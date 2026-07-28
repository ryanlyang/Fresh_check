#!/usr/bin/env python3
"""Train or evaluate one immutable RETB Stage-D native-HLT fusion row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.native_fusion import (  # noqa: E402
    NATIVE_FUSION_REGISTRATION_CONTRACT,
    NativeFusionTrainingConfig,
    build_native_fusion_model,
    evaluate_native_hlt_fusion,
    load_native_fusion_cache,
    train_native_hlt_fusion,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step6 import (  # noqa: E402
    resolve_stage_d_confirmation_run,
    resolve_stage_d_run,
    validate_stage_d_confirmation_registry,
    validate_stage_d_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)

import torch  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--confirmation-registry", type=Path)
    parser.add_argument("--model-train-cache", required=True, type=Path)
    parser.add_argument("--val-stop-cache", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_d_runs.json"
    )
    base_registry_sha = validate_stage_d_run_registry(registry)
    if args.confirmation_registry is None:
        registry_sha = base_registry_sha
        run = resolve_stage_d_run(registry, run_id=args.run_id)
    else:
        confirmation = load_hashed_json(args.confirmation_registry)
        registry_sha = validate_stage_d_confirmation_registry(confirmation)
        if (
            confirmation["stage_d_run_registry_sha256"] != base_registry_sha
            or confirmation.get("source") != campaign.get("source")
        ):
            raise ValueError("Stage-D confirmation lineage differs")
        run = resolve_stage_d_confirmation_run(
            confirmation, run_id=args.run_id
        )
    configuration = run["configuration"]
    if configuration.get("kind") != "NATIVE_HLT_FUSION":
        raise ValueError("Stage-D row is not a native HLT fusion")
    train_meta, _ = load_native_fusion_cache(args.model_train_cache)
    val_meta, _ = load_native_fusion_cache(args.val_stop_cache)
    if (
        train_meta.get("source") != campaign.get("source")
        or val_meta.get("source") != campaign.get("source")
    ):
        raise ValueError("native fusion caches belong to another source snapshot")
    variant = configuration["fusion_variant"]
    if train_meta["shape_id"] != configuration["shape_id"]:
        raise ValueError("native fusion cache shape differs from the run")
    bank_dimensions = {
        name: int(shape[1]) for name, shape in train_meta["allocation"].items()
    }
    output = args.output_dir or (
        args.campaign_root
        / "runs"
        / "stage_d"
        / "native_fusions"
        / args.run_id
        / f"seed_{run['seed']}"
    )
    profile = campaign["campaign_profile"]
    miniature = profile == "miniature_test"
    result = {
        "dry_run": bool(args.dry_run),
        "run": run,
        "run_registry_sha256": registry_sha,
        "model_train_cache_sha256": train_meta["content_hash"],
        "val_stop_cache_sha256": val_meta["content_hash"],
        "output_dir": str(output.resolve()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="model_train"
    )
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="val_stop"
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    torch.manual_seed(int(run["seed"]))
    model = build_native_fusion_model(
        variant, bank_dimensions=bank_dimensions
    )
    if configuration["fixed_epochs"] == 0:
        metrics = evaluate_native_hlt_fusion(
            model=model,
            manifest_path=args.val_stop_cache,
            batch_size=args.batch_size,
            device=device,
        )
        registration = bind_source(
            with_content_hash(
                {
                    "contract": NATIVE_FUSION_REGISTRATION_CONTRACT,
                    "schema_version": 1,
                    "run_id": args.run_id,
                    "variant": variant,
                    "pipeline_seed": int(run["seed"]),
                    "realization_policy": configuration["realization_policy"],
                    "shape_id": train_meta["shape_id"],
                    "run_registry_sha256": registry_sha,
                    "model_train_cache_sha256": train_meta["content_hash"],
                    "val_stop_cache_sha256": val_meta["content_hash"],
                    "checkpoint_sha256": None,
                    "val_stop_metrics": metrics,
                    "selected_epoch": None,
                    "epochs_completed": 0,
                    "fixed_epoch_budget_completed": True,
                    "offline_targets_consumed": False,
                    "performance_based_termination": False,
                }
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        output.mkdir(parents=True, exist_ok=True)
        write_immutable_json(output / "fusion_registration.json", registration)
    else:
        step6 = load_hashed_json(
            args.campaign_root
            / "registry"
            / "retb_step6_native_hlt_bundle.json"
        )
        config = NativeFusionTrainingConfig(
            seed=int(run["seed"]),
            variant=variant,
            realization_policy=configuration["realization_policy"],
            maximum_epochs=2 if miniature else 40,
            batch_size=args.batch_size,
            campaign_profile="miniature_test" if miniature else "production",
        )
        registration = train_native_hlt_fusion(
            model=model,
            model_train_manifest=args.model_train_cache,
            val_stop_manifest=args.val_stop_cache,
            output_dir=output,
            run_id=args.run_id,
            run_registry_sha256=registry_sha,
            native_fusion_contract_sha256=step6["artifact_hashes"][
                "native_fusion"
            ],
            global_determinism_sha256=campaign["parent_artifact_hashes"][
                "global_determinism"
            ],
            config=config,
            device=device,
        )
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

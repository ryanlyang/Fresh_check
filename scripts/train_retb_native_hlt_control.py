#!/usr/bin/env python3
"""Train one immutable RETB Stage-D H_BASE or H_WIDE control."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_retb_native_hlt_expert import _labels, _mapping  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_controls import (  # noqa: E402
    NativeHLTControlTrainingConfig,
    build_hlt_matched_control_model,
    train_native_hlt_control,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (  # noqa: E402
    HLT_EVALUATION_REALIZATION_POLICY,
    NativeHLTExpertDataset,
    make_native_hlt_expert_loader,
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
from teacher_logit_reco.relational_part.capacity import (  # noqa: E402
    select_wide_widths,
)

import torch  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--confirmation-registry", type=Path)
    parser.add_argument("--train-cache", action="append", default=[])
    parser.add_argument("--val-stop-cache", action="append", default=[])
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--val-stop-labels", required=True, type=Path)
    parser.add_argument("--wide-capacity-artifact", type=Path)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
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
    control = configuration.get("control_id")
    if configuration.get("kind") != "NATIVE_HLT_MATCHED_CONTROL" or control not in {
        "H_BASE",
        "H_WIDE",
    }:
        raise ValueError("Stage-D row is not a single-model matched control")
    train_paths = _mapping(args.train_cache, argument="--train-cache")
    val_paths = _mapping(args.val_stop_cache, argument="--val-stop-cache")
    train_arrays, train_metadata = {}, {}
    val_arrays, val_metadata = {}, {}
    for replica, path in train_paths.items():
        train_arrays[replica], train_metadata[replica] = load_hlt_v3_cache(path)
    for replica, path in val_paths.items():
        val_arrays[replica], val_metadata[replica] = load_hlt_v3_cache(path)
    train_labels, train_ids = _labels(args.train_labels)
    val_labels, val_ids = _labels(args.val_stop_labels)
    train_dataset = NativeHLTExpertDataset(
        replica_arrays=train_arrays,
        replica_metadata=train_metadata,
        labels=train_labels,
        identities=train_ids,
        logical_role="model_train",
        realization_policy="R_MULTI",
    )
    val_dataset = NativeHLTExpertDataset(
        replica_arrays=val_arrays,
        replica_metadata=val_metadata,
        labels=val_labels,
        identities=val_ids,
        logical_role="val_stop",
        realization_policy=HLT_EVALUATION_REALIZATION_POLICY,
    )
    miniature = campaign["campaign_profile"] == "miniature_test"
    config = NativeHLTControlTrainingConfig(
        seed=int(run["seed"]),
        control_id=control,
        maximum_epochs=2 if miniature else 40,
        microbatch_size=args.microbatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=(
            args.microbatch_size * args.gradient_accumulation_steps
        ),
        campaign_profile="miniature_test" if miniature else "production",
    )
    output = args.output_dir or (
        args.campaign_root
        / "runs"
        / "stage_d"
        / "matched_controls"
        / args.run_id
        / f"seed_{run['seed']}"
    )
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "run": run,
                "output_dir": str(output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="model_train"
    )
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="val_stop"
    )
    capacity = (
        load_hashed_json(args.wide_capacity_artifact)
        if args.wide_capacity_artifact is not None
        else None
    )
    if control == "H_WIDE" and capacity is None:
        capacity = select_wide_widths()
    if control == "H_WIDE":
        write_immutable_json(output / "locked_wide_capacity.json", capacity)
    torch.manual_seed(int(run["seed"]))
    model = build_hlt_matched_control_model(
        control,
        weaver_module=importlib.import_module(
            "weaver.nn.model.ParticleTransformer"
        ),
        wide_capacity_artifact=capacity,
    )
    registration = train_native_hlt_control(
        model=model,
        train_loader=make_native_hlt_expert_loader(
            train_dataset,
            seed=config.seed,
            training=True,
            batch_size=config.microbatch_size,
        ),
        val_stop_loader=make_native_hlt_expert_loader(
            val_dataset,
            seed=0,
            training=False,
            batch_size=config.microbatch_size,
        ),
        output_dir=output,
        run_id=args.run_id,
        run_registry_sha256=registry_sha,
        lineage_hashes={
            "campaign_spec": campaign["content_hash"],
            **(
                {"wide_capacity": capacity["content_hash"]}
                if control == "H_WIDE"
                else {}
            ),
            **{
                f"model_train_hlt_replica_{key}": value["content_hash"]
                for key, value in train_metadata.items()
            },
            **{
                f"val_stop_hlt_replica_{key}": value["content_hash"]
                for key, value in val_metadata.items()
            },
        },
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        config=config,
        device=(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.device == "auto"
            else torch.device(args.device)
        ),
    )
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

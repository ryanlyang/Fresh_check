#!/usr/bin/env python3
"""Train or evaluate one immutable RETB Stage-D native-HLT fusion row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows development host
    fcntl = None

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
    publish_native_fusion_cache,
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
import numpy as np  # noqa: E402

from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)


def _hlt_manifest(root: Path, split: str, replica: int) -> Path:
    policy = (
        "R_MULTI"
        if split in {"model_train", "scale_train"}
        else "R_FIXED"
    )
    return (
        root
        / "inputs"
        / "hlt_v3"
        / split
        / f"replica_{replica}"
        / policy
        / "D_NOMINAL"
        / "hlt_v3_metadata.json"
    )


def _expert_row(
    registry: dict,
    *,
    confirmation: dict | None,
    shape: str,
    expert: str,
    seed: int,
) -> dict:
    if confirmation is not None:
        rows = [
            row
            for row in confirmation["rows"]
            if (
                row["component"] == "HLT_EXPERT"
                and int(row["seed"]) == int(seed)
                and row["configuration"]["shape_id"] == shape
                and row["configuration"]["expert_id"] == expert
                and row["configuration"]["mode"] == "HE_SCRATCH_CE"
                and row["configuration"]["realization_policy"] == "R_MULTI"
                and not row["configuration"]["measurement_embedding"]
            )
        ]
        if len(rows) != 1:
            raise ValueError(
                "selected native fusion expert confirmation differs for "
                f"{shape}/{expert}/{seed}"
            )
        return rows[0]
    rows = []
    for section in (
        "scratch_expert_rows",
        "encoder_screen_rows",
        "bridge_parent_expert_rows",
    ):
        for row in registry[section]:
            config = row["configuration"]
            if (
                int(row["seed"]) == int(seed)
                and config["shape_id"] == shape
                and config["expert_id"] == expert
                and config["mode"] == "HE_SCRATCH_CE"
                and config["realization_policy"] == "R_MULTI"
                and not config["measurement_embedding"]
            ):
                rows.append(row)
    unique = {row["run_id"]: row for row in rows}
    if len(unique) != 1:
        raise ValueError(
            f"native fusion expert parent differs for {shape}/{expert}/{seed}"
        )
    return next(iter(unique.values()))


def _ensure_cache(
    *,
    path: Path,
    root: Path,
    registry: dict,
    confirmation: dict | None,
    shape: str,
    seed: int,
    split: str,
) -> None:
    if path.is_file():
        return
    lock_path = path.parent.parent / f".{split}_cache.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            return
        replica_ids = (
            (0, 1, 2, 3)
            if split in {"model_train", "scale_train"}
            else (0,)
        )
        identities = labels = None
        banks = {replica: {} for replica in replica_ids}
        logits = {replica: {} for replica in replica_ids}
        states, masks, registrations = {}, {}, {}
        for expert in EXPERT_ORDER:
            row = _expert_row(
                registry,
                confirmation=confirmation,
                shape=shape,
                expert=expert,
                seed=seed,
            )
            parent = (
                root
                / "runs"
                / "stage_d"
                / "hlt_experts"
                / row["run_id"]
                / f"seed_{seed}"
            )
            registration = load_hashed_json(
                parent / "checkpoint_registration.json"
            )
            manifest = load_hashed_json(parent / "native_output_manifest.json")
            if (
                manifest.get("contract")
                not in {
                    "retb_native_hlt_expert_outputs_v5",
                    "retb_native_hlt_expert_outputs_v6",
                }
                or manifest["expert_registration_sha256"]
                != registration["content_hash"]
            ):
                raise ValueError("native fusion expert output lineage differs")
            registrations[expert] = registration["content_hash"]
            for replica in replica_ids:
                record = manifest["files"][f"{split}_replica_{replica}"]
                with np.load(
                    parent / record["relative_path"], allow_pickle=False
                ) as payload:
                    current_ids = np.asarray(payload["identities"])
                    current_labels = np.asarray(
                        payload["labels"], dtype=np.int64
                    )
                    banks[replica][expert] = np.asarray(
                        payload["tokens"], dtype=np.float32
                    )
                    logits[replica][expert] = np.asarray(
                        payload["logits"], dtype=np.float32
                    )
                    if expert == "BASE4":
                        states[replica] = np.asarray(
                            payload["particle_states"], dtype=np.float32
                        )
                        masks[replica] = np.asarray(
                            payload["particle_mask"], dtype=bool
                        )
                if identities is None:
                    identities, labels = current_ids, current_labels
                elif not np.array_equal(
                    identities, current_ids
                ) or not np.array_equal(labels, current_labels):
                    raise ValueError(
                        "native fusion expert populations differ"
                    )
        hlt_hashes = {
            replica: load_hashed_json(
                _hlt_manifest(root, split, replica)
            )["content_hash"]
            for replica in replica_ids
        }
        identity_parent = load_hashed_json(
            root
            / "inputs"
            / "offline"
            / split
            / "offline_input_manifest.json"
        )["content_hash"]
        publish_native_fusion_cache(
            output_dir=path.parent,
            split=split,
            pipeline_seed=seed,
            shape_id=shape,
            realization_policy="R_MULTI",
            identities=[str(value) for value in identities.tolist()],
            labels=labels,
            token_banks_by_replica=banks,
            expert_logits_by_replica=logits,
            unbiased_particle_states_by_replica=states,
            particle_masks_by_replica=masks,
            expert_registration_hashes=registrations,
            hlt_cache_hashes_by_replica=hlt_hashes,
            identity_manifest_sha256=identity_parent,
            label_manifest_sha256=identity_parent,
            source_snapshot=source_snapshot(REPO_ROOT),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--pipeline-seed-override",
        type=int,
        help=(
            "Stage-M only: instantiate the locked native-fusion recipe for "
            "a matched scale seed without reopening variant selection."
        ),
    )
    parser.add_argument("--confirmation-registry", type=Path)
    parser.add_argument(
        "--training-role",
        choices=("model_train", "scale_train"),
        default="model_train",
    )
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
        confirmation = None
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
    if args.pipeline_seed_override is not None:
        if (
            args.training_role != "scale_train"
            or args.pipeline_seed_override not in {101, 202, 303}
        ):
            raise ValueError("native-fusion seed override is Stage-M only")
        run = {
            **run,
            "seed": int(args.pipeline_seed_override),
            "run_id": (
                f"RETB_SCALE_NATIVE_FUSION_"
                f"{configuration['shape_id']}_"
                f"{configuration['fusion_variant']}_"
                f"S{int(args.pipeline_seed_override)}"
            ),
        }
        args.run_id = run["run_id"]
    _ensure_cache(
        path=args.model_train_cache,
        root=args.campaign_root,
        registry=registry,
        confirmation=confirmation,
        shape=configuration["shape_id"],
        seed=int(run["seed"]),
        split=args.training_role,
    )
    _ensure_cache(
        path=args.val_stop_cache,
        root=args.campaign_root,
        registry=registry,
        confirmation=confirmation,
        shape=configuration["shape_id"],
        seed=int(run["seed"]),
        split="val_stop",
    )
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
        worker_role=(
            "scale_training_worker"
            if args.training_role == "scale_train"
            else "training_worker"
        ),
        requested_resource=args.training_role,
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
            training_split=args.training_role,
            device=device,
        )
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

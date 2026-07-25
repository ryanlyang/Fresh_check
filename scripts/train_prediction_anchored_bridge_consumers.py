#!/usr/bin/env python3
"""Validate Step 3 recipes and publish paired bridge-consumer replicas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge_consumer import (  # noqa: E402
    PAIRED_SEED_IDS,
    STEP3_RUN_IDS,
    T10_CLEAN,
    T10_ALL50_CLEAN,
    T10_ROBUST,
    ConsumerCampaignConfig,
    ReplicaResult,
    build_consumer_replica_manifest,
    publish_evaluated_teacher_replica,
    publish_paired_replicas,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "publish"), default="plan")
    parser.add_argument("--baseline-steps", type=int, default=10_000)
    parser.add_argument("--bridge-finetune-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-interval-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--model-size", choices=("tiny", "base", "large"), default="base")
    parser.add_argument(
        "--data-profile",
        choices=("pilot_250k", "high_data_3m"),
        default="pilot_250k",
    )
    parser.add_argument("--output", default="", help="immutable plan JSON for plan mode")
    parser.add_argument("--output-dir", default="", help="empty publication directory for publish mode")
    parser.add_argument("--replica-dir", default="", help="directory containing seed checkpoint/metric pairs")
    parser.add_argument(
        "--selection-aggregate",
        default="",
        help="teacher model_val_select aggregate; retains its exact evaluated replica bytes",
    )
    parser.add_argument("--run-id", choices=STEP3_RUN_IDS)
    parser.add_argument("--reservations", default="", help="immutable campaign reservations")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config(args: argparse.Namespace) -> ConsumerCampaignConfig:
    return ConsumerCampaignConfig(
        baseline_steps=args.baseline_steps,
        bridge_finetune_steps=args.bridge_finetune_steps,
        batch_size=args.batch_size,
        evaluation_interval_steps=args.evaluation_interval_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        model_size=args.model_size,
        data_profile=args.data_profile,
    )


def _load_replicas(root: Path, run_id: str) -> list[ReplicaResult]:
    import torch

    replicas: list[ReplicaResult] = []
    for seed in PAIRED_SEED_IDS:
        checkpoint = root / f"{run_id}__seed{seed}.pt"
        metrics_path = root / f"{run_id}__seed{seed}.metrics.json"
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise FileNotFoundError(f"missing/unsafe replica checkpoint: {checkpoint}")
        if metrics_path.is_symlink() or not metrics_path.is_file():
            raise FileNotFoundError(f"missing/unsafe replica metrics: {metrics_path}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        replicas.append(
            ReplicaResult(
                run_id=run_id,
                seed_id=seed,
                metrics=metrics,
                weights_payload=payload,
            )
        )
    return replicas


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _config(args)
    manifest = build_consumer_replica_manifest(config)
    if args.mode == "plan":
        if not args.dry_run and not args.output:
            raise ValueError("plan mode requires --output unless --dry-run is used")
        result: dict[str, object] = {"dry_run": bool(args.dry_run), "manifest": manifest}
        if not args.dry_run:
            result["publication"] = write_immutable_json(args.output, manifest)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.dry_run:
        raise ValueError("publish mode cannot use --dry-run")
    if not args.run_id or not args.replica_dir or not args.output_dir:
        raise ValueError("publish mode requires --run-id, --replica-dir, and --output-dir")
    if not args.reservations:
        raise ValueError("production publication requires --reservations")
    reservations = load_hashed_json(
        args.reservations, expected_contract="prediction_anchored_step9_campaign_reservations_v1"
    )
    reservation_bytes = int(reservations["run_reservations_bytes"][args.run_id])
    if reservation_bytes <= 0:
        raise PermissionError("run has no positive persistent reservation")
    replicas = _load_replicas(Path(args.replica_dir), args.run_id)
    is_bridge_teacher = args.run_id in {T10_CLEAN, T10_ROBUST, T10_ALL50_CLEAN}
    if is_bridge_teacher:
        if not args.selection_aggregate:
            raise ValueError("teacher publication requires --selection-aggregate")
        aggregate_path = Path(args.selection_aggregate)
        if aggregate_path.is_symlink() or not aggregate_path.is_file():
            raise FileNotFoundError("teacher selection aggregate is absent or unsafe")
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        publication = publish_evaluated_teacher_replica(
            run_id=args.run_id,
            selection_aggregate=aggregate,
            replica_checkpoint_paths={
                int(seed): Path(args.replica_dir) / f"{args.run_id}__seed{seed}.pt"
                for seed in PAIRED_SEED_IDS
            },
            output_dir=args.output_dir,
            reservation_bytes=reservation_bytes,
        )
    else:
        if args.selection_aggregate:
            raise ValueError("non-teacher publication cannot use --selection-aggregate")
        publication = publish_paired_replicas(
            replicas,
            output_dir=args.output_dir,
            reservation_bytes=reservation_bytes,
        )
    print(json.dumps({"manifest_sha256": manifest["content_hash"], **publication}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

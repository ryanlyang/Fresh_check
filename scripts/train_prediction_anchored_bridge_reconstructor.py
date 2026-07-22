#!/usr/bin/env python3
"""Plan/measure C0 or execute/publish a packed reconstruction sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    C0_CANONICAL_RUN_IDS,
    C0CampaignConfig,
    RECONSTRUCTION_RUN_IDS,
    ReconstructionCampaignConfig,
    ReconstructionReplicaResult,
    build_c0_campaign_manifest,
    measure_c0_registry_states,
    publish_reconstruction_paired_replicas,
    publish_l0_early_replay_manifest,
    run_reconstruction_pack_from_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign import (  # noqa: E402
    PAIRED_SEED_IDS,
    validate_campaign_registry,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "plan",
            "measure",
            "execute",
            "publish",
            "publish-l0-early",
            "publish-l0-postteacher",
        ),
        default="plan",
    )
    parser.add_argument("--scaler", default="", help="immutable physical45 scaler JSON")
    parser.add_argument("--registry", default="", help="campaign registry JSON for measure mode")
    parser.add_argument("--output", default="", help="immutable plan JSON")
    parser.add_argument("--output-dir", default="", help="empty measure/publication directory")
    parser.add_argument("--replica-dir", default="", help="seed checkpoint/metric directory")
    parser.add_argument("--execution-report", default="", help="immutable RAM/storage telemetry report")
    parser.add_argument("--run-id", choices=RECONSTRUCTION_RUN_IDS)
    parser.add_argument("--reservations", default="", help="immutable campaign reservations")
    parser.add_argument("--execution-spec", default="")
    parser.add_argument("--graph", default="")
    parser.add_argument("--node-id", default="")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--r0-checkpoint", default="")
    parser.add_argument("--r0-registration", default="")
    parser.add_argument("--all50-scaler", default="")
    parser.add_argument("--absolute-scaler", default="")
    parser.add_argument("--deployed-resource-reference", default="")
    parser.add_argument("--ram-root", default="")
    parser.add_argument("--allocation-id", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--field-warmup-steps", type=int, default=2_000)
    parser.add_argument("--phase2-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--model-width", type=int, default=160)
    parser.add_argument("--particle-mlp-layers", type=int, default=2)
    parser.add_argument("--head-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-capacity-bytes", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--allow-unverified-test-root", action="store_true", help=argparse.SUPPRESS)
    return parser


def _read_json(path: str, *, label: str) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"missing/unsafe {label}: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _campaign_config(args: argparse.Namespace) -> C0CampaignConfig:
    return C0CampaignConfig(
        field_warmup_steps=args.field_warmup_steps,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        kd_temperature=args.kd_temperature,
        early_stop_patience=args.early_stop_patience,
        model_width=args.model_width,
        particle_mlp_layers=args.particle_mlp_layers,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    )


def _load_replicas(root: Path, run_id: str) -> list[ReconstructionReplicaResult]:
    import torch

    replicas: list[ReconstructionReplicaResult] = []
    for seed in PAIRED_SEED_IDS:
        checkpoint = root / f"{run_id}__seed{seed}.pt"
        metrics_path = root / f"{run_id}__seed{seed}.metrics.json"
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise FileNotFoundError(f"missing/unsafe reconstruction replica checkpoint: {checkpoint}")
        if metrics_path.is_symlink() or not metrics_path.is_file():
            raise FileNotFoundError(f"missing/unsafe reconstruction replica metrics: {metrics_path}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        replicas.append(
            ReconstructionReplicaResult(
                run_id=run_id,
                seed_id=int(seed),
                metrics=metrics,
                weights_payload=payload,
                source_checkpoint_sha256=sha256_file(checkpoint),
            )
        )
    return replicas


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode in {"publish", "publish-l0-early", "publish-l0-postteacher"}:
        if args.dry_run:
            raise ValueError("publish mode cannot use --dry-run")
        if not args.run_id or not args.replica_dir or not args.output_dir:
            raise ValueError("publish mode requires --run-id, --replica-dir, and --output-dir")
        if not args.reservations:
            raise ValueError("production publication requires --reservations")
        reservations = load_hashed_json(
            args.reservations,
            expected_contract="prediction_anchored_step9_campaign_reservations_v1",
        )
        reservation_bytes = int(reservations["run_reservations_bytes"][args.run_id])
        if reservation_bytes <= 0:
            raise PermissionError("run has no positive persistent reservation")
        replicas = _load_replicas(Path(args.replica_dir), args.run_id)
        if args.mode == "publish-l0-early":
            result = publish_l0_early_replay_manifest(
                replicas, output_dir=args.output_dir
            )
        else:
            result = publish_reconstruction_paired_replicas(
                replicas,
                output_dir=args.output_dir,
                l0_postteacher=args.mode == "publish-l0-postteacher",
                reservation_bytes=reservation_bytes,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.mode == "execute":
        required = {
            "--execution-spec": args.execution_spec,
            "--graph": args.graph,
            "--node-id": args.node_id,
            "--artifact-root": args.artifact_root,
            "--r0-checkpoint": args.r0_checkpoint,
            "--r0-registration": args.r0_registration,
            "--scaler": args.scaler,
            "--all50-scaler": args.all50_scaler,
            "--replica-dir": args.replica_dir,
            "--ram-root": args.ram_root,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("execute mode is missing " + ", ".join(missing))
        if args.dry_run:
            raise ValueError("execute mode dry-run is handled by the immutable graph planner")
        result = run_reconstruction_pack_from_execution_spec(
            args.execution_spec,
            graph_path=args.graph,
            node_id=args.node_id,
            artifact_root=args.artifact_root,
            r0_checkpoint_path=args.r0_checkpoint,
            r0_registration_path=args.r0_registration,
            physical45_scaler_path=args.scaler,
            all50_scaler_path=args.all50_scaler,
            absolute_scaler_path=args.absolute_scaler or None,
            deployed_reference_path=args.deployed_resource_reference or None,
            replica_output_dir=args.replica_dir,
            ram_root=args.ram_root,
            allocation_id=args.allocation_id or None,
            device=args.device,
            shard_size=int(args.shard_size),
            config=ReconstructionCampaignConfig(
                field_warmup_steps=args.field_warmup_steps,
                phase2_epochs=args.phase2_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                kd_temperature=args.kd_temperature,
                early_stop_patience=args.early_stop_patience,
                c0_model_width=args.model_width,
                dropout=args.dropout,
            ),
            capacity_bytes=(int(args.test_capacity_bytes) or None),
            allow_unverified_test_root=bool(args.allow_unverified_test_root),
        )
        if args.execution_report:
            result = dict(result)
            result["execution_report_publication"] = write_immutable_json(
                args.execution_report, result
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if not args.scaler:
        raise ValueError(f"{args.mode} mode requires --scaler")
    scaler = _read_json(args.scaler, label="physical45 scaler")
    validate_content_hash(scaler)

    if args.mode == "plan":
        manifest = build_c0_campaign_manifest(_campaign_config(args), scaler_artifact=scaler)
        if not args.dry_run:
            if not args.output:
                raise ValueError("plan mode requires --output unless --dry-run is used")
            write_immutable_json(args.output, manifest)
        print(
            json.dumps(
                {
                    "dry_run": bool(args.dry_run),
                    "manifest": manifest,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.registry:
        raise ValueError("measure mode requires --registry")
    registry = _read_json(args.registry, label="campaign registry")
    validate_campaign_registry(registry)
    updated, measurement = measure_c0_registry_states(
        registry,
        scaler_artifact=scaler,
        model_width=args.model_width,
    )
    output: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "measurement": measurement,
        "updated_registry_sha256": updated["content_hash"],
    }
    if not args.dry_run:
        if not args.output_dir:
            raise ValueError("measure mode requires --output-dir unless --dry-run is used")
        root = Path(args.output_dir)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"C0 measurement directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        write_immutable_json(root / "c0_measurement.json", measurement)
        write_immutable_json(root / "campaign_registry_step5.json", updated)
        output["persistent_artifacts"] = [
            "c0_measurement.json",
            "campaign_registry_step5.json",
        ]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

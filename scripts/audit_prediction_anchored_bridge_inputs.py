#!/usr/bin/env python3
"""Stage and audit one prediction-anchored bridge source in low-storage mode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge import (  # noqa: E402
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    BridgeScalers,
    build_bridge_recipe,
    fit_bridge_scalers,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    canonical_sha256,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_ram import (  # noqa: E402
    AllocationNpzStager,
    AllocationRamLedger,
    FrozenR0Runner,
    PredictionAnchoredBridgeProvider,
)
from teacher_logit_reco.local_particle_residual_field.targets import (  # noqa: E402
    DEFAULT_LOCAL_RESIDUAL_RADII,
    local_particle_residual_field_layout,
)


AUDIT_CONTRACT = "prediction_anchored_bridge_step2_audit_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlt-npz", required=True)
    parser.add_argument("--hlt-metadata", required=True)
    parser.add_argument("--offline-npz", required=True)
    parser.add_argument("--offline-metadata", required=True)
    parser.add_argument("--r0-checkpoint", required=True)
    parser.add_argument("--ram-root", required=True)
    parser.add_argument("--allocation-id", default=os.environ.get("SLURM_JOB_ID", "local_audit"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--preprocessing-sha256", default="")
    parser.add_argument("--target-schema-sha256", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--derived-capacity-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-fit-jets", type=int, default=0, help="0 fits the complete staged split")
    parser.add_argument("--fit-batch-size", type=int, default=512)
    parser.add_argument("--test-capacity-bytes", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--allow-unverified-test-root", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _default_hashes(args: argparse.Namespace) -> tuple[str, str]:
    names, groups, _ = local_particle_residual_field_layout(DEFAULT_LOCAL_RESIDUAL_RADII)
    target = args.target_schema_sha256 or canonical_sha256(
        {"radii": list(DEFAULT_LOCAL_RESIDUAL_RADII), "field_names": names, "field_groups": groups}
    )
    preprocessing = args.preprocessing_sha256 or canonical_sha256(
        {"source": "fixed_hlt_raw_tokens", "dtype": "float32", "mask": "bool", "units": "repository_native"}
    )
    return preprocessing, target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preprocessing_hash, target_hash = _default_hashes(args)
    dry_report = {
        "contract": AUDIT_CONTRACT,
        "dry_run": bool(args.dry_run),
        "single_node_required": True,
        "one_persistent_npz_open_per_source": True,
        "raw_shard_size": int(args.shard_size),
        "persistent_output_allowlist": [
            "bridge_recipe_all50.json",
            "bridge_recipe_physical45.json",
            "bridge_scalers_all50.json",
            "bridge_scalers_physical45.json",
            "step2_audit_metrics.json",
        ],
        "persistent_field_tensors_written": False,
    }
    if args.dry_run:
        print(json.dumps(dry_report, indent=2, sort_keys=True))
        return 0

    ledger = AllocationRamLedger(
        args.ram_root,
        allocation_id=str(args.allocation_id),
        capacity_bytes=(int(args.test_capacity_bytes) or None),
        allow_unverified_test_root=bool(args.allow_unverified_test_root),
    )
    provider = None
    try:
        stager = AllocationNpzStager(ledger, rank=0, world_size=int(args.world_size))
        hlt, offline, stage_report = stager.stage_pair(
            hlt_npz=args.hlt_npz,
            hlt_metadata=args.hlt_metadata,
            offline_npz=args.offline_npz,
            offline_metadata=args.offline_metadata,
            shard_size=int(args.shard_size),
        )
        r0 = FrozenR0Runner(args.r0_checkpoint, device=args.device)
        provider = PredictionAnchoredBridgeProvider(
            hlt=hlt,
            offline=offline,
            r0=r0,
            ledger=ledger,
            rank=0,
            world_size=1,
            derived_capacity_bytes=int(args.derived_capacity_bytes),
        )
        fit_count = hlt.n_events if int(args.max_fit_jets) <= 0 else min(hlt.n_events, int(args.max_fit_jets))
        fit_indices = np.arange(fit_count, dtype=np.int64)
        audit_indices = fit_indices[: min(16, fit_count)]
        audit_f0_a, audit_h0_a = provider.r0_for_indices(audit_indices)
        audit_f0_b, audit_h0_b = provider.r0_for_indices(audit_indices)
        if not np.array_equal(audit_f0_a, audit_f0_b) or not np.array_equal(audit_h0_a, audit_h0_b):
            raise RuntimeError("frozen R0 regeneration is not deterministic")
        audit_truth_a, audit_mask_a = provider.truth_for_indices(audit_indices)
        audit_truth_b, audit_mask_b = provider.truth_for_indices(audit_indices)
        if not np.array_equal(audit_truth_a, audit_truth_b) or not np.array_equal(audit_mask_a, audit_mask_b):
            raise RuntimeError("streamed f_true regeneration is not deterministic")
        particle_slots = int(hlt.manifest["arrays"]["tokens"]["shape"][1])
        scaler_value_bytes = fit_count * particle_slots * 50 * 2 * np.dtype(np.float32).itemsize
        # Float64 per-channel quantile work arrays temporarily coexist with the
        # two float32 retained value banks.
        scaler_working_bytes = int(np.ceil(1.10 * scaler_value_bytes)) + 1024 * 1024
        scaler_reservation = ledger.reserve(
            owner="rank0",
            role="exact_scaler_fit_working_values",
            expected_bytes=scaler_working_bytes,
            category="derived",
        )
        ledger.commit(scaler_reservation, measured_bytes=scaler_working_bytes)

        def batches():
            for start in range(0, fit_count, int(args.fit_batch_size)):
                selected = fit_indices[start : start + int(args.fit_batch_size)]
                f_true, mask = provider.truth_for_indices(selected)
                f0, _ = provider.r0_for_indices(selected)
                yield f0, f_true, mask

        parent_hashes = {
            "source_manifest_sha256": canonical_sha256(stage_report["source_content_hash"]),
            "r0_checkpoint_sha256": r0.checkpoint_sha256,
            "target_schema_sha256": target_hash,
            "mask_sha256": stage_report["event_order_sha256"],
            "fit_code_sha256": canonical_sha256({"fit_version": "physical_valid_particle_exact_channel_quantiles_v1"}),
        }
        try:
            physical_scalers = fit_bridge_scalers(
                batches,
                parent_hashes=parent_hashes,
                channel_policy=BRIDGE_CHANNEL_PHYSICAL45,
            )
        finally:
            ledger.release(scaler_reservation)
        all50_scalers = BridgeScalers(
            mu_f0=physical_scalers.mu_f0,
            sigma_f0=physical_scalers.sigma_f0,
            q99_delta=physical_scalers.q99_delta,
            sigma_delta=physical_scalers.sigma_delta,
            trust_scale=physical_scalers.trust_scale,
            epsilon=physical_scalers.epsilon,
            active=physical_scalers.q99_delta > 0,
            sparse_nonzero_fallback=physical_scalers.sparse_nonzero_fallback,
            valid_count=physical_scalers.valid_count,
            parent_hashes=parent_hashes,
            channel_policy=BRIDGE_CHANNEL_ALL50,
        )
        recipe_kwargs = {
            "rho": "0.100",
            "r0_checkpoint_sha256": r0.checkpoint_sha256,
            "hlt_source_sha256": stage_report["source_file_sha256"]["hlt"],
            "offline_source_sha256": stage_report["source_file_sha256"]["offline"],
            "split_manifest_sha256": str(args.split_manifest_sha256),
            "target_schema_sha256": target_hash,
            "preprocessing_sha256": preprocessing_hash,
            "event_order_sha256": stage_report["event_order_sha256"],
            "audit_summary": {
                "n_events": hlt.n_events,
                "fit_events": fit_count,
                "one_open_verified": all(value == 1 for value in stage_report["persistent_npz_open_counts"].values()),
            },
        }
        physical_recipe = build_bridge_recipe(channel_policy=BRIDGE_CHANNEL_PHYSICAL45, **recipe_kwargs)
        all50_recipe = build_bridge_recipe(channel_policy=BRIDGE_CHANNEL_ALL50, **recipe_kwargs)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_immutable_json(output / "bridge_recipe_physical45.json", physical_recipe)
        write_immutable_json(output / "bridge_recipe_all50.json", all50_recipe)
        write_immutable_json(output / "bridge_scalers_physical45.json", physical_scalers.to_artifact())
        write_immutable_json(output / "bridge_scalers_all50.json", all50_scalers.to_artifact())
        metrics = {
            **dry_report,
            "dry_run": False,
            "n_events": hlt.n_events,
            "fit_events": fit_count,
            "raw_stage": stage_report,
            "provider_telemetry": provider.telemetry(),
            "regeneration_agreement": {
                "r0_f0_bitwise": True,
                "r0_h0_bitwise": True,
                "f_true_bitwise": True,
                "masked_f0_exact_zero": bool(np.all(audit_f0_a[~audit_mask_a] == 0)),
                "masked_h0_exact_zero": bool(np.all(audit_h0_a[~audit_mask_a] == 0)),
            },
            "ledger_peak_reserved_bytes": ledger.snapshot()["peak_reserved_bytes"],
            "recipes": {
                "physical45": physical_recipe["content_hash"],
                "all50": all50_recipe["content_hash"],
            },
            "scalers": {
                "physical45": physical_scalers.to_artifact()["content_hash"],
                "all50": all50_scalers.to_artifact()["content_hash"],
            },
        }
        write_immutable_json(output / "step2_audit_metrics.json", with_content_hash(metrics))
        names = sorted(path.name for path in output.iterdir())
        if names != sorted(dry_report["persistent_output_allowlist"]):
            raise RuntimeError(f"production audit wrote unexpected persistent artifacts: {names}")
        print(json.dumps({"ok": True, "output_dir": str(output), "persistent_artifacts": names}, indent=2))
        return 0
    finally:
        if provider is not None:
            provider.close()
        ledger.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

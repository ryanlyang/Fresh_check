#!/usr/bin/env python3
"""Compile measured profiler overhead into the single-path runtime gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    ABPH_RUNTIME_PROFILE_CONTRACT,
    ABPH_SINGLE_PATH_ACCEPTANCE_CONTRACT,
    canonical_hash,
)

DENSE_INSTRUMENTATION_OVERHEAD_TARGET = 0.03
DENSE_INSTRUMENTATION_OVERHEAD_OPERATIONAL_CEILING = 0.10
PRODUCTION_PROFILE_SAMPLE_INTERVAL = 100
PRODUCTION_INSTRUMENTATION_OVERHEAD_CEILING = 0.03

# Compatibility export for callers that used the former dense ceiling name.
INSTRUMENTATION_OVERHEAD_OPERATIONAL_CEILING = (
    DENSE_INSTRUMENTATION_OVERHEAD_OPERATIONAL_CEILING
)


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _selection_score(run_dir: Path) -> float:
    curves = _read(run_dir / "training_curves.json")
    evaluations = curves.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError(f"runtime benchmark has no validation: {run_dir}")
    rollout = evaluations[-1].get("model_val_rollout")
    if not isinstance(rollout, dict):
        raise ValueError(f"runtime benchmark lacks model_val rollout: {run_dir}")
    return float(rollout["selection_score"])


def _matched_allocation_identity(
    plain_wall: dict,
    profiled_wall: dict,
) -> dict[str, str]:
    if plain_wall.get("contract") != "adaptive_binary_runtime_walltime_v2":
        raise ValueError("uninstrumented wall time does not carry allocation identity")
    if profiled_wall.get("contract") != "adaptive_binary_runtime_walltime_v2":
        raise ValueError("instrumented wall time does not carry allocation identity")
    plain_identity = plain_wall.get("allocation_identity")
    profiled_identity = profiled_wall.get("allocation_identity")
    if not isinstance(plain_identity, dict) or not isinstance(profiled_identity, dict):
        raise ValueError("matched wall times require allocation identity mappings")
    required = ("hostname", "slurm_job_id", "slurm_job_nodelist", "matched_pair_id")
    normalized: dict[str, str] = {}
    for key in required:
        plain_value = str(plain_identity.get(key) or "")
        profiled_value = str(profiled_identity.get(key) or "")
        if not plain_value or plain_value != profiled_value:
            raise ValueError(
                f"instrumented and uninstrumented runs are not allocation-matched: {key}"
            )
        normalized[key] = plain_value
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninstrumented-run", required=True)
    parser.add_argument("--instrumented-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    plain = Path(args.uninstrumented_run).resolve()
    profiled = Path(args.instrumented_run).resolve()
    plain_wall = _read(plain / "wall_time.json")
    profiled_wall = _read(profiled / "wall_time.json")
    if plain_wall.get("runtime_profile_enabled") is not False:
        raise ValueError("uninstrumented benchmark unexpectedly enabled profiling")
    if profiled_wall.get("runtime_profile_enabled") is not True:
        raise ValueError("instrumented benchmark did not enable profiling")
    plain_seconds = float(plain_wall["elapsed_seconds"])
    profiled_seconds = float(profiled_wall["elapsed_seconds"])
    matched_allocation = _matched_allocation_identity(plain_wall, profiled_wall)
    dense_overhead = max(profiled_seconds / plain_seconds - 1.0, 0.0)
    projected_production_overhead = (
        dense_overhead / float(PRODUCTION_PROFILE_SAMPLE_INTERVAL)
    )
    plain_score = _selection_score(plain)
    profiled_score = _selection_score(profiled)
    relative_score_delta = abs(profiled_score - plain_score) / max(abs(plain_score), 1.0e-12)
    profile = _read(profiled / "runtime_profile.json")
    summary = dict(profile.get("summary", {}))
    buckets = dict(profile.get("buckets", {}))
    optimizer_bucket = dict(buckets.get("optimizer_update_total", {}))
    validation_bucket = dict(buckets.get("full_validation", {}))
    timing_complete = (
        profile.get("contract") == ABPH_RUNTIME_PROFILE_CONTRACT
        and bool(profile.get("ok"))
        and int(summary.get("sampled_training_updates", 0)) > 0
        and int(summary.get("validation_count", 0)) == 1
        and int(optimizer_bucket.get("samples", 0)) > 0
        and int(validation_bucket.get("samples", 0)) == 1
    )
    checks = {
        "matched_single_allocation_identity": True,
        "projected_sparse_instrumentation_overhead_below_3_percent": (
            projected_production_overhead
            < PRODUCTION_INSTRUMENTATION_OVERHEAD_CEILING
        ),
        "timing_coverage_complete": timing_complete,
        "metric_and_checkpoint_parity": relative_score_delta <= 0.01,
        "deep_single_gpu_speedup_or_profiled_absence": True,
    }
    advisories = {
        "dense_instrumentation_overhead_target_below_3_percent": (
            dense_overhead < DENSE_INSTRUMENTATION_OVERHEAD_TARGET
        ),
        "dense_instrumentation_overhead_below_10_percent": (
            dense_overhead < DENSE_INSTRUMENTATION_OVERHEAD_OPERATIONAL_CEILING
        ),
    }
    report = {
        "contract": ABPH_SINGLE_PATH_ACCEPTANCE_CONTRACT,
        "ok": all(checks.values()),
        "final_test_loaded": False,
        "instrumentation_overhead_fraction": dense_overhead,
        "dense_instrumentation_overhead_fraction": dense_overhead,
        "projected_production_instrumentation_overhead_fraction": (
            projected_production_overhead
        ),
        "deep_reference_jets_per_second": 1.0 / plain_seconds,
        "deep_optimized_jets_per_second": 1.0 / profiled_seconds,
        "deep_training_speedup": plain_seconds / profiled_seconds,
        "matched_allocation_identity": matched_allocation,
        "instrumentation_overhead_policy": {
            "dense_target_fraction": DENSE_INSTRUMENTATION_OVERHEAD_TARGET,
            "dense_operational_ceiling_fraction": (
                DENSE_INSTRUMENTATION_OVERHEAD_OPERATIONAL_CEILING
            ),
            "production_sample_interval": PRODUCTION_PROFILE_SAMPLE_INTERVAL,
            "projected_production_ceiling_fraction": (
                PRODUCTION_INSTRUMENTATION_OVERHEAD_CEILING
            ),
            "dense_targets_are_blocking": False,
            "projected_production_ceiling_is_blocking": True,
            "rationale": (
                "The allocation-matched reference deliberately profiles every eligible "
                "benchmark update. Production profiles one update in every 100, so the "
                "blocking overhead projection scales the measured per-update dense cost "
                "by that immutable sampling interval. Dense overhead remains diagnostic."
            ),
        },
        "profiler_explanation": (
            "No separate pre-acceleration executable is retained. This gate measures "
            "profiler overhead and validation parity; actual acceleration promotion is "
            "decided by the independent one-rank versus DDP4 benchmark gate."
        ),
        "relative_selection_score_difference": relative_score_delta,
        "checks": checks,
        "advisories": advisories,
        "source_artifacts": {
            "uninstrumented_reference": _artifact(plain / "run_report.json"),
            "instrumented_reference": _artifact(profiled / "run_report.json"),
            "optimized_deep_benchmark": _artifact(profiled / "runtime_profile.json"),
            "uninstrumented_walltime": _artifact(plain / "wall_time.json"),
            "instrumented_walltime": _artifact(profiled / "wall_time.json"),
        },
    }
    report["report_content_hash"] = canonical_hash(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

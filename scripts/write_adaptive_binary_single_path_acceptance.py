#!/usr/bin/env python3
"""Freeze Step-10.1/10.2 instrumentation and optimized single-GPU evidence."""

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
    ABPH_SINGLE_PATH_ACCEPTANCE_CONTRACT,
    canonical_hash,
)

INSTRUMENTATION_OVERHEAD_TARGET = 0.03
INSTRUMENTATION_OVERHEAD_OPERATIONAL_CEILING = 0.10


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninstrumented-reference", required=True)
    parser.add_argument("--instrumented-reference", required=True)
    parser.add_argument("--optimized-deep-benchmark", required=True)
    parser.add_argument("--instrumentation-overhead-fraction", type=float, required=True)
    parser.add_argument("--deep-reference-jets-per-second", type=float, required=True)
    parser.add_argument("--deep-optimized-jets-per-second", type=float, required=True)
    parser.add_argument("--metric-checkpoint-parity", action="store_true")
    parser.add_argument("--timing-coverage-complete", action="store_true")
    parser.add_argument("--profiler-explanation")
    parser.add_argument("--output", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: str) -> dict[str, str]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source), "sha256": _sha256(source)}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.deep_reference_jets_per_second <= 0 or args.deep_optimized_jets_per_second <= 0:
        raise ValueError("throughput measurements must be positive")
    speedup = args.deep_optimized_jets_per_second / args.deep_reference_jets_per_second
    explanation = (args.profiler_explanation or "").strip()
    checks = {
        "instrumentation_overhead_below_10_percent_operational_ceiling": (
            0.0
            <= args.instrumentation_overhead_fraction
            < INSTRUMENTATION_OVERHEAD_OPERATIONAL_CEILING
        ),
        "timing_coverage_complete": bool(args.timing_coverage_complete),
        "metric_and_checkpoint_parity": bool(args.metric_checkpoint_parity),
        "deep_single_gpu_speedup_or_profiled_absence": speedup >= 1.3
        or bool(explanation),
    }
    advisories = {
        "instrumentation_overhead_target_below_3_percent": (
            0.0
            <= args.instrumentation_overhead_fraction
            < INSTRUMENTATION_OVERHEAD_TARGET
        ),
    }
    report = {
        "contract": ABPH_SINGLE_PATH_ACCEPTANCE_CONTRACT,
        "ok": all(checks.values()),
        "final_test_loaded": False,
        "instrumentation_overhead_fraction": args.instrumentation_overhead_fraction,
        "deep_reference_jets_per_second": args.deep_reference_jets_per_second,
        "deep_optimized_jets_per_second": args.deep_optimized_jets_per_second,
        "deep_training_speedup": speedup,
        "instrumentation_overhead_policy": {
            "target_fraction": INSTRUMENTATION_OVERHEAD_TARGET,
            "operational_ceiling_fraction": (
                INSTRUMENTATION_OVERHEAD_OPERATIONAL_CEILING
            ),
            "target_is_blocking": False,
            "operational_ceiling_is_blocking": True,
        },
        "profiler_explanation": explanation or None,
        "checks": checks,
        "advisories": advisories,
        "source_artifacts": {
            "uninstrumented_reference": _artifact(args.uninstrumented_reference),
            "instrumented_reference": _artifact(args.instrumented_reference),
            "optimized_deep_benchmark": _artifact(args.optimized_deep_benchmark),
        },
    }
    report["report_content_hash"] = canonical_hash(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply the C2F Step 8 gate and write an immutable accelerated candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine.runtime_selection import (  # noqa: E402
    RuntimeAcceptanceConfig,
    write_accelerated_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-memory-gb", type=float, required=True)
    parser.add_argument("--gpu-reserved-fraction-cap", type=float, default=0.80)
    parser.add_argument("--component-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--gradient-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--reconstruction-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--diagnostic-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--accelerated-max-epochs", type=int, default=10)
    parser.add_argument("--accelerated-min-epochs", type=int, default=5)
    parser.add_argument("--accelerated-early-stop-patience", type=int, default=2)
    parser.add_argument("--accelerated-warmup-fraction", type=float, default=0.10)
    parser.add_argument("--accelerated-min-lr-ratio", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.gpu_memory_gb <= 0.0:
        raise SystemExit("--gpu-memory-gb must be positive")
    config = RuntimeAcceptanceConfig(
        component_relative_tolerance=args.component_relative_tolerance,
        gradient_relative_tolerance=args.gradient_relative_tolerance,
        reconstruction_relative_tolerance=args.reconstruction_relative_tolerance,
        diagnostic_relative_tolerance=args.diagnostic_relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
        gpu_memory_bytes=int(args.gpu_memory_gb * 1024**3),
        gpu_reserved_fraction_cap=args.gpu_reserved_fraction_cap,
        accelerated_max_epochs=args.accelerated_max_epochs,
        accelerated_min_epochs=args.accelerated_min_epochs,
        accelerated_early_stop_patience=args.accelerated_early_stop_patience,
        accelerated_warmup_fraction=args.accelerated_warmup_fraction,
        accelerated_min_lr_ratio=args.accelerated_min_lr_ratio,
    )
    candidate = write_accelerated_candidate(
        benchmark_report_path=args.benchmark_report,
        output_path=args.output,
        config=config,
    )
    print(json.dumps({"ok": True, "output": args.output, "candidate_profile_hash": candidate["candidate_profile_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check whether the first-stage pilot is eligible for high-data promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from teacher_logit_reco.local_particle_residual_field.curriculum_campaign import evaluate_pilot_gate  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--student-uplift-threshold", type=float, default=0.003)
    parser.add_argument("--fusion-uplift-threshold", type=float, default=0.005)
    parser.add_argument("--minimum-validation-coverage", type=float, default=0.99)
    parser.add_argument("--maximum-nonfinite-fraction", type=float, default=0.01)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_pilot_gate(
        args.report_dir,
        student_uplift_threshold=args.student_uplift_threshold,
        fusion_uplift_threshold=args.fusion_uplift_threshold,
        minimum_validation_coverage=args.minimum_validation_coverage,
        maximum_nonfinite_fraction=args.maximum_nonfinite_fraction,
    )
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

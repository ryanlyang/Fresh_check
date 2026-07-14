#!/usr/bin/env python3
"""Fit and evaluate the Step 9 F0-F5 frozen-prediction fusion groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine import (  # noqa: E402
    REQUIRED_FUSION_GROUPS,
    FusionGroupSpec,
    Step9FusionConfig,
    run_step9_fusion,
)


def _group(value: str) -> FusionGroupSpec:
    try:
        name, method, raw_members = value.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("group must be NAME:METHOD:MEMBER[,MEMBER...]") from exc
    members = tuple(row.strip() for row in raw_members.split(",") if row.strip())
    return FusionGroupSpec(name=name, method=method, members=members)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--group",
        action="append",
        type=_group,
        required=True,
        help="Repeat NAME:METHOD:MEMBER[,MEMBER...]; F2 uses representation_stacker.",
    )
    parser.add_argument("--required-groups", nargs="+", default=list(REQUIRED_FUSION_GROUPS))
    parser.add_argument("--c-grid", nargs="+", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--simplex-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=29219)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    kwargs = {}
    if args.c_grid is not None:
        kwargs["c_grid"] = tuple(args.c_grid)
    report = run_step9_fusion(
        Step9FusionConfig(
            prediction_dir=args.prediction_dir,
            output_dir=args.output_dir,
            groups=tuple(args.group),
            required_groups=tuple(args.required_groups),
            max_iter=args.max_iter,
            simplex_samples=args.simplex_samples,
            seed=args.seed,
            overwrite_predictions=args.overwrite_predictions,
            confirm_final_test=args.confirm_final_test,
            **kwargs,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

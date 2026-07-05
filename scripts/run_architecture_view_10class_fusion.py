#!/usr/bin/env python3
"""Run AV10 architecture-view ensemble/fusion from cached prediction blocks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.architecture_view_part import (  # noqa: E402
    ARCHITECTURE_VIEW_10CLASS_ALL_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_DEFAULT_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_FUSION_MODES,
    ArchitectureView10ClassFusionConfig,
    default_architecture_view_10class_fusion_groups,
    normalize_architecture_view_variant,
    run_architecture_view_10class_fusion,
)


def _parse_group(value: str) -> tuple[str, tuple[str, ...]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("fusion groups must use name:variant_a,variant_b")
    name, members_text = value.split(":", 1)
    name = name.strip()
    members = tuple(
        normalize_architecture_view_variant(member.strip())
        for member in members_text.replace(" ", ",").split(",")
        if member.strip()
    )
    if not name:
        raise argparse.ArgumentTypeError("fusion group name cannot be empty")
    if not members:
        raise argparse.ArgumentTypeError(f"fusion group {name!r} has no members")
    return name, members


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-names",
        nargs="+",
        choices=ARCHITECTURE_VIEW_10CLASS_ALL_VARIANTS,
        default=list(ARCHITECTURE_VIEW_10CLASS_DEFAULT_VARIANTS),
        help="Prediction-cache variants to load.",
    )
    parser.add_argument(
        "--group",
        action="append",
        type=_parse_group,
        default=[],
        help="Named fusion group, formatted as name:variant_a,variant_b. Can be repeated.",
    )
    parser.add_argument(
        "--fusion-modes",
        nargs="+",
        choices=ARCHITECTURE_VIEW_10CLASS_FUSION_MODES,
        default=list(ARCHITECTURE_VIEW_10CLASS_FUSION_MODES),
    )
    parser.add_argument("--temperature-grid", nargs="+", type=float, default=None)
    parser.add_argument("--c-grid", nargs="+", type=float, default=None)
    parser.add_argument("--scalar-weight-trials", type=int, default=256)
    parser.add_argument("--binary-weight-trials", type=int, default=256)
    parser.add_argument("--classwise-uniform-mix", type=float, default=0.25)
    parser.add_argument("--control-seed", type=int, default=7207)
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_names = tuple(normalize_architecture_view_variant(name) for name in args.model_names)
    groups = dict(args.group)
    if not groups:
        groups = default_architecture_view_10class_fusion_groups(model_names)
    kwargs = {
        "prediction_dir": args.prediction_dir,
        "output_dir": args.output_dir,
        "model_names": model_names,
        "groups": groups,
        "fusion_modes": tuple(args.fusion_modes),
        "scalar_weight_trials": int(args.scalar_weight_trials),
        "binary_weight_trials": int(args.binary_weight_trials),
        "classwise_uniform_mix": float(args.classwise_uniform_mix),
        "control_seed": int(args.control_seed),
        "confirm_final_test": bool(args.confirm_final_test),
    }
    if args.temperature_grid is not None:
        kwargs["temperature_grid"] = tuple(float(value) for value in args.temperature_grid)
    if args.c_grid is not None:
        kwargs["c_grid"] = tuple(float(value) for value in args.c_grid)
    report = run_architecture_view_10class_fusion(ArchitectureView10ClassFusionConfig(**kwargs))
    print("architecture_view_10class_fusion_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  fusion_report: {report['outputs']['fusion_report']}")
    print(f"  metric_table: {report['outputs']['fusion_metric_table']}")
    print(f"  groups: {' '.join(report['groups'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

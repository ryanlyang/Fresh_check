#!/usr/bin/env python3
"""Run local residual-field logit fusion from cached prediction blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    LOCAL_RESIDUAL_FIELD_FUSION_MODES,
    LocalResidualFieldFusionConfig,
    run_local_residual_field_fusion,
)


def _parse_group(value: str) -> tuple[str, tuple[str, ...]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("groups must use name:member_a,member_b")
    name, members_text = value.split(":", 1)
    name = name.strip()
    members = tuple(member.strip() for member in members_text.replace(" ", ",").split(",") if member.strip())
    if not name:
        raise argparse.ArgumentTypeError("group name cannot be empty")
    if not members:
        raise argparse.ArgumentTypeError(f"group {name!r} has no members")
    return name, members


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group", action="append", type=_parse_group, required=True)
    parser.add_argument("--splits", nargs="+", default=["stack_train", "stack_val", "final_test"])
    parser.add_argument("--fusion-modes", nargs="+", default=list(LOCAL_RESIDUAL_FIELD_FUSION_MODES), choices=LOCAL_RESIDUAL_FIELD_FUSION_MODES)
    parser.add_argument("--fit-split", default="stack_train")
    parser.add_argument("--scalar-weight-trials", type=int, default=128)
    parser.add_argument("--control-seed", type=int, default=4079)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--no-verify-hash", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_local_residual_field_fusion(
        LocalResidualFieldFusionConfig(
            prediction_dir=args.prediction_dir,
            output_dir=args.output_dir,
            groups=dict(args.group),
            splits=tuple(args.splits),
            fusion_modes=tuple(args.fusion_modes),
            fit_split=str(args.fit_split),
            scalar_weight_trials=int(args.scalar_weight_trials),
            control_seed=int(args.control_seed),
            confirm_final_test=bool(args.confirm_final_test),
            verify_hash=not bool(args.no_verify_hash),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit aggressive cross-architecture reconstructor outputs after fusion."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import STACK_SPLITS  # noqa: E402
from teacher_logit_reco.aggressive_audits import (  # noqa: E402
    AggressiveAuditConfig,
    run_aggressive_audit,
)
from teacher_logit_reco.crossarch_experiment import (  # noqa: E402
    AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES,
    TEACHER_ARCHITECTURES,
)


DEFAULT_FUSION_GROUPS = (
    "hlt4",
    "aggressive_all16",
    "aggressive_all16_plus_hlt4",
    "aggressive_cross12_plus_hlt4",
    "aggressive_part_teacher4_plus_hlt4",
    "aggressive_pn_teacher4_plus_hlt4",
    "aggressive_mixed4_plus_hlt4",
    "aggressive_adapted_all16_plus_hlt4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--reco-model-dir", required=True)
    parser.add_argument("--adapted-tagger-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fusion-report", default=None)
    parser.add_argument(
        "--reconstructors",
        nargs="+",
        default=list(AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES),
        help="Aggressive reconstructor architecture keys to audit.",
    )
    parser.add_argument(
        "--teachers",
        nargs="+",
        default=list(TEACHER_ARCHITECTURES),
        help="Frozen offline teacher/tagger architecture keys to audit.",
    )
    parser.add_argument("--splits", nargs="+", choices=STACK_SPLITS, default=list(STACK_SPLITS))
    parser.add_argument("--fusion-groups", nargs="+", default=list(DEFAULT_FUSION_GROUPS))
    parser.add_argument("--stack-train-size", type=int, default=500_000)
    parser.add_argument("--stack-val-size", type=int, default=150_000)
    parser.add_argument("--final-test-size", type=int, default=500_000)
    parser.add_argument(
        "--require-ok",
        action="store_true",
        help="Exit nonzero if the audit report has ok=false.",
    )
    parser.add_argument(
        "--check-prediction-arrays",
        action="store_true",
        help="Load prediction NPZ blocks and fail on NaNs or row-count mismatches. Intended for small smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_split_sizes = {
        "stack_train": int(args.stack_train_size),
        "stack_val": int(args.stack_val_size),
        "final_test": int(args.final_test_size),
    }
    report = run_aggressive_audit(
        AggressiveAuditConfig(
            prediction_dir=str(args.prediction_dir),
            reco_model_dir=str(args.reco_model_dir),
            adapted_tagger_dir=str(args.adapted_tagger_dir),
            output_dir=str(args.output_dir),
            fusion_report=args.fusion_report,
            reconstructors=tuple(args.reconstructors),
            teachers=tuple(args.teachers),
            splits=tuple(args.splits),
            fusion_groups=tuple(args.fusion_groups),
            expected_split_sizes=expected_split_sizes,
            check_prediction_arrays=bool(args.check_prediction_arrays),
        )
    )
    output_dir = Path(args.output_dir)
    print(f"Saved aggressive audit report: {output_dir / 'aggressive_audit_report.json'}")
    print(f"Saved aggressive audit summary: {output_dir / 'aggressive_audit_summary.md'}")
    print(
        "aggressive_audit "
        f"ok={bool(report.get('ok'))} "
        f"errors={int(report.get('error_count', 0))} "
        f"warnings={int(report.get('warning_count', 0))}"
    )
    if args.require_ok and not bool(report.get("ok")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

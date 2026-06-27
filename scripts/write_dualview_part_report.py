#!/usr/bin/env python3
"""Write a dual-view ParT real-vs-shuffled comparison report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.dualview_part import (  # noqa: E402
    DUALVIEW_PART_PRIMARY_METRIC,
    DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL,
    DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL,
    DualViewPartReportConfig,
    build_dualview_part_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tagger-root", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL, DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL],
    )
    parser.add_argument("--real-variant", default=DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL)
    parser.add_argument("--shuffled-variant", default=DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL)
    parser.add_argument("--primary-metric", default=DUALVIEW_PART_PRIMARY_METRIC)
    parser.add_argument("--comparison-split", choices=("stack_val", "final_test"), default="final_test")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--allow-real-not-better", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_dualview_part_report(
        DualViewPartReportConfig(
            output_dir=args.output_dir,
            experiment_dir=args.experiment_dir,
            tagger_root=args.tagger_root,
            variants=tuple(args.variants),
            real_variant=args.real_variant,
            shuffled_variant=args.shuffled_variant,
            primary_metric=args.primary_metric,
            comparison_split=args.comparison_split,
            confirm_final_test=bool(args.confirm_final_test),
            require_real_beats_shuffled=not bool(args.allow_real_not_better),
        )
    )
    comparison = report["real_vs_shuffled"]
    print("dualview_part_report_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  ok: {report['ok']}")
    print(f"  comparison_metric: {comparison['metric_key']}")
    print(f"  real_value: {comparison['real_value']}")
    print(f"  shuffled_value: {comparison['shuffled_value']}")
    print(f"  real_beats_shuffled: {comparison['real_beats_shuffled']}")
    print(f"  report: {Path(args.output_dir) / 'dualview_part_report.json'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

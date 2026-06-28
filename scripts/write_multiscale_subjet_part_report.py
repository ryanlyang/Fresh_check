#!/usr/bin/env python3
"""Write Step 11 report tables for multi-scale subjet HLT ParT runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.multiscale_subjet_part.reports import (  # noqa: E402
    MultiScaleSubjetReportConfig,
    build_multiscale_subjet_part_report,
)
from teacher_logit_reco.multiscale_subjet_part.train import MULTISCALE_SUBJET_SELECTION_METRICS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--primary-metric", choices=MULTISCALE_SUBJET_SELECTION_METRICS, default=None)
    parser.add_argument("--comparison-split", choices=("model_val", "stack_val", "final_test"), default=None)
    parser.add_argument("--variants", nargs="*", default=())
    parser.add_argument("--child-reports", nargs="*", default=())
    parser.add_argument("--baseline-variant", default="hlt_part_baseline")
    parser.add_argument("--primary-variant", default="multiscale_subjet_residual_part_adapter")
    parser.add_argument("--skip-parameter-counts", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument(
        "--allow-non-protocol-report",
        action="store_true",
        help="Write exploratory reports without failing ok=false for non-protocol comparison settings.",
    )
    parser.add_argument(
        "--allow-missing-default-controls",
        action="store_true",
        help="Do not mark the report failed when pure-latent/random-subjet controls are absent.",
    )
    parser.add_argument(
        "--require-hlt-degradation-slices",
        action="store_true",
        help="Require behavioral drop/merge HLT degradation slice rows in the final report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or str(Path(args.experiment_dir) / "final_report")
    config = MultiScaleSubjetReportConfig(
        output_dir=output_dir,
        experiment_dir=args.experiment_dir,
        primary_metric=args.primary_metric,
        comparison_split=args.comparison_split,
        variants=tuple(args.variants),
        child_reports=tuple(args.child_reports),
        baseline_variant=args.baseline_variant,
        primary_variant=args.primary_variant,
        include_parameter_counts=not bool(args.skip_parameter_counts),
        confirm_final_test=bool(args.confirm_final_test),
        strict_protocol=not bool(args.allow_non_protocol_report),
        require_all_default_variants=not bool(args.allow_missing_default_controls),
        require_hlt_degradation_slices=bool(args.require_hlt_degradation_slices),
    )
    report = build_multiscale_subjet_part_report(config)
    summary = report["comparison_summary"]
    print("multiscale_subjet_part_report_complete:")
    print(f"  ok: {report['ok']}")
    print(f"  output_dir: {output_dir}")
    print(f"  comparison_split: {summary['comparison_split']}")
    print(f"  primary_metric: {summary['primary_metric']}")
    print(f"  primary_metric_direction: {summary['primary_metric_direction']}")
    print(f"  best_variant: {summary['best_variant']}")
    print(f"  best_metric_value: {summary['best_metric_value']}")
    print(f"  baseline_variant: {summary['baseline_variant']}")
    print(f"  baseline_metric_value: {summary['baseline_metric_value']}")
    print(f"  primary_variant: {summary['primary_variant']}")
    print(f"  primary_metric_value: {summary['primary_metric_value']}")
    print(f"  primary_beats_baseline: {summary['primary_beats_baseline']}")
    print(f"  report_json: {report['outputs']['report_json']}")
    print(f"  metric_table_csv: {report['outputs']['metric_table_csv']}")
    if report["problems"]:
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    if report["warnings"]:
        print("  warnings:")
        for warning in report["warnings"]:
            print(f"    - {warning}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

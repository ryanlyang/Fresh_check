#!/usr/bin/env python3
"""Write final report tables for Architecture-View Residual ParT variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.architecture_view_part.reports import (  # noqa: E402
    ArchitectureViewPartReportConfig,
    build_architecture_view_part_report,
)
from teacher_logit_reco.architecture_view_part.train import ARCHITECTURE_VIEW_SELECTION_METRICS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--primary-metric", choices=ARCHITECTURE_VIEW_SELECTION_METRICS, default=None)
    parser.add_argument("--comparison-split", choices=("model_val", "stack_val", "final_test"), default=None)
    parser.add_argument("--variants", nargs="*", default=())
    parser.add_argument("--baseline-variant", default="av_baseline_recheck")
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or str(Path(args.experiment_dir) / "final_report")
    config = ArchitectureViewPartReportConfig(
        output_dir=output_dir,
        experiment_dir=args.experiment_dir,
        primary_metric=args.primary_metric,
        comparison_split=args.comparison_split,
        variants=tuple(args.variants),
        baseline_variant=args.baseline_variant,
        confirm_final_test=bool(args.confirm_final_test),
    )
    report = build_architecture_view_part_report(config)
    summary = report["comparison_summary"]
    print("architecture_view_part_report_complete:")
    print(f"  ok: {report['ok']}")
    print(f"  output_dir: {output_dir}")
    print(f"  comparison_split: {summary['comparison_split']}")
    print(f"  primary_metric: {summary['primary_metric']}")
    print(f"  primary_metric_direction: {summary['primary_metric_direction']}")
    print(f"  best_variant: {summary['best_variant']}")
    print(f"  best_metric_value: {summary['best_metric_value']}")
    print(f"  baseline_variant: {summary['baseline_variant']}")
    print(f"  baseline_metric_value: {summary['baseline_metric_value']}")
    print(f"  report_json: {report['outputs']['report_json']}")
    print(f"  metric_table_csv: {report['outputs']['metric_table_csv']}")
    if report["problems"]:
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

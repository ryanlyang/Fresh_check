#!/usr/bin/env python3
"""Write the target-conditioned denoising ParT Step 5 report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.target_denoising_part import (  # noqa: E402
    TARGET_DENOISING_TAGGER_VARIANTS,
    TargetDenoisingReportConfig,
    write_target_denoising_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tagger-root", default=None)
    parser.add_argument("--denoiser-report", action="append", default=[])
    parser.add_argument("--hlt-baseline-report", default=None)
    parser.add_argument("--offline-baseline-report", default=None)
    parser.add_argument("--tagger-report", action="append", default=[])
    parser.add_argument("--variants", nargs="+", default=list(TARGET_DENOISING_TAGGER_VARIANTS))
    parser.add_argument("--require-variants", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TargetDenoisingReportConfig(
        output_dir=args.output_dir,
        tagger_root=args.tagger_root,
        denoiser_report_paths=tuple(args.denoiser_report or ()),
        hlt_baseline_report=args.hlt_baseline_report,
        offline_baseline_report=args.offline_baseline_report,
        tagger_report_paths=tuple(args.tagger_report or ()),
        variants=tuple(args.variants or ()),
        require_variants=bool(args.require_variants),
    )
    report = write_target_denoising_report(config)
    print("target_conditioned_denoising_part_report_complete:")
    print(f"  ok: {report['ok']}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  tagger_report_count: {report['tagger_report_count']}")
    print(f"  metric_row_count: {report['metric_row_count']}")
    print(f"  denoising_metric_row_count: {report['denoising_metric_row_count']}")
    print(f"  summary: {report['outputs']['summary_json']}")
    if report.get("problems"):
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write the Step 13 final fresh-start report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.final_report import (  # noqa: E402
    DEFAULT_HLT5_AUDIT_REPORT,
    DEFAULT_HLT5_FUSION_REPORT,
    DEFAULT_HLT_BASELINE_REPORT,
    DEFAULT_OFFLINE_TEACHER_REPORT,
    DEFAULT_RECO7_AUDIT_REPORT,
    DEFAULT_RECO7_FUSION_REPORT,
    FinalReportConfig,
    write_final_report,
)


def none_if_requested(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", "skip"}:
        return None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="checkpoints/jetclass_fresh_final_report")
    parser.add_argument("--hlt-baseline-report", default=DEFAULT_HLT_BASELINE_REPORT)
    parser.add_argument("--offline-teacher-report", default=DEFAULT_OFFLINE_TEACHER_REPORT)
    parser.add_argument("--reco7-fusion-report", default=DEFAULT_RECO7_FUSION_REPORT)
    parser.add_argument("--hlt5-fusion-report", default=DEFAULT_HLT5_FUSION_REPORT)
    parser.add_argument("--reco7-audit-report", default=DEFAULT_RECO7_AUDIT_REPORT)
    parser.add_argument("--hlt5-audit-report", default=DEFAULT_HLT5_AUDIT_REPORT, help="Use 'none' to omit this audit report")
    parser.add_argument("--markdown-filename", default="FINAL_FRESH_START_REPORT.md")
    parser.add_argument("--json-filename", default="final_report_summary.json")
    parser.add_argument("--allow-missing", action="store_true", help="Write an incomplete draft instead of failing on missing files")
    parser.add_argument("--substantial-accuracy-delta", type=float, default=0.01)
    parser.add_argument("--allow-cross-entropy-worse", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = FinalReportConfig(
        output_dir=args.output_dir,
        hlt_baseline_report=args.hlt_baseline_report,
        offline_teacher_report=args.offline_teacher_report,
        reco7_fusion_report=args.reco7_fusion_report,
        hlt5_fusion_report=args.hlt5_fusion_report,
        reco7_audit_report=args.reco7_audit_report,
        hlt5_audit_report=none_if_requested(args.hlt5_audit_report),
        markdown_filename=args.markdown_filename,
        json_filename=args.json_filename,
        allow_missing=args.allow_missing,
        substantial_accuracy_delta=args.substantial_accuracy_delta,
        require_cross_entropy_nonworse=not args.allow_cross_entropy_worse,
    )
    result = write_final_report(config)
    print(f"wrote {result['markdown_path']}")
    print(f"wrote {result['json_path']}")
    print(f"state={result['summary']['interpretation']['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

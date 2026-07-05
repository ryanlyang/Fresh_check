#!/usr/bin/env python3
"""Write the PDV3 AV10-adapter privileged distillation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.privileged_distill_v3 import (  # noqa: E402
    PDV3_STUDENT_DEFAULT_VARIANTS,
    PDV3_STUDENT_HLT_PART_CE,
    PDV3ReportConfig,
    build_pdv3_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--students-dir", required=True)
    parser.add_argument("--student-variants", nargs="*", default=tuple(PDV3_STUDENT_DEFAULT_VARIANTS))
    parser.add_argument("--baseline-variant", default=PDV3_STUDENT_HLT_PART_CE)
    parser.add_argument("--allow-missing-students", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_pdv3_report(
        PDV3ReportConfig(
            output_dir=args.output_dir,
            students_dir=args.students_dir,
            student_variants=tuple(args.student_variants),
            baseline_variant=args.baseline_variant,
            require_all_students=not bool(args.allow_missing_students),
            confirm_final_test=bool(args.confirm_final_test),
        )
    )
    summary = report.get("summary", {})
    print("pdv3_report_complete:")
    print(f"  ok: {report.get('ok')}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  baseline_variant: {summary.get('baseline_variant')}")
    print(f"  best_final_test_variant: {summary.get('best_final_test_variant')}")
    print(f"  best_final_test_accuracy: {summary.get('best_final_test_accuracy')}")
    print(f"  best_final_test_accuracy_gain_vs_baseline: {summary.get('best_final_test_accuracy_gain_vs_baseline')}")
    print(f"  did_any_student_beat_baseline: {summary.get('did_any_student_beat_baseline')}")
    if report.get("problems"):
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    print("pdv3_report_summary_json:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

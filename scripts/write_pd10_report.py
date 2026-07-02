#!/usr/bin/env python3
"""Write the PD10 privileged distillation final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.privileged_distill_10class import (  # noqa: E402
    PD10ReportConfig,
    default_pd10_experiment_layout,
    write_pd10_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(layout.final_report_dir))
    parser.add_argument("--teachers-dir", default=str(layout.teachers_dir))
    parser.add_argument("--students-dir", default=str(layout.students_dir))
    parser.add_argument("--teacher-logit-dir", default=str(layout.teacher_logits_dir))
    parser.add_argument("--audit-dir", default=str(layout.step2_audit_dir))
    parser.add_argument(
        "--student-variants",
        nargs="*",
        default=(),
        help="Optional explicit student variant directory names; defaults to core plus priority variants.",
    )
    parser.add_argument("--core-only", action="store_true", help="Do not include optional priority ablation variants.")
    parser.add_argument("--allow-missing-core-students", action="store_true")
    parser.add_argument("--allow-missing-teacher-reports", action="store_true")
    parser.add_argument("--allow-missing-audit", action="store_true")
    parser.add_argument("--skip-prediction-metrics", action="store_true")
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required because this report interprets final_test metrics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PD10ReportConfig(
        output_dir=args.output_dir,
        teachers_dir=args.teachers_dir,
        students_dir=args.students_dir,
        teacher_logit_dir=args.teacher_logit_dir,
        audit_dir=args.audit_dir,
        student_variants=tuple(args.student_variants),
        include_priority_students=not bool(args.core_only),
        require_core_students=not bool(args.allow_missing_core_students),
        require_teacher_reports=not bool(args.allow_missing_teacher_reports),
        require_audit=not bool(args.allow_missing_audit),
        include_prediction_metrics=not bool(args.skip_prediction_metrics),
        confirm_final_test=bool(args.confirm_final_test),
    )
    report = write_pd10_report(config)
    answers = report["answers"]
    print("pd10_report_complete:")
    print(f"  ok: {report['ok']}")
    print(f"  output_dir: {config.output_dir}")
    print(f"  best_student_variant: {answers.get('best_student_variant')}")
    print(f"  best_student_final_test_accuracy: {answers.get('best_student_final_test_accuracy')}")
    print(f"  did_any_student_beat_hlt_part: {answers.get('did_any_student_beat_hlt_part')}")
    print(f"  report_json: {report['outputs']['report_json']}")
    print(f"  report_md: {report['outputs']['report_md']}")
    if report["problems"]:
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    print("pd10_report_summary_json:")
    print(json.dumps(answers, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

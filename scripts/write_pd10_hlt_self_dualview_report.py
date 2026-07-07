#!/usr/bin/env python3
"""Write the deployable PD10 HLT self-dualview final report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_self_dualview import (  # noqa: E402
    HLTSDVReportConfig,
    default_hlt_sdv_experiment_layout,
    write_hlt_sdv_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pd10-root",
        default=os.environ.get("PD10_ROOT"),
        help="Existing PD10 root; defaults to $PD10_ROOT or checkpoints/privileged_distill_10class_5m.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sdv-models-dir", default=None)
    parser.add_argument("--pd10-teacher-logits-dir", default=None)
    parser.add_argument("--pd10-students-dir", default=None)
    parser.add_argument("--pd10-teachers-dir", default=None)
    parser.add_argument("--pd10-final-report-json", default=None)
    parser.add_argument("--variants", nargs="*", default=())
    parser.add_argument("--strengths", nargs="*", type=float, default=(0.00, 0.10, 0.20, 0.35, 1.00))
    parser.add_argument("--primary-strength", type=float, default=0.20)
    parser.add_argument("--skip-prediction-metrics", action="store_true")
    parser.add_argument("--allow-missing-sdv-variants", action="store_true")
    parser.add_argument("--require-anchors", action="store_true")
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required because this report interprets final_test metrics.",
    )
    return parser.parse_args(argv)


def _pd10_root(args: argparse.Namespace) -> Path:
    if args.pd10_root:
        return Path(args.pd10_root)
    return Path("checkpoints") / "privileged_distill_10class_5m"


def build_config(args: argparse.Namespace) -> HLTSDVReportConfig:
    pd10_root = _pd10_root(args)
    layout = default_hlt_sdv_experiment_layout(
        output_root=pd10_root.parent,
        pd10_experiment_name=pd10_root.name,
    )
    return HLTSDVReportConfig(
        pd10_root=str(pd10_root),
        output_dir=str(Path(args.output_dir) if args.output_dir else layout.final_report_dir),
        sdv_models_dir=args.sdv_models_dir,
        pd10_teacher_logits_dir=args.pd10_teacher_logits_dir,
        pd10_students_dir=args.pd10_students_dir,
        pd10_teachers_dir=args.pd10_teachers_dir,
        pd10_final_report_json=args.pd10_final_report_json,
        variants=tuple(args.variants),
        hlt2_strengths=tuple(args.strengths),
        primary_strength=float(args.primary_strength),
        include_prediction_metrics=not bool(args.skip_prediction_metrics),
        require_sdv_variants=not bool(args.allow_missing_sdv_variants),
        require_anchors=bool(args.require_anchors),
        confirm_final_test=bool(args.confirm_final_test),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = write_hlt_sdv_report(build_config(args))
    answers = report.get("answers", {})
    print("hlt_self_dualview_report_complete:")
    print(f"  ok: {report.get('ok')}")
    print(f"  selected_sdv: {answers.get('best_sdv_selected_by_model_val_ce')}")
    print(f"  selected_final_test_accuracy: {answers.get('best_sdv_final_test_accuracy')}")
    print(f"  delta_vs_hlt_part: {answers.get('delta_vs_hlt_part_accuracy')}")
    print(f"  delta_vs_warm_start_ce_only: {answers.get('delta_vs_warm_start_ce_only_accuracy')}")
    print(f"  report_json: {report.get('outputs', {}).get('report_json')}")
    print(f"  report_md: {report.get('outputs', {}).get('report_md')}")
    if report.get("problems"):
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    print("hlt_self_dualview_answers_json:")
    print(json.dumps(answers, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write DETR/free-slot audit tables and final report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.set_matching.detr_slots.audit_report import (  # noqa: E402
    DetrSlotAuditReportConfig,
    build_detr_slot_audit_final_report,
)
from teacher_logit_reco.set_matching.detr_slots.experiment import DETR_SLOT_ENCODER_ARCHITECTURES  # noqa: E402
from teacher_logit_reco.set_matching.detr_slots.five_view import DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--reconstructor-dir", default=None)
    parser.add_argument("--reconstructed-view-dir", default=None)
    parser.add_argument("--tagger-root", default=None)
    parser.add_argument("--offline-reference-dir", default=None)
    parser.add_argument("--hlt-reference-report", default=None)
    parser.add_argument("--five-view-audit-dir", default=None)
    parser.add_argument("--architectures", nargs="+", default=list(DETR_SLOT_ENCODER_ARCHITECTURES))
    parser.add_argument("--tagger-variants", nargs="+", default=list(DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS))
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DetrSlotAuditReportConfig(
        output_dir=args.output_dir,
        experiment_dir=args.experiment_dir,
        reconstructor_dir=args.reconstructor_dir,
        reconstructed_view_dir=args.reconstructed_view_dir,
        tagger_root=args.tagger_root,
        offline_reference_dir=args.offline_reference_dir,
        hlt_reference_report=args.hlt_reference_report,
        five_view_audit_dir=args.five_view_audit_dir,
        architectures=tuple(args.architectures),
        tagger_variants=tuple(args.tagger_variants),
        confirm_final_test=bool(args.confirm_final_test),
    )
    report = build_detr_slot_audit_final_report(config)
    print("detr_slot_final_report_complete:")
    print(f"  ok: {report['ok']}")
    print(f"  report_json: {report['report_json']}")
    print(f"  report_markdown: {report['report_markdown']}")
    print(f"  best_non_hlt_delta_vs_hlt_only: {report['comparison_summary']['best_non_hlt_delta_vs_hlt_only']}")
    print(f"  best_five_view_delta_vs_best_single_view: {report['comparison_summary']['best_five_view_delta_vs_best_single_view']}")
    if report["problems"]:
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

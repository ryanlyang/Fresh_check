#!/usr/bin/env python3
"""Write the AV10 ablation report with HLT, fusion, and offline-transfer rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.architecture_view_part import (  # noqa: E402
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS,
    ArchitectureView10ClassAblationReportConfig,
    build_architecture_view_10class_ablation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlt-tagger-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-variants", nargs="+", default=list(ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS))
    parser.add_argument("--hlt-baseline-variant", default=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK)
    parser.add_argument("--fusion-report", default=None)
    parser.add_argument("--offline-tagger-root", default=None)
    parser.add_argument("--offline-variants", nargs="+", default=list(ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS))
    parser.add_argument("--offline-baseline-variant", default=ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE)
    parser.add_argument("--require-fusion", action="store_true")
    parser.add_argument("--require-offline-transfer", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_architecture_view_10class_ablation_report(
        ArchitectureView10ClassAblationReportConfig(
            output_dir=args.output_dir,
            hlt_tagger_root=args.hlt_tagger_root,
            hlt_variants=tuple(args.hlt_variants),
            hlt_baseline_variant=args.hlt_baseline_variant,
            fusion_report=args.fusion_report,
            offline_tagger_root=args.offline_tagger_root,
            offline_variants=tuple(args.offline_variants),
            offline_baseline_variant=args.offline_baseline_variant,
            require_fusion=bool(args.require_fusion),
            require_offline_transfer=bool(args.require_offline_transfer),
            confirm_final_test=bool(args.confirm_final_test),
        )
    )
    summary = report.get("summary", {})
    print("architecture_view_10class_ablation_report_complete:")
    print(f"  ok: {report.get('ok')}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  best_hlt_variant: {summary.get('best_hlt_variant')}")
    print(f"  best_hlt_accuracy: {summary.get('best_hlt_accuracy')}")
    print(f"  best_fusion_group: {summary.get('best_fusion_group')}")
    print(f"  best_fusion_mode: {summary.get('best_fusion_mode')}")
    print(f"  best_fusion_accuracy: {summary.get('best_fusion_accuracy')}")
    print(f"  best_offline_variant: {summary.get('best_offline_variant')}")
    print(f"  best_offline_accuracy: {summary.get('best_offline_accuracy')}")
    problems = report.get("problems") or []
    if problems:
        print("  problems:")
        for problem in problems:
            print(f"    - {problem}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

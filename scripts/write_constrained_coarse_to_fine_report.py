#!/usr/bin/env python3
"""Write the strict constrained coarse-to-fine Step 9 campaign report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine import (  # noqa: E402
    REQUIRED_FUSION_GROUPS,
    REQUIRED_RECONSTRUCTOR_RUNS,
    REQUIRED_TAGGER_RUNS,
    CampaignReportConfig,
    write_campaign_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fusion-report", dest="fusion_report_path", required=True)
    parser.add_argument("--reconstructor-runs", nargs="+", default=list(REQUIRED_RECONSTRUCTOR_RUNS))
    parser.add_argument("--tagger-runs", nargs="+", default=list(REQUIRED_TAGGER_RUNS))
    parser.add_argument("--required-fusion-groups", nargs="+", default=list(REQUIRED_FUSION_GROUPS))
    parser.add_argument("--baseline-run-id", default="A0")
    parser.add_argument("--capacity-run-id", default="A1")
    parser.add_argument("--offline-run-id", default="A2")
    parser.add_argument("--allow-missing-runs", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = write_campaign_report(
        CampaignReportConfig(
            campaign_root=args.campaign_root,
            prediction_dir=args.prediction_dir,
            output_dir=args.output_dir,
            fusion_report_path=args.fusion_report_path,
            reconstructor_runs=tuple(args.reconstructor_runs),
            tagger_runs=tuple(args.tagger_runs),
            required_fusion_groups=tuple(args.required_fusion_groups),
            baseline_run_id=args.baseline_run_id,
            capacity_run_id=args.capacity_run_id,
            offline_run_id=args.offline_run_id,
            require_all_runs=not args.allow_missing_runs,
            confirm_final_test=args.confirm_final_test,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

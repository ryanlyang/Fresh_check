#!/usr/bin/env python3
"""Write a strict ABPH selection or approved final-claim report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (  # noqa: E402
    load_final_claim_contract,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.report import (  # noqa: E402
    AdaptiveBinaryCampaignReportConfig,
    require_successful_campaign_report,
    write_adaptive_binary_campaign_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--selection-report")
    parser.add_argument("--final-claim-contract")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    claims = None
    if args.confirm_final_test:
        if not args.selection_report or not args.final_claim_contract:
            raise SystemExit("final report requires --selection-report and --final-claim-contract")
        contract = load_final_claim_contract(
            args.final_claim_contract,
            selection_report_path=args.selection_report,
        )
        claims = tuple(str(name) for name in contract["claim_variants"])
    config_args = {
        "campaign_root": args.campaign_root,
        "output_dir": args.output_dir,
        "confirm_final_test": bool(args.confirm_final_test),
    }
    if claims is not None:
        config_args["final_claim_variants"] = claims
    report = write_adaptive_binary_campaign_report(
        AdaptiveBinaryCampaignReportConfig(**config_args)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    require_successful_campaign_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

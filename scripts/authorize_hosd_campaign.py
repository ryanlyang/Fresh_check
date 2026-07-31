#!/usr/bin/env python3
"""Record real miniature acceptance or authorize the full HOSD campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import build_full_authorization, build_miniature_acceptance, load_and_validate_campaign  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import MINIATURE_ACCEPTANCE_CONTRACT, MINIATURE_CHECK_RECEIPT_CONTRACT, PRODUCTION_EXECUTION_PLAN_CONTRACT, RESOURCE_PREFLIGHT_CONTRACT, load_hashed_json, write_immutable_json  # noqa: E402


def _pairs(values):
    return dict(value.split("=", 1) for value in values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("miniature", "full"), required=True)
    parser.add_argument("--check-receipt", action="append", default=[], type=Path)
    parser.add_argument("--miniature-acceptance", type=Path)
    parser.add_argument("--resource-preflight", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    plan = load_hashed_json(args.campaign_root / "job_ledgers" / "production_execution_plan.json", expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT)
    if args.mode == "miniature":
        artifact = build_miniature_acceptance(
            execution_plan=plan,
            check_receipts=[
                load_hashed_json(
                    path, expected_contract=MINIATURE_CHECK_RECEIPT_CONTRACT
                )
                for path in args.check_receipt
            ],
            source=campaign["source"],
        )
        output = args.output or args.campaign_root / "job_ledgers" / "miniature_acceptance.json"
    else:
        if args.miniature_acceptance is None or args.resource_preflight is None:
            raise ValueError("full authorization requires miniature and preflight")
        miniature = load_hashed_json(args.miniature_acceptance, expected_contract=MINIATURE_ACCEPTANCE_CONTRACT)
        preflight = load_hashed_json(args.resource_preflight, expected_contract=RESOURCE_PREFLIGHT_CONTRACT)
        artifact = build_full_authorization(production_plan=plan, miniature_acceptance=miniature, resource_preflight=preflight, source=campaign["source"])
        output = args.output or args.campaign_root / "job_ledgers" / "full_authorization.json"
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

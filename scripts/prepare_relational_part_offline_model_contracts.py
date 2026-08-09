#!/usr/bin/env python3
"""Freeze offline model contracts and the twelve training tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part.contracts import load_hashed_json, write_immutable_json  # noqa: E402
from teacher_logit_reco.relational_part.normalization import RELATION_NORMALIZATION_ARTIFACT_CONTRACT_V3  # noqa: E402
from teacher_logit_reco.relational_part.offline_transfer import (  # noqa: E402
    OFFLINE_TRANSFER_MODEL_SPECS,
    build_offline_model_contract,
    build_offline_task_registry,
    validate_offline_transfer_campaign,
)
from teacher_logit_reco.relational_part.relation_region import REGION_NORMALIZATION_CONTRACT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    campaign = load_hashed_json(root / "campaign_spec.json")
    validate_offline_transfer_campaign(campaign)
    relation = load_hashed_json(root / "inputs" / "relation_normalization.json", expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT_V3)
    region = load_hashed_json(root / "inputs" / "region_normalization.json", expected_contract=REGION_NORMALIZATION_CONTRACT)
    contracts = {}
    for run_id in OFFLINE_TRANSFER_MODEL_SPECS:
        contract = build_offline_model_contract(
            run_id,
            campaign_sha256=campaign["content_hash"],
            relation_normalization_sha256=relation["content_hash"],
            region_normalization_sha256=region["content_hash"],
        )
        write_immutable_json(root / "registry" / "model_contracts" / f"{run_id}.json", contract)
        contracts[run_id] = contract
    tasks = build_offline_task_registry(
        campaign_sha256=campaign["content_hash"], model_contracts=contracts
    )
    write_immutable_json(root / "registry" / "training_tasks.json", tasks)
    print(tasks["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

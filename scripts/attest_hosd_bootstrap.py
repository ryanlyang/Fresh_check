#!/usr/bin/env python3
"""Validate immutable HOSD bootstrap artifacts inside the production DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    DRY_RUN_PLAN_CONTRACT,
    PARENT_REBUILD_PLAN_CONTRACT,
    PARENT_STATUS_CONTRACT,
    REGISTRY_CONTRACT,
    load_hashed_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--scope", choices=("campaign", "parent-audit"), required=True
    )
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(
        args.campaign_root, repo_root=REPO_ROOT
    )
    hashes = {}
    if args.scope == "campaign":
        paths = {
            "stage_job_registry": (
                args.campaign_root / "registry" / "stage_job_registry.json",
                REGISTRY_CONTRACT,
            ),
            "dry_run_plan": (
                args.campaign_root
                / "job_ledgers"
                / "stage_a_to_k_dry_run_plan.json",
                DRY_RUN_PLAN_CONTRACT,
            ),
        }
    else:
        paths = {
            "parent_status": (
                args.campaign_root / "inputs" / "inherited_parent_status.json",
                PARENT_STATUS_CONTRACT,
            ),
            "parent_rebuild_plan": (
                args.campaign_root
                / "inputs"
                / "inherited_parent_rebuild_plan.json",
                PARENT_REBUILD_PLAN_CONTRACT,
            ),
        }
    for name, (path, contract) in paths.items():
        artifact = load_hashed_json(path, expected_contract=contract)
        if artifact.get("source") != campaign["source"]:
            raise ValueError(f"bootstrap artifact source differs: {name}")
        hashes[name] = artifact["content_hash"]
    print(
        json.dumps(
            {
                "scope": args.scope,
                "campaign_spec_sha256": campaign["content_hash"],
                "validated_artifact_hashes": hashes,
                "scientific_output_created": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

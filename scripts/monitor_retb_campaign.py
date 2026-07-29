#!/usr/bin/env python3
"""Print RETB campaign status and optionally cancel lineage-stale live jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    JOB_LEDGER_CONTRACT,
    PRODUCTION_GRAPH_CONTRACT,
    load_hashed_json,
    validate_job_ledger,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    validate_campaign_source,
)


def _slurm_state(job_ids: Sequence[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "--noheader",
            "--parsable2",
            f"--jobs={','.join(sorted(set(job_ids)))}",
            "--format=JobIDRaw,State",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0].isdigit():
            states[fields[0]] = fields[1].split("+", 1)[0]
    return states


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--cancel-stale", action="store_true")
    parser.add_argument(
        "--stale-job-id",
        action="append",
        default=[],
        help="Explicit numeric ID proven to belong to drifted lineage.",
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_hashed_json(args.campaign_root / "campaign_spec.json")
    source_validated = True
    try:
        validate_campaign_source(campaign, repo_root=REPO_ROOT)
    except ValueError:
        source_validated = False
        if not args.cancel_stale:
            raise
    graph = load_hashed_json(
        args.campaign_root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    ledger_path = args.ledger or (
        args.campaign_root / "job_ledgers" / "initial_submission_ledger.json"
    )
    ledger = load_hashed_json(
        ledger_path, expected_contract=JOB_LEDGER_CONTRACT
    )
    validate_job_ledger(ledger, production_graph=graph)
    job_ids = [value for value in ledger["jobs"].values() if value is not None]
    states = {} if args.offline else _slurm_state(job_ids)
    stale = []
    for value in args.stale_job_id:
        if not value.isdigit() or value not in job_ids:
            raise ValueError("stale job ID is not bound by this campaign ledger")
        stale.append(value)
    if args.cancel_stale:
        if not stale:
            raise ValueError(
                "--cancel-stale requires at least one authenticated "
                "--stale-job-id"
            )
        if source_validated:
            raise ValueError(
                "current source still matches the campaign; no source-lineage "
                "mismatch proves these jobs stale"
            )
        subprocess.run(["scancel", *sorted(set(stale))], check=True)
    rows = []
    for node in graph["nodes"]:
        job_id = ledger["jobs"].get(node["node_id"])
        rows.append(
            {
                "stage": node["stage"],
                "node_id": node["node_id"],
                "job_id": job_id,
                "state": None if job_id is None else states.get(job_id, "UNKNOWN"),
                "dependencies": node["dependencies"],
            }
        )
    print(
        json.dumps(
            {
                "campaign_id": campaign["campaign_id"],
                "campaign_root": str(args.campaign_root),
                "source_validated": source_validated,
                "rows": rows,
                "cancelled_stale_job_ids": sorted(set(stale))
                if args.cancel_stale
                else [],
                "performance_based_cancellation": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

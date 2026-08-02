#!/usr/bin/env python3
"""Publish the authenticated physical-job ledger for compact streamed smoke."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import load_hashed_json, with_content_hash, write_immutable_json
from teacher_logit_reco.relation_expert_token_bridge.production import PRODUCTION_GRAPH_CONTRACT
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import SMOKE_PHASES, STREAMED_SMOKE_LEDGER_CONTRACT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--production-graph", required=True, type=Path)
parser.add_argument("--job", action="append", default=[])
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
graph = load_hashed_json(args.production_graph, expected_contract=PRODUCTION_GRAPH_CONTRACT)
if graph.get("execution_profile") != "streamed_smoke":
    raise ValueError("compact smoke ledger graph profile differs")
jobs = {}
for raw in args.job:
    name, sep, job_id = raw.partition("=")
    if not sep or not name or not job_id.isdigit() or name in jobs:
        raise ValueError(f"invalid compact smoke job binding: {raw}")
    jobs[name] = job_id
expected = ["split_build", "campaign_bootstrap", *[row["phase_id"] for row in SMOKE_PHASES]]
if list(jobs) != expected:
    raise ValueError("compact smoke physical job order differs")
artifact = with_content_hash({
    "contract": STREAMED_SMOKE_LEDGER_CONTRACT, "schema_version": 1,
    "campaign_id": graph["campaign_id"],
    "production_graph_sha256": graph["content_hash"],
    "physical_jobs": jobs, "physical_job_count": len(jobs),
    "physical_job_limit": 30,
    "all_job_names_campaign_prefixed": True,
    "restart_policy": "phase_artifact_revalidation_then_resubmit_missing_suffix",
    "production_evidence_eligible": False,
})
print(json.dumps({"artifact": artifact, "publication": write_immutable_json(args.output, artifact)}, indent=2, sort_keys=True))

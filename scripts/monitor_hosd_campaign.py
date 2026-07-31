#!/usr/bin/env python3
"""Write a deterministic HOSD restart/repair view from scheduler state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import build_campaign_monitor, load_and_validate_campaign  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import PRODUCTION_EXECUTION_PLAN_CONTRACT, load_hashed_json, write_immutable_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--states-json", required=True, type=Path)
    parser.add_argument("--artifact-validity-json", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    plan = load_hashed_json(args.campaign_root / "job_ledgers" / "production_execution_plan.json", expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT)
    artifact = build_campaign_monitor(execution_plan=plan, node_states=json.loads(args.states_json.read_text(encoding="utf-8")), artifact_validity=json.loads(args.artifact_validity_json.read_text(encoding="utf-8")), source=campaign["source"])
    output = args.output or args.campaign_root / "job_ledgers" / "monitor.json"
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"], "restart_nodes": artifact["restart_nodes"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

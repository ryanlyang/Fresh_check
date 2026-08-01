#!/usr/bin/env python3
"""Execute one registered command coordinate from a pinned HOSD DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
    node_execution,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    PRODUCTION_EXECUTION_PLAN_CONTRACT,
    load_hashed_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--campaign-source-root", type=Path)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--coordinate", type=int, default=0)
    args = parser.parse_args(argv)
    campaign_source_root = (
        REPO_ROOT
        if args.campaign_source_root is None
        else args.campaign_source_root.resolve()
    )
    campaign = load_and_validate_campaign(
        args.campaign_root, repo_root=campaign_source_root
    )
    plan = load_hashed_json(args.campaign_root / "job_ledgers" / "production_execution_plan.json", expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT)
    if plan.get("source") != campaign["source"]:
        raise ValueError("production execution plan source differs")
    node = node_execution(plan, node_id=args.node_id)
    if not 0 <= args.coordinate < len(node["commands"]):
        raise ValueError("node command coordinate lies outside its array")
    command = list(node["commands"][args.coordinate])
    if args.node_id == "resolved_parent_lock":
        command = [
            sys.executable,
            "-s",
            str(REPO_ROOT / "scripts" / "lock_hosd_inherited_parents.py"),
            "--campaign-root",
            str(args.campaign_root.resolve()),
            "--campaign-source-root",
            str(campaign_source_root),
        ]
    completed = subprocess.run(command, cwd=campaign_source_root, check=False)
    if completed.returncode:
        raise RuntimeError(f"registered node command failed with {completed.returncode}")
    print(json.dumps({"node_id": args.node_id, "coordinate": args.coordinate, "command_completed": True, "scientific_performance_inspected": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

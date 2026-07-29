#!/usr/bin/env python3
"""Build a non-mutating RETB resume plan from authenticated node states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    JOB_LEDGER_CONTRACT,
    PRODUCTION_GRAPH_CONTRACT,
    build_resume_plan,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--production-graph", required=True, type=Path)
    parser.add_argument("--previous-ledger", required=True, type=Path)
    parser.add_argument(
        "--completed-node",
        action="append",
        default=[],
        metavar="NODE=OUTPUT_SHA256",
    )
    parser.add_argument("--failed-node", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    graph = load_hashed_json(
        args.production_graph, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    ledger = load_hashed_json(
        args.previous_ledger, expected_contract=JOB_LEDGER_CONTRACT
    )
    completed = {}
    for raw in args.completed_node:
        name, separator, digest = raw.partition("=")
        if not separator or not name or name in completed:
            raise ValueError("--completed-node requires unique NODE=SHA256")
        completed[name] = digest
    artifact = build_resume_plan(
        production_graph=graph,
        previous_ledger=ledger,
        completed_nodes=completed,
        failed_nodes=args.failed_node,
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "resume_plan_sha256": artifact["content_hash"],
        "reusable_completed_nodes": artifact["reusable_completed_nodes"],
        "ready_to_resubmit": artifact["ready_to_resubmit"],
        "blocked_until_dependencies_complete": artifact[
            "blocked_until_dependencies_complete"
        ],
    }
    if args.output is not None and not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

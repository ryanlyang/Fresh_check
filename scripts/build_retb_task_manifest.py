#!/usr/bin/env python3
"""Publish one authenticated manifest for a bounded RETB Slurm array."""

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
    PRODUCTION_GRAPH_CONTRACT,
    build_task_manifest,
    load_hashed_json,
    validate_task_manifest_for_graph,
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
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--rows-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    graph = load_hashed_json(
        args.production_graph, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    if args.node_id not in nodes or nodes[args.node_id]["array"] is None:
        raise ValueError("task manifest node is absent or not an array")
    declaration = nodes[args.node_id]["array"]
    rows = json.loads(args.rows_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("task rows JSON must be an array")
    if len(rows) > int(declaration["maximum_tasks"]):
        raise ValueError("task manifest exceeds its declared maximum")
    artifact = build_task_manifest(
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        node_id=args.node_id,
        rows=rows,
        maximum_concurrent_tasks=int(
            declaration["maximum_concurrent_tasks"]
        ),
    )
    validate_task_manifest_for_graph(
        artifact,
        production_graph=graph,
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "task_manifest_sha256": artifact["content_hash"],
        "task_count": artifact["task_count"],
        "slurm_array": artifact["slurm_array"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

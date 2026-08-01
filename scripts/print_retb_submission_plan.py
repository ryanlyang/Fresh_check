#!/usr/bin/env python3
"""Print a shell-safe, pipe-delimited RETB production-node table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    PRODUCTION_GRAPH_CONTRACT,
    load_hashed_json,
    offline_submission_node_ids,
    validate_production_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-graph", required=True, type=Path)
    parser.add_argument(
        "--submission-scope",
        choices=("complete", "offline_abc"),
        default="complete",
    )
    args = parser.parse_args()
    graph = load_hashed_json(
        args.production_graph, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    validate_production_graph(graph)
    execution = {
        row["node_id"]: row
        for row in graph["node_execution_registry"]["entries"]
    }
    selected = (
        set(offline_submission_node_ids(graph))
        if args.submission_scope == "offline_abc"
        else {str(node["node_id"]) for node in graph["nodes"]}
    )
    for node in graph["nodes"]:
        if node["node_id"] not in selected:
            continue
        fields = (
            node["node_id"],
            node["stage"],
            ":".join(node["dependencies"]),
            node["resource"],
            node["worker"],
            "1" if node["array"] is not None else "0",
            "" if node["virtual_alias_of"] is None else node["virtual_alias_of"],
            execution[node["node_id"]]["dispatch_mode"],
        )
        if any("|" in str(value) or "\n" in str(value) for value in fields):
            raise ValueError("production node is not shell-table safe")
        print("|".join(fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

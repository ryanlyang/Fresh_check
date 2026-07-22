#!/usr/bin/env python3
"""Print immutable fields for one production node without mutating campaign state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    validate_prediction_anchored_tigris_graph,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument(
        "--field",
        choices=("configuration-run-ids", "teacher-namespace", "node-json"),
        default="node-json",
    )
    args = parser.parse_args(argv)
    graph = load_hashed_json(args.graph)
    validate_prediction_anchored_tigris_graph(graph)
    matches = [row for row in graph["nodes"] if row["node_id"] == args.node_id]
    if len(matches) != 1:
        raise KeyError(f"production graph has no unique node {args.node_id!r}")
    node = matches[0]
    if args.field == "configuration-run-ids":
        for run_id in node["configuration_run_ids"]:
            print(run_id)
    elif args.field == "teacher-namespace":
        if node["teacher_namespace"] is not None:
            print(node["teacher_namespace"])
    else:
        print(json.dumps(node, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

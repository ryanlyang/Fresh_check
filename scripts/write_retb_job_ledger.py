#!/usr/bin/env python3
"""Publish authenticated initial, resumed, or completed RETB Slurm ledgers."""

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
    build_job_ledger,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    write_immutable_json,
)


def _pairs(values: Sequence[str]) -> dict[str, str]:
    output = {}
    for raw in values:
        name, separator, job_id = raw.partition("=")
        if (
            not separator
            or not name
            or name in output
            or not job_id.isdigit()
            or int(job_id) <= 0
        ):
            raise ValueError(f"invalid job binding {raw!r}")
        output[name] = job_id
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-graph", required=True, type=Path)
    parser.add_argument("--job", action="append", default=[])
    parser.add_argument("--previous-ledger", type=Path)
    parser.add_argument(
        "--submission-mode",
        choices=(
            "dry_run",
            "smoke_simulation",
            "smoke_submitted",
            "production_submitted",
            "resumed",
            "completed",
        ),
        required=True,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    graph = load_hashed_json(
        args.production_graph, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    jobs = {}
    if args.previous_ledger is not None:
        previous = load_hashed_json(
            args.previous_ledger, expected_contract="retb_tigris_job_ledger_v1"
        )
        jobs.update(
            {
                name: job_id
                for name, job_id in previous["jobs"].items()
                if job_id is not None
            }
        )
    explicit = _pairs(args.job)
    overlap = set(jobs) & set(explicit)
    if any(jobs[name] != explicit[name] for name in overlap):
        raise ValueError("explicit job binding differs from previous ledger")
    jobs.update(explicit)
    ledger = build_job_ledger(
        production_graph=graph,
        jobs=jobs,
        submission_mode=args.submission_mode,
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "job_ledger_sha256": ledger["content_hash"],
        "submitted_node_count": ledger["submitted_node_count"],
        "all_nodes_bound": ledger["all_nodes_bound"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, ledger)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

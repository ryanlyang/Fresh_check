#!/usr/bin/env python3
"""Query or atomically update the source-bound RETB dynamic-job ledger."""

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
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


CONTRACT = "retb_tigris_dynamic_job_ledger_v1"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--production-graph", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--logical-name", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--dependency", default="")
    parser.add_argument("--task-manifest-sha256")
    parser.add_argument("--query", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    graph = load_hashed_json(
        args.production_graph, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    node_ids = {row["node_id"] for row in graph["nodes"]}
    if args.logical_name not in node_ids:
        raise ValueError("dynamic job is not in the production graph")
    if args.ledger.is_file():
        ledger = load_hashed_json(args.ledger, expected_contract=CONTRACT)
        if (
            ledger["campaign_spec_sha256"] != campaign["content_hash"]
            or ledger["production_graph_sha256"] != graph["content_hash"]
        ):
            raise ValueError("dynamic ledger belongs to another campaign")
        jobs = dict(ledger["jobs"])
    else:
        jobs = {}
    if args.query:
        row = jobs.get(args.logical_name)
        if row is not None:
            print(row["job_id"])
        return 0
    if args.job_id is None or not args.job_id.isdigit():
        raise ValueError("recording a dynamic job requires numeric --job-id")
    manifest_sha = args.task_manifest_sha256
    if manifest_sha is not None and (
        len(manifest_sha) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha)
    ):
        raise ValueError("dynamic task-manifest hash differs")
    row = {
        "job_id": args.job_id,
        "dependency": args.dependency,
        "task_manifest_sha256": manifest_sha,
    }
    existing = jobs.get(args.logical_name)
    if existing is not None and existing != row:
        raise ValueError("dynamic job already has another binding")
    jobs[args.logical_name] = row
    artifact = with_content_hash(
        {
            "contract": CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign["content_hash"],
            "production_graph_sha256": graph["content_hash"],
            "jobs": dict(sorted(jobs.items())),
            "job_count": len(jobs),
            "performance_based_job_suppression": False,
        }
    )
    _write(args.ledger, artifact)
    print(args.job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

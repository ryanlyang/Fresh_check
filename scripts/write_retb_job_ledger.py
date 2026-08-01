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
    JOB_LEDGER_CONTRACT,
    PRODUCTION_GRAPH_CONTRACT,
    build_job_ledger,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    FINAL_TEST_EVALUATION_CONTRACT,
    FINAL_TEST_EXECUTION_LOCK_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_reporting import (  # noqa: E402
    STAGE_MN_REPORT_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_selection import (  # noqa: E402
    LOCKED_SCALE_FINALISTS_CONTRACT,
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


def _hash_pairs(values: Sequence[str]) -> dict[str, str]:
    output = {}
    for raw in values:
        name, separator, digest = raw.partition("=")
        if (
            not separator
            or not name
            or name in output
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid completion-artifact binding {raw!r}")
        output[name] = digest
    return output


def _discover_completion_artifacts(campaign_root: Path) -> dict[str, str]:
    expected = {
        "locked_scale_finalists": LOCKED_SCALE_FINALISTS_CONTRACT,
        "final_test_execution_lock": FINAL_TEST_EXECUTION_LOCK_CONTRACT,
        "sealed_final_test_evaluation": FINAL_TEST_EVALUATION_CONTRACT,
        "final_report": STAGE_MN_REPORT_CONTRACT,
    }
    found: dict[str, list[dict]] = {name: [] for name in expected}
    by_contract = {contract: name for name, contract in expected.items()}
    for path in campaign_root.rglob("*.json"):
        try:
            payload = load_hashed_json(path)
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
            continue
        name = by_contract.get(payload.get("contract"))
        if name is not None:
            found[name].append(payload)
    if any(len(rows) != 1 for rows in found.values()):
        counts = {name: len(rows) for name, rows in found.items()}
        raise ValueError(
            f"completed ledger final-artifact coverage differs: {counts}"
        )
    return {
        name: rows[0]["content_hash"] for name, rows in found.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-graph", required=True, type=Path)
    parser.add_argument("--job", action="append", default=[])
    parser.add_argument("--previous-ledger", type=Path)
    parser.add_argument(
        "--completion-artifact", action="append", default=[]
    )
    parser.add_argument(
        "--submission-mode",
        choices=(
            "dry_run",
            "smoke_simulation",
            "smoke_submitted",
            "offline_production_submitted",
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
            args.previous_ledger, expected_contract=JOB_LEDGER_CONTRACT
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
    completion = _hash_pairs(args.completion_artifact)
    if args.submission_mode == "completed" and not completion:
        completion = _discover_completion_artifacts(
            Path(graph["campaign_root"])
        )
    ledger = build_job_ledger(
        production_graph=graph,
        jobs=jobs,
        submission_mode=args.submission_mode,
        completion_artifact_hashes=completion,
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

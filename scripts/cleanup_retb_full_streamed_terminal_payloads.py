#!/usr/bin/env python3
"""Safely prune registered recomputable payloads after full RETB completion."""

from __future__ import annotations

import hashlib
import json
import fnmatch
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (  # noqa: E402
    JOB_LEDGER_CONTRACT,
    PRODUCTION_GRAPH_CONTRACT,
    validate_job_ledger,
    validate_production_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import (  # noqa: E402
    FULL_STREAMED_PROFILE,
    STREAMED_TERMINAL_CLEANUP_PLAN_CONTRACT,
    STREAMED_TERMINAL_CLEANUP_RECEIPT_CONTRACT,
    TERMINAL_ROLLING_PAYLOAD_PATTERNS,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _eligible(relative_path: str) -> bool:
    relative = PurePosixPath(relative_path).as_posix()
    return any(
        fnmatch.fnmatchcase(relative, pattern)
        for pattern in TERMINAL_ROLLING_PAYLOAD_PATTERNS
    )


def _validate_plan(
    payload: Mapping[str, Any], *, campaign: Mapping[str, Any],
    graph: Mapping[str, Any], campaign_root: Path,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STREAMED_TERMINAL_CLEANUP_PLAN_CONTRACT
    )
    records = payload.get("payloads")
    if (
        payload.get("campaign_spec_sha256") != campaign["content_hash"]
        or payload.get("production_graph_sha256") != graph["content_hash"]
        or payload.get("source") != campaign.get("source")
        or not isinstance(records, list)
        or int(payload.get("payload_count", -1)) != len(records)
        or int(payload.get("payload_bytes", -1))
        != sum(int(row["bytes"]) for row in records)
    ):
        raise ValueError("full-streamed cleanup plan lineage differs")
    paths = [str(row.get("relative_path", "")) for row in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("full-streamed cleanup plan path order differs")
    root = campaign_root.resolve()
    for row, relative in zip(records, paths):
        if not relative or not _eligible(relative):
            raise ValueError("full-streamed cleanup plan gained an unsafe path")
        candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
        if root not in candidate.parents or candidate == root:
            raise ValueError("full-streamed cleanup path escaped campaign root")
        if (
            len(str(row.get("sha256", ""))) != 64
            or int(row.get("bytes", -1)) < 0
        ):
            raise ValueError("full-streamed cleanup payload evidence differs")
    return digest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    graph = load_hashed_json(
        root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    validate_production_graph(graph)
    if graph.get("execution_profile") != FULL_STREAMED_PROFILE:
        raise ValueError("terminal streamed cleanup received another profile")
    completed = load_hashed_json(
        root / "job_ledgers" / "completed_job_ledger.json",
        expected_contract=JOB_LEDGER_CONTRACT,
    )
    validate_job_ledger(completed, production_graph=graph)
    if completed.get("submission_mode") != "completed":
        raise ValueError("terminal cleanup requires the completed ledger")

    plan_path = root / "job_ledgers" / "streamed_terminal_cleanup_plan.json"
    receipt_path = root / "job_ledgers" / "streamed_terminal_cleanup_receipt.json"
    if plan_path.is_file():
        plan = load_hashed_json(
            plan_path, expected_contract=STREAMED_TERMINAL_CLEANUP_PLAN_CONTRACT
        )
    else:
        artifacts: dict[str, Path] = {}
        for pattern in TERMINAL_ROLLING_PAYLOAD_PATTERNS:
            for path in root.glob(pattern):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(root).as_posix()
                if not _eligible(relative):
                    raise ValueError("terminal cleanup discovery escaped policy")
                artifacts[relative] = path
        records = [
            {
                "relative_path": relative,
                "sha256": _sha256(artifacts[relative]),
                "bytes": artifacts[relative].stat().st_size,
            }
            for relative in sorted(artifacts)
        ]
        plan = with_content_hash(
            {
                "contract": STREAMED_TERMINAL_CLEANUP_PLAN_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign["content_hash"],
                "production_graph_sha256": graph["content_hash"],
                "completed_job_ledger_sha256": completed["content_hash"],
                "eligible_patterns": list(TERMINAL_ROLLING_PAYLOAD_PATTERNS),
                "payloads": records,
                "payload_count": len(records),
                "payload_bytes": sum(int(row["bytes"]) for row in records),
                "durable_scientific_outputs_eligible_for_cleanup": False,
                "source": campaign["source"],
            }
        )
        write_immutable_json(plan_path, plan)
    _validate_plan(plan, campaign=campaign, graph=graph, campaign_root=root)
    if (
        plan.get("completed_job_ledger_sha256") != completed["content_hash"]
        or plan.get("eligible_patterns") != list(TERMINAL_ROLLING_PAYLOAD_PATTERNS)
        or plan.get("durable_scientific_outputs_eligible_for_cleanup") is not False
    ):
        raise ValueError("full-streamed cleanup plan policy differs")

    for row in plan["payloads"]:
        path = root / Path(*PurePosixPath(row["relative_path"]).parts)
        if path.is_symlink():
            raise ValueError("refusing to clean a symlinked streamed payload")
        if path.is_file():
            if _sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise ValueError("streamed payload changed before cleanup")
            path.unlink()
    if any(
        (root / Path(*PurePosixPath(row["relative_path"]).parts)).exists()
        for row in plan["payloads"]
    ):
        raise RuntimeError("one or more streamed payloads survived cleanup")

    receipt = with_content_hash(
        {
            "contract": STREAMED_TERMINAL_CLEANUP_RECEIPT_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign["content_hash"],
            "production_graph_sha256": graph["content_hash"],
            "completed_job_ledger_sha256": completed["content_hash"],
            "cleanup_plan_sha256": plan["content_hash"],
            "removed_payload_count": plan["payload_count"],
            "removed_payload_bytes": plan["payload_bytes"],
            "all_registered_payloads_absent": True,
            "durable_scientific_outputs_removed": False,
            "source": campaign["source"],
        }
    )
    publication = write_immutable_json(receipt_path, receipt)
    print(json.dumps({"receipt": receipt, "publication": publication}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

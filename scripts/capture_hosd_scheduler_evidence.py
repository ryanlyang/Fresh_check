#!/usr/bin/env python3
"""Derive miniature scheduler and interruption evidence from Slurm ledgers."""

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
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    CAMPAIGN_MONITOR_CONTRACT,
    PRODUCTION_EXECUTION_PLAN_CONTRACT,
    SLURM_SUBMISSION_LEDGER_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def _ledgers(root: Path, plan, source):
    values = []
    for path in sorted(
        (root / "job_ledgers").glob("slurm_submission_attempt_*.json")
    ):
        value = load_hashed_json(
            path, expected_contract=SLURM_SUBMISSION_LEDGER_CONTRACT
        )
        if value["submission_mode"] == "dry_run":
            continue
        if (
            value.get("source") != source
            or value.get("execution_plan_sha256") != plan["content_hash"]
        ):
            raise ValueError("scheduler evidence submission lineage differs")
        values.append(value)
    return sorted(values, key=lambda row: int(row["attempt"]))


def _sacct(job_ids):
    completed = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(sorted(set(job_ids))),
            "-o",
            "JobIDRaw,State,ExitCode",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        row[0]: {"state": row[1].rstrip("+"), "exit_code": row[2]}
        for raw in completed.stdout.splitlines()
        if raw.strip()
        for row in [raw.split("|")]
        if len(row) >= 3
    }


def _interruption(
    *,
    node_id: str,
    monitor,
    ledgers,
    scheduler,
    root: Path,
    plan_by_id,
):
    attempts = [
        (row["attempt"], str(row["jobs"][node_id]))
        for row in ledgers
        if node_id in row["jobs"]
    ]
    if len(attempts) < 2:
        raise ValueError(f"{node_id} lacks a recovery submission")
    interrupted_job = attempts[-2][1]
    resumed_job = attempts[-1][1]
    interrupted_tasks = [
        (job_id, row)
        for job_id, row in scheduler.items()
        if job_id.startswith(interrupted_job + "_")
        and not row["state"].startswith("COMPLETED")
    ]
    if not interrupted_tasks:
        raise ValueError(f"{node_id} lacks an interrupted array coordinate")
    task_id, interrupted = sorted(interrupted_tasks)[0]
    coordinate = task_id.split("_", 1)[1]
    resumed_task = scheduler.get(f"{resumed_job}_{coordinate}")
    if resumed_task is None or not resumed_task["state"].startswith(
        "COMPLETED"
    ):
        raise ValueError(f"{node_id} recovery coordinate did not complete")
    monitor_row = next(
        row for row in monitor["nodes"] if row["node_id"] == node_id
    )
    if (
        monitor_row["artifact_valid"]
        or not monitor_row["needs_repair"]
        or node_id not in monitor["recovery_submission_nodes"]
    ):
        raise ValueError(f"{node_id} monitor did not reject partial state")
    for relative in plan_by_id[node_id]["outputs"]:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"{node_id} recovered output is absent: {relative}"
            )
        if path.suffix == ".json":
            load_hashed_json(path)
    return {
        "node_id": node_id,
        "coordinate": int(coordinate),
        "interrupted_job_id": interrupted_job,
        "resumed_job_id": resumed_job,
        "interrupted_state": interrupted["state"],
        "resumed_state": resumed_task["state"],
        "same_source_and_coordinate": True,
        "partial_artifact_rejected": True,
        "resume_reused_only_valid_state": True,
        "monitor_sha256": monitor["content_hash"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--target-monitor", required=True, type=Path)
    parser.add_argument("--training-monitor", required=True, type=Path)
    parser.add_argument(
        "--target-node", default="canonical_target_build"
    )
    parser.add_argument("--training-node", default="baseline_train")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "job_ledgers" / "production_execution_plan.json",
        expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT,
    )
    ledgers = _ledgers(root, plan, campaign["source"])
    all_jobs = [
        str(job_id)
        for ledger in ledgers
        for job_id in ledger["jobs"].values()
    ]
    scheduler = _sacct(all_jobs)
    latest_jobs = {}
    for ledger in ledgers:
        latest_jobs.update(
            {
                str(node_id): str(job_id)
                for node_id, job_id in ledger["jobs"].items()
            }
        )
    expected_nodes = {row["node_id"] for row in plan["nodes"]}
    if set(latest_jobs) != expected_nodes:
        raise ValueError("terminal scheduler evidence lacks node coverage")
    terminal = {}
    for node_id, job_id in latest_jobs.items():
        row = scheduler.get(job_id)
        if row is None or not row["state"].startswith("COMPLETED"):
            raise ValueError(
                f"terminal scheduler node is not complete: {node_id}"
            )
        terminal[node_id] = "COMPLETED"
    monitors = [
        load_hashed_json(
            args.target_monitor, expected_contract=CAMPAIGN_MONITOR_CONTRACT
        ),
        load_hashed_json(
            args.training_monitor, expected_contract=CAMPAIGN_MONITOR_CONTRACT
        ),
    ]
    if any(
        row.get("source") != campaign["source"]
        or row.get("execution_plan_sha256") != plan["content_hash"]
        for row in monitors
    ):
        raise ValueError("interrupt monitor lineage differs")
    plan_by_id = {row["node_id"]: row for row in plan["nodes"]}
    artifact = with_content_hash(
        {
            "contract": "hosd_miniature_scheduler_evidence_v1",
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "research_compute": True,
            "scheduler": "slurm",
            "execution_plan_sha256": plan["content_hash"],
            "submission_ledger_hashes": [
                row["content_hash"] for row in ledgers
            ],
            "terminal_state_by_node": terminal,
            "interrupt_resume": {
                "target_shard": _interruption(
                    node_id=args.target_node,
                    monitor=monitors[0],
                    ledgers=ledgers,
                    scheduler=scheduler,
                    root=root,
                    plan_by_id=plan_by_id,
                ),
                "training_row": _interruption(
                    node_id=args.training_node,
                    monitor=monitors[1],
                    ledgers=ledgers,
                    scheduler=scheduler,
                    root=root,
                    plan_by_id=plan_by_id,
                ),
            },
            "manual_artifact_injection": False,
            "performance_based_cancellation": False,
        }
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

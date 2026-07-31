#!/usr/bin/env python3
"""Publish the terminal source-bound Slurm and output-completion ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    COMPLETED_JOB_LEDGER_CONTRACT,
    PRODUCTION_EXECUTION_PLAN_CONTRACT,
    SLURM_SUBMISSION_LEDGER_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def _submission_jobs(root: Path, plan, source) -> tuple[dict[str, str], list[str]]:
    jobs: dict[str, str] = {}
    ledgers = []
    paths = sorted(
        (root / "job_ledgers").glob("slurm_submission_attempt_*.json")
    )
    for path in paths:
        ledger = load_hashed_json(
            path, expected_contract=SLURM_SUBMISSION_LEDGER_CONTRACT
        )
        if (
            ledger.get("source") != source
            or ledger.get("execution_plan_sha256") != plan["content_hash"]
        ):
            raise ValueError("submission ledger lineage differs")
        if ledger["submission_mode"] == "dry_run":
            continue
        for node_id, job_id in ledger["jobs"].items():
            jobs[str(node_id)] = str(job_id)
        ledgers.append(ledger["content_hash"])
    return jobs, ledgers


def _scheduler_rows(job_ids: list[str]) -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "-o",
            "JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = []
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        fields = raw.split("|")
        if len(fields) < 8:
            raise ValueError("sacct completion row is malformed")
        rows.append(
            dict(
                zip(
                    (
                        "job_id_raw",
                        "state",
                        "exit_code",
                        "elapsed",
                        "allocated_cpus",
                        "requested_memory",
                        "maximum_rss",
                        "total_cpu",
                    ),
                    fields[:8],
                )
            )
        )
    return rows


def _completed_rows_for_job(
    scheduler_rows: list[dict[str, str]], job_id: str
) -> list[dict[str, str]]:
    matching = [
        row
        for row in scheduler_rows
        if row["job_id_raw"] == job_id
        or row["job_id_raw"].startswith(job_id + "_")
    ]
    if not matching or any(
        not row["state"].rstrip("+").startswith("COMPLETED")
        or not row["exit_code"].startswith("0:")
        for row in matching
    ):
        raise RuntimeError("Slurm job is not completely successful")
    return matching


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "job_ledgers" / "production_execution_plan.json",
        expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT,
    )
    current_node = os.environ.get("HOSD_NODE_ID", "campaign_completion")
    if current_node != "campaign_completion":
        raise ValueError("completed ledger must run as campaign_completion")
    required_nodes = [
        row["node_id"]
        for row in plan["nodes"]
        if row["node_id"] != current_node
    ]
    jobs, submission_hashes = _submission_jobs(
        root, plan, campaign["source"]
    )
    if set(required_nodes) - set(jobs):
        raise ValueError("completed ledger lacks submitted upstream jobs")
    scheduler = _scheduler_rows([jobs[node] for node in required_nodes])
    node_states = {}
    for node_id in required_nodes:
        job_id = jobs[node_id]
        try:
            rows = _completed_rows_for_job(scheduler, job_id)
        except RuntimeError as error:
            raise RuntimeError(
                f"upstream Slurm job is not completely successful: {node_id}"
            ) from error
        node_states[node_id] = {
            "job_id": job_id,
            "scheduler_rows": rows,
            "completed": True,
        }
    output_hashes = {}
    for node in plan["nodes"]:
        if node["node_id"] == current_node:
            continue
        for relative in node["outputs"]:
            path = root / relative
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(
                    f"registered terminal output is absent: {relative}"
                )
            output_hashes[f"{node['node_id']}::{relative}"] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
            )
    artifact = with_content_hash(
        {
            "contract": COMPLETED_JOB_LEDGER_CONTRACT,
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "campaign_spec_sha256": campaign["content_hash"],
            "production_execution_plan_sha256": plan["content_hash"],
            "submission_ledger_hashes": submission_hashes,
            "node_states": node_states,
            "completed_upstream_node_count": len(node_states),
            "registered_output_hashes": dict(sorted(output_hashes.items())),
            "registered_output_count": len(output_hashes),
            "all_upstream_scheduler_states_completed": True,
            "all_registered_outputs_present": True,
            "scientific_result_sign_ignored": True,
        }
    )
    output = args.output or (
        root / "job_ledgers" / "completed_job_ledger.json"
    )
    publication = write_immutable_json(output, artifact)
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

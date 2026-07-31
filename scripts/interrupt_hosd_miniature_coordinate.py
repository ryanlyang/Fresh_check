#!/usr/bin/env python3
"""Intentionally interrupt one running miniature array coordinate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    SLURM_SUBMISSION_LEDGER_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--node", required=True)
    parser.add_argument("--coordinate", type=int)
    parser.add_argument("--wait-seconds", type=int, default=21600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    ledgers = []
    for path in sorted(
        (root / "job_ledgers").glob("slurm_submission_attempt_*.json")
    ):
        value = load_hashed_json(
            path, expected_contract=SLURM_SUBMISSION_LEDGER_CONTRACT
        )
        if args.node in value.get("jobs", {}):
            ledgers.append(value)
    if not ledgers:
        raise ValueError("interruption node has not been submitted")
    ledger = max(ledgers, key=lambda row: int(row["attempt"]))
    if ledger.get("source") != campaign["source"]:
        raise ValueError("interruption submission source differs")
    parent_job = str(ledger["jobs"][args.node])
    deadline = time.monotonic() + int(args.wait_seconds)
    chosen = None
    while time.monotonic() < deadline:
        queued = subprocess.run(
            [
                "squeue",
                "-h",
                "-j",
                parent_job,
                "-o",
                "%i|%T",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        rows = [
            raw.split("|", 1)
            for raw in queued.stdout.splitlines()
            if raw.strip() and "|" in raw
        ]
        running = [
            job_id
            for job_id, state in rows
            if state.upper() == "RUNNING" and "_" in job_id
        ]
        if args.coordinate is not None:
            requested = f"{parent_job}_{int(args.coordinate)}"
            if requested in running:
                chosen = requested
        elif running:
            chosen = sorted(
                running,
                key=lambda value: int(value.split("_", 1)[1]),
            )[0]
        if chosen is not None:
            break
        time.sleep(2)
    if chosen is None:
        raise TimeoutError("no requested array coordinate became RUNNING")
    coordinate = int(chosen.split("_", 1)[1])
    subprocess.run(["scancel", chosen], check=True)
    while time.monotonic() < deadline:
        accounting = subprocess.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                parent_job,
                "-o",
                "JobIDRaw,State",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        parent_states = [
            row[1].rstrip("+").upper()
            for raw in accounting.stdout.splitlines()
            if raw.strip()
            for row in [raw.split("|")]
            if len(row) >= 2 and row[0] == parent_job
        ]
        if parent_states and all(
            not state.startswith(
                ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING")
            )
            for state in parent_states
        ):
            break
        time.sleep(2)
    else:
        raise TimeoutError("interrupted array did not become terminal")
    artifact = with_content_hash(
        {
            "contract": "hosd_intentional_interruption_v1",
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "submission_ledger_sha256": ledger["content_hash"],
            "node_id": args.node,
            "parent_job_id": parent_job,
            "coordinate": coordinate,
            "array_task_id": chosen,
            "requested_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "required_real_miniature_interrupt_resume_acceptance",
            "scientific_performance_inspected": False,
        }
    )
    output = args.output or (
        root
        / "job_ledgers"
        / "interruptions"
        / f"{args.node}__job_{parent_job}__coordinate_{coordinate}.json"
    )
    publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
                "cancelled_array_task": chosen,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a recovery monitor directly from Slurm and registered artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_campaign_monitor,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    PRODUCTION_EXECUTION_PLAN_CONTRACT,
    SLURM_SUBMISSION_LEDGER_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def _jobs(root: Path, plan, source) -> dict[str, str]:
    jobs = {}
    for path in sorted(
        (root / "job_ledgers").glob("slurm_submission_attempt_*.json")
    ):
        ledger = load_hashed_json(
            path, expected_contract=SLURM_SUBMISSION_LEDGER_CONTRACT
        )
        if (
            ledger.get("source") != source
            or ledger.get("execution_plan_sha256") != plan["content_hash"]
        ):
            raise ValueError("scheduler snapshot submission lineage differs")
        if ledger["submission_mode"] == "dry_run":
            continue
        jobs.update(
            {
                str(node_id): str(job_id)
                for node_id, job_id in ledger["jobs"].items()
            }
        )
    expected = {row["node_id"] for row in plan["nodes"]}
    if set(jobs) != expected:
        raise ValueError("scheduler snapshot lacks submitted node coverage")
    return jobs


def _slurm_states(job_ids) -> dict[str, str]:
    completed = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "-o",
            "JobIDRaw,State",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    states = {}
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        job_id, state, *_ = raw.split("|")
        if job_id in job_ids:
            states[job_id] = state.rstrip("+").upper()
    return states


def _normalized(state: str) -> str:
    if state.startswith("COMPLETED"):
        return "completed"
    if state.startswith("RUNNING") or state.startswith("COMPLETING"):
        return "running"
    if state.startswith("PENDING") or state.startswith("CONFIGURING"):
        return "pending"
    if state.startswith("TIMEOUT"):
        return "timeout"
    if state.startswith("CANCELLED") or state.startswith("PREEMPTED"):
        return "cancelled"
    return "failed"


def _artifact_valid(root: Path, node, source) -> bool:
    try:
        for relative in node["outputs"]:
            path = root / relative
            if not path.is_file() or path.is_symlink():
                return False
            if path.suffix == ".json":
                value = load_hashed_json(path)
                if value.get("source") not in (None, source):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cancel-recovery-closure",
        action="store_true",
        help="Cancel active jobs in the computed recovery closure, then resnapshot.",
    )
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "job_ledgers" / "production_execution_plan.json",
        expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT,
    )
    jobs = _jobs(root, plan, campaign["source"])

    def snapshot():
        observed = _slurm_states(list(jobs.values()))
        states = {
            node_id: {
                "state": _normalized(observed.get(job_id, "PENDING")),
                "job_id": job_id,
            }
            for node_id, job_id in jobs.items()
        }
        validity = {
            node["node_id"]: _artifact_valid(
                root, node, campaign["source"]
            )
            for node in plan["nodes"]
        }
        return build_campaign_monitor(
            execution_plan=plan,
            node_states=states,
            artifact_validity=validity,
            source=campaign["source"],
        )

    monitor = snapshot()
    if args.cancel_recovery_closure:
        by_id = {row["node_id"]: row for row in monitor["nodes"]}
        active = [
            jobs[node_id]
            for node_id in monitor["recovery_submission_nodes"]
            if by_id[node_id]["state"] in {"pending", "running"}
        ]
        if active:
            subprocess.run(["scancel", *active], check=True)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                observed = _slurm_states(active)
                if all(
                    _normalized(observed.get(job_id, "CANCELLED"))
                    not in {"pending", "running"}
                    for job_id in active
                ):
                    break
                time.sleep(2)
            else:
                raise TimeoutError(
                    "recovery-closure cancellation did not become terminal"
                )
            monitor = snapshot()
    publication = write_immutable_json(args.output, monitor)
    print(
        json.dumps(
            {
                "content_hash": monitor["content_hash"],
                "publication": publication["status"],
                "complete": monitor["complete"],
                "recovery_submission_nodes": monitor[
                    "recovery_submission_nodes"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

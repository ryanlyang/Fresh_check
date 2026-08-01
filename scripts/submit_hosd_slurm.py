#!/usr/bin/env python3
"""Dry-run, submit, or recover the exhaustive integrity-gated HOSD DAG."""

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
    build_slurm_submission_ledger,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    CAMPAIGN_MONITOR_CONTRACT,
    FULL_AUTHORIZATION_CONTRACT,
    PRODUCTION_EXECUTION_PLAN_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke-submit", action="store_true")
    mode.add_argument("--full-submit", action="store_true")
    mode.add_argument("--resume-submit", action="store_true")
    parser.add_argument("--monitor", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--account", default="reu-aisocial")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--campaign-source-root",
        type=Path,
        help=(
            "Validated frozen source worktree used by scientific workers. "
            "This permits a newer control-plane launcher to recover a campaign "
            "without changing its source-bound scientific code."
        ),
    )
    return parser


def _topological_nodes(nodes):
    """Return a stable dependency order independent of registry presentation."""
    by_id = {str(row["node_id"]): row for row in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("execution plan contains duplicate node IDs")
    unknown = {
        dependency
        for row in nodes
        for dependency in row["dependencies"]
        if dependency not in by_id
    }
    if unknown:
        raise ValueError(f"execution plan has unknown dependencies: {sorted(unknown)}")
    pending = list(nodes)
    emitted = set()
    ordered = []
    while pending:
        ready = [
            row
            for row in pending
            if set(row["dependencies"]).issubset(emitted)
        ]
        if not ready:
            raise ValueError("execution plan dependency graph is cyclic")
        for row in ready:
            ordered.append(row)
            emitted.add(str(row["node_id"]))
            pending.remove(row)
    return ordered


def _submission_profile_and_mode(args, plan):
    if args.smoke_submit:
        return "miniature_test", "smoke_submit", False
    if args.dry_run:
        return plan["profile"], "dry_run", False
    if args.resume_submit:
        profile = str(plan["profile"])
        return (
            profile,
            "resume_submit",
            profile == "production_500k_scale3m",
        )
    return "production_500k_scale3m", "full_submit", True


def main(argv=None):
    args = _parser().parse_args(argv)
    campaign_source_root = (
        REPO_ROOT
        if args.campaign_source_root is None
        else args.campaign_source_root.resolve()
    )
    campaign = load_and_validate_campaign(
        args.campaign_root, repo_root=campaign_source_root
    )
    plan = load_hashed_json(
        args.campaign_root / "job_ledgers" / "production_execution_plan.json",
        expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT,
    )
    if plan["source"] != campaign["source"]:
        raise ValueError("submission source differs")
    expected_profile, mode, requires_authorization = (
        _submission_profile_and_mode(args, plan)
    )
    if plan["profile"] != expected_profile:
        raise ValueError("submission profile differs")

    if requires_authorization:
        authorization_path = args.authorization or (
            args.campaign_root / "job_ledgers" / "full_authorization.json"
        )
        authorization = load_hashed_json(
            authorization_path, expected_contract=FULL_AUTHORIZATION_CONTRACT
        )
        if (
            authorization.get("production_execution_plan_sha256")
            != plan["content_hash"]
            or authorization.get("source") != campaign["source"]
            or not authorization.get("full_campaign_submission_authorized")
        ):
            raise ValueError("full campaign is not authorized")

    ordered_nodes = _topological_nodes(list(plan["nodes"]))
    all_ids = [node["node_id"] for node in ordered_nodes]
    if args.resume_submit:
        if args.monitor is None:
            raise ValueError("resume submission requires --monitor")
        monitor = load_hashed_json(
            args.monitor, expected_contract=CAMPAIGN_MONITOR_CONTRACT
        )
        if (
            monitor.get("execution_plan_sha256") != plan["content_hash"]
            or monitor.get("source") != campaign["source"]
        ):
            raise ValueError("recovery monitor lineage differs")
        selected = list(monitor["recovery_submission_nodes"])
        state = {row["node_id"]: row["state"] for row in monitor["nodes"]}
        active = [
            node_id
            for node_id in selected
            if state[node_id] in {"running", "pending"}
        ]
        if active:
            raise RuntimeError(
                "recovery nodes still have active scheduler state; cancel or "
                f"finish them before resubmission: {active}"
            )
    else:
        selected = all_ids

    maximum = args.max_parallel
    if maximum is None:
        maximum = 8
    if int(maximum) <= 0:
        raise ValueError("max parallel must be positive")
    jobs: dict[str, str | None] = {}
    selected_set = set(selected)
    for node in ordered_nodes:
        node_id = node["node_id"]
        if node_id not in selected_set:
            continue
        dependencies = []
        for value in node["dependencies"]:
            if value not in selected_set:
                continue
            if value not in jobs:
                raise RuntimeError(
                    f"dependency {value} was not submitted before {node_id}"
                )
            dependencies.append(jobs[value])
        resource = node["resource"]
        command = [
            "sbatch",
            "--parsable",
            f"--account={args.account}",
            f"--partition={resource['partition']}",
            f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}",
            f"--time={resource['time']}",
            f"--job-name=hosd_{node_id}",
            f"--output={args.campaign_root}/job_ledgers/slurm/%x_%A_%a.out",
            f"--error={args.campaign_root}/job_ledgers/slurm/%x_%A_%a.err",
            (
                "--export=ALL,"
                f"PROJECT_DIR={campaign_source_root},HOSD_LAUNCHER_ROOT={REPO_ROOT},"
                f"CAMPAIGN_ROOT={args.campaign_root},"
                f"HOSD_NODE_ID={node_id}"
            ),
        ]
        if resource["gres"]:
            command.append(f"--gres={resource['gres']}")
        if dependencies:
            command.append(
                "--dependency=afterok:" + ":".join(dependencies)
            )
        coordinates = len(node["commands"])
        if coordinates > 1:
            command.append(
                f"--array=0-{coordinates - 1}%{int(maximum)}"
            )
        command.append(
            str(REPO_ROOT / "sbatch" / "run_hosd_registered_node.sh")
        )
        if args.dry_run:
            jobs[node_id] = None
            continue
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError("sbatch did not return a numeric job ID")
        jobs[node_id] = job_id
    ledger = build_slurm_submission_ledger(
        execution_plan=plan,
        jobs=jobs,
        submission_mode=mode,
        attempt=args.attempt,
        selected_node_ids=selected,
        source=campaign["source"],
    )
    ledger_path = (
        args.campaign_root
        / "job_ledgers"
        / f"slurm_submission_attempt_{args.attempt:03d}.json"
    )
    publication = write_immutable_json(ledger_path, ledger)
    print(
        json.dumps(
            {
                **ledger,
                "publication": publication["status"],
                "ledger_path": str(ledger_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

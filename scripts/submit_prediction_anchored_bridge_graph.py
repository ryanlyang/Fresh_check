#!/usr/bin/env python3
"""Submit an immutable prediction-anchored Tigris graph only with --execute."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    TIGRIS_ACCOUNT,
    TIGRIS_PARTITION,
    build_prediction_anchored_job_ledger,
    render_tigris_sbatch_commands,
    validate_prediction_anchored_tigris_graph,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--ledger-output", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-final-test", action="store_true")
    parser.add_argument("--approve-final-test", action="store_true")
    parser.add_argument("--sbatch-bin", default="sbatch")
    parser.add_argument("--log-dir", default="fresh_check_logs")
    return parser


def _argv(node, job_ids, *, graph_path: Path, artifact_root: str, log_dir: Path, sbatch_bin: str):
    resources = node["resources"]
    argv = [
        sbatch_bin,
        "--parsable",
        f"--account={TIGRIS_ACCOUNT}",
        f"--partition={TIGRIS_PARTITION}",
        "--nodes=1",
        f"--cpus-per-task={resources['cpus_per_task']}",
        f"--mem={resources['host_memory_gib']}G",
        f"--time={resources['walltime']}",
        "--kill-on-invalid-dep=yes",
        f"--job-name=pab_{node['node_id'][:36]}",
        f"--output={log_dir.as_posix()}/%x_%j.out",
        f"--error={log_dir.as_posix()}/%x_%j.err",
        (
            "--export=ALL,PYTHONNOUSERSITE=1,"
            f"PREDICTION_ANCHORED_GRAPH={graph_path.as_posix()},"
            f"PREDICTION_ANCHORED_NODE_ID={node['node_id']},"
            f"PREDICTION_ANCHORED_ARTIFACT_ROOT={artifact_root}"
        ),
    ]
    if int(resources["gpus_per_node"]) > 0:
        argv.append(f"--gres=gpu:{resources['gpus_per_node']}")
    if node["dependencies"]:
        argv.append(
            "--dependency=afterok:"
            + ":".join(str(job_ids[value]) for value in node["dependencies"])
        )
    argv.extend([f"sbatch/{node['runner']}", *node["arguments"]])
    return argv


def _validate_required_executors(graph, *, include_final_test: bool) -> dict[str, str]:
    required = {
        "PAB_CONSUMER_EXECUTOR": any(
            row["runner"] == "run_train_prediction_anchored_bridge_consumer.sh"
            for row in graph["nodes"]
        ),
        "PAB_RECONSTRUCTOR_EXECUTOR": any(
            row["runner"] == "run_train_prediction_anchored_bridge_reconstructor.sh"
            for row in graph["nodes"]
        ),
        "PAB_TEACHER_FORWARD_EXECUTOR": any(
            row["runner"] == "run_cache_prediction_anchored_bridge_logits.sh"
            for row in graph["nodes"]
        ),
        "PAB_DEPLOYABLE_EXPORT_EXECUTOR": True,
        "PAB_FINAL_TEST_EXECUTOR": bool(include_final_test),
    }
    resolved = {}
    for name, needed in required.items():
        if not needed:
            continue
        raw = os.environ.get(name, "")
        path = Path(raw)
        if not raw or path.is_symlink() or not path.is_file():
            raise PermissionError(
                f"actual production submission requires a safe {name} executable path"
            )
        resolved[name] = str(path.resolve())
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise ValueError("--execute and --dry-run are mutually exclusive")
    if args.include_final_test and not args.approve_final_test:
        raise PermissionError("final-test submission requires --approve-final-test")
    if args.approve_final_test and not args.include_final_test:
        raise ValueError("--approve-final-test has no effect without --include-final-test")
    graph_path = Path(args.graph).resolve()
    graph = load_hashed_json(graph_path)
    validation = validate_prediction_anchored_tigris_graph(graph)
    if not args.execute:
        rendered = render_tigris_sbatch_commands(
            graph, include_final_test=bool(args.include_final_test)
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "submission_executed": False,
                    "validation": validation,
                    "rendered": rendered,
                    "execution_hint": (
                        "rerun with --execute only after reviewing this immutable graph"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.ledger_output:
        raise ValueError("actual submission requires --ledger-output")
    executor_preflight = _validate_required_executors(
        graph, include_final_test=bool(args.include_final_test)
    )
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    by_id = {row["node_id"]: row for row in graph["nodes"]}
    job_ids = {}
    commands = []
    for node_id in graph["topological_node_ids"]:
        node = by_id[node_id]
        if node["protected_final_test"] and not args.include_final_test:
            continue
        command = _argv(
            node,
            job_ids,
            graph_path=graph_path,
            artifact_root=str(graph["artifact_root"]),
            log_dir=log_dir,
            sbatch_bin=str(args.sbatch_bin),
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Slurm submission failed for {node_id}: {completed.stderr.strip()}"
            )
        token = completed.stdout.strip().split(";", 1)[0]
        if not token.isdigit():
            raise RuntimeError(f"Slurm returned an invalid job ID for {node_id}: {token!r}")
        job_ids[node_id] = token
        commands.append({"node_id": node_id, "job_id": token, "argv": command})

    ledger = build_prediction_anchored_job_ledger(
        graph,
        job_ids=job_ids,
        include_final_test=bool(args.include_final_test),
    )
    publication = write_immutable_json(args.ledger_output, ledger)
    print(
        json.dumps(
            {
                "dry_run": False,
                "submission_executed": True,
                "executor_preflight": executor_preflight,
                "submitted_jobs": commands,
                "ledger": ledger,
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

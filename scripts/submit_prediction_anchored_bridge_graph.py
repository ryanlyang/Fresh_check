#!/usr/bin/env python3
"""Submit an immutable prediction-anchored Tigris graph only with --execute."""

from __future__ import annotations

import argparse
import json
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
from teacher_logit_reco.local_particle_residual_field.bridge_campaign import (  # noqa: E402
    validate_campaign_registry,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign_policy import (  # noqa: E402
    PREDICTION_ANCHORED_CAMPAIGN_RESERVATION_CONTRACT,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_execution import (  # noqa: E402
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--registry", default="")
    parser.add_argument("--reservations", default="")
    parser.add_argument("--execution-spec", default="")
    parser.add_argument("--ledger-output", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-final-test", action="store_true")
    parser.add_argument("--approve-final-test", action="store_true")
    parser.add_argument("--sbatch-bin", default="sbatch")
    parser.add_argument("--log-dir", default="fresh_check_logs")
    return parser


def _argv(
    node,
    job_ids,
    *,
    graph_path: Path,
    artifact_root: str,
    registry_path: Path,
    reservations_path: Path,
    execution_spec_path: Path,
    log_dir: Path,
    sbatch_bin: str,
):
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
            f"PREDICTION_ANCHORED_ARTIFACT_ROOT={artifact_root},"
            f"PAB_REGISTRY={registry_path.as_posix()},"
            f"PAB_RESERVATIONS={reservations_path.as_posix()},"
            f"PAB_EXECUTION_SPEC={execution_spec_path.as_posix()}"
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
    del graph, include_final_test
    return {
        "repository_owned_deployable_export": "scripts/deploy_prediction_anchored_bridge.py",
        "repository_owned_final_test": "scripts/deploy_prediction_anchored_bridge.py",
    }


def _submission_bindings(args, graph) -> tuple[Path, Path, Path]:
    missing = [
        option
        for option, value in (
            ("--registry", args.registry),
            ("--reservations", args.reservations),
            ("--execution-spec", args.execution_spec),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "actual submission requires explicit scientific bindings: "
            + ", ".join(missing)
        )
    paths = tuple(Path(value).resolve() for value in (
        args.registry, args.reservations, args.execution_spec
    ))
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing/unsafe submission binding: {path}")
        if "," in path.as_posix():
            raise ValueError("Slurm export paths may not contain commas")
    registry_path, reservations_path, execution_spec_path = paths
    registry = load_hashed_json(registry_path)
    validate_campaign_registry(registry)
    reservations = load_hashed_json(
        reservations_path,
        expected_contract=PREDICTION_ANCHORED_CAMPAIGN_RESERVATION_CONTRACT,
    )
    execution_spec = load_hashed_json(
        execution_spec_path,
        expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    )
    validate_prediction_anchored_execution_spec(
        execution_spec, verify_file_hashes=True
    )
    if registry["content_hash"] != graph.get("registry_sha256"):
        raise ValueError("submission registry differs from the immutable graph")
    if reservations["content_hash"] != graph.get("reservations_sha256"):
        raise ValueError("submission reservations differ from the immutable graph")
    if reservations.get("registry_sha256") != registry["content_hash"]:
        raise ValueError("submission reservations belong to another registry")
    expected_bindings = {
        "execution_spec_sha256": execution_spec["content_hash"],
        "child_manifest_sha256": execution_spec["child_manifest"]["content_hash"],
        "parent_manifest_file_sha256": execution_spec["parent_manifest"]["sha256"],
    }
    for name, expected in expected_bindings.items():
        if reservations.get(name) != expected or graph.get(name) != expected:
            raise ValueError(
                f"submission {name} differs across execution spec, reservations, and graph"
            )
    return registry_path, reservations_path, execution_spec_path


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
    registry_path, reservations_path, execution_spec_path = _submission_bindings(
        args, graph
    )
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
            registry_path=registry_path,
            reservations_path=reservations_path,
            execution_spec_path=execution_spec_path,
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

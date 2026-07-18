#!/usr/bin/env python3
"""Submit one dependency-safe ABPH pilot, high-data, or final-claim graph."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (  # noqa: E402
    ABPH_RECONSTRUCTOR_PARALLELISM_MODES,
    ABPH_STAGE_MODES,
    AdaptiveBinarySubmissionConfig,
    SlurmJobSpec,
    SlurmResourceProfile,
    build_submission_graph,
    canonical_hash,
    require_partial_stage_inputs,
    submission_manifest,
)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("ABPH_DATA_DIR", "/home/ryreu/atlas/PracticeTagging/data/jetclass_part1"),
    )
    parser.add_argument("--campaign-mode", choices=("pilot", "highdata"), default="pilot")
    parser.add_argument("--stage-mode", choices=ABPH_STAGE_MODES, default="full")
    parser.add_argument("--cluster", choices=("tigris", "tier3"), default="tigris")
    parser.add_argument("--account", default=os.environ.get("ABPH_SBATCH_ACCOUNT"))
    parser.add_argument("--project-dir", default=os.environ.get("PROJECT_DIR", str(REPO_ROOT)))
    parser.add_argument("--approve-highdata", action="store_true", default=_bool_env("ABPH_APPROVE_HIGHDATA"))
    parser.add_argument("--pilot-report", default=os.environ.get("ABPH_PILOT_REPORT_PATH"))
    parser.add_argument("--approve-final-test", action="store_true", default=_bool_env("ABPH_APPROVE_FINAL_TEST"))
    parser.add_argument("--selection-report", default=os.environ.get("ABPH_SELECTION_REPORT_PATH"))
    parser.add_argument("--final-claim-contract", default=os.environ.get("ABPH_FINAL_CLAIM_CONTRACT"))
    parser.add_argument("--confirm-final-test", action="store_true", default=_bool_env("CONFIRM_FINAL_TEST"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-manifest")
    parser.add_argument("--gpu-memory")
    parser.add_argument("--cpu-memory")
    parser.add_argument("--gpu-cpus", type=int)
    parser.add_argument("--cpu-cpus", type=int)
    parser.add_argument(
        "--reconstructor-parallelism",
        choices=ABPH_RECONSTRUCTOR_PARALLELISM_MODES,
        default=os.environ.get("ABPH_RECONSTRUCTOR_PARALLELISM", "single"),
        help="Distributed topology for B/C/D reconstructor jobs only.",
    )
    parser.add_argument(
        "--runtime-acceptance",
        default=os.environ.get("ABPH_RUNTIME_ACCEPTANCE_PATH"),
        help="Immutable Step-10 acceptance artifact required by full DDP4 campaigns.",
    )
    return parser


def _config(args: argparse.Namespace) -> AdaptiveBinarySubmissionConfig:
    partial = args.stage_mode in {"predictions", "fusion", "diagnostics", "report", "final_claims"}
    reuse_preparation = args.stage_mode == "models"
    return AdaptiveBinarySubmissionConfig(
        campaign_root=args.campaign_root,
        data_dir=args.data_dir,
        campaign_mode=args.campaign_mode,
        stage_mode=args.stage_mode,
        cluster=args.cluster,
        account=args.account,
        approve_highdata=bool(args.approve_highdata),
        pilot_report_path=args.pilot_report,
        approve_final_test=bool(args.approve_final_test),
        selection_report_path=args.selection_report,
        final_claim_contract_path=args.final_claim_contract,
        confirm_final_test=bool(args.confirm_final_test),
        rebuild_inputs=not partial and not reuse_preparation,
        rebuild_targets=not partial and not reuse_preparation,
        rebuild_models=not partial,
        rebuild_predictions=args.stage_mode in {
            "full",
            "models",
            "predictions",
            "final_claims",
        },
        reconstructor_parallelism=args.reconstructor_parallelism,
        runtime_acceptance_path=args.runtime_acceptance,
    )


def _resource(
    args: argparse.Namespace,
    config: AdaptiveBinarySubmissionConfig,
) -> SlurmResourceProfile:
    profile = SlurmResourceProfile.for_cluster(args.cluster, account=args.account)
    topology = config.reconstructor_topology
    return replace(
        profile,
        gpu_memory=args.gpu_memory or profile.gpu_memory,
        cpu_memory=args.cpu_memory or profile.cpu_memory,
        gpu_cpus=args.gpu_cpus or profile.gpu_cpus,
        cpu_cpus=args.cpu_cpus or profile.cpu_cpus,
        nodes=int(topology["nodes"]),
        ntasks=int(topology["ntasks"]),
        ntasks_per_node=int(topology["ntasks_per_node"]),
        gpus_per_node=int(topology["gpus_per_node"]),
        distributed_world_size=int(topology["distributed_world_size"]),
        launcher=str(topology["launcher"]),
    )


_EXECUTOR_DEFAULTS = {
    "ABPH_VARIANT_EXECUTOR": "scripts/train_adaptive_binary_pseudooffline_variant.py",
    "ABPH_PREDICTION_EXECUTOR": "scripts/predict_adaptive_binary_pseudooffline.py",
    "ABPH_DIAGNOSTIC_EXECUTOR": "scripts/diagnose_adaptive_binary_pseudooffline.py",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_executor_names(stage_mode: str) -> tuple[str, ...]:
    if stage_mode in {"full", "models"}:
        return tuple(_EXECUTOR_DEFAULTS)
    if stage_mode in {"predictions", "final_claims"}:
        return ("ABPH_PREDICTION_EXECUTOR",)
    if stage_mode == "diagnostics":
        return ("ABPH_DIAGNOSTIC_EXECUTOR",)
    return ()


def _resolve_runtime_executors(
    *,
    stage_mode: str,
    project_dir: Path,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    environment: dict[str, str] = {}
    provenance: dict[str, dict[str, str]] = {}
    for name in _required_executor_names(stage_mode):
        configured = os.environ.get(name, _EXECUTOR_DEFAULTS[name])
        path = Path(configured)
        if not path.is_absolute():
            path = project_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{stage_mode} submission requires {name}, but no executor exists at {path}. "
                "Set the environment variable to the reviewed production entry point before queueing."
            )
        environment[name] = str(path)
        provenance[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    return environment, provenance


def _sbatch_command(
    job: SlurmJobSpec,
    *,
    dependency_ids: Sequence[str],
    resource: SlurmResourceProfile,
    project_dir: Path,
) -> list[str]:
    script = project_dir / "sbatch" / job.script
    if not script.is_file():
        raise FileNotFoundError(f"ABPH Slurm worker is missing: {script}")
    command = [
        "sbatch",
        "--parsable",
        f"--job-name=abph_{job.stage}"[:128],
        f"--partition={resource.partition}",
        f"--nodes={job.nodes}",
        f"--ntasks={job.ntasks}",
        f"--ntasks-per-node={job.ntasks_per_node}",
        f"--cpus-per-task={resource.gpu_cpus if job.gpu else resource.cpu_cpus}",
        f"--mem={resource.gpu_memory if job.gpu else resource.cpu_memory}",
        f"--time={resource.gpu_time if job.gpu else resource.cpu_time}",
    ]
    if resource.account:
        command.append(f"--account={resource.account}")
    if job.gpu:
        command.append(f"--gres={resource.gpu_gres}")
    if dependency_ids:
        command.append("--dependency=afterok:" + ":".join(dependency_ids))
    command.extend((str(script), *job.arguments))
    return command


def _submit(
    jobs: Sequence[SlurmJobSpec],
    *,
    resource: SlurmResourceProfile,
    project_dir: Path,
    dry_run: bool,
    runtime_environment: dict[str, str],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    job_ids: dict[str, str] = {}
    commands: list[dict[str, object]] = []
    for index, job in enumerate(jobs, start=1):
        dependencies = [job_ids[name] for name in job.dependencies]
        command = _sbatch_command(
            job,
            dependency_ids=dependencies,
            resource=resource,
            project_dir=project_dir,
        )
        environment = os.environ.copy()
        environment.update(runtime_environment)
        environment.update(job.environment)
        environment.update(
            {
                "PROJECT_DIR": str(project_dir),
                "PYTHONNOUSERSITE": "1",
                "ABPH_SBATCH_ACCOUNT": resource.account,
                "ABPH_SBATCH_PARTITION": resource.partition,
                "ABPH_JOB_LAUNCHER": job.launcher,
                "ABPH_DISTRIBUTED_NODES": str(job.nodes),
                "ABPH_DISTRIBUTED_NTASKS": str(job.ntasks),
                "ABPH_DISTRIBUTED_NTASKS_PER_NODE": str(job.ntasks_per_node),
                "ABPH_DISTRIBUTED_GPUS_PER_NODE": str(job.resolved_gpus_per_node),
                "ABPH_DISTRIBUTED_WORLD_SIZE": str(job.distributed_world_size),
            }
        )
        if dry_run:
            job_id = f"DRYRUN_{index:04d}"
        else:
            result = subprocess.run(
                command,
                cwd=project_dir,
                env=environment,
                check=True,
                text=True,
                capture_output=True,
            )
            token = result.stdout.strip().split(";", 1)[0]
            if not token.isdigit():
                raise RuntimeError(f"sbatch returned invalid job id for {job.key}: {result.stdout!r}")
            job_id = token
            print(result.stdout.strip(), file=sys.stderr)
        job_ids[job.key] = job_id
        commands.append(
            {
                "key": job.key,
                "job_id": job_id,
                "dependencies": dependencies,
                "command": command,
                "environment": {
                    **runtime_environment,
                    **dict(job.environment),
                    "ABPH_JOB_LAUNCHER": job.launcher,
                    "ABPH_DISTRIBUTED_NODES": str(job.nodes),
                    "ABPH_DISTRIBUTED_NTASKS": str(job.ntasks),
                    "ABPH_DISTRIBUTED_NTASKS_PER_NODE": str(job.ntasks_per_node),
                    "ABPH_DISTRIBUTED_GPUS_PER_NODE": str(job.resolved_gpus_per_node),
                    "ABPH_DISTRIBUTED_WORLD_SIZE": str(job.distributed_world_size),
                },
            }
        )
    return job_ids, commands


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _config(args)
    reuse_preflight = require_partial_stage_inputs(config)
    jobs = build_submission_graph(config)
    resource = _resource(args, config)
    project_dir = Path(args.project_dir).resolve()
    runtime_environment, runtime_executors = _resolve_runtime_executors(
        stage_mode=args.stage_mode,
        project_dir=project_dir,
    )
    job_ids, commands = _submit(
        jobs,
        resource=resource,
        project_dir=project_dir,
        dry_run=bool(args.dry_run),
        runtime_environment=runtime_environment,
    )
    manifest = submission_manifest(config, jobs)
    manifest.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": bool(args.dry_run),
            "resource_profile": resource.__dict__,
            "job_ids": job_ids,
            "submission_commands": commands,
            "reuse_preflight": reuse_preflight,
            "runtime_executors": runtime_executors,
        }
    )
    destination = (
        Path(args.output_manifest)
        if args.output_manifest
        else Path(args.campaign_root) / "submission_logs" / f"abph_{args.stage_mode}_submission.json"
    )
    parallelism_path = destination.parent / "abph_reconstructor_parallelism.json"
    parallelism_manifest = {
        **dict(manifest["reconstructor_parallelism"]),
        "campaign_root": str(config.paths.root),
        "cluster": config.cluster,
        "account": resource.account,
        "partition": resource.partition,
        "gpu_gres_per_node": resource.gpu_gres,
        "gpu_cpus_per_rank": resource.gpu_cpus,
        "gpu_memory_per_node": resource.gpu_memory,
        "submission_graph_hash": manifest["graph_hash"],
    }
    parallelism_manifest["content_hash"] = parallelism_content_hash = canonical_hash(
        parallelism_manifest
    )
    manifest["parallelism_manifest"] = {
        "path": str(parallelism_path),
        "content_hash": parallelism_content_hash,
    }
    if not args.dry_run or args.output_manifest:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(destination, manifest)
        _write_json_atomic(parallelism_path, parallelism_manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

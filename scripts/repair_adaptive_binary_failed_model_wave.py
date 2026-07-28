#!/usr/bin/env python3
"""Replay failed ABPH model jobs and rewire their pending descendants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repair_adaptive_binary_runtime_batch_contract import (
    _live_dependency,
    _rows,
    replace_dependency_job,
)
from scripts.repair_adaptive_binary_storage_acceptance_graph import (
    _ACTIVE_SLURM_STATES,
    _job_environment,
    _remaining_dependencies,
    _require_pending,
    _slurm_job_state,
    _source_identity,
)


DEFAULT_VARIANTS = (
    "B0_pooled_mlp_root",
    "B2_semantic_query_probabilistic",
    "B4_oracle_root_diagnostic",
    "D6_true_offline_particles",
)
ORACLE_REFERENCE_VARIANTS = {
    "B4_oracle_root_diagnostic",
    "D6_true_offline_particles",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-manifest", required=True)
    parser.add_argument("--project-dir", default="/home/ryreu/atlas/Fresh_check")
    parser.add_argument("--log-dir")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def repaired_variant_command(
    row: Mapping[str, Any],
    *,
    variant: str,
    dependencies: Sequence[str],
    log_dir: Path,
) -> list[str]:
    raw = row.get("command")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{row.get('key')} lacks its original sbatch command")
    oracle_reference = variant in ORACLE_REFERENCE_VARIANTS
    removed_prefixes = ("--dependency=", "--error=", "--job-name=", "--output=")
    if oracle_reference:
        removed_prefixes = (
            *removed_prefixes,
            "--nodes=",
            "--ntasks=",
            "--ntasks-per-node=",
            "--gres=",
        )
    command = [
        str(token)
        for token in raw
        if not str(token).startswith(removed_prefixes)
    ]
    script_index = next(
        (
            index
            for index, token in enumerate(command)
            if token.endswith("/sbatch/run_adaptive_binary_variant.sh")
        ),
        None,
    )
    if script_index is None:
        raise ValueError(f"{row.get('key')} does not use the ABPH variant worker")
    label = re.sub(r"[^A-Za-z0-9_]+", "_", variant)
    log_root = log_dir.as_posix()
    options = [
        f"--job-name=abph_repair_{label}",
        f"--output={log_root}/abph_repair_{label}_%j.out",
        f"--error={log_root}/abph_repair_{label}_%j.err",
    ]
    if oracle_reference:
        options.extend(
            (
                "--nodes=1",
                "--ntasks=1",
                "--ntasks-per-node=1",
                "--gres=gpu:gh200:1",
            )
        )
    if dependencies:
        if not all(str(value).isdigit() for value in dependencies):
            raise ValueError(f"{variant} has invalid dependencies: {dependencies}")
        options.append("--dependency=afterok:" + ":".join(dependencies))
    command[script_index:script_index] = options
    return command


def _variant_environment(
    row: Mapping[str, Any],
    project_dir: Path,
    *,
    variant: str,
    source_identity: tuple[str, str] | None,
) -> dict[str, str]:
    environment = _job_environment(
        row,
        project_dir,
        source_identity=source_identity,
    )
    if variant in ORACLE_REFERENCE_VARIANTS:
        environment.update(
            {
                "ABPH_RECONSTRUCTOR_PARALLELISM": "single",
                "ABPH_JOB_LAUNCHER": "direct",
                "ABPH_DISTRIBUTED_NODES": "1",
                "ABPH_DISTRIBUTED_NTASKS": "1",
                "ABPH_DISTRIBUTED_NTASKS_PER_NODE": "1",
                "ABPH_DISTRIBUTED_GPUS_PER_NODE": "1",
                "ABPH_DISTRIBUTED_WORLD_SIZE": "1",
            }
        )
    return environment


def _submit(
    row: Mapping[str, Any],
    *,
    variant: str,
    dependencies: Sequence[str],
    project_dir: Path,
    log_dir: Path,
    dry_run: bool,
    source_identity: tuple[str, str] | None,
) -> str:
    command = repaired_variant_command(
        row,
        variant=variant,
        dependencies=dependencies,
        log_dir=log_dir,
    )
    if dry_run:
        print(shlex.join(command))
        return f"DRYRUN_{variant}"
    try:
        completed = subprocess.run(
            command,
            cwd=project_dir,
            env=_variant_environment(
                row,
                project_dir,
                variant=variant,
                source_identity=source_identity,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"sbatch rejected repaired {variant}: "
            f"{(exc.stderr or exc.stdout or '').strip()}"
        ) from exc
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned invalid job id: {completed.stdout!r}")
    return job_id


def _numeric_job_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("job_id", ""))
    if not value.isdigit():
        raise ValueError(f"{row.get('key')} has invalid Slurm job id {value!r}")
    return value


def _contract_job_id(rows: Mapping[str, Mapping[str, Any]], variant: str) -> str | None:
    row = rows.get(f"runtime_batch_contract:{variant}")
    return None if row is None else _numeric_job_id(row)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    manifest = Path(args.submission_manifest).resolve()
    log_dir = Path(args.log_dir or project_dir / "fresh_check_logs").resolve()
    rows = _rows(manifest)
    requested = tuple(dict.fromkeys(str(value) for value in args.variants))
    unknown = [
        variant for variant in requested if f"variant:{variant}" not in rows
    ]
    if unknown:
        raise ValueError(f"submission manifest lacks variants: {unknown}")
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    source_identity = None if args.dry_run else _source_identity(project_dir)

    results: list[dict[str, Any]] = []
    for variant in requested:
        row = rows[f"variant:{variant}"]
        old_job_id = _numeric_job_id(row)
        consumers = [
            candidate
            for candidate in rows.values()
            if old_job_id
            in tuple(str(value) for value in candidate.get("dependencies", ()))
        ]
        consumer_ids = [_numeric_job_id(candidate) for candidate in consumers]
        if not consumers:
            raise RuntimeError(f"{variant} has no recorded downstream consumers")
        if not args.dry_run:
            _require_pending(consumer_ids)

        drop_ids = set()
        if variant in ORACLE_REFERENCE_VARIANTS:
            contract_job_id = _contract_job_id(rows, variant)
            if contract_job_id is not None:
                drop_ids.add(contract_job_id)
        prerequisite_ids = [
            str(value)
            for value in row.get("dependencies", ())
            if str(value) not in drop_ids
        ]
        dependencies = _remaining_dependencies(
            prerequisite_ids, dry_run=args.dry_run
        )
        new_job_id = _submit(
            row,
            variant=variant,
            dependencies=dependencies,
            project_dir=project_dir,
            log_dir=log_dir,
            dry_run=args.dry_run,
            source_identity=source_identity,
        )

        rewired = []
        for consumer, consumer_id in zip(consumers, consumer_ids, strict=True):
            current = (
                f"afterok:{old_job_id}"
                if args.dry_run
                else _live_dependency(consumer_id)
            )
            dependency = replace_dependency_job(
                current,
                old_job_id=old_job_id,
                new_job_id=new_job_id,
            )
            command = (
                "scontrol",
                "update",
                f"JobId={consumer_id}",
                f"Dependency={dependency}",
            )
            if args.dry_run:
                print(shlex.join(command))
            else:
                subprocess.run(command, check=True)
            rewired.append(
                {
                    "key": str(consumer["key"]),
                    "job_id": consumer_id,
                    "dependency": dependency,
                }
            )

        old_state = None if args.dry_run else _slurm_job_state(old_job_id)
        if old_state in _ACTIVE_SLURM_STATES:
            subprocess.run(("scancel", old_job_id), check=True)
        results.append(
            {
                "variant": variant,
                "old_job_id": old_job_id,
                "old_state": old_state,
                "new_job_id": new_job_id,
                "dependencies": dependencies,
                "rewired": rewired,
            }
        )

    print(json.dumps({"ok": True, "repairs": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

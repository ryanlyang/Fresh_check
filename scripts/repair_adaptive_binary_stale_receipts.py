#!/usr/bin/env python3
"""Resubmit poisoned ABPH receipt jobs and rewire their pending consumers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repair_adaptive_binary_failed_model_wave import (
    _drop_completed_afterok_dependencies,
)
from scripts.repair_adaptive_binary_runtime_batch_contract import (
    _live_dependency,
    _rows,
    replace_dependency_job,
    update_pending_dependency,
)
from scripts.repair_adaptive_binary_storage_acceptance_graph import (
    _job_environment,
    _require_pending,
    _slurm_job_state,
    _source_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-manifest", required=True)
    parser.add_argument("--job-ids", nargs="+", required=True)
    parser.add_argument("--project-dir", default="/home/ryreu/atlas/Fresh_check")
    parser.add_argument("--log-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def repaired_receipt_command(
    row: Mapping[str, Any],
    *,
    old_job_id: str,
    dependency: str,
    log_dir: Path,
) -> list[str]:
    raw = row.get("command")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{row.get('key')} lacks its original sbatch command")
    command = [
        str(token)
        for token in raw
        if not str(token).startswith(
            ("--dependency=", "--error=", "--job-name=", "--output=")
        )
    ]
    script_index = next(
        (
            index
            for index, token in enumerate(command)
            if "/sbatch/" in token and token.endswith(".sh")
        ),
        None,
    )
    if script_index is None:
        raise ValueError(f"{row.get('key')} lacks an sbatch worker")
    log_root = log_dir.as_posix()
    options = [
        f"--job-name=abph_repair_receipt_{old_job_id}",
        f"--output={log_root}/abph_repair_receipt_{old_job_id}_%j.out",
        f"--error={log_root}/abph_repair_receipt_{old_job_id}_%j.err",
        f"--dependency={dependency}",
    ]
    command[script_index:script_index] = options
    return command


def _submit(
    row: Mapping[str, Any],
    *,
    old_job_id: str,
    dependency: str,
    project_dir: Path,
    log_dir: Path,
    source_identity: tuple[str, str] | None,
    dry_run: bool,
) -> str:
    command = repaired_receipt_command(
        row,
        old_job_id=old_job_id,
        dependency=dependency,
        log_dir=log_dir,
    )
    if dry_run:
        print(shlex.join(command))
        return f"DRYRUN_{old_job_id}"
    completed = subprocess.run(
        command,
        cwd=project_dir,
        env=_job_environment(
            row,
            project_dir,
            source_identity=source_identity,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned invalid job id: {completed.stdout!r}")
    return job_id


def _numeric_job_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("job_id", ""))
    if not value.isdigit():
        raise ValueError(f"{row.get('key')} has invalid Slurm job id {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    log_dir = Path(args.log_dir or project_dir / "fresh_check_logs").resolve()
    manifest = Path(args.submission_manifest).resolve()
    rows = _rows(manifest)
    rows_by_job = {
        _numeric_job_id(row): row
        for row in rows.values()
    }
    requested = tuple(dict.fromkeys(str(value) for value in args.job_ids))
    if not all(value.isdigit() for value in requested):
        raise ValueError("--job-ids must contain numeric Slurm job IDs")
    missing = sorted(set(requested) - set(rows_by_job))
    if missing:
        raise ValueError(f"submission manifest lacks jobs: {missing}")
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    source_identity = None if args.dry_run else _source_identity(project_dir)

    repairs: list[dict[str, Any]] = []
    for old_job_id in requested:
        row = rows_by_job[old_job_id]
        if not str(row.get("key", "")).startswith("receipt:"):
            raise ValueError(f"{old_job_id} is not an ABPH receipt job")
        if not args.dry_run and _slurm_job_state(old_job_id) != "PENDING":
            raise RuntimeError(f"stale receipt {old_job_id} is not pending")
        dependency = (
            "afterok:DRYRUN"
            if args.dry_run
            else _drop_completed_afterok_dependencies(
                _live_dependency(old_job_id),
                state_for_job=_slurm_job_state,
            )
        )
        if not dependency:
            raise RuntimeError(f"stale receipt {old_job_id} has no live prerequisite")

        consumers = [
            candidate
            for candidate in rows.values()
            if old_job_id
            in tuple(str(value) for value in candidate.get("dependencies", ()))
        ]
        consumer_ids = [_numeric_job_id(candidate) for candidate in consumers]
        if not consumers:
            raise RuntimeError(f"stale receipt {old_job_id} has no consumers")
        if not args.dry_run:
            _require_pending(consumer_ids)

        new_job_id = _submit(
            row,
            old_job_id=old_job_id,
            dependency=dependency,
            project_dir=project_dir,
            log_dir=log_dir,
            source_identity=source_identity,
            dry_run=args.dry_run,
        )
        rewired = []
        for consumer, consumer_id in zip(consumers, consumer_ids, strict=True):
            current = (
                f"afterok:{old_job_id}"
                if args.dry_run
                else _live_dependency(consumer_id)
            )
            repaired = replace_dependency_job(
                current,
                old_job_id=old_job_id,
                new_job_id=new_job_id,
            )
            if not args.dry_run:
                repaired = _drop_completed_afterok_dependencies(
                    repaired,
                    state_for_job=_slurm_job_state,
                )
            command = (
                "scontrol",
                "update",
                f"JobId={consumer_id}",
                f"Dependency={repaired}",
            )
            if args.dry_run:
                print(shlex.join(command))
            else:
                update_pending_dependency(consumer_id, repaired)
            rewired.append(
                {
                    "key": str(consumer["key"]),
                    "job_id": consumer_id,
                    "dependency": repaired,
                }
            )
        if args.dry_run:
            print(shlex.join(("scancel", old_job_id)))
        else:
            subprocess.run(("scancel", old_job_id), check=True)
        repairs.append(
            {
                "old_job_id": old_job_id,
                "new_job_id": new_job_id,
                "dependency": dependency,
                "rewired": rewired,
            }
        )
    print(json.dumps({"ok": True, "repairs": repairs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

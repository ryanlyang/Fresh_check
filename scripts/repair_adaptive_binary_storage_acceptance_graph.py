#!/usr/bin/env python3
"""Replace failed storage-acceptance nodes in an existing ABPH Slurm graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


TEST_KEY = "acceptance:component_parity_tests"
SMOKE_KEY = "acceptance:ram_lifecycle_smoke"
COMPILE_KEY = "acceptance:storage_smoke"
REQUIRED_KEYS = (TEST_KEY, SMOKE_KEY, COMPILE_KEY)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-manifest", required=True)
    parser.add_argument("--project-dir", default="/home/ryreu/atlas/Fresh_check")
    parser.add_argument("--log-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("submission_commands")
    if not isinstance(raw_rows, list):
        raise ValueError(f"submission manifest lacks submission_commands: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("key"), str):
            raise ValueError(f"invalid submission row in {path}")
        rows[str(raw["key"])] = dict(raw)
    missing = sorted(set(REQUIRED_KEYS).difference(rows))
    if missing:
        raise ValueError(f"submission manifest lacks required jobs: {missing}")
    return rows


def _numeric_job_id(row: Mapping[str, Any], *, key: str) -> str:
    value = str(row.get("job_id", ""))
    if not value.isdigit():
        raise ValueError(f"{key} has an invalid Slurm job id: {value!r}")
    return value


def repaired_sbatch_command(
    row: Mapping[str, Any],
    *,
    label: str,
    dependencies: Sequence[str],
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
            if token.endswith("/sbatch/run_adaptive_binary_storage_acceptance.sh")
        ),
        None,
    )
    if script_index is None:
        raise ValueError(f"{row.get('key')} does not use the storage worker")
    options = [
        f"--job-name=abph_repair_{label}",
        f"--output={log_dir}/abph_repair_{label}_%j.out",
        f"--error={log_dir}/abph_repair_{label}_%j.err",
    ]
    if dependencies:
        if not all(str(value).isdigit() for value in dependencies):
            raise ValueError(f"{label} has invalid dependency ids: {dependencies}")
        options.append("--dependency=afterok:" + ":".join(dependencies))
    command[script_index:script_index] = options
    return command


def _job_environment(row: Mapping[str, Any], project_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    recorded = row.get("environment")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"{row.get('key')} lacks its recorded environment")
    environment.update({str(key): str(value) for key, value in recorded.items()})
    environment.update(
        {
            "ABPH_SUPPRESS_SLURM_LOGS": "0",
            "PROJECT_DIR": str(project_dir),
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _submit(
    row: Mapping[str, Any],
    *,
    label: str,
    dependencies: Sequence[str],
    project_dir: Path,
    log_dir: Path,
    dry_run: bool,
) -> str:
    command = repaired_sbatch_command(
        row, label=label, dependencies=dependencies, log_dir=log_dir
    )
    if dry_run:
        print(shlex.join(command))
        return f"DRYRUN_{label}"
    completed = subprocess.run(
        command,
        cwd=project_dir,
        env=_job_environment(row, project_dir),
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned an invalid job id: {completed.stdout!r}")
    return job_id


def _require_pending(job_ids: Sequence[str]) -> None:
    for job_id in job_ids:
        completed = subprocess.run(
            ("scontrol", "show", "job", "-o", job_id),
            check=True,
            capture_output=True,
            text=True,
        )
        if "JobState=PENDING" not in completed.stdout:
            raise RuntimeError(
                f"refusing to alter non-pending downstream job {job_id}: "
                f"{completed.stdout.strip()}"
            )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    manifest = Path(args.submission_manifest).resolve()
    log_dir = Path(args.log_dir or project_dir / "fresh_check_logs").resolve()
    worker = project_dir / "sbatch" / "run_adaptive_binary_storage_acceptance.sh"
    worker_source = worker.read_text(encoding="utf-8")
    if 'source "${PROJECT_DIR}/sbatch/common.sh"' not in worker_source:
        raise RuntimeError(f"the repaired project-anchored worker is not present: {worker}")
    rows = _load_rows(manifest)

    old_test = _numeric_job_id(rows[TEST_KEY], key=TEST_KEY)
    old_smoke = _numeric_job_id(rows[SMOKE_KEY], key=SMOKE_KEY)
    old_compile = _numeric_job_id(rows[COMPILE_KEY], key=COMPILE_KEY)
    descendants = [
        row
        for row in rows.values()
        if old_compile in [str(value) for value in row.get("dependencies", ())]
    ]
    descendant_ids = [
        _numeric_job_id(row, key=str(row["key"])) for row in descendants
    ]
    if not descendants:
        raise RuntimeError(f"no direct descendants reference old compiler {old_compile}")
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        _require_pending(descendant_ids)

    test_dependencies = [str(value) for value in rows[TEST_KEY]["dependencies"]]
    smoke_dependencies = [str(value) for value in rows[SMOKE_KEY]["dependencies"]]
    new_test = _submit(
        rows[TEST_KEY],
        label="storage_tests",
        dependencies=test_dependencies,
        project_dir=project_dir,
        log_dir=log_dir,
        dry_run=args.dry_run,
    )
    new_smoke = _submit(
        rows[SMOKE_KEY],
        label="storage_smoke",
        dependencies=smoke_dependencies,
        project_dir=project_dir,
        log_dir=log_dir,
        dry_run=args.dry_run,
    )
    replacements = {old_test: new_test, old_smoke: new_smoke}
    compile_dependencies = [
        replacements.get(str(value), str(value))
        for value in rows[COMPILE_KEY]["dependencies"]
    ]
    new_compile = _submit(
        rows[COMPILE_KEY],
        label="storage_compile",
        dependencies=compile_dependencies,
        project_dir=project_dir,
        log_dir=log_dir,
        dry_run=args.dry_run,
    )

    rewired: list[dict[str, Any]] = []
    for row, job_id in zip(descendants, descendant_ids, strict=True):
        dependencies = [
            new_compile if str(value) == old_compile else str(value)
            for value in row["dependencies"]
        ]
        dependency = "afterok:" + ":".join(dependencies)
        if args.dry_run:
            print(shlex.join(("scontrol", "update", f"JobId={job_id}", f"Dependency={dependency}")))
        else:
            subprocess.run(
                ("scontrol", "update", f"JobId={job_id}", f"Dependency={dependency}"),
                check=True,
            )
        rewired.append(
            {"key": row["key"], "job_id": job_id, "dependency": dependency}
        )
    if args.dry_run:
        print(shlex.join(("scancel", old_compile)))
    else:
        subprocess.run(
            ("scancel", old_compile),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print(
        json.dumps(
            {
                "ok": True,
                "dry_run": bool(args.dry_run),
                "manifest": str(manifest),
                "replacements": {
                    TEST_KEY: {"old": old_test, "new": new_test},
                    SMOKE_KEY: {"old": old_smoke, "new": new_smoke},
                    COMPILE_KEY: {"old": old_compile, "new": new_compile},
                },
                "rewired": rewired,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

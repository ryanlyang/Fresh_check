#!/usr/bin/env python3
"""Reprobe one failed ABPH runtime batch contract and rewire its live consumers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repair_adaptive_binary_storage_acceptance_graph import (
    _job_environment,
    _require_pending,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-manifest", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--project-dir", default="/home/ryreu/atlas/Fresh_check")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("submission_commands")
    if not isinstance(raw, list):
        raise ValueError(f"submission manifest lacks submission_commands: {path}")
    return {
        str(row["key"]): dict(row)
        for row in raw
        if isinstance(row, Mapping) and isinstance(row.get("key"), str)
    }


def _job_id(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get("job_id", ""))
    if not value.isdigit():
        raise ValueError(f"{key} has invalid Slurm job id {value!r}")
    return value


def repaired_command(
    row: Mapping[str, Any],
    *,
    label: str,
    dependencies: Sequence[str],
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
    options = [
        f"--job-name=abph_repair_{label}",
        "--output=/dev/null",
        "--error=/dev/null",
    ]
    if dependencies:
        if not all(str(value).isdigit() for value in dependencies):
            raise ValueError(f"{label} has invalid dependency ids: {dependencies}")
        options.append("--dependency=afterok:" + ":".join(dependencies))
    command[script_index:script_index] = options
    return command


def _submit(
    row: Mapping[str, Any],
    *,
    label: str,
    dependencies: Sequence[str],
    project_dir: Path,
    dry_run: bool,
) -> str:
    command = repaired_command(row, label=label, dependencies=dependencies)
    if dry_run:
        print(shlex.join(command))
        return f"DRYRUN_{label}"
    try:
        completed = subprocess.run(
            command,
            cwd=project_dir,
            env=_job_environment(row, project_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"sbatch rejected repaired {label} job: "
            f"{(exc.stderr or exc.stdout or '').strip()}"
        ) from exc
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned invalid job id: {completed.stdout!r}")
    return job_id


def replace_dependency_job(
    dependency: str,
    *,
    old_job_id: str,
    new_job_id: str,
) -> str:
    if dependency in {"", "(null)"}:
        raise ValueError("consumer no longer declares a dependency")
    pattern = re.compile(rf"(?<!\d){re.escape(old_job_id)}(?!\d)")
    repaired, count = pattern.subn(new_job_id, dependency)
    if count != 1:
        raise ValueError(
            f"expected one reference to {old_job_id}, found {count}: {dependency}"
        )
    return repaired


def _live_dependency(job_id: str) -> str:
    completed = subprocess.run(
        ("scontrol", "show", "job", "-o", job_id),
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(?:^|\s)Dependency=(\S+)", completed.stdout)
    if match is None:
        raise RuntimeError(f"pending job {job_id} has no readable dependency")
    return re.sub(r"\((?:un)?fulfilled\)", "", match.group(1))


def _archive_stale_evidence(root: Path, variant: str, *, dry_run: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "archives" / f"runtime_batch_repair_{stamp}" / variant
    sources = (
        root / "runtime_batch_measurements" / variant,
        root / "runtime_batch_contracts" / variant,
    )
    if dry_run:
        for source in sources:
            if source.exists():
                print(f"archive {source} -> {archive / source.parent.name}")
        return archive
    for source in sources:
        if not source.exists():
            continue
        destination = archive / source.parent.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    return archive


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    manifest = Path(args.submission_manifest).resolve()
    rows = _rows(manifest)
    variant = str(args.variant)
    contract_key = f"runtime_batch_contract:{variant}"
    probe_prefix = f"runtime_batch_probe:{variant}:"
    if contract_key not in rows:
        raise ValueError(f"manifest has no runtime contract for {variant}")
    probes = [row for key, row in rows.items() if key.startswith(probe_prefix)]
    if not probes:
        raise ValueError(f"manifest has no runtime probes for {variant}")
    contract_row = rows[contract_key]
    old_contract = _job_id(contract_row, contract_key)
    consumers = [
        row
        for row in rows.values()
        if old_contract in [str(value) for value in row.get("dependencies", ())]
    ]
    consumer_ids = [_job_id(row, str(row["key"])) for row in consumers]
    if not consumers:
        raise RuntimeError(f"no consumers reference failed contract {old_contract}")
    if not args.dry_run:
        _require_pending(consumer_ids)

    root = manifest.parents[1]
    acceptance_path = root / "storage" / "storage_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("ok") is not True:
        raise RuntimeError("campaign storage acceptance is not successful")
    archive = _archive_stale_evidence(root, variant, dry_run=args.dry_run)

    new_probes = [
        _submit(
            row,
            label=f"{variant}_{index}",
            dependencies=(),
            project_dir=project_dir,
            dry_run=args.dry_run,
        )
        for index, row in enumerate(probes)
    ]
    new_contract = _submit(
        contract_row,
        label=f"{variant}_contract",
        dependencies=new_probes,
        project_dir=project_dir,
        dry_run=args.dry_run,
    )

    rewired: list[dict[str, str]] = []
    for row, job_id in zip(consumers, consumer_ids, strict=True):
        current = (
            "afterok:" + old_contract
            if args.dry_run
            else _live_dependency(job_id)
        )
        dependency = replace_dependency_job(
            current,
            old_job_id=old_contract,
            new_job_id=new_contract,
        )
        if args.dry_run:
            print(
                shlex.join(
                    ("scontrol", "update", f"JobId={job_id}", f"Dependency={dependency}")
                )
            )
        else:
            subprocess.run(
                ("scontrol", "update", f"JobId={job_id}", f"Dependency={dependency}"),
                check=True,
            )
        rewired.append(
            {"key": str(row["key"]), "job_id": job_id, "dependency": dependency}
        )

    print(
        json.dumps(
            {
                "ok": True,
                "variant": variant,
                "old_contract_job_id": old_contract,
                "new_probe_job_ids": new_probes,
                "new_contract_job_id": new_contract,
                "archive": str(archive),
                "rewired": rewired,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

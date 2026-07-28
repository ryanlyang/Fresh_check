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
    _ACTIVE_SLURM_STATES,
    _FAILED_SLURM_STATES,
    _job_environment,
    _remaining_dependencies,
    _require_pending,
    _slurm_job_state,
    _source_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-manifest", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--project-dir", default="/home/ryreu/atlas/Fresh_check")
    parser.add_argument("--log-dir")
    parser.add_argument(
        "--dependency-replacement",
        action="append",
        default=[],
        metavar="OLD_JOB_ID=NEW_JOB_ID",
        help="Replace an additional failed prerequisite while rewiring consumers.",
    )
    parser.add_argument(
        "--superseded-contract-job",
        help=(
            "Failed replacement contract currently referenced by consumers; "
            "use this when repairing an already repaired contract."
        ),
    )
    parser.add_argument(
        "--resubmit-failed-consumers",
        action="store_true",
        help=(
            "Resubmit a failed variant/renderer consumer against the repaired "
            "contract and rewire its pending direct descendants."
        ),
    )
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
    log_dir: Path | None = None,
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
    options = [f"--job-name=abph_repair_{label}"]
    if log_dir is None:
        options.extend(("--output=/dev/null", "--error=/dev/null"))
    else:
        options.extend(
            (
                f"--output={log_dir}/abph_repair_{label}_%j.out",
                f"--error={log_dir}/abph_repair_{label}_%j.err",
            )
        )
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
    log_dir: Path | None,
    dry_run: bool,
    source_identity: tuple[str, str] | None = None,
) -> str:
    command = repaired_command(
        row,
        label=label,
        dependencies=dependencies,
        log_dir=log_dir,
    )
    if dry_run:
        print(shlex.join(command))
        return f"DRYRUN_{label}"
    try:
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


def _job_replacements(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        old_job_id, separator, new_job_id = str(value).partition("=")
        if (
            not separator
            or not old_job_id.isdigit()
            or not new_job_id.isdigit()
        ):
            raise ValueError(
                "--dependency-replacement must use OLD_JOB_ID=NEW_JOB_ID"
            )
        if old_job_id in result:
            raise ValueError(f"duplicate replacement for job {old_job_id}")
        result[old_job_id] = new_job_id
    return result


def _replace_dependency_if_present(
    dependency: str,
    *,
    old_job_id: str,
    new_job_id: str,
) -> str:
    old_pattern = re.compile(rf"(?<!\d){re.escape(old_job_id)}(?!\d)")
    new_pattern = re.compile(rf"(?<!\d){re.escape(new_job_id)}(?!\d)")
    old_count = len(old_pattern.findall(dependency))
    new_count = len(new_pattern.findall(dependency))
    if old_count == 1 and new_count == 0:
        return replace_dependency_job(
            dependency,
            old_job_id=old_job_id,
            new_job_id=new_job_id,
        )
    if old_count == 0 and new_count in {0, 1}:
        return dependency
    raise ValueError(
        "dependency replacement is ambiguous: "
        f"old_count={old_count}, new_count={new_count}, dependency={dependency}"
    )


def replace_one_dependency_job(
    dependency: str,
    *,
    old_job_ids: Sequence[str],
    new_job_id: str,
) -> str:
    """Replace exactly one original or superseded dependency reference."""

    candidates = tuple(dict.fromkeys(str(value) for value in old_job_ids))
    matches = [
        value
        for value in candidates
        if re.search(rf"(?<!\d){re.escape(value)}(?!\d)", dependency)
    ]
    if re.search(rf"(?<!\d){re.escape(new_job_id)}(?!\d)", dependency):
        if matches:
            raise ValueError(
                "dependency contains both a stale and the new contract job"
            )
        return dependency
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one original or superseded dependency, found "
            f"{matches}: {dependency}"
        )
    return replace_dependency_job(
        dependency,
        old_job_id=matches[0],
        new_job_id=new_job_id,
    )


def update_pending_dependency(job_id: str, dependency: str) -> bool:
    """Update a pending job, thawing Slurm's frozen DNS state if necessary.

    Returns ``True`` when the requeue-hold recovery path was required.
    """

    command = (
        "scontrol",
        "update",
        f"JobId={job_id}",
        f"Dependency={dependency}",
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return False
    message = (completed.stderr or completed.stdout or "").strip()
    if "Job dependency problem" not in message:
        raise RuntimeError(
            f"scontrol rejected dependency repair for {job_id}: {message}"
        )
    subprocess.run(("scontrol", "requeuehold", job_id), check=True)
    subprocess.run(command, check=True)
    subprocess.run(("scontrol", "release", job_id), check=True)
    return True


def _consumer_dependencies(
    row: Mapping[str, Any],
    *,
    old_contract: str,
    new_contract: str,
    dry_run: bool,
) -> list[str]:
    replaced = [
        new_contract if str(value) == old_contract else str(value)
        for value in row.get("dependencies", ())
    ]
    if old_contract in replaced:
        raise RuntimeError("failed consumer retained the obsolete runtime contract")
    return _remaining_dependencies(replaced, dry_run=dry_run)


def _drop_completed_afterok_dependencies(
    dependency: str,
    *,
    state_for_job: Any,
) -> str:
    retained: list[str] = []
    for term in dependency.split(","):
        pieces = term.split(":")
        if len(pieces) < 2 or pieces[0] != "afterok":
            raise ValueError(f"unsupported dependency term: {term}")
        active_ids = [
            job_id
            for job_id in pieces[1:]
            if state_for_job(job_id) != "COMPLETED"
        ]
        if active_ids:
            retained.append("afterok:" + ":".join(active_ids))
    return ",".join(retained)


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
    dependency = match.group(1)
    # Slurm adds display-only state annotations after dependencies freeze.
    # Feeding those annotations back to `scontrol update` is invalid.
    return re.sub(r"\((?:fulfilled|unfulfilled|failed)\)", "", dependency)


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
    log_dir = Path(args.log_dir or project_dir / "fresh_check_logs").resolve()
    manifest = Path(args.submission_manifest).resolve()
    companion_replacements = _job_replacements(args.dependency_replacement)
    superseded_contract = args.superseded_contract_job
    if superseded_contract is not None and not str(superseded_contract).isdigit():
        raise ValueError("--superseded-contract-job must be a numeric Slurm job ID")
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
    consumer_states: dict[str, str | None] = {}
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        consumer_states = {
            job_id: _slurm_job_state(job_id) for job_id in consumer_ids
        }
        if args.resubmit_failed_consumers:
            unsupported = {
                job_id: state
                for job_id, state in consumer_states.items()
                if state != "PENDING" and state not in _FAILED_SLURM_STATES
            }
            if unsupported:
                raise RuntimeError(
                    "runtime contract consumers are neither pending nor failed: "
                    f"{unsupported}"
                )
        else:
            _require_pending(consumer_ids)
    source_identity = None if args.dry_run else _source_identity(project_dir)

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
            log_dir=log_dir,
            dry_run=args.dry_run,
            source_identity=source_identity,
        )
        for index, row in enumerate(probes)
    ]
    new_contract = _submit(
        contract_row,
        label=f"{variant}_contract",
        dependencies=new_probes,
        project_dir=project_dir,
        log_dir=log_dir,
        dry_run=args.dry_run,
        source_identity=source_identity,
    )

    rewired: list[dict[str, Any]] = []
    for row, job_id in zip(consumers, consumer_ids, strict=True):
        state = consumer_states.get(job_id)
        if state in _FAILED_SLURM_STATES:
            if not args.resubmit_failed_consumers:
                raise RuntimeError(
                    f"consumer {job_id} failed and resubmission was not approved"
                )
            key = str(row.get("key", ""))
            if not key.startswith("variant:"):
                raise RuntimeError(
                    f"failed runtime-contract consumer is not a variant job: {key}"
                )
            consumer_variant = key.split(":", 1)[1]
            replacement_dependencies = _consumer_dependencies(
                row,
                old_contract=old_contract,
                new_contract=new_contract,
                dry_run=args.dry_run,
            )
            replacement_job = _submit(
                row,
                label=f"{consumer_variant}_consumer",
                dependencies=replacement_dependencies,
                project_dir=project_dir,
                log_dir=log_dir,
                dry_run=args.dry_run,
                source_identity=source_identity,
            )
            descendants = [
                candidate
                for candidate in rows.values()
                if job_id
                in tuple(
                    str(value)
                    for value in candidate.get("dependencies", ())
                )
            ]
            if not descendants:
                raise RuntimeError(
                    f"failed consumer {job_id} has no recorded descendants"
                )
            descendant_ids = [
                _job_id(candidate, str(candidate["key"]))
                for candidate in descendants
            ]
            if not args.dry_run:
                _require_pending(descendant_ids)
            descendant_repairs = []
            for descendant, descendant_id in zip(
                descendants, descendant_ids, strict=True
            ):
                current = (
                    f"afterok:{job_id}"
                    if args.dry_run
                    else _live_dependency(descendant_id)
                )
                dependency = replace_one_dependency_job(
                    current,
                    old_job_ids=(job_id,),
                    new_job_id=replacement_job,
                )
                if not args.dry_run:
                    dependency = _drop_completed_afterok_dependencies(
                        dependency,
                        state_for_job=_slurm_job_state,
                    )
                    update_pending_dependency(descendant_id, dependency)
                else:
                    print(
                        shlex.join(
                            (
                                "scontrol",
                                "update",
                                f"JobId={descendant_id}",
                                f"Dependency={dependency}",
                            )
                        )
                    )
                descendant_repairs.append(
                    {
                        "key": str(descendant["key"]),
                        "job_id": descendant_id,
                        "dependency": dependency,
                    }
                )
            rewired.append(
                {
                    "key": key,
                    "job_id": job_id,
                    "state": state,
                    "replacement_job_id": replacement_job,
                    "replacement_dependencies": replacement_dependencies,
                    "descendants": descendant_repairs,
                }
            )
            continue

        current = (
            "afterok:" + old_contract
            if args.dry_run
            else _live_dependency(job_id)
        )
        dependency = replace_one_dependency_job(
            current,
            old_job_ids=tuple(
                value
                for value in (old_contract, superseded_contract)
                if value is not None
            ),
            new_job_id=new_contract,
        )
        for old_job_id, new_job_id in companion_replacements.items():
            dependency = _replace_dependency_if_present(
                dependency,
                old_job_id=old_job_id,
                new_job_id=new_job_id,
            )
        if not args.dry_run:
            dependency = _drop_completed_afterok_dependencies(
                dependency,
                state_for_job=_slurm_job_state,
            )
        if not dependency:
            raise RuntimeError(
                f"repairing {row['key']} removed every dependency"
            )
        if args.dry_run:
            print(
                shlex.join(
                    ("scontrol", "update", f"JobId={job_id}", f"Dependency={dependency}")
                )
            )
        else:
            update_pending_dependency(job_id, dependency)
        rewired.append(
            {"key": str(row["key"]), "job_id": job_id, "dependency": dependency}
        )

    if not args.dry_run and superseded_contract is not None:
        if _slurm_job_state(superseded_contract) in _ACTIVE_SLURM_STATES:
            subprocess.run(("scancel", superseded_contract), check=True)

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

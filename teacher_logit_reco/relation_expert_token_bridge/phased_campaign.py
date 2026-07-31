"""Authenticated internal phase waves for scientifically ordered campaigns.

The public production graph deliberately stays stable.  A public controller
node may use this module to execute an ordered sequence of internal arrays
whose later membership depends on an immutable selector produced by an
earlier phase.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


INTERNAL_PHASE_PLAN_CONTRACT = "retb_internal_phase_plan_v1"
INTERNAL_PHASE_ROW_COMPLETION_CONTRACT = (
    "retb_internal_phase_row_completion_v1"
)
INTERNAL_PHASE_COMPLETION_CONTRACT = "retb_internal_phase_completion_v1"
PHASED_CONTROLLER_COMPLETION_CONTRACT = (
    "retb_phased_controller_completion_v1"
)
PHASE_RESOURCES = {"cpu", "gpu"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("internal phase path escapes campaign root") from error
    return path


def build_internal_phase_plan(
    *,
    campaign_root: str | Path,
    campaign_spec_sha256: str,
    production_graph_sha256: str,
    controller_id: str,
    phase_id: str,
    sequence_index: int,
    resource: str,
    maximum_concurrent_tasks: int,
    rows: Sequence[Mapping[str, Any]],
    prerequisite_completion_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    if (
        not controller_id
        or not phase_id
        or int(sequence_index) < 0
        or resource not in PHASE_RESOURCES
        or int(maximum_concurrent_tasks) <= 0
        or not rows
    ):
        raise ValueError("internal phase declaration differs")
    normalized = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        if set(row) != {
            "task_id",
            "argv",
            "environment",
            "expected_outputs",
            "input_artifact_hashes",
        }:
            raise ValueError("internal phase row fields differ")
        if row["task_id"] != f"{phase_id}:{index}":
            raise ValueError("internal phase row identity differs")
        argv = [str(value) for value in row["argv"]]
        outputs = [
            str(_inside(root, value)) for value in row["expected_outputs"]
        ]
        if (
            len(argv) < 2
            or argv[0] not in {"python", "python3"}
            or not argv[1].replace("\\", "/").startswith("scripts/")
            or not outputs
            or len(outputs) != len(set(outputs))
        ):
            raise ValueError("internal phase worker declaration differs")
        normalized.append(
            {
                "task_id": row["task_id"],
                "argv": argv,
                "environment": {
                    str(name): str(value)
                    for name, value in sorted(row["environment"].items())
                },
                "expected_outputs": outputs,
                "input_artifact_hashes": {
                    str(name): require_sha256(
                        value, name=f"input_artifact_hashes.{name}"
                    )
                    for name, value in sorted(
                        row["input_artifact_hashes"].items()
                    )
                },
            }
        )
    return with_content_hash(
        {
            "contract": INTERNAL_PHASE_PLAN_CONTRACT,
            "schema_version": 1,
            "campaign_root": str(root),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "production_graph_sha256": require_sha256(
                production_graph_sha256, name="production_graph_sha256"
            ),
            "controller_id": str(controller_id),
            "phase_id": str(phase_id),
            "sequence_index": int(sequence_index),
            "resource": resource,
            "maximum_concurrent_tasks": int(maximum_concurrent_tasks),
            "prerequisite_completion_hashes": {
                str(name): require_sha256(
                    value,
                    name=f"prerequisite_completion_hashes.{name}",
                )
                for name, value in sorted(
                    prerequisite_completion_hashes.items()
                )
            },
            "rows": normalized,
            "row_count": len(normalized),
            "scientific_performance_used_to_omit_rows": False,
            "negative_scientific_results_block_continuation": False,
            "source": dict(source),
        }
    )


def validate_internal_phase_plan(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=INTERNAL_PHASE_PLAN_CONTRACT
    )
    expected = build_internal_phase_plan(
        campaign_root=payload["campaign_root"],
        campaign_spec_sha256=payload["campaign_spec_sha256"],
        production_graph_sha256=payload["production_graph_sha256"],
        controller_id=payload["controller_id"],
        phase_id=payload["phase_id"],
        sequence_index=int(payload["sequence_index"]),
        resource=payload["resource"],
        maximum_concurrent_tasks=int(
            payload["maximum_concurrent_tasks"]
        ),
        rows=payload["rows"],
        prerequisite_completion_hashes=payload[
            "prerequisite_completion_hashes"
        ],
        source=payload["source"],
    )
    if dict(payload) != expected:
        raise ValueError("internal phase plan semantics differ")
    return digest


def phase_plan_path(
    campaign_root: str | Path, *, controller_id: str, phase_id: str
) -> Path:
    return (
        Path(campaign_root).resolve()
        / "job_ledgers"
        / "internal_phases"
        / controller_id
        / phase_id
        / "phase_plan.json"
    )


def phase_row_completion_path(
    campaign_root: str | Path,
    *,
    controller_id: str,
    phase_id: str,
    task_index: int,
) -> Path:
    return (
        phase_plan_path(
            campaign_root,
            controller_id=controller_id,
            phase_id=phase_id,
        ).parent
        / "rows"
        / f"{int(task_index):06d}.json"
    )


def phase_completion_path(
    campaign_root: str | Path, *, controller_id: str, phase_id: str
) -> Path:
    return (
        phase_plan_path(
            campaign_root,
            controller_id=controller_id,
            phase_id=phase_id,
        ).parent
        / "phase_completion.json"
    )


def _expected_output_hashes(row: Mapping[str, Any]) -> dict[str, str]:
    output = {}
    for value in row["expected_outputs"]:
        path = Path(value)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"internal phase output is absent or unsafe: {path}"
            )
        output[str(path.resolve())] = _file_sha256(path)
    return output


def reusable_internal_phase_row(
    *,
    plan: Mapping[str, Any],
    task_index: int,
) -> dict[str, Any] | None:
    validate_internal_phase_plan(plan)
    index = int(task_index)
    path = phase_row_completion_path(
        plan["campaign_root"],
        controller_id=plan["controller_id"],
        phase_id=plan["phase_id"],
        task_index=index,
    )
    if not path.is_file():
        return None
    receipt = load_hashed_json(
        path, expected_contract=INTERNAL_PHASE_ROW_COMPLETION_CONTRACT
    )
    row = plan["rows"][index]
    if (
        receipt.get("phase_plan_sha256") != plan["content_hash"]
        or int(receipt.get("task_index", -1)) != index
        or receipt.get("task_id") != row["task_id"]
        or receipt.get("output_file_sha256")
        != _expected_output_hashes(row)
        or receipt.get("source") != plan.get("source")
    ):
        raise ValueError("reusable internal phase row lineage differs")
    return receipt


def execute_internal_phase_row(
    *,
    plan: Mapping[str, Any],
    task_index: int,
    repo_root: str | Path,
) -> dict[str, Any]:
    validate_internal_phase_plan(plan)
    index = int(task_index)
    if index < 0 or index >= int(plan["row_count"]):
        raise IndexError("internal phase task index is outside the plan")
    reused = reusable_internal_phase_row(plan=plan, task_index=index)
    if reused is not None:
        return reused
    row = plan["rows"][index]
    environment = dict(os.environ)
    environment.update(row["environment"])
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        row["argv"],
        cwd=Path(repo_root).resolve(),
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"internal phase worker failed with {completed.returncode}: "
            f"{row['task_id']}"
        )
    receipt = with_content_hash(
        {
            "contract": INTERNAL_PHASE_ROW_COMPLETION_CONTRACT,
            "schema_version": 1,
            "phase_plan_sha256": plan["content_hash"],
            "controller_id": plan["controller_id"],
            "phase_id": plan["phase_id"],
            "task_index": index,
            "task_id": row["task_id"],
            "input_artifact_hashes": row["input_artifact_hashes"],
            "output_file_sha256": _expected_output_hashes(row),
            "source": plan["source"],
        }
    )
    write_immutable_json(
        phase_row_completion_path(
            plan["campaign_root"],
            controller_id=plan["controller_id"],
            phase_id=plan["phase_id"],
            task_index=index,
        ),
        receipt,
    )
    return receipt


def publish_internal_phase_completion(
    *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    validate_internal_phase_plan(plan)
    receipts = []
    for index in range(int(plan["row_count"])):
        receipt = reusable_internal_phase_row(
            plan=plan, task_index=index
        )
        if receipt is None:
            raise ValueError("internal phase row coverage is incomplete")
        receipts.append(receipt)
    artifact = with_content_hash(
        {
            "contract": INTERNAL_PHASE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "phase_plan_sha256": plan["content_hash"],
            "controller_id": plan["controller_id"],
            "phase_id": plan["phase_id"],
            "sequence_index": plan["sequence_index"],
            "row_count": plan["row_count"],
            "row_completion_hashes": [
                receipt["content_hash"] for receipt in receipts
            ],
            "all_outputs_revalidated_after_last_row": True,
            "scientific_result_sign_used_as_completion_gate": False,
            "source": plan["source"],
        }
    )
    write_immutable_json(
        phase_completion_path(
            plan["campaign_root"],
            controller_id=plan["controller_id"],
            phase_id=plan["phase_id"],
        ),
        artifact,
    )
    return artifact


def reusable_internal_phase_completion(
    *, plan: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = phase_completion_path(
        plan["campaign_root"],
        controller_id=plan["controller_id"],
        phase_id=plan["phase_id"],
    )
    if not path.is_file():
        return None
    retained = load_hashed_json(
        path, expected_contract=INTERNAL_PHASE_COMPLETION_CONTRACT
    )
    current = publish_internal_phase_completion(plan=plan)
    if retained != current:
        raise ValueError("reusable internal phase completion differs")
    return retained


def execute_internal_phase(
    *,
    plan: Mapping[str, Any],
    repo_root: str | Path,
    slurm_task_script: str | Path,
) -> dict[str, Any]:
    """Execute one complete phase locally or as a nested Slurm array."""

    validate_internal_phase_plan(plan)
    retained = reusable_internal_phase_completion(plan=plan)
    if retained is not None:
        return retained
    if os.environ.get("SLURM_JOB_ID"):
        maximum = min(
            int(plan["maximum_concurrent_tasks"]), int(plan["row_count"])
        )
        arguments = [
            "sbatch",
            "--parsable",
            "--wait",
            f"--array=0-{int(plan['row_count']) - 1}%{maximum}",
            f"--job-name={plan['controller_id']}_{plan['phase_id']}",
            (
                "--output="
                f"{plan['campaign_root']}/job_ledgers/slurm/"
                "%x_%A_%a.out"
            ),
            (
                "--error="
                f"{plan['campaign_root']}/job_ledgers/slurm/"
                "%x_%A_%a.err"
            ),
            "--cpus-per-task=16",
            "--mem=220G" if plan["resource"] == "gpu" else "--mem=192G",
        ]
        if plan["resource"] == "gpu":
            arguments.append("--gres=gpu:gh200:1")
        arguments.extend(
            [
                (
                    "--export=ALL,"
                    f"RETB_INTERNAL_PHASE_PLAN={phase_plan_path(plan['campaign_root'], controller_id=plan['controller_id'], phase_id=plan['phase_id'])},"
                    f"CAMPAIGN_ROOT={plan['campaign_root']}"
                ),
                str(Path(slurm_task_script).resolve()),
            ]
        )
        completed = subprocess.run(
            arguments,
            cwd=Path(repo_root).resolve(),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"nested Slurm phase failed: {plan['phase_id']}"
            )
    else:
        for index in range(int(plan["row_count"])):
            execute_internal_phase_row(
                plan=plan, task_index=index, repo_root=repo_root
            )
    return publish_internal_phase_completion(plan=plan)


def execute_phased_controller(
    *,
    campaign_root: str | Path,
    controller_id: str,
    phase_ids: Sequence[str],
    phase_builder: Callable[
        [str, int, Mapping[str, str]], Mapping[str, Any]
    ],
    repo_root: str | Path,
    slurm_task_script: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build and execute an ordered, restart-safe internal phase sequence."""

    completions: dict[str, str] = {}
    plans: dict[str, str] = {}
    for index, phase_id in enumerate(phase_ids):
        plan = dict(phase_builder(phase_id, index, dict(completions)))
        validate_internal_phase_plan(plan)
        if (
            plan["controller_id"] != controller_id
            or plan["phase_id"] != phase_id
            or int(plan["sequence_index"]) != index
            or plan["prerequisite_completion_hashes"] != completions
        ):
            raise ValueError("internal phase sequence binding differs")
        path = phase_plan_path(
            campaign_root,
            controller_id=controller_id,
            phase_id=phase_id,
        )
        if path.is_file():
            retained = load_hashed_json(
                path, expected_contract=INTERNAL_PHASE_PLAN_CONTRACT
            )
            if retained != plan:
                raise ValueError(
                    "restarted internal phase plan differs from its lock"
                )
        else:
            write_immutable_json(path, plan)
        completion = execute_internal_phase(
            plan=plan,
            repo_root=repo_root,
            slurm_task_script=slurm_task_script,
        )
        plans[phase_id] = plan["content_hash"]
        completions[phase_id] = completion["content_hash"]
    artifact = with_content_hash(
        {
            "contract": PHASED_CONTROLLER_COMPLETION_CONTRACT,
            "schema_version": 1,
            "controller_id": controller_id,
            "phase_order": list(phase_ids),
            "phase_plan_hashes": plans,
            "phase_completion_hashes": completions,
            "all_phase_outputs_revalidated": True,
            "scientific_result_sign_used_as_continuation_gate": False,
            "source": plan["source"],
        }
    )
    write_immutable_json(output_path, artifact)
    return artifact


__all__ = [
    "INTERNAL_PHASE_COMPLETION_CONTRACT",
    "INTERNAL_PHASE_PLAN_CONTRACT",
    "INTERNAL_PHASE_ROW_COMPLETION_CONTRACT",
    "PHASED_CONTROLLER_COMPLETION_CONTRACT",
    "build_internal_phase_plan",
    "execute_internal_phase",
    "execute_internal_phase_row",
    "execute_phased_controller",
    "phase_completion_path",
    "phase_plan_path",
    "phase_row_completion_path",
    "publish_internal_phase_completion",
    "reusable_internal_phase_completion",
    "reusable_internal_phase_row",
    "validate_internal_phase_plan",
]

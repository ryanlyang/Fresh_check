"""Concrete scientific-task adapters for production runtime handlers.

Factories own cache/model construction; this module owns the locked operation
dispatch, exact registry coverage, artifact authentication, and task-result
publication. A factory cannot replace the scientific operation with a no-op.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import (
    load_hashed_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .registry import validate_particle_view_registry
from .runtime import build_runtime_task_result


PARTICLE_VIEW_SCIENTIFIC_TASK_CATALOG_CONTRACT = (
    "particle_view_scientific_task_catalog_v1"
)

SCIENTIFIC_OPERATIONS = (
    "source_preflight",
    "teacher_training",
    "existing_teacher_registration",
    "direct_control_training",
    "target_discovery",
    "consumer_training",
    "recovery_probe_training",
    "selected_view_publication",
    "pview0_training",
    "residual_sampler_fit",
    "robust_consumer_training",
    "frozen_distillation",
    "joint_finetuning",
    "structural_control_evaluation",
    "configuration_selection",
    "fairness_closure",
    "stack_evaluation",
    "fusion",
    "reporting",
    "bundle_export",
    "bundle_reload",
    "final_test",
)

_CATEGORY_OPERATIONS = {
    "source": {"source_preflight"},
    "baseline": {
        "teacher_training",
        "existing_teacher_registration",
        "direct_control_training",
    },
    "target_generator": {"target_discovery"},
    "consumer_interface": {"target_discovery", "consumer_training"},
    "target_selection": {"configuration_selection"},
    "view_publication": {
        "selected_view_publication",
        "consumer_training",
    },
    "representation": {
        "pview0_training",
        "residual_sampler_fit",
        "robust_consumer_training",
    },
    "predictor_architecture": {"frozen_distillation"},
    "distillation": {"frozen_distillation", "joint_finetuning"},
    "focused_interaction": {"frozen_distillation", "joint_finetuning"},
    "trained_control": {
        "teacher_training",
        "frozen_distillation",
        "joint_finetuning",
    },
    "structural_control": {"structural_control_evaluation"},
    "confirmation_role": {
        "frozen_distillation",
        "joint_finetuning",
        "teacher_training",
    },
    "winner_selection": {"configuration_selection"},
    "fairness_ledger": {"fairness_closure"},
    "fairness_control": {"fairness_closure", "teacher_training"},
    "stack_winner": {"stack_evaluation"},
    "stack_fairness_control": {"stack_evaluation"},
    "stack_static": {"stack_evaluation", "fusion"},
    "report_export": {"reporting", "bundle_export", "bundle_reload"},
    "final_test": {"final_test"},
}


def _category(role: str) -> str:
    category, separator, _ = role.partition(":")
    if not separator:
        raise ValueError("scientific role has no category")
    return category


def build_scientific_task_catalog(
    *,
    registry: Mapping[str, Any],
    task_specs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Bind every registry run to one nonreplaceable scientific operation."""

    validate_particle_view_registry(registry)
    run_by_id = {row["run_id"]: row for row in registry["runs"]}
    if set(task_specs) != set(run_by_id):
        raise ValueError(
            "scientific task coverage mismatch: "
            f"missing={sorted(set(run_by_id) - set(task_specs))}, "
            f"extra={sorted(set(task_specs) - set(run_by_id))}"
        )
    rows = []
    for run_id in sorted(run_by_id):
        spec = task_specs[run_id]
        if set(spec) != {
            "operation",
            "factory",
            "factory_config_path",
            "factory_config_sha256",
        }:
            raise ValueError("scientific task spec field inventory mismatch")
        operation = str(spec["operation"])
        category = _category(run_by_id[run_id]["scientific_role"])
        if operation not in _CATEGORY_OPERATIONS.get(category, set()):
            raise ValueError(
                f"operation {operation} is invalid for category {category}"
            )
        factory = str(spec["factory"])
        if factory.count(":") != 1 or not all(factory.split(":", 1)):
            raise ValueError("factory must use module:function syntax")
        config_path = Path(spec["factory_config_path"]).resolve()
        config_hash = require_sha256(
            "factory_config_sha256",
            spec["factory_config_sha256"],
        )
        if not config_path.is_file() or sha256_file(config_path) != config_hash:
            raise ValueError(f"factory config is absent or stale for {run_id}")
        rows.append(
            {
                "run_id": run_id,
                "category": category,
                "operation": operation,
                "factory": factory,
                "factory_config_path": str(config_path),
                "factory_config_sha256": config_hash,
            }
        )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_SCIENTIFIC_TASK_CATALOG_CONTRACT,
            "registry_sha256": registry["content_hash"],
            "rows": rows,
            "run_count": len(rows),
            "factory_contract": "cache_backed_kwargs_and_artifacts_v1",
            "no_op_success_forbidden": True,
        }
    )


def validate_scientific_task_catalog(
    catalog: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        catalog,
        expected_contract=PARTICLE_VIEW_SCIENTIFIC_TASK_CATALOG_CONTRACT,
    )
    if catalog.get("registry_sha256") != registry["content_hash"]:
        raise ValueError("scientific catalog is bound to another registry")
    rebuilt = build_scientific_task_catalog(
        registry=registry,
        task_specs={
            row["run_id"]: {
                key: row[key]
                for key in (
                    "operation",
                    "factory",
                    "factory_config_path",
                    "factory_config_sha256",
                )
            }
            for row in catalog["rows"]
        },
    )
    if rebuilt != dict(catalog):
        raise ValueError("scientific task catalog is noncanonical")
    return {
        "ok": True,
        "content_hash": catalog["content_hash"],
        "run_count": catalog["run_count"],
    }


def build_scientific_handler_commands(
    *,
    catalog: Mapping[str, Any],
    catalog_path: str,
    python_executable: str = "python",
) -> dict[str, list[str]]:
    """Generate category handlers for the generic scientific-task executor."""

    validate_content_hash(
        catalog,
        expected_contract=PARTICLE_VIEW_SCIENTIFIC_TASK_CATALOG_CONTRACT,
    )
    if not python_executable:
        raise ValueError("python_executable must be nonempty")
    categories = sorted({row["category"] for row in catalog["rows"]})
    path = str(Path(catalog_path).resolve())
    return {
        category: [
            python_executable,
            "-u",
            "scripts/execute_particle_view_scientific_task.py",
            "--catalog",
            path,
            "--registry",
            "{registry_path}",
            "--run-id",
            "{run_id}",
            "--seed",
            "{seed}",
            "--task-id",
            "{task_id}",
            "--output-dir",
            "{task_output_dir}",
        ]
        for category in categories
    }


def _load_factory(path: str) -> Callable[..., Mapping[str, Any]]:
    module_name, function_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError("scientific task factory is not callable")
    return factory


def _operation_callable(operation: str) -> Callable[..., Any] | None:
    if operation == "consumer_training":
        from .consumer_train import train_particle_view_consumer

        return train_particle_view_consumer
    if operation == "existing_teacher_registration":
        from .production_factories import register_existing_teacher_source

        return register_existing_teacher_source
    if operation == "direct_control_training":
        from .direct_control import train_direct_hlt_control

        return train_direct_hlt_control
    if operation == "target_discovery":
        from .target_runtime import run_target_discovery_operation

        return run_target_discovery_operation
    if operation == "recovery_probe_training":
        from .recovery_probe import train_recovery_probe

        return train_recovery_probe
    if operation == "pview0_training":
        from .train_predictor import train_pview0

        return train_pview0
    if operation == "residual_sampler_fit":
        from .robust_consumer import fit_correlated_residual_sampler

        return fit_correlated_residual_sampler
    if operation == "robust_consumer_training":
        from .robust_consumer import train_robust_consumer

        return train_robust_consumer
    if operation == "frozen_distillation":
        from .distillation import train_frozen_consumer_distillation

        return train_frozen_consumer_distillation
    if operation == "joint_finetuning":
        from .distillation import train_joint_finetuning

        return train_joint_finetuning
    return None


def execute_scientific_task(
    *,
    catalog: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Execute one cache-backed factory and publish the runtime task result."""

    validate_scientific_task_catalog(catalog, registry=registry)
    try:
        row = next(item for item in catalog["rows"] if item["run_id"] == run_id)
    except StopIteration as exc:
        raise ValueError("scientific task run_id is absent") from exc
    config_path = Path(row["factory_config_path"])
    if sha256_file(config_path) != row["factory_config_sha256"]:
        raise ValueError("scientific task factory config changed")
    config = load_hashed_json(config_path)
    factory = _load_factory(row["factory"])
    prepared = factory(
        operation=row["operation"],
        config=config,
        registry=registry,
        run_id=run_id,
        seed=int(seed),
        task_id=task_id,
        output_dir=str(Path(output_dir).resolve()),
    )
    if not isinstance(prepared, Mapping) or set(prepared) != {
        "kwargs",
        "artifact_paths",
        "action",
    }:
        raise ValueError(
            "scientific factory must return kwargs/artifact_paths/action"
        )
    operation = row["operation"]
    locked = _operation_callable(operation)
    if locked is not None:
        if prepared["action"] is not None:
            raise ValueError("locked training operation cannot be replaced")
        locked(**dict(prepared["kwargs"]))
    else:
        action = prepared["action"]
        if not callable(action):
            raise ValueError(
                f"operation {operation} requires a callable factory action"
            )
        action(**dict(prepared["kwargs"]))
    artifacts = []
    for raw_path in prepared["artifact_paths"]:
        path = Path(raw_path).resolve()
        artifacts.append({"path": str(path), "sha256": sha256_file(path)})
    result = build_runtime_task_result(
        task_id=task_id,
        artifacts=artifacts,
    )
    destination = Path(output_dir) / "task_result.json"
    write_immutable_json(destination, result)
    return result


__all__ = [
    "PARTICLE_VIEW_SCIENTIFIC_TASK_CATALOG_CONTRACT",
    "SCIENTIFIC_OPERATIONS",
    "build_scientific_handler_commands",
    "build_scientific_task_catalog",
    "execute_scientific_task",
    "validate_scientific_task_catalog",
]

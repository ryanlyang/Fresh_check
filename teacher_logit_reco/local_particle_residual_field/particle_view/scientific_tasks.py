"""Concrete scientific-task adapters for production runtime handlers.

Factories own cache/model construction; this module owns the locked operation
dispatch, exact registry coverage, artifact authentication, and task-result
publication. A factory cannot replace the scientific operation with a no-op.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

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
    "consumer_interface_screen",
    "consumer_training",
    "recovery_probe_training",
    "selected_view_publication",
    "pview0_training",
    "residual_sampler_fit",
    "robust_consumer_training",
    "frozen_distillation",
    "joint_finetuning",
    "focused_composite_training",
    "trained_control_training",
    "structural_control_evaluation",
    "confirmation_training",
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
    "consumer_interface": {"consumer_interface_screen"},
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
    "focused_interaction": {
        "frozen_distillation",
        "joint_finetuning",
        "focused_composite_training",
    },
    "trained_control": {
        "teacher_training",
        "frozen_distillation",
        "joint_finetuning",
        "trained_control_training",
    },
    "structural_control": {"structural_control_evaluation"},
    "confirmation_role": {
        "confirmation_training",
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
    if operation == "teacher_training":
        from .teacher_train import train_particle_view_teacher

        return train_particle_view_teacher
    if operation == "consumer_interface_screen":
        from .consumer_interface_runtime import (
            run_consumer_interface_screen,
        )

        return run_consumer_interface_screen
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
        from .post_target_runtime import run_residual_sampler_fit

        return run_residual_sampler_fit
    if operation == "robust_consumer_training":
        from .robust_consumer import train_robust_consumer

        return train_robust_consumer
    if operation == "frozen_distillation":
        from .distillation import train_frozen_consumer_distillation

        return train_frozen_consumer_distillation
    if operation == "joint_finetuning":
        from .distillation import train_joint_finetuning

        return train_joint_finetuning
    if operation == "focused_composite_training":
        from .focused_control_runtime import (
            run_focused_composite_training,
        )

        return run_focused_composite_training
    if operation == "trained_control_training":
        from .focused_control_runtime import run_trained_control_training

        return run_trained_control_training
    if operation == "structural_control_evaluation":
        from .confirmation_runtime import (
            run_structural_control_evaluation,
        )

        return run_structural_control_evaluation
    if operation == "confirmation_training":
        from .confirmation_runtime import run_confirmation_training

        return run_confirmation_training
    if operation == "stack_evaluation":
        from .stack_runtime import run_stack_evaluation

        return run_stack_evaluation
    if operation == "fusion":
        from .stack_runtime import run_stack_fusion

        return run_stack_fusion
    if operation == "bundle_export":
        from .report_runtime import run_report_bundle_export

        return run_report_bundle_export
    if operation == "bundle_reload":
        from .report_runtime import run_report_bundle_reload

        return run_report_bundle_reload
    if operation == "reporting":
        # Both the aggregate report and family permit are locked PV09
        # operations.  The factory supplies the exact callable-shaped kwargs;
        # dispatch on their authenticated field inventory rather than allowing
        # an arbitrary factory action.
        from .report_runtime import (
            run_report_aggregate,
            run_report_final_permit,
        )

        def run_locked_reporting(**kwargs):
            if "permit_family" in kwargs:
                return run_report_final_permit(**kwargs)
            return run_report_aggregate(**kwargs)

        return run_locked_reporting
    if operation == "final_test":
        from .final_runtime import run_final_test_campaign

        return run_final_test_campaign
    return None


def _warning_candidates(value: Any, *, key: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        code = value.get("warning_code")
        if isinstance(code, str) and code.startswith("WARN_"):
            rows.append(dict(value))
        for child_key, child in value.items():
            if (
                child_key in {
                    "warning",
                    "warnings",
                    "quality_warning",
                    "quality_warnings",
                    "scientific_warning",
                    "scientific_warnings",
                }
                or isinstance(child, (Mapping, list, tuple))
            ):
                rows.extend(_warning_candidates(child, key=str(child_key)))
    elif isinstance(value, (list, tuple)):
        for child in value:
            rows.extend(_warning_candidates(child, key=key))
    elif isinstance(value, str) and value.startswith("WARN_"):
        rows.append({"warning_code": value})
    return rows


def _published_path_string(value: str) -> str:
    """Map a task-transaction path to its immutable published location."""

    raw_attempt = os.environ.get("PARTICLE_VIEW_TASK_OUTPUT_DIR")
    raw_final = os.environ.get("PARTICLE_VIEW_TASK_FINAL_OUTPUT_DIR")
    if not raw_attempt or not raw_final:
        return value
    attempt = str(Path(raw_attempt).resolve())
    final = str(Path(raw_final).resolve())
    if value == attempt:
        return final
    normalized = value.replace("\\", "/")
    normalized_attempt = attempt.replace("\\", "/").rstrip("/")
    prefix = normalized_attempt + "/"
    if normalized.startswith(prefix):
        return str(Path(final) / Path(normalized[len(prefix) :]))
    return value


def _replace_strings(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _replace_strings(child, replacements)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_replace_strings(child, replacements) for child in value]
    if isinstance(value, str):
        return _published_path_string(replacements.get(value, value))
    return value


def _encoded_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _replace_file(path: Path, encoded: bytes) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.canonical.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonicalize_published_json_artifacts(
    artifact_paths: Sequence[str | Path],
) -> None:
    """Rewrite transaction paths and same-task hashes before publication."""

    documents: dict[Path, dict[str, Any]] = {}
    jsonl_paths: list[Path] = []
    old_identifiers: dict[str, Path] = {}
    for raw_path in artifact_paths:
        path = Path(raw_path).resolve()
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            documents[path] = payload
            old_identifiers[sha256_file(path)] = path
            content_hash = payload.get("content_hash")
            if isinstance(content_hash, str) and len(content_hash) == 64:
                old_identifiers[content_hash] = path
        elif path.suffix == ".jsonl":
            jsonl_paths.append(path)

    visiting: set[Path] = set()
    complete: set[Path] = set()
    replacements: dict[str, str] = {}

    def identifiers(value: Any) -> set[str]:
        if isinstance(value, Mapping):
            return {
                item
                for child in value.values()
                for item in identifiers(child)
            }
        if isinstance(value, (list, tuple)):
            return {item for child in value for item in identifiers(child)}
        return {value} if isinstance(value, str) else set()

    def publish(path: Path) -> None:
        if path in complete:
            return
        if path in visiting:
            raise ValueError("task JSON artifacts contain cyclic hash lineage")
        visiting.add(path)
        payload = documents[path]
        for identifier in identifiers(payload):
            dependency = old_identifiers.get(identifier)
            if dependency is not None and dependency != path:
                publish(dependency)
        updated = _replace_strings(payload, replacements)
        old_content_hash = payload.get("content_hash")
        if old_content_hash is not None:
            unhashed = dict(updated)
            unhashed.pop("content_hash", None)
            updated = with_content_hash(unhashed)
        old_file_hash = sha256_file(path)
        _replace_file(path, _encoded_json(updated))
        if isinstance(old_content_hash, str):
            replacements[old_content_hash] = str(updated["content_hash"])
        replacements[old_file_hash] = sha256_file(path)
        documents[path] = updated
        visiting.remove(path)
        complete.add(path)

    for document_path in documents:
        publish(document_path)

    for path in jsonl_paths:
        rows = [
            _replace_strings(json.loads(line), replacements)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _replace_file(
            path,
            b"".join(
                json.dumps(
                    row,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
                for row in rows
            ),
        )


def _normalized_warning_severity(value: Any) -> str:
    normalized = str(value if value is not None else "warning").strip().lower()
    normalized = {
        "scientific": "warning",
        "scientific_warning": "warning",
        "warn": "warning",
        "error": "high",
        "critical": "high",
    }.get(normalized, normalized)
    return normalized if normalized in {"info", "warning", "high"} else "warning"


def _publish_task_quality_warnings(
    *,
    artifact_rows: list[dict[str, str]],
    output_dir: Path,
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    source_commit: str,
) -> tuple[Path, list[str]] | None:
    from .production import (
        LOGICAL_NODE_LAYOUT,
        build_quality_warning,
        write_quality_warning_jsonl,
    )

    stage = next(
        row["stage"] for row in registry["runs"] if row["run_id"] == run_id
    )
    graph_node = next(
        node_id
        for node_id, stages, _ in LOGICAL_NODE_LAYOUT
        if stage in stages
    )
    warnings = []
    seen = set()
    for artifact in artifact_rows:
        path = Path(artifact["path"])
        if path.suffix not in {".json", ".jsonl"} or path.name == "task_result.json":
            continue
        try:
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                candidates = _warning_candidates(payload)
            else:
                candidates = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        candidates.extend(_warning_candidates(json.loads(line)))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for candidate in candidates:
            code = str(candidate["warning_code"])
            identity = (code, artifact["sha256"])
            if identity in seen:
                continue
            seen.add(identity)
            warnings.append(
                build_quality_warning(
                    warning_code=code,
                    severity=_normalized_warning_severity(
                        candidate.get("severity", "warning")
                    ),
                    graph_node=graph_node,
                    configuration_id=str(
                        candidate.get("configuration_id", run_id)
                    ),
                    seed=int(candidate.get("seed", seed)),
                    split=str(
                        candidate.get(
                            "split",
                            candidate.get("selection_split", "not_applicable"),
                        )
                    ),
                    observed_value=candidate.get("observed_value"),
                    reference_value=candidate.get("reference_value"),
                    warning_threshold=candidate.get(
                        "warning_threshold",
                        candidate.get("declared_warning_threshold"),
                    ),
                    interpretation=str(
                        candidate.get(
                            "interpretation",
                            f"{code} emitted by {path.name}",
                        )
                    ),
                    suggested_diagnostic=str(
                        candidate.get(
                            "suggested_diagnostic",
                            "Inspect the bound supporting artifact.",
                        )
                    ),
                    supporting_artifacts=[
                        {
                            **artifact,
                            "path": _published_path_string(artifact["path"]),
                        }
                    ],
                    source_commit=source_commit,
                )
            )
    if not warnings:
        return None
    path = output_dir / "quality_warnings.jsonl"
    write_quality_warning_jsonl(path, warnings)
    return path, [row["content_hash"] for row in warnings]


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
    _canonicalize_published_json_artifacts(prepared["artifact_paths"])
    artifacts = []
    for raw_path in prepared["artifact_paths"]:
        path = Path(raw_path).resolve()
        artifacts.append({"path": str(path), "sha256": sha256_file(path)})
    source_commit = str(config.get("source_commit", "0" * 40))
    graph_path = os.environ.get("PARTICLE_VIEW_GRAPH")
    if graph_path:
        graph = load_hashed_json(graph_path)
        source_commit = str(graph["source_commit"])
    warning_publication = _publish_task_quality_warnings(
        artifact_rows=artifacts,
        output_dir=Path(output_dir),
        registry=registry,
        run_id=run_id,
        seed=int(seed),
        source_commit=source_commit,
    )
    warning_hashes: list[str] = []
    if warning_publication is not None:
        warning_path, warning_hashes = warning_publication
        artifacts.append(
            {
                "path": str(warning_path.resolve()),
                "sha256": sha256_file(warning_path),
            }
        )
    result = build_runtime_task_result(
        task_id=task_id,
        artifacts=artifacts,
        warning_sha256=warning_hashes,
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

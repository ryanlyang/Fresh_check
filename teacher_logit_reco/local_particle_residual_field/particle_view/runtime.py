"""Seed-expanded, resumable runtime execution for the low-data campaign."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .production import LOGICAL_NODE_LAYOUT
from .registry import validate_particle_view_registry


PARTICLE_VIEW_RUNTIME_HANDLER_CATALOG_CONTRACT = (
    "particle_view_runtime_handler_catalog_v1"
)
PARTICLE_VIEW_RUNTIME_EXECUTION_MANIFEST_CONTRACT = (
    "particle_view_runtime_execution_manifest_v1"
)
PARTICLE_VIEW_RUNTIME_TASK_CONTRACT = "particle_view_runtime_task_v1"
PARTICLE_VIEW_RUNTIME_TASK_RESULT_CONTRACT = (
    "particle_view_runtime_task_result_v1"
)
PARTICLE_VIEW_RUNTIME_TASK_COMPLETION_CONTRACT = (
    "particle_view_runtime_task_completion_v1"
)
PARTICLE_VIEW_RUNTIME_NODE_REPORT_CONTRACT = (
    "particle_view_runtime_node_report_v1"
)

_PLACEHOLDERS = {
    "run_id",
    "seed",
    "task_id",
    "task_output_dir",
    "artifact_root",
    "registry_path",
    "scientific_role",
}
_REQUIRED_TEMPLATE_PLACEHOLDERS = {
    "run_id",
    "seed",
    "task_output_dir",
}
_TOKEN_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def _role_category(role: str) -> str:
    if not isinstance(role, str) or ":" not in role:
        raise ValueError("scientific_role must contain a category prefix")
    category, detail = role.split(":", 1)
    if not category or not detail:
        raise ValueError("scientific_role category/detail must be nonempty")
    return category


def _stage_node_map() -> dict[str, str]:
    result = {}
    for node_id, stages, _ in LOGICAL_NODE_LAYOUT:
        for stage in stages:
            if stage in result:
                raise RuntimeError("logical stage is assigned to multiple nodes")
            result[stage] = node_id
    return result


def _topological_runs(
    runs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    result: list[str] = []
    state: dict[str, int] = {}

    def visit(run_id: str) -> None:
        marker = state.get(run_id, 0)
        if marker == 1:
            raise ValueError(f"runtime registry cycle at {run_id}")
        if marker == 2:
            return
        state[run_id] = 1
        for parent in runs[run_id]["parent_run_ids"]:
            visit(parent)
        state[run_id] = 2
        result.append(run_id)

    for run_id in sorted(runs):
        visit(run_id)
    return result


def build_runtime_handler_catalog(
    handler_commands: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Freeze category-level argv templates used to generate every task."""

    if not handler_commands:
        raise ValueError("runtime handler catalog cannot be empty")
    handlers = []
    for category in sorted(handler_commands):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", category):
            raise ValueError(f"invalid runtime handler category {category!r}")
        command = list(handler_commands[category])
        if not command or any(
            not isinstance(token, str) or not token for token in command
        ):
            raise ValueError(f"handler {category} has an invalid command")
        observed = {
            match.group(1)
            for token in command
            for match in _TOKEN_PLACEHOLDER.finditer(token)
        }
        unknown = observed - _PLACEHOLDERS
        missing = _REQUIRED_TEMPLATE_PLACEHOLDERS - observed
        if unknown or missing:
            raise ValueError(
                f"handler {category} placeholder mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        handlers.append(
            {
                "category": category,
                "command_template": command,
            }
        )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_RUNTIME_HANDLER_CATALOG_CONTRACT,
            "handlers": handlers,
            "required_result_contract": (
                PARTICLE_VIEW_RUNTIME_TASK_RESULT_CONTRACT
            ),
            "shell_execution": False,
        }
    )


def validate_runtime_handler_catalog(
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        catalog,
        expected_contract=PARTICLE_VIEW_RUNTIME_HANDLER_CATALOG_CONTRACT,
    )
    if set(catalog) != {
        "contract",
        "handlers",
        "required_result_contract",
        "shell_execution",
        "content_hash",
    }:
        raise ValueError("runtime handler catalog field inventory mismatch")
    if (
        catalog["required_result_contract"]
        != PARTICLE_VIEW_RUNTIME_TASK_RESULT_CONTRACT
        or catalog["shell_execution"] is not False
    ):
        raise ValueError("runtime handler execution contract changed")
    rebuilt = build_runtime_handler_catalog(
        {
            row["category"]: row["command_template"]
            for row in catalog["handlers"]
        }
    )
    if rebuilt != dict(catalog):
        raise ValueError("runtime handler catalog is noncanonical")
    return {
        "ok": True,
        "content_hash": catalog["content_hash"],
        "categories": [
            row["category"] for row in catalog["handlers"]
        ],
    }


def _render_command(
    template: Sequence[str],
    values: Mapping[str, str],
) -> list[str]:
    rendered = []
    for token in template:
        unknown = {
            match.group(1)
            for match in _TOKEN_PLACEHOLDER.finditer(token)
        } - set(values)
        if unknown:
            raise ValueError(
                f"runtime command has unresolved placeholders {sorted(unknown)}"
            )
        value = token
        for key, replacement in values.items():
            value = value.replace("{" + key + "}", replacement)
        if _TOKEN_PLACEHOLDER.search(value):
            raise ValueError("runtime command retains an unresolved placeholder")
        rendered.append(value)
    return rendered


def build_runtime_execution_manifest(
    *,
    registry: Mapping[str, Any],
    registry_path: str,
    handler_catalog: Mapping[str, Any],
    handler_catalog_path: str,
    artifact_root: str,
) -> dict[str, Any]:
    """Expand every registry role and seed into an authenticated task."""

    registry_audit = validate_particle_view_registry(registry)
    handler_audit = validate_runtime_handler_catalog(handler_catalog)
    runs = {row["run_id"]: row for row in registry["runs"]}
    categories = {
        _role_category(row["scientific_role"]) for row in runs.values()
    }
    handler_categories = set(handler_audit["categories"])
    if categories != handler_categories:
        raise ValueError(
            "runtime handler coverage mismatch: "
            f"missing={sorted(categories - handler_categories)}, "
            f"extra={sorted(handler_categories - categories)}"
        )
    handlers = {
        row["category"]: row["command_template"]
        for row in handler_catalog["handlers"]
    }
    root = Path(artifact_root).resolve()
    stage_nodes = _stage_node_map()
    tasks = []
    task_ids: dict[tuple[str, int], str] = {}
    for run_id in _topological_runs(runs):
        run = runs[run_id]
        for seed in run["seed_ids"]:
            task_ids[(run_id, int(seed))] = f"{run_id}__seed_{int(seed)}"
    for run_id in _topological_runs(runs):
        run = runs[run_id]
        category = _role_category(run["scientific_role"])
        for seed_value in run["seed_ids"]:
            seed = int(seed_value)
            task_id = task_ids[(run_id, seed)]
            parent_task_ids = []
            for parent_run_id in run["parent_run_ids"]:
                parent_seeds = [int(value) for value in runs[parent_run_id]["seed_ids"]]
                parent_seed = seed if seed in parent_seeds else 101
                if parent_seed not in parent_seeds:
                    raise ValueError(
                        f"run {run_id} cannot bind seed {seed} to "
                        f"parent {parent_run_id}"
                    )
                parent_task_ids.append(task_ids[(parent_run_id, parent_seed)])
            output_dir = root / "runtime_tasks" / task_id
            values = {
                "run_id": run_id,
                "seed": str(seed),
                "task_id": task_id,
                "task_output_dir": str(output_dir),
                "artifact_root": str(root),
                "registry_path": str(Path(registry_path).resolve()),
                "scientific_role": str(run["scientific_role"]),
            }
            command = _render_command(handlers[category], values)
            tasks.append(
                {
                    "contract": PARTICLE_VIEW_RUNTIME_TASK_CONTRACT,
                    "task_id": task_id,
                    "run_id": run_id,
                    "seed": seed,
                    "stage": run["stage"],
                    "node_id": stage_nodes[run["stage"]],
                    "scientific_role": run["scientific_role"],
                    "category": category,
                    "parent_task_ids": parent_task_ids,
                    "command": command,
                    "command_sha256": canonical_sha256(command),
                    "output_dir": str(output_dir),
                    "result_path": str(output_dir / "task_result.json"),
                    "completion_path": str(
                        root / "runtime_completions" / f"{task_id}.json"
                    ),
                    "quality_policy": run["quality_policy"],
                }
            )
    manifest = with_content_hash(
        {
            "contract": PARTICLE_VIEW_RUNTIME_EXECUTION_MANIFEST_CONTRACT,
            "registry_sha256": registry["content_hash"],
            "registry_path": str(Path(registry_path).resolve()),
            "handler_catalog_sha256": handler_catalog["content_hash"],
            "handler_catalog_path": str(Path(handler_catalog_path).resolve()),
            "artifact_root": str(root),
            "tasks": tasks,
            "declared_run_count": registry_audit["run_count"],
            "seed_expanded_task_count": len(tasks),
            "quality_gates": False,
            "resume_policy": "hash_authenticated_task_completion_v1",
        }
    )
    validate_runtime_execution_manifest(manifest, registry=registry)
    return manifest


def validate_runtime_execution_manifest(
    manifest: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_content_hash(
        manifest,
        expected_contract=PARTICLE_VIEW_RUNTIME_EXECUTION_MANIFEST_CONTRACT,
    )
    expected = {
        "contract",
        "registry_sha256",
        "registry_path",
        "handler_catalog_sha256",
        "handler_catalog_path",
        "artifact_root",
        "tasks",
        "declared_run_count",
        "seed_expanded_task_count",
        "quality_gates",
        "resume_policy",
        "content_hash",
    }
    if set(manifest) != expected:
        raise ValueError("runtime execution manifest field inventory mismatch")
    require_sha256("registry_sha256", manifest["registry_sha256"])
    require_sha256(
        "handler_catalog_sha256", manifest["handler_catalog_sha256"]
    )
    if (
        manifest["quality_gates"] is not False
        or manifest["resume_policy"]
        != "hash_authenticated_task_completion_v1"
    ):
        raise ValueError("runtime quality/resume policy changed")
    raw_tasks = manifest["tasks"]
    if (
        not isinstance(raw_tasks, list)
        or len(raw_tasks) != manifest["seed_expanded_task_count"]
    ):
        raise ValueError("runtime task count mismatch")
    tasks: dict[str, Mapping[str, Any]] = {}
    for task in raw_tasks:
        if set(task) != {
            "contract",
            "task_id",
            "run_id",
            "seed",
            "stage",
            "node_id",
            "scientific_role",
            "category",
            "parent_task_ids",
            "command",
            "command_sha256",
            "output_dir",
            "result_path",
            "completion_path",
            "quality_policy",
        }:
            raise ValueError("runtime task field inventory mismatch")
        if task["contract"] != PARTICLE_VIEW_RUNTIME_TASK_CONTRACT:
            raise ValueError("runtime task contract mismatch")
        task_id = task["task_id"]
        if task_id in tasks:
            raise ValueError("duplicate runtime task ID")
        if task["seed"] not in {101, 202, 303}:
            raise ValueError("runtime task seed is invalid")
        if task["command_sha256"] != canonical_sha256(task["command"]):
            raise ValueError("runtime task command hash mismatch")
        if task["category"] != _role_category(task["scientific_role"]):
            raise ValueError("runtime task category mismatch")
        tasks[task_id] = task
    for task in raw_tasks:
        for parent in task["parent_task_ids"]:
            if parent not in tasks:
                raise ValueError("runtime task references unknown parent")
    _topological_runs(
        {
            task_id: {"parent_run_ids": task["parent_task_ids"]}
            for task_id, task in tasks.items()
        }
    )
    if registry is not None:
        registry_audit = validate_particle_view_registry(registry)
        if manifest["registry_sha256"] != registry["content_hash"]:
            raise ValueError("runtime manifest is bound to another registry")
        expected_tasks = sum(
            len(row["seed_ids"]) for row in registry["runs"]
        )
        if (
            manifest["declared_run_count"] != registry_audit["run_count"]
            or manifest["seed_expanded_task_count"] != expected_tasks
        ):
            raise ValueError("runtime manifest/registry count mismatch")
        observed = {
            (task["run_id"], int(task["seed"])) for task in raw_tasks
        }
        expected_pairs = {
            (row["run_id"], int(seed))
            for row in registry["runs"]
            for seed in row["seed_ids"]
        }
        if observed != expected_pairs:
            raise ValueError("runtime manifest omits or adds registry replicas")
    node_counts = Counter(task["node_id"] for task in raw_tasks)
    return {
        "ok": True,
        "content_hash": manifest["content_hash"],
        "task_count": len(tasks),
        "node_task_counts": {
            key: node_counts[key] for key in sorted(node_counts)
        },
        "tasks": tasks,
    }


def build_runtime_command_catalog(
    *,
    execution_manifest_path: str,
    python_executable: str = "python",
    execution_manifest_sha256: str | None = None,
) -> dict[str, list[str]]:
    """Generate the 11 graph-node commands; no hand-authored catalog remains."""

    if not python_executable:
        raise ValueError("python_executable must be nonempty")
    manifest_path = str(Path(execution_manifest_path).resolve())
    if execution_manifest_sha256 is None and Path(manifest_path).is_file():
        manifest = load_hashed_json(manifest_path)
        validate_runtime_execution_manifest(manifest)
        execution_manifest_sha256 = manifest["content_hash"]
    if execution_manifest_sha256 is not None:
        require_sha256(
            "execution_manifest_sha256", execution_manifest_sha256
        )
    manifest_binding = (
        [
            "--expected-manifest-sha256",
            execution_manifest_sha256,
        ]
        if execution_manifest_sha256 is not None
        else []
    )
    return {
        node_id: [
            python_executable,
            "-u",
            "scripts/run_particle_view_campaign_node.py",
            "--execution-manifest",
            manifest_path,
            *manifest_binding,
            "--node-id",
            node_id,
        ]
        for node_id, _, _ in LOGICAL_NODE_LAYOUT
    }


def build_runtime_task_result(
    *,
    task_id: str,
    artifacts: Sequence[Mapping[str, str]],
    warning_sha256: Sequence[str] = (),
) -> dict[str, Any]:
    normalized = []
    for row in artifacts:
        path = Path(str(row["path"])).resolve()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"runtime task artifact is absent: {path}")
        observed = sha256_file(path)
        declared = require_sha256("artifact sha256", row["sha256"])
        if observed != declared:
            raise ValueError("runtime task artifact hash mismatch")
        normalized.append({"path": str(path), "sha256": declared})
    if not normalized:
        raise ValueError("runtime task result requires at least one artifact")
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_RUNTIME_TASK_RESULT_CONTRACT,
            "task_id": task_id,
            "artifacts": normalized,
            "warning_sha256": [
                require_sha256("warning_sha256", value)
                for value in warning_sha256
            ],
            "scientific_warnings_non_gating": True,
            "status": "complete",
        }
    )


def validate_runtime_task_result(
    result: Mapping[str, Any],
    *,
    expected_task_id: str,
) -> dict[str, Any]:
    validate_content_hash(
        result,
        expected_contract=PARTICLE_VIEW_RUNTIME_TASK_RESULT_CONTRACT,
    )
    if result.get("task_id") != expected_task_id:
        raise ValueError("runtime result belongs to another task")
    if (
        result.get("status") != "complete"
        or result.get("scientific_warnings_non_gating") is not True
    ):
        raise ValueError("runtime task result is incomplete or gating")
    if not result.get("artifacts"):
        raise ValueError("runtime task result has no artifacts")
    for row in result["artifacts"]:
        path = Path(row["path"])
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("runtime result artifact is absent or unsafe")
        if sha256_file(path) != require_sha256(
            "artifact sha256", row["sha256"]
        ):
            raise ValueError("runtime result artifact changed")
    for value in result["warning_sha256"]:
        require_sha256("warning_sha256", value)
    return {
        "ok": True,
        "content_hash": result["content_hash"],
        "artifact_count": len(result["artifacts"]),
        "warning_count": len(result["warning_sha256"]),
    }


def _reject_transaction_path_leakage(
    *,
    attempt_dir: Path,
    artifact_rows: Sequence[Mapping[str, str]],
) -> None:
    """Reject metadata that would point at the renamed attempt directory."""

    attempt_text = str(attempt_dir.resolve())
    for row in artifact_rows:
        path = Path(row["path"]).resolve()
        if path.suffix not in {".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8")
        if attempt_text in text or ".attempt_" in text:
            raise ValueError(
                f"transaction-attempt path leaked into published metadata: {path}"
            )


def _build_task_completion(
    *,
    manifest: Mapping[str, Any],
    task: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_RUNTIME_TASK_COMPLETION_CONTRACT,
            "execution_manifest_sha256": manifest["content_hash"],
            "registry_sha256": manifest["registry_sha256"],
            "task_id": task["task_id"],
            "run_id": task["run_id"],
            "seed": task["seed"],
            "node_id": task["node_id"],
            "command_sha256": task["command_sha256"],
            "result_sha256": result["content_hash"],
            "artifact_sha256": [
                row["sha256"] for row in result["artifacts"]
            ],
            "warning_sha256": list(result["warning_sha256"]),
            "status": "complete",
            "exit_code": 0,
        }
    )


def _load_valid_completion(
    *,
    manifest: Mapping[str, Any],
    task: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    path = Path(task["completion_path"])
    if not path.exists():
        return None
    completion = load_hashed_json(path)
    validate_content_hash(
        completion,
        expected_contract=PARTICLE_VIEW_RUNTIME_TASK_COMPLETION_CONTRACT,
    )
    expected = {
        "execution_manifest_sha256": manifest["content_hash"],
        "registry_sha256": manifest["registry_sha256"],
        "task_id": task["task_id"],
        "run_id": task["run_id"],
        "seed": task["seed"],
        "node_id": task["node_id"],
        "command_sha256": task["command_sha256"],
        "status": "complete",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(
                f"stale runtime completion for {task['task_id']}: {key}"
            )
    result = load_hashed_json(task["result_path"])
    validate_runtime_task_result(
        result,
        expected_task_id=task["task_id"],
    )
    if completion["result_sha256"] != result["content_hash"]:
        raise ValueError("runtime completion/result hash mismatch")
    return completion


def execute_runtime_node(
    *,
    manifest: Mapping[str, Any],
    node_id: str,
    task_ids: Sequence[str] | None = None,
    dry_run: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    """Run or resume one node while preserving independent work on failure."""

    audit = validate_runtime_execution_manifest(manifest)
    if node_id not in {row[0] for row in LOGICAL_NODE_LAYOUT}:
        raise ValueError("unknown runtime node")
    ordered = [
        task
        for task in manifest["tasks"]
        if task["node_id"] == node_id
    ]
    if task_ids is not None:
        requested = list(task_ids)
        if (
            not requested
            or len(requested) != len(set(requested))
            or any(
                task_id not in {row["task_id"] for row in ordered}
                for task_id in requested
            )
        ):
            raise ValueError("runtime task filter is empty, duplicate, or cross-node")
        requested_set = set(requested)
        ordered = [
            task for task in ordered if task["task_id"] in requested_set
        ]
    task_by_id = audit["tasks"]
    records = []
    failed_or_blocked: set[str] = set()
    for task in ordered:
        completion = _load_valid_completion(manifest=manifest, task=task)
        if completion is not None:
            records.append(
                {
                    "task_id": task["task_id"],
                    "action": "reuse_complete",
                    "exit_code": 0,
                    "completion_sha256": completion["content_hash"],
                }
            )
            continue
        result_path = Path(task["result_path"])
        if result_path.is_file():
            try:
                recovered_result = load_hashed_json(result_path)
                validate_runtime_task_result(
                    recovered_result,
                    expected_task_id=task["task_id"],
                )
                completion = _build_task_completion(
                    manifest=manifest,
                    task=task,
                    result=recovered_result,
                )
                write_immutable_json(task["completion_path"], completion)
                records.append(
                    {
                        "task_id": task["task_id"],
                        "action": "recover_published_result",
                        "exit_code": 0,
                        "completion_sha256": completion["content_hash"],
                    }
                )
                continue
            except Exception:
                # An incomplete historical attempt must never be written into
                # on retry. Preserve it under a campaign-scoped quarantine.
                pass
        blocking = [
            parent
            for parent in task["parent_task_ids"]
            if parent in failed_or_blocked
        ]
        if blocking:
            failed_or_blocked.add(task["task_id"])
            records.append(
                {
                    "task_id": task["task_id"],
                    "action": "blocked_by_failed_parent",
                    "exit_code": None,
                    "blocking_parent_task_ids": blocking,
                }
            )
            continue
        missing_parents = []
        for parent_id in task["parent_task_ids"]:
            parent = task_by_id[parent_id]
            if _load_valid_completion(manifest=manifest, task=parent) is None:
                missing_parents.append(parent_id)
        if missing_parents and not dry_run:
            failed_or_blocked.add(task["task_id"])
            records.append(
                {
                    "task_id": task["task_id"],
                    "action": "missing_parent_completion",
                    "exit_code": None,
                    "missing_parent_task_ids": missing_parents,
                }
            )
            continue
        if dry_run:
            records.append(
                {
                    "task_id": task["task_id"],
                    "action": "would_execute",
                    "exit_code": None,
                    "command": list(task["command"]),
                }
            )
            continue
        output_dir = Path(task["output_dir"]).resolve()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            if output_dir.is_symlink() or not output_dir.is_dir():
                raise ValueError("runtime task output is unsafe")
            quarantine_root = (
                Path(manifest["artifact_root"]).resolve()
                / "failed_task_attempts"
            )
            quarantine_root.mkdir(parents=True, exist_ok=True)
            quarantine = quarantine_root / (
                f"{task['task_id']}__{time.time_ns()}"
            )
            output_dir.replace(quarantine)
        attempt_dir = output_dir.parent / (
            f".{output_dir.name}.attempt_{time.time_ns()}"
        )
        attempt_dir.mkdir(parents=False, exist_ok=False)
        task_command = [
            str(attempt_dir) if token == str(output_dir) else token
            for token in task["command"]
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "PARTICLE_VIEW_TASK_ID": task["task_id"],
                "PARTICLE_VIEW_RUN_ID": task["run_id"],
                "PARTICLE_VIEW_SEED": str(task["seed"]),
                "PARTICLE_VIEW_TASK_OUTPUT_DIR": str(attempt_dir),
                "PARTICLE_VIEW_TASK_FINAL_OUTPUT_DIR": str(output_dir),
                "PARTICLE_VIEW_EXECUTION_MANIFEST_SHA256": manifest[
                    "content_hash"
                ],
                "PYTHONNOUSERSITE": "1",
            }
        )
        completed = runner(
            task_command,
            check=False,
            env=environment,
        )
        returncode = int(completed.returncode)
        if returncode != 0:
            failed_or_blocked.add(task["task_id"])
            records.append(
                {
                    "task_id": task["task_id"],
                    "action": "failed",
                    "exit_code": returncode,
                    "attempt_dir": str(attempt_dir),
                }
            )
            continue
        try:
            attempt_result_path = attempt_dir / "task_result.json"
            result = load_hashed_json(attempt_result_path)
            validate_runtime_task_result(
                result,
                expected_task_id=task["task_id"],
            )
            _reject_transaction_path_leakage(
                attempt_dir=attempt_dir,
                artifact_rows=result["artifacts"],
            )
            mapped_artifacts = []
            for artifact in result["artifacts"]:
                artifact_path = Path(artifact["path"]).resolve()
                try:
                    relative = artifact_path.relative_to(attempt_dir)
                except ValueError:
                    mapped_path = artifact_path
                else:
                    mapped_path = output_dir / relative
                mapped_artifacts.append(
                    {"path": str(mapped_path), "sha256": artifact["sha256"]}
                )
            attempt_result_path.unlink()
            attempt_dir.replace(output_dir)
            result = build_runtime_task_result(
                task_id=task["task_id"],
                artifacts=mapped_artifacts,
                warning_sha256=result["warning_sha256"],
            )
            write_immutable_json(task["result_path"], result)
            completion = _build_task_completion(
                manifest=manifest,
                task=task,
                result=result,
            )
            write_immutable_json(task["completion_path"], completion)
        except Exception as exc:
            failed_or_blocked.add(task["task_id"])
            records.append(
                {
                    "task_id": task["task_id"],
                    "action": "invalid_result",
                    "exit_code": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                    "attempt_dir": str(attempt_dir),
                }
            )
            continue
        records.append(
            {
                "task_id": task["task_id"],
                "action": "completed",
                "exit_code": 0,
                "completion_sha256": completion["content_hash"],
            }
        )
    failed_count = sum(
        row["action"]
        in {"failed", "invalid_result", "missing_parent_completion"}
        for row in records
    )
    blocked_count = sum(
        row["action"] == "blocked_by_failed_parent" for row in records
    )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_RUNTIME_NODE_REPORT_CONTRACT,
            "execution_manifest_sha256": manifest["content_hash"],
            "node_id": node_id,
            "task_filter": (
                None if task_ids is None else list(task_ids)
            ),
            "dry_run": bool(dry_run),
            "records": records,
            "task_count": len(ordered),
            "completed_count": sum(
                row["action"] in {"completed", "reuse_complete"}
                or row["action"] == "recover_published_result"
                for row in records
            ),
            "failed_count": failed_count,
            "blocked_count": blocked_count,
            "exit_code": 1 if failed_count or blocked_count else 0,
            "scientific_warnings_are_non_gating": True,
        }
    )


__all__ = [
    "PARTICLE_VIEW_RUNTIME_EXECUTION_MANIFEST_CONTRACT",
    "PARTICLE_VIEW_RUNTIME_HANDLER_CATALOG_CONTRACT",
    "PARTICLE_VIEW_RUNTIME_NODE_REPORT_CONTRACT",
    "PARTICLE_VIEW_RUNTIME_TASK_COMPLETION_CONTRACT",
    "PARTICLE_VIEW_RUNTIME_TASK_CONTRACT",
    "PARTICLE_VIEW_RUNTIME_TASK_RESULT_CONTRACT",
    "build_runtime_command_catalog",
    "build_runtime_execution_manifest",
    "build_runtime_handler_catalog",
    "build_runtime_task_result",
    "execute_runtime_node",
    "validate_runtime_execution_manifest",
    "validate_runtime_handler_catalog",
    "validate_runtime_task_result",
]

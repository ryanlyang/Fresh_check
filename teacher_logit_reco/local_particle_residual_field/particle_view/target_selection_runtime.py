"""Production target ranking/forwarding for the Stage-B screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .campaign import CONSUMER_SCREEN_IDS, TARGET_SCREEN_IDS
from .consumer_interface_runtime import (
    PARTICLE_VIEW_CONSUMER_SCREEN_METRICS_CONTRACT,
    select_consumer_interface,
)
from .contracts import (
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .oracle_discovery import (
    PARTICLE_VIEW_TARGET_METRICS_CONTRACT,
    build_scientific_warnings,
    select_target_candidates,
    write_scientific_warnings,
)
from .registry import validate_particle_view_registry
from .runtime_data import resolve_parent_task_artifacts
from .target_runtime import (
    CANONICAL_TARGET_DISCOVERY_RUN_ID,
    PARTICLE_VIEW_TARGET_UNAVAILABLE_CONTRACT,
    TARGET_SCREEN_FORWARD_COUNT,
)


PARTICLE_VIEW_TARGET_SELECTION_FACTORY_CONFIG_CONTRACT = (
    "particle_view_target_selection_factory_config_v1"
)
PARTICLE_VIEW_TARGET_SELECTION_RESULT_CONTRACT = (
    "particle_view_target_selection_result_v1"
)
TARGET_SELECTION_RUN_ID = "SELECT_TARGET_DEFINITIONS"


def build_target_selection_factory_config(
    *,
    source_commit: str,
) -> dict[str, Any]:
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("source_commit must be nonempty")
    artifact = with_content_hash(
        {
            "contract": (
                PARTICLE_VIEW_TARGET_SELECTION_FACTORY_CONFIG_CONTRACT
            ),
            "source_commit": source_commit,
            "candidate_run_ids": list(TARGET_SCREEN_IDS),
            "consumer_run_ids": [
                f"SCREEN_{consumer_id}"
                for consumer_id in CONSUMER_SCREEN_IDS
            ],
            "canonical_target_id": CANONICAL_TARGET_DISCOVERY_RUN_ID,
            "forward_count": TARGET_SCREEN_FORWARD_COUNT,
            "ranking_split": "model_val_select",
            "quality_threshold_used_as_gate": False,
            "warnings_stop_execution": False,
        }
    )
    validate_target_selection_factory_config(artifact)
    return artifact


def validate_target_selection_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload,
        expected_contract=(
            PARTICLE_VIEW_TARGET_SELECTION_FACTORY_CONFIG_CONTRACT
        ),
    )
    if set(payload) != {
        "contract",
        "source_commit",
        "candidate_run_ids",
        "consumer_run_ids",
        "canonical_target_id",
        "forward_count",
        "ranking_split",
        "quality_threshold_used_as_gate",
        "warnings_stop_execution",
        "content_hash",
    }:
        raise ValueError("target-selection factory field inventory mismatch")
    if (
        not isinstance(payload["source_commit"], str)
        or not payload["source_commit"]
        or payload["candidate_run_ids"] != list(TARGET_SCREEN_IDS)
        or payload["consumer_run_ids"]
        != [f"SCREEN_{value}" for value in CONSUMER_SCREEN_IDS]
        or payload["canonical_target_id"]
        != CANONICAL_TARGET_DISCOVERY_RUN_ID
        or payload["forward_count"] != TARGET_SCREEN_FORWARD_COUNT
        or payload["ranking_split"] != "model_val_select"
        or payload["quality_threshold_used_as_gate"] is not False
        or payload["warnings_stop_execution"] is not False
    ):
        raise ValueError("target-selection factory policy changed")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "candidate_count": len(TARGET_SCREEN_IDS),
    }


def run_target_selection(
    *,
    candidates: Sequence[Mapping[str, Any]],
    consumer_metrics: Sequence[Mapping[str, Any]],
    unavailable_targets: Sequence[Mapping[str, Any]],
    source_commit: str,
    output_path: str,
    consumer_output_path: str,
    warnings_path: str,
    result_path: str,
) -> None:
    by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        validate_content_hash(
            candidate,
            expected_contract=PARTICLE_VIEW_TARGET_METRICS_CONTRACT,
        )
        target_id = str(candidate["target_id"])
        if target_id in by_id:
            raise ValueError("target selection contains a duplicate target")
        by_id[target_id] = candidate
    unavailable_by_id = {}
    for unavailable in unavailable_targets:
        validate_content_hash(
            unavailable,
            expected_contract=PARTICLE_VIEW_TARGET_UNAVAILABLE_CONTRACT,
        )
        target_id = str(unavailable["run_id"])
        if target_id in by_id or target_id in unavailable_by_id:
            raise ValueError("target availability inventory has duplicates")
        unavailable_by_id[target_id] = unavailable
    if set(by_id) | set(unavailable_by_id) != set(TARGET_SCREEN_IDS):
        raise ValueError("target selection screen coverage mismatch")
    ordered = [
        by_id[run_id] for run_id in TARGET_SCREEN_IDS if run_id in by_id
    ]
    selection = select_target_candidates(
        ordered,
        canonical_target_id=CANONICAL_TARGET_DISCOVERY_RUN_ID,
        forward_count=TARGET_SCREEN_FORWARD_COUNT,
    )
    write_immutable_json(output_path, selection)
    consumer_selection = select_consumer_interface(
        list(consumer_metrics)
    )
    write_immutable_json(consumer_output_path, consumer_selection)
    warnings = []
    for row in ordered:
        warnings.extend(
            build_scientific_warnings(
                row,
                graph_node="stage_b_target_selection",
                configuration_id=row["target_id"],
                seed=101,
                split="model_val_select",
                supporting_metric_sha256=row["content_hash"],
                source_commit=source_commit,
            )
        )
    warning_destination = Path(warnings_path)
    if warning_destination.exists():
        raise FileExistsError("refusing to append to existing warning ledger")
    write_scientific_warnings(warning_destination, warnings)
    result = with_content_hash(
        {
            "contract": PARTICLE_VIEW_TARGET_SELECTION_RESULT_CONTRACT,
            "selection_sha256": selection["content_hash"],
            "consumer_interface_selection_sha256": consumer_selection[
                "content_hash"
            ],
            "candidate_metric_sha256_by_run": {
                run_id: by_id[run_id]["content_hash"]
                for run_id in TARGET_SCREEN_IDS
                if run_id in by_id
            },
            "unavailable_target_sha256_by_run": {
                run_id: unavailable_by_id[run_id]["content_hash"]
                for run_id in TARGET_SCREEN_IDS
                if run_id in unavailable_by_id
            },
            "forwarded_target_ids": selection["forwarded_target_ids"],
            "warning_count": len(warnings),
            "warnings_stop_execution": False,
            "quality_threshold_used_as_gate": False,
        }
    )
    write_immutable_json(result_path, result)


def build_target_selection_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    validate_target_selection_factory_config(config)
    validate_particle_view_registry(registry)
    if (
        operation != "configuration_selection"
        or run_id != TARGET_SELECTION_RUN_ID
        or task_id != f"{run_id}__seed_{int(seed)}"
    ):
        raise ValueError("target-selection task identity is invalid")
    rows = {row["run_id"]: row for row in registry["runs"]}
    row = rows.get(run_id)
    if row is None or int(seed) not in row["seed_ids"]:
        raise ValueError("target-selection seed is not registered")
    root = Path(output_dir).resolve().parent.parent
    parents = resolve_parent_task_artifacts(
        registry=registry,
        artifact_root=root,
        run_id=run_id,
        seed=int(seed),
    )
    candidates = []
    unavailable_targets = []
    for target_id in TARGET_SCREEN_IDS:
        artifacts = parents[target_id]["artifacts"]
        binding = artifacts.get(
                "target_candidate_metrics.json"
            )
        expected_contract = PARTICLE_VIEW_TARGET_METRICS_CONTRACT
        if binding is None:
            binding = artifacts.get("target_unavailable.json")
            expected_contract = PARTICLE_VIEW_TARGET_UNAVAILABLE_CONTRACT
        if binding is None:
            raise ValueError(
                f"target parent {target_id} omitted ranking metrics"
            )
        path = Path(binding["path"]).resolve()
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"target metric changed for {target_id}")
        metric = load_hashed_json(path)
        validate_content_hash(metric, expected_contract=expected_contract)
        identity = (
            metric.get("target_id")
            if expected_contract == PARTICLE_VIEW_TARGET_METRICS_CONTRACT
            else metric.get("run_id")
        )
        if identity != target_id:
            raise ValueError("target metric/run identity mismatch")
        (
            candidates
            if expected_contract == PARTICLE_VIEW_TARGET_METRICS_CONTRACT
            else unavailable_targets
        ).append(metric)
    consumer_metrics = []
    for consumer_id in CONSUMER_SCREEN_IDS:
        run = f"SCREEN_{consumer_id}"
        binding = parents[run]["artifacts"].get(
            "consumer_interface_metrics.json"
        )
        if binding is None:
            raise ValueError(
                f"consumer parent {run} omitted interface metrics"
            )
        path = Path(binding["path"]).resolve()
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"consumer metric changed for {run}")
        metric = load_hashed_json(path)
        validate_content_hash(
            metric,
            expected_contract=PARTICLE_VIEW_CONSUMER_SCREEN_METRICS_CONTRACT,
        )
        if metric["run_id"] != run:
            raise ValueError("consumer metric/run identity mismatch")
        consumer_metrics.append(metric)
    output = Path(output_dir).resolve()
    return {
        "kwargs": {
            "candidates": candidates,
            "consumer_metrics": consumer_metrics,
            "unavailable_targets": unavailable_targets,
            "source_commit": config["source_commit"],
            "output_path": str(output / "selected_targets.json"),
            "consumer_output_path": str(
                output / "selected_consumer_interface.json"
            ),
            "warnings_path": str(output / "scientific_warnings.jsonl"),
            "result_path": str(output / "target_selection_result.json"),
        },
        "artifact_paths": [
            str(output / "selected_targets.json"),
            str(output / "selected_consumer_interface.json"),
            str(output / "scientific_warnings.jsonl"),
            str(output / "target_selection_result.json"),
        ],
        "action": run_target_selection,
    }


def build_target_selection_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    path = Path(factory_config_path).resolve()
    validate_target_selection_factory_config(load_hashed_json(path))
    return {
        TARGET_SELECTION_RUN_ID: {
            "operation": "configuration_selection",
            "factory": (
                "teacher_logit_reco.local_particle_residual_field."
                "particle_view.target_selection_runtime:"
                "build_target_selection_factory"
            ),
            "factory_config_path": str(path),
            "factory_config_sha256": sha256_file(path),
        }
    }


__all__ = [
    "PARTICLE_VIEW_TARGET_SELECTION_FACTORY_CONFIG_CONTRACT",
    "PARTICLE_VIEW_TARGET_SELECTION_RESULT_CONTRACT",
    "TARGET_SELECTION_RUN_ID",
    "build_target_selection_factory",
    "build_target_selection_factory_config",
    "build_target_selection_task_specs",
    "run_target_selection",
    "validate_target_selection_factory_config",
]

"""Immutable Stage-M scale refits, graph runs, and completion aggregation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .confirmation import (
    SCALE_SHORTLIST_CONTRACT,
    mean_log_selection_rejection,
)
from .contracts import (
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .evaluation import CLASSIFICATION_METRICS_CONTRACT
from .predictor_bundle import PIPELINE_SEEDS


SCALE_REFIT_BUNDLE_CONTRACT = "retb_scale_refit_bundle_v1"
SCALE_GRAPH_RUN_CONTRACT = "retb_scale_graph_run_v1"
SCALE_COMPLETION_CONTRACT = "retb_scale_completion_v1"

SCALE_REFIT_KEYS = frozenset(
    {
        "offline_input",
        "offline_relation",
        "offline_REGION",
        "shared_HLT_input",
        "shared_HLT_relation",
        "shared_HLT_REGION",
        "target_token",
        "uncertainty_calibrator",
    }
)
SCALE_COMPONENT_KEYS = frozenset(
    {
        "offline_experts",
        "offline_fusion",
        "scale_offline_target_cache",
        "scale_target_token_normalizer",
        "native_HLT_experts",
        "native_HLT_fusion",
        "predictor_bundle",
        "token_refiner_or_identity",
        "final_consumer",
        "deployable_export",
        "complete_graph_capacity",
        "training_curve",
        "val_stop_metrics",
        "pre_stack_val_confirmation_prediction",
        "pre_stack_val_confirmation_metrics",
    }
)


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is nonfinite")
    return result


def build_scale_refit_bundle(
    *,
    graph_id: str,
    pipeline_seed: int,
    locked_scale_shortlist_sha256: str,
    scale_train_manifest_sha256: str,
    val_design_identity_manifest_sha256: str,
    refits: Mapping[str, Mapping[str, Any]],
    five_hundred_k_artifact_hashes: Sequence[str],
) -> dict[str, Any]:
    if (
        not str(graph_id)
        or int(pipeline_seed) not in PIPELINE_SEEDS
        or set(refits) != set(SCALE_REFIT_KEYS)
    ):
        raise ValueError("scale-refit identity or coverage differs")
    scale_manifest = require_sha256(
        scale_train_manifest_sha256,
        name="scale_train_manifest_sha256",
    )
    design_manifest = require_sha256(
        val_design_identity_manifest_sha256,
        name="val_design_identity_manifest_sha256",
    )
    old_hashes = {
        require_sha256(value, name="five_hundred_k_artifact_hash")
        for value in five_hundred_k_artifact_hashes
    }
    if not old_hashes:
        raise ValueError("scale refit must enumerate rejected 500k parents")
    expected_population = {
        "offline_input": ("offline_scale", scale_manifest),
        "offline_relation": ("offline_scale", scale_manifest),
        "offline_REGION": ("offline_scale", scale_manifest),
        "shared_HLT_input": ("shared_hlt_scale", scale_manifest),
        "shared_HLT_relation": ("shared_hlt_scale", scale_manifest),
        "shared_HLT_REGION": ("shared_hlt_scale", scale_manifest),
        "target_token": (
            "scale_train_offline_targets",
            scale_manifest,
        ),
        "uncertainty_calibrator": (
            "val_design_label_free",
            design_manifest,
        ),
    }
    checked = {}
    for name in sorted(SCALE_REFIT_KEYS):
        row = dict(refits[name])
        if set(row) != {
            "artifact_sha256",
            "population",
            "identity_manifest_sha256",
            "recipe_sha256",
            "fitted_values_sha256",
            "labels_consumed",
            "replica_ids",
        }:
            raise ValueError(f"scale-refit fields differ for {name}")
        population, identity_sha = expected_population[name]
        artifact_sha = require_sha256(
            row["artifact_sha256"], name=f"refits.{name}.artifact_sha256"
        )
        expected_replicas = (
            [0, 1, 2, 3] if name.startswith("shared_HLT_") else []
        )
        if (
            artifact_sha in old_hashes
            or row["population"] != population
            or row["identity_manifest_sha256"] != identity_sha
            or bool(row["labels_consumed"])
            or row["replica_ids"] != expected_replicas
        ):
            raise ValueError(f"scale-refit lineage differs for {name}")
        checked[name] = {
            "artifact_sha256": artifact_sha,
            "population": population,
            "identity_manifest_sha256": require_sha256(
                row["identity_manifest_sha256"],
                name=f"refits.{name}.identity_manifest_sha256",
            ),
            "recipe_sha256": require_sha256(
                row["recipe_sha256"],
                name=f"refits.{name}.recipe_sha256",
            ),
            "fitted_values_sha256": require_sha256(
                row["fitted_values_sha256"],
                name=f"refits.{name}.fitted_values_sha256",
            ),
            "labels_consumed": False,
            "replica_ids": expected_replicas,
        }
    return with_content_hash(
        {
            "contract": SCALE_REFIT_BUNDLE_CONTRACT,
            "schema_version": 1,
            "graph_id": str(graph_id),
            "pipeline_seed": int(pipeline_seed),
            "locked_scale_shortlist_sha256": require_sha256(
                locked_scale_shortlist_sha256,
                name="locked_scale_shortlist_sha256",
            ),
            "scale_train_manifest_sha256": scale_manifest,
            "val_design_identity_manifest_sha256": design_manifest,
            "refits": checked,
            "rejected_500k_artifact_hashes": sorted(old_hashes),
            "all_train_derived_statistics_refitted": True,
            "uncertainty_calibrator_fitted_label_free": True,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
    )


def validate_scale_refit_bundle(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SCALE_REFIT_BUNDLE_CONTRACT
    )
    expected = build_scale_refit_bundle(
        graph_id=payload.get("graph_id", ""),
        pipeline_seed=payload.get("pipeline_seed", -1),
        locked_scale_shortlist_sha256=payload.get(
            "locked_scale_shortlist_sha256"
        ),
        scale_train_manifest_sha256=payload.get(
            "scale_train_manifest_sha256"
        ),
        val_design_identity_manifest_sha256=payload.get(
            "val_design_identity_manifest_sha256"
        ),
        refits=payload.get("refits", {}),
        five_hundred_k_artifact_hashes=payload.get(
            "rejected_500k_artifact_hashes", []
        ),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("scale-refit bundle semantics differ")
    return digest


def build_scale_graph_run(
    *,
    locked_scale_shortlist: Mapping[str, Any],
    graph_id: str,
    pipeline_seed: int,
    scale_refit_bundle: Mapping[str, Any],
    component_hashes: Mapping[str, str],
    selected_epoch: int,
    val_stop_accuracy: float,
    val_stop_cross_entropy: float,
    analytical_flops_batch1: int,
    analytical_flops_batch128: int,
    parameter_count: int,
    pre_stack_confirmation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    shortlist_sha = validate_content_hash(
        locked_scale_shortlist, expected_contract=SCALE_SHORTLIST_CONTRACT
    )
    graph = str(graph_id)
    definitions = locked_scale_shortlist.get(
        "locked_graph_definitions", {}
    )
    if (
        graph not in locked_scale_shortlist.get("SCALE_SHORTLIST", [])
        or graph not in definitions
        or int(pipeline_seed) not in PIPELINE_SEEDS
        or set(component_hashes) != set(SCALE_COMPONENT_KEYS)
    ):
        raise ValueError("scale graph run is outside the locked shortlist")
    refit_sha = validate_scale_refit_bundle(scale_refit_bundle)
    validate_content_hash(
        pre_stack_confirmation_metrics,
        expected_contract=CLASSIFICATION_METRICS_CONTRACT,
    )
    if (
        scale_refit_bundle.get("graph_id") != graph
        or int(scale_refit_bundle.get("pipeline_seed", -1))
        != int(pipeline_seed)
        or scale_refit_bundle.get("locked_scale_shortlist_sha256")
        != shortlist_sha
        or pre_stack_confirmation_metrics.get("split") != "val_design"
        or pre_stack_confirmation_metrics.get("source")
        != locked_scale_shortlist.get("source")
        or scale_refit_bundle.get("source")
        != locked_scale_shortlist.get("source")
        or component_hashes.get("pre_stack_val_confirmation_metrics")
        != pre_stack_confirmation_metrics.get("content_hash")
        or component_hashes.get("scale_target_token_normalizer")
        != scale_refit_bundle["refits"]["target_token"][
            "artifact_sha256"
        ]
        or not 1 <= int(selected_epoch) <= 40
        or min(
            int(analytical_flops_batch1),
            int(analytical_flops_batch128),
            int(parameter_count),
        )
        <= 0
    ):
        raise ValueError("scale graph run lineage/metrics differ")
    val_accuracy = _finite(
        val_stop_accuracy, name="val_stop_accuracy"
    )
    val_ce = _finite(
        val_stop_cross_entropy, name="val_stop_cross_entropy"
    )
    if not 0.0 <= val_accuracy <= 1.0 or val_ce < 0.0:
        raise ValueError("scale graph val_stop metrics differ")
    return with_content_hash(
        {
            "contract": SCALE_GRAPH_RUN_CONTRACT,
            "schema_version": 1,
            "locked_scale_shortlist_sha256": shortlist_sha,
            "graph_id": graph,
            "pipeline_seed": int(pipeline_seed),
            "locked_graph_definition": dict(definitions[graph]),
            "scale_refit_bundle_sha256": refit_sha,
            "scale_refit_bundle": dict(scale_refit_bundle),
            "component_hashes": {
                name: require_sha256(
                    value, name=f"component_hashes.{name}"
                )
                for name, value in sorted(component_hashes.items())
            },
            "checkpoint_selection": {
                "split": "val_stop",
                "selected_epoch": int(selected_epoch),
                "maximum_epochs": 40,
                "accuracy": val_accuracy,
                "cross_entropy": val_ce,
                "window": 0.0001,
                "tie_order": [
                    "maximum_accuracy",
                    "lower_cross_entropy",
                    "earliest_epoch",
                ],
            },
            "pre_stack_confirmation": {
                "split": "val_design",
                "metrics_sha256": pre_stack_confirmation_metrics[
                    "content_hash"
                ],
                "metrics": dict(pre_stack_confirmation_metrics),
                "mean_log_Jeffreys_selection_rejection": (
                    mean_log_selection_rejection(
                        pre_stack_confirmation_metrics
                    )
                ),
            },
            "capacity": {
                "analytical_flops_batch1": int(
                    analytical_flops_batch1
                ),
                "analytical_flops_batch128": int(
                    analytical_flops_batch128
                ),
                "parameter_count": int(parameter_count),
            },
            "training_population": "scale_train",
            "epoch_selection_population": "val_stop",
            "architecture_reselection_performed": False,
            "component_reselection_performed": False,
            "performance_based_termination": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
    )


def validate_scale_graph_run(
    payload: Mapping[str, Any],
    *,
    locked_scale_shortlist: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SCALE_GRAPH_RUN_CONTRACT
    )
    checkpoint = payload.get("checkpoint_selection", {})
    capacity = payload.get("capacity", {})
    confirmation = payload.get("pre_stack_confirmation", {})
    expected = build_scale_graph_run(
        locked_scale_shortlist=locked_scale_shortlist,
        graph_id=payload.get("graph_id", ""),
        pipeline_seed=payload.get("pipeline_seed", -1),
        scale_refit_bundle=payload.get("scale_refit_bundle", {}),
        component_hashes=payload.get("component_hashes", {}),
        selected_epoch=checkpoint.get("selected_epoch", -1),
        val_stop_accuracy=checkpoint.get("accuracy", math.nan),
        val_stop_cross_entropy=checkpoint.get(
            "cross_entropy", math.nan
        ),
        analytical_flops_batch1=capacity.get(
            "analytical_flops_batch1", 0
        ),
        analytical_flops_batch128=capacity.get(
            "analytical_flops_batch128", 0
        ),
        parameter_count=capacity.get("parameter_count", 0),
        pre_stack_confirmation_metrics=confirmation.get("metrics", {}),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("scale graph run semantics differ")
    return digest


def aggregate_scale_completion(
    *,
    locked_scale_shortlist: Mapping[str, Any],
    scale_graph_runs: Sequence[Mapping[str, Any]],
    step14_bundle_sha256: str,
    scale_train_manifest_sha256: str,
) -> dict[str, Any]:
    shortlist_sha = validate_content_hash(
        locked_scale_shortlist, expected_contract=SCALE_SHORTLIST_CONTRACT
    )
    expected = {
        (graph_id, seed)
        for graph_id in locked_scale_shortlist["SCALE_SHORTLIST"]
        for seed in PIPELINE_SEEDS
    }
    if len(scale_graph_runs) != len(expected):
        raise ValueError("scale completion row count differs")
    checked, seen = [], set()
    source = locked_scale_shortlist.get("source")
    for row in scale_graph_runs:
        key = (str(row.get("graph_id")), int(row.get("pipeline_seed", -1)))
        if key not in expected or key in seen:
            raise ValueError("scale completion graph/seed coverage differs")
        seen.add(key)
        run_sha = validate_scale_graph_run(
            row, locked_scale_shortlist=locked_scale_shortlist
        )
        if (
            row.get("source") != source
            or row["scale_refit_bundle"].get(
                "scale_train_manifest_sha256"
            )
            != scale_train_manifest_sha256
        ):
            raise ValueError("scale completion source/population differs")
        checked.append(
            {
                "graph_id": key[0],
                "pipeline_seed": key[1],
                "scale_graph_run_sha256": run_sha,
                "complete_graph_definition_sha256": row[
                    "locked_graph_definition"
                ]["complete_graph_definition_sha256"],
                "deployable_export_sha256": row["component_hashes"][
                    "deployable_export"
                ],
                "complete_graph_capacity_sha256": row[
                    "component_hashes"
                ]["complete_graph_capacity"],
                "pre_stack_val_confirmation_prediction_sha256": row[
                    "component_hashes"
                ]["pre_stack_val_confirmation_prediction"],
                "pre_stack_val_confirmation_metrics_sha256": row[
                    "component_hashes"
                ]["pre_stack_val_confirmation_metrics"],
                "capacity": row["capacity"],
                "pre_stack_confirmation": row[
                    "pre_stack_confirmation"
                ],
                "component_hashes": row["component_hashes"],
                "scale_refit_bundle_sha256": row[
                    "scale_refit_bundle_sha256"
                ],
            }
        )
    if seen != expected:
        raise ValueError("scale completion lacks a shortlisted graph/seed")
    return with_content_hash(
        {
            "contract": SCALE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "parents": {
                "locked_scale_shortlist": shortlist_sha,
                "step14_bundle": require_sha256(
                    step14_bundle_sha256, name="step14_bundle_sha256"
                ),
                "scale_train_manifest": require_sha256(
                    scale_train_manifest_sha256,
                    name="scale_train_manifest_sha256",
                ),
            },
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "shortlisted_graph_ids": list(
                locked_scale_shortlist["SCALE_SHORTLIST"]
            ),
            "expected_run_count": len(expected),
            "runs": sorted(
                checked,
                key=lambda row: (
                    row["graph_id"],
                    row["pipeline_seed"],
                ),
            ),
            "every_and_only_shortlisted_graph_trained": True,
            "all_shortlisted_3M_checkpoints_immutable": True,
            "all_scale_statistics_refitted": True,
            "stack_val_authorized_after_this_artifact": True,
            "architecture_reselection_performed": False,
            "performance_based_termination": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
    )


def validate_scale_completion(
    payload: Mapping[str, Any],
    *,
    locked_scale_shortlist: Mapping[str, Any],
    scale_graph_runs: Sequence[Mapping[str, Any]],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SCALE_COMPLETION_CONTRACT
    )
    expected = aggregate_scale_completion(
        locked_scale_shortlist=locked_scale_shortlist,
        scale_graph_runs=scale_graph_runs,
        step14_bundle_sha256=payload.get("parents", {}).get(
            "step14_bundle"
        ),
        scale_train_manifest_sha256=payload.get("parents", {}).get(
            "scale_train_manifest"
        ),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("scale completion semantics differ")
    return digest


__all__ = [
    "SCALE_COMPONENT_KEYS",
    "SCALE_COMPLETION_CONTRACT",
    "SCALE_GRAPH_RUN_CONTRACT",
    "SCALE_REFIT_BUNDLE_CONTRACT",
    "SCALE_REFIT_KEYS",
    "aggregate_scale_completion",
    "build_scale_graph_run",
    "build_scale_refit_bundle",
    "validate_scale_completion",
    "validate_scale_graph_run",
    "validate_scale_refit_bundle",
]

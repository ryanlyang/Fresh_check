"""Deterministic phased orchestration for the Stage-F/G/H predictor campaign."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .contracts import canonical_sha256, with_content_hash
from .step9 import (
    ARCHITECTURE_CONTEXT_ROWS,
    LOSS_IDS,
    REPRESENTATIVE_EXPERTS,
    SCREEN_SHAPES,
    build_stage_f_registry,
    build_stage_g_registry,
)


PREDICTOR_PHASE_SELECTION_CONTRACT = (
    "retb_predictor_phase_selection_v1"
)
PREDICTOR_FOLLOWUP_REGISTRY_CONTRACT = (
    "retb_predictor_optimizer_followup_registry_v1"
)
PHASED_CONTROLLER_TOPOLOGY_CONTRACT = (
    "retb_scientific_phased_controller_topology_v2"
)
PREDICTOR_PHASE_ORDER = (
    "F_ARCHITECTURE_SCREEN",
    "F_ARCHITECTURE_SELECT",
    "F_OPTIMIZER_SCREEN",
    "F_OPTIMIZER_SELECT",
    "G_OBJECTIVE_SCREEN",
    "G_CONFIGURATION_SELECT",
    "H_EVIDENCE_SELECT",
    "H_EVIDENCE_CONFIRM",
    "H_EVIDENCE_CACHE",
    "H_EVIDENCE_FUSION_CONFIRM",
    "H_CONFIRMATION",
)
JOINT_PHASE_ORDER = (
    "J0_J4_CANDIDATES",
    "J4_BLOCK_SELECT",
    "J5_END_TO_END",
)
FINAL_CONSUMER_PHASE_ORDER = (
    "FINAL_DATASET_PREP",
    "TOKEN_REFINER_WAVE",
    "TOKEN_REFINER_SELECT",
    "FINAL_CONSUMER_WAVE",
)


def build_phased_controller_topology(
    controller_id: str,
) -> dict[str, Any]:
    if controller_id == "predictor_campaign":
        public_target = "predictor_training"
        phases = PREDICTOR_PHASE_ORDER
        selection_edges = {
            "F_ARCHITECTURE_SELECT": "F_ARCHITECTURE_SCREEN",
            "F_OPTIMIZER_SCREEN": "F_ARCHITECTURE_SELECT",
            "F_OPTIMIZER_SELECT": "F_OPTIMIZER_SCREEN",
            "G_OBJECTIVE_SCREEN": "F_OPTIMIZER_SELECT",
            "G_CONFIGURATION_SELECT": "G_OBJECTIVE_SCREEN",
            "H_EVIDENCE_SELECT": "G_CONFIGURATION_SELECT",
            "H_EVIDENCE_CONFIRM": "H_EVIDENCE_SELECT",
            "H_EVIDENCE_CACHE": "H_EVIDENCE_CONFIRM",
            "H_EVIDENCE_FUSION_CONFIRM": "H_EVIDENCE_CACHE",
            "H_CONFIRMATION": "H_EVIDENCE_FUSION_CONFIRM",
        }
    elif controller_id == "joint_predictor_campaign":
        public_target = "joint_predictor_training"
        phases = JOINT_PHASE_ORDER
        selection_edges = {
            "J4_BLOCK_SELECT": "J0_J4_CANDIDATES",
            "J5_END_TO_END": "J4_BLOCK_SELECT",
        }
    elif controller_id == "final_consumer_campaign":
        public_target = "final_consumer_training"
        phases = FINAL_CONSUMER_PHASE_ORDER
        selection_edges = {
            "TOKEN_REFINER_WAVE": "FINAL_DATASET_PREP",
            "TOKEN_REFINER_SELECT": "TOKEN_REFINER_WAVE",
            "FINAL_CONSUMER_WAVE": "TOKEN_REFINER_SELECT",
        }
    else:
        raise ValueError("unknown RETB phased controller")
    return with_content_hash(
        {
            "contract": PHASED_CONTROLLER_TOPOLOGY_CONTRACT,
            "schema_version": 2,
            "controller_id": controller_id,
            "public_target": public_target,
            "public_target_count_preserved": True,
            "internal_phase_order": list(phases),
            "selection_edges": selection_edges,
            "later_phase_plan_binds_all_prior_phase_completions": True,
            "selection_result_sign_blocks_continuation": False,
            "incomplete_or_stale_phase_blocks_continuation": True,
        }
    )


def _complete_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_ids: set[str],
    id_field: str,
) -> dict[str, dict[str, Any]]:
    required = {
        id_field,
        "val_design_accuracy",
        "normalized_token_error",
        "parameter_count",
        "result_sha256",
    }
    output: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        if not required.issubset(row):
            raise ValueError("predictor phase result fields differ")
        identity = str(row[id_field])
        if identity in output:
            raise ValueError("predictor phase result is duplicated")
        output[identity] = row
    if set(output) != expected_ids:
        raise ValueError("predictor phase result coverage differs")
    return output


def _aggregate_and_select(
    groups: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]],
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    aggregates = []
    for key, rows in sorted(groups.items()):
        aggregates.append(
            {
                "configuration": list(key),
                "mean_val_design_accuracy": sum(
                    float(row["val_design_accuracy"]) for row in rows
                )
                / len(rows),
                "mean_normalized_token_error": sum(
                    float(row["normalized_token_error"]) for row in rows
                )
                / len(rows),
                "parameter_count": max(
                    int(row["parameter_count"]) for row in rows
                ),
                "result_hashes": sorted(
                    str(row["result_sha256"]) for row in rows
                ),
            }
        )
    maximum = max(row["mean_val_design_accuracy"] for row in aggregates)
    eligible = [
        row
        for row in aggregates
        if maximum - row["mean_val_design_accuracy"] <= 0.0001
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["mean_normalized_token_error"],
            row["parameter_count"],
            tuple(str(value) for value in row["configuration"]),
        ),
    )
    return tuple(selected["configuration"]), aggregates


def select_stage_f_architecture_families(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    registry = build_stage_f_registry()
    by_id = _complete_results(
        results,
        expected_ids={str(row["run_id"]) for row in registry["rows"]},
        id_field="run_id",
    )
    selections, traces = {}, {}
    for family, architecture in (
        ("selected_direct", "A3_SLOT_DECODER_DIRECT"),
        ("selected_gated", "A4_SLOT_DECODER_GATED"),
    ):
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for row in registry["rows"]:
            if row["architecture"] == architecture:
                groups[(row["architecture"], row["context"])].append(
                    by_id[row["run_id"]]
                )
        selected, trace = _aggregate_and_select(groups)
        selections[family] = {
            "architecture": selected[0],
            "context": selected[1],
        }
        traces[family] = trace
    return with_content_hash(
        {
            "contract": PREDICTOR_PHASE_SELECTION_CONTRACT,
            "schema_version": 1,
            "phase": "F_ARCHITECTURE_CONTEXT",
            "stage_f_registry_sha256": registry["content_hash"],
            "selected_families": selections,
            "selection_traces": traces,
            "complete_result_count": len(by_id),
            "scientific_underperformance_blocks_continuation": False,
        }
    )


def build_stage_f_optimizer_followup_registry(
    architecture_selection: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        architecture_selection.get("contract")
        != PREDICTOR_PHASE_SELECTION_CONTRACT
        or architecture_selection.get("phase")
        != "F_ARCHITECTURE_CONTEXT"
    ):
        raise ValueError("predictor architecture selection differs")
    rows = []
    for family, selected in sorted(
        architecture_selection["selected_families"].items()
    ):
        for expert in ("PT", "TRACK"):
            for shape in SCREEN_SHAPES:
                for learning_rate in (2.0e-4, 5.0e-4, 1.0e-3):
                    for dropout in (0.0, 0.1):
                        identity = {
                            "family": family,
                            "architecture": selected["architecture"],
                            "context": selected["context"],
                            "expert_id": expert,
                            "shape_alias": shape,
                            "learning_rate": learning_rate,
                            "dropout": dropout,
                            "pipeline_seed": 101,
                        }
                        rows.append(
                            {
                                **identity,
                                "run_id": (
                                    "RETB_FOPT_"
                                    + canonical_sha256(identity)[:20]
                                ),
                                "simplicity_control": (
                                    learning_rate == 5.0e-4
                                    and dropout == 0.0
                                ),
                            }
                        )
    return with_content_hash(
        {
            "contract": PREDICTOR_FOLLOWUP_REGISTRY_CONTRACT,
            "schema_version": 1,
            "architecture_selection_sha256": (
                architecture_selection["content_hash"]
            ),
            "membership_count": len(rows),
            "rows": rows,
            "performance_based_termination": False,
        }
    )


def select_stage_f_optimizer_configurations(
    *,
    registry: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        registry.get("contract")
        != PREDICTOR_FOLLOWUP_REGISTRY_CONTRACT
        or int(registry.get("membership_count", -1))
        != len(registry.get("rows", ()))
    ):
        raise ValueError("predictor optimizer registry differs")
    by_id = _complete_results(
        results,
        expected_ids={str(row["run_id"]) for row in registry["rows"]},
        id_field="run_id",
    )
    selections, traces = {}, {}
    for family in ("selected_direct", "selected_gated"):
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for row in registry["rows"]:
            if row["family"] == family:
                groups[
                    (float(row["learning_rate"]), float(row["dropout"]))
                ].append(by_id[row["run_id"]])
        selected, trace = _aggregate_and_select(groups)
        selections[family] = {
            "learning_rate": selected[0],
            "dropout": selected[1],
            "simplicity_control": {
                "learning_rate": 5.0e-4,
                "dropout": 0.0,
            },
        }
        traces[family] = trace
    return with_content_hash(
        {
            "contract": PREDICTOR_PHASE_SELECTION_CONTRACT,
            "schema_version": 1,
            "phase": "F_OPTIMIZER",
            "optimizer_registry_sha256": registry["content_hash"],
            "selected_families": selections,
            "selection_traces": traces,
            "complete_result_count": len(by_id),
            "scientific_underperformance_blocks_continuation": False,
        }
    )


def select_stage_g_configurations(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    registry = build_stage_g_registry()
    by_id = _complete_results(
        results,
        expected_ids={
            str(row["template_id"]) for row in registry["templates"]
        },
        id_field="template_id",
    )
    selections, traces = {}, {}
    for family in ("SELECTED_DIRECT", "SELECTED_GATED"):
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for row in registry["templates"]:
            if row["architecture_family"] == family:
                key = (
                    row["objective_id"],
                    row["uncertainty_head"],
                    row["normalization_mode"],
                )
                groups[key].append(by_id[row["template_id"]])
        selected, trace = _aggregate_and_select(groups)
        selections[family.lower()] = {
            "objective_id": selected[0],
            "uncertainty_head": selected[1],
            "normalization_mode": selected[2],
        }
        traces[family.lower()] = trace
    return with_content_hash(
        {
            "contract": PREDICTOR_PHASE_SELECTION_CONTRACT,
            "schema_version": 1,
            "phase": "G_OBJECTIVE_UNCERTAINTY_NORMALIZATION",
            "stage_g_registry_sha256": registry["content_hash"],
            "selected_families": selections,
            "selection_traces": traces,
            "complete_result_count": len(by_id),
            "scientific_underperformance_blocks_continuation": False,
        }
    )


__all__ = [
    "FINAL_CONSUMER_PHASE_ORDER",
    "JOINT_PHASE_ORDER",
    "PHASED_CONTROLLER_TOPOLOGY_CONTRACT",
    "PREDICTOR_PHASE_ORDER",
    "PREDICTOR_FOLLOWUP_REGISTRY_CONTRACT",
    "PREDICTOR_PHASE_SELECTION_CONTRACT",
    "build_phased_controller_topology",
    "build_stage_f_optimizer_followup_registry",
    "select_stage_f_architecture_families",
    "select_stage_f_optimizer_configurations",
    "select_stage_g_configurations",
]

"""Resolve selection locks into executable, immutable HOSD graph definitions."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    COMBINATION_SELECTION_CONTRACT,
    FEEDBACK_SELECTION_CONTRACT,
    GRAPH_REGISTRY_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    STAGE_F_PLAN_CONTRACT,
    validate_content_hash,
    with_content_hash,
)


def build_locked_graph_registry(
    *,
    single_family_selection: Mapping[str, Any],
    feedback_selection: Mapping[str, Any],
    combination_selection: Mapping[str, Any],
    stage_f_plan: Mapping[str, Any],
    retb_comparators: Mapping[str, Mapping[str, Any] | None],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        single_family_selection,
        expected_contract=SINGLE_FAMILY_SELECTION_CONTRACT,
    )
    validate_content_hash(
        feedback_selection, expected_contract=FEEDBACK_SELECTION_CONTRACT
    )
    validate_content_hash(
        combination_selection, expected_contract=COMBINATION_SELECTION_CONTRACT
    )
    validate_content_hash(stage_f_plan, expected_contract=STAGE_F_PLAN_CONTRACT)
    physical = [
        row
        for row in single_family_selection["cross_family_order"]
        if str(row["target_id"]).startswith("T_OFFLINE_")
        and not str(row["target_id"]).startswith(
            ("T_OFFLINE_LOGITS_", "T_OFFLINE_POOLED_LATENT")
        )
    ]
    if not physical:
        raise ValueError("graph registry lacks a selected physical auxiliary")
    physical_target = str(physical[0]["target_id"])
    physical_definition = dict(
        single_family_selection["selected_definition_by_target"][physical_target]
    )
    physical_definition["row_id"] = single_family_selection[
        "selected_row_by_target"
    ][physical_target]
    mandatory = {
        row["combination_id"]: row
        for row in stage_f_plan["mandatory_combinations"]
    }
    if "C_PHYSICAL_KD" not in mandatory:
        raise ValueError("graph registry lacks C_PHYSICAL_KD")
    definitions = {
        "H_BASE": {
            "graph_id": "H_BASE",
            "graph_kind": "BASELINE",
            "baseline_id": "H_BASE",
        },
        "H_BASE_LONG": {
            "graph_id": "H_BASE_LONG",
            "graph_kind": "BASELINE",
            "baseline_id": "H_BASE_LONG",
        },
        "H_PARTICLENET": {
            "graph_id": "H_PARTICLENET",
            "graph_kind": "BASELINE",
            "baseline_id": "H_PARTICLENET",
        },
        "H_KD_O_BASE": {
            "graph_id": "H_KD_LOGIT_O_BASE",
            "graph_kind": "BASELINE",
            "baseline_id": "H_KD_LOGIT_O_BASE",
        },
        "H_KD_O_FULLREL": {
            "graph_id": "H_KD_LOGIT_O_FULLREL",
            "graph_kind": "BASELINE",
            "baseline_id": "H_KD_LOGIT_O_FULLREL",
        },
        "BEST_PHYSICAL_AUX": {
            "graph_id": physical_definition["row_id"],
            "graph_kind": "AUXILIARY",
            "row": physical_definition,
        },
        "BEST_FEEDBACK": {
            "graph_id": feedback_selection["selected_feedback_row_id"],
            "graph_kind": "FEEDBACK",
            "row": dict(feedback_selection["selected_feedback_definition"]),
        },
        **{
            role: {
                "graph_id": definition["row_id"],
                "graph_kind": "FEEDBACK",
                "row": dict(definition),
                "graph_role": "REFERENCE_BASELINE",
                "eligible_for_hosd_family_selection": False,
                "eligible_for_overall_finalist_selection": True,
            }
            for role, definition in sorted(
                feedback_selection["reference_graph_definitions"].items()
            )
        },
        "BEST_COMBINATION": {
            "graph_id": combination_selection["selected_combination_graph_id"],
            "graph_kind": "COMBINATION",
            "graph": dict(combination_selection["selected_graph_definition"]),
        },
        "C_PHYSICAL_KD": {
            "graph_id": mandatory["C_PHYSICAL_KD"]["graph_id"],
            "graph_kind": "COMBINATION",
            "graph": dict(mandatory["C_PHYSICAL_KD"]),
        },
    }
    retb = {}
    for role, definition in sorted(retb_comparators.items()):
        if definition is None:
            retb[str(role)] = None
        else:
            checked = dict(definition)
            if not checked.get("graph_id") or not checked.get("export_sha256"):
                raise ValueError("compatible RETB comparator definition is incomplete")
            retb[str(role)] = checked
    return with_content_hash(
        {
            "contract": GRAPH_REGISTRY_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "parent_locks": {
                "single_family_selection": single_family_selection["content_hash"],
                "feedback_selection": feedback_selection["content_hash"],
                "combination_selection": combination_selection["content_hash"],
                "stage_f_plan": stage_f_plan["content_hash"],
            },
            "definitions_by_role": definitions,
            "retb_comparators": retb,
            "role_order": list(definitions),
            "all_graphs_executable": True,
            "manual_graph_id_entry": False,
        }
    )


__all__ = ["build_locked_graph_registry"]

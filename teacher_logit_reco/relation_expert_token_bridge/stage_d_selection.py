"""Deterministic complete-coverage selection of native HLT evidence modes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .registry import EXPERT_ORDER
from .step6 import validate_stage_d_run_registry


STAGE_D_EVIDENCE_SELECTION_CONTRACT = (
    "retb_stage_d_evidence_mode_selection_v1"
)
EVIDENCE_MODES = (
    "HE_SCRATCH_CE",
    "HE_OFFLINE_INIT",
    "HE_DUAL_OBJECTIVE",
)
SCREEN_SHAPES = ("SHAPE_COMPACT", "SHAPE_HIGH")


def select_stage_d_evidence_modes(
    *,
    registry: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    registry_sha = validate_stage_d_run_registry(registry)
    candidates = [
        row
        for row in registry["encoder_screen_rows"]
        if row["configuration"]["shape_id"] in SCREEN_SHAPES
        and row["configuration"]["mode"] in EVIDENCE_MODES
    ]
    expected = {str(row["run_id"]) for row in candidates}
    by_id: dict[str, dict[str, Any]] = {}
    for source in results:
        row = dict(source)
        required = {
            "run_id",
            "val_design_accuracy",
            "val_design_cross_entropy",
            "result_sha256",
        }
        if not required <= set(row):
            raise ValueError("Stage-D evidence result fields differ")
        run_id = str(row["run_id"])
        if run_id in by_id:
            raise ValueError("Stage-D evidence result is duplicated")
        require_sha256(row["result_sha256"], name="result_sha256")
        by_id[run_id] = row
    if set(by_id) != expected:
        raise ValueError("Stage-D evidence result coverage differs")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    rows_by_id = {str(row["run_id"]): row for row in candidates}
    for run_id, result in by_id.items():
        config = rows_by_id[run_id]["configuration"]
        groups[
            (
                str(config["shape_id"]),
                str(config["expert_id"]),
                str(config["mode"]),
            )
        ].append(result)
    required_groups = {
        (shape, expert, mode)
        for shape in SCREEN_SHAPES
        for expert in EXPERT_ORDER
        for mode in EVIDENCE_MODES
    }
    if set(groups) != required_groups:
        raise ValueError("Stage-D evidence selection groups differ")

    selected = []
    traces = []
    for key in sorted(groups):
        group = groups[key]
        maximum = max(float(row["val_design_accuracy"]) for row in group)
        window = [
            row
            for row in group
            if maximum - float(row["val_design_accuracy"]) <= 0.0001
        ]

        def rank(result: Mapping[str, Any]) -> tuple[Any, ...]:
            config = rows_by_id[str(result["run_id"])]["configuration"]
            return (
                float(result["val_design_cross_entropy"]),
                0 if config["realization_policy"] == "R_MULTI" else 1,
                0 if not config["measurement_embedding"] else 1,
                float(config["lambda_token"]) + float(config["lambda_logit"]),
                float(config["lambda_token"]),
                float(config["lambda_logit"]),
                str(result["run_id"]),
            )

        winner = min(window, key=rank)
        source = rows_by_id[str(winner["run_id"])]
        selected.append(
            {
                "shape_id": key[0],
                "expert_id": key[1],
                "mode": key[2],
                "selected_screen_run_id": source["run_id"],
                "configuration": source["configuration"],
                "result_sha256": winner["result_sha256"],
            }
        )
        traces.append(
            {
                "group": list(key),
                "maximum_accuracy": maximum,
                "accuracy_window": 0.0001,
                "candidate_run_ids": sorted(
                    str(row["run_id"]) for row in group
                ),
                "selected_run_id": source["run_id"],
            }
        )
    fusion_rows = [
        row
        for row in registry["native_fusion_rows"]
        if row["configuration"]["shape_id"] in SCREEN_SHAPES
        and row["configuration"]["fusion_variant"]
        in {"HF_NATIVE", "HF_TRAINED_LOGIT"}
    ]
    if len(fusion_rows) != 4:
        raise ValueError("Stage-D selected native fusion coverage differs")
    return with_content_hash(
        {
            "contract": STAGE_D_EVIDENCE_SELECTION_CONTRACT,
            "schema_version": 1,
            "stage_d_run_registry_sha256": registry_sha,
            "screen_seed": 101,
            "shapes": list(SCREEN_SHAPES),
            "expert_order": list(EXPERT_ORDER),
            "evidence_modes": list(EVIDENCE_MODES),
            "selected_rows": selected,
            "selected_native_fusion_run_ids": [
                str(row["run_id"]) for row in fusion_rows
            ],
            "selection_traces": traces,
            "complete_result_count": len(by_id),
            "scientific_underperformance_blocks_continuation": False,
        }
    )


def validate_stage_d_evidence_selection(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_D_EVIDENCE_SELECTION_CONTRACT
    )
    selected = list(payload.get("selected_rows", ()))
    coordinates = {
        (row["shape_id"], row["expert_id"], row["mode"])
        for row in selected
    }
    expected = {
        (shape, expert, mode)
        for shape in SCREEN_SHAPES
        for expert in EXPERT_ORDER
        for mode in EVIDENCE_MODES
    }
    if (
        coordinates != expected
        or len(selected) != len(expected)
        or payload.get("scientific_underperformance_blocks_continuation")
        is not False
    ):
        raise ValueError("Stage-D evidence selection semantics differ")
    return digest


__all__ = [
    "EVIDENCE_MODES",
    "SCREEN_SHAPES",
    "STAGE_D_EVIDENCE_SELECTION_CONTRACT",
    "select_stage_d_evidence_modes",
    "validate_stage_d_evidence_selection",
]

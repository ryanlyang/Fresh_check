"""Deterministic complete-graph capacity matching and label-exposure controls."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import require_sha256, with_content_hash


CAPACITY_CONTROL_CONTRACT = "retb_complete_graph_capacity_controls_v1"
OFFLINE_LONG_EXPOSURE_CONTRACT = "retb_offline_long_exposure_ledger_v1"


def build_capacity_control_registry() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": "retb_capacity_control_registry_v1",
            "schema_version": 1,
            "offline_controls": [
                "O_BASE",
                "O_WIDE",
                "O_MONO_PARAM",
                "O_MONO_FLOP",
                "O_BASE_LONG",
                "O_FULLREL",
                "O_GROUPED_HEAD_REL",
                "O_7X_UNBIASED_ENSEMBLE",
                "O_7X_UNBIASED_TOKEN_FUSION",
                "O_RELATION_EXPERT_TOKEN_FUSION",
            ],
            "monolithic_grid_axes": [
                "particle_hidden_width",
                "feed_forward_expansion",
                "attention_head_count",
                "particle_block_count",
                "class_block_count",
            ],
            "head_count_must_divide_width": True,
            "complete_deployed_graph_target_required": True,
            "analytical_flop_batches": [1, 128],
            "latency_is_diagnostic": True,
            "long_baseline_uses_label_presentation_ledger": True,
        }
    )


def select_monolithic_capacity_controls(
    *,
    target_parameters: int,
    target_flops_batch1: float,
    target_flops_batch128: float,
    candidates: Sequence[Mapping[str, Any]],
    domain: str = "offline",
    target_complete_graph_sha256: str | None = None,
) -> dict[str, Any]:
    if domain not in {"offline", "hlt"}:
        raise ValueError("capacity-control domain differs")
    if min(int(target_parameters), float(target_flops_batch1), float(target_flops_batch128)) <= 0:
        raise ValueError("capacity target totals must be positive")
    checked = []
    for row in candidates:
        configuration = tuple(row["configuration"])
        width, expansion, heads, particle_blocks, class_blocks = map(
            int, configuration
        )
        if width % heads:
            raise ValueError("capacity candidate width is not divisible by heads")
        parameters = int(row["parameter_count"])
        flops1 = float(row["inference_flops_batch1"])
        flops128 = float(row["inference_flops_batch128"])
        if min(parameters, flops1, flops128) <= 0:
            raise ValueError("capacity candidate totals must be positive")
        checked.append(
            {
                **dict(row),
                "configuration": list(configuration),
                "parameter_mismatch": abs(parameters - target_parameters),
                "flops_batch1_mismatch": abs(flops1 - target_flops_batch1),
                "flops_batch128_mismatch": abs(flops128 - target_flops_batch128),
                "depth_plus_width": width + particle_blocks + class_blocks,
            }
        )
    if not checked:
        raise ValueError("capacity selector requires candidates")
    parameter_match = min(
        checked,
        key=lambda row: (
            row["parameter_mismatch"],
            row["flops_batch1_mismatch"],
            row["depth_plus_width"],
            tuple(row["configuration"]),
        ),
    )
    flop_match = min(
        checked,
        key=lambda row: (
            row["flops_batch1_mismatch"],
            row["parameter_mismatch"],
            row["depth_plus_width"],
            tuple(row["configuration"]),
        ),
    )
    prefix = "O" if domain == "offline" else "H"
    target = {
        "parameter_count": int(target_parameters),
        "inference_flops_batch1": float(target_flops_batch1),
        "inference_flops_batch128": float(target_flops_batch128),
        "includes": [
            "expert_encoders",
            "summary_tokenizers",
            "dimension_projections",
            "uncertainty_or_reliability_heads",
            "token_refiner",
            "final_consumer",
        ],
        "excludes": ["training_only_teachers", "target_caches"],
    }
    if target_complete_graph_sha256 is not None:
        target["complete_graph_sha256"] = require_sha256(
            target_complete_graph_sha256,
            name="target_complete_graph_sha256",
        )
        target["includes"].append(
            "deployable_frozen_offline_heads_and_fusion"
        )
    payload = {
        "contract": (
            CAPACITY_CONTROL_CONTRACT
            if domain == "offline"
            and target_complete_graph_sha256 is None
            else "retb_complete_graph_capacity_controls_v2"
        ),
        "schema_version": (
            1
            if domain == "offline"
            and target_complete_graph_sha256 is None
            else 2
        ),
        "target": target,
        f"{prefix}_MONO_PARAM": parameter_match,
        f"{prefix}_MONO_FLOP": flop_match,
        "within_5_percent": {
                f"{prefix}_MONO_PARAM_parameters": (
                    parameter_match["parameter_mismatch"] / target_parameters <= 0.05
                ),
                f"{prefix}_MONO_PARAM_batch1_flops": (
                    parameter_match["flops_batch1_mismatch"] / target_flops_batch1 <= 0.05
                ),
                f"{prefix}_MONO_FLOP_parameters": (
                    flop_match["parameter_mismatch"] / target_parameters <= 0.05
                ),
                f"{prefix}_MONO_FLOP_batch1_flops": (
                    flop_match["flops_batch1_mismatch"] / target_flops_batch1 <= 0.05
                ),
        },
        "measured_latency_used_for_selection": False,
    }
    if payload["schema_version"] == 2:
        payload["domain"] = domain
    return with_content_hash(payload)


def build_offline_long_exposure_ledger(
    *,
    component_rows: Sequence[Mapping[str, Any]],
    obase_effective_batch_size: int = 128,
) -> dict[str, Any]:
    if int(obase_effective_batch_size) <= 0:
        raise ValueError("O_BASE effective batch size must be positive")
    rows = []
    total = 0
    for row in component_rows:
        if row.get("component_kind") not in {
            "offline_expert",
            "primary_frozen_fusion",
            "attachment_pretraining",
        }:
            raise ValueError("offline label-exposure component is unregistered")
        presentations = int(row["labeled_example_presentations"])
        if presentations <= 0:
            raise ValueError("label presentations must be positive")
        total += presentations
        rows.append(
            {
                "component_id": str(row["component_id"]),
                "component_kind": row["component_kind"],
                "labeled_example_presentations": presentations,
                "parent_sha256": require_sha256(
                    row["parent_sha256"],
                    name=f"component_rows.{row['component_id']}.parent_sha256",
                ),
            }
        )
    if not rows:
        raise ValueError("offline long-exposure ledger is empty")
    updates = math.ceil(total / int(obase_effective_batch_size))
    return with_content_hash(
        {
            "contract": OFFLINE_LONG_EXPOSURE_CONTRACT,
            "schema_version": 1,
            "component_rows": rows,
            "total_labeled_example_presentations": total,
            "obase_effective_batch_size": int(obase_effective_batch_size),
            "optimizer_update_budget": updates,
            "rounding": "ceil",
            "same_warmup_fraction_and_cosine_schedule": True,
            "early_stopping": False,
        }
    )


__all__ = [
    "CAPACITY_CONTROL_CONTRACT",
    "OFFLINE_LONG_EXPOSURE_CONTRACT",
    "build_offline_long_exposure_ledger",
    "build_capacity_control_registry",
    "select_monolithic_capacity_controls",
]

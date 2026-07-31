"""Deterministic complete-graph capacity matching and label-exposure controls."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import require_sha256, validate_content_hash, with_content_hash


CAPACITY_CONTROL_CONTRACT = "retb_complete_graph_capacity_controls_v1"
OFFLINE_LONG_EXPOSURE_CONTRACT = "retb_offline_long_exposure_ledger_v1"
OFFLINE_CAPACITY_EXECUTION_CONTRACT = (
    "retb_offline_capacity_execution_registry_v1"
)
OFFLINE_CAPACITY_REGISTRATION_CONTRACT = (
    "retb_offline_capacity_control_registration_v2"
)

OFFLINE_CAPACITY_CONTROL_ORDER = (
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
)


def build_offline_capacity_execution_registry() -> dict[str, Any]:
    """Freeze how every scientific capacity control is actually produced."""

    return with_content_hash(
        {
            "contract": OFFLINE_CAPACITY_EXECUTION_CONTRACT,
            "schema_version": 1,
            "control_order": list(OFFLINE_CAPACITY_CONTROL_ORDER),
            "recipes": {
                "O_BASE": {
                    "kind": "single_particle_transformer",
                    "pair_relations": ["base4"],
                    "classifier": "ordinary_weaver_class_attention",
                    "training": "fixed_40_epoch_CE",
                },
                "O_WIDE": {
                    "kind": "single_particle_transformer",
                    "pair_relations": ["base4"],
                    "pair_stem": "locked_RPT_BASE_WIDE_MAX",
                    "classifier": "ordinary_weaver_class_attention",
                    "training": "fixed_40_epoch_CE",
                },
                "O_MONO_PARAM": {
                    "kind": "selected_monolithic_particle_transformer",
                    "selector": "minimum_complete_graph_parameter_mismatch",
                    "pair_relations": ["base4"],
                    "training": "fixed_40_epoch_CE",
                },
                "O_MONO_FLOP": {
                    "kind": "selected_monolithic_particle_transformer",
                    "selector": "minimum_complete_graph_batch1_FLOP_mismatch",
                    "pair_relations": ["base4"],
                    "training": "fixed_40_epoch_CE",
                },
                "O_BASE_LONG": {
                    "kind": "single_particle_transformer",
                    "pair_relations": ["base4"],
                    "classifier": "ordinary_weaver_class_attention",
                    "training": "fixed_serialized_optimizer_update_budget",
                    "early_stopping": False,
                },
                "O_FULLREL": {
                    "kind": "single_particle_transformer",
                    "pair_relations": [
                        "base4",
                        "PT",
                        "TRACK",
                        "PID",
                        "CHARGE",
                        "DENSITY",
                        "REGION",
                    ],
                    "classifier": "ordinary_weaver_class_attention",
                    "training": "fixed_40_epoch_CE",
                },
                "O_GROUPED_HEAD_REL": {
                    "kind": "single_particle_transformer",
                    "pair_relations": [
                        "head0_base4",
                        "head1_base4",
                        "head2_base4_plus_PT",
                        "head3_base4_plus_TRACK",
                        "head4_base4_plus_PID",
                        "head5_base4_plus_CHARGE",
                        "head6_base4_plus_DENSITY",
                        "head7_base4_plus_REGION",
                    ],
                    "softmax_scope": "independent_per_head",
                    "training": "fixed_40_epoch_CE",
                },
                "O_7X_UNBIASED_ENSEMBLE": {
                    "kind": "seven_independently_trained_BASE4_experts",
                    "member_seeds": [101, 202, 303, 404, 505, 606, 707],
                    "combiner": "arithmetic_mean_logits",
                    "member_training": "fixed_40_epoch_CE",
                },
                "O_7X_UNBIASED_TOKEN_FUSION": {
                    "kind": "seven_independently_trained_BASE4_token_experts",
                    "member_seeds": [101, 202, 303, 404, 505, 606, 707],
                    "topology": "selected_relation_expert_token_topology",
                    "fusion": "fresh_F_TOKEN_TRANSFORMER",
                    "member_training": "fixed_40_epoch_CE",
                    "fusion_training": "fixed_40_epoch_CE",
                },
                "O_RELATION_EXPERT_TOKEN_FUSION": {
                    "kind": "selected_seven_relation_expert_token_graph",
                    "expert_order": [
                        "BASE4",
                        "PT",
                        "TRACK",
                        "PID",
                        "CHARGE",
                        "DENSITY",
                        "REGION",
                    ],
                    "fusion": "selected_frozen_token_fusion",
                    "training": "authenticated_Stage_C_confirmations",
                },
            },
            "required_splits": ["model_train", "val_stop", "val_design"],
            "complete_graph_profile_batches": [1, 128],
            "measured_latency_used_for_selection": False,
            "all_rows_execute_when_scientifically_negative": True,
        }
    )


def build_offline_capacity_control_registration(
    *,
    control_id: str,
    execution_registry_sha256: str,
    checkpoint_hashes: Sequence[str],
    parameter_count: int,
    inference_flops_batch1: int,
    inference_flops_batch128: int,
    profile_sha256: str,
    labeled_example_presentations: int,
    label_exposure_ledger_sha256: str,
    training_artifact_hashes: Sequence[str],
    val_design_prediction_sha256: str,
    val_design_metrics_sha256: str,
    fixed_budget_completed: bool,
) -> dict[str, Any]:
    if control_id not in OFFLINE_CAPACITY_CONTROL_ORDER:
        raise ValueError("offline capacity control ID is unknown")
    if not checkpoint_hashes or not training_artifact_hashes:
        raise ValueError("capacity control lacks executed training lineage")
    if (
        min(
            int(parameter_count),
            int(inference_flops_batch1),
            int(inference_flops_batch128),
            int(labeled_example_presentations),
        )
        <= 0
        or int(inference_flops_batch128)
        != 128 * int(inference_flops_batch1)
        or not bool(fixed_budget_completed)
    ):
        raise ValueError("capacity control execution evidence is incomplete")
    return with_content_hash(
        {
            "contract": OFFLINE_CAPACITY_REGISTRATION_CONTRACT,
            "schema_version": 2,
            "control_id": str(control_id),
            "execution_registry_sha256": require_sha256(
                execution_registry_sha256,
                name="execution_registry_sha256",
            ),
            "checkpoint_hashes": [
                require_sha256(value, name="checkpoint_hashes")
                for value in checkpoint_hashes
            ],
            "checkpoint_sha256": (
                require_sha256(checkpoint_hashes[0], name="checkpoint_sha256")
                if len(checkpoint_hashes) == 1
                else None
            ),
            "complete_graph_capacity": {
                "parameter_count": int(parameter_count),
                "inference_flops_batch1": int(inference_flops_batch1),
                "inference_flops_batch128": int(inference_flops_batch128),
                "profile_sha256": require_sha256(
                    profile_sha256, name="profile_sha256"
                ),
            },
            "label_exposure": {
                "labeled_example_presentations": int(
                    labeled_example_presentations
                ),
                "ledger_sha256": require_sha256(
                    label_exposure_ledger_sha256,
                    name="label_exposure_ledger_sha256",
                ),
            },
            "training_artifact_hashes": [
                require_sha256(value, name="training_artifact_hashes")
                for value in training_artifact_hashes
            ],
            "val_design_prediction_sha256": require_sha256(
                val_design_prediction_sha256,
                name="val_design_prediction_sha256",
            ),
            "val_design_metrics_sha256": require_sha256(
                val_design_metrics_sha256,
                name="val_design_metrics_sha256",
            ),
            "fixed_budget_completed": True,
            "performance_based_termination": False,
            "complete_graph_profiled": True,
            "label_exposure_authenticated": True,
        }
    )


def validate_offline_capacity_control_registration(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=OFFLINE_CAPACITY_REGISTRATION_CONTRACT
    )
    capacity = payload.get("complete_graph_capacity", {})
    exposure = payload.get("label_exposure", {})
    if (
        payload.get("control_id") not in OFFLINE_CAPACITY_CONTROL_ORDER
        or payload.get("fixed_budget_completed") is not True
        or payload.get("performance_based_termination") is not False
        or payload.get("complete_graph_profiled") is not True
        or payload.get("label_exposure_authenticated") is not True
        or min(
            int(capacity.get("parameter_count", 0)),
            int(capacity.get("inference_flops_batch1", 0)),
            int(capacity.get("inference_flops_batch128", 0)),
            int(exposure.get("labeled_example_presentations", 0)),
        )
        <= 0
        or int(capacity["inference_flops_batch128"])
        != 128 * int(capacity["inference_flops_batch1"])
        or not payload.get("checkpoint_hashes")
        or not payload.get("training_artifact_hashes")
    ):
        raise ValueError("offline capacity registration differs")
    for name in (
        "execution_registry_sha256",
        "val_design_prediction_sha256",
        "val_design_metrics_sha256",
    ):
        require_sha256(payload.get(name), name=name)
    require_sha256(capacity.get("profile_sha256"), name="profile_sha256")
    require_sha256(exposure.get("ledger_sha256"), name="ledger_sha256")
    for name in ("checkpoint_hashes", "training_artifact_hashes"):
        for value in payload[name]:
            require_sha256(value, name=name)
    expected_checkpoint = (
        payload["checkpoint_hashes"][0]
        if len(payload["checkpoint_hashes"]) == 1
        else None
    )
    if payload.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError("capacity primary checkpoint semantics differ")
    return digest


def build_capacity_control_registry() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": "retb_capacity_control_registry_v1",
            "schema_version": 1,
            "offline_controls": list(OFFLINE_CAPACITY_CONTROL_ORDER),
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
    "OFFLINE_CAPACITY_CONTROL_ORDER",
    "OFFLINE_CAPACITY_EXECUTION_CONTRACT",
    "OFFLINE_CAPACITY_REGISTRATION_CONTRACT",
    "OFFLINE_LONG_EXPOSURE_CONTRACT",
    "build_offline_capacity_control_registration",
    "build_offline_capacity_execution_registry",
    "build_offline_long_exposure_ledger",
    "build_capacity_control_registry",
    "select_monolithic_capacity_controls",
    "validate_offline_capacity_control_registration",
]

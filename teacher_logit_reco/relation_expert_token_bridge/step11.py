"""Step-11 joint bridge optimization registries and immutable contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    bind_source,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .joint_bridge import JOINT_INPUT_POLICY, JOINT_VARIANTS
from .joint_bridge_training import VARIANT_LOSS_WEIGHTS
from .predictor_bundle import (
    PIPELINE_SEEDS,
    PREDICTOR_BUNDLE_LOCK_CONTRACT,
)
from .registry import EXPERT_ORDER


STAGE_J_POLICY_CONTRACT = "retb_stage_j_joint_bridge_policy_v1"
STAGE_J_REGISTRY_CONTRACT = "retb_stage_j_joint_bridge_registry_v1"
STAGE_J_RUN_CONTRACT = "retb_materialized_stage_j_run_v1"
J4_SELECTION_CONTRACT = "retb_j4_block_selection_v1"
STEP11_BUNDLE_CONTRACT = "retb_step11_joint_bridge_bundle_v1"
STEP11_REPORT_CONTRACT = "retb_step11_report_v1"

SEMANTIC_LABELS = {
    "J0_INDEPENDENT": "FAITHFUL_FROZEN_EVIDENCE_RECONSTRUCTION",
    "J1_SHARED_CONTEXT": "COORDINATED_TOKEN_RECONSTRUCTION",
    "J2_COUPLED_DECODER": "COUPLED_TOKEN_RECONSTRUCTION",
    "J3_INDEPENDENT_PLUS_ADAPTER": "DOWNSTREAM_ADAPTATION_CONTROL",
    "J4_BRIDGE_FINETUNE": "BRIDGE_TUNED_PRIVILEGED_RECONSTRUCTION",
    "J5_END_TO_END": "END_TO_END_MAXIMUM_PERFORMANCE",
}


def build_stage_j_policy() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": STAGE_J_POLICY_CONTRACT,
            "schema_version": 1,
            "variants": list(JOINT_VARIANTS),
            "input_policy": JOINT_INPUT_POLICY,
            "common_view_invariant": {
                "one_canonical_identity": True,
                "one_R_MULTI_replica_per_epoch_identity": True,
                "one_degraded_HLT_particle_array": True,
                "all_seven_experts_consume_same_array": True,
                "expert_specific_relations_rebuilt_from_same_array": True,
                "cross_replica_bank_mixing_permitted": False,
            },
            "J0": {
                "selected_predictors_frozen": True,
                "offline_fusion_frozen": True,
                "joint_training": False,
            },
            "J1": {
                "shared_HLT_context_computed_once": True,
                "independent_decoders_retained": True,
                "loss": "mean_selected_predictor_objective_plus_all_bank_fusion_KD",
            },
            "J2": {
                "one_decoder": True,
                "queries": "all_expert_slot_queries_with_expert_and_slot_identity",
                "query_self_attention_across_experts": True,
                "non_autoregressive": True,
            },
            "J3": {
                "selected_predictors_frozen": True,
                "train_only_deployable_residual_adapter": True,
                "residual_scalar_initialization": 0.0,
            },
            "HE_BRIDGE_TUNED": {
                "unfrozen": [
                    "HLT_summary_tokenizer",
                    "HLT_relation_encoder_and_dual_path_gates",
                    "final_N_particle_attention_blocks",
                    "predictor",
                ],
                "final_particle_block_candidates": [2, 4],
                "simplicity_control": 2,
                "learning_rates": {
                    "HLT_particle_relation": 5.0e-5,
                    "HLT_tokenizer": 1.0e-4,
                    "predictor": 2.0e-4,
                },
                "offline_targets_heads_and_fusion_frozen": True,
                "epoch_selection_split": "val_stop",
                "block_count_selection_split": "val_design",
            },
            "J5": {
                "initialization": "selected_J4",
                "all_HLT_evidence_encoders_trainable": True,
                "all_predictors_or_coupled_decoder_trainable": True,
                "deployable_final_fusion_trainable": True,
                "offline_target_encoders_and_caches_frozen": True,
                "maximum_performance_claim_only": True,
            },
            "loss_weights": VARIANT_LOSS_WEIGHTS,
            "semantic_labels": SEMANTIC_LABELS,
            "task_distillation_definition": (
                "any_W_LOGIT_ONLY_or_T1_TASK_BRIDGE_predictor_lineage"
            ),
            "semantic_categories_remain_separate_when_all_rows_lose": True,
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "fixed_epochs": 40,
            "performance_based_termination": False,
            "stack_val_permitted": False,
            "final_test_permitted": False,
        }
    )


def validate_stage_j_policy(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_J_POLICY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_stage_j_policy()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-J policy semantics differ")
    return digest


def _validate_predictor_lock(lock: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        lock, expected_contract=PREDICTOR_BUNDLE_LOCK_CONTRACT
    )
    if (
        lock.get("expert_order") != list(EXPERT_ORDER)
        or set(lock.get("candidate_hashes", {})) != set(EXPERT_ORDER)
        or set(lock.get("seed_specific_artifacts", {}))
        != {str(seed) for seed in PIPELINE_SEEDS}
        or not lock.get("configuration_shared_across_pipeline_seeds")
        or lock.get("per_seed_selection_permitted")
        or not lock.get("locked_before_joint_training")
    ):
        raise ValueError("Stage-J predictor bundle lock differs")
    return digest


def build_stage_j_registry(
    *,
    predictor_bundle_lock: Mapping[str, Any],
    policy_sha256: str | None = None,
) -> dict[str, Any]:
    lock_sha = _validate_predictor_lock(predictor_bundle_lock)
    rows = []
    for variant in JOINT_VARIANTS:
        block_rows = (
            (2, 4) if variant == "J4_BRIDGE_FINETUNE" else (None,)
        )
        for blocks in block_rows:
            for seed in PIPELINE_SEEDS:
                suffix = (
                    "" if blocks is None else f"_N{int(blocks)}"
                )
                rows.append(
                    {
                        "run_id": f"RETB_{variant}_S{seed}{suffix}",
                        "stage": "J",
                        "variant": variant,
                        "pipeline_seed": seed,
                        "final_particle_blocks": blocks,
                        "input_policy": JOINT_INPUT_POLICY,
                        "semantic_label": SEMANTIC_LABELS[variant],
                        "role": (
                            "reference_baseline"
                            if variant == "J0_INDEPENDENT"
                            else "scientific_candidate"
                        ),
                        "trainable": variant != "J0_INDEPENDENT",
                        "predictor_bundle_lock_sha256": lock_sha,
                    }
                )
    selected_task_distilled = any(
        (
            row.get("configuration", {}).get("objective_id")
            == "W_LOGIT_ONLY"
            or row.get("target_mode") == "T1_TASK_BRIDGE"
        )
        for row in predictor_bundle_lock.get(
            "selected_candidate_descriptors", {}
        ).values()
    )
    return with_content_hash(
        {
            "contract": STAGE_J_REGISTRY_CONTRACT,
            "schema_version": 1,
            "policy_sha256": (
                build_stage_j_policy()["content_hash"]
                if policy_sha256 is None
                else require_sha256(policy_sha256, name="policy_sha256")
            ),
            "predictor_bundle_lock_sha256": lock_sha,
            "input_policy": JOINT_INPUT_POLICY,
            "membership_count": len(rows),
            "rows": rows,
            "semantic_comparison_registry": {
                "FAITHFUL": ["J0_INDEPENDENT"],
                "COORDINATED": [
                    "J1_SHARED_CONTEXT",
                    "J2_COUPLED_DECODER",
                ],
                "LOGIT_DISTILLED": [
                    "locked_Step9_W_LOGIT_ONLY_and_T1_TASK_controls"
                ],
                "BRIDGE_TUNED": ["J4_BRIDGE_FINETUNE"],
                "END_TO_END": ["J5_END_TO_END"],
            },
            "selected_bundle_is_task_distilled": selected_task_distilled,
            "all_categories_reported_even_when_worse": True,
            "J4_block_selection": {
                "candidates": [2, 4],
                "split": "val_design",
                "simplicity_control": 2,
                "all_three_pipeline_seeds_required": True,
            },
            "J5_materialization_requires_J4_selection_lock": True,
            "performance_based_termination": False,
        }
    )


def validate_stage_j_registry(
    payload: Mapping[str, Any],
    *,
    predictor_bundle_lock: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_J_REGISTRY_CONTRACT
    )
    expected = build_stage_j_registry(
        predictor_bundle_lock=predictor_bundle_lock,
        policy_sha256=payload.get("policy_sha256"),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-J registry semantics differ")
    return digest


def materialize_stage_j_run(
    *,
    run_id: str,
    variant: str,
    pipeline_seed: int,
    final_particle_blocks: int | None,
    predictor_bundle_lock_sha256: str,
    step11_bundle_sha256: str,
    parent_hashes: Mapping[str, str],
    semantic_label: str,
) -> dict[str, Any]:
    required = {
        "model_train_identity_manifest",
        "val_stop_identity_manifest",
        "val_design_identity_manifest",
        "val_design_label_manifest",
        "model_train_R_MULTI_view_cache",
        "val_stop_R_MULTI_view_cache",
        "val_design_fixed_view_cache",
        "offline_target_cache",
        "target_normalizer_set",
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
        "selected_predictor_seed_artifacts",
    }
    if variant in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}:
        required.add("selected_HLT_expert_seed_artifacts")
    if variant == "J5_END_TO_END":
        required.update(
            {
                "j4_block_selection",
                "selected_J4_bridge_initialization",
            }
        )
    if (
        variant not in JOINT_VARIANTS
        or int(pipeline_seed) not in PIPELINE_SEEDS
        or (
            variant == "J4_BRIDGE_FINETUNE"
            and final_particle_blocks not in {2, 4}
        )
        or (
            variant != "J4_BRIDGE_FINETUNE"
            and final_particle_blocks is not None
        )
        or semantic_label != SEMANTIC_LABELS[variant]
        or set(parent_hashes) != required
    ):
        raise ValueError("materialized Stage-J run semantics differ")
    return with_content_hash(
        {
            "contract": STAGE_J_RUN_CONTRACT,
            "schema_version": 1,
            "run_id": str(run_id),
            "variant": variant,
            "pipeline_seed": int(pipeline_seed),
            "final_particle_blocks": final_particle_blocks,
            "input_policy": JOINT_INPUT_POLICY,
            "semantic_label": semantic_label,
            "role": (
                "reference_baseline"
                if variant == "J0_INDEPENDENT"
                else "scientific_candidate"
            ),
            "predictor_bundle_lock_sha256": require_sha256(
                predictor_bundle_lock_sha256,
                name="predictor_bundle_lock_sha256",
            ),
            "step11_bundle_sha256": require_sha256(
                step11_bundle_sha256, name="step11_bundle_sha256"
            ),
            "parent_hashes": {
                name: require_sha256(
                    parent_hashes[name], name=f"parent_hashes.{name}"
                )
                for name in sorted(parent_hashes)
            },
            "one_identity_one_R_MULTI_view_required": True,
            "fixed_budget": variant != "J0_INDEPENDENT",
            "performance_based_termination": False,
        }
    )


def validate_materialized_stage_j_run(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_J_RUN_CONTRACT
    )
    expected = materialize_stage_j_run(
        run_id=payload.get("run_id"),
        variant=payload.get("variant"),
        pipeline_seed=int(payload.get("pipeline_seed", -1)),
        final_particle_blocks=payload.get("final_particle_blocks"),
        predictor_bundle_lock_sha256=payload.get(
            "predictor_bundle_lock_sha256"
        ),
        step11_bundle_sha256=payload.get("step11_bundle_sha256"),
        parent_hashes=payload.get("parent_hashes", {}),
        semantic_label=payload.get("semantic_label"),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("materialized Stage-J run semantics differ")
    return digest


def select_j4_block_count(
    rows: Sequence[Mapping[str, Any]],
    *,
    predictor_bundle_lock_sha256: str,
    label_manifest_sha256: str,
) -> dict[str, Any]:
    expected = {
        (blocks, seed)
        for blocks in (2, 4)
        for seed in PIPELINE_SEEDS
    }
    seen = {
        (int(row.get("final_particle_blocks", -1)), int(row.get("pipeline_seed", -1)))
        for row in rows
    }
    if seen != expected:
        raise ValueError("J4 block selector coverage differs")
    aggregates = []
    for blocks in (2, 4):
        selected = [
            row for row in rows if int(row["final_particle_blocks"]) == blocks
        ]
        for row in selected:
            if (
                row.get("variant") != "J4_BRIDGE_FINETUNE"
                or row.get("split") != "val_design"
                or not all(
                    math.isfinite(float(row.get(name, math.nan)))
                    for name in (
                        "accuracy",
                        "cross_entropy",
                        "normalized_token_error",
                    )
                )
                or int(row.get("inference_flops", 0)) <= 0
                or int(row.get("parameter_count", 0)) <= 0
            ):
                raise ValueError("J4 block selector row differs")
        aggregates.append(
            {
                "final_particle_blocks": blocks,
                "mean_accuracy": sum(
                    float(row["accuracy"]) for row in selected
                )
                / len(selected),
                "mean_cross_entropy": sum(
                    float(row["cross_entropy"]) for row in selected
                )
                / len(selected),
                "mean_normalized_token_error": sum(
                    float(row["normalized_token_error"]) for row in selected
                )
                / len(selected),
                "inference_flops": max(
                    int(row["inference_flops"]) for row in selected
                ),
                "parameter_count": max(
                    int(row["parameter_count"]) for row in selected
                ),
                "registration_hashes": {
                    str(row["pipeline_seed"]): require_sha256(
                        row["registration_sha256"],
                        name="registration_sha256",
                    )
                    for row in selected
                },
            }
        )
    maximum = max(row["mean_accuracy"] for row in aggregates)
    eligible = [
        row
        for row in aggregates
        if maximum - row["mean_accuracy"] <= 1.0e-4 + 1.0e-15
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["mean_cross_entropy"],
            row["mean_normalized_token_error"],
            row["inference_flops"],
            row["parameter_count"],
            row["final_particle_blocks"],
        ),
    )
    return with_content_hash(
        {
            "contract": J4_SELECTION_CONTRACT,
            "schema_version": 1,
            "predictor_bundle_lock_sha256": require_sha256(
                predictor_bundle_lock_sha256,
                name="predictor_bundle_lock_sha256",
            ),
            "label_manifest_sha256": require_sha256(
                label_manifest_sha256, name="label_manifest_sha256"
            ),
            "split": "val_design",
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "ranking": [
                "maximum_mean_accuracy",
                "global_window_0.0001",
                "lower_mean_cross_entropy",
                "lower_mean_normalized_token_error",
                "lower_inference_FLOPs",
                "lower_parameter_count",
                "smaller_final_particle_block_count",
            ],
            "candidates": aggregates,
            "selected_final_particle_blocks": selected[
                "final_particle_blocks"
            ],
            "selected_registration_hashes": selected[
                "registration_hashes"
            ],
            "simplicity_control": 2,
            "all_candidates_worse_than_baseline_does_not_block": True,
            "performance_based_termination": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
    )


def build_step11_bundle(
    *,
    campaign_spec_sha256: str,
    step10_bundle_sha256: str,
    predictor_bundle_lock: Mapping[str, Any],
    global_determinism_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    policy = bind_source(
        build_stage_j_policy(), source_snapshot=source_snapshot
    )
    registry = bind_source(
        build_stage_j_registry(
            predictor_bundle_lock=predictor_bundle_lock,
            policy_sha256=policy["content_hash"],
        ),
        source_snapshot=source_snapshot,
    )
    lock_sha = _validate_predictor_lock(predictor_bundle_lock)
    manifest = bind_source(
        with_content_hash(
            {
                "contract": STEP11_BUNDLE_CONTRACT,
                "schema_version": 1,
                "parents": {
                    "campaign_spec": require_sha256(
                        campaign_spec_sha256, name="campaign_spec_sha256"
                    ),
                    "step10_bundle": require_sha256(
                        step10_bundle_sha256, name="step10_bundle_sha256"
                    ),
                    "predictor_bundle_lock": lock_sha,
                    "global_determinism": require_sha256(
                        global_determinism_sha256,
                        name="global_determinism_sha256",
                    ),
                },
                "artifact_hashes": {
                    "stage_j_policy": policy["content_hash"],
                    "stage_j_registry": registry["content_hash"],
                },
                "J0_through_J5_registered": True,
                "HE_BRIDGE_TUNED_registered": True,
                "one_identity_one_R_MULTI_view_required": True,
                "semantic_claims_separated": True,
                "performance_based_termination": False,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    report = bind_source(
        with_content_hash(
            {
                "contract": STEP11_REPORT_CONTRACT,
                "schema_version": 1,
                "step11_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "immutable_predictor_bundle_parent": True,
                    "shared_common_view_contract": True,
                    "independent_shared_and_coupled_graphs": True,
                    "bridge_tuned_N2_N4_screen": True,
                    "J5_requires_J4_lock": True,
                    "faithful_logit_distilled_bridge_and_end_to_end_labels": True,
                    "all_negative_campaign_continues": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {
        "stage_j_policy": policy,
        "stage_j_registry": registry,
        "step11_bundle": manifest,
        "step11_report": report,
    }


def validate_step11_bundle(
    bundle: Mapping[str, Mapping[str, Any]],
    *,
    predictor_bundle_lock: Mapping[str, Any],
) -> str:
    if set(bundle) != {
        "stage_j_policy",
        "stage_j_registry",
        "step11_bundle",
        "step11_report",
    }:
        raise ValueError("Step-11 bundle members differ")
    policy_sha = validate_stage_j_policy(bundle["stage_j_policy"])
    registry_sha = validate_stage_j_registry(
        bundle["stage_j_registry"],
        predictor_bundle_lock=predictor_bundle_lock,
    )
    digest = validate_content_hash(
        bundle["step11_bundle"], expected_contract=STEP11_BUNDLE_CONTRACT
    )
    validate_content_hash(
        bundle["step11_report"], expected_contract=STEP11_REPORT_CONTRACT
    )
    manifest = bundle["step11_bundle"]
    if (
        manifest["artifact_hashes"]
        != {
            "stage_j_policy": policy_sha,
            "stage_j_registry": registry_sha,
        }
        or bundle["stage_j_registry"]["policy_sha256"] != policy_sha
        or manifest["parents"]["predictor_bundle_lock"]
        != _validate_predictor_lock(predictor_bundle_lock)
        or not manifest["J0_through_J5_registered"]
        or not manifest["HE_BRIDGE_TUNED_registered"]
        or not manifest["one_identity_one_R_MULTI_view_required"]
        or not manifest["semantic_claims_separated"]
        or manifest["performance_based_termination"]
        or manifest["stack_val_consumed"]
        or manifest["final_test_consumed"]
        or bundle["step11_report"]["step11_bundle_sha256"] != digest
        or bundle["step11_report"]["scientific_results_inspected"]
        or len({repr(row.get("source")) for row in bundle.values()}) != 1
    ):
        raise ValueError("Step-11 bundle semantics differ")
    return digest


def publish_step11_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Mapping[str, Any]],
    predictor_bundle_lock: Mapping[str, Any],
) -> dict[str, Any]:
    digest = validate_step11_bundle(
        bundle, predictor_bundle_lock=predictor_bundle_lock
    )
    root = Path(campaign_root)
    paths = {
        "stage_j_policy": root / "registry" / "retb_stage_j_policy.json",
        "stage_j_registry": root / "registry" / "retb_stage_j_registry.json",
        "step11_bundle": (
            root / "registry" / "retb_step11_joint_bridge_bundle.json"
        ),
        "step11_report": root / "reports" / "retb_step11_report.json",
    }
    return {
        "step11_bundle_sha256": digest,
        "publications": {
            name: write_immutable_json(paths[name], bundle[name])
            for name in paths
        },
    }


__all__ = [
    "J4_SELECTION_CONTRACT",
    "SEMANTIC_LABELS",
    "STAGE_J_POLICY_CONTRACT",
    "STAGE_J_REGISTRY_CONTRACT",
    "STAGE_J_RUN_CONTRACT",
    "STEP11_BUNDLE_CONTRACT",
    "STEP11_REPORT_CONTRACT",
    "build_stage_j_policy",
    "build_stage_j_registry",
    "build_step11_bundle",
    "materialize_stage_j_run",
    "publish_step11_bundle",
    "select_j4_block_count",
    "validate_materialized_stage_j_run",
    "validate_stage_j_policy",
    "validate_stage_j_registry",
    "validate_step11_bundle",
]

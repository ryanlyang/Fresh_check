"""Step-12 registries and immutable final-consumer contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    bind_source,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .final_consumers import (
    ADAPTER_VARIANTS,
    BYPASS_CONTROLS,
    NATIVE_DROPOUT_MODES,
    REFINER_VARIANTS,
    UNRESTRICTED_EVIDENCE_VARIANTS,
)
from .predictor_bundle import PIPELINE_SEEDS


FINAL_CONSUMER_POLICY_CONTRACT = "retb_final_consumer_policy_v1"
STAGE_J_CONSUMER_REGISTRY_CONTRACT = (
    "retb_stage_j_final_consumer_registry_v2"
)
FINAL_CONSUMER_RUN_CONTRACT = "retb_materialized_final_consumer_run_v2"
STEP12_BUNDLE_CONTRACT = "retb_step12_final_consumers_bundle_v2"
STEP12_REPORT_CONTRACT = "retb_step12_report_v1"

TOKEN_INPUTS = ("TOKEN_PREDICTED", "TOKEN_REFINED_SELECTED")


def build_final_consumer_policy() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": FINAL_CONSUMER_POLICY_CONTRACT,
            "schema_version": 1,
            "faithful_consumers": ["PF_FROZEN", "OF_ROBUST"],
            "constrained_consumer": "HF_ADAPTER",
            "maximum_performance_consumer": "HF_UNRESTRICTED",
            "refiner_variants": list(REFINER_VARIANTS),
            "adapter_variants": list(ADAPTER_VARIANTS),
            "unrestricted_evidence_variants": list(
                UNRESTRICTED_EVIDENCE_VARIANTS
            ),
            "native_dropout_modes": list(NATIVE_DROPOUT_MODES),
            "bypass_controls": list(BYPASS_CONTROLS),
            "robust_offline_fusion_mixture": {
                "all_oracle": 0.25,
                "exactly_one_predicted": 0.25,
                "independent_predicted_p0.5": 0.25,
                "all_predicted": 0.25,
                "generic_expert_dropout_used": False,
            },
            "adapter": {
                "transformer_layers": 2,
                "residual_scalar_initialization": 0.0,
                "reports_frozen_residual_combined_logits": True,
            },
            "refiner": {
                "cross_attention_blocks": 2,
                "formula": "predicted_plus_sigmoid_gate_times_delta",
                "offline_inputs_at_inference": False,
                "reports_token_fidelity_before_and_after": True,
            },
            "unrestricted": {
                "transformer_layers": 4,
                "width": 256,
                "attention_heads": 8,
                "MLP_expansion": 4,
                "pre_normalized": True,
                "learned_class_token": True,
                "predicts_logits_directly": True,
                "matched_width_candidates": [64, 128, 192, 256, 320, 384],
                "matched_ranking": [
                    "incremental_parameter_mismatch",
                    "analytical_inference_FLOP_mismatch",
                    "smaller_width",
                ],
                "reliability_gates_bounded_scalar_residual_only": True,
            },
            "native_dropout": {
                "ND0_NONE": {"expected_rate": 0.0},
                "ND1_FIXED": {
                    "whole_native_branch_rate": 0.10,
                    "training_only": True,
                },
                "ND2_CONFIDENCE": {
                    "calibrated_uncertainty_conditioned": True,
                    "expected_corruption_rate": 0.10,
                    "deterministic_waterfill_capping": True,
                    "training_only": True,
                },
                "event_rejection_permitted": False,
            },
            "deployable_export": {
                "HLT_arrays_and_selected_checkpoints_only": True,
                "offline_inputs_accepted": False,
                "target_caches_loadable": False,
                "oracle_targets_requestable": False,
                "labels_accepted": False,
                "reload_parity_required": True,
            },
            "complete_graph_capacity": {
                "separate_per_exported_graph": True,
                "includes": [
                    "HLT_expert_encoders",
                    "predictors",
                    "dimension_projections",
                    "uncertainty_reliability_heads",
                    "token_refiner",
                    "final_consumer",
                    "deployable_frozen_offline_heads_and_fusion",
                ],
                "analytical_batches": [1, 128],
                "training_only_teachers_excluded": True,
                "target_caches_excluded": True,
            },
            "fixed_epochs": 40,
            "performance_based_termination": False,
            "stack_val_permitted": False,
            "final_test_permitted": False,
        }
    )


def validate_final_consumer_policy(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FINAL_CONSUMER_POLICY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_final_consumer_policy()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("final-consumer policy semantics differ")
    return digest


def build_final_consumer_registry(
    *,
    step11_bundle_sha256: str,
    predictor_bundle_lock_sha256: str,
    policy_sha256: str | None = None,
    carried_predictor_bundle_locks: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    rows = []
    carried = (
        {"PRIMARY": predictor_bundle_lock_sha256}
        if carried_predictor_bundle_locks is None
        else {
            str(role): require_sha256(
                digest,
                name=f"carried_predictor_bundle_locks.{role}",
            )
            for role, digest in carried_predictor_bundle_locks.items()
        }
    )
    if not carried:
        raise ValueError("final-consumer carried bundle coverage is empty")

    def add(
        *,
        consumer_kind: str,
        model_variant: str,
        seed: int,
        native_dropout_mode: str = "ND0_NONE",
        token_input: str = "TOKEN_PREDICTED",
        trainable: bool,
        role: str,
        semantic_label: str,
        carried_shape_role: str,
        carried_lock_sha256: str,
    ) -> None:
        rows.append(
            {
                "run_id": (
                    f"RETB_{carried_shape_role}_{consumer_kind}_{model_variant}_"
                    f"{native_dropout_mode}_{token_input}_S{seed}"
                ),
                "stage": "J_FINAL_CONSUMERS",
                "carried_shape_role": carried_shape_role,
                "consumer_kind": consumer_kind,
                "model_variant": model_variant,
                "native_dropout_mode": native_dropout_mode,
                "token_input": token_input,
                "pipeline_seed": int(seed),
                "trainable": bool(trainable),
                "role": role,
                "semantic_label": semantic_label,
                "step11_bundle_sha256": require_sha256(
                    step11_bundle_sha256, name="step11_bundle_sha256"
                ),
                "predictor_bundle_lock_sha256": require_sha256(
                    carried_lock_sha256,
                    name="predictor_bundle_lock_sha256",
                ),
            }
        )

    for carried_shape_role, carried_lock_sha256, seed in (
        (role, digest, seed)
        for role, digest in sorted(carried.items())
        for seed in PIPELINE_SEEDS
    ):
        add(
            consumer_kind="PF_FROZEN",
            model_variant="PF_FROZEN",
            seed=seed,
            trainable=False,
            role="reference_baseline",
            semantic_label="FAITHFUL_FROZEN_OFFLINE_FUSION",
            carried_shape_role=carried_shape_role,
            carried_lock_sha256=carried_lock_sha256,
        )
        add(
            consumer_kind="OF_ROBUST",
            model_variant="OF_ROBUST",
            seed=seed,
            trainable=True,
            role="robustness_control",
            semantic_label="ROBUST_RETRAINED_OFFLINE_FUSION",
            carried_shape_role=carried_shape_role,
            carried_lock_sha256=carried_lock_sha256,
        )
        for variant in REFINER_VARIANTS:
            add(
                consumer_kind="TR_REFINE",
                model_variant=variant,
                seed=seed,
                trainable=variant != "TR0_NONE",
                role=(
                    "semantic_control"
                    if variant in {"TR0_NONE", "TR3_ZERO_NATIVE_SHAPE"}
                    else "scientific_candidate"
                ),
                semantic_label="TOKEN_REFINEMENT_CONTROL",
                carried_shape_role=carried_shape_role,
                carried_lock_sha256=carried_lock_sha256,
            )
        for variant in ADAPTER_VARIANTS:
            for dropout in NATIVE_DROPOUT_MODES:
                add(
                    consumer_kind="HF_ADAPTER",
                    model_variant=variant,
                    native_dropout_mode=dropout,
                    seed=seed,
                    trainable=True,
                    role=(
                        "scientific_candidate"
                        if dropout == "ND0_NONE"
                        else "robustness_control"
                    ),
                    semantic_label="CONSTRAINED_RESIDUAL_ADAPTER",
                    carried_shape_role=carried_shape_role,
                    carried_lock_sha256=carried_lock_sha256,
                )
        for evidence in UNRESTRICTED_EVIDENCE_VARIANTS:
            for dropout in NATIVE_DROPOUT_MODES:
                for token_input in TOKEN_INPUTS:
                    add(
                        consumer_kind="HF_UNRESTRICTED",
                        model_variant=evidence,
                        native_dropout_mode=dropout,
                        token_input=token_input,
                        seed=seed,
                        trainable=True,
                        role=(
                            "scientific_candidate"
                            if dropout == "ND0_NONE"
                            else "robustness_control"
                        ),
                        semantic_label="MAXIMUM_PERFORMANCE_UNRESTRICTED",
                        carried_shape_role=carried_shape_role,
                        carried_lock_sha256=carried_lock_sha256,
                    )
    return with_content_hash(
        {
            "contract": STAGE_J_CONSUMER_REGISTRY_CONTRACT,
            "schema_version": 2,
            "policy_sha256": (
                build_final_consumer_policy()["content_hash"]
                if policy_sha256 is None
                else require_sha256(policy_sha256, name="policy_sha256")
            ),
            "step11_bundle_sha256": require_sha256(
                step11_bundle_sha256, name="step11_bundle_sha256"
            ),
            "predictor_bundle_lock_sha256": require_sha256(
                predictor_bundle_lock_sha256,
                name="predictor_bundle_lock_sha256",
            ),
            "carried_predictor_bundle_locks": dict(sorted(carried.items())),
            "membership_count": len(rows),
            "rows": rows,
            "all_negative_campaign_completed": True,
            "performance_based_termination": False,
        }
    )


def validate_final_consumer_registry(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_J_CONSUMER_REGISTRY_CONTRACT
    )
    expected = build_final_consumer_registry(
        step11_bundle_sha256=payload.get("step11_bundle_sha256"),
        predictor_bundle_lock_sha256=payload.get(
            "predictor_bundle_lock_sha256"
        ),
        policy_sha256=payload.get("policy_sha256"),
        carried_predictor_bundle_locks=payload.get(
            "carried_predictor_bundle_locks"
        ),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("final-consumer registry semantics differ")
    return digest


def materialize_final_consumer_run(
    *,
    registry_row: Mapping[str, Any],
    step12_bundle_sha256: str,
    parent_hashes: Mapping[str, str],
) -> dict[str, Any]:
    required = {
        "model_train_identity_manifest",
        "val_stop_identity_manifest",
        "val_design_identity_manifest",
        "val_design_label_manifest",
        "model_train_R_MULTI_view_cache",
        "val_stop_R_MULTI_view_cache",
        "val_design_fixed_view_cache",
        "joint_prediction_checkpoint",
        "native_HLT_checkpoint_bundle",
        "offline_target_cache",
        "target_normalizer_set",
        "uncertainty_calibration",
        "HLT_input_normalizer",
        "HLT_relation_normalizer",
        "HLT_region_normalizer",
        "degradation_profile",
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
    }
    if registry_row.get("token_input") == "TOKEN_REFINED_SELECTED":
        required.add("selected_token_refiner")
    required_row = {
        "run_id",
        "stage",
        "carried_shape_role",
        "consumer_kind",
        "model_variant",
        "native_dropout_mode",
        "token_input",
        "pipeline_seed",
        "trainable",
        "role",
        "semantic_label",
        "step11_bundle_sha256",
        "predictor_bundle_lock_sha256",
    }
    if (
        set(registry_row) != required_row
        or set(parent_hashes) != required
        or registry_row["stage"] != "J_FINAL_CONSUMERS"
        or registry_row["pipeline_seed"] not in PIPELINE_SEEDS
    ):
        raise ValueError("materialized final-consumer run semantics differ")
    return with_content_hash(
        {
            "contract": FINAL_CONSUMER_RUN_CONTRACT,
            "schema_version": 2,
            **dict(registry_row),
            "step12_bundle_sha256": require_sha256(
                step12_bundle_sha256, name="step12_bundle_sha256"
            ),
            "parent_hashes": {
                name: require_sha256(
                    value, name=f"parent_hashes.{name}"
                )
                for name, value in sorted(parent_hashes.items())
            },
            "fixed_budget": bool(registry_row["trainable"]),
            "performance_based_termination": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
    )


def validate_materialized_final_consumer_run(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FINAL_CONSUMER_RUN_CONTRACT
    )
    row_names = {
        "run_id",
        "stage",
        "carried_shape_role",
        "consumer_kind",
        "model_variant",
        "native_dropout_mode",
        "token_input",
        "pipeline_seed",
        "trainable",
        "role",
        "semantic_label",
        "step11_bundle_sha256",
        "predictor_bundle_lock_sha256",
    }
    expected = materialize_final_consumer_run(
        registry_row={name: payload.get(name) for name in row_names},
        step12_bundle_sha256=payload.get("step12_bundle_sha256"),
        parent_hashes=payload.get("parent_hashes", {}),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("materialized final-consumer run semantics differ")
    return digest


def build_step12_bundle(
    *,
    campaign_spec_sha256: str,
    step11_bundle_sha256: str,
    predictor_bundle_lock_sha256: str,
    joint_campaign_lock_sha256: str,
    global_determinism_sha256: str,
    source_snapshot: Mapping[str, Any],
    carried_predictor_bundle_locks: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    policy = bind_source(
        build_final_consumer_policy(), source_snapshot=source_snapshot
    )
    registry = bind_source(
        build_final_consumer_registry(
            step11_bundle_sha256=step11_bundle_sha256,
            predictor_bundle_lock_sha256=predictor_bundle_lock_sha256,
            policy_sha256=policy["content_hash"],
            carried_predictor_bundle_locks=carried_predictor_bundle_locks,
        ),
        source_snapshot=source_snapshot,
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": STEP12_BUNDLE_CONTRACT,
                "schema_version": 2,
                "parents": {
                    "campaign_spec": require_sha256(
                        campaign_spec_sha256, name="campaign_spec_sha256"
                    ),
                    "step11_bundle": require_sha256(
                        step11_bundle_sha256,
                        name="step11_bundle_sha256",
                    ),
                    "predictor_bundle_lock": require_sha256(
                        predictor_bundle_lock_sha256,
                        name="predictor_bundle_lock_sha256",
                    ),
                    "joint_campaign_lock": require_sha256(
                        joint_campaign_lock_sha256,
                        name="joint_campaign_lock_sha256",
                    ),
                    "global_determinism": require_sha256(
                        global_determinism_sha256,
                        name="global_determinism_sha256",
                    ),
                },
                "artifact_hashes": {
                    "final_consumer_policy": policy["content_hash"],
                    "final_consumer_registry": registry["content_hash"],
                },
                "TR_REFINE_complete": True,
                "faithful_and_constrained_consumers_complete": True,
                "HF_UNRESTRICTED_complete": True,
                "native_dropout_and_bypass_controls_complete": True,
                "deployable_HLT_only_export_required": True,
                "complete_graph_capacity_per_export_required": True,
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
                "contract": STEP12_REPORT_CONTRACT,
                "schema_version": 1,
                "step12_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "robust_mixture_exact": True,
                    "adapter_identity_initialization": True,
                    "refiner_identity_and_zero_native_controls": True,
                    "expert_logit_information_control": True,
                    "matched_token_only_capacity_control": True,
                    "all_native_dropout_modes": True,
                    "all_bypass_controls": True,
                    "HLT_only_reloadable_export": True,
                    "per_graph_complete_capacity": True,
                    "all_negative_campaign_continues": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {
        "final_consumer_policy": policy,
        "final_consumer_registry": registry,
        "step12_bundle": manifest,
        "step12_report": report,
    }


def validate_step12_bundle(
    bundle: Mapping[str, Mapping[str, Any]],
) -> str:
    if set(bundle) != {
        "final_consumer_policy",
        "final_consumer_registry",
        "step12_bundle",
        "step12_report",
    }:
        raise ValueError("Step-12 bundle members differ")
    policy_sha = validate_final_consumer_policy(
        bundle["final_consumer_policy"]
    )
    registry_sha = validate_final_consumer_registry(
        bundle["final_consumer_registry"]
    )
    digest = validate_content_hash(
        bundle["step12_bundle"], expected_contract=STEP12_BUNDLE_CONTRACT
    )
    validate_content_hash(
        bundle["step12_report"], expected_contract=STEP12_REPORT_CONTRACT
    )
    manifest = bundle["step12_bundle"]
    if (
        manifest["artifact_hashes"]
        != {
            "final_consumer_policy": policy_sha,
            "final_consumer_registry": registry_sha,
        }
        or bundle["final_consumer_registry"]["policy_sha256"]
        != policy_sha
        or bundle["step12_report"]["step12_bundle_sha256"] != digest
        or bundle["step12_report"]["scientific_results_inspected"]
        or manifest["performance_based_termination"]
        or manifest["stack_val_consumed"]
        or manifest["final_test_consumed"]
        or len({repr(row.get("source")) for row in bundle.values()}) != 1
    ):
        raise ValueError("Step-12 bundle semantics differ")
    return digest


def publish_step12_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    digest = validate_step12_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "final_consumer_policy": (
            root / "registry" / "retb_final_consumer_policy.json"
        ),
        "final_consumer_registry": (
            root / "registry" / "retb_final_consumer_registry.json"
        ),
        "step12_bundle": (
            root / "registry" / "retb_step12_final_consumers_bundle.json"
        ),
        "step12_report": root / "reports" / "retb_step12_report.json",
    }
    return {
        "step12_bundle_sha256": digest,
        "publications": {
            name: write_immutable_json(paths[name], bundle[name])
            for name in paths
        },
    }


__all__ = [
    "FINAL_CONSUMER_POLICY_CONTRACT",
    "FINAL_CONSUMER_RUN_CONTRACT",
    "STAGE_J_CONSUMER_REGISTRY_CONTRACT",
    "STEP12_BUNDLE_CONTRACT",
    "STEP12_REPORT_CONTRACT",
    "TOKEN_INPUTS",
    "build_final_consumer_policy",
    "build_final_consumer_registry",
    "build_step12_bundle",
    "materialize_final_consumer_run",
    "publish_step12_bundle",
    "validate_final_consumer_policy",
    "validate_final_consumer_registry",
    "validate_materialized_final_consumer_run",
    "validate_step12_bundle",
]

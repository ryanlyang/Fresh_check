"""Static, training-code-independent RETB registries and run IDs."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .contracts import canonical_sha256, validate_content_hash, with_content_hash
from .replicas import DOMAIN_SEEDS, RANDOM_MULTIPLIERS, REALIZATION_POLICIES


REGISTRY_CONTRACT = "retb_static_registry_v1"
EXPERT_ORDER = ("BASE4", "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION")
TOKEN_SHAPES = {
    "S1_128": {"K": 1, "D": 128},
    "S2_128": {"K": 2, "D": 128},
    "S4_128": {"K": 4, "D": 128},
    "S8_128": {"K": 8, "D": 128},
    "S16_128": {"K": 16, "D": 128},
    "S8_64": {"K": 8, "D": 64},
    "S16_64": {"K": 16, "D": 64},
}
PIPELINE_SEEDS = (101, 202, 303)
RUN_COMPONENTS = {
    "A": ("DATA_AUDIT", "HLT_V3_AUDIT", "O_BASE", "O_WIDE"),
    "B": ("OFFLINE_EXPERT", "TOKEN_SHAPE", "OPTIMIZATION_CONTROL"),
    "C": ("OFFLINE_FUSION", "COMPLEMENTARITY", "CAPACITY_CONTROL"),
    "D": ("HLT_EXPERT", "NATIVE_HLT_FUSION"),
    "E": ("BRIDGE_PILOT", "BRIDGE_TARGET", "TARGET_TUPLE"),
    "F": ("OFFLINE_TARGET_CACHE", "TOKEN_NORMALIZER"),
    "G": ("PREDICTOR", "UNCERTAINTY_CALIBRATOR"),
    "H": ("PREDICTOR_BUNDLE", "ORACLE_SUBSTITUTION"),
    "I": ("JOINT_PREDICTOR", "BRIDGE_FINETUNE"),
    "J": ("TOKEN_REFINER", "FINAL_ADAPTER", "UNRESTRICTED_FUSION"),
    "K": ("ROBUSTNESS", "SEMANTIC_CONTROL"),
    "L": ("CONFIRMATION_500K", "SCALE_SHORTLIST"),
    "M": ("SCALE_GRAPH_3M",),
    "N": (
        "STACK_VAL_INFERENCE",
        "SCALE_FINALIST_SELECTOR",
        "FINALIST_CONTROL",
        "FINAL_TEST",
    ),
}


def _registry(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": REGISTRY_CONTRACT,
            "schema_version": 1,
            "registry_name": name,
            **dict(payload),
        }
    )


def resolve_run_id(
    *,
    stage: str,
    component: str,
    seed: int,
    configuration: Mapping[str, Any],
) -> str:
    stage_key = str(stage).upper()
    component_key = str(component).upper()
    if re.fullmatch(r"[A-N]", stage_key) is None:
        raise ValueError("stage must be one of A through N")
    if re.fullmatch(r"[A-Z0-9][A-Z0-9_]{0,95}", component_key) is None:
        raise ValueError("component ID is unsafe")
    if component_key not in RUN_COMPONENTS[stage_key]:
        raise ValueError(
            f"component {component_key!r} is not registered for Stage {stage_key}"
        )
    if int(seed) < 0:
        raise ValueError("seed must be nonnegative")
    digest = canonical_sha256(
        {
            "run_id_contract": "retb_run_id_v1",
            "stage": stage_key,
            "component": component_key,
            "seed": int(seed),
            "configuration": dict(configuration),
        }
    )[:12]
    return f"retb_{stage_key.lower()}_{component_key.lower()}_s{int(seed)}_{digest}"


def build_registries() -> dict[str, dict[str, Any]]:
    experts = _registry(
        "expert_registry",
        {
            "canonical_order": list(EXPERT_ORDER),
            "experts": {
                "BASE4": {"pair_input": ["base4"], "all_particle_fields": True},
                "PT": {"pair_input": ["base4", "PT"], "all_particle_fields": True},
                "TRACK": {
                    "pair_input": ["base4", "TRACK"],
                    "all_particle_fields": True,
                },
                "PID": {"pair_input": ["base4", "PID"], "all_particle_fields": True},
                "CHARGE": {
                    "pair_input": ["base4", "CHARGE"],
                    "all_particle_fields": True,
                },
                "DENSITY": {
                    "pair_input": ["base4", "DENSITY"],
                    "all_particle_fields": True,
                },
                "REGION": {
                    "pair_input": ["base4", "REGION"],
                    "all_particle_fields": True,
                },
            },
            "primary_multi_relation_expert_allowed": False,
        },
    )
    expert_losses = _registry(
        "expert_loss_registry",
        {
            "temperature": 2.0,
            "candidates": {
                "ELOSS_CE": {"ce": 1.0, "kd": 0.0, "teacher": None},
                "ELOSS_BASE_LOW": {"ce": 1.0, "kd": 0.10, "teacher": "O_BASE"},
                "ELOSS_BASE": {"ce": 1.0, "kd": 0.50, "teacher": "O_BASE"},
                "ELOSS_FULLREL": {
                    "ce": 1.0,
                    "kd": 0.50,
                    "teacher": "O_FULLREL",
                },
                "ELOSS_ENSEMBLE": {
                    "ce": 1.0,
                    "kd": 0.50,
                    "teacher": "mean_probability_O_BASE_O_FULLREL",
                },
                "ELOSS_KD_DOMINANT": {
                    "ce": 0.25,
                    "kd": 1.0,
                    "teacher": "selected_strongest",
                },
            },
        },
    )
    token_registry = _registry(
        "token_registry",
        {
            "canonical": {
                "id": "TOK_CANONICAL",
                "blocks": 2,
                "query_identity": "learned_slot_index",
                "classification_bypass": False,
                "attention_dropout": 0.0,
                "residual_dropout": 0.1,
            },
            "controls": [
                "TOK_WEAVER_CLASS",
                "TOK_MASKED_MEAN",
                "TOK_ONE_QUERY_NO_SELF",
                "TOK_K_QUERY_NO_SELF",
                "TOK_MULTI_DEPTH",
            ],
        },
    )
    shape_registry = _registry(
        "token_shape_registry",
        {
            "shapes": TOKEN_SHAPES,
            "canonical_screen_order": list(TOKEN_SHAPES),
            "heterogeneous": {
                "HET_PHYSICS": {
                    "BASE4": 4,
                    "PT": 8,
                    "TRACK": 16,
                    "PID": 4,
                    "CHARGE": 4,
                    "DENSITY": 4,
                    "REGION": 16,
                },
                "HET_SELECTED": {"allowed_K": [1, 2, 4, 8, 16], "slot_cap": 56},
                "HET_BEAM": {
                    "allowed_K": [1, 2, 4, 8, 16],
                    "slot_cap": 56,
                    "beam_width": 32,
                },
            },
        },
    )
    targets = _registry(
        "target_mode_registry",
        {
            "modes": [
                "T0_PURE",
                "T1_ANCHORED_BRIDGE",
                "T1_TASK_BRIDGE",
                "T2_PROJECT",
                "T3_LOGIT",
            ],
            "primary": "T1_ANCHORED_BRIDGE",
            "pure_reference": "T0_PURE",
            "coordinate_tuple_beam_width": 16,
            "readout_seed": 41703,
        },
    )
    topology = _registry(
        "relation_bias_topology_registry",
        {
            "topologies": ["B_CONCAT", "B_DUAL_FIXED", "B_DUAL_GATED"],
            "dual_path_contract": "retb_layerwise_pair_bias_v1",
            "gate_range": [0.0, 2.0],
            "gate_initial_value": 1.0,
            "materialize_B_L_H_N_N": False,
        },
    )
    pilots = _registry(
        "bridge_pilot_registry",
        {
            "pilots": ["PILOT_T0"],
            "negative_count": 31,
            "temperature": 0.1,
            "training_negative_namespace": "retb_t1_negatives_v1",
            "certification_negative_namespace": "retb_t1_cert_negatives_v1",
        },
    )
    bridge_content = _registry(
        "bridge_content_contract",
        {
            "offline_noninferiority": {
                "mean_accuracy_deficit_max": 0.0020,
                "mean_cross_entropy_increase_max": 0.0050,
                "worst_per_class_efficiency_deficit_max": 0.0100,
            },
            "content_certification": {
                "class_agreement_min": 0.990,
                "median_variance_ratio_min": 0.50,
                "low_variance_channel_fraction_max": 0.05,
                "effective_rank_ratio_min": 0.80,
                "relative_covariance_error_max": 0.25,
                "retrieval_accuracy_min": 0.20,
            },
        },
    )
    fusions = _registry(
        "fusion_registry",
        {
            "offline": [
                "F_BEST_SINGLE",
                "F_UNIFORM_LOGIT_MEAN",
                "F_TRAINED_LOGIT_LINEAR",
                "F_POOLED_MLP",
                "F_TOKEN_TRANSFORMER",
                "F_TOKEN_TRANSFORMER_LIGHT_FINETUNE",
                "F_TOKEN_TRANSFORMER_FULL_FINETUNE",
            ],
            "final": [
                "PF_FROZEN",
                "OF_ROBUST",
                "HF_ADAPTER",
                "HF_UNRESTRICTED",
            ],
            "evidence": [
                "F_TOKEN_ONLY",
                "F_TOKEN_PLUS_EXPERT_LOGITS",
                "F_TOKEN_ONLY_MATCHED",
            ],
        },
    )
    predictors = _registry(
        "predictor_registry",
        {
            "architectures": [
                "A0_AFFINE",
                "A1_RESMLP",
                "A2_TOKEN_ENCODER",
                "A3_SLOT_DECODER_DIRECT",
                "A4_SLOT_DECODER_GATED",
            ],
            "contexts": ["C0_SELF", "C1_NATIVE", "C2_ALL", "C3_ALL_PARTICLE"],
            "joint": [
                "J0_INDEPENDENT",
                "J1_SHARED_CONTEXT",
                "J2_COUPLED_DECODER",
                "J3_INDEPENDENT_PLUS_ADAPTER",
                "J4_BRIDGE_FINETUNE",
                "J5_END_TO_END",
            ],
            "bundle_beam_width": 32,
            "gate_bias": -2.0,
            "log_variance_clip": [-8.0, 4.0],
        },
    )
    losses = _registry(
        "loss_registry",
        {
            "weights": {
                "W_TOKEN_ONLY": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "W_TOKEN_HEAVY": [1.0, 0.25, 0.10, 0.25, 0.25, 0.10],
                "W_CANONICAL": [1.0, 0.25, 0.10, 0.50, 0.50, 0.25],
                "W_TASK_HEAVY": [0.50, 0.10, 0.05, 1.0, 1.0, 0.50],
                "W_LOGIT_ONLY": [0.0, 0.0, 0.0, 1.0, 1.0, 0.50],
            },
            "column_order": [
                "token",
                "cosine",
                "relation",
                "expertKD",
                "swapKD",
                "CE",
            ],
            "gradnorm_control": "W_GRADNORM",
        },
    )
    uncertainty = _registry(
        "uncertainty_registry",
        {
            "heads": {
                "U_SLOT": "one_per_slot",
                "U_GROUP4": "four_contiguous_channel_groups_per_slot",
                "U_DIAGONAL": "one_per_channel_per_slot",
            },
            "normalizations": ["N_UNCLIPPED", "N_CLIP16", "N_CLIP8"],
            "calibration_population": "val_design_label_free",
        },
    )
    realization = _registry(
        "realization_policy_registry",
        {"policies": REALIZATION_POLICIES},
    )
    degradation_seeds = _registry(
        "degradation_domain_seed_registry",
        {
            "domain_seeds": DOMAIN_SEEDS,
            "random_multipliers": RANDOM_MULTIPLIERS,
            "event_seed_namespace": "retb_hlt_v3_rng_v1",
        },
    )
    normalizers = _registry(
        "normalizer_population_registry",
        {
            "populations": {
                "offline_500k": "model_train_offline_once_per_identity",
                "hlt_shared_500k": "model_train_x_R_MULTI_0_1_2_3_equal_weight",
                "offline_scale": "scale_train_offline_once_per_identity",
                "hlt_shared_scale": "scale_train_x_R_MULTI_0_1_2_3_equal_weight",
                "token_500k": "model_train_offline_targets",
                "token_scale": "scale_train_offline_targets",
            },
            "validation_or_test_fit_allowed": False,
        },
    )
    deployed = _registry(
        "deployed_graph_registry",
        {
            "adapter_inputs": [
                "R0_PREDICTED_ONLY",
                "R1_PREDICTED_PLUS_NATIVE_BASE",
                "R2_PREDICTED_PLUS_ALL_NATIVE_EXPERTS",
                "R3_NATIVE_ONLY_MATCHED_TO_R2",
            ],
            "native_dropout": ["ND0_NONE", "ND1_FIXED", "ND2_CONFIDENCE"],
            "token_refiners": [
                "TR0_NONE",
                "TR1_NATIVE_BASE",
                "TR2_ALL_NATIVE",
                "TR3_ZERO_NATIVE_SHAPE",
            ],
            "capacity_controls": [
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
                "H_BASE",
                "H_WIDE",
                "H_BASE_LONG",
            ],
            "performance_failure_blocks_future_runs": False,
        },
    )
    stages = _registry(
        "campaign_stage_registry",
        {
            "stage_order": list("ABCDEFGHIJKLMN"),
            "stage_roles": {
                "A": "data_degradation_exact_baselines",
                "B": "offline_expert_shape_screen",
                "C": "offline_fusion_complementarity",
                "D": "native_hlt_evidence",
                "E": "bridge_target_coordinates",
                "F": "offline_target_cache",
                "G": "predictor_screen",
                "H": "predictor_bundle",
                "I": "joint_bridge",
                "J": "final_consumers",
                "K": "robustness_semantic_controls",
                "L": "500k_confirmation_scale_shortlist",
                "M": "3m_scale_shortlist_training",
                "N": "stack_selection_locks_final_test",
            },
            "dataset_access": {
                "training": ["model_train", "val_stop"],
                "scale_training": ["scale_train", "val_stop"],
                "design": ["val_design"],
                "stage_n_selection_inference": ["stack_val_features_no_labels"],
                "stage_n_selector": ["stack_val_label_manifest_no_features"],
                "final": ["final_test_after_execution_lock"],
            },
            "run_id_contract": {
                "id": "retb_run_id_v1",
                "format": "retb_<stage>_<component>_s<seed>_<config_sha256_12>",
                "resolver": "registry.resolve_run_id",
            },
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "screen_seed": 101,
            "beam_readout_seeds": [41701, 41702, 41703],
        },
    )
    run_ids = _registry(
        "run_id_registry",
        {
            "contract_id": "retb_run_id_v1",
            "format": "retb_<stage>_<component>_s<seed>_<config_sha256_12>",
            "configuration_hash": (
                "sha256_canonical_json_of_stage_component_seed_configuration"
            ),
            "stage_component_namespaces": {
                stage: list(components)
                for stage, components in RUN_COMPONENTS.items()
            },
            "configuration_axes_are_registry_ids_only": True,
            "scientific_result_may_change_run_id": False,
            "resolver": "registry.resolve_run_id",
        },
    )
    return {
        "expert_registry": experts,
        "expert_loss_registry": expert_losses,
        "token_registry": token_registry,
        "token_shape_registry": shape_registry,
        "target_mode_registry": targets,
        "relation_bias_topology_registry": topology,
        "bridge_pilot_registry": pilots,
        "bridge_content_contract": bridge_content,
        "fusion_registry": fusions,
        "predictor_registry": predictors,
        "loss_registry": losses,
        "uncertainty_registry": uncertainty,
        "realization_policy_registry": realization,
        "degradation_domain_seed_registry": degradation_seeds,
        "normalizer_population_registry": normalizers,
        "deployed_graph_registry": deployed,
        "campaign_stage_registry": stages,
        "run_id_registry": run_ids,
    }


def validate_registry(payload: Mapping[str, Any], *, name: str) -> str:
    digest = validate_content_hash(payload, expected_contract=REGISTRY_CONTRACT)
    if payload.get("registry_name") != name:
        raise ValueError(f"registry name mismatch: expected {name!r}")
    expected = build_registries()[name]
    actual = dict(payload)
    actual.pop("source", None)
    actual.pop("content_hash", None)
    expected.pop("content_hash", None)
    if actual != expected:
        raise ValueError(f"registry semantics differ for {name}")
    return digest


__all__ = [
    "EXPERT_ORDER",
    "PIPELINE_SEEDS",
    "REGISTRY_CONTRACT",
    "RUN_COMPONENTS",
    "TOKEN_SHAPES",
    "build_registries",
    "resolve_run_id",
    "validate_registry",
]

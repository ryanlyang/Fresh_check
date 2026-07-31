"""Step-14 policy and source-bound bundle for RETB Stages M and N."""

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
from .predictor_bundle import PIPELINE_SEEDS


STAGE_MN_POLICY_CONTRACT = "retb_stage_mn_scale_final_seal_policy_v1"
STEP14_BUNDLE_CONTRACT = "retb_step14_scale_final_seal_bundle_v2"
STEP14_PREFLIGHT_CONTRACT = "retb_step14_preflight_report_v2"


def build_stage_mn_policy() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": STAGE_MN_POLICY_CONTRACT,
            "schema_version": 1,
            "stage_M": {
                "training_population": "scale_train",
                "epoch_selection_population": "val_stop",
                "pipeline_seeds": list(PIPELINE_SEEDS),
                "maximum_epochs": 40,
                "train_every_and_only_locked_shortlist_graph": True,
                "architecture_reselection_allowed": False,
                "component_reselection_allowed": False,
                "stack_val_allowed": False,
                "final_test_allowed": False,
                "required_refits": [
                    "offline_relation_and_REGION",
                    "shared_HLT_relation_and_REGION_R_MULTI_0_1_2_3",
                    "all_locked_scale_input_standardizers",
                    "scale_teacher_target_token_normalizers",
                    "label_free_uncertainty_calibrator_on_val_design",
                ],
                "five_hundred_k_normalizer_substitution_allowed": False,
            },
            "stage_N_pre_finalist_lock": {
                "only_model_output": (
                    "label_free_deployable_stack_val_logits_probabilities"
                ),
                "selection_prediction_fields": [
                    "canonical_identity",
                    "graph_id",
                    "pipeline_seed",
                    "float32_logits",
                    "float32_probabilities",
                ],
                "forbidden_prediction_fields": [
                    "label",
                    "offline_target",
                    "oracle_token",
                    "oracle_logit",
                ],
                "selector_alone_joins_label_manifest": True,
                "prediction_shards_remain_label_free": True,
                "final_test_model_output_allowed": False,
            },
            "dual_finalist_selection": {
                "accuracy_primary": "three_seed_mean_balanced_accuracy",
                "accuracy_window": 0.0001,
                "accuracy_ties": [
                    "lower_mean_cross_entropy",
                    "lower_analytical_batch1_FLOPs",
                    "lower_parameter_count",
                    "lexicographic_graph_id",
                ],
                "rejection_primary": (
                    "three_seed_mean_of_18_log_Jeffreys_rejections"
                ),
                "rejection_window": 0.005,
                "rejection_ties": [
                    "higher_mean_accuracy",
                    "lower_mean_cross_entropy",
                    "lower_analytical_batch1_FLOPs",
                    "lower_parameter_count",
                    "lexicographic_graph_id",
                ],
                "same_graph_may_win_both": True,
                "scientific_underperformance_blocks_lock": False,
            },
            "post_finalist_lock": {
                "stack_val_oracle_diagnostics_selection_eligible": False,
                "final_test_targets_require_scale_teachers": True,
                "five_hundred_k_teacher_or_target_allowed": False,
                "finalist_controls_resolved_after_lock": True,
                "controls_may_replace_finalist": False,
            },
            "final_test": {
                "requires_locked_scale_finalists": True,
                "requires_final_test_execution_lock": True,
                "only_registered_rows_allowed": True,
                "checkpoint_substitution_allowed": False,
                "test_result_may_select_replacement": False,
                "one_immutable_evaluation_per_registered_row": True,
            },
            "performance_based_termination": False,
        }
    )


def validate_stage_mn_policy(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_MN_POLICY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_stage_mn_policy()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-M/N policy semantics differ")
    return digest


def build_step14_bundle(
    *,
    campaign_spec_sha256: str,
    step13_bundle_sha256: str,
    locked_scale_shortlist_sha256: str,
    shortlisted_500k_controls_sha256: str,
    global_determinism_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    policy = bind_source(
        build_stage_mn_policy(), source_snapshot=source_snapshot
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": STEP14_BUNDLE_CONTRACT,
                "schema_version": 2,
                "parents": {
                    "campaign_spec": require_sha256(
                        campaign_spec_sha256,
                        name="campaign_spec_sha256",
                    ),
                    "step13_bundle": require_sha256(
                        step13_bundle_sha256,
                        name="step13_bundle_sha256",
                    ),
                    "locked_scale_shortlist": require_sha256(
                        locked_scale_shortlist_sha256,
                        name="locked_scale_shortlist_sha256",
                    ),
                    "shortlisted_500k_controls": require_sha256(
                        shortlisted_500k_controls_sha256,
                        name="shortlisted_500k_controls_sha256",
                    ),
                    "global_determinism": require_sha256(
                        global_determinism_sha256,
                        name="global_determinism_sha256",
                    ),
                },
                "artifact_hashes": {
                    "stage_mn_policy": policy["content_hash"]
                },
                "scale_completion_required_before_stack_val": True,
                "dual_finalist_lock_required": True,
                "postlock_targets_and_controls_required": True,
                "execution_lock_required_before_final_inference": True,
                "negative_campaign_continues": True,
            }
        ),
        source_snapshot=source_snapshot,
    )
    preflight = bind_source(
        with_content_hash(
            {
                "contract": STEP14_PREFLIGHT_CONTRACT,
                "schema_version": 2,
                "step14_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "shortlist_already_immutable": True,
                    "real_shortlisted_500k_controls_complete": True,
                    "every_and_only_scale_graph_rule_frozen": True,
                    "stack_val_label_feature_separation_frozen": True,
                    "accuracy_and_rejection_selectors_frozen": True,
                    "postlock_oracle_timing_frozen": True,
                    "two_lock_final_test_sequence_frozen": True,
                    "performance_based_termination_disabled": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {
        "stage_mn_policy": policy,
        "step14_bundle": manifest,
        "step14_preflight_report": preflight,
    }


def validate_step14_bundle(
    bundle: Mapping[str, Mapping[str, Any]],
) -> str:
    if set(bundle) != {
        "stage_mn_policy",
        "step14_bundle",
        "step14_preflight_report",
    }:
        raise ValueError("Step-14 bundle members differ")
    policy_sha = validate_stage_mn_policy(bundle["stage_mn_policy"])
    digest = validate_content_hash(
        bundle["step14_bundle"],
        expected_contract=STEP14_BUNDLE_CONTRACT,
    )
    validate_content_hash(
        bundle["step14_preflight_report"],
        expected_contract=STEP14_PREFLIGHT_CONTRACT,
    )
    if (
        bundle["step14_bundle"]["artifact_hashes"]
        != {"stage_mn_policy": policy_sha}
        or bundle["step14_preflight_report"]["step14_bundle_sha256"]
        != digest
        or bundle["step14_preflight_report"][
            "scientific_results_inspected"
        ]
        or len({repr(row.get("source")) for row in bundle.values()}) != 1
    ):
        raise ValueError("Step-14 bundle semantics differ")
    return digest


def publish_step14_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    digest = validate_step14_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "stage_mn_policy": root / "registry" / "retb_stage_mn_policy.json",
        "step14_bundle": (
            root / "registry" / "retb_step14_scale_final_seal_bundle.json"
        ),
        "step14_preflight_report": (
            root / "reports" / "retb_step14_preflight_report.json"
        ),
    }
    return {
        "step14_bundle_sha256": digest,
        "publications": {
            name: write_immutable_json(paths[name], bundle[name])
            for name in paths
        },
    }


__all__ = [
    "STAGE_MN_POLICY_CONTRACT",
    "STEP14_BUNDLE_CONTRACT",
    "STEP14_PREFLIGHT_CONTRACT",
    "build_stage_mn_policy",
    "build_step14_bundle",
    "publish_step14_bundle",
    "validate_stage_mn_policy",
    "validate_step14_bundle",
]

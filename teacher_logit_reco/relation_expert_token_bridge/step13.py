"""Step-13 contracts for 500k confirmation and scale shortlisting."""

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


STAGE_L_POLICY_CONTRACT = "retb_stage_l_confirmation_shortlist_policy_v1"
STEP13_BUNDLE_CONTRACT = "retb_step13_confirmation_shortlist_bundle_v1"
STEP13_PREFLIGHT_REPORT_CONTRACT = "retb_step13_preflight_report_v1"

FAILURE_INTERPRETATIONS = (
    {
        "interpretation_id": "RELATION_EXPERTS_NOT_ABOVE_O_BASE",
        "meaning": (
            "Relation bias did not improve a complete full-field expert; "
            "token reconstruction may still be studied without claiming "
            "improved offline reasoning."
        ),
    },
    {
        "interpretation_id": "CAPACITY_CONTROLS_MATCH_ORACLE_FUSION",
        "meaning": (
            "The gain is attributable to capacity or ensembling rather than "
            "demonstrated relation specialization."
        ),
    },
    {
        "interpretation_id": "EXPERTS_STRONG_BUT_REDUNDANT",
        "meaning": (
            "The biases produce similar decisions; report negative "
            "complementarity without retroactively forcing diversity."
        ),
    },
    {
        "interpretation_id": "SMALL_K_PREDICTS_POORLY",
        "meaning": (
            "Compact slots may be overloaded or unstable; prefer the locked "
            "higher-shape candidate only if bridge-aware ordering selects it."
        ),
    },
    {
        "interpretation_id": "LARGE_K_PREDICTS_BETTER",
        "meaning": (
            "Additional slots may factorize evidence into easier targets; "
            "output dimension alone is not reconstruction difficulty."
        ),
    },
    {
        "interpretation_id": "LOW_DIMENSIONAL_OR_HETEROGENEOUS_WINS",
        "meaning": (
            "Slot count is not the sole bottleneck; report scalar width and "
            "per-expert allocation rather than making a K-only claim."
        ),
    },
    {
        "interpretation_id": "BRIDGE_AWARE_DEPLOYS_BETTER",
        "meaning": (
            "Pure-offline coordinates were class-useful but unnecessarily "
            "hard to infer; this is not proof of coordinate reconstruction."
        ),
    },
    {
        "interpretation_id": "BRIDGE_NONINFERIORITY_FAIL",
        "meaning": (
            "The bridge-aware target is ineligible; retain T0_PURE and do not "
            "relax the frozen noninferiority threshold."
        ),
    },
    {
        "interpretation_id": "TASK_BRIDGE_CONTENT_FAIL",
        "meaning": (
            "The target is task distillation, not faithful instance-level "
            "representation recovery."
        ),
    },
    {
        "interpretation_id": "JOINT_BUNDLE_INTERACTION",
        "meaning": (
            "Prediction errors interact across banks; report the independent "
            "hybrid and joint-beam results without assigning individual gain."
        ),
    },
    {
        "interpretation_id": "TOKEN_ERROR_NO_CONSUMER_GAIN",
        "meaning": (
            "Raw coordinate error overweights irrelevant coordinates; frozen "
            "consumer utility remains the predictor endpoint."
        ),
    },
    {
        "interpretation_id": "LOGIT_ONLY_NOT_FAITHFUL",
        "meaning": (
            "A logit-only win is successful task distillation, not faithful "
            "token reconstruction."
        ),
    },
    {
        "interpretation_id": "NATIVE_MATCHES_RECONSTRUCTION",
        "meaning": (
            "Relation specialization helps HLT but offline-token "
            "reconstruction adds no deployable value."
        ),
    },
    {
        "interpretation_id": "ORACLE_GAP",
        "meaning": (
            "Useful oracle information is unavailable or ambiguous in HLT; "
            "oracle substitution identifies the dominant expert-bank gap."
        ),
    },
    {
        "interpretation_id": "ADAPTER_GAIN_WITHOUT_FROZEN_GAIN",
        "meaning": (
            "HLT correction is useful but evidence for offline "
            "representation recovery is weak."
        ),
    },
    {
        "interpretation_id": "UNRESTRICTED_WITHOUT_FAITHFUL_GAIN",
        "meaning": (
            "The system is useful, but its gain cannot be attributed to "
            "recovering offline token reasoning."
        ),
    },
    {
        "interpretation_id": "EXPERT_LOGIT_GAIN_NEEDS_MATCHED_CONTROL",
        "meaning": (
            "Compare with F_TOKEN_ONLY_MATCHED before attributing the gain to "
            "information rather than capacity."
        ),
    },
    {
        "interpretation_id": "MULTI_REALIZATION_ROBUSTNESS_ONLY",
        "meaning": (
            "The benefit is domain randomization rather than improved "
            "fixed-proxy reconstruction."
        ),
    },
    {
        "interpretation_id": "OFFLINE_INIT_DEPENDENCE",
        "meaning": (
            "The bridge depends on privileged alignment or initialization "
            "and must not be presented as HLT-label-only learning."
        ),
    },
    {
        "interpretation_id": "TRACK_DEGRADATION_GAP",
        "meaning": (
            "A severe track-dominant gap is a scientific result, not a job "
            "failure; do not retune the nominal profile after inspection."
        ),
    },
    {
        "interpretation_id": "SCALE_RANKING_CHANGE",
        "meaning": (
            "A 500k-to-3M ranking change is an expected scale effect and may "
            "be resolved only by the locked Stage-N selectors."
        ),
    },
    {
        "interpretation_id": "NO_MODEL_IMPROVES",
        "meaning": (
            "A provenance-complete negative result is a successful campaign "
            "outcome and does not block shortlist emission or scale-up."
        ),
    },
)


def build_stage_l_policy() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": STAGE_L_POLICY_CONTRACT,
            "schema_version": 1,
            "stage": "L",
            "population": "500k_model_train_with_val_design_selection",
            "pipeline_seeds": list(PIPELINE_SEEDS),
            "pipeline_seed_lineage": {
                "kind": "PRIMARY_MATCHED_SEED",
                "same_seed_required": [
                    "offline_experts",
                    "offline_fusion",
                    "offline_target_cache",
                    "native_HLT_experts",
                    "predictor_bundle",
                    "refiner_or_adapter",
                    "final_consumer",
                ],
                "fixed_teacher_controls_selection_eligible": False,
            },
            "confirmed_categories": [
                "PRIMARY_BASELINE",
                "UNIFORM_FINALIST",
                "HETEROGENEOUS_FINALIST",
                "NATIVE_HLT_FUSION",
                "FROZEN_RECONSTRUCTION",
                "TOKEN_REFINER",
                "CONSTRAINED_ADAPTER",
                "UNRESTRICTED_FUSION",
            ],
            "bridge_shape_selection": {
                "candidates": ["SHAPE_COMPACT", "SHAPE_HIGH"],
                "ordering": [
                    "higher_mean_paired_gain_over_shape_matched_HF_NATIVE",
                    "lower_mean_frozen_fusion_cross_entropy",
                    "lower_mean_normalized_token_error",
                    "fewer_total_scalars",
                    "smaller_K",
                    "smaller_D",
                ],
                "split": "val_design",
            },
            "shortlist": {
                "accuracy_top_k": 3,
                "accuracy_window": 0.0001,
                "rejection_top_k": 3,
                "rejection_score": (
                    "mean_of_18_log_Jeffreys_smoothed_rejections"
                ),
                "rejection_window": 0.005,
                "union": "canonical_duplicate_free_graph_ID_union",
                "minimum_size": 1,
                "maximum_size": 6,
                "ineligible_only_when": [
                    "incomplete_three_seed_coverage",
                    "nonfinite_required_metric",
                    "not_deployable",
                    "lineage_invalid",
                ],
                "underperformance_removes_graph": False,
            },
            "statistics": {
                "paired_unit": "canonical_jet_identity",
                "bootstrap_seed": 917301,
                "bootstrap_replicates": 10000,
                "class_stratified_balanced_counts": True,
                "quantiles": [0.025, 0.975],
                "quantile_method": "linear",
                "rejection_threshold_recomputed_per_resample": True,
                "all_18_rejection_terms_recomputed_per_resample": True,
            },
            "locked_shortlist_contains": [
                "complete_500k_graph_definitions",
                "accuracy_and_rejection_ranking_traces",
                "duplicate_removal",
                "SHAPE_BRIDGE",
            ],
            "locked_shortlist_forbids": [
                "3M_checkpoint",
                "final_finalist_identity",
                "stack_val_metric",
                "final_test_output",
            ],
            "post_shortlist_controls": {
                "required_for_every_shortlisted_graph": [
                    "complete_graph_capacity",
                    "monolithic_parameter_match",
                    "monolithic_FLOP_match",
                    "H_BASE_LONG_label_exposure_match",
                ],
                "resolved_only_after_shortlist_lock": True,
                "may_change_shortlist_membership": False,
            },
            "failure_interpretations": [
                dict(row) for row in FAILURE_INTERPRETATIONS
            ],
            "stage_M_must_train_every_and_only_shortlisted_graph": True,
            "fixed_epochs": 40,
            "performance_based_termination": False,
            "stack_val_permitted": False,
            "final_test_permitted": False,
        }
    )


def validate_stage_l_policy(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_L_POLICY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_stage_l_policy()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("Stage-L policy semantics differ")
    return digest


def build_step13_bundle(
    *,
    campaign_spec_sha256: str,
    step12_bundle_sha256: str,
    global_determinism_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    policy = bind_source(
        build_stage_l_policy(), source_snapshot=source_snapshot
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": STEP13_BUNDLE_CONTRACT,
                "schema_version": 1,
                "parents": {
                    "campaign_spec": require_sha256(
                        campaign_spec_sha256,
                        name="campaign_spec_sha256",
                    ),
                    "step12_bundle": require_sha256(
                        step12_bundle_sha256,
                        name="step12_bundle_sha256",
                    ),
                    "global_determinism": require_sha256(
                        global_determinism_sha256,
                        name="global_determinism_sha256",
                    ),
                },
                "artifact_hashes": {
                    "stage_l_policy": policy["content_hash"],
                },
                "matched_seed_confirmation_required": True,
                "bridge_shape_selection_required": True,
                "bounded_dual_metric_shortlist_required": True,
                "complete_reports_required": True,
                "all_negative_campaign_continues": True,
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
                "contract": STEP13_PREFLIGHT_REPORT_CONTRACT,
                "schema_version": 1,
                "step13_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "three_matched_pipeline_seeds": True,
                    "SHAPE_BRIDGE_order_frozen": True,
                    "accuracy_top3_union_rejection_top3": True,
                    "all_negative_campaign_continues": True,
                    "val_design_only": True,
                    "stack_val_forbidden": True,
                    "final_test_forbidden": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {
        "stage_l_policy": policy,
        "step13_bundle": manifest,
        "step13_preflight_report": report,
    }


def validate_step13_bundle(
    bundle: Mapping[str, Mapping[str, Any]],
) -> str:
    if set(bundle) != {
        "stage_l_policy",
        "step13_bundle",
        "step13_preflight_report",
    }:
        raise ValueError("Step-13 bundle members differ")
    policy_sha = validate_stage_l_policy(bundle["stage_l_policy"])
    digest = validate_content_hash(
        bundle["step13_bundle"],
        expected_contract=STEP13_BUNDLE_CONTRACT,
    )
    validate_content_hash(
        bundle["step13_preflight_report"],
        expected_contract=STEP13_PREFLIGHT_REPORT_CONTRACT,
    )
    manifest = bundle["step13_bundle"]
    if (
        manifest["artifact_hashes"] != {"stage_l_policy": policy_sha}
        or bundle["step13_preflight_report"]["step13_bundle_sha256"]
        != digest
        or bundle["step13_preflight_report"][
            "scientific_results_inspected"
        ]
        or manifest["performance_based_termination"]
        or manifest["stack_val_consumed"]
        or manifest["final_test_consumed"]
        or len({repr(row.get("source")) for row in bundle.values()}) != 1
    ):
        raise ValueError("Step-13 bundle semantics differ")
    return digest


def publish_step13_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    digest = validate_step13_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "stage_l_policy": root / "registry" / "retb_stage_l_policy.json",
        "step13_bundle": (
            root / "registry" / "retb_step13_confirmation_shortlist_bundle.json"
        ),
        "step13_preflight_report": (
            root / "reports" / "retb_step13_preflight_report.json"
        ),
    }
    return {
        "step13_bundle_sha256": digest,
        "publications": {
            name: write_immutable_json(paths[name], bundle[name])
            for name in paths
        },
    }


__all__ = [
    "STAGE_L_POLICY_CONTRACT",
    "STEP13_BUNDLE_CONTRACT",
    "STEP13_PREFLIGHT_REPORT_CONTRACT",
    "FAILURE_INTERPRETATIONS",
    "build_stage_l_policy",
    "build_step13_bundle",
    "publish_step13_bundle",
    "validate_stage_l_policy",
    "validate_step13_bundle",
]

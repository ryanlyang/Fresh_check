"""Immutable, source-bound contracts for HOSD.

Step 1 deliberately reuses RETB's canonical JSON and source-snapshot
semantics.  HOSD artifacts have independent contract IDs and scientific
meaning, but byte serialization and source equality do not fork.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    SOURCE_STATUS_HASH_POLICY,
    bind_source,
    canonical_json_bytes,
    canonical_sha256,
    load_hashed_json,
    require_git_object_id,
    require_sha256,
    source_record,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)


CANONICAL_JSON_CONTRACT = "hosd_canonical_json_v1"
CAMPAIGN_SPEC_CONTRACT = "hosd_campaign_spec_v4"
ARTIFACT_LAYOUT_CONTRACT = "hosd_artifact_layout_v1"
DESIGN_PARTITION_CONTRACT = "hosd_design_partition_manifest_v1"
DRY_RUN_PLAN_CONTRACT = "hosd_stage_a_to_k_dry_run_plan_v1"
STEP1_REPORT_CONTRACT = "hosd_step1_report_v1"
REGISTRY_CONTRACT = "hosd_registry_v1"
PARENT_STATUS_CONTRACT = "hosd_inherited_parent_status_v2"
PARENT_REBUILD_PLAN_CONTRACT = "hosd_parent_rebuild_plan_v2"
TARGET_CAPABILITY_AUDIT_CONTRACT = "hosd_target_capability_audit_v1"
STRUCTURE_TARGET_REGISTRY_CONTRACT = "hosd_structure_target_registry_v1"
TARGET_CACHE_SPEC_CONTRACT = "hosd_target_cache_spec_v1"
TARGET_SHARD_CONTRACT = "hosd_target_shard_v1"
TARGET_CACHE_MANIFEST_CONTRACT = "hosd_target_cache_manifest_v1"
TARGET_NORMALIZER_CONTRACT = "hosd_target_normalizer_v1"
STREAMED_TARGET_NORMALIZER_CONTRACT = "hosd_streamed_target_normalizer_v1"
RESIDUAL_CACHE_SPEC_CONTRACT = "hosd_residual_cache_spec_v1"
CONDITIONAL_RESIDUAL_CONTRACT = "hosd_conditional_residual_v1"
HETEROSCEDASTIC_METADATA_CONTRACT = "hosd_heteroscedastic_metadata_v1"
TARGET_SHUFFLE_PLAN_CONTRACT = "hosd_target_shuffle_plan_v1"
TARGET_CONTROL_MANIFEST_CONTRACT = "hosd_target_control_manifest_v1"
STORAGE_MEASUREMENT_CONTRACT = "hosd_storage_measurements_v2"
STORAGE_PROBE_EVIDENCE_CONTRACT = "hosd_storage_probe_evidence_v1"
TEACHER_TRAINING_MANIFEST_CONTRACT = "hosd_teacher_training_manifest_v1"
TEACHER_LOCK_CONTRACT = "hosd_teacher_lock_v1"
TEACHER_OUTPUT_MANIFEST_CONTRACT = "hosd_teacher_output_manifest_v2"
LATENT_WHITENING_CONTRACT = "hosd_latent_whitening_v2"
RIDGE_ADAPTER_CONTRACT = "hosd_latent_ridge_adapter_v1"
TARGET_AUDIT_CONTRACT = "hosd_target_audit_v1"
SPLIT_FORWARD_CONTRACT = "hosd_weaver_split_forward_v1"
TARGET_HEAD_CONTRACT = "hosd_target_head_registry_v2"
BASELINE_REGISTRY_CONTRACT = "hosd_stage_c_baseline_registry_v1"
STAGE_C_PLAN_CONTRACT = "hosd_stage_c_execution_plan_v1"
BASELINE_CHECKPOINT_CONTRACT = "hosd_baseline_checkpoint_v1"
BASELINE_COMPLETION_CONTRACT = "hosd_baseline_completion_v1"
PROBE_ENCODER_LOCK_CONTRACT = "hosd_probe_encoder_lock_v1"
PROBE_CHECKPOINT_CONTRACT = "hosd_probe_checkpoint_v1"
PROBE_RESULT_CONTRACT = "hosd_probe_result_v1"
PROBE_COMPLETION_CONTRACT = "hosd_probe_completion_v1"
PREDICTABILITY_MATRIX_CONTRACT = "hosd_predictability_matrix_v1"
STAGE_D_PLAN_CONTRACT = "hosd_stage_d_execution_plan_v1"
AUXILIARY_OBJECTIVE_CONTRACT = "hosd_auxiliary_objective_v1"
AUXILIARY_CHECKPOINT_CONTRACT = "hosd_auxiliary_checkpoint_v2"
AUXILIARY_COMPLETION_CONTRACT = "hosd_auxiliary_completion_v2"
AUXILIARY_PREDICTION_CONTRACT = "hosd_auxiliary_prediction_v2"
SINGLE_FAMILY_PHASE_LOCK_CONTRACT = "hosd_single_family_phase_lock_v1"
SINGLE_FAMILY_SELECTION_CONTRACT = "hosd_single_family_selection_v2"
FEEDBACK_INTERFACE_CONTRACT = "hosd_feedback_interface_v2"
STAGE_E_PLAN_CONTRACT = "hosd_stage_e_execution_plan_v3"
FEEDBACK_CHECKPOINT_CONTRACT = "hosd_feedback_checkpoint_v3"
FEEDBACK_RESULT_CONTRACT = "hosd_feedback_result_v4"
FEEDBACK_COMPLETION_CONTRACT = "hosd_feedback_completion_v4"
FEEDBACK_SELECTION_CONTRACT = "hosd_feedback_selection_v3"
STAGE_F_PLAN_CONTRACT = "hosd_stage_f_combination_plan_v2"
COMBINATION_RESULT_CONTRACT = "hosd_combination_result_v1"
COMBINATION_SELECTION_CONTRACT = "hosd_combination_selection_v1"
COMBINATION_BEAM_COMPLETION_CONTRACT = "hosd_combination_beam_completion_v1"
COMBINATION_BEAM_PROMOTION_CONTRACT = "hosd_combination_beam_promotion_v1"
COMBINATION_WAVE_COMPLETION_CONTRACT = "hosd_combination_wave_completion_v1"
GRADIENT_CONFLICT_CONTRACT = "hosd_gradient_conflict_v1"
MECHANISM_CONTROL_PLAN_CONTRACT = "hosd_mechanism_control_plan_v1"
MECHANISM_RESULT_CONTRACT = "hosd_mechanism_result_v1"
MECHANISM_SUMMARY_CONTRACT = "hosd_mechanism_summary_v1"
HOSD_METRICS_CONTRACT = "hosd_metrics_v1"
ROBUSTNESS_PLAN_CONTRACT = "hosd_robustness_plan_v1"
ROBUSTNESS_RESULT_CONTRACT = "hosd_robustness_result_v1"
ROBUSTNESS_SUMMARY_CONTRACT = "hosd_robustness_summary_v1"
EFFICIENCY_PROFILE_CONTRACT = "hosd_efficiency_profile_v3"
PLOT_BUNDLE_CONTRACT = "hosd_plot_bundle_v1"
HOSD_REPORT_CONTRACT = "hosd_report_v1"
HOSD_PAIRED_STATISTICS_CONTRACT = "hosd_paired_statistics_v1"
CONFIRMATION_PLAN_CONTRACT = "hosd_confirmation_plan_v3"
GRAPH_REGISTRY_CONTRACT = "hosd_locked_graph_registry_v2"
CONFIRMATION_RESULT_CONTRACT = "hosd_confirmation_result_v2"
CONFIRMATION_TRAINING_CHECKPOINT_CONTRACT = (
    "hosd_confirmation_training_checkpoint_v1"
)
CONFIRMATION_TRAINING_COMPLETION_CONTRACT = (
    "hosd_confirmation_training_completion_v1"
)
CONFIRMATION_TRAINING_PREDICTION_CONTRACT = (
    "hosd_confirmation_training_prediction_v1"
)
CONFIRMATION_WAVE_COMPLETION_CONTRACT = (
    "hosd_confirmation_wave_completion_v1"
)
CONFIRMATION_SUMMARY_CONTRACT = "hosd_confirmation_summary_v1"
SCALE_SHORTLIST_CONTRACT = "hosd_scale_shortlist_v2"
SCALE_EXECUTION_PLAN_CONTRACT = "hosd_scale_execution_plan_v4"
SCALE_INPUT_COMPLETION_CONTRACT = "hosd_scale_input_completion_v4"
SCALE_TREE_WAVE_COMPLETION_CONTRACT = "hosd_scale_tree_wave_completion_v1"
SCALE_NORMALIZER_COMPLETION_CONTRACT = (
    "hosd_scale_normalizer_completion_v1"
)
SCALE_TARGET_COMPLETION_CONTRACT = "hosd_scale_target_completion_v1"
SCALE_TARGET_WAVE_COMPLETION_CONTRACT = "hosd_scale_target_wave_completion_v1"
SCALE_NATIVE_RELATION_WAVE_CONTRACT = (
    "hosd_scale_native_relation_target_wave_v2"
)
SCALE_GRAPH_WAVE_COMPLETION_CONTRACT = "hosd_scale_graph_wave_completion_v1"
SCALE_TRAINING_CHECKPOINT_CONTRACT = "hosd_scale_training_checkpoint_v2"
SCALE_TRAINING_COMPLETION_CONTRACT = "hosd_scale_training_completion_v2"
SCALE_TRAINING_PREDICTION_CONTRACT = "hosd_scale_training_prediction_v1"
STAGE_B_WAVE_COMPLETION_CONTRACT = "hosd_stage_b_wave_completion_v2"
TARGET_NORMALIZATION_WAVE_CONTRACT = "hosd_target_normalization_wave_v1"
TARGET_CONTROL_WAVE_CONTRACT = "hosd_target_control_wave_v1"
TARGET_AUDIT_WAVE_CONTRACT = "hosd_target_audit_wave_v1"
ROW_WAVE_COMPLETION_CONTRACT = "hosd_row_wave_completion_v1"
SCALE_ROW_RESULT_CONTRACT = "hosd_scale_row_result_v3"
SCALE_COMPLETION_CONTRACT = "hosd_scale_completion_v3"
DEPLOYABLE_EXPORT_AUDIT_CONTRACT = "hosd_deployable_export_audit_v1"
CAPACITY_GRID_CONTRACT = "hosd_capacity_grid_v1"
CAPACITY_PROFILE_CONTRACT = "hosd_capacity_profile_v1"
CAPACITY_CONTROL_COMPILATION_CONTRACT = "hosd_capacity_control_compilation_v1"
CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT = "hosd_capacity_control_execution_plan_v1"
CAPACITY_CONTROL_RESULT_CONTRACT = "hosd_capacity_control_result_v2"
STACK_PREDICTION_MANIFEST_CONTRACT = "hosd_stack_prediction_manifest_v1"
STACK_SELECTOR_TRACE_CONTRACT = "hosd_stack_selector_trace_v1"
FINALIST_LOCK_CONTRACT = "hosd_finalist_lock_v1"
POSTLOCK_ORACLE_CONTRACT = "hosd_postlock_oracle_v2"
FINAL_INPUT_PREPARATION_CONTRACT = "hosd_final_input_preparation_v1"
FINAL_EXECUTION_LOCK_CONTRACT = "hosd_final_execution_lock_v1"
FINAL_EXECUTION_CLAIM_CONTRACT = "hosd_final_execution_claim_v1"
FINAL_ROW_RESULT_CONTRACT = "hosd_final_row_result_v1"
FINAL_EVALUATION_CONTRACT = "hosd_final_evaluation_v1"
PRODUCTION_EXECUTION_PLAN_CONTRACT = "hosd_production_execution_plan_v2"
CAMPAIGN_MONITOR_CONTRACT = "hosd_campaign_monitor_v1"
SLURM_SUBMISSION_LEDGER_CONTRACT = "hosd_slurm_submission_ledger_v2"
MINIATURE_ACCEPTANCE_CONTRACT = "hosd_miniature_acceptance_v2"
MINIATURE_CHECK_RECEIPT_CONTRACT = "hosd_miniature_check_receipt_v2"
FULL_AUTHORIZATION_CONTRACT = "hosd_full_authorization_v10"
RESOURCE_PREFLIGHT_CONTRACT = "hosd_resource_preflight_v10"
RESOURCE_MEASUREMENT_EVIDENCE_CONTRACT = (
    "hosd_miniature_resource_measurement_evidence_v9"
)
RESOURCE_MEASUREMENTS_CONTRACT = "hosd_resource_measurements_v10"
INPUT_VIEW_MANIFEST_CONTRACT = "hosd_label_blind_input_view_v4"
PARENT_GROUP_COMPLETION_CONTRACT = "hosd_parent_group_completion_v1"
RUNTIME_MANIFEST_CONTRACT = "hosd_runtime_manifest_v3"
NODE_FACTORY_REGISTRY_CONTRACT = "hosd_node_factory_registry_v1"
COMPLETED_JOB_LEDGER_CONTRACT = "hosd_completed_job_ledger_v1"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def require_safe_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} contains unsafe characters")
    return value


def require_source_equal(
    artifact: Mapping[str, Any],
    *,
    source_snapshot: Mapping[str, Any],
    name: str,
) -> None:
    expected = source_record(source_snapshot)
    if artifact.get("source") != expected:
        raise ValueError(f"{name} is bound to a different source snapshot")


__all__ = [
    "ARTIFACT_LAYOUT_CONTRACT",
    "CAMPAIGN_SPEC_CONTRACT",
    "CANONICAL_JSON_CONTRACT",
    "DESIGN_PARTITION_CONTRACT",
    "DRY_RUN_PLAN_CONTRACT",
    "PARENT_REBUILD_PLAN_CONTRACT",
    "PARENT_STATUS_CONTRACT",
    "REGISTRY_CONTRACT",
    "SOURCE_STATUS_HASH_POLICY",
    "STORAGE_MEASUREMENT_CONTRACT",
    "STORAGE_PROBE_EVIDENCE_CONTRACT",
    "STRUCTURE_TARGET_REGISTRY_CONTRACT",
    "STEP1_REPORT_CONTRACT",
    "TARGET_CACHE_MANIFEST_CONTRACT",
    "TARGET_CACHE_SPEC_CONTRACT",
    "STAGE_B_WAVE_COMPLETION_CONTRACT",
    "TARGET_NORMALIZATION_WAVE_CONTRACT",
    "TARGET_CONTROL_WAVE_CONTRACT",
    "TARGET_AUDIT_WAVE_CONTRACT",
    "ROW_WAVE_COMPLETION_CONTRACT",
    "TARGET_CAPABILITY_AUDIT_CONTRACT",
    "TARGET_AUDIT_CONTRACT",
    "SPLIT_FORWARD_CONTRACT",
    "TARGET_HEAD_CONTRACT",
    "BASELINE_REGISTRY_CONTRACT",
    "STAGE_C_PLAN_CONTRACT",
    "BASELINE_CHECKPOINT_CONTRACT",
    "BASELINE_COMPLETION_CONTRACT",
    "PROBE_ENCODER_LOCK_CONTRACT",
    "PROBE_CHECKPOINT_CONTRACT",
    "PROBE_RESULT_CONTRACT",
    "PROBE_COMPLETION_CONTRACT",
    "PREDICTABILITY_MATRIX_CONTRACT",
    "STAGE_D_PLAN_CONTRACT",
    "AUXILIARY_OBJECTIVE_CONTRACT",
    "AUXILIARY_CHECKPOINT_CONTRACT",
    "AUXILIARY_COMPLETION_CONTRACT",
    "AUXILIARY_PREDICTION_CONTRACT",
    "SINGLE_FAMILY_PHASE_LOCK_CONTRACT",
    "SINGLE_FAMILY_SELECTION_CONTRACT",
    "FEEDBACK_INTERFACE_CONTRACT",
    "STAGE_E_PLAN_CONTRACT",
    "FEEDBACK_CHECKPOINT_CONTRACT",
    "FEEDBACK_RESULT_CONTRACT",
    "FEEDBACK_COMPLETION_CONTRACT",
    "FEEDBACK_SELECTION_CONTRACT",
    "STAGE_F_PLAN_CONTRACT",
    "COMBINATION_RESULT_CONTRACT",
    "COMBINATION_SELECTION_CONTRACT",
    "GRADIENT_CONFLICT_CONTRACT",
    "MECHANISM_CONTROL_PLAN_CONTRACT",
    "MECHANISM_RESULT_CONTRACT",
    "MECHANISM_SUMMARY_CONTRACT",
    "HOSD_METRICS_CONTRACT",
    "ROBUSTNESS_PLAN_CONTRACT",
    "ROBUSTNESS_RESULT_CONTRACT",
    "ROBUSTNESS_SUMMARY_CONTRACT",
    "EFFICIENCY_PROFILE_CONTRACT",
    "PLOT_BUNDLE_CONTRACT",
    "HOSD_REPORT_CONTRACT",
    "HOSD_PAIRED_STATISTICS_CONTRACT",
    "CONFIRMATION_PLAN_CONTRACT",
    "CONFIRMATION_RESULT_CONTRACT",
    "CONFIRMATION_TRAINING_CHECKPOINT_CONTRACT",
    "CONFIRMATION_TRAINING_COMPLETION_CONTRACT",
    "CONFIRMATION_TRAINING_PREDICTION_CONTRACT",
    "CONFIRMATION_WAVE_COMPLETION_CONTRACT",
    "GRAPH_REGISTRY_CONTRACT",
    "CONFIRMATION_SUMMARY_CONTRACT",
    "SCALE_SHORTLIST_CONTRACT",
    "SCALE_EXECUTION_PLAN_CONTRACT",
    "SCALE_TARGET_COMPLETION_CONTRACT",
    "SCALE_TARGET_WAVE_COMPLETION_CONTRACT",
    "SCALE_NATIVE_RELATION_WAVE_CONTRACT",
    "SCALE_GRAPH_WAVE_COMPLETION_CONTRACT",
    "SCALE_TRAINING_CHECKPOINT_CONTRACT",
    "SCALE_TRAINING_COMPLETION_CONTRACT",
    "SCALE_TRAINING_PREDICTION_CONTRACT",
    "SCALE_ROW_RESULT_CONTRACT",
    "SCALE_COMPLETION_CONTRACT",
    "DEPLOYABLE_EXPORT_AUDIT_CONTRACT",
    "CAPACITY_GRID_CONTRACT",
    "CAPACITY_PROFILE_CONTRACT",
    "CAPACITY_CONTROL_COMPILATION_CONTRACT",
    "CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT",
    "CAPACITY_CONTROL_RESULT_CONTRACT",
    "STACK_PREDICTION_MANIFEST_CONTRACT",
    "STACK_SELECTOR_TRACE_CONTRACT",
    "FINALIST_LOCK_CONTRACT",
    "POSTLOCK_ORACLE_CONTRACT",
    "FINAL_INPUT_PREPARATION_CONTRACT",
    "FINAL_EXECUTION_LOCK_CONTRACT",
    "FINAL_EXECUTION_CLAIM_CONTRACT",
    "FINAL_ROW_RESULT_CONTRACT",
    "FINAL_EVALUATION_CONTRACT",
    "PRODUCTION_EXECUTION_PLAN_CONTRACT",
    "CAMPAIGN_MONITOR_CONTRACT",
    "SLURM_SUBMISSION_LEDGER_CONTRACT",
    "MINIATURE_ACCEPTANCE_CONTRACT",
    "MINIATURE_CHECK_RECEIPT_CONTRACT",
    "FULL_AUTHORIZATION_CONTRACT",
    "RESOURCE_PREFLIGHT_CONTRACT",
    "RESOURCE_MEASUREMENTS_CONTRACT",
    "RESOURCE_MEASUREMENT_EVIDENCE_CONTRACT",
    "INPUT_VIEW_MANIFEST_CONTRACT",
    "PARENT_GROUP_COMPLETION_CONTRACT",
    "RUNTIME_MANIFEST_CONTRACT",
    "NODE_FACTORY_REGISTRY_CONTRACT",
    "COMPLETED_JOB_LEDGER_CONTRACT",
    "TARGET_CONTROL_MANIFEST_CONTRACT",
    "TARGET_NORMALIZER_CONTRACT",
    "STREAMED_TARGET_NORMALIZER_CONTRACT",
    "TARGET_SHARD_CONTRACT",
    "TARGET_SHUFFLE_PLAN_CONTRACT",
    "TEACHER_LOCK_CONTRACT",
    "TEACHER_OUTPUT_MANIFEST_CONTRACT",
    "TEACHER_TRAINING_MANIFEST_CONTRACT",
    "LATENT_WHITENING_CONTRACT",
    "RIDGE_ADAPTER_CONTRACT",
    "RESIDUAL_CACHE_SPEC_CONTRACT",
    "CONDITIONAL_RESIDUAL_CONTRACT",
    "HETEROSCEDASTIC_METADATA_CONTRACT",
    "bind_source",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_hashed_json",
    "require_git_object_id",
    "require_safe_id",
    "require_sha256",
    "require_source_equal",
    "source_record",
    "validate_content_hash",
    "with_content_hash",
    "write_immutable_bytes",
    "write_immutable_json",
]

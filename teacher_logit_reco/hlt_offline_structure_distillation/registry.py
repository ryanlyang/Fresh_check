"""Frozen HOSD Stage A--K job, producer, split, and access registries."""

from __future__ import annotations

from typing import Any, Mapping

from .access import build_access_role_registry, validate_access_role_registry
from .contracts import REGISTRY_CONTRACT, validate_content_hash, with_content_hash
from .parents import PARENT_REQUIREMENTS


STAGE_ORDER = tuple("ABCDEFGHIJK")


def _node(
    node_id: str,
    stage: str,
    role: str,
    entrypoint: str,
    dependencies: tuple[str, ...],
    outputs: tuple[str, ...],
    *,
    resource: str = "cpu",
    implementation_step: int,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "stage": stage,
        "worker_role": role,
        "entrypoint": entrypoint,
        "dependencies": list(dependencies),
        "outputs": list(outputs),
        "resource": resource,
        "implementation_step": implementation_step,
        "performance_can_omit_or_cancel": False,
        "failure_policy": "runtime_or_integrity_failure_blocks_dependents_only",
    }


_NODES = (
    _node(
        "campaign_bootstrap",
        "A",
        "campaign_builder",
        "scripts/attest_hosd_bootstrap.py",
        (),
        (
            "campaign_spec.json",
            "registry/stage_job_registry.json",
            "registry/producer_registry.json",
            "registry/access_role_registry.json",
            "registry/split_role_registry.json",
            "registry/parent_requirement_registry.json",
            "job_ledgers/stage_a_to_k_dry_run_plan.json",
        ),
        implementation_step=1,
    ),
    _node(
        "inherited_parent_audit",
        "A",
        "campaign_builder",
        "scripts/attest_hosd_bootstrap.py",
        ("campaign_bootstrap",),
        (
            "inputs/inherited_parent_status.json",
            "inputs/inherited_parent_rebuild_plan.json",
        ),
        implementation_step=1,
    ),
    _node(
        "shared_hlt_parent_rebuild",
        "A",
        "campaign_builder",
        "scripts/build_hosd_shared_hlt_parents.py",
        ("inherited_parent_audit",),
        ("inputs/shared_hlt_parent_completion.json",),
        implementation_step=1,
    ),
    _node(
        "tree_parent_rebuild",
        "A",
        "campaign_builder",
        "scripts/build_hosd_tree_parents.py",
        ("inherited_parent_audit",),
        ("inputs/tree_parent_completion.json",),
        implementation_step=1,
    ),
    _node(
        "relation_normalizer_rebuild",
        "A",
        "target_builder",
        "scripts/fit_hosd_relation_normalizers.py",
        ("shared_hlt_parent_rebuild", "tree_parent_rebuild"),
        ("inputs/relation_normalizer_parent_completion.json",),
        implementation_step=1,
    ),
    _node(
        "resolved_parent_lock",
        "A",
        "campaign_builder",
        "scripts/lock_hosd_inherited_parents.py",
        ("relation_normalizer_rebuild",),
        ("inputs/resolved_inherited_parent_lock.json",),
        implementation_step=1,
    ),
    _node(
        "capability_audit",
        "A",
        "target_builder",
        "scripts/audit_hosd_target_capability.py",
        ("resolved_parent_lock",),
        ("capability/target_capability_audit.json",),
        implementation_step=2,
    ),
    _node(
        "target_registry_compile",
        "A",
        "target_builder",
        "scripts/audit_hosd_target_capability.py",
        ("capability_audit",),
        ("registry/structure_target_registry.json",),
        implementation_step=2,
    ),
    _node(
        "storage_measurement",
        "A",
        "target_builder",
        "scripts/measure_hosd_storage.py",
        ("target_registry_compile",),
        (
            "job_ledgers/storage_probe_evidence.json",
            "job_ledgers/runtime_storage_measurements.json",
        ),
        implementation_step=2,
    ),
    _node(
        "offline_teacher_train",
        "B",
        "teacher_inference",
        "scripts/train_hosd_offline_teacher.py",
        ("storage_measurement",),
        ("teachers/training_manifest.json",),
        resource="gpu",
        implementation_step=4,
    ),
    _node(
        "offline_teacher_lock",
        "B",
        "teacher_inference",
        "scripts/lock_hosd_teachers.py",
        ("offline_teacher_train",),
        ("teachers/teacher_lock.json",),
        implementation_step=4,
    ),
    _node(
        "canonical_target_build",
        "B",
        "target_builder",
        "scripts/build_hosd_targets.py",
        ("offline_teacher_lock",),
        ("targets/canonical/target_manifest.json",),
        implementation_step=4,
    ),
    _node(
        "hlt_analogue_target_build",
        "B",
        "target_builder",
        "scripts/build_hosd_targets.py",
        ("offline_teacher_lock",),
        ("targets/hlt_analogues/completion.json",),
        implementation_step=4,
    ),
    _node(
        "teacher_target_inference",
        "B",
        "teacher_inference",
        "scripts/infer_hosd_teacher_targets.py",
        ("offline_teacher_lock",),
        ("teachers/teacher_output_cache_completion.json",),
        resource="gpu",
        implementation_step=4,
    ),
    _node(
        "teacher_target_finalize",
        "B",
        "teacher_inference",
        "scripts/finalize_hosd_teacher_outputs.py",
        ("teacher_target_inference",),
        ("teachers/teacher_output_manifest.json",),
        implementation_step=4,
    ),
    _node(
        "residual_target_build",
        "B",
        "target_builder",
        "scripts/build_hosd_target_derivatives.py",
        ("canonical_target_build", "hlt_analogue_target_build"),
        ("targets/residuals/completion.json",),
        implementation_step=4,
    ),
    _node(
        "target_normalization",
        "B",
        "target_builder",
        "scripts/execute_hosd_normalization_wave.py",
        ("canonical_target_build", "residual_target_build", "teacher_target_finalize"),
        (
            "normalization/target_500k/normalizer_manifest.json",
            "normalization/residual_500k/normalizer_manifest.json",
            "normalization/target_500k/heteroscedastic_metadata.json",
            "normalization/target_500k/latent_whitening.json",
            "normalization/residual_500k/conditional_completion.json",
        ),
        implementation_step=4,
    ),
    _node(
        "target_controls",
        "B",
        "label_auditor",
        "scripts/execute_hosd_control_wave.py",
        ("target_normalization",),
        ("targets/controls/control_plan_completion.json",),
        implementation_step=4,
    ),
    _node(
        "target_audit",
        "B",
        "label_auditor",
        "scripts/execute_hosd_target_audit_wave.py",
        ("target_controls",),
        ("targets/target_audit.json",),
        implementation_step=4,
    ),
    _node(
        "baseline_train",
        "C",
        "train_worker",
        "scripts/train_hosd_baseline.py",
        ("target_audit",),
        ("baselines/baseline_completion.json",),
        resource="gpu",
        implementation_step=5,
    ),
    _node(
        "probe_tap_capture",
        "C",
        "probe_worker",
        "scripts/build_hosd_probe_taps.py",
        ("baseline_train",),
        (
            "probes/frozen_taps/probe_encoder_lock.json",
            "probes/frozen_taps/tap_cache_manifest.json",
        ),
        resource="gpu",
        implementation_step=5,
    ),
    _node(
        "probe_input_materialization",
        "C",
        "probe_worker",
        "scripts/materialize_hosd_probe_inputs.py",
        ("probe_tap_capture",),
        ("probes/inputs/input_completion.json",),
        implementation_step=5,
    ),
    _node(
        "probe_train",
        "C",
        "probe_worker",
        "scripts/train_hosd_probe.py",
        ("probe_input_materialization",),
        ("probes/probe_completion.json",),
        resource="gpu",
        implementation_step=5,
    ),
    _node(
        "predictability_aggregate",
        "C",
        "design_selector",
        "scripts/aggregate_hosd_predictability.py",
        ("probe_train",),
        ("probes/predictability_matrix.json",),
        implementation_step=5,
    ),
    _node(
        "auxiliary_train",
        "D",
        "train_worker",
        "scripts/train_hosd_auxiliary.py",
        ("predictability_aggregate",),
        ("auxiliary/primary_scientific_completion.json",),
        resource="gpu",
        implementation_step=6,
    ),
    _node(
        "single_family_phase_lock",
        "D",
        "design_selector",
        "scripts/select_hosd_single_targets.py",
        ("auxiliary_train",),
        ("auxiliary/single_family_phase_lock.json",),
        implementation_step=6,
    ),
    _node(
        "relation_het_auxiliary_train",
        "D",
        "train_worker",
        "scripts/train_hosd_auxiliary.py",
        ("single_family_phase_lock",),
        ("auxiliary/relation_het_completion.json",),
        resource="gpu",
        implementation_step=6,
    ),
    _node(
        "hlt_self_auxiliary_train",
        "D",
        "train_worker",
        "scripts/train_hosd_auxiliary.py",
        ("single_family_phase_lock",),
        ("auxiliary/hlt_self_control_completion.json",),
        resource="gpu",
        implementation_step=6,
    ),
    _node(
        "auxiliary_controls",
        "D",
        "train_worker",
        "scripts/train_hosd_auxiliary.py",
        ("predictability_aggregate",),
        ("auxiliary/null_control_completion.json",),
        resource="gpu",
        implementation_step=6,
    ),
    _node(
        "single_family_select",
        "D",
        "design_selector",
        "scripts/select_hosd_single_targets.py",
        (
            "relation_het_auxiliary_train",
            "hlt_self_auxiliary_train",
            "auxiliary_controls",
        ),
        ("auxiliary/locked_single_family_choices.json",),
        implementation_step=6,
    ),
    _node(
        "feedback_train",
        "E",
        "train_worker",
        "scripts/train_hosd_feedback.py",
        ("single_family_select",),
        ("feedback/scientific_row_completion.json",),
        resource="gpu",
        implementation_step=7,
    ),
    _node(
        "feedback_controls",
        "E",
        "train_worker",
        "scripts/train_hosd_feedback.py",
        ("single_family_select", "feedback_train"),
        ("feedback/mechanism_control_completion.json",),
        resource="gpu",
        implementation_step=7,
    ),
    _node(
        "feedback_select",
        "E",
        "design_selector",
        "scripts/select_hosd_feedback.py",
        ("feedback_train", "feedback_controls"),
        ("feedback/locked_feedback_choices.json",),
        implementation_step=7,
    ),
    _node(
        "combination_beam",
        "F",
        "train_worker",
        "scripts/train_hosd_combination.py",
        ("feedback_select",),
        ("combinations/beam_completion.json",),
        resource="gpu",
        implementation_step=8,
    ),
    _node(
        "combination_train",
        "F",
        "train_worker",
        "scripts/train_hosd_combination.py",
        ("combination_beam",),
        ("combinations/full_completion.json",),
        resource="gpu",
        implementation_step=8,
    ),
    _node(
        "pcgrad_control",
        "F",
        "train_worker",
        "scripts/train_hosd_combination.py",
        ("combination_beam",),
        ("combinations/pcgrad_completion.json",),
        resource="gpu",
        implementation_step=8,
    ),
    _node(
        "combination_select",
        "F",
        "design_selector",
        "scripts/train_hosd_combination.py",
        ("combination_train", "pcgrad_control"),
        ("combinations/locked_combination_choices.json",),
        implementation_step=8,
    ),
    _node(
        "mechanism_controls",
        "G",
        "design_confirmer",
        "scripts/run_hosd_mechanism_controls.py",
        ("combination_select",),
        ("mechanism_controls/control_completion.json",),
        resource="gpu",
        implementation_step=8,
    ),
    _node(
        "mechanism_confirm",
        "G",
        "design_confirmer",
        "scripts/run_hosd_mechanism_controls.py",
        ("mechanism_controls",),
        ("mechanism_controls/design_confirm_summary.json",),
        implementation_step=8,
    ),
    _node(
        "robustness_cache_build",
        "H",
        "target_builder",
        "scripts/build_hosd_robustness_cache.py",
        ("mechanism_confirm",),
        ("robustness/cache_completion.json",),
        resource="cpu",
        implementation_step=9,
    ),
    _node(
        "robustness_plan_compile",
        "H",
        "reporter",
        "scripts/evaluate_hosd_robustness.py",
        ("mechanism_confirm", "graph_registry_compile"),
        ("robustness/evaluation_plan.json",),
        implementation_step=9,
    ),
    _node(
        "robustness_evaluation",
        "H",
        "design_inference",
        "scripts/evaluate_hosd_robustness.py",
        (
            "robustness_cache_build",
            "robustness_plan_compile",
            "discovery_export",
        ),
        ("robustness/evaluation_completion.json",),
        resource="gpu",
        implementation_step=9,
    ),
    _node(
        "robustness_report",
        "H",
        "reporter",
        "scripts/evaluate_hosd_robustness.py",
        ("robustness_evaluation",),
        ("robustness/summary.json",),
        implementation_step=9,
    ),
    _node(
        "graph_registry_compile",
        "H",
        "design_confirmer",
        "scripts/build_hosd_graph_registry.py",
        ("mechanism_confirm",),
        ("registry/locked_graph_registry.json",),
        implementation_step=10,
    ),
    _node(
        "confirmation_native_relation_build",
        "I",
        "target_builder",
        "scripts/build_hosd_confirmation_native_relations.py",
        ("graph_registry_compile", "robustness_report"),
        (
            "targets/native_relations/design_confirm/replica_0.npz",
            "targets/native_relations/design_confirm/replica_0.manifest.json",
        ),
        implementation_step=10,
    ),
    _node(
        "confirmation_compile",
        "I",
        "reporter",
        "scripts/aggregate_hosd_confirmation.py",
        ("graph_registry_compile", "robustness_report"),
        ("confirmation_500k/execution_plan.json",),
        implementation_step=10,
    ),
    _node(
        "discovery_export",
        "H",
        "reporter",
        "scripts/execute_hosd_discovery_export.py",
        ("graph_registry_compile",),
        ("confirmation_500k/discovery_export_completion.json",),
        resource="gpu",
        implementation_step=10,
    ),
    _node(
        "confirmation_train",
        "I",
        "train_worker",
        "scripts/execute_hosd_confirmation_row.py",
        (
            "confirmation_compile",
            "discovery_export",
            "confirmation_native_relation_build",
        ),
        ("confirmation_500k/training_completion.json",),
        resource="gpu",
        implementation_step=10,
    ),
    _node(
        "capacity_compile",
        "I",
        "reporter",
        "scripts/compile_hosd_capacity_wave.py",
        ("confirmation_compile", "discovery_export"),
        ("confirmation_500k/capacity_execution_plan.json",),
        implementation_step=10,
    ),
    _node(
        "capacity_controls",
        "I",
        "train_worker",
        "scripts/execute_hosd_capacity_control_row.py",
        ("capacity_compile",),
        ("confirmation_500k/capacity_control_completion.json",),
        resource="gpu",
        implementation_step=10,
    ),
    _node(
        "confirmation_aggregate",
        "I",
        "reporter",
        "scripts/aggregate_hosd_confirmation.py",
        ("confirmation_train", "capacity_controls", "capacity_compile"),
        ("confirmation_500k/summary.json",),
        implementation_step=10,
    ),
    _node(
        "scale_shortlist",
        "I",
        "design_selector",
        "scripts/select_hosd_scale_shortlist.py",
        ("confirmation_aggregate",),
        ("selection/locked_scale_shortlist.json",),
        implementation_step=10,
    ),
    _node(
        "scale_plan_compile",
        "J",
        "reporter",
        "scripts/train_hosd_scale.py",
        ("scale_shortlist",),
        ("scale_up/execution_plan.json",),
        implementation_step=10,
    ),
    _node(
        "scale_input_prepare",
        "J",
        "target_builder",
        "scripts/prepare_hosd_scale_inputs.py",
        ("scale_plan_compile",),
        ("scale_up/inputs/completion.json",),
        implementation_step=10,
    ),
    _node(
        "scale_tree_build",
        "J",
        "target_builder",
        "scripts/build_hosd_scale_tree.py",
        ("scale_input_prepare",),
        ("scale_up/trees/completion.json",),
        implementation_step=10,
    ),
    _node(
        "scale_normalization",
        "J",
        "target_builder",
        "scripts/fit_hosd_scale_normalizers.py",
        ("scale_tree_build",),
        ("scale_up/normalization/completion.json",),
        implementation_step=10,
    ),
    _node(
        "scale_teacher_train",
        "J",
        "teacher_inference",
        "scripts/execute_hosd_scale_row.py",
        ("scale_normalization",),
        ("scale_up/teacher_completion.json",),
        resource="gpu",
        implementation_step=10,
    ),
    _node(
        "scale_teacher_lock",
        "J",
        "teacher_inference",
        "scripts/execute_hosd_scale_row.py",
        ("scale_teacher_train",),
        ("scale_up/teachers/teacher_lock.json",),
        implementation_step=10,
    ),
    _node(
        "scale_teacher_adapter_compile",
        "J",
        "teacher_inference",
        "scripts/compile_hosd_scale_teacher_adapters.py",
        ("scale_teacher_lock",),
        ("scale_up/teacher_outputs/adapter_configs/completion.json",),
        implementation_step=10,
    ),
    _node(
        "scale_teacher_target_inference",
        "J",
        "teacher_inference",
        "scripts/execute_hosd_scale_row.py",
        ("scale_teacher_adapter_compile",),
        ("scale_up/teacher_outputs/completion.json",),
        resource="gpu",
        implementation_step=10,
    ),
    _node(
        "scale_target_build",
        "J",
        "target_builder",
        "scripts/execute_hosd_scale_row.py",
        ("scale_teacher_target_inference",),
        ("scale_up/target_completion.json",),
        implementation_step=10,
    ),
    _node(
        "scale_native_relation_build",
        "J",
        "target_builder",
        "scripts/build_hosd_scale_native_relations.py",
        ("scale_target_build",),
        ("scale_up/targets/native_relations/completion.json",),
        implementation_step=10,
    ),
    _node(
        "scale_graph_train",
        "J",
        "train_worker",
        "scripts/execute_hosd_scale_row.py",
        ("scale_native_relation_build",),
        ("scale_up/graph_completion.json",),
        resource="gpu",
        implementation_step=10,
    ),
    _node(
        "scale_finalize",
        "J",
        "reporter",
        "scripts/train_hosd_scale.py",
        ("scale_graph_train",),
        ("scale_up/completion.json",),
        implementation_step=10,
    ),
    _node(
        "scale_export_audit",
        "J",
        "reporter",
        "scripts/audit_hosd_deployment.py",
        ("scale_finalize",),
        ("scale_up/export_audit.json",),
        implementation_step=10,
    ),
    _node(
        "scale_efficiency",
        "J",
        "reporter",
        "scripts/profile_hosd_scale_efficiency.py",
        ("scale_export_audit",),
        ("scale_up/efficiency/completion.json",),
        resource="gpu",
        implementation_step=10,
    ),
    _node(
        "stack_capacity_compile",
        "K",
        "reporter",
        "scripts/build_hosd_stack_capacity.py",
        ("scale_export_audit", "scale_efficiency"),
        ("selection_predictions/stack_val/capacity.json",),
        implementation_step=11,
    ),
    _node(
        "stack_inference",
        "K",
        "stack_inference",
        "scripts/infer_hosd_stack_val.py",
        ("scale_export_audit", "stack_capacity_compile"),
        ("selection_predictions/stack_val/completion.json",),
        resource="gpu",
        implementation_step=11,
    ),
    _node(
        "stack_selector",
        "K",
        "stack_selector",
        "scripts/select_hosd_finalists.py",
        ("stack_inference", "stack_capacity_compile"),
        ("selection/stack_selector_trace.json",),
        implementation_step=11,
    ),
    _node(
        "finalist_lock",
        "K",
        "stack_selector",
        "scripts/select_hosd_finalists.py",
        ("stack_selector",),
        ("selection/locked_hosd_finalists.json",),
        implementation_step=11,
    ),
    _node(
        "postlock_oracle",
        "K",
        "postlock_oracle_diagnostic",
        "scripts/build_hosd_postlock_oracles.py",
        ("finalist_lock",),
        ("postlock_oracle_diagnostics/completion.json",),
        resource="gpu",
        implementation_step=11,
    ),
    _node(
        "finalist_controls",
        "K",
        "train_worker",
        "scripts/build_hosd_finalist_controls.py",
        ("finalist_lock",),
        ("selection/finalist_control_completion.json",),
        resource="gpu",
        implementation_step=11,
    ),
    _node(
        "final_input_preparation",
        "K",
        "final_input_preparer",
        "scripts/write_hosd_final_test_execution_lock.py",
        ("finalist_lock",),
        ("final_test/prepared_inputs.json",),
        implementation_step=11,
    ),
    _node(
        "execution_lock",
        "K",
        "reporter",
        "scripts/write_hosd_final_test_execution_lock.py",
        ("postlock_oracle", "finalist_controls", "final_input_preparation"),
        ("selection/final_test_execution_lock.json",),
        implementation_step=11,
    ),
    _node(
        "final_test",
        "K",
        "final_inference",
        "scripts/evaluate_hosd_final_test.py",
        ("execution_lock",),
        ("final_test/final_evaluation.json",),
        resource="gpu",
        implementation_step=11,
    ),
    _node(
        "final_report",
        "K",
        "reporter",
        "scripts/write_hosd_report.py",
        ("final_test",),
        ("reports/final_report.json", "reports/final_report.md"),
        implementation_step=11,
    ),
    _node(
        "final_plots",
        "K",
        "reporter",
        "scripts/plot_hosd_report.py",
        ("final_report",),
        (
            "reports/plots/manifest.json",
            "reports/plots/balanced_accuracy_difference.png",
            "reports/plots/mean_log_rejection_difference.png",
        ),
        implementation_step=11,
    ),
    _node(
        "campaign_completion",
        "K",
        "reporter",
        "scripts/complete_hosd_job_ledger.py",
        ("final_plots",),
        ("job_ledgers/completed_job_ledger.json",),
        implementation_step=12,
    ),
)


def build_stage_job_registry(*, source: Mapping[str, Any]) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": REGISTRY_CONTRACT,
            "schema_version": 1,
            "registry_id": "stage_job_registry",
            "stage_order": list(STAGE_ORDER),
            "nodes": [dict(node) for node in _NODES],
            "node_count": len(_NODES),
            "all_stages_enumerated": True,
            "scientific_underperformance_can_omit_registered_node": False,
            "source": dict(source),
        }
    )


def build_producer_registry(
    stage_jobs: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_stage_job_registry(stage_jobs)
    entries = []
    seen: set[str] = set()
    for node in stage_jobs["nodes"]:
        for output in node["outputs"]:
            if output in seen:
                raise ValueError(f"duplicate producer output {output!r}")
            seen.add(output)
            entries.append(
                {
                    "artifact": output,
                    "producer_node_id": node["node_id"],
                    "entrypoint": node["entrypoint"],
                    "stage": node["stage"],
                    "implementation_step": node["implementation_step"],
                }
            )
    return with_content_hash(
        {
            "contract": REGISTRY_CONTRACT,
            "schema_version": 1,
            "registry_id": "producer_registry",
            "stage_job_registry_sha256": stage_jobs["content_hash"],
            "entries": sorted(entries, key=lambda row: row["artifact"]),
            "entry_count": len(entries),
            "every_output_has_exactly_one_producer": True,
            "source": dict(source),
        }
    )


def build_split_role_registry(*, source: Mapping[str, Any]) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": REGISTRY_CONTRACT,
            "schema_version": 1,
            "registry_id": "split_role_registry",
            "roles": [
                {
                    "repository_split": "model_train",
                    "cache_logical_role": "model_train",
                    "access_role": "model_train",
                },
                {
                    "repository_split": "model_val",
                    "cache_logical_role": "val_stop",
                    "access_role": "val_stop",
                },
                {
                    "repository_split": "model_val",
                    "cache_logical_role": "val_design",
                    "access_role": "val_design",
                },
                {
                    "repository_split": "stack_val",
                    "cache_logical_role": "stack_val",
                    "access_role": "final_select",
                },
                {
                    "repository_split": "final_test",
                    "cache_logical_role": "final_test",
                    "access_role": "final_test",
                },
            ],
            "design_subroles": ["design_select", "design_confirm"],
            "stack_train_role": "unused",
            "source": dict(source),
        }
    )


def build_parent_requirement_registry(*, source: Mapping[str, Any]) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": REGISTRY_CONTRACT,
            "schema_version": 1,
            "registry_id": "parent_requirement_registry",
            "requirements": [
                {
                    "parent_id": item.parent_id,
                    "expected_contract": item.expected_contract,
                    "required_before_stage": item.required_before_stage,
                    "canonical_path": item.canonical_path,
                    "rebuild_entrypoint": item.rebuild_entrypoint,
                    "rebuild_group": item.rebuild_group,
                }
                for item in PARENT_REQUIREMENTS
            ],
            "semantic_aliasing_allowed": False,
            "source_drift_reuse_allowed": False,
            "source": dict(source),
        }
    )


def build_registries(*, source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    stage = build_stage_job_registry(source=source)
    return {
        "access_role_registry": build_access_role_registry(source=source),
        "parent_requirement_registry": build_parent_requirement_registry(
            source=source
        ),
        "producer_registry": build_producer_registry(stage, source=source),
        "split_role_registry": build_split_role_registry(source=source),
        "stage_job_registry": stage,
    }


def validate_stage_job_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(payload, expected_contract=REGISTRY_CONTRACT)
    if payload.get("registry_id") != "stage_job_registry":
        raise ValueError("stage-job registry ID differs")
    nodes = list(payload.get("nodes", ()))
    ids = [str(node.get("node_id")) for node in nodes]
    if len(ids) != len(set(ids)) or len(nodes) != int(payload.get("node_count", -1)):
        raise ValueError("stage-job node coverage differs")
    by_id = {node["node_id"]: node for node in nodes}
    if set(node["stage"] for node in nodes) != set(STAGE_ORDER):
        raise ValueError("stage-job registry does not cover Stage A through K")
    stage_index = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    for node in nodes:
        if node["worker_role"] not in {
            *build_access_role_registry(source=payload["source"])["roles"].keys()
        }:
            raise ValueError(f"node {node['node_id']} has unknown access role")
        for dependency in node["dependencies"]:
            if dependency not in by_id:
                raise ValueError(f"node {node['node_id']} has unknown dependency")
            if stage_index[by_id[dependency]["stage"]] > stage_index[node["stage"]]:
                raise ValueError("stage-job dependency points to a later stage")
        if bool(node["performance_can_omit_or_cancel"]):
            raise ValueError("scientific performance can omit a registered job")
    return digest


def validate_registry(payload: Mapping[str, Any]) -> str:
    registry_id = payload.get("registry_id")
    if registry_id == "stage_job_registry":
        return validate_stage_job_registry(payload)
    if registry_id == "access_role_registry":
        return validate_access_role_registry(payload)
    digest = validate_content_hash(payload, expected_contract=REGISTRY_CONTRACT)
    if registry_id not in {
        "parent_requirement_registry",
        "producer_registry",
        "split_role_registry",
    }:
        raise ValueError(f"unknown HOSD registry {registry_id!r}")
    return digest


__all__ = [
    "STAGE_ORDER",
    "build_parent_requirement_registry",
    "build_producer_registry",
    "build_registries",
    "build_split_role_registry",
    "build_stage_job_registry",
    "validate_registry",
    "validate_stage_job_registry",
]

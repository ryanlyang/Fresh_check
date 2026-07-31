"""Authenticated Step-15 Tigris production graph and execution ledgers.

The production graph is intentionally independent of training imports.  It
describes every Stage A--N dependency, while runtime task manifests resolve
the exact rows of arrays whose size depends on an earlier immutable selector.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    require_git_object_id,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .plan_factory_registry import (
    MANIFEST_PLAN_FACTORY_REGISTRY_CONTRACT,
    build_manifest_plan_factory_registry,
    validate_manifest_plan_factory_registry,
)


PRODUCTION_GRAPH_CONTRACT = "retb_tigris_production_graph_v29"
NODE_EXECUTION_REGISTRY_CONTRACT = (
    "retb_production_node_execution_registry_v14"
)
JOB_LEDGER_CONTRACT = "retb_tigris_job_ledger_v2"
RESOURCE_PROBE_CONTRACT = "retb_tigris_resource_probe_v1"
TARGET_SHARD_PLAN_CONTRACT = "retb_target_shard_execution_plan_v1"
TASK_MANIFEST_CONTRACT = "retb_tigris_task_manifest_v1"
RESUME_PLAN_CONTRACT = "retb_tigris_resume_plan_v1"
STEP15_BUNDLE_CONTRACT = "retb_step15_production_bundle_v23"

TIGRIS_DEFAULTS = {
    "project_dir": "/home/ryreu/atlas/Fresh_check",
    "data_dir": "/home/ryreu/atlas/PracticeTagging/data",
    "output_root": "/home/ryreu/atlas/Fresh_check/checkpoints",
    "conda_base": "/home/ryreu/miniforge3-aarch64",
    "conda_env": "atlas_kd_tigris",
    "python_no_user_site": "1",
    "account": "reu-aisocial",
    "partition": "tigris",
    "gpu_gres": "gpu:gh200:1",
    "gpu_cpus_per_task": 16,
    "gpu_memory": "220G",
    "cpu_cpus_per_task": 16,
    "cpu_memory": "192G",
}

DEFAULT_CONCURRENCY = {
    "cpu_cache": 64,
    "gpu_expert": 64,
    "gpu_predictor": 64,
    "gpu_scale": 64,
    "gpu_final": 64,
}

PRODUCTION_SPLIT_SIZES = {
    "model_train": 500_000,
    "model_val": 100_000,
    "stack_train": 0,
    "stack_val": 50_000,
    "final_test": 300_000,
    "scale_train": 3_000_000,
}
MINIATURE_SPLIT_SIZES = {
    "model_train": 400,
    "model_val": 640,
    "stack_train": 0,
    "stack_val": 100,
    "final_test": 320,
    "scale_train": 400,
}

# These nodes are submitted as their declared worker directly.  Every other
# non-alias node is executed through an authenticated task manifest.
DIRECT_WORKER_NODES = frozenset(
    {
        "split_build",
        "campaign_bootstrap",
        "compiled_region_backend",
        "cpu_resource_probe",
        "gpu_resource_probe",
        "step3_architecture_contracts",
        "step4_offline_training_contracts",
        "step5_offline_fusion_contracts",
        "step6_native_hlt_contracts",
        "step7_bridge_contracts",
        "step8_target_cache_contracts",
        "step9_predictor_contracts",
        "step10_predictor_bundle_contracts",
        "step11_joint_bridge_contracts",
        "step12_final_consumer_contracts",
        "step13_confirmation_contracts",
        "step14_scale_final_contracts",
        "stage_n_evidence_join",
        "completed_job_ledger",
    }
)

# Explicit ownership is deliberate: adding a manifest-driven graph node without
# adding it here is a contract failure.  Task 2--7 producers may materialize
# different row contents, but the responsible upstream invocation is frozen
# here before any campaign is submitted.
TASK_MANIFEST_PRODUCER_NODES = {
    "offline_input_cache": "campaign_bootstrap",
    "hlt_v3_cache": "campaign_bootstrap",
    "region_tree_cache": "campaign_bootstrap",
    "region_tree_finalize": "campaign_bootstrap",
    "input_audit": "campaign_bootstrap",
    "normalizers_500k": "campaign_bootstrap",
    "offline_expert_training": "campaign_bootstrap",
    "offline_expert_confirmation": "campaign_bootstrap",
    "offline_fusion_cache": "campaign_bootstrap",
    "offline_shape_selector": "offline_fusion_training",
    "offline_optimization_selector": "offline_expert_training",
    "offline_fusion_training": "campaign_bootstrap",
    "offline_complementarity": "offline_fusion_training",
    "offline_capacity_controls": "offline_shape_selector",
    "native_hlt_expert_training": "campaign_bootstrap",
    "native_hlt_fusion_training": "campaign_bootstrap",
    "bridge_pilot_training": "campaign_bootstrap",
    "bridge_target_training": "bridge_pilot_training",
    "bridge_content_certification": "bridge_target_training",
    "target_coordinate_selector": "bridge_content_certification",
    "target_cache_build": "step8_target_cache_contracts",
    "target_normalizers": "target_cache_build",
    "predictor_training": "step9_predictor_contracts",
    "uncertainty_calibration": "predictor_training",
    "predictor_bundle_selector": "step10_predictor_bundle_contracts",
    "oracle_substitutions": "predictor_bundle_selector",
    "joint_predictor_training": "step11_joint_bridge_contracts",
    "joint_predictor_selector": "joint_predictor_training",
    "final_consumer_training": "step12_final_consumer_contracts",
    "deployable_export": "final_consumer_training",
    "robustness_controls": "deployable_export",
    "semantic_controls": "deployable_export",
    "stage_l_graph_registration": "step13_confirmation_contracts",
    "confirmation_500k": "stage_l_graph_registration",
    "confirmation_summary": "confirmation_500k",
    "bridge_shape_selector": "confirmation_summary",
    "scale_shortlist_selector": "bridge_shape_selector",
    "shortlisted_500k_control_training": "scale_shortlist_selector",
    "shortlisted_500k_controls": "shortlisted_500k_control_training",
    "scale_refit_normalizers": "step14_scale_final_contracts",
    "scale_refit_teachers": "scale_refit_normalizers",
    "scale_refit_offline_experts": "scale_refit_teachers",
    "scale_refit_targets": "scale_refit_offline_experts",
    "scale_refit_native": "scale_refit_targets",
    "scale_refit_native_fusion": "scale_refit_native",
    "scale_refit_predictors": "scale_refit_native_fusion",
    "scale_refit_calibrations": "scale_refit_predictors",
    "scale_refits": "scale_refit_calibrations",
    "scale_joint_training": "scale_refits",
    "scale_graph_datasets": "scale_joint_training",
    "scale_refiner_training": "scale_graph_datasets",
    "scale_final_consumer_training": "scale_refiner_training",
    "scale_graph_export": "scale_final_consumer_training",
    "scale_graph_training": "scale_graph_export",
    "scale_completion": "scale_graph_training",
    "prelock_final_inputs": "input_audit",
    "stack_val_inference": "scale_completion",
    "accuracy_finalist_selector": "stack_val_inference",
    "postlock_oracle_targets": "accuracy_finalist_selector",
    "finalist_controls": "accuracy_finalist_selector",
    "final_test_execution_lock": "stage_n_evidence_join",
    "sealed_final_test": "final_test_execution_lock",
    "final_report": "sealed_final_test",
}

BOOTSTRAP_INPUT_MANIFEST_NODES = frozenset(
    {
        "offline_input_cache",
        "hlt_v3_cache",
        "region_tree_cache",
        "region_tree_finalize",
        "normalizers_500k",
        "input_audit",
    }
)

STATIC_EXPERIMENT_MANIFEST_NODES = frozenset(
    {
        "offline_expert_training",
        "offline_expert_confirmation",
        "offline_fusion_cache",
        "offline_fusion_training",
        "native_hlt_expert_training",
        "native_hlt_fusion_training",
        "bridge_pilot_training",
    }
)

MIDDLE_CONTINUATION_MANIFEST_NODES = frozenset(
    {
        "target_cache_build",
        "target_normalizers",
        "predictor_training",
        "uncertainty_calibration",
        "predictor_bundle_selector",
        "oracle_substitutions",
        "joint_predictor_training",
        "joint_predictor_selector",
        "final_consumer_training",
        "deployable_export",
    }
)

LATE_CONTINUATION_MANIFEST_NODES = frozenset(
    {
        "robustness_controls",
        "semantic_controls",
        "stage_l_graph_registration",
        "confirmation_500k",
        "confirmation_summary",
        "bridge_shape_selector",
        "scale_shortlist_selector",
        "shortlisted_500k_control_training",
        "shortlisted_500k_controls",
        "scale_refit_normalizers",
        "scale_refit_teachers",
        "scale_refit_offline_experts",
        "scale_refit_targets",
        "scale_refit_native",
        "scale_refit_native_fusion",
        "scale_refit_predictors",
        "scale_refit_calibrations",
        "scale_refits",
        "scale_joint_training",
        "scale_graph_datasets",
        "scale_refiner_training",
        "scale_final_consumer_training",
        "scale_graph_export",
        "scale_graph_training",
        "scale_completion",
    }
)
LATE_CONTINUATION_GATE_CONTRACT = (
    "retb_stage_k_m_continuation_gate_v1"
)
LATE_NODE_ENTRYPOINTS: dict[str, tuple[str, ...]] = {
    "robustness_controls": (
        "scripts/execute_retb_robustness_campaign.py",
        "scripts/evaluate_retb_final_consumer_reference.py",
        "scripts/evaluate_retb_final_consumer_bypass_controls.py",
        "scripts/evaluate_retb_stage_i_substitutions.py",
    ),
    "semantic_controls": (
        "scripts/execute_retb_semantic_control_campaign.py",
        "scripts/evaluate_retb_relation_predictor_semantics.py",
        "scripts/evaluate_retb_bias_scale_semantics.py",
        "scripts/evaluate_retb_final_consumer_bypass_controls.py",
        "scripts/evaluate_retb_stage_i_substitutions.py",
        "scripts/finalize_retb_semantic_control_campaign.py",
    ),
    "stage_l_graph_registration": (
        "scripts/execute_retb_stage_l_registration.py",
        "scripts/register_retb_stage_l_graphs.py",
    ),
    "confirmation_500k": (
        "scripts/execute_retb_500k_seed_confirmation.py",
    ),
    "confirmation_summary": ("scripts/aggregate_retb_confirmation.py",),
    "bridge_shape_selector": ("scripts/select_retb_bridge_shape.py",),
    "scale_shortlist_selector": (
        "scripts/select_retb_scale_shortlist.py",
    ),
    "shortlisted_500k_control_training": (
        "scripts/train_retb_scale_finalist_control.py",
    ),
    "shortlisted_500k_controls": (
        "scripts/aggregate_retb_shortlisted_500k_controls.py",
    ),
    "scale_refits": ("scripts/execute_retb_scale_seed_refit.py",),
    "scale_refit_normalizers": ("scripts/execute_retb_scale_seed_refit.py",),
    "scale_refit_teachers": ("scripts/execute_retb_scale_seed_refit.py",),
    "scale_refit_offline_experts": ("scripts/execute_retb_scale_seed_refit.py",),
    "scale_refit_targets": ("scripts/execute_retb_scale_seed_refit.py",),
    "scale_refit_native": ("scripts/execute_retb_scale_seed_refit.py",),
    "scale_refit_native_fusion": (
        "scripts/execute_retb_scale_seed_refit.py",
    ),
    "scale_refit_predictors": ("scripts/execute_retb_scale_seed_refit.py",),
    "scale_refit_calibrations": ("scripts/execute_retb_scale_seed_refit.py",),
    "scale_graph_training": (
        "scripts/execute_retb_scale_graph_pipeline.py",
    ),
    "scale_joint_training": ("scripts/execute_retb_scale_graph_pipeline.py",),
    "scale_graph_datasets": ("scripts/execute_retb_scale_graph_pipeline.py",),
    "scale_refiner_training": ("scripts/execute_retb_scale_graph_pipeline.py",),
    "scale_final_consumer_training": ("scripts/execute_retb_scale_graph_pipeline.py",),
    "scale_graph_export": ("scripts/execute_retb_scale_graph_pipeline.py",),
    "scale_completion": (
        "scripts/aggregate_retb_scale_completion.py",
    ),
}
if set(LATE_NODE_ENTRYPOINTS) != set(
    LATE_CONTINUATION_MANIFEST_NODES
):
    raise RuntimeError("Stage K--M entry-point coverage differs")

FINAL_CONTINUATION_MANIFEST_NODES = frozenset(
    {
        "prelock_final_inputs",
        "stack_val_inference",
        "accuracy_finalist_selector",
        "postlock_oracle_targets",
        "finalist_controls",
        "final_test_execution_lock",
        "sealed_final_test",
        "final_report",
    }
)
FINAL_CONTINUATION_GATE_CONTRACT = (
    "retb_stage_n_continuation_gate_v1"
)
FINAL_NODE_ENTRYPOINTS: dict[str, tuple[str, ...]] = {
    "prelock_final_inputs": (
        "scripts/prepare_retb_final_test_inputs.py",
    ),
    "stack_val_inference": (
        "scripts/execute_retb_stack_val_inference.py",
    ),
    "accuracy_finalist_selector": (
        "scripts/select_retb_scale_finalists.py",
    ),
    "postlock_oracle_targets": (
        "scripts/execute_retb_postlock_oracle_target.py",
    ),
    "finalist_controls": (
        "scripts/train_retb_scale_finalist_control.py",
    ),
    "final_test_execution_lock": (
        "scripts/write_retb_final_test_execution_lock.py",
    ),
    "sealed_final_test": (
        "scripts/execute_retb_sealed_final_test.py",
    ),
    "final_report": (
        "scripts/write_retb_step14_report.py",
    ),
}
if set(FINAL_NODE_ENTRYPOINTS) != set(
    FINAL_CONTINUATION_MANIFEST_NODES
):
    raise RuntimeError("Stage-N entry-point coverage differs")

MIDDLE_CONTINUATION_GATE_CONTRACT = (
    "retb_stage_f_j_continuation_gate_v1"
)
MIDDLE_NODE_ENTRYPOINTS: dict[str, str] = {
    "target_cache_build": "scripts/execute_retb_target_cache_row.py",
    "target_normalizers": "scripts/fit_retb_target_normalizers.py",
    "predictor_training": "scripts/execute_retb_predictor_campaign.py",
    "uncertainty_calibration": "scripts/calibrate_retb_uncertainty.py",
    "predictor_bundle_selector": (
        "scripts/execute_retb_predictor_bundle_selection.py"
    ),
    "oracle_substitutions": (
        "scripts/execute_retb_stage_i_oracle_wave.py"
    ),
    "joint_predictor_training": "scripts/execute_retb_joint_campaign.py",
    "joint_predictor_selector": "scripts/finalize_retb_joint_campaign.py",
    "final_consumer_training": (
        "scripts/execute_retb_final_consumer_campaign.py"
    ),
    "deployable_export": (
        "scripts/execute_retb_deployable_export_campaign.py"
    ),
}
if set(MIDDLE_NODE_ENTRYPOINTS) != set(
    MIDDLE_CONTINUATION_MANIFEST_NODES
):
    raise RuntimeError("Stage F--J entry-point coverage differs")


def _array(
    *,
    task_manifest: str,
    concurrency: int,
    maximum_tasks: int,
    smoke_tasks: int = 1,
) -> dict[str, Any]:
    if (
        int(concurrency) <= 0
        or int(maximum_tasks) <= 0
        or int(smoke_tasks) <= 0
        or int(smoke_tasks) > int(maximum_tasks)
    ):
        raise ValueError("invalid bounded-array declaration")
    return {
        "task_manifest": task_manifest,
        "index_contract": "zero_based_contiguous_manifest_rows",
        "task_count_resolution": "authenticated_manifest_at_submission",
        "maximum_tasks": int(maximum_tasks),
        "smoke_tasks": int(smoke_tasks),
        "maximum_concurrent_tasks": int(concurrency),
        "empty_manifest_allowed": False,
    }


def _node(
    node_id: str,
    *,
    stage: str,
    worker: str,
    dependencies: Sequence[str] = (),
    resource: str = "cpu",
    array: Mapping[str, Any] | None = None,
    dynamic: bool = False,
    access: str = "none",
    resumable: bool = False,
    virtual_alias_of: str | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "stage": stage,
        "worker": worker,
        "dependencies": list(dependencies),
        "dependency_mode": "afterok",
        "resource": resource,
        "array": None if array is None else dict(array),
        "dynamic_continuation": bool(dynamic),
        "dataset_access": access,
        "resumable": bool(resumable),
        "virtual_alias_of": virtual_alias_of,
        "performance_warning_blocks_dependency": False,
        "scientific_underperformance_blocks_dependency": False,
        "provenance_or_execution_failure_blocks_dependency": True,
    }


def _nodes(concurrency: Mapping[str, int]) -> list[dict[str, Any]]:
    cpu = int(concurrency["cpu_cache"])
    expert = int(concurrency["gpu_expert"])
    predictor = int(concurrency["gpu_predictor"])
    scale = int(concurrency["gpu_scale"])
    final = int(concurrency["gpu_final"])
    return [
        _node("split_build", stage="A", worker="run_retb_build_splits.sh"),
        _node(
            "campaign_bootstrap",
            stage="A",
            worker="run_retb_build_campaign.sh",
            dependencies=("split_build",),
        ),
        _node(
            "compiled_region_backend",
            stage="A",
            worker="run_retb_compiled_region_backend.sh",
            dependencies=("campaign_bootstrap",),
        ),
        _node(
            "cpu_resource_probe",
            stage="A",
            worker="run_retb_resource_probe.sh",
            dependencies=("compiled_region_backend",),
        ),
        _node(
            "gpu_resource_probe",
            stage="A",
            worker="run_retb_resource_probe.sh",
            dependencies=("compiled_region_backend",),
            resource="gpu",
        ),
        _node(
            "offline_input_cache",
            stage="A",
            worker="run_retb_build_offline_inputs.sh",
            dependencies=("cpu_resource_probe",),
            array=_array(
                task_manifest="job_ledgers/tasks/offline_input_cache.json",
                concurrency=cpu,
                maximum_tasks=6,
                smoke_tasks=2,
            ),
            resumable=True,
            access="checkpoint_free_input_preparation",
        ),
        _node(
            "hlt_v3_cache",
            stage="A",
            worker="run_retb_build_hlt_v3.sh",
            dependencies=("cpu_resource_probe", "offline_input_cache"),
            array=_array(
                task_manifest="job_ledgers/tasks/hlt_v3_cache.json",
                concurrency=cpu,
                maximum_tasks=64,
                smoke_tasks=2,
            ),
            resumable=True,
            access="checkpoint_free_input_preparation",
        ),
        _node(
            "region_tree_cache",
            stage="A",
            worker="run_retb_build_region_trees.sh",
            dependencies=("hlt_v3_cache", "compiled_region_backend"),
            array=_array(
                task_manifest="job_ledgers/tasks/region_tree_cache.json",
                concurrency=cpu,
                maximum_tasks=512,
                smoke_tasks=9,
            ),
            resumable=True,
            access="checkpoint_free_input_preparation",
        ),
        _node(
            "region_tree_finalize",
            stage="A",
            worker="run_retb_finalize_region_trees.sh",
            dependencies=("region_tree_cache",),
            access="checkpoint_free_input_preparation",
        ),
        _node(
            "normalizers_500k",
            stage="A",
            worker="run_retb_fit_normalizers.sh",
            dependencies=("region_tree_finalize",),
            access="model_train_features_only",
        ),
        _node(
            "input_audit",
            stage="A",
            worker="run_retb_audit_inputs.sh",
            dependencies=(
                "hlt_v3_cache",
                "compiled_region_backend",
                "normalizers_500k",
            ),
        ),
        _node(
            "step3_architecture_contracts",
            stage="A",
            worker="run_retb_build_step3_contracts.sh",
            dependencies=("input_audit", "gpu_resource_probe"),
        ),
        _node(
            "step4_offline_training_contracts",
            stage="B",
            worker="run_retb_build_step4_contracts.sh",
            dependencies=("step3_architecture_contracts",),
        ),
        _node(
            "offline_expert_training",
            stage="B",
            worker="run_retb_train_offline_expert.sh",
            dependencies=("step4_offline_training_contracts",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_b_offline_experts.json",
                concurrency=expert,
                maximum_tasks=1024,
            ),
            resumable=True,
            access="model_train_and_val_stop",
        ),
        _node(
            "offline_optimization_selector",
            stage="B",
            worker="run_retb_select_optimization.sh",
            dependencies=("offline_expert_training",),
            dynamic=True,
            resource="gpu",
            access="val_design_only",
        ),
        _node(
            "step5_offline_fusion_contracts",
            stage="C",
            worker="run_retb_build_step5_contracts.sh",
            dependencies=("step4_offline_training_contracts",),
        ),
        _node(
            "offline_expert_confirmation",
            stage="C",
            worker="run_retb_train_offline_expert_confirmation.sh",
            dependencies=(
                "step5_offline_fusion_contracts",
                "offline_expert_training",
            ),
            resource="gpu",
            array=_array(
                task_manifest=(
                    "job_ledgers/tasks/"
                    "stage_c_offline_expert_confirmations.json"
                ),
                concurrency=expert,
                maximum_tasks=256,
                smoke_tasks=7,
            ),
            resumable=True,
            access="model_train_and_val_stop",
        ),
        _node(
            "offline_fusion_cache",
            stage="C",
            worker="run_retb_build_offline_fusion_cache.sh",
            dependencies=("offline_expert_confirmation",),
            resource="gpu",
            array=_array(
                task_manifest=(
                    "job_ledgers/tasks/stage_c_offline_fusion_cache.json"
                ),
                concurrency=expert,
                maximum_tasks=128,
                smoke_tasks=3,
            ),
            resumable=True,
            access="model_train_val_stop_and_val_design_inference",
        ),
        _node(
            "offline_fusion_training",
            stage="C",
            worker="run_retb_train_offline_fusion.sh",
            dependencies=(
                "step5_offline_fusion_contracts",
                "offline_fusion_cache",
            ),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_c_offline_fusions.json",
                concurrency=expert,
                maximum_tasks=256,
            ),
            resumable=True,
            access="model_train_and_val_stop",
        ),
        _node(
            "offline_shape_selector",
            stage="C",
            worker="run_retb_select_shapes.sh",
            dependencies=("offline_fusion_training",),
            dynamic=True,
            resource="gpu",
            access="val_design_only",
        ),
        _node(
            "offline_complementarity",
            stage="C",
            worker="run_retb_analyze_complementarity.sh",
            dependencies=("offline_fusion_training",),
            resource="gpu",
            access="val_design_only",
        ),
        _node(
            "offline_capacity_controls",
            stage="C",
            worker="run_retb_capacity_controls.sh",
            dependencies=("offline_shape_selector",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_c_capacity_controls.json",
                concurrency=expert,
                maximum_tasks=128,
            ),
            resumable=True,
        ),
        _node(
            "step6_native_hlt_contracts",
            stage="D",
            worker="run_retb_build_step6_contracts.sh",
            dependencies=(
                "offline_shape_selector",
                "offline_optimization_selector",
                "offline_complementarity",
                "offline_capacity_controls",
                "input_audit",
            ),
        ),
        _node(
            "native_hlt_expert_training",
            stage="D",
            worker="run_retb_train_native_hlt_expert.sh",
            dependencies=("step6_native_hlt_contracts",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_d_hlt_experts.json",
                concurrency=expert,
                maximum_tasks=1024,
            ),
            resumable=True,
            access="model_train_and_val_stop",
        ),
        _node(
            "native_hlt_fusion_training",
            stage="D",
            worker="run_retb_train_native_hlt_fusion.sh",
            dependencies=("native_hlt_expert_training",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_d_hlt_fusions.json",
                concurrency=expert,
                maximum_tasks=128,
            ),
            resumable=True,
        ),
        _node(
            "step7_bridge_contracts",
            stage="E",
            worker="run_retb_build_step7_contracts.sh",
            dependencies=(
                "offline_fusion_training",
                "native_hlt_expert_training",
                "step6_native_hlt_contracts",
            ),
            resource="gpu",
        ),
        _node(
            "bridge_pilot_training",
            stage="E",
            worker="run_retb_train_bridge_pilot.sh",
            dependencies=("step7_bridge_contracts",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_e_bridge_pilots.json",
                concurrency=predictor,
                maximum_tasks=128,
            ),
            resumable=True,
        ),
        _node(
            "bridge_target_training",
            stage="E",
            worker="run_retb_train_bridge_targets.sh",
            dependencies=("bridge_pilot_training",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_e_bridge_targets.json",
                concurrency=predictor,
                maximum_tasks=2048,
            ),
            dynamic=True,
            resumable=True,
        ),
        _node(
            "bridge_content_certification",
            stage="E",
            worker="run_retb_certify_bridge_content.sh",
            dependencies=("bridge_target_training",),
            resource="gpu",
            access="val_design_only",
        ),
        _node(
            "target_coordinate_selector",
            stage="E",
            worker="run_retb_select_bridge_coordinates.sh",
            dependencies=("bridge_content_certification",),
            access="val_design_only",
        ),
        _node(
            "step8_target_cache_contracts",
            stage="F",
            worker="run_retb_build_step8_contracts.sh",
            dependencies=("target_coordinate_selector",),
        ),
        _node(
            "target_cache_build",
            stage="F",
            worker="run_retb_build_target_cache.sh",
            dependencies=("step8_target_cache_contracts",),
            array=_array(
                task_manifest="job_ledgers/tasks/stage_f_target_caches.json",
                concurrency=cpu,
                maximum_tasks=512,
            ),
            dynamic=True,
            resumable=True,
            access="model_train_val_stop_val_design_only",
        ),
        _node(
            "target_normalizers",
            stage="F",
            worker="run_retb_fit_target_normalizers.sh",
            dependencies=("target_cache_build",),
            dynamic=True,
        ),
        _node(
            "step9_predictor_contracts",
            stage="G",
            worker="run_retb_build_step9_contracts.sh",
            dependencies=("target_normalizers", "native_hlt_expert_training"),
        ),
        _node(
            "predictor_training",
            stage="G",
            worker="run_retb_train_predictor.sh",
            dependencies=("step9_predictor_contracts",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_g_predictors.json",
                concurrency=predictor,
                maximum_tasks=4096,
            ),
            dynamic=True,
            resumable=True,
            access="model_train_and_val_stop",
        ),
        _node(
            "uncertainty_calibration",
            stage="G",
            worker="run_retb_calibrate_uncertainty.sh",
            dependencies=("predictor_training",),
            array=_array(
                task_manifest="job_ledgers/tasks/stage_g_calibrators.json",
                concurrency=cpu,
                maximum_tasks=1024,
            ),
            dynamic=True,
            access="label_free_val_design",
        ),
        _node(
            "step10_predictor_bundle_contracts",
            stage="H",
            worker="run_retb_build_step10_contracts.sh",
            dependencies=("uncertainty_calibration",),
        ),
        _node(
            "predictor_bundle_selector",
            stage="H",
            worker="run_retb_select_predictor_bundle.sh",
            dependencies=("step10_predictor_bundle_contracts",),
            dynamic=True,
            access="val_design_only",
        ),
        _node(
            "oracle_substitutions",
            stage="H",
            worker="run_retb_stage_i_substitutions.sh",
            dependencies=("predictor_bundle_selector",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_h_oracle_substitutions.json",
                concurrency=predictor,
                maximum_tasks=256,
            ),
            dynamic=True,
            access="val_design_only",
        ),
        _node(
            "step11_joint_bridge_contracts",
            stage="I",
            worker="run_retb_build_step11_contracts.sh",
            dependencies=("predictor_bundle_selector",),
        ),
        _node(
            "joint_predictor_training",
            stage="I",
            worker="run_retb_train_joint_bridge.sh",
            dependencies=("step11_joint_bridge_contracts",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_i_joint_predictors.json",
                concurrency=predictor,
                maximum_tasks=512,
            ),
            dynamic=True,
            resumable=True,
        ),
        _node(
            "joint_predictor_selector",
            stage="I",
            worker="run_retb_select_joint_bundle.sh",
            dependencies=("joint_predictor_training", "oracle_substitutions"),
            dynamic=True,
            access="val_design_only",
        ),
        _node(
            "step12_final_consumer_contracts",
            stage="J",
            worker="run_retb_build_step12_contracts.sh",
            dependencies=("joint_predictor_selector",),
        ),
        _node(
            "final_consumer_training",
            stage="J",
            worker="run_retb_train_final_consumer.sh",
            dependencies=("step12_final_consumer_contracts",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_j_final_consumers.json",
                concurrency=predictor,
                maximum_tasks=1024,
            ),
            dynamic=True,
            resumable=True,
        ),
        _node(
            "deployable_export",
            stage="J",
            worker="run_retb_export_deployable_graph.sh",
            dependencies=("final_consumer_training",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_j_exports.json",
                concurrency=predictor,
                maximum_tasks=512,
            ),
            dynamic=True,
        ),
        _node(
            "robustness_controls",
            stage="K",
            worker="run_retb_robustness_controls.sh",
            dependencies=("deployable_export",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_k_robustness.json",
                concurrency=predictor,
                maximum_tasks=2048,
            ),
            dynamic=True,
            access="val_design_only",
        ),
        _node(
            "semantic_controls",
            stage="K",
            worker="run_retb_semantic_controls.sh",
            dependencies=("deployable_export",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_k_semantics.json",
                concurrency=predictor,
                maximum_tasks=1024,
            ),
            dynamic=True,
            access="val_design_only",
        ),
        _node(
            "step13_confirmation_contracts",
            stage="L",
            worker="run_retb_build_step13_contracts.sh",
            dependencies=("robustness_controls", "semantic_controls"),
        ),
        _node(
            "stage_l_graph_registration",
            stage="L",
            worker="run_retb_register_stage_l_graphs.sh",
            dependencies=("step13_confirmation_contracts",),
            dynamic=True,
        ),
        _node(
            "confirmation_500k",
            stage="L",
            worker="run_retb_register_500k_seed_confirmation.sh",
            dependencies=("stage_l_graph_registration",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_l_confirmations.json",
                concurrency=predictor,
                maximum_tasks=2048,
            ),
            dynamic=True,
            resumable=True,
            access="val_design_only",
        ),
        _node(
            "confirmation_summary",
            stage="L",
            worker="run_retb_confirm.sh",
            dependencies=("confirmation_500k",),
            dynamic=True,
            access="val_design_only",
        ),
        _node(
            "bridge_shape_selector",
            stage="L",
            worker="run_retb_select_bridge_shape.sh",
            dependencies=("confirmation_summary",),
            dynamic=True,
            access="val_design_only",
        ),
        _node(
            "scale_shortlist_selector",
            stage="L",
            worker="run_retb_scale_shortlist.sh",
            dependencies=("confirmation_summary", "bridge_shape_selector"),
            dynamic=True,
            access="val_design_only",
        ),
        _node(
            "shortlisted_500k_control_training",
            stage="L",
            worker="run_retb_train_shortlisted_500k_controls.sh",
            dependencies=("scale_shortlist_selector",),
            resource="gpu",
            array=_array(
                task_manifest=(
                    "job_ledgers/tasks/shortlisted_500k_control_training.json"
                ),
                concurrency=scale,
                maximum_tasks=54,
                smoke_tasks=3,
            ),
            dynamic=True,
            resumable=True,
            access="model_train_val_stop_and_val_design",
        ),
        _node(
            "shortlisted_500k_controls",
            stage="L",
            worker="run_retb_aggregate_shortlisted_500k_controls.sh",
            dependencies=("shortlisted_500k_control_training",),
            dynamic=True,
            resumable=True,
            access="val_design_only",
        ),
        _node(
            "step14_scale_final_contracts",
            stage="M",
            worker="run_retb_build_step14_contracts.sh",
            dependencies=("shortlisted_500k_controls",),
        ),
        *[
            _node(
                node_id,
                stage="M",
                worker="run_retb_production_task.sh",
                dependencies=(dependency,),
                resource="gpu",
                array=_array(
                    task_manifest=f"job_ledgers/tasks/{node_id}.json",
                    concurrency=scale,
                    # Five possible carried roles x seven experts x two
                    # source/allocated shapes x three seeds is the exact
                    # worst-case offline-expert component bound.  Other
                    # refit phases are strict subsets of this ceiling.
                    maximum_tasks=210,
                    smoke_tasks=1,
                ),
                dynamic=True,
                resumable=True,
                access="scale_train_and_label_free_val_design",
            )
            for node_id, dependency in (
                ("scale_refit_normalizers", "step14_scale_final_contracts"),
                ("scale_refit_teachers", "scale_refit_normalizers"),
                ("scale_refit_offline_experts", "scale_refit_teachers"),
                ("scale_refit_targets", "scale_refit_offline_experts"),
                ("scale_refit_native", "scale_refit_targets"),
                ("scale_refit_native_fusion", "scale_refit_native"),
                ("scale_refit_predictors", "scale_refit_native_fusion"),
                ("scale_refit_calibrations", "scale_refit_predictors"),
                ("scale_refits", "scale_refit_calibrations"),
            )
        ],
        *[
            _node(
                node_id,
                stage="M",
                worker="run_retb_production_task.sh",
                dependencies=(dependency,),
                resource="gpu",
                array=_array(
                    task_manifest=f"job_ledgers/tasks/{node_id}.json",
                    concurrency=scale,
                    maximum_tasks=21,
                    smoke_tasks=1,
                ),
                dynamic=True,
                resumable=True,
                access="scale_train_and_val_stop",
            )
            for node_id, dependency in (
                ("scale_joint_training", "scale_refits"),
                ("scale_graph_datasets", "scale_joint_training"),
                ("scale_refiner_training", "scale_graph_datasets"),
                ("scale_final_consumer_training", "scale_refiner_training"),
                ("scale_graph_export", "scale_final_consumer_training"),
                ("scale_graph_training", "scale_graph_export"),
            )
        ],
        _node(
            "prelock_final_inputs",
            stage="N",
            worker="run_retb_prepare_final_inputs.sh",
            dependencies=("input_audit",),
            dynamic=True,
            access="checkpoint_free_final_input_preparation",
            resumable=True,
        ),
        _node(
            "scale_completion",
            stage="M",
            worker="run_retb_scale_completion.sh",
            dependencies=("scale_graph_training", "prelock_final_inputs"),
            dynamic=True,
        ),
        _node(
            "stack_val_inference",
            stage="N",
            worker="run_retb_infer_scale_stack_val.sh",
            dependencies=("scale_completion",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_n_stack_val.json",
                concurrency=scale,
                maximum_tasks=18,
            ),
            dynamic=True,
            resumable=True,
            access="label_free_stack_val_features",
        ),
        _node(
            "accuracy_finalist_selector",
            stage="N",
            worker="run_retb_select_scale_finalists.sh",
            dependencies=("stack_val_inference",),
            dynamic=True,
            access="selector_only_stack_val_labels",
        ),
        _node(
            "rejection_finalist_selector",
            stage="N",
            worker="run_retb_select_scale_finalists.sh",
            dependencies=("stack_val_inference",),
            access="selector_only_stack_val_labels",
            virtual_alias_of="accuracy_finalist_selector",
        ),
        _node(
            "locked_scale_finalists",
            stage="N",
            worker="run_retb_select_scale_finalists.sh",
            dependencies=(
                "accuracy_finalist_selector",
                "rejection_finalist_selector",
            ),
            virtual_alias_of="accuracy_finalist_selector",
        ),
        _node(
            "postlock_oracle_targets",
            stage="N",
            worker="run_retb_postlock_targets.sh",
            dependencies=("locked_scale_finalists",),
            array=_array(
                task_manifest="job_ledgers/tasks/stage_n_postlock_targets.json",
                concurrency=cpu,
                maximum_tasks=12,
            ),
            dynamic=True,
            resumable=True,
            access="postlock_stack_val_and_final_test_targets",
        ),
        _node(
            "finalist_controls",
            stage="N",
            worker="run_retb_finalist_controls.sh",
            dependencies=("locked_scale_finalists",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_n_finalist_controls.json",
                concurrency=final,
                maximum_tasks=18,
            ),
            dynamic=True,
            resumable=True,
        ),
        _node(
            "stage_n_evidence_join",
            stage="N",
            worker="run_retb_stage_n_evidence_join.sh",
            dependencies=(
                "postlock_oracle_targets",
                "finalist_controls",
                "prelock_final_inputs",
            ),
        ),
        _node(
            "final_test_execution_lock",
            stage="N",
            worker="run_retb_final_execution_lock.sh",
            dependencies=("stage_n_evidence_join",),
            dynamic=True,
        ),
        _node(
            "sealed_final_test",
            stage="N",
            worker="run_retb_final_test.sh",
            dependencies=("final_test_execution_lock",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_n_final_test.json",
                concurrency=final,
                maximum_tasks=1,
            ),
            dynamic=True,
            resumable=True,
            access="sealed_final_test_once",
        ),
        _node(
            "final_report",
            stage="N",
            worker="run_retb_step14_report.sh",
            dependencies=("sealed_final_test",),
            dynamic=True,
            access="authenticated_final_predictions_only",
        ),
        _node(
            "completed_job_ledger",
            stage="N",
            worker="run_retb_finalize_job_ledger.sh",
            dependencies=("final_report",),
        ),
    ]


def _task_manifest_path(node: Mapping[str, Any]) -> str:
    array = node["array"]
    if array is not None:
        return str(array["task_manifest"])
    return f"job_ledgers/tasks/{node['node_id']}.json"


def task_manifest_path_for_graph(
    production_graph: Mapping[str, Any],
    *,
    node_id: str,
    campaign_root: str | Path | None = None,
) -> Path:
    """Resolve the graph-authoritative task-manifest path for one node."""

    validate_production_graph(production_graph)
    matches = [
        node
        for node in production_graph["nodes"]
        if node["node_id"] == str(node_id)
    ]
    if len(matches) != 1:
        raise ValueError("task-manifest node is absent or duplicated")
    relative = Path(_task_manifest_path(matches[0]))
    root = Path(
        production_graph["campaign_root"]
        if campaign_root is None
        else campaign_root
    ).resolve()
    if root != Path(production_graph["campaign_root"]).resolve():
        raise ValueError("task-manifest campaign root differs")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("task-manifest path escapes campaign root") from error
    return resolved


def build_node_execution_registry(
    *,
    nodes: Sequence[Mapping[str, Any]],
    manifest_producer_nodes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the exhaustive production-node execution/manifest registry."""

    by_id = {str(node["node_id"]): node for node in nodes}
    if len(by_id) != len(nodes) or not by_id:
        raise ValueError("node execution registry input identities differ")
    producers = dict(
        TASK_MANIFEST_PRODUCER_NODES
        if manifest_producer_nodes is None
        else manifest_producer_nodes
    )
    alias_nodes = {
        node_id
        for node_id, node in by_id.items()
        if node["virtual_alias_of"] is not None
    }
    direct_nodes = set(DIRECT_WORKER_NODES) & set(by_id)
    unknown_direct = set(DIRECT_WORKER_NODES) - set(by_id)
    manifest_nodes = set(by_id) - direct_nodes - alias_nodes
    if unknown_direct:
        raise ValueError(
            "direct-worker registry contains unknown nodes "
            f"{sorted(unknown_direct)}"
        )
    if set(producers) != manifest_nodes:
        missing = sorted(manifest_nodes - set(producers))
        extra = sorted(set(producers) - manifest_nodes)
        raise ValueError(
            "task-manifest producer coverage differs: "
            f"missing={missing}, extra={extra}"
        )

    entries: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node["node_id"])
        dependencies = [str(value) for value in node["dependencies"]]
        alias = node["virtual_alias_of"]
        if alias is not None:
            mode = "virtual_alias"
            row_resolution = "not_applicable"
            manifest_path = None
            producer = None
            expected_outputs = [f"node_output:{alias}"]
        elif node_id in DIRECT_WORKER_NODES:
            mode = "direct_worker"
            row_resolution = "not_applicable"
            manifest_path = None
            producer = None
            expected_outputs = [f"node_completion:{node_id}"]
        else:
            mode = "task_manifest_worker"
            row_resolution = (
                "dynamic"
                if bool(node["dynamic_continuation"])
                else "static"
            )
            manifest_path = _task_manifest_path(node)
            producer_node = str(producers[node_id])
            if node_id in BOOTSTRAP_INPUT_MANIFEST_NODES:
                producer_entrypoint = "scripts/bootstrap_retb_input_tasks.py"
            elif node_id in STATIC_EXPERIMENT_MANIFEST_NODES:
                producer_entrypoint = (
                    "scripts/compile_retb_static_experiment_manifests.py"
                )
            elif node_id in MIDDLE_CONTINUATION_MANIFEST_NODES:
                producer_entrypoint = (
                    "scripts/continue_retb_stage_f_j.py"
                )
            elif node_id in LATE_CONTINUATION_MANIFEST_NODES:
                producer_entrypoint = (
                    "scripts/continue_retb_stage_k_m.py"
                )
            elif node_id in FINAL_CONTINUATION_MANIFEST_NODES:
                producer_entrypoint = "scripts/continue_retb_stage_n.py"
            else:
                producer_entrypoint = "scripts/build_retb_task_manifest.py"
            producer = {
                "node_id": producer_node,
                "entrypoint": producer_entrypoint,
                "publication_mode": (
                    "campaign_bootstrap"
                    if producer_node == "campaign_bootstrap"
                    else "upstream_dependency"
                ),
            }
            expected_outputs = [f"node_completion:{node_id}"]
        required_inputs = [
            "campaign_spec",
            "production_graph",
            *(f"node_output:{dependency}" for dependency in dependencies),
        ]
        if manifest_path is not None:
            required_inputs.append(f"task_manifest:{manifest_path}")
        entries.append(
            {
                "node_id": node_id,
                "worker": str(node["worker"]),
                "dispatch_mode": mode,
                "manifest_required": mode == "task_manifest_worker",
                "task_manifest_path": manifest_path,
                "manifest_producer": producer,
                "required_inputs": required_inputs,
                "expected_outputs": expected_outputs,
                "resource": str(node["resource"]),
                "row_resolution": row_resolution,
            }
        )
    artifact = with_content_hash(
        {
            "contract": NODE_EXECUTION_REGISTRY_CONTRACT,
            "schema_version": 12,
            "entries": entries,
            "node_count": len(entries),
            "manifest_driven_node_count": len(manifest_nodes),
            "direct_worker_node_count": len(direct_nodes),
            "virtual_alias_node_count": len(alias_nodes),
            "missing_manifest_producers": [],
        }
    )
    validate_node_execution_registry(artifact, nodes=nodes)
    return artifact


def validate_node_execution_registry(
    payload: Mapping[str, Any],
    *,
    nodes: Sequence[Mapping[str, Any]],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=NODE_EXECUTION_REGISTRY_CONTRACT
    )
    if int(payload.get("schema_version", -1)) != 12:
        raise ValueError("node execution registry schema version differs")
    by_id = {str(node["node_id"]): node for node in nodes}
    entries = list(payload.get("entries", ()))
    by_entry = {str(entry["node_id"]): entry for entry in entries}
    if (
        len(by_id) != len(nodes)
        or len(by_entry) != len(entries)
        or set(by_entry) != set(by_id)
    ):
        missing = sorted(set(by_id) - set(by_entry))
        extra = sorted(set(by_entry) - set(by_id))
        raise ValueError(
            "node execution registry coverage differs: "
            f"missing={missing}, extra={extra}"
        )
    if [entry["node_id"] for entry in entries] != [
        node["node_id"] for node in nodes
    ]:
        raise ValueError("node execution registry order differs")
    manifest_count = 0
    direct_count = 0
    alias_count = 0
    for node_id, node in by_id.items():
        entry = by_entry[node_id]
        alias = node["virtual_alias_of"]
        if alias is not None:
            expected_mode = "virtual_alias"
            alias_count += 1
        elif node_id in DIRECT_WORKER_NODES:
            expected_mode = "direct_worker"
            direct_count += 1
        else:
            expected_mode = "task_manifest_worker"
            manifest_count += 1
        if (
            entry["worker"] != node["worker"]
            or entry["resource"] != node["resource"]
            or entry["dispatch_mode"] != expected_mode
            or bool(entry["manifest_required"])
            != (expected_mode == "task_manifest_worker")
        ):
            raise ValueError(f"{node_id} execution registry semantics differ")
        expected_required = [
            "campaign_spec",
            "production_graph",
            *(
                f"node_output:{dependency}"
                for dependency in node["dependencies"]
            ),
        ]
        if expected_mode == "task_manifest_worker":
            expected_path = _task_manifest_path(node)
            expected_required.append(f"task_manifest:{expected_path}")
            producer = entry["manifest_producer"]
            if (
                entry["task_manifest_path"] != expected_path
                or not isinstance(producer, Mapping)
                or not str(producer.get("node_id", ""))
                or producer.get("entrypoint")
                not in {
                    "scripts/bootstrap_retb_input_tasks.py",
                    "scripts/compile_retb_static_experiment_manifests.py",
                    "scripts/continue_retb_stage_f_j.py",
                    "scripts/continue_retb_stage_k_m.py",
                    "scripts/continue_retb_stage_n.py",
                    "scripts/build_retb_task_manifest.py",
                }
                or producer.get("publication_mode")
                not in {"campaign_bootstrap", "upstream_dependency"}
            ):
                raise ValueError(
                    f"{node_id} lacks an automatic manifest producer"
                )
            producer_node = str(producer["node_id"])
            if (
                producer_node not in by_id
                or producer_node == node_id
                or producer_node not in _ancestors(by_id, node_id)
            ):
                raise ValueError(
                    f"{node_id} manifest producer is not an ancestor"
                )
            expected_producer = TASK_MANIFEST_PRODUCER_NODES.get(node_id)
            if node_id in BOOTSTRAP_INPUT_MANIFEST_NODES:
                expected_entrypoint = "scripts/bootstrap_retb_input_tasks.py"
            elif node_id in STATIC_EXPERIMENT_MANIFEST_NODES:
                expected_entrypoint = (
                    "scripts/compile_retb_static_experiment_manifests.py"
                )
            elif node_id in MIDDLE_CONTINUATION_MANIFEST_NODES:
                expected_entrypoint = "scripts/continue_retb_stage_f_j.py"
            elif node_id in LATE_CONTINUATION_MANIFEST_NODES:
                expected_entrypoint = "scripts/continue_retb_stage_k_m.py"
            elif node_id in FINAL_CONTINUATION_MANIFEST_NODES:
                expected_entrypoint = "scripts/continue_retb_stage_n.py"
            else:
                expected_entrypoint = "scripts/build_retb_task_manifest.py"
            if (
                producer_node != expected_producer
                or producer["entrypoint"] != expected_entrypoint
            ):
                raise ValueError(
                    f"{node_id} automatic manifest producer differs"
                )
            expected_resolution = (
                "dynamic"
                if bool(node["dynamic_continuation"])
                else "static"
            )
        else:
            if (
                entry["task_manifest_path"] is not None
                or entry["manifest_producer"] is not None
            ):
                raise ValueError(
                    f"{node_id} unexpectedly declares a task manifest"
                )
            expected_resolution = "not_applicable"
        expected_outputs = [
            (
                f"node_output:{alias}"
                if alias is not None
                else f"node_completion:{node_id}"
            )
        ]
        if (
            entry["required_inputs"] != expected_required
            or entry["expected_outputs"] != expected_outputs
            or entry["row_resolution"] != expected_resolution
        ):
            raise ValueError(f"{node_id} execution I/O registry differs")
    if (
        int(payload.get("node_count", -1)) != len(nodes)
        or int(payload.get("manifest_driven_node_count", -1))
        != manifest_count
        or int(payload.get("direct_worker_node_count", -1)) != direct_count
        or int(payload.get("virtual_alias_node_count", -1)) != alias_count
        or payload.get("missing_manifest_producers") != []
    ):
        raise ValueError("node execution registry coverage summary differs")
    return digest


def build_production_graph(
    *,
    campaign_root: str | Path,
    campaign_id: str,
    source_commit: str,
    source_status_sha256: str,
    storage_measurements_sha256: str,
    miniature: bool = False,
    concurrency: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    resolved = dict(DEFAULT_CONCURRENCY)
    if concurrency is not None:
        if set(concurrency) != set(DEFAULT_CONCURRENCY):
            raise ValueError("production concurrency keys differ")
        resolved = {key: int(value) for key, value in concurrency.items()}
    if any(value <= 0 for value in resolved.values()):
        raise ValueError("production concurrency must be positive")
    nodes = _nodes(resolved)
    execution_registry = build_node_execution_registry(nodes=nodes)
    bootstrap_manifest_targets = (
        set(BOOTSTRAP_INPUT_MANIFEST_NODES)
        | set(STATIC_EXPERIMENT_MANIFEST_NODES)
    )
    plan_factory_registry = build_manifest_plan_factory_registry(
        nodes=nodes,
        manifest_producer_nodes=TASK_MANIFEST_PRODUCER_NODES,
        bootstrap_targets=bootstrap_manifest_targets,
    )
    profile = (
        "nonproduction_miniature_test"
        if miniature
        else "production_500k_100k_50k_300k_scale3m"
    )
    artifact = with_content_hash(
        {
            "contract": PRODUCTION_GRAPH_CONTRACT,
            "schema_version": 27,
            "campaign_id": str(campaign_id),
            "campaign_root": str(Path(campaign_root)),
            "campaign_profile": profile,
            "scientific_results_allowed": not miniature,
            "source_commit": require_git_object_id(
                source_commit, name="source_commit"
            ),
            "source_status_sha256": require_sha256(
                source_status_sha256, name="source_status_sha256"
            ),
            "storage_measurements_sha256": require_sha256(
                storage_measurements_sha256,
                name="storage_measurements_sha256",
            ),
            "degradation_profile": "D_NOMINAL",
            "degradation_profile_implicit_override_allowed": False,
            "split_sizes": dict(
                MINIATURE_SPLIT_SIZES if miniature else PRODUCTION_SPLIT_SIZES
            ),
            "tigris_defaults": dict(TIGRIS_DEFAULTS),
            "bounded_concurrency": resolved,
            "nodes": nodes,
            "node_execution_registry": execution_registry,
            "manifest_plan_factory_registry": plan_factory_registry,
            "stage_order": list("ABCDEFGHIJKLMN"),
            "two_stage_n_selectors": [
                "accuracy_finalist_selector",
                "rejection_finalist_selector",
            ],
            "performance_based_termination": False,
            "negative_campaign_continues_to_final_report": True,
            "final_test_before_both_locks_allowed": False,
            "preproduction_validation_order": [
                "local_synthetic_DAG",
                "local_miniature_worker_interfaces",
                "smoke_simulate",
                "execution_complete_manifest_plan_audit",
                "real_miniature_Tigris_smoke",
                "production_dry_run_authenticated_storage",
            ],
            "full_submission_requires_operational_authorization": True,
            "production_submission_performed": False,
            "monitoring": {
                "queue": 'squeue -u "$USER" -o "%i %j %T %R"',
                "accounting": (
                    "sacct -X --starttime today --name retb_ "
                    "--format=JobID,JobName,State,Elapsed,ExitCode"
                ),
                "resume": (
                    "python scripts/plan_retb_resume.py "
                    "--campaign-root \"${CAMPAIGN_ROOT}\" "
                    "--production-graph \"${CAMPAIGN_ROOT}/job_ledgers/production_graph.json\" "
                    "--previous-ledger \"${CAMPAIGN_ROOT}/job_ledgers/initial_submission_ledger.json\" "
                    "--dry-run"
                ),
                "cancel_stale": (
                    "python scripts/monitor_retb_campaign.py "
                    "--campaign-root \"${CAMPAIGN_ROOT}\" --cancel-stale"
                ),
            },
        }
    )
    validate_production_graph(artifact)
    return artifact


def _ancestors(
    nodes: Mapping[str, Mapping[str, Any]], node_id: str
) -> set[str]:
    result: set[str] = set()
    pending = list(nodes[node_id]["dependencies"])
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        pending.extend(nodes[current]["dependencies"])
    return result


def validate_production_graph(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    if int(payload.get("schema_version", -1)) != 27:
        raise ValueError("production graph schema version differs")
    nodes = list(payload.get("nodes", ()))
    by_id = {str(node["node_id"]): node for node in nodes}
    if len(nodes) != len(by_id) or not nodes:
        raise ValueError("production graph node identities differ")
    seen: set[str] = set()
    for node in nodes:
        node_id = str(node["node_id"])
        if node["stage"] not in "ABCDEFGHIJKLMN":
            raise ValueError("production graph stage differs")
        if node["dependency_mode"] != "afterok":
            raise ValueError("scientific dependencies must use afterok")
        if any(parent not in by_id for parent in node["dependencies"]):
            raise ValueError(f"{node_id} has an unknown dependency")
        if any(parent not in seen for parent in node["dependencies"]):
            raise ValueError("production graph is not topologically ordered")
        if (
            node["performance_warning_blocks_dependency"] is not False
            or node["scientific_underperformance_blocks_dependency"] is not False
            or node["provenance_or_execution_failure_blocks_dependency"] is not True
        ):
            raise ValueError("production failure semantics differ")
        array = node["array"]
        if array is not None:
            _array(
                task_manifest=array["task_manifest"],
                concurrency=int(array["maximum_concurrent_tasks"]),
                maximum_tasks=int(array["maximum_tasks"]),
                smoke_tasks=int(array["smoke_tasks"]),
            )
        alias = node["virtual_alias_of"]
        if alias is not None and alias not in by_id:
            raise ValueError("virtual job alias differs")
        seen.add(node_id)
    if set(node["stage"] for node in nodes) != set("ABCDEFGHIJKLMN"):
        raise ValueError("production graph does not cover every Stage A-N")
    required_ancestors = {
        "offline_expert_training": {"input_audit", "normalizers_500k"},
        "predictor_training": {
            "target_cache_build",
            "native_hlt_expert_training",
        },
        "scale_graph_training": {
            "scale_shortlist_selector",
            "confirmation_summary",
        },
        "stack_val_inference": {"scale_completion", "scale_graph_training"},
        "locked_scale_finalists": {
            "accuracy_finalist_selector",
            "rejection_finalist_selector",
            "stack_val_inference",
        },
        "postlock_oracle_targets": {"locked_scale_finalists"},
        "final_test_execution_lock": {
            "locked_scale_finalists",
            "postlock_oracle_targets",
            "finalist_controls",
            "prelock_final_inputs",
        },
        "sealed_final_test": {
            "locked_scale_finalists",
            "final_test_execution_lock",
        },
        "final_report": {"sealed_final_test"},
    }
    for node_id, required in required_ancestors.items():
        missing = required - _ancestors(by_id, node_id)
        if missing:
            raise ValueError(f"{node_id} lacks ancestors {sorted(missing)}")
    validate_node_execution_registry(
        payload["node_execution_registry"], nodes=nodes
    )
    validate_manifest_plan_factory_registry(
        payload["manifest_plan_factory_registry"],
        nodes=nodes,
        manifest_producer_nodes=TASK_MANIFEST_PRODUCER_NODES,
        bootstrap_targets=(
            set(BOOTSTRAP_INPUT_MANIFEST_NODES)
            | set(STATIC_EXPERIMENT_MANIFEST_NODES)
        ),
    )
    if (
        payload["degradation_profile"] != "D_NOMINAL"
        or payload["degradation_profile_implicit_override_allowed"] is not False
        or payload["performance_based_termination"] is not False
        or payload["negative_campaign_continues_to_final_report"] is not True
        or payload["final_test_before_both_locks_allowed"] is not False
        or payload["preproduction_validation_order"]
        != [
            "local_synthetic_DAG",
            "local_miniature_worker_interfaces",
            "smoke_simulate",
            "execution_complete_manifest_plan_audit",
            "real_miniature_Tigris_smoke",
            "production_dry_run_authenticated_storage",
        ]
        or payload["full_submission_requires_operational_authorization"]
        is not True
        or payload["two_stage_n_selectors"]
        != ["accuracy_finalist_selector", "rejection_finalist_selector"]
    ):
        raise ValueError("production scientific controls differ")
    for key, expected in TIGRIS_DEFAULTS.items():
        if payload["tigris_defaults"].get(key) != expected:
            raise ValueError(f"Tigris default {key} differs")
    return digest


def validate_production_campaign_binding(
    production_graph: Mapping[str, Any],
    campaign_spec: Mapping[str, Any],
) -> str:
    graph_sha = validate_production_graph(production_graph)
    profile_map = {
        "miniature_test": "nonproduction_miniature_test",
        "production_500k_scale3m": (
            "production_500k_100k_50k_300k_scale3m"
        ),
    }
    if campaign_spec["campaign_profile"] not in profile_map:
        raise ValueError("campaign profile is not a production-DAG profile")
    expected_profile = profile_map[campaign_spec["campaign_profile"]]
    if (
        production_graph["campaign_id"] != campaign_spec["campaign_id"]
        or Path(production_graph["campaign_root"]).name
        != campaign_spec["campaign_id"]
        or production_graph["source_commit"]
        != campaign_spec["source"]["commit"]
        or production_graph["source_status_sha256"]
        != campaign_spec["source"]["status_sha256"]
        or production_graph["storage_measurements_sha256"]
        != campaign_spec["parent_artifact_hashes"]["storage_measurements"]
        or production_graph["campaign_profile"] != expected_profile
    ):
        raise ValueError("production graph differs from campaign binding")
    return graph_sha


def build_job_ledger(
    *,
    production_graph: Mapping[str, Any],
    jobs: Mapping[str, str | None],
    submission_mode: str,
    resolved_arrays: Mapping[str, Mapping[str, Any]] | None = None,
    completion_artifact_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    graph_sha = validate_production_graph(production_graph)
    if submission_mode not in {
        "dry_run",
        "smoke_simulation",
        "smoke_submitted",
        "production_submitted",
        "resumed",
        "completed",
    }:
        raise ValueError("job-ledger submission mode differs")
    node_ids = {str(node["node_id"]) for node in production_graph["nodes"]}
    if set(jobs) - node_ids:
        raise ValueError("job ledger contains unknown nodes")
    normalized: dict[str, str | None] = {}
    for node_id in sorted(node_ids):
        value = jobs.get(node_id)
        if value is not None and (
            not str(value).isdigit() or int(str(value)) <= 0
        ):
            raise ValueError(f"invalid Slurm job ID for {node_id}")
        normalized[node_id] = None if value is None else str(value)
    arrays = {}
    for node_id, row in sorted((resolved_arrays or {}).items()):
        if node_id not in node_ids:
            raise ValueError("resolved array belongs to an unknown node")
        count = int(row["task_count"])
        concurrency = int(row["maximum_concurrent_tasks"])
        if count <= 0 or concurrency <= 0:
            raise ValueError("resolved array dimensions differ")
        arrays[node_id] = {
            "task_manifest_sha256": require_sha256(
                row["task_manifest_sha256"],
                name=f"{node_id}.task_manifest_sha256",
            ),
            "task_count": count,
            "maximum_concurrent_tasks": concurrency,
            "slurm_array": f"0-{count - 1}%{concurrency}",
        }
    completion_keys = frozenset(
        {
            "locked_scale_finalists",
            "final_test_execution_lock",
            "sealed_final_test_evaluation",
            "final_report",
        }
    )
    completion = dict(completion_artifact_hashes or {})
    all_nodes_bound = all(
        value is not None for value in normalized.values()
    )
    if submission_mode == "completed":
        if not all_nodes_bound or set(completion) != set(completion_keys):
            raise ValueError(
                "completed ledger lacks all jobs or final artifacts"
            )
    elif completion:
        raise ValueError(
            "non-completed ledger may not bind final artifacts"
        )
    return with_content_hash(
        {
            "contract": JOB_LEDGER_CONTRACT,
            "schema_version": 2,
            "production_graph_sha256": graph_sha,
            "campaign_id": production_graph["campaign_id"],
            "campaign_root": production_graph["campaign_root"],
            "submission_mode": submission_mode,
            "jobs": normalized,
            "resolved_arrays": arrays,
            "submitted_node_count": sum(
                value is not None for value in normalized.values()
            ),
            "all_nodes_bound": all_nodes_bound,
            "completion_artifact_hashes": {
                name: require_sha256(
                    value, name=f"completion_artifact_hashes.{name}"
                )
                for name, value in sorted(completion.items())
            },
            "completed_after_final_report": submission_mode == "completed",
            "performance_based_cancellation_allowed": False,
            "stale_job_cancellation_requires_lineage_mismatch": True,
        }
    )


def validate_job_ledger(
    payload: Mapping[str, Any], *, production_graph: Mapping[str, Any]
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=JOB_LEDGER_CONTRACT
    )
    expected = build_job_ledger(
        production_graph=production_graph,
        jobs=payload["jobs"],
        submission_mode=payload["submission_mode"],
        resolved_arrays=payload["resolved_arrays"],
        completion_artifact_hashes=payload[
            "completion_artifact_hashes"
        ],
    )
    if payload != expected:
        raise ValueError("job-ledger semantics differ")
    return digest


def build_resource_probe(
    *,
    campaign_spec_sha256: str,
    storage_measurements_sha256: str,
    resource_kind: str,
    node_name: str,
    available_memory_bytes: int,
    available_storage_bytes: int,
    measured_items_per_second: float,
    compiler_backend_parity_passed: bool,
    requested_memory_bytes: int,
    projected_peak_storage_bytes: int,
) -> dict[str, Any]:
    if resource_kind not in {"cpu", "gpu"}:
        raise ValueError("resource probe kind differs")
    numeric = (
        int(available_memory_bytes),
        int(available_storage_bytes),
        int(requested_memory_bytes),
        int(projected_peak_storage_bytes),
    )
    rate = float(measured_items_per_second)
    if any(value < 0 for value in numeric) or not math.isfinite(rate) or rate < 0:
        raise ValueError("resource probe measurement differs")
    admitted = (
        int(available_memory_bytes) >= int(requested_memory_bytes)
        and int(available_storage_bytes) >= int(projected_peak_storage_bytes)
        and bool(compiler_backend_parity_passed)
    )
    return with_content_hash(
        {
            "contract": RESOURCE_PROBE_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "storage_measurements_sha256": require_sha256(
                storage_measurements_sha256,
                name="storage_measurements_sha256",
            ),
            "resource_kind": resource_kind,
            "node_name": str(node_name),
            "available_memory_bytes": int(available_memory_bytes),
            "available_storage_bytes": int(available_storage_bytes),
            "requested_memory_bytes": int(requested_memory_bytes),
            "projected_peak_storage_bytes": int(projected_peak_storage_bytes),
            "measured_items_per_second": rate,
            "compiled_region_backend_parity_passed": bool(
                compiler_backend_parity_passed
            ),
            "resource_admitted": admitted,
            "throughput_changes_scientific_selection": False,
            "underperformance_is_not_a_resource_failure": True,
        }
    )


def validate_resource_probe(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=RESOURCE_PROBE_CONTRACT
    )
    expected = build_resource_probe(
        campaign_spec_sha256=payload["campaign_spec_sha256"],
        storage_measurements_sha256=payload["storage_measurements_sha256"],
        resource_kind=payload["resource_kind"],
        node_name=payload["node_name"],
        available_memory_bytes=int(payload["available_memory_bytes"]),
        available_storage_bytes=int(payload["available_storage_bytes"]),
        measured_items_per_second=float(payload["measured_items_per_second"]),
        compiler_backend_parity_passed=bool(
            payload["compiled_region_backend_parity_passed"]
        ),
        requested_memory_bytes=int(payload["requested_memory_bytes"]),
        projected_peak_storage_bytes=int(
            payload["projected_peak_storage_bytes"]
        ),
    )
    if payload != expected:
        raise ValueError("resource-probe semantics differ")
    return digest


def build_target_shard_plan(
    *,
    campaign_spec_sha256: str,
    target_cache_specification_sha256: str,
    identity_order_sha256: str,
    event_count: int,
    shard_size: int,
    maximum_concurrent_tasks: int,
) -> dict[str, Any]:
    if (
        int(event_count) <= 0
        or int(shard_size) <= 0
        or int(maximum_concurrent_tasks) <= 0
    ):
        raise ValueError("target shard plan dimensions differ")
    rows = []
    for index, start in enumerate(range(0, int(event_count), int(shard_size))):
        stop = min(start + int(shard_size), int(event_count))
        rows.append(
            {
                "shard_index": index,
                "start_index": start,
                "stop_index_exclusive": stop,
                "manifest_filename": f"shard_{index:06d}.json",
                "npz_filename": f"shard_{index:06d}.npz",
            }
        )
    return with_content_hash(
        {
            "contract": TARGET_SHARD_PLAN_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "target_cache_specification_sha256": require_sha256(
                target_cache_specification_sha256,
                name="target_cache_specification_sha256",
            ),
            "identity_order_sha256": require_sha256(
                identity_order_sha256, name="identity_order_sha256"
            ),
            "event_count": int(event_count),
            "shard_size": int(shard_size),
            "shard_count": len(rows),
            "rows": rows,
            "maximum_concurrent_tasks": int(maximum_concurrent_tasks),
            "slurm_array": (
                f"0-{len(rows) - 1}%{int(maximum_concurrent_tasks)}"
            ),
            "publication": "immutable_npz_then_immutable_manifest",
            "resume_rule": "reuse_only_after_manifest_and_npz_hash_validation",
            "partial_npz_without_manifest_fails_closed": True,
        }
    )


def validate_target_shard_plan(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=TARGET_SHARD_PLAN_CONTRACT
    )
    expected = build_target_shard_plan(
        campaign_spec_sha256=payload["campaign_spec_sha256"],
        target_cache_specification_sha256=payload[
            "target_cache_specification_sha256"
        ],
        identity_order_sha256=payload["identity_order_sha256"],
        event_count=int(payload["event_count"]),
        shard_size=int(payload["shard_size"]),
        maximum_concurrent_tasks=int(payload["maximum_concurrent_tasks"]),
    )
    if payload != expected:
        raise ValueError("target-shard plan semantics differ")
    return digest


def build_task_manifest(
    *,
    campaign_spec_sha256: str,
    production_graph_sha256: str,
    node_id: str,
    rows: Sequence[Mapping[str, Any]],
    maximum_concurrent_tasks: int,
) -> dict[str, Any]:
    if int(maximum_concurrent_tasks) <= 0 or not rows:
        raise ValueError("task manifest dimensions differ")
    normalized = []
    for index, raw in enumerate(rows):
        if set(raw) != {
            "task_id",
            "argv",
            "environment",
            "expected_outputs",
            "input_artifact_hashes",
        }:
            raise ValueError("task-manifest row fields differ")
        argv = [str(value) for value in raw["argv"]]
        environment = {
            str(key): str(value)
            for key, value in sorted(raw["environment"].items())
        }
        outputs = [str(value) for value in raw["expected_outputs"]]
        input_hashes = {
            str(key): require_sha256(value, name=f"input_artifact_hashes.{key}")
            for key, value in sorted(raw["input_artifact_hashes"].items())
        }
        if (
            str(raw["task_id"]) != f"{node_id}:{index}"
            or not argv
            or any(not value for value in argv)
            or any(not key or "\x00" in value for key, value in environment.items())
            or not outputs
        ):
            raise ValueError("task-manifest row semantics differ")
        normalized.append(
            {
                "task_index": index,
                "task_id": str(raw["task_id"]),
                "argv": argv,
                "environment": environment,
                "expected_outputs": outputs,
                "input_artifact_hashes": input_hashes,
            }
        )
    return with_content_hash(
        {
            "contract": TASK_MANIFEST_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "production_graph_sha256": require_sha256(
                production_graph_sha256, name="production_graph_sha256"
            ),
            "node_id": str(node_id),
            "task_count": len(normalized),
            "maximum_concurrent_tasks": int(maximum_concurrent_tasks),
            "slurm_array": (
                f"0-{len(normalized) - 1}%{int(maximum_concurrent_tasks)}"
            ),
            "rows": normalized,
            "row_order": "zero_based_immutable_registration_order",
            "performance_based_row_skipping": False,
        }
    )


def validate_task_manifest(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=TASK_MANIFEST_CONTRACT
    )
    rows = [
        {
            key: value
            for key, value in row.items()
            if key != "task_index"
        }
        for row in payload["rows"]
    ]
    expected = build_task_manifest(
        campaign_spec_sha256=payload["campaign_spec_sha256"],
        production_graph_sha256=payload["production_graph_sha256"],
        node_id=payload["node_id"],
        rows=rows,
        maximum_concurrent_tasks=int(payload["maximum_concurrent_tasks"]),
    )
    if payload != expected:
        raise ValueError("task-manifest semantics differ")
    return digest


def validate_task_manifest_for_graph(
    payload: Mapping[str, Any],
    *,
    production_graph: Mapping[str, Any],
    campaign_root: str | Path,
    repo_root: str | Path,
) -> str:
    digest = validate_task_manifest(payload)
    graph_sha = validate_production_graph(production_graph)
    nodes = {
        str(node["node_id"]): node for node in production_graph["nodes"]
    }
    node_id = str(payload["node_id"])
    node = nodes.get(node_id)
    execution_entries = {
        str(row["node_id"]): row
        for row in production_graph["node_execution_registry"]["entries"]
    }
    execution = execution_entries.get(node_id)
    node_array = None if node is None else node["array"]
    if node_array is None:
        dimensions_differ = (
            int(payload["task_count"]) != 1
            or int(payload["maximum_concurrent_tasks"]) != 1
        )
    else:
        dimensions_differ = (
            int(payload["task_count"]) > int(node_array["maximum_tasks"])
            or int(payload["maximum_concurrent_tasks"])
            != int(node_array["maximum_concurrent_tasks"])
        )
    if (
        payload["production_graph_sha256"] != graph_sha
        or node is None
        or execution is None
        or execution["manifest_required"] is not True
        or dimensions_differ
    ):
        raise ValueError("task manifest differs from production graph")
    root = Path(campaign_root).resolve()
    source = Path(repo_root).resolve()
    for row in payload["rows"]:
        argv = row["argv"]
        executable = Path(argv[0]).name.lower()
        if (
            len(argv) < 2
            or not executable.startswith("python")
            or Path(argv[1]).is_absolute()
            or Path(argv[1]).suffix != ".py"
        ):
            raise ValueError("task command is not a repository Python entry point")
        entrypoint = (source / argv[1]).resolve()
        try:
            entrypoint.relative_to(source)
        except ValueError as error:
            raise ValueError("task entry point escapes the repository") from error
        if not entrypoint.is_file():
            raise FileNotFoundError(f"task entry point is absent: {entrypoint}")
        if (
            node_id in MIDDLE_NODE_ENTRYPOINTS
            and argv[1].replace("\\", "/")
            != MIDDLE_NODE_ENTRYPOINTS[node_id]
        ):
            raise ValueError(
                f"{node_id} task entry point differs from its Stage F--J "
                "execution contract"
            )
        if (
            node_id in LATE_NODE_ENTRYPOINTS
            and argv[1].replace("\\", "/")
            not in LATE_NODE_ENTRYPOINTS[node_id]
        ):
            raise ValueError(
                f"{node_id} task entry point differs from its Stage K--M "
                "execution contract"
            )
        if (
            node_id in FINAL_NODE_ENTRYPOINTS
            and argv[1].replace("\\", "/")
            not in FINAL_NODE_ENTRYPOINTS[node_id]
        ):
            raise ValueError(
                f"{node_id} task entry point differs from its sealed "
                "Stage-N execution contract"
            )
        for output in row["expected_outputs"]:
            destination = Path(output).resolve()
            try:
                destination.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "task expected output escapes the campaign root"
                ) from error
        if node_id == "hlt_v3_cache":
            occurrences = [
                index
                for index, value in enumerate(argv)
                if value == "--profile-id"
            ]
            if len(occurrences) != 1:
                raise ValueError("HLT-v3 task profile declaration differs")
            profile_index = occurrences[0]
            if (
                profile_index + 1 >= len(argv)
                or argv[profile_index + 1]
                != production_graph["degradation_profile"]
            ):
                raise ValueError(
                    "HLT-v3 task changes the production degradation profile"
                )
    return digest


def build_resume_plan(
    *,
    production_graph: Mapping[str, Any],
    previous_ledger: Mapping[str, Any],
    completed_nodes: Mapping[str, str],
    failed_nodes: Sequence[str],
) -> dict[str, Any]:
    graph_sha = validate_production_graph(production_graph)
    ledger_sha = validate_job_ledger(
        previous_ledger, production_graph=production_graph
    )
    nodes = {
        str(node["node_id"]): node for node in production_graph["nodes"]
    }
    completed_hashes = {
        str(name): require_sha256(
            digest, name=f"completed_nodes.{name}"
        )
        for name, digest in completed_nodes.items()
    }
    completed = set(completed_hashes)
    failed = set(str(value) for value in failed_nodes)
    if (
        completed - set(nodes)
        or failed - set(nodes)
        or completed & failed
    ):
        raise ValueError("resume node states differ")
    reusable = sorted(completed)
    resubmit = []
    blocked = []
    for node_id, node in nodes.items():
        if node_id in completed:
            continue
        missing = [
            parent
            for parent in node["dependencies"]
            if parent not in completed
        ]
        row = {
            "node_id": node_id,
            "previous_job_id": previous_ledger["jobs"].get(node_id),
            "missing_completed_dependencies": missing,
            "previously_failed": node_id in failed,
        }
        if missing:
            blocked.append(row)
        else:
            resubmit.append(row)
    return with_content_hash(
        {
            "contract": RESUME_PLAN_CONTRACT,
            "schema_version": 1,
            "production_graph_sha256": graph_sha,
            "previous_job_ledger_sha256": ledger_sha,
            "reusable_completed_nodes": [
                {
                    "node_id": node_id,
                    "output_artifact_sha256": completed_hashes[node_id],
                }
                for node_id in reusable
            ],
            "ready_to_resubmit": resubmit,
            "blocked_until_dependencies_complete": blocked,
            "reuse_requires_authenticated_output_hashes": True,
            "performance_based_resubmission": False,
            "failed_execution_may_be_resubmitted": True,
            "final_test_may_run_more_than_once": False,
        }
    )


def build_step15_bundle(
    *,
    production_graph: Mapping[str, Any],
    dry_run_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    graph_sha = validate_production_graph(production_graph)
    ledger_sha = validate_job_ledger(
        dry_run_ledger, production_graph=production_graph
    )
    if dry_run_ledger["submission_mode"] not in {
        "dry_run",
        "smoke_simulation",
    }:
        raise ValueError("Step-15 bundle requires a non-mutating ledger")
    return with_content_hash(
        {
            "contract": STEP15_BUNDLE_CONTRACT,
            "schema_version": 23,
            "production_graph_sha256": graph_sha,
            "dry_run_job_ledger_sha256": ledger_sha,
            "stage_coverage": list("ABCDEFGHIJKLMN"),
            "both_stage_n_selectors_present": True,
            "scale_up_present": True,
            "bounded_arrays_present": True,
            "resumable_target_shards_present": True,
            "dynamic_continuation_present": True,
            "dynamic_continuation_contracts": {
                "intent": "retb_dynamic_continuation_intent_v1",
                "binding": "retb_dynamic_continuation_binding_v1",
                "task_manifest": TASK_MANIFEST_CONTRACT,
            },
            "dynamic_manifest_execution_requires_binding_receipt": True,
            "stage_f_j_completion_contracts": {
                "row": "retb_task_row_completion_v1",
                "manifest": "retb_task_manifest_completion_v1",
                "continuation_gate": MIDDLE_CONTINUATION_GATE_CONTRACT,
                "continuation_bundle": (
                    "retb_stage_f_j_continuation_bundle_v1"
                ),
            },
            "stage_f_j_dependents_require_complete_parent_attestations": True,
            "stage_f_j_row_reuse_requires_output_revalidation": True,
            "stage_k_m_completion_contracts": {
                "continuation_gate": LATE_CONTINUATION_GATE_CONTRACT,
                "continuation_bundle": (
                    "retb_stage_k_m_continuation_bundle_v1"
                ),
                "confirmation_execution_plan": (
                    "retb_500k_confirmation_execution_plan_v1"
                ),
                "scale_refit_execution_plan": (
                    "retb_scale_refit_execution_plan_v1"
                ),
                "scale_graph_execution_plan": (
                    "retb_scale_graph_execution_plan_v1"
                ),
                "task_manifest_completion": (
                    "retb_task_manifest_completion_v1"
                ),
                "scale_completion": "retb_scale_completion_v2",
                "shortlisted_500k_controls": (
                    "retb_shortlisted_500k_controls_v3"
                ),
                "semantic_controls": (
                    "retb_stage_k_semantic_controls_bundle_v5"
                ),
                "scale_refit_phase_completion": (
                    "retb_scale_refit_phase_completion_v2"
                ),
                "scale_graph_phase_completion": (
                    "retb_scale_graph_phase_completion_v1"
                ),
            },
            "stage_k_m_dependents_require_complete_parent_attestations": True,
            "stage_l_m_registration_only_rows_forbidden": True,
            "negative_control_or_scale_results_continue": True,
            "all_shortlisted_and_named_baseline_graph_seed_rows_required": True,
            "stage_n_completion_contracts": {
                "continuation_gate": FINAL_CONTINUATION_GATE_CONTRACT,
                "continuation_bundle": (
                    "retb_stage_n_continuation_bundle_v1"
                ),
                "stack_inference_execution_plan": (
                    "retb_stack_val_inference_execution_plan_v1"
                ),
                "locked_scale_finalists": (
                    "retb_locked_scale_finalists_v1"
                ),
                "deployable_inference_input": (
                    "retb_deployable_inference_input_v1"
                ),
                "deployable_inference_input_binding": (
                    "retb_deployable_inference_input_v2"
                ),
                "shared_deployable_inference_payload": (
                    "retb_shared_deployable_inference_payload_v1"
                ),
                "deployable_inference_entrypoint": (
                    "scripts/run_retb_deployable_inference.py"
                ),
                "postlock_target_execution_plan": (
                    "retb_postlock_target_execution_plan_v1"
                ),
                "finalist_controls_execution_plan": (
                    "retb_finalist_controls_execution_plan_v1"
                ),
                "finalist_controls": (
                    "retb_scale_finalist_controls_v1"
                ),
                "evidence_join": "retb_stage_n_evidence_join_v1",
                "final_test_execution_lock": (
                    "retb_final_test_execution_lock_v1"
                ),
                "sealed_final_test_execution_plan": (
                    "retb_sealed_final_test_execution_plan_v3"
                ),
                "final_test_execution_claim": (
                    "retb_final_test_execution_claim_v3"
                ),
                "final_test_inference_attestation": (
                    "retb_final_test_inference_attestation_v2"
                ),
                "final_test_row_completion": (
                    "retb_final_test_row_completion_v3"
                ),
                "sealed_final_test_evaluation": (
                    "retb_sealed_final_test_evaluation_v3"
                ),
                "final_report": "retb_stage_mn_final_report_v3",
                "completed_job_ledger": JOB_LEDGER_CONTRACT,
            },
            "sealed_final_test_task_count": 1,
            "final_test_evaluation_exactly_once": True,
            "task8_operational_validation_contracts": {
                "local_report": "retb_local_operational_report_v3",
                "tigris_smoke_evidence": (
                    "retb_tigris_smoke_evidence_v1"
                ),
                "production_dry_run_evidence": (
                    "retb_production_dry_run_evidence_v1"
                ),
                "full_submission_authorization": (
                    "retb_full_submission_authorization_v2"
                ),
            },
            "manifest_orchestration_contracts": {
                "materialization_plan": (
                    "retb_manifest_materialization_plan_v2"
                ),
                "producer_receipt": "retb_manifest_producer_receipt_v3",
                "plan_factory_registry": (
                    MANIFEST_PLAN_FACTORY_REGISTRY_CONTRACT
                ),
                "plan_producer_audit": (
                    "retb_manifest_plan_producer_audit_v1"
                ),
                "post_completion_hook_required": True,
                "missing_or_drifted_plan_fails_closed": True,
                "reused_manifest_requires_plan_completion_and_trigger": True,
                "hook_presence_is_not_plan_factory_evidence": True,
                "real_miniature_execution_required_for_readiness": True,
                "scientific_performance_used_as_gate": False,
            },
            "full_submission_requires_current_source_authorization": True,
            "smoke_submission_supported": True,
            "full_submission_supported": True,
            "monitoring_supported": True,
            "node_execution_registry_present": True,
            "manifest_producer_registration_coverage_complete": True,
            "automatic_manifest_producer_invocation_requires_task8_audit": True,
            "performance_based_termination": False,
        }
    )


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DIRECT_WORKER_NODES",
    "FINAL_CONTINUATION_GATE_CONTRACT",
    "FINAL_CONTINUATION_MANIFEST_NODES",
    "FINAL_NODE_ENTRYPOINTS",
    "JOB_LEDGER_CONTRACT",
    "LATE_CONTINUATION_GATE_CONTRACT",
    "LATE_CONTINUATION_MANIFEST_NODES",
    "LATE_NODE_ENTRYPOINTS",
    "MANIFEST_PLAN_FACTORY_REGISTRY_CONTRACT",
    "MIDDLE_CONTINUATION_GATE_CONTRACT",
    "MIDDLE_CONTINUATION_MANIFEST_NODES",
    "MIDDLE_NODE_ENTRYPOINTS",
    "MINIATURE_SPLIT_SIZES",
    "NODE_EXECUTION_REGISTRY_CONTRACT",
    "PRODUCTION_GRAPH_CONTRACT",
    "PRODUCTION_SPLIT_SIZES",
    "RESOURCE_PROBE_CONTRACT",
    "RESUME_PLAN_CONTRACT",
    "STEP15_BUNDLE_CONTRACT",
    "STATIC_EXPERIMENT_MANIFEST_NODES",
    "TASK_MANIFEST_CONTRACT",
    "TARGET_SHARD_PLAN_CONTRACT",
    "TASK_MANIFEST_PRODUCER_NODES",
    "TIGRIS_DEFAULTS",
    "build_job_ledger",
    "build_node_execution_registry",
    "build_production_graph",
    "build_resource_probe",
    "build_resume_plan",
    "build_step15_bundle",
    "build_task_manifest",
    "build_target_shard_plan",
    "task_manifest_path_for_graph",
    "validate_job_ledger",
    "validate_node_execution_registry",
    "validate_production_graph",
    "validate_production_campaign_binding",
    "validate_resource_probe",
    "validate_task_manifest",
    "validate_task_manifest_for_graph",
    "validate_target_shard_plan",
]

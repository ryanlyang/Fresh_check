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


PRODUCTION_GRAPH_CONTRACT = "retb_tigris_production_graph_v1"
JOB_LEDGER_CONTRACT = "retb_tigris_job_ledger_v1"
RESOURCE_PROBE_CONTRACT = "retb_tigris_resource_probe_v1"
TARGET_SHARD_PLAN_CONTRACT = "retb_target_shard_execution_plan_v1"
TASK_MANIFEST_CONTRACT = "retb_tigris_task_manifest_v1"
RESUME_PLAN_CONTRACT = "retb_tigris_resume_plan_v1"
STEP15_BUNDLE_CONTRACT = "retb_step15_production_bundle_v1"

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
    "cpu_cache": 12,
    "gpu_expert": 4,
    "gpu_predictor": 4,
    "gpu_scale": 3,
    "gpu_final": 3,
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
    "model_train": 20,
    "model_val": 20,
    "stack_train": 0,
    "stack_val": 10,
    "final_test": 20,
    "scale_train": 40,
}


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
            "input_audit",
            stage="A",
            worker="run_retb_audit_inputs.sh",
            dependencies=("hlt_v3_cache", "compiled_region_backend"),
        ),
        _node(
            "normalizers_500k",
            stage="A",
            worker="run_retb_fit_normalizers.sh",
            dependencies=("input_audit",),
            access="model_train_features_only",
        ),
        _node(
            "step3_architecture_contracts",
            stage="A",
            worker="run_retb_build_step3_contracts.sh",
            dependencies=("normalizers_500k", "gpu_resource_probe"),
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
            "offline_shape_selector",
            stage="B",
            worker="run_retb_select_shapes.sh",
            dependencies=("offline_expert_training",),
            dynamic=True,
            access="val_design_only",
        ),
        _node(
            "offline_optimization_selector",
            stage="B",
            worker="run_retb_select_optimization.sh",
            dependencies=("offline_expert_training",),
            dynamic=True,
            access="val_design_only",
        ),
        _node(
            "step5_offline_fusion_contracts",
            stage="C",
            worker="run_retb_build_step5_contracts.sh",
            dependencies=(
                "offline_shape_selector",
                "offline_optimization_selector",
            ),
        ),
        _node(
            "offline_fusion_training",
            stage="C",
            worker="run_retb_train_offline_fusion.sh",
            dependencies=(
                "step5_offline_fusion_contracts",
            ),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_c_offline_fusions.json",
                concurrency=expert,
                maximum_tasks=256,
            ),
            dynamic=True,
            resumable=True,
            access="model_train_and_val_stop",
        ),
        _node(
            "offline_complementarity",
            stage="C",
            worker="run_retb_analyze_complementarity.sh",
            dependencies=("offline_fusion_training",),
            access="val_design_only",
        ),
        _node(
            "offline_capacity_controls",
            stage="C",
            worker="run_retb_capacity_controls.sh",
            dependencies=("offline_fusion_training",),
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
            dependencies=("step5_offline_fusion_contracts", "input_audit"),
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
                maximum_tasks=512,
            ),
            dynamic=True,
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
            dynamic=True,
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
            dynamic=True,
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
                maximum_tasks=512,
            ),
            dynamic=True,
            resumable=True,
        ),
        _node(
            "bridge_content_certification",
            stage="E",
            worker="run_retb_certify_bridge_content.sh",
            dependencies=("bridge_target_training",),
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
            access="val_design_only",
        ),
        _node(
            "bridge_shape_selector",
            stage="L",
            worker="run_retb_select_bridge_shape.sh",
            dependencies=("confirmation_summary",),
            access="val_design_only",
        ),
        _node(
            "scale_shortlist_selector",
            stage="L",
            worker="run_retb_scale_shortlist.sh",
            dependencies=("confirmation_summary", "bridge_shape_selector"),
            access="val_design_only",
        ),
        _node(
            "step14_scale_final_contracts",
            stage="M",
            worker="run_retb_build_step14_contracts.sh",
            dependencies=("scale_shortlist_selector",),
        ),
        _node(
            "scale_refits",
            stage="M",
            worker="run_retb_register_scale_refits.sh",
            dependencies=("step14_scale_final_contracts",),
            array=_array(
                task_manifest="job_ledgers/tasks/stage_m_scale_refits.json",
                concurrency=cpu,
                maximum_tasks=64,
            ),
            dynamic=True,
            resumable=True,
            access="scale_train_and_label_free_val_design",
        ),
        _node(
            "scale_graph_training",
            stage="M",
            worker="run_retb_train_scale_shortlist.sh",
            dependencies=("scale_refits",),
            resource="gpu",
            array=_array(
                task_manifest="job_ledgers/tasks/stage_m_scale_graphs.json",
                concurrency=scale,
                maximum_tasks=18,
                smoke_tasks=1,
            ),
            dynamic=True,
            resumable=True,
            access="scale_train_and_val_stop",
        ),
        _node(
            "scale_completion",
            stage="M",
            worker="run_retb_scale_completion.sh",
            dependencies=("scale_graph_training",),
        ),
        _node(
            "prelock_final_inputs",
            stage="N",
            worker="run_retb_prepare_final_inputs.sh",
            dependencies=("input_audit",),
            access="checkpoint_free_final_input_preparation",
            resumable=True,
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
                maximum_tasks=30,
            ),
            dynamic=True,
            resumable=True,
        ),
        _node(
            "final_test_execution_lock",
            stage="N",
            worker="run_retb_final_execution_lock.sh",
            dependencies=(
                "postlock_oracle_targets",
                "finalist_controls",
                "prelock_final_inputs",
            ),
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
                maximum_tasks=30,
            ),
            dynamic=True,
            resumable=False,
            access="sealed_final_test_once",
        ),
        _node(
            "final_report",
            stage="N",
            worker="run_retb_step14_report.sh",
            dependencies=("sealed_final_test",),
            access="authenticated_final_predictions_only",
        ),
        _node(
            "completed_job_ledger",
            stage="N",
            worker="run_retb_finalize_job_ledger.sh",
            dependencies=("final_report",),
        ),
    ]


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
    profile = (
        "nonproduction_miniature_test"
        if miniature
        else "production_500k_100k_50k_300k_scale3m"
    )
    artifact = with_content_hash(
        {
            "contract": PRODUCTION_GRAPH_CONTRACT,
            "schema_version": 1,
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
            "stage_order": list("ABCDEFGHIJKLMN"),
            "two_stage_n_selectors": [
                "accuracy_finalist_selector",
                "rejection_finalist_selector",
            ],
            "performance_based_termination": False,
            "negative_campaign_continues_to_final_report": True,
            "final_test_before_both_locks_allowed": False,
            "production_submission_performed": False,
            "monitoring": {
                "queue": 'squeue -u "$USER" -o "%i %j %T %R"',
                "accounting": (
                    "sacct -X --starttime today --name retb_ "
                    "--format=JobID,JobName,State,Elapsed,ExitCode"
                ),
                "resume": (
                    "bash sbatch/submit_retb_tigris_full.sh "
                    "--resume \"${CAMPAIGN_ROOT}\""
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
    if (
        payload["degradation_profile"] != "D_NOMINAL"
        or payload["degradation_profile_implicit_override_allowed"] is not False
        or payload["performance_based_termination"] is not False
        or payload["negative_campaign_continues_to_final_report"] is not True
        or payload["final_test_before_both_locks_allowed"] is not False
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
    return with_content_hash(
        {
            "contract": JOB_LEDGER_CONTRACT,
            "schema_version": 1,
            "production_graph_sha256": graph_sha,
            "campaign_id": production_graph["campaign_id"],
            "campaign_root": production_graph["campaign_root"],
            "submission_mode": submission_mode,
            "jobs": normalized,
            "resolved_arrays": arrays,
            "submitted_node_count": sum(
                value is not None for value in normalized.values()
            ),
            "all_nodes_bound": all(
                value is not None for value in normalized.values()
            ),
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
    if (
        payload["production_graph_sha256"] != graph_sha
        or node_id not in nodes
        or nodes[node_id]["array"] is None
        or int(payload["task_count"])
        > int(nodes[node_id]["array"]["maximum_tasks"])
        or int(payload["maximum_concurrent_tasks"])
        != int(nodes[node_id]["array"]["maximum_concurrent_tasks"])
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
            "schema_version": 1,
            "production_graph_sha256": graph_sha,
            "dry_run_job_ledger_sha256": ledger_sha,
            "stage_coverage": list("ABCDEFGHIJKLMN"),
            "both_stage_n_selectors_present": True,
            "scale_up_present": True,
            "bounded_arrays_present": True,
            "resumable_target_shards_present": True,
            "dynamic_continuation_present": True,
            "smoke_submission_supported": True,
            "full_submission_supported": True,
            "monitoring_supported": True,
            "performance_based_termination": False,
        }
    )


__all__ = [
    "DEFAULT_CONCURRENCY",
    "JOB_LEDGER_CONTRACT",
    "MINIATURE_SPLIT_SIZES",
    "PRODUCTION_GRAPH_CONTRACT",
    "PRODUCTION_SPLIT_SIZES",
    "RESOURCE_PROBE_CONTRACT",
    "RESUME_PLAN_CONTRACT",
    "STEP15_BUNDLE_CONTRACT",
    "TASK_MANIFEST_CONTRACT",
    "TARGET_SHARD_PLAN_CONTRACT",
    "TIGRIS_DEFAULTS",
    "build_job_ledger",
    "build_production_graph",
    "build_resource_probe",
    "build_resume_plan",
    "build_step15_bundle",
    "build_task_manifest",
    "build_target_shard_plan",
    "validate_job_ledger",
    "validate_production_graph",
    "validate_production_campaign_binding",
    "validate_resource_probe",
    "validate_task_manifest",
    "validate_task_manifest_for_graph",
    "validate_target_shard_plan",
]

"""Central contract for production RETB manifest-plan factories.

The registry deliberately separates declaring who must build a downstream
task plan from proving that the factory exists and has run.  Closure blocks
2--5 add the executable factory implementations; until then the operational
readiness audit remains fail-closed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import validate_content_hash, with_content_hash


MANIFEST_PLAN_FACTORY_REGISTRY_CONTRACT = (
    "retb_manifest_plan_factory_registry_v8"
)
MANIFEST_PLAN_PRODUCER_AUDIT_CONTRACT = (
    "retb_manifest_plan_producer_audit_v1"
)
PLAN_FACTORY_ENTRYPOINT = (
    "scripts/produce_retb_downstream_manifest_plans.py"
)


_EARLY_ALLOWED_WORKERS: dict[str, tuple[str, ...]] = {
    "offline_optimization_selector": (
        "scripts/execute_retb_offline_optimization_wave.py",
    ),
    "offline_shape_selector": (
        "scripts/execute_retb_offline_shape_wave.py",
    ),
    "offline_complementarity": (
        "scripts/execute_retb_offline_complementarity_wave.py",
    ),
    "offline_capacity_controls": (
        "scripts/execute_retb_offline_capacity_wave.py",
    ),
    "bridge_target_training": ("scripts/train_retb_bridge_target.py",),
    "bridge_content_certification": (
        "scripts/execute_retb_bridge_certification_wave.py",
    ),
    "target_coordinate_selector": (
        "scripts/execute_retb_target_coordinate_selection.py",
    ),
}

_MIDDLE_ALLOWED_WORKERS: dict[str, tuple[str, ...]] = {
    "target_cache_build": (
        "scripts/execute_retb_target_cache_row.py",
    ),
    "target_normalizers": ("scripts/fit_retb_target_normalizers.py",),
    "predictor_training": (
        "scripts/execute_retb_predictor_campaign.py",
    ),
    "uncertainty_calibration": (
        "scripts/calibrate_retb_uncertainty.py",
    ),
    "predictor_bundle_selector": (
        "scripts/execute_retb_predictor_bundle_selection.py",
    ),
    "oracle_substitutions": (
        "scripts/execute_retb_stage_i_oracle_wave.py",
    ),
    "joint_predictor_training": (
        "scripts/execute_retb_joint_campaign.py",
    ),
    "joint_predictor_selector": (
        "scripts/finalize_retb_joint_campaign.py",
    ),
    "final_consumer_training": (
        "scripts/execute_retb_final_consumer_campaign.py",
    ),
    "deployable_export": (
        "scripts/execute_retb_deployable_export_campaign.py",
    ),
}

_LATE_ALLOWED_WORKERS: dict[str, tuple[str, ...]] = {
    "robustness_controls": (
        "scripts/execute_retb_robustness_campaign.py",
        "scripts/evaluate_retb_final_consumer_reference.py",
        "scripts/evaluate_retb_final_consumer_bypass_controls.py",
        "scripts/evaluate_retb_stage_i_substitutions.py",
    ),
    "semantic_controls": (
        "scripts/execute_retb_semantic_control_campaign.py",
        "scripts/evaluate_retb_final_consumer_bypass_controls.py",
        "scripts/evaluate_retb_stage_i_substitutions.py",
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
    "scale_refits": ("scripts/execute_retb_scale_seed_refit.py",),
    "scale_graph_training": (
        "scripts/execute_retb_scale_graph_pipeline.py",
    ),
    "scale_completion": ("scripts/aggregate_retb_scale_completion.py",),
}

_FINAL_ALLOWED_WORKERS: dict[str, tuple[str, ...]] = {
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
    "sealed_final_test": ("scripts/execute_retb_sealed_final_test.py",),
    "final_report": ("scripts/write_retb_step14_report.py",),
}

ALLOWED_WORKER_ENTRYPOINTS = {
    **_EARLY_ALLOWED_WORKERS,
    **_MIDDLE_ALLOWED_WORKERS,
    **_LATE_ALLOWED_WORKERS,
    **_FINAL_ALLOWED_WORKERS,
}


def _dataset_access_role(target: str, stage: str) -> str:
    if target == "prelock_final_inputs":
        return "final_test_raw_inputs_only_no_model_outputs"
    if target == "stack_val_inference":
        return "stack_val_deployable_label_free_inference"
    if target == "accuracy_finalist_selector":
        return "stack_val_selector_label_join"
    if target == "postlock_oracle_targets":
        return "postlock_selection_ineligible_oracle_targets"
    if target in {
        "finalist_controls",
        "final_test_execution_lock",
        "sealed_final_test",
        "final_report",
    }:
        return "locked_final_evaluation"
    if stage == "M":
        return "scale_train_and_validation_only"
    return "model_train_model_val_and_val_design_only"


def _build_manifest_plan_factory_registry(
    *,
    nodes: Sequence[Mapping[str, Any]],
    manifest_producer_nodes: Mapping[str, str],
    bootstrap_targets: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Declare every non-bootstrap plan factory required by the graph."""

    by_id = {str(node["node_id"]): node for node in nodes}
    downstream = sorted(set(manifest_producer_nodes) - set(bootstrap_targets))
    if set(ALLOWED_WORKER_ENTRYPOINTS) != set(downstream):
        missing = sorted(set(downstream) - set(ALLOWED_WORKER_ENTRYPOINTS))
        extra = sorted(set(ALLOWED_WORKER_ENTRYPOINTS) - set(downstream))
        raise ValueError(
            "manifest plan worker registry coverage differs: "
            f"missing={missing}, extra={extra}"
        )
    entries: list[dict[str, Any]] = []
    for target in downstream:
        node = by_id[target]
        producer = str(manifest_producer_nodes[target])
        producer_node = by_id[producer]
        dynamic = bool(node["dynamic_continuation"])
        producer_completion_contract = (
            "retb_task_manifest_completion_v1"
            if producer in manifest_producer_nodes
            else f"node_completion:{producer}"
        )
        entries.append(
            {
                "factory_id": f"retb_plan_factory:{target}:v7",
                "target_node_id": target,
                "producer_node_id": producer,
                "plan_factory_entrypoint": PLAN_FACTORY_ENTRYPOINT,
                "plan_factory_symbol": f"build_{target}_manifest_plan",
                "required_producer_artifacts": [
                    f"node_completion:{producer}",
                    *(
                        f"node_completion:{dependency}"
                        for dependency in producer_node["dependencies"]
                    ),
                ],
                "required_producer_contracts": [
                    producer_completion_contract,
                ],
                "trigger_artifact_rule": (
                    "factory_declared_immutable_trigger"
                    if dynamic
                    else "not_applicable_static_plan"
                ),
                "trigger_hash_field": (
                    "content_hash" if dynamic else None
                ),
                "row_count_rule": (
                    "deterministic_from_complete_authenticated_"
                    f"{producer}_artifacts"
                ),
                "allowed_worker_entrypoints": list(
                    ALLOWED_WORKER_ENTRYPOINTS[target]
                ),
                "expected_output_contracts": [
                    "retb_task_row_completion_v1",
                    "retb_task_manifest_completion_v1",
                ],
                "row_resolution": "dynamic" if dynamic else "static",
                "dataset_access_role": _dataset_access_role(
                    target, str(node["stage"])
                ),
                "resource_type": str(node["resource"]),
                "scientific_underperformance_blocks_continuation": False,
                "performance_independent_continuation_required": True,
            }
        )
    artifact = with_content_hash(
        {
            "contract": MANIFEST_PLAN_FACTORY_REGISTRY_CONTRACT,
            "schema_version": 7,
            "entries": entries,
            "entry_count": len(entries),
            "bootstrap_manifest_target_count": len(bootstrap_targets),
            "manifest_target_count": len(manifest_producer_nodes),
            "factory_implementation_must_emit_authenticated_plan": True,
            "shared_materialization_hook_is_not_factory_evidence": True,
            "synthetic_rows_are_not_production_evidence": True,
        }
    )
    return artifact


def build_manifest_plan_factory_registry(
    *,
    nodes: Sequence[Mapping[str, Any]],
    manifest_producer_nodes: Mapping[str, str],
    bootstrap_targets: set[str] | frozenset[str],
) -> dict[str, Any]:
    artifact = _build_manifest_plan_factory_registry(
        nodes=nodes,
        manifest_producer_nodes=manifest_producer_nodes,
        bootstrap_targets=bootstrap_targets,
    )
    validate_manifest_plan_factory_registry(
        artifact,
        nodes=nodes,
        manifest_producer_nodes=manifest_producer_nodes,
        bootstrap_targets=bootstrap_targets,
    )
    return artifact


def validate_manifest_plan_factory_registry(
    payload: Mapping[str, Any],
    *,
    nodes: Sequence[Mapping[str, Any]],
    manifest_producer_nodes: Mapping[str, str],
    bootstrap_targets: set[str] | frozenset[str],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=MANIFEST_PLAN_FACTORY_REGISTRY_CONTRACT
    )
    entries = list(payload.get("entries", ()))
    targets = [
        str(entry.get("target_node_id", "")) for entry in entries
    ]
    expected_targets = sorted(
        set(manifest_producer_nodes) - set(bootstrap_targets)
    )
    if (
        len(targets) != len(set(targets))
        or targets != expected_targets
        or int(payload.get("entry_count", -1)) != len(expected_targets)
    ):
        raise ValueError("manifest plan factory registration coverage differs")
    expected = _build_manifest_plan_factory_registry(
        nodes=nodes,
        manifest_producer_nodes=manifest_producer_nodes,
        bootstrap_targets=bootstrap_targets,
    )
    if dict(payload) != expected:
        raise ValueError("manifest plan factory registry semantics differ")
    return digest


__all__ = [
    "ALLOWED_WORKER_ENTRYPOINTS",
    "MANIFEST_PLAN_FACTORY_REGISTRY_CONTRACT",
    "MANIFEST_PLAN_PRODUCER_AUDIT_CONTRACT",
    "PLAN_FACTORY_ENTRYPOINT",
    "build_manifest_plan_factory_registry",
    "validate_manifest_plan_factory_registry",
]

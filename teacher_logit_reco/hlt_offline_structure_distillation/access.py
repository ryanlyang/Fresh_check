"""Fail-closed dataset and artifact capabilities for HOSD workers."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import REGISTRY_CONTRACT, validate_content_hash, with_content_hash


ROLE_CAPABILITIES: dict[str, dict[str, tuple[str, ...]]] = {
    "campaign_builder": {
        "read": (
            "source_snapshot",
            "parent_split_manifest",
            "storage_measurements",
            "inherited_parent_artifacts",
        ),
        "write": ("campaign_control_plane", "parent_rebuild_submissions"),
    },
    "target_builder": {
        "read": (
            "authenticated_raw_schema",
            "resolved_inherited_parent_lock",
            "model_train_offline",
            "val_stop_offline",
            "val_design_offline",
        ),
        "write": ("capability_audit", "target_registry", "canonical_targets"),
    },
    "teacher_inference": {
        "read": ("model_train_offline", "val_stop_offline", "val_design_offline"),
        "write": ("teacher_targets",),
    },
    "train_worker": {
        "read": (
            "model_train_hlt",
            "model_train_labels",
            "model_train_targets",
            "val_stop_hlt",
            "val_stop_labels",
            "val_stop_targets",
            "design_select_hlt",
            "design_select_labels",
            "design_select_targets",
        ),
        "write": ("training_checkpoints", "training_curves"),
    },
    "probe_worker": {
        "read": (
            "model_train_hlt",
            "model_train_labels",
            "model_train_targets",
            "val_stop_hlt",
            "val_stop_labels",
            "val_stop_targets",
            "design_select_hlt",
            "design_select_labels",
            "design_select_targets",
        ),
        "write": ("probe_checkpoints", "probe_predictions"),
    },
    "design_inference": {
        "read": ("val_design_hlt",),
        "write": ("val_design_predictions",),
    },
    "design_selector": {
        "read": (
            "design_select_predictions",
            "design_select_labels",
            "design_select_targets",
        ),
        "write": ("design_selection_locks",),
    },
    "design_confirmer": {
        "read": (
            "design_confirm_predictions",
            "design_confirm_labels",
            "design_confirm_targets",
            "design_selection_locks",
        ),
        "write": ("mechanism_confirmation",),
    },
    "stack_inference": {
        "read": ("stack_val_hlt", "locked_scale_graphs"),
        "write": ("stack_val_identity_logits_probabilities",),
    },
    "stack_selector": {
        "read": (
            "stack_val_identity_logits_probabilities",
            "final_select_label_manifest",
        ),
        "write": ("locked_hosd_finalists",),
    },
    "postlock_oracle_diagnostic": {
        "read": (
            "stack_val_hlt",
            "stack_val_offline",
            "locked_hosd_finalists",
        ),
        "write": ("postlock_oracle_diagnostics",),
    },
    "final_input_preparer": {
        "read": ("final_test_raw_inputs", "locked_hosd_finalists"),
        "write": ("final_test_prepared_inputs",),
    },
    "final_inference": {
        "read": ("final_test_prepared_inputs", "final_test_execution_lock"),
        "write": ("final_test_predictions",),
    },
    "label_auditor": {
        "read": ("published_identity_target_manifest", "split_label_manifest"),
        "write": ("label_correlation_audit", "shuffle_plans"),
    },
    "reporter": {
        "read": ("authenticated_metrics", "job_ledgers", "selection_locks"),
        "write": ("reports",),
    },
}


FORBIDDEN_RUNTIME_RESOURCES = (
    "offline_input",
    "offline_target",
    "target_loss_mask",
    "truth_target",
    "oracle_feedback",
    "teacher_state",
    "class_label",
    "degradation_source_lineage",
)


def build_access_role_registry(*, source: Mapping[str, Any]) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": REGISTRY_CONTRACT,
            "schema_version": 1,
            "registry_id": "access_role_registry",
            "roles": {
                role: {
                    "read": list(capabilities["read"]),
                    "write": list(capabilities["write"]),
                }
                for role, capabilities in ROLE_CAPABILITIES.items()
            },
            "runtime_forbidden_resources": list(FORBIDDEN_RUNTIME_RESOURCES),
            "training_provenance_may_reach_privileged_evidence": True,
            "runtime_reachability_must_be_hlt_only": True,
            "performance_based_access_change_allowed": False,
            "source": dict(source),
        }
    )


def authorize_access(
    *,
    worker_role: str,
    requested_resource: str,
    operation: str = "read",
) -> None:
    if worker_role not in ROLE_CAPABILITIES:
        raise ValueError(f"unknown HOSD worker role {worker_role!r}")
    if operation not in {"read", "write"}:
        raise ValueError(f"unknown HOSD access operation {operation!r}")
    if requested_resource not in ROLE_CAPABILITIES[worker_role][operation]:
        raise PermissionError(
            f"{worker_role} is not authorized to {operation} "
            f"{requested_resource}"
        )


def validate_access_role_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(payload, expected_contract=REGISTRY_CONTRACT)
    if payload.get("registry_id") != "access_role_registry":
        raise ValueError("access-role registry ID differs")
    expected = build_access_role_registry(source=payload["source"])
    if dict(payload) != expected:
        raise ValueError("access-role registry semantics differ")
    return digest


__all__ = [
    "FORBIDDEN_RUNTIME_RESOURCES",
    "ROLE_CAPABILITIES",
    "authorize_access",
    "build_access_role_registry",
    "validate_access_role_registry",
]

"""Fail-closed source and dataset-role authorization for RETB workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    CAMPAIGN_SPEC_CONTRACT,
    load_hashed_json,
    validate_content_hash,
)
from .provenance import source_snapshot


ROLE_ACCESS = {
    "training_worker": {"model_train", "val_stop"},
    "scale_training_worker": {"scale_train", "val_stop"},
    "design_worker": {"val_design"},
    "stage_n_selection_inference": {"stack_val_features"},
    "stage_n_selector": {"final_select_label_manifest"},
    "postlock_stack_diagnostic": {"stack_val_features", "stack_val_oracle_targets"},
    "final_test_input_preparation": {"final_test_inputs"},
    "final_test_worker": {"final_test_inputs", "final_test_targets"},
}


def validate_campaign_source(
    campaign_spec: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    validate_content_hash(campaign_spec, expected_contract=CAMPAIGN_SPEC_CONTRACT)
    current = source_snapshot(repo_root)
    expected = campaign_spec["source"]
    actual = {
        "commit": current["source_commit"],
        "status_sha256": current["source_status_sha256"],
        "dirty": bool(current["source_dirty"]),
        "status_hash_policy": expected["status_hash_policy"],
    }
    if actual != expected:
        raise ValueError("active repository source snapshot differs from campaign spec")
    return actual


def load_and_validate_campaign_source(
    campaign_root: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    campaign = load_hashed_json(
        Path(campaign_root) / "campaign_spec.json",
        expected_contract=CAMPAIGN_SPEC_CONTRACT,
    )
    validate_campaign_source(campaign, repo_root=repo_root)
    return campaign


def authorize_dataset_access(*, worker_role: str, requested_resource: str) -> None:
    if worker_role not in ROLE_ACCESS:
        raise ValueError(f"unknown worker role {worker_role!r}")
    if requested_resource not in ROLE_ACCESS[worker_role]:
        raise PermissionError(
            f"{worker_role} is not authorized for {requested_resource}"
        )


__all__ = [
    "ROLE_ACCESS",
    "authorize_dataset_access",
    "load_and_validate_campaign_source",
    "validate_campaign_source",
]

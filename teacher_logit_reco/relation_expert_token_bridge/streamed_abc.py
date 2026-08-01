"""Authenticated execution policy for storage-bounded offline Stages A--C.

The policy changes materialization, not science: the complete production
offline split, configurations, seeds, run IDs, optimizers, and selectors are
unchanged.  Future-only inputs are deferred and frozen-token banks are scoped
to one ``(shape_id, pipeline_seed)`` worker allocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import validate_content_hash, with_content_hash


STREAMED_ABC_PROFILE = "offline_abc_streamed"
STREAMED_ABC_EXECUTION_CONTRACT = "retb_streamed_abc_execution_profile_v1"
STREAMED_ABC_TREE_INDEX_CONTRACT = "retb_streamed_abc_tree_index_v1"
STREAMED_ABC_INPUT_AUDIT_CONTRACT = "retb_streamed_abc_input_audit_v1"
STREAMED_ABC_FUSION_RECEIPT_CONTRACT = (
    "retb_streamed_abc_fusion_cache_receipt_v1"
)

STREAMED_OFFLINE_ROLES = ("model_train", "val_stop", "val_design")
STREAMED_HLT_VIEWS = tuple(
    ("model_train", replica, "R_MULTI") for replica in range(4)
)


def build_streamed_abc_execution_profile(
    *,
    campaign_id: str,
    campaign_root: str | Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the global storage/execution choices before results exist."""

    return with_content_hash(
        {
            "contract": STREAMED_ABC_EXECUTION_CONTRACT,
            "schema_version": 1,
            "campaign_id": str(campaign_id),
            "campaign_root": str(Path(campaign_root)),
            "source": dict(source),
            "submission_scope": STREAMED_ABC_PROFILE,
            "scientific_scope": ["A", "B", "C"],
            "offline_roles_materialized": list(STREAMED_OFFLINE_ROLES),
            "hlt_views_materialized": [
                {
                    "logical_role": role,
                    "replica_id": replica,
                    "realization_policy": policy,
                }
                for role, replica, policy in STREAMED_HLT_VIEWS
            ],
            "deferred_roles": ["stack_val", "final_test", "scale_train"],
            "frozen_token_cache_lifetime": (
                "single_shape_seed_slurm_allocation"
            ),
            "frozen_token_cache_persistent": False,
            "fusion_wave_coordinate": ["shape_id", "pipeline_seed"],
            "fusion_wave_count": 21,
            "task_local_root_precedence": [
                "RETB_STREAM_ROOT",
                "SLURM_TMPDIR",
                "/dev/shm",
            ],
            "task_local_cleanup": "finally_even_on_failure",
            "persistent_outputs": [
                "normalizers",
                "expert_checkpoints",
                "fusion_checkpoints",
                "predictions",
                "metrics",
                "selectors",
                "content_hashed_receipts",
            ],
            "resume_state_after_success": "removed_after_output_attestation",
            "performance_based_termination": False,
            "scientific_underperformance_blocks_continuation": False,
            "provenance_or_execution_failure_blocks_dependents": True,
            "later_expansion_requires_full_stage_a_audit": True,
        }
    )


def validate_streamed_abc_execution_profile(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STREAMED_ABC_EXECUTION_CONTRACT
    )
    expected = build_streamed_abc_execution_profile(
        campaign_id=str(payload["campaign_id"]),
        campaign_root=str(payload["campaign_root"]),
        source=payload["source"],
    )
    if dict(payload) != expected:
        raise ValueError("streamed A-C execution profile semantics differ")
    return digest


def build_streamed_tree_index(
    *,
    campaign_spec_sha256: str,
    backend_manifest_sha256: str,
    angular_tree_resource_sha256: str,
    execution_profile_sha256: str,
    views: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": STREAMED_ABC_TREE_INDEX_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_spec_sha256,
            "backend_manifest_sha256": backend_manifest_sha256,
            "angular_tree_resource_sha256": angular_tree_resource_sha256,
            "execution_profile_sha256": execution_profile_sha256,
            "views": [dict(row) for row in views],
            "view_ids": [str(row["view_id"]) for row in views],
            "view_count": len(views),
            "source": dict(source),
        }
    )


def build_streamed_input_audit(
    *,
    campaign_spec_sha256: str,
    execution_profile_sha256: str,
    offline_views: Sequence[Mapping[str, Any]],
    hlt_views: Sequence[Mapping[str, Any]],
    tree_index_sha256: str,
    normalizer_bundle_sha256: str,
    hlt_v3_degradation_audit_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": STREAMED_ABC_INPUT_AUDIT_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_spec_sha256,
            "execution_profile_sha256": execution_profile_sha256,
            "offline_views": [dict(row) for row in offline_views],
            "hlt_views": [dict(row) for row in hlt_views],
            "offline_role_order": list(STREAMED_OFFLINE_ROLES),
            "hlt_view_coordinates": [list(row) for row in STREAMED_HLT_VIEWS],
            "tree_index_sha256": tree_index_sha256,
            "normalizer_bundle_sha256": normalizer_bundle_sha256,
            "hlt_v3_degradation_audit_sha256": (
                hlt_v3_degradation_audit_sha256
            ),
            "future_roles_audited": False,
            "valid_for_scientific_stages": ["A", "B", "C"],
            "valid_for_stages_d_through_n": False,
            "source": dict(source),
        }
    )


__all__ = [
    "STREAMED_ABC_EXECUTION_CONTRACT",
    "STREAMED_ABC_FUSION_RECEIPT_CONTRACT",
    "STREAMED_ABC_INPUT_AUDIT_CONTRACT",
    "STREAMED_ABC_PROFILE",
    "STREAMED_ABC_TREE_INDEX_CONTRACT",
    "STREAMED_HLT_VIEWS",
    "STREAMED_OFFLINE_ROLES",
    "build_streamed_abc_execution_profile",
    "build_streamed_input_audit",
    "build_streamed_tree_index",
    "validate_streamed_abc_execution_profile",
]

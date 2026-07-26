"""Immutable provenance for candidate privileged particle-view targets."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .target_generator import (
    PARTICLE_VIEW_PAIR_FEATURE_CONTRACT,
    ParticleViewGeneratorConfig,
    particle_view_generator_config_from_payload,
)


PARTICLE_VIEW_TARGET_CANDIDATE_CONTRACT = "particle_view_target_candidate_v1"
PARTICLE_VIEW_TARGET_SELECTION_STATUSES = (
    "canonical_selectable",
    "selectable",
    "performance_control",
    "diagnostic_nonselectable",
)
PARTICLE_VIEW_TARGET_INDEX_CONTRACT = (
    "hlt_jet_identity_plus_hlt_particle_index_v1"
)


def _validate_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if any(character not in allowed for character in value):
        raise ValueError(f"{name} contains unsupported characters")
    lowered = value.lower()
    if "crossfit" in lowered or "cross_fit" in lowered:
        raise ValueError(f"{name} cannot identify a cross-fit artifact")
    return value


def build_target_candidate_registration(
    *,
    target_id: str,
    campaign_id: str,
    selection_status: str,
    seed: int,
    generator_config: ParticleViewGeneratorConfig,
    source_manifest_sha256: str,
    unified_split_manifest_sha256: str,
    train_split_sha256: str,
    train_identity_sha256: str,
    query_tap_registration_sha256: str,
    query_checkpoint_sha256: str,
    memory_tap_registration_sha256: str,
    memory_checkpoint_sha256: str,
    staged_tap_source_role: str,
    staged_tap_reservation_sha256: str,
    staged_tap_manifest_sha256: str,
    staged_tap_logical_content_sha256: str,
    generator_checkpoint_sha256: str,
    offline_source_sha256: str | None,
    privileged_claim_eligible: bool,
    deployment_control_eligible: bool,
) -> dict[str, Any]:
    """Register a candidate without consumer, predictor, or logit descendants."""

    _validate_identifier("target_id", target_id)
    _validate_identifier("campaign_id", campaign_id)
    if selection_status not in PARTICLE_VIEW_TARGET_SELECTION_STATUSES:
        raise ValueError("target selection_status is invalid")
    if seed not in {101, 202, 303}:
        raise ValueError("target seed must be 101, 202, or 303")
    for name, value in (
        ("privileged_claim_eligible", privileged_claim_eligible),
        ("deployment_control_eligible", deployment_control_eligible),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
    hashes = {
        "source_manifest_sha256": source_manifest_sha256,
        "unified_split_manifest_sha256": unified_split_manifest_sha256,
        "train_split_sha256": train_split_sha256,
        "train_identity_sha256": train_identity_sha256,
        "query_tap_registration_sha256": query_tap_registration_sha256,
        "query_checkpoint_sha256": query_checkpoint_sha256,
        "memory_tap_registration_sha256": memory_tap_registration_sha256,
        "memory_checkpoint_sha256": memory_checkpoint_sha256,
        "staged_tap_reservation_sha256": staged_tap_reservation_sha256,
        "staged_tap_manifest_sha256": staged_tap_manifest_sha256,
        "staged_tap_logical_content_sha256": staged_tap_logical_content_sha256,
        "generator_checkpoint_sha256": generator_checkpoint_sha256,
    }
    for name, value in hashes.items():
        require_sha256(name, value)
    if offline_source_sha256 is not None:
        require_sha256("offline_source_sha256", offline_source_sha256)

    if generator_config.memory_source == "offline":
        if staged_tap_source_role != "offline_teacher":
            raise ValueError("offline-memory target requires offline-teacher RAM staging")
        if offline_source_sha256 is None:
            raise ValueError("offline-memory targets require an offline source hash")
        if not privileged_claim_eligible or deployment_control_eligible:
            raise ValueError("offline-memory target eligibility flags are inconsistent")
        if generator_config.self_mask_same_particle:
            raise ValueError("offline-memory target cannot self-mask HLT identity")
    else:
        if staged_tap_source_role != "hlt_memory_control":
            raise ValueError("HLT-memory control requires HLT-only RAM staging")
        if offline_source_sha256 is not None:
            raise ValueError("HLT-memory controls cannot bind an offline source")
        if privileged_claim_eligible:
            raise ValueError("HLT-memory controls are privileged-claim ineligible")
        if memory_checkpoint_sha256 != query_checkpoint_sha256:
            raise ValueError("HLT query and memory must use the exact A0 checkpoint")
        if memory_tap_registration_sha256 != query_tap_registration_sha256:
            raise ValueError("HLT query and memory must use the exact contextual tap")
        if generator_config.self_mask_same_particle:
            if selection_status != "diagnostic_nonselectable":
                raise ValueError("self-masked HLT memory must remain diagnostic")
            if deployment_control_eligible:
                raise ValueError("self-masked HLT memory is not deployment eligible")
        else:
            if selection_status != "performance_control":
                raise ValueError("self-inclusive HLT memory is a performance control")
            if not deployment_control_eligible:
                raise ValueError("self-inclusive HLT memory is deployment eligible")

    payload = with_content_hash(
        {
            "contract": PARTICLE_VIEW_TARGET_CANDIDATE_CONTRACT,
            "target_id": target_id,
            "campaign_id": campaign_id,
            "selection_status": selection_status,
            "seed": seed,
            "generator_config": generator_config.to_payload(),
            "generator_config_sha256": generator_config.content_hash,
            **hashes,
            "staged_tap_source_role": staged_tap_source_role,
            "offline_source_sha256": offline_source_sha256,
            "memory_source": generator_config.memory_source,
            "privileged_claim_eligible": privileged_claim_eligible,
            "deployment_control_eligible": deployment_control_eligible,
            "pair_feature_schema": PARTICLE_VIEW_PAIR_FEATURE_CONTRACT,
            "matching_policy": "none_matching_free_cross_attention_v1",
            "target_index_contract": PARTICLE_VIEW_TARGET_INDEX_CONTRACT,
            "consumer_dependencies": [],
            "predictor_dependencies": [],
            "target_logit_dependencies": [],
            "persistent_contextual_tap_allowed": False,
        }
    )
    return payload


def validate_target_candidate_registration(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_TARGET_CANDIDATE_CONTRACT
    )
    expected_fields = {
        "contract",
        "target_id",
        "campaign_id",
        "selection_status",
        "seed",
        "generator_config",
        "generator_config_sha256",
        "source_manifest_sha256",
        "unified_split_manifest_sha256",
        "train_split_sha256",
        "train_identity_sha256",
        "query_tap_registration_sha256",
        "query_checkpoint_sha256",
        "memory_tap_registration_sha256",
        "memory_checkpoint_sha256",
        "staged_tap_source_role",
        "staged_tap_reservation_sha256",
        "staged_tap_manifest_sha256",
        "staged_tap_logical_content_sha256",
        "generator_checkpoint_sha256",
        "offline_source_sha256",
        "memory_source",
        "privileged_claim_eligible",
        "deployment_control_eligible",
        "pair_feature_schema",
        "matching_policy",
        "target_index_contract",
        "consumer_dependencies",
        "predictor_dependencies",
        "target_logit_dependencies",
        "persistent_contextual_tap_allowed",
        "content_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("target candidate field inventory mismatch")
    config = particle_view_generator_config_from_payload(
        payload["generator_config"]
    )
    if payload["generator_config_sha256"] != config.content_hash:
        raise ValueError("target candidate generator config hash mismatch")
    rebuilt = build_target_candidate_registration(
        target_id=payload["target_id"],
        campaign_id=payload["campaign_id"],
        selection_status=payload["selection_status"],
        seed=payload["seed"],
        generator_config=config,
        source_manifest_sha256=payload["source_manifest_sha256"],
        unified_split_manifest_sha256=payload[
            "unified_split_manifest_sha256"
        ],
        train_split_sha256=payload["train_split_sha256"],
        train_identity_sha256=payload["train_identity_sha256"],
        query_tap_registration_sha256=payload[
            "query_tap_registration_sha256"
        ],
        query_checkpoint_sha256=payload["query_checkpoint_sha256"],
        memory_tap_registration_sha256=payload[
            "memory_tap_registration_sha256"
        ],
        memory_checkpoint_sha256=payload["memory_checkpoint_sha256"],
        staged_tap_source_role=payload["staged_tap_source_role"],
        staged_tap_reservation_sha256=payload[
            "staged_tap_reservation_sha256"
        ],
        staged_tap_manifest_sha256=payload["staged_tap_manifest_sha256"],
        staged_tap_logical_content_sha256=payload[
            "staged_tap_logical_content_sha256"
        ],
        generator_checkpoint_sha256=payload[
            "generator_checkpoint_sha256"
        ],
        offline_source_sha256=payload["offline_source_sha256"],
        privileged_claim_eligible=payload["privileged_claim_eligible"],
        deployment_control_eligible=payload["deployment_control_eligible"],
    )
    if rebuilt != dict(payload):
        raise ValueError("target candidate registration is not canonical")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "target_id": payload["target_id"],
        "memory_source": payload["memory_source"],
        "matching_free": True,
        "privileged_claim_eligible": payload["privileged_claim_eligible"],
        "deployment_control_eligible": payload[
            "deployment_control_eligible"
        ],
    }


__all__ = [
    "PARTICLE_VIEW_TARGET_CANDIDATE_CONTRACT",
    "PARTICLE_VIEW_TARGET_INDEX_CONTRACT",
    "PARTICLE_VIEW_TARGET_SELECTION_STATUSES",
    "build_target_candidate_registration",
    "validate_target_candidate_registration",
]

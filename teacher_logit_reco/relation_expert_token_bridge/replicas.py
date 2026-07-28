"""HLT realization identities and deterministic replica selection."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .contracts import require_sha256, validate_content_hash, with_content_hash


HLT_REPLICA_MANIFEST_CONTRACT = "retb_hlt_replica_manifest_v1"
REALIZATION_POLICIES = {
    "R_FIXED": {
        "training_replicas": [0],
        "selection": "replica_0_every_epoch",
        "domain": "nominal",
    },
    "R_MULTI": {
        "training_replicas": [0, 1, 2, 3],
        "selection": "(epoch+identity_hash_low_two_bits)%4",
        "domain": "nominal",
    },
    "R_RANDOM": {
        "training_replicas": [0, 1, 2, 3],
        "selection": "(epoch+identity_hash_low_two_bits)%4",
        "domain": "fixed_domain_randomized",
    },
}
DOMAIN_SEEDS = {
    "model_train": 3053,
    "scale_train": 3053,
    "val_stop": 3054,
    "val_design": 3054,
    "stack_val": 3056,
    "final_test": 3057,
}
RANDOM_MULTIPLIERS = {
    "0": {
        "kinematic": 1.00,
        "track_loss": 1.00,
        "track_core_noise": 1.00,
        "tail_probability": 1.00,
    },
    "1": {
        "kinematic": 0.80,
        "track_loss": 1.20,
        "track_core_noise": 1.10,
        "tail_probability": 0.75,
    },
    "2": {
        "kinematic": 1.20,
        "track_loss": 0.80,
        "track_core_noise": 0.90,
        "tail_probability": 1.25,
    },
    "3": {
        "kinematic": 1.00,
        "track_loss": 1.00,
        "track_core_noise": 1.20,
        "tail_probability": 1.50,
    },
}


def identity_hash_low_two_bits(canonical_identity: str) -> int:
    if not canonical_identity:
        raise ValueError("canonical identity must be nonempty")
    digest = hashlib.sha256(
        b"retb_replica_cycle_v1\0" + canonical_identity.encode("utf-8")
    ).digest()
    return int(digest[-1] & 0b11)


def replica_for(
    *,
    policy: str,
    logical_role: str,
    epoch: int,
    canonical_identity: str,
) -> int:
    if policy not in REALIZATION_POLICIES:
        raise ValueError(f"unknown realization policy {policy!r}")
    if logical_role not in {
        "model_train",
        "scale_train",
        "val_stop",
        "val_design",
        "stack_val",
        "final_test",
    }:
        raise ValueError(f"unknown logical role {logical_role!r}")
    if int(epoch) < 0:
        raise ValueError("epoch must be nonnegative")
    if logical_role not in {"model_train", "scale_train"}:
        return 0
    if policy == "R_FIXED":
        return 0
    return (int(epoch) + identity_hash_low_two_bits(canonical_identity)) % 4


def event_rng_seed(
    *,
    logical_role: str,
    replica_id: int,
    canonical_identity: str,
) -> int:
    if logical_role not in DOMAIN_SEEDS:
        raise ValueError(f"unknown logical role {logical_role!r}")
    if int(replica_id) not in range(4):
        raise ValueError("replica_id must be in [0,3]")
    digest = hashlib.sha256()
    digest.update(b"retb_hlt_v3_rng_v1\0")
    digest.update(str(DOMAIN_SEEDS[logical_role]).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(int(replica_id)).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_identity.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def build_hlt_replica_manifest(
    *,
    split_manifest_sha256: str,
    validation_partition_sha256: str,
    scale_train_manifest_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": HLT_REPLICA_MANIFEST_CONTRACT,
            "schema_version": 1,
            "split_manifest_sha256": require_sha256(
                split_manifest_sha256, name="split_manifest_sha256"
            ),
            "validation_partition_sha256": require_sha256(
                validation_partition_sha256,
                name="validation_partition_sha256",
            ),
            "scale_train_manifest_sha256": require_sha256(
                scale_train_manifest_sha256,
                name="scale_train_manifest_sha256",
            ),
            "profile": "HLT_V3_TRACK_DOMINANT",
            "profile_semantics_deferred_to_step": 2,
            "replica_ids": [0, 1, 2, 3],
            "training_realization_count": 4,
            "evaluation_replica_id": 0,
            "policies": REALIZATION_POLICIES,
            "domain_seeds": DOMAIN_SEEDS,
            "random_multipliers": RANDOM_MULTIPLIERS,
            "replica_cycle": {
                "formula": "(zero_based_epoch+h(identity))%4",
                "hash": (
                    "low_two_bits_sha256(retb_replica_cycle_v1||canonical_identity)"
                ),
                "resume_at_epoch_boundary_exact": True,
                "label_exposure_multiplier": 1,
            },
            "event_rng": (
                "sha256(retb_hlt_v3_rng_v1||domain_seed||replica_id||identity)"
            ),
            "random_substreams": "fixed_per_corruption_family",
            "batch_worker_shard_invariant": True,
            "model_train_scale_train_shared_identity_bytes_required": True,
        }
    )


def validate_hlt_replica_manifest(payload: Mapping[str, Any]) -> str:
    return validate_content_hash(
        payload, expected_contract=HLT_REPLICA_MANIFEST_CONTRACT
    )


__all__ = [
    "DOMAIN_SEEDS",
    "HLT_REPLICA_MANIFEST_CONTRACT",
    "RANDOM_MULTIPLIERS",
    "REALIZATION_POLICIES",
    "build_hlt_replica_manifest",
    "event_rng_seed",
    "identity_hash_low_two_bits",
    "replica_for",
    "validate_hlt_replica_manifest",
]

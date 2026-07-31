"""Immutable HOSD target-mean and shuffle-control construction."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contracts import (
    TARGET_CONTROL_MANIFEST_CONTRACT,
    TARGET_SHUFFLE_PLAN_CONTRACT,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .extractors import TargetBatch
from .normalization import target_mean_values
from .target_cache import LoadedTargetCache, identity_order_sha256


SHUFFLE_KINDS = ("global", "within_class")


def _rank(
    identity: str, *, shuffle_kind: str, target_id: str, split: str
) -> tuple[str, str]:
    digest = hashlib.sha256()
    for item in (
        "hosd_target_shuffle_v1",
        shuffle_kind,
        target_id,
        split,
        identity,
    ):
        encoded = item.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest(), identity


def build_target_shuffle_plan(
    identities: Sequence[str],
    *,
    labels: Sequence[int],
    target_id: str,
    split: str,
    shuffle_kind: str,
    label_manifest_sha256: str,
    canonical_cache_manifest_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic donor map in a label-capable auditor only."""

    if shuffle_kind not in SHUFFLE_KINDS:
        raise ValueError(f"unknown shuffle kind {shuffle_kind!r}")
    require_sha256(label_manifest_sha256, name="label_manifest_sha256")
    require_sha256(
        canonical_cache_manifest_sha256, name="canonical_cache_manifest_sha256"
    )
    identity_values = tuple(str(value) for value in identities)
    label_values = np.asarray(labels, dtype=np.int64)
    if label_values.shape != (len(identity_values),):
        raise ValueError("shuffle labels do not match identity population")
    if len(set(identity_values)) != len(identity_values):
        raise ValueError("shuffle identities contain duplicates")
    mapping = np.arange(len(identity_values), dtype=np.int64)
    groups = (
        [np.arange(len(identity_values), dtype=np.int64)]
        if shuffle_kind == "global"
        else [
            np.flatnonzero(label_values == label)
            for label in sorted(set(label_values.tolist()))
        ]
    )
    singleton_count = 0
    group_records = []
    for group in groups:
        ordered = sorted(
            (int(index) for index in group),
            key=lambda index: _rank(
                identity_values[index],
                shuffle_kind=shuffle_kind,
                target_id=target_id,
                split=split,
            ),
        )
        if len(ordered) == 1:
            singleton_count += 1
        for position, recipient in enumerate(ordered):
            mapping[recipient] = ordered[(position + 1) % len(ordered)]
        group_records.append(
            {
                "event_count": len(ordered),
                "recipient_index_sha256": hashlib.sha256(
                    np.asarray(sorted(ordered), dtype=np.int64).tobytes()
                ).hexdigest(),
                "permutation_sha256": hashlib.sha256(
                    np.asarray([mapping[index] for index in sorted(ordered)], dtype=np.int64)
                    .tobytes()
                ).hexdigest(),
            }
        )
    if sorted(mapping.tolist()) != list(range(len(identity_values))):
        raise AssertionError("shuffle donor mapping is not a permutation")
    if shuffle_kind == "within_class" and not np.array_equal(
        label_values, label_values[mapping]
    ):
        raise AssertionError("within-class shuffle changed class marginals")
    return with_content_hash(
        {
            "contract": TARGET_SHUFFLE_PLAN_CONTRACT,
            "schema_version": 1,
            "shuffle_kind": shuffle_kind,
            "target_id": target_id,
            "split": split,
            "event_count": len(identity_values),
            "canonical_identity_order_sha256": identity_order_sha256(identity_values),
            "canonical_cache_manifest_sha256": canonical_cache_manifest_sha256,
            "label_manifest_sha256": label_manifest_sha256,
            "mapping_recipient_to_donor": mapping.tolist(),
            "mapping_sha256": hashlib.sha256(mapping.tobytes()).hexdigest(),
            "construction_rule": (
                "sort_sha256(hosd_target_shuffle_v1||kind||target||split||identity)"
                "_then_rotate_left_one"
            ),
            "groups": group_records,
            "singleton_self_map_count": singleton_count,
            "labels_stored": False,
            "target_values_stored": False,
            "canonical_cache_mutated": False,
            "source": dict(source),
        }
    )


def apply_target_shuffle(
    values: np.ndarray,
    masks: np.ndarray,
    *,
    plan: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    validate_content_hash(plan, expected_contract=TARGET_SHUFFLE_PLAN_CONTRACT)
    values = np.asarray(values, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    mapping = np.asarray(plan["mapping_recipient_to_donor"], dtype=np.int64)
    if values.shape != masks.shape or values.shape[0] != mapping.size:
        raise ValueError("shuffle plan population differs from target arrays")
    return values[mapping].copy(), masks[mapping].copy()


def build_control_batches(
    canonical: LoadedTargetCache,
    *,
    normalizer: Mapping[str, Any],
    control_kind: str,
    shuffle_plans: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, TargetBatch]:
    if control_kind not in {"target_mean", "global_shuffle", "within_class_shuffle"}:
        raise ValueError(f"unknown target control {control_kind!r}")
    output = {}
    for target_id in canonical.manifest["persisted_target_ids"]:
        masks = canonical.masks[target_id]
        if control_kind == "target_mean":
            values = target_mean_values(
                masks, target_id=target_id, normalizer=normalizer
            )
            control_masks = masks.copy()
            plan_hash = None
        else:
            if shuffle_plans is None or target_id not in shuffle_plans:
                raise ValueError(f"missing shuffle plan for {target_id}")
            values, control_masks = apply_target_shuffle(
                canonical.values[target_id],
                masks,
                plan=shuffle_plans[target_id],
            )
            plan_hash = shuffle_plans[target_id]["content_hash"]
        output[target_id] = TargetBatch(
            target_id=target_id,
            component_names=tuple(
                canonical.manifest["target_components"][target_id]
            ),
            availability_groups=("control_target_available",),
            values=torch.from_numpy(values),
            loss_mask=torch.from_numpy(control_masks),
            diagnostics={
                "control_kind": control_kind,
                "canonical_cache_manifest_sha256": canonical.manifest["content_hash"],
                "shuffle_plan_sha256": plan_hash,
                "canonical_cache_mutated": False,
            },
        )
    return output


def build_target_control_manifest(
    *,
    control_kind: str,
    canonical_cache_manifest_sha256: str,
    control_cache_manifest_sha256: str,
    target_normalizer_sha256: str,
    shuffle_plan_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if control_kind not in {"target_mean", "global_shuffle", "within_class_shuffle"}:
        raise ValueError("unknown target control kind")
    for name, value in {
        "canonical_cache_manifest_sha256": canonical_cache_manifest_sha256,
        "control_cache_manifest_sha256": control_cache_manifest_sha256,
        "target_normalizer_sha256": target_normalizer_sha256,
    }.items():
        require_sha256(value, name=name)
    return with_content_hash(
        {
            "contract": TARGET_CONTROL_MANIFEST_CONTRACT,
            "schema_version": 1,
            "control_kind": control_kind,
            "canonical_cache_manifest_sha256": canonical_cache_manifest_sha256,
            "control_cache_manifest_sha256": control_cache_manifest_sha256,
            "target_normalizer_sha256": target_normalizer_sha256,
            "shuffle_plan_hashes": {
                target_id: require_sha256(value, name=f"shuffle_plan.{target_id}")
                for target_id, value in sorted(shuffle_plan_hashes.items())
            },
            "canonical_cache_mutated": False,
            "artifact_identity_distinct_from_canonical": (
                control_cache_manifest_sha256 != canonical_cache_manifest_sha256
            ),
            "source": dict(source),
        }
    )


__all__ = [
    "SHUFFLE_KINDS",
    "apply_target_shuffle",
    "build_control_batches",
    "build_target_control_manifest",
    "build_target_shuffle_plan",
]

"""Step 1 split and validation-access contracts for the bridge pilot."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.jetclass_data import (
    LABEL_NAMES,
    SPLIT_ORDER,
    JetIdentity,
    SplitManifest,
    audit_split_manifest,
    manifest_hash,
)

from .bridge_contracts import (
    canonical_json_bytes,
    canonical_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


PREDICTION_ANCHORED_SPLIT_CONTRACT = "prediction_anchored_child_splits_v1"
PREDICTION_ANCHORED_CHILD_SPLIT_CONTRACT = "prediction_anchored_child_split_v1"
PREDICTION_ANCHORED_VALIDATION_UNLOCK_CONTRACT = (
    "prediction_anchored_validation_unlock_v1"
)
PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT = (
    "prediction_anchored_split_access_receipt_v1"
)


@dataclass(frozen=True)
class ChildSplitSpec:
    name: str
    count: int
    purpose: str
    seal_kind: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": int(self.count),
            "purpose": self.purpose,
            "seal_kind": self.seal_kind,
        }


@dataclass(frozen=True)
class ParentPartitionSpec:
    parent_split: str
    seed: int
    children: tuple[ChildSplitSpec, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "parent_split": self.parent_split,
            "seed": int(self.seed),
            "children": [child.to_payload() for child in self.children],
        }


def _default_parent_counts() -> tuple[tuple[str, int], ...]:
    return (
        ("model_train", 500_000),
        ("model_val", 150_000),
        ("stack_train", 500_000),
        ("stack_val", 150_000),
        ("final_test", 150_000),
    )


def _default_partitions() -> tuple[ParentPartitionSpec, ...]:
    return (
        ParentPartitionSpec(
            parent_split="stack_train",
            seed=810_101,
            children=(
                ChildSplitSpec(
                    "stack_train_consumer", 250_000, "consumer_training"
                ),
                ChildSplitSpec(
                    "stack_train_distill", 250_000, "reconstructor_training"
                ),
            ),
        ),
        ParentPartitionSpec(
            parent_split="model_val",
            seed=810_202,
            children=(
                ChildSplitSpec("model_val_stop", 75_000, "checkpoint_selection"),
                ChildSplitSpec(
                    "model_val_select", 75_000, "configuration_selection"
                ),
            ),
        ),
        ParentPartitionSpec(
            parent_split="stack_val",
            seed=810_303,
            children=(
                ChildSplitSpec(
                    "stack_val_consumer",
                    75_000,
                    "consumer_confirmation",
                    "consumer_preconfirmation",
                ),
                ChildSplitSpec(
                    "stack_val_deploy",
                    75_000,
                    "deployable_confirmation",
                    "deployable_preconfirmation",
                ),
            ),
        ),
    )


def _high_data_3m_parent_counts() -> tuple[tuple[str, int], ...]:
    """Locked parent inventory for the storage-safe high-data campaign.

    The requested 6M labeled training pool remains the 3M/3M ``stack_train``
    partition.  ``model_train`` is a separate 500k parent because R0 must not
    learn from either the consumer or distillation examples.
    """

    return (
        ("model_train", 500_000),
        ("model_val", 500_000),
        ("stack_train", 6_000_000),
        ("stack_val", 500_000),
        ("final_test", 1_000_000),
    )


def _high_data_3m_partitions() -> tuple[ParentPartitionSpec, ...]:
    return (
        ParentPartitionSpec(
            parent_split="stack_train",
            seed=8_100_101,
            children=(
                ChildSplitSpec(
                    "stack_train_consumer", 3_000_000, "consumer_training"
                ),
                ChildSplitSpec(
                    "stack_train_distill", 3_000_000, "reconstructor_training"
                ),
            ),
        ),
        ParentPartitionSpec(
            parent_split="model_val",
            seed=8_100_202,
            children=(
                ChildSplitSpec("model_val_stop", 250_000, "checkpoint_selection"),
                ChildSplitSpec(
                    "model_val_select", 250_000, "configuration_selection"
                ),
            ),
        ),
        ParentPartitionSpec(
            parent_split="stack_val",
            seed=8_100_303,
            children=(
                ChildSplitSpec(
                    "stack_val_consumer",
                    250_000,
                    "consumer_confirmation",
                    "consumer_preconfirmation",
                ),
                ChildSplitSpec(
                    "stack_val_deploy",
                    250_000,
                    "deployable_confirmation",
                    "deployable_preconfirmation",
                ),
            ),
        ),
    )


@dataclass(frozen=True)
class PredictionAnchoredSplitConfig:
    """Versioned, hashable split configuration.

    Tests may provide a miniature count-preserving configuration.  Production
    callers use the locked defaults below; the serialized config hash prevents
    a miniature or stale contract from entering a real campaign.
    """

    contract: str = "prediction_anchored_split_config_v1"
    class_names: tuple[str, ...] = tuple(LABEL_NAMES)
    parent_split_counts: tuple[tuple[str, int], ...] = field(
        default_factory=_default_parent_counts
    )
    partitions: tuple[ParentPartitionSpec, ...] = field(
        default_factory=_default_partitions
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "class_names": list(self.class_names),
            "parent_split_counts": {
                name: int(count) for name, count in self.parent_split_counts
            },
            "partitions": [partition.to_payload() for partition in self.partitions],
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


LOCKED_PILOT_SPLIT_CONFIG = PredictionAnchoredSplitConfig()
LOCKED_HIGH_DATA_3M_SPLIT_CONFIG = PredictionAnchoredSplitConfig(
    contract="prediction_anchored_high_data_3m_split_config_v1",
    parent_split_counts=_high_data_3m_parent_counts(),
    partitions=_high_data_3m_partitions(),
)


def prediction_anchored_split_config(profile: str) -> PredictionAnchoredSplitConfig:
    profiles = {
        "pilot_250k": LOCKED_PILOT_SPLIT_CONFIG,
        "high_data_3m": LOCKED_HIGH_DATA_3M_SPLIT_CONFIG,
    }
    try:
        return profiles[str(profile)]
    except KeyError as exc:
        raise ValueError(
            f"unknown prediction-anchored split profile {profile!r}; "
            f"expected one of {sorted(profiles)}"
        ) from exc


def prediction_anchored_split_config_from_payload(
    payload: Mapping[str, Any],
) -> PredictionAnchoredSplitConfig:
    """Reconstruct and validate the exact split config embedded in a manifest."""

    if not isinstance(payload, Mapping):
        raise ValueError("split config payload must be an object")
    raw_counts = payload.get("parent_split_counts")
    raw_partitions = payload.get("partitions")
    raw_classes = payload.get("class_names")
    if not isinstance(raw_counts, Mapping) or not isinstance(raw_partitions, list):
        raise ValueError("split config payload is missing counts or partitions")
    if not isinstance(raw_classes, list):
        raise ValueError("split config payload is missing class_names")
    partitions: list[ParentPartitionSpec] = []
    for raw_partition in raw_partitions:
        if not isinstance(raw_partition, Mapping) or not isinstance(
            raw_partition.get("children"), list
        ):
            raise ValueError("invalid split partition payload")
        children = tuple(
            ChildSplitSpec(
                name=str(child["name"]),
                count=int(child["count"]),
                purpose=str(child["purpose"]),
                seal_kind=(
                    None if child.get("seal_kind") is None else str(child["seal_kind"])
                ),
            )
            for child in raw_partition["children"]
        )
        partitions.append(
            ParentPartitionSpec(
                parent_split=str(raw_partition["parent_split"]),
                seed=int(raw_partition["seed"]),
                children=children,
            )
        )
    config = PredictionAnchoredSplitConfig(
        contract=str(payload.get("contract", "")),
        class_names=tuple(str(value) for value in raw_classes),
        parent_split_counts=tuple(
            (name, int(raw_counts[name])) for name in SPLIT_ORDER
        ),
        partitions=tuple(partitions),
    )
    _require_split_config(config)
    if config.to_payload() != dict(payload):
        raise ValueError("split config payload is not canonical")
    return config


@dataclass(frozen=True)
class SplitAccessRule:
    purpose: str
    seal_kind: str | None = None
    one_shot: bool = False


SPLIT_ACCESS_RULES: dict[str, SplitAccessRule] = {
    "model_train": SplitAccessRule("r0_training"),
    "stack_train_consumer": SplitAccessRule("consumer_training"),
    "stack_train_distill": SplitAccessRule("reconstructor_training"),
    "model_val_stop": SplitAccessRule("checkpoint_selection"),
    "model_val_select": SplitAccessRule("configuration_selection"),
    "stack_val_consumer": SplitAccessRule(
        "consumer_confirmation", "consumer_preconfirmation", True
    ),
    "stack_val_deploy": SplitAccessRule(
        "deployable_confirmation", "deployable_preconfirmation", True
    ),
    "final_test": SplitAccessRule("final_evaluation", "locked_deployable", True),
}


def _identity_digest(identities: Sequence[JetIdentity]) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        digest.update(canonical_json_bytes(identity.to_dict()))
        digest.update(b"\n")
    return digest.hexdigest()


def _rank_key(seed: int, label: int, identity: JetIdentity, *, domain: str) -> tuple[str, str]:
    key = identity.key()
    encoded = f"{domain}\0{int(seed)}\0{int(label)}\0{key}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), key


def _require_split_config(config: PredictionAnchoredSplitConfig) -> None:
    parent_counts = dict(config.parent_split_counts)
    if tuple(parent_counts) != tuple(SPLIT_ORDER):
        raise ValueError(
            "split config parent counts must use the canonical five-way split order"
        )
    if tuple(config.class_names) != tuple(LABEL_NAMES):
        raise ValueError("split config must preserve the canonical ten-class ordering")
    child_names: set[str] = set()
    partition_parents: set[str] = set()
    n_classes = len(config.class_names)
    for partition in config.partitions:
        if partition.parent_split in partition_parents:
            raise ValueError(f"duplicate child partition parent: {partition.parent_split}")
        partition_parents.add(partition.parent_split)
        if partition.parent_split not in parent_counts:
            raise ValueError(f"unknown child partition parent: {partition.parent_split}")
        if sum(child.count for child in partition.children) != parent_counts[partition.parent_split]:
            raise ValueError(
                f"children do not cover parent split {partition.parent_split} exactly"
            )
        for child in partition.children:
            if child.name in child_names or child.name in parent_counts:
                raise ValueError(f"duplicate or ambiguous child split name: {child.name}")
            child_names.add(child.name)
            if child.count <= 0 or child.count % n_classes:
                raise ValueError(
                    f"child split {child.name} count must be positive and class divisible"
                )
            rule = SPLIT_ACCESS_RULES.get(child.name)
            if rule is None or rule.purpose != child.purpose or rule.seal_kind != child.seal_kind:
                raise ValueError(f"access policy mismatch for child split {child.name}")


def _audit_parent_manifest(
    parent: SplitManifest,
    config: PredictionAnchoredSplitConfig,
) -> dict[str, Any]:
    _require_split_config(config)
    if tuple(parent.class_names) != tuple(config.class_names):
        raise ValueError("parent manifest class ordering does not match the split contract")
    expected_counts = dict(config.parent_split_counts)
    for split in SPLIT_ORDER:
        actual = len(parent.splits.get(split, ()))
        declared = int(parent.split_sizes.get(split, -1))
        expected = int(expected_counts[split])
        if actual != expected or declared != expected:
            raise ValueError(
                f"parent split {split} count mismatch: "
                f"actual={actual}, declared={declared}, expected={expected}"
            )
        counts = [0] * len(config.class_names)
        for identity in parent.splits[split]:
            if identity.label < 0 or identity.label >= len(counts):
                raise ValueError(f"invalid label in parent split {split}")
            counts[identity.label] += 1
        if len(set(counts)) != 1:
            raise ValueError(f"parent split {split} is not exactly class balanced: {counts}")
    base_audit = audit_split_manifest(parent)
    if not base_audit.get("ok"):
        raise ValueError(f"parent split manifest failed its base audit: {base_audit}")
    return base_audit


def build_child_split_manifest(
    parent: SplitManifest,
    *,
    config: PredictionAnchoredSplitConfig = LOCKED_PILOT_SPLIT_CONFIG,
) -> dict[str, Any]:
    """Create deterministic, class-stratified child membership by parent index."""

    _audit_parent_manifest(parent, config)
    parent_sha256 = manifest_hash(parent)
    n_classes = len(config.class_names)
    parent_order_hashes = {
        split: _identity_digest(parent.splits[split]) for split in SPLIT_ORDER
    }
    children: dict[str, dict[str, Any]] = {}
    partition_audits: list[dict[str, Any]] = []

    for partition in config.partitions:
        rows = parent.splits[partition.parent_split]
        by_label: list[list[int]] = [[] for _ in range(n_classes)]
        for index, identity in enumerate(rows):
            by_label[identity.label].append(index)
        for label, indices in enumerate(by_label):
            indices.sort(
                key=lambda index: _rank_key(
                    partition.seed,
                    label,
                    rows[index],
                    domain="membership",
                )
            )

        offsets = [0] * n_classes
        partition_indices: list[int] = []
        for child in partition.children:
            per_class = child.count // n_classes
            selected: list[int] = []
            for label in range(n_classes):
                start = offsets[label]
                stop = start + per_class
                selected.extend(by_label[label][start:stop])
                offsets[label] = stop
            selected.sort(
                key=lambda index: _rank_key(
                    partition.seed,
                    rows[index].label,
                    rows[index],
                    domain=f"row-order:{child.name}",
                )
            )
            selected_identities = [rows[index] for index in selected]
            class_counts = {
                config.class_names[label]: sum(
                    identity.label == label for identity in selected_identities
                )
                for label in range(n_classes)
            }
            child_payload = with_content_hash(
                {
                    "contract": PREDICTION_ANCHORED_CHILD_SPLIT_CONTRACT,
                    "name": child.name,
                    "parent_split": partition.parent_split,
                    "parent_manifest_sha256": parent_sha256,
                    "parent_split_order_sha256": parent_order_hashes[
                        partition.parent_split
                    ],
                    "split_config_sha256": config.content_hash,
                    "partition_seed": int(partition.seed),
                    "purpose": child.purpose,
                    "seal_kind": child.seal_kind,
                    "count": len(selected),
                    "class_counts": class_counts,
                    "parent_row_indices": selected,
                    "ordered_identity_sha256": _identity_digest(selected_identities),
                }
            )
            children[child.name] = child_payload
            partition_indices.extend(selected)

        partition_audits.append(
            {
                "parent_split": partition.parent_split,
                "parent_count": len(rows),
                "child_count": len(partition_indices),
                "coverage_complete": sorted(partition_indices) == list(range(len(rows))),
                "overlap_count": len(partition_indices) - len(set(partition_indices)),
            }
        )

    payload = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_SPLIT_CONTRACT,
            "split_config": config.to_payload(),
            "split_config_sha256": config.content_hash,
            "parent_manifest_sha256": parent_sha256,
            "parent_split_order_sha256": parent_order_hashes,
            "children": children,
            "partition_audits": partition_audits,
        }
    )
    audit_child_split_manifest(payload, parent=parent, config=config)
    return payload


def audit_child_split_manifest(
    payload: Mapping[str, Any],
    *,
    parent: SplitManifest,
    config: PredictionAnchoredSplitConfig = LOCKED_PILOT_SPLIT_CONFIG,
) -> dict[str, Any]:
    """Fail closed on tampering, parent reordering, overlap, or stale lineage."""

    validate_content_hash(payload, expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT)
    _audit_parent_manifest(parent, config)
    if payload.get("split_config") != config.to_payload():
        raise ValueError("child manifest split config payload mismatch")
    if payload.get("split_config_sha256") != config.content_hash:
        raise ValueError("child manifest split config hash mismatch")
    current_parent_hash = manifest_hash(parent)
    if payload.get("parent_manifest_sha256") != current_parent_hash:
        raise ValueError("child manifest is bound to a stale or reordered parent manifest")

    expected_specs = {
        child.name: (partition, child)
        for partition in config.partitions
        for child in partition.children
    }
    raw_children = payload.get("children")
    if not isinstance(raw_children, Mapping) or set(raw_children) != set(expected_specs):
        raise ValueError("child manifest has missing or unexpected child IDs")

    summaries: dict[str, Any] = {}
    for name, (partition, spec) in expected_specs.items():
        child = raw_children[name]
        if not isinstance(child, Mapping):
            raise ValueError(f"child split {name} is not an object")
        validate_content_hash(
            child, expected_contract=PREDICTION_ANCHORED_CHILD_SPLIT_CONTRACT
        )
        if child.get("name") != name or child.get("parent_split") != partition.parent_split:
            raise ValueError(f"child split identity mismatch for {name}")
        if child.get("purpose") != spec.purpose or child.get("seal_kind") != spec.seal_kind:
            raise ValueError(f"child split access contract mismatch for {name}")
        if child.get("parent_manifest_sha256") != current_parent_hash:
            raise ValueError(f"child split {name} has stale parent provenance")
        rows = parent.splits[partition.parent_split]
        expected_parent_order_hash = _identity_digest(rows)
        if child.get("parent_split_order_sha256") != expected_parent_order_hash:
            raise ValueError(f"child split {name} parent ordering changed")
        indices = child.get("parent_row_indices")
        if not isinstance(indices, list) or any(
            not isinstance(index, int) or isinstance(index, bool) for index in indices
        ):
            raise ValueError(f"child split {name} has invalid row indices")
        if len(indices) != spec.count or int(child.get("count", -1)) != spec.count:
            raise ValueError(f"child split {name} count mismatch")
        if len(set(indices)) != len(indices):
            raise ValueError(f"child split {name} contains duplicate parent rows")
        if any(index < 0 or index >= len(rows) for index in indices):
            raise ValueError(f"child split {name} contains an out-of-range parent row")
        identities = [rows[index] for index in indices]
        if child.get("ordered_identity_sha256") != _identity_digest(identities):
            raise ValueError(f"child split {name} identity ordering mismatch")
        actual_class_counts = {
            config.class_names[label]: sum(identity.label == label for identity in identities)
            for label in range(len(config.class_names))
        }
        expected_per_class = spec.count // len(config.class_names)
        if child.get("class_counts") != actual_class_counts or set(
            actual_class_counts.values()
        ) != {expected_per_class}:
            raise ValueError(f"child split {name} is not exactly class stratified")
        summaries[name] = {
            "content_hash": child["content_hash"],
            "count": spec.count,
            "class_counts": actual_class_counts,
            "parent_split": partition.parent_split,
            "purpose": spec.purpose,
            "seal_kind": spec.seal_kind,
        }

    for partition in config.partitions:
        combined = [
            index
            for spec in partition.children
            for index in raw_children[spec.name]["parent_row_indices"]
        ]
        expected = list(range(len(parent.splits[partition.parent_split])))
        if len(combined) != len(set(combined)):
            raise ValueError(f"child splits overlap within {partition.parent_split}")
        if sorted(combined) != expected:
            raise ValueError(f"child splits do not cover {partition.parent_split}")

    return {
        "ok": True,
        "contract": PREDICTION_ANCHORED_SPLIT_CONTRACT,
        "content_hash": payload["content_hash"],
        "parent_manifest_sha256": current_parent_hash,
        "children": summaries,
        "coverage_complete": True,
        "overlap_count": 0,
    }


def child_split_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_hash(payload, expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT)
    return {
        "content_hash": payload["content_hash"],
        "parent_manifest_sha256": payload["parent_manifest_sha256"],
        "split_config_sha256": payload["split_config_sha256"],
        "children": {
            name: {
                "content_hash": child["content_hash"],
                "parent_split": child["parent_split"],
                "count": child["count"],
                "class_counts": child["class_counts"],
                "purpose": child["purpose"],
                "seal_kind": child["seal_kind"],
            }
            for name, child in sorted(payload["children"].items())
        },
    }


def build_validation_unlock(
    *,
    split_name: str,
    parent_manifest_sha256: str,
    bound_split_sha256: str,
    selection_sha256: str,
) -> dict[str, Any]:
    rule = SPLIT_ACCESS_RULES.get(split_name)
    if rule is None or rule.seal_kind is None:
        raise ValueError(f"split {split_name!r} does not use a validation unlock")
    for name, value in (
        ("parent_manifest_sha256", parent_manifest_sha256),
        ("bound_split_sha256", bound_split_sha256),
        ("selection_sha256", selection_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a SHA-256 hex digest")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_VALIDATION_UNLOCK_CONTRACT,
            "status": "LOCKED",
            "seal_kind": rule.seal_kind,
            "target_split": split_name,
            "parent_manifest_sha256": parent_manifest_sha256,
            "bound_split_sha256": bound_split_sha256,
            "selection_sha256": selection_sha256,
        }
    )


def authorize_split_access(
    *,
    split_name: str,
    purpose: str,
    parent_manifest_sha256: str,
    bound_split_sha256: str,
    unlock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rule = SPLIT_ACCESS_RULES.get(split_name)
    if rule is None:
        raise PermissionError(f"split {split_name!r} is not authorized by this campaign")
    if purpose != rule.purpose:
        raise PermissionError(
            f"split {split_name!r} permits purpose {rule.purpose!r}, not {purpose!r}"
        )
    unlock_hash: str | None = None
    selection_hash: str | None = None
    if rule.seal_kind is not None:
        if unlock is None:
            raise PermissionError(f"split {split_name!r} is sealed until selection is locked")
        validate_content_hash(
            unlock, expected_contract=PREDICTION_ANCHORED_VALIDATION_UNLOCK_CONTRACT
        )
        expected = {
            "status": "LOCKED",
            "seal_kind": rule.seal_kind,
            "target_split": split_name,
            "parent_manifest_sha256": parent_manifest_sha256,
            "bound_split_sha256": bound_split_sha256,
        }
        for name, value in expected.items():
            if unlock.get(name) != value:
                raise PermissionError(
                    f"validation unlock field {name!r} does not authorize {split_name!r}"
                )
        unlock_hash = str(unlock["content_hash"])
        selection_hash = str(unlock["selection_sha256"])
    elif unlock is not None:
        raise PermissionError(f"unsealed split {split_name!r} must not receive an unlock")

    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT,
            "status": "AUTHORIZED",
            "split_name": split_name,
            "purpose": purpose,
            "parent_manifest_sha256": parent_manifest_sha256,
            "bound_split_sha256": bound_split_sha256,
            "seal_kind": rule.seal_kind,
            "one_shot": rule.one_shot,
            "unlock_sha256": unlock_hash,
            "selection_sha256": selection_hash,
        }
    )


def split_binding(
    child_manifest: Mapping[str, Any],
    split_name: str,
) -> tuple[str, str]:
    """Resolve parent and split hashes from a verified child-manifest contract."""

    validate_content_hash(
        child_manifest, expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    parent_hash = str(child_manifest["parent_manifest_sha256"])
    children = child_manifest["children"]
    if split_name in children:
        child = children[split_name]
        validate_content_hash(
            child, expected_contract=PREDICTION_ANCHORED_CHILD_SPLIT_CONTRACT
        )
        return parent_hash, str(child["content_hash"])
    parent_order_hashes = child_manifest["parent_split_order_sha256"]
    if split_name in {"model_train", "final_test"}:
        return parent_hash, str(parent_order_hashes[split_name])
    raise KeyError(f"split {split_name!r} has no campaign binding")


def build_manifest_validation_unlock(
    child_manifest: Mapping[str, Any],
    *,
    split_name: str,
    selection_sha256: str,
) -> dict[str, Any]:
    parent_hash, bound_hash = split_binding(child_manifest, split_name)
    return build_validation_unlock(
        split_name=split_name,
        parent_manifest_sha256=parent_hash,
        bound_split_sha256=bound_hash,
        selection_sha256=selection_sha256,
    )


def authorize_manifest_split_access(
    child_manifest: Mapping[str, Any],
    *,
    parent: SplitManifest,
    split_name: str,
    purpose: str,
    unlock: Mapping[str, Any] | None = None,
    config: PredictionAnchoredSplitConfig = LOCKED_PILOT_SPLIT_CONFIG,
) -> dict[str, Any]:
    """Authorize access only after validating the current parent and child lineage."""

    audit_child_split_manifest(child_manifest, parent=parent, config=config)
    parent_hash, bound_hash = split_binding(child_manifest, split_name)
    return authorize_split_access(
        split_name=split_name,
        purpose=purpose,
        parent_manifest_sha256=parent_hash,
        bound_split_sha256=bound_hash,
        unlock=unlock,
    )


def claim_split_access(
    receipt_path: str | Path,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist an authorization receipt; sealed one-shot claims cannot repeat."""

    validate_content_hash(
        authorization, expected_contract=PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT
    )
    destination = Path(receipt_path)
    if bool(authorization.get("one_shot")) and destination.exists():
        raise PermissionError(
            f"one-shot split access was already claimed: {authorization.get('split_name')}"
        )
    receipt = write_immutable_json(destination, authorization)
    if bool(authorization.get("one_shot")) and receipt["status"] != "published":
        raise PermissionError(
            f"one-shot split access was already claimed: {authorization.get('split_name')}"
        )
    return receipt


__all__ = [
    "ChildSplitSpec",
    "LOCKED_HIGH_DATA_3M_SPLIT_CONFIG",
    "LOCKED_PILOT_SPLIT_CONFIG",
    "PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT",
    "PREDICTION_ANCHORED_CHILD_SPLIT_CONTRACT",
    "PREDICTION_ANCHORED_SPLIT_CONTRACT",
    "PREDICTION_ANCHORED_VALIDATION_UNLOCK_CONTRACT",
    "ParentPartitionSpec",
    "PredictionAnchoredSplitConfig",
    "SPLIT_ACCESS_RULES",
    "SplitAccessRule",
    "audit_child_split_manifest",
    "authorize_manifest_split_access",
    "authorize_split_access",
    "build_child_split_manifest",
    "build_manifest_validation_unlock",
    "build_validation_unlock",
    "child_split_summary",
    "claim_split_access",
    "prediction_anchored_split_config",
    "split_binding",
]

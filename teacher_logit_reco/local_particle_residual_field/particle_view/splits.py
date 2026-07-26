"""Unified single-training-pool split contract for particle-view campaigns."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.jetclass_data import (
    LABEL_NAMES,
    MAX_CONSTITUENTS,
    SPLIT_ORDER,
    JetIdentity,
    SplitManifest,
    audit_split_manifest,
    manifest_hash,
)

from .contracts import (
    canonical_json_bytes,
    canonical_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


PARTICLE_VIEW_SPLIT_CONFIG_CONTRACT = "particle_view_split_config_v1"
PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT = "particle_view_unified_splits_v1"
PARTICLE_VIEW_LOGICAL_SPLIT_CONTRACT = "particle_view_logical_split_v1"
PARTICLE_VIEW_STEP1_REPORT_CONTRACT = "particle_view_step1_report_v1"
PARTICLE_VIEW_TRAINING_TOPOLOGY = "single_pool_no_crossfit_v1"
PARTICLE_VIEW_MODEL_VAL_PARTITION_SEED = 9_120_202

PARTICLE_VIEW_LOGICAL_SPLITS = (
    "train",
    "model_val_stop",
    "model_val_select",
    "stack_val",
    "final_test",
)
PARTICLE_VIEW_FORBIDDEN_TRAINING_SPLIT_FRAGMENTS = (
    "consumer",
    "distill",
    "fold",
    "crossfit",
    "cross_fit",
)
PARTICLE_VIEW_TRAINABLE_COMPONENTS = (
    "A0_view",
    "A0_view_long_deploy",
    "A0_view_total_label_budget",
    "Toff_view",
    "Gview",
    "Cview_discovery",
    "Cview_probe",
    "Pview_probe",
    "Cview_clean",
    "Pview_0",
    "Cview_robust",
    "Pview_final_objectives",
    "equal_capacity_direct_controls",
)

_LOGICAL_PURPOSES: dict[str, tuple[str, str | None]] = {
    "train": ("all_model_fitting", None),
    "model_val_stop": ("checkpoint_selection", None),
    "model_val_select": ("configuration_selection", None),
    "stack_val": ("sealed_confirmation_and_fusion", "winner_selection_locked"),
    "final_test": ("hlt_only_final_evaluation", "final_test_authorized"),
}


@dataclass(frozen=True)
class ParticleViewSplitConfig:
    """Versioned inventory and deterministic validation-partition recipe."""

    contract: str = PARTICLE_VIEW_SPLIT_CONFIG_CONTRACT
    class_names: tuple[str, ...] = tuple(LABEL_NAMES)
    max_particles: int = MAX_CONSTITUENTS
    train_parent_split: str = "model_train"
    train_count: int = 500_000
    model_val_parent_split: str = "model_val"
    model_val_count: int = 150_000
    stack_val_parent_split: str = "stack_val"
    stack_val_count: int = 150_000
    final_test_parent_split: str = "final_test"
    final_test_count: int = 150_000
    unused_parent_splits: tuple[str, ...] = ("stack_train",)
    unused_parent_split_counts: tuple[tuple[str, int], ...] = (
        ("stack_train", 500_000),
    )
    model_val_partition_seed: int = PARTICLE_VIEW_MODEL_VAL_PARTITION_SEED

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "class_names": list(self.class_names),
            "max_particles": int(self.max_particles),
            "source_mapping": {
                "train": self.train_parent_split,
                "model_val_stop": self.model_val_parent_split,
                "model_val_select": self.model_val_parent_split,
                "stack_val": self.stack_val_parent_split,
                "final_test": self.final_test_parent_split,
            },
            "source_counts": {
                self.train_parent_split: int(self.train_count),
                self.model_val_parent_split: int(self.model_val_count),
                self.stack_val_parent_split: int(self.stack_val_count),
                self.final_test_parent_split: int(self.final_test_count),
                **{
                    name: int(count)
                    for name, count in self.unused_parent_split_counts
                },
            },
            "unused_parent_splits": list(self.unused_parent_splits),
            "model_val_partition_seed": int(self.model_val_partition_seed),
            "logical_split_order": list(PARTICLE_VIEW_LOGICAL_SPLITS),
            "training_topology": PARTICLE_VIEW_TRAINING_TOPOLOGY,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG = ParticleViewSplitConfig()


def _identity_digest(identities: Sequence[JetIdentity]) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        digest.update(canonical_json_bytes(identity.to_dict()))
        digest.update(b"\n")
    return digest.hexdigest()


def _rank_key(seed: int, identity: JetIdentity, *, domain: str) -> tuple[str, str]:
    key = identity.key()
    value = f"particle-view\0{domain}\0{seed}\0{identity.label}\0{key}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest(), key


def _require_config(config: ParticleViewSplitConfig) -> None:
    if tuple(config.class_names) != tuple(LABEL_NAMES):
        raise ValueError("particle-view split config must preserve class order")
    if config.max_particles != MAX_CONSTITUENTS:
        raise ValueError("particle-view split config must use 128 particles")
    source_splits = (
        config.train_parent_split,
        config.model_val_parent_split,
        config.stack_val_parent_split,
        config.final_test_parent_split,
        *config.unused_parent_splits,
    )
    if len(set(source_splits)) != len(source_splits):
        raise ValueError("logical train/validation/test sources must be distinct")
    if any(name not in SPLIT_ORDER for name in source_splits):
        raise ValueError("particle-view config references an unknown parent split")
    n_classes = len(config.class_names)
    counts = (
        config.train_count,
        config.model_val_count,
        config.stack_val_count,
        config.final_test_count,
        *(count for _, count in config.unused_parent_split_counts),
    )
    if any(
        not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or count % n_classes
        for count in counts
    ):
        raise ValueError("all particle-view split counts must be positive/class-balanced")
    if config.model_val_count % (2 * n_classes):
        raise ValueError("model_val must split evenly and class-stratified into halves")
    if not isinstance(config.model_val_partition_seed, int) or isinstance(
        config.model_val_partition_seed, bool
    ):
        raise ValueError("model_val_partition_seed must be an integer")
    if tuple(name for name, _ in config.unused_parent_split_counts) != tuple(
        config.unused_parent_splits
    ):
        raise ValueError("unused parent split/count inventories disagree")
    if config.to_payload()["training_topology"] != PARTICLE_VIEW_TRAINING_TOPOLOGY:
        raise ValueError("particle-view training topology mismatch")


def _audit_parent(
    parent: SplitManifest, config: ParticleViewSplitConfig
) -> dict[str, Any]:
    _require_config(config)
    if tuple(parent.class_names) != tuple(config.class_names):
        raise ValueError("parent class order does not match particle-view config")
    if int(parent.max_constits) != int(config.max_particles):
        raise ValueError("parent maximum-particle contract mismatch")
    base = audit_split_manifest(parent)
    if not base.get("ok"):
        raise ValueError(f"parent split manifest failed base audit: {base}")
    expected = {
        config.train_parent_split: config.train_count,
        config.model_val_parent_split: config.model_val_count,
        config.stack_val_parent_split: config.stack_val_count,
        config.final_test_parent_split: config.final_test_count,
        **dict(config.unused_parent_split_counts),
    }
    for split, count in expected.items():
        rows = parent.splits.get(split, ())
        declared = int(parent.split_sizes.get(split, -1))
        if len(rows) != count or declared != count:
            raise ValueError(
                f"parent split {split} count mismatch: "
                f"actual={len(rows)}, declared={declared}, expected={count}"
            )
        by_label = [0] * len(config.class_names)
        for identity in rows:
            if identity.label < 0 or identity.label >= len(by_label):
                raise ValueError(f"invalid label in parent split {split}")
            by_label[identity.label] += 1
        if len(set(by_label)) != 1:
            raise ValueError(f"parent split {split} is not class balanced: {by_label}")
    return base


def _logical_payload(
    *,
    name: str,
    parent_split: str,
    parent: SplitManifest,
    indices: list[int] | None,
    config: ParticleViewSplitConfig,
) -> dict[str, Any]:
    rows = parent.splits[parent_split]
    identities = (
        list(rows)
        if indices is None
        else [rows[index] for index in indices]
    )
    purpose, seal_kind = _LOGICAL_PURPOSES[name]
    class_counts = {
        config.class_names[label]: sum(identity.label == label for identity in identities)
        for label in range(len(config.class_names))
    }
    payload: dict[str, Any] = {
        "contract": PARTICLE_VIEW_LOGICAL_SPLIT_CONTRACT,
        "name": name,
        "parent_split": parent_split,
        "membership_kind": (
            "complete_parent_alias"
            if indices is None
            else "parent_row_indices"
        ),
        "count": len(identities),
        "class_counts": class_counts,
        "ordered_identity_sha256": _identity_digest(identities),
        "purpose": purpose,
        "seal_kind": seal_kind,
    }
    if indices is not None:
        payload["parent_row_indices"] = list(indices)
        payload["partition_seed"] = int(config.model_val_partition_seed)
    return with_content_hash(payload)


def _model_val_partition_indices(
    parent: SplitManifest,
    config: ParticleViewSplitConfig,
) -> tuple[list[int], list[int]]:
    """Return the exact seeded, class-stratified stop/select membership."""

    model_val_rows = parent.splits[config.model_val_parent_split]
    by_label: list[list[int]] = [[] for _ in config.class_names]
    for index, identity in enumerate(model_val_rows):
        by_label[identity.label].append(index)
    for indices in by_label:
        indices.sort(
            key=lambda index: _rank_key(
                config.model_val_partition_seed,
                model_val_rows[index],
                domain="model-val-membership",
            )
        )

    per_class_stop = (config.model_val_count // 2) // len(config.class_names)
    stop_indices: list[int] = []
    select_indices: list[int] = []
    for indices in by_label:
        stop_indices.extend(indices[:per_class_stop])
        select_indices.extend(indices[per_class_stop:])
    stop_indices.sort(
        key=lambda index: _rank_key(
            config.model_val_partition_seed,
            model_val_rows[index],
            domain="model-val-stop-order",
        )
    )
    select_indices.sort(
        key=lambda index: _rank_key(
            config.model_val_partition_seed,
            model_val_rows[index],
            domain="model-val-select-order",
        )
    )
    return stop_indices, select_indices


def build_unified_split_manifest(
    parent: SplitManifest,
    *,
    config: ParticleViewSplitConfig = LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG,
) -> dict[str, Any]:
    """Alias one training pool and deterministically split only model_val."""

    _audit_parent(parent, config)
    stop_indices, select_indices = _model_val_partition_indices(parent, config)

    logical = {
        "train": _logical_payload(
            name="train",
            parent_split=config.train_parent_split,
            parent=parent,
            indices=None,
            config=config,
        ),
        "model_val_stop": _logical_payload(
            name="model_val_stop",
            parent_split=config.model_val_parent_split,
            parent=parent,
            indices=stop_indices,
            config=config,
        ),
        "model_val_select": _logical_payload(
            name="model_val_select",
            parent_split=config.model_val_parent_split,
            parent=parent,
            indices=select_indices,
            config=config,
        ),
        "stack_val": _logical_payload(
            name="stack_val",
            parent_split=config.stack_val_parent_split,
            parent=parent,
            indices=None,
            config=config,
        ),
        "final_test": _logical_payload(
            name="final_test",
            parent_split=config.final_test_parent_split,
            parent=parent,
            indices=None,
            config=config,
        ),
    }
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
            "campaign": "particle_view_500k_v1",
            "parent_manifest_sha256": manifest_hash(parent),
            "split_config": config.to_payload(),
            "split_config_sha256": config.content_hash,
            "training_topology": {
                "contract": PARTICLE_VIEW_TRAINING_TOPOLOGY,
                "logical_training_split": "train",
                "parent_training_split": config.train_parent_split,
                "training_subpartitions": [],
                "cross_fit": False,
                "fold_assignments": False,
                "unused_parent_splits": list(config.unused_parent_splits),
            },
            "logical_splits": logical,
            "component_training_bindings": {
                component: {
                    "logical_split": "train",
                    "train_split_sha256": logical["train"]["content_hash"],
                    "train_identity_sha256": logical["train"][
                        "ordered_identity_sha256"
                    ],
                }
                for component in PARTICLE_VIEW_TRAINABLE_COMPONENTS
            },
        }
    )
    audit_unified_split_manifest(artifact, parent=parent, config=config)
    return artifact


def audit_unified_split_manifest(
    payload: Mapping[str, Any],
    *,
    parent: SplitManifest,
    config: ParticleViewSplitConfig = LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG,
) -> dict[str, Any]:
    """Fail closed on stale parents, overlap, topology drift, or tampering."""

    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT
    )
    expected_top_level_fields = {
        "contract",
        "campaign",
        "parent_manifest_sha256",
        "split_config",
        "split_config_sha256",
        "training_topology",
        "logical_splits",
        "component_training_bindings",
        "content_hash",
    }
    if set(payload) != expected_top_level_fields:
        raise ValueError("unified split manifest field inventory mismatch")
    if payload.get("campaign") != "particle_view_500k_v1":
        raise ValueError("unified split manifest campaign mismatch")
    _audit_parent(parent, config)
    if payload.get("parent_manifest_sha256") != manifest_hash(parent):
        raise ValueError("unified split manifest is bound to a stale parent")
    if payload.get("split_config") != config.to_payload():
        raise ValueError("unified split config payload mismatch")
    if payload.get("split_config_sha256") != config.content_hash:
        raise ValueError("unified split config hash mismatch")
    topology = payload.get("training_topology")
    expected_topology = {
        "contract": PARTICLE_VIEW_TRAINING_TOPOLOGY,
        "logical_training_split": "train",
        "parent_training_split": config.train_parent_split,
        "training_subpartitions": [],
        "cross_fit": False,
        "fold_assignments": False,
        "unused_parent_splits": list(config.unused_parent_splits),
    }
    if topology != expected_topology:
        raise ValueError("particle-view training topology must be one unsplit pool")

    logical = payload.get("logical_splits")
    if not isinstance(logical, Mapping) or set(logical) != set(
        PARTICLE_VIEW_LOGICAL_SPLITS
    ):
        raise ValueError("unified manifest logical split order/set mismatch")
    expected_parent = {
        "train": config.train_parent_split,
        "model_val_stop": config.model_val_parent_split,
        "model_val_select": config.model_val_parent_split,
        "stack_val": config.stack_val_parent_split,
        "final_test": config.final_test_parent_split,
    }
    expected_count = {
        "train": config.train_count,
        "model_val_stop": config.model_val_count // 2,
        "model_val_select": config.model_val_count // 2,
        "stack_val": config.stack_val_count,
        "final_test": config.final_test_count,
    }
    expected_stop, expected_select = _model_val_partition_indices(parent, config)
    expected_child_indices = {
        "model_val_stop": expected_stop,
        "model_val_select": expected_select,
    }
    identity_sets: dict[str, set[str]] = {}
    summaries: dict[str, Any] = {}
    for name in PARTICLE_VIEW_LOGICAL_SPLITS:
        split_payload = logical[name]
        if not isinstance(split_payload, Mapping):
            raise ValueError(f"logical split {name} must be an object")
        validate_content_hash(
            split_payload, expected_contract=PARTICLE_VIEW_LOGICAL_SPLIT_CONTRACT
        )
        if split_payload.get("name") != name:
            raise ValueError(f"logical split name mismatch for {name}")
        parent_split = expected_parent[name]
        if split_payload.get("parent_split") != parent_split:
            raise ValueError(f"logical split parent mismatch for {name}")
        purpose, seal_kind = _LOGICAL_PURPOSES[name]
        if (
            split_payload.get("purpose") != purpose
            or split_payload.get("seal_kind") != seal_kind
        ):
            raise ValueError(f"logical split access policy mismatch for {name}")
        is_validation_child = name in expected_child_indices
        base_fields = {
            "contract",
            "name",
            "parent_split",
            "membership_kind",
            "count",
            "class_counts",
            "ordered_identity_sha256",
            "purpose",
            "seal_kind",
            "content_hash",
        }
        expected_fields = (
            base_fields | {"parent_row_indices", "partition_seed"}
            if is_validation_child
            else base_fields
        )
        if set(split_payload) != expected_fields:
            raise ValueError(f"logical split {name} field inventory mismatch")
        if split_payload.get("membership_kind") != (
            "parent_row_indices"
            if is_validation_child
            else "complete_parent_alias"
        ):
            raise ValueError(f"logical split {name} membership kind mismatch")
        if split_payload.get("count") != expected_count[name]:
            raise ValueError(f"logical split {name} count mismatch")
        parent_rows = parent.splits[parent_split]
        if is_validation_child:
            indices = split_payload.get("parent_row_indices")
            if not isinstance(indices, list) or any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in indices
            ):
                raise ValueError(f"logical split {name} has invalid row indices")
            if (
                indices != expected_child_indices[name]
                or split_payload.get("partition_seed")
                != config.model_val_partition_seed
            ):
                raise ValueError(
                    f"logical split {name} deterministic membership changed"
                )
            if len(indices) != len(set(indices)):
                raise ValueError(f"logical split {name} contains duplicate rows")
            if any(index < 0 or index >= len(parent_rows) for index in indices):
                raise ValueError(
                    f"logical split {name} contains out-of-range rows"
                )
            identities = [parent_rows[index] for index in indices]
        else:
            identities = list(parent_rows)
        if split_payload.get("ordered_identity_sha256") != _identity_digest(
            identities
        ):
            raise ValueError(f"logical split {name} identity digest mismatch")
        actual_counts = {
            config.class_names[label]: sum(
                identity.label == label for identity in identities
            )
            for label in range(len(config.class_names))
        }
        if split_payload.get("class_counts") != actual_counts or len(
            set(actual_counts.values())
        ) != 1:
            raise ValueError(f"logical split {name} is not class balanced")
        identity_sets[name] = {identity.key() for identity in identities}
        summaries[name] = {
            "content_hash": split_payload["content_hash"],
            "ordered_identity_sha256": split_payload["ordered_identity_sha256"],
            "count": expected_count[name],
            "class_counts": actual_counts,
            "parent_split": parent_split,
            "purpose": purpose,
            "seal_kind": seal_kind,
        }

    stop = set(expected_stop)
    select = set(expected_select)
    if stop & select or stop | select != set(range(config.model_val_count)):
        raise ValueError("model_val stop/select children must be disjoint and complete")
    for index, first in enumerate(PARTICLE_VIEW_LOGICAL_SPLITS):
        for second in PARTICLE_VIEW_LOGICAL_SPLITS[index + 1 :]:
            if identity_sets[first] & identity_sets[second]:
                raise ValueError(f"logical splits overlap: {first}, {second}")
    expected_binding = {
        "logical_split": "train",
        "train_split_sha256": logical["train"]["content_hash"],
        "train_identity_sha256": logical["train"]["ordered_identity_sha256"],
    }
    component_bindings = payload.get("component_training_bindings")
    if not isinstance(component_bindings, Mapping) or set(component_bindings) != set(
        PARTICLE_VIEW_TRAINABLE_COMPONENTS
    ):
        raise ValueError("trainable-component binding inventory mismatch")
    for component in PARTICLE_VIEW_TRAINABLE_COMPONENTS:
        if component_bindings[component] != expected_binding:
            raise ValueError(
                f"trainable component {component} does not use the unified train pool"
            )
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "parent_manifest_sha256": payload["parent_manifest_sha256"],
        "training_identity_sha256": logical["train"]["ordered_identity_sha256"],
        "logical_splits": summaries,
        "single_training_pool": True,
        "cross_fit": False,
        "trainable_component_count": len(PARTICLE_VIEW_TRAINABLE_COMPONENTS),
        "unused_parent_splits": list(config.unused_parent_splits),
    }


def logical_split_identities(
    payload: Mapping[str, Any],
    *,
    parent: SplitManifest,
    split_name: str,
    config: ParticleViewSplitConfig = LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG,
) -> list[JetIdentity]:
    audit_unified_split_manifest(payload, parent=parent, config=config)
    if split_name not in PARTICLE_VIEW_LOGICAL_SPLITS:
        raise KeyError(f"unknown particle-view logical split {split_name!r}")
    split_payload = payload["logical_splits"][split_name]
    rows = parent.splits[split_payload["parent_split"]]
    if split_payload["membership_kind"] == "complete_parent_alias":
        return list(rows)
    return [rows[index] for index in split_payload["parent_row_indices"]]


def logical_split_binding(
    payload: Mapping[str, Any], split_name: str
) -> tuple[str, str, str]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT
    )
    logical = payload.get("logical_splits")
    if not isinstance(logical, Mapping) or split_name not in logical:
        raise KeyError(f"unknown particle-view logical split {split_name!r}")
    split_payload = logical[split_name]
    validate_content_hash(
        split_payload, expected_contract=PARTICLE_VIEW_LOGICAL_SPLIT_CONTRACT
    )
    return (
        str(payload["parent_manifest_sha256"]),
        str(split_payload["content_hash"]),
        str(split_payload["ordered_identity_sha256"]),
    )


def write_unified_split_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT
    )
    return write_immutable_json(path, payload)


__all__ = [
    "LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG",
    "PARTICLE_VIEW_FORBIDDEN_TRAINING_SPLIT_FRAGMENTS",
    "PARTICLE_VIEW_LOGICAL_SPLIT_CONTRACT",
    "PARTICLE_VIEW_LOGICAL_SPLITS",
    "PARTICLE_VIEW_MODEL_VAL_PARTITION_SEED",
    "PARTICLE_VIEW_SPLIT_CONFIG_CONTRACT",
    "PARTICLE_VIEW_STEP1_REPORT_CONTRACT",
    "PARTICLE_VIEW_TRAINING_TOPOLOGY",
    "PARTICLE_VIEW_TRAINABLE_COMPONENTS",
    "PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT",
    "ParticleViewSplitConfig",
    "audit_unified_split_manifest",
    "build_unified_split_manifest",
    "logical_split_binding",
    "logical_split_identities",
    "write_unified_split_manifest",
]

"""Identity-only split, validation-role, and 3M scale-pool contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import heapq
from typing import Any, Iterable, Mapping

from jetclass_fresh.jetclass_data import (
    DEFAULT_SPLIT_SEEDS,
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    SPLIT_ORDER,
    JetIdentity,
    SplitManifest,
    audit_split_manifest,
    manifest_hash,
)

from .contracts import require_sha256, with_content_hash


VALIDATION_PARTITION_CONTRACT = "retb_validation_partition_manifest_v1"
SCALE_TRAIN_MANIFEST_CONTRACT = "retb_scale_train_manifest_v1"
FINAL_SELECT_LABEL_MANIFEST_CONTRACT = "retb_final_select_label_manifest_v1"
SPLIT_AUDIT_CONTRACT = "retb_split_audit_v1"
SCALE_TRAIN_AUDIT_CONTRACT = "retb_scale_train_audit_v1"

PRODUCTION_SPLIT_SIZES = {
    "model_train": 500_000,
    "model_val": 100_000,
    "stack_train": 0,
    "stack_val": 50_000,
    "final_test": 300_000,
}
LOGICAL_ROLES = {
    "model_train": "train",
    "model_val": "validation_partition_source",
    "stack_train": "unused",
    "stack_val": "final_select",
    "final_test": "final_test",
}
ACCESS_ROLES = {
    "train": ["model_weight_training", "train_only_fitting"],
    "val_stop": ["epoch_checkpoint_selection"],
    "val_design": ["calibration", "certification", "component_selection"],
    "final_select": ["stage_n_complete_graph_selection"],
    "final_test": ["post_execution_lock_evaluation"],
}


@dataclass(frozen=True)
class RetbSplitConfig:
    split_sizes: Mapping[str, int]
    split_seeds: Mapping[str, int]
    train_per_class: int
    val_stop_per_class: int
    val_design_per_class: int
    final_select_per_class: int
    final_test_per_class: int
    scale_train_per_class: int
    profile: str

    @classmethod
    def production(cls) -> "RetbSplitConfig":
        return cls(
            split_sizes=dict(PRODUCTION_SPLIT_SIZES),
            split_seeds=dict(DEFAULT_SPLIT_SEEDS),
            train_per_class=50_000,
            val_stop_per_class=5_000,
            val_design_per_class=5_000,
            final_select_per_class=5_000,
            final_test_per_class=30_000,
            scale_train_per_class=300_000,
            profile="production_500k_scale3m",
        )

    @classmethod
    def miniature(
        cls,
        *,
        train_per_class: int = 2,
        validation_role_per_class: int = 1,
        final_select_per_class: int = 1,
        final_test_per_class: int = 2,
        scale_train_per_class: int = 4,
    ) -> "RetbSplitConfig":
        values = (
            train_per_class,
            validation_role_per_class,
            final_select_per_class,
            final_test_per_class,
            scale_train_per_class,
        )
        if min(values) < 1 or scale_train_per_class < train_per_class:
            raise ValueError("invalid miniature RETB split counts")
        classes = len(LABEL_NAMES)
        return cls(
            split_sizes={
                "model_train": classes * train_per_class,
                "model_val": classes * validation_role_per_class * 2,
                "stack_train": 0,
                "stack_val": classes * final_select_per_class,
                "final_test": classes * final_test_per_class,
            },
            split_seeds=dict(DEFAULT_SPLIT_SEEDS),
            train_per_class=train_per_class,
            val_stop_per_class=validation_role_per_class,
            val_design_per_class=validation_role_per_class,
            final_select_per_class=final_select_per_class,
            final_test_per_class=final_test_per_class,
            scale_train_per_class=scale_train_per_class,
            profile="miniature_test",
        )


def _identity_rank(namespace: str, identity: JetIdentity) -> str:
    return hashlib.sha256(
        namespace.encode("utf-8") + b"\0" + identity.key().encode("utf-8")
    ).hexdigest()


def _identity_rows(identities: Iterable[JetIdentity]) -> list[dict[str, Any]]:
    return [identity.to_dict() for identity in identities]


def _identity_hash(identities: Iterable[JetIdentity]) -> str:
    digest = hashlib.sha256()
    digest.update(b"retb_identity_order_v1\0")
    for identity in identities:
        digest.update(identity.key().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(identity.label)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_source_split_manifest(
    manifest: SplitManifest,
    *,
    config: RetbSplitConfig,
) -> dict[str, Any]:
    problems: list[str] = []
    if int(manifest.max_constits) != 128:
        problems.append("max_constits must equal 128")
    if list(manifest.class_names) != list(LABEL_NAMES):
        problems.append("class names/order differ from canonical JetClass order")
    if dict(manifest.file_prefix_to_label) != dict(FILE_PREFIX_TO_LABEL):
        problems.append("file-prefix label mapping differs")
    if {key: int(value) for key, value in manifest.split_sizes.items()} != {
        key: int(config.split_sizes[key]) for key in SPLIT_ORDER
    }:
        problems.append("split sizes differ from the RETB profile")
    if {key: int(value) for key, value in manifest.split_seeds.items()} != {
        key: int(config.split_seeds[key]) for key in SPLIT_ORDER
    }:
        problems.append("split seeds differ from the RETB profile")
    if set(manifest.splits) != set(SPLIT_ORDER):
        problems.append("split keys differ from the five-way manifest")
    base = audit_split_manifest(manifest)
    if not bool(base.get("ok")):
        problems.append("base split duplicate/non-overlap audit failed")
    records = {record.path: record for record in manifest.file_records}
    if len(records) != len(manifest.file_records):
        problems.append("duplicate source file records")
    if not records:
        problems.append("source file records are required")

    class_counts: dict[str, dict[str, int]] = {}
    expected = {
        "model_train": config.train_per_class,
        "model_val": config.val_stop_per_class + config.val_design_per_class,
        "stack_train": 0,
        "stack_val": config.final_select_per_class,
        "final_test": config.final_test_per_class,
    }
    for split in SPLIT_ORDER:
        counts = Counter(int(row.label) for row in manifest.splits.get(split, []))
        class_counts[split] = {
            name: int(counts.get(index, 0))
            for index, name in enumerate(LABEL_NAMES)
        }
        if any(counts.get(index, 0) != expected[split] for index in range(10)):
            problems.append(f"{split} is not balanced at {expected[split]}/class")
        for row in manifest.splits.get(split, []):
            record = records.get(row.file)
            if record is None:
                problems.append(f"{split} identity references unknown file")
            elif (
                int(record.label) != int(row.label)
                or int(row.entry) < 0
                or int(row.entry) >= int(record.num_entries)
            ):
                problems.append(f"{split} identity violates its file record")
    if manifest.splits.get("stack_train"):
        problems.append("stack_train must be empty")
    if problems:
        raise ValueError("RETB split validation failed: " + "; ".join(problems))
    return {
        "ok": True,
        "profile": config.profile,
        "manifest_hash": manifest_hash(manifest),
        "split_counts": {
            split: len(manifest.splits[split]) for split in SPLIT_ORDER
        },
        "class_counts": class_counts,
        "cross_split_overlap_count": int(base["cross_split_overlap_count"]),
        "duplicate_within_split_count": int(base["duplicate_within_split_count"]),
    }


def build_validation_partition(
    manifest: SplitManifest,
    *,
    source_manifest_sha256: str,
    config: RetbSplitConfig,
) -> dict[str, Any]:
    require_sha256(source_manifest_sha256, name="source_manifest_sha256")
    by_label: dict[int, list[JetIdentity]] = {index: [] for index in range(10)}
    for identity in manifest.splits["model_val"]:
        by_label[int(identity.label)].append(identity)
    stop: list[JetIdentity] = []
    design: list[JetIdentity] = []
    for label in range(10):
        ordered = sorted(
            by_label[label],
            key=lambda row: (
                _identity_rank("retb_model_val_partition_v1", row),
                row.key(),
            ),
        )
        stop.extend(ordered[: config.val_stop_per_class])
        design.extend(ordered[config.val_stop_per_class :])
    return with_content_hash(
        {
            "contract": VALIDATION_PARTITION_CONTRACT,
            "schema_version": 1,
            "source_manifest_sha256": source_manifest_sha256,
            "partition_rule": (
                "per_class_sha256(retb_model_val_partition_v1||identity),identity"
            ),
            "role_order": ["val_stop", "val_design"],
            "access_roles": {
                "val_stop": ACCESS_ROLES["val_stop"],
                "val_design": ACCESS_ROLES["val_design"],
            },
            "roles": {
                "val_stop": _identity_rows(stop),
                "val_design": _identity_rows(design),
            },
            "identity_hashes": {
                "val_stop": _identity_hash(stop),
                "val_design": _identity_hash(design),
            },
            "counts": {
                "val_stop": len(stop),
                "val_design": len(design),
            },
        }
    )


def _candidate_identities(
    manifest: SplitManifest,
    *,
    label: int,
    excluded: set[str],
) -> Iterable[JetIdentity]:
    for record in sorted(manifest.file_records, key=lambda row: row.path):
        if int(record.label) != label:
            continue
        for entry in range(int(record.num_entries)):
            identity = JetIdentity(file=record.path, entry=entry, label=label)
            if identity.key() not in excluded:
                yield identity


def build_scale_train_manifest(
    manifest: SplitManifest,
    *,
    source_manifest_sha256: str,
    config: RetbSplitConfig,
) -> dict[str, Any]:
    require_sha256(source_manifest_sha256, name="source_manifest_sha256")
    all_assigned = {
        identity.key()
        for split in SPLIT_ORDER
        for identity in manifest.splits[split]
    }
    train_by_label: dict[int, list[JetIdentity]] = {index: [] for index in range(10)}
    for identity in manifest.splits["model_train"]:
        train_by_label[int(identity.label)].append(identity)

    scale_rows: list[JetIdentity] = []
    added_rows: list[JetIdentity] = []
    for label in range(10):
        needed = config.scale_train_per_class - len(train_by_label[label])
        if needed < 0:
            raise ValueError("scale pool is smaller than model_train")
        ranked = heapq.nsmallest(
            needed,
            (
                (
                    _identity_rank("retb_scale_train_v1", identity),
                    identity.key(),
                    identity,
                )
                for identity in _candidate_identities(
                    manifest, label=label, excluded=all_assigned
                )
            ),
        )
        if len(ranked) != needed:
            raise ValueError(
                f"insufficient unused source identities for scale class {label}: "
                f"needed={needed}, found={len(ranked)}"
            )
        added = [row[2] for row in ranked]
        added_rows.extend(added)
        combined = train_by_label[label] + added
        combined.sort(
            key=lambda row: (
                _identity_rank("retb_scale_train_order_v1", row),
                row.key(),
            )
        )
        scale_rows.extend(combined)

    model_train_keys = {row.key() for row in manifest.splits["model_train"]}
    scale_keys = {row.key() for row in scale_rows}
    held_out_keys = {
        row.key()
        for split in ("model_val", "stack_val", "final_test")
        for row in manifest.splits[split]
    }
    if not model_train_keys.issubset(scale_keys):
        raise AssertionError("scale_train lost a model_train identity")
    if scale_keys & held_out_keys:
        raise AssertionError("scale_train overlaps a held-out split")
    if len(scale_keys) != len(scale_rows):
        raise AssertionError("scale_train contains duplicate identities")

    return with_content_hash(
        {
            "contract": SCALE_TRAIN_MANIFEST_CONTRACT,
            "schema_version": 1,
            "source_manifest_sha256": source_manifest_sha256,
            "selection_rule": (
                "model_train_union_per_class_smallest_"
                "sha256(retb_scale_train_v1||unused_identity)"
            ),
            "ordering": (
                "class_order_then_sha256(retb_scale_train_order_v1||identity)"
            ),
            "model_train_is_exact_subset": True,
            "held_out_disjoint": True,
            "count": len(scale_rows),
            "per_class_count": config.scale_train_per_class,
            "added_count": len(added_rows),
            "identity_hash": _identity_hash(scale_rows),
            "added_identity_hash": _identity_hash(added_rows),
            "identities": _identity_rows(scale_rows),
        }
    )


def build_final_select_label_manifest(
    manifest: SplitManifest,
    *,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    require_sha256(source_manifest_sha256, name="source_manifest_sha256")
    ordered = sorted(
        manifest.splits["stack_val"],
        key=lambda row: (row.key(), int(row.label)),
    )
    return with_content_hash(
        {
            "contract": FINAL_SELECT_LABEL_MANIFEST_CONTRACT,
            "schema_version": 1,
            "source_manifest_sha256": source_manifest_sha256,
            "role": "stage_n_selector_only",
            "feature_access_allowed": False,
            "selection_inference_access_allowed": False,
            "ordering": "canonical_identity_then_label",
            "count": len(ordered),
            "identity_hash": _identity_hash(ordered),
            "rows": [
                {"identity": row.key(), "label": int(row.label)} for row in ordered
            ],
        }
    )


def build_split_audits(
    manifest: SplitManifest,
    *,
    source_audit: Mapping[str, Any],
    validation_partition: Mapping[str, Any],
    scale_manifest: Mapping[str, Any],
    final_select_labels: Mapping[str, Any],
    config: RetbSplitConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_sha = require_sha256(
        source_audit.get("manifest_hash"), name="source_audit.manifest_hash"
    )
    validation_roles = validation_partition["roles"]
    stop_keys = {
        JetIdentity.from_dict(row).key() for row in validation_roles["val_stop"]
    }
    design_keys = {
        JetIdentity.from_dict(row).key() for row in validation_roles["val_design"]
    }
    scale_rows = [JetIdentity.from_dict(row) for row in scale_manifest["identities"]]
    scale_counts = Counter(row.label for row in scale_rows)
    split_audit = with_content_hash(
        {
            "contract": SPLIT_AUDIT_CONTRACT,
            "schema_version": 1,
            "source_manifest_sha256": source_sha,
            "source_validation": dict(source_audit),
            "validation_partition_sha256": validation_partition["content_hash"],
            "final_select_label_manifest_sha256": final_select_labels["content_hash"],
            "validation_roles_disjoint": not bool(stop_keys & design_keys),
            "validation_roles_cover_model_val": (
                len(stop_keys | design_keys) == len(manifest.splits["model_val"])
            ),
            "logical_roles": dict(LOGICAL_ROLES),
            "access_roles": dict(ACCESS_ROLES),
        }
    )
    scale_audit = with_content_hash(
        {
            "contract": SCALE_TRAIN_AUDIT_CONTRACT,
            "schema_version": 1,
            "source_manifest_sha256": source_sha,
            "scale_train_manifest_sha256": scale_manifest["content_hash"],
            "count": len(scale_rows),
            "expected_count": config.scale_train_per_class * 10,
            "class_counts": {
                LABEL_NAMES[label]: int(scale_counts.get(label, 0))
                for label in range(10)
            },
            "model_train_subset": True,
            "held_out_disjoint": True,
            "duplicate_count": len(scale_rows) - len({row.key() for row in scale_rows}),
        }
    )
    return split_audit, scale_audit


__all__ = [
    "ACCESS_ROLES",
    "FINAL_SELECT_LABEL_MANIFEST_CONTRACT",
    "LOGICAL_ROLES",
    "PRODUCTION_SPLIT_SIZES",
    "RetbSplitConfig",
    "SCALE_TRAIN_AUDIT_CONTRACT",
    "SCALE_TRAIN_MANIFEST_CONTRACT",
    "SPLIT_AUDIT_CONTRACT",
    "VALIDATION_PARTITION_CONTRACT",
    "build_final_select_label_manifest",
    "build_scale_train_manifest",
    "build_split_audits",
    "build_validation_partition",
    "validate_source_split_manifest",
]

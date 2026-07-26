"""Locked split and fixed-HLT provenance for the relational ParT campaign."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from jetclass_fixed_hlt import HLT_PROFILE_V1
from jetclass_fresh.hlt_cache import (
    DEFAULT_HLT_SEEDS,
    HLT_ARRAY_FILENAME,
    HLT_METADATA_FILENAME,
    fixed_hlt_params_dict,
    fixed_hlt_params_from_profile,
    hash_arrays,
    hlt_profile_version_from_params,
    jet_identity_hash,
    load_cached_hlt_view,
)
from jetclass_fresh.jetclass_data import (
    DEFAULT_SPLIT_SEEDS,
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    RAW_TOKEN_DIM,
    SPLIT_ORDER,
    SplitManifest,
    audit_split_manifest,
    manifest_hash,
)

from .contracts import require_sha256, sha256_file, with_content_hash


RELATIONAL_SPLIT_BINDING_CONTRACT = "relational_part_split_binding_v1"
RELATIONAL_HLT_EXPECTATION_CONTRACT = "relational_part_hlt_expectation_v1"
RELATIONAL_HLT_BINDING_CONTRACT = "relational_part_hlt_binding_v1"

PRODUCTION_SPLIT_SIZES = {
    "model_train": 1_000_000,
    "model_val": 125_000,
    "stack_train": 0,
    "stack_val": 125_000,
    "final_test": 500_000,
}
PRODUCTION_LOGICAL_ROLES = {
    "model_train": "train",
    "model_val": "val_stop",
    "stack_train": "unused",
    "stack_val": "val_select",
    "final_test": "final_test",
}
LOCKED_HLT_SEEDS = {
    "model_train": 1053,
    "model_val": 1054,
    "stack_val": 1056,
    "final_test": 1057,
}
NONEMPTY_SPLITS = tuple(split for split in SPLIT_ORDER if split != "stack_train")


@dataclass(frozen=True)
class RelationalSplitConfig:
    split_sizes: Mapping[str, int]
    split_seeds: Mapping[str, int]
    class_names: tuple[str, ...] = tuple(LABEL_NAMES)
    file_prefix_to_label: tuple[tuple[str, int], ...] = tuple(
        sorted(FILE_PREFIX_TO_LABEL.items())
    )
    max_constituents: int = 128

    @classmethod
    def production(cls) -> "RelationalSplitConfig":
        return cls(
            split_sizes=dict(PRODUCTION_SPLIT_SIZES),
            split_seeds=dict(DEFAULT_SPLIT_SEEDS),
        )

    @classmethod
    def miniature(
        cls,
        *,
        per_class_train: int = 2,
        per_class_validation: int = 1,
        per_class_test: int = 2,
    ) -> "RelationalSplitConfig":
        if min(per_class_train, per_class_validation, per_class_test) < 1:
            raise ValueError("miniature nonempty splits require at least one jet/class")
        classes = len(LABEL_NAMES)
        return cls(
            split_sizes={
                "model_train": classes * per_class_train,
                "model_val": classes * per_class_validation,
                "stack_train": 0,
                "stack_val": classes * per_class_validation,
                "final_test": classes * per_class_test,
            },
            split_seeds=dict(DEFAULT_SPLIT_SEEDS),
        )

    def normalized_sizes(self) -> dict[str, int]:
        return {split: int(self.split_sizes[split]) for split in SPLIT_ORDER}

    def normalized_seeds(self) -> dict[str, int]:
        return {split: int(self.split_seeds[split]) for split in SPLIT_ORDER}

    def prefix_map(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in self.file_prefix_to_label}


def _require_exact_keys(mapping: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def validate_relational_split_manifest(
    manifest: SplitManifest,
    *,
    config: RelationalSplitConfig | None = None,
) -> dict[str, Any]:
    """Fail closed on every split invariant used by the campaign."""

    config = config or RelationalSplitConfig.production()
    expected_sizes = config.normalized_sizes()
    expected_seeds = config.normalized_seeds()
    _require_exact_keys(manifest.split_sizes, set(SPLIT_ORDER), name="split_sizes")
    _require_exact_keys(manifest.split_seeds, set(SPLIT_ORDER), name="split_seeds")
    _require_exact_keys(manifest.splits, set(SPLIT_ORDER), name="splits")

    problems: list[str] = []
    if int(manifest.max_constits) != int(config.max_constituents):
        problems.append(
            f"max_constits={manifest.max_constits}, expected={config.max_constituents}"
        )
    if list(manifest.class_names) != list(config.class_names):
        problems.append("class_names/order differs from locked JetClass ten-class order")
    if dict(manifest.file_prefix_to_label) != config.prefix_map():
        problems.append("file_prefix_to_label differs from locked mapping")
    if {key: int(value) for key, value in manifest.split_sizes.items()} != expected_sizes:
        problems.append("split_sizes differ from locked profile")
    if {key: int(value) for key, value in manifest.split_seeds.items()} != expected_seeds:
        problems.append("split_seeds differ from locked profile")

    production_profile = expected_sizes == PRODUCTION_SPLIT_SIZES
    records_by_path: dict[str, Any] = {}
    for record in manifest.file_records:
        if record.path in records_by_path:
            problems.append(f"duplicate file record for {record.path}")
            continue
        records_by_path[record.path] = record
        if int(record.label) not in range(len(config.class_names)):
            problems.append(f"file record {record.path} has invalid label")
        if int(record.num_entries) <= 0:
            problems.append(f"file record {record.path} has nonpositive entry count")
    if production_profile and not records_by_path:
        problems.append("production manifest must include source file records")

    base_audit = audit_split_manifest(manifest)
    if not bool(base_audit.get("ok")):
        problems.append("base manifest duplicate/count/non-overlap audit failed")

    class_counts: dict[str, dict[str, int]] = {}
    expected_per_class: dict[str, int] = {}
    for split in SPLIT_ORDER:
        total = expected_sizes[split]
        if total % len(config.class_names) != 0:
            problems.append(f"{split} count {total} is not divisible by class count")
            continue
        quota = total // len(config.class_names)
        expected_per_class[split] = quota
        counts = Counter(int(item.label) for item in manifest.splits[split])
        for identity in manifest.splits[split]:
            if int(identity.entry) < 0:
                problems.append(f"{split} contains a negative entry index")
            if records_by_path:
                record = records_by_path.get(identity.file)
                if record is None:
                    problems.append(
                        f"{split} identity references unregistered file {identity.file}"
                    )
                elif int(identity.label) != int(record.label):
                    problems.append(
                        f"{split} identity label differs from file record"
                    )
                elif int(identity.entry) >= int(record.num_entries):
                    problems.append(
                        f"{split} identity entry exceeds source file bounds"
                    )
        class_counts[split] = {
            str(config.class_names[label]): int(counts.get(label, 0))
            for label in range(len(config.class_names))
        }
        invalid = sorted(label for label in counts if label not in range(len(config.class_names)))
        if invalid:
            problems.append(f"{split} contains invalid labels {invalid}")
        if any(counts.get(label, 0) != quota for label in range(len(config.class_names))):
            problems.append(f"{split} is not exactly class-balanced at {quota}/class")

    if manifest.splits["stack_train"]:
        problems.append("stack_train must be empty")

    if problems:
        raise ValueError("relational split provenance failed: " + "; ".join(problems))

    return {
        "ok": True,
        "manifest_hash": manifest_hash(manifest),
        "split_counts": {
            split: len(manifest.splits[split]) for split in SPLIT_ORDER
        },
        "expected_per_class": expected_per_class,
        "class_counts": class_counts,
        "cross_split_overlap_count": int(base_audit["cross_split_overlap_count"]),
        "duplicate_within_split_count": int(
            base_audit["duplicate_within_split_count"]
        ),
    }


def build_split_binding(
    manifest: SplitManifest,
    *,
    config: RelationalSplitConfig | None = None,
) -> dict[str, Any]:
    config = config or RelationalSplitConfig.production()
    audit = validate_relational_split_manifest(manifest, config=config)
    source_hash = manifest_hash(manifest)
    return with_content_hash(
        {
            "contract": RELATIONAL_SPLIT_BINDING_CONTRACT,
            "schema_version": 1,
            "source_manifest_hash": source_hash,
            "max_constituents": int(config.max_constituents),
            "class_names": list(config.class_names),
            "file_prefix_to_label": config.prefix_map(),
            "split_order": list(SPLIT_ORDER),
            "split_sizes": config.normalized_sizes(),
            "split_seeds": config.normalized_seeds(),
            "logical_roles": dict(PRODUCTION_LOGICAL_ROLES),
            "split_identity_hashes": {
                split: jet_identity_hash(manifest.splits[split])
                for split in SPLIT_ORDER
            },
            "split_audit": audit,
            "exact_class_balance_required": True,
            "disjoint_identity_required": True,
        }
    )


def build_hlt_expectation(*, split_binding_sha256: str) -> dict[str, Any]:
    require_sha256(split_binding_sha256, name="split_binding_sha256")
    params = fixed_hlt_params_from_profile(HLT_PROFILE_V1, strength=0.6)
    return with_content_hash(
        {
            "contract": RELATIONAL_HLT_EXPECTATION_CONTRACT,
            "schema_version": 1,
            "split_binding_sha256": split_binding_sha256,
            "hlt_profile": HLT_PROFILE_V1,
            "hlt_profile_version": hlt_profile_version_from_params(params),
            "hlt_degradation_strength": 0.6,
            "hlt_params": fixed_hlt_params_dict(params),
            "max_constituents": 128,
            "raw_token_dimension": int(RAW_TOKEN_DIM),
            "derived_part_particle_input_dimension": 17,
            "required_splits": list(NONEMPTY_SPLITS),
            "forbidden_splits": ["stack_train"],
            "seeds": dict(LOCKED_HLT_SEEDS),
            "source_view": "offline",
            "persistent_offline_arrays_forbidden": True,
        }
    )


def _load_metadata(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"HLT metadata is absent or unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"HLT metadata must be an object: {path}")
    return payload


def build_hlt_binding(
    *,
    cache_dir: str | Path,
    manifest: SplitManifest,
    split_binding: Mapping[str, Any],
    hlt_expectation: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate every required cache and reject stale or extra split caches."""

    cache_root = Path(cache_dir)
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise FileNotFoundError(f"HLT cache directory is absent or unsafe: {cache_root}")
    split_hash = require_sha256(
        split_binding.get("content_hash"), name="split_binding.content_hash"
    )
    expected_parent = require_sha256(
        hlt_expectation.get("split_binding_sha256"),
        name="hlt_expectation.split_binding_sha256",
    )
    if split_hash != expected_parent:
        raise ValueError("HLT expectation belongs to a different split binding")
    source_manifest_hash = require_sha256(
        split_binding.get("source_manifest_hash"),
        name="split_binding.source_manifest_hash",
    )

    allowed_names = {
        HLT_ARRAY_FILENAME.format(split=split)
        for split in NONEMPTY_SPLITS
    } | {
        HLT_METADATA_FILENAME.format(split=split)
        for split in NONEMPTY_SPLITS
    }
    actual_relevant = {
        item.name
        for item in cache_root.iterdir()
        if item.name.endswith("_fixed_hlt.npz")
        or item.name.endswith("_fixed_hlt_metadata.json")
    }
    unexpected = sorted(actual_relevant - allowed_names)
    if unexpected:
        raise ValueError(f"unexpected HLT cache artifacts: {unexpected}")

    reports: dict[str, Any] = {}
    expected_params = hlt_expectation["hlt_params"]
    for split in NONEMPTY_SPLITS:
        array_path = cache_root / HLT_ARRAY_FILENAME.format(split=split)
        metadata_path = cache_root / HLT_METADATA_FILENAME.format(split=split)
        if array_path.is_symlink() or not array_path.is_file():
            raise FileNotFoundError(f"required HLT array is absent or unsafe: {array_path}")
        metadata = _load_metadata(metadata_path)
        exact = {
            "view": "fixed_hlt",
            "split": split,
            "seed": int(LOCKED_HLT_SEEDS[split]),
            "hlt_profile": hlt_expectation["hlt_profile"],
            "hlt_profile_version": hlt_expectation["hlt_profile_version"],
            "hlt_degradation_strength": float(
                hlt_expectation["hlt_degradation_strength"]
            ),
            "hlt_params": expected_params,
            "source_manifest_hash": source_manifest_hash,
            "source_view": "offline",
            "max_constits": int(hlt_expectation["max_constituents"]),
            "raw_token_dim": int(hlt_expectation["raw_token_dimension"]),
            "n_jets": len(manifest.splits[split]),
            "jet_identity_hash": split_binding["split_identity_hashes"][split],
            "generator": {
                "module": "jetclass_fixed_hlt",
                "function": "build_fixed_hlt_view",
                "params_class": "FixedHLTParams",
            },
        }
        mismatches = [
            key for key, expected in exact.items() if metadata.get(key) != expected
        ]
        for digest_key in ("hlt_content_hash", "source_content_hash", "diagnostics_hash"):
            try:
                require_sha256(metadata.get(digest_key), name=f"{split}.{digest_key}")
            except ValueError:
                mismatches.append(digest_key)
        if mismatches:
            raise ValueError(
                f"HLT metadata provenance mismatch for {split}: {sorted(set(mismatches))}"
            )
        cached_view = load_cached_hlt_view(cache_root, split, verify_hash=True)
        if list(cached_view.jet_ids) != list(manifest.splits[split]):
            raise ValueError(
                f"HLT cache ordered identities differ from manifest for {split}"
            )
        expected_shape = (
            len(manifest.splits[split]),
            int(hlt_expectation["max_constituents"]),
            int(hlt_expectation["raw_token_dimension"]),
        )
        if tuple(cached_view.tokens.shape) != expected_shape:
            raise ValueError(
                f"HLT cache shape mismatch for {split}: "
                f"actual={tuple(cached_view.tokens.shape)}, expected={expected_shape}"
            )
        if tuple(cached_view.mask.shape) != expected_shape[:2]:
            raise ValueError(f"HLT mask shape mismatch for {split}")
        actual_diagnostics_hash = hash_arrays(
            {
                f"diag_{key}": value
                for key, value in cached_view.metadata["diagnostics"].items()
            }
        )
        if actual_diagnostics_hash != metadata["diagnostics_hash"]:
            raise ValueError(f"HLT diagnostics content hash mismatch for {split}")
        reports[split] = {
            "seed": int(metadata["seed"]),
            "n_jets": int(metadata["n_jets"]),
            "jet_identity_hash": metadata["jet_identity_hash"],
            "hlt_content_hash": metadata["hlt_content_hash"],
            "source_content_hash": metadata["source_content_hash"],
            "diagnostics_hash": metadata["diagnostics_hash"],
            "array_sha256": sha256_file(array_path),
            "metadata_sha256": sha256_file(metadata_path),
            "array_bytes": int(array_path.stat().st_size),
            "metadata_bytes": int(metadata_path.stat().st_size),
        }

    hlt_hashes = [row["hlt_content_hash"] for row in reports.values()]
    if len(hlt_hashes) != len(set(hlt_hashes)):
        raise ValueError("HLT content hashes must be distinct across nonempty splits")

    if any(
        (cache_root / HLT_ARRAY_FILENAME.format(split="stack_train")).exists()
        or (cache_root / HLT_METADATA_FILENAME.format(split="stack_train")).exists()
        for _ in (0,)
    ):
        raise ValueError("stack_train is empty and must not have an HLT cache")

    return with_content_hash(
        {
            "contract": RELATIONAL_HLT_BINDING_CONTRACT,
            "schema_version": 1,
            "split_binding_sha256": split_hash,
            "hlt_expectation_sha256": require_sha256(
                hlt_expectation.get("content_hash"),
                name="hlt_expectation.content_hash",
            ),
            "source_manifest_hash": source_manifest_hash,
            "cache_root": str(cache_root.resolve()),
            "split_reports": reports,
            "stack_train_cache_present": False,
            "ok": True,
        }
    )


__all__ = [
    "LOCKED_HLT_SEEDS",
    "NONEMPTY_SPLITS",
    "PRODUCTION_LOGICAL_ROLES",
    "PRODUCTION_SPLIT_SIZES",
    "RELATIONAL_HLT_BINDING_CONTRACT",
    "RELATIONAL_HLT_EXPECTATION_CONTRACT",
    "RELATIONAL_SPLIT_BINDING_CONTRACT",
    "RelationalSplitConfig",
    "build_hlt_binding",
    "build_hlt_expectation",
    "build_split_binding",
    "validate_relational_split_manifest",
]

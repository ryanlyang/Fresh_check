"""Sharded, hash-bound caches for adaptive-binary hierarchy targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetIdentity, LABEL_NAMES, load_split_manifest, manifest_hash
from teacher_logit_reco.architecture_view_part import load_cached_offline_view

from .config import (
    ABPH_HLT_DEGRADATION_STRENGTH,
    ABPH_HLT_PROFILE,
    ABPH_HLT_PROFILE_VERSION,
    ABPH_OFFLINE_SUPERVISION_SPLITS,
)
from .inputs import validate_hlt_view_contract, validate_offline_view_contract
from .schemas import schema_manifest
from .targets import (
    ABPH_LEVEL_CAPACITIES,
    ABPH_TARGET_BUILDER_CONTRACT,
    ABPH_TARGET_BUILDER_VERSION,
    GROUP_FEATURE_NAMES,
    PARTICLE_TARGET_NAMES,
    ROOT_FEATURE_NAMES,
    AdaptiveBinaryHierarchyLayout,
    AdaptiveBinaryTargetBatch,
    adaptive_binary_target_invariant_report,
    build_adaptive_binary_targets,
    require_adaptive_binary_target_invariants,
)


ABPH_TARGET_CACHE_CONTRACT = "adaptive_binary_pseudooffline_target_cache_v1"
ABPH_TARGET_CACHE_SET_CONTRACT = "adaptive_binary_pseudooffline_target_cache_set_v1"
ABPH_TARGET_CACHE_SET_FILENAME = "adaptive_binary_target_cache_manifest.json"
ABPH_TARGET_CACHE_SPLITS: tuple[str, ...] = ABPH_OFFLINE_SUPERVISION_SPLITS


def _array_keys() -> tuple[str, ...]:
    keys = [
        "root_features",
        "root_identities",
        "particle_targets",
        "particle_mask",
        "hlt_axis_eta",
        "hlt_axis_phi",
        "valid_hlt_counts",
        "valid_offline_counts",
    ]
    for depth in range(len(ABPH_LEVEL_CAPACITIES)):
        prefix = f"level{depth + 1}"
        keys.extend(
            (
                f"{prefix}_features",
                f"{prefix}_mask",
                f"{prefix}_topology",
                f"{prefix}_parent_indices",
                f"{prefix}_membership",
                f"{prefix}_identities",
            )
        )
    keys.extend(("labels", "jet_file_indices", "jet_entries"))
    return tuple(keys)


_SHARD_ARRAY_KEYS = _array_keys()


@dataclass(frozen=True)
class AdaptiveBinaryTargetShard:
    targets: AdaptiveBinaryTargetBatch
    labels: np.ndarray
    jet_ids: tuple[JetIdentity, ...]
    split: str
    grouping: str
    shard_index: int
    start: int
    stop: int
    metadata: Mapping[str, Any]


def adaptive_binary_target_cache_paths(
    cache_dir: str | Path,
    split: str,
    grouping: str,
) -> tuple[Path, Path]:
    layout = AdaptiveBinaryHierarchyLayout(grouping=grouping)
    stem = f"{split}_{layout.grouping}_adaptive_binary_targets"
    root = Path(cache_dir)
    return root / stem, root / f"{stem}_metadata.json"


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_arrays(
    jet_ids: Sequence[JetIdentity],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    files: list[str] = []
    lookup: dict[str, int] = {}
    file_indices: list[int] = []
    entries: list[int] = []
    for identity in jet_ids:
        name = str(identity.file)
        if name not in lookup:
            lookup[name] = len(files)
            files.append(name)
        file_indices.append(lookup[name])
        entries.append(int(identity.entry))
    return files, np.asarray(file_indices, dtype=np.int32), np.asarray(entries, dtype=np.int64)


def _identities_from_arrays(
    files: Sequence[str],
    file_indices: np.ndarray,
    entries: np.ndarray,
    labels: np.ndarray,
) -> tuple[JetIdentity, ...]:
    return tuple(
        JetIdentity(file=str(files[int(file_index)]), entry=int(entry), label=int(label))
        for file_index, entry, label in zip(file_indices, entries, labels)
    )


def _cache_hash_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_contract": metadata["cache_contract"],
        "builder_contract": metadata["builder_contract"],
        "builder_version": metadata["builder_version"],
        "split": metadata["split"],
        "grouping": metadata["grouping"],
        "n_jets": metadata["n_jets"],
        "feature_dtype": metadata["feature_dtype"],
        "root_feature_names": metadata["root_feature_names"],
        "group_feature_names": metadata["group_feature_names"],
        "particle_target_names": metadata["particle_target_names"],
        "source_manifest_hash": metadata["source_manifest_hash"],
        "hlt_content_hash": metadata["hlt_content_hash"],
        "offline_content_hash": metadata["offline_content_hash"],
        "jet_identity_hash": metadata["jet_identity_hash"],
        "schema_manifest_hash": metadata["schema_manifest_hash"],
        "layout": metadata["layout"],
        "shards": [
            {
                key: shard[key]
                for key in (
                    "filename",
                    "shard_index",
                    "start",
                    "stop",
                    "n_jets",
                    "content_hash",
                    "jet_identity_hash",
                )
            }
            for shard in metadata["shards"]
        ],
    }


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(LABEL_NAMES))
    return {name: int(counts[index]) for index, name in enumerate(LABEL_NAMES)}


def _arrays_for_shard(
    targets: AdaptiveBinaryTargetBatch,
    *,
    labels: np.ndarray,
    file_indices: np.ndarray,
    entries: np.ndarray,
    feature_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    arrays = targets.array_dict()
    for key in tuple(arrays):
        if key.endswith("_features") or key == "particle_targets":
            arrays[key] = np.asarray(arrays[key], dtype=feature_dtype)
        else:
            arrays[key] = np.asarray(arrays[key])
    arrays.update(
        {
            "labels": np.asarray(labels, dtype=np.int64),
            "jet_file_indices": np.asarray(file_indices, dtype=np.int32),
            "jet_entries": np.asarray(entries, dtype=np.int64),
        }
    )
    return {key: np.ascontiguousarray(arrays[key]) for key in _SHARD_ARRAY_KEYS}


def _targets_from_arrays(
    arrays: Mapping[str, np.ndarray],
    layout: AdaptiveBinaryHierarchyLayout,
) -> AdaptiveBinaryTargetBatch:
    level_features = []
    level_masks = []
    level_topology = []
    level_parent_indices = []
    level_membership = []
    level_identities = []
    for depth in range(len(layout.level_capacities)):
        prefix = f"level{depth + 1}"
        level_features.append(np.asarray(arrays[f"{prefix}_features"]))
        level_masks.append(np.asarray(arrays[f"{prefix}_mask"], dtype=bool))
        level_topology.append(np.asarray(arrays[f"{prefix}_topology"], dtype=np.int8))
        level_parent_indices.append(np.asarray(arrays[f"{prefix}_parent_indices"], dtype=np.int16))
        level_membership.append(np.asarray(arrays[f"{prefix}_membership"], dtype=bool))
        level_identities.append(np.asarray(arrays[f"{prefix}_identities"], dtype="S64"))
    targets = AdaptiveBinaryTargetBatch(
        root_features=np.asarray(arrays["root_features"]),
        root_identities=np.asarray(arrays["root_identities"], dtype="S64"),
        level_features=tuple(level_features),
        level_masks=tuple(level_masks),
        level_topology=tuple(level_topology),
        level_parent_indices=tuple(level_parent_indices),
        level_membership=tuple(level_membership),
        level_identities=tuple(level_identities),
        particle_targets=np.asarray(arrays["particle_targets"]),
        particle_mask=np.asarray(arrays["particle_mask"], dtype=bool),
        hlt_axis_eta=np.asarray(arrays["hlt_axis_eta"], dtype=np.float32),
        hlt_axis_phi=np.asarray(arrays["hlt_axis_phi"], dtype=np.float32),
        valid_hlt_counts=np.asarray(arrays["valid_hlt_counts"], dtype=np.int32),
        valid_offline_counts=np.asarray(arrays["valid_offline_counts"], dtype=np.int32),
        layout=layout,
        diagnostics={"all_finite": True, "loaded_from_cache": True},
    )
    require_adaptive_binary_target_invariants(targets)
    return targets


def build_adaptive_binary_target_cache(
    *,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
    output_cache_dir: str | Path,
    split: str,
    grouping: str,
    chunk_size: int = 512,
    feature_dtype: str = "float32",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one grouping/split as independently verifiable NPZ shards."""

    split = str(split)
    if split not in ABPH_TARGET_CACHE_SPLITS:
        raise ValueError(
            f"offline hierarchy targets are restricted to {ABPH_TARGET_CACHE_SPLITS}; got {split!r}"
        )
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be positive")
    dtype = np.dtype(feature_dtype)
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("feature_dtype must be float16 or float32")
    layout = AdaptiveBinaryHierarchyLayout(grouping=grouping)
    shard_dir, metadata_path = adaptive_binary_target_cache_paths(
        output_cache_dir, split, layout.grouping
    )
    if not overwrite and (shard_dir.exists() or metadata_path.exists()):
        raise FileExistsError(f"target cache already exists: {shard_dir}")
    if overwrite and shard_dir.exists():
        shutil.rmtree(shard_dir)
    if overwrite and metadata_path.exists():
        metadata_path.unlink()
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_split_manifest(manifest_path)
    if split not in manifest.splits:
        raise ValueError(f"split {split!r} is absent from the manifest")
    expected_n_jets = len(manifest.splits[split])
    hlt_view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True)
    offline_view = load_cached_offline_view(offline_cache_dir, split, verify_hash=True)
    hlt_report = validate_hlt_view_contract(
        hlt_view, manifest, split, expected_n_jets=expected_n_jets
    )
    offline_report = validate_offline_view_contract(
        offline_view,
        manifest,
        split,
        expected_n_jets=expected_n_jets,
        hlt_view=hlt_view,
    )
    jet_ids = tuple(hlt_view.jet_ids)
    labels = np.asarray(hlt_view.labels, dtype=np.int64)
    jet_files, file_indices, entries = _identity_arrays(jet_ids)
    shard_reports: list[dict[str, Any]] = []

    try:
        for shard_index, start in enumerate(range(0, expected_n_jets, chunk)):
            stop = min(start + chunk, expected_n_jets)
            targets = build_adaptive_binary_targets(
                hlt_view.tokens[start:stop],
                hlt_view.mask[start:stop],
                offline_view.tokens[start:stop],
                offline_view.mask[start:stop],
                jet_ids=jet_ids[start:stop],
                layout=layout,
            )
            arrays = _arrays_for_shard(
                targets,
                labels=labels[start:stop],
                file_indices=file_indices[start:stop],
                entries=entries[start:stop],
                feature_dtype=dtype,
            )
            content_hash = hash_arrays(arrays)
            filename = f"shard_{shard_index:06d}.npz"
            np.savez_compressed(shard_dir / filename, **arrays)
            invariant_report = adaptive_binary_target_invariant_report(targets)
            shard_reports.append(
                {
                    "filename": filename,
                    "shard_index": int(shard_index),
                    "start": int(start),
                    "stop": int(stop),
                    "n_jets": int(stop - start),
                    "content_hash": content_hash,
                    "jet_identity_hash": jet_identity_hash(jet_ids[start:stop]),
                    "target_identity_hash": invariant_report["target_identity_hash"],
                    "max_frontier_particle_multiplicity": invariant_report[
                        "max_frontier_particle_multiplicity"
                    ],
                }
            )
    except Exception:
        shutil.rmtree(shard_dir, ignore_errors=True)
        raise

    schemas = schema_manifest()
    metadata: dict[str, Any] = {
        "cache_contract": ABPH_TARGET_CACHE_CONTRACT,
        "builder_contract": ABPH_TARGET_BUILDER_CONTRACT,
        "builder_version": ABPH_TARGET_BUILDER_VERSION,
        "split": split,
        "grouping": layout.grouping,
        "layout": layout.to_dict(),
        "n_jets": expected_n_jets,
        "n_shards": len(shard_reports),
        "chunk_size": chunk,
        "feature_dtype": str(dtype),
        "root_feature_names": list(ROOT_FEATURE_NAMES),
        "group_feature_names": list(GROUP_FEATURE_NAMES),
        "particle_target_names": list(PARTICLE_TARGET_NAMES),
        "schema_manifest_hash": schemas["manifest_hash"],
        "schemas": schemas["schemas"],
        "source_manifest_hash": manifest_hash(manifest),
        "hlt_content_hash": hlt_report["hlt_content_hash"],
        "offline_content_hash": offline_report["offline_content_hash"],
        "jet_identity_hash": jet_identity_hash(jet_ids),
        "label_hash": hlt_report["label_hash"],
        "label_names": list(LABEL_NAMES),
        "class_counts": _class_counts(labels),
        "jet_files": jet_files,
        "hlt_profile": ABPH_HLT_PROFILE,
        "hlt_profile_version": ABPH_HLT_PROFILE_VERSION,
        "hlt_degradation_strength": ABPH_HLT_DEGRADATION_STRENGTH,
        "offline_target_splits": list(ABPH_TARGET_CACHE_SPLITS),
        "offline_final_test_loaded": False,
        "target_semantics": {
            "root": "all valid offline constituents",
            "recursive_split": "independently recluster each non-singleton parent to exactly two subjets",
            "terminal": "carry singleton unchanged; never materialize an empty sibling",
            "membership": "each active frontier is an exact partition of valid offline constituents",
            "coordinates": "HLT-axis centered eta and wrapped phi; no offline rotation",
            "tie_breaking": "SHA256(builder, grouping, jet file/entry, original constituent memberships); labels excluded",
        },
        "shards": shard_reports,
    }
    metadata["target_content_hash"] = _json_hash(_cache_hash_payload(metadata))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _load_metadata(
    cache_dir: str | Path,
    split: str,
    grouping: str,
) -> tuple[Path, dict[str, Any], AdaptiveBinaryHierarchyLayout]:
    layout = AdaptiveBinaryHierarchyLayout(grouping=grouping)
    shard_dir, metadata_path = adaptive_binary_target_cache_paths(
        cache_dir, split, layout.grouping
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_contract") != ABPH_TARGET_CACHE_CONTRACT:
        raise ValueError("adaptive binary target cache contract mismatch")
    if metadata.get("builder_version") != ABPH_TARGET_BUILDER_VERSION:
        raise ValueError("adaptive binary target builder version mismatch")
    if metadata.get("split") != split or metadata.get("grouping") != layout.grouping:
        raise ValueError("adaptive binary target cache split/grouping mismatch")
    if metadata.get("schema_manifest_hash") != schema_manifest()["manifest_hash"]:
        raise ValueError("adaptive binary target cache schema hash is stale")
    if metadata.get("layout") != layout.to_dict():
        raise ValueError("adaptive binary target cache layout is stale")
    if metadata.get("hlt_profile") != ABPH_HLT_PROFILE:
        raise ValueError("adaptive binary target cache HLT profile mismatch")
    if metadata.get("hlt_profile_version") != ABPH_HLT_PROFILE_VERSION:
        raise ValueError("adaptive binary target cache HLT profile version mismatch")
    if abs(
        float(metadata.get("hlt_degradation_strength", -1.0))
        - ABPH_HLT_DEGRADATION_STRENGTH
    ) > 1.0e-12:
        raise ValueError("adaptive binary target cache HLT strength mismatch")
    expected_hash = _json_hash(_cache_hash_payload(metadata))
    if metadata.get("target_content_hash") != expected_hash:
        raise ValueError("adaptive binary target aggregate content hash mismatch")
    return shard_dir, metadata, layout


def load_adaptive_binary_target_shard(
    cache_dir: str | Path,
    split: str,
    grouping: str,
    shard_index: int,
    *,
    verify_hash: bool = True,
) -> AdaptiveBinaryTargetShard:
    shard_dir, metadata, layout = _load_metadata(cache_dir, split, grouping)
    index = int(shard_index)
    shards = metadata["shards"]
    if not 0 <= index < len(shards):
        raise IndexError(f"shard index {index} outside [0, {len(shards)})")
    shard_metadata = dict(shards[index])
    if int(shard_metadata.get("shard_index", -1)) != index:
        raise ValueError("noncontiguous adaptive binary shard metadata")
    with np.load(shard_dir / shard_metadata["filename"], allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in _SHARD_ARRAY_KEYS}
    if verify_hash and hash_arrays(arrays) != shard_metadata.get("content_hash"):
        raise ValueError(f"adaptive binary target shard hash mismatch for {split}/{grouping}/{index}")
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    jet_ids = _identities_from_arrays(
        metadata["jet_files"], arrays["jet_file_indices"], arrays["jet_entries"], labels
    )
    shard_identity_hash = jet_identity_hash(jet_ids)
    if shard_identity_hash != shard_metadata.get("jet_identity_hash"):
        raise ValueError("adaptive binary target shard identity hash mismatch")
    return AdaptiveBinaryTargetShard(
        targets=_targets_from_arrays(arrays, layout),
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        grouping=layout.grouping,
        shard_index=index,
        start=int(shard_metadata["start"]),
        stop=int(shard_metadata["stop"]),
        metadata={**metadata, "active_shard": shard_metadata},
    )


def load_adaptive_binary_target_cache_metadata(
    cache_dir: str | Path,
    split: str,
    grouping: str,
) -> dict[str, Any]:
    """Load and aggregate-hash-validate one target cache's metadata."""

    _, metadata, _ = _load_metadata(cache_dir, split, grouping)
    return dict(metadata)


def iter_adaptive_binary_target_shards(
    cache_dir: str | Path,
    split: str,
    grouping: str,
    *,
    verify_hash: bool = True,
) -> Iterator[AdaptiveBinaryTargetShard]:
    _, metadata, _ = _load_metadata(cache_dir, split, grouping)
    for index in range(int(metadata["n_shards"])):
        yield load_adaptive_binary_target_shard(
            cache_dir, split, grouping, index, verify_hash=verify_hash
        )


def audit_adaptive_binary_target_cache(
    cache_dir: str | Path,
    *,
    manifest_path: str | Path,
    splits: Sequence[str] = ABPH_TARGET_CACHE_SPLITS,
    groupings: Sequence[str] = ("exclusive_kt", "cambridge_aachen"),
    verify_hash: bool = True,
) -> dict[str, Any]:
    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    reports: dict[str, Any] = {}
    problems: list[str] = []
    for split in tuple(str(value) for value in splits):
        for grouping_value in groupings:
            grouping = AdaptiveBinaryHierarchyLayout(grouping=grouping_value).grouping
            key = f"{split}/{grouping}"
            local: list[str] = []
            try:
                _, metadata, _ = _load_metadata(cache_dir, split, grouping)
                expected_ids = tuple(manifest.splits[split])
                if metadata.get("source_manifest_hash") != manifest_sha:
                    local.append("source_manifest_hash mismatch")
                if metadata.get("jet_identity_hash") != jet_identity_hash(expected_ids):
                    local.append("jet_identity_hash mismatch")
                observed: list[JetIdentity] = []
                expected_start = 0
                for shard in iter_adaptive_binary_target_shards(
                    cache_dir, split, grouping, verify_hash=verify_hash
                ):
                    if shard.start != expected_start:
                        local.append(f"shard {shard.shard_index} start is not contiguous")
                    observed.extend(shard.jet_ids)
                    expected_start = shard.stop
                if tuple(observed) != expected_ids:
                    local.append("shard identities/order differ from manifest")
                if expected_start != len(expected_ids):
                    local.append("shards do not cover the full split")
            except Exception as exc:
                metadata = {}
                local.append(str(exc))
            problems.extend(f"{key}: {problem}" for problem in local)
            reports[key] = {
                "ok": not local,
                "problems": local,
                "n_jets": metadata.get("n_jets"),
                "n_shards": metadata.get("n_shards"),
                "target_content_hash": metadata.get("target_content_hash"),
            }
    return {
        "ok": not problems,
        "cache_contract": ABPH_TARGET_CACHE_CONTRACT,
        "source_manifest_hash": manifest_sha,
        "reports": reports,
        "problems": problems,
    }


def build_adaptive_binary_target_caches(
    *,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
    output_cache_dir: str | Path,
    splits: Sequence[str] = ABPH_TARGET_CACHE_SPLITS,
    groupings: Sequence[str] = ("exclusive_kt", "cambridge_aachen"),
    chunk_size: int = 512,
    feature_dtype: str = "float32",
    overwrite: bool = False,
) -> dict[str, Any]:
    requested_splits = tuple(str(value) for value in splits)
    if any(split not in ABPH_TARGET_CACHE_SPLITS for split in requested_splits):
        raise ValueError(f"target splits must be a subset of {ABPH_TARGET_CACHE_SPLITS}")
    requested_groupings = tuple(
        AdaptiveBinaryHierarchyLayout(grouping=value).grouping for value in groupings
    )
    if len(requested_groupings) != len(set(requested_groupings)):
        raise ValueError("duplicate hierarchy grouping requested")
    reports: dict[str, Any] = {}
    for split in requested_splits:
        for grouping in requested_groupings:
            key = f"{split}/{grouping}"
            reports[key] = build_adaptive_binary_target_cache(
                manifest_path=manifest_path,
                hlt_cache_dir=hlt_cache_dir,
                offline_cache_dir=offline_cache_dir,
                output_cache_dir=output_cache_dir,
                split=split,
                grouping=grouping,
                chunk_size=chunk_size,
                feature_dtype=feature_dtype,
                overwrite=overwrite,
            )
    manifest_payload: dict[str, Any] = {
        "cache_set_contract": ABPH_TARGET_CACHE_SET_CONTRACT,
        "source_manifest_hash": manifest_hash(load_split_manifest(manifest_path)),
        "splits": list(requested_splits),
        "groupings": list(requested_groupings),
        "offline_final_test_loaded": False,
        "caches": {
            key: {
                "target_content_hash": value["target_content_hash"],
                "n_jets": value["n_jets"],
                "n_shards": value["n_shards"],
                "hlt_content_hash": value["hlt_content_hash"],
                "offline_content_hash": value["offline_content_hash"],
            }
            for key, value in reports.items()
        },
    }
    manifest_payload["cache_set_hash"] = _json_hash(manifest_payload)
    output = Path(output_cache_dir) / ABPH_TARGET_CACHE_SET_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_payload


__all__ = [
    "ABPH_TARGET_CACHE_CONTRACT",
    "ABPH_TARGET_CACHE_SET_CONTRACT",
    "ABPH_TARGET_CACHE_SET_FILENAME",
    "ABPH_TARGET_CACHE_SPLITS",
    "AdaptiveBinaryTargetShard",
    "adaptive_binary_target_cache_paths",
    "audit_adaptive_binary_target_cache",
    "build_adaptive_binary_target_cache",
    "build_adaptive_binary_target_caches",
    "iter_adaptive_binary_target_shards",
    "load_adaptive_binary_target_cache_metadata",
    "load_adaptive_binary_target_shard",
]

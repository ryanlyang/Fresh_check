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
from .compact_target_codec import (
    ABPH_COMPACT_TARGET_CODEC_CONTRACT,
    ABPH_COMPACT_TARGET_CODEC_NAME,
    ABPH_LEGACY_TARGET_CODEC_NAME,
    ABPH_TARGET_STORAGE_CODECS,
    decode_compact_target_arrays,
    encode_compact_target_arrays,
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
    reconstruct_target_identity_arrays,
    require_adaptive_binary_target_invariants,
)


ABPH_TARGET_CACHE_CONTRACT = "adaptive_binary_pseudooffline_target_cache_v1"
ABPH_TARGET_CACHE_SET_CONTRACT = "adaptive_binary_pseudooffline_target_cache_set_v1"
ABPH_TARGET_CACHE_SET_FILENAME = "adaptive_binary_target_cache_manifest.json"
ABPH_TARGET_CACHE_SPLITS: tuple[str, ...] = ABPH_OFFLINE_SUPERVISION_SPLITS
ABPH_FORENSIC_IDENTITY_SAMPLE_FILENAME = "forensic_identity_sample.npz"


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
_TARGET_IDENTITY_ARRAY_KEYS = (
    "root_identities",
    *(f"level{depth + 1}_identities" for depth in range(len(ABPH_LEVEL_CAPACITIES))),
)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    payload = {
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
    if "storage_codec" in metadata:
        payload.update(
            {
                "storage_codec": metadata["storage_codec"],
                "codec_contract": metadata.get("codec_contract"),
                "forensic_identity_sample": metadata.get(
                    "forensic_identity_sample"
                ),
                "storage_summary": metadata.get("storage_summary"),
            }
        )
        for payload_shard, shard in zip(payload["shards"], metadata["shards"]):
            for key in (
                "encoded_content_hash",
                "file_sha256",
                "storage_bytes",
                "codec_manifest_hash",
                "target_identity_hash",
                "logical_uncompressed_bytes",
                "encoded_uncompressed_bytes",
            ):
                if key in shard:
                    payload_shard[key] = shard[key]
    return payload


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
    storage_codec: str = ABPH_LEGACY_TARGET_CODEC_NAME,
    forensic_jets_per_class: int = 2,
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
    codec = str(storage_codec)
    if codec not in ABPH_TARGET_STORAGE_CODECS:
        raise ValueError(f"unknown target storage codec {codec!r}")
    if codec == ABPH_COMPACT_TARGET_CODEC_NAME and dtype != np.dtype("float32"):
        raise ValueError("compact lossless targets require float32 feature values")
    forensic_per_class = int(forensic_jets_per_class)
    if forensic_per_class < 0:
        raise ValueError("forensic_jets_per_class cannot be negative")
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
    forensic_parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            *_TARGET_IDENTITY_ARRAY_KEYS,
            "labels",
            "jet_file_indices",
            "jet_entries",
        )
    }
    forensic_global_indices: list[int] = []
    forensic_class_counts = np.zeros(len(LABEL_NAMES), dtype=np.int64)

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
            shard_path = shard_dir / filename
            codec_manifest = None
            encoded_content_hash = None
            codec_manifest_hash = None
            if codec == ABPH_COMPACT_TARGET_CODEC_NAME:
                stored_arrays, codec_manifest = encode_compact_target_arrays(
                    arrays,
                    omitted_identity_keys=_TARGET_IDENTITY_ARRAY_KEYS,
                )
                encoded_content_hash = codec_manifest["encoded_content_hash"]
                codec_manifest_hash = codec_manifest["manifest_hash"]
                np.savez_compressed(shard_path, **stored_arrays)
                encoded_uncompressed_bytes = sum(
                    int(value.nbytes) for value in stored_arrays.values()
                )
            else:
                np.savez_compressed(shard_path, **arrays)
                encoded_uncompressed_bytes = sum(
                    int(value.nbytes) for value in arrays.values()
                )
            invariant_report = adaptive_binary_target_invariant_report(targets)
            shard_report = {
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
                    "file_sha256": _sha256_file(shard_path),
                    "storage_bytes": int(shard_path.stat().st_size),
                    "logical_uncompressed_bytes": sum(
                        int(value.nbytes) for value in arrays.values()
                    ),
                    "encoded_uncompressed_bytes": encoded_uncompressed_bytes,
                }
            if encoded_content_hash is not None:
                shard_report.update(
                    {
                        "encoded_content_hash": encoded_content_hash,
                        "codec_manifest_hash": codec_manifest_hash,
                    }
                )
            shard_reports.append(shard_report)

            selected_local: list[int] = []
            if codec == ABPH_COMPACT_TARGET_CODEC_NAME and forensic_per_class:
                for local_index, label in enumerate(labels[start:stop]):
                    label_index = int(label)
                    if forensic_class_counts[label_index] < forensic_per_class:
                        forensic_class_counts[label_index] += 1
                        selected_local.append(local_index)
                        forensic_global_indices.append(start + local_index)
            if selected_local:
                selected = np.asarray(selected_local, dtype=np.int64)
                for key in forensic_parts:
                    forensic_parts[key].append(np.ascontiguousarray(arrays[key][selected]))
    except Exception:
        shutil.rmtree(shard_dir, ignore_errors=True)
        raise

    schemas = schema_manifest()
    forensic_report = None
    if codec == ABPH_COMPACT_TARGET_CODEC_NAME and forensic_global_indices:
        forensic_arrays = {
            key: np.concatenate(parts, axis=0)
            for key, parts in forensic_parts.items()
        }
        forensic_arrays["global_indices"] = np.asarray(
            forensic_global_indices, dtype=np.int64
        )
        forensic_path = shard_dir / ABPH_FORENSIC_IDENTITY_SAMPLE_FILENAME
        np.savez_compressed(forensic_path, **forensic_arrays)
        forensic_report = {
            "filename": ABPH_FORENSIC_IDENTITY_SAMPLE_FILENAME,
            "n_jets": len(forensic_global_indices),
            "global_indices": forensic_global_indices,
            "class_counts": _class_counts(forensic_arrays["labels"]),
            "content_hash": hash_arrays(forensic_arrays),
            "file_sha256": _sha256_file(forensic_path),
            "storage_bytes": int(forensic_path.stat().st_size),
            "identity_keys": list(_TARGET_IDENTITY_ARRAY_KEYS),
        }
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
        "storage_codec": codec,
        "codec_contract": (
            ABPH_COMPACT_TARGET_CODEC_CONTRACT
            if codec == ABPH_COMPACT_TARGET_CODEC_NAME
            else ABPH_LEGACY_TARGET_CODEC_NAME
        ),
        "forensic_identity_sample": forensic_report,
        "storage_summary": {
            "shard_storage_bytes": sum(
                int(row["storage_bytes"]) for row in shard_reports
            ),
            "forensic_storage_bytes": (
                int(forensic_report["storage_bytes"])
                if forensic_report is not None
                else 0
            ),
            "logical_uncompressed_bytes": sum(
                int(row["logical_uncompressed_bytes"]) for row in shard_reports
            ),
            "encoded_uncompressed_bytes": sum(
                int(row["encoded_uncompressed_bytes"]) for row in shard_reports
            ),
        },
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
    storage_codec = metadata.get("storage_codec", ABPH_LEGACY_TARGET_CODEC_NAME)
    if storage_codec not in ABPH_TARGET_STORAGE_CODECS:
        raise ValueError("adaptive binary target storage codec is unknown")
    if storage_codec == ABPH_COMPACT_TARGET_CODEC_NAME:
        if metadata.get("codec_contract") != ABPH_COMPACT_TARGET_CODEC_CONTRACT:
            raise ValueError("adaptive binary compact target codec contract mismatch")
        if metadata.get("feature_dtype") != "float32":
            raise ValueError("adaptive binary compact targets are not float32")
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
    shard_path = shard_dir / shard_metadata["filename"]
    if verify_hash and shard_metadata.get("file_sha256"):
        if _sha256_file(shard_path) != shard_metadata["file_sha256"]:
            raise ValueError(
                f"adaptive binary target shard hash mismatch for {split}/{grouping}/{index} "
                "(file bytes)"
            )
    storage_codec = metadata.get("storage_codec", ABPH_LEGACY_TARGET_CODEC_NAME)
    with np.load(shard_path, allow_pickle=False) as source:
        if storage_codec == ABPH_COMPACT_TARGET_CODEC_NAME:
            encoded = {key: np.asarray(source[key]) for key in source.files}
            arrays, codec_manifest = decode_compact_target_arrays(
                encoded,
                expected_encoded_content_hash=(
                    shard_metadata.get("encoded_content_hash") if verify_hash else None
                ),
            )
            if codec_manifest.get("manifest_hash") != shard_metadata.get(
                "codec_manifest_hash"
            ):
                raise ValueError("adaptive binary compact codec manifest hash mismatch")
        else:
            arrays = {key: np.asarray(source[key]) for key in _SHARD_ARRAY_KEYS}
            codec_manifest = None
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    jet_ids = _identities_from_arrays(
        metadata["jet_files"], arrays["jet_file_indices"], arrays["jet_entries"], labels
    )
    if storage_codec == ABPH_COMPACT_TARGET_CODEC_NAME:
        root_identities, level_identities = reconstruct_target_identity_arrays(
            jet_ids=jet_ids,
            layout=layout,
            particle_mask=arrays["particle_mask"],
            level_masks=tuple(
                arrays[f"level{depth + 1}_mask"]
                for depth in range(len(layout.level_capacities))
            ),
            level_membership=tuple(
                arrays[f"level{depth + 1}_membership"]
                for depth in range(len(layout.level_capacities))
            ),
        )
        arrays["root_identities"] = root_identities
        for depth, identities in enumerate(level_identities):
            arrays[f"level{depth + 1}_identities"] = identities
        if codec_manifest.get("logical_content_hash") != shard_metadata.get(
            "content_hash"
        ):
            raise ValueError("adaptive binary compact logical hash metadata mismatch")
    if set(arrays) != set(_SHARD_ARRAY_KEYS):
        raise ValueError("adaptive binary target shard schema is incomplete")
    if verify_hash and hash_arrays(arrays) != shard_metadata.get("content_hash"):
        raise ValueError(f"adaptive binary target shard hash mismatch for {split}/{grouping}/{index}")
    shard_identity_hash = jet_identity_hash(jet_ids)
    if shard_identity_hash != shard_metadata.get("jet_identity_hash"):
        raise ValueError("adaptive binary target shard identity hash mismatch")
    targets = _targets_from_arrays(arrays, layout)
    invariant_report = adaptive_binary_target_invariant_report(targets)
    if invariant_report["target_identity_hash"] != shard_metadata.get(
        "target_identity_hash"
    ):
        raise ValueError("adaptive binary target shard target-identity hash mismatch")
    return AdaptiveBinaryTargetShard(
        targets=targets,
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


def _load_forensic_identity_sample(
    shard_dir: Path,
    metadata: Mapping[str, Any],
    *,
    verify_hash: bool,
) -> dict[str, np.ndarray] | None:
    report = metadata.get("forensic_identity_sample")
    if report is None:
        return None
    if not isinstance(report, Mapping):
        raise ValueError("forensic target identity sample metadata is malformed")
    path = shard_dir / str(report["filename"])
    if verify_hash and _sha256_file(path) != report.get("file_sha256"):
        raise ValueError("forensic target identity sample file hash mismatch")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    if verify_hash and hash_arrays(arrays) != report.get("content_hash"):
        raise ValueError("forensic target identity sample content hash mismatch")
    expected_keys = {
        *_TARGET_IDENTITY_ARRAY_KEYS,
        "labels",
        "jet_file_indices",
        "jet_entries",
        "global_indices",
    }
    if set(arrays) != expected_keys:
        raise ValueError("forensic target identity sample schema mismatch")
    indices = np.asarray(arrays["global_indices"], dtype=np.int64)
    if indices.ndim != 1 or indices.size != int(report["n_jets"]):
        raise ValueError("forensic target identity sample length mismatch")
    if indices.tolist() != list(report["global_indices"]):
        raise ValueError("forensic target identity sample indices mismatch")
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError("forensic target identity sample indices are not ordered")
    return arrays


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
                shard_dir, metadata, _ = _load_metadata(cache_dir, split, grouping)
                expected_ids = tuple(manifest.splits[split])
                if metadata.get("source_manifest_hash") != manifest_sha:
                    local.append("source_manifest_hash mismatch")
                if metadata.get("jet_identity_hash") != jet_identity_hash(expected_ids):
                    local.append("jet_identity_hash mismatch")
                observed: list[JetIdentity] = []
                expected_start = 0
                forensic = _load_forensic_identity_sample(
                    shard_dir, metadata, verify_hash=verify_hash
                )
                forensic_lookup = (
                    {
                        int(global_index): sample_index
                        for sample_index, global_index in enumerate(
                            forensic["global_indices"]
                        )
                    }
                    if forensic is not None
                    else {}
                )
                checked_forensic: set[int] = set()
                for shard in iter_adaptive_binary_target_shards(
                    cache_dir, split, grouping, verify_hash=verify_hash
                ):
                    if shard.start != expected_start:
                        local.append(f"shard {shard.shard_index} start is not contiguous")
                    observed.extend(shard.jet_ids)
                    for global_index in range(shard.start, shard.stop):
                        sample_index = forensic_lookup.get(global_index)
                        if sample_index is None:
                            continue
                        local_index = global_index - shard.start
                        if (
                            shard.targets.root_identities[local_index]
                            != forensic["root_identities"][sample_index]
                        ):
                            local.append(
                                f"forensic root identity mismatch at jet {global_index}"
                            )
                        for depth in range(len(shard.targets.level_identities)):
                            key_name = f"level{depth + 1}_identities"
                            if not np.array_equal(
                                shard.targets.level_identities[depth][local_index],
                                forensic[key_name][sample_index],
                            ):
                                local.append(
                                    f"forensic level identity mismatch at jet {global_index} depth {depth + 1}"
                                )
                        expected_identity = shard.jet_ids[local_index]
                        forensic_identity = JetIdentity(
                            file=str(
                                metadata["jet_files"][
                                    int(forensic["jet_file_indices"][sample_index])
                                ]
                            ),
                            entry=int(forensic["jet_entries"][sample_index]),
                            label=int(forensic["labels"][sample_index]),
                        )
                        if expected_identity != forensic_identity:
                            local.append(
                                f"forensic jet identity mismatch at jet {global_index}"
                            )
                        checked_forensic.add(global_index)
                    expected_start = shard.stop
                if tuple(observed) != expected_ids:
                    local.append("shard identities/order differ from manifest")
                if expected_start != len(expected_ids):
                    local.append("shards do not cover the full split")
                if checked_forensic != set(forensic_lookup):
                    local.append("forensic identity sample was not fully audited")
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
                "storage_codec": metadata.get(
                    "storage_codec", ABPH_LEGACY_TARGET_CODEC_NAME
                ),
                "forensic_identity_sample_count": (
                    metadata.get("forensic_identity_sample") or {}
                ).get("n_jets", 0),
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
    storage_codec: str = ABPH_LEGACY_TARGET_CODEC_NAME,
    forensic_jets_per_class: int = 2,
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
                storage_codec=storage_codec,
                forensic_jets_per_class=forensic_jets_per_class,
                overwrite=overwrite,
            )
    manifest_payload: dict[str, Any] = {
        "cache_set_contract": ABPH_TARGET_CACHE_SET_CONTRACT,
        "source_manifest_hash": manifest_hash(load_split_manifest(manifest_path)),
        "splits": list(requested_splits),
        "groupings": list(requested_groupings),
        "offline_final_test_loaded": False,
        "storage_codec": str(storage_codec),
        "forensic_jets_per_class": int(forensic_jets_per_class),
        "caches": {
            key: {
                "target_content_hash": value["target_content_hash"],
                "n_jets": value["n_jets"],
                "n_shards": value["n_shards"],
                "hlt_content_hash": value["hlt_content_hash"],
                "offline_content_hash": value["offline_content_hash"],
                "storage_codec": value["storage_codec"],
                "codec_contract": value["codec_contract"],
                "storage_summary": value["storage_summary"],
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
    "ABPH_FORENSIC_IDENTITY_SAMPLE_FILENAME",
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

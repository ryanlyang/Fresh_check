"""Sharded cache I/O and provenance audits for hierarchy targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from jetclass_fixed_hlt import HLT_PROFILE_V2_REALISTIC, HLT_PROFILE_V2_REALISTIC_VERSION
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash, load_cached_hlt_view, normalize_hlt_profile
from jetclass_fresh.jetclass_data import JetIdentity, LABEL_NAMES, load_split_manifest, manifest_hash
from teacher_logit_reco.architecture_view_part import load_cached_offline_view

from .layout import (
    ACCOUNTING_FIELD_NAMES,
    DERIVED_DIAGNOSTIC_FIELD_NAMES,
    HIERARCHY_LAYOUT_VERSION,
    LEVEL_CELL_COUNTS,
    HierarchyTargetLayout,
    default_hierarchy_target_layout,
)
from .targets import (
    HIERARCHY_TARGET_BUILDER_VERSION,
    HierarchyTargetOutput,
    build_hierarchy_targets,
    derive_accounting_diagnostics,
    fit_radial_boundary_from_hlt,
    hierarchy_consistency_report,
    require_hierarchy_consistency,
)


HIERARCHY_TARGET_CACHE_CONTRACT = "constrained_coarse_to_fine_hierarchy_target_cache_v1"
HIERARCHY_TARGET_CACHE_SET_CONTRACT = "constrained_coarse_to_fine_hierarchy_target_cache_set_v1"
HIERARCHY_TARGET_EXPECTED_HLT_PROFILE = HLT_PROFILE_V2_REALISTIC
HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION = HLT_PROFILE_V2_REALISTIC_VERSION
HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH = 2.5
HIERARCHY_TARGET_DEFAULT_SPLITS: tuple[str, ...] = ("model_train", "model_val")
HIERARCHY_TARGET_ALLOWED_SPLITS: tuple[str, ...] = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
)
HIERARCHY_TARGET_CACHE_SET_FILENAME = "hierarchy_target_cache_manifest.json"

_SHARD_ARRAY_KEYS: tuple[str, ...] = (
    "global_accounting",
    "level1_accounting",
    "level2_accounting",
    "level3_accounting",
    "final_cell_indices",
    "reference_eta",
    "reference_phi",
    "valid_hlt_counts",
    "valid_offline_counts",
    "unknown_pid_counts",
    "clipped_particle_counts",
    "labels",
    "jet_file_indices",
    "jet_entries",
)


@dataclass(frozen=True)
class HierarchyTargetShard:
    output: HierarchyTargetOutput
    labels: np.ndarray
    jet_ids: tuple[JetIdentity, ...]
    split: str
    shard_index: int
    start: int
    stop: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HierarchyTargetCache:
    output: HierarchyTargetOutput
    labels: np.ndarray
    jet_ids: tuple[JetIdentity, ...]
    split: str
    metadata: dict[str, Any]


def hierarchy_target_cache_paths(cache_dir: str | Path, split: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    return (
        root / f"{split}_hierarchy_targets",
        root / f"{split}_hierarchy_targets_metadata.json",
    )


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = {name: 0 for name in LABEL_NAMES}
    for label in np.asarray(labels, dtype=np.int64):
        counts[LABEL_NAMES[int(label)]] += 1
    return counts


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values)
    if arr.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _jet_identity_arrays(jet_ids: Sequence[JetIdentity]) -> tuple[list[str], np.ndarray, np.ndarray]:
    jet_files: list[str] = []
    file_to_index: dict[str, int] = {}
    file_indices: list[int] = []
    entries: list[int] = []
    for identity in jet_ids:
        file_name = str(identity.file)
        if file_name not in file_to_index:
            file_to_index[file_name] = len(jet_files)
            jet_files.append(file_name)
        file_indices.append(file_to_index[file_name])
        entries.append(int(identity.entry))
    return jet_files, np.asarray(file_indices, dtype=np.int32), np.asarray(entries, dtype=np.int64)


def _jet_ids_from_arrays(
    jet_files: Sequence[str],
    file_indices: np.ndarray,
    entries: np.ndarray,
    labels: np.ndarray,
) -> tuple[JetIdentity, ...]:
    return tuple(
        JetIdentity(file=str(jet_files[int(file_index)]), entry=int(entry), label=int(label))
        for file_index, entry, label in zip(file_indices, entries, labels)
    )


def _layout_from_metadata(metadata: Mapping[str, Any]) -> HierarchyTargetLayout:
    layout_payload = metadata.get("layout")
    if not isinstance(layout_payload, Mapping):
        raise ValueError("hierarchy target metadata is missing layout")
    layout = default_hierarchy_target_layout(
        radial_boundary=float(layout_payload["radial_boundary"]),
        coordinate_extent=float(layout_payload["coordinate_extent"]),
    )
    if str(layout_payload.get("layout_version")) != HIERARCHY_LAYOUT_VERSION:
        raise ValueError(
            f"hierarchy layout version mismatch: {layout_payload.get('layout_version')} != {HIERARCHY_LAYOUT_VERSION}"
        )
    if tuple(layout_payload.get("field_names", ())) != ACCOUNTING_FIELD_NAMES:
        raise ValueError("hierarchy target field_names do not match the current accounting contract")
    if tuple(int(value) for value in layout_payload.get("level_cell_counts", ())) != LEVEL_CELL_COUNTS:
        raise ValueError("hierarchy target level_cell_counts do not match the current hierarchy contract")
    return layout


def _validate_source_pair(
    *,
    split: str,
    manifest_path: str | Path,
    hlt_view: Any,
    offline_view: Any,
) -> tuple[str, tuple[JetIdentity, ...]]:
    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    if str(split) not in manifest.splits:
        raise ValueError(f"split {split!r} is missing from split manifest")
    expected_ids = tuple(manifest.splits[str(split)])
    hlt_ids = tuple(hlt_view.jet_ids)
    offline_ids = tuple(offline_view.jet_ids)
    if hlt_ids != offline_ids:
        raise ValueError(f"HLT/offline jet identities do not match for split {split}")
    if hlt_ids != expected_ids:
        raise ValueError(f"HLT/offline cache identities do not match the manifest for split {split}")
    if not np.array_equal(np.asarray(hlt_view.labels), np.asarray(offline_view.labels)):
        raise ValueError(f"HLT/offline labels do not match for split {split}")
    for source_name, view in (("HLT", hlt_view), ("offline", offline_view)):
        source_manifest = view.metadata.get("source_manifest_hash")
        if source_manifest != manifest_sha:
            raise ValueError(
                f"{source_name} source_manifest_hash mismatch for {split}: {source_manifest} != {manifest_sha}"
            )

    actual_profile = normalize_hlt_profile(hlt_view.metadata.get("hlt_profile"))
    if actual_profile != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE:
        raise ValueError(
            f"HLT profile mismatch for {split}: {actual_profile} != {HIERARCHY_TARGET_EXPECTED_HLT_PROFILE}"
        )
    actual_version = str(hlt_view.metadata.get("hlt_profile_version") or "")
    if actual_version != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION:
        raise ValueError(
            f"HLT profile version mismatch for {split}: "
            f"{actual_version} != {HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION}"
        )
    actual_strength = hlt_view.metadata.get("hlt_degradation_strength")
    if actual_strength is None or abs(float(actual_strength) - HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH) > 1.0e-12:
        raise ValueError(
            f"HLT degradation strength mismatch for {split}: "
            f"{actual_strength} != {HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH}"
        )
    if not hlt_view.metadata.get("hlt_content_hash"):
        raise ValueError(f"HLT cache for {split} is missing hlt_content_hash")
    if not (offline_view.metadata.get("offline_content_hash") or offline_view.metadata.get("content_hash")):
        raise ValueError(f"offline cache for {split} is missing offline_content_hash")
    return manifest_sha, hlt_ids


def _output_arrays(
    output: HierarchyTargetOutput,
    *,
    labels: np.ndarray,
    file_indices: np.ndarray,
    entries: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "global_accounting": np.asarray(output.global_accounting),
        "level1_accounting": np.asarray(output.level1_accounting),
        "level2_accounting": np.asarray(output.level2_accounting),
        "level3_accounting": np.asarray(output.level3_accounting),
        "final_cell_indices": np.asarray(output.final_cell_indices, dtype=np.int16),
        "reference_eta": np.asarray(output.reference_eta, dtype=np.float32),
        "reference_phi": np.asarray(output.reference_phi, dtype=np.float32),
        "valid_hlt_counts": np.asarray(output.valid_hlt_counts, dtype=np.int32),
        "valid_offline_counts": np.asarray(output.valid_offline_counts, dtype=np.int32),
        "unknown_pid_counts": np.asarray(output.unknown_pid_counts, dtype=np.int32),
        "clipped_particle_counts": np.asarray(output.clipped_particle_counts, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int64),
        "jet_file_indices": np.asarray(file_indices, dtype=np.int32),
        "jet_entries": np.asarray(entries, dtype=np.int64),
    }


def _shard_hash(arrays: Mapping[str, np.ndarray]) -> str:
    return hash_arrays({key: np.asarray(arrays[key]) for key in _SHARD_ARRAY_KEYS})


def _cache_content_hash_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_contract": metadata["cache_contract"],
        "builder_version": metadata["builder_version"],
        "split": metadata["split"],
        "n_jets": metadata["n_jets"],
        "target_dtype": metadata["target_dtype"],
        "source_manifest_hash": metadata["source_manifest_hash"],
        "hlt_content_hash": metadata["hlt_content_hash"],
        "offline_content_hash": metadata["offline_content_hash"],
        "jet_identity_hash": metadata["jet_identity_hash"],
        "layout": metadata["layout"],
        "shards": [
            {
                "filename": shard["filename"],
                "shard_index": shard["shard_index"],
                "start": shard["start"],
                "stop": shard["stop"],
                "n_jets": shard["n_jets"],
                "content_hash": shard["content_hash"],
            }
            for shard in metadata["shards"]
        ],
    }


def _shard_diagnostics(output: HierarchyTargetOutput) -> dict[str, Any]:
    global_derived = derive_accounting_diagnostics(output.global_accounting)
    total_pt_index = ACCOUNTING_FIELD_NAMES.index("total_pT")
    level_summaries: dict[str, Any] = {}
    for level_name, accounting in (
        ("level1", output.level1_accounting),
        ("level2", output.level2_accounting),
        ("level3", output.level3_accounting),
    ):
        cell_pt = np.asarray(accounting[..., total_pt_index], dtype=np.float64)
        jet_pt = np.maximum(cell_pt.sum(axis=1), 1.0e-8)
        level_summaries[level_name] = {
            "active_cell_count": _summary(np.sum(cell_pt > 0.0, axis=1)),
            "max_cell_pT_fraction": _summary(np.max(cell_pt, axis=1) / jet_pt),
        }
    return {
        "valid_hlt_counts": _summary(output.valid_hlt_counts),
        "valid_offline_counts": _summary(output.valid_offline_counts),
        "unknown_pid_counts": _summary(output.unknown_pid_counts),
        "clipped_particle_counts": _summary(output.clipped_particle_counts),
        "global_derived_mean": {
            name: float(global_derived[:, index].mean()) if global_derived.shape[0] else 0.0
            for index, name in enumerate(DERIVED_DIAGNOSTIC_FIELD_NAMES)
        },
        "levels": level_summaries,
        "consistency": hierarchy_consistency_report(output),
        "all_finite": bool(output.diagnostics.get("all_finite", False)),
    }


def build_hierarchy_target_cache(
    *,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
    output_cache_dir: str | Path,
    split: str,
    layout: HierarchyTargetLayout,
    overwrite: bool = False,
    chunk_size: int = 8192,
    target_dtype: str = "float32",
) -> dict[str, Any]:
    """Build one split as bounded, independently hashed NPZ shards."""

    split = str(split)
    if split == "final_test":
        raise ValueError("offline final_test hierarchy targets are forbidden by the deployable campaign contract")
    if split not in HIERARCHY_TARGET_ALLOWED_SPLITS:
        raise ValueError(f"unsupported hierarchy-target split {split!r}")
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be positive")
    dtype = np.dtype(str(target_dtype))
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("target_dtype must be float16 or float32")

    shard_dir, metadata_path = hierarchy_target_cache_paths(output_cache_dir, split)
    if not overwrite and (shard_dir.exists() or metadata_path.exists()):
        raise FileExistsError(f"hierarchy target cache already exists for {split}: {shard_dir}")
    if overwrite and shard_dir.exists():
        shutil.rmtree(shard_dir)
    if overwrite and metadata_path.exists():
        metadata_path.unlink()
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    hlt_view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True)
    offline_view = load_cached_offline_view(offline_cache_dir, split, verify_hash=True)
    manifest_sha, jet_ids = _validate_source_pair(
        split=split,
        manifest_path=manifest_path,
        hlt_view=hlt_view,
        offline_view=offline_view,
    )
    n_jets = int(hlt_view.tokens.shape[0])
    jet_files, all_file_indices, all_entries = _jet_identity_arrays(jet_ids)
    labels = np.asarray(hlt_view.labels, dtype=np.int64)
    shard_reports: list[dict[str, Any]] = []
    aggregate_valid_hlt: list[np.ndarray] = []
    aggregate_valid_offline: list[np.ndarray] = []
    aggregate_unknown_pid: list[np.ndarray] = []
    aggregate_clipped: list[np.ndarray] = []

    for shard_index, start in enumerate(range(0, n_jets, chunk)):
        stop = min(start + chunk, n_jets)
        output = build_hierarchy_targets(
            hlt_view.tokens[start:stop],
            hlt_view.mask[start:stop],
            offline_view.tokens[start:stop],
            offline_view.mask[start:stop],
            layout=layout,
            target_dtype=dtype,
        )
        require_hierarchy_consistency(output, atol=5.0e-3 if dtype == np.dtype("float16") else 2.0e-4)
        arrays = _output_arrays(
            output,
            labels=labels[start:stop],
            file_indices=all_file_indices[start:stop],
            entries=all_entries[start:stop],
        )
        arrays = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
        filename = f"shard_{shard_index:05d}.npz"
        shard_path = shard_dir / filename
        np.savez_compressed(shard_path, **arrays)
        content_hash = _shard_hash(arrays)
        report = {
            "filename": filename,
            "shard_index": int(shard_index),
            "start": int(start),
            "stop": int(stop),
            "n_jets": int(stop - start),
            "content_hash": content_hash,
            "diagnostics": _shard_diagnostics(output),
        }
        shard_reports.append(report)
        aggregate_valid_hlt.append(output.valid_hlt_counts)
        aggregate_valid_offline.append(output.valid_offline_counts)
        aggregate_unknown_pid.append(output.unknown_pid_counts)
        aggregate_clipped.append(output.clipped_particle_counts)

    identity_hash = jet_identity_hash(jet_ids)
    metadata: dict[str, Any] = {
        "cache_contract": HIERARCHY_TARGET_CACHE_CONTRACT,
        "builder_version": HIERARCHY_TARGET_BUILDER_VERSION,
        "split": split,
        "oracle_only": False,
        "allowed_for_primary_training": True,
        "array_directory": str(shard_dir),
        "metadata_path": str(metadata_path),
        "n_jets": n_jets,
        "n_shards": len(shard_reports),
        "chunk_size": chunk,
        "target_dtype": str(dtype),
        "field_dim": len(ACCOUNTING_FIELD_NAMES),
        "field_names": list(ACCOUNTING_FIELD_NAMES),
        "level_cell_counts": list(LEVEL_CELL_COUNTS),
        "derived_diagnostic_field_names": list(DERIVED_DIAGNOSTIC_FIELD_NAMES),
        "label_names": list(LABEL_NAMES),
        "class_counts": _class_counts(labels),
        "jet_files": jet_files,
        "jet_identity_hash": identity_hash,
        "source_manifest_hash": manifest_sha,
        "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
        "offline_content_hash": offline_view.metadata.get("offline_content_hash")
        or offline_view.metadata.get("content_hash"),
        "hlt_jet_identity_hash": hlt_view.metadata.get("jet_identity_hash") or identity_hash,
        "offline_jet_identity_hash": offline_view.metadata.get("jet_identity_hash") or identity_hash,
        "hlt_profile": normalize_hlt_profile(hlt_view.metadata.get("hlt_profile")),
        "hlt_profile_version": str(hlt_view.metadata.get("hlt_profile_version")),
        "hlt_degradation_strength": float(hlt_view.metadata.get("hlt_degradation_strength")),
        "hlt_params": hlt_view.metadata.get("hlt_params"),
        "layout": layout.to_dict(),
        "target_semantics": {
            "reference_axis": "pT-weighted HLT eta and circular phi",
            "offline_supervision": "offline particles aligned by manifest jet identity",
            "coordinate_overflow": "project deta/dphi to configured rendering boundary before cell assignment and moments",
            "pid_unknown_fallback": "neutral_hadron, counted in unknown_pid_counts",
            "total_pT": "derived exactly from five stored PID-category pT channels",
            "expected_constituent_count": "derived exactly from five stored PID-category count channels",
            "final_test": "offline hierarchy targets forbidden",
        },
        "shards": shard_reports,
        "diagnostics_summary": {
            "valid_hlt_counts": _summary(np.concatenate(aggregate_valid_hlt)),
            "valid_offline_counts": _summary(np.concatenate(aggregate_valid_offline)),
            "unknown_pid_counts": _summary(np.concatenate(aggregate_unknown_pid)),
            "clipped_particle_counts": _summary(np.concatenate(aggregate_clipped)),
            "all_finite": bool(all(shard["diagnostics"]["all_finite"] for shard in shard_reports)),
            "max_parent_child_closure_error": float(
                max(
                    (
                        max(shard["diagnostics"]["consistency"]["parent_child_closure"].values())
                        for shard in shard_reports
                    ),
                    default=0.0,
                )
            ),
        },
    }
    metadata["target_content_hash"] = _json_hash(_cache_content_hash_payload(metadata))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _load_metadata(cache_dir: str | Path, split: str) -> tuple[Path, dict[str, Any], HierarchyTargetLayout]:
    shard_dir, metadata_path = hierarchy_target_cache_paths(cache_dir, split)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_contract") != HIERARCHY_TARGET_CACHE_CONTRACT:
        raise ValueError(f"hierarchy target cache contract mismatch for {split}")
    if metadata.get("builder_version") != HIERARCHY_TARGET_BUILDER_VERSION:
        raise ValueError(f"hierarchy target builder version mismatch for {split}")
    if str(metadata.get("split")) != str(split):
        raise ValueError(f"hierarchy target split mismatch: {metadata.get('split')} != {split}")
    layout = _layout_from_metadata(metadata)
    expected_content_hash = _json_hash(_cache_content_hash_payload(metadata))
    if metadata.get("target_content_hash") != expected_content_hash:
        raise ValueError(f"hierarchy target aggregate content hash mismatch for {split}")
    return shard_dir, metadata, layout


def _output_from_arrays(arrays: Mapping[str, np.ndarray], layout: HierarchyTargetLayout) -> HierarchyTargetOutput:
    output = HierarchyTargetOutput(
        global_accounting=np.asarray(arrays["global_accounting"]),
        level1_accounting=np.asarray(arrays["level1_accounting"]),
        level2_accounting=np.asarray(arrays["level2_accounting"]),
        level3_accounting=np.asarray(arrays["level3_accounting"]),
        final_cell_indices=np.asarray(arrays["final_cell_indices"], dtype=np.int16),
        reference_eta=np.asarray(arrays["reference_eta"], dtype=np.float32),
        reference_phi=np.asarray(arrays["reference_phi"], dtype=np.float32),
        valid_hlt_counts=np.asarray(arrays["valid_hlt_counts"], dtype=np.int32),
        valid_offline_counts=np.asarray(arrays["valid_offline_counts"], dtype=np.int32),
        unknown_pid_counts=np.asarray(arrays["unknown_pid_counts"], dtype=np.int32),
        clipped_particle_counts=np.asarray(arrays["clipped_particle_counts"], dtype=np.int32),
        layout=layout,
        diagnostics={
            "all_finite": bool(
                np.isfinite(arrays["global_accounting"]).all()
                and np.isfinite(arrays["level1_accounting"]).all()
                and np.isfinite(arrays["level2_accounting"]).all()
                and np.isfinite(arrays["level3_accounting"]).all()
            )
        },
    )
    output.diagnostics.update(hierarchy_consistency_report(output))
    return output


def load_hierarchy_target_shard(
    cache_dir: str | Path,
    split: str,
    shard_index: int,
    *,
    verify_hash: bool = True,
) -> HierarchyTargetShard:
    shard_dir, metadata, layout = _load_metadata(cache_dir, split)
    index = int(shard_index)
    shards = metadata.get("shards", [])
    if not 0 <= index < len(shards):
        raise IndexError(f"shard_index {index} is outside [0, {len(shards)})")
    shard_metadata = dict(shards[index])
    if int(shard_metadata.get("shard_index", -1)) != index:
        raise ValueError(f"non-contiguous hierarchy target shard metadata at index {index}")
    shard_path = shard_dir / str(shard_metadata["filename"])
    with np.load(shard_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in _SHARD_ARRAY_KEYS}
    if verify_hash:
        actual_hash = _shard_hash(arrays)
        if actual_hash != shard_metadata.get("content_hash"):
            raise ValueError(
                f"hierarchy target shard hash mismatch for {split}/{index}: "
                f"{actual_hash} != {shard_metadata.get('content_hash')}"
            )
    output = _output_from_arrays(arrays, layout)
    tolerance = 5.0e-3 if output.global_accounting.dtype == np.float16 else 2.0e-4
    require_hierarchy_consistency(output, atol=tolerance)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    jet_ids = _jet_ids_from_arrays(
        metadata["jet_files"],
        arrays["jet_file_indices"],
        arrays["jet_entries"],
        labels,
    )
    return HierarchyTargetShard(
        output=output,
        labels=labels,
        jet_ids=jet_ids,
        split=str(split),
        shard_index=index,
        start=int(shard_metadata["start"]),
        stop=int(shard_metadata["stop"]),
        metadata={**metadata, "active_shard": shard_metadata},
    )


def iter_hierarchy_target_shards(
    cache_dir: str | Path,
    split: str,
    *,
    verify_hash: bool = True,
) -> Iterator[HierarchyTargetShard]:
    _, metadata, _ = _load_metadata(cache_dir, split)
    for shard_index in range(int(metadata["n_shards"])):
        yield load_hierarchy_target_shard(cache_dir, split, shard_index, verify_hash=verify_hash)


def load_hierarchy_target_cache(
    cache_dir: str | Path,
    split: str,
    *,
    verify_hash: bool = True,
    max_concat_jets: int = 250_000,
) -> HierarchyTargetCache:
    """Concatenate a small cache; high-data training should consume shards."""

    _, metadata, layout = _load_metadata(cache_dir, split)
    if int(metadata["n_jets"]) > int(max_concat_jets):
        raise MemoryError(
            f"refusing to concatenate {metadata['n_jets']} hierarchy targets; "
            "use iter_hierarchy_target_shards for high-data caches"
        )
    shards = list(iter_hierarchy_target_shards(cache_dir, split, verify_hash=verify_hash))
    if not shards:
        raise ValueError(f"hierarchy target cache for {split} has no shards")
    outputs = [shard.output for shard in shards]
    output = HierarchyTargetOutput(
        global_accounting=np.concatenate([item.global_accounting for item in outputs], axis=0),
        level1_accounting=np.concatenate([item.level1_accounting for item in outputs], axis=0),
        level2_accounting=np.concatenate([item.level2_accounting for item in outputs], axis=0),
        level3_accounting=np.concatenate([item.level3_accounting for item in outputs], axis=0),
        final_cell_indices=np.concatenate([item.final_cell_indices for item in outputs], axis=0),
        reference_eta=np.concatenate([item.reference_eta for item in outputs], axis=0),
        reference_phi=np.concatenate([item.reference_phi for item in outputs], axis=0),
        valid_hlt_counts=np.concatenate([item.valid_hlt_counts for item in outputs], axis=0),
        valid_offline_counts=np.concatenate([item.valid_offline_counts for item in outputs], axis=0),
        unknown_pid_counts=np.concatenate([item.unknown_pid_counts for item in outputs], axis=0),
        clipped_particle_counts=np.concatenate([item.clipped_particle_counts for item in outputs], axis=0),
        layout=layout,
        diagnostics={"all_finite": True},
    )
    output.diagnostics.update(hierarchy_consistency_report(output))
    require_hierarchy_consistency(output)
    labels = np.concatenate([shard.labels for shard in shards], axis=0)
    jet_ids = tuple(identity for shard in shards for identity in shard.jet_ids)
    if jet_identity_hash(jet_ids) != metadata.get("jet_identity_hash"):
        raise ValueError(f"hierarchy target concatenated jet identity hash mismatch for {split}")
    return HierarchyTargetCache(output=output, labels=labels, jet_ids=jet_ids, split=str(split), metadata=metadata)


def audit_hierarchy_target_cache(
    cache_dir: str | Path,
    *,
    manifest_path: str | Path,
    splits: Sequence[str] = HIERARCHY_TARGET_DEFAULT_SPLITS,
    expected_split_sizes: Mapping[str, int] | None = None,
    verify_shard_hashes: bool = True,
) -> dict[str, Any]:
    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    split_reports: dict[str, Any] = {}
    problems: list[str] = []
    for split in tuple(str(value) for value in splits):
        split_problems: list[str] = []
        try:
            _, metadata, _ = _load_metadata(cache_dir, split)
            expected_ids = tuple(manifest.splits[split])
            if metadata.get("source_manifest_hash") != manifest_sha:
                split_problems.append("source_manifest_hash mismatch")
            if int(metadata.get("n_jets", -1)) != len(expected_ids):
                split_problems.append("n_jets does not match manifest")
            if expected_split_sizes is not None and int(metadata.get("n_jets", -1)) != int(expected_split_sizes[split]):
                split_problems.append("n_jets does not match expected split size")
            if metadata.get("jet_identity_hash") != jet_identity_hash(expected_ids):
                split_problems.append("jet_identity_hash does not match manifest")
            if metadata.get("hlt_profile") != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE:
                split_problems.append("HLT profile mismatch")
            if metadata.get("hlt_profile_version") != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION:
                split_problems.append("HLT profile version mismatch")
            if abs(float(metadata.get("hlt_degradation_strength", -1.0)) - HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH) > 1.0e-12:
                split_problems.append("HLT degradation strength mismatch")

            observed_ids: list[JetIdentity] = []
            expected_start = 0
            for shard in iter_hierarchy_target_shards(cache_dir, split, verify_hash=verify_shard_hashes):
                if shard.start != expected_start:
                    split_problems.append(f"shard {shard.shard_index} starts at {shard.start}, expected {expected_start}")
                observed_ids.extend(shard.jet_ids)
                expected_start = shard.stop
            if expected_start != int(metadata["n_jets"]):
                split_problems.append("shards do not cover n_jets exactly")
            if tuple(observed_ids) != expected_ids:
                split_problems.append("shard jet identities do not match manifest ordering")
        except Exception as exc:
            split_problems.append(str(exc))
            metadata = {}
        if split_problems:
            problems.extend(f"{split}: {problem}" for problem in split_problems)
        split_reports[split] = {
            "ok": not split_problems,
            "problems": split_problems,
            "n_jets": metadata.get("n_jets"),
            "n_shards": metadata.get("n_shards"),
            "target_content_hash": metadata.get("target_content_hash"),
            "hlt_content_hash": metadata.get("hlt_content_hash"),
            "offline_content_hash": metadata.get("offline_content_hash"),
            "jet_identity_hash": metadata.get("jet_identity_hash"),
        }
    return {
        "ok": not problems,
        "cache_contract": HIERARCHY_TARGET_CACHE_CONTRACT,
        "source_manifest_hash": manifest_sha,
        "splits": list(splits),
        "split_reports": split_reports,
        "problems": problems,
    }


def build_hierarchy_target_caches(
    *,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
    output_cache_dir: str | Path,
    splits: Sequence[str] = HIERARCHY_TARGET_DEFAULT_SPLITS,
    radial_boundary: float | None = None,
    coordinate_extent: float = 0.8,
    radial_histogram_bins: int = 4096,
    radial_fit_chunk_size: int = 8192,
    chunk_size: int = 8192,
    target_dtype: str = "float32",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit one model_train geometry and build aligned target caches."""

    requested_splits = tuple(str(split) for split in splits)
    if "final_test" in requested_splits:
        raise ValueError("final_test offline hierarchy targets are forbidden")
    unknown = sorted(set(requested_splits) - set(HIERARCHY_TARGET_ALLOWED_SPLITS))
    if unknown:
        raise ValueError(f"unsupported hierarchy-target splits: {unknown}")

    if radial_boundary is None:
        model_train_hlt = load_cached_hlt_view(hlt_cache_dir, "model_train", verify_hash=True)
        source_manifest_sha = manifest_hash(load_split_manifest(manifest_path))
        actual_profile = normalize_hlt_profile(model_train_hlt.metadata.get("hlt_profile"))
        actual_version = str(model_train_hlt.metadata.get("hlt_profile_version") or "")
        actual_strength = model_train_hlt.metadata.get("hlt_degradation_strength")
        if model_train_hlt.metadata.get("source_manifest_hash") != source_manifest_sha:
            raise ValueError("model_train HLT source_manifest_hash does not match the active manifest")
        if actual_profile != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE:
            raise ValueError("model_train HLT profile is not fixed_hlt_v2_realistic")
        if actual_version != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION:
            raise ValueError("model_train HLT profile version does not match the hierarchy-target contract")
        if actual_strength is None or abs(float(actual_strength) - HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH) > 1.0e-12:
            raise ValueError("model_train HLT degradation strength is not 2.5")
        resolved_boundary, radial_fit = fit_radial_boundary_from_hlt(
            model_train_hlt.tokens,
            model_train_hlt.mask,
            coordinate_extent=float(coordinate_extent),
            histogram_bins=int(radial_histogram_bins),
            chunk_size=int(radial_fit_chunk_size),
        )
    else:
        resolved_boundary = float(radial_boundary)
        radial_fit = {
            "method": "explicit",
            "radial_boundary": resolved_boundary,
            "coordinate_extent": float(coordinate_extent),
        }
    layout = default_hierarchy_target_layout(
        radial_boundary=resolved_boundary,
        coordinate_extent=float(coordinate_extent),
    )

    reports: dict[str, Any] = {}
    for split in requested_splits:
        reports[split] = build_hierarchy_target_cache(
            manifest_path=manifest_path,
            hlt_cache_dir=hlt_cache_dir,
            offline_cache_dir=offline_cache_dir,
            output_cache_dir=output_cache_dir,
            split=split,
            layout=layout,
            overwrite=bool(overwrite),
            chunk_size=int(chunk_size),
            target_dtype=str(target_dtype),
        )

    manifest = load_split_manifest(manifest_path)
    cache_set = {
        "cache_set_contract": HIERARCHY_TARGET_CACHE_SET_CONTRACT,
        "builder_version": HIERARCHY_TARGET_BUILDER_VERSION,
        "source_manifest_hash": manifest_hash(manifest),
        "hlt_profile": HIERARCHY_TARGET_EXPECTED_HLT_PROFILE,
        "hlt_profile_version": HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION,
        "hlt_degradation_strength": HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH,
        "splits": list(requested_splits),
        "layout": layout.to_dict(),
        "radial_fit": radial_fit,
        "target_dtype": str(np.dtype(target_dtype)),
        "split_target_content_hashes": {
            split: reports[split]["target_content_hash"] for split in requested_splits
        },
    }
    cache_set["cache_set_content_hash"] = _json_hash(cache_set)
    output_root = Path(output_cache_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / HIERARCHY_TARGET_CACHE_SET_FILENAME).write_text(
        json.dumps(cache_set, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"cache_set": cache_set, "reports": reports}

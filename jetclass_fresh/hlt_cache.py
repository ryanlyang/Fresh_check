"""Fixed-HLT view generation and caching for Step 3 of the replication."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from jetclass_fixed_hlt import FixedHLTParams, build_fixed_hlt_view, summarize_hlt_diagnostics

from .jetclass_data import (
    JetIdentity,
    JetView,
    SplitManifest,
    SPLIT_ORDER,
    load_offline_view,
    manifest_hash,
)


DEFAULT_HLT_SEEDS = {
    "model_train": 1053,
    "model_val": 1054,
    "stack_train": 1055,
    "stack_val": 1056,
    "final_test": 1057,
}

HLT_ARRAY_FILENAME = "{split}_fixed_hlt.npz"
HLT_METADATA_FILENAME = "{split}_fixed_hlt_metadata.json"


def fixed_hlt_params_dict(params: FixedHLTParams | None = None) -> Dict[str, float]:
    """Return the exact fixed HLT parameter profile as plain JSON values."""

    return {key: float(value) for key, value in asdict(params or FixedHLTParams()).items()}


def _hash_update_array(hasher: "hashlib._Hash", name: str, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(json.dumps(arr.shape).encode("utf-8"))
    hasher.update(arr.tobytes())


def hash_arrays(named_arrays: Mapping[str, np.ndarray]) -> str:
    """Stable SHA256 hash for numeric numpy arrays."""

    hasher = hashlib.sha256()
    for name in sorted(named_arrays):
        _hash_update_array(hasher, name, named_arrays[name])
    return hasher.hexdigest()


def jet_identity_hash(jet_ids: Sequence[JetIdentity]) -> str:
    """Stable SHA256 hash for ordered jet identities."""

    hasher = hashlib.sha256()
    for identity in jet_ids:
        hasher.update(identity.file.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(int(identity.entry)).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(str(int(identity.label)).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _count_summary(counts: np.ndarray) -> Dict[str, float]:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return {
            "n_jets": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    return {
        "n_jets": int(counts.size),
        "min": float(np.min(counts)),
        "max": float(np.max(counts)),
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "p10": float(np.percentile(counts, 10)),
        "p50": float(np.percentile(counts, 50)),
        "p90": float(np.percentile(counts, 90)),
    }


def _cache_paths(cache_dir: str | Path, split: str) -> tuple[Path, Path]:
    cache_root = Path(cache_dir)
    return (
        cache_root / HLT_ARRAY_FILENAME.format(split=split),
        cache_root / HLT_METADATA_FILENAME.format(split=split),
    )


def _identity_arrays(jet_ids: Sequence[JetIdentity]) -> tuple[list[str], np.ndarray, np.ndarray]:
    unique_files: list[str] = []
    file_to_index: Dict[str, int] = {}
    file_indices = np.zeros((len(jet_ids),), dtype=np.int32)
    entries = np.zeros((len(jet_ids),), dtype=np.int64)
    for index, identity in enumerate(jet_ids):
        if identity.file not in file_to_index:
            file_to_index[identity.file] = len(unique_files)
            unique_files.append(identity.file)
        file_indices[index] = file_to_index[identity.file]
        entries[index] = int(identity.entry)
    return unique_files, file_indices, entries


def _metadata_jsonable(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    return {str(k): convert(v) for k, v in metadata.items()}


def build_fixed_hlt_jet_view(
    offline_view: JetView,
    *,
    seed: int,
    params: FixedHLTParams | None = None,
    show_progress: bool = False,
) -> tuple[JetView, Dict[str, np.ndarray], Dict[str, Any]]:
    """Generate a fixed HLT view from one offline split view."""

    params = params or FixedHLTParams()
    hlt_tokens, hlt_mask, diagnostics = build_fixed_hlt_view(
        offline_view.tokens,
        offline_view.mask,
        seed=int(seed),
        params=params,
        show_progress=show_progress,
    )
    hlt_view = JetView(
        tokens=hlt_tokens.astype(np.float32, copy=False),
        mask=hlt_mask.astype(bool, copy=False),
        labels=offline_view.labels.astype(np.int64, copy=True),
        jet_ids=list(offline_view.jet_ids),
        split=offline_view.split,
        metadata={
            "view": "fixed_hlt",
            "source_view": offline_view.metadata.get("view", "offline"),
            "seed": int(seed),
            "hlt_params": fixed_hlt_params_dict(params),
            "source_manifest_hash": offline_view.metadata.get("source_manifest_hash"),
        },
    )
    metadata = build_hlt_metadata(
        hlt_view,
        offline_view=offline_view,
        diagnostics=diagnostics,
        seed=seed,
        params=params,
    )
    hlt_view.metadata.update(metadata)
    return hlt_view, diagnostics, metadata


def build_hlt_metadata(
    hlt_view: JetView,
    *,
    offline_view: JetView,
    diagnostics: Mapping[str, np.ndarray],
    seed: int,
    params: FixedHLTParams | None = None,
) -> Dict[str, Any]:
    """Build metadata proving a split's HLT cache identity and diagnostics."""

    params = params or FixedHLTParams()
    offline_counts = np.sum(offline_view.mask, axis=1).astype(np.int32)
    hlt_counts = np.sum(hlt_view.mask, axis=1).astype(np.int32)
    unique_files, file_indices, entries = _identity_arrays(hlt_view.jet_ids)
    diagnostics_hash = hash_arrays({f"diag_{key}": np.asarray(value) for key, value in diagnostics.items()})
    hlt_content_hash = hash_arrays(
        {
            "tokens": hlt_view.tokens,
            "mask": hlt_view.mask,
            "labels": hlt_view.labels,
            "jet_file_indices": file_indices,
            "jet_entries": entries,
        }
    )
    source_content_hash = hash_arrays(
        {
            "offline_tokens": offline_view.tokens,
            "offline_mask": offline_view.mask,
            "labels": offline_view.labels,
            "jet_file_indices": file_indices,
            "jet_entries": entries,
        }
    )

    return {
        "version": 1,
        "view": "fixed_hlt",
        "split": hlt_view.split,
        "seed": int(seed),
        "hlt_params": fixed_hlt_params_dict(params),
        "source_manifest_hash": offline_view.metadata.get("source_manifest_hash"),
        "source_view": offline_view.metadata.get("view", "offline"),
        "max_constits": int(hlt_view.tokens.shape[1]),
        "raw_token_dim": int(hlt_view.tokens.shape[2]),
        "n_jets": int(hlt_view.tokens.shape[0]),
        "jet_files": unique_files,
        "jet_identity_hash": jet_identity_hash(hlt_view.jet_ids),
        "source_content_hash": source_content_hash,
        "hlt_content_hash": hlt_content_hash,
        "diagnostics_hash": diagnostics_hash,
        "offline_constit_count_summary": _count_summary(offline_counts),
        "hlt_constit_count_summary": _count_summary(hlt_counts),
        "hlt_diagnostics_summary": summarize_hlt_diagnostics(dict(diagnostics)),
        "generator": {
            "module": "jetclass_fixed_hlt",
            "function": "build_fixed_hlt_view",
            "params_class": "FixedHLTParams",
        },
        "leakage_note": (
            "HLT tokens were generated from offline constituents once for this split. "
            "Downstream HLT-side models must consume only this cached HLT view."
        ),
    }


def save_hlt_cache(
    hlt_view: JetView,
    diagnostics: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    cache_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Save one split's HLT tokens, masks, diagnostics, and metadata."""

    array_path, metadata_path = _cache_paths(cache_dir, hlt_view.split)
    if not overwrite and (array_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"HLT cache already exists for split {hlt_view.split}: {array_path}")

    array_path.parent.mkdir(parents=True, exist_ok=True)
    unique_files, file_indices, entries = _identity_arrays(hlt_view.jet_ids)
    if list(unique_files) != list(metadata.get("jet_files", [])):
        raise ValueError("Metadata jet_files do not match HLT view jet identities")

    arrays: Dict[str, np.ndarray] = {
        "tokens": hlt_view.tokens.astype(np.float32, copy=False),
        "mask": hlt_view.mask.astype(bool, copy=False),
        "labels": hlt_view.labels.astype(np.int64, copy=False),
        "jet_file_indices": file_indices,
        "jet_entries": entries,
    }
    for key, value in diagnostics.items():
        arrays[f"diag_{key}"] = np.asarray(value, dtype=np.float32)

    np.savez_compressed(array_path, **arrays)
    saved_metadata = dict(_metadata_jsonable(metadata))
    saved_metadata.update(
        {
            "array_path": str(array_path),
            "metadata_path": str(metadata_path),
            "array_keys": sorted(arrays.keys()),
        }
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(saved_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return saved_metadata


def generate_and_cache_hlt_view(
    offline_view: JetView,
    cache_dir: str | Path,
    *,
    seed: int,
    params: FixedHLTParams | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Generate and save one split's fixed-HLT cache from an offline view."""

    hlt_view, diagnostics, metadata = build_fixed_hlt_jet_view(
        offline_view,
        seed=int(seed),
        params=params,
        show_progress=show_progress,
    )
    return save_hlt_cache(hlt_view, diagnostics, metadata, cache_dir, overwrite=overwrite)


def generate_and_cache_hlt_split(
    manifest: SplitManifest,
    split: str,
    cache_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    seed: int | None = None,
    params: FixedHLTParams | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
    verify_label_branches: bool = False,
    read_chunk_size: int = 50_000,
) -> Dict[str, Any]:
    """Load a Step 2 offline split and cache its fixed HLT view."""

    if split not in SPLIT_ORDER:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_ORDER}")
    seed = DEFAULT_HLT_SEEDS[split] if seed is None else int(seed)
    offline_view = load_offline_view(
        manifest,
        split,
        data_dir=data_dir,
        verify_label_branches=verify_label_branches,
        read_chunk_size=read_chunk_size,
    )
    return generate_and_cache_hlt_view(
        offline_view,
        cache_dir,
        seed=seed,
        params=params,
        overwrite=overwrite,
        show_progress=show_progress,
    )


def load_hlt_metadata(cache_dir: str | Path, split: str) -> Dict[str, Any]:
    """Load one split's fixed-HLT metadata JSON."""

    _, metadata_path = _cache_paths(cache_dir, split)
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cached_hlt_view(
    cache_dir: str | Path,
    split: str,
    *,
    verify_hash: bool = True,
) -> JetView:
    """Load one cached fixed-HLT split as a JetView."""

    array_path, _ = _cache_paths(cache_dir, split)
    metadata = load_hlt_metadata(cache_dir, split)
    with np.load(array_path, allow_pickle=False) as data:
        tokens = data["tokens"].astype(np.float32, copy=False)
        mask = data["mask"].astype(bool, copy=False)
        labels = data["labels"].astype(np.int64, copy=False)
        file_indices = data["jet_file_indices"].astype(np.int64, copy=False)
        entries = data["jet_entries"].astype(np.int64, copy=False)
        diagnostics = {
            key.removeprefix("diag_"): data[key].astype(np.float32, copy=False)
            for key in data.files
            if key.startswith("diag_")
        }

    jet_files = [str(path) for path in metadata["jet_files"]]
    jet_ids = [
        JetIdentity(file=jet_files[int(file_index)], entry=int(entry), label=int(label))
        for file_index, entry, label in zip(file_indices, entries, labels)
    ]

    if verify_hash:
        actual_content_hash = hash_arrays(
            {
                "tokens": tokens,
                "mask": mask,
                "labels": labels,
                "jet_file_indices": file_indices.astype(np.int32),
                "jet_entries": entries,
            }
        )
        if actual_content_hash != metadata.get("hlt_content_hash"):
            raise ValueError(
                f"HLT cache content hash mismatch for {split}: "
                f"{actual_content_hash} != {metadata.get('hlt_content_hash')}"
            )
        actual_identity_hash = jet_identity_hash(jet_ids)
        if actual_identity_hash != metadata.get("jet_identity_hash"):
            raise ValueError(
                f"HLT cache identity hash mismatch for {split}: "
                f"{actual_identity_hash} != {metadata.get('jet_identity_hash')}"
            )

    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={**metadata, "diagnostics": diagnostics},
    )


def audit_hlt_cache(
    manifest: SplitManifest,
    cache_dir: str | Path,
    *,
    splits: Iterable[str] = SPLIT_ORDER,
) -> Dict[str, Any]:
    """Verify cached HLT views match split identities and fixed profile metadata."""

    manifest_sha = manifest_hash(manifest)
    split_reports: Dict[str, Any] = {}
    ok = True
    expected_params = fixed_hlt_params_dict()

    for split in splits:
        split_ok = True
        problems: list[str] = []
        try:
            hlt_view = load_cached_hlt_view(cache_dir, split, verify_hash=True)
            metadata = hlt_view.metadata
        except Exception as exc:  # pragma: no cover - exercised by CLI failures
            ok = False
            split_reports[split] = {"ok": False, "problems": [str(exc)]}
            continue

        expected_ids = manifest.splits[split]
        if len(hlt_view.jet_ids) != len(expected_ids):
            split_ok = False
            problems.append(f"expected {len(expected_ids)} jet ids, found {len(hlt_view.jet_ids)}")
        else:
            for index, (actual, expected) in enumerate(zip(hlt_view.jet_ids, expected_ids)):
                if actual != expected:
                    split_ok = False
                    problems.append(f"jet id mismatch at row {index}: {actual} != {expected}")
                    break

        if metadata.get("source_manifest_hash") not in (None, manifest_sha):
            split_ok = False
            problems.append("source_manifest_hash does not match manifest")
        if metadata.get("hlt_params") != expected_params:
            split_ok = False
            problems.append("HLT parameters do not match FixedHLTParams defaults")
        expected_seed = DEFAULT_HLT_SEEDS.get(split)
        if expected_seed is not None and int(metadata.get("seed", -1)) != int(expected_seed):
            split_ok = False
            problems.append(f"seed is {metadata.get('seed')}, expected {expected_seed}")

        split_reports[split] = {
            "ok": bool(split_ok),
            "problems": problems,
            "n_jets": int(metadata.get("n_jets", 0)),
            "seed": int(metadata.get("seed", -1)),
            "hlt_content_hash": metadata.get("hlt_content_hash"),
            "hlt_diagnostics_summary": metadata.get("hlt_diagnostics_summary"),
        }
        ok = ok and split_ok

    hashes = [report.get("hlt_content_hash") for report in split_reports.values() if report.get("ok")]
    return {
        "ok": bool(ok),
        "manifest_hash": manifest_sha,
        "split_reports": split_reports,
        "all_splits_have_distinct_content_hashes": len(hashes) == len(set(hashes)),
    }

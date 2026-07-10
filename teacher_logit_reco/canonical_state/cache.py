"""Cache writer and audit helpers for canonical jet-state Phi artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetIdentity, JetView, LABEL_NAMES, load_split_manifest, manifest_hash
from teacher_logit_reco.architecture_view_part import load_cached_offline_view

from .config import CANONICAL_STATE_SPLIT_ORDER, canonical_state_split_sizes
from .layout import (
    CANONICAL_STATE_FIELD_NAMES,
    CanonicalJetStateLayout,
    default_canonical_jet_state_layout,
)
from .phi import CANONICAL_STATE_PHI_BUILDER_VERSION, build_canonical_jet_state_phi_from_view


CANONICAL_STATE_PHI_CACHE_CONTRACT = "canonical_state_phi_cache_v1"
CANONICAL_STATE_PHI_HLT_SOURCE = "hlt"
CANONICAL_STATE_PHI_OFFLINE_SOURCE = "offline"
CANONICAL_STATE_PHI_HLT_SPLITS: tuple[str, ...] = CANONICAL_STATE_SPLIT_ORDER
CANONICAL_STATE_PHI_OFFLINE_PRIMARY_SPLITS: tuple[str, ...] = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
)
CANONICAL_STATE_PHI_OFFLINE_ORACLE_SPLITS: tuple[str, ...] = CANONICAL_STATE_SPLIT_ORDER


@dataclass(frozen=True)
class CanonicalPhiCache:
    """Loaded Phi cache arrays and metadata."""

    phi_tokens: np.ndarray
    state_mask: np.ndarray
    labels: np.ndarray
    jet_ids: tuple[JetIdentity, ...]
    token_type_ids: np.ndarray
    scale_ids: np.ndarray
    slot_ids: np.ndarray
    valid_particle_counts: np.ndarray
    masked_particle_counts: np.ndarray
    state_valid_counts: np.ndarray
    finite_particle_fraction: np.ndarray
    split: str
    source_view: str
    metadata: dict[str, Any]


def normalize_phi_source_view(source_view: str) -> str:
    value = str(source_view).strip().lower()
    aliases = {
        "hlt": CANONICAL_STATE_PHI_HLT_SOURCE,
        "fixed_hlt": CANONICAL_STATE_PHI_HLT_SOURCE,
        "phi_hlt": CANONICAL_STATE_PHI_HLT_SOURCE,
        "offline": CANONICAL_STATE_PHI_OFFLINE_SOURCE,
        "off": CANONICAL_STATE_PHI_OFFLINE_SOURCE,
        "phi_off": CANONICAL_STATE_PHI_OFFLINE_SOURCE,
        "phi_offline": CANONICAL_STATE_PHI_OFFLINE_SOURCE,
    }
    if value not in aliases:
        raise ValueError(f"unknown canonical Phi source_view {source_view!r}; expected hlt or offline")
    return aliases[value]


def phi_cache_paths(cache_dir: str | Path, split: str, source_view: str) -> tuple[Path, Path]:
    source = normalize_phi_source_view(source_view)
    root = Path(cache_dir)
    return (
        root / f"{split}_phi_{source}.npz",
        root / f"{split}_phi_{source}_metadata.json",
    )


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
    return (
        jet_files,
        np.asarray(file_indices, dtype=np.int32),
        np.asarray(entries, dtype=np.int64),
    )


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


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = {name: 0 for name in LABEL_NAMES}
    for label in np.asarray(labels, dtype=np.int64):
        counts[LABEL_NAMES[int(label)]] += 1
    return counts


def _source_cache_hash(view: JetView, source_view: str) -> tuple[str, str | None]:
    metadata = dict(view.metadata)
    if source_view == CANONICAL_STATE_PHI_HLT_SOURCE:
        return "hlt_content_hash", metadata.get("hlt_content_hash")
    return "offline_content_hash", metadata.get("offline_content_hash") or metadata.get("content_hash")


def _content_hash(arrays: Mapping[str, np.ndarray]) -> str:
    return hash_arrays(
        {
            "phi_tokens": np.asarray(arrays["phi_tokens"], dtype=np.float32),
            "state_mask": np.asarray(arrays["state_mask"], dtype=bool),
            "labels": np.asarray(arrays["labels"], dtype=np.int64),
            "jet_file_indices": np.asarray(arrays["jet_file_indices"], dtype=np.int32),
            "jet_entries": np.asarray(arrays["jet_entries"], dtype=np.int64),
            "token_type_ids": np.asarray(arrays["token_type_ids"], dtype=np.int16),
            "scale_ids": np.asarray(arrays["scale_ids"], dtype=np.int16),
            "slot_ids": np.asarray(arrays["slot_ids"], dtype=np.int16),
        }
    )


def _diagnostic_arrays(phi_output: Any) -> dict[str, np.ndarray]:
    diagnostics = dict(phi_output.diagnostics)
    return {
        "valid_particle_counts": np.asarray(diagnostics.get("valid_particle_counts", []), dtype=np.int32),
        "masked_particle_counts": np.asarray(diagnostics.get("masked_particle_counts", []), dtype=np.int32),
        "state_valid_counts": np.asarray(diagnostics.get("state_valid_counts", []), dtype=np.int32),
        "finite_particle_fraction": np.asarray(diagnostics.get("finite_particle_fraction", []), dtype=np.float32),
    }


def save_canonical_phi_cache(
    view: JetView,
    cache_dir: str | Path,
    *,
    source_view: str,
    layout: CanonicalJetStateLayout | None = None,
    overwrite: bool = False,
    allow_final_test_offline_oracle: bool = False,
) -> dict[str, Any]:
    """Build and persist one split's canonical Phi cache."""

    source = normalize_phi_source_view(source_view)
    if source == CANONICAL_STATE_PHI_OFFLINE_SOURCE and view.split == "final_test" and not allow_final_test_offline_oracle:
        raise ValueError("offline final_test Phi is oracle-only; pass allow_final_test_offline_oracle=True to write it")
    layout = default_canonical_jet_state_layout() if layout is None else layout
    array_path, metadata_path = phi_cache_paths(cache_dir, view.split, source)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (array_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"canonical Phi cache already exists for {source}/{view.split}: {array_path}")

    phi_output = build_canonical_jet_state_phi_from_view(
        view,
        layout=layout,
        source_metadata={"source_view": source},
    )
    jet_files, file_indices, entries = _jet_identity_arrays(view.jet_ids)
    token_type_ids = np.asarray(layout.token_type_ids, dtype=np.int16)
    scale_ids = np.asarray(layout.scale_ids, dtype=np.int16)
    slot_ids = np.asarray(layout.slot_ids, dtype=np.int16)
    diagnostic_arrays = _diagnostic_arrays(phi_output)
    arrays = {
        "phi_tokens": np.asarray(phi_output.phi_tokens, dtype=np.float32),
        "state_mask": np.asarray(phi_output.state_mask, dtype=bool),
        "labels": np.asarray(view.labels, dtype=np.int64),
        "jet_file_indices": file_indices,
        "jet_entries": entries,
        "token_type_ids": token_type_ids,
        "scale_ids": scale_ids,
        "slot_ids": slot_ids,
        **diagnostic_arrays,
    }
    np.savez_compressed(array_path, **arrays)
    phi_content_hash = _content_hash(arrays)
    identity_hash = jet_identity_hash(view.jet_ids)
    source_hash_name, source_hash_value = _source_cache_hash(view, source)
    oracle_only = bool(source == CANONICAL_STATE_PHI_OFFLINE_SOURCE and view.split == "final_test")
    metadata = {
        "cache_contract": CANONICAL_STATE_PHI_CACHE_CONTRACT,
        "split": str(view.split),
        "source_view": source,
        "oracle_only": oracle_only,
        "allowed_for_primary_training": not oracle_only,
        "array_path": str(array_path),
        "metadata_path": str(metadata_path),
        "n_jets": int(arrays["phi_tokens"].shape[0]),
        "k_state": int(arrays["phi_tokens"].shape[1]),
        "d_phi": int(arrays["phi_tokens"].shape[2]),
        "label_names": list(LABEL_NAMES),
        "class_counts": _class_counts(arrays["labels"]),
        "jet_files": jet_files,
        "jet_identity_hash": identity_hash,
        "source_manifest_hash": view.metadata.get("source_manifest_hash"),
        "source_cache_hash_name": source_hash_name,
        "source_cache_hash": source_hash_value,
        source_hash_name: source_hash_value,
        "source_jet_identity_hash": view.metadata.get("jet_identity_hash") or identity_hash,
        "phi_content_hash": phi_content_hash,
        "layout_version": layout.config.layout_version,
        "field_names": list(layout.config.field_names),
        "token_names": list(layout.token_names),
        "token_type_ids": token_type_ids.astype(int).tolist(),
        "scale_ids": scale_ids.astype(int).tolist(),
        "slot_ids": slot_ids.astype(int).tolist(),
        "layout_metadata": layout.to_dict(),
        "phi_builder_version": CANONICAL_STATE_PHI_BUILDER_VERSION,
        "normalization_metadata": {
            "feature_scales": {name: float(layout.config.feature_scales[name]) for name in layout.config.field_names},
            "residual_scales": {name: float(layout.config.residual_scales[name]) for name in layout.config.field_names},
        },
        "diagnostics_summary": {
            "valid_particle_counts": _summary(diagnostic_arrays["valid_particle_counts"]),
            "masked_particle_counts": _summary(diagnostic_arrays["masked_particle_counts"]),
            "state_valid_counts": _summary(diagnostic_arrays["state_valid_counts"]),
            "finite_particle_fraction": _summary(diagnostic_arrays["finite_particle_fraction"]),
            "all_finite": bool(np.isfinite(arrays["phi_tokens"]).all()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_canonical_phi_cache(
    cache_dir: str | Path,
    split: str,
    *,
    source_view: str,
    verify_hash: bool = True,
    expected_layout: CanonicalJetStateLayout | None = None,
    allow_oracle_final_test: bool = False,
) -> CanonicalPhiCache:
    """Load and validate one canonical Phi cache split."""

    source = normalize_phi_source_view(source_view)
    array_path, metadata_path = phi_cache_paths(cache_dir, split, source)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("cache_contract") != CANONICAL_STATE_PHI_CACHE_CONTRACT:
        raise ValueError(f"unexpected Phi cache contract for {source}/{split}: {metadata.get('cache_contract')}")
    if str(metadata.get("split")) != str(split):
        raise ValueError(f"Phi cache split mismatch: {metadata.get('split')} != {split}")
    if normalize_phi_source_view(metadata.get("source_view", source)) != source:
        raise ValueError(f"Phi cache source_view mismatch: {metadata.get('source_view')} != {source}")
    if bool(metadata.get("oracle_only")) and not bool(allow_oracle_final_test):
        raise ValueError(f"{source}/{split} Phi cache is oracle-only and cannot be loaded for primary training")
    if metadata.get("phi_builder_version") != CANONICAL_STATE_PHI_BUILDER_VERSION:
        raise ValueError(
            f"Phi builder version mismatch: {metadata.get('phi_builder_version')} != {CANONICAL_STATE_PHI_BUILDER_VERSION}"
        )
    layout = expected_layout or default_canonical_jet_state_layout()
    if metadata.get("layout_version") != layout.config.layout_version:
        raise ValueError(f"layout version mismatch: {metadata.get('layout_version')} != {layout.config.layout_version}")
    if tuple(metadata.get("field_names") or []) != tuple(layout.config.field_names):
        raise ValueError("Phi cache field_names do not match expected canonical layout")
    if tuple(metadata.get("token_names") or []) != tuple(layout.token_names):
        raise ValueError("Phi cache token_names do not match expected canonical layout")
    with np.load(array_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
        phi_tokens = arrays["phi_tokens"].astype(np.float32, copy=False)
        state_mask = arrays["state_mask"].astype(bool, copy=False)
        labels = arrays["labels"].astype(np.int64, copy=False)
        file_indices = arrays["jet_file_indices"].astype(np.int64, copy=False)
        entries = arrays["jet_entries"].astype(np.int64, copy=False)
        token_type_ids = arrays["token_type_ids"].astype(np.int16, copy=False)
        scale_ids = arrays["scale_ids"].astype(np.int16, copy=False)
        slot_ids = arrays["slot_ids"].astype(np.int16, copy=False)
        valid_particle_counts = arrays["valid_particle_counts"].astype(np.int32, copy=False)
        masked_particle_counts = arrays["masked_particle_counts"].astype(np.int32, copy=False)
        state_valid_counts = arrays["state_valid_counts"].astype(np.int32, copy=False)
        finite_particle_fraction = arrays["finite_particle_fraction"].astype(np.float32, copy=False)

    if phi_tokens.shape != (int(metadata.get("n_jets", -1)), layout.k_state, layout.d_phi):
        raise ValueError(f"Phi token shape {phi_tokens.shape} does not match metadata/layout")
    if state_mask.shape != phi_tokens.shape[:2]:
        raise ValueError(f"state_mask shape {state_mask.shape} does not match phi_tokens")
    if tuple(token_type_ids.astype(int).tolist()) != tuple(layout.token_type_ids):
        raise ValueError("token_type_ids do not match expected canonical layout")
    if tuple(scale_ids.astype(int).tolist()) != tuple(layout.scale_ids):
        raise ValueError("scale_ids do not match expected canonical layout")
    if tuple(slot_ids.astype(int).tolist()) != tuple(layout.slot_ids):
        raise ValueError("slot_ids do not match expected canonical layout")
    jet_ids = _jet_ids_from_arrays(metadata["jet_files"], file_indices, entries, labels)
    if verify_hash:
        content_hash = _content_hash(
            {
                "phi_tokens": phi_tokens,
                "state_mask": state_mask,
                "labels": labels,
                "jet_file_indices": file_indices.astype(np.int32),
                "jet_entries": entries,
                "token_type_ids": token_type_ids,
                "scale_ids": scale_ids,
                "slot_ids": slot_ids,
            }
        )
        if content_hash != metadata.get("phi_content_hash"):
            raise ValueError(f"Phi cache content hash mismatch for {source}/{split}")
        identity_hash = jet_identity_hash(jet_ids)
        if identity_hash != metadata.get("jet_identity_hash"):
            raise ValueError(f"Phi cache identity hash mismatch for {source}/{split}")
    return CanonicalPhiCache(
        phi_tokens=phi_tokens,
        state_mask=state_mask,
        labels=labels,
        jet_ids=jet_ids,
        token_type_ids=token_type_ids,
        scale_ids=scale_ids,
        slot_ids=slot_ids,
        valid_particle_counts=valid_particle_counts,
        masked_particle_counts=masked_particle_counts,
        state_valid_counts=state_valid_counts,
        finite_particle_fraction=finite_particle_fraction,
        split=str(split),
        source_view=source,
        metadata=metadata,
    )


def build_phi_cache_from_hlt_cache(
    hlt_cache_dir: str | Path,
    phi_cache_dir: str | Path,
    *,
    splits: Sequence[str] = CANONICAL_STATE_PHI_HLT_SPLITS,
    overwrite: bool = False,
    layout: CanonicalJetStateLayout | None = None,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for split in splits:
        view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True)
        reports[split] = save_canonical_phi_cache(
            view,
            phi_cache_dir,
            source_view=CANONICAL_STATE_PHI_HLT_SOURCE,
            layout=layout,
            overwrite=overwrite,
        )
    return reports


def build_phi_cache_from_offline_cache(
    offline_cache_dir: str | Path,
    phi_cache_dir: str | Path,
    *,
    splits: Sequence[str] = CANONICAL_STATE_PHI_OFFLINE_PRIMARY_SPLITS,
    overwrite: bool = False,
    layout: CanonicalJetStateLayout | None = None,
    allow_final_test_offline_oracle: bool = False,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for split in splits:
        view = load_cached_offline_view(offline_cache_dir, split, verify_hash=True)
        reports[split] = save_canonical_phi_cache(
            view,
            phi_cache_dir,
            source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE,
            layout=layout,
            overwrite=overwrite,
            allow_final_test_offline_oracle=allow_final_test_offline_oracle,
        )
    return reports


def audit_canonical_phi_cache(
    cache_dir: str | Path,
    *,
    source_view: str,
    manifest_path: str | Path,
    splits: Sequence[str],
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_layout: CanonicalJetStateLayout | None = None,
    allow_oracle_final_test: bool = False,
) -> dict[str, Any]:
    source = normalize_phi_source_view(source_view)
    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    expected_sizes = canonical_state_split_sizes(expected_split_sizes)
    layout = expected_layout or default_canonical_jet_state_layout()
    split_reports: dict[str, Any] = {}
    problems: list[str] = []
    for split in splits:
        try:
            cache = load_canonical_phi_cache(
                cache_dir,
                split,
                source_view=source,
                expected_layout=layout,
                allow_oracle_final_test=allow_oracle_final_test,
            )
            metadata = cache.metadata
            expected_ids = tuple(manifest.splits[split])
            item_problems: list[str] = []
            if int(cache.phi_tokens.shape[0]) != int(expected_sizes[split]):
                item_problems.append(f"n_jets is {cache.phi_tokens.shape[0]}, expected {expected_sizes[split]}")
            if cache.jet_ids != expected_ids:
                item_problems.append("Phi cache jet identities do not match split manifest")
            if metadata.get("source_manifest_hash") != manifest_sha:
                item_problems.append("source_manifest_hash does not match manifest hash")
            if not metadata.get("source_cache_hash"):
                item_problems.append("missing source cache hash")
            if metadata.get("source_jet_identity_hash") != metadata.get("jet_identity_hash"):
                item_problems.append("source jet identity hash does not match Phi jet identity hash")
            if source == CANONICAL_STATE_PHI_OFFLINE_SOURCE and split == "final_test" and not allow_oracle_final_test:
                item_problems.append("offline final_test Phi is oracle-only and not allowed for primary audit")
            item = {
                "ok": not item_problems,
                "split": split,
                "source_view": source,
                "n_jets": int(cache.phi_tokens.shape[0]),
                "k_state": int(cache.phi_tokens.shape[1]),
                "d_phi": int(cache.phi_tokens.shape[2]),
                "phi_content_hash": metadata.get("phi_content_hash"),
                "jet_identity_hash": metadata.get("jet_identity_hash"),
                "source_manifest_hash": metadata.get("source_manifest_hash"),
                "source_cache_hash_name": metadata.get("source_cache_hash_name"),
                "source_cache_hash": metadata.get("source_cache_hash"),
                "oracle_only": bool(metadata.get("oracle_only")),
                "field_names": metadata.get("field_names"),
                "layout_version": metadata.get("layout_version"),
                "phi_builder_version": metadata.get("phi_builder_version"),
                "diagnostics_summary": metadata.get("diagnostics_summary"),
                "problems": item_problems,
            }
        except Exception as exc:
            item = {"ok": False, "split": split, "source_view": source, "problems": [str(exc)]}
        split_reports[split] = item
        for problem in item.get("problems") or []:
            problems.append(f"{source}/{split}: {problem}")
    return {
        "ok": bool(not problems and all(bool(item.get("ok")) for item in split_reports.values())),
        "cache_contract": CANONICAL_STATE_PHI_CACHE_CONTRACT,
        "source_view": source,
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "splits": list(splits),
        "layout_version": layout.config.layout_version,
        "field_names": list(layout.config.field_names),
        "split_reports": split_reports,
        "problems": problems,
    }


def audit_canonical_phi_pair_alignment(
    phi_hlt_cache_dir: str | Path,
    phi_offline_cache_dir: str | Path,
    *,
    manifest_path: str | Path,
    splits: Sequence[str] = CANONICAL_STATE_PHI_OFFLINE_PRIMARY_SPLITS,
    expected_split_sizes: Mapping[str, int] | None = None,
    allow_oracle_final_test: bool = False,
) -> dict[str, Any]:
    hlt_audit = audit_canonical_phi_cache(
        phi_hlt_cache_dir,
        source_view=CANONICAL_STATE_PHI_HLT_SOURCE,
        manifest_path=manifest_path,
        splits=splits,
        expected_split_sizes=expected_split_sizes,
    )
    offline_audit = audit_canonical_phi_cache(
        phi_offline_cache_dir,
        source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE,
        manifest_path=manifest_path,
        splits=splits,
        expected_split_sizes=expected_split_sizes,
        allow_oracle_final_test=allow_oracle_final_test,
    )
    split_reports: dict[str, Any] = {}
    problems: list[str] = []
    for split in splits:
        hlt_item = hlt_audit.get("split_reports", {}).get(split, {})
        off_item = offline_audit.get("split_reports", {}).get(split, {})
        item_problems: list[str] = []
        if hlt_item.get("jet_identity_hash") != off_item.get("jet_identity_hash"):
            item_problems.append("HLT/offline Phi jet_identity_hash mismatch")
        if hlt_item.get("field_names") != off_item.get("field_names"):
            item_problems.append("HLT/offline Phi field_names mismatch")
        if hlt_item.get("layout_version") != off_item.get("layout_version"):
            item_problems.append("HLT/offline Phi layout_version mismatch")
        item = {
            "ok": bool(hlt_item.get("ok") and off_item.get("ok") and not item_problems),
            "split": split,
            "hlt_phi_content_hash": hlt_item.get("phi_content_hash"),
            "offline_phi_content_hash": off_item.get("phi_content_hash"),
            "jet_identity_hash": hlt_item.get("jet_identity_hash"),
            "problems": item_problems,
        }
        split_reports[split] = item
        for problem in item_problems:
            problems.append(f"{split}: {problem}")
    problems = list(hlt_audit.get("problems") or []) + list(offline_audit.get("problems") or []) + problems
    return {
        "ok": bool(hlt_audit.get("ok") and offline_audit.get("ok") and not problems),
        "cache_contract": CANONICAL_STATE_PHI_CACHE_CONTRACT,
        "manifest_path": str(manifest_path),
        "manifest_hash": hlt_audit.get("manifest_hash"),
        "splits": list(splits),
        "hlt_audit": hlt_audit,
        "offline_audit": offline_audit,
        "split_reports": split_reports,
        "problems": problems,
    }

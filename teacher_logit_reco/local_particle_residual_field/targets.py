"""Target-cache builder for local per-particle residual fields.

The cache is an HLT-only query grid with offline-supervised targets.  For each
valid HLT particle, it summarizes how offline local energy flow differs from
HLT local energy flow at several radii.  The target is intentionally local and
per-particle; it does not require one-to-one HLT/offline particle matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetIdentity, LABEL_NAMES, load_split_manifest, manifest_hash
from teacher_logit_reco.architecture_view_part import load_cached_offline_view


LOCAL_PARTICLE_RESIDUAL_FIELD_CACHE_CONTRACT = "local_particle_residual_field_target_cache_v1"
LOCAL_PARTICLE_RESIDUAL_FIELD_BUILDER_VERSION = "local_particle_residual_field_builder_v1"
LOCAL_PARTICLE_RESIDUAL_FIELD_PRIMARY_SPLITS: tuple[str, ...] = (
    "model_train",
    "model_val",
    "stack_val",
)
LOCAL_PARTICLE_RESIDUAL_FIELD_ALL_SPLITS: tuple[str, ...] = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)
DEFAULT_LOCAL_RESIDUAL_RADII: tuple[float, ...] = (0.02, 0.05, 0.10)
PID_FIELD_NAMES: tuple[str, ...] = (
    "charged_hadron",
    "neutral_hadron",
    "photon",
    "electron",
    "muon",
)
PID_TOKEN_SLICE = slice(5, 10)
EPSILON = 1.0e-6


@dataclass(frozen=True)
class LocalParticleResidualFieldCache:
    """Loaded local residual-field target cache."""

    target_fields: np.ndarray
    target_mask: np.ndarray
    labels: np.ndarray
    jet_ids: tuple[JetIdentity, ...]
    field_names: tuple[str, ...]
    field_groups: dict[str, list[int]]
    radii: tuple[float, ...]
    split: str
    metadata: dict[str, Any]


def local_particle_residual_field_cache_paths(cache_dir: str | Path, split: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    return (
        root / f"{split}_local_particle_residual_fields.npz",
        root / f"{split}_local_particle_residual_fields_metadata.json",
    )


def _radius_tag(radius: float) -> str:
    text = f"{float(radius):.4g}".replace("-", "m").replace(".", "p")
    return f"r{text}"


def local_particle_residual_field_layout(
    radii: Sequence[float] = DEFAULT_LOCAL_RESIDUAL_RADII,
) -> tuple[list[str], dict[str, list[int]], list[str]]:
    """Return field names, field-group index map, and radius tags."""

    field_names: list[str] = []
    field_groups: dict[str, list[int]] = {
        "pt_density": [],
        "centroid": [],
        "multiplicity": [],
        "composition": [],
        "reliability": [],
    }
    radius_tags = [_radius_tag(radius) for radius in radii]

    def add(group: str, name: str) -> None:
        field_groups[group].append(len(field_names))
        field_names.append(name)

    for tag in radius_tags:
        for name in (
            "delta_log_pt_sum",
            "delta_pt_frac",
            "missing_pt_frac",
            "extra_pt_frac",
            "delta_log_energy_sum",
        ):
            add("pt_density", f"{tag}.{name}")
        for name in ("delta_eta_centroid", "delta_phi_centroid", "delta_r_centroid"):
            add("centroid", f"{tag}.{name}")
        for name in ("delta_log_n", "missing_n_frac"):
            add("multiplicity", f"{tag}.{name}")
        for pid_name in PID_FIELD_NAMES:
            add("composition", f"{tag}.delta_{pid_name}_frac")

    for name in (
        "flag.is_merged_token",
        "flag.has_missing_local_activity",
        "flag.has_large_local_shift",
        "local_reliability_score",
        "target_uncertainty_scale",
    ):
        add("reliability", name)

    return field_names, field_groups, radius_tags


def _wrap_phi(delta: np.ndarray) -> np.ndarray:
    return ((delta + np.pi) % (2.0 * np.pi)) - np.pi


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / np.maximum(den, EPSILON)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = {name: 0 for name in LABEL_NAMES}
    for label in np.asarray(labels, dtype=np.int64):
        counts[LABEL_NAMES[int(label)]] += 1
    return counts


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


def _zero_baseline_metrics(
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    field_names: Sequence[str],
    field_groups: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool)
    if not bool(np.any(valid)):
        return {
            "n_valid_particles": 0,
            "mae": None,
            "mse": None,
            "per_field_mae": {},
            "per_group_mae": {},
        }
    arr = np.asarray(targets, dtype=np.float64)[valid]
    abs_arr = np.abs(arr)
    sq_arr = arr * arr
    per_field_mae = {
        str(name): float(abs_arr[:, index].mean())
        for index, name in enumerate(field_names)
    }
    per_group_mae = {
        str(group): float(abs_arr[:, list(indices)].mean())
        for group, indices in field_groups.items()
        if len(indices) > 0
    }
    return {
        "n_valid_particles": int(valid.sum()),
        "mae": float(abs_arr.mean()),
        "mse": float(sq_arr.mean()),
        "per_field_mae": per_field_mae,
        "per_group_mae": per_group_mae,
    }


def _local_summaries_for_radius(
    query_tokens: np.ndarray,
    query_mask: np.ndarray,
    source_tokens: np.ndarray,
    source_mask: np.ndarray,
    radius: float,
) -> dict[str, np.ndarray]:
    """Summarize source particles around each HLT query particle."""

    q_eta = query_tokens[:, :, 1].astype(np.float32)
    q_phi = query_tokens[:, :, 2].astype(np.float32)
    src_pt = source_tokens[:, :, 0].astype(np.float32)
    src_eta = source_tokens[:, :, 1].astype(np.float32)
    src_phi = source_tokens[:, :, 2].astype(np.float32)
    src_energy = source_tokens[:, :, 3].astype(np.float32)
    src_pid = source_tokens[:, :, PID_TOKEN_SLICE].astype(np.float32)

    d_eta = src_eta[:, None, :] - q_eta[:, :, None]
    d_phi = _wrap_phi(src_phi[:, None, :] - q_phi[:, :, None]).astype(np.float32)
    d_r2 = d_eta * d_eta + d_phi * d_phi
    radius = max(float(radius), EPSILON)
    weights = np.exp(-0.5 * d_r2 / float(radius * radius)).astype(np.float32)
    weights *= source_mask[:, None, :].astype(np.float32)
    weights *= query_mask[:, :, None].astype(np.float32)

    sum_w = weights.sum(axis=2, dtype=np.float32)
    pt_w = weights * src_pt[:, None, :]
    sum_pt = pt_w.sum(axis=2, dtype=np.float32)
    sum_energy = (weights * src_energy[:, None, :]).sum(axis=2, dtype=np.float32)
    eta_centroid = _safe_div((pt_w * d_eta).sum(axis=2, dtype=np.float32), sum_pt).astype(np.float32)
    phi_centroid = _safe_div((pt_w * d_phi).sum(axis=2, dtype=np.float32), sum_pt).astype(np.float32)
    pid_pt = (pt_w[:, :, :, None] * src_pid[:, None, :, :]).sum(axis=2, dtype=np.float32)
    pid_frac = _safe_div(pid_pt, sum_pt[:, :, None]).astype(np.float32)

    return {
        "sum_w": sum_w.astype(np.float32),
        "sum_pt": sum_pt.astype(np.float32),
        "sum_energy": sum_energy.astype(np.float32),
        "eta_centroid": eta_centroid,
        "phi_centroid": phi_centroid,
        "pid_frac": pid_frac,
    }


def compute_local_particle_residual_fields(
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
    offline_tokens: np.ndarray,
    offline_mask: np.ndarray,
    *,
    radii: Sequence[float] = DEFAULT_LOCAL_RESIDUAL_RADII,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list[int]], dict[str, Any]]:
    """Compute oracle local residual fields for HLT query particles."""

    hlt_tokens = np.asarray(hlt_tokens, dtype=np.float32)
    offline_tokens = np.asarray(offline_tokens, dtype=np.float32)
    hlt_mask = np.asarray(hlt_mask, dtype=bool)
    offline_mask = np.asarray(offline_mask, dtype=bool)
    if hlt_tokens.ndim != 3 or offline_tokens.ndim != 3:
        raise ValueError("hlt_tokens and offline_tokens must have shape [N, P, D]")
    if hlt_mask.shape != hlt_tokens.shape[:2]:
        raise ValueError("hlt_mask must match hlt_tokens leading shape")
    if offline_mask.shape != offline_tokens.shape[:2]:
        raise ValueError("offline_mask must match offline_tokens leading shape")
    if hlt_tokens.shape[0] != offline_tokens.shape[0]:
        raise ValueError("HLT and offline token arrays must have the same number of jets")

    radii = tuple(float(radius) for radius in radii)
    field_names, field_groups, _ = local_particle_residual_field_layout(radii)
    output = np.zeros((hlt_tokens.shape[0], hlt_tokens.shape[1], len(field_names)), dtype=np.float32)
    jet_pt = np.sum(hlt_tokens[:, :, 0] * hlt_mask.astype(np.float32), axis=1, dtype=np.float32)
    field_index = 0
    medium_radius_index = min(1, len(radii) - 1)
    medium_values: dict[str, np.ndarray] = {}

    for radius_index, radius in enumerate(radii):
        hlt_summary = _local_summaries_for_radius(hlt_tokens, hlt_mask, hlt_tokens, hlt_mask, radius)
        offline_summary = _local_summaries_for_radius(hlt_tokens, hlt_mask, offline_tokens, offline_mask, radius)

        delta_log_pt = np.log(offline_summary["sum_pt"] + EPSILON) - np.log(hlt_summary["sum_pt"] + EPSILON)
        delta_pt = offline_summary["sum_pt"] - hlt_summary["sum_pt"]
        missing_pt = np.maximum(delta_pt, 0.0)
        extra_pt = np.maximum(-delta_pt, 0.0)
        delta_log_energy = np.log(offline_summary["sum_energy"] + EPSILON) - np.log(hlt_summary["sum_energy"] + EPSILON)
        delta_eta = offline_summary["eta_centroid"] - hlt_summary["eta_centroid"]
        delta_phi = offline_summary["phi_centroid"] - hlt_summary["phi_centroid"]
        delta_r = np.sqrt(delta_eta * delta_eta + delta_phi * delta_phi)
        delta_log_n = np.log(offline_summary["sum_w"] + 1.0) - np.log(hlt_summary["sum_w"] + 1.0)
        missing_n = np.maximum(offline_summary["sum_w"] - hlt_summary["sum_w"], 0.0)
        missing_n_frac = missing_n / np.maximum(offline_summary["sum_w"] + 1.0, 1.0)
        delta_pid = offline_summary["pid_frac"] - hlt_summary["pid_frac"]

        values = (
            delta_log_pt,
            _safe_div(delta_pt, jet_pt[:, None]),
            _safe_div(missing_pt, jet_pt[:, None]),
            _safe_div(extra_pt, jet_pt[:, None]),
            delta_log_energy,
            delta_eta,
            delta_phi,
            delta_r,
            delta_log_n,
            missing_n_frac,
        )
        for value in values:
            output[:, :, field_index] = np.asarray(value, dtype=np.float32)
            field_index += 1
        output[:, :, field_index : field_index + len(PID_FIELD_NAMES)] = delta_pid.astype(np.float32)
        field_index += len(PID_FIELD_NAMES)

        if radius_index == medium_radius_index:
            medium_values = {
                "delta_log_pt": delta_log_pt.astype(np.float32),
                "missing_pt_frac": _safe_div(missing_pt, jet_pt[:, None]).astype(np.float32),
                "delta_r": delta_r.astype(np.float32),
                "delta_log_n": delta_log_n.astype(np.float32),
                "hlt_n": hlt_summary["sum_w"].astype(np.float32),
                "offline_n": offline_summary["sum_w"].astype(np.float32),
            }

    if not medium_values:
        raise ValueError("at least one radius is required")
    is_merged = (medium_values["offline_n"] > medium_values["hlt_n"] + 1.5).astype(np.float32)
    has_missing_activity = (medium_values["missing_pt_frac"] > 0.02).astype(np.float32)
    has_large_shift = (medium_values["delta_r"] > 0.02).astype(np.float32)
    uncertainty = np.sqrt(
        medium_values["delta_log_pt"] ** 2
        + medium_values["delta_r"] ** 2
        + medium_values["delta_log_n"] ** 2
    ).astype(np.float32)
    reliability = np.exp(-np.minimum(uncertainty, 20.0)).astype(np.float32)
    for value in (is_merged, has_missing_activity, has_large_shift, reliability, uncertainty):
        output[:, :, field_index] = value.astype(np.float32)
        field_index += 1

    output *= hlt_mask[:, :, None].astype(np.float32)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    diagnostics = {
        "valid_hlt_particles": hlt_mask.sum(axis=1).astype(np.int32),
        "valid_offline_particles": offline_mask.sum(axis=1).astype(np.int32),
        "mean_abs_target_by_field": np.mean(np.abs(output), axis=(0, 1), dtype=np.float64).astype(np.float64),
        "all_finite": bool(np.isfinite(output).all()),
    }
    return output, hlt_mask.copy(), field_names, field_groups, diagnostics


def _content_hash(arrays: Mapping[str, np.ndarray]) -> str:
    return hash_arrays(
        {
            "target_fields": np.asarray(arrays["target_fields"], dtype=np.float32),
            "target_mask": np.asarray(arrays["target_mask"], dtype=bool),
            "labels": np.asarray(arrays["labels"], dtype=np.int64),
            "jet_file_indices": np.asarray(arrays["jet_file_indices"], dtype=np.int32),
            "jet_entries": np.asarray(arrays["jet_entries"], dtype=np.int64),
        }
    )


def _validate_pair_and_manifest(
    *,
    split: str,
    manifest_path: str | Path,
    hlt_view: Any,
    offline_view: Any,
) -> tuple[str, tuple[JetIdentity, ...]]:
    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    expected_ids = tuple(manifest.splits[str(split)])
    hlt_ids = tuple(hlt_view.jet_ids)
    offline_ids = tuple(offline_view.jet_ids)
    if hlt_ids != offline_ids:
        raise ValueError(f"HLT/offline jet identities do not match for split {split}")
    if hlt_ids != expected_ids:
        raise ValueError(f"cache jet identities do not match split manifest for split {split}")
    if not np.array_equal(np.asarray(hlt_view.labels), np.asarray(offline_view.labels)):
        raise ValueError(f"HLT/offline labels do not match for split {split}")
    hlt_manifest = hlt_view.metadata.get("source_manifest_hash")
    offline_manifest = offline_view.metadata.get("source_manifest_hash")
    if hlt_manifest not in (None, manifest_sha):
        raise ValueError(f"HLT source_manifest_hash mismatch for split {split}: {hlt_manifest} != {manifest_sha}")
    if offline_manifest not in (None, manifest_sha):
        raise ValueError(f"offline source_manifest_hash mismatch for split {split}: {offline_manifest} != {manifest_sha}")
    return manifest_sha, hlt_ids


def build_local_particle_residual_field_cache(
    *,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
    output_cache_dir: str | Path,
    split: str,
    radii: Sequence[float] = DEFAULT_LOCAL_RESIDUAL_RADII,
    overwrite: bool = False,
    allow_final_test_targets: bool = False,
    chunk_size: int = 1024,
    target_dtype: str = "float16",
) -> dict[str, Any]:
    """Build and persist one split's local residual-field target cache."""

    split = str(split)
    if split == "final_test" and not bool(allow_final_test_targets):
        raise ValueError("offline final_test residual fields are oracle-only; pass allow_final_test_targets=True")
    array_path, metadata_path = local_particle_residual_field_cache_paths(output_cache_dir, split)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (array_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"local residual-field cache already exists for split {split}: {array_path}")

    hlt_view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True)
    offline_view = load_cached_offline_view(offline_cache_dir, split, verify_hash=True)
    manifest_sha, jet_ids = _validate_pair_and_manifest(
        split=split,
        manifest_path=manifest_path,
        hlt_view=hlt_view,
        offline_view=offline_view,
    )

    n_jets = int(hlt_view.tokens.shape[0])
    chunk = int(chunk_size)
    if chunk <= 0:
        chunk = n_jets
    field_names, field_groups, radius_tags = local_particle_residual_field_layout(radii)
    dtype = np.dtype(str(target_dtype))
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("target_dtype must be float16 or float32")
    target_fields = np.empty((n_jets, hlt_view.tokens.shape[1], len(field_names)), dtype=dtype)
    target_mask = np.empty((n_jets, hlt_view.tokens.shape[1]), dtype=bool)
    valid_hlt_particles = np.empty(n_jets, dtype=np.int32)
    valid_offline_particles = np.empty(n_jets, dtype=np.int32)
    all_finite = True

    for start in range(0, n_jets, chunk):
        stop = min(start + chunk, n_jets)
        fields, mask, _, _, diagnostics = compute_local_particle_residual_fields(
            hlt_view.tokens[start:stop],
            hlt_view.mask[start:stop],
            offline_view.tokens[start:stop],
            offline_view.mask[start:stop],
            radii=radii,
        )
        target_fields[start:stop] = fields.astype(dtype, copy=False)
        target_mask[start:stop] = mask
        valid_hlt_particles[start:stop] = diagnostics["valid_hlt_particles"]
        valid_offline_particles[start:stop] = diagnostics["valid_offline_particles"]
        all_finite = all_finite and bool(diagnostics["all_finite"])

    jet_files, file_indices, entries = _jet_identity_arrays(jet_ids)
    arrays = {
        "target_fields": target_fields,
        "target_mask": target_mask,
        "labels": np.asarray(hlt_view.labels, dtype=np.int64),
        "jet_file_indices": file_indices,
        "jet_entries": entries,
        "valid_hlt_particles": valid_hlt_particles,
        "valid_offline_particles": valid_offline_particles,
    }
    np.savez_compressed(array_path, **arrays)
    target_content_hash = _content_hash(arrays)
    identity_hash = jet_identity_hash(jet_ids)
    zero_metrics = _zero_baseline_metrics(
        target_fields,
        target_mask,
        field_names=field_names,
        field_groups=field_groups,
    )
    metadata = {
        "cache_contract": LOCAL_PARTICLE_RESIDUAL_FIELD_CACHE_CONTRACT,
        "builder_version": LOCAL_PARTICLE_RESIDUAL_FIELD_BUILDER_VERSION,
        "split": split,
        "oracle_only": bool(split == "final_test"),
        "allowed_for_primary_training": bool(split != "final_test"),
        "array_path": str(array_path),
        "metadata_path": str(metadata_path),
        "n_jets": int(n_jets),
        "max_particles": int(target_fields.shape[1]),
        "field_dim": int(target_fields.shape[2]),
        "target_dtype": str(dtype),
        "radii": [float(radius) for radius in radii],
        "radius_tags": radius_tags,
        "field_names": list(field_names),
        "field_groups": {str(key): [int(index) for index in value] for key, value in field_groups.items()},
        "label_names": list(LABEL_NAMES),
        "class_counts": _class_counts(arrays["labels"]),
        "jet_files": jet_files,
        "jet_identity_hash": identity_hash,
        "source_manifest_hash": manifest_sha,
        "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
        "offline_content_hash": offline_view.metadata.get("offline_content_hash")
        or offline_view.metadata.get("content_hash"),
        "hlt_jet_identity_hash": hlt_view.metadata.get("jet_identity_hash") or identity_hash,
        "offline_jet_identity_hash": offline_view.metadata.get("jet_identity_hash") or identity_hash,
        "target_content_hash": target_content_hash,
        "zero_baseline_metrics": zero_metrics,
        "diagnostics_summary": {
            "valid_hlt_particles": _summary(valid_hlt_particles),
            "valid_offline_particles": _summary(valid_offline_particles),
            "target_mask_valid_fraction": float(np.mean(target_mask)),
            "all_finite": bool(all_finite and np.isfinite(target_fields).all()),
        },
        "target_semantics": {
            "query": "valid HLT particles",
            "source_offline": "soft local neighborhoods around each HLT particle",
            "matching": "no one-to-one particle matching; all targets are local energy-flow summaries",
            "delta_log_pt_sum": "log(local offline pt sum + eps) - log(local HLT pt sum + eps)",
            "composition": list(PID_FIELD_NAMES),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def build_local_particle_residual_field_caches(
    *,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
    output_cache_dir: str | Path,
    splits: Sequence[str] = LOCAL_PARTICLE_RESIDUAL_FIELD_PRIMARY_SPLITS,
    radii: Sequence[float] = DEFAULT_LOCAL_RESIDUAL_RADII,
    overwrite: bool = False,
    allow_final_test_targets: bool = False,
    chunk_size: int = 1024,
    target_dtype: str = "float16",
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for split in splits:
        reports[str(split)] = build_local_particle_residual_field_cache(
            manifest_path=manifest_path,
            hlt_cache_dir=hlt_cache_dir,
            offline_cache_dir=offline_cache_dir,
            output_cache_dir=output_cache_dir,
            split=str(split),
            radii=radii,
            overwrite=overwrite,
            allow_final_test_targets=allow_final_test_targets,
            chunk_size=chunk_size,
            target_dtype=target_dtype,
        )
    return reports


def load_local_particle_residual_field_cache(
    cache_dir: str | Path,
    split: str,
    *,
    verify_hash: bool = True,
    allow_final_test_targets: bool = False,
) -> LocalParticleResidualFieldCache:
    array_path, metadata_path = local_particle_residual_field_cache_paths(cache_dir, split)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("cache_contract") != LOCAL_PARTICLE_RESIDUAL_FIELD_CACHE_CONTRACT:
        raise ValueError(f"unexpected local residual-field cache contract: {metadata.get('cache_contract')}")
    if str(metadata.get("split")) != str(split):
        raise ValueError(f"local residual-field split mismatch: {metadata.get('split')} != {split}")
    if bool(metadata.get("oracle_only")) and not bool(allow_final_test_targets):
        raise ValueError(f"{split} residual-field cache is oracle-only and cannot be loaded for primary training")

    with np.load(array_path, allow_pickle=False) as data:
        target_fields = data["target_fields"].astype(np.float32, copy=False)
        target_mask = data["target_mask"].astype(bool, copy=False)
        labels = data["labels"].astype(np.int64, copy=False)
        file_indices = data["jet_file_indices"].astype(np.int64, copy=False)
        entries = data["jet_entries"].astype(np.int64, copy=False)
    jet_ids = _jet_ids_from_arrays(metadata["jet_files"], file_indices, entries, labels)
    if verify_hash:
        actual_hash = _content_hash(
            {
                "target_fields": target_fields,
                "target_mask": target_mask,
                "labels": labels,
                "jet_file_indices": file_indices.astype(np.int32),
                "jet_entries": entries,
            }
        )
        if actual_hash != metadata.get("target_content_hash"):
            raise ValueError(
                f"local residual-field content hash mismatch for {split}: "
                f"{actual_hash} != {metadata.get('target_content_hash')}"
            )
        actual_identity_hash = jet_identity_hash(jet_ids)
        if actual_identity_hash != metadata.get("jet_identity_hash"):
            raise ValueError(
                f"local residual-field identity hash mismatch for {split}: "
                f"{actual_identity_hash} != {metadata.get('jet_identity_hash')}"
            )

    return LocalParticleResidualFieldCache(
        target_fields=target_fields,
        target_mask=target_mask,
        labels=labels,
        jet_ids=jet_ids,
        field_names=tuple(str(name) for name in metadata["field_names"]),
        field_groups={
            str(key): [int(index) for index in value]
            for key, value in dict(metadata["field_groups"]).items()
        },
        radii=tuple(float(radius) for radius in metadata["radii"]),
        split=str(split),
        metadata=metadata,
    )

"""Build and audit deterministic HLT2 caches derived only from PD10 HLT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from jetclass_fixed_hlt import (
    FixedHLTV2Params,
    RAW_DIM,
    apply_hlt_single_jet_v2_realistic,
    summarize_hlt_diagnostics,
)
from jetclass_fresh.hlt_cache import (
    hash_arrays,
    jet_identity_hash,
    load_cached_hlt_view,
    save_hlt_cache,
)
from jetclass_fresh.jetclass_data import JetIdentity, JetView, SplitManifest, manifest_hash

from .config import (
    HLT_SDV_ALLOWED_INPUTS,
    HLT_SDV_DEPLOYMENT_INPUTS,
    HLT_SDV_HLT2_PROFILE_NAME,
    HLT_SDV_HLT2_PROFILE_VERSION,
    HLT_SDV_IDENTITY_STRENGTH,
    HLT_SDV_ROOT_DIRNAME,
    build_hlt_sdv_cache_strengths,
    default_hlt_sdv_experiment_layout,
    hlt_sdv_strength_tag,
    normalize_hlt_sdv_strength,
)

HLT2_CACHE_CONTRACT = "hlt_self_dualview_hlt2_cache_v1"
HLT2_CACHE_AUDIT_REPORT = "hlt2_cache_audit_report.json"
HLT2_CACHE_AUDIT_SUMMARY = "hlt2_cache_audit_summary.md"
HLT2_DEFAULT_BASE_SEED = 710_053
HLT2_AUDIT_SPLITS = ("model_train", "model_val", "final_test")


@dataclass(frozen=True)
class HLTSecondDegradeProfile:
    """Strength-1 knobs for the mild second-layer HLT degradation."""

    extra_pt_threshold: float = 0.08
    extra_merge_radius: float = 0.0025
    extra_merge_probability: float = 0.15
    extra_reassign_scale: float = 0.08
    extra_smear_scale: float = 0.10
    extra_eff_plateau_loss_barrel: float = 0.012
    extra_eff_plateau_loss_endcap: float = 0.018
    extra_turnon_pt_barrel: float = 0.08
    extra_turnon_pt_endcap: float = 0.12
    extra_eff_width_pt_barrel: float = 0.03
    extra_eff_width_pt_endcap: float = 0.04
    extra_density_loss_scale: float = 0.008
    extra_jet_quality_sigma: float = 0.010
    extra_tail_probability_base: float = 0.0008
    extra_tail_probability_eta: float = 0.0006
    extra_tail_probability_density: float = 0.0006


def hlt2_profile_dict(profile: HLTSecondDegradeProfile | None = None) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(profile or HLTSecondDegradeProfile()).items()}


def hlt2_params_from_strength(
    strength: float | int | str,
    *,
    profile: HLTSecondDegradeProfile | None = None,
) -> FixedHLTV2Params:
    """Convert the HLT2 profile and strength into existing mild-HLT primitives."""

    s = normalize_hlt_sdv_strength(strength)
    p = profile or HLTSecondDegradeProfile()
    return FixedHLTV2Params(
        profile_name=HLT_SDV_HLT2_PROFILE_NAME,
        profile_version=HLT_SDV_HLT2_PROFILE_VERSION,
        hlt_pt_threshold=float(p.extra_pt_threshold) * s,
        merge_radius=float(p.extra_merge_radius) * s,
        merge_probability=float(p.extra_merge_probability) * s,
        eff_plateau_barrel=float(np.clip(1.0 - float(p.extra_eff_plateau_loss_barrel) * s, 0.0, 1.0)),
        eff_plateau_endcap=float(np.clip(1.0 - float(p.extra_eff_plateau_loss_endcap) * s, 0.0, 1.0)),
        eff_turnon_pt_barrel=float(p.extra_turnon_pt_barrel) * s,
        eff_turnon_pt_endcap=float(p.extra_turnon_pt_endcap) * s,
        eff_width_pt_barrel=float(p.extra_eff_width_pt_barrel) * s,
        eff_width_pt_endcap=float(p.extra_eff_width_pt_endcap) * s,
        density_loss_scale=float(p.extra_density_loss_scale) * s,
        jet_quality_sigma=float(p.extra_jet_quality_sigma) * s,
        smear_scale=float(p.extra_smear_scale) * s,
        tail_probability_base=float(p.extra_tail_probability_base) * s,
        tail_probability_eta=float(p.extra_tail_probability_eta) * s,
        tail_probability_density=float(p.extra_tail_probability_density) * s,
        reassign_scale=float(p.extra_reassign_scale) * s,
    )


def _identity_arrays(jet_ids: Sequence[JetIdentity]) -> tuple[list[str], np.ndarray, np.ndarray]:
    unique_files: list[str] = []
    file_to_index: dict[str, int] = {}
    file_indices = np.zeros((len(jet_ids),), dtype=np.int32)
    entries = np.zeros((len(jet_ids),), dtype=np.int64)
    for index, identity in enumerate(jet_ids):
        if identity.file not in file_to_index:
            file_to_index[identity.file] = len(unique_files)
            unique_files.append(identity.file)
        file_indices[index] = file_to_index[identity.file]
        entries[index] = int(identity.entry)
    return unique_files, file_indices, entries


def count_summary(counts: np.ndarray) -> dict[str, Any]:
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


def hlt_view_content_hash(view: JetView) -> str:
    _, file_indices, entries = _identity_arrays(view.jet_ids)
    return hash_arrays(
        {
            "tokens": view.tokens,
            "mask": view.mask,
            "labels": view.labels,
            "jet_file_indices": file_indices,
            "jet_entries": entries,
        }
    )


def hlt2_seed_for_identity(
    identity: JetIdentity,
    *,
    split: str,
    strength: float | int | str,
    base_seed: int = HLT2_DEFAULT_BASE_SEED,
    profile_name: str = HLT_SDV_HLT2_PROFILE_NAME,
    profile_version: str = HLT_SDV_HLT2_PROFILE_VERSION,
) -> int:
    """Return a deterministic per-jet seed independent of processing order."""

    payload = "\0".join(
        [
            profile_name,
            profile_version,
            split,
            hlt_sdv_strength_tag(strength),
            str(int(base_seed)),
            identity.file,
            str(int(identity.entry)),
            str(int(identity.label)),
        ]
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def slice_jet_view(view: JetView, max_jets: int | None = None) -> JetView:
    if max_jets is None:
        return view
    n = min(max(int(max_jets), 0), int(view.tokens.shape[0]))
    return JetView(
        tokens=view.tokens[:n].copy(),
        mask=view.mask[:n].copy(),
        labels=view.labels[:n].copy(),
        jet_ids=list(view.jet_ids[:n]),
        split=view.split,
        metadata={**view.metadata, "sliced_max_jets": int(max_jets), "sliced_n_jets": int(n)},
    )


def build_hlt2_view_from_hlt_view(
    hlt_view: JetView,
    *,
    strength: float | int | str,
    base_seed: int = HLT2_DEFAULT_BASE_SEED,
    profile: HLTSecondDegradeProfile | None = None,
    show_progress: bool = False,
) -> tuple[JetView, dict[str, np.ndarray], dict[str, Any]]:
    """Derive a deterministic HLT2 view from an already-cached HLT view."""

    strength_value = normalize_hlt_sdv_strength(strength)
    params = hlt2_params_from_strength(strength_value, profile=profile)
    n_jets, max_constits, raw_dim = hlt_view.tokens.shape
    if raw_dim != RAW_DIM:
        raise ValueError(f"Expected raw token dim {RAW_DIM}, got {raw_dim}.")

    out_tokens = np.zeros_like(hlt_view.tokens, dtype=np.float32)
    out_mask = np.zeros_like(hlt_view.mask, dtype=bool)
    diag_keys = (
        "n_offline",
        "n_after_eff",
        "n_after_threshold",
        "n_after_merge",
        "drop_eff",
        "drop_threshold",
        "drop_merge",
        "drop_total",
        "merge_count",
    )
    diag_values = {key: np.zeros((n_jets,), dtype=np.float32) for key in diag_keys}

    iterator: Iterable[int] = range(n_jets)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"Building HLT2 {hlt_sdv_strength_tag(strength_value)}")
        except Exception:
            pass

    for index in iterator:
        seed = hlt2_seed_for_identity(
            hlt_view.jet_ids[index],
            split=hlt_view.split,
            strength=strength_value,
            base_seed=base_seed,
        )
        rng = np.random.RandomState(seed)
        tokens_i, mask_i, diag_i = apply_hlt_single_jet_v2_realistic(
            hlt_view.tokens[index],
            hlt_view.mask[index],
            params,
            rng,
            max_constits,
        )
        out_tokens[index] = tokens_i
        out_mask[index] = mask_i
        for key in diag_keys:
            diag_values[key][index] = float(diag_i.get(key, 0.0))

    hlt2_view = JetView(
        tokens=out_tokens.astype(np.float32, copy=False),
        mask=out_mask.astype(bool, copy=False),
        labels=hlt_view.labels.astype(np.int64, copy=True),
        jet_ids=list(hlt_view.jet_ids),
        split=hlt_view.split,
        metadata={
            "view": "hlt2",
            "source_view": "HLT",
            "derived_view": "HLT2",
            "hlt2_strength": strength_value,
            "hlt2_profile_name": HLT_SDV_HLT2_PROFILE_NAME,
            "hlt2_profile_version": HLT_SDV_HLT2_PROFILE_VERSION,
        },
    )
    return hlt2_view, diag_values, {
        "hlt2_params": asdict(params),
        "hlt2_profile": hlt2_profile_dict(profile),
        "hlt2_strength": strength_value,
        "hlt2_seed": int(base_seed),
    }


def build_hlt2_metadata(
    hlt2_view: JetView,
    *,
    source_hlt_view: JetView,
    diagnostics: Mapping[str, np.ndarray],
    split_manifest_hash: str,
    source_hlt_cache_dir: str | Path,
    strength: float | int | str,
    base_seed: int = HLT2_DEFAULT_BASE_SEED,
    profile: HLTSecondDegradeProfile | None = None,
) -> dict[str, Any]:
    strength_value = normalize_hlt_sdv_strength(strength)
    source_files, source_file_indices, source_entries = _identity_arrays(source_hlt_view.jet_ids)
    _, hlt2_file_indices, hlt2_entries = _identity_arrays(hlt2_view.jet_ids)
    source_hlt_content_hash = hlt_view_content_hash(source_hlt_view)
    hlt2_content_hash = hlt_view_content_hash(hlt2_view)
    source_counts = np.sum(source_hlt_view.mask, axis=1).astype(np.int32)
    hlt2_counts = np.sum(hlt2_view.mask, axis=1).astype(np.int32)
    source_total = max(float(np.sum(source_counts)), 1.0)
    drop_fraction = float(max(np.sum(source_counts) - np.sum(hlt2_counts), 0) / source_total)
    diagnostics_hash = hash_arrays({f"diag_{key}": np.asarray(value) for key, value in diagnostics.items()})

    return {
        "version": 1,
        "contract": HLT2_CACHE_CONTRACT,
        "view": "hlt2",
        "split": hlt2_view.split,
        "source_view": "HLT",
        "derived_view": "HLT2",
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "uses_offline_particles": False,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "deterministic_by_jet_identity": True,
        "source_hlt_cache_dir": str(Path(source_hlt_cache_dir)),
        "source_hlt_content_hash": source_hlt_content_hash,
        "source_hlt_full_content_hash": source_hlt_view.metadata.get("hlt_content_hash"),
        "source_hlt_jet_identity_hash": jet_identity_hash(source_hlt_view.jet_ids),
        "source_hlt_full_jet_identity_hash": source_hlt_view.metadata.get("jet_identity_hash"),
        "source_hlt_array_path": source_hlt_view.metadata.get("array_path"),
        "source_manifest_hash": split_manifest_hash,
        "split_manifest_hash": split_manifest_hash,
        "hlt2_profile_name": HLT_SDV_HLT2_PROFILE_NAME,
        "hlt2_profile_version": HLT_SDV_HLT2_PROFILE_VERSION,
        "hlt2_profile": hlt2_profile_dict(profile),
        "hlt2_strength": strength_value,
        "hlt2_strength_tag": hlt_sdv_strength_tag(strength_value),
        "hlt2_seed": int(base_seed),
        "hlt2_params": asdict(hlt2_params_from_strength(strength_value, profile=profile)),
        "max_constits": int(hlt2_view.tokens.shape[1]),
        "raw_token_dim": int(hlt2_view.tokens.shape[2]),
        "n_jets": int(hlt2_view.tokens.shape[0]),
        "jet_files": source_files,
        "jet_identity_hash": jet_identity_hash(hlt2_view.jet_ids),
        "source_content_hash": source_hlt_content_hash,
        "hlt_content_hash": hlt2_content_hash,
        "hlt2_content_hash": hlt2_content_hash,
        "diagnostics_hash": diagnostics_hash,
        "source_hlt_particle_count_summary": count_summary(source_counts),
        "hlt2_particle_count_summary": count_summary(hlt2_counts),
        "hlt2_drop_summary": {
            "total_source_particles": int(np.sum(source_counts)),
            "total_hlt2_particles": int(np.sum(hlt2_counts)),
            "additional_drop_fraction_relative_to_hlt": drop_fraction,
            "mean_count_delta": float(np.mean(hlt2_counts - source_counts)) if len(hlt2_counts) else 0.0,
        },
        "hlt2_diagnostics_summary": summarize_hlt_diagnostics(dict(diagnostics)),
        "generator": {
            "module": "teacher_logit_reco.hlt_self_dualview.hlt2_cache",
            "function": "build_hlt2_view_from_hlt_view",
            "profile": HLT_SDV_HLT2_PROFILE_NAME,
        },
        "parent_hlt_identity_arrays_hash": hash_arrays(
            {
                "labels": source_hlt_view.labels,
                "jet_file_indices": source_file_indices,
                "jet_entries": source_entries,
            }
        ),
        "hlt2_identity_arrays_hash": hash_arrays(
            {
                "labels": hlt2_view.labels,
                "jet_file_indices": hlt2_file_indices,
                "jet_entries": hlt2_entries,
            }
        ),
        "leakage_note": (
            "HLT2 was generated only from the cached PD10 HLT view. No offline particles, "
            "teacher logits, or teacher representations are referenced."
        ),
    }


def generate_and_cache_hlt2_split(
    source_hlt_cache_dir: str | Path,
    hlt2_cache_dir: str | Path,
    *,
    split: str,
    split_manifest_hash: str,
    strength: float | int | str,
    base_seed: int = HLT2_DEFAULT_BASE_SEED,
    profile: HLTSecondDegradeProfile | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
    max_jets: int | None = None,
) -> dict[str, Any]:
    source_view = load_cached_hlt_view(source_hlt_cache_dir, split, verify_hash=True)
    source_view = slice_jet_view(source_view, max_jets=max_jets)
    hlt2_view, diagnostics, _ = build_hlt2_view_from_hlt_view(
        source_view,
        strength=strength,
        base_seed=base_seed,
        profile=profile,
        show_progress=show_progress,
    )
    metadata = build_hlt2_metadata(
        hlt2_view,
        source_hlt_view=source_view,
        diagnostics=diagnostics,
        split_manifest_hash=split_manifest_hash,
        source_hlt_cache_dir=source_hlt_cache_dir,
        strength=strength,
        base_seed=base_seed,
        profile=profile,
    )
    return save_hlt_cache(hlt2_view, diagnostics, metadata, hlt2_cache_dir, overwrite=overwrite)


def generate_and_cache_hlt2_view(
    manifest: SplitManifest,
    source_hlt_cache_dir: str | Path,
    hlt2_cache_dir: str | Path,
    *,
    strength: float | int | str,
    splits: Iterable[str] = HLT2_AUDIT_SPLITS,
    base_seed: int = HLT2_DEFAULT_BASE_SEED,
    profile: HLTSecondDegradeProfile | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
    max_jets_by_split: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    split_manifest_hash = manifest_hash(manifest)
    reports: dict[str, Any] = {}
    for split in splits:
        reports[split] = generate_and_cache_hlt2_split(
            source_hlt_cache_dir,
            hlt2_cache_dir,
            split=split,
            split_manifest_hash=split_manifest_hash,
            strength=strength,
            base_seed=base_seed,
            profile=profile,
            overwrite=overwrite,
            show_progress=show_progress,
            max_jets=None if max_jets_by_split is None else max_jets_by_split.get(split),
        )
    return {
        "ok": True,
        "contract": HLT2_CACHE_CONTRACT,
        "hlt2_cache_dir": str(Path(hlt2_cache_dir)),
        "source_hlt_cache_dir": str(Path(source_hlt_cache_dir)),
        "split_manifest_hash": split_manifest_hash,
        "hlt2_strength": normalize_hlt_sdv_strength(strength),
        "hlt2_strength_tag": hlt_sdv_strength_tag(strength),
        "splits": list(splits),
        "reports": reports,
    }


def hlt2_split_audit_problems(
    *,
    split: str,
    parent_view: JetView,
    hlt2_view: JetView,
    metadata: Mapping[str, Any],
    manifest: SplitManifest,
    manifest_sha: str,
    strength: float,
    expected_n_jets: int | None = None,
) -> list[str]:
    problems: list[str] = []
    expected_n = len(manifest.splits.get(split, [])) if expected_n_jets is None else int(expected_n_jets)
    if int(hlt2_view.tokens.shape[0]) != expected_n:
        problems.append(f"n_jets is {hlt2_view.tokens.shape[0]}, expected {expected_n}")
    if not np.array_equal(parent_view.labels, hlt2_view.labels):
        problems.append("HLT2 labels differ from parent HLT labels")
    if jet_identity_hash(parent_view.jet_ids) != jet_identity_hash(hlt2_view.jet_ids):
        problems.append("HLT2 jet identity hash differs from parent HLT")
    if metadata.get("contract") != HLT2_CACHE_CONTRACT:
        problems.append(f"contract is {metadata.get('contract')!r}, expected {HLT2_CACHE_CONTRACT!r}")
    if metadata.get("source_view") != "HLT" or metadata.get("derived_view") != "HLT2":
        problems.append("source_view/derived_view metadata is not HLT -> HLT2")
    if metadata.get("allowed_inputs") != HLT_SDV_ALLOWED_INPUTS:
        problems.append("allowed_inputs is not HLT_only")
    if bool(metadata.get("uses_offline_particles")):
        problems.append("uses_offline_particles must be false")
    if metadata.get("student_deployment_inputs") != HLT_SDV_DEPLOYMENT_INPUTS:
        problems.append("student_deployment_inputs metadata is wrong")
    if metadata.get("split_manifest_hash") != manifest_sha:
        problems.append("split_manifest_hash does not match manifest")
    if metadata.get("source_hlt_content_hash") != hlt_view_content_hash(parent_view):
        problems.append("source_hlt_content_hash does not match parent HLT rows")
    if metadata.get("source_hlt_jet_identity_hash") != jet_identity_hash(parent_view.jet_ids):
        problems.append("source_hlt_jet_identity_hash does not match parent HLT rows")
    if metadata.get("hlt2_content_hash") != metadata.get("hlt_content_hash"):
        problems.append("hlt2_content_hash and hlt_content_hash differ")
    if metadata.get("hlt2_content_hash") != hlt_view_content_hash(hlt2_view):
        problems.append("hlt2_content_hash does not match HLT2 arrays")
    if not bool(metadata.get("deterministic_by_jet_identity")):
        problems.append("deterministic_by_jet_identity must be true")
    if not np.isfinite(hlt2_view.tokens).all():
        problems.append("HLT2 tokens contain non-finite values")
    if hlt2_view.mask.dtype != np.bool_:
        problems.append("HLT2 mask is not boolean")

    source_counts = np.sum(parent_view.mask, axis=1)
    hlt2_counts = np.sum(hlt2_view.mask, axis=1)
    total_source_particles = int(np.sum(source_counts))
    if strength == HLT_SDV_IDENTITY_STRENGTH:
        if not np.array_equal(parent_view.tokens, hlt2_view.tokens):
            problems.append("s0p00 HLT2 tokens are not exactly equal to parent HLT tokens")
        if not np.array_equal(parent_view.mask, hlt2_view.mask):
            problems.append("s0p00 HLT2 mask is not exactly equal to parent HLT mask")
        if metadata.get("hlt2_content_hash") != metadata.get("source_hlt_content_hash"):
            problems.append("s0p00 HLT2 content hash differs from parent HLT hash")
    elif total_source_particles > 0:
        if metadata.get("hlt2_content_hash") == metadata.get("source_hlt_content_hash"):
            problems.append("HLT2 content hash should differ from parent HLT for strength > 0")
        if float(np.mean(hlt2_counts)) > float(np.mean(source_counts)) + 1.0e-6:
            problems.append("mean HLT2 particle count is greater than mean parent HLT count")
        if float(np.mean(source_counts)) > 0 and float(np.mean(hlt2_counts)) < 0.50 * float(np.mean(source_counts)):
            problems.append("mean HLT2 particle count collapsed below 50% of parent HLT")

    return problems


def audit_hlt2_cache(
    manifest: SplitManifest,
    source_hlt_cache_dir: str | Path,
    hlt2_cache_dir: str | Path,
    *,
    strength: float | int | str,
    splits: Iterable[str] = HLT2_AUDIT_SPLITS,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    strength_value = normalize_hlt_sdv_strength(strength)
    manifest_sha = manifest_hash(manifest)
    split_reports: dict[str, Any] = {}
    ok = True
    for split in splits:
        try:
            parent_view = load_cached_hlt_view(source_hlt_cache_dir, split, verify_hash=True)
            if expected_counts is not None and split in expected_counts:
                parent_view = slice_jet_view(parent_view, max_jets=int(expected_counts[split]))
            hlt2_view = load_cached_hlt_view(hlt2_cache_dir, split, verify_hash=True)
            metadata = hlt2_view.metadata
            problems = hlt2_split_audit_problems(
                split=split,
                parent_view=parent_view,
                hlt2_view=hlt2_view,
                metadata=metadata,
                manifest=manifest,
                manifest_sha=manifest_sha,
                strength=strength_value,
                expected_n_jets=None if expected_counts is None else expected_counts.get(split),
            )
            source_counts = np.sum(parent_view.mask, axis=1)
            hlt2_counts = np.sum(hlt2_view.mask, axis=1)
            split_ok = not problems
            split_reports[split] = {
                "ok": bool(split_ok),
                "problems": problems,
                "n_jets": int(hlt2_view.tokens.shape[0]),
                "source_hlt_content_hash": metadata.get("source_hlt_content_hash"),
                "source_hlt_jet_identity_hash": metadata.get("source_hlt_jet_identity_hash"),
                "hlt2_content_hash": metadata.get("hlt2_content_hash"),
                "hlt2_strength": metadata.get("hlt2_strength"),
                "hlt2_seed": metadata.get("hlt2_seed"),
                "source_hlt_particle_count_summary": count_summary(source_counts),
                "hlt2_particle_count_summary": count_summary(hlt2_counts),
                "hlt2_drop_summary": metadata.get("hlt2_drop_summary"),
                "hlt2_diagnostics_summary": metadata.get("hlt2_diagnostics_summary"),
            }
            ok = ok and split_ok
        except Exception as exc:  # pragma: no cover - exercised by compute-side failures
            ok = False
            split_reports[split] = {"ok": False, "problems": [str(exc)], "n_jets": 0}

    problems: list[str] = []
    for split, report in split_reports.items():
        for problem in report.get("problems") or []:
            problems.append(f"{split}: {problem}")
    return {
        "ok": bool(ok),
        "contract": HLT2_CACHE_CONTRACT,
        "source_view": "HLT",
        "derived_view": "HLT2",
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "uses_offline_particles": False,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "source_hlt_cache_dir": str(Path(source_hlt_cache_dir)),
        "hlt2_cache_dir": str(Path(hlt2_cache_dir)),
        "split_manifest_hash": manifest_sha,
        "hlt2_profile_name": HLT_SDV_HLT2_PROFILE_NAME,
        "hlt2_profile_version": HLT_SDV_HLT2_PROFILE_VERSION,
        "hlt2_strength": strength_value,
        "hlt2_strength_tag": hlt_sdv_strength_tag(strength_value),
        "splits": list(splits),
        "split_reports": split_reports,
        "problems": problems,
    }


def write_hlt2_audit_reports(
    manifest: SplitManifest,
    source_hlt_cache_dir: str | Path,
    hlt2_cache_dir: str | Path,
    output_dir: str | Path,
    *,
    strength: float | int | str,
    splits: Iterable[str] = HLT2_AUDIT_SPLITS,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    report = audit_hlt2_cache(
        manifest,
        source_hlt_cache_dir,
        hlt2_cache_dir,
        strength=strength,
        splits=splits,
        expected_counts=expected_counts,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / HLT2_CACHE_AUDIT_REPORT
    summary_path = output / HLT2_CACHE_AUDIT_SUMMARY
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary_path.write_text(build_hlt2_audit_summary(report), encoding="utf-8")
    return {
        "ok": bool(report["ok"]),
        "output_dir": str(output),
        "audit_report": str(report_path),
        "summary": str(summary_path),
    }


def build_hlt2_audit_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "# HLT Self-Dualview HLT2 Cache Audit",
        "",
        f"overall_ok: {bool(report.get('ok'))}",
        f"contract: `{report.get('contract')}`",
        f"strength: `{report.get('hlt2_strength_tag')}`",
        f"uses_offline_particles: {report.get('uses_offline_particles')}",
        "",
        "| split | jets | HLT mean count | HLT2 mean count | drop fraction | HLT2 hash | ok |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for split, item in (report.get("split_reports") or {}).items():
        source_summary = item.get("source_hlt_particle_count_summary") or {}
        hlt2_summary = item.get("hlt2_particle_count_summary") or {}
        drop_summary = item.get("hlt2_drop_summary") or {}
        digest = str(item.get("hlt2_content_hash") or "")
        digest_text = f"`{digest[:12]}...`" if digest else "missing"
        lines.append(
            f"| {split} | {item.get('n_jets')} | "
            f"{float(source_summary.get('mean', 0.0)):.4f} | "
            f"{float(hlt2_summary.get('mean', 0.0)):.4f} | "
            f"{float(drop_summary.get('additional_drop_fraction_relative_to_hlt', 0.0)):.6f} | "
            f"{digest_text} | {item.get('ok')} |"
        )
    if report.get("problems"):
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in report["problems"])
    return "\n".join(lines) + "\n"


def expected_counts_from_maxima(
    *,
    model_train: int | None = None,
    model_val: int | None = None,
    final_test: int | None = None,
) -> dict[str, int] | None:
    values = {
        "model_train": model_train,
        "model_val": model_val,
        "final_test": final_test,
    }
    if not any(value is not None for value in values.values()):
        return None
    return {split: int(value) for split, value in values.items() if value is not None}


def default_hlt2_audit_dir(pd10_root: str | Path, strength: float | int | str) -> Path:
    return Path(pd10_root) / HLT_SDV_ROOT_DIRNAME / "audits" / "hlt2_cache" / hlt_sdv_strength_tag(strength)


def default_hlt2_cache_dir(pd10_root: str | Path, strength: float | int | str) -> Path:
    layout = default_hlt_sdv_experiment_layout(
        output_root=Path(pd10_root).parent,
        pd10_experiment_name=Path(pd10_root).name,
    )
    return layout.hlt2_cache_dir(strength)


__all__ = [
    "HLT2_AUDIT_SPLITS",
    "HLT2_CACHE_AUDIT_REPORT",
    "HLT2_CACHE_AUDIT_SUMMARY",
    "HLT2_CACHE_CONTRACT",
    "HLT2_DEFAULT_BASE_SEED",
    "HLTSecondDegradeProfile",
    "audit_hlt2_cache",
    "build_hlt2_audit_summary",
    "build_hlt2_metadata",
    "build_hlt2_view_from_hlt_view",
    "build_hlt_sdv_cache_strengths",
    "count_summary",
    "default_hlt2_audit_dir",
    "default_hlt2_cache_dir",
    "expected_counts_from_maxima",
    "generate_and_cache_hlt2_split",
    "generate_and_cache_hlt2_view",
    "hlt2_params_from_strength",
    "hlt2_profile_dict",
    "hlt2_seed_for_identity",
    "hlt2_split_audit_problems",
    "hlt_view_content_hash",
    "slice_jet_view",
    "write_hlt2_audit_reports",
]

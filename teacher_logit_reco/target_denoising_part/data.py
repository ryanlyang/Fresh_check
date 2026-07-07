"""Alignment-aware datasets for target-conditioned particle denoising.

Step 1 deliberately stops at data construction: paired HLT/offline views,
per-particle residual targets, canonical ParT batch inputs, and strict metadata
checks.  The denoising model itself is introduced in later steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.hlt_cache import (
    fixed_hlt_params_from_profile,
    jet_identity_hash,
    load_cached_hlt_view,
    normalize_hlt_profile,
)
from jetclass_fresh.jetclass_data import (
    JetIdentity,
    JetView,
    load_offline_view,
    load_split_manifest,
    manifest_hash,
)
from jetclass_fresh.part_inputs import (
    PF_FEATURE_NAMES,
    build_particle_transformer_inputs_from_tokens,
)


TARGET_DENOISING_STEP1 = "target_conditioned_denoising_part_step1_dataset"
TARGET_DENOISING_DATASET_CONTRACT = "target_conditioned_pairwise_denoising_dataset_v1"
TARGET_DENOISING_ALLOWED_INPUTS = "HLT_only_at_inference_offline_targets_train_time_only"

DENOISING_TARGET_NAMES = (
    "delta_log_pt",
    "delta_eta",
    "delta_phi",
    "delta_log_energy",
)

TARGET_STATUS_NO_TARGET = 0
TARGET_STATUS_DIRECT = 1
TARGET_SUMMARY_PREVIEW_JETS = 4096

ALIGNMENT_MODE_RANK_DIRECT = "rank_direct"
ALIGNMENT_MODE_ALIGNED_DIRECT = "aligned_direct"
SUPPORTED_ALIGNMENT_MODES = (ALIGNMENT_MODE_RANK_DIRECT, ALIGNMENT_MODE_ALIGNED_DIRECT)

EPS = 1.0e-8


def wrap_delta_phi_np(delta_phi: np.ndarray) -> np.ndarray:
    """Wrap angular differences to [-pi, pi)."""

    value = np.asarray(delta_phi, dtype=np.float32)
    return ((value + np.float32(math.pi)) % np.float32(2.0 * math.pi) - np.float32(math.pi)).astype(np.float32)


def _identity_key(identity: JetIdentity) -> tuple[str, int, int]:
    return str(identity.file), int(identity.entry), int(identity.label)


def _as_optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class TargetDenoisingDatasetConfig:
    """Filesystem and split contract for Step 1 target-denoising datasets."""

    manifest_path: str
    hlt_cache_dir: str
    split: str = "model_train"
    data_dir: str | None = None
    max_jets: int | None = None
    alignment_mode: str = ALIGNMENT_MODE_ALIGNED_DIRECT
    expected_hlt_profile: str | None = None
    expected_hlt_degradation_strength: float | None = None
    expected_hlt_profile_version: str | None = None
    require_hlt_contract: bool = True
    require_same_manifest_hash: bool = True
    require_same_jet_identity: bool = True
    require_same_labels: bool = True
    require_same_mask_for_rank_alignment: bool = False
    require_identical_masks_for_aligned_direct_without_provenance: bool = True
    require_order_preserving_for_aligned_direct_without_provenance: bool = True
    allow_final_test_targets: bool = False
    verify_hlt_hash: bool = True
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000

    def __post_init__(self) -> None:
        mode = str(self.alignment_mode)
        if mode not in SUPPORTED_ALIGNMENT_MODES:
            raise ValueError(f"alignment_mode must be one of {SUPPORTED_ALIGNMENT_MODES}, got {mode!r}")
        max_jets = self.max_jets
        if max_jets is not None and int(max_jets) <= 0:
            raise ValueError("max_jets must be positive when provided")
        if self.expected_hlt_profile is not None:
            object.__setattr__(self, "expected_hlt_profile", normalize_hlt_profile(self.expected_hlt_profile))
        object.__setattr__(
            self,
            "expected_hlt_degradation_strength",
            _as_optional_float(self.expected_hlt_degradation_strength),
        )

    def expected_hlt_params(self) -> Mapping[str, Any] | None:
        if self.expected_hlt_profile is None or self.expected_hlt_degradation_strength is None:
            return None
        params = fixed_hlt_params_from_profile(
            self.expected_hlt_profile,
            float(self.expected_hlt_degradation_strength),
        )
        from jetclass_fresh.hlt_cache import fixed_hlt_params_dict

        return fixed_hlt_params_dict(params)


@dataclass(frozen=True)
class TargetResidualArrays:
    """Batch or split arrays for per-particle HLT-to-offline residual targets."""

    residuals: np.ndarray
    target_mask: np.ndarray
    target_weights: np.ndarray
    target_status: np.ndarray
    target_names: tuple[str, ...] = DENOISING_TARGET_NAMES
    alignment_mode: str = ALIGNMENT_MODE_ALIGNED_DIRECT
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.residuals.ndim != 3 or int(self.residuals.shape[-1]) != len(DENOISING_TARGET_NAMES):
            raise ValueError(
                f"residuals must have shape [N, P, {len(DENOISING_TARGET_NAMES)}], got {self.residuals.shape}"
            )
        expected_shape = self.residuals.shape[:2]
        for name, value in (
            ("target_mask", self.target_mask),
            ("target_weights", self.target_weights),
            ("target_status", self.target_status),
        ):
            if tuple(value.shape) != tuple(expected_shape):
                raise ValueError(f"{name} shape {value.shape} does not match residual shape {expected_shape}")

    def summary(self) -> dict[str, Any]:
        valid = np.asarray(self.target_mask, dtype=bool)
        residuals = np.asarray(self.residuals, dtype=np.float32)
        if np.any(valid):
            valid_residuals = residuals[valid]
            abs_mean = np.mean(np.abs(valid_residuals), axis=0)
            rms = np.sqrt(np.mean(valid_residuals * valid_residuals, axis=0))
        else:
            abs_mean = np.zeros((len(DENOISING_TARGET_NAMES),), dtype=np.float32)
            rms = np.zeros((len(DENOISING_TARGET_NAMES),), dtype=np.float32)
        return {
            "target_names": list(self.target_names),
            "alignment_mode": self.alignment_mode,
            "n_jets": int(residuals.shape[0]),
            "n_particles": int(residuals.shape[1]),
            "target_count": int(np.count_nonzero(valid)),
            "target_fraction": float(np.mean(valid)) if valid.size else 0.0,
            "abs_mean_by_target": {
                name: float(abs_mean[index]) for index, name in enumerate(DENOISING_TARGET_NAMES)
            },
            "rms_by_target": {name: float(rms[index]) for index, name in enumerate(DENOISING_TARGET_NAMES)},
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TargetDenoisingDatasetMetadata:
    """Serializable audit block for one paired denoising dataset."""

    contract: str = TARGET_DENOISING_DATASET_CONTRACT
    experiment_step: str = TARGET_DENOISING_STEP1
    split: str = ""
    n_jets: int = 0
    alignment_mode: str = ALIGNMENT_MODE_ALIGNED_DIRECT
    target_names: tuple[str, ...] = DENOISING_TARGET_NAMES
    hlt_profile: str | None = None
    hlt_profile_version: str | None = None
    hlt_degradation_strength: float | None = None
    hlt_content_hash: str | None = None
    hlt_jet_identity_hash: str | None = None
    offline_jet_identity_hash: str | None = None
    source_manifest_hash: str | None = None
    expected_manifest_hash: str | None = None
    returns_offline_particles: bool = True
    offline_targets_train_time_only: bool = True
    student_allowed_inputs: str = TARGET_DENOISING_ALLOWED_INPUTS
    target_summary: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "experiment_step": self.experiment_step,
            "split": self.split,
            "n_jets": int(self.n_jets),
            "alignment_mode": self.alignment_mode,
            "target_names": list(self.target_names),
            "hlt_profile": self.hlt_profile,
            "hlt_profile_version": self.hlt_profile_version,
            "hlt_degradation_strength": self.hlt_degradation_strength,
            "hlt_content_hash": self.hlt_content_hash,
            "hlt_jet_identity_hash": self.hlt_jet_identity_hash,
            "offline_jet_identity_hash": self.offline_jet_identity_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "expected_manifest_hash": self.expected_manifest_hash,
            "returns_offline_particles": bool(self.returns_offline_particles),
            "offline_targets_train_time_only": bool(self.offline_targets_train_time_only),
            "student_allowed_inputs": self.student_allowed_inputs,
            "target_summary": dict(self.target_summary),
            "notes": list(self.notes),
        }


def _validate_fixed_hlt_contract(
    hlt_view: JetView,
    *,
    expected_profile: str | None,
    expected_profile_version: str | None,
    expected_strength: float | None,
    expected_params: Mapping[str, Any] | None,
    expected_manifest_hash: str | None,
    require_same_manifest_hash: bool,
) -> None:
    metadata = dict(hlt_view.metadata)
    if metadata.get("view") not in (None, "fixed_hlt"):
        raise ValueError(f"Target denoising requires a fixed_hlt view, got {metadata.get('view')!r}")
    if expected_profile is not None:
        actual_profile = normalize_hlt_profile(metadata.get("hlt_profile"))
        if actual_profile != normalize_hlt_profile(expected_profile):
            raise ValueError(f"HLT profile mismatch: {actual_profile} != {expected_profile}")
    if expected_profile_version is not None:
        actual_version = str(metadata.get("hlt_profile_version") or "")
        if actual_version != str(expected_profile_version):
            raise ValueError(f"HLT profile version mismatch: {actual_version} != {expected_profile_version}")
    if expected_strength is not None:
        actual_strength = metadata.get("hlt_degradation_strength")
        if actual_strength is None or abs(float(actual_strength) - float(expected_strength)) > 1.0e-12:
            raise ValueError(f"HLT strength mismatch: {actual_strength} != {expected_strength}")
    if expected_params is not None and metadata.get("hlt_params") != dict(expected_params):
        raise ValueError("HLT params do not match expected profile/strength")
    if require_same_manifest_hash and expected_manifest_hash is not None:
        actual_manifest_hash = metadata.get("source_manifest_hash")
        if actual_manifest_hash not in (None, expected_manifest_hash):
            raise ValueError(f"HLT source manifest hash mismatch: {actual_manifest_hash} != {expected_manifest_hash}")


def _validate_paired_views(
    hlt_view: JetView,
    offline_view: JetView,
    *,
    config: TargetDenoisingDatasetConfig,
    expected_manifest_hash: str | None,
) -> None:
    if hlt_view.split != offline_view.split:
        raise ValueError(f"split mismatch: {hlt_view.split} != {offline_view.split}")
    if bool(config.require_same_jet_identity) and [_identity_key(x) for x in hlt_view.jet_ids] != [
        _identity_key(x) for x in offline_view.jet_ids
    ]:
        raise ValueError("HLT and offline views are not aligned by jet identity")
    if bool(config.require_same_labels) and not np.array_equal(hlt_view.labels, offline_view.labels):
        raise ValueError("HLT and offline labels differ")
    if bool(config.require_same_manifest_hash) and expected_manifest_hash is not None:
        offline_hash = offline_view.metadata.get("source_manifest_hash")
        if offline_hash != expected_manifest_hash:
            raise ValueError(f"offline source manifest hash mismatch: {offline_hash} != {expected_manifest_hash}")
    if bool(config.require_hlt_contract):
        _validate_fixed_hlt_contract(
            hlt_view,
            expected_profile=config.expected_hlt_profile,
            expected_profile_version=config.expected_hlt_profile_version,
            expected_strength=config.expected_hlt_degradation_strength,
            expected_params=config.expected_hlt_params(),
            expected_manifest_hash=expected_manifest_hash,
            require_same_manifest_hash=bool(config.require_same_manifest_hash),
        )
    if hlt_view.split == "final_test" and not bool(config.allow_final_test_targets):
        raise ValueError(
            "final_test offline denoising targets are guarded. Set allow_final_test_targets=True only for "
            "one-shot final reporting or explicitly marked diagnostics."
        )
    if bool(config.require_same_mask_for_rank_alignment) and not np.array_equal(hlt_view.mask, offline_view.mask):
        raise ValueError("rank-aligned denoising targets require identical HLT/offline masks")
    if (
        config.alignment_mode == ALIGNMENT_MODE_ALIGNED_DIRECT
        and bool(config.require_identical_masks_for_aligned_direct_without_provenance)
        and not np.array_equal(hlt_view.mask, offline_view.mask)
    ):
        raise ValueError(
            "aligned_direct target denoising needs per-particle HLT provenance when HLT/offline masks differ. "
            "Current fixed-HLT caches do not expose provenance, so realistic drop/merge caches fail closed. "
            "Use rank_direct only as an explicit debug control."
        )
    if (
        config.alignment_mode == ALIGNMENT_MODE_ALIGNED_DIRECT
        and bool(config.require_order_preserving_for_aligned_direct_without_provenance)
        and not _has_aligned_direct_order_guarantee(hlt_view.metadata)
    ):
        raise ValueError(
            "aligned_direct target denoising requires a cache-level guarantee that HLT particle order is inherited "
            "from the offline particle order. Current fixed-HLT caches sort particles after degradation and do not "
            "store per-particle provenance, so same-mask identity swaps can silently corrupt residual targets. "
            "Use rank_direct only as an explicit debug control, or build an identity/order-preserving target cache."
        )


def _truthy_metadata_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _has_aligned_direct_order_guarantee(metadata: Mapping[str, Any]) -> bool:
    """Return whether the cache explicitly supports same-rank residual targets."""

    metadata = dict(metadata or {})
    strength = metadata.get("hlt_degradation_strength")
    try:
        if strength is not None and abs(float(strength)) <= 1.0e-12:
            return True
    except (TypeError, ValueError):
        pass
    for key in (
        "target_denoising_order_preserving",
        "hlt_preserves_offline_particle_order",
        "particle_order_preserving",
        "identity_order_preserving",
        "same_rank_targets_safe",
    ):
        if _truthy_metadata_flag(metadata.get(key)):
            return True
    order_value = str(
        metadata.get("hlt_particle_order")
        or metadata.get("particle_order")
        or metadata.get("hlt_output_order")
        or ""
    ).strip().lower()
    return order_value in {
        "offline",
        "offline_order",
        "input",
        "input_order",
        "identity",
        "identity_preserving",
        "order_preserving",
    }


def _limit_view(view: JetView, max_jets: int | None) -> JetView:
    if max_jets is None or int(max_jets) >= int(len(view.labels)):
        return view
    limit = int(max_jets)
    return JetView(
        tokens=np.asarray(view.tokens[:limit], dtype=np.float32),
        mask=np.asarray(view.mask[:limit], dtype=bool),
        labels=np.asarray(view.labels[:limit], dtype=np.int64),
        jet_ids=list(view.jet_ids[:limit]),
        split=view.split,
        metadata={**dict(view.metadata), "target_denoising_max_jets": limit},
    )


def build_rank_aligned_residual_targets(
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
    offline_tokens: np.ndarray,
    offline_mask: np.ndarray,
    *,
    alignment_mode: str = ALIGNMENT_MODE_ALIGNED_DIRECT,
    eps: float = EPS,
) -> TargetResidualArrays:
    """Build direct same-rank residuals from HLT and offline token arrays.

    ``aligned_direct`` is only safe when upstream validation has proven that
    HLT/offline masks remain identical, or when a future cache provides true
    per-particle provenance. For degraded caches with drops/merges, callers
    should fail closed before reaching this helper.
    """

    hlt_tokens = np.asarray(hlt_tokens, dtype=np.float32)
    offline_tokens = np.asarray(offline_tokens, dtype=np.float32)
    hlt_mask = np.asarray(hlt_mask, dtype=bool)
    offline_mask = np.asarray(offline_mask, dtype=bool)
    if hlt_tokens.shape != offline_tokens.shape:
        raise ValueError(f"HLT/offline token shape mismatch: {hlt_tokens.shape} != {offline_tokens.shape}")
    if hlt_mask.shape != offline_mask.shape or hlt_mask.shape != hlt_tokens.shape[:2]:
        raise ValueError("HLT/offline mask shapes must match token row shapes")
    if hlt_tokens.ndim != 3 or hlt_tokens.shape[-1] < 4:
        raise ValueError("tokens must have shape [N, P, raw_dim>=4]")
    if alignment_mode not in SUPPORTED_ALIGNMENT_MODES:
        raise ValueError(f"Unsupported alignment_mode {alignment_mode!r}")

    valid = hlt_mask & offline_mask
    finite = np.isfinite(hlt_tokens[:, :, :4]).all(axis=-1) & np.isfinite(offline_tokens[:, :, :4]).all(axis=-1)
    positive = (hlt_tokens[:, :, 0] > 0.0) & (offline_tokens[:, :, 0] > 0.0)
    positive &= (hlt_tokens[:, :, 3] > 0.0) & (offline_tokens[:, :, 3] > 0.0)
    target_mask = valid & finite & positive

    residuals = np.zeros((*hlt_tokens.shape[:2], len(DENOISING_TARGET_NAMES)), dtype=np.float32)
    residuals[:, :, 0] = np.log(np.maximum(offline_tokens[:, :, 0], eps)) - np.log(np.maximum(hlt_tokens[:, :, 0], eps))
    residuals[:, :, 1] = offline_tokens[:, :, 1] - hlt_tokens[:, :, 1]
    residuals[:, :, 2] = wrap_delta_phi_np(offline_tokens[:, :, 2] - hlt_tokens[:, :, 2])
    residuals[:, :, 3] = np.log(np.maximum(offline_tokens[:, :, 3], eps)) - np.log(np.maximum(hlt_tokens[:, :, 3], eps))
    residuals = np.where(target_mask[:, :, None], residuals, 0.0).astype(np.float32)
    residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    target_weights = target_mask.astype(np.float32)
    target_status = np.where(target_mask, TARGET_STATUS_DIRECT, TARGET_STATUS_NO_TARGET).astype(np.int8)
    notes = (
        f"{alignment_mode} residuals use same-rank particles only after the dataset-level safety checks pass.",
        "Targets are train-time privileged supervision and must not be required for HLT-only inference.",
    )
    return TargetResidualArrays(
        residuals=residuals,
        target_mask=target_mask.astype(bool),
        target_weights=target_weights,
        target_status=target_status,
        alignment_mode=alignment_mode,
        notes=notes,
    )


class TargetDenoisingPairedDataset:
    """Paired cached-HLT/offline dataset for Step 1 target denoising."""

    def __init__(
        self,
        hlt_view: JetView,
        offline_view: JetView,
        *,
        config: TargetDenoisingDatasetConfig,
        expected_manifest_hash: str | None = None,
    ) -> None:
        _validate_paired_views(
            hlt_view,
            offline_view,
            config=config,
            expected_manifest_hash=expected_manifest_hash,
        )
        self.hlt_view = _limit_view(hlt_view, config.max_jets)
        self.offline_view = _limit_view(offline_view, config.max_jets)
        _validate_paired_views(self.hlt_view, self.offline_view, config=config, expected_manifest_hash=expected_manifest_hash)
        preview_limit = min(int(len(self.hlt_view.labels)), TARGET_SUMMARY_PREVIEW_JETS)
        preview_targets = build_rank_aligned_residual_targets(
            self.hlt_view.tokens[:preview_limit],
            self.hlt_view.mask[:preview_limit],
            self.offline_view.tokens[:preview_limit],
            self.offline_view.mask[:preview_limit],
            alignment_mode=config.alignment_mode,
        )
        self.config = config
        self.labels = np.asarray(self.hlt_view.labels, dtype=np.int64)
        self.jet_ids = list(self.hlt_view.jet_ids)
        self.split = self.hlt_view.split
        self.metadata = TargetDenoisingDatasetMetadata(
            split=self.split,
            n_jets=int(len(self.labels)),
            alignment_mode=config.alignment_mode,
            hlt_profile=self.hlt_view.metadata.get("hlt_profile"),
            hlt_profile_version=self.hlt_view.metadata.get("hlt_profile_version"),
            hlt_degradation_strength=self.hlt_view.metadata.get("hlt_degradation_strength"),
            hlt_content_hash=self.hlt_view.metadata.get("hlt_content_hash"),
            hlt_jet_identity_hash=jet_identity_hash(self.hlt_view.jet_ids),
            offline_jet_identity_hash=jet_identity_hash(self.offline_view.jet_ids),
            source_manifest_hash=self.hlt_view.metadata.get("source_manifest_hash")
            or self.offline_view.metadata.get("source_manifest_hash"),
            expected_manifest_hash=expected_manifest_hash,
            target_summary={
                **preview_targets.summary(),
                "summary_is_preview": True,
                "summary_preview_jets": int(preview_limit),
            },
            notes=(
                "Offline particles and residual targets are train-time denoising supervision only.",
                "The deployable tagger must consume HLT particles plus predicted denoiser outputs, not offline targets.",
            ),
        )

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        return {
            "hlt_tokens": np.asarray(self.hlt_view.tokens[index], dtype=np.float32),
            "hlt_mask": np.asarray(self.hlt_view.mask[index], dtype=bool),
            "offline_tokens": np.asarray(self.offline_view.tokens[index], dtype=np.float32),
            "offline_mask": np.asarray(self.offline_view.mask[index], dtype=bool),
            "label": np.int64(self.labels[index]),
            "jet_id": self.jet_ids[index],
            "alignment_mode": self.config.alignment_mode,
        }

    def to_metadata(self) -> dict[str, Any]:
        return self.metadata.to_dict()


def _stack_samples(samples: list[Mapping[str, Any]], key: str, dtype: Any) -> np.ndarray:
    return np.stack([np.asarray(sample[key], dtype=dtype) for sample in samples], axis=0)


def _batch_feature_rows(part_inputs: Any) -> np.ndarray:
    return np.transpose(part_inputs.pf_features, (0, 2, 1)).astype(np.float32, copy=False)


def collate_target_denoising_batch(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate a batch with canonical HLT/offline ParT inputs and residual targets."""

    if not samples:
        raise ValueError("cannot collate an empty target-denoising batch")
    torch = require_torch()
    hlt_tokens = _stack_samples(samples, "hlt_tokens", np.float32)
    hlt_mask = _stack_samples(samples, "hlt_mask", bool)
    offline_tokens = _stack_samples(samples, "offline_tokens", np.float32)
    offline_mask = _stack_samples(samples, "offline_mask", bool)
    labels = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    alignment_mode = str(samples[0].get("alignment_mode", ALIGNMENT_MODE_ALIGNED_DIRECT))
    targets = build_rank_aligned_residual_targets(
        hlt_tokens,
        hlt_mask,
        offline_tokens,
        offline_mask,
        alignment_mode=alignment_mode,
    )

    hlt_inputs = build_particle_transformer_inputs_from_tokens(
        hlt_tokens,
        hlt_mask,
        labels=labels,
        source_view="fixed_hlt",
    )
    offline_inputs = build_particle_transformer_inputs_from_tokens(
        offline_tokens,
        offline_mask,
        labels=labels,
        source_view="offline",
    )
    jet_ids = [sample["jet_id"] for sample in samples]
    return {
        "hlt_tokens": torch.from_numpy(hlt_tokens).float(),
        "hlt_constituent_mask": torch.from_numpy(hlt_mask).bool(),
        "offline_tokens": torch.from_numpy(offline_tokens).float(),
        "offline_constituent_mask": torch.from_numpy(offline_mask).bool(),
        "points": torch.from_numpy(hlt_inputs.pf_points).float(),
        "features": torch.from_numpy(hlt_inputs.pf_features).float(),
        "lorentz_vectors": torch.from_numpy(hlt_inputs.pf_vectors).float(),
        "mask": torch.from_numpy(hlt_inputs.pf_mask).bool(),
        "hlt_feature_rows": torch.from_numpy(_batch_feature_rows(hlt_inputs)).float(),
        "offline_points": torch.from_numpy(offline_inputs.pf_points).float(),
        "offline_features": torch.from_numpy(offline_inputs.pf_features).float(),
        "offline_lorentz_vectors": torch.from_numpy(offline_inputs.pf_vectors).float(),
        "offline_mask": torch.from_numpy(offline_inputs.pf_mask).bool(),
        "offline_feature_rows": torch.from_numpy(_batch_feature_rows(offline_inputs)).float(),
        "target_residuals": torch.from_numpy(targets.residuals).float(),
        "target_mask": torch.from_numpy(targets.target_mask).bool(),
        "target_weights": torch.from_numpy(targets.target_weights).float(),
        "target_status": torch.from_numpy(targets.target_status.astype(np.int64)).long(),
        "target_names": list(DENOISING_TARGET_NAMES),
        "labels": torch.from_numpy(labels).long(),
        "jet_ids": jet_ids,
        "jet_files": [identity.file for identity in jet_ids],
        "jet_entries": torch.tensor([int(identity.entry) for identity in jet_ids], dtype=torch.long),
        "jet_identity_labels": torch.tensor([int(identity.label) for identity in jet_ids], dtype=torch.long),
        "jet_keys": [identity.key() for identity in jet_ids],
        "student_allowed_inputs": TARGET_DENOISING_ALLOWED_INPUTS,
        "returns_offline_particles": True,
        "offline_targets_train_time_only": True,
        "pf_feature_names": list(PF_FEATURE_NAMES),
    }


def make_target_denoising_loader(
    dataset: TargetDenoisingPairedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 12345,
):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_target_denoising_batch,
        generator=generator,
    )


def target_denoising_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    """Move tensor values to a device while preserving jet-id lists/metadata."""

    torch = require_torch()
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_target_denoising_views(config: TargetDenoisingDatasetConfig) -> tuple[JetView, JetView, str]:
    """Load aligned cached-HLT and offline views for a denoising split."""

    if config.split == "final_test" and not bool(config.allow_final_test_targets):
        raise ValueError(
            "final_test offline denoising targets are guarded. Set allow_final_test_targets=True only for "
            "one-shot final reporting or explicitly marked diagnostics."
        )
    manifest = load_split_manifest(config.manifest_path)
    manifest_sha = manifest_hash(manifest)
    hlt_view = load_cached_hlt_view(config.hlt_cache_dir, config.split, verify_hash=bool(config.verify_hlt_hash))
    offline_view = load_offline_view(
        manifest,
        config.split,
        data_dir=config.data_dir,
        verify_label_branches=bool(config.verify_label_branches),
        read_chunk_size=int(config.read_chunk_size),
    )
    _validate_paired_views(hlt_view, offline_view, config=config, expected_manifest_hash=manifest_sha)
    return hlt_view, offline_view, manifest_sha


def load_target_denoising_dataset(config: TargetDenoisingDatasetConfig) -> TargetDenoisingPairedDataset:
    """Load one strict Step 1 target-denoising dataset from manifest/cache paths."""

    hlt_view, offline_view, manifest_sha = load_target_denoising_views(config)
    return TargetDenoisingPairedDataset(
        hlt_view,
        offline_view,
        config=config,
        expected_manifest_hash=manifest_sha,
    )

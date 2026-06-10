"""Shared view interfaces for teacher-logit reconstruction.

This module deliberately contains no reconstructor model, no teacher model, and
no training loop.  It defines the data boundary that later teacher-logit
reconstructors will use:

``fixed HLT JetView + offline JetView -> SoftReconstructedView -> tagger inputs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import (
    JetIdentity,
    JetView,
    RAW_TOKEN_DIM,
    SPLIT_ORDER,
    SplitManifest,
    load_offline_view,
    load_split_manifest,
    manifest_hash,
)
from jetclass_fresh.part_inputs import (
    ParticleTransformerInputs,
    build_particle_transformer_inputs_from_tokens,
    compute_view_jet_features,
)


def _as_float32(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return arr


def _as_bool(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def _as_int64(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _maybe_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch


def _is_torch_tensor(value: Any) -> bool:
    torch = _maybe_torch()
    return torch is not None and isinstance(value, torch.Tensor)


def _to_numpy_debug(value: Any) -> np.ndarray:
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_view_alignment(left: JetView, right: JetView, *, left_name: str = "left", right_name: str = "right") -> None:
    """Require two views to describe the exact same ordered jets."""

    if left.split != right.split:
        raise ValueError(f"Split mismatch between {left_name} and {right_name}: {left.split!r} != {right.split!r}")
    if left.tokens.shape[0] != right.tokens.shape[0]:
        raise ValueError(
            f"Jet count mismatch between {left_name} and {right_name}: "
            f"{left.tokens.shape[0]} != {right.tokens.shape[0]}"
        )
    if not np.array_equal(left.labels, right.labels):
        raise ValueError(f"Label mismatch between {left_name} and {right_name}")
    if list(left.jet_ids) != list(right.jet_ids):
        raise ValueError(f"Jet identity mismatch between {left_name} and {right_name}")


def slice_jet_view(view: JetView, max_jets: int | None) -> JetView:
    """Return the first ``max_jets`` rows of a ``JetView`` without mutating it."""

    if max_jets is None:
        return view
    limit = int(max_jets)
    if limit < 0:
        raise ValueError("max_jets must be non-negative")
    limit = min(limit, int(view.tokens.shape[0]))
    metadata = dict(view.metadata)
    metadata.update(
        {
            "row_limit_applied": True,
            "row_limit": int(limit),
            "source_n_jets_before_limit": int(view.tokens.shape[0]),
        }
    )
    return JetView(
        tokens=view.tokens[:limit].copy(),
        mask=view.mask[:limit].copy(),
        labels=view.labels[:limit].copy(),
        jet_ids=list(view.jet_ids[:limit]),
        split=view.split,
        metadata=metadata,
    )


def limit_split_manifest(manifest: SplitManifest, split: str, max_jets: int | None) -> SplitManifest:
    """Return a manifest copy with one split truncated to ``max_jets`` rows.

    This is useful for smoke/debug loaders because ``load_offline_view`` loads
    the identities listed in the manifest.  The returned manifest records the
    source manifest hash in metadata so the truncation is auditable.
    """

    if split not in SPLIT_ORDER:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_ORDER}")
    if max_jets is None:
        return manifest
    limit = int(max_jets)
    if limit < 0:
        raise ValueError("max_jets must be non-negative")
    splits = {name: list(rows) for name, rows in manifest.splits.items()}
    original_count = len(splits[split])
    splits[split] = splits[split][: min(limit, original_count)]
    split_sizes = dict(manifest.split_sizes)
    split_sizes[split] = len(splits[split])
    metadata = dict(manifest.metadata)
    metadata.update(
        {
            "limited_for_teacher_logit_reco_smoke": True,
            "limited_split": split,
            "limited_split_max_jets": int(limit),
            "limited_split_original_count": int(original_count),
            "source_manifest_hash_before_limit": manifest_hash(manifest),
        }
    )
    return SplitManifest(
        data_dir=manifest.data_dir,
        max_constits=manifest.max_constits,
        class_names=list(manifest.class_names),
        file_prefix_to_label=dict(manifest.file_prefix_to_label),
        split_sizes=split_sizes,
        split_seeds=dict(manifest.split_seeds),
        file_records=list(manifest.file_records),
        splits=splits,
        metadata=metadata,
    )


@dataclass
class PairedJetViews:
    """Parent container for aligned fixed-HLT and offline views."""

    hlt: JetView
    offline: JetView
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_view_alignment(self.hlt, self.offline, left_name="hlt", right_name="offline")
        self.metadata = {
            "split": self.hlt.split,
            "n_jets": int(self.hlt.tokens.shape[0]),
            "hlt_view": self.hlt.metadata.get("view"),
            "offline_view": self.offline.metadata.get("view"),
            "jet_identity_hash": jet_identity_hash(self.hlt.jet_ids),
            **dict(self.metadata),
        }

    @property
    def split(self) -> str:
        return self.hlt.split

    @property
    def labels(self) -> np.ndarray:
        return self.hlt.labels

    @property
    def jet_ids(self) -> list[JetIdentity]:
        return list(self.hlt.jet_ids)

    def slice(self, max_jets: int | None) -> "PairedJetViews":
        return PairedJetViews(
            hlt=slice_jet_view(self.hlt, max_jets),
            offline=slice_jet_view(self.offline, max_jets),
            metadata=dict(self.metadata),
        )


@dataclass
class SoftReconstructedView:
    """Differentiable reconstructed particle view boundary.

    ``weights`` are not folded into the stored tokens.  They are folded into
    `pt` and `energy` only when converting to tagger inputs.
    """

    tokens: np.ndarray
    mask: np.ndarray
    weights: np.ndarray
    labels: np.ndarray
    jet_ids: list[JetIdentity]
    split: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if _is_torch_tensor(self.tokens):
            self._post_init_torch()
            return
        self.tokens = _as_float32("tokens", self.tokens)
        self.mask = _as_bool("mask", self.mask)
        self.weights = _as_float32("weights", self.weights)
        self.labels = _as_int64("labels", self.labels)
        if self.tokens.ndim != 3 or self.tokens.shape[-1] != RAW_TOKEN_DIM:
            raise ValueError(f"tokens must have shape [N, P, {RAW_TOKEN_DIM}], got {self.tokens.shape}")
        if self.mask.shape != self.tokens.shape[:2]:
            raise ValueError(f"mask shape {self.mask.shape} does not match tokens {self.tokens.shape[:2]}")
        if self.weights.shape != self.tokens.shape[:2]:
            raise ValueError(f"weights shape {self.weights.shape} does not match tokens {self.tokens.shape[:2]}")
        if self.labels.shape[0] != self.tokens.shape[0]:
            raise ValueError("labels length does not match number of jets")
        if len(self.jet_ids) != self.tokens.shape[0]:
            raise ValueError("jet_ids length does not match number of jets")
        if np.any(self.weights < 0.0):
            raise ValueError("weights must be non-negative")
        self.weights = np.where(self.mask, self.weights, 0.0).astype(np.float32)
        self.tokens = np.where(self.mask[:, :, None], self.tokens, 0.0).astype(np.float32)
        self.metadata = {
            "view": "teacher_logit_soft_reco",
            "n_jets": int(self.tokens.shape[0]),
            "n_candidates": int(self.tokens.shape[1]),
            "raw_token_dim": int(self.tokens.shape[2]),
            "jet_identity_hash": jet_identity_hash(self.jet_ids),
            **dict(self.metadata),
        }

    def _post_init_torch(self) -> None:
        torch = _maybe_torch()
        if torch is None:  # pragma: no cover - guarded by _is_torch_tensor
            raise ImportError("PyTorch is required for tensor SoftReconstructedView inputs")
        if not _is_torch_tensor(self.mask) or not _is_torch_tensor(self.weights):
            raise TypeError("tokens, mask, and weights must all be torch tensors for torch soft views")
        self.tokens = self.tokens.float()
        self.mask = self.mask.bool()
        self.weights = self.weights.float()
        if _is_torch_tensor(self.labels):
            self.labels = self.labels.to(device=self.tokens.device, dtype=torch.long)
        else:
            self.labels = torch.as_tensor(self.labels, dtype=torch.long, device=self.tokens.device)
        if not bool(torch.isfinite(self.tokens).all()):
            raise FloatingPointError("tokens contains non-finite values")
        if not bool(torch.isfinite(self.weights).all()):
            raise FloatingPointError("weights contains non-finite values")
        if self.tokens.ndim != 3 or int(self.tokens.shape[-1]) != RAW_TOKEN_DIM:
            raise ValueError(f"tokens must have shape [N, P, {RAW_TOKEN_DIM}], got {tuple(self.tokens.shape)}")
        if tuple(self.mask.shape) != tuple(self.tokens.shape[:2]):
            raise ValueError(f"mask shape {tuple(self.mask.shape)} does not match tokens {tuple(self.tokens.shape[:2])}")
        if tuple(self.weights.shape) != tuple(self.tokens.shape[:2]):
            raise ValueError(
                f"weights shape {tuple(self.weights.shape)} does not match tokens {tuple(self.tokens.shape[:2])}"
            )
        if int(self.labels.shape[0]) != int(self.tokens.shape[0]):
            raise ValueError("labels length does not match number of jets")
        if len(self.jet_ids) != int(self.tokens.shape[0]):
            raise ValueError("jet_ids length does not match number of jets")
        if bool((self.weights < 0.0).any()):
            raise ValueError("weights must be non-negative")
        self.weights = torch.where(self.mask, self.weights, torch.zeros_like(self.weights))
        self.tokens = torch.where(self.mask[:, :, None], self.tokens, torch.zeros_like(self.tokens))
        self.metadata = {
            "view": "teacher_logit_soft_reco",
            "n_jets": int(self.tokens.shape[0]),
            "n_candidates": int(self.tokens.shape[1]),
            "raw_token_dim": int(self.tokens.shape[2]),
            "jet_identity_hash": jet_identity_hash(self.jet_ids),
            **dict(self.metadata),
        }

    @property
    def effective_mask(self):
        return self.mask & (self.weights > 0.0)

    def as_jet_view(self) -> JetView:
        """Return an unweighted `JetView` wrapper for metadata/debugging."""

        return JetView(
            tokens=_to_numpy_debug(self.tokens).copy(),
            mask=_to_numpy_debug(self.mask).astype(bool, copy=True),
            labels=_to_numpy_debug(self.labels).astype(np.int64, copy=True),
            jet_ids=list(self.jet_ids),
            split=self.split,
            metadata=dict(self.metadata),
        )

    def to_particle_transformer_inputs(self, *, weight_threshold: float = 0.0) -> ParticleTransformerInputs:
        return soft_view_to_particle_transformer_inputs(self, weight_threshold=weight_threshold)


def soft_view_to_particle_transformer_inputs(
    view: SoftReconstructedView,
    *,
    weight_threshold: float = 0.0,
) -> ParticleTransformerInputs:
    """Convert a soft reconstructed view into canonical tagger inputs."""

    return build_particle_transformer_inputs_from_tokens(
        _to_numpy_debug(view.tokens),
        _to_numpy_debug(view.mask).astype(bool),
        labels=_to_numpy_debug(view.labels).astype(np.int64),
        split=view.split,
        source_view=view.metadata.get("view", "teacher_logit_soft_reco"),
        candidate_weights=_to_numpy_debug(view.weights),
        fold_weights=True,
        weight_threshold=float(weight_threshold),
    )


def make_identity_soft_view(view: JetView, *, metadata: Mapping[str, Any] | None = None) -> SoftReconstructedView:
    """Wrap an existing view as a soft view with unit weights on valid tokens."""

    payload = {
        "construction": "identity",
        "source_view": view.metadata.get("view"),
        **dict(metadata or {}),
    }
    return SoftReconstructedView(
        tokens=view.tokens.copy(),
        mask=view.mask.copy(),
        weights=view.mask.astype(np.float32),
        labels=view.labels.copy(),
        jet_ids=list(view.jet_ids),
        split=view.split,
        metadata=payload,
    )


def make_soft_view_from_parents_and_extras(
    *,
    parent_tokens: np.ndarray,
    parent_mask: np.ndarray,
    parent_weights: np.ndarray,
    extra_tokens: np.ndarray | None,
    extra_weights: np.ndarray | None,
    labels: np.ndarray,
    jet_ids: Sequence[JetIdentity],
    split: str,
    extra_mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SoftReconstructedView:
    """Build a soft view from corrected parent tokens plus extra candidates."""

    parent_tokens = _as_float32("parent_tokens", parent_tokens)
    parent_mask = _as_bool("parent_mask", parent_mask)
    parent_weights = _as_float32("parent_weights", parent_weights)
    if parent_tokens.ndim != 3 or parent_tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"parent_tokens must have shape [N, P, {RAW_TOKEN_DIM}], got {parent_tokens.shape}")
    if parent_mask.shape != parent_tokens.shape[:2]:
        raise ValueError("parent_mask shape does not match parent_tokens")
    if parent_weights.shape != parent_tokens.shape[:2]:
        raise ValueError("parent_weights shape does not match parent_tokens")

    if extra_tokens is None:
        tokens = parent_tokens
        mask = parent_mask
        weights = parent_weights
        n_extra = 0
    else:
        extra_tokens = _as_float32("extra_tokens", extra_tokens)
        if extra_tokens.ndim != 3 or extra_tokens.shape[0] != parent_tokens.shape[0] or extra_tokens.shape[-1] != RAW_TOKEN_DIM:
            raise ValueError(
                "extra_tokens must have shape [N, K, RAW_TOKEN_DIM] with the same N as parent_tokens"
            )
        if extra_weights is None:
            raise ValueError("extra_weights must be provided when extra_tokens is provided")
        extra_weights = _as_float32("extra_weights", extra_weights)
        if extra_weights.shape != extra_tokens.shape[:2]:
            raise ValueError("extra_weights shape does not match extra_tokens")
        if extra_mask is None:
            extra_mask = np.ones(extra_tokens.shape[:2], dtype=bool)
        else:
            extra_mask = _as_bool("extra_mask", extra_mask)
            if extra_mask.shape != extra_tokens.shape[:2]:
                raise ValueError("extra_mask shape does not match extra_tokens")
        tokens = np.concatenate([parent_tokens, extra_tokens], axis=1)
        mask = np.concatenate([parent_mask, extra_mask], axis=1)
        weights = np.concatenate([parent_weights, extra_weights], axis=1)
        n_extra = int(extra_tokens.shape[1])

    payload = {
        "construction": "parents_plus_extras",
        "n_parent_candidates": int(parent_tokens.shape[1]),
        "n_extra_candidates": int(n_extra),
        **dict(metadata or {}),
    }
    return SoftReconstructedView(
        tokens=tokens,
        mask=mask,
        weights=weights,
        labels=labels,
        jet_ids=list(jet_ids),
        split=split,
        metadata=payload,
    )


def load_paired_jet_views(
    *,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    split: str,
    data_dir: str | Path | None = None,
    max_jets: int | None = None,
    verify_hlt_hash: bool = True,
    verify_label_branches: bool = False,
    read_chunk_size: int = 50_000,
) -> PairedJetViews:
    """Load aligned fixed-HLT and offline views for one split."""

    manifest = load_split_manifest(manifest_path)
    if split not in SPLIT_ORDER:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_ORDER}")
    hlt_view = slice_jet_view(
        load_cached_hlt_view(hlt_cache_dir, split, verify_hash=verify_hlt_hash),
        max_jets,
    )
    offline_manifest = limit_split_manifest(manifest, split, max_jets)
    offline_view = load_offline_view(
        offline_manifest,
        split,
        data_dir=data_dir,
        verify_label_branches=verify_label_branches,
        read_chunk_size=read_chunk_size,
    )
    return PairedJetViews(
        hlt=hlt_view,
        offline=offline_view,
        metadata={
            "manifest_path": str(manifest_path),
            "hlt_cache_dir": str(hlt_cache_dir),
            "data_dir": str(data_dir) if data_dir is not None else manifest.data_dir,
            "max_jets": None if max_jets is None else int(max_jets),
            "source_manifest_hash": manifest_hash(manifest),
        },
    )


def _count_stats(mask: np.ndarray, weights: np.ndarray | None = None) -> Dict[str, float]:
    if weights is None:
        counts = np.sum(mask, axis=1).astype(np.float64)
    else:
        counts = np.sum(np.where(mask, weights, 0.0), axis=1).astype(np.float64)
    if counts.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(np.min(counts)),
        "mean": float(np.mean(counts)),
        "max": float(np.max(counts)),
    }


def summarize_soft_view(view: SoftReconstructedView) -> Dict[str, Any]:
    tokens = _to_numpy_debug(view.tokens).astype(np.float32)
    mask = _to_numpy_debug(view.mask).astype(bool)
    weights = _to_numpy_debug(view.weights).astype(np.float32)
    particle, jet_features = compute_view_jet_features(tokens, mask)
    del particle
    return {
        "split": view.split,
        "n_jets": int(tokens.shape[0]),
        "n_candidates": int(tokens.shape[1]),
        "valid_candidate_count": _count_stats(mask),
        "weighted_candidate_count": _count_stats(mask, weights),
        "min_weight": float(np.min(weights)) if weights.size else 0.0,
        "max_weight": float(np.max(weights)) if weights.size else 0.0,
        "mean_weight_on_valid": float(np.mean(weights[mask])) if np.any(mask) else 0.0,
        "mean_jet_pt_unweighted": float(np.mean(jet_features[:, 0])) if len(jet_features) else 0.0,
        "mean_jet_mass_unweighted": float(np.mean(jet_features[:, 4])) if len(jet_features) else 0.0,
        "metadata": dict(view.metadata),
    }


def summarize_paired_jet_views(pair: PairedJetViews) -> Dict[str, Any]:
    return {
        "split": pair.split,
        "n_jets": int(pair.hlt.tokens.shape[0]),
        "jet_identity_hash": pair.metadata.get("jet_identity_hash"),
        "hlt_shape": list(pair.hlt.tokens.shape),
        "offline_shape": list(pair.offline.tokens.shape),
        "hlt_valid_count": _count_stats(pair.hlt.mask),
        "offline_valid_count": _count_stats(pair.offline.mask),
        "hlt_metadata_view": pair.hlt.metadata.get("view"),
        "offline_metadata_view": pair.offline.metadata.get("view"),
        "metadata": dict(pair.metadata),
    }

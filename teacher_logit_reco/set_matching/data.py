"""Paired HLT/offline set dataset for set-matching reconstruction.

This module is the data boundary for the set-matching branch.  It reuses the
existing fixed-HLT cache and offline JetClass split loader, but exposes a new
dataset/collate/statistics interface that is not coupled to teacher-logit
losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import jet_identity_hash
from jetclass_fresh.jetclass_data import (
    JetIdentity,
    JetView,
    RAW_TOKEN_DIM,
    SPLIT_ORDER,
    SplitManifest,
    load_split_manifest,
    manifest_hash,
)
from teacher_logit_reco.views import PairedJetViews, load_paired_jet_views, validate_view_alignment


FEATURE_STAT_EPS = 1.0e-6
DEFAULT_FEATURE_NAMES: tuple[str, ...] = tuple(f"raw_token_{index}" for index in range(RAW_TOKEN_DIM))


def _maybe_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch


def require_torch():
    torch = _maybe_torch()
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("Set-matching DataLoader utilities require PyTorch")
    return torch


def _as_float32_tokens(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D [jets, particles, features], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return arr


def _as_bool_mask(name: str, value: np.ndarray, token_shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(value, dtype=bool)
    if arr.shape != token_shape[:2]:
        raise ValueError(f"{name} shape {arr.shape} does not match tokens {token_shape[:2]}")
    return arr


def _count_summary(mask: np.ndarray) -> dict[str, float]:
    counts = np.sum(np.asarray(mask, dtype=bool), axis=1).astype(np.float64)
    if counts.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    return {
        "min": float(np.min(counts)),
        "mean": float(np.mean(counts)),
        "max": float(np.max(counts)),
        "std": float(np.std(counts)),
    }


def _identity_key(identity: JetIdentity) -> str:
    return identity.key()


@dataclass(frozen=True)
class SetMatchingJetSample:
    """One aligned fixed-HLT/offline jet pair."""

    hlt_tokens: np.ndarray
    hlt_mask: np.ndarray
    offline_tokens: np.ndarray
    offline_mask: np.ndarray
    label: int
    jet_id: JetIdentity
    index: int
    split: str

    def __post_init__(self) -> None:
        hlt_tokens = np.asarray(self.hlt_tokens, dtype=np.float32)
        offline_tokens = np.asarray(self.offline_tokens, dtype=np.float32)
        hlt_mask = np.asarray(self.hlt_mask, dtype=bool)
        offline_mask = np.asarray(self.offline_mask, dtype=bool)
        if hlt_tokens.ndim != 2 or offline_tokens.ndim != 2:
            raise ValueError("sample tokens must be 2D [particles, features]")
        if hlt_tokens.shape[-1] != offline_tokens.shape[-1]:
            raise ValueError("HLT and offline sample feature dimensions must match")
        if hlt_mask.shape != hlt_tokens.shape[:1]:
            raise ValueError("hlt_mask length does not match hlt_tokens")
        if offline_mask.shape != offline_tokens.shape[:1]:
            raise ValueError("offline_mask length does not match offline_tokens")
        if not np.isfinite(hlt_tokens).all() or not np.isfinite(offline_tokens).all():
            raise FloatingPointError("sample tokens contain non-finite values")
        object.__setattr__(self, "hlt_tokens", hlt_tokens)
        object.__setattr__(self, "offline_tokens", offline_tokens)
        object.__setattr__(self, "hlt_mask", hlt_mask)
        object.__setattr__(self, "offline_mask", offline_mask)
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "index", int(self.index))


@dataclass(frozen=True)
class FeatureNormalizationStats:
    """Masked per-feature normalization constants."""

    mean: np.ndarray
    std: np.ndarray
    count: int
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES
    source_view: str = "unknown"
    eps: float = FEATURE_STAT_EPS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError("mean and std must be 1D arrays")
        if mean.shape != std.shape:
            raise ValueError("mean and std shapes must match")
        if len(self.feature_names) != mean.shape[0]:
            raise ValueError("feature_names length must match mean/std")
        if int(self.count) < 0:
            raise ValueError("count cannot be negative")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise FloatingPointError("normalization stats contain non-finite values")
        std = np.where(std <= float(self.eps), 1.0, std).astype(np.float32)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "feature_names", tuple(str(name) for name in self.feature_names))
        object.__setattr__(self, "eps", float(self.eps))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    def transform(self, tokens: np.ndarray) -> np.ndarray:
        arr = np.asarray(tokens, dtype=np.float32)
        if arr.shape[-1] != self.feature_dim:
            raise ValueError(f"tokens feature dimension {arr.shape[-1]} != stats feature_dim {self.feature_dim}")
        return ((arr - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_view": self.source_view,
            "count": int(self.count),
            "feature_dim": int(self.feature_dim),
            "feature_names": list(self.feature_names),
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "eps": float(self.eps),
            "metadata": dict(self.metadata),
        }


class SetMatchingJetDataset:
    """Dataset over aligned fixed-HLT inputs and offline set targets."""

    def __init__(
        self,
        pair: PairedJetViews,
        *,
        max_jets: int | None = None,
        trim_to_valid: bool = False,
        copy_arrays: bool = False,
    ) -> None:
        pair = pair.slice(max_jets)
        validate_view_alignment(pair.hlt, pair.offline, left_name="hlt", right_name="offline")
        self.hlt_tokens = _as_float32_tokens("hlt_tokens", pair.hlt.tokens)
        self.hlt_mask = _as_bool_mask("hlt_mask", pair.hlt.mask, self.hlt_tokens.shape)
        self.offline_tokens = _as_float32_tokens("offline_tokens", pair.offline.tokens)
        self.offline_mask = _as_bool_mask("offline_mask", pair.offline.mask, self.offline_tokens.shape)
        if self.hlt_tokens.shape[0] != self.offline_tokens.shape[0]:
            raise ValueError("HLT and offline jet counts do not match")
        if self.hlt_tokens.shape[-1] != self.offline_tokens.shape[-1]:
            raise ValueError("HLT and offline feature dimensions do not match")
        self.labels = np.asarray(pair.labels, dtype=np.int64)
        self.jet_ids = list(pair.jet_ids)
        self.split = pair.split
        self.trim_to_valid = bool(trim_to_valid)
        metadata = dict(pair.metadata)
        metadata.update(
            {
                "dataset": "set_matching_paired_hlt_offline",
                "split": self.split,
                "n_jets": int(self.labels.shape[0]),
                "feature_dim": int(self.hlt_tokens.shape[-1]),
                "trim_to_valid": bool(trim_to_valid),
                "jet_identity_hash": jet_identity_hash(self.jet_ids),
            }
        )
        self.metadata = metadata
        if copy_arrays:
            self.hlt_tokens = self.hlt_tokens.copy()
            self.hlt_mask = self.hlt_mask.copy()
            self.offline_tokens = self.offline_tokens.copy()
            self.offline_mask = self.offline_mask.copy()
            self.labels = self.labels.copy()

    @classmethod
    def from_paths(
        cls,
        *,
        manifest_path: str | Path,
        hlt_cache_dir: str | Path,
        split: str,
        data_dir: str | Path | None = None,
        max_jets: int | None = None,
        trim_to_valid: bool = False,
        verify_hlt_hash: bool = True,
        verify_label_branches: bool = False,
        read_chunk_size: int = 50_000,
    ) -> "SetMatchingJetDataset":
        pair = load_paired_jet_views(
            manifest_path=manifest_path,
            hlt_cache_dir=hlt_cache_dir,
            split=split,
            data_dir=data_dir,
            max_jets=max_jets,
            verify_hlt_hash=verify_hlt_hash,
            verify_label_branches=verify_label_branches,
            read_chunk_size=read_chunk_size,
        )
        return cls(pair, trim_to_valid=trim_to_valid)

    @property
    def feature_dim(self) -> int:
        return int(self.hlt_tokens.shape[-1])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _row(self, tokens: np.ndarray, mask: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        row_tokens = tokens[index]
        row_mask = mask[index]
        if not self.trim_to_valid:
            return row_tokens, row_mask
        return row_tokens[row_mask], np.ones((int(np.sum(row_mask)),), dtype=bool)

    def __getitem__(self, index: int) -> SetMatchingJetSample:
        idx = int(index)
        hlt_tokens, hlt_mask = self._row(self.hlt_tokens, self.hlt_mask, idx)
        offline_tokens, offline_mask = self._row(self.offline_tokens, self.offline_mask, idx)
        return SetMatchingJetSample(
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            offline_tokens=offline_tokens,
            offline_mask=offline_mask,
            label=int(self.labels[idx]),
            jet_id=self.jet_ids[idx],
            index=idx,
            split=self.split,
        )


def _pad_token_rows(rows: Sequence[np.ndarray], masks: Sequence[np.ndarray], *, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("cannot collate an empty sample list")
    feature_dims = {np.asarray(row).shape[-1] for row in rows}
    if len(feature_dims) != 1:
        raise ValueError(f"{name} rows have inconsistent feature dimensions: {sorted(feature_dims)}")
    max_len = max(int(np.asarray(row).shape[0]) for row in rows)
    feature_dim = int(next(iter(feature_dims)))
    tokens = np.zeros((len(rows), max_len, feature_dim), dtype=np.float32)
    padded_mask = np.zeros((len(rows), max_len), dtype=bool)
    for index, (row, mask) in enumerate(zip(rows, masks)):
        row = np.asarray(row, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if row.ndim != 2:
            raise ValueError(f"{name} row {index} must be 2D, got {row.shape}")
        if mask.shape != row.shape[:1]:
            raise ValueError(f"{name} mask {index} shape {mask.shape} does not match row {row.shape[:1]}")
        length = int(row.shape[0])
        tokens[index, :length] = row
        padded_mask[index, :length] = mask
    return tokens, padded_mask


def _sample_attr(sample: SetMatchingJetSample | Mapping[str, Any], name: str) -> Any:
    if isinstance(sample, Mapping):
        return sample[name]
    return getattr(sample, name)


def collate_set_matching_samples(
    samples: Sequence[SetMatchingJetSample | Mapping[str, Any]],
    *,
    as_torch: bool = True,
) -> dict[str, Any]:
    """Pad and collate variable-length paired set samples."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    splits = {str(_sample_attr(sample, "split")) for sample in samples}
    if len(splits) != 1:
        raise ValueError(f"all samples in a batch must come from one split, got {sorted(splits)}")
    hlt_tokens, hlt_mask = _pad_token_rows(
        [_sample_attr(sample, "hlt_tokens") for sample in samples],
        [_sample_attr(sample, "hlt_mask") for sample in samples],
        name="hlt",
    )
    offline_tokens, offline_mask = _pad_token_rows(
        [_sample_attr(sample, "offline_tokens") for sample in samples],
        [_sample_attr(sample, "offline_mask") for sample in samples],
        name="offline",
    )
    labels = np.asarray([int(_sample_attr(sample, "label")) for sample in samples], dtype=np.int64)
    indices = np.asarray([int(_sample_attr(sample, "index")) for sample in samples], dtype=np.int64)
    jet_ids = [_sample_attr(sample, "jet_id") for sample in samples]
    batch: dict[str, Any] = {
        "hlt_tokens": hlt_tokens,
        "hlt_mask": hlt_mask,
        "offline_tokens": offline_tokens,
        "offline_mask": offline_mask,
        "labels": labels,
        "indices": indices,
        "jet_ids": jet_ids,
        "split": next(iter(splits)),
    }
    if not as_torch:
        return batch

    torch = require_torch()
    return {
        "hlt_tokens": torch.from_numpy(hlt_tokens).float(),
        "hlt_mask": torch.from_numpy(hlt_mask).bool(),
        "offline_tokens": torch.from_numpy(offline_tokens).float(),
        "offline_mask": torch.from_numpy(offline_mask).bool(),
        "labels": torch.from_numpy(labels).long(),
        "indices": torch.from_numpy(indices).long(),
        "jet_ids": jet_ids,
        "split": batch["split"],
    }


def make_set_matching_loader(
    dataset: SetMatchingJetDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 12345,
):
    """Build a PyTorch DataLoader for set-matching reconstruction training."""

    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_set_matching_samples,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def compute_feature_normalization_stats(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    source_view: str = "unknown",
    eps: float = FEATURE_STAT_EPS,
    metadata: Mapping[str, Any] | None = None,
) -> FeatureNormalizationStats:
    """Compute masked per-feature mean/std over valid particles."""

    tokens = _as_float32_tokens("tokens", tokens)
    mask = _as_bool_mask("mask", mask, tokens.shape)
    valid = tokens[mask]
    if valid.shape[0] == 0:
        raise ValueError("cannot compute normalization stats with zero valid particles")
    names = tuple(feature_names or tuple(f"raw_token_{index}" for index in range(tokens.shape[-1])))
    if len(names) != tokens.shape[-1]:
        raise ValueError("feature_names length must match token feature dimension")
    std = valid.std(axis=0).astype(np.float32)
    std = np.where(std <= float(eps), 1.0, std).astype(np.float32)
    return FeatureNormalizationStats(
        mean=valid.mean(axis=0).astype(np.float32),
        std=std,
        count=int(valid.shape[0]),
        feature_names=names,
        source_view=source_view,
        eps=float(eps),
        metadata=dict(metadata or {}),
    )


def build_set_matching_feature_normalization(
    pair_or_dataset: PairedJetViews | SetMatchingJetDataset,
    *,
    include_combined: bool = True,
    feature_names: Sequence[str] | None = None,
    eps: float = FEATURE_STAT_EPS,
) -> dict[str, FeatureNormalizationStats]:
    """Build HLT/offline/combined masked feature normalization stats."""

    if isinstance(pair_or_dataset, SetMatchingJetDataset):
        hlt_tokens = pair_or_dataset.hlt_tokens
        hlt_mask = pair_or_dataset.hlt_mask
        offline_tokens = pair_or_dataset.offline_tokens
        offline_mask = pair_or_dataset.offline_mask
        split = pair_or_dataset.split
        identity_hash = pair_or_dataset.metadata.get("jet_identity_hash")
    else:
        hlt_tokens = pair_or_dataset.hlt.tokens
        hlt_mask = pair_or_dataset.hlt.mask
        offline_tokens = pair_or_dataset.offline.tokens
        offline_mask = pair_or_dataset.offline.mask
        split = pair_or_dataset.split
        identity_hash = pair_or_dataset.metadata.get("jet_identity_hash")

    stats = {
        "hlt": compute_feature_normalization_stats(
            hlt_tokens,
            hlt_mask,
            feature_names=feature_names,
            source_view="fixed_hlt",
            eps=eps,
            metadata={"split": split, "jet_identity_hash": identity_hash},
        ),
        "offline": compute_feature_normalization_stats(
            offline_tokens,
            offline_mask,
            feature_names=feature_names,
            source_view="offline",
            eps=eps,
            metadata={"split": split, "jet_identity_hash": identity_hash},
        ),
    }
    if include_combined:
        combined_values = np.concatenate([hlt_tokens[hlt_mask], offline_tokens[offline_mask]], axis=0)
        combined_tokens = combined_values.reshape(1, combined_values.shape[0], combined_values.shape[1])
        combined_mask = np.ones((1, combined_values.shape[0]), dtype=bool)
        stats["combined"] = compute_feature_normalization_stats(
            combined_tokens,
            combined_mask,
            feature_names=feature_names,
            source_view="hlt_plus_offline",
            eps=eps,
            metadata={"split": split, "jet_identity_hash": identity_hash},
        )
    return stats


def audit_set_matching_pair(
    pair_or_dataset: PairedJetViews | SetMatchingJetDataset,
    *,
    manifest: SplitManifest | None = None,
    expected_split: str | None = None,
) -> dict[str, Any]:
    """Audit split alignment and identity hygiene for paired set data."""

    if isinstance(pair_or_dataset, SetMatchingJetDataset):
        split = pair_or_dataset.split
        hlt_tokens = pair_or_dataset.hlt_tokens
        hlt_mask = pair_or_dataset.hlt_mask
        offline_tokens = pair_or_dataset.offline_tokens
        offline_mask = pair_or_dataset.offline_mask
        labels = pair_or_dataset.labels
        jet_ids = pair_or_dataset.jet_ids
    else:
        validate_view_alignment(pair_or_dataset.hlt, pair_or_dataset.offline, left_name="hlt", right_name="offline")
        split = pair_or_dataset.split
        hlt_tokens = pair_or_dataset.hlt.tokens
        hlt_mask = pair_or_dataset.hlt.mask
        offline_tokens = pair_or_dataset.offline.tokens
        offline_mask = pair_or_dataset.offline.mask
        labels = pair_or_dataset.labels
        jet_ids = pair_or_dataset.jet_ids

    problems: list[str] = []
    if expected_split is not None and split != expected_split:
        problems.append(f"split {split!r} != expected_split {expected_split!r}")
    if split not in SPLIT_ORDER:
        problems.append(f"unknown split {split!r}")
    if hlt_tokens.shape[0] != offline_tokens.shape[0]:
        problems.append("hlt/offline jet count mismatch")
    if hlt_tokens.shape[-1] != offline_tokens.shape[-1]:
        problems.append("hlt/offline feature dimension mismatch")
    if len(labels) != len(jet_ids):
        problems.append("label count does not match jet identity count")

    label_mismatch_count = int(sum(int(label) != int(jet_id.label) for label, jet_id in zip(labels, jet_ids)))
    if label_mismatch_count:
        problems.append(f"{label_mismatch_count} labels do not match jet identity labels")

    identity_keys = [_identity_key(identity) for identity in jet_ids]
    duplicate_identity_count = int(len(identity_keys) - len(set(identity_keys)))
    if duplicate_identity_count:
        problems.append(f"{duplicate_identity_count} duplicate jet identities inside split")

    manifest_hash_value = None
    manifest_match = None
    if manifest is not None:
        manifest_hash_value = manifest_hash(manifest)
        expected = list(manifest.splits.get(split, []))
        manifest_match = list(jet_ids) == expected[: len(jet_ids)]
        if not manifest_match:
            problems.append("dataset jet identities do not match manifest split order/prefix")

    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "split": split,
        "n_jets": int(len(jet_ids)),
        "feature_dim": int(hlt_tokens.shape[-1]) if hlt_tokens.ndim == 3 else None,
        "jet_identity_hash": jet_identity_hash(jet_ids),
        "duplicate_identity_count": duplicate_identity_count,
        "label_mismatch_count": label_mismatch_count,
        "hlt_shape": list(hlt_tokens.shape),
        "offline_shape": list(offline_tokens.shape),
        "hlt_valid_count": _count_summary(hlt_mask),
        "offline_valid_count": _count_summary(offline_mask),
        "manifest_hash": manifest_hash_value,
        "matches_manifest_split_prefix": manifest_match,
    }


def load_set_matching_dataset(
    *,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    split: str,
    data_dir: str | Path | None = None,
    max_jets: int | None = None,
    trim_to_valid: bool = False,
    verify_hlt_hash: bool = True,
    verify_label_branches: bool = False,
    read_chunk_size: int = 50_000,
) -> SetMatchingJetDataset:
    """Load one leakage-clean paired split as a set-matching dataset."""

    return SetMatchingJetDataset.from_paths(
        manifest_path=manifest_path,
        hlt_cache_dir=hlt_cache_dir,
        split=split,
        data_dir=data_dir,
        max_jets=max_jets,
        trim_to_valid=trim_to_valid,
        verify_hlt_hash=verify_hlt_hash,
        verify_label_branches=verify_label_branches,
        read_chunk_size=read_chunk_size,
    )


def load_manifest_for_set_matching(path: str | Path) -> SplitManifest:
    """Small named wrapper so set-matching scripts do not import old runners."""

    return load_split_manifest(path)


__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "FEATURE_STAT_EPS",
    "FeatureNormalizationStats",
    "SetMatchingJetDataset",
    "SetMatchingJetSample",
    "audit_set_matching_pair",
    "build_set_matching_feature_normalization",
    "collate_set_matching_samples",
    "compute_feature_normalization_stats",
    "load_manifest_for_set_matching",
    "load_set_matching_dataset",
    "make_set_matching_loader",
    "require_torch",
]

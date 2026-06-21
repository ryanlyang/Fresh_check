"""Five-view dataset and collator for set-matching multi-view taggers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetIdentity

from .experiment import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_TOKENS_PER_VIEW,
    DEFAULT_MIN_TOKENS_PER_VIEW,
    HLT_VIEW_NAME,
    SET_RECONSTRUCTOR_ARCHITECTURES,
    SOURCE_TYPE_ORIGINAL_HLT,
    SOURCE_TYPE_RECONSTRUCTED,
    VIEW_NAMES,
    SetMatchingMultiViewLayout,
    normalize_set_reconstructor_architecture,
    normalize_split_name,
    normalize_view_name,
    view_name_for_reconstructor,
)


SET_MATCHING_FIVE_VIEW_DATA_STEP = "set_matching_multiview_step7_five_view_dataset"
FIVE_VIEW_SELECTION_MODES: tuple[str, str] = ("topk_or_threshold", "all_slots")
FIVE_VIEW_SOURCE_TYPE_IDS = {
    SOURCE_TYPE_ORIGINAL_HLT: 0,
    SOURCE_TYPE_RECONSTRUCTED: 1,
}


def _positive_int(value: int, *, field_name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _nonnegative_int(value: int, *, field_name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return value


def _as_float32_3d(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D [jets, tokens, features], got {array.shape}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return array


def _as_bool_2d(name: str, value: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match expected {expected_shape}")
    return array


def _as_float32_2d(name: str, value: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match expected {expected_shape}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return array


def _metadata_path_for_reco_cache(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metadata.json")


def _load_optional_metadata(path: Path) -> dict[str, Any]:
    metadata_path = _metadata_path_for_reco_cache(path)
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _reco_identity_rows(
    *,
    jet_files: Sequence[str],
    file_indices: np.ndarray,
    entries: np.ndarray,
    labels: np.ndarray,
) -> list[JetIdentity]:
    files = [str(item) for item in jet_files]
    return [
        JetIdentity(file=files[int(file_index)], entry=int(entry), label=int(label))
        for file_index, entry, label in zip(file_indices, entries, labels)
    ]


def _stable_unique(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.int64, copy=False)
    return np.array(list(dict.fromkeys(int(value) for value in values.tolist())), dtype=np.int64)


def _normalize_label_filter(label_filter: Sequence[int] | None) -> tuple[int, ...]:
    if label_filter is None:
        return ()
    labels = tuple(int(label) for label in label_filter)
    if len(set(labels)) != len(labels):
        raise ValueError(f"label_filter contains duplicates: {labels}")
    return labels


def _filter_cache_arrays_by_labels(view: "FiveViewCacheArrays", label_filter: Sequence[int]) -> "FiveViewCacheArrays":
    labels = _normalize_label_filter(label_filter)
    if not labels:
        return view
    keep_labels = set(labels)
    keep = np.asarray([int(label) in keep_labels for label in view.labels], dtype=bool)
    metadata = dict(view.metadata)
    metadata.update(
        {
            "label_filter_applied": True,
            "label_filter": list(labels),
            "n_jets_before_label_filter": int(view.labels.shape[0]),
            "n_jets_after_label_filter": int(np.sum(keep)),
        }
    )
    return FiveViewCacheArrays(
        name=view.name,
        tokens=view.tokens[keep].copy(),
        mask=view.mask[keep].copy(),
        confidence=view.confidence[keep].copy(),
        labels=view.labels[keep].copy(),
        jet_ids=[identity for identity, keep_row in zip(view.jet_ids, keep) if bool(keep_row)],
        source_type=view.source_type,
        metadata=metadata,
    )


def _confidence_order(indices: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    if indices.size <= 1:
        return indices.astype(np.int64, copy=False)
    order = np.lexsort((indices, -confidence[indices]))
    return indices[order].astype(np.int64, copy=False)


def _select_indices(
    *,
    mask: np.ndarray,
    confidence: np.ndarray,
    is_hlt: bool,
    selection_mode: str,
    max_tokens_per_view: int,
    min_tokens_per_view: int,
    confidence_threshold: float,
) -> np.ndarray:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return valid.astype(np.int64, copy=False)
    if selection_mode == "all_slots" or is_hlt:
        return valid[:max_tokens_per_view].astype(np.int64, copy=False)
    above_threshold = valid[confidence[valid] >= float(confidence_threshold)]
    if above_threshold.size < int(min_tokens_per_view):
        top_min = _confidence_order(valid, confidence)[: min(int(min_tokens_per_view), valid.size)]
        selected = _stable_unique(np.concatenate([above_threshold, top_min], axis=0))
    else:
        selected = above_threshold.astype(np.int64, copy=False)
    selected = _confidence_order(selected, confidence)
    return selected[:max_tokens_per_view].astype(np.int64, copy=False)


def _filter_view_tokens(
    *,
    tokens: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    is_hlt: bool,
    selection_mode: str,
    max_tokens_per_view: int,
    min_tokens_per_view: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n_jets, _, feature_dim = tokens.shape
    output_tokens = np.zeros((n_jets, int(max_tokens_per_view), feature_dim), dtype=np.float32)
    output_mask = np.zeros((n_jets, int(max_tokens_per_view)), dtype=bool)
    output_confidence = np.zeros((n_jets, int(max_tokens_per_view)), dtype=np.float32)
    source_indices = np.full((n_jets, int(max_tokens_per_view)), -1, dtype=np.int32)
    kept_counts = np.zeros((n_jets,), dtype=np.int32)
    raw_counts = mask.sum(axis=1).astype(np.int32)
    threshold_counts = ((confidence >= float(confidence_threshold)) & mask).sum(axis=1).astype(np.int32)

    for jet_index in range(n_jets):
        selected = _select_indices(
            mask=mask[jet_index],
            confidence=confidence[jet_index],
            is_hlt=is_hlt,
            selection_mode=selection_mode,
            max_tokens_per_view=int(max_tokens_per_view),
            min_tokens_per_view=int(min_tokens_per_view),
            confidence_threshold=float(confidence_threshold),
        )
        length = int(selected.size)
        if length:
            output_tokens[jet_index, :length] = tokens[jet_index, selected]
            output_mask[jet_index, :length] = True
            output_confidence[jet_index, :length] = confidence[jet_index, selected]
            source_indices[jet_index, :length] = selected.astype(np.int32, copy=False)
        kept_counts[jet_index] = length

    diagnostics = {
        "raw_count_min": float(raw_counts.min()) if raw_counts.size else 0.0,
        "raw_count_mean": float(raw_counts.mean()) if raw_counts.size else 0.0,
        "raw_count_max": float(raw_counts.max()) if raw_counts.size else 0.0,
        "threshold_count_mean": float(threshold_counts.mean()) if threshold_counts.size else 0.0,
        "kept_count_min": float(kept_counts.min()) if kept_counts.size else 0.0,
        "kept_count_mean": float(kept_counts.mean()) if kept_counts.size else 0.0,
        "kept_count_max": float(kept_counts.max()) if kept_counts.size else 0.0,
        "truncated_fraction": float(np.mean(raw_counts > int(max_tokens_per_view))) if raw_counts.size else 0.0,
    }
    return output_tokens, output_mask, output_confidence, source_indices, diagnostics


@dataclass(frozen=True)
class FiveViewCacheArrays:
    """One raw loaded view before final five-view stacking/filtering."""

    name: str
    tokens: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    labels: np.ndarray
    jet_ids: list[JetIdentity]
    source_type: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        name = normalize_view_name(self.name)
        tokens = _as_float32_3d("tokens", self.tokens)
        mask = _as_bool_2d("mask", self.mask, tokens.shape[:2])
        confidence = _as_float32_2d("confidence", self.confidence, tokens.shape[:2])
        labels = np.asarray(self.labels, dtype=np.int64)
        if labels.shape != (tokens.shape[0],):
            raise ValueError(f"labels shape {labels.shape} does not match token rows {tokens.shape[0]}")
        if len(self.jet_ids) != int(tokens.shape[0]):
            raise ValueError("jet_ids length does not match token rows")
        if self.source_type not in FIVE_VIEW_SOURCE_TYPE_IDS:
            raise ValueError(f"unknown source_type {self.source_type!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "confidence", np.where(mask, confidence, 0.0).astype(np.float32))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class FiveViewJetSample:
    """One five-view jet sample after view filtering."""

    view_features: np.ndarray
    view_masks: np.ndarray
    view_confidence: np.ndarray
    view_ids: np.ndarray
    source_type_ids: np.ndarray
    source_indices: np.ndarray
    label: int
    jet_id: JetIdentity
    index: int
    split: str


@dataclass
class FiveViewDatasetConfig:
    """Configuration for loading/filtering five-view cached inputs."""

    output_dir: str
    hlt_cache_dir: str
    split: str
    reconstructed_view_dir: str | None = None
    architectures: tuple[str, ...] = SET_RECONSTRUCTOR_ARCHITECTURES
    max_tokens_per_view: int = DEFAULT_MAX_TOKENS_PER_VIEW
    min_tokens_per_view: int = DEFAULT_MIN_TOKENS_PER_VIEW
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    selection_mode: str = "topk_or_threshold"
    drop_views: tuple[str, ...] = ()
    label_filter: tuple[int, ...] = ()
    shuffle_view_labels: bool = False
    view_label_shuffle_seed: int = 1205
    keep_hlt_label_anchor: bool = True
    verify_hlt_hash: bool = True

    def __post_init__(self) -> None:
        self.split = normalize_split_name(self.split)
        self.architectures = tuple(normalize_set_reconstructor_architecture(value) for value in self.architectures)
        if self.architectures != SET_RECONSTRUCTOR_ARCHITECTURES:
            raise ValueError(f"architectures must be exactly {SET_RECONSTRUCTOR_ARCHITECTURES}")
        self.max_tokens_per_view = _positive_int(self.max_tokens_per_view, field_name="max_tokens_per_view")
        self.min_tokens_per_view = _nonnegative_int(self.min_tokens_per_view, field_name="min_tokens_per_view")
        if int(self.min_tokens_per_view) > int(self.max_tokens_per_view):
            raise ValueError("min_tokens_per_view cannot exceed max_tokens_per_view")
        if self.selection_mode not in FIVE_VIEW_SELECTION_MODES:
            raise ValueError(f"selection_mode must be one of {FIVE_VIEW_SELECTION_MODES}")
        self.confidence_threshold = float(self.confidence_threshold)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.drop_views = tuple(normalize_view_name(view) for view in self.drop_views)
        if len(set(self.drop_views)) != len(self.drop_views):
            raise ValueError(f"drop_views contains duplicates: {self.drop_views}")
        self.label_filter = _normalize_label_filter(self.label_filter)

    @property
    def layout(self) -> SetMatchingMultiViewLayout:
        output_path = Path(self.output_dir)
        return SetMatchingMultiViewLayout(output_root=output_path.parent, experiment_name=output_path.name)

    @property
    def view_names(self) -> tuple[str, ...]:
        return (HLT_VIEW_NAME,) + tuple(view_name_for_reconstructor(arch) for arch in self.architectures)

    def reco_cache_path(self, architecture: str) -> Path:
        arch = normalize_set_reconstructor_architecture(architecture)
        if self.reconstructed_view_dir is not None:
            return Path(self.reconstructed_view_dir) / arch / f"{self.split}_reconstructed_view.npz"
        return self.layout.reconstructed_view_cache_path(arch, self.split)


class FiveViewJetDataset:
    """Dataset over HLT plus four cached set-matching reconstructed views."""

    def __init__(
        self,
        *,
        view_features: np.ndarray,
        view_masks: np.ndarray,
        view_confidence: np.ndarray,
        labels: np.ndarray,
        jet_ids: Sequence[JetIdentity],
        split: str,
        view_names: Sequence[str],
        source_types: Sequence[str],
        view_ids: Sequence[int],
        source_type_ids: Sequence[int],
        source_indices: np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.view_features = np.asarray(view_features, dtype=np.float32)
        self.view_masks = np.asarray(view_masks, dtype=bool)
        self.view_confidence = np.asarray(view_confidence, dtype=np.float32)
        if self.view_features.ndim != 4:
            raise ValueError(f"view_features must be 4D, got {self.view_features.shape}")
        if self.view_masks.shape != self.view_features.shape[:3]:
            raise ValueError("view_masks shape must match view_features first three dimensions")
        if self.view_confidence.shape != self.view_features.shape[:3]:
            raise ValueError("view_confidence shape must match view_features first three dimensions")
        self.labels = np.asarray(labels, dtype=np.int64)
        if self.labels.shape != (self.view_features.shape[0],):
            raise ValueError("labels length must match number of jets")
        self.jet_ids = list(jet_ids)
        if len(self.jet_ids) != int(self.view_features.shape[0]):
            raise ValueError("jet_ids length must match number of jets")
        self.split = normalize_split_name(split)
        self.view_names = tuple(normalize_view_name(name) for name in view_names)
        self.source_types = tuple(str(value) for value in source_types)
        if len(self.view_names) != int(self.view_features.shape[1]):
            raise ValueError("view_names length must match number of views")
        if len(self.source_types) != len(self.view_names):
            raise ValueError("source_types length must match view_names")
        self.view_ids = np.asarray(view_ids, dtype=np.int64)
        self.source_type_ids = np.asarray(source_type_ids, dtype=np.int64)
        if self.view_ids.shape != (len(self.view_names),):
            raise ValueError("view_ids length must match number of views")
        if self.source_type_ids.shape != (len(self.view_names),):
            raise ValueError("source_type_ids length must match number of views")
        if source_indices is None:
            self.source_indices = np.full(self.view_masks.shape, -1, dtype=np.int32)
        else:
            self.source_indices = np.asarray(source_indices, dtype=np.int32)
            if self.source_indices.shape != self.view_masks.shape:
                raise ValueError("source_indices shape must match view_masks")
        self.metadata = dict(metadata or {})

    @classmethod
    def from_caches(
        cls,
        config: FiveViewDatasetConfig,
        *,
        reconstructed_view_paths: Mapping[str, str | Path] | None = None,
    ) -> "FiveViewJetDataset":
        hlt = load_cached_hlt_view(config.hlt_cache_dir, config.split, verify_hash=bool(config.verify_hlt_hash))
        hlt_confidence = np.where(hlt.mask, 1.0, 0.0).astype(np.float32)
        views = [
            FiveViewCacheArrays(
                name=HLT_VIEW_NAME,
                tokens=hlt.tokens,
                mask=hlt.mask,
                confidence=hlt_confidence,
                labels=hlt.labels,
                jet_ids=list(hlt.jet_ids),
                source_type=SOURCE_TYPE_ORIGINAL_HLT,
                metadata=hlt.metadata,
            )
        ]
        reco_metadata: dict[str, Any] = {}
        for architecture in config.architectures:
            view_name = view_name_for_reconstructor(architecture)
            if reconstructed_view_paths and architecture in reconstructed_view_paths:
                path = Path(reconstructed_view_paths[architecture])
            elif reconstructed_view_paths and view_name in reconstructed_view_paths:
                path = Path(reconstructed_view_paths[view_name])
            else:
                path = config.reco_cache_path(architecture)
            arrays = load_reconstructed_view_cache(path, expected_architecture=architecture, expected_split=config.split)
            views.append(arrays)
            reco_metadata[view_name] = dict(arrays.metadata)
        if config.label_filter:
            views = [_filter_cache_arrays_by_labels(view, config.label_filter) for view in views]
        report = audit_five_view_alignment(views)
        if not report.get("ok", False):
            raise ValueError(f"Five-view alignment audit failed: {report['problems']}")
        return build_five_view_dataset_from_arrays(
            views,
            config=config,
            alignment_audit=report,
            extra_metadata={"reconstructed_view_metadata": reco_metadata},
        )

    @property
    def n_views(self) -> int:
        return int(self.view_features.shape[1])

    @property
    def max_tokens_per_view(self) -> int:
        return int(self.view_features.shape[2])

    @property
    def feature_dim(self) -> int:
        return int(self.view_features.shape[3])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> FiveViewJetSample:
        idx = int(index)
        return FiveViewJetSample(
            view_features=self.view_features[idx],
            view_masks=self.view_masks[idx],
            view_confidence=self.view_confidence[idx],
            view_ids=self.view_ids,
            source_type_ids=self.source_type_ids,
            source_indices=self.source_indices[idx],
            label=int(self.labels[idx]),
            jet_id=self.jet_ids[idx],
            index=idx,
            split=self.split,
        )


def load_reconstructed_view_cache(
    path: str | Path,
    *,
    expected_architecture: str | None = None,
    expected_split: str | None = None,
) -> FiveViewCacheArrays:
    """Load one Step 6 reconstructed-view NPZ and sidecar metadata."""

    cache_path = Path(path)
    metadata = _load_optional_metadata(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        tokens = data["tokens"] if "tokens" in data.files else data["features"]
        mask = data["mask"] if "mask" in data.files else data["candidate_mask"]
        if "confidence" in data.files:
            confidence = data["confidence"]
        elif "candidate_weights" in data.files:
            confidence = data["candidate_weights"]
        elif "existence_logits" in data.files:
            confidence = 1.0 / (1.0 + np.exp(-data["existence_logits"]))
        else:
            confidence = np.asarray(mask, dtype=np.float32)
        labels = data["labels"].astype(np.int64, copy=False)
        file_indices = data["jet_file_indices"].astype(np.int64, copy=False)
        entries = data["jet_entries"].astype(np.int64, copy=False)
    architecture = metadata.get("architecture")
    if expected_architecture is not None:
        expected_architecture = normalize_set_reconstructor_architecture(expected_architecture)
        if architecture is not None and normalize_set_reconstructor_architecture(str(architecture)) != expected_architecture:
            raise ValueError(f"reconstructed-view cache {cache_path} architecture {architecture!r} != {expected_architecture!r}")
        architecture = expected_architecture
    if architecture is None:
        raise ValueError(f"reconstructed-view cache metadata is missing architecture: {cache_path}")
    if expected_split is not None:
        expected_split = normalize_split_name(expected_split)
        split = metadata.get("split")
        if split is not None and str(split) != expected_split:
            raise ValueError(f"reconstructed-view cache {cache_path} split {split!r} != {expected_split!r}")
    jet_files = metadata.get("jet_files")
    if not jet_files:
        raise ValueError(f"reconstructed-view cache metadata is missing jet_files: {cache_path}")
    jet_ids = _reco_identity_rows(
        jet_files=jet_files,
        file_indices=file_indices,
        entries=entries,
        labels=labels,
    )
    return FiveViewCacheArrays(
        name=view_name_for_reconstructor(str(architecture)),
        tokens=tokens,
        mask=mask,
        confidence=confidence,
        labels=labels,
        jet_ids=jet_ids,
        source_type=SOURCE_TYPE_RECONSTRUCTED,
        metadata=metadata,
    )


def audit_five_view_alignment(views: Sequence[FiveViewCacheArrays]) -> dict[str, Any]:
    """Check that all loaded views have identical rows, labels, identities, and feature dimensions."""

    if not views:
        raise ValueError("at least one view is required")
    reference = views[0]
    problems: list[str] = []
    identity_hashes = {reference.name: jet_identity_hash(reference.jet_ids)}
    label_hashes = {reference.name: hash_arrays({"labels": reference.labels})}
    feature_dims = {reference.name: int(reference.tokens.shape[-1])}
    row_counts = {reference.name: int(reference.tokens.shape[0])}
    for view in views[1:]:
        identity_hash = jet_identity_hash(view.jet_ids)
        identity_hashes[view.name] = identity_hash
        label_hashes[view.name] = hash_arrays({"labels": view.labels})
        feature_dims[view.name] = int(view.tokens.shape[-1])
        row_counts[view.name] = int(view.tokens.shape[0])
        if int(view.tokens.shape[0]) != int(reference.tokens.shape[0]):
            problems.append(f"{view.name} row count {view.tokens.shape[0]} != {reference.name} {reference.tokens.shape[0]}")
        if int(view.tokens.shape[-1]) != int(reference.tokens.shape[-1]):
            problems.append(
                f"{view.name} feature_dim {view.tokens.shape[-1]} != {reference.name} {reference.tokens.shape[-1]}"
            )
        if not np.array_equal(view.labels, reference.labels):
            problems.append(f"{view.name} labels do not match {reference.name}")
        if identity_hash != identity_hashes[reference.name]:
            problems.append(f"{view.name} jet identity hash does not match {reference.name}")
    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "view_names": [view.name for view in views],
        "row_counts": row_counts,
        "feature_dims": feature_dims,
        "identity_hashes": identity_hashes,
        "label_hashes": label_hashes,
        "reference_view": reference.name,
        "reference_identity_hash": identity_hashes[reference.name],
    }


def _view_ids_for_dataset(view_names: Sequence[str]) -> np.ndarray:
    return np.arange(len(view_names), dtype=np.int64)


def _stable_split_seed_offset(split: str) -> int:
    text = normalize_split_name(split)
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 1_000_003
    return int(value)


def _shuffle_reconstructed_view_contents_per_jet(
    *,
    view_features: np.ndarray,
    view_masks: np.ndarray,
    view_confidence: np.ndarray,
    source_indices: np.ndarray,
    view_names: Sequence[str],
    seed: int,
    split: str,
    keep_hlt_anchor: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Mismatch reconstructed view contents against stable semantic view ids.

    The first version of the shuffle control only permuted ``view_ids`` once
    for the whole dataset.  A transformer can simply relearn that stable
    relabeling.  This control instead keeps the semantic view labels fixed and
    independently shuffles the reconstructed-view contents for each jet.  HLT is
    anchored by default, so the control asks whether stable labels for the four
    reconstructed sources carry usable information.
    """

    if len(view_names) < 2:
        return view_features, view_masks, view_confidence, source_indices, {
            "applied": False,
            "reason": "fewer_than_two_views",
        }
    if keep_hlt_anchor and normalize_view_name(view_names[0]) == HLT_VIEW_NAME:
        semantic_positions = np.arange(1, len(view_names), dtype=np.int64)
    else:
        semantic_positions = np.arange(len(view_names), dtype=np.int64)
    if semantic_positions.size <= 1:
        return view_features, view_masks, view_confidence, source_indices, {
            "applied": False,
            "reason": "fewer_than_two_shuffleable_views",
        }

    rng = np.random.RandomState(int(seed) + _stable_split_seed_offset(split))
    shuffled_features = view_features.copy()
    shuffled_masks = view_masks.copy()
    shuffled_confidence = view_confidence.copy()
    shuffled_indices = source_indices.copy()
    examples: list[dict[str, Any]] = []
    unchanged_count = 0

    for jet_index in range(int(view_features.shape[0])):
        source_positions = semantic_positions.copy()
        rng.shuffle(source_positions)
        if np.array_equal(source_positions, semantic_positions):
            unchanged_count += 1
            source_positions = np.roll(source_positions, 1)
        shuffled_features[jet_index, semantic_positions] = view_features[jet_index, source_positions]
        shuffled_masks[jet_index, semantic_positions] = view_masks[jet_index, source_positions]
        shuffled_confidence[jet_index, semantic_positions] = view_confidence[jet_index, source_positions]
        shuffled_indices[jet_index, semantic_positions] = source_indices[jet_index, source_positions]
        if len(examples) < 5:
            examples.append(
                {
                    "jet_index": int(jet_index),
                    "semantic_view_names": [str(view_names[int(index)]) for index in semantic_positions],
                    "content_source_view_names": [str(view_names[int(index)]) for index in source_positions],
                }
            )

    return shuffled_features, shuffled_masks, shuffled_confidence, shuffled_indices, {
        "applied": True,
        "mode": "per_jet_reconstructed_content_shuffle_with_fixed_view_ids",
        "seed": int(seed),
        "effective_seed": int(seed) + _stable_split_seed_offset(split),
        "keep_hlt_anchor": bool(keep_hlt_anchor),
        "shuffleable_view_names": [str(view_names[int(index)]) for index in semantic_positions],
        "n_jets": int(view_features.shape[0]),
        "unchanged_random_permutation_count_before_forced_roll": int(unchanged_count),
        "examples": examples,
    }


def build_five_view_dataset_from_arrays(
    views: Sequence[FiveViewCacheArrays],
    *,
    config: FiveViewDatasetConfig,
    alignment_audit: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> FiveViewJetDataset:
    """Build a filtered/stacked five-view dataset from loaded view arrays."""

    if config.label_filter:
        views = tuple(_filter_cache_arrays_by_labels(view, config.label_filter) for view in views)
    view_names = tuple(view.name for view in views)
    if view_names != VIEW_NAMES:
        raise ValueError(f"views must be ordered as {VIEW_NAMES}, got {view_names}")
    alignment_report = dict(alignment_audit or audit_five_view_alignment(views))
    if not alignment_report.get("ok", False):
        raise ValueError(f"Five-view alignment audit failed: {alignment_report['problems']}")
    source_types = tuple(view.source_type for view in views)
    filtered_tokens = []
    filtered_masks = []
    filtered_confidences = []
    filtered_indices = []
    filtering_diagnostics = {}
    for view in views:
        tokens, mask, confidence, source_indices, diagnostics = _filter_view_tokens(
            tokens=view.tokens,
            mask=view.mask,
            confidence=view.confidence,
            is_hlt=view.name == HLT_VIEW_NAME,
            selection_mode=config.selection_mode,
            max_tokens_per_view=int(config.max_tokens_per_view),
            min_tokens_per_view=int(config.min_tokens_per_view),
            confidence_threshold=float(config.confidence_threshold),
        )
        filtered_tokens.append(tokens)
        filtered_masks.append(mask)
        filtered_confidences.append(confidence)
        filtered_indices.append(source_indices)
        filtering_diagnostics[view.name] = diagnostics

    view_features = np.stack(filtered_tokens, axis=1).astype(np.float32, copy=False)
    view_masks = np.stack(filtered_masks, axis=1).astype(bool, copy=False)
    view_confidence = np.stack(filtered_confidences, axis=1).astype(np.float32, copy=False)
    source_indices = np.stack(filtered_indices, axis=1).astype(np.int32, copy=False)

    shuffle_control_diagnostics = {"applied": False}
    if bool(config.shuffle_view_labels):
        (
            view_features,
            view_masks,
            view_confidence,
            source_indices,
            shuffle_control_diagnostics,
        ) = _shuffle_reconstructed_view_contents_per_jet(
            view_features=view_features,
            view_masks=view_masks,
            view_confidence=view_confidence,
            source_indices=source_indices,
            view_names=view_names,
            seed=int(config.view_label_shuffle_seed),
            split=config.split,
            keep_hlt_anchor=bool(config.keep_hlt_label_anchor),
        )

    dropped = set(config.drop_views)
    for view_index, view_name in enumerate(view_names):
        if view_name not in dropped:
            continue
        view_features[:, view_index] = 0.0
        view_masks[:, view_index] = False
        view_confidence[:, view_index] = 0.0
        source_indices[:, view_index] = -1

    view_ids = _view_ids_for_dataset(view_names)
    source_type_ids = np.asarray([FIVE_VIEW_SOURCE_TYPE_IDS[source_type] for source_type in source_types], dtype=np.int64)
    labels = views[0].labels
    content_hash = hash_arrays(
        {
            "view_features": view_features,
            "view_masks": view_masks,
            "view_confidence": view_confidence,
            "source_indices": source_indices,
            "labels": labels,
            "view_ids": view_ids,
            "source_type_ids": source_type_ids,
        }
    )
    metadata = {
        "experiment_step": SET_MATCHING_FIVE_VIEW_DATA_STEP,
        "split": config.split,
        "view_names": list(view_names),
        "source_types": list(source_types),
        "n_views": int(len(view_names)),
        "n_jets": int(view_features.shape[0]),
        "max_tokens_per_view": int(view_features.shape[2]),
        "feature_dim": int(view_features.shape[3]),
        "selection_mode": config.selection_mode,
        "confidence_threshold": float(config.confidence_threshold),
        "min_tokens_per_view": int(config.min_tokens_per_view),
        "drop_views": list(config.drop_views),
        "label_filter": list(config.label_filter),
        "label_filter_applied": bool(config.label_filter),
        "shuffle_view_labels": bool(config.shuffle_view_labels),
        "view_label_shuffle_control": shuffle_control_diagnostics,
        "view_ids": view_ids.astype(int).tolist(),
        "source_type_ids": source_type_ids.astype(int).tolist(),
        "jet_identity_hash": jet_identity_hash(views[0].jet_ids),
        "five_view_content_hash": content_hash,
        "alignment_audit": alignment_report,
        "filtering_diagnostics": filtering_diagnostics,
        "config": config.__dict__.copy(),
    }
    metadata.update(dict(extra_metadata or {}))
    return FiveViewJetDataset(
        view_features=view_features,
        view_masks=view_masks,
        view_confidence=view_confidence,
        labels=labels,
        jet_ids=views[0].jet_ids,
        split=config.split,
        view_names=view_names,
        source_types=source_types,
        view_ids=view_ids,
        source_type_ids=source_type_ids,
        source_indices=source_indices,
        metadata=metadata,
    )


def _sample_attr(sample: FiveViewJetSample | Mapping[str, Any], name: str) -> Any:
    if isinstance(sample, Mapping):
        return sample[name]
    return getattr(sample, name)


def collate_five_view_samples(
    samples: Sequence[FiveViewJetSample | Mapping[str, Any]],
    *,
    as_torch: bool = True,
) -> dict[str, Any]:
    """Collate five-view samples into tagger-ready batch tensors."""

    if not samples:
        raise ValueError("cannot collate an empty five-view batch")
    splits = {str(_sample_attr(sample, "split")) for sample in samples}
    if len(splits) != 1:
        raise ValueError(f"all samples in a batch must come from one split, got {sorted(splits)}")
    view_features = np.stack([_sample_attr(sample, "view_features") for sample in samples], axis=0).astype(np.float32)
    view_masks = np.stack([_sample_attr(sample, "view_masks") for sample in samples], axis=0).astype(bool)
    view_confidence = np.stack([_sample_attr(sample, "view_confidence") for sample in samples], axis=0).astype(np.float32)
    source_indices = np.stack([_sample_attr(sample, "source_indices") for sample in samples], axis=0).astype(np.int64)
    labels = np.asarray([int(_sample_attr(sample, "label")) for sample in samples], dtype=np.int64)
    indices = np.asarray([int(_sample_attr(sample, "index")) for sample in samples], dtype=np.int64)
    view_ids = np.asarray(_sample_attr(samples[0], "view_ids"), dtype=np.int64)
    source_type_ids = np.asarray(_sample_attr(samples[0], "source_type_ids"), dtype=np.int64)
    for sample in samples[1:]:
        if not np.array_equal(view_ids, np.asarray(_sample_attr(sample, "view_ids"), dtype=np.int64)):
            raise ValueError("all samples must use the same view_ids")
        if not np.array_equal(source_type_ids, np.asarray(_sample_attr(sample, "source_type_ids"), dtype=np.int64)):
            raise ValueError("all samples must use the same source_type_ids")
    batch: dict[str, Any] = {
        "view_features": view_features,
        "view_masks": view_masks,
        "view_confidence": view_confidence,
        "view_ids": view_ids,
        "source_type_ids": source_type_ids,
        "source_indices": source_indices,
        "labels": labels,
        "indices": indices,
        "jet_ids": [_sample_attr(sample, "jet_id") for sample in samples],
        "split": next(iter(splits)),
    }
    if not as_torch:
        return batch
    torch = require_torch()
    return {
        "view_features": torch.from_numpy(view_features).float(),
        "view_masks": torch.from_numpy(view_masks).bool(),
        "view_confidence": torch.from_numpy(view_confidence).float(),
        "view_ids": torch.from_numpy(view_ids).long(),
        "source_type_ids": torch.from_numpy(source_type_ids).long(),
        "source_indices": torch.from_numpy(source_indices).long(),
        "labels": torch.from_numpy(labels).long(),
        "indices": torch.from_numpy(indices).long(),
        "jet_ids": batch["jet_ids"],
        "split": batch["split"],
    }


def make_five_view_loader(
    dataset: FiveViewJetDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 1205,
):
    """Build a PyTorch DataLoader for five-view tagger training."""

    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_five_view_samples,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


__all__ = [
    "FIVE_VIEW_SELECTION_MODES",
    "FIVE_VIEW_SOURCE_TYPE_IDS",
    "SET_MATCHING_FIVE_VIEW_DATA_STEP",
    "FiveViewCacheArrays",
    "FiveViewDatasetConfig",
    "FiveViewJetDataset",
    "FiveViewJetSample",
    "audit_five_view_alignment",
    "build_five_view_dataset_from_arrays",
    "collate_five_view_samples",
    "load_reconstructed_view_cache",
    "make_five_view_loader",
]

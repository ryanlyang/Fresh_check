"""Dual-view HLT + PN reconstructed-view dataset for Step 3.

This branch intentionally consumes only HLT-derived objects at training and
inference time: the cached fixed-HLT view and the cached PN reconstructed view
produced from that HLT view.  The loader reuses the set-matching reconstructed
view cache contract so the PN output does not need a second storage format.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.dual_view import build_part_inputs_torch
from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.hlt_cache import (
    fixed_hlt_params_dict,
    fixed_hlt_params_from_strength,
    hash_arrays,
    jet_identity_hash,
    load_cached_hlt_view,
)
from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.set_matching.experiment import (
    HLT_VIEW_NAME,
    SOURCE_TYPE_ORIGINAL_HLT,
    SOURCE_TYPE_RECONSTRUCTED,
    normalize_set_reconstructor_architecture,
    normalize_split_name,
    view_name_for_reconstructor,
)
from teacher_logit_reco.set_matching.five_view_data import (
    FiveViewCacheArrays,
    audit_five_view_alignment,
    load_reconstructed_view_cache,
)

from .config import (
    DUALVIEW_PART_BINARY_LABEL_FILTER_NAMES,
    DUALVIEW_PART_HLT_DEGRADATION_STRENGTH,
    DUALVIEW_PART_RECONSTRUCTOR_ARCHITECTURE,
    DUALVIEW_PART_SOURCE_LABEL_NAMES,
    DUALVIEW_PART_SPLIT_SIZES,
    DUALVIEW_PART_VIEW_HLT,
    DUALVIEW_PART_VIEW_PN_RECO,
    DualViewPartExperimentLayout,
    default_dualview_part_layout,
)


DUALVIEW_PART_STEP3 = "reliability_gated_dualview_part_step3_dataset"
DUALVIEW_PART_DATA_CONTRACT = "hlt_part_inputs_plus_pn_reco_cache_v1"
DUALVIEW_PART_SELECTION_MODES: tuple[str, str] = ("topk_or_threshold", "all_slots")
DUALVIEW_PART_COMPACT_LABEL_FILTER: tuple[int, int] = tuple(
    int(value) for value in DUALVIEW_PART_BINARY_LABEL_FILTER_NAMES
)


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


def _as_float32_3d(name: str, value: np.ndarray, *, feature_dim: int = RAW_TOKEN_DIM) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or int(array.shape[-1]) != int(feature_dim):
        raise ValueError(f"{name} must have shape [jets, tokens, {feature_dim}], got {array.shape}")
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


def _normalize_label_filter(label_filter: Sequence[int] | None) -> tuple[int, ...]:
    if label_filter is None:
        return ()
    labels = tuple(int(label) for label in label_filter)
    if len(set(labels)) != len(labels):
        raise ValueError(f"label_filter contains duplicates: {labels}")
    return labels


def _canonical_hlt_params() -> dict[str, float]:
    return fixed_hlt_params_dict(fixed_hlt_params_from_strength(DUALVIEW_PART_HLT_DEGRADATION_STRENGTH))


def _params_match_canonical(params: Mapping[str, Any]) -> bool:
    expected = _canonical_hlt_params()
    for key, expected_value in expected.items():
        if key not in params:
            return False
        if abs(float(params[key]) - float(expected_value)) > 1.0e-9:
            return False
    return True


def _metadata_label_names(metadata: Mapping[str, Any]) -> tuple[str, ...] | None:
    for key in ("source_label_names", "selected_class_names", "label_names", "class_names"):
        value = metadata.get(key)
        if value is not None:
            return tuple(str(item) for item in value)
    return None


def _validate_compact_labels(labels: np.ndarray, *, allowed_labels: Sequence[int]) -> None:
    allowed = set(int(label) for label in allowed_labels)
    present = set(int(label) for label in np.asarray(labels, dtype=np.int64).tolist())
    disallowed = sorted(present - allowed)
    if disallowed:
        raise ValueError(f"Dual-view dataset labels must be compact QCD/Hgg labels {sorted(allowed)}, got {disallowed}")


def _validate_hlt_cache_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    view = str(metadata.get("view", "")).strip().lower()
    if view and view not in {"fixed_hlt", "hlt"}:
        raise ValueError(f"HLT cache must be a fixed-HLT view, got view={view!r}")
    params = metadata.get("hlt_params")
    if not isinstance(params, Mapping) or not _params_match_canonical(params):
        raise ValueError("HLT cache hlt_params do not match the canonical HLT0.6 profile")
    return {
        "view": view or "fixed_hlt",
        "hlt_degradation_strength": DUALVIEW_PART_HLT_DEGRADATION_STRENGTH,
        "hlt_params_match": True,
    }


def _validate_optional_label_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    names = _metadata_label_names(metadata)
    if names is None:
        return {"label_metadata_present": False}
    if names != DUALVIEW_PART_SOURCE_LABEL_NAMES:
        raise ValueError(f"Dual-view source labels must be {DUALVIEW_PART_SOURCE_LABEL_NAMES}, got {names}")
    return {"label_metadata_present": True, "source_label_names": list(names)}


def _filter_rows_by_labels(
    *,
    hlt: FiveViewCacheArrays,
    pn: FiveViewCacheArrays,
    label_filter: Sequence[int],
) -> tuple[FiveViewCacheArrays, FiveViewCacheArrays, dict[str, Any]]:
    labels = _normalize_label_filter(label_filter)
    if not labels:
        return hlt, pn, {"applied": False}
    keep_labels = set(labels)
    keep = np.asarray([int(label) in keep_labels for label in hlt.labels], dtype=bool)
    if not np.array_equal(keep, np.asarray([int(label) in keep_labels for label in pn.labels], dtype=bool)):
        raise ValueError("HLT and PN label filters would keep different rows")

    def _slice(view: FiveViewCacheArrays) -> FiveViewCacheArrays:
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

    return _slice(hlt), _slice(pn), {
        "applied": True,
        "label_filter": list(labels),
        "n_jets_before_label_filter": int(hlt.labels.shape[0]),
        "n_jets_after_label_filter": int(np.sum(keep)),
    }


def _confidence_order(indices: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    if indices.size <= 1:
        return indices.astype(np.int64, copy=False)
    order = np.lexsort((indices, -confidence[indices]))
    return indices[order].astype(np.int64, copy=False)


def _stable_unique(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.int64, copy=False)
    return np.array(list(dict.fromkeys(int(value) for value in values.tolist())), dtype=np.int64)


def _select_pn_indices(
    *,
    mask: np.ndarray,
    confidence: np.ndarray,
    selection_mode: str,
    max_tokens: int,
    min_tokens: int,
    confidence_threshold: float,
) -> np.ndarray:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return valid.astype(np.int64, copy=False)
    if selection_mode == "all_slots":
        return valid[:max_tokens].astype(np.int64, copy=False)
    above_threshold = valid[confidence[valid] >= float(confidence_threshold)]
    if above_threshold.size < int(min_tokens):
        top_min = _confidence_order(valid, confidence)[: min(int(min_tokens), valid.size)]
        selected = _stable_unique(np.concatenate([above_threshold, top_min], axis=0))
    else:
        selected = above_threshold.astype(np.int64, copy=False)
    selected = _confidence_order(selected, confidence)
    return selected[:max_tokens].astype(np.int64, copy=False)


def _select_pn_tokens(
    *,
    tokens: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    selection_mode: str,
    max_tokens: int,
    min_tokens: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n_jets, _, feature_dim = tokens.shape
    selected_tokens = np.zeros((n_jets, int(max_tokens), feature_dim), dtype=np.float32)
    selected_mask = np.zeros((n_jets, int(max_tokens)), dtype=bool)
    selected_confidence = np.zeros((n_jets, int(max_tokens)), dtype=np.float32)
    source_indices = np.full((n_jets, int(max_tokens)), -1, dtype=np.int32)
    raw_counts = mask.sum(axis=1).astype(np.int32)
    kept_counts = np.zeros((n_jets,), dtype=np.int32)
    threshold_counts = ((confidence >= float(confidence_threshold)) & mask).sum(axis=1).astype(np.int32)

    for jet_index in range(n_jets):
        selected = _select_pn_indices(
            mask=mask[jet_index],
            confidence=confidence[jet_index],
            selection_mode=selection_mode,
            max_tokens=int(max_tokens),
            min_tokens=int(min_tokens),
            confidence_threshold=float(confidence_threshold),
        )
        length = int(selected.size)
        if length:
            selected_tokens[jet_index, :length] = tokens[jet_index, selected]
            selected_mask[jet_index, :length] = True
            selected_confidence[jet_index, :length] = confidence[jet_index, selected]
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
        "truncated_fraction": float(np.mean(raw_counts > int(max_tokens))) if raw_counts.size else 0.0,
    }
    return selected_tokens, selected_mask, selected_confidence, source_indices, diagnostics


@dataclass
class DualViewPartDatasetConfig:
    """Configuration for loading one dual-view split from cached files."""

    output_dir: str | None = None
    hlt_cache_dir: str | None = None
    pn_reconstructed_view_dir: str | None = None
    split: str = "stack_train"
    pn_reconstructor_architecture: str = DUALVIEW_PART_RECONSTRUCTOR_ARCHITECTURE
    max_hlt_constits: int = 128
    max_pn_tokens: int = 128
    min_pn_tokens: int = 8
    confidence_threshold: float = 0.05
    selection_mode: str = "topk_or_threshold"
    label_filter: tuple[int, ...] = DUALVIEW_PART_COMPACT_LABEL_FILTER
    verify_hlt_hash: bool = True
    enforce_canonical_contract: bool = True
    enforce_split_size: bool = False

    def __post_init__(self) -> None:
        self.split = normalize_split_name(self.split)
        self.pn_reconstructor_architecture = normalize_set_reconstructor_architecture(self.pn_reconstructor_architecture)
        if self.pn_reconstructor_architecture != DUALVIEW_PART_RECONSTRUCTOR_ARCHITECTURE:
            raise ValueError("dual-view ParT Step 3 is locked to the PN reconstructed view")
        self.max_hlt_constits = _positive_int(self.max_hlt_constits, field_name="max_hlt_constits")
        self.max_pn_tokens = _positive_int(self.max_pn_tokens, field_name="max_pn_tokens")
        self.min_pn_tokens = _nonnegative_int(self.min_pn_tokens, field_name="min_pn_tokens")
        if int(self.min_pn_tokens) > int(self.max_pn_tokens):
            raise ValueError("min_pn_tokens cannot exceed max_pn_tokens")
        self.confidence_threshold = float(self.confidence_threshold)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.selection_mode not in DUALVIEW_PART_SELECTION_MODES:
            raise ValueError(f"selection_mode must be one of {DUALVIEW_PART_SELECTION_MODES}")
        self.label_filter = _normalize_label_filter(self.label_filter)
        if bool(self.enforce_canonical_contract):
            invalid = sorted(set(self.label_filter) - set(DUALVIEW_PART_COMPACT_LABEL_FILTER))
            if invalid:
                raise ValueError(
                    "canonical dual-view dataset label_filter must be a subset of compact QCD/Hgg labels "
                    f"{DUALVIEW_PART_COMPACT_LABEL_FILTER}, got {self.label_filter}"
                )
        self.enforce_canonical_contract = bool(self.enforce_canonical_contract)
        self.enforce_split_size = bool(self.enforce_split_size)

    @property
    def layout(self) -> DualViewPartExperimentLayout:
        if self.output_dir is None:
            return default_dualview_part_layout()
        output_path = Path(self.output_dir)
        return DualViewPartExperimentLayout(output_root=output_path.parent, experiment_name=output_path.name)

    @property
    def resolved_hlt_cache_dir(self) -> Path:
        if self.hlt_cache_dir is not None:
            return Path(self.hlt_cache_dir)
        return self.layout.hlt_cache_dir

    @property
    def resolved_pn_reconstructed_view_dir(self) -> Path:
        if self.pn_reconstructed_view_dir is not None:
            return Path(self.pn_reconstructed_view_dir)
        return self.layout.pn_reconstructed_view_dir

    @property
    def pn_reconstructed_view_path(self) -> Path:
        return self.resolved_pn_reconstructed_view_dir / f"{self.split}_reconstructed_view.npz"


@dataclass(frozen=True)
class DualViewPartSample:
    """One aligned HLT + PN reconstructed-view sample."""

    hlt_tokens: np.ndarray
    hlt_mask: np.ndarray
    pn_reco_tokens: np.ndarray
    pn_reco_mask: np.ndarray
    pn_reco_confidence: np.ndarray
    pn_source_indices: np.ndarray
    label: int
    jet_id: JetIdentity
    index: int
    split: str


class DualViewPartJetDataset:
    """Dataset over a fixed-HLT view and its aligned PN reconstructed view."""

    def __init__(
        self,
        *,
        hlt_tokens: np.ndarray,
        hlt_mask: np.ndarray,
        pn_reco_tokens: np.ndarray,
        pn_reco_mask: np.ndarray,
        pn_reco_confidence: np.ndarray,
        labels: np.ndarray,
        jet_ids: Sequence[JetIdentity],
        split: str,
        pn_source_indices: np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.hlt_tokens = _as_float32_3d("hlt_tokens", hlt_tokens)
        self.hlt_mask = _as_bool_2d("hlt_mask", hlt_mask, self.hlt_tokens.shape[:2])
        self.pn_reco_tokens = _as_float32_3d("pn_reco_tokens", pn_reco_tokens)
        self.pn_reco_mask = _as_bool_2d("pn_reco_mask", pn_reco_mask, self.pn_reco_tokens.shape[:2])
        self.pn_reco_confidence = _as_float32_2d(
            "pn_reco_confidence",
            pn_reco_confidence,
            self.pn_reco_tokens.shape[:2],
        )
        self.pn_reco_confidence = np.where(self.pn_reco_mask, self.pn_reco_confidence, 0.0).astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        if self.labels.shape != (self.hlt_tokens.shape[0],):
            raise ValueError("labels length must match number of HLT jets")
        if int(self.pn_reco_tokens.shape[0]) != int(self.hlt_tokens.shape[0]):
            raise ValueError("PN reconstructed view row count must match HLT row count")
        self.jet_ids = list(jet_ids)
        if len(self.jet_ids) != int(self.hlt_tokens.shape[0]):
            raise ValueError("jet_ids length must match number of jets")
        self.split = normalize_split_name(split)
        if pn_source_indices is None:
            self.pn_source_indices = np.full(self.pn_reco_mask.shape, -1, dtype=np.int32)
        else:
            self.pn_source_indices = np.asarray(pn_source_indices, dtype=np.int32)
            if self.pn_source_indices.shape != self.pn_reco_mask.shape:
                raise ValueError("pn_source_indices shape must match pn_reco_mask")
        self.metadata = dict(metadata or {})

    @classmethod
    def from_caches(cls, config: DualViewPartDatasetConfig) -> "DualViewPartJetDataset":
        """Load one split from fixed-HLT and PN reconstructed-view caches."""

        hlt_view = load_cached_hlt_view(
            config.resolved_hlt_cache_dir,
            config.split,
            verify_hash=bool(config.verify_hlt_hash),
        )
        contract_audit = {
            "enforced": bool(config.enforce_canonical_contract),
            "expected_label_filter": list(DUALVIEW_PART_COMPACT_LABEL_FILTER),
            "expected_source_label_names": list(DUALVIEW_PART_SOURCE_LABEL_NAMES),
            "expected_hlt_degradation_strength": float(DUALVIEW_PART_HLT_DEGRADATION_STRENGTH),
            "expected_split_size": int(DUALVIEW_PART_SPLIT_SIZES[config.split]),
            "actual_n_jets_before_filter": int(hlt_view.labels.shape[0]),
        }
        if bool(config.enforce_canonical_contract):
            contract_audit["hlt_cache"] = _validate_hlt_cache_contract(hlt_view.metadata)
            contract_audit["label_metadata"] = _validate_optional_label_metadata(hlt_view.metadata)
            _validate_compact_labels(hlt_view.labels, allowed_labels=DUALVIEW_PART_COMPACT_LABEL_FILTER)
            if bool(config.enforce_split_size) and int(hlt_view.labels.shape[0]) != int(DUALVIEW_PART_SPLIT_SIZES[config.split]):
                raise ValueError(
                    f"Dual-view split {config.split} expected {DUALVIEW_PART_SPLIT_SIZES[config.split]} jets, "
                    f"got {hlt_view.labels.shape[0]}"
                )
        hlt_confidence = np.where(hlt_view.mask, 1.0, 0.0).astype(np.float32)
        hlt = FiveViewCacheArrays(
            name=HLT_VIEW_NAME,
            tokens=hlt_view.tokens,
            mask=hlt_view.mask,
            confidence=hlt_confidence,
            labels=hlt_view.labels,
            jet_ids=list(hlt_view.jet_ids),
            source_type=SOURCE_TYPE_ORIGINAL_HLT,
            metadata=hlt_view.metadata,
        )
        pn = load_reconstructed_view_cache(
            config.pn_reconstructed_view_path,
            expected_architecture=DUALVIEW_PART_RECONSTRUCTOR_ARCHITECTURE,
            expected_split=config.split,
        )
        expected_pn_name = view_name_for_reconstructor(DUALVIEW_PART_RECONSTRUCTOR_ARCHITECTURE)
        if pn.name != expected_pn_name:
            raise ValueError(f"expected PN reconstructed view {expected_pn_name!r}, got {pn.name!r}")
        full_alignment_audit = audit_five_view_alignment([hlt, pn])
        if not full_alignment_audit.get("ok", False):
            raise ValueError(f"Dual-view alignment audit failed: {full_alignment_audit['problems']}")
        hlt, pn, label_filter_report = _filter_rows_by_labels(
            hlt=hlt,
            pn=pn,
            label_filter=config.label_filter,
        )
        if bool(config.enforce_canonical_contract):
            _validate_compact_labels(hlt.labels, allowed_labels=config.label_filter)
            _validate_compact_labels(pn.labels, allowed_labels=config.label_filter)
            contract_audit["actual_n_jets_after_filter"] = int(hlt.labels.shape[0])
            contract_audit["label_filter"] = list(config.label_filter)
        alignment_audit = audit_five_view_alignment([hlt, pn])
        if not alignment_audit.get("ok", False):
            raise ValueError(f"Dual-view post-filter alignment audit failed: {alignment_audit['problems']}")
        pn_tokens, pn_mask, pn_confidence, pn_source_indices, filtering_diagnostics = _select_pn_tokens(
            tokens=pn.tokens,
            mask=pn.mask,
            confidence=pn.confidence,
            selection_mode=config.selection_mode,
            max_tokens=int(config.max_pn_tokens),
            min_tokens=int(config.min_pn_tokens),
            confidence_threshold=float(config.confidence_threshold),
        )
        metadata = {
            "experiment_step": DUALVIEW_PART_STEP3,
            "output_contract": DUALVIEW_PART_DATA_CONTRACT,
            "split": config.split,
            "view_names": [DUALVIEW_PART_VIEW_HLT, DUALVIEW_PART_VIEW_PN_RECO],
            "n_jets": int(hlt.tokens.shape[0]),
            "hlt_shape": list(hlt.tokens.shape),
            "pn_reco_shape": list(pn_tokens.shape),
            "max_hlt_constits": int(config.max_hlt_constits),
            "max_pn_tokens": int(config.max_pn_tokens),
            "min_pn_tokens": int(config.min_pn_tokens),
            "selection_mode": config.selection_mode,
            "confidence_threshold": float(config.confidence_threshold),
            "label_filter": list(config.label_filter),
            "label_filter_report": label_filter_report,
            "contract_audit": contract_audit,
            "jet_identity_hash": jet_identity_hash(hlt.jet_ids),
            "content_hash": hash_arrays(
                {
                    "hlt_tokens": hlt.tokens,
                    "hlt_mask": hlt.mask,
                    "pn_reco_tokens": pn_tokens,
                    "pn_reco_mask": pn_mask,
                    "pn_reco_confidence": pn_confidence,
                    "labels": hlt.labels,
                    "pn_source_indices": pn_source_indices,
                }
            ),
            "full_alignment_audit": full_alignment_audit,
            "alignment_audit": alignment_audit,
            "filtering_diagnostics": filtering_diagnostics,
            "hlt_metadata": dict(hlt.metadata),
            "pn_reco_metadata": dict(pn.metadata),
            "config": {
                "output_dir": config.output_dir,
                "hlt_cache_dir": str(config.resolved_hlt_cache_dir),
                "pn_reconstructed_view_dir": str(config.resolved_pn_reconstructed_view_dir),
                "split": config.split,
                "pn_reconstructor_architecture": config.pn_reconstructor_architecture,
                "max_hlt_constits": int(config.max_hlt_constits),
                "max_pn_tokens": int(config.max_pn_tokens),
                "min_pn_tokens": int(config.min_pn_tokens),
                "confidence_threshold": float(config.confidence_threshold),
                "selection_mode": config.selection_mode,
                "label_filter": list(config.label_filter),
                "verify_hlt_hash": bool(config.verify_hlt_hash),
                "enforce_canonical_contract": bool(config.enforce_canonical_contract),
                "enforce_split_size": bool(config.enforce_split_size),
            },
        }
        return cls(
            hlt_tokens=hlt.tokens,
            hlt_mask=hlt.mask,
            pn_reco_tokens=pn_tokens,
            pn_reco_mask=pn_mask,
            pn_reco_confidence=pn_confidence,
            labels=hlt.labels,
            jet_ids=hlt.jet_ids,
            split=config.split,
            pn_source_indices=pn_source_indices,
            metadata=metadata,
        )

    @property
    def n_jets(self) -> int:
        return int(self.labels.shape[0])

    @property
    def pn_max_tokens(self) -> int:
        return int(self.pn_reco_tokens.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.hlt_tokens.shape[-1])

    def __len__(self) -> int:
        return self.n_jets

    def __getitem__(self, index: int) -> DualViewPartSample:
        idx = int(index)
        return DualViewPartSample(
            hlt_tokens=self.hlt_tokens[idx],
            hlt_mask=self.hlt_mask[idx],
            pn_reco_tokens=self.pn_reco_tokens[idx],
            pn_reco_mask=self.pn_reco_mask[idx],
            pn_reco_confidence=self.pn_reco_confidence[idx],
            pn_source_indices=self.pn_source_indices[idx],
            label=int(self.labels[idx]),
            jet_id=self.jet_ids[idx],
            index=idx,
            split=self.split,
        )


def _sample_attr(sample: DualViewPartSample | Mapping[str, Any], name: str) -> Any:
    if isinstance(sample, Mapping):
        return sample[name]
    return getattr(sample, name)


def _numpy_hlt_inputs(tokens: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        mask,
        labels=labels,
        source_view="fixed_hlt",
    )
    return {
        "points": inputs.pf_points,
        "features": inputs.pf_features,
        "lorentz_vectors": inputs.pf_vectors,
        "mask": inputs.pf_mask,
    }


def collate_dualview_part_samples(
    samples: Sequence[DualViewPartSample | Mapping[str, Any]],
    *,
    as_torch: bool = True,
    max_hlt_constits: int = 128,
    hlt_weight_threshold: float = 0.0,
) -> dict[str, Any]:
    """Collate dual-view samples into anchor-ready batches."""

    if not samples:
        raise ValueError("cannot collate an empty dual-view batch")
    splits = {str(_sample_attr(sample, "split")) for sample in samples}
    if len(splits) != 1:
        raise ValueError(f"all samples in a batch must come from one split, got {sorted(splits)}")
    hlt_tokens = np.stack([_sample_attr(sample, "hlt_tokens") for sample in samples], axis=0).astype(np.float32)
    hlt_mask = np.stack([_sample_attr(sample, "hlt_mask") for sample in samples], axis=0).astype(bool)
    pn_reco_tokens = np.stack([_sample_attr(sample, "pn_reco_tokens") for sample in samples], axis=0).astype(np.float32)
    pn_reco_mask = np.stack([_sample_attr(sample, "pn_reco_mask") for sample in samples], axis=0).astype(bool)
    pn_reco_confidence = np.stack(
        [_sample_attr(sample, "pn_reco_confidence") for sample in samples],
        axis=0,
    ).astype(np.float32)
    pn_source_indices = np.stack([_sample_attr(sample, "pn_source_indices") for sample in samples], axis=0).astype(np.int64)
    labels = np.asarray([int(_sample_attr(sample, "label")) for sample in samples], dtype=np.int64)
    indices = np.asarray([int(_sample_attr(sample, "index")) for sample in samples], dtype=np.int64)
    jet_ids = [_sample_attr(sample, "jet_id") for sample in samples]
    split = next(iter(splits))

    if not as_torch:
        return {
            "hlt_inputs": _numpy_hlt_inputs(hlt_tokens, hlt_mask, labels),
            "hlt_tokens": hlt_tokens,
            "hlt_mask": hlt_mask,
            "pn_reco_tokens": pn_reco_tokens,
            "pn_reco_mask": pn_reco_mask,
            "pn_reco_confidence": pn_reco_confidence,
            "pn_source_indices": pn_source_indices,
            "labels": labels,
            "indices": indices,
            "jet_ids": jet_ids,
            "split": split,
        }

    torch = require_torch()
    hlt_tokens_t = torch.from_numpy(hlt_tokens).float()
    hlt_mask_t = torch.from_numpy(hlt_mask).bool()
    return {
        "hlt_inputs": build_part_inputs_torch(
            hlt_tokens_t,
            hlt_mask_t,
            max_constits=int(max_hlt_constits),
            weight_threshold=float(hlt_weight_threshold),
        ),
        "hlt_tokens": hlt_tokens_t,
        "hlt_mask": hlt_mask_t,
        "pn_reco_tokens": torch.from_numpy(pn_reco_tokens).float(),
        "pn_reco_mask": torch.from_numpy(pn_reco_mask).bool(),
        "pn_reco_confidence": torch.from_numpy(pn_reco_confidence).float(),
        "pn_source_indices": torch.from_numpy(pn_source_indices).long(),
        "labels": torch.from_numpy(labels).long(),
        "indices": torch.from_numpy(indices).long(),
        "jet_ids": jet_ids,
        "split": split,
    }


def make_dualview_part_loader(
    dataset: DualViewPartJetDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 2205,
    max_hlt_constits: int = 128,
    hlt_weight_threshold: float = 0.0,
):
    """Build a PyTorch DataLoader for the dual-view ParT branch."""

    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=partial(
            collate_dualview_part_samples,
            max_hlt_constits=int(max_hlt_constits),
            hlt_weight_threshold=float(hlt_weight_threshold),
        ),
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


__all__ = [
    "DUALVIEW_PART_DATA_CONTRACT",
    "DUALVIEW_PART_SELECTION_MODES",
    "DUALVIEW_PART_STEP3",
    "DualViewPartDatasetConfig",
    "DualViewPartJetDataset",
    "DualViewPartSample",
    "collate_dualview_part_samples",
    "make_dualview_part_loader",
]

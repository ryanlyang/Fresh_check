"""Baseline-logit caches for local-graph residual expert training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device
from jetclass_fresh.hlt_cache import load_cached_hlt_view

from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader

from .fusion import (
    binary_margin_from_logits,
    binary_metrics_from_signal_scores,
    sigmoid_np,
)
from .protocol import LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
from .train import _verify_hlt_cache_protocol, load_local_graph_tagger_checkpoint
from .model import LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE


LOCAL_GRAPH_BASELINE_LOGIT_CACHE_STEP = "local_graph_residual_step1_baseline_logit_cache"
LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CONTRACT = "local_graph_residual_baseline_logits_v1"
LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES = (
    "z_base",
    "p_base",
    "delta_tau50",
    "abs_delta_tau50",
    "delta_tau30",
    "near_tau50_weight",
)
LOCAL_GRAPH_RESIDUAL_DEFAULT_SPLITS = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)
LOCAL_GRAPH_BASELINE_LOGIT_CACHE_DEFAULT_METRIC_SPLITS = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
)
LOCAL_GRAPH_BASELINE_CONDITION_REFERENCE_SPLIT = "model_train"
LOCAL_GRAPH_BASELINE_CHECKPOINT_VARIANT = LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _optional_positive_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive when provided")
    return value


def baseline_logit_cache_paths(output_dir: str | Path, split: str) -> tuple[Path, Path]:
    root = Path(output_dir)
    return root / f"{split}_baseline_logits.npz", root / f"{split}_baseline_logits_metadata.json"


def operating_point_from_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    target_signal_efficiency: float,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError(f"labels/scores shape mismatch: {labels.shape} vs {scores.shape}")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("operating-point labels must be encoded as 0/1")
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return {
            "target_signal_efficiency": float(target_signal_efficiency),
            "threshold": float("nan"),
            "signal_efficiency": float("nan"),
            "false_positive_rate": float("nan"),
            "background_rejection": float("nan"),
            "n_signal": int(n_pos),
            "n_background": int(n_neg),
        }
    positive_scores = np.sort(scores[positives])[::-1]
    threshold_index = min(max(int(np.ceil(float(target_signal_efficiency) * n_pos)) - 1, 0), n_pos - 1)
    threshold = float(positive_scores[threshold_index])
    signal_efficiency = float(np.mean(scores[positives] >= threshold))
    false_positive_rate = float(np.mean(scores[negatives] >= threshold))
    return {
        "target_signal_efficiency": float(target_signal_efficiency),
        "threshold": threshold,
        "signal_efficiency": signal_efficiency,
        "false_positive_rate": false_positive_rate,
        "background_rejection": float("inf") if false_positive_rate == 0.0 else float(1.0 / false_positive_rate),
        "n_signal": int(n_pos),
        "n_background": int(n_neg),
    }


def baseline_condition_features(
    z_base: np.ndarray,
    *,
    tau50: float,
    tau30: float,
    near_scale: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    z_base = np.asarray(z_base, dtype=np.float64).reshape(-1)
    if not np.isfinite(z_base).all():
        raise FloatingPointError("z_base contains non-finite values")
    tau50 = float(tau50)
    tau30 = float(tau30)
    delta_tau50 = z_base - tau50
    abs_delta_tau50 = np.abs(delta_tau50)
    if near_scale is None:
        finite_abs = abs_delta_tau50[np.isfinite(abs_delta_tau50)]
        scale = float(np.quantile(finite_abs, 0.25)) if finite_abs.size else 1.0
    else:
        scale = float(near_scale)
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    near_tau50_weight = np.exp(-abs_delta_tau50 / scale)
    features = np.stack(
        [
            z_base,
            sigmoid_np(z_base),
            delta_tau50,
            abs_delta_tau50,
            z_base - tau30,
            near_tau50_weight,
        ],
        axis=1,
    ).astype(np.float32)
    metadata = {
        "feature_names": list(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES),
        "tau50": tau50,
        "tau30": tau30,
        "near_tau50_scale": float(scale),
    }
    return features, metadata


def baseline_condition_reference_from_block(
    block: "LocalGraphBaselineLogitBlock",
    *,
    source_split: str | None = None,
) -> dict[str, Any]:
    """Build one label-dependent conditioning reference from a trusted split.

    The residual model is allowed to know where the frozen HLT ParT operating
    point lies, but that reference must come from the training/selection side
    of the experiment rather than from stack/final labels.
    """

    margin_ops = block.operating_points()["margin"]
    tau50 = float(margin_ops["signal_eff_0p50"]["threshold"])
    tau30 = float(margin_ops["signal_eff_0p30"]["threshold"])
    _features, metadata = baseline_condition_features(block.z_base, tau50=tau50, tau30=tau30)
    return {
        "source_split": str(source_split or block.split),
        "feature_names": list(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES),
        "tau50": float(metadata["tau50"]),
        "tau30": float(metadata["tau30"]),
        "near_tau50_scale": float(metadata["near_tau50_scale"]),
        "n_jets": int(block.labels.shape[0]),
        "label_dependent": True,
        "rule": "Computed once from the reference split and reused for every cached split.",
    }


def _condition_reference_from_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    reference = metadata.get("condition_reference")
    if isinstance(reference, Mapping):
        return reference
    condition = metadata.get("condition_features")
    if isinstance(condition, Mapping):
        nested = condition.get("reference")
        if isinstance(nested, Mapping):
            return nested
    return None


def _condition_features_from_reference(
    z_base: np.ndarray,
    reference: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    features, metadata = baseline_condition_features(
        z_base,
        tau50=float(reference["tau50"]),
        tau30=float(reference["tau30"]),
        near_scale=float(reference["near_tau50_scale"]),
    )
    metadata["reference"] = dict(reference)
    metadata["condition_source"] = "shared_reference"
    return features, metadata


def _condition_metadata_from_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": list(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES),
        "tau50": float(reference["tau50"]),
        "tau30": float(reference["tau30"]),
        "near_tau50_scale": float(reference["near_tau50_scale"]),
        "reference": dict(reference),
        "condition_source": "shared_reference",
    }


@dataclass
class LocalGraphBaselineLogitBlock:
    split: str
    logits: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    condition_features_array: np.ndarray | None = None
    _index_to_position: dict[int, int] = field(init=False, repr=False, default_factory=dict)
    _condition_features_cache: tuple[np.ndarray, dict[str, Any]] | None = field(
        init=False,
        repr=False,
        default=None,
    )

    def __post_init__(self) -> None:
        self.split = str(self.split)
        self.logits = np.asarray(self.logits, dtype=np.float32)
        if self.logits.ndim != 2 or self.logits.shape[1] != 2:
            raise ValueError(f"baseline logits must have shape [N, 2], got {self.logits.shape}")
        if not np.isfinite(self.logits).all():
            raise FloatingPointError(f"{self.split}: baseline logits contain non-finite values")
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        self.indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        if self.labels.shape[0] != self.logits.shape[0]:
            raise ValueError(f"{self.split}: label/logit length mismatch")
        if self.indices.shape[0] != self.logits.shape[0]:
            raise ValueError(f"{self.split}: index/logit length mismatch")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError(f"{self.split}: labels must be encoded as 0/1")
        if self.indices.size and np.unique(self.indices).shape[0] != self.indices.shape[0]:
            raise ValueError(f"{self.split}: baseline cache indices must be unique")
        self._index_to_position = {int(index): int(pos) for pos, index in enumerate(self.indices.tolist())}
        if self.condition_features_array is not None:
            features = np.asarray(self.condition_features_array, dtype=np.float32)
            expected = (self.logits.shape[0], len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES))
            if features.shape != expected:
                raise ValueError(f"{self.split}: condition_features must have shape {expected}, got {features.shape}")
            if not np.isfinite(features).all():
                raise FloatingPointError(f"{self.split}: condition_features contain non-finite values")
            self.condition_features_array = features

    @property
    def z_base(self) -> np.ndarray:
        return binary_margin_from_logits(self.logits).astype(np.float64)

    @property
    def p_base(self) -> np.ndarray:
        return sigmoid_np(self.z_base)

    @property
    def preds(self) -> np.ndarray:
        return (self.p_base >= 0.5).astype(np.int64)

    def metrics(self) -> dict[str, Any]:
        return binary_metrics_from_signal_scores(self.z_base, self.labels)

    def operating_points(self) -> dict[str, Any]:
        prob_metrics = self.metrics().get("binary_metrics", {}).get("fpr_at_signal_efficiency", {})
        return {
            "margin": {
                "signal_eff_0p30": operating_point_from_scores(self.labels, self.z_base, 0.30),
                "signal_eff_0p50": operating_point_from_scores(self.labels, self.z_base, 0.50),
            },
            "probability": prob_metrics,
        }

    def condition_reference(self, *, require: bool = False) -> Mapping[str, Any] | None:
        reference = _condition_reference_from_metadata(self.metadata)
        if reference is None and bool(require):
            raise ValueError(
                f"{self.split}: baseline logit cache is missing a shared condition_reference; "
                "rebuild the baseline-logit cache to avoid held-out label leakage"
            )
        if reference is not None and str(reference.get("source_split")) != LOCAL_GRAPH_BASELINE_CONDITION_REFERENCE_SPLIT:
            raise ValueError(
                f"{self.split}: condition_reference source_split must be "
                f"{LOCAL_GRAPH_BASELINE_CONDITION_REFERENCE_SPLIT!r}, got {reference.get('source_split')!r}"
            )
        return reference

    def positions_for_indices(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        positions: list[int] = []
        missing: list[int] = []
        for index in indices.tolist():
            pos = self._index_to_position.get(int(index))
            if pos is None:
                missing.append(int(index))
            else:
                positions.append(pos)
        if missing:
            preview = ", ".join(str(item) for item in missing[:5])
            raise IndexError(f"{self.split}: baseline logit cache is missing dataset indices: {preview}")
        return np.asarray(positions, dtype=np.int64)

    def condition_features(self, *, require_reference: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        if self._condition_features_cache is not None:
            _features, cached_metadata = self._condition_features_cache
            if not bool(require_reference) or cached_metadata.get("condition_source") == "shared_reference":
                return self._condition_features_cache
        reference = self.condition_reference(require=bool(require_reference))
        if reference is not None:
            if self.condition_features_array is not None:
                output = self.condition_features_array, _condition_metadata_from_reference(reference)
            else:
                output = _condition_features_from_reference(self.z_base, reference)
            self._condition_features_cache = output
            return output
        margin_ops = self.operating_points()["margin"]
        tau50 = margin_ops["signal_eff_0p50"]["threshold"]
        tau30 = margin_ops["signal_eff_0p30"]["threshold"]
        features, metadata = baseline_condition_features(self.z_base, tau50=tau50, tau30=tau30)
        metadata["condition_source"] = "legacy_split_labels"
        metadata["leakage_warning"] = (
            "This fallback uses labels from the same split. Production residual training/reporting "
            "requires a shared condition_reference."
        )
        self._condition_features_cache = (features, metadata)
        return self._condition_features_cache

    def condition_features_for_positions(
        self,
        positions: np.ndarray,
        *,
        require_reference: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        features, metadata = self.condition_features(require_reference=require_reference)
        return features[positions], metadata

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "n_jets": int(self.labels.shape[0]),
            "logits_shape": list(self.logits.shape),
            "index_min": int(np.min(self.indices)) if self.indices.size else None,
            "index_max": int(np.max(self.indices)) if self.indices.size else None,
            "metadata": dict(self.metadata),
        }


def save_baseline_logit_block(
    block: LocalGraphBaselineLogitBlock,
    output_dir: str | Path,
    *,
    condition_reference: Mapping[str, Any] | None = None,
    compute_metrics: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    npz_path, metadata_path = baseline_logit_cache_paths(output_dir, block.split)
    if not bool(overwrite) and (npz_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"baseline cache already exists for split {block.split}: {npz_path}")
    if condition_reference is not None:
        block.metadata["condition_reference"] = dict(condition_reference)
    features, feature_metadata = block.condition_features(require_reference=condition_reference is not None)
    metrics = block.metrics() if bool(compute_metrics) else None
    operating_points = block.operating_points() if bool(compute_metrics) else None
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        logits=block.logits.astype(np.float32, copy=False),
        labels=block.labels.astype(np.int64, copy=False),
        indices=block.indices.astype(np.int64, copy=False),
        z_base=block.z_base.astype(np.float32),
        p_base=block.p_base.astype(np.float32),
        condition_features=features.astype(np.float32, copy=False),
        condition_feature_names=np.asarray(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES),
    )
    metadata = {
        "step": LOCAL_GRAPH_BASELINE_LOGIT_CACHE_STEP,
        "contract": LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CONTRACT,
        **block.to_manifest_row(),
        "npz_path": str(npz_path),
        "metadata_path": str(metadata_path),
        "condition_features": feature_metadata,
        "condition_reference": dict(condition_reference) if condition_reference is not None else None,
        "metrics_computed": bool(compute_metrics),
        "operating_points": operating_points,
        "metrics": metrics,
    }
    metadata.update(dict(block.metadata))
    save_json(metadata_path, metadata)
    return metadata


def _metadata_dataset(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    dataset = metadata.get("dataset")
    return dataset if isinstance(dataset, Mapping) else {}


def baseline_checkpoint_identity(block: LocalGraphBaselineLogitBlock) -> dict[str, Any]:
    return {
        "checkpoint": block.metadata.get("checkpoint"),
        "checkpoint_variant": block.metadata.get("checkpoint_variant"),
        "checkpoint_epoch": block.metadata.get("checkpoint_epoch"),
        "run_report": block.metadata.get("run_report"),
    }


def verify_baseline_logit_block_alignment(
    block: LocalGraphBaselineLogitBlock,
    dataset_metadata: Mapping[str, Any],
    *,
    split: str | None = None,
    dataset_length: int | None = None,
    require_hashes: bool = True,
) -> dict[str, Any]:
    """Verify that a cached baseline-logit block belongs to the active HLT dataset."""

    problems: list[str] = []
    if split is not None and str(block.split) != str(split):
        problems.append(f"split mismatch: block={block.split} active={split}")
    if dataset_length is not None and int(block.labels.shape[0]) < int(dataset_length):
        problems.append(
            f"cache has {int(block.labels.shape[0])} rows but active dataset needs {int(dataset_length)}"
        )
    active_split = dataset_metadata.get("split")
    if active_split is not None and str(active_split) != str(block.split):
        problems.append(f"dataset metadata split mismatch: block={block.split} active={active_split}")
    cached_dataset = _metadata_dataset(block.metadata)
    for key in ("hlt_content_hash", "jet_identity_hash"):
        cached = cached_dataset.get(key)
        active = dataset_metadata.get(key)
        if cached is None or active is None:
            if bool(require_hashes):
                problems.append(f"missing {key}: cached={cached!r} active={active!r}")
        elif str(cached) != str(active):
            problems.append(f"{key} mismatch: cached={cached!r} active={active!r}")
    cached_params = cached_dataset.get("hlt_params")
    active_params = dataset_metadata.get("hlt_params")
    if cached_params is not None and active_params is not None and _jsonable(cached_params) != _jsonable(active_params):
        problems.append("hlt_params mismatch between baseline cache and active dataset")
    active_n = dataset_metadata.get("n_jets")
    cached_n = cached_dataset.get("n_jets")
    if active_n is not None and cached_n is not None and int(cached_n) < int(active_n):
        problems.append(f"cached dataset n_jets={cached_n} is smaller than active n_jets={active_n}")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "ok": True,
        "split": block.split,
        "dataset_length": None if dataset_length is None else int(dataset_length),
        "cache_rows": int(block.labels.shape[0]),
        "checkpoint_identity": baseline_checkpoint_identity(block),
        "condition_reference": dict(block.condition_reference(require=False) or {}),
    }


def verify_baseline_logit_cache_family(
    blocks: Sequence[LocalGraphBaselineLogitBlock],
    *,
    require_condition_reference: bool = True,
    required_condition_source_split: str | None = LOCAL_GRAPH_BASELINE_CONDITION_REFERENCE_SPLIT,
    expected_checkpoint_variant: str | None = LOCAL_GRAPH_BASELINE_CHECKPOINT_VARIANT,
) -> dict[str, Any]:
    """Verify all split caches came from the same checkpoint and conditioning reference."""

    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("at least one baseline logit block is required")
    identities = [baseline_checkpoint_identity(block) for block in blocks]
    first_identity = identities[0]
    problems: list[str] = []
    if expected_checkpoint_variant is not None:
        for block, identity in zip(blocks, identities):
            if str(identity.get("checkpoint_variant")) != str(expected_checkpoint_variant):
                problems.append(
                    f"{block.split}: checkpoint_variant must be {expected_checkpoint_variant!r}, "
                    f"got {identity.get('checkpoint_variant')!r}"
                )
    for block, identity in zip(blocks[1:], identities[1:]):
        if identity != first_identity:
            problems.append(f"{block.split}: checkpoint identity differs from {blocks[0].split}")
    references = []
    for block in blocks:
        reference = block.condition_reference(require=bool(require_condition_reference))
        references.append(dict(reference or {}))
    first_reference = references[0]
    if required_condition_source_split is not None:
        for block, reference in zip(blocks, references):
            if str(reference.get("source_split")) != str(required_condition_source_split):
                problems.append(
                    f"{block.split}: condition_reference source_split must be "
                    f"{required_condition_source_split!r}, got {reference.get('source_split')!r}"
                )
    for block, reference in zip(blocks[1:], references[1:]):
        comparable = {
            key: reference.get(key)
            for key in ("source_split", "tau50", "tau30", "near_tau50_scale")
        }
        first_comparable = {
            key: first_reference.get(key)
            for key in ("source_split", "tau50", "tau30", "near_tau50_scale")
        }
        if comparable != first_comparable:
            problems.append(f"{block.split}: condition_reference differs from {blocks[0].split}")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "ok": True,
        "splits": [block.split for block in blocks],
        "checkpoint_identity": first_identity,
        "condition_reference": first_reference,
    }


def load_baseline_logit_block(
    output_dir: str | Path,
    split: str,
    *,
    require_metadata: bool = False,
) -> LocalGraphBaselineLogitBlock:
    npz_path, metadata_path = baseline_logit_cache_paths(output_dir, split)
    if not npz_path.exists():
        raise FileNotFoundError(f"missing baseline logit cache: {npz_path}")
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    elif bool(require_metadata):
        raise FileNotFoundError(f"missing baseline logit metadata: {metadata_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        condition_features_array = data["condition_features"] if "condition_features" in data.files else None
        return LocalGraphBaselineLogitBlock(
            split=str(split),
            logits=data["logits"],
            labels=data["labels"],
            indices=data["indices"],
            metadata=metadata,
            condition_features_array=condition_features_array,
        )


@dataclass
class LocalGraphBaselineLogitCacheConfig:
    output_dir: str
    hlt_cache_dir: str
    checkpoint_path: str
    splits: tuple[str, ...] = LOCAL_GRAPH_RESIDUAL_DEFAULT_SPLITS
    metric_splits: tuple[str, ...] = LOCAL_GRAPH_BASELINE_LOGIT_CACHE_DEFAULT_METRIC_SPLITS
    max_jets_by_split: Mapping[str, int | None] = field(default_factory=dict)
    batch_size: int = 256
    num_workers: int = 0
    device: str = "auto"
    seed: int = 7717
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    expected_hlt_degradation_strength: float = 0.6
    expected_checkpoint_variant: str = LOCAL_GRAPH_BASELINE_CHECKPOINT_VARIANT
    overwrite: bool = False
    label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
    label_filter: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "splits", tuple(str(split) for split in self.splits))
        object.__setattr__(self, "metric_splits", tuple(str(split) for split in self.metric_splits))
        if not self.splits:
            raise ValueError("at least one split is required")
        unknown_metric_splits = sorted(set(self.metric_splits) - set(self.splits))
        if unknown_metric_splits:
            raise ValueError(f"metric_splits must be a subset of splits, got {unknown_metric_splits}")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "num_workers", int(self.num_workers))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "expected_hlt_degradation_strength", float(self.expected_hlt_degradation_strength))
        object.__setattr__(self, "expected_checkpoint_variant", str(self.expected_checkpoint_variant))
        object.__setattr__(self, "label_names", tuple(str(name) for name in self.label_names))
        object.__setattr__(self, "label_filter", tuple(int(label) for label in self.label_filter))
        if len(self.label_names) != 2 or tuple(self.label_names) != LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES:
            raise ValueError("baseline cache is currently fixed to QCD/Hgg label names")
        if tuple(self.label_filter) != (0, 1):
            raise ValueError("baseline cache expects binary-cache labels QCD=0, Hgg=1")
        clean_max: dict[str, int | None] = {}
        for split in self.splits:
            clean_max[str(split)] = _optional_positive_int(
                self.max_jets_by_split.get(str(split)) if isinstance(self.max_jets_by_split, Mapping) else None,
                field_name=f"max_jets_by_split[{split}]",
            )
        object.__setattr__(self, "max_jets_by_split", clean_max)


def _load_baseline_dataset(
    config: LocalGraphBaselineLogitCacheConfig,
    split: str,
) -> SubtokenHLTJetDataset:
    view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    view.metadata["hlt_protocol_audit"] = _verify_hlt_cache_protocol(
        view.metadata,
        split=split,
        expected_strength=float(config.expected_hlt_degradation_strength),
        required=bool(config.verify_hlt_params),
    )
    dataset = SubtokenHLTJetDataset(
        view,
        label_filter=tuple(config.label_filter),
        label_names=tuple(config.label_names),
        max_jets=config.max_jets_by_split.get(str(split)),
    )
    dataset.metadata["hlt_protocol_audit"] = view.metadata["hlt_protocol_audit"]
    return dataset


def predict_baseline_logits_for_split(
    model,
    config: LocalGraphBaselineLogitCacheConfig,
    split: str,
    *,
    device,
    checkpoint_payload: Mapping[str, Any],
) -> LocalGraphBaselineLogitBlock:
    torch = require_torch()
    dataset = _load_baseline_dataset(config, split)
    loader = make_subtoken_hlt_loader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    indices_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = model(tokens, mask)
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
            indices_rows.append(batch["indices"].detach().cpu().numpy().astype(np.int64))
    logits_np = np.concatenate(logits_rows, axis=0) if logits_rows else np.zeros((0, 2), dtype=np.float32)
    labels_np = np.concatenate(labels_rows, axis=0) if labels_rows else np.zeros((0,), dtype=np.int64)
    indices_np = np.concatenate(indices_rows, axis=0) if indices_rows else np.zeros((0,), dtype=np.int64)
    return LocalGraphBaselineLogitBlock(
        split=str(split),
        logits=logits_np,
        labels=labels_np,
        indices=indices_np,
        metadata={
            "checkpoint": str(config.checkpoint_path),
            "checkpoint_variant": checkpoint_payload.get("variant"),
            "checkpoint_epoch": checkpoint_payload.get("epoch"),
            "run_report": str(Path(config.checkpoint_path).parent / "run_report.json"),
            "dataset": dict(dataset.metadata),
        },
    )


def cache_local_graph_baseline_logits(config: LocalGraphBaselineLogitCacheConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    model, payload = load_local_graph_tagger_checkpoint(config.checkpoint_path, device=device)
    checkpoint_variant = payload.get("variant")
    if str(checkpoint_variant) != str(config.expected_checkpoint_variant):
        raise ValueError(
            "Baseline-logit cache must be built from the frozen HLT ParT baseline checkpoint. "
            f"Expected variant {config.expected_checkpoint_variant!r}, got {checkpoint_variant!r}."
        )
    blocks: dict[str, LocalGraphBaselineLogitBlock] = {}
    for split in config.splits:
        blocks[str(split)] = predict_baseline_logits_for_split(
            model,
            config,
            split,
            device=device,
            checkpoint_payload=payload,
        )
    if LOCAL_GRAPH_BASELINE_CONDITION_REFERENCE_SPLIT not in blocks:
        raise ValueError(
            f"baseline-logit cache must include {LOCAL_GRAPH_BASELINE_CONDITION_REFERENCE_SPLIT!r} "
            "so the label-dependent conditioning reference never comes from held-out splits"
        )
    reference_split = LOCAL_GRAPH_BASELINE_CONDITION_REFERENCE_SPLIT
    condition_reference = baseline_condition_reference_from_block(blocks[reference_split], source_split=reference_split)
    split_reports: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []
    metric_split_set = set(config.metric_splits)
    for split in config.splits:
        block = blocks[str(split)]
        report = save_baseline_logit_block(
            block,
            output_dir,
            condition_reference=condition_reference,
            compute_metrics=str(split) in metric_split_set,
            overwrite=bool(config.overwrite),
        )
        split_reports[str(split)] = report
        manifest_rows.append(
            {
                "split": str(split),
                "n_jets": int(block.labels.shape[0]),
                "npz_path": report.get("npz_path"),
                "metadata_path": report.get("metadata_path"),
                "metrics_computed": bool(report.get("metrics_computed")),
                "fpr_at_signal_eff_0p50": (
                    ((report.get("metrics") or {}).get("binary_metrics") or {}).get("fpr_at_signal_eff_0p50")
                ),
                "threshold_margin_0p50": (
                    (((report.get("operating_points") or {}).get("margin") or {}).get("signal_eff_0p50") or {}).get(
                        "threshold"
                    )
                ),
            }
        )
    manifest = {
        "step": LOCAL_GRAPH_BASELINE_LOGIT_CACHE_STEP,
        "contract": LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CONTRACT,
        "config": asdict(config),
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_variant": payload.get("variant"),
        "expected_checkpoint_variant": str(config.expected_checkpoint_variant),
        "checkpoint_epoch": payload.get("epoch"),
        "device": str(device),
        "splits": split_reports,
        "manifest_rows": manifest_rows,
        "condition_reference": condition_reference,
        "outputs": {
            "manifest": str(output_dir / "baseline_logit_manifest.json"),
            "run_report": str(output_dir / "run_report.json"),
        },
    }
    save_json(output_dir / "baseline_logit_manifest.json", manifest)
    save_json(output_dir / "run_report.json", manifest)
    return manifest


__all__ = [
    "LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES",
    "LOCAL_GRAPH_BASELINE_CONDITION_REFERENCE_SPLIT",
    "LOCAL_GRAPH_BASELINE_CHECKPOINT_VARIANT",
    "LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CONTRACT",
    "LOCAL_GRAPH_BASELINE_LOGIT_CACHE_DEFAULT_METRIC_SPLITS",
    "LOCAL_GRAPH_BASELINE_LOGIT_CACHE_STEP",
    "LOCAL_GRAPH_RESIDUAL_DEFAULT_SPLITS",
    "LocalGraphBaselineLogitBlock",
    "LocalGraphBaselineLogitCacheConfig",
    "baseline_checkpoint_identity",
    "baseline_condition_features",
    "baseline_condition_reference_from_block",
    "baseline_logit_cache_paths",
    "cache_local_graph_baseline_logits",
    "load_baseline_logit_block",
    "operating_point_from_scores",
    "predict_baseline_logits_for_split",
    "save_baseline_logit_block",
    "verify_baseline_logit_block_alignment",
    "verify_baseline_logit_cache_family",
]

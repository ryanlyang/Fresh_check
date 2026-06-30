"""Baseline-logit and penultimate-embedding cache for residual expert V2.

V2 differs from the V1 residual cache in one important way: the cache is built
from the exact frozen 2-class HLT ParT baseline and stores the true
penultimate ParT embedding captured by :mod:`residual_v2_anchor`.  This module
keeps that output under a separate contract so V1 widened-head caches cannot be
mistaken for V2 inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device
from jetclass_fresh.hlt_cache import load_cached_hlt_view

from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader

from .fusion import binary_margin_from_logits, binary_metrics_from_signal_scores, sigmoid_np
from .protocol import LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
from .residual_cache import (
    LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES,
    baseline_condition_features,
    operating_point_from_scores,
    save_json,
)
from .residual_v2_anchor import HLTPartEmbeddingAnchor, load_hlt_part_embedding_anchor
from .residual_v2_protocol import (
    LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT,
    LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_CACHE_SPLITS,
    LOCAL_GRAPH_RESIDUAL_V2_CACHE_STEP,
    LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_INDEX,
    LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_NAME,
    LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
    default_local_graph_residual_v2_protocol,
    local_graph_residual_v2_protocol_manifest,
)
from .train import _verify_hlt_cache_protocol


LOCAL_GRAPH_RESIDUAL_V2_CACHE_DEFAULT_METRIC_SPLITS = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
)
LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT = "model_train"


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
    if hasattr(value, "detach"):
        try:
            if int(value.numel()) == 1:
                return float(value.detach().cpu().item())
        except Exception:
            return str(value)
    return value


def sha256_file(path: str | Path) -> str:
    """Return a SHA256 digest for a checkpoint file."""

    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def residual_v2_embedding_cache_paths(output_dir: str | Path, split: str) -> tuple[Path, Path]:
    root = Path(output_dir)
    return root / f"{split}_baseline_embedding_cache.npz", root / f"{split}_baseline_embedding_cache_metadata.json"


def _condition_reference_from_arrays(
    *,
    split: str,
    labels: np.ndarray,
    margin: np.ndarray,
) -> dict[str, Any]:
    op50 = operating_point_from_scores(labels, margin, 0.50)
    op30 = operating_point_from_scores(labels, margin, 0.30)
    _features, metadata = baseline_condition_features(
        margin,
        tau50=float(op50["threshold"]),
        tau30=float(op30["threshold"]),
    )
    return {
        "source_split": str(split),
        "feature_names": list(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES),
        "tau50": float(metadata["tau50"]),
        "tau30": float(metadata["tau30"]),
        "near_tau50_scale": float(metadata["near_tau50_scale"]),
        "n_jets": int(np.asarray(labels).shape[0]),
        "label_dependent": True,
        "rule": "Computed once from model_train and reused for every V2 cached split.",
    }


def _condition_features_from_reference(margin: np.ndarray, reference: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    features, metadata = baseline_condition_features(
        margin,
        tau50=float(reference["tau50"]),
        tau30=float(reference["tau30"]),
        near_scale=float(reference["near_tau50_scale"]),
    )
    metadata["reference"] = dict(reference)
    metadata["condition_source"] = "model_train_shared_reference"
    return features.astype(np.float32, copy=False), metadata


@dataclass
class LocalGraphResidualV2BaselineEmbeddingBlock:
    """One split of exact HLT ParT baseline logits plus true ParT embeddings."""

    split: str
    logits: np.ndarray
    embedding: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    condition_features_array: np.ndarray | None = None
    _index_to_position: dict[int, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.split = str(self.split)
        self.logits = np.asarray(self.logits, dtype=np.float32)
        self.embedding = np.asarray(self.embedding, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        self.indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        if self.logits.ndim != 2 or self.logits.shape[1] != 2:
            raise ValueError(f"{self.split}: logits must have shape [N, 2], got {self.logits.shape}")
        if self.embedding.ndim != 2:
            raise ValueError(f"{self.split}: embedding must have shape [N, D], got {self.embedding.shape}")
        n_rows = int(self.logits.shape[0])
        if int(self.embedding.shape[0]) != n_rows:
            raise ValueError(f"{self.split}: embedding/logit length mismatch")
        if int(self.labels.shape[0]) != n_rows:
            raise ValueError(f"{self.split}: label/logit length mismatch")
        if int(self.indices.shape[0]) != n_rows:
            raise ValueError(f"{self.split}: index/logit length mismatch")
        if n_rows and np.unique(self.indices).shape[0] != n_rows:
            raise ValueError(f"{self.split}: indices must be unique")
        self._index_to_position = {int(index): int(pos) for pos, index in enumerate(self.indices.tolist())}
        if not np.isfinite(self.logits).all():
            raise FloatingPointError(f"{self.split}: logits contain non-finite values")
        if not np.isfinite(self.embedding).all():
            raise FloatingPointError(f"{self.split}: embeddings contain non-finite values")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError(f"{self.split}: labels must be encoded as 0/1")
        if self.condition_features_array is not None:
            features = np.asarray(self.condition_features_array, dtype=np.float32)
            expected = (n_rows, len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES))
            if features.shape != expected:
                raise ValueError(f"{self.split}: condition_features must have shape {expected}, got {features.shape}")
            if not np.isfinite(features).all():
                raise FloatingPointError(f"{self.split}: condition_features contain non-finite values")
            self.condition_features_array = features

    @property
    def margin(self) -> np.ndarray:
        return binary_margin_from_logits(self.logits).astype(np.float64)

    @property
    def prob_hgg(self) -> np.ndarray:
        return sigmoid_np(self.margin)

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding.shape[1])

    def metrics(self) -> dict[str, Any]:
        return binary_metrics_from_signal_scores(self.margin, self.labels)

    def operating_points(self) -> dict[str, Any]:
        return {
            "margin": {
                "signal_eff_0p30": operating_point_from_scores(self.labels, self.margin, 0.30),
                "signal_eff_0p50": operating_point_from_scores(self.labels, self.margin, 0.50),
            }
        }

    def condition_reference(self, *, require: bool = False) -> Mapping[str, Any] | None:
        reference = self.metadata.get("condition_reference")
        if not isinstance(reference, Mapping):
            condition = self.metadata.get("condition_features")
            if isinstance(condition, Mapping) and isinstance(condition.get("reference"), Mapping):
                reference = condition["reference"]
        if reference is None:
            if bool(require):
                raise ValueError(f"{self.split}: V2 cache is missing condition_reference")
            return None
        if str(reference.get("source_split")) != LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT:
            raise ValueError(
                f"{self.split}: V2 condition_reference source_split must be "
                f"{LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT!r}, got {reference.get('source_split')!r}"
            )
        return reference

    def positions_for_indices(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        positions: list[int] = []
        missing: list[int] = []
        for index in indices.tolist():
            position = self._index_to_position.get(int(index))
            if position is None:
                missing.append(int(index))
            else:
                positions.append(position)
        if missing:
            preview = ", ".join(str(item) for item in missing[:5])
            raise IndexError(f"{self.split}: V2 embedding cache is missing dataset indices: {preview}")
        return np.asarray(positions, dtype=np.int64)

    def condition_features(self, *, require_reference: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
        reference = self.condition_reference(require=bool(require_reference))
        if self.condition_features_array is not None:
            if reference is not None:
                return self.condition_features_array, {
                    "feature_names": list(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES),
                    "reference": dict(reference),
                    "condition_source": "model_train_shared_reference",
                }
            return self.condition_features_array, {
                "feature_names": list(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES),
                "condition_source": "cache_array_no_reference",
            }
        if reference is None:
            raise ValueError(f"{self.split}: V2 cache is missing condition_features and condition_reference")
        return _condition_features_from_reference(self.margin, reference)

    def arrays_for_indices(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        positions = self.positions_for_indices(indices)
        condition_features, _condition_metadata = self.condition_features(require_reference=True)
        return {
            "positions": positions,
            "logits": self.logits[positions],
            "margin": self.margin[positions].astype(np.float32),
            "prob_hgg": self.prob_hgg[positions].astype(np.float32),
            "embedding": self.embedding[positions],
            "labels": self.labels[positions],
            "indices": self.indices[positions],
            "condition_features": condition_features[positions],
        }

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "n_jets": int(self.labels.shape[0]),
            "logits_shape": list(self.logits.shape),
            "embedding_shape": list(self.embedding.shape),
            "embedding_dim": int(self.embedding_dim),
            "index_min": int(np.min(self.indices)) if self.indices.size else None,
            "index_max": int(np.max(self.indices)) if self.indices.size else None,
        }


def _checkpoint_identity(anchor: HLTPartEmbeddingAnchor, checkpoint_path: str | Path | None) -> dict[str, Any]:
    payload = dict(getattr(anchor, "payload", {}) or {})
    path = None if checkpoint_path is None else str(Path(checkpoint_path))
    file_sha = None
    if path is not None and Path(path).exists():
        file_sha = sha256_file(path)
    identity = {
        "checkpoint_path": path,
        "checkpoint_sha256": file_sha,
        "checkpoint_variant": payload.get("variant") or LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT,
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_output_contract": payload.get("output_contract"),
        "final_head_name": getattr(anchor, "final_head_name", None),
        "embedding_source": anchor.config.embedding_source,
        "required_embedding_role": anchor.config.required_embedding_role,
    }
    identity["checkpoint_identity_hash"] = _stable_json_hash(identity)
    return identity


def save_residual_v2_embedding_block(
    block: LocalGraphResidualV2BaselineEmbeddingBlock,
    output_dir: str | Path,
    *,
    condition_reference: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    metric_splits: Sequence[str],
    overwrite: bool = False,
) -> dict[str, Any]:
    npz_path, metadata_path = residual_v2_embedding_cache_paths(output_dir, block.split)
    if not bool(overwrite) and (npz_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"V2 embedding cache already exists for split {block.split}: {npz_path}")
    if str(condition_reference.get("source_split")) != LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT:
        raise ValueError("V2 condition_reference must come from model_train")
    features, feature_metadata = _condition_features_from_reference(block.margin, condition_reference)
    metrics_computed = str(block.split) in set(str(split) for split in metric_splits)
    metrics = block.metrics() if metrics_computed else None
    operating_points = block.operating_points() if metrics_computed else None
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        logits=block.logits.astype(np.float32, copy=False),
        margin=block.margin.astype(np.float32),
        prob_hgg=block.prob_hgg.astype(np.float32),
        embedding=block.embedding.astype(np.float32, copy=False),
        labels=block.labels.astype(np.int64, copy=False),
        indices=block.indices.astype(np.int64, copy=False),
        condition_features=features.astype(np.float32, copy=False),
        condition_feature_names=np.asarray(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES),
    )
    metadata = {
        "step": LOCAL_GRAPH_RESIDUAL_V2_CACHE_STEP,
        "contract": LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
        **block.to_manifest_row(),
        "npz_path": str(npz_path),
        "metadata_path": str(metadata_path),
        "label_names": list(LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES),
        "positive_class_name": LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_NAME,
        "positive_class_index": int(LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_INDEX),
        "checkpoint_path": checkpoint_identity.get("checkpoint_path"),
        "checkpoint_identity": dict(checkpoint_identity),
        "checkpoint_identity_hash": checkpoint_identity.get("checkpoint_identity_hash"),
        "model_config": block.metadata.get("model_config"),
        "embedding_dim": int(block.embedding_dim),
        "embedding_source": block.metadata.get("embedding_source"),
        "required_embedding_role": LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
        "embedding_reproduces_logits_check": bool(block.metadata.get("embedding_reproduces_logits_check", True)),
        "hlt_degradation_strength": block.metadata.get("hlt_degradation_strength"),
        "hlt_cache_dir": block.metadata.get("hlt_cache_dir"),
        "hlt_content_hash": (block.metadata.get("dataset") or {}).get("hlt_content_hash"),
        "jet_identity_hash": (block.metadata.get("dataset") or {}).get("jet_identity_hash"),
        "split_manifest_hash": (block.metadata.get("dataset") or {}).get("split_manifest_hash"),
        "condition_features": feature_metadata,
        "condition_reference": dict(condition_reference),
        "metric_splits": list(str(split) for split in metric_splits),
        "metrics_computed": bool(metrics_computed),
        "operating_points": operating_points,
        "metrics": metrics,
    }
    metadata.update(dict(block.metadata))
    metadata["condition_reference"] = dict(condition_reference)
    metadata["condition_features"] = feature_metadata
    save_json(metadata_path, metadata)
    return metadata


def load_residual_v2_embedding_block(
    output_dir: str | Path,
    split: str,
    *,
    require_metadata: bool = True,
) -> LocalGraphResidualV2BaselineEmbeddingBlock:
    npz_path, metadata_path = residual_v2_embedding_cache_paths(output_dir, split)
    if not npz_path.exists():
        raise FileNotFoundError(f"missing V2 baseline embedding cache: {npz_path}")
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    elif bool(require_metadata):
        raise FileNotFoundError(f"missing V2 baseline embedding metadata: {metadata_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        return LocalGraphResidualV2BaselineEmbeddingBlock(
            split=str(split),
            logits=data["logits"],
            embedding=data["embedding"],
            labels=data["labels"],
            indices=data["indices"],
            metadata=metadata,
            condition_features_array=data["condition_features"] if "condition_features" in data.files else None,
        )


def residual_v2_checkpoint_identity(block: LocalGraphResidualV2BaselineEmbeddingBlock) -> dict[str, Any]:
    identity = block.metadata.get("checkpoint_identity")
    if isinstance(identity, Mapping):
        return dict(identity)
    return {
        "checkpoint_path": block.metadata.get("checkpoint_path") or block.metadata.get("checkpoint"),
        "checkpoint_identity_hash": block.metadata.get("checkpoint_identity_hash"),
        "checkpoint_variant": block.metadata.get("checkpoint_variant"),
        "checkpoint_epoch": block.metadata.get("checkpoint_epoch"),
        "embedding_source": block.metadata.get("embedding_source"),
        "required_embedding_role": block.metadata.get("required_embedding_role"),
    }


def _metadata_value(block: LocalGraphResidualV2BaselineEmbeddingBlock, key: str) -> Any:
    value = block.metadata.get(key)
    if value is not None:
        return value
    dataset = block.metadata.get("dataset")
    if isinstance(dataset, Mapping):
        return dataset.get(key)
    return None


def _compare_optional_metadata(
    *,
    problems: list[str],
    name: str,
    cached: Any,
    active: Any,
    require: bool,
) -> None:
    if cached is None or active is None:
        if bool(require):
            problems.append(f"missing {name}: cached={cached!r} active={active!r}")
        return
    if str(cached) != str(active):
        problems.append(f"{name} mismatch: cached={cached!r} active={active!r}")


def _condition_reference_signature(reference: Mapping[str, Any] | None) -> dict[str, Any]:
    if reference is None:
        return {}
    return {
        "source_split": reference.get("source_split"),
        "tau50": None if reference.get("tau50") is None else float(reference["tau50"]),
        "tau30": None if reference.get("tau30") is None else float(reference["tau30"]),
        "near_tau50_scale": (
            None if reference.get("near_tau50_scale") is None else float(reference["near_tau50_scale"])
        ),
        "feature_names": list(reference.get("feature_names") or []),
    }


def verify_residual_v2_embedding_block_alignment(
    block: LocalGraphResidualV2BaselineEmbeddingBlock,
    dataset_metadata: Mapping[str, Any],
    *,
    split: str | None = None,
    dataset_length: int | None = None,
    expected_indices: np.ndarray | None = None,
    expected_labels: np.ndarray | None = None,
    expected_checkpoint_identity: Mapping[str, Any] | None = None,
    expected_condition_reference: Mapping[str, Any] | None = None,
    expected_embedding_dim: int | None = None,
    require_hashes: bool = True,
) -> dict[str, Any]:
    """Verify one V2 embedding cache block belongs to the active HLT dataset."""

    problems: list[str] = []
    if str(block.metadata.get("contract")) != LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT:
        problems.append(
            f"contract mismatch: expected {LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT!r}, "
            f"got {block.metadata.get('contract')!r}"
        )
    if split is not None and str(block.split) != str(split):
        problems.append(f"split mismatch: block={block.split} active={split}")
    active_split = dataset_metadata.get("split")
    if active_split is not None and str(active_split) != str(block.split):
        problems.append(f"dataset metadata split mismatch: block={block.split} active={active_split}")
    if dataset_length is not None and int(block.labels.shape[0]) < int(dataset_length):
        problems.append(
            f"cache has {int(block.labels.shape[0])} rows but active dataset needs {int(dataset_length)}"
        )
    label_names = tuple(str(item) for item in block.metadata.get("label_names") or ())
    if label_names != tuple(LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES):
        problems.append(f"label_names mismatch: {label_names!r}")
    if str(block.metadata.get("positive_class_name")) != LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_NAME:
        problems.append(f"positive_class_name mismatch: {block.metadata.get('positive_class_name')!r}")
    if int(block.metadata.get("positive_class_index", -1)) != int(LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_INDEX):
        problems.append(f"positive_class_index mismatch: {block.metadata.get('positive_class_index')!r}")
    if str(block.metadata.get("required_embedding_role")) != LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE:
        problems.append(f"required_embedding_role mismatch: {block.metadata.get('required_embedding_role')!r}")
    cached_embedding_dim = int(block.metadata.get("embedding_dim", block.embedding_dim))
    if cached_embedding_dim != int(block.embedding_dim):
        problems.append(
            f"metadata embedding_dim={cached_embedding_dim} does not match array dim={int(block.embedding_dim)}"
        )
    if expected_embedding_dim is not None and int(block.embedding_dim) != int(expected_embedding_dim):
        problems.append(f"embedding_dim mismatch: cached={int(block.embedding_dim)} expected={int(expected_embedding_dim)}")

    _compare_optional_metadata(
        problems=problems,
        name="hlt_content_hash",
        cached=_metadata_value(block, "hlt_content_hash"),
        active=dataset_metadata.get("hlt_content_hash"),
        require=bool(require_hashes),
    )
    _compare_optional_metadata(
        problems=problems,
        name="jet_identity_hash",
        cached=_metadata_value(block, "jet_identity_hash"),
        active=dataset_metadata.get("jet_identity_hash"),
        require=bool(require_hashes),
    )
    active_manifest_hash = (
        dataset_metadata.get("split_manifest_hash")
        or dataset_metadata.get("source_manifest_hash")
        or dataset_metadata.get("manifest_hash")
    )
    _compare_optional_metadata(
        problems=problems,
        name="split_manifest_hash",
        cached=_metadata_value(block, "split_manifest_hash"),
        active=active_manifest_hash,
        require=False,
    )
    if expected_indices is not None:
        expected_indices = np.asarray(expected_indices, dtype=np.int64).reshape(-1)
        if expected_indices.shape[0] > block.indices.shape[0]:
            problems.append(
                f"expected_indices has {expected_indices.shape[0]} rows but cache has {block.indices.shape[0]}"
            )
        elif not np.array_equal(block.indices[: expected_indices.shape[0]], expected_indices):
            problems.append("split indices/order mismatch")
    if expected_labels is not None:
        expected_labels = np.asarray(expected_labels, dtype=np.int64).reshape(-1)
        if expected_labels.shape[0] > block.labels.shape[0]:
            problems.append(f"expected_labels has {expected_labels.shape[0]} rows but cache has {block.labels.shape[0]}")
        elif not np.array_equal(block.labels[: expected_labels.shape[0]], expected_labels):
            problems.append("labels mismatch")
    identity = residual_v2_checkpoint_identity(block)
    if expected_checkpoint_identity is not None:
        expected_hash = expected_checkpoint_identity.get("checkpoint_identity_hash")
        cached_hash = identity.get("checkpoint_identity_hash")
        if expected_hash is not None and cached_hash is not None:
            if str(expected_hash) != str(cached_hash):
                problems.append(f"checkpoint_identity_hash mismatch: cached={cached_hash!r} expected={expected_hash!r}")
        elif _jsonable(identity) != _jsonable(dict(expected_checkpoint_identity)):
            problems.append("checkpoint identity mismatch")
    reference = block.condition_reference(require=True)
    if expected_condition_reference is not None:
        if _condition_reference_signature(reference) != _condition_reference_signature(expected_condition_reference):
            problems.append("condition_reference mismatch")
    if block.condition_features_array is None:
        problems.append("condition_features array is missing from V2 cache")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "ok": True,
        "split": block.split,
        "dataset_length": None if dataset_length is None else int(dataset_length),
        "cache_rows": int(block.labels.shape[0]),
        "embedding_dim": int(block.embedding_dim),
        "checkpoint_identity": identity,
        "condition_reference": dict(reference or {}),
        "hlt_content_hash": _metadata_value(block, "hlt_content_hash"),
        "jet_identity_hash": _metadata_value(block, "jet_identity_hash"),
        "split_manifest_hash": _metadata_value(block, "split_manifest_hash"),
    }


def verify_residual_v2_embedding_cache_family(
    blocks: Sequence[LocalGraphResidualV2BaselineEmbeddingBlock],
    *,
    require_condition_reference: bool = True,
    expected_checkpoint_identity: Mapping[str, Any] | None = None,
    expected_checkpoint_variant: str = LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT,
) -> dict[str, Any]:
    """Verify a group of V2 split caches share one baseline and condition reference."""

    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("at least one V2 embedding cache block is required")
    problems: list[str] = []
    splits = [block.split for block in blocks]
    if len(set(splits)) != len(splits):
        problems.append(f"duplicate split names in V2 cache family: {splits}")
    identities = [residual_v2_checkpoint_identity(block) for block in blocks]
    first_identity = identities[0]
    first_hash = first_identity.get("checkpoint_identity_hash")
    if expected_checkpoint_identity is not None:
        expected_hash = expected_checkpoint_identity.get("checkpoint_identity_hash")
        if expected_hash is not None and first_hash is not None and str(expected_hash) != str(first_hash):
            problems.append(f"checkpoint_identity_hash mismatch: cached={first_hash!r} expected={expected_hash!r}")
    for block, identity in zip(blocks, identities):
        if str(identity.get("checkpoint_variant")) != str(expected_checkpoint_variant):
            problems.append(
                f"{block.split}: checkpoint_variant must be {expected_checkpoint_variant!r}, "
                f"got {identity.get('checkpoint_variant')!r}"
            )
    for block, identity in zip(blocks[1:], identities[1:]):
        if first_hash is not None and identity.get("checkpoint_identity_hash") is not None:
            if str(identity["checkpoint_identity_hash"]) != str(first_hash):
                problems.append(f"{block.split}: checkpoint_identity_hash differs from {blocks[0].split}")
        elif _jsonable(identity) != _jsonable(first_identity):
            problems.append(f"{block.split}: checkpoint identity differs from {blocks[0].split}")
    embedding_dim = int(blocks[0].embedding_dim)
    for block in blocks:
        if int(block.embedding_dim) != embedding_dim:
            problems.append(f"{block.split}: embedding_dim differs from {blocks[0].split}")
        if str(block.metadata.get("contract")) != LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT:
            problems.append(f"{block.split}: unexpected V2 cache contract {block.metadata.get('contract')!r}")
        if tuple(str(item) for item in block.metadata.get("label_names") or ()) != tuple(LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES):
            problems.append(f"{block.split}: label_names mismatch")
    references = []
    for block in blocks:
        reference = block.condition_reference(require=bool(require_condition_reference))
        references.append(reference)
    first_reference_signature = _condition_reference_signature(references[0])
    for block, reference in zip(blocks, references):
        signature = _condition_reference_signature(reference)
        if str(signature.get("source_split")) != LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT:
            problems.append(
                f"{block.split}: condition_reference source_split must be "
                f"{LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT!r}, got {signature.get('source_split')!r}"
            )
        if signature != first_reference_signature:
            problems.append(f"{block.split}: condition_reference differs from {blocks[0].split}")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "ok": True,
        "splits": splits,
        "embedding_dim": embedding_dim,
        "checkpoint_identity_hash": first_identity.get("checkpoint_identity_hash"),
        "checkpoint_identity": first_identity,
        "condition_reference": dict(references[0] or {}),
    }


@dataclass
class LocalGraphResidualV2EmbeddingCacheConfig:
    """Configuration for writing the strict V2 frozen-baseline embedding cache."""

    output_dir: str
    hlt_cache_dir: str
    checkpoint_path: str | None = None
    splits: tuple[str, ...] = LOCAL_GRAPH_RESIDUAL_V2_CACHE_SPLITS
    metric_splits: tuple[str, ...] = LOCAL_GRAPH_RESIDUAL_V2_CACHE_DEFAULT_METRIC_SPLITS
    max_jets_by_split: Mapping[str, int | None] = field(default_factory=dict)
    batch_size: int = 256
    num_workers: int = 0
    device: str = "auto"
    seed: int = 7751
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    expected_hlt_degradation_strength: float = 0.6
    overwrite: bool = False
    label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
    label_filter: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        protocol = default_local_graph_residual_v2_protocol()
        object.__setattr__(self, "splits", tuple(str(split) for split in self.splits))
        object.__setattr__(self, "metric_splits", tuple(str(split) for split in self.metric_splits))
        if not self.splits:
            raise ValueError("at least one V2 cache split is required")
        unknown_metric_splits = sorted(set(self.metric_splits) - set(self.splits))
        if unknown_metric_splits:
            raise ValueError(f"metric_splits must be a subset of splits, got {unknown_metric_splits}")
        if LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT not in self.splits:
            raise ValueError("V2 embedding cache must include model_train for the shared condition reference")
        if "final_test" in set(self.metric_splits):
            raise ValueError("V2 embedding cache does not compute final_test metrics by default; use final report")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "num_workers", int(self.num_workers))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "expected_hlt_degradation_strength", float(self.expected_hlt_degradation_strength))
        if abs(float(self.expected_hlt_degradation_strength) - float(protocol.hlt_degradation_strength)) > 1.0e-12:
            raise ValueError("V2 embedding cache is fixed to HLT degradation strength 0.6")
        labels = tuple(str(name) for name in self.label_names)
        if labels != tuple(protocol.label_names):
            raise ValueError("V2 embedding cache is fixed to QCD/Hgg labels")
        object.__setattr__(self, "label_names", labels)
        label_filter = tuple(int(label) for label in self.label_filter)
        if label_filter != tuple(protocol.binary_label_filter):
            raise ValueError("V2 embedding cache expects binary-cache labels QCD=0, Hgg=1")
        object.__setattr__(self, "label_filter", label_filter)
        clean_max: dict[str, int | None] = {}
        for split in self.splits:
            value = self.max_jets_by_split.get(str(split)) if isinstance(self.max_jets_by_split, Mapping) else None
            if value is None:
                clean_max[str(split)] = None
            else:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"max_jets_by_split[{split}] must be positive when provided")
                clean_max[str(split)] = value
        object.__setattr__(self, "max_jets_by_split", clean_max)


def _load_residual_v2_dataset(
    config: LocalGraphResidualV2EmbeddingCacheConfig,
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
    dataset.metadata["split_manifest_hash"] = (
        view.metadata.get("source_manifest_hash")
        or view.metadata.get("manifest_hash")
        or view.metadata.get("split_manifest_hash")
    )
    dataset.metadata["source_manifest_hash"] = view.metadata.get("source_manifest_hash")
    dataset.metadata["hlt_cache_dir"] = str(config.hlt_cache_dir)
    return dataset


def predict_residual_v2_embeddings_for_split(
    anchor: HLTPartEmbeddingAnchor,
    config: LocalGraphResidualV2EmbeddingCacheConfig,
    split: str,
    *,
    device,
    checkpoint_identity: Mapping[str, Any],
) -> LocalGraphResidualV2BaselineEmbeddingBlock:
    torch = require_torch()
    dataset = _load_residual_v2_dataset(config, split)
    loader = make_subtoken_hlt_loader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    logits_rows: list[np.ndarray] = []
    embedding_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    indices_rows: list[np.ndarray] = []
    anchor.eval()
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            output = anchor.forward_outputs(tokens, mask)
            logits_rows.append(output.logits.detach().cpu().numpy().astype(np.float32))
            embedding_rows.append(output.embedding.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
            indices_rows.append(batch["indices"].detach().cpu().numpy().astype(np.int64))
    logits_np = np.concatenate(logits_rows, axis=0) if logits_rows else np.zeros((0, 2), dtype=np.float32)
    if embedding_rows:
        embedding_np = np.concatenate(embedding_rows, axis=0).astype(np.float32, copy=False)
    else:
        embedding_np = np.zeros((0, 0), dtype=np.float32)
    labels_np = np.concatenate(labels_rows, axis=0) if labels_rows else np.zeros((0,), dtype=np.int64)
    indices_np = np.concatenate(indices_rows, axis=0) if indices_rows else np.zeros((0,), dtype=np.int64)
    anchor_metadata = anchor.metadata()
    return LocalGraphResidualV2BaselineEmbeddingBlock(
        split=str(split),
        logits=logits_np,
        embedding=embedding_np,
        labels=labels_np,
        indices=indices_np,
        metadata={
            "checkpoint": checkpoint_identity.get("checkpoint_path"),
            "checkpoint_variant": checkpoint_identity.get("checkpoint_variant"),
            "checkpoint_epoch": checkpoint_identity.get("checkpoint_epoch"),
            "checkpoint_identity_hash": checkpoint_identity.get("checkpoint_identity_hash"),
            "run_report": (
                str(Path(str(checkpoint_identity["checkpoint_path"])).parent / "run_report.json")
                if checkpoint_identity.get("checkpoint_path")
                else None
            ),
            "dataset": dict(dataset.metadata),
            "model_config": anchor_metadata.get("config"),
            "anchor_metadata": anchor_metadata,
            "embedding_source": anchor_metadata.get("embedding_source"),
            "embedding_reproduces_logits_check": True,
            "hlt_degradation_strength": float(config.expected_hlt_degradation_strength),
            "hlt_cache_dir": str(config.hlt_cache_dir),
        },
    )


def cache_local_graph_residual_v2_baseline_embeddings(
    config: LocalGraphResidualV2EmbeddingCacheConfig,
    *,
    anchor: HLTPartEmbeddingAnchor | None = None,
) -> dict[str, Any]:
    """Write exact HLT ParT logits and true penultimate embeddings for V2."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    if anchor is None:
        if config.checkpoint_path is None:
            raise ValueError("checkpoint_path is required when anchor is not provided")
        anchor = load_hlt_part_embedding_anchor(config.checkpoint_path, device=str(device))
    anchor = anchor.to(device)
    anchor.eval()
    checkpoint_identity = _checkpoint_identity(anchor, config.checkpoint_path or anchor.checkpoint_path)
    if str(checkpoint_identity.get("checkpoint_variant")) != LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT:
        raise ValueError("V2 embedding cache must be built from hlt_part_baseline")
    blocks: dict[str, LocalGraphResidualV2BaselineEmbeddingBlock] = {}
    for split in config.splits:
        blocks[str(split)] = predict_residual_v2_embeddings_for_split(
            anchor,
            config,
            split,
            device=device,
            checkpoint_identity=checkpoint_identity,
        )
    reference_split = LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT
    reference_block = blocks[reference_split]
    condition_reference = _condition_reference_from_arrays(
        split=reference_split,
        labels=reference_block.labels,
        margin=reference_block.margin,
    )
    split_reports: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []
    for split in config.splits:
        block = blocks[str(split)]
        report = save_residual_v2_embedding_block(
            block,
            output_dir,
            condition_reference=condition_reference,
            checkpoint_identity=checkpoint_identity,
            metric_splits=tuple(config.metric_splits),
            overwrite=bool(config.overwrite),
        )
        split_reports[str(split)] = report
        manifest_rows.append(
            {
                "split": str(split),
                "n_jets": int(block.labels.shape[0]),
                "embedding_dim": int(block.embedding_dim),
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
        "step": LOCAL_GRAPH_RESIDUAL_V2_CACHE_STEP,
        "contract": LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
        "protocol": local_graph_residual_v2_protocol_manifest(),
        "config": asdict(config),
        "checkpoint_identity": checkpoint_identity,
        "device": str(device),
        "splits": split_reports,
        "manifest_rows": manifest_rows,
        "condition_reference": condition_reference,
        "embedding_source": anchor.config.embedding_source,
        "required_embedding_role": anchor.config.required_embedding_role,
        "outputs": {
            "manifest": str(output_dir / "baseline_embedding_manifest.json"),
            "run_report": str(output_dir / "run_report.json"),
        },
    }
    save_json(output_dir / "baseline_embedding_manifest.json", manifest)
    save_json(output_dir / "run_report.json", manifest)
    return manifest


__all__ = [
    "LOCAL_GRAPH_RESIDUAL_V2_CACHE_DEFAULT_METRIC_SPLITS",
    "LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT",
    "LocalGraphResidualV2BaselineEmbeddingBlock",
    "LocalGraphResidualV2EmbeddingCacheConfig",
    "cache_local_graph_residual_v2_baseline_embeddings",
    "load_residual_v2_embedding_block",
    "predict_residual_v2_embeddings_for_split",
    "residual_v2_embedding_cache_paths",
    "residual_v2_checkpoint_identity",
    "save_residual_v2_embedding_block",
    "sha256_file",
    "verify_residual_v2_embedding_block_alignment",
    "verify_residual_v2_embedding_cache_family",
]

"""Ensemble and fusion utilities for AV10 prediction caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays
from jetclass_fresh.hlt_baseline import save_json
from jetclass_fresh.jetclass_data import JetIdentity

from teacher_logit_reco.set_matching.five_view_train import (
    binary_classification_metrics_from_logits,
    classification_metrics_from_predictions,
)
from teacher_logit_reco.set_matching.train import source_metadata

from .config import (
    ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
    normalize_architecture_view_variant,
)
from .predictions import (
    ARCHITECTURE_VIEW_10CLASS_PREDICTION_SPLITS,
    architecture_view_10class_prediction_paths,
    load_architecture_view_10class_prediction_metadata,
)
from .train import architecture_view_binary_projection_metrics


ARCHITECTURE_VIEW_10CLASS_FUSION_STEP = "architecture_view_10class_step3_fusion"
ARCHITECTURE_VIEW_10CLASS_FUSION_CONTRACT = "architecture_view_10class_fusion_v1"
ARCHITECTURE_VIEW_10CLASS_FUSION_MODES = (
    "uniform_logit_mean",
    "uniform_probability_mean",
    "temperature_scaled_logit_mean",
    "scalar_weighted_logit_mean",
    "classwise_weighted_logit_mean",
    "ridge_logit_stacker",
    "binary_projection_weighted",
)
ARCHITECTURE_VIEW_10CLASS_FUSION_FIT_SPLIT = "stack_train"
ARCHITECTURE_VIEW_10CLASS_FUSION_SELECTION_SPLIT = "stack_val"
ARCHITECTURE_VIEW_10CLASS_FUSION_EVAL_SPLITS = ("stack_train", "stack_val", "final_test")
ARCHITECTURE_VIEW_10CLASS_BINARY_PROJECTION_PAIRS = (
    ("QCD", "Hgg"),
    ("QCD", "Hbb"),
    ("QCD", "Tbqq"),
    ("QCD", "Wqq"),
    ("QCD", "Zqq"),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.clip(exp.sum(axis=1, keepdims=True), 1.0e-300, None)).astype(np.float32)


def _log_probs(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), 1.0e-12, 1.0)).astype(np.float32)


def _cross_entropy_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return float("nan")
    picked = np.clip(probs[np.arange(labels.size), labels], 1.0e-12, 1.0)
    return float(-np.mean(np.log(picked)))


def _selection_score(metrics: Mapping[str, Any]) -> tuple[float, float]:
    accuracy = float(metrics.get("accuracy", 0.0) or 0.0)
    ce = float(metrics.get("cross_entropy", metrics.get("loss", float("inf"))) or float("inf"))
    return accuracy, -ce


def _labels_hash(labels: np.ndarray) -> str:
    import hashlib

    arr = np.asarray(labels, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _read_prediction_arrays(array_path: Path) -> dict[str, np.ndarray]:
    with np.load(array_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@dataclass(frozen=True)
class ArchitectureView10ClassPredictionBlock:
    """Loaded AV10 prediction block."""

    variant: str
    split: str
    logits: np.ndarray
    probs: np.ndarray
    labels: np.ndarray
    preds: np.ndarray
    jet_ids: tuple[JetIdentity, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ArchitectureView10ClassFusionConfig:
    """Configuration for Step 3 AV10 fusion."""

    prediction_dir: str
    output_dir: str
    model_names: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]] | None = None
    fusion_modes: tuple[str, ...] = ARCHITECTURE_VIEW_10CLASS_FUSION_MODES
    temperature_grid: tuple[float, ...] = (0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0)
    c_grid: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    scalar_weight_trials: int = 256
    binary_weight_trials: int = 256
    classwise_uniform_mix: float = 0.25
    control_seed: int = 7207
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if not str(self.prediction_dir):
            raise ValueError("prediction_dir is required")
        if not str(self.output_dir):
            raise ValueError("output_dir is required")
        models = tuple(normalize_architecture_view_variant(name) for name in self.model_names)
        if not models:
            raise ValueError("at least one model_name is required")
        if len(set(models)) != len(models):
            raise ValueError(f"duplicate model_names are not allowed: {models}")
        modes = tuple(str(mode) for mode in self.fusion_modes)
        unknown_modes = [mode for mode in modes if mode not in ARCHITECTURE_VIEW_10CLASS_FUSION_MODES]
        if unknown_modes:
            raise ValueError(f"unknown AV10 fusion modes: {unknown_modes}")
        groups = self.groups
        if groups is None:
            groups = default_architecture_view_10class_fusion_groups(models)
        normalized_groups: dict[str, tuple[str, ...]] = {}
        available = set(models)
        for group_name, members in groups.items():
            members_tuple = tuple(normalize_architecture_view_variant(member) for member in members)
            if len(members_tuple) < 1:
                raise ValueError(f"fusion group {group_name!r} has no members")
            missing = sorted(set(members_tuple) - available)
            if missing:
                raise ValueError(f"fusion group {group_name!r} references missing models: {missing}")
            normalized_groups[str(group_name)] = members_tuple
        if "final_test" in ARCHITECTURE_VIEW_10CLASS_FUSION_EVAL_SPLITS and not bool(self.confirm_final_test):
            raise ValueError("Refusing to evaluate final_test fusion without confirm_final_test=True")
        object.__setattr__(self, "model_names", models)
        object.__setattr__(self, "groups", normalized_groups)
        object.__setattr__(self, "fusion_modes", modes)
        object.__setattr__(self, "temperature_grid", tuple(float(value) for value in self.temperature_grid))
        object.__setattr__(self, "c_grid", tuple(float(value) for value in self.c_grid))
        if int(self.scalar_weight_trials) < 0:
            raise ValueError("scalar_weight_trials cannot be negative")
        if int(self.binary_weight_trials) < 0:
            raise ValueError("binary_weight_trials cannot be negative")
        object.__setattr__(self, "scalar_weight_trials", int(self.scalar_weight_trials))
        object.__setattr__(self, "binary_weight_trials", int(self.binary_weight_trials))
        mix = float(self.classwise_uniform_mix)
        if mix < 0.0 or mix > 1.0:
            raise ValueError("classwise_uniform_mix must be in [0, 1]")
        object.__setattr__(self, "classwise_uniform_mix", mix)
        object.__setattr__(self, "control_seed", int(self.control_seed))


def default_architecture_view_10class_fusion_groups(model_names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    available = {normalize_architecture_view_variant(name) for name in model_names}
    groups: dict[str, tuple[str, ...]] = {}
    core = tuple(
        name
        for name in (
            ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
            ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
            ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART,
            ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART,
        )
        if name in available
    )
    if len(core) >= 2:
        groups["av10_architecture_view_core"] = core
    control = tuple(
        name
        for name in (
            ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
            ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL,
            ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
            ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
            ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART,
        )
        if name in available
    )
    if len(control) >= 2:
        groups["av10_with_controls"] = control
    if len(available) >= 2:
        groups["av10_all_available"] = tuple(name for name in model_names if normalize_architecture_view_variant(name) in available)
    if not groups and model_names:
        groups["av10_singleton"] = (normalize_architecture_view_variant(model_names[0]),)
    return groups


def load_architecture_view_10class_prediction_block(
    prediction_dir: str | Path,
    variant: str,
    split: str,
    *,
    verify_hash: bool = True,
) -> ArchitectureView10ClassPredictionBlock:
    """Load one AV10 prediction cache block."""

    variant = normalize_architecture_view_variant(variant)
    metadata = load_architecture_view_10class_prediction_metadata(prediction_dir, variant, split)
    array_path, _metadata_path = architecture_view_10class_prediction_paths(prediction_dir, variant, split)
    arrays = _read_prediction_arrays(array_path)
    if bool(verify_hash) and hash_arrays(arrays) != metadata.get("prediction_content_hash"):
        raise ValueError(f"prediction content hash mismatch for {variant}/{split}")
    logits = np.asarray(arrays["logits"], dtype=np.float32)
    probs = np.asarray(arrays.get("probs", _softmax_np(logits)), dtype=np.float32)
    labels = np.asarray(arrays["labels"], dtype=np.int64).reshape(-1)
    preds = np.asarray(arrays.get("preds", np.argmax(logits, axis=1)), dtype=np.int64).reshape(-1)
    files = [str(path) for path in metadata.get("jet_files", ())]
    file_indices = np.asarray(arrays["jet_file_indices"], dtype=np.int64)
    entries = np.asarray(arrays["jet_entries"], dtype=np.int64)
    jet_ids = tuple(
        JetIdentity(file=files[int(file_index)], entry=int(entry), label=int(label))
        for file_index, entry, label in zip(file_indices, entries, labels)
    )
    if logits.ndim != 2 or int(logits.shape[1]) != 10:
        raise ValueError(f"AV10 logits must have shape [N, 10], got {logits.shape}")
    if probs.shape != logits.shape:
        raise ValueError("AV10 probs/logits shape mismatch")
    if labels.shape[0] != logits.shape[0] or preds.shape[0] != logits.shape[0]:
        raise ValueError("AV10 labels/preds/logits row mismatch")
    return ArchitectureView10ClassPredictionBlock(
        variant=variant,
        split=str(split),
        logits=logits,
        probs=probs,
        labels=labels,
        preds=preds,
        jet_ids=jet_ids,
        metadata=metadata,
    )


def load_architecture_view_10class_blocks_for_split(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    split: str,
) -> list[ArchitectureView10ClassPredictionBlock]:
    blocks = [load_architecture_view_10class_prediction_block(prediction_dir, name, split) for name in model_names]
    validate_architecture_view_10class_block_alignment(blocks)
    return blocks


def validate_architecture_view_10class_block_alignment(
    blocks: Sequence[ArchitectureView10ClassPredictionBlock],
) -> None:
    if not blocks:
        raise ValueError("no prediction blocks were provided")
    first = blocks[0]
    for block in blocks[1:]:
        if block.split != first.split:
            raise ValueError(f"split mismatch: {block.split} != {first.split}")
        if not np.array_equal(block.labels, first.labels):
            raise ValueError(f"label mismatch between {first.variant} and {block.variant}")
        if block.jet_ids != first.jet_ids:
            raise ValueError(f"jet identity mismatch between {first.variant} and {block.variant}")
        if tuple(block.metadata.get("label_names", ())) != tuple(first.metadata.get("label_names", ())):
            raise ValueError(f"label name mismatch between {first.variant} and {block.variant}")


def av10_metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    logits_for_projection: np.ndarray | None = None,
) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.argmax(probs, axis=1).astype(np.int64)
    ce = _cross_entropy_from_probs(probs, labels)
    metrics = classification_metrics_from_predictions(
        preds=preds,
        labels=labels,
        loss_sum=ce * float(max(labels.size, 1)),
        logits=None,
        label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    )
    metrics["cross_entropy"] = ce
    metrics["loss"] = ce
    if logits_for_projection is not None:
        metrics["binary_projection_metrics"] = architecture_view_binary_projection_metrics(
            np.asarray(logits_for_projection, dtype=np.float32),
            labels,
            label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
            pairs=ARCHITECTURE_VIEW_10CLASS_BINARY_PROJECTION_PAIRS,
        )
    return metrics


def av10_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float32)
    return av10_metrics_from_probs(_softmax_np(logits), labels, logits_for_projection=logits)


def _stack_logits(blocks: Sequence[ArchitectureView10ClassPredictionBlock]) -> np.ndarray:
    validate_architecture_view_10class_block_alignment(blocks)
    return np.stack([block.logits for block in blocks], axis=0).astype(np.float32)


def _stack_probs(blocks: Sequence[ArchitectureView10ClassPredictionBlock]) -> np.ndarray:
    validate_architecture_view_10class_block_alignment(blocks)
    return np.stack([block.probs for block in blocks], axis=0).astype(np.float32)


def _candidate_scalar_weights(n_models: int, *, trials: int, seed: int) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    uniform = np.full((n_models,), 1.0 / float(max(n_models, 1)), dtype=np.float64)
    candidates.append(uniform)
    for index in range(n_models):
        one_hot = np.zeros((n_models,), dtype=np.float64)
        one_hot[index] = 1.0
        candidates.append(one_hot)
    rng = np.random.RandomState(int(seed))
    for _ in range(int(trials)):
        candidates.append(rng.dirichlet(np.ones((n_models,), dtype=np.float64)))
    return candidates


def _weighted_logit_mean(logits_by_model: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("mnc,m->nc", np.asarray(logits_by_model, dtype=np.float64), np.asarray(weights, dtype=np.float64)).astype(np.float32)


def _classwise_weighted_logit_mean(logits_by_model: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("mnc,mc->nc", np.asarray(logits_by_model, dtype=np.float64), np.asarray(weights, dtype=np.float64)).astype(np.float32)


def _fit_temperature_scaled(blocks_train: Sequence[ArchitectureView10ClassPredictionBlock], *, grid: Sequence[float]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    temperatures: list[float] = []
    for block in blocks_train:
        candidates = []
        for temp in grid:
            temp = float(temp)
            metrics = av10_metrics_from_logits(block.logits / max(temp, 1.0e-6), block.labels)
            candidates.append({"temperature": temp, "metrics": metrics})
        best = min(candidates, key=lambda row: row["metrics"]["cross_entropy"])
        temperatures.append(float(best["temperature"]))
        rows.append({"variant": block.variant, "selected_temperature": float(best["temperature"]), "candidates": candidates})
    return {"temperatures": np.asarray(temperatures, dtype=np.float64), "per_model": rows}


def _fit_scalar_weights(
    blocks_train: Sequence[ArchitectureView10ClassPredictionBlock],
    blocks_val: Sequence[ArchitectureView10ClassPredictionBlock],
    *,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    train_logits = _stack_logits(blocks_train)
    val_logits = _stack_logits(blocks_val)
    val_labels = blocks_val[0].labels
    candidates = []
    for weights in _candidate_scalar_weights(len(blocks_train), trials=trials, seed=seed):
        train_metrics = av10_metrics_from_logits(_weighted_logit_mean(train_logits, weights), blocks_train[0].labels)
        val_metrics = av10_metrics_from_logits(_weighted_logit_mean(val_logits, weights), val_labels)
        candidates.append({"weights": weights, "train_metrics": train_metrics, "stack_val_metrics": val_metrics})
    best = max(candidates, key=lambda row: _selection_score(row["stack_val_metrics"]))
    return {
        "weights": np.asarray(best["weights"], dtype=np.float64),
        "selected_stack_val_metrics": best["stack_val_metrics"],
        "candidate_count": len(candidates),
        "candidates": [
            {"weights": row["weights"].tolist(), "stack_val_metrics": row["stack_val_metrics"]}
            for row in candidates[: min(len(candidates), 32)]
        ],
    }


def _fit_classwise_weights(
    blocks_train: Sequence[ArchitectureView10ClassPredictionBlock],
    *,
    uniform_mix: float,
) -> dict[str, Any]:
    n_models = len(blocks_train)
    n_classes = int(blocks_train[0].logits.shape[1])
    weights = np.full((n_models, n_classes), 1.0 / float(max(n_models, 1)), dtype=np.float64)
    for class_index in range(n_classes):
        class_losses: list[float] = []
        mask = blocks_train[0].labels == class_index
        for block in blocks_train:
            if not bool(mask.any()):
                class_losses.append(0.0)
                continue
            probs = _softmax_np(block.logits[mask])
            class_losses.append(_cross_entropy_from_probs(probs, block.labels[mask]))
        losses = np.asarray(class_losses, dtype=np.float64)
        centered = losses - float(np.min(losses))
        raw = np.exp(-centered)
        raw = raw / np.clip(raw.sum(), 1.0e-12, None)
        weights[:, class_index] = (1.0 - float(uniform_mix)) * raw + float(uniform_mix) / float(n_models)
    return {"weights": weights, "uniform_mix": float(uniform_mix)}


def _feature_matrix(blocks: Sequence[ArchitectureView10ClassPredictionBlock]) -> np.ndarray:
    validate_architecture_view_10class_block_alignment(blocks)
    return np.concatenate([block.logits for block in blocks], axis=1).astype(np.float64)


def _standardize(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    return (x_train - mean) / scale, mean, scale


def _ridge_fit(x_train: np.ndarray, y_train: np.ndarray, *, C: float, num_classes: int) -> dict[str, np.ndarray | float]:
    x_std, mean, scale = _standardize(np.asarray(x_train, dtype=np.float64))
    x_aug = np.concatenate([x_std, np.ones((x_std.shape[0], 1), dtype=np.float64)], axis=1)
    y = np.eye(num_classes, dtype=np.float64)[np.asarray(y_train, dtype=np.int64)]
    l2 = 1.0 / max(float(C), 1.0e-9)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * l2
    penalty[-1, -1] = 0.0
    gram = x_aug.T @ x_aug + penalty
    rhs = x_aug.T @ y
    try:
        params = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        params = np.linalg.pinv(gram) @ rhs
    return {
        "coef": params[:-1].T.astype(np.float64),
        "intercept": params[-1].astype(np.float64),
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
        "C": float(C),
    }


def _ridge_predict(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    x = (np.asarray(x, dtype=np.float64) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return (x @ np.asarray(model["coef"]).T + np.asarray(model["intercept"])).astype(np.float32)


def _fit_ridge_stacker(
    blocks_train: Sequence[ArchitectureView10ClassPredictionBlock],
    blocks_val: Sequence[ArchitectureView10ClassPredictionBlock],
    *,
    c_grid: Sequence[float],
) -> dict[str, Any]:
    x_train = _feature_matrix(blocks_train)
    y_train = blocks_train[0].labels
    x_val = _feature_matrix(blocks_val)
    y_val = blocks_val[0].labels
    num_classes = int(blocks_train[0].logits.shape[1])
    candidates = []
    for c_value in c_grid:
        model = _ridge_fit(x_train, y_train, C=float(c_value), num_classes=num_classes)
        val_metrics = av10_metrics_from_logits(_ridge_predict(model, x_val), y_val)
        candidates.append({"C": float(c_value), "model": model, "stack_val_metrics": val_metrics})
    best = max(candidates, key=lambda row: _selection_score(row["stack_val_metrics"]))
    return {
        "model": best["model"],
        "selected_C": float(best["C"]),
        "selected_stack_val_metrics": best["stack_val_metrics"],
        "candidates": [
            {"C": row["C"], "stack_val_metrics": row["stack_val_metrics"]}
            for row in candidates
        ],
    }


def _binary_score_metrics(scores: np.ndarray, labels_binary: np.ndarray) -> dict[str, Any]:
    logits = np.stack([-np.asarray(scores, dtype=np.float64), np.asarray(scores, dtype=np.float64)], axis=1).astype(np.float32)
    return binary_classification_metrics_from_logits(
        logits=logits,
        labels=np.asarray(labels_binary, dtype=np.int64),
        label_names=("background", "signal"),
    )


def _projection_scores(blocks: Sequence[ArchitectureView10ClassPredictionBlock], *, negative_label: str, positive_label: str) -> tuple[np.ndarray, np.ndarray]:
    labels = blocks[0].labels
    names = tuple(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES)
    neg_index = names.index(str(negative_label))
    pos_index = names.index(str(positive_label))
    keep = (labels == neg_index) | (labels == pos_index)
    scores = []
    for block in blocks:
        scores.append(block.logits[keep, pos_index] - block.logits[keep, neg_index])
    return np.stack(scores, axis=0).astype(np.float64), (labels[keep] == pos_index).astype(np.int64)


def _fit_binary_projection_weights(
    blocks_train: Sequence[ArchitectureView10ClassPredictionBlock],
    blocks_val: Sequence[ArchitectureView10ClassPredictionBlock],
    *,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    rng_seed = int(seed)
    output: dict[str, Any] = {}
    for pair_index, (negative_label, positive_label) in enumerate(ARCHITECTURE_VIEW_10CLASS_BINARY_PROJECTION_PAIRS):
        train_scores, train_binary = _projection_scores(blocks_train, negative_label=negative_label, positive_label=positive_label)
        val_scores, val_binary = _projection_scores(blocks_val, negative_label=negative_label, positive_label=positive_label)
        if train_binary.size == 0 or val_binary.size == 0:
            output[f"{negative_label}_vs_{positive_label}"] = {"available": False, "reason": "missing labels"}
            continue
        candidates = []
        for weights in _candidate_scalar_weights(len(blocks_train), trials=trials, seed=rng_seed + pair_index):
            score_val = np.einsum("mn,m->n", val_scores, weights)
            metrics_val = _binary_score_metrics(score_val, val_binary)
            candidates.append({"weights": weights, "stack_val_metrics": metrics_val})
        best = min(
            candidates,
            key=lambda row: (
                float(row["stack_val_metrics"].get("fpr_at_signal_eff_0p50", float("inf"))),
                -float(row["stack_val_metrics"].get("auc", 0.0) or 0.0),
            ),
        )
        output[f"{negative_label}_vs_{positive_label}"] = {
            "available": True,
            "negative_label": negative_label,
            "positive_label": positive_label,
            "weights": np.asarray(best["weights"], dtype=np.float64),
            "selected_stack_val_metrics": best["stack_val_metrics"],
            "candidate_count": len(candidates),
        }
    return output


def _evaluate_binary_projection_weights(
    fit: Mapping[str, Any],
    blocks_by_split: Mapping[str, Sequence[ArchitectureView10ClassPredictionBlock]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pair_key, row in fit.items():
        if not isinstance(row, Mapping) or not row.get("available"):
            output[pair_key] = dict(row) if isinstance(row, Mapping) else {"available": False}
            continue
        weights = np.asarray(row["weights"], dtype=np.float64)
        negative_label = str(row["negative_label"])
        positive_label = str(row["positive_label"])
        split_metrics = {}
        for split, blocks in blocks_by_split.items():
            scores, labels_binary = _projection_scores(blocks, negative_label=negative_label, positive_label=positive_label)
            fused_scores = np.einsum("mn,m->n", scores, weights)
            split_metrics[split] = _binary_score_metrics(fused_scores, labels_binary)
            split_metrics[split]["n_jets"] = int(labels_binary.size)
        output[pair_key] = {
            **{key: value for key, value in row.items() if key != "weights"},
            "weights": weights.tolist(),
            "fit_split": ARCHITECTURE_VIEW_10CLASS_FUSION_FIT_SPLIT,
            "selection_split": ARCHITECTURE_VIEW_10CLASS_FUSION_SELECTION_SPLIT,
            "metrics": split_metrics,
        }
    return output


def _evaluate_mode(
    mode: str,
    blocks_by_split: Mapping[str, Sequence[ArchitectureView10ClassPredictionBlock]],
    *,
    config: ArchitectureView10ClassFusionConfig,
    group_name: str,
) -> dict[str, Any]:
    train_blocks = blocks_by_split[ARCHITECTURE_VIEW_10CLASS_FUSION_FIT_SPLIT]
    val_blocks = blocks_by_split[ARCHITECTURE_VIEW_10CLASS_FUSION_SELECTION_SPLIT]
    if mode == "binary_projection_weighted":
        fit = _fit_binary_projection_weights(
            train_blocks,
            val_blocks,
            trials=int(config.binary_weight_trials),
            seed=int(config.control_seed) + len(group_name),
        )
        return {
            "mode": mode,
            "fit_split": ARCHITECTURE_VIEW_10CLASS_FUSION_FIT_SPLIT,
            "selection_split": ARCHITECTURE_VIEW_10CLASS_FUSION_SELECTION_SPLIT,
            "binary_projection_results": _evaluate_binary_projection_weights(fit, blocks_by_split),
        }
    fit: dict[str, Any] = {}
    if mode == "temperature_scaled_logit_mean":
        fit = _fit_temperature_scaled(train_blocks, grid=config.temperature_grid)
    elif mode == "scalar_weighted_logit_mean":
        fit = _fit_scalar_weights(
            train_blocks,
            val_blocks,
            trials=int(config.scalar_weight_trials),
            seed=int(config.control_seed) + 17 * len(group_name),
        )
    elif mode == "classwise_weighted_logit_mean":
        fit = _fit_classwise_weights(train_blocks, uniform_mix=float(config.classwise_uniform_mix))
    elif mode == "ridge_logit_stacker":
        fit = _fit_ridge_stacker(train_blocks, val_blocks, c_grid=config.c_grid)
    metrics_by_split: dict[str, Any] = {}
    for split, blocks in blocks_by_split.items():
        labels = blocks[0].labels
        logits_by_model = _stack_logits(blocks)
        if mode == "uniform_logit_mean":
            fused_logits = np.mean(logits_by_model, axis=0).astype(np.float32)
        elif mode == "uniform_probability_mean":
            fused_probs = np.mean(_stack_probs(blocks), axis=0).astype(np.float32)
            fused_logits = _log_probs(fused_probs)
        elif mode == "temperature_scaled_logit_mean":
            temperatures = np.asarray(fit["temperatures"], dtype=np.float64)[:, None, None]
            fused_logits = np.mean(logits_by_model / np.clip(temperatures, 1.0e-6, None), axis=0).astype(np.float32)
        elif mode == "scalar_weighted_logit_mean":
            fused_logits = _weighted_logit_mean(logits_by_model, np.asarray(fit["weights"], dtype=np.float64))
        elif mode == "classwise_weighted_logit_mean":
            fused_logits = _classwise_weighted_logit_mean(logits_by_model, np.asarray(fit["weights"], dtype=np.float64))
        elif mode == "ridge_logit_stacker":
            fused_logits = _ridge_predict(fit["model"], _feature_matrix(blocks))
        else:
            raise ValueError(f"unsupported fusion mode {mode!r}")
        metrics_by_split[split] = av10_metrics_from_logits(fused_logits, labels)
    fit_report = dict(fit)
    for key in ("weights", "temperatures"):
        if key in fit_report and isinstance(fit_report[key], np.ndarray):
            fit_report[key] = fit_report[key].tolist()
    if "model" in fit_report:
        model = fit_report["model"]
        fit_report["model"] = {
            "coef_shape": list(np.asarray(model["coef"]).shape),
            "intercept_shape": list(np.asarray(model["intercept"]).shape),
            "C": float(model["C"]),
        }
    return {
        "mode": mode,
        "fit_split": ARCHITECTURE_VIEW_10CLASS_FUSION_FIT_SPLIT,
        "selection_split": ARCHITECTURE_VIEW_10CLASS_FUSION_SELECTION_SPLIT,
        "fit": fit_report,
        "metrics": metrics_by_split,
    }


def _individual_metrics(blocks_by_split: Mapping[str, Mapping[str, ArchitectureView10ClassPredictionBlock]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_name, by_split in blocks_by_split.items():
        output[model_name] = {}
        for split, block in by_split.items():
            output[model_name][split] = av10_metrics_from_logits(block.logits, block.labels)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                keys.append(str(key))
    if not keys:
        keys = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_jsonable(row.get(key)), sort_keys=True) if isinstance(row.get(key), (Mapping, list, tuple)) else _jsonable(row.get(key)) for key in keys})


def _metric_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, group in report.get("groups", {}).items():
        for mode, mode_report in group.get("fusion_modes", {}).items():
            metrics = mode_report.get("metrics")
            if isinstance(metrics, Mapping):
                for split, split_metrics in metrics.items():
                    if not isinstance(split_metrics, Mapping):
                        continue
                    rows.append(
                        {
                            "group": group_name,
                            "mode": mode,
                            "split": split,
                            "metric_family": "10class",
                            "accuracy": split_metrics.get("accuracy"),
                            "macro_per_class_accuracy": split_metrics.get("macro_per_class_accuracy"),
                            "cross_entropy": split_metrics.get("cross_entropy"),
                            "n_jets": split_metrics.get("n_jets"),
                        }
                    )
            binary_results = mode_report.get("binary_projection_results")
            if isinstance(binary_results, Mapping):
                for pair_name, pair_report in binary_results.items():
                    if not isinstance(pair_report, Mapping) or not pair_report.get("available"):
                        continue
                    for split, split_metrics in pair_report.get("metrics", {}).items():
                        if not isinstance(split_metrics, Mapping):
                            continue
                        rows.append(
                            {
                                "group": group_name,
                                "mode": mode,
                                "split": split,
                                "metric_family": "binary_projection",
                                "binary_pair": pair_name,
                                "auc": split_metrics.get("auc"),
                                "fpr_at_signal_eff_0p30": split_metrics.get("fpr_at_signal_eff_0p30"),
                                "fpr_at_signal_eff_0p50": split_metrics.get("fpr_at_signal_eff_0p50"),
                                "background_rejection_at_signal_eff_0p50": split_metrics.get(
                                    "background_rejection_at_signal_eff_0p50"
                                ),
                                "n_jets": split_metrics.get("n_jets"),
                            }
                        )
    return rows


def run_architecture_view_10class_fusion(config: ArchitectureView10ClassFusionConfig) -> dict[str, Any]:
    """Run all requested AV10 fusion modes from cached predictions."""

    prediction_dir = Path(config.prediction_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks_by_model: dict[str, dict[str, ArchitectureView10ClassPredictionBlock]] = {}
    for model_name in config.model_names:
        blocks_by_model[model_name] = {}
        for split in ARCHITECTURE_VIEW_10CLASS_FUSION_EVAL_SPLITS:
            blocks_by_model[model_name][split] = load_architecture_view_10class_prediction_block(
                prediction_dir,
                model_name,
                split,
            )
    split_audit = {}
    for split in ARCHITECTURE_VIEW_10CLASS_FUSION_EVAL_SPLITS:
        split_blocks = [blocks_by_model[model_name][split] for model_name in config.model_names]
        validate_architecture_view_10class_block_alignment(split_blocks)
        split_audit[split] = {
            "n_jets": int(split_blocks[0].labels.shape[0]),
            "labels_hash": _labels_hash(split_blocks[0].labels),
            "jet_identity_hash": split_blocks[0].metadata.get("jet_identity_hash"),
            "per_model_prediction_hash": {
                block.variant: block.metadata.get("prediction_content_hash")
                for block in split_blocks
            },
        }
    group_reports: dict[str, Any] = {}
    for group_name, model_names in dict(config.groups or {}).items():
        blocks_by_split = {
            split: [blocks_by_model[model_name][split] for model_name in model_names]
            for split in ARCHITECTURE_VIEW_10CLASS_FUSION_EVAL_SPLITS
        }
        for blocks in blocks_by_split.values():
            validate_architecture_view_10class_block_alignment(blocks)
        modes = {}
        for mode in config.fusion_modes:
            modes[mode] = _evaluate_mode(
                mode,
                blocks_by_split,
                config=config,
                group_name=group_name,
            )
        group_reports[group_name] = {
            "model_names": list(model_names),
            "fusion_modes": modes,
        }
    report = {
        "contract": ARCHITECTURE_VIEW_10CLASS_FUSION_CONTRACT,
        "step": ARCHITECTURE_VIEW_10CLASS_FUSION_STEP,
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "source": source_metadata(),
        "leakage_rules": {
            "fit_split": ARCHITECTURE_VIEW_10CLASS_FUSION_FIT_SPLIT,
            "selection_split": ARCHITECTURE_VIEW_10CLASS_FUSION_SELECTION_SPLIT,
            "final_test_role": "evaluated only after stack_train fitting and stack_val fusion selection",
        },
        "split_audit": split_audit,
        "individual_metrics": _individual_metrics(blocks_by_model),
        "groups": group_reports,
    }
    rows = _metric_rows(report)
    output_paths = {
        "fusion_report": str(output_dir / "fusion_report.json"),
        "fusion_metric_table": str(output_dir / "fusion_metric_table.csv"),
        "run_report": str(output_dir / "run_report.json"),
    }
    report["outputs"] = output_paths
    save_json(output_dir / "fusion_report.json", _jsonable(report))
    save_json(output_dir / "run_report.json", _jsonable(report))
    _write_csv(output_dir / "fusion_metric_table.csv", rows)
    return _jsonable(report)


__all__ = [
    "ARCHITECTURE_VIEW_10CLASS_BINARY_PROJECTION_PAIRS",
    "ARCHITECTURE_VIEW_10CLASS_FUSION_CONTRACT",
    "ARCHITECTURE_VIEW_10CLASS_FUSION_EVAL_SPLITS",
    "ARCHITECTURE_VIEW_10CLASS_FUSION_FIT_SPLIT",
    "ARCHITECTURE_VIEW_10CLASS_FUSION_MODES",
    "ARCHITECTURE_VIEW_10CLASS_FUSION_SELECTION_SPLIT",
    "ARCHITECTURE_VIEW_10CLASS_FUSION_STEP",
    "ArchitectureView10ClassFusionConfig",
    "ArchitectureView10ClassPredictionBlock",
    "av10_metrics_from_logits",
    "av10_metrics_from_probs",
    "default_architecture_view_10class_fusion_groups",
    "load_architecture_view_10class_blocks_for_split",
    "load_architecture_view_10class_prediction_block",
    "run_architecture_view_10class_fusion",
    "validate_architecture_view_10class_block_alignment",
]

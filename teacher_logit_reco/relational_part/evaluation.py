"""Deterministic metrics and inference for the relational ParT campaign."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import require_sha256, with_content_hash

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EVALUATION_CONTRACT = "relational_part_evaluation_v1"
CLASS_NAMES = (
    "QCD",
    "Hbb",
    "Hcc",
    "Hgg",
    "H4q",
    "Hqql",
    "Zqq",
    "Wqq",
    "Tbqq",
    "Tbl",
)


def stable_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES):
        raise ValueError("logits must have shape [events,10]")
    if not np.isfinite(values).all():
        raise FloatingPointError("logits contain NaN or infinity")
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=bool)
    positives = int(positive.sum())
    negatives = int(len(positive) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires positive and negative support")
    ranks = _average_ranks(np.asarray(scores, dtype=np.float64))
    positive_rank_sum = float(ranks[positive].sum(dtype=np.float64))
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    bins: list[dict[str, Any]] = []
    ece = np.float64(0.0)
    for index in range(15):
        selected = (confidence >= edges[index]) & (
            (confidence < edges[index + 1])
            if index < 14
            else (confidence <= edges[index + 1])
        )
        count = int(selected.sum())
        if count:
            accuracy = float(correct[selected].mean(dtype=np.float64))
            mean_confidence = float(confidence[selected].mean(dtype=np.float64))
            contribution = (
                count / len(labels) * abs(accuracy - mean_confidence)
            )
        else:
            accuracy = None
            mean_confidence = None
            contribution = 0.0
        ece += np.float64(contribution)
        bins.append(
            {
                "index": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "right_inclusive": index == 14,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "contribution": float(contribution),
            }
        )
    return {"value": float(ece), "bins": bins}


def qcd_signal_rejection(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    signal_index: int,
    target_efficiency: float,
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not 1 <= int(signal_index) < len(CLASS_NAMES):
        raise ValueError("signal_index must identify a non-QCD class")
    signal = labels == int(signal_index)
    qcd = labels == 0
    signal_support = int(signal.sum())
    qcd_support = int(qcd.sum())
    if not signal_support or not qcd_support:
        raise ValueError("QCD rejection requires signal and QCD support")
    if not 0 < float(target_efficiency) <= 1:
        raise ValueError("target efficiency must lie in (0,1]")
    score = values[:, signal_index] - values[:, 0]
    rank = int(math.ceil(float(target_efficiency) * signal_support))
    threshold = float(np.sort(score[signal])[::-1][rank - 1])
    passed = score >= threshold
    signal_pass = int((passed & signal).sum())
    qcd_pass = int((passed & qcd).sum())
    false_positive_rate = qcd_pass / qcd_support
    return {
        "signal_class": CLASS_NAMES[signal_index],
        "target_signal_efficiency": float(target_efficiency),
        "achieved_signal_efficiency": signal_pass / signal_support,
        "discriminant": "logit_signal_minus_logit_QCD",
        "threshold": threshold,
        "pass_rule": "score_greater_than_or_equal_to_threshold",
        "signal_support": signal_support,
        "signal_pass_count": signal_pass,
        "qcd_support": qcd_support,
        "qcd_false_positive_count": qcd_pass,
        "qcd_false_positive_rate": false_positive_rate,
        "background_rejection": (
            None if qcd_pass == 0 else 1.0 / false_positive_rate
        ),
        "background_rejection_is_infinite": qcd_pass == 0,
    }


def evaluate_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape != (len(truth), len(CLASS_NAMES)):
        raise ValueError("logits and labels have incompatible shapes")
    if len(truth) == 0:
        raise ValueError("evaluation requires at least one event")
    if not np.isfinite(values).all():
        raise FloatingPointError("evaluation logits are nonfinite")
    if bool(((truth < 0) | (truth >= len(CLASS_NAMES))).any()):
        raise ValueError("labels lie outside the canonical class order")
    probabilities = stable_probabilities(values)
    predictions = probabilities.argmax(axis=1)
    log_norm = np.log(np.exp(values - values.max(1, keepdims=True)).sum(1))
    log_norm += values.max(1)
    cross_entropy = float(
        (log_norm - values[np.arange(len(truth)), truth]).mean(dtype=np.float64)
    )
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    supports = confusion.sum(axis=1)
    if bool((supports == 0).any()):
        missing = [CLASS_NAMES[index] for index in np.flatnonzero(supports == 0)]
        raise ValueError(f"evaluation lacks canonical class support: {missing}")
    efficiencies = np.diag(confusion) / supports
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[truth]
    brier = float(
        np.square(probabilities - one_hot).sum(axis=1).mean(dtype=np.float64)
    )
    auc = {
        name: _binary_auc(probabilities[:, index], truth == index)
        for index, name in enumerate(CLASS_NAMES)
    }
    rejection = {
        CLASS_NAMES[index]: {
            str(target): qcd_signal_rejection(
                values,
                truth,
                signal_index=index,
                target_efficiency=target,
            )
            for target in (0.30, 0.50)
        }
        for index in range(1, len(CLASS_NAMES))
    }
    calibration = expected_calibration_error(probabilities, truth)
    return with_content_hash(
        {
            "contract": EVALUATION_CONTRACT,
            "schema_version": 1,
            "split": str(split),
            "event_count": len(truth),
            "class_order": list(CLASS_NAMES),
            "accuracy": float((predictions == truth).mean(dtype=np.float64)),
            "cross_entropy": cross_entropy,
            "macro_per_class_accuracy": float(
                efficiencies.mean(dtype=np.float64)
            ),
            "per_class_efficiency": {
                name: float(efficiencies[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "one_vs_rest_auc": auc,
            "ece_15_bin_top_label": calibration,
            "brier_score": brier,
            "confusion_matrix": confusion.tolist(),
            "qcd_signal_rejection": rejection,
            "calculation_dtype": "float64",
        }
    )


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for model evaluation")
    moved: dict[str, Any] = {}
    for name, value in batch.items():
        moved[name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def model_forward(model: Any, batch: Mapping[str, Any]) -> Any:
    parameters = inspect.signature(model.forward).parameters
    aliases = {
        "lorentz_vectors": ("lorentz_vectors", "vectors"),
        "raw_tokens": ("raw_tokens", "tokens"),
    }
    kwargs: dict[str, Any] = {}
    for name in parameters:
        candidates = aliases.get(name, (name,))
        for candidate in candidates:
            if candidate in batch:
                kwargs[name] = batch[candidate]
                break
    required = [
        name
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if required:
        raise ValueError(f"batch lacks required model inputs: {required}")
    return model(**kwargs)


def evaluate_model(
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    *,
    split: str,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for model evaluation")
    resolved_device = torch.device(device)
    was_training = bool(model.training)
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, resolved_device)
            if "labels" not in batch:
                raise ValueError("evaluation batch lacks labels")
            output = model_forward(model, batch)
            if output.ndim != 2 or int(output.shape[1]) != len(CLASS_NAMES):
                raise ValueError("model output must have shape [events,10]")
            logits.append(output.detach().float().cpu().numpy())
            labels.append(batch["labels"].detach().long().cpu().numpy())
    if was_training:
        model.train()
    if not logits:
        raise ValueError("evaluation loader is empty")
    return evaluate_logits(
        np.concatenate(logits, axis=0),
        np.concatenate(labels, axis=0),
        split=split,
    )


def build_evaluation_contract(*, global_determinism_sha256: str) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": EVALUATION_CONTRACT,
            "schema_version": 1,
            "global_determinism_sha256": require_sha256(
                global_determinism_sha256, name="global_determinism_sha256"
            ),
            "class_order": list(CLASS_NAMES),
            "metrics": [
                "accuracy",
                "cross_entropy",
                "macro_per_class_accuracy",
                "per_class_efficiency",
                "one_vs_rest_auc",
                "15_bin_top_label_ECE",
                "multiclass_Brier",
                "confusion_matrix",
                "QCD_signal_rejection_at_0.30_and_0.50",
            ],
            "val_stop_selects_checkpoint": True,
            "val_select_selects_checkpoint": False,
            "val_select_evaluations_per_checkpoint": 1,
            "calculation_dtype": "float64",
        }
    )


__all__ = [
    "CLASS_NAMES",
    "EVALUATION_CONTRACT",
    "build_evaluation_contract",
    "evaluate_logits",
    "evaluate_model",
    "expected_calibration_error",
    "model_forward",
    "qcd_signal_rejection",
    "stable_probabilities",
]

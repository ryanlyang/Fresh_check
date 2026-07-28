"""Exact RETB classification metrics frozen before scientific inspection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import with_content_hash


CLASSIFICATION_METRICS_CONTRACT = "retb_classification_metrics_v1"
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


def _finite_number_or_infinity(value: float) -> float | str:
    """Return an RFC-8259-safe representation of an extended real number."""

    if math.isinf(float(value)):
        return "positive_infinity" if float(value) > 0 else "negative_infinity"
    if math.isnan(float(value)):
        raise FloatingPointError("metric value is NaN")
    return float(value)


def stable_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError("classification logits must have shape [N,10]")
    if not np.isfinite(values).all():
        raise FloatingPointError("classification logits are nonfinite")
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
    rank_sum = float(ranks[positive].sum(dtype=np.float64))
    return (
        rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    confidence = np.asarray(probabilities, dtype=np.float64).max(axis=1)
    prediction = np.asarray(probabilities, dtype=np.float64).argmax(axis=1)
    truth = np.asarray(labels, dtype=np.int64)
    edges = np.asarray([index / 15 for index in range(16)], dtype=np.float64)
    bins = []
    total = len(truth)
    value = 0.0
    for index in range(15):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == 14
            else confidence < edges[index + 1]
        )
        count = int(selected.sum())
        if count:
            accuracy = float((prediction[selected] == truth[selected]).mean())
            mean_confidence = float(confidence[selected].mean(dtype=np.float64))
            contribution = count / total * abs(mean_confidence - accuracy)
        else:
            accuracy = None
            mean_confidence = None
            contribution = 0.0
        value += contribution
        bins.append(
            {
                "index": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "left_inclusive": True,
                "right_inclusive": index == 14,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "contribution": float(contribution),
            }
        )
    return {
        "kind": "top_label_multiclass",
        "value": float(value),
        "bins": bins,
    }


def qcd_signal_rejection(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    signal_index: int,
    target_efficiency: float,
) -> dict[str, Any]:
    probability = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if int(signal_index) not in range(1, 10):
        raise ValueError("signal_index must identify a non-QCD class")
    if not 0.0 < float(target_efficiency) <= 1.0:
        raise ValueError("target signal efficiency lies outside (0,1]")
    denominator = probability[:, signal_index] + probability[:, 0]
    if bool((denominator <= 0).any()):
        raise FloatingPointError("QCD-versus-signal denominator is nonpositive")
    score = probability[:, signal_index] / denominator
    signal = truth == signal_index
    qcd = truth == 0
    signal_scores = np.sort(score[signal])
    qcd_scores = np.sort(score[qcd])
    if len(signal_scores) == 0 or len(qcd_scores) == 0:
        raise ValueError("QCD rejection requires signal and QCD support")
    unique = np.unique(score)
    thresholds = np.concatenate(
        (
            np.asarray([np.inf]),
            unique[::-1],
            np.asarray([-np.inf]),
        )
    )
    signal_pass = len(signal_scores) - np.searchsorted(
        signal_scores, thresholds, side="left"
    )
    achieved = signal_pass / len(signal_scores)
    distances = np.abs(achieved - float(target_efficiency))
    minimum_distance = distances.min()
    eligible = np.flatnonzero(distances == minimum_distance)
    maximum_achieved = achieved[eligible].max()
    eligible = eligible[achieved[eligible] == maximum_achieved]
    selected_index = int(eligible[0])
    threshold = float(thresholds[selected_index])
    qcd_pass = int(
        len(qcd_scores)
        - np.searchsorted(qcd_scores, threshold, side="left")
    )
    background_efficiency = qcd_pass / len(qcd_scores)
    return {
        "signal_class": CLASS_NAMES[signal_index],
        "discriminant": "p_signal/(p_signal+p_QCD)",
        "target_signal_efficiency": float(target_efficiency),
        "threshold": _finite_number_or_infinity(threshold),
        "pass_rule": "score_greater_than_or_equal_to_threshold",
        "threshold_candidates": (
            "positive_infinity_union_unique_scores_union_negative_infinity"
        ),
        "achieved_signal_efficiency": float(achieved[selected_index]),
        "signal_support": int(len(signal_scores)),
        "signal_pass_count": int(signal_pass[selected_index]),
        "qcd_support": int(len(qcd_scores)),
        "qcd_pass_count": qcd_pass,
        "background_efficiency": float(background_efficiency),
        "background_rejection": (
            "positive_infinity"
            if qcd_pass == 0
            else float(1.0 / background_efficiency)
        ),
        "background_rejection_is_infinite": qcd_pass == 0,
        "finite_selection_rejection": (
            (len(qcd_scores) + 1) / (qcd_pass + 0.5)
        ),
    }


def evaluate_classification(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.shape != (len(truth), 10) or len(truth) == 0:
        raise ValueError("classification logits and labels have incompatible shapes")
    if bool(((truth < 0) | (truth >= 10)).any()):
        raise ValueError("classification labels lie outside 0..9")
    probabilities = stable_probabilities(values)
    prediction = probabilities.argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    supports = confusion.sum(axis=1)
    if bool((supports == 0).any()):
        missing = [CLASS_NAMES[index] for index in np.flatnonzero(supports == 0)]
        raise ValueError(f"classification metrics lack class support: {missing}")
    shifted = values - values.max(axis=1, keepdims=True)
    cross_entropy = float(
        (
            np.log(np.exp(shifted).sum(axis=1))
            - shifted[np.arange(len(truth)), truth]
        ).mean(dtype=np.float64)
    )
    efficiency = np.diag(confusion) / supports
    one_hot = np.eye(10, dtype=np.float64)[truth]
    rejection = {
        CLASS_NAMES[index]: {
            str(target): qcd_signal_rejection(
                probabilities,
                truth,
                signal_index=index,
                target_efficiency=target,
            )
            for target in (0.30, 0.50)
        }
        for index in range(1, 10)
    }
    return with_content_hash(
        {
            "contract": CLASSIFICATION_METRICS_CONTRACT,
            "schema_version": 1,
            "split": str(split),
            "event_count": int(len(truth)),
            "class_order": list(CLASS_NAMES),
            "accuracy": float((prediction == truth).mean(dtype=np.float64)),
            "cross_entropy": cross_entropy,
            "macro_per_class_accuracy": float(efficiency.mean(dtype=np.float64)),
            "per_class_efficiency": {
                name: float(efficiency[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "one_vs_rest_auc": {
                name: _binary_auc(probabilities[:, index], truth == index)
                for index, name in enumerate(CLASS_NAMES)
            },
            "confusion_matrix": confusion.tolist(),
            "brier_score": float(
                np.square(probabilities - one_hot)
                .sum(axis=1)
                .mean(dtype=np.float64)
            ),
            "ece_15_bin": expected_calibration_error(probabilities, truth),
            "qcd_signal_rejection": rejection,
        }
    )


__all__ = [
    "CLASSIFICATION_METRICS_CONTRACT",
    "CLASS_NAMES",
    "evaluate_classification",
    "expected_calibration_error",
    "qcd_signal_rejection",
    "stable_probabilities",
]

"""Shared rich prediction metrics for PD10 teacher and student reports."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import classification_metrics_from_logits, softmax_np
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .config import PD10_NUM_CLASSES


PD10_SIGNAL_EFFICIENCIES: tuple[float, ...] = (0.30, 0.50)


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _eff_key(efficiency: float) -> str:
    return f"{float(efficiency):.2f}".replace(".", "p")


def pd10_confusion_matrix_from_logits(logits: np.ndarray, labels: np.ndarray) -> list[list[int]]:
    preds = np.argmax(np.asarray(logits), axis=1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    matrix = np.zeros((PD10_NUM_CLASSES, PD10_NUM_CLASSES), dtype=np.int64)
    for truth, pred in zip(labels, preds.astype(np.int64)):
        if 0 <= truth < PD10_NUM_CLASSES and 0 <= pred < PD10_NUM_CLASSES:
            matrix[int(truth), int(pred)] += 1
    return matrix.tolist()


def pd10_calibration_from_logits(logits: np.ndarray, labels: np.ndarray, *, bins: int = 10) -> dict[str, Any]:
    probs = softmax_np(logits)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = preds == labels
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    bin_rows: list[dict[str, Any]] = []
    for index in range(int(bins)):
        low = edges[index]
        high = edges[index + 1]
        if index == int(bins) - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        count = int(mask.sum())
        if count:
            acc = float(correct[mask].mean())
            conf = float(confidences[mask].mean())
            ece += (count / max(1, int(labels.shape[0]))) * abs(acc - conf)
        else:
            acc = None
            conf = None
        bin_rows.append(
            {
                "bin": int(index),
                "low": float(low),
                "high": float(high),
                "n_jets": count,
                "accuracy": acc,
                "confidence": conf,
            }
        )
    return {
        "expected_calibration_error": float(ece),
        "mean_confidence": float(confidences.mean()) if len(confidences) else None,
        "calibration_bins": bin_rows,
    }


def _average_rank_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive, dtype=bool).reshape(-1)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.zeros(scores.shape[0], dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        # One-based average rank for ties.
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    pos_rank_sum = float(ranks[positive].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _threshold_for_signal_efficiency(signal_scores: np.ndarray, efficiency: float) -> float | None:
    signal_scores = np.asarray(signal_scores, dtype=np.float64).reshape(-1)
    if signal_scores.size == 0:
        return None
    ordered = np.sort(signal_scores)[::-1]
    keep = max(1, int(np.ceil(float(efficiency) * float(signal_scores.size))))
    return float(ordered[min(keep - 1, ordered.size - 1)])


def _fpr_at_threshold(background_scores: np.ndarray, threshold: float | None) -> float | None:
    if threshold is None:
        return None
    background_scores = np.asarray(background_scores, dtype=np.float64).reshape(-1)
    if background_scores.size == 0:
        return None
    return float(np.mean(background_scores >= float(threshold)))


def _efficiency_at_threshold(signal_scores: np.ndarray, threshold: float | None) -> float | None:
    if threshold is None:
        return None
    signal_scores = np.asarray(signal_scores, dtype=np.float64).reshape(-1)
    if signal_scores.size == 0:
        return None
    return float(np.mean(signal_scores >= float(threshold)))


def _threshold_metrics(scores: np.ndarray, positive: np.ndarray) -> dict[str, Any]:
    positive = np.asarray(positive, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    result: dict[str, Any] = {}
    signal_scores = scores[positive]
    background_scores = scores[~positive]
    for efficiency in PD10_SIGNAL_EFFICIENCIES:
        key = _eff_key(efficiency)
        threshold = _threshold_for_signal_efficiency(signal_scores, efficiency)
        fpr = _fpr_at_threshold(background_scores, threshold)
        result[f"threshold_at_signal_eff_{key}"] = threshold
        result[f"signal_eff_at_threshold_{key}"] = _efficiency_at_threshold(signal_scores, threshold)
        result[f"fpr_at_signal_eff_{key}"] = fpr
        result[f"background_rejection_at_signal_eff_{key}"] = None if not fpr else float(1.0 / fpr)
    return result


def pd10_score_thresholds_by_class(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    efficiencies: Sequence[float] = PD10_SIGNAL_EFFICIENCIES,
) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    thresholds: dict[str, Any] = {}
    for class_index, label_name in enumerate(LABEL_NAMES):
        signal_scores = probs[labels == class_index, class_index]
        class_thresholds: dict[str, Any] = {}
        for efficiency in efficiencies:
            key = _eff_key(efficiency)
            class_thresholds[key] = {
                "signal_efficiency_target": float(efficiency),
                "threshold": _threshold_for_signal_efficiency(signal_scores, efficiency),
            }
        thresholds[label_name] = class_thresholds
    return thresholds


def pd10_apply_score_thresholds_by_class(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds_by_class: Mapping[str, Any],
) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    result: dict[str, Any] = {}
    macro_fprs: dict[str, list[float]] = {_eff_key(eff): [] for eff in PD10_SIGNAL_EFFICIENCIES}
    for class_index, label_name in enumerate(LABEL_NAMES):
        class_thresholds = thresholds_by_class.get(label_name, {})
        if not isinstance(class_thresholds, Mapping):
            continue
        signal_scores = probs[labels == class_index, class_index]
        background_scores = probs[labels != class_index, class_index]
        class_result: dict[str, Any] = {}
        for key, item in class_thresholds.items():
            if not isinstance(item, Mapping):
                continue
            threshold = _finite_or_none(item.get("threshold"))
            fpr = _fpr_at_threshold(background_scores, threshold)
            signal_eff = _efficiency_at_threshold(signal_scores, threshold)
            class_result[str(key)] = {
                "threshold": threshold,
                "signal_efficiency": signal_eff,
                "fpr": fpr,
                "background_rejection": None if not fpr else float(1.0 / fpr),
                "n_signal": int(signal_scores.size),
                "n_background": int(background_scores.size),
            }
            if fpr is not None:
                macro_fprs.setdefault(str(key), []).append(float(fpr))
        result[label_name] = class_result
    for key, values in macro_fprs.items():
        result[f"macro_fpr_{key}"] = float(np.mean(values)) if values else None
    return result


def pd10_per_class_metrics_from_confusion(confusion_matrix: Sequence[Sequence[int]]) -> dict[str, Any]:
    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    if matrix.shape != (PD10_NUM_CLASSES, PD10_NUM_CLASSES):
        raise ValueError(f"confusion matrix must be {PD10_NUM_CLASSES}x{PD10_NUM_CLASSES}")
    total = float(matrix.sum())
    rows: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    per_class_accuracy: dict[str, float | None] = {}
    for class_index, label_name in enumerate(LABEL_NAMES):
        tp = float(matrix[class_index, class_index])
        fp = float(matrix[:, class_index].sum() - tp)
        fn = float(matrix[class_index, :].sum() - tp)
        tn = float(total - tp - fp - fn)
        support = tp + fn
        predicted = tp + fp
        precision = None if predicted <= 0.0 else tp / predicted
        recall = None if support <= 0.0 else tp / support
        f1 = None if precision is None or recall is None or (precision + recall) <= 0.0 else 2.0 * precision * recall / (precision + recall)
        specificity = None if (tn + fp) <= 0.0 else tn / (tn + fp)
        one_vs_rest_accuracy = None if total <= 0.0 else (tp + tn) / total
        if precision is not None:
            precisions.append(float(precision))
        if recall is not None:
            recalls.append(float(recall))
            per_class_accuracy[label_name] = float(recall)
        else:
            per_class_accuracy[label_name] = None
        if f1 is not None:
            f1s.append(float(f1))
        rows.append(
            {
                "class_index": int(class_index),
                "label": label_name,
                "support": int(support),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "one_vs_rest_accuracy": one_vs_rest_accuracy,
            }
        )
    return {
        "per_class_metrics": rows,
        "per_class_accuracy": per_class_accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else None,
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
    }


def pd10_binary_projection_metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    class_pairs: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if class_pairs is None:
        class_pairs = tuple(combinations(range(PD10_NUM_CLASSES), 2))
    results: dict[str, Any] = {}
    for signal_class, background_class in class_pairs:
        signal_class = int(signal_class)
        background_class = int(background_class)
        mask = (labels == signal_class) | (labels == background_class)
        task_name = f"{LABEL_NAMES[signal_class]}_vs_{LABEL_NAMES[background_class]}"
        n_signal = int(np.sum(labels[mask] == signal_class))
        n_background = int(np.sum(labels[mask] == background_class))
        if n_signal == 0 or n_background == 0:
            results[task_name] = {
                "available": False,
                "reason": "missing signal or background examples",
                "signal_label": LABEL_NAMES[signal_class],
                "background_label": LABEL_NAMES[background_class],
                "n_signal": n_signal,
                "n_background": n_background,
            }
            continue
        denom = probs[mask, signal_class] + probs[mask, background_class] + 1.0e-12
        scores = probs[mask, signal_class] / denom
        positive = labels[mask] == signal_class
        threshold_metrics = _threshold_metrics(scores, positive)
        results[task_name] = {
            "available": True,
            "signal_label": LABEL_NAMES[signal_class],
            "background_label": LABEL_NAMES[background_class],
            "n_signal": n_signal,
            "n_background": n_background,
            "auc": _average_rank_auc(scores, positive),
            **threshold_metrics,
        }
    return results


def pd10_binary_score_thresholds_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    class_pairs: Sequence[tuple[int, int]] | None = None,
    efficiencies: Sequence[float] = PD10_SIGNAL_EFFICIENCIES,
) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if class_pairs is None:
        class_pairs = tuple(combinations(range(PD10_NUM_CLASSES), 2))
    thresholds: dict[str, Any] = {}
    for signal_class, background_class in class_pairs:
        signal_class = int(signal_class)
        background_class = int(background_class)
        mask = (labels == signal_class) | (labels == background_class)
        task_name = f"{LABEL_NAMES[signal_class]}_vs_{LABEL_NAMES[background_class]}"
        positive = labels[mask] == signal_class
        if int(positive.sum()) == 0 or int((~positive).sum()) == 0:
            thresholds[task_name] = {
                "available": False,
                "reason": "missing signal or background examples",
                "signal_label": LABEL_NAMES[signal_class],
                "background_label": LABEL_NAMES[background_class],
                "signal_class": signal_class,
                "background_class": background_class,
            }
            continue
        denom = probs[mask, signal_class] + probs[mask, background_class] + 1.0e-12
        scores = probs[mask, signal_class] / denom
        signal_scores = scores[positive]
        thresholds[task_name] = {
            "available": True,
            "signal_label": LABEL_NAMES[signal_class],
            "background_label": LABEL_NAMES[background_class],
            "signal_class": signal_class,
            "background_class": background_class,
            "thresholds": {
                _eff_key(efficiency): {
                    "signal_efficiency_target": float(efficiency),
                    "threshold": _threshold_for_signal_efficiency(signal_scores, efficiency),
                }
                for efficiency in efficiencies
            },
        }
    return thresholds


def pd10_apply_binary_score_thresholds_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds_by_task: Mapping[str, Any],
) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    results: dict[str, Any] = {}
    macro_fprs: dict[str, list[float]] = {_eff_key(eff): [] for eff in PD10_SIGNAL_EFFICIENCIES}
    for task_name, payload in thresholds_by_task.items():
        if not isinstance(payload, Mapping) or not bool(payload.get("available", True)):
            results[str(task_name)] = {
                "available": False,
                "reason": None if not isinstance(payload, Mapping) else payload.get("reason"),
            }
            continue
        signal_class = int(payload.get("signal_class"))
        background_class = int(payload.get("background_class"))
        mask = (labels == signal_class) | (labels == background_class)
        positive = labels[mask] == signal_class
        n_signal = int(positive.sum())
        n_background = int((~positive).sum())
        if n_signal == 0 or n_background == 0:
            results[str(task_name)] = {
                "available": False,
                "reason": "missing signal or background examples",
                "signal_label": payload.get("signal_label"),
                "background_label": payload.get("background_label"),
                "n_signal": n_signal,
                "n_background": n_background,
            }
            continue
        denom = probs[mask, signal_class] + probs[mask, background_class] + 1.0e-12
        scores = probs[mask, signal_class] / denom
        signal_scores = scores[positive]
        background_scores = scores[~positive]
        task_thresholds = payload.get("thresholds", {})
        task_result: dict[str, Any] = {
            "available": True,
            "signal_label": payload.get("signal_label"),
            "background_label": payload.get("background_label"),
            "signal_class": signal_class,
            "background_class": background_class,
            "n_signal": n_signal,
            "n_background": n_background,
        }
        if not isinstance(task_thresholds, Mapping):
            task_thresholds = {}
        for key, item in task_thresholds.items():
            if not isinstance(item, Mapping):
                continue
            threshold = _finite_or_none(item.get("threshold"))
            fpr = _fpr_at_threshold(background_scores, threshold)
            signal_eff = _efficiency_at_threshold(signal_scores, threshold)
            key = str(key)
            task_result[key] = {
                "threshold": threshold,
                "signal_efficiency": signal_eff,
                "fpr": fpr,
                "background_rejection": None if not fpr else float(1.0 / fpr),
            }
            if fpr is not None:
                macro_fprs.setdefault(key, []).append(float(fpr))
        results[str(task_name)] = task_result
    for key, values in macro_fprs.items():
        results[f"macro_fpr_{key}"] = float(np.mean(values)) if values else None
    return results


def pd10_ovr_fpr_summary_from_probs(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    aucs: list[float] = []
    fprs_by_eff: dict[str, list[float]] = {_eff_key(eff): [] for eff in PD10_SIGNAL_EFFICIENCIES}
    for class_index in range(PD10_NUM_CLASSES):
        positive = labels == class_index
        if int(positive.sum()) == 0 or int((~positive).sum()) == 0:
            continue
        scores = probs[:, class_index]
        auc = _average_rank_auc(scores, positive)
        if auc is not None:
            aucs.append(float(auc))
        threshold_metrics = _threshold_metrics(scores, positive)
        for efficiency in PD10_SIGNAL_EFFICIENCIES:
            key = _eff_key(efficiency)
            fpr = _finite_or_none(threshold_metrics.get(f"fpr_at_signal_eff_{key}"))
            if fpr is not None:
                fprs_by_eff[key].append(float(fpr))
    result: dict[str, Any] = {
        "macro_ovr_auc": float(np.mean(aucs)) if aucs else None,
    }
    for key, values in fprs_by_eff.items():
        macro_fpr = float(np.mean(values)) if values else None
        result[f"fpr_at_signal_eff_{key}_macro"] = macro_fpr
        result[f"background_rejection_at_signal_eff_{key}_macro"] = None if not macro_fpr else float(1.0 / macro_fpr)
    return result


def pd10_prediction_metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    validation_thresholds_by_class: Mapping[str, Any] | None = None,
    validation_binary_thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = softmax_np(logits)
    metrics: dict[str, Any] = dict(classification_metrics_from_logits(logits, labels))
    confusion = pd10_confusion_matrix_from_logits(logits, labels)
    metrics["confusion_matrix"] = confusion
    metrics.update(pd10_per_class_metrics_from_confusion(confusion))
    metrics.update(pd10_calibration_from_logits(logits, labels))
    metrics.update(pd10_ovr_fpr_summary_from_probs(probs, labels))
    metrics["binary_metrics"] = pd10_binary_projection_metrics_from_probs(probs, labels)
    metrics["binary_projection_results"] = metrics["binary_metrics"]
    thresholds = pd10_score_thresholds_by_class(probs, labels)
    metrics["score_thresholds_by_class"] = thresholds
    binary_thresholds = pd10_binary_score_thresholds_from_probs(probs, labels)
    metrics["binary_score_thresholds"] = binary_thresholds
    if validation_thresholds_by_class is not None:
        applied = pd10_apply_score_thresholds_by_class(probs, labels, validation_thresholds_by_class)
        metrics["validation_threshold_fpr"] = applied
        for efficiency in PD10_SIGNAL_EFFICIENCIES:
            key = _eff_key(efficiency)
            metrics[f"validation_threshold_fpr_at_signal_eff_{key}_macro"] = applied.get(f"macro_fpr_{key}")
    if validation_binary_thresholds is not None:
        binary_applied = pd10_apply_binary_score_thresholds_from_probs(probs, labels, validation_binary_thresholds)
        metrics["validation_binary_threshold_fpr"] = binary_applied
        binary_metrics = dict(metrics["binary_metrics"])
        for task_name, task_applied in binary_applied.items():
            if not isinstance(task_applied, Mapping) or str(task_name).startswith("macro_fpr_"):
                continue
            task_metrics = dict(binary_metrics.get(task_name, {}))
            task_metrics["validation_thresholds_from_split"] = "model_val"
            for efficiency in PD10_SIGNAL_EFFICIENCIES:
                key = _eff_key(efficiency)
                item = task_applied.get(key)
                if not isinstance(item, Mapping):
                    continue
                task_metrics[f"validation_threshold_at_signal_eff_{key}"] = item.get("threshold")
                task_metrics[f"validation_signal_eff_at_threshold_{key}"] = item.get("signal_efficiency")
                task_metrics[f"validation_fpr_at_signal_eff_{key}"] = item.get("fpr")
                task_metrics[f"validation_background_rejection_at_signal_eff_{key}"] = item.get(
                    "background_rejection"
                )
            binary_metrics[task_name] = task_metrics
        metrics["binary_metrics"] = binary_metrics
        metrics["binary_projection_results"] = binary_metrics
        for efficiency in PD10_SIGNAL_EFFICIENCIES:
            key = _eff_key(efficiency)
            macro_fpr = binary_applied.get(f"macro_fpr_{key}")
            metrics[f"validation_binary_fpr_at_signal_eff_{key}_macro"] = macro_fpr
            metrics[f"validation_binary_background_rejection_at_signal_eff_{key}_macro"] = (
                None if not macro_fpr else float(1.0 / float(macro_fpr))
            )
    return metrics


__all__ = [
    "PD10_SIGNAL_EFFICIENCIES",
    "pd10_apply_binary_score_thresholds_from_probs",
    "pd10_apply_score_thresholds_by_class",
    "pd10_binary_projection_metrics_from_probs",
    "pd10_binary_score_thresholds_from_probs",
    "pd10_calibration_from_logits",
    "pd10_confusion_matrix_from_logits",
    "pd10_ovr_fpr_summary_from_probs",
    "pd10_per_class_metrics_from_confusion",
    "pd10_prediction_metrics_from_logits",
    "pd10_score_thresholds_by_class",
]

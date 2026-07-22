"""Shared leakage-safe metrics for the local residual-field fusion campaign."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


LOCAL_RESIDUAL_FIELD_MULTICLASS_METRICS_CONTRACT = "local_residual_field_multiclass_metrics_v1"
LOCAL_RESIDUAL_FIELD_BINARY_PROJECTION_CONTRACT = "local_residual_field_binary_projection_metrics_v2"
LOCAL_RESIDUAL_FIELD_COMPLEMENTARITY_CONTRACT = "local_residual_field_complementarity_metrics_v1"
LOCAL_RESIDUAL_FIELD_BOOTSTRAP_CONTRACT = "local_residual_field_paired_bootstrap_v1"
LOCAL_RESIDUAL_FIELD_BINARY_SIGNAL_EFFICIENCIES = (0.30, 0.50)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _validate_logits_labels(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    label_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    names = tuple(str(name) for name in label_names)
    if values.ndim != 2 or truth.shape != (values.shape[0],):
        raise ValueError("logits/labels must have shapes [N,C] and [N]")
    if values.shape[1] != len(names) or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("label_names do not match the logit columns")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    if truth.size and (int(truth.min()) < 0 or int(truth.max()) >= len(names)):
        raise ValueError("labels contain an out-of-range class index")
    return values, truth, names


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(scores.size - n_positive)
    if not n_positive or not n_negative:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / float(n_positive * n_negative)


def local_residual_field_multiclass_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    label_names: Sequence[str],
    calibration_bins: int = 15,
) -> dict[str, Any]:
    """Compute the complete shared ten-class metric payload."""

    values, truth, names = _validate_logits_labels(logits, labels, label_names=label_names)
    if not truth.size:
        raise ValueError("multiclass metrics require at least one row")
    bins = int(calibration_bins)
    if bins <= 1:
        raise ValueError("calibration_bins must be greater than one")
    probabilities = _softmax(values)
    predictions = np.argmax(values, axis=1).astype(np.int64)
    picked = np.clip(probabilities[np.arange(len(truth)), truth], 1.0e-12, 1.0)
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    per_class: list[dict[str, Any]] = []
    auc_values: list[float] = []
    for index, name in enumerate(names):
        support = int(confusion[index].sum())
        correct = int(confusion[index, index])
        auc = _binary_auc(probabilities[:, index], truth == index)
        if auc is not None:
            auc_values.append(float(auc))
        per_class.append(
            {
                "class_index": index,
                "class_name": name,
                "support": support,
                "correct": correct,
                "accuracy": None if not support else correct / float(support),
                "one_vs_rest_auc": auc,
            }
        )
    confidence = np.max(probabilities, axis=1)
    correct = predictions == truth
    ece = 0.0
    calibration_rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        keep = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        count = int(np.count_nonzero(keep))
        if count:
            bin_accuracy = float(np.mean(correct[keep]))
            bin_confidence = float(np.mean(confidence[keep]))
            ece += count / float(len(truth)) * abs(bin_accuracy - bin_confidence)
        else:
            bin_accuracy = None
            bin_confidence = None
        calibration_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": bin_accuracy,
                "mean_confidence": bin_confidence,
            }
        )
    one_hot = np.eye(len(names), dtype=np.float64)[truth]
    return {
        "contract": LOCAL_RESIDUAL_FIELD_MULTICLASS_METRICS_CONTRACT,
        "n_jets": int(len(truth)),
        "label_names": list(names),
        "accuracy": float(np.mean(correct)),
        "cross_entropy": float(-np.mean(np.log(picked))),
        "macro_one_vs_rest_auc": None if not auc_values else float(np.mean(auc_values)),
        "macro_per_class_accuracy": float(
            np.mean([row["accuracy"] for row in per_class if row["accuracy"] is not None])
        ),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "expected_calibration_error": float(ece),
        "calibration_bins": calibration_rows,
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def _wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / float(total)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def _threshold_for_efficiency(signal_scores: np.ndarray, target_efficiency: float) -> float:
    signal = np.asarray(signal_scores, dtype=np.float64)
    if not signal.size:
        raise ValueError("cannot derive a threshold without signal rows")
    order = np.argsort(-signal, kind="mergesort")
    index = min(max(int(np.ceil(float(target_efficiency) * signal.size)) - 1, 0), signal.size - 1)
    return float(signal[order[index]])


def _operating_point(
    scores: np.ndarray,
    positive: np.ndarray,
    *,
    target_efficiency: float,
    threshold: float | None,
    convention: str,
) -> dict[str, Any]:
    signal_scores = np.asarray(scores[positive], dtype=np.float64)
    background_scores = np.asarray(scores[~positive], dtype=np.float64)
    signal_support = int(signal_scores.size)
    background_support = int(background_scores.size)
    if not signal_support or not background_support:
        return {
            "available": False,
            "threshold_convention": convention,
            "target_signal_efficiency": float(target_efficiency),
            "signal_support": signal_support,
            "qcd_support": background_support,
        }
    active_threshold = (
        _threshold_for_efficiency(signal_scores, target_efficiency)
        if threshold is None
        else float(threshold)
    )
    signal_pass = int(np.count_nonzero(signal_scores >= active_threshold))
    false_positives = int(np.count_nonzero(background_scores >= active_threshold))
    fpr = false_positives / float(background_support)
    interval = _wilson_interval(false_positives, background_support)
    return {
        "available": True,
        "threshold_convention": convention,
        "target_signal_efficiency": float(target_efficiency),
        "realized_signal_efficiency": signal_pass / float(signal_support),
        "threshold": active_threshold,
        "signal_pass_count": signal_pass,
        "signal_support": signal_support,
        "qcd_false_positive_count": false_positives,
        "qcd_support": background_support,
        "false_positive_rate": fpr,
        "false_positive_rate_interval_95": interval,
        "interval_method": "wilson_95",
        "jeffreys_smoothed_false_positive_rate": (false_positives + 0.5) / float(background_support + 1),
        "background_rejection": None if fpr <= 0.0 else float(1.0 / fpr),
        "background_rejection_lower_bound_95": (
            None if interval is None or interval[1] <= 0.0 else float(1.0 / interval[1])
        ),
    }


def local_residual_field_binary_projection_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    label_names: Sequence[str],
    signal_efficiencies: Sequence[float] = LOCAL_RESIDUAL_FIELD_BINARY_SIGNAL_EFFICIENCIES,
    frozen_thresholds: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Compute all QCD-vs-signal diagnostics, optionally applying frozen thresholds."""

    values, truth, names = _validate_logits_labels(logits, labels, label_names=label_names)
    if "QCD" not in names:
        raise ValueError("binary projection metrics require a QCD class")
    efficiencies = tuple(float(value) for value in signal_efficiencies)
    if not efficiencies or any(value <= 0.0 or value > 1.0 for value in efficiencies):
        raise ValueError("signal efficiencies must lie in (0, 1]")
    qcd_index = names.index("QCD")
    projections: dict[str, Any] = {}
    for signal_index, signal_name in enumerate(names):
        if signal_index == qcd_index:
            continue
        keep = (truth == qcd_index) | (truth == signal_index)
        pair_truth = truth[keep]
        pair_scores = values[keep, signal_index] - values[keep, qcd_index]
        positive = pair_truth == signal_index
        key = f"QCD_vs_{signal_name}"
        within = {
            f"signal_efficiency_{efficiency:.2f}": _operating_point(
                pair_scores,
                positive,
                target_efficiency=efficiency,
                threshold=None,
                convention="within_split_matched_efficiency",
            )
            for efficiency in efficiencies
        }
        frozen: dict[str, Any] = {}
        thresholds_for_pair = None if frozen_thresholds is None else frozen_thresholds.get(key)
        if thresholds_for_pair is not None:
            for efficiency in efficiencies:
                efficiency_key = f"signal_efficiency_{efficiency:.2f}"
                if efficiency_key not in thresholds_for_pair:
                    raise ValueError(f"frozen thresholds are missing {key}/{efficiency_key}")
                frozen[efficiency_key] = _operating_point(
                    pair_scores,
                    positive,
                    target_efficiency=efficiency,
                    threshold=float(thresholds_for_pair[efficiency_key]),
                    convention="stack_val_frozen_threshold",
                )
        projections[key] = {
            "available": bool(np.any(positive) and np.any(~positive)),
            "score_definition": f"logit_{signal_name}-logit_QCD",
            "negative_class_name": "QCD",
            "positive_class_name": signal_name,
            "n_jets": int(np.count_nonzero(keep)),
            "qcd_support": int(np.count_nonzero(~positive)),
            "signal_support": int(np.count_nonzero(positive)),
            "operating_points": within,
            "frozen_threshold_operating_points": frozen,
        }
    return {
        "contract": LOCAL_RESIDUAL_FIELD_BINARY_PROJECTION_CONTRACT,
        "label_names": list(names),
        "signal_efficiencies": list(efficiencies),
        "frozen_threshold_source": None if frozen_thresholds is None else "stack_val",
        "projections": projections,
    }


def freeze_binary_projection_thresholds(metrics: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    if metrics.get("contract") != LOCAL_RESIDUAL_FIELD_BINARY_PROJECTION_CONTRACT:
        raise ValueError("cannot freeze thresholds from an incompatible binary metric artifact")
    output: dict[str, dict[str, float]] = {}
    for pair, row in dict(metrics.get("projections") or {}).items():
        operating = row.get("operating_points") if isinstance(row, Mapping) else None
        if not isinstance(operating, Mapping):
            raise ValueError(f"binary projection {pair} lacks operating points")
        output[str(pair)] = {
            str(key): float(value["threshold"])
            for key, value in operating.items()
            if isinstance(value, Mapping) and value.get("available") is True
        }
    return output


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size < 2 or float(np.std(a)) <= 0.0 or float(np.std(b)) <= 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def local_residual_field_complementarity_metrics(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    *,
    label_names: Sequence[str],
    member_a: str = "A0",
    member_b: str = "member_B",
) -> dict[str, Any]:
    a, truth, names = _validate_logits_labels(logits_a, labels, label_names=label_names)
    b, other_truth, _ = _validate_logits_labels(logits_b, labels, label_names=label_names)
    if a.shape != b.shape or not np.array_equal(truth, other_truth):
        raise ValueError("complementarity members are not aligned")
    pred_a = np.argmax(a, axis=1)
    pred_b = np.argmax(b, axis=1)
    correct_a = pred_a == truth
    correct_b = pred_b == truth
    both_correct = int(np.count_nonzero(correct_a & correct_b))
    a_only = int(np.count_nonzero(correct_a & ~correct_b))
    b_only = int(np.count_nonzero(~correct_a & correct_b))
    both_wrong = int(np.count_nonzero(~correct_a & ~correct_b))
    error_union = a_only + b_only + both_wrong
    probabilities_a = _softmax(a)
    probabilities_b = _softmax(b)
    per_class = []
    for index, name in enumerate(names):
        keep = truth == index
        support = int(np.count_nonzero(keep))
        per_class.append(
            {
                "class_index": index,
                "class_name": name,
                "support": support,
                "disagreement_rate": None if not support else float(np.mean(pred_a[keep] != pred_b[keep])),
                "logit_correlation": _safe_correlation(a[keep, index], b[keep, index]),
                "probability_correlation": _safe_correlation(
                    probabilities_a[keep, index], probabilities_b[keep, index]
                ),
            }
        )
    return {
        "contract": LOCAL_RESIDUAL_FIELD_COMPLEMENTARITY_CONTRACT,
        "member_a": str(member_a),
        "member_b": str(member_b),
        "n_jets": int(len(truth)),
        "disagreement_count": int(np.count_nonzero(pred_a != pred_b)),
        "disagreement_rate": float(np.mean(pred_a != pred_b)),
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "error_overlap_jaccard": None if not error_union else both_wrong / float(error_union),
        "gain_on_a_error_count": b_only,
        "gain_on_a_error_rate": None if not (b_only + both_wrong) else b_only / float(b_only + both_wrong),
        "loss_on_a_correct_count": a_only,
        "loss_on_a_correct_rate": None if not (a_only + both_correct) else a_only / float(a_only + both_correct),
        "flattened_logit_correlation": _safe_correlation(a, b),
        "flattened_probability_correlation": _safe_correlation(probabilities_a, probabilities_b),
        "per_class": per_class,
    }


def _row_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probabilities = _softmax(logits)
    return -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-12, 1.0))


def paired_multiclass_bootstrap(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    *,
    label_names: Sequence[str],
    replicates: int = 1000,
    seed: int = 7319,
) -> dict[str, Any]:
    a, truth, _ = _validate_logits_labels(logits_a, labels, label_names=label_names)
    b, _, _ = _validate_logits_labels(logits_b, labels, label_names=label_names)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap members are not aligned")
    count = int(replicates)
    if count < 1000:
        raise ValueError("paired bootstrap requires at least 1000 replicates")
    rng = np.random.default_rng(int(seed))
    by_class = [np.flatnonzero(truth == index) for index in range(a.shape[1])]
    if any(not rows.size for rows in by_class):
        raise ValueError("stratified paired bootstrap requires every class")
    pred_a = np.argmax(a, axis=1)
    pred_b = np.argmax(b, axis=1)
    ce_a = _row_cross_entropy(a, truth)
    ce_b = _row_cross_entropy(b, truth)
    accuracy_deltas = np.empty(count, dtype=np.float64)
    ce_deltas = np.empty(count, dtype=np.float64)
    sample_digest = hashlib.sha256()
    for replicate in range(count):
        sampled = np.concatenate([rng.choice(rows, size=rows.size, replace=True) for rows in by_class])
        sample_digest.update(np.asarray(sampled, dtype=np.int64).tobytes())
        accuracy_deltas[replicate] = np.mean(pred_b[sampled] == truth[sampled]) - np.mean(
            pred_a[sampled] == truth[sampled]
        )
        ce_deltas[replicate] = np.mean(ce_b[sampled] - ce_a[sampled])
    def summary(values: np.ndarray, estimate: float) -> dict[str, Any]:
        return {
            "estimate": float(estimate),
            "mean": float(np.mean(values)),
            "interval_95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        }
    return {
        "contract": LOCAL_RESIDUAL_FIELD_BOOTSTRAP_CONTRACT,
        "replicates": count,
        "seed": int(seed),
        "sampled_index_hash": sample_digest.hexdigest(),
        "stratified_by_class": True,
        "accuracy_delta_b_minus_a": summary(
            accuracy_deltas,
            float(np.mean(pred_b == truth) - np.mean(pred_a == truth)),
        ),
        "cross_entropy_delta_b_minus_a": summary(
            ce_deltas,
            float(np.mean(ce_b - ce_a)),
        ),
    }


def paired_binary_projection_bootstrap(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    *,
    label_names: Sequence[str],
    signal_name: str,
    target_signal_efficiency: float = 0.50,
    replicates: int = 1000,
    seed: int = 7321,
) -> dict[str, Any]:
    """Stratified paired interval for one QCD operating-point comparison."""

    a, truth, names = _validate_logits_labels(logits_a, labels, label_names=label_names)
    b, other_truth, _ = _validate_logits_labels(logits_b, labels, label_names=label_names)
    if a.shape != b.shape or not np.array_equal(truth, other_truth):
        raise ValueError("paired binary bootstrap members are not aligned")
    if "QCD" not in names or str(signal_name) not in names or str(signal_name) == "QCD":
        raise ValueError("paired binary bootstrap requires a valid non-QCD signal_name")
    efficiency = float(target_signal_efficiency)
    if efficiency <= 0.0 or efficiency > 1.0:
        raise ValueError("target_signal_efficiency must lie in (0,1]")
    count = int(replicates)
    if count < 1000:
        raise ValueError("paired binary bootstrap requires at least 1000 replicates")
    qcd_index = names.index("QCD")
    signal_index = names.index(str(signal_name))
    qcd_rows = np.flatnonzero(truth == qcd_index)
    signal_rows = np.flatnonzero(truth == signal_index)
    if not qcd_rows.size or not signal_rows.size:
        raise ValueError("paired binary bootstrap requires QCD and signal support")
    scores_a = a[:, signal_index] - a[:, qcd_index]
    scores_b = b[:, signal_index] - b[:, qcd_index]

    def fpr(scores: np.ndarray, sampled_qcd: np.ndarray, sampled_signal: np.ndarray) -> tuple[float, float]:
        threshold = _threshold_for_efficiency(scores[sampled_signal], efficiency)
        false_positives = int(np.count_nonzero(scores[sampled_qcd] >= threshold))
        raw = false_positives / float(len(sampled_qcd))
        smoothed = (false_positives + 0.5) / float(len(sampled_qcd) + 1)
        return raw, smoothed

    rng = np.random.default_rng(int(seed))
    fpr_deltas = np.empty(count, dtype=np.float64)
    log_fpr_deltas = np.empty(count, dtype=np.float64)
    rejection_ratio = np.empty(count, dtype=np.float64)
    sample_digest = hashlib.sha256()
    for replicate in range(count):
        sampled_qcd = rng.choice(qcd_rows, size=qcd_rows.size, replace=True)
        sampled_signal = rng.choice(signal_rows, size=signal_rows.size, replace=True)
        sample_digest.update(np.asarray(sampled_qcd, dtype=np.int64).tobytes())
        sample_digest.update(np.asarray(sampled_signal, dtype=np.int64).tobytes())
        raw_a, smooth_a = fpr(scores_a, sampled_qcd, sampled_signal)
        raw_b, smooth_b = fpr(scores_b, sampled_qcd, sampled_signal)
        fpr_deltas[replicate] = raw_b - raw_a
        log_fpr_deltas[replicate] = np.log(smooth_b) - np.log(smooth_a)
        rejection_ratio[replicate] = smooth_a / smooth_b
    original_qcd = qcd_rows
    original_signal = signal_rows
    raw_a, smooth_a = fpr(scores_a, original_qcd, original_signal)
    raw_b, smooth_b = fpr(scores_b, original_qcd, original_signal)

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "contract": LOCAL_RESIDUAL_FIELD_BOOTSTRAP_CONTRACT,
        "comparison": "paired_binary_projection",
        "signal_name": str(signal_name),
        "negative_name": "QCD",
        "target_signal_efficiency": efficiency,
        "replicates": count,
        "seed": int(seed),
        "sampled_index_hash": sample_digest.hexdigest(),
        "stratified_by_class": True,
        "qcd_support": int(qcd_rows.size),
        "signal_support": int(signal_rows.size),
        "false_positive_rate_a": raw_a,
        "false_positive_rate_b": raw_b,
        "false_positive_rate_delta_b_minus_a": {
            "estimate": raw_b - raw_a,
            "interval_95": interval(fpr_deltas),
        },
        "log_smoothed_fpr_delta_b_minus_a": {
            "estimate": float(np.log(smooth_b) - np.log(smooth_a)),
            "interval_95": interval(log_fpr_deltas),
        },
        "smoothed_rejection_ratio_b_over_a": {
            "estimate": smooth_a / smooth_b,
            "interval_95": interval(rejection_ratio),
        },
    }


__all__ = [
    "LOCAL_RESIDUAL_FIELD_MULTICLASS_METRICS_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_BINARY_PROJECTION_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_COMPLEMENTARITY_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_BOOTSTRAP_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_BINARY_SIGNAL_EFFICIENCIES",
    "local_residual_field_multiclass_metrics",
    "local_residual_field_binary_projection_metrics",
    "freeze_binary_projection_thresholds",
    "local_residual_field_complementarity_metrics",
    "paired_multiclass_bootstrap",
    "paired_binary_projection_bootstrap",
]

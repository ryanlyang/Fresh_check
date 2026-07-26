"""Locked Step-8 classification, calibration, and paired statistics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import require_sha256, with_content_hash


PARTICLE_VIEW_PAIRED_STATISTICS_CONTRACT = "particle_view_paired_statistics_v1"
PARTICLE_VIEW_CLASSIFICATION_METRICS_CONTRACT = (
    "particle_view_classification_metrics_v1"
)
PAIRED_BOOTSTRAP_SEED = 917_301
PAIRED_BOOTSTRAP_REPLICATES = 10_000
ECE_BINS = 15


def _arrays(
    logits: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or target.ndim != 1 or scores.shape[0] != target.size:
        raise ValueError("logits/labels shape mismatch")
    if scores.shape[1] < 2 or target.size == 0:
        raise ValueError("classification metrics require events and >=2 classes")
    if not np.isfinite(scores).all():
        raise FloatingPointError("logits contain non-finite values")
    if target.min() < 0 or target.max() >= scores.shape[1]:
        raise ValueError("labels are outside the class order")
    return scores, target


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = ECE_BINS
) -> float:
    """Top-label ECE with the plan's 15 equal-width bins."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    # floor maps confidence 1.0 to bins, so clamp it into the closed final bin.
    indices = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    result = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return result


def multiclass_brier_score(
    probabilities: np.ndarray, labels: np.ndarray
) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(labels.size), labels] = 1.0
    return float(np.square(probabilities - one_hot).sum(axis=1).mean())


def classification_metrics(
    logits: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[int],
    *,
    split: str,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    scores, target = _arrays(logits, labels)
    probabilities = probabilities_from_logits(scores)
    prediction = probabilities.argmax(axis=1)
    count = scores.shape[1]
    if class_names is None:
        class_names = tuple(str(index) for index in range(count))
    if len(class_names) != count or len(set(class_names)) != count:
        raise ValueError("class_names do not match the logit width")
    confusion = np.zeros((count, count), dtype=np.int64)
    np.add.at(confusion, (target, prediction), 1)
    per_class = []
    for index, name in enumerate(class_names):
        denominator = int((target == index).sum())
        per_class.append(
            {
                "class_name": str(name),
                "support": denominator,
                "accuracy": (
                    float(((prediction == index) & (target == index)).sum())
                    / denominator
                    if denominator
                    else None
                ),
            }
        )
    cross_entropy = float(
        -np.log(np.clip(probabilities[np.arange(target.size), target], 1.0e-300, 1.0)).mean()
    )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_CLASSIFICATION_METRICS_CONTRACT,
            "split": split,
            "event_count": int(target.size),
            "class_names": list(class_names),
            "accuracy": float((prediction == target).mean()),
            "cross_entropy": cross_entropy,
            "macro_per_class_accuracy": float(
                np.mean(
                    [
                        row["accuracy"]
                        for row in per_class
                        if row["accuracy"] is not None
                    ]
                )
            ),
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
            "ece_top_label_15_equal_width": expected_calibration_error(
                probabilities, target
            ),
            "multiclass_brier": multiclass_brier_score(probabilities, target),
        }
    )


def exact_two_sided_mcnemar_pvalue(
    baseline_correct: np.ndarray | Sequence[bool],
    candidate_correct: np.ndarray | Sequence[bool],
) -> tuple[int, int, float]:
    baseline = np.asarray(baseline_correct, dtype=bool)
    candidate = np.asarray(candidate_correct, dtype=bool)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired correctness arrays must be same-shape vectors")
    baseline_only = int(np.sum(baseline & ~candidate))
    candidate_only = int(np.sum(~baseline & candidate))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return baseline_only, candidate_only, 1.0
    tail = min(baseline_only, candidate_only)
    logs = np.asarray(
        [
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
            - discordant * math.log(2.0)
            for k in range(tail + 1)
        ],
        dtype=np.float64,
    )
    maximum = float(logs.max())
    probability = math.exp(maximum) * float(np.exp(logs - maximum).sum())
    return baseline_only, candidate_only, min(1.0, 2.0 * probability)


def paired_stratified_bootstrap_accuracy_gain(
    baseline_correct: np.ndarray | Sequence[bool],
    candidate_correct: np.ndarray | Sequence[bool],
    labels: np.ndarray | Sequence[int],
    *,
    replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_correct, dtype=np.float64)
    candidate = np.asarray(candidate_correct, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if (
        baseline.ndim != 1
        or baseline.shape != candidate.shape
        or baseline.shape != target.shape
        or baseline.size == 0
    ):
        raise ValueError("paired bootstrap arrays must be nonempty aligned vectors")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    classes = [np.flatnonzero(target == value) for value in np.unique(target)]
    if any(indices.size == 0 for indices in classes):
        raise ValueError("bootstrap class strata cannot be empty")
    rng = np.random.default_rng(seed)
    gains = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        total = 0.0
        number = 0
        for indices in classes:
            sampled = indices[rng.integers(0, indices.size, size=indices.size)]
            total += float((candidate[sampled] - baseline[sampled]).sum())
            number += int(indices.size)
        gains[replicate] = total / number
    lower, upper = np.percentile(gains, [2.5, 97.5])
    return {
        "seed": int(seed),
        "replicates": int(replicates),
        "stratified_by_class": True,
        "observed_accuracy_gain": float((candidate - baseline).mean()),
        "mean_bootstrap_gain": float(gains.mean()),
        "ci95_percentile": [float(lower), float(upper)],
    }


def build_paired_statistics_report(
    *,
    baseline_logits: np.ndarray | Sequence[Sequence[float]],
    candidate_logits: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[int],
    split: str,
    baseline_artifact_sha256: str,
    candidate_artifact_sha256: str,
    split_sha256: str,
    event_identity_sha256: str,
    replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    baseline_scores, target = _arrays(baseline_logits, labels)
    candidate_scores, candidate_target = _arrays(candidate_logits, labels)
    if not np.array_equal(target, candidate_target) or baseline_scores.shape != candidate_scores.shape:
        raise ValueError("paired model outputs are not aligned")
    for name, value in (
        ("baseline_artifact_sha256", baseline_artifact_sha256),
        ("candidate_artifact_sha256", candidate_artifact_sha256),
        ("split_sha256", split_sha256),
        ("event_identity_sha256", event_identity_sha256),
    ):
        require_sha256(name, value)
    baseline_correct = baseline_scores.argmax(axis=1) == target
    candidate_correct = candidate_scores.argmax(axis=1) == target
    baseline_only, candidate_only, pvalue = exact_two_sided_mcnemar_pvalue(
        baseline_correct, candidate_correct
    )
    bootstrap = paired_stratified_bootstrap_accuracy_gain(
        baseline_correct,
        candidate_correct,
        target,
        replicates=replicates,
    )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_PAIRED_STATISTICS_CONTRACT,
            "split": split,
            "split_sha256": split_sha256,
            "event_identity_sha256": event_identity_sha256,
            "baseline_artifact_sha256": baseline_artifact_sha256,
            "candidate_artifact_sha256": candidate_artifact_sha256,
            "event_count": int(target.size),
            "wins": candidate_only,
            "losses": baseline_only,
            "ties": int(target.size - candidate_only - baseline_only),
            "mcnemar": {
                "baseline_only_correct": baseline_only,
                "candidate_only_correct": candidate_only,
                "discordant": baseline_only + candidate_only,
                "exact_two_sided_binomial_p": pvalue,
            },
            "paired_bootstrap": bootstrap,
        }
    )


def aggregate_calibration(
    replica_metrics: Sequence[Mapping[str, Any]],
    *,
    accuracy_tolerance: float = 0.0005,
) -> dict[str, Any]:
    """Apply the locked comparable-accuracy/ECE/Brier precedence."""

    if len(replica_metrics) != 3:
        raise ValueError("calibration aggregation requires three seeds")
    accuracy = np.asarray([row["accuracy"] for row in replica_metrics], dtype=float)
    ece = np.asarray(
        [row["ece_top_label_15_equal_width"] for row in replica_metrics], dtype=float
    )
    brier = np.asarray([row["multiclass_brier"] for row in replica_metrics], dtype=float)
    if not np.isfinite(np.concatenate((accuracy, ece, brier))).all():
        raise FloatingPointError("calibration aggregation contains non-finite values")
    return {
        "mean_accuracy": float(accuracy.mean()),
        "sample_accuracy_std": float(accuracy.std(ddof=1)),
        "minimum_accuracy": float(accuracy.min()),
        "mean_ece": float(ece.mean()),
        "mean_brier": float(brier.mean()),
        "comparable_accuracy_absolute_tolerance": float(accuracy_tolerance),
        "calibration_precedence": ["lower_mean_ece", "lower_mean_brier"],
        "seed_variability_is_diagnostic": True,
    }


__all__ = [
    "ECE_BINS",
    "PAIRED_BOOTSTRAP_REPLICATES",
    "PAIRED_BOOTSTRAP_SEED",
    "PARTICLE_VIEW_CLASSIFICATION_METRICS_CONTRACT",
    "PARTICLE_VIEW_PAIRED_STATISTICS_CONTRACT",
    "aggregate_calibration",
    "build_paired_statistics_report",
    "classification_metrics",
    "exact_two_sided_mcnemar_pvalue",
    "expected_calibration_error",
    "multiclass_brier_score",
    "paired_stratified_bootstrap_accuracy_gain",
    "probabilities_from_logits",
]

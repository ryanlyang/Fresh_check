"""Step 4 evaluation and deterministic consumer-selection contracts.

The functions in this module deliberately operate on logits rather than on a
particular ParT implementation.  A production runner therefore performs every
field-condition forward with one loaded checkpoint, while the scientific
selection and provenance rules remain small, deterministic, and CPU-testable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bridge import (
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    BRIDGE_CONTROLS,
    PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT,
    RESPONSE_RHOS,
)
from .bridge_campaign import PAIRED_SEED_IDS
from .bridge_consumer import T10_ALL50_CLEAN, T10_CLEAN, T10_ROBUST
from .bridge_contracts import (
    canonical_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


PREDICTION_ANCHORED_CLASSIFICATION_METRICS_CONTRACT = (
    "prediction_anchored_classification_metrics_v1"
)
PREDICTION_ANCHORED_CONSUMER_REPLICA_EVALUATION_CONTRACT = (
    "prediction_anchored_consumer_replica_evaluation_v1"
)
PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT = (
    "prediction_anchored_consumer_selection_aggregate_v1"
)
PREDICTION_ANCHORED_CONSUMER_PRECONFIRMATION_CONTRACT = (
    "prediction_anchored_consumer_preconfirmation_v1"
)
PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT = "selected_bridge_consumer_v2"
PREDICTION_ANCHORED_STOPPED_CAMPAIGN_CONTRACT = (
    "prediction_anchored_stopped_campaign_v1"
)
PREDICTION_ANCHORED_TEACHER_BINDING_CONTRACT = "teacher_binding_v1"

PRIMARY_TEACHER_NAMESPACE = "physical45_selected_bridge_teacher"
ALL50_TEACHER_NAMESPACE = "all50_selected_bridge_teacher"
ALTERNATE_TEACHER_NAMESPACE = "physical45_alternate_bridge_teacher"
N3_F0_TEACHER_NAMESPACE = "physical45_selected_teacher_on_f0_control"
TEACHER_LOGIT_CACHE_SCHEMA = "prediction_anchored_teacher_logit_cache_v1"

SELECTION_RECIPE_IDS = (T10_CLEAN, T10_ROBUST)
REQUIRED_RESPONSE_KEYS = tuple(f"rho_{rho}" for rho in RESPONSE_RHOS)
REQUIRED_DIAGNOSTIC_KEYS = (
    "oracle_physical45",
    "oracle_all50",
    "reliability5_only",
    "zero_field_consumer_diagnostic",
)
DEFAULT_BACKGROUND_REJECTION_EFFICIENCIES = (0.30, 0.50)


def _sha256(value: Any, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _all_numeric_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, (bool, str)) or value is None:
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return True


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def _weighted_auc(scores: np.ndarray, positive: np.ndarray, weights: np.ndarray) -> float | None:
    positive_weight = float(weights[positive].sum())
    negative_weight = float(weights[~positive].sum())
    if positive_weight <= 0 or negative_weight <= 0:
        return None
    # Group equal scores so the trapezoid is invariant to their input order.
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_positive = positive[order]
    sorted_weights = weights[order]
    tp = [0.0]
    fp = [0.0]
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and sorted_scores[stop] == sorted_scores[cursor]:
            stop += 1
        group_positive = sorted_positive[cursor:stop]
        group_weights = sorted_weights[cursor:stop]
        tp.append(tp[-1] + float(group_weights[group_positive].sum()))
        fp.append(fp[-1] + float(group_weights[~group_positive].sum()))
        cursor = stop
    tpr = np.asarray(tp, dtype=np.float64) / positive_weight
    fpr = np.asarray(fp, dtype=np.float64) / negative_weight
    # Keep compatibility with both NumPy 1.x and 2.x without relying on the
    # renamed trapz/trapezoid public symbol.
    widths = fpr[1:] - fpr[:-1]
    return float(np.sum(widths * (tpr[1:] + tpr[:-1]) * 0.5))


def _background_rejection(
    scores: np.ndarray,
    positive: np.ndarray,
    weights: np.ndarray,
    target_efficiency: float,
) -> float | None:
    positive_weight = float(weights[positive].sum())
    negative_weight = float(weights[~positive].sum())
    if positive_weight <= 0 or negative_weight <= 0:
        return None
    positive_scores = scores[positive]
    positive_weights = weights[positive]
    order = np.argsort(-positive_scores, kind="mergesort")
    cumulative = np.cumsum(positive_weights[order]) / positive_weight
    index = min(int(np.searchsorted(cumulative, target_efficiency, side="left")), len(order) - 1)
    threshold = float(positive_scores[order[index]])
    background_efficiency = float(weights[(~positive) & (scores >= threshold)].sum()) / negative_weight
    # JSON contracts forbid infinity.  Null means no background passed.
    return None if background_efficiency <= 0 else float(1.0 / background_efficiency)


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    class_order: Sequence[str],
    sample_weights: np.ndarray | None = None,
    ece_bins: int = 15,
    rejection_signal_efficiencies: Sequence[float] = DEFAULT_BACKGROUND_REJECTION_EFFICIENCIES,
) -> dict[str, Any]:
    """Compute the locked ten-class discrimination/calibration summary."""

    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    names = tuple(str(value) for value in class_order)
    if values.ndim != 2 or truth.shape != (values.shape[0],):
        raise ValueError("logits/labels must have shapes [N,C] and [N]")
    if values.shape[1] != len(names) or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("class_order does not match unique logit columns")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("classification metrics require finite nonempty logits")
    if np.any(truth < 0) or np.any(truth >= len(names)):
        raise ValueError("label is outside class_order")
    weights = (
        np.ones(values.shape[0], dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if weights.shape != truth.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample weights must be finite nonnegative [N]")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("sample weights have zero total")
    probabilities = _softmax(values)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == truth
    accuracy = float(weights[correct].sum() / total_weight)
    chosen = np.maximum(probabilities[np.arange(len(truth)), truth], np.finfo(np.float64).tiny)
    cross_entropy = float(np.sum(weights * -np.log(chosen)) / total_weight)
    confusion = np.zeros((len(names), len(names)), dtype=np.float64)
    np.add.at(confusion, (truth, predictions), weights)
    class_efficiency: list[float | None] = []
    for class_index in range(len(names)):
        denominator = float(confusion[class_index].sum())
        class_efficiency.append(
            None if denominator <= 0 else float(confusion[class_index, class_index] / denominator)
        )
    finite_efficiencies = [value for value in class_efficiency if value is not None]
    macro = float(np.mean(finite_efficiencies)) if finite_efficiencies else None

    confidence = probabilities.max(axis=1)
    boundaries = np.linspace(0.0, 1.0, int(ece_bins) + 1)
    ece = 0.0
    ece_rows: list[dict[str, Any]] = []
    for index in range(int(ece_bins)):
        in_bin = (confidence >= boundaries[index]) & (
            confidence <= boundaries[index + 1]
            if index == int(ece_bins) - 1
            else confidence < boundaries[index + 1]
        )
        bin_weight = float(weights[in_bin].sum())
        if bin_weight <= 0:
            continue
        bin_accuracy = float(weights[in_bin & correct].sum() / bin_weight)
        bin_confidence = float(np.sum(weights[in_bin] * confidence[in_bin]) / bin_weight)
        ece += (bin_weight / total_weight) * abs(bin_accuracy - bin_confidence)
        ece_rows.append(
            {
                "low": float(boundaries[index]),
                "high": float(boundaries[index + 1]),
                "weight": bin_weight,
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
            }
        )
    targets = np.eye(len(names), dtype=np.float64)[truth]
    brier = float(np.sum(weights[:, None] * np.square(probabilities - targets)) / total_weight)
    auc: dict[str, float | None] = {}
    rejection: dict[str, dict[str, float | None]] = {}
    for class_index, name in enumerate(names):
        positive = truth == class_index
        scores = probabilities[:, class_index]
        auc[name] = _weighted_auc(scores, positive, weights)
        rejection[name] = {
            f"signal_efficiency_{float(efficiency):.2f}": _background_rejection(
                scores, positive, weights, float(efficiency)
            )
            for efficiency in rejection_signal_efficiencies
        }
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CLASSIFICATION_METRICS_CONTRACT,
            "event_count": int(len(truth)),
            "weight_sum": total_weight,
            "class_order": list(names),
            "accuracy": accuracy,
            "cross_entropy": cross_entropy,
            "macro_per_class_accuracy": macro,
            "confusion_matrix": confusion.tolist(),
            "classwise_efficiency": dict(zip(names, class_efficiency)),
            "one_vs_rest_auc": auc,
            "background_rejection": rejection,
            "ece_bins": int(ece_bins),
            "expected_calibration_error": float(ece),
            "brier_score": brier,
            "ece_rows": ece_rows,
        }
    )


def slice_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    class_order: Sequence[str],
    slices: Mapping[str, Sequence[Any]],
    sample_weights: np.ndarray | None = None,
    minimum_events: int = 1,
) -> dict[str, Any]:
    """Report finite per-slice accuracy/CE without using slices for selection."""

    values = np.asarray(logits)
    truth = np.asarray(labels)
    weights = None if sample_weights is None else np.asarray(sample_weights)
    output: dict[str, Any] = {}
    for slice_name, raw_groups in sorted(slices.items()):
        groups = np.asarray(raw_groups)
        if groups.shape != truth.shape:
            raise ValueError(f"slice {slice_name!r} does not align with labels")
        rows: dict[str, Any] = {}
        for group in sorted(set(str(value) for value in groups.tolist())):
            selected = np.asarray([str(value) == group for value in groups.tolist()], dtype=bool)
            count = int(selected.sum())
            if count < int(minimum_events):
                rows[group] = {"event_count": count, "status": "INSUFFICIENT_EVENTS"}
                continue
            metrics = classification_metrics(
                values[selected],
                truth[selected],
                class_order=class_order,
                sample_weights=None if weights is None else weights[selected],
            )
            rows[group] = {
                "event_count": count,
                "status": "REPORTED_DIAGNOSTIC_ONLY",
                "accuracy": metrics["accuracy"],
                "cross_entropy": metrics["cross_entropy"],
                "macro_per_class_accuracy": metrics["macro_per_class_accuracy"],
                "expected_calibration_error": metrics["expected_calibration_error"],
                "brier_score": metrics["brier_score"],
            }
        output[str(slice_name)] = rows
    return output


def _event_manifest_hash(event_ids: Sequence[str]) -> str:
    identities = [str(value) for value in event_ids]
    if len(identities) != len(set(identities)):
        raise ValueError("evaluation event identities contain duplicates")
    return canonical_sha256(identities)


def paired_bootstrap_difference(
    left_correct: np.ndarray,
    right_correct: np.ndarray,
    *,
    event_ids: Sequence[str],
    sample_weights: np.ndarray | None = None,
    seed: int = 4_180_101,
    resamples: int = 4_000,
    confidence: float = 0.90,
) -> dict[str, Any]:
    left = np.asarray(left_correct, dtype=np.float64)
    right = np.asarray(right_correct, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.shape[0] != len(event_ids):
        raise ValueError("paired bootstrap arrays/event IDs do not align")
    if int(resamples) <= 0 or not 0 < float(confidence) < 1:
        raise ValueError("invalid bootstrap configuration")
    weights = (
        np.ones(left.shape[0], dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if weights.shape != left.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("paired bootstrap sample weights are invalid")
    if left.size == 0 or weights.sum() <= 0:
        raise ValueError("paired bootstrap is empty")
    event_hash = _event_manifest_hash(event_ids)
    rng = np.random.default_rng(int(seed))
    differences = left - right
    values = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        selected = rng.integers(0, len(differences), size=len(differences))
        selected_weights = weights[selected]
        values[index] = float(np.sum(selected_weights * differences[selected]) / selected_weights.sum())
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "seed": int(seed),
        "resamples": int(resamples),
        "confidence": float(confidence),
        "event_manifest_sha256": event_hash,
        "point_difference": float(np.sum(weights * differences) / weights.sum()),
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
        "paired_by_event": True,
    }


@dataclass(frozen=True)
class ConsumerReplicaEvaluation:
    """Hashed metrics plus ephemeral paired arrays for aggregate bootstrap."""

    artifact: Mapping[str, Any]
    bridge_correct: np.ndarray = field(repr=False)
    f0_correct: np.ndarray = field(repr=False)
    event_ids: tuple[str, ...] = field(repr=False)
    sample_weights: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        validate_content_hash(
            self.artifact,
            expected_contract=PREDICTION_ANCHORED_CONSUMER_REPLICA_EVALUATION_CONTRACT,
        )
        if self.bridge_correct.shape != self.f0_correct.shape or self.bridge_correct.ndim != 1:
            raise ValueError("replica paired arrays do not align")
        if len(self.event_ids) != self.bridge_correct.shape[0] or self.sample_weights.shape != self.bridge_correct.shape:
            raise ValueError("replica event IDs/weights do not align")
        if self.artifact["event_manifest_sha256"] != _event_manifest_hash(self.event_ids):
            raise ValueError("replica paired arrays changed after metric construction")


def build_consumer_replica_evaluation(
    *,
    run_id: str,
    seed_id: int,
    checkpoint_sha256: str,
    recipe_sha256: str,
    split_sha256: str,
    ram_audit_sha256: str,
    class_order: Sequence[str],
    labels: np.ndarray,
    event_ids: Sequence[str],
    logits_by_condition: Mapping[str, np.ndarray],
    matched_compute_f0_accuracy: float,
    sample_weights: np.ndarray | None = None,
    slices: Mapping[str, Sequence[Any]] | None = None,
    bootstrap_seed: int = 4_180_101,
    bootstrap_resamples: int = 4_000,
    provenance_valid: bool = True,
) -> ConsumerReplicaEvaluation:
    """Evaluate one selected-on-stop checkpoint once on model_val_select."""

    if str(run_id) not in SELECTION_RECIPE_IDS and str(run_id) != T10_ALL50_CLEAN:
        raise ValueError("unknown bridge-consumer recipe")
    if int(seed_id) not in PAIRED_SEED_IDS:
        raise ValueError("consumer evaluation seed is not paired")
    checkpoint = _sha256(checkpoint_sha256, name="checkpoint_sha256")
    recipe = _sha256(recipe_sha256, name="recipe_sha256")
    split = _sha256(split_sha256, name="split_sha256")
    ram_audit = _sha256(ram_audit_sha256, name="ram_audit_sha256")
    required = {"f0", *REQUIRED_RESPONSE_KEYS, *BRIDGE_CONTROLS, *REQUIRED_DIAGNOSTIC_KEYS}
    if set(logits_by_condition) != required:
        missing = sorted(required - set(logits_by_condition))
        extra = sorted(set(logits_by_condition) - required)
        raise ValueError(f"same-consumer evaluation condition mismatch; missing={missing}, extra={extra}")
    truth = np.asarray(labels, dtype=np.int64)
    identities = tuple(str(value) for value in event_ids)
    weights = (
        np.ones(len(truth), dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    metrics_by_condition: dict[str, Any] = {}
    for condition, raw_logits in sorted(logits_by_condition.items()):
        metrics_by_condition[condition] = classification_metrics(
            raw_logits,
            truth,
            class_order=class_order,
            sample_weights=weights,
        )
    if not np.allclose(
        np.asarray(logits_by_condition["f0"], dtype=np.float32),
        np.asarray(logits_by_condition["rho_0.000"], dtype=np.float32),
        atol=1.0e-6,
        rtol=1.0e-5,
    ):
        raise AssertionError("rho=0 response logits do not reproduce the same consumer on f0")
    bridge_logits = np.asarray(logits_by_condition["rho_0.100"])
    f0_logits = np.asarray(logits_by_condition["f0"])
    bridge_correct = (np.argmax(bridge_logits, axis=1) == truth).astype(np.float64)
    f0_correct = (np.argmax(f0_logits, axis=1) == truth).astype(np.float64)
    bootstrap = paired_bootstrap_difference(
        bridge_correct,
        f0_correct,
        event_ids=identities,
        sample_weights=weights,
        seed=int(bootstrap_seed) + int(seed_id),
        resamples=int(bootstrap_resamples),
    )
    response = [
        {
            "rho": float(rho),
            "accuracy": metrics_by_condition[f"rho_{rho}"]["accuracy"],
            "cross_entropy": metrics_by_condition[f"rho_{rho}"]["cross_entropy"],
            "checkpoint_sha256": checkpoint,
        }
        for rho in RESPONSE_RHOS
    ]
    f0_accuracy = float(metrics_by_condition["f0"]["accuracy"])
    bridge_accuracy = float(metrics_by_condition["rho_0.100"]["accuracy"])
    delta_same = bridge_accuracy - f0_accuracy
    controls = {
        name: {
            "accuracy": metrics_by_condition[name]["accuracy"],
            "cross_entropy": metrics_by_condition[name]["cross_entropy"],
            "gain_over_f0": float(metrics_by_condition[name]["accuracy"] - f0_accuracy),
            "checkpoint_sha256": checkpoint,
        }
        for name in BRIDGE_CONTROLS
    }
    artifact = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CONSUMER_REPLICA_EVALUATION_CONTRACT,
            "run_id": str(run_id),
            "seed_id": int(seed_id),
            "checkpoint_sha256": checkpoint,
            "bridge_recipe_sha256": recipe,
            "split_sha256": split,
            "ram_audit_sha256": ram_audit,
            "evaluation_split": "model_val_select",
            "selected_checkpoint_source": "model_val_stop_only",
            "same_checkpoint_all_conditions": True,
            "event_manifest_sha256": _event_manifest_hash(identities),
            "event_count": len(identities),
            "class_order": list(class_order),
            "f0": metrics_by_condition["f0"],
            "bridge_0p10": metrics_by_condition["rho_0.100"],
            "delta_same": float(delta_same),
            "matched_compute_f0_accuracy": _finite(
                matched_compute_f0_accuracy, name="matched_compute_f0_accuracy"
            ),
            "matched_compute_gain": float(bridge_accuracy - float(matched_compute_f0_accuracy)),
            "paired_bootstrap_90": bootstrap,
            "response_curve": response,
            "negative_controls": controls,
            "diagnostics": {
                name: metrics_by_condition[name] for name in REQUIRED_DIAGNOSTIC_KEYS
            },
            "slice_metrics": slice_classification_metrics(
                bridge_logits,
                truth,
                class_order=class_order,
                slices=slices or {},
                sample_weights=weights,
            ),
            "provenance_valid": bool(provenance_valid),
            "persistent_per_event_outcomes_written": False,
        }
    )
    return ConsumerReplicaEvaluation(
        artifact=artifact,
        bridge_correct=bridge_correct,
        f0_correct=f0_correct,
        event_ids=identities,
        sample_weights=weights,
    )


def evaluate_bound_consumer_conditions(
    *,
    run_id: str,
    seed_id: int,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    recipe_sha256: str,
    split_sha256: str,
    ram_audit_sha256: str,
    class_order: Sequence[str],
    batches: Sequence[Mapping[str, Any]],
    forward_fn: Any,
    matched_compute_f0_accuracy: float,
    bootstrap_seed: int = 4_180_101,
    bootstrap_resamples: int = 4_000,
    provenance_valid: bool = True,
) -> ConsumerReplicaEvaluation:
    """Run every response/control condition through one bound checkpoint.

    ``forward_fn(batch, condition)`` must expose ``checkpoint_sha256``.  This
    small adapter contract prevents a caller from accidentally evaluating the
    target side with one consumer and controls with another.
    """

    expected_checkpoint = _sha256(checkpoint_sha256, name="checkpoint_sha256")
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(f"consumer checkpoint is absent or unsafe: {checkpoint}")
    if sha256_file(checkpoint) != expected_checkpoint:
        raise ValueError("consumer evaluation checkpoint bytes changed")
    if getattr(forward_fn, "checkpoint_sha256", None) != expected_checkpoint:
        raise ValueError("consumer forward callback is not bound to the checkpoint hash")
    required_conditions = (
        "f0",
        *REQUIRED_RESPONSE_KEYS,
        *BRIDGE_CONTROLS,
        *REQUIRED_DIAGNOSTIC_KEYS,
    )
    logit_parts: dict[str, list[np.ndarray]] = {
        condition: [] for condition in required_conditions
    }
    label_parts: list[np.ndarray] = []
    event_ids: list[str] = []
    weight_parts: list[np.ndarray] = []
    slice_parts: dict[str, list[np.ndarray]] = {}
    try:
        import torch

        no_grad = torch.no_grad
    except ImportError:  # pragma: no cover - production has torch
        from contextlib import nullcontext

        no_grad = nullcontext
    for batch in batches:
        if not {"labels", "event_ids"}.issubset(batch):
            raise ValueError("same-consumer evaluation batch lacks labels/event_ids")
        labels = np.asarray(batch["labels"], dtype=np.int64)
        ids = [str(value) for value in batch["event_ids"]]
        if labels.ndim != 1 or len(ids) != len(labels):
            raise ValueError("same-consumer evaluation batch identities do not align")
        weights = np.asarray(
            batch.get("sample_weights", np.ones(len(labels), dtype=np.float64)),
            dtype=np.float64,
        )
        if weights.shape != labels.shape:
            raise ValueError("same-consumer evaluation batch weights do not align")
        with no_grad():
            for condition in required_conditions:
                raw_output = forward_fn(batch, condition)
                output = getattr(raw_output, "logits", raw_output)
                try:
                    if isinstance(output, torch.Tensor):
                        output = output.detach().float().cpu().numpy()
                except NameError:  # pragma: no cover
                    pass
                logits = np.asarray(output, dtype=np.float32)
                if logits.shape != (len(labels), len(class_order)):
                    raise ValueError(
                        f"consumer condition {condition!r} logits do not align with its batch"
                    )
                logit_parts[condition].append(logits)
        raw_slices = batch.get("slices", {})
        if not isinstance(raw_slices, Mapping):
            raise ValueError("same-consumer evaluation slices must be a mapping")
        if slice_parts and set(raw_slices) != set(slice_parts):
            raise ValueError("same-consumer evaluation slice inventory changed between batches")
        for name, groups in raw_slices.items():
            values = np.asarray(groups)
            if values.shape != labels.shape:
                raise ValueError(f"evaluation slice {name!r} does not align with its batch")
            slice_parts.setdefault(str(name), []).append(values)
        label_parts.append(labels)
        event_ids.extend(ids)
        weight_parts.append(weights)
    if not label_parts:
        raise ValueError("same-consumer evaluation received no batches")
    return build_consumer_replica_evaluation(
        run_id=run_id,
        seed_id=seed_id,
        checkpoint_sha256=expected_checkpoint,
        recipe_sha256=recipe_sha256,
        split_sha256=split_sha256,
        ram_audit_sha256=ram_audit_sha256,
        class_order=class_order,
        labels=np.concatenate(label_parts),
        event_ids=event_ids,
        logits_by_condition={
            condition: np.concatenate(parts, axis=0)
            for condition, parts in logit_parts.items()
        },
        matched_compute_f0_accuracy=matched_compute_f0_accuracy,
        sample_weights=np.concatenate(weight_parts),
        slices={
            name: np.concatenate(parts, axis=0) for name, parts in slice_parts.items()
        },
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        provenance_valid=provenance_valid,
    )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _sample_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _aggregate_paired_bootstrap(
    evaluations: Sequence[ConsumerReplicaEvaluation],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(resamples), dtype=np.float64)
    for draw in range(int(resamples)):
        seed_deltas: list[float] = []
        for evaluation in evaluations:
            n_events = len(evaluation.event_ids)
            selected = rng.integers(0, n_events, size=n_events)
            weights = evaluation.sample_weights[selected]
            differences = evaluation.bridge_correct[selected] - evaluation.f0_correct[selected]
            seed_deltas.append(float(np.sum(weights * differences) / weights.sum()))
        draws[draw] = float(np.mean(seed_deltas))
    return {
        "seed": int(seed),
        "resamples": int(resamples),
        "confidence": 0.90,
        "paired_by_event_and_seed": True,
        "event_manifest_sha256_by_seed": {
            str(item.artifact["seed_id"]): item.artifact["event_manifest_sha256"]
            for item in evaluations
        },
        "lower": float(np.quantile(draws, 0.05)),
        "upper": float(np.quantile(draws, 0.95)),
    }


def aggregate_consumer_evaluations(
    evaluations: Sequence[ConsumerReplicaEvaluation],
    *,
    bootstrap_seed: int = 5_180_101,
    bootstrap_resamples: int = 4_000,
) -> dict[str, Any]:
    """Aggregate three paired replicas and apply all Section 18.1 rules."""

    if len(evaluations) != 3:
        raise ValueError("consumer aggregate requires exactly three evaluations")
    artifacts = [dict(item.artifact) for item in evaluations]
    run_ids = {str(item["run_id"]) for item in artifacts}
    seeds = {int(item["seed_id"]) for item in artifacts}
    if len(run_ids) != 1 or seeds != set(PAIRED_SEED_IDS):
        raise ValueError("aggregate requires one recipe and seeds 101/202/303")
    checkpoint_hashes = [str(item["checkpoint_sha256"]) for item in artifacts]
    if len(set(checkpoint_hashes)) != 3:
        raise ValueError("each paired replica must bind its own exact checkpoint")
    for artifact in artifacts:
        validate_content_hash(
            artifact,
            expected_contract=PREDICTION_ANCHORED_CONSUMER_REPLICA_EVALUATION_CONTRACT,
        )
    deltas = [float(item["delta_same"]) for item in artifacts]
    bridge_accuracy = [float(item["bridge_0p10"]["accuracy"]) for item in artifacts]
    f0_accuracy = [float(item["f0"]["accuracy"]) for item in artifacts]
    matched = [float(item["matched_compute_f0_accuracy"]) for item in artifacts]
    bridge_ce = [float(item["bridge_0p10"]["cross_entropy"]) for item in artifacts]
    ece = [float(item["bridge_0p10"]["expected_calibration_error"]) for item in artifacts]
    brier = [float(item["bridge_0p10"]["brier_score"]) for item in artifacts]
    ordering = sorted(
        artifacts,
        key=lambda item: (
            float(item["bridge_0p10"]["accuracy"]),
            float(item["delta_same"]),
            -float(item["bridge_0p10"]["cross_entropy"]),
            int(item["seed_id"]),
        ),
    )
    median = ordering[1]
    mean_delta = _mean(deltas)
    mean_response = []
    for index, rho in enumerate(RESPONSE_RHOS):
        mean_response.append(
            {
                "rho": float(rho),
                "accuracy": _mean(
                    [float(item["response_curve"][index]["accuracy"]) for item in artifacts]
                ),
            }
        )
    adjacent_decreases = sum(
        mean_response[index + 1]["accuracy"] < mean_response[index]["accuracy"] - 0.0005
        for index in range(4)
    )
    endpoint_within_maximum = (
        mean_response[-1]["accuracy"] >= max(row["accuracy"] for row in mean_response) - 0.0005
    )
    control_gains = {
        control: _mean(
            [float(item["negative_controls"][control]["gain_over_f0"]) for item in artifacts]
        )
        for control in BRIDGE_CONTROLS
    }
    control_limit = max(0.00025, 0.25 * mean_delta)
    bootstrap = _aggregate_paired_bootstrap(
        evaluations,
        seed=int(bootstrap_seed),
        resamples=int(bootstrap_resamples),
    )
    rules = {
        "rule_1_positive_mean_delta_same": mean_delta > 0,
        "rule_2_bootstrap_or_effect_size": bootstrap["lower"] > 0 or mean_delta >= 0.001,
        "rule_3_positive_matched_compute_gain": _mean(bridge_accuracy) - _mean(matched) > 0,
        "rule_4_replica_and_median_positive": sum(value > 0 for value in deltas) >= 2
        and float(median["delta_same"]) > 0,
        "rule_5_f0_degradation_at_most_0p002": _mean(f0_accuracy) >= _mean(matched) - 0.002,
        "rule_6_response_curve": adjacent_decreases <= 1 and endpoint_within_maximum,
        "rule_7_controls_below_limit": all(value <= control_limit for value in control_gains.values()),
        "rule_8_provenance_valid": all(bool(item["provenance_valid"]) for item in artifacts),
    }
    payload = {
        "contract": PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
        "run_id": next(iter(run_ids)),
        "paired_seed_ids": list(PAIRED_SEED_IDS),
        "mean_bridge_accuracy": _mean(bridge_accuracy),
        "sample_std_bridge_accuracy": _sample_std(bridge_accuracy),
        "mean_f0_accuracy": _mean(f0_accuracy),
        "mean_matched_compute_f0_accuracy": _mean(matched),
        "mean_matched_compute_gain": _mean(bridge_accuracy) - _mean(matched),
        "mean_delta_same": mean_delta,
        "sample_std_delta_same": _sample_std(deltas),
        "mean_bridge_cross_entropy": _mean(bridge_ce),
        "mean_expected_calibration_error": _mean(ece),
        "mean_brier_score": _mean(brier),
        "aggregate_paired_bootstrap_90": bootstrap,
        "mean_response_curve": mean_response,
        "adjacent_decreases_over_0p0005": int(adjacent_decreases),
        "rho_0p10_within_0p0005_of_max": bool(endpoint_within_maximum),
        "mean_negative_control_gains": control_gains,
        "negative_control_gain_limit": float(control_limit),
        "validity_rules": rules,
        "eligible": all(rules.values()),
        "ordered_seed_ids": [int(item["seed_id"]) for item in ordering],
        "median_seed_id": int(median["seed_id"]),
        "median_checkpoint_sha256": str(median["checkpoint_sha256"]),
        "best_seed_id": int(ordering[-1]["seed_id"]),
        "best_seed_checkpoint_rejected": True,
        "replica_metrics": [
            deepcopy(item) for item in sorted(artifacts, key=lambda value: int(value["seed_id"]))
        ],
    }
    return with_content_hash(payload)


def select_bridge_consumer_preconfirmation(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    f0_checkpoint_sha256: str,
    bridge_recipe_sha256: str,
    bridge_channel_policy: str = BRIDGE_CHANNEL_PHYSICAL45,
    selected_checkpoint_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Select once from fixed aggregates; never inspect an individual best seed."""

    if bridge_channel_policy != BRIDGE_CHANNEL_PHYSICAL45:
        raise ValueError("primary consumer selection is physical45")
    by_recipe: dict[str, Mapping[str, Any]] = {}
    for raw in aggregates:
        validate_content_hash(raw, expected_contract=PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT)
        recipe = str(raw["run_id"])
        if recipe not in SELECTION_RECIPE_IDS or recipe in by_recipe:
            raise ValueError("selector requires unique T10_clean and T10_robust aggregates")
        by_recipe[recipe] = raw
    if set(by_recipe) != set(SELECTION_RECIPE_IDS):
        raise ValueError("selector requires both predeclared consumer recipes")
    eligible = [raw for raw in by_recipe.values() if bool(raw["eligible"])]
    if not eligible:
        raise RuntimeError("no bridge-consumer recipe satisfies Section 18.1")
    best_score = max(float(raw["mean_bridge_accuracy"]) for raw in eligible)
    tie_pool = [raw for raw in eligible if best_score - float(raw["mean_bridge_accuracy"]) <= 0.0005]
    ranked = sorted(
        tie_pool,
        key=lambda raw: (
            -float(raw["mean_f0_accuracy"]),
            float(raw["mean_bridge_cross_entropy"]),
            float(raw["sample_std_delta_same"]),
            float(raw["mean_expected_calibration_error"]),
            float(raw["mean_brier_score"]),
            str(raw["run_id"]),
        ),
    )
    selected = ranked[0]
    recipe = str(selected["run_id"])
    median_seed = int(selected["median_seed_id"])
    checkpoint_hash = _sha256(selected["median_checkpoint_sha256"], name="checkpoint_sha256")
    checkpoint_path = None
    if selected_checkpoint_paths is not None:
        if recipe not in selected_checkpoint_paths:
            raise ValueError("selected checkpoint path mapping is incomplete")
        checkpoint_path = str(selected_checkpoint_paths[recipe])
    reason = {
        "best_score": best_score,
        "tie_tolerance": 0.0005,
        "tie_pool": sorted(str(raw["run_id"]) for raw in tie_pool),
        "secondary_order": [
            "higher_mean_f0_accuracy",
            "lower_mean_bridge_cross_entropy",
            "lower_sample_std_delta_same",
            "lower_mean_ece",
            "lower_mean_brier",
            "lexicographic_recipe_id",
        ],
        "winner": recipe,
    }
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CONSUMER_PRECONFIRMATION_CONTRACT,
            "schema": PREDICTION_ANCHORED_CONSUMER_PRECONFIRMATION_CONTRACT,
            "status": "LOCKED_AWAITING_STACK_VAL_CONSUMER",
            "selected_consumer_recipe": recipe,
            "selected_consumer_id": f"{recipe}_seed_{median_seed}",
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "selected_median_seed_id": median_seed,
            "selected_rho_endpoint": 0.10,
            "selection_source": "model_val_stop_then_model_val_select__stack_val_consumer_confirmation_only",
            "selection_reason": reason,
            "recipe_aggregate_metrics": deepcopy(dict(selected)),
            "all_recipe_aggregate_hashes": {
                name: str(value["content_hash"]) for name, value in sorted(by_recipe.items())
            },
            "replica_metrics": deepcopy(list(selected["replica_metrics"])),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_path": checkpoint_path,
            "f0_checkpoint_sha256": _sha256(f0_checkpoint_sha256, name="f0_checkpoint_sha256"),
            "bridge_recipe_sha256": _sha256(bridge_recipe_sha256, name="bridge_recipe_sha256"),
            "bridge_channel_policy": bridge_channel_policy,
            "model_val_select_bridge_response": deepcopy(list(selected["mean_response_curve"])),
            "negative_control_results": deepcopy(dict(selected["mean_negative_control_gains"])),
            "best_individual_seed_selectable": False,
            "stack_val_consumer_opened": False,
            "deployable": False,
        }
    )


def finalize_consumer_confirmation(
    preconfirmation: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    *,
    access_receipt: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Apply the one-shot confirmation gate with no runner-up fallback."""

    validate_content_hash(
        preconfirmation,
        expected_contract=PREDICTION_ANCHORED_CONSUMER_PRECONFIRMATION_CONTRACT,
    )
    validate_content_hash(access_receipt)
    required_receipt = {
        "status": "AUTHORIZED",
        "split_name": "stack_val_consumer",
        "purpose": "consumer_confirmation",
        "seal_kind": "consumer_preconfirmation",
        "one_shot": True,
        "selection_sha256": preconfirmation["content_hash"],
    }
    for key, expected in required_receipt.items():
        if access_receipt.get(key) != expected:
            raise PermissionError(f"confirmation receipt field {key!r} is invalid")
    if str(confirmation.get("checkpoint_sha256")) != preconfirmation["checkpoint_sha256"]:
        raise ValueError("confirmation used a different consumer checkpoint")
    if str(confirmation.get("bridge_recipe_sha256")) != preconfirmation["bridge_recipe_sha256"]:
        raise ValueError("confirmation used a different bridge recipe")
    try:
        raw_f0_accuracy = float(confirmation.get("f0_accuracy"))
        raw_bridge_accuracy = float(confirmation.get("bridge_0p10_accuracy"))
    except (TypeError, ValueError):
        raw_f0_accuracy = math.nan
        raw_bridge_accuracy = math.nan
    all_metrics_finite = (
        math.isfinite(raw_f0_accuracy)
        and math.isfinite(raw_bridge_accuracy)
        and _all_numeric_finite(confirmation)
    )
    f0_accuracy = raw_f0_accuracy if all_metrics_finite else None
    bridge_accuracy = raw_bridge_accuracy if all_metrics_finite else None
    delta = (
        raw_bridge_accuracy - raw_f0_accuracy
        if all_metrics_finite
        else None
    )
    passed = (
        delta is not None
        and delta >= 0
        and bool(confirmation.get("provenance_valid"))
    )
    record = {
        "split": "stack_val_consumer",
        "access_receipt_sha256": access_receipt["content_hash"],
        "checkpoint_sha256": preconfirmation["checkpoint_sha256"],
        "bridge_recipe_sha256": preconfirmation["bridge_recipe_sha256"],
        "f0_accuracy": f0_accuracy,
        "bridge_0p10_accuracy": bridge_accuracy,
        "delta_same": None if delta is None else float(delta),
        "all_metrics_finite": bool(all_metrics_finite),
        "provenance_valid": bool(confirmation.get("provenance_valid")),
        "passed": bool(passed),
    }
    root = None if output_dir is None else Path(output_dir)
    if not passed:
        stopped = with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_STOPPED_CAMPAIGN_CONTRACT,
                "status": "STOPPED_CONSUMER_CONFIRMATION_FAILED",
                "pre_confirmation_selection_sha256": preconfirmation["content_hash"],
                "selected_consumer_id": preconfirmation["selected_consumer_id"],
                "stack_val_consumer_confirmation": record,
                "runner_up_considered": False,
                "refit_performed": False,
                "downstream_submission_allowed": False,
            }
        )
        if root is not None:
            write_immutable_json(root / "stopped_campaign.json", stopped)
        return stopped
    final_payload = {
        key: deepcopy(value)
        for key, value in preconfirmation.items()
        if key not in {"contract", "schema", "content_hash", "status", "stack_val_consumer_opened"}
    }
    final_payload.update(
        {
            "contract": PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
            "schema": PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
            "status": "CONFIRMED_LOCKED",
            "pre_confirmation_selection_sha256": preconfirmation["content_hash"],
            "stack_val_consumer_confirmation": record,
            "stack_val_consumer_opened": True,
            "runner_up_considered": False,
            "refit_performed": False,
            "deployable": False,
        }
    )
    final = with_content_hash(final_payload)
    if root is not None:
        write_immutable_json(root / "selected_bridge_consumer.json", final)
    return final


def build_teacher_binding(
    *,
    binding_kind: str,
    run_id: str,
    aggregate: Mapping[str, Any],
    checkpoint_path: str,
    checkpoint_sha256: str,
    channel_policy: str,
    validation_manifest_hashes: Mapping[str, str],
    target_cache_namespace: str,
    bridge_recipe_sha256: str,
    primary_selection: Mapping[str, Any] | None = None,
    all50_scaler_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind an exact median teacher before any target-logit cache exists."""

    if binding_kind not in {"primary", "all50", "alternate"}:
        raise ValueError("unknown teacher binding kind")
    validate_content_hash(aggregate)
    checkpoint = _sha256(checkpoint_sha256, name="checkpoint_sha256")
    bridge_recipe = _sha256(bridge_recipe_sha256, name="bridge_recipe_sha256")
    expected = {
        "primary": (BRIDGE_CHANNEL_PHYSICAL45, PRIMARY_TEACHER_NAMESPACE),
        "all50": (BRIDGE_CHANNEL_ALL50, ALL50_TEACHER_NAMESPACE),
        "alternate": (BRIDGE_CHANNEL_PHYSICAL45, ALTERNATE_TEACHER_NAMESPACE),
    }[binding_kind]
    if (channel_policy, target_cache_namespace) != expected:
        raise ValueError("binding channel/namespace does not match its kind")
    for key, value in validation_manifest_hashes.items():
        _sha256(value, name=f"validation_manifest_hashes[{key!r}]")
    selection_hash = None
    if binding_kind == "primary":
        if primary_selection is None:
            raise ValueError("primary teacher binding requires selected_bridge_consumer.json")
        validate_content_hash(
            primary_selection,
            expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
        )
        if primary_selection.get("status") != "CONFIRMED_LOCKED":
            raise ValueError("primary consumer has not passed confirmation")
        expected_fields = {
            "selected_consumer_recipe": run_id,
            "checkpoint_sha256": checkpoint,
            "bridge_recipe_sha256": bridge_recipe,
            "bridge_channel_policy": channel_policy,
        }
        for key, value in expected_fields.items():
            if primary_selection.get(key) != value:
                raise ValueError(f"primary binding disagrees with selection field {key!r}")
        if (
            primary_selection.get("recipe_aggregate_metrics", {}).get("content_hash")
            != aggregate["content_hash"]
        ):
            raise ValueError("primary binding aggregate is not the locked selected aggregate")
        selection_hash = primary_selection["content_hash"]
    elif primary_selection is not None:
        raise ValueError("non-primary binding must not reuse the primary selection")
    all50_scaler_hash = None
    all50_extra_statistics = None
    if all50_scaler_artifact is not None:
        if binding_kind != "all50":
            raise ValueError("only the all50 binding may contain all50 correction scales")
        validate_content_hash(
            all50_scaler_artifact,
            expected_contract=PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT,
        )
        if (
            all50_scaler_artifact.get("channel_policy") != BRIDGE_CHANNEL_ALL50
            or all50_scaler_artifact.get("fit_split") != "stack_train_distill"
        ):
            raise ValueError("all50 correction scales must be fit on stack_train_distill")
        all50_scaler_hash = all50_scaler_artifact["content_hash"]
        all50_extra_statistics = {
            name: list(all50_scaler_artifact[name][45:])
            for name in (
                "q99_delta", "sigma_delta", "trust_scale", "epsilon",
                "active", "sparse_nonzero_fallback",
            )
        }
    if int(aggregate.get("median_seed_id", -1)) not in PAIRED_SEED_IDS:
        raise ValueError("teacher aggregate does not identify a paired median replica")
    if str(aggregate.get("median_checkpoint_sha256")) != checkpoint:
        raise ValueError("teacher binding checkpoint is not the aggregate median")
    if binding_kind == "all50" and str(run_id) != T10_ALL50_CLEAN:
        raise ValueError("all50 binding must use T10_all50_clean")
    if binding_kind in {"primary", "alternate"} and not bool(aggregate.get("eligible")):
        raise ValueError("physical45 binding recipe did not pass aggregate validity")
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_symlink() or not checkpoint_file.is_file():
        raise FileNotFoundError(f"bound teacher checkpoint is absent or unsafe: {checkpoint_file}")
    actual_checkpoint_sha256 = sha256_file(checkpoint_file)
    if actual_checkpoint_sha256 != checkpoint:
        raise ValueError("bound teacher checkpoint bytes disagree with checkpoint_sha256")
    if binding_kind == "primary" and primary_selection.get("checkpoint_path") not in {
        None,
        str(checkpoint_path),
    }:
        raise ValueError("primary binding checkpoint path differs from the locked selection")
    artifact = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_TEACHER_BINDING_CONTRACT,
            "binding_kind": binding_kind,
            "run_id": str(run_id),
            "recipe_aggregate_sha256": aggregate["content_hash"],
            "recipe_aggregate_metrics": deepcopy(dict(aggregate)),
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "selected_median_seed_id": int(aggregate["median_seed_id"]),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint,
            "checkpoint_size_bytes": int(checkpoint_file.stat().st_size),
            "checkpoint_bytes_verified": True,
            "channel_policy": channel_policy,
            "bridge_recipe_sha256": bridge_recipe,
            "validation_manifest_hashes": dict(validation_manifest_hashes),
            "primary_selection_sha256": selection_hash,
            "target_cache_namespace": target_cache_namespace,
            "target_cache_schema": TEACHER_LOGIT_CACHE_SCHEMA,
            "binding_created_before_cache": True,
            "checkpoint_refit_forbidden": True,
            "cache_artifact_sha256": None,
            "all50_correction_scaler_sha256": all50_scaler_hash,
            "all50_extra_correction_statistics": all50_extra_statistics,
            "all50_extra_statistics_fit_split": (
                "stack_train_distill" if all50_scaler_hash is not None else None
            ),
            "deployable": False,
        }
    )
    validate_teacher_binding(
        artifact,
        expected_kind=binding_kind,
        primary_selection=primary_selection,
    )
    return artifact


def validate_teacher_binding(
    binding: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
    primary_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_content_hash(binding, expected_contract=PREDICTION_ANCHORED_TEACHER_BINDING_CONTRACT)
    kind = str(binding.get("binding_kind"))
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"teacher binding kind mismatch: expected {expected_kind!r}, got {kind!r}")
    expected = {
        "primary": (BRIDGE_CHANNEL_PHYSICAL45, PRIMARY_TEACHER_NAMESPACE),
        "all50": (BRIDGE_CHANNEL_ALL50, ALL50_TEACHER_NAMESPACE),
        "alternate": (BRIDGE_CHANNEL_PHYSICAL45, ALTERNATE_TEACHER_NAMESPACE),
    }.get(kind)
    if expected is None or (binding.get("channel_policy"), binding.get("target_cache_namespace")) != expected:
        raise ValueError("teacher binding channel/namespace is invalid")
    _sha256(binding.get("checkpoint_sha256"), name="checkpoint_sha256")
    _sha256(binding.get("bridge_recipe_sha256"), name="bridge_recipe_sha256")
    if binding.get("target_cache_schema") != TEACHER_LOGIT_CACHE_SCHEMA:
        raise ValueError("teacher binding target cache schema changed")
    if binding.get("cache_artifact_sha256") is not None:
        raise ValueError("teacher binding circularly references a cache artifact")
    if not bool(binding.get("checkpoint_refit_forbidden")):
        raise ValueError("teacher binding permits a post-cache refit")
    all50_scaler_hash = binding.get("all50_correction_scaler_sha256")
    all50_statistics = binding.get("all50_extra_correction_statistics")
    if kind == "all50" and all50_scaler_hash is not None:
        _sha256(all50_scaler_hash, name="all50_correction_scaler_sha256")
        if binding.get("all50_extra_statistics_fit_split") != "stack_train_distill":
            raise ValueError("all50 extra correction statistics used another fit split")
        expected_statistics = {
            "q99_delta", "sigma_delta", "trust_scale", "epsilon",
            "active", "sparse_nonzero_fallback",
        }
        if not isinstance(all50_statistics, Mapping) or set(all50_statistics) != expected_statistics:
            raise ValueError("all50 binding correction statistics are incomplete")
        if any(not isinstance(all50_statistics[name], list) or len(all50_statistics[name]) != 5 for name in expected_statistics):
            raise ValueError("all50 binding must contain exactly five extra-channel statistics")
    elif all50_scaler_hash is not None or all50_statistics is not None or binding.get(
        "all50_extra_statistics_fit_split"
    ) is not None:
        raise ValueError("non-all50 binding contains all50 correction statistics")
    checkpoint_path = Path(str(binding.get("checkpoint_path")))
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"bound teacher checkpoint is absent or unsafe: {checkpoint_path}")
    if sha256_file(checkpoint_path) != binding.get("checkpoint_sha256"):
        raise ValueError("bound teacher checkpoint bytes changed")
    if int(binding.get("checkpoint_size_bytes", -1)) != int(checkpoint_path.stat().st_size):
        raise ValueError("bound teacher checkpoint size changed")
    if not bool(binding.get("checkpoint_bytes_verified")):
        raise ValueError("teacher binding did not verify checkpoint bytes")
    if kind == "primary":
        if primary_selection is None:
            raise ValueError("validating a primary binding requires its selection")
        validate_content_hash(
            primary_selection,
            expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
        )
        if binding.get("primary_selection_sha256") != primary_selection["content_hash"]:
            raise ValueError("primary teacher binding points to a different selection")
        if binding.get("checkpoint_sha256") != primary_selection["checkpoint_sha256"]:
            raise ValueError("primary teacher binding checkpoint changed")
    elif binding.get("primary_selection_sha256") is not None or primary_selection is not None:
        raise ValueError("non-primary teacher binding cannot substitute a primary selection")
    return {
        "ok": True,
        "binding_kind": kind,
        "binding_sha256": binding["content_hash"],
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "target_cache_namespace": binding["target_cache_namespace"],
    }


__all__ = [
    "PREDICTION_ANCHORED_CLASSIFICATION_METRICS_CONTRACT",
    "PREDICTION_ANCHORED_CONSUMER_REPLICA_EVALUATION_CONTRACT",
    "PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT",
    "PREDICTION_ANCHORED_CONSUMER_PRECONFIRMATION_CONTRACT",
    "PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT",
    "PREDICTION_ANCHORED_STOPPED_CAMPAIGN_CONTRACT",
    "PREDICTION_ANCHORED_TEACHER_BINDING_CONTRACT",
    "PRIMARY_TEACHER_NAMESPACE",
    "ALL50_TEACHER_NAMESPACE",
    "ALTERNATE_TEACHER_NAMESPACE",
    "N3_F0_TEACHER_NAMESPACE",
    "TEACHER_LOGIT_CACHE_SCHEMA",
    "ConsumerReplicaEvaluation",
    "classification_metrics",
    "slice_classification_metrics",
    "paired_bootstrap_difference",
    "build_consumer_replica_evaluation",
    "evaluate_bound_consumer_conditions",
    "aggregate_consumer_evaluations",
    "select_bridge_consumer_preconfirmation",
    "finalize_consumer_confirmation",
    "build_teacher_binding",
    "validate_teacher_binding",
]

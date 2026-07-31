"""Frozen H_BASE probe models, statistical controls, and metrics."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    PROBE_ENCODER_LOCK_CONTRACT,
    PROBE_RESULT_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .heads import GlobalTargetHead, PairTargetHead
from .taps import HBaseParticleTransformer, TAP_BLOCKS

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_hash(module: Any) -> str:
    rows = []
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        rows.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        )
    return canonical_sha256(rows)


def build_probe_encoder_lock(
    *,
    encoder: HBaseParticleTransformer,
    checkpoint_path: str | Path,
    checkpoint_registration_sha256: str,
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(encoder, HBaseParticleTransformer):
        raise TypeError("probe encoder must be exact H_BASE")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return with_content_hash(
        {
            "contract": PROBE_ENCODER_LOCK_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "baseline_id": "H_BASE",
            "seed": 101,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "checkpoint_registration_sha256": require_sha256(
                checkpoint_registration_sha256,
                name="checkpoint_registration_sha256",
            ),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "encoder_state_sha256": _state_hash(encoder),
            "tap_contract_sha256": encoder._hosd_split_adapter.contract["content_hash"],
            "frozen_before_target_results": True,
            "all_parameters_require_grad_false": all(
                not parameter.requires_grad for parameter in encoder.parameters()
            ),
        }
    )


def validate_frozen_encoder(
    encoder: HBaseParticleTransformer, lock: Mapping[str, Any]
) -> str:
    digest = validate_content_hash(
        lock, expected_contract=PROBE_ENCODER_LOCK_CONTRACT
    )
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("frozen probe encoder has trainable parameters")
    if _state_hash(encoder) != lock["encoder_state_sha256"]:
        raise RuntimeError("frozen probe encoder state drifted")
    return digest


def masked_mean(states: Any, mask: Any) -> Any:
    weight = mask.bool().unsqueeze(-1)
    count = weight.sum(dim=1).clamp_min(1)
    return states.masked_fill(~weight, 0).sum(dim=1) / count


class LinearTapProbe(torch.nn.Module if torch is not None else object):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        availability_groups: int = 1,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD probes")
        super().__init__()
        self.output = torch.nn.Linear(input_dimension, output_dimension)
        self.availability = torch.nn.Linear(input_dimension, availability_groups)

    def forward(self, states: Any, mask: Any) -> dict[str, Any]:
        pooled = masked_mean(states, mask)
        return {
            "value": self.output(pooled),
            "availability_logits": self.availability(pooled),
        }


class ShallowTapProbe(GlobalTargetHead):
    pass


class RawSummaryProbe(torch.nn.Module if torch is not None else object):
    """Two-layer MLP over the matching HLT summary plus five jet scalars."""

    def __init__(
        self,
        summary_dimension: int,
        output_dimension: int,
        availability_groups: int = 1,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD probes")
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.LayerNorm(summary_dimension + 5),
            torch.nn.Linear(summary_dimension + 5, 128),
            torch.nn.GELU(),
        )
        self.output = torch.nn.Linear(128, output_dimension)
        self.availability = torch.nn.Linear(128, availability_groups)

    def forward(self, matching_hlt_summary: Any, jet_context: Any) -> dict[str, Any]:
        if jet_context.shape[-1] != 5:
            raise ValueError("raw probe requires pt, eta, mass, multiplicity, track fraction")
        hidden = self.trunk(torch.cat((matching_hlt_summary, jet_context), -1))
        return {
            "value": self.output(hidden),
            "availability_logits": self.availability(hidden),
        }


class TargetToClassOracle(torch.nn.Module if torch is not None else object):
    def __init__(self, target_dimension: int, availability_dimension: int) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD probes")
        super().__init__()
        width = int(target_dimension) + int(availability_dimension)
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, target: Any, availability: Any) -> Any:
        return self.network(torch.cat((target, availability.float()), dim=-1))


def build_tap_probe(
    *,
    probe_kind: str,
    tap: str,
    input_dimension: int,
    target_dimension: int,
    head_type: str = "global",
    symmetric: bool = False,
    availability_groups: int = 1,
) -> Any:
    if tap not in TAP_BLOCKS:
        raise ValueError("unknown probe tap")
    if probe_kind not in {"P_LINEAR", "P_SHALLOW"}:
        raise ValueError("unknown learned tap probe")
    if head_type == "pair":
        if probe_kind == "P_LINEAR":
            # A one-layer pair probe is the linear counterpart to the locked
            # two-layer width-128 shallow pair head.
            class LinearPair(PairTargetHead):
                def __init__(self):
                    super().__init__(
                        input_dimension, target_dimension, symmetric=symmetric
                    )
                    factor = 3 if symmetric else 4
                    self.network = torch.nn.Linear(
                        factor * input_dimension, target_dimension
                    )

            return LinearPair()
        return PairTargetHead(
            input_dimension, target_dimension, symmetric=symmetric
        )
    if head_type != "global":
        raise ValueError("current-source probes support global or pair targets")
    if probe_kind == "P_LINEAR":
        return LinearTapProbe(
            input_dimension, target_dimension, availability_groups
        )
    return ShallowTapProbe(
        target_dimension,
        input_dimension=input_dimension,
        availability_groups=availability_groups,
        heteroscedastic=False,
    )


def statistical_references(
    targets: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    target_kind: str,
) -> dict[str, Any]:
    values = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(masks, dtype=bool)
    truth = np.asarray(labels, dtype=np.int64)
    if (
        values.shape != valid.shape
        or values.ndim < 2
        or values.shape[0] != truth.shape[0]
    ):
        raise ValueError("statistical-reference arrays disagree")
    if target_kind not in {
        "continuous",
        "categorical",
        "teacher_logits",
        "mixed_region_pair",
    }:
        raise ValueError("unknown statistical-reference target kind")

    component_count = int(values.shape[-1])

    def statistic(selected_values: np.ndarray, selected_mask: np.ndarray) -> list[float]:
        selected_values = selected_values.reshape(-1, component_count)
        selected_mask = selected_mask.reshape(-1, component_count)
        if target_kind == "teacher_logits":
            row_valid = selected_mask.all(axis=1)
            logits = selected_values[row_valid]
            if len(logits) == 0:
                return [0.1] * component_count
            shifted = logits - logits.max(axis=1, keepdims=True)
            probability = np.exp(shifted)
            probability /= probability.sum(axis=1, keepdims=True)
            return [float(value) for value in probability.mean(axis=0)]
        output = []
        for index in range(component_count):
            component = selected_values[:, index][selected_mask[:, index]]
            if component.size == 0:
                output.append(0.0)
            elif target_kind == "categorical" or (
                target_kind == "mixed_region_pair" and index < 3
            ):
                ones = float(component.mean())
                output.append(ones)
            else:
                output.append(float(np.median(component)))
        return output

    prior = statistic(values, valid)
    per_class = {}
    for class_index in range(10):
        selected = truth == class_index
        per_class[str(class_index)] = statistic(values[selected], valid[selected])
    return {
        "P_PRIOR": prior,
        "P_CLASS_CONDITIONAL_ORACLE": per_class,
        "fit_population": "model_train_only",
        "labels_used_only_by_class_conditional_non_deployable_oracle": True,
    }


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def continuous_probe_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if predicted.shape != truth.shape or truth.shape != valid.shape:
        raise ValueError("continuous probe metric shapes differ")
    residual = predicted[valid] - truth[valid]
    if residual.size == 0:
        raise ValueError("continuous probe has no valid values")
    centered = truth[valid] - truth[valid].mean()
    denominator = float(np.square(centered).sum())
    r2 = (
        0.0
        if denominator == 0
        else 1.0 - float(np.square(residual).sum()) / denominator
    )
    rank_x, rank_y = _ranks(predicted[valid]), _ranks(truth[valid])
    if rank_x.std() == 0 or rank_y.std() == 0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return {
        "normalized_mae": float(np.abs(residual).mean()),
        "normalized_rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": r2,
        "spearman": spearman,
        "normalized_median_absolute_error": float(np.median(np.abs(residual))),
        "valid_target_coverage": float(valid.mean()),
    }


def teacher_probe_metrics(
    prediction: np.ndarray, target: np.ndarray, *, temperature: float = 2.0
) -> dict[str, Any]:
    student = np.asarray(prediction, dtype=np.float64)
    teacher = np.asarray(target, dtype=np.float64)
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("teacher probe logits differ")
    def softmax(value):
        shifted = value / temperature
        shifted -= shifted.max(axis=1, keepdims=True)
        result = np.exp(shifted)
        return result / result.sum(axis=1, keepdims=True)
    p, q = softmax(teacher), softmax(student)
    kl = (temperature**2) * np.sum(
        p * (np.log(np.clip(p, 1e-12, None)) - np.log(np.clip(q, 1e-12, None))),
        axis=1,
    )
    correlations = []
    for index in range(teacher.shape[1]):
        if teacher[:, index].std() == 0 or student[:, index].std() == 0:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(teacher[:, index], student[:, index])[0, 1]))
    agreement = float((student.argmax(1) == teacher.argmax(1)).mean())
    return {
        "temperature_2_kl": float(kl.mean()),
        "top1_agreement": agreement,
        "per_class_logit_correlation": correlations,
        "frozen_teacher_decision_preservation": agreement,
    }


def latent_probe_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    frozen_head_prediction: np.ndarray | None = None,
    frozen_head_target: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if predicted.shape != truth.shape or truth.shape != valid.shape or truth.ndim != 2:
        raise ValueError("latent probe arrays differ")
    clean_pred = np.where(valid, predicted, 0.0)
    clean_truth = np.where(valid, truth, 0.0)
    residual = clean_pred - clean_truth
    absolute = np.abs(residual)
    huber = np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)
    cosine = np.sum(clean_pred * clean_truth, axis=1) / np.clip(
        np.linalg.norm(clean_pred, axis=1) * np.linalg.norm(clean_truth, axis=1),
        1e-12,
        None,
    )
    x = clean_pred - clean_pred.mean(axis=0, keepdims=True)
    y = clean_truth - clean_truth.mean(axis=0, keepdims=True)
    numerator = float(np.square(x.T @ y).sum())
    denominator = math.sqrt(
        float(np.square(x.T @ x).sum()) * float(np.square(y.T @ y).sum())
    )

    def effective_rank(value: np.ndarray) -> float:
        singular = np.linalg.svd(value, compute_uv=False)
        probability = singular / np.clip(singular.sum(), 1e-12, None)
        entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, None)))
        return float(np.exp(entropy))

    metrics: dict[str, Any] = {
        "normalized_huber": float(huber[valid].mean()),
        "normalized_rmse": float(np.sqrt(np.square(residual[valid]).mean())),
        "cosine_similarity": float(cosine.mean()),
        "linear_cka": 0.0 if denominator == 0 else numerator / denominator,
        "predicted_effective_rank": effective_rank(clean_pred),
        "target_effective_rank": effective_rank(clean_truth),
        "effective_rank_retention": (
            effective_rank(clean_pred) / max(effective_rank(clean_truth), 1e-12)
        ),
    }
    if (
        frozen_head_prediction is None
        or frozen_head_target is None
        or labels is None
    ):
        metrics.update(
            {
                "frozen_head_accuracy": None,
                "frozen_head_agreement": None,
                "frozen_head_status": "requires_locked_teacher_head_outputs",
            }
        )
    else:
        left = np.asarray(frozen_head_prediction).argmax(axis=1)
        right = np.asarray(frozen_head_target).argmax(axis=1)
        truth_labels = np.asarray(labels, dtype=np.int64)
        metrics.update(
            {
                "frozen_head_accuracy": float((left == truth_labels).mean()),
                "frozen_head_agreement": float((left == right).mean()),
                "frozen_head_status": "evaluated",
            }
        )
    return metrics


def _binary_auc(probability: np.ndarray, target: np.ndarray) -> float | None:
    truth = np.asarray(target, dtype=bool)
    positive, negative = int(truth.sum()), int((~truth).sum())
    if positive == 0 or negative == 0:
        return None
    ranks = _ranks(np.asarray(probability, dtype=np.float64)) + 1.0
    return float(
        (ranks[truth].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def _binary_pr_auc(probability: np.ndarray, target: np.ndarray) -> float | None:
    truth = np.asarray(target, dtype=bool)
    positives = int(truth.sum())
    if positives == 0:
        return None
    order = np.argsort(-np.asarray(probability), kind="mergesort")
    sorted_truth = truth[order].astype(np.float64)
    true_positive = np.cumsum(sorted_truth)
    precision = true_positive / np.arange(1, len(order) + 1)
    recall_delta = sorted_truth / positives
    return float(np.sum(precision * recall_delta))


def pair_probe_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    binary_channels: int,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if predicted.shape != truth.shape or truth.shape != valid.shape or truth.ndim != 4:
        raise ValueError("pair metric arrays differ")
    binary = []
    for channel in range(int(binary_channels)):
        selected = valid[..., channel]
        logits = predicted[..., channel][selected]
        labels = truth[..., channel][selected]
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        decision = probability >= 0.5
        positive = labels >= 0.5
        tpr = None if not positive.any() else float(decision[positive].mean())
        negative = ~positive
        tnr = None if not negative.any() else float((~decision[negative]).mean())
        binary.append(
            {
                "channel": channel,
                "roc_auc": _binary_auc(probability, positive),
                "pr_auc": _binary_pr_auc(probability, positive),
                "balanced_accuracy": (
                    None if tpr is None or tnr is None else 0.5 * (tpr + tnr)
                ),
                "prevalence": float(positive.mean()) if len(positive) else None,
                "probability_calibration_error": (
                    float(np.abs(probability - labels).mean())
                    if len(labels)
                    else None
                ),
            }
        )
    continuous = (
        None
        if binary_channels == truth.shape[-1]
        else continuous_probe_metrics(
            predicted[..., binary_channels:],
            truth[..., binary_channels:],
            valid[..., binary_channels:],
        )
    )
    return {
        "binary_channels": binary,
        "continuous_channels": continuous,
        "pair_reduction": "per_jet_training;all_applicable_pairs_validation",
    }


def build_probe_result(
    *,
    row: Mapping[str, Any],
    split: str,
    metrics: Mapping[str, Any],
    identity_order_sha256: str,
    target_cache_manifest_sha256: str,
    probe_encoder_lock_sha256: str | None,
    checkpoint_sha256: str | None,
    input_artifact_hashes: Mapping[str, str] | None = None,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if split not in {"val_stop", "design_select"}:
        raise ValueError("probe result split is not registered")
    return with_content_hash(
        {
            "contract": PROBE_RESULT_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "row_id": str(row["row_id"]),
            "target_id": str(row["target_id"]),
            "probe_kind": str(row["probe_kind"]),
            "tap": row.get("tap"),
            "pipeline_seed": (
                None if row.get("pipeline_seed") is None else int(row["pipeline_seed"])
            ),
            "component_seed": (
                None if row.get("component_seed") is None else int(row["component_seed"])
            ),
            "split": split,
            "metrics": dict(metrics),
            "identity_order_sha256": require_sha256(
                identity_order_sha256, name="identity_order_sha256"
            ),
            "target_cache_manifest_sha256": require_sha256(
                target_cache_manifest_sha256,
                name="target_cache_manifest_sha256",
            ),
            "probe_encoder_lock_sha256": (
                None
                if probe_encoder_lock_sha256 is None
                else require_sha256(
                    probe_encoder_lock_sha256,
                    name="probe_encoder_lock_sha256",
                )
            ),
            "checkpoint_sha256": (
                None
                if checkpoint_sha256 is None
                else require_sha256(checkpoint_sha256, name="checkpoint_sha256")
            ),
            "input_artifact_hashes": {
                str(name): require_sha256(value, name=f"input_artifact_hashes.{name}")
                for name, value in sorted((input_artifact_hashes or {}).items())
            },
            "selection_eligible": False,
            "can_cancel_later_stage": False,
        }
    )


__all__ = [
    "LinearTapProbe",
    "RawSummaryProbe",
    "ShallowTapProbe",
    "TargetToClassOracle",
    "build_probe_encoder_lock",
    "build_probe_result",
    "build_tap_probe",
    "continuous_probe_metrics",
    "masked_mean",
    "latent_probe_metrics",
    "pair_probe_metrics",
    "statistical_references",
    "teacher_probe_metrics",
    "validate_frozen_encoder",
]

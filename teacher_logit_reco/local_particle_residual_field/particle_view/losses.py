"""Masked representation and uncertainty losses for particle-view training."""

from __future__ import annotations

from typing import Any

import torch

from .contracts import require_sha256, with_content_hash
from .predictor import ParticleViewPredictorOutput
from .recovery_probe import (
    view_cosine_loss,
    view_huber_loss,
    view_relational_loss,
)


PARTICLE_VIEW_REPRESENTATION_OBJECTIVE_CONTRACT = (
    "particle_view_representation_objective_v1"
)
PARTICLE_VIEW_UNCERTAINTY_METRICS_CONTRACT = (
    "particle_view_uncertainty_metrics_v1"
)
PARTICLE_VIEW_UNCERTAINTY_CALIBRATION_REPORT_CONTRACT = (
    "particle_view_uncertainty_calibration_report_v1"
)


def heteroscedastic_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Global masked heteroscedastic Huber likelihood from the plan."""

    if (
        prediction.shape != target.shape
        or log_variance.shape != target.shape
        or mask.shape != target.shape[:2]
        or mask.dtype != torch.bool
    ):
        raise ValueError("uncertainty loss tensor shapes differ")
    if beta != 0.1:
        raise ValueError("particle-view uncertainty Huber beta changed")
    error = (prediction - target).abs()
    penalty = torch.where(
        error <= beta,
        error.square() / (2.0 * beta),
        error - beta / 2.0,
    )
    bounded = log_variance.clamp(-6.0, 3.0)
    values = 0.5 * torch.exp(-bounded) * penalty + 0.5 * bounded
    valid = mask[:, :, None].expand_as(values)
    if not valid.any():
        return values.new_zeros(())
    return values[valid].mean()


def particle_view_representation_losses(
    output: ParticleViewPredictorOutput,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Canonical four-epoch Pview_0 objective, including balance guarding."""

    if output.mean.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError("representation output/target/mask shapes differ")
    invalid = ~mask[:, :, None]
    if invalid.any():
        if output.mean[invalid.expand_as(output.mean)].abs().max().item() != 0:
            raise ValueError("predictor mean is nonzero on padding")
        if (
            output.log_variance[invalid.expand_as(output.log_variance)]
            .abs()
            .max()
            .item()
            != 0
        ):
            raise ValueError("predictor log variance is nonzero on padding")
    huber = view_huber_loss(output.mean, target, mask, beta=0.1)
    cosine = view_cosine_loss(output.mean, target, mask)
    relational = view_relational_loss(output.mean, target, mask)
    uncertainty = heteroscedastic_huber_loss(
        output.mean, target, output.log_variance, mask
    )
    total = (
        huber
        + 0.25 * cosine
        + 0.15 * relational
        + 0.05 * uncertainty
        + output.balance_loss
    )
    if not torch.isfinite(total):
        raise FloatingPointError("particle-view representation loss is non-finite")
    return {
        "total": total,
        "huber": huber,
        "cosine": cosine,
        "relational": relational,
        "uncertainty": uncertainty,
        "balance": output.balance_loss,
    }


def _average_tie_rank(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().double().flatten()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    _, counts = torch.unique_consecutive(
        sorted_values, return_counts=True
    )
    stops = counts.cumsum(dim=0)
    starts = stops - counts
    average = 0.5 * (starts + stops - 1).to(values.dtype)
    sorted_ranks = torch.repeat_interleave(average, counts)
    ranks = torch.empty_like(values)
    ranks[order] = sorted_ranks
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().double().flatten()
    right = right.detach().double().flatten()
    if left.numel() < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator.item() <= 0:
        return 0.0
    return float((left * right).sum().div(denominator).item())


def uncertainty_calibration_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    if (
        prediction.shape != target.shape
        or log_variance.shape != target.shape
        or mask.shape != target.shape[:2]
    ):
        raise ValueError("uncertainty metric tensor shapes differ")
    valid = mask[:, :, None].expand_as(target)
    if not valid.any():
        return {
            "contract": PARTICLE_VIEW_UNCERTAINTY_METRICS_CONTRACT,
            "valid_entries": 0,
            "nll": 0.0,
            "coverage_1sigma": 0.0,
            "coverage_1p96sigma": 0.0,
            "absolute_error_variance_spearman": 0.0,
            "fraction_log_variance_at_min": 0.0,
            "fraction_log_variance_at_max": 0.0,
        }
    error = (prediction - target)[valid].detach().float()
    bounded = log_variance.clamp(-6.0, 3.0)[valid].detach().float()
    variance = bounded.exp()
    sigma = variance.sqrt()
    absolute = error.abs()
    penalty = torch.where(
        absolute <= 0.1,
        absolute.square() / 0.2,
        absolute - 0.05,
    )
    nll = 0.5 * torch.exp(-bounded) * penalty + 0.5 * bounded
    return {
        "contract": PARTICLE_VIEW_UNCERTAINTY_METRICS_CONTRACT,
        "valid_entries": int(error.numel()),
        "nll": float(nll.mean().item()),
        "coverage_1sigma": float((absolute <= sigma).float().mean().item()),
        "coverage_1p96sigma": float(
            (absolute <= 1.96 * sigma).float().mean().item()
        ),
        "absolute_error_variance_spearman": _pearson(
            _average_tie_rank(absolute), _average_tie_rank(variance)
        ),
        "fraction_log_variance_at_min": float(
            bounded.le(-6.0).float().mean().item()
        ),
        "fraction_log_variance_at_max": float(
            bounded.ge(3.0).float().mean().item()
        ),
    }


def build_uncertainty_calibration_report(
    *,
    model_val_stop_metrics: dict[str, Any],
    model_val_select_metrics: dict[str, Any],
    pview0_checkpoint_sha256: str,
    coordinate_binding_sha256: str,
    model_val_stop_split_sha256: str,
    model_val_select_split_sha256: str,
) -> dict[str, Any]:
    for name, value in (
        ("pview0_checkpoint_sha256", pview0_checkpoint_sha256),
        ("coordinate_binding_sha256", coordinate_binding_sha256),
        ("model_val_stop_split_sha256", model_val_stop_split_sha256),
        ("model_val_select_split_sha256", model_val_select_split_sha256),
    ):
        require_sha256(name, value)
    for name, metrics in (
        ("model_val_stop", model_val_stop_metrics),
        ("model_val_select", model_val_select_metrics),
    ):
        if metrics.get("contract") != PARTICLE_VIEW_UNCERTAINTY_METRICS_CONTRACT:
            raise ValueError(f"{name} uncertainty metric contract mismatch")
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_UNCERTAINTY_CALIBRATION_REPORT_CONTRACT,
            "pview0_checkpoint_sha256": pview0_checkpoint_sha256,
            "coordinate_binding_sha256": coordinate_binding_sha256,
            "model_val_stop_split_sha256": model_val_stop_split_sha256,
            "model_val_select_split_sha256": model_val_select_split_sha256,
            "calibration_selection_split": "model_val_stop",
            "configuration_ranking_split": "model_val_select",
            "model_val_stop": model_val_stop_metrics,
            "model_val_select": model_val_select_metrics,
            "model_val_select_evaluated_once": True,
        }
    )


__all__ = [
    "PARTICLE_VIEW_REPRESENTATION_OBJECTIVE_CONTRACT",
    "PARTICLE_VIEW_UNCERTAINTY_METRICS_CONTRACT",
    "PARTICLE_VIEW_UNCERTAINTY_CALIBRATION_REPORT_CONTRACT",
    "build_uncertainty_calibration_report",
    "heteroscedastic_huber_loss",
    "particle_view_representation_losses",
    "uncertainty_calibration_metrics",
]

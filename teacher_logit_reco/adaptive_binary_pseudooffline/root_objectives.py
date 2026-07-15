"""Distribution-aware root objectives and physical validation diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .root_compiler import CompiledRootState
from .root_model import ABPH_CHARGE_SUPPORT_MIN, SemanticRootPrediction
from .root_transforms import (
    ROOT_AUXILIARY_FEATURE_NAMES,
    ROOT_FEATURE_INDEX,
    ROOT_RESIDUAL_CHANNEL_NAMES,
    ROOT_SHAPE_FEATURE_NAMES,
    RootNormalizationStats,
    RootResidualTargets,
    wrap_phi_tensor,
)
from .schemas import ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES


ABPH_ROOT_OBJECTIVE_CONTRACT = "adaptive_binary_pseudooffline_root_objective_v1"
ABPH_MIN_REQUIRED_ROOT_LOSS_WEIGHT = 1.0e-4


@dataclass(frozen=True)
class RootLossWeights:
    """Fixed-minimum primary root weights; no required channel can disappear."""

    p4_nll: float = 1.0
    count_nll: float = 0.75
    count_ordinal: float = 0.15
    delta_count_nll: float = 0.20
    composition_nll: float = 0.75
    charge_nll: float = 0.35
    shape_nll: float = 0.50
    auxiliary_huber: float = 0.20
    physical_huber: float = 0.20

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < ABPH_MIN_REQUIRED_ROOT_LOSS_WEIGHT:
                raise ValueError(
                    f"required root loss {name} must be finite and at least "
                    f"{ABPH_MIN_REQUIRED_ROOT_LOSS_WEIGHT}"
                )

    def to_dict(self) -> dict[str, float | str]:
        return {"contract": ABPH_ROOT_OBJECTIVE_CONTRACT, **asdict(self)}


@dataclass(frozen=True)
class RootLossOutput:
    total: Any
    components: Mapping[str, Any]
    weights: RootLossWeights

    def detached_components(self) -> dict[str, float]:
        return {
            name: float(value.detach().cpu())
            for name, value in self.components.items()
        }


def _gaussian_nll_from_error(error: Any, log_scale: Any) -> Any:
    torch = require_torch()
    scale = torch.exp(log_scale).clamp_min(1.0e-6)
    return (0.5 * (error / scale).square() + log_scale + 0.5 * math.log(2.0 * math.pi)).mean()


def _simplex_logistic_normal_nll(logits: Any, target: Any, log_scale: Any) -> Any:
    torch = require_torch()
    target = target.clamp_min(1.0e-6)
    target = target / target.sum(dim=-1, keepdim=True)
    target_log_ratio = target.log() - target.log().mean(dim=-1, keepdim=True)
    prediction_log_ratio = logits - logits.mean(dim=-1, keepdim=True)
    nll = _gaussian_nll_from_error(target_log_ratio - prediction_log_ratio, log_scale)
    cross_entropy = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    return nll + 0.25 * cross_entropy


def _physical_p4_residual_mean(
    prediction: SemanticRootPrediction,
    normalization: RootNormalizationStats | None,
) -> Any:
    if normalization is None:
        return prediction.p4_residual_mean
    return normalization.denormalize_named(
        prediction.p4_residual_mean, ROOT_RESIDUAL_CHANNEL_NAMES[:4]
    )


def _normalized_shape(
    values: Any,
    normalization: RootNormalizationStats | None,
) -> Any:
    if normalization is None:
        return values
    return normalization.normalize_named(values, ROOT_SHAPE_FEATURE_NAMES)


def _compiled_auxiliary(compiled: CompiledRootState) -> Any:
    torch = require_torch()
    return torch.stack(
        tuple(
            compiled.root_ledger[:, ROOT_FEATURE_INDEX[name]]
            for name in ROOT_AUXILIARY_FEATURE_NAMES
        ),
        dim=-1,
    )


def compute_root_losses(
    prediction: SemanticRootPrediction,
    compiled: CompiledRootState,
    targets: RootResidualTargets,
    *,
    normalization: RootNormalizationStats | None = None,
    weights: RootLossWeights | None = None,
) -> RootLossOutput:
    """Compute all required root losses with bounded, explicit channel weights."""

    torch = require_torch()
    resolved = weights or RootLossWeights()
    physical_mean = _physical_p4_residual_mean(prediction, normalization)
    physical_error = physical_mean - targets.p4_residuals
    physical_error = physical_error.clone()
    physical_error[:, 2] = wrap_phi_tensor(physical_error[:, 2])
    if normalization is not None:
        p4_scales = normalization.scale_named(
            ROOT_RESIDUAL_CHANNEL_NAMES[:4], like=physical_error
        )
        p4_error = physical_error / p4_scales
    else:
        p4_error = physical_error
    p4_nll = _gaussian_nll_from_error(p4_error, prediction.p4_residual_log_scale)

    count_nll = torch.nn.functional.cross_entropy(
        prediction.count_logits, targets.count_index
    )
    count_support = torch.arange(
        1,
        ABPH_MAX_PARTICLES + 1,
        dtype=prediction.count_logits.dtype,
        device=prediction.count_logits.device,
    )
    expected_count = (prediction.count_probabilities() * count_support).sum(dim=-1)
    target_count = targets.count_index.to(expected_count.dtype) + 1.0
    count_ordinal = torch.nn.functional.smooth_l1_loss(expected_count, target_count)

    if normalization is None:
        delta_target = targets.delta_count
    else:
        delta_target = normalization.normalize_named(
            targets.delta_count.unsqueeze(-1), (ROOT_RESIDUAL_CHANNEL_NAMES[4],)
        ).squeeze(-1)
    delta_count_nll = _gaussian_nll_from_error(
        prediction.delta_count_mean - delta_target,
        prediction.delta_count_log_scale,
    )

    pid_count = len(ABPH_PID_CATEGORIES)
    composition_nll = sum(
        (
            _simplex_logistic_normal_nll(
                logits,
                target,
                prediction.composition_log_scale[:, start : start + pid_count],
            )
            for start, logits, target in (
                (0, prediction.type_count_logits, targets.type_count_fractions),
                (pid_count, prediction.type_pt_logits, targets.type_pt_fractions),
                (2 * pid_count, prediction.type_energy_logits, targets.type_energy_fractions),
            )
        )
    ) / 3.0

    charge_index = targets.integer_charge - ABPH_CHARGE_SUPPORT_MIN
    if bool(((charge_index < 0) | (charge_index >= prediction.charge_logits.shape[-1])).any()):
        raise ValueError("root target charge lies outside the declared support")
    charge_nll = torch.nn.functional.cross_entropy(
        prediction.charge_logits, charge_index
    )

    predicted_shape = _normalized_shape(compiled.shape_features, normalization)
    target_shape = _normalized_shape(targets.shape_features, normalization)
    shape_nll = _gaussian_nll_from_error(
        predicted_shape - target_shape, prediction.shape_log_scale
    )

    predicted_auxiliary = _compiled_auxiliary(compiled)
    auxiliary_scale = targets.auxiliary_features.detach().abs().clamp_min(1.0)
    auxiliary_huber = torch.nn.functional.smooth_l1_loss(
        predicted_auxiliary / auxiliary_scale,
        targets.auxiliary_features / auxiliary_scale,
    )
    predicted_p4 = compiled.kinematics.four_vector()
    target_p4 = targets.physical.four_vector()
    p4_scale = target_p4.detach().abs().clamp_min(1.0)
    physical_huber = torch.nn.functional.smooth_l1_loss(
        predicted_p4 / p4_scale, target_p4 / p4_scale
    )

    components = {
        "p4_nll": p4_nll,
        "count_nll": count_nll,
        "count_ordinal": count_ordinal,
        "delta_count_nll": delta_count_nll,
        "composition_nll": composition_nll,
        "charge_nll": charge_nll,
        "shape_nll": shape_nll,
        "auxiliary_huber": auxiliary_huber,
        "physical_huber": physical_huber,
    }
    total = sum(
        components[name] * float(getattr(resolved, name))
        for name in components
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("root objective produced a nonfinite total loss")
    return RootLossOutput(total=total, components=components, weights=resolved)


def _mean(value: Any) -> float:
    return float(value.detach().float().mean().cpu())


def _rmse(value: Any) -> float:
    torch = require_torch()
    return float(torch.sqrt(value.detach().float().square().mean()).cpu())


def _coverage(error: Any, log_scale: Any, multiplier: float) -> float:
    torch = require_torch()
    bound = float(multiplier) * torch.exp(log_scale)
    return _mean(error.abs() <= bound)


def compute_root_metrics(
    prediction: SemanticRootPrediction,
    compiled: CompiledRootState,
    targets: RootResidualTargets,
    *,
    normalization: RootNormalizationStats | None = None,
    losses: RootLossOutput | None = None,
) -> dict[str, Any]:
    """Return model-validation diagnostics in physical units."""

    torch = require_torch()
    physical_residual_mean = _physical_p4_residual_mean(prediction, normalization)
    residual_error = physical_residual_mean - targets.p4_residuals
    residual_error = residual_error.clone()
    residual_error[:, 2] = wrap_phi_tensor(residual_error[:, 2])
    if normalization is None:
        coverage_error = residual_error
    else:
        coverage_error = residual_error / normalization.scale_named(
            ROOT_RESIDUAL_CHANNEL_NAMES[:4], like=residual_error
        )
    count_target = targets.count_index + 1
    count_predicted = compiled.constituent_count
    confusion = torch.zeros(
        (ABPH_MAX_PARTICLES, ABPH_MAX_PARTICLES), dtype=torch.long
    )
    flat_index = (
        (count_target.detach().cpu() - 1) * ABPH_MAX_PARTICLES
        + (count_predicted.detach().cpu() - 1)
    )
    confusion.view(-1).scatter_add_(
        0, flat_index, torch.ones_like(flat_index, dtype=torch.long)
    )
    type_count_error = compiled.type_counts.to(torch.float32) - targets.type_counts.to(torch.float32)
    type_fraction_error = compiled.type_count_fractions - targets.type_count_fractions
    type_pt_error = compiled.type_pt_fractions - targets.type_pt_fractions
    shape_error = compiled.shape_features - targets.shape_features
    charge_error = compiled.integer_charge - targets.integer_charge
    p4_error = compiled.kinematics.four_vector() - targets.physical.four_vector()
    metrics: dict[str, Any] = {
        "contract": ABPH_ROOT_OBJECTIVE_CONTRACT,
        "n_jets": int(count_target.shape[0]),
        "normalization_hash": None if normalization is None else normalization.normalization_hash,
        "physical": {
            "pt_response_mean": _mean(
                compiled.kinematics.pt / targets.physical.pt.clamp_min(1.0e-8)
            ),
            "pt_mae_gev": _mean((compiled.kinematics.pt - targets.physical.pt).abs()),
            "eta_mae": _mean((compiled.kinematics.eta - targets.physical.eta).abs()),
            "phi_mae": _mean(
                wrap_phi_tensor(compiled.kinematics.phi - targets.physical.phi).abs()
            ),
            "mass_response_mean": _mean(
                compiled.kinematics.mass / targets.physical.mass.clamp_min(1.0e-8)
            ),
            "mass_mae_gev": _mean((compiled.kinematics.mass - targets.physical.mass).abs()),
            "four_vector_rmse": _rmse(p4_error),
        },
        "residuals": {
            name: {
                "bias": _mean(residual_error[:, index]),
                "rmse": _rmse(residual_error[:, index]),
                "coverage_68": _coverage(
                    coverage_error[:, index], prediction.p4_residual_log_scale[:, index], 1.0
                ),
                "coverage_95": _coverage(
                    coverage_error[:, index], prediction.p4_residual_log_scale[:, index], 1.96
                ),
            }
            for index, name in enumerate(ROOT_RESIDUAL_CHANNEL_NAMES[:4])
        },
        "count": {
            "exact_accuracy": _mean(count_predicted == count_target),
            "mae": _mean((count_predicted - count_target).abs()),
            "confusion_matrix": confusion.tolist(),
        },
        "composition": {
            "all_type_counts_exact": _mean(
                (compiled.type_counts == targets.type_counts).all(dim=-1)
            ),
            "type_count_mae": {
                name: _mean(type_count_error[:, index].abs())
                for index, name in enumerate(ABPH_PID_CATEGORIES)
            },
            "type_count_fraction_mae": _mean(type_fraction_error.abs()),
            "type_pt_fraction_mae": _mean(type_pt_error.abs()),
        },
        "charge": {
            "exact_accuracy": _mean(compiled.integer_charge == targets.integer_charge),
            "mae": _mean(charge_error.abs()),
            "all_compiled_feasible": bool(compiled.diagnostics["ok"]),
        },
        "shape": {
            "mae": _mean(shape_error.abs()),
            "per_channel_mae": {
                name: _mean(shape_error[:, index].abs())
                for index, name in enumerate(ROOT_SHAPE_FEATURE_NAMES)
            },
        },
        "compiler": dict(compiled.diagnostics),
    }
    if losses is not None:
        metrics["losses"] = {
            "total": float(losses.total.detach().cpu()),
            **losses.detached_components(),
            "weights": losses.weights.to_dict(),
        }
    return metrics


def root_head_gradient_norms(model: Any) -> dict[str, float]:
    """Expose required-head gradient norms so learned balancing cannot hide a head."""

    groups = {
        "semantic_queries": ("query_tokens", "query_type_embedding", "blocks"),
        "p4": ("p4_head",),
        "count": ("count_head",),
        "composition": ("composition_head",),
        "shape": ("shape_head",),
        "charge": ("charge_head",),
        "uncertainty": ("uncertainty_head",),
    }
    totals = {name: 0.0 for name in groups}
    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        squared = float(parameter.grad.detach().float().square().sum().cpu())
        for group, prefixes in groups.items():
            if parameter_name.startswith(prefixes):
                totals[group] += squared
                break
    return {name: math.sqrt(value) for name, value in totals.items()}


def root_acceptance_report(
    prediction: SemanticRootPrediction,
    compiled: CompiledRootState,
) -> dict[str, Any]:
    """Compact deployable acceptance record used by smoke tests and run reports."""

    torch = require_torch()
    type_closure = compiled.type_counts.sum(dim=-1) - compiled.constituent_count
    fractions = (
        compiled.type_count_fractions,
        compiled.type_pt_fractions,
        compiled.type_energy_fractions,
    )
    fraction_valid = all(
        bool(torch.isfinite(value).all())
        and not bool((value < 0.0).any())
        and bool(
            torch.allclose(
                value.sum(dim=-1),
                torch.ones_like(value.sum(dim=-1)),
                atol=2.0e-6,
                rtol=2.0e-6,
            )
        )
        for value in fractions
    )
    offline_loaded = bool(prediction.diagnostics.get("offline_inputs_loaded", True))
    return {
        "ok": bool(compiled.diagnostics["ok"])
        and not offline_loaded
        and bool((type_closure == 0).all())
        and fraction_valid,
        "compiler_ok": bool(compiled.diagnostics["ok"]),
        "max_type_count_closure": int(type_closure.abs().max().detach().cpu()),
        "all_fractions_valid": fraction_valid,
        "offline_inputs_loaded": offline_loaded,
        "teacher_logits_loaded": bool(
            prediction.diagnostics.get("teacher_logits_loaded", True)
        ),
    }


__all__ = [
    "ABPH_MIN_REQUIRED_ROOT_LOSS_WEIGHT",
    "ABPH_ROOT_OBJECTIVE_CONTRACT",
    "RootLossOutput",
    "RootLossWeights",
    "compute_root_losses",
    "compute_root_metrics",
    "root_acceptance_report",
    "root_head_gradient_norms",
]

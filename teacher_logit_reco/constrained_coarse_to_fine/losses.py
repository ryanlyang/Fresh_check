"""Hierarchy reconstruction losses and diagnostics for Step 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from .constraints import ACCOUNTING_INDEX, PID_PT_INDICES
from .layout import ACCOUNTING_FIELD_NAMES, DERIVED_DIAGNOSTIC_FIELD_NAMES, MOMENT_FIELD_NAMES
from .model import CoarseToFineReconstructorOutput


HIERARCHY_RECONSTRUCTION_LOSS_CONTRACT = "constrained_coarse_to_fine_hierarchy_loss_v1"


@dataclass(frozen=True)
class HierarchyReconstructionLossConfig:
    global_weight: float = 1.0
    grid_weight: float = 1.0
    relative_weight: float = 0.25
    auxiliary_weight: float = 0.25
    allocation_kl_weight: float = 0.10
    # This is auxiliary calibration, not the primary accounting objective.  A
    # modest weight keeps an initially overconfident uncertainty head from
    # overwhelming the deterministic reconstruction gradients.
    uncertainty_weight: float = 0.01
    high_pt_cell_weight: float = 0.75
    huber_beta: float = 0.20
    # exp(-2 * sigma) is the learned precision.  -2 caps that multiplier at
    # exp(4) ~= 55, avoiding the exp(8) ~= 3,000 amplification that made the
    # B4 no-moment ablation numerically unstable on real batches.
    uncertainty_log_sigma_floor: float = -2.0
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "uncertainty_log_sigma_floor":
                continue
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if float(self.huber_beta) <= 0.0 or float(self.epsilon) <= 0.0:
            raise ValueError("huber_beta and epsilon must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = HIERARCHY_RECONSTRUCTION_LOSS_CONTRACT
        return payload


@dataclass(frozen=True)
class HierarchyReconstructionLossOutput:
    loss: torch.Tensor
    components: Mapping[str, torch.Tensor]
    metrics: Mapping[str, torch.Tensor]

    def detached_summary(self) -> dict[str, Any]:
        return {
            "contract": HIERARCHY_RECONSTRUCTION_LOSS_CONTRACT,
            "loss": float(self.loss.detach().cpu().item()),
            "components": {
                name: float(value.detach().cpu().item()) for name, value in self.components.items()
            },
            "metrics": {
                name: float(value.detach().cpu().item()) for name, value in self.metrics.items()
            },
        }


def accounting_diagnostics_torch(
    accounting: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Differentiable equivalent of the deterministic target diagnostics."""

    if int(accounting.shape[-1]) != len(ACCOUNTING_FIELD_NAMES):
        raise ValueError("accounting has the wrong field dimension")
    total_pt = accounting[..., ACCOUNTING_INDEX["total_pT"]]
    total_count = accounting[..., ACCOUNTING_INDEX["expected_constituent_count"]]
    safe_pt = total_pt.clamp_min(epsilon)
    safe_count = total_count.clamp_min(epsilon)
    axis_deta = (
        accounting[..., ACCOUNTING_INDEX["sum_pT_abs_deta_pos"]]
        - accounting[..., ACCOUNTING_INDEX["sum_pT_abs_deta_neg"]]
    ) / safe_pt
    axis_dphi = (
        accounting[..., ACCOUNTING_INDEX["sum_pT_abs_dphi_pos"]]
        - accounting[..., ACCOUNTING_INDEX["sum_pT_abs_dphi_neg"]]
    ) / safe_pt
    width_eta = (
        accounting[..., ACCOUNTING_INDEX["sum_pT_deta2"]] / safe_pt - axis_deta.square()
    ).clamp_min(0.0)
    width_phi = (
        accounting[..., ACCOUNTING_INDEX["sum_pT_dphi2"]] / safe_pt - axis_dphi.square()
    ).clamp_min(0.0)
    result = torch.stack(
        (
            *(accounting[..., index] / safe_pt for index in PID_PT_INDICES),
            axis_deta,
            axis_dphi,
            width_eta,
            width_phi,
            accounting[..., ACCOUNTING_INDEX["sum_pT_r"]] / safe_pt,
            torch.sqrt((accounting[..., ACCOUNTING_INDEX["sum_pT_r2"]] / safe_pt).clamp_min(0.0)),
            total_pt / safe_count,
            accounting[..., ACCOUNTING_INDEX["total_energy"]] / safe_pt,
        ),
        dim=-1,
    )
    result = torch.where((total_pt > epsilon).unsqueeze(-1), result, torch.zeros_like(result))
    if int(result.shape[-1]) != len(DERIVED_DIAGNOSTIC_FIELD_NAMES):
        raise AssertionError("derived diagnostic tensor has the wrong field dimension")
    return result


def _masked_field_mean(value: torch.Tensor, field_mask: torch.Tensor) -> torch.Tensor:
    mask = field_mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(0)
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)


def _composition_fractions(accounting: torch.Tensor, epsilon: float) -> torch.Tensor:
    category_pt = accounting[..., list(PID_PT_INDICES)].clamp_min(0.0)
    return category_pt / category_pt.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def _accounting_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_sigma: torch.Tensor,
    field_mask: torch.Tensor,
    config: HierarchyReconstructionLossConfig,
    *,
    cell_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = target.to(device=prediction.device, dtype=prediction.dtype).clamp_min(0.0)
    log_error = torch.log1p(prediction.clamp_min(0.0)) - torch.log1p(target)
    element = F.smooth_l1_loss(
        torch.log1p(prediction.clamp_min(0.0)),
        torch.log1p(target),
        beta=float(config.huber_beta),
        reduction="none",
    )
    sigma = log_sigma.to(dtype=prediction.dtype).clamp_min(float(config.uncertainty_log_sigma_floor))
    nll_element = 0.5 * log_error.square() * torch.exp(-2.0 * sigma) + sigma
    if cell_weights is not None:
        weights = cell_weights.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
        element = element * weights
        nll_element = nll_element * weights
    accounting_loss = _masked_field_mean(element, field_mask)
    uncertainty_nll = _masked_field_mean(nll_element, field_mask)
    if bool(field_mask[list(PID_PT_INDICES)].all()):
        relative = (
            _composition_fractions(prediction, config.epsilon)
            - _composition_fractions(target, config.epsilon)
        ).abs().mean()
    else:
        relative = prediction.new_zeros(())
    return accounting_loss, relative, uncertainty_nll


def _auxiliary_field_mask(field_mask: torch.Tensor) -> torch.Tensor:
    """Prevent derived losses from bypassing B4-B6 channel ablations."""

    result = torch.ones(len(DERIVED_DIAGNOSTIC_FIELD_NAMES), dtype=torch.bool, device=field_mask.device)
    result[:5] = field_mask[list(PID_PT_INDICES)].all()
    moment_available = field_mask[[ACCOUNTING_INDEX[name] for name in MOMENT_FIELD_NAMES]].all()
    result[5:11] = moment_available
    result[11] = field_mask[ACCOUNTING_INDEX["expected_constituent_count"]]
    result[12] = field_mask[ACCOUNTING_INDEX["total_energy"]]
    return result


def _cell_weights(target: torch.Tensor, config: HierarchyReconstructionLossConfig) -> torch.Tensor:
    pt = target[..., ACCOUNTING_INDEX["total_pT"]].clamp_min(0.0)
    share = pt / pt.sum(dim=1, keepdim=True).clamp_min(config.epsilon)
    emphasis = torch.sqrt(share * float(max(1, int(pt.shape[1])))).clamp(max=4.0)
    weights = 1.0 + float(config.high_pt_cell_weight) * emphasis
    return weights / weights.mean(dim=1, keepdim=True).clamp_min(config.epsilon)


def _allocation_kl(prediction: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
    pred_pt = prediction[..., ACCOUNTING_INDEX["total_pT"]].clamp_min(0.0)
    target_pt = target[..., ACCOUNTING_INDEX["total_pT"]].clamp_min(0.0)
    pred_distribution = (pred_pt + epsilon) / (pred_pt.sum(dim=1, keepdim=True) + epsilon * pred_pt.shape[1])
    target_distribution = (target_pt + epsilon) / (
        target_pt.sum(dim=1, keepdim=True) + epsilon * target_pt.shape[1]
    )
    return (
        target_distribution
        * (torch.log(target_distribution.clamp_min(epsilon)) - torch.log(pred_distribution.clamp_min(epsilon)))
    ).sum(dim=1).mean()


def _high_pt_cell_recall(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cells = int(prediction.shape[1])
    top_k = max(1, cells // 8)
    pred_top = prediction[..., ACCOUNTING_INDEX["total_pT"]].topk(top_k, dim=1).indices
    target_top = target[..., ACCOUNTING_INDEX["total_pT"]].topk(top_k, dim=1).indices
    matches = (pred_top.unsqueeze(-1) == target_top.unsqueeze(-2)).any(dim=-1).float()
    return matches.mean()


def compute_hierarchy_reconstruction_loss(
    output: CoarseToFineReconstructorOutput,
    targets: Mapping[str, torch.Tensor],
    config: HierarchyReconstructionLossConfig | Mapping[str, Any] | None = None,
) -> HierarchyReconstructionLossOutput:
    """Supervise global and every produced hierarchy level."""

    if config is None:
        config = HierarchyReconstructionLossConfig()
    elif not isinstance(config, HierarchyReconstructionLossConfig):
        config = HierarchyReconstructionLossConfig(**dict(config))
    if "global_accounting" not in targets:
        raise KeyError("targets are missing global_accounting")
    field_mask = output.supervised_field_mask.to(device=output.global_accounting.device)
    global_target = targets["global_accounting"].to(
        device=output.global_accounting.device, dtype=output.global_accounting.dtype
    )
    global_accounting, global_relative, global_nll = _accounting_terms(
        output.global_accounting,
        global_target,
        output.global_log_sigma,
        field_mask,
        config,
    )
    target_aux = accounting_diagnostics_torch(global_target, epsilon=config.epsilon)
    auxiliary_element = F.smooth_l1_loss(
        output.global_auxiliary,
        target_aux,
        beta=float(config.huber_beta),
        reduction="none",
    )
    auxiliary = _masked_field_mean(auxiliary_element, _auxiliary_field_mask(field_mask))
    global_loss = (
        global_accounting
        + float(config.relative_weight) * global_relative
        + float(config.auxiliary_weight) * auxiliary
        + float(config.uncertainty_weight) * global_nll
    )
    components: dict[str, torch.Tensor] = {
        "global_accounting": global_accounting,
        "global_relative": global_relative,
        "global_auxiliary": auxiliary,
        "global_uncertainty_nll": global_nll,
    }
    metrics: dict[str, torch.Tensor] = {}
    total_pt_index = ACCOUNTING_INDEX["total_pT"]
    count_index = ACCOUNTING_INDEX["expected_constituent_count"]
    metrics["global_total_pT_mae"] = (
        output.global_accounting[..., total_pt_index] - global_target[..., total_pt_index]
    ).abs().mean()
    metrics["global_total_pT_relative_mae"] = (
        (output.global_accounting[..., total_pt_index] - global_target[..., total_pt_index]).abs()
        / global_target[..., total_pt_index].clamp_min(config.epsilon)
    ).mean()
    metrics["global_energy_relative_mae"] = (
        (output.global_accounting[..., ACCOUNTING_INDEX["total_energy"]] - global_target[..., ACCOUNTING_INDEX["total_energy"]]).abs()
        / global_target[..., ACCOUNTING_INDEX["total_energy"]].clamp_min(config.epsilon)
    ).mean()
    metrics["global_count_mae"] = (
        output.global_accounting[..., count_index] - global_target[..., count_index]
    ).abs().mean()
    metrics["global_composition_mae"] = (
        _composition_fractions(output.global_accounting, config.epsilon)
        - _composition_fractions(global_target, config.epsilon)
    ).abs().mean()
    metrics["global_axis_delta_mae"] = (
        output.global_auxiliary[..., 5:7] - target_aux[..., 5:7]
    ).abs().mean()
    metrics["global_width_mae"] = (
        output.global_auxiliary[..., 7:9] - target_aux[..., 7:9]
    ).abs().mean()

    grid_losses: list[torch.Tensor] = []
    parent_accounting = output.global_accounting.unsqueeze(1)
    for level in output.levels:
        key = f"level{int(level.level)}_accounting"
        if key not in targets:
            raise KeyError(f"targets are missing {key}")
        target = targets[key].to(device=level.accounting.device, dtype=level.accounting.dtype)
        weights = _cell_weights(target, config)
        accounting, relative, uncertainty = _accounting_terms(
            level.accounting,
            target,
            level.log_sigma,
            field_mask,
            config,
            cell_weights=weights,
        )
        allocation_kl = _allocation_kl(level.accounting, target, config.epsilon)
        level_loss = (
            accounting
            + float(config.relative_weight) * relative
            + float(config.allocation_kl_weight) * allocation_kl
            + float(config.uncertainty_weight) * uncertainty
        )
        prefix = f"level{int(level.level)}"
        components[f"{prefix}_accounting"] = accounting
        components[f"{prefix}_relative"] = relative
        components[f"{prefix}_allocation_kl"] = allocation_kl
        components[f"{prefix}_uncertainty_nll"] = uncertainty
        grid_losses.append(level_loss)
        metrics[f"{prefix}_accounting_mae"] = (level.accounting - target).abs().mean()
        metrics[f"{prefix}_pT_allocation_kl"] = allocation_kl
        metrics[f"{prefix}_composition_allocation_error"] = (
            _composition_fractions(level.accounting, config.epsilon)
            - _composition_fractions(target, config.epsilon)
        ).abs().mean()
        metrics[f"{prefix}_high_pT_cell_recall"] = _high_pt_cell_recall(level.accounting, target)
        metrics[f"{prefix}_count_calibration_mae"] = (
            level.accounting[..., count_index] - target[..., count_index]
        ).abs().mean()
        metrics[f"{prefix}_parent_child_consistency_max"] = level.parent_closure_error(
            parent_accounting
        ).amax()
        parent_accounting = level.accounting
    grid_loss = torch.stack(grid_losses).mean() if grid_losses else global_loss.new_zeros(())
    components["grid"] = grid_loss
    loss = float(config.global_weight) * global_loss + float(config.grid_weight) * grid_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("hierarchy reconstruction loss is non-finite")
    return HierarchyReconstructionLossOutput(loss=loss, components=components, metrics=metrics)


__all__ = [
    "HIERARCHY_RECONSTRUCTION_LOSS_CONTRACT",
    "HierarchyReconstructionLossConfig",
    "HierarchyReconstructionLossOutput",
    "accounting_diagnostics_torch",
    "compute_hierarchy_reconstruction_loss",
]

"""Losses for local-graph residual experts.

These losses train a local-graph expert to correct a frozen HLT ParT logit
margin near the QCD/Hgg FPR@50 operating point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch

from .residual_cache import operating_point_from_scores


LOCAL_GRAPH_RESIDUAL_LOSS_STEP = "local_graph_residual_expert_step3_losses"
LOCAL_GRAPH_RESIDUAL_LOSS_CONTRACT = "local_graph_residual_losses_v1"

LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE = "residual_weighted_bce"
LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE = "residual_boundary_pairwise"
LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR = "residual_boundary_pairwise_bce_anchor"
LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR = (
    "residual_boundary_pairwise_soft_fpr_bce_anchor"
)
LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR_ALPHA_SHRINK = (
    "residual_boundary_pairwise_soft_fpr_bce_anchor_alpha_shrink"
)
LOCAL_GRAPH_RESIDUAL_LOSS_MODES = (
    LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
    LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE,
    LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
)
LOCAL_GRAPH_RESIDUAL_REPORT_POLICY_ALPHA_SHRINK = "model_val_gamma_shrink"
LOCAL_GRAPH_RESIDUAL_ALPHA_GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25)


def normalize_local_graph_residual_loss_mode(value: str) -> str:
    clean = str(value).strip().lower().replace("-", "_")
    aliases = {
        "a": LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
        "weighted_bce": LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
        "residual_weighted_bce": LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
        "b": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE,
        "boundary_pairwise": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE,
        "residual_boundary_pairwise": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE,
        "c": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
        "boundary_pairwise_bce_anchor": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
        "residual_boundary_pairwise_bce_anchor": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
        "d": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
        "boundary_pairwise_soft_fpr_bce_anchor": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
        "residual_boundary_pairwise_soft_fpr_bce_anchor": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
    }
    if clean in {
        "e",
        "boundary_pairwise_soft_fpr_bce_anchor_alpha_shrink",
        "residual_boundary_pairwise_soft_fpr_bce_anchor_alpha_shrink",
    }:
        raise ValueError(
            "Residual ladder E is no longer a separate training loss. Train mode D and use the "
            "validation-shrunk report rows instead."
        )
    if clean not in aliases:
        raise ValueError(f"residual loss mode must be one of {LOCAL_GRAPH_RESIDUAL_LOSS_MODES}, got {value!r}")
    return aliases[clean]


def _optional_positive_float(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be positive when provided")
    return value


@dataclass(frozen=True)
class LocalGraphResidualLossConfig:
    """Configuration for the Step 3 residual-expert objective ladder."""

    mode: str = LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE
    bce_anchor_weight: float = 0.10
    soft_fpr_weight: float = 0.25
    residual_l2_weight: float = 1.0e-4
    pairwise_weight: float = 1.0
    weighted_bce_weight: float = 1.0

    pairwise_temperature: float = 0.20
    soft_fpr_epsilon: float = 0.20
    cvar_top_fraction: float = 0.50
    hard_background_fraction: float = 0.20
    signal_boundary_quantile_low: float = 0.40
    signal_boundary_quantile_high: float = 0.60

    bce_low_margin_weight: float = 1.0
    bce_baseline_false_positive_weight: float = 4.0
    bce_baseline_false_negative_weight: float = 2.0
    bce_signal_boundary_weight: float = 1.0
    bce_boundary_scale: float | None = None
    normalize_bce_weights: bool = True

    default_tau50: float = 0.0
    default_tau30: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_local_graph_residual_loss_mode(self.mode))
        for field_name in (
            "bce_anchor_weight",
            "soft_fpr_weight",
            "residual_l2_weight",
            "pairwise_weight",
            "weighted_bce_weight",
            "pairwise_temperature",
            "soft_fpr_epsilon",
            "bce_low_margin_weight",
            "bce_baseline_false_positive_weight",
            "bce_baseline_false_negative_weight",
            "bce_signal_boundary_weight",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative")
            object.__setattr__(self, field_name, value)
        if float(self.pairwise_temperature) <= 0.0:
            raise ValueError("pairwise_temperature must be positive")
        if float(self.soft_fpr_epsilon) <= 0.0:
            raise ValueError("soft_fpr_epsilon must be positive")
        for field_name in ("cvar_top_fraction", "hard_background_fraction"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be in (0, 1]")
            object.__setattr__(self, field_name, value)
        q_low = float(self.signal_boundary_quantile_low)
        q_high = float(self.signal_boundary_quantile_high)
        if not (0.0 <= q_low <= q_high <= 1.0):
            raise ValueError("signal boundary quantiles must satisfy 0 <= low <= high <= 1")
        object.__setattr__(self, "signal_boundary_quantile_low", q_low)
        object.__setattr__(self, "signal_boundary_quantile_high", q_high)
        object.__setattr__(
            self,
            "bce_boundary_scale",
            _optional_positive_float(self.bce_boundary_scale, field_name="bce_boundary_scale"),
        )
        object.__setattr__(self, "normalize_bce_weights", bool(self.normalize_bce_weights))
        object.__setattr__(self, "default_tau50", float(self.default_tau50))
        object.__setattr__(self, "default_tau30", float(self.default_tau30))

    def active_weights(self) -> dict[str, float]:
        if self.mode == LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE:
            return {
                "weighted_bce": float(self.weighted_bce_weight),
                "boundary_pairwise": 0.0,
                "soft_fpr50": 0.0,
                "residual_l2": float(self.residual_l2_weight),
            }
        if self.mode == LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE:
            return {
                "weighted_bce": 0.0,
                "boundary_pairwise": float(self.pairwise_weight),
                "soft_fpr50": 0.0,
                "residual_l2": float(self.residual_l2_weight),
            }
        if self.mode == LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR:
            return {
                "weighted_bce": float(self.bce_anchor_weight),
                "boundary_pairwise": float(self.pairwise_weight),
                "soft_fpr50": 0.0,
                "residual_l2": float(self.residual_l2_weight),
            }
        return {
            "weighted_bce": float(self.bce_anchor_weight),
            "boundary_pairwise": float(self.pairwise_weight),
            "soft_fpr50": float(self.soft_fpr_weight),
            "residual_l2": float(self.residual_l2_weight),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_weights"] = self.active_weights()
        payload["step"] = LOCAL_GRAPH_RESIDUAL_LOSS_STEP
        payload["contract"] = LOCAL_GRAPH_RESIDUAL_LOSS_CONTRACT
        return payload


@dataclass(frozen=True)
class LocalGraphResidualLossOutput:
    total_loss: Any
    components: Mapping[str, Any]
    weights: Mapping[str, float]
    diagnostics: Mapping[str, Any]
    config: LocalGraphResidualLossConfig

    def scalar_components(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for key, value in self.components.items():
            if hasattr(value, "detach"):
                output[str(key)] = float(value.detach().cpu().item())
            else:
                output[str(key)] = float(value)
        return output

    def scalar_diagnostics(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for key, value in self.diagnostics.items():
            if hasattr(value, "detach"):
                if int(value.numel()) == 1:
                    output[str(key)] = float(value.detach().cpu().item())
            elif isinstance(value, (int, float)):
                output[str(key)] = float(value)
        return output


def _as_tensor_1d(value: Any, *, dtype: Any | None = None, device: Any | None = None, field_name: str) -> Any:
    torch = require_torch()
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=dtype or torch.float32, device=device)
    else:
        if dtype is not None or device is not None:
            value = value.to(dtype=dtype or value.dtype, device=device or value.device)
    value = value.reshape(-1)
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{field_name} contains non-finite values")
    return value


def _threshold_tensor(value: Any | None, reference: Any, fallback: float) -> Any:
    torch = require_torch()
    if value is None:
        return torch.tensor(float(fallback), dtype=reference.dtype, device=reference.device)
    if not isinstance(value, torch.Tensor):
        return torch.tensor(float(value), dtype=reference.dtype, device=reference.device)
    return value.detach().to(dtype=reference.dtype, device=reference.device).reshape(())


def _zero_loss(reference: Any) -> Any:
    return reference.sum() * 0.0


def _quantile_detached(values: Any, q: float) -> Any:
    torch = require_torch()
    values = values.detach().reshape(-1)
    if int(values.numel()) == 0:
        raise ValueError("cannot compute quantile on empty tensor")
    sorted_values = torch.sort(values).values
    index = int(round(float(q) * float(int(sorted_values.numel()) - 1)))
    index = min(max(index, 0), int(sorted_values.numel()) - 1)
    return sorted_values[index]


def _top_fraction_mean(values: Any, fraction: float) -> Any:
    values = values.reshape(-1)
    if int(values.numel()) == 0:
        return _zero_loss(values)
    k = max(1, int(math.ceil(float(fraction) * int(values.numel()))))
    return values.topk(k, largest=True).values.mean()


def residual_bce_weights(
    labels: Any,
    baseline_logit: Any,
    *,
    tau50: Any | None = None,
    config: LocalGraphResidualLossConfig | None = None,
) -> Any:
    """Build BCE weights emphasizing baseline boundary mistakes."""

    torch = require_torch()
    config = config or LocalGraphResidualLossConfig()
    labels = _as_tensor_1d(labels, dtype=torch.float32, field_name="labels")
    baseline_logit = _as_tensor_1d(
        baseline_logit,
        dtype=torch.float32,
        device=labels.device,
        field_name="baseline_logit",
    )
    if tuple(labels.shape) != tuple(baseline_logit.shape):
        raise ValueError("labels and baseline_logit must have matching shape")
    tau50_t = _threshold_tensor(tau50, baseline_logit, float(config.default_tau50))
    delta = baseline_logit.detach() - tau50_t
    abs_delta = torch.abs(delta)
    if config.bce_boundary_scale is None:
        scale = torch.clamp(_quantile_detached(abs_delta, 0.25), min=1.0e-3)
    else:
        scale = torch.tensor(float(config.bce_boundary_scale), dtype=baseline_logit.dtype, device=baseline_logit.device)
    near = torch.exp(-abs_delta / scale)
    signal = labels >= 0.5
    background = ~signal
    baseline_fp = background & (baseline_logit.detach() >= tau50_t)
    baseline_fn = signal & (baseline_logit.detach() < tau50_t)
    weights = torch.ones_like(baseline_logit)
    weights = weights + float(config.bce_low_margin_weight) * near
    weights = weights + float(config.bce_baseline_false_positive_weight) * baseline_fp.to(dtype=weights.dtype)
    weights = weights + float(config.bce_baseline_false_negative_weight) * baseline_fn.to(dtype=weights.dtype) * near
    weights = weights + float(config.bce_signal_boundary_weight) * signal.to(dtype=weights.dtype) * near
    if bool(config.normalize_bce_weights):
        weights = weights / torch.clamp(weights.mean().detach(), min=1.0e-6)
    return weights


def weighted_residual_bce_loss(
    fused_logit: Any,
    labels: Any,
    baseline_logit: Any,
    *,
    tau50: Any | None = None,
    config: LocalGraphResidualLossConfig | None = None,
) -> tuple[Any, Any]:
    """Weighted BCE on the fused additive logit."""

    torch = require_torch()
    config = config or LocalGraphResidualLossConfig()
    fused_logit = _as_tensor_1d(fused_logit, dtype=torch.float32, field_name="fused_logit")
    labels = _as_tensor_1d(labels, dtype=fused_logit.dtype, device=fused_logit.device, field_name="labels")
    baseline_logit = _as_tensor_1d(
        baseline_logit,
        dtype=fused_logit.dtype,
        device=fused_logit.device,
        field_name="baseline_logit",
    )
    if tuple(labels.shape) != tuple(fused_logit.shape) or tuple(baseline_logit.shape) != tuple(fused_logit.shape):
        raise ValueError("fused_logit, labels, and baseline_logit must have matching shape")
    weights = residual_bce_weights(labels, baseline_logit, tau50=tau50, config=config).to(
        device=fused_logit.device,
        dtype=fused_logit.dtype,
    )
    per_example = torch.nn.functional.binary_cross_entropy_with_logits(
        fused_logit,
        labels,
        reduction="none",
    )
    return (per_example * weights).mean(), weights


def residual_boundary_masks(
    labels: Any,
    baseline_logit: Any,
    *,
    tau50: Any | None = None,
    config: LocalGraphResidualLossConfig | None = None,
) -> tuple[Any, Any]:
    """Return (signal-boundary mask, hard-background mask)."""

    torch = require_torch()
    config = config or LocalGraphResidualLossConfig()
    labels = _as_tensor_1d(labels, dtype=torch.float32, field_name="labels")
    baseline_logit = _as_tensor_1d(
        baseline_logit,
        dtype=torch.float32,
        device=labels.device,
        field_name="baseline_logit",
    )
    tau50_t = _threshold_tensor(tau50, baseline_logit, float(config.default_tau50))
    signal = labels >= 0.5
    background = ~signal
    signal_mask = torch.zeros_like(signal, dtype=torch.bool)
    if int(signal.sum().item()) > 0:
        signal_scores = baseline_logit.detach()[signal]
        q_low = _quantile_detached(signal_scores, float(config.signal_boundary_quantile_low))
        q_high = _quantile_detached(signal_scores, float(config.signal_boundary_quantile_high))
        if bool((q_high < q_low).item()):
            q_low, q_high = q_high, q_low
        signal_mask = signal & (baseline_logit.detach() >= q_low) & (baseline_logit.detach() <= q_high)
        if int(signal_mask.sum().item()) == 0:
            distances = torch.abs(signal_scores - tau50_t)
            chosen_local = torch.argmin(distances)
            signal_indices = torch.nonzero(signal, as_tuple=False).reshape(-1)
            signal_mask[signal_indices[chosen_local]] = True
    background_mask = background & (baseline_logit.detach() >= tau50_t)
    if int(background_mask.sum().item()) == 0 and int(background.sum().item()) > 0:
        background_scores = baseline_logit.detach()[background]
        k = max(1, int(math.ceil(float(config.hard_background_fraction) * int(background_scores.numel()))))
        local = torch.topk(background_scores, k=k, largest=True).indices
        background_indices = torch.nonzero(background, as_tuple=False).reshape(-1)
        background_mask[background_indices[local]] = True
    return signal_mask, background_mask


def boundary_pairwise_loss(
    fused_logit: Any,
    labels: Any,
    baseline_logit: Any,
    *,
    tau50: Any | None = None,
    config: LocalGraphResidualLossConfig | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Pairwise rank loss pushing hard QCD below Hgg boundary jets."""

    torch = require_torch()
    config = config or LocalGraphResidualLossConfig()
    fused_logit = _as_tensor_1d(fused_logit, dtype=torch.float32, field_name="fused_logit")
    labels = _as_tensor_1d(labels, dtype=fused_logit.dtype, device=fused_logit.device, field_name="labels")
    baseline_logit = _as_tensor_1d(
        baseline_logit,
        dtype=fused_logit.dtype,
        device=fused_logit.device,
        field_name="baseline_logit",
    )
    signal_mask, background_mask = residual_boundary_masks(labels, baseline_logit, tau50=tau50, config=config)
    n_signal = int(signal_mask.sum().item())
    n_background = int(background_mask.sum().item())
    if n_signal == 0 or n_background == 0:
        return _zero_loss(fused_logit), {
            "boundary_signal_count": torch.tensor(float(n_signal), device=fused_logit.device),
            "hard_background_count": torch.tensor(float(n_background), device=fused_logit.device),
            "pair_count": torch.zeros((), dtype=fused_logit.dtype, device=fused_logit.device),
        }
    signal_scores = fused_logit[signal_mask]
    background_scores = fused_logit[background_mask]
    pair_terms = torch.nn.functional.softplus(
        (background_scores[:, None] - signal_scores[None, :]) / float(config.pairwise_temperature)
    )
    per_background = pair_terms.mean(dim=1)
    loss = _top_fraction_mean(per_background, float(config.cvar_top_fraction))
    return loss, {
        "boundary_signal_count": torch.tensor(float(n_signal), dtype=fused_logit.dtype, device=fused_logit.device),
        "hard_background_count": torch.tensor(float(n_background), dtype=fused_logit.dtype, device=fused_logit.device),
        "pair_count": torch.tensor(float(n_signal * n_background), dtype=fused_logit.dtype, device=fused_logit.device),
        "pairwise_term_mean": pair_terms.detach().mean(),
        "pairwise_term_max": pair_terms.detach().amax(),
    }


def soft_fpr50_loss(
    fused_logit: Any,
    labels: Any,
    *,
    tau50: Any | None = None,
    config: LocalGraphResidualLossConfig | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Soft FPR@50 surrogate using the frozen baseline signal threshold."""

    torch = require_torch()
    config = config or LocalGraphResidualLossConfig()
    fused_logit = _as_tensor_1d(fused_logit, dtype=torch.float32, field_name="fused_logit")
    labels = _as_tensor_1d(labels, dtype=fused_logit.dtype, device=fused_logit.device, field_name="labels")
    tau50_t = _threshold_tensor(tau50, fused_logit, float(config.default_tau50))
    background = labels < 0.5
    if int(background.sum().item()) == 0:
        return _zero_loss(fused_logit), {
            "soft_fpr_background_count": torch.zeros((), dtype=fused_logit.dtype, device=fused_logit.device),
            "soft_fpr_tau50": tau50_t.detach(),
        }
    background_scores = fused_logit[background]
    values = torch.sigmoid((background_scores - tau50_t) / float(config.soft_fpr_epsilon))
    return values.mean(), {
        "soft_fpr_background_count": torch.tensor(
            float(int(background.sum().item())),
            dtype=fused_logit.dtype,
            device=fused_logit.device,
        ),
        "soft_fpr_tau50": tau50_t.detach(),
        "soft_fpr_surrogate_mean": values.detach().mean(),
    }


def residual_l2_loss(residual_logit: Any, alpha: Any | None = None) -> Any:
    """Penalty on the actual additive correction alpha * residual."""

    torch = require_torch()
    residual_logit = _as_tensor_1d(residual_logit, dtype=torch.float32, field_name="residual_logit")
    if alpha is None:
        correction = residual_logit
    else:
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(float(alpha), dtype=residual_logit.dtype, device=residual_logit.device)
        else:
            alpha = alpha.to(dtype=residual_logit.dtype, device=residual_logit.device)
        correction = alpha.reshape(()) * residual_logit
    return torch.mean(correction * correction)


def compute_local_graph_residual_loss(
    *,
    fused_logit: Any,
    labels: Any,
    baseline_logit: Any,
    residual_logit: Any,
    alpha: Any | None = None,
    tau50: Any | None = None,
    tau30: Any | None = None,
    config: LocalGraphResidualLossConfig | None = None,
) -> LocalGraphResidualLossOutput:
    """Compute the selected residual-expert objective ladder."""

    del tau30
    config = config or LocalGraphResidualLossConfig()
    fused_logit = _as_tensor_1d(fused_logit, dtype=require_torch().float32, field_name="fused_logit")
    labels = _as_tensor_1d(labels, dtype=fused_logit.dtype, device=fused_logit.device, field_name="labels")
    baseline_logit = _as_tensor_1d(
        baseline_logit,
        dtype=fused_logit.dtype,
        device=fused_logit.device,
        field_name="baseline_logit",
    )
    residual_logit = _as_tensor_1d(
        residual_logit,
        dtype=fused_logit.dtype,
        device=fused_logit.device,
        field_name="residual_logit",
    )
    bce, bce_weights = weighted_residual_bce_loss(
        fused_logit,
        labels,
        baseline_logit,
        tau50=tau50,
        config=config,
    )
    pairwise, pairwise_diag = boundary_pairwise_loss(
        fused_logit,
        labels,
        baseline_logit,
        tau50=tau50,
        config=config,
    )
    soft_fpr, soft_fpr_diag = soft_fpr50_loss(fused_logit, labels, tau50=tau50, config=config)
    l2 = residual_l2_loss(residual_logit, alpha=alpha)
    components = {
        "weighted_bce": bce,
        "boundary_pairwise": pairwise,
        "soft_fpr50": soft_fpr,
        "residual_l2": l2,
    }
    weights = config.active_weights()
    total = _zero_loss(fused_logit)
    for key, value in components.items():
        total = total + float(weights.get(key, 0.0)) * value
    diagnostics = {
        "step": LOCAL_GRAPH_RESIDUAL_LOSS_STEP,
        "contract": LOCAL_GRAPH_RESIDUAL_LOSS_CONTRACT,
        "mode": config.mode,
        "bce_weight_mean": bce_weights.detach().mean(),
        "bce_weight_min": bce_weights.detach().amin(),
        "bce_weight_max": bce_weights.detach().amax(),
        "fused_logit_mean": fused_logit.detach().mean(),
        "baseline_logit_mean": baseline_logit.detach().mean(),
        "residual_abs_mean": residual_logit.detach().abs().mean(),
        **pairwise_diag,
        **soft_fpr_diag,
    }
    if alpha is not None and hasattr(alpha, "detach"):
        diagnostics["alpha"] = alpha.detach().reshape(())
    return LocalGraphResidualLossOutput(
        total_loss=total,
        components=components,
        weights=weights,
        diagnostics=diagnostics,
        config=config,
    )


def compute_local_graph_residual_loss_from_output(
    output: Any,
    labels: Any,
    *,
    tau50: Any | None = None,
    tau30: Any | None = None,
    config: LocalGraphResidualLossConfig | None = None,
) -> LocalGraphResidualLossOutput:
    """Convenience wrapper for ``LocalGraphResidualExpertOutput``."""

    return compute_local_graph_residual_loss(
        fused_logit=output.fused_logit,
        labels=labels,
        baseline_logit=output.baseline_logit,
        residual_logit=output.residual_logit,
        alpha=output.alpha,
        tau50=tau50,
        tau30=tau30,
        config=config,
    )


def select_alpha_shrinkage(
    *,
    labels: Sequence[int] | np.ndarray,
    baseline_logit: Sequence[float] | np.ndarray,
    residual_logit: Sequence[float] | np.ndarray,
    alpha_grid: Sequence[float] = LOCAL_GRAPH_RESIDUAL_ALPHA_GRID,
    target_signal_efficiency: float = 0.50,
) -> dict[str, Any]:
    """Select alpha on validation labels by minimizing FPR at target signal efficiency."""

    labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    baseline_np = np.asarray(baseline_logit, dtype=np.float64).reshape(-1)
    residual_np = np.asarray(residual_logit, dtype=np.float64).reshape(-1)
    if labels_np.shape != baseline_np.shape or labels_np.shape != residual_np.shape:
        raise ValueError("labels, baseline_logit, and residual_logit must have matching shape")
    if not np.isin(labels_np, [0, 1]).all():
        raise ValueError("labels must be binary encoded as 0/1")
    if not np.isfinite(baseline_np).all() or not np.isfinite(residual_np).all():
        raise FloatingPointError("baseline/residual logits must be finite")
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for alpha in alpha_grid:
        alpha_f = float(alpha)
        if not np.isfinite(alpha_f) or alpha_f < 0.0:
            raise ValueError("alpha_grid values must be finite and nonnegative")
        scores = baseline_np + alpha_f * residual_np
        op = operating_point_from_scores(labels_np, scores, float(target_signal_efficiency))
        preds = (scores >= 0.0).astype(np.int64)
        accuracy = float(np.mean(preds == labels_np)) if labels_np.size else float("nan")
        fpr = float(op["false_positive_rate"])
        signal_eff = float(op["signal_efficiency"])
        row = {
            "alpha": alpha_f,
            "false_positive_rate": fpr,
            "signal_efficiency": signal_eff,
            "threshold": float(op["threshold"]),
            "background_rejection": float(op["background_rejection"]),
            "accuracy": accuracy,
            "score_mean": float(np.mean(scores)) if scores.size else float("nan"),
            "score_std": float(np.std(scores)) if scores.size else float("nan"),
        }
        rows.append(row)
        score = (fpr, -signal_eff, -accuracy, abs(alpha_f))
        if best_row is None:
            best_row = {**row, "_selection_tuple": score}
        else:
            if score < best_row["_selection_tuple"]:
                best_row = {**row, "_selection_tuple": score}
    if best_row is None:
        raise ValueError("alpha_grid cannot be empty")
    best = dict(best_row)
    best.pop("_selection_tuple", None)
    baseline_op = operating_point_from_scores(labels_np, baseline_np, float(target_signal_efficiency))
    return {
        "step": LOCAL_GRAPH_RESIDUAL_LOSS_STEP,
        "contract": LOCAL_GRAPH_RESIDUAL_LOSS_CONTRACT,
        "target_signal_efficiency": float(target_signal_efficiency),
        "alpha_grid": [float(alpha) for alpha in alpha_grid],
        "rows": rows,
        "best": best,
        "selected_alpha": float(best["alpha"]),
        "selected_fpr": float(best["false_positive_rate"]),
        "baseline_fpr": float(baseline_op["false_positive_rate"]),
        "delta_fpr_vs_baseline": float(best["false_positive_rate"]) - float(baseline_op["false_positive_rate"]),
        "collapsed_to_zero": bool(abs(float(best["alpha"])) <= 1.0e-12),
    }


__all__ = [
    "LOCAL_GRAPH_RESIDUAL_ALPHA_GRID",
    "LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE",
    "LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR",
    "LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR",
    "LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR_ALPHA_SHRINK",
    "LOCAL_GRAPH_RESIDUAL_LOSS_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_LOSS_MODES",
    "LOCAL_GRAPH_RESIDUAL_LOSS_STEP",
    "LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE",
    "LOCAL_GRAPH_RESIDUAL_REPORT_POLICY_ALPHA_SHRINK",
    "LocalGraphResidualLossConfig",
    "LocalGraphResidualLossOutput",
    "boundary_pairwise_loss",
    "compute_local_graph_residual_loss",
    "compute_local_graph_residual_loss_from_output",
    "normalize_local_graph_residual_loss_mode",
    "residual_bce_weights",
    "residual_boundary_masks",
    "residual_l2_loss",
    "select_alpha_shrinkage",
    "soft_fpr50_loss",
    "weighted_residual_bce_loss",
]

"""Loss ladder for the local-graph residual expert V2.

The V2 losses operate on an exact frozen HLT ParT baseline score plus a small
learned correction from :class:`LocalGraphResidualExpertV2`.  Ladder E is a
report-time validation-shrinkage policy, not a training loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .residual_v2_protocol import (
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
)


LOCAL_GRAPH_RESIDUAL_V2_LOSS_STEP = "local_graph_residual_expert_v2_step7_losses"
LOCAL_GRAPH_RESIDUAL_V2_LOSS_CONTRACT = "local_graph_residual_expert_v2_losses_v1"
LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES = (
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
)


def normalize_local_graph_residual_v2_loss_mode(value: str) -> str:
    """Normalize A-D V2 loss aliases and reject E as a training mode."""

    clean = str(value).strip().lower().replace("-", "_")
    aliases = {
        "a": LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
        "weighted_bce": LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
        "residual_v2_weighted_bce": LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
        LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE: LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
        "b": LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE,
        "boundary_pairwise": LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE,
        "residual_v2_boundary_pairwise": LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE,
        LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE: LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE,
        "c": LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
        "boundary_pairwise_bce_anchor": LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
        "residual_v2_boundary_pairwise_bce_anchor": LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
        LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR: (
            LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR
        ),
        "d": LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
        "boundary_pairwise_soft_fpr_bce_anchor": (
            LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR
        ),
        "residual_v2_boundary_pairwise_soft_fpr_bce_anchor": (
            LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR
        ),
        LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR: (
            LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR
        ),
    }
    if clean in {
        "e",
        "alpha_shrink",
        "gamma_shrink",
        "validation_shrinkage",
        "residual_v2_boundary_pairwise_soft_fpr_bce_anchor_alpha_shrink",
    }:
        raise ValueError(
            "V2 ladder E is not a training loss. Train A/C/D and apply validation shrinkage in reporting."
        )
    if clean not in aliases:
        raise ValueError(f"V2 residual loss mode must be one of {LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES}, got {value!r}")
    return aliases[clean]


def _optional_positive_float(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be positive when provided")
    return value


@dataclass(frozen=True)
class LocalGraphResidualV2LossConfig:
    """Configuration for the Step 7 V2 objective ladder."""

    mode: str = LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE
    bce_anchor_weight: float = 0.05
    soft_fpr_weight: float = 0.25
    correction_l2_weight: float = 1.0e-4
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
        object.__setattr__(self, "mode", normalize_local_graph_residual_v2_loss_mode(self.mode))
        for field_name in (
            "bce_anchor_weight",
            "soft_fpr_weight",
            "correction_l2_weight",
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
        if self.mode == LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE:
            return {
                "weighted_bce": float(self.weighted_bce_weight),
                "boundary_pairwise": 0.0,
                "soft_fpr50": 0.0,
                "correction_l2": float(self.correction_l2_weight),
            }
        if self.mode == LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE:
            return {
                "weighted_bce": 0.0,
                "boundary_pairwise": float(self.pairwise_weight),
                "soft_fpr50": 0.0,
                "correction_l2": float(self.correction_l2_weight),
            }
        if self.mode == LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR:
            return {
                "weighted_bce": float(self.bce_anchor_weight),
                "boundary_pairwise": float(self.pairwise_weight),
                "soft_fpr50": 0.0,
                "correction_l2": float(self.correction_l2_weight),
            }
        return {
            "weighted_bce": float(self.bce_anchor_weight),
            "boundary_pairwise": float(self.pairwise_weight),
            "soft_fpr50": float(self.soft_fpr_weight),
            "correction_l2": float(self.correction_l2_weight),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_weights"] = self.active_weights()
        payload["step"] = LOCAL_GRAPH_RESIDUAL_V2_LOSS_STEP
        payload["contract"] = LOCAL_GRAPH_RESIDUAL_V2_LOSS_CONTRACT
        return payload


@dataclass(frozen=True)
class LocalGraphResidualV2LossOutput:
    total_loss: Any
    components: Mapping[str, Any]
    weights: Mapping[str, float]
    diagnostics: Mapping[str, Any]
    config: LocalGraphResidualV2LossConfig

    def scalar_components(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for key, value in self.components.items():
            output[str(key)] = float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
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


def residual_v2_bce_weights(
    labels: Any,
    baseline_logit: Any,
    *,
    tau50: Any | None = None,
    config: LocalGraphResidualV2LossConfig | None = None,
) -> Any:
    """Build BCE weights that emphasize baseline boundary mistakes."""

    torch = require_torch()
    config = config or LocalGraphResidualV2LossConfig()
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


def weighted_residual_v2_bce_loss(
    fused_logit: Any,
    labels: Any,
    baseline_logit: Any,
    *,
    tau50: Any | None = None,
    config: LocalGraphResidualV2LossConfig | None = None,
) -> tuple[Any, Any]:
    """Weighted BCE on the fused additive V2 logit."""

    torch = require_torch()
    config = config or LocalGraphResidualV2LossConfig()
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
    weights = residual_v2_bce_weights(labels, baseline_logit, tau50=tau50, config=config).to(
        device=fused_logit.device,
        dtype=fused_logit.dtype,
    )
    per_example = torch.nn.functional.binary_cross_entropy_with_logits(fused_logit, labels, reduction="none")
    return (per_example * weights).mean(), weights


def residual_v2_boundary_masks(
    labels: Any,
    baseline_logit: Any,
    *,
    tau50: Any | None = None,
    config: LocalGraphResidualV2LossConfig | None = None,
) -> tuple[Any, Any]:
    """Return (signal-boundary mask, hard-background mask)."""

    torch = require_torch()
    config = config or LocalGraphResidualV2LossConfig()
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


def boundary_pairwise_v2_loss(
    fused_logit: Any,
    labels: Any,
    baseline_logit: Any,
    *,
    tau50: Any | None = None,
    config: LocalGraphResidualV2LossConfig | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Pairwise rank loss pushing hard QCD below boundary Hgg jets."""

    torch = require_torch()
    config = config or LocalGraphResidualV2LossConfig()
    fused_logit = _as_tensor_1d(fused_logit, dtype=torch.float32, field_name="fused_logit")
    labels = _as_tensor_1d(labels, dtype=fused_logit.dtype, device=fused_logit.device, field_name="labels")
    baseline_logit = _as_tensor_1d(
        baseline_logit,
        dtype=fused_logit.dtype,
        device=fused_logit.device,
        field_name="baseline_logit",
    )
    signal_mask, background_mask = residual_v2_boundary_masks(labels, baseline_logit, tau50=tau50, config=config)
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


def soft_fpr50_v2_loss(
    fused_logit: Any,
    labels: Any,
    *,
    tau50: Any | None = None,
    config: LocalGraphResidualV2LossConfig | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Soft FPR@50 surrogate for background scores around train-derived tau50."""

    torch = require_torch()
    config = config or LocalGraphResidualV2LossConfig()
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


def correction_l2_v2_loss(correction_logit: Any) -> Any:
    correction_logit = _as_tensor_1d(correction_logit, dtype=require_torch().float32, field_name="correction_logit")
    return (correction_logit * correction_logit).mean()


def compute_local_graph_residual_v2_loss(
    *,
    fused_logit: Any,
    labels: Any,
    baseline_logit: Any,
    correction_logit: Any,
    tau50: Any | None = None,
    tau30: Any | None = None,
    config: LocalGraphResidualV2LossConfig | None = None,
) -> LocalGraphResidualV2LossOutput:
    """Compute the selected V2 residual-expert objective."""

    del tau30
    torch = require_torch()
    config = config or LocalGraphResidualV2LossConfig()
    fused_logit = _as_tensor_1d(fused_logit, dtype=torch.float32, field_name="fused_logit")
    labels = _as_tensor_1d(labels, dtype=fused_logit.dtype, device=fused_logit.device, field_name="labels")
    baseline_logit = _as_tensor_1d(
        baseline_logit,
        dtype=fused_logit.dtype,
        device=fused_logit.device,
        field_name="baseline_logit",
    )
    correction_logit = _as_tensor_1d(
        correction_logit,
        dtype=fused_logit.dtype,
        device=fused_logit.device,
        field_name="correction_logit",
    )
    if tuple(labels.shape) != tuple(fused_logit.shape) or tuple(baseline_logit.shape) != tuple(fused_logit.shape):
        raise ValueError("fused_logit, labels, and baseline_logit must have matching shape")
    if tuple(correction_logit.shape) != tuple(fused_logit.shape):
        raise ValueError("correction_logit must have the same shape as fused_logit")

    bce, bce_weights = weighted_residual_v2_bce_loss(
        fused_logit,
        labels,
        baseline_logit,
        tau50=tau50,
        config=config,
    )
    pairwise, pairwise_diag = boundary_pairwise_v2_loss(
        fused_logit,
        labels,
        baseline_logit,
        tau50=tau50,
        config=config,
    )
    soft_fpr, soft_fpr_diag = soft_fpr50_v2_loss(fused_logit, labels, tau50=tau50, config=config)
    l2 = correction_l2_v2_loss(correction_logit)
    components = {
        "weighted_bce": bce,
        "boundary_pairwise": pairwise,
        "soft_fpr50": soft_fpr,
        "correction_l2": l2,
    }
    weights = config.active_weights()
    total = _zero_loss(fused_logit)
    for key, value in components.items():
        total = total + float(weights.get(key, 0.0)) * value
    diagnostics = {
        "step": LOCAL_GRAPH_RESIDUAL_V2_LOSS_STEP,
        "contract": LOCAL_GRAPH_RESIDUAL_V2_LOSS_CONTRACT,
        "mode": config.mode,
        "bce_weight_mean": bce_weights.detach().mean(),
        "bce_weight_min": bce_weights.detach().amin(),
        "bce_weight_max": bce_weights.detach().amax(),
        "fused_logit_mean": fused_logit.detach().mean(),
        "baseline_logit_mean": baseline_logit.detach().mean(),
        "correction_abs_mean": correction_logit.detach().abs().mean(),
        **pairwise_diag,
        **soft_fpr_diag,
    }
    return LocalGraphResidualV2LossOutput(
        total_loss=total,
        components=components,
        weights=weights,
        diagnostics=diagnostics,
        config=config,
    )


def compute_local_graph_residual_v2_loss_from_output(
    output: Any,
    labels: Any,
    *,
    tau50: Any | None = None,
    tau30: Any | None = None,
    config: LocalGraphResidualV2LossConfig | None = None,
) -> LocalGraphResidualV2LossOutput:
    return compute_local_graph_residual_v2_loss(
        fused_logit=output.fused_logit,
        labels=labels,
        baseline_logit=output.baseline_logit,
        correction_logit=output.correction_logit,
        tau50=tau50,
        tau30=tau30,
        config=config,
    )


__all__ = [
    "LOCAL_GRAPH_RESIDUAL_V2_LOSS_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES",
    "LOCAL_GRAPH_RESIDUAL_V2_LOSS_STEP",
    "LocalGraphResidualV2LossConfig",
    "LocalGraphResidualV2LossOutput",
    "boundary_pairwise_v2_loss",
    "compute_local_graph_residual_v2_loss",
    "compute_local_graph_residual_v2_loss_from_output",
    "correction_l2_v2_loss",
    "normalize_local_graph_residual_v2_loss_mode",
    "residual_v2_bce_weights",
    "residual_v2_boundary_masks",
    "soft_fpr50_v2_loss",
    "weighted_residual_v2_bce_loss",
]

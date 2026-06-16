"""Losses for teacher-logit global transformer reconstruction.

Step 4 keeps the training objective explicit:

``teacher_KL + CE + correction_budget + weak_jet_summary_loss``

The functions here are intentionally small and auditable.  They do not own the
data loader, optimizer, or checkpoint logic; they only compute one differentiable
training step from a reconstructed soft view and frozen-teacher logits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .global_transformer import EPS, physical_energy_floor
from .views import SoftReconstructedView


@dataclass
class TeacherLogitRecoLossConfig:
    """Weights and temperatures for the first teacher-logit objective."""

    teacher_kl_weight: float = 1.0
    ce_weight: float = 0.3
    correction_budget_weight: float = 0.01
    jet_summary_weight: float = 0.05
    temperature: float = 2.0
    parent_delta_budget_weight: float = 1.0
    parent_weight_budget_weight: float = 0.25
    extra_weight_budget_weight: float = 0.25
    extra_pt_fraction_budget_weight: float = 0.25
    jet_summary_huber_beta: float = 0.2
    summary_log_scale: bool = True
    aggressive_extra_budget_weight: float = 0.0
    aggressive_parent_weight_budget_weight: float = 0.0
    aggressive_global_calibration_budget_weight: float = 0.0
    max_total_extra_pt_fraction: float = 0.50
    extra_count_budget_weight: float = 0.05
    min_parent_weight_fraction: float = 0.25
    parent_prune_budget_weight: float = 0.05

    def __post_init__(self) -> None:
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        for name in (
            "teacher_kl_weight",
            "ce_weight",
            "correction_budget_weight",
            "jet_summary_weight",
            "parent_delta_budget_weight",
            "parent_weight_budget_weight",
            "extra_weight_budget_weight",
            "extra_pt_fraction_budget_weight",
            "jet_summary_huber_beta",
            "aggressive_extra_budget_weight",
            "aggressive_parent_weight_budget_weight",
            "aggressive_global_calibration_budget_weight",
            "max_total_extra_pt_fraction",
            "extra_count_budget_weight",
            "min_parent_weight_fraction",
            "parent_prune_budget_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.min_parent_weight_fraction) > 1.0:
            raise ValueError("min_parent_weight_fraction must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "TeacherLogitRecoLossConfig" | None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))

    @classmethod
    def aggressive_defaults(cls, **overrides: Any) -> "TeacherLogitRecoLossConfig":
        """Return first-pass opt-in weights for aggressive reconstructors.

        The base dataclass keeps the new losses disabled to preserve existing
        conservative training behavior.  Aggressive training CLIs can use this
        constructor, or pass equivalent explicit weights, once the aggressive
        reconstructor architectures are wired in.
        """

        payload = {
            "teacher_kl_weight": 1.0,
            "ce_weight": 0.5,
            "correction_budget_weight": 0.02,
            "jet_summary_weight": 0.05,
            "aggressive_extra_budget_weight": 0.05,
            "aggressive_parent_weight_budget_weight": 0.02,
            "aggressive_global_calibration_budget_weight": 0.02,
        }
        payload.update(overrides)
        return cls(**payload)


@dataclass
class TeacherLogitRecoLossOutput:
    """Structured output from one loss computation."""

    total_loss: Any
    components: Dict[str, Any]
    weighted_components: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def detached_float_dict(self) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        for prefix, values in (
            ("component", self.components),
            ("weighted", self.weighted_components),
            ("metric", self.metrics),
        ):
            for name, value in values.items():
                payload[f"{prefix}_{name}"] = detach_float(value)
        payload["total_loss"] = detach_float(self.total_loss)
        return payload


def detach_float(value: Any) -> float:
    try:
        return float(value.detach().cpu().item())
    except AttributeError:
        return float(value)


def _safe_mean(value):
    torch = require_torch()
    if value.numel() == 0:
        return torch.zeros((), dtype=value.dtype, device=value.device)
    return value.mean()


def masked_mean(value, mask):
    torch = require_torch()
    mask = mask.bool()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(dtype=value.dtype).expand_as(value)
    denom = torch.clamp(weights.sum(), min=1.0)
    return (value * weights).sum() / denom


def teacher_kl_loss(offline_logits, reco_logits, *, temperature: float = 2.0):
    """Temperature-scaled KL from offline-teacher distribution to reco distribution."""

    torch = require_torch()
    functional = torch.nn.functional
    temp = float(temperature)
    target_probs = functional.softmax(offline_logits.detach() / temp, dim=-1)
    reco_log_probs = functional.log_softmax(reco_logits / temp, dim=-1)
    return (temp * temp) * functional.kl_div(reco_log_probs, target_probs, reduction="batchmean")


def teacher_cross_entropy_loss(reco_logits, labels):
    torch = require_torch()
    labels = labels.to(device=reco_logits.device, dtype=torch.long)
    return torch.nn.functional.cross_entropy(reco_logits, labels)


def weighted_jet_summary(tokens, mask, *, weights=None):
    """Differentiable coarse jet summaries from token kinematics."""

    torch = require_torch()
    mask = mask.bool()
    if weights is None:
        weights = torch.ones_like(tokens[:, :, 0])
    weights = torch.clamp(weights.float(), min=0.0)
    effective = mask.float() * weights

    pt = torch.clamp(tokens[:, :, 0], min=0.0) * effective
    eta = tokens[:, :, 1]
    phi = tokens[:, :, 2]
    energy = torch.clamp(tokens[:, :, 3], min=0.0) * effective

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(torch.clamp(eta, -5.0, 5.0))
    jet_px = px.sum(dim=1)
    jet_py = py.sum(dim=1)
    jet_pz = pz.sum(dim=1)
    jet_energy = energy.sum(dim=1)
    jet_pt = torch.sqrt(torch.clamp(jet_px * jet_px + jet_py * jet_py, min=0.0))
    jet_mass2 = jet_energy * jet_energy - jet_px * jet_px - jet_py * jet_py - jet_pz * jet_pz
    jet_mass = torch.sqrt(torch.clamp(jet_mass2, min=0.0))
    weighted_multiplicity = effective.sum(dim=1)
    leading_pt_fraction = torch.where(
        jet_pt > EPS,
        torch.max(pt, dim=1).values / torch.clamp(jet_pt, min=EPS),
        torch.zeros_like(jet_pt),
    )
    return {
        "jet_pt": jet_pt,
        "jet_energy": jet_energy,
        "jet_mass": jet_mass,
        "weighted_multiplicity": weighted_multiplicity,
        "leading_pt_fraction": leading_pt_fraction,
    }


def jet_summary_tensor(summary: Mapping[str, Any], *, log_scale: bool = True):
    torch = require_torch()
    if log_scale:
        pieces = [
            torch.log1p(torch.clamp(summary["jet_pt"], min=0.0)),
            torch.log1p(torch.clamp(summary["jet_energy"], min=0.0)),
            torch.log1p(torch.clamp(summary["jet_mass"], min=0.0)),
            torch.log1p(torch.clamp(summary["weighted_multiplicity"], min=0.0)),
            torch.clamp(summary["leading_pt_fraction"], 0.0, 10.0),
        ]
    else:
        pieces = [
            summary["jet_pt"],
            summary["jet_energy"],
            summary["jet_mass"],
            summary["weighted_multiplicity"],
            summary["leading_pt_fraction"],
        ]
    return torch.stack(pieces, dim=1)


def weak_jet_summary_loss(
    reco_view: SoftReconstructedView,
    offline_tokens,
    offline_mask,
    *,
    config: TeacherLogitRecoLossConfig | Mapping[str, Any] | None = None,
):
    """Weakly anchor reconstructed jets to offline coarse summaries."""

    torch = require_torch()
    cfg = TeacherLogitRecoLossConfig.from_mapping(config)
    offline_tokens = offline_tokens.to(device=reco_view.tokens.device, dtype=torch.float32)
    offline_mask = offline_mask.to(device=reco_view.tokens.device, dtype=torch.bool)
    reco_summary = weighted_jet_summary(reco_view.tokens, reco_view.mask, weights=reco_view.weights)
    offline_summary = weighted_jet_summary(offline_tokens, offline_mask, weights=None)
    reco_tensor = jet_summary_tensor(reco_summary, log_scale=bool(cfg.summary_log_scale))
    offline_tensor = jet_summary_tensor(offline_summary, log_scale=bool(cfg.summary_log_scale))
    return torch.nn.functional.smooth_l1_loss(
        reco_tensor,
        offline_tensor.detach(),
        beta=float(cfg.jet_summary_huber_beta),
        reduction="mean",
    )


def correction_budget_loss(
    reco_view: SoftReconstructedView,
    *,
    config: TeacherLogitRecoLossConfig | Mapping[str, Any] | None = None,
) -> tuple[Any, Dict[str, Any]]:
    """Penalize excessive parent edits and extra candidate usage."""

    torch = require_torch()
    cfg = TeacherLogitRecoLossConfig.from_mapping(config)
    aux = dict(reco_view.aux)
    parent_delta = aux.get("parent_delta")
    parent_mask = aux.get("sanitized_hlt_mask")
    parent_weights = aux.get("parent_weights")
    extra_weights = aux.get("extra_weights")
    extra_tokens = aux.get("extra_tokens")
    hlt_tokens = aux.get("sanitized_hlt_tokens")
    hlt_mask = aux.get("sanitized_hlt_mask")

    zero = reco_view.tokens.new_zeros(())
    if parent_delta is None or parent_mask is None:
        parent_delta_term = zero
    else:
        parent_delta_term = masked_mean(parent_delta * parent_delta, parent_mask)

    if parent_weights is None or parent_mask is None:
        parent_weight_term = zero
    else:
        parent_weight_term = masked_mean((parent_weights - 1.0) ** 2, parent_mask)

    extra_weight_term = _safe_mean(extra_weights.sum(dim=1)) if extra_weights is not None else zero

    if extra_tokens is not None and extra_weights is not None and hlt_tokens is not None and hlt_mask is not None:
        extra_pt = (torch.clamp(extra_tokens[:, :, 0], min=0.0) * torch.clamp(extra_weights, min=0.0)).sum(dim=1)
        hlt_pt = (torch.clamp(hlt_tokens[:, :, 0], min=0.0) * hlt_mask.float()).sum(dim=1)
        extra_pt_fraction_term = torch.mean(extra_pt / torch.clamp(hlt_pt, min=EPS))
    else:
        extra_pt_fraction_term = zero

    components = {
        "parent_delta": parent_delta_term,
        "parent_weight": parent_weight_term,
        "extra_weight": extra_weight_term,
        "extra_pt_fraction": extra_pt_fraction_term,
    }
    total = (
        float(cfg.parent_delta_budget_weight) * parent_delta_term
        + float(cfg.parent_weight_budget_weight) * parent_weight_term
        + float(cfg.extra_weight_budget_weight) * extra_weight_term
        + float(cfg.extra_pt_fraction_budget_weight) * extra_pt_fraction_term
    )
    return total, components


def extra_budget_loss(
    reco_view: SoftReconstructedView,
    *,
    config: TeacherLogitRecoLossConfig | Mapping[str, Any] | None = None,
) -> tuple[Any, Dict[str, Any]]:
    """Budget aggressive generated-particle capacity.

    This is intentionally separate from the legacy correction budget.  It uses
    the explicit per-jet tensors produced by ``AggressiveSoftViewHead`` when
    available, and falls back to recomputing from extra tokens/weights for older
    soft views.  Missing extra candidates produce a differentiable zero.
    """

    torch = require_torch()
    cfg = TeacherLogitRecoLossConfig.from_mapping(config)
    aux = dict(reco_view.aux)
    zero = reco_view.tokens.new_zeros(())

    extra_weights = aux.get("extra_weights")
    extra_tokens = aux.get("extra_tokens")
    hlt_tokens = aux.get("sanitized_hlt_tokens")
    hlt_mask = aux.get("sanitized_hlt_mask")

    extra_weight_sum = aux.get("extra_weight_sum")
    if extra_weight_sum is None:
        if extra_weights is None:
            extra_weight_sum = reco_view.tokens.new_zeros((int(reco_view.tokens.shape[0]),))
        else:
            extra_weight_sum = torch.clamp(extra_weights, min=0.0, max=1.0).sum(dim=1)

    extra_pt_fraction = aux.get("extra_pt_fraction")
    if extra_pt_fraction is None:
        if extra_tokens is None or extra_weights is None or hlt_tokens is None or hlt_mask is None:
            extra_pt_fraction = reco_view.tokens.new_zeros((int(reco_view.tokens.shape[0]),))
        else:
            extra_pt = (torch.clamp(extra_tokens[:, :, 0], min=0.0) * torch.clamp(extra_weights, min=0.0, max=1.0)).sum(dim=1)
            parent_pt = (torch.clamp(hlt_tokens[:, :, 0], min=0.0) * hlt_mask.float()).sum(dim=1)
            extra_pt_fraction = extra_pt / torch.clamp(parent_pt, min=EPS)

    if int(extra_pt_fraction.numel()) == 0:
        excess = zero
        count_term = zero
        mean_fraction = zero
    else:
        excess = torch.mean(torch.relu(extra_pt_fraction - float(cfg.max_total_extra_pt_fraction)) ** 2)
        count_term = _safe_mean(extra_weight_sum)
        mean_fraction = _safe_mean(extra_pt_fraction)

    components = {
        "extra_pt_fraction_excess": excess,
        "extra_weight_sum": count_term,
        "extra_pt_fraction_mean": mean_fraction,
    }
    total = excess + float(cfg.extra_count_budget_weight) * count_term
    return total, components


def parent_weight_budget_loss(
    reco_view: SoftReconstructedView,
    *,
    config: TeacherLogitRecoLossConfig | Mapping[str, Any] | None = None,
) -> tuple[Any, Dict[str, Any]]:
    """Budget aggressive parent pruning/reweighting.

    The keep term prevents collapse to all-zero parent weights.  The prune term
    is intentionally mild and records how far the model moves from preserving
    the original HLT constituents.
    """

    torch = require_torch()
    cfg = TeacherLogitRecoLossConfig.from_mapping(config)
    aux = dict(reco_view.aux)
    parent_weights = aux.get("parent_weights")
    parent_mask = aux.get("sanitized_hlt_mask")
    zero = reco_view.tokens.new_zeros(())
    if parent_weights is None or parent_mask is None:
        return zero, {"parent_keep": zero, "parent_prune": zero, "parent_weight_fraction_mean": zero}

    parent_mask = parent_mask.bool()
    valid = parent_mask.float()
    parent_count = valid.sum(dim=1)
    parent_weight_sum = (torch.clamp(parent_weights, min=0.0, max=1.0) * valid).sum(dim=1)
    parent_weight_fraction = parent_weight_sum / torch.clamp(parent_count, min=1.0)
    keep = torch.mean(torch.relu(float(cfg.min_parent_weight_fraction) - parent_weight_fraction) ** 2)
    prune = masked_mean((parent_weights - 1.0) ** 2, parent_mask)
    components = {
        "parent_keep": keep,
        "parent_prune": prune,
        "parent_weight_fraction_mean": _safe_mean(parent_weight_fraction),
    }
    total = keep + float(cfg.parent_prune_budget_weight) * prune
    return total, components


def global_calibration_budget_loss(
    reco_view: SoftReconstructedView,
    *,
    config: TeacherLogitRecoLossConfig | Mapping[str, Any] | None = None,
) -> tuple[Any, Dict[str, Any]]:
    """Budget aggressive jet-level global calibration parameters."""

    del config
    aux = dict(reco_view.aux)
    global_correction = aux.get("global_correction") or aux.get("global_calibration")
    zero = reco_view.tokens.new_zeros(())
    if not global_correction:
        return zero, {
            "global_logpt_scale_abs": zero,
            "global_loge_scale_abs": zero,
            "global_eta_shift_abs": zero,
            "global_phi_shift_abs": zero,
        }

    def component(name: str):
        value = global_correction.get(name)
        if value is None:
            return zero
        return _safe_mean(value.abs())

    components = {
        "global_logpt_scale_abs": component("logpt_scale"),
        "global_loge_scale_abs": component("loge_scale"),
        "global_eta_shift_abs": component("eta_shift"),
        "global_phi_shift_abs": component("phi_shift"),
    }
    total = sum(components.values(), zero)
    return total, components


def compute_teacher_logit_reco_loss(
    *,
    offline_logits,
    reco_logits,
    labels,
    reco_view: SoftReconstructedView,
    offline_tokens,
    offline_mask,
    config: TeacherLogitRecoLossConfig | Mapping[str, Any] | None = None,
) -> TeacherLogitRecoLossOutput:
    """Combine the Step 4 loss components for one batch."""

    cfg = TeacherLogitRecoLossConfig.from_mapping(config)
    kl = teacher_kl_loss(offline_logits, reco_logits, temperature=float(cfg.temperature))
    ce = teacher_cross_entropy_loss(reco_logits, labels)
    budget, budget_terms = correction_budget_loss(reco_view, config=cfg)
    summary = weak_jet_summary_loss(reco_view, offline_tokens, offline_mask, config=cfg)
    extra_budget, extra_budget_terms = extra_budget_loss(reco_view, config=cfg)
    parent_weight_budget, parent_weight_budget_terms = parent_weight_budget_loss(reco_view, config=cfg)
    global_budget, global_budget_terms = global_calibration_budget_loss(reco_view, config=cfg)

    components = {
        "teacher_kl": kl,
        "ce": ce,
        "correction_budget": budget,
        "jet_summary": summary,
        "extra_budget": extra_budget,
        "parent_weight_budget": parent_weight_budget,
        "global_calibration_budget": global_budget,
        **{f"budget_{name}": value for name, value in budget_terms.items()},
        **{f"extra_budget_{name}": value for name, value in extra_budget_terms.items()},
        **{f"parent_weight_budget_{name}": value for name, value in parent_weight_budget_terms.items()},
        **{f"global_calibration_budget_{name}": value for name, value in global_budget_terms.items()},
    }
    weighted_components = {
        "teacher_kl": float(cfg.teacher_kl_weight) * kl,
        "ce": float(cfg.ce_weight) * ce,
        "correction_budget": float(cfg.correction_budget_weight) * budget,
        "jet_summary": float(cfg.jet_summary_weight) * summary,
        "extra_budget": float(cfg.aggressive_extra_budget_weight) * extra_budget,
        "parent_weight_budget": float(cfg.aggressive_parent_weight_budget_weight) * parent_weight_budget,
        "global_calibration_budget": float(cfg.aggressive_global_calibration_budget_weight) * global_budget,
    }
    total = sum(weighted_components.values())
    metrics = {
        "offline_teacher_confidence": require_torch().softmax(offline_logits.detach(), dim=-1).max(dim=-1).values.mean(),
        "reco_teacher_confidence": require_torch().softmax(reco_logits.detach(), dim=-1).max(dim=-1).values.mean(),
        "reco_argmax_accuracy": (reco_logits.detach().argmax(dim=-1) == labels.to(reco_logits.device)).float().mean(),
    }
    return TeacherLogitRecoLossOutput(
        total_loss=total,
        components=components,
        weighted_components=weighted_components,
        metrics=metrics,
    )


def global_transformer_teacher_training_step(
    *,
    reconstructor,
    teacher,
    hlt_tokens,
    hlt_mask,
    offline_tokens,
    offline_mask,
    labels,
    optimizer=None,
    config: TeacherLogitRecoLossConfig | Mapping[str, Any] | None = None,
    jet_ids=None,
    split: str = "model_train",
) -> tuple[TeacherLogitRecoLossOutput, SoftReconstructedView, Any, Any]:
    """Run one optional optimizer step for the teacher-logit reconstructor."""

    torch = require_torch()
    cfg = TeacherLogitRecoLossConfig.from_mapping(config)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    reco_view = reconstructor(hlt_tokens, hlt_mask, labels=labels, jet_ids=jet_ids, split=split)
    with torch.no_grad():
        offline_logits = teacher.forward_view_no_grad(offline_tokens, offline_mask)
    reco_logits = teacher.forward_soft_view(reco_view)
    loss = compute_teacher_logit_reco_loss(
        offline_logits=offline_logits,
        reco_logits=reco_logits,
        labels=labels,
        reco_view=reco_view,
        offline_tokens=offline_tokens,
        offline_mask=offline_mask,
        config=cfg,
    )
    if optimizer is not None:
        loss.total_loss.backward()
        optimizer.step()
    return loss, reco_view, offline_logits, reco_logits

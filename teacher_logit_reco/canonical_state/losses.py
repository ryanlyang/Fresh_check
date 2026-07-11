"""Loss composition for Canonical Multi-Scale Jet State training.

Step 7 collects the objectives used by the later training runner without
coupling them to a particular DataLoader.  The loss helper is intentionally
strict about final-test teacher supervision: teacher logits are a training and
model-val diagnostic object, not an input to primary final-test evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .layout import CanonicalJetStateLayout, default_canonical_jet_state_layout


CANONICAL_STATE_LOSS_CONTRACT = "canonical_state_loss_composition_v1"


@dataclass(frozen=True)
class CanonicalStateLossWeights:
    """Weights for the Step 7 objective family."""

    ce: float = 1.0
    state_huber: float = 0.0
    state_l1: float = 0.0
    logit_kd: float = 0.0
    delta_norm: float = 0.0
    smoothness: float = 0.0
    uncertainty_state: float = 0.0
    kd_temperature: float = 2.0
    huber_beta: float = 1.0
    allow_teacher_logits_on_final_test: bool = False

    def __post_init__(self) -> None:
        for name in (
            "ce",
            "state_huber",
            "state_l1",
            "logit_kd",
            "delta_norm",
            "smoothness",
            "uncertainty_state",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if float(self.kd_temperature) <= 0.0:
            raise ValueError("kd_temperature must be positive")
        if float(self.huber_beta) <= 0.0:
            raise ValueError("huber_beta must be positive")
        object.__setattr__(self, "kd_temperature", float(self.kd_temperature))
        object.__setattr__(self, "huber_beta", float(self.huber_beta))
        object.__setattr__(self, "allow_teacher_logits_on_final_test", bool(self.allow_teacher_logits_on_final_test))

    def active_terms(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "ce",
                "state_huber",
                "state_l1",
                "logit_kd",
                "delta_norm",
                "smoothness",
                "uncertainty_state",
            )
            if float(getattr(self, name)) > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = CANONICAL_STATE_LOSS_CONTRACT
        payload["active_terms"] = list(self.active_terms())
        return payload


@dataclass(frozen=True)
class CanonicalStateLossOutput:
    total: Any
    terms: Mapping[str, Any]
    weighted_terms: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    @property
    def loss(self) -> Any:
        return self.total


def teacher_logits_allowed_for_split(split: str | None, *, allow_final_test_teacher: bool = False) -> bool:
    """Return whether teacher logits may be consumed for this split."""

    split_name = str(split or "").strip().lower()
    if split_name == "final_test" and not bool(allow_final_test_teacher):
        return False
    return True


def _as_tensor(value: Any, *, device: Any | None = None, dtype: Any | None = None) -> Any:
    torch = require_torch()
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
    if device is not None:
        tensor = tensor.to(device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _optional_tensor(value: Any | None, *, device: Any, dtype: Any) -> Any | None:
    if value is None:
        return None
    return _as_tensor(value, device=device, dtype=dtype)


def _field_importance_vector(layout: CanonicalJetStateLayout, *, device: Any, dtype: Any) -> Any:
    torch = require_torch()
    high = {
        "sum_pt_frac",
        "sum_energy_frac",
        "mean_pt_frac",
        "max_pt_frac",
        "mass_proxy",
        "charged_pt_frac",
        "neutral_pt_frac",
        "photon_pt_frac",
        "electron_pt_frac",
        "muon_pt_frac",
        "hadron_pt_frac",
    }
    medium = {
        "log1p_count",
        "pt_weighted_mean_deta",
        "pt_weighted_mean_dphi",
        "pt_weighted_var_deta",
        "pt_weighted_var_dphi",
        "width_proxy",
    }
    weights = []
    for name in layout.config.field_names:
        if name in high:
            weights.append(2.0)
        elif name in medium:
            weights.append(1.0)
        else:
            weights.append(0.35)
    return torch.tensor(weights, device=device, dtype=dtype)


def _normalized_residual(value: Any, layout: CanonicalJetStateLayout) -> Any:
    torch = require_torch()
    scales = torch.tensor(layout.residual_scale_vector(), device=value.device, dtype=value.dtype).clamp_min(1.0e-8)
    return value / scales[None, None, :]


def _masked_mean(value: Any, mask: Any) -> Any:
    weights = mask.to(device=value.device, dtype=value.dtype)
    while int(weights.ndim) < int(value.ndim):
        weights = weights.unsqueeze(-1)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def _state_smoothness(delta_phi: Any, state_mask: Any, layout: CanonicalJetStateLayout) -> Any:
    torch = require_torch()
    pieces = []
    for family, (start, end) in layout.family_slices().items():
        del family
        if int(end) - int(start) < 2:
            continue
        diffs = delta_phi[:, start + 1 : end, :] - delta_phi[:, start : end - 1, :]
        valid = state_mask[:, start + 1 : end] & state_mask[:, start : end - 1]
        if bool(valid.any()):
            pieces.append(_masked_mean(diffs.square().sum(dim=-1), valid))
    if not pieces:
        return delta_phi.new_zeros(())
    return torch.stack(pieces).mean()


def _kd_loss(student_logits: Any, teacher_logits: Any, temperature: float) -> Any:
    torch = require_torch()
    t = float(temperature)
    student_log_prob = torch.nn.functional.log_softmax(student_logits / t, dim=-1)
    teacher_prob = torch.nn.functional.softmax(teacher_logits / t, dim=-1)
    return torch.nn.functional.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (t * t)


def compute_canonical_state_losses(
    *,
    logits: Any,
    labels: Any,
    weights: CanonicalStateLossWeights | Mapping[str, Any] | None = None,
    split: str | None = "model_train",
    phi_hlt: Any | None = None,
    phi_off: Any | None = None,
    delta_phi_pred: Any | None = None,
    log_sigma: Any | None = None,
    teacher_logits: Any | None = None,
    state_mask: Any | None = None,
    layout: CanonicalJetStateLayout | None = None,
) -> CanonicalStateLossOutput:
    """Compute the configured Step 7 loss terms."""

    torch = require_torch()
    config = weights if isinstance(weights, CanonicalStateLossWeights) else CanonicalStateLossWeights(**dict(weights or {}))
    resolved_layout = default_canonical_jet_state_layout() if layout is None else layout
    logits = _as_tensor(logits).float()
    labels = _as_tensor(labels, device=logits.device).long()
    terms: dict[str, Any] = {}
    weighted: dict[str, Any] = {}

    if float(config.ce) > 0.0:
        terms["ce"] = torch.nn.functional.cross_entropy(logits, labels)
        weighted["ce"] = terms["ce"] * float(config.ce)

    if teacher_logits is not None:
        if not teacher_logits_allowed_for_split(
            split,
            allow_final_test_teacher=bool(config.allow_teacher_logits_on_final_test),
        ):
            raise ValueError("teacher logits are not allowed for primary final_test loss/evaluation")
        teacher_logits = _as_tensor(teacher_logits, device=logits.device, dtype=logits.dtype)
    if float(config.logit_kd) > 0.0:
        if teacher_logits is None:
            raise ValueError("logit_kd loss is active but teacher_logits were not provided")
        terms["logit_kd"] = _kd_loss(logits, teacher_logits, float(config.kd_temperature))
        weighted["logit_kd"] = terms["logit_kd"] * float(config.logit_kd)

    state_terms_active = any(
        float(getattr(config, name)) > 0.0
        for name in ("state_huber", "state_l1", "uncertainty_state", "delta_norm", "smoothness")
    )
    if state_terms_active:
        if delta_phi_pred is None:
            raise ValueError("state/delta losses are active but delta_phi_pred was not provided")
        delta_phi_pred = _as_tensor(delta_phi_pred, device=logits.device, dtype=logits.dtype)
        if tuple(delta_phi_pred.shape[-2:]) != (resolved_layout.k_state, resolved_layout.d_phi):
            raise ValueError("delta_phi_pred has wrong canonical-state trailing shape")
        if state_mask is None:
            state_mask = torch.ones(delta_phi_pred.shape[:2], device=logits.device, dtype=torch.bool)
        else:
            state_mask = _as_tensor(state_mask, device=logits.device).bool()
        if tuple(state_mask.shape) != tuple(delta_phi_pred.shape[:2]):
            raise ValueError("state_mask must match delta_phi_pred leading shape")
    else:
        state_mask = None

    if any(float(getattr(config, name)) > 0.0 for name in ("state_huber", "state_l1", "uncertainty_state")):
        if phi_hlt is None or phi_off is None:
            raise ValueError("state residual losses require phi_hlt and phi_off")
        phi_hlt = _as_tensor(phi_hlt, device=logits.device, dtype=logits.dtype)
        phi_off = _as_tensor(phi_off, device=logits.device, dtype=logits.dtype)
        target_delta = phi_off - phi_hlt
        pred_norm = _normalized_residual(delta_phi_pred, resolved_layout)
        target_norm = _normalized_residual(target_delta, resolved_layout)
        error_norm = pred_norm - target_norm
        field_weights = _field_importance_vector(resolved_layout, device=logits.device, dtype=logits.dtype)
        if float(config.state_huber) > 0.0:
            huber = torch.nn.functional.smooth_l1_loss(
                pred_norm,
                target_norm,
                reduction="none",
                beta=float(config.huber_beta),
            )
            terms["state_huber"] = _masked_mean(huber * field_weights[None, None, :], state_mask)
            weighted["state_huber"] = terms["state_huber"] * float(config.state_huber)
        if float(config.state_l1) > 0.0:
            terms["state_l1"] = _masked_mean(error_norm.abs() * field_weights[None, None, :], state_mask)
            weighted["state_l1"] = terms["state_l1"] * float(config.state_l1)
        if float(config.uncertainty_state) > 0.0:
            if log_sigma is None:
                raise ValueError("uncertainty_state loss is active but log_sigma was not provided")
            log_sigma = _as_tensor(log_sigma, device=logits.device, dtype=logits.dtype).clamp(min=-5.0, max=5.0)
            inv_var = torch.exp(-2.0 * log_sigma)
            nll = 0.5 * error_norm.square() * inv_var + log_sigma
            terms["uncertainty_state"] = _masked_mean(nll * field_weights[None, None, :], state_mask)
            weighted["uncertainty_state"] = terms["uncertainty_state"] * float(config.uncertainty_state)

    if float(config.delta_norm) > 0.0:
        delta_normed = _normalized_residual(delta_phi_pred, resolved_layout)
        terms["delta_norm"] = _masked_mean(delta_normed.square().sum(dim=-1), state_mask)
        weighted["delta_norm"] = terms["delta_norm"] * float(config.delta_norm)
    if float(config.smoothness) > 0.0:
        terms["smoothness"] = _state_smoothness(_normalized_residual(delta_phi_pred, resolved_layout), state_mask, resolved_layout)
        weighted["smoothness"] = terms["smoothness"] * float(config.smoothness)

    if weighted:
        total = torch.stack([term for term in weighted.values()]).sum()
    else:
        total = logits.new_zeros(())
    diagnostics = {
        "contract": CANONICAL_STATE_LOSS_CONTRACT,
        "split": str(split or ""),
        "active_terms": list(config.active_terms()),
        "teacher_logits_used": teacher_logits is not None,
        "teacher_logits_allowed": teacher_logits_allowed_for_split(
            split,
            allow_final_test_teacher=bool(config.allow_teacher_logits_on_final_test),
        ),
        "loss_weights": config.to_dict(),
    }
    for name, term in terms.items():
        diagnostics[f"{name}_loss"] = float(term.detach().cpu().item())
        diagnostics[f"{name}_weighted_loss"] = float(weighted.get(name, term.new_zeros(())).detach().cpu().item())
    diagnostics["total_loss"] = float(total.detach().cpu().item())
    return CanonicalStateLossOutput(total=total, terms=terms, weighted_terms=weighted, diagnostics=diagnostics)

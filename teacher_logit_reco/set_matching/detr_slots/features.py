"""Canonical feature adapters for DETR/free-slot reconstruction.

The DETR branch predicts free slots, but the downstream taggers still consume
the usual raw particle-token layout.  This module is the translation boundary:
raw tokens become canonical matching features, and slot-head outputs become raw
tokens again.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


DETR_SLOT_FEATURE_STEP = "detr_free_slot_step2_feature_adapter"
DETR_SLOT_FEATURE_EPS = 1.0e-6
DETR_SLOT_CORE_FEATURE_NAMES: tuple[str, str, str, str] = (
    "log_pt",
    "eta",
    "phi",
    "log_energy",
)
DETR_SLOT_RAW_KINEMATIC_FEATURE_NAMES: tuple[str, str, str, str] = (
    "pt",
    "eta",
    "phi",
    "energy",
)


def _maybe_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch


def require_torch():
    torch = _maybe_torch()
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("DETR slot feature adapters require PyTorch")
    return torch


@dataclass(frozen=True)
class DetrSlotFeatureConfig:
    """Feature-index and numerical-stability settings for DETR slot outputs."""

    feature_dim: int = RAW_TOKEN_DIM
    pt_index: int = 0
    eta_index: int = 1
    phi_index: int = 2
    energy_index: int = 3
    aux_indices: tuple[int, ...] | None = None
    signed_aux_indices: tuple[int, ...] = (4, 10, 12)
    unit_interval_aux_indices: tuple[int, ...] = (5, 6, 7, 8, 9, 11, 13)
    binary_aux_indices: tuple[int, ...] = (5, 6, 7, 8, 9)
    eps: float = DETR_SLOT_FEATURE_EPS
    max_abs_eta: float = 5.0
    min_log_value: float = -20.0
    max_log_value: float = 20.0
    enforce_energy_ge_pt_cosh_eta: bool = True

    def __post_init__(self) -> None:
        if int(self.feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        core_indices = (int(self.pt_index), int(self.eta_index), int(self.phi_index), int(self.energy_index))
        if len(set(core_indices)) != 4:
            raise ValueError(f"core feature indices must be distinct, got {core_indices}")
        for name, index in (
            ("pt_index", self.pt_index),
            ("eta_index", self.eta_index),
            ("phi_index", self.phi_index),
            ("energy_index", self.energy_index),
        ):
            if int(index) < 0 or int(index) >= int(self.feature_dim):
                raise ValueError(f"{name}={index} is outside feature_dim={self.feature_dim}")
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive")
        if float(self.max_abs_eta) <= 0.0:
            raise ValueError("max_abs_eta must be positive")
        if float(self.min_log_value) >= float(self.max_log_value):
            raise ValueError("min_log_value must be smaller than max_log_value")
        if self.aux_indices is not None:
            normalized = tuple(int(index) for index in self.aux_indices)
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"aux_indices contain duplicates: {normalized}")
            core = set(core_indices)
            for index in normalized:
                if index < 0 or index >= int(self.feature_dim):
                    raise ValueError(f"aux index {index} is outside feature_dim={self.feature_dim}")
                if index in core:
                    raise ValueError(f"aux index {index} overlaps a core kinematic index")
            object.__setattr__(self, "aux_indices", normalized)
        selected_aux = None if self.aux_indices is None else set(int(index) for index in self.aux_indices)
        signed = tuple(
            int(index)
            for index in self.signed_aux_indices
            if int(index) < int(self.feature_dim) and (selected_aux is None or int(index) in selected_aux)
        )
        unit_interval = tuple(
            int(index)
            for index in self.unit_interval_aux_indices
            if int(index) < int(self.feature_dim) and (selected_aux is None or int(index) in selected_aux)
        )
        binary = tuple(
            int(index)
            for index in self.binary_aux_indices
            if int(index) < int(self.feature_dim) and (selected_aux is None or int(index) in selected_aux)
        )
        for name, values in (
            ("signed_aux_indices", signed),
            ("unit_interval_aux_indices", unit_interval),
            ("binary_aux_indices", binary),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates: {values}")
            for index in values:
                if index in core_indices:
                    raise ValueError(f"{name} index {index} overlaps a core kinematic index")
                if index < 0:
                    raise ValueError(f"{name} index {index} is outside feature_dim={self.feature_dim}")
        overlap = set(signed) & set(unit_interval)
        if overlap:
            raise ValueError(f"signed_aux_indices and unit_interval_aux_indices overlap: {sorted(overlap)}")
        outside_unit = set(binary) - set(unit_interval)
        if outside_unit:
            raise ValueError(f"binary_aux_indices must be a subset of unit_interval_aux_indices: {sorted(outside_unit)}")
        object.__setattr__(self, "signed_aux_indices", signed)
        object.__setattr__(self, "unit_interval_aux_indices", unit_interval)
        object.__setattr__(self, "binary_aux_indices", binary)

    @property
    def core_indices(self) -> tuple[int, int, int, int]:
        return (
            int(self.pt_index),
            int(self.eta_index),
            int(self.phi_index),
            int(self.energy_index),
        )

    def aux_feature_indices(self, feature_dim: int | None = None) -> tuple[int, ...]:
        dim = int(self.feature_dim if feature_dim is None else feature_dim)
        if dim <= max(self.core_indices):
            raise ValueError(f"feature_dim={dim} is too small for core indices {self.core_indices}")
        if self.aux_indices is not None:
            return tuple(index for index in self.aux_indices if index < dim)
        core = set(self.core_indices)
        return tuple(index for index in range(dim) if index not in core)

    @property
    def aux_dim(self) -> int:
        return len(self.aux_feature_indices())

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": int(self.feature_dim),
            "pt_index": int(self.pt_index),
            "eta_index": int(self.eta_index),
            "phi_index": int(self.phi_index),
            "energy_index": int(self.energy_index),
            "aux_indices": None if self.aux_indices is None else list(self.aux_indices),
            "signed_aux_indices": list(self.signed_aux_indices),
            "unit_interval_aux_indices": list(self.unit_interval_aux_indices),
            "binary_aux_indices": list(self.binary_aux_indices),
            "eps": float(self.eps),
            "max_abs_eta": float(self.max_abs_eta),
            "min_log_value": float(self.min_log_value),
            "max_log_value": float(self.max_log_value),
            "enforce_energy_ge_pt_cosh_eta": bool(self.enforce_energy_ge_pt_cosh_eta),
            "core_feature_names": list(DETR_SLOT_CORE_FEATURE_NAMES),
            "raw_kinematic_feature_names": list(DETR_SLOT_RAW_KINEMATIC_FEATURE_NAMES),
        }


def _as_tensor(value, *, dtype=None, device=None):
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return torch.as_tensor(value, dtype=dtype, device=device)


def _sanitize_finite(value, *, fill: float = 0.0, limit: float = 1.0e6):
    torch = require_torch()
    tensor = value
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    return torch.nan_to_num(tensor, nan=float(fill), posinf=float(limit), neginf=-float(limit))


def _require_finite(value, *, name: str):
    torch = require_torch()
    tensor = value
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return tensor


def _validate_token_tensor(name: str, tokens) -> None:
    if tokens.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions [..., particles, features], got {tuple(tokens.shape)}")


def _validate_feature_dim(name: str, tokens, config: DetrSlotFeatureConfig) -> None:
    feature_dim = int(tokens.shape[-1])
    required = max(config.core_indices) + 1
    if feature_dim < required:
        raise ValueError(f"{name} feature_dim={feature_dim} is too small for core indices {config.core_indices}")


def wrap_phi(phi):
    """Wrap angles into [-pi, pi] with torch operations."""

    torch = require_torch()
    phi = _as_tensor(phi)
    if not torch.is_floating_point(phi):
        phi = phi.float()
    return torch.atan2(torch.sin(phi), torch.cos(phi))


def wrapped_phi_difference(left_phi, right_phi):
    """Return ``left_phi - right_phi`` wrapped into [-pi, pi]."""

    left = _as_tensor(left_phi)
    right = _as_tensor(right_phi, dtype=left.dtype, device=left.device)
    return wrap_phi(left - right)


def safe_log_pt_energy(pt, energy, *, eps: float = DETR_SLOT_FEATURE_EPS, max_log_value: float = 20.0):
    """Return finite ``log(pt)`` and ``log(energy)`` tensors with safe clamping."""

    torch = require_torch()
    pt_tensor = _sanitize_finite(_as_tensor(pt), fill=float(eps))
    energy_tensor = _sanitize_finite(_as_tensor(energy, dtype=pt_tensor.dtype, device=pt_tensor.device), fill=float(eps))
    pt_safe = torch.clamp(pt_tensor, min=float(eps))
    energy_safe = torch.clamp(energy_tensor, min=float(eps))
    log_pt = torch.log(pt_safe).clamp(min=-float(max_log_value), max=float(max_log_value))
    log_energy = torch.log(energy_safe).clamp(min=-float(max_log_value), max=float(max_log_value))
    return log_pt, log_energy


def raw_to_core_features(tokens, config: DetrSlotFeatureConfig | None = None):
    """Convert raw particle tokens to ``[log_pt, eta, wrapped_phi, log_energy]``."""

    tokens = _as_tensor(tokens)
    config = config or DetrSlotFeatureConfig(feature_dim=int(tokens.shape[-1]))
    _validate_token_tensor("tokens", tokens)
    _validate_feature_dim("tokens", tokens, config)
    tokens = _sanitize_finite(tokens)

    log_pt, log_energy = safe_log_pt_energy(
        tokens[..., int(config.pt_index)],
        tokens[..., int(config.energy_index)],
        eps=float(config.eps),
        max_log_value=float(config.max_log_value),
    )
    eta = torch_clamp_eta(tokens[..., int(config.eta_index)], config)
    phi = wrap_phi(tokens[..., int(config.phi_index)])
    torch = require_torch()
    return torch.stack((log_pt, eta, phi, log_energy), dim=-1)


def raw_to_aux_features(tokens, config: DetrSlotFeatureConfig | None = None):
    """Return non-core auxiliary particle features in stable index order."""

    tokens = _as_tensor(tokens)
    config = config or DetrSlotFeatureConfig(feature_dim=int(tokens.shape[-1]))
    _validate_token_tensor("tokens", tokens)
    _validate_feature_dim("tokens", tokens, config)
    tokens = _sanitize_finite(tokens)
    indices = config.aux_feature_indices(int(tokens.shape[-1]))
    if not indices:
        shape = tuple(tokens.shape[:-1]) + (0,)
        return tokens.new_zeros(shape)
    return tokens[..., list(indices)]


def torch_clamp_eta(eta, config: DetrSlotFeatureConfig):
    torch = require_torch()
    return torch.clamp(_sanitize_finite(_as_tensor(eta)), min=-float(config.max_abs_eta), max=float(config.max_abs_eta))


def smooth_bound(value, *, lower: float, upper: float):
    """Smoothly map unconstrained values into ``[lower, upper]``."""

    torch = require_torch()
    value = _as_tensor(value)
    if not torch.is_floating_point(value):
        value = value.float()
    _require_finite(value, name="smooth_bound input")
    lower = float(lower)
    upper = float(upper)
    if lower >= upper:
        raise ValueError("lower must be smaller than upper")
    center = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower)
    return center + radius * torch.tanh(value / max(radius, float(DETR_SLOT_FEATURE_EPS)))


def smooth_physical_energy(energy, pt, eta, *, config: DetrSlotFeatureConfig):
    """Smoothly enforce ``energy >= pt * cosh(eta)`` without a hard max."""

    torch = require_torch()
    if not bool(config.enforce_energy_ge_pt_cosh_eta):
        return energy
    min_energy = pt * torch.cosh(eta)
    gap = energy - min_energy
    return min_energy + torch.nn.functional.softplus(gap) + float(config.eps)


def _normalized_aux_outputs(aux_outputs, *, core, config: DetrSlotFeatureConfig, out_dim: int):
    indices = config.aux_feature_indices(out_dim)
    if aux_outputs is None:
        return None, indices
    aux = _as_tensor(aux_outputs, dtype=core.dtype, device=core.device)
    if tuple(aux.shape[:-1]) != tuple(core.shape[:-1]):
        raise ValueError(
            f"aux_outputs leading shape {tuple(aux.shape[:-1])} does not match core {tuple(core.shape[:-1])}"
        )
    if int(aux.shape[-1]) == len(indices):
        return aux, indices
    if int(aux.shape[-1]) == out_dim:
        if not indices:
            return aux[..., :0], indices
        index_tensor = require_torch().as_tensor(indices, dtype=require_torch().long, device=aux.device)
        return aux.index_select(-1, index_tensor), indices
    if indices:
        raise ValueError(
            "aux_outputs last dimension must equal configured aux_dim "
            f"({len(indices)}) or feature_dim ({out_dim}), got {int(aux.shape[-1])}"
        )
    if int(aux.shape[-1]) != 0:
        raise ValueError(f"aux_outputs provided but config has no aux slots: got {tuple(aux.shape)}")
    return aux, indices


def _apply_aux_constraints(aux, indices: tuple[int, ...], config: DetrSlotFeatureConfig, *, sanitize: bool):
    torch = require_torch()
    aux = _sanitize_finite(aux) if sanitize else _require_finite(aux, name="aux_outputs")
    if not indices:
        return aux
    pieces = []
    signed = set(int(index) for index in config.signed_aux_indices)
    unit_interval = set(int(index) for index in config.unit_interval_aux_indices)
    for source_position, feature_index in enumerate(indices):
        values = aux[..., int(source_position)]
        if int(feature_index) in signed:
            values = torch.tanh(values)
        elif int(feature_index) in unit_interval:
            values = torch.sigmoid(values)
        pieces.append(values)
    return torch.stack(pieces, dim=-1)


def decode_slot_outputs_to_raw_tokens(
    core_outputs,
    aux_outputs=None,
    *,
    config: DetrSlotFeatureConfig | None = None,
    feature_dim: int | None = None,
):
    """Decode slot-head outputs into raw particle tokens.

    ``core_outputs`` must have last dimension 4 and is interpreted as
    ``[log_pt, eta, phi, log_energy]``.  Auxiliary outputs may either contain
    exactly the configured auxiliary dimensions, or a full raw-feature vector
    from which non-core auxiliary columns are copied.
    """

    torch = require_torch()
    core = _as_tensor(core_outputs)
    if core.ndim < 2 or int(core.shape[-1]) != 4:
        raise ValueError(f"core_outputs must have shape [..., 4], got {tuple(core.shape)}")
    core = core.float()
    out_dim = int(feature_dim if feature_dim is not None else (config.feature_dim if config is not None else RAW_TOKEN_DIM))
    config = config or DetrSlotFeatureConfig(feature_dim=out_dim)
    if int(config.feature_dim) != out_dim:
        config = DetrSlotFeatureConfig(
            feature_dim=out_dim,
            pt_index=config.pt_index,
            eta_index=config.eta_index,
            phi_index=config.phi_index,
            energy_index=config.energy_index,
            aux_indices=config.aux_indices,
            signed_aux_indices=config.signed_aux_indices,
            unit_interval_aux_indices=config.unit_interval_aux_indices,
            binary_aux_indices=config.binary_aux_indices,
            eps=config.eps,
            max_abs_eta=config.max_abs_eta,
            min_log_value=config.min_log_value,
            max_log_value=config.max_log_value,
            enforce_energy_ge_pt_cosh_eta=config.enforce_energy_ge_pt_cosh_eta,
        )
    core = _sanitize_finite(core)

    log_pt = core[..., 0].clamp(min=float(config.min_log_value), max=float(config.max_log_value))
    eta = torch_clamp_eta(core[..., 1], config)
    phi = wrap_phi(core[..., 2])
    log_energy = core[..., 3].clamp(min=float(config.min_log_value), max=float(config.max_log_value))

    pt = torch.exp(log_pt).clamp(min=float(config.eps))
    energy = torch.exp(log_energy).clamp(min=float(config.eps))
    energy = smooth_physical_energy(energy, pt, eta, config=config)

    output = core.new_zeros(tuple(core.shape[:-1]) + (out_dim,))
    output[..., int(config.pt_index)] = pt
    output[..., int(config.eta_index)] = eta
    output[..., int(config.phi_index)] = phi
    output[..., int(config.energy_index)] = energy

    aux, indices = _normalized_aux_outputs(aux_outputs, core=core, config=config, out_dim=out_dim)
    if aux is not None and indices:
        aux = _apply_aux_constraints(aux, indices, config, sanitize=True)
        for source_position, feature_index in enumerate(indices):
            output[..., int(feature_index)] = aux[..., int(source_position)]
    return output


def decode_slot_outputs_to_loss_features(
    core_outputs,
    aux_outputs=None,
    *,
    config: DetrSlotFeatureConfig | None = None,
    feature_dim: int | None = None,
):
    """Decode slot-head outputs into smooth loss-facing particle features.

    This path deliberately avoids hard clamps and ``nan_to_num``.  It still
    returns the raw feature layout expected by existing set-matching losses, but
    kinematics are obtained through smooth bounded transforms so gradients do
    not silently die before the Hungarian loss sees the prediction.
    """

    torch = require_torch()
    core = _as_tensor(core_outputs)
    if core.ndim < 2 or int(core.shape[-1]) != 4:
        raise ValueError(f"core_outputs must have shape [..., 4], got {tuple(core.shape)}")
    core = core.float()
    core = _require_finite(core, name="core_outputs")
    out_dim = int(feature_dim if feature_dim is not None else (config.feature_dim if config is not None else RAW_TOKEN_DIM))
    config = config or DetrSlotFeatureConfig(feature_dim=out_dim)
    if int(config.feature_dim) != out_dim:
        config = DetrSlotFeatureConfig(
            feature_dim=out_dim,
            pt_index=config.pt_index,
            eta_index=config.eta_index,
            phi_index=config.phi_index,
            energy_index=config.energy_index,
            aux_indices=config.aux_indices,
            signed_aux_indices=config.signed_aux_indices,
            unit_interval_aux_indices=config.unit_interval_aux_indices,
            binary_aux_indices=config.binary_aux_indices,
            eps=config.eps,
            max_abs_eta=config.max_abs_eta,
            min_log_value=config.min_log_value,
            max_log_value=config.max_log_value,
            enforce_energy_ge_pt_cosh_eta=config.enforce_energy_ge_pt_cosh_eta,
        )

    log_pt = smooth_bound(core[..., 0], lower=float(config.min_log_value), upper=float(config.max_log_value))
    eta = float(config.max_abs_eta) * torch.tanh(core[..., 1] / float(config.max_abs_eta))
    phi = wrap_phi(core[..., 2])
    log_energy = smooth_bound(core[..., 3], lower=float(config.min_log_value), upper=float(config.max_log_value))
    output = core.new_zeros(tuple(core.shape[:-1]) + (out_dim,))
    output[..., int(config.pt_index)] = torch.exp(log_pt)
    output[..., int(config.eta_index)] = eta
    output[..., int(config.phi_index)] = phi
    pt = torch.exp(log_pt)
    energy = torch.exp(log_energy)
    output[..., int(config.energy_index)] = smooth_physical_energy(energy, pt, eta, config=config)

    aux, indices = _normalized_aux_outputs(aux_outputs, core=core, config=config, out_dim=out_dim)
    if aux is not None and indices:
        aux = _apply_aux_constraints(aux, indices, config, sanitize=False)
        for source_position, feature_index in enumerate(indices):
            output[..., int(feature_index)] = aux[..., int(source_position)]
    return output


def default_detr_slot_feature_config(*, feature_dim: int = RAW_TOKEN_DIM) -> DetrSlotFeatureConfig:
    return DetrSlotFeatureConfig(feature_dim=int(feature_dim))


def feature_indices_report(config: DetrSlotFeatureConfig | None = None) -> dict[str, Any]:
    config = config or DetrSlotFeatureConfig()
    return {
        "step": DETR_SLOT_FEATURE_STEP,
        "core": {
            name: int(index)
            for name, index in zip(DETR_SLOT_RAW_KINEMATIC_FEATURE_NAMES, config.core_indices)
        },
        "aux_indices": list(config.aux_feature_indices()),
        "signed_aux_indices": list(config.signed_aux_indices),
        "unit_interval_aux_indices": list(config.unit_interval_aux_indices),
        "binary_aux_indices": list(config.binary_aux_indices),
        "config": config.to_dict(),
    }

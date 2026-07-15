"""Physical root summaries, residual targets, and robust normalization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch

from .schemas import ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .targets import ROOT_FEATURE_NAMES


ABPH_ROOT_TRANSFORM_CONTRACT = "adaptive_binary_pseudooffline_root_transform_v1"
ABPH_ROOT_TRANSFORM_VERSION = "v1"
ABPH_ROOT_EPSILON = 1.0e-8

ROOT_FEATURE_INDEX: Mapping[str, int] = {
    name: index for index, name in enumerate(ROOT_FEATURE_NAMES)
}
ROOT_SHAPE_FEATURE_NAMES: tuple[str, ...] = (
    "eta_first_moment",
    "phi_first_moment",
    "eta_second_moment",
    "phi_second_moment",
    "eta_phi_cross_moment",
    "radial_first_moment",
    "radial_second_moment",
    "covariance_cholesky[0]",
    "covariance_cholesky[1]",
    "covariance_cholesky[2]",
    "radial_quantiles[0]",
    "radial_quantiles[1]",
    "radial_quantiles[2]",
    "leading_pt_fractions[0]",
    "leading_pt_fractions[1]",
    "leading_pt_fractions[2]",
    "leading_pt_fractions[3]",
)
ROOT_AUXILIARY_FEATURE_NAMES: tuple[str, ...] = tuple(
    name
    for name in ROOT_FEATURE_NAMES
    if name
    not in {
        "energy",
        "px",
        "py",
        "pz",
        "constituent_count",
        *(f"count_{pid}" for pid in ABPH_PID_CATEGORIES),
        "integer_charge",
        "minimum_mass_budget",
        "feasible_charge_min",
        "feasible_charge_max",
    }
)
ROOT_RESIDUAL_CHANNEL_NAMES: tuple[str, ...] = (
    "delta_log_pt",
    "delta_eta",
    "delta_phi",
    "delta_mass_excess",
    "delta_count",
)


def wrap_phi_tensor(values: Any) -> Any:
    torch = require_torch()
    return torch.remainder(values + torch.pi, 2.0 * torch.pi) - torch.pi


def inverse_softplus(values: Any, *, epsilon: float = ABPH_ROOT_EPSILON) -> Any:
    torch = require_torch()
    values = torch.as_tensor(values)
    positive = values.clamp_min(float(epsilon))
    return positive + torch.log(-torch.expm1(-positive))


@dataclass(frozen=True)
class RootPhysicalKinematics:
    pt: Any
    eta: Any
    phi: Any
    mass: Any
    energy: Any
    px: Any
    py: Any
    pz: Any

    def four_vector(self) -> Any:
        torch = require_torch()
        return torch.stack((self.energy, self.px, self.py, self.pz), dim=-1)


@dataclass(frozen=True)
class HLTRootSummary:
    kinematics: RootPhysicalKinematics
    constituent_count: Any
    type_counts: Any
    integer_charge: Any
    scalar_sum_pt: Any
    type_scalar_pt: Any
    type_energy: Any
    particle_mask: Any


@dataclass(frozen=True)
class RootResidualTargets:
    p4_residuals: Any
    count_index: Any
    delta_count: Any
    type_counts: Any
    type_count_fractions: Any
    type_pt_fractions: Any
    type_energy_fractions: Any
    integer_charge: Any
    minimum_mass_budget: Any
    shape_features: Any
    auxiliary_features: Any
    physical: RootPhysicalKinematics

    def normalization_matrix(self) -> Any:
        torch = require_torch()
        return torch.cat(
            (
                self.p4_residuals,
                self.delta_count.unsqueeze(-1),
                self.shape_features,
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class RootNormalizationStats:
    channel_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    fit_split: str = "model_train"
    method: str = "median_iqr"
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if not self.channel_names:
            raise ValueError("normalization statistics require at least one channel")
        if len(self.center) != len(self.channel_names) or len(self.scale) != len(
            self.channel_names
        ):
            raise ValueError("normalization statistics dimensions do not agree")
        if any(not np.isfinite(value) for value in self.center + self.scale):
            raise ValueError("normalization statistics must be finite")
        if any(value <= 0.0 for value in self.scale):
            raise ValueError("normalization scales must be positive")
        if self.fit_split != "model_train":
            raise ValueError("root normalization may only be fit on model_train")

    @property
    def normalization_hash(self) -> str:
        payload = self.to_dict(include_hash=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract": ABPH_ROOT_TRANSFORM_CONTRACT,
            "version": ABPH_ROOT_TRANSFORM_VERSION,
            "channel_names": list(self.channel_names),
            "center": list(self.center),
            "scale": list(self.scale),
            "fit_split": self.fit_split,
            "method": self.method,
            "epsilon": self.epsilon,
        }
        if include_hash:
            payload["normalization_hash"] = self.normalization_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RootNormalizationStats":
        result = cls(
            channel_names=tuple(str(value) for value in payload["channel_names"]),
            center=tuple(float(value) for value in payload["center"]),
            scale=tuple(float(value) for value in payload["scale"]),
            fit_split=str(payload.get("fit_split", "model_train")),
            method=str(payload.get("method", "median_iqr")),
            epsilon=float(payload.get("epsilon", 1.0e-6)),
        )
        expected = payload.get("normalization_hash")
        if expected not in (None, result.normalization_hash):
            raise ValueError("root normalization hash mismatch")
        return result

    def normalize(self, values: Any) -> Any:
        torch = require_torch()
        center = torch.as_tensor(self.center, dtype=values.dtype, device=values.device)
        scale = torch.as_tensor(self.scale, dtype=values.dtype, device=values.device)
        return (values - center) / scale

    def denormalize(self, values: Any) -> Any:
        torch = require_torch()
        center = torch.as_tensor(self.center, dtype=values.dtype, device=values.device)
        scale = torch.as_tensor(self.scale, dtype=values.dtype, device=values.device)
        return values * scale + center

    def _indices(self, names: tuple[str, ...]) -> tuple[int, ...]:
        lookup = {name: index for index, name in enumerate(self.channel_names)}
        missing = tuple(name for name in names if name not in lookup)
        if missing:
            raise ValueError(f"normalization statistics are missing channels: {missing}")
        return tuple(lookup[name] for name in names)

    def normalize_named(self, values: Any, names: tuple[str, ...]) -> Any:
        """Normalize a channel subset without relying on positional coincidence."""

        torch = require_torch()
        if values.shape[-1] != len(names):
            raise ValueError("named normalization shape does not match the channel list")
        indices = self._indices(names)
        center = torch.as_tensor(
            tuple(self.center[index] for index in indices),
            dtype=values.dtype,
            device=values.device,
        )
        scale = torch.as_tensor(
            tuple(self.scale[index] for index in indices),
            dtype=values.dtype,
            device=values.device,
        )
        return (values - center) / scale

    def denormalize_named(self, values: Any, names: tuple[str, ...]) -> Any:
        """Invert a named channel subset using the hash-bound model-train stats."""

        torch = require_torch()
        if values.shape[-1] != len(names):
            raise ValueError("named denormalization shape does not match the channel list")
        indices = self._indices(names)
        center = torch.as_tensor(
            tuple(self.center[index] for index in indices),
            dtype=values.dtype,
            device=values.device,
        )
        scale = torch.as_tensor(
            tuple(self.scale[index] for index in indices),
            dtype=values.dtype,
            device=values.device,
        )
        return values * scale + center

    def scale_named(self, names: tuple[str, ...], *, like: Any) -> Any:
        torch = require_torch()
        indices = self._indices(names)
        return torch.as_tensor(
            tuple(self.scale[index] for index in indices),
            dtype=like.dtype,
            device=like.device,
        )


def fit_root_normalization_stats(
    targets: RootResidualTargets,
    *,
    fit_split: str = "model_train",
    epsilon: float = 1.0e-6,
) -> RootNormalizationStats:
    torch = require_torch()
    if fit_split != "model_train":
        raise ValueError("root normalization may only be fit on model_train")
    matrix = targets.normalization_matrix().detach().float()
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("normalization target matrix must be nonempty and rank two")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("normalization targets contain nonfinite values")
    center = torch.quantile(matrix, 0.5, dim=0)
    q25 = torch.quantile(matrix, 0.25, dim=0)
    q75 = torch.quantile(matrix, 0.75, dim=0)
    robust_scale = (q75 - q25) / 1.349
    fallback = matrix.std(dim=0, unbiased=False)
    scale = torch.where(robust_scale > float(epsilon), robust_scale, fallback)
    scale = scale.clamp_min(float(epsilon))
    names = ROOT_RESIDUAL_CHANNEL_NAMES + ROOT_SHAPE_FEATURE_NAMES
    return RootNormalizationStats(
        channel_names=names,
        center=tuple(float(value) for value in center.cpu()),
        scale=tuple(float(value) for value in scale.cpu()),
        fit_split=fit_split,
        epsilon=epsilon,
    )


def kinematics_from_four_vector(four_vector: Any) -> RootPhysicalKinematics:
    torch = require_torch()
    p4 = torch.as_tensor(four_vector).float()
    if p4.shape[-1] != 4:
        raise ValueError("four_vector must end with [energy, px, py, pz]")
    energy, px, py, pz = p4.unbind(dim=-1)
    pt = torch.sqrt((px.square() + py.square()).clamp_min(0.0))
    phi = wrap_phi_tensor(torch.atan2(py, px))
    eta = torch.asinh(pz / pt.clamp_min(ABPH_ROOT_EPSILON))
    mass_squared = energy.square() - px.square() - py.square() - pz.square()
    mass = torch.sqrt(mass_squared.clamp_min(0.0))
    return RootPhysicalKinematics(pt, eta, phi, mass, energy, px, py, pz)


def kinematics_from_pt_eta_phi_mass(pt: Any, eta: Any, phi: Any, mass: Any) -> RootPhysicalKinematics:
    torch = require_torch()
    pt = torch.as_tensor(pt).float()
    eta = torch.as_tensor(eta, device=pt.device).float()
    phi = wrap_phi_tensor(torch.as_tensor(phi, device=pt.device).float())
    mass = torch.as_tensor(mass, device=pt.device).float().clamp_min(0.0)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    momentum_squared = pt.square() + pz.square()
    energy = torch.sqrt((momentum_squared + mass.square()).clamp_min(0.0))
    return RootPhysicalKinematics(pt, eta, phi, mass, energy, px, py, pz)


def summarize_hlt_root(hlt_tokens: Any, hlt_mask: Any) -> HLTRootSummary:
    """Derive deployable root evidence exclusively from raw HLT particles."""

    torch = require_torch()
    tokens = torch.as_tensor(hlt_tokens).float()
    mask = torch.as_tensor(hlt_mask, device=tokens.device).bool()
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (ABPH_MAX_PARTICLES, 14):
        raise ValueError("HLT tokens must have shape [batch, 128, 14]")
    if mask.shape != tokens.shape[:2]:
        raise ValueError("HLT mask shape does not match HLT tokens")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("root prediction requires at least one valid HLT particle per jet")
    valid = mask.to(tokens.dtype)
    pt = tokens[..., 0].clamp_min(0.0) * valid
    eta = tokens[..., 1]
    phi = tokens[..., 2]
    energy_particle = tokens[..., 3] * valid
    px_particle = pt * torch.cos(phi)
    py_particle = pt * torch.sin(phi)
    pz_particle = pt * torch.sinh(eta)
    p4 = torch.stack(
        (
            energy_particle.sum(dim=1),
            px_particle.sum(dim=1),
            py_particle.sum(dim=1),
            pz_particle.sum(dim=1),
        ),
        dim=-1,
    )
    pid_logits = tokens[..., 5:10]
    known = pid_logits.max(dim=-1).values > 0.5
    pid_indices = pid_logits.argmax(dim=-1)
    pid_indices = torch.where(
        known, pid_indices, torch.full_like(pid_indices, len(ABPH_PID_CATEGORIES) - 1)
    )
    one_hot = torch.nn.functional.one_hot(
        pid_indices, num_classes=len(ABPH_PID_CATEGORIES)
    ).to(tokens.dtype)
    one_hot = one_hot * valid.unsqueeze(-1)
    type_counts = one_hot.sum(dim=1).to(torch.long)
    type_scalar_pt = (one_hot * pt.unsqueeze(-1)).sum(dim=1)
    type_energy = (one_hot * energy_particle.unsqueeze(-1)).sum(dim=1)
    charge = torch.round(tokens[..., 4]) * valid
    return HLTRootSummary(
        kinematics=kinematics_from_four_vector(p4),
        constituent_count=mask.sum(dim=1).to(torch.long),
        type_counts=type_counts,
        integer_charge=charge.sum(dim=1).to(torch.long),
        scalar_sum_pt=pt.sum(dim=1),
        type_scalar_pt=type_scalar_pt,
        type_energy=type_energy,
        particle_mask=mask,
    )


def root_ledger_kinematics(root_ledger: Any) -> RootPhysicalKinematics:
    torch = require_torch()
    ledger = torch.as_tensor(root_ledger).float()
    if ledger.ndim != 2 or ledger.shape[-1] != len(ROOT_FEATURE_NAMES):
        raise ValueError(f"root ledger must have shape [batch, {len(ROOT_FEATURE_NAMES)}]")
    p4 = torch.stack(
        tuple(ledger[:, ROOT_FEATURE_INDEX[name]] for name in ("energy", "px", "py", "pz")),
        dim=-1,
    )
    return kinematics_from_four_vector(p4)


def root_physical_to_residual(
    hlt: RootPhysicalKinematics,
    target: RootPhysicalKinematics,
    minimum_mass_budget: Any,
    *,
    epsilon: float = ABPH_ROOT_EPSILON,
) -> Any:
    """Transform physical targets to the stable residual parameterization."""

    torch = require_torch()
    minimum_mass = torch.as_tensor(
        minimum_mass_budget, dtype=target.mass.dtype, device=target.mass.device
    )
    delta_log_pt = torch.log(target.pt.clamp_min(epsilon) / hlt.pt.clamp_min(epsilon))
    delta_eta = target.eta - hlt.eta
    delta_phi = wrap_phi_tensor(target.phi - hlt.phi)
    target_excess = (target.mass - minimum_mass).clamp_min(epsilon)
    hlt_excess = (hlt.mass - minimum_mass).clamp_min(epsilon)
    delta_mass = inverse_softplus(target_excess, epsilon=epsilon) - inverse_softplus(
        hlt_excess, epsilon=epsilon
    )
    return torch.stack((delta_log_pt, delta_eta, delta_phi, delta_mass), dim=-1)


def root_residual_to_physical(
    hlt: RootPhysicalKinematics,
    residuals: Any,
    minimum_mass_budget: Any,
    *,
    epsilon: float = ABPH_ROOT_EPSILON,
) -> RootPhysicalKinematics:
    """Invert root residuals and deterministically compile the four-vector."""

    torch = require_torch()
    residual = torch.as_tensor(residuals).float()
    if residual.shape[-1] != 4:
        raise ValueError("root physical residuals must have four channels")
    minimum_mass = torch.as_tensor(
        minimum_mass_budget, dtype=residual.dtype, device=residual.device
    )
    pt = hlt.pt * torch.exp(residual[..., 0].clamp(-12.0, 12.0))
    eta = hlt.eta + residual[..., 1]
    phi = wrap_phi_tensor(hlt.phi + residual[..., 2])
    base_excess = (hlt.mass - minimum_mass).clamp_min(epsilon)
    mass_excess = torch.nn.functional.softplus(
        inverse_softplus(base_excess, epsilon=epsilon) + residual[..., 3]
    )
    mass = minimum_mass + mass_excess
    return kinematics_from_pt_eta_phi_mass(pt, eta, phi, mass)


def build_root_residual_targets(
    hlt_tokens: Any,
    hlt_mask: Any,
    root_ledger: Any,
) -> RootResidualTargets:
    """Convert cached offline root ledgers into trainable residual targets."""

    torch = require_torch()
    hlt = summarize_hlt_root(hlt_tokens, hlt_mask)
    ledger = torch.as_tensor(root_ledger, device=hlt.kinematics.pt.device).float()
    physical = root_ledger_kinematics(ledger)
    count = ledger[:, ROOT_FEATURE_INDEX["constituent_count"]].round().to(torch.long)
    if bool(((count < 1) | (count > ABPH_MAX_PARTICLES)).any()):
        raise ValueError("offline root count lies outside [1, 128]")
    type_counts = torch.stack(
        tuple(
            ledger[:, ROOT_FEATURE_INDEX[f"count_{pid}"]].round().to(torch.long)
            for pid in ABPH_PID_CATEGORIES
        ),
        dim=-1,
    )
    if not bool((type_counts.sum(dim=-1) == count).all()):
        raise ValueError("offline root type counts do not sum to the total count")
    minimum_mass = ledger[:, ROOT_FEATURE_INDEX["minimum_mass_budget"]]
    residual = root_physical_to_residual(hlt.kinematics, physical, minimum_mass)
    type_pt = torch.stack(
        tuple(
            ledger[:, ROOT_FEATURE_INDEX[f"scalar_pt_{pid}"]]
            for pid in ABPH_PID_CATEGORIES
        ),
        dim=-1,
    ).clamp_min(0.0)
    type_energy = torch.stack(
        tuple(
            ledger[:, ROOT_FEATURE_INDEX[f"energy_{pid}"]]
            for pid in ABPH_PID_CATEGORIES
        ),
        dim=-1,
    ).clamp_min(0.0)
    count_fractions = type_counts.to(ledger.dtype) / count.unsqueeze(-1).to(ledger.dtype)
    pt_fractions = type_pt / type_pt.sum(dim=-1, keepdim=True).clamp_min(ABPH_ROOT_EPSILON)
    energy_fractions = type_energy / type_energy.sum(dim=-1, keepdim=True).clamp_min(
        ABPH_ROOT_EPSILON
    )
    shape = torch.stack(
        tuple(ledger[:, ROOT_FEATURE_INDEX[name]] for name in ROOT_SHAPE_FEATURE_NAMES), dim=-1
    )
    auxiliary = torch.stack(
        tuple(ledger[:, ROOT_FEATURE_INDEX[name]] for name in ROOT_AUXILIARY_FEATURE_NAMES),
        dim=-1,
    )
    return RootResidualTargets(
        p4_residuals=residual,
        count_index=count - 1,
        delta_count=count.to(ledger.dtype) - hlt.constituent_count.to(ledger.dtype),
        type_counts=type_counts,
        type_count_fractions=count_fractions,
        type_pt_fractions=pt_fractions,
        type_energy_fractions=energy_fractions,
        integer_charge=ledger[:, ROOT_FEATURE_INDEX["integer_charge"]].round().to(torch.long),
        minimum_mass_budget=minimum_mass,
        shape_features=shape,
        auxiliary_features=auxiliary,
        physical=physical,
    )


__all__ = [
    "ABPH_ROOT_EPSILON",
    "ABPH_ROOT_TRANSFORM_CONTRACT",
    "ABPH_ROOT_TRANSFORM_VERSION",
    "HLTRootSummary",
    "ROOT_AUXILIARY_FEATURE_NAMES",
    "ROOT_FEATURE_INDEX",
    "ROOT_RESIDUAL_CHANNEL_NAMES",
    "ROOT_SHAPE_FEATURE_NAMES",
    "RootNormalizationStats",
    "RootPhysicalKinematics",
    "RootResidualTargets",
    "build_root_residual_targets",
    "fit_root_normalization_stats",
    "inverse_softplus",
    "kinematics_from_four_vector",
    "kinematics_from_pt_eta_phi_mass",
    "root_ledger_kinematics",
    "root_physical_to_residual",
    "root_residual_to_physical",
    "summarize_hlt_root",
    "wrap_phi_tensor",
]

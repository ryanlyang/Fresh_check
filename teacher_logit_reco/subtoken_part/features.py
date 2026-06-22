"""Feature grouping for reliability-gated subtoken Particle Transformers.

This module is deliberately model-free.  It validates the raw JetClass token
contract, splits each particle into named modality groups, and builds the same
derived kinematic features used by the existing Particle Transformer input
builder in :mod:`jetclass_fresh.part_inputs`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.part_inputs import EPS, JET_FEATURE_NAMES, PF_FEATURE_NAMES, PF_POINT_NAMES, PF_VECTOR_NAMES

from .config import (
    SUBTOKEN_MODALITY_IDENTITY,
    SUBTOKEN_MODALITY_KINEMATICS,
    SUBTOKEN_MODALITY_TRACK,
    SUBTOKEN_PART_RAW_FEATURE_NAMES,
    SubtokenFeatureConfig,
)


SUBTOKEN_DERIVED_KINEMATIC_FEATURE_NAMES: tuple[str, ...] = (
    "part_pt_log",
    "part_e_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "part_deta",
    "part_dphi",
)

SUBTOKEN_DERIVED_TRACK_FEATURE_NAMES: tuple[str, ...] = (
    "part_d0",
    "part_d0err",
    "part_dz",
    "part_dzerr",
)


@dataclass(frozen=True)
class SubtokenDerivedKinematics:
    """ParT-style derived kinematic features for one raw token view."""

    part_features: Any
    part_feature_names: tuple[str, ...]
    part_points: Any
    part_point_names: tuple[str, ...]
    part_vectors: Any
    part_vector_names: tuple[str, ...]
    jet_features: Any
    jet_feature_names: tuple[str, ...]
    feature_map: Mapping[str, Any]

    def feature_tensor(self, names: tuple[str, ...]) -> Any:
        if not names:
            torch = require_torch()
            return self.part_features.new_zeros((*self.part_features.shape[:2], 0))
        missing = [name for name in names if name not in self.feature_map]
        if missing:
            raise KeyError(f"Unknown derived feature names: {missing}")
        return require_torch().stack([self.feature_map[name] for name in names], dim=-1)


@dataclass(frozen=True)
class SubtokenModalityInputs:
    """Raw modality split before derived kinematics are appended."""

    modality_values: Mapping[str, Any]
    modality_feature_names: Mapping[str, tuple[str, ...]]
    mask: Any
    raw_tokens: Any
    feature_config: SubtokenFeatureConfig

    @property
    def kin_values(self) -> Any:
        return self.modality_values[SUBTOKEN_MODALITY_KINEMATICS]

    @property
    def id_values(self) -> Any:
        return self.modality_values[SUBTOKEN_MODALITY_IDENTITY]

    @property
    def track_values(self) -> Any:
        return self.modality_values[SUBTOKEN_MODALITY_TRACK]

    def values_for(self, modality: str) -> Any:
        return self.modality_values[modality]


@dataclass(frozen=True)
class SubtokenInputs(SubtokenModalityInputs):
    """Final Step-2 feature object consumed by later subtoken encoders."""

    derived_kinematics: SubtokenDerivedKinematics | None = None

    @property
    def raw_kin_values(self) -> Any:
        kin_modalities = [modality for modality in self.feature_config.modalities if modality.name == SUBTOKEN_MODALITY_KINEMATICS]
        if not kin_modalities:
            raise KeyError(f"feature_config does not contain {SUBTOKEN_MODALITY_KINEMATICS!r}")
        raw_indices = kin_modalities[0].raw_indices
        return _select_feature_indices(self.raw_tokens, raw_indices)

    def summary(self) -> dict[str, Any]:
        return {
            "raw_tokens_shape": list(self.raw_tokens.shape),
            "mask_shape": list(self.mask.shape),
            "modalities": {
                name: {
                    "shape": list(values.shape),
                    "feature_names": list(self.modality_feature_names[name]),
                }
                for name, values in self.modality_values.items()
            },
            "has_derived_kinematics": self.derived_kinematics is not None,
        }


def normalize_feature_config(config: SubtokenFeatureConfig | Mapping[str, Any] | None = None) -> SubtokenFeatureConfig:
    if config is None:
        return SubtokenFeatureConfig()
    if isinstance(config, SubtokenFeatureConfig):
        return config
    return SubtokenFeatureConfig(**dict(config))


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _manual_transform_torch(
    value: Any,
    *,
    subtract: float,
    multiply: float,
    clip_min: float = -5.0,
    clip_max: float = 5.0,
) -> Any:
    torch = require_torch()
    return torch.clamp((value - float(subtract)) * float(multiply), min=float(clip_min), max=float(clip_max))


def _safe_log_torch(value: Any) -> Any:
    torch = require_torch()
    return torch.log(torch.clamp(value, min=float(EPS)))


def wrap_phi_torch(phi: Any) -> Any:
    torch = require_torch()
    return torch.remainder(phi + math.pi, 2.0 * math.pi) - math.pi


def _select_feature_indices(values: Any, indices: tuple[int, ...]) -> Any:
    torch = require_torch()
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=values.device)
    return values.index_select(dim=-1, index=index_tensor)


def _zero_masked(values: Any, mask: Any) -> Any:
    torch = require_torch()
    return torch.where(mask[:, :, None], values, torch.zeros_like(values))


def prepare_subtoken_tokens_and_mask(
    tokens: Any,
    mask: Any,
    config: SubtokenFeatureConfig | Mapping[str, Any] | None = None,
) -> tuple[Any, Any, SubtokenFeatureConfig]:
    """Validate, cast, sanitize, and mask raw JetClass token tensors."""

    torch = require_torch()
    feature_config = normalize_feature_config(config)
    if not hasattr(tokens, "shape") or not hasattr(mask, "shape"):
        raise TypeError("tokens and mask must be torch tensors or tensor-like objects")
    tokens = tokens.float()
    mask = mask.bool()
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(tokens.shape[-1]) != int(feature_config.raw_token_dim):
        raise ValueError(
            f"tokens last dimension must be raw_token_dim={int(feature_config.raw_token_dim)}, "
            f"got {int(tokens.shape[-1])}"
        )
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape [batch, particles], got {tuple(mask.shape)}")
    if tuple(tokens.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"tokens/mask leading shapes differ: {tuple(tokens.shape[:2])} vs {tuple(mask.shape)}")
    tokens = _nan_to_num_torch(tokens)
    return torch.where(mask[:, :, None], tokens, torch.zeros_like(tokens)), mask, feature_config


def split_raw_tokens_into_modalities(
    tokens: Any,
    mask: Any,
    config: SubtokenFeatureConfig | Mapping[str, Any] | None = None,
) -> SubtokenModalityInputs:
    """Split raw `[B, N, 14]` tokens into configured modality tensors."""

    tokens, mask, feature_config = prepare_subtoken_tokens_and_mask(tokens, mask, config=config)
    modality_values: dict[str, Any] = {}
    modality_feature_names: dict[str, tuple[str, ...]] = {}
    for modality in feature_config.modalities:
        values = _select_feature_indices(tokens, modality.raw_indices)
        feature_names = tuple(SUBTOKEN_PART_RAW_FEATURE_NAMES[index] for index in modality.raw_indices)
        modality_values[modality.name] = _zero_masked(values, mask)
        modality_feature_names[modality.name] = feature_names
    return SubtokenModalityInputs(
        modality_values=modality_values,
        modality_feature_names=modality_feature_names,
        mask=mask,
        raw_tokens=tokens,
        feature_config=feature_config,
    )


def build_derived_kinematics(tokens: Any, mask: Any) -> SubtokenDerivedKinematics:
    """Build torch equivalents of `jetclass_fresh.part_inputs` features.

    Returned tensors are shaped `[batch, particles, features]`, unlike the
    existing numpy ParT builder which stores particle features channel-first.
    """

    torch = require_torch()
    tokens, mask, _ = prepare_subtoken_tokens_and_mask(tokens, mask, config=SubtokenFeatureConfig())
    mask_float = mask.to(dtype=tokens.dtype)

    pt = tokens[:, :, 0]
    eta = tokens[:, :, 1]
    phi = tokens[:, :, 2]
    energy = tokens[:, :, 3]

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)

    jet_px = (px * mask_float).sum(dim=1)
    jet_py = (py * mask_float).sum(dim=1)
    jet_pz = (pz * mask_float).sum(dim=1)
    jet_energy = (energy * mask_float).sum(dim=1)
    jet_pt = torch.sqrt(torch.clamp(jet_px * jet_px + jet_py * jet_py, min=0.0))
    jet_phi = torch.atan2(jet_py, jet_px)
    jet_eta = torch.asinh(jet_pz / torch.clamp(jet_pt, min=float(EPS)))
    jet_eta = torch.where(jet_pt > float(EPS), jet_eta, torch.zeros_like(jet_eta))
    jet_phi = torch.where(jet_pt > float(EPS), jet_phi, torch.zeros_like(jet_phi))
    mass2 = jet_energy * jet_energy - jet_px * jet_px - jet_py * jet_py - jet_pz * jet_pz
    jet_mass = torch.sqrt(torch.clamp(mass2, min=0.0))
    jet_nparticles = mask_float.sum(dim=1)

    jet_eta_col = jet_eta[:, None]
    jet_phi_col = jet_phi[:, None]
    jet_pt_col = jet_pt[:, None]
    jet_energy_col = jet_energy[:, None]

    eta_sign = torch.sign(jet_eta_col)
    eta_sign = torch.where(eta_sign == 0.0, torch.ones_like(eta_sign), eta_sign)
    part_deta = (eta - jet_eta_col) * eta_sign
    part_dphi = wrap_phi_torch(phi - jet_phi_col)
    part_delta_r = torch.sqrt(torch.clamp(part_deta * part_deta + part_dphi * part_dphi, min=0.0))

    feature_map = {
        "part_pt_log": _manual_transform_torch(_safe_log_torch(pt), subtract=1.7, multiply=0.7),
        "part_e_log": _manual_transform_torch(_safe_log_torch(energy), subtract=2.0, multiply=0.7),
        "part_logptrel": _manual_transform_torch(
            _safe_log_torch(pt / torch.clamp(jet_pt_col, min=float(EPS))),
            subtract=-4.7,
            multiply=0.7,
        ),
        "part_logerel": _manual_transform_torch(
            _safe_log_torch(energy / torch.clamp(jet_energy_col, min=float(EPS))),
            subtract=-4.7,
            multiply=0.7,
        ),
        "part_deltaR": _manual_transform_torch(part_delta_r, subtract=0.2, multiply=4.0),
        "part_charge": tokens[:, :, 4],
        "part_isChargedHadron": tokens[:, :, 5],
        "part_isNeutralHadron": tokens[:, :, 6],
        "part_isPhoton": tokens[:, :, 7],
        "part_isElectron": tokens[:, :, 8],
        "part_isMuon": tokens[:, :, 9],
        "part_d0": torch.tanh(tokens[:, :, 10]),
        "part_d0err": _manual_transform_torch(tokens[:, :, 11], subtract=0.0, multiply=1.0, clip_min=0.0, clip_max=1.0),
        "part_dz": torch.tanh(tokens[:, :, 12]),
        "part_dzerr": _manual_transform_torch(tokens[:, :, 13], subtract=0.0, multiply=1.0, clip_min=0.0, clip_max=1.0),
        "part_deta": part_deta,
        "part_dphi": part_dphi,
    }
    feature_map = {
        name: torch.where(mask, _nan_to_num_torch(value), torch.zeros_like(value))
        for name, value in feature_map.items()
    }

    part_points = torch.stack([feature_map[name] for name in PF_POINT_NAMES], dim=-1)
    part_features = torch.stack([feature_map[name] for name in PF_FEATURE_NAMES], dim=-1)
    part_vectors = torch.stack([px, py, pz, energy], dim=-1)
    part_vectors = _zero_masked(_nan_to_num_torch(part_vectors), mask)
    jet_features = torch.stack([jet_pt, jet_eta, jet_phi, jet_energy, jet_mass, jet_nparticles], dim=-1)
    jet_features = _nan_to_num_torch(jet_features)

    return SubtokenDerivedKinematics(
        part_features=part_features,
        part_feature_names=tuple(PF_FEATURE_NAMES),
        part_points=part_points,
        part_point_names=tuple(PF_POINT_NAMES),
        part_vectors=part_vectors,
        part_vector_names=tuple(PF_VECTOR_NAMES),
        jet_features=jet_features,
        jet_feature_names=tuple(JET_FEATURE_NAMES),
        feature_map=feature_map,
    )


def build_subtoken_inputs(
    tokens: Any,
    mask: Any,
    config: SubtokenFeatureConfig | Mapping[str, Any] | None = None,
) -> SubtokenInputs:
    """Build named modality inputs for the subtoken branch."""

    raw = split_raw_tokens_into_modalities(tokens, mask, config=config)
    derived = (
        build_derived_kinematics(raw.raw_tokens, raw.mask)
        if raw.feature_config.include_part_style_derived_features
        else None
    )
    modality_values = dict(raw.modality_values)
    modality_feature_names = dict(raw.modality_feature_names)
    if derived is not None:
        derived_kin = derived.feature_tensor(SUBTOKEN_DERIVED_KINEMATIC_FEATURE_NAMES)
        modality_values[SUBTOKEN_MODALITY_KINEMATICS] = _zero_masked(
            require_torch().cat([raw.kin_values, derived_kin], dim=-1),
            raw.mask,
        )
        modality_feature_names[SUBTOKEN_MODALITY_KINEMATICS] = (
            *modality_feature_names[SUBTOKEN_MODALITY_KINEMATICS],
            *SUBTOKEN_DERIVED_KINEMATIC_FEATURE_NAMES,
        )
        derived_track = derived.feature_tensor(SUBTOKEN_DERIVED_TRACK_FEATURE_NAMES)
        modality_values[SUBTOKEN_MODALITY_TRACK] = _zero_masked(
            require_torch().cat([raw.track_values, derived_track], dim=-1),
            raw.mask,
        )
        modality_feature_names[SUBTOKEN_MODALITY_TRACK] = (
            *modality_feature_names[SUBTOKEN_MODALITY_TRACK],
            *SUBTOKEN_DERIVED_TRACK_FEATURE_NAMES,
        )
    return SubtokenInputs(
        modality_values=modality_values,
        modality_feature_names=modality_feature_names,
        mask=raw.mask,
        raw_tokens=raw.raw_tokens,
        feature_config=raw.feature_config,
        derived_kinematics=derived,
    )


__all__ = [
    "SUBTOKEN_DERIVED_KINEMATIC_FEATURE_NAMES",
    "SUBTOKEN_DERIVED_TRACK_FEATURE_NAMES",
    "SubtokenDerivedKinematics",
    "SubtokenInputs",
    "SubtokenModalityInputs",
    "build_derived_kinematics",
    "build_subtoken_inputs",
    "normalize_feature_config",
    "prepare_subtoken_tokens_and_mask",
    "split_raw_tokens_into_modalities",
    "wrap_phi_torch",
]

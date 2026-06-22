"""Shared feature calibration for auxiliary subtoken losses."""

from __future__ import annotations

from typing import Any, Sequence
import math

from jetclass_fresh.hlt_baseline import require_torch

from .config import (
    SUBTOKEN_MODALITY_IDENTITY,
    SUBTOKEN_MODALITY_KINEMATICS,
    SUBTOKEN_MODALITY_TRACK,
)


SUBTOKEN_PART_AUXILIARY_FEATURE_CALIBRATION = "part_style_auxiliary_feature_calibration_v1"


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _default_feature_names(modality_name: str, dim: int) -> tuple[str, ...]:
    if modality_name == SUBTOKEN_MODALITY_KINEMATICS:
        names = (
            "pt",
            "eta",
            "phi",
            "energy",
            "part_pt_log",
            "part_e_log",
            "part_logptrel",
            "part_logerel",
            "part_deltaR",
            "part_deta",
            "part_dphi",
        )
    elif modality_name == SUBTOKEN_MODALITY_IDENTITY:
        names = (
            "charge",
            "isChargedHadron",
            "isNeutralHadron",
            "isPhoton",
            "isElectron",
            "isMuon",
        )
    elif modality_name == SUBTOKEN_MODALITY_TRACK:
        names = (
            "d0",
            "d0err",
            "dz",
            "dzerr",
            "part_d0",
            "part_d0err",
            "part_dz",
            "part_dzerr",
        )
    else:
        names = tuple(f"{modality_name}_{index}" for index in range(int(dim)))
    if len(names) == int(dim):
        return names
    if int(dim) < len(names):
        return names[: int(dim)]
    return tuple(f"{modality_name}_{index}" for index in range(int(dim)))


def _calibrate_named_feature(value: Any, feature_name: str) -> Any:
    torch = require_torch()
    name = str(feature_name)
    if name in {"pt", "energy"}:
        return torch.log1p(torch.clamp(value, min=0.0)) / 8.0
    if name in {"eta", "part_deta"}:
        return torch.clamp(value / 5.0, min=-2.0, max=2.0)
    if name in {"phi", "part_dphi"}:
        return torch.clamp(value / math.pi, min=-2.0, max=2.0)
    if name in {"part_pt_log", "part_e_log", "part_logptrel", "part_logerel", "part_deltaR"}:
        return torch.clamp(value / 5.0, min=-2.0, max=2.0)
    if name in {"charge", "part_charge"}:
        return torch.clamp(value, min=-1.0, max=1.0)
    if name in {
        "isChargedHadron",
        "isNeutralHadron",
        "isPhoton",
        "isElectron",
        "isMuon",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
    }:
        return torch.clamp(value, min=0.0, max=1.0)
    if name in {"d0", "dz"}:
        return torch.tanh(value)
    if name in {"d0err", "dzerr", "part_d0err", "part_dzerr"}:
        return torch.clamp(value, min=0.0, max=1.0)
    if name in {"part_d0", "part_dz"}:
        return torch.clamp(value, min=-1.0, max=1.0)
    return torch.clamp(value, min=-5.0, max=5.0)


def _calibrate_named_prediction(value: Any, feature_name: str) -> Any:
    torch = require_torch()
    name = str(feature_name)
    if name in {"pt", "energy"}:
        return torch.log1p(torch.nn.functional.softplus(value)) / 8.0
    if name in {"eta", "part_deta"}:
        return 2.0 * torch.tanh(value / 10.0)
    if name in {"phi", "part_dphi"}:
        return 2.0 * torch.tanh(value / (2.0 * math.pi))
    if name in {"part_pt_log", "part_e_log", "part_logptrel", "part_logerel", "part_deltaR"}:
        return 2.0 * torch.tanh(value / 10.0)
    if name in {"charge", "part_charge", "d0", "dz", "part_d0", "part_dz"}:
        return torch.tanh(value)
    if name in {
        "isChargedHadron",
        "isNeutralHadron",
        "isPhoton",
        "isElectron",
        "isMuon",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
        "d0err",
        "dzerr",
        "part_d0err",
        "part_dzerr",
    }:
        return torch.sigmoid(value)
    return 5.0 * torch.tanh(value / 5.0)


def calibrate_modality_values(
    modality_name: str,
    values: Any,
    feature_names: Sequence[str] | None = None,
) -> Any:
    """Map heterogeneous modality features to a bounded comparison space.

    The subtoken inputs still receive their original values.  This calibration is
    only for auxiliary residual/masked losses, where raw pT or energy should not
    dominate PID, impact-parameter, or derived ParT-style features.
    """

    torch = require_torch()
    values = _nan_to_num_torch(values.float())
    if int(values.ndim) < 1:
        raise ValueError("values must have a feature dimension")
    dim = int(values.shape[-1])
    names = tuple(str(name) for name in (feature_names or _default_feature_names(str(modality_name), dim)))
    if len(names) != dim:
        names = _default_feature_names(str(modality_name), dim)
    calibrated = [_calibrate_named_feature(values[..., index], names[index]) for index in range(dim)]
    return _nan_to_num_torch(torch.stack(calibrated, dim=-1))


def calibrate_modality_predictions(
    modality_name: str,
    values: Any,
    feature_names: Sequence[str] | None = None,
) -> Any:
    """Differentiably map masked-head outputs to the comparison space.

    Targets use hard clipping because they are fixed data.  Predictions use
    smooth positive/bounded transforms so auxiliary losses do not create
    zero-gradient regions when an unconstrained head starts on the wrong side of
    a physical boundary.
    """

    torch = require_torch()
    values = _nan_to_num_torch(values.float())
    if int(values.ndim) < 1:
        raise ValueError("values must have a feature dimension")
    dim = int(values.shape[-1])
    names = tuple(str(name) for name in (feature_names or _default_feature_names(str(modality_name), dim)))
    if len(names) != dim:
        names = _default_feature_names(str(modality_name), dim)
    calibrated = [_calibrate_named_prediction(values[..., index], names[index]) for index in range(dim)]
    return _nan_to_num_torch(torch.stack(calibrated, dim=-1))


__all__ = [
    "SUBTOKEN_PART_AUXILIARY_FEATURE_CALIBRATION",
    "calibrate_modality_predictions",
    "calibrate_modality_values",
]

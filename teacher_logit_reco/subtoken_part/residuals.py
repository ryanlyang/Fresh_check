"""Coarse modality residual targets for privileged subtoken training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .auxiliary import calibrate_modality_values
from .classifier import SubtokenClassifierOutput
from .config import (
    SUBTOKEN_MODALITY_IDENTITY,
    SUBTOKEN_MODALITY_KINEMATICS,
    SUBTOKEN_MODALITY_TRACK,
    SubtokenFeatureConfig,
    SubtokenPartConfig,
)
from .features import SubtokenInputs, build_subtoken_inputs

try:  # Keep imports lightweight on systems without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


SUBTOKEN_PART_RESIDUAL_STEP = "subtoken_part_step17_modality_residuals"
SUBTOKEN_PART_RESIDUAL_CONTRACT = "privileged_modality_residual_targets_v1"
SUBTOKEN_PART_RESIDUAL_TARGET_MODES = ("jet", "nearest", "jet_plus_nearest")


@dataclass(frozen=True)
class ModalityResidualTargetOutput:
    """Coarse residual targets for each HLT particle and modality."""

    targets: Any
    mask: Any
    jet_targets: Any
    nearest_targets: Any
    modality_names: tuple[str, ...]
    target_mode: str

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_RESIDUAL_CONTRACT,
            "target_mode": self.target_mode,
            "targets_shape": list(self.targets.shape),
            "mask_shape": list(self.mask.shape),
            "jet_targets_shape": list(self.jet_targets.shape),
            "nearest_targets_shape": list(self.nearest_targets.shape),
            "modality_names": list(self.modality_names),
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        dtype = self.targets.dtype
        active = self.mask.to(dtype=dtype)
        denom = torch.clamp(active.sum(), min=1.0)
        valid_particle_denom = torch.clamp(self.mask.any(dim=2).sum().to(dtype=dtype), min=1.0)
        by_modality_denom = torch.clamp(active.sum(dim=(0, 1)), min=1.0)
        return {
            "mean_target": (self.targets * active).sum() / denom,
            "max_target": torch.where(self.mask, self.targets, torch.zeros_like(self.targets)).amax(),
            "mean_target_by_modality": (self.targets * active).sum(dim=(0, 1)) / by_modality_denom,
            "valid_particle_fraction": self.mask.any(dim=2).to(dtype=dtype).sum() / valid_particle_denom,
        }


@dataclass(frozen=True)
class ModalityResidualPredictionOutput:
    """Predicted residual magnitude per particle and modality."""

    residual_logits: Any
    residual_pred_by_modality: Any
    mask: Any
    modality_names: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_RESIDUAL_CONTRACT,
            "residual_logits_shape": list(self.residual_logits.shape),
            "residual_pred_by_modality_shape": list(self.residual_pred_by_modality.shape),
            "mask_shape": list(self.mask.shape),
            "modality_names": list(self.modality_names),
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        active = self.mask.to(dtype=self.residual_pred_by_modality.dtype)
        denom = torch.clamp(active.sum(), min=1.0)
        by_modality_denom = torch.clamp(active.sum(dim=(0, 1)), min=1.0)
        return {
            "mean_pred": (self.residual_pred_by_modality * active).sum() / denom,
            "mean_pred_by_modality": (
                self.residual_pred_by_modality * active
            ).sum(dim=(0, 1)) / by_modality_denom,
        }


def normalize_modality_residual_target_mode(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "broadcast": "jet",
        "jet_level": "jet",
        "nearest_neighbor": "nearest",
        "nearest_neighbour": "nearest",
        "combined": "jet_plus_nearest",
        "both": "jet_plus_nearest",
    }
    mode = aliases.get(key, key)
    if mode not in SUBTOKEN_PART_RESIDUAL_TARGET_MODES:
        raise ValueError(
            f"Unknown modality residual target mode {value!r}; expected one of {SUBTOKEN_PART_RESIDUAL_TARGET_MODES}"
        )
    return mode


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _normalize_feature_config(config: SubtokenFeatureConfig | Mapping[str, Any] | None = None) -> SubtokenFeatureConfig:
    if config is None:
        return SubtokenFeatureConfig()
    if isinstance(config, SubtokenFeatureConfig):
        return config
    return SubtokenFeatureConfig(**dict(config))


def _build_inputs(tokens: Any, mask: Any, config: SubtokenFeatureConfig | Mapping[str, Any]) -> SubtokenInputs:
    return build_subtoken_inputs(tokens, mask, config=_normalize_feature_config(config))


def _masked_mean(values: Any, mask: Any) -> Any:
    torch = require_torch()
    mask_float = mask.to(dtype=values.dtype)
    denom = torch.clamp(mask_float.sum(dim=1, keepdim=True), min=1.0)
    return (values * mask_float[:, :, None]).sum(dim=1) / denom


def _modality_distance(
    values_a: Any,
    values_b: Any,
    *,
    modality_name: str,
    feature_names: tuple[str, ...],
) -> Any:
    torch = require_torch()
    if int(values_a.shape[-1]) != int(values_b.shape[-1]):
        raise ValueError(f"Modality feature dimensions differ: {int(values_a.shape[-1])} vs {int(values_b.shape[-1])}")
    values_a = calibrate_modality_values(modality_name, values_a, feature_names)
    values_b = calibrate_modality_values(modality_name, values_b, feature_names)
    distance = torch.nn.functional.smooth_l1_loss(values_a, values_b, reduction="none", beta=1.0)
    return _nan_to_num_torch(distance.mean(dim=-1))


def _pairwise_delta_r(hlt_tokens: Any, hlt_mask: Any, offline_tokens: Any, offline_mask: Any) -> Any:
    torch = require_torch()
    hlt_eta = hlt_tokens[:, :, 1]
    hlt_phi = hlt_tokens[:, :, 2]
    offline_eta = offline_tokens[:, :, 1]
    offline_phi = offline_tokens[:, :, 2]
    deta = hlt_eta[:, :, None] - offline_eta[:, None, :]
    dphi = torch.remainder(hlt_phi[:, :, None] - offline_phi[:, None, :] + torch.pi, 2.0 * torch.pi) - torch.pi
    delta_r = torch.sqrt(torch.clamp(deta * deta + dphi * dphi, min=0.0))
    valid_pair = hlt_mask[:, :, None] & offline_mask[:, None, :]
    return delta_r.masked_fill(~valid_pair, torch.finfo(delta_r.dtype).max)


def _gather_offline_nearest(values: Any, nearest_index: Any) -> Any:
    gather_index = nearest_index[:, :, None].expand(-1, -1, int(values.shape[-1]))
    return values.gather(dim=1, index=gather_index)


def _modality_names_from_inputs(inputs: SubtokenInputs) -> tuple[str, ...]:
    names = (
        SUBTOKEN_MODALITY_KINEMATICS,
        SUBTOKEN_MODALITY_IDENTITY,
        SUBTOKEN_MODALITY_TRACK,
    )
    missing = [name for name in names if name not in inputs.modality_values]
    if missing:
        raise ValueError(f"Missing expected modality inputs: {missing}")
    return names


def compute_modality_residual_targets(
    hlt_tokens: Any,
    hlt_mask: Any,
    offline_tokens: Any,
    offline_mask: Any,
    *,
    feature_config: SubtokenFeatureConfig | Mapping[str, Any] | None = None,
    target_mode: str = "jet",
) -> ModalityResidualTargetOutput:
    """Build coarse HLT/offline residual targets by modality.

    ``jet`` compares mean modality summaries and broadcasts the result to every
    HLT particle. ``nearest`` uses angular nearest-neighbor matching from each
    HLT particle to the offline view. ``jet_plus_nearest`` averages the two.
    """

    torch = require_torch()
    target_mode = normalize_modality_residual_target_mode(target_mode)
    hlt_tokens = torch.as_tensor(hlt_tokens, dtype=torch.float32)
    hlt_mask = torch.as_tensor(hlt_mask, dtype=torch.bool, device=hlt_tokens.device)
    offline_tokens = torch.as_tensor(offline_tokens, dtype=hlt_tokens.dtype, device=hlt_tokens.device)
    offline_mask = torch.as_tensor(offline_mask, dtype=torch.bool, device=hlt_tokens.device)
    if tuple(hlt_tokens.shape[:2]) != tuple(hlt_mask.shape):
        raise ValueError("hlt_tokens and hlt_mask leading shapes differ")
    if tuple(offline_tokens.shape[:2]) != tuple(offline_mask.shape):
        raise ValueError("offline_tokens and offline_mask leading shapes differ")
    if int(hlt_tokens.shape[0]) != int(offline_tokens.shape[0]):
        raise ValueError("HLT/offline batch sizes differ")

    config = _normalize_feature_config(feature_config)
    hlt_inputs = _build_inputs(hlt_tokens, hlt_mask, config)
    offline_inputs = _build_inputs(offline_tokens, offline_mask, config)
    modality_names = _modality_names_from_inputs(hlt_inputs)
    if _modality_names_from_inputs(offline_inputs) != modality_names:
        raise ValueError("HLT/offline modality names differ")

    jet_rows = []
    nearest_rows = []
    delta_r = _pairwise_delta_r(hlt_tokens, hlt_mask, offline_tokens, offline_mask)
    nearest_index = delta_r.argmin(dim=2)
    has_offline = offline_mask.any(dim=1)
    for name in modality_names:
        hlt_values = hlt_inputs.modality_values[name]
        offline_values = offline_inputs.modality_values[name]
        feature_names = tuple(hlt_inputs.modality_feature_names[name])
        hlt_summary = _masked_mean(hlt_values, hlt_mask)
        offline_summary = _masked_mean(offline_values, offline_mask)
        jet_distance = _modality_distance(
            hlt_summary[:, None, :],
            offline_summary[:, None, :],
            modality_name=name,
            feature_names=feature_names,
        ).squeeze(1)
        jet_rows.append(jet_distance)

        nearest_offline_values = _gather_offline_nearest(offline_values, nearest_index)
        nearest_distance = _modality_distance(
            hlt_values,
            nearest_offline_values,
            modality_name=name,
            feature_names=feature_names,
        )
        nearest_distance = torch.where(
            hlt_mask & has_offline[:, None],
            nearest_distance,
            torch.zeros_like(nearest_distance),
        )
        nearest_rows.append(nearest_distance)

    jet_targets = torch.stack(jet_rows, dim=1)
    nearest_targets = torch.stack(nearest_rows, dim=2)
    broadcast_jet = jet_targets[:, None, :].expand_as(nearest_targets)
    if target_mode == "jet":
        targets = broadcast_jet
    elif target_mode == "nearest":
        targets = nearest_targets
    else:
        targets = 0.5 * (broadcast_jet + nearest_targets)
    mask = hlt_mask[:, :, None].expand_as(targets)
    targets = torch.where(mask, _nan_to_num_torch(targets), torch.zeros_like(targets))
    return ModalityResidualTargetOutput(
        targets=targets,
        mask=mask,
        jet_targets=_nan_to_num_torch(jet_targets),
        nearest_targets=torch.where(mask, _nan_to_num_torch(nearest_targets), torch.zeros_like(nearest_targets)),
        modality_names=modality_names,
        target_mode=target_mode,
    )


def _make_mlp(input_dim: int, hidden_dim: int, *, dropout: float = 0.0):
    torch = require_torch()
    return torch.nn.Sequential(
        torch.nn.LayerNorm(int(input_dim)),
        torch.nn.Linear(int(input_dim), int(hidden_dim)),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(int(hidden_dim), 1),
    )


def _normalize_model_config(config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> SubtokenPartConfig:
    if config is None:
        return SubtokenPartConfig(num_classes=2)
    if isinstance(config, SubtokenPartConfig):
        return config
    return SubtokenPartConfig(**dict(config))


class ModalityResidualHead(_ModuleBase):
    """Predict HLT/offline residual magnitude from local modality and context tokens."""

    def __init__(
        self,
        config: SubtokenPartConfig | Mapping[str, Any] | None = None,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.hidden_dim = int(hidden_dim or max(32, 2 * self.embed_dim))
        self.head = _make_mlp(3 * self.embed_dim, self.hidden_dim, dropout=float(self.config.dropout))

    def forward(self, output: SubtokenClassifierOutput) -> ModalityResidualPredictionOutput:
        torch = require_torch()
        local_tokens = output.mixed.local_tokens
        mask = output.mixed.mask
        modality_mask = output.mixed.modality_mask
        if output.context is None:
            context_tokens = torch.zeros_like(output.pooled.provisional_particles)
        else:
            context_tokens = output.context.context_tokens
        pooled = output.pooled.provisional_particles
        context_expanded = context_tokens[:, :, None, :].expand_as(local_tokens)
        pooled_expanded = pooled[:, :, None, :].expand_as(local_tokens)
        features = torch.cat([local_tokens, context_expanded, pooled_expanded], dim=-1)
        logits = self.head(features).squeeze(-1)
        pred = torch.nn.functional.softplus(logits)
        pred = torch.where(modality_mask & mask[:, :, None], _nan_to_num_torch(pred), torch.zeros_like(pred))
        logits = torch.where(modality_mask & mask[:, :, None], _nan_to_num_torch(logits), torch.zeros_like(logits))
        return ModalityResidualPredictionOutput(
            residual_logits=logits,
            residual_pred_by_modality=pred,
            mask=modality_mask & mask[:, :, None],
            modality_names=tuple(output.mixed.modality_names),
        )


def compute_modality_residual_loss(
    prediction: ModalityResidualPredictionOutput,
    target: ModalityResidualTargetOutput,
    *,
    huber_beta: float = 1.0,
) -> Any:
    torch = require_torch()
    if tuple(prediction.residual_pred_by_modality.shape) != tuple(target.targets.shape):
        raise ValueError(
            "prediction/target shape mismatch: "
            f"{tuple(prediction.residual_pred_by_modality.shape)} vs {tuple(target.targets.shape)}"
        )
    mask = prediction.mask & target.mask
    loss = torch.nn.functional.smooth_l1_loss(
        torch.log1p(prediction.residual_pred_by_modality),
        torch.log1p(target.targets),
        reduction="none",
        beta=float(huber_beta),
    )
    mask_float = mask.to(dtype=loss.dtype)
    denom = torch.clamp(mask_float.sum(), min=1.0)
    loss = (loss * mask_float).sum() / denom
    if not bool(torch.isfinite(loss).detach().cpu().item()):
        raise FloatingPointError("modality residual loss is non-finite")
    return loss


def compute_gate_residual_regularization(
    gates: Any,
    targets: Any,
    mask: Any,
    *,
    temperature: float = 1.0,
) -> Any:
    """Encourage high gates for low-residual modalities."""

    torch = require_torch()
    temperature = max(float(temperature), 1.0e-6)
    active = mask.bool()
    scores = (-targets / temperature).masked_fill(~active, torch.finfo(targets.dtype).min)
    target_weights = torch.softmax(scores, dim=2)
    target_weights = torch.where(active, target_weights, torch.zeros_like(target_weights))
    target_weights = target_weights / torch.clamp(target_weights.sum(dim=2, keepdim=True), min=1.0e-12)
    gates = torch.where(active, gates, torch.zeros_like(gates))
    loss = torch.nn.functional.smooth_l1_loss(gates, target_weights, reduction="none", beta=0.25)
    active_float = active.to(dtype=loss.dtype)
    loss = (loss * active_float).sum() / torch.clamp(active_float.sum(), min=1.0)
    if not bool(torch.isfinite(loss).detach().cpu().item()):
        raise FloatingPointError("gate residual regularization is non-finite")
    return loss


__all__ = [
    "SUBTOKEN_PART_RESIDUAL_CONTRACT",
    "SUBTOKEN_PART_RESIDUAL_STEP",
    "SUBTOKEN_PART_RESIDUAL_TARGET_MODES",
    "ModalityResidualHead",
    "ModalityResidualPredictionOutput",
    "ModalityResidualTargetOutput",
    "compute_gate_residual_regularization",
    "compute_modality_residual_loss",
    "compute_modality_residual_targets",
    "normalize_modality_residual_target_mode",
]

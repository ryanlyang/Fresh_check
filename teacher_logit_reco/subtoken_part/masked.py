"""Masked modality-subtoken objective for privileged subtoken training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .auxiliary import calibrate_modality_predictions, calibrate_modality_values
from .classifier import SubtokenClassifierOutput
from .config import (
    SUBTOKEN_MODALITY_IDENTITY,
    SUBTOKEN_MODALITY_KINEMATICS,
    SUBTOKEN_MODALITY_TRACK,
    SubtokenFeatureConfig,
    SubtokenPartConfig,
)
from .encoders import subtoken_modality_input_dims
from .features import build_subtoken_inputs

try:  # Keep imports lightweight on systems without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


SUBTOKEN_PART_MASKED_STEP = "subtoken_part_step18_masked_subtoken_objective"
SUBTOKEN_PART_MASKED_CONTRACT = "masked_modality_subtoken_prediction_v1"
SUBTOKEN_PART_MASKED_TARGET_HLT_SELF = "hlt_self"
SUBTOKEN_PART_MASKED_TARGET_OFFLINE = "offline"
SUBTOKEN_PART_MASKED_TARGET_OFFLINE_SLOT = "offline_slot"
SUBTOKEN_PART_MASKED_TARGET_MODES = (
    SUBTOKEN_PART_MASKED_TARGET_HLT_SELF,
    SUBTOKEN_PART_MASKED_TARGET_OFFLINE,
    SUBTOKEN_PART_MASKED_TARGET_OFFLINE_SLOT,
)


@dataclass(frozen=True)
class MaskedSubtokenMaskOutput:
    """Input mask plus target mask for a masked-modality forward pass."""

    input_modality_mask: Any
    prediction_mask: Any
    particle_mask: Any
    modality_names: tuple[str, ...]
    mask_probability: float

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_MASKED_CONTRACT,
            "input_modality_mask_shape": list(self.input_modality_mask.shape),
            "prediction_mask_shape": list(self.prediction_mask.shape),
            "particle_mask_shape": list(self.particle_mask.shape),
            "modality_names": list(self.modality_names),
            "mask_probability": float(self.mask_probability),
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        valid_particles = torch.clamp(self.particle_mask.sum().to(dtype=torch.float32), min=1.0)
        masked_by_modality = self.prediction_mask.to(dtype=torch.float32).sum(dim=(0, 1))
        return {
            "masked_particle_fraction": self.prediction_mask.any(dim=2).to(dtype=torch.float32).sum() / valid_particles,
            "masked_count": self.prediction_mask.to(dtype=torch.float32).sum(),
            "masked_by_modality": masked_by_modality,
        }


@dataclass(frozen=True)
class MaskedSubtokenTargetOutput:
    """Target modality values for masked-subtoken prediction."""

    target_values: Mapping[str, Any]
    prediction_mask: Any
    particle_mask: Any
    modality_names: tuple[str, ...]
    target_mode: str
    modality_feature_names: Mapping[str, tuple[str, ...]]
    matching_metadata: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_MASKED_CONTRACT,
            "target_mode": self.target_mode,
            "prediction_mask_shape": list(self.prediction_mask.shape),
            "particle_mask_shape": list(self.particle_mask.shape),
            "modality_names": list(self.modality_names),
            "target_shapes": {name: list(values.shape) for name, values in self.target_values.items()},
            "modality_feature_names": {name: list(values) for name, values in self.modality_feature_names.items()},
            "matching_metadata": dict(self.matching_metadata),
        }


@dataclass(frozen=True)
class MaskedSubtokenPredictionOutput:
    """Predicted target values for each modality."""

    predictions: Mapping[str, Any]
    prediction_mask: Any
    particle_mask: Any
    modality_names: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_MASKED_CONTRACT,
            "prediction_mask_shape": list(self.prediction_mask.shape),
            "particle_mask_shape": list(self.particle_mask.shape),
            "modality_names": list(self.modality_names),
            "prediction_shapes": {name: list(values.shape) for name, values in self.predictions.items()},
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        active = self.prediction_mask.to(dtype=torch.float32)
        denom = torch.clamp(active.sum(), min=1.0)
        return {
            "prediction_masked_fraction": self.prediction_mask.any(dim=2).to(dtype=torch.float32).sum()
            / torch.clamp(self.particle_mask.sum().to(dtype=torch.float32), min=1.0),
            "prediction_mask_count": active.sum(),
            "prediction_abs_mean": sum(
                (
                    values.abs()
                    * self.prediction_mask[:, :, index, None].to(dtype=values.dtype)
                ).sum()
                / torch.clamp(
                    self.prediction_mask[:, :, index].to(dtype=values.dtype).sum() * int(values.shape[-1]),
                    min=1.0,
                )
                for index, (name, values) in enumerate(self.predictions.items())
            )
            / denom.new_tensor(max(1, len(self.predictions))),
        }


def normalize_masked_subtoken_target_mode(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "hlt": "hlt_self",
        "self": "hlt_self",
        "self_reconstruction": "hlt_self",
        "offline_target": "offline",
        "privileged_offline": "offline",
        "offline_matched": "offline",
        "nearest": "offline",
        "nearest_offline": "offline",
        "offline_nearest": "offline",
        "slot": "offline_slot",
        "slot_aligned": "offline_slot",
        "offline_aligned": "offline_slot",
        "offline_slot_aligned": "offline_slot",
    }
    mode = aliases.get(key, key)
    if mode not in SUBTOKEN_PART_MASKED_TARGET_MODES:
        raise ValueError(
            f"Unknown masked subtoken target mode {value!r}; expected one of {SUBTOKEN_PART_MASKED_TARGET_MODES}"
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


def _normalize_model_config(config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> SubtokenPartConfig:
    if config is None:
        return SubtokenPartConfig(num_classes=2)
    if isinstance(config, SubtokenPartConfig):
        return config
    return SubtokenPartConfig(**dict(config))


def _default_modality_names() -> tuple[str, ...]:
    return (
        SUBTOKEN_MODALITY_KINEMATICS,
        SUBTOKEN_MODALITY_IDENTITY,
        SUBTOKEN_MODALITY_TRACK,
    )


def sample_masked_subtoken_modality_mask(
    particle_mask: Any,
    *,
    num_modalities: int,
    mask_probability: float,
    base_modality_mask: Any | None = None,
    modality_names: tuple[str, ...] | None = None,
    force_at_least_one: bool = True,
) -> MaskedSubtokenMaskOutput:
    """Sample one held-out modality for a subset of valid particles."""

    torch = require_torch()
    particle_mask = torch.as_tensor(particle_mask, dtype=torch.bool)
    if int(particle_mask.ndim) != 2:
        raise ValueError(f"particle_mask must have shape [batch, particles], got {tuple(particle_mask.shape)}")
    num_modalities = int(num_modalities)
    if num_modalities <= 0:
        raise ValueError("num_modalities must be positive")
    probability = float(mask_probability)
    if probability < 0.0 or probability >= 1.0:
        raise ValueError("mask_probability must be in [0, 1)")
    if base_modality_mask is None:
        base_modality_mask = particle_mask[:, :, None].expand(*particle_mask.shape, num_modalities)
    else:
        base_modality_mask = torch.as_tensor(base_modality_mask, dtype=torch.bool, device=particle_mask.device)
    if tuple(base_modality_mask.shape) != (*tuple(particle_mask.shape), num_modalities):
        raise ValueError(
            f"base_modality_mask shape {tuple(base_modality_mask.shape)} does not match "
            f"{(*tuple(particle_mask.shape), num_modalities)}"
        )
    base_modality_mask = base_modality_mask & particle_mask[:, :, None]
    if modality_names is None:
        modality_names = tuple(f"modality_{index}" for index in range(num_modalities))
    if len(modality_names) != num_modalities:
        raise ValueError("modality_names length must match num_modalities")

    should_mask = (torch.rand(particle_mask.shape, device=particle_mask.device) < probability) & particle_mask
    choice = torch.randint(num_modalities, particle_mask.shape, dtype=torch.long, device=particle_mask.device)
    prediction_mask = torch.nn.functional.one_hot(choice, num_classes=num_modalities).bool()
    prediction_mask = prediction_mask & should_mask[:, :, None] & base_modality_mask
    invalid_choice = should_mask & ~prediction_mask.any(dim=2)
    if bool(invalid_choice.any().detach().cpu().item()):
        first_active = base_modality_mask.to(dtype=torch.long).argmax(dim=2)
        fallback = torch.nn.functional.one_hot(first_active, num_classes=num_modalities).bool()
        prediction_mask = prediction_mask | (invalid_choice[:, :, None] & fallback & base_modality_mask)

    if bool(force_at_least_one) and not bool(prediction_mask.any().detach().cpu().item()) and bool(particle_mask.any().detach().cpu().item()):
        first_particle = particle_mask.reshape(-1).nonzero(as_tuple=False)[0, 0]
        batch_index = first_particle // int(particle_mask.shape[1])
        particle_index = first_particle % int(particle_mask.shape[1])
        first_active = int(base_modality_mask[batch_index, particle_index].to(dtype=torch.long).argmax().item())
        prediction_mask[batch_index, particle_index, first_active] = True

    input_modality_mask = base_modality_mask & ~prediction_mask
    return MaskedSubtokenMaskOutput(
        input_modality_mask=input_modality_mask,
        prediction_mask=prediction_mask,
        particle_mask=particle_mask,
        modality_names=tuple(modality_names),
        mask_probability=probability,
    )


def _pairwise_delta_r(reference_tokens: Any, reference_mask: Any, target_tokens: Any, target_mask: Any) -> Any:
    torch = require_torch()
    reference_eta = reference_tokens[:, :, 1]
    reference_phi = reference_tokens[:, :, 2]
    target_eta = target_tokens[:, :, 1]
    target_phi = target_tokens[:, :, 2]
    deta = reference_eta[:, :, None] - target_eta[:, None, :]
    dphi = torch.remainder(reference_phi[:, :, None] - target_phi[:, None, :] + torch.pi, 2.0 * torch.pi) - torch.pi
    delta_r = torch.sqrt(torch.clamp(deta * deta + dphi * dphi, min=0.0))
    valid_pair = reference_mask[:, :, None] & target_mask[:, None, :]
    return delta_r.masked_fill(~valid_pair, torch.finfo(delta_r.dtype).max)


def _matched_offline_tokens_for_hlt(
    *,
    reference_tokens: Any,
    reference_mask: Any,
    offline_tokens: Any,
    offline_mask: Any,
    max_delta_r: float | None = 0.4,
) -> tuple[Any, Any, Any]:
    torch = require_torch()
    reference_tokens = torch.as_tensor(reference_tokens, dtype=torch.float32)
    reference_mask = torch.as_tensor(reference_mask, dtype=torch.bool, device=reference_tokens.device)
    offline_tokens = torch.as_tensor(offline_tokens, dtype=reference_tokens.dtype, device=reference_tokens.device)
    offline_mask = torch.as_tensor(offline_mask, dtype=torch.bool, device=reference_tokens.device)
    if tuple(reference_tokens.shape[:2]) != tuple(reference_mask.shape):
        raise ValueError("reference_tokens and reference_mask leading shapes differ")
    if tuple(offline_tokens.shape[:2]) != tuple(offline_mask.shape):
        raise ValueError("offline_tokens and offline_mask leading shapes differ")
    if int(reference_tokens.shape[0]) != int(offline_tokens.shape[0]):
        raise ValueError("reference/offline batch sizes differ")
    if int(reference_tokens.shape[-1]) != int(offline_tokens.shape[-1]):
        raise ValueError("reference/offline feature dimensions differ")
    delta_r = _pairwise_delta_r(reference_tokens, reference_mask, offline_tokens, offline_mask)
    nearest_delta_r, nearest_index = delta_r.min(dim=2)
    gather_index = nearest_index[:, :, None].expand(-1, -1, int(offline_tokens.shape[-1]))
    matched_tokens = offline_tokens.gather(dim=1, index=gather_index)
    has_offline = offline_mask.any(dim=1)
    matched_mask = reference_mask & has_offline[:, None]
    if max_delta_r is not None:
        matched_mask = matched_mask & (nearest_delta_r <= float(max_delta_r))
    matched_tokens = torch.where(matched_mask[:, :, None], _nan_to_num_torch(matched_tokens), torch.zeros_like(matched_tokens))
    return matched_tokens, matched_mask, nearest_delta_r


def build_masked_subtoken_targets(
    target_tokens: Any,
    target_mask: Any,
    prediction_mask: Any,
    *,
    feature_config: SubtokenFeatureConfig | Mapping[str, Any] | None = None,
    target_mode: str = "hlt_self",
    reference_tokens: Any | None = None,
    reference_mask: Any | None = None,
    max_match_delta_r: float | None = 0.4,
) -> MaskedSubtokenTargetOutput:
    """Build target modality tensors for the sampled masked positions."""

    torch = require_torch()
    target_mode = normalize_masked_subtoken_target_mode(target_mode)
    config = _normalize_feature_config(feature_config)
    matching_metadata: dict[str, Any] = {
        "matching": "slot_aligned",
        "max_match_delta_r": None,
        "mean_nearest_delta_r": None,
        "matched_fraction": None,
    }
    if target_mode == SUBTOKEN_PART_MASKED_TARGET_OFFLINE:
        if reference_tokens is None or reference_mask is None:
            raise ValueError(
                "target_mode='offline' requires reference_tokens/reference_mask so offline targets can be "
                "nearest-matched to HLT slots. Use target_mode='offline_slot' only for explicit legacy slot alignment."
            )
        target_tokens, target_mask, nearest_delta_r = _matched_offline_tokens_for_hlt(
            reference_tokens=reference_tokens,
            reference_mask=reference_mask,
            offline_tokens=target_tokens,
            offline_mask=target_mask,
            max_delta_r=max_match_delta_r,
        )
        target_mask_float = target_mask.to(dtype=torch.float32)
        reference_mask_tensor = torch.as_tensor(reference_mask, dtype=torch.bool, device=nearest_delta_r.device)
        reference_mask_float = reference_mask_tensor.to(dtype=torch.float32)
        mean_nearest_delta_r = (
            (nearest_delta_r.to(dtype=torch.float32) * reference_mask_float).sum()
            / torch.clamp(reference_mask_float.sum(), min=1.0)
        )
        matched_fraction = target_mask_float.sum() / torch.clamp(reference_mask_float.sum(), min=1.0)
        matching_metadata = {
            "matching": "nearest_delta_r",
            "max_match_delta_r": None if max_match_delta_r is None else float(max_match_delta_r),
            "mean_nearest_delta_r": float(mean_nearest_delta_r.detach().cpu().item()),
            "matched_fraction": float(matched_fraction.detach().cpu().item()),
        }
    inputs = build_subtoken_inputs(
        torch.as_tensor(target_tokens, dtype=torch.float32),
        torch.as_tensor(target_mask, dtype=torch.bool),
        config=config,
    )
    modality_names = _default_modality_names()
    missing = [name for name in modality_names if name not in inputs.modality_values]
    if missing:
        raise ValueError(f"Missing target modalities: {missing}")
    prediction_mask = torch.as_tensor(prediction_mask, dtype=torch.bool, device=inputs.mask.device)
    if tuple(prediction_mask.shape[:2]) != tuple(inputs.mask.shape):
        raise ValueError("prediction_mask and target mask leading shapes differ")
    if int(prediction_mask.shape[2]) != len(modality_names):
        raise ValueError("prediction_mask modality dimension does not match default subtoken modalities")
    prediction_mask = prediction_mask & inputs.mask[:, :, None]
    return MaskedSubtokenTargetOutput(
        target_values={name: _nan_to_num_torch(inputs.modality_values[name]) for name in modality_names},
        prediction_mask=prediction_mask,
        particle_mask=inputs.mask,
        modality_names=modality_names,
        target_mode=target_mode,
        modality_feature_names={name: tuple(inputs.modality_feature_names[name]) for name in modality_names},
        matching_metadata=matching_metadata,
    )


def _make_head(input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float = 0.0):
    torch = require_torch()
    return torch.nn.Sequential(
        torch.nn.LayerNorm(int(input_dim)),
        torch.nn.Linear(int(input_dim), int(hidden_dim)),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(int(hidden_dim), int(output_dim)),
    )


class MaskedSubtokenPredictionHead(_ModuleBase):
    """Predict held-out modality values from a masked subtoken forward pass."""

    def __init__(
        self,
        config: SubtokenPartConfig | Mapping[str, Any] | None = None,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        torch = require_torch()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.modality_names = _default_modality_names()
        dims = subtoken_modality_input_dims(self.config.feature_config)
        missing = [name for name in self.modality_names if name not in dims]
        if missing:
            raise ValueError(f"Missing modality dims for masked prediction: {missing}")
        self.target_dims = {name: int(dims[name]) for name in self.modality_names}
        self.hidden_dim = int(hidden_dim or max(32, 2 * self.embed_dim))
        input_dim = 4 * self.embed_dim
        self.heads = torch.nn.ModuleDict(
            {
                name: _make_head(input_dim, self.hidden_dim, self.target_dims[name], dropout=float(self.config.dropout))
                for name in self.modality_names
            }
        )

    def forward(self, output: SubtokenClassifierOutput, mask_output: MaskedSubtokenMaskOutput) -> MaskedSubtokenPredictionOutput:
        torch = require_torch()
        particle_mask = output.particles.mask
        if tuple(mask_output.prediction_mask.shape[:2]) != tuple(particle_mask.shape):
            raise ValueError("masked prediction mask and model output particle masks differ")
        if output.context is None:
            context_tokens = torch.zeros_like(output.pooled.provisional_particles)
        else:
            context_tokens = output.context.context_tokens
        base_features = torch.cat(
            [
                output.pooled.provisional_particles,
                context_tokens,
                output.particles.particle_tokens,
                output.encoded.anchor,
            ],
            dim=-1,
        )
        base_features = _nan_to_num_torch(base_features)
        predictions = {
            name: torch.where(
                particle_mask[:, :, None],
                _nan_to_num_torch(self.heads[name](base_features)),
                torch.zeros((*particle_mask.shape, self.target_dims[name]), dtype=base_features.dtype, device=base_features.device),
            )
            for name in self.modality_names
        }
        return MaskedSubtokenPredictionOutput(
            predictions=predictions,
            prediction_mask=mask_output.prediction_mask.to(device=particle_mask.device) & particle_mask[:, :, None],
            particle_mask=particle_mask,
            modality_names=self.modality_names,
        )


def compute_masked_subtoken_loss(
    prediction: MaskedSubtokenPredictionOutput,
    target: MaskedSubtokenTargetOutput,
    *,
    huber_beta: float = 1.0,
) -> Any:
    torch = require_torch()
    if prediction.modality_names != target.modality_names:
        raise ValueError(f"Prediction/target modality names differ: {prediction.modality_names} vs {target.modality_names}")
    if tuple(prediction.prediction_mask.shape) != tuple(target.prediction_mask.shape):
        raise ValueError("Prediction/target masks have different shapes")
    total = None
    denom = None
    for index, name in enumerate(prediction.modality_names):
        pred = prediction.predictions[name]
        target_values = target.target_values[name].to(device=pred.device, dtype=pred.dtype)
        if tuple(pred.shape) != tuple(target_values.shape):
            raise ValueError(f"{name} prediction/target shapes differ: {tuple(pred.shape)} vs {tuple(target_values.shape)}")
        mask = (prediction.prediction_mask[:, :, index] & target.prediction_mask[:, :, index]).to(device=pred.device)
        feature_names = tuple(target.modality_feature_names.get(name, ()))
        calibrated_pred = calibrate_modality_predictions(name, pred, feature_names)
        calibrated_target = calibrate_modality_values(name, target_values, feature_names)
        loss = torch.nn.functional.smooth_l1_loss(
            calibrated_pred,
            calibrated_target,
            reduction="none",
            beta=float(huber_beta),
        )
        mask_float = mask[:, :, None].to(dtype=loss.dtype)
        weighted = (loss * mask_float).sum()
        count = torch.clamp(mask_float.sum() * int(pred.shape[-1]), min=1.0)
        total = weighted if total is None else total + weighted
        denom = count if denom is None else denom + count
    if total is None or denom is None:
        return prediction.particle_mask.new_zeros((), dtype=torch.float32)
    loss = total / torch.clamp(denom, min=1.0)
    if not bool(torch.isfinite(loss).detach().cpu().item()):
        raise FloatingPointError("masked subtoken loss is non-finite")
    return loss


__all__ = [
    "SUBTOKEN_PART_MASKED_CONTRACT",
    "SUBTOKEN_PART_MASKED_STEP",
    "SUBTOKEN_PART_MASKED_TARGET_HLT_SELF",
    "SUBTOKEN_PART_MASKED_TARGET_OFFLINE",
    "SUBTOKEN_PART_MASKED_TARGET_OFFLINE_SLOT",
    "SUBTOKEN_PART_MASKED_TARGET_MODES",
    "MaskedSubtokenMaskOutput",
    "MaskedSubtokenPredictionHead",
    "MaskedSubtokenPredictionOutput",
    "MaskedSubtokenTargetOutput",
    "build_masked_subtoken_targets",
    "compute_masked_subtoken_loss",
    "normalize_masked_subtoken_target_mode",
    "sample_masked_subtoken_modality_mask",
]

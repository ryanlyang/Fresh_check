"""Canonical raw HLT input handling for architecture-view branches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .config import ARCHITECTURE_VIEW_RAW_FEATURE_NAMES


ARCHITECTURE_VIEW_INPUTS_CONTRACT = "architecture_view_raw_hlt_inputs_v1"


@dataclass(frozen=True)
class ArchitectureViewInputs:
    """Sanitized raw HLT tokens plus masks and simple quality diagnostics."""

    tokens: Any
    mask: Any
    original_finite_mask: Any
    quality_features: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def raw_token_dim(self) -> int:
        return int(self.tokens.shape[2])

    def summary(self) -> dict[str, Any]:
        torch = require_torch()
        valid = self.mask.to(dtype=torch.bool)
        denom = torch.clamp(valid.sum().to(dtype=self.tokens.dtype), min=1.0)
        finite_fraction = (
            (self.original_finite_mask & valid).sum().to(dtype=self.tokens.dtype) / denom
        )
        return {
            "contract": ARCHITECTURE_VIEW_INPUTS_CONTRACT,
            "tokens_shape": list(self.tokens.shape),
            "mask_shape": list(self.mask.shape),
            "quality_features_shape": list(self.quality_features.shape),
            "valid_particle_count": int(valid.sum().detach().cpu().item()),
            "original_finite_valid_fraction": float(finite_fraction.detach().cpu().item()),
            **dict(self.metadata),
        }


def wrap_architecture_view_phi(delta_phi: Any) -> Any:
    """Wrap an angle tensor to ``[-pi, pi]`` using differentiable torch ops."""

    torch = require_torch()
    return torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))


def _coerce_tokens_and_mask(tokens: Any, mask: Any | None = None) -> tuple[Any, Any]:
    torch = require_torch()
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens, dtype=torch.float32)
    else:
        tokens = tokens.float()
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"tokens last dimension must be {RAW_TOKEN_DIM}, got {tokens.shape[-1]}")
    if mask is None:
        mask_tensor = torch.isfinite(tokens).all(dim=-1) & (tokens[:, :, 0] > 0.0)
    elif isinstance(mask, torch.Tensor):
        mask_tensor = mask.to(device=tokens.device, dtype=torch.bool)
    else:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=tokens.device)
    if int(mask_tensor.ndim) != 2 or tuple(mask_tensor.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask_tensor.shape)} does not match tokens shape {tuple(tokens.shape[:2])}")
    return tokens, mask_tensor


def sanitize_architecture_view_tokens(tokens: Any, mask: Any | None = None) -> ArchitectureViewInputs:
    """Return finite raw tokens with invalid particles zeroed.

    The original finite mask is recorded before sanitization so view branches can
    expose true input-quality signals instead of merely proving cleanup worked.
    """

    torch = require_torch()
    tokens, mask_tensor = _coerce_tokens_and_mask(tokens, mask)
    original_finite_mask = torch.isfinite(tokens).all(dim=-1)
    valid_mask = mask_tensor & original_finite_mask
    sanitized = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
    sanitized = torch.where(valid_mask[:, :, None], sanitized, torch.zeros_like(sanitized))
    sanitized_phi = wrap_architecture_view_phi(sanitized[:, :, 2])
    sanitized = sanitized.clone()
    sanitized[:, :, 2] = torch.where(valid_mask, sanitized_phi, torch.zeros_like(sanitized_phi))
    quality = build_architecture_view_quality_features(
        sanitized,
        valid_mask,
        original_finite_mask=original_finite_mask,
    )
    return ArchitectureViewInputs(
        tokens=sanitized,
        mask=valid_mask,
        original_finite_mask=original_finite_mask,
        quality_features=quality,
        metadata={
            "raw_feature_names": list(ARCHITECTURE_VIEW_RAW_FEATURE_NAMES),
            "source": "raw_hlt_tokens",
        },
    )


def build_architecture_view_quality_features(
    tokens: Any,
    mask: Any,
    *,
    original_finite_mask: Any | None = None,
) -> Any:
    """Build lightweight per-particle reliability hints from raw HLT tokens."""

    torch = require_torch()
    tokens, mask = _coerce_tokens_and_mask(tokens, mask)
    if original_finite_mask is None:
        original_finite_mask = torch.isfinite(tokens).all(dim=-1)
    else:
        original_finite_mask = original_finite_mask.to(device=tokens.device, dtype=torch.bool)
    pt = torch.clamp(tokens[:, :, 0], min=0.0)
    energy = torch.clamp(tokens[:, :, 3], min=0.0)
    charge = tokens[:, :, 4]
    pid_sum = tokens[:, :, 5:10].sum(dim=-1)
    d0err = torch.clamp(tokens[:, :, 11], min=0.0)
    dzerr = torch.clamp(tokens[:, :, 13], min=0.0)
    pt_positive = (pt > 0.0).to(dtype=tokens.dtype)
    energy_positive = (energy > 0.0).to(dtype=tokens.dtype)
    finite = original_finite_mask.to(dtype=tokens.dtype)
    pid_oneish = torch.exp(-torch.abs(pid_sum - 1.0))
    charged_hint = torch.clamp(torch.abs(charge), max=1.0)
    track_error = torch.log1p(d0err + dzerr)
    quality = torch.stack(
        [
            finite,
            pt_positive,
            energy_positive,
            pid_oneish,
            charged_hint,
            track_error,
        ],
        dim=-1,
    )
    return torch.where(mask[:, :, None], quality, torch.zeros_like(quality))


def architecture_view_inputs_from_batch(batch: Mapping[str, Any]) -> ArchitectureViewInputs:
    """Best-effort extraction from common HLT cache/data-loader batch names."""

    token_keys = ("tokens", "hlt_tokens", "raw_tokens", "particles")
    mask_keys = ("mask", "hlt_mask", "particle_mask", "masks")
    tokens = None
    mask = None
    for key in token_keys:
        if key in batch:
            tokens = batch[key]
            break
    for key in mask_keys:
        if key in batch:
            mask = batch[key]
            break
    if tokens is None:
        raise KeyError(f"batch does not contain raw tokens under any of {token_keys}")
    return sanitize_architecture_view_tokens(tokens, mask)

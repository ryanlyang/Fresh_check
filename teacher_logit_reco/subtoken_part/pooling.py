"""Local pooling from modality subtokens to provisional particle tokens."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import (
    SUBTOKEN_PART_POOL_CLS_TOKEN,
    SUBTOKEN_PART_POOL_LEARNED_QUERY,
    SUBTOKEN_PART_POOL_MEAN,
    SubtokenPartConfig,
    normalize_subtoken_pool_mode,
)
from .mixer import SubtokenMixerOutput

try:  # Keep imports lightweight on systems without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


SUBTOKEN_PART_LOCAL_POOL_STEP = "subtoken_part_step5_attention_pool"
SUBTOKEN_PART_LOCAL_POOL_CONTRACT = "provisional_particles_pool_weights_v1"


@dataclass(frozen=True)
class SubtokenPoolOutput:
    """Provisional particle tokens produced by local modality pooling."""

    provisional_particles: Any
    pool_weights: Any
    mask: Any
    modality_mask: Any
    modality_names: tuple[str, ...]
    local_tokens: Any
    pool_mode: str

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_LOCAL_POOL_CONTRACT,
            "pool_mode": self.pool_mode,
            "provisional_particles_shape": list(self.provisional_particles.shape),
            "pool_weights_shape": list(self.pool_weights.shape),
            "mask_shape": list(self.mask.shape),
            "modality_mask_shape": list(self.modality_mask.shape),
            "modality_names": list(self.modality_names),
        }


def _normalize_model_config(config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> SubtokenPartConfig:
    if config is None:
        return SubtokenPartConfig(num_classes=2)
    if isinstance(config, SubtokenPartConfig):
        return config
    return SubtokenPartConfig(**dict(config))


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _validate_local_tokens(local_tokens: Any, mask: Any, *, embed_dim: int) -> tuple[Any, Any]:
    local_tokens = _nan_to_num_torch(local_tokens.float())
    mask = mask.bool()
    if int(local_tokens.ndim) != 4:
        raise ValueError(
            f"local_tokens must have shape [batch, particles, modalities, embed_dim], got {tuple(local_tokens.shape)}"
        )
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape [batch, particles], got {tuple(mask.shape)}")
    if tuple(local_tokens.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"local_tokens/mask leading shapes differ: {tuple(local_tokens.shape[:2])} vs {tuple(mask.shape)}")
    if int(local_tokens.shape[-1]) != int(embed_dim):
        raise ValueError(f"local_tokens last dimension must be embed_dim={int(embed_dim)}, got {int(local_tokens.shape[-1])}")
    if int(local_tokens.shape[2]) <= 0:
        raise ValueError("local_tokens must contain at least one modality")
    return local_tokens, mask


def _validate_modality_mask(modality_mask: Any, local_tokens: Any) -> Any:
    modality_mask = modality_mask.bool()
    expected_shape = tuple(local_tokens.shape[:3])
    if tuple(modality_mask.shape) != expected_shape:
        raise ValueError(f"modality_mask shape {tuple(modality_mask.shape)} does not match {expected_shape}")
    return modality_mask


def _ensure_one_active_modality(modality_mask: Any, particle_mask: Any) -> Any:
    """Guarantee every valid particle has at least one unmasked modality."""

    torch = require_torch()
    missing_valid_particle = particle_mask & ~modality_mask.any(dim=2)
    fallback = torch.nn.functional.one_hot(
        torch.zeros_like(particle_mask, dtype=torch.long),
        num_classes=int(modality_mask.shape[2]),
    ).bool()
    return (modality_mask | (missing_valid_particle[:, :, None] & fallback)) & particle_mask[:, :, None]


def _zero_invalid_particles(values: Any, mask: Any) -> Any:
    torch = require_torch()
    return torch.where(mask[:, :, None], values, torch.zeros_like(values))


class SubtokenAttentionPool(_ModuleBase):
    """Pool locally mixed modality subtokens into one token per particle."""

    def __init__(
        self,
        config: SubtokenPartConfig | Mapping[str, Any] | None = None,
        *,
        pool_mode: str | None = None,
    ) -> None:
        super().__init__()
        torch = require_torch()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.pool_mode = normalize_subtoken_pool_mode(pool_mode or self.config.local_pool_mode)
        if self.pool_mode == SUBTOKEN_PART_POOL_LEARNED_QUERY:
            self.score_norm = torch.nn.LayerNorm(self.embed_dim)
            self.query = torch.nn.Parameter(torch.empty(self.embed_dim))
            torch.nn.init.normal_(self.query, mean=0.0, std=self.embed_dim ** -0.5)
        elif self.pool_mode == SUBTOKEN_PART_POOL_CLS_TOKEN:
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
            self.cls_attention = torch.nn.MultiheadAttention(
                self.embed_dim,
                num_heads=int(self.config.local_heads),
                dropout=float(self.config.attention_dropout),
                batch_first=True,
            )
            self.cls_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        local_tokens_or_output: Any,
        mask: Any | None = None,
        modality_mask: Any | None = None,
    ) -> SubtokenPoolOutput:
        torch = require_torch()
        if isinstance(local_tokens_or_output, SubtokenMixerOutput):
            local_tokens = local_tokens_or_output.local_tokens
            mask = local_tokens_or_output.mask
            modality_mask = local_tokens_or_output.modality_mask
            modality_names = local_tokens_or_output.modality_names
        else:
            if mask is None:
                raise ValueError("mask is required when passing raw local token tensors")
            local_tokens = local_tokens_or_output
            modality_names = None

        local_tokens, mask = _validate_local_tokens(local_tokens, mask, embed_dim=self.embed_dim)
        if modality_names is None:
            modality_names = tuple(f"modality_{index}" for index in range(int(local_tokens.shape[2])))
        batch_size, num_particles, num_modalities, embed_dim = local_tokens.shape
        if modality_mask is None:
            modality_mask = mask[:, :, None].expand(batch_size, num_particles, num_modalities)
        modality_mask = _validate_modality_mask(modality_mask, local_tokens)
        modality_mask = _ensure_one_active_modality(modality_mask & mask[:, :, None], mask)
        local_tokens = torch.where(modality_mask[:, :, :, None], local_tokens, torch.zeros_like(local_tokens))

        if self.pool_mode == SUBTOKEN_PART_POOL_MEAN:
            active_counts = modality_mask.sum(dim=2, keepdim=True).clamp(min=1).to(dtype=local_tokens.dtype)
            pool_weights = modality_mask.to(dtype=local_tokens.dtype) / active_counts
            provisional = (local_tokens * pool_weights[:, :, :, None]).sum(dim=2)
        elif self.pool_mode == SUBTOKEN_PART_POOL_LEARNED_QUERY:
            scores = (self.score_norm(local_tokens) * self.query.view(1, 1, 1, embed_dim)).sum(dim=-1)
            scores = scores / math.sqrt(float(embed_dim))
            scores = scores.masked_fill(~modality_mask, torch.finfo(scores.dtype).min)
            pool_weights = torch.softmax(scores, dim=-1)
            pool_weights = torch.where(modality_mask, pool_weights, torch.zeros_like(pool_weights))
            pool_weights = pool_weights / torch.clamp(pool_weights.sum(dim=2, keepdim=True), min=float(1.0e-12))
            provisional = (local_tokens * pool_weights[:, :, :, None]).sum(dim=2)
        elif self.pool_mode == SUBTOKEN_PART_POOL_CLS_TOKEN:
            flat_tokens = local_tokens.reshape(batch_size * num_particles, num_modalities, embed_dim)
            query = self.cls_token.expand(batch_size, num_particles, 1, embed_dim).reshape(batch_size * num_particles, 1, embed_dim)
            flat_modality_mask = modality_mask.reshape(batch_size * num_particles, num_modalities)
            safe_flat_modality_mask = flat_modality_mask.clone()
            all_masked = ~safe_flat_modality_mask.any(dim=1)
            safe_flat_modality_mask[all_masked, 0] = True
            attended, weights = self.cls_attention(
                query,
                flat_tokens,
                flat_tokens,
                key_padding_mask=~safe_flat_modality_mask,
                need_weights=True,
            )
            provisional = self.cls_norm(attended.squeeze(1)).reshape(batch_size, num_particles, embed_dim)
            pool_weights = weights.squeeze(1).reshape(batch_size, num_particles, num_modalities)
            pool_weights = torch.where(modality_mask, pool_weights, torch.zeros_like(pool_weights))
            pool_weights = pool_weights / torch.clamp(pool_weights.sum(dim=2, keepdim=True), min=float(1.0e-12))
        else:  # pragma: no cover - guarded by normalize_subtoken_pool_mode
            raise AssertionError(f"Unhandled subtoken pool mode {self.pool_mode!r}")

        provisional = _zero_invalid_particles(_nan_to_num_torch(provisional), mask)
        pool_weights = torch.where(modality_mask, _nan_to_num_torch(pool_weights), torch.zeros_like(pool_weights))
        pool_weights = torch.where(mask[:, :, None], pool_weights, torch.zeros_like(pool_weights))

        return SubtokenPoolOutput(
            provisional_particles=provisional,
            pool_weights=pool_weights,
            mask=mask,
            modality_mask=modality_mask,
            modality_names=tuple(modality_names),
            local_tokens=local_tokens,
            pool_mode=self.pool_mode,
        )


__all__ = [
    "SUBTOKEN_PART_LOCAL_POOL_CONTRACT",
    "SUBTOKEN_PART_LOCAL_POOL_STEP",
    "SubtokenAttentionPool",
    "SubtokenPoolOutput",
]

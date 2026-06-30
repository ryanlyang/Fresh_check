"""Provisional particle-token pooling for local-compression adapters.

Step 6 compresses the within-particle modality sequence into one provisional
particle token.  These tokens are not yet globally contextualized; Step 7 owns
jet-level particle attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .compressor import LocalCompressionCompressorOutput
from .config import (
    LOCAL_COMPRESSION_PART_CONTRACT,
    LOCAL_COMPRESSION_POOL_LEARNED_QUERY,
    LOCAL_COMPRESSION_POOL_MEAN,
    LocalCompressionPartConfig,
    normalize_local_compression_pool_mode,
)

try:  # Keep package imports cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_COMPRESSION_POOLING_STEP = "local_compression_part_step6_pooling"
LOCAL_COMPRESSION_POOLING_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_provisional_pooling_v1"


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _normalize_model_config(config: LocalCompressionPartConfig | Mapping[str, Any] | None = None) -> LocalCompressionPartConfig:
    if config is None:
        return LocalCompressionPartConfig()
    if isinstance(config, LocalCompressionPartConfig):
        return config
    return LocalCompressionPartConfig(**dict(config))


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
    expected = tuple(local_tokens.shape[:3])
    if tuple(modality_mask.shape) != expected:
        raise ValueError(f"modality_mask shape {tuple(modality_mask.shape)} does not match {expected}")
    return modality_mask


@dataclass(frozen=True)
class LocalCompressionPoolOutput:
    """Provisional particle tokens and modality pooling diagnostics."""

    local_particle_token: Any
    pool_weights: Any
    mask: Any
    modality_mask: Any
    modality_names: tuple[str, ...]
    local_tokens: Any
    pool_mode: str

    def __post_init__(self) -> None:
        torch = require_torch()
        if int(self.local_particle_token.ndim) != 3:
            raise ValueError("local_particle_token must have shape [batch, particles, embed_dim]")
        batch_size, num_particles, embed_dim = tuple(self.local_particle_token.shape)
        if int(self.local_tokens.ndim) != 4:
            raise ValueError("local_tokens must have shape [batch, particles, modalities, embed_dim]")
        num_modalities = int(self.local_tokens.shape[2])
        expected_mask = (batch_size, num_particles)
        expected_weights = (batch_size, num_particles, num_modalities)
        expected_tokens = (batch_size, num_particles, num_modalities, embed_dim)
        if tuple(self.mask.shape) != expected_mask:
            raise ValueError(f"mask has shape {tuple(self.mask.shape)}, expected {expected_mask}")
        if tuple(self.pool_weights.shape) != expected_weights:
            raise ValueError(f"pool_weights has shape {tuple(self.pool_weights.shape)}, expected {expected_weights}")
        if tuple(self.modality_mask.shape) != expected_weights:
            raise ValueError(f"modality_mask has shape {tuple(self.modality_mask.shape)}, expected {expected_weights}")
        if tuple(self.local_tokens.shape) != expected_tokens:
            raise ValueError(f"local_tokens has shape {tuple(self.local_tokens.shape)}, expected {expected_tokens}")
        if len(tuple(self.modality_names)) != num_modalities:
            raise ValueError("modality_names length must match modality dimension")
        if not bool(torch.isfinite(self.local_particle_token).all()):
            raise ValueError("local_particle_token contains non-finite values")
        if not bool(torch.isfinite(self.pool_weights).all()):
            raise ValueError("pool_weights contain non-finite values")
        inactive_weight = torch.where(self.modality_mask, torch.zeros_like(self.pool_weights), self.pool_weights.abs())
        if float(inactive_weight.max().detach().cpu().item()) > 1.0e-6:
            raise ValueError("inactive modalities must have zero pool weight")
        active_particle_mask = self.modality_mask.any(dim=-1)
        weight_sums = self.pool_weights.sum(dim=-1)
        expected_sums = active_particle_mask.to(dtype=weight_sums.dtype)
        if not bool(torch.allclose(weight_sums, expected_sums, atol=1.0e-5, rtol=0.0)):
            raise ValueError("pool weights must sum to one for active particles and zero otherwise")
        invalid_tokens = torch.where(self.mask[:, :, None], torch.zeros_like(self.local_particle_token), self.local_particle_token.abs())
        if float(invalid_tokens.max().detach().cpu().item()) > 1.0e-6:
            raise ValueError("invalid particles must have zero local_particle_token")
        object.__setattr__(self, "modality_names", tuple(self.modality_names))
        object.__setattr__(self, "pool_mode", normalize_local_compression_pool_mode(self.pool_mode))

    @property
    def batch_size(self) -> int:
        return int(self.local_particle_token.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.local_particle_token.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self.local_particle_token.shape[2])

    @property
    def num_modalities(self) -> int:
        return int(self.pool_weights.shape[2])

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_COMPRESSION_POOLING_CONTRACT,
            "pool_mode": self.pool_mode,
            "local_particle_token_shape": list(self.local_particle_token.shape),
            "pool_weights_shape": list(self.pool_weights.shape),
            "mask_shape": list(self.mask.shape),
            "modality_mask_shape": list(self.modality_mask.shape),
            "modality_names": list(self.modality_names),
            "active_particle_count": int(self.modality_mask.any(dim=-1).detach().cpu().sum().item()),
            "active_modality_count": int(self.modality_mask.detach().cpu().sum().item()),
        }


class LocalCompressionProvisionalPooler(_ModuleBase):
    """Masked pooling from local modality tokens to provisional particle tokens."""

    def __init__(self, config: LocalCompressionPartConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.pool_mode = normalize_local_compression_pool_mode(self.config.pool_mode)
        if self.pool_mode == LOCAL_COMPRESSION_POOL_LEARNED_QUERY:
            self.query = torch.nn.Parameter(torch.zeros(self.embed_dim))
            self.score_norm = torch.nn.LayerNorm(self.embed_dim)
        else:
            self.register_parameter("query", None)
            self.score_norm = torch.nn.Identity()

    def _mean_pool_weights(self, modality_mask: Any) -> Any:
        torch = require_torch()
        counts = modality_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(dtype=torch.float32)
        weights = modality_mask.to(dtype=torch.float32) / counts
        return torch_where_modality(modality_mask, weights)

    def _learned_query_pool_weights(self, local_tokens: Any, modality_mask: Any) -> Any:
        torch = require_torch()
        scores = (self.score_norm(local_tokens) * self.query.view(1, 1, 1, -1)).sum(dim=-1)
        scores = scores / math.sqrt(float(self.embed_dim))
        masked_scores = scores.masked_fill(~modality_mask, -1.0e9)
        weights = torch.softmax(masked_scores, dim=-1)
        weights = torch.where(modality_mask, weights, torch.zeros_like(weights))
        normalizer = weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-12)
        weights = torch.where(modality_mask.any(dim=-1, keepdim=True), weights / normalizer, torch.zeros_like(weights))
        return weights

    def forward(
        self,
        compressed_or_tokens: Any,
        mask: Any | None = None,
        modality_mask: Any | None = None,
    ) -> LocalCompressionPoolOutput:
        torch = require_torch()
        if isinstance(compressed_or_tokens, LocalCompressionCompressorOutput):
            local_tokens = compressed_or_tokens.local_tokens
            mask = compressed_or_tokens.mask
            modality_mask = compressed_or_tokens.modality_mask
            modality_names = compressed_or_tokens.modality_names
        else:
            if mask is None:
                raise ValueError("mask is required when passing raw local token tensors")
            local_tokens = compressed_or_tokens
            modality_names = tuple(f"modality_{index}" for index in range(int(local_tokens.shape[2])))

        local_tokens, mask = _validate_local_tokens(local_tokens, mask, embed_dim=self.embed_dim)
        batch_size, num_particles, num_modalities, _embed_dim = tuple(local_tokens.shape)
        if modality_mask is None:
            modality_mask = mask[:, :, None].expand(batch_size, num_particles, num_modalities)
        modality_mask = _validate_modality_mask(modality_mask, local_tokens)
        modality_mask = modality_mask & mask[:, :, None]
        masked_tokens = torch.where(modality_mask[:, :, :, None], local_tokens, torch.zeros_like(local_tokens))

        if self.pool_mode == LOCAL_COMPRESSION_POOL_MEAN:
            pool_weights = self._mean_pool_weights(modality_mask)
        elif self.pool_mode == LOCAL_COMPRESSION_POOL_LEARNED_QUERY:
            pool_weights = self._learned_query_pool_weights(masked_tokens, modality_mask)
        else:  # pragma: no cover - normalized config should prevent this.
            raise ValueError(f"unsupported pool_mode {self.pool_mode!r}")

        local_particle_token = (pool_weights[:, :, :, None] * masked_tokens).sum(dim=2)
        local_particle_token = _nan_to_num_torch(local_particle_token)
        local_particle_token = torch.where(mask[:, :, None], local_particle_token, torch.zeros_like(local_particle_token))
        return LocalCompressionPoolOutput(
            local_particle_token=local_particle_token,
            pool_weights=pool_weights,
            mask=mask,
            modality_mask=modality_mask,
            modality_names=tuple(modality_names),
            local_tokens=masked_tokens,
            pool_mode=self.pool_mode,
        )


def torch_where_modality(modality_mask: Any, weights: Any) -> Any:
    torch = require_torch()
    return torch.where(modality_mask, weights, torch.zeros_like(weights))


__all__ = [
    "LOCAL_COMPRESSION_POOLING_CONTRACT",
    "LOCAL_COMPRESSION_POOLING_STEP",
    "LocalCompressionPoolOutput",
    "LocalCompressionProvisionalPooler",
]

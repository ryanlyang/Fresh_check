"""Particle-context stage for reliability-gated subtoken ParT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import SubtokenPartConfig
from .pooling import SubtokenPoolOutput

try:  # Keep imports lightweight on systems without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


SUBTOKEN_PART_CONTEXT_STEP = "subtoken_part_step6_particle_context"
SUBTOKEN_PART_CONTEXT_CONTRACT = "masked_particle_context_tokens_v1"


@dataclass(frozen=True)
class ParticleContextOutput:
    """Contextualized provisional particle tokens."""

    context_tokens: Any
    mask: Any
    input_particles: Any

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_CONTEXT_CONTRACT,
            "context_tokens_shape": list(self.context_tokens.shape),
            "input_particles_shape": list(self.input_particles.shape),
            "mask_shape": list(self.mask.shape),
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


def _validate_particle_inputs(particles: Any, mask: Any, *, embed_dim: int) -> tuple[Any, Any]:
    particles = _nan_to_num_torch(particles.float())
    mask = mask.bool()
    if int(particles.ndim) != 3:
        raise ValueError(f"particles must have shape [batch, particles, embed_dim], got {tuple(particles.shape)}")
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape [batch, particles], got {tuple(mask.shape)}")
    if tuple(particles.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"particles/mask leading shapes differ: {tuple(particles.shape[:2])} vs {tuple(mask.shape)}")
    if int(particles.shape[-1]) != int(embed_dim):
        raise ValueError(f"particles last dimension must be embed_dim={int(embed_dim)}, got {int(particles.shape[-1])}")
    return particles, mask


class ParticleContextTransformer(_ModuleBase):
    """Lightweight transformer that gives provisional particles jet context."""

    def __init__(self, config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.num_layers = int(self.config.context_layers)
        self.num_heads = int(self.config.context_heads)
        if self.num_layers <= 0:
            raise ValueError("context_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("context_heads must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by context_heads")

        transformer_dropout = max(float(self.config.dropout), float(self.config.attention_dropout))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.embed_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.output_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(self, particles_or_output: Any, mask: Any | None = None) -> ParticleContextOutput:
        torch = require_torch()
        if isinstance(particles_or_output, SubtokenPoolOutput):
            particles = particles_or_output.provisional_particles
            mask = particles_or_output.mask
        else:
            if mask is None:
                raise ValueError("mask is required when passing raw provisional particle tensors")
            particles = particles_or_output

        particles, mask = _validate_particle_inputs(particles, mask, embed_dim=self.embed_dim)
        masked_particles = torch.where(mask[:, :, None], particles, torch.zeros_like(particles))

        safe_mask = mask.clone()
        all_masked = ~safe_mask.any(dim=1)
        safe_mask[all_masked, 0] = True
        context_tokens = self.encoder(masked_particles, src_key_padding_mask=~safe_mask)
        context_tokens = self.output_norm(context_tokens)
        context_tokens = torch.where(mask[:, :, None], _nan_to_num_torch(context_tokens), torch.zeros_like(context_tokens))

        return ParticleContextOutput(
            context_tokens=context_tokens,
            mask=mask,
            input_particles=masked_particles,
        )


__all__ = [
    "SUBTOKEN_PART_CONTEXT_CONTRACT",
    "SUBTOKEN_PART_CONTEXT_STEP",
    "ParticleContextOutput",
    "ParticleContextTransformer",
]

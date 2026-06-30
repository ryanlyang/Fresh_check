"""Shallow particle-context block for local-compression adapters.

Step 7 gives provisional particle tokens a small amount of jet-level context
before reliability gates and delta-F prediction.  This is not the main ParT
classifier.  It is a lightweight, mask-safe context stage used only by the
adapter path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import LOCAL_COMPRESSION_PART_CONTRACT, LocalCompressionPartConfig
from .pooling import LocalCompressionPoolOutput

try:  # Keep package import cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_COMPRESSION_CONTEXT_STEP = "local_compression_part_step7_particle_context"
LOCAL_COMPRESSION_CONTEXT_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_particle_context_v1"


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


def _validate_particle_tokens(tokens: Any, mask: Any, *, embed_dim: int) -> tuple[Any, Any]:
    tokens = _nan_to_num_torch(tokens.float())
    mask = mask.bool()
    if int(tokens.ndim) != 3:
        raise ValueError(f"particle tokens must have shape [batch, particles, embed_dim], got {tuple(tokens.shape)}")
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape [batch, particles], got {tuple(mask.shape)}")
    if tuple(tokens.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"particle tokens/mask leading shapes differ: {tuple(tokens.shape[:2])} vs {tuple(mask.shape)}")
    if int(tokens.shape[-1]) != int(embed_dim):
        raise ValueError(f"particle tokens last dimension must be embed_dim={int(embed_dim)}, got {int(tokens.shape[-1])}")
    return tokens, mask


class _ParticleContextTransformerLayer(_ModuleBase):
    """Pre-norm particle mixer with separated attention and residual dropout."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        torch = require_torch()
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(attention_dropout),
            batch_first=True,
        )
        self.norm1 = torch.nn.LayerNorm(int(embed_dim))
        self.norm2 = torch.nn.LayerNorm(int(embed_dim))
        self.linear1 = torch.nn.Linear(int(embed_dim), int(ff_dim))
        self.linear2 = torch.nn.Linear(int(ff_dim), int(embed_dim))
        self.activation = torch.nn.GELU()
        self.residual_dropout = torch.nn.Dropout(float(dropout))
        self.ffn_dropout = torch.nn.Dropout(float(dropout))

    def forward(self, tokens: Any, *, key_padding_mask: Any) -> Any:
        normed = self.norm1(tokens)
        attended, _weights = self.self_attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.residual_dropout(attended)
        normed = self.norm2(tokens)
        ffn = self.linear2(self.ffn_dropout(self.activation(self.linear1(normed))))
        return tokens + self.residual_dropout(ffn)


@dataclass(frozen=True)
class ParticleContextOutput:
    """Contextualized provisional particle tokens."""

    context_tokens: Any
    mask: Any
    input_particles: Any

    def __post_init__(self) -> None:
        torch = require_torch()
        if int(self.context_tokens.ndim) != 3:
            raise ValueError("context_tokens must have shape [batch, particles, embed_dim]")
        if tuple(self.input_particles.shape) != tuple(self.context_tokens.shape):
            raise ValueError("input_particles shape must match context_tokens")
        if tuple(self.mask.shape) != tuple(self.context_tokens.shape[:2]):
            raise ValueError("mask shape must match context_tokens leading dimensions")
        if not bool(torch.isfinite(self.context_tokens).all()):
            raise ValueError("context_tokens contain non-finite values")
        if not bool(torch.isfinite(self.input_particles).all()):
            raise ValueError("input_particles contain non-finite values")
        invalid_tokens = torch.where(self.mask[:, :, None], torch.zeros_like(self.context_tokens), self.context_tokens.abs())
        if float(invalid_tokens.max().detach().cpu().item()) > 1.0e-6:
            raise ValueError("invalid particles must have zero context_tokens")

    @property
    def batch_size(self) -> int:
        return int(self.context_tokens.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.context_tokens.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self.context_tokens.shape[2])

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_COMPRESSION_CONTEXT_CONTRACT,
            "context_tokens_shape": list(self.context_tokens.shape),
            "input_particles_shape": list(self.input_particles.shape),
            "mask_shape": list(self.mask.shape),
            "valid_particle_count": int(self.mask.detach().cpu().sum().item()),
        }


class ParticleContextBlock(_ModuleBase):
    """Lightweight global self-attention over provisional particle tokens."""

    def __init__(self, config: LocalCompressionPartConfig | Mapping[str, Any] | None = None) -> None:
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
        ff_dim = int(round(float(self.config.mlp_ratio) * self.embed_dim))
        if ff_dim <= 0:
            raise ValueError("particle context feed-forward dimension must be positive")
        self.layers = torch.nn.ModuleList(
            [
                _ParticleContextTransformerLayer(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ff_dim=ff_dim,
                    dropout=float(self.config.dropout),
                    attention_dropout=float(self.config.attention_dropout),
                )
                for _index in range(self.num_layers)
            ]
        )
        self.output_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(self, particles_or_output: Any, mask: Any | None = None) -> ParticleContextOutput:
        torch = require_torch()
        if isinstance(particles_or_output, LocalCompressionPoolOutput):
            particles = particles_or_output.local_particle_token
            mask = particles_or_output.mask
        elif mask is None and hasattr(particles_or_output, "local_particle_token") and hasattr(particles_or_output, "mask"):
            particles = particles_or_output.local_particle_token
            mask = particles_or_output.mask
        elif mask is None and hasattr(particles_or_output, "provisional_particles") and hasattr(particles_or_output, "mask"):
            particles = particles_or_output.provisional_particles
            mask = particles_or_output.mask
        else:
            if mask is None:
                raise ValueError("mask is required when passing raw particle-token tensors")
            particles = particles_or_output

        particles, mask = _validate_particle_tokens(particles, mask, embed_dim=self.embed_dim)
        input_particles = torch.where(mask[:, :, None], particles, torch.zeros_like(particles))

        safe_mask = mask.clone()
        all_masked = ~safe_mask.any(dim=1)
        if bool(all_masked.any().detach().cpu().item()):
            safe_mask[all_masked, 0] = True
        key_padding_mask = ~safe_mask
        context_tokens = input_particles
        for layer in self.layers:
            context_tokens = layer(context_tokens, key_padding_mask=key_padding_mask)
        context_tokens = self.output_norm(_nan_to_num_torch(context_tokens))
        context_tokens = torch.where(mask[:, :, None], context_tokens, torch.zeros_like(context_tokens))
        return ParticleContextOutput(
            context_tokens=context_tokens,
            mask=mask,
            input_particles=input_particles,
        )


__all__ = [
    "LOCAL_COMPRESSION_CONTEXT_CONTRACT",
    "LOCAL_COMPRESSION_CONTEXT_STEP",
    "ParticleContextBlock",
    "ParticleContextOutput",
]

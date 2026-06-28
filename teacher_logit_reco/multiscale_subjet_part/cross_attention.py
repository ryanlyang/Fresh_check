"""Particle-subjet cross-attention and residual feature readback.

Step 7 is the bridge from the multi-scale subjet hierarchy back into the exact
canonical Particle Transformer input stream.  It does not alter Lorentz vectors
or the ParT mask; it only predicts a gated residual ``delta_F`` for canonical PF
features.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .features import (
    CANONICAL_PART_FEATURE_NAMES,
    CanonicalPartInputs,
    MultiscaleSubjetFeatureConfig,
    build_canonical_part_inputs,
)
from .subjet_transformer import MultiScaleSubjetTransformerOutput


try:  # Keep imports lightweight on systems without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


MULTISCALE_SUBJET_CROSS_ATTENTION_CONTRACT = "multiscale_subjet_particle_cross_attention_readback_v1"
MULTISCALE_SUBJET_CROSS_ATTENTION_STEP = "multiscale_subjet_part_step7_particle_subjet_cross_attention"


@dataclass(frozen=True)
class ParticleSubjetCrossAttentionConfig:
    """Configuration for particle-subjet readback into canonical ParT features."""

    feature_dim: int = len(CANONICAL_PART_FEATURE_NAMES)
    subjet_token_dim: int = 128
    hidden_dim: int = 128
    num_heads: int = 4
    delta_hidden_dim: int = 256
    dropout: float = 0.05
    attention_dropout: float = 0.05
    residual_gamma_init: float = 0.0
    max_constits: int | None = None
    weight_threshold: float = 0.0
    use_particle_reads_subjets: bool = True
    use_subjets_read_particles: bool = True

    def __post_init__(self) -> None:
        for name in ("feature_dim", "subjet_token_dim", "hidden_dim", "num_heads", "delta_hidden_dim"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if int(self.hidden_dim) % int(self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be finite and satisfy 0 <= {name} < 1")
            object.__setattr__(self, name, value)
        residual_gamma_init = float(self.residual_gamma_init)
        if not math.isfinite(residual_gamma_init):
            raise ValueError("residual_gamma_init must be finite")
        object.__setattr__(self, "residual_gamma_init", residual_gamma_init)
        max_constits = self.max_constits
        if max_constits is not None:
            max_constits = int(max_constits)
            if max_constits <= 0:
                raise ValueError("max_constits must be positive when provided")
        object.__setattr__(self, "max_constits", max_constits)
        weight_threshold = float(self.weight_threshold)
        if not math.isfinite(weight_threshold) or weight_threshold < 0.0:
            raise ValueError("weight_threshold must be non-negative and finite")
        object.__setattr__(self, "weight_threshold", weight_threshold)
        object.__setattr__(self, "use_particle_reads_subjets", bool(self.use_particle_reads_subjets))
        object.__setattr__(self, "use_subjets_read_particles", bool(self.use_subjets_read_particles))


@dataclass(frozen=True)
class ParticleSubjetCrossAttentionOutput:
    """Canonical ParT inputs after gated particle-subjet feature readback."""

    canonical_inputs: CanonicalPartInputs
    adapted_features: Any
    feature_delta: Any
    effective_feature_delta: Any
    particle_context: Any
    updated_subjet_tokens: Any
    particle_to_subjet_attention: Any | None
    subjet_to_particle_attention: Any | None
    gamma_F: Any
    diagnostics: Mapping[str, Any]
    config: ParticleSubjetCrossAttentionConfig

    @property
    def part_points(self) -> Any:
        return self.canonical_inputs.points

    @property
    def part_features(self) -> Any:
        return self.adapted_features.transpose(1, 2).contiguous()

    @property
    def part_lorentz_vectors(self) -> Any:
        return self.canonical_inputs.lorentz_vectors

    @property
    def part_mask(self) -> Any:
        return self.canonical_inputs.mask

    def part_inputs(self) -> dict[str, Any]:
        return {
            "points": self.part_points,
            "features": self.part_features,
            "lorentz_vectors": self.part_lorentz_vectors,
            "mask": self.part_mask,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "contract": MULTISCALE_SUBJET_CROSS_ATTENTION_CONTRACT,
            "step": MULTISCALE_SUBJET_CROSS_ATTENTION_STEP,
            "canonical_features_shape": list(self.canonical_inputs.features.shape),
            "adapted_features_shape": list(self.adapted_features.shape),
            "feature_delta_shape": list(self.feature_delta.shape),
            "particle_context_shape": list(self.particle_context.shape),
            "updated_subjet_tokens_shape": list(self.updated_subjet_tokens.shape),
            "lorentz_vectors_unchanged": True,
            "diagnostics": dict(self.diagnostics),
        }


def normalize_particle_subjet_cross_attention_config(
    config: ParticleSubjetCrossAttentionConfig | Mapping[str, Any] | None = None,
) -> ParticleSubjetCrossAttentionConfig:
    if config is None:
        return ParticleSubjetCrossAttentionConfig()
    if isinstance(config, ParticleSubjetCrossAttentionConfig):
        return config
    return ParticleSubjetCrossAttentionConfig(**dict(config))


def _masked_mean(value: Any, mask: Any) -> float:
    torch = require_torch()
    weight = mask.to(dtype=value.dtype)
    denom = torch.clamp(weight.sum(), min=1.0)
    return float((value * weight).sum().detach().cpu().item() / float(denom.detach().cpu().item()))


def _attention_entropy(attention: Any | None, query_mask: Any, eps: float = 1.0e-8) -> float | None:
    if attention is None:
        return None
    torch = require_torch()
    entropy = -(attention * torch.log(torch.clamp(attention, min=float(eps)))).sum(dim=-1).mean(dim=1)
    return _masked_mean(entropy, query_mask)


class MaskedCrossAttention(_ModuleBase):
    """Small batch-first multi-head cross-attention with explicit masks."""

    def __init__(self, *, hidden_dim: int, num_heads: int, dropout: float = 0.0, attention_dropout: float = 0.0) -> None:
        torch = require_torch()
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = torch.nn.Dropout(float(attention_dropout))
        self.out_dropout = torch.nn.Dropout(float(dropout))
        self.query_norm = torch.nn.LayerNorm(hidden_dim)
        self.key_norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, query_tokens: Any, key_value_tokens: Any, query_mask: Any, key_value_mask: Any) -> tuple[Any, Any]:
        torch = require_torch()
        batch_size, num_queries, _ = query_tokens.shape
        num_keys = int(key_value_tokens.shape[1])
        q = self.q_proj(self.query_norm(query_tokens)).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.key_norm(key_value_tokens)).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.key_norm(key_value_tokens)).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        key_mask = key_value_mask[:, None, None, :]
        query_mask_4d = query_mask[:, None, :, None]
        scores = scores.masked_fill(~key_mask, -1.0e4)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(key_mask & query_mask_4d, attention, torch.zeros_like(attention))
        dropped_attention = self.attn_dropout(attention)
        context = torch.matmul(dropped_attention, v).transpose(1, 2).reshape(batch_size, num_queries, self.hidden_dim)
        context = self.out_dropout(self.out_proj(context))
        context = torch.where(query_mask[:, :, None], context, torch.zeros_like(context))
        return context, attention


class ParticleSubjetCrossAttentionReadback(_ModuleBase):
    """Cross-attend particles and subjets, then predict gated ``delta_F``."""

    def __init__(self, config: ParticleSubjetCrossAttentionConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = normalize_particle_subjet_cross_attention_config(config)
        self.particle_embed = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.feature_dim)),
            torch.nn.Linear(int(self.config.feature_dim), int(self.config.hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(self.config.hidden_dim), int(self.config.hidden_dim)),
        )
        self.subjet_embed = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.subjet_token_dim)),
            torch.nn.Linear(int(self.config.subjet_token_dim), int(self.config.hidden_dim)),
        )
        self.particle_reads_subjets = MaskedCrossAttention(
            hidden_dim=int(self.config.hidden_dim),
            num_heads=int(self.config.num_heads),
            dropout=float(self.config.dropout),
            attention_dropout=float(self.config.attention_dropout),
        )
        self.subjets_read_particles = MaskedCrossAttention(
            hidden_dim=int(self.config.hidden_dim),
            num_heads=int(self.config.num_heads),
            dropout=float(self.config.dropout),
            attention_dropout=float(self.config.attention_dropout),
        )
        self.subjet_update_norm = torch.nn.LayerNorm(int(self.config.hidden_dim))
        self.delta_head = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * int(self.config.hidden_dim)),
            torch.nn.Linear(2 * int(self.config.hidden_dim), int(self.config.delta_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(self.config.delta_hidden_dim), int(self.config.feature_dim)),
        )
        self.gamma_F = torch.nn.Parameter(torch.tensor(float(self.config.residual_gamma_init), dtype=torch.float32))

    def forward(
        self,
        tokens: Any,
        mask: Any,
        subjet_output: MultiScaleSubjetTransformerOutput,
        *,
        feature_config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None = None,
    ) -> ParticleSubjetCrossAttentionOutput:
        torch = require_torch()
        tokens = tokens.float()
        mask = mask.to(device=tokens.device, dtype=torch.bool)
        max_constits = int(tokens.shape[1]) if self.config.max_constits is None else int(self.config.max_constits)
        canonical = build_canonical_part_inputs(
            tokens,
            mask,
            max_constits=max_constits,
            weight_threshold=float(self.config.weight_threshold),
            config=feature_config,
        )
        features = canonical.feature_rows()
        if int(features.shape[-1]) != int(self.config.feature_dim):
            raise ValueError(
                f"canonical feature dim {int(features.shape[-1])} does not match configured "
                f"{int(self.config.feature_dim)}"
            )
        particle_mask = canonical.mask.squeeze(1).to(device=tokens.device, dtype=torch.bool)
        subjet_tokens = subjet_output.subjet_tokens.to(device=tokens.device, dtype=features.dtype)
        subjet_mask = subjet_output.subjet_mask.to(device=tokens.device, dtype=torch.bool)
        if int(subjet_tokens.shape[-1]) != int(self.config.subjet_token_dim):
            raise ValueError(
                f"subjet token dim {int(subjet_tokens.shape[-1])} does not match configured "
                f"{int(self.config.subjet_token_dim)}"
            )

        particle_hidden = self.particle_embed(features)
        particle_hidden = torch.where(particle_mask[:, :, None], particle_hidden, torch.zeros_like(particle_hidden))
        subjet_hidden = self.subjet_embed(subjet_tokens)
        subjet_hidden = torch.where(subjet_mask[:, :, None], subjet_hidden, torch.zeros_like(subjet_hidden))

        if bool(self.config.use_particle_reads_subjets):
            particle_context, particle_to_subjet_attention = self.particle_reads_subjets(
                particle_hidden,
                subjet_hidden,
                particle_mask,
                subjet_mask,
            )
        else:
            particle_context = torch.zeros_like(particle_hidden)
            particle_to_subjet_attention = None

        if bool(self.config.use_subjets_read_particles):
            subjet_context, subjet_to_particle_attention = self.subjets_read_particles(
                subjet_hidden,
                particle_hidden,
                subjet_mask,
                particle_mask,
            )
            updated_subjets = self.subjet_update_norm(subjet_hidden + subjet_context)
            updated_subjets = torch.where(subjet_mask[:, :, None], updated_subjets, torch.zeros_like(updated_subjets))
        else:
            subjet_to_particle_attention = None
            updated_subjets = subjet_hidden

        delta_input = torch.cat([particle_hidden, particle_context], dim=-1)
        feature_delta = self.delta_head(delta_input)
        feature_delta = torch.where(particle_mask[:, :, None], feature_delta, torch.zeros_like(feature_delta))
        gamma = self.gamma_F.to(device=features.device, dtype=features.dtype)
        effective_delta = gamma * feature_delta
        adapted_features = features + effective_delta
        adapted_features = torch.where(particle_mask[:, :, None], adapted_features, torch.zeros_like(adapted_features))
        for name, value in (
            ("particle_context", particle_context),
            ("updated_subjet_tokens", updated_subjets),
            ("feature_delta", feature_delta),
            ("adapted_features", adapted_features),
        ):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"{name} contains non-finite values")

        feature_norm = features.norm(dim=-1)
        delta_norm = feature_delta.norm(dim=-1)
        effective_delta_norm = effective_delta.norm(dim=-1)
        diagnostics = {
            "step": MULTISCALE_SUBJET_CROSS_ATTENTION_STEP,
            "contract": MULTISCALE_SUBJET_CROSS_ATTENTION_CONTRACT,
            "gamma_F": float(gamma.detach().cpu().item()),
            "valid_particle_fraction": float(particle_mask.float().mean().detach().cpu().item()),
            "valid_subjet_fraction": float(subjet_mask.float().mean().detach().cpu().item()),
            "feature_delta_norm_mean": _masked_mean(delta_norm, particle_mask),
            "effective_feature_delta_norm_mean": _masked_mean(effective_delta_norm, particle_mask),
            "feature_norm_mean": _masked_mean(feature_norm, particle_mask),
            "particle_to_subjet_attention_entropy_mean": _attention_entropy(particle_to_subjet_attention, particle_mask),
            "subjet_to_particle_attention_entropy_mean": _attention_entropy(subjet_to_particle_attention, subjet_mask),
            "lorentz_vectors_unchanged": True,
        }
        return ParticleSubjetCrossAttentionOutput(
            canonical_inputs=canonical,
            adapted_features=adapted_features,
            feature_delta=feature_delta,
            effective_feature_delta=effective_delta,
            particle_context=particle_context,
            updated_subjet_tokens=updated_subjets,
            particle_to_subjet_attention=particle_to_subjet_attention,
            subjet_to_particle_attention=subjet_to_particle_attention,
            gamma_F=self.gamma_F,
            diagnostics=diagnostics,
            config=self.config,
        )


MultiscaleParticleSubjetCrossAttentionConfig = ParticleSubjetCrossAttentionConfig
MultiscaleParticleSubjetCrossAttentionOutput = ParticleSubjetCrossAttentionOutput
MultiscaleParticleSubjetCrossAttentionReadback = ParticleSubjetCrossAttentionReadback

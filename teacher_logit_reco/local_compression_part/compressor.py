"""Within-particle local modality compressor.

Step 5 mixes modality subtokens inside each particle only.  It does not perform
jet-level reasoning; that belongs to the later shallow context and exact ParT
backbone stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import LOCAL_COMPRESSION_PART_CONTRACT, LocalCompressionPartConfig
from .subtokens import LocalCompressionSubtokenOutput

try:  # Keep package imports cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_COMPRESSION_COMPRESSOR_STEP = "local_compression_part_step5_compressor"
LOCAL_COMPRESSION_COMPRESSOR_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_local_modality_compressor_v1"


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


def _validate_subtokens(subtokens: Any, mask: Any, *, embed_dim: int) -> tuple[Any, Any]:
    subtokens = _nan_to_num_torch(subtokens.float())
    mask = mask.bool()
    if int(subtokens.ndim) != 4:
        raise ValueError(f"subtokens must have shape [batch, particles, modalities, embed_dim], got {tuple(subtokens.shape)}")
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape [batch, particles], got {tuple(mask.shape)}")
    if tuple(subtokens.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"subtokens/mask leading shapes differ: {tuple(subtokens.shape[:2])} vs {tuple(mask.shape)}")
    if int(subtokens.shape[-1]) != int(embed_dim):
        raise ValueError(f"subtokens last dimension must be embed_dim={int(embed_dim)}, got {int(subtokens.shape[-1])}")
    if int(subtokens.shape[2]) <= 0:
        raise ValueError("subtokens must contain at least one modality")
    return subtokens, mask


def _validate_modality_mask(modality_mask: Any, subtokens: Any) -> Any:
    modality_mask = modality_mask.bool()
    expected = tuple(subtokens.shape[:3])
    if tuple(modality_mask.shape) != expected:
        raise ValueError(f"modality_mask shape {tuple(modality_mask.shape)} does not match {expected}")
    return modality_mask


class _LocalCompressionTransformerLayer(_ModuleBase):
    """Pre-norm modality mixer with separate attention and residual dropout."""

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
        tokens = tokens + self.residual_dropout(ffn)
        return tokens


@dataclass(frozen=True)
class LocalCompressionCompressorOutput:
    """Locally compressed modality tokens for every particle."""

    local_tokens: Any
    mask: Any
    modality_mask: Any
    modality_names: tuple[str, ...]
    input_subtokens: Any

    def __post_init__(self) -> None:
        torch = require_torch()
        if int(self.local_tokens.ndim) != 4:
            raise ValueError("local_tokens must have shape [batch, particles, modalities, embed_dim]")
        batch_size, num_particles, num_modalities, embed_dim = tuple(self.local_tokens.shape)
        expected_mask = (batch_size, num_particles)
        expected_modality_mask = (batch_size, num_particles, num_modalities)
        if tuple(self.mask.shape) != expected_mask:
            raise ValueError(f"mask has shape {tuple(self.mask.shape)}, expected {expected_mask}")
        if tuple(self.modality_mask.shape) != expected_modality_mask:
            raise ValueError(f"modality_mask has shape {tuple(self.modality_mask.shape)}, expected {expected_modality_mask}")
        if tuple(self.input_subtokens.shape) != (batch_size, num_particles, num_modalities, embed_dim):
            raise ValueError("input_subtokens shape must match local_tokens")
        if len(tuple(self.modality_names)) != num_modalities:
            raise ValueError("modality_names length must match local token modality dimension")
        if not bool(torch.isfinite(self.local_tokens).all()):
            raise ValueError("local_tokens contain non-finite values")
        object.__setattr__(self, "modality_names", tuple(self.modality_names))

    @property
    def batch_size(self) -> int:
        return int(self.local_tokens.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.local_tokens.shape[1])

    @property
    def num_modalities(self) -> int:
        return int(self.local_tokens.shape[2])

    @property
    def embed_dim(self) -> int:
        return int(self.local_tokens.shape[3])

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_COMPRESSION_COMPRESSOR_CONTRACT,
            "local_tokens_shape": list(self.local_tokens.shape),
            "input_subtokens_shape": list(self.input_subtokens.shape),
            "mask_shape": list(self.mask.shape),
            "modality_mask_shape": list(self.modality_mask.shape),
            "modality_names": list(self.modality_names),
            "active_modality_count": int(self.modality_mask.detach().cpu().sum().item()),
        }


class LocalModalityCompressor(_ModuleBase):
    """Tiny Transformer over modalities inside each particle."""

    def __init__(self, config: LocalCompressionPartConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.num_layers = int(self.config.local_layers)
        self.num_heads = int(self.config.local_heads)
        if self.num_layers <= 0:
            raise ValueError("local_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("local_heads must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by local_heads")
        ff_dim = max(self.embed_dim, int(round(float(self.config.mlp_ratio) * self.embed_dim)))
        self.layers = torch.nn.ModuleList(
            [
                _LocalCompressionTransformerLayer(
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

    def forward(
        self,
        subtokens_or_output: Any,
        mask: Any | None = None,
        modality_mask: Any | None = None,
    ) -> LocalCompressionCompressorOutput:
        torch = require_torch()
        if isinstance(subtokens_or_output, LocalCompressionSubtokenOutput):
            subtokens = subtokens_or_output.subtokens
            mask = subtokens_or_output.mask
            modality_mask = subtokens_or_output.modality_mask
            modality_names = subtokens_or_output.modality_names
        else:
            if mask is None:
                raise ValueError("mask is required when passing raw subtoken tensors")
            subtokens = subtokens_or_output
            modality_names = tuple(f"modality_{index}" for index in range(int(subtokens.shape[2])))

        subtokens, mask = _validate_subtokens(subtokens, mask, embed_dim=self.embed_dim)
        batch_size, num_particles, num_modalities, embed_dim = tuple(subtokens.shape)
        if modality_mask is None:
            modality_mask = mask[:, :, None].expand(batch_size, num_particles, num_modalities)
        modality_mask = _validate_modality_mask(modality_mask, subtokens)
        modality_mask = modality_mask & mask[:, :, None]
        input_subtokens = torch.where(modality_mask[:, :, :, None], subtokens, torch.zeros_like(subtokens))

        flat = input_subtokens.reshape(batch_size * num_particles, num_modalities, embed_dim)
        flat_modality_mask = modality_mask.reshape(batch_size * num_particles, num_modalities)

        # PyTorch attention returns NaNs for rows where every key is padded.  Give
        # those dummy rows one visible zero token, then zero them again after the
        # transformer.
        safe_flat_modality_mask = flat_modality_mask.clone()
        all_masked = ~safe_flat_modality_mask.any(dim=1)
        if bool(all_masked.any()):
            safe_flat_modality_mask[all_masked, 0] = True
        key_padding_mask = ~safe_flat_modality_mask
        mixed = flat
        for layer in self.layers:
            mixed = layer(mixed, key_padding_mask=key_padding_mask)
        mixed = self.output_norm(_nan_to_num_torch(mixed))
        local_tokens = mixed.reshape(batch_size, num_particles, num_modalities, embed_dim)
        local_tokens = torch.where(modality_mask[:, :, :, None], local_tokens, torch.zeros_like(local_tokens))
        return LocalCompressionCompressorOutput(
            local_tokens=local_tokens,
            mask=mask,
            modality_mask=modality_mask,
            modality_names=tuple(modality_names),
            input_subtokens=input_subtokens,
        )


__all__ = [
    "LOCAL_COMPRESSION_COMPRESSOR_CONTRACT",
    "LOCAL_COMPRESSION_COMPRESSOR_STEP",
    "LocalCompressionCompressorOutput",
    "LocalModalityCompressor",
]

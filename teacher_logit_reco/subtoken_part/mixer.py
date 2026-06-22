"""Within-particle subtoken mixing for reliability-gated subtoken ParT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import SubtokenPartConfig
from .encoders import SubtokenEncoderOutput

try:  # Keep imports lightweight on systems without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


SUBTOKEN_PART_LOCAL_MIXER_STEP = "subtoken_part_step4_within_particle_mixer"
SUBTOKEN_PART_LOCAL_MIXER_CONTRACT = "local_tokens_mask_modalities_v1"


@dataclass(frozen=True)
class SubtokenMixerOutput:
    """Locally mixed modality subtokens for every valid particle."""

    local_tokens: Any
    mask: Any
    modality_mask: Any
    modality_names: tuple[str, ...]
    input_subtokens: Any

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_LOCAL_MIXER_CONTRACT,
            "local_tokens_shape": list(self.local_tokens.shape),
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


def _validate_subtoken_inputs(subtokens: Any, mask: Any, *, embed_dim: int) -> tuple[Any, Any]:
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
    expected_shape = tuple(subtokens.shape[:3])
    if tuple(modality_mask.shape) != expected_shape:
        raise ValueError(f"modality_mask shape {tuple(modality_mask.shape)} does not match {expected_shape}")
    return modality_mask


class WithinParticleSubtokenTransformer(_ModuleBase):
    """Tiny self-attention block over modalities inside each particle."""

    def __init__(self, config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> None:
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

    def forward(
        self,
        subtokens_or_output: Any,
        mask: Any | None = None,
        modality_mask: Any | None = None,
    ) -> SubtokenMixerOutput:
        torch = require_torch()
        if isinstance(subtokens_or_output, SubtokenEncoderOutput):
            subtokens = subtokens_or_output.subtokens
            mask = subtokens_or_output.mask
            modality_mask = subtokens_or_output.modality_mask
            modality_names = subtokens_or_output.modality_names
        else:
            if mask is None:
                raise ValueError("mask is required when passing raw subtoken tensors")
            subtokens = subtokens_or_output
            modality_names = None

        subtokens, mask = _validate_subtoken_inputs(subtokens, mask, embed_dim=self.embed_dim)
        if modality_names is None:
            modality_names = tuple(f"modality_{index}" for index in range(int(subtokens.shape[2])))
        batch_size, num_particles, num_modalities, embed_dim = subtokens.shape
        if modality_mask is None:
            modality_mask = mask[:, :, None].expand(batch_size, num_particles, num_modalities)
        modality_mask = _validate_modality_mask(modality_mask, subtokens)
        modality_mask = modality_mask & mask[:, :, None]
        masked_subtokens = torch.where(modality_mask[:, :, :, None], subtokens, torch.zeros_like(subtokens))

        flat = masked_subtokens.reshape(batch_size * num_particles, num_modalities, embed_dim)
        flat_modality_mask = modality_mask.reshape(batch_size * num_particles, num_modalities)
        safe_flat_modality_mask = flat_modality_mask.clone()
        all_masked = ~safe_flat_modality_mask.any(dim=1)
        safe_flat_modality_mask[all_masked, 0] = True
        mixed = self.encoder(flat, src_key_padding_mask=~safe_flat_modality_mask)
        mixed = self.output_norm(mixed)
        local_tokens = mixed.reshape(batch_size, num_particles, num_modalities, embed_dim)
        local_tokens = torch.where(modality_mask[:, :, :, None], local_tokens, torch.zeros_like(local_tokens))

        return SubtokenMixerOutput(
            local_tokens=local_tokens,
            mask=mask,
            modality_mask=modality_mask,
            modality_names=tuple(modality_names),
            input_subtokens=masked_subtokens,
        )


__all__ = [
    "SUBTOKEN_PART_LOCAL_MIXER_CONTRACT",
    "SUBTOKEN_PART_LOCAL_MIXER_STEP",
    "SubtokenMixerOutput",
    "WithinParticleSubtokenTransformer",
]

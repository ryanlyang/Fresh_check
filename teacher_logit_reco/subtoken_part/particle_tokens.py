"""Build reliability-aware particle tokens from modality subtokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import SUBTOKEN_PART_GATE_NONE, SubtokenPartConfig
from .encoders import SubtokenEncoderOutput
from .gates import ReliabilityGateOutput
from .mixer import SubtokenMixerOutput
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


SUBTOKEN_PART_PARTICLE_TOKEN_STEP = "subtoken_part_step8_particle_tokens"
SUBTOKEN_PART_PARTICLE_TOKEN_CONTRACT = "reliability_aware_particle_tokens_v1"


@dataclass(frozen=True)
class ReliabilityAwareParticleOutput:
    """Particle tokens ready for the global particle-level transformer."""

    particle_tokens: Any
    weighted_modalities: Any
    anchor: Any
    gates: Any
    mask: Any
    modality_mask: Any
    modality_names: tuple[str, ...]
    gate_mode: str
    used_reliability_gates: bool

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_PARTICLE_TOKEN_CONTRACT,
            "gate_mode": self.gate_mode,
            "used_reliability_gates": bool(self.used_reliability_gates),
            "particle_tokens_shape": list(self.particle_tokens.shape),
            "weighted_modalities_shape": list(self.weighted_modalities.shape),
            "anchor_shape": list(self.anchor.shape),
            "gates_shape": list(self.gates.shape),
            "mask_shape": list(self.mask.shape),
            "modality_mask_shape": list(self.modality_mask.shape),
            "modality_names": list(self.modality_names),
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        valid_count = torch.clamp(self.mask.sum().to(dtype=self.particle_tokens.dtype), min=1.0)
        particle_norm = torch.linalg.vector_norm(self.particle_tokens, dim=-1)
        weighted_norm = torch.linalg.vector_norm(self.weighted_modalities, dim=-1)
        anchor_norm = torch.linalg.vector_norm(self.anchor, dim=-1)
        return {
            "mean_particle_token_norm": particle_norm.sum() / valid_count,
            "mean_weighted_modality_norm": weighted_norm.sum() / valid_count,
            "mean_anchor_norm": anchor_norm.sum() / valid_count,
            "mean_gate_by_particle": self.gates.sum(dim=(0, 1)) / valid_count,
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


def _zero_invalid_particles(values: Any, mask: Any) -> Any:
    torch = require_torch()
    return torch.where(mask[:, :, None], values, torch.zeros_like(values))


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
    return local_tokens, mask


def _validate_particle_tensor(name: str, values: Any, mask: Any, *, embed_dim: int) -> Any:
    values = _nan_to_num_torch(values.float())
    if int(values.ndim) != 3:
        raise ValueError(f"{name} must have shape [batch, particles, embed_dim], got {tuple(values.shape)}")
    if tuple(values.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"{name}/mask leading shapes differ: {tuple(values.shape[:2])} vs {tuple(mask.shape)}")
    if int(values.shape[-1]) != int(embed_dim):
        raise ValueError(f"{name} last dimension must be embed_dim={int(embed_dim)}, got {int(values.shape[-1])}")
    return values


def _validate_modality_values(name: str, values: Any, local_tokens: Any) -> Any:
    values = _nan_to_num_torch(values.float())
    expected_shape = tuple(local_tokens.shape[:3])
    if tuple(values.shape) != expected_shape:
        raise ValueError(f"{name} shape {tuple(values.shape)} does not match {expected_shape}")
    return values


def _validate_modality_mask(modality_mask: Any, local_tokens: Any) -> Any:
    modality_mask = modality_mask.bool()
    expected_shape = tuple(local_tokens.shape[:3])
    if tuple(modality_mask.shape) != expected_shape:
        raise ValueError(f"modality_mask shape {tuple(modality_mask.shape)} does not match {expected_shape}")
    return modality_mask


class ReliabilityAwareParticleTokenBuilder(_ModuleBase):
    """Combine anchor and modality evidence into one token per particle."""

    def __init__(self, config: SubtokenPartConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.gate_mode = self.config.gate_mode

    def _extract_local_inputs(
        self,
        local_tokens_or_output: Any,
        mask: Any | None,
        modality_mask: Any | None,
    ) -> tuple[Any, Any, Any, tuple[str, ...]]:
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
        batch_size, num_particles, num_modalities, _ = local_tokens.shape
        if modality_mask is None:
            modality_mask = mask[:, :, None].expand(batch_size, num_particles, num_modalities)
        modality_mask = _validate_modality_mask(modality_mask, local_tokens)
        modality_mask = modality_mask & mask[:, :, None]
        local_tokens = require_torch().where(modality_mask[:, :, :, None], local_tokens, require_torch().zeros_like(local_tokens))
        if modality_names is None:
            modality_names = tuple(f"modality_{index}" for index in range(num_modalities))
        return local_tokens, mask, modality_mask, tuple(modality_names)

    def forward(
        self,
        local_tokens_or_output: Any,
        pooled_or_output: Any | None = None,
        gate_or_output: ReliabilityGateOutput | None = None,
        anchor_or_output: Any | None = None,
        *,
        mask: Any | None = None,
        modality_mask: Any | None = None,
    ) -> ReliabilityAwareParticleOutput:
        torch = require_torch()
        local_tokens, mask, modality_mask, modality_names = self._extract_local_inputs(
            local_tokens_or_output,
            mask,
            modality_mask,
        )

        if isinstance(pooled_or_output, SubtokenPoolOutput):
            provisional_particles = pooled_or_output.provisional_particles
            pool_weights = pooled_or_output.pool_weights
        else:
            provisional_particles = pooled_or_output
            pool_weights = None
        if provisional_particles is None:
            raise ValueError("pooled_or_output is required")
        provisional_particles = _validate_particle_tensor(
            "provisional_particles",
            provisional_particles,
            mask,
            embed_dim=self.embed_dim,
        )

        if isinstance(anchor_or_output, SubtokenEncoderOutput):
            anchor = anchor_or_output.anchor
        else:
            anchor = anchor_or_output
        if anchor is None:
            anchor = torch.zeros_like(provisional_particles)
        anchor = _validate_particle_tensor("anchor", anchor, mask, embed_dim=self.embed_dim)

        if gate_or_output is None and self.gate_mode != SUBTOKEN_PART_GATE_NONE:
            raise ValueError(f"gate_mode={self.gate_mode!r} requires gate_or_output")
        use_gates = gate_or_output is not None and gate_or_output.gate_mode != SUBTOKEN_PART_GATE_NONE
        if not use_gates:
            if pool_weights is None:
                active_counts = modality_mask.sum(dim=2, keepdim=True).clamp(min=1).to(dtype=local_tokens.dtype)
                gates = modality_mask.to(dtype=local_tokens.dtype) / active_counts
            else:
                gates = _validate_modality_values("pool_weights", pool_weights, local_tokens)
                gates = torch.where(modality_mask, gates, torch.zeros_like(gates))
            weighted_modalities = provisional_particles
            particle_tokens = provisional_particles
            gate_mode = SUBTOKEN_PART_GATE_NONE
        else:
            gates = _validate_modality_values("gates", gate_or_output.gates, local_tokens)
            gate_mask = _validate_modality_mask(gate_or_output.modality_mask, local_tokens)
            if tuple(gate_or_output.mask.shape) != tuple(mask.shape):
                raise ValueError(f"gate mask shape {tuple(gate_or_output.mask.shape)} does not match {tuple(mask.shape)}")
            modality_mask = modality_mask & gate_mask & gate_or_output.mask.bool()[:, :, None]
            gates = torch.where(modality_mask, gates, torch.zeros_like(gates))
            weighted_modalities = (local_tokens * gates[:, :, :, None]).sum(dim=2)
            particle_tokens = anchor + weighted_modalities
            gate_mode = gate_or_output.gate_mode

        weighted_modalities = _zero_invalid_particles(_nan_to_num_torch(weighted_modalities), mask)
        particle_tokens = _zero_invalid_particles(_nan_to_num_torch(particle_tokens), mask)
        anchor = _zero_invalid_particles(_nan_to_num_torch(anchor), mask)
        gates = torch.where(modality_mask, _nan_to_num_torch(gates), torch.zeros_like(gates))
        gates = torch.where(mask[:, :, None], gates, torch.zeros_like(gates))

        return ReliabilityAwareParticleOutput(
            particle_tokens=particle_tokens,
            weighted_modalities=weighted_modalities,
            anchor=anchor,
            gates=gates,
            mask=mask,
            modality_mask=modality_mask,
            modality_names=modality_names,
            gate_mode=gate_mode,
            used_reliability_gates=bool(use_gates),
        )


__all__ = [
    "SUBTOKEN_PART_PARTICLE_TOKEN_CONTRACT",
    "SUBTOKEN_PART_PARTICLE_TOKEN_STEP",
    "ReliabilityAwareParticleOutput",
    "ReliabilityAwareParticleTokenBuilder",
]

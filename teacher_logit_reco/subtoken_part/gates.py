"""Context-aware reliability gates for subtoken particle modalities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import (
    SUBTOKEN_PART_GATE_CONTEXT_SIGMOID,
    SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
    SUBTOKEN_PART_GATE_LOCAL_SOFTMAX,
    SUBTOKEN_PART_GATE_NONE,
    SubtokenPartConfig,
)
from .context import ParticleContextOutput
from .encoders import SubtokenEncoderOutput
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


SUBTOKEN_PART_GATE_STEP = "subtoken_part_step7_reliability_gates"
SUBTOKEN_PART_GATE_CONTRACT = "context_aware_modality_gates_v1"


@dataclass(frozen=True)
class ReliabilityGateOutput:
    """Reliability weights for each modality token of each particle."""

    gate_logits: Any
    gates: Any
    gate_entropy: Any
    mask: Any
    modality_mask: Any
    modality_names: tuple[str, ...]
    gate_mode: str

    def summary(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_GATE_CONTRACT,
            "gate_mode": self.gate_mode,
            "gate_logits_shape": list(self.gate_logits.shape),
            "gates_shape": list(self.gates.shape),
            "gate_entropy_shape": list(self.gate_entropy.shape),
            "mask_shape": list(self.mask.shape),
            "modality_mask_shape": list(self.modality_mask.shape),
            "modality_names": list(self.modality_names),
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        valid_count = torch.clamp(self.mask.sum().to(dtype=self.gates.dtype), min=1.0)
        active_count_by_modality = torch.clamp(
            self.modality_mask.sum(dim=(0, 1)).to(dtype=self.gates.dtype),
            min=1.0,
        )
        mean_gate_by_particle = self.gates.sum(dim=(0, 1)) / valid_count
        mean_gate_when_active = self.gates.sum(dim=(0, 1)) / active_count_by_modality
        mean_entropy = self.gate_entropy.sum() / valid_count
        return {
            "mean_gate_by_particle": mean_gate_by_particle,
            "mean_gate_when_active": mean_gate_when_active,
            "mean_gate_entropy": mean_entropy,
            "active_modality_fraction": self.modality_mask.to(dtype=self.gates.dtype).sum(dim=(0, 1)) / valid_count,
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


def _make_mlp(input_dim: int, hidden_dim: int, *, dropout: float = 0.0) -> Any:
    torch = require_torch()
    input_dim = int(input_dim)
    hidden_dim = int(hidden_dim)
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_dim),
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(hidden_dim, 1),
    )


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


def _validate_modality_mask(modality_mask: Any, local_tokens: Any) -> Any:
    modality_mask = modality_mask.bool()
    expected_shape = tuple(local_tokens.shape[:3])
    if tuple(modality_mask.shape) != expected_shape:
        raise ValueError(f"modality_mask shape {tuple(modality_mask.shape)} does not match {expected_shape}")
    return modality_mask


def _ensure_one_active_modality(modality_mask: Any, particle_mask: Any) -> Any:
    torch = require_torch()
    missing_valid_particle = particle_mask & ~modality_mask.any(dim=2)
    fallback = torch.nn.functional.one_hot(
        torch.zeros_like(particle_mask, dtype=torch.long),
        num_classes=int(modality_mask.shape[2]),
    ).bool()
    return (modality_mask | (missing_valid_particle[:, :, None] & fallback)) & particle_mask[:, :, None]


def _masked_softmax(logits: Any, modality_mask: Any, particle_mask: Any) -> Any:
    torch = require_torch()
    masked_logits = logits.masked_fill(~modality_mask, torch.finfo(logits.dtype).min)
    gates = torch.softmax(masked_logits, dim=-1)
    gates = torch.where(modality_mask, gates, torch.zeros_like(gates))
    gates = gates / torch.clamp(gates.sum(dim=2, keepdim=True), min=float(1.0e-12))
    return torch.where(particle_mask[:, :, None], gates, torch.zeros_like(gates))


def _gate_entropy(gates: Any, particle_mask: Any) -> Any:
    torch = require_torch()
    probs = gates / torch.clamp(gates.sum(dim=2, keepdim=True), min=float(1.0e-12))
    log_probs = torch.where(probs > 0.0, torch.log(torch.clamp(probs, min=float(1.0e-12))), torch.zeros_like(probs))
    entropy = -(probs * log_probs).sum(dim=2)
    return torch.where(particle_mask, _nan_to_num_torch(entropy), torch.zeros_like(entropy))


class ReliabilityGateHead(_ModuleBase):
    """Predict context-aware reliability weights over particle modality tokens."""

    def __init__(
        self,
        config: SubtokenPartConfig | Mapping[str, Any] | None = None,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.gate_mode = self.config.gate_mode
        self.hidden_dim = int(hidden_dim or 2 * self.embed_dim)
        self.logit_head = _make_mlp(
            4 * self.embed_dim,
            self.hidden_dim,
            dropout=float(self.config.dropout),
        )

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
        modality_mask = _ensure_one_active_modality(modality_mask & mask[:, :, None], mask)
        if modality_names is None:
            modality_names = tuple(f"modality_{index}" for index in range(num_modalities))
        return local_tokens, mask, modality_mask, tuple(modality_names)

    def forward(
        self,
        local_tokens_or_output: Any,
        provisional_particles_or_output: Any | None = None,
        particle_context_or_output: Any | None = None,
        anchor_or_output: Any | None = None,
        *,
        mask: Any | None = None,
        modality_mask: Any | None = None,
    ) -> ReliabilityGateOutput:
        torch = require_torch()
        local_tokens, mask, modality_mask, modality_names = self._extract_local_inputs(
            local_tokens_or_output,
            mask,
            modality_mask,
        )
        batch_size, num_particles, num_modalities, embed_dim = local_tokens.shape

        if isinstance(provisional_particles_or_output, SubtokenPoolOutput):
            provisional_particles = provisional_particles_or_output.provisional_particles
        else:
            provisional_particles = provisional_particles_or_output
        if provisional_particles is None:
            raise ValueError("provisional_particles_or_output is required")
        provisional_particles = _validate_particle_tensor(
            "provisional_particles",
            provisional_particles,
            mask,
            embed_dim=embed_dim,
        )

        if isinstance(anchor_or_output, SubtokenEncoderOutput):
            anchor = anchor_or_output.anchor
        else:
            anchor = anchor_or_output
        if anchor is None:
            raise ValueError("anchor_or_output is required")
        anchor = _validate_particle_tensor("anchor", anchor, mask, embed_dim=embed_dim)

        if isinstance(particle_context_or_output, ParticleContextOutput):
            particle_context = particle_context_or_output.context_tokens
        else:
            particle_context = particle_context_or_output
        if particle_context is None:
            if self.gate_mode in {SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX, SUBTOKEN_PART_GATE_CONTEXT_SIGMOID}:
                raise ValueError(f"gate_mode={self.gate_mode!r} requires particle_context_or_output")
            particle_context = torch.zeros_like(provisional_particles)
        particle_context = _validate_particle_tensor("particle_context", particle_context, mask, embed_dim=embed_dim)

        if self.gate_mode == SUBTOKEN_PART_GATE_NONE:
            gate_logits = local_tokens.new_zeros((batch_size, num_particles, num_modalities))
            gates = _masked_softmax(gate_logits, modality_mask, mask)
        else:
            if self.gate_mode == SUBTOKEN_PART_GATE_LOCAL_SOFTMAX:
                particle_context = torch.zeros_like(particle_context)
            particle_context_expanded = particle_context[:, :, None, :].expand_as(local_tokens)
            provisional_expanded = provisional_particles[:, :, None, :].expand_as(local_tokens)
            anchor_expanded = anchor[:, :, None, :].expand_as(local_tokens)
            gate_features = torch.cat(
                [local_tokens, provisional_expanded, particle_context_expanded, anchor_expanded],
                dim=-1,
            )
            raw_logits = self.logit_head(gate_features).squeeze(-1)
            raw_logits = _nan_to_num_torch(raw_logits)
            if self.gate_mode in {SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX, SUBTOKEN_PART_GATE_LOCAL_SOFTMAX}:
                gates = _masked_softmax(raw_logits, modality_mask, mask)
            elif self.gate_mode == SUBTOKEN_PART_GATE_CONTEXT_SIGMOID:
                gates = torch.sigmoid(raw_logits)
                gates = torch.where(modality_mask, gates, torch.zeros_like(gates))
                gates = torch.where(mask[:, :, None], gates, torch.zeros_like(gates))
            else:  # pragma: no cover - guarded by config normalization
                raise AssertionError(f"Unhandled gate mode {self.gate_mode!r}")
            gate_logits = torch.where(modality_mask & mask[:, :, None], raw_logits, torch.zeros_like(raw_logits))

        gate_entropy = _gate_entropy(gates, mask)
        gates = torch.where(modality_mask & mask[:, :, None], _nan_to_num_torch(gates), torch.zeros_like(gates))
        gate_logits = torch.where(mask[:, :, None], _nan_to_num_torch(gate_logits), torch.zeros_like(gate_logits))

        return ReliabilityGateOutput(
            gate_logits=gate_logits,
            gates=gates,
            gate_entropy=gate_entropy,
            mask=mask,
            modality_mask=modality_mask,
            modality_names=modality_names,
            gate_mode=self.gate_mode,
        )


__all__ = [
    "SUBTOKEN_PART_GATE_CONTRACT",
    "SUBTOKEN_PART_GATE_STEP",
    "ReliabilityGateHead",
    "ReliabilityGateOutput",
]

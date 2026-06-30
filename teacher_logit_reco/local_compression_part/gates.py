"""Context-aware modality gates for local-compression adapters.

Step 8 predicts one sigmoid gate per particle modality.  The gates are used by
later delta-F adapters as reliability/importance signals; they are not the main
classifier and they never see offline targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .compressor import LocalCompressionCompressorOutput
from .config import (
    LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID,
    LOCAL_COMPRESSION_GATE_NONE,
    LOCAL_COMPRESSION_PART_CONTRACT,
    LocalCompressionPartConfig,
    normalize_local_compression_gate_mode,
)
from .context import ParticleContextOutput
from .features import LocalCompressionCanonicalInputs

try:  # Keep package imports cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_COMPRESSION_GATES_STEP = "local_compression_part_step8_context_gates"
LOCAL_COMPRESSION_GATES_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_context_modality_gates_v1"


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


def _make_mlp(input_dim: int, output_dim: int, *, hidden_dim: int, dropout: float) -> Any:
    torch = require_torch()
    return torch.nn.Sequential(
        torch.nn.LayerNorm(int(input_dim)),
        torch.nn.Linear(int(input_dim), int(hidden_dim)),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(int(hidden_dim), int(output_dim)),
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


def _validate_context_tokens(context_tokens: Any, mask: Any, *, embed_dim: int) -> Any:
    context_tokens = _nan_to_num_torch(context_tokens.float())
    if int(context_tokens.ndim) != 3:
        raise ValueError(f"context_tokens must have shape [batch, particles, embed_dim], got {tuple(context_tokens.shape)}")
    if tuple(context_tokens.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"context_tokens/mask leading shapes differ: {tuple(context_tokens.shape[:2])} vs {tuple(mask.shape)}")
    if int(context_tokens.shape[-1]) != int(embed_dim):
        raise ValueError(f"context_tokens last dimension must be embed_dim={int(embed_dim)}, got {int(context_tokens.shape[-1])}")
    return context_tokens


def _validate_feature_rows(feature_rows: Any, mask: Any, *, feature_dim: int) -> Any:
    feature_rows = _nan_to_num_torch(feature_rows.float())
    if int(feature_rows.ndim) != 3:
        raise ValueError(f"feature_rows must have shape [batch, particles, feature_dim], got {tuple(feature_rows.shape)}")
    if tuple(feature_rows.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"feature_rows/mask leading shapes differ: {tuple(feature_rows.shape[:2])} vs {tuple(mask.shape)}")
    if int(feature_rows.shape[-1]) != int(feature_dim):
        raise ValueError(f"feature_rows last dimension must be feature_dim={int(feature_dim)}, got {int(feature_rows.shape[-1])}")
    return feature_rows


def _validate_modality_mask(modality_mask: Any, local_tokens: Any) -> Any:
    modality_mask = modality_mask.bool()
    expected = tuple(local_tokens.shape[:3])
    if tuple(modality_mask.shape) != expected:
        raise ValueError(f"modality_mask shape {tuple(modality_mask.shape)} does not match {expected}")
    return modality_mask


@dataclass(frozen=True)
class LocalCompressionGateOutput:
    """Sigmoid modality gates and gate diagnostics."""

    gates: Any
    gate_logits: Any
    diagnostic_weights: Any
    mask: Any
    modality_mask: Any
    modality_names: tuple[str, ...]
    local_tokens: Any
    context_tokens: Any
    feature_rows: Any
    feature_summary: Any
    gate_mode: str

    def __post_init__(self) -> None:
        torch = require_torch()
        if int(self.gates.ndim) != 3:
            raise ValueError("gates must have shape [batch, particles, modalities]")
        batch_size, num_particles, num_modalities = tuple(self.gates.shape)
        if tuple(self.gate_logits.shape) != (batch_size, num_particles, num_modalities):
            raise ValueError("gate_logits shape must match gates")
        if tuple(self.diagnostic_weights.shape) != (batch_size, num_particles, num_modalities):
            raise ValueError("diagnostic_weights shape must match gates")
        if tuple(self.mask.shape) != (batch_size, num_particles):
            raise ValueError("mask shape must match gates leading dimensions")
        if tuple(self.modality_mask.shape) != (batch_size, num_particles, num_modalities):
            raise ValueError("modality_mask shape must match gates")
        if int(self.local_tokens.ndim) != 4 or tuple(self.local_tokens.shape[:3]) != (batch_size, num_particles, num_modalities):
            raise ValueError("local_tokens shape must be [batch, particles, modalities, embed_dim]")
        embed_dim = int(self.local_tokens.shape[-1])
        if tuple(self.context_tokens.shape) != (batch_size, num_particles, embed_dim):
            raise ValueError("context_tokens shape must be [batch, particles, embed_dim]")
        if tuple(self.feature_summary.shape) != (batch_size, num_particles, embed_dim):
            raise ValueError("feature_summary shape must be [batch, particles, embed_dim]")
        if tuple(self.feature_rows.shape[:2]) != (batch_size, num_particles):
            raise ValueError("feature_rows leading dimensions must match gates")
        if len(tuple(self.modality_names)) != num_modalities:
            raise ValueError("modality_names length must match gates modality dimension")
        for name, value in {
            "gates": self.gates,
            "gate_logits": self.gate_logits,
            "diagnostic_weights": self.diagnostic_weights,
            "local_tokens": self.local_tokens,
            "context_tokens": self.context_tokens,
            "feature_rows": self.feature_rows,
            "feature_summary": self.feature_summary,
        }.items():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        if float(self.gates.min().detach().cpu().item()) < -1.0e-6 or float(self.gates.max().detach().cpu().item()) > 1.0 + 1.0e-6:
            raise ValueError("gates must stay in [0, 1]")
        inactive_gates = torch.where(self.modality_mask, torch.zeros_like(self.gates), self.gates.abs())
        inactive_diag = torch.where(self.modality_mask, torch.zeros_like(self.diagnostic_weights), self.diagnostic_weights.abs())
        if float(inactive_gates.max().detach().cpu().item()) > 1.0e-6:
            raise ValueError("inactive modalities must have zero gates")
        if float(inactive_diag.max().detach().cpu().item()) > 1.0e-6:
            raise ValueError("inactive modalities must have zero diagnostic weight")
        invalid_gates = torch.where(self.mask[:, :, None], torch.zeros_like(self.gates), self.gates.abs())
        if float(invalid_gates.max().detach().cpu().item()) > 1.0e-6:
            raise ValueError("invalid particles must have zero gates")
        object.__setattr__(self, "modality_names", tuple(self.modality_names))
        object.__setattr__(self, "gate_mode", normalize_local_compression_gate_mode(self.gate_mode))

    @property
    def batch_size(self) -> int:
        return int(self.gates.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.gates.shape[1])

    @property
    def num_modalities(self) -> int:
        return int(self.gates.shape[2])

    def summary(self) -> dict[str, Any]:
        torch = require_torch()
        active = self.modality_mask.bool()
        if bool(active.any()):
            active_gates = self.gates[active]
            gate_mean = float(active_gates.detach().cpu().mean().item())
            gate_min = float(active_gates.detach().cpu().min().item())
            gate_max = float(active_gates.detach().cpu().max().item())
        else:
            gate_mean = gate_min = gate_max = 0.0
        return {
            "contract": LOCAL_COMPRESSION_GATES_CONTRACT,
            "gate_mode": self.gate_mode,
            "gates_shape": list(self.gates.shape),
            "gate_logits_shape": list(self.gate_logits.shape),
            "diagnostic_weights_shape": list(self.diagnostic_weights.shape),
            "mask_shape": list(self.mask.shape),
            "modality_mask_shape": list(self.modality_mask.shape),
            "modality_names": list(self.modality_names),
            "active_particle_count": int(self.modality_mask.any(dim=-1).detach().cpu().sum().item()),
            "active_modality_count": int(active.detach().cpu().sum().item()),
            "active_gate_mean": gate_mean,
            "active_gate_min": gate_min,
            "active_gate_max": gate_max,
        }


class LocalCompressionContextGate(_ModuleBase):
    """Predict context-aware sigmoid gates for each particle modality."""

    def __init__(self, config: LocalCompressionPartConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.feature_dim = len(tuple(self.config.feature_config.canonical_feature_names))
        self.gate_mode = normalize_local_compression_gate_mode(self.config.gate_mode)
        hidden_dim = max(self.embed_dim, int(round(float(self.config.mlp_ratio) * self.embed_dim)))
        self.feature_summary = _make_mlp(
            self.feature_dim,
            self.embed_dim,
            hidden_dim=hidden_dim,
            dropout=float(self.config.dropout),
        )
        if self.gate_mode == LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID:
            self.gate_mlp = _make_mlp(
                3 * self.embed_dim,
                1,
                hidden_dim=hidden_dim,
                dropout=float(self.config.dropout),
            )
            final_linear = self.gate_mlp[-1]
            torch.nn.init.normal_(final_linear.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(final_linear.bias)
        elif self.gate_mode == LOCAL_COMPRESSION_GATE_NONE:
            self.gate_mlp = None
        else:  # pragma: no cover - normalized config should prevent this.
            raise ValueError(f"unsupported gate_mode {self.gate_mode!r}")

    def _feature_rows_from_input(self, feature_rows_or_canonical: Any, mask: Any) -> Any:
        if isinstance(feature_rows_or_canonical, LocalCompressionCanonicalInputs):
            if not bool((feature_rows_or_canonical.particle_mask.bool() == mask.bool()).all()):
                raise ValueError("canonical particle_mask does not match gate mask")
            feature_rows = feature_rows_or_canonical.feature_rows()
        elif hasattr(feature_rows_or_canonical, "feature_rows"):
            feature_rows = feature_rows_or_canonical.feature_rows()
            particle_mask = getattr(feature_rows_or_canonical, "particle_mask", None)
            if particle_mask is not None and not bool((particle_mask.bool() == mask.bool()).all()):
                raise ValueError("feature_rows source particle_mask does not match gate mask")
        else:
            feature_rows = feature_rows_or_canonical
        return _validate_feature_rows(feature_rows, mask, feature_dim=self.feature_dim)

    def forward(
        self,
        local_or_output: Any,
        context_or_output: Any,
        feature_rows_or_canonical: Any,
        *,
        mask: Any | None = None,
        modality_mask: Any | None = None,
    ) -> LocalCompressionGateOutput:
        torch = require_torch()
        if isinstance(local_or_output, LocalCompressionCompressorOutput):
            local_tokens = local_or_output.local_tokens
            mask = local_or_output.mask
            modality_mask = local_or_output.modality_mask
            modality_names = local_or_output.modality_names
        else:
            if mask is None:
                raise ValueError("mask is required when passing raw local token tensors")
            local_tokens = local_or_output
            modality_names = tuple(f"modality_{index}" for index in range(int(local_tokens.shape[2])))

        local_tokens, mask = _validate_local_tokens(local_tokens, mask, embed_dim=self.embed_dim)
        batch_size, num_particles, num_modalities, _embed_dim = tuple(local_tokens.shape)
        if modality_mask is None:
            modality_mask = mask[:, :, None].expand(batch_size, num_particles, num_modalities)
        modality_mask = _validate_modality_mask(modality_mask, local_tokens)
        modality_mask = modality_mask & mask[:, :, None]
        local_tokens = torch.where(modality_mask[:, :, :, None], local_tokens, torch.zeros_like(local_tokens))

        if isinstance(context_or_output, ParticleContextOutput):
            context_tokens = context_or_output.context_tokens
            context_mask = context_or_output.mask
            if not bool(torch.equal(context_mask.bool(), mask.bool())):
                raise ValueError("context mask does not match gate mask")
        elif hasattr(context_or_output, "context_tokens"):
            context_tokens = context_or_output.context_tokens
            context_mask = getattr(context_or_output, "mask", None)
            if context_mask is not None and not bool(torch.equal(context_mask.bool(), mask.bool())):
                raise ValueError("context mask does not match gate mask")
        else:
            context_tokens = context_or_output
        context_tokens = _validate_context_tokens(context_tokens, mask, embed_dim=self.embed_dim)
        context_tokens = torch.where(mask[:, :, None], context_tokens, torch.zeros_like(context_tokens))

        feature_rows = self._feature_rows_from_input(feature_rows_or_canonical, mask)
        feature_rows = torch.where(mask[:, :, None], feature_rows, torch.zeros_like(feature_rows))
        feature_summary = self.feature_summary(feature_rows)
        feature_summary = torch.where(mask[:, :, None], _nan_to_num_torch(feature_summary), torch.zeros_like(feature_summary))

        if self.gate_mode == LOCAL_COMPRESSION_GATE_NONE:
            gate_logits = torch.zeros(batch_size, num_particles, num_modalities, dtype=local_tokens.dtype, device=local_tokens.device)
            gates = modality_mask.to(dtype=local_tokens.dtype)
        else:
            context_expanded = context_tokens[:, :, None, :].expand(batch_size, num_particles, num_modalities, self.embed_dim)
            feature_expanded = feature_summary[:, :, None, :].expand(batch_size, num_particles, num_modalities, self.embed_dim)
            gate_input = torch.cat([local_tokens, context_expanded, feature_expanded], dim=-1)
            gate_logits = self.gate_mlp(gate_input).squeeze(-1)
            gate_logits = _nan_to_num_torch(gate_logits)
            gates = torch.sigmoid(gate_logits)
            gates = torch.where(modality_mask, gates, torch.zeros_like(gates))
            gate_logits = torch.where(modality_mask, gate_logits, torch.zeros_like(gate_logits))
        diagnostic_weights = torch.where(modality_mask, gates, torch.zeros_like(gates))
        return LocalCompressionGateOutput(
            gates=gates,
            gate_logits=gate_logits,
            diagnostic_weights=diagnostic_weights,
            mask=mask,
            modality_mask=modality_mask,
            modality_names=tuple(modality_names),
            local_tokens=local_tokens,
            context_tokens=context_tokens,
            feature_rows=feature_rows,
            feature_summary=feature_summary,
            gate_mode=self.gate_mode,
        )


ContextAwareModalityGate = LocalCompressionContextGate


__all__ = [
    "LOCAL_COMPRESSION_GATES_CONTRACT",
    "LOCAL_COMPRESSION_GATES_STEP",
    "ContextAwareModalityGate",
    "LocalCompressionContextGate",
    "LocalCompressionGateOutput",
]

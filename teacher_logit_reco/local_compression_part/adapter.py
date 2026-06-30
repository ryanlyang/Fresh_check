"""Delta-F adapter for local-compression Particle Transformer inputs.

Step 9 is the first module that edits the canonical ParT feature rows.  The
final projection is zero-initialized so the exact HLT ParT baseline is recovered
until training learns a useful residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .config import (
    LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES,
    LOCAL_COMPRESSION_PART_CONTRACT,
    LocalCompressionPartConfig,
)
from .context import ParticleContextOutput
from .features import LocalCompressionCanonicalInputs
from .gates import LocalCompressionGateOutput
from .pooling import LocalCompressionPoolOutput

try:  # Keep package imports cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_COMPRESSION_ADAPTER_STEP = "local_compression_part_step9_delta_f_adapter"
LOCAL_COMPRESSION_ADAPTER_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_delta_f_adapter_v1"

LOCAL_COMPRESSION_PID_FEATURES = (
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
)
LOCAL_COMPRESSION_GEOMETRY_FEATURES = ("part_deltaR", "part_deta", "part_dphi")
LOCAL_COMPRESSION_PID_DELTA_FEATURES = LOCAL_COMPRESSION_PID_FEATURES
LOCAL_COMPRESSION_GEOMETRY_DELTA_FEATURES = LOCAL_COMPRESSION_GEOMETRY_FEATURES


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


def _make_body(input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float) -> Any:
    torch = require_torch()
    return torch.nn.Sequential(
        torch.nn.LayerNorm(int(input_dim)),
        torch.nn.Linear(int(input_dim), int(hidden_dim)),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(int(hidden_dim), int(output_dim)),
    )


def _validate_feature_rows(feature_rows: Any, mask: Any, *, feature_dim: int) -> Any:
    feature_rows = _nan_to_num_torch(feature_rows.float())
    mask = mask.bool()
    if int(feature_rows.ndim) != 3:
        raise ValueError(f"feature_rows must have shape [batch, particles, feature_dim], got {tuple(feature_rows.shape)}")
    if tuple(feature_rows.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"feature_rows/mask leading shapes differ: {tuple(feature_rows.shape[:2])} vs {tuple(mask.shape)}")
    if int(feature_rows.shape[-1]) != int(feature_dim):
        raise ValueError(f"feature_rows last dimension must be feature_dim={int(feature_dim)}, got {int(feature_rows.shape[-1])}")
    return feature_rows


def _validate_particle_tokens(tokens: Any, mask: Any, *, embed_dim: int, name: str) -> Any:
    tokens = _nan_to_num_torch(tokens.float())
    if int(tokens.ndim) != 3:
        raise ValueError(f"{name} must have shape [batch, particles, embed_dim], got {tuple(tokens.shape)}")
    if tuple(tokens.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"{name}/mask leading shapes differ: {tuple(tokens.shape[:2])} vs {tuple(mask.shape)}")
    if int(tokens.shape[-1]) != int(embed_dim):
        raise ValueError(f"{name} last dimension must be embed_dim={int(embed_dim)}, got {int(tokens.shape[-1])}")
    return tokens


def _validate_gates(gates: Any, mask: Any, *, num_modalities: int) -> Any:
    gates = _nan_to_num_torch(gates.float())
    if int(gates.ndim) != 3:
        raise ValueError(f"gates must have shape [batch, particles, modalities], got {tuple(gates.shape)}")
    if tuple(gates.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"gates/mask leading shapes differ: {tuple(gates.shape[:2])} vs {tuple(mask.shape)}")
    if int(gates.shape[-1]) != int(num_modalities):
        raise ValueError(f"gates last dimension must be num_modalities={int(num_modalities)}, got {int(gates.shape[-1])}")
    return torch_clamp_gates(gates)


def torch_clamp_gates(gates: Any) -> Any:
    torch = require_torch()
    return torch.clamp(gates, min=0.0, max=1.0)


@dataclass(frozen=True)
class LocalCompressionDeltaFOutput:
    """Bounded residual feature edits for canonical ParT input rows."""

    delta_F_rows: Any
    raw_delta_rows: Any
    bounded_delta_rows: Any
    adapted_feature_rows: Any
    mask: Any
    feature_rows: Any
    local_particle_token: Any
    context_tokens: Any
    gate_weighted_token: Any
    gates: Any
    feature_delta_scales: Any
    feature_active_mask: Any
    feature_names: tuple[str, ...]
    adapter_diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = require_torch()
        if int(self.delta_F_rows.ndim) != 3:
            raise ValueError("delta_F_rows must have shape [batch, particles, feature_dim]")
        batch_size, num_particles, feature_dim = tuple(self.delta_F_rows.shape)
        expected_feature_shape = (batch_size, num_particles, feature_dim)
        for name in ("raw_delta_rows", "bounded_delta_rows", "adapted_feature_rows", "feature_rows"):
            value = getattr(self, name)
            if tuple(value.shape) != expected_feature_shape:
                raise ValueError(f"{name} shape must match delta_F_rows")
        if tuple(self.mask.shape) != (batch_size, num_particles):
            raise ValueError("mask shape must match delta_F_rows leading dimensions")
        embed_dim = int(self.local_particle_token.shape[-1])
        for name in ("local_particle_token", "context_tokens", "gate_weighted_token"):
            value = getattr(self, name)
            if tuple(value.shape) != (batch_size, num_particles, embed_dim):
                raise ValueError(f"{name} must have shape [batch, particles, embed_dim]")
        if int(self.gates.ndim) != 3 or tuple(self.gates.shape[:2]) != (batch_size, num_particles):
            raise ValueError("gates must have shape [batch, particles, modalities]")
        if tuple(self.feature_delta_scales.shape) != (feature_dim,):
            raise ValueError("feature_delta_scales must have shape [feature_dim]")
        if tuple(self.feature_active_mask.shape) != (feature_dim,):
            raise ValueError("feature_active_mask must have shape [feature_dim]")
        if len(tuple(self.feature_names)) != feature_dim:
            raise ValueError("feature_names length must match feature_dim")
        for name in (
            "delta_F_rows",
            "raw_delta_rows",
            "bounded_delta_rows",
            "adapted_feature_rows",
            "feature_rows",
            "local_particle_token",
            "context_tokens",
            "gate_weighted_token",
            "gates",
            "feature_delta_scales",
            "feature_active_mask",
        ):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} contains non-finite values")
        invalid_delta = torch.where(self.mask[:, :, None], torch.zeros_like(self.delta_F_rows), self.delta_F_rows.abs())
        if float(invalid_delta.max().detach().cpu().item()) > 1.0e-6:
            raise ValueError("invalid particles must have zero delta_F_rows")
        scale = self.feature_delta_scales.abs() * self.feature_active_mask.abs()
        over_bound = torch.clamp(self.delta_F_rows.abs() - scale.view(1, 1, -1), min=0.0)
        if float(over_bound.max().detach().cpu().item()) > 1.0e-5:
            raise ValueError("delta_F_rows exceed configured feature bounds")
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "adapter_diagnostics", dict(self.adapter_diagnostics))

    @property
    def batch_size(self) -> int:
        return int(self.delta_F_rows.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.delta_F_rows.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.delta_F_rows.shape[2])

    @property
    def delta_f_rows(self) -> Any:
        return self.delta_F_rows

    @property
    def feature_delta_mask(self) -> Any:
        return self.feature_active_mask

    def adapted_canonical_inputs(self, canonical: LocalCompressionCanonicalInputs) -> LocalCompressionCanonicalInputs:
        return canonical.with_features(self.adapted_feature_rows)

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_COMPRESSION_ADAPTER_CONTRACT,
            "delta_F_rows_shape": list(self.delta_F_rows.shape),
            "raw_delta_rows_shape": list(self.raw_delta_rows.shape),
            "bounded_delta_rows_shape": list(self.bounded_delta_rows.shape),
            "adapted_feature_rows_shape": list(self.adapted_feature_rows.shape),
            "mask_shape": list(self.mask.shape),
            "feature_names": list(self.feature_names),
            "delta_abs_max": float(self.delta_F_rows.detach().abs().max().cpu().item()),
            "delta_l2_mean": float(self.delta_F_rows.detach().pow(2).sum(dim=-1)[self.mask].mean().cpu().item())
            if bool(self.mask.any())
            else 0.0,
            "active_feature_count": int(self.feature_active_mask.detach().cpu().sum().item()),
            "diagnostics": dict(self.adapter_diagnostics),
        }


class LocalCompressionDeltaFAdapter(_ModuleBase):
    """Predict bounded residual edits to canonical PF feature rows."""

    def __init__(self, config: LocalCompressionPartConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        torch = require_torch()
        self.config = _normalize_model_config(config)
        self.embed_dim = int(self.config.embed_dim)
        self.feature_names = tuple(self.config.feature_config.canonical_feature_names)
        self.feature_dim = len(self.feature_names)
        self.num_modalities = int(self.config.feature_config.num_modalities)
        hidden_dim = max(self.embed_dim, int(round(float(self.config.mlp_ratio) * self.embed_dim)))
        input_dim = self.feature_dim + 3 * self.embed_dim + self.num_modalities
        self.projector = _make_body(input_dim, hidden_dim, self.feature_dim, dropout=float(self.config.dropout))
        if bool(self.config.zero_init_delta_projection):
            final_linear = self.projector[-1]
            torch.nn.init.zeros_(final_linear.weight)
            torch.nn.init.zeros_(final_linear.bias)
        scales = torch.as_tensor(
            tuple(self.config.feature_config.feature_delta_scales),
            dtype=torch.float32,
        )
        if not bool(self.config.feature_config.use_feature_wise_delta_scales):
            scales = torch.ones_like(scales)
        scales = scales * float(self.config.delta_scale)
        active = torch.ones(self.feature_dim, dtype=torch.float32)
        if bool(self.config.freeze_pid_deltas):
            for index, name in enumerate(self.feature_names):
                if name in LOCAL_COMPRESSION_PID_FEATURES:
                    active[index] = 0.0
        if bool(self.config.freeze_geometry_deltas):
            for index, name in enumerate(self.feature_names):
                if name in LOCAL_COMPRESSION_GEOMETRY_FEATURES:
                    active[index] = 0.0
        self.register_buffer("feature_delta_scales", scales)
        self.register_buffer("feature_active_mask", active)

    def _feature_rows_from_input(self, feature_rows_or_canonical: Any) -> tuple[Any, Any]:
        if isinstance(feature_rows_or_canonical, LocalCompressionCanonicalInputs):
            return feature_rows_or_canonical.feature_rows(), feature_rows_or_canonical.particle_mask
        if hasattr(feature_rows_or_canonical, "feature_rows"):
            feature_rows = feature_rows_or_canonical.feature_rows()
            mask = getattr(feature_rows_or_canonical, "particle_mask", None)
            if mask is None:
                raise ValueError("feature_rows source must expose particle_mask")
            return feature_rows, mask
        raise ValueError("feature_rows_or_canonical must be LocalCompressionCanonicalInputs or expose feature_rows()")

    def forward(
        self,
        feature_rows_or_canonical: Any,
        local_or_pool: Any,
        context_or_output: Any,
        gates_or_output: Any,
        *,
        mask: Any | None = None,
    ) -> LocalCompressionDeltaFOutput:
        torch = require_torch()
        if mask is None:
            feature_rows, mask = self._feature_rows_from_input(feature_rows_or_canonical)
        else:
            if isinstance(feature_rows_or_canonical, LocalCompressionCanonicalInputs) or hasattr(feature_rows_or_canonical, "feature_rows"):
                feature_rows, source_mask = self._feature_rows_from_input(feature_rows_or_canonical)
                if not bool(torch.equal(source_mask.bool(), mask.bool())):
                    raise ValueError("feature source mask does not match explicit mask")
            else:
                feature_rows = feature_rows_or_canonical
        mask = mask.bool()
        feature_rows = _validate_feature_rows(feature_rows, mask, feature_dim=self.feature_dim)
        feature_rows = torch.where(mask[:, :, None], feature_rows, torch.zeros_like(feature_rows))

        if isinstance(local_or_pool, LocalCompressionPoolOutput):
            local_particle_token = local_or_pool.local_particle_token
            if not bool(torch.equal(local_or_pool.mask.bool(), mask.bool())):
                raise ValueError("pool mask does not match adapter mask")
        elif hasattr(local_or_pool, "local_particle_token"):
            local_particle_token = local_or_pool.local_particle_token
            source_mask = getattr(local_or_pool, "mask", None)
            if source_mask is not None and not bool(torch.equal(source_mask.bool(), mask.bool())):
                raise ValueError("local token source mask does not match adapter mask")
        else:
            local_particle_token = local_or_pool
        local_particle_token = _validate_particle_tokens(
            local_particle_token,
            mask,
            embed_dim=self.embed_dim,
            name="local_particle_token",
        )
        local_particle_token = torch.where(mask[:, :, None], local_particle_token, torch.zeros_like(local_particle_token))

        if isinstance(context_or_output, ParticleContextOutput):
            context_tokens = context_or_output.context_tokens
            if not bool(torch.equal(context_or_output.mask.bool(), mask.bool())):
                raise ValueError("context mask does not match adapter mask")
        elif hasattr(context_or_output, "context_tokens"):
            context_tokens = context_or_output.context_tokens
            source_mask = getattr(context_or_output, "mask", None)
            if source_mask is not None and not bool(torch.equal(source_mask.bool(), mask.bool())):
                raise ValueError("context source mask does not match adapter mask")
        else:
            context_tokens = context_or_output
        context_tokens = _validate_particle_tokens(context_tokens, mask, embed_dim=self.embed_dim, name="context_tokens")
        context_tokens = torch.where(mask[:, :, None], context_tokens, torch.zeros_like(context_tokens))

        gate_weighted_token = None
        if isinstance(gates_or_output, LocalCompressionGateOutput):
            gates = gates_or_output.gates
            if not bool(torch.equal(gates_or_output.mask.bool(), mask.bool())):
                raise ValueError("gate mask does not match adapter mask")
            gate_weighted_token = (gates_or_output.local_tokens * gates[:, :, :, None]).sum(dim=2)
        elif hasattr(gates_or_output, "gates"):
            gates = gates_or_output.gates
            source_mask = getattr(gates_or_output, "mask", None)
            if source_mask is not None and not bool(torch.equal(source_mask.bool(), mask.bool())):
                raise ValueError("gate source mask does not match adapter mask")
            local_tokens = getattr(gates_or_output, "local_tokens", None)
            if local_tokens is not None:
                gate_weighted_token = (local_tokens * gates[:, :, :, None]).sum(dim=2)
        else:
            gates = gates_or_output
        gates = _validate_gates(gates, mask, num_modalities=self.num_modalities)
        gates = torch.where(mask[:, :, None], gates, torch.zeros_like(gates))
        if gate_weighted_token is None:
            gate_weighted_token = local_particle_token * gates.mean(dim=-1, keepdim=True)
        gate_weighted_token = _validate_particle_tokens(
            gate_weighted_token,
            mask,
            embed_dim=self.embed_dim,
            name="gate_weighted_token",
        )
        gate_weighted_token = torch.where(mask[:, :, None], gate_weighted_token, torch.zeros_like(gate_weighted_token))

        adapter_input = torch.cat(
            [feature_rows, local_particle_token, context_tokens, gate_weighted_token, gates],
            dim=-1,
        )
        raw_delta = self.projector(adapter_input)
        raw_delta = _nan_to_num_torch(raw_delta)
        bounded_delta = torch.tanh(raw_delta)
        scale = (self.feature_delta_scales * self.feature_active_mask).to(dtype=bounded_delta.dtype, device=bounded_delta.device)
        delta = bounded_delta * scale.view(1, 1, -1)
        delta = torch.where(mask[:, :, None], delta, torch.zeros_like(delta))
        raw_delta = torch.where(mask[:, :, None], raw_delta, torch.zeros_like(raw_delta))
        bounded_delta = torch.where(mask[:, :, None], bounded_delta, torch.zeros_like(bounded_delta))
        adapted = feature_rows + delta
        adapted = torch.where(mask[:, :, None], adapted, torch.zeros_like(adapted))
        diagnostics = {
            "zero_init_delta_projection": bool(self.config.zero_init_delta_projection),
            "delta_scale": float(self.config.delta_scale),
            "uses_feature_wise_delta_scales": bool(self.config.feature_config.use_feature_wise_delta_scales),
            "freeze_pid_deltas": bool(self.config.freeze_pid_deltas),
            "freeze_geometry_deltas": bool(self.config.freeze_geometry_deltas),
            "active_feature_count": int(self.feature_active_mask.detach().cpu().sum().item()),
        }
        return LocalCompressionDeltaFOutput(
            delta_F_rows=delta,
            raw_delta_rows=raw_delta,
            bounded_delta_rows=bounded_delta,
            adapted_feature_rows=adapted,
            mask=mask,
            feature_rows=feature_rows,
            local_particle_token=local_particle_token,
            context_tokens=context_tokens,
            gate_weighted_token=gate_weighted_token,
            gates=gates,
            feature_delta_scales=self.feature_delta_scales.to(dtype=delta.dtype, device=delta.device).detach(),
            feature_active_mask=self.feature_active_mask.to(dtype=delta.dtype, device=delta.device).detach(),
            feature_names=self.feature_names,
            adapter_diagnostics=diagnostics,
        )


__all__ = [
    "LOCAL_COMPRESSION_ADAPTER_CONTRACT",
    "LOCAL_COMPRESSION_ADAPTER_STEP",
    "LOCAL_COMPRESSION_GEOMETRY_FEATURES",
    "LOCAL_COMPRESSION_GEOMETRY_DELTA_FEATURES",
    "LOCAL_COMPRESSION_PID_FEATURES",
    "LOCAL_COMPRESSION_PID_DELTA_FEATURES",
    "LocalCompressionDeltaFAdapter",
    "LocalCompressionDeltaFOutput",
]

"""Target-conditioned pairwise denoising modules.

The Step 2 model predicts per-particle HLT-to-offline correction hypotheses
from HLT particles only.  Offline particles remain targets supplied by the
Step 1 dataset and are never inputs to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .data import DENOISING_TARGET_NAMES

try:  # Keep module importable where torch is absent.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


TARGET_DENOISING_STEP2 = "target_conditioned_denoising_part_step2_model"
TARGET_DENOISING_MODEL_CONTRACT = "target_conditioned_pairwise_denoising_model_v1"

PAIRWISE_DENOISING_FEATURE_NAMES = (
    "delta_eta",
    "sin_delta_phi",
    "cos_delta_phi",
    "delta_r",
    "log_delta_r",
    "delta_log_pt",
    "log_pair_mass",
    "log_relative_kt",
    "relative_kt",
    "z",
    "abs_delta_eta",
    "abs_delta_phi",
    "charge_product",
    "same_charge",
    "same_pid",
    "target_local_density",
    "context_local_density",
    "same_particle",
)
PAIRWISE_DENOISING_FEATURE_DIM = len(PAIRWISE_DENOISING_FEATURE_NAMES)


@dataclass(frozen=True)
class PairwiseDenoisingFeatureConfig:
    """Raw-token indices and scaling for denoising pair features."""

    raw_feature_dim: int = RAW_TOKEN_DIM
    pt_index: int = 0
    eta_index: int = 1
    phi_index: int = 2
    energy_index: int = 3
    charge_index: int = 4
    pid_start_index: int = 5
    pid_count: int = 5
    eps: float = 1.0e-6
    local_density_radius: float = 0.08
    local_density_scale: float = 16.0

    def __post_init__(self) -> None:
        raw_feature_dim = int(self.raw_feature_dim)
        if raw_feature_dim <= 0:
            raise ValueError("raw_feature_dim must be positive")
        for name in ("pt_index", "eta_index", "phi_index", "energy_index", "charge_index"):
            index = int(getattr(self, name))
            if index < 0 or index >= raw_feature_dim:
                raise ValueError(f"{name}={index} is outside raw_feature_dim={raw_feature_dim}")
            object.__setattr__(self, name, index)
        pid_start = int(self.pid_start_index)
        pid_count = int(self.pid_count)
        if pid_start < 0 or pid_count <= 0 or pid_start + pid_count > raw_feature_dim:
            raise ValueError("PID feature range is outside raw_feature_dim")
        for name in ("eps", "local_density_radius", "local_density_scale"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "raw_feature_dim", raw_feature_dim)
        object.__setattr__(self, "pid_start_index", pid_start)
        object.__setattr__(self, "pid_count", pid_count)


@dataclass(frozen=True)
class PairwiseDenoisingFeatureOutput:
    """Pairwise denoising features plus masks and useful raw geometry."""

    pair_features: Any
    pair_mask: Any
    token_mask: Any
    delta_r: Any
    feature_names: tuple[str, ...] = PAIRWISE_DENOISING_FEATURE_NAMES

    def summary(self) -> dict[str, Any]:
        return {
            "contract": TARGET_DENOISING_MODEL_CONTRACT,
            "step": TARGET_DENOISING_STEP2,
            "pair_features_shape": list(self.pair_features.shape),
            "pair_mask_shape": list(self.pair_mask.shape),
            "token_mask_shape": list(self.token_mask.shape),
            "feature_names": list(self.feature_names),
        }


@dataclass(frozen=True)
class DenoisingPairBiasConfig:
    """MLP shape for converting denoising pair features to per-head bias."""

    num_heads: int
    pair_feature_dim: int = PAIRWISE_DENOISING_FEATURE_DIM
    hidden_dim: int = 64
    dropout: float = 0.0
    max_abs_bias: float = 4.0
    zero_init: bool = True

    def __post_init__(self) -> None:
        for name in ("num_heads", "pair_feature_dim", "hidden_dim"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        dropout = float(self.dropout)
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        max_abs_bias = float(self.max_abs_bias)
        if max_abs_bias <= 0.0:
            raise ValueError("max_abs_bias must be positive")
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "max_abs_bias", max_abs_bias)
        object.__setattr__(self, "zero_init", bool(self.zero_init))


@dataclass(frozen=True)
class DenoisingResidualHeadConfig:
    """Residual/uncertainty head config."""

    embed_dim: int
    hidden_dim: int = 128
    dropout: float = 0.0
    delta_bounds: tuple[float, float, float, float] = (0.30, 0.08, 0.08, 0.30)
    max_abs_log_variance: float = 6.0
    zero_init: bool = True

    def __post_init__(self) -> None:
        for name in ("embed_dim", "hidden_dim"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        dropout = float(self.dropout)
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        bounds = tuple(float(value) for value in self.delta_bounds)
        if len(bounds) != len(DENOISING_TARGET_NAMES) or any(value <= 0.0 for value in bounds):
            raise ValueError(f"delta_bounds must contain {len(DENOISING_TARGET_NAMES)} positive values")
        max_abs_log_variance = float(self.max_abs_log_variance)
        if max_abs_log_variance <= 0.0:
            raise ValueError("max_abs_log_variance must be positive")
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "delta_bounds", bounds)
        object.__setattr__(self, "max_abs_log_variance", max_abs_log_variance)
        object.__setattr__(self, "zero_init", bool(self.zero_init))


@dataclass(frozen=True)
class TargetConditionedDenoiserConfig:
    """Config for the target-conditioned pairwise denoiser."""

    raw_feature_dim: int = RAW_TOKEN_DIM
    embed_dim: int = 64
    num_heads: int = 4
    pair_hidden_dim: int = 64
    head_hidden_dim: int = 128
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    use_pair_bias: bool = True
    use_local_kernel: bool = True
    local_kernel_radius: float = 0.12
    local_kernel_init: float = 0.0
    pair_bias_max_abs: float = 4.0
    zero_init: bool = True
    delta_bounds: tuple[float, float, float, float] = (0.30, 0.08, 0.08, 0.30)

    def __post_init__(self) -> None:
        for name in ("raw_feature_dim", "embed_dim", "num_heads", "pair_hidden_dim", "head_hidden_dim"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if int(self.embed_dim) % int(self.num_heads) != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
            object.__setattr__(self, name, value)
        for name in ("mlp_ratio", "local_kernel_radius", "pair_bias_max_abs"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "local_kernel_init", float(self.local_kernel_init))
        bounds = tuple(float(value) for value in self.delta_bounds)
        if len(bounds) != len(DENOISING_TARGET_NAMES) or any(value <= 0.0 for value in bounds):
            raise ValueError(f"delta_bounds must contain {len(DENOISING_TARGET_NAMES)} positive values")
        object.__setattr__(self, "delta_bounds", bounds)
        object.__setattr__(self, "use_pair_bias", bool(self.use_pair_bias))
        object.__setattr__(self, "use_local_kernel", bool(self.use_local_kernel))
        object.__setattr__(self, "zero_init", bool(self.zero_init))


@dataclass(frozen=True)
class TargetDenoisingOutput:
    """Forward output of the target-conditioned denoiser."""

    deltas: Any
    log_variances: Any
    reliability: Any
    token_mask: Any
    attention_weights: Any | None
    attention_bias: Any
    pairwise_features: Any
    pair_mask: Any
    target_names: tuple[str, ...] = DENOISING_TARGET_NAMES
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "contract": TARGET_DENOISING_MODEL_CONTRACT,
            "step": TARGET_DENOISING_STEP2,
            "deltas_shape": list(self.deltas.shape),
            "log_variances_shape": list(self.log_variances.shape),
            "reliability_shape": list(self.reliability.shape),
            "attention_bias_shape": list(self.attention_bias.shape),
            "has_attention_weights": self.attention_weights is not None,
            "target_names": list(self.target_names),
            **dict(self.diagnostics),
        }


def _normalize_feature_config(config: PairwiseDenoisingFeatureConfig | Mapping[str, Any] | None) -> PairwiseDenoisingFeatureConfig:
    if config is None:
        return PairwiseDenoisingFeatureConfig()
    if isinstance(config, PairwiseDenoisingFeatureConfig):
        return config
    return PairwiseDenoisingFeatureConfig(**dict(config))


def _normalize_pair_bias_config(config: DenoisingPairBiasConfig | Mapping[str, Any]) -> DenoisingPairBiasConfig:
    if isinstance(config, DenoisingPairBiasConfig):
        return config
    return DenoisingPairBiasConfig(**dict(config))


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))


def wrap_delta_phi_torch(delta_phi: Any) -> Any:
    """Wrap angular differences to [-pi, pi]."""

    torch = require_torch()
    return torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))


def _coerce_tokens_and_mask(tokens: Any, mask: Any) -> tuple[Any, Any]:
    torch = require_torch()
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens, dtype=torch.float32)
    else:
        tokens = tokens.float()
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=tokens.device)
    else:
        mask = mask.to(device=tokens.device, dtype=torch.bool)
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match token rows {tuple(tokens.shape[:2])}")
    return _nan_to_num_torch(tokens), mask


class PairwiseDenoisingFeatureBuilder(_ModuleBase):
    """Build target-context pair features for denoising attention bias."""

    def __init__(self, config: PairwiseDenoisingFeatureConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = _normalize_feature_config(config)

    def forward(self, raw_tokens: Any, mask: Any) -> PairwiseDenoisingFeatureOutput:
        torch = require_torch()
        tokens, mask = _coerce_tokens_and_mask(raw_tokens, mask)
        feature_dim = int(tokens.shape[-1])
        if feature_dim < int(self.config.raw_feature_dim):
            raise ValueError(f"tokens feature dim {feature_dim} is smaller than raw_feature_dim={self.config.raw_feature_dim}")
        pt = torch.clamp(tokens[:, :, int(self.config.pt_index)].abs(), min=float(self.config.eps))
        eta = tokens[:, :, int(self.config.eta_index)]
        phi = tokens[:, :, int(self.config.phi_index)]
        energy = torch.clamp(tokens[:, :, int(self.config.energy_index)].abs(), min=float(self.config.eps))
        charge = tokens[:, :, int(self.config.charge_index)]
        pid = tokens[:, :, int(self.config.pid_start_index) : int(self.config.pid_start_index) + int(self.config.pid_count)]

        delta_eta = eta[:, :, None] - eta[:, None, :]
        delta_phi = wrap_delta_phi_torch(phi[:, :, None] - phi[:, None, :])
        delta_r = torch.sqrt(torch.clamp(delta_eta * delta_eta + delta_phi * delta_phi, min=0.0))
        delta_log_pt = torch.log(pt[:, :, None]) - torch.log(pt[:, None, :])
        relative_kt = torch.minimum(pt[:, :, None], pt[:, None, :]) * delta_r
        z = torch.minimum(pt[:, :, None], pt[:, None, :]) / torch.clamp(pt[:, :, None] + pt[:, None, :], min=float(self.config.eps))

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(torch.clamp(eta, min=-20.0, max=20.0))
        pair_energy = energy[:, :, None] + energy[:, None, :]
        pair_px = px[:, :, None] + px[:, None, :]
        pair_py = py[:, :, None] + py[:, None, :]
        pair_pz = pz[:, :, None] + pz[:, None, :]
        pair_mass2 = torch.clamp(
            pair_energy * pair_energy - pair_px * pair_px - pair_py * pair_py - pair_pz * pair_pz,
            min=float(self.config.eps) * float(self.config.eps),
        )
        pair_mass = torch.sqrt(pair_mass2)

        charge_product = torch.clamp(charge[:, :, None] * charge[:, None, :], -1.0, 1.0)
        same_charge = ((charge[:, :, None] != 0.0) & (charge[:, :, None] == charge[:, None, :])).to(tokens.dtype)
        same_pid = torch.sum(pid[:, :, None, :] * pid[:, None, :, :], dim=-1).clamp(0.0, 1.0)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        nearby = (delta_r < float(self.config.local_density_radius)) & pair_mask
        num_particles = int(tokens.shape[1])
        same_particle = torch.eye(num_particles, dtype=tokens.dtype, device=tokens.device)[None, :, :]
        nearby = nearby & ~same_particle.bool()
        density = nearby.sum(dim=-1).to(tokens.dtype) / float(self.config.local_density_scale)
        density = density.clamp(0.0, 4.0)

        features = torch.stack(
            [
                torch.clamp(delta_eta / 5.0, -2.0, 2.0),
                torch.sin(delta_phi),
                torch.cos(delta_phi),
                torch.clamp(delta_r / 5.0, 0.0, 4.0),
                torch.clamp(torch.log(delta_r + float(self.config.eps)), -14.0, 4.0),
                torch.clamp(delta_log_pt, -8.0, 8.0),
                torch.clamp(torch.log(pair_mass + float(self.config.eps)), -14.0, 14.0),
                torch.clamp(torch.log(relative_kt + float(self.config.eps)), -14.0, 14.0),
                torch.clamp(relative_kt / 1000.0, 0.0, 10.0),
                torch.clamp(z, 0.0, 0.5),
                torch.clamp(delta_eta.abs() / 5.0, 0.0, 4.0),
                torch.clamp(delta_phi.abs() / math.pi, 0.0, 1.0),
                charge_product,
                same_charge,
                same_pid,
                density[:, :, None].expand_as(delta_r),
                density[:, None, :].expand_as(delta_r),
                same_particle.expand(int(tokens.shape[0]), num_particles, num_particles),
            ],
            dim=-1,
        )
        features = torch.where(pair_mask[:, :, :, None], _nan_to_num_torch(features), torch.zeros_like(features))
        delta_r = torch.where(pair_mask, _nan_to_num_torch(delta_r), torch.zeros_like(delta_r))
        return PairwiseDenoisingFeatureOutput(
            pair_features=features,
            pair_mask=pair_mask,
            token_mask=mask,
            delta_r=delta_r,
        )


class DenoisingPairBiasEncoder(_ModuleBase):
    """Convert denoising pair features into additive per-head attention bias."""

    def __init__(self, config: DenoisingPairBiasConfig | Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        self.config = _normalize_pair_bias_config(config)
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.pair_feature_dim)),
            torch.nn.Linear(int(self.config.pair_feature_dim), int(self.config.hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(self.config.hidden_dim), int(self.config.num_heads)),
        )
        if bool(self.config.zero_init):
            final = self.network[-1]
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

    def forward(self, pair_features_or_output: Any, pair_mask: Any | None = None) -> Any:
        torch = require_torch()
        if isinstance(pair_features_or_output, PairwiseDenoisingFeatureOutput):
            pair_features = pair_features_or_output.pair_features
            pair_mask = pair_features_or_output.pair_mask
        else:
            pair_features = pair_features_or_output
        pair_features = _nan_to_num_torch(pair_features.float())
        if int(pair_features.ndim) != 4:
            raise ValueError(f"pair_features must have shape [B, N, N, F], got {tuple(pair_features.shape)}")
        if int(pair_features.shape[-1]) != int(self.config.pair_feature_dim):
            raise ValueError(
                f"pair_features last dim must be {int(self.config.pair_feature_dim)}, got {int(pair_features.shape[-1])}"
            )
        if pair_mask is None:
            pair_mask = torch.ones(pair_features.shape[:3], dtype=torch.bool, device=pair_features.device)
        else:
            pair_mask = pair_mask.to(device=pair_features.device, dtype=torch.bool)
        raw_bias = self.network(pair_features)
        bias = float(self.config.max_abs_bias) * torch.tanh(raw_bias / float(self.config.max_abs_bias))
        bias = bias.permute(0, 3, 1, 2).contiguous()
        bias = torch.where(pair_mask[:, None, :, :], _nan_to_num_torch(bias), torch.zeros_like(bias))
        return bias


class DenoisingResidualHead(_ModuleBase):
    """Predict bounded correction deltas, log variances, and reliability."""

    def __init__(self, config: DenoisingResidualHeadConfig | Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config if isinstance(config, DenoisingResidualHeadConfig) else DenoisingResidualHeadConfig(**dict(config))
        out_dim = 2 * len(DENOISING_TARGET_NAMES) + 1
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.embed_dim)),
            torch.nn.Linear(int(self.config.embed_dim), int(self.config.hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(self.config.hidden_dim), out_dim),
        )
        if bool(self.config.zero_init):
            final = self.network[-1]
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)
        bounds = torch.tensor(tuple(self.config.delta_bounds), dtype=torch.float32)
        self.register_buffer("delta_bounds", bounds, persistent=False)

    def forward(self, tokens: Any, mask: Any) -> tuple[Any, Any, Any]:
        torch = require_torch()
        tokens = _nan_to_num_torch(tokens.float())
        mask = mask.to(device=tokens.device, dtype=torch.bool)
        if int(tokens.ndim) != 3 or int(tokens.shape[-1]) != int(self.config.embed_dim):
            raise ValueError(f"tokens must have shape [B, N, {int(self.config.embed_dim)}], got {tuple(tokens.shape)}")
        raw = self.network(tokens)
        n_targets = len(DENOISING_TARGET_NAMES)
        raw_delta = raw[:, :, :n_targets]
        raw_log_var = raw[:, :, n_targets : 2 * n_targets]
        raw_reliability = raw[:, :, 2 * n_targets]
        deltas = self.delta_bounds.to(device=tokens.device, dtype=tokens.dtype)[None, None, :] * torch.tanh(raw_delta)
        log_variances = float(self.config.max_abs_log_variance) * torch.tanh(
            raw_log_var / float(self.config.max_abs_log_variance)
        )
        reliability = torch.sigmoid(raw_reliability)
        deltas = torch.where(mask[:, :, None], _nan_to_num_torch(deltas), torch.zeros_like(deltas))
        log_variances = torch.where(mask[:, :, None], _nan_to_num_torch(log_variances), torch.zeros_like(log_variances))
        reliability = torch.where(mask, _nan_to_num_torch(reliability), torch.zeros_like(reliability))
        return deltas, log_variances, reliability


class TargetConditionedPairwiseDenoiser(_ModuleBase):
    """Target-query/full-context denoiser with ParT-style pairwise attention bias."""

    def __init__(self, config: TargetConditionedDenoiserConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config if isinstance(config, TargetConditionedDenoiserConfig) else TargetConditionedDenoiserConfig(**dict(config or {}))
        self.feature_builder = PairwiseDenoisingFeatureBuilder(
            PairwiseDenoisingFeatureConfig(raw_feature_dim=int(self.config.raw_feature_dim))
        )
        self.pair_bias_encoder = DenoisingPairBiasEncoder(
            DenoisingPairBiasConfig(
                num_heads=int(self.config.num_heads),
                hidden_dim=int(self.config.pair_hidden_dim),
                dropout=float(self.config.dropout),
                max_abs_bias=float(self.config.pair_bias_max_abs),
                zero_init=bool(self.config.zero_init),
            )
        )
        self.target_embed = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.raw_feature_dim)),
            torch.nn.Linear(int(self.config.raw_feature_dim), int(self.config.embed_dim)),
            torch.nn.GELU(),
        )
        self.context_embed = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.raw_feature_dim)),
            torch.nn.Linear(int(self.config.raw_feature_dim), int(self.config.embed_dim)),
            torch.nn.GELU(),
        )
        self.attention = torch.nn.MultiheadAttention(
            int(self.config.embed_dim),
            int(self.config.num_heads),
            dropout=float(self.config.attention_dropout),
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(float(self.config.dropout))
        self.norm_after_attention = torch.nn.LayerNorm(int(self.config.embed_dim))
        hidden_dim = int(round(float(self.config.mlp_ratio) * int(self.config.embed_dim)))
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(int(self.config.embed_dim), hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(hidden_dim, int(self.config.embed_dim)),
        )
        self.residual_head = DenoisingResidualHead(
            DenoisingResidualHeadConfig(
                embed_dim=int(self.config.embed_dim),
                hidden_dim=int(self.config.head_hidden_dim),
                dropout=float(self.config.dropout),
                delta_bounds=tuple(self.config.delta_bounds),
                zero_init=bool(self.config.zero_init),
            )
        )
        self.pair_gate = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.local_kernel_gate = torch.nn.Parameter(torch.tensor(float(self.config.local_kernel_init), dtype=torch.float32))

    @property
    def output_contract(self) -> str:
        return TARGET_DENOISING_MODEL_CONTRACT

    def _attention_mask(self, attention_bias: Any, token_mask: Any, *, dtype: Any, device: Any) -> Any:
        torch = require_torch()
        attention_bias = attention_bias.to(device=device, dtype=dtype)
        token_mask = token_mask.to(device=device, dtype=torch.bool)
        batch_size, num_heads, num_targets, num_context = (int(value) for value in attention_bias.shape)
        if num_targets != num_context:
            raise ValueError("Step 2 denoiser currently expects square target/context particle sets")
        if tuple(token_mask.shape) != (batch_size, num_context):
            raise ValueError(f"token_mask shape {tuple(token_mask.shape)} does not match {(batch_size, num_context)}")
        masked_bias = attention_bias.masked_fill(~token_mask[:, None, None, :], -1.0e4)
        return masked_bias.reshape(batch_size * num_heads, num_targets, num_context)

    def _local_kernel_bias(self, pairwise: PairwiseDenoisingFeatureOutput, *, dtype: Any) -> Any:
        torch = require_torch()
        if not bool(self.config.use_local_kernel):
            return torch.zeros(
                (int(pairwise.delta_r.shape[0]), int(self.config.num_heads), int(pairwise.delta_r.shape[1]), int(pairwise.delta_r.shape[2])),
                dtype=dtype,
                device=pairwise.delta_r.device,
            )
        radius = float(self.config.local_kernel_radius)
        kernel = torch.exp(-torch.square(pairwise.delta_r.to(dtype=dtype) / radius))
        kernel = torch.where(pairwise.pair_mask, kernel, torch.zeros_like(kernel))
        return self.local_kernel_gate.to(dtype=dtype) * kernel[:, None, :, :]

    def forward(self, raw_tokens: Any, mask: Any, *, need_weights: bool = False) -> TargetDenoisingOutput:
        torch = require_torch()
        tokens, mask = _coerce_tokens_and_mask(raw_tokens, mask)
        if int(tokens.shape[-1]) != int(self.config.raw_feature_dim):
            raise ValueError(
                f"raw token feature dim must be {int(self.config.raw_feature_dim)}, got {int(tokens.shape[-1])}"
            )
        pairwise = self.feature_builder(tokens, mask)
        target = self.target_embed(tokens)
        context = self.context_embed(tokens)
        target = torch.where(mask[:, :, None], target, torch.zeros_like(target))
        context = torch.where(mask[:, :, None], context, torch.zeros_like(context))

        pair_bias = self.pair_bias_encoder(pairwise) if bool(self.config.use_pair_bias) else torch.zeros(
            (int(tokens.shape[0]), int(self.config.num_heads), int(tokens.shape[1]), int(tokens.shape[1])),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        attention_bias = self.pair_gate.to(dtype=tokens.dtype) * pair_bias
        attention_bias = attention_bias + self._local_kernel_bias(pairwise, dtype=tokens.dtype)
        attention_bias = torch.where(pairwise.pair_mask[:, None, :, :], _nan_to_num_torch(attention_bias), torch.zeros_like(attention_bias))
        attn_mask = self._attention_mask(attention_bias, mask, dtype=target.dtype, device=target.device)
        attended, weights = self.attention(
            target,
            context,
            context,
            attn_mask=attn_mask,
            need_weights=bool(need_weights),
            average_attn_weights=False,
        )
        x = target + self.dropout(attended)
        x = x + self.dropout(self.feed_forward(self.norm_after_attention(x)))
        x = torch.where(mask[:, :, None], _nan_to_num_torch(x), torch.zeros_like(x))
        deltas, log_variances, reliability = self.residual_head(x, mask)
        if weights is not None:
            weights = torch.where(mask[:, None, :, None], _nan_to_num_torch(weights), torch.zeros_like(weights))
            weights = torch.where(mask[:, None, None, :], weights, torch.zeros_like(weights))
        diagnostics = self._diagnostics(
            deltas=deltas,
            log_variances=log_variances,
            reliability=reliability,
            attention_bias=attention_bias,
            attention_weights=weights,
            mask=mask,
        )
        return TargetDenoisingOutput(
            deltas=deltas,
            log_variances=log_variances,
            reliability=reliability,
            token_mask=mask,
            attention_weights=weights,
            attention_bias=attention_bias,
            pairwise_features=pairwise.pair_features,
            pair_mask=pairwise.pair_mask,
            diagnostics=diagnostics,
        )

    def _diagnostics(
        self,
        *,
        deltas: Any,
        log_variances: Any,
        reliability: Any,
        attention_bias: Any,
        attention_weights: Any | None,
        mask: Any,
    ) -> dict[str, Any]:
        torch = require_torch()
        valid = mask.bool()
        valid_delta = deltas[valid] if bool(valid.any()) else deltas.reshape(-1, deltas.shape[-1])[:0]
        result: dict[str, Any] = {
            "pair_gate": self.pair_gate.detach(),
            "local_kernel_gate": self.local_kernel_gate.detach(),
            "delta_abs_mean": valid_delta.abs().mean() if valid_delta.numel() else deltas.new_zeros(()),
            "log_variance_abs_mean": log_variances[valid].abs().mean() if bool(valid.any()) else log_variances.new_zeros(()),
            "reliability_mean": reliability[valid].mean() if bool(valid.any()) else reliability.new_zeros(()),
            "attention_bias_abs_mean": attention_bias.abs().mean() if attention_bias.numel() else attention_bias.new_zeros(()),
            "attention_bias_abs_max": attention_bias.abs().amax() if attention_bias.numel() else attention_bias.new_zeros(()),
            "valid_particle_fraction": valid.float().mean() if valid.numel() else torch.zeros((), device=deltas.device),
        }
        if attention_weights is not None:
            result["attention_entropy_mean"] = self._attention_entropy(attention_weights, mask)
        return result

    @staticmethod
    def _attention_entropy(attention_weights: Any, mask: Any) -> Any:
        torch = require_torch()
        weights = torch.clamp(attention_weights, min=1.0e-12)
        entropy = -(weights * torch.log(weights)).sum(dim=-1)
        valid = mask[:, None, :]
        return entropy[valid.expand_as(entropy)].mean() if bool(valid.any()) else entropy.new_zeros(())


__all__ = [
    "TARGET_DENOISING_MODEL_CONTRACT",
    "TARGET_DENOISING_STEP2",
    "PAIRWISE_DENOISING_FEATURE_DIM",
    "PAIRWISE_DENOISING_FEATURE_NAMES",
    "DenoisingPairBiasConfig",
    "DenoisingPairBiasEncoder",
    "DenoisingResidualHead",
    "DenoisingResidualHeadConfig",
    "PairwiseDenoisingFeatureBuilder",
    "PairwiseDenoisingFeatureConfig",
    "PairwiseDenoisingFeatureOutput",
    "TargetConditionedDenoiserConfig",
    "TargetConditionedPairwiseDenoiser",
    "TargetDenoisingOutput",
    "wrap_delta_phi_torch",
]

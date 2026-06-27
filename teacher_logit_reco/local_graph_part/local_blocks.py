"""Local graph residual blocks for graph-augmented Particle Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .edge_features import (
    LOCAL_GRAPH_EDGE_FEATURE_DIM,
    LocalEdgeFeatureBuilder,
    LocalEdgeFeatureConfig,
    LocalEdgeFeatureOutput,
)
from .knn import LocalKnnOutput, gather_local_neighbors

try:  # Keep imports cheap when PyTorch is not available.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_GRAPH_POOL_MEAN_MAX = "mean_max"
LOCAL_GRAPH_POOL_MEAN = "mean"
LOCAL_GRAPH_POOL_MAX = "max"
LOCAL_GRAPH_POOL_MODES = (LOCAL_GRAPH_POOL_MEAN_MAX, LOCAL_GRAPH_POOL_MEAN, LOCAL_GRAPH_POOL_MAX)

LOCAL_GRAPH_EDGECONV_ADAPTER_STEP = "local_graph_part_step3_edgeconv_adapter"
LOCAL_GRAPH_EDGECONV_ADAPTER_CONTRACT = "local_graph_edgeconv_residual_adapter_v1"
LOCAL_GRAPH_POINT_ATTENTION_ADAPTER_STEP = "local_graph_part_step4_point_attention_adapter"
LOCAL_GRAPH_POINT_ATTENTION_ADAPTER_CONTRACT = "local_graph_point_attention_residual_adapter_v1"


@dataclass(frozen=True)
class EdgeConvLocalAdapterConfig:
    """Configuration for the residual local EdgeConv adapter."""

    input_dim: int
    k: int = 16
    hidden_dim: int | None = None
    dropout: float = 0.05
    pool_mode: str = LOCAL_GRAPH_POOL_MEAN_MAX
    residual_gamma_init: float = 0.0
    edge_features: LocalEdgeFeatureConfig | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        input_dim = int(self.input_dim)
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        object.__setattr__(self, "input_dim", input_dim)
        k = int(self.k)
        if k <= 0:
            raise ValueError("k must be positive")
        object.__setattr__(self, "k", k)
        hidden_dim = self.hidden_dim
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive when provided")
        object.__setattr__(self, "hidden_dim", hidden_dim)
        dropout = float(self.dropout)
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        object.__setattr__(self, "dropout", dropout)
        pool_mode = str(self.pool_mode)
        if pool_mode not in LOCAL_GRAPH_POOL_MODES:
            raise ValueError(f"pool_mode must be one of {LOCAL_GRAPH_POOL_MODES}, got {pool_mode!r}")
        object.__setattr__(self, "pool_mode", pool_mode)
        edge_features = self.edge_features
        if edge_features is None:
            edge_features = LocalEdgeFeatureConfig()
        elif isinstance(edge_features, Mapping):
            edge_features = LocalEdgeFeatureConfig(**dict(edge_features))
        if not isinstance(edge_features, LocalEdgeFeatureConfig):
            raise TypeError("edge_features must be a LocalEdgeFeatureConfig, mapping, or None")
        if int(edge_features.knn.k) != k:
            edge_features = LocalEdgeFeatureConfig(
                raw_feature_dim=int(edge_features.raw_feature_dim),
                pt_index=int(edge_features.pt_index),
                eta_index=int(edge_features.eta_index),
                phi_index=int(edge_features.phi_index),
                energy_index=int(edge_features.energy_index),
                charge_index=int(edge_features.charge_index),
                pid_indices=tuple(edge_features.pid_indices),
                d0_index=int(edge_features.d0_index),
                dz_index=int(edge_features.dz_index),
                eps=float(edge_features.eps),
                knn={
                    "k": k,
                    "raw_feature_dim": int(edge_features.knn.raw_feature_dim),
                    "eta_index": int(edge_features.knn.eta_index),
                    "phi_index": int(edge_features.knn.phi_index),
                    "include_self": bool(edge_features.knn.include_self),
                    "eta_clip": float(edge_features.knn.eta_clip),
                },
            )
        object.__setattr__(self, "edge_features", edge_features)
        object.__setattr__(self, "residual_gamma_init", float(self.residual_gamma_init))


@dataclass(frozen=True)
class PointAttentionLocalAdapterConfig:
    """Configuration for the residual local point-attention adapter."""

    input_dim: int
    k: int = 16
    num_heads: int = 4
    hidden_dim: int | None = None
    dropout: float = 0.05
    attention_dropout: float = 0.05
    residual_gamma_init: float = 0.0
    edge_features: LocalEdgeFeatureConfig | Mapping[str, Any] | None = None
    mask_value: float = -1.0e4
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        input_dim = int(self.input_dim)
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        object.__setattr__(self, "input_dim", input_dim)
        k = int(self.k)
        if k <= 0:
            raise ValueError("k must be positive")
        object.__setattr__(self, "k", k)
        num_heads = int(self.num_heads)
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")
        object.__setattr__(self, "num_heads", num_heads)
        hidden_dim = self.hidden_dim
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive when provided")
        object.__setattr__(self, "hidden_dim", hidden_dim)
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
            object.__setattr__(self, name, value)
        edge_features = self.edge_features
        if edge_features is None:
            edge_features = LocalEdgeFeatureConfig()
        elif isinstance(edge_features, Mapping):
            edge_features = LocalEdgeFeatureConfig(**dict(edge_features))
        if not isinstance(edge_features, LocalEdgeFeatureConfig):
            raise TypeError("edge_features must be a LocalEdgeFeatureConfig, mapping, or None")
        if int(edge_features.knn.k) != k:
            edge_features = LocalEdgeFeatureConfig(
                raw_feature_dim=int(edge_features.raw_feature_dim),
                pt_index=int(edge_features.pt_index),
                eta_index=int(edge_features.eta_index),
                phi_index=int(edge_features.phi_index),
                energy_index=int(edge_features.energy_index),
                charge_index=int(edge_features.charge_index),
                pid_indices=tuple(edge_features.pid_indices),
                d0_index=int(edge_features.d0_index),
                dz_index=int(edge_features.dz_index),
                eps=float(edge_features.eps),
                knn={
                    "k": k,
                    "raw_feature_dim": int(edge_features.knn.raw_feature_dim),
                    "eta_index": int(edge_features.knn.eta_index),
                    "phi_index": int(edge_features.knn.phi_index),
                    "include_self": bool(edge_features.knn.include_self),
                    "eta_clip": float(edge_features.knn.eta_clip),
                },
            )
        eps = float(self.eps)
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        object.__setattr__(self, "edge_features", edge_features)
        object.__setattr__(self, "residual_gamma_init", float(self.residual_gamma_init))
        object.__setattr__(self, "mask_value", float(self.mask_value))
        object.__setattr__(self, "eps", eps)


@dataclass(frozen=True)
class EdgeConvLocalAdapterOutput:
    """Output and diagnostics from a residual EdgeConv local adapter."""

    tokens: Any
    local_update: Any
    edge_features: LocalEdgeFeatureOutput
    gamma: Any
    diagnostics: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_GRAPH_EDGECONV_ADAPTER_CONTRACT,
            "tokens_shape": list(self.tokens.shape),
            "local_update_shape": list(self.local_update.shape),
            "edge_features_shape": list(self.edge_features.edge_features.shape),
            "edge_mask_shape": list(self.edge_features.edge_mask.shape),
            "edge_feature_names": list(self.edge_features.feature_names),
            "diagnostic_keys": sorted(str(key) for key in self.diagnostics),
        }


@dataclass(frozen=True)
class PointAttentionLocalAdapterOutput:
    """Output and diagnostics from a residual local point-attention adapter."""

    tokens: Any
    local_update: Any
    attention_weights: Any
    edge_features: LocalEdgeFeatureOutput
    gamma: Any
    diagnostics: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_GRAPH_POINT_ATTENTION_ADAPTER_CONTRACT,
            "tokens_shape": list(self.tokens.shape),
            "local_update_shape": list(self.local_update.shape),
            "attention_weights_shape": list(self.attention_weights.shape),
            "edge_features_shape": list(self.edge_features.edge_features.shape),
            "edge_mask_shape": list(self.edge_features.edge_mask.shape),
            "edge_feature_names": list(self.edge_features.feature_names),
            "diagnostic_keys": sorted(str(key) for key in self.diagnostics),
        }


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _masked_mean(values: Any, mask: Any, dim: int) -> Any:
    torch = require_torch()
    weights = mask.to(dtype=values.dtype)
    numerator = torch.sum(values * weights.unsqueeze(-1), dim=dim)
    denominator = torch.clamp(torch.sum(weights, dim=dim, keepdim=True), min=1.0)
    return numerator / denominator


def _masked_max(values: Any, mask: Any, dim: int) -> Any:
    torch = require_torch()
    very_negative = torch.finfo(values.dtype).min / 16.0
    masked = values.masked_fill(~mask.unsqueeze(-1), very_negative)
    pooled = torch.max(masked, dim=dim).values
    has_any = mask.any(dim=dim)
    return torch.where(has_any.unsqueeze(-1), pooled, torch.zeros_like(pooled))


class EdgeConvLocalAdapter(_ModuleBase):
    """Residual EdgeConv adapter over eta-phi kNN neighborhoods.

    With the default ``residual_gamma_init=0``, valid output tokens initially
    match the input embeddings exactly.  This lets later ParT integration start
    at the HLT baseline and only use local graph information if training finds
    it useful.
    """

    def __init__(self, config: EdgeConvLocalAdapterConfig | Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        if isinstance(config, Mapping):
            config = EdgeConvLocalAdapterConfig(**dict(config))
        self.config = config
        self.edge_builder = LocalEdgeFeatureBuilder(config.edge_features)
        input_dim = int(config.input_dim)
        hidden_dim = int(config.hidden_dim or max(input_dim, 128))
        edge_input_dim = 2 * input_dim + LOCAL_GRAPH_EDGE_FEATURE_DIM
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_input_dim),
            torch.nn.Linear(edge_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(hidden_dim, input_dim),
            torch.nn.GELU(),
        )
        pool_dim = input_dim * (2 if str(config.pool_mode) == LOCAL_GRAPH_POOL_MEAN_MAX else 1)
        self.update_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(pool_dim),
            torch.nn.Linear(pool_dim, input_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(input_dim, input_dim),
        )
        self.update_norm = torch.nn.LayerNorm(input_dim)
        self.gamma = torch.nn.Parameter(torch.tensor(float(config.residual_gamma_init), dtype=torch.float32))

    def forward(
        self,
        particle_embeddings: Any,
        raw_tokens: Any,
        mask: Any,
        knn: LocalKnnOutput | None = None,
    ) -> EdgeConvLocalAdapterOutput:
        torch = require_torch()
        if not isinstance(particle_embeddings, torch.Tensor):
            particle_embeddings = torch.as_tensor(particle_embeddings, dtype=torch.float32)
        else:
            particle_embeddings = particle_embeddings.float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.bool, device=particle_embeddings.device)
        else:
            mask = mask.to(device=particle_embeddings.device, dtype=torch.bool)
        if int(particle_embeddings.ndim) != 3:
            raise ValueError(f"particle_embeddings must have shape [batch, particles, dim], got {tuple(particle_embeddings.shape)}")
        if int(particle_embeddings.shape[-1]) != int(self.config.input_dim):
            raise ValueError(
                f"particle_embeddings last dimension must be {int(self.config.input_dim)}, got {int(particle_embeddings.shape[-1])}"
            )
        if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(particle_embeddings.shape[:2]):
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match embeddings shape {tuple(particle_embeddings.shape[:2])}")

        embeddings = _nan_to_num_torch(particle_embeddings)
        embeddings = torch.where(mask[:, :, None], embeddings, torch.zeros_like(embeddings))
        edge_output = self.edge_builder(raw_tokens, mask, knn)
        edge_mask = edge_output.edge_mask.to(device=embeddings.device, dtype=torch.bool)
        edge_features = edge_output.edge_features.to(device=embeddings.device, dtype=embeddings.dtype)
        neighbor_embeddings = gather_local_neighbors(embeddings, edge_output.knn.indices.to(device=embeddings.device))
        center_embeddings = embeddings[:, :, None, :].expand_as(neighbor_embeddings)
        edge_input = torch.cat([center_embeddings, neighbor_embeddings - center_embeddings, edge_features], dim=-1)
        messages = self.edge_mlp(edge_input)
        messages = torch.where(edge_mask[:, :, :, None], messages, torch.zeros_like(messages))

        pool_mode = str(self.config.pool_mode)
        if pool_mode == LOCAL_GRAPH_POOL_MEAN_MAX:
            pooled = torch.cat(
                [
                    _masked_mean(messages, edge_mask, dim=2),
                    _masked_max(messages, edge_mask, dim=2),
                ],
                dim=-1,
            )
        elif pool_mode == LOCAL_GRAPH_POOL_MEAN:
            pooled = _masked_mean(messages, edge_mask, dim=2)
        else:
            pooled = _masked_max(messages, edge_mask, dim=2)
        local_update = self.update_norm(self.update_proj(pooled))
        local_update = torch.where(mask[:, :, None], _nan_to_num_torch(local_update), torch.zeros_like(local_update))
        output_tokens = embeddings + self.gamma.to(dtype=embeddings.dtype) * local_update
        output_tokens = torch.where(mask[:, :, None], _nan_to_num_torch(output_tokens), torch.zeros_like(output_tokens))

        valid_edges = edge_mask.to(dtype=embeddings.dtype)
        valid_particles = mask.to(dtype=embeddings.dtype)
        diagnostics = {
            "gamma": self.gamma.detach(),
            "mean_valid_neighbors": valid_edges.sum(dim=-1).sum() / torch.clamp(valid_particles.sum(), min=1.0),
            "mean_neighbor_delta_r": edge_output.knn.distances.to(device=embeddings.device, dtype=embeddings.dtype).sum()
            / torch.clamp(valid_edges.sum(), min=1.0),
            "local_update_norm_mean": local_update.norm(dim=-1).sum() / torch.clamp(valid_particles.sum(), min=1.0),
            "embedding_norm_mean": embeddings.norm(dim=-1).sum() / torch.clamp(valid_particles.sum(), min=1.0),
        }
        return EdgeConvLocalAdapterOutput(
            tokens=output_tokens,
            local_update=local_update,
            edge_features=edge_output,
            gamma=self.gamma,
            diagnostics=diagnostics,
        )


class PointAttentionLocalAdapter(_ModuleBase):
    """Residual local point-attention adapter over eta-phi kNN neighborhoods.

    Queries come from each center particle. Neighbor keys/values are shifted by
    learned projections of local edge features, so the adapter can learn
    geometry-aware reliability and substructure patterns before the global
    ParT-style stage.
    """

    def __init__(self, config: PointAttentionLocalAdapterConfig | Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        if isinstance(config, Mapping):
            config = PointAttentionLocalAdapterConfig(**dict(config))
        self.config = config
        self.edge_builder = LocalEdgeFeatureBuilder(config.edge_features)
        input_dim = int(config.input_dim)
        hidden_dim = int(config.hidden_dim or max(input_dim, 128))
        self.query_proj = torch.nn.Linear(input_dim, input_dim)
        self.key_proj = torch.nn.Linear(input_dim, input_dim)
        self.value_proj = torch.nn.Linear(input_dim, input_dim)
        self.edge_key_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(LOCAL_GRAPH_EDGE_FEATURE_DIM),
            torch.nn.Linear(LOCAL_GRAPH_EDGE_FEATURE_DIM, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )
        self.edge_value_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(LOCAL_GRAPH_EDGE_FEATURE_DIM),
            torch.nn.Linear(LOCAL_GRAPH_EDGE_FEATURE_DIM, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )
        self.edge_logit_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(LOCAL_GRAPH_EDGE_FEATURE_DIM),
            torch.nn.Linear(LOCAL_GRAPH_EDGE_FEATURE_DIM, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, int(config.num_heads)),
        )
        self.attention_dropout = torch.nn.Dropout(float(config.attention_dropout))
        self.output_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(hidden_dim, input_dim),
        )
        self.update_norm = torch.nn.LayerNorm(input_dim)
        self.gamma = torch.nn.Parameter(torch.tensor(float(config.residual_gamma_init), dtype=torch.float32))

    def forward(
        self,
        particle_embeddings: Any,
        raw_tokens: Any,
        mask: Any,
        knn: LocalKnnOutput | None = None,
    ) -> PointAttentionLocalAdapterOutput:
        torch = require_torch()
        if not isinstance(particle_embeddings, torch.Tensor):
            particle_embeddings = torch.as_tensor(particle_embeddings, dtype=torch.float32)
        else:
            particle_embeddings = particle_embeddings.float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.bool, device=particle_embeddings.device)
        else:
            mask = mask.to(device=particle_embeddings.device, dtype=torch.bool)
        if int(particle_embeddings.ndim) != 3:
            raise ValueError(f"particle_embeddings must have shape [batch, particles, dim], got {tuple(particle_embeddings.shape)}")
        if int(particle_embeddings.shape[-1]) != int(self.config.input_dim):
            raise ValueError(
                f"particle_embeddings last dimension must be {int(self.config.input_dim)}, got {int(particle_embeddings.shape[-1])}"
            )
        if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(particle_embeddings.shape[:2]):
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match embeddings shape {tuple(particle_embeddings.shape[:2])}")

        embeddings = _nan_to_num_torch(particle_embeddings)
        embeddings = torch.where(mask[:, :, None], embeddings, torch.zeros_like(embeddings))
        edge_output = self.edge_builder(raw_tokens, mask, knn)
        edge_mask = edge_output.edge_mask.to(device=embeddings.device, dtype=torch.bool)
        edge_features = edge_output.edge_features.to(device=embeddings.device, dtype=embeddings.dtype)
        neighbor_embeddings = gather_local_neighbors(embeddings, edge_output.knn.indices.to(device=embeddings.device))

        batch_size, num_particles, num_neighbors, input_dim = neighbor_embeddings.shape
        num_heads = int(self.config.num_heads)
        head_dim = input_dim // num_heads
        center_queries = self.query_proj(embeddings).view(batch_size, num_particles, num_heads, head_dim)
        neighbor_keys = self.key_proj(neighbor_embeddings) + self.edge_key_proj(edge_features)
        neighbor_values = self.value_proj(neighbor_embeddings) + self.edge_value_proj(edge_features)
        neighbor_keys = neighbor_keys.view(batch_size, num_particles, num_neighbors, num_heads, head_dim).permute(0, 1, 3, 2, 4)
        neighbor_values = neighbor_values.view(batch_size, num_particles, num_neighbors, num_heads, head_dim).permute(0, 1, 3, 2, 4)
        edge_logits = self.edge_logit_proj(edge_features).permute(0, 1, 3, 2)

        scores = (center_queries[:, :, :, None, :] * neighbor_keys).sum(dim=-1) / (float(head_dim) ** 0.5)
        scores = scores + edge_logits
        masked_scores = scores.masked_fill(~edge_mask[:, :, None, :], float(self.config.mask_value))
        attention = torch.softmax(masked_scores, dim=-1)
        attention = attention * edge_mask[:, :, None, :].to(dtype=attention.dtype)
        attention = attention / torch.clamp(attention.sum(dim=-1, keepdim=True), min=float(self.config.eps))
        attention = _nan_to_num_torch(attention)
        dropped_attention = self.attention_dropout(attention)

        context = torch.sum(dropped_attention[:, :, :, :, None] * neighbor_values, dim=3)
        context = context.permute(0, 1, 2, 3).contiguous().view(batch_size, num_particles, input_dim)
        local_update = self.update_norm(self.output_proj(context))
        local_update = torch.where(mask[:, :, None], _nan_to_num_torch(local_update), torch.zeros_like(local_update))
        output_tokens = embeddings + self.gamma.to(dtype=embeddings.dtype) * local_update
        output_tokens = torch.where(mask[:, :, None], _nan_to_num_torch(output_tokens), torch.zeros_like(output_tokens))

        valid_edges = edge_mask.to(dtype=embeddings.dtype)
        valid_particles = mask.to(dtype=embeddings.dtype)
        valid_particle_heads = mask[:, :, None].expand(batch_size, num_particles, num_heads).to(dtype=embeddings.dtype)
        entropy = -torch.sum(attention * torch.log(torch.clamp(attention, min=float(self.config.eps))), dim=-1)
        max_attention = torch.max(attention, dim=-1).values
        head_denom = torch.clamp(valid_particle_heads.sum(), min=1.0)
        diagnostics = {
            "gamma": self.gamma.detach(),
            "mean_valid_neighbors": valid_edges.sum(dim=-1).sum() / torch.clamp(valid_particles.sum(), min=1.0),
            "mean_neighbor_delta_r": edge_output.knn.distances.to(device=embeddings.device, dtype=embeddings.dtype).sum()
            / torch.clamp(valid_edges.sum(), min=1.0),
            "attention_entropy_mean": (entropy * valid_particle_heads).sum() / head_denom,
            "attention_max_mean": (max_attention * valid_particle_heads).sum() / head_denom,
            "local_update_norm_mean": local_update.norm(dim=-1).sum() / torch.clamp(valid_particles.sum(), min=1.0),
            "embedding_norm_mean": embeddings.norm(dim=-1).sum() / torch.clamp(valid_particles.sum(), min=1.0),
        }
        return PointAttentionLocalAdapterOutput(
            tokens=output_tokens,
            local_update=local_update,
            attention_weights=attention,
            edge_features=edge_output,
            gamma=self.gamma,
            diagnostics=diagnostics,
        )


__all__ = [
    "LOCAL_GRAPH_EDGECONV_ADAPTER_CONTRACT",
    "LOCAL_GRAPH_EDGECONV_ADAPTER_STEP",
    "LOCAL_GRAPH_POINT_ATTENTION_ADAPTER_CONTRACT",
    "LOCAL_GRAPH_POINT_ATTENTION_ADAPTER_STEP",
    "LOCAL_GRAPH_POOL_MAX",
    "LOCAL_GRAPH_POOL_MEAN",
    "LOCAL_GRAPH_POOL_MEAN_MAX",
    "LOCAL_GRAPH_POOL_MODES",
    "EdgeConvLocalAdapter",
    "EdgeConvLocalAdapterConfig",
    "EdgeConvLocalAdapterOutput",
    "PointAttentionLocalAdapter",
    "PointAttentionLocalAdapterConfig",
    "PointAttentionLocalAdapterOutput",
]

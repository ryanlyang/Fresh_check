"""Local edge feature construction for graph-augmented Particle Transformers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .knn import (
    LocalKnnConfig,
    LocalKnnOutput,
    build_local_knn_graph,
    gather_local_neighbors,
    wrap_local_delta_phi,
)

try:  # Keep imports cheap when PyTorch is not available.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_GRAPH_EDGE_FEATURE_CONTRACT = "local_graph_edge_features_v1"
LOCAL_GRAPH_EDGE_FEATURE_NAMES = (
    "delta_eta",
    "delta_phi",
    "sin_delta_phi",
    "cos_delta_phi",
    "delta_r",
    "log_delta_r",
    "log_pt_center",
    "log_pt_neighbor",
    "delta_log_pt",
    "log_pair_mass",
    "log_relative_kt",
    "relative_kt",
    "z",
    "same_charge_or_both_neutral",
    "pid_dot",
    "abs_delta_d0",
    "abs_delta_dz",
    "neighbor_pt_fraction",
    "neighbor_rank",
    "self_edge",
)
LOCAL_GRAPH_EDGE_FEATURE_DIM = len(LOCAL_GRAPH_EDGE_FEATURE_NAMES)


@dataclass(frozen=True)
class LocalEdgeFeatureConfig:
    """Raw-token feature indices used for local graph edge features."""

    raw_feature_dim: int = RAW_TOKEN_DIM
    pt_index: int = 0
    eta_index: int = 1
    phi_index: int = 2
    energy_index: int = 3
    charge_index: int = 4
    pid_indices: tuple[int, ...] = (5, 6, 7, 8, 9)
    d0_index: int = 10
    dz_index: int = 12
    eps: float = 1.0e-6
    knn: LocalKnnConfig = LocalKnnConfig()

    def __post_init__(self) -> None:
        raw_feature_dim = int(self.raw_feature_dim)
        if raw_feature_dim <= 0:
            raise ValueError("raw_feature_dim must be positive")
        object.__setattr__(self, "raw_feature_dim", raw_feature_dim)
        for name in ("pt_index", "eta_index", "phi_index", "energy_index", "charge_index", "d0_index", "dz_index"):
            index = int(getattr(self, name))
            if index < 0 or index >= raw_feature_dim:
                raise ValueError(f"{name}={index} is outside raw_feature_dim={raw_feature_dim}")
            object.__setattr__(self, name, index)
        pid_indices = tuple(int(index) for index in self.pid_indices)
        if not pid_indices:
            raise ValueError("pid_indices must not be empty")
        for index in pid_indices:
            if index < 0 or index >= raw_feature_dim:
                raise ValueError(f"pid index {index} is outside raw_feature_dim={raw_feature_dim}")
        object.__setattr__(self, "pid_indices", pid_indices)
        eps = float(self.eps)
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        object.__setattr__(self, "eps", eps)
        knn = self.knn
        if isinstance(knn, Mapping):
            knn = LocalKnnConfig(**dict(knn))
        if not isinstance(knn, LocalKnnConfig):
            raise TypeError("knn must be a LocalKnnConfig or mapping")
        if int(knn.raw_feature_dim) != raw_feature_dim:
            knn = LocalKnnConfig(
                k=int(knn.k),
                raw_feature_dim=raw_feature_dim,
                eta_index=int(knn.eta_index),
                phi_index=int(knn.phi_index),
                include_self=bool(knn.include_self),
                eta_clip=float(knn.eta_clip),
            )
        object.__setattr__(self, "knn", knn)


@dataclass(frozen=True)
class LocalEdgeFeatureOutput:
    """Local edge features aligned to a kNN neighborhood."""

    edge_features: Any
    edge_mask: Any
    knn: LocalKnnOutput
    feature_names: tuple[str, ...] = LOCAL_GRAPH_EDGE_FEATURE_NAMES

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_GRAPH_EDGE_FEATURE_CONTRACT,
            "edge_features_shape": list(self.edge_features.shape),
            "edge_mask_shape": list(self.edge_mask.shape),
            "edge_feature_names": list(self.feature_names),
            "knn": self.knn.summary(),
        }


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


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
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match tokens shape {tuple(tokens.shape[:2])}")
    return tokens, mask


def _pair_mass(pt_i: Any, eta_i: Any, phi_i: Any, energy_i: Any, pt_j: Any, eta_j: Any, phi_j: Any, energy_j: Any, eps: float) -> Any:
    torch = require_torch()
    px_i = pt_i * torch.cos(phi_i)
    py_i = pt_i * torch.sin(phi_i)
    pz_i = pt_i * torch.sinh(torch.clamp(eta_i, -20.0, 20.0))
    px_j = pt_j * torch.cos(phi_j)
    py_j = pt_j * torch.sin(phi_j)
    pz_j = pt_j * torch.sinh(torch.clamp(eta_j, -20.0, 20.0))
    mass2 = torch.clamp(
        (energy_i + energy_j).square()
        - (px_i + px_j).square()
        - (py_i + py_j).square()
        - (pz_i + pz_j).square(),
        min=float(eps) * float(eps),
    )
    return torch.sqrt(mass2)


class LocalEdgeFeatureBuilder(_ModuleBase):
    """Build local edge features for each particle and its kNN neighbors."""

    def __init__(self, config: LocalEdgeFeatureConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            config = LocalEdgeFeatureConfig()
        elif isinstance(config, Mapping):
            config = LocalEdgeFeatureConfig(**dict(config))
        self.config = config

    def forward(self, raw_tokens: Any, mask: Any, knn: LocalKnnOutput | None = None) -> LocalEdgeFeatureOutput:
        torch = require_torch()
        tokens, mask = _coerce_tokens_and_mask(raw_tokens, mask)
        feature_dim = int(tokens.shape[-1])
        if feature_dim < int(self.config.raw_feature_dim):
            raise ValueError(f"tokens last dimension {feature_dim} is smaller than configured raw_feature_dim={self.config.raw_feature_dim}")
        tokens = _nan_to_num_torch(tokens)
        if knn is None:
            knn = build_local_knn_graph(tokens, mask, self.config.knn)
        if tuple(knn.indices.shape[:2]) != tuple(tokens.shape[:2]):
            raise ValueError(f"knn indices shape {tuple(knn.indices.shape[:2])} does not match tokens shape {tuple(tokens.shape[:2])}")

        neighbors = gather_local_neighbors(tokens, knn.indices)
        centers = tokens[:, :, None, :].expand_as(neighbors)
        edge_mask = knn.neighbor_mask.to(device=tokens.device, dtype=torch.bool)
        eps = float(self.config.eps)

        pt_i = torch.clamp(centers[..., int(self.config.pt_index)].abs(), min=eps)
        pt_j = torch.clamp(neighbors[..., int(self.config.pt_index)].abs(), min=eps)
        eta_i = centers[..., int(self.config.eta_index)]
        eta_j = neighbors[..., int(self.config.eta_index)]
        phi_i = centers[..., int(self.config.phi_index)]
        phi_j = neighbors[..., int(self.config.phi_index)]
        energy_i = torch.clamp(centers[..., int(self.config.energy_index)].abs(), min=eps)
        energy_j = torch.clamp(neighbors[..., int(self.config.energy_index)].abs(), min=eps)

        delta_eta = eta_i - eta_j
        delta_phi = wrap_local_delta_phi(phi_i - phi_j)
        delta_r = torch.sqrt(torch.clamp(delta_eta.square() + delta_phi.square(), min=0.0))
        log_pt_i = torch.log(pt_i)
        log_pt_j = torch.log(pt_j)
        delta_log_pt = log_pt_j - log_pt_i
        relative_kt = torch.minimum(pt_i, pt_j) * delta_r
        z = torch.minimum(pt_i, pt_j) / torch.clamp(pt_i + pt_j, min=eps)
        pair_mass = _pair_mass(pt_i, eta_i, phi_i, energy_i, pt_j, eta_j, phi_j, energy_j, eps)

        charge_i = centers[..., int(self.config.charge_index)]
        charge_j = neighbors[..., int(self.config.charge_index)]
        same_charge = (torch.abs(charge_i - charge_j) < 0.5).float()
        both_neutral = ((torch.abs(charge_i) < 0.5) & (torch.abs(charge_j) < 0.5)).float()
        same_charge_or_neutral = torch.maximum(same_charge, both_neutral)

        pid_i = torch.clamp(centers[..., list(self.config.pid_indices)], 0.0, 1.0)
        pid_j = torch.clamp(neighbors[..., list(self.config.pid_indices)], 0.0, 1.0)
        pid_dot = torch.sum(pid_i * pid_j, dim=-1)

        abs_delta_d0 = torch.abs(torch.tanh(centers[..., int(self.config.d0_index)]) - torch.tanh(neighbors[..., int(self.config.d0_index)]))
        abs_delta_dz = torch.abs(torch.tanh(centers[..., int(self.config.dz_index)]) - torch.tanh(neighbors[..., int(self.config.dz_index)]))
        local_pt_sum = torch.sum(torch.where(edge_mask, pt_j, torch.zeros_like(pt_j)), dim=-1, keepdim=True)
        neighbor_pt_fraction = pt_j / torch.clamp(local_pt_sum, min=eps)
        num_neighbors = int(knn.indices.shape[-1])
        if num_neighbors <= 1:
            neighbor_rank = torch.zeros_like(delta_r)
        else:
            rank = torch.linspace(0.0, 1.0, steps=num_neighbors, dtype=tokens.dtype, device=tokens.device)
            neighbor_rank = rank[None, None, :].expand_as(delta_r)
        center_indices = torch.arange(int(tokens.shape[1]), dtype=knn.indices.dtype, device=tokens.device)[None, :, None]
        self_edge = (knn.indices.to(device=tokens.device) == center_indices).to(dtype=tokens.dtype)

        edge_features = torch.stack(
            [
                torch.clamp(delta_eta / 5.0, -2.0, 2.0),
                delta_phi,
                torch.sin(delta_phi),
                torch.cos(delta_phi),
                torch.clamp(delta_r / 5.0, 0.0, 4.0),
                torch.clamp(torch.log(delta_r + eps), -14.0, 4.0),
                torch.clamp(log_pt_i, -14.0, 14.0),
                torch.clamp(log_pt_j, -14.0, 14.0),
                torch.clamp(delta_log_pt, -8.0, 8.0),
                torch.clamp(torch.log(pair_mass + eps), -14.0, 14.0),
                torch.clamp(torch.log(relative_kt + eps), -14.0, 14.0),
                torch.clamp(relative_kt / 1000.0, 0.0, 10.0),
                torch.clamp(z, 0.0, 0.5),
                same_charge_or_neutral,
                torch.clamp(pid_dot, 0.0, 1.0),
                torch.clamp(abs_delta_d0, 0.0, 2.0),
                torch.clamp(abs_delta_dz, 0.0, 2.0),
                torch.clamp(neighbor_pt_fraction, 0.0, 1.0),
                neighbor_rank,
                self_edge,
            ],
            dim=-1,
        )
        edge_features = torch.where(edge_mask[:, :, :, None], _nan_to_num_torch(edge_features), torch.zeros_like(edge_features))
        return LocalEdgeFeatureOutput(edge_features=edge_features, edge_mask=edge_mask, knn=knn)

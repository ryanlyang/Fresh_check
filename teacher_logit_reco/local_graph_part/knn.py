"""Valid-particle-aware eta-phi kNN utilities for local graph Particle Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

try:  # Keep imports cheap when PyTorch is not available.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None


LOCAL_GRAPH_COORDINATE_DIM = 2
LOCAL_GRAPH_KNN_CONTRACT = "local_graph_eta_phi_knn_v1"


@dataclass(frozen=True)
class LocalKnnConfig:
    """Configuration for exact eta-phi kNN graph construction."""

    k: int = 16
    raw_feature_dim: int = RAW_TOKEN_DIM
    eta_index: int = 1
    phi_index: int = 2
    include_self: bool = True
    eta_clip: float = 5.0

    def __post_init__(self) -> None:
        raw_feature_dim = int(self.raw_feature_dim)
        if raw_feature_dim <= 0:
            raise ValueError("raw_feature_dim must be positive")
        object.__setattr__(self, "raw_feature_dim", raw_feature_dim)
        k = int(self.k)
        if k <= 0:
            raise ValueError("k must be positive")
        object.__setattr__(self, "k", k)
        for name in ("eta_index", "phi_index"):
            index = int(getattr(self, name))
            if index < 0 or index >= raw_feature_dim:
                raise ValueError(f"{name}={index} is outside raw_feature_dim={raw_feature_dim}")
            object.__setattr__(self, name, index)
        eta_clip = float(self.eta_clip)
        if eta_clip <= 0.0:
            raise ValueError("eta_clip must be positive")
        object.__setattr__(self, "eta_clip", eta_clip)
        object.__setattr__(self, "include_self", bool(self.include_self))


@dataclass(frozen=True)
class LocalKnnOutput:
    """kNN indices plus masks/distances for local graph neighborhoods."""

    indices: Any
    neighbor_mask: Any
    distances: Any
    coordinates: Any
    particle_mask: Any
    include_self: bool

    @property
    def valid_neighbor_counts(self) -> Any:
        """Number of real, unpadded neighbors for every query particle."""

        return self.neighbor_mask.sum(dim=-1)

    def diagnostics(self) -> dict[str, Any]:
        """Tensor diagnostics that keep underfilled neighborhoods visible."""

        torch = require_torch()
        valid_queries = self.particle_mask.to(dtype=torch.bool)
        counts = self.valid_neighbor_counts.to(dtype=self.distances.dtype)
        requested_k = int(self.indices.shape[-1])
        valid_query_count = torch.clamp(valid_queries.sum().to(dtype=self.distances.dtype), min=1.0)
        valid_counts = counts[valid_queries]
        if int(valid_counts.numel()) == 0:
            zero = self.distances.new_zeros(())
            return {
                "mean_valid_neighbors": zero,
                "min_valid_neighbors": zero,
                "max_valid_neighbors": zero,
                "underfilled_particle_fraction": zero,
                "masked_placeholder_fraction": zero,
            }
        underfilled = (counts < requested_k) & valid_queries
        masked_placeholders = (~self.neighbor_mask) & valid_queries[:, :, None]
        placeholder_denominator = torch.clamp(
            valid_query_count * max(float(requested_k), 1.0),
            min=1.0,
        )
        return {
            "mean_valid_neighbors": valid_counts.mean(),
            "min_valid_neighbors": valid_counts.min(),
            "max_valid_neighbors": valid_counts.max(),
            "underfilled_particle_fraction": underfilled.sum().to(dtype=self.distances.dtype) / valid_query_count,
            "masked_placeholder_fraction": masked_placeholders.sum().to(dtype=self.distances.dtype) / placeholder_denominator,
        }

    def summary(self) -> dict[str, Any]:
        payload = {
            "contract": LOCAL_GRAPH_KNN_CONTRACT,
            "indices_shape": list(self.indices.shape),
            "neighbor_mask_shape": list(self.neighbor_mask.shape),
            "distances_shape": list(self.distances.shape),
            "coordinates_shape": list(self.coordinates.shape),
            "particle_mask_shape": list(self.particle_mask.shape),
            "include_self": bool(self.include_self),
        }
        for key, value in self.diagnostics().items():
            payload[key] = float(value.detach().cpu().item())
        return payload


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


def _coerce_coords_and_mask(coords: Any, mask: Any) -> tuple[Any, Any]:
    torch = require_torch()
    if not isinstance(coords, torch.Tensor):
        coords = torch.as_tensor(coords, dtype=torch.float32)
    else:
        coords = coords.float()
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=coords.device)
    else:
        mask = mask.to(device=coords.device, dtype=torch.bool)
    if int(coords.ndim) != 3:
        raise ValueError(f"coords must have shape [batch, particles, dims], got {tuple(coords.shape)}")
    if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(coords.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match coords shape {tuple(coords.shape[:2])}")
    if int(coords.shape[-1]) != LOCAL_GRAPH_COORDINATE_DIM:
        raise ValueError(f"coords last dimension must be {LOCAL_GRAPH_COORDINATE_DIM}, got {coords.shape[-1]}")
    return coords, mask


def wrap_local_delta_phi(delta_phi: Any) -> Any:
    """Wrap a phi difference tensor to ``[-pi, pi]``."""

    torch = require_torch()
    return torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))


def local_eta_phi_coordinates(tokens: Any, mask: Any, config: LocalKnnConfig | None = None) -> Any:
    """Return finite ``(eta, phi)`` coordinates from raw JetClass tokens."""

    torch = require_torch()
    config = config or LocalKnnConfig()
    tokens, mask = _coerce_tokens_and_mask(tokens, mask)
    feature_dim = int(tokens.shape[-1])
    for name in ("eta_index", "phi_index"):
        index = int(getattr(config, name))
        if index >= feature_dim:
            raise ValueError(f"{name}={index} is outside input feature dimension {feature_dim}")
    tokens = _nan_to_num_torch(tokens)
    eta = torch.clamp(tokens[:, :, int(config.eta_index)], -float(config.eta_clip), float(config.eta_clip))
    phi = wrap_local_delta_phi(tokens[:, :, int(config.phi_index)])
    coords = torch.stack([eta, phi], dim=-1)
    return torch.where(mask[:, :, None], coords, torch.zeros_like(coords))


def pairwise_eta_phi_distance(coords: Any) -> Any:
    """Return exact pairwise ``deltaR`` distances from ``(eta, phi)`` coordinates."""

    torch = require_torch()
    if not isinstance(coords, torch.Tensor):
        coords = torch.as_tensor(coords, dtype=torch.float32)
    else:
        coords = coords.float()
    if int(coords.ndim) != 3 or int(coords.shape[-1]) != LOCAL_GRAPH_COORDINATE_DIM:
        raise ValueError(f"coords must have shape [batch, particles, {LOCAL_GRAPH_COORDINATE_DIM}], got {tuple(coords.shape)}")
    coords = _nan_to_num_torch(coords)
    delta_eta = coords[:, :, None, 0] - coords[:, None, :, 0]
    delta_phi = wrap_local_delta_phi(coords[:, :, None, 1] - coords[:, None, :, 1])
    return torch.sqrt(torch.clamp(delta_eta * delta_eta + delta_phi * delta_phi, min=0.0))


def _topk_with_masked_padding(distances: Any, candidate_mask: Any, query_mask: Any, k: int) -> tuple[Any, Any, Any]:
    torch = require_torch()
    batch_size, num_particles, _ = distances.shape
    if int(num_particles) == 0:
        empty_idx = torch.empty(batch_size, 0, k, dtype=torch.long, device=distances.device)
        empty_mask = torch.empty(batch_size, 0, k, dtype=torch.bool, device=distances.device)
        empty_dist = distances.new_empty(batch_size, 0, k)
        return empty_idx, empty_mask, empty_dist

    large = torch.finfo(distances.dtype).max / 16.0
    masked_distances = distances.masked_fill(~candidate_mask, large)
    topk_count = min(int(k), int(num_particles))
    _, indices = torch.topk(masked_distances, k=topk_count, dim=-1, largest=False, sorted=True)
    selected_valid = torch.gather(candidate_mask, dim=2, index=indices) & query_mask[:, :, None]
    fallback_index = candidate_mask.to(dtype=torch.long).argmax(dim=2, keepdim=True)
    indices = torch.where(selected_valid, indices, fallback_index.expand_as(indices))

    if topk_count < int(k):
        pad_count = int(k) - topk_count
        indices = torch.cat([indices, fallback_index.expand(-1, -1, pad_count)], dim=2)
        selected_valid = torch.cat([selected_valid, torch.zeros_like(selected_valid[:, :, :1]).expand(-1, -1, pad_count)], dim=2)

    indices = torch.where(query_mask[:, :, None], indices, torch.zeros_like(indices))
    selected_valid = selected_valid & query_mask[:, :, None]
    selected_distances = torch.gather(distances, dim=2, index=indices)
    selected_distances = torch.where(selected_valid, selected_distances, torch.zeros_like(selected_distances))
    return indices.long(), selected_valid.bool(), selected_distances


def build_local_knn_graph(tokens: Any, mask: Any, config: LocalKnnConfig | None = None) -> LocalKnnOutput:
    """Build an exact valid-aware kNN graph in eta-phi space.

    Invalid padded particles are never valid neighbors. When there are fewer
    than ``k`` available candidates, a valid index is repeated only as a safe
    gather placeholder and the corresponding ``neighbor_mask`` entry is false.
    """

    torch = require_torch()
    config = config or LocalKnnConfig()
    tokens, mask = _coerce_tokens_and_mask(tokens, mask)
    coords = local_eta_phi_coordinates(tokens, mask, config)
    coords, mask = _coerce_coords_and_mask(coords, mask)
    batch_size, num_particles, _ = coords.shape
    distances = pairwise_eta_phi_distance(coords)
    finite_coords = torch.isfinite(coords).all(dim=-1)
    valid_particles = mask & finite_coords
    query_mask = valid_particles
    candidate_mask = valid_particles[:, None, :].expand(batch_size, num_particles, num_particles).clone()
    if not bool(config.include_self) and int(num_particles) > 0:
        eye = torch.eye(num_particles, dtype=torch.bool, device=coords.device)[None, :, :]
        candidate_mask = candidate_mask & ~eye
    candidate_mask = candidate_mask & query_mask[:, :, None]

    indices, neighbor_mask, selected_distances = _topk_with_masked_padding(
        distances,
        candidate_mask,
        query_mask,
        int(config.k),
    )
    return LocalKnnOutput(
        indices=indices,
        neighbor_mask=neighbor_mask,
        distances=selected_distances,
        coordinates=coords,
        particle_mask=valid_particles,
        include_self=bool(config.include_self),
    )


def gather_local_neighbors(features: Any, indices: Any) -> Any:
    """Gather ``features[b, indices[b, i, j]]`` for every query particle ``i``."""

    torch = require_torch()
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if not isinstance(indices, torch.Tensor):
        indices = torch.as_tensor(indices, dtype=torch.long, device=features.device)
    else:
        indices = indices.to(device=features.device, dtype=torch.long)
    if int(features.ndim) != 3:
        raise ValueError(f"features must have shape [batch, particles, channels], got {tuple(features.shape)}")
    if int(indices.ndim) != 3:
        raise ValueError(f"indices must have shape [batch, particles, neighbors], got {tuple(indices.shape)}")
    if tuple(features.shape[:2]) != tuple(indices.shape[:2]):
        raise ValueError(f"features/indices leading shapes differ: {tuple(features.shape[:2])} vs {tuple(indices.shape[:2])}")
    batch_size, num_particles, channels = features.shape
    _, _, num_neighbors = indices.shape
    if int(num_particles) == 0:
        return features.new_empty(batch_size, 0, num_neighbors, channels)
    if bool((indices < 0).any()) or bool((indices >= int(num_particles)).any()):
        raise IndexError("neighbor indices are out of range for features")
    expanded_features = features[:, None, :, :].expand(-1, num_particles, -1, -1)
    gather_index = indices[:, :, :, None].expand(-1, -1, -1, channels)
    return torch.gather(expanded_features, dim=2, index=gather_index)

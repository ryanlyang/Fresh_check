"""ParticleNet-style graph utilities for teacher-logit reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM

from .global_transformer import (
    ENERGY_EPS,
    EPS,
    jet_axes_from_tokens,
    physical_energy_floor,
    placeholder_jet_ids,
    sanitize_hlt_tokens,
    wrap_phi_torch,
)
from .views import SoftReconstructedView

try:  # Keep imports lightweight on machines without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


PARTICLE_NET_INPUT_FEATURE_DIM = 15
PARTICLE_NET_KNN_COORD_DIM = 3


@dataclass
class ParticleNetReconstructorConfig:
    """Configuration for the teacher-logit ParticleNet-style reconstructor."""

    input_dim: int = RAW_TOKEN_DIM
    edgeconv_dims: tuple[int, ...] = (64, 128, 128)
    k: int = 16
    dropout: float = 0.05
    num_extra_candidates: int = 32
    max_delta_logpt: float = 0.50
    max_delta_eta: float = 0.25
    max_delta_phi: float = 0.25
    max_delta_loge: float = 0.50
    parent_weight_bias: float = 4.0
    extra_weight_bias: float = -3.0
    max_total_extra_pt_fraction: float = 0.20
    max_extra_delta_eta: float = 1.25
    max_extra_delta_phi: float = 1.25
    eta_limit: float = 5.0
    min_pt: float = 1.0e-4
    energy_eps: float = ENERGY_EPS

    def __post_init__(self) -> None:
        self.edgeconv_dims = tuple(int(dim) for dim in self.edgeconv_dims)
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        if not self.edgeconv_dims:
            raise ValueError("edgeconv_dims must contain at least one block dimension")
        if any(dim <= 0 for dim in self.edgeconv_dims):
            raise ValueError("edgeconv_dims must all be positive")
        if int(self.k) <= 0:
            raise ValueError("k must be positive")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if int(self.num_extra_candidates) < 0:
            raise ValueError("num_extra_candidates must be non-negative")
        if float(self.max_total_extra_pt_fraction) < 0.0:
            raise ValueError("max_total_extra_pt_fraction must be non-negative")
        if float(self.min_pt) <= 0.0:
            raise ValueError("min_pt must be positive")
        if float(self.energy_eps) <= 0.0:
            raise ValueError("energy_eps must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["edgeconv_dims"] = list(self.edgeconv_dims)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "ParticleNetReconstructorConfig" | None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        payload = dict(value)
        if "edgeconv_dims" in payload:
            payload["edgeconv_dims"] = tuple(payload["edgeconv_dims"])
        return cls(**payload)


def _nan_to_num_torch(value, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0):
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value) + float(nan))


def _validate_token_inputs(tokens, mask) -> None:
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape (batch, particles, features), got {tuple(tokens.shape)}")
    if int(tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"tokens last dimension must be RAW_TOKEN_DIM={RAW_TOKEN_DIM}, got {tokens.shape[-1]}")
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape (batch, particles), got {tuple(mask.shape)}")
    if tuple(tokens.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"tokens/mask leading shapes differ: {tuple(tokens.shape[:2])} vs {tuple(mask.shape)}")


def particle_net_input_features(tokens, mask):
    """Build finite per-particle features for a PN-style graph encoder.

    The feature convention intentionally mirrors the Global Transformer
    embedding features so architecture comparisons are about the graph bias,
    not a different raw feature set.
    """

    torch = require_torch()
    _validate_token_inputs(tokens, mask)
    tokens = _nan_to_num_torch(tokens.float())
    mask = mask.bool()

    pt = torch.clamp(tokens[:, :, 0], min=EPS)
    eta = torch.clamp(tokens[:, :, 1], -5.0, 5.0)
    phi = wrap_phi_torch(tokens[:, :, 2])
    energy = torch.clamp(tokens[:, :, 3], min=EPS)
    pieces = [
        0.2 * torch.log(pt),
        0.2 * torch.log(energy),
        eta / 5.0,
        torch.sin(phi),
        torch.cos(phi),
        torch.clamp(tokens[:, :, 4], -1.0, 1.0),
        torch.clamp(tokens[:, :, 5], 0.0, 1.0),
        torch.clamp(tokens[:, :, 6], 0.0, 1.0),
        torch.clamp(tokens[:, :, 7], 0.0, 1.0),
        torch.clamp(tokens[:, :, 8], 0.0, 1.0),
        torch.clamp(tokens[:, :, 9], 0.0, 1.0),
        torch.tanh(tokens[:, :, 10]),
        torch.clamp(tokens[:, :, 11], 0.0, 1.0),
        torch.tanh(tokens[:, :, 12]),
        torch.clamp(tokens[:, :, 13], 0.0, 1.0),
    ]
    features = torch.stack(pieces, dim=-1)
    return torch.where(mask[:, :, None], features, torch.zeros_like(features))


def particle_net_knn_coordinates(tokens, mask):
    """Return physical kNN coordinates ``(eta, phi, log(pt))``."""

    torch = require_torch()
    _validate_token_inputs(tokens, mask)
    tokens = _nan_to_num_torch(tokens.float())
    mask = mask.bool()
    pt = torch.clamp(tokens[:, :, 0], min=EPS)
    coords = torch.stack(
        [
            torch.clamp(tokens[:, :, 1], -5.0, 5.0),
            wrap_phi_torch(tokens[:, :, 2]),
            torch.log(pt),
        ],
        dim=-1,
    )
    return torch.where(mask[:, :, None], coords, torch.zeros_like(coords))


def _pairwise_particle_net_distance(coords):
    torch = require_torch()
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    if int(coords.shape[-1]) >= 2:
        diff = diff.clone()
        diff[:, :, :, 1] = wrap_phi_torch(diff[:, :, :, 1])
    return torch.sum(diff * diff, dim=-1)


def masked_knn_indices(coords, mask, k: int):
    """Return nearest valid neighbor indices with shape ``(B, N, k)``.

    Candidate particles with ``mask=False`` are never selected when a jet has at
    least one valid particle.  If ``k`` exceeds the number of valid candidates,
    valid neighbors are repeated instead of filling with padded particles.  For
    completely empty jets there is no valid candidate to choose, so index zero
    is used as a harmless placeholder.
    """

    torch = require_torch()
    if int(coords.ndim) != 3:
        raise ValueError(f"coords must have shape (batch, particles, dims), got {tuple(coords.shape)}")
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape (batch, particles), got {tuple(mask.shape)}")
    if tuple(coords.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"coords/mask leading shapes differ: {tuple(coords.shape[:2])} vs {tuple(mask.shape)}")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")

    batch_size, num_particles, _ = coords.shape
    if int(num_particles) == 0:
        return torch.empty(batch_size, 0, k, dtype=torch.long, device=coords.device)

    coords = _nan_to_num_torch(coords.float())
    mask = mask.bool()
    finite_coords = torch.isfinite(coords).all(dim=-1)
    valid_candidates = mask & finite_coords

    distances = _pairwise_particle_net_distance(coords)
    large = torch.finfo(distances.dtype).max / 16.0
    distances = distances.masked_fill(~valid_candidates[:, None, :], large)

    topk_count = min(k, int(num_particles))
    _, indices = torch.topk(distances, k=topk_count, dim=-1, largest=False, sorted=True)

    selected_valid = torch.gather(
        valid_candidates[:, None, :].expand(-1, num_particles, -1),
        dim=2,
        index=indices,
    )
    first_index = indices[:, :, :1]
    indices = torch.where(selected_valid, indices, first_index.expand_as(indices))

    if topk_count < k:
        pad = indices[:, :, -1:].expand(-1, -1, k - topk_count)
        indices = torch.cat([indices, pad], dim=2)
    has_valid_candidate = valid_candidates.any(dim=1)
    indices = torch.where(has_valid_candidate[:, None, None], indices, torch.zeros_like(indices))
    return indices.long()


def gather_neighbor_features(features, indices):
    """Gather ``features[b, indices[b, i, j]]`` for every query particle ``i``."""

    torch = require_torch()
    if int(features.ndim) != 3:
        raise ValueError(f"features must have shape (batch, particles, channels), got {tuple(features.shape)}")
    if int(indices.ndim) != 3:
        raise ValueError(f"indices must have shape (batch, particles, neighbors), got {tuple(indices.shape)}")
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


def _validate_graph_inputs(features, coords, mask) -> None:
    if int(features.ndim) != 3:
        raise ValueError(f"features must have shape (batch, particles, channels), got {tuple(features.shape)}")
    if int(coords.ndim) != 3:
        raise ValueError(f"coords must have shape (batch, particles, dims), got {tuple(coords.shape)}")
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape (batch, particles), got {tuple(mask.shape)}")
    if tuple(features.shape[:2]) != tuple(coords.shape[:2]):
        raise ValueError(f"features/coords leading shapes differ: {tuple(features.shape[:2])} vs {tuple(coords.shape[:2])}")
    if tuple(features.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"features/mask leading shapes differ: {tuple(features.shape[:2])} vs {tuple(mask.shape)}")


class EdgeConvBlock(_ModuleBase):
    """Dynamic EdgeConv block using masked kNN in fixed physical coordinates."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        k: int = 16,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        residual: bool = True,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.k = int(k)
        self.dropout = float(dropout)
        self.use_residual = bool(residual)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        hidden = int(hidden_dim or max(self.input_dim, self.output_dim))
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive when provided")

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self.input_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden, self.output_dim),
            torch.nn.GELU(),
        )
        self.residual_proj = None
        if self.use_residual:
            if self.input_dim == self.output_dim:
                self.residual_proj = torch.nn.Identity()
            else:
                self.residual_proj = torch.nn.Linear(self.input_dim, self.output_dim)
        self.output_norm = torch.nn.LayerNorm(self.output_dim)

    def forward(self, features, coords, mask):
        torch = require_torch()
        _validate_graph_inputs(features, coords, mask)
        if int(features.shape[-1]) != self.input_dim:
            raise ValueError(f"features last dimension must be {self.input_dim}, got {features.shape[-1]}")

        features = _nan_to_num_torch(features.float())
        coords = _nan_to_num_torch(coords.float())
        mask = mask.bool()
        features = torch.where(mask[:, :, None], features, torch.zeros_like(features))

        indices = masked_knn_indices(coords, mask, self.k)
        neighbors = gather_neighbor_features(features, indices)
        centers = features[:, :, None, :].expand_as(neighbors)
        edge_input = torch.cat([centers, neighbors - centers], dim=-1)
        edge_features = self.edge_mlp(edge_input)
        aggregated = edge_features.max(dim=2).values
        if self.residual_proj is not None:
            aggregated = aggregated + self.residual_proj(features)
        aggregated = self.output_norm(aggregated)
        return torch.where(mask[:, :, None], aggregated, torch.zeros_like(aggregated))


class ParticleNetEncoder(_ModuleBase):
    """Stack of EdgeConv blocks for parent-particle encoding."""

    def __init__(
        self,
        input_dim: int = PARTICLE_NET_INPUT_FEATURE_DIM,
        hidden_dims: Sequence[int] = (64, 128, 128),
        *,
        k: int = 16,
        dropout: float = 0.05,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.k = int(k)
        self.dropout = float(dropout)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one block dimension")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must all be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")

        dims = (self.input_dim,) + self.hidden_dims
        self.blocks = torch.nn.ModuleList(
            [
                EdgeConvBlock(
                    dims[index],
                    dims[index + 1],
                    k=self.k,
                    hidden_dim=max(dims[index], dims[index + 1]),
                    dropout=self.dropout,
                    residual=True,
                )
                for index in range(len(self.hidden_dims))
            ]
        )
        self.output_dim = self.hidden_dims[-1]

    def forward(self, features, coords, mask):
        torch = require_torch()
        _validate_graph_inputs(features, coords, mask)
        if int(features.shape[-1]) != self.input_dim:
            raise ValueError(f"features last dimension must be {self.input_dim}, got {features.shape[-1]}")

        mask = mask.bool()
        x = _nan_to_num_torch(features.float())
        x = torch.where(mask[:, :, None], x, torch.zeros_like(x))
        coords = _nan_to_num_torch(coords.float())
        for block in self.blocks:
            x = block(x, coords, mask)
        return torch.where(mask[:, :, None], x, torch.zeros_like(x))


def _masked_mean(values, mask):
    torch = require_torch()
    weights = mask.float()
    denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
    return (values * weights[:, :, None]).sum(dim=1) / denom


def _masked_max(values, mask):
    torch = require_torch()
    if int(values.shape[1]) == 0:
        return values.new_zeros(values.shape[0], values.shape[-1])
    very_negative = torch.finfo(values.dtype).min / 8.0
    masked = values.masked_fill(~mask[:, :, None].bool(), very_negative)
    max_values = masked.max(dim=1).values
    has_valid = mask.any(dim=1)
    return torch.where(has_valid[:, None], max_values, torch.zeros_like(max_values))


class ParticleNetReconstructor(_ModuleBase):
    """PN-style reconstructor with bounded parent edits and soft extra slots."""

    def __init__(self, config: Mapping[str, Any] | ParticleNetReconstructorConfig | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = ParticleNetReconstructorConfig.from_mapping(config)
        self.encoder = ParticleNetEncoder(
            input_dim=PARTICLE_NET_INPUT_FEATURE_DIM,
            hidden_dims=self.config.edgeconv_dims,
            k=int(self.config.k),
            dropout=float(self.config.dropout),
        )
        encoder_dim = int(self.encoder.output_dim)
        self.parent_head = torch.nn.Sequential(
            torch.nn.LayerNorm(encoder_dim),
            torch.nn.Linear(encoder_dim, encoder_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(encoder_dim, 5),
        )

        self.global_context_dim = 2 * encoder_dim
        self.num_extra_candidates = int(self.config.num_extra_candidates)
        if self.num_extra_candidates > 0:
            self.extra_slot_embeddings = torch.nn.Parameter(torch.empty(self.num_extra_candidates, encoder_dim))
            torch.nn.init.normal_(self.extra_slot_embeddings, mean=0.0, std=0.02)
            self.extra_head = torch.nn.Sequential(
                torch.nn.LayerNorm(self.global_context_dim + encoder_dim),
                torch.nn.Linear(self.global_context_dim + encoder_dim, encoder_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(float(self.config.dropout)),
                torch.nn.Linear(encoder_dim, RAW_TOKEN_DIM + 1),
            )
        else:
            self.register_parameter("extra_slot_embeddings", None)
            self.extra_head = None

    def _bounded_parent_corrections(self, raw):
        torch = require_torch()
        deltas = torch.stack(
            [
                float(self.config.max_delta_logpt) * torch.tanh(raw[:, :, 0]),
                float(self.config.max_delta_eta) * torch.tanh(raw[:, :, 1]),
                float(self.config.max_delta_phi) * torch.tanh(raw[:, :, 2]),
                float(self.config.max_delta_loge) * torch.tanh(raw[:, :, 3]),
            ],
            dim=-1,
        )
        parent_weight = torch.sigmoid(raw[:, :, 4] + float(self.config.parent_weight_bias))
        return deltas, parent_weight

    def _apply_parent_corrections(self, tokens, mask, deltas):
        torch = require_torch()
        out = tokens.clone()
        pt = torch.clamp(tokens[:, :, 0], min=float(self.config.min_pt)) * torch.exp(deltas[:, :, 0])
        eta = torch.clamp(tokens[:, :, 1] + deltas[:, :, 1], -float(self.config.eta_limit), float(self.config.eta_limit))
        phi = wrap_phi_torch(tokens[:, :, 2] + deltas[:, :, 2])
        energy = torch.clamp(tokens[:, :, 3], min=float(self.config.energy_eps)) * torch.exp(deltas[:, :, 3])
        energy = torch.maximum(energy, physical_energy_floor(pt, eta, eps=float(self.config.energy_eps)))
        out[:, :, 0] = pt
        out[:, :, 1] = eta
        out[:, :, 2] = phi
        out[:, :, 3] = energy
        return torch.where(mask[:, :, None], out, torch.zeros_like(out))

    def _global_graph_context(self, encoded, mask):
        return torch.cat([_masked_mean(encoded, mask), _masked_max(encoded, mask)], dim=1)

    def _make_extra_candidates(self, global_context, jet_axes):
        torch = require_torch()
        batch_size = int(global_context.shape[0])
        if self.num_extra_candidates == 0:
            empty_tokens = global_context.new_zeros(batch_size, 0, RAW_TOKEN_DIM)
            empty_weights = global_context.new_zeros(batch_size, 0)
            empty_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=global_context.device)
            return empty_tokens, empty_weights, empty_mask

        slots = self.extra_slot_embeddings[None, :, :].expand(batch_size, -1, -1)
        context = global_context[:, None, :].expand(-1, self.num_extra_candidates, -1)
        raw = self.extra_head(torch.cat([context, slots], dim=-1))

        per_slot_pt_fraction = float(self.config.max_total_extra_pt_fraction) / max(self.num_extra_candidates, 1)
        pt = torch.clamp(jet_axes["pt"][:, None], min=float(self.config.min_pt)) * (
            per_slot_pt_fraction * torch.sigmoid(raw[:, :, 0])
        )
        pt = torch.clamp(pt, min=float(self.config.min_pt))
        eta = torch.clamp(
            jet_axes["eta"][:, None] + float(self.config.max_extra_delta_eta) * torch.tanh(raw[:, :, 1]),
            -float(self.config.eta_limit),
            float(self.config.eta_limit),
        )
        phi = wrap_phi_torch(jet_axes["phi"][:, None] + float(self.config.max_extra_delta_phi) * torch.tanh(raw[:, :, 2]))
        energy_scale = torch.exp(float(self.config.max_delta_loge) * torch.tanh(raw[:, :, 3]))
        energy = physical_energy_floor(pt, eta, eps=float(self.config.energy_eps)) * energy_scale
        energy = torch.maximum(energy, physical_energy_floor(pt, eta, eps=float(self.config.energy_eps)))

        tokens = raw.new_zeros(batch_size, self.num_extra_candidates, RAW_TOKEN_DIM)
        tokens[:, :, 0] = pt
        tokens[:, :, 1] = eta
        tokens[:, :, 2] = phi
        tokens[:, :, 3] = energy
        tokens[:, :, 4] = torch.tanh(raw[:, :, 4])
        tokens[:, :, 5:10] = torch.softmax(raw[:, :, 5:10], dim=-1)
        tokens[:, :, 10] = torch.tanh(raw[:, :, 10])
        tokens[:, :, 11] = torch.sigmoid(raw[:, :, 11])
        tokens[:, :, 12] = torch.tanh(raw[:, :, 12])
        tokens[:, :, 13] = torch.sigmoid(raw[:, :, 13])
        weights = torch.sigmoid(raw[:, :, RAW_TOKEN_DIM] + float(self.config.extra_weight_bias))
        extra_mask = torch.ones(batch_size, self.num_extra_candidates, dtype=torch.bool, device=global_context.device)
        return tokens, weights, extra_mask

    def forward(
        self,
        hlt_tokens,
        hlt_mask,
        *,
        labels=None,
        jet_ids: list[JetIdentity] | None = None,
        split: str = "in_memory",
    ) -> SoftReconstructedView:
        torch = require_torch()
        hlt_tokens, hlt_mask, diagnostics = sanitize_hlt_tokens(hlt_tokens, hlt_mask, config=self.config)
        features = particle_net_input_features(hlt_tokens, hlt_mask)
        coords = particle_net_knn_coordinates(hlt_tokens, hlt_mask)
        encoded = self.encoder(features, coords, hlt_mask)

        parent_raw = self.parent_head(encoded)
        parent_delta, parent_weights = self._bounded_parent_corrections(parent_raw)
        parent_tokens = self._apply_parent_corrections(hlt_tokens, hlt_mask, parent_delta)
        parent_weights = torch.where(hlt_mask, parent_weights, torch.zeros_like(parent_weights))

        jet_axes = jet_axes_from_tokens(hlt_tokens, hlt_mask)
        global_context = self._global_graph_context(encoded, hlt_mask)
        extra_tokens, extra_weights, extra_mask = self._make_extra_candidates(global_context, jet_axes)
        tokens = torch.cat([parent_tokens, extra_tokens], dim=1)
        mask = torch.cat([hlt_mask, extra_mask], dim=1)
        weights = torch.cat([parent_weights, extra_weights], dim=1)

        batch_size = int(tokens.shape[0])
        if labels is None:
            labels = torch.full((batch_size,), -1, dtype=torch.long, device=tokens.device)
        else:
            labels = labels.to(device=tokens.device, dtype=torch.long) if isinstance(labels, torch.Tensor) else labels
        if jet_ids is None:
            jet_ids = placeholder_jet_ids(batch_size, labels=labels)

        aux = {
            "sanitized_hlt_tokens": hlt_tokens,
            "sanitized_hlt_mask": hlt_mask,
            "parent_tokens": parent_tokens,
            "parent_delta": parent_delta,
            "parent_weights": parent_weights,
            "extra_tokens": extra_tokens,
            "extra_weights": extra_weights,
            "extra_mask": extra_mask,
            "jet_axes": jet_axes,
            "diagnostics": diagnostics,
            "particle_net_features": features,
            "particle_net_coords": coords,
            "encoded_parent_features": encoded,
            "global_context": global_context,
        }
        return SoftReconstructedView(
            tokens=tokens,
            mask=mask,
            weights=weights,
            labels=labels,
            jet_ids=jet_ids,
            split=split,
            metadata={
                "construction": "particle_net_parents_plus_extras",
                "model_family": "teacher_logit_particle_net",
                "reconstructor_architecture": "particle_net",
                "n_parent_candidates": int(parent_tokens.shape[1]),
                "n_extra_candidates": int(extra_tokens.shape[1]),
                "config": self.config.to_dict(),
                "diagnostics": diagnostics,
            },
            aux=aux,
        )


def build_particle_net_reconstructor(
    config: Mapping[str, Any] | ParticleNetReconstructorConfig | None = None,
) -> ParticleNetReconstructor:
    return ParticleNetReconstructor(config)

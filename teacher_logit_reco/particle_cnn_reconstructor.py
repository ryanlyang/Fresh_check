"""Particle-CNN style teacher-logit reconstructor.

This module implements the rank-convolution reconstructor family.  It uses the
fixed-HLT cache order as the canonical particle-rank axis, applies masked
Conv1d residual blocks over that axis, and emits the same soft reconstructed
view contract as the other teacher-logit reconstructors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM

from .global_transformer import (
    ENERGY_EPS,
    EPS,
    _nan_to_num_torch,
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


PARTICLE_CNN_RECONSTRUCTOR_ARCHITECTURE = "particle_cnn"
PARTICLE_CNN_ORDERING_ASSUMPTION = "fixed_hlt_cache_order_is_canonical_rank_axis"
PARTICLE_CNN_BASE_FEATURE_NAMES = (
    "log_pt",
    "log_energy",
    "eta_scaled",
    "sin_phi",
    "cos_phi",
    "log_pt_fraction",
    "log_energy_fraction",
    "charge",
    "is_charged_hadron",
    "is_neutral_hadron",
    "is_photon",
    "is_electron",
    "is_muon",
    "d0",
    "d0err",
    "dz",
    "dzerr",
    "valid_mask",
)
PARTICLE_CNN_RANK_FEATURE_NAMES = (
    "rank_fraction",
    "log_rank",
    "tail_fraction",
    "is_leading",
    "is_top3",
)
PARTICLE_CNN_FEATURE_NAMES = PARTICLE_CNN_BASE_FEATURE_NAMES + PARTICLE_CNN_RANK_FEATURE_NAMES
PARTICLE_CNN_INPUT_FEATURE_DIM = len(PARTICLE_CNN_FEATURE_NAMES)


def _as_positive_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        dims = (int(value),)
    else:
        dims = tuple(int(dim) for dim in value)
    if not dims:
        raise ValueError(f"{field_name} must contain at least one value")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"{field_name} must contain only positive values")
    return dims


def _require_finite_float(value: Any, *, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _validate_token_inputs(tokens, mask) -> None:
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape (batch, particles, features), got {tuple(tokens.shape)}")
    if int(tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"tokens last dimension must be RAW_TOKEN_DIM={RAW_TOKEN_DIM}, got {tokens.shape[-1]}")
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape (batch, particles), got {tuple(mask.shape)}")
    if tuple(tokens.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"tokens/mask leading shapes differ: {tuple(tokens.shape[:2])} vs {tuple(mask.shape)}")


def _validate_particle_last_inputs(values, mask) -> None:
    if int(values.ndim) != 3:
        raise ValueError(f"values must have shape (batch, particles, features), got {tuple(values.shape)}")
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape (batch, particles), got {tuple(mask.shape)}")
    if tuple(values.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"values/mask leading shapes differ: {tuple(values.shape[:2])} vs {tuple(mask.shape)}")


def _validate_channel_first_inputs(values, mask) -> None:
    if int(values.ndim) != 3:
        raise ValueError(f"values must have shape (batch, channels, particles), got {tuple(values.shape)}")
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape (batch, particles), got {tuple(mask.shape)}")
    if int(values.shape[0]) != int(mask.shape[0]) or int(values.shape[2]) != int(mask.shape[1]):
        raise ValueError(f"values/mask shapes differ: {tuple(values.shape)} vs {tuple(mask.shape)}")


def build_rank_features(mask):
    """Build finite rank-position features for the P-CNN particle axis.

    The features are intentionally based on absolute array position, not on the
    count of valid particles, because the P-CNN bias is tied to canonical cache
    order.  Invalid positions are zeroed by the mask.
    """

    torch = require_torch()
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape (batch, particles), got {tuple(mask.shape)}")
    mask = mask.bool()
    batch_size, num_particles = int(mask.shape[0]), int(mask.shape[1])
    dtype = torch.float32
    device = mask.device
    if num_particles == 0:
        return torch.zeros(batch_size, 0, len(PARTICLE_CNN_RANK_FEATURE_NAMES), dtype=dtype, device=device)

    rank = torch.arange(num_particles, dtype=dtype, device=device)
    rank_fraction = rank / max(num_particles - 1, 1)
    log_rank = torch.log1p(rank) / math.log1p(max(num_particles - 1, 1))
    tail_fraction = 1.0 - rank_fraction
    is_leading = (rank == 0).to(dtype)
    is_top3 = (rank < 3).to(dtype)
    features = torch.stack([rank_fraction, log_rank, tail_fraction, is_leading, is_top3], dim=-1)
    features = features[None, :, :].expand(batch_size, -1, -1)
    return torch.where(mask[:, :, None], features, torch.zeros_like(features))


def build_particle_cnn_features(tokens, mask):
    """Build finite per-particle features for the P-CNN-style reconstructor."""

    torch = require_torch()
    _validate_token_inputs(tokens, mask)
    tokens = _nan_to_num_torch(tokens.float())
    mask = mask.bool()

    pt = torch.clamp(tokens[:, :, 0], min=EPS)
    eta = torch.clamp(tokens[:, :, 1], -5.0, 5.0)
    phi = wrap_phi_torch(tokens[:, :, 2])
    energy = torch.clamp(tokens[:, :, 3], min=EPS)

    mask_float = mask.to(dtype=tokens.dtype)
    sum_pt = torch.clamp((pt * mask_float).sum(dim=1, keepdim=True), min=EPS)
    sum_energy = torch.clamp((energy * mask_float).sum(dim=1, keepdim=True), min=EPS)

    base_pieces = [
        0.2 * torch.log(pt),
        0.2 * torch.log(energy),
        eta / 5.0,
        torch.sin(phi),
        torch.cos(phi),
        torch.log(pt / sum_pt),
        torch.log(energy / sum_energy),
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
        mask_float,
    ]
    base_features = torch.stack(base_pieces, dim=-1)
    rank_features = build_rank_features(mask).to(dtype=base_features.dtype)
    features = torch.cat([base_features, rank_features], dim=-1)
    features = _nan_to_num_torch(features)
    return torch.where(mask[:, :, None], features, torch.zeros_like(features))


def audit_particle_cnn_cache_order(tokens, mask) -> dict[str, Any]:
    """Summarize whether valid adjacent particles are mostly descending in pt.

    The P-CNN intentionally does not sort inside ``forward``.  This audit makes
    the ordering assumption visible in model metadata and saved reports.
    """

    torch = require_torch()
    _validate_token_inputs(tokens, mask)
    tokens = _nan_to_num_torch(tokens.float())
    mask = mask.bool()
    batch_size = int(tokens.shape[0])
    num_particles = int(tokens.shape[1])
    if num_particles < 2:
        return {
            "ordering_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
            "n_jets": batch_size,
            "n_particles": num_particles,
            "checked_adjacent_valid_pairs": 0,
            "descending_adjacent_valid_pairs": 0,
            "non_descending_adjacent_valid_pairs": 0,
            "descending_adjacent_valid_pair_fraction": 1.0,
            "jets_with_any_non_descending_adjacent_pair": 0,
        }

    valid_pairs = mask[:, :-1] & mask[:, 1:]
    left_pt = tokens[:, :-1, 0]
    right_pt = tokens[:, 1:, 0]
    descending = left_pt + EPS >= right_pt
    non_descending_pairs = valid_pairs & ~descending
    checked_pairs = int(valid_pairs.sum().detach().cpu().item())
    non_descending_count = int(non_descending_pairs.sum().detach().cpu().item())
    descending_count = checked_pairs - non_descending_count
    fraction = float(descending_count / checked_pairs) if checked_pairs else 1.0
    return {
        "ordering_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
        "n_jets": batch_size,
        "n_particles": num_particles,
        "checked_adjacent_valid_pairs": checked_pairs,
        "descending_adjacent_valid_pairs": descending_count,
        "non_descending_adjacent_valid_pairs": non_descending_count,
        "descending_adjacent_valid_pair_fraction": fraction,
        "jets_with_any_non_descending_adjacent_pair": int(
            non_descending_pairs.any(dim=1).sum().detach().cpu().item()
        ),
    }


def apply_particle_mask_channels(values, mask):
    """Zero padded particle positions in a Conv1d-style ``[B, C, N]`` tensor."""

    _validate_channel_first_inputs(values, mask)
    values = _nan_to_num_torch(values) if values.is_floating_point() else values
    return torch_where_mask_channel_first(values, mask)


def torch_where_mask_channel_first(values, mask):
    torch = require_torch()
    mask = mask.bool()
    return torch.where(mask[:, None, :], values, torch.zeros_like(values))


def masked_sum_pool(values, mask):
    """Mask-safe sum over particle dimension for ``[B, N, F]`` tensors."""

    _validate_particle_last_inputs(values, mask)
    values = _nan_to_num_torch(values) if values.is_floating_point() else values.float()
    mask = mask.bool()
    return (values * mask[:, :, None].to(dtype=values.dtype)).sum(dim=1)


def masked_mean_pool(values, mask):
    """Mask-safe mean over particle dimension for ``[B, N, F]`` tensors."""

    torch = require_torch()
    _validate_particle_last_inputs(values, mask)
    values = _nan_to_num_torch(values) if values.is_floating_point() else values.float()
    mask = mask.bool()
    valid_count = torch.clamp(mask.sum(dim=1).to(dtype=values.dtype), min=1.0)
    return masked_sum_pool(values, mask) / valid_count[:, None]


def masked_max_pool(values, mask):
    """Mask-safe max over particle dimension for ``[B, N, F]`` tensors."""

    torch = require_torch()
    _validate_particle_last_inputs(values, mask)
    values = _nan_to_num_torch(values) if values.is_floating_point() else values.float()
    mask = mask.bool()
    masked = torch.where(mask[:, :, None], values, torch.full_like(values, -1.0e30))
    pooled = masked.max(dim=1).values
    return torch.where(mask.any(dim=1)[:, None], pooled, torch.zeros_like(pooled))


def _build_particle_cnn_mlp(input_dim: int, hidden_dims: tuple[int, ...], *, dropout: float):
    torch = require_torch()
    input_dim = int(input_dim)
    hidden_dims = _as_positive_int_tuple(hidden_dims, field_name="hidden_dims")
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if float(dropout) < 0.0 or float(dropout) >= 1.0:
        raise ValueError("dropout must be in [0, 1)")

    layers = [torch.nn.LayerNorm(input_dim)]
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(torch.nn.Linear(current_dim, int(hidden_dim)))
        layers.append(torch.nn.GELU())
        if float(dropout) > 0.0:
            layers.append(torch.nn.Dropout(float(dropout)))
        current_dim = int(hidden_dim)
    layers.append(torch.nn.LayerNorm(current_dim))
    return torch.nn.Sequential(*layers)


def _build_particle_cnn_head(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    *,
    dropout: float,
):
    torch = require_torch()
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    hidden_dims = _as_positive_int_tuple(hidden_dims, field_name="hidden_dims")
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if float(dropout) < 0.0 or float(dropout) >= 1.0:
        raise ValueError("dropout must be in [0, 1)")

    layers = [torch.nn.LayerNorm(input_dim)]
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(torch.nn.Linear(current_dim, int(hidden_dim)))
        layers.append(torch.nn.GELU())
        if float(dropout) > 0.0:
            layers.append(torch.nn.Dropout(float(dropout)))
        current_dim = int(hidden_dim)
    layers.append(torch.nn.Linear(current_dim, output_dim))
    return torch.nn.Sequential(*layers)


@dataclass
class ParticleCnnEncoderOutput:
    """Outputs from the P-CNN rank-convolution encoder and context builder."""

    particle_embeddings: Any
    jet_context: Any
    rank_features: Any
    pooling_report: dict[str, Any]


class ParticleCnnBlock(_ModuleBase):
    """Masked residual Conv1d block over canonical particle rank."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.05,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.dropout = float(dropout)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.dilation <= 0:
            raise ValueError("dilation must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        padding = self.dilation * (self.kernel_size - 1) // 2
        self.norm = torch.nn.LayerNorm(self.channels)
        self.rank_conv = torch.nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=padding,
        )
        self.activation = torch.nn.GELU()
        self.drop = torch.nn.Dropout(self.dropout) if self.dropout > 0.0 else torch.nn.Identity()
        self.pointwise = torch.nn.Conv1d(self.channels, self.channels, kernel_size=1)

    def forward(self, values, mask):
        _validate_channel_first_inputs(values, mask)
        if int(values.shape[1]) != self.channels:
            raise ValueError(f"channel dimension must be {self.channels}, got {values.shape[1]}")
        residual = apply_particle_mask_channels(values, mask)
        x = residual.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.rank_conv(x)
        x = apply_particle_mask_channels(x, mask)
        x = self.activation(x)
        x = self.drop(x)
        x = self.pointwise(x)
        x = apply_particle_mask_channels(x, mask)
        return apply_particle_mask_channels(residual + x, mask)


class ParticleCnnContextBuilder(_ModuleBase):
    """Build a whole-jet context from masked rank-convolution embeddings."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        context_dim: int = 256,
        context_mlp_dims: tuple[int, ...] = (256, 256),
        summary_dim: int = 0,
        dropout: float = 0.05,
    ) -> None:
        require_torch()
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.context_dim = int(context_dim)
        self.context_mlp_dims = _as_positive_int_tuple(context_mlp_dims, field_name="context_mlp_dims")
        self.summary_dim = int(summary_dim)
        self.dropout = float(dropout)
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.summary_dim < 0:
            raise ValueError("summary_dim must be non-negative")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        pooled_dim = 3 * self.embedding_dim + 1 + self.summary_dim
        self.context_mlp = _build_particle_cnn_mlp(
            pooled_dim,
            self.context_mlp_dims + (self.context_dim,),
            dropout=self.dropout,
        )

    def forward(self, particle_embeddings, mask, *, summary_features=None):
        torch = require_torch()
        _validate_particle_last_inputs(particle_embeddings, mask)
        if int(particle_embeddings.shape[-1]) != self.embedding_dim:
            raise ValueError(
                f"particle embedding dimension must be {self.embedding_dim}, got {particle_embeddings.shape[-1]}"
            )
        particle_embeddings = _nan_to_num_torch(particle_embeddings) if particle_embeddings.is_floating_point() else particle_embeddings.float()
        mask = mask.bool()
        batch_size = int(particle_embeddings.shape[0])
        if summary_features is None:
            summary_features = particle_embeddings.new_zeros(batch_size, self.summary_dim)
        else:
            if int(summary_features.ndim) != 2:
                raise ValueError(f"summary_features must have shape (batch, features), got {tuple(summary_features.shape)}")
            if int(summary_features.shape[0]) != batch_size:
                raise ValueError(
                    f"summary batch size differs from embeddings: {summary_features.shape[0]} vs {batch_size}"
                )
            if int(summary_features.shape[1]) != self.summary_dim:
                raise ValueError(f"summary feature dimension must be {self.summary_dim}, got {summary_features.shape[1]}")
            summary_features = _nan_to_num_torch(summary_features.float())

        sum_pool = masked_sum_pool(particle_embeddings, mask)
        mean_pool = masked_mean_pool(particle_embeddings, mask)
        max_pool = masked_max_pool(particle_embeddings, mask)
        valid_count = mask.sum(dim=1).to(dtype=particle_embeddings.dtype)
        count_feature = torch.log1p(valid_count)[:, None]
        pooled_context_input = torch.cat(
            [sum_pool, mean_pool, max_pool, count_feature, summary_features],
            dim=-1,
        )
        jet_context = self.context_mlp(pooled_context_input)
        jet_context = torch.where(mask.any(dim=1)[:, None], jet_context, torch.zeros_like(jet_context))
        pooling_report = {
            "sum_pool": sum_pool,
            "mean_pool": mean_pool,
            "max_pool": max_pool,
            "valid_count": valid_count,
            "summary_features": summary_features,
            "pooled_context_input": pooled_context_input,
        }
        return jet_context, pooling_report


class ParticleCnnEncoder(_ModuleBase):
    """P-CNN feature projection, masked residual Conv1d stack, and context."""

    def __init__(
        self,
        *,
        input_dim: int = PARTICLE_CNN_INPUT_FEATURE_DIM,
        hidden_channels: int = 128,
        kernel_sizes: tuple[int, ...] = (5, 5, 3, 3, 3, 3),
        dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4),
        context_dim: int = 256,
        context_mlp_dims: tuple[int, ...] = (256, 256),
        summary_dim: int = 0,
        dropout: float = 0.05,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_channels = int(hidden_channels)
        self.kernel_sizes = _as_positive_int_tuple(kernel_sizes, field_name="kernel_sizes")
        self.dilations = _as_positive_int_tuple(dilations, field_name="dilations")
        self.output_dim = self.hidden_channels
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if len(self.kernel_sizes) != len(self.dilations):
            raise ValueError("kernel_sizes and dilations must have the same length")

        self.input_projection = _build_particle_cnn_mlp(
            self.input_dim,
            (self.hidden_channels,),
            dropout=float(dropout),
        )
        self.blocks = torch.nn.ModuleList(
            [
                ParticleCnnBlock(
                    self.hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=float(dropout),
                )
                for kernel_size, dilation in zip(self.kernel_sizes, self.dilations)
            ]
        )
        self.context_builder = ParticleCnnContextBuilder(
            embedding_dim=self.hidden_channels,
            context_dim=int(context_dim),
            context_mlp_dims=tuple(int(dim) for dim in context_mlp_dims),
            summary_dim=int(summary_dim),
            dropout=float(dropout),
        )
        self.context_dim = int(context_dim)

    def forward(self, features, mask, *, summary_features=None) -> ParticleCnnEncoderOutput:
        torch = require_torch()
        _validate_particle_last_inputs(features, mask)
        if int(features.shape[-1]) != self.input_dim:
            raise ValueError(f"feature dimension must be {self.input_dim}, got {features.shape[-1]}")
        features = _nan_to_num_torch(features) if features.is_floating_point() else features.float()
        mask = mask.bool()
        rank_features = build_rank_features(mask).to(dtype=features.dtype)
        projected = self.input_projection(features)
        projected = torch.where(mask[:, :, None], projected, torch.zeros_like(projected))
        conv_features = projected.transpose(1, 2)
        conv_features = apply_particle_mask_channels(conv_features, mask)
        for block in self.blocks:
            conv_features = block(conv_features, mask)
        particle_embeddings = conv_features.transpose(1, 2)
        particle_embeddings = torch.where(mask[:, :, None], particle_embeddings, torch.zeros_like(particle_embeddings))
        jet_context, pooling_report = self.context_builder(
            particle_embeddings,
            mask,
            summary_features=summary_features,
        )
        return ParticleCnnEncoderOutput(
            particle_embeddings=particle_embeddings,
            jet_context=jet_context,
            rank_features=rank_features,
            pooling_report=pooling_report,
        )


@dataclass
class ParticleCnnReconstructorConfig:
    """Configuration for the teacher-logit P-CNN-style reconstructor."""

    input_dim: int = RAW_TOKEN_DIM
    hidden_channels: int = 128
    num_blocks: int = 6
    kernel_sizes: tuple[int, ...] = (5, 5, 3, 3, 3, 3)
    dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4)
    context_dim: int = 256
    context_mlp_dims: tuple[int, ...] = (256, 256)
    decoder_dims: tuple[int, ...] = (256, 128)
    slot_dim: int | None = None
    dropout: float = 0.05
    num_extra_candidates: int = 32
    max_delta_logpt: float = 1.0
    max_delta_eta: float = 0.35
    max_delta_phi: float = 0.35
    max_delta_loge: float = 1.0
    parent_weight_bias: float = 2.0
    extra_weight_bias: float = -3.0
    max_total_extra_pt_fraction: float = 0.20
    max_extra_delta_eta: float = 1.25
    max_extra_delta_phi: float = 1.25
    eta_limit: float = 5.0
    min_pt: float = 1.0e-4
    energy_eps: float = ENERGY_EPS

    def __post_init__(self) -> None:
        self.input_dim = int(self.input_dim)
        self.hidden_channels = int(self.hidden_channels)
        self.num_blocks = int(self.num_blocks)
        self.kernel_sizes = _as_positive_int_tuple(self.kernel_sizes, field_name="kernel_sizes")
        self.dilations = _as_positive_int_tuple(self.dilations, field_name="dilations")
        self.context_dim = int(self.context_dim)
        self.context_mlp_dims = _as_positive_int_tuple(self.context_mlp_dims, field_name="context_mlp_dims")
        self.decoder_dims = _as_positive_int_tuple(self.decoder_dims, field_name="decoder_dims")
        if self.slot_dim is not None:
            self.slot_dim = int(self.slot_dim)

        if self.input_dim != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if len(self.kernel_sizes) != self.num_blocks:
            raise ValueError("kernel_sizes length must match num_blocks")
        if len(self.dilations) != self.num_blocks:
            raise ValueError("dilations length must match num_blocks")
        if any(kernel % 2 == 0 for kernel in self.kernel_sizes):
            raise ValueError("kernel_sizes must be odd so Conv1d blocks preserve rank alignment")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.slot_dim is not None and self.slot_dim <= 0:
            raise ValueError("slot_dim must be positive when provided")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if int(self.num_extra_candidates) < 0:
            raise ValueError("num_extra_candidates must be non-negative")
        if float(self.max_total_extra_pt_fraction) < 0.0:
            raise ValueError("max_total_extra_pt_fraction must be non-negative")
        if float(self.eta_limit) <= 0.0:
            raise ValueError("eta_limit must be positive")
        if float(self.min_pt) <= 0.0:
            raise ValueError("min_pt must be positive")
        if float(self.energy_eps) <= 0.0:
            raise ValueError("energy_eps must be positive")

        for field_name in (
            "dropout",
            "max_delta_logpt",
            "max_delta_eta",
            "max_delta_phi",
            "max_delta_loge",
            "parent_weight_bias",
            "extra_weight_bias",
            "max_total_extra_pt_fraction",
            "max_extra_delta_eta",
            "max_extra_delta_phi",
            "eta_limit",
            "min_pt",
            "energy_eps",
        ):
            setattr(self, field_name, _require_finite_float(getattr(self, field_name), field_name=field_name))

        self.num_extra_candidates = int(self.num_extra_candidates)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kernel_sizes"] = list(self.kernel_sizes)
        payload["dilations"] = list(self.dilations)
        payload["context_mlp_dims"] = list(self.context_mlp_dims)
        payload["decoder_dims"] = list(self.decoder_dims)
        payload["reconstructor_architecture"] = PARTICLE_CNN_RECONSTRUCTOR_ARCHITECTURE
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "ParticleCnnReconstructorConfig" | None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        payload = dict(value)
        payload.pop("reconstructor_architecture", None)
        payload.pop("architecture", None)
        if "context_dims" in payload and "context_mlp_dims" not in payload:
            payload["context_mlp_dims"] = payload.pop("context_dims")
        for key in ("kernel_sizes", "dilations", "context_mlp_dims", "decoder_dims"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)


class ParticleCnnReconstructor(_ModuleBase):
    """P-CNN-style reconstructor with rank-conv parent edits and extras."""

    def __init__(self, config: ParticleCnnReconstructorConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = ParticleCnnReconstructorConfig.from_mapping(config)
        self.reconstructor_architecture = PARTICLE_CNN_RECONSTRUCTOR_ARCHITECTURE
        self.encoder = ParticleCnnEncoder(
            input_dim=PARTICLE_CNN_INPUT_FEATURE_DIM,
            hidden_channels=int(self.config.hidden_channels),
            kernel_sizes=self.config.kernel_sizes,
            dilations=self.config.dilations,
            context_dim=int(self.config.context_dim),
            context_mlp_dims=self.config.context_mlp_dims,
            summary_dim=0,
            dropout=float(self.config.dropout),
        )
        parent_input_dim = int(self.encoder.output_dim) + int(self.config.context_dim) + PARTICLE_CNN_INPUT_FEATURE_DIM
        self.parent_head = _build_particle_cnn_head(
            parent_input_dim,
            self.config.decoder_dims,
            5,
            dropout=float(self.config.dropout),
        )
        self.num_extra_candidates = int(self.config.num_extra_candidates)
        self.slot_dim = int(self.config.slot_dim or self.config.context_dim)
        if self.num_extra_candidates > 0:
            self.extra_slot_embeddings = torch.nn.Parameter(torch.empty(self.num_extra_candidates, self.slot_dim))
            torch.nn.init.normal_(self.extra_slot_embeddings, mean=0.0, std=0.02)
            self.extra_head = _build_particle_cnn_head(
                int(self.config.context_dim) + self.slot_dim,
                self.config.decoder_dims,
                RAW_TOKEN_DIM + 1,
                dropout=float(self.config.dropout),
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

    def _make_extra_candidates(self, jet_context, jet_axes):
        torch = require_torch()
        batch_size = int(jet_context.shape[0])
        if self.num_extra_candidates == 0:
            empty_tokens = jet_context.new_zeros(batch_size, 0, RAW_TOKEN_DIM)
            empty_weights = jet_context.new_zeros(batch_size, 0)
            empty_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=jet_context.device)
            return empty_tokens, empty_weights, empty_mask

        slots = self.extra_slot_embeddings[None, :, :].expand(batch_size, -1, -1)
        context = jet_context[:, None, :].expand(-1, self.num_extra_candidates, -1)
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
        extra_mask = torch.ones(batch_size, self.num_extra_candidates, dtype=torch.bool, device=jet_context.device)
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
        cache_order_audit = audit_particle_cnn_cache_order(hlt_tokens, hlt_mask)
        features = build_particle_cnn_features(hlt_tokens, hlt_mask)
        encoder_output = self.encoder(features, hlt_mask)
        particle_embeddings = encoder_output.particle_embeddings
        jet_context = encoder_output.jet_context

        context_per_particle = jet_context[:, None, :].expand(-1, int(hlt_tokens.shape[1]), -1)
        parent_decoder_input = torch.cat([particle_embeddings, context_per_particle, features], dim=-1)
        parent_raw = self.parent_head(parent_decoder_input)
        parent_delta, parent_weights = self._bounded_parent_corrections(parent_raw)
        parent_tokens = self._apply_parent_corrections(hlt_tokens, hlt_mask, parent_delta)
        parent_weights = torch.where(hlt_mask, parent_weights, torch.zeros_like(parent_weights))

        jet_axes = jet_axes_from_tokens(hlt_tokens, hlt_mask)
        extra_tokens, extra_weights, extra_mask = self._make_extra_candidates(jet_context, jet_axes)
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

        diagnostics = {
            **diagnostics,
            "cache_order_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
            "cache_order_audit": cache_order_audit,
        }
        aux = {
            "sanitized_hlt_tokens": hlt_tokens,
            "sanitized_hlt_mask": hlt_mask,
            "particle_cnn_features": features,
            "rank_features": encoder_output.rank_features,
            "particle_embeddings": particle_embeddings,
            "jet_context": jet_context,
            "pooling_report": encoder_output.pooling_report,
            "parent_decoder_input": parent_decoder_input,
            "parent_tokens": parent_tokens,
            "parent_delta": parent_delta,
            "parent_weights": parent_weights,
            "extra_tokens": extra_tokens,
            "extra_weights": extra_weights,
            "extra_mask": extra_mask,
            "jet_axes": jet_axes,
            "cache_order_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
            "cache_order_audit": cache_order_audit,
            "diagnostics": diagnostics,
        }
        return SoftReconstructedView(
            tokens=tokens,
            mask=mask,
            weights=weights,
            labels=labels,
            jet_ids=jet_ids,
            split=split,
            metadata={
                "construction": "particle_cnn_parents_plus_extras",
                "model_family": "teacher_logit_particle_cnn",
                "reconstructor_architecture": PARTICLE_CNN_RECONSTRUCTOR_ARCHITECTURE,
                "ordering_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
                "cache_order_audit": cache_order_audit,
                "n_parent_candidates": int(parent_tokens.shape[1]),
                "n_extra_candidates": int(extra_tokens.shape[1]),
                "config": self.config.to_dict(),
                "diagnostics": diagnostics,
            },
            aux=aux,
        )


def build_particle_cnn_reconstructor(
    config: ParticleCnnReconstructorConfig | Mapping[str, Any] | None = None,
) -> ParticleCnnReconstructor:
    return ParticleCnnReconstructor(ParticleCnnReconstructorConfig.from_mapping(config))


__all__ = [
    "PARTICLE_CNN_BASE_FEATURE_NAMES",
    "PARTICLE_CNN_FEATURE_NAMES",
    "PARTICLE_CNN_INPUT_FEATURE_DIM",
    "PARTICLE_CNN_ORDERING_ASSUMPTION",
    "PARTICLE_CNN_RANK_FEATURE_NAMES",
    "PARTICLE_CNN_RECONSTRUCTOR_ARCHITECTURE",
    "ParticleCnnBlock",
    "ParticleCnnContextBuilder",
    "ParticleCnnEncoder",
    "ParticleCnnEncoderOutput",
    "ParticleCnnReconstructor",
    "ParticleCnnReconstructorConfig",
    "apply_particle_mask_channels",
    "audit_particle_cnn_cache_order",
    "build_particle_cnn_features",
    "build_particle_cnn_reconstructor",
    "build_rank_features",
    "masked_max_pool",
    "masked_mean_pool",
    "masked_sum_pool",
]

"""Particle Flow Network style reconstructor registration and utilities.

This module starts the PFN reconstructor line with the stable public config and
builder contract.  Step 2 adds the shared PFN input features and masked pooling
helpers.  The encoder, context builder, and soft-view forward pass are
implemented in later plan steps.
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
    sanitize_reconstructed_view_tensors,
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


PARTICLE_FLOW_RECONSTRUCTOR_ARCHITECTURE = "particle_flow"
PARTICLE_FLOW_FEATURE_NAMES = (
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
PARTICLE_FLOW_INPUT_FEATURE_DIM = len(PARTICLE_FLOW_FEATURE_NAMES)
PARTICLE_FLOW_SUMMARY_FEATURE_NAMES = (
    "log_total_pt",
    "log_total_energy",
    "log_valid_count",
    "pt_weighted_eta",
    "pt_weighted_sin_phi",
    "pt_weighted_cos_phi",
    "mean_abs_eta",
    "charged_hadron_pt_fraction",
    "neutral_hadron_pt_fraction",
    "photon_pt_fraction",
    "electron_pt_fraction",
    "muon_pt_fraction",
)
PARTICLE_FLOW_SUMMARY_FEATURE_DIM = len(PARTICLE_FLOW_SUMMARY_FEATURE_NAMES)


def _as_positive_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        dims = (int(value),)
    else:
        dims = tuple(int(dim) for dim in value)
    if not dims:
        raise ValueError(f"{field_name} must contain at least one dimension")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"{field_name} must contain only positive dimensions")
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


def _validate_pool_inputs(values, mask) -> None:
    if int(values.ndim) != 3:
        raise ValueError(f"values must have shape (batch, particles, features), got {tuple(values.shape)}")
    if int(mask.ndim) != 2:
        raise ValueError(f"mask must have shape (batch, particles), got {tuple(mask.shape)}")
    if tuple(values.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"values/mask leading shapes differ: {tuple(values.shape[:2])} vs {tuple(mask.shape)}")


def build_particle_flow_features(tokens, mask):
    """Build finite per-particle features for the PFN-style reconstructor.

    The PFN feature set keeps the same raw token convention as the other
    teacher-logit reconstructors, but adds explicit relative-to-jet energy-flow
    channels and a valid-particle mask channel for pooled-set modeling.
    """

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

    pieces = [
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
    features = torch.stack(pieces, dim=-1)
    features = _nan_to_num_torch(features)
    return torch.where(mask[:, :, None], features, torch.zeros_like(features))


def build_particle_flow_summary_features(tokens, mask):
    """Build coarse permutation-invariant HLT jet summary features."""

    torch = require_torch()
    _validate_token_inputs(tokens, mask)
    tokens = _nan_to_num_torch(tokens.float())
    mask = mask.bool()
    mask_float = mask.to(dtype=tokens.dtype)

    pt = torch.clamp(tokens[:, :, 0], min=EPS)
    eta = torch.clamp(tokens[:, :, 1], -5.0, 5.0)
    phi = wrap_phi_torch(tokens[:, :, 2])
    energy = torch.clamp(tokens[:, :, 3], min=EPS)

    valid_count = mask_float.sum(dim=1)
    has_valid = valid_count > 0
    weighted_pt = pt * mask_float
    weighted_energy = energy * mask_float
    total_pt = weighted_pt.sum(dim=1)
    total_energy = weighted_energy.sum(dim=1)
    safe_total_pt = torch.clamp(total_pt, min=EPS)
    safe_total_energy = torch.clamp(total_energy, min=EPS)
    safe_count = torch.clamp(valid_count, min=1.0)

    pt_weighted_eta = (weighted_pt * eta).sum(dim=1) / safe_total_pt / 5.0
    pt_weighted_sin_phi = (weighted_pt * torch.sin(phi)).sum(dim=1) / safe_total_pt
    pt_weighted_cos_phi = (weighted_pt * torch.cos(phi)).sum(dim=1) / safe_total_pt
    mean_abs_eta = (mask_float * eta.abs()).sum(dim=1) / safe_count / 5.0

    pid_pt_fractions = []
    for column in range(5, 10):
        pid = torch.clamp(tokens[:, :, column], 0.0, 1.0)
        pid_pt_fractions.append((weighted_pt * pid).sum(dim=1) / safe_total_pt)

    pieces = [
        torch.where(has_valid, 0.2 * torch.log(safe_total_pt), torch.zeros_like(total_pt)),
        torch.where(has_valid, 0.2 * torch.log(safe_total_energy), torch.zeros_like(total_energy)),
        torch.log1p(valid_count),
        torch.where(has_valid, pt_weighted_eta, torch.zeros_like(pt_weighted_eta)),
        torch.where(has_valid, pt_weighted_sin_phi, torch.zeros_like(pt_weighted_sin_phi)),
        torch.where(has_valid, pt_weighted_cos_phi, torch.zeros_like(pt_weighted_cos_phi)),
        torch.where(has_valid, mean_abs_eta, torch.zeros_like(mean_abs_eta)),
        *[
            torch.where(has_valid, fraction, torch.zeros_like(fraction))
            for fraction in pid_pt_fractions
        ],
    ]
    summary = torch.stack(pieces, dim=-1)
    return _nan_to_num_torch(summary)


def _floating_pool_values(values):
    if values.is_floating_point():
        return _nan_to_num_torch(values)
    return values.float()


def masked_sum_pool(values, mask):
    """Sum particle features over valid particles only."""

    _validate_pool_inputs(values, mask)
    values = _floating_pool_values(values)
    weights = mask.bool().to(dtype=values.dtype)
    return (values * weights[:, :, None]).sum(dim=1)


def masked_mean_pool(values, mask):
    """Mean-pool particle features, returning zeros for empty jets."""

    torch = require_torch()
    _validate_pool_inputs(values, mask)
    values = _floating_pool_values(values)
    weights = mask.bool().to(dtype=values.dtype)
    denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
    return (values * weights[:, :, None]).sum(dim=1) / denom


def masked_max_pool(values, mask):
    """Max-pool particle features, returning zeros for empty jets."""

    torch = require_torch()
    _validate_pool_inputs(values, mask)
    values = _floating_pool_values(values)
    if int(values.shape[1]) == 0:
        return values.new_zeros((int(values.shape[0]), int(values.shape[2])))

    mask = mask.bool()
    fill_value = -torch.finfo(values.dtype).max
    masked_values = values.masked_fill(~mask[:, :, None], fill_value)
    pooled = masked_values.max(dim=1).values
    return torch.where(mask.any(dim=1)[:, None], pooled, torch.zeros_like(pooled))


def _build_particle_flow_mlp(input_dim: int, hidden_dims: tuple[int, ...], *, dropout: float):
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


def _build_particle_flow_head(
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
class ParticleFlowEncoderOutput:
    """Outputs from the PFN shared-phi encoder and pooled context builder."""

    particle_embeddings: Any
    jet_context: Any
    pooling_report: dict[str, Any]


class ParticleFlowContextBuilder(_ModuleBase):
    """Build a permutation-invariant PFN jet context from particle embeddings."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        context_dim: int = 256,
        context_mlp_dims: tuple[int, ...] = (256, 256),
        summary_dim: int = PARTICLE_FLOW_SUMMARY_FEATURE_DIM,
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

        pooled_dim = 3 * self.embedding_dim + 1 + self.summary_dim
        self.context_mlp = _build_particle_flow_mlp(
            pooled_dim,
            self.context_mlp_dims + (self.context_dim,),
            dropout=self.dropout,
        )

    def forward(self, particle_embeddings, mask, *, summary_features=None):
        torch = require_torch()
        _validate_pool_inputs(particle_embeddings, mask)
        if int(particle_embeddings.shape[-1]) != self.embedding_dim:
            raise ValueError(
                f"particle embedding dimension must be {self.embedding_dim}, got {particle_embeddings.shape[-1]}"
            )

        particle_embeddings = _floating_pool_values(particle_embeddings)
        mask = mask.bool()
        batch_size = int(particle_embeddings.shape[0])
        if summary_features is None:
            summary_features = particle_embeddings.new_zeros(batch_size, self.summary_dim)
        else:
            if int(summary_features.ndim) != 2:
                raise ValueError(
                    f"summary_features must have shape (batch, features), got {tuple(summary_features.shape)}"
                )
            if int(summary_features.shape[0]) != batch_size:
                raise ValueError(
                    f"summary batch size differs from embeddings: {summary_features.shape[0]} vs {batch_size}"
                )
            if int(summary_features.shape[1]) != self.summary_dim:
                raise ValueError(
                    f"summary feature dimension must be {self.summary_dim}, got {summary_features.shape[1]}"
                )
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


class ParticleFlowEncoder(_ModuleBase):
    """PFN shared-phi encoder plus permutation-invariant context builder."""

    def __init__(
        self,
        *,
        input_dim: int = PARTICLE_FLOW_INPUT_FEATURE_DIM,
        phi_dims: tuple[int, ...] = (128, 128, 128),
        context_dim: int = 256,
        context_mlp_dims: tuple[int, ...] = (256, 256),
        summary_dim: int = PARTICLE_FLOW_SUMMARY_FEATURE_DIM,
        dropout: float = 0.05,
    ) -> None:
        require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.phi_dims = _as_positive_int_tuple(phi_dims, field_name="phi_dims")
        self.output_dim = int(self.phi_dims[-1])
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.phi_mlp = _build_particle_flow_mlp(self.input_dim, self.phi_dims, dropout=float(dropout))
        self.context_builder = ParticleFlowContextBuilder(
            embedding_dim=self.output_dim,
            context_dim=int(context_dim),
            context_mlp_dims=tuple(int(dim) for dim in context_mlp_dims),
            summary_dim=int(summary_dim),
            dropout=float(dropout),
        )
        self.context_dim = int(context_dim)

    def forward(self, features, mask, *, summary_features=None) -> ParticleFlowEncoderOutput:
        torch = require_torch()
        _validate_pool_inputs(features, mask)
        if int(features.shape[-1]) != self.input_dim:
            raise ValueError(f"feature dimension must be {self.input_dim}, got {features.shape[-1]}")
        features = _floating_pool_values(features)
        mask = mask.bool()
        particle_embeddings = self.phi_mlp(features)
        particle_embeddings = torch.where(mask[:, :, None], particle_embeddings, torch.zeros_like(particle_embeddings))
        jet_context, pooling_report = self.context_builder(
            particle_embeddings,
            mask,
            summary_features=summary_features,
        )
        return ParticleFlowEncoderOutput(
            particle_embeddings=particle_embeddings,
            jet_context=jet_context,
            pooling_report=pooling_report,
        )


@dataclass
class ParticleFlowReconstructorConfig:
    """Configuration for the teacher-logit PFN-style reconstructor."""

    input_dim: int = RAW_TOKEN_DIM
    phi_dims: tuple[int, ...] = (128, 128, 128)
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
        self.phi_dims = _as_positive_int_tuple(self.phi_dims, field_name="phi_dims")
        self.context_dim = int(self.context_dim)
        self.context_mlp_dims = _as_positive_int_tuple(
            self.context_mlp_dims,
            field_name="context_mlp_dims",
        )
        self.decoder_dims = _as_positive_int_tuple(self.decoder_dims, field_name="decoder_dims")
        if self.slot_dim is not None:
            self.slot_dim = int(self.slot_dim)

        if self.input_dim != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
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
        payload["phi_dims"] = list(self.phi_dims)
        payload["context_mlp_dims"] = list(self.context_mlp_dims)
        payload["decoder_dims"] = list(self.decoder_dims)
        payload["reconstructor_architecture"] = PARTICLE_FLOW_RECONSTRUCTOR_ARCHITECTURE
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "ParticleFlowReconstructorConfig" | None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        payload = dict(value)
        payload.pop("reconstructor_architecture", None)
        payload.pop("architecture", None)
        if "context_dims" in payload and "context_mlp_dims" not in payload:
            payload["context_mlp_dims"] = payload.pop("context_dims")
        for key in ("phi_dims", "context_mlp_dims", "decoder_dims"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)


class ParticleFlowReconstructor(_ModuleBase):
    """PFN-style reconstructor with global-context parent edits and extras."""

    def __init__(self, config: ParticleFlowReconstructorConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = ParticleFlowReconstructorConfig.from_mapping(config)
        self.reconstructor_architecture = PARTICLE_FLOW_RECONSTRUCTOR_ARCHITECTURE
        self.encoder = ParticleFlowEncoder(
            input_dim=PARTICLE_FLOW_INPUT_FEATURE_DIM,
            phi_dims=self.config.phi_dims,
            context_dim=self.config.context_dim,
            context_mlp_dims=self.config.context_mlp_dims,
            summary_dim=PARTICLE_FLOW_SUMMARY_FEATURE_DIM,
            dropout=float(self.config.dropout),
        )
        parent_input_dim = int(self.encoder.output_dim) + int(self.config.context_dim) + PARTICLE_FLOW_INPUT_FEATURE_DIM
        self.parent_head = _build_particle_flow_head(
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
            self.extra_head = _build_particle_flow_head(
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
        pt = torch.clamp(tokens[:, :, 0], min=float(self.config.min_pt)) * torch.exp(deltas[:, :, 0])
        eta = torch.clamp(tokens[:, :, 1] + deltas[:, :, 1], -float(self.config.eta_limit), float(self.config.eta_limit))
        phi = wrap_phi_torch(tokens[:, :, 2] + deltas[:, :, 2])
        energy = torch.clamp(tokens[:, :, 3], min=float(self.config.energy_eps)) * torch.exp(deltas[:, :, 3])
        energy = torch.maximum(energy, physical_energy_floor(pt, eta, eps=float(self.config.energy_eps)))
        out = torch.cat(
            [
                pt[:, :, None],
                eta[:, :, None],
                phi[:, :, None],
                energy[:, :, None],
                tokens[:, :, 4:],
            ],
            dim=-1,
        )
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

        tokens = torch.cat(
            [
                pt[:, :, None],
                eta[:, :, None],
                phi[:, :, None],
                energy[:, :, None],
                torch.tanh(raw[:, :, 4])[:, :, None],
                torch.softmax(raw[:, :, 5:10], dim=-1),
                torch.tanh(raw[:, :, 10])[:, :, None],
                torch.sigmoid(raw[:, :, 11])[:, :, None],
                torch.tanh(raw[:, :, 12])[:, :, None],
                torch.sigmoid(raw[:, :, 13])[:, :, None],
            ],
            dim=-1,
        )
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
        features = build_particle_flow_features(hlt_tokens, hlt_mask)
        summary_features = build_particle_flow_summary_features(hlt_tokens, hlt_mask)
        encoder_output = self.encoder(features, hlt_mask, summary_features=summary_features)
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
        tokens, mask, weights, reco_diagnostics = sanitize_reconstructed_view_tensors(
            tokens,
            mask,
            weights,
            config=self.config,
        )
        diagnostics = {**diagnostics, **reco_diagnostics}
        n_parent_candidates = int(parent_tokens.shape[1])
        parent_tokens = tokens[:, :n_parent_candidates, :]
        parent_weights = weights[:, :n_parent_candidates]
        extra_tokens = tokens[:, n_parent_candidates:, :]
        extra_weights = weights[:, n_parent_candidates:]
        extra_mask = mask[:, n_parent_candidates:]

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
            "particle_flow_features": features,
            "particle_flow_summary_features": summary_features,
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
                "construction": "particle_flow_parents_plus_extras",
                "model_family": "teacher_logit_particle_flow",
                "reconstructor_architecture": PARTICLE_FLOW_RECONSTRUCTOR_ARCHITECTURE,
                "n_parent_candidates": int(parent_tokens.shape[1]),
                "n_extra_candidates": int(extra_tokens.shape[1]),
                "config": self.config.to_dict(),
                "diagnostics": diagnostics,
            },
            aux=aux,
        )


def build_particle_flow_reconstructor(
    config: ParticleFlowReconstructorConfig | Mapping[str, Any] | None = None,
) -> ParticleFlowReconstructor:
    return ParticleFlowReconstructor(ParticleFlowReconstructorConfig.from_mapping(config))


__all__ = [
    "PARTICLE_FLOW_RECONSTRUCTOR_ARCHITECTURE",
    "PARTICLE_FLOW_FEATURE_NAMES",
    "PARTICLE_FLOW_INPUT_FEATURE_DIM",
    "PARTICLE_FLOW_SUMMARY_FEATURE_NAMES",
    "PARTICLE_FLOW_SUMMARY_FEATURE_DIM",
    "ParticleFlowContextBuilder",
    "ParticleFlowEncoder",
    "ParticleFlowEncoderOutput",
    "ParticleFlowReconstructor",
    "ParticleFlowReconstructorConfig",
    "build_particle_flow_reconstructor",
    "build_particle_flow_features",
    "build_particle_flow_summary_features",
    "masked_max_pool",
    "masked_mean_pool",
    "masked_sum_pool",
]

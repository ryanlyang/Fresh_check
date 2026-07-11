"""Residual predictor models for Canonical Multi-Scale Jet State tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .layout import (
    CANONICAL_STATE_TOKEN_FAMILIES,
    CanonicalJetStateLayout,
    default_canonical_jet_state_layout,
)


CANONICAL_STATE_PREDICTOR_CONTRACT = "canonical_state_residual_predictor_v1"

PREDICTOR_VARIANT_GEOMETRY_BIASED = "P0_geometry_biased_decoder"
PREDICTOR_VARIANT_NO_GEOMETRY_BIAS = "P1_no_geometry_bias"
PREDICTOR_VARIANT_DEEPSETS = "P2_deepsets_global_pooled"
PREDICTOR_VARIANT_STATE_ONLY = "P3_state_only"
PREDICTOR_VARIANT_PARTICLE_ONLY_QUERIES = "P4_particle_only_learned_queries"
PREDICTOR_VARIANT_HARD_LOCALITY = "P5_hard_locality"
PREDICTOR_VARIANT_UNCERTAINTY = "P6_uncertainty"
PREDICTOR_VARIANT_NO_STATE_SELF_ATTENTION = "P7_no_state_self_attention"

CANONICAL_STATE_PREDICTOR_VARIANTS: tuple[str, ...] = (
    PREDICTOR_VARIANT_GEOMETRY_BIASED,
    PREDICTOR_VARIANT_NO_GEOMETRY_BIAS,
    PREDICTOR_VARIANT_DEEPSETS,
    PREDICTOR_VARIANT_STATE_ONLY,
    PREDICTOR_VARIANT_PARTICLE_ONLY_QUERIES,
    PREDICTOR_VARIANT_HARD_LOCALITY,
    PREDICTOR_VARIANT_UNCERTAINTY,
    PREDICTOR_VARIANT_NO_STATE_SELF_ATTENTION,
)

_VARIANT_ALIASES = {
    "P0": PREDICTOR_VARIANT_GEOMETRY_BIASED,
    "geometry": PREDICTOR_VARIANT_GEOMETRY_BIASED,
    "geometry_biased": PREDICTOR_VARIANT_GEOMETRY_BIASED,
    "P1": PREDICTOR_VARIANT_NO_GEOMETRY_BIAS,
    "no_geometry": PREDICTOR_VARIANT_NO_GEOMETRY_BIAS,
    "P2": PREDICTOR_VARIANT_DEEPSETS,
    "deepsets": PREDICTOR_VARIANT_DEEPSETS,
    "P3": PREDICTOR_VARIANT_STATE_ONLY,
    "state_only": PREDICTOR_VARIANT_STATE_ONLY,
    "P4": PREDICTOR_VARIANT_PARTICLE_ONLY_QUERIES,
    "particle_only_queries": PREDICTOR_VARIANT_PARTICLE_ONLY_QUERIES,
    "P5": PREDICTOR_VARIANT_HARD_LOCALITY,
    "hard_locality": PREDICTOR_VARIANT_HARD_LOCALITY,
    "P6": PREDICTOR_VARIANT_UNCERTAINTY,
    "uncertainty": PREDICTOR_VARIANT_UNCERTAINTY,
    "P7": PREDICTOR_VARIANT_NO_STATE_SELF_ATTENTION,
    "no_state_self_attention": PREDICTOR_VARIANT_NO_STATE_SELF_ATTENTION,
}


def normalize_predictor_variant(value: str) -> str:
    key = str(value).strip()
    if key in CANONICAL_STATE_PREDICTOR_VARIANTS:
        return key
    if key in _VARIANT_ALIASES:
        return _VARIANT_ALIASES[key]
    raise ValueError(f"unknown canonical state predictor variant {value!r}")


@dataclass(frozen=True)
class CanonicalStateResidualPredictorConfig:
    """Configuration for Step 5 residual predictor models."""

    variant: str = PREDICTOR_VARIANT_GEOMETRY_BIASED
    particle_dim: int = RAW_TOKEN_DIM
    d_model: int = 128
    num_heads: int = 4
    particle_encoder_layers: int = 1
    decoder_layers: int = 3
    mlp_ratio: float = 2.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    max_particles: int = 256
    max_slots: int = 32
    geometry_bias_clip: float = 8.0
    hard_locality_factor: float = 1.5
    zero_init_delta_projection: bool = True
    layout: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        variant = normalize_predictor_variant(self.variant)
        if int(self.particle_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"particle_dim must be RAW_TOKEN_DIM={RAW_TOKEN_DIM}")
        for name in ("d_model", "num_heads", "particle_encoder_layers", "decoder_layers", "max_particles", "max_slots"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.d_model) % int(self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must be in [0, 1)")
        if float(self.geometry_bias_clip) <= 0.0:
            raise ValueError("geometry_bias_clip must be positive")
        if float(self.hard_locality_factor) <= 0.0:
            raise ValueError("hard_locality_factor must be positive")
        if not bool(self.zero_init_delta_projection):
            raise ValueError("zero_init_delta_projection is required for Phi_hlt recovery at initialization")
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "particle_dim", int(self.particle_dim))
        object.__setattr__(self, "d_model", int(self.d_model))
        object.__setattr__(self, "num_heads", int(self.num_heads))
        object.__setattr__(self, "particle_encoder_layers", int(self.particle_encoder_layers))
        object.__setattr__(self, "decoder_layers", int(self.decoder_layers))
        object.__setattr__(self, "mlp_ratio", float(self.mlp_ratio))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "attention_dropout", float(self.attention_dropout))
        object.__setattr__(self, "max_particles", int(self.max_particles))
        object.__setattr__(self, "max_slots", int(self.max_slots))
        object.__setattr__(self, "geometry_bias_clip", float(self.geometry_bias_clip))
        object.__setattr__(self, "hard_locality_factor", float(self.hard_locality_factor))
        object.__setattr__(self, "zero_init_delta_projection", bool(self.zero_init_delta_projection))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = CANONICAL_STATE_PREDICTOR_CONTRACT
        return payload


@dataclass(frozen=True)
class CanonicalStateResidualPredictorOutput:
    delta_phi: torch.Tensor
    phi_pred: torch.Tensor
    diagnostics: dict[str, Any]
    raw_delta: torch.Tensor
    log_sigma: torch.Tensor | None = None


def _as_bool_mask(mask: torch.Tensor | None, tokens: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.isfinite(tokens).all(dim=-1) & (tokens[..., 0] > 0.0)
    return mask.to(device=tokens.device, dtype=torch.bool)


def _wrap_phi_torch(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


class _FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, d_model),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GeometryBiasedCrossAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dropout: float,
        num_families: int,
        geometry_bias_clip: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = int(d_model) // int(num_heads)
        self.geometry_bias_clip = float(geometry_bias_clip)
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(float(dropout))
        self.family_bias_scale = torch.nn.Parameter(torch.ones(num_families, num_heads))

    def forward(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        *,
        particle_mask: torch.Tensor,
        geometry_bias: torch.Tensor | None,
        token_type_ids: torch.Tensor,
        local_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, k_state, _ = state.shape
        particles = memory.shape[1]
        q = self.q_proj(state).reshape(batch, k_state, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).reshape(batch, particles, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).reshape(batch, particles, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhkd,bhnd->bhkn", q, k) / math.sqrt(float(self.head_dim))
        if geometry_bias is not None:
            scales = torch.nn.functional.softplus(self.family_bias_scale[token_type_ids]).transpose(0, 1)
            scaled_bias = geometry_bias[:, None, :, :] * scales[None, :, :, None]
            scores = scores + torch.clamp(scaled_bias, min=-self.geometry_bias_clip, max=0.0)
        valid = particle_mask[:, None, None, :].expand(batch, self.num_heads, k_state, particles)
        if local_mask is not None:
            local = local_mask[:, None, :, :].expand_as(valid)
            local_valid = valid & local
            has_local = local_valid.any(dim=-1, keepdim=True)
            valid = torch.where(has_local, local_valid, valid)
        scores = scores.masked_fill(~valid, -1.0e4)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * valid.to(dtype=attention.dtype)
        attention = attention / torch.clamp(attention.sum(dim=-1, keepdim=True), min=1.0e-8)
        context = torch.einsum("bhkn,bhnd->bhkd", self.dropout(attention), v)
        context = context.transpose(1, 2).reshape(batch, k_state, self.d_model)
        output = self.out_proj(context)
        diagnostics: dict[str, torch.Tensor] = {
            "attention": attention if return_attention else attention.detach(),
            "pre_bias_scores": scores.detach(),
        }
        return output, diagnostics


class _DecoderBlock(torch.nn.Module):
    def __init__(self, config: CanonicalStateResidualPredictorConfig, *, num_families: int) -> None:
        super().__init__()
        d_model = int(config.d_model)
        hidden_dim = int(round(float(config.mlp_ratio) * d_model))
        self.self_attn = torch.nn.MultiheadAttention(
            d_model,
            int(config.num_heads),
            dropout=float(config.attention_dropout),
            batch_first=True,
        )
        self.cross_attn = _GeometryBiasedCrossAttention(
            d_model=d_model,
            num_heads=int(config.num_heads),
            dropout=float(config.attention_dropout),
            num_families=num_families,
            geometry_bias_clip=float(config.geometry_bias_clip),
        )
        self.ffn = _FeedForward(d_model, hidden_dim, float(config.dropout))
        self.norm_self = torch.nn.LayerNorm(d_model)
        self.norm_cross = torch.nn.LayerNorm(d_model)
        self.norm_ffn = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(float(config.dropout))

    def forward(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        *,
        particle_mask: torch.Tensor,
        geometry_bias: torch.Tensor | None,
        token_type_ids: torch.Tensor,
        local_mask: torch.Tensor | None,
        enable_self_attention: bool,
        return_attention: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        diagnostics: dict[str, torch.Tensor] = {}
        if enable_self_attention:
            residual = state
            x = self.norm_self(state)
            self_out, self_weights = self.self_attn(x, x, x, need_weights=return_attention)
            state = residual + self.dropout(self_out)
            if return_attention:
                diagnostics["self_attention"] = self_weights.detach()
        residual = state
        cross_out, cross_diag = self.cross_attn(
            self.norm_cross(state),
            memory,
            particle_mask=particle_mask,
            geometry_bias=geometry_bias,
            token_type_ids=token_type_ids,
            local_mask=local_mask,
            return_attention=return_attention,
        )
        state = residual + self.dropout(cross_out)
        state = state + self.ffn(self.norm_ffn(state))
        diagnostics.update(cross_diag)
        return state, diagnostics


class GeometryBiasedStateResidualDecoder(torch.nn.Module):
    """Step 5 residual predictor over canonical state tokens."""

    def __init__(
        self,
        config: CanonicalStateResidualPredictorConfig | None = None,
        *,
        layout: CanonicalJetStateLayout | None = None,
    ) -> None:
        super().__init__()
        self.config = CanonicalStateResidualPredictorConfig() if config is None else config
        self.layout = default_canonical_jet_state_layout() if layout is None else layout
        d_model = int(self.config.d_model)
        self.particle_projection = torch.nn.Linear(int(self.config.particle_dim), d_model)
        self.particle_rank_embedding = torch.nn.Embedding(int(self.config.max_particles), d_model)
        particle_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(self.config.num_heads),
            dim_feedforward=int(round(float(self.config.mlp_ratio) * d_model)),
            dropout=float(self.config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.particle_encoder = torch.nn.TransformerEncoder(
            particle_layer,
            num_layers=int(self.config.particle_encoder_layers),
        )
        self.state_value_projection = torch.nn.Linear(self.layout.d_phi, d_model)
        self.token_type_embedding = torch.nn.Embedding(len(CANONICAL_STATE_TOKEN_FAMILIES), d_model)
        self.scale_embedding = torch.nn.Embedding(len(CANONICAL_STATE_TOKEN_FAMILIES), d_model)
        self.slot_embedding = torch.nn.Embedding(int(self.config.max_slots), d_model)
        self.geometry_mlp = torch.nn.Sequential(
            torch.nn.Linear(8, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.learned_state_queries = torch.nn.Parameter(torch.zeros(self.layout.k_state, d_model))
        torch.nn.init.normal_(self.learned_state_queries, mean=0.0, std=0.02)
        self.decoder_blocks = torch.nn.ModuleList(
            [
                _DecoderBlock(self.config, num_families=len(CANONICAL_STATE_TOKEN_FAMILIES))
                for _ in range(int(self.config.decoder_layers))
            ]
        )
        self.state_only_layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=int(self.config.num_heads),
                    dim_feedforward=int(round(float(self.config.mlp_ratio) * d_model)),
                    dropout=float(self.config.dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(self.config.decoder_layers))
            ]
        )
        self.deepsets_update = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * d_model),
            torch.nn.Linear(2 * d_model, int(round(float(self.config.mlp_ratio) * d_model))),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(round(float(self.config.mlp_ratio) * d_model)), d_model),
        )
        self.output_norm = torch.nn.LayerNorm(d_model)
        self.delta_hidden = torch.nn.Sequential(
            torch.nn.Linear(d_model, int(round(float(self.config.mlp_ratio) * d_model))),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
        )
        self.delta_out = torch.nn.Linear(int(round(float(self.config.mlp_ratio) * d_model)), self.layout.d_phi)
        self.log_sigma_out = torch.nn.Linear(int(round(float(self.config.mlp_ratio) * d_model)), self.layout.d_phi)
        if bool(self.config.zero_init_delta_projection):
            torch.nn.init.zeros_(self.delta_out.weight)
            torch.nn.init.zeros_(self.delta_out.bias)
            torch.nn.init.zeros_(self.log_sigma_out.weight)
            torch.nn.init.zeros_(self.log_sigma_out.bias)
        self._register_layout_buffers()

    def _register_layout_buffers(self) -> None:
        self.register_buffer(
            "token_type_ids",
            torch.tensor(self.layout.token_type_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("scale_ids", torch.tensor(self.layout.scale_ids, dtype=torch.long), persistent=False)
        slot_ids = [min(int(slot), int(self.config.max_slots) - 1) for slot in self.layout.slot_ids]
        self.register_buffer("slot_ids", torch.tensor(slot_ids, dtype=torch.long), persistent=False)
        self.register_buffer(
            "residual_scales",
            torch.tensor(self.layout.residual_scale_vector(), dtype=torch.float32),
            persistent=False,
        )
        geometry = []
        radial_centers = []
        radial_widths = []
        angular_centers = []
        angular_widths = []
        anchor_radii = []
        anchor_deta = []
        anchor_dphi = []
        for spec in self.layout.token_specs:
            if spec.family == "radial":
                inner = float(spec.radius_inner or 0.0)
                outer = 0.5 if spec.radius_outer is None else float(spec.radius_outer)
                center = 0.5 * (inner + outer)
                width = max(outer - inner, 1.0e-3)
            else:
                center = 0.0
                width = 1.0
            radial_centers.append(center)
            radial_widths.append(width)
            angular_centers.append(0.0 if spec.angular_center is None else float(spec.angular_center))
            angular_widths.append(2.0 * math.pi if spec.angular_width is None else float(spec.angular_width))
            anchor_radii.append(0.0 if spec.anchor_radius is None else float(spec.anchor_radius))
            anchor_deta.append(0.0 if spec.anchor_deta is None else float(spec.anchor_deta))
            anchor_dphi.append(0.0 if spec.anchor_dphi is None else float(spec.anchor_dphi))
            geometry.append(
                [
                    center,
                    width,
                    0.0 if spec.angular_center is None else float(spec.angular_center),
                    0.0 if spec.angular_width is None else float(spec.angular_width),
                    0.0 if spec.anchor_radius is None else float(spec.anchor_radius),
                    0.0 if spec.anchor_deta is None else float(spec.anchor_deta),
                    0.0 if spec.anchor_dphi is None else float(spec.anchor_dphi),
                    float(spec.slot_id),
                ]
            )
        self.register_buffer("state_geometry_features", torch.tensor(geometry, dtype=torch.float32), persistent=False)
        self.register_buffer("radial_centers", torch.tensor(radial_centers, dtype=torch.float32), persistent=False)
        self.register_buffer("radial_widths", torch.tensor(radial_widths, dtype=torch.float32), persistent=False)
        self.register_buffer("angular_centers", torch.tensor(angular_centers, dtype=torch.float32), persistent=False)
        self.register_buffer("angular_widths", torch.tensor(angular_widths, dtype=torch.float32), persistent=False)
        self.register_buffer("anchor_radii", torch.tensor(anchor_radii, dtype=torch.float32), persistent=False)
        self.register_buffer("anchor_deta", torch.tensor(anchor_deta, dtype=torch.float32), persistent=False)
        self.register_buffer("anchor_dphi", torch.tensor(anchor_dphi, dtype=torch.float32), persistent=False)

    @property
    def uses_geometry_bias(self) -> bool:
        return self.config.variant in {
            PREDICTOR_VARIANT_GEOMETRY_BIASED,
            PREDICTOR_VARIANT_HARD_LOCALITY,
            PREDICTOR_VARIANT_UNCERTAINTY,
            PREDICTOR_VARIANT_NO_STATE_SELF_ATTENTION,
        }

    @property
    def uses_hard_locality(self) -> bool:
        return self.config.variant == PREDICTOR_VARIANT_HARD_LOCALITY

    @property
    def uses_uncertainty(self) -> bool:
        return self.config.variant == PREDICTOR_VARIANT_UNCERTAINTY

    @property
    def uses_state_self_attention(self) -> bool:
        return self.config.variant != PREDICTOR_VARIANT_NO_STATE_SELF_ATTENTION

    def _particle_memory(self, particles: torch.Tensor, particle_mask: torch.Tensor) -> torch.Tensor:
        batch, n_particles, _ = particles.shape
        ranks = torch.arange(n_particles, device=particles.device).clamp(max=int(self.config.max_particles) - 1)
        x = self.particle_projection(torch.nan_to_num(particles, nan=0.0, posinf=0.0, neginf=0.0))
        x = x + self.particle_rank_embedding(ranks)[None, :, :]
        padding_mask = ~particle_mask
        all_invalid = padding_mask.all(dim=1)
        if bool(all_invalid.any()):
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False
        encoded = self.particle_encoder(x, src_key_padding_mask=padding_mask)
        return encoded * particle_mask[:, :, None].to(dtype=encoded.dtype)

    def _state_queries(self, phi_hlt: torch.Tensor) -> torch.Tensor:
        batch = phi_hlt.shape[0]
        metadata = (
            self.token_type_embedding(self.token_type_ids)
            + self.scale_embedding(self.scale_ids)
            + self.slot_embedding(self.slot_ids)
            + self.geometry_mlp(self.state_geometry_features)
        )
        if self.config.variant == PREDICTOR_VARIANT_PARTICLE_ONLY_QUERIES:
            value = self.learned_state_queries[None, :, :].expand(batch, -1, -1)
        else:
            value = self.state_value_projection(phi_hlt) + self.learned_state_queries[None, :, :]
        return value + metadata[None, :, :]

    def geometry_bias_and_mask(
        self,
        particles: torch.Tensor,
        particle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_particles, _ = particles.shape
        eta = particles[..., 1]
        phi = _wrap_phi_torch(particles[..., 2])
        radius = torch.sqrt(torch.clamp(eta * eta + phi * phi, min=0.0))
        bias = particles.new_zeros((batch, self.layout.k_state, n_particles))
        local = torch.zeros((batch, self.layout.k_state, n_particles), dtype=torch.bool, device=particles.device)

        for spec in self.layout.token_specs:
            index = int(spec.index)
            if spec.family == "radial":
                center = self.radial_centers[index]
                width = torch.clamp(self.radial_widths[index], min=1.0e-3)
                delta = (radius - center) / width
                bias[:, index, :] = -(delta * delta)
                local[:, index, :] = torch.abs(radius - center) <= float(self.config.hard_locality_factor) * width
            elif spec.family == "angular":
                center = self.angular_centers[index]
                width = torch.clamp(self.angular_widths[index], min=1.0e-3)
                delta = _wrap_phi_torch(phi - center) / width
                bias[:, index, :] = -(delta * delta)
                local[:, index, :] = torch.abs(_wrap_phi_torch(phi - center)) <= float(self.config.hard_locality_factor) * width
            elif spec.family.startswith("anchor_"):
                sigma = torch.clamp(self.anchor_radii[index], min=1.0e-3)
                deta = eta - self.anchor_deta[index]
                dphi = _wrap_phi_torch(phi - self.anchor_dphi[index])
                delta2 = (deta * deta + dphi * dphi) / (sigma * sigma)
                bias[:, index, :] = -delta2
                local[:, index, :] = (delta2 <= float(self.config.hard_locality_factor) ** 2) & particle_mask
            else:
                local[:, index, :] = particle_mask
        bias = torch.clamp(bias, min=-float(self.config.geometry_bias_clip), max=0.0)
        return bias, local

    def _masked_mean(self, memory: torch.Tensor, particle_mask: torch.Tensor) -> torch.Tensor:
        weights = particle_mask.to(dtype=memory.dtype)
        denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
        return (memory * weights[:, :, None]).sum(dim=1) / denom

    def forward(
        self,
        particles: torch.Tensor,
        particle_mask: torch.Tensor | None,
        phi_hlt: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> CanonicalStateResidualPredictorOutput:
        particle_mask = _as_bool_mask(particle_mask, particles)
        phi_hlt = torch.nan_to_num(phi_hlt, nan=0.0, posinf=0.0, neginf=0.0)
        if phi_hlt.shape[1:] != (self.layout.k_state, self.layout.d_phi):
            raise ValueError(f"phi_hlt must have shape [B, {self.layout.k_state}, {self.layout.d_phi}], got {tuple(phi_hlt.shape)}")
        state_mask = (
            torch.ones(phi_hlt.shape[:2], device=phi_hlt.device, dtype=torch.bool)
            if state_mask is None
            else state_mask.to(device=phi_hlt.device, dtype=torch.bool)
        )
        memory = self._particle_memory(particles, particle_mask)
        state = self._state_queries(phi_hlt)
        geometry_bias, local_mask = self.geometry_bias_and_mask(particles, particle_mask)
        geometry_for_attention = geometry_bias if self.uses_geometry_bias else None
        local_for_attention = local_mask if self.uses_hard_locality else None
        attention_snapshot: torch.Tensor | None = None

        if self.config.variant == PREDICTOR_VARIANT_DEEPSETS:
            pooled = self._masked_mean(memory, particle_mask)
            state = state + self.deepsets_update(torch.cat([state, pooled[:, None, :].expand_as(state)], dim=-1))
        elif self.config.variant == PREDICTOR_VARIANT_STATE_ONLY:
            for layer in self.state_only_layers:
                state = layer(state)
        else:
            for block in self.decoder_blocks:
                state, block_diag = block(
                    state,
                    memory,
                    particle_mask=particle_mask,
                    geometry_bias=geometry_for_attention,
                    token_type_ids=self.token_type_ids,
                    local_mask=local_for_attention,
                    enable_self_attention=self.uses_state_self_attention,
                    return_attention=return_attention,
                )
                attention_snapshot = block_diag.get("attention", attention_snapshot)

        hidden = self.delta_hidden(self.output_norm(state))
        raw_delta = self.delta_out(hidden)
        residual_scales = self.residual_scales.to(device=raw_delta.device, dtype=raw_delta.dtype)
        delta_phi = torch.tanh(raw_delta) * residual_scales[None, None, :]
        delta_phi = delta_phi * state_mask[:, :, None].to(dtype=delta_phi.dtype)
        phi_pred = phi_hlt + delta_phi
        log_sigma = None
        if self.uses_uncertainty:
            log_sigma = torch.clamp(self.log_sigma_out(hidden), min=-5.0, max=5.0)

        family_norms: dict[str, float] = {}
        with torch.no_grad():
            for family, (start, end) in self.layout.family_slices().items():
                family_delta = delta_phi[:, start:end, :]
                family_norms[f"delta_norm_mean.{family}"] = float(torch.linalg.vector_norm(family_delta, dim=-1).mean().detach().cpu())
            diagnostics: dict[str, Any] = {
                "contract": CANONICAL_STATE_PREDICTOR_CONTRACT,
                "variant": self.config.variant,
                "geometry_bias_applied": bool(self.uses_geometry_bias),
                "hard_locality_applied": bool(self.uses_hard_locality),
                "state_self_attention_enabled": bool(self.uses_state_self_attention),
                "uncertainty_enabled": bool(self.uses_uncertainty),
                "particle_valid_count_mean": float(particle_mask.sum(dim=1).float().mean().detach().cpu()),
                "state_valid_count_mean": float(state_mask.sum(dim=1).float().mean().detach().cpu()),
                "delta_norm_mean": float(torch.linalg.vector_norm(delta_phi, dim=-1).mean().detach().cpu()),
                "delta_abs_max": float(torch.max(torch.abs(delta_phi)).detach().cpu()),
                "geometry_bias_min": float(torch.min(geometry_bias).detach().cpu()),
                "geometry_bias_max": float(torch.max(geometry_bias).detach().cpu()),
                **family_norms,
            }
            if attention_snapshot is not None:
                invalid_mass = (
                    attention_snapshot * (~particle_mask[:, None, None, :]).to(dtype=attention_snapshot.dtype)
                ).sum(dim=-1)
                diagnostics["invalid_particle_attention_mass_max"] = float(torch.max(invalid_mass).detach().cpu())
                entropy = -torch.sum(
                    attention_snapshot * torch.log(torch.clamp(attention_snapshot, min=1.0e-8)),
                    dim=-1,
                )
                diagnostics["cross_attention_entropy_mean"] = float(entropy.mean().detach().cpu())
                diagnostics["cross_attention_shape"] = list(attention_snapshot.shape)
            if log_sigma is not None:
                diagnostics["log_sigma_mean"] = float(log_sigma.mean().detach().cpu())
                diagnostics["log_sigma_abs_max"] = float(torch.max(torch.abs(log_sigma)).detach().cpu())

        return CanonicalStateResidualPredictorOutput(
            delta_phi=delta_phi,
            phi_pred=phi_pred,
            diagnostics=diagnostics,
            raw_delta=raw_delta,
            log_sigma=log_sigma,
        )


def build_canonical_state_residual_predictor(
    variant: str = PREDICTOR_VARIANT_GEOMETRY_BIASED,
    **kwargs: Any,
) -> GeometryBiasedStateResidualDecoder:
    config = CanonicalStateResidualPredictorConfig(variant=variant, **kwargs)
    return GeometryBiasedStateResidualDecoder(config)

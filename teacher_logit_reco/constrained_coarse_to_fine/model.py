"""ParT-style HLT encoder and constrained hierarchy reconstructor.

Step 3 deliberately stops before particle-slot rendering.  It turns an HLT
particle view into a global accounting prediction and, depending on the B-tier
variant, one or more increasingly fine grids.  B0-B6 allocate every primitive
child field from its parent by construction; B7 predicts independent positive
child totals as the capacity-matched no-consistency control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .constraints import (
    ACCOUNTING_INDEX,
    PID_COUNT_INDICES,
    PID_PT_INDICES,
    PRIMITIVE_TO_FULL_INDICES,
    AccountingAllocationOutput,
    PositiveAccountingParameterization,
    SoftmaxAccountingAllocator,
    assemble_accounting,
    primitive_accounting,
)
from .layout import (
    ACCOUNTING_FIELD_NAMES,
    DERIVED_DIAGNOSTIC_FIELD_NAMES,
    LEVEL_CELL_COUNTS,
    MOMENT_FIELD_NAMES,
    PID_CATEGORY_NAMES,
    PRIMITIVE_ACCOUNTING_FIELD_NAMES,
    HierarchyTargetLayout,
    default_hierarchy_target_layout,
)


COARSE_TO_FINE_RECONSTRUCTOR_CONTRACT = "constrained_coarse_to_fine_global_grid_reconstructor_v1"

B0_GLOBAL_ONLY = "B0_global_only"
B1_GLOBAL_8 = "B1_global_8"
B2_GLOBAL_8_32 = "B2_global_8_32"
B3_FULL_HIERARCHY = "B3_global_8_32_128"
B4_NO_MOMENTS = "B4_no_moments"
B5_NO_COMPOSITION = "B5_no_composition"
B6_NO_COUNTS = "B6_no_counts"
B7_DIRECT_CHILD_TOTALS = "B7_direct_child_totals"

B_TIER_VARIANTS: tuple[str, ...] = (
    B0_GLOBAL_ONLY,
    B1_GLOBAL_8,
    B2_GLOBAL_8_32,
    B3_FULL_HIERARCHY,
    B4_NO_MOMENTS,
    B5_NO_COMPOSITION,
    B6_NO_COUNTS,
    B7_DIRECT_CHILD_TOTALS,
)

_B_VARIANT_ALIASES = {
    "B0": B0_GLOBAL_ONLY,
    "B1": B1_GLOBAL_8,
    "B2": B2_GLOBAL_8_32,
    "B3": B3_FULL_HIERARCHY,
    "B4": B4_NO_MOMENTS,
    "B5": B5_NO_COMPOSITION,
    "B6": B6_NO_COUNTS,
    "B7": B7_DIRECT_CHILD_TOTALS,
    "global": B0_GLOBAL_ONLY,
    "full": B3_FULL_HIERARCHY,
}


def normalize_b_tier_variant(value: str) -> str:
    key = str(value).strip()
    if key in B_TIER_VARIANTS:
        return key
    if key in _B_VARIANT_ALIASES:
        return _B_VARIANT_ALIASES[key]
    raise ValueError(f"unknown B-tier hierarchy variant {value!r}")


@dataclass(frozen=True)
class BTierVariantSpec:
    name: str
    max_grid_level: int
    hard_allocation: bool = True
    use_moments: bool = True
    learn_composition: bool = True
    use_counts: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def b_tier_variant_spec(value: str) -> BTierVariantSpec:
    variant = normalize_b_tier_variant(value)
    if variant == B0_GLOBAL_ONLY:
        return BTierVariantSpec(variant, 0)
    if variant == B1_GLOBAL_8:
        return BTierVariantSpec(variant, 1)
    if variant == B2_GLOBAL_8_32:
        return BTierVariantSpec(variant, 2)
    if variant == B3_FULL_HIERARCHY:
        return BTierVariantSpec(variant, 3)
    if variant == B4_NO_MOMENTS:
        return BTierVariantSpec(variant, 3, use_moments=False)
    if variant == B5_NO_COMPOSITION:
        return BTierVariantSpec(variant, 3, learn_composition=False)
    if variant == B6_NO_COUNTS:
        return BTierVariantSpec(variant, 3, use_counts=False)
    if variant == B7_DIRECT_CHILD_TOTALS:
        return BTierVariantSpec(variant, 3, hard_allocation=False)
    raise AssertionError(f"unhandled variant {variant}")


@dataclass(frozen=True)
class CoarseToFineReconstructorConfig:
    """Architecture and hierarchy settings for the Step 3 model."""

    variant: str = B3_FULL_HIERARCHY
    feature_dim: int = 17
    point_dim: int = 2
    vector_dim: int = 4
    d_model: int = 256
    num_heads: int = 8
    encoder_layers: int = 8
    pool_layers: int = 2
    decoder_layers_per_level: int = 3
    ffn_multiplier: float = 4.0
    pair_hidden_dim: int = 64
    dropout: float = 0.05
    attention_dropout: float = 0.05
    allocation_temperature: float = 1.0
    uncertainty_min: float = -8.0
    uncertainty_max: float = 8.0
    radial_boundary: float = 0.16
    coordinate_extent: float = 0.8

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", normalize_b_tier_variant(self.variant))
        for name in (
            "feature_dim",
            "point_dim",
            "vector_dim",
            "d_model",
            "num_heads",
            "encoder_layers",
            "pool_layers",
            "decoder_layers_per_level",
            "pair_hidden_dim",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if int(self.d_model) % int(self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if float(self.ffn_multiplier) <= 0.0:
            raise ValueError("ffn_multiplier must be positive")
        if float(self.allocation_temperature) <= 0.0:
            raise ValueError("allocation_temperature must be positive")
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if float(self.uncertainty_min) >= float(self.uncertainty_max):
            raise ValueError("uncertainty_min must be smaller than uncertainty_max")
        if float(self.radial_boundary) <= 0.0 or float(self.coordinate_extent) <= 0.0:
            raise ValueError("hierarchy geometry scales must be positive")

    @property
    def variant_spec(self) -> BTierVariantSpec:
        return b_tier_variant_spec(self.variant)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = COARSE_TO_FINE_RECONSTRUCTOR_CONTRACT
        payload["variant_spec"] = self.variant_spec.to_dict()
        return payload


@dataclass(frozen=True)
class HLTEncoderOutput:
    particle_embeddings: torch.Tensor
    jet_embedding: torch.Tensor
    particle_mask: torch.Tensor
    pool_attention: torch.Tensor
    pair_bias: torch.Tensor


@dataclass(frozen=True)
class HierarchyLevelOutput:
    name: str
    level: int
    accounting: torch.Tensor
    cell_tokens: torch.Tensor
    log_sigma: torch.Tensor
    parent_indices: torch.Tensor
    allocation_logits: torch.Tensor
    primitive_fractions: torch.Tensor | None
    hard_allocation: bool

    def parent_closure_error(self, parent_accounting: torch.Tensor) -> torch.Tensor:
        """Return absolute primitive closure errors grouped by parent cell."""

        if int(self.level) == 1:
            grouped = self.accounting.unsqueeze(1)
        else:
            children_per_parent = int(self.accounting.shape[1]) // int(parent_accounting.shape[1])
            grouped = self.accounting.reshape(
                self.accounting.shape[0], parent_accounting.shape[1], children_per_parent, -1
            )
        return (primitive_accounting(grouped).sum(dim=-2) - primitive_accounting(parent_accounting)).abs()


@dataclass(frozen=True)
class CoarseToFineReconstructorOutput:
    variant: str
    global_accounting: torch.Tensor
    global_log_sigma: torch.Tensor
    global_auxiliary: torch.Tensor
    global_auxiliary_names: tuple[str, ...]
    global_token: torch.Tensor
    levels: tuple[HierarchyLevelOutput, ...]
    hlt: HLTEncoderOutput
    supervised_field_mask: torch.Tensor
    diagnostics: Mapping[str, Any]

    def level(self, level: int | str) -> HierarchyLevelOutput:
        if isinstance(level, str):
            normalized = level.lower().replace("_", "")
            if normalized.startswith("level"):
                level = int(normalized[5:])
            else:
                level = int(normalized)
        for output in self.levels:
            if int(output.level) == int(level):
                return output
        raise KeyError(f"hierarchy level {level!r} was not produced by {self.variant}")

    def summary(self) -> dict[str, Any]:
        return {
            "contract": COARSE_TO_FINE_RECONSTRUCTOR_CONTRACT,
            "variant": self.variant,
            "global_accounting_shape": list(self.global_accounting.shape),
            "global_log_sigma_shape": list(self.global_log_sigma.shape),
            "global_auxiliary_shape": list(self.global_auxiliary.shape),
            "global_auxiliary_names": list(self.global_auxiliary_names),
            "particle_embedding_shape": list(self.hlt.particle_embeddings.shape),
            "jet_embedding_shape": list(self.hlt.jet_embedding.shape),
            "levels": [
                {
                    "name": output.name,
                    "level": int(output.level),
                    "accounting_shape": list(output.accounting.shape),
                    "cell_token_shape": list(output.cell_tokens.shape),
                    "hard_allocation": bool(output.hard_allocation),
                }
                for output in self.levels
            ],
            "diagnostics": dict(self.diagnostics),
        }


def _batch_first(value: torch.Tensor, channels: int, *, name: str) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError(f"{name} must be rank 3, got {tuple(value.shape)}")
    if int(value.shape[-1]) == int(channels):
        return value
    if int(value.shape[1]) == int(channels):
        return value.transpose(1, 2).contiguous()
    raise ValueError(f"{name} must contain channel dimension {channels}, got {tuple(value.shape)}")


def _particle_mask(mask: torch.Tensor, batch: int, particles: int) -> torch.Tensor:
    if mask.ndim == 3 and int(mask.shape[1]) == 1:
        mask = mask[:, 0, :]
    elif mask.ndim == 3 and int(mask.shape[-1]) == 1:
        mask = mask[..., 0]
    if tuple(mask.shape) != (batch, particles):
        raise ValueError(f"mask must have shape {(batch, particles)} or one singleton channel, got {tuple(mask.shape)}")
    return mask.to(dtype=torch.bool)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=value.dtype).unsqueeze(-1)
    return (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _accounting_diagnostics(accounting: torch.Tensor, *, epsilon: float = 1.0e-8) -> torch.Tensor:
    """Torch equivalent of Step 1's deterministic accounting diagnostics."""

    if int(accounting.shape[-1]) != len(ACCOUNTING_FIELD_NAMES):
        raise ValueError("accounting has the wrong field dimension")
    total_pt = accounting[..., ACCOUNTING_INDEX["total_pT"]]
    total_count = accounting[..., ACCOUNTING_INDEX["expected_constituent_count"]]
    safe_pt = total_pt.clamp_min(epsilon)
    safe_count = total_count.clamp_min(epsilon)
    axis_deta = (
        accounting[..., ACCOUNTING_INDEX["sum_pT_abs_deta_pos"]]
        - accounting[..., ACCOUNTING_INDEX["sum_pT_abs_deta_neg"]]
    ) / safe_pt
    axis_dphi = (
        accounting[..., ACCOUNTING_INDEX["sum_pT_abs_dphi_pos"]]
        - accounting[..., ACCOUNTING_INDEX["sum_pT_abs_dphi_neg"]]
    ) / safe_pt
    width_eta = (
        accounting[..., ACCOUNTING_INDEX["sum_pT_deta2"]] / safe_pt - axis_deta.square()
    ).clamp_min(0.0)
    width_phi = (
        accounting[..., ACCOUNTING_INDEX["sum_pT_dphi2"]] / safe_pt - axis_dphi.square()
    ).clamp_min(0.0)
    result = torch.stack(
        (
            *(accounting[..., index] / safe_pt for index in PID_PT_INDICES),
            axis_deta,
            axis_dphi,
            width_eta,
            width_phi,
            accounting[..., ACCOUNTING_INDEX["sum_pT_r"]] / safe_pt,
            torch.sqrt((accounting[..., ACCOUNTING_INDEX["sum_pT_r2"]] / safe_pt).clamp_min(0.0)),
            total_pt / safe_count,
            accounting[..., ACCOUNTING_INDEX["total_energy"]] / safe_pt,
        ),
        dim=-1,
    )
    result = torch.where((total_pt > epsilon).unsqueeze(-1), result, torch.zeros_like(result))
    if int(result.shape[-1]) != len(DERIVED_DIAGNOSTIC_FIELD_NAMES):
        raise AssertionError("derived diagnostic tensor does not match its declared layout")
    return result


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _PairwiseBias(nn.Module):
    """Physics-aware ParT-style pair bias from points and four-vectors."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, points: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        deta = points[..., 0].unsqueeze(2) - points[..., 0].unsqueeze(1)
        dphi = torch.remainder(
            points[..., 1].unsqueeze(2) - points[..., 1].unsqueeze(1) + math.pi,
            2.0 * math.pi,
        ) - math.pi
        dr2 = deta.square() + dphi.square()
        dr = torch.sqrt(dr2 + 1.0e-12)
        px, py, pz, energy = vectors.unbind(dim=-1)
        pt = torch.sqrt(px.square() + py.square() + 1.0e-12)
        pt_i = pt.unsqueeze(2)
        pt_j = pt.unsqueeze(1)
        pair_e = energy.unsqueeze(2) + energy.unsqueeze(1)
        pair_px = px.unsqueeze(2) + px.unsqueeze(1)
        pair_py = py.unsqueeze(2) + py.unsqueeze(1)
        pair_pz = pz.unsqueeze(2) + pz.unsqueeze(1)
        mass2 = (pair_e.square() - pair_px.square() - pair_py.square() - pair_pz.square()).clamp_min(0.0)
        minimum_pt = torch.minimum(pt_i, pt_j)
        pt_sum = (pt_i + pt_j).clamp_min(1.0e-8)
        pair_features = torch.stack(
            (
                deta,
                torch.sin(dphi),
                torch.cos(dphi),
                torch.log1p(dr),
                torch.log1p(minimum_pt * dr),
                torch.log1p(torch.sqrt(mass2 + 1.0e-12)),
                minimum_pt / pt_sum,
                torch.tanh(torch.log(pt_i.clamp_min(1.0e-8)) - torch.log(pt_j.clamp_min(1.0e-8))),
            ),
            dim=-1,
        )
        bias = self.net(pair_features).permute(0, 3, 1, 2).contiguous()
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        return bias * pair_mask.to(dtype=bias.dtype)


class _PairBiasedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(d_model) // int(num_heads)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        pair_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, particles, d_model = value.shape
        qkv = self.qkv(value).reshape(batch, particles, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(float(self.head_dim))
        scores = scores + pair_bias
        valid = mask[:, None, :, None] & mask[:, None, None, :]
        scores = scores.masked_fill(~valid, -1.0e4)
        attention = torch.softmax(scores, dim=-1) * valid.to(dtype=scores.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        attended = torch.einsum("bhij,bhjd->bhid", self.dropout(attention), v)
        attended = attended.transpose(1, 2).reshape(batch, particles, d_model)
        return self.out(attended), attention


class _ParticleEncoderBlock(nn.Module):
    def __init__(self, config: CoarseToFineReconstructorConfig) -> None:
        super().__init__()
        hidden = int(round(config.ffn_multiplier * config.d_model))
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.attention = _PairBiasedAttention(config.d_model, config.num_heads, config.attention_dropout)
        self.norm_ffn = nn.LayerNorm(config.d_model)
        self.ffn = _FeedForward(config.d_model, hidden, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        pair_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update, attention = self.attention(self.norm_attention(value), mask, pair_bias)
        value = value + self.dropout(update)
        value = value + self.ffn(self.norm_ffn(value))
        return value * mask.unsqueeze(-1).to(dtype=value.dtype), attention


class _JetPoolBlock(nn.Module):
    def __init__(self, config: CoarseToFineReconstructorConfig) -> None:
        super().__init__()
        hidden = int(round(config.ffn_multiplier * config.d_model))
        self.query_norm = nn.LayerNorm(config.d_model)
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = _FeedForward(config.d_model, hidden, config.dropout)

    def forward(
        self,
        query: torch.Tensor,
        particles: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update, weights = self.attention(
            self.query_norm(query),
            self.memory_norm(particles),
            self.memory_norm(particles),
            key_padding_mask=~mask,
            need_weights=True,
            average_attn_weights=False,
        )
        query = query + update
        query = query + self.ffn(self.ffn_norm(query))
        return query, weights


class ParTStyleHLTEncoder(nn.Module):
    """HLT-only particle encoder with ParT-like learned pair biases."""

    def __init__(self, config: CoarseToFineReconstructorConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_norm = nn.LayerNorm(config.feature_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.pair_bias = _PairwiseBias(config.pair_hidden_dim, config.num_heads)
        self.blocks = nn.ModuleList([_ParticleEncoderBlock(config) for _ in range(config.encoder_layers)])
        self.particle_norm = nn.LayerNorm(config.d_model)
        self.jet_query = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.trunc_normal_(self.jet_query, std=0.02)
        self.pool_blocks = nn.ModuleList([_JetPoolBlock(config) for _ in range(config.pool_layers)])
        self.mean_projection = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.jet_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> HLTEncoderOutput:
        features = _batch_first(features, self.config.feature_dim, name="features").to(dtype=torch.float32)
        points = _batch_first(points, self.config.point_dim, name="points").to(
            device=features.device, dtype=features.dtype
        )
        vectors = _batch_first(lorentz_vectors, self.config.vector_dim, name="lorentz_vectors").to(
            device=features.device, dtype=features.dtype
        )
        batch, particles, _ = features.shape
        if points.shape[:2] != (batch, particles) or vectors.shape[:2] != (batch, particles):
            raise ValueError("points, features, and lorentz_vectors must have aligned batch/particle dimensions")
        original_mask = _particle_mask(mask.to(device=features.device), batch, particles)
        safe_mask = original_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if bool(empty.any()):
            safe_mask[empty, 0] = True
            features = features.clone()
            points = points.clone()
            vectors = vectors.clone()
            features[empty, 0] = 0.0
            points[empty, 0] = 0.0
            vectors[empty, 0] = 0.0
            vectors[empty, 0, 3] = 1.0e-8
        hidden = self.feature_projection(self.feature_norm(features))
        hidden = hidden * safe_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pair_bias = self.pair_bias(points, vectors, safe_mask)
        for block in self.blocks:
            hidden, _ = block(hidden, safe_mask, pair_bias)
        hidden = self.particle_norm(hidden)
        query = self.jet_query.expand(batch, -1, -1)
        pool_attention = hidden.new_zeros(batch, self.config.num_heads, 1, particles)
        for block in self.pool_blocks:
            query, pool_attention = block(query, hidden, safe_mask)
        jet = self.jet_norm(query[:, 0] + self.mean_projection(_masked_mean(hidden, safe_mask)))
        exposed_hidden = hidden * original_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return HLTEncoderOutput(
            particle_embeddings=exposed_hidden,
            jet_embedding=jet,
            particle_mask=original_mask,
            pool_attention=pool_attention,
            pair_bias=pair_bias,
        )


def _hlt_composition_priors(
    features: torch.Tensor,
    vectors: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic HLT pT/count category fractions for B5."""

    pid_start = 6
    if int(features.shape[-1]) < pid_start + len(PID_CATEGORY_NAMES):
        uniform = features.new_full((features.shape[0], len(PID_CATEGORY_NAMES)), 1.0 / len(PID_CATEGORY_NAMES))
        return uniform, uniform
    pid = features[..., pid_start : pid_start + len(PID_CATEGORY_NAMES)].clamp_min(0.0)
    valid = mask.to(dtype=features.dtype).unsqueeze(-1)
    pt = torch.sqrt(vectors[..., 0].square() + vectors[..., 1].square() + epsilon).unsqueeze(-1)
    pt_totals = (pid * pt * valid).sum(dim=1)
    count_totals = (pid * valid).sum(dim=1)
    uniform = torch.full_like(pt_totals, 1.0 / float(len(PID_CATEGORY_NAMES)))
    pt_prior = torch.where(
        pt_totals.sum(dim=-1, keepdim=True) > epsilon,
        pt_totals / pt_totals.sum(dim=-1, keepdim=True).clamp_min(epsilon),
        uniform,
    )
    count_prior = torch.where(
        count_totals.sum(dim=-1, keepdim=True) > epsilon,
        count_totals / count_totals.sum(dim=-1, keepdim=True).clamp_min(epsilon),
        uniform,
    )
    return pt_prior, count_prior


class _AccountingPolicy(nn.Module):
    """Apply B4-B6 channel semantics while preserving one common schema."""

    def __init__(self, spec: BTierVariantSpec) -> None:
        super().__init__()
        self.spec = spec
        primitive_names = tuple(PRIMITIVE_ACCOUNTING_FIELD_NAMES)
        self.primitive_pt_indices = tuple(primitive_names.index(name) for name in primitive_names if name.endswith("_pT"))
        self.primitive_count_indices = tuple(
            primitive_names.index(name) for name in primitive_names if name.endswith("_count")
        )
        self.primitive_moment_indices = tuple(primitive_names.index(name) for name in MOMENT_FIELD_NAMES)

    def global_accounting(
        self,
        raw: torch.Tensor,
        pt_prior: torch.Tensor,
        count_prior: torch.Tensor,
    ) -> torch.Tensor:
        primitive = F.softplus(raw)
        if not self.spec.use_moments:
            primitive = primitive.clone()
            primitive[..., list(self.primitive_moment_indices)] = 0.0
        if not self.spec.use_counts:
            primitive = primitive.clone()
            primitive[..., list(self.primitive_count_indices)] = 0.0
        if not self.spec.learn_composition:
            primitive = primitive.clone()
            predicted_pt_total = primitive[..., list(self.primitive_pt_indices)].sum(dim=-1, keepdim=True)
            primitive[..., list(self.primitive_pt_indices)] = predicted_pt_total * pt_prior
            if self.spec.use_counts:
                predicted_count_total = primitive[..., list(self.primitive_count_indices)].sum(dim=-1, keepdim=True)
                primitive[..., list(self.primitive_count_indices)] = predicted_count_total * count_prior
        return assemble_accounting(primitive)

    def allocation_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.spec.learn_composition:
            return logits
        logits = logits.clone()
        shared_pt = logits[..., list(self.primitive_pt_indices)].mean(dim=-1, keepdim=True)
        logits[..., list(self.primitive_pt_indices)] = shared_pt
        if self.spec.use_counts:
            shared_count = logits[..., list(self.primitive_count_indices)].mean(dim=-1, keepdim=True)
            logits[..., list(self.primitive_count_indices)] = shared_count
        return logits

    def direct_accounting(self, raw: torch.Tensor) -> torch.Tensor:
        return PositiveAccountingParameterization(minimum=0.0)(raw)

    def supervised_field_mask(self, reference: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(len(ACCOUNTING_FIELD_NAMES), dtype=torch.bool, device=reference.device)
        if not self.spec.use_moments:
            mask[list(ACCOUNTING_INDEX[name] for name in MOMENT_FIELD_NAMES)] = False
        if not self.spec.use_counts:
            mask[list(ACCOUNTING_INDEX[name] for name in ACCOUNTING_FIELD_NAMES if name.endswith("_count"))] = False
            mask[ACCOUNTING_INDEX["expected_constituent_count"]] = False
        if not self.spec.learn_composition:
            mask[list(ACCOUNTING_INDEX[name] for name in ACCOUNTING_FIELD_NAMES if name.endswith("_pT") and name != "total_pT")] = False
            mask[list(ACCOUNTING_INDEX[name] for name in ACCOUNTING_FIELD_NAMES if name.endswith("_count"))] = False
        return mask


class _GlobalPredictor(nn.Module):
    def __init__(self, config: CoarseToFineReconstructorConfig) -> None:
        super().__init__()
        hidden = int(round(config.ffn_multiplier * config.d_model))
        self.context = nn.Sequential(
            nn.LayerNorm(3 * config.d_model),
            nn.Linear(3 * config.d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.d_model),
            nn.GELU(),
        )
        self.accounting = nn.Linear(config.d_model, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES))
        self.log_sigma = nn.Linear(config.d_model, len(ACCOUNTING_FIELD_NAMES))

    def forward(self, hlt: HLTEncoderOutput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = _masked_mean(hlt.particle_embeddings, hlt.particle_mask)
        masked = hlt.particle_embeddings.masked_fill(~hlt.particle_mask.unsqueeze(-1), -1.0e4)
        maximum = masked.max(dim=1).values
        maximum = torch.where(hlt.particle_mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        context = self.context(torch.cat((hlt.jet_embedding, mean, maximum), dim=-1))
        return context, self.accounting(context), self.log_sigma(context)


def _geometry_tensor(layout: HierarchyTargetLayout, level: int) -> torch.Tensor:
    rows = layout.cell_geometry(level)
    count = float(len(rows))
    max_parent = float(max(1, LEVEL_CELL_COUNTS[level - 1] - 1))
    extent = float(layout.coordinate_extent)
    radial_scale = math.sqrt(2.0) * extent
    values = []
    for row in rows:
        eta_center = 0.5 * (row["eta_min"] + row["eta_max"])
        phi_center = 0.5 * (row["phi_min"] + row["phi_max"])
        values.append(
            (
                eta_center / extent,
                phi_center / extent,
                (row["eta_max"] - row["eta_min"]) / (2.0 * extent),
                (row["phi_max"] - row["phi_min"]) / (2.0 * extent),
                row["radial_min"] / radial_scale,
                row["radial_max"] / radial_scale,
                float(row["radial_bin"]),
                float(row["eta_bin"]) / max(1.0, float(2**level - 1)),
                float(row["phi_bin"]) / max(1.0, float(2**level - 1)),
                float(row["parent_id"]) / max_parent,
                float(row["cell_id"]) / max(1.0, count - 1.0),
                float(level) / 3.0,
            )
        )
    return torch.tensor(values, dtype=torch.float32)


def _geometry_feasibility(layout: HierarchyTargetLayout, level: int) -> torch.Tensor:
    """Identify rectangle/radial-shell intersections used by physical cells."""

    feasible = []
    for row in layout.cell_geometry(level):
        eta_min, eta_max = float(row["eta_min"]), float(row["eta_max"])
        phi_min, phi_max = float(row["phi_min"]), float(row["phi_max"])
        nearest_eta = 0.0 if eta_min <= 0.0 <= eta_max else min(abs(eta_min), abs(eta_max))
        nearest_phi = 0.0 if phi_min <= 0.0 <= phi_max else min(abs(phi_min), abs(phi_max))
        minimum_radius = math.hypot(nearest_eta, nearest_phi)
        maximum_radius = max(
            math.hypot(eta, phi)
            for eta in (eta_min, eta_max)
            for phi in (phi_min, phi_max)
        )
        feasible.append(
            maximum_radius + 1.0e-12 >= float(row["radial_min"])
            and minimum_radius <= float(row["radial_max"]) + 1.0e-12
        )
    return torch.as_tensor(feasible, dtype=torch.bool)


class _CellQueryBlock(nn.Module):
    def __init__(self, config: CoarseToFineReconstructorConfig) -> None:
        super().__init__()
        hidden = int(round(config.ffn_multiplier * config.d_model))
        kwargs = dict(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.self_attention = nn.MultiheadAttention(**kwargs)
        self.hlt_attention = nn.MultiheadAttention(**kwargs)
        self.ancestor_attention = nn.MultiheadAttention(**kwargs)
        self.norm_self = nn.LayerNorm(config.d_model)
        self.norm_hlt = nn.LayerNorm(config.d_model)
        self.norm_ancestor = nn.LayerNorm(config.d_model)
        self.norm_ffn = nn.LayerNorm(config.d_model)
        self.ffn = _FeedForward(config.d_model, hidden, config.dropout)

    def forward(
        self,
        queries: torch.Tensor,
        hlt_particles: torch.Tensor,
        hlt_mask: torch.Tensor,
        ancestors: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm_self(queries)
        update, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        queries = queries + update
        update, _ = self.hlt_attention(
            self.norm_hlt(queries),
            hlt_particles,
            hlt_particles,
            key_padding_mask=~hlt_mask,
            need_weights=False,
        )
        queries = queries + update
        update, _ = self.ancestor_attention(
            self.norm_ancestor(queries), ancestors, ancestors, need_weights=False
        )
        queries = queries + update
        return queries + self.ffn(self.norm_ffn(queries))


class _GridLevelDecoder(nn.Module):
    def __init__(
        self,
        config: CoarseToFineReconstructorConfig,
        layout: HierarchyTargetLayout,
        level: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.level = int(level)
        self.num_cells = int(LEVEL_CELL_COUNTS[level])
        self.num_parents = int(LEVEL_CELL_COUNTS[level - 1])
        self.children_per_parent = self.num_cells // self.num_parents
        parent_indices = torch.as_tensor(layout.parent_indices(level), dtype=torch.long)
        self.register_buffer("parent_indices", parent_indices, persistent=True)
        self.register_buffer("geometry", _geometry_tensor(layout, level), persistent=True)
        self.register_buffer("feasible_cells", _geometry_feasibility(layout, level), persistent=True)
        self.query = nn.Parameter(torch.zeros(1, self.num_cells, config.d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.geometry_projection = nn.Sequential(
            nn.Linear(int(self.geometry.shape[-1]), config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.parent_accounting_projection = nn.Sequential(
            nn.LayerNorm(len(ACCOUNTING_FIELD_NAMES)),
            nn.Linear(len(ACCOUNTING_FIELD_NAMES), config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.parent_token_projection = nn.Linear(config.d_model, config.d_model)
        self.blocks = nn.ModuleList(
            [_CellQueryBlock(config) for _ in range(config.decoder_layers_per_level)]
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.allocation_head = nn.Linear(config.d_model, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES))
        self.direct_head = nn.Linear(config.d_model, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES))
        self.uncertainty_head = nn.Linear(config.d_model, len(ACCOUNTING_FIELD_NAMES))
        self.allocator = SoftmaxAccountingAllocator(
            self.children_per_parent, temperature=config.allocation_temperature
        )

    def forward(
        self,
        hlt: HLTEncoderOutput,
        parent_accounting: torch.Tensor,
        parent_tokens: torch.Tensor,
        ancestor_tokens: torch.Tensor,
        policy: _AccountingPolicy,
    ) -> HierarchyLevelOutput:
        batch = int(hlt.jet_embedding.shape[0])
        indices = self.parent_indices
        selected_parent_accounting = parent_accounting.index_select(1, indices)
        selected_parent_tokens = parent_tokens.index_select(1, indices)
        queries = self.query.expand(batch, -1, -1)
        queries = queries + self.geometry_projection(self.geometry)[None, :, :]
        queries = queries + self.parent_token_projection(selected_parent_tokens)
        queries = queries + self.parent_accounting_projection(torch.log1p(selected_parent_accounting.clamp_min(0.0)))
        for block in self.blocks:
            queries = block(queries, hlt.particle_embeddings, hlt.particle_mask, ancestor_tokens)
        tokens = self.output_norm(queries)
        raw_logits = policy.allocation_logits(self.allocation_head(tokens))
        grouped_logits = raw_logits.reshape(
            batch,
            self.num_parents,
            self.children_per_parent,
            len(PRIMITIVE_ACCOUNTING_FIELD_NAMES),
        )
        if policy.spec.hard_allocation:
            grouped_feasible = self.feasible_cells.reshape(
                self.num_parents, self.children_per_parent
            )
            parent_has_feasible_child = grouped_feasible.any(dim=-1, keepdim=True)
            allocation_mask = grouped_feasible | ~parent_has_feasible_child
            grouped_logits = grouped_logits.masked_fill(
                ~allocation_mask[None, :, :, None],
                float("-inf"),
            )
            allocated: AccountingAllocationOutput = self.allocator(parent_accounting, grouped_logits)
            accounting = allocated.children.reshape(batch, self.num_cells, len(ACCOUNTING_FIELD_NAMES))
            fractions = allocated.primitive_fractions.reshape(
                batch, self.num_cells, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES)
            )
        else:
            accounting = policy.direct_accounting(self.direct_head(tokens))
            fractions = None
        log_sigma = self.uncertainty_head(tokens).clamp(
            min=self.config.uncertainty_min, max=self.config.uncertainty_max
        )
        return HierarchyLevelOutput(
            name=f"level{self.level}",
            level=self.level,
            accounting=accounting,
            cell_tokens=tokens,
            log_sigma=log_sigma,
            parent_indices=indices,
            allocation_logits=raw_logits,
            primitive_fractions=fractions,
            hard_allocation=policy.spec.hard_allocation,
        )


class ConstrainedCoarseToFineReconstructor(nn.Module):
    """Step 3 HLT encoder plus global/grid reconstruction hierarchy."""

    def __init__(
        self,
        config: CoarseToFineReconstructorConfig | Mapping[str, Any] | None = None,
        *,
        layout: HierarchyTargetLayout | None = None,
        hlt_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = CoarseToFineReconstructorConfig()
        elif not isinstance(config, CoarseToFineReconstructorConfig):
            config = CoarseToFineReconstructorConfig(**dict(config))
        self.config = config
        self.spec = config.variant_spec
        self.layout = layout or default_hierarchy_target_layout(
            radial_boundary=config.radial_boundary,
            coordinate_extent=config.coordinate_extent,
        )
        self.hlt_encoder = hlt_encoder or ParTStyleHLTEncoder(config)
        self.global_predictor = _GlobalPredictor(config)
        self.policy = _AccountingPolicy(self.spec)
        self.level_decoders = nn.ModuleList(
            [_GridLevelDecoder(config, self.layout, level) for level in range(1, self.spec.max_grid_level + 1)]
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> CoarseToFineReconstructorOutput:
        hlt: HLTEncoderOutput = self.hlt_encoder(points, features, lorentz_vectors, mask)
        feature_rows = _batch_first(features, self.config.feature_dim, name="features").to(
            device=hlt.particle_embeddings.device, dtype=hlt.particle_embeddings.dtype
        )
        vector_rows = _batch_first(lorentz_vectors, self.config.vector_dim, name="lorentz_vectors").to(
            device=hlt.particle_embeddings.device, dtype=hlt.particle_embeddings.dtype
        )
        pt_prior, count_prior = _hlt_composition_priors(
            feature_rows, vector_rows, hlt.particle_mask
        )
        global_token, raw_global, raw_log_sigma = self.global_predictor(hlt)
        global_accounting = self.policy.global_accounting(raw_global, pt_prior, count_prior)
        global_log_sigma = raw_log_sigma.clamp(
            min=self.config.uncertainty_min, max=self.config.uncertainty_max
        )
        global_auxiliary = _accounting_diagnostics(global_accounting)
        parent_accounting = global_accounting.unsqueeze(1)
        parent_tokens = global_token.unsqueeze(1)
        ancestor_tokens = parent_tokens
        levels: list[HierarchyLevelOutput] = []
        for decoder in self.level_decoders:
            level_output = decoder(
                hlt,
                parent_accounting,
                parent_tokens,
                ancestor_tokens,
                self.policy,
            )
            levels.append(level_output)
            parent_accounting = level_output.accounting
            parent_tokens = level_output.cell_tokens
            ancestor_tokens = torch.cat((ancestor_tokens, parent_tokens), dim=1)
        supervised_mask = self.policy.supervised_field_mask(global_accounting)
        closure_errors = []
        parent = global_accounting.unsqueeze(1)
        for level in levels:
            closure_errors.append(level.parent_closure_error(parent).amax().detach())
            parent = level.accounting
        diagnostics: dict[str, Any] = {
            "contract": COARSE_TO_FINE_RECONSTRUCTOR_CONTRACT,
            "variant": self.config.variant,
            "max_grid_level": int(self.spec.max_grid_level),
            "hard_allocation": bool(self.spec.hard_allocation),
            "use_moments": bool(self.spec.use_moments),
            "learn_composition": bool(self.spec.learn_composition),
            "use_counts": bool(self.spec.use_counts),
            "hlt_only_inputs": True,
            "composition_ablation_policy": (
                "learned_total_with_deterministic_hlt_category_priors_and_shared_spatial_allocations"
                if not self.spec.learn_composition
                else "learned_per_category"
            ),
            "closure_abs_max": (
                torch.stack(closure_errors).amax() if closure_errors else global_accounting.new_tensor(0.0)
            ),
            "valid_hlt_particles_mean": hlt.particle_mask.sum(dim=1).float().mean().detach(),
        }
        return CoarseToFineReconstructorOutput(
            variant=self.config.variant,
            global_accounting=global_accounting,
            global_log_sigma=global_log_sigma,
            global_auxiliary=global_auxiliary,
            global_auxiliary_names=tuple(DERIVED_DIAGNOSTIC_FIELD_NAMES),
            global_token=global_token,
            levels=tuple(levels),
            hlt=hlt,
            supervised_field_mask=supervised_mask,
            diagnostics=diagnostics,
        )

    def load_hlt_encoder_warm_start(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Load an HLT-encoder checkpoint while accepting common wrapper prefixes.

        This supports warm starts from a separately trained copy of this
        ParT-style encoder. Exact Weaver ParT checkpoints use a different block
        implementation and should be converted explicitly rather than silently
        partially loaded.
        """

        cleaned: dict[str, torch.Tensor] = {}
        prefixes = ("module.hlt_encoder.", "hlt_encoder.", "module.")
        for key, value in state_dict.items():
            normalized = str(key)
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    break
            cleaned[normalized] = value
        result = self.hlt_encoder.load_state_dict(cleaned, strict=strict)
        loaded = sorted(set(self.hlt_encoder.state_dict()) - set(result.missing_keys))
        if not loaded:
            raise ValueError("warm-start state did not match any HLT encoder parameters")
        return {
            "loaded_keys": loaded,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "strict": bool(strict),
        }


def build_coarse_to_fine_reconstructor(
    config: CoarseToFineReconstructorConfig | Mapping[str, Any] | None = None,
    *,
    layout: HierarchyTargetLayout | None = None,
    hlt_encoder: nn.Module | None = None,
) -> ConstrainedCoarseToFineReconstructor:
    return ConstrainedCoarseToFineReconstructor(config, layout=layout, hlt_encoder=hlt_encoder)

"""Multi-scale soft-subjet token construction.

Step 5 turns particle-to-subjet assignment weights into actual subjet tokens.
The module keeps the exact physics summaries raw-order based, then learns a
compact token representation with scale embeddings for later subjet-level
transformer and particle readback stages.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .assignment import (
    SoftSubjetAssignment,
    SoftSubjetAssignmentConfig,
    SoftSubjetAssignmentOutput,
    normalize_soft_assignment_config,
)
from .features import (
    CANONICAL_PART_FEATURE_NAMES,
    MultiscaleSubjetFeatureConfig,
    build_canonical_part_inputs,
    build_prepared_subjet_inputs,
    pairwise_delta_r,
)
from .seeds import SubjetSeedOutput


try:  # Keep package imports cheap on machines without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


MULTISCALE_SUBJET_TOKEN_BUILDER_CONTRACT = "multiscale_subjet_token_builder_v1"
MULTISCALE_SUBJET_TOKEN_BUILDER_STEP = "multiscale_subjet_part_step5_token_builder"

MULTISCALE_SUBJET_FOUR_VECTOR_NAMES = ("px", "py", "pz", "energy")
MULTISCALE_SUBJET_FOUR_VECTOR_FEATURE_NAMES = (
    "soft_log_pt",
    "soft_eta",
    "soft_phi",
    "soft_log_energy",
    "soft_log_mass",
    "soft_pt_fraction",
    "soft_mass_over_pt",
)
MULTISCALE_SUBJET_PAIR_OBSERVABLE_NAMES = (
    "internal_pair_weight_sum",
    "internal_delta_r_mean",
    "internal_delta_r_rms",
    "internal_log_pair_mass_mean",
    "internal_log_relative_kt_mean",
    "internal_z_mean",
)


@dataclass(frozen=True)
class MultiScaleSubjetTokenBuilderConfig:
    """Configuration for the Step 5 multi-scale soft-subjet token builder."""

    assignment_config: SoftSubjetAssignmentConfig | Mapping[str, Any] | None = None
    token_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.05
    use_scale_embedding: bool = True
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        assignment_config = normalize_soft_assignment_config(self.assignment_config)
        token_dim = int(self.token_dim)
        hidden_dim = int(self.hidden_dim)
        dropout = float(self.dropout)
        eps = float(self.eps)
        if token_dim <= 0 or hidden_dim <= 0:
            raise ValueError("token_dim and hidden_dim must be positive")
        if not math.isfinite(dropout) or dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be finite and satisfy 0 <= dropout < 1")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        object.__setattr__(self, "assignment_config", assignment_config)
        object.__setattr__(self, "token_dim", token_dim)
        object.__setattr__(self, "hidden_dim", hidden_dim)
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "use_scale_embedding", bool(self.use_scale_embedding))
        object.__setattr__(self, "eps", eps)

    @property
    def total_num_subjets(self) -> int:
        return int(self.assignment_config.total_num_subjets)

    @property
    def num_scales(self) -> int:
        return int(self.assignment_config.num_scales)


@dataclass(frozen=True)
class MultiScaleSubjetTokenBuilderOutput:
    """Learned subjet tokens plus explicit physics summaries.

    ``assignment_weights`` are per-subjet attention distributions over
    particles.  ``cluster_weights`` are scale-wise fuzzy partitions of particles
    across subjets; physics summaries use these cluster weights so four-vectors
    behave like soft cluster sums rather than averaged particle proxies.
    """

    subjet_tokens: Any
    subjet_mask: Any
    assignment_weights: Any
    cluster_weights: Any
    assignment_logits: Any
    estimated_centers: Any
    estimated_pt_fraction: Any
    cluster_pt_fraction: Any
    soft_four_vectors: Any
    soft_four_vector_features: Any
    soft_pair_observable_summaries: Any
    scale_index: Any
    scale_radius: Any
    assignment_output: SoftSubjetAssignmentOutput
    diagnostics: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "contract": MULTISCALE_SUBJET_TOKEN_BUILDER_CONTRACT,
            "step": MULTISCALE_SUBJET_TOKEN_BUILDER_STEP,
            "subjet_tokens_shape": list(self.subjet_tokens.shape),
            "subjet_mask_shape": list(self.subjet_mask.shape),
            "assignment_weights_shape": list(self.assignment_weights.shape),
            "cluster_weights_shape": list(self.cluster_weights.shape),
            "estimated_centers_shape": list(self.estimated_centers.shape),
            "soft_four_vectors_shape": list(self.soft_four_vectors.shape),
            "soft_four_vector_feature_names": list(MULTISCALE_SUBJET_FOUR_VECTOR_FEATURE_NAMES),
            "soft_pair_observable_names": list(MULTISCALE_SUBJET_PAIR_OBSERVABLE_NAMES),
            "diagnostics": dict(self.diagnostics),
        }


def normalize_subjet_token_builder_config(
    config: MultiScaleSubjetTokenBuilderConfig | Mapping[str, Any] | None = None,
) -> MultiScaleSubjetTokenBuilderConfig:
    if config is None:
        return MultiScaleSubjetTokenBuilderConfig()
    if isinstance(config, MultiScaleSubjetTokenBuilderConfig):
        return config
    return MultiScaleSubjetTokenBuilderConfig(**dict(config))


def _masked_mean(value: Any, mask: Any) -> float:
    torch = require_torch()
    weight = mask.to(dtype=value.dtype)
    denom = torch.clamp(weight.sum(), min=1.0)
    return float((value * weight).sum().detach().cpu().item() / float(denom.detach().cpu().item()))


def _soft_four_vector_features(soft_four_vectors: Any, estimated_pt_fraction: Any, eps: float) -> Any:
    torch = require_torch()
    soft_four_vectors = torch.nan_to_num(soft_four_vectors.float(), nan=0.0, posinf=0.0, neginf=0.0)
    estimated_pt_fraction = torch.nan_to_num(estimated_pt_fraction.float(), nan=0.0, posinf=0.0, neginf=0.0)
    px = soft_four_vectors[:, :, 0]
    py = soft_four_vectors[:, :, 1]
    pz = soft_four_vectors[:, :, 2]
    energy = torch.clamp(soft_four_vectors[:, :, 3], min=0.0)
    pt = torch.sqrt(torch.clamp(px * px + py * py, min=0.0))
    p2 = px * px + py * py + pz * pz
    mass2 = torch.clamp(energy * energy - p2, min=0.0)
    mass = torch.sqrt(mass2)
    eta = torch.asinh(pz / torch.clamp(pt, min=float(eps)))
    eta = torch.where(pt > float(eps), eta, torch.zeros_like(eta))
    eta = torch.clamp(eta, -5.0, 5.0)
    phi = torch.atan2(py, px)
    phi = torch.where(pt > float(eps), phi, torch.zeros_like(phi))
    mass_over_pt = torch.where(pt > float(eps), mass / torch.clamp(pt, min=float(eps)), torch.zeros_like(mass))
    mass_over_pt = torch.clamp(mass_over_pt, min=0.0, max=10.0)
    return torch.stack(
        [
            torch.log1p(pt),
            eta,
            phi,
            torch.log1p(energy),
            torch.log1p(mass),
            estimated_pt_fraction,
            mass_over_pt,
        ],
        dim=-1,
    )


def _particle_pair_observables(prepared: Any, eps: float) -> Any:
    torch = require_torch()
    distances = pairwise_delta_r(prepared.coordinates)
    pt_i = prepared.pt[:, :, None]
    pt_j = prepared.pt[:, None, :]
    px_i = prepared.px[:, :, None]
    px_j = prepared.px[:, None, :]
    py_i = prepared.py[:, :, None]
    py_j = prepared.py[:, None, :]
    pz_i = prepared.pz[:, :, None]
    pz_j = prepared.pz[:, None, :]
    energy_i = prepared.energy[:, :, None]
    energy_j = prepared.energy[:, None, :]
    pair_px = px_i + px_j
    pair_py = py_i + py_j
    pair_pz = pz_i + pz_j
    pair_energy = energy_i + energy_j
    pair_mass2 = torch.clamp(
        pair_energy * pair_energy - pair_px * pair_px - pair_py * pair_py - pair_pz * pair_pz,
        min=0.0,
    )
    pair_mass = torch.sqrt(pair_mass2)
    min_pt = torch.minimum(pt_i, pt_j)
    relative_kt = min_pt * distances
    z = min_pt / torch.clamp(pt_i + pt_j, min=float(eps))
    return torch.stack(
        [
            distances,
            distances * distances,
            torch.log1p(pair_mass),
            torch.log1p(relative_kt),
            z,
        ],
        dim=-1,
    )


def _soft_pair_summaries(weights: Any, prepared: Any, subjet_mask: Any, eps: float) -> Any:
    torch = require_torch()
    batch_size, _num_subjets, num_particles = weights.shape
    if int(num_particles) <= 1:
        return weights.new_zeros((batch_size, int(weights.shape[1]), len(MULTISCALE_SUBJET_PAIR_OBSERVABLE_NAMES)))
    pair_observables = _particle_pair_observables(prepared, eps)
    upper = torch.triu(torch.ones((num_particles, num_particles), dtype=torch.bool, device=weights.device), diagonal=1)
    valid_pairs = (prepared.mask[:, :, None] & prepared.mask[:, None, :] & upper[None, :, :]).to(dtype=weights.dtype)
    pair_weights = weights[:, :, :, None] * weights[:, :, None, :] * valid_pairs[:, None, :, :]
    pair_weight_sum = pair_weights.sum(dim=(2, 3))
    summary_values = (
        pair_weights[:, :, :, :, None] * pair_observables[:, None, :, :, :]
    ).sum(dim=(2, 3)) / torch.clamp(pair_weight_sum[:, :, None], min=float(eps))
    pair_summaries = torch.cat([pair_weight_sum[:, :, None], summary_values], dim=-1)
    return torch.where(subjet_mask[:, :, None], pair_summaries, torch.zeros_like(pair_summaries))


def _weighted_center_delta_r_rms(weights: Any, centers: Any, prepared: Any, subjet_mask: Any, eps: float) -> Any:
    torch = require_torch()
    delta_eta = centers[:, :, None, 0] - prepared.coordinates[:, None, :, 0]
    delta_phi = torch.atan2(
        torch.sin(centers[:, :, None, 1] - prepared.coordinates[:, None, :, 1]),
        torch.cos(centers[:, :, None, 1] - prepared.coordinates[:, None, :, 1]),
    )
    delta_r2 = delta_eta * delta_eta + delta_phi * delta_phi
    weighted = weights * prepared.mask[:, None, :].to(dtype=weights.dtype)
    denom = torch.clamp(weighted.sum(dim=-1), min=float(eps))
    rms = torch.sqrt(torch.clamp((weighted * delta_r2).sum(dim=-1) / denom, min=0.0))
    return torch.where(subjet_mask, rms, torch.zeros_like(rms))


class MultiScaleSubjetTokenBuilder(_ModuleBase):
    """Build small/medium/large learned soft-subjet tokens from HLT particles."""

    def __init__(self, config: MultiScaleSubjetTokenBuilderConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = normalize_subjet_token_builder_config(config)
        self.assignment = SoftSubjetAssignment(self.config.assignment_config)
        input_dim = (
            RAW_TOKEN_DIM
            + len(CANONICAL_PART_FEATURE_NAMES)
            + 2
            + 1
            + len(MULTISCALE_SUBJET_FOUR_VECTOR_FEATURE_NAMES)
            + len(MULTISCALE_SUBJET_PAIR_OBSERVABLE_NAMES)
            + 1
        )
        self.input_dim = int(input_dim)
        self.token_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim),
            torch.nn.Linear(self.input_dim, int(self.config.hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(self.config.hidden_dim), int(self.config.token_dim)),
        )
        self.scale_embedding = torch.nn.Embedding(int(self.config.num_scales), int(self.config.token_dim))
        self.output_norm = torch.nn.LayerNorm(int(self.config.token_dim))
        torch.nn.init.normal_(self.scale_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: Any,
        mask: Any,
        *,
        seed_output: SubjetSeedOutput | None = None,
        particle_assignment_mask: Any | None = None,
        subjet_assignment_mask: Any | None = None,
        query_mode: str | None = None,
        feature_config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None = None,
    ) -> MultiScaleSubjetTokenBuilderOutput:
        torch = require_torch()
        prepared = build_prepared_subjet_inputs(tokens, mask, config=feature_config)
        canonical = build_canonical_part_inputs(prepared.tokens, prepared.mask, max_constits=prepared.tokens.shape[1])
        assignment_output = self.assignment(
            prepared.tokens,
            prepared.mask,
            seed_output=seed_output,
            particle_assignment_mask=particle_assignment_mask,
            subjet_assignment_mask=subjet_assignment_mask,
            query_mode=query_mode,
            feature_config=prepared.config,
            prepared_inputs=prepared,
            canonical_inputs=canonical,
        )
        weights = assignment_output.assignment_weights.float()
        cluster_weights = assignment_output.cluster_weights.float()
        subjet_mask = assignment_output.subjet_mask
        raw_pool = torch.einsum("bmn,bnf->bmf", weights, prepared.tokens.float())
        canonical_pool = torch.einsum("bmn,bnf->bmf", weights, canonical.feature_rows().float())
        particle_four_vectors = torch.stack(
            [prepared.px.float(), prepared.py.float(), prepared.pz.float(), prepared.energy.float()],
            dim=-1,
        )
        soft_four_vectors = torch.einsum("bmn,bnf->bmf", cluster_weights, particle_four_vectors)
        cluster_pt_fraction = torch.einsum("bmn,bn->bm", cluster_weights, prepared.pt_fraction.float())
        attention_radius = _weighted_center_delta_r_rms(
            weights,
            assignment_output.estimated_centers,
            prepared,
            subjet_mask,
            float(self.config.eps),
        )
        cluster_radius = _weighted_center_delta_r_rms(
            cluster_weights,
            assignment_output.estimated_centers,
            prepared,
            subjet_mask,
            float(self.config.eps),
        )
        soft_four_features = _soft_four_vector_features(
            soft_four_vectors,
            cluster_pt_fraction,
            float(self.config.eps),
        )
        pair_summaries = _soft_pair_summaries(cluster_weights, prepared, subjet_mask, float(self.config.eps))
        scale_radius = assignment_output.scale_radius.to(device=prepared.tokens.device, dtype=torch.float32)
        token_inputs = torch.cat(
            [
                raw_pool,
                canonical_pool,
                assignment_output.estimated_centers,
                assignment_output.estimated_pt_fraction[:, :, None],
                soft_four_features,
                pair_summaries,
                scale_radius[None, :, None].expand(prepared.tokens.shape[0], -1, -1),
            ],
            dim=-1,
        )
        if int(token_inputs.shape[-1]) != self.input_dim:
            raise RuntimeError(f"token input dim {int(token_inputs.shape[-1])} does not match expected {self.input_dim}")
        token_inputs = torch.nan_to_num(token_inputs.float(), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        token_inputs = torch.clamp(token_inputs, min=-1.0e6, max=1.0e6)
        autocast_context = torch.cuda.amp.autocast(enabled=False) if token_inputs.is_cuda else contextlib.nullcontext()
        with autocast_context:
            projected_tokens = self.token_projection(token_inputs.float())
            if bool(self.config.use_scale_embedding):
                scale_embedding = self.scale_embedding(assignment_output.scale_index.to(device=prepared.tokens.device))[None, :, :].float()
            else:
                scale_embedding = projected_tokens.new_zeros(projected_tokens.shape)
            subjet_tokens = self.output_norm(projected_tokens + scale_embedding)
        subjet_tokens = torch.where(subjet_mask[:, :, None], subjet_tokens, torch.zeros_like(subjet_tokens))
        cluster_pt_fraction = torch.where(subjet_mask, cluster_pt_fraction, torch.zeros_like(cluster_pt_fraction))
        soft_four_vectors = torch.where(subjet_mask[:, :, None], soft_four_vectors, torch.zeros_like(soft_four_vectors))
        soft_four_features = torch.where(subjet_mask[:, :, None], soft_four_features, torch.zeros_like(soft_four_features))
        pair_summaries = torch.where(subjet_mask[:, :, None], pair_summaries, torch.zeros_like(pair_summaries))
        for name, value in (
            ("subjet_tokens", subjet_tokens),
            ("cluster_weights", cluster_weights),
            ("soft_four_vectors", soft_four_vectors),
            ("soft_four_vector_features", soft_four_features),
            ("soft_pair_observable_summaries", pair_summaries),
        ):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"{name} contains non-finite values")

        valid = subjet_mask
        token_norm = subjet_tokens.norm(dim=-1)
        diagnostics = {
            "step": MULTISCALE_SUBJET_TOKEN_BUILDER_STEP,
            "contract": MULTISCALE_SUBJET_TOKEN_BUILDER_CONTRACT,
            "assignment_contract": assignment_output.diagnostics.get("contract"),
            "query_mode": assignment_output.query_mode,
            "use_token_scale_embedding": bool(self.config.use_scale_embedding),
            "valid_subjet_fraction": float(valid.float().mean().detach().cpu().item()),
            "subjet_token_norm_mean": _masked_mean(token_norm, valid),
            "attention_pt_fraction_mean": _masked_mean(assignment_output.estimated_pt_fraction, valid),
            "cluster_pt_fraction_mean": _masked_mean(cluster_pt_fraction, valid),
            "attention_cluster_pt_fraction_abs_diff_mean": _masked_mean(
                torch.abs(assignment_output.estimated_pt_fraction - cluster_pt_fraction),
                valid,
            ),
            "attention_radius_mean": _masked_mean(attention_radius, valid),
            "cluster_radius_mean": _masked_mean(cluster_radius, valid),
            "attention_cluster_radius_abs_diff_mean": _masked_mean(torch.abs(attention_radius - cluster_radius), valid),
            "soft_log_pt_mean": _masked_mean(soft_four_features[:, :, 0], valid),
            "soft_log_mass_mean": _masked_mean(soft_four_features[:, :, 4], valid),
            "internal_pair_weight_sum_mean": _masked_mean(pair_summaries[:, :, 0], valid),
            "internal_delta_r_mean": _masked_mean(pair_summaries[:, :, 1], valid),
            "assignment_entropy_mean": assignment_output.diagnostics.get("entropy_mean"),
            "assignment_dead_token_fraction": assignment_output.diagnostics.get("dead_token_fraction"),
        }
        return MultiScaleSubjetTokenBuilderOutput(
            subjet_tokens=subjet_tokens,
            subjet_mask=subjet_mask,
            assignment_weights=weights,
            cluster_weights=cluster_weights,
            assignment_logits=assignment_output.logits,
            estimated_centers=assignment_output.estimated_centers,
            estimated_pt_fraction=assignment_output.estimated_pt_fraction,
            cluster_pt_fraction=cluster_pt_fraction,
            soft_four_vectors=soft_four_vectors,
            soft_four_vector_features=soft_four_features,
            soft_pair_observable_summaries=pair_summaries,
            scale_index=assignment_output.scale_index,
            scale_radius=assignment_output.scale_radius,
            assignment_output=assignment_output,
            diagnostics=diagnostics,
        )


MultiscaleSubjetTokenBuilderConfig = MultiScaleSubjetTokenBuilderConfig
MultiscaleSubjetTokenBuilderOutput = MultiScaleSubjetTokenBuilderOutput
MultiscaleSubjetTokenBuilder = MultiScaleSubjetTokenBuilder

"""HLT-only hierarchical particle-query predictor for privileged views.

The module deliberately owns every geometric and pooling operation used by
the predictor.  In particular, physical four-vectors are never multiplied by
the normalized transverse-momentum weights used for embedding aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from .contracts import canonical_sha256
from .target_generator import masked_particle_mean_center, wrap_delta_phi


PARTICLE_VIEW_PREDICTOR_CONFIG_CONTRACT = "particle_view_predictor_config_v1"
PARTICLE_VIEW_PREDICTOR_RESOURCE_CONTRACT = "particle_view_predictor_resource_v1"
PARTICLE_VIEW_FLOP_COUNTER = "particle_view_flops_v1"
PARTICLE_VIEW_FLOP_FIXTURE = "flop_fixture_v1"
PARTICLE_VIEW_FLOP_FIXTURE_SEED = 44_017
PVA3_CANONICAL_ARCHITECTURE = "P_HIER_DECODER_REFINE"
PVA3_CANONICAL_NAME = "PVA3_hierarchical_particle_query_decoder"

PARTICLE_PAIR_FEATURE_ORDER = (
    "delta_eta",
    "delta_phi_wrapped",
    "log1p_delta_r",
    "log_pt_ratio",
    "log1p_pair_mass",
    "cos_delta_phi",
)

PARTICLE_VIEW_PREDICTOR_ARCHITECTURES = (
    "P_C0_PARTICLE",
    "P_PART_BASIC",
    "P_LOCAL",
    "P_LOCAL_GLOBAL",
    "P_HIER_NO_DECODER",
    "P_HIER_DECODER",
    "P_HIER_DECODER_REFINE",
    "P_NO_PAIR_BIAS",
    "P_NO_REFINEMENT",
    "P_REGIONS_8_4",
    "P_REGIONS_16_8_4",
    "P_REGIONS_16_8_4_2",
    "P_NO_BALANCE",
    "P_WIDTH128",
    "P_WIDTH192",
    "P_WIDTH256",
    "P_SHARED_CONSUMER_STEM",
)

NONSELECTABLE_PREDICTOR_ARCHITECTURES = frozenset({"P_SHARED_CONSUMER_STEM"})


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ParticleViewPredictorConfig:
    """Immutable architecture contract for the canonical model and controls."""

    view_dim: int
    input_dim: int = 17
    architecture_id: str = PVA3_CANONICAL_ARCHITECTURE
    width: int = 192
    num_heads: int = 8
    local_blocks: int = 3
    global_region_blocks: int = 4
    particle_global_blocks: int = 0
    decoder_blocks: int = 2
    final_refinement_blocks: int = 1
    hierarchy_sizes: tuple[int, ...] = (16, 8, 4)
    use_hierarchy: bool = True
    decoder_mode: str = "cross_attention"
    use_pair_bias: bool = True
    use_balance_loss: bool = True
    balance_weight: float = 0.01
    ffn_expansion: int = 4
    dropout: float = 0.05
    embedding_epsilon: float = 1.0e-8
    empty_occupancy_threshold: float = 1.0e-3
    empty_weight_threshold: float = 1.0e-5
    log_variance_min: float = -6.0
    log_variance_max: float = 3.0
    predict_trust: bool = True
    shared_consumer_stem: bool = False
    contract: str = PARTICLE_VIEW_PREDICTOR_CONFIG_CONTRACT

    def __post_init__(self) -> None:
        if self.view_dim not in {1, 2, 4, 8}:
            raise ValueError("view_dim must be one of 1, 2, 4, or 8")
        for name in (
            "input_dim",
            "width",
            "num_heads",
            "ffn_expansion",
        ):
            _positive_int(name, getattr(self, name))
        for name in (
            "local_blocks",
            "global_region_blocks",
            "particle_global_blocks",
            "decoder_blocks",
            "final_refinement_blocks",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.architecture_id not in PARTICLE_VIEW_PREDICTOR_ARCHITECTURES:
            raise ValueError("unknown particle-view predictor architecture")
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")
        if self.decoder_mode not in {"none", "broadcast", "cross_attention"}:
            raise ValueError("invalid decoder_mode")
        if self.use_hierarchy:
            if len(self.hierarchy_sizes) < 2:
                raise ValueError("hierarchical predictors need at least two levels")
            if any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
                for size in self.hierarchy_sizes
            ):
                raise ValueError("hierarchy sizes must be positive integers")
            if any(
                left <= right
                for left, right in zip(
                    self.hierarchy_sizes, self.hierarchy_sizes[1:]
                )
            ):
                raise ValueError("hierarchy sizes must decrease sequentially")
        elif self.hierarchy_sizes:
            raise ValueError("non-hierarchical predictors use an empty hierarchy")
        if not self.use_hierarchy and self.use_balance_loss:
            raise ValueError("balance loss is undefined without a hierarchy")
        if self.decoder_mode == "cross_attention" and not self.use_hierarchy:
            raise ValueError("cross-attention decoder requires hierarchy")
        if self.decoder_mode != "cross_attention" and self.decoder_blocks:
            raise ValueError("decoder blocks require cross_attention mode")
        if self.global_region_blocks and not self.use_hierarchy:
            raise ValueError("region blocks require hierarchy")
        for name in (
            "embedding_epsilon",
            "empty_occupancy_threshold",
            "empty_weight_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.balance_weight not in {0.0, 0.01}:
            raise ValueError("balance weight must be the locked 0.01 or zero")
        if self.use_balance_loss != (self.balance_weight > 0):
            raise ValueError("balance flag and weight disagree")
        if self.dropout not in {0.0, 0.05}:
            raise ValueError("dropout must be zero or the canonical 0.05")
        if not self.log_variance_min < self.log_variance_max:
            raise ValueError("invalid log-variance bounds")
        if self.shared_consumer_stem != (
            self.architecture_id == "P_SHARED_CONSUMER_STEM"
        ):
            raise ValueError("shared stem is restricted to its diagnostic control")

    @property
    def canonical(self) -> bool:
        return self.architecture_id in {
            "P_HIER_DECODER_REFINE",
            "P_REGIONS_16_8_4",
            "P_WIDTH192",
        }

    @property
    def selectable(self) -> bool:
        return self.architecture_id not in NONSELECTABLE_PREDICTOR_ARCHITECTURES

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "architecture_id": self.architecture_id,
            "architecture_name": (
                PVA3_CANONICAL_NAME if self.canonical else self.architecture_id
            ),
            "view_dim": self.view_dim,
            "input_dim": self.input_dim,
            "width": self.width,
            "num_heads": self.num_heads,
            "local_blocks": self.local_blocks,
            "global_region_blocks": self.global_region_blocks,
            "particle_global_blocks": self.particle_global_blocks,
            "decoder_blocks": self.decoder_blocks,
            "final_refinement_blocks": self.final_refinement_blocks,
            "hierarchy_sizes": list(self.hierarchy_sizes),
            "use_hierarchy": self.use_hierarchy,
            "decoder_mode": self.decoder_mode,
            "use_pair_bias": self.use_pair_bias,
            "use_balance_loss": self.use_balance_loss,
            "balance_weight": self.balance_weight,
            "ffn_expansion": self.ffn_expansion,
            "dropout": self.dropout,
            "embedding_epsilon": self.embedding_epsilon,
            "empty_occupancy_threshold": self.empty_occupancy_threshold,
            "empty_weight_threshold": self.empty_weight_threshold,
            "log_variance_bounds": [
                self.log_variance_min,
                self.log_variance_max,
            ],
            "predict_trust": self.predict_trust,
            "shared_consumer_stem": self.shared_consumer_stem,
            "selectable": self.selectable,
            "four_vector_order": ["px", "py", "pz", "energy"],
            "particle_pair_feature_order": list(PARTICLE_PAIR_FEATURE_ORDER),
            "pooling": {
                "sequential": True,
                "embedding_weight": "normalized_particle_pt",
                "physical_four_vector_weight": "assignment_once",
                "particle_to_fine_centroid_refinement_passes": 1,
                "higher_transition_geometry": (
                    "single_embedding_proposal_then_registered_pair_assignment"
                ),
                "iterative_refinement": False,
                "invented_initial_coordinates": False,
            },
            "soft_hierarchy_relation": True,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())

    @property
    def structural_hash(self) -> str:
        payload = self.to_payload()
        for field in ("architecture_id", "architecture_name", "selectable"):
            payload.pop(field)
        return canonical_sha256(payload)


def build_predictor_architecture_config(
    architecture_id: str,
    *,
    view_dim: int,
    input_dim: int = 17,
) -> ParticleViewPredictorConfig:
    """Return one locked member of the Step-5 architecture screen."""

    if architecture_id not in PARTICLE_VIEW_PREDICTOR_ARCHITECTURES:
        raise ValueError(f"unknown architecture_id {architecture_id!r}")
    values: dict[str, Any] = {
        "view_dim": view_dim,
        "input_dim": input_dim,
        "architecture_id": architecture_id,
    }
    if architecture_id == "P_C0_PARTICLE":
        values.update(
            local_blocks=0,
            global_region_blocks=0,
            decoder_blocks=0,
            final_refinement_blocks=0,
            hierarchy_sizes=(),
            use_hierarchy=False,
            decoder_mode="none",
            use_pair_bias=False,
            use_balance_loss=False,
            balance_weight=0.0,
        )
    elif architecture_id == "P_PART_BASIC":
        values.update(
            local_blocks=0,
            global_region_blocks=0,
            decoder_blocks=0,
            final_refinement_blocks=0,
            hierarchy_sizes=(),
            use_hierarchy=False,
            decoder_mode="none",
            use_pair_bias=False,
            particle_global_blocks=3,
            use_balance_loss=False,
            balance_weight=0.0,
        )
    elif architecture_id == "P_LOCAL":
        values.update(
            global_region_blocks=0,
            decoder_blocks=0,
            final_refinement_blocks=0,
            hierarchy_sizes=(),
            use_hierarchy=False,
            decoder_mode="none",
            use_balance_loss=False,
            balance_weight=0.0,
        )
    elif architecture_id == "P_LOCAL_GLOBAL":
        values.update(
            global_region_blocks=0,
            decoder_blocks=0,
            final_refinement_blocks=0,
            hierarchy_sizes=(),
            use_hierarchy=False,
            decoder_mode="none",
            particle_global_blocks=4,
            use_balance_loss=False,
            balance_weight=0.0,
        )
    elif architecture_id == "P_HIER_NO_DECODER":
        values.update(decoder_blocks=0, final_refinement_blocks=0, decoder_mode="broadcast")
    elif architecture_id == "P_HIER_DECODER":
        values.update(final_refinement_blocks=0)
    elif architecture_id in {
        "P_HIER_DECODER_REFINE",
        "P_REGIONS_16_8_4",
        "P_WIDTH192",
    }:
        pass
    elif architecture_id == "P_NO_PAIR_BIAS":
        values.update(use_pair_bias=False)
    elif architecture_id == "P_NO_REFINEMENT":
        values.update(final_refinement_blocks=0)
    elif architecture_id == "P_REGIONS_8_4":
        values.update(hierarchy_sizes=(8, 4))
    elif architecture_id == "P_REGIONS_16_8_4_2":
        values.update(hierarchy_sizes=(16, 8, 4, 2))
    elif architecture_id == "P_NO_BALANCE":
        values.update(use_balance_loss=False, balance_weight=0.0)
    elif architecture_id == "P_WIDTH128":
        values.update(width=128)
    elif architecture_id == "P_WIDTH256":
        values.update(width=256)
    elif architecture_id == "P_SHARED_CONSUMER_STEM":
        values.update(shared_consumer_stem=True)
    return ParticleViewPredictorConfig(**values)


def build_predictor_architecture_screen(
    *, view_dim: int, input_dim: int = 17
) -> dict[str, ParticleViewPredictorConfig]:
    return {
        name: build_predictor_architecture_config(
            name, view_dim=view_dim, input_dim=input_dim
        )
        for name in PARTICLE_VIEW_PREDICTOR_ARCHITECTURES
    }


def _normalise_mask(mask: torch.Tensor, batch: int, particles: int) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise ValueError("particle mask must be boolean")
    if mask.shape == (batch, particles):
        result = mask
    elif mask.shape == (batch, 1, particles):
        result = mask[:, 0]
    else:
        raise ValueError("particle mask must be [B,P] or [B,1,P]")
    if (~result.any(dim=1)).any():
        raise ValueError("predictor does not accept all-padding events")
    return result


def _normalise_features(
    features: torch.Tensor, *, input_dim: int
) -> tuple[torch.Tensor, int, int]:
    if features.ndim != 3:
        raise ValueError("HLT features must be rank three")
    if features.shape[1] == input_dim:
        values = features.transpose(1, 2)
    elif features.shape[2] == input_dim:
        values = features
    else:
        raise ValueError("HLT features do not match predictor input_dim")
    if not torch.isfinite(values).all():
        raise ValueError("HLT features must be finite")
    return values, values.shape[0], values.shape[1]


def _normalise_four_vectors(
    vectors: torch.Tensor, *, batch: int, particles: int
) -> torch.Tensor:
    if vectors.shape == (batch, 4, particles):
        values = vectors.transpose(1, 2)
    elif vectors.shape == (batch, particles, 4):
        values = vectors
    else:
        raise ValueError("four-vectors must be [B,4,P] or [B,P,4]")
    if not torch.isfinite(values).all():
        raise ValueError("four-vectors must be finite")
    return values


def cartesian_four_vector_kinematics(
    four_vectors: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return ``[pt, eta, phi, mass]`` from ``[px,py,pz,E]``."""

    if four_vectors.ndim != 3 or four_vectors.shape[-1] != 4:
        raise ValueError("four_vectors must be [B,N,4]")
    px, py, pz, energy = four_vectors.unbind(dim=-1)
    pt2 = px.square() + py.square()
    pt = torch.where(
        pt2 > 0,
        torch.sqrt(pt2.clamp_min(epsilon * epsilon)),
        torch.zeros_like(pt2),
    )
    eta = torch.asinh(pz / pt.clamp_min(epsilon))
    phi = torch.atan2(py, px)
    mass2 = energy.square() - px.square() - py.square() - pz.square()
    mass = torch.where(
        mass2 > 0,
        torch.sqrt(mass2.clamp_min(epsilon * epsilon)),
        torch.zeros_like(mass2),
    )
    nonzero = pt > epsilon
    eta = torch.where(nonzero, eta, torch.zeros_like(eta))
    phi = torch.where(nonzero, phi, torch.zeros_like(phi))
    return torch.stack((pt, eta, phi, mass), dim=-1)


def build_particle_pair_features(
    source_four_vectors: torch.Tensor,
    destination_four_vectors: torch.Tensor | None = None,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Build directed HLT-only ParT-style pair features."""

    destination_four_vectors = (
        source_four_vectors
        if destination_four_vectors is None
        else destination_four_vectors
    )
    source = cartesian_four_vector_kinematics(
        source_four_vectors, epsilon=epsilon
    )
    destination = cartesian_four_vector_kinematics(
        destination_four_vectors, epsilon=epsilon
    )
    source_pt, source_eta, source_phi, _ = source.unbind(-1)
    dest_pt, dest_eta, dest_phi, _ = destination.unbind(-1)
    deta = source_eta[:, :, None] - dest_eta[:, None, :]
    dphi = wrap_delta_phi(source_phi[:, :, None] - dest_phi[:, None, :])
    dr2 = deta.square() + dphi.square()
    dr = torch.where(
        dr2 > 0,
        torch.sqrt(dr2.clamp_min(epsilon * epsilon)),
        torch.zeros_like(dr2),
    )
    log_pt_ratio = torch.log(
        (source_pt[:, :, None] + epsilon)
        / (dest_pt[:, None, :] + epsilon)
    ).clamp(-12.0, 12.0)
    pair_vector = (
        source_four_vectors[:, :, None, :]
        + destination_four_vectors[:, None, :, :]
    )
    pair_kinematics = cartesian_four_vector_kinematics(
        pair_vector.reshape(pair_vector.shape[0], -1, 4),
        epsilon=epsilon,
    ).reshape(*pair_vector.shape[:-1], 4)
    pair_mass = pair_kinematics[..., 3]
    return torch.stack(
        (
            deta,
            dphi,
            torch.log1p(dr),
            log_pt_ratio,
            torch.log1p(pair_mass),
            torch.cos(dphi),
        ),
        dim=-1,
    )


class _PairAttentionBlock(nn.Module):
    """Pre-norm attention with an optional learned directed pair bias."""

    def __init__(
        self,
        width: int,
        heads: int,
        pair_dim: int,
        *,
        dropout: float,
        expansion: int,
        cross_attention: bool = False,
        use_pair_bias: bool = True,
    ) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.cross_attention = cross_attention
        self.use_pair_bias = use_pair_bias
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width) if cross_attention else None
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.pair_bias = (
            nn.Sequential(
                nn.Linear(pair_dim, width),
                nn.GELU(),
                nn.Linear(width, heads),
            )
            if use_pair_bias
            else None
        )
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * width, width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        query_mask: torch.Tensor,
        pair_features: torch.Tensor,
        *,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = queries if memory is None else memory
        memory_mask = query_mask if memory_mask is None else memory_mask
        qn = self.query_norm(queries)
        mn = (
            self.memory_norm(memory)
            if self.memory_norm is not None
            else self.query_norm(memory)
        )
        batch, query_count, _ = qn.shape
        memory_count = mn.shape[1]
        q = self.query(qn).reshape(
            batch, query_count, self.heads, self.head_width
        ).transpose(1, 2)
        k = self.key(mn).reshape(
            batch, memory_count, self.heads, self.head_width
        ).transpose(1, 2)
        v = self.value(mn).reshape(
            batch, memory_count, self.heads, self.head_width
        ).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            self.head_width
        )
        if self.pair_bias is not None:
            scores = scores + self.pair_bias(pair_features).permute(0, 3, 1, 2)
        pair_mask = query_mask[:, None, :, None] & memory_mask[:, None, None, :]
        scores = scores.masked_fill(~pair_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(pair_mask, attention, torch.zeros_like(attention))
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )
        message = torch.matmul(attention, v).transpose(1, 2).reshape(
            batch, query_count, self.width
        )
        values = queries + self.dropout(self.output(message))
        values = torch.where(
            query_mask[:, :, None], values, torch.zeros_like(values)
        )
        values = values + self.ff(self.ff_norm(values))
        return torch.where(
            query_mask[:, :, None], values, torch.zeros_like(values)
        )


class _AssignmentTransition(nn.Module):
    """One embedding-only proposal followed by one geometric assignment.

    At particle-to-fine this is the mandated single centroid-refinement pass.
    A sequential higher transition also needs a provisional centroid before
    ``p_rs`` can exist; B/C are each evaluated once and are never iterated.
    """

    def __init__(
        self,
        width: int,
        output_slots: int,
        *,
        pair_dim: int,
    ) -> None:
        super().__init__()
        self.output_slots = output_slots
        self.slot_seeds = nn.Parameter(torch.empty(output_slots, width))
        nn.init.normal_(self.slot_seeds, std=width**-0.5)
        self.initial = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.refine = nn.Sequential(
            nn.Linear(2 * width + pair_dim, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.empty_state = nn.Parameter(torch.zeros(width))

    def _expanded(
        self, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, width = source.shape
        left = source[:, :, None, :].expand(
            batch, count, self.output_slots, width
        )
        right = self.slot_seeds[None, None, :, :].expand(
            batch, count, self.output_slots, width
        )
        return left, right

    def forward(
        self,
        source: torch.Tensor,
        source_four_vectors: torch.Tensor,
        source_weight: torch.Tensor,
        source_occupancy: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        epsilon: float,
        empty_occupancy_threshold: float,
        empty_weight_threshold: float,
        use_pair_bias: bool,
    ) -> "RegionLevel":
        left, right = self._expanded(source)
        initial_logits = self.initial(torch.cat((left, right), dim=-1)).squeeze(-1)
        initial_assignment = torch.softmax(initial_logits, dim=-1)
        initial_assignment = torch.where(
            source_mask[:, :, None],
            initial_assignment,
            torch.zeros_like(initial_assignment),
        )
        provisional_four_vectors = torch.einsum(
            "bsr,bsk->brk", initial_assignment, source_four_vectors
        )
        pair_features = build_particle_pair_features(
            source_four_vectors,
            provisional_four_vectors,
            epsilon=epsilon,
        )
        if not use_pair_bias:
            pair_features = torch.zeros_like(pair_features)
        refined_logits = self.refine(
            torch.cat((left, right, pair_features), dim=-1)
        ).squeeze(-1)
        assignment = torch.softmax(refined_logits, dim=-1)
        assignment = torch.where(
            source_mask[:, :, None], assignment, torch.zeros_like(assignment)
        )

        valid_source_weight = torch.where(
            source_mask, source_weight, torch.zeros_like(source_weight)
        )
        valid_source_occupancy = torch.where(
            source_mask, source_occupancy, torch.zeros_like(source_occupancy)
        )
        inherited_weight = torch.einsum(
            "bsr,bs->br", assignment, valid_source_weight
        )
        occupancy = torch.einsum(
            "bsr,bs->br", assignment, valid_source_occupancy
        )
        four_vectors = torch.einsum(
            "bsr,bsk->brk",
            assignment,
            torch.where(
                source_mask[:, :, None],
                source_four_vectors,
                torch.zeros_like(source_four_vectors),
            ),
        )
        numerator = torch.einsum(
            "bsr,bs,bsw->brw",
            assignment,
            valid_source_weight,
            source,
        )
        embeddings = numerator / inherited_weight[:, :, None].clamp_min(epsilon)
        valid = (occupancy >= empty_occupancy_threshold) & (
            inherited_weight >= empty_weight_threshold
        )
        embeddings = torch.where(
            valid[:, :, None],
            embeddings,
            self.empty_state[None, None, :].expand_as(embeddings),
        )
        four_vectors = torch.where(
            valid[:, :, None], four_vectors, torch.zeros_like(four_vectors)
        )
        return RegionLevel(
            embeddings=embeddings,
            four_vectors=four_vectors,
            weight=inherited_weight,
            occupancy=occupancy,
            valid=valid,
            assignment=assignment,
            provisional_assignment=initial_assignment,
        )


@dataclass
class RegionLevel:
    embeddings: torch.Tensor
    four_vectors: torch.Tensor
    weight: torch.Tensor
    occupancy: torch.Tensor
    valid: torch.Tensor
    assignment: torch.Tensor
    provisional_assignment: torch.Tensor


@dataclass
class ParticleViewHierarchy:
    levels: tuple[RegionLevel, ...]
    directed_relation: torch.Tensor
    relation_complement: torch.Tensor
    level_ids: torch.Tensor
    balance_terms: torch.Tensor
    assignment_entropy: torch.Tensor
    maximum_slot_mass: torch.Tensor
    empty_rate: torch.Tensor


@dataclass
class ParticleViewPredictorOutput:
    mean: torch.Tensor
    log_variance: torch.Tensor
    trust: torch.Tensor | None
    balance_loss: torch.Tensor
    hierarchy: ParticleViewHierarchy | None
    local_embeddings: torch.Tensor
    decoded_embeddings: torch.Tensor


def _balance_term(weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    normalised = weight / weight.sum(dim=1, keepdim=True).clamp_min(epsilon)
    target = 1.0 / weight.shape[1]
    return (normalised - target).square().mean(dim=1)


def _assignment_diagnostics(
    assignment: torch.Tensor, source_mask: torch.Tensor, level: RegionLevel
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entropy_per_source = -(
        assignment.clamp_min(1.0e-12) * assignment.clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    entropy = (
        (entropy_per_source * source_mask).sum(dim=1)
        / source_mask.sum(dim=1).clamp_min(1)
    )
    mass = assignment.sum(dim=1)
    maximum = mass.max(dim=1).values / source_mask.sum(dim=1).clamp_min(1)
    empty = (~level.valid).float().mean(dim=1)
    return entropy, maximum, empty


def build_soft_hierarchy_relations(
    levels: Sequence[RegionLevel],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(levels) < 2:
        raise ValueError("soft hierarchy relation requires multiple levels")
    batch = levels[0].embeddings.shape[0]
    sizes = [level.embeddings.shape[1] for level in levels]
    total = sum(sizes)
    relation = levels[0].embeddings.new_zeros((batch, total, total))
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    for index, level in enumerate(levels):
        start, stop = offsets[index], offsets[index + 1]
        identity = torch.eye(
            sizes[index],
            device=relation.device,
            dtype=relation.dtype,
        )[None].expand(batch, -1, -1)
        relation[:, start:stop, start:stop] = identity
    for source_index in range(len(levels) - 1):
        product = levels[source_index + 1].assignment
        for destination_index in range(source_index + 1, len(levels)):
            if destination_index > source_index + 1:
                product = torch.matmul(
                    product, levels[destination_index].assignment
                )
            left = slice(offsets[source_index], offsets[source_index + 1])
            right = slice(
                offsets[destination_index], offsets[destination_index + 1]
            )
            relation[:, left, right] = product
            relation[:, right, left] = product.transpose(1, 2)
    valid = torch.cat([level.valid for level in levels], dim=1)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    relation = torch.where(pair_valid, relation, torch.zeros_like(relation))
    complement = torch.where(
        pair_valid, 1.0 - relation, torch.zeros_like(relation)
    )
    level_ids = torch.cat(
        [
            torch.full(
                (size,),
                index,
                dtype=torch.long,
                device=relation.device,
            )
            for index, size in enumerate(sizes)
        ]
    )
    return relation, complement, level_ids


class HierarchicalParticleViewPredictor(nn.Module):
    """Canonical PVA3 predictor and its declared architecture controls."""

    def __init__(
        self,
        config: ParticleViewPredictorConfig,
        *,
        shared_particle_embedding: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if config.shared_consumer_stem:
            if shared_particle_embedding is None:
                raise ValueError(
                    "shared-stem diagnostic requires shared_particle_embedding"
                )
            self.input_embedding = shared_particle_embedding
        elif shared_particle_embedding is not None:
            raise ValueError("shared embedding supplied to a separate predictor")
        else:
            self.input_embedding = nn.Sequential(
                nn.Linear(config.input_dim, config.width),
                nn.LayerNorm(config.width),
                nn.GELU(),
            )
        pair_dim = len(PARTICLE_PAIR_FEATURE_ORDER)
        self.local_blocks = nn.ModuleList(
            _PairAttentionBlock(
                config.width,
                config.num_heads,
                pair_dim,
                dropout=config.dropout,
                expansion=config.ffn_expansion,
                use_pair_bias=config.use_pair_bias,
            )
            for _ in range(config.local_blocks)
        )
        self.particle_global_blocks = nn.ModuleList(
            _PairAttentionBlock(
                config.width,
                config.num_heads,
                pair_dim,
                dropout=config.dropout,
                expansion=config.ffn_expansion,
                use_pair_bias=False,
            )
            for _ in range(config.particle_global_blocks)
        )
        self.transitions = nn.ModuleList()
        if config.use_hierarchy:
            for size in config.hierarchy_sizes:
                self.transitions.append(
                    _AssignmentTransition(
                        config.width, size, pair_dim=pair_dim
                    )
                )
        self.scale_embeddings = (
            nn.Parameter(torch.empty(len(config.hierarchy_sizes), config.width))
            if config.use_hierarchy
            else None
        )
        if self.scale_embeddings is not None:
            nn.init.normal_(self.scale_embeddings, std=config.width**-0.5)
        global_pair_dim = pair_dim + 4
        self.global_region_blocks = nn.ModuleList(
            _PairAttentionBlock(
                config.width,
                config.num_heads,
                global_pair_dim,
                dropout=config.dropout,
                expansion=config.ffn_expansion,
                use_pair_bias=config.use_pair_bias,
            )
            for _ in range(config.global_region_blocks)
        )
        decoder_pair_dim = pair_dim + 3
        self.decoder_blocks = nn.ModuleList(
            _PairAttentionBlock(
                config.width,
                config.num_heads,
                decoder_pair_dim,
                dropout=config.dropout,
                expansion=config.ffn_expansion,
                cross_attention=True,
                use_pair_bias=config.use_pair_bias,
            )
            for _ in range(config.decoder_blocks)
        )
        self.broadcast_projection = (
            nn.Linear(2 * config.width, config.width)
            if config.decoder_mode == "broadcast"
            else None
        )
        self.refinement_input = (
            nn.Linear(2 * config.width, config.width)
            if config.decoder_mode in {"broadcast", "cross_attention"}
            else None
        )
        self.final_refinement = nn.ModuleList(
            _PairAttentionBlock(
                config.width,
                config.num_heads,
                pair_dim,
                dropout=config.dropout,
                expansion=config.ffn_expansion,
                use_pair_bias=config.use_pair_bias,
            )
            for _ in range(config.final_refinement_blocks)
        )
        self.control_mlp = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )
        self.mean_head = nn.Linear(config.width, config.view_dim)
        self.log_variance_head = nn.Linear(config.width, config.view_dim)
        self.trust_head = (
            nn.Linear(config.width, 1) if config.predict_trust else None
        )

    def _hierarchy(
        self,
        local: torch.Tensor,
        four_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> ParticleViewHierarchy:
        config = self.config
        kinematics = cartesian_four_vector_kinematics(
            four_vectors, epsilon=config.embedding_epsilon
        )
        pt = torch.where(mask, kinematics[..., 0], torch.zeros_like(kinematics[..., 0]))
        source_weight = pt / pt.sum(dim=1, keepdim=True).clamp_min(
            config.embedding_epsilon
        )
        source_occupancy = mask.to(local.dtype)
        source, source_four_vectors, source_mask = local, four_vectors, mask
        levels: list[RegionLevel] = []
        balances: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        maximums: list[torch.Tensor] = []
        empty_rates: list[torch.Tensor] = []
        for transition in self.transitions:
            level = transition(
                source,
                source_four_vectors,
                source_weight,
                source_occupancy,
                source_mask,
                epsilon=config.embedding_epsilon,
                empty_occupancy_threshold=config.empty_occupancy_threshold,
                empty_weight_threshold=config.empty_weight_threshold,
                use_pair_bias=config.use_pair_bias,
            )
            levels.append(level)
            balances.append(_balance_term(level.weight, config.embedding_epsilon))
            entropy, maximum, empty = _assignment_diagnostics(
                level.assignment, source_mask, level
            )
            entropies.append(entropy)
            maximums.append(maximum)
            empty_rates.append(empty)
            source = level.embeddings
            source_four_vectors = level.four_vectors
            source_weight = level.weight
            source_occupancy = level.occupancy
            source_mask = level.valid
        relation, complement, level_ids = build_soft_hierarchy_relations(levels)
        return ParticleViewHierarchy(
            levels=tuple(levels),
            directed_relation=relation,
            relation_complement=complement,
            level_ids=level_ids,
            balance_terms=torch.stack(balances, dim=1),
            assignment_entropy=torch.stack(entropies, dim=1),
            maximum_slot_mass=torch.stack(maximums, dim=1),
            empty_rate=torch.stack(empty_rates, dim=1),
        )

    def _region_pair_features(
        self, hierarchy: ParticleViewHierarchy
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = torch.cat(
            [
                level.embeddings + self.scale_embeddings[index][None, None, :]
                for index, level in enumerate(hierarchy.levels)
            ],
            dim=1,
        )
        four_vectors = torch.cat(
            [level.four_vectors for level in hierarchy.levels], dim=1
        )
        valid = torch.cat([level.valid for level in hierarchy.levels], dim=1)
        geometry = build_particle_pair_features(
            four_vectors, epsilon=self.config.embedding_epsilon
        )
        count = embeddings.shape[1]
        denominator = max(len(hierarchy.levels) - 1, 1)
        source_level = (
            hierarchy.level_ids.to(embeddings.dtype)[None, :, None]
            .expand(embeddings.shape[0], count, count)
            / denominator
        )
        destination_level = (
            hierarchy.level_ids.to(embeddings.dtype)[None, None, :]
            .expand(embeddings.shape[0], count, count)
            / denominator
        )
        pair = torch.cat(
            (
                geometry,
                hierarchy.directed_relation[..., None],
                hierarchy.relation_complement[..., None],
                source_level[..., None],
                destination_level[..., None],
            ),
            dim=-1,
        )
        pair_valid = valid[:, :, None] & valid[:, None, :]
        pair = torch.where(
            pair_valid[:, :, :, None], pair, torch.zeros_like(pair)
        )
        return embeddings, four_vectors, valid, pair

    def _decoder_pair_features(
        self,
        particles: torch.Tensor,
        hierarchy: ParticleViewHierarchy,
    ) -> torch.Tensor:
        region_four_vectors = torch.cat(
            [level.four_vectors for level in hierarchy.levels], dim=1
        )
        geometry = build_particle_pair_features(
            particles,
            region_four_vectors,
            epsilon=self.config.embedding_epsilon,
        )
        occupancy = torch.cat(
            [level.occupancy for level in hierarchy.levels], dim=1
        )
        valid = torch.cat([level.valid for level in hierarchy.levels], dim=1)
        level = hierarchy.level_ids.to(geometry.dtype)
        level = level / max(len(hierarchy.levels) - 1, 1)
        batch, particles_count, regions = geometry.shape[:3]
        return torch.cat(
            (
                geometry,
                level[None, None, :, None].expand(
                    batch, particles_count, regions, 1
                ),
                torch.log1p(occupancy)[:, None, :, None].expand(
                    batch, particles_count, regions, 1
                ),
                (~valid).to(geometry.dtype)[:, None, :, None].expand(
                    batch, particles_count, regions, 1
                ),
            ),
            dim=-1,
        )

    def forward(
        self,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> ParticleViewPredictorOutput:
        values, batch, particles = _normalise_features(
            features, input_dim=self.config.input_dim
        )
        valid = _normalise_mask(mask, batch, particles)
        four_vectors = _normalise_four_vectors(
            lorentz_vectors, batch=batch, particles=particles
        )
        four_vectors = torch.where(
            valid[:, :, None], four_vectors, torch.zeros_like(four_vectors)
        )
        values = self.input_embedding(values)
        if values.shape != (batch, particles, self.config.width):
            raise ValueError("particle embedding output shape changed")
        values = torch.where(valid[:, :, None], values, torch.zeros_like(values))
        particle_pair = build_particle_pair_features(
            four_vectors, epsilon=self.config.embedding_epsilon
        )
        pair_valid = valid[:, :, None] & valid[:, None, :]
        particle_pair = torch.where(
            pair_valid[:, :, :, None],
            particle_pair,
            torch.zeros_like(particle_pair),
        )
        for block in self.local_blocks:
            values = block(values, valid, particle_pair)
        local = values
        for block in self.particle_global_blocks:
            values = block(values, valid, particle_pair)
        hierarchy = None
        decoded = values
        if self.config.use_hierarchy:
            hierarchy = self._hierarchy(local, four_vectors, valid)
            regions, _region_four_vectors, region_valid, region_pair = (
                self._region_pair_features(hierarchy)
            )
            for block in self.global_region_blocks:
                regions = block(regions, region_valid, region_pair)
            if self.config.decoder_mode == "cross_attention":
                decoder_pair = self._decoder_pair_features(
                    four_vectors, hierarchy
                )
                decoded = local
                for block in self.decoder_blocks:
                    decoded = block(
                        decoded,
                        valid,
                        decoder_pair,
                        memory=regions,
                        memory_mask=region_valid,
                    )
            elif self.config.decoder_mode == "broadcast":
                weights = region_valid.to(regions.dtype)
                pooled = (regions * weights[:, :, None]).sum(dim=1)
                pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1)
                pooled = pooled[:, None, :].expand(-1, particles, -1)
                decoded = self.broadcast_projection(
                    torch.cat((local, pooled), dim=-1)
                )
                decoded = torch.where(
                    valid[:, :, None], decoded, torch.zeros_like(decoded)
                )
            if self.refinement_input is not None:
                decoded = self.refinement_input(
                    torch.cat((local, decoded), dim=-1)
                )
                decoded = torch.where(
                    valid[:, :, None], decoded, torch.zeros_like(decoded)
                )
        if (
            self.config.architecture_id == "P_C0_PARTICLE"
            and not self.config.use_hierarchy
        ):
            decoded = decoded + self.control_mlp(decoded)
            decoded = torch.where(
                valid[:, :, None], decoded, torch.zeros_like(decoded)
            )
        for block in self.final_refinement:
            decoded = block(decoded, valid, particle_pair)
        mean = masked_particle_mean_center(self.mean_head(decoded), valid)
        log_variance = self.log_variance_head(decoded).clamp(
            self.config.log_variance_min, self.config.log_variance_max
        )
        log_variance = torch.where(
            valid[:, :, None],
            log_variance,
            torch.zeros_like(log_variance),
        )
        trust = (
            torch.sigmoid(self.trust_head(decoded))
            if self.trust_head is not None
            else None
        )
        if trust is not None:
            trust = torch.where(
                valid[:, :, None], trust, torch.zeros_like(trust)
            )
        if hierarchy is not None and self.config.use_balance_loss:
            balance = (
                hierarchy.balance_terms.mean()
                * self.config.balance_weight
            )
        else:
            balance = decoded.new_zeros(())
        return ParticleViewPredictorOutput(
            mean=mean,
            log_variance=log_variance,
            trust=trust,
            balance_loss=balance,
            hierarchy=hierarchy,
            local_embeddings=local,
            decoded_embeddings=decoded,
        )


def build_canonical_particle_view_predictor(
    *, view_dim: int, input_dim: int = 17
) -> HierarchicalParticleViewPredictor:
    return HierarchicalParticleViewPredictor(
        build_predictor_architecture_config(
            PVA3_CANONICAL_ARCHITECTURE,
            view_dim=view_dim,
            input_dim=input_dim,
        )
    )


def build_flop_fixture(
    *,
    input_dim: int = 17,
    particles: int = 128,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build the locked finite 128-particle resource fixture."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(PARTICLE_VIEW_FLOP_FIXTURE_SEED)
    features = torch.randn(
        (1, input_dim, particles), generator=generator, dtype=torch.float32
    )
    momentum = torch.randn(
        (1, 3, particles), generator=generator, dtype=torch.float32
    )
    energy = torch.sqrt(momentum.square().sum(dim=1, keepdim=True) + 1.0)
    vectors = torch.cat((momentum, energy), dim=1)
    mask = torch.ones((1, 1, particles), dtype=torch.bool)
    return {
        "features": features.to(device),
        "lorentz_vectors": vectors.to(device),
        "mask": mask.to(device),
    }


def flop_fixture_sha256(
    fixture: Mapping[str, torch.Tensor] | None = None,
    *,
    input_dim: int = 17,
    particles: int = 128,
) -> str:
    """Hash tensor names, shapes, dtypes, and canonical CPU bytes."""

    if fixture is None:
        fixture = build_flop_fixture(
            input_dim=input_dim, particles=particles, device="cpu"
        )
    expected = {"features", "lorentz_vectors", "mask"}
    if set(fixture) != expected:
        raise ValueError("FLOP fixture tensor inventory mismatch")
    entries = []
    for name in sorted(fixture):
        value = fixture[name].detach().cpu().contiguous()
        entries.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes_sha256": hashlib.sha256(
                    value.numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
    return canonical_sha256(
        {
            "contract": PARTICLE_VIEW_FLOP_FIXTURE,
            "seed": PARTICLE_VIEW_FLOP_FIXTURE_SEED,
            "tensors": entries,
        }
    )


def count_unique_parameters(
    modules: nn.Module | Iterable[nn.Module],
    *,
    trainable_only: bool = False,
) -> int:
    """Count learned scalars once, deduplicating shared storage."""

    if isinstance(modules, nn.Module):
        modules = (modules,)
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            if trainable_only and not parameter.requires_grad:
                continue
            identity = (
                str(parameter.device),
                int(parameter.untyped_storage().data_ptr()),
                int(parameter.storage_offset()),
            )
            if identity in seen:
                continue
            seen.add(identity)
            total += parameter.numel()
    return total


class _SemanticFlopCounter:
    """Unfused deterministic counter for the predictor's learned operations."""

    def __init__(self) -> None:
        self.breakdown: dict[str, int] = {}
        self.handles: list[Any] = []

    def add(self, name: str, value: int) -> None:
        self.breakdown[name] = self.breakdown.get(name, 0) + int(value)

    def _linear(self, module: nn.Linear, _inputs, output) -> None:
        rows = output.numel() // module.out_features
        self.add("linear_matmul", 2 * rows * module.in_features * module.out_features)
        if module.bias is not None:
            self.add("linear_bias", rows * module.out_features)

    def _layer_norm(self, module: nn.LayerNorm, _inputs, output) -> None:
        width = math.prod(module.normalized_shape)
        rows = output.numel() // width
        # Mean, centering, square, variance, eps/sqrt, normalize, affine.
        self.add("layer_norm", rows * (7 * width + 2))

    def _gelu(self, _module: nn.GELU, _inputs, output) -> None:
        self.add("gelu", 5 * output.numel())

    def _attention(self, module: _PairAttentionBlock, inputs, output) -> None:
        queries = inputs[0]
        pair = inputs[2]
        batch, query_count = queries.shape[:2]
        memory_count = pair.shape[2]
        heads, depth = module.heads, module.head_width
        self.add(
            "attention_qk_matmul",
            2 * batch * heads * query_count * memory_count * depth,
        )
        self.add(
            "attention_value_matmul",
            2 * batch * heads * query_count * memory_count * depth,
        )
        self.add(
            "attention_scale",
            batch * heads * query_count * memory_count,
        )
        if module.use_pair_bias:
            self.add(
                "attention_pair_bias_add",
                batch * heads * query_count * memory_count,
            )
        self.add(
            "softmax",
            batch
            * heads
            * query_count
            * (3 * memory_count - 1),
        )
        self.add(
            "attention_probability_renormalization",
            batch
            * heads
            * query_count
            * (2 * memory_count - 1),
        )
        self.add("attention_residual_add", 2 * output.numel())

    def __enter__(self):
        return self

    def attach(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear))
            elif isinstance(module, nn.LayerNorm):
                self.handles.append(module.register_forward_hook(self._layer_norm))
            elif isinstance(module, nn.GELU):
                self.handles.append(module.register_forward_hook(self._gelu))
            elif isinstance(module, _PairAttentionBlock):
                self.handles.append(module.register_forward_hook(self._attention))

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def predictor_semantic_flops(
    model: HierarchicalParticleViewPredictor,
    *,
    particles: int = 128,
) -> dict[str, Any]:
    """Count deterministic semantic FLOPs on the locked float32 fixture.

    The Step-8 exported predictor-plus-consumer counter extends the same
    implementation across the consumer.  This Step-5 profile covers the
    predictor boundary and records that scope explicitly.
    """

    fixture = build_flop_fixture(
        input_dim=model.config.input_dim,
        particles=particles,
        device=next(model.parameters()).device,
    )
    training = model.training
    model.eval()
    counter = _SemanticFlopCounter()
    with torch.no_grad(), counter:
        counter.attach(model)
        model(**fixture)
    model.train(training)
    # Explicit HLT geometry and pooling arithmetic is versioned separately
    # from learned modules, so it cannot silently disappear from comparisons.
    # Counts below follow the decompositions stated in Section 17: masking,
    # reshape, transpose, concatenation, and memory movement are zero-cost.
    def kinematic_flops(count: int) -> int:
        # pt2/sqrt, eta, phi, invariant mass, and nonzero comparisons.
        return 18 * count

    def pair_feature_flops(source: int, destination: int) -> int:
        pairs = source * destination
        # Source/destination kinematics plus directed angular/pt/mass features.
        return (
            kinematic_flops(source)
            + kinematic_flops(destination)
            + 43 * pairs
        )

    config = model.config
    width = config.width
    levels = config.hierarchy_sizes
    counter.add(
        "particle_pair_feature_arithmetic",
        pair_feature_flops(particles, particles),
    )
    if config.architecture_id == "P_C0_PARTICLE":
        counter.add("particle_control_residual_add", particles * width)
    if levels:
        counter.add(
            "particle_pt_weight_normalization",
            kinematic_flops(particles) + (particles - 1) + particles,
        )
        source = particles
        for destination in levels:
            counter.add(
                "hierarchy_pair_feature_arithmetic",
                pair_feature_flops(source, destination),
            )
            counter.add(
                "hierarchy_assignment_softmax",
                2 * source * (3 * destination - 1),
            )
            # Provisional p4, inherited W/n/p4, embedding numerator/divide,
            # and the two empty predicates plus their conjunction.
            counter.add(
                "hierarchy_pooling",
                8 * source * destination
                + 2 * source * destination
                + 2 * source * destination
                + 8 * source * destination
                + source * destination
                + 2 * source * destination * width
                + destination * width
                + 3 * destination,
            )
            # Balance and entropy/max-mass/empty-rate diagnostics.
            counter.add("hierarchy_balance", 5 * destination - 1)
            counter.add(
                "hierarchy_diagnostics",
                (3 * source * destination + 2 * source)
                + (source * destination + source - 1)
                + 2 * destination,
            )
            source = destination
        total_regions = sum(levels)
        # Non-adjacent soft relation products. Adjacent relations and
        # transposes are views/copies and therefore zero-cost.
        for start in range(len(levels) - 2):
            product_source = levels[start]
            product_middle = levels[start + 1]
            for destination_index in range(start + 2, len(levels)):
                product_destination = levels[destination_index]
                counter.add(
                    "soft_hierarchy_relation_matmul",
                    2
                    * product_source
                    * product_middle
                    * product_destination,
                )
                product_middle = product_destination
        counter.add(
            "soft_hierarchy_relation_complement",
            total_regions * total_regions,
        )
        counter.add("region_scale_embedding_add", total_regions * width)
        counter.add(
            "region_pair_feature_arithmetic",
            pair_feature_flops(total_regions, total_regions),
        )
        counter.add("region_level_normalization", 2 * total_regions)
        if config.decoder_mode == "cross_attention":
            counter.add(
                "decoder_pair_feature_arithmetic",
                pair_feature_flops(particles, total_regions),
            )
            counter.add(
                "decoder_level_occupancy_features",
                3 * total_regions,
            )
        elif config.decoder_mode == "broadcast":
            counter.add(
                "region_broadcast_pooling",
                total_regions * width
                + width * (total_regions - 1)
                + width,
            )
        if config.use_balance_loss:
            counter.add(
                "weighted_balance_loss",
                len(levels) + 1,
            )
    view_values = particles * config.view_dim
    counter.add(
        "masked_output_mean_center",
        4 * view_values + particles - 1,
    )
    counter.add("log_variance_bounds", 2 * view_values)
    if config.predict_trust:
        counter.add("trust_sigmoid", particles)
    return {
        "counter": PARTICLE_VIEW_FLOP_COUNTER,
        "scope": "predictor_only_preprocessed_hlt_to_view",
        "fixture_contract": PARTICLE_VIEW_FLOP_FIXTURE,
        "fixture_sha256": flop_fixture_sha256(fixture),
        "particles": particles,
        "batch_size": 1,
        "dtype": "float32",
        "exact_integer_total": int(sum(counter.breakdown.values())),
        "per_operator": dict(sorted(counter.breakdown.items())),
    }


def profile_predictor_resources(
    model: HierarchicalParticleViewPredictor,
    *,
    particles: int = 128,
    warmup: int = 2,
    repetitions: int = 5,
    campaign_batch_size: int = 32,
) -> dict[str, Any]:
    """Profile parameters, semantic FLOPs, memory, and diagnostic latency."""

    if particles != 128:
        raise ValueError("resource profiles require the locked 128-particle fixture")
    if (
        warmup < 0
        or repetitions <= 0
        or not isinstance(campaign_batch_size, int)
        or isinstance(campaign_batch_size, bool)
        or campaign_batch_size <= 0
    ):
        raise ValueError("invalid profiling repetition counts")
    parameter = next(model.parameters())
    device = parameter.device
    fixture = build_flop_fixture(
        input_dim=model.config.input_dim,
        particles=particles,
        device=device,
    )
    training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(**fixture)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(repetitions):
            start = time.perf_counter()
            model(**fixture)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1_000.0)
        batch1_inference_peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
        campaign_fixture = {
            name: value.expand(
                campaign_batch_size, *([-1] * (value.ndim - 1))
            ).contiguous()
            for name, value in fixture.items()
        }
        campaign_samples = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repetitions):
            start = time.perf_counter()
            model(**campaign_fixture)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            campaign_samples.append((time.perf_counter() - start) * 1_000.0)
        campaign_inference_peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
        del campaign_fixture
    training_peak = None
    if device.type == "cuda":
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        output = model(**fixture)
        (
            output.mean.square().mean()
            + output.log_variance.square().mean()
            + output.balance_loss
        ).backward()
        torch.cuda.synchronize(device)
        training_peak = int(torch.cuda.max_memory_allocated(device))
        model.zero_grad(set_to_none=True)
    model.train(training)
    profile = {
        "contract": PARTICLE_VIEW_PREDICTOR_RESOURCE_CONTRACT,
        "architecture_config_sha256": model.config.content_hash,
        "architecture_structural_sha256": model.config.structural_hash,
        "architecture_config": model.config.to_payload(),
        "architecture_id": model.config.architecture_id,
        "total_parameters": count_unique_parameters(model),
        "trainable_parameters": count_unique_parameters(
            model, trainable_only=True
        ),
        "forward_flops": predictor_semantic_flops(
            model, particles=particles
        ),
        "inference_peak_memory_bytes": campaign_inference_peak,
        "batch1_inference_peak_memory_bytes": batch1_inference_peak,
        "training_peak_memory_bytes": training_peak,
        "latency_diagnostic": {
            "device": str(device),
            "batch_size": 1,
            "valid_particles": particles,
            "warmup": warmup,
            "repetitions": repetitions,
            "median_milliseconds": statistics.median(samples),
            "precision": "float32",
            "campaign_batch_size": campaign_batch_size,
            "campaign_batch_median_milliseconds": statistics.median(
                campaign_samples
            ),
        },
        "predictor_consumer_weights_shared": model.config.shared_consumer_stem,
        "peak_memory_warning": (
            None
            if device.type == "cuda"
            else "WARN_CPU_NATIVE_PEAK_MEMORY_NOT_MEASURED"
        ),
    }
    profile["content_hash"] = canonical_sha256(profile)
    return profile


def validate_predictor_resource_profile(
    profile: Mapping[str, Any],
    *,
    expected_config_sha256: str | None = None,
) -> None:
    """Validate a persisted Step-5 resource record."""

    if profile.get("contract") != PARTICLE_VIEW_PREDICTOR_RESOURCE_CONTRACT:
        raise ValueError("predictor resource contract mismatch")
    if expected_config_sha256 is not None and profile.get(
        "architecture_config_sha256"
    ) != expected_config_sha256:
        raise ValueError("predictor resource architecture hash mismatch")
    config_payload = profile.get("architecture_config")
    if (
        not isinstance(config_payload, Mapping)
        or canonical_sha256(config_payload)
        != profile.get("architecture_config_sha256")
    ):
        raise ValueError("predictor resource embedded architecture mismatch")
    for field in ("total_parameters", "trainable_parameters"):
        value = profile.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"invalid predictor resource {field}")
    if profile["trainable_parameters"] > profile["total_parameters"]:
        raise ValueError("trainable parameter count exceeds total")
    flops = profile.get("forward_flops")
    if not isinstance(flops, Mapping) or flops.get("counter") != PARTICLE_VIEW_FLOP_COUNTER:
        raise ValueError("predictor FLOP-counter contract mismatch")
    if flops.get("particles") != 128 or flops.get("batch_size") != 1:
        raise ValueError("predictor FLOP fixture must be batch-1/128-particle")
    if (
        flops.get("fixture_contract") != PARTICLE_VIEW_FLOP_FIXTURE
        or flops.get("fixture_sha256")
        != flop_fixture_sha256(
            input_dim=profile["architecture_config"]["input_dim"],
            particles=128,
        )
    ):
        raise ValueError("predictor FLOP fixture identity mismatch")
    total = flops.get("exact_integer_total")
    breakdown = flops.get("per_operator")
    if (
        not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
        or not isinstance(breakdown, Mapping)
        or sum(breakdown.values()) != total
    ):
        raise ValueError("predictor FLOP totals do not reconcile")
    for field in (
        "inference_peak_memory_bytes",
        "batch1_inference_peak_memory_bytes",
        "training_peak_memory_bytes",
    ):
        value = profile.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(f"invalid predictor resource {field}")
    payload = dict(profile)
    observed = payload.pop("content_hash", None)
    if observed != canonical_sha256(payload):
        raise ValueError("predictor resource content hash mismatch")

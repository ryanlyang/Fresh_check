"""Cell-local constrained particle-slot rendering for the C-tier models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .constraints import (
    ACCOUNTING_INDEX,
    CategoryCountSlotAllocator,
    CategoryPtSlotAllocator,
    CategorySlotAllocationOutput,
    CellBoundCoordinateTransform,
    TOTAL_ENERGY_INDEX,
    assemble_accounting,
)
from .layout import (
    ACCOUNTING_FIELD_NAMES,
    MOMENT_FIELD_NAMES,
    PID_CATEGORY_NAMES,
    PRIMITIVE_ACCOUNTING_FIELD_NAMES,
    HierarchyTargetLayout,
)
from .model import (
    B1_GLOBAL_8,
    B2_GLOBAL_8_32,
    B3_FULL_HIERARCHY,
    CoarseToFineReconstructorConfig,
    CoarseToFineReconstructorOutput,
    ConstrainedCoarseToFineReconstructor,
    build_coarse_to_fine_reconstructor,
)


PARTICLE_SLOT_DECODER_CONTRACT = "constrained_coarse_to_fine_particle_slot_decoder_v1"

C0_DETERMINISTIC_K8 = "C0_deterministic_k8"
C1_DETERMINISTIC_K16 = "C1_deterministic_k16"
C2_NO_DUST = "C2_no_dust"
C3_SINKHORN = "C3_sinkhorn"
C4_HUNGARIAN = "C4_hungarian"
C5_UNCERTAINTY = "C5_uncertainty"
C6_MULTIVIEW = "C6_multiview"
C5_B1 = "C5-B1"
C5_B2 = "C5-B2"
C5_B3 = "C5-B3"

C_TIER_VARIANTS: tuple[str, ...] = (
    C0_DETERMINISTIC_K8,
    C1_DETERMINISTIC_K16,
    C2_NO_DUST,
    C3_SINKHORN,
    C4_HUNGARIAN,
    C5_UNCERTAINTY,
    C6_MULTIVIEW,
    C5_B1,
    C5_B2,
    C5_B3,
)

_C_ALIASES = {
    "C0": C0_DETERMINISTIC_K8,
    "C1": C1_DETERMINISTIC_K16,
    "C2": C2_NO_DUST,
    "C3": C3_SINKHORN,
    "C4": C4_HUNGARIAN,
    "C5": C5_UNCERTAINTY,
    "C6": C6_MULTIVIEW,
    "C5_B1": C5_B1,
    "C5_B2": C5_B2,
    "C5_B3": C5_B3,
}


def normalize_c_tier_variant(value: str) -> str:
    key = str(value).strip()
    if key in C_TIER_VARIANTS:
        return key
    if key in _C_ALIASES:
        return _C_ALIASES[key]
    raise ValueError(f"unknown C-tier slot variant {value!r}")


@dataclass(frozen=True)
class ParticleSlotDecoderSpec:
    """Resolved rendering recipe independent of hierarchy depth."""

    num_real_slots: int = 16
    include_dust: bool = True
    matching_mode: str = "sinkhorn"
    use_uncertainty: bool = True
    num_views: int = 1
    stochastic_latent_dim: int = 32
    slot_layers: int = 3
    max_hlt_particles_per_cell: int = 24

    def __post_init__(self) -> None:
        for name in (
            "num_real_slots",
            "num_views",
            "stochastic_latent_dim",
            "slot_layers",
            "max_hlt_particles_per_cell",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.matching_mode not in {"ordered", "sinkhorn", "hungarian"}:
            raise ValueError("matching_mode must be ordered, sinkhorn, or hungarian")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


C5_SHARED_SLOT_SPEC = ParticleSlotDecoderSpec()


@dataclass(frozen=True)
class CTierVariantSpec:
    name: str
    hierarchy_variant: str
    slot_spec: ParticleSlotDecoderSpec

    @property
    def hierarchy_depth(self) -> int:
        return {B1_GLOBAL_8: 1, B2_GLOBAL_8_32: 2, B3_FULL_HIERARCHY: 3}[self.hierarchy_variant]

    @property
    def terminal_cell_count(self) -> int:
        return (8, 32, 128)[self.hierarchy_depth - 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hierarchy_variant": self.hierarchy_variant,
            "hierarchy_depth": self.hierarchy_depth,
            "terminal_cell_count": self.terminal_cell_count,
            "slot_spec": self.slot_spec.to_dict(),
        }


def c_tier_variant_spec(value: str) -> CTierVariantSpec:
    variant = normalize_c_tier_variant(value)
    hierarchy = B3_FULL_HIERARCHY
    if variant == C0_DETERMINISTIC_K8:
        slots = ParticleSlotDecoderSpec(
            num_real_slots=8,
            matching_mode="ordered",
            use_uncertainty=False,
            stochastic_latent_dim=32,
        )
    elif variant == C1_DETERMINISTIC_K16:
        slots = replace(C5_SHARED_SLOT_SPEC, matching_mode="ordered", use_uncertainty=False)
    elif variant == C2_NO_DUST:
        slots = replace(C5_SHARED_SLOT_SPEC, include_dust=False, matching_mode="ordered", use_uncertainty=False)
    elif variant == C3_SINKHORN:
        slots = replace(C5_SHARED_SLOT_SPEC, use_uncertainty=False)
    elif variant == C4_HUNGARIAN:
        slots = replace(C5_SHARED_SLOT_SPEC, matching_mode="hungarian", use_uncertainty=False)
    elif variant == C5_UNCERTAINTY:
        slots = C5_SHARED_SLOT_SPEC
    elif variant == C6_MULTIVIEW:
        slots = replace(C5_SHARED_SLOT_SPEC, num_views=4)
    elif variant == C5_B1:
        hierarchy = B1_GLOBAL_8
        slots = C5_SHARED_SLOT_SPEC
    elif variant == C5_B2:
        hierarchy = B2_GLOBAL_8_32
        slots = C5_SHARED_SLOT_SPEC
    elif variant == C5_B3:
        hierarchy = B3_FULL_HIERARCHY
        slots = C5_SHARED_SLOT_SPEC
    else:  # pragma: no cover - normalize_c_tier_variant is exhaustive
        raise AssertionError(variant)
    return CTierVariantSpec(variant, hierarchy, slots)


@dataclass(frozen=True)
class ParticleSlotDecoderConfig:
    """Neural settings for one resolved cell-local slot decoder."""

    variant: str = C5_UNCERTAINTY
    d_model: int = 256
    num_heads: int = 8
    ffn_multiplier: float = 4.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    uncertainty_min: float = -8.0
    uncertainty_max: float = 8.0
    constrain_accounting: bool = True
    direct_particle_decoding: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", normalize_c_tier_variant(self.variant))
        for name in ("d_model", "num_heads"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.d_model) % int(self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if float(self.ffn_multiplier) <= 0.0:
            raise ValueError("ffn_multiplier must be positive")
        for name in ("dropout", "attention_dropout"):
            if not 0.0 <= float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if float(self.uncertainty_min) >= float(self.uncertainty_max):
            raise ValueError("uncertainty bounds are reversed")
        if bool(self.direct_particle_decoding) and bool(self.constrain_accounting):
            raise ValueError("direct particle decoding cannot use hierarchical accounting constraints")

    @property
    def variant_spec(self) -> CTierVariantSpec:
        return c_tier_variant_spec(self.variant)

    @property
    def slot_spec(self) -> ParticleSlotDecoderSpec:
        return self.variant_spec.slot_spec

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = PARTICLE_SLOT_DECODER_CONTRACT
        payload["variant_spec"] = self.variant_spec.to_dict()
        return payload


@dataclass(frozen=True)
class ParticleSlotDecoderOutput:
    variant: str
    terminal_level: int
    terminal_accounting: torch.Tensor
    terminal_cell_tokens: torch.Tensor
    real_slot_embeddings: torch.Tensor
    local_coordinates: torch.Tensor
    total_pt: torch.Tensor
    category_pt: torch.Tensor
    total_energy: torch.Tensor
    expected_count: torch.Tensor
    category_count: torch.Tensor
    pid_probabilities: torch.Tensor
    raw_pid_logits: torch.Tensor
    charge_logits: torch.Tensor
    existence_logits: torch.Tensor
    log_sigma: torch.Tensor | None
    reliability: torch.Tensor
    dust_total_pt: torch.Tensor | None
    dust_category_pt: torch.Tensor | None
    dust_total_energy: torch.Tensor | None
    rendered_accounting: torch.Tensor
    stochastic_latent: torch.Tensor | None
    diagnostics: Mapping[str, Any]

    @property
    def num_views(self) -> int:
        return int(self.total_pt.shape[1])

    @property
    def num_cells(self) -> int:
        return int(self.total_pt.shape[2])

    @property
    def num_real_slots(self) -> int:
        return int(self.total_pt.shape[3])

    def summary(self) -> dict[str, Any]:
        return {
            "contract": PARTICLE_SLOT_DECODER_CONTRACT,
            "variant": self.variant,
            "terminal_level": int(self.terminal_level),
            "terminal_accounting_shape": list(self.terminal_accounting.shape),
            "real_slot_embedding_shape": list(self.real_slot_embeddings.shape),
            "local_coordinate_shape": list(self.local_coordinates.shape),
            "num_views": self.num_views,
            "num_cells": self.num_cells,
            "num_real_slots": self.num_real_slots,
            "dust_enabled": self.dust_total_pt is not None,
            "uncertainty_enabled": self.log_sigma is not None,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class CTierReconstructorOutput:
    hierarchy: CoarseToFineReconstructorOutput
    slots: ParticleSlotDecoderOutput

    def summary(self) -> dict[str, Any]:
        return {"hierarchy": self.hierarchy.summary(), "slots": self.slots.summary()}


def _batch_first(value: torch.Tensor, channels: int, *, name: str) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError(f"{name} must be rank 3")
    if int(value.shape[-1]) == int(channels):
        return value
    if int(value.shape[1]) == int(channels):
        return value.transpose(1, 2).contiguous()
    raise ValueError(f"{name} has no channel dimension of size {channels}: {tuple(value.shape)}")


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _SlotBlock(nn.Module):
    def __init__(self, config: ParticleSlotDecoderConfig) -> None:
        super().__init__()
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
        self.ffn = _FeedForward(
            config.d_model,
            int(round(config.ffn_multiplier * config.d_model)),
            config.dropout,
        )

    def forward(
        self,
        slots: torch.Tensor,
        hlt_memory: torch.Tensor,
        hlt_mask: torch.Tensor,
        ancestor_memory: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm_self(slots)
        update, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        slots = slots + update
        update, _ = self.hlt_attention(
            self.norm_hlt(slots),
            hlt_memory,
            hlt_memory,
            key_padding_mask=~hlt_mask,
            need_weights=False,
        )
        slots = slots + update
        update, _ = self.ancestor_attention(
            self.norm_ancestor(slots), ancestor_memory, ancestor_memory, need_weights=False
        )
        slots = slots + update
        return slots + self.ffn(self.norm_ffn(slots))


def _terminal_inputs(
    hierarchy: CoarseToFineReconstructorOutput,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hierarchy.levels:
        raise ValueError("particle slots require at least a B1 hierarchy with spatial cells")
    terminal = hierarchy.levels[-1]
    terminal_level = int(terminal.level)
    count = int(terminal.accounting.shape[1])
    ancestors = [hierarchy.global_token[:, None, :].expand(-1, count, -1)]
    for previous in hierarchy.levels[:-1]:
        divisor = 4 ** (terminal_level - int(previous.level))
        indices = torch.arange(count, device=terminal.accounting.device, dtype=torch.long) // divisor
        ancestors.append(previous.cell_tokens.index_select(1, indices))
    ancestor_tokens = torch.stack(ancestors, dim=2)
    return terminal_level, terminal.accounting, terminal.cell_tokens, ancestor_tokens


def _cell_geometry(
    layout: HierarchyTargetLayout,
    level: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = layout.cell_geometry(level)
    geometry_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    bounds = torch.tensor(
        [[row["eta_min"], row["eta_max"], row["phi_min"], row["phi_max"]] for row in rows],
        device=device,
        dtype=geometry_dtype,
    )
    radial_bounds = torch.tensor(
        [[row["radial_min"], row["radial_max"]] for row in rows],
        device=device,
        dtype=geometry_dtype,
    )
    features = []
    radial_scale = math.sqrt(2.0) * float(layout.coordinate_extent)
    for row in rows:
        features.append(
            [
                0.5 * (row["eta_min"] + row["eta_max"]),
                0.5 * (row["phi_min"] + row["phi_max"]),
                row["eta_max"] - row["eta_min"],
                row["phi_max"] - row["phi_min"],
                row["radial_min"] / radial_scale,
                row["radial_max"] / radial_scale,
                float(row["radial_bin"]),
                float(level) / 3.0,
            ]
        )
    return bounds, radial_bounds, torch.tensor(features, device=device, dtype=geometry_dtype)


def _geometry_feasibility(bounds: torch.Tensor, radial_bounds: torch.Tensor) -> torch.Tensor:
    eta_min, eta_max, phi_min, phi_max = bounds.unbind(dim=-1)
    nearest_eta = torch.where(
        (eta_min <= 0.0) & (eta_max >= 0.0), torch.zeros_like(eta_min), torch.minimum(eta_min.abs(), eta_max.abs())
    )
    nearest_phi = torch.where(
        (phi_min <= 0.0) & (phi_max >= 0.0), torch.zeros_like(phi_min), torch.minimum(phi_min.abs(), phi_max.abs())
    )
    minimum_radius = torch.sqrt(nearest_eta.square() + nearest_phi.square())
    maximum_radius = torch.sqrt(
        torch.maximum(eta_min.abs(), eta_max.abs()).square()
        + torch.maximum(phi_min.abs(), phi_max.abs()).square()
    )
    return (maximum_radius >= radial_bounds[..., 0]) & (minimum_radius <= radial_bounds[..., 1])


def _hlt_axis_relative_coordinates(
    lorentz_vectors: torch.Tensor,
    mask: torch.Tensor,
    *,
    coordinate_extent: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = _batch_first(lorentz_vectors, 4, name="lorentz_vectors")
    valid = mask.bool()
    px, py, pz, _ = vectors.unbind(dim=-1)
    pt = torch.sqrt(px.square() + py.square())
    eta = torch.asinh(pz / pt.clamp_min(1.0e-12))
    phi = torch.atan2(py, px)
    valid = valid & torch.isfinite(vectors).all(dim=-1) & (pt > 0.0)
    weight = torch.where(valid, pt, torch.zeros_like(pt))
    denominator = weight.sum(dim=1).clamp_min(1.0e-12)
    reference_eta = (weight * torch.where(valid, eta, torch.zeros_like(eta))).sum(dim=1) / denominator
    reference_phi = torch.atan2(
        (weight * torch.sin(torch.where(valid, phi, torch.zeros_like(phi)))).sum(dim=1),
        (weight * torch.cos(torch.where(valid, phi, torch.zeros_like(phi)))).sum(dim=1),
    )
    deta = eta - reference_eta[:, None]
    dphi = torch.remainder(phi - reference_phi[:, None] + math.pi, 2.0 * math.pi) - math.pi
    extent = float(coordinate_extent)
    coordinates = torch.stack(
        (deta.clamp(-extent, extent), dphi.clamp(-extent, extent)),
        dim=-1,
    )
    coordinates = torch.where(valid.unsqueeze(-1), coordinates, torch.zeros_like(coordinates))
    return coordinates, valid


def _select_local_hlt_memory(
    hierarchy: CoarseToFineReconstructorOutput,
    lorentz_vectors: torch.Tensor,
    cell_bounds: torch.Tensor,
    radial_bounds: torch.Tensor,
    max_particles: int,
    *,
    coordinate_extent: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    particles = hierarchy.hlt.particle_embeddings
    point_rows, mask = _hlt_axis_relative_coordinates(
        lorentz_vectors,
        hierarchy.hlt.particle_mask,
        coordinate_extent=coordinate_extent,
    )
    point_rows = point_rows.to(device=particles.device, dtype=particles.dtype)
    centers = torch.stack(
        (
            0.5 * (cell_bounds[:, 0] + cell_bounds[:, 1]),
            0.5 * (cell_bounds[:, 2] + cell_bounds[:, 3]),
        ),
        dim=-1,
    )
    deta = point_rows[:, None, :, 0] - centers[None, :, None, 0]
    dphi = torch.remainder(
        point_rows[:, None, :, 1] - centers[None, :, None, 1] + math.pi,
        2.0 * math.pi,
    ) - math.pi
    distance = deta.square() + dphi.square()
    radius = torch.linalg.vector_norm(point_rows, dim=-1)
    inside = (
        (point_rows[:, None, :, 0] >= cell_bounds[None, :, None, 0])
        & (point_rows[:, None, :, 0] <= cell_bounds[None, :, None, 1])
        & (point_rows[:, None, :, 1] >= cell_bounds[None, :, None, 2])
        & (point_rows[:, None, :, 1] <= cell_bounds[None, :, None, 3])
        & (radius[:, None, :] >= radial_bounds[None, :, None, 0])
        & (radius[:, None, :] <= radial_bounds[None, :, None, 1])
    )
    distance = distance + (~inside).to(distance.dtype) * 1.0e4
    distance = distance.masked_fill(~mask[:, None, :], float("inf"))
    selected_count = min(int(max_particles), int(particles.shape[1]))
    indices = torch.topk(distance, k=selected_count, dim=-1, largest=False).indices
    expanded_particles = particles[:, None, :, :].expand(-1, int(cell_bounds.shape[0]), -1, -1)
    selected = expanded_particles.gather(
        2, indices.unsqueeze(-1).expand(-1, -1, -1, int(particles.shape[-1]))
    )
    selected_mask = mask[:, None, :].expand(-1, int(cell_bounds.shape[0]), -1).gather(2, indices)
    empty = ~selected_mask.any(dim=-1)
    if bool(empty.any()):
        selected = selected.clone()
        selected_mask = selected_mask.clone()
        empty_indices = torch.nonzero(empty, as_tuple=False)
        selected_mask[empty_indices[:, 0], empty_indices[:, 1], 0] = True
        selected[empty_indices[:, 0], empty_indices[:, 1], 0] = 0.0
    return selected, selected_mask


def render_slot_accounting(
    *,
    category_pt_all: torch.Tensor,
    category_count_real: torch.Tensor,
    energy_all: torch.Tensor,
    coordinates_all: torch.Tensor,
) -> torch.Tensor:
    """Rebuild additive cell accounting from constrained slot outputs."""

    if int(category_pt_all.shape[-1]) != len(PID_CATEGORY_NAMES):
        raise ValueError("category_pt_all has the wrong PID dimension")
    total_pt_all = category_pt_all.sum(dim=-1)
    primitive = category_pt_all.new_zeros(
        *category_pt_all.shape[:-2], len(PRIMITIVE_ACCOUNTING_FIELD_NAMES)
    )
    primitive_index = {name: index for index, name in enumerate(PRIMITIVE_ACCOUNTING_FIELD_NAMES)}
    primitive[..., primitive_index["total_energy"]] = energy_all.sum(dim=-1)
    for category_index, category in enumerate(PID_CATEGORY_NAMES):
        primitive[..., primitive_index[f"{category}_pT"]] = category_pt_all[..., category_index].sum(dim=-1)
        primitive[..., primitive_index[f"{category}_count"]] = category_count_real[..., category_index].sum(dim=-1)
    deta = coordinates_all[..., 0]
    dphi = coordinates_all[..., 1]
    radius = torch.sqrt(deta.square() + dphi.square() + 1.0e-12)
    moment_values = {
        "sum_pT_abs_deta_pos": total_pt_all * deta.clamp_min(0.0),
        "sum_pT_abs_deta_neg": total_pt_all * (-deta).clamp_min(0.0),
        "sum_pT_abs_dphi_pos": total_pt_all * dphi.clamp_min(0.0),
        "sum_pT_abs_dphi_neg": total_pt_all * (-dphi).clamp_min(0.0),
        "sum_pT_deta2": total_pt_all * deta.square(),
        "sum_pT_dphi2": total_pt_all * dphi.square(),
        "sum_pT_r": total_pt_all * radius,
        "sum_pT_r2": total_pt_all * radius.square(),
    }
    for name in MOMENT_FIELD_NAMES:
        primitive[..., primitive_index[name]] = moment_values[name].sum(dim=-1)
    return assemble_accounting(primitive)


class ParticleSlotDecoder(nn.Module):
    """Render constrained real slots and an optional unresolved dust slot."""

    UNCERTAINTY_NAMES = ("log_pT", "deta", "dphi", "energy", "pid")

    def __init__(
        self,
        config: ParticleSlotDecoderConfig | Mapping[str, Any] | None = None,
        *,
        layout: HierarchyTargetLayout,
    ) -> None:
        super().__init__()
        if config is None:
            config = ParticleSlotDecoderConfig()
        elif not isinstance(config, ParticleSlotDecoderConfig):
            config = ParticleSlotDecoderConfig(**dict(config))
        self.config = config
        self.spec = config.slot_spec
        self.layout = layout
        total_queries = self.spec.num_real_slots + int(self.spec.include_dust)
        self.slot_queries = nn.Parameter(torch.zeros(1, 1, total_queries, config.d_model))
        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        self.accounting_projection = nn.Sequential(
            nn.LayerNorm(len(ACCOUNTING_FIELD_NAMES)),
            nn.Linear(len(ACCOUNTING_FIELD_NAMES), config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.cell_projection = nn.Linear(config.d_model, config.d_model)
        self.geometry_projection = nn.Sequential(
            nn.Linear(8, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.latent_projection = nn.Linear(self.spec.stochastic_latent_dim, config.d_model)
        self.view_embedding = nn.Embedding(max(8, self.spec.num_views), config.d_model)
        self.blocks = nn.ModuleList([_SlotBlock(config) for _ in range(self.spec.slot_layers)])
        self.output_norm = nn.LayerNorm(config.d_model)
        hidden = int(round(config.ffn_multiplier * config.d_model))

        def head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, output_dim),
            )

        self.category_pt_head = head(len(PID_CATEGORY_NAMES))
        self.category_count_head = head(len(PID_CATEGORY_NAMES))
        self.energy_head = head(1)
        self.coordinate_head = head(2)
        self.pid_head = head(len(PID_CATEGORY_NAMES))
        self.charge_head = head(3)
        self.existence_head = head(1)
        self.uncertainty_head = head(len(self.UNCERTAINTY_NAMES))
        self.reliability_head = head(1)
        self.pt_allocator = CategoryPtSlotAllocator()
        self.count_allocator = CategoryCountSlotAllocator()
        self.coordinate_transform = CellBoundCoordinateTransform()

    def forward(
        self,
        hierarchy: CoarseToFineReconstructorOutput,
        lorentz_vectors: torch.Tensor,
        *,
        stochastic_latent: torch.Tensor | None = None,
    ) -> ParticleSlotDecoderOutput:
        terminal_level, accounting, cell_tokens, ancestors = _terminal_inputs(hierarchy)
        if terminal_level != self.config.variant_spec.hierarchy_depth:
            raise ValueError(
                f"{self.config.variant} requires hierarchy depth {self.config.variant_spec.hierarchy_depth}, "
                f"but received level {terminal_level}"
            )
        batch, cells, _ = accounting.shape
        views = int(self.spec.num_views)
        bounds, radial_bounds, geometry = _cell_geometry(
            self.layout,
            terminal_level,
            device=accounting.device,
            dtype=accounting.dtype,
        )
        geometry_feasible = _geometry_feasibility(bounds, radial_bounds)
        local_hlt, local_hlt_mask = _select_local_hlt_memory(
            hierarchy,
            lorentz_vectors,
            bounds,
            radial_bounds,
            self.spec.max_hlt_particles_per_cell,
            coordinate_extent=float(self.layout.coordinate_extent),
        )
        if stochastic_latent is None and views > 1:
            stochastic_latent = torch.randn(
                batch,
                views,
                self.spec.stochastic_latent_dim,
                device=accounting.device,
                dtype=accounting.dtype,
            )
        elif stochastic_latent is not None:
            expected = (batch, views, self.spec.stochastic_latent_dim)
            if tuple(stochastic_latent.shape) != expected:
                raise ValueError(f"stochastic_latent must have shape {expected}, got {tuple(stochastic_latent.shape)}")
            stochastic_latent = stochastic_latent.to(device=accounting.device, dtype=accounting.dtype)
        total_queries = self.spec.num_real_slots + int(self.spec.include_dust)
        slots = self.slot_queries.expand(batch, cells, -1, -1)
        if self.config.direct_particle_decoding:
            conditioning = self.cell_projection(hierarchy.global_token)[:, None, :].expand(-1, cells, -1)
            conditioning = conditioning + self.geometry_projection(geometry)[None, :, :]
        else:
            conditioning = self.cell_projection(cell_tokens)
            conditioning = conditioning + self.accounting_projection(torch.log1p(accounting.clamp_min(0.0)))
            conditioning = conditioning + self.geometry_projection(geometry)[None, :, :]
        slots = slots[:, None, :, :, :].expand(-1, views, -1, -1, -1)
        slots = slots + conditioning[:, None, :, None, :]
        view_ids = torch.arange(views, device=accounting.device)
        slots = slots + self.view_embedding(view_ids)[None, :, None, None, :]
        if stochastic_latent is not None:
            slots = slots + self.latent_projection(stochastic_latent)[:, :, None, None, :]

        flat_slots = slots.reshape(batch * views * cells, total_queries, self.config.d_model)
        hlt_memory = local_hlt[:, None, :, :, :].expand(-1, views, -1, -1, -1).reshape(
            batch * views * cells, local_hlt.shape[2], self.config.d_model
        )
        hlt_mask = local_hlt_mask[:, None, :, :].expand(-1, views, -1, -1).reshape(
            batch * views * cells, local_hlt_mask.shape[2]
        )
        if self.config.direct_particle_decoding:
            ancestor_memory = hierarchy.global_token[:, None, None, :].expand(-1, views, cells, -1)
            ancestor_memory = ancestor_memory.reshape(batch * views * cells, 1, self.config.d_model)
        else:
            ancestor_memory = ancestors[:, None, :, :, :].expand(-1, views, -1, -1, -1).reshape(
                batch * views * cells, ancestors.shape[2], self.config.d_model
            )
        for block in self.blocks:
            flat_slots = block(flat_slots, hlt_memory, hlt_mask, ancestor_memory)
        hidden_all = self.output_norm(flat_slots).reshape(
            batch, views, cells, total_queries, self.config.d_model
        )
        real_hidden = hidden_all[..., : self.spec.num_real_slots, :]

        expanded_accounting = accounting[:, None, :, :].expand(-1, views, -1, -1)
        category_pt_logits = self.category_pt_head(hidden_all)
        category_count_logits = self.category_count_head(real_hidden)
        energy_logits = self.energy_head(hidden_all).squeeze(-1)
        if self.config.constrain_accounting:
            pt_allocation = self.pt_allocator(expanded_accounting, category_pt_logits)
            count_allocation = self.count_allocator(expanded_accounting, category_count_logits)
            energy_fractions = torch.softmax(energy_logits, dim=-1)
            energy_all = expanded_accounting[..., TOTAL_ENERGY_INDEX].unsqueeze(-1) * energy_fractions
        else:
            if self.config.direct_particle_decoding:
                category_per_slot = F.softplus(category_pt_logits)
                category_per_slot = category_per_slot * geometry_feasible[None, None, :, None, None]
            else:
                pt_scale = expanded_accounting[..., ACCOUNTING_INDEX["total_pT"]].unsqueeze(-1).unsqueeze(-1)
                pt_scale = pt_scale / float(total_queries * len(PID_CATEGORY_NAMES))
                category_per_slot = F.softplus(category_pt_logits) * (pt_scale / math.log(2.0))
            total_per_slot = category_per_slot.sum(dim=-1)
            category_probabilities = category_per_slot / total_per_slot.unsqueeze(-1).clamp_min(1.0e-8)
            pt_allocation = CategorySlotAllocationOutput(
                category_per_slot=category_per_slot,
                total_per_slot=total_per_slot,
                category_probabilities=category_probabilities,
                fractions=category_per_slot
                / category_per_slot.sum(dim=-2, keepdim=True).clamp_min(1.0e-8),
            )
            if self.config.direct_particle_decoding:
                category_count = F.softplus(category_count_logits)
                category_count = category_count * geometry_feasible[None, None, :, None, None]
            else:
                count_scale = expanded_accounting[..., ACCOUNTING_INDEX["expected_constituent_count"]]
                count_scale = count_scale.unsqueeze(-1).unsqueeze(-1) / float(
                    self.spec.num_real_slots * len(PID_CATEGORY_NAMES)
                )
                category_count = F.softplus(category_count_logits) * (count_scale / math.log(2.0))
            count_total = category_count.sum(dim=-1)
            count_allocation = CategorySlotAllocationOutput(
                category_per_slot=category_count,
                total_per_slot=count_total,
                category_probabilities=category_count / count_total.unsqueeze(-1).clamp_min(1.0e-8),
                fractions=category_count
                / category_count.sum(dim=-2, keepdim=True).clamp_min(1.0e-8),
            )
            if self.config.direct_particle_decoding:
                energy_all = F.softplus(energy_logits)
                energy_all = energy_all * geometry_feasible[None, None, :, None]
            else:
                energy_scale = expanded_accounting[..., TOTAL_ENERGY_INDEX].unsqueeze(-1) / float(total_queries)
                energy_all = F.softplus(energy_logits) * (energy_scale / math.log(2.0))

        raw_coordinates = self.coordinate_head(real_hidden)
        coordinates = self.coordinate_transform(
            raw_coordinates,
            bounds[None, None, :, None, :],
            radial_bounds[None, None, :, None, :],
        )
        pid_logits = self.pid_head(real_hidden)
        charge_logits = self.charge_head(real_hidden)
        existence_logits = self.existence_head(real_hidden).squeeze(-1)
        if self.config.direct_particle_decoding:
            existence_logits = existence_logits.masked_fill(
                ~geometry_feasible[None, None, :, None], -20.0
            )
        reliability = torch.sigmoid(self.reliability_head(real_hidden).squeeze(-1))
        log_sigma = None
        if self.spec.use_uncertainty:
            log_sigma = self.uncertainty_head(real_hidden).clamp(
                min=self.config.uncertainty_min,
                max=self.config.uncertainty_max,
            )

        if self.spec.include_dust:
            real_category_pt = pt_allocation.category_per_slot[..., :-1, :]
            dust_category_pt = pt_allocation.category_per_slot[..., -1, :]
            real_energy = energy_all[..., :-1]
            dust_energy = energy_all[..., -1]
            center = self.coordinate_transform(
                torch.zeros(cells, 1, 2, device=bounds.device, dtype=bounds.dtype),
                bounds[:, None, :],
                radial_bounds[:, None, :],
            ).squeeze(-2)
            coordinates_all = torch.cat(
                (
                    coordinates,
                    center[None, None, :, None, :].expand(batch, views, -1, -1, -1),
                ),
                dim=-2,
            )
            dust_total_pt = dust_category_pt.sum(dim=-1)
        else:
            real_category_pt = pt_allocation.category_per_slot
            dust_category_pt = None
            real_energy = energy_all
            dust_energy = None
            coordinates_all = coordinates
            dust_total_pt = None
        rendered = render_slot_accounting(
            category_pt_all=pt_allocation.category_per_slot,
            category_count_real=count_allocation.category_per_slot,
            energy_all=energy_all,
            coordinates_all=coordinates_all,
        )
        diagnostics: dict[str, Any] = {
            "contract": PARTICLE_SLOT_DECODER_CONTRACT,
            "variant": self.config.variant,
            "matching_mode": self.spec.matching_mode,
            "terminal_level": terminal_level,
            "num_cells": cells,
            "num_real_slots": self.spec.num_real_slots,
            "num_views": views,
            "dust_enabled": self.spec.include_dust,
            "uncertainty_enabled": self.spec.use_uncertainty,
            "accounting_constraints_enabled": bool(self.config.constrain_accounting),
            "direct_particle_decoding": bool(self.config.direct_particle_decoding),
            "geometrically_feasible_cells": int(geometry_feasible.sum().detach().cpu().item()),
            "local_hlt_memory_size": int(local_hlt.shape[2]),
            "category_pt_closure_abs_max": (
                pt_allocation.category_per_slot.sum(dim=-2)
                - expanded_accounting[..., [ACCOUNTING_INDEX[f"{name}_pT"] for name in PID_CATEGORY_NAMES]]
            ).abs().amax().detach(),
            "category_count_closure_abs_max": (
                count_allocation.category_per_slot.sum(dim=-2)
                - expanded_accounting[..., [ACCOUNTING_INDEX[f"{name}_count"] for name in PID_CATEGORY_NAMES]]
            ).abs().amax().detach(),
            "energy_closure_abs_max": (
                energy_all.sum(dim=-1) - expanded_accounting[..., TOTAL_ENERGY_INDEX]
            ).abs().amax().detach(),
        }
        return ParticleSlotDecoderOutput(
            variant=self.config.variant,
            terminal_level=terminal_level,
            terminal_accounting=accounting,
            terminal_cell_tokens=cell_tokens,
            real_slot_embeddings=real_hidden,
            local_coordinates=coordinates,
            total_pt=real_category_pt.sum(dim=-1),
            category_pt=real_category_pt,
            total_energy=real_energy,
            expected_count=count_allocation.total_per_slot,
            category_count=count_allocation.category_per_slot,
            pid_probabilities=pt_allocation.category_probabilities[..., : self.spec.num_real_slots, :],
            raw_pid_logits=pid_logits,
            charge_logits=charge_logits,
            existence_logits=existence_logits,
            log_sigma=log_sigma,
            reliability=reliability,
            dust_total_pt=dust_total_pt,
            dust_category_pt=dust_category_pt,
            dust_total_energy=dust_energy,
            rendered_accounting=rendered,
            stochastic_latent=stochastic_latent,
            diagnostics=diagnostics,
        )


class CTierParticleReconstructor(nn.Module):
    """End-to-end Step 3 hierarchy and Step 4 particle-slot model."""

    def __init__(
        self,
        hierarchy: ConstrainedCoarseToFineReconstructor,
        slot_decoder: ParticleSlotDecoder,
    ) -> None:
        super().__init__()
        self.hierarchy = hierarchy
        self.slot_decoder = slot_decoder
        if hierarchy.config.variant != slot_decoder.config.variant_spec.hierarchy_variant:
            raise ValueError("hierarchy and C-tier slot variant depths do not agree")
        if hierarchy.config.d_model != slot_decoder.config.d_model:
            raise ValueError("hierarchy and slot decoder d_model must agree")

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        *,
        stochastic_latent: torch.Tensor | None = None,
    ) -> CTierReconstructorOutput:
        hierarchy_output = self.hierarchy(points, features, lorentz_vectors, mask)
        slot_output = self.slot_decoder(
            hierarchy_output,
            lorentz_vectors,
            stochastic_latent=stochastic_latent,
        )
        return CTierReconstructorOutput(hierarchy=hierarchy_output, slots=slot_output)


def build_c_tier_reconstructor(
    variant: str,
    *,
    hierarchy_overrides: Mapping[str, Any] | None = None,
    slot_overrides: Mapping[str, Any] | None = None,
    layout: HierarchyTargetLayout | None = None,
) -> CTierParticleReconstructor:
    spec = c_tier_variant_spec(variant)
    hierarchy_payload = dict(hierarchy_overrides or {})
    hierarchy_payload["variant"] = spec.hierarchy_variant
    hierarchy_config = CoarseToFineReconstructorConfig(**hierarchy_payload)
    hierarchy = build_coarse_to_fine_reconstructor(hierarchy_config, layout=layout)
    slot_payload = dict(slot_overrides or {})
    slot_payload["variant"] = spec.name
    slot_payload.setdefault("d_model", hierarchy_config.d_model)
    slot_payload.setdefault("num_heads", hierarchy_config.num_heads)
    slot_config = ParticleSlotDecoderConfig(**slot_payload)
    decoder = ParticleSlotDecoder(slot_config, layout=hierarchy.layout)
    return CTierParticleReconstructor(hierarchy, decoder)

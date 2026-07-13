"""Step 7 uncertainty-aware dual-stream taggers for constrained pseudo-particles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from jetclass_fresh.dual_view import build_part_inputs_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .layout import ACCOUNTING_FIELD_NAMES, PID_CATEGORY_NAMES, default_hierarchy_target_layout
from .model import B3_FULL_HIERARCHY, CoarseToFineReconstructorConfig, HLTEncoderOutput, ParTStyleHLTEncoder


DUAL_STREAM_FUSION_CONTRACT = "constrained_coarse_to_fine_dual_stream_fusion_v1"
PSEUDO_VIEW_INPUT_CONTRACT = "constrained_coarse_to_fine_pseudo_view_input_v1"

D0_PSEUDO_ONLY = "D0"
D1_LATE_LOGIT_FUSION = "D1"
D2_REPRESENTATION_FUSION = "D2"
D3_CROSS_ATTENTION = "D3"
D4_UNCERTAINTY_GATED = "D4"
D5_END_TO_END = "D5"
D5_B1 = "D5-B1"
D5_B2 = "D5-B2"
D5_B3 = "D5-B3"
D6_MULTIVIEW = "D6"
D7_GRID_ONLY = "D7"
D8_MULTIDEPTH = "D8"

E0_SHUFFLED_CELLS = "E0"
E1_RANDOM_COORDINATES = "E1"
E2_SHUFFLED_COMPOSITION = "E2"
E3_NO_UNCERTAINTY = "E3"
E4_UNCONSTRAINED_SOURCE = "E4"
E5_NO_SLOT_LOSS_SOURCE = "E5"
E6_CAPACITY_MATCHED_HLT = "E6"

ARCH_PSEUDO_ONLY = "pseudo_only"
ARCH_LATE_LOGIT = "late_logit"
ARCH_REPRESENTATION = "representation"
ARCH_CROSS_ATTENTION = "cross_attention"
ARCH_GATED_CROSS_ATTENTION = "uncertainty_gated_cross_attention"
ARCH_HLT_CAPACITY_CONTROL = "hlt_capacity_control"

CONTROL_NONE = "none"
CONTROL_SHUFFLED_CELLS = "shuffled_cells"
CONTROL_RANDOM_COORDINATES = "random_coordinates"
CONTROL_SHUFFLED_COMPOSITION = "shuffled_composition"
CONTROL_NO_UNCERTAINTY = "no_uncertainty"

PSEUDO_SIDE_FEATURE_NAMES = (
    "candidate_weight",
    "existence_probability",
    "reliability",
    "log1p_expected_count",
    "dust_flag",
    "uncertainty_available",
    "log_sigma_log_pT",
    "log_sigma_deta",
    "log_sigma_dphi",
    "log_sigma_energy",
    "log_sigma_pid",
    "normalized_cell_index",
    "normalized_slot_index",
)


@dataclass(frozen=True)
class FusionVariantSpec:
    name: str
    architecture: str
    default_view_names: tuple[str, ...]
    source_recipe: str
    control: str = CONTROL_NONE
    requires_multiview: bool = False
    requires_multidepth: bool = False
    requires_grid_tokens: bool = False
    requires_end_to_end_schedule: bool = False
    description: str = ""

    @property
    def num_pseudo_views(self) -> int:
        return len(self.default_view_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "default_view_names": list(self.default_view_names),
            "num_pseudo_views": self.num_pseudo_views,
        }


def _variant_specs() -> tuple[FusionVariantSpec, ...]:
    canonical = ("canonical",)
    return (
        FusionVariantSpec(D0_PSEUDO_ONLY, ARCH_PSEUDO_ONLY, canonical, "best_c", description="Pseudo-only ParT-style tagger."),
        FusionVariantSpec(D1_LATE_LOGIT_FUSION, ARCH_LATE_LOGIT, canonical, "best_c", description="Mean of jointly trained HLT and pseudo branch logits."),
        FusionVariantSpec(D2_REPRESENTATION_FUSION, ARCH_REPRESENTATION, canonical, "best_c", description="Pre-classifier HLT/pseudo representation fusion."),
        FusionVariantSpec(D3_CROSS_ATTENTION, ARCH_CROSS_ATTENTION, canonical, "best_c", description="Ungated bidirectional token cross-attention."),
        FusionVariantSpec(D4_UNCERTAINTY_GATED, ARCH_GATED_CROSS_ATTENTION, canonical, "best_c", description="Uncertainty-gated bidirectional cross-attention."),
        FusionVariantSpec(D5_END_TO_END, ARCH_GATED_CROSS_ATTENTION, canonical, "best_c", requires_end_to_end_schedule=True, description="D4 architecture with the Step 8 gentle end-to-end schedule."),
        FusionVariantSpec(D5_B1, ARCH_GATED_CROSS_ATTENTION, ("c5_b1",), "c5_b1", requires_end_to_end_schedule=True),
        FusionVariantSpec(D5_B2, ARCH_GATED_CROSS_ATTENTION, ("c5_b2",), "c5_b2", requires_end_to_end_schedule=True),
        FusionVariantSpec(D5_B3, ARCH_GATED_CROSS_ATTENTION, ("c5_b3",), "c5_b3", requires_end_to_end_schedule=True),
        FusionVariantSpec(D6_MULTIVIEW, ARCH_GATED_CROSS_ATTENTION, tuple(f"stochastic_{index}" for index in range(4)), "c6", requires_multiview=True, requires_end_to_end_schedule=True),
        FusionVariantSpec(D7_GRID_ONLY, ARCH_GATED_CROSS_ATTENTION, ("grid",), "best_b_grid", requires_grid_tokens=True, description="Grid accounting tokens without particle slots."),
        FusionVariantSpec(D8_MULTIDEPTH, ARCH_GATED_CROSS_ATTENTION, ("best_c", "c5_b1", "c5_b2", "c5_b3"), "multidepth", requires_multidepth=True, requires_end_to_end_schedule=True),
        FusionVariantSpec(E0_SHUFFLED_CELLS, ARCH_GATED_CROSS_ATTENTION, canonical, "best_c", control=CONTROL_SHUFFLED_CELLS),
        FusionVariantSpec(E1_RANDOM_COORDINATES, ARCH_GATED_CROSS_ATTENTION, canonical, "best_c", control=CONTROL_RANDOM_COORDINATES),
        FusionVariantSpec(E2_SHUFFLED_COMPOSITION, ARCH_GATED_CROSS_ATTENTION, canonical, "best_c", control=CONTROL_SHUFFLED_COMPOSITION),
        FusionVariantSpec(E3_NO_UNCERTAINTY, ARCH_GATED_CROSS_ATTENTION, canonical, "best_c", control=CONTROL_NO_UNCERTAINTY),
        FusionVariantSpec(E4_UNCONSTRAINED_SOURCE, ARCH_GATED_CROSS_ATTENTION, canonical, "unconstrained_particle_reconstructor"),
        FusionVariantSpec(E5_NO_SLOT_LOSS_SOURCE, ARCH_GATED_CROSS_ATTENTION, canonical, "no_slot_loss_reconstructor"),
        FusionVariantSpec(E6_CAPACITY_MATCHED_HLT, ARCH_HLT_CAPACITY_CONTROL, (), "hlt_only", description="Second HLT encoder and fusion capacity, no pseudo input."),
    )


FUSION_VARIANTS = {spec.name: spec for spec in _variant_specs()}
D_TIER_VARIANTS = tuple(name for name in FUSION_VARIANTS if name.startswith("D"))
E_TIER_VARIANTS = tuple(name for name in FUSION_VARIANTS if name.startswith("E"))


def normalize_fusion_variant(value: str) -> str:
    candidate = str(value).strip().upper().replace("_", "-")
    aliases = {name.upper().replace("_", "-"): name for name in FUSION_VARIANTS}
    if candidate not in aliases:
        raise ValueError(f"unknown D/E-tier fusion variant {value!r}; expected one of {tuple(FUSION_VARIANTS)}")
    return aliases[candidate]


def fusion_variant_spec(value: str) -> FusionVariantSpec:
    return FUSION_VARIANTS[normalize_fusion_variant(value)]


@dataclass(frozen=True)
class FusionTaggerConfig:
    variant: str = D4_UNCERTAINTY_GATED
    num_classes: int = 10
    feature_dim: int = 17
    d_model: int = 256
    num_heads: int = 8
    hlt_encoder_layers: int = 6
    hlt_pool_layers: int = 2
    pseudo_local_layers: int = 2
    pseudo_global_layers: int = 3
    fusion_layers: int = 3
    ffn_multiplier: float = 4.0
    pair_hidden_dim: int = 64
    dropout: float = 0.05
    attention_dropout: float = 0.05
    pseudo_view_dropout: float = 0.15
    view_names: tuple[str, ...] | None = None
    control_seed: int = 27071

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", normalize_fusion_variant(self.variant))
        if self.view_names is not None:
            names = tuple(str(name) for name in self.view_names)
            if len(names) != len(set(names)):
                raise ValueError("view_names must be unique")
            object.__setattr__(self, "view_names", names)
        for name in (
            "num_classes",
            "feature_dim",
            "d_model",
            "num_heads",
            "hlt_encoder_layers",
            "hlt_pool_layers",
            "pseudo_local_layers",
            "pseudo_global_layers",
            "fusion_layers",
            "pair_hidden_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.d_model) % int(self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if float(self.ffn_multiplier) <= 0.0:
            raise ValueError("ffn_multiplier must be positive")
        for name in ("dropout", "attention_dropout", "pseudo_view_dropout"):
            if not 0.0 <= float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        spec = self.variant_spec
        resolved = self.resolved_view_names
        if spec.architecture == ARCH_HLT_CAPACITY_CONTROL and resolved:
            raise ValueError("E6 cannot declare pseudo views")
        if spec.requires_multiview and len(resolved) < 2:
            raise ValueError("D6 requires multiple stochastic pseudo views")
        if spec.requires_multidepth and len(resolved) < 2:
            raise ValueError("D8 requires at least two unique structural views")
        if spec.architecture != ARCH_HLT_CAPACITY_CONTROL and not resolved:
            raise ValueError(f"{self.variant} requires at least one pseudo view")

    @property
    def variant_spec(self) -> FusionVariantSpec:
        return fusion_variant_spec(self.variant)

    @property
    def resolved_view_names(self) -> tuple[str, ...]:
        return self.variant_spec.default_view_names if self.view_names is None else tuple(self.view_names)

    @property
    def pseudo_feature_dim(self) -> int:
        return int(self.feature_dim) + len(PSEUDO_SIDE_FEATURE_NAMES) + len(ACCOUNTING_FIELD_NAMES)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "view_names": list(self.resolved_view_names),
            "variant_spec": self.variant_spec.to_dict(),
            "pseudo_feature_dim": self.pseudo_feature_dim,
            "contract": DUAL_STREAM_FUSION_CONTRACT,
        }


@dataclass(frozen=True)
class ParticleStreamInput:
    points: torch.Tensor
    features: torch.Tensor
    lorentz_vectors: torch.Tensor
    mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class PseudoParticleViewInput(ParticleStreamInput):
    name: str
    raw_tokens: torch.Tensor
    candidate_weights: torch.Tensor
    existence_probability: torch.Tensor
    reliability: torch.Tensor
    expected_count: torch.Tensor
    slot_log_sigma: torch.Tensor
    uncertainty_mask: torch.Tensor
    cell_indices: torch.Tensor
    slot_indices: torch.Tensor
    is_dust: torch.Tensor
    parent_accounting: torch.Tensor
    terminal_level: int
    view_kind: str = "particle"
    source_variant: str | None = None

    @property
    def num_particles(self) -> int:
        return int(self.raw_tokens.shape[1])

    def validate(self) -> None:
        batch, particles, raw_dim = self.raw_tokens.shape
        if raw_dim != RAW_TOKEN_DIM:
            raise ValueError(f"{self.name} raw token dimension is {raw_dim}, expected {RAW_TOKEN_DIM}")
        if self.features.shape != (batch, 17, particles):
            raise ValueError(f"{self.name} feature shape is not [B,17,P]")
        if self.points.shape != (batch, 2, particles) or self.lorentz_vectors.shape != (batch, 4, particles):
            raise ValueError(f"{self.name} kinematic inputs are misaligned")
        if self.mask.shape not in {(batch, particles), (batch, 1, particles)}:
            raise ValueError(f"{self.name} mask is misaligned")
        for field_name in (
            "candidate_weights",
            "existence_probability",
            "reliability",
            "expected_count",
            "uncertainty_mask",
        ):
            if getattr(self, field_name).shape != (batch, particles):
                raise ValueError(f"{self.name} {field_name} is misaligned")
        if self.slot_log_sigma.shape != (batch, particles, 5):
            raise ValueError(f"{self.name} slot_log_sigma must be [B,P,5]")
        if self.parent_accounting.shape != (batch, particles, len(ACCOUNTING_FIELD_NAMES)):
            raise ValueError(f"{self.name} parent accounting is misaligned")
        for field_name in ("cell_indices", "slot_indices", "is_dust"):
            value = getattr(self, field_name)
            if value.ndim != 1 or int(value.shape[0]) != particles:
                raise ValueError(f"{self.name} {field_name} must be a shared [P] layout")


@dataclass(frozen=True)
class PseudoStreamEncoding:
    token_embeddings: torch.Tensor
    token_mask: torch.Tensor
    jet_embedding: torch.Tensor
    token_trust: torch.Tensor
    token_uncertainty: torch.Tensor
    slot_embeddings: torch.Tensor
    slot_mask: torch.Tensor
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class FusionTaggerOutput:
    logits: torch.Tensor
    hlt_logits: torch.Tensor | None
    pseudo_logits: torch.Tensor | None
    hlt_representation: torch.Tensor
    pseudo_representations: Mapping[str, torch.Tensor]
    token_gates: torch.Tensor | None
    pooled_gates: torch.Tensor | None
    gate_entropy_regularizer: torch.Tensor
    diagnostics: Mapping[str, Any]


def particle_stream_from_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> ParticleStreamInput:
    inputs = build_part_inputs_torch(tokens, mask, max_constits=int(tokens.shape[1]))
    return ParticleStreamInput(
        points=inputs["points"],
        features=inputs["features"],
        lorentz_vectors=inputs["lorentz_vectors"],
        mask=inputs["mask"],
    )


def _tensor(value: Any, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = value if torch.is_tensor(value) else torch.as_tensor(value)
    if device is not None:
        result = result.to(device=device)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def _terminal_level_and_accounting(arrays: Mapping[str, Any]) -> tuple[int, torch.Tensor, torch.Tensor]:
    for level in (3, 2, 1):
        accounting = _tensor(arrays[f"level{level}_accounting"], dtype=torch.float32)
        if int(accounting.shape[1]) > 0:
            return level, accounting, _tensor(arrays[f"level{level}_log_sigma"], dtype=torch.float32)
    raise ValueError("pseudo cache has no terminal hierarchy accounting")


def _view_name(prefix: str, index: int, views: int) -> str:
    return prefix if views == 1 else f"{prefix}_{index}"


def pseudo_particle_views_from_arrays(
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    view_name_prefix: str | None = None,
    device: torch.device | str | None = None,
    grid_only: bool = False,
) -> tuple[PseudoParticleViewInput, ...]:
    """Convert one Step 6 shard/cache into model-ready pseudo streams."""

    if grid_only:
        return (grid_view_from_arrays(arrays, metadata, name=view_name_prefix or "grid", device=device),)
    raw = _tensor(arrays["tokens"], device=device, dtype=torch.float32)
    mask = _tensor(arrays["mask"], device=device).bool()
    if raw.ndim != 4:
        raise ValueError("pseudo tokens must have shape [B,V,P,19]")
    batch, views, particles, _ = raw.shape
    level, terminal, _ = _terminal_level_and_accounting(arrays)
    terminal = terminal.to(device=raw.device)
    cell_indices = _tensor(arrays["token_cell_indices"], device=raw.device).long()
    slot_indices = _tensor(arrays["token_slot_indices"], device=raw.device).long()
    is_dust = _tensor(arrays["token_is_dust"], device=raw.device).bool()
    parent = terminal[:, cell_indices]
    prefix = str(view_name_prefix or metadata.get("variant") or "pseudo").lower().replace("-", "_")
    result = []
    for view_index in range(views):
        view_tokens = raw[:, view_index]
        part = build_part_inputs_torch(view_tokens, mask[:, view_index], max_constits=particles)
        view = PseudoParticleViewInput(
            name=_view_name(prefix, view_index, views),
            raw_tokens=view_tokens,
            points=part["points"],
            features=part["features"],
            lorentz_vectors=part["lorentz_vectors"],
            mask=part["mask"],
            candidate_weights=_tensor(arrays["candidate_weights"], device=raw.device, dtype=torch.float32)[:, view_index],
            existence_probability=_tensor(arrays["existence_probability"], device=raw.device, dtype=torch.float32)[:, view_index],
            reliability=_tensor(arrays["reliability"], device=raw.device, dtype=torch.float32)[:, view_index],
            expected_count=_tensor(arrays["expected_count"], device=raw.device, dtype=torch.float32)[:, view_index],
            slot_log_sigma=_tensor(arrays["slot_log_sigma"], device=raw.device, dtype=torch.float32)[:, view_index],
            uncertainty_mask=_tensor(arrays["uncertainty_mask"], device=raw.device).bool()[:, view_index],
            cell_indices=cell_indices,
            slot_indices=slot_indices,
            is_dust=is_dust,
            parent_accounting=parent,
            terminal_level=level,
            view_kind="particle",
            source_variant=str(metadata.get("variant") or "unknown"),
        )
        view.validate()
        result.append(view)
    return tuple(result)


def grid_view_from_arrays(
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    name: str = "grid",
    device: torch.device | str | None = None,
) -> PseudoParticleViewInput:
    """Represent predicted terminal accounting cells without particle slots for D7."""

    level, accounting, log_sigma = _terminal_level_and_accounting(arrays)
    accounting = accounting.to(device=device)
    log_sigma = log_sigma.to(device=device)
    batch, cells, _ = accounting.shape
    model_payload = metadata.get("source_checkpoint_model")
    hierarchy_payload = model_payload.get("hierarchy_config") if isinstance(model_payload, Mapping) else None
    radial_boundary = float((hierarchy_payload or {}).get("radial_boundary", 0.16))
    coordinate_extent = float((hierarchy_payload or {}).get("coordinate_extent", 0.8))
    layout = default_hierarchy_target_layout(
        radial_boundary=radial_boundary,
        coordinate_extent=coordinate_extent,
    )
    geometry = layout.cell_geometry(level)
    if len(geometry) != cells:
        raise ValueError(f"terminal accounting has {cells} cells, layout level {level} has {len(geometry)}")
    centers = accounting.new_tensor(
        [[0.5 * (row["eta_min"] + row["eta_max"]), 0.5 * (row["phi_min"] + row["phi_max"])] for row in geometry]
    )
    reference_eta = _tensor(arrays["reference_eta"], device=accounting.device, dtype=torch.float32)
    reference_phi = _tensor(arrays["reference_phi"], device=accounting.device, dtype=torch.float32)
    field = {field_name: index for index, field_name in enumerate(ACCOUNTING_FIELD_NAMES)}
    pt = accounting[..., field["total_pT"]]
    energy = accounting[..., field["total_energy"]]
    expected_count = accounting[..., field["expected_constituent_count"]]
    category_pt = torch.stack(
        [accounting[..., field[f"{category}_pT"]] for category in PID_CATEGORY_NAMES],
        dim=-1,
    )
    pid = category_pt / pt.unsqueeze(-1).clamp_min(1.0e-8)
    pid = torch.where(
        (pt > 0.0).unsqueeze(-1),
        pid,
        torch.full_like(pid, 1.0 / len(PID_CATEGORY_NAMES)),
    )
    absolute_eta = reference_eta[:, None] + centers[None, :, 0]
    absolute_phi = torch.remainder(reference_phi[:, None] + centers[None, :, 1] + math.pi, 2.0 * math.pi) - math.pi
    raw = accounting.new_zeros(batch, cells, RAW_TOKEN_DIM)
    raw[..., 0] = pt
    raw[..., 1] = absolute_eta
    raw[..., 2] = absolute_phi
    raw[..., 3] = energy
    raw[..., 5:10] = pid
    mask = pt > 0.0
    selected_uncertainty = torch.stack(
        (
            log_sigma[..., field["total_pT"]],
            log_sigma[..., field["sum_pT_abs_deta_pos"]],
            log_sigma[..., field["sum_pT_abs_dphi_pos"]],
            log_sigma[..., field["total_energy"]],
            log_sigma[..., field[f"{PID_CATEGORY_NAMES[0]}_pT"]],
        ),
        dim=-1,
    ).clamp(-8.0, 8.0)
    reliability = torch.exp(-selected_uncertainty.mean(dim=-1).clamp_min(0.0)).clamp(0.0, 1.0)
    part = build_part_inputs_torch(raw, mask, max_constits=cells)
    view = PseudoParticleViewInput(
        name=str(name),
        raw_tokens=raw,
        points=part["points"],
        features=part["features"],
        lorentz_vectors=part["lorentz_vectors"],
        mask=part["mask"],
        candidate_weights=reliability * mask.float(),
        existence_probability=mask.float(),
        reliability=reliability,
        expected_count=expected_count,
        slot_log_sigma=selected_uncertainty,
        uncertainty_mask=mask,
        cell_indices=torch.arange(cells, device=accounting.device),
        slot_indices=torch.full((cells,), -1, device=accounting.device, dtype=torch.long),
        is_dust=torch.zeros(cells, device=accounting.device, dtype=torch.bool),
        parent_accounting=accounting,
        terminal_level=level,
        view_kind="grid",
        source_variant=str(metadata.get("variant") or "unknown"),
    )
    view.validate()
    return view


def _mask_2d(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3 and int(mask.shape[1]) == 1:
        return mask[:, 0].bool()
    if mask.ndim == 2:
        return mask.bool()
    raise ValueError(f"mask must be [B,P] or [B,1,P], got {tuple(mask.shape)}")


def _signed_log1p(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(value.abs())


def _masked_weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = torch.clamp(weight, min=0.0)
    denominator = weight.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (value * weight.unsqueeze(-1)).sum(dim=1) / denominator


def _encoder_config(config: FusionTaggerConfig, *, feature_dim: int, encoder_layers: int) -> CoarseToFineReconstructorConfig:
    return CoarseToFineReconstructorConfig(
        variant=B3_FULL_HIERARCHY,
        feature_dim=int(feature_dim),
        d_model=int(config.d_model),
        num_heads=int(config.num_heads),
        encoder_layers=int(encoder_layers),
        pool_layers=int(config.hlt_pool_layers),
        ffn_multiplier=float(config.ffn_multiplier),
        pair_hidden_dim=int(config.pair_hidden_dim),
        dropout=float(config.dropout),
        attention_dropout=float(config.attention_dropout),
    )


class HierarchicalPseudoStreamEncoder(nn.Module):
    """Encode all particle slots locally, then globally attend over cell summaries."""

    def __init__(self, config: FusionTaggerConfig, *, view_index: int) -> None:
        super().__init__()
        self.config = config
        self.view_index = int(view_index)
        hidden = int(round(config.ffn_multiplier * config.d_model))
        self.feature_norm = nn.LayerNorm(config.pseudo_feature_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(config.pseudo_feature_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        local_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=config.pseudo_local_layers)
        global_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(global_layer, num_layers=config.pseudo_global_layers)
        self.slot_norm = nn.LayerNorm(config.d_model)
        self.cell_norm = nn.LayerNorm(config.d_model)
        self.jet_norm = nn.LayerNorm(config.d_model)
        self.view_embedding = nn.Parameter(torch.empty(config.d_model))
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

    def _enriched_features(self, view: PseudoParticleViewInput) -> torch.Tensor:
        base = view.features.transpose(1, 2).float()
        particles = int(base.shape[1])
        cell_scale = max(int(view.cell_indices.max().item()), 1)
        real_slots = view.slot_indices[view.slot_indices >= 0]
        slot_scale = max(int(real_slots.max().item()) if real_slots.numel() else 1, 1)
        cell_index = view.cell_indices.float()[None].expand(base.shape[0], -1) / float(cell_scale)
        slot_index = view.slot_indices.clamp_min(0).float()[None].expand(base.shape[0], -1) / float(slot_scale)
        uncertainty_available = view.uncertainty_mask.float()
        side = torch.cat(
            (
                view.candidate_weights.unsqueeze(-1),
                view.existence_probability.unsqueeze(-1),
                view.reliability.unsqueeze(-1),
                torch.log1p(view.expected_count.clamp_min(0.0)).unsqueeze(-1),
                view.is_dust.float()[None, :, None].expand(base.shape[0], -1, -1),
                uncertainty_available.unsqueeze(-1),
                view.slot_log_sigma,
                cell_index.unsqueeze(-1),
                slot_index.unsqueeze(-1),
            ),
            dim=-1,
        )
        if side.shape[:2] != (base.shape[0], particles):
            raise ValueError(f"{view.name} side-channel assembly is misaligned")
        return torch.cat((base, side, _signed_log1p(view.parent_accounting)), dim=-1)

    def forward(self, view: PseudoParticleViewInput) -> PseudoStreamEncoding:
        view.validate()
        mask = _mask_2d(view.mask)
        enriched = self._enriched_features(view)
        geometry = torch.cat((view.points.transpose(1, 2), view.lorentz_vectors.transpose(1, 2)), dim=-1)
        hidden = self.feature_projection(self.feature_norm(enriched)) + self.geometry_projection(_signed_log1p(geometry))
        hidden = hidden + self.view_embedding[None, None]
        hidden = hidden * mask.unsqueeze(-1).float()
        cells = int(view.cell_indices.max().item()) + 1
        selected_by_cell = [torch.nonzero(view.cell_indices == cell, as_tuple=False).flatten() for cell in range(cells)]
        counts = {int(selected.numel()) for selected in selected_by_cell}
        if len(counts) != 1:
            raise ValueError(f"{view.name} pseudo layout must have equal slots per cell, got {sorted(counts)}")
        slots_per_cell = next(iter(counts))
        order = torch.cat(selected_by_cell)
        batch = int(hidden.shape[0])
        local = hidden[:, order].reshape(batch * cells, slots_per_cell, self.config.d_model)
        local_mask = mask[:, order].reshape(batch * cells, slots_per_cell)
        safe_local_mask = local_mask.clone()
        empty = ~safe_local_mask.any(dim=1)
        if bool(empty.any()):
            safe_local_mask[empty, 0] = True
            local = local.clone()
            local[empty, 0] = 0.0
        local = self.local_encoder(local, src_key_padding_mask=~safe_local_mask)
        local = self.slot_norm(local) * local_mask.unsqueeze(-1).float()
        candidate = view.candidate_weights[:, order].reshape(batch * cells, slots_per_cell)
        pooling_weight = candidate * local_mask.float()
        fallback = local_mask.float()
        use_fallback = pooling_weight.sum(dim=1, keepdim=True) <= 1.0e-8
        pooling_weight = torch.where(use_fallback, fallback, pooling_weight)
        cell_tokens = _masked_weighted_mean(local, pooling_weight).reshape(batch, cells, self.config.d_model)
        cell_mask = local_mask.reshape(batch, cells, slots_per_cell).any(dim=-1)
        safe_cell_mask = cell_mask.clone()
        empty_jet = ~safe_cell_mask.any(dim=1)
        if bool(empty_jet.any()):
            safe_cell_mask[empty_jet, 0] = True
            cell_tokens = cell_tokens.clone()
            cell_tokens[empty_jet, 0] = 0.0
        cell_tokens = self.global_encoder(cell_tokens, src_key_padding_mask=~safe_cell_mask)
        cell_tokens = self.cell_norm(cell_tokens) * cell_mask.unsqueeze(-1).float()
        reliability = view.reliability[:, order].reshape(batch, cells, slots_per_cell)
        uncertainty = torch.exp(view.slot_log_sigma[:, order].clamp(-8.0, 8.0)).mean(dim=-1)
        uncertainty = uncertainty.reshape(batch, cells, slots_per_cell)
        cell_weight = pooling_weight.reshape(batch, cells, slots_per_cell)
        cell_trust = (reliability * cell_weight).sum(dim=-1) / cell_weight.sum(dim=-1).clamp_min(1.0e-8)
        cell_uncertainty = (uncertainty * cell_weight).sum(dim=-1) / cell_weight.sum(dim=-1).clamp_min(1.0e-8)
        cell_trust = cell_trust * cell_mask.float()
        cell_uncertainty = cell_uncertainty * cell_mask.float()
        jet_weight = cell_trust * cell_mask.float()
        jet = self.jet_norm(_masked_weighted_mean(cell_tokens, jet_weight + 1.0e-4 * cell_mask.float()))
        return PseudoStreamEncoding(
            token_embeddings=cell_tokens,
            token_mask=cell_mask,
            jet_embedding=jet,
            token_trust=cell_trust,
            token_uncertainty=cell_uncertainty,
            slot_embeddings=local.reshape(batch, cells * slots_per_cell, self.config.d_model),
            slot_mask=local_mask.reshape(batch, cells * slots_per_cell),
            diagnostics={
                "view_name": view.name,
                "view_kind": view.view_kind,
                "terminal_level": int(view.terminal_level),
                "num_cells": cells,
                "slots_per_cell": slots_per_cell,
                "active_cell_fraction": cell_mask.float().mean().detach(),
                "mean_cell_trust": cell_trust[cell_mask].mean().detach() if bool(cell_mask.any()) else cell_trust.new_zeros(()),
            },
        )


class _CrossViewLayer(nn.Module):
    def __init__(self, config: FusionTaggerConfig, *, num_views: int, gated: bool) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.gated = bool(gated)
        hidden = int(round(config.ffn_multiplier * config.d_model))
        self.hlt_norm = nn.LayerNorm(config.d_model)
        self.pseudo_norms = nn.ModuleList(nn.LayerNorm(config.d_model) for _ in range(num_views))
        self.hlt_to_pseudo = nn.ModuleList(
            nn.MultiheadAttention(
                config.d_model,
                config.num_heads,
                dropout=config.attention_dropout,
                batch_first=True,
            )
            for _ in range(num_views)
        )
        self.pseudo_to_hlt = nn.ModuleList(
            nn.MultiheadAttention(
                config.d_model,
                config.num_heads,
                dropout=config.attention_dropout,
                batch_first=True,
            )
            for _ in range(num_views)
        )
        self.hlt_update_norm = nn.LayerNorm(config.d_model)
        self.pseudo_update_norms = nn.ModuleList(nn.LayerNorm(config.d_model) for _ in range(num_views))
        self.hlt_ffn = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.pseudo_ffns = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, config.d_model),
                nn.Dropout(config.dropout),
            )
            for _ in range(num_views)
        )
        self.token_gate = (
            nn.ModuleList(
                nn.Sequential(
                    nn.LayerNorm(2 * config.d_model + 2),
                    nn.Linear(2 * config.d_model + 2, config.d_model),
                    nn.GELU(),
                    nn.Linear(config.d_model, 1),
                )
                for _ in range(num_views)
            )
            if gated
            else None
        )

    def forward(
        self,
        hlt: torch.Tensor,
        hlt_mask: torch.Tensor,
        pseudo_tokens: Sequence[torch.Tensor],
        pseudo_masks: Sequence[torch.Tensor],
        pseudo_trust: Sequence[torch.Tensor],
        pseudo_uncertainty: Sequence[torch.Tensor],
        view_available: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor, Mapping[str, Any]]:
        hlt_updates = []
        pseudo_updates = []
        gate_logits = []
        attended_trust_rows = []
        attended_uncertainty_rows = []
        normalized_hlt = self.hlt_norm(hlt)
        for index in range(self.num_views):
            normalized_pseudo = self.pseudo_norms[index](pseudo_tokens[index])
            hlt_update, attention = self.hlt_to_pseudo[index](
                normalized_hlt,
                normalized_pseudo,
                normalized_pseudo,
                key_padding_mask=~pseudo_masks[index],
                need_weights=True,
                average_attn_weights=False,
            )
            pseudo_update, _ = self.pseudo_to_hlt[index](
                normalized_pseudo,
                normalized_hlt,
                normalized_hlt,
                key_padding_mask=~hlt_mask,
                need_weights=False,
            )
            mean_attention = attention.mean(dim=1)
            attended_trust = torch.einsum("bqp,bp->bq", mean_attention, pseudo_trust[index])
            attended_uncertainty = torch.einsum("bqp,bp->bq", mean_attention, pseudo_uncertainty[index])
            hlt_updates.append(hlt_update)
            pseudo_updates.append(
                self.pseudo_update_norms[index](pseudo_tokens[index] + pseudo_update)
                + self.pseudo_ffns[index](pseudo_tokens[index] + pseudo_update)
            )
            attended_trust_rows.append(attended_trust)
            attended_uncertainty_rows.append(attended_uncertainty)
            if self.gated:
                gate_logits.append(
                    self.token_gate[index](
                        torch.cat(
                            (
                                hlt,
                                hlt_update,
                                attended_trust.unsqueeze(-1),
                                torch.log1p(attended_uncertainty).unsqueeze(-1),
                            ),
                            dim=-1,
                        )
                    ).squeeze(-1)
                )
        if self.gated:
            stacked_logits = torch.stack(gate_logits, dim=-1)
            stacked_logits = stacked_logits.masked_fill(~view_available[:, None, :], -1.0e4)
            skip_logits = torch.zeros_like(stacked_logits[..., :1])
            normalized_gates = torch.softmax(torch.cat((skip_logits, stacked_logits), dim=-1), dim=-1)[..., 1:]
        else:
            denominator = view_available.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=hlt.dtype)
            normalized_gates = view_available[:, None, :].to(dtype=hlt.dtype) / denominator[:, None, :]
            normalized_gates = normalized_gates.expand(-1, hlt.shape[1], -1)
        stacked_updates = torch.stack(hlt_updates, dim=-2)
        combined_update = (stacked_updates * normalized_gates.unsqueeze(-1)).sum(dim=-2)
        next_hlt = self.hlt_update_norm(hlt + combined_update)
        next_hlt = (next_hlt + self.hlt_ffn(next_hlt)) * hlt_mask.unsqueeze(-1).float()
        next_pseudo = tuple(
            value * pseudo_masks[index].unsqueeze(-1).float()
            for index, value in enumerate(pseudo_updates)
        )
        return next_hlt, next_pseudo, normalized_gates, {
            "attended_trust": torch.stack(attended_trust_rows, dim=-1),
            "attended_uncertainty": torch.stack(attended_uncertainty_rows, dim=-1),
        }


def _fixed_permutation(indices: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(int(indices.numel()), generator=generator)
    return indices[permutation.to(device=indices.device)]


def _rebuild_view_from_raw(view: PseudoParticleViewInput, raw: torch.Tensor) -> PseudoParticleViewInput:
    mask = _mask_2d(view.mask)
    inputs = build_part_inputs_torch(raw, mask, max_constits=int(raw.shape[1]))
    result = replace(
        view,
        raw_tokens=raw,
        points=inputs["points"],
        features=inputs["features"],
        lorentz_vectors=inputs["lorentz_vectors"],
        mask=inputs["mask"],
    )
    result.validate()
    return result


def apply_fusion_control(
    view: PseudoParticleViewInput,
    control: str,
    *,
    seed: int,
) -> PseudoParticleViewInput:
    """Apply one declared E-tier intervention without changing unrelated channels."""

    if control == CONTROL_NONE:
        return view
    if control == CONTROL_NO_UNCERTAINTY:
        return replace(
            view,
            candidate_weights=view.existence_probability * _mask_2d(view.mask).float(),
            reliability=torch.ones_like(view.reliability),
            slot_log_sigma=torch.zeros_like(view.slot_log_sigma),
            uncertainty_mask=torch.zeros_like(view.uncertainty_mask),
        )
    if control == CONTROL_SHUFFLED_CELLS:
        cells = int(view.cell_indices.max().item()) + 1
        remapped_cells = torch.remainder(view.cell_indices + 1, cells)
        parent_by_cell = torch.stack(
            [
                view.parent_accounting[:, torch.nonzero(view.cell_indices == cell, as_tuple=False)[0, 0]]
                for cell in range(cells)
            ],
            dim=1,
        )
        return replace(
            view,
            cell_indices=remapped_cells,
            parent_accounting=parent_by_cell[:, remapped_cells],
        )
    raw = view.raw_tokens.clone()
    for cell in range(int(view.cell_indices.max().item()) + 1):
        selected = torch.nonzero(view.cell_indices == cell, as_tuple=False).flatten()
        if int(selected.numel()) <= 1:
            continue
        source = _fixed_permutation(selected, int(seed) + 97 * cell)
        if control == CONTROL_RANDOM_COORDINATES:
            raw[:, selected, 1:3] = view.raw_tokens[:, source, 1:3]
        elif control == CONTROL_SHUFFLED_COMPOSITION:
            raw[:, selected, 4:10] = view.raw_tokens[:, source, 4:10]
        else:
            raise ValueError(f"unsupported fusion control {control!r}")
    return _rebuild_view_from_raw(view, raw)


class _CapacityResidualBank(nn.Module):
    """Use an exact number of additional HLT-only parameters as residual refiners."""

    def __init__(self, d_model: int, parameter_count: int) -> None:
        super().__init__()
        if int(parameter_count) < 0:
            raise ValueError("capacity residual parameter_count must be nonnegative")
        self.d_model = int(d_model)
        self.parameter_count = int(parameter_count)
        self.bank = nn.Parameter(torch.zeros(self.parameter_count))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.parameter_count == 0:
            return value
        cursor = 0
        block_size = self.d_model * self.d_model + self.d_model
        while self.parameter_count - cursor >= block_size:
            weight = self.bank[cursor : cursor + self.d_model * self.d_model].reshape(
                self.d_model, self.d_model
            )
            cursor += self.d_model * self.d_model
            bias = self.bank[cursor : cursor + self.d_model]
            cursor += self.d_model
            value = value + torch.nn.functional.gelu(torch.nn.functional.linear(value, weight, bias))
        remaining = self.bank[cursor:]
        if int(remaining.numel()) >= self.d_model:
            value = value + remaining[: self.d_model]
            remaining = remaining[self.d_model :]
        if int(remaining.numel()):
            value = value + torch.tanh(remaining.mean()) * torch.tanh(value)
        return value


class ConstrainedDualStreamTagger(nn.Module):
    """D/E-tier tagger with an ungated HLT identity path and structured pseudo streams."""

    def __init__(self, config: FusionTaggerConfig) -> None:
        super().__init__()
        self.config = config
        self.spec = config.variant_spec
        hlt_config = _encoder_config(
            config,
            feature_dim=config.feature_dim,
            encoder_layers=config.hlt_encoder_layers,
        )
        self.hlt_encoder = ParTStyleHLTEncoder(hlt_config)
        self.view_names = config.resolved_view_names
        self.pseudo_encoders = nn.ModuleDict(
            {
                name: HierarchicalPseudoStreamEncoder(config, view_index=index)
                for index, name in enumerate(self.view_names)
            }
        )
        self.shadow_hlt_encoder = (
            ParTStyleHLTEncoder(hlt_config)
            if self.spec.architecture == ARCH_HLT_CAPACITY_CONTROL
            else None
        )
        self.cross_layers = nn.ModuleList(
            _CrossViewLayer(
                config,
                num_views=len(self.view_names),
                gated=self.spec.architecture == ARCH_GATED_CROSS_ATTENTION,
            )
            for _ in range(config.fusion_layers)
        )
        self.pooled_gate = (
            nn.ModuleList(
                nn.Sequential(
                    nn.LayerNorm(2 * config.d_model + 2),
                    nn.Linear(2 * config.d_model + 2, config.d_model),
                    nn.GELU(),
                    nn.Linear(config.d_model, 1),
                )
                for _ in self.view_names
            )
            if self.spec.architecture == ARCH_GATED_CROSS_ATTENTION
            else None
        )
        self.hlt_head = nn.Linear(config.d_model, config.num_classes)
        self.pseudo_head = nn.Linear(config.d_model, config.num_classes)
        if self.spec.architecture == ARCH_REPRESENTATION or self.spec.architecture == ARCH_HLT_CAPACITY_CONTROL:
            classifier_dim = 4 * config.d_model
        elif self.spec.architecture in {ARCH_CROSS_ATTENTION, ARCH_GATED_CROSS_ATTENTION}:
            classifier_dim = 5 * config.d_model
        else:
            classifier_dim = config.d_model
        hidden = 2 * config.d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes),
        )
        self.capacity_residual = None
        self.parameter_match_reference = None
        if self.spec.architecture == ARCH_HLT_CAPACITY_CONTROL:
            with torch.random.fork_rng(devices=[]):
                reference = ConstrainedDualStreamTagger(
                    replace(config, variant=D4_UNCERTAINTY_GATED, view_names=("canonical",))
                )
            target = sum(parameter.numel() for parameter in reference.parameters())
            del reference
            current = sum(parameter.numel() for parameter in self.parameters())
            self.capacity_residual = _CapacityResidualBank(config.d_model, target - current)
            self.parameter_match_reference = D4_UNCERTAINTY_GATED

    def _validate_views(self, pseudo_views: Sequence[PseudoParticleViewInput]) -> tuple[PseudoParticleViewInput, ...]:
        if self.spec.architecture == ARCH_HLT_CAPACITY_CONTROL:
            if pseudo_views:
                raise ValueError("E6 must not receive pseudo views")
            return ()
        if len(pseudo_views) != len(self.view_names):
            raise ValueError(
                f"{self.config.variant} expects {len(self.view_names)} pseudo views {self.view_names}, "
                f"got {len(pseudo_views)}"
            )
        by_name = {view.name: view for view in pseudo_views}
        if set(by_name) != set(self.view_names):
            raise ValueError(f"pseudo view names {tuple(by_name)} do not match configured names {self.view_names}")
        ordered = tuple(by_name[name] for name in self.view_names)
        if self.spec.requires_grid_tokens and any(view.view_kind != "grid" for view in ordered):
            raise ValueError("D7 requires grid-token input")
        if not self.spec.requires_grid_tokens and any(view.view_kind != "particle" for view in ordered):
            raise ValueError(f"{self.config.variant} requires pseudo-particle input")
        return tuple(
            apply_fusion_control(view, self.spec.control, seed=self.config.control_seed + index)
            for index, view in enumerate(ordered)
        )

    def _view_availability(self, batch: int, device: torch.device) -> torch.Tensor:
        available = torch.ones(batch, len(self.view_names), dtype=torch.bool, device=device)
        if self.training and self.config.pseudo_view_dropout > 0.0 and len(self.view_names):
            available = torch.rand(batch, len(self.view_names), device=device) >= self.config.pseudo_view_dropout
        return available

    @staticmethod
    def _mean_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _masked_weighted_mean(tokens, mask.float())

    def forward_detailed(
        self,
        hlt: ParticleStreamInput,
        pseudo_views: Sequence[PseudoParticleViewInput] = (),
    ) -> FusionTaggerOutput:
        views = self._validate_views(tuple(pseudo_views))
        hlt_output = self.hlt_encoder(hlt.points, hlt.features, hlt.lorentz_vectors, hlt.mask)
        hlt_tokens = hlt_output.particle_embeddings
        hlt_mask = hlt_output.particle_mask
        zero = hlt_tokens.new_zeros(())
        if self.spec.architecture == ARCH_HLT_CAPACITY_CONTROL:
            shadow = self.shadow_hlt_encoder(hlt.points, hlt.features, hlt.lorentz_vectors, hlt.mask)
            shadow_jet = self.capacity_residual(shadow.jet_embedding)
            fused = torch.cat(
                (
                    hlt_output.jet_embedding,
                    shadow_jet,
                    torch.abs(hlt_output.jet_embedding - shadow_jet),
                    hlt_output.jet_embedding * shadow_jet,
                ),
                dim=-1,
            )
            logits = self.classifier(fused)
            return FusionTaggerOutput(
                logits=logits,
                hlt_logits=self.hlt_head(hlt_output.jet_embedding),
                pseudo_logits=None,
                hlt_representation=hlt_output.jet_embedding,
                pseudo_representations={},
                token_gates=None,
                pooled_gates=None,
                gate_entropy_regularizer=zero,
                diagnostics={
                    "contract": DUAL_STREAM_FUSION_CONTRACT,
                    "variant": self.config.variant,
                    "hlt_skip_ungated": True,
                    "parameter_match_reference": self.parameter_match_reference,
                    "capacity_residual_parameter_count": self.capacity_residual.parameter_count,
                },
            )
        encoded = tuple(self.pseudo_encoders[name](view) for name, view in zip(self.view_names, views))
        pseudo_jets = torch.stack([row.jet_embedding for row in encoded], dim=1)
        pseudo_mean = pseudo_jets.mean(dim=1)
        hlt_logits = self.hlt_head(hlt_output.jet_embedding)
        pseudo_logits = self.pseudo_head(pseudo_mean)
        if self.spec.architecture == ARCH_PSEUDO_ONLY:
            logits = self.classifier(pseudo_mean)
            return self._simple_output(logits, hlt_logits, pseudo_logits, hlt_output, encoded, zero)
        if self.spec.architecture == ARCH_LATE_LOGIT:
            logits = 0.5 * (hlt_logits + pseudo_logits)
            return self._simple_output(logits, hlt_logits, pseudo_logits, hlt_output, encoded, zero)
        if self.spec.architecture == ARCH_REPRESENTATION:
            fused = torch.cat(
                (
                    hlt_output.jet_embedding,
                    pseudo_mean,
                    torch.abs(hlt_output.jet_embedding - pseudo_mean),
                    hlt_output.jet_embedding * pseudo_mean,
                ),
                dim=-1,
            )
            logits = self.classifier(fused)
            return self._simple_output(logits, hlt_logits, pseudo_logits, hlt_output, encoded, zero)
        pseudo_tokens = tuple(row.token_embeddings for row in encoded)
        pseudo_masks = tuple(row.token_mask for row in encoded)
        view_available = self._view_availability(hlt.batch_size, hlt_tokens.device)
        token_gates = None
        layer_diagnostics = []
        for layer in self.cross_layers:
            hlt_tokens, pseudo_tokens, token_gates, diagnostics = layer(
                hlt_tokens,
                hlt_mask,
                pseudo_tokens,
                pseudo_masks,
                tuple(row.token_trust for row in encoded),
                tuple(row.token_uncertainty for row in encoded),
                view_available,
            )
            layer_diagnostics.append(diagnostics)
        fused_hlt = hlt_output.jet_embedding + self._mean_tokens(hlt_tokens, hlt_mask)
        updated_pseudo_jets = torch.stack(
            [self._mean_tokens(tokens, mask) for tokens, mask in zip(pseudo_tokens, pseudo_masks)],
            dim=1,
        )
        mean_trust = torch.stack(
            [
                (row.token_trust * row.token_mask.float()).sum(dim=1)
                / row.token_mask.float().sum(dim=1).clamp_min(1.0)
                for row in encoded
            ],
            dim=1,
        )
        mean_uncertainty = torch.stack(
            [
                (row.token_uncertainty * row.token_mask.float()).sum(dim=1)
                / row.token_mask.float().sum(dim=1).clamp_min(1.0)
                for row in encoded
            ],
            dim=1,
        )
        if self.pooled_gate is not None:
            pooled_logits = torch.stack(
                [
                    gate(
                        torch.cat(
                            (
                                hlt_output.jet_embedding,
                                updated_pseudo_jets[:, index],
                                mean_trust[:, index : index + 1],
                                torch.log1p(mean_uncertainty[:, index : index + 1]),
                            ),
                            dim=-1,
                        )
                    ).squeeze(-1)
                    for index, gate in enumerate(self.pooled_gate)
                ],
                dim=-1,
            )
            pooled_logits = pooled_logits.masked_fill(~view_available, -1.0e4)
            pooled_gates = torch.softmax(
                torch.cat((torch.zeros_like(pooled_logits[:, :1]), pooled_logits), dim=-1),
                dim=-1,
            )[:, 1:]
        else:
            pooled_gates = view_available.float() / view_available.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_pseudo = (updated_pseudo_jets * pooled_gates.unsqueeze(-1)).sum(dim=1)
        fused = torch.cat(
            (
                hlt_output.jet_embedding,
                fused_hlt,
                pooled_pseudo,
                torch.abs(fused_hlt - pooled_pseudo),
                fused_hlt * pooled_pseudo,
            ),
            dim=-1,
        )
        logits = self.classifier(fused)
        gate_probabilities = pooled_gates.clamp_min(1.0e-8)
        gate_entropy = -(gate_probabilities * gate_probabilities.log()).sum(dim=-1).mean()
        return FusionTaggerOutput(
            logits=logits,
            hlt_logits=hlt_logits,
            pseudo_logits=pseudo_logits,
            hlt_representation=fused_hlt,
            pseudo_representations={name: updated_pseudo_jets[:, index] for index, name in enumerate(self.view_names)},
            token_gates=token_gates,
            pooled_gates=pooled_gates,
            gate_entropy_regularizer=-gate_entropy,
            diagnostics={
                "contract": DUAL_STREAM_FUSION_CONTRACT,
                "variant": self.config.variant,
                "architecture": self.spec.architecture,
                "hlt_skip_ungated": True,
                "view_names": self.view_names,
                "view_available": view_available,
                "pooled_gate_mean": pooled_gates.mean(dim=0).detach(),
                "mean_view_trust": mean_trust.detach(),
                "mean_view_uncertainty": mean_uncertainty.detach(),
                "pseudo_streams": {name: row.diagnostics for name, row in zip(self.view_names, encoded)},
                "cross_layers": layer_diagnostics,
            },
        )

    def _simple_output(
        self,
        logits: torch.Tensor,
        hlt_logits: torch.Tensor,
        pseudo_logits: torch.Tensor,
        hlt_output: HLTEncoderOutput,
        encoded: Sequence[PseudoStreamEncoding],
        zero: torch.Tensor,
    ) -> FusionTaggerOutput:
        return FusionTaggerOutput(
            logits=logits,
            hlt_logits=hlt_logits,
            pseudo_logits=pseudo_logits,
            hlt_representation=hlt_output.jet_embedding,
            pseudo_representations={name: row.jet_embedding for name, row in zip(self.view_names, encoded)},
            token_gates=None,
            pooled_gates=None,
            gate_entropy_regularizer=zero,
            diagnostics={
                "contract": DUAL_STREAM_FUSION_CONTRACT,
                "variant": self.config.variant,
                "architecture": self.spec.architecture,
                "hlt_skip_ungated": self.spec.architecture != ARCH_PSEUDO_ONLY,
                "view_names": self.view_names,
                "pseudo_streams": {name: row.diagnostics for name, row in zip(self.view_names, encoded)},
            },
        )

    def forward(
        self,
        hlt: ParticleStreamInput,
        pseudo_views: Sequence[PseudoParticleViewInput] = (),
    ) -> torch.Tensor:
        return self.forward_detailed(hlt, pseudo_views).logits


def build_dual_stream_fusion_tagger(
    variant: str,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ConstrainedDualStreamTagger:
    return ConstrainedDualStreamTagger(FusionTaggerConfig(variant=variant, **dict(overrides or {})))


__all__ = [
    "ARCH_CROSS_ATTENTION",
    "ARCH_GATED_CROSS_ATTENTION",
    "ARCH_HLT_CAPACITY_CONTROL",
    "ARCH_LATE_LOGIT",
    "ARCH_PSEUDO_ONLY",
    "ARCH_REPRESENTATION",
    "CONTROL_NONE",
    "CONTROL_NO_UNCERTAINTY",
    "CONTROL_RANDOM_COORDINATES",
    "CONTROL_SHUFFLED_CELLS",
    "CONTROL_SHUFFLED_COMPOSITION",
    "D0_PSEUDO_ONLY",
    "D1_LATE_LOGIT_FUSION",
    "D2_REPRESENTATION_FUSION",
    "D3_CROSS_ATTENTION",
    "D4_UNCERTAINTY_GATED",
    "D5_END_TO_END",
    "D5_B1",
    "D5_B2",
    "D5_B3",
    "D6_MULTIVIEW",
    "D7_GRID_ONLY",
    "D8_MULTIDEPTH",
    "D_TIER_VARIANTS",
    "DUAL_STREAM_FUSION_CONTRACT",
    "E0_SHUFFLED_CELLS",
    "E1_RANDOM_COORDINATES",
    "E2_SHUFFLED_COMPOSITION",
    "E3_NO_UNCERTAINTY",
    "E4_UNCONSTRAINED_SOURCE",
    "E5_NO_SLOT_LOSS_SOURCE",
    "E6_CAPACITY_MATCHED_HLT",
    "E_TIER_VARIANTS",
    "FUSION_VARIANTS",
    "FusionTaggerConfig",
    "FusionTaggerOutput",
    "FusionVariantSpec",
    "HierarchicalPseudoStreamEncoder",
    "PSEUDO_SIDE_FEATURE_NAMES",
    "PSEUDO_VIEW_INPUT_CONTRACT",
    "ParticleStreamInput",
    "PseudoParticleViewInput",
    "PseudoStreamEncoding",
    "ConstrainedDualStreamTagger",
    "apply_fusion_control",
    "build_dual_stream_fusion_tagger",
    "fusion_variant_spec",
    "grid_view_from_arrays",
    "normalize_fusion_variant",
    "particle_stream_from_tokens",
    "pseudo_particle_views_from_arrays",
]

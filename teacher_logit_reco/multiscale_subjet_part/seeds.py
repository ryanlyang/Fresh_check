"""Deterministic seed proposals for multi-scale soft subjet assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .features import (
    MULTISCALE_SUBJET_FEATURE_CONTRACT,
    MultiscaleSubjetFeatureConfig,
    SubjetScaleSpec,
    build_prepared_subjet_inputs,
    default_subjet_scale_specs,
    local_density_features,
    pairwise_delta_r,
)


MULTISCALE_SUBJET_SEED_CONTRACT = "multiscale_subjet_seed_builder_v1"
MULTISCALE_SUBJET_SEED_BUILDER_STEP = "multiscale_subjet_part_step3_seed_builder"
MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD = "leading_pt"
MULTISCALE_SUBJET_LOCAL_DENSITY_SEED_METHOD = "local_density"
MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD = "farthest_point"
MULTISCALE_SUBJET_SEED_METHODS = (
    MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD,
    MULTISCALE_SUBJET_LOCAL_DENSITY_SEED_METHOD,
    MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD,
)


def _default_scale_specs() -> tuple[SubjetScaleSpec, ...]:
    return default_subjet_scale_specs()


@dataclass(frozen=True)
class SubjetSeedBuilderConfig:
    """Configuration for deterministic small/medium/large seed proposals."""

    scale_specs: tuple[SubjetScaleSpec, ...] = field(default_factory=_default_scale_specs)
    method_by_scale: Mapping[str, str] | tuple[tuple[str, str], ...] | None = None
    density_pt_weight: float = 1.0
    include_self_in_density: bool = False
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        specs = tuple(self.scale_specs)
        if not specs:
            raise ValueError("scale_specs must be non-empty")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"scale names must be unique, got {names}")
        if any(int(spec.num_tokens) <= 0 for spec in specs):
            raise ValueError("each scale spec must request at least one token")
        raw_methods = {} if self.method_by_scale is None else dict(self.method_by_scale)
        unknown_names = set(raw_methods).difference(names)
        if unknown_names:
            raise ValueError(f"method_by_scale contains unknown scale names: {sorted(unknown_names)}")
        normalized_methods = {}
        for scale_name, method in raw_methods.items():
            normalized = _normalize_seed_method(method)
            if normalized not in MULTISCALE_SUBJET_SEED_METHODS:
                raise ValueError(f"unknown seed selection method {method!r}")
            normalized_methods[str(scale_name)] = normalized
        density_pt_weight = float(self.density_pt_weight)
        eps = float(self.eps)
        if not math.isfinite(density_pt_weight) or density_pt_weight < 0.0:
            raise ValueError("density_pt_weight must be non-negative and finite")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        object.__setattr__(self, "scale_specs", specs)
        object.__setattr__(self, "method_by_scale", tuple(sorted(normalized_methods.items())))
        object.__setattr__(self, "density_pt_weight", density_pt_weight)
        object.__setattr__(self, "include_self_in_density", bool(self.include_self_in_density))
        object.__setattr__(self, "eps", eps)

    @property
    def total_num_seeds(self) -> int:
        return int(sum(int(spec.num_tokens) for spec in self.scale_specs))


@dataclass(frozen=True)
class SubjetSeedOutput:
    """Seed particles and geometry centers for all subjet scales."""

    centers: Any
    mask: Any
    indices: Any
    scale_index: Any
    scale_radius: Any
    seed_tokens: Any
    seed_pt_fraction: Any
    scale_names: tuple[str, ...]
    selection_methods: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    @property
    def total_num_seeds(self) -> int:
        return int(self.mask.shape[1])

    def summary(self) -> dict[str, Any]:
        return {
            "contract": MULTISCALE_SUBJET_SEED_CONTRACT,
            "feature_contract": MULTISCALE_SUBJET_FEATURE_CONTRACT,
            "centers_shape": list(self.centers.shape),
            "mask_shape": list(self.mask.shape),
            "indices_shape": list(self.indices.shape),
            "total_num_seeds": self.total_num_seeds,
            "valid_seed_count": int(self.mask.sum().detach().cpu().item()),
            "scale_names": list(self.scale_names),
            "selection_methods": list(self.selection_methods),
            "diagnostics": dict(self.diagnostics),
        }


def normalize_seed_builder_config(
    config: SubjetSeedBuilderConfig | Mapping[str, Any] | None = None,
) -> SubjetSeedBuilderConfig:
    if config is None:
        return SubjetSeedBuilderConfig()
    if isinstance(config, SubjetSeedBuilderConfig):
        return config
    return SubjetSeedBuilderConfig(**dict(config))


def _normalize_seed_method(method: str) -> str:
    normalized = str(method).strip().lower().replace("-", "_")
    if normalized in {"pt", "leading", "leadingpt", "leading_p_t"}:
        return MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD
    if normalized in {"density", "local"}:
        return MULTISCALE_SUBJET_LOCAL_DENSITY_SEED_METHOD
    if normalized in {"farthest", "fps", "farthest_point_sampling"}:
        return MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD
    return normalized


def _selection_method_for_scale(scale: SubjetScaleSpec, config: SubjetSeedBuilderConfig) -> str:
    configured = dict(config.method_by_scale)
    if str(scale.name) in configured:
        return configured[str(scale.name)]
    name = str(scale.name).lower()
    if "small" in name:
        return MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD
    if "medium" in name:
        return MULTISCALE_SUBJET_LOCAL_DENSITY_SEED_METHOD
    if "large" in name:
        return MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD
    return MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD


def _topk_indices(scores: Any, mask: Any, k: int) -> tuple[Any, Any]:
    torch = require_torch()
    k = int(k)
    batch_size, num_particles = mask.shape
    take = min(k, int(num_particles))
    if take <= 0:
        indices = torch.zeros((batch_size, k), dtype=torch.long, device=mask.device)
        valid = torch.zeros((batch_size, k), dtype=torch.bool, device=mask.device)
        return indices, valid
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    values, indices = torch.topk(masked_scores, k=take, dim=1, largest=True, sorted=True)
    valid = torch.isfinite(values) & mask.gather(1, indices)
    if take < k:
        pad = k - take
        indices = torch.cat([indices, torch.zeros((batch_size, pad), dtype=torch.long, device=mask.device)], dim=1)
        valid = torch.cat([valid, torch.zeros((batch_size, pad), dtype=torch.bool, device=mask.device)], dim=1)
    return indices, valid


def _farthest_point_indices(coords: Any, mask: Any, pt: Any, k: int) -> tuple[Any, Any]:
    torch = require_torch()
    k = int(k)
    batch_size, num_particles = mask.shape
    out_indices = torch.zeros((batch_size, k), dtype=torch.long, device=mask.device)
    out_valid = torch.zeros((batch_size, k), dtype=torch.bool, device=mask.device)
    if k <= 0 or num_particles <= 0:
        return out_indices, out_valid
    distances = pairwise_delta_r(coords)
    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        if int(valid_indices.numel()) == 0:
            continue
        batch_pt = pt[batch_index, valid_indices]
        first_local = int(torch.argmax(batch_pt).detach().cpu().item())
        selected = [int(valid_indices[first_local].detach().cpu().item())]
        out_indices[batch_index, 0] = selected[0]
        out_valid[batch_index, 0] = True
        available = mask[batch_index].clone()
        available[selected[0]] = False
        for slot in range(1, k):
            candidates = torch.nonzero(available, as_tuple=False).flatten()
            if int(candidates.numel()) == 0:
                break
            selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=mask.device)
            min_distance = distances[batch_index, candidates][:, selected_tensor].min(dim=1).values
            score = min_distance + 1.0e-6 * pt[batch_index, candidates]
            next_index = int(candidates[int(torch.argmax(score).detach().cpu().item())].detach().cpu().item())
            selected.append(next_index)
            out_indices[batch_index, slot] = next_index
            out_valid[batch_index, slot] = True
            available[next_index] = False
    return out_indices, out_valid


def _gather_rows(values: Any, indices: Any, valid: Any) -> Any:
    torch = require_torch()
    if int(values.ndim) == 2:
        gathered = torch.gather(values, 1, torch.clamp(indices, min=0))
        return torch.where(valid, gathered, torch.zeros_like(gathered))
    if int(values.ndim) != 3:
        raise ValueError(f"values must have rank 2 or 3, got {int(values.ndim)}")
    safe_indices = torch.clamp(indices, min=0)[:, :, None].expand(-1, -1, int(values.shape[-1]))
    gathered = torch.gather(values, 1, safe_indices)
    return torch.where(valid[:, :, None], gathered, torch.zeros_like(gathered))


def _duplicate_valid_seed_fraction(indices: Any, valid: Any) -> float:
    torch = require_torch()
    if int(indices.numel()) == 0:
        return 0.0
    duplicate_count = 0
    valid_count = 0
    for batch_index in range(int(indices.shape[0])):
        row = indices[batch_index][valid[batch_index]]
        valid_count += int(row.numel())
        if int(row.numel()) <= 1:
            continue
        unique = torch.unique(row)
        duplicate_count += int(row.numel()) - int(unique.numel())
    if valid_count <= 0:
        return 0.0
    return float(duplicate_count) / float(valid_count)


def build_multiscale_subjet_seeds(
    tokens: Any,
    mask: Any,
    *,
    config: SubjetSeedBuilderConfig | Mapping[str, Any] | None = None,
    feature_config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None = None,
) -> SubjetSeedOutput:
    """Build deterministic seed particles for each subjet scale."""

    torch = require_torch()
    seed_config = normalize_seed_builder_config(config)
    prepared = build_prepared_subjet_inputs(tokens, mask, config=feature_config)
    centers_by_scale = []
    mask_by_scale = []
    indices_by_scale = []
    tokens_by_scale = []
    pt_fraction_by_scale = []
    scale_indices: list[int] = []
    scale_radii: list[float] = []
    scale_names: list[str] = []
    selection_methods: list[str] = []
    per_scale_valid_mean: dict[str, float] = {}

    for scale_idx, scale in enumerate(seed_config.scale_specs):
        method = _selection_method_for_scale(scale, seed_config)
        k = int(scale.num_tokens)
        if method == MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD:
            indices, valid = _topk_indices(prepared.pt, prepared.mask, k)
        elif method == MULTISCALE_SUBJET_LOCAL_DENSITY_SEED_METHOD:
            radius = max(float(scale.radius_center), float(seed_config.eps))
            density = local_density_features(
                prepared.tokens,
                prepared.mask,
                radii=(radius,),
                include_self=bool(seed_config.include_self_in_density),
                config=prepared.config,
            )
            score = density.counts[:, :, 0] + float(seed_config.density_pt_weight) * density.pt_fraction_sums[:, :, 0]
            indices, valid = _topk_indices(score, prepared.mask, k)
        elif method == MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD:
            indices, valid = _farthest_point_indices(prepared.coordinates, prepared.mask, prepared.pt, k)
        else:
            raise ValueError(f"unknown seed selection method {method!r}")

        centers_by_scale.append(_gather_rows(prepared.coordinates, indices, valid))
        tokens_by_scale.append(_gather_rows(prepared.tokens, indices, valid))
        pt_fraction_by_scale.append(_gather_rows(prepared.pt_fraction, indices, valid))
        mask_by_scale.append(valid)
        indices_by_scale.append(torch.where(valid, indices, torch.full_like(indices, -1)))
        scale_indices.extend([scale_idx] * k)
        scale_radii.extend([float(scale.radius_center)] * k)
        scale_names.extend([str(scale.name)] * k)
        selection_methods.extend([method] * k)
        per_scale_valid_mean[f"{scale.name}_valid_seed_mean"] = float(valid.float().mean().detach().cpu().item())

    seed_mask = torch.cat(mask_by_scale, dim=1)
    seed_indices = torch.cat(indices_by_scale, dim=1)
    valid_counts = seed_mask.sum(dim=1)
    method_valid_counts = {}
    for method in MULTISCALE_SUBJET_SEED_METHODS:
        method_slots = torch.as_tensor(
            [slot_method == method for slot_method in selection_methods],
            dtype=torch.bool,
            device=prepared.tokens.device,
        )
        if int(method_slots.numel()) == 0:
            method_valid_counts[method] = 0
        else:
            method_valid_counts[method] = int((seed_mask & method_slots[None, :]).sum().detach().cpu().item())
    return SubjetSeedOutput(
        centers=torch.cat(centers_by_scale, dim=1),
        mask=seed_mask,
        indices=seed_indices,
        scale_index=torch.as_tensor(scale_indices, dtype=torch.long, device=prepared.tokens.device),
        scale_radius=torch.as_tensor(scale_radii, dtype=prepared.tokens.dtype, device=prepared.tokens.device),
        seed_tokens=torch.cat(tokens_by_scale, dim=1),
        seed_pt_fraction=torch.cat(pt_fraction_by_scale, dim=1),
        scale_names=tuple(scale_names),
        selection_methods=tuple(selection_methods),
        diagnostics={
            "step": MULTISCALE_SUBJET_SEED_BUILDER_STEP,
            "contract": MULTISCALE_SUBJET_SEED_CONTRACT,
            "valid_seed_fraction": float(seed_mask.float().mean().detach().cpu().item()),
            "valid_seed_count_mean": float(valid_counts.float().mean().detach().cpu().item()),
            "valid_seed_count_min": int(valid_counts.min().detach().cpu().item()) if int(valid_counts.numel()) else 0,
            "valid_seed_count_max": int(valid_counts.max().detach().cpu().item()) if int(valid_counts.numel()) else 0,
            "empty_jet_fraction": float((~prepared.mask.any(dim=1)).float().mean().detach().cpu().item()),
            "empty_jet_count": int((~prepared.mask.any(dim=1)).sum().detach().cpu().item()),
            "duplicate_valid_seed_fraction": _duplicate_valid_seed_fraction(seed_indices, seed_mask),
            "total_num_seeds": int(seed_config.total_num_seeds),
            "method_valid_counts": method_valid_counts,
            **per_scale_valid_mean,
        },
    )

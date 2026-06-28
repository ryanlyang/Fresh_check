"""Differentiable particle-to-subjet soft assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .features import (
    CANONICAL_PART_FEATURE_NAMES,
    MultiscaleSubjetFeatureConfig,
    SubjetScaleSpec,
    build_canonical_part_inputs,
    build_prepared_subjet_inputs,
    default_subjet_scale_specs,
    local_density_features,
    wrap_delta_phi,
)
from .seeds import (
    MULTISCALE_SUBJET_SEED_CONTRACT,
    SubjetSeedBuilderConfig,
    SubjetSeedOutput,
    build_multiscale_subjet_seeds,
    normalize_seed_builder_config,
)


try:  # Keep protocol/config imports cheap on machines without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


MULTISCALE_SUBJET_ASSIGNMENT_CONTRACT = "multiscale_subjet_soft_assignment_v1"
MULTISCALE_SUBJET_ASSIGNMENT_STEP = "multiscale_subjet_part_step4_soft_assignment"
MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED = "seeded"
MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED = "learned"
MULTISCALE_SUBJET_ASSIGNMENT_QUERY_MODES = (
    MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED,
    MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED,
)


def _normalize_assignment_query_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"seed", "seeds", "seed_conditioned", "seed_conditioned_queries"}:
        return MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED
    if normalized in {"latent", "learned_query", "learned_queries", "pure_learned"}:
        return MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED
    return normalized


def _default_scale_specs() -> tuple[SubjetScaleSpec, ...]:
    return default_subjet_scale_specs()


@dataclass(frozen=True)
class SoftSubjetAssignmentConfig:
    """Configuration for the Step 4 soft assignment layer."""

    scale_specs: tuple[SubjetScaleSpec, ...] = field(default_factory=_default_scale_specs)
    query_mode: str = MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED
    embed_dim: int = 64
    hidden_dim: int = 128
    temperature: float = 1.0
    geometry_bias_strength: float = 2.0
    use_scale_embedding: bool = True
    radius_floor: float = 0.03
    dead_token_weight_threshold: float = 1.0e-3
    seed_config: SubjetSeedBuilderConfig | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        specs = tuple(self.scale_specs)
        if not specs:
            raise ValueError("scale_specs must be non-empty")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"scale names must be unique, got {names}")
        mode = _normalize_assignment_query_mode(str(self.query_mode))
        if mode not in MULTISCALE_SUBJET_ASSIGNMENT_QUERY_MODES:
            raise ValueError(f"query_mode must be one of {MULTISCALE_SUBJET_ASSIGNMENT_QUERY_MODES}, got {mode!r}")
        embed_dim = int(self.embed_dim)
        hidden_dim = int(self.hidden_dim)
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be positive")
        temperature = float(self.temperature)
        geometry_bias_strength = float(self.geometry_bias_strength)
        radius_floor = float(self.radius_floor)
        dead_token_weight_threshold = float(self.dead_token_weight_threshold)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        if not math.isfinite(geometry_bias_strength) or geometry_bias_strength < 0.0:
            raise ValueError("geometry_bias_strength must be non-negative and finite")
        if not math.isfinite(radius_floor) or radius_floor <= 0.0:
            raise ValueError("radius_floor must be positive and finite")
        if not math.isfinite(dead_token_weight_threshold) or dead_token_weight_threshold < 0.0:
            raise ValueError("dead_token_weight_threshold must be non-negative and finite")
        seed_config = normalize_seed_builder_config(self.seed_config)
        if tuple(seed_config.scale_specs) != specs:
            seed_config = SubjetSeedBuilderConfig(
                scale_specs=specs,
                method_by_scale=seed_config.method_by_scale,
                density_pt_weight=seed_config.density_pt_weight,
                include_self_in_density=seed_config.include_self_in_density,
                eps=seed_config.eps,
            )
        object.__setattr__(self, "scale_specs", specs)
        object.__setattr__(self, "query_mode", mode)
        object.__setattr__(self, "embed_dim", embed_dim)
        object.__setattr__(self, "hidden_dim", hidden_dim)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "geometry_bias_strength", geometry_bias_strength)
        object.__setattr__(self, "use_scale_embedding", bool(self.use_scale_embedding))
        object.__setattr__(self, "radius_floor", radius_floor)
        object.__setattr__(self, "dead_token_weight_threshold", dead_token_weight_threshold)
        object.__setattr__(self, "seed_config", seed_config)

    @property
    def total_num_subjets(self) -> int:
        return int(sum(int(spec.num_tokens) for spec in self.scale_specs))

    @property
    def num_scales(self) -> int:
        return int(len(self.scale_specs))


@dataclass(frozen=True)
class SoftSubjetAssignmentOutput:
    """Assignment weights and derived soft subjet geometry."""

    assignment_weights: Any
    cluster_weights: Any
    logits: Any
    subjet_mask: Any
    particle_mask: Any
    seed_output: SubjetSeedOutput | None
    query_mode: str
    scale_index: Any
    scale_radius: Any
    estimated_centers: Any
    estimated_pt_fraction: Any
    diagnostics: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "contract": MULTISCALE_SUBJET_ASSIGNMENT_CONTRACT,
            "seed_contract": None if self.seed_output is None else MULTISCALE_SUBJET_SEED_CONTRACT,
            "query_mode": self.query_mode,
            "assignment_weights_shape": list(self.assignment_weights.shape),
            "cluster_weights_shape": list(self.cluster_weights.shape),
            "subjet_mask_shape": list(self.subjet_mask.shape),
            "particle_mask_shape": list(self.particle_mask.shape),
            "estimated_centers_shape": list(self.estimated_centers.shape),
            "diagnostics": dict(self.diagnostics),
        }


def normalize_soft_assignment_config(
    config: SoftSubjetAssignmentConfig | Mapping[str, Any] | None = None,
) -> SoftSubjetAssignmentConfig:
    if config is None:
        return SoftSubjetAssignmentConfig()
    if isinstance(config, SoftSubjetAssignmentConfig):
        return config
    return SoftSubjetAssignmentConfig(**dict(config))


class SoftSubjetAssignment(_ModuleBase):
    """Assign HLT particles softly to multi-scale subjet tokens.

    The serious default is seed-conditioned: deterministic seed particles define
    geometry centers and the learned assignment queries are conditioned on those
    seed tokens.  Pure learned queries are available for capacity controls.
    """

    def __init__(self, config: SoftSubjetAssignmentConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = normalize_soft_assignment_config(config)
        density_dim = 2 * len(MultiscaleSubjetFeatureConfig().default_density_radii)
        self.particle_input_dim = int(RAW_TOKEN_DIM + len(CANONICAL_PART_FEATURE_NAMES) + 2 + 1 + density_dim)
        self.seed_input_dim = int(RAW_TOKEN_DIM + 2 + 1 + 1)
        self.particle_key = torch.nn.Sequential(
            torch.nn.LayerNorm(self.particle_input_dim),
            torch.nn.Linear(self.particle_input_dim, int(self.config.hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.hidden_dim), int(self.config.embed_dim)),
        )
        self.seed_query = torch.nn.Sequential(
            torch.nn.LayerNorm(self.seed_input_dim),
            torch.nn.Linear(self.seed_input_dim, int(self.config.hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.hidden_dim), int(self.config.embed_dim)),
        )
        self.learned_queries = torch.nn.Parameter(
            torch.empty(int(self.config.total_num_subjets), int(self.config.embed_dim))
        )
        self.scale_embedding = torch.nn.Embedding(int(self.config.num_scales), int(self.config.embed_dim))
        torch.nn.init.normal_(self.learned_queries, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.scale_embedding.weight, mean=0.0, std=0.02)
        self.register_buffer(
            "scale_index",
            torch.as_tensor(
                [scale_idx for scale_idx, scale in enumerate(self.config.scale_specs) for _ in range(int(scale.num_tokens))],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "scale_radius",
            torch.as_tensor(
                [float(scale.radius_center) for scale in self.config.scale_specs for _ in range(int(scale.num_tokens))],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _cluster_membership_weights(self, logits: Any, particle_mask: Any, subjet_mask: Any) -> Any:
        """Return membership-style weights normalized over subjets within each scale."""

        torch = require_torch()
        cluster_weights = torch.zeros_like(logits)
        for scale_id in torch.unique(self.scale_index.to(device=logits.device)).tolist():
            scale_slots = self.scale_index.to(device=logits.device) == int(scale_id)
            scale_logits = logits[:, scale_slots, :]
            scale_subjet_mask = subjet_mask[:, scale_slots]
            has_valid_subjet = scale_subjet_mask.any(dim=1, keepdim=True)
            scale_logits = scale_logits.masked_fill(~scale_subjet_mask[:, :, None], -1.0e9)
            weights = torch.softmax(scale_logits, dim=1)
            weights = weights * scale_subjet_mask[:, :, None].to(dtype=logits.dtype)
            weights = weights * particle_mask[:, None, :].to(dtype=logits.dtype)
            weights = weights * has_valid_subjet[:, :, None].to(dtype=logits.dtype)
            weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0e-12)
            weights = torch.where(
                has_valid_subjet[:, :, None] & particle_mask[:, None, :],
                weights,
                torch.zeros_like(weights),
            )
            cluster_weights[:, scale_slots, :] = weights
        return cluster_weights

    def _particle_inputs(
        self,
        tokens: Any,
        mask: Any,
        feature_config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None,
        prepared_inputs: Any | None = None,
        canonical_inputs: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        torch = require_torch()
        prepared = prepared_inputs
        if prepared is None:
            prepared = build_prepared_subjet_inputs(tokens, mask, config=feature_config)
        canonical = canonical_inputs
        if canonical is None:
            canonical = build_canonical_part_inputs(prepared.tokens, prepared.mask, max_constits=prepared.tokens.shape[1])
        density = local_density_features(prepared.tokens, prepared.mask, config=prepared.config)
        particle_features = torch.cat(
            [
                prepared.tokens,
                canonical.feature_rows(),
                prepared.coordinates,
                prepared.pt_fraction[:, :, None],
                density.counts,
                density.pt_fraction_sums,
            ],
            dim=-1,
        )
        return prepared, canonical, particle_features

    def _queries(
        self,
        prepared: Any,
        seed_output: SubjetSeedOutput | None,
        mode: str,
    ) -> tuple[Any, Any, SubjetSeedOutput | None]:
        torch = require_torch()
        batch_size = int(prepared.tokens.shape[0])
        base_queries = self.learned_queries[None, :, :].expand(batch_size, -1, -1)
        if bool(self.config.use_scale_embedding):
            scale_queries = self.scale_embedding(self.scale_index.to(device=prepared.tokens.device))[None, :, :]
        else:
            scale_queries = base_queries.new_zeros(base_queries.shape)
        if mode == MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED:
            subjet_mask = prepared.mask.any(dim=1, keepdim=True).expand(batch_size, int(self.config.total_num_subjets))
            return base_queries + scale_queries, subjet_mask, None
        if seed_output is None:
            seed_output = build_multiscale_subjet_seeds(prepared.tokens, prepared.mask, config=self.config.seed_config)
        if tuple(seed_output.mask.shape) != (batch_size, int(self.config.total_num_subjets)):
            raise ValueError(
                f"seed mask shape {tuple(seed_output.mask.shape)} does not match "
                f"{(batch_size, int(self.config.total_num_subjets))}"
            )
        seed_inputs = torch.cat(
            [
                seed_output.seed_tokens,
                seed_output.centers,
                seed_output.seed_pt_fraction[:, :, None],
                seed_output.scale_radius.to(device=prepared.tokens.device, dtype=prepared.tokens.dtype)[None, :, None].expand(
                    batch_size, -1, -1
                ),
            ],
            dim=-1,
        )
        return base_queries + scale_queries + self.seed_query(seed_inputs), seed_output.mask, seed_output

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
        prepared_inputs: Any | None = None,
        canonical_inputs: Any | None = None,
    ) -> SoftSubjetAssignmentOutput:
        torch = require_torch()
        mode = self.config.query_mode if query_mode is None else _normalize_assignment_query_mode(str(query_mode))
        if mode not in MULTISCALE_SUBJET_ASSIGNMENT_QUERY_MODES:
            raise ValueError(f"query_mode must be one of {MULTISCALE_SUBJET_ASSIGNMENT_QUERY_MODES}, got {mode!r}")
        prepared, _canonical, particle_features = self._particle_inputs(
            tokens,
            mask,
            feature_config,
            prepared_inputs=prepared_inputs,
            canonical_inputs=canonical_inputs,
        )
        particle_mask = prepared.mask
        if particle_assignment_mask is not None:
            extra_particle_mask = torch.as_tensor(particle_assignment_mask, device=prepared.tokens.device).bool()
            if tuple(extra_particle_mask.shape) != tuple(particle_mask.shape):
                raise ValueError(
                    f"particle_assignment_mask shape {tuple(extra_particle_mask.shape)} does not match {tuple(particle_mask.shape)}"
                )
            particle_mask = particle_mask & extra_particle_mask

        queries, subjet_mask, seed_output = self._queries(prepared, seed_output, mode)
        if subjet_assignment_mask is not None:
            extra_subjet_mask = torch.as_tensor(subjet_assignment_mask, device=prepared.tokens.device).bool()
            if tuple(extra_subjet_mask.shape) != tuple(subjet_mask.shape):
                raise ValueError(
                    f"subjet_assignment_mask shape {tuple(extra_subjet_mask.shape)} does not match {tuple(subjet_mask.shape)}"
                )
            subjet_mask = subjet_mask & extra_subjet_mask
        any_assignable_particle = particle_mask.any(dim=1, keepdim=True)
        subjet_mask = subjet_mask & any_assignable_particle

        keys = self.particle_key(particle_features)
        logits = torch.einsum("bme,bne->bmn", queries, keys) / math.sqrt(float(self.config.embed_dim))
        logits = logits / float(self.config.temperature)
        scale_radius = self.scale_radius.to(device=prepared.tokens.device, dtype=prepared.tokens.dtype)
        if mode == MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED and seed_output is not None:
            delta_eta = seed_output.centers[:, :, None, 0] - prepared.coordinates[:, None, :, 0]
            delta_phi = wrap_delta_phi(seed_output.centers[:, :, None, 1] - prepared.coordinates[:, None, :, 1])
            delta_r2 = delta_eta * delta_eta + delta_phi * delta_phi
            radius_source = seed_output.scale_radius.to(device=prepared.tokens.device, dtype=prepared.tokens.dtype)
            radius = torch.clamp(radius_source, min=float(self.config.radius_floor))[None, :, None]
            geometry_bias = -float(self.config.geometry_bias_strength) * delta_r2 / torch.clamp(radius * radius, min=1.0e-8)
            logits = logits + geometry_bias

        logits = logits.masked_fill(~particle_mask[:, None, :], -1.0e9)
        logits = logits.masked_fill(~subjet_mask[:, :, None], -1.0e9)
        weights = torch.softmax(logits, dim=-1) * particle_mask[:, None, :].to(dtype=prepared.tokens.dtype)
        weights = weights * subjet_mask[:, :, None].to(dtype=prepared.tokens.dtype)
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1.0e-12)
        weights = torch.where(subjet_mask[:, :, None], weights, torch.zeros_like(weights))
        cluster_weights = self._cluster_membership_weights(logits, particle_mask, subjet_mask)

        estimated_centers = torch.einsum("bmn,bnd->bmd", weights, prepared.coordinates)
        estimated_pt_fraction = torch.einsum("bmn,bn->bm", weights, prepared.pt_fraction)
        valid_weights = weights.clamp_min(1.0e-12)
        entropy = -(weights * valid_weights.log()).sum(dim=-1)
        max_weight = weights.max(dim=-1).values
        effective_count = 1.0 / torch.clamp((weights * weights).sum(dim=-1), min=1.0e-12)
        valid_subjets = subjet_mask
        valid_count = torch.clamp(valid_subjets.float().sum(), min=1.0)
        dead = (max_weight < float(self.config.dead_token_weight_threshold)) & valid_subjets
        diagnostics = {
            "step": MULTISCALE_SUBJET_ASSIGNMENT_STEP,
            "contract": MULTISCALE_SUBJET_ASSIGNMENT_CONTRACT,
            "query_mode": mode,
            "seeded_default": bool(self.config.query_mode == MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED),
            "geometry_bias_applied": bool(mode == MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED and seed_output is not None),
            "use_scale_embedding": bool(self.config.use_scale_embedding),
            "valid_subjet_fraction": float(valid_subjets.float().mean().detach().cpu().item()),
            "entropy_mean": float((entropy * valid_subjets.float()).sum().detach().cpu().item() / float(valid_count.detach().cpu().item())),
            "max_weight_mean": float((max_weight * valid_subjets.float()).sum().detach().cpu().item() / float(valid_count.detach().cpu().item())),
            "effective_particle_count_mean": float(
                (effective_count * valid_subjets.float()).sum().detach().cpu().item() / float(valid_count.detach().cpu().item())
            ),
            "dead_token_fraction": float(dead.float().sum().detach().cpu().item() / float(valid_count.detach().cpu().item())),
            "cluster_weight_sum_mean": float(
                (cluster_weights.sum(dim=-1) * valid_subjets.float()).sum().detach().cpu().item()
                / float(valid_count.detach().cpu().item())
            ),
            "geometry_bias_strength": float(self.config.geometry_bias_strength),
            "temperature": float(self.config.temperature),
        }
        return SoftSubjetAssignmentOutput(
            assignment_weights=weights,
            cluster_weights=cluster_weights,
            logits=logits,
            subjet_mask=subjet_mask,
            particle_mask=particle_mask,
            seed_output=seed_output,
            query_mode=mode,
            scale_index=self.scale_index.to(device=prepared.tokens.device),
            scale_radius=scale_radius,
            estimated_centers=estimated_centers,
            estimated_pt_fraction=estimated_pt_fraction,
            diagnostics=diagnostics,
        )

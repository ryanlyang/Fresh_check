"""Matching-free privileged per-particle view generator.

The generator consumes contextual HLT query tokens and contextual memory
tokens.  The canonical memory is produced by a frozen offline teacher; the
mandatory HLT-memory controls pass frozen HLT tokens through the same
interface.  No particle matching, nearest-neighbour assignment, transport
plan, or hard radius selection exists in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import canonical_sha256


PARTICLE_VIEW_GENERATOR_CONFIG_CONTRACT = "particle_view_generator_config_v2"
PARTICLE_VIEW_PAIR_FEATURE_CONTRACT = "particle_view_hlt_memory_pair_features_v1"
PARTICLE_VIEW_QUERY_TAP_CONTRACT = "particle_view_contextual_query_tap_v1"
PARTICLE_VIEW_RATE_LOSS_CONTRACT = "particle_view_rate_covariance_losses_v1"

PARTICLE_GEOMETRY_ORDER = ("pt", "eta", "phi", "mass")
PARTICLE_VIEW_PAIR_FEATURE_ORDER = (
    "delta_eta",
    "wrapped_delta_phi",
    "log_delta_r",
    "relative_log_pt",
)
PARTICLE_VIEW_QUERY_SOURCES = (
    "raw",
    "embedding",
    "middle",
    "penultimate",
    "mix3",
)
PARTICLE_VIEW_MEMORY_SOURCES = ("offline", "hlt")
PARTICLE_VIEW_HLT_MEMORY_CONTROL_IDS = (
    "VGEN_MEMORY_HLT",
    "VGEN_MEMORY_HLT_SELFMASK",
)


def _require_bool_mask(
    mask: torch.Tensor,
    *,
    batch: int,
    particles: int,
    name: str,
) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask[:, 0, :]
    if mask.shape != (batch, particles):
        raise ValueError(
            f"{name} must have shape [{batch},{particles}] or "
            f"[{batch},1,{particles}], got {tuple(mask.shape)}"
        )
    if mask.dtype is not torch.bool:
        raise ValueError(f"{name} must be boolean")
    return mask


def _require_tokens(tokens: torch.Tensor, *, name: str) -> tuple[int, int, int]:
    if tokens.ndim != 3:
        raise ValueError(f"{name} must have shape [batch,particles,width]")
    if not tokens.dtype.is_floating_point:
        raise ValueError(f"{name} must be floating point")
    return int(tokens.shape[0]), int(tokens.shape[1]), int(tokens.shape[2])


def _require_geometry(
    geometry: torch.Tensor,
    *,
    batch: int,
    particles: int,
    name: str,
) -> torch.Tensor:
    if geometry.shape != (batch, particles, len(PARTICLE_GEOMETRY_ORDER)):
        raise ValueError(
            f"{name} must have shape [{batch},{particles},4] in "
            f"{PARTICLE_GEOMETRY_ORDER} order"
        )
    if not geometry.dtype.is_floating_point:
        raise ValueError(f"{name} must be floating point")
    return geometry


def wrap_delta_phi(delta_phi: torch.Tensor) -> torch.Tensor:
    """Wrap angular differences to ``[-pi, pi]`` without branch cuts."""

    return torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))


def build_hlt_memory_pair_features(
    query_geometry: torch.Tensor,
    memory_geometry: torch.Tensor,
    *,
    query_mask: torch.Tensor,
    memory_mask: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Build HLT-anchor-to-memory geometric features.

    Geometry rows use the explicit ``(pt, eta, phi, mass)`` order.  Invalid
    pairs are exactly zero and are separately excluded by attention masks.
    """

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    batch, query_particles, _ = query_geometry.shape
    if memory_geometry.ndim != 3 or memory_geometry.shape[0] != batch:
        raise ValueError("query and memory geometry batch dimensions differ")
    memory_particles = int(memory_geometry.shape[1])
    _require_geometry(
        query_geometry,
        batch=batch,
        particles=query_particles,
        name="query_geometry",
    )
    _require_geometry(
        memory_geometry,
        batch=batch,
        particles=memory_particles,
        name="memory_geometry",
    )
    query_mask = _require_bool_mask(
        query_mask,
        batch=batch,
        particles=query_particles,
        name="query_mask",
    )
    memory_mask = _require_bool_mask(
        memory_mask,
        batch=batch,
        particles=memory_particles,
        name="memory_mask",
    )
    query_safe = torch.where(
        query_mask[:, :, None], query_geometry, torch.zeros_like(query_geometry)
    )
    memory_safe = torch.where(
        memory_mask[:, :, None], memory_geometry, torch.zeros_like(memory_geometry)
    )
    query_pt = query_safe[..., 0].clamp_min(epsilon)
    memory_pt = memory_safe[..., 0].clamp_min(epsilon)
    delta_eta = query_safe[:, :, None, 1] - memory_safe[:, None, :, 1]
    delta_phi = wrap_delta_phi(
        query_safe[:, :, None, 2] - memory_safe[:, None, :, 2]
    )
    delta_r = torch.sqrt(delta_eta.square() + delta_phi.square())
    log_delta_r = torch.log(delta_r + epsilon)
    relative_log_pt = (
        torch.log(query_pt)[:, :, None] - torch.log(memory_pt)[:, None, :]
    )
    features = torch.stack(
        (delta_eta, delta_phi, log_delta_r, relative_log_pt), dim=-1
    )
    pair_mask = query_mask[:, :, None] & memory_mask[:, None, :]
    features = torch.where(
        pair_mask[:, :, :, None], features, torch.zeros_like(features)
    )
    if not torch.isfinite(features).all():
        raise ValueError("pair features contain nonfinite values")
    return features


def masked_particle_mean_center(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Subtract each jet's valid-particle mean and zero invalid particles."""

    batch, particles, _ = _require_tokens(values, name="values")
    mask = _require_bool_mask(
        mask, batch=batch, particles=particles, name="mask"
    )
    valid = mask[:, :, None].to(dtype=values.dtype)
    denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (torch.where(mask[:, :, None], values, torch.zeros_like(values)) * valid).sum(
        dim=1, keepdim=True
    ) / denominator
    centered = (values - mean) * valid
    return torch.where(mask[:, :, None], centered, torch.zeros_like(centered))


@dataclass(frozen=True)
class ParticleViewGeneratorConfig:
    query_dim: int
    memory_dim: int
    width: int = 160
    num_heads: int = 8
    num_cross_attention_blocks: int = 2
    feed_forward_expansion: int = 4
    bottleneck_width: int = 4
    pair_feature_dim: int = len(PARTICLE_VIEW_PAIR_FEATURE_ORDER)
    use_pair_bias: bool = True
    use_null_token: bool = True
    center_output: bool = True
    coordinate_dropout_probability: float = 0.10
    coordinate_noise_sigma: float = 0.05
    memory_source: str = "offline"
    self_mask_same_particle: bool = False
    hard_local_radius: float | None = None
    contract: str = PARTICLE_VIEW_GENERATOR_CONFIG_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != PARTICLE_VIEW_GENERATOR_CONFIG_CONTRACT:
            raise ValueError("generator config contract mismatch")
        integer_fields = {
            "query_dim": self.query_dim,
            "memory_dim": self.memory_dim,
            "width": self.width,
            "num_heads": self.num_heads,
            "num_cross_attention_blocks": self.num_cross_attention_blocks,
            "feed_forward_expansion": self.feed_forward_expansion,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")
        if (
            not isinstance(self.bottleneck_width, int)
            or isinstance(self.bottleneck_width, bool)
            or self.bottleneck_width not in {1, 2, 4, 8}
        ):
            raise ValueError("bottleneck_width must be one of 1, 2, 4, or 8")
        if self.pair_feature_dim != len(PARTICLE_VIEW_PAIR_FEATURE_ORDER):
            raise ValueError("pair_feature_dim does not match the locked schema")
        for name in ("use_pair_bias", "use_null_token", "center_output"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name, value in (
            ("coordinate_dropout_probability", self.coordinate_dropout_probability),
            ("coordinate_noise_sigma", self.coordinate_noise_sigma),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.coordinate_dropout_probability >= 1.0:
            raise ValueError("coordinate_dropout_probability must be below one")
        if self.memory_source not in PARTICLE_VIEW_MEMORY_SOURCES:
            raise ValueError(f"memory_source must be one of {PARTICLE_VIEW_MEMORY_SOURCES}")
        if not isinstance(self.self_mask_same_particle, bool):
            raise ValueError("self_mask_same_particle must be boolean")
        if self.self_mask_same_particle and self.memory_source != "hlt":
            raise ValueError("same-particle self masking is only valid for HLT memory")
        if self.hard_local_radius is not None:
            if (
                not isinstance(self.hard_local_radius, (int, float))
                or isinstance(self.hard_local_radius, bool)
                or not math.isfinite(self.hard_local_radius)
                or self.hard_local_radius <= 0.0
            ):
                raise ValueError("hard_local_radius must be positive and finite")
            if float(self.hard_local_radius) not in {0.2, 0.4}:
                raise ValueError(
                    "hard_local_radius is reserved for the 0.2/0.4 diagnostics"
                )

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "query_dim": self.query_dim,
            "memory_dim": self.memory_dim,
            "width": self.width,
            "num_heads": self.num_heads,
            "num_cross_attention_blocks": self.num_cross_attention_blocks,
            "feed_forward_expansion": self.feed_forward_expansion,
            "bottleneck_width": self.bottleneck_width,
            "pair_feature_schema": PARTICLE_VIEW_PAIR_FEATURE_CONTRACT,
            "pair_feature_order": list(PARTICLE_VIEW_PAIR_FEATURE_ORDER),
            "pair_feature_dim": self.pair_feature_dim,
            "use_pair_bias": self.use_pair_bias,
            "use_null_token": self.use_null_token,
            "center_output": self.center_output,
            "coordinate_dropout_probability": self.coordinate_dropout_probability,
            "coordinate_noise_sigma": self.coordinate_noise_sigma,
            "memory_source": self.memory_source,
            "self_mask_same_particle": self.self_mask_same_particle,
            "hard_local_radius": self.hard_local_radius,
            "matching_policy": "none_matching_free_cross_attention_v1",
            "output_operation_order": [
                "tanh",
                "masked_particle_mean_center"
                if self.center_output
                else "uncentered_diagnostic",
                "whole_coordinate_event_dropout",
                "valid_entry_gaussian_noise",
                "masked_particle_mean_recenter"
                if self.center_output
                else "uncentered_after_noise",
                "invalid_particle_zeroing",
            ],
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


def particle_view_generator_config_from_payload(
    payload: Mapping[str, Any],
) -> ParticleViewGeneratorConfig:
    required = {
        "contract",
        "query_dim",
        "memory_dim",
        "width",
        "num_heads",
        "num_cross_attention_blocks",
        "feed_forward_expansion",
        "bottleneck_width",
        "pair_feature_schema",
        "pair_feature_order",
        "pair_feature_dim",
        "use_pair_bias",
        "use_null_token",
        "center_output",
        "coordinate_dropout_probability",
        "coordinate_noise_sigma",
        "memory_source",
        "self_mask_same_particle",
        "hard_local_radius",
        "matching_policy",
        "output_operation_order",
    }
    if set(payload) != required:
        raise ValueError("generator config field inventory mismatch")
    config = ParticleViewGeneratorConfig(
        contract=str(payload["contract"]),
        query_dim=payload["query_dim"],
        memory_dim=payload["memory_dim"],
        width=payload["width"],
        num_heads=payload["num_heads"],
        num_cross_attention_blocks=payload["num_cross_attention_blocks"],
        feed_forward_expansion=payload["feed_forward_expansion"],
        bottleneck_width=payload["bottleneck_width"],
        pair_feature_dim=payload["pair_feature_dim"],
        use_pair_bias=payload["use_pair_bias"],
        use_null_token=payload["use_null_token"],
        center_output=payload["center_output"],
        coordinate_dropout_probability=payload[
            "coordinate_dropout_probability"
        ],
        coordinate_noise_sigma=payload["coordinate_noise_sigma"],
        memory_source=str(payload["memory_source"]),
        self_mask_same_particle=payload["self_mask_same_particle"],
        hard_local_radius=payload["hard_local_radius"],
    )
    if config.to_payload() != dict(payload):
        raise ValueError("generator config payload is not canonical")
    return config


def build_mandatory_hlt_memory_control_configs(
    *,
    token_dim: int,
    width: int = 160,
    num_heads: int = 8,
    num_cross_attention_blocks: int = 2,
    feed_forward_expansion: int = 4,
    bottleneck_width: int = 4,
) -> dict[str, ParticleViewGeneratorConfig]:
    """Return the locked self-inclusive and same-particle-masked controls."""

    common = {
        "query_dim": token_dim,
        "memory_dim": token_dim,
        "width": width,
        "num_heads": num_heads,
        "num_cross_attention_blocks": num_cross_attention_blocks,
        "feed_forward_expansion": feed_forward_expansion,
        "bottleneck_width": bottleneck_width,
        "memory_source": "hlt",
    }
    return {
        "VGEN_MEMORY_HLT": ParticleViewGeneratorConfig(
            **common, self_mask_same_particle=False
        ),
        "VGEN_MEMORY_HLT_SELFMASK": ParticleViewGeneratorConfig(
            **common, self_mask_same_particle=True
        ),
    }


@dataclass(frozen=True)
class ContextualQueryTapConfig:
    source: str
    available_layers: tuple[str, ...]
    checkpoint_sha256: str
    input_normalization_sha256: str
    tensor_location: str = "after_both_residuals_before_next_block_and_pooling"
    dropout_disabled: bool = True
    detach_frozen_tokens: bool = True

    def __post_init__(self) -> None:
        if self.source not in PARTICLE_VIEW_QUERY_SOURCES:
            raise ValueError(f"unknown query source {self.source!r}")
        required = {
            "raw": ("raw",),
            "embedding": ("embedding",),
            "middle": ("middle",),
            "penultimate": ("penultimate",),
            "mix3": ("final_minus2", "final_minus1", "final"),
        }[self.source]
        if not set(required).issubset(self.available_layers):
            raise ValueError(f"query source {self.source} omits required frozen taps")
        for name, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("input_normalization_sha256", self.input_normalization_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.dropout_disabled or not self.detach_frozen_tokens:
            raise ValueError("contextual query taps must be frozen with dropout disabled")

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": PARTICLE_VIEW_QUERY_TAP_CONTRACT,
            "source": self.source,
            "available_layers": list(self.available_layers),
            "checkpoint_sha256": self.checkpoint_sha256,
            "input_normalization_sha256": self.input_normalization_sha256,
            "tensor_location": self.tensor_location,
            "dropout_disabled": self.dropout_disabled,
            "detach_frozen_tokens": self.detach_frozen_tokens,
        }


class ContextualHLTQueryTap(nn.Module):
    """Select or mix authenticated frozen A0 contextual particle taps."""

    def __init__(self, config: ContextualQueryTapConfig) -> None:
        super().__init__()
        self.config = config
        if config.source == "mix3":
            self.mixture_logits = nn.Parameter(torch.zeros(3))
        else:
            self.register_parameter("mixture_logits", None)

    def forward(
        self,
        frozen_taps: Mapping[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        required = {
            "raw": ("raw",),
            "embedding": ("embedding",),
            "middle": ("middle",),
            "penultimate": ("penultimate",),
            "mix3": ("final_minus2", "final_minus1", "final"),
        }[self.config.source]
        if any(name not in frozen_taps for name in required):
            raise ValueError("frozen tap tensor inventory is incomplete")
        tensors = [frozen_taps[name].detach() for name in required]
        shape = tensors[0].shape
        if any(tensor.shape != shape for tensor in tensors):
            raise ValueError("frozen query tap shapes differ")
        batch, particles, _ = _require_tokens(tensors[0], name="frozen_query_tap")
        mask = _require_bool_mask(
            mask, batch=batch, particles=particles, name="query_mask"
        )
        if self.config.source == "mix3":
            weights = torch.softmax(self.mixture_logits, dim=0)
            selected = sum(
                weight * tensor for weight, tensor in zip(weights, tensors)
            )
        else:
            selected = tensors[0]
        return torch.where(
            mask[:, :, None], selected, torch.zeros_like(selected)
        )

    def provenance_payload(self) -> dict[str, Any]:
        payload = self.config.to_payload()
        if self.mixture_logits is not None:
            weights = torch.softmax(self.mixture_logits.detach().cpu(), dim=0)
            payload["mixture_weights"] = [float(value) for value in weights]
        else:
            payload["mixture_weights"] = None
        payload["content_hash"] = canonical_sha256(payload)
        return payload


class _GeometricCrossAttentionBlock(nn.Module):
    def __init__(self, config: ParticleViewGeneratorConfig) -> None:
        super().__init__()
        self.width = config.width
        self.num_heads = config.num_heads
        self.head_dim = config.width // config.num_heads
        self.use_pair_bias = config.use_pair_bias
        self.hard_local_radius = config.hard_local_radius
        self.query_norm = nn.LayerNorm(config.width)
        self.memory_norm = nn.LayerNorm(config.width)
        self.q_projection = nn.Linear(config.width, config.width)
        self.k_projection = nn.Linear(config.width, config.width)
        self.v_projection = nn.Linear(config.width, config.width)
        self.output_projection = nn.Linear(config.width, config.width)
        pair_hidden = max(32, config.width // 2)
        self.pair_bias = (
            nn.Sequential(
                nn.Linear(config.pair_feature_dim, pair_hidden),
                nn.GELU(),
                nn.Linear(pair_hidden, config.num_heads),
            )
            if config.use_pair_bias
            else None
        )
        self.null_pair_bias = nn.Parameter(torch.zeros(config.num_heads))
        self.feed_forward_norm = nn.LayerNorm(config.width)
        hidden = config.width * config.feed_forward_expansion
        self.feed_forward = nn.Sequential(
            nn.Linear(config.width, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.width),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_mask: torch.Tensor,
        memory_mask: torch.Tensor,
        pair_features: torch.Tensor,
        self_mask_same_particle: bool,
        null_token_present: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_particles, _ = query.shape
        memory_particles = int(memory.shape[1])
        normalized_query = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        q = self.q_projection(normalized_query).reshape(
            batch, query_particles, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_projection(normalized_memory).reshape(
            batch, memory_particles, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_projection(normalized_memory).reshape(
            batch, memory_particles, self.num_heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if self.pair_bias is not None:
            nonnull_memory = memory_particles - int(null_token_present)
            bias = self.pair_bias(pair_features).permute(0, 3, 1, 2)
            if null_token_present:
                null_bias = self.null_pair_bias[None, :, None, None].expand(
                    batch, -1, query_particles, 1
                )
                bias = torch.cat((bias, null_bias), dim=-1)
            if bias.shape[-1] != memory_particles or nonnull_memory != pair_features.shape[2]:
                raise RuntimeError("pair-bias/memory shape mismatch")
            scores = scores + bias

        allowed = memory_mask[:, None, None, :].expand(
            batch, self.num_heads, query_particles, memory_particles
        ).clone()
        if self.hard_local_radius is not None:
            nonnull_memory = memory_particles - int(null_token_present)
            delta_r = torch.sqrt(
                pair_features[..., 0].square()
                + pair_features[..., 1].square()
            )
            local = delta_r <= float(self.hard_local_radius)
            allowed[..., :nonnull_memory] &= local[:, None]
        if self_mask_same_particle:
            nonnull_memory = memory_particles - int(null_token_present)
            if nonnull_memory != query_particles:
                raise ValueError(
                    "self-masked HLT memory requires aligned query/memory lengths"
                )
            diagonal = torch.eye(
                query_particles, dtype=torch.bool, device=query.device
            )[None, None, :, :]
            allowed[..., :nonnull_memory] &= ~diagonal
        if not null_token_present and (~allowed).all(dim=-1).any():
            raise ValueError("a query has no valid memory key and no null token")
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(
            query_mask[:, None, :, None],
            attention,
            torch.zeros_like(attention),
        )
        context = torch.matmul(attention, v).transpose(1, 2).reshape(
            batch, query_particles, self.width
        )
        query = query + self.output_projection(context)
        query = query + self.feed_forward(self.feed_forward_norm(query))
        query = torch.where(
            query_mask[:, :, None], query, torch.zeros_like(query)
        )
        return query, attention


@dataclass
class ParticleViewGeneratorOutput:
    view: torch.Tensor
    deterministic_centered_view: torch.Tensor
    rich_context: torch.Tensor
    attention: torch.Tensor
    null_attention_fraction: torch.Tensor
    query_mask: torch.Tensor


class MatchingFreeParticleViewGenerator(nn.Module):
    """Canonical HLT-query/offline-memory privileged target generator."""

    def __init__(self, config: ParticleViewGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.query_input_norm = nn.LayerNorm(config.query_dim)
        self.memory_input_norm = nn.LayerNorm(config.memory_dim)
        self.query_projection = nn.Linear(config.query_dim, config.width)
        self.memory_projection = nn.Linear(config.memory_dim, config.width)
        self.null_token = (
            nn.Parameter(torch.zeros(1, 1, config.width))
            if config.use_null_token
            else None
        )
        if self.null_token is not None:
            nn.init.normal_(self.null_token, std=0.02)
        self.blocks = nn.ModuleList(
            _GeometricCrossAttentionBlock(config)
            for _ in range(config.num_cross_attention_blocks)
        )
        self.rich_context_norm = nn.LayerNorm(config.width)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.bottleneck_width),
        )

    def _augment(
        self,
        deterministic: torch.Tensor,
        mask: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        augmented = deterministic
        if self.config.coordinate_dropout_probability > 0.0:
            keep = torch.rand(
                (
                    deterministic.shape[0],
                    1,
                    deterministic.shape[2],
                ),
                device=deterministic.device,
                generator=generator,
            ) >= self.config.coordinate_dropout_probability
            augmented = augmented * keep.to(dtype=augmented.dtype)
        if self.config.coordinate_noise_sigma > 0.0:
            noise = torch.randn(
                augmented.shape,
                device=augmented.device,
                dtype=augmented.dtype,
                generator=generator,
            ) * self.config.coordinate_noise_sigma
            augmented = torch.where(
                mask[:, :, None], augmented + noise, augmented
            )
        if self.config.center_output:
            augmented = masked_particle_mean_center(augmented, mask)
        return torch.where(
            mask[:, :, None], augmented, torch.zeros_like(augmented)
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        *,
        query_geometry: torch.Tensor,
        memory_geometry: torch.Tensor,
        query_mask: torch.Tensor,
        memory_mask: torch.Tensor,
        apply_output_augmentation: bool | None = None,
        augmentation_generator: torch.Generator | None = None,
    ) -> ParticleViewGeneratorOutput:
        batch, query_particles, query_dim = _require_tokens(
            query_tokens, name="query_tokens"
        )
        memory_batch, memory_particles, memory_dim = _require_tokens(
            memory_tokens, name="memory_tokens"
        )
        if memory_batch != batch:
            raise ValueError("query and memory token batch dimensions differ")
        if query_dim != self.config.query_dim or memory_dim != self.config.memory_dim:
            raise ValueError("query or memory token width differs from generator config")
        query_mask = _require_bool_mask(
            query_mask,
            batch=batch,
            particles=query_particles,
            name="query_mask",
        )
        memory_mask = _require_bool_mask(
            memory_mask,
            batch=batch,
            particles=memory_particles,
            name="memory_mask",
        )
        query_geometry = _require_geometry(
            query_geometry,
            batch=batch,
            particles=query_particles,
            name="query_geometry",
        )
        memory_geometry = _require_geometry(
            memory_geometry,
            batch=batch,
            particles=memory_particles,
            name="memory_geometry",
        )
        if self.config.memory_source == "hlt":
            if query_particles != memory_particles:
                raise ValueError("HLT-memory controls require aligned particle counts")
            if not torch.equal(query_mask, memory_mask):
                raise ValueError("HLT-memory controls require the exact query mask")
            if not torch.equal(query_tokens[query_mask], memory_tokens[memory_mask]):
                raise ValueError(
                    "HLT-memory controls require the exact frozen query tokens"
                )
            if not torch.equal(
                query_geometry[query_mask], memory_geometry[memory_mask]
            ):
                raise ValueError(
                    "HLT-memory controls require exact HLT-to-HLT geometry"
                )

        safe_query = torch.where(
            query_mask[:, :, None], query_tokens, torch.zeros_like(query_tokens)
        )
        safe_memory = torch.where(
            memory_mask[:, :, None], memory_tokens, torch.zeros_like(memory_tokens)
        )
        if not torch.isfinite(safe_query).all() or not torch.isfinite(safe_memory).all():
            raise ValueError("valid query/memory tokens must be finite")
        query = self.query_projection(self.query_input_norm(safe_query))
        memory = self.memory_projection(self.memory_input_norm(safe_memory))
        pair_features = build_hlt_memory_pair_features(
            query_geometry,
            memory_geometry,
            query_mask=query_mask,
            memory_mask=memory_mask,
        )
        if self.null_token is not None:
            null = self.null_token.expand(batch, -1, -1)
            memory = torch.cat((memory, null), dim=1)
            memory_mask_with_null = torch.cat(
                (
                    memory_mask,
                    torch.ones((batch, 1), dtype=torch.bool, device=memory.device),
                ),
                dim=1,
            )
        else:
            memory_mask_with_null = memory_mask

        attention: torch.Tensor | None = None
        for block in self.blocks:
            query, attention = block(
                query,
                memory,
                query_mask=query_mask,
                memory_mask=memory_mask_with_null,
                pair_features=pair_features,
                self_mask_same_particle=self.config.self_mask_same_particle,
                null_token_present=self.null_token is not None,
            )
        if attention is None:
            raise RuntimeError("generator has no cross-attention blocks")
        rich = self.rich_context_norm(query)
        rich = torch.where(query_mask[:, :, None], rich, torch.zeros_like(rich))
        bounded = torch.tanh(self.bottleneck(rich))
        deterministic = (
            masked_particle_mean_center(bounded, query_mask)
            if self.config.center_output
            else torch.where(
                query_mask[:, :, None], bounded, torch.zeros_like(bounded)
            )
        )
        if not torch.isfinite(deterministic).all():
            raise ValueError("generator produced nonfinite coordinates")
        if float(deterministic.detach().abs().max().cpu()) > 2.0 + 1.0e-6:
            raise ValueError("centered generator coordinates exceed [-2,2]")

        if apply_output_augmentation is None:
            apply_output_augmentation = self.training
        if apply_output_augmentation:
            view = self._augment(
                deterministic,
                query_mask,
                generator=augmentation_generator,
            )
        else:
            view = deterministic
        if self.null_token is None:
            null_fraction = view.new_zeros(())
        else:
            valid = query_mask[:, None, :, None].to(dtype=attention.dtype)
            denominator = valid.sum() * attention.shape[1]
            null_fraction = (
                (attention[..., -1:] * valid).sum()
                / denominator.clamp_min(1.0)
            )
        return ParticleViewGeneratorOutput(
            view=view,
            deterministic_centered_view=deterministic,
            rich_context=rich,
            attention=attention,
            null_attention_fraction=null_fraction,
            query_mask=query_mask,
        )


def particle_view_rate_covariance_losses(
    view: torch.Tensor,
    mask: torch.Tensor,
    *,
    variance_floor: float = 0.02,
    rate_variance_per_dimension: float = 0.50,
) -> dict[str, torch.Tensor | str]:
    """Compute the locked variance-floor, rate, and covariance penalties."""

    batch, particles, width = _require_tokens(view, name="view")
    for name, value in (
        ("variance_floor", variance_floor),
        ("rate_variance_per_dimension", rate_variance_per_dimension),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    mask = _require_bool_mask(mask, batch=batch, particles=particles, name="mask")
    valid = view[mask]
    if valid.numel() == 0:
        raise ValueError("rate/covariance losses require at least one valid particle")
    if not torch.isfinite(valid).all():
        raise ValueError("rate/covariance inputs contain nonfinite values")
    centered = valid - valid.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1).matmul(centered) / valid.shape[0]
    variances = torch.diagonal(covariance)
    floor_loss = torch.relu(
        view.new_tensor(variance_floor) - variances
    ).mean()
    rate_limit = view.new_tensor(rate_variance_per_dimension * width)
    rate_loss = torch.relu(torch.trace(covariance) - rate_limit).square()
    if width == 1:
        covariance_loss = view.new_zeros(())
    else:
        off_diagonal = covariance - torch.diag_embed(variances)
        covariance_loss = off_diagonal.square().sum() / (width * (width - 1))
    return {
        "contract": PARTICLE_VIEW_RATE_LOSS_CONTRACT,
        "variance_floor_loss": floor_loss,
        "rate_loss": rate_loss,
        "covariance_loss": covariance_loss,
        "variances": variances,
        "covariance": covariance,
    }


__all__ = [
    "ContextualHLTQueryTap",
    "ContextualQueryTapConfig",
    "MatchingFreeParticleViewGenerator",
    "PARTICLE_GEOMETRY_ORDER",
    "PARTICLE_VIEW_GENERATOR_CONFIG_CONTRACT",
    "PARTICLE_VIEW_HLT_MEMORY_CONTROL_IDS",
    "PARTICLE_VIEW_MEMORY_SOURCES",
    "PARTICLE_VIEW_PAIR_FEATURE_CONTRACT",
    "PARTICLE_VIEW_PAIR_FEATURE_ORDER",
    "PARTICLE_VIEW_QUERY_SOURCES",
    "PARTICLE_VIEW_QUERY_TAP_CONTRACT",
    "PARTICLE_VIEW_RATE_LOSS_CONTRACT",
    "ParticleViewGeneratorConfig",
    "ParticleViewGeneratorOutput",
    "build_hlt_memory_pair_features",
    "build_mandatory_hlt_memory_control_configs",
    "masked_particle_mean_center",
    "particle_view_rate_covariance_losses",
    "particle_view_generator_config_from_payload",
    "wrap_delta_phi",
]

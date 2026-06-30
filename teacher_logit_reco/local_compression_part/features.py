"""Canonical input helpers for local-compression residual feature adapters.

Step 2 owns the boundary between fixed-HLT raw tokens and the exact canonical
Particle Transformer input contract.  It deliberately reuses the shared ParT
input builder from ``jetclass_fresh.dual_view`` while also returning the
sanitized/top-k raw tokens in the same particle order, so later modality
subtoken builders cannot drift out of alignment with ParT features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.dual_view import (
    _nan_to_num_torch,
    build_part_inputs_torch,
    sanitize_tokens_for_part_inputs,
)
from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .config import (
    LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES,
    LOCAL_COMPRESSION_CANONICAL_POINT_NAMES,
    LOCAL_COMPRESSION_CANONICAL_VECTOR_NAMES,
    LOCAL_COMPRESSION_DERIVED_FIELD_NAMES,
    LOCAL_COMPRESSION_MODALITIES,
    LOCAL_COMPRESSION_PART_CONTRACT,
    LOCAL_COMPRESSION_RAW_FEATURE_NAMES,
    LocalCompressionFeatureConfig,
    LocalCompressionModalitySpec,
)


LOCAL_COMPRESSION_FEATURES_STEP = "local_compression_part_step2_features"
LOCAL_COMPRESSION_FEATURES_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_canonical_inputs_v1"


def _as_float_tensor(value: Any) -> Any:
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.as_tensor(value, dtype=torch.float32)


def _as_bool_tensor(value: Any, *, device: Any) -> Any:
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.bool)
    return torch.as_tensor(value, dtype=torch.bool, device=device)


def normalize_local_compression_feature_config(
    config: LocalCompressionFeatureConfig | Mapping[str, Any] | None = None,
) -> LocalCompressionFeatureConfig:
    """Return a validated local-compression feature config."""

    if config is None:
        return LocalCompressionFeatureConfig()
    if isinstance(config, LocalCompressionFeatureConfig):
        return config
    return LocalCompressionFeatureConfig(**dict(config))


def _safe_tensor_sum_int(value: Any) -> int:
    return int(value.detach().cpu().sum().item())


@dataclass(frozen=True)
class PreparedLocalCompressionInputs:
    """Sanitized raw HLT tokens aligned to the canonical ParT particle order."""

    tokens: Any
    mask: Any
    weights: Any | None
    original_all_finite: Any
    max_constits: int
    weight_threshold: float
    feature_config: LocalCompressionFeatureConfig

    def __post_init__(self) -> None:
        if int(self.tokens.ndim) != 3:
            raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(self.tokens.shape)}")
        if int(self.tokens.shape[-1]) != int(self.feature_config.raw_token_dim):
            raise ValueError(
                f"tokens last dimension must be raw_token_dim={int(self.feature_config.raw_token_dim)}, "
                f"got {int(self.tokens.shape[-1])}"
            )
        if int(self.mask.ndim) != 2 or tuple(self.mask.shape) != tuple(self.tokens.shape[:2]):
            raise ValueError(f"mask shape {tuple(self.mask.shape)} does not match tokens shape {tuple(self.tokens.shape[:2])}")
        if self.weights is not None and tuple(self.weights.shape) != tuple(self.mask.shape):
            raise ValueError(f"weights shape {tuple(self.weights.shape)} does not match mask shape {tuple(self.mask.shape)}")
        if tuple(self.original_all_finite.shape) != tuple(self.mask.shape):
            raise ValueError(
                f"original_all_finite shape {tuple(self.original_all_finite.shape)} does not match "
                f"mask shape {tuple(self.mask.shape)}"
            )
        if int(self.max_constits) <= 0:
            raise ValueError("max_constits must be positive")
        if float(self.weight_threshold) < 0.0:
            raise ValueError("weight_threshold must be non-negative")

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def raw_token_dim(self) -> int:
        return int(self.tokens.shape[2])

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_COMPRESSION_FEATURES_CONTRACT,
            "tokens_shape": list(self.tokens.shape),
            "mask_shape": list(self.mask.shape),
            "weights_shape": None if self.weights is None else list(self.weights.shape),
            "original_all_finite_shape": list(self.original_all_finite.shape),
            "raw_token_dim": int(self.raw_token_dim),
            "max_constits": int(self.max_constits),
            "weight_threshold": float(self.weight_threshold),
            "valid_particle_count": _safe_tensor_sum_int(self.mask),
            "original_nonfinite_particle_count": int((~self.original_all_finite.bool()).detach().cpu().sum().item()),
        }


@dataclass(frozen=True)
class LocalCompressionCanonicalInputs:
    """Canonical ParT tensors plus aligned raw-token rows for LC adapters."""

    points: Any
    features: Any
    lorentz_vectors: Any
    mask: Any
    selected_tokens: Any
    particle_mask: Any
    selected_weights: Any | None = None
    selected_original_all_finite: Any | None = None
    feature_names: tuple[str, ...] = LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES
    point_names: tuple[str, ...] = LOCAL_COMPRESSION_CANONICAL_POINT_NAMES
    vector_names: tuple[str, ...] = LOCAL_COMPRESSION_CANONICAL_VECTOR_NAMES
    max_constits: int = 128
    weight_threshold: float = 0.0

    def __post_init__(self) -> None:
        if int(self.features.ndim) != 3:
            raise ValueError(f"features must have shape [batch, features, particles], got {tuple(self.features.shape)}")
        batch_size = int(self.features.shape[0])
        num_particles = int(self.features.shape[2])
        expected = {
            "points": (batch_size, len(self.point_names), num_particles),
            "features": (batch_size, len(self.feature_names), num_particles),
            "lorentz_vectors": (batch_size, len(self.vector_names), num_particles),
            "mask": (batch_size, 1, num_particles),
            "selected_tokens": (batch_size, num_particles, RAW_TOKEN_DIM),
            "particle_mask": (batch_size, num_particles),
        }
        actual = {
            "points": tuple(self.points.shape),
            "features": tuple(self.features.shape),
            "lorentz_vectors": tuple(self.lorentz_vectors.shape),
            "mask": tuple(self.mask.shape),
            "selected_tokens": tuple(self.selected_tokens.shape),
            "particle_mask": tuple(self.particle_mask.shape),
        }
        for key, shape in expected.items():
            if actual[key] != shape:
                raise ValueError(f"{key} has shape {actual[key]}, expected {shape}")
        if self.selected_weights is not None and tuple(self.selected_weights.shape) != tuple(self.particle_mask.shape):
            raise ValueError(
                f"selected_weights shape {tuple(self.selected_weights.shape)} does not match "
                f"particle_mask shape {tuple(self.particle_mask.shape)}"
            )
        if self.selected_original_all_finite is not None and tuple(self.selected_original_all_finite.shape) != tuple(
            self.particle_mask.shape
        ):
            raise ValueError(
                "selected_original_all_finite shape must match particle_mask shape, got "
                f"{tuple(self.selected_original_all_finite.shape)} vs {tuple(self.particle_mask.shape)}"
            )
        if tuple(self.feature_names) != LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES:
            raise ValueError("feature_names must match canonical PF_FEATURE_NAMES order")
        if tuple(self.point_names) != LOCAL_COMPRESSION_CANONICAL_POINT_NAMES:
            raise ValueError("point_names must match canonical PF_POINT_NAMES order")
        if tuple(self.vector_names) != LOCAL_COMPRESSION_CANONICAL_VECTOR_NAMES:
            raise ValueError("vector_names must match canonical PF_VECTOR_NAMES order")
        if int(self.max_constits) <= 0:
            raise ValueError("max_constits must be positive")
        if float(self.weight_threshold) < 0.0:
            raise ValueError("weight_threshold must be non-negative")
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "point_names", tuple(self.point_names))
        object.__setattr__(self, "vector_names", tuple(self.vector_names))
        object.__setattr__(self, "max_constits", int(self.max_constits))
        object.__setattr__(self, "weight_threshold", float(self.weight_threshold))

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.features.shape[2])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def feature_rows(self) -> Any:
        """Return ``[batch, particles, PF_FEATURE_NAMES]`` for delta-F adapters."""

        return self.features.transpose(1, 2).contiguous()

    def vector_rows(self) -> Any:
        """Return ``[batch, particles, PF_VECTOR_NAMES]`` for diagnostics."""

        return self.lorentz_vectors.transpose(1, 2).contiguous()

    def point_rows(self) -> Any:
        """Return ``[batch, particles, PF_POINT_NAMES]`` for diagnostics."""

        return self.points.transpose(1, 2).contiguous()

    def as_part_kwargs(self) -> dict[str, Any]:
        """Return tensors with the keyword names expected by the HLT ParT wrapper."""

        return {
            "points": self.points,
            "features": self.features,
            "lorentz_vectors": self.lorentz_vectors,
            "mask": self.mask,
        }

    def with_features(self, feature_rows: Any) -> "LocalCompressionCanonicalInputs":
        """Return a new canonical input object with adapted feature rows.

        ``feature_rows`` must be ``[batch, particles, PF_FEATURE_NAMES]``.  This
        helper keeps Step 9/10 code from manually transposing tensors.
        """

        torch = require_torch()
        feature_rows = torch.as_tensor(feature_rows, device=self.features.device, dtype=self.features.dtype)
        if tuple(feature_rows.shape) != (self.batch_size, self.num_particles, self.feature_dim):
            raise ValueError(
                f"feature_rows shape {tuple(feature_rows.shape)} does not match "
                f"{(self.batch_size, self.num_particles, self.feature_dim)}"
            )
        adapted = feature_rows.transpose(1, 2).contiguous()
        adapted = _nan_to_num_torch(adapted)
        adapted = adapted * self.mask.to(dtype=adapted.dtype)
        return LocalCompressionCanonicalInputs(
            points=self.points,
            features=adapted,
            lorentz_vectors=self.lorentz_vectors,
            mask=self.mask,
            selected_tokens=self.selected_tokens,
            particle_mask=self.particle_mask,
            selected_weights=self.selected_weights,
            selected_original_all_finite=self.selected_original_all_finite,
            feature_names=self.feature_names,
            point_names=self.point_names,
            vector_names=self.vector_names,
            max_constits=self.max_constits,
            weight_threshold=self.weight_threshold,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_COMPRESSION_FEATURES_CONTRACT,
            "points_shape": list(self.points.shape),
            "features_shape": list(self.features.shape),
            "lorentz_vectors_shape": list(self.lorentz_vectors.shape),
            "mask_shape": list(self.mask.shape),
            "selected_tokens_shape": list(self.selected_tokens.shape),
            "selected_weights_shape": None if self.selected_weights is None else list(self.selected_weights.shape),
            "selected_original_all_finite_shape": (
                None if self.selected_original_all_finite is None else list(self.selected_original_all_finite.shape)
            ),
            "feature_names": list(self.feature_names),
            "point_names": list(self.point_names),
            "vector_names": list(self.vector_names),
            "max_constits": int(self.max_constits),
            "weight_threshold": float(self.weight_threshold),
            "valid_particle_count": _safe_tensor_sum_int(self.particle_mask),
        }


def _select_topk_with_quality(tokens: Any, mask: Any, weights: Any | None, original_all_finite: Any, *, max_constits: int):
    """Select top-k rows and carry original finite-quality metadata with them."""

    torch = require_torch()
    n_constits = int(tokens.shape[1])
    if n_constits <= int(max_constits):
        return tokens, mask, weights, original_all_finite
    score = torch.clamp(tokens[:, :, 0], min=0.0)
    if weights is not None:
        score = score * torch.clamp(weights, min=0.0)
    score = score.masked_fill(~mask.bool(), -1.0)
    _, indices = torch.topk(score, k=int(max_constits), dim=1, largest=True, sorted=True)
    token_indices = indices[:, :, None].expand(-1, -1, tokens.shape[2])
    selected_tokens = torch.gather(tokens, dim=1, index=token_indices)
    selected_mask = torch.gather(mask.bool(), dim=1, index=indices)
    selected_weights = None if weights is None else torch.gather(weights, dim=1, index=indices)
    selected_original_all_finite = torch.gather(original_all_finite.bool(), dim=1, index=indices)
    return selected_tokens, selected_mask, selected_weights, selected_original_all_finite


def prepare_local_compression_tokens_and_mask(
    tokens: Any,
    mask: Any,
    *,
    weights: Any | None = None,
    max_constits: int | None = 128,
    weight_threshold: float = 0.0,
    config: LocalCompressionFeatureConfig | Mapping[str, Any] | None = None,
) -> PreparedLocalCompressionInputs:
    """Validate, sanitize, and top-k select HLT tokens for local compression.

    The returned tokens are aligned with the canonical ParT particle order.  If
    ``weights`` are supplied, the weights are preserved separately so canonical
    ParT feature construction can use the shared ``build_part_inputs_torch``
    weighted path without mutating the raw-token columns seen by modalities.
    """

    torch = require_torch()
    feature_config = normalize_local_compression_feature_config(config)
    tokens = _as_float_tensor(tokens)
    mask = _as_bool_tensor(mask, device=tokens.device)
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(tokens.shape[-1]) != int(feature_config.raw_token_dim):
        raise ValueError(
            f"tokens last dimension must be raw_token_dim={int(feature_config.raw_token_dim)}, "
            f"got {int(tokens.shape[-1])}"
        )
    if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match tokens shape {tuple(tokens.shape[:2])}")
    if float(weight_threshold) < 0.0:
        raise ValueError("weight_threshold must be non-negative")

    original_all_finite = torch.isfinite(tokens).all(dim=-1)
    tokens, mask = sanitize_tokens_for_part_inputs(tokens, mask)
    resolved_weights = None
    if weights is not None:
        resolved_weights = _as_float_tensor(weights).to(device=tokens.device)
        if tuple(resolved_weights.shape) != tuple(mask.shape):
            raise ValueError(f"weights shape {tuple(resolved_weights.shape)} does not match mask shape {tuple(mask.shape)}")
        finite_weights = torch.isfinite(resolved_weights)
        resolved_weights = torch.clamp(_nan_to_num_torch(resolved_weights), min=0.0)
        mask = mask & finite_weights & (resolved_weights > float(weight_threshold))

    resolved_max_constits = int(tokens.shape[1]) if max_constits is None else int(max_constits)
    if resolved_max_constits <= 0:
        raise ValueError("max_constits must be positive")
    tokens, mask, resolved_weights, original_all_finite = _select_topk_with_quality(
        tokens,
        mask,
        resolved_weights,
        original_all_finite,
        max_constits=resolved_max_constits,
    )

    tokens = _nan_to_num_torch(tokens) * mask[:, :, None].to(dtype=tokens.dtype)
    return PreparedLocalCompressionInputs(
        tokens=tokens,
        mask=mask,
        weights=resolved_weights,
        original_all_finite=original_all_finite,
        max_constits=resolved_max_constits,
        weight_threshold=float(weight_threshold),
        feature_config=feature_config,
    )


def build_local_compression_canonical_inputs(
    tokens: Any,
    mask: Any,
    *,
    weights: Any | None = None,
    max_constits: int | None = 128,
    weight_threshold: float = 0.0,
    config: LocalCompressionFeatureConfig | Mapping[str, Any] | None = None,
) -> LocalCompressionCanonicalInputs:
    """Build exact ParT tensors and aligned raw tokens for LC adapters."""

    torch = require_torch()
    prepared = prepare_local_compression_tokens_and_mask(
        tokens,
        mask,
        weights=weights,
        max_constits=max_constits,
        weight_threshold=weight_threshold,
        config=config,
    )
    part_inputs = build_part_inputs_torch(
        prepared.tokens,
        prepared.mask,
        weights=prepared.weights,
        max_constits=prepared.num_particles,
        weight_threshold=prepared.weight_threshold,
    )
    points = torch.as_tensor(part_inputs["points"], device=prepared.tokens.device).float()
    features = torch.as_tensor(part_inputs["features"], device=prepared.tokens.device).float()
    lorentz_vectors = torch.as_tensor(part_inputs["lorentz_vectors"], device=prepared.tokens.device).float()
    part_mask = torch.as_tensor(part_inputs["mask"], device=prepared.tokens.device).bool()
    particle_mask = part_mask[:, 0, :].bool()
    if tuple(particle_mask.shape) != tuple(prepared.mask.shape):
        raise ValueError("canonical ParT mask shape drifted from prepared local-compression mask")
    if not bool(torch.equal(particle_mask, prepared.mask)):
        raise ValueError("canonical ParT mask is not aligned with prepared local-compression mask")
    return LocalCompressionCanonicalInputs(
        points=points,
        features=features,
        lorentz_vectors=lorentz_vectors,
        mask=part_mask,
        selected_tokens=prepared.tokens,
        particle_mask=particle_mask,
        selected_weights=prepared.weights,
        selected_original_all_finite=prepared.original_all_finite,
        feature_names=LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES,
        point_names=LOCAL_COMPRESSION_CANONICAL_POINT_NAMES,
        vector_names=LOCAL_COMPRESSION_CANONICAL_VECTOR_NAMES,
        max_constits=prepared.max_constits,
        weight_threshold=prepared.weight_threshold,
    )


# Short alias used by later model code, mirroring the multiscale package naming.
build_canonical_part_inputs = build_local_compression_canonical_inputs


@dataclass(frozen=True)
class LocalCompressionModalities:
    """Semantic modality feature tensors for one canonical HLT particle batch."""

    values_by_modality: Mapping[str, Any]
    modality_mask: Any
    particle_mask: Any
    feature_names_by_modality: Mapping[str, tuple[str, ...]]
    source_names_by_modality: Mapping[str, tuple[str, ...]]
    modality_names: tuple[str, ...]
    feature_config: LocalCompressionFeatureConfig

    def __post_init__(self) -> None:
        torch = require_torch()
        modality_names = tuple(str(name) for name in self.modality_names)
        if modality_names != self.feature_config.modality_names:
            raise ValueError("modality_names must match feature_config modality order")
        values_by_modality = dict(self.values_by_modality)
        feature_names_by_modality = {
            str(name): tuple(str(item) for item in names)
            for name, names in dict(self.feature_names_by_modality).items()
        }
        source_names_by_modality = {
            str(name): tuple(str(item) for item in names)
            for name, names in dict(self.source_names_by_modality).items()
        }
        if set(values_by_modality) != set(modality_names):
            raise ValueError("values_by_modality keys must match modality_names")
        if set(feature_names_by_modality) != set(modality_names):
            raise ValueError("feature_names_by_modality keys must match modality_names")
        if set(source_names_by_modality) != set(modality_names):
            raise ValueError("source_names_by_modality keys must match modality_names")
        if int(self.particle_mask.ndim) != 2:
            raise ValueError("particle_mask must have shape [batch, particles]")
        batch_size, num_particles = tuple(self.particle_mask.shape)
        expected_modality_mask_shape = (batch_size, num_particles, len(modality_names))
        if tuple(self.modality_mask.shape) != expected_modality_mask_shape:
            raise ValueError(
                f"modality_mask has shape {tuple(self.modality_mask.shape)}, expected {expected_modality_mask_shape}"
            )
        cleaned_values: dict[str, Any] = {}
        for name in modality_names:
            values = torch.as_tensor(values_by_modality[name], device=self.particle_mask.device).float()
            if int(values.ndim) != 3 or tuple(values.shape[:2]) != (batch_size, num_particles):
                raise ValueError(f"modality {name!r} values must have shape [batch, particles, features]")
            if int(values.shape[-1]) != len(feature_names_by_modality[name]):
                raise ValueError(
                    f"modality {name!r} has {int(values.shape[-1])} features but "
                    f"{len(feature_names_by_modality[name])} names"
                )
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"modality {name!r} contains non-finite values")
            masked_values = values * self.particle_mask[:, :, None].to(dtype=values.dtype)
            cleaned_values[name] = masked_values
        object.__setattr__(self, "values_by_modality", cleaned_values)
        object.__setattr__(self, "feature_names_by_modality", feature_names_by_modality)
        object.__setattr__(self, "source_names_by_modality", source_names_by_modality)
        object.__setattr__(self, "modality_names", modality_names)

    @property
    def batch_size(self) -> int:
        return int(self.particle_mask.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.particle_mask.shape[1])

    @property
    def num_modalities(self) -> int:
        return len(self.modality_names)

    @property
    def feature_dim_by_modality(self) -> dict[str, int]:
        return {name: int(self.values_by_modality[name].shape[-1]) for name in self.modality_names}

    def stacked_values(self) -> Any:
        """Return padded modality values as ``[B, N, M, max_C]`` for diagnostics.

        Step 4 uses per-modality embedders, so padding is not part of the model
        path.  This helper is useful for tests and cheap summaries.
        """

        torch = require_torch()
        max_dim = max(self.feature_dim_by_modality.values())
        rows = []
        for name in self.modality_names:
            values = self.values_by_modality[name]
            pad = max_dim - int(values.shape[-1])
            if pad > 0:
                values = torch.nn.functional.pad(values, (0, pad))
            rows.append(values)
        return torch.stack(rows, dim=2)

    def summary(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_COMPRESSION_FEATURES_CONTRACT,
            "batch_size": int(self.batch_size),
            "num_particles": int(self.num_particles),
            "num_modalities": int(self.num_modalities),
            "modality_names": list(self.modality_names),
            "feature_dim_by_modality": self.feature_dim_by_modality,
            "feature_names_by_modality": {
                name: list(self.feature_names_by_modality[name]) for name in self.modality_names
            },
            "source_names_by_modality": {
                name: list(self.source_names_by_modality[name]) for name in self.modality_names
            },
            "modality_mask_shape": list(self.modality_mask.shape),
            "valid_particle_count": _safe_tensor_sum_int(self.particle_mask),
            "active_modality_count": _safe_tensor_sum_int(self.modality_mask),
        }


def _raw_feature_map(canonical: LocalCompressionCanonicalInputs) -> dict[str, Any]:
    return {
        name: canonical.selected_tokens[:, :, index]
        for index, name in enumerate(LOCAL_COMPRESSION_RAW_FEATURE_NAMES)
    }


def _pf_feature_map(canonical: LocalCompressionCanonicalInputs) -> dict[str, Any]:
    rows = canonical.feature_rows()
    return {
        name: rows[:, :, index]
        for index, name in enumerate(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES)
    }


def _pt_rank_features(canonical: LocalCompressionCanonicalInputs) -> tuple[Any, Any]:
    torch = require_torch()
    pt = canonical.selected_tokens[:, :, 0]
    mask = canonical.particle_mask.bool()
    scores = torch.where(mask, pt, torch.full_like(pt, -1.0))
    order = torch.argsort(scores, dim=1, descending=True)
    rank = torch.empty_like(order)
    positions = torch.arange(order.shape[1], device=order.device, dtype=order.dtype)[None, :].expand_as(order)
    rank.scatter_(1, order, positions)
    rank = rank.to(dtype=pt.dtype)
    valid_count = torch.clamp(mask.sum(dim=1, keepdim=True).to(dtype=pt.dtype), min=1.0)
    rank_norm = rank / torch.clamp(valid_count - 1.0, min=1.0)
    log_rank = torch.log1p(rank) / torch.log1p(valid_count)
    rank_norm = torch.where(mask, rank_norm, torch.zeros_like(rank_norm))
    log_rank = torch.where(mask, log_rank, torch.zeros_like(log_rank))
    return rank_norm, log_rank


def _charged_pid_mask(tokens: Any) -> Any:
    return (tokens[:, :, 5] + tokens[:, :, 8] + tokens[:, :, 9]) > 0.5


def _neutral_pid_mask(tokens: Any) -> Any:
    return (tokens[:, :, 6] + tokens[:, :, 7]) > 0.5


def _derived_feature_map(canonical: LocalCompressionCanonicalInputs) -> dict[str, Any]:
    torch = require_torch()
    tokens = canonical.selected_tokens
    mask = canonical.particle_mask.bool()
    vectors = canonical.vector_rows()
    phi = tokens[:, :, 2]
    charge = tokens[:, :, 4]
    charged_like = _charged_pid_mask(tokens)
    neutral_like = _neutral_pid_mask(tokens)
    any_pid = charged_like | neutral_like
    charge_nonzero = torch.abs(charge) > 0.5
    charged_consistent = charged_like & charge_nonzero
    neutral_consistent = neutral_like & ~charge_nonzero
    unknown_consistent = ~any_pid
    charged_pid_consistency = (charged_consistent | neutral_consistent | unknown_consistent).to(dtype=tokens.dtype)
    track_signal = (
        torch.abs(tokens[:, :, 10])
        + torch.abs(tokens[:, :, 12])
        + torch.clamp(tokens[:, :, 11], min=0.0)
        + torch.clamp(tokens[:, :, 13], min=0.0)
    )
    neutral_track_applicability = torch.where(
        neutral_like,
        (track_signal <= 1.0e-6).to(dtype=tokens.dtype),
        torch.ones_like(track_signal),
    )
    d0err = torch.clamp(tokens[:, :, 11], min=0.0, max=1.0)
    dzerr = torch.clamp(tokens[:, :, 13], min=0.0, max=1.0)
    track_error_summary = 0.5 * (d0err + dzerr)
    pt_rank, log_pt_rank = _pt_rank_features(canonical)
    if canonical.selected_original_all_finite is None:
        original_all_finite = torch.isfinite(tokens).all(dim=-1)
    else:
        original_all_finite = canonical.selected_original_all_finite.bool()
    all_finite = original_all_finite.to(dtype=tokens.dtype)
    valid_float = mask.to(dtype=tokens.dtype)
    derived = {
        "sin_phi": torch.sin(phi),
        "cos_phi": torch.cos(phi),
        "pt_rank": pt_rank,
        "log_pt_rank": log_pt_rank,
        "part_px": vectors[:, :, 0],
        "part_py": vectors[:, :, 1],
        "part_pz": vectors[:, :, 2],
        "part_energy": vectors[:, :, 3],
        "valid_mask": valid_float,
        "all_finite": all_finite,
        "charged_pid_consistency": charged_pid_consistency,
        "neutral_track_applicability": neutral_track_applicability,
        "track_error_summary": track_error_summary,
    }
    missing = sorted(set(LOCAL_COMPRESSION_DERIVED_FIELD_NAMES) - set(derived))
    if missing:
        raise AssertionError(f"missing derived local-compression fields: {missing}")
    return {
        name: torch.where(mask, _nan_to_num_torch(value), torch.zeros_like(value))
        for name, value in derived.items()
    }


def _values_for_modality(
    spec: LocalCompressionModalitySpec,
    *,
    raw_map: Mapping[str, Any],
    pf_map: Mapping[str, Any],
    derived_map: Mapping[str, Any],
    particle_mask: Any,
) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    torch = require_torch()
    values = []
    feature_names = []
    source_names = []
    for name in spec.raw_feature_names:
        values.append(raw_map[name])
        feature_names.append(name)
        source_names.append(f"raw:{name}")
    for name in spec.pf_feature_names:
        values.append(pf_map[name])
        feature_names.append(name)
        source_names.append(f"pf:{name}")
    for name in spec.derived_feature_names:
        values.append(derived_map[name])
        feature_names.append(name)
        source_names.append(f"derived:{name}")
    if not values:
        raise ValueError(f"modality {spec.name!r} has no fields")
    stacked = torch.stack(values, dim=-1).float()
    stacked = _nan_to_num_torch(stacked)
    stacked = stacked * particle_mask[:, :, None].to(dtype=stacked.dtype)
    return stacked, tuple(feature_names), tuple(source_names)


def build_local_compression_modalities(
    canonical: LocalCompressionCanonicalInputs,
    *,
    config: LocalCompressionFeatureConfig | Mapping[str, Any] | None = None,
) -> LocalCompressionModalities:
    """Build semantic modality features from aligned raw and canonical tensors."""

    torch = require_torch()
    feature_config = normalize_local_compression_feature_config(config)
    if tuple(canonical.feature_names) != tuple(feature_config.canonical_feature_names):
        raise ValueError("canonical feature order does not match local-compression feature config")
    raw_map = _raw_feature_map(canonical)
    pf_map = _pf_feature_map(canonical)
    derived_map = _derived_feature_map(canonical)
    values_by_modality: dict[str, Any] = {}
    feature_names_by_modality: dict[str, tuple[str, ...]] = {}
    source_names_by_modality: dict[str, tuple[str, ...]] = {}
    for spec in feature_config.modalities:
        values, feature_names, source_names = _values_for_modality(
            spec,
            raw_map=raw_map,
            pf_map=pf_map,
            derived_map=derived_map,
            particle_mask=canonical.particle_mask,
        )
        values_by_modality[spec.name] = values
        feature_names_by_modality[spec.name] = feature_names
        source_names_by_modality[spec.name] = source_names
    modality_mask = canonical.particle_mask[:, :, None].expand(
        canonical.batch_size,
        canonical.num_particles,
        feature_config.num_modalities,
    )
    modality_mask = modality_mask.clone().to(dtype=torch.bool)
    return LocalCompressionModalities(
        values_by_modality=values_by_modality,
        modality_mask=modality_mask,
        particle_mask=canonical.particle_mask,
        feature_names_by_modality=feature_names_by_modality,
        source_names_by_modality=source_names_by_modality,
        modality_names=feature_config.modality_names,
        feature_config=feature_config,
    )


def build_local_compression_modalities_from_tokens(
    tokens: Any,
    mask: Any,
    *,
    weights: Any | None = None,
    max_constits: int | None = 128,
    weight_threshold: float = 0.0,
    config: LocalCompressionFeatureConfig | Mapping[str, Any] | None = None,
) -> tuple[LocalCompressionCanonicalInputs, LocalCompressionModalities]:
    """Convenience wrapper for tests and later lightweight scripts."""

    feature_config = normalize_local_compression_feature_config(config)
    canonical = build_local_compression_canonical_inputs(
        tokens,
        mask,
        weights=weights,
        max_constits=max_constits,
        weight_threshold=weight_threshold,
        config=feature_config,
    )
    modalities = build_local_compression_modalities(canonical, config=feature_config)
    return canonical, modalities

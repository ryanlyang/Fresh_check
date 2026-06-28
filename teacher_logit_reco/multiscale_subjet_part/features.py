"""Raw-token feature and geometry helpers for multi-scale subjet models.

This module is intentionally model-free.  It validates the raw JetClass HLT
token contract, builds canonical eta-phi/kinematic helper tensors, and provides
scale metadata used by later seeded soft-subjet assignment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.dual_view import build_part_inputs_torch
from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES, PF_POINT_NAMES, PF_VECTOR_NAMES


MULTISCALE_SUBJET_FEATURE_CONTRACT = "multiscale_subjet_feature_helpers_v1"
MULTISCALE_SUBJET_COORDINATE_DIM = 2
MULTISCALE_SUBJET_CANONICAL_PART_INPUT_CONTRACT = "multiscale_subjet_canonical_part_inputs_v1"

CANONICAL_PART_FEATURE_NAMES = tuple(PF_FEATURE_NAMES)
CANONICAL_PART_POINT_NAMES = tuple(PF_POINT_NAMES)
CANONICAL_PART_VECTOR_NAMES = tuple(PF_VECTOR_NAMES)


@dataclass(frozen=True)
class SubjetScaleSpec:
    """One named subjet scale family."""

    name: str
    num_tokens: int
    radius_min: float
    radius_max: float
    role: str

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ValueError("scale name must be non-empty")
        num_tokens = int(self.num_tokens)
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        radius_min = float(self.radius_min)
        radius_max = float(self.radius_max)
        if not math.isfinite(radius_min) or not math.isfinite(radius_max):
            raise ValueError("scale radii must be finite")
        if radius_min < 0.0 or radius_max <= 0.0 or radius_min >= radius_max:
            raise ValueError("scale radii must satisfy 0 <= radius_min < radius_max")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "num_tokens", num_tokens)
        object.__setattr__(self, "radius_min", radius_min)
        object.__setattr__(self, "radius_max", radius_max)
        object.__setattr__(self, "role", str(self.role))

    @property
    def radius_center(self) -> float:
        return 0.5 * (float(self.radius_min) + float(self.radius_max))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["radius_center"] = self.radius_center
        return payload


MULTISCALE_SUBJET_DEFAULT_SCALE_SPECS: tuple[SubjetScaleSpec, ...] = (
    SubjetScaleSpec("small", 8, 0.05, 0.12, "tight local cores and HLT merge/drop artifacts"),
    SubjetScaleSpec("medium", 8, 0.12, 0.25, "candidate prongs and proto-subjets"),
    SubjetScaleSpec("large", 4, 0.25, 0.50, "broad radiation regions and jet-scale context"),
)
MULTISCALE_SUBJET_SCALE_PROFILES = (
    "default",
    "one_scale_small",
    "one_scale_medium",
    "one_scale_large",
    "few_subjets",
    "many_subjets",
)


@dataclass(frozen=True)
class MultiscaleSubjetFeatureConfig:
    """Column contract for raw HLT token feature helpers."""

    raw_token_dim: int = RAW_TOKEN_DIM
    pt_index: int = 0
    eta_index: int = 1
    phi_index: int = 2
    energy_index: int = 3
    eta_clip: float = 5.0
    eps: float = 1.0e-8
    energy_floor_eps: float = 1.0e-4
    default_density_radii: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)

    def __post_init__(self) -> None:
        raw_token_dim = int(self.raw_token_dim)
        if raw_token_dim != RAW_TOKEN_DIM:
            raise ValueError(f"multi-scale subjet helpers expect RAW_TOKEN_DIM={RAW_TOKEN_DIM}")
        object.__setattr__(self, "raw_token_dim", raw_token_dim)
        for name in ("pt_index", "eta_index", "phi_index", "energy_index"):
            index = int(getattr(self, name))
            if index < 0 or index >= raw_token_dim:
                raise ValueError(f"{name}={index} is outside raw_token_dim={raw_token_dim}")
            object.__setattr__(self, name, index)
        eta_clip = float(self.eta_clip)
        eps = float(self.eps)
        energy_floor_eps = float(self.energy_floor_eps)
        if not math.isfinite(eta_clip) or eta_clip <= 0.0:
            raise ValueError("eta_clip must be positive and finite")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        if not math.isfinite(energy_floor_eps) or energy_floor_eps <= 0.0:
            raise ValueError("energy_floor_eps must be positive and finite")
        radii = tuple(float(radius) for radius in self.default_density_radii)
        if not radii or any((not math.isfinite(radius) or radius <= 0.0) for radius in radii):
            raise ValueError("default_density_radii must be positive finite values")
        if tuple(sorted(radii)) != radii:
            raise ValueError("default_density_radii must be sorted ascending")
        object.__setattr__(self, "eta_clip", eta_clip)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "energy_floor_eps", energy_floor_eps)
        object.__setattr__(self, "default_density_radii", radii)


@dataclass(frozen=True)
class PreparedSubjetInputs:
    """Canonical raw-token helper tensors for one HLT particle batch."""

    tokens: Any
    mask: Any
    pt: Any
    eta: Any
    phi: Any
    energy: Any
    px: Any
    py: Any
    pz: Any
    coordinates: Any
    pt_fraction: Any
    config: MultiscaleSubjetFeatureConfig

    def summary(self) -> dict[str, Any]:
        return {
            "contract": MULTISCALE_SUBJET_FEATURE_CONTRACT,
            "tokens_shape": list(self.tokens.shape),
            "mask_shape": list(self.mask.shape),
            "coordinates_shape": list(self.coordinates.shape),
            "raw_token_dim": int(self.config.raw_token_dim),
            "valid_particle_count": int(self.mask.sum().detach().cpu().item()),
        }


@dataclass(frozen=True)
class LocalDensityOutput:
    """Per-particle local density features at multiple eta-phi radii."""

    counts: Any
    pt_fraction_sums: Any
    radii: tuple[float, ...]
    include_self: bool

    def summary(self) -> dict[str, Any]:
        return {
            "contract": MULTISCALE_SUBJET_FEATURE_CONTRACT,
            "counts_shape": list(self.counts.shape),
            "pt_fraction_sums_shape": list(self.pt_fraction_sums.shape),
            "radii": list(self.radii),
            "include_self": bool(self.include_self),
        }


@dataclass(frozen=True)
class CanonicalPartInputs:
    """Canonical Particle Transformer tensors built from the same raw HLT view.

    The multi-scale subjet branch will use raw ``eta``/``phi`` helpers to build
    soft subjet tokens, but its serious comparison path must still anchor into
    the exact ParT feature contract used by the HLT baseline.  This wrapper is a
    small typed boundary around ``jetclass_fresh.dual_view.build_part_inputs_torch``.
    """

    points: Any
    features: Any
    lorentz_vectors: Any
    mask: Any
    feature_names: tuple[str, ...] = CANONICAL_PART_FEATURE_NAMES
    point_names: tuple[str, ...] = CANONICAL_PART_POINT_NAMES
    vector_names: tuple[str, ...] = CANONICAL_PART_VECTOR_NAMES
    max_constits: int = 128
    weight_threshold: float = 0.0

    def __post_init__(self) -> None:
        if int(self.features.ndim) != 3:
            raise ValueError(f"features must have shape [batch, features, particles], got {tuple(self.features.shape)}")
        batch_size = int(self.features.shape[0])
        n_constits = int(self.features.shape[2])
        expected = {
            "points": (batch_size, len(self.point_names), n_constits),
            "features": (batch_size, len(self.feature_names), n_constits),
            "lorentz_vectors": (batch_size, len(self.vector_names), n_constits),
            "mask": (batch_size, 1, n_constits),
        }
        actual = {
            "points": tuple(self.points.shape),
            "features": tuple(self.features.shape),
            "lorentz_vectors": tuple(self.lorentz_vectors.shape),
            "mask": tuple(self.mask.shape),
        }
        for key, shape in expected.items():
            if actual[key] != shape:
                raise ValueError(f"{key} has shape {actual[key]}, expected {shape}")
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
    def num_constits(self) -> int:
        return int(self.features.shape[2])

    def feature_rows(self) -> Any:
        """Return ``[batch, particles, PF_FEATURE_NAMES]`` for local adapters."""

        return self.features.transpose(1, 2).contiguous()

    def as_part_kwargs(self) -> dict[str, Any]:
        """Return tensors with the keyword names expected by ParT wrappers."""

        return {
            "points": self.points,
            "features": self.features,
            "lorentz_vectors": self.lorentz_vectors,
            "mask": self.mask,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "contract": MULTISCALE_SUBJET_CANONICAL_PART_INPUT_CONTRACT,
            "points_shape": list(self.points.shape),
            "features_shape": list(self.features.shape),
            "lorentz_vectors_shape": list(self.lorentz_vectors.shape),
            "mask_shape": list(self.mask.shape),
            "feature_names": list(self.feature_names),
            "point_names": list(self.point_names),
            "vector_names": list(self.vector_names),
            "max_constits": int(self.max_constits),
            "weight_threshold": float(self.weight_threshold),
            "valid_particle_count": int(self.mask.sum().detach().cpu().item()),
        }


def normalize_feature_config(
    config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None = None,
) -> MultiscaleSubjetFeatureConfig:
    if config is None:
        return MultiscaleSubjetFeatureConfig()
    if isinstance(config, MultiscaleSubjetFeatureConfig):
        return config
    return MultiscaleSubjetFeatureConfig(**dict(config))


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _as_float_tensor(value: Any) -> Any:
    torch = require_torch()
    if not isinstance(value, torch.Tensor):
        return torch.as_tensor(value, dtype=torch.float32)
    return value.float()


def _as_bool_tensor(value: Any, *, device: Any) -> Any:
    torch = require_torch()
    if not isinstance(value, torch.Tensor):
        return torch.as_tensor(value, dtype=torch.bool, device=device)
    return value.to(device=device, dtype=torch.bool)


def prepare_subjet_tokens_and_mask(
    tokens: Any,
    mask: Any,
    config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None = None,
) -> tuple[Any, Any, MultiscaleSubjetFeatureConfig]:
    """Validate, sanitize, cast, and mask raw HLT tokens."""

    torch = require_torch()
    feature_config = normalize_feature_config(config)
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
    tokens = _nan_to_num_torch(tokens)
    pt = torch.clamp(tokens[:, :, int(feature_config.pt_index)], min=0.0)
    eta = torch.clamp(
        tokens[:, :, int(feature_config.eta_index)],
        -float(feature_config.eta_clip),
        float(feature_config.eta_clip),
    )
    phi = wrap_phi(tokens[:, :, int(feature_config.phi_index)])
    energy_floor = pt * torch.cosh(eta) + float(feature_config.energy_floor_eps)
    energy = torch.maximum(
        torch.clamp(tokens[:, :, int(feature_config.energy_index)], min=float(feature_config.energy_floor_eps)),
        energy_floor,
    )
    tokens = tokens.clone()
    tokens[:, :, int(feature_config.pt_index)] = pt
    tokens[:, :, int(feature_config.eta_index)] = eta
    tokens[:, :, int(feature_config.phi_index)] = phi
    tokens[:, :, int(feature_config.energy_index)] = energy
    tokens = torch.where(mask[:, :, None], tokens, torch.zeros_like(tokens))
    return tokens, mask, feature_config


def wrap_delta_phi(delta_phi: Any) -> Any:
    """Wrap a phi difference tensor to ``[-pi, pi]``."""

    torch = require_torch()
    delta_phi = _as_float_tensor(delta_phi)
    return torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))


def wrap_phi(phi: Any) -> Any:
    """Wrap an absolute phi tensor to ``[-pi, pi]``."""

    return wrap_delta_phi(phi)


def eta_phi_coordinates(
    tokens: Any,
    mask: Any,
    config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None = None,
) -> Any:
    """Return finite ``(eta, phi)`` coordinates from raw HLT tokens."""

    torch = require_torch()
    tokens, mask, feature_config = prepare_subjet_tokens_and_mask(tokens, mask, config=config)
    eta = torch.clamp(tokens[:, :, int(feature_config.eta_index)], -float(feature_config.eta_clip), float(feature_config.eta_clip))
    phi = wrap_phi(tokens[:, :, int(feature_config.phi_index)])
    coords = torch.stack([eta, phi], dim=-1)
    return torch.where(mask[:, :, None], coords, torch.zeros_like(coords))


def pairwise_eta_phi_deltas(coords: Any) -> tuple[Any, Any]:
    """Return pairwise ``delta_eta`` and wrapped ``delta_phi`` from coordinates."""

    torch = require_torch()
    coords = _as_float_tensor(coords)
    if int(coords.ndim) != 3 or int(coords.shape[-1]) != MULTISCALE_SUBJET_COORDINATE_DIM:
        raise ValueError(
            f"coords must have shape [batch, particles, {MULTISCALE_SUBJET_COORDINATE_DIM}], got {tuple(coords.shape)}"
        )
    coords = _nan_to_num_torch(coords)
    delta_eta = coords[:, :, None, 0] - coords[:, None, :, 0]
    delta_phi = wrap_delta_phi(coords[:, :, None, 1] - coords[:, None, :, 1])
    return delta_eta, delta_phi


def pairwise_delta_r(coords: Any) -> Any:
    """Return exact pairwise ``deltaR`` distances from ``(eta, phi)`` coordinates."""

    torch = require_torch()
    delta_eta, delta_phi = pairwise_eta_phi_deltas(coords)
    return torch.sqrt(torch.clamp(delta_eta * delta_eta + delta_phi * delta_phi, min=0.0))


def particle_pt_fraction(
    tokens: Any,
    mask: Any,
    config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None = None,
) -> Any:
    """Return per-particle ``pt / sum_valid_pt`` with masked particles set to zero."""

    torch = require_torch()
    tokens, mask, feature_config = prepare_subjet_tokens_and_mask(tokens, mask, config=config)
    pt = torch.clamp(tokens[:, :, int(feature_config.pt_index)], min=0.0)
    pt = torch.where(mask, pt, torch.zeros_like(pt))
    denom = torch.clamp(pt.sum(dim=1, keepdim=True), min=float(feature_config.eps))
    return torch.where(mask, pt / denom, torch.zeros_like(pt))


def build_prepared_subjet_inputs(
    tokens: Any,
    mask: Any,
    config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None = None,
) -> PreparedSubjetInputs:
    """Build canonical per-particle kinematic helper tensors."""

    torch = require_torch()
    tokens, mask, feature_config = prepare_subjet_tokens_and_mask(tokens, mask, config=config)
    pt = torch.clamp(tokens[:, :, int(feature_config.pt_index)], min=0.0)
    eta = torch.clamp(tokens[:, :, int(feature_config.eta_index)], -float(feature_config.eta_clip), float(feature_config.eta_clip))
    phi = wrap_phi(tokens[:, :, int(feature_config.phi_index)])
    energy = tokens[:, :, int(feature_config.energy_index)]
    pt = torch.where(mask, pt, torch.zeros_like(pt))
    eta = torch.where(mask, eta, torch.zeros_like(eta))
    phi = torch.where(mask, phi, torch.zeros_like(phi))
    energy = torch.where(mask, energy, torch.zeros_like(energy))
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    coordinates = torch.stack([eta, phi], dim=-1)
    pt_frac = particle_pt_fraction(tokens, mask, feature_config)
    return PreparedSubjetInputs(
        tokens=tokens,
        mask=mask,
        pt=pt,
        eta=eta,
        phi=phi,
        energy=energy,
        px=px,
        py=py,
        pz=pz,
        coordinates=coordinates,
        pt_fraction=pt_frac,
        config=feature_config,
    )


def _coerce_radii(
    radii: tuple[float, ...] | list[float] | None,
    config: MultiscaleSubjetFeatureConfig,
) -> tuple[float, ...]:
    values = tuple(float(radius) for radius in (config.default_density_radii if radii is None else radii))
    if not values:
        raise ValueError("radii must be non-empty")
    if any((not math.isfinite(radius) or radius <= 0.0) for radius in values):
        raise ValueError("radii must be positive finite values")
    if tuple(sorted(values)) != values:
        raise ValueError("radii must be sorted ascending")
    return values


def local_density_features(
    tokens: Any,
    mask: Any,
    *,
    radii: tuple[float, ...] | list[float] | None = None,
    include_self: bool = False,
    config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None = None,
) -> LocalDensityOutput:
    """Count local valid neighbors and pt-fraction mass inside eta-phi radii."""

    torch = require_torch()
    tokens, mask, feature_config = prepare_subjet_tokens_and_mask(tokens, mask, config=config)
    radii_tuple = _coerce_radii(radii, feature_config)
    coords = eta_phi_coordinates(tokens, mask, feature_config)
    distances = pairwise_delta_r(coords)
    batch_size, num_particles = mask.shape
    candidate_mask = mask[:, None, :].expand(batch_size, num_particles, num_particles).clone()
    query_mask = mask[:, :, None]
    valid_pairs = candidate_mask & query_mask
    if not bool(include_self) and int(num_particles) > 0:
        eye = torch.eye(num_particles, dtype=torch.bool, device=mask.device)[None, :, :]
        valid_pairs = valid_pairs & ~eye

    pt_fraction = particle_pt_fraction(tokens, mask, feature_config)
    count_rows = []
    pt_rows = []
    for radius in radii_tuple:
        within = valid_pairs & (distances <= float(radius))
        count_rows.append(within.sum(dim=-1).to(dtype=tokens.dtype))
        pt_rows.append((within.to(dtype=tokens.dtype) * pt_fraction[:, None, :]).sum(dim=-1))
    counts = torch.stack(count_rows, dim=-1)
    pt_fraction_sums = torch.stack(pt_rows, dim=-1)
    counts = torch.where(mask[:, :, None], counts, torch.zeros_like(counts))
    pt_fraction_sums = torch.where(mask[:, :, None], pt_fraction_sums, torch.zeros_like(pt_fraction_sums))
    return LocalDensityOutput(
        counts=counts,
        pt_fraction_sums=pt_fraction_sums,
        radii=radii_tuple,
        include_self=bool(include_self),
    )


def build_canonical_part_inputs(
    tokens: Any,
    mask: Any,
    *,
    weights: Any | None = None,
    max_constits: int | None = 128,
    weight_threshold: float = 0.0,
    config: MultiscaleSubjetFeatureConfig | Mapping[str, Any] | None = None,
) -> CanonicalPartInputs:
    """Build the exact canonical ParT tensors for the supplied raw HLT tokens.

    Later multi-scale subjet models should call this rather than reimplementing
    ParT preprocessing.  That keeps the baseline and adapter paths on the same
    normalized features, points, Lorentz vectors, mask convention, and top-k
    constituent ordering.
    """

    torch = require_torch()
    tokens, mask, _ = prepare_subjet_tokens_and_mask(tokens, mask, config=config)
    resolved_max_constits = int(tokens.shape[1]) if max_constits is None else int(max_constits)
    if resolved_max_constits <= 0:
        raise ValueError("max_constits must be positive")
    if float(weight_threshold) < 0.0:
        raise ValueError("weight_threshold must be non-negative")
    resolved_weights = None
    if weights is not None:
        resolved_weights = _as_float_tensor(weights).to(device=tokens.device)
        if tuple(resolved_weights.shape) != tuple(mask.shape):
            raise ValueError(f"weights shape {tuple(resolved_weights.shape)} does not match mask shape {tuple(mask.shape)}")

    part_inputs = build_part_inputs_torch(
        tokens,
        mask,
        weights=resolved_weights,
        max_constits=resolved_max_constits,
        weight_threshold=float(weight_threshold),
    )
    points = torch.as_tensor(part_inputs["points"], device=tokens.device).float()
    features = torch.as_tensor(part_inputs["features"], device=tokens.device).float()
    lorentz_vectors = torch.as_tensor(part_inputs["lorentz_vectors"], device=tokens.device).float()
    part_mask = torch.as_tensor(part_inputs["mask"], device=tokens.device).bool()
    return CanonicalPartInputs(
        points=points,
        features=features,
        lorentz_vectors=lorentz_vectors,
        mask=part_mask,
        feature_names=CANONICAL_PART_FEATURE_NAMES,
        point_names=CANONICAL_PART_POINT_NAMES,
        vector_names=CANONICAL_PART_VECTOR_NAMES,
        max_constits=resolved_max_constits,
        weight_threshold=float(weight_threshold),
    )


def multiscale_subjet_scale_specs_for_profile(profile: str = "default") -> tuple[SubjetScaleSpec, ...]:
    """Return named Step 14 scale layouts for serious and ablation runs."""

    clean = str(profile).strip().lower().replace("-", "_")
    aliases = {
        "default": "default",
        "multi": "default",
        "multiscale": "default",
        "three_scale": "default",
        "small_only": "one_scale_small",
        "one_scale_small": "one_scale_small",
        "medium_only": "one_scale_medium",
        "one_scale": "one_scale_medium",
        "one_scale_medium": "one_scale_medium",
        "large_only": "one_scale_large",
        "one_scale_large": "one_scale_large",
        "few": "few_subjets",
        "fewer": "few_subjets",
        "few_subjets": "few_subjets",
        "many": "many_subjets",
        "more": "many_subjets",
        "many_subjets": "many_subjets",
    }
    if clean not in aliases:
        raise ValueError(f"unknown multiscale subjet scale profile {profile!r}; expected one of {MULTISCALE_SUBJET_SCALE_PROFILES}")
    clean = aliases[clean]
    if clean == "default":
        return tuple(MULTISCALE_SUBJET_DEFAULT_SCALE_SPECS)
    if clean == "one_scale_small":
        return (SubjetScaleSpec("small", 20, 0.05, 0.12, "single-scale tight local subjet ablation"),)
    if clean == "one_scale_medium":
        return (SubjetScaleSpec("medium", 20, 0.12, 0.25, "single-scale proto-subjet ablation"),)
    if clean == "one_scale_large":
        return (SubjetScaleSpec("large", 20, 0.25, 0.50, "single-scale broad-context ablation"),)
    if clean == "few_subjets":
        return (
            SubjetScaleSpec("small", 4, 0.05, 0.12, "low-token tight local cores"),
            SubjetScaleSpec("medium", 4, 0.12, 0.25, "low-token proto-subjets"),
            SubjetScaleSpec("large", 2, 0.25, 0.50, "low-token broad context"),
        )
    if clean == "many_subjets":
        return (
            SubjetScaleSpec("small", 16, 0.05, 0.12, "high-token tight local cores"),
            SubjetScaleSpec("medium", 16, 0.12, 0.25, "high-token proto-subjets"),
            SubjetScaleSpec("large", 8, 0.25, 0.50, "high-token broad context"),
        )
    raise AssertionError(f"unhandled scale profile {clean!r}")


def default_subjet_scale_specs() -> tuple[SubjetScaleSpec, ...]:
    """Return the default small/medium/large scale metadata."""

    return multiscale_subjet_scale_specs_for_profile("default")


def scale_radius_bounds(scale_specs: tuple[SubjetScaleSpec, ...] | None = None) -> dict[str, tuple[float, float]]:
    """Map scale name to ``(radius_min, radius_max)``."""

    specs = default_subjet_scale_specs() if scale_specs is None else tuple(scale_specs)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"scale names must be unique, got {names}")
    return {spec.name: (float(spec.radius_min), float(spec.radius_max)) for spec in specs}

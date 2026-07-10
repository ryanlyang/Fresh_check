"""Canonical jet-state token layout for CMS-JS experiments.

This module is intentionally deterministic and data-free.  It freezes the
token and field coordinate system that later Phi builders, residual
predictors, caches, and reports must all share.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


CANONICAL_STATE_LAYOUT_VERSION = "canonical_jet_state_layout_v1"

CANONICAL_STATE_TOKEN_FAMILIES: tuple[str, ...] = (
    "global",
    "radial",
    "angular",
    "anchor_coarse",
    "anchor_medium",
    "anchor_fine",
)
CANONICAL_STATE_TOKEN_TYPE_IDS: dict[str, int] = {
    family: index for index, family in enumerate(CANONICAL_STATE_TOKEN_FAMILIES)
}
CANONICAL_STATE_SCALE_IDS: dict[str, int] = dict(CANONICAL_STATE_TOKEN_TYPE_IDS)

CANONICAL_STATE_GLOBAL_TOKENS: tuple[str, ...] = (
    "global_energy_shape",
    "global_pid_fractions",
    "global_multiplicity_softness",
    "global_leading_structure",
)

CANONICAL_STATE_FIELD_NAMES: tuple[str, ...] = (
    "sum_pt_frac",
    "sum_energy_frac",
    "log1p_count",
    "mean_pt_frac",
    "max_pt_frac",
    "pt_weighted_mean_deta",
    "pt_weighted_mean_dphi",
    "pt_weighted_var_deta",
    "pt_weighted_var_dphi",
    "mass_proxy",
    "width_proxy",
    "charged_pt_frac",
    "neutral_pt_frac",
    "photon_pt_frac",
    "electron_pt_frac",
    "muon_pt_frac",
    "hadron_pt_frac",
    "quality_or_missingness_proxy",
)

CANONICAL_STATE_FEATURE_SCALES: dict[str, float] = {
    "sum_pt_frac": 1.0,
    "sum_energy_frac": 1.0,
    "log1p_count": 5.0,
    "mean_pt_frac": 1.0,
    "max_pt_frac": 1.0,
    "pt_weighted_mean_deta": 0.4,
    "pt_weighted_mean_dphi": 0.4,
    "pt_weighted_var_deta": 0.16,
    "pt_weighted_var_dphi": 0.16,
    "mass_proxy": 0.5,
    "width_proxy": 0.5,
    "charged_pt_frac": 1.0,
    "neutral_pt_frac": 1.0,
    "photon_pt_frac": 1.0,
    "electron_pt_frac": 1.0,
    "muon_pt_frac": 1.0,
    "hadron_pt_frac": 1.0,
    "quality_or_missingness_proxy": 1.0,
}

CANONICAL_STATE_RESIDUAL_SCALES: dict[str, float] = {
    "sum_pt_frac": 0.35,
    "sum_energy_frac": 0.35,
    "log1p_count": 0.75,
    "mean_pt_frac": 0.25,
    "max_pt_frac": 0.30,
    "pt_weighted_mean_deta": 0.12,
    "pt_weighted_mean_dphi": 0.12,
    "pt_weighted_var_deta": 0.05,
    "pt_weighted_var_dphi": 0.05,
    "mass_proxy": 0.25,
    "width_proxy": 0.20,
    "charged_pt_frac": 0.35,
    "neutral_pt_frac": 0.35,
    "photon_pt_frac": 0.35,
    "electron_pt_frac": 0.20,
    "muon_pt_frac": 0.20,
    "hadron_pt_frac": 0.35,
    "quality_or_missingness_proxy": 0.50,
}

CANONICAL_STATE_RADIAL_BIN_EDGES: tuple[float | None, ...] = (
    0.00,
    0.03,
    0.06,
    0.10,
    0.15,
    0.22,
    0.30,
    0.40,
    None,
)
CANONICAL_STATE_ANGULAR_SECTORS = 8
CANONICAL_STATE_ANCHOR_COUNTS: dict[str, int] = {
    "anchor_coarse": 4,
    "anchor_medium": 8,
    "anchor_fine": 16,
}
CANONICAL_STATE_ANCHOR_RADII: dict[str, float] = {
    "anchor_coarse": 0.30,
    "anchor_medium": 0.18,
    "anchor_fine": 0.10,
}


def _require_exact_mapping_keys(name: str, mapping: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    actual = set(mapping.keys())
    expected = set(keys)
    if actual != expected:
        missing = tuple(key for key in keys if key not in actual)
        extra = tuple(key for key in mapping if key not in expected)
        raise ValueError(f"{name} keys must match {keys}; missing={missing}, extra={extra}")


@dataclass(frozen=True)
class CanonicalStateTokenSpec:
    """One fixed coordinate in the canonical state token sequence."""

    index: int
    name: str
    family: str
    token_type_id: int
    scale_id: int
    slot_id: int
    ring_id: int = -1
    sector_id: int = -1
    radius_inner: float | None = None
    radius_outer: float | None = None
    angular_center: float | None = None
    angular_width: float | None = None
    anchor_radius: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalStateTokenSpec":
        return cls(
            index=int(payload["index"]),
            name=str(payload["name"]),
            family=str(payload["family"]),
            token_type_id=int(payload["token_type_id"]),
            scale_id=int(payload["scale_id"]),
            slot_id=int(payload["slot_id"]),
            ring_id=int(payload.get("ring_id", -1)),
            sector_id=int(payload.get("sector_id", -1)),
            radius_inner=None if payload.get("radius_inner") is None else float(payload["radius_inner"]),
            radius_outer=None if payload.get("radius_outer") is None else float(payload["radius_outer"]),
            angular_center=None if payload.get("angular_center") is None else float(payload["angular_center"]),
            angular_width=None if payload.get("angular_width") is None else float(payload["angular_width"]),
            anchor_radius=None if payload.get("anchor_radius") is None else float(payload["anchor_radius"]),
        )


@dataclass(frozen=True)
class CanonicalJetStateConfig:
    """Frozen field and token-layout configuration for `Phi(jet)`."""

    layout_version: str = CANONICAL_STATE_LAYOUT_VERSION
    field_names: tuple[str, ...] = CANONICAL_STATE_FIELD_NAMES
    feature_scales: Mapping[str, float] = field(default_factory=lambda: dict(CANONICAL_STATE_FEATURE_SCALES))
    residual_scales: Mapping[str, float] = field(default_factory=lambda: dict(CANONICAL_STATE_RESIDUAL_SCALES))
    global_tokens: tuple[str, ...] = CANONICAL_STATE_GLOBAL_TOKENS
    radial_bin_edges: tuple[float | None, ...] = CANONICAL_STATE_RADIAL_BIN_EDGES
    angular_sectors: int = CANONICAL_STATE_ANGULAR_SECTORS
    anchor_counts: Mapping[str, int] = field(default_factory=lambda: dict(CANONICAL_STATE_ANCHOR_COUNTS))
    anchor_radii: Mapping[str, float] = field(default_factory=lambda: dict(CANONICAL_STATE_ANCHOR_RADII))

    def __post_init__(self) -> None:
        if str(self.layout_version) != CANONICAL_STATE_LAYOUT_VERSION:
            raise ValueError(f"layout_version must be {CANONICAL_STATE_LAYOUT_VERSION}")
        field_names = tuple(str(name) for name in self.field_names)
        if field_names != CANONICAL_STATE_FIELD_NAMES:
            raise ValueError("Canonical state field order is frozen; do not reorder or rename fields")
        feature_scales = {str(key): float(value) for key, value in self.feature_scales.items()}
        residual_scales = {str(key): float(value) for key, value in self.residual_scales.items()}
        _require_exact_mapping_keys("feature_scales", feature_scales, field_names)
        _require_exact_mapping_keys("residual_scales", residual_scales, field_names)
        feature_scales = {name: feature_scales[name] for name in field_names}
        residual_scales = {name: residual_scales[name] for name in field_names}
        for name, value in feature_scales.items():
            if value <= 0.0:
                raise ValueError(f"feature scale for {name} must be positive")
        for name, value in residual_scales.items():
            if not (0.0 < value <= 1.0):
                raise ValueError(f"residual scale for {name} must be in (0, 1]")
        global_tokens = tuple(str(name) for name in self.global_tokens)
        if global_tokens != CANONICAL_STATE_GLOBAL_TOKENS:
            raise ValueError("global token order is frozen")
        radial_edges = tuple(self.radial_bin_edges)
        if len(radial_edges) != 9:
            raise ValueError("radial_bin_edges must define 8 rings")
        if radial_edges[-1] is not None:
            raise ValueError("final radial bin must be open-ended and encoded as None")
        finite_edges = [float(value) for value in radial_edges[:-1]]
        if finite_edges[0] != 0.0 or any(b <= a for a, b in zip(finite_edges, finite_edges[1:])):
            raise ValueError("finite radial bin edges must be strictly increasing from 0.0")
        angular_sectors = int(self.angular_sectors)
        if angular_sectors != CANONICAL_STATE_ANGULAR_SECTORS:
            raise ValueError(f"angular_sectors must be {CANONICAL_STATE_ANGULAR_SECTORS}")
        anchor_counts = {str(key): int(value) for key, value in self.anchor_counts.items()}
        anchor_radii = {str(key): float(value) for key, value in self.anchor_radii.items()}
        _require_exact_mapping_keys("anchor_counts", anchor_counts, tuple(CANONICAL_STATE_ANCHOR_COUNTS.keys()))
        _require_exact_mapping_keys("anchor_radii", anchor_radii, tuple(CANONICAL_STATE_ANCHOR_RADII.keys()))
        anchor_counts = {family: anchor_counts[family] for family in CANONICAL_STATE_ANCHOR_COUNTS}
        anchor_radii = {family: anchor_radii[family] for family in CANONICAL_STATE_ANCHOR_RADII}
        for family, count in anchor_counts.items():
            if count != CANONICAL_STATE_ANCHOR_COUNTS[family]:
                raise ValueError(f"{family} count must be {CANONICAL_STATE_ANCHOR_COUNTS[family]}")
        for family, radius in anchor_radii.items():
            if abs(radius - CANONICAL_STATE_ANCHOR_RADII[family]) > 1.0e-12:
                raise ValueError(f"{family} anchor radius must be {CANONICAL_STATE_ANCHOR_RADII[family]}")
        object.__setattr__(self, "field_names", field_names)
        object.__setattr__(self, "feature_scales", feature_scales)
        object.__setattr__(self, "residual_scales", residual_scales)
        object.__setattr__(self, "global_tokens", global_tokens)
        object.__setattr__(self, "radial_bin_edges", tuple(radial_edges))
        object.__setattr__(self, "angular_sectors", angular_sectors)
        object.__setattr__(self, "anchor_counts", anchor_counts)
        object.__setattr__(self, "anchor_radii", anchor_radii)

    @property
    def d_phi(self) -> int:
        return len(self.field_names)

    @property
    def k_state(self) -> int:
        return (
            len(self.global_tokens)
            + (len(self.radial_bin_edges) - 1)
            + int(self.angular_sectors)
            + sum(int(value) for value in self.anchor_counts.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_version": self.layout_version,
            "field_names": list(self.field_names),
            "feature_scales": {name: float(self.feature_scales[name]) for name in self.field_names},
            "residual_scales": {name: float(self.residual_scales[name]) for name in self.field_names},
            "global_tokens": list(self.global_tokens),
            "radial_bin_edges": list(self.radial_bin_edges),
            "angular_sectors": int(self.angular_sectors),
            "anchor_counts": dict(self.anchor_counts),
            "anchor_radii": dict(self.anchor_radii),
            "k_state": int(self.k_state),
            "d_phi": int(self.d_phi),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalJetStateConfig":
        return cls(
            layout_version=str(payload.get("layout_version", CANONICAL_STATE_LAYOUT_VERSION)),
            field_names=tuple(str(name) for name in payload["field_names"]),
            feature_scales=dict(payload["feature_scales"]),
            residual_scales=dict(payload["residual_scales"]),
            global_tokens=tuple(str(name) for name in payload["global_tokens"]),
            radial_bin_edges=tuple(payload["radial_bin_edges"]),
            angular_sectors=int(payload["angular_sectors"]),
            anchor_counts=dict(payload["anchor_counts"]),
            anchor_radii=dict(payload["anchor_radii"]),
        )


@dataclass(frozen=True)
class CanonicalJetStateLayout:
    """Concrete token metadata generated from `CanonicalJetStateConfig`."""

    config: CanonicalJetStateConfig = field(default_factory=CanonicalJetStateConfig)
    token_specs: tuple[CanonicalStateTokenSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        config = self.config if isinstance(self.config, CanonicalJetStateConfig) else CanonicalJetStateConfig.from_dict(self.config)
        token_specs = tuple(self.token_specs) if self.token_specs else build_token_specs(config)
        if len(token_specs) != config.k_state:
            raise ValueError(f"token_specs length is {len(token_specs)}, expected {config.k_state}")
        for expected_index, spec in enumerate(token_specs):
            if int(spec.index) != expected_index:
                raise ValueError("token_specs must be stored in index order")
            if spec.family not in CANONICAL_STATE_TOKEN_TYPE_IDS:
                raise ValueError(f"unknown token family {spec.family!r}")
            if spec.token_type_id != CANONICAL_STATE_TOKEN_TYPE_IDS[spec.family]:
                raise ValueError(f"token_type_id mismatch for {spec.name}")
            if spec.scale_id != CANONICAL_STATE_SCALE_IDS[spec.family]:
                raise ValueError(f"scale_id mismatch for {spec.name}")
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "token_specs", token_specs)

    @property
    def k_state(self) -> int:
        return len(self.token_specs)

    @property
    def d_phi(self) -> int:
        return self.config.d_phi

    @property
    def token_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.token_specs)

    @property
    def token_type_ids(self) -> tuple[int, ...]:
        return tuple(spec.token_type_id for spec in self.token_specs)

    @property
    def scale_ids(self) -> tuple[int, ...]:
        return tuple(spec.scale_id for spec in self.token_specs)

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(spec.slot_id for spec in self.token_specs)

    def family_slices(self) -> dict[str, tuple[int, int]]:
        slices: dict[str, tuple[int, int]] = {}
        start = 0
        for family in CANONICAL_STATE_TOKEN_FAMILIES:
            count = sum(1 for spec in self.token_specs if spec.family == family)
            slices[family] = (start, start + count)
            start += count
        return slices

    def feature_scale_vector(self) -> tuple[float, ...]:
        return tuple(float(self.config.feature_scales[name]) for name in self.config.field_names)

    def residual_scale_vector(self) -> tuple[float, ...]:
        return tuple(float(self.config.residual_scales[name]) for name in self.config.field_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_version": self.config.layout_version,
            "config": self.config.to_dict(),
            "token_specs": [spec.to_dict() for spec in self.token_specs],
            "token_names": list(self.token_names),
            "token_type_ids": list(self.token_type_ids),
            "scale_ids": list(self.scale_ids),
            "slot_ids": list(self.slot_ids),
            "family_slices": {key: list(value) for key, value in self.family_slices().items()},
            "k_state": int(self.k_state),
            "d_phi": int(self.d_phi),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalJetStateLayout":
        config = CanonicalJetStateConfig.from_dict(payload["config"])
        token_specs = tuple(CanonicalStateTokenSpec.from_dict(item) for item in payload["token_specs"])
        return cls(config=config, token_specs=token_specs)


def _token_spec(
    *,
    index: int,
    name: str,
    family: str,
    slot_id: int,
    ring_id: int = -1,
    sector_id: int = -1,
    radius_inner: float | None = None,
    radius_outer: float | None = None,
    angular_center: float | None = None,
    angular_width: float | None = None,
    anchor_radius: float | None = None,
) -> CanonicalStateTokenSpec:
    return CanonicalStateTokenSpec(
        index=int(index),
        name=name,
        family=family,
        token_type_id=CANONICAL_STATE_TOKEN_TYPE_IDS[family],
        scale_id=CANONICAL_STATE_SCALE_IDS[family],
        slot_id=int(slot_id),
        ring_id=int(ring_id),
        sector_id=int(sector_id),
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        angular_center=angular_center,
        angular_width=angular_width,
        anchor_radius=anchor_radius,
    )


def build_token_specs(config: CanonicalJetStateConfig | None = None) -> tuple[CanonicalStateTokenSpec, ...]:
    config = CanonicalJetStateConfig() if config is None else config
    specs: list[CanonicalStateTokenSpec] = []

    for slot_id, name in enumerate(config.global_tokens):
        specs.append(_token_spec(index=len(specs), name=name, family="global", slot_id=slot_id))

    for ring_id, (inner, outer) in enumerate(zip(config.radial_bin_edges[:-1], config.radial_bin_edges[1:])):
        name = f"radial_ring_{ring_id:02d}"
        specs.append(
            _token_spec(
                index=len(specs),
                name=name,
                family="radial",
                slot_id=ring_id,
                ring_id=ring_id,
                radius_inner=float(inner),
                radius_outer=None if outer is None else float(outer),
            )
        )

    angular_width = 2.0 * math.pi / float(config.angular_sectors)
    for sector_id in range(config.angular_sectors):
        center = -math.pi + (float(sector_id) + 0.5) * angular_width
        specs.append(
            _token_spec(
                index=len(specs),
                name=f"angular_sector_{sector_id:02d}",
                family="angular",
                slot_id=sector_id,
                sector_id=sector_id,
                angular_center=center,
                angular_width=angular_width,
            )
        )

    for family in ("anchor_coarse", "anchor_medium", "anchor_fine"):
        for slot_id in range(int(config.anchor_counts[family])):
            specs.append(
                _token_spec(
                    index=len(specs),
                    name=f"{family}_slot_{slot_id:02d}",
                    family=family,
                    slot_id=slot_id,
                    anchor_radius=float(config.anchor_radii[family]),
                )
            )

    return tuple(specs)


def default_canonical_jet_state_config() -> CanonicalJetStateConfig:
    return CanonicalJetStateConfig()


def default_canonical_jet_state_layout() -> CanonicalJetStateLayout:
    return CanonicalJetStateLayout(default_canonical_jet_state_config())


def canonical_jet_state_layout_manifest() -> dict[str, Any]:
    return default_canonical_jet_state_layout().to_dict()

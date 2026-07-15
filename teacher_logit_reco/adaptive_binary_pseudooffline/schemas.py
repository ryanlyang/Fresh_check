"""Versioned tensor schemas for adaptive binary pseudo-offline reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping


ABPH_SCHEMA_CONTRACT = "adaptive_binary_pseudooffline_schema_v1"
ABPH_SCHEMA_VERSION = "v1"
ABPH_MAX_PARTICLES = 128
ABPH_PID_CATEGORIES: tuple[str, ...] = (
    "charged_hadron",
    "neutral_hadron",
    "photon",
    "electron",
    "muon",
    "other",
)
ABPH_EFFECTIVE_MASS_GEV: Mapping[str, float] = MappingProxyType(
    {
        "charged_hadron": 0.13957039,
        "neutral_hadron": 0.0,
        "photon": 0.0,
        "electron": 0.00051099895,
        "muon": 0.1056583755,
        "other": 0.0,
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(row) for key, row in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(row) for row in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(row) for key, row in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(row) for row in value]
    return value


@dataclass(frozen=True)
class SchemaField:
    """One ordered field in a persisted tensor schema."""

    name: str
    dtype: str
    shape: tuple[int, ...] = ()
    unit: str = "unitless"
    role: str = "feature"
    required: bool = True
    hard_constraint: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("schema field name must be non-empty")
        dtype = str(self.dtype).strip()
        if not dtype:
            raise ValueError(f"schema field {name!r} dtype must be non-empty")
        shape = tuple(int(size) for size in self.shape)
        if any(size <= 0 for size in shape):
            raise ValueError(f"schema field {name!r} shape must contain positive dimensions")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "unit", str(self.unit))
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "hard_constraint", bool(self.hard_constraint))
        object.__setattr__(self, "description", str(self.description))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "unit": self.unit,
            "role": self.role,
            "required": self.required,
            "hard_constraint": self.hard_constraint,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchemaField":
        return cls(
            name=str(payload["name"]),
            dtype=str(payload["dtype"]),
            shape=tuple(int(size) for size in payload.get("shape", ())),
            unit=str(payload.get("unit", "unitless")),
            role=str(payload.get("role", "feature")),
            required=bool(payload.get("required", True)),
            hard_constraint=bool(payload.get("hard_constraint", False)),
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True)
class VersionedTensorSchema:
    """Ordered, hash-bound schema used by caches, checkpoints, and reports."""

    name: str
    version: str
    fields: tuple[SchemaField, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    contract: str = ABPH_SCHEMA_CONTRACT

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        version = str(self.version).strip()
        fields = tuple(self.fields)
        if not name or not version:
            raise ValueError("schema name and version must be non-empty")
        if not fields:
            raise ValueError(f"schema {name!r} must contain at least one field")
        field_names = tuple(row.name for row in fields)
        if len(field_names) != len(set(field_names)):
            raise ValueError(f"schema {name!r} contains duplicate field names")
        metadata = _thaw_json(_freeze_json(dict(self.metadata)))
        _canonical_json(metadata)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "metadata", _freeze_json(metadata))
        object.__setattr__(self, "contract", str(self.contract))

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.fields)

    @property
    def feature_order_hash(self) -> str:
        # Deliberately covers every schema attribute. Any semantic schema drift
        # must invalidate cached features, not only a reordered field name.
        return _sha256_json(self._hash_payload())

    @property
    def schema_hash(self) -> str:
        return self.feature_order_hash

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "name": self.name,
            "version": self.version,
            "fields": [row.to_dict() for row in self.fields],
            "metadata": _thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["feature_order_hash"] = self.feature_order_hash
        payload["schema_hash"] = self.schema_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VersionedTensorSchema":
        schema = cls(
            name=str(payload["name"]),
            version=str(payload["version"]),
            fields=tuple(SchemaField.from_dict(row) for row in payload["fields"]),
            metadata=dict(payload.get("metadata", {})),
            contract=str(payload.get("contract", ABPH_SCHEMA_CONTRACT)),
        )
        for key in ("feature_order_hash", "schema_hash"):
            expected = payload.get(key)
            if expected not in (None, schema.schema_hash):
                raise ValueError(
                    f"serialized {schema.name} {key} mismatch: {expected} != {schema.schema_hash}"
                )
        return schema


def _field(
    name: str,
    dtype: str = "float32",
    *,
    shape: tuple[int, ...] = (),
    unit: str = "unitless",
    role: str,
    hard: bool = False,
    description: str = "",
) -> SchemaField:
    return SchemaField(
        name=name,
        dtype=dtype,
        shape=shape,
        unit=unit,
        role=role,
        hard_constraint=hard,
        description=description,
    )


def _root_fields() -> tuple[SchemaField, ...]:
    hard_fields = (
        _field("energy", unit="GeV", role="hard_four_momentum", hard=True),
        _field("px", unit="GeV", role="hard_four_momentum", hard=True),
        _field("py", unit="GeV", role="hard_four_momentum", hard=True),
        _field("pz", unit="GeV", role="hard_four_momentum", hard=True),
        _field("constituent_count", "int32", role="hard_discrete", hard=True),
        *tuple(
            _field(f"count_{pid}", "int32", role="hard_type_count", hard=True)
            for pid in ABPH_PID_CATEGORIES
        ),
        _field("integer_charge", "int32", unit="e", role="hard_discrete", hard=True),
        _field("minimum_mass_budget", unit="GeV", role="feasibility", hard=True),
        _field("feasible_charge_min", "int32", unit="e", role="feasibility", hard=True),
        _field("feasible_charge_max", "int32", unit="e", role="feasibility", hard=True),
    )
    auxiliary_fields = (
        _field("scalar_sum_pt", unit="GeV", role="auxiliary_accounting"),
        *tuple(
            _field(f"energy_{pid}", unit="GeV", role="auxiliary_type_energy")
            for pid in ABPH_PID_CATEGORIES
        ),
        *tuple(
            _field(f"scalar_pt_{pid}", unit="GeV", role="auxiliary_type_pt")
            for pid in ABPH_PID_CATEGORIES
        ),
        _field("absolute_charge_sum", unit="e", role="auxiliary_accounting"),
        _field("eta_first_moment", role="auxiliary_moment"),
        _field("phi_first_moment", role="auxiliary_moment"),
        _field("eta_second_moment", role="auxiliary_moment"),
        _field("phi_second_moment", role="auxiliary_moment"),
        _field("eta_phi_cross_moment", role="auxiliary_moment"),
        _field("radial_first_moment", role="auxiliary_moment"),
        _field("radial_second_moment", role="auxiliary_moment"),
        _field("covariance_cholesky", shape=(3,), role="auxiliary_shape"),
        _field("radial_quantiles", shape=(3,), role="auxiliary_shape"),
        _field("leading_pt_fractions", shape=(4,), role="auxiliary_shape"),
    )
    return hard_fields + auxiliary_fields


ROOT_LEDGER_SCHEMA = VersionedTensorSchema(
    name="abph_root_ledger",
    version=ABPH_SCHEMA_VERSION,
    fields=_root_fields(),
    metadata={
        "coordinate_frame": "hlt_jet_axis_centered",
        "phi_wrapping": "[-pi,pi)",
        "pid_categories": list(ABPH_PID_CATEGORIES),
        "compiler_order": [
            "count",
            "type_counts",
            "charge",
            "minimum_mass",
            "four_momentum",
            "auxiliary_observables",
        ],
    },
)


GROUP_TARGET_SCHEMA = VersionedTensorSchema(
    name="abph_binary_group_target",
    version=ABPH_SCHEMA_VERSION,
    fields=(
        *_root_fields(),
        _field("parent_index", "int32", role="topology"),
        _field("depth", "int16", role="topology"),
        _field("topology_state", "int8", role="topology"),
        _field("centroid_eta", role="support_geometry"),
        _field("centroid_phi", role="support_geometry"),
        _field("support_covariance_cholesky", shape=(3,), role="support_geometry"),
        _field("support_radial_quantiles", shape=(3,), role="support_geometry"),
        _field("maximum_member_radius", role="support_geometry"),
        _field("principal_axis_sin", role="support_geometry"),
        _field("principal_axis_cos", role="support_geometry"),
        _field("member_count", "int32", role="membership"),
        _field(
            "member_indices",
            "int16",
            shape=(ABPH_MAX_PARTICLES,),
            role="membership",
        ),
        _field(
            "member_mask",
            "bool",
            shape=(ABPH_MAX_PARTICLES,),
            role="membership",
        ),
    ),
    metadata={
        "topology_states": ["padding", "active_and_terminal", "active_and_split"],
        "level_capacities": [2, 4, 8, 16, 32],
        "sibling_order": "unordered",
    },
)


PARTICLE_TARGET_SCHEMA = VersionedTensorSchema(
    name="abph_particle_target",
    version=ABPH_SCHEMA_VERSION,
    fields=(
        _field("pt", unit="GeV", role="particle_four_vector"),
        _field("eta_hlt_relative", role="particle_four_vector"),
        _field("phi_hlt_relative", role="particle_four_vector"),
        _field("energy", unit="GeV", role="particle_four_vector"),
        _field("charge", unit="e", role="particle_identity"),
        *tuple(
            _field(f"pid_{pid}", role="particle_identity")
            for pid in ABPH_PID_CATEGORIES
        ),
        _field("d0_value", role="track_feature"),
        _field("d0_error", role="track_feature"),
        _field("dz_value", role="track_feature"),
        _field("dz_error", role="track_feature"),
        _field("source_constituent_index", "int16", role="identity"),
        _field("target_microgroup_index", "int16", role="membership"),
    ),
    metadata={
        "coordinate_origin": "hlt_jet_axis",
        "offline_axis_rotation": False,
        "max_particles": ABPH_MAX_PARTICLES,
        "pid_categories": list(ABPH_PID_CATEGORIES),
    },
)


ABPH_SCHEMAS: Mapping[str, VersionedTensorSchema] = MappingProxyType(
    {
        ROOT_LEDGER_SCHEMA.name: ROOT_LEDGER_SCHEMA,
        GROUP_TARGET_SCHEMA.name: GROUP_TARGET_SCHEMA,
        PARTICLE_TARGET_SCHEMA.name: PARTICLE_TARGET_SCHEMA,
    }
)


def schema_manifest(schemas: Iterable[VersionedTensorSchema] | None = None) -> dict[str, Any]:
    rows = tuple(ABPH_SCHEMAS.values()) if schemas is None else tuple(schemas)
    payload = {row.name: row.to_dict() for row in rows}
    return {
        "contract": ABPH_SCHEMA_CONTRACT,
        "schema_version": ABPH_SCHEMA_VERSION,
        "schemas": payload,
        "manifest_hash": _sha256_json(payload),
    }


def schema_from_dict(payload: Mapping[str, Any]) -> VersionedTensorSchema:
    """Round-trip helper that verifies any serialized schema hash."""

    return VersionedTensorSchema.from_dict(payload)


__all__ = [
    "ABPH_MAX_PARTICLES",
    "ABPH_PID_CATEGORIES",
    "ABPH_SCHEMA_CONTRACT",
    "ABPH_SCHEMA_VERSION",
    "ABPH_SCHEMAS",
    "GROUP_TARGET_SCHEMA",
    "PARTICLE_TARGET_SCHEMA",
    "ROOT_LEDGER_SCHEMA",
    "SchemaField",
    "VersionedTensorSchema",
    "schema_from_dict",
    "schema_manifest",
]

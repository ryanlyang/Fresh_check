"""Deterministic adaptive-binary hierarchy targets for pseudo-offline training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import math
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM

from .schemas import (
    ABPH_MAX_PARTICLES,
    ABPH_EFFECTIVE_MASS_GEV,
    ABPH_PID_CATEGORIES,
    GROUP_TARGET_SCHEMA,
    PARTICLE_TARGET_SCHEMA,
    ROOT_LEDGER_SCHEMA,
    SchemaField,
    VersionedTensorSchema,
)


ABPH_TARGET_BUILDER_CONTRACT = "adaptive_binary_pseudooffline_target_builder_v1"
ABPH_TARGET_BUILDER_VERSION = "v1"
ABPH_HIERARCHY_GROUPINGS: tuple[str, ...] = ("exclusive_kt", "cambridge_aachen")
ABPH_LEVEL_CAPACITIES: tuple[int, ...] = (2, 4, 8, 16, 32)
ABPH_CLUSTER_RADIUS = 0.8

TOPOLOGY_PADDING = np.int8(0)
TOPOLOGY_ACTIVE_TERMINAL = np.int8(1)
TOPOLOGY_ACTIVE_SPLIT = np.int8(2)

_PID_MASSES_GEV = ABPH_EFFECTIVE_MASS_GEV
_GROUP_SEPARATE_FIELDS = frozenset(("member_indices", "member_mask"))


def wrap_phi(values: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to the single canonical interval ``[-pi, pi)``."""

    array = np.asarray(values)
    wrapped = (array + np.pi) % (2.0 * np.pi) - np.pi
    if np.isscalar(values):
        return float(wrapped)
    return wrapped


def _field_size(field: SchemaField) -> int:
    return int(np.prod(field.shape, dtype=np.int64)) if field.shape else 1


def _flat_field_names(
    schema: VersionedTensorSchema,
    *,
    excluded: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    names: list[str] = []
    for field in schema.fields:
        if field.name in excluded:
            continue
        size = _field_size(field)
        if size == 1:
            names.append(field.name)
        else:
            names.extend(f"{field.name}[{index}]" for index in range(size))
    return tuple(names)


ROOT_FEATURE_NAMES = _flat_field_names(ROOT_LEDGER_SCHEMA)
GROUP_FEATURE_NAMES = _flat_field_names(GROUP_TARGET_SCHEMA, excluded=_GROUP_SEPARATE_FIELDS)
PARTICLE_TARGET_NAMES = _flat_field_names(PARTICLE_TARGET_SCHEMA)


def _flatten_mapping(
    schema: VersionedTensorSchema,
    values: Mapping[str, Any],
    *,
    excluded: frozenset[str] = frozenset(),
) -> np.ndarray:
    result: list[np.ndarray] = []
    for field in schema.fields:
        if field.name in excluded:
            continue
        if field.name not in values:
            raise KeyError(f"missing {schema.name} field {field.name!r}")
        value = np.asarray(values[field.name])
        expected_shape = field.shape
        if expected_shape and tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{schema.name}.{field.name} shape {value.shape} != {expected_shape}"
            )
        if not expected_shape and value.ndim != 0:
            raise ValueError(f"{schema.name}.{field.name} must be scalar, got {value.shape}")
        result.append(value.reshape(-1).astype(np.float64, copy=False))
    return np.concatenate(result).astype(np.float32, copy=False)


def _normalize_grouping(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace("/", "_")
    aliases = {
        "kt": "exclusive_kt",
        "exclusivekt": "exclusive_kt",
        "exclusive_k_t": "exclusive_kt",
        "ca": "cambridge_aachen",
        "c_a": "cambridge_aachen",
        "cambridgeaachen": "cambridge_aachen",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ABPH_HIERARCHY_GROUPINGS:
        raise ValueError(
            f"unknown grouping {value!r}; expected one of {ABPH_HIERARCHY_GROUPINGS}"
        )
    return normalized


@dataclass(frozen=True)
class AdaptiveBinaryHierarchyLayout:
    """Fixed hierarchy layout and generalized-kT clustering semantics."""

    grouping: str = "exclusive_kt"
    radius: float = ABPH_CLUSTER_RADIUS
    level_capacities: tuple[int, ...] = ABPH_LEVEL_CAPACITIES

    def __post_init__(self) -> None:
        grouping = _normalize_grouping(self.grouping)
        radius = float(self.radius)
        capacities = tuple(int(value) for value in self.level_capacities)
        if radius <= 0.0:
            raise ValueError("cluster radius must be positive")
        if capacities != ABPH_LEVEL_CAPACITIES:
            raise ValueError(f"level capacities must be exactly {ABPH_LEVEL_CAPACITIES}")
        object.__setattr__(self, "grouping", grouping)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "level_capacities", capacities)

    @property
    def exponent(self) -> float:
        return 1.0 if self.grouping == "exclusive_kt" else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": ABPH_TARGET_BUILDER_CONTRACT,
            "builder_version": ABPH_TARGET_BUILDER_VERSION,
            "grouping": self.grouping,
            "generalized_kt_exponent": self.exponent,
            "radius": self.radius,
            "recombination": "E_scheme",
            "coordinate_frame": "hlt_jet_axis_centered_eta_phi",
            "phi_wrapping": "[-pi,pi)",
            "level_capacities": list(self.level_capacities),
            "parent_reclustering": "independent_recursive_exclusive_two_subjets",
            "terminal_policy": "carry_singleton_without_empty_sibling",
        }


@dataclass(frozen=True)
class AdaptiveBinaryTargetBatch:
    """One fixed-shape batch of root, frontier, and particle targets."""

    root_features: np.ndarray
    root_identities: np.ndarray
    level_features: tuple[np.ndarray, ...]
    level_masks: tuple[np.ndarray, ...]
    level_topology: tuple[np.ndarray, ...]
    level_parent_indices: tuple[np.ndarray, ...]
    level_membership: tuple[np.ndarray, ...]
    level_identities: tuple[np.ndarray, ...]
    particle_targets: np.ndarray
    particle_mask: np.ndarray
    hlt_axis_eta: np.ndarray
    hlt_axis_phi: np.ndarray
    valid_hlt_counts: np.ndarray
    valid_offline_counts: np.ndarray
    layout: AdaptiveBinaryHierarchyLayout
    diagnostics: Mapping[str, Any]

    @property
    def n_jets(self) -> int:
        return int(self.root_features.shape[0])

    def array_dict(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "root_features": np.asarray(self.root_features),
            "root_identities": np.asarray(self.root_identities),
            "particle_targets": np.asarray(self.particle_targets),
            "particle_mask": np.asarray(self.particle_mask),
            "hlt_axis_eta": np.asarray(self.hlt_axis_eta),
            "hlt_axis_phi": np.asarray(self.hlt_axis_phi),
            "valid_hlt_counts": np.asarray(self.valid_hlt_counts),
            "valid_offline_counts": np.asarray(self.valid_offline_counts),
        }
        for depth in range(len(self.level_features)):
            prefix = f"level{depth + 1}"
            arrays[f"{prefix}_features"] = np.asarray(self.level_features[depth])
            arrays[f"{prefix}_mask"] = np.asarray(self.level_masks[depth])
            arrays[f"{prefix}_topology"] = np.asarray(self.level_topology[depth])
            arrays[f"{prefix}_parent_indices"] = np.asarray(
                self.level_parent_indices[depth]
            )
            arrays[f"{prefix}_membership"] = np.asarray(self.level_membership[depth])
            arrays[f"{prefix}_identities"] = np.asarray(self.level_identities[depth])
        return arrays


@dataclass
class _Cluster:
    uid: int
    members: tuple[int, ...]
    energy: float
    px: float
    py: float
    pz: float

    @property
    def pt(self) -> float:
        return math.hypot(self.px, self.py)

    @property
    def eta(self) -> float:
        return math.asinh(self.pz / max(self.pt, 1.0e-12))

    @property
    def phi(self) -> float:
        return math.atan2(self.py, self.px)


def _jet_identity_key(identity: JetIdentity | str | tuple[Any, ...]) -> str:
    if isinstance(identity, JetIdentity):
        return f"{identity.file}\0{int(identity.entry)}"
    if isinstance(identity, tuple):
        if len(identity) < 2:
            raise ValueError("tuple jet identity must contain at least file and entry")
        # A common identity tuple also contains the class label in position 2.
        # Deliberately exclude it from all clustering and tie-breaking semantics.
        return f"{identity[0]}\0{identity[1]}"
    return str(identity)


def _digest(*parts: Any) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, (tuple, list)):
            encoded = ",".join(str(value) for value in part)
        else:
            encoded = str(part)
        hasher.update(encoded.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _cluster_from_particle(tokens: np.ndarray, index: int) -> _Cluster:
    pt, eta, phi, energy = (float(value) for value in tokens[index, :4])
    return _Cluster(
        uid=int(index),
        members=(int(index),),
        energy=energy,
        px=pt * math.cos(phi),
        py=pt * math.sin(phi),
        pz=pt * math.sinh(eta),
    )


def _pair_distance(left: _Cluster, right: _Cluster, *, exponent: float, radius: float) -> float:
    deta = left.eta - right.eta
    dphi = float(wrap_phi(left.phi - right.phi))
    angular = (deta * deta + dphi * dphi) / (radius * radius)
    if exponent == 0.0:
        return angular
    return min(max(left.pt, 1.0e-12) ** (2.0 * exponent), max(right.pt, 1.0e-12) ** (2.0 * exponent)) * angular


def _pair_tie_key(
    left: _Cluster,
    right: _Cluster,
    *,
    grouping: str,
    jet_key: str,
) -> str:
    first, second = sorted((left.members, right.members))
    return _digest(ABPH_TARGET_BUILDER_VERSION, grouping, jet_key, "pair", first, second)


def exclusive_binary_partition(
    tokens: np.ndarray,
    member_indices: Sequence[int],
    *,
    grouping: str,
    jet_identity: JetIdentity | str | tuple[Any, ...],
    radius: float = ABPH_CLUSTER_RADIUS,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Recluster one parent and return exactly two nonempty E-scheme subjets."""

    grouping = _normalize_grouping(grouping)
    members = tuple(sorted(int(index) for index in member_indices))
    if len(members) < 2:
        raise ValueError("exclusive binary partition requires at least two particles")
    array = np.asarray(tokens)
    if array.ndim != 2 or array.shape[1] != RAW_TOKEN_DIM:
        raise ValueError(f"tokens must have shape [particles, {RAW_TOKEN_DIM}]")
    if min(members) < 0 or max(members) >= array.shape[0]:
        raise IndexError("member index lies outside the token array")

    exponent = 1.0 if grouping == "exclusive_kt" else 0.0
    jet_key = _jet_identity_key(jet_identity)
    active: dict[int, _Cluster] = {
        index: _cluster_from_particle(array, index) for index in members
    }
    heap: list[tuple[float, str, int, int]] = []

    def push_pair(left: _Cluster, right: _Cluster) -> None:
        first_uid, second_uid = sorted((left.uid, right.uid))
        heapq.heappush(
            heap,
            (
                _pair_distance(left, right, exponent=exponent, radius=float(radius)),
                _pair_tie_key(left, right, grouping=grouping, jet_key=jet_key),
                first_uid,
                second_uid,
            ),
        )

    initial = list(active.values())
    for left_index, left in enumerate(initial):
        for right in initial[left_index + 1 :]:
            push_pair(left, right)

    next_uid = max(active) + 1
    while len(active) > 2:
        while heap:
            _, _, left_uid, right_uid = heapq.heappop(heap)
            if left_uid in active and right_uid in active:
                break
        else:
            raise RuntimeError("exclusive clustering heap exhausted before two clusters remained")
        left = active.pop(left_uid)
        right = active.pop(right_uid)
        merged = _Cluster(
            uid=next_uid,
            members=tuple(sorted(left.members + right.members)),
            energy=left.energy + right.energy,
            px=left.px + right.px,
            py=left.py + right.py,
            pz=left.pz + right.pz,
        )
        next_uid += 1
        for other in active.values():
            push_pair(merged, other)
        active[merged.uid] = merged

    children = sorted(
        (cluster.members for cluster in active.values()),
        key=lambda child: (_digest(ABPH_TARGET_BUILDER_VERSION, grouping, jet_key, "child", child), child),
    )
    if len(children) != 2 or not children[0] or not children[1]:
        raise RuntimeError("exclusive clustering did not produce two nonempty children")
    if set(children[0]).intersection(children[1]) or set(children[0]).union(children[1]) != set(members):
        raise RuntimeError("exclusive clustering children do not partition their parent")
    return children[0], children[1]


def _pid_indices(tokens: np.ndarray, members: np.ndarray) -> np.ndarray:
    pid = np.asarray(tokens[members, 5:10], dtype=np.float64)
    best = np.argmax(pid, axis=1)
    known = np.max(pid, axis=1) > 0.5
    return np.where(known, best, len(ABPH_PID_CATEGORIES) - 1).astype(np.int64)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    if values.size == 0:
        return np.zeros(len(tuple(quantiles)), dtype=np.float64)
    order = np.lexsort((np.arange(values.size), values))
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order], 0.0)
    total = float(sorted_weights.sum())
    if total <= 0.0:
        return np.quantile(sorted_values, quantiles)
    cumulative = np.cumsum(sorted_weights) / total
    return np.asarray(
        [sorted_values[min(int(np.searchsorted(cumulative, q, side="left")), values.size - 1)] for q in quantiles],
        dtype=np.float64,
    )


def _covariance_cholesky(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=np.float64)
    first = math.sqrt(max(float(cov[0, 0]), 0.0))
    lower = float(cov[1, 0]) / first if first > 1.0e-12 else 0.0
    second = math.sqrt(max(float(cov[1, 1]) - lower * lower, 0.0))
    return np.asarray((first, lower, second), dtype=np.float64)


def _hlt_axis(tokens: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    members = np.flatnonzero(mask)
    if members.size == 0:
        return 0.0, 0.0
    pt = np.maximum(np.asarray(tokens[members, 0], dtype=np.float64), 0.0)
    total = float(pt.sum())
    weights = pt / total if total > 0.0 else np.full(members.size, 1.0 / members.size)
    eta = float(np.sum(weights * np.asarray(tokens[members, 1], dtype=np.float64)))
    phi_values = np.asarray(tokens[members, 2], dtype=np.float64)
    phi = math.atan2(float(np.sum(weights * np.sin(phi_values))), float(np.sum(weights * np.cos(phi_values))))
    return eta, float(wrap_phi(phi))


def _accounting_values(
    tokens: np.ndarray,
    members: np.ndarray,
    *,
    axis_eta: float,
    axis_phi: float,
) -> dict[str, Any]:
    selected = np.asarray(tokens[members], dtype=np.float64)
    pt = np.maximum(selected[:, 0], 0.0)
    eta = selected[:, 1]
    phi = selected[:, 2]
    energy = selected[:, 3]
    charge = np.rint(selected[:, 4]).astype(np.int64)
    pid_indices = _pid_indices(tokens, members)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    total_pt = float(pt.sum())
    weights = pt / total_pt if total_pt > 0.0 else np.full(members.size, 1.0 / members.size)
    eta_rel = eta - float(axis_eta)
    phi_rel = np.asarray(wrap_phi(phi - float(axis_phi)), dtype=np.float64)
    eta_mean = float(np.sum(weights * eta_rel))
    phi_mean = float(np.sum(weights * phi_rel))
    eta_second = float(np.sum(weights * eta_rel * eta_rel))
    phi_second = float(np.sum(weights * phi_rel * phi_rel))
    cross = float(np.sum(weights * eta_rel * phi_rel))
    centered = np.stack((eta_rel - eta_mean, phi_rel - phi_mean), axis=1)
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
    radius = np.sqrt(eta_rel * eta_rel + phi_rel * phi_rel)
    counts = np.bincount(pid_indices, minlength=len(ABPH_PID_CATEGORIES)).astype(np.int64)
    type_energy = np.bincount(pid_indices, weights=energy, minlength=len(ABPH_PID_CATEGORIES))
    type_pt = np.bincount(pid_indices, weights=pt, minlength=len(ABPH_PID_CATEGORIES))
    charged_slots = int(counts[0] + counts[3] + counts[4] + counts[5])
    leading = np.zeros(4, dtype=np.float64)
    if total_pt > 0.0:
        sorted_pt = np.sort(pt)[::-1]
        leading[: min(4, sorted_pt.size)] = sorted_pt[:4] / total_pt
    values: dict[str, Any] = {
        "energy": float(energy.sum()),
        "px": float(px.sum()),
        "py": float(py.sum()),
        "pz": float(pz.sum()),
        "constituent_count": int(members.size),
        "integer_charge": int(charge.sum()),
        "minimum_mass_budget": float(
            sum(float(counts[index]) * _PID_MASSES_GEV[name] for index, name in enumerate(ABPH_PID_CATEGORIES))
        ),
        "feasible_charge_min": -charged_slots,
        "feasible_charge_max": charged_slots,
        "scalar_sum_pt": total_pt,
        "absolute_charge_sum": float(np.abs(charge).sum()),
        "eta_first_moment": eta_mean,
        "phi_first_moment": phi_mean,
        "eta_second_moment": eta_second,
        "phi_second_moment": phi_second,
        "eta_phi_cross_moment": cross,
        "radial_first_moment": float(np.sum(weights * radius)),
        "radial_second_moment": float(np.sum(weights * radius * radius)),
        "covariance_cholesky": _covariance_cholesky(covariance),
        "radial_quantiles": _weighted_quantiles(radius, pt, (0.5, 0.8, 0.95)),
        "leading_pt_fractions": leading,
    }
    for index, name in enumerate(ABPH_PID_CATEGORIES):
        values[f"count_{name}"] = int(counts[index])
        values[f"energy_{name}"] = float(type_energy[index])
        values[f"scalar_pt_{name}"] = float(type_pt[index])
    return values


def _group_values(
    tokens: np.ndarray,
    members: np.ndarray,
    *,
    axis_eta: float,
    axis_phi: float,
    parent_index: int,
    depth: int,
    topology_state: int,
) -> dict[str, Any]:
    values = _accounting_values(tokens, members, axis_eta=axis_eta, axis_phi=axis_phi)
    selected = np.asarray(tokens[members], dtype=np.float64)
    pt = np.maximum(selected[:, 0], 0.0)
    total_pt = float(pt.sum())
    weights = pt / total_pt if total_pt > 0.0 else np.full(members.size, 1.0 / members.size)
    centroid_eta_abs = float(np.sum(weights * selected[:, 1]))
    centroid_phi_abs = math.atan2(
        float(np.sum(weights * np.sin(selected[:, 2]))),
        float(np.sum(weights * np.cos(selected[:, 2]))),
    )
    deta = selected[:, 1] - centroid_eta_abs
    dphi = np.asarray(wrap_phi(selected[:, 2] - centroid_phi_abs), dtype=np.float64)
    centered = np.stack((deta, dphi), axis=1)
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
    radii = np.sqrt(deta * deta + dphi * dphi)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if axis[0] < 0.0 or (abs(float(axis[0])) <= 1.0e-15 and axis[1] < 0.0):
        axis = -axis
    member_indices = np.full(ABPH_MAX_PARTICLES, -1, dtype=np.int16)
    member_mask = np.zeros(ABPH_MAX_PARTICLES, dtype=bool)
    member_indices[: members.size] = members.astype(np.int16)
    member_mask[: members.size] = True
    values.update(
        {
            "parent_index": int(parent_index),
            "depth": int(depth),
            "topology_state": int(topology_state),
            "centroid_eta": centroid_eta_abs - float(axis_eta),
            "centroid_phi": float(wrap_phi(centroid_phi_abs - float(axis_phi))),
            "support_covariance_cholesky": _covariance_cholesky(covariance),
            "support_radial_quantiles": _weighted_quantiles(radii, pt, (0.5, 0.8, 0.95)),
            "maximum_member_radius": float(np.max(radii, initial=0.0)),
            "principal_axis_sin": float(axis[1]),
            "principal_axis_cos": float(axis[0]),
            "member_count": int(members.size),
            "member_indices": member_indices,
            "member_mask": member_mask,
        }
    )
    return values


def _particle_values(
    token: np.ndarray,
    *,
    source_index: int,
    microgroup_index: int,
    axis_eta: float,
    axis_phi: float,
) -> dict[str, Any]:
    pid = np.asarray(token[5:10], dtype=np.float64)
    pid_values = np.zeros(len(ABPH_PID_CATEGORIES), dtype=np.float64)
    if pid.size and float(np.max(pid)) > 0.5:
        pid_values[int(np.argmax(pid))] = 1.0
    else:
        pid_values[-1] = 1.0
    values: dict[str, Any] = {
        "pt": float(token[0]),
        "eta_hlt_relative": float(token[1]) - float(axis_eta),
        "phi_hlt_relative": float(wrap_phi(float(token[2]) - float(axis_phi))),
        "energy": float(token[3]),
        "charge": float(token[4]),
        "d0_value": float(token[10]),
        "d0_error": float(token[11]),
        "dz_value": float(token[12]),
        "dz_error": float(token[13]),
        "source_constituent_index": int(source_index),
        "target_microgroup_index": int(microgroup_index),
    }
    for index, name in enumerate(ABPH_PID_CATEGORIES):
        values[f"pid_{name}"] = float(pid_values[index])
    return values


def _target_identity(
    *,
    grouping: str,
    jet_key: str,
    depth: int,
    members: Sequence[int],
) -> str:
    return _digest(
        ABPH_TARGET_BUILDER_CONTRACT,
        ABPH_TARGET_BUILDER_VERSION,
        grouping,
        jet_key,
        int(depth),
        tuple(sorted(int(value) for value in members)),
    )


def build_adaptive_binary_targets(
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
    offline_tokens: np.ndarray,
    offline_mask: np.ndarray,
    *,
    jet_ids: Sequence[JetIdentity | str | tuple[Any, ...]],
    layout: AdaptiveBinaryHierarchyLayout | None = None,
) -> AdaptiveBinaryTargetBatch:
    """Build recursive binary targets for one aligned HLT/offline batch."""

    resolved_layout = layout or AdaptiveBinaryHierarchyLayout()
    hlt = np.asarray(hlt_tokens, dtype=np.float32)
    hlt_valid = np.asarray(hlt_mask, dtype=bool)
    offline = np.asarray(offline_tokens, dtype=np.float32)
    offline_valid = np.asarray(offline_mask, dtype=bool)
    if hlt.shape != offline.shape or hlt.shape[-2:] != (ABPH_MAX_PARTICLES, RAW_TOKEN_DIM):
        raise ValueError(
            f"HLT/offline tokens must share shape [N, {ABPH_MAX_PARTICLES}, {RAW_TOKEN_DIM}]"
        )
    if hlt_valid.shape != hlt.shape[:2] or offline_valid.shape != offline.shape[:2]:
        raise ValueError("HLT/offline masks must match the first two token dimensions")
    n_jets = int(hlt.shape[0])
    if len(jet_ids) != n_jets:
        raise ValueError(f"jet_ids length {len(jet_ids)} != n_jets {n_jets}")
    if not np.isfinite(hlt[hlt_valid]).all() or not np.isfinite(offline[offline_valid]).all():
        raise ValueError("valid HLT/offline target inputs must be finite")

    root_features = np.zeros((n_jets, len(ROOT_FEATURE_NAMES)), dtype=np.float32)
    root_identities = np.zeros(n_jets, dtype="S64")
    level_features = tuple(
        np.zeros((n_jets, capacity, len(GROUP_FEATURE_NAMES)), dtype=np.float32)
        for capacity in resolved_layout.level_capacities
    )
    level_masks = tuple(
        np.zeros((n_jets, capacity), dtype=bool) for capacity in resolved_layout.level_capacities
    )
    level_topology = tuple(
        np.zeros((n_jets, capacity), dtype=np.int8) for capacity in resolved_layout.level_capacities
    )
    level_parent_indices = tuple(
        np.full((n_jets, capacity), -1, dtype=np.int16) for capacity in resolved_layout.level_capacities
    )
    level_membership = tuple(
        np.zeros((n_jets, capacity, ABPH_MAX_PARTICLES), dtype=bool)
        for capacity in resolved_layout.level_capacities
    )
    level_identities = tuple(
        np.zeros((n_jets, capacity), dtype="S64") for capacity in resolved_layout.level_capacities
    )
    particle_targets = np.zeros(
        (n_jets, ABPH_MAX_PARTICLES, len(PARTICLE_TARGET_NAMES)), dtype=np.float32
    )
    particle_mask = offline_valid.copy()
    hlt_axis_eta = np.zeros(n_jets, dtype=np.float32)
    hlt_axis_phi = np.zeros(n_jets, dtype=np.float32)
    valid_hlt_counts = hlt_valid.sum(axis=1, dtype=np.int32)
    valid_offline_counts = offline_valid.sum(axis=1, dtype=np.int32)

    for jet_index, identity in enumerate(jet_ids):
        members = tuple(int(value) for value in np.flatnonzero(offline_valid[jet_index]))
        if not members:
            raise ValueError(f"jet {jet_index} has no valid offline constituents")
        axis_eta, axis_phi = _hlt_axis(hlt[jet_index], hlt_valid[jet_index])
        hlt_axis_eta[jet_index] = axis_eta
        hlt_axis_phi[jet_index] = axis_phi
        jet_key = _jet_identity_key(identity)
        root_values = _accounting_values(
            offline[jet_index], np.asarray(members, dtype=np.int64), axis_eta=axis_eta, axis_phi=axis_phi
        )
        root_features[jet_index] = _flatten_mapping(ROOT_LEDGER_SCHEMA, root_values)
        root_identities[jet_index] = _target_identity(
            grouping="shared_root", jet_key=jet_key, depth=0, members=members
        )

        current: list[tuple[tuple[int, ...], int]] = [(members, 0)]
        for depth_index, capacity in enumerate(resolved_layout.level_capacities):
            depth = depth_index + 1
            next_frontier: list[tuple[tuple[int, ...], int]] = []
            for parent_index, (parent_members, _) in enumerate(current):
                if len(parent_members) == 1:
                    next_frontier.append((parent_members, parent_index))
                else:
                    children = exclusive_binary_partition(
                        offline[jet_index],
                        parent_members,
                        grouping=resolved_layout.grouping,
                        jet_identity=identity,
                        radius=resolved_layout.radius,
                    )
                    next_frontier.extend((child, parent_index) for child in children)
            if len(next_frontier) > capacity:
                raise RuntimeError(
                    f"frontier depth {depth} produced {len(next_frontier)} groups above capacity {capacity}"
                )
            for group_index, (group_members_tuple, parent_index) in enumerate(next_frontier):
                group_members = np.asarray(group_members_tuple, dtype=np.int64)
                is_terminal = len(group_members_tuple) == 1 or depth_index == len(resolved_layout.level_capacities) - 1
                topology = TOPOLOGY_ACTIVE_TERMINAL if is_terminal else TOPOLOGY_ACTIVE_SPLIT
                values = _group_values(
                    offline[jet_index],
                    group_members,
                    axis_eta=axis_eta,
                    axis_phi=axis_phi,
                    parent_index=parent_index,
                    depth=depth,
                    topology_state=int(topology),
                )
                level_features[depth_index][jet_index, group_index] = _flatten_mapping(
                    GROUP_TARGET_SCHEMA, values, excluded=_GROUP_SEPARATE_FIELDS
                )
                level_masks[depth_index][jet_index, group_index] = True
                level_topology[depth_index][jet_index, group_index] = topology
                level_parent_indices[depth_index][jet_index, group_index] = parent_index
                level_membership[depth_index][jet_index, group_index, group_members] = True
                level_identities[depth_index][jet_index, group_index] = _target_identity(
                    grouping=resolved_layout.grouping,
                    jet_key=jet_key,
                    depth=depth,
                    members=group_members_tuple,
                )
            current = next_frontier

        microgroup_for_particle = np.full(ABPH_MAX_PARTICLES, -1, dtype=np.int16)
        final_membership = level_membership[-1][jet_index]
        for microgroup_index in np.flatnonzero(level_masks[-1][jet_index]):
            microgroup_for_particle[final_membership[microgroup_index]] = int(microgroup_index)
        for source_index in members:
            microgroup_index = int(microgroup_for_particle[source_index])
            if microgroup_index < 0:
                raise RuntimeError("valid target particle is absent from the final hierarchy frontier")
            particle_targets[jet_index, source_index] = _flatten_mapping(
                PARTICLE_TARGET_SCHEMA,
                _particle_values(
                    offline[jet_index, source_index],
                    source_index=source_index,
                    microgroup_index=microgroup_index,
                    axis_eta=axis_eta,
                    axis_phi=axis_phi,
                ),
            )

    output = AdaptiveBinaryTargetBatch(
        root_features=root_features,
        root_identities=root_identities,
        level_features=level_features,
        level_masks=level_masks,
        level_topology=level_topology,
        level_parent_indices=level_parent_indices,
        level_membership=level_membership,
        level_identities=level_identities,
        particle_targets=particle_targets,
        particle_mask=particle_mask,
        hlt_axis_eta=hlt_axis_eta,
        hlt_axis_phi=hlt_axis_phi,
        valid_hlt_counts=valid_hlt_counts,
        valid_offline_counts=valid_offline_counts,
        layout=resolved_layout,
        diagnostics={
            "empty_hlt_jets": int(np.sum(valid_hlt_counts == 0)),
            "all_finite": bool(np.isfinite(root_features).all() and np.isfinite(particle_targets).all()),
        },
    )
    require_adaptive_binary_target_invariants(output)
    return output


def adaptive_binary_target_invariant_report(output: AdaptiveBinaryTargetBatch) -> dict[str, Any]:
    """Audit exact frontier coverage, local partitions, masks, and identities."""

    problems: list[str] = []
    max_duplicate_count = 0
    for jet_index in range(output.n_jets):
        expected = output.particle_mask[jet_index]
        previous_membership = expected[None, :]
        previous_mask = np.asarray((True,), dtype=bool)
        root_state = (
            TOPOLOGY_ACTIVE_TERMINAL
            if int(expected.sum()) == 1
            else TOPOLOGY_ACTIVE_SPLIT
        )
        previous_topology = np.asarray((root_state,), dtype=np.int8)
        for depth_index, capacity in enumerate(output.layout.level_capacities):
            mask = output.level_masks[depth_index][jet_index]
            topology = output.level_topology[depth_index][jet_index]
            parents = output.level_parent_indices[depth_index][jet_index]
            membership = output.level_membership[depth_index][jet_index]
            identities = output.level_identities[depth_index][jet_index]
            if np.any(mask[1:] & ~mask[:-1]):
                problems.append(f"jet {jet_index} depth {depth_index + 1}: active groups are not packed")
            if np.any(membership[~mask]):
                problems.append(f"jet {jet_index} depth {depth_index + 1}: padding owns particles")
            if np.any(topology[~mask] != TOPOLOGY_PADDING):
                problems.append(f"jet {jet_index} depth {depth_index + 1}: padding topology is nonzero")
            if np.any(parents[~mask] != -1):
                problems.append(f"jet {jet_index} depth {depth_index + 1}: padding parent is not -1")
            if np.any(identities[mask] == b"") or np.any(identities[~mask] != b""):
                problems.append(f"jet {jet_index} depth {depth_index + 1}: identity mask mismatch")
            active_membership = membership[mask]
            counts = active_membership.sum(axis=1)
            if np.any(counts <= 0):
                problems.append(f"jet {jet_index} depth {depth_index + 1}: empty active child")
            coverage = active_membership.sum(axis=0)
            max_duplicate_count = max(max_duplicate_count, int(coverage.max(initial=0)))
            if not np.array_equal(coverage, expected.astype(np.int64)):
                problems.append(f"jet {jet_index} depth {depth_index + 1}: frontier is not an exact partition")
            if np.any(~np.isin(topology[mask], (TOPOLOGY_ACTIVE_TERMINAL, TOPOLOGY_ACTIVE_SPLIT))):
                problems.append(f"jet {jet_index} depth {depth_index + 1}: invalid active topology")

            for parent_index in np.flatnonzero(previous_mask):
                child_indices = np.flatnonzero(mask & (parents == parent_index))
                parent_members = previous_membership[parent_index]
                parent_state = previous_topology[parent_index]
                expected_children = 1 if parent_state == TOPOLOGY_ACTIVE_TERMINAL else 2
                if child_indices.size != expected_children:
                    problems.append(
                        f"jet {jet_index} depth {depth_index + 1}: parent {parent_index} has "
                        f"{child_indices.size} children, expected {expected_children}"
                    )
                    continue
                child_sum = membership[child_indices].sum(axis=0)
                if not np.array_equal(child_sum, parent_members.astype(np.int64)):
                    problems.append(
                        f"jet {jet_index} depth {depth_index + 1}: siblings do not partition parent {parent_index}"
                    )
                if parent_state == TOPOLOGY_ACTIVE_TERMINAL and not np.array_equal(
                    membership[child_indices[0]], parent_members
                ):
                    problems.append(
                        f"jet {jet_index} depth {depth_index + 1}: terminal parent was not carried exactly"
                    )
            previous_membership = membership
            previous_mask = mask
            previous_topology = topology

        if np.any(np.abs(output.hlt_axis_phi[jet_index]) > np.pi):
            problems.append(f"jet {jet_index}: HLT phi axis is not wrapped")
    return {
        "ok": not problems,
        "problems": problems,
        "n_jets": output.n_jets,
        "grouping": output.layout.grouping,
        "level_capacities": list(output.layout.level_capacities),
        "max_frontier_particle_multiplicity": max_duplicate_count,
        "target_identity_hash": hashlib.sha256(
            output.root_identities.tobytes()
            + b"".join(level.tobytes() for level in output.level_identities)
        ).hexdigest(),
    }


def require_adaptive_binary_target_invariants(output: AdaptiveBinaryTargetBatch) -> None:
    report = adaptive_binary_target_invariant_report(output)
    if not report["ok"]:
        raise ValueError("adaptive binary target invariant failure: " + "; ".join(report["problems"][:20]))


__all__ = [
    "ABPH_CLUSTER_RADIUS",
    "ABPH_HIERARCHY_GROUPINGS",
    "ABPH_LEVEL_CAPACITIES",
    "ABPH_TARGET_BUILDER_CONTRACT",
    "ABPH_TARGET_BUILDER_VERSION",
    "GROUP_FEATURE_NAMES",
    "PARTICLE_TARGET_NAMES",
    "ROOT_FEATURE_NAMES",
    "TOPOLOGY_ACTIVE_SPLIT",
    "TOPOLOGY_ACTIVE_TERMINAL",
    "TOPOLOGY_PADDING",
    "AdaptiveBinaryHierarchyLayout",
    "AdaptiveBinaryTargetBatch",
    "adaptive_binary_target_invariant_report",
    "build_adaptive_binary_targets",
    "exclusive_binary_partition",
    "require_adaptive_binary_target_invariants",
    "wrap_phi",
]

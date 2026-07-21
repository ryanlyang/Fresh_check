"""Virtual prediction-anchored residual bridges and immutable scalers.

This module intentionally contains no cache writer for dense fields.  A bridge
is a small hashed recipe which is evaluated on aligned ``f0`` and ``f_true``
batches in physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .bridge_contracts import canonical_sha256, with_content_hash
from .targets import DEFAULT_LOCAL_RESIDUAL_RADII, local_particle_residual_field_layout


PREDICTION_ANCHORED_BRIDGE_RECIPE_CONTRACT = "prediction_anchored_residual_bridge_recipe_v2"
PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT = "prediction_anchored_bridge_scalers_v1"
MATCHED_WRONG_EVENT_MAP_CONTRACT = "matched_wrong_event_map_v1"
BRIDGE_FORMULA_VERSION = "f0_plus_rho_times_ftrue_minus_f0_v1"
BRIDGE_STORAGE_POLICY = "virtual_fields_ram_only_v1"
BRIDGE_SCALER_FIT_VERSION = "physical_valid_particle_exact_channel_quantiles_v1"

BRIDGE_CHANNEL_PHYSICAL45 = "physical45"
BRIDGE_CHANNEL_ALL50 = "all50"
BRIDGE_CHANNEL_POLICIES = (BRIDGE_CHANNEL_PHYSICAL45, BRIDGE_CHANNEL_ALL50)
PHYSICAL_FIELD_COUNT = 45
FIELD_COUNT = 50
RESPONSE_RHOS = ("0.000", "0.025", "0.050", "0.075", "0.100")

CONTROL_EVENT_SHUFFLED = "event_shuffled_delta"
CONTROL_PARTICLE_SHUFFLED = "particle_shuffled_delta"
CONTROL_SIGN_REVERSED = "sign_reversed_delta"
CONTROL_SAME_NORM_RANDOM = "same_norm_random_delta"
CONTROL_RADIUS_PERMUTED = "radius_group_permuted_delta"
BRIDGE_CONTROLS = (
    CONTROL_EVENT_SHUFFLED,
    CONTROL_PARTICLE_SHUFFLED,
    CONTROL_SIGN_REVERSED,
    CONTROL_SAME_NORM_RANDOM,
    CONTROL_RADIUS_PERMUTED,
)


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    if len(text) != 64:
        return False
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def _decimal_rho(value: str | float | Decimal) -> str:
    try:
        rho = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid bridge rho {value!r}") from exc
    if not rho.is_finite() or rho < 0 or rho > 1:
        raise ValueError("bridge rho must be finite and lie in [0, 1]")
    quantized = rho.quantize(Decimal("0.001"))
    if quantized != rho:
        raise ValueError("bridge rho must be exactly representable to three decimal places")
    return format(quantized, "f")


def physical_loss_groups() -> dict[str, list[int]]:
    """Return the twelve radius-by-semantics groups from the plan."""

    semantics = {
        "pt_density": range(0, 5),
        "centroid": range(5, 8),
        "multiplicity": range(8, 10),
        "composition": range(10, 15),
    }
    output: dict[str, list[int]] = {}
    for radius_index, radius in enumerate(DEFAULT_LOCAL_RESIDUAL_RADII):
        for semantic, offsets in semantics.items():
            output[f"r{float(radius):.2f}.{semantic}"] = [
                15 * radius_index + int(offset) for offset in offsets
            ]
    return output


def build_bridge_recipe(
    *,
    rho: str | float | Decimal,
    channel_policy: str,
    r0_checkpoint_sha256: str,
    hlt_source_sha256: str,
    offline_source_sha256: str,
    split_manifest_sha256: str,
    target_schema_sha256: str,
    preprocessing_sha256: str,
    event_order_sha256: str,
    control_type: str | None = None,
    control_seed: int | None = None,
    audit_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical virtual recipe; no dense tensor is created."""

    policy = str(channel_policy)
    if policy not in BRIDGE_CHANNEL_POLICIES:
        raise ValueError(f"unknown bridge channel policy {channel_policy!r}")
    if control_type is not None and str(control_type) not in BRIDGE_CONTROLS:
        raise ValueError(f"unknown bridge control {control_type!r}")
    if control_type is not None and control_seed is None:
        raise ValueError("controlled bridge recipes require control_seed")
    hashes = {
        "r0_checkpoint_sha256": r0_checkpoint_sha256,
        "hlt_source_sha256": hlt_source_sha256,
        "offline_source_sha256": offline_source_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "target_schema_sha256": target_schema_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "event_order_sha256": event_order_sha256,
    }
    if any(not _valid_sha256(value) for value in hashes.values()):
        raise ValueError("every bridge parent hash must be a SHA-256 hex string")
    names, groups, _ = local_particle_residual_field_layout(DEFAULT_LOCAL_RESIDUAL_RADII)
    payload = {
        "contract": PREDICTION_ANCHORED_BRIDGE_RECIPE_CONTRACT,
        "rho_decimal": _decimal_rho(rho),
        "formula_version": BRIDGE_FORMULA_VERSION,
        "channel_policy": policy,
        "corrected_channel_indices": list(
            range(PHYSICAL_FIELD_COUNT if policy == BRIDGE_CHANNEL_PHYSICAL45 else FIELD_COUNT)
        ),
        "pass_through_channel_indices": (
            list(range(PHYSICAL_FIELD_COUNT, FIELD_COUNT))
            if policy == BRIDGE_CHANNEL_PHYSICAL45
            else []
        ),
        "parent_hashes": hashes,
        "radii": [float(value) for value in DEFAULT_LOCAL_RESIDUAL_RADII],
        "field_names": names,
        "field_groups": groups,
        "physical_loss_groups": physical_loss_groups(),
        "control_type": None if control_type is None else str(control_type),
        "control_seed": None if control_seed is None else int(control_seed),
        "audit_summary": dict(audit_summary or {}),
        "ram_execution_mode": "single_node_one_open_allocation",
        "storage_policy": BRIDGE_STORAGE_POLICY,
        "field_space": "physical_repository_native_units",
        "dense_bridge_artifact": False,
    }
    return with_content_hash(payload)


def validate_bridge_recipe(recipe: Mapping[str, Any]) -> None:
    if recipe.get("contract") != PREDICTION_ANCHORED_BRIDGE_RECIPE_CONTRACT:
        raise ValueError("unexpected bridge recipe contract")
    claimed = recipe.get("content_hash")
    unhashed = dict(recipe)
    unhashed.pop("content_hash", None)
    if claimed != canonical_sha256(unhashed):
        raise ValueError("bridge recipe content hash mismatch")
    if recipe.get("channel_policy") not in BRIDGE_CHANNEL_POLICIES:
        raise ValueError("bridge recipe has invalid channel policy")
    _decimal_rho(str(recipe.get("rho_decimal")))


def _validate_fields(
    f0: np.ndarray,
    f_true: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor = np.asarray(f0, dtype=np.float32)
    truth = np.asarray(f_true, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if anchor.shape != truth.shape or anchor.ndim != 3 or anchor.shape[-1] != FIELD_COUNT:
        raise ValueError("f0 and f_true must have the same [N, P, 50] shape")
    if valid.shape != anchor.shape[:2]:
        raise ValueError("field mask must match [N, P]")
    if not np.isfinite(anchor).all() or not np.isfinite(truth).all():
        raise ValueError("bridge inputs contain non-finite values")
    # Padding is semantically zero, not merely ignored by a later loss.
    if np.any(anchor[~valid] != 0) or np.any(truth[~valid] != 0):
        raise ValueError("masked bridge inputs must be exactly zero")
    return anchor, truth, valid


def validate_aligned_bridge_batch(
    f0: np.ndarray,
    f_true: np.ndarray,
    f0_mask: np.ndarray,
    f_true_mask: np.ndarray,
    *,
    f0_event_ids: Sequence[Any],
    f_true_event_ids: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fail closed on every alignment condition before bridge construction."""

    anchor_mask = np.asarray(f0_mask, dtype=bool)
    truth_mask = np.asarray(f_true_mask, dtype=bool)
    if not np.array_equal(anchor_mask, truth_mask):
        raise ValueError("f0 and f_true masks do not match")
    left = [str(value) for value in f0_event_ids]
    right = [str(value) for value in f_true_event_ids]
    if len(left) != np.asarray(f0).shape[0] or len(right) != np.asarray(f_true).shape[0]:
        raise ValueError("event IDs do not align with bridge batches")
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("duplicate event IDs are forbidden")
    if left != right:
        raise ValueError("f0 and f_true event/particle order does not match")
    return _validate_fields(f0, f_true, anchor_mask)


def virtual_bridge(
    f0: np.ndarray,
    f_true: np.ndarray,
    mask: np.ndarray,
    *,
    rho: str | float | Decimal,
    channel_policy: str = BRIDGE_CHANNEL_PHYSICAL45,
    delta_override: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate a bridge in physical units with exact five-channel pass-through."""

    anchor, truth, valid = _validate_fields(f0, f_true, mask)
    policy = str(channel_policy)
    if policy not in BRIDGE_CHANNEL_POLICIES:
        raise ValueError(f"unknown bridge channel policy {policy!r}")
    weight = np.float32(float(Decimal(_decimal_rho(rho))))
    delta = truth - anchor if delta_override is None else np.asarray(delta_override, dtype=np.float32)
    if delta.shape != anchor.shape or not np.isfinite(delta).all():
        raise ValueError("delta override must be finite and match f0")
    output = anchor.copy()
    stop = PHYSICAL_FIELD_COUNT if policy == BRIDGE_CHANNEL_PHYSICAL45 else FIELD_COUNT
    output[..., :stop] = anchor[..., :stop] + weight * delta[..., :stop]
    output[~valid] = 0.0
    if policy == BRIDGE_CHANNEL_PHYSICAL45 and not np.array_equal(
        output[..., PHYSICAL_FIELD_COUNT:], anchor[..., PHYSICAL_FIELD_COUNT:]
    ):
        raise AssertionError("physical45 bridge modified pass-through channels")
    return output


def bridge_response(
    f0: np.ndarray,
    f_true: np.ndarray,
    mask: np.ndarray,
    *,
    channel_policy: str = BRIDGE_CHANNEL_PHYSICAL45,
) -> dict[str, np.ndarray]:
    return {
        rho: virtual_bridge(f0, f_true, mask, rho=rho, channel_policy=channel_policy)
        for rho in RESPONSE_RHOS
    }


def _seed_for(seed: int, event_key: Any, suffix: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{event_key}\0{suffix}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _bin(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="right").astype(np.int16)


def hlt_matching_features(tokens: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    particles = np.asarray(tokens, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if particles.ndim != 3 or valid.shape != particles.shape[:2] or particles.shape[-1] < 4:
        raise ValueError("HLT matching features require [N,P,D>=4] tokens and [N,P] mask")
    if not np.isfinite(particles).all():
        raise ValueError("HLT tokens contain non-finite values")
    pt = np.where(valid, particles[..., 0], 0.0)
    eta = particles[..., 1]
    phi = particles[..., 2]
    energy = np.where(valid, particles[..., 3], 0.0)
    px = np.sum(pt * np.cos(phi), axis=1)
    py = np.sum(pt * np.sin(phi), axis=1)
    pz = np.sum(pt * np.sinh(eta), axis=1)
    e = np.sum(energy, axis=1)
    mass = np.sqrt(np.maximum(e * e - px * px - py * py - pz * pz, 0.0))
    return {
        "multiplicity": valid.sum(axis=1).astype(np.float64),
        "jet_pt": np.sqrt(px * px + py * py),
        "jet_mass": mass,
    }


def fit_matched_bin_edges(tokens: np.ndarray, mask: np.ndarray) -> dict[str, list[float]]:
    features = hlt_matching_features(tokens, mask)
    return {
        "multiplicity": np.quantile(features["multiplicity"], [1 / 3, 2 / 3]).astype(float).tolist(),
        "jet_pt": np.quantile(features["jet_pt"], [0.25, 0.5, 0.75]).astype(float).tolist(),
        "jet_mass": np.quantile(features["jet_mass"], [0.25, 0.5, 0.75]).astype(float).tolist(),
    }


def _identity_digest(value: Any, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}\0{value}".encode("utf-8")).hexdigest()


def build_matched_wrong_event_map(
    *,
    tokens: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    event_ids: Sequence[Any],
    seed: int,
    bin_edges: Mapping[str, Sequence[float]] | None = None,
    logical_block_size: int = 8192,
) -> dict[str, Any]:
    """Build the within-class/block deterministic derangement from Section 10."""

    n_events = int(np.asarray(tokens).shape[0])
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.shape != (n_events,) or len(event_ids) != n_events:
        raise ValueError("labels/event IDs must align with HLT tokens")
    identity_text = [str(value) for value in event_ids]
    if len(set(identity_text)) != n_events:
        raise ValueError("duplicate event IDs are forbidden")
    block_size = int(logical_block_size)
    if block_size <= 0:
        raise ValueError("logical_block_size must be positive")
    edges = dict(bin_edges or fit_matched_bin_edges(tokens, mask))
    features = hlt_matching_features(tokens, mask)
    bins = np.stack(
        [
            _bin(features["multiplicity"], edges["multiplicity"]),
            _bin(features["jet_pt"], edges["jet_pt"]),
            _bin(features["jet_mass"], edges["jet_mass"]),
        ],
        axis=1,
    )
    permutation = np.full(n_events, -1, dtype=np.int64)
    merge_records: list[dict[str, Any]] = []
    blocks = np.arange(n_events, dtype=np.int64) // block_size
    for block in np.unique(blocks):
        for label in np.unique(labels_array[blocks == block]):
            population = np.flatnonzero((blocks == block) & (labels_array == label))
            if population.size < 2:
                raise ValueError(
                    f"matched control invalid: block {int(block)} class {int(label)} has fewer than two events"
                )
            groups: dict[tuple[int, int, int], list[int]] = {}
            for index in population.tolist():
                groups.setdefault(tuple(int(v) for v in bins[index]), []).append(index)
            # Deterministically absorb singleton components until every group can
            # support a true cyclic derangement.
            while any(len(indices) == 1 for indices in groups.values()):
                source = min(key for key, indices in groups.items() if len(indices) == 1)
                candidates = [key for key in groups if key != source]
                target = min(
                    candidates,
                    key=lambda key: (
                        sum(abs(source[d] - key[d]) for d in range(3)),
                        abs(source[0] - key[0]),
                        abs(source[1] - key[1]),
                        abs(source[2] - key[2]),
                        key,
                    ),
                )
                groups[target].extend(groups.pop(source))
                merge_records.append(
                    {"block": int(block), "label": int(label), "source_bin": list(source), "target_bin": list(target)}
                )
            for bin_key, indices in sorted(groups.items()):
                ordered = sorted(indices, key=lambda idx: (_identity_digest(identity_text[idx], seed), idx))
                for position, index in enumerate(ordered):
                    permutation[index] = ordered[(position + 1) % len(ordered)]
    if np.any(permutation < 0) or np.any(permutation == np.arange(n_events)):
        raise AssertionError("matched wrong-event map is not a complete derangement")
    if not np.array_equal(labels_array, labels_array[permutation]):
        raise AssertionError("matched wrong-event map crossed classes")
    if not np.array_equal(blocks, blocks[permutation]):
        raise AssertionError("matched wrong-event map crossed logical blocks")
    payload = {
        "contract": MATCHED_WRONG_EVENT_MAP_CONTRACT,
        "seed": int(seed),
        "logical_block_size": block_size,
        "bin_edges": {key: [float(v) for v in values] for key, values in edges.items()},
        "event_order_sha256": canonical_sha256(identity_text),
        "permutation": permutation.tolist(),
        "merge_records": merge_records,
        "within_class": True,
        "within_block": True,
        "true_derangement": True,
    }
    return with_content_hash(payload)


def apply_bridge_control(
    f0: np.ndarray,
    f_true: np.ndarray,
    mask: np.ndarray,
    *,
    control_type: str,
    seed: int,
    rho: str | float | Decimal = "0.100",
    channel_policy: str = BRIDGE_CHANNEL_PHYSICAL45,
    event_ids: Sequence[Any] | None = None,
    wrong_event_map: Mapping[str, Any] | Sequence[int] | None = None,
) -> np.ndarray:
    """Generate one deterministic negative control without persistent tensors."""

    anchor, truth, valid = _validate_fields(f0, f_true, mask)
    kind = str(control_type)
    if kind not in BRIDGE_CONTROLS:
        raise ValueError(f"unknown bridge control {kind!r}")
    delta = truth - anchor
    controlled = delta.copy()
    ids = list(range(anchor.shape[0])) if event_ids is None else list(event_ids)
    if len(ids) != anchor.shape[0]:
        raise ValueError("event_ids must align with fields")
    if kind == CONTROL_EVENT_SHUFFLED:
        if wrong_event_map is None:
            raise ValueError("event_shuffled_delta requires a matched wrong-event map")
        if isinstance(wrong_event_map, Mapping):
            if wrong_event_map.get("contract") != MATCHED_WRONG_EVENT_MAP_CONTRACT:
                raise ValueError("wrong-event map contract mismatch")
            unhashed_map = dict(wrong_event_map)
            claimed_map_hash = unhashed_map.pop("content_hash", None)
            if claimed_map_hash != canonical_sha256(unhashed_map):
                raise ValueError("wrong-event map content hash mismatch")
            if event_ids is not None and wrong_event_map.get("event_order_sha256") != canonical_sha256(
                [str(value) for value in event_ids]
            ):
                raise ValueError("wrong-event map event order mismatch")
        permutation = np.asarray(
            wrong_event_map.get("permutation") if isinstance(wrong_event_map, Mapping) else wrong_event_map,
            dtype=np.int64,
        )
        if permutation.shape != (anchor.shape[0],) or np.any(permutation == np.arange(anchor.shape[0])):
            raise ValueError("wrong-event map must be an aligned true derangement")
        controlled = delta[permutation].copy()
        source_mask = valid[permutation]
        controlled *= source_mask[..., None]
        # A recipient cannot acquire values in its padding slots.
        controlled *= valid[..., None]
    elif kind == CONTROL_PARTICLE_SHUFFLED:
        controlled.fill(0.0)
        for event in range(anchor.shape[0]):
            indices = np.flatnonzero(valid[event])
            if indices.size < 2:
                controlled[event, indices] = delta[event, indices]
                continue
            rng = np.random.default_rng(_seed_for(seed, ids[event], kind))
            shift = int(rng.integers(1, indices.size))
            controlled[event, indices] = delta[event, np.roll(indices, shift)]
    elif kind == CONTROL_SIGN_REVERSED:
        controlled = -delta
    elif kind == CONTROL_SAME_NORM_RANDOM:
        controlled.fill(0.0)
        for event in range(anchor.shape[0]):
            indices = np.flatnonzero(valid[event])
            if not indices.size:
                continue
            rng = np.random.default_rng(_seed_for(seed, ids[event], kind))
            for group_indices in physical_loss_groups().values():
                target_values = delta[event][np.ix_(indices, group_indices)]
                random_values = rng.normal(size=target_values.shape).astype(np.float32)
                target_norm = float(np.linalg.norm(target_values.astype(np.float64)))
                random_norm = float(np.linalg.norm(random_values.astype(np.float64)))
                if target_norm > 0 and random_norm > 0:
                    random_values *= np.float32(target_norm / random_norm)
                else:
                    random_values.fill(0.0)
                controlled[event][np.ix_(indices, group_indices)] = random_values
    elif kind == CONTROL_RADIUS_PERMUTED:
        # All three radius blocks have the same 15-coordinate layout.
        shift = 1 + int(seed) % 2
        source_radii = np.roll(np.arange(3), shift)
        for destination, source in enumerate(source_radii.tolist()):
            controlled[..., 15 * destination : 15 * (destination + 1)] = delta[
                ..., 15 * source : 15 * (source + 1)
            ]
    controlled[~valid] = 0.0
    return virtual_bridge(
        anchor,
        truth,
        valid,
        rho=rho,
        channel_policy=channel_policy,
        delta_override=controlled,
    )


BatchFactory = Callable[[], Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]]


def _factory(batches: BatchFactory | Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> BatchFactory:
    if callable(batches):
        return batches
    materialized = tuple(batches)
    return lambda: iter(materialized)


def _channel_values(
    factory: BatchFactory,
    channel: int,
    *,
    value: str,
    rho: float,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for f0, f_true, mask in factory():
        anchor, truth, valid = _validate_fields(f0, f_true, mask)
        if value == "abs_f0":
            data = np.abs(anchor[..., channel][valid])
        elif value == "abs_correction":
            data = np.abs((np.float32(rho) * (truth - anchor))[..., channel][valid])
        else:
            raise ValueError(value)
        pieces.append(np.asarray(data, dtype=np.float64))
    if not pieces or sum(piece.size for piece in pieces) == 0:
        raise ValueError("scaler fit has no valid particles")
    return np.concatenate(pieces)


@dataclass(frozen=True)
class BridgeScalers:
    mu_f0: np.ndarray
    sigma_f0: np.ndarray
    q99_delta: np.ndarray
    sigma_delta: np.ndarray
    trust_scale: np.ndarray
    epsilon: np.ndarray
    active: np.ndarray
    sparse_nonzero_fallback: np.ndarray
    valid_count: int
    parent_hashes: Mapping[str, str]
    channel_policy: str
    rho_decimal: str = "0.100"

    def __post_init__(self) -> None:
        for name in ("mu_f0", "sigma_f0", "q99_delta", "sigma_delta", "trust_scale", "epsilon"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (FIELD_COUNT,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain 50 finite values")
            object.__setattr__(self, name, value)
        for name in ("active", "sparse_nonzero_fallback"):
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != (FIELD_COUNT,):
                raise ValueError(f"{name} must contain 50 flags")
            object.__setattr__(self, name, value)
        if np.any(self.sigma_f0 < 1e-6) or np.any(self.sigma_delta <= 0) or np.any(self.trust_scale <= 0):
            raise ValueError("scaler floors were not enforced")
        if self.channel_policy not in BRIDGE_CHANNEL_POLICIES:
            raise ValueError("invalid scaler channel policy")
        object.__setattr__(self, "rho_decimal", _decimal_rho(self.rho_decimal))

    def conditioning_standardize(self, fields: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = np.asarray(fields, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool)
        if values.shape[-1] != FIELD_COUNT or valid.shape != values.shape[:2]:
            raise ValueError("conditioning fields/mask have incompatible shapes")
        output = ((values.astype(np.float64) - self.mu_f0) / self.sigma_f0).astype(np.float32)
        output[~valid] = 0.0
        return output

    def conditioning_inverse(self, standardized: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = np.asarray(standardized, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool)
        if values.shape[-1] != FIELD_COUNT or valid.shape != values.shape[:2]:
            raise ValueError("standardized fields/mask have incompatible shapes")
        output = (values.astype(np.float64) * self.sigma_f0 + self.mu_f0).astype(np.float32)
        output[~valid] = 0.0
        return output

    def bounded_physical_correction(self, raw_standardized: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_standardized, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool)
        if raw.shape[-1] != FIELD_COUNT or valid.shape != raw.shape[:2]:
            raise ValueError("correction/mask have incompatible shapes")
        physical_raw = raw.astype(np.float64) * self.sigma_delta
        bounded = self.trust_scale * np.tanh(physical_raw / self.trust_scale)
        bounded[..., ~self.active] = 0.0
        if self.channel_policy == BRIDGE_CHANNEL_PHYSICAL45:
            bounded[..., PHYSICAL_FIELD_COUNT:] = 0.0
        bounded[~valid] = 0.0
        return bounded.astype(np.float32)

    def corrected_physical_fields(
        self,
        f0_physical: np.ndarray,
        raw_standardized: np.ndarray,
        mask: np.ndarray,
        *,
        input_space: str = "physical",
    ) -> np.ndarray:
        """Inverse-scale/bound first, then add to the physical R0 anchor."""

        if str(input_space) != "physical":
            raise ValueError("corrections may only be added to physical-space f0")
        anchor = np.asarray(f0_physical, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool)
        if anchor.shape != np.asarray(raw_standardized).shape or valid.shape != anchor.shape[:2]:
            raise ValueError("physical anchor/correction/mask shapes do not align")
        if not np.isfinite(anchor).all():
            raise ValueError("physical f0 contains non-finite values")
        output = anchor + self.bounded_physical_correction(raw_standardized, valid)
        output[~valid] = 0.0
        if self.channel_policy == BRIDGE_CHANNEL_PHYSICAL45:
            output[..., PHYSICAL_FIELD_COUNT:] = anchor[..., PHYSICAL_FIELD_COUNT:]
        return output.astype(np.float32)

    def to_artifact(self) -> dict[str, Any]:
        payload = {
            "contract": PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT,
            "fit_version": BRIDGE_SCALER_FIT_VERSION,
            "fit_split": "stack_train_distill",
            "rho_decimal": self.rho_decimal,
            "channel_policy": self.channel_policy,
            "valid_particle_count": int(self.valid_count),
            "mu_f0": self.mu_f0.tolist(),
            "sigma_f0": self.sigma_f0.tolist(),
            "q99_delta": self.q99_delta.tolist(),
            "sigma_delta": self.sigma_delta.tolist(),
            "trust_scale": self.trust_scale.tolist(),
            "epsilon": self.epsilon.tolist(),
            "active": self.active.tolist(),
            "sparse_nonzero_fallback": self.sparse_nonzero_fallback.tolist(),
            "parent_hashes": dict(self.parent_hashes),
            "numerical_space": "physical_fields_conditioning_standardized_corrections_standardized_v1",
        }
        return with_content_hash(payload)

    @classmethod
    def from_artifact(cls, payload: Mapping[str, Any]) -> "BridgeScalers":
        if payload.get("contract") != PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT:
            raise ValueError("unexpected scaler contract")
        unhashed = dict(payload)
        claimed = unhashed.pop("content_hash", None)
        if claimed != canonical_sha256(unhashed):
            raise ValueError("scaler content hash mismatch")
        return cls(
            mu_f0=np.asarray(payload["mu_f0"]),
            sigma_f0=np.asarray(payload["sigma_f0"]),
            q99_delta=np.asarray(payload["q99_delta"]),
            sigma_delta=np.asarray(payload["sigma_delta"]),
            trust_scale=np.asarray(payload["trust_scale"]),
            epsilon=np.asarray(payload["epsilon"]),
            active=np.asarray(payload["active"]),
            sparse_nonzero_fallback=np.asarray(payload["sparse_nonzero_fallback"]),
            valid_count=int(payload["valid_particle_count"]),
            parent_hashes=dict(payload["parent_hashes"]),
            channel_policy=str(payload["channel_policy"]),
            rho_decimal=str(payload["rho_decimal"]),
        )


def fit_bridge_scalers(
    batches: BatchFactory | Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    parent_hashes: Mapping[str, str],
    channel_policy: str = BRIDGE_CHANNEL_PHYSICAL45,
    rho: str | float | Decimal = "0.100",
) -> BridgeScalers:
    """Fit exact per-channel statistics in one streamed model pass.

    Valid values are retained only in host RAM for the duration of fitting so an
    expensive frozen R0 is evaluated once.  They are never written as a dense
    persistent artifact.  The caller's allocation ledger must include this
    transient scaler-fit working set in its derived reservation.
    """

    factory = _factory(batches)
    if not parent_hashes or any(not _valid_sha256(value) for value in parent_hashes.values()):
        raise ValueError("scaler provenance requires nonempty SHA-256 parent hashes")
    rho_decimal = _decimal_rho(rho)
    rho_value = float(Decimal(rho_decimal))
    count = 0
    sums = np.zeros(FIELD_COUNT, dtype=np.float64)
    squares = np.zeros(FIELD_COUNT, dtype=np.float64)
    anchor_pieces: list[np.ndarray] = []
    correction_pieces: list[np.ndarray] = []
    for f0, f_true, mask in factory():
        anchor, truth, valid = _validate_fields(f0, f_true, mask)
        selected_f32 = anchor[valid].astype(np.float32, copy=True)
        selected = selected_f32.astype(np.float64)
        count += int(selected.shape[0])
        sums += selected.sum(axis=0, dtype=np.float64)
        squares += np.square(selected).sum(axis=0, dtype=np.float64)
        anchor_pieces.append(selected_f32)
        correction_pieces.append(
            (np.float32(rho_value) * (truth - anchor))[valid].astype(np.float32, copy=True)
        )
    if count == 0:
        raise ValueError("cannot fit bridge scalers without valid particles")
    mu = sums / count
    variance = np.maximum(squares / count - mu * mu, 0.0)
    sigma_f0 = np.maximum(np.sqrt(variance), 1e-6)
    q99 = np.zeros(FIELD_COUNT, dtype=np.float64)
    epsilon = np.zeros(FIELD_COUNT, dtype=np.float64)
    active = np.ones(FIELD_COUNT, dtype=bool)
    fallback = np.zeros(FIELD_COUNT, dtype=bool)
    for channel in range(FIELD_COUNT):
        abs_f0 = np.abs(np.concatenate([piece[:, channel] for piece in anchor_pieces])).astype(np.float64)
        epsilon[channel] = max(1e-6, 1e-3 * float(np.quantile(abs_f0, 0.99)))
        correction = np.abs(
            np.concatenate([piece[:, channel] for piece in correction_pieces])
        ).astype(np.float64)
        provisional = float(np.quantile(correction, 0.99))
        nonzero = correction[correction != 0]
        if provisional == 0.0 and nonzero.size:
            provisional = float(np.quantile(nonzero, 0.99))
            fallback[channel] = True
        if nonzero.size == 0:
            active[channel] = False
            provisional = 0.0
        q99[channel] = provisional
    anchor_pieces.clear()
    correction_pieces.clear()
    sigma_delta = np.maximum(q99, epsilon)
    trust = np.maximum(2.0 * q99, epsilon)
    if channel_policy == BRIDGE_CHANNEL_PHYSICAL45:
        active[PHYSICAL_FIELD_COUNT:] = False
    return BridgeScalers(
        mu_f0=mu,
        sigma_f0=sigma_f0,
        q99_delta=q99,
        sigma_delta=sigma_delta,
        trust_scale=trust,
        epsilon=epsilon,
        active=active,
        sparse_nonzero_fallback=fallback,
        valid_count=count,
        parent_hashes=dict(parent_hashes),
        channel_policy=channel_policy,
        rho_decimal=rho_decimal,
    )


__all__ = [
    "PREDICTION_ANCHORED_BRIDGE_RECIPE_CONTRACT",
    "PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT",
    "MATCHED_WRONG_EVENT_MAP_CONTRACT",
    "BRIDGE_CHANNEL_PHYSICAL45",
    "BRIDGE_CHANNEL_ALL50",
    "BRIDGE_CONTROLS",
    "RESPONSE_RHOS",
    "BridgeScalers",
    "apply_bridge_control",
    "bridge_response",
    "build_bridge_recipe",
    "build_matched_wrong_event_map",
    "fit_bridge_scalers",
    "fit_matched_bin_edges",
    "hlt_matching_features",
    "physical_loss_groups",
    "validate_bridge_recipe",
    "validate_aligned_bridge_batch",
    "virtual_bridge",
]

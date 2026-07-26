"""Locked robust-normalization contracts and the Step-3 train-only fitter."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .registry import validate_relation_family_registry
from .provenance import validate_raw_input_schema_contract

try:  # Keep contract-only imports usable without the training stack.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None


NORMALIZATION_CONTRACT = "relational_part_normalization_contract_v3"
RELATION_NORMALIZATION_ARTIFACT_CONTRACT = (
    "relational_part_relation_normalization_v2"
)
GLOBAL_EPSILON = 1.0e-6
NORMALIZATION_JET_SALT = "relational_part_normalization_jets_v1"
NORMALIZATION_PAIR_SALT = "relational_part_normalization_pairs_v1"
NORMALIZATION_QUANTILE_METHOD = "linear"
NORMALIZATION_JET_LIMIT = 50_000
NORMALIZATION_PAIR_LIMIT_PER_JET = 64

PT_RAW_FEATURE_NAMES = (
    "query_pt_fraction",
    "context_pt_fraction",
    "query_log_pt_fraction",
    "context_log_pt_fraction",
    "context_minus_query_log_pt_fraction",
    "log_pair_scalar_pt_fraction",
    "signed_context_minus_query_pt_asymmetry",
    "query_average_normalized_pt_rank",
    "context_average_normalized_pt_rank",
    "context_minus_query_pt_rank",
)
PT_ROBUST_FEATURE_NAMES = (
    "query_log_pt_fraction",
    "context_log_pt_fraction",
    "context_minus_query_log_pt_fraction",
    "log_pair_scalar_pt_fraction",
)
CHARGE_RAW_FEATURE_NAMES = (
    "query_charge",
    "context_charge",
    "charge_product",
    "half_absolute_charge_difference",
    "both_neutral",
    "exactly_one_charged",
    "same_nonzero_sign",
    "opposite_nonzero_sign",
)
CHARGE_ROBUST_FEATURE_NAMES = CHARGE_RAW_FEATURE_NAMES[:4]
TRACK_NODE_CONTINUOUS_NAMES = (
    "d0",
    "dz",
    "log_d0_sigma_effective",
    "log_dz_sigma_effective",
    "asinh_d0_significance",
    "asinh_dz_significance",
)
TRACK_COMPATIBILITY_FEATURE_NAMES = (
    "log1p_chi2",
    "exp_minus_half_clipped_chi2",
    "minimum_abs_d0_significance",
    "maximum_abs_d0_significance",
    "minimum_abs_dz_significance",
    "maximum_abs_dz_significance",
    "d0_significance_product",
    "dz_significance_product",
    "context_minus_query_normalized_d0",
    "context_minus_query_normalized_dz",
    "sin_query_minus_context_delta_phi",
    "cos_query_minus_context_delta_phi",
    "log_delta_r",
)
TRACK_COMPATIBILITY_ROBUST_NAMES = (
    "log1p_chi2",
    "minimum_abs_d0_significance",
    "maximum_abs_d0_significance",
    "minimum_abs_dz_significance",
    "maximum_abs_dz_significance",
    "d0_significance_product",
    "dz_significance_product",
    "context_minus_query_normalized_d0",
    "context_minus_query_normalized_dz",
    "log_delta_r",
)
DENSITY_NODE_FEATURE_NAMES = (
    "annulus_0_count",
    "annulus_0_pt_fraction",
    "annulus_1_count",
    "annulus_1_pt_fraction",
    "annulus_2_count",
    "annulus_2_pt_fraction",
    "annulus_3_count",
    "annulus_3_pt_fraction",
    "smooth_0_count",
    "smooth_0_pt_fraction",
    "smooth_1_count",
    "smooth_1_pt_fraction",
    "smooth_2_count",
    "smooth_2_pt_fraction",
    "smooth_3_count",
    "smooth_3_pt_fraction",
    "local_charged_pt_fraction",
    "local_neutral_hadron_pt_fraction",
    "local_photon_pt_fraction",
    "local_displaced_track_pt_fraction",
    "valid_neighbor_fraction_R0p40",
    "self_share_R0p20",
)


if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


def build_normalization_contract(
    *,
    split_binding_sha256: str,
) -> dict[str, Any]:
    split_binding_sha256 = require_sha256(
        split_binding_sha256, name="split_binding_sha256"
    )
    return with_content_hash(
        {
            "contract": NORMALIZATION_CONTRACT,
            "schema_version": 3,
            "split_binding_sha256": split_binding_sha256,
            "fit_split": "model_train",
            "global_epsilon": GLOBAL_EPSILON,
            "selection": {
                "jet_count": NORMALIZATION_JET_LIMIT,
                "jet_policy": "lowest_salted_identity_sha256",
                "jet_salt": NORMALIZATION_JET_SALT,
                "jet_hash_preimage": "utf8(salt)+NUL+utf8(file#entry)",
                "maximum_directed_pairs_per_jet": (
                    NORMALIZATION_PAIR_LIMIT_PER_JET
                ),
                "pair_policy": "lowest_salted_directed_pair_sha256",
                "pair_salt": NORMALIZATION_PAIR_SALT,
                "pair_hash_preimage": (
                    "utf8(salt)+NUL+utf8(file#entry)+NUL+ascii(query)"
                    "+NUL+ascii(context)"
                ),
                "pair_domain": "all_ordered_valid_pairs_including_diagonal",
                "node_features_counted_once_per_particle": True,
            },
            "continuous_transform": {
                "center": "median",
                "quartile_method": NORMALIZATION_QUANTILE_METHOD,
                "quantile_arithmetic": "numpy_float64",
                "scale": "max((q75-q25)/1.349,1e-6)",
                "clip_min": -8.0,
                "clip_max": 8.0,
            },
            "never_normalized": [
                "binary_indicators",
                "categorical_embedding_indices",
                "explicit_fixed_scale_channels",
            ],
            "applicability_rules": {
                "PT_pair": "all_valid_directed_particle_pairs",
                "CHARGE_pair": "all_valid_directed_particle_pairs",
                "TRACK_node": "valid_tracks_counted_once",
                "TRACK_compatibility": (
                    "valid_valid_distinct_or_diagonal_as_declared"
                ),
                "DENSITY_node": "valid_particles_counted_once",
                "REGION_node": (
                    "valid_particles_per_defined_resolution_counted_once"
                ),
                "REGION_pair": (
                    "valid_directed_pairs_per_defined_resolution"
                ),
                "REGION_merge": "valid_distinct_directed_pairs_only",
            },
            "invalid_policy": {
                "excluded_from_fit": True,
                "safe_placeholder_before_numeric_transform": True,
                "zero_after_normalization": True,
                "masked_after_learned_encoder": True,
            },
            "track_uncertainty_floor": {
                "fit_population": (
                    "all_valid_positive_model_train_track_errors"
                ),
                "d0_formula": "max(q01(valid_positive_d0err),1e-6)",
                "dz_formula": "max(q01(valid_positive_dzerr),1e-6)",
                "recorded_quantiles": [0.01, 0.05, 0.5, 0.95, 0.99],
                "quantile_method": NORMALIZATION_QUANTILE_METHOD,
                "quantile_arithmetic": "numpy_float64",
            },
            "raw_input_schema_required": "jetclass_hlt_raw_v1",
            "track_sentinel_policy_source": "locked_raw_input_schema_only",
            "required_channel_record_fields": [
                "family_id",
                "feature_name",
                "applicability_rule_id",
                "applicable_count",
                "median",
                "q25",
                "q75",
                "robust_scale",
                "applicable_zero_fraction",
                "post_normalization_clip_fraction",
            ],
            "validation_or_test_statistics_allowed": False,
        }
    )


def validate_normalization_contract(contract: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        contract,
        expected_contract=NORMALIZATION_CONTRACT,
    )
    split_sha = require_sha256(
        contract.get("split_binding_sha256"),
        name="normalization_contract.split_binding_sha256",
    )
    semantic = dict(contract)
    semantic.pop("content_hash", None)
    semantic.pop("source", None)
    expected = build_normalization_contract(split_binding_sha256=split_sha)
    expected.pop("content_hash")
    if semantic != expected:
        raise ValueError("normalization contract differs from the locked contract")
    return digest


def _identity_key(identity: Any) -> str:
    key_method = getattr(identity, "key", None)
    if callable(key_method):
        key = key_method()
    elif isinstance(identity, Mapping):
        key = f"{identity['file']}#{int(identity['entry'])}"
    elif isinstance(identity, str):
        key = identity
    else:
        raise TypeError("jet identities must expose key(), be mappings, or be strings")
    key = str(key)
    if not key or "\x00" in key:
        raise ValueError("canonical jet identity is empty or contains NUL")
    return key


def _salted_digest(*parts: str) -> bytes:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).digest()


def select_normalization_jet_indices(
    identities: Sequence[Any],
    *,
    limit: int = NORMALIZATION_JET_LIMIT,
) -> np.ndarray:
    """Select jets by the locked salted identity ordering."""

    if int(limit) <= 0:
        raise ValueError("normalization jet limit must be positive")
    keys = [_identity_key(identity) for identity in identities]
    if len(keys) != len(set(keys)):
        raise ValueError("normalization input contains duplicate jet identities")
    order = sorted(
        range(len(keys)),
        key=lambda index: (
            _salted_digest(NORMALIZATION_JET_SALT, keys[index]),
            keys[index],
        ),
    )
    return np.asarray(order[: min(int(limit), len(order))], dtype=np.int64)


def select_normalization_pairs(
    identity: Any,
    valid_indices: Sequence[int],
    *,
    limit: int = NORMALIZATION_PAIR_LIMIT_PER_JET,
) -> tuple[tuple[int, int], ...]:
    """Select ordered valid pairs, including diagonal pairs, by salted hash."""

    if int(limit) <= 0:
        raise ValueError("normalization pair limit must be positive")
    key = _identity_key(identity)
    indices = tuple(int(index) for index in valid_indices)
    if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError("valid particle indices must be unique and nonnegative")
    pairs = [(query, context) for query in indices for context in indices]
    pairs.sort(
        key=lambda pair: (
            _salted_digest(
                NORMALIZATION_PAIR_SALT,
                key,
                str(pair[0]),
                str(pair[1]),
            ),
            pair,
        )
    )
    return tuple(pairs[: min(int(limit), len(pairs))])


def _average_tied_descending_rank_numpy(pt: np.ndarray) -> np.ndarray:
    if pt.size <= 1:
        return np.zeros_like(pt, dtype=np.float64)
    greater = (pt[None, :] > pt[:, None]).sum(axis=1, dtype=np.int64)
    equal = (pt[None, :] == pt[:, None]).sum(axis=1, dtype=np.int64)
    return (greater + 0.5 * (equal - 1)) / float(pt.size - 1)


def _step3_pair_values(
    tokens: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    valid_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return (
            np.empty((0, len(PT_ROBUST_FEATURE_NAMES)), dtype=np.float64),
            np.empty((0, len(CHARGE_ROBUST_FEATURE_NAMES)), dtype=np.float64),
        )
    valid_indices = np.asarray(tuple(int(index) for index in valid_indices))
    pt_valid = np.asarray(tokens[valid_indices, 0], dtype=np.float64)
    if np.any(pt_valid < 0) or not np.isfinite(pt_valid).all():
        raise ValueError("valid HLT transverse momenta must be finite and nonnegative")
    pt_sum = float(pt_valid.sum(dtype=np.float64))
    fractions = pt_valid / (pt_sum + GLOBAL_EPSILON)
    log_fractions = np.log((pt_valid + GLOBAL_EPSILON) / (pt_sum + GLOBAL_EPSILON))
    rank = _average_tied_descending_rank_numpy(pt_valid)
    position = {int(original): offset for offset, original in enumerate(valid_indices)}

    pt_rows: list[list[float]] = []
    charge_rows: list[list[float]] = []
    for query, context in pairs:
        qi = position[query]
        cj = position[context]
        pair_log = np.log(
            (pt_valid[qi] + pt_valid[cj]) / (pt_sum + GLOBAL_EPSILON)
            + GLOBAL_EPSILON
        )
        pt_rows.append(
            [
                log_fractions[qi],
                log_fractions[cj],
                log_fractions[cj] - log_fractions[qi],
                pair_log,
            ]
        )

        charges = np.asarray(tokens[[query, context], 4], dtype=np.float64)
        locked = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
        distances = np.abs(charges[:, None] - locked[None, :])
        nearest = locked[np.argmin(distances, axis=1)]
        if np.any(np.min(distances, axis=1) > GLOBAL_EPSILON):
            raise ValueError("valid HLT charges violate the locked integer tolerance")
        q_i, q_j = map(float, nearest)
        charge_rows.append(
            [q_i, q_j, q_i * q_j, abs(q_i - q_j) / 2.0]
        )
    return np.asarray(pt_rows), np.asarray(charge_rows)


def _track_sentinel_policy_from_schema(
    raw_input_schema: Mapping[str, Any],
) -> dict[str, float | None]:
    fields = raw_input_schema.get("track_fields")
    if not isinstance(fields, Mapping) or set(fields) != {
        "d0",
        "d0err",
        "dz",
        "dzerr",
    }:
        raise ValueError("raw-input schema has an invalid track-field declaration")
    output: dict[str, float | None] = {}
    for name in ("d0", "d0err", "dz", "dzerr"):
        declaration = fields[name]
        if not isinstance(declaration, Mapping):
            raise ValueError(f"raw-input schema {name} declaration is invalid")
        policy = declaration.get("missing_policy")
        if policy == "no_numeric_sentinel":
            output[name] = None
        elif policy == "numeric_sentinel":
            value = float(declaration.get("sentinel"))
            if not np.isfinite(value):
                raise ValueError(f"raw-input schema {name} sentinel is nonfinite")
            output[name] = value
        else:
            raise ValueError(f"unsupported {name} missing policy {policy!r}")
    return output


def _track_valid_numpy(
    row: np.ndarray,
    particle_valid: np.ndarray,
    pid_category: np.ndarray,
    sentinel_policy: Mapping[str, float | None],
) -> np.ndarray:
    charged = (pid_category == 0) | (pid_category == 3) | (pid_category == 4)
    values = {
        "d0": np.asarray(row[:, 10], dtype=np.float64),
        "d0err": np.asarray(row[:, 11], dtype=np.float64),
        "dz": np.asarray(row[:, 12], dtype=np.float64),
        "dzerr": np.asarray(row[:, 13], dtype=np.float64),
    }
    valid = np.asarray(particle_valid, dtype=bool) & charged
    for value in values.values():
        valid &= np.isfinite(value)
    valid &= values["d0err"] > 0
    valid &= values["dzerr"] > 0
    for name, sentinel in sentinel_policy.items():
        if sentinel is not None:
            valid &= values[name] != sentinel
    return valid


def _uncertainty_floor_audit(
    values: np.ndarray,
) -> dict[str, Any]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0 or not np.isfinite(sample).all() or np.any(sample <= 0):
        raise ValueError("uncertainty-floor population must be positive and finite")
    quantile_levels = [0.01, 0.05, 0.50, 0.95, 0.99]
    quantiles = np.quantile(
        sample,
        quantile_levels,
        method=NORMALIZATION_QUANTILE_METHOD,
    )
    return {
        "applicable_count": int(sample.size),
        "quantiles": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
        "floor": max(float(quantiles[0]), GLOBAL_EPSILON),
    }


def _track_node_values_numpy(
    row: np.ndarray,
    track_valid: np.ndarray,
    *,
    d0_floor: float,
    dz_floor: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    d0 = np.nan_to_num(
        np.asarray(row[:, 10], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    d0err = np.nan_to_num(
        np.asarray(row[:, 11], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    dz = np.nan_to_num(
        np.asarray(row[:, 12], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    dzerr = np.nan_to_num(
        np.asarray(row[:, 13], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    sigma_d0 = np.sqrt(d0err**2 + float(d0_floor) ** 2)
    sigma_dz = np.sqrt(dzerr**2 + float(dz_floor) ** 2)
    raw_s0 = d0 / sigma_d0
    raw_sz = dz / sigma_dz
    values = np.stack(
        (
            d0,
            dz,
            np.log(np.maximum(sigma_d0, GLOBAL_EPSILON)),
            np.log(np.maximum(sigma_dz, GLOBAL_EPSILON)),
            np.arcsinh(raw_s0),
            np.arcsinh(raw_sz),
        ),
        axis=1,
    )
    return values[track_valid], {
        "d0": d0,
        "dz": dz,
        "sigma_d0": sigma_d0,
        "sigma_dz": sigma_dz,
        "s0": np.arcsinh(raw_s0),
        "sz": np.arcsinh(raw_sz),
        "raw_s0": raw_s0,
        "raw_sz": raw_sz,
    }


def _track_pair_values_numpy(
    row: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    node: Mapping[str, np.ndarray],
) -> np.ndarray:
    rows: list[list[float]] = []
    for query, context in pairs:
        d0_denominator = np.sqrt(
            node["sigma_d0"][query] ** 2 + node["sigma_d0"][context] ** 2
        )
        dz_denominator = np.sqrt(
            node["sigma_dz"][query] ** 2 + node["sigma_dz"][context] ** 2
        )
        delta_d0 = (
            node["d0"][context] - node["d0"][query]
        ) / max(float(d0_denominator), GLOBAL_EPSILON)
        delta_dz = (
            node["dz"][context] - node["dz"][query]
        ) / max(float(dz_denominator), GLOBAL_EPSILON)
        chi2 = delta_d0**2 + delta_dz**2
        delta_phi = np.arctan2(
            np.sin(float(row[query, 2] - row[context, 2])),
            np.cos(float(row[query, 2] - row[context, 2])),
        )
        delta_eta = float(row[query, 1] - row[context, 1])
        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
        full = [
            np.log1p(chi2),
            np.exp(-0.5 * min(chi2, 25.0)),
            min(abs(node["s0"][query]), abs(node["s0"][context])),
            max(abs(node["s0"][query]), abs(node["s0"][context])),
            min(abs(node["sz"][query]), abs(node["sz"][context])),
            max(abs(node["sz"][query]), abs(node["sz"][context])),
            node["s0"][query] * node["s0"][context],
            node["sz"][query] * node["sz"][context],
            delta_d0,
            delta_dz,
            np.sin(delta_phi),
            np.cos(delta_phi),
            np.log(delta_r + GLOBAL_EPSILON),
        ]
        rows.append(
            [
                full[TRACK_COMPATIBILITY_FEATURE_NAMES.index(name)]
                for name in TRACK_COMPATIBILITY_ROBUST_NAMES
            ]
        )
    return np.asarray(rows, dtype=np.float64).reshape(
        -1, len(TRACK_COMPATIBILITY_ROBUST_NAMES)
    )


def _density_node_values_numpy(
    row: np.ndarray,
    particle_valid: np.ndarray,
    pid_category: np.ndarray,
    track_valid: np.ndarray,
    node: Mapping[str, np.ndarray],
) -> np.ndarray:
    pt = np.where(particle_valid, row[:, 0], 0.0).astype(np.float64)
    if np.any(pt < 0):
        raise ValueError("valid transverse momentum must be nonnegative")
    eta = np.where(particle_valid, row[:, 1], 0.0).astype(np.float64)
    phi = np.where(particle_valid, row[:, 2], 0.0).astype(np.float64)
    delta_eta = eta[:, None] - eta[None, :]
    delta_phi = np.arctan2(
        np.sin(phi[:, None] - phi[None, :]),
        np.cos(phi[:, None] - phi[None, :]),
    )
    delta_r = np.sqrt(np.maximum(delta_eta**2 + delta_phi**2, 0.0))
    length = row.shape[0]
    neighbors = (
        particle_valid[:, None]
        & particle_valid[None, :]
        & ~np.eye(length, dtype=bool)
    )
    jet_pt = float(pt.sum(dtype=np.float64))
    outputs: list[np.ndarray] = []
    boundaries = (0.0, 0.05, 0.10, 0.20, 0.40)
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        annulus = neighbors & (delta_r <= upper)
        if index:
            annulus &= delta_r > lower
        count = np.log1p(annulus.sum(axis=1)) / np.log1p(128.0)
        pt_sum = (annulus * pt[None, :]).sum(axis=1)
        outputs.extend((count, pt_sum / (jet_pt + GLOBAL_EPSILON)))
    denominator = max(int(particle_valid.sum()) - 1, 1)
    log_radius = np.log(delta_r + GLOBAL_EPSILON)
    for center in (0.025, 0.071, 0.141, 0.283):
        kernel = np.exp(
            -(log_radius - np.log(center)) ** 2 / (2.0 * 0.45**2)
        ) * neighbors
        count_response = kernel.sum(axis=1) / denominator
        pt_response = (kernel * pt[None, :]).sum(axis=1) / (
            jet_pt + GLOBAL_EPSILON
        )
        if int(particle_valid.sum()) <= 1:
            count_response[:] = 0.0
            pt_response[:] = 0.0
        outputs.extend((count_response, pt_response))
    local = neighbors & (delta_r <= 0.20)
    local_pt = (local * pt[None, :]).sum(axis=1)

    def fraction(category_mask: np.ndarray) -> np.ndarray:
        numerator = (local * category_mask[None, :] * pt[None, :]).sum(axis=1)
        return np.divide(
            numerator,
            local_pt,
            out=np.zeros_like(numerator),
            where=local_pt > 0,
        )

    charged = (pid_category == 0) | (pid_category == 3) | (pid_category == 4)
    displaced = track_valid & (
        (np.abs(node["raw_s0"]) >= 2.0)
        | (np.abs(node["raw_sz"]) >= 2.0)
    )
    outputs.extend(
        (
            fraction(charged),
            fraction(pid_category == 1),
            fraction(pid_category == 2),
            fraction(displaced),
        )
    )
    within_040 = neighbors & (delta_r <= 0.40)
    neighbor_fraction = within_040.sum(axis=1) / denominator
    if int(particle_valid.sum()) <= 1:
        neighbor_fraction[:] = 0.0
    self_denominator = pt + local_pt
    self_share = np.divide(
        pt,
        self_denominator,
        out=np.zeros_like(pt),
        where=self_denominator > 0,
    )
    outputs.extend((neighbor_fraction, self_share))
    descriptor = np.stack(outputs, axis=1)
    return descriptor[particle_valid]


def _identity_sequence_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fit_records(
    values: np.ndarray,
    *,
    family_id: str,
    feature_names: Sequence[str],
    applicability_rule_id: str,
) -> list[dict[str, Any]]:
    if values.ndim != 2 or int(values.shape[1]) != len(feature_names):
        raise ValueError("normalization sample matrix has the wrong shape")
    if int(values.shape[0]) == 0:
        raise ValueError(f"{family_id} has no applicable model_train samples")
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{family_id} normalization samples are nonfinite")
    records: list[dict[str, Any]] = []
    for column, feature_name in enumerate(feature_names):
        sample = np.asarray(values[:, column], dtype=np.float64)
        q25, median, q75 = np.quantile(
            sample,
            [0.25, 0.5, 0.75],
            method=NORMALIZATION_QUANTILE_METHOD,
        )
        scale = max(float((q75 - q25) / 1.349), GLOBAL_EPSILON)
        standardized = (sample - median) / scale
        records.append(
            {
                "family_id": family_id,
                "feature_name": str(feature_name),
                "applicability_rule_id": applicability_rule_id,
                "applicable_count": int(sample.size),
                "median": float(median),
                "q25": float(q25),
                "q75": float(q75),
                "robust_scale": float(scale),
                "applicable_zero_fraction": float(np.mean(sample == 0.0)),
                "post_normalization_clip_fraction": float(
                    np.mean((standardized < -8.0) | (standardized > 8.0))
                ),
            }
        )
    return records


def fit_relation_normalization(
    tokens: np.ndarray,
    mask: np.ndarray,
    identities: Sequence[Any],
    *,
    normalization_contract: Mapping[str, Any],
    relation_registry: Mapping[str, Any],
    raw_input_schema: Mapping[str, Any],
    hlt_binding_sha256: str,
    source_manifest_sha256: str,
    hlt_model_train_content_sha256: str,
) -> dict[str, Any]:
    """Fit every non-tree relation scaler from locked model-train populations."""

    if normalization_contract.get("fit_split") != "model_train":
        raise ValueError("relation normalization may only fit model_train")
    normalization_sha = validate_normalization_contract(normalization_contract)
    relation_sha = validate_relation_family_registry(relation_registry)
    raw_input_schema_sha = validate_raw_input_schema_contract(raw_input_schema)
    sentinel_policy = _track_sentinel_policy_from_schema(raw_input_schema)
    array = np.asarray(tokens)
    raw_mask = np.asarray(mask)
    if array.dtype != np.float32:
        raise TypeError("HLT normalization tokens must use float32")
    if raw_mask.dtype != np.bool_:
        raise TypeError("HLT normalization mask must use bool")
    valid = raw_mask
    if array.ndim != 3 or int(array.shape[2]) != 14:
        raise ValueError("HLT tokens must have shape [jets,particles,14]")
    if valid.shape != array.shape[:2]:
        raise ValueError("HLT mask shape does not match tokens")
    if len(identities) != int(array.shape[0]):
        raise ValueError("jet identity count does not match HLT tokens")
    if not np.isfinite(array[valid]).all():
        raise FloatingPointError("valid HLT tokens contain NaN or infinity")

    pid = np.asarray(array[:, :, 5:10], dtype=np.float64)
    pid_distance = np.minimum(np.abs(pid), np.abs(pid - 1.0))
    invalid_pid = valid[:, :, None] & (
        (~np.isfinite(pid)) | (pid_distance > GLOBAL_EPSILON)
    )
    if bool(invalid_pid.any()):
        raise ValueError("valid PID flags violate the locked binary tolerance")
    selected_pid = pid >= 0.5
    selected_pid_count = selected_pid.sum(axis=-1)
    multi_hot = valid & (selected_pid_count > 1)
    if bool(multi_hot.any()):
        raise ValueError("model_train contains forbidden multi-hot PID states")
    zero_hot = valid & (selected_pid_count == 0)
    pid_category = np.argmax(selected_pid, axis=-1)
    pid_category = np.where(selected_pid_count == 1, pid_category, 5)
    pid_category_counts = np.bincount(
        pid_category[valid], minlength=6
    ).astype(np.int64)

    locked_charge = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    charge = np.asarray(array[:, :, 4], dtype=np.float64)
    charge_distance = np.abs(charge[:, :, None] - locked_charge)
    charge_state = np.argmin(charge_distance, axis=-1)
    invalid_charge = valid & (
        (~np.isfinite(charge))
        | (np.min(charge_distance, axis=-1) > GLOBAL_EPSILON)
    )
    if bool(invalid_charge.any()):
        raise ValueError("valid charges violate the locked integer tolerance")
    charge_state_counts = np.bincount(
        charge_state[valid], minlength=3
    ).astype(np.int64)

    track_valid = np.zeros_like(valid)
    d0_error_population: list[np.ndarray] = []
    dz_error_population: list[np.ndarray] = []
    for row_index in range(int(array.shape[0])):
        track_valid[row_index] = _track_valid_numpy(
            array[row_index],
            valid[row_index],
            pid_category[row_index],
            sentinel_policy,
        )
        if bool(track_valid[row_index].any()):
            d0_error_population.append(
                np.asarray(
                    array[row_index, track_valid[row_index], 11],
                    dtype=np.float64,
                )
            )
            dz_error_population.append(
                np.asarray(
                    array[row_index, track_valid[row_index], 13],
                    dtype=np.float64,
                )
            )
    if not d0_error_population:
        raise ValueError("model_train contains no valid tracks for floor fitting")
    floor_audits = {
        "d0": _uncertainty_floor_audit(np.concatenate(d0_error_population)),
        "dz": _uncertainty_floor_audit(np.concatenate(dz_error_population)),
    }

    selected = select_normalization_jet_indices(identities)
    selected_keys = [_identity_key(identities[int(index)]) for index in selected]
    pair_keys: list[str] = []
    track_node_keys: list[str] = []
    track_pair_keys: list[str] = []
    density_node_keys: list[str] = []
    pt_chunks: list[np.ndarray] = []
    charge_chunks: list[np.ndarray] = []
    track_node_chunks: list[np.ndarray] = []
    track_pair_chunks: list[np.ndarray] = []
    density_node_chunks: list[np.ndarray] = []
    for row in selected:
        row_index = int(row)
        identity_key = _identity_key(identities[row_index])
        valid_indices = np.flatnonzero(valid[row_index]).tolist()
        pairs = select_normalization_pairs(identities[row_index], valid_indices)
        if pairs:
            pair_keys.extend(
                f"{identity_key}#{query}>{context}"
                for query, context in pairs
            )
            pt_values, charge_values = _step3_pair_values(
                array[row_index], pairs, valid_indices
            )
            pt_chunks.append(pt_values)
            charge_chunks.append(charge_values)

        row_track_valid = track_valid[row_index]
        track_indices = np.flatnonzero(row_track_valid).tolist()
        node_values, node_details = _track_node_values_numpy(
            array[row_index],
            row_track_valid,
            d0_floor=floor_audits["d0"]["floor"],
            dz_floor=floor_audits["dz"]["floor"],
        )
        if node_values.size:
            track_node_chunks.append(node_values)
            track_node_keys.extend(
                f"{identity_key}#track-node:{index}"
                for index in track_indices
            )
        track_pairs = select_normalization_pairs(
            identities[row_index], track_indices
        )
        if track_pairs:
            track_pair_chunks.append(
                _track_pair_values_numpy(
                    array[row_index], track_pairs, node_details
                )
            )
            track_pair_keys.extend(
                f"{identity_key}#track:{query}>{context}"
                for query, context in track_pairs
            )

        density_values = _density_node_values_numpy(
            array[row_index],
            valid[row_index],
            pid_category[row_index],
            row_track_valid,
            node_details,
        )
        if density_values.size:
            density_node_chunks.append(density_values)
            density_node_keys.extend(
                f"{identity_key}#density-node:{index}"
                for index in valid_indices
            )
    if not pt_chunks:
        raise ValueError("selected model_train jets contain no valid directed pairs")
    if not track_node_chunks or not track_pair_chunks:
        raise ValueError("selected model_train jets contain no valid TRACK samples")
    if not density_node_chunks:
        raise ValueError("selected model_train jets contain no valid DENSITY nodes")
    samples = {
        "PT_pair": np.concatenate(pt_chunks, axis=0),
        "CHARGE_pair": np.concatenate(charge_chunks, axis=0),
        "TRACK_node": np.concatenate(track_node_chunks, axis=0),
        "TRACK_compatibility": np.concatenate(track_pair_chunks, axis=0),
        "DENSITY_node": np.concatenate(density_node_chunks, axis=0),
    }
    records = [
        *_fit_records(
            samples["PT_pair"],
            family_id="PT",
            feature_names=PT_ROBUST_FEATURE_NAMES,
            applicability_rule_id="PT_pair",
        ),
        *_fit_records(
            samples["CHARGE_pair"],
            family_id="CHARGE",
            feature_names=CHARGE_ROBUST_FEATURE_NAMES,
            applicability_rule_id="CHARGE_pair",
        ),
        *_fit_records(
            samples["TRACK_node"],
            family_id="TRACK",
            feature_names=TRACK_NODE_CONTINUOUS_NAMES,
            applicability_rule_id="TRACK_node",
        ),
        *_fit_records(
            samples["TRACK_compatibility"],
            family_id="TRACK",
            feature_names=TRACK_COMPATIBILITY_ROBUST_NAMES,
            applicability_rule_id="TRACK_compatibility",
        ),
        *_fit_records(
            samples["DENSITY_node"],
            family_id="DENSITY",
            feature_names=DENSITY_NODE_FEATURE_NAMES,
            applicability_rule_id="DENSITY_node",
        ),
    ]
    sample_sets = {
        "PT_pair": {
            "applicable_count": len(pair_keys),
            "sample_identity_sha256": _identity_sequence_hash(pair_keys),
        },
        "CHARGE_pair": {
            "applicable_count": len(pair_keys),
            "sample_identity_sha256": _identity_sequence_hash(pair_keys),
        },
        "TRACK_node": {
            "applicable_count": len(track_node_keys),
            "sample_identity_sha256": _identity_sequence_hash(track_node_keys),
        },
        "TRACK_compatibility": {
            "applicable_count": len(track_pair_keys),
            "sample_identity_sha256": _identity_sequence_hash(track_pair_keys),
        },
        "DENSITY_node": {
            "applicable_count": len(density_node_keys),
            "sample_identity_sha256": _identity_sequence_hash(density_node_keys),
        },
    }
    artifact = with_content_hash(
        {
            "contract": RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
            "schema_version": 2,
            "fit_split": "model_train",
            "normalization_contract_sha256": normalization_sha,
            "relation_registry_sha256": relation_sha,
            "raw_input_schema_sha256": raw_input_schema_sha,
            "hlt_binding_sha256": require_sha256(
                hlt_binding_sha256, name="hlt_binding_sha256"
            ),
            "source_manifest_sha256": require_sha256(
                source_manifest_sha256, name="source_manifest_sha256"
            ),
            "hlt_model_train_content_sha256": require_sha256(
                hlt_model_train_content_sha256,
                name="hlt_model_train_content_sha256",
            ),
            "fit_families": ["PT", "TRACK", "CHARGE", "DENSITY"],
            "selected_jet_count": int(selected.size),
            "selected_jet_identity_sha256": _identity_sequence_hash(selected_keys),
            "selected_directed_pair_count": len(pair_keys),
            "selected_directed_pair_identity_sha256": _identity_sequence_hash(
                pair_keys
            ),
            "sample_sets": sample_sets,
            "track_sentinel_policy": sentinel_policy,
            "track_uncertainty_floors": floor_audits,
            "quantile_method": NORMALIZATION_QUANTILE_METHOD,
            "float_accumulation_dtype": "float64",
            "input_audit": {
                "valid_particle_count": int(valid.sum()),
                "pid_threshold": 0.5,
                "pid_binary_tolerance": GLOBAL_EPSILON,
                "pid_zero_hot_count": int(zero_hot.sum()),
                "pid_multi_hot_count": 0,
                "pid_category_order": [
                    "charged_hadron",
                    "neutral_hadron",
                    "photon",
                    "electron",
                    "muon",
                    "unknown",
                ],
                "pid_category_counts": [
                    int(value) for value in pid_category_counts
                ],
                "charge_integer_tolerance": GLOBAL_EPSILON,
                "charge_state_order": [-1, 0, 1],
                "charge_state_counts": [
                    int(value) for value in charge_state_counts
                ],
                "track_valid_count": int(track_valid.sum()),
                "track_invalid_count": int(valid.sum() - track_valid.sum()),
                "track_sentinel_policy_source": "locked_raw_input_schema",
            },
            "records": records,
        }
    )
    validate_relation_normalization_artifact(
        artifact,
        normalization_contract_sha256=normalization_sha,
        relation_registry_sha256=relation_sha,
        raw_input_schema_sha256=raw_input_schema_sha,
    )
    return artifact


def fit_step3_relation_normalization(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility name for the complete non-tree relation fitter."""

    return fit_relation_normalization(*args, **kwargs)


def validate_relation_normalization_artifact(
    artifact: Mapping[str, Any],
    *,
    normalization_contract_sha256: str | None = None,
    relation_registry_sha256: str | None = None,
    raw_input_schema_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        artifact, expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT
    )
    if artifact.get("fit_split") != "model_train":
        raise ValueError("relation normalizer was not fitted on model_train")
    if artifact.get("quantile_method") != NORMALIZATION_QUANTILE_METHOD:
        raise ValueError("relation normalizer uses an unsupported quantile method")
    if int(artifact.get("schema_version", 0)) != 2:
        raise ValueError("relation normalizer schema version differs")
    if artifact.get("fit_families") != ["PT", "TRACK", "CHARGE", "DENSITY"]:
        raise ValueError("relation normalizer has unsupported family coverage")
    for field in (
        "normalization_contract_sha256",
        "relation_registry_sha256",
        "raw_input_schema_sha256",
        "hlt_binding_sha256",
        "source_manifest_sha256",
        "hlt_model_train_content_sha256",
        "selected_jet_identity_sha256",
        "selected_directed_pair_identity_sha256",
    ):
        require_sha256(artifact.get(field), name=field)
    selected_jets = int(artifact.get("selected_jet_count", 0))
    selected_pairs = int(artifact.get("selected_directed_pair_count", 0))
    if (
        selected_jets < 1
        or selected_jets > NORMALIZATION_JET_LIMIT
        or selected_pairs < 1
        or selected_pairs
        > selected_jets * NORMALIZATION_PAIR_LIMIT_PER_JET
    ):
        raise ValueError("relation normalizer sample counts violate the lock")
    sample_sets = artifact.get("sample_sets")
    expected_sample_sets = {
        "PT_pair",
        "CHARGE_pair",
        "TRACK_node",
        "TRACK_compatibility",
        "DENSITY_node",
    }
    if not isinstance(sample_sets, Mapping) or set(sample_sets) != expected_sample_sets:
        raise ValueError("relation normalizer sample-set coverage differs")
    for name, sample_set in sample_sets.items():
        if not isinstance(sample_set, Mapping):
            raise ValueError(f"invalid normalization sample set {name}")
        if int(sample_set.get("applicable_count", 0)) < 1:
            raise ValueError(f"empty normalization sample set {name}")
        require_sha256(
            sample_set.get("sample_identity_sha256"),
            name=f"sample_sets.{name}.sample_identity_sha256",
        )
    if (
        int(sample_sets["PT_pair"]["applicable_count"]) != selected_pairs
        or int(sample_sets["CHARGE_pair"]["applicable_count"]) != selected_pairs
        or sample_sets["PT_pair"]["sample_identity_sha256"]
        != artifact.get("selected_directed_pair_identity_sha256")
        or sample_sets["CHARGE_pair"]["sample_identity_sha256"]
        != artifact.get("selected_directed_pair_identity_sha256")
    ):
        raise ValueError("legacy pair sample metadata is inconsistent")
    floor_audits = artifact.get("track_uncertainty_floors")
    if not isinstance(floor_audits, Mapping) or set(floor_audits) != {"d0", "dz"}:
        raise ValueError("relation normalizer lacks uncertainty-floor audits")
    for name, floor_audit in floor_audits.items():
        if not isinstance(floor_audit, Mapping):
            raise ValueError(f"invalid {name} uncertainty-floor audit")
        quantiles = floor_audit.get("quantiles")
        if not isinstance(quantiles, Mapping) or set(quantiles) != {
            "q01", "q05", "q50", "q95", "q99"
        }:
            raise ValueError(f"invalid {name} uncertainty quantiles")
        numeric = [float(value) for value in quantiles.values()]
        floor = float(floor_audit.get("floor", 0.0))
        if (
            int(floor_audit.get("applicable_count", 0)) < 1
            or not np.isfinite(numeric).all()
            or not np.isfinite(floor)
            or floor < GLOBAL_EPSILON
            or floor != max(float(quantiles["q01"]), GLOBAL_EPSILON)
        ):
            raise ValueError(f"invalid {name} uncertainty-floor audit")
    sentinel = artifact.get("track_sentinel_policy")
    if not isinstance(sentinel, Mapping) or set(sentinel) != {
        "d0", "d0err", "dz", "dzerr"
    }:
        raise ValueError("relation normalizer sentinel policy differs")
    for value in sentinel.values():
        if value is not None and not np.isfinite(float(value)):
            raise ValueError("relation normalizer has a nonfinite sentinel")
    audit = artifact.get("input_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("relation normalizer lacks its model_train input audit")
    valid_count = int(audit.get("valid_particle_count", -1))
    pid_counts = list(audit.get("pid_category_counts", ()))
    charge_counts = list(audit.get("charge_state_counts", ()))
    if (
        valid_count < 1
        or len(pid_counts) != 6
        or len(charge_counts) != 3
        or sum(int(value) for value in pid_counts) != valid_count
        or sum(int(value) for value in charge_counts) != valid_count
        or int(audit.get("pid_multi_hot_count", -1)) != 0
    ):
        raise ValueError("relation normalizer input audit is inconsistent")
    if normalization_contract_sha256 is not None and artifact.get(
        "normalization_contract_sha256"
    ) != require_sha256(
        normalization_contract_sha256, name="normalization_contract_sha256"
    ):
        raise ValueError("relation normalizer belongs to another normalization contract")
    if relation_registry_sha256 is not None and artifact.get(
        "relation_registry_sha256"
    ) != require_sha256(
        relation_registry_sha256, name="relation_registry_sha256"
    ):
        raise ValueError("relation normalizer belongs to another relation registry")
    if raw_input_schema_sha256 is not None and artifact.get(
        "raw_input_schema_sha256"
    ) != require_sha256(
        raw_input_schema_sha256, name="raw_input_schema_sha256"
    ):
        raise ValueError("relation normalizer belongs to another raw-input schema")
    expected = {
        ("PT", name, "PT_pair") for name in PT_ROBUST_FEATURE_NAMES
    } | {
        ("CHARGE", name, "CHARGE_pair")
        for name in CHARGE_ROBUST_FEATURE_NAMES
    } | {
        ("TRACK", name, "TRACK_node")
        for name in TRACK_NODE_CONTINUOUS_NAMES
    } | {
        ("TRACK", name, "TRACK_compatibility")
        for name in TRACK_COMPATIBILITY_ROBUST_NAMES
    } | {
        ("DENSITY", name, "DENSITY_node")
        for name in DENSITY_NODE_FEATURE_NAMES
    }
    records = artifact.get("records")
    if not isinstance(records, list):
        raise ValueError("relation normalizer records must be a list")
    actual: set[tuple[str, str, str]] = set()
    for record in records:
        key = (
            str(record.get("family_id")),
            str(record.get("feature_name")),
            str(record.get("applicability_rule_id")),
        )
        if key in actual:
            raise ValueError(f"duplicate normalization record {key}")
        actual.add(key)
        count = int(record.get("applicable_count", 0))
        scale = float(record.get("robust_scale", 0.0))
        numeric = [
            float(record.get(name))
            for name in (
                "median",
                "q25",
                "q75",
                "robust_scale",
                "applicable_zero_fraction",
                "post_normalization_clip_fraction",
            )
        ]
        if (
            count != int(sample_sets[key[2]]["applicable_count"])
            or not np.isfinite(numeric).all()
            or scale < GLOBAL_EPSILON
        ):
            raise ValueError(f"invalid normalization record {key}")
    if actual != expected:
        raise ValueError(
            f"relation normalization coverage differs: missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    return digest


class FeaturewiseNormalizer(_ModuleBase):
    """Torch module applying only registered robust channels."""

    def __init__(
        self,
        *,
        family_id: str,
        raw_feature_names: Sequence[str],
        robust_feature_names: Sequence[str],
        artifact: Mapping[str, Any],
    ) -> None:
        if _torch is None:  # pragma: no cover - environment dependent
            raise ImportError("FeaturewiseNormalizer requires PyTorch")
        super().__init__()
        validate_relation_normalization_artifact(artifact)
        raw = tuple(str(name) for name in raw_feature_names)
        robust = tuple(str(name) for name in robust_feature_names)
        if not set(robust).issubset(raw):
            raise ValueError("robust feature names must be a subset of raw features")
        lookup = {
            (str(record["family_id"]), str(record["feature_name"])): record
            for record in artifact["records"]
        }
        center = np.zeros(len(raw), dtype=np.float32)
        scale = np.ones(len(raw), dtype=np.float32)
        robust_mask = np.zeros(len(raw), dtype=bool)
        for index, name in enumerate(raw):
            if name not in robust:
                continue
            try:
                record = lookup[(family_id, name)]
            except KeyError as exc:
                raise ValueError(
                    f"normalizer lacks {family_id}.{name}"
                ) from exc
            center[index] = float(record["median"])
            scale[index] = float(record["robust_scale"])
            robust_mask[index] = True
        self.family_id = str(family_id)
        self.raw_feature_names = raw
        self.robust_feature_names = robust
        self.artifact_sha256 = str(artifact["content_hash"])
        self.register_buffer("center", _torch.from_numpy(center))
        self.register_buffer("scale", _torch.from_numpy(scale))
        self.register_buffer("robust_mask", _torch.from_numpy(robust_mask))

    def forward(self, values: Any, applicability_mask: Any) -> Any:
        if values.ndim < 3 or int(values.shape[1]) != len(
            self.raw_feature_names
        ):
            raise ValueError("family raw tensor has an incompatible shape")
        if tuple(applicability_mask.shape) != (
            int(values.shape[0]), 1, *map(int, values.shape[2:])
        ):
            raise ValueError("family applicability mask has an incompatible shape")
        safe = _torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        broadcast = (1, -1, *(1 for _ in values.shape[2:]))
        center = self.center.to(dtype=safe.dtype).view(*broadcast)
        scale = self.scale.to(dtype=safe.dtype).view(*broadcast)
        robust = self.robust_mask.view(*broadcast)
        transformed = _torch.clamp((safe - center) / scale, -8.0, 8.0)
        output = _torch.where(robust, transformed, safe)
        return output.masked_fill(~applicability_mask.bool(), 0.0)


__all__ = [
    "CHARGE_RAW_FEATURE_NAMES",
    "CHARGE_ROBUST_FEATURE_NAMES",
    "DENSITY_NODE_FEATURE_NAMES",
    "FeaturewiseNormalizer",
    "GLOBAL_EPSILON",
    "NORMALIZATION_CONTRACT",
    "NORMALIZATION_JET_LIMIT",
    "NORMALIZATION_JET_SALT",
    "NORMALIZATION_PAIR_LIMIT_PER_JET",
    "NORMALIZATION_PAIR_SALT",
    "NORMALIZATION_QUANTILE_METHOD",
    "PT_RAW_FEATURE_NAMES",
    "PT_ROBUST_FEATURE_NAMES",
    "RELATION_NORMALIZATION_ARTIFACT_CONTRACT",
    "TRACK_COMPATIBILITY_FEATURE_NAMES",
    "TRACK_COMPATIBILITY_ROBUST_NAMES",
    "TRACK_NODE_CONTINUOUS_NAMES",
    "build_normalization_contract",
    "fit_relation_normalization",
    "fit_step3_relation_normalization",
    "select_normalization_jet_indices",
    "select_normalization_pairs",
    "validate_relation_normalization_artifact",
    "validate_normalization_contract",
]

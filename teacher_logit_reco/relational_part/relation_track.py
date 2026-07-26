"""Measurement-aware TRACK relation family."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import require_sha256, with_content_hash
from .normalization import FeaturewiseNormalizer, GLOBAL_EPSILON
from .pair_base import require_torch
from .relation_pid_charge import pid_categories
from .relation_pt import valid_pair_mask

try:
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None


TRACK_RELATION_CONTRACT = "relational_part_track_relation_v1"
TRACK_NODE_CONTINUOUS_NAMES = (
    "d0",
    "dz",
    "log_d0_sigma_effective",
    "log_dz_sigma_effective",
    "asinh_d0_significance",
    "asinh_dz_significance",
)
TRACK_NODE_FEATURE_NAMES = (*TRACK_NODE_CONTINUOUS_NAMES, "track_valid")
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
TRACK_VALIDITY_STATE_NAMES = (
    "invalid_invalid",
    "valid_invalid",
    "invalid_valid",
    "valid_valid",
)
TRACK_NODE_ENCODED_DIMENSION = 16
TRACK_EXPLICIT_PAIR_DIMENSION = 17
TRACK_PAIR_INPUT_DIMENSION = 81
TRACK_ENCODED_DIMENSION = 12
TRACK_SENTINEL_FIELDS = ("d0", "d0err", "dz", "dzerr")


if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


def normalize_track_sentinel_policy(
    policy: Mapping[str, Any] | None,
) -> dict[str, float | None]:
    """Normalize an explicit per-field sentinel declaration."""

    if policy is None:
        return {field: None for field in TRACK_SENTINEL_FIELDS}
    if set(policy) != set(TRACK_SENTINEL_FIELDS):
        raise ValueError("track sentinel policy must declare all four fields")
    normalized: dict[str, float | None] = {}
    for field in TRACK_SENTINEL_FIELDS:
        value = policy[field]
        if value is None or value == "no_numeric_sentinel":
            normalized[field] = None
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field} sentinel must be finite")
        normalized[field] = numeric
    return normalized


def track_validity(
    raw_tokens: Any,
    mask: Any,
    *,
    sentinel_policy: Mapping[str, Any] | None = None,
) -> Any:
    """Apply the locked charged-PID, finite, positive-error validity rule."""

    torch = require_torch()
    if (
        not isinstance(raw_tokens, torch.Tensor)
        or raw_tokens.ndim != 3
        or int(raw_tokens.shape[2]) != 14
    ):
        raise ValueError("raw_tokens must have shape [batch,particles,14]")
    if tuple(mask.shape) != (
        int(raw_tokens.shape[0]),
        1,
        int(raw_tokens.shape[1]),
    ):
        raise ValueError("raw token and mask shapes disagree")
    particle_valid = mask[:, 0].bool()
    category = pid_categories(raw_tokens[:, :, 5:10].transpose(1, 2), mask)
    charged_pid = (category == 0) | (category == 3) | (category == 4)
    values = {
        "d0": raw_tokens[:, :, 10],
        "d0err": raw_tokens[:, :, 11],
        "dz": raw_tokens[:, :, 12],
        "dzerr": raw_tokens[:, :, 13],
    }
    measurement_valid = torch.ones_like(particle_valid)
    for value in values.values():
        measurement_valid &= torch.isfinite(value)
    measurement_valid &= values["d0err"] > 0
    measurement_valid &= values["dzerr"] > 0
    sentinels = normalize_track_sentinel_policy(sentinel_policy)
    for field, sentinel in sentinels.items():
        if sentinel is not None:
            measurement_valid &= values[field] != sentinel
    return particle_valid & charged_pid & measurement_valid


def build_track_node_features(
    raw_tokens: Any,
    mask: Any,
    *,
    d0_uncertainty_floor: float,
    dz_uncertainty_floor: float,
    sentinel_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    d0_floor = max(float(d0_uncertainty_floor), GLOBAL_EPSILON)
    dz_floor = max(float(dz_uncertainty_floor), GLOBAL_EPSILON)
    valid = track_validity(
        raw_tokens,
        mask,
        sentinel_policy=sentinel_policy,
    )
    work = raw_tokens.float()
    d0 = torch.nan_to_num(work[:, :, 10], nan=0.0, posinf=0.0, neginf=0.0)
    d0err = torch.nan_to_num(work[:, :, 11], nan=0.0, posinf=0.0, neginf=0.0)
    dz = torch.nan_to_num(work[:, :, 12], nan=0.0, posinf=0.0, neginf=0.0)
    dzerr = torch.nan_to_num(work[:, :, 13], nan=0.0, posinf=0.0, neginf=0.0)
    sigma_d0 = torch.sqrt(d0err.square() + d0_floor**2)
    sigma_dz = torch.sqrt(dzerr.square() + dz_floor**2)
    raw_d0_significance = d0 / sigma_d0
    raw_dz_significance = dz / sigma_dz
    node_continuous = torch.stack(
        (
            d0,
            dz,
            torch.log(sigma_d0.clamp_min(GLOBAL_EPSILON)),
            torch.log(sigma_dz.clamp_min(GLOBAL_EPSILON)),
            torch.asinh(raw_d0_significance),
            torch.asinh(raw_dz_significance),
        ),
        dim=1,
    ).to(dtype=raw_tokens.dtype)
    node_continuous = node_continuous.masked_fill(
        ~valid.unsqueeze(1), 0.0
    )
    return {
        "continuous": node_continuous,
        "track_valid": valid,
        "d0": d0,
        "dz": dz,
        "sigma_d0_effective": sigma_d0,
        "sigma_dz_effective": sigma_dz,
        "raw_d0_significance": raw_d0_significance,
        "raw_dz_significance": raw_dz_significance,
    }


def build_track_compatibility(
    raw_tokens: Any,
    mask: Any,
    *,
    d0_uncertainty_floor: float,
    dz_uncertainty_floor: float,
    sentinel_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    node = build_track_node_features(
        raw_tokens,
        mask,
        d0_uncertainty_floor=d0_uncertainty_floor,
        dz_uncertainty_floor=dz_uncertainty_floor,
        sentinel_policy=sentinel_policy,
    )
    valid = node["track_valid"]
    both_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    particle_pairs = valid_pair_mask(mask)
    validity_index = (
        valid.to(dtype=torch.int64).unsqueeze(-1)
        + valid.to(dtype=torch.int64).unsqueeze(-2) * 2
    )
    validity_one_hot = torch.nn.functional.one_hot(
        validity_index,
        num_classes=4,
    ).permute(0, 3, 1, 2).to(dtype=raw_tokens.dtype)
    validity_one_hot = validity_one_hot.masked_fill(~particle_pairs, 0.0)

    def query(value: Any) -> Any:
        return value.unsqueeze(-1)

    def context(value: Any) -> Any:
        return value.unsqueeze(-2)

    d0_i, d0_j = query(node["d0"]), context(node["d0"])
    dz_i, dz_j = query(node["dz"]), context(node["dz"])
    sd0_i = query(node["sigma_d0_effective"])
    sd0_j = context(node["sigma_d0_effective"])
    sdz_i = query(node["sigma_dz_effective"])
    sdz_j = context(node["sigma_dz_effective"])
    d0_denominator = torch.sqrt(sd0_i.square() + sd0_j.square())
    dz_denominator = torch.sqrt(sdz_i.square() + sdz_j.square())
    delta_d0 = (d0_j - d0_i) / d0_denominator.clamp_min(GLOBAL_EPSILON)
    delta_dz = (dz_j - dz_i) / dz_denominator.clamp_min(GLOBAL_EPSILON)
    chi2 = delta_d0.square() + delta_dz.square()
    s0_i = query(node["continuous"][:, 4])
    s0_j = context(node["continuous"][:, 4])
    sz_i = query(node["continuous"][:, 5])
    sz_j = context(node["continuous"][:, 5])
    eta = raw_tokens[:, :, 1].float()
    phi = raw_tokens[:, :, 2].float()
    delta_eta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
    delta_phi = torch.atan2(
        torch.sin(phi.unsqueeze(-1) - phi.unsqueeze(-2)),
        torch.cos(phi.unsqueeze(-1) - phi.unsqueeze(-2)),
    )
    delta_r = torch.sqrt(
        torch.clamp(delta_eta.square() + delta_phi.square(), min=0.0)
    )
    compatibility = torch.stack(
        (
            torch.log1p(chi2),
            torch.exp(-0.5 * chi2.clamp(max=25.0)),
            torch.minimum(s0_i.abs(), s0_j.abs()),
            torch.maximum(s0_i.abs(), s0_j.abs()),
            torch.minimum(sz_i.abs(), sz_j.abs()),
            torch.maximum(sz_i.abs(), sz_j.abs()),
            s0_i * s0_j,
            sz_i * sz_j,
            delta_d0,
            delta_dz,
            torch.sin(delta_phi),
            torch.cos(delta_phi),
            torch.log(delta_r + GLOBAL_EPSILON),
        ),
        dim=1,
    ).to(dtype=raw_tokens.dtype)
    compatibility = compatibility.masked_fill(
        ~both_valid.unsqueeze(1), 0.0
    )
    return {
        **node,
        "validity_index": validity_index,
        "validity_one_hot": validity_one_hot,
        "both_tracks_valid": both_valid.unsqueeze(1),
        "compatibility": compatibility,
        "chi2": chi2.masked_fill(~both_valid, 0.0),
        "delta_phi": delta_phi,
        "delta_r": delta_r,
    }


class TrackEncoder(_ModuleBase):
    raw_feature_names = (
        *TRACK_NODE_FEATURE_NAMES,
        *TRACK_COMPATIBILITY_FEATURE_NAMES,
    )
    encoded_dimension = TRACK_ENCODED_DIMENSION

    def __init__(self, normalization_artifact: Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        floor_audit = normalization_artifact.get("track_uncertainty_floors")
        if not isinstance(floor_audit, Mapping):
            raise ValueError("relation normalizer lacks track uncertainty floors")
        self.uncertainty_floor_audit = {
            name: {
                "applicable_count": int(floor_audit[name]["applicable_count"]),
                "quantiles": {
                    key: float(value)
                    for key, value in floor_audit[name]["quantiles"].items()
                },
                "floor": float(floor_audit[name]["floor"]),
            }
            for name in ("d0", "dz")
        }
        self.d0_uncertainty_floor = float(floor_audit["d0"]["floor"])
        self.dz_uncertainty_floor = float(floor_audit["dz"]["floor"])
        self.sentinel_policy = normalize_track_sentinel_policy(
            normalization_artifact.get("track_sentinel_policy")
        )
        self.node_normalizer = FeaturewiseNormalizer(
            family_id="TRACK",
            raw_feature_names=TRACK_NODE_CONTINUOUS_NAMES,
            robust_feature_names=TRACK_NODE_CONTINUOUS_NAMES,
            artifact=normalization_artifact,
        )
        self.compatibility_normalizer = FeaturewiseNormalizer(
            family_id="TRACK",
            raw_feature_names=TRACK_COMPATIBILITY_FEATURE_NAMES,
            robust_feature_names=TRACK_COMPATIBILITY_ROBUST_NAMES,
            artifact=normalization_artifact,
        )
        self.track_encoder = torch.nn.Sequential(
            torch.nn.Linear(7, 32),
            torch.nn.GELU(),
            torch.nn.RMSNorm(32, eps=GLOBAL_EPSILON),
            torch.nn.Linear(32, TRACK_NODE_ENCODED_DIMENSION),
        )
        self.pair_encoder = torch.nn.Sequential(
            torch.nn.Linear(TRACK_PAIR_INPUT_DIMENSION, 48),
            torch.nn.GELU(),
            torch.nn.RMSNorm(48, eps=GLOBAL_EPSILON),
            torch.nn.Linear(48, TRACK_ENCODED_DIMENSION),
        )

    def forward(
        self,
        raw_tokens: Any,
        mask: Any,
        *,
        return_details: bool = False,
    ) -> Any:
        torch = require_torch()
        details = build_track_compatibility(
            raw_tokens,
            mask,
            d0_uncertainty_floor=self.d0_uncertainty_floor,
            dz_uncertainty_floor=self.dz_uncertainty_floor,
            sentinel_policy=self.sentinel_policy,
        )
        track_mask = details["track_valid"].unsqueeze(1)
        normalized_node = self.node_normalizer(
            details["continuous"],
            track_mask,
        )
        node_input = torch.cat(
            (
                normalized_node,
                details["track_valid"].unsqueeze(1).to(raw_tokens.dtype),
            ),
            dim=1,
        )
        node_encoded = self.track_encoder(node_input.transpose(1, 2))
        node_encoded = node_encoded.transpose(1, 2).contiguous()
        particle_valid = mask.bool()
        node_encoded = node_encoded.masked_fill(~particle_valid, 0.0)
        normalized_compatibility = self.compatibility_normalizer(
            details["compatibility"],
            details["both_tracks_valid"],
        )
        explicit = torch.cat(
            (details["validity_one_hot"], normalized_compatibility),
            dim=1,
        )
        g_i = node_encoded.unsqueeze(-1).expand(-1, -1, -1, node_encoded.shape[-1])
        g_j = node_encoded.unsqueeze(-2).expand(-1, -1, node_encoded.shape[-1], -1)
        pair_input = torch.cat(
            (g_i, g_j, g_j - g_i, g_i * g_j, explicit),
            dim=1,
        )
        if int(pair_input.shape[1]) != TRACK_PAIR_INPUT_DIMENSION:
            raise RuntimeError("TRACK pair input dimension drifted")
        encoded = self.pair_encoder(pair_input.permute(0, 2, 3, 1))
        encoded = encoded.permute(0, 3, 1, 2).contiguous()
        encoded = encoded.masked_fill(~valid_pair_mask(mask), 0.0)
        if return_details:
            return {
                **details,
                "normalized_node": normalized_node,
                "node_encoder_input": node_input,
                "node_encoded": node_encoded,
                "normalized_compatibility": normalized_compatibility,
                "explicit_pair": explicit,
                "pair_encoder_input": pair_input,
                "encoded": encoded,
                "pair_mask": valid_pair_mask(mask),
            }
        return encoded

    def diagnostics(self, raw_tokens: Any, mask: Any) -> dict[str, Any]:
        torch = require_torch()
        details = build_track_compatibility(
            raw_tokens,
            mask,
            d0_uncertainty_floor=self.d0_uncertainty_floor,
            dz_uncertainty_floor=self.dz_uncertainty_floor,
            sentinel_policy=self.sentinel_policy,
        )
        pair_mask = valid_pair_mask(mask)[:, 0]
        state_counts = torch.bincount(
            details["validity_index"].masked_select(pair_mask),
            minlength=4,
        )
        both = details["both_tracks_valid"][:, 0]
        return {
            "validity_state_order": list(TRACK_VALIDITY_STATE_NAMES),
            "validity_state_counts": [int(value) for value in state_counts.cpu()],
            "validity_state_fractions": [
                float(value)
                for value in (
                    state_counts.float() / state_counts.sum().clamp_min(1)
                ).cpu()
            ],
            "track_valid_count": int(details["track_valid"].sum().cpu()),
            "d0_uncertainty_floor": self.d0_uncertainty_floor,
            "dz_uncertainty_floor": self.dz_uncertainty_floor,
            "uncertainty_floor_audit": self.uncertainty_floor_audit,
            "chi2_mean": (
                float(details["chi2"].masked_select(both).mean().cpu())
                if bool(both.any())
                else 0.0
            ),
            "raw_d0_mean": (
                float(details["d0"].masked_select(details["track_valid"]).mean().cpu())
                if bool(details["track_valid"].any())
                else 0.0
            ),
            "raw_dz_mean": (
                float(details["dz"].masked_select(details["track_valid"]).mean().cpu())
                if bool(details["track_valid"].any())
                else 0.0
            ),
            "raw_absolute_d0_significance_mean": (
                float(
                    details["raw_d0_significance"]
                    .abs()
                    .masked_select(details["track_valid"])
                    .mean()
                    .cpu()
                )
                if bool(details["track_valid"].any())
                else 0.0
            ),
            "raw_absolute_dz_significance_mean": (
                float(
                    details["raw_dz_significance"]
                    .abs()
                    .masked_select(details["track_valid"])
                    .mean()
                    .cpu()
                )
                if bool(details["track_valid"].any())
                else 0.0
            ),
            "asinh_absolute_d0_significance_mean": (
                float(
                    details["continuous"][:, 4]
                    .abs()
                    .masked_select(details["track_valid"])
                    .mean()
                    .cpu()
                )
                if bool(details["track_valid"].any())
                else 0.0
            ),
            "asinh_absolute_dz_significance_mean": (
                float(
                    details["continuous"][:, 5]
                    .abs()
                    .masked_select(details["track_valid"])
                    .mean()
                    .cpu()
                )
                if bool(details["track_valid"].any())
                else 0.0
            ),
            "sentinel_policy": dict(self.sentinel_policy),
        }


def build_track_relation_contract(
    *,
    relation_registry_sha256: str,
    relation_normalization_sha256: str,
    raw_input_schema_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": TRACK_RELATION_CONTRACT,
            "schema_version": 1,
            "relation_registry_sha256": require_sha256(
                relation_registry_sha256, name="relation_registry_sha256"
            ),
            "relation_normalization_sha256": require_sha256(
                relation_normalization_sha256,
                name="relation_normalization_sha256",
            ),
            "raw_input_schema_sha256": require_sha256(
                raw_input_schema_sha256, name="raw_input_schema_sha256"
            ),
            "family_id": "TRACK",
            "node_feature_names": list(TRACK_NODE_FEATURE_NAMES),
            "node_continuous_normalization": "TRACK_node",
            "node_encoder": [
                "Linear(7,32)",
                "GELU",
                "RMSNorm(32,eps=1e-6)",
                "Linear(32,16)",
            ],
            "siamese_endpoint_weights_shared": True,
            "validity_state_order": list(TRACK_VALIDITY_STATE_NAMES),
            "compatibility_feature_names": list(
                TRACK_COMPATIBILITY_FEATURE_NAMES
            ),
            "compatibility_robust_feature_names": list(
                TRACK_COMPATIBILITY_ROBUST_NAMES
            ),
            "delta_phi_direction": "wrapped_query_minus_context",
            "normalized_displacement_direction": "context_minus_query",
            "chi2_clip_for_exponential_only": 25.0,
            "pair_encoder_input": (
                "[g_i,g_j,g_j-g_i,g_i*g_j,validity_one_hot,compatibility]"
            ),
            "pair_input_dimension": TRACK_PAIR_INPUT_DIMENSION,
            "pair_encoder": [
                "Linear(81,48)",
                "GELU",
                "RMSNorm(48,eps=1e-6)",
                "Linear(48,12)",
            ],
            "encoded_dimension": TRACK_ENCODED_DIMENSION,
            "invalid_compatibility_policy": "zero_unless_both_tracks_valid",
            "dropout": 0.0,
        }
    )


__all__ = [
    "TRACK_COMPATIBILITY_FEATURE_NAMES",
    "TRACK_COMPATIBILITY_ROBUST_NAMES",
    "TRACK_ENCODED_DIMENSION",
    "TRACK_EXPLICIT_PAIR_DIMENSION",
    "TRACK_NODE_CONTINUOUS_NAMES",
    "TRACK_NODE_ENCODED_DIMENSION",
    "TRACK_NODE_FEATURE_NAMES",
    "TRACK_PAIR_INPUT_DIMENSION",
    "TRACK_RELATION_CONTRACT",
    "TRACK_SENTINEL_FIELDS",
    "TRACK_VALIDITY_STATE_NAMES",
    "TrackEncoder",
    "build_track_compatibility",
    "build_track_node_features",
    "build_track_relation_contract",
    "normalize_track_sentinel_policy",
    "track_validity",
]

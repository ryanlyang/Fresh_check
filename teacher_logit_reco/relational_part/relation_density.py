"""Multiscale local-activity DENSITY relation family."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import require_sha256, with_content_hash
from .normalization import FeaturewiseNormalizer, GLOBAL_EPSILON
from .pair_base import require_torch
from .relation_pid_charge import pid_categories
from .relation_pt import valid_pair_mask
from .relation_track import (
    normalize_track_sentinel_policy,
    track_validity,
)

try:
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None


DENSITY_RELATION_CONTRACT = "relational_part_density_relation_v1"
DENSITY_ANNULUS_BOUNDARIES = (0.0, 0.05, 0.10, 0.20, 0.40)
DENSITY_SMOOTH_RADIUS_CENTERS = (0.025, 0.071, 0.141, 0.283)
DENSITY_SMOOTH_LOG_RADIUS_SIGMA = 0.45
DENSITY_MAX_CONSTITUENTS = 128
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
DENSITY_NODE_DIMENSION = 22
DENSITY_PAIR_INPUT_DIMENSION = 66
DENSITY_ENCODED_DIMENSION = 12


if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


def build_density_node_features(
    raw_tokens: Any,
    mask: Any,
    *,
    d0_uncertainty_floor: float,
    dz_uncertainty_floor: float,
    sentinel_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    valid = mask[:, 0].bool()
    work = raw_tokens.float()
    if not bool(torch.isfinite(work[valid]).all()):
        raise FloatingPointError("valid raw tokens contain NaN or infinity")
    pt = work[:, :, 0].masked_fill(~valid, 0.0)
    if bool((pt < 0).any()):
        raise ValueError("valid transverse momentum must be nonnegative")
    eta = torch.nan_to_num(work[:, :, 1], nan=0.0, posinf=0.0, neginf=0.0)
    phi = torch.nan_to_num(work[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0)
    eta = eta.masked_fill(~valid, 0.0)
    phi = phi.masked_fill(~valid, 0.0)
    delta_eta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
    delta_phi = torch.atan2(
        torch.sin(phi.unsqueeze(-1) - phi.unsqueeze(-2)),
        torch.cos(phi.unsqueeze(-1) - phi.unsqueeze(-2)),
    )
    delta_r = torch.sqrt(
        torch.clamp(delta_eta.square() + delta_phi.square(), min=0.0)
    )
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    length = int(raw_tokens.shape[1])
    not_self = ~torch.eye(length, dtype=torch.bool, device=raw_tokens.device)
    neighbors = pair_valid & not_self.unsqueeze(0)
    jet_pt = pt.sum(dim=-1, keepdim=True)
    features: list[Any] = []
    annulus_masks: list[Any] = []
    for index, (lower, upper) in enumerate(
        zip(DENSITY_ANNULUS_BOUNDARIES[:-1], DENSITY_ANNULUS_BOUNDARIES[1:])
    ):
        radial = delta_r <= upper
        if index > 0:
            radial &= delta_r > lower
        annulus = neighbors & radial
        annulus_masks.append(annulus)
        count = annulus.sum(dim=-1).float()
        count = torch.log1p(count) / math.log1p(DENSITY_MAX_CONSTITUENTS)
        pt_sum = (annulus.to(pt.dtype) * pt.unsqueeze(-2)).sum(dim=-1)
        features.extend(
            (count, pt_sum / (jet_pt + GLOBAL_EPSILON))
        )

    valid_count = valid.sum(dim=-1, keepdim=True)
    neighbor_denominator = (valid_count - 1).clamp_min(1).float()
    smooth_responses: list[Any] = []
    log_radius = torch.log(delta_r + GLOBAL_EPSILON)
    for center in DENSITY_SMOOTH_RADIUS_CENTERS:
        kernel = torch.exp(
            -(
                log_radius - math.log(center)
            ).square()
            / (2.0 * DENSITY_SMOOTH_LOG_RADIUS_SIGMA**2)
        )
        kernel = kernel * neighbors.to(kernel.dtype)
        count_response = kernel.sum(dim=-1) / neighbor_denominator
        pt_response = (
            kernel * pt.unsqueeze(-2)
        ).sum(dim=-1) / (jet_pt + GLOBAL_EPSILON)
        one_or_fewer = valid_count <= 1
        count_response = count_response.masked_fill(one_or_fewer, 0.0)
        pt_response = pt_response.masked_fill(one_or_fewer, 0.0)
        smooth_responses.extend((count_response, pt_response))
    features.extend(smooth_responses)

    category = pid_categories(raw_tokens[:, :, 5:10].transpose(1, 2), mask)
    charged = (category == 0) | (category == 3) | (category == 4)
    neutral_hadron = category == 1
    photon = category == 2
    local = neighbors & (delta_r <= 0.20)
    local_pt = (local.to(pt.dtype) * pt.unsqueeze(-2)).sum(dim=-1)

    def composition_fraction(category_mask: Any) -> Any:
        numerator = (
            local.to(pt.dtype)
            * category_mask.unsqueeze(-2).to(pt.dtype)
            * pt.unsqueeze(-2)
        ).sum(dim=-1)
        return torch.where(
            local_pt > 0,
            numerator / local_pt.clamp_min(GLOBAL_EPSILON),
            torch.zeros_like(numerator),
        )

    track_valid = track_validity(
        raw_tokens,
        mask,
        sentinel_policy=normalize_track_sentinel_policy(sentinel_policy),
    )
    d0err = torch.nan_to_num(work[:, :, 11], nan=0.0)
    dzerr = torch.nan_to_num(work[:, :, 13], nan=0.0)
    sigma_d0 = torch.sqrt(
        d0err.square() + max(float(d0_uncertainty_floor), GLOBAL_EPSILON) ** 2
    )
    sigma_dz = torch.sqrt(
        dzerr.square() + max(float(dz_uncertainty_floor), GLOBAL_EPSILON) ** 2
    )
    displaced = track_valid & (
        (work[:, :, 10].abs() / sigma_d0 >= 2.0)
        | (work[:, :, 12].abs() / sigma_dz >= 2.0)
    )
    features.extend(
        (
            composition_fraction(charged),
            composition_fraction(neutral_hadron),
            composition_fraction(photon),
            composition_fraction(displaced),
        )
    )

    within_040 = neighbors & (delta_r <= 0.40)
    neighbor_fraction = (
        within_040.sum(dim=-1).float() / neighbor_denominator
    )
    neighbor_fraction = neighbor_fraction.masked_fill(valid_count <= 1, 0.0)
    self_denominator = pt + local_pt
    self_share = torch.where(
        self_denominator > 0,
        pt / self_denominator.clamp_min(GLOBAL_EPSILON),
        torch.zeros_like(pt),
    )
    features.extend((neighbor_fraction, self_share))
    descriptor = torch.stack(features, dim=1).to(dtype=raw_tokens.dtype)
    if int(descriptor.shape[1]) != DENSITY_NODE_DIMENSION:
        raise RuntimeError("DENSITY node dimension drifted")
    descriptor = descriptor.masked_fill(~mask.bool(), 0.0)
    return {
        "descriptor": descriptor,
        "delta_r": delta_r,
        "annulus_masks": torch.stack(annulus_masks, dim=1),
        "neighbor_mask": neighbors.unsqueeze(1),
        "local_mask_R0p20": local.unsqueeze(1),
        "track_valid": track_valid,
        "displaced_track": displaced,
    }


class DensityEncoder(_ModuleBase):
    raw_feature_names = DENSITY_NODE_FEATURE_NAMES
    encoded_dimension = DENSITY_ENCODED_DIMENSION

    def __init__(self, normalization_artifact: Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        floor_audit = normalization_artifact.get("track_uncertainty_floors")
        if not isinstance(floor_audit, Mapping):
            raise ValueError("relation normalizer lacks track uncertainty floors")
        self.d0_uncertainty_floor = float(floor_audit["d0"]["floor"])
        self.dz_uncertainty_floor = float(floor_audit["dz"]["floor"])
        self.sentinel_policy = normalize_track_sentinel_policy(
            normalization_artifact.get("track_sentinel_policy")
        )
        self.node_normalizer = FeaturewiseNormalizer(
            family_id="DENSITY",
            raw_feature_names=DENSITY_NODE_FEATURE_NAMES,
            robust_feature_names=DENSITY_NODE_FEATURE_NAMES,
            artifact=normalization_artifact,
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(DENSITY_PAIR_INPUT_DIMENSION, 32),
            torch.nn.GELU(),
            torch.nn.RMSNorm(32, eps=GLOBAL_EPSILON),
            torch.nn.Linear(32, DENSITY_ENCODED_DIMENSION),
        )

    def forward(
        self,
        raw_tokens: Any,
        mask: Any,
        *,
        return_details: bool = False,
    ) -> Any:
        torch = require_torch()
        details = build_density_node_features(
            raw_tokens,
            mask,
            d0_uncertainty_floor=self.d0_uncertainty_floor,
            dz_uncertainty_floor=self.dz_uncertainty_floor,
            sentinel_policy=self.sentinel_policy,
        )
        normalized = self.node_normalizer(details["descriptor"], mask.bool())
        d_i = normalized.unsqueeze(-1).expand(-1, -1, -1, normalized.shape[-1])
        d_j = normalized.unsqueeze(-2).expand(-1, -1, normalized.shape[-1], -1)
        pair_input = torch.cat((d_i, d_j, d_j - d_i), dim=1)
        encoded = self.encoder(pair_input.permute(0, 2, 3, 1))
        encoded = encoded.permute(0, 3, 1, 2).contiguous()
        encoded = encoded.masked_fill(~valid_pair_mask(mask), 0.0)
        if return_details:
            return {
                **details,
                "normalized_descriptor": normalized,
                "pair_encoder_input": pair_input,
                "encoded": encoded,
                "pair_mask": valid_pair_mask(mask),
            }
        return encoded

    def diagnostics(self, raw_tokens: Any, mask: Any) -> dict[str, Any]:
        details = build_density_node_features(
            raw_tokens,
            mask,
            d0_uncertainty_floor=self.d0_uncertainty_floor,
            dz_uncertainty_floor=self.dz_uncertainty_floor,
            sentinel_policy=self.sentinel_policy,
        )
        valid = mask[:, 0].bool()
        descriptor = details["descriptor"]
        valid_count = int(valid.sum().cpu())
        annulus_counts = [
            int(
                (
                    details["annulus_masks"][:, index].sum(dim=-1) > 0
                )[valid].sum().cpu()
            )
            for index in range(4)
        ]
        neighbor_sum = float(
            descriptor[:, 20].masked_select(valid).sum().cpu()
        )
        self_share_sum = float(
            descriptor[:, 21].masked_select(valid).sum().cpu()
        )
        return {
            "valid_particle_count": valid_count,
            "annulus_nonempty_fractions": [
                0.0 if valid_count == 0 else count / valid_count
                for count in annulus_counts
            ],
            "mean_neighbor_fraction_R0p40": (
                0.0 if valid_count == 0 else neighbor_sum / valid_count
            ),
            "mean_self_share_R0p20": (
                0.0 if valid_count == 0 else self_share_sum / valid_count
            ),
            "_population_statistics": {
                "valid_particle_count": {
                    "kind": "sum",
                    "value": valid_count,
                },
                "annulus_nonempty_fractions": {
                    "kind": "ratio",
                    "numerator": annulus_counts,
                    "denominator": [valid_count] * 4,
                },
                "mean_neighbor_fraction_R0p40": {
                    "kind": "ratio",
                    "numerator": neighbor_sum,
                    "denominator": valid_count,
                },
                "mean_self_share_R0p20": {
                    "kind": "ratio",
                    "numerator": self_share_sum,
                    "denominator": valid_count,
                },
            },
        }


def build_density_relation_contract(
    *,
    relation_registry_sha256: str,
    relation_normalization_sha256: str,
    track_relation_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": DENSITY_RELATION_CONTRACT,
            "schema_version": 1,
            "relation_registry_sha256": require_sha256(
                relation_registry_sha256, name="relation_registry_sha256"
            ),
            "relation_normalization_sha256": require_sha256(
                relation_normalization_sha256,
                name="relation_normalization_sha256",
            ),
            "track_relation_sha256": require_sha256(
                track_relation_sha256, name="track_relation_sha256"
            ),
            "family_id": "DENSITY",
            "annulus_boundaries": list(DENSITY_ANNULUS_BOUNDARIES),
            "annulus_endpoint_rule": (
                "[0,0.05],(0.05,0.10],(0.10,0.20],(0.20,0.40]"
            ),
            "smooth_radius_centers": list(DENSITY_SMOOTH_RADIUS_CENTERS),
            "smooth_log_radius_sigma": DENSITY_SMOOTH_LOG_RADIUS_SIGMA,
            "smooth_hidden_cutoff": None,
            "max_constituents_count_reference": DENSITY_MAX_CONSTITUENTS,
            "node_feature_names": list(DENSITY_NODE_FEATURE_NAMES),
            "node_normalization": "fit_once_per_valid_particle",
            "pair_input": "[d_i,d_j,d_j-d_i]",
            "pair_input_dimension": DENSITY_PAIR_INPUT_DIMENSION,
            "encoder": [
                "Linear(66,32)",
                "GELU",
                "RMSNorm(32,eps=1e-6)",
                "Linear(32,12)",
            ],
            "encoded_dimension": DENSITY_ENCODED_DIMENSION,
            "self_excluded_from_neighbor_summaries": True,
            "dropout": 0.0,
        }
    )


__all__ = [
    "DENSITY_ANNULUS_BOUNDARIES",
    "DENSITY_ENCODED_DIMENSION",
    "DENSITY_MAX_CONSTITUENTS",
    "DENSITY_NODE_DIMENSION",
    "DENSITY_NODE_FEATURE_NAMES",
    "DENSITY_PAIR_INPUT_DIMENSION",
    "DENSITY_RELATION_CONTRACT",
    "DENSITY_SMOOTH_LOG_RADIUS_SIGMA",
    "DENSITY_SMOOTH_RADIUS_CENTERS",
    "DensityEncoder",
    "build_density_node_features",
    "build_density_relation_contract",
]

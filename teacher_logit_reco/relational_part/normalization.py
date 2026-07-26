"""Immutable normalization specification for future relation implementations."""

from __future__ import annotations

from typing import Any

from .contracts import require_sha256, with_content_hash


NORMALIZATION_CONTRACT = "relational_part_normalization_contract_v1"
GLOBAL_EPSILON = 1.0e-6


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
            "schema_version": 1,
            "split_binding_sha256": split_binding_sha256,
            "fit_split": "model_train",
            "global_epsilon": GLOBAL_EPSILON,
            "selection": {
                "jet_count": 50_000,
                "jet_policy": "lowest_salted_identity_sha256",
                "jet_salt": "relational_part_normalization_jets_v1",
                "maximum_directed_pairs_per_jet": 64,
                "pair_policy": "lowest_salted_directed_pair_sha256",
                "pair_salt": "relational_part_normalization_pairs_v1",
                "node_features_counted_once_per_particle": True,
            },
            "continuous_transform": {
                "center": "median",
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
                "TRACK_node": "valid_tracks_counted_once",
                "TRACK_compatibility": "valid_valid_distinct_or_diagonal_as_declared",
                "DENSITY_node": "valid_particles_counted_once",
                "REGION_node": "valid_particles_per_defined_resolution_counted_once",
                "REGION_pair": "valid_directed_pairs_per_defined_resolution",
                "REGION_merge": "valid_distinct_directed_pairs_only",
            },
            "invalid_policy": {
                "excluded_from_fit": True,
                "safe_placeholder_before_numeric_transform": True,
                "zero_after_normalization": True,
                "masked_after_learned_encoder": True,
            },
            "track_uncertainty_floor": {
                "fit_population": "all_valid_positive_model_train_track_errors",
                "d0_formula": "max(q01(valid_positive_d0err),1e-6)",
                "dz_formula": "max(q01(valid_positive_dzerr),1e-6)",
                "recorded_quantiles": [0.01, 0.05, 0.5, 0.95, 0.99],
            },
            "required_channel_record_fields": [
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


__all__ = [
    "GLOBAL_EPSILON",
    "NORMALIZATION_CONTRACT",
    "build_normalization_contract",
]

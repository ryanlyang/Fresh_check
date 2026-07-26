"""Training-independent registries for relational Particle Transformer runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import require_sha256, validate_content_hash, with_content_hash


RELATION_FAMILY_REGISTRY_CONTRACT = "relational_part_relation_family_registry_v1"
SCREENING_REGISTRY_CONTRACT = "relational_part_screening_registry_v1"
CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT = (
    "relational_part_confirmation_architecture_registry_v1"
)
SEMANTIC_CONTROL_REGISTRY_CONTRACT = "relational_part_semantic_control_registry_v1"

CANONICAL_FAMILY_ORDER = ("base4", "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION")
CONFIGURATION_ROLES = (
    "reference_baseline",
    "capacity_control",
    "architecture_control",
    "scientific_finalist",
    "semantic_control",
)


def build_relation_family_registry() -> dict[str, Any]:
    families = [
        {
            "family_id": "base4",
            "kind": "standard_part_pair_features",
            "feature_source": "Weaver.pairwise_lv_fts(num_outputs=4)",
            "raw_dimension": 4,
            "encoded_dimension": None,
            "requires_tree": False,
            "normalization": "reference_Weaver_contract",
        },
        {
            "family_id": "PT",
            "kind": "directional_continuous_pair",
            "raw_feature_names": [
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
            ],
            "raw_dimension": 10,
            "encoded_dimension": 8,
            "requires_tree": False,
            "normalization": "robust_featurewise_model_train",
        },
        {
            "family_id": "TRACK",
            "kind": "siamese_track_and_directed_compatibility",
            "node_feature_names": [
                "d0",
                "dz",
                "log_d0_sigma_effective",
                "log_dz_sigma_effective",
                "asinh_d0_significance",
                "asinh_dz_significance",
                "track_valid",
            ],
            "explicit_pair_feature_groups": {
                "validity_one_hot": 4,
                "chi2_and_compatibility": 2,
                "minmax_significance": 4,
                "significance_products": 2,
                "signed_normalized_displacement_differences": 2,
                "angular": 3,
            },
            "node_raw_dimension": 7,
            "node_encoded_dimension": 16,
            "explicit_pair_dimension": 17,
            "pair_encoder_input_dimension": 81,
            "encoded_dimension": 12,
            "requires_tree": False,
            "normalization": "applicability_aware_robust_model_train",
        },
        {
            "family_id": "PID",
            "kind": "factorized_directional_categorical_pair",
            "category_order": [
                "charged_hadron",
                "neutral_hadron",
                "photon",
                "electron",
                "muon",
                "unknown",
            ],
            "category_count": 6,
            "directed_pair_state_count": 36,
            "encoded_dimension": 8,
            "requires_tree": False,
            "normalization": "none_categorical",
        },
        {
            "family_id": "CHARGE",
            "kind": "directional_mixed_pair",
            "continuous_feature_names": [
                "query_charge",
                "context_charge",
                "charge_product",
                "half_absolute_charge_difference",
                "both_neutral",
                "exactly_one_charged",
                "same_nonzero_sign",
                "opposite_nonzero_sign",
            ],
            "directed_charge_state_count": 9,
            "continuous_dimension": 8,
            "categorical_embedding_dimension": 4,
            "family_encoder_input_dimension": 12,
            "encoded_dimension": 6,
            "requires_tree": False,
            "normalization": "continuous_only_robust_model_train",
        },
        {
            "family_id": "DENSITY",
            "kind": "multiscale_local_node_to_pair",
            "annulus_boundaries": [0.0, 0.05, 0.1, 0.2, 0.4],
            "smooth_radius_centers": [0.025, 0.071, 0.141, 0.283],
            "smooth_log_radius_sigma": 0.45,
            "node_feature_groups": {
                "hard_annular_count_and_pt": 8,
                "smooth_kernel_count_and_pt": 8,
                "local_pid_and_displacement_composition": 4,
                "neighbor_fraction_and_self_share": 2,
            },
            "node_dimension": 22,
            "raw_pair_dimension": 66,
            "encoded_dimension": 12,
            "requires_tree": False,
            "normalization": "node_once_robust_model_train",
        },
        {
            "family_id": "REGION",
            "kind": "beam_free_exclusive_angular_tree_pair",
            "raw_feature_groups": {
                "same_cluster_indicators_K2_K4_K8": 3,
                "endpoint_cluster_descriptors": 36,
                "lca_and_merge_features": 2,
            },
            "raw_dimension": 41,
            "encoded_dimension": 12,
            "requires_tree": True,
            "tree_contract": "relational_ca_tree_v1",
            "exclusive_resolutions": [2, 4, 8],
            "normalization": "node_and_pair_applicability_aware_model_train",
        },
    ]
    if [row["family_id"] for row in families] != list(CANONICAL_FAMILY_ORDER):
        raise AssertionError("relation registry violates canonical family order")
    return with_content_hash(
        {
            "contract": RELATION_FAMILY_REGISTRY_CONTRACT,
            "schema_version": 1,
            "canonical_family_order": list(CANONICAL_FAMILY_ORDER),
            "families": families,
            "invalid_pair_policy": "zero_after_every_learned_encoder",
            "query_context_direction": "i_is_query_j_is_context",
            "global_epsilon": 1.0e-6,
        }
    )


@dataclass(frozen=True)
class _Run:
    run_id: str
    relations: tuple[str, ...]
    purpose: str
    role: str = "scientific_finalist"
    relation_input_mode: str = "active"
    architecture: str = "shared_bias"

    def to_dict(self) -> dict[str, Any]:
        eligible = self.role == "scientific_finalist"
        return {
            "run_id": self.run_id,
            "enabled_relations": ["base4", *self.relations],
            "new_relation_families": list(self.relations),
            "purpose": self.purpose,
            "configuration_role": self.role,
            "relational_selection_eligible": eligible,
            "relation_input_mode": self.relation_input_mode,
            "attention_architecture": self.architecture,
            "initialization": "from_scratch",
        }


_SCREENING_RUNS = (
    _Run("RPT_BASE", (), "exact matched HLT ParT", "reference_baseline"),
    _Run(
        "RPT_BASE_WIDE_MAX",
        (),
        "active incremental parameter-matched capacity control",
        "capacity_control",
        architecture="wide_pair_encoder",
    ),
    _Run(
        "RPT_FULL_ZERO_REL",
        ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"),
        "exact full-relation tensor-shape control with zero new channels",
        "capacity_control",
        relation_input_mode="forced_zero",
    ),
    _Run("RPT_PT", ("PT",), "single PT relation"),
    _Run("RPT_TRACK", ("TRACK",), "single TRACK relation"),
    _Run("RPT_PID", ("PID",), "single PID relation"),
    _Run("RPT_CHARGE", ("CHARGE",), "single CHARGE relation"),
    _Run("RPT_DENSITY", ("DENSITY",), "single DENSITY relation"),
    _Run("RPT_REGION", ("REGION",), "single REGION relation"),
    _Run("RPT_PT_TRACK", ("PT", "TRACK"), "high-value pair"),
    _Run("RPT_TRACK_PID", ("TRACK", "PID"), "high-value pair"),
    _Run("RPT_TRACK_CHARGE", ("TRACK", "CHARGE"), "high-value pair"),
    _Run("RPT_PID_CHARGE", ("PID", "CHARGE"), "high-value pair"),
    _Run("RPT_PT_DENSITY", ("PT", "DENSITY"), "high-value pair"),
    _Run("RPT_PT_REGION", ("PT", "REGION"), "high-value pair"),
    _Run("RPT_TRACK_REGION", ("TRACK", "REGION"), "high-value pair"),
    _Run(
        "RPT_TRACK_PID_CHARGE",
        ("TRACK", "PID", "CHARGE"),
        "high-potential higher-order combination",
    ),
    _Run(
        "RPT_PT_TRACK_DENSITY",
        ("PT", "TRACK", "DENSITY"),
        "high-potential higher-order combination",
    ),
    _Run(
        "RPT_PT_TRACK_REGION",
        ("PT", "TRACK", "REGION"),
        "high-potential higher-order combination",
    ),
    _Run(
        "RPT_PT_TRACK_PID_CHARGE",
        ("PT", "TRACK", "PID", "CHARGE"),
        "high-potential higher-order combination",
    ),
    _Run(
        "RPT_FULL_ALL",
        ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"),
        "all relation families",
    ),
)


def _validate_run_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    ids = [str(row.get("run_id")) for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("run registry contains duplicate run IDs")
    family_index = {name: index for index, name in enumerate(CANONICAL_FAMILY_ORDER)}
    for row in output:
        role = row.get("configuration_role")
        if role not in CONFIGURATION_ROLES:
            raise ValueError(f"{row.get('run_id')} has invalid role {role!r}")
        expected_eligible = role == "scientific_finalist"
        if row.get("relational_selection_eligible") is not expected_eligible:
            raise ValueError(f"{row.get('run_id')} selectability contradicts role")
        enabled = list(row.get("enabled_relations", []))
        if not enabled or enabled[0] != "base4":
            raise ValueError(f"{row.get('run_id')} must start with base4")
        if len(enabled) != len(set(enabled)):
            raise ValueError(f"{row.get('run_id')} repeats a relation family")
        if any(str(name).startswith("<resolved_") for name in enabled):
            if enabled[0] != "base4" or len(enabled) != 2:
                raise ValueError(
                    f"{row.get('run_id')} has an invalid dynamic relation placeholder"
                )
        else:
            try:
                indices = [family_index[name] for name in enabled]
            except KeyError as exc:
                raise ValueError(f"{row.get('run_id')} has an unknown family") from exc
            if indices != sorted(indices):
                raise ValueError(f"{row.get('run_id')} violates canonical family order")
    return output


def build_screening_registry(*, relation_registry_sha256: str) -> dict[str, Any]:
    require_sha256(relation_registry_sha256, name="relation_registry_sha256")
    rows = _validate_run_rows(run.to_dict() for run in _SCREENING_RUNS)
    if len(rows) != 21:
        raise AssertionError("the locked screening registry must contain 21 rows")
    return with_content_hash(
        {
            "contract": SCREENING_REGISTRY_CONTRACT,
            "schema_version": 1,
            "relation_registry_sha256": relation_registry_sha256,
            "screening_seed": 101,
            "fixed_before_validation": True,
            "rows": rows,
            "row_count": len(rows),
            "selection_roles": ["scientific_finalist"],
            "performance_failure_cancels_campaign": False,
            "wide_capacity_search": {
                "variables": ["pair_stem_hidden_1", "pair_stem_hidden_2", "pair_output_hidden"],
                "integer_range_inclusive": [64, 256],
                "target": "P(RPT_FULL_ALL)-P(RPT_BASE)",
                "candidate": "P(RPT_BASE_WIDE_MAX)-P(RPT_BASE)",
                "maximum_relative_incremental_mismatch": 0.02,
                "tie_breaks": [
                    "minimum_absolute_incremental_parameter_mismatch",
                    "lower_measured_forward_FLOPs",
                    "smaller_width_tuple_lexicographically",
                ],
            },
        }
    )


def validate_screening_registry(registry: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        registry, expected_contract=SCREENING_REGISTRY_CONTRACT
    )
    rows = _validate_run_rows(registry.get("rows", []))
    if len(rows) != 21 or int(registry.get("row_count", -1)) != 21:
        raise ValueError("screening registry must contain exactly 21 rows")
    expected_ids = [run.run_id for run in _SCREENING_RUNS]
    if [row["run_id"] for row in rows] != expected_ids:
        raise ValueError("screening registry row order or IDs differ from the lock")
    return digest


def build_confirmation_architecture_registry(
    *, relation_registry_sha256: str, screening_registry_sha256: str
) -> dict[str, Any]:
    require_sha256(relation_registry_sha256, name="relation_registry_sha256")
    require_sha256(screening_registry_sha256, name="screening_registry_sha256")
    templates = _validate_run_rows(
        [
            _Run(
                "RPT_SELECTED_UNION",
                (),
                "union of the two highest-ranked screening singles",
            ).to_dict()
            | {
                "enabled_relations": ["base4", "<resolved_selected_union>"],
                "new_relation_families": ["<resolved_selected_union>"],
                "resolution_rule": "union_of_top_two_single_family_rows",
            },
            _Run(
                "RPT_BASE_LAYERWISE",
                (),
                "base4 layer-specific projection architecture control",
                "architecture_control",
                architecture="layerwise_bias",
            ).to_dict(),
            _Run(
                "RPT_BASE_EDGEVALUE",
                (),
                "base4 edge-value architecture control",
                "architecture_control",
                architecture="layerwise_bias_and_edge_value",
            ).to_dict(),
            _Run(
                "RPT_SELECTED_LAYERWISE",
                (),
                "selected relations with layer-specific bias projections",
                architecture="layerwise_bias",
            ).to_dict()
            | {
                "enabled_relations": ["base4", "<resolved_selected_relation_set>"],
                "new_relation_families": ["<resolved_selected_relation_set>"],
                "resolution_rule": "selected_shared_bias_relation_set",
            },
            _Run(
                "RPT_SELECTED_EDGEVALUE",
                (),
                "selected relations with layerwise bias and edge-value messages",
                architecture="layerwise_bias_and_edge_value",
            ).to_dict()
            | {
                "enabled_relations": ["base4", "<resolved_selected_relation_set>"],
                "new_relation_families": ["<resolved_selected_relation_set>"],
                "resolution_rule": "selected_shared_bias_relation_set",
            },
        ]
    )
    return with_content_hash(
        {
            "contract": CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT,
            "schema_version": 1,
            "relation_registry_sha256": relation_registry_sha256,
            "screening_registry_sha256": screening_registry_sha256,
            "confirmation_seeds": [101, 202, 303],
            "templates": templates,
            "mandatory_rows": {
                "fixed": [
                    "RPT_BASE",
                    "RPT_BASE_WIDE_MAX",
                    "RPT_FULL_ZERO_REL",
                    "RPT_PT",
                    "RPT_TRACK",
                    "RPT_PID",
                    "RPT_CHARGE",
                    "RPT_DENSITY",
                    "RPT_REGION",
                    "RPT_FULL_ALL",
                    "RPT_BASE_LAYERWISE",
                    "RPT_BASE_EDGEVALUE",
                    "RPT_SELECTED_LAYERWISE",
                    "RPT_SELECTED_EDGEVALUE",
                ],
                "rank_resolved": [
                    "best_two_predeclared_non_full_combinations",
                    "RPT_SELECTED_UNION_when_distinct",
                ],
            },
            "seed_101_reuse_requires_exact_hash_match": True,
            "architecture_contracts": {
                "shared_pair_stem_evaluations_per_batch": 1,
                "layerwise_bias": (
                    "independent_final_head_projection_per_particle_attention_layer"
                ),
                "edge_value": {
                    "projection": "per_layer_per_head_linear_no_bias",
                    "materialize_B_H_N_N_dh": False,
                    "aggregation": "attention_weighted_relation_then_linear_projection",
                },
                "base_controls_use_base4_only": True,
                "all_models_initialized_from_scratch": True,
            },
        }
    )


def build_semantic_control_registry(
    *, relation_registry_sha256: str, confirmation_registry_sha256: str
) -> dict[str, Any]:
    require_sha256(relation_registry_sha256, name="relation_registry_sha256")
    require_sha256(confirmation_registry_sha256, name="confirmation_registry_sha256")
    controls = [
        {
            "control_id": "RPT_SELECTED_WITHIN_JET_SHUFFLED",
            "configuration_role": "semantic_control",
            "kind": "inference_perturbation",
            "split": "stack_val",
            "rule": "nonidentity_valid_index_PEPt_tokens_fixed",
            "relational_selection_eligible": False,
        },
        {
            "control_id": "RPT_SELECTED_WRONG_EVENT",
            "configuration_role": "semantic_control",
            "kind": "inference_perturbation",
            "split": "stack_val",
            "rule": "class_blind_exact_Nvalid_derangement_rank_aligned",
            "relational_selection_eligible": False,
        },
        {
            "control_id": "RPT_SELECTED_DIRECTIONAL_SWAP",
            "configuration_role": "semantic_control",
            "kind": "inference_perturbation",
            "split": "stack_val",
            "rule": "transpose_new_relation_channels_keep_base4",
            "relational_selection_eligible": False,
        },
        {
            "control_id": "RPT_SELECTED_UNARY",
            "configuration_role": "semantic_control",
            "kind": "trained_unary_endpoint_control",
            "split": "stack_val",
            "seeds": [101, 202, 303],
            "relational_selection_eligible": False,
            "source_relation_rule": "nominal_winner_active_family_set",
            "reference_rule": "ordinary_shared_bias_row_with_same_family_set",
            "adapter_location": "add_to_128d_particle_embedding_before_first_attention",
            "explicit_pair_only_quantities_forbidden": True,
            "unary_endpoint_features": {
                "PT": [
                    "pt_fraction",
                    "log_pt_fraction",
                    "average_normalized_pt_rank",
                ],
                "TRACK": [
                    "d0",
                    "dz",
                    "log_d0_sigma_effective",
                    "log_dz_sigma_effective",
                    "asinh_d0_significance",
                    "asinh_dz_significance",
                    "track_valid",
                ],
                "PID": ["independent_embedding_6x8"],
                "CHARGE": ["quantized_charge", "independent_embedding_3x4"],
                "DENSITY": ["complete_normalized_22_channel_node_descriptor"],
                "REGION": [
                    "K2_six_channel_endpoint_descriptor",
                    "K4_six_channel_endpoint_descriptor",
                    "K8_six_channel_endpoint_descriptor",
                ],
            },
            "unary_feature_order": list(CANONICAL_FAMILY_ORDER[1:]),
            "categorical_embedding_reuse": False,
            "adapter": [
                "Linear(Du,h1)",
                "GELU",
                "RMSNorm(h1)",
                "Linear(h1,h2)",
                "GELU",
                "RMSNorm(h2)",
                "Linear(h2,128)",
            ],
            "forbidden_pair_only_features": [
                "track_chi2_and_endpoint_products",
                "pid_directed_residual_table",
                "charge_pair_states",
                "region_same_cluster_lca_merge",
                "all_query_context_differences",
            ],
            "parameter_match": {
                "target": "P(unary_reference_run)-P(RPT_BASE)",
                "candidate": (
                    "P(independent_PID_charge_embeddings)+P(UnaryAdapter)"
                ),
                "integer_width_search": {"h1": [1, 512], "h2": [1, 512]},
                "maximum_relative_incremental_mismatch": 0.02,
                "tie_breaks": [
                    "minimum_absolute_incremental_parameter_mismatch",
                    "lower_measured_adapter_FLOPs",
                    "smaller_h1",
                    "smaller_h2",
                ],
            },
        },
    ]
    return with_content_hash(
        {
            "contract": SEMANTIC_CONTROL_REGISTRY_CONTRACT,
            "schema_version": 1,
            "relation_registry_sha256": relation_registry_sha256,
            "confirmation_registry_sha256": confirmation_registry_sha256,
            "controls": controls,
            "validation_only_perturbations": [
                row["control_id"] for row in controls if row["kind"] == "inference_perturbation"
            ],
            "performance_gate": False,
        }
    )


def resolve_registered_run(
    run_id: str,
    *,
    screening_registry: Mapping[str, Any],
    confirmation_registry: Mapping[str, Any] | None = None,
    semantic_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a run ID without importing any training or model code."""

    screening_hash = validate_screening_registry(screening_registry)
    candidates = list(screening_registry.get("rows", []))
    if confirmation_registry is not None:
        confirmation_hash = validate_content_hash(
            confirmation_registry,
            expected_contract=CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT,
        )
        if confirmation_registry.get("screening_registry_sha256") != screening_hash:
            raise ValueError("confirmation registry belongs to another screening registry")
        candidates.extend(confirmation_registry.get("templates", []))
    else:
        confirmation_hash = None
    if semantic_registry is not None:
        if confirmation_hash is None:
            raise ValueError("semantic registry requires its confirmation registry")
        validate_content_hash(
            semantic_registry,
            expected_contract=SEMANTIC_CONTROL_REGISTRY_CONTRACT,
        )
        if semantic_registry.get("confirmation_registry_sha256") != confirmation_hash:
            raise ValueError("semantic registry belongs to another confirmation registry")
        candidates.extend(
            {"run_id": row["control_id"], **row}
            for row in semantic_registry.get("controls", [])
        )
    matches = [dict(row) for row in candidates if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise KeyError(f"run ID {run_id!r} resolved to {len(matches)} rows")
    return matches[0]


__all__ = [
    "CANONICAL_FAMILY_ORDER",
    "CONFIGURATION_ROLES",
    "CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT",
    "RELATION_FAMILY_REGISTRY_CONTRACT",
    "SCREENING_REGISTRY_CONTRACT",
    "SEMANTIC_CONTROL_REGISTRY_CONTRACT",
    "build_confirmation_architecture_registry",
    "build_relation_family_registry",
    "build_screening_registry",
    "build_semantic_control_registry",
    "resolve_registered_run",
    "validate_screening_registry",
]

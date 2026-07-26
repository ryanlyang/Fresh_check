"""Globally frozen numerical, metric, statistical, and schedule conventions."""

from __future__ import annotations

from typing import Any

import math
from typing import Mapping

from .contracts import validate_content_hash, with_content_hash


GLOBAL_DETERMINISM_CONTRACT = "relational_part_global_determinism_v3"
DIAGNOSTIC_BIN_EDGES = {
    "track_raw_displacement": (
        -math.inf, -1.0, -0.1, 0.0, 0.1, 1.0, math.inf
    ),
    "track_absolute_significance": (0.0, 1.0, 2.0, 4.0, 8.0, math.inf),
    "track_compatibility_chi2": (
        0.0, 1.0, 4.0, 9.0, 16.0, 25.0, math.inf
    ),
    "density_local_activity": (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    "jet_multiplicity": (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, math.inf),
    "leading_particle_pt_fraction": (
        0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0
    ),
    "region_lca_depth": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "region_log_merge_delta_r": (
        -math.inf, -4.0, -3.0, -2.0, -1.0, 0.0, math.inf
    ),
    "region_log_merge_kt": (
        -math.inf, -4.0, -2.0, 0.0, 2.0, 4.0, math.inf
    ),
    "region_merge_z": (0.0, 0.05, 0.1, 0.2, 0.35, 0.5),
    "region_log_merge_mass_fraction": (
        -math.inf, -6.0, -4.0, -2.0, -1.0, 0.0, math.inf
    ),
    "region_log_cluster_pt_fraction": (
        -math.inf, -4.0, -2.0, -1.0, -0.5, 0.0
    ),
    "region_log_cluster_mass_fraction": (
        -math.inf, -6.0, -4.0, -2.0, -1.0, 0.0
    ),
    "region_cluster_multiplicity_fraction": (
        0.0, 0.1, 0.25, 0.5, 0.75, 1.0
    ),
    "region_tree_depth": (0.0, 4.0, 8.0, 12.0, 16.0, math.inf),
    "region_hard_prong_count": (0.0, 2.0, 4.0, 6.0, 8.0, math.inf),
}


def _serialized_diagnostic_edges() -> dict[str, list[float | str]]:
    return {
        name: [
            "-inf" if value == -math.inf else "+inf" if value == math.inf else value
            for value in edges
        ]
        for name, edges in DIAGNOSTIC_BIN_EDGES.items()
    }


def build_global_determinism_contract() -> dict[str, Any]:
    """Build conventions that are immutable across every campaign model.

    These choices are fixed before any scientific result is inspected.  Later
    implementation steps consume this artifact rather than selecting local
    tolerances, metric definitions, or scheduler rounding rules.
    """

    return with_content_hash(
        {
            "contract": GLOBAL_DETERMINISM_CONTRACT,
            "schema_version": 3,
            "fixed_before_scientific_results": True,
            "model_specific_override_allowed": False,
            "attention_diagnostics": {
                "population": "complete_val_select",
                "aggregation": (
                    "schema_aware_sufficient_statistics_across_complete_"
                    "validation_population"
                ),
                "mean_denominators": {
                    "attention": "valid_particle_queries",
                    "pair_bias": "applicable_valid_directed_pairs",
                    "performance": "events_in_declared_bin",
                },
                "context_groups": {
                    "leading": "all_valid_particles_at_maximum_pt",
                    "subleading": (
                        "all_valid_particles_at_next_distinct_pt"
                    ),
                    "soft": "all_remaining_valid_particles",
                },
                "angular_band_edges": [0.0, 0.05, 0.10, 0.20, 0.40],
                "angular_band_endpoint_policy": (
                    "[0,0.05],(0.05,0.10],(0.10,0.20],"
                    "(0.20,0.40],(0.40,inf)"
                ),
                "combination_family_dropouts": (
                    "zero_exactly_one_encoded_family_after_base4_"
                    "concatenation_without_retraining"
                ),
                "region_resolution_dropouts": [2, 4, 8],
                "diagnostic_bin_edges": _serialized_diagnostic_edges(),
                "diagnostic_bin_endpoint_policy": (
                    "first bin [left,right], later bins (left,right]"
                ),
            },
            "parity": {
                "authoritative_weaver_explicit_uu": {
                    "device_path": "real_installed_weaver",
                    "dtype": "float32",
                    "autocast_enabled": False,
                    "gradient_scaler_enabled": False,
                    "evaluation_mode": True,
                    "atol": 1.0e-6,
                    "rtol": 1.0e-6,
                    "applies_to": [
                        "standard_four_pair_features",
                        "logits",
                        "input_gradients",
                        "parameter_gradients",
                        "valid_token_padding_invariance",
                    ],
                    "nonfinite_is_failure": True,
                },
                "state_dictionary": {
                    "keys_exact": True,
                    "ordered_keys_exact": True,
                    "shapes_exact": True,
                    "dtypes_exact": True,
                    "copied_initial_tensor_values_bitwise_exact": True,
                    "missing_or_extra_gradient_exact": True,
                },
                "region_python_cpp_float64": {
                    "atol": 1.0e-10,
                    "rtol": 1.0e-10,
                    "continuous_values_only": True,
                },
                "region_float32_sidecar_runtime": {
                    "comparison_dtype": "float32",
                    "atol": 2.0e-6,
                    "rtol": 2.0e-6,
                    "continuous_values_only": True,
                },
                "relation_permutation_fp32_logits": {
                    "atol": 2.0e-6,
                    "rtol": 2.0e-6,
                },
                "exact_fields": [
                    "integer_tree_topology",
                    "parent_child_indices",
                    "exclusive_cluster_assignments",
                    "particle_masks",
                    "pair_masks",
                    "categorical_states",
                    "event_identities",
                    "state_dictionary_structure",
                ],
                "mixed_precision": {
                    "authoritative_equivalence_allowed": False,
                    "required_checks": [
                        "finite_outputs",
                        "finite_gradients",
                        "exact_masks",
                        "exact_categories",
                        "exact_identities",
                    ],
                },
            },
            "paired_bootstrap": {
                "seed": 917_301,
                "replicates": 10_000,
                "bit_generator": "numpy.PCG64",
                "random_draw": (
                    "Generator(PCG64(seed)).integers("
                    "0,class_count,size=class_count,endpoint=False,dtype=int64)"
                ),
                "seed_reused_for_every_paired_comparison": True,
                "sampling_unit": "aligned_event_identity",
                "sampling": "with_replacement",
                "pairing": (
                    "one sampled identity indexes candidate and matched baseline"
                ),
                "stratification": {
                    "field": "true_class",
                    "class_order": [
                        "QCD",
                        "Hbb",
                        "Hcc",
                        "Hgg",
                        "H4q",
                        "Hqql",
                        "Zqq",
                        "Wqq",
                        "Tbqq",
                        "Tbl",
                    ],
                    "draws_per_class": "original_event_count_in_that_class",
                    "within_class_source_order": (
                        "split_manifest_event_identity_order"
                    ),
                    "empty_class_policy": "fail",
                    "balanced_split_effect": (
                        "every replicate preserves exact class balance"
                    ),
                },
                "statistic": (
                    "candidate_accuracy_minus_matched_baseline_accuracy_over_"
                    "the_concatenated_class_stratified_resample"
                ),
                "interval": {
                    "kind": "two_sided_percentile",
                    "lower_percent": 2.5,
                    "upper_percent": 97.5,
                    "quantile_method": "Hyndman_Fan_type_7_linear",
                    "numpy_method": "linear",
                    "bootstrap_mean_used_as_endpoint": False,
                },
            },
            "calibration": {
                "ece": {
                    "kind": "top_label_multiclass",
                    "bin_count": 15,
                    "bin_edges": "equal_width_linspace_0_to_1_inclusive",
                    "membership": (
                        "left_closed_right_open_except_final_bin_closed"
                    ),
                    "confidence": "maximum_class_probability",
                    "probabilities": "float64_stable_softmax_from_logits",
                    "prediction_tie_rule": "lowest_canonical_class_index",
                    "correctness": "argmax_class_equals_true_class",
                    "per_bin_term": (
                        "event_fraction_times_absolute_accuracy_minus_"
                        "mean_confidence"
                    ),
                    "empty_bin_contribution": 0.0,
                    "sum_dtype": "float64",
                },
                "brier": {
                    "kind": "multiclass",
                    "definition": (
                        "event_mean_of_sum_over_classes_probability_minus_"
                        "one_hot_squared"
                    ),
                    "calculation_dtype": "float64",
                },
            },
            "qcd_signal_rejection": {
                "event_subset": "true_label_is_QCD_or_requested_signal",
                "discriminant": "logit_signal_minus_logit_QCD",
                "calculation_dtype": "float64",
                "target_signal_efficiencies": [0.30, 0.50],
                "threshold": {
                    "rank": "ceil(target_efficiency_times_signal_support)",
                    "value": "that_ranked_signal_score_in_descending_order",
                    "pass_rule": "score_greater_than_or_equal_to_threshold",
                    "tie_policy": (
                        "all_events_tied_at_threshold_pass_and_achieved_"
                        "efficiency_may_exceed_target"
                    ),
                },
                "reported_counts": [
                    "signal_support",
                    "signal_pass_count",
                    "qcd_support",
                    "qcd_false_positive_count",
                ],
                "reported_rates": [
                    "target_signal_efficiency",
                    "achieved_signal_efficiency",
                    "qcd_false_positive_rate",
                ],
                "background_rejection": "one_over_qcd_false_positive_rate",
                "zero_background_behavior": {
                    "background_rejection": None,
                    "background_rejection_is_infinite": True,
                    "qcd_false_positive_rate": 0.0,
                    "qcd_false_positive_count": 0,
                    "reason": "avoid_nonfinite_JSON_while_preserving_exact_meaning",
                },
                "missing_signal_or_qcd_support": "fail",
            },
            "auc": {
                "kind": "one_vs_rest",
                "score": "class_probability",
                "tie_handling": "average_ranks",
                "calculation_dtype": "float64",
                "missing_positive_or_negative_support": "fail",
            },
            "continuous_quantiles": {
                "calculation_dtype": "float64",
                "method": "Hyndman_Fan_type_7_linear",
                "numpy_method": "linear",
                "nonfinite_input": "fail",
            },
            "optimizer_update_schedule": {
                "microbatch_size": 64,
                "gradient_accumulation_steps": 2,
                "accumulation_groups_cross_epoch_boundary": False,
                "dataloader_drop_last": False,
                "final_partial_microbatch_allowed": True,
                "final_partial_accumulation_group": (
                    "optimizer_step_after_all_remaining_microbatches"
                ),
                "gradient_normalization": (
                    "sum_of_event_losses_divided_by_actual_event_count_"
                    "in_accumulation_group"
                ),
                "microbatches_per_epoch": "ceil(training_event_count/64)",
                "optimizer_updates_per_epoch": (
                    "ceil(microbatches_per_epoch/2)"
                ),
                "total_optimizer_updates": (
                    "maximum_epochs_times_optimizer_updates_per_epoch"
                ),
                "warmup_updates": (
                    "zero_if_total_updates_is_zero_else_"
                    "min(total_updates,max(1,ceil(0.05*total_updates)))"
                ),
                "update_ordinal": "one_based",
                "warmup_lr": (
                    "base_lr_times_update_ordinal_divided_by_warmup_updates"
                ),
                "post_warmup_progress": (
                    "(update_ordinal-warmup_updates)/"
                    "(total_updates-warmup_updates)"
                ),
                "post_warmup_lr": (
                    "min_lr+0.5*(base_lr-min_lr)*"
                    "(1+cos(pi*post_warmup_progress))"
                ),
                "total_equals_warmup_behavior": (
                    "the_only_updates_use_warmup_lr_and_the_last_equals_base_lr"
                ),
                "early_stopping_changes_planned_schedule": False,
                "resume_restores_optimizer_update_ordinal_exactly": True,
            },
        }
    )


def optimizer_update_counts(
    *,
    training_event_count: int,
    maximum_epochs: int,
) -> dict[str, int]:
    """Evaluate the locked integer update-count and warm-up definitions."""

    training_event_count = int(training_event_count)
    maximum_epochs = int(maximum_epochs)
    if training_event_count < 0 or maximum_epochs < 0:
        raise ValueError("training_event_count and maximum_epochs must be nonnegative")
    microbatches_per_epoch = (training_event_count + 63) // 64
    optimizer_updates_per_epoch = (microbatches_per_epoch + 1) // 2
    total_optimizer_updates = maximum_epochs * optimizer_updates_per_epoch
    warmup_updates = (
        0
        if total_optimizer_updates == 0
        else min(
            total_optimizer_updates,
            max(1, math.ceil(0.05 * total_optimizer_updates)),
        )
    )
    return {
        "microbatches_per_epoch": microbatches_per_epoch,
        "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
        "total_optimizer_updates": total_optimizer_updates,
        "warmup_updates": warmup_updates,
    }


def scheduled_learning_rate(
    *,
    update_ordinal: int,
    total_optimizer_updates: int,
    warmup_updates: int,
    base_lr: float,
    minimum_lr: float,
) -> float:
    """Evaluate the locked one-based linear-warmup/cosine schedule."""

    update_ordinal = int(update_ordinal)
    total_optimizer_updates = int(total_optimizer_updates)
    warmup_updates = int(warmup_updates)
    base_lr = float(base_lr)
    minimum_lr = float(minimum_lr)
    if total_optimizer_updates <= 0:
        raise ValueError("a learning rate requires at least one optimizer update")
    if not 1 <= update_ordinal <= total_optimizer_updates:
        raise ValueError("update_ordinal is outside the planned update range")
    if not 1 <= warmup_updates <= total_optimizer_updates:
        raise ValueError("warmup_updates is outside the planned update range")
    if not (math.isfinite(base_lr) and math.isfinite(minimum_lr)):
        raise ValueError("learning rates must be finite")
    if base_lr <= 0.0 or minimum_lr < 0.0 or minimum_lr > base_lr:
        raise ValueError("learning rates violate 0 <= minimum_lr <= base_lr")
    if update_ordinal <= warmup_updates:
        return base_lr * update_ordinal / warmup_updates
    progress = (update_ordinal - warmup_updates) / (
        total_optimizer_updates - warmup_updates
    )
    return minimum_lr + 0.5 * (base_lr - minimum_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def validate_global_determinism_contract(payload: Mapping[str, Any]) -> str:
    """Reject rehashed policy variants, including legacy or locally tuned ones."""

    digest = validate_content_hash(
        payload, expected_contract=GLOBAL_DETERMINISM_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_global_determinism_contract()
    expected.pop("content_hash", None)
    if actual != expected:
        raise ValueError(
            "global deterministic conventions differ from the locked v3 policy"
        )
    return digest


__all__ = [
    "GLOBAL_DETERMINISM_CONTRACT",
    "DIAGNOSTIC_BIN_EDGES",
    "build_global_determinism_contract",
    "optimizer_update_counts",
    "scheduled_learning_rate",
    "validate_global_determinism_contract",
]

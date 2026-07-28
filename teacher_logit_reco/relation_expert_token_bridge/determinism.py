"""Globally frozen numerical and statistical conventions for RETB."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import validate_content_hash, with_content_hash


GLOBAL_DETERMINISM_CONTRACT = "retb_global_determinism_v1"


def build_global_determinism() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": GLOBAL_DETERMINISM_CONTRACT,
            "schema_version": 1,
            "fixed_before_scientific_results": True,
            "model_specific_override_allowed": False,
            "canonical_class_order": [
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
            "parity": {
                "authoritative_weaver": {
                    "dtype": "float32",
                    "autocast_enabled": False,
                    "gradient_scaler_enabled": False,
                    "atol": 1.0e-6,
                    "rtol": 1.0e-6,
                    "exact": [
                        "integer_topology",
                        "masks",
                        "categories",
                        "identities",
                        "state_dictionary_structure",
                    ],
                },
                "mixed_precision_is_authoritative": False,
            },
            "paired_bootstrap": {
                "seed": 917_301,
                "replicates": 10_000,
                "bit_generator": "numpy.PCG64",
                "sampling_unit": "canonical_jet_identity",
                "paired": True,
                "sampling": "with_replacement",
                "stratification": "true_class",
                "class_counts": "preserve_original_balanced_count_per_class",
                "quantiles_percent": [2.5, 97.5],
                "quantile_method": "Hyndman_Fan_type_7_linear",
                "numpy_method": "linear",
            },
            "ece": {
                "kind": "top_label_multiclass",
                "bin_count": 15,
                "edges": [index / 15 for index in range(16)],
                "membership": "left_closed_right_open_except_final_bin_closed",
                "confidence": "largest_softmax_probability",
                "exact_logit_tie": "lowest_class_index",
                "empty_bin_contribution": 0.0,
                "aggregation": (
                    "sample_count_weighted_absolute_mean_confidence_minus_accuracy"
                ),
                "calculation_dtype": "float64",
            },
            "qcd_signal_rejection": {
                "discriminant": "p_signal/(p_signal+p_QCD)",
                "target_signal_efficiencies": [0.30, 0.50],
                "threshold_candidates": [
                    "positive_infinity",
                    "every_unique_observed_score",
                    "negative_infinity",
                ],
                "pass_rule": "score_greater_than_or_equal_to_threshold",
                "selection": [
                    "minimum_absolute_achieved_minus_target_efficiency",
                    "larger_achieved_efficiency",
                    "larger_threshold",
                ],
                "report_achieved_efficiency": True,
                "zero_background_display": "positive_infinity",
                "finite_selector": "(N_QCD+1)/(n_pass+0.5)",
            },
            "optimizer_schedule": {
                "maximum_epochs": 40,
                "total_updates": (
                    "maximum_epochs*ceil(ceil(training_events/microbatch)/"
                    "gradient_accumulation_steps)"
                ),
                "warmup_updates": "min(T,max(1,ceil(0.05*T))) for T>0",
                "total_updates_zero": "invalid",
                "total_updates_one": "sole_update_uses_base_lr",
                "update_ordinal": "one_based",
                "post_warmup": "cosine_to_minimum_lr",
                "resume_recomputes_schedule": False,
            },
            "retrieval": {
                "flatten_order": "slot_major_then_channel",
                "dot_and_norm_dtype": "float32",
                "cosine_denominator_epsilon": 1.0e-8,
                "zero_norm_similarity": 0.0,
                "temperature": 0.1,
                "exact_tie": "ascending_canonical_identity",
                "info_nce_logsumexp_dtype": "float32",
            },
            "covariance": {
                "kind": "centered_population_1_over_B_per_slot",
                "dtype": "float32",
                "effective_batch": "all_accumulation_steps_and_distributed_ranks",
                "distributed_sufficient_statistics": "differentiable_all_reduce",
                "relative_frobenius_epsilon": 1.0e-8,
                "effective_batch_below_two": "invalid",
            },
            "effective_rank": {
                "matrix": "n_events_by_K_times_D_slot_major_train_normalized",
                "dtype_backend": "C_contiguous_CPU_float64_numpy_linalg_svd",
                "full_matrices": False,
                "zero_threshold_relative_to_smax": 1.0e-12,
                "all_zero_rank": 0.0,
                "version_binding": "numpy_and_linked_LAPACK",
            },
            "scientific_performance_failure_stops_run": False,
            "nonfinite_required_quantity": "fail_closed",
        }
    )


def validate_global_determinism(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=GLOBAL_DETERMINISM_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_global_determinism()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("global deterministic conventions differ from RETB v1")
    return digest


def optimizer_update_counts(
    *,
    training_event_count: int,
    maximum_epochs: int = 40,
    microbatch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, int]:
    values = (
        int(training_event_count),
        int(maximum_epochs),
        int(microbatch_size),
        int(gradient_accumulation_steps),
    )
    if values[0] <= 0 or values[1] <= 0 or values[2] <= 0 or values[3] <= 0:
        raise ValueError("training and schedule integers must be positive")
    microbatches = math.ceil(values[0] / values[2])
    updates_per_epoch = math.ceil(microbatches / values[3])
    total = values[1] * updates_per_epoch
    warmup = min(total, max(1, math.ceil(0.05 * total)))
    return {
        "microbatches_per_epoch": microbatches,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "total_optimizer_updates": total,
        "warmup_updates": warmup,
    }


def scheduled_learning_rate(
    *,
    update_ordinal: int,
    total_optimizer_updates: int,
    warmup_updates: int,
    base_learning_rate: float,
    minimum_learning_rate: float = 1.0e-5,
) -> float:
    ordinal = int(update_ordinal)
    total = int(total_optimizer_updates)
    warmup = int(warmup_updates)
    base = float(base_learning_rate)
    minimum = float(minimum_learning_rate)
    if total <= 0 or not 1 <= ordinal <= total:
        raise ValueError("optimizer update ordinal lies outside the schedule")
    if not 1 <= warmup <= total:
        raise ValueError("warm-up update count lies outside the schedule")
    if not 0.0 <= minimum <= base:
        raise ValueError("learning-rate endpoints are invalid")
    if ordinal <= warmup:
        return base * ordinal / warmup
    if total == warmup:
        return base
    progress = (ordinal - warmup) / (total - warmup)
    return minimum + 0.5 * (base - minimum) * (
        1.0 + math.cos(math.pi * progress)
    )


def replica_cycle(*, epoch: int, identity_hash_low_two_bits: int) -> int:
    if int(epoch) < 0:
        raise ValueError("epoch must be nonnegative")
    if int(identity_hash_low_two_bits) not in range(4):
        raise ValueError("identity hash low bits must be in [0,3]")
    return (int(epoch) + int(identity_hash_low_two_bits)) % 4


__all__ = [
    "GLOBAL_DETERMINISM_CONTRACT",
    "build_global_determinism",
    "optimizer_update_counts",
    "replica_cycle",
    "scheduled_learning_rate",
    "validate_global_determinism",
]

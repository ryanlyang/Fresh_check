"""Deterministic paired event statistics for RETB confirmation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .evaluation import (
    CLASS_NAMES,
    qcd_signal_rejection,
    stable_probabilities,
)


PAIRED_STATISTICS_CONTRACT = "retb_paired_confirmation_statistics_v1"
BOOTSTRAP_SEED = 917_301
BOOTSTRAP_REPLICATES = 10_000


def _validate_inputs(
    *,
    identities: Sequence[str],
    labels: np.ndarray,
    candidate_logits: np.ndarray,
    baseline_logits: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    ids = tuple(str(value) for value in identities)
    truth = np.asarray(labels, dtype=np.int64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    baseline = np.asarray(baseline_logits, dtype=np.float64)
    counts = np.bincount(truth, minlength=10)
    if (
        not ids
        or len(ids) != len(set(ids))
        or ids != tuple(sorted(ids))
        or truth.shape != (len(ids),)
        or candidate.shape != (len(ids), 10)
        or baseline.shape != candidate.shape
        or not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
        or bool(((truth < 0) | (truth >= 10)).any())
        or bool((counts == 0).any())
        or len(set(counts.tolist())) != 1
    ):
        raise ValueError(
            "paired confirmation requires finite, identity-unique, "
            "lexicographically identity-sorted, class-balanced common "
            "predictions"
        )
    return ids, truth, candidate, baseline


def _rejection_terms(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    logs, central = [], []
    for signal_index, signal in enumerate(CLASS_NAMES[1:], start=1):
        for target in (0.30, 0.50):
            row = qcd_signal_rejection(
                probabilities,
                labels,
                signal_index=signal_index,
                target_efficiency=target,
            )
            logs.append(math.log(row["finite_selection_rejection"]))
            central.append(
                {
                    "signal_class": signal,
                    "target_signal_efficiency": target,
                    "achieved_signal_efficiency": row[
                        "achieved_signal_efficiency"
                    ],
                    "qcd_support": row["qcd_support"],
                    "qcd_pass_count": row["qcd_pass_count"],
                    "background_efficiency": row[
                        "background_efficiency"
                    ],
                    "background_rejection": row[
                        "background_rejection"
                    ],
                    "finite_selection_rejection": row[
                        "finite_selection_rejection"
                    ],
                }
            )
    return np.asarray(logs, dtype=np.float64), central


def _bootstrap_rejection_terms(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Exact selector terms while sharing threshold work across targets."""

    terms = []
    qcd = labels == 0
    qcd_support = int(qcd.sum())
    for signal_index in range(1, 10):
        denominator = (
            probabilities[:, signal_index] + probabilities[:, 0]
        )
        if bool((denominator <= 0).any()):
            raise FloatingPointError(
                "QCD-versus-signal denominator is nonpositive"
            )
        score = probabilities[:, signal_index] / denominator
        signal = labels == signal_index
        signal_support = int(signal.sum())
        signal_scores = np.sort(score[signal])
        qcd_scores = np.sort(score[qcd])
        thresholds = np.concatenate(
            (
                np.asarray([np.inf]),
                np.unique(signal_scores)[::-1],
                np.asarray([-np.inf]),
            )
        )
        signal_pass = signal_support - np.searchsorted(
            signal_scores, thresholds, side="left"
        )
        achieved = signal_pass / signal_support
        for target in (0.30, 0.50):
            distances = np.abs(achieved - target)
            eligible = np.flatnonzero(distances == distances.min())
            maximum_achieved = achieved[eligible].max()
            eligible = eligible[achieved[eligible] == maximum_achieved]
            threshold = thresholds[int(eligible[0])]
            qcd_pass = int(
                qcd_support
                - np.searchsorted(qcd_scores, threshold, side="left")
            )
            terms.append(
                math.log((qcd_support + 1) / (qcd_pass + 0.5))
            )
    return np.asarray(terms, dtype=np.float64)


def _linear_interval(values: np.ndarray) -> list[float]:
    quantiles = np.quantile(
        np.asarray(values, dtype=np.float64),
        [0.025, 0.975],
        method="linear",
    )
    return [float(quantiles[0]), float(quantiles[1])]


def build_paired_confirmation_statistics(
    *,
    identities: Sequence[str],
    labels: np.ndarray,
    candidate_logits: np.ndarray,
    baseline_logits: np.ndarray,
    candidate_graph_id: str,
    baseline_graph_id: str,
    pipeline_seed: int,
    candidate_prediction_sha256: str,
    baseline_prediction_sha256: str,
) -> dict[str, Any]:
    ids, truth, candidate, baseline = _validate_inputs(
        identities=identities,
        labels=labels,
        candidate_logits=candidate_logits,
        baseline_logits=baseline_logits,
    )
    if int(pipeline_seed) not in {101, 202, 303}:
        raise ValueError("paired-statistics pipeline seed differs")
    candidate_probability = stable_probabilities(candidate)
    baseline_probability = stable_probabilities(baseline)
    candidate_prediction = candidate_probability.argmax(axis=1)
    baseline_prediction = baseline_probability.argmax(axis=1)
    candidate_correct = candidate_prediction == truth
    baseline_correct = baseline_prediction == truth
    candidate_accuracy = float(candidate_correct.mean(dtype=np.float64))
    baseline_accuracy = float(baseline_correct.mean(dtype=np.float64))
    candidate_terms, candidate_central = _rejection_terms(
        candidate_probability, truth
    )
    baseline_terms, baseline_central = _rejection_terms(
        baseline_probability, truth
    )
    class_indices = [np.flatnonzero(truth == index) for index in range(10)]
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    accuracy_differences = np.empty(
        BOOTSTRAP_REPLICATES, dtype=np.float64
    )
    candidate_term_samples = np.empty(
        (BOOTSTRAP_REPLICATES, 18), dtype=np.float64
    )
    baseline_term_samples = np.empty_like(candidate_term_samples)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [
                indices[
                    generator.integers(
                        0, len(indices), size=len(indices)
                    )
                ]
                for indices in class_indices
            ]
        )
        sampled_truth = truth[sampled]
        accuracy_differences[replicate] = (
            candidate_correct[sampled].mean(dtype=np.float64)
            - baseline_correct[sampled].mean(dtype=np.float64)
        )
        candidate_term_samples[replicate] = _bootstrap_rejection_terms(
            candidate_probability[sampled], sampled_truth
        )
        baseline_term_samples[replicate] = _bootstrap_rejection_terms(
            baseline_probability[sampled], sampled_truth
        )
    candidate_mean_log = candidate_term_samples.mean(axis=1)
    baseline_mean_log = baseline_term_samples.mean(axis=1)
    rejection_difference = candidate_mean_log - baseline_mean_log
    per_term = []
    for index, candidate_row in enumerate(candidate_central):
        candidate_interval = _linear_interval(
            candidate_term_samples[:, index]
        )
        baseline_interval = _linear_interval(
            baseline_term_samples[:, index]
        )
        per_term.append(
            {
                "signal_class": candidate_row["signal_class"],
                "target_signal_efficiency": candidate_row[
                    "target_signal_efficiency"
                ],
                "candidate_central": candidate_row,
                "baseline_central": baseline_central[index],
                "candidate_log_rejection_interval_95": (
                    candidate_interval
                ),
                "candidate_rejection_interval_95": [
                    float(math.exp(value))
                    for value in candidate_interval
                ],
                "baseline_log_rejection_interval_95": baseline_interval,
                "baseline_rejection_interval_95": [
                    float(math.exp(value))
                    for value in baseline_interval
                ],
            }
        )
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        selected = truth == index
        per_class[name] = (
            float(candidate_correct[selected].mean(dtype=np.float64))
            - float(baseline_correct[selected].mean(dtype=np.float64))
        )
    baseline_error = 1.0 - baseline_accuracy
    return with_content_hash(
        {
            "contract": PAIRED_STATISTICS_CONTRACT,
            "schema_version": 1,
            "candidate_graph_id": str(candidate_graph_id),
            "baseline_graph_id": str(baseline_graph_id),
            "pipeline_seed": int(pipeline_seed),
            "identity_count": len(ids),
            "identity_order_sha256": canonical_sha256(list(ids)),
            "identity_order": "lexicographic_canonical_jet_identity",
            "class_order": list(CLASS_NAMES),
            "balanced_count_per_class": len(class_indices[0]),
            "parents": {
                "candidate_prediction": str(
                    require_sha256(
                        candidate_prediction_sha256,
                        name="candidate_prediction_sha256",
                    )
                ),
                "baseline_prediction": require_sha256(
                    baseline_prediction_sha256,
                    name="baseline_prediction_sha256",
                ),
            },
            "central": {
                "candidate_accuracy": candidate_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "paired_accuracy_difference": (
                    candidate_accuracy - baseline_accuracy
                ),
                "relative_error_reduction": (
                    None
                    if baseline_error == 0.0
                    else (
                        baseline_error - (1.0 - candidate_accuracy)
                    )
                    / baseline_error
                ),
                "McNemar_candidate_correct_baseline_wrong": int(
                    (candidate_correct & ~baseline_correct).sum()
                ),
                "McNemar_candidate_wrong_baseline_correct": int(
                    (~candidate_correct & baseline_correct).sum()
                ),
                "per_class_accuracy_difference": per_class,
                "candidate_mean_log_rejection": float(
                    candidate_terms.mean(dtype=np.float64)
                ),
                "baseline_mean_log_rejection": float(
                    baseline_terms.mean(dtype=np.float64)
                ),
                "mean_log_rejection_difference": float(
                    (candidate_terms - baseline_terms).mean(
                        dtype=np.float64
                    )
                ),
            },
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "bit_generator": "numpy.PCG64",
                "replicates": BOOTSTRAP_REPLICATES,
                "paired_sampling_unit": "canonical_jet_identity",
                "stratification": "true_class",
                "class_counts_preserved": True,
                "sampling": "with_replacement",
                "quantile_percent": [2.5, 97.5],
                "quantile_method": "linear",
                "paired_accuracy_difference_interval_95": (
                    _linear_interval(accuracy_differences)
                ),
                "candidate_mean_log_rejection_interval_95": (
                    _linear_interval(candidate_mean_log)
                ),
                "baseline_mean_log_rejection_interval_95": (
                    _linear_interval(baseline_mean_log)
                ),
                "paired_mean_log_rejection_difference_interval_95": (
                    _linear_interval(rejection_difference)
                ),
                "thresholds_recomputed_in_every_resample": True,
                "all_18_terms_recomputed_in_every_resample": True,
            },
            "per_signal_rejection": per_term,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }
    )


def validate_paired_confirmation_statistics(
    payload: Mapping[str, Any],
) -> str:
    """Validate the frozen sampling/statistics contract without raw logits."""

    digest = validate_content_hash(
        payload, expected_contract=PAIRED_STATISTICS_CONTRACT
    )
    bootstrap = payload.get("bootstrap", {})
    parents = payload.get("parents", {})
    central = payload.get("central", {})
    rows = payload.get("per_signal_rejection", [])
    expected_terms = {
        (signal, target)
        for signal in CLASS_NAMES[1:]
        for target in (0.30, 0.50)
    }
    actual_terms = {
        (
            row.get("signal_class"),
            float(row.get("target_signal_efficiency", -1.0)),
        )
        for row in rows
    }
    finite_central_keys = (
        "candidate_accuracy",
        "baseline_accuracy",
        "paired_accuracy_difference",
        "candidate_mean_log_rejection",
        "baseline_mean_log_rejection",
        "mean_log_rejection_difference",
    )
    if (
        not str(payload.get("candidate_graph_id", ""))
        or not str(payload.get("baseline_graph_id", ""))
        or int(payload.get("pipeline_seed", -1)) not in {101, 202, 303}
        or int(payload.get("identity_count", 0)) <= 0
        or int(payload.get("balanced_count_per_class", 0)) * 10
        != int(payload.get("identity_count", -1))
        or payload.get("identity_order")
        != "lexicographic_canonical_jet_identity"
        or payload.get("class_order") != list(CLASS_NAMES)
        or set(parents) != {"candidate_prediction", "baseline_prediction"}
        or any(
            require_sha256(value, name=f"parents.{name}") != value
            for name, value in parents.items()
        )
        or bootstrap.get("seed") != BOOTSTRAP_SEED
        or bootstrap.get("bit_generator") != "numpy.PCG64"
        or bootstrap.get("replicates") != BOOTSTRAP_REPLICATES
        or bootstrap.get("paired_sampling_unit") != "canonical_jet_identity"
        or bootstrap.get("stratification") != "true_class"
        or bootstrap.get("class_counts_preserved") is not True
        or bootstrap.get("sampling") != "with_replacement"
        or bootstrap.get("quantile_percent") != [2.5, 97.5]
        or bootstrap.get("quantile_method") != "linear"
        or bootstrap.get("thresholds_recomputed_in_every_resample")
        is not True
        or bootstrap.get("all_18_terms_recomputed_in_every_resample")
        is not True
        or len(rows) != 18
        or actual_terms != expected_terms
        or any(
            not math.isfinite(float(central.get(key, math.nan)))
            for key in finite_central_keys
        )
        or payload.get("stack_val_consumed") is not False
        or payload.get("final_test_consumed") is not False
    ):
        raise ValueError("paired-confirmation statistics semantics differ")
    for row in rows:
        for prefix in ("candidate", "baseline"):
            interval = row.get(f"{prefix}_log_rejection_interval_95")
            rejection_interval = row.get(
                f"{prefix}_rejection_interval_95"
            )
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not isinstance(rejection_interval, list)
                or len(rejection_interval) != 2
                or any(
                    not math.isfinite(float(value))
                    for value in (*interval, *rejection_interval)
                )
                or interval[0] > interval[1]
                or rejection_interval[0] > rejection_interval[1]
            ):
                raise ValueError(
                    "paired-confirmation rejection interval differs"
                )
    return digest


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "PAIRED_STATISTICS_CONTRACT",
    "build_paired_confirmation_statistics",
    "validate_paired_confirmation_statistics",
]

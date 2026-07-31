"""Exact inherited metrics plus HOSD utility, robustness, and reports."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (
    mean_log_selection_rejection,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
    expected_calibration_error,
)
from teacher_logit_reco.relation_expert_token_bridge.paired_statistics import (
    _bootstrap_rejection_terms,
    _rejection_terms,
)

from .contracts import (
    HOSD_METRICS_CONTRACT,
    HOSD_REPORT_CONTRACT,
    HOSD_PAIRED_STATISTICS_CONTRACT,
    ROBUSTNESS_PLAN_CONTRACT,
    ROBUSTNESS_RESULT_CONTRACT,
    ROBUSTNESS_SUMMARY_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    stable_probabilities,
)


DEGRADATION_PROFILES = (
    "D_OFFLINE_IDENTITY",
    "D_KIN_ONLY",
    "D_TRACK_ONLY",
    "D_MISSING_ONLY",
    "D_NOMINAL",
    "D_MILD",
    "D_SEVERE",
    "D_LEGACY_V1",
    "D_LEGACY_V2",
)
REPLICA_POLICIES = ("R_FIXED", "R_MULTI", "R_RANDOM")
BOOTSTRAP_SEED = 917_301
BOOTSTRAP_REPLICATES = 10_000


def evaluate_hosd_classification(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
    identities: Sequence[str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if len(identities) != len(set(identities)) or len(identities) != len(labels):
        raise ValueError("HOSD metric identity population differs")
    inherited = evaluate_classification(logits, labels, split=split)
    return with_content_hash(
        {
            "contract": HOSD_METRICS_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "split": split,
            "identity_count": len(identities),
            "identity_order_sha256": canonical_sha256(list(identities)),
            "classification_metrics": inherited,
            "mean_log_selection_rejection": mean_log_selection_rejection(
                inherited
            ),
            "inherited_exact_metric_contract": inherited["content_hash"],
        }
    )


def tagging_utility(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float | None]:
    left = candidate["classification_metrics"]
    right = baseline["classification_metrics"]
    candidate_accuracy = float(left["macro_per_class_accuracy"])
    baseline_accuracy = float(right["macro_per_class_accuracy"])
    delta_accuracy = candidate_accuracy - baseline_accuracy
    baseline_error = 1.0 - baseline_accuracy
    candidate_rejection = mean_log_selection_rejection(left)
    baseline_rejection = mean_log_selection_rejection(right)
    return {
        "accuracy_difference": delta_accuracy,
        "relative_error_reduction": (
            None if baseline_error <= 0 else delta_accuracy / baseline_error
        ),
        "mean_log_rejection_difference": (
            candidate_rejection - baseline_rejection
        ),
    }


def offline_gap_closure(
    candidate: Mapping[str, Any],
    hlt_baseline: Mapping[str, Any],
    offline_baseline: Mapping[str, Any],
) -> dict[str, float | None]:
    c = candidate["classification_metrics"]
    h = hlt_baseline["classification_metrics"]
    o = offline_baseline["classification_metrics"]
    h_acc, o_acc = (
        float(h["macro_per_class_accuracy"]),
        float(o["macro_per_class_accuracy"]),
    )
    acc_denominator = o_acc - h_acc
    h_rejection = mean_log_selection_rejection(h)
    rejection_denominator = mean_log_selection_rejection(o) - h_rejection
    return {
        "accuracy": (
            None
            if acc_denominator <= 0
            else (
                float(c["macro_per_class_accuracy"]) - h_acc
            )
            / acc_denominator
        ),
        "mean_log_rejection": (
            None
            if rejection_denominator <= 0
            else (
                mean_log_selection_rejection(c) - h_rejection
            )
            / rejection_denominator
        ),
        "values_clipped": False,
    }


def feedback_decomposition(
    *,
    baseline: Mapping[str, Any],
    auxiliary: Mapping[str, Any],
    feedback: Mapping[str, Any],
    oracle_trained: Mapping[str, Any],
    oracle_substitution: Mapping[str, Any],
    unrestricted: Mapping[str, Any],
) -> dict[str, Any]:
    rows = {
        "auxiliary_gain": (auxiliary, baseline),
        "feedback_gain": (feedback, auxiliary),
        "oracle_room": (oracle_trained, feedback),
        "substitution": (oracle_substitution, feedback),
        "semantic_gain": (feedback, unrestricted),
    }
    return {
        name: {
            "accuracy": float(left["classification_metrics"]["macro_per_class_accuracy"])
            - float(right["classification_metrics"]["macro_per_class_accuracy"]),
            "mean_log_rejection": mean_log_selection_rejection(
                left["classification_metrics"]
            )
            - mean_log_selection_rejection(right["classification_metrics"]),
        }
        for name, (left, right) in rows.items()
    }


def target_component_statistics(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if predicted.shape != truth.shape or truth.shape != valid.shape:
        raise ValueError("target statistic arrays differ")
    rows = []
    for component in range(truth.shape[-1]):
        selected = valid[..., component].reshape(-1)
        x = predicted[..., component].reshape(-1)[selected]
        y = truth[..., component].reshape(-1)[selected]
        if len(x) == 0:
            rows.append({"component": component, "count": 0, "r2": None, "spearman": None})
            continue
        error = float(np.square(x - y).sum())
        variance = float(np.square(y - y.mean()).sum())
        r2 = None if variance == 0 and error == 0 else 0.0 if variance == 0 else 1 - error / variance
        if len(x) < 2:
            spearman = None
        else:
            xr, yr = _average_ranks(x), _average_ranks(y)
            spearman = _correlation(xr, yr)
        rows.append(
            {
                "component": component,
                "count": len(x),
                "r2": r2,
                "spearman": spearman,
                "mae": float(np.abs(x - y).mean()),
                "rmse": float(np.sqrt(np.square(x - y).mean())),
            }
        )
    defined = [row["spearman"] for row in rows if row["spearman"] is not None]
    return {
        "components": rows,
        "macro_spearman": None if not defined else float(np.mean(defined)),
        "defined_spearman_component_count": len(defined),
        "sampling_unit": "jet_after_per_jet_target_reduction",
    }


def categorical_target_statistics(
    logits: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    """Masked categorical metrics with the global 15-bin ECE convention."""

    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(target, dtype=np.int64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim < 2 or values.shape[:-1] != truth.shape or truth.shape != valid.shape:
        raise ValueError("categorical target arrays differ")
    selected_logits = values.reshape(-1, values.shape[-1])[valid.reshape(-1)]
    selected_truth = truth.reshape(-1)[valid.reshape(-1)]
    if len(selected_truth) == 0:
        return {
            "count": 0,
            "cross_entropy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "brier": None,
            "target_ece_15": None,
            "confusion_matrix": None,
        }
    if (
        bool((selected_truth < 0).any())
        or bool((selected_truth >= selected_logits.shape[1]).any())
        or not np.isfinite(selected_logits).all()
    ):
        raise ValueError("categorical target values differ")
    shifted = selected_logits - selected_logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    classes = int(values.shape[-1])
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (selected_truth, predicted), 1)
    recalls, f1s = [], []
    for class_index in range(classes):
        true_positive = int(confusion[class_index, class_index])
        support = int(confusion[class_index].sum())
        predicted_count = int(confusion[:, class_index].sum())
        if support:
            recalls.append(true_positive / support)
        denominator = 2 * true_positive + (predicted_count - true_positive) + (
            support - true_positive
        )
        if denominator:
            f1s.append(2 * true_positive / denominator)
    one_hot = np.eye(classes, dtype=np.float64)[selected_truth]
    brier = np.square(probabilities - one_hot).sum(axis=1).mean()
    confidence = probabilities.max(axis=1)
    correct = predicted == selected_truth
    edges = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    ece = 0.0
    bins = []
    for index in range(15):
        in_bin = (
            (confidence >= edges[index])
            & (
                confidence <= edges[index + 1]
                if index == 14
                else confidence < edges[index + 1]
            )
        )
        count = int(in_bin.sum())
        if count:
            mean_confidence = float(confidence[in_bin].mean())
            accuracy = float(correct[in_bin].mean())
            contribution = count / len(correct) * abs(mean_confidence - accuracy)
            ece += contribution
        else:
            mean_confidence, accuracy, contribution = None, None, 0.0
        bins.append(
            {
                "index": index,
                "left": float(edges[index]),
                "right": float(edges[index + 1]),
                "left_inclusive": True,
                "right_inclusive": index == 14,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "weighted_contribution": contribution,
            }
        )
    log_probability = np.log(
        np.clip(probabilities[np.arange(len(selected_truth)), selected_truth], 1e-300, 1)
    )
    return {
        "count": len(selected_truth),
        "cross_entropy": float(-log_probability.mean()),
        "balanced_accuracy": None if not recalls else float(np.mean(recalls)),
        "macro_f1": None if not f1s else float(np.mean(f1s)),
        "brier": float(brier),
        "target_ece_15": float(ece),
        "ece_bins": bins,
        "confusion_matrix": confusion.tolist(),
        "top_label_multiclass_ece": True,
        "empty_bin_contribution": 0.0,
    }


def build_hosd_paired_statistics(
    *,
    identities: Sequence[str],
    labels: np.ndarray,
    candidate_logits_by_seed: Mapping[int, np.ndarray],
    baseline_logits_by_seed: Mapping[int, np.ndarray],
    candidate_graph_id: str,
    baseline_graph_id: str,
    prediction_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    candidate_target_error_by_seed: Mapping[int, np.ndarray] | None = None,
    baseline_target_error_by_seed: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    ids = tuple(str(value) for value in identities)
    truth = np.asarray(labels, dtype=np.int64)
    seeds = tuple(sorted(int(value) for value in candidate_logits_by_seed))
    if (
        seeds != (202, 303, 404)
        or set(baseline_logits_by_seed) != set(seeds)
        or ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or truth.shape != (len(ids),)
    ):
        raise ValueError("HOSD confirmation paired population/seeds differ")
    counts = np.bincount(truth, minlength=10)
    if bool((counts == 0).any()) or len(set(counts.tolist())) != 1:
        raise ValueError("HOSD paired population must be class balanced")
    candidate_probabilities, baseline_probabilities = {}, {}
    per_seed = []
    for seed in seeds:
        candidate = np.asarray(candidate_logits_by_seed[seed], dtype=np.float64)
        baseline = np.asarray(baseline_logits_by_seed[seed], dtype=np.float64)
        if (
            candidate.shape != (len(ids), 10)
            or baseline.shape != candidate.shape
            or not np.isfinite(candidate).all()
            or not np.isfinite(baseline).all()
        ):
            raise ValueError("HOSD paired logits differ")
        candidate_probabilities[seed] = stable_probabilities(candidate)
        baseline_probabilities[seed] = stable_probabilities(baseline)
        candidate_correct = candidate.argmax(axis=1) == truth
        baseline_correct = baseline.argmax(axis=1) == truth
        candidate_terms, candidate_central = _rejection_terms(
            candidate_probabilities[seed], truth
        )
        baseline_terms, baseline_central = _rejection_terms(
            baseline_probabilities[seed], truth
        )
        per_seed.append(
            {
                "seed": seed,
                "candidate_accuracy": float(candidate_correct.mean()),
                "baseline_accuracy": float(baseline_correct.mean()),
                "accuracy_difference": float(
                    candidate_correct.mean() - baseline_correct.mean()
                ),
                "mean_log_rejection_difference": float(
                    candidate_terms.mean() - baseline_terms.mean()
                ),
                "mcnemar_candidate_only_correct": int(
                    (candidate_correct & ~baseline_correct).sum()
                ),
                "mcnemar_baseline_only_correct": int(
                    (~candidate_correct & baseline_correct).sum()
                ),
                "candidate_rejection": candidate_central,
                "baseline_rejection": baseline_central,
            }
        )
    class_indices = [np.flatnonzero(truth == index) for index in range(10)]
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    accuracy_samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    rejection_samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    per_signal_samples = np.empty(
        (BOOTSTRAP_REPLICATES, 18), dtype=np.float64
    )
    target_errors_enabled = (
        candidate_target_error_by_seed is not None
        or baseline_target_error_by_seed is not None
    )
    if target_errors_enabled:
        if (
            candidate_target_error_by_seed is None
            or baseline_target_error_by_seed is None
            or set(candidate_target_error_by_seed) != set(seeds)
            or set(baseline_target_error_by_seed) != set(seeds)
        ):
            raise ValueError("paired target-error seed coverage differs")
        target_differences = {}
        for seed in seeds:
            candidate_error = np.asarray(
                candidate_target_error_by_seed[seed], dtype=np.float64
            )
            baseline_error = np.asarray(
                baseline_target_error_by_seed[seed], dtype=np.float64
            )
            if (
                candidate_error.shape != (len(ids),)
                or baseline_error.shape != candidate_error.shape
                or not np.isfinite(candidate_error).all()
                or not np.isfinite(baseline_error).all()
            ):
                raise ValueError("paired per-jet target errors differ")
            target_differences[seed] = candidate_error - baseline_error
        target_error_samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    else:
        target_differences = {}
        target_error_samples = None
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [
                indices[
                    generator.integers(0, len(indices), size=len(indices))
                ]
                for indices in class_indices
            ]
        )
        sampled_truth = truth[sampled]
        seed_accuracy, seed_terms = [], []
        for seed in seeds:
            candidate_probability = candidate_probabilities[seed][sampled]
            baseline_probability = baseline_probabilities[seed][sampled]
            seed_accuracy.append(
                float(
                    (
                        candidate_probability.argmax(axis=1) == sampled_truth
                    ).mean()
                    - (
                        baseline_probability.argmax(axis=1) == sampled_truth
                    ).mean()
                )
            )
            seed_terms.append(
                _bootstrap_rejection_terms(
                    candidate_probability, sampled_truth
                )
                - _bootstrap_rejection_terms(
                    baseline_probability, sampled_truth
                )
            )
        accuracy_samples[replicate] = np.mean(seed_accuracy)
        per_signal_samples[replicate] = np.mean(seed_terms, axis=0)
        rejection_samples[replicate] = per_signal_samples[replicate].mean()
        if target_error_samples is not None:
            target_error_samples[replicate] = float(
                np.mean(
                    [
                        target_differences[seed][sampled].mean()
                        for seed in seeds
                    ]
                )
            )
    interval = lambda values: [
        float(value)
        for value in np.quantile(values, [0.025, 0.975], method="linear")
    ]
    payload = {
            "contract": HOSD_PAIRED_STATISTICS_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "candidate_graph_id": str(candidate_graph_id),
            "baseline_graph_id": str(baseline_graph_id),
            "identity_count": len(ids),
            "identity_order_sha256": canonical_sha256(list(ids)),
            "seeds": list(seeds),
            "prediction_hashes": {
                key: require_sha256(value, name=f"prediction.{key}")
                for key, value in sorted(prediction_hashes.items())
            },
            "per_seed": per_seed,
            "three_seed_accuracy_difference_mean": float(
                np.mean([row["accuracy_difference"] for row in per_seed])
            ),
            "three_seed_accuracy_difference_sample_std": float(
                np.std(
                    [row["accuracy_difference"] for row in per_seed], ddof=1
                )
            ),
            "paired_accuracy_difference_interval_95": interval(
                accuracy_samples
            ),
            "paired_mean_log_rejection_difference_interval_95": interval(
                rejection_samples
            ),
            "per_signal_log_rejection_difference_intervals_95": [
                interval(per_signal_samples[:, index]) for index in range(18)
            ],
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "sampling_unit": "canonical_jet_identity",
                "stratification": "within_each_balanced_class_original_count",
                "paired_across_models_targets_and_seeds": True,
                "quantiles": [0.025, 0.975],
                "quantile_method": "linear_r_equals_n_minus_1_times_q",
                "rejection_thresholds_recomputed_each_resample": True,
                "mean_log_terms_recomputed_per_resample": 18,
            },
        }
    if target_error_samples is not None:
        per_seed_target = [
            {
                "seed": seed,
                "paired_mean_target_error_difference": float(
                    target_differences[seed].mean()
                ),
            }
            for seed in seeds
        ]
        payload["paired_target_error"] = {
            "per_seed": per_seed_target,
            "three_seed_mean": float(
                np.mean(
                    [
                        row["paired_mean_target_error_difference"]
                        for row in per_seed_target
                    ]
                )
            ),
            "three_seed_sample_std": float(
                np.std(
                    [
                        row["paired_mean_target_error_difference"]
                        for row in per_seed_target
                    ],
                    ddof=1,
                )
            ),
            "interval_95": interval(target_error_samples),
            "sampling_unit": "jet_after_per_jet_target_reduction",
        }
    else:
        payload["paired_target_error"] = None
    return with_content_hash(payload)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = left - left.mean(), right - right.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return None if denominator == 0 else float((x @ y) / denominator)


def build_robustness_plan(
    *,
    graph_ids: Sequence[str],
    mechanism_summary_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if not graph_ids or len(graph_ids) != len(set(graph_ids)):
        raise ValueError("robustness graph set is empty or duplicated")
    rows = [
        {
            "row_id": f"ROBUST_{canonical_sha256([graph, profile, replica])[:16]}",
            "graph_id": graph,
            "degradation_profile": profile,
            "replica_policy": replica,
            "selection_eligible": False,
        }
        for graph in graph_ids
        for profile in DEGRADATION_PROFILES
        for replica in REPLICA_POLICIES
    ]
    return with_content_hash(
        {
            "contract": ROBUSTNESS_PLAN_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "mechanism_summary_sha256": require_sha256(
                mechanism_summary_sha256, name="mechanism_summary_sha256"
            ),
            "graph_ids": list(graph_ids),
            "degradation_profiles": list(DEGRADATION_PROFILES),
            "replica_policies": list(REPLICA_POLICIES),
            "rows": rows,
            "row_count": len(rows),
            "selection_reopened": False,
        }
    )


def build_robustness_summary(
    *,
    plan: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(plan, expected_contract=ROBUSTNESS_PLAN_CONTRACT)
    by_id = {row["row_id"]: row for row in results}
    expected = {row["row_id"] for row in plan["rows"]}
    if set(by_id) != expected:
        raise ValueError("robustness result coverage differs")
    expected_by_id = {row["row_id"]: row for row in plan["rows"]}
    for row_id, result in by_id.items():
        validate_content_hash(result, expected_contract=ROBUSTNESS_RESULT_CONTRACT)
        expected_row = expected_by_id[row_id]
        if (
            result.get("source") != dict(source)
            or result.get("robustness_plan_sha256") != plan["content_hash"]
            or any(
                result.get(key) != expected_row[key]
                for key in ("graph_id", "degradation_profile", "replica_policy")
            )
        ):
            raise ValueError("robustness result lineage differs")
    return with_content_hash(
        {
            "contract": ROBUSTNESS_SUMMARY_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "robustness_plan_sha256": plan["content_hash"],
            "result_hashes": {
                key: str(by_id[key]["content_hash"]) for key in sorted(by_id)
            },
            "rows": [by_id[key] for key in sorted(by_id)],
            "complete": True,
            "negative_results_reported": True,
            "selection_reopened": False,
        }
    )


def build_robustness_result(
    *,
    plan: Mapping[str, Any],
    row_id: str,
    identities: Sequence[str],
    labels: np.ndarray,
    logits: np.ndarray,
    subgroup_values: Mapping[str, np.ndarray],
    subgroup_edges: Mapping[str, Sequence[float]],
    prediction_sha256: str,
    export_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(plan, expected_contract=ROBUSTNESS_PLAN_CONTRACT)
    rows = {row["row_id"]: row for row in plan["rows"]}
    if row_id not in rows:
        raise ValueError("robustness row is absent from its plan")
    ids = tuple(str(value) for value in identities)
    truth = np.asarray(labels, dtype=np.int64)
    values = np.asarray(logits, dtype=np.float64)
    if (
        len(ids) != len(set(ids))
        or truth.shape != (len(ids),)
        or values.shape != (len(ids), 10)
    ):
        raise ValueError("robustness prediction population differs")
    overall = evaluate_hosd_classification(
        values,
        truth,
        split="design_confirm",
        identities=ids,
        source=source,
    )
    required = {
        "jet_pt",
        "abs_jet_eta",
        "valid_multiplicity",
        "valid_track_fraction",
    }
    if not required.issubset(subgroup_values) or not required.issubset(
        subgroup_edges
    ):
        raise ValueError("robustness subgroup covariates are incomplete")
    subgroups = {}
    for name, raw in sorted(subgroup_values.items()):
        coordinate = np.asarray(raw, dtype=np.float64)
        if coordinate.shape != (len(ids),) or not np.isfinite(coordinate).all():
            raise ValueError(f"robustness subgroup {name} differs")
        edges = np.asarray(subgroup_edges[name], dtype=np.float64)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or bool((np.diff(edges) <= 0).any())
        ):
            raise ValueError(f"robustness subgroup edges {name} differ")
        bins = []
        for index in range(len(edges) - 1):
            selected = (coordinate >= edges[index]) & (
                coordinate <= edges[index + 1]
                if index == len(edges) - 2
                else coordinate < edges[index + 1]
            )
            bins.append(
                {
                    "bin": index,
                    "left": float(edges[index]),
                    "right": float(edges[index + 1]),
                    "count": int(selected.sum()),
                    "classification_metrics": (
                        None
                        if not bool(selected.any())
                        else evaluate_classification(
                            values[selected],
                            truth[selected],
                            split="design_confirm",
                        )
                    ),
                }
            )
        subgroups[name] = {
            "edges": edges.tolist(),
            "bins": bins,
            "last_bin_right_inclusive": True,
        }
    return with_content_hash(
        {
            "contract": ROBUSTNESS_RESULT_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "robustness_plan_sha256": plan["content_hash"],
            **rows[row_id],
            "identity_count": len(ids),
            "identity_order_sha256": canonical_sha256(list(ids)),
            "prediction_sha256": require_sha256(
                prediction_sha256, name="prediction_sha256"
            ),
            "export_sha256": require_sha256(export_sha256, name="export_sha256"),
            "classification": overall,
            "subgroups": subgroups,
            "selection_eligible": False,
        }
    )


def build_hosd_report(
    *,
    title: str,
    artifact_hashes: Mapping[str, str],
    result_rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if not result_rows:
        raise ValueError("HOSD report requires result rows")
    checked = {
        key: require_sha256(value, name=f"artifact.{key}")
        for key, value in sorted(artifact_hashes.items())
    }
    artifact = with_content_hash(
        {
            "contract": HOSD_REPORT_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "title": str(title),
            "artifact_hashes": checked,
            "rows": [dict(row) for row in result_rows],
            "row_count": len(result_rows),
            "positive_and_negative_rows_included": True,
            "manual_scientific_text_required": False,
        }
    )
    lines = [
        f"# {title}",
        "",
        f"Report contract: `{artifact['content_hash']}`",
        "",
        (
            "| Graph | Split | Balanced accuracy | Difference vs H_BASE | "
            "Deployed parameters | Inference FLOPs (B1,N128) | "
            "Batch-1 median ms |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            (
                "| {graph} | {split} | {accuracy:.8f} | "
                "{difference:+.8f} | {parameters} | {flops} | "
                "{latency} |"
            ).format(
                graph=row["graph_id"],
                split=row["split"],
                accuracy=float(row["balanced_accuracy"]),
                difference=float(row["accuracy_difference_vs_h_base"]),
                parameters=row.get(
                    "deployed_trainable_parameters", "n/a"
                ),
                flops=row.get(
                    "analytical_inference_flops_by_batch", {}
                ).get("1", "n/a"),
                latency=(
                    "n/a"
                    if "latency" not in row
                    else f"{float(row['latency']['1']['median_milliseconds']):.6f}"
                ),
            )
        )
    lines.extend(
        [
            "",
            "All registered rows are shown, including null and negative gains.",
            "",
        ]
    )
    return artifact, "\n".join(lines)


__all__ = [
    "DEGRADATION_PROFILES",
    "REPLICA_POLICIES",
    "build_hosd_report",
    "build_hosd_paired_statistics",
    "build_robustness_plan",
    "build_robustness_summary",
    "build_robustness_result",
    "categorical_target_statistics",
    "evaluate_hosd_classification",
    "feedback_decomposition",
    "offline_gap_closure",
    "tagging_utility",
    "target_component_statistics",
]

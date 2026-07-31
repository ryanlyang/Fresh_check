from __future__ import annotations

import numpy as np
import pytest

from teacher_logit_reco.hlt_offline_structure_distillation import (
    build_hosd_report,
    build_robustness_plan,
    build_robustness_summary,
    build_robustness_result,
    build_hosd_paired_statistics,
    categorical_target_statistics,
    evaluate_hosd_classification,
    feedback_decomposition,
    offline_gap_closure,
    tagging_utility,
    target_component_statistics,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    with_content_hash,
)


SOURCE = {
    "commit": "a" * 40,
    "status_sha256": "b" * 64,
    "dirty": True,
    "status_hash_policy": "test",
}


def _metric(strength: float):
    labels = np.tile(np.arange(10), 4)
    logits = np.zeros((40, 10), dtype=np.float64)
    logits[np.arange(40), labels] = strength
    identities = [f"jet-{index:03d}" for index in range(40)]
    return evaluate_hosd_classification(
        logits, labels, split="design_confirm", identities=identities, source=SOURCE
    )


def test_exact_classification_utility_gap_and_feedback_decomposition():
    weak, middle, strong = _metric(0.0), _metric(1.0), _metric(2.0)
    utility = tagging_utility(middle, weak)
    assert utility["accuracy_difference"] >= 0
    closure = offline_gap_closure(middle, weak, strong)
    assert closure["values_clipped"] is False
    assert closure["accuracy"] is not None
    undefined = offline_gap_closure(middle, strong, weak)
    assert undefined["accuracy"] is None
    decomposition = feedback_decomposition(
        baseline=weak,
        auxiliary=middle,
        feedback=strong,
        oracle_trained=strong,
        oracle_substitution=middle,
        unrestricted=weak,
    )
    assert set(decomposition) == {
        "auxiliary_gain",
        "feedback_gain",
        "oracle_room",
        "substitution",
        "semantic_gain",
    }


def test_target_statistics_freeze_constant_and_tie_conventions():
    target = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    prediction = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    stats = target_component_statistics(
        prediction, target, np.ones_like(target, dtype=bool)
    )
    assert stats["components"][0]["r2"] == 0.0
    assert stats["components"][0]["spearman"] is None
    assert stats["components"][1]["r2"] == 1.0
    perfect_constant = target_component_statistics(
        target[:, :1], target[:, :1], np.ones((3, 1), dtype=bool)
    )
    assert perfect_constant["components"][0]["r2"] is None


def test_robustness_matrix_is_complete_and_never_reopens_selection():
    plan = build_robustness_plan(
        graph_ids=["H_BASE", "C_WINNER"],
        mechanism_summary_sha256="c" * 64,
        source=SOURCE,
    )
    assert plan["row_count"] == 2 * 9 * 3
    results = [
        build_robustness_result(
            plan=plan,
            row_id=row["row_id"],
            identities=[f"jet-{index:03d}" for index in range(40)],
            labels=np.tile(np.arange(10), 4),
            logits=np.zeros((40, 10)),
            subgroup_values={
                "jet_pt": np.arange(40, dtype=float),
                "abs_jet_eta": np.arange(40, dtype=float),
                "valid_multiplicity": np.arange(40, dtype=float),
                "valid_track_fraction": np.linspace(0, 1, 40),
            },
            subgroup_edges={
                "jet_pt": [-1, 20, 41],
                "abs_jet_eta": [-1, 20, 41],
                "valid_multiplicity": [-1, 20, 41],
                "valid_track_fraction": [-0.1, 0.5, 1.1],
            },
            prediction_sha256="d" * 64,
            export_sha256="e" * 64,
            source=SOURCE,
        )
        for row in plan["rows"]
    ]
    with pytest.raises(ValueError, match="coverage"):
        build_robustness_summary(plan=plan, results=results[:-1], source=SOURCE)
    summary = build_robustness_summary(
        plan=plan, results=results, source=SOURCE
    )
    assert summary["complete"]
    assert summary["negative_results_reported"]
    assert summary["selection_reopened"] is False


def test_report_includes_negative_rows_without_manual_text():
    rows = [
        {
            "graph_id": "H_BASE",
            "split": "final_test",
            "balanced_accuracy": 0.7,
            "accuracy_difference_vs_h_base": 0.0,
        },
        {
            "graph_id": "HOSD_NEGATIVE",
            "split": "final_test",
            "balanced_accuracy": 0.6,
            "accuracy_difference_vs_h_base": -0.1,
        },
    ]
    artifact, markdown = build_hosd_report(
        title="Fixture",
        artifact_hashes={"metrics": "d" * 64},
        result_rows=rows,
        source=SOURCE,
    )
    assert artifact["positive_and_negative_rows_included"]
    assert "HOSD_NEGATIVE" in markdown
    assert "-0.10000000" in markdown


def test_target_ece_brier_and_endpoint_contract():
    logits = np.asarray(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [0.0, 0.0],
        ]
    )
    result = categorical_target_statistics(
        logits, np.asarray([0, 1, 1]), np.asarray([True, True, True])
    )
    assert result["count"] == 3
    assert result["top_label_multiclass_ece"]
    assert len(result["ece_bins"]) == 15
    assert result["ece_bins"][-1]["right_inclusive"] is True
    assert result["brier"] >= 0


def test_three_seed_paired_bootstrap_includes_target_error_interval(monkeypatch):
    import teacher_logit_reco.hlt_offline_structure_distillation.metrics as module

    monkeypatch.setattr(module, "BOOTSTRAP_REPLICATES", 50)
    labels = np.repeat(np.arange(10), 2)
    identities = [f"jet-{index:03d}" for index in range(20)]
    base = np.zeros((20, 10), dtype=np.float64)
    candidate = base.copy()
    candidate[np.arange(20), labels] = 1.0
    seeds = (202, 303, 404)
    artifact = build_hosd_paired_statistics(
        identities=identities,
        labels=labels,
        candidate_logits_by_seed={seed: candidate for seed in seeds},
        baseline_logits_by_seed={seed: base for seed in seeds},
        candidate_graph_id="candidate",
        baseline_graph_id="baseline",
        prediction_hashes={f"seed_{seed}": "f" * 64 for seed in seeds},
        source=SOURCE,
        candidate_target_error_by_seed={
            seed: np.zeros(20) for seed in seeds
        },
        baseline_target_error_by_seed={
            seed: np.ones(20) for seed in seeds
        },
    )
    assert artifact["bootstrap"]["seed"] == 917301
    assert artifact["paired_target_error"]["three_seed_mean"] == -1.0
    assert artifact["paired_target_error"]["interval_95"] == [-1.0, -1.0]

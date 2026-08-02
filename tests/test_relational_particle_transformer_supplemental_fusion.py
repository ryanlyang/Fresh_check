from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.evaluate_relational_part_supplemental_fusion import (
    COMPARISON_RELATION_RUN_IDS,
    CONTROL_RUN_IDS,
    PAIR_FUSIONS,
    RELATION_RUN_IDS,
    TRACK_CHARGE_DENSITY_RUN_IDS,
    fit_temperature,
    fuse_equal_weight_logits,
    pair_fusion_definitions,
    pairwise_diversity,
)
from scripts.aggregate_relational_part_supplemental_fusion import (
    aggregate_seed_artifacts,
)


def test_temperature_fit_is_deterministic_and_reduces_val_stop_loss() -> None:
    generator = np.random.default_rng(1717)
    labels = np.tile(np.arange(3, dtype=np.int64), 200)
    logits = generator.normal(size=(len(labels), 3))
    logits[np.arange(len(labels)), labels] += 0.8
    logits *= 5.0

    first = fit_temperature(logits, labels)
    second = fit_temperature(logits, labels)

    assert first == second
    assert first["fit_split"] == "val_stop"
    assert first["temperature"] > 1.0
    assert (
        first["val_stop_cross_entropy_after"]
        < first["val_stop_cross_entropy_before"]
    )


def test_equal_weight_logit_fusion_centers_and_temperature_scales() -> None:
    logits = {
        "A": np.asarray([[3.0, 1.0], [2.0, 4.0]]),
        "B": np.asarray([[8.0, 4.0], [7.0, 3.0]]),
    }
    fused = fuse_equal_weight_logits(
        logits,
        ("A", "B"),
        inverse_temperatures={"A": 0.5, "B": 2.0},
    )
    expected_a = (logits["A"] - logits["A"].mean(1, keepdims=True)) * 0.5
    expected_b = (logits["B"] - logits["B"].mean(1, keepdims=True)) * 2.0

    np.testing.assert_allclose(fused, (expected_a + expected_b) / 2.0)
    np.testing.assert_allclose(fused.mean(axis=1), 0.0, atol=1e-15)


def test_pairwise_diversity_reports_disagreement_and_oracle_accuracy() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    predictions = {
        "A": np.asarray([0, 1, 1, 0], dtype=np.int64),
        "B": np.asarray([0, 0, 0, 0], dtype=np.int64),
        "C": np.asarray([1, 0, 1, 1], dtype=np.int64),
    }

    result = pairwise_diversity(predictions, labels, ("A", "B", "C"))

    assert result["oracle_any_member_correct_accuracy"] == 1.0
    assert result["all_members_correct_fraction"] == 0.0
    assert result["all_members_wrong_fraction"] == 0.0
    ab = result["pairwise"][0]
    assert (ab["left_run_id"], ab["right_run_id"]) == ("A", "B")
    assert ab["prediction_disagreement"] == 0.5
    assert ab["both_wrong_fraction"] == 0.25
    assert ab["exactly_one_correct_fraction"] == 0.5


def test_fusion_groups_and_fail_closed_cli_contract_are_fixed() -> None:
    assert CONTROL_RUN_IDS == (
        "RPT_BASE",
        "RPT_BASE_WIDE_MAX",
        "RPT_FULL_ZERO_REL",
    )
    assert RELATION_RUN_IDS == ("RPT_TRACK", "RPT_CHARGE", "RPT_PT")
    assert TRACK_CHARGE_DENSITY_RUN_IDS == (
        "RPT_TRACK",
        "RPT_CHARGE",
        "RPT_DENSITY",
    )
    assert COMPARISON_RELATION_RUN_IDS == {
        "track_charge_pt": RELATION_RUN_IDS,
        "track_charge_density": TRACK_CHARGE_DENSITY_RUN_IDS,
    }
    assert set(PAIR_FUSIONS.values()) == {
        ("RPT_TRACK", "RPT_CHARGE"),
        ("RPT_TRACK", "RPT_PT"),
        ("RPT_CHARGE", "RPT_PT"),
    }
    assert pair_fusion_definitions(TRACK_CHARGE_DENSITY_RUN_IDS) == {
        "RPT_TRACK_CHARGE_LOGIT_FUSION": ("RPT_TRACK", "RPT_CHARGE"),
        "RPT_TRACK_DENSITY_LOGIT_FUSION": ("RPT_TRACK", "RPT_DENSITY"),
        "RPT_CHARGE_DENSITY_LOGIT_FUSION": ("RPT_CHARGE", "RPT_DENSITY"),
    }

    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "evaluate_relational_part_supplemental_fusion.py"
    ).read_text(encoding="utf-8")
    assert "validate_campaign_source" in source
    assert "--allow-source-status-drift" in source
    assert "supplemental fusion source commit differs from campaign" in source
    assert '"official_campaign_workers_affected": False' in source
    assert "val_select replay differs" in source
    assert "load_region_tree_split" in source
    assert '"final_test_opened": False' in source
    assert "build_final_test_loader" not in source
    assert "target_efficiency=0.75" in source
    assert '"eligible_for_model_selection": False' in source
    assert '"track_charge_density"' in source


def _fake_seed_fusion(seed: int, difference: float, false_positives: int) -> dict:
    control_accuracy = 0.74 + seed / 1_000_000
    relation_accuracy = control_accuracy + difference

    def metrics(accuracy: float) -> dict:
        return {
            "accuracy": accuracy,
            "cross_entropy": 0.7,
            "brier_score": 0.35,
            "ece_15_bin_top_label": {"value": 0.02},
        }

    def rejection() -> dict:
        return {
            "H4q": {
                "qcd_false_positive_count": false_positives,
                "qcd_support": 100,
            }
        }

    return {
        "contract": "relational_part_posthoc_fusion_comparison_v2",
        "comparison_id": "track_charge_density",
        "content_hash": f"{seed:064d}",
        "seed": seed,
        "official_campaign_metric": False,
        "eligible_for_model_selection": False,
        "final_test_opened": False,
        "campaign_spec_sha256": "a" * 64,
        "global_determinism_sha256": "b" * 64,
        "hlt_cache_hashes": {"stack_val": "c" * 64},
        "event_identity_hashes": {"val_select": "d" * 64},
        "class_order": ["QCD", "H4q"],
        "control_member_run_ids": [
            "RPT_BASE",
            "RPT_BASE_WIDE_MAX",
            "RPT_FULL_ZERO_REL",
        ],
        "relation_member_run_ids": [
            "RPT_TRACK",
            "RPT_CHARGE",
            "RPT_DENSITY",
        ],
        "calibration_split": "val_stop",
        "evaluation_split": "val_select",
        "fusion_results": {
            "CONTROL_LOGIT_FUSION": {
                "primary_metrics": metrics(control_accuracy),
                "primary_qcd_signal_rejection_at_0p75": rejection(),
            },
            "RELATION_LOGIT_FUSION": {
                "primary_metrics": metrics(relation_accuracy),
                "primary_qcd_signal_rejection_at_0p75": rejection(),
            },
        },
        "primary_relation_minus_control_paired_statistics": {
            "paired_absolute_accuracy_difference": difference,
            "per_class_paired_accuracy_difference": {"QCD": difference},
        },
    }


def test_three_seed_fusion_aggregation_is_matched_and_pools_false_positives() -> None:
    artifacts = [
        _fake_seed_fusion(101, 0.001, 1),
        _fake_seed_fusion(202, 0.002, 2),
        _fake_seed_fusion(303, -0.001, 0),
    ]

    result = aggregate_seed_artifacts(artifacts)

    comparison = result["relation_minus_control_accuracy"]
    assert comparison["per_seed"] == {
        "101": 0.001,
        "202": 0.002,
        "303": -0.001,
    }
    assert comparison["mean"] == np.mean([0.001, 0.002, -0.001])
    assert comparison["seeds_relation_fusion_beats_control_fusion"] == 2
    pooled = result["relation_fusion_pooled_qcd_rejection_at_0p75"]["H4q"]
    assert pooled["pooled_qcd_false_positive_count"] == 3
    assert pooled["pooled_qcd_support"] == 300
    assert pooled["pooled_background_rejection"] == 100.0


def test_density_fusion_worker_fixes_seed_array_and_validation_only_comparison() -> None:
    source = (
        Path(__file__).parents[1]
        / "sbatch"
        / "run_relational_part_supplemental_fusion_density.sh"
    ).read_text(encoding="utf-8")

    assert "seeds=(101 202 303)" in source
    assert "--comparison-id track_charge_density" in source
    assert "--allow-source-status-drift" in source
    assert '--source-root "${PINNED_SOURCE_ROOT}"' in source
    assert "final_test" not in source

    aggregate_source = (
        Path(__file__).parents[1]
        / "sbatch"
        / "run_aggregate_relational_part_supplemental_fusion_density.sh"
    ).read_text(encoding="utf-8")
    assert "aggregate_relational_part_supplemental_fusion.py" in aggregate_source
    assert '--source-root "${PINNED_SOURCE_ROOT}"' in aggregate_source

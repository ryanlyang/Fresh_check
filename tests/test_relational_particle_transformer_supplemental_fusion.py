from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.evaluate_relational_part_supplemental_fusion import (
    CONTROL_RUN_IDS,
    PAIR_FUSIONS,
    RELATION_RUN_IDS,
    fit_temperature,
    fuse_equal_weight_logits,
    pairwise_diversity,
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
    assert set(PAIR_FUSIONS.values()) == {
        ("RPT_TRACK", "RPT_CHARGE"),
        ("RPT_TRACK", "RPT_PT"),
        ("RPT_CHARGE", "RPT_PT"),
    }

    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "evaluate_relational_part_supplemental_fusion.py"
    ).read_text(encoding="utf-8")
    assert "validate_campaign_source" in source
    assert "val_select replay differs" in source
    assert "load_region_tree_split" in source
    assert '"final_test_opened": False' in source
    assert "build_final_test_loader" not in source
    assert "target_efficiency=0.75" in source
    assert '"eligible_for_model_selection": False' in source

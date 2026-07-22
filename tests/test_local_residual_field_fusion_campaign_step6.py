from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block
from jetclass_fresh.jetclass_data import JetIdentity, LABEL_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    LOCAL_RESIDUAL_FIELD_FUSION_TEST_STATUS,
    audit_local_residual_field_pilot_metric_reproduction,
    freeze_binary_projection_thresholds,
    local_residual_field_binary_projection_metrics,
    local_residual_field_complementarity_metrics,
    local_residual_field_multiclass_metrics,
    paired_binary_projection_bootstrap,
    paired_multiclass_bootstrap,
)
from teacher_logit_reco.local_particle_residual_field import fusion_sources as source_module
from tests.test_local_residual_field_fusion_campaign_step3 import _OracleFreeModel, build_source_fixture


REPO_ROOT = Path(__file__).resolve().parents[1]


def _balanced_logits(rows_per_class: int = 4) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(len(LABEL_NAMES), dtype=np.int64), rows_per_class)
    logits = np.full((len(labels), len(LABEL_NAMES)), -1.0, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 2.0
    return logits, labels


def test_step6_shared_multiclass_metrics_include_auc_calibration_and_brier() -> None:
    logits, labels = _balanced_logits()
    metrics = local_residual_field_multiclass_metrics(logits, labels, label_names=LABEL_NAMES)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_one_vs_rest_auc"] == 1.0
    assert len(metrics["per_class"]) == 10
    assert len(metrics["confusion_matrix"]) == 10
    assert metrics["expected_calibration_error"] >= 0.0
    assert metrics["brier_score"] >= 0.0


def test_step6_rejection_has_counts_intervals_and_distinct_frozen_thresholds() -> None:
    logits, labels = _balanced_logits(rows_per_class=6)
    stack = local_residual_field_binary_projection_metrics(logits, labels, label_names=LABEL_NAMES)
    thresholds = freeze_binary_projection_thresholds(stack)
    shifted = logits.copy()
    shifted[labels == 0, 3] += 8.0
    final = local_residual_field_binary_projection_metrics(
        shifted,
        labels,
        label_names=LABEL_NAMES,
        frozen_thresholds=thresholds,
    )

    within = final["projections"]["QCD_vs_Hgg"]["operating_points"]["signal_efficiency_0.50"]
    frozen = final["projections"]["QCD_vs_Hgg"]["frozen_threshold_operating_points"]["signal_efficiency_0.50"]
    assert within["threshold_convention"] == "within_split_matched_efficiency"
    assert frozen["threshold_convention"] == "stack_val_frozen_threshold"
    assert frozen["qcd_false_positive_count"] == 6
    assert frozen["false_positive_rate_interval_95"][0] >= 0.0
    assert frozen["background_rejection_lower_bound_95"] > 0.0


def test_step6_complementarity_and_deterministic_paired_bootstrap() -> None:
    logits_a, labels = _balanced_logits(rows_per_class=4)
    logits_b = logits_a.copy()
    logits_b[0, 0] = -2.0
    logits_b[0, 1] = 3.0
    complementarity = local_residual_field_complementarity_metrics(
        logits_a, logits_b, labels, label_names=LABEL_NAMES, member_a="A0", member_b="P7b"
    )
    first = paired_multiclass_bootstrap(logits_a, logits_b, labels, label_names=LABEL_NAMES, seed=17)
    second = paired_multiclass_bootstrap(logits_a, logits_b, labels, label_names=LABEL_NAMES, seed=17)

    assert complementarity["disagreement_count"] == 1
    assert complementarity["loss_on_a_correct_count"] == 1
    assert complementarity["gain_on_a_error_count"] == 0
    assert len(complementarity["per_class"]) == 10
    assert first["replicates"] == 1000
    assert first["sampled_index_hash"] == second["sampled_index_hash"]
    with pytest.raises(ValueError, match="at least 1000"):
        paired_multiclass_bootstrap(logits_a, logits_b, labels, label_names=LABEL_NAMES, replicates=999)


def test_step6_binary_bootstrap_is_paired_stratified_and_deterministic() -> None:
    logits_a, labels = _balanced_logits(rows_per_class=5)
    logits_b = logits_a.copy()
    logits_b[labels == 0, 3] += np.linspace(0.0, 6.0, 5, dtype=np.float32)
    first = paired_binary_projection_bootstrap(
        logits_a, logits_b, labels, label_names=LABEL_NAMES, signal_name="Hgg", seed=29
    )
    second = paired_binary_projection_bootstrap(
        logits_a, logits_b, labels, label_names=LABEL_NAMES, signal_name="Hgg", seed=29
    )

    assert first["stratified_by_class"] is True
    assert first["replicates"] == 1000
    assert first["sampled_index_hash"] == second["sampled_index_hash"]
    assert len(first["log_smoothed_fpr_delta_b_minus_a"]["interval_95"]) == 2


def test_step6_reproduction_audit_requires_confirmation_and_is_nonselectable(tmp_path: Path) -> None:
    config, payload = build_source_fixture(tmp_path)
    audit = source_module.audit_fusion_source_artifacts(
        config,
        model_loader=lambda path, device="cpu": (_OracleFreeModel(), payload),
    )
    with pytest.raises(ValueError, match="explicit confirmation"):
        audit_local_residual_field_pilot_metric_reproduction(
            config.output_path,
            output_path=tmp_path / "metrics.json",
            confirm_exploratory_final_test=False,
        )

    logits, labels = _balanced_logits(rows_per_class=2)
    logits[0, 3] = 5.0
    identities = [JetIdentity(file="final.root", entry=index, label=int(label)) for index, label in enumerate(labels)]
    for member in ("A0", "P7b"):
        save_prediction_block(
            PredictionBlock(
                model_name=member,
                split="final_test",
                logits=logits,
                probs=np.zeros_like(logits),
                labels=labels,
                jet_ids=identities,
                metadata={
                    "runtime_inputs": "HLT_only",
                    "uses_true_fields": False,
                    "uses_offline_particles": False,
                    "uses_teacher_logits_at_runtime": False,
                    "deployable": True,
                },
            ),
            config.a0_prediction_dir,
        )
    expectations = {
        member: {
            "accuracy": 0.95,
            "hgg_qcd_50_false_positives": 1,
            "qcd_support": 2,
            "hgg_qcd_50_rejection": 2.0,
        }
        for member in ("A0", "P7b")
    }
    report = audit_local_residual_field_pilot_metric_reproduction(
        config.output_path,
        output_path=tmp_path / "metric_reproduction.json",
        confirm_exploratory_final_test=True,
        expectations=expectations,
    )

    assert report["ok"] is True
    assert report["selection_allowed"] is False
    assert report["test_status"] == LOCAL_RESIDUAL_FIELD_FUSION_TEST_STATUS
    assert report["members"]["A0"]["checks"]["all_nine_qcd_projections"] is True
    assert report["complementarity"]["n_jets"] == 20


def test_step6_metric_slurm_requires_explicit_final_confirmation_and_tigris_settings() -> None:
    shell = (REPO_ROOT / "sbatch" / "run_validate_local_residual_field_fusion_metrics.sh").read_text(encoding="utf-8")
    assert "#SBATCH --account=reu-aisocial" in shell
    assert "export PYTHONNOUSERSITE=1" in shell
    assert "CONFIRM_FINAL_TEST" in shell
    assert "--confirm-exploratory-final-test" in shell

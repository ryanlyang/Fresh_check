from __future__ import annotations

import csv
from pathlib import Path

from teacher_logit_reco.local_particle_residual_field import fusion_campaign_report as report_module
from teacher_logit_reco.local_particle_residual_field import fusion_bootstrap_audit as bootstrap_module
from teacher_logit_reco.local_particle_residual_field.fusion_campaign import default_fusion_candidate_specs, stable_fusion_json_hash


def _multiclass(accuracy: float) -> dict:
    return {
        "n_jets": 100, "accuracy": accuracy, "cross_entropy": 0.4,
        "macro_one_vs_rest_auc": 0.9, "macro_per_class_accuracy": accuracy,
        "expected_calibration_error": 0.02, "brier_score": 0.1,
        "confusion_matrix": [[1]], "per_class": [],
    }


def _binary() -> dict:
    point = {
        "available": True, "target_signal_efficiency": 0.5, "realized_signal_efficiency": 0.5,
        "threshold": 0.2, "signal_pass_count": 10, "signal_support": 20,
        "qcd_false_positive_count": 2, "qcd_support": 50, "false_positive_rate": 0.04,
        "false_positive_rate_interval_95": [0.01, 0.1], "background_rejection": 25.0,
        "background_rejection_lower_bound_95": 10.0,
    }
    return {"projections": {"QCD_vs_Hgg": {"positive_class_name": "Hgg",
            "operating_points": {"signal_efficiency_0.50": point},
            "frozen_threshold_operating_points": {"signal_efficiency_0.50": {**point, "threshold_convention": "stack_val_frozen_threshold"}}}}}


def _bootstrap() -> dict:
    return {
        "replicates": 1000, "seed": 7319, "sampled_index_hash": "b" * 64,
        "accuracy_delta_b_minus_a": {"estimate": 0.01, "mean": 0.01, "interval_95": [0.0, 0.02]},
        "cross_entropy_delta_b_minus_a": {"estimate": -0.01, "mean": -0.01, "interval_95": [-0.02, 0.0]},
    }


def _binary_bootstrap() -> dict:
    row = {"estimate": -0.01, "interval_95": [-0.02, 0.0]}
    return {
        "replicates": 1000, "seed": 7321, "sampled_index_hash": "c" * 64,
        "false_positive_rate_delta_b_minus_a": row,
        "log_smoothed_fpr_delta_b_minus_a": row,
        "smoothed_rejection_ratio_b_over_a": {"estimate": 1.2, "interval_95": [1.0, 1.4]},
    }


def _complementarity(member_b: str) -> dict:
    return {
        "member_a": "A0", "member_b": member_b, "n_jets": 100, "disagreement_count": 12,
        "disagreement_rate": 0.12, "both_correct": 75, "a_only_correct": 5,
        "b_only_correct": 7, "both_wrong": 13, "error_overlap_jaccard": 0.52,
        "gain_on_a_error_count": 7, "gain_on_a_error_rate": 0.35,
        "loss_on_a_correct_count": 5, "loss_on_a_correct_rate": 0.0625,
        "flattened_logit_correlation": 0.8, "flattened_probability_correlation": 0.82,
        "per_class": [{"class_index": 0, "class_name": "QCD", "support": 20,
                       "disagreement_rate": 0.1, "logit_correlation": 0.8, "probability_correlation": 0.8}],
    }


def test_step11_writes_complete_atomic_report_with_identified_fusion_rows(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "campaign"
    selection_path = root / "selection" / "selected_fusion.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text("{}", encoding="utf-8")
    g0_fit = root / "g0_fit.json"
    g0_fit.write_text('{"g0_development_reproduction":{"ok":true,"reference_id":"G0"}}', encoding="utf-8")
    selected = {
        "campaign_id": "unit", "artifact_hash": "a" * 64, "source_artifact_audit_hash": "d" * 64,
        "representation_stability_union": [],
        "selections": [
            {"group_id": group, "candidate_id": candidate, "champion_role": role,
             "member_ids": members, "hyperparameters": {},
             "selection_metrics": {"multiclass": _multiclass(accuracy), "binary_projection": _binary()}}
            for group, candidate, members, accuracy in (
                ("F_method", "L0_mean_logits", ["A0", "P7b"], 0.82),
                ("F_seed", "L1_mean_probs", ["A0", "A0_seed1"], 0.81),
            )
            for role in ("accuracy_champion", "rejection_champion")
        ],
    }
    final_results = []
    for group, candidate, member_b, accuracy in (
        ("F_method", "L0_mean_logits", "P7b", 0.825),
        ("F_seed", "L1_mean_probs", "A0_seed1", 0.812),
    ):
        final_results.append({
            "run_id": f"{group}/{candidate}", "group_id": group, "candidate_id": candidate,
            "champion_roles": ["accuracy_champion", "rejection_champion"], "multiclass": _multiclass(accuracy),
            "binary_projection": _binary(), "member_complementarity": _complementarity(member_b),
            "paired_bootstrap_vs_A0": _bootstrap(), "paired_binary_bootstrap_vs_A0": {"Hgg": _binary_bootstrap()},
            "runtime_inputs": "HLT_only", "deployable": True, "uses_true_fields": False,
            "uses_offline_particles": False, "uses_teacher_logits_at_runtime": False,
        })
    final = {
        "selected_fusion_artifact_hash": selected["artifact_hash"], "final_test_status": "exploratory",
        "runtime_inputs": "HLT_only", "deployable": True, "uses_true_fields": False,
        "uses_offline_particles": False, "uses_teacher_logits_at_runtime": False,
        "member_metrics": {member: {
                               "multiclass": _multiclass(acc), "checkpoint_hash": member * 8,
                               "runtime_inputs": "HLT_only", "deployable": True,
                               "uses_true_fields": False, "uses_offline_particles": False,
                               "uses_teacher_logits_at_runtime": False,
                           }
                           for member, acc in (("A0", 0.80), ("A0_seed1", 0.805), ("P7b", 0.81))},
        "selected_results": final_results,
    }
    replay = {"source_selected_fusion_artifact_hash": selected["artifact_hash"], "candidate_id": "L0_mean_logits",
              "hyperparameter_search_performed": False, "final_test_opened": False,
              "metrics": {"stack_val": {"multiclass": _multiclass(0.808)}}}
    runtime = {
        "selected_fusion_artifact_hash": selected["artifact_hash"],
        "final_evaluation_sha256": "f" * 64,
        "runtime_inputs": "HLT_only", "deployable": True, "uses_true_fields": False,
        "uses_offline_particles": False, "uses_teacher_logits_at_runtime": False,
        "member_rows": [{
            "run_id": "member/A0", "member_id": "A0", "median_batch_latency_ms": 2.0,
            "runtime_inputs": "HLT_only", "deployable": True, "uses_true_fields": False,
            "uses_offline_particles": False, "uses_teacher_logits_at_runtime": False,
        }],
        "fusion_rows": [{"run_id": row["run_id"], "group_id": row["group_id"], "candidate_id": row["candidate_id"],
                         "median_batch_latency_ms": 5.0, "runtime_inputs": "HLT_only", "deployable": True,
                         "uses_true_fields": False, "uses_offline_particles": False,
                         "uses_teacher_logits_at_runtime": False}
                        for row in final_results],
    }
    bootstrap_audit = dict(runtime)
    candidates = {}
    for group in ("F_method", "F_seed"):
        for index, spec in enumerate(default_fusion_candidate_specs()):
            candidates[(group, spec.candidate_id)] = {
                "group_id": group, "candidate_id": spec.candidate_id, "family": spec.family, "phase": "complete",
                "head_seeds": [], "trainable_parameter_count": index, "selected_hyperparameters": {},
                "metrics": {"stack_val": {"multiclass": _multiclass(0.7 + index / 1000)}},
                "artifact_hash": str(index), "fit_artifacts": [{"path": str(g0_fit)}],
            }
    monkeypatch.setattr(report_module, "load_selected_fusion_set", lambda _path: selected)
    monkeypatch.setattr(report_module, "_validate_selection_dependencies", lambda *_args: ({"split_manifest_hash": "m", "hlt_content_hashes": {}}, {}))
    monkeypatch.setattr(report_module, "_candidate_reports", lambda *_args: candidates)
    monkeypatch.setattr(report_module, "sha256_file", lambda _path: "f" * 64)
    monkeypatch.setattr(
        report_module, "_validated_metric_reproduction",
        lambda *_args, **_kwargs: {"audit_hash": "e" * 64},
    )
    monkeypatch.setattr(report_module, "_validated_hashed_json", lambda path, contract: (
        final if path.name == "final_evaluation.json" else replay if path.name == "recipe_replay.json"
        else bootstrap_audit if path.name == "bootstrap_audit.json" else runtime
    ))

    report = report_module.write_local_residual_field_fusion_campaign_report(selection_path)
    output = root / "final_report"
    assert report["ok"] is True
    assert {path.name for path in output.iterdir()} == set(report_module.FINAL_REPORT_FILENAMES)
    with (output / "selected_fusion_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["run_id"] and row["group_id"] and row["candidate_id"] for row in rows)
    assert "Best raw member" in (output / "summary.md").read_text(encoding="utf-8")
    assert "oracle diagnostics are reported separately" in (output / "summary.md").read_text(encoding="utf-8")


def test_step11_refuses_existing_report_directory(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "campaign" / "selection" / "selected_fusion.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    (tmp_path / "campaign" / "final_report").mkdir()
    monkeypatch.setattr(report_module, "load_selected_fusion_set", lambda _path: (_ for _ in ()).throw(FileExistsError("immutable")))
    try:
        report_module.write_local_residual_field_fusion_campaign_report(path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable report rerun should fail")


def test_step11_bootstrap_gate_requires_all_headline_stratified_resamples(tmp_path: Path, monkeypatch) -> None:
    selected = {"campaign_id": "unit", "artifact_hash": "a" * 64}
    payload = {"replicates": 1000, "sampled_index_hash": "b" * 64, "stratified_by_class": True}
    final = {
        "ok": True, "contract": "local_residual_field_fusion_final_evaluation_v1",
        "selected_fusion_artifact_hash": selected["artifact_hash"],
        "runtime_inputs": "HLT_only", "deployable": True, "uses_true_fields": False,
        "uses_offline_particles": False, "uses_teacher_logits_at_runtime": False,
        "selected_results": [{
            "run_id": "F_method/L0_mean_logits", "group_id": "F_method", "candidate_id": "L0_mean_logits",
            "paired_bootstrap_vs_A0": payload,
            "paired_binary_bootstrap_vs_A0": {signal: payload for signal in bootstrap_module.FUSION_HEADLINE_SIGNALS},
            "runtime_inputs": "HLT_only", "deployable": True, "uses_true_fields": False,
            "uses_offline_particles": False, "uses_teacher_logits_at_runtime": False,
        }],
    }
    final["artifact_hash"] = stable_fusion_json_hash(final)
    written = {}
    monkeypatch.setattr(bootstrap_module, "load_selected_fusion_set", lambda _path: selected)
    monkeypatch.setattr(bootstrap_module, "_read_json", lambda _path: final)
    monkeypatch.setattr(bootstrap_module, "sha256_file", lambda _path: "c" * 64)
    monkeypatch.setattr(bootstrap_module, "_atomic_json", lambda path, report: written.update(path=path, report=report))
    report = bootstrap_module.audit_selected_fusion_bootstraps(tmp_path / "selection" / "selected_fusion.json")
    assert report["ok"] is True
    assert report["rows"][0]["minimum_binary_replicates"] == 1000
    assert written["path"].name == "bootstrap_audit.json"

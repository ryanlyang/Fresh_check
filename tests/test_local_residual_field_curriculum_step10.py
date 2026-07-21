from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.check_local_residual_curriculum_pilot_gate as pilot_gate_script
import scripts.validate_local_residual_curriculum_reused_inputs as reused_inputs_script
from teacher_logit_reco.local_particle_residual_field.curriculum import (
    LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
)
from teacher_logit_reco.local_particle_residual_field.curriculum_campaign import (
    LOCAL_RESIDUAL_FIELD_STUDENT_SELECTION_CONTRACT,
    curve_shape_metrics,
    evaluate_pilot_gate,
    select_best_curriculum_student,
    select_curriculum_consumer,
)


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "sbatch"


def _read(name: str) -> str:
    return (SBATCH / name).read_text(encoding="utf-8")


def _curve(values: dict[float, float]) -> dict[str, dict[str, float]]:
    return {str(alpha): {"accuracy": accuracy, "cross_entropy": 0.7} for alpha, accuracy in values.items()}


def _alpha_report(consumer: str, model: dict[float, float], stack: dict[float, float]) -> dict:
    return {
        "ok": True,
        "run_id": "D_alpha_eval_Ofull" if consumer == "Ofull" else "D_alpha_eval_Orobust",
        "consumer_id": consumer,
        "model_val_alpha_curve": _curve(model),
        "stack_val_alpha_curve": _curve(stack),
    }


def test_step10_requested_submitters_and_job_runners_exist() -> None:
    for name in (
        "submit_lprf_curriculum_pilot.sh",
        "submit_lprf_curriculum_tigris_pilot.sh",
        "submit_lprf_curriculum_tigris_rebuild_and_pilot.sh",
        "submit_lprf_curriculum_highdata.sh",
        "submit_lprf_curriculum_tigris_highdata.sh",
        "run_evaluate_local_residual_oracle_alpha.sh",
        "run_select_local_residual_curriculum_consumer.sh",
        "run_train_local_residual_field_curriculum_student.sh",
        "run_select_local_residual_curriculum_student.sh",
        "run_predict_selected_local_residual_curriculum_student.sh",
        "run_validate_local_residual_curriculum_reused_inputs.sh",
    ):
        assert (SBATCH / name).is_file(), name


def test_step10_pilot_submitter_has_explicit_stages_guardrails_and_dependencies() -> None:
    text = _read("submit_lprf_curriculum_pilot.sh")
    for stage in ("stage1a", "select_consumer", "stage1b", "full_first_stage"):
        assert stage in text
    assert "first_stage_pilot" in text
    assert "O0 Ofull Orobust_light" in text
    assert "D_alpha_eval_Ofull D_alpha_eval_Orobust" in text
    assert "P2 P4 P7a P7b" in text
    assert "Q0 Q3" in text
    assert "G0" in text
    assert "LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_TOTAL:=12" in text
    assert "LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_BEFORE_SELECTOR:=6" in text
    assert "LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_AFTER_SELECTOR:=6" in text
    assert '--dependency="afterok:${dependency}"' in text
    assert 'selector_dep="$(join_colon "${ofull_alpha_jid}" "${robust_alpha_jid}")"' in text
    assert 'stage1b_parent="$(join_colon "${input_audit_jid}" "${selector_jid}")"' in text
    assert "selected_consumer.json" in text
    assert "SELECTED_AT_RUNTIME" not in text
    assert "LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_FULL_FAMILY" in text


def test_step10_final_test_is_cached_only_after_best_p_is_frozen() -> None:
    submitter = _read("submit_lprf_curriculum_pilot.sh")
    selected_runner = _read("run_predict_selected_local_residual_curriculum_student.sh")
    fusion = _read("run_local_residual_field_fusion.sh")

    assert "LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS=stack_train stack_val" in submitter
    assert "lprf_predict_selected_P_final" in submitter
    assert 'selected_final_dep="$(join_colon "${best_p_jid}" "${prediction_jobs[P2]}"' in submitter
    assert "LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS=final_test" in selected_runner
    assert "selected_curriculum_student.json" in selected_runner
    assert "LOCAL_RESIDUAL_FIELD_FUSION_MODES=uniform_logit_mean" in submitter
    assert "LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON" in fusion
    assert 'G0:A0,${selected_p}' in fusion


def test_step10_stage1b_runner_reads_selector_and_keeps_q0_oracle_path_disabled() -> None:
    text = _read("run_train_local_residual_field_curriculum_student.sh")
    assert "--selected-consumer-json" in text
    assert '["selected_consumer_id"]' in text
    assert 'if [[ "${RUN_ID}" != Q0 ]]' in text
    assert "--oracle-teacher-checkpoint" in text
    assert "P7b" in text and "student-warm-start-checkpoint" in text
    assert "--predictor-warm-start-checkpoint" in text


def test_step10_tigris_wrappers_use_full_account_and_disable_user_site() -> None:
    for name in (
        "submit_lprf_curriculum_tigris_pilot.sh",
        "submit_lprf_curriculum_tigris_highdata.sh",
        "submit_lprf_curriculum_tigris_rebuild_and_pilot.sh",
    ):
        text = _read(name)
        assert "LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT:=reu-aisocial" in text
        assert "LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT:=reu-aisoc}" not in text
        assert "PYTHONNOUSERSITE:=1" in text
        assert "export PROJECT_DIR PD10_DATA_DIR OUTPUT_ROOT CONDA_BASE CONDA_ENV PYTHONNOUSERSITE DEVICE" in text
        assert "gpu:gh200:1" in text


def test_step10_rebuild_wrapper_queues_full_first_stage_after_a0_and_c0() -> None:
    wrapper = _read("submit_lprf_curriculum_tigris_rebuild_and_pilot.sh")
    bootstrap = _read("submit_local_particle_residual_field_experiment.sh")
    curriculum = _read("submit_lprf_curriculum_pilot.sh")
    audit = _read("run_validate_local_residual_curriculum_reused_inputs.sh")

    assert "LOCAL_RESIDUAL_FIELD_RECON_RUN_IDS=C0" in wrapper
    assert "LOCAL_RESIDUAL_FIELD_TAGGER_RUN_IDS=A0" in wrapper
    for disabled in ("TEACHER_LOGITS", "PREDICTIONS", "FUSION", "REPORT"):
        assert f"LOCAL_RESIDUAL_FIELD_SUBMIT_{disabled}=0" in wrapper
    assert "LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE=full_first_stage" in wrapper
    assert 'LOCAL_RESIDUAL_FIELD_CURRICULUM_UPSTREAM_DEPENDENCY="${bootstrap_dependency}"' in wrapper
    assert "submit_lprf_curriculum_tigris_pilot.sh" in wrapper

    assert 'bootstrap_reco_jid="${reco_jobs[C0]:-}"' in bootstrap
    assert 'bootstrap_a0_jid="${tagger_jobs[A0]:-}"' in bootstrap
    assert 'printf \'%s\\n\' "${bootstrap_dependency}"' in bootstrap
    assert 'submit_job lprf_input_audit cpu "${LOCAL_RESIDUAL_FIELD_CURRICULUM_UPSTREAM_DEPENDENCY}"' in curriculum
    assert 'fresh_require_file "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/A0/best_model_val.pt"' in audit
    assert 'fresh_require_file "${LOCAL_RESIDUAL_FIELD_RECON_ROOT}/C0/best_model_val.pt"' in audit


def test_step10_highdata_is_off_by_default_and_requires_actual_uplift_gate() -> None:
    generic = _read("submit_lprf_curriculum_highdata.sh")
    tigris = _read("submit_lprf_curriculum_tigris_highdata.sh")
    assert "LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_HIGHDATA:=0" in generic
    assert "LOCAL_RESIDUAL_FIELD_CURRICULUM_PILOT_REPORT_OK:=0" in generic
    assert "LOCAL_RESIDUAL_FIELD_CURRICULUM_OVERRIDE_PILOT_GATE:=0" in generic
    assert "scripts/check_local_residual_curriculum_pilot_gate.py" in generic
    assert "0.003" in generic and "0.005" in generic
    assert "LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_HIGHDATA:=0" in tigris


def test_step10_reused_cache_audit_enforces_hashes_profile_and_strength() -> None:
    runner = _read("run_validate_local_residual_curriculum_reused_inputs.sh")
    script = (ROOT / "scripts" / "validate_local_residual_curriculum_reused_inputs.py").read_text(encoding="utf-8")
    assert "LOCAL_RESIDUAL_FIELD_REUSE_SPLIT_MANIFEST:=1" in runner
    assert "LOCAL_RESIDUAL_FIELD_REUSE_HLT_CACHE:=1" in runner
    assert "LOCAL_RESIDUAL_FIELD_REUSE_TARGET_CACHE:=1" in runner
    assert "--expected-hlt-profile" in runner
    assert "--expected-hlt-degradation-strength" in runner
    for field in ("source_manifest_hash", "hlt_content_hash", "offline_content_hash", "target_content_hash"):
        assert field in script


def test_step10_reused_cache_audit_writes_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "nested" / "reused_inputs_report.json"
    monkeypatch.setattr(reused_inputs_script, "load_split_manifest", lambda path: {"path": str(path)})
    monkeypatch.setattr(reused_inputs_script, "manifest_hash", lambda manifest: "manifest_hash")
    monkeypatch.setattr(
        reused_inputs_script,
        "_audit_metadata_dir",
        lambda *args, **kwargs: [{"path": "metadata.json", "content_hashes": {"hlt_content_hash": "hlt"}}],
    )

    result = reused_inputs_script.main(
        [
            "--manifest-path",
            str(tmp_path / "manifest.json.gz"),
            "--hlt-cache-dir",
            str(tmp_path / "hlt_cache"),
            "--expected-hlt-profile",
            "fixed_hlt_v2_realistic",
            "--expected-hlt-degradation-strength",
            "2.5",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def test_step10_pilot_gate_cli_writes_optional_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "nested" / "pilot_gate.json"
    monkeypatch.setattr(
        pilot_gate_script,
        "evaluate_pilot_gate",
        lambda *args, **kwargs: {"ok": True, "decision": "promote"},
    )

    result = pilot_gate_script.main(
        [
            "--report-dir",
            str(tmp_path / "report"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {"decision": "promote", "ok": True}


def test_step10_selector_is_model_val_primary_and_prefers_smooth_close_consumer(tmp_path: Path) -> None:
    ofull = _alpha_report(
        "Ofull",
        {0.0: 0.700, 0.25: 0.710, 0.50: 0.719, 0.75: 0.7185, 1.0: 0.7195},
        {0.0: 0.700, 0.25: 0.709, 0.50: 0.710, 0.75: 0.707, 1.0: 0.706},
    )
    robust = _alpha_report(
        "Orobust_light",
        {0.0: 0.700, 0.25: 0.706, 0.50: 0.711, 0.75: 0.716, 1.0: 0.719},
        {0.0: 0.700, 0.25: 0.706, 0.50: 0.711, 0.75: 0.715, 1.0: 0.718},
    )
    output = tmp_path / "selected_consumer.json"

    selected = select_curriculum_consumer((ofull, robust), output_path=output)

    assert selected["contract"] == LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT
    assert selected["selected_consumer_id"] == "Orobust_light"
    assert selected["selected_alpha_endpoint"] == 1.0
    assert selected["selection_primary_split"] == "model_val"
    assert selected["selection_confirmation_split"] == "stack_val"
    assert json.loads(output.read_text(encoding="utf-8"))["selected_consumer_id"] == "Orobust_light"
    assert curve_shape_metrics(robust["model_val_alpha_curve"])["monotonicity_score"] == 1.0


def test_step10_selector_hard_stops_when_both_weak_alpha_responses_fail(tmp_path: Path) -> None:
    flat = {0.0: 0.700, 0.25: 0.7005, 0.75: 0.7004, 1.0: 0.7003}
    with pytest.raises(ValueError, match="pilot hard stop"):
        select_curriculum_consumer(
            (_alpha_report("Ofull", flat, flat), _alpha_report("Orobust_light", flat, flat)),
            output_path=tmp_path / "selected.json",
        )


def _student_report(run_id: str, model_accuracy: float, stack_accuracy: float) -> dict:
    return {
        "ok": True,
        "run_id": run_id,
        "deployable": True,
        "runtime_inputs": "HLT_only",
        "uses_true_fields": False,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "best_model_val": {"accuracy": model_accuracy},
        "stack_val": {"accuracy": stack_accuracy},
        "checkpoint": f"/{run_id}/best_model_val.pt",
    }


def test_step10_best_p_selection_uses_model_val_then_close_tie_confirmation(tmp_path: Path) -> None:
    reports = (
        _student_report("P2", 0.7200, 0.740),
        _student_report("P4", 0.7300, 0.725),
        _student_report("P7a", 0.7305, 0.735),
        _student_report("P7b", 0.7250, 0.750),
    )
    selected = select_best_curriculum_student(reports, output_path=tmp_path / "selected_p.json")
    assert selected["contract"] == LOCAL_RESIDUAL_FIELD_STUDENT_SELECTION_CONTRACT
    assert selected["selected_run_id"] == "P7a"
    assert selected["selection_primary_split"] == "model_val"


def test_step10_pilot_gate_accepts_student_or_fusion_uplift(tmp_path: Path) -> None:
    (tmp_path / "run_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (tmp_path / "provenance_audit.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    with (tmp_path / "deployable_leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "group", "split", "accuracy"])
        writer.writeheader()
        writer.writerow({"run_id": "A0", "split": "stack_val", "accuracy": 0.776})
        writer.writerow({"run_id": "P7a", "split": "stack_val", "accuracy": 0.780})
        writer.writerow({"group": "G0", "split": "stack_val", "accuracy": 0.779})
    with (tmp_path / "curriculum_student_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id", "split", "n_jets", "attempted_jets", "valid_fraction",
                "nonfinite_fraction", "valid_for_selection",
            ],
        )
        writer.writeheader()
        for run_id in ("P2", "P4", "P7a", "P7b"):
            writer.writerow(
                {
                    "run_id": run_id,
                    "split": "stack_val",
                    "n_jets": 100,
                    "attempted_jets": 100,
                    "valid_fraction": 1.0,
                    "nonfinite_fraction": 0.0,
                    "valid_for_selection": True,
                }
            )
    gate = evaluate_pilot_gate(tmp_path)
    assert gate["ok"] is True
    assert gate["student_uplift"] == pytest.approx(0.004)

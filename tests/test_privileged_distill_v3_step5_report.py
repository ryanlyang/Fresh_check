from __future__ import annotations

import json
from pathlib import Path

from teacher_logit_reco.privileged_distill_v3 import (
    PDV3_LABEL_NAMES,
    PDV3_REPORT_CONTRACT,
    PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
    PDV3_STUDENT_HLT_PART_CE,
    PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD,
    PDV3_STEP5_REPORT_STEP,
    PDV3ReportConfig,
    build_pdv3_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"


def _per_class(offset: float = 0.0) -> list[dict[str, object]]:
    rows = []
    for index, name in enumerate(PDV3_LABEL_NAMES):
        rows.append(
            {
                "class_index": index,
                "class_name": name,
                "support": 10,
                "correct": 8,
                "accuracy": 0.80 + offset,
            }
        )
    return rows


def _dataset_meta(split: str, *, teacher_supervision_loaded: bool = False) -> dict[str, object]:
    return {
        "split": split,
        "source_manifest_hash": "manifest-hash-1",
        "hlt_content_hash": f"hlt-content-{split}",
        "jet_identity_hash": f"jets-{split}",
        "source_hlt_jet_identity_hash": f"jets-{split}",
        "teacher_logits_train_time_only": bool(teacher_supervision_loaded),
        "teacher_representations_train_time_only": bool(teacher_supervision_loaded),
    }


def _write_student(
    root: Path,
    variant: str,
    *,
    final_accuracy: float,
    val_accuracy: float,
    final_teacher_loaded: bool = False,
    baseline_checkpoint_hash: str | None = "baseline-hash-1",
    baseline_checkpoint_split_manifest_hash: str | None = "manifest-hash-1",
    baseline_from_scratch: bool = False,
    checkpoint_hash: str = "student-checkpoint-hash",
    final_test_teacher_diagnostic_metrics: dict[str, object] | None = None,
) -> None:
    path = root / variant
    path.mkdir(parents=True)
    report = {
        "student_variant": variant,
        "architecture_view_variant": "av10_feature_mlp_adapter",
        "best_epoch": 3,
        "selection_metric": "accuracy",
        "best_model_val_metrics": {
            "accuracy": val_accuracy,
            "loss": 0.5,
            "ce_loss": 0.45,
            "macro_per_class_accuracy": val_accuracy - 0.01,
            "kd_loss": 0.12,
            "rep_loss": 0.05,
            "teacher_student_logit_kl": 0.12,
            "teacher_student_representation_cosine": 0.95,
            "teacher_student_top1_agreement": 0.88,
            "teacher_entropy_mean": 0.40,
            "student_entropy_mean_with_teacher": 0.45,
            "per_class_accuracy": _per_class(val_accuracy - 0.80),
            "diagnostics": {
                "delta_h_norm_mean": 0.03,
                "gate_mean": 0.02,
                "effective_kd_alpha": 0.5,
            },
        },
        "final_test_metrics": {
            "accuracy": final_accuracy,
            "loss": 0.52,
            "ce_loss": 0.47,
            "macro_per_class_accuracy": final_accuracy - 0.01,
            "kd_loss": 0.13,
            "rep_loss": 0.06,
            "teacher_student_logit_kl": 0.13,
            "teacher_student_representation_cosine": 0.94,
            "teacher_student_top1_agreement": None,
            "teacher_entropy_mean": None,
            "student_entropy_mean_with_teacher": None,
            "per_class_accuracy": _per_class(final_accuracy - 0.80),
            "diagnostics": {
                "delta_h_norm_mean": 0.04,
                "adapter_output_norm_mean": 0.09,
                "teacher_student_logit_kl": 0.13,
            },
        },
        "parameter_accounting": {
            "total_params": 1000,
            "trainable_params": 100,
            "part_params": 900,
            "adapter_params": 100,
            "trainable_adapter_params": 100,
            "dormant_adapter_params": 0,
            "active_adapter_module_names": ["context_control"],
        },
        "representation_projector_config": {"source": "test", "input_dim": 16, "output_dim": 8},
        "runtime": {"elapsed_seconds": 12.0, "elapsed_minutes": 0.2},
        "epochs_completed": 4,
        "checkpoint": f"/checkpoints/{variant}/best_model_val.pt",
        "checkpoint_hash": checkpoint_hash,
        "last_checkpoint": f"/checkpoints/{variant}/last.pt",
        "last_checkpoint_hash": f"{checkpoint_hash}-last",
        "final_test_loaded_during_training": False,
        "inference_consumes_hlt_only": True,
        "manifest": {"manifest_hash": "manifest-hash-1"},
        "baseline_from_scratch": bool(baseline_from_scratch),
        "baseline_checkpoint_hash": baseline_checkpoint_hash,
        "baseline_checkpoint_split_manifest_hash": baseline_checkpoint_split_manifest_hash,
        "baseline_checkpoint_path": "/checkpoints/baseline/best_model_val.pt",
        "final_test_teacher_diagnostic_metrics": final_test_teacher_diagnostic_metrics,
        "train_dataset": _dataset_meta("model_train", teacher_supervision_loaded=True),
        "val_dataset": _dataset_meta("model_val", teacher_supervision_loaded=True),
        "final_test_dataset": _dataset_meta("final_test", teacher_supervision_loaded=final_teacher_loaded),
    }
    (path / "run_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_step5_report_writes_tables_and_ranks_against_baseline(tmp_path):
    students = tmp_path / "students"
    _write_student(students, PDV3_STUDENT_HLT_PART_CE, final_accuracy=0.900, val_accuracy=0.890)
    _write_student(students, PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD, final_accuracy=0.925, val_accuracy=0.910)
    _write_student(students, PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD, final_accuracy=0.915, val_accuracy=0.905)

    report = build_pdv3_report(
        PDV3ReportConfig(
            output_dir=str(tmp_path / "report"),
            students_dir=str(students),
            student_variants=(
                PDV3_STUDENT_HLT_PART_CE,
                PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
                PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD,
            ),
            confirm_final_test=True,
        )
    )

    assert report["experiment_step"] == PDV3_STEP5_REPORT_STEP
    assert report["report_contract"] == PDV3_REPORT_CONTRACT
    assert report["ok"] is True
    assert report["summary"]["best_final_test_variant"] == PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD
    assert report["summary"]["best_final_test_accuracy_gain_vs_baseline"] == 0.025000000000000022
    assert report["summary"]["best_final_test_relative_error_reduction_vs_baseline"] > 0.24
    for filename in (
        "pdv3_report.json",
        "pdv3_report.md",
        "student_metrics.csv",
        "comparison_table.csv",
        "per_class_metrics.csv",
        "confusion_matrix.csv",
        "parameter_accounting.csv",
        "adapter_diagnostics.csv",
        "kd_diagnostics.csv",
        "runtime_table.csv",
    ):
        assert (tmp_path / "report" / filename).exists(), filename


def test_step5_report_records_missing_students_when_required(tmp_path):
    students = tmp_path / "students"
    _write_student(students, PDV3_STUDENT_HLT_PART_CE, final_accuracy=0.900, val_accuracy=0.890)

    report = build_pdv3_report(
        PDV3ReportConfig(
            output_dir=str(tmp_path / "report"),
            students_dir=str(students),
            student_variants=(PDV3_STUDENT_HLT_PART_CE, PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD),
            require_all_students=True,
            confirm_final_test=True,
        )
    )

    assert report["ok"] is False
    assert any("missing required PDV3 students" in problem for problem in report["problems"])


def test_step5_report_rejects_final_test_teacher_cache_dependency(tmp_path):
    students = tmp_path / "students"
    _write_student(students, PDV3_STUDENT_HLT_PART_CE, final_accuracy=0.900, val_accuracy=0.890)
    _write_student(
        students,
        PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
        final_accuracy=0.925,
        val_accuracy=0.910,
        final_teacher_loaded=True,
    )

    report = build_pdv3_report(
        PDV3ReportConfig(
            output_dir=str(tmp_path / "report"),
            students_dir=str(students),
            student_variants=(PDV3_STUDENT_HLT_PART_CE, PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD),
            confirm_final_test=True,
        )
    )

    assert report["ok"] is False
    assert any("final_test_dataset loaded privileged teacher caches" in problem for problem in report["problems"])


def test_step5_report_accepts_scratch_baseline_checkpoint_identity(tmp_path):
    students = tmp_path / "students"
    _write_student(
        students,
        PDV3_STUDENT_HLT_PART_CE,
        final_accuracy=0.900,
        val_accuracy=0.890,
        baseline_checkpoint_hash=None,
        baseline_checkpoint_split_manifest_hash=None,
        baseline_from_scratch=True,
        checkpoint_hash="scratch-baseline-checkpoint-hash",
    )
    _write_student(
        students,
        PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
        final_accuracy=0.925,
        val_accuracy=0.910,
        baseline_checkpoint_hash="scratch-baseline-checkpoint-hash",
        baseline_checkpoint_split_manifest_hash="manifest-hash-1",
    )

    report = build_pdv3_report(
        PDV3ReportConfig(
            output_dir=str(tmp_path / "report"),
            students_dir=str(students),
            student_variants=(PDV3_STUDENT_HLT_PART_CE, PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD),
            confirm_final_test=True,
        )
    )

    assert report["ok"] is True
    assert not report["problems"]


def test_step5_report_surfaces_optional_final_test_teacher_diagnostics(tmp_path):
    students = tmp_path / "students"
    _write_student(students, PDV3_STUDENT_HLT_PART_CE, final_accuracy=0.900, val_accuracy=0.890)
    _write_student(
        students,
        PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
        final_accuracy=0.925,
        val_accuracy=0.910,
        final_test_teacher_diagnostic_metrics={
            "diagnostic_only": True,
            "teacher_student_top1_agreement": 0.77,
            "teacher_entropy_mean": 0.41,
            "diagnostics": {"teacher_confidence_mean": 0.82},
        },
    )

    report = build_pdv3_report(
        PDV3ReportConfig(
            output_dir=str(tmp_path / "report"),
            students_dir=str(students),
            student_variants=(PDV3_STUDENT_HLT_PART_CE, PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD),
            confirm_final_test=True,
        )
    )

    rows = report["tables"]["kd_diagnostics"]
    assert any(
        row["variant"] == PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD
        and row["split"] == "final_test_teacher_diagnostics"
        and row["diagnostic"] == "teacher_student_top1_agreement"
        and row["value"] == 0.77
        and row["diagnostic_only"] is True
        for row in rows
    )


def test_step5_sbatch_runner_is_wired_to_report_cli():
    runner = (SBATCH_DIR / "run_pdv3_write_report.sh").read_text(encoding="utf-8")
    common = (SBATCH_DIR / "common.sh").read_text(encoding="utf-8")

    assert "scripts/write_pdv3_report.py" in runner
    assert "--students-dir" in runner
    assert "--confirm-final-test" in runner
    assert "PDV3_REPORT_STUDENT_VARIANTS" in common
    assert "PDV3_REPORT_ALLOW_MISSING_STUDENTS:=0" in common

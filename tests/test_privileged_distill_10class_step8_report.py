import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.privileged_distill_10class import (
    PD10_NUM_CLASSES,
    PD10_REPORT_CONTRACT,
    PD10_REPORT_JSON,
    PD10_REPORT_MD,
    PD10_REPORT_TABLES,
    PD10_STEP4_EXPERIMENT_STEP,
    PD10_STEP8_EXPERIMENT_STEP,
    PD10_TARGET_FULL_LOGITS,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_HLT,
    PD10_TEACHER_LOGIT_CACHE_CONTRACT,
    PD10_TEACHER_NONE,
    PD10_TEACHER_OFFLINE,
    PD10ReportConfig,
    build_pd10_core_student_variants,
    build_pd10_priority_student_variants,
    default_pd10_experiment_layout,
    pd10_report_dir,
    pd10_teacher_model_name,
    write_pd10_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "write_pd10_report.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("write_pd10_report", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def confusion(confused_count: int) -> list[list[int]]:
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    matrix[0][0] = 95
    matrix[0][1] = int(confused_count)
    for label in range(1, 10):
        matrix[label][label] = 100
    return matrix


def write_teacher_report(root: Path, target: str, *, model_val_acc: float, final_acc: float, confused_count: int) -> None:
    model_name = pd10_teacher_model_name(target)
    write_json(
        root / "teachers" / model_name / "run_report.json",
        {
            "ok": True,
            "teacher_target": target,
            "model_name": model_name,
            "best_model_val_metrics": {
                "accuracy": model_val_acc,
                "cross_entropy": 1.0 - model_val_acc,
                "n_jets": 100,
            },
            "final_test_metrics": {
                "accuracy": final_acc,
                "cross_entropy": 1.0 - final_acc,
                "n_jets": 100,
                "expected_calibration_error": 0.03,
                "mean_confidence": 0.81,
                "confusion_matrix": confusion(confused_count),
            },
        },
    )


def write_audit(root: Path) -> None:
    write_json(
        root / "audits" / "pd10_step2_audit_report.json",
        {
            "ok": True,
            "experiment_name": "privileged_distill_10class_5m",
            "manifest_hash": "manifest-hash",
            "hlt_degradation_strength": 0.6,
            "problems": [],
            "audits": {
                "split_manifest": {
                    "ok": True,
                    "split_audit": {
                        "duplicate_within_split_count": 0,
                        "cross_split_overlap_count": 0,
                        "file_level_separation_claimed": True,
                    },
                },
                "hlt_cache": {"ok": True, "distinct_hlt_hashes_ok": True},
            },
        },
    )


def student_accuracy(spec) -> float:
    table = {
        ("scratch", PD10_TEACHER_NONE, PD10_TARGET_FULL_LOGITS, 2.0, 0.0): 0.730,
        ("scratch", PD10_TEACHER_HLT, PD10_TARGET_FULL_LOGITS, 2.0, 0.5): 0.735,
        ("scratch", PD10_TEACHER_OFFLINE, PD10_TARGET_FULL_LOGITS, 2.0, 0.5): 0.740,
        ("scratch", PD10_TEACHER_DUAL_VIEW, PD10_TARGET_FULL_LOGITS, 2.0, 0.5): 0.750,
        ("warm_start", PD10_TEACHER_NONE, PD10_TARGET_FULL_LOGITS, 2.0, 0.0): 0.745,
        ("warm_start", PD10_TEACHER_HLT, PD10_TARGET_FULL_LOGITS, 2.0, 0.5): 0.750,
        ("warm_start", PD10_TEACHER_OFFLINE, PD10_TARGET_FULL_LOGITS, 2.0, 0.5): 0.755,
        ("warm_start", PD10_TEACHER_DUAL_VIEW, PD10_TARGET_FULL_LOGITS, 2.0, 0.5): 0.770,
        ("warm_start", PD10_TEACHER_DUAL_VIEW, PD10_TARGET_FULL_LOGITS, 4.0, 0.5): 0.775,
        ("warm_start", PD10_TEACHER_DUAL_VIEW, PD10_TARGET_FULL_LOGITS, 2.0, 0.3): 0.765,
        ("warm_start", PD10_TEACHER_DUAL_VIEW, "top3", 2.0, 0.5): 0.768,
        ("warm_start", PD10_TEACHER_DUAL_VIEW, "confidence_weighted", 2.0, 0.5): 0.766,
    }
    return table[(spec.init_mode, spec.teacher_target, spec.target_mode, float(spec.temperature), float(spec.kd_alpha))]


def write_student_report(root: Path, spec) -> None:
    acc = student_accuracy(spec)
    metrics = {
        "accuracy": acc,
        "ce_loss": 1.0 - acc,
        "n_jets": 100,
        "expected_calibration_error": 0.04,
        "mean_confidence": 0.79,
    }
    if spec.teacher_target == PD10_TEACHER_DUAL_VIEW and spec.init_mode == "warm_start":
        metrics["binary_metrics"] = {
            "QCD_vs_Hgg": {
                "available": True,
                "auc": 0.91,
                "fpr_at_signal_eff_0p50": 0.12,
            }
        }
    write_json(
        root / "students" / spec.name / "run_report.json",
        {
            "ok": True,
            "variant_name": spec.name,
            "student_init": spec.init_mode,
            "teacher_target": spec.teacher_target,
            "target_mode": spec.target_mode,
            "temperature": spec.temperature,
            "kd_alpha": spec.kd_alpha,
            "top_k": spec.top_k,
            "best_epoch": 3,
            "selection_metric": "model_val_accuracy",
            "checkpoint": str(root / "students" / spec.name / "best_model_val.pt"),
            "best_model_val_metrics": {
                "accuracy": acc - 0.01,
                "ce_loss": 1.01 - acc,
                "n_jets": 100,
            },
            "final_test_metrics": metrics,
            "teacher_logits_train_time_only": True,
            "inference_requires_teacher_logits": False,
            "inference_requires_offline_inputs": False,
        },
    )


def write_complete_fixture(root: Path) -> None:
    write_teacher_report(root, PD10_TEACHER_HLT, model_val_acc=0.710, final_acc=0.720, confused_count=6)
    write_teacher_report(root, PD10_TEACHER_OFFLINE, model_val_acc=0.810, final_acc=0.820, confused_count=2)
    write_teacher_report(root, PD10_TEACHER_DUAL_VIEW, model_val_acc=0.800, final_acc=0.815, confused_count=1)
    write_audit(root)
    for spec in list(build_pd10_core_student_variants()) + list(build_pd10_priority_student_variants()):
        write_student_report(root, spec)


class PD10Step8ReportTests(unittest.TestCase):
    def test_report_writes_sections_answers_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_fixture(root)

            cfg = PD10ReportConfig(
                output_dir=str(root / "final_report"),
                teachers_dir=str(root / "teachers"),
                students_dir=str(root / "students"),
                teacher_logit_dir=str(root / "teacher_logits"),
                audit_dir=str(root / "audits"),
                confirm_final_test=True,
            )
            report = write_pd10_report(cfg)

            self.assertTrue(report["ok"], report["problems"])
            self.assertEqual(report["contract"], PD10_REPORT_CONTRACT)
            self.assertEqual(report["experiment_step"], PD10_STEP8_EXPERIMENT_STEP)
            self.assertEqual(len(report["student_core_matrix"]), 8)
            self.assertTrue(report["answers"]["did_any_student_beat_hlt_part"])
            self.assertEqual(report["answers"]["did_dual_view_kd_beat_hlt_self_kd"]["warm_start"], True)
            self.assertEqual(report["answers"]["did_dual_view_kd_beat_offline_kd"]["warm_start"], True)
            self.assertTrue(report["answers"]["did_warm_start_kd_beat_warm_start_ce_only"])
            self.assertTrue(report["answers"]["did_scratch_kd_beat_scratch_ce_only"])
            self.assertEqual(report["answers"]["best_student_variant"], "pd10_student_warm_start_dual_view_full_logits_t4_a0p5")
            self.assertAlmostEqual(report["answers"]["best_gap_closure_fraction"], 0.55)
            self.assertTrue(report["answers"]["class_pair_improvements_available"])
            self.assertTrue(any(row.get("available") is True for row in report["binary_projection_rows"]))
            self.assertTrue(any(row.get("available") is True for row in report["calibration_rows"]))

            output = root / "final_report"
            self.assertTrue((output / PD10_REPORT_JSON).exists())
            self.assertTrue((output / PD10_REPORT_MD).exists())
            self.assertTrue((output / "run_report.json").exists())
            for filename in PD10_REPORT_TABLES.values():
                self.assertTrue((output / filename).exists(), filename)

            saved = json.loads((output / PD10_REPORT_JSON).read_text(encoding="utf-8"))
            self.assertEqual(saved["answers"]["best_student_variant"], report["answers"]["best_student_variant"])

    def test_missing_core_student_is_problem_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_teacher_report(root, PD10_TEACHER_HLT, model_val_acc=0.710, final_acc=0.720, confused_count=6)
            write_teacher_report(root, PD10_TEACHER_OFFLINE, model_val_acc=0.810, final_acc=0.820, confused_count=2)
            write_teacher_report(root, PD10_TEACHER_DUAL_VIEW, model_val_acc=0.800, final_acc=0.815, confused_count=1)
            write_audit(root)
            write_student_report(root, build_pd10_core_student_variants()[0])

            strict = PD10ReportConfig(
                output_dir=str(root / "strict_report"),
                teachers_dir=str(root / "teachers"),
                students_dir=str(root / "students"),
                audit_dir=str(root / "audits"),
                include_priority_students=False,
                confirm_final_test=True,
            )
            strict_report = write_pd10_report(strict)
            self.assertFalse(strict_report["ok"])
            self.assertTrue(any("missing student run_report" in item for item in strict_report["problems"]))

            permissive = PD10ReportConfig(
                output_dir=str(root / "permissive_report"),
                teachers_dir=str(root / "teachers"),
                students_dir=str(root / "students"),
                audit_dir=str(root / "audits"),
                include_priority_students=False,
                require_core_students=False,
                confirm_final_test=True,
            )
            permissive_report = write_pd10_report(permissive)
            self.assertTrue(permissive_report["ok"], permissive_report["problems"])
            self.assertTrue(permissive_report["warnings"])

    def test_teacher_rows_merge_sparse_run_report_with_cached_logit_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_name = pd10_teacher_model_name(PD10_TEACHER_HLT)
            write_json(
                root / "teachers" / model_name / "run_report.json",
                {
                    "ok": True,
                    "teacher_target": PD10_TEACHER_HLT,
                    "model_name": model_name,
                    "final_test_metrics": {"accuracy": 0.1, "n_jets": 4},
                },
            )
            labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
            logits = np.full((4, PD10_NUM_CLASSES), -2.0, dtype=np.float32)
            logits[np.arange(4), labels] = 3.0
            for split in ("model_val", "final_test"):
                jet_ids = [
                    JetIdentity(file=f"{split}_class{int(label)}.root", entry=index, label=int(label))
                    for index, label in enumerate(labels)
                ]
                block = PredictionBlock(
                    model_name=model_name,
                    split=split,
                    logits=logits,
                    probs=softmax_np(logits),
                    labels=labels,
                    jet_ids=jet_ids,
                    metadata={
                        "contract": PD10_TEACHER_LOGIT_CACHE_CONTRACT,
                        "experiment_step": PD10_STEP4_EXPERIMENT_STEP,
                        "teacher_target": PD10_TEACHER_HLT,
                        "model_name": model_name,
                        "source_view": "fixed_hlt",
                        "allowed_inputs": "HLT_only",
                        "hlt_content_hash": f"{split}-hlt-hash",
                        "no_offline_inputs_loaded": True,
                    },
                )
                save_prediction_block(block, root / "teacher_logits")

            cfg = PD10ReportConfig(
                output_dir=str(root / "report"),
                teachers_dir=str(root / "teachers"),
                students_dir=str(root / "students"),
                teacher_logit_dir=str(root / "teacher_logits"),
                include_priority_students=False,
                require_core_students=False,
                require_teacher_reports=False,
                require_audit=False,
                confirm_final_test=True,
            )
            report = write_pd10_report(cfg)
            hlt_final = next(
                row
                for row in report["teacher_metric_rows"]
                if row.get("teacher_target") == PD10_TEACHER_HLT and row.get("split") == "final_test"
            )

            self.assertEqual(hlt_final["metrics_source"], "teacher_logit_cache+run_report")
            self.assertGreater(hlt_final["accuracy"], 0.9)
            self.assertIn("confusion_matrix", hlt_final["metrics"])
            self.assertIn("expected_calibration_error", hlt_final["metrics"])
            self.assertIn("fpr_at_signal_eff_0p50_macro", hlt_final["metrics"])
            self.assertIn("validation_threshold_fpr_at_signal_eff_0p50_macro", hlt_final["metrics"])
            self.assertIn("validation_binary_fpr_at_signal_eff_0p50_macro", hlt_final["metrics"])
            binary_row = next(
                row
                for row in report["binary_projection_rows"]
                if row.get("source") == PD10_TEACHER_HLT
                and row.get("split") == "final_test"
                and row.get("binary_task") == "QCD_vs_Hbb"
            )
            self.assertEqual(binary_row["validation_thresholds_from_split"], "model_val")
            self.assertIsNotNone(binary_row["validation_fpr_at_signal_eff_0p50"])

    def test_config_and_cli_defaults_are_canonical(self):
        layout = default_pd10_experiment_layout(output_root="checkpoints")
        self.assertEqual(pd10_report_dir(output_root="checkpoints"), layout.final_report_dir)

        with self.assertRaises(ValueError):
            PD10ReportConfig(
                output_dir="out",
                teachers_dir="teachers",
                students_dir="students",
            )

        module = load_script_module()
        args = module.parse_args(["--confirm-final-test"])
        self.assertEqual(args.output_dir, str(layout.final_report_dir))
        self.assertEqual(args.teachers_dir, str(layout.teachers_dir))
        self.assertEqual(args.students_dir, str(layout.students_dir))
        self.assertEqual(args.teacher_logit_dir, str(layout.teacher_logits_dir))
        self.assertEqual(args.audit_dir, str(layout.step2_audit_dir))
        self.assertTrue(args.confirm_final_test)
        self.assertFalse(args.core_only)
        self.assertFalse(args.allow_missing_core_students)


if __name__ == "__main__":
    unittest.main()

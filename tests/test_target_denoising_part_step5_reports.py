import csv
import json
import tempfile
import unittest
from pathlib import Path

from teacher_logit_reco.target_denoising_part import (
    TARGET_DENOISING_REPORT_CONTRACT,
    TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN,
    TARGET_DENOISING_VARIANT_HLT_PART_BASELINE,
    TargetDenoisingReportConfig,
    write_target_denoising_report,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TargetDenoisingPartStep5ReportTests(unittest.TestCase):
    def test_report_builder_separates_selection_final_and_diagnostic_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tagger_root = root / "taggers"
            hlt_report = {
                "variant": TARGET_DENOISING_VARIANT_HLT_PART_BASELINE,
                "best_epoch": 2,
                "selection_metric": "accuracy",
                "best_model_val_metrics": {
                    "accuracy": 0.70,
                    "loss": 0.90,
                    "n_jets": 100,
                    "per_class_accuracy": {"0": 0.75, "1": 0.65},
                    "confusion_matrix": [[40, 10], [20, 30]],
                },
                "final_test_metrics": {
                    "accuracy": 0.68,
                    "loss": 0.95,
                    "n_jets": 80,
                    "per_class_accuracy": [0.70, 0.66],
                    "confusion_matrix": [[28, 12], [14, 26]],
                },
                "variant_behavior": {"baseline": True},
            }
            denoise_tagger_report = {
                "variant": TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN,
                "best_epoch": 3,
                "selection_metric": "accuracy",
                "best_model_val_metrics": {
                    "accuracy": 0.73,
                    "loss": 0.82,
                    "n_jets": 100,
                    "per_class_accuracy": {"0": 0.77, "1": 0.69},
                    "confusion_matrix": [[42, 8], [19, 31]],
                },
                "final_test_metrics": {
                    "accuracy": 0.71,
                    "loss": 0.88,
                    "n_jets": 80,
                    "per_class_accuracy": {"0": 0.73, "1": 0.69},
                    "confusion_matrix": [[30, 10], [13, 27]],
                },
                "adapter_diagnostics": {"adapter_output_norm_mean": 0.02, "gate_mean": 0.12},
                "injection_summary": {"injection_applied": True, "delta_h_abs_max": 0.04},
                "diagnostic_only_teacher_offline": {
                    "split": "model_val",
                    "teacher_student_top1_agreement": 0.61,
                },
                "model_config": {"variant_behavior": {"uses_denoiser": True}},
            }
            denoiser_report = {
                "best_epoch": 1,
                "selection_metric": "normalized_rmse",
                "best_model_val_metrics": {
                    "normalized_rmse": 0.14,
                    "nll_loss": 0.31,
                    "smooth_l1_loss": 0.06,
                    "n_jets": 100,
                },
            }
            _write_json(tagger_root / TARGET_DENOISING_VARIANT_HLT_PART_BASELINE / "run_report.json", hlt_report)
            _write_json(
                tagger_root / TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN / "run_report.json",
                denoise_tagger_report,
            )
            denoiser_path = root / "denoiser" / "run_report.json"
            _write_json(denoiser_path, denoiser_report)

            output_dir = root / "report"
            summary = write_target_denoising_report(
                TargetDenoisingReportConfig(
                    output_dir=str(output_dir),
                    tagger_root=str(tagger_root),
                    denoiser_report=str(denoiser_path),
                    variants=(
                        TARGET_DENOISING_VARIANT_HLT_PART_BASELINE,
                        TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN,
                    ),
                    require_variants=True,
                )
            )

            self.assertTrue(summary["ok"])
            self.assertEqual(summary["contract"], TARGET_DENOISING_REPORT_CONTRACT)
            for name in (
                "summary.json",
                "tagger_metrics.csv",
                "denoising_metrics.csv",
                "per_class_accuracy.csv",
                "adapter_attention_diagnostics.csv",
                "diagnostic_only_teacher_offline.csv",
                "mechanism_ablation_metrics.csv",
                "confusion_matrices.json",
            ):
                self.assertTrue((output_dir / name).exists(), name)

            tagger_rows = _read_csv(output_dir / "tagger_metrics.csv")
            self.assertIn("model_val_selection", {row["split_group"] for row in tagger_rows})
            self.assertIn("final_test_one_shot", {row["split_group"] for row in tagger_rows})
            frozen_final = [
                row
                for row in tagger_rows
                if row["variant"] == TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN
                and row["split"] == "final_test"
            ][0]
            self.assertEqual(frozen_final["accuracy"], "0.71")

            denoising_rows = _read_csv(output_dir / "denoising_metrics.csv")
            self.assertEqual(denoising_rows[0]["normalized_rmse"], "0.14")

            diagnostic_rows = _read_csv(output_dir / "diagnostic_only_teacher_offline.csv")
            self.assertEqual(diagnostic_rows[0]["split_group"], "diagnostic_only_teacher_offline")

            mechanism_rows = _read_csv(output_dir / "mechanism_ablation_metrics.csv")
            frozen_model_val = [
                row
                for row in mechanism_rows
                if row["variant"] == TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN
                and row["split"] == "model_val"
            ][0]
            self.assertAlmostEqual(float(frozen_model_val["accuracy_delta_vs_hlt"]), 0.03, places=6)

            confusion = json.loads((output_dir / "confusion_matrices.json").read_text(encoding="utf-8"))
            self.assertEqual(len(confusion["entries"]), 4)

    def test_report_builder_marks_missing_required_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = write_target_denoising_report(
                TargetDenoisingReportConfig(
                    output_dir=str(root / "report"),
                    tagger_root=str(root / "taggers"),
                    variants=(TARGET_DENOISING_VARIANT_HLT_PART_BASELINE,),
                    require_variants=True,
                )
            )

            self.assertFalse(summary["ok"])
            self.assertIn("missing tagger report", summary["problems"][0])


if __name__ == "__main__":
    unittest.main()

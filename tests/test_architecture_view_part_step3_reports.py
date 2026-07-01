from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_REPORT_CONTRACT,
    ARCHITECTURE_VIEW_REPORT_STEP,
    ARCHITECTURE_VIEW_VALIDATION_THRESHOLD_METRIC,
    ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_VARIANT_PN_ONLY,
    ArchitectureViewPartReportConfig,
    build_architecture_view_part_report,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def metrics(*, fpr50: float, auc: float, accuracy: float, arrays: dict | None = None) -> dict:
    payload = {
        "accuracy": float(accuracy),
        "loss": 0.4,
        "n_jets": 8,
        "binary_metrics": {
            "auc": float(auc),
            "fpr_at_signal_eff_0p30": float(fpr50) / 2.0,
            "fpr_at_signal_eff_0p50": float(fpr50),
            "background_rejection_at_signal_eff_0p30": 2.0 / float(fpr50),
            "background_rejection_at_signal_eff_0p50": 1.0 / float(fpr50),
        },
        "diagnostics": {"delta_h_norm_mean": 0.001},
    }
    if arrays is not None:
        payload["_prediction_arrays"] = arrays
    return payload


def run_report(variant: str, *, final_fpr50: float, val_fpr50: float, arrays: bool = False) -> dict:
    model_val_arrays = None
    final_test_arrays = None
    if arrays:
        model_val_arrays = {
            "labels": [1, 1, 1, 1, 0, 0, 0, 0],
            "scores": [0.9, 0.8, 0.4, 0.2, 0.7, 0.5, 0.3, 0.1],
        }
        final_test_arrays = {
            "labels": [1, 1, 1, 1, 0, 0, 0, 0],
            "scores": [0.95, 0.55, 0.45, 0.15, 0.65, 0.35, 0.25, 0.05],
        }
    return {
        "experiment_step": "architecture_view_part_step3_train",
        "output_contract": "toy_contract",
        "variant": variant,
        "variant_behavior": {"variant": variant},
        "checkpoint": "best_model_val.pt",
        "best_epoch": 3,
        "epochs_completed": 4,
        "selection_metric": "fpr_at_signal_eff_0p50",
        "selection_metric_direction": "minimize",
        "best_model_selection_metric_value": val_fpr50,
        "best_model_val_metrics": metrics(fpr50=val_fpr50, auc=0.81, accuracy=0.7, arrays=model_val_arrays),
        "stack_val_metrics": metrics(fpr50=val_fpr50 + 0.01, auc=0.8, accuracy=0.69),
        "final_test_metrics": metrics(fpr50=final_fpr50, auc=0.84, accuracy=0.73, arrays=final_test_arrays),
        "final_test_evaluated": True,
        "num_classes": 2,
        "label_names": ["QCD", "Hgg"],
        "label_filter": [0, 1],
        "inference_consumes_hlt_only": True,
        "runtime": {"elapsed_seconds": 12.0, "elapsed_minutes": 0.2},
        "walltime_seconds": 12.0,
    }


class ArchitectureViewStep3ReportTests(unittest.TestCase):
    def test_report_sorts_by_final_test_fpr50_and_compares_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK / "run_report.json"
            all_views = root / ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS / "run_report.json"
            pn_only = root / ARCHITECTURE_VIEW_VARIANT_PN_ONLY / "run_report.json"
            write_json(baseline, run_report(ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK, final_fpr50=0.30, val_fpr50=0.29))
            write_json(all_views, run_report(ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS, final_fpr50=0.20, val_fpr50=0.24))
            write_json(pn_only, run_report(ARCHITECTURE_VIEW_VARIANT_PN_ONLY, final_fpr50=0.25, val_fpr50=0.28))
            write_json(
                root / "variant_suite_report.json",
                {"run_reports": [str(baseline.relative_to(root)), str(all_views.relative_to(root)), str(pn_only.relative_to(root))]},
            )

            report = build_architecture_view_part_report(
                ArchitectureViewPartReportConfig(
                    output_dir=str(root / "final_report"),
                    experiment_dir=str(root),
                    variants=(
                        ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
                        ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
                        ARCHITECTURE_VIEW_VARIANT_PN_ONLY,
                    ),
                    confirm_final_test=True,
                )
            )

            self.assertTrue(report["ok"], report["problems"])
            self.assertEqual(report["experiment_step"], ARCHITECTURE_VIEW_REPORT_STEP)
            self.assertEqual(report["output_contract"], ARCHITECTURE_VIEW_REPORT_CONTRACT)
            self.assertEqual(report["comparison_summary"]["best_variant"], ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS)
            self.assertEqual(report["metric_table"][0]["variant"], ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS)
            row = next(row for row in report["metric_table"] if row["variant"] == ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS)
            self.assertEqual(row["final_test_oracle_fpr_at_signal_eff_0p50"], 0.20)
            self.assertGreater(row["primary_metric_improvement_vs_baseline"], 0.0)
            self.assertTrue(row["beats_baseline"])
            self.assertTrue((root / "final_report" / "architecture_view_part_final_report.json").exists())
            self.assertTrue((root / "final_report" / "metric_table.csv").exists())
            self.assertTrue((root / "final_report" / "run_report.json").exists())

    def test_validation_threshold_final_test_metric_is_computed_when_arrays_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS / "run_report.json",
                run_report(ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS, final_fpr50=0.10, val_fpr50=0.20, arrays=True),
            )
            write_json(
                root / ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK / "run_report.json",
                run_report(ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK, final_fpr50=0.30, val_fpr50=0.29),
            )

            report = build_architecture_view_part_report(
                ArchitectureViewPartReportConfig(
                    output_dir=str(root / "final_report"),
                    experiment_dir=str(root),
                    variants=(ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK, ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS),
                    confirm_final_test=True,
                )
            )

            row = next(row for row in report["metric_table"] if row["variant"] == ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS)
            self.assertTrue(row["validation_threshold_final_test_available"])
            self.assertEqual(row["validation_threshold_final_test_threshold"], 0.8)
            self.assertEqual(row[ARCHITECTURE_VIEW_VALIDATION_THRESHOLD_METRIC], 0.0)
            self.assertEqual(row["validation_threshold_final_test_signal_efficiency"], 0.25)

    def test_missing_child_reports_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK / "run_report.json",
                run_report(ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK, final_fpr50=0.30, val_fpr50=0.29),
            )

            report = build_architecture_view_part_report(
                ArchitectureViewPartReportConfig(
                    output_dir=str(root / "final_report"),
                    experiment_dir=str(root),
                    variants=(ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK, ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS),
                    confirm_final_test=True,
                )
            )

            self.assertFalse(report["ok"])
            self.assertTrue(any("missing child run_report" in problem for problem in report["problems"]))


if __name__ == "__main__":
    unittest.main()

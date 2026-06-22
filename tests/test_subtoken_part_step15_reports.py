import json
import tempfile
import unittest
from pathlib import Path

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_DEFAULT_BINARY_PRIMARY_METRIC,
    SUBTOKEN_PART_REPORT_CONTRACT,
    SUBTOKEN_PART_REPORT_STEP,
    SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
    SubtokenPartReportConfig,
    build_subtoken_part_final_report,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SubtokenPartStep15ReportTests(unittest.TestCase):
    def make_child_report(self, *, variant: str, fpr50: float, accuracy: float, gate_diag: bool = False) -> dict:
        diagnostics = {}
        if gate_diag:
            diagnostics = {
                "gate_mean_gate_entropy": 0.72,
                "gate_mean_gate_by_particle": [0.2, 0.5, 0.3],
            }
        return {
            "experiment_step": "child",
            "variant": variant,
            "output_contract": "test_child",
            "best_epoch": 3,
            "epochs_completed": 4,
            "selection_metric": "accuracy",
            "selection_metric_direction": "maximize",
            "best_model_selection_metric_value": accuracy,
            "checkpoint": "best_model_val.pt",
            "label_names": ["QCD", "Hgg"],
            "label_filter": [0, 1],
            "num_classes": 2,
            "inference_consumes_hlt_only": True,
            "best_model_val_metrics": {"accuracy": accuracy - 0.02, "loss": 0.4, "n_jets": 20},
            "stack_val_metrics": {
                "accuracy": accuracy - 0.01,
                "loss": 0.35,
                "n_jets": 30,
                "binary_metrics": {
                    "auc": 0.91,
                    "fpr_at_signal_eff_0p50": fpr50 + 0.02,
                    "background_rejection_at_signal_eff_0p50": 1.0 / (fpr50 + 0.02),
                },
                "diagnostics": diagnostics,
            },
            "final_test_metrics": {
                "accuracy": accuracy,
                "loss": 0.33,
                "n_jets": 40,
                "binary_metrics": {
                    "auc": 0.94,
                    "fpr_at_signal_eff_0p50": fpr50,
                    "background_rejection_at_signal_eff_0p50": 1.0 / fpr50,
                },
                "diagnostics": diagnostics,
            },
        }

    def test_binary_report_defaults_to_lower_fpr50_and_baseline_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            output = Path(tmp) / "report"
            baseline_path = root / SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE / "run_report.json"
            subtoken_path = root / SUBTOKEN_PART_VARIANT_CONTEXT_GATE / "run_report.json"
            write_json(
                baseline_path,
                self.make_child_report(
                    variant=SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
                    fpr50=0.30,
                    accuracy=0.83,
                ),
            )
            write_json(
                subtoken_path,
                self.make_child_report(
                    variant=SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
                    fpr50=0.18,
                    accuracy=0.81,
                    gate_diag=True,
                ),
            )
            write_json(
                root / "run_report.json",
                {
                    "num_classes": 2,
                    "comparison_split": "final_test",
                    "primary_metric": "accuracy",
                    "child_reports": {
                        SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE: str(baseline_path),
                        SUBTOKEN_PART_VARIANT_CONTEXT_GATE: str(subtoken_path),
                    },
                },
            )

            report = build_subtoken_part_final_report(
                SubtokenPartReportConfig(
                    output_dir=str(output),
                    experiment_dir=str(root),
                    include_parameter_counts=False,
                    confirm_final_test=True,
                )
            )

            summary = report["comparison_summary"]
            self.assertTrue(report["ok"])
            self.assertEqual(report["experiment_step"], SUBTOKEN_PART_REPORT_STEP)
            self.assertEqual(report["output_contract"], SUBTOKEN_PART_REPORT_CONTRACT)
            self.assertEqual(summary["primary_metric"], SUBTOKEN_PART_DEFAULT_BINARY_PRIMARY_METRIC)
            self.assertEqual(summary["primary_metric_direction"], "minimize")
            self.assertEqual(summary["best_variant"], SUBTOKEN_PART_VARIANT_CONTEXT_GATE)
            self.assertAlmostEqual(summary["best_metric_value"], 0.18)
            rows = {row["variant"]: row for row in report["metric_table"]}
            self.assertTrue(rows[SUBTOKEN_PART_VARIANT_CONTEXT_GATE]["beats_baseline"])
            self.assertAlmostEqual(
                rows[SUBTOKEN_PART_VARIANT_CONTEXT_GATE]["primary_metric_improvement_vs_baseline"],
                0.12,
            )
            self.assertTrue((output / "metric_table.csv").exists())
            self.assertTrue((output / "gate_diagnostics.csv").exists())
            self.assertTrue((output / "subtoken_part_final_report.md").exists())

    def test_report_extracts_gate_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            output = Path(tmp) / "report"
            subtoken_path = root / SUBTOKEN_PART_VARIANT_CONTEXT_GATE / "run_report.json"
            write_json(
                subtoken_path,
                self.make_child_report(
                    variant=SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
                    fpr50=0.20,
                    accuracy=0.85,
                    gate_diag=True,
                ),
            )

            report = build_subtoken_part_final_report(
                SubtokenPartReportConfig(
                    output_dir=str(output),
                    experiment_dir=str(root),
                    variants=(SUBTOKEN_PART_VARIANT_CONTEXT_GATE,),
                    primary_metric="accuracy",
                    comparison_split="final_test",
                    baseline_variant=SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
                    include_parameter_counts=False,
                )
            )

            diagnostics = report["gate_diagnostics"]
            names = {row["diagnostic"] for row in diagnostics}
            self.assertIn("gate_mean_gate_entropy", names)
            self.assertIn("gate_mean_gate_by_particle[0]", names)
            self.assertEqual(report["comparison_summary"]["best_variant"], SUBTOKEN_PART_VARIANT_CONTEXT_GATE)


if __name__ == "__main__":
    unittest.main()

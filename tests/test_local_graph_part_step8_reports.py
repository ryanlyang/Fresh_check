import json
import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.hlt_baseline import require_torch

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
    LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
    LOCAL_GRAPH_PART_REPORT_CONTRACT,
    LOCAL_GRAPH_PART_REPORT_STEP,
    LocalGraphPartReportConfig,
    build_local_graph_part_report,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class LocalGraphPartStep8ReportTests(unittest.TestCase):
    def make_child_report(self, *, variant: str, fpr50: float, accuracy: float, local_diag: bool = False) -> dict:
        diagnostics = {
            "valid_particle_count_mean": 5.0,
            "logit_abs_mean": 0.25,
        }
        if local_diag:
            diagnostics.update(
                {
                    "local_gamma": 0.07,
                    "local_mean_valid_neighbors": 3.5,
                    "local_mean_neighbor_delta_r": 0.18,
                    "local_attention_entropy_mean": 1.1,
                }
            )
        metadata = {
            "source_view": "fixed_hlt",
            "n_jets": 40,
            "label_counts": {"QCD": 20, "Hgg": 20},
            "hlt_content_hash": "toy-hlt-hash",
            "jet_identity_hash": "toy-jet-hash",
            "hlt_seed": 123,
            "hlt_params": {"strength": 0.6, "drop_probability": 0.12},
        }
        return {
            "experiment_step": "local_graph_part_step6_train_baseline_and_adapters",
            "variant": variant,
            "output_contract": "test_child",
            "best_epoch": 3,
            "epochs_completed": 4,
            "selection_metric": "fpr_at_signal_eff_0p50",
            "selection_metric_direction": "minimize",
            "best_model_selection_metric_value": fpr50,
            "checkpoint": "best_model_val.pt",
            "label_names": ["QCD", "Hgg"],
            "label_filter": [0, 1],
            "num_classes": 2,
            "inference_consumes_hlt_only": True,
            "model_config": {"uses_reference_part_backbone": True},
            "runtime": {"elapsed_seconds": 12.5, "elapsed_minutes": 12.5 / 60.0, "epochs_completed": 4},
            "walltime_seconds": 12.5,
            "train_dataset": {**metadata, "split": "model_train", "n_jets": 80},
            "val_dataset": {**metadata, "split": "model_val", "n_jets": 20},
            "stack_val_dataset": {**metadata, "split": "stack_val", "n_jets": 30},
            "final_test_dataset": {**metadata, "split": "final_test", "n_jets": 40},
            "best_model_val_metrics": {
                "accuracy": accuracy - 0.02,
                "loss": 0.4,
                "n_jets": 20,
                "binary_metrics": {"auc": 0.91, "fpr_at_signal_eff_0p50": fpr50 + 0.02},
            },
            "stack_val_metrics": {
                "accuracy": accuracy - 0.01,
                "loss": 0.35,
                "n_jets": 30,
                "binary_metrics": {
                    "auc": 0.92,
                    "fpr_at_signal_eff_0p50": fpr50 + 0.01,
                    "background_rejection_at_signal_eff_0p50": 1.0 / (fpr50 + 0.01),
                },
                "diagnostics": diagnostics,
            },
            "final_test_metrics": {
                "accuracy": accuracy,
                "loss": 0.33,
                "n_jets": 40,
                "binary_metrics": {
                    "auc": 0.94,
                    "fpr_at_signal_eff_0p30": fpr50 / 2.0,
                    "fpr_at_signal_eff_0p50": fpr50,
                    "background_rejection_at_signal_eff_0p50": 1.0 / fpr50,
                },
                "diagnostics": diagnostics,
            },
        }

    def write_checkpoint(self, path: Path) -> None:
        torch = require_torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {
                    "linear.weight": torch.zeros(2, 3),
                    "linear.bias": torch.zeros(2),
                }
            },
            path,
        )

    def test_report_defaults_to_fpr50_and_extracts_step8_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            output = Path(tmp) / "report"
            baseline_path = root / LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE / "run_report.json"
            adapter_path = root / LOCAL_GRAPH_MODEL_VARIANT_EDGECONV / "run_report.json"
            write_json(
                baseline_path,
                self.make_child_report(
                    variant=LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
                    fpr50=0.30,
                    accuracy=0.84,
                ),
            )
            write_json(
                adapter_path,
                self.make_child_report(
                    variant=LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
                    fpr50=0.18,
                    accuracy=0.82,
                    local_diag=True,
                ),
            )
            self.write_checkpoint(baseline_path.parent / "best_model_val.pt")
            self.write_checkpoint(adapter_path.parent / "best_model_val.pt")
            write_json(
                root / "run_report.json",
                {
                    "num_classes": 2,
                    "comparison_split": "final_test",
                    "primary_metric": "accuracy",
                    "child_reports": {
                        LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE: str(
                            Path(root.name) / LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE / "run_report.json"
                        ),
                        LOCAL_GRAPH_MODEL_VARIANT_EDGECONV: str(
                            Path(root.name) / LOCAL_GRAPH_MODEL_VARIANT_EDGECONV / "run_report.json"
                        ),
                    },
                },
            )

            report = build_local_graph_part_report(
                LocalGraphPartReportConfig(
                    output_dir=str(output),
                    experiment_dir=str(root),
                    include_parameter_counts=True,
                    confirm_final_test=True,
                )
            )

            summary = report["comparison_summary"]
            self.assertTrue(report["ok"])
            self.assertEqual(report["experiment_step"], LOCAL_GRAPH_PART_REPORT_STEP)
            self.assertEqual(report["output_contract"], LOCAL_GRAPH_PART_REPORT_CONTRACT)
            self.assertEqual(summary["primary_metric"], "fpr_at_signal_eff_0p50")
            self.assertEqual(summary["primary_metric_direction"], "minimize")
            self.assertEqual(summary["best_variant"], LOCAL_GRAPH_MODEL_VARIANT_EDGECONV)
            self.assertAlmostEqual(summary["best_metric_value"], 0.18)
            rows = {row["variant"]: row for row in report["metric_table"]}
            self.assertTrue(rows[LOCAL_GRAPH_MODEL_VARIANT_EDGECONV]["beats_baseline"])
            self.assertAlmostEqual(
                rows[LOCAL_GRAPH_MODEL_VARIANT_EDGECONV]["primary_metric_improvement_vs_baseline"],
                0.12,
            )
            self.assertTrue((output / "metric_table.csv").exists())
            self.assertTrue((output / "adapter_diagnostics.csv").exists())
            self.assertTrue((output / "parameter_counts.csv").exists())
            self.assertTrue((output / "runtime_summary.csv").exists())
            self.assertTrue((output / "hlt_degradation_summary.csv").exists())
            self.assertTrue((output / "local_graph_part_report.md").exists())

            diagnostic_names = {row["diagnostic"] for row in report["adapter_diagnostics"]}
            self.assertIn("local_mean_valid_neighbors", diagnostic_names)
            diagnostic_categories = {row["category"] for row in report["adapter_diagnostics"]}
            self.assertIn("local_graph", diagnostic_categories)
            self.assertIn("adapter", diagnostic_categories)
            self.assertTrue(any(row.get("parameter_count") == 8 for row in report["parameter_counts"]))
            self.assertTrue(any(row.get("metadata_hlt_degradation_strength") == 0.6 for row in report["hlt_degradation_summary"]))

    def test_report_can_use_explicit_variant_and_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            output = Path(tmp) / "report"
            adapter_path = root / "warm_point" / "run_report.json"
            write_json(
                adapter_path,
                self.make_child_report(
                    variant=LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
                    fpr50=0.20,
                    accuracy=0.88,
                    local_diag=True,
                ),
            )

            report = build_local_graph_part_report(
                LocalGraphPartReportConfig(
                    output_dir=str(output),
                    experiment_dir=str(root),
                    variants=("warm_point",),
                    primary_metric="accuracy",
                    comparison_split="final_test",
                    baseline_variant="warm_point",
                    include_parameter_counts=False,
                )
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["comparison_summary"]["primary_metric"], "accuracy")
            self.assertEqual(report["comparison_summary"]["best_variant"], "warm_point")
            self.assertEqual(report["metric_table"][0]["model_variant"], LOCAL_GRAPH_MODEL_VARIANT_EDGECONV)


if __name__ == "__main__":
    unittest.main()

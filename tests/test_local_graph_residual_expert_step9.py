import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_RESIDUAL_BASELINE_VARIANT,
    LOCAL_GRAPH_RESIDUAL_PRECOMPUTED_EVAL_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_REPORT_CONTRACT,
    LocalGraphBaselineLogitBlock,
    LocalGraphResidualExpertReportConfig,
    baseline_condition_reference_from_block,
    binary_logits_from_log_odds,
    binary_metrics_from_signal_scores,
    build_local_graph_residual_expert_report,
    save_baseline_logit_block,
)


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "write_local_graph_residual_expert_report.py"
    spec = importlib.util.spec_from_file_location("write_local_graph_residual_expert_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LocalGraphResidualExpertStep9ReportTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _make_baseline_cache(self, root: Path) -> None:
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        scores = np.asarray([0.8, 0.6, -1.0, -2.0, 0.7, 0.5, 0.2, -0.5], dtype=np.float32)
        logits = binary_logits_from_log_odds(scores).astype(np.float32)
        blocks = {}
        for split in ("model_train", "model_val", "stack_val", "final_test"):
            blocks[split] = LocalGraphBaselineLogitBlock(
                split=split,
                logits=logits,
                labels=labels,
                indices=np.arange(labels.shape[0], dtype=np.int64),
                metadata={
                    "split": split,
                    "checkpoint": "baseline.pt",
                    "checkpoint_variant": "hlt_part_baseline",
                    "checkpoint_epoch": 3,
                    "run_report": "baseline/run_report.json",
                    "dataset": {
                        "split": split,
                        "n_jets": int(labels.shape[0]),
                        "hlt_content_hash": "toy-hlt",
                        "jet_identity_hash": "toy-jets",
                        "hlt_params": {"strength": 0.6},
                    },
                },
            )
        reference = baseline_condition_reference_from_block(blocks["model_train"], source_split="model_train")
        for split, block in blocks.items():
            save_baseline_logit_block(block, root, condition_reference=reference, overwrite=True)
        self._write_json(root / "baseline_logit_manifest.json", {"ok": True})
        self._write_json(root / "run_report.json", {"ok": True})

    def _metrics(self, scores):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        return binary_metrics_from_signal_scores(np.asarray(scores, dtype=np.float32), labels)

    def _make_residual_reports(self, root: Path) -> None:
        learned_metrics = self._metrics([0.8, 0.6, -1.0, -2.0, 0.7, 0.5, 0.2, -0.5])
        selected_metrics = self._metrics([0.4, 0.2, -1.0, -2.0, 0.8, 0.7, 0.2, -0.5])
        residual_metrics = self._metrics([-0.4, -0.4, -0.4, -0.4, 0.1, 0.1, 0.1, 0.1])
        baseline_metrics = self._metrics([0.8, 0.6, -1.0, -2.0, 0.7, 0.5, 0.2, -0.5])
        payload = {
            "precomputed_evaluation_contract": LOCAL_GRAPH_RESIDUAL_PRECOMPUTED_EVAL_CONTRACT,
            "variant": "local_point_attention_adapter_residual_expert",
            "best_epoch": 3,
            "checkpoint": "best_model_val.pt",
            "loss_config": {"mode": "residual_boundary_pairwise_soft_fpr_bce_anchor"},
            "alpha_shrinkage_model_val": {
                "selected_alpha": 0.5,
                "collapsed_to_zero": False,
                "shrinkage_applies_to": "learned_correction_delta",
            },
            "baseline_cache_alignment": {
                "family": {
                    "condition_reference": {
                        "source_split": "model_train",
                    },
                    "checkpoint_identity": {
                        "checkpoint": "baseline.pt",
                        "checkpoint_variant": "hlt_part_baseline",
                        "checkpoint_epoch": 3,
                        "run_report": "baseline/run_report.json",
                    },
                },
            },
            "evaluations": {
                split: {
                    "fused_metrics": learned_metrics,
                    "baseline_metrics": baseline_metrics,
                    "residual_metrics": residual_metrics,
                    "selected_alpha_metrics": selected_metrics,
                    "selected_alpha": 0.5,
                    "residual_diagnostics": {
                        "fused_delta_FPR50_vs_baseline": -0.5,
                        "false_positive_overlap": {
                            "old_false_positives_removed": 2,
                            "new_false_positives_introduced": 0,
                        },
                    },
                    "source": "unit_test_precomputed",
                }
                for split in ("model_val", "stack_val", "final_test")
            },
        }
        self._write_json(root / "d" / "run_report.json", payload)

    def _make_standalone_and_fusion_reports(self, root: Path, fusion_path: Path) -> None:
        standalone = {
            "variant": "local_point_attention_adapter",
            "best_epoch": 4,
            "final_test_metrics": self._metrics([0.7, 0.3, -1.0, -2.0, 0.8, 0.6, 0.1, -0.5]),
            "stack_val_metrics": self._metrics([0.7, 0.3, -1.0, -2.0, 0.8, 0.6, 0.1, -0.5]),
            "best_model_val_metrics": self._metrics([0.7, 0.3, -1.0, -2.0, 0.8, 0.6, 0.1, -0.5]),
        }
        self._write_json(root / "local_point_attention_adapter" / "run_report.json", standalone)
        fusion_row = {
            "method": "weighted_average",
            "model_set": "baseline_plus_local",
            "is_control": False,
            "final_test_accuracy": 0.75,
            "final_test_auc": 0.95,
            "final_test_fpr_at_signal_eff_0p30": 0.0,
            "final_test_fpr_at_signal_eff_0p50": 0.25,
            "final_test_background_rejection_at_signal_eff_0p50": 4.0,
        }
        self._write_json(fusion_path, {"fusion_metric_table": [fusion_row]})

    def test_report_builder_compares_baseline_residual_standalone_and_fusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_logits"
            residual_root = root / "residual_experts"
            standalone_root = root / "taggers"
            output = root / "final_report"
            fusion_path = root / "score_fusion" / "fusion_report.json"
            self._make_baseline_cache(baseline)
            self._make_residual_reports(residual_root)
            self._make_standalone_and_fusion_reports(standalone_root, fusion_path)

            report = build_local_graph_residual_expert_report(
                LocalGraphResidualExpertReportConfig(
                    output_dir=str(output),
                    hlt_cache_dir=str(root / "unused_hlt_cache"),
                    baseline_logit_cache_dir=str(baseline),
                    residual_expert_root=str(residual_root),
                    residual_variants=("d",),
                    standalone_tagger_root=str(standalone_root),
                    standalone_variants=("local_point_attention_adapter",),
                    score_fusion_report_path=str(fusion_path),
                    confirm_final_test=True,
                    evaluate_checkpoints=False,
                    allow_precomputed_evaluations=True,
                )
            )

            self.assertTrue(report["ok"], report["problems"])
            self.assertEqual(report["output_contract"], LOCAL_GRAPH_RESIDUAL_REPORT_CONTRACT)
            self.assertEqual(report["comparison_summary"]["primary_metric"], LOCAL_GRAPH_PART_PRIMARY_METRIC)
            self.assertEqual(report["comparison_summary"]["baseline_variant"], LOCAL_GRAPH_RESIDUAL_BASELINE_VARIANT)
            self.assertEqual(report["comparison_summary"]["best_source_type"], "residual_fused_val_shrunk")
            self.assertEqual(report["comparison_summary"]["best_variant"], "d__val_shrunk")
            source_types = {row["source_type"] for row in report["metric_table"]}
            self.assertIn("baseline", source_types)
            self.assertIn("residual_fused_learned_alpha", source_types)
            self.assertIn("residual_fused_val_shrunk", source_types)
            self.assertIn("standalone_local_graph", source_types)
            self.assertIn("score_fusion_control", source_types)
            self.assertTrue((output / "local_graph_residual_expert_report.json").exists())
            self.assertTrue((output / "residual_metric_table.csv").exists())
            self.assertTrue((output / "baseline_comparison.csv").exists())

    def test_baseline_rows_respect_report_max_jets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_logits"
            output = root / "final_report"
            self._make_baseline_cache(baseline)

            report = build_local_graph_residual_expert_report(
                LocalGraphResidualExpertReportConfig(
                    output_dir=str(output),
                    hlt_cache_dir=str(root / "unused_hlt_cache"),
                    baseline_logit_cache_dir=str(baseline),
                    residual_expert_root=str(root / "empty_residuals"),
                    residual_variants=(),
                    confirm_final_test=True,
                    require_all_residual_variants=False,
                    max_final_test_jets=4,
                )
            )

            final_baseline = [
                row for row in report["metric_table"]
                if row["source_type"] == "baseline" and row["split"] == "final_test"
            ][0]
            self.assertEqual(final_baseline["n_jets"], 4)

    def test_final_test_report_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "confirm_final_test"):
                LocalGraphResidualExpertReportConfig(
                    output_dir=str(Path(tmp) / "out"),
                    hlt_cache_dir="hlt",
                    baseline_logit_cache_dir="baseline",
                    residual_expert_root="residual",
                    comparison_split="final_test",
                )

    def test_cli_builds_report_config(self):
        script = _load_script_module()
        args = script.parse_args(
            [
                "--output-dir",
                "out",
                "--hlt-cache-dir",
                "hlt",
                "--baseline-logit-cache-dir",
                "baseline",
                "--residual-expert-root",
                "residual",
                "--residual-variants",
                "a",
                "b",
                "--skip-checkpoint-evaluation",
                "--allow-precomputed-evaluations",
                "--allow-missing-residual-variants",
                "--confirm-final-test",
            ]
        )
        config = script.build_config(args)
        self.assertFalse(config.evaluate_checkpoints)
        self.assertTrue(config.allow_precomputed_evaluations)
        self.assertFalse(config.require_all_residual_variants)
        self.assertEqual(config.residual_variants, ("a", "b"))

        unsafe_args = script.parse_args(
            [
                "--output-dir",
                "out",
                "--hlt-cache-dir",
                "hlt",
                "--baseline-logit-cache-dir",
                "baseline",
                "--residual-expert-root",
                "residual",
                "--skip-checkpoint-evaluation",
                "--confirm-final-test",
            ]
        )
        with self.assertRaisesRegex(ValueError, "unsafe"):
            script.build_config(unsafe_args)

    def test_precomputed_report_rejects_old_alpha_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_logits"
            residual_root = root / "residual_experts"
            output = root / "final_report"
            self._make_baseline_cache(baseline)
            self._make_residual_reports(residual_root)
            report_path = residual_root / "d" / "run_report.json"
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["alpha_shrinkage_model_val"].pop("shrinkage_applies_to", None)
            self._write_json(report_path, payload)

            report = build_local_graph_residual_expert_report(
                LocalGraphResidualExpertReportConfig(
                    output_dir=str(output),
                    hlt_cache_dir=str(root / "unused_hlt_cache"),
                    baseline_logit_cache_dir=str(baseline),
                    residual_expert_root=str(residual_root),
                    residual_variants=("d",),
                    confirm_final_test=True,
                    evaluate_checkpoints=False,
                    allow_precomputed_evaluations=True,
                )
            )

            self.assertFalse(report["ok"])
            self.assertTrue(any("learned_correction_delta" in problem for problem in report["problems"]))


if __name__ == "__main__":
    unittest.main()

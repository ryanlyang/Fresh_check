import json
import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.fusion import prediction_paths
from teacher_logit_reco.pn_first_experiment import (
    EXPERIMENT_STEP,
    DEFAULT_FIRST_EXPERIMENT_SPLITS,
    PredictionComparisonSpec,
    TeacherLogitParticleNetFirstExperimentConfig,
    build_first_experiment_report,
    comparison_delta_summary,
    prediction_metric_summary,
)


def write_prediction_metadata(root, model_name, split, *, accuracy, cross_entropy):
    _, meta_path = prediction_paths(root, model_name, split)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "split": split,
        "n_jets": 12,
        "model_kind": "synthetic",
        "allowed_inputs": "cached_fixed_hlt_only",
        "metrics": {"accuracy": float(accuracy), "cross_entropy": float(cross_entropy)},
    }
    meta_path.write_text(json.dumps(payload), encoding="utf-8")


class TeacherLogitParticleNetFirstExperimentConfigTests(unittest.TestCase):
    def test_config_defaults_to_modest_non_final_splits(self):
        cfg = TeacherLogitParticleNetFirstExperimentConfig(
            output_dir="out",
            teacher_checkpoint="teacher.pt",
        )
        self.assertEqual(cfg.teacher_architecture, "part")
        self.assertEqual(cfg.resolved_model_name, "pn_reco_to_part_teacher")
        self.assertEqual(cfg.splits, DEFAULT_FIRST_EXPERIMENT_SPLITS)
        self.assertEqual(cfg.max_train_jets, 50_000)
        self.assertEqual(cfg.max_val_jets, 10_000)
        self.assertEqual(cfg.max_prediction_jets_per_split, 50_000)

    def test_final_test_requires_confirmation(self):
        with self.assertRaises(ValueError):
            TeacherLogitParticleNetFirstExperimentConfig(
                output_dir="out",
                teacher_checkpoint="teacher.pt",
                splits=["stack_train", "stack_val", "final_test"],
            )
        cfg = TeacherLogitParticleNetFirstExperimentConfig(
            output_dir="out",
            teacher_checkpoint="teacher.pt",
            splits=["stack_train", "stack_val", "final_test"],
            confirm_final_test=True,
        )
        self.assertIn("final_test", cfg.splits)

    def test_prediction_metric_summary_and_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            pn_dir = Path(tmp) / "pn_predictions"
            hlt_dir = Path(tmp) / "hlt_predictions"
            for split in ("stack_train", "stack_val"):
                write_prediction_metadata(pn_dir, "pn_reco_to_part_teacher", split, accuracy=0.72, cross_entropy=0.9)
                write_prediction_metadata(hlt_dir, "hlt_baseline", split, accuracy=0.70, cross_entropy=1.0)

            primary = prediction_metric_summary(
                name="particle_net_reco",
                prediction_dir=pn_dir,
                model_name="pn_reco_to_part_teacher",
                splits=["stack_train", "stack_val"],
            )
            comparison = prediction_metric_summary(
                name="raw_hlt",
                prediction_dir=hlt_dir,
                model_name="hlt_baseline",
                splits=["stack_train", "stack_val"],
            )
            deltas = comparison_delta_summary(primary, [comparison])
            self.assertFalse(primary["splits"]["stack_val"]["missing"])
            self.assertAlmostEqual(deltas["stack_val"]["raw_hlt"]["metric_deltas"]["accuracy"], 0.02)
            self.assertAlmostEqual(deltas["stack_val"]["raw_hlt"]["metric_deltas"]["cross_entropy"], -0.1)

    def test_build_first_experiment_report_includes_comparison_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pn_dir = root / "experiment" / "predictions"
            comparison_dir = root / "comparison_predictions"
            write_prediction_metadata(pn_dir, "pn_reco_to_part_teacher", "stack_val", accuracy=0.73, cross_entropy=0.8)
            write_prediction_metadata(comparison_dir, "gt_reco_to_part_teacher", "stack_val", accuracy=0.71, cross_entropy=0.85)
            cfg = TeacherLogitParticleNetFirstExperimentConfig(
                output_dir=str(root / "experiment"),
                teacher_checkpoint="teacher.pt",
                splits=["stack_val"],
                comparison_specs=[
                    PredictionComparisonSpec(
                        name="gt_reco",
                        prediction_dir=str(comparison_dir),
                        model_name="gt_reco_to_part_teacher",
                    )
                ],
            )
            report = build_first_experiment_report(
                config=cfg,
                train_report={"checkpoint": "best_model_val.pt"},
                prediction_report={"model_name": "pn_reco_to_part_teacher"},
            )
            self.assertEqual(report["experiment_step"], EXPERIMENT_STEP)
            self.assertEqual(report["model_name"], "pn_reco_to_part_teacher")
            self.assertEqual(report["comparison_metrics"][0]["name"], "gt_reco")
            delta = report["comparison_deltas"]["stack_val"]["gt_reco"]["metric_deltas"]["accuracy"]
            self.assertAlmostEqual(delta, 0.02)


if __name__ == "__main__":
    unittest.main()

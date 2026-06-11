import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jetclass_fresh.fusion import prediction_paths
from teacher_logit_reco.particle_cnn_reconstructor import PARTICLE_CNN_ORDERING_ASSUMPTION
from teacher_logit_reco.pcnn_first_experiment import (
    EXPERIMENT_STEP,
    PredictionComparisonSpec,
    TeacherLogitParticleCnnFirstExperimentConfig,
    build_first_experiment_report,
    comparison_delta_summary,
    prediction_metric_summary,
    run_teacher_logit_particle_cnn_first_experiment,
)


class TeacherLogitParticleCnnFirstExperimentConfigTests(unittest.TestCase):
    def test_config_final_test_requires_confirmation_and_validates_shape(self):
        with self.assertRaises(ValueError):
            TeacherLogitParticleCnnFirstExperimentConfig(
                output_dir="out",
                teacher_checkpoint="teacher.pt",
                splits=["stack_val", "final_test"],
            )
        with self.assertRaises(ValueError):
            TeacherLogitParticleCnnFirstExperimentConfig(
                output_dir="out",
                teacher_checkpoint="teacher.pt",
                num_blocks=2,
                kernel_sizes=(5,),
                dilations=(1, 2),
            )
        cfg = TeacherLogitParticleCnnFirstExperimentConfig(
            output_dir="out",
            teacher_checkpoint="teacher.pt",
            splits=["stack_val", "final_test"],
            confirm_final_test=True,
            num_blocks=1,
            kernel_sizes=(3,),
            dilations=(1,),
        )
        self.assertEqual(cfg.resolved_model_name, "pcnn_reco_to_part_teacher")
        self.assertEqual(str(cfg.train_output_dir), str(Path("out") / "train" / "pcnn_reco_to_part_teacher"))


class TeacherLogitParticleCnnFirstExperimentReportTests(unittest.TestCase):
    def test_prediction_metric_summary_reads_saved_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, meta_path = prediction_paths(tmp, "pcnn_reco_to_part_teacher", "stack_val")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(
                json.dumps(
                    {
                        "n_jets": 12,
                        "metrics": {"accuracy": 0.71, "cross_entropy": 0.88},
                        "model_kind": "teacher_logit_particle_cnn_reco",
                        "allowed_inputs": "cached_fixed_hlt_only_then_reconstructed_soft_view_to_frozen_teacher",
                        "reconstructor_architecture": "particle_cnn",
                        "reconstructor_ordering_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
                        "experiment_step": "predict_step",
                    }
                ),
                encoding="utf-8",
            )
            summary = prediction_metric_summary(
                name="pcnn",
                prediction_dir=tmp,
                model_name="pcnn_reco_to_part_teacher",
                splits=["stack_val", "final_test"],
            )
            self.assertFalse(summary["splits"]["stack_val"]["missing"])
            self.assertEqual(summary["splits"]["stack_val"]["n_jets"], 12)
            self.assertEqual(summary["splits"]["stack_val"]["metrics"]["accuracy"], 0.71)
            self.assertTrue(summary["splits"]["final_test"]["missing"])

    def test_comparison_delta_summary_uses_matching_metrics(self):
        primary = {
            "splits": {
                "stack_val": {
                    "metrics": {"accuracy": 0.72, "cross_entropy": 0.82, "note": "skip"},
                }
            }
        }
        comparisons = [
            {
                "name": "raw_hlt",
                "model_name": "hlt_part",
                "splits": {
                    "stack_val": {
                        "missing": False,
                        "metrics": {"accuracy": 0.70, "cross_entropy": 0.90},
                    }
                },
            }
        ]
        deltas = comparison_delta_summary(primary, comparisons)
        self.assertAlmostEqual(deltas["stack_val"]["raw_hlt"]["metric_deltas"]["accuracy"], 0.02)
        self.assertAlmostEqual(deltas["stack_val"]["raw_hlt"]["metric_deltas"]["cross_entropy"], -0.08)
        self.assertNotIn("note", deltas["stack_val"]["raw_hlt"]["metric_deltas"])

    def test_build_report_records_pcnn_research_context(self):
        cfg = TeacherLogitParticleCnnFirstExperimentConfig(
            output_dir="out",
            teacher_checkpoint="teacher.pt",
            comparison_specs=[
                PredictionComparisonSpec(
                    name="raw_hlt",
                    prediction_dir="preds",
                    model_name="hlt_part",
                )
            ],
        )
        report = build_first_experiment_report(
            config=cfg,
            train_report={"checkpoint": "best.pt"},
            prediction_report={"model_name": cfg.resolved_model_name},
        )
        self.assertEqual(report["experiment_step"], EXPERIMENT_STEP)
        self.assertEqual(report["reconstructor_architecture"], "particle_cnn")
        self.assertEqual(report["ordering_assumption"], PARTICLE_CNN_ORDERING_ASSUMPTION)
        self.assertFalse(report["fits_final_stacker"])
        self.assertEqual(report["comparison_metrics"][0]["name"], "raw_hlt")

    def test_run_harness_orchestrates_train_and_predict_without_final_stacker(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TeacherLogitParticleCnnFirstExperimentConfig(
                output_dir=tmp,
                teacher_checkpoint="teacher.pt",
                epochs=1,
                num_blocks=1,
                kernel_sizes=(3,),
                dilations=(1,),
                max_train_jets=4,
                max_val_jets=4,
                max_prediction_jets_per_split=4,
            )
            train_report = {"checkpoint": str(Path(tmp) / "train" / "best_model_val.pt"), "best_epoch": 1}
            prediction_report = {"model_name": cfg.resolved_model_name, "prediction_dir": str(cfg.prediction_dir)}
            with patch(
                "teacher_logit_reco.pcnn_first_experiment.train_teacher_logit_particle_cnn_reco",
                return_value=train_report,
            ) as train_mock, patch(
                "teacher_logit_reco.pcnn_first_experiment.collect_teacher_logit_particle_cnn_predictions",
                return_value=prediction_report,
            ) as predict_mock:
                report = run_teacher_logit_particle_cnn_first_experiment(cfg)
            self.assertEqual(report["train_report"], train_report)
            self.assertEqual(report["prediction_report"], prediction_report)
            self.assertTrue((Path(tmp) / "first_experiment_report.json").exists())
            train_mock.assert_called_once()
            predict_mock.assert_called_once()

        with tempfile.TemporaryDirectory() as tmp:
            cfg = TeacherLogitParticleCnnFirstExperimentConfig(
                output_dir=tmp,
                teacher_checkpoint="teacher.pt",
                fit_final_stacker=True,
            )
            with self.assertRaises(NotImplementedError):
                run_teacher_logit_particle_cnn_first_experiment(cfg)


if __name__ == "__main__":
    unittest.main()

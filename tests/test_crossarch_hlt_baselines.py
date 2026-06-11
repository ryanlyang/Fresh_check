from pathlib import Path
import unittest

from teacher_logit_reco.crossarch_experiment import CrossArchExperimentLayout
from teacher_logit_reco.crossarch_hlt_baselines import (
    CrossArchHLTBaselinePredictionConfig,
    CrossArchHLTBaselineTrainConfig,
    crossarch_hlt_baseline_dir,
    crossarch_hlt_checkpoint_path,
    normalize_crossarch_hlt_architecture,
    split_size_for_crossarch_hlt_prediction,
)


class CrossArchHLTBaselineConfigTests(unittest.TestCase):
    def test_normalizes_architecture_and_paths(self):
        self.assertEqual(normalize_crossarch_hlt_architecture("ParticleTransformer"), "part")
        self.assertEqual(normalize_crossarch_hlt_architecture("ParticleNet"), "pn")
        self.assertEqual(normalize_crossarch_hlt_architecture("P-CNN"), "pcnn")
        layout = CrossArchExperimentLayout(output_root="/tmp/checkpoints")
        self.assertEqual(crossarch_hlt_baseline_dir("PFN", output_root="/tmp/checkpoints"), layout.hlt_baseline_dir("pfn"))
        self.assertEqual(
            crossarch_hlt_checkpoint_path("PFN", output_root="/tmp/checkpoints"),
            layout.hlt_baseline_dir("pfn") / "best_model_val.pt",
        )
        with self.assertRaises(ValueError):
            normalize_crossarch_hlt_architecture("bad")

    def test_train_config_allows_only_model_train_and_model_val(self):
        cfg = CrossArchHLTBaselineTrainConfig(
            architecture="pn",
            output_dir="out",
            cache_dir="cache",
        )
        self.assertEqual(cfg.architecture, "pn")
        with self.assertRaises(ValueError):
            CrossArchHLTBaselineTrainConfig(
                architecture="part",
                output_dir="out",
                cache_dir="cache",
                train_split="stack_train",
            )
        with self.assertRaises(ValueError):
            CrossArchHLTBaselineTrainConfig(
                architecture="part",
                output_dir="out",
                cache_dir="cache",
                epochs=0,
            )

    def test_prediction_config_guardrails_and_split_sizes(self):
        with self.assertRaises(ValueError):
            CrossArchHLTBaselinePredictionConfig(
                architecture="part",
                checkpoint="ckpt.pt",
                cache_dir="cache",
                prediction_dir="predictions",
                output_dir="run",
                splits=["stack_train", "stack_val", "final_test"],
                confirm_final_test=False,
            )
        cfg = CrossArchHLTBaselinePredictionConfig(
            architecture="part",
            checkpoint="ckpt.pt",
            cache_dir="cache",
            prediction_dir="predictions",
            output_dir="run",
            splits=["stack_train", "stack_val", "final_test"],
            confirm_final_test=True,
        )
        self.assertEqual(split_size_for_crossarch_hlt_prediction(cfg, "stack_train"), 500_000)
        self.assertEqual(split_size_for_crossarch_hlt_prediction(cfg, "stack_val"), 150_000)
        self.assertEqual(split_size_for_crossarch_hlt_prediction(cfg, "final_test"), 500_000)
        with self.assertRaises(ValueError):
            CrossArchHLTBaselinePredictionConfig(
                architecture="part",
                checkpoint="ckpt.pt",
                cache_dir="cache",
                prediction_dir="predictions",
                output_dir="run",
                splits=["model_train"],
            )


if __name__ == "__main__":
    unittest.main()

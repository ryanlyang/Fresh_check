import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.ensemble_analysis import (
    UncertaintyStackerConfig,
    build_uncertainty_feature_matrix,
    run_diversity_audit,
    run_uncertainty_feature_stackers,
)
from jetclass_fresh.fusion import PredictionBlock, load_blocks_for_split, save_prediction_block
from jetclass_fresh.jetclass_data import JetIdentity


def make_logits(labels, *, num_classes=3, strength=2.0, offset=0):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.zeros((len(labels), num_classes), dtype=np.float32)
    for row, label in enumerate(labels):
        target = (int(label) + int(offset)) % int(num_classes)
        logits[row, target] = float(strength)
        logits[row] += np.linspace(-0.1, 0.1, num_classes, dtype=np.float32)
    return logits


def write_prediction_fixture(prediction_dir, *, n_per_split=36):
    labels = np.tile(np.arange(3, dtype=np.int64), n_per_split // 3 + 1)[:n_per_split]
    model_specs = {
        "hlt_baseline": {"strength": 2.0, "offset": 0},
        "m2_base": {"strength": 2.4, "offset": 0},
        "m2_genlow": {"strength": 1.8, "offset": 0},
    }
    for split in ("stack_train", "stack_val", "final_test"):
        jet_ids = [
            JetIdentity(file=f"{split}.root", entry=index, label=int(label))
            for index, label in enumerate(labels)
        ]
        for model_name, spec in model_specs.items():
            logits = make_logits(labels, strength=spec["strength"], offset=spec["offset"])
            block = PredictionBlock(
                model_name=model_name,
                split=split,
                logits=logits,
                probs=np.zeros_like(logits),
                labels=labels.copy(),
                jet_ids=list(jet_ids),
                metadata={"fixture": True},
            )
            save_prediction_block(block, prediction_dir)


class EnsembleAnalysisTests(unittest.TestCase):
    def test_build_uncertainty_features_has_named_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            write_prediction_fixture(prediction_dir)
            blocks = load_blocks_for_split(prediction_dir, ["hlt_baseline", "m2_base"], "stack_train")
            features, columns = build_uncertainty_feature_matrix(blocks, feature_mode="mean_uncertainty")
            self.assertEqual(features.shape[0], len(blocks[0].labels))
            self.assertEqual(features.shape[1], len(columns))
            self.assertIn("ensemble_entropy", columns)
            self.assertIn("vote_disagreement", columns)

    def test_diversity_audit_and_uncertainty_stacker_write_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dir = root / "predictions"
            diversity_dir = root / "diversity"
            uncertainty_dir = root / "uncertainty"
            write_prediction_fixture(prediction_dir)

            diversity = run_diversity_audit(
                prediction_dir=prediction_dir,
                output_dir=diversity_dir,
                model_names=["hlt_baseline", "m2_base", "m2_genlow"],
                groups={
                    "m2_only": ["m2_base", "m2_genlow"],
                    "hlt_plus_m2": ["hlt_baseline", "m2_base", "m2_genlow"],
                },
                confirm_final_test=True,
            )
            self.assertTrue((diversity_dir / "diversity_report.json").exists())
            self.assertTrue((diversity_dir / "pairwise_diversity.csv").exists())
            final_oracle = [
                row for row in diversity["group_oracle_summary"]
                if row["group"] == "hlt_plus_m2" and row["split"] == "final_test"
            ][0]
            self.assertGreaterEqual(
                final_oracle["oracle_any_model_correct_accuracy"],
                final_oracle["best_single_model_accuracy"],
            )

            config = UncertaintyStackerConfig(
                prediction_dir=str(prediction_dir),
                output_dir=str(uncertainty_dir),
                model_names=["hlt_baseline", "m2_base", "m2_genlow"],
                groups={"hlt_plus_m2": ["hlt_baseline", "m2_base", "m2_genlow"]},
                feature_modes=["uncertainty", "mean_uncertainty"],
                c_grid=[0.1],
                max_iter=100,
                confirm_final_test=True,
            )
            report = run_uncertainty_feature_stackers(config)
            self.assertTrue((uncertainty_dir / "uncertainty_stacker_report.json").exists())
            self.assertTrue((uncertainty_dir / "uncertainty_stacker_metrics.csv").exists())
            modes = report["group_uncertainty_stacker_metrics"]["hlt_plus_m2"]["feature_modes"]
            self.assertIn("uncertainty", modes)
            self.assertIn("mean_uncertainty", modes)
            self.assertGreater(modes["mean_uncertainty"]["n_features"], modes["uncertainty"]["n_features"])


if __name__ == "__main__":
    unittest.main()

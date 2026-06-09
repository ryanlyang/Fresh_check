import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block
from jetclass_fresh.independent_fusion import (
    IndependentFusionConfig,
    default_groups_for_models,
    discover_prediction_models,
    run_independent_fusion,
    validate_fusion_groups,
)
from jetclass_fresh.jetclass_data import JetIdentity


def make_logits(labels, *, num_classes=3, strength=2.5, rotate=False):
    logits = np.zeros((len(labels), num_classes), dtype=np.float32)
    for index, label in enumerate(labels):
        target = (int(label) + 1) % num_classes if rotate else int(label)
        logits[index, target] = strength
        logits[index] += np.linspace(-0.2, 0.2, num_classes, dtype=np.float32)
    return logits


def write_blocks(prediction_dir, model_names=("hlt_baseline", "m2_base"), n_per_split=30):
    labels = np.tile(np.arange(3, dtype=np.int64), n_per_split // 3 + 1)[:n_per_split]
    for split_index, split in enumerate(("stack_train", "stack_val", "final_test")):
        jet_ids = [
            JetIdentity(file=f"{split}.root", entry=i, label=int(label))
            for i, label in enumerate(labels)
        ]
        for model_name in model_names:
            logits = make_logits(
                labels,
                strength=2.5 if model_name == "hlt_baseline" else 3.0,
                rotate=False,
            )
            if split_index == 2 and model_name == "hlt_baseline":
                logits = make_logits(labels, strength=2.0, rotate=False)
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


class IndependentFusionTests(unittest.TestCase):
    def test_discovers_models_and_default_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            write_blocks(prediction_dir)
            models = discover_prediction_models(prediction_dir)
            self.assertEqual(models, ["hlt_baseline", "m2_base"])
            groups = default_groups_for_models(models)
            self.assertEqual(groups["m2_only"], ["m2_base"])
            self.assertEqual(groups["hlt_plus_m2"], ["hlt_baseline", "m2_base"])

    def test_run_independent_fusion_writes_required_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dir = root / "predictions"
            output_dir = root / "fusion"
            write_blocks(prediction_dir)
            config = IndependentFusionConfig(
                prediction_dir=str(prediction_dir),
                output_dir=str(output_dir),
                model_names=["hlt_baseline", "m2_base"],
                groups={"hlt_plus_m2": ["hlt_baseline", "m2_base"]},
                feature_modes=["logits_probs"],
                c_grid=[0.1, 1.0],
                max_iter=200,
                confirm_final_test=True,
                run_controls=False,
            )
            report = run_independent_fusion(config)
            self.assertTrue((output_dir / "fusion_report.json").exists())
            self.assertTrue((output_dir / "raw_source_metrics.csv").exists())
            self.assertTrue((output_dir / "singleton_stacker_metrics.csv").exists())
            self.assertTrue((output_dir / "group_fusion_metrics.csv").exists())
            self.assertTrue((output_dir / "controls.json").exists())
            self.assertTrue((output_dir / "stack_split_hash_audit.json").exists())
            self.assertTrue(report["stack_split_hash_audit"]["ok"])
            final = report["group_fusion_metrics"]["hlt_plus_m2"]["feature_modes"]["logits_probs"]["metrics"]["final_test"]
            self.assertGreaterEqual(final["accuracy"], 0.0)
            self.assertEqual(report["leakage_rules"]["stacker_fit_split"], "stack_train")
            self.assertEqual(report["leakage_rules"]["stacker_selection_split"], "stack_val")

    def test_offline_teacher_is_rejected_from_fusion_groups(self):
        with self.assertRaises(ValueError):
            validate_fusion_groups(
                {"bad": ["hlt_baseline", "offline_teacher"]},
                ["hlt_baseline", "offline_teacher"],
            )


if __name__ == "__main__":
    unittest.main()

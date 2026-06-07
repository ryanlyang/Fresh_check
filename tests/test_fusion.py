from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.fusion import (
    FusionRunConfig,
    PredictionBlock,
    classification_metrics_from_logits,
    evaluate_fusion_methods,
    load_prediction_block,
    run_reco7_fusion,
    save_prediction_block,
    select_weighted_average_weights,
    softmax_np,
    stack_feature_matrix,
)
from jetclass_fresh.jetclass_data import JetIdentity


def make_block(model_name, split, labels, *, strength=3.0, offset=0):
    n_classes = int(np.max(labels)) + 1
    logits = np.full((len(labels), n_classes), -1.0, dtype=np.float32)
    for row, label in enumerate(labels):
        logits[row, int(label)] = strength
        logits[row] += 0.05 * np.sin(row + offset + np.arange(n_classes))
    jet_ids = [JetIdentity(file=f"{split}.root", entry=i, label=int(label)) for i, label in enumerate(labels)]
    return PredictionBlock(
        model_name=model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=np.asarray(labels, dtype=np.int64),
        jet_ids=jet_ids,
        metadata={"model_kind": "synthetic"},
    )


class FusionStep10Tests(unittest.TestCase):
    def test_prediction_block_roundtrip_and_metrics(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        block = make_block("model_a", "stack_val", labels)
        with tempfile.TemporaryDirectory() as tmp:
            metadata = save_prediction_block(block, tmp)
            loaded = load_prediction_block(tmp, "model_a", "stack_val")
        self.assertEqual(metadata["n_jets"], 6)
        self.assertEqual(loaded.jet_ids, block.jet_ids)
        self.assertTrue(np.allclose(loaded.probs, block.probs))
        self.assertEqual(classification_metrics_from_logits(loaded.logits, loaded.labels)["accuracy"], 1.0)

    def test_stack_feature_matrix_and_weight_selection(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        blocks = [
            make_block("model_a", "stack_val", labels, strength=3.0),
            make_block("model_b", "stack_val", labels, strength=2.0, offset=5),
        ]
        features = stack_feature_matrix(blocks, feature_mode="logits_probs")
        self.assertEqual(features.shape, (6, 12))
        weights, report = select_weighted_average_weights(blocks, mode="probs", max_steps=3)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertEqual(report["mode"], "probs")

    def test_evaluate_fusion_methods_with_synthetic_predictions(self):
        labels = np.asarray([0, 1, 2] * 8, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            for split in ["stack_train", "stack_val", "final_test"]:
                save_prediction_block(make_block("model_a", split, labels, strength=3.0), tmp)
                save_prediction_block(make_block("model_b", split, labels, strength=1.5, offset=7), tmp)
            result = evaluate_fusion_methods(
                tmp,
                ["model_a", "model_b"],
                C_grid=[0.1, 1.0],
                max_iter=100,
            )
        report = result["report"]
        self.assertIn("stacked_logistic_regression", report)
        self.assertEqual(report["stacked_logistic_regression"]["metrics"]["final_test"]["accuracy"], 1.0)
        self.assertTrue(report["final_test_evaluated"])

    def test_final_test_requires_explicit_confirmation(self):
        cfg = FusionRunConfig(
            output_dir="/tmp/unused",
            hlt_cache_dir="/tmp/hlt",
            hlt_checkpoint="/tmp/hlt.pt",
            splits=["stack_train", "stack_val", "final_test"],
            confirm_final_test=False,
        )
        with self.assertRaises(ValueError):
            run_reco7_fusion(cfg)


if __name__ == "__main__":
    unittest.main()

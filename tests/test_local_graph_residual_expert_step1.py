import tempfile
import unittest
from pathlib import Path

import numpy as np

from teacher_logit_reco.local_graph_part.fusion import binary_logits_from_log_odds
from teacher_logit_reco.local_graph_part.residual_cache import (
    LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES,
    LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CONTRACT,
    LocalGraphBaselineLogitBlock,
    baseline_condition_features,
    baseline_logit_cache_paths,
    load_baseline_logit_block,
    operating_point_from_scores,
    save_baseline_logit_block,
)


class LocalGraphResidualExpertStep1Tests(unittest.TestCase):
    def test_operating_point_matches_project_threshold_convention(self):
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
        scores = np.asarray([0.9, 0.4, 0.8, 0.1], dtype=np.float64)

        op50 = operating_point_from_scores(labels, scores, 0.50)
        op100 = operating_point_from_scores(labels, scores, 1.00)

        self.assertEqual(op50["threshold"], 0.9)
        self.assertEqual(op50["signal_efficiency"], 0.5)
        self.assertEqual(op50["false_positive_rate"], 0.0)
        self.assertEqual(op100["threshold"], 0.4)
        self.assertEqual(op100["signal_efficiency"], 1.0)
        self.assertEqual(op100["false_positive_rate"], 0.5)

    def test_baseline_condition_features_are_finite_and_named(self):
        z_base = np.asarray([-2.0, -0.1, 0.2, 1.5], dtype=np.float64)

        features, metadata = baseline_condition_features(z_base, tau50=0.2, tau30=1.0)

        self.assertEqual(features.shape, (4, len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES)))
        self.assertEqual(tuple(metadata["feature_names"]), LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[:, 0], z_base, rtol=0, atol=1e-6)
        np.testing.assert_allclose(features[:, 2], z_base - 0.2, rtol=0, atol=1e-6)
        self.assertGreaterEqual(float(features[:, -1].min()), 0.0)
        self.assertLessEqual(float(features[:, -1].max()), 1.0)

    def test_save_and_load_baseline_logit_cache_contract(self):
        margins = np.asarray([-2.0, 1.5, 0.5, -0.25, 2.0, -1.0], dtype=np.float64)
        labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int64)
        indices = np.arange(labels.shape[0], dtype=np.int64) + 10
        logits = binary_logits_from_log_odds(margins).astype(np.float32)
        block = LocalGraphBaselineLogitBlock(
            split="model_val",
            logits=logits,
            labels=labels,
            indices=indices,
            metadata={"checkpoint": "baseline.pt", "checkpoint_epoch": 3},
        )

        with tempfile.TemporaryDirectory() as tmp:
            report = save_baseline_logit_block(block, tmp)
            npz_path, metadata_path = baseline_logit_cache_paths(tmp, "model_val")
            self.assertTrue(npz_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(report["contract"], LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CONTRACT)
            self.assertIn("operating_points", report)
            self.assertIn("condition_features", report)

            with np.load(npz_path, allow_pickle=False) as data:
                self.assertIn("z_base", data.files)
                self.assertIn("p_base", data.files)
                self.assertIn("condition_features", data.files)
                self.assertEqual(data["condition_features"].shape[1], len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES))

            loaded = load_baseline_logit_block(Path(tmp), "model_val", require_metadata=True)
            np.testing.assert_allclose(loaded.logits, logits)
            np.testing.assert_array_equal(loaded.labels, labels)
            np.testing.assert_array_equal(loaded.indices, indices)
            self.assertEqual(loaded.metadata["contract"], LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CONTRACT)

    def test_save_cache_can_defer_split_metrics(self):
        margins = np.asarray([-0.5, 0.5], dtype=np.float64)
        labels = np.asarray([0, 1], dtype=np.int64)
        logits = binary_logits_from_log_odds(margins).astype(np.float32)
        block = LocalGraphBaselineLogitBlock(
            split="final_test",
            logits=logits,
            labels=labels,
            indices=np.arange(labels.shape[0], dtype=np.int64),
            metadata={"checkpoint": "baseline.pt", "checkpoint_epoch": 3},
        )

        with tempfile.TemporaryDirectory() as tmp:
            report = save_baseline_logit_block(block, tmp, compute_metrics=False)

            self.assertFalse(report["metrics_computed"])
            self.assertIsNone(report["metrics"])
            self.assertIsNone(report["operating_points"])
            loaded = load_baseline_logit_block(Path(tmp), "final_test", require_metadata=True)
            self.assertFalse(loaded.metadata["metrics_computed"])


if __name__ == "__main__":
    unittest.main()

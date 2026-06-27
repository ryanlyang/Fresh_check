import tempfile
import unittest
from pathlib import Path

import numpy as np

from teacher_logit_reco.local_graph_part.fusion import (
    LocalGraphPredictionBlock,
    binary_metrics_from_signal_scores,
    build_score_feature_block,
    fit_binary_logistic_stackers_selecting_c,
    load_prediction_block,
    save_prediction_block,
    select_weighted_average_on_stack,
    validate_prediction_alignment,
)


def make_block(variant: str, scores: np.ndarray, labels: np.ndarray) -> LocalGraphPredictionBlock:
    scores = np.asarray(scores, dtype=np.float32)
    logits = np.stack([-0.5 * scores, 0.5 * scores], axis=1).astype(np.float32)
    return LocalGraphPredictionBlock(
        variant=variant,
        split="stack_val",
        logits=logits,
        labels=np.asarray(labels, dtype=np.int64),
        indices=np.arange(len(labels), dtype=np.int64),
    )


class LocalGraphScoreFusionStep1Tests(unittest.TestCase):
    def test_binary_metrics_use_lower_fpr_at_signal_efficiency(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        good_scores = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        bad_scores = -good_scores

        good = binary_metrics_from_signal_scores(good_scores, labels)
        bad = binary_metrics_from_signal_scores(bad_scores, labels)

        self.assertLess(
            good["binary_metrics"]["fpr_at_signal_eff_0p50"],
            bad["binary_metrics"]["fpr_at_signal_eff_0p50"],
        )

    def test_prediction_alignment_rejects_label_mismatch(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        left = make_block("hlt_part_baseline", [-1.0, 1.0, -0.5, 0.5], labels)
        right = make_block("local_edgeconv_adapter", [-1.0, 1.0, -0.5, 0.5], 1 - labels)

        with self.assertRaisesRegex(ValueError, "Label mismatch|label mismatch"):
            validate_prediction_alignment([left, right])

    def test_weighted_average_selects_better_model_by_fpr50(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        bad = make_block("hlt_part_baseline", [3.0, 2.0, 1.0, -1.0, -2.0, -3.0], labels)
        good = make_block("local_point_attention_adapter", [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], labels)
        features = build_score_feature_block([bad, good], mode="margin")

        selection, rows = select_weighted_average_on_stack(features, step=1.0)

        self.assertEqual(len(rows), 2)
        self.assertEqual(selection.weights.tolist(), [0.0, 1.0])
        self.assertEqual(selection.stack_metrics["binary_metrics"]["fpr_at_signal_eff_0p50"], 0.0)

    def test_numpy_logistic_stacker_predicts_scores(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        block = make_block("hlt_part_baseline", [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], labels)
        features = build_score_feature_block([block], mode="margin")

        stacker, selection = fit_binary_logistic_stackers_selecting_c(
            features,
            c_grid=[0.1],
            max_iter=50,
            prefer_sklearn=False,
        )

        scores = stacker.predict_score(features.features)
        self.assertEqual(scores.shape, labels.shape)
        self.assertIn("selected_metric_value", selection)

    def test_prediction_cache_round_trips_core_arrays(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        block = make_block("hlt_part_baseline", [-2.0, 2.0, -1.0, 1.0], labels)
        with tempfile.TemporaryDirectory() as tmp:
            metadata = save_prediction_block(block, tmp)
            loaded = load_prediction_block(tmp, "hlt_part_baseline", "stack_val")

            self.assertTrue((Path(tmp) / "hlt_part_baseline" / "stack_val_predictions.npz").exists())
            self.assertEqual(metadata["variant"], "hlt_part_baseline")
            self.assertTrue(np.array_equal(loaded.labels, block.labels))
            self.assertTrue(np.allclose(loaded.logits, block.logits))


if __name__ == "__main__":
    unittest.main()


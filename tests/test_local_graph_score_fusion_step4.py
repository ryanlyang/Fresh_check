import unittest
from unittest.mock import patch

import numpy as np

from teacher_logit_reco.local_graph_part.fusion import (
    BinaryLogisticStacker,
    FusionFeatureBlock,
    LocalGraphPredictionBlock,
    binary_metrics_from_signal_scores,
    build_score_feature_block,
    fusion_metric_score,
    select_weighted_average_on_stack,
    shuffle_non_baseline_columns,
)
from scripts import run_local_graph_score_fusion as fusion_runner


def make_block(
    variant: str,
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    *,
    split: str = "stack_val",
) -> LocalGraphPredictionBlock:
    scores = np.asarray(scores, dtype=np.float32)
    logits = np.stack([-0.5 * scores, 0.5 * scores], axis=1).astype(np.float32)
    return LocalGraphPredictionBlock(
        variant=variant,
        split=split,
        logits=logits,
        labels=np.asarray(labels, dtype=np.int64),
        indices=np.arange(len(labels), dtype=np.int64),
    )


class LocalGraphScoreFusionStep4Tests(unittest.TestCase):
    def test_fpr50_selection_direction_is_lower_is_better(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        good_scores = np.asarray([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
        bad_scores = -good_scores

        good_metrics = binary_metrics_from_signal_scores(good_scores, labels)
        bad_metrics = binary_metrics_from_signal_scores(bad_scores, labels)
        good_score, good_value = fusion_metric_score(good_metrics, "fpr_at_signal_eff_0p50")
        bad_score, bad_value = fusion_metric_score(bad_metrics, "fpr_at_signal_eff_0p50")

        self.assertLess(good_value, bad_value)
        self.assertGreater(good_score, bad_score)

    def test_baseline_only_calibration_uses_one_baseline_feature(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        baseline = make_block("hlt_part_baseline", [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], labels)
        features = build_score_feature_block([baseline], mode="margin")

        stacker, selection = fusion_runner.fit_binary_logistic_stackers_selecting_c(
            features,
            c_grid=[0.1],
            max_iter=50,
            selection_metric="fpr_at_signal_eff_0p50",
            prefer_sklearn=False,
        )

        self.assertEqual(stacker.variants, ("hlt_part_baseline",))
        self.assertEqual(stacker.feature_names, ("hlt_part_baseline__margin",))
        self.assertEqual(stacker.coef.shape, (1,))
        self.assertEqual(selection["selection_metric"], "fpr_at_signal_eff_0p50")

    def test_weighted_average_selection_can_choose_true_mixture(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        left = make_block("hlt_part_baseline", [-3.0, -2.0, 2.0, 3.0, -1.0, 1.0, 2.0, 3.0], labels)
        right = make_block("local_edgeconv_adapter", [3.0, 2.0, -2.0, -3.0, 1.0, -1.0, 2.0, 3.0], labels)
        features = build_score_feature_block([left, right], mode="margin")

        selection, candidates = select_weighted_average_on_stack(
            features,
            step=0.5,
            selection_metric="fpr_at_signal_eff_0p50",
        )

        self.assertEqual(len(candidates), 3)
        self.assertTrue(np.allclose(selection.weights, np.asarray([0.5, 0.5])))
        self.assertEqual(selection.stack_metrics["binary_metrics"]["fpr_at_signal_eff_0p50"], 0.0)

    def test_logistic_runner_fits_on_stack_val_not_final_test(self):
        stack_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        final_labels = 1 - stack_labels
        stack_blocks = [
            make_block("hlt_part_baseline", [-2.0, -1.0, 1.0, 2.0], stack_labels, split="stack_val"),
            make_block("local_edgeconv_adapter", [-1.0, -2.0, 2.0, 1.0], stack_labels, split="stack_val"),
        ]
        final_blocks = [
            make_block("hlt_part_baseline", [-2.0, -1.0, 1.0, 2.0], final_labels, split="final_test"),
            make_block("local_edgeconv_adapter", [-1.0, -2.0, 2.0, 1.0], final_labels, split="final_test"),
        ]
        blocks_by_split = {
            "stack_val": {block.variant: block for block in stack_blocks},
            "final_test": {block.variant: block for block in final_blocks},
        }
        captured_fit_labels: list[np.ndarray] = []

        def fake_fit(feature_block, **_kwargs):
            captured_fit_labels.append(feature_block.labels.copy())
            stacker = BinaryLogisticStacker(
                coef=np.ones(feature_block.features.shape[1], dtype=np.float64),
                intercept=0.0,
                mean=np.zeros(feature_block.features.shape[1], dtype=np.float64),
                scale=np.ones(feature_block.features.shape[1], dtype=np.float64),
                C=1.0,
                feature_names=feature_block.feature_names,
                variants=feature_block.variants,
                solver="test_stub",
            )
            return stacker, {
                "selected_metrics": binary_metrics_from_signal_scores(
                    stacker.predict_score(feature_block.features),
                    feature_block.labels,
                ),
                "selection_metric": "fpr_at_signal_eff_0p50",
            }

        rows: list[dict] = []
        weight_rows: list[dict] = []
        with patch.object(fusion_runner, "fit_binary_logistic_stackers_selecting_c", side_effect=fake_fit):
            fusion_runner._add_logistic_rows(
                rows,
                weight_rows,
                blocks_by_split=blocks_by_split,
                model_sets=[("hlt_part_baseline", "local_edgeconv_adapter")],
                primary_metric="fpr_at_signal_eff_0p50",
                c_grid=[1.0],
                max_iter=3,
                prefer_sklearn=False,
            )

        self.assertTrue(captured_fit_labels)
        for labels in captured_fit_labels:
            self.assertTrue(np.array_equal(labels, stack_labels))
            self.assertFalse(np.array_equal(labels, final_labels))
        self.assertTrue(rows)

    def test_row_shuffled_controls_keep_baseline_and_shuffle_local_columns(self):
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        baseline = make_block("hlt_part_baseline", [-3.0, 3.0, -2.0, 2.0, -1.0, 1.0], labels)
        local = make_block("local_point_attention_adapter", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], labels)
        features = build_score_feature_block([baseline, local], mode="margin")

        shuffled = shuffle_non_baseline_columns(
            features,
            baseline_variant="hlt_part_baseline",
            seed=17,
        )

        self.assertTrue(np.array_equal(shuffled.labels, features.labels))
        self.assertTrue(np.allclose(shuffled.features[:, 0], features.features[:, 0]))
        self.assertCountEqual(shuffled.features[:, 1].tolist(), features.features[:, 1].tolist())
        self.assertFalse(np.array_equal(shuffled.features[:, 1], features.features[:, 1]))
        self.assertEqual(shuffled.feature_names[0], "hlt_part_baseline__margin")
        self.assertTrue(shuffled.feature_names[1].startswith("row_shuffled__local_point_attention_adapter"))


if __name__ == "__main__":
    unittest.main()

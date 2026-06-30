import unittest

import numpy as np

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_RESIDUAL_V2_GAMMA_SHRINKAGE_GRID,
    select_local_graph_residual_v2_gamma_shrinkage,
)
from teacher_logit_reco.local_graph_part.fusion import binary_logits_from_log_odds


class LocalGraphResidualV2Step9ShrinkageTest(unittest.TestCase):
    def test_gamma_shrinkage_applies_to_learned_correction_and_can_help(self):
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
        baseline = np.asarray([0.5, 0.0, 0.6, -1.0], dtype=np.float32)
        learned_correction = np.asarray([0.0, 0.0, -1.0, 0.0], dtype=np.float32)

        report = select_local_graph_residual_v2_gamma_shrinkage(
            labels=labels,
            baseline_logits=binary_logits_from_log_odds(baseline),
            correction_logits=binary_logits_from_log_odds(learned_correction),
            gamma_grid=(0.0, 0.1, 0.2, 0.5),
        )

        self.assertEqual(report["shrinkage_applies_to"], "learned_correction_delta")
        self.assertEqual(report["score_formula"], "z_base + gamma_val * (learned_gamma * delta)")
        self.assertGreater(report["selected_gamma"], 0.0)
        self.assertLess(report["selected_fpr"], report["baseline_fpr"])

    def test_gamma_shrinkage_selects_zero_when_learned_correction_hurts(self):
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
        baseline = np.asarray([0.5, 0.0, 0.4, -1.0], dtype=np.float32)
        learned_correction = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        report = select_local_graph_residual_v2_gamma_shrinkage(
            labels=labels,
            baseline_logits=binary_logits_from_log_odds(baseline),
            correction_logits=binary_logits_from_log_odds(learned_correction),
            gamma_grid=(0.0, 0.2, 0.5, 1.0),
        )

        self.assertEqual(report["selected_gamma"], 0.0)
        self.assertTrue(report["collapsed_to_zero"])
        self.assertEqual(report["selected_fpr"], report["baseline_fpr"])

    def test_default_grid_matches_plan_shape(self):
        self.assertEqual(LOCAL_GRAPH_RESIDUAL_V2_GAMMA_SHRINKAGE_GRID[0], 0.0)
        self.assertIn(0.01, LOCAL_GRAPH_RESIDUAL_V2_GAMMA_SHRINKAGE_GRID)
        self.assertIn(1.5, LOCAL_GRAPH_RESIDUAL_V2_GAMMA_SHRINKAGE_GRID)


if __name__ == "__main__":
    unittest.main()

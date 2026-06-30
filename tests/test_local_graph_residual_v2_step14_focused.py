import unittest

from teacher_logit_reco.local_graph_part.residual_v2_report import _best_row, _metric_direction


class LocalGraphResidualV2Step14FocusedTest(unittest.TestCase):
    def test_report_ranking_prefers_lower_final_test_fpr50_over_accuracy(self):
        rows = [
            {
                "source_type": "baseline",
                "variant": "accurate_but_bad_fpr",
                "split": "final_test",
                "primary_metric": "fpr_at_signal_eff_0p50",
                "primary_metric_value": 0.30,
                "accuracy": 0.99,
            },
            {
                "source_type": "v2_residual_fused_val_shrunk",
                "variant": "worse_accuracy_better_fpr",
                "split": "final_test",
                "primary_metric": "fpr_at_signal_eff_0p50",
                "primary_metric_value": 0.12,
                "accuracy": 0.80,
            },
        ]

        best = _best_row(rows, _metric_direction("fpr_at_signal_eff_0p50"))

        self.assertEqual(best["variant"], "worse_accuracy_better_fpr")
        self.assertLess(best["primary_metric_value"], rows[0]["primary_metric_value"])

    def test_metric_direction_for_binary_report_metric_is_minimize(self):
        self.assertEqual(_metric_direction("fpr_at_signal_eff_0p50"), "minimize")
        self.assertEqual(_metric_direction("background_rejection_at_signal_eff_0p50"), "maximize")


if __name__ == "__main__":
    unittest.main()

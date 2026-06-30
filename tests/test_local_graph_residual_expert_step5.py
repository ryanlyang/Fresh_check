import unittest

import numpy as np

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_CONTRACT,
    LocalGraphResidualDiagnosticsConfig,
    binary_logits_from_log_odds,
    residual_correction_diagnostics,
)


class LocalGraphResidualExpertStep5Tests(unittest.TestCase):
    def test_diagnostics_measure_removed_false_positive_overlap(self):
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
        baseline_margin = np.asarray([1.0, 0.2, 1.1, -1.0], dtype=np.float64)
        fused_margin = np.asarray([1.0, 0.2, 0.0, -1.0], dtype=np.float64)
        residual_margin = fused_margin - baseline_margin

        report = residual_correction_diagnostics(
            labels=labels,
            baseline_logits=binary_logits_from_log_odds(baseline_margin),
            fused_logits=binary_logits_from_log_odds(fused_margin),
            residual_logits=binary_logits_from_log_odds(residual_margin),
            indices=np.asarray([10, 11, 12, 13], dtype=np.int64),
            config=LocalGraphResidualDiagnosticsConfig(
                near_tau_fraction=1.0,
                include_index_samples=True,
            ),
        )

        self.assertEqual(report["contract"], LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_CONTRACT)
        self.assertLess(report["fused_delta_FPR50_vs_baseline"], 0.0)
        fp = report["false_positive_overlap"]
        self.assertEqual(fp["baseline_fp_count"], 1)
        self.assertEqual(fp["fused_fp_count"], 0)
        self.assertEqual(fp["old_false_positives_removed"], 1)
        self.assertEqual(fp["new_false_positives_introduced"], 0)
        self.assertEqual(fp["old_false_positive_removed_fraction"], 1.0)
        self.assertEqual(report["index_samples"]["old_false_positives_removed"], [12])

    def test_diagnostics_report_boundary_and_residual_summaries(self):
        labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64)
        baseline_margin = np.asarray([1.2, 0.5, -0.1, 1.1, 0.2, -0.8], dtype=np.float64)
        residual_margin = np.asarray([0.1, 0.2, 0.3, -0.8, -0.2, 0.0], dtype=np.float64)
        fused_margin = baseline_margin + residual_margin

        report = residual_correction_diagnostics(
            labels=labels,
            baseline_logit=baseline_margin,
            fused_logit=fused_margin,
            residual_logit=residual_margin,
            alpha_report={
                "selected_alpha": 0.5,
                "selected_fpr": 0.1,
                "baseline_fpr": 0.2,
                "delta_fpr_vs_baseline": -0.1,
                "collapsed_to_zero": False,
            },
        )

        self.assertIn("boundary_corrections", report)
        self.assertIn("region_shift_summary", report)
        self.assertIn("residual_summary", report)
        self.assertEqual(report["alpha_summary"]["selected_alpha"], 0.5)
        self.assertGreaterEqual(report["residual_summary"]["abs_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()

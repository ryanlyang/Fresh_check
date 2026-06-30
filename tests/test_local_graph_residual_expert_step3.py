import unittest

from jetclass_fresh.hlt_baseline import require_torch

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE,
    LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
    LocalGraphResidualLossConfig,
    boundary_pairwise_loss,
    compute_local_graph_residual_loss,
    residual_bce_weights,
    select_alpha_shrinkage,
    soft_fpr50_loss,
)


_TORCH = require_torch()


class LocalGraphResidualExpertStep3Tests(unittest.TestCase):
    def labels_and_baseline(self):
        torch = require_torch()
        labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)
        baseline = torch.tensor([1.2, 0.2, -0.6, 1.0, 0.3, -1.4], dtype=torch.float32)
        return labels, baseline

    def config(self, mode=LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE):
        return LocalGraphResidualLossConfig(
            mode=mode,
            default_tau50=0.2,
            pairwise_temperature=0.5,
            soft_fpr_epsilon=0.5,
            residual_l2_weight=1.0e-3,
            bce_anchor_weight=0.1,
            soft_fpr_weight=0.25,
            cvar_top_fraction=1.0,
            hard_background_fraction=0.5,
            bce_boundary_scale=1.0,
        )

    def test_weighted_bce_emphasizes_baseline_boundary_mistakes(self):
        labels, baseline = self.labels_and_baseline()
        weights = residual_bce_weights(labels, baseline, tau50=0.2, config=self.config())

        hard_qcd = weights[3]
        easy_qcd = weights[5]
        boundary_signal = weights[1]

        self.assertGreater(float(hard_qcd), float(easy_qcd))
        self.assertGreater(float(boundary_signal), 0.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)

    def test_boundary_pairwise_loss_rewards_pushing_qcd_below_signal_boundary(self):
        torch = require_torch()
        labels, baseline = self.labels_and_baseline()
        config = self.config(LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE)
        bad_fused = torch.tensor([1.2, 0.2, -0.6, 1.1, 0.7, -1.4], dtype=torch.float32)
        good_fused = torch.tensor([1.2, 0.2, -0.6, -0.4, -0.8, -1.4], dtype=torch.float32)

        bad_loss, bad_diag = boundary_pairwise_loss(bad_fused, labels, baseline, tau50=0.2, config=config)
        good_loss, good_diag = boundary_pairwise_loss(good_fused, labels, baseline, tau50=0.2, config=config)

        self.assertGreater(float(bad_loss), float(good_loss))
        self.assertGreater(float(bad_diag["pair_count"]), 0.0)
        self.assertEqual(float(bad_diag["pair_count"]), float(good_diag["pair_count"]))

    def test_soft_fpr_loss_rewards_qcd_below_threshold(self):
        torch = require_torch()
        labels, _baseline = self.labels_and_baseline()
        config = self.config(LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR)
        high_qcd = torch.tensor([1.0, 0.3, -0.5, 1.0, 0.6, -0.2], dtype=torch.float32)
        low_qcd = torch.tensor([1.0, 0.3, -0.5, -1.0, -0.6, -0.2], dtype=torch.float32)

        high_loss, _ = soft_fpr50_loss(high_qcd, labels, tau50=0.2, config=config)
        low_loss, _ = soft_fpr50_loss(low_qcd, labels, tau50=0.2, config=config)

        self.assertGreater(float(high_loss), float(low_loss))

    def test_all_loss_modes_are_finite_and_backprop(self):
        torch = require_torch()
        labels, baseline = self.labels_and_baseline()
        residual = torch.tensor([0.1, 0.2, -0.1, -0.4, -0.2, 0.1], dtype=torch.float32, requires_grad=True)
        alpha = torch.tensor(0.5, dtype=torch.float32)
        modes = (
            LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
            LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE,
            LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
            LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
        )
        for mode in modes:
            residual.grad = None
            fused = baseline + alpha * residual
            output = compute_local_graph_residual_loss(
                fused_logit=fused,
                labels=labels,
                baseline_logit=baseline,
                residual_logit=residual,
                alpha=alpha,
                tau50=0.2,
                config=self.config(mode),
            )
            self.assertTrue(torch.isfinite(output.total_loss))
            output.total_loss.backward(retain_graph=True)
            self.assertIsNotNone(residual.grad)
            self.assertTrue(torch.isfinite(residual.grad).all())

    def test_ladder_e_is_report_policy_not_training_loss(self):
        with self.assertRaisesRegex(ValueError, "no longer a separate training loss"):
            self.config("E")

    def test_alpha_shrinkage_selects_useful_positive_alpha(self):
        labels = [1, 1, 0, 0]
        baseline = [0.7, 0.4, 0.8, -0.2]
        residual = [0.1, 0.1, -1.5, 0.0]

        report = select_alpha_shrinkage(
            labels=labels,
            baseline_logit=baseline,
            residual_logit=residual,
            alpha_grid=(0.0, 0.25, 0.5, 1.0),
            target_signal_efficiency=0.5,
        )

        self.assertGreater(report["selected_alpha"], 0.0)
        self.assertLess(report["selected_fpr"], report["baseline_fpr"])
        self.assertFalse(report["collapsed_to_zero"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import math
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.losses import (
        SetMatchingLossConfig,
        compute_set_matching_loss,
        pairwise_core_cost,
        wrapped_delta_phi,
    )
else:  # pragma: no cover - environment dependent
    torch = None


def make_feature_tensor(rows):
    values = torch.zeros((len(rows), 14), dtype=torch.float32)
    for index, (pt, eta, phi, energy) in enumerate(rows):
        values[index, 0] = float(pt)
        values[index, 1] = float(eta)
        values[index, 2] = float(phi)
        values[index, 3] = float(energy)
        values[index, 4] = 1.0
        values[index, 5] = 1.0
    return values


def isolated_assignment_config():
    return SetMatchingLossConfig(
        matched_aux_weight=0.0,
        existence_weight=0.0,
        count_weight=0.0,
        jet_summary_weight=0.0,
        correction_budget_weight=0.0,
        chamfer_weight=0.0,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SetMatchingLossTests(unittest.TestCase):
    def test_hungarian_loss_is_permutation_invariant(self):
        pred = make_feature_tensor(
            [
                (10.0, 0.0, 0.1, 10.5),
                (20.0, 0.4, -0.2, 21.0),
                (5.0, -0.3, 0.8, 5.2),
            ]
        ).unsqueeze(0)
        target = pred.clone()
        permuted_target = target[:, [2, 0, 1], :]
        mask = torch.ones((1, 3), dtype=torch.bool)
        logits = torch.full((1, 3), 5.0, dtype=torch.float32)
        cfg = isolated_assignment_config()

        base = compute_set_matching_loss(
            predicted_features=pred,
            existence_logits=logits,
            candidate_mask=mask,
            offline_features=target,
            offline_mask=mask,
            config=cfg,
        )
        permuted = compute_set_matching_loss(
            predicted_features=pred,
            existence_logits=logits,
            candidate_mask=mask,
            offline_features=permuted_target,
            offline_mask=mask,
            config=cfg,
        )

        self.assertAlmostEqual(
            float(base.components["matched_core_loss"].detach()),
            float(permuted.components["matched_core_loss"].detach()),
            places=7,
        )
        self.assertAlmostEqual(float(base.total_loss.detach()), float(permuted.total_loss.detach()), places=7)
        self.assertEqual(base.assignments[0].matched_count, 3)
        self.assertEqual(permuted.assignments[0].matched_count, 3)

    def test_existence_targets_mark_matched_and_unmatched_slots(self):
        pred = make_feature_tensor([(10.0, 0.0, 0.0, 10.0), (50.0, 2.0, 2.0, 55.0)]).unsqueeze(0)
        target = make_feature_tensor([(10.0, 0.0, 0.0, 10.0)]).unsqueeze(0)
        candidate_mask = torch.ones((1, 2), dtype=torch.bool)
        target_mask = torch.ones((1, 1), dtype=torch.bool)
        logits = torch.zeros((1, 2), dtype=torch.float32)
        cfg = SetMatchingLossConfig(
            matched_core_weight=0.0,
            matched_aux_weight=0.0,
            existence_weight=1.0,
            count_weight=0.0,
            jet_summary_weight=0.0,
            correction_budget_weight=0.0,
            chamfer_weight=0.0,
        )

        out = compute_set_matching_loss(
            predicted_features=pred,
            existence_logits=logits,
            candidate_mask=candidate_mask,
            offline_features=target,
            offline_mask=target_mask,
            config=cfg,
        )

        self.assertEqual(out.assignments[0].pred_indices.tolist(), [0])
        self.assertEqual(out.assignments[0].target_indices.tolist(), [0])
        self.assertEqual(out.assignments[0].existence_targets.tolist(), [1.0, 0.0])
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.zeros((2,), dtype=torch.float32),
            torch.tensor([1.0, 0.0], dtype=torch.float32),
        )
        self.assertAlmostEqual(float(out.components["existence_loss"].detach()), float(expected), places=6)
        self.assertAlmostEqual(float(out.diagnostics["existence_target_mean"].detach()), 0.5, places=6)

    def test_phi_wraparound_keeps_boundary_particles_close(self):
        delta = wrapped_delta_phi(
            torch.tensor([math.pi - 0.01], dtype=torch.float32)
            - torch.tensor([-math.pi + 0.01], dtype=torch.float32)
        )
        self.assertLess(float(delta.abs().item()), 0.025)

        pred = make_feature_tensor([(10.0, 0.0, math.pi - 0.01, 10.0)])
        target_wrapped = make_feature_tensor([(10.0, 0.0, -math.pi + 0.01, 10.0)])
        target_far = make_feature_tensor([(10.0, 0.0, 0.0, 10.0)])
        cfg = isolated_assignment_config()
        wrapped_cost = pairwise_core_cost(pred, target_wrapped, cfg)
        far_cost = pairwise_core_cost(pred, target_far, cfg)

        self.assertLess(float(wrapped_cost.item()), 0.001)
        self.assertGreater(float(far_cost.item()), 1.0)

    def test_loss_report_contains_components_and_metrics(self):
        pred = make_feature_tensor([(10.0, 0.0, 0.1, 10.5)]).unsqueeze(0)
        target = pred.clone()
        mask = torch.ones((1, 1), dtype=torch.bool)
        logits = torch.ones((1, 1), dtype=torch.float32)
        out = compute_set_matching_loss(
            predicted_features=pred,
            existence_logits=logits,
            candidate_mask=mask,
            offline_features=target,
            offline_mask=mask,
            config=isolated_assignment_config(),
        )
        report = out.detached_float_dict()
        self.assertIn("loss_total", report)
        self.assertIn("loss_matched_core_loss", report)
        self.assertIn("metric_matched_count_mean", report)
        self.assertTrue(np.isfinite(list(report.values())).all())


if __name__ == "__main__":
    unittest.main()

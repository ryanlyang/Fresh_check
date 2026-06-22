import importlib.util
import math
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.detr_slots import (
        DETR_SLOT_HUNGARIAN_LOSS_STEP,
        DetrSlotHungarianLossConfig,
        compute_detr_slot_hungarian_loss,
        default_detr_slot_hungarian_loss_config,
        detr_slot_output_from_tensors,
        pairwise_detr_slot_assignment_cost,
        pairwise_detr_slot_core_cost,
    )
    from teacher_logit_reco.set_matching.detr_slots.features import raw_to_core_features, wrapped_phi_difference
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
    return DetrSlotHungarianLossConfig(
        assignment_aux_weight=0.0,
        matched_aux_weight=0.0,
        existence_weight=0.0,
        count_weight=0.0,
        jet_summary_weight=0.0,
        hlt_support_weight=0.0,
        duplicate_weight=0.0,
        allow_bruteforce_fallback=True,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DetrFreeSlotStep10HungarianLossTests(unittest.TestCase):
    def test_default_config_records_step_and_rejects_bad_values(self):
        cfg = default_detr_slot_hungarian_loss_config()
        self.assertEqual(cfg.to_dict()["step"], DETR_SLOT_HUNGARIAN_LOSS_STEP)
        self.assertFalse(cfg.allow_bruteforce_fallback)

        bad_configs = [
            {"core_weights": (1.0, 1.0, 1.0)},
            {"core_weights": (1.0, -1.0, 1.0, 1.0)},
            {"assignment_aux_weight": -0.1},
            {"existence_negative_weight": -0.1},
            {"duplicate_probability_threshold": 1.5},
            {"duplicate_delta_r_scale": 0.0},
            {"max_nearest_hlt_delta_r": 0.0},
            {"max_count_for_summary": 0.0},
            {"huber_beta": 0.0},
            {"brute_force_fallback_limit": 0},
            {"allow_bruteforce_fallback": "yes"},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    DetrSlotHungarianLossConfig(**kwargs)

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

        base = compute_detr_slot_hungarian_loss(
            predicted_features=pred,
            existence_logits=logits,
            candidate_mask=mask,
            offline_features=target,
            offline_mask=mask,
            config=cfg,
        )
        permuted = compute_detr_slot_hungarian_loss(
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

    def test_perfect_prediction_has_lower_loss_than_noisy_prediction(self):
        target = make_feature_tensor(
            [
                (10.0, 0.0, 0.1, 10.5),
                (20.0, 0.4, -0.2, 21.0),
            ]
        ).unsqueeze(0)
        noisy = target.clone()
        noisy[0, 0, 0] = 15.0
        noisy[0, 1, 1] = 1.4
        mask = torch.ones((1, 2), dtype=torch.bool)
        logits = torch.full((1, 2), 5.0, dtype=torch.float32)
        cfg = isolated_assignment_config()

        perfect = compute_detr_slot_hungarian_loss(
            predicted_features=target,
            existence_logits=logits,
            candidate_mask=mask,
            offline_features=target,
            offline_mask=mask,
            config=cfg,
        )
        degraded = compute_detr_slot_hungarian_loss(
            predicted_features=noisy,
            existence_logits=logits,
            candidate_mask=mask,
            offline_features=target,
            offline_mask=mask,
            config=cfg,
        )

        self.assertLess(float(perfect.total_loss.detach()), float(degraded.total_loss.detach()))

    def test_count_loss_moves_down_when_logits_predict_right_count(self):
        pred = make_feature_tensor(
            [
                (10.0, 0.0, 0.0, 10.0),
                (20.0, 0.2, 0.3, 21.0),
                (4.0, -0.4, 1.0, 5.0),
            ]
        ).unsqueeze(0)
        target = pred[:, :2, :].clone()
        candidate_mask = torch.ones((1, 3), dtype=torch.bool)
        target_mask = torch.ones((1, 2), dtype=torch.bool)
        too_many_logits = torch.full((1, 3), 5.0, dtype=torch.float32)
        right_count_logits = torch.tensor([[5.0, 5.0, -5.0]], dtype=torch.float32)
        cfg = DetrSlotHungarianLossConfig(
            assignment_aux_weight=0.0,
            matched_core_weight=0.0,
            matched_aux_weight=0.0,
            existence_weight=0.0,
            count_weight=1.0,
            jet_summary_weight=0.0,
            allow_bruteforce_fallback=True,
        )

        too_many = compute_detr_slot_hungarian_loss(
            predicted_features=pred,
            existence_logits=too_many_logits,
            candidate_mask=candidate_mask,
            offline_features=target,
            offline_mask=target_mask,
            config=cfg,
        )
        right_count = compute_detr_slot_hungarian_loss(
            predicted_features=pred,
            existence_logits=right_count_logits,
            candidate_mask=candidate_mask,
            offline_features=target,
            offline_mask=target_mask,
            config=cfg,
        )

        self.assertLess(
            float(right_count.components["count_loss"].detach()),
            float(too_many.components["count_loss"].detach()),
        )

    def test_aux_aware_assignment_resolves_same_core_different_pid(self):
        pred = make_feature_tensor([(10.0, 0.0, 0.0, 10.0), (10.0, 0.0, 0.0, 10.0)])
        target = make_feature_tensor([(10.0, 0.0, 0.0, 10.0), (10.0, 0.0, 0.0, 10.0)])
        pred[:, 4:] = 0.0
        target[:, 4:] = 0.0
        pred[0, 5] = 1.0
        pred[1, 7] = 1.0
        target[0, 7] = 1.0
        target[1, 5] = 1.0
        cfg = DetrSlotHungarianLossConfig(
            assignment_aux_weight=1.0,
            matched_core_weight=0.0,
            matched_aux_weight=0.0,
            existence_weight=0.0,
            count_weight=0.0,
            jet_summary_weight=0.0,
            allow_bruteforce_fallback=True,
        )
        cost = pairwise_detr_slot_assignment_cost(pred, target, cfg)
        self.assertLess(float(cost[0, 1].detach()), float(cost[0, 0].detach()))
        self.assertLess(float(cost[1, 0].detach()), float(cost[1, 1].detach()))

        out = compute_detr_slot_hungarian_loss(
            predicted_features=pred.unsqueeze(0),
            existence_logits=torch.full((1, 2), 5.0, dtype=torch.float32),
            candidate_mask=torch.ones((1, 2), dtype=torch.bool),
            offline_features=target.unsqueeze(0),
            offline_mask=torch.ones((1, 2), dtype=torch.bool),
            config=cfg,
        )
        self.assertEqual(out.assignments[0].pred_indices.tolist(), [0, 1])
        self.assertEqual(out.assignments[0].target_indices.tolist(), [1, 0])

    def test_unit_interval_aux_uses_logits_for_bce_supervision(self):
        pred = make_feature_tensor([(10.0, 0.0, 0.0, 10.0)]).unsqueeze(0)
        target = make_feature_tensor([(10.0, 0.0, 0.0, 10.0)]).unsqueeze(0)
        pred[:, :, 4:] = 0.0
        target[:, :, 4:] = 0.0
        target[0, 0, 5] = 1.0
        cfg = DetrSlotHungarianLossConfig(
            assignment_aux_weight=0.0,
            matched_core_weight=0.0,
            matched_aux_weight=1.0,
            existence_weight=0.0,
            count_weight=0.0,
            jet_summary_weight=0.0,
            allow_bruteforce_fallback=True,
        )
        aux_indices = cfg.feature_config.aux_feature_indices()
        pos5 = aux_indices.index(5)
        good_logits = torch.full((1, 1, cfg.feature_config.aux_dim), -5.0, dtype=torch.float32)
        bad_logits = torch.full_like(good_logits, -5.0)
        good_logits[0, 0, pos5] = 5.0
        bad_logits[0, 0, pos5] = -5.0

        good = compute_detr_slot_hungarian_loss(
            predicted_features=pred,
            predicted_aux_logits=good_logits,
            existence_logits=torch.ones((1, 1), dtype=torch.float32),
            candidate_mask=torch.ones((1, 1), dtype=torch.bool),
            offline_features=target,
            offline_mask=torch.ones((1, 1), dtype=torch.bool),
            config=cfg,
        )
        bad = compute_detr_slot_hungarian_loss(
            predicted_features=pred,
            predicted_aux_logits=bad_logits,
            existence_logits=torch.ones((1, 1), dtype=torch.float32),
            candidate_mask=torch.ones((1, 1), dtype=torch.bool),
            offline_features=target,
            offline_mask=torch.ones((1, 1), dtype=torch.bool),
            config=cfg,
        )
        self.assertLess(
            float(good.components["matched_aux_loss"].detach()),
            float(bad.components["matched_aux_loss"].detach()),
        )

    def test_bounded_continuous_aux_is_regressed_not_bce_classified(self):
        pred = make_feature_tensor([(10.0, 0.0, 0.0, 10.0)]).unsqueeze(0)
        target = make_feature_tensor([(10.0, 0.0, 0.0, 10.0)]).unsqueeze(0)
        pred[:, :, 4:] = 0.0
        target[:, :, 4:] = 0.0
        pred[0, 0, 11] = 0.7
        target[0, 0, 11] = 0.7
        cfg = DetrSlotHungarianLossConfig(
            assignment_aux_weight=0.0,
            matched_core_weight=0.0,
            matched_aux_weight=1.0,
            existence_weight=0.0,
            count_weight=0.0,
            jet_summary_weight=0.0,
            allow_bruteforce_fallback=True,
        )
        aux_indices = cfg.feature_config.aux_feature_indices()
        pos11 = aux_indices.index(11)
        neutral_logits = torch.zeros((1, 1, cfg.feature_config.aux_dim), dtype=torch.float32)
        changed_continuous_logit = neutral_logits.clone()
        changed_continuous_logit[0, 0, pos11] = 10.0

        neutral = compute_detr_slot_hungarian_loss(
            predicted_features=pred,
            predicted_aux_logits=neutral_logits,
            existence_logits=torch.ones((1, 1), dtype=torch.float32),
            candidate_mask=torch.ones((1, 1), dtype=torch.bool),
            offline_features=target,
            offline_mask=torch.ones((1, 1), dtype=torch.bool),
            config=cfg,
        )
        changed = compute_detr_slot_hungarian_loss(
            predicted_features=pred,
            predicted_aux_logits=changed_continuous_logit,
            existence_logits=torch.ones((1, 1), dtype=torch.float32),
            candidate_mask=torch.ones((1, 1), dtype=torch.bool),
            offline_features=target,
            offline_mask=torch.ones((1, 1), dtype=torch.bool),
            config=cfg,
        )
        self.assertAlmostEqual(
            float(neutral.components["matched_aux_loss"].detach()),
            float(changed.components["matched_aux_loss"].detach()),
            places=7,
        )

    def test_existence_targets_mark_matched_and_unmatched_slots(self):
        pred = make_feature_tensor([(10.0, 0.0, 0.0, 10.0), (50.0, 2.0, 2.0, 55.0)]).unsqueeze(0)
        target = make_feature_tensor([(10.0, 0.0, 0.0, 10.0)]).unsqueeze(0)
        candidate_mask = torch.ones((1, 2), dtype=torch.bool)
        target_mask = torch.ones((1, 1), dtype=torch.bool)
        logits = torch.zeros((1, 2), dtype=torch.float32)
        cfg = DetrSlotHungarianLossConfig(
            matched_core_weight=0.0,
            matched_aux_weight=0.0,
            existence_weight=1.0,
            existence_positive_weight=1.0,
            existence_negative_weight=0.2,
            count_weight=0.0,
            jet_summary_weight=0.0,
            allow_bruteforce_fallback=True,
        )

        out = compute_detr_slot_hungarian_loss(
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
        self.assertGreater(float(out.components["existence_loss"].detach()), 0.0)
        self.assertAlmostEqual(float(out.diagnostics["existence_target_mean"].detach()), 0.5, places=6)

    def test_phi_wraparound_keeps_boundary_particles_close(self):
        delta = wrapped_phi_difference(
            torch.tensor([math.pi - 0.01], dtype=torch.float32),
            torch.tensor([-math.pi + 0.01], dtype=torch.float32),
        )
        self.assertLess(float(delta.abs().item()), 0.025)

        pred = make_feature_tensor([(10.0, 0.0, math.pi - 0.01, 10.0)])
        target_wrapped = make_feature_tensor([(10.0, 0.0, -math.pi + 0.01, 10.0)])
        target_far = make_feature_tensor([(10.0, 0.0, 0.0, 10.0)])
        cfg = isolated_assignment_config()
        wrapped_cost = pairwise_detr_slot_core_cost(raw_to_core_features(pred), raw_to_core_features(target_wrapped), cfg)
        far_cost = pairwise_detr_slot_core_cost(raw_to_core_features(pred), raw_to_core_features(target_far), cfg)

        self.assertLess(float(wrapped_cost.item()), 0.001)
        self.assertGreater(float(far_cost.item()), 1.0)

    def test_loss_report_contains_components_and_metrics(self):
        pred = make_feature_tensor([(10.0, 0.0, 0.1, 10.5)]).unsqueeze(0)
        target = pred.clone()
        mask = torch.ones((1, 1), dtype=torch.bool)
        logits = torch.ones((1, 1), dtype=torch.float32)
        out = compute_detr_slot_hungarian_loss(
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
        self.assertIn("metric_matched_delta_r_mean", report)
        self.assertIn("metric_count_mae", report)
        self.assertIn("metric_existence_precision", report)
        self.assertIn("metric_existence_recall", report)
        self.assertIn("metric_matched_delta_r_p90", report)
        self.assertIn("metric_jet_sum_pt_relative_error_mean", report)
        self.assertTrue(np.isfinite(list(report.values())).all())

    def test_output_to_loss_kwargs_feeds_hungarian_loss(self):
        tokens = make_feature_tensor(
            [
                (10.0, 0.0, 0.0, 10.5),
                (15.0, 0.3, 0.2, 16.0),
                (2.0, 2.0, 2.5, 3.0),
            ]
        ).unsqueeze(0)
        logits = torch.tensor([[5.0, 5.0, -5.0]], dtype=torch.float32)
        slot_mask = torch.ones((1, 3), dtype=torch.bool)
        aux_outputs = torch.zeros((1, 3, 10), dtype=torch.float32)
        output = detr_slot_output_from_tensors(tokens, logits, slot_mask, loss_features=tokens, aux_outputs=aux_outputs)
        offline = tokens[:, :2, :].clone()
        offline_mask = torch.ones((1, 2), dtype=torch.bool)
        loss_kwargs = output.to_loss_kwargs(
            offline_features=offline,
            offline_mask=offline_mask,
            include_aux_logits=True,
        )
        self.assertIn("predicted_aux_logits", loss_kwargs)

        out = compute_detr_slot_hungarian_loss(
            **loss_kwargs,
            config=isolated_assignment_config(),
        )

        self.assertEqual(out.assignments[0].matched_count, 2)
        self.assertEqual(out.assignments[0].existence_targets.tolist(), [1.0, 1.0, 0.0])

    def test_empty_targets_train_all_candidates_as_no_object(self):
        pred = make_feature_tensor([(10.0, 0.0, 0.0, 10.0), (20.0, 0.2, 0.3, 21.0)]).unsqueeze(0)
        target = make_feature_tensor([(1.0, 0.0, 0.0, 1.0)]).unsqueeze(0)
        candidate_mask = torch.ones((1, 2), dtype=torch.bool)
        target_mask = torch.zeros((1, 1), dtype=torch.bool)
        logits = torch.zeros((1, 2), dtype=torch.float32)
        cfg = DetrSlotHungarianLossConfig(
            matched_core_weight=0.0,
            matched_aux_weight=0.0,
            existence_weight=1.0,
            count_weight=0.0,
            jet_summary_weight=0.0,
            allow_bruteforce_fallback=True,
        )

        out = compute_detr_slot_hungarian_loss(
            predicted_features=pred,
            existence_logits=logits,
            candidate_mask=candidate_mask,
            offline_features=target,
            offline_mask=target_mask,
            config=cfg,
        )

        self.assertEqual(out.assignments[0].matched_count, 0)
        self.assertEqual(out.assignments[0].method, "none")
        self.assertEqual(out.assignments[0].existence_targets.tolist(), [0.0, 0.0])
        self.assertGreater(float(out.components["existence_loss"].detach()), 0.0)

    def test_optional_support_and_duplicate_terms_are_finite(self):
        pred = make_feature_tensor(
            [
                (10.0, 0.0, 0.0, 10.0),
                (10.1, 0.01, 0.01, 10.2),
            ]
        ).unsqueeze(0)
        target = pred.clone()
        hlt = make_feature_tensor([(10.0, 0.0, 0.0, 10.0)]).unsqueeze(0)
        mask = torch.ones((1, 2), dtype=torch.bool)
        hlt_mask = torch.ones((1, 1), dtype=torch.bool)
        logits = torch.full((1, 2), 5.0, dtype=torch.float32)
        cfg = DetrSlotHungarianLossConfig(
            hlt_support_weight=0.01,
            duplicate_weight=0.01,
            duplicate_probability_threshold=0.1,
            brute_force_fallback_limit=4,
            allow_bruteforce_fallback=True,
        )

        out = compute_detr_slot_hungarian_loss(
            predicted_features=pred,
            existence_logits=logits,
            candidate_mask=mask,
            offline_features=target,
            offline_mask=mask,
            hlt_features=hlt,
            hlt_mask=hlt_mask,
            config=cfg,
        )

        self.assertTrue(torch.isfinite(out.total_loss))
        self.assertGreaterEqual(float(out.components["support_loss"].detach()), 0.0)
        self.assertGreaterEqual(float(out.components["duplicate_loss"].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()

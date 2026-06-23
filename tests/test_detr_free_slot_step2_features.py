import importlib.util
import math
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.detr_slots import (
        DetrSlotFeatureConfig,
        decode_slot_outputs_to_loss_features,
        decode_slot_outputs_to_raw_tokens,
        default_detr_slot_feature_config,
        feature_indices_report,
        raw_to_aux_features,
        raw_to_core_features,
        safe_log_pt_energy,
        wrapped_phi_difference,
    )
else:  # pragma: no cover - environment dependent
    torch = None


SIGNED_AUX_INDICES = (4, 10, 12)
UNIT_INTERVAL_AUX_INDICES = (5, 6, 7, 8, 9, 11, 13)
BINARY_AUX_INDICES = (5, 6, 7, 8, 9)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DetrFreeSlotStep2FeatureTests(unittest.TestCase):
    def test_raw_to_core_features_uses_log_kinematics_and_wrapped_phi(self):
        tokens = torch.zeros((2, 3, RAW_TOKEN_DIM), dtype=torch.float32)
        tokens[..., 0] = 4.0
        tokens[..., 1] = 0.5
        tokens[..., 2] = 3.0 * math.pi
        tokens[..., 3] = 8.0

        core = raw_to_core_features(tokens)

        self.assertEqual(tuple(core.shape), (2, 3, 4))
        self.assertTrue(torch.allclose(core[..., 0], torch.log(torch.full((2, 3), 4.0))))
        self.assertTrue(torch.allclose(core[..., 1], torch.full((2, 3), 0.5)))
        self.assertTrue(torch.allclose(core[..., 2].abs(), torch.full((2, 3), math.pi), atol=1.0e-6))
        self.assertTrue(torch.allclose(core[..., 3], torch.log(torch.full((2, 3), 8.0))))

    def test_wrapped_phi_difference_crosses_pi_boundary(self):
        left = torch.tensor([math.pi - 0.01], dtype=torch.float32)
        right = torch.tensor([-math.pi + 0.01], dtype=torch.float32)
        delta = wrapped_phi_difference(left, right)

        self.assertLess(float(delta.abs().item()), 0.025)

    def test_raw_to_aux_features_returns_non_core_columns(self):
        tokens = torch.arange(2 * 4 * RAW_TOKEN_DIM, dtype=torch.float32).reshape(2, 4, RAW_TOKEN_DIM)
        aux = raw_to_aux_features(tokens)

        self.assertEqual(tuple(aux.shape), (2, 4, RAW_TOKEN_DIM - 4))
        self.assertTrue(torch.equal(aux[..., 0], tokens[..., 4]))
        self.assertTrue(torch.equal(aux[..., -1], tokens[..., RAW_TOKEN_DIM - 1]))

    def test_custom_aux_indices_support_nonstandard_feature_groups(self):
        config = DetrSlotFeatureConfig(feature_dim=8, aux_indices=(5, 7))
        tokens = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
        aux = raw_to_aux_features(tokens, config)

        self.assertEqual(tuple(aux.shape), (2, 3, 2))
        self.assertTrue(torch.equal(aux[..., 0], tokens[..., 5]))
        self.assertTrue(torch.equal(aux[..., 1], tokens[..., 7]))

    def test_feature_config_from_mapping_ignores_report_only_keys(self):
        config = DetrSlotFeatureConfig(feature_dim=RAW_TOKEN_DIM, aux_indices=(4, 5, 6))
        payload = config.to_dict()

        restored = DetrSlotFeatureConfig.from_mapping(payload)

        self.assertEqual(restored.feature_dim, config.feature_dim)
        self.assertEqual(restored.aux_indices, config.aux_indices)
        self.assertIn("core_feature_names", payload)
        self.assertIn("raw_kinematic_feature_names", payload)

    def test_decode_slot_outputs_to_raw_tokens_enforces_physical_kinematics(self):
        config = default_detr_slot_feature_config(feature_dim=RAW_TOKEN_DIM)
        core = torch.tensor(
            [[[math.log(2.0), 1.0, 4.0 * math.pi + 0.25, math.log(0.5)]]],
            dtype=torch.float32,
        )
        aux = torch.ones((1, 1, config.aux_dim), dtype=torch.float32)

        raw = decode_slot_outputs_to_raw_tokens(core, aux, config=config)

        self.assertEqual(tuple(raw.shape), (1, 1, RAW_TOKEN_DIM))
        self.assertGreater(float(raw[0, 0, 0]), 0.0)
        self.assertGreaterEqual(float(raw[0, 0, 3]), float(raw[0, 0, 0] * torch.cosh(raw[0, 0, 1])))
        self.assertGreaterEqual(float(raw[0, 0, 2]), -math.pi)
        self.assertLessEqual(float(raw[0, 0, 2]), math.pi)
        for index in SIGNED_AUX_INDICES:
            self.assertGreaterEqual(float(raw[0, 0, index]), -1.0)
            self.assertLessEqual(float(raw[0, 0, index]), 1.0)
        for index in UNIT_INTERVAL_AUX_INDICES:
            self.assertGreaterEqual(float(raw[0, 0, index]), 0.0)
            self.assertLessEqual(float(raw[0, 0, index]), 1.0)

    def test_decode_accepts_full_dim_aux_outputs(self):
        config = DetrSlotFeatureConfig(feature_dim=6)
        core = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
        full_aux = torch.zeros((1, 1, 6), dtype=torch.float32)
        full_aux[..., 4] = 7.0
        full_aux[..., 5] = 8.0

        raw = decode_slot_outputs_to_raw_tokens(core, full_aux, config=config)

        self.assertLess(float(raw[0, 0, 4]), 1.0)
        self.assertGreater(float(raw[0, 0, 4]), 0.99)
        self.assertLess(float(raw[0, 0, 5]), 1.0)
        self.assertGreater(float(raw[0, 0, 5]), 0.99)

    def test_loss_feature_decode_uses_smooth_bounded_kinematics(self):
        config = default_detr_slot_feature_config(feature_dim=RAW_TOKEN_DIM)
        core = torch.tensor([[[100.0, 100.0, 4.0 * math.pi + 0.1, -100.0]]], dtype=torch.float32, requires_grad=True)
        aux = torch.zeros((1, 1, config.aux_dim), dtype=torch.float32, requires_grad=True)

        loss_features = decode_slot_outputs_to_loss_features(core, aux, config=config)
        loss = loss_features[..., [0, 1, 2, 3]].sum()
        loss.backward()

        self.assertEqual(tuple(loss_features.shape), (1, 1, RAW_TOKEN_DIM))
        self.assertTrue(torch.isfinite(loss_features).all())
        self.assertGreater(float(loss_features[0, 0, 0]), 0.0)
        self.assertGreater(float(loss_features[0, 0, 3]), 0.0)
        self.assertGreaterEqual(
            float(loss_features[0, 0, 3]),
            float(loss_features[0, 0, 0] * torch.cosh(loss_features[0, 0, 1])),
        )
        self.assertGreaterEqual(float(loss_features[0, 0, 1]), -5.0)
        self.assertLessEqual(float(loss_features[0, 0, 1]), 5.0)
        self.assertIsNotNone(core.grad)
        self.assertTrue(torch.isfinite(core.grad).all())

    def test_half_precision_decode_promotes_exp_math_to_float32(self):
        config = default_detr_slot_feature_config(feature_dim=RAW_TOKEN_DIM)
        core = torch.tensor([[[20.0, 5.0, 0.0, 20.0]]], dtype=torch.float16, requires_grad=True)
        aux = torch.zeros((1, 1, config.aux_dim), dtype=torch.float16, requires_grad=True)

        raw = decode_slot_outputs_to_raw_tokens(core, aux, config=config)
        loss_features = decode_slot_outputs_to_loss_features(core, aux, config=config)
        loss = raw[..., [0, 3]].sum() + loss_features[..., [0, 3]].sum()
        loss.backward()

        self.assertEqual(raw.dtype, torch.float32)
        self.assertEqual(loss_features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(raw).all())
        self.assertTrue(torch.isfinite(loss_features).all())
        self.assertIsNotNone(core.grad)
        self.assertTrue(torch.isfinite(core.grad).all())

    def test_safe_log_pt_energy_sanitizes_bad_inputs(self):
        pt = torch.tensor([4.0, -2.0, float("nan")], dtype=torch.float32)
        energy = torch.tensor([8.0, 0.0, float("inf")], dtype=torch.float32)
        log_pt, log_energy = safe_log_pt_energy(pt, energy)

        self.assertEqual(tuple(log_pt.shape), (3,))
        self.assertTrue(torch.isfinite(log_pt).all())
        self.assertTrue(torch.isfinite(log_energy).all())

    def test_feature_report_is_binary_task_agnostic(self):
        report = feature_indices_report(DetrSlotFeatureConfig(feature_dim=RAW_TOKEN_DIM))

        self.assertEqual(report["core"]["pt"], 0)
        self.assertEqual(report["core"]["eta"], 1)
        self.assertEqual(report["core"]["phi"], 2)
        self.assertEqual(report["core"]["energy"], 3)
        self.assertEqual(len(report["aux_indices"]), RAW_TOKEN_DIM - 4)
        self.assertEqual(tuple(report["signed_aux_indices"]), SIGNED_AUX_INDICES)
        self.assertEqual(tuple(report["unit_interval_aux_indices"]), UNIT_INTERVAL_AUX_INDICES)
        self.assertEqual(tuple(report["binary_aux_indices"]), BINARY_AUX_INDICES)


if __name__ == "__main__":
    unittest.main()

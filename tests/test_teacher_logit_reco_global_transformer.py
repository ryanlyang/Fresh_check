import importlib.util
import math
import unittest

import numpy as np

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.global_transformer import (
    GlobalTransformerReconstructor,
    GlobalTransformerReconstructorConfig,
    physical_energy_floor,
    sanitize_reconstructed_view_tensors,
)
from teacher_logit_reco.views import SoftReconstructedView

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


class GlobalTransformerConfigTests(unittest.TestCase):
    def test_config_roundtrip_and_validation(self):
        cfg = GlobalTransformerReconstructorConfig(hidden_dim=64, num_heads=4, num_layers=1, num_extra_candidates=3)
        self.assertEqual(cfg.to_dict()["hidden_dim"], 64)
        self.assertEqual(GlobalTransformerReconstructorConfig.from_mapping(cfg).hidden_dim, 64)
        self.assertEqual(GlobalTransformerReconstructorConfig.from_mapping(cfg.to_dict()).num_extra_candidates, 3)
        with self.assertRaises(ValueError):
            GlobalTransformerReconstructorConfig(hidden_dim=65, num_heads=4)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class GlobalTransformerForwardTests(unittest.TestCase):
    def make_batch(self):
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((2, 5), dtype=torch.bool)
        mask[0, :3] = True
        mask[1, :2] = True
        for b in range(2):
            for i in range(5):
                if not bool(mask[b, i]):
                    continue
                pt = 10.0 + b + i
                eta = 0.1 * i
                phi = -0.2 + 0.1 * i
                tokens[b, i, 0] = pt
                tokens[b, i, 1] = eta
                tokens[b, i, 2] = phi
                tokens[b, i, 3] = pt * math.cosh(eta) + 0.5
                tokens[b, i, 4] = 1.0
                tokens[b, i, 5 + (i % 5)] = 1.0
                tokens[b, i, 10:14] = torch.tensor([0.1, 0.01, -0.2, 0.02])
        labels = torch.tensor([1, 8], dtype=torch.long)
        return tokens, mask, labels

    def make_model(self):
        torch.manual_seed(123)
        return GlobalTransformerReconstructor(
            GlobalTransformerReconstructorConfig(
                hidden_dim=32,
                num_heads=4,
                num_layers=1,
                num_extra_candidates=3,
                dropout=0.0,
            )
        )

    def test_forward_returns_soft_view_with_correct_shapes_and_masks(self):
        model = self.make_model()
        tokens, mask, labels = self.make_batch()
        view = model(tokens, mask, labels=labels, split="model_train")
        self.assertIsInstance(view, SoftReconstructedView)
        self.assertEqual(tuple(view.tokens.shape), (2, 8, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.weights.shape), (2, 8))
        self.assertEqual(tuple(view.mask[:, :5].shape), tuple(mask.shape))
        self.assertTrue(torch.equal(view.mask[:, :5], mask))
        self.assertTrue(bool(view.mask[:, 5:].all()))
        self.assertTrue(bool(torch.isfinite(view.tokens).all()))
        self.assertTrue(bool(torch.isfinite(view.weights).all()))
        self.assertTrue(bool((view.weights >= 0.0).all()))
        self.assertTrue(bool((view.weights <= 1.0).all()))
        self.assertTrue(bool((view.weights[:, :5][~mask] == 0.0).all()))
        self.assertTrue(bool((view.tokens[:, :5][~mask] == 0.0).all()))

    def test_parent_corrections_are_bounded_and_physical(self):
        model = self.make_model()
        tokens, mask, labels = self.make_batch()
        view = model(tokens, mask, labels=labels)
        delta = view.aux["parent_delta"]
        cfg = model.config
        self.assertLessEqual(float(delta[:, :, 0].abs().max()), cfg.max_delta_logpt + 1.0e-6)
        self.assertLessEqual(float(delta[:, :, 1].abs().max()), cfg.max_delta_eta + 1.0e-6)
        self.assertLessEqual(float(delta[:, :, 2].abs().max()), cfg.max_delta_phi + 1.0e-6)
        self.assertLessEqual(float(delta[:, :, 3].abs().max()), cfg.max_delta_loge + 1.0e-6)

        valid = view.mask
        pt = view.tokens[:, :, 0][valid]
        eta = view.tokens[:, :, 1][valid]
        phi = view.tokens[:, :, 2][valid]
        energy = view.tokens[:, :, 3][valid]
        self.assertTrue(bool((pt >= cfg.min_pt).all()))
        self.assertTrue(bool((eta.abs() <= cfg.eta_limit + 1.0e-6).all()))
        self.assertTrue(bool((phi >= -math.pi - 1.0e-6).all()))
        self.assertTrue(bool((phi < math.pi + 1.0e-6).all()))
        floor = physical_energy_floor(pt, eta, eps=cfg.energy_eps)
        self.assertTrue(bool((energy + 1.0e-5 >= floor).all()))

    def test_empty_input_jets_get_safe_fallback_parent(self):
        model = self.make_model()
        tokens = torch.zeros((1, 4, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((1, 4), dtype=torch.bool)
        view = model(tokens, mask)
        self.assertTrue(bool(view.mask[0, 0]))
        self.assertEqual(view.aux["diagnostics"]["empty_input_jet_count"], 1)
        self.assertTrue(bool(torch.isfinite(view.tokens).all()))

    def test_reconstructed_view_sanitizer_masks_nonfinite_candidates(self):
        cfg = GlobalTransformerReconstructorConfig()
        tokens = torch.zeros((1, 3, RAW_TOKEN_DIM), dtype=torch.float32)
        tokens[:, :, 0] = 1.0
        tokens[:, :, 3] = 2.0
        tokens[0, 1, 0] = float("nan")
        mask = torch.ones((1, 3), dtype=torch.bool)
        weights = torch.tensor([[1.0, 0.5, float("inf")]], dtype=torch.float32)

        clean_tokens, clean_mask, clean_weights, diagnostics = sanitize_reconstructed_view_tensors(
            tokens,
            mask,
            weights,
            config=cfg,
        )

        self.assertTrue(bool(torch.isfinite(clean_tokens).all()))
        self.assertTrue(bool(torch.isfinite(clean_weights).all()))
        self.assertTrue(bool(clean_mask[0, 0]))
        self.assertFalse(bool(clean_mask[0, 1]))
        self.assertFalse(bool(clean_mask[0, 2]))
        self.assertEqual(diagnostics["nonfinite_reco_token_count"], 1)
        self.assertEqual(diagnostics["nonfinite_reco_weight_count"], 1)


if __name__ == "__main__":
    unittest.main()

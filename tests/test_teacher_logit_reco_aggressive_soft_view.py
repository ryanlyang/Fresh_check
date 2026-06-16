import importlib.util
import math
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.aggressive_soft_view import (
    AGGRESSION_LEVEL,
    AggressiveSoftViewConfig,
    AggressiveSoftViewHead,
)
from teacher_logit_reco.views import SoftReconstructedView

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


class AggressiveSoftViewConfigTests(unittest.TestCase):
    def test_default_config_and_roundtrip(self):
        cfg = AggressiveSoftViewConfig(embedding_dim=32, num_extra_candidates=4)
        self.assertEqual(cfg.aggression_level, AGGRESSION_LEVEL)
        self.assertEqual(cfg.num_extra_candidates, 4)
        self.assertEqual(AggressiveSoftViewConfig.from_mapping(cfg.to_dict()).embedding_dim, 32)
        with self.assertRaises(ValueError):
            AggressiveSoftViewConfig(embedding_dim=0)
        with self.assertRaises(ValueError):
            AggressiveSoftViewConfig(num_extra_candidates=-1)
        with self.assertRaises(ValueError):
            AggressiveSoftViewConfig(extra_usage_weight_threshold=1.5)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class AggressiveSoftViewHeadTests(unittest.TestCase):
    def make_batch(self):
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((2, 5), dtype=torch.bool)
        mask[0, :3] = True
        mask[1, :2] = True
        for b in range(2):
            for i in range(5):
                if not bool(mask[b, i]):
                    continue
                pt = 20.0 + b + i
                eta = 0.05 * i
                phi = -0.15 + 0.07 * i
                tokens[b, i, 0] = pt
                tokens[b, i, 1] = eta
                tokens[b, i, 2] = phi
                tokens[b, i, 3] = pt * math.cosh(eta) + 0.25
                tokens[b, i, 4] = 1.0
                tokens[b, i, 5 + (i % 5)] = 1.0
                tokens[b, i, 10:14] = torch.tensor([0.1, 0.01, -0.2, 0.02])
        labels = torch.tensor([1, 8], dtype=torch.long)
        return tokens, mask, labels

    def make_head(self):
        torch.manual_seed(123)
        cfg = AggressiveSoftViewConfig(
            embedding_dim=16,
            num_extra_candidates=4,
            dropout=0.0,
        )
        return AggressiveSoftViewHead(cfg)

    def test_forward_returns_soft_view_with_expected_contract(self):
        head = self.make_head()
        tokens, mask, labels = self.make_batch()
        embeddings = torch.randn(tokens.shape[0], tokens.shape[1], head.config.embedding_dim)
        context = torch.randn(tokens.shape[0], head.config.embedding_dim)
        view = head(tokens, mask, embeddings, context, labels=labels, split="model_train")

        self.assertIsInstance(view, SoftReconstructedView)
        self.assertEqual(tuple(view.tokens.shape), (2, 9, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.mask.shape), (2, 9))
        self.assertEqual(tuple(view.weights.shape), (2, 9))
        self.assertEqual(view.metadata["aggression_level"], AGGRESSION_LEVEL)
        self.assertTrue(view.metadata["parent_reweighting_enabled"])
        self.assertTrue(view.metadata["global_calibration_enabled"])
        self.assertEqual(view.metadata["n_extra_candidates"], 4)
        self.assertTrue(bool(torch.equal(view.mask[:, :5], mask)))
        self.assertTrue(bool(view.mask[:, 5:].all()))
        self.assertTrue(bool(torch.isfinite(view.tokens).all()))
        self.assertTrue(bool(torch.isfinite(view.weights).all()))
        self.assertTrue(bool((view.weights >= 0.0).all()))
        self.assertTrue(bool((view.weights <= 1.0).all()))
        self.assertIn("global_correction", view.aux)
        self.assertIn("global_calibration", view.aux)
        self.assertIn("global_raw", view.aux)
        self.assertIn("parent_raw", view.aux)
        self.assertIn("parent_weight_logits", view.aux)
        self.assertIn("extra_raw", view.aux)
        self.assertIn("extra_weight_logits", view.aux)
        self.assertIn("extra_weight_sum", view.aux)
        self.assertIn("extra_pt_fraction", view.aux)
        self.assertIn("extra_slot_active_mask", view.aux)
        self.assertIn("extra_slot_usage", view.aux)
        self.assertIn("extra_pt_fraction_mean", view.metadata["diagnostics"])
        self.assertIn("extra_slot_usage_mean", view.metadata["diagnostics"])
        self.assertIn("extra_slot_usage_histogram", view.metadata["diagnostics"])
        self.assertIn("parent_weight_min", view.metadata["diagnostics"])
        self.assertIn("parent_delta_logpt_abs_mean", view.metadata["diagnostics"])

    def test_global_corrections_are_bounded(self):
        head = self.make_head()
        tokens, mask, labels = self.make_batch()
        embeddings = torch.randn(tokens.shape[0], tokens.shape[1], head.config.embedding_dim)
        context = torch.randn(tokens.shape[0], head.config.embedding_dim)
        view = head(tokens, mask, embeddings, context, labels=labels)
        global_correction = view.aux["global_correction"]
        self.assertLessEqual(
            float(global_correction["logpt_scale"].abs().max()),
            head.config.max_global_logpt_scale + 1.0e-6,
        )
        self.assertLessEqual(
            float(global_correction["loge_scale"].abs().max()),
            head.config.max_global_loge_scale + 1.0e-6,
        )
        self.assertLessEqual(
            float(global_correction["eta_shift"].abs().max()),
            head.config.max_global_eta_shift + 1.0e-6,
        )
        self.assertLessEqual(
            float(global_correction["phi_shift"].abs().max()),
            head.config.max_global_phi_shift + 1.0e-6,
        )

    def test_parent_edits_and_weights_are_bounded(self):
        head = self.make_head()
        tokens, mask, labels = self.make_batch()
        embeddings = torch.randn(tokens.shape[0], tokens.shape[1], head.config.embedding_dim)
        context = torch.randn(tokens.shape[0], head.config.embedding_dim)
        view = head(tokens, mask, embeddings, context, labels=labels)
        parent_delta = view.aux["parent_delta"]
        parent_weights = view.aux["parent_weights"]
        diagnostics = view.metadata["diagnostics"]

        self.assertLessEqual(float(parent_delta[:, :, 0].abs().max()), head.config.max_delta_logpt + 1.0e-6)
        self.assertLessEqual(float(parent_delta[:, :, 1].abs().max()), head.config.max_delta_eta + 1.0e-6)
        self.assertLessEqual(float(parent_delta[:, :, 2].abs().max()), head.config.max_delta_phi + 1.0e-6)
        self.assertLessEqual(float(parent_delta[:, :, 3].abs().max()), head.config.max_delta_loge + 1.0e-6)
        self.assertTrue(bool((parent_weights >= 0.0).all()))
        self.assertTrue(bool((parent_weights <= 1.0).all()))
        self.assertTrue(bool((parent_weights[~mask] == 0.0).all()))
        self.assertGreaterEqual(diagnostics["parent_weight_min"], 0.0)
        self.assertLessEqual(diagnostics["parent_weight_max"], 1.0)
        self.assertLessEqual(diagnostics["parent_delta_eta_abs_mean"], head.config.max_delta_eta + 1.0e-6)

    def test_extra_candidates_have_budget_diagnostics(self):
        head = self.make_head()
        tokens, mask, labels = self.make_batch()
        embeddings = torch.randn(tokens.shape[0], tokens.shape[1], head.config.embedding_dim)
        context = torch.randn(tokens.shape[0], head.config.embedding_dim)
        view = head(tokens, mask, embeddings, context, labels=labels)
        diagnostics = view.metadata["diagnostics"]

        self.assertEqual(tuple(view.tokens.shape), (2, 5 + head.config.num_extra_candidates, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.aux["extra_tokens"].shape), (2, head.config.num_extra_candidates, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.aux["extra_raw"].shape), (2, head.config.num_extra_candidates, RAW_TOKEN_DIM + 1))
        self.assertEqual(tuple(view.aux["extra_weight_logits"].shape), (2, head.config.num_extra_candidates))
        self.assertEqual(tuple(view.aux["extra_pt_fraction"].shape), (2,))
        self.assertEqual(tuple(view.aux["extra_weight_sum"].shape), (2,))
        self.assertEqual(tuple(view.aux["extra_slot_active_mask"].shape), (2, head.config.num_extra_candidates))
        self.assertEqual(tuple(view.aux["extra_slot_usage"].shape), (2,))
        self.assertTrue(bool(torch.isfinite(view.aux["extra_tokens"]).all()))
        self.assertTrue(bool(torch.isfinite(view.aux["extra_weights"]).all()))
        self.assertTrue(bool((view.aux["extra_weights"] >= 0.0).all()))
        self.assertTrue(bool((view.aux["extra_weights"] <= 1.0).all()))
        self.assertTrue(bool(view.aux["extra_mask"].all()))
        self.assertIn("extra_weight_sum_mean", diagnostics)
        self.assertIn("extra_weight_sum_min", diagnostics)
        self.assertIn("extra_weight_sum_max", diagnostics)
        self.assertIn("extra_pt_fraction_mean", diagnostics)
        self.assertIn("extra_pt_fraction_max", diagnostics)
        self.assertIn("extra_slot_usage_mean", diagnostics)
        self.assertIn("extra_slot_usage_min", diagnostics)
        self.assertIn("extra_slot_usage_max", diagnostics)
        self.assertIn("extra_slot_usage_histogram", diagnostics)
        self.assertEqual(len(diagnostics["extra_slot_usage_histogram"]), head.config.num_extra_candidates + 1)
        self.assertEqual(sum(diagnostics["extra_slot_usage_histogram"]), tokens.shape[0])

    def test_shape_guards_catch_wrong_encoder_outputs(self):
        head = self.make_head()
        tokens, mask, labels = self.make_batch()
        embeddings = torch.randn(tokens.shape[0], tokens.shape[1], head.config.embedding_dim + 1)
        context = torch.randn(tokens.shape[0], head.config.embedding_dim)
        with self.assertRaises(ValueError):
            head(tokens, mask, embeddings, context, labels=labels)
        embeddings = torch.randn(tokens.shape[0], tokens.shape[1], head.config.embedding_dim)
        bad_context = torch.randn(tokens.shape[0], head.config.embedding_dim + 1)
        with self.assertRaises(ValueError):
            head(tokens, mask, embeddings, bad_context, labels=labels)

    def test_zero_extra_candidates_is_supported(self):
        cfg = AggressiveSoftViewConfig(embedding_dim=8, num_extra_candidates=0, dropout=0.0)
        head = AggressiveSoftViewHead(cfg)
        tokens, mask, labels = self.make_batch()
        embeddings = torch.randn(tokens.shape[0], tokens.shape[1], cfg.embedding_dim)
        context = torch.randn(tokens.shape[0], cfg.embedding_dim)
        view = head(tokens, mask, embeddings, context, labels=labels)
        self.assertEqual(tuple(view.tokens.shape), (2, 5, RAW_TOKEN_DIM))
        self.assertEqual(view.metadata["n_extra_candidates"], 0)
        self.assertEqual(tuple(view.aux["extra_tokens"].shape), (2, 0, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.aux["extra_raw"].shape), (2, 0, RAW_TOKEN_DIM + 1))
        self.assertEqual(view.metadata["diagnostics"]["extra_slot_usage_histogram"], [2])


if __name__ == "__main__":
    unittest.main()

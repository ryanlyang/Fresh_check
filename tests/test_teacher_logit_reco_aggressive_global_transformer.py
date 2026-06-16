import importlib.util
import math
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.aggressive_global_transformer import (
    AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR,
    AggressiveGlobalTransformerReconstructor,
    AggressiveGlobalTransformerReconstructorConfig,
)
from teacher_logit_reco.aggressive_soft_view import AGGRESSION_LEVEL
from teacher_logit_reco.views import SoftReconstructedView

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


class AggressiveGlobalTransformerConfigTests(unittest.TestCase):
    def test_config_accepts_top_level_head_keys(self):
        cfg = AggressiveGlobalTransformerReconstructorConfig.from_mapping(
            {
                "architecture": "aggressive_gt",
                "hidden_dim": 32,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "num_extra_candidates": 5,
                "max_delta_logpt": 0.75,
            }
        )
        self.assertEqual(cfg.reconstructor_architecture, AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR)
        self.assertEqual(cfg.aggression_level, AGGRESSION_LEVEL)
        self.assertEqual(cfg.aggressive_head_config["embedding_dim"], 32)
        self.assertEqual(cfg.aggressive_head_config["num_extra_candidates"], 5)
        self.assertEqual(cfg.aggressive_head_config["max_delta_logpt"], 0.75)
        self.assertEqual(AggressiveGlobalTransformerReconstructorConfig.from_mapping(cfg.to_dict()).hidden_dim, 32)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            AggressiveGlobalTransformerReconstructorConfig(hidden_dim=30, num_heads=8)
        with self.assertRaises(ValueError):
            AggressiveGlobalTransformerReconstructorConfig(dropout=1.0)
        with self.assertRaises(ValueError):
            AggressiveGlobalTransformerReconstructorConfig(
                hidden_dim=32,
                num_heads=4,
                aggressive_head_config={"embedding_dim": 16},
            )
        with self.assertRaises(TypeError):
            AggressiveGlobalTransformerReconstructorConfig.from_mapping({"not_a_config_key": 1})


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class AggressiveGlobalTransformerForwardTests(unittest.TestCase):
    def make_batch(self):
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((2, 5), dtype=torch.bool)
        mask[0, :3] = True
        mask[1, :2] = True
        for b in range(2):
            for i in range(5):
                if not bool(mask[b, i]):
                    continue
                pt = 12.0 + b + i
                eta = 0.08 * i
                phi = -0.25 + 0.12 * i
                tokens[b, i, 0] = pt
                tokens[b, i, 1] = eta
                tokens[b, i, 2] = phi
                tokens[b, i, 3] = pt * math.cosh(eta) + 0.4
                tokens[b, i, 4] = 1.0
                tokens[b, i, 5 + (i % 5)] = 1.0
                tokens[b, i, 10:14] = torch.tensor([0.1, 0.01, -0.2, 0.02])
        labels = torch.tensor([1, 8], dtype=torch.long)
        return tokens, mask, labels

    def make_model(self):
        torch.manual_seed(123)
        return AggressiveGlobalTransformerReconstructor(
            {
                "hidden_dim": 32,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "num_extra_candidates": 4,
            }
        )

    def test_forward_uses_aggressive_soft_view_contract(self):
        model = self.make_model()
        tokens, mask, labels = self.make_batch()
        view = model(tokens, mask, labels=labels, split="model_train")

        self.assertIsInstance(view, SoftReconstructedView)
        self.assertEqual(tuple(view.tokens.shape), (2, 9, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.mask.shape), (2, 9))
        self.assertEqual(tuple(view.weights.shape), (2, 9))
        self.assertTrue(bool(torch.equal(view.mask[:, :5], mask)))
        self.assertTrue(bool(view.mask[:, 5:].all()))
        self.assertTrue(bool(torch.isfinite(view.tokens).all()))
        self.assertTrue(bool(torch.isfinite(view.weights).all()))
        self.assertTrue(bool((view.weights >= 0.0).all()))
        self.assertTrue(bool((view.weights <= 1.0).all()))
        self.assertEqual(view.metadata["reconstructor_architecture"], AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR)
        self.assertEqual(view.metadata["model_family"], "teacher_logit_aggressive_global_transformer")
        self.assertEqual(view.metadata["aggression_level"], AGGRESSION_LEVEL)
        self.assertEqual(view.metadata["n_extra_candidates"], 4)
        self.assertTrue(view.metadata["global_calibration_enabled"])
        self.assertTrue(view.metadata["parent_reweighting_enabled"])
        self.assertIn("encoder_diagnostics", view.metadata)
        self.assertIn("parent_raw", view.aux)
        self.assertIn("extra_raw", view.aux)
        self.assertIn("global_raw", view.aux)
        self.assertIn("extra_slot_usage_mean", view.metadata["diagnostics"])
        self.assertIn("global_logpt_scale_abs_mean", view.metadata["diagnostics"])

    def test_encode_returns_expected_embedding_contract(self):
        model = self.make_model()
        tokens, mask, _ = self.make_batch()
        hlt_tokens, hlt_mask, particle_embeddings, global_context, diagnostics = model.encode(tokens, mask)
        self.assertEqual(tuple(hlt_tokens.shape), tuple(tokens.shape))
        self.assertTrue(bool(torch.equal(hlt_mask, mask)))
        self.assertEqual(tuple(particle_embeddings.shape), (2, 5, model.config.hidden_dim))
        self.assertEqual(tuple(global_context.shape), (2, model.config.hidden_dim))
        self.assertIn("empty_input_jet_count", diagnostics)


if __name__ == "__main__":
    unittest.main()

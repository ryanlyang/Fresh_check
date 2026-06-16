import importlib.util
import math
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.aggressive_particle_reconstructors import (
    AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR,
    AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR,
    AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR,
    AggressiveParticleCnnReconstructor,
    AggressiveParticleCnnReconstructorConfig,
    AggressiveParticleFlowReconstructor,
    AggressiveParticleFlowReconstructorConfig,
    AggressiveParticleNetReconstructor,
    AggressiveParticleNetReconstructorConfig,
)
from teacher_logit_reco.aggressive_soft_view import AGGRESSION_LEVEL
from teacher_logit_reco.views import SoftReconstructedView

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


class AggressiveParticleReconstructorConfigTests(unittest.TestCase):
    def test_particle_net_config_routes_head_keys(self):
        cfg = AggressiveParticleNetReconstructorConfig.from_mapping(
            {
                "architecture": "aggressive_pn",
                "edgeconv_dims": [16, 24],
                "k": 4,
                "dropout": 0.0,
                "embedding_dim": 20,
                "num_extra_candidates": 6,
                "max_delta_logpt": 0.8,
            }
        )
        self.assertEqual(cfg.reconstructor_architecture, AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR)
        self.assertEqual(cfg.aggression_level, AGGRESSION_LEVEL)
        self.assertEqual(cfg.edgeconv_dims, (16, 24))
        self.assertEqual(cfg.aggressive_head_config["embedding_dim"], 20)
        self.assertEqual(cfg.aggressive_head_config["num_extra_candidates"], 6)
        self.assertEqual(cfg.aggressive_head_config["max_delta_logpt"], 0.8)
        with self.assertRaises(ValueError):
            AggressiveParticleNetReconstructorConfig(k=0)

    def test_particle_flow_config_routes_head_keys_and_aliases(self):
        cfg = AggressiveParticleFlowReconstructorConfig.from_mapping(
            {
                "architecture": "aggressive_pfn",
                "phi_dims": [16, 24],
                "context_dim": 32,
                "context_dims": [32],
                "embedding_dim": 20,
                "num_extra_candidates": 5,
            }
        )
        self.assertEqual(cfg.reconstructor_architecture, AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR)
        self.assertEqual(cfg.phi_dims, (16, 24))
        self.assertEqual(cfg.context_mlp_dims, (32,))
        self.assertEqual(cfg.aggressive_head_config["embedding_dim"], 20)
        self.assertEqual(cfg.aggressive_head_config["num_extra_candidates"], 5)
        with self.assertRaises(ValueError):
            AggressiveParticleFlowReconstructorConfig(embedding_dim=0)

    def test_particle_cnn_config_routes_head_keys_and_validates_rank_convs(self):
        cfg = AggressiveParticleCnnReconstructorConfig.from_mapping(
            {
                "architecture": "aggressive_pcnn",
                "hidden_channels": 16,
                "num_blocks": 2,
                "kernel_sizes": [5, 3],
                "dilations": [1, 2],
                "context_dim": 32,
                "context_dims": [32],
                "embedding_dim": 20,
                "num_extra_candidates": 4,
            }
        )
        self.assertEqual(cfg.reconstructor_architecture, AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR)
        self.assertEqual(cfg.kernel_sizes, (5, 3))
        self.assertEqual(cfg.dilations, (1, 2))
        self.assertEqual(cfg.context_mlp_dims, (32,))
        self.assertEqual(cfg.aggressive_head_config["embedding_dim"], 20)
        self.assertEqual(cfg.aggressive_head_config["num_extra_candidates"], 4)
        with self.assertRaises(ValueError):
            AggressiveParticleCnnReconstructorConfig(num_blocks=1, kernel_sizes=(4,), dilations=(1,))

    def test_unknown_config_keys_raise(self):
        with self.assertRaises(TypeError):
            AggressiveParticleNetReconstructorConfig.from_mapping({"not_a_real_key": 1})


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class AggressiveParticleReconstructorForwardTests(unittest.TestCase):
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
                eta = 0.06 * i
                phi = -0.3 + 0.11 * i
                tokens[b, i, 0] = pt
                tokens[b, i, 1] = eta
                tokens[b, i, 2] = phi
                tokens[b, i, 3] = pt * math.cosh(eta) + 0.3
                tokens[b, i, 4] = 1.0
                tokens[b, i, 5 + (i % 5)] = 1.0
                tokens[b, i, 10:14] = torch.tensor([0.1, 0.01, -0.2, 0.02])
        labels = torch.tensor([1, 8], dtype=torch.long)
        return tokens, mask, labels

    def model_specs(self):
        torch.manual_seed(456)
        return [
            (
                AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR,
                AggressiveParticleNetReconstructor(
                    {
                        "edgeconv_dims": [16, 20],
                        "embedding_dim": 24,
                        "k": 4,
                        "dropout": 0.0,
                        "num_extra_candidates": 2,
                    }
                ),
                "particle_net_features",
            ),
            (
                AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR,
                AggressiveParticleFlowReconstructor(
                    {
                        "phi_dims": [16],
                        "context_dim": 20,
                        "context_mlp_dims": [20],
                        "embedding_dim": 24,
                        "dropout": 0.0,
                        "num_extra_candidates": 2,
                    }
                ),
                "particle_flow_features",
            ),
            (
                AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR,
                AggressiveParticleCnnReconstructor(
                    {
                        "hidden_channels": 16,
                        "num_blocks": 2,
                        "kernel_sizes": [3, 3],
                        "dilations": [1, 2],
                        "context_dim": 20,
                        "context_mlp_dims": [20],
                        "embedding_dim": 24,
                        "dropout": 0.0,
                        "num_extra_candidates": 2,
                    }
                ),
                "particle_cnn_features",
            ),
        ]

    def test_forward_uses_shared_aggressive_head_contract(self):
        tokens, mask, labels = self.make_batch()
        for architecture, model, feature_key in self.model_specs():
            with self.subTest(architecture=architecture):
                view = model(tokens, mask, labels=labels, split="model_train")
                self.assertIsInstance(view, SoftReconstructedView)
                self.assertEqual(tuple(view.tokens.shape), (2, 7, RAW_TOKEN_DIM))
                self.assertEqual(tuple(view.mask.shape), (2, 7))
                self.assertEqual(tuple(view.weights.shape), (2, 7))
                self.assertTrue(bool(torch.equal(view.mask[:, :5], mask)))
                self.assertTrue(bool(view.mask[:, 5:].all()))
                self.assertTrue(bool(torch.isfinite(view.tokens).all()))
                self.assertTrue(bool(torch.isfinite(view.weights).all()))
                self.assertTrue(bool((view.weights >= 0.0).all()))
                self.assertTrue(bool((view.weights <= 1.0).all()))
                self.assertEqual(view.metadata["reconstructor_architecture"], architecture)
                self.assertEqual(view.metadata["aggression_level"], AGGRESSION_LEVEL)
                self.assertEqual(view.metadata["n_extra_candidates"], 2)
                self.assertTrue(view.metadata["global_calibration_enabled"])
                self.assertTrue(view.metadata["parent_reweighting_enabled"])
                self.assertIn(feature_key, view.aux)
                self.assertIn("parent_raw", view.aux)
                self.assertIn("extra_raw", view.aux)
                self.assertIn("global_raw", view.aux)
                self.assertIn("extra_slot_usage_mean", view.metadata["diagnostics"])

    def test_encode_returns_embedding_and_context_contract(self):
        tokens, mask, _ = self.make_batch()
        for architecture, model, _ in self.model_specs():
            with self.subTest(architecture=architecture):
                hlt_tokens, hlt_mask, particle_embeddings, global_context, diagnostics, encoder_aux = model.encode(tokens, mask)
                self.assertEqual(tuple(hlt_tokens.shape), tuple(tokens.shape))
                self.assertTrue(bool(torch.equal(hlt_mask, mask)))
                self.assertEqual(tuple(particle_embeddings.shape), (2, 5, model.config.embedding_dim))
                self.assertEqual(tuple(global_context.shape), (2, model.config.embedding_dim))
                self.assertIn("empty_input_jet_count", diagnostics)
                self.assertIn("global_context", encoder_aux)


if __name__ == "__main__":
    unittest.main()

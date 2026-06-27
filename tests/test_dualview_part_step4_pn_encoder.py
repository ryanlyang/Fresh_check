import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.dualview_part import (
        DUALVIEW_PART_PN_ENCODER_CONTRACT,
        DUALVIEW_PART_STEP4,
        PNMemoryEncoderConfig,
        build_pn_memory_encoder,
    )
else:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DualViewPartStep4PNEncoderTests(unittest.TestCase):
    def make_batch(self, batch_size=4, n_tokens=7, feature_dim=RAW_TOKEN_DIM):
        generator = torch.Generator().manual_seed(17)
        tokens = torch.randn(batch_size, n_tokens, feature_dim, generator=generator)
        mask = torch.ones(batch_size, n_tokens, dtype=torch.bool)
        if batch_size > 1:
            mask[1, -2:] = False
        if batch_size > 2:
            mask[2, :] = False
        confidence = torch.rand(batch_size, n_tokens, generator=generator)
        confidence = torch.where(mask, confidence, torch.zeros_like(confidence))
        return tokens, mask, confidence

    def test_config_defaults_match_step4_contract(self):
        cfg = PNMemoryEncoderConfig(embed_dim=32, num_layers=1, num_heads=4)

        self.assertEqual(cfg.raw_token_dim, RAW_TOKEN_DIM)
        self.assertEqual(cfg.output_contract, DUALVIEW_PART_PN_ENCODER_CONTRACT)
        self.assertEqual(cfg.experiment_step, DUALVIEW_PART_STEP4)
        self.assertEqual(cfg.view_name, "pn_reco")
        self.assertEqual(cfg.source_architecture, "pn")
        self.assertTrue(cfg.use_confidence)

    def test_config_validation_rejects_bad_shapes(self):
        bad_configs = [
            {"raw_token_dim": 0},
            {"embed_dim": 30, "num_heads": 8},
            {"num_layers": 0},
            {"dropout": 1.0},
            {"attention_dropout": -0.1},
            {"use_confidence": "yes"},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    PNMemoryEncoderConfig(**kwargs)

    def test_encoder_returns_masked_memory_and_finite_context(self):
        model = build_pn_memory_encoder(embed_dim=32, num_layers=1, num_heads=4, dropout=0.0)
        tokens, mask, confidence = self.make_batch()
        output = model(tokens, mask, confidence, return_diagnostics=True)

        self.assertEqual(tuple(output.memory.shape), (4, 7, 32))
        self.assertEqual(tuple(output.context.shape), (4, 32))
        self.assertEqual(tuple(output.memory_mask.shape), (4, 7))
        self.assertTrue(torch.equal(output.memory_mask, mask))
        self.assertTrue(bool(torch.isfinite(output.memory).all()))
        self.assertTrue(bool(torch.isfinite(output.context).all()))
        self.assertTrue(bool((output.memory[~mask] == 0.0).all()))
        self.assertEqual(output.diagnostics["output_contract"], DUALVIEW_PART_PN_ENCODER_CONTRACT)
        self.assertGreaterEqual(output.diagnostics["empty_jet_fraction"], 0.0)

    def test_all_empty_batch_is_finite(self):
        model = build_pn_memory_encoder(embed_dim=24, num_layers=1, num_heads=4, dropout=0.0)
        tokens = torch.randn(3, 5, RAW_TOKEN_DIM)
        mask = torch.zeros(3, 5, dtype=torch.bool)
        output = model(tokens, mask)

        self.assertTrue(bool(torch.isfinite(output.memory).all()))
        self.assertTrue(bool(torch.isfinite(output.context).all()))
        self.assertTrue(bool((output.memory == 0.0).all()))

    def test_confidence_changes_memory_when_enabled(self):
        torch.manual_seed(23)
        model = build_pn_memory_encoder(embed_dim=32, num_layers=1, num_heads=4, dropout=0.0)
        model.eval()
        tokens, mask, confidence = self.make_batch(batch_size=2, n_tokens=6)
        high = torch.where(mask, torch.ones_like(confidence), torch.zeros_like(confidence))
        low = torch.zeros_like(confidence)
        with torch.no_grad():
            high_output = model(tokens, mask, high)
            low_output = model(tokens, mask, low)

        delta = (high_output.context - low_output.context).abs().max().item()
        self.assertGreater(delta, 1.0e-5)

    def test_confidence_can_be_disabled(self):
        torch.manual_seed(29)
        model = build_pn_memory_encoder(embed_dim=32, num_layers=1, num_heads=4, dropout=0.0, use_confidence=False)
        model.eval()
        tokens, mask, confidence = self.make_batch(batch_size=2, n_tokens=6)
        with torch.no_grad():
            output_a = model(tokens, mask, confidence)
            output_b = model(tokens, mask, torch.zeros_like(confidence))

        self.assertLess((output_a.context - output_b.context).abs().max().item(), 1.0e-6)

    def test_gradient_flows_through_valid_tokens(self):
        torch.manual_seed(31)
        model = build_pn_memory_encoder(embed_dim=32, num_layers=1, num_heads=4, dropout=0.0)
        tokens, mask, confidence = self.make_batch(batch_size=2, n_tokens=5)
        tokens.requires_grad_(True)
        output = model(tokens, mask, confidence)
        loss = output.context.square().mean() + output.memory[mask].square().mean()
        loss.backward()

        self.assertIsNotNone(tokens.grad)
        self.assertTrue(bool(torch.isfinite(tokens.grad).all()))
        self.assertGreater(tokens.grad[mask].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()

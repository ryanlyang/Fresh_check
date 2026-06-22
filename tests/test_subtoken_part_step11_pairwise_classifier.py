import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
    SUBTOKEN_PART_GATE_LOCAL_SOFTMAX,
    SUBTOKEN_PART_PAIRWISE_CLASSIFIER_CONTRACT,
    SubtokenPartConfig,
    SubtokenParticleTransformerClassifier,
    build_subtoken_particle_transformer_classifier,
)


class SubtokenPartStep11PairwiseClassifierTests(unittest.TestCase):
    def make_config(self, **kwargs):
        defaults = {
            "num_classes": 2,
            "embed_dim": 16,
            "local_layers": 1,
            "local_heads": 4,
            "context_layers": 1,
            "context_heads": 4,
            "global_layers": 2,
            "global_heads": 4,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "use_pairwise_bias": True,
        }
        defaults.update(kwargs)
        return SubtokenPartConfig(**defaults)

    def make_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, True, False, True],
                [True, False, True, True, False],
            ],
            dtype=torch.bool,
        )
        tokens[:, :, 0] = torch.tensor(
            [
                [80.0, 45.0, 20.0, 900.0, 12.0],
                [60.0, 800.0, 18.0, 10.0, 700.0],
            ]
        )
        tokens[:, :, 1] = torch.tensor(
            [
                [0.1, -0.2, 0.7, 9.0, -1.1],
                [-0.5, 8.0, 0.3, 1.2, -7.0],
            ]
        )
        tokens[:, :, 2] = torch.tensor(
            [
                [0.2, 2.4, -2.8, -8.0, 1.3],
                [-2.5, 8.0, 0.6, -0.4, 7.0],
            ]
        )
        tokens[:, :, 3] = tokens[:, :, 0] + 15.0
        tokens[:, :, 4] = torch.tensor(
            [
                [1.0, -1.0, 0.0, 3.0, 1.0],
                [-1.0, 5.0, 0.0, 1.0, 5.0],
            ]
        )
        pid_rows = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [9, 9, 9, 9, 9],
                [0, 0, 0, 1, 0],
            ],
            dtype=torch.float32,
        )
        tokens[0, :, 5:10] = pid_rows
        tokens[1, :, 5:10] = pid_rows.roll(1, dims=0)
        tokens[:, :, 10] = torch.tensor(
            [
                [0.2, -0.3, 0.4, 99.0, -0.1],
                [0.5, 8.0, -0.2, 0.1, 7.0],
            ]
        )
        tokens[:, :, 11] = torch.tensor(
            [
                [0.1, 0.2, 0.3, 99.0, 0.4],
                [0.4, 8.0, 0.5, 0.6, 7.0],
            ]
        )
        tokens[:, :, 12] = torch.tensor(
            [
                [-0.1, 0.2, -0.4, 99.0, 0.1],
                [0.7, 8.0, 0.3, -0.2, 7.0],
            ]
        )
        tokens[:, :, 13] = torch.tensor(
            [
                [0.6, 0.7, 0.8, 99.0, 0.2],
                [0.9, 8.0, 1.0, 0.1, 7.0],
            ]
        )
        return tokens, mask

    def test_pairwise_classifier_forward_returns_serious_contract(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        model = SubtokenParticleTransformerClassifier(self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX))

        output = model(tokens, mask, return_outputs=True)
        summary = output.summary()

        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(summary["contract"], SUBTOKEN_PART_PAIRWISE_CLASSIFIER_CONTRACT)
        self.assertTrue(summary["serious_comparison_ready"])
        self.assertFalse(summary["smoke_only_without_pairwise_bias"])
        self.assertIsNotNone(output.pairwise_features)
        self.assertIsNotNone(output.attention_bias)
        self.assertEqual(tuple(output.global_mask.shape), (2, 6))
        self.assertEqual(tuple(output.attention_bias.shape), (2, 4, 6, 6))
        self.assertTrue(bool(torch.equal(output.global_mask[:, 1:], mask)))
        self.assertTrue(bool(torch.isfinite(output.logits).all()))

    def test_pairwise_classifier_diagnostics_include_bias_stats(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        model = SubtokenParticleTransformerClassifier(self.make_config(gate_mode=SUBTOKEN_PART_GATE_LOCAL_SOFTMAX))

        diagnostics = model(tokens, mask, return_outputs=True).diagnostics()

        self.assertEqual(diagnostics["contract"], SUBTOKEN_PART_PAIRWISE_CLASSIFIER_CONTRACT)
        self.assertTrue(diagnostics["serious_comparison_ready"])
        self.assertTrue(bool(torch.isfinite(diagnostics["pairwise_attention_bias_abs_mean"]).all()))
        self.assertTrue(bool(torch.isfinite(diagnostics["pairwise_attention_bias_abs_max"]).all()))

    def test_pairwise_bias_encoder_receives_gradients(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        labels = torch.tensor([0, 1], dtype=torch.long)
        model = SubtokenParticleTransformerClassifier(self.make_config())

        output = model(tokens, mask, return_outputs=True)
        loss = torch.nn.functional.cross_entropy(output.logits, labels) + 0.001 * output.attention_bias.square().mean()
        loss.backward()

        grad_sum = sum(
            float(param.grad.detach().abs().sum().cpu().item())
            for param in model.pairwise_bias_encoder.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)

    def test_raw_kinematics_change_pairwise_attention_bias(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        changed = tokens.clone()
        changed[:, 0, 2] = changed[:, 0, 2] + 0.7
        model = SubtokenParticleTransformerClassifier(self.make_config())
        model.eval()

        baseline = model(tokens, mask, return_outputs=True).attention_bias
        shifted = model(changed, mask, return_outputs=True).attention_bias

        delta = (baseline - shifted).abs().sum()
        self.assertGreater(float(delta.detach().cpu().item()), 1.0e-6)

    def test_default_builder_uses_pairwise_classifier(self):
        model = build_subtoken_particle_transformer_classifier(
            num_classes=2,
            embed_dim=16,
            local_heads=4,
            context_heads=4,
            global_heads=4,
            global_layers=1,
            dropout=0.0,
            attention_dropout=0.0,
        )

        self.assertTrue(model.config.use_pairwise_bias)
        self.assertEqual(model.output_contract, SUBTOKEN_PART_PAIRWISE_CLASSIFIER_CONTRACT)


if __name__ == "__main__":
    unittest.main()

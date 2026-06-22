import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_CLASSIFIER_CONTRACT,
    SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
    SUBTOKEN_PART_GATE_LOCAL_SOFTMAX,
    SUBTOKEN_PART_GATE_NONE,
    SubtokenPartConfig,
    SubtokenParticleTransformerClassifier,
    build_subtoken_particle_transformer_classifier,
)


class SubtokenPartStep9ClassifierTests(unittest.TestCase):
    def make_config(self, **kwargs):
        defaults = {
            "num_classes": 2,
            "embed_dim": 16,
            "local_layers": 1,
            "local_heads": 4,
            "context_layers": 1,
            "context_heads": 4,
            "global_layers": 1,
            "global_heads": 4,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "use_pairwise_bias": False,
        }
        defaults.update(kwargs)
        return SubtokenPartConfig(**defaults)

    def make_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((3, 6, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, False, True, False, True, False],
                [True, True, True, True, True, True],
            ],
            dtype=torch.bool,
        )
        tokens[:, :, 0] = torch.tensor(
            [
                [50.0, 20.0, 5.0, 900.0, 900.0, 900.0],
                [30.0, 800.0, 12.0, 700.0, 6.0, 700.0],
                [100.0, 80.0, 60.0, 30.0, 20.0, 10.0],
            ]
        )
        tokens[:, :, 1] = torch.tensor(
            [
                [0.2, -0.5, 0.1, 9.0, 9.0, 9.0],
                [-0.3, 8.0, 0.7, -7.0, 0.4, -7.0],
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            ]
        )
        tokens[:, :, 2] = torch.tensor(
            [
                [0.1, 2.9, -2.1, -8.0, -8.0, -8.0],
                [1.5, 8.0, -2.1, 7.0, -0.6, 7.0],
                [-3.0, 2.8, 0.5, -0.8, 1.2, -1.4],
            ]
        )
        tokens[:, :, 3] = tokens[:, :, 0] + 10.0
        tokens[:, :, 4] = torch.tensor(
            [
                [1.0, -1.0, 0.0, 3.0, 3.0, 3.0],
                [0.0, -2.0, 1.0, 5.0, -1.0, 5.0],
                [1.0, -1.0, 0.0, 1.0, -1.0, 0.0],
            ]
        )
        pid_rows = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [9, 9, 9, 9, 9],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        tokens[0, :, 5:10] = pid_rows
        tokens[1, :, 5:10] = pid_rows.roll(1, dims=0)
        tokens[2, :, 5:10] = pid_rows.roll(2, dims=0)
        tokens[:, :, 10] = torch.tensor(
            [
                [0.2, -0.3, 0.4, 99.0, 99.0, 99.0],
                [0.5, 8.0, -0.2, 7.0, 0.2, 7.0],
                [0.1, -0.1, 0.2, -0.2, 0.3, -0.3],
            ]
        )
        tokens[:, :, 11] = torch.tensor(
            [
                [0.1, 0.2, 0.3, 99.0, 99.0, 99.0],
                [0.4, 8.0, 0.5, 7.0, 0.6, 7.0],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            ]
        )
        tokens[:, :, 12] = torch.tensor(
            [
                [-0.1, 0.2, -0.4, 99.0, 99.0, 99.0],
                [0.7, 8.0, 0.3, 7.0, -0.2, 7.0],
                [-0.2, 0.2, -0.3, 0.3, -0.4, 0.4],
            ]
        )
        tokens[:, :, 13] = torch.tensor(
            [
                [0.6, 0.7, 0.8, 99.0, 99.0, 99.0],
                [0.9, 8.0, 1.0, 7.0, 0.1, 7.0],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            ]
        )
        return tokens, mask

    def test_forward_pass_returns_binary_logits_and_outputs(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        model = SubtokenParticleTransformerClassifier(self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX))

        logits = model(tokens, mask)
        output = model(tokens, mask, return_outputs=True)

        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(tuple(output.logits.shape), (3, 2))
        self.assertEqual(output.summary()["contract"], SUBTOKEN_PART_CLASSIFIER_CONTRACT)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(bool(torch.isfinite(output.cls_embedding).all()))
        self.assertIsNotNone(output.context)
        self.assertIsNotNone(output.gates)
        self.assertTrue(bool((output.particles.particle_tokens[~mask] == 0.0).all()))

    def test_multiclass_output_dims(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        model = SubtokenParticleTransformerClassifier(self.make_config(num_classes=5, gate_mode=SUBTOKEN_PART_GATE_LOCAL_SOFTMAX))

        logits, diagnostics = model({"tokens": tokens, "mask": mask}, return_diagnostics=True)

        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertEqual(diagnostics["contract"], SUBTOKEN_PART_CLASSIFIER_CONTRACT)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(bool(torch.isfinite(diagnostics["valid_particle_count_mean"]).all()))

    def test_no_gate_variant_skips_context_and_gate_outputs(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        model = SubtokenParticleTransformerClassifier(self.make_config(gate_mode=SUBTOKEN_PART_GATE_NONE))

        output = model(tokens, mask, return_outputs=True)

        self.assertIsNone(output.context)
        self.assertIsNone(output.gates)
        self.assertFalse(output.particles.used_reliability_gates)
        self.assertTrue(bool(torch.allclose(output.particles.particle_tokens, output.pooled.provisional_particles, atol=1.0e-6)))

    def test_loss_backward_updates_model_parameters(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        labels = torch.tensor([0, 1, 0], dtype=torch.long)
        model = SubtokenParticleTransformerClassifier(self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX))

        logits = model(tokens, mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        grad_sum = sum(
            float(param.grad.detach().abs().sum().cpu().item())
            for param in model.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)

    def test_variable_particle_counts_are_finite(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        mask = mask.clone()
        mask[1] = torch.tensor([True, False, False, False, False, False])
        model = SubtokenParticleTransformerClassifier(self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX))

        output = model(tokens, mask, return_outputs=True)

        self.assertTrue(bool(torch.isfinite(output.logits).all()))
        self.assertTrue(bool((output.global_mask[:, 0] == True).all()))
        self.assertTrue(bool(torch.equal(output.global_mask[:, 1:], mask)))
        self.assertTrue(bool((output.particles.particle_tokens[~mask] == 0.0).all()))

    def test_pairwise_bias_disabled_is_smoke_only_ablation(self):
        tokens, mask = self.make_tokens()
        model = SubtokenParticleTransformerClassifier(self.make_config(use_pairwise_bias=False))

        output = model(tokens, mask, return_outputs=True)
        diagnostics = output.diagnostics()

        self.assertIsNone(output.pairwise_features)
        self.assertIsNone(output.attention_bias)
        self.assertTrue(output.summary()["smoke_only_without_pairwise_bias"])
        self.assertFalse(output.summary()["serious_comparison_ready"])
        self.assertTrue(diagnostics["smoke_only_without_pairwise_bias"])
        self.assertFalse(diagnostics["serious_comparison_ready"])

    def test_build_helper_accepts_overrides(self):
        model = build_subtoken_particle_transformer_classifier(
            num_classes=3,
            embed_dim=16,
            local_heads=4,
            context_heads=4,
            global_heads=4,
            use_pairwise_bias=False,
            dropout=0.0,
            attention_dropout=0.0,
        )

        self.assertEqual(model.config.num_classes, 3)
        self.assertEqual(model.output_contract, SUBTOKEN_PART_CLASSIFIER_CONTRACT)


if __name__ == "__main__":
    unittest.main()

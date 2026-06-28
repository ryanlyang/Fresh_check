import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.multiscale_subjet_part import (
    CANONICAL_PART_FEATURE_NAMES,
    MULTISCALE_SUBJET_CROSS_ATTENTION_CONTRACT,
    MULTISCALE_SUBJET_CROSS_ATTENTION_STEP,
    MultiScaleSubjetTokenBuilder,
    MultiScaleSubjetTokenBuilderConfig,
    ParticleSubjetCrossAttentionConfig,
    ParticleSubjetCrossAttentionReadback,
    SoftSubjetAssignmentConfig,
    SubjetScaleSpec,
    SubjetSubjetTransformer,
    SubjetTransformerConfig,
)

torch = require_torch()


def make_tokens():
    tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    rows = [
        [(50.0, 0.00, 0.00, 55.0), (20.0, 0.05, 0.01, 22.0), (10.0, 1.00, 1.00, 12.0)],
        [(30.0, -0.10, 0.20, 35.0), (5.0, 2.00, -2.00, 6.0)],
    ]
    for batch_index, batch_rows in enumerate(rows):
        for particle_index, (pt, eta, phi, energy) in enumerate(batch_rows):
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = energy
            tokens[batch_index, particle_index, 5] = 1.0
    return tokens, mask


def make_subjet_output(tokens, mask):
    assignment = SoftSubjetAssignmentConfig(
        scale_specs=(
            SubjetScaleSpec("small", 2, 0.05, 0.12, "tight"),
            SubjetScaleSpec("medium", 1, 0.12, 0.25, "medium"),
            SubjetScaleSpec("large", 1, 0.25, 0.50, "wide"),
        ),
        embed_dim=16,
        hidden_dim=32,
    )
    token_builder = MultiScaleSubjetTokenBuilder(
        MultiScaleSubjetTokenBuilderConfig(
            assignment_config=assignment,
            token_dim=24,
            hidden_dim=48,
            dropout=0.0,
        )
    )
    token_output = token_builder(tokens, mask)
    transformer = SubjetSubjetTransformer(
        SubjetTransformerConfig(
            token_dim=24,
            num_layers=1,
            num_heads=4,
            ffn_dim=48,
            dropout=0.0,
            attention_dropout=0.0,
            num_scales=3,
        )
    )
    return transformer(token_output)


class MultiscaleSubjetPartStep7CrossAttentionTests(unittest.TestCase):
    def test_zero_gamma_readback_is_exact_canonical_feature_noop(self):
        tokens, mask = make_tokens()
        subjet_output = make_subjet_output(tokens, mask)
        readback = ParticleSubjetCrossAttentionReadback(
            ParticleSubjetCrossAttentionConfig(
                feature_dim=len(CANONICAL_PART_FEATURE_NAMES),
                subjet_token_dim=24,
                hidden_dim=32,
                num_heads=4,
                delta_hidden_dim=64,
                dropout=0.0,
                attention_dropout=0.0,
                residual_gamma_init=0.0,
            )
        )
        output = readback(tokens, mask, subjet_output)

        canonical_rows = output.canonical_inputs.feature_rows()
        self.assertTrue(bool(torch.allclose(output.adapted_features, canonical_rows, atol=0.0, rtol=0.0)))
        self.assertEqual(float(output.effective_feature_delta.abs().sum().item()), 0.0)
        self.assertEqual(output.summary()["contract"], MULTISCALE_SUBJET_CROSS_ATTENTION_CONTRACT)
        self.assertEqual(output.diagnostics["step"], MULTISCALE_SUBJET_CROSS_ATTENTION_STEP)
        self.assertTrue(output.diagnostics["lorentz_vectors_unchanged"])
        self.assertTrue(bool(torch.equal(output.part_lorentz_vectors, output.canonical_inputs.lorentz_vectors)))
        self.assertEqual(tuple(output.part_features.shape), tuple(output.canonical_inputs.features.shape))
        self.assertEqual(tuple(output.part_mask.shape), tuple(output.canonical_inputs.mask.shape))

    def test_cross_attention_returns_masks_attention_and_diagnostics(self):
        tokens, mask = make_tokens()
        subjet_output = make_subjet_output(tokens, mask)
        readback = ParticleSubjetCrossAttentionReadback(
            ParticleSubjetCrossAttentionConfig(
                feature_dim=len(CANONICAL_PART_FEATURE_NAMES),
                subjet_token_dim=24,
                hidden_dim=32,
                num_heads=4,
                delta_hidden_dim=64,
                dropout=0.0,
                attention_dropout=0.0,
                residual_gamma_init=0.1,
            )
        )
        output = readback(tokens, mask, subjet_output)

        particle_mask = output.canonical_inputs.mask.squeeze(1)
        self.assertEqual(tuple(output.particle_context.shape[:2]), tuple(particle_mask.shape))
        self.assertEqual(tuple(output.particle_to_subjet_attention.shape[:3]), (2, 4, particle_mask.shape[1]))
        self.assertEqual(float(output.particle_context[~particle_mask].abs().sum().item()), 0.0)
        self.assertIn("particle_to_subjet_attention_entropy_mean", output.diagnostics)
        self.assertIn("subjet_to_particle_attention_entropy_mean", output.diagnostics)
        self.assertGreater(float(output.feature_delta[particle_mask].abs().sum().item()), 0.0)
        self.assertGreater(float(output.effective_feature_delta[particle_mask].abs().sum().item()), 0.0)

    def test_cross_attention_can_disable_subjet_to_particle_read(self):
        tokens, mask = make_tokens()
        subjet_output = make_subjet_output(tokens, mask)
        readback = ParticleSubjetCrossAttentionReadback(
            ParticleSubjetCrossAttentionConfig(
                feature_dim=len(CANONICAL_PART_FEATURE_NAMES),
                subjet_token_dim=24,
                hidden_dim=32,
                num_heads=4,
                delta_hidden_dim=64,
                dropout=0.0,
                use_subjets_read_particles=False,
            )
        )
        output = readback(tokens, mask, subjet_output)

        self.assertIsNone(output.subjet_to_particle_attention)
        self.assertTrue(bool(torch.isfinite(output.updated_subjet_tokens).all()))

    def test_readback_backpropagates_when_gate_is_open(self):
        tokens, mask = make_tokens()
        subjet_output = make_subjet_output(tokens, mask)
        readback = ParticleSubjetCrossAttentionReadback(
            ParticleSubjetCrossAttentionConfig(
                feature_dim=len(CANONICAL_PART_FEATURE_NAMES),
                subjet_token_dim=24,
                hidden_dim=32,
                num_heads=4,
                delta_hidden_dim=64,
                dropout=0.0,
                residual_gamma_init=0.2,
            )
        )
        output = readback(tokens, mask, subjet_output)
        particle_mask = output.canonical_inputs.mask.squeeze(1)
        loss = output.adapted_features[particle_mask].pow(2).mean()
        loss.backward()

        self.assertIsNotNone(readback.particle_reads_subjets.q_proj.weight.grad)
        self.assertGreater(float(readback.particle_reads_subjets.q_proj.weight.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(readback.gamma_F.grad)
        self.assertGreater(float(readback.gamma_F.grad.abs().sum().item()), 0.0)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ParticleSubjetCrossAttentionConfig(hidden_dim=30, num_heads=4)
        with self.assertRaisesRegex(ValueError, "dropout"):
            ParticleSubjetCrossAttentionConfig(dropout=1.0)


if __name__ == "__main__":
    unittest.main()

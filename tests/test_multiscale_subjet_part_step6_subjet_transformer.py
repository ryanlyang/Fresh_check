import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.multiscale_subjet_part import (
    MULTISCALE_SUBJET_PAIR_FEATURE_DIM,
    MULTISCALE_SUBJET_PAIR_FEATURE_NAMES,
    MULTISCALE_SUBJET_TRANSFORMER_CONTRACT,
    MULTISCALE_SUBJET_TRANSFORMER_STEP,
    MultiScaleSubjetTokenBuilder,
    MultiScaleSubjetTokenBuilderConfig,
    SoftSubjetAssignmentConfig,
    SubjetScaleSpec,
    SubjetSubjetTransformer,
    SubjetTransformerConfig,
    build_subjet_pair_features,
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
        [(50.0, 0.00, 3.13, 55.0), (20.0, 0.05, -3.13, 22.0), (10.0, 1.00, 1.00, 12.0)],
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


def small_token_builder():
    assignment = SoftSubjetAssignmentConfig(
        scale_specs=(
            SubjetScaleSpec("small", 2, 0.05, 0.12, "tight"),
            SubjetScaleSpec("medium", 1, 0.12, 0.25, "medium"),
            SubjetScaleSpec("large", 1, 0.25, 0.50, "wide"),
        ),
        embed_dim=16,
        hidden_dim=32,
    )
    return MultiScaleSubjetTokenBuilder(
        MultiScaleSubjetTokenBuilderConfig(
            assignment_config=assignment,
            token_dim=24,
            hidden_dim=48,
            dropout=0.0,
        )
    )


class MultiscaleSubjetPartStep6TransformerTests(unittest.TestCase):
    def test_pair_features_have_expected_shape_and_physics_ranges(self):
        tokens, mask = make_tokens()
        token_output = small_token_builder()(tokens, mask)
        pair_output = build_subjet_pair_features(token_output)

        self.assertEqual(tuple(pair_output.pair_features.shape), (2, 4, 4, MULTISCALE_SUBJET_PAIR_FEATURE_DIM))
        self.assertEqual(tuple(pair_output.pair_mask.shape), (2, 4, 4))
        self.assertEqual(pair_output.feature_names, MULTISCALE_SUBJET_PAIR_FEATURE_NAMES)
        self.assertTrue(bool(torch.isfinite(pair_output.pair_features).all()))
        z_index = MULTISCALE_SUBJET_PAIR_FEATURE_NAMES.index("z")
        containment_i_index = MULTISCALE_SUBJET_PAIR_FEATURE_NAMES.index("containment_i_in_j")
        containment_j_index = MULTISCALE_SUBJET_PAIR_FEATURE_NAMES.index("containment_j_in_i")
        self.assertGreaterEqual(float(pair_output.pair_features[..., z_index].min().item()), 0.0)
        self.assertLessEqual(float(pair_output.pair_features[..., z_index].max().item()), 0.5)
        self.assertGreaterEqual(float(pair_output.pair_features[..., containment_i_index].min().item()), 0.0)
        self.assertLessEqual(float(pair_output.pair_features[..., containment_i_index].max().item()), 1.0)
        self.assertGreaterEqual(float(pair_output.pair_features[..., containment_j_index].min().item()), 0.0)
        self.assertLessEqual(float(pair_output.pair_features[..., containment_j_index].max().item()), 1.0)

    def test_transformer_returns_masked_tokens_and_diagnostics(self):
        tokens, mask = make_tokens()
        token_output = small_token_builder()(tokens, mask)
        transformer = SubjetSubjetTransformer(
            SubjetTransformerConfig(
                token_dim=24,
                num_layers=2,
                num_heads=4,
                ffn_dim=48,
                dropout=0.0,
                attention_dropout=0.0,
                num_scales=3,
            )
        )
        output = transformer(token_output, need_weights=True)

        self.assertEqual(tuple(output.subjet_tokens.shape), (2, 4, 24))
        self.assertEqual(output.summary()["contract"], MULTISCALE_SUBJET_TRANSFORMER_CONTRACT)
        self.assertEqual(output.diagnostics["step"], MULTISCALE_SUBJET_TRANSFORMER_STEP)
        self.assertIsNotNone(output.pair_bias)
        self.assertEqual(tuple(output.attention_weights.shape[:2]), (2, 2))
        self.assertTrue(bool(torch.isfinite(output.subjet_tokens).all()))
        invalid = ~output.subjet_mask
        self.assertEqual(float(output.subjet_tokens[invalid].abs().sum().item()), 0.0)

    def test_transformer_without_pairwise_bias_is_supported(self):
        tokens, mask = make_tokens()
        token_output = small_token_builder()(tokens, mask)
        transformer = SubjetSubjetTransformer(
            SubjetTransformerConfig(
                token_dim=24,
                num_layers=1,
                num_heads=4,
                ffn_dim=48,
                dropout=0.0,
                attention_dropout=0.0,
                use_pairwise_bias=False,
            )
        )
        output = transformer(token_output)

        self.assertIsNone(output.pair_bias)
        self.assertFalse(output.diagnostics["use_pairwise_bias"])
        self.assertTrue(bool(torch.isfinite(output.subjet_tokens).all()))

    def test_transformer_backpropagates(self):
        tokens, mask = make_tokens()
        token_builder = small_token_builder()
        transformer = SubjetSubjetTransformer(
            SubjetTransformerConfig(token_dim=24, num_layers=1, num_heads=4, ffn_dim=48, dropout=0.0)
        )
        token_output = token_builder(tokens, mask)
        output = transformer(token_output)
        loss = output.subjet_tokens[output.subjet_mask].pow(2).mean()
        loss.backward()

        self.assertIsNotNone(transformer.layers[0].attention.in_proj_weight.grad)
        self.assertGreater(float(transformer.layers[0].attention.in_proj_weight.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(token_builder.token_projection[1].weight.grad)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "divide"):
            SubjetTransformerConfig(token_dim=25, num_heads=4)
        with self.assertRaisesRegex(ValueError, "dropout"):
            SubjetTransformerConfig(dropout=1.0)


if __name__ == "__main__":
    unittest.main()

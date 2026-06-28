import unittest
from unittest import mock

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.multiscale_subjet_part import (
    MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED,
    MULTISCALE_SUBJET_FOUR_VECTOR_FEATURE_NAMES,
    MULTISCALE_SUBJET_FOUR_VECTOR_NAMES,
    MULTISCALE_SUBJET_PAIR_OBSERVABLE_NAMES,
    MULTISCALE_SUBJET_TOKEN_BUILDER_CONTRACT,
    MULTISCALE_SUBJET_TOKEN_BUILDER_STEP,
    MultiScaleSubjetTokenBuilder,
    MultiScaleSubjetTokenBuilderConfig,
    SoftSubjetAssignmentConfig,
    SubjetScaleSpec,
)
from teacher_logit_reco.multiscale_subjet_part import assignment as assignment_module

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


def small_token_config(**kwargs):
    assignment = SoftSubjetAssignmentConfig(
        scale_specs=(
            SubjetScaleSpec("small", 2, 0.05, 0.12, "tight"),
            SubjetScaleSpec("medium", 1, 0.12, 0.25, "medium"),
            SubjetScaleSpec("large", 1, 0.25, 0.50, "wide"),
        ),
        embed_dim=16,
        hidden_dim=32,
    )
    return MultiScaleSubjetTokenBuilderConfig(assignment_config=assignment, token_dim=24, hidden_dim=48, dropout=0.0, **kwargs)


class MultiscaleSubjetPartStep5TokenBuilderTests(unittest.TestCase):
    def test_token_builder_returns_step5_outputs_and_diagnostics(self):
        tokens, mask = make_tokens()
        builder = MultiScaleSubjetTokenBuilder(small_token_config())
        output = builder(tokens, mask)

        self.assertEqual(tuple(output.subjet_tokens.shape), (2, 4, 24))
        self.assertEqual(tuple(output.subjet_mask.shape), (2, 4))
        self.assertEqual(tuple(output.assignment_weights.shape), (2, 4, 5))
        self.assertEqual(tuple(output.cluster_weights.shape), (2, 4, 5))
        self.assertEqual(tuple(output.estimated_centers.shape), (2, 4, 2))
        self.assertEqual(tuple(output.cluster_pt_fraction.shape), (2, 4))
        self.assertEqual(tuple(output.soft_four_vectors.shape), (2, 4, len(MULTISCALE_SUBJET_FOUR_VECTOR_NAMES)))
        self.assertEqual(tuple(output.soft_four_vector_features.shape), (2, 4, len(MULTISCALE_SUBJET_FOUR_VECTOR_FEATURE_NAMES)))
        self.assertEqual(
            tuple(output.soft_pair_observable_summaries.shape),
            (2, 4, len(MULTISCALE_SUBJET_PAIR_OBSERVABLE_NAMES)),
        )
        self.assertTrue(bool(torch.isfinite(output.subjet_tokens).all()))
        self.assertTrue(bool(torch.isfinite(output.soft_pair_observable_summaries).all()))
        self.assertEqual(output.summary()["contract"], MULTISCALE_SUBJET_TOKEN_BUILDER_CONTRACT)
        self.assertEqual(output.diagnostics["step"], MULTISCALE_SUBJET_TOKEN_BUILDER_STEP)
        self.assertIn("assignment_entropy_mean", output.diagnostics)
        self.assertIn("attention_pt_fraction_mean", output.diagnostics)
        self.assertIn("cluster_pt_fraction_mean", output.diagnostics)
        self.assertIn("attention_cluster_pt_fraction_abs_diff_mean", output.diagnostics)
        self.assertIn("attention_radius_mean", output.diagnostics)
        self.assertIn("cluster_radius_mean", output.diagnostics)
        self.assertIn("attention_cluster_radius_abs_diff_mean", output.diagnostics)

    def test_token_builder_reuses_canonical_inputs_inside_assignment(self):
        tokens, mask = make_tokens()
        builder = MultiScaleSubjetTokenBuilder(small_token_config())
        with mock.patch.object(
            assignment_module,
            "build_canonical_part_inputs",
            side_effect=AssertionError("assignment recomputed canonical inputs"),
        ):
            output = builder(tokens, mask)

        self.assertTrue(bool(output.subjet_mask.any()))

    def test_hard_geometry_assignment_keeps_attention_proxy_but_sums_cluster_four_vector(self):
        tokens = torch.zeros((1, 3, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor([[True, True, True]], dtype=torch.bool)
        tokens[0, 0, 0] = 100.0
        tokens[0, 0, 1] = 0.0
        tokens[0, 0, 2] = 0.0
        tokens[0, 0, 3] = 105.0
        tokens[0, 1, 0] = 10.0
        tokens[0, 1, 1] = 1.0
        tokens[0, 1, 2] = 1.0
        tokens[0, 1, 3] = 12.0
        tokens[0, 2, 0] = 5.0
        tokens[0, 2, 1] = -1.0
        tokens[0, 2, 2] = -1.0
        tokens[0, 2, 3] = 6.0
        config = MultiScaleSubjetTokenBuilderConfig(
            assignment_config=SoftSubjetAssignmentConfig(
                scale_specs=(SubjetScaleSpec("small", 1, 0.05, 0.12, "tight"),),
                embed_dim=8,
                hidden_dim=16,
                geometry_bias_strength=50.0,
            ),
            token_dim=12,
            hidden_dim=24,
            dropout=0.0,
        )
        builder = MultiScaleSubjetTokenBuilder(config)
        for parameter in builder.assignment.parameters():
            torch.nn.init.zeros_(parameter)
        output = builder(tokens, mask)

        pt = tokens[0, :, 0]
        eta = tokens[0, :, 1]
        phi = tokens[0, :, 2]
        energy = torch.maximum(torch.clamp(tokens[0, :, 3], min=1.0e-4), pt * torch.cosh(eta) + 1.0e-4)
        expected = torch.stack([pt * torch.cos(phi), pt * torch.sin(phi), pt * torch.sinh(eta), energy], dim=-1).sum(dim=0)
        self.assertGreater(float(output.assignment_weights[0, 0, 0].item()), 0.999)
        self.assertTrue(bool(torch.allclose(output.cluster_weights[0, 0], torch.ones(3), atol=1.0e-6)))
        self.assertTrue(bool(torch.allclose(output.soft_four_vectors[0, 0], expected, atol=2.0e-2)))
        self.assertAlmostEqual(float(output.cluster_pt_fraction[0, 0].item()), 1.0, places=5)
        self.assertAlmostEqual(float(output.soft_four_vector_features[0, 0, 5].item()), 1.0, places=5)

    def test_assignment_masks_zero_invalid_subjet_outputs(self):
        tokens, mask = make_tokens()
        builder = MultiScaleSubjetTokenBuilder(small_token_config())
        subjet_assignment_mask = torch.ones((2, 4), dtype=torch.bool)
        subjet_assignment_mask[:, -1] = False
        output = builder(tokens, mask, subjet_assignment_mask=subjet_assignment_mask)

        self.assertFalse(bool(output.subjet_mask[:, -1].any()))
        self.assertEqual(float(output.subjet_tokens[:, -1].abs().sum().item()), 0.0)
        self.assertEqual(float(output.soft_four_vectors[:, -1].abs().sum().item()), 0.0)
        self.assertEqual(float(output.soft_pair_observable_summaries[:, -1].abs().sum().item()), 0.0)
        self.assertEqual(float(output.cluster_pt_fraction[:, -1].abs().sum().item()), 0.0)

    def test_learned_query_empty_jets_are_safe(self):
        tokens = torch.zeros((2, 4, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor(
            [
                [False, False, False, False],
                [True, False, False, False],
            ],
            dtype=torch.bool,
        )
        tokens[1, 0, 0] = 10.0
        tokens[1, 0, 3] = 11.0
        config = small_token_config()
        config = MultiScaleSubjetTokenBuilderConfig(
            assignment_config=SoftSubjetAssignmentConfig(
                scale_specs=config.assignment_config.scale_specs,
                query_mode=MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED,
                embed_dim=16,
                hidden_dim=32,
            ),
            token_dim=24,
            hidden_dim=48,
            dropout=0.0,
        )
        output = MultiScaleSubjetTokenBuilder(config)(tokens, mask)

        self.assertFalse(bool(output.subjet_mask[0].any()))
        self.assertEqual(float(output.subjet_tokens[0].abs().sum().item()), 0.0)
        self.assertTrue(bool(torch.isfinite(output.soft_four_vector_features).all()))
        self.assertTrue(bool(output.subjet_mask[1].all()))

    def test_particle_assignment_mask_invalidates_token_outputs_for_empty_rows(self):
        tokens, mask = make_tokens()
        particle_assignment_mask = mask.clone()
        particle_assignment_mask[0, :] = False
        output = MultiScaleSubjetTokenBuilder(small_token_config())(
            tokens,
            mask,
            particle_assignment_mask=particle_assignment_mask,
        )

        self.assertFalse(bool(output.subjet_mask[0].any()))
        self.assertEqual(float(output.subjet_tokens[0].abs().sum().item()), 0.0)
        self.assertEqual(float(output.cluster_weights[0].abs().sum().item()), 0.0)
        self.assertTrue(bool(output.subjet_mask[1].any()))

    def test_four_vector_features_are_clamped_for_tiny_pt_outliers(self):
        tokens = torch.zeros((1, 2, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor([[True, False]], dtype=torch.bool)
        tokens[0, 0, 0] = 1.0e-9
        tokens[0, 0, 1] = 20.0
        tokens[0, 0, 3] = 100.0
        builder = MultiScaleSubjetTokenBuilder(
            MultiScaleSubjetTokenBuilderConfig(
                assignment_config=SoftSubjetAssignmentConfig(
                    scale_specs=(SubjetScaleSpec("small", 1, 0.05, 0.12, "tight"),),
                    embed_dim=8,
                    hidden_dim=16,
                ),
                token_dim=12,
                hidden_dim=24,
                dropout=0.0,
            )
        )
        output = builder(tokens, mask)

        eta_feature = float(output.soft_four_vector_features[0, 0, 1].item())
        mass_over_pt = float(output.soft_four_vector_features[0, 0, 6].item())
        self.assertLessEqual(abs(eta_feature), 5.0)
        self.assertLessEqual(mass_over_pt, 10.0)

    def test_pair_summaries_are_zero_when_only_one_particle_can_contribute(self):
        tokens = torch.zeros((1, 3, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor([[True, False, False]], dtype=torch.bool)
        tokens[0, 0, 0] = 12.0
        tokens[0, 0, 3] = 13.0
        builder = MultiScaleSubjetTokenBuilder(
            MultiScaleSubjetTokenBuilderConfig(
                assignment_config=SoftSubjetAssignmentConfig(
                    scale_specs=(SubjetScaleSpec("small", 1, 0.05, 0.12, "tight"),),
                    embed_dim=8,
                    hidden_dim=16,
                ),
                token_dim=12,
                hidden_dim=24,
                dropout=0.0,
            )
        )
        output = builder(tokens, mask)

        self.assertTrue(bool(output.subjet_mask[0, 0]))
        self.assertEqual(float(output.soft_pair_observable_summaries.abs().sum().item()), 0.0)

    def test_token_builder_has_parameter_gradients(self):
        tokens, mask = make_tokens()
        builder = MultiScaleSubjetTokenBuilder(small_token_config())
        output = builder(tokens, mask)
        loss = output.subjet_tokens[output.subjet_mask].pow(2).mean()
        loss.backward()

        first_linear = builder.token_projection[1]
        self.assertIsNotNone(first_linear.weight.grad)
        self.assertGreater(float(first_linear.weight.grad.abs().sum().item()), 0.0)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "token_dim"):
            MultiScaleSubjetTokenBuilderConfig(token_dim=0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            MultiScaleSubjetTokenBuilderConfig(dropout=1.0)


if __name__ == "__main__":
    unittest.main()

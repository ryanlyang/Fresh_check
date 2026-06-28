import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.multiscale_subjet_part import (
    MULTISCALE_SUBJET_ASSIGNMENT_CONTRACT,
    MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED,
    MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED,
    MULTISCALE_SUBJET_ASSIGNMENT_STEP,
    MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD,
    MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD,
    SoftSubjetAssignment,
    SoftSubjetAssignmentConfig,
    SubjetScaleSpec,
    SubjetSeedBuilderConfig,
)

torch = require_torch()


def make_assignment_tokens():
    tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    batch0 = [
        (50.0, 0.00, 0.00, 55.0),
        (20.0, 0.04, 0.02, 22.0),
        (15.0, 1.20, 1.00, 18.0),
        (10.0, -1.20, -1.00, 12.0),
    ]
    batch1 = [
        (30.0, 0.10, -0.10, 35.0),
        (25.0, 0.20, -0.15, 28.0),
        (5.0, 2.00, 2.00, 6.0),
    ]
    for batch_index, values in enumerate((batch0, batch1)):
        for idx, (pt, eta, phi, energy) in enumerate(values):
            tokens[batch_index, idx, 0] = pt
            tokens[batch_index, idx, 1] = eta
            tokens[batch_index, idx, 2] = phi
            tokens[batch_index, idx, 3] = energy
            tokens[batch_index, idx, 5] = 1.0
    return tokens, mask


def small_assignment_config(**kwargs):
    return SoftSubjetAssignmentConfig(
        scale_specs=(
            SubjetScaleSpec("small", 2, 0.05, 0.12, "tight"),
            SubjetScaleSpec("medium", 1, 0.12, 0.25, "medium"),
            SubjetScaleSpec("large", 1, 0.25, 0.50, "wide"),
        ),
        embed_dim=16,
        hidden_dim=32,
        **kwargs,
    )


class MultiscaleSubjetPartStep4AssignmentTests(unittest.TestCase):
    def test_seeded_assignment_is_default_and_respects_masks(self):
        tokens, mask = make_assignment_tokens()
        module = SoftSubjetAssignment(small_assignment_config())
        output = module(tokens, mask)

        self.assertEqual(output.query_mode, MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED)
        self.assertIsNotNone(output.seed_output)
        self.assertEqual(tuple(output.assignment_weights.shape), (2, 4, 5))
        self.assertEqual(tuple(output.subjet_mask.shape), (2, 4))
        self.assertEqual(tuple(output.estimated_centers.shape), (2, 4, 2))
        valid_sums = output.assignment_weights.sum(dim=-1)
        self.assertTrue(bool(torch.allclose(valid_sums[output.subjet_mask], torch.ones_like(valid_sums[output.subjet_mask]), atol=1.0e-6)))
        self.assertEqual(float(output.assignment_weights[:, :, 4].abs().sum().item()), 0.0)
        self.assertEqual(output.summary()["contract"], MULTISCALE_SUBJET_ASSIGNMENT_CONTRACT)
        self.assertEqual(output.diagnostics["step"], MULTISCALE_SUBJET_ASSIGNMENT_STEP)
        self.assertTrue(output.diagnostics["geometry_bias_applied"])
        self.assertIn("entropy_mean", output.diagnostics)
        self.assertIn("dead_token_fraction", output.diagnostics)

    def test_assignment_supports_explicit_particle_and_subjet_masks(self):
        tokens, mask = make_assignment_tokens()
        module = SoftSubjetAssignment(small_assignment_config())
        particle_assignment_mask = mask.clone()
        particle_assignment_mask[:, 1:] = False
        subjet_assignment_mask = torch.ones((2, 4), dtype=torch.bool)
        subjet_assignment_mask[:, -1] = False
        output = module(
            tokens,
            mask,
            particle_assignment_mask=particle_assignment_mask,
            subjet_assignment_mask=subjet_assignment_mask,
        )

        self.assertEqual(float(output.assignment_weights[:, :, 1:].sum().item()), 0.0)
        self.assertEqual(float(output.assignment_weights[:, -1, :].sum().item()), 0.0)
        self.assertTrue(bool(torch.allclose(output.assignment_weights[:, :-1, 0], torch.ones_like(output.assignment_weights[:, :-1, 0]))))

    def test_particle_assignment_mask_can_invalidate_whole_batch_row(self):
        tokens, mask = make_assignment_tokens()
        module = SoftSubjetAssignment(small_assignment_config())
        particle_assignment_mask = mask.clone()
        particle_assignment_mask[0, :] = False
        output = module(tokens, mask, particle_assignment_mask=particle_assignment_mask)

        self.assertFalse(bool(output.subjet_mask[0].any()))
        self.assertEqual(float(output.assignment_weights[0].abs().sum().item()), 0.0)
        self.assertEqual(float(output.cluster_weights[0].abs().sum().item()), 0.0)
        self.assertTrue(bool(output.subjet_mask[1].any()))

    def test_pure_learned_queries_are_available_as_control(self):
        tokens, mask = make_assignment_tokens()
        module = SoftSubjetAssignment(small_assignment_config(query_mode=MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED))
        output = module(tokens, mask)

        self.assertEqual(output.query_mode, MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED)
        self.assertIsNone(output.seed_output)
        self.assertFalse(output.diagnostics["geometry_bias_applied"])
        self.assertEqual(tuple(output.cluster_weights.shape), tuple(output.assignment_weights.shape))
        self.assertEqual(int(output.subjet_mask.sum().item()), 8)
        valid_sums = output.assignment_weights.sum(dim=-1)
        self.assertTrue(bool(torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1.0e-6)))

    def test_query_mode_aliases_are_normalized(self):
        tokens, mask = make_assignment_tokens()
        module = SoftSubjetAssignment(small_assignment_config(query_mode="seed-conditioned"))
        seeded_output = module(tokens, mask)
        learned_output = module(tokens, mask, query_mode="pure-learned")

        self.assertEqual(seeded_output.query_mode, MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED)
        self.assertEqual(learned_output.query_mode, MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED)
        self.assertIsNone(learned_output.seed_output)

    def test_custom_seed_method_mapping_survives_assignment_config_normalization(self):
        tokens, mask = make_assignment_tokens()
        scale_specs = (
            SubjetScaleSpec("small", 2, 0.05, 0.12, "tight"),
            SubjetScaleSpec("medium", 1, 0.12, 0.25, "medium"),
        )
        seed_config = SubjetSeedBuilderConfig(
            method_by_scale={
                "small": MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD,
                "medium": MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD,
            }
        )
        config = SoftSubjetAssignmentConfig(scale_specs=scale_specs, seed_config=seed_config, embed_dim=16, hidden_dim=32)
        module = SoftSubjetAssignment(config)
        output = module(tokens, mask)

        self.assertIsNotNone(output.seed_output)
        self.assertEqual(
            output.seed_output.selection_methods,
            (
                MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD,
                MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD,
                MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD,
            ),
        )

    def test_learned_query_empty_jets_have_invalid_zero_assignments(self):
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
        module = SoftSubjetAssignment(small_assignment_config(query_mode=MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED))
        output = module(tokens, mask)

        self.assertFalse(bool(output.subjet_mask[0].any()))
        self.assertEqual(float(output.assignment_weights[0].abs().sum().item()), 0.0)
        self.assertTrue(bool(output.subjet_mask[1].all()))
        self.assertTrue(bool(torch.allclose(output.assignment_weights[1, :, 0], torch.ones_like(output.assignment_weights[1, :, 0]))))

    def test_seed_geometry_bias_can_dominate_when_logits_are_zeroed(self):
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
        config = SoftSubjetAssignmentConfig(
            scale_specs=(SubjetScaleSpec("small", 1, 0.05, 0.12, "tight"),),
            embed_dim=8,
            hidden_dim=16,
            geometry_bias_strength=50.0,
        )
        module = SoftSubjetAssignment(config)
        for parameter in module.parameters():
            torch.nn.init.zeros_(parameter)
        output = module(tokens, mask)

        self.assertGreater(float(output.assignment_weights[0, 0, 0].item()), 0.999)
        self.assertLess(float(output.assignment_weights[0, 0, 1].item()), 1.0e-6)
        self.assertLess(float(output.assignment_weights[0, 0, 2].item()), 1.0e-6)

    def test_assignment_has_parameter_gradients(self):
        tokens, mask = make_assignment_tokens()
        module = SoftSubjetAssignment(small_assignment_config())
        output = module(tokens, mask)
        loss = output.assignment_weights[:, :, 0].sum() + output.estimated_pt_fraction.sum()
        loss.backward()

        self.assertIsNotNone(module.learned_queries.grad)
        self.assertGreater(float(module.learned_queries.grad.abs().sum().item()), 0.0)

    def test_bad_masks_are_rejected(self):
        tokens, mask = make_assignment_tokens()
        module = SoftSubjetAssignment(small_assignment_config())
        with self.assertRaisesRegex(ValueError, "particle_assignment_mask shape"):
            module(tokens, mask, particle_assignment_mask=mask[:, :3])
        with self.assertRaisesRegex(ValueError, "subjet_assignment_mask shape"):
            module(tokens, mask, subjet_assignment_mask=torch.ones((2, 3), dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()

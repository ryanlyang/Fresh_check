import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.multiscale_subjet_part import (
    MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD,
    MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD,
    MULTISCALE_SUBJET_LOCAL_DENSITY_SEED_METHOD,
    MULTISCALE_SUBJET_SEED_BUILDER_STEP,
    MULTISCALE_SUBJET_SEED_CONTRACT,
    SubjetScaleSpec,
    SubjetSeedBuilderConfig,
    build_multiscale_subjet_seeds,
)

torch = require_torch()


def make_tokens(batch_size=1, num_particles=4):
    return torch.zeros((batch_size, num_particles, RAW_TOKEN_DIM), dtype=torch.float32)


class MultiscaleSubjetPartStep3SeedTests(unittest.TestCase):
    def test_leading_pt_seeds_ignore_padded_particles_and_do_not_mark_duplicates_valid(self):
        tokens = make_tokens()
        mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)
        tokens[0, 0, 0] = 10.0
        tokens[0, 0, 1] = 0.1
        tokens[0, 0, 2] = 0.2
        tokens[0, 1, 0] = 20.0
        tokens[0, 1, 1] = 0.3
        tokens[0, 1, 2] = 0.4
        tokens[0, 2, 0] = 999.0
        tokens[0, 2, 1] = 999.0
        tokens[0, 2, 2] = 999.0

        scale = SubjetScaleSpec("small", 4, 0.05, 0.12, "test leading seeds")
        config = SubjetSeedBuilderConfig(
            scale_specs=(scale,),
            method_by_scale={"small": MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD},
        )
        output = build_multiscale_subjet_seeds(tokens, mask, config=config)

        self.assertEqual(output.indices[0].tolist(), [1, 0, -1, -1])
        self.assertEqual(output.mask[0].tolist(), [True, True, False, False])
        self.assertTrue(bool(torch.all(output.centers[0, 2:] == 0.0)))
        self.assertTrue(bool(torch.all(output.seed_tokens[0, 2:] == 0.0)))
        self.assertEqual(output.diagnostics["valid_seed_count_min"], 2)
        self.assertEqual(output.diagnostics["duplicate_valid_seed_fraction"], 0.0)

    def test_empty_jets_return_invalid_finite_seed_outputs(self):
        tokens = make_tokens(batch_size=2, num_particles=3)
        mask = torch.zeros((2, 3), dtype=torch.bool)

        scale = SubjetScaleSpec("small", 3, 0.05, 0.12, "empty test")
        output = build_multiscale_subjet_seeds(tokens, mask, config=SubjetSeedBuilderConfig(scale_specs=(scale,)))

        self.assertEqual(tuple(output.centers.shape), (2, 3, 2))
        self.assertFalse(bool(output.mask.any()))
        self.assertTrue(bool(torch.all(output.indices == -1)))
        self.assertTrue(bool(torch.isfinite(output.centers).all()))
        self.assertTrue(bool(torch.all(output.centers == 0.0)))
        self.assertEqual(output.diagnostics["empty_jet_count"], 2)
        self.assertEqual(output.diagnostics["valid_seed_count_max"], 0)

    def test_farthest_point_seeds_use_wrapped_phi_distances(self):
        tokens = make_tokens(batch_size=1, num_particles=3)
        mask = torch.ones((1, 3), dtype=torch.bool)
        tokens[0, 0, 0] = 100.0
        tokens[0, 0, 1] = 0.0
        tokens[0, 0, 2] = math.pi - 0.01
        tokens[0, 1, 0] = 90.0
        tokens[0, 1, 1] = 0.0
        tokens[0, 1, 2] = -math.pi + 0.01
        tokens[0, 2, 0] = 80.0
        tokens[0, 2, 1] = 0.0
        tokens[0, 2, 2] = 0.0

        scale = SubjetScaleSpec("large", 2, 0.25, 0.50, "farthest point")
        config = SubjetSeedBuilderConfig(
            scale_specs=(scale,),
            method_by_scale={"large": MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD},
        )
        output = build_multiscale_subjet_seeds(tokens, mask, config=config)

        self.assertEqual(output.indices[0].tolist(), [0, 2])
        self.assertEqual(output.selection_methods, (MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD,) * 2)

    def test_local_density_seeds_choose_cluster_not_isolated_particle(self):
        tokens = make_tokens(batch_size=1, num_particles=4)
        mask = torch.ones((1, 4), dtype=torch.bool)
        coords = [(0.00, 0.00), (0.03, 0.00), (0.00, 0.03), (2.00, 2.00)]
        for index, (eta, phi) in enumerate(coords):
            tokens[0, index, 0] = 10.0
            tokens[0, index, 1] = eta
            tokens[0, index, 2] = phi

        scale = SubjetScaleSpec("medium", 2, 0.10, 0.30, "density peaks")
        config = SubjetSeedBuilderConfig(
            scale_specs=(scale,),
            method_by_scale={"medium": MULTISCALE_SUBJET_LOCAL_DENSITY_SEED_METHOD},
        )
        output = build_multiscale_subjet_seeds(tokens, mask, config=config)

        self.assertEqual(output.mask[0].tolist(), [True, True])
        self.assertTrue(set(output.indices[0].tolist()).issubset({0, 1, 2}))
        self.assertNotIn(3, output.indices[0].tolist())
        self.assertEqual(output.diagnostics["method_valid_counts"][MULTISCALE_SUBJET_LOCAL_DENSITY_SEED_METHOD], 2)

    def test_default_multiscale_seed_builder_is_deterministic_and_reports_scale_metadata(self):
        tokens = make_tokens(batch_size=2, num_particles=5)
        mask = torch.tensor(
            [
                [True, True, True, True, False],
                [True, True, False, False, False],
            ],
            dtype=torch.bool,
        )
        for batch in range(2):
            for particle in range(5):
                tokens[batch, particle, 0] = float(10 + batch + particle)
                tokens[batch, particle, 1] = float(0.1 * particle)
                tokens[batch, particle, 2] = float(-0.2 * particle)

        first = build_multiscale_subjet_seeds(tokens, mask)
        second = build_multiscale_subjet_seeds(tokens, mask)

        self.assertTrue(bool(torch.equal(first.indices, second.indices)))
        self.assertTrue(bool(torch.equal(first.mask, second.mask)))
        self.assertTrue(bool(torch.allclose(first.centers, second.centers)))
        self.assertEqual(first.summary()["contract"], MULTISCALE_SUBJET_SEED_CONTRACT)
        self.assertEqual(first.diagnostics["step"], MULTISCALE_SUBJET_SEED_BUILDER_STEP)
        self.assertEqual(first.total_num_seeds, 20)
        self.assertEqual(len(first.scale_names), 20)
        self.assertIn("small", first.scale_names)
        self.assertIn("medium", first.scale_names)
        self.assertIn("large", first.scale_names)
        self.assertIn(MULTISCALE_SUBJET_LEADING_PT_SEED_METHOD, first.selection_methods)
        self.assertIn(MULTISCALE_SUBJET_LOCAL_DENSITY_SEED_METHOD, first.selection_methods)
        self.assertIn(MULTISCALE_SUBJET_FARTHEST_POINT_SEED_METHOD, first.selection_methods)

    def test_config_rejects_bad_scale_method_mapping(self):
        scale = SubjetScaleSpec("small", 1, 0.05, 0.12, "validation")
        with self.assertRaisesRegex(ValueError, "unknown scale"):
            SubjetSeedBuilderConfig(scale_specs=(scale,), method_by_scale={"large": "leading_pt"})
        with self.assertRaisesRegex(ValueError, "unknown seed selection method"):
            SubjetSeedBuilderConfig(scale_specs=(scale,), method_by_scale={"small": "mystery"})


if __name__ == "__main__":
    unittest.main()

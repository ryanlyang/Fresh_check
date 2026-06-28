import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES, PF_POINT_NAMES, PF_VECTOR_NAMES

from teacher_logit_reco.multiscale_subjet_part import (
    CANONICAL_PART_FEATURE_NAMES,
    CANONICAL_PART_POINT_NAMES,
    CANONICAL_PART_VECTOR_NAMES,
    MULTISCALE_SUBJET_DEFAULT_SCALE_SPECS,
    CanonicalPartInputs,
    MultiscaleSubjetFeatureConfig,
    build_canonical_part_inputs,
    build_prepared_subjet_inputs,
    default_subjet_scale_specs,
    eta_phi_coordinates,
    local_density_features,
    pairwise_delta_r,
    particle_pt_fraction,
    prepare_subjet_tokens_and_mask,
    scale_radius_bounds,
    wrap_delta_phi,
    wrap_phi,
)

torch = require_torch()


def make_tokens():
    tokens = torch.zeros((2, 4, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )
    # Batch 0: particles 0 and 1 are close across the phi boundary.
    tokens[0, 0, 0] = 10.0
    tokens[0, 0, 1] = 0.00
    tokens[0, 0, 2] = math.pi - 0.02
    tokens[0, 0, 3] = 11.0
    tokens[0, 1, 0] = 20.0
    tokens[0, 1, 1] = 0.01
    tokens[0, 1, 2] = -math.pi + 0.03
    tokens[0, 1, 3] = 21.0
    tokens[0, 2, 0] = 30.0
    tokens[0, 2, 1] = 0.50
    tokens[0, 2, 2] = 0.50
    tokens[0, 2, 3] = 35.0
    tokens[0, 3, 0] = 999.0
    tokens[0, 3, 1] = 999.0
    tokens[0, 3, 2] = 999.0
    tokens[0, 3, 3] = 999.0

    tokens[1, 0, 0] = 5.0
    tokens[1, 0, 1] = -0.20
    tokens[1, 0, 2] = 0.10
    tokens[1, 0, 3] = 5.5
    tokens[1, 1, 0] = 15.0
    tokens[1, 1, 1] = -0.25
    tokens[1, 1, 2] = 0.12
    tokens[1, 1, 3] = 16.0
    return tokens, mask


class MultiscaleSubjetPartStep2FeatureTests(unittest.TestCase):
    def test_prepare_validates_raw_dim_and_zeros_masked_tokens(self):
        tokens, mask = make_tokens()
        prepared, prepared_mask, config = prepare_subjet_tokens_and_mask(tokens, mask)

        self.assertEqual(config.raw_token_dim, RAW_TOKEN_DIM)
        self.assertTrue(bool(torch.equal(prepared_mask, mask)))
        self.assertTrue(bool(torch.all(prepared[0, 3] == 0.0)))
        with self.assertRaisesRegex(ValueError, "raw_token_dim"):
            prepare_subjet_tokens_and_mask(tokens[:, :, :13], mask)
        with self.assertRaisesRegex(ValueError, "mask shape"):
            prepare_subjet_tokens_and_mask(tokens, mask[:, :3])

    def test_phi_wrapping_handles_boundary(self):
        delta = torch.tensor([2.0 * math.pi - 0.05, -2.0 * math.pi + 0.05, 0.25], dtype=torch.float32)
        wrapped = wrap_delta_phi(delta)
        self.assertTrue(bool(torch.all(torch.abs(wrapped[:2]) < 0.06)))
        self.assertAlmostEqual(float(wrapped[2]), 0.25, places=6)

        phi = wrap_phi(torch.tensor([3.5 * math.pi], dtype=torch.float32))
        self.assertGreaterEqual(float(phi.item()), -math.pi)
        self.assertLessEqual(float(phi.item()), math.pi)

    def test_eta_phi_coordinates_and_pairwise_delta_r_respect_phi_wrap(self):
        tokens, mask = make_tokens()
        coords = eta_phi_coordinates(tokens, mask)
        distances = pairwise_delta_r(coords)

        self.assertEqual(tuple(coords.shape), (2, 4, 2))
        self.assertEqual(tuple(distances.shape), (2, 4, 4))
        self.assertTrue(bool(torch.allclose(distances, distances.transpose(1, 2), atol=1.0e-6)))
        self.assertTrue(bool(torch.allclose(torch.diagonal(distances, dim1=1, dim2=2), torch.zeros((2, 4)))))
        self.assertLess(float(distances[0, 0, 1].item()), 0.06)
        self.assertEqual(float(coords[0, 3].abs().sum().item()), 0.0)

    def test_particle_pt_fraction_sums_to_one_over_valid_particles(self):
        tokens, mask = make_tokens()
        pt_fraction = particle_pt_fraction(tokens, mask)

        self.assertEqual(tuple(pt_fraction.shape), (2, 4))
        self.assertTrue(bool(torch.allclose(pt_fraction[0, :3].sum(), torch.tensor(1.0))))
        self.assertTrue(bool(torch.allclose(pt_fraction[1, :2].sum(), torch.tensor(1.0))))
        self.assertEqual(float(pt_fraction[0, 3].item()), 0.0)
        self.assertAlmostEqual(float(pt_fraction[0, 2].item()), 0.5, places=6)

    def test_build_prepared_inputs_contains_four_vector_helpers(self):
        tokens, mask = make_tokens()
        prepared = build_prepared_subjet_inputs(tokens, mask)

        self.assertEqual(tuple(prepared.pt.shape), (2, 4))
        self.assertEqual(tuple(prepared.coordinates.shape), (2, 4, 2))
        self.assertTrue(bool(torch.allclose(prepared.px, prepared.pt * torch.cos(prepared.phi))))
        self.assertTrue(bool(torch.allclose(prepared.py, prepared.pt * torch.sin(prepared.phi))))
        self.assertEqual(float(prepared.energy[0, 3].item()), 0.0)
        summary = prepared.summary()
        self.assertEqual(summary["contract"], "multiscale_subjet_feature_helpers_v1")
        self.assertEqual(summary["raw_token_dim"], RAW_TOKEN_DIM)

    def test_prepared_inputs_apply_physical_energy_floor_like_part_inputs(self):
        tokens, mask = make_tokens()
        tokens[0, 0, 0] = 20.0
        tokens[0, 0, 1] = 2.0
        tokens[0, 0, 3] = 1.0
        prepared = build_prepared_subjet_inputs(tokens, mask)

        expected_floor = 20.0 * math.cosh(2.0) + 1.0e-4
        self.assertAlmostEqual(float(prepared.energy[0, 0].item()), expected_floor, places=4)
        self.assertAlmostEqual(float(prepared.tokens[0, 0, 3].item()), expected_floor, places=4)

    def test_local_density_counts_neighbors_and_pt_fraction_inside_radii(self):
        tokens, mask = make_tokens()
        density = local_density_features(tokens, mask, radii=(0.06, 3.00), include_self=False)

        self.assertEqual(tuple(density.counts.shape), (2, 4, 2))
        self.assertEqual(tuple(density.pt_fraction_sums.shape), (2, 4, 2))
        self.assertEqual(float(density.counts[0, 0, 0].item()), 1.0)
        self.assertEqual(float(density.counts[0, 0, 1].item()), 2.0)
        self.assertAlmostEqual(float(density.pt_fraction_sums[0, 0, 0].item()), 20.0 / 60.0, places=6)
        self.assertEqual(float(density.counts[0, 3].sum().item()), 0.0)

    def test_canonical_part_inputs_match_particle_transformer_contract(self):
        tokens, mask = make_tokens()
        canonical = build_canonical_part_inputs(tokens, mask, max_constits=4)

        self.assertIsInstance(canonical, CanonicalPartInputs)
        self.assertEqual(canonical.feature_names, tuple(PF_FEATURE_NAMES))
        self.assertEqual(canonical.point_names, tuple(PF_POINT_NAMES))
        self.assertEqual(canonical.vector_names, tuple(PF_VECTOR_NAMES))
        self.assertEqual(CANONICAL_PART_FEATURE_NAMES, tuple(PF_FEATURE_NAMES))
        self.assertEqual(CANONICAL_PART_POINT_NAMES, tuple(PF_POINT_NAMES))
        self.assertEqual(CANONICAL_PART_VECTOR_NAMES, tuple(PF_VECTOR_NAMES))
        self.assertEqual(tuple(canonical.points.shape), (2, len(PF_POINT_NAMES), 4))
        self.assertEqual(tuple(canonical.features.shape), (2, len(PF_FEATURE_NAMES), 4))
        self.assertEqual(tuple(canonical.lorentz_vectors.shape), (2, len(PF_VECTOR_NAMES), 4))
        self.assertEqual(tuple(canonical.mask.shape), (2, 1, 4))
        self.assertEqual(tuple(canonical.feature_rows().shape), (2, 4, len(PF_FEATURE_NAMES)))
        self.assertTrue(bool(torch.equal(canonical.as_part_kwargs()["features"], canonical.features)))
        self.assertEqual(float(canonical.features[0, :, 3].abs().sum().item()), 0.0)
        self.assertEqual(float(canonical.lorentz_vectors[0, :, 3].abs().sum().item()), 0.0)
        summary = canonical.summary()
        self.assertEqual(summary["contract"], "multiscale_subjet_canonical_part_inputs_v1")
        self.assertEqual(summary["valid_particle_count"], int(mask.sum().item()))

    def test_canonical_part_inputs_support_weighted_views_and_thresholding(self):
        tokens, mask = make_tokens()
        weights = torch.ones_like(mask, dtype=torch.float32)
        weights[0, 2] = 0.0
        weights[1, 0] = 0.01
        canonical = build_canonical_part_inputs(tokens, mask, weights=weights, max_constits=4, weight_threshold=0.05)

        self.assertEqual(int(canonical.mask.sum().item()), 3)
        self.assertEqual(int(canonical.mask[0].sum().item()), 2)
        self.assertEqual(int(canonical.mask[1].sum().item()), 1)
        self.assertEqual(float(canonical.mask[0, 0, 3].item()), 0.0)
        self.assertEqual(float(canonical.features[0, :, ~canonical.mask[0, 0]].abs().sum().item()), 0.0)
        self.assertEqual(float(canonical.features[1, :, ~canonical.mask[1, 0]].abs().sum().item()), 0.0)
        with self.assertRaisesRegex(ValueError, "weights shape"):
            build_canonical_part_inputs(tokens, mask, weights=weights[:, :3])
        with self.assertRaisesRegex(ValueError, "weight_threshold"):
            build_canonical_part_inputs(tokens, mask, weight_threshold=-0.1)

    def test_default_scale_metadata_is_ordered_and_named(self):
        specs = default_subjet_scale_specs()
        self.assertEqual(specs, MULTISCALE_SUBJET_DEFAULT_SCALE_SPECS)
        self.assertEqual(tuple(spec.name for spec in specs), ("small", "medium", "large"))
        self.assertEqual(tuple(spec.num_tokens for spec in specs), (8, 8, 4))
        bounds = scale_radius_bounds(specs)
        self.assertEqual(bounds["small"], (0.05, 0.12))
        self.assertEqual(bounds["large"], (0.25, 0.50))
        self.assertLess(specs[0].radius_center, specs[-1].radius_center)

    def test_config_rejects_bad_density_radii(self):
        with self.assertRaisesRegex(ValueError, "sorted"):
            MultiscaleSubjetFeatureConfig(default_density_radii=(0.2, 0.1))
        with self.assertRaisesRegex(ValueError, "positive"):
            MultiscaleSubjetFeatureConfig(default_density_radii=(0.0,))


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from jetclass_fresh.dual_view import build_part_inputs_torch
from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES, PF_POINT_NAMES, PF_VECTOR_NAMES

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES,
    LOCAL_COMPRESSION_FEATURES_CONTRACT,
    LocalCompressionCanonicalInputs,
    LocalCompressionFeatureConfig,
    build_local_compression_canonical_inputs,
    prepare_local_compression_tokens_and_mask,
)


torch = require_torch()


def make_tokens(num_particles: int = 5):
    tokens = torch.zeros((2, num_particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((2, num_particles), dtype=torch.bool)
    mask[0, : min(num_particles, 4)] = True
    mask[1, : min(num_particles, 3)] = True
    for batch in range(2):
        for idx in range(num_particles):
            pt = float(5.0 + 10.0 * idx + 3.0 * batch)
            eta = -0.4 + 0.1 * idx + 0.05 * batch
            phi = -math.pi + 0.2 * idx
            tokens[batch, idx, 0] = pt
            tokens[batch, idx, 1] = eta
            tokens[batch, idx, 2] = phi
            tokens[batch, idx, 3] = pt * math.cosh(eta) + 0.5
            tokens[batch, idx, 4] = -1.0 if idx % 2 else 1.0
            tokens[batch, idx, 5 + (idx % 5)] = 1.0
            tokens[batch, idx, 10] = 0.01 * idx
            tokens[batch, idx, 11] = 0.1 + 0.01 * idx
            tokens[batch, idx, 12] = -0.02 * idx
            tokens[batch, idx, 13] = 0.2 + 0.01 * idx
    return tokens, mask


class LocalCompressionPartStep2FeatureTests(unittest.TestCase):
    def test_prepare_validates_raw_dim_and_zeros_invalid_particles(self):
        tokens, mask = make_tokens()
        prepared = prepare_local_compression_tokens_and_mask(tokens, mask, max_constits=5)

        self.assertEqual(prepared.raw_token_dim, RAW_TOKEN_DIM)
        self.assertEqual(tuple(prepared.tokens.shape), (2, 5, RAW_TOKEN_DIM))
        self.assertTrue(bool(torch.equal(prepared.mask, mask)))
        self.assertEqual(float(prepared.tokens[0, 4].abs().sum().item()), 0.0)
        self.assertEqual(prepared.summary()["valid_particle_count"], int(mask.sum().item()))

        with self.assertRaisesRegex(ValueError, "raw_token_dim"):
            prepare_local_compression_tokens_and_mask(tokens[:, :, :13], mask)
        with self.assertRaisesRegex(ValueError, "mask shape"):
            prepare_local_compression_tokens_and_mask(tokens, mask[:, :3])

    def test_prepare_topk_keeps_tokens_aligned_by_pt(self):
        tokens, mask = make_tokens(num_particles=6)
        prepared = prepare_local_compression_tokens_and_mask(tokens, mask, max_constits=3)

        self.assertEqual(tuple(prepared.tokens.shape), (2, 3, RAW_TOKEN_DIM))
        self.assertTrue(bool(torch.all(prepared.tokens[0, :, 0].diff() <= 0.0)))
        self.assertTrue(bool(torch.all(prepared.tokens[1, :, 0].diff() <= 0.0)))
        self.assertEqual(int(prepared.mask[0].sum().item()), 3)
        self.assertEqual(int(prepared.mask[1].sum().item()), 3)

    def test_prepare_applies_weight_threshold_without_mutating_raw_tokens(self):
        tokens, mask = make_tokens()
        weights = torch.ones_like(mask, dtype=torch.float32)
        weights[0, 0] = 0.0
        weights[0, 1] = 0.5
        weights[1, 0] = 0.01
        prepared = prepare_local_compression_tokens_and_mask(
            tokens,
            mask,
            weights=weights,
            max_constits=5,
            weight_threshold=0.05,
        )

        self.assertEqual(int(prepared.mask.sum().item()), int(mask.sum().item()) - 2)
        self.assertEqual(float(prepared.tokens[0, 0].abs().sum().item()), 0.0)
        self.assertAlmostEqual(float(prepared.tokens[0, 1, 0].item()), float(tokens[0, 1, 0].item()), places=6)
        self.assertAlmostEqual(float(prepared.weights[0, 1].item()), 0.5, places=6)
        with self.assertRaisesRegex(ValueError, "weights shape"):
            prepare_local_compression_tokens_and_mask(tokens, mask, weights=weights[:, :3])

    def test_prepare_preserves_original_finite_quality_metadata(self):
        tokens, mask = make_tokens()
        tokens[0, 1, 11] = float("nan")
        prepared = prepare_local_compression_tokens_and_mask(tokens, mask, max_constits=5)

        self.assertFalse(bool(prepared.original_all_finite[0, 1].item()))
        self.assertFalse(bool(prepared.mask[0, 1].item()))
        self.assertEqual(prepared.summary()["original_nonfinite_particle_count"], 1)

    def test_canonical_inputs_match_exact_part_builder_for_unweighted_tokens(self):
        tokens, mask = make_tokens()
        canonical = build_local_compression_canonical_inputs(tokens, mask, max_constits=5)
        part_inputs = build_part_inputs_torch(tokens, mask, max_constits=5)

        self.assertIsInstance(canonical, LocalCompressionCanonicalInputs)
        self.assertEqual(canonical.feature_names, tuple(PF_FEATURE_NAMES))
        self.assertEqual(canonical.point_names, tuple(PF_POINT_NAMES))
        self.assertEqual(canonical.vector_names, tuple(PF_VECTOR_NAMES))
        self.assertEqual(canonical.feature_names, LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES)
        self.assertEqual(tuple(canonical.points.shape), tuple(part_inputs["points"].shape))
        self.assertEqual(tuple(canonical.features.shape), tuple(part_inputs["features"].shape))
        self.assertEqual(tuple(canonical.lorentz_vectors.shape), tuple(part_inputs["lorentz_vectors"].shape))
        self.assertTrue(bool(torch.allclose(canonical.points, part_inputs["points"])))
        self.assertTrue(bool(torch.allclose(canonical.features, part_inputs["features"])))
        self.assertTrue(bool(torch.allclose(canonical.lorentz_vectors, part_inputs["lorentz_vectors"])))
        self.assertTrue(bool(torch.equal(canonical.mask, part_inputs["mask"])))
        self.assertEqual(tuple(canonical.feature_rows().shape), (2, 5, len(PF_FEATURE_NAMES)))
        self.assertEqual(tuple(canonical.vector_rows().shape), (2, 5, len(PF_VECTOR_NAMES)))
        self.assertEqual(tuple(canonical.point_rows().shape), (2, 5, len(PF_POINT_NAMES)))

    def test_canonical_inputs_keep_selected_tokens_aligned_with_topk(self):
        tokens, mask = make_tokens(num_particles=7)
        canonical = build_local_compression_canonical_inputs(tokens, mask, max_constits=3)

        self.assertEqual(tuple(canonical.selected_tokens.shape), (2, 3, RAW_TOKEN_DIM))
        self.assertTrue(bool(torch.equal(canonical.particle_mask, canonical.mask[:, 0, :])))
        self.assertTrue(bool(torch.allclose(canonical.selected_tokens[0, :, 0], torch.tensor([35.0, 25.0, 15.0]))))
        self.assertEqual(canonical.summary()["contract"], LOCAL_COMPRESSION_FEATURES_CONTRACT)
        self.assertEqual(canonical.summary()["valid_particle_count"], int(canonical.particle_mask.sum().item()))

    def test_canonical_with_features_masks_invalid_particles(self):
        tokens, mask = make_tokens()
        canonical = build_local_compression_canonical_inputs(tokens, mask, max_constits=5)
        rows = canonical.feature_rows().clone()
        rows[0, 4] = 123.0
        adapted = canonical.with_features(rows)

        self.assertEqual(float(adapted.features[0, :, 4].abs().sum().item()), 0.0)
        self.assertTrue(bool(torch.allclose(adapted.features[0, :, :4], rows[0, :4].transpose(0, 1))))
        with self.assertRaisesRegex(ValueError, "feature_rows shape"):
            canonical.with_features(rows[:, :, :3])

    def test_config_rejects_noncanonical_feature_contract(self):
        with self.assertRaisesRegex(ValueError, "PF_FEATURE_NAMES"):
            LocalCompressionFeatureConfig(canonical_feature_names=("bad",))


if __name__ == "__main__":
    unittest.main()

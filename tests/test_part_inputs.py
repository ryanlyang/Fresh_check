import unittest

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView
from jetclass_fresh.part_inputs import (
    JET_FEATURE_NAMES,
    PF_FEATURE_NAMES,
    PF_POINT_NAMES,
    PF_VECTOR_NAMES,
    build_particle_transformer_inputs,
    summarize_particle_transformer_inputs,
    wrap_phi,
)


def make_view(tokens, mask, *, view_name):
    n_jets = tokens.shape[0]
    return JetView(
        tokens=tokens.astype(np.float32),
        mask=mask.astype(bool),
        labels=np.arange(n_jets, dtype=np.int64) % 10,
        jet_ids=[JetIdentity(file=f"{view_name}_{idx}.root", entry=idx, label=idx % 10) for idx in range(n_jets)],
        split="model_val",
        metadata={
            "view": view_name,
            "offline_jet_pt": 999999.0,
            "offline_jet_eta": -999999.0,
        },
    )


def base_tokens():
    tokens = np.zeros((2, 4, 14), dtype=np.float32)
    mask = np.zeros((2, 4), dtype=bool)
    mask[:, :3] = True

    values = [
        [(20.0, 0.10, 0.00), (10.0, 0.20, 0.35), (5.0, -0.10, -0.40)],
        [(12.0, -0.30, 1.00), (8.0, -0.10, 1.30), (4.0, -0.50, 0.70)],
    ]
    for jet_index, particles in enumerate(values):
        for part_index, (pt, eta, phi) in enumerate(particles):
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta) + 0.2
            tokens[jet_index, part_index, 4] = 1.0 if part_index == 0 else 0.0
            tokens[jet_index, part_index, 5 + part_index] = 1.0
            tokens[jet_index, part_index, 10] = 0.2 * (part_index + 1)
            tokens[jet_index, part_index, 11] = 0.05 * (part_index + 1)
            tokens[jet_index, part_index, 12] = -0.1 * (part_index + 1)
            tokens[jet_index, part_index, 13] = 0.03 * (part_index + 1)
    return tokens, mask


class ParticleTransformerInputStep4Tests(unittest.TestCase):
    def test_builder_matches_reference_shapes_and_names(self):
        tokens, mask = base_tokens()
        inputs = build_particle_transformer_inputs(make_view(tokens, mask, view_name="offline"))

        self.assertEqual(inputs.pf_points.shape, (2, 2, 4))
        self.assertEqual(inputs.pf_features.shape, (2, 17, 4))
        self.assertEqual(inputs.pf_vectors.shape, (2, 4, 4))
        self.assertEqual(inputs.pf_mask.shape, (2, 1, 4))
        self.assertEqual(inputs.metadata["pf_point_names"], PF_POINT_NAMES)
        self.assertEqual(inputs.metadata["pf_feature_names"], PF_FEATURE_NAMES)
        self.assertEqual(inputs.metadata["pf_vector_names"], PF_VECTOR_NAMES)
        self.assertFalse(summarize_particle_transformer_inputs(inputs)["has_nan"])

    def test_hlt_view_uses_hlt_axis_not_offline_metadata(self):
        offline_tokens, mask = base_tokens()
        hlt_tokens = offline_tokens.copy()
        hlt_tokens[:, :, 0] *= 0.60
        hlt_tokens[:, :, 1] += 0.35
        hlt_tokens[:, :, 2] = wrap_phi(hlt_tokens[:, :, 2] - 0.25)
        hlt_tokens[:, :, 3] = hlt_tokens[:, :, 0] * np.cosh(hlt_tokens[:, :, 1]) + 0.1

        offline_inputs = build_particle_transformer_inputs(make_view(offline_tokens, mask, view_name="offline"))
        hlt_inputs = build_particle_transformer_inputs(make_view(hlt_tokens, mask, view_name="fixed_hlt"))

        jet_pt_index = JET_FEATURE_NAMES.index("jet_pt")
        self.assertNotAlmostEqual(
            float(offline_inputs.jet_features[0, jet_pt_index]),
            float(hlt_inputs.jet_features[0, jet_pt_index]),
        )
        expected_hlt_px = np.sum(hlt_tokens[0, :3, 0] * np.cos(hlt_tokens[0, :3, 2]))
        expected_hlt_py = np.sum(hlt_tokens[0, :3, 0] * np.sin(hlt_tokens[0, :3, 2]))
        expected_hlt_pt = np.hypot(expected_hlt_px, expected_hlt_py)
        self.assertAlmostEqual(float(hlt_inputs.jet_features[0, jet_pt_index]), float(expected_hlt_pt), places=5)

    def test_dummy_reconstructed_view_can_fold_candidate_weights(self):
        tokens, mask = base_tokens()
        reco_view = make_view(tokens[:1], mask[:1], view_name="reconstructed")
        weights = np.array([[1.0, 0.50, 0.0, 1.0]], dtype=np.float32)
        inputs = build_particle_transformer_inputs(reco_view, candidate_weights=weights)

        px_index = PF_VECTOR_NAMES.index("part_px")
        original_second_px = tokens[0, 1, 0] * np.cos(tokens[0, 1, 2])
        self.assertAlmostEqual(float(inputs.pf_vectors[0, px_index, 1]), float(0.5 * original_second_px), places=5)
        self.assertFalse(bool(inputs.pf_mask[0, 0, 2]))
        self.assertTrue(inputs.metadata["candidate_weights_folded"])

    def test_padding_positions_are_zeroed(self):
        tokens, mask = base_tokens()
        inputs = build_particle_transformer_inputs(make_view(tokens, mask, view_name="offline"))
        self.assertTrue(np.all(inputs.pf_points[:, :, 3] == 0.0))
        self.assertTrue(np.all(inputs.pf_features[:, :, 3] == 0.0))
        self.assertTrue(np.all(inputs.pf_vectors[:, :, 3] == 0.0))


if __name__ == "__main__":
    unittest.main()

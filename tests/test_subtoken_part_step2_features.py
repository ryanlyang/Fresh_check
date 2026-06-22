import unittest

import numpy as np

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES, build_particle_transformer_inputs_from_tokens
from jetclass_fresh.hlt_baseline import require_torch

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_DERIVED_KINEMATIC_FEATURE_NAMES,
    SUBTOKEN_DERIVED_TRACK_FEATURE_NAMES,
    SUBTOKEN_MODALITY_IDENTITY,
    SUBTOKEN_MODALITY_KINEMATICS,
    SUBTOKEN_MODALITY_TRACK,
    SubtokenFeatureConfig,
    build_derived_kinematics,
    build_subtoken_inputs,
    split_raw_tokens_into_modalities,
)


class SubtokenPartStep2FeatureTests(unittest.TestCase):
    def make_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((2, 4, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, False, True],
                [True, False, True, False],
            ],
            dtype=torch.bool,
        )

        tokens[:, :, 0] = torch.tensor(
            [
                [50.0, 20.0, 999.0, 8.0],
                [30.0, 777.0, 12.0, 999.0],
            ],
            dtype=torch.float32,
        )
        tokens[:, :, 1] = torch.tensor(
            [
                [0.2, -0.5, 99.0, 1.0],
                [-0.3, 42.0, 0.7, -99.0],
            ],
            dtype=torch.float32,
        )
        tokens[:, :, 2] = torch.tensor(
            [
                [0.1, 2.9, -88.0, -2.7],
                [1.5, 88.0, -2.1, 13.0],
            ],
            dtype=torch.float32,
        )
        tokens[:, :, 3] = torch.tensor(
            [
                [55.0, 24.0, 999.0, 13.0],
                [35.0, 777.0, 15.0, 999.0],
            ],
            dtype=torch.float32,
        )
        tokens[:, :, 4] = torch.tensor(
            [
                [1.0, -1.0, 3.0, 0.0],
                [0.0, -2.0, 1.0, 5.0],
            ],
            dtype=torch.float32,
        )
        tokens[:, :, 5:10] = torch.tensor(
            [
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [9, 9, 9, 9, 9], [0, 0, 1, 0, 0]],
                [[0, 0, 0, 1, 0], [8, 8, 8, 8, 8], [0, 0, 0, 0, 1], [7, 7, 7, 7, 7]],
            ],
            dtype=torch.float32,
        )
        tokens[:, :, 10] = torch.tensor([[0.2, -0.3, 99.0, 0.4], [0.5, 8.0, -0.2, 7.0]])
        tokens[:, :, 11] = torch.tensor([[0.1, 0.2, 99.0, 0.3], [0.4, 8.0, 0.5, 7.0]])
        tokens[:, :, 12] = torch.tensor([[-0.1, 0.2, 99.0, -0.4], [0.7, 8.0, 0.3, 7.0]])
        tokens[:, :, 13] = torch.tensor([[0.6, 0.7, 99.0, 0.8], [0.9, 8.0, 1.0, 7.0]])
        return tokens, mask

    def test_split_raw_tokens_into_modalities_shapes_and_masks(self):
        tokens, mask = self.make_tokens()

        grouped = split_raw_tokens_into_modalities(tokens, mask)

        self.assertEqual(tuple(grouped.kin_values.shape), (2, 4, 4))
        self.assertEqual(tuple(grouped.id_values.shape), (2, 4, 6))
        self.assertEqual(tuple(grouped.track_values.shape), (2, 4, 4))
        self.assertEqual(grouped.modality_feature_names[SUBTOKEN_MODALITY_KINEMATICS], ("pt", "eta", "phi", "energy"))
        self.assertEqual(
            grouped.modality_feature_names[SUBTOKEN_MODALITY_IDENTITY],
            ("charge", "isChargedHadron", "isNeutralHadron", "isPhoton", "isElectron", "isMuon"),
        )
        self.assertEqual(grouped.modality_feature_names[SUBTOKEN_MODALITY_TRACK], ("d0", "d0err", "dz", "dzerr"))
        self.assertTrue(bool((grouped.kin_values[~mask] == 0.0).all()))
        self.assertTrue(bool((grouped.id_values[~mask] == 0.0).all()))
        self.assertTrue(bool((grouped.track_values[~mask] == 0.0).all()))

    def test_split_rejects_non_14_dim_raw_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((1, 2, RAW_TOKEN_DIM + 1), dtype=torch.float32)
        mask = torch.ones((1, 2), dtype=torch.bool)

        with self.assertRaises(ValueError):
            split_raw_tokens_into_modalities(tokens, mask)

    def test_derived_kinematics_matches_particle_transformer_builder(self):
        tokens, mask = self.make_tokens()

        derived = build_derived_kinematics(tokens, mask)
        expected = build_particle_transformer_inputs_from_tokens(
            tokens.detach().cpu().numpy(),
            mask.detach().cpu().numpy(),
        )

        self.assertEqual(tuple(derived.part_features.shape), (2, 4, len(PF_FEATURE_NAMES)))
        np.testing.assert_allclose(
            derived.part_features.detach().cpu().numpy(),
            np.transpose(expected.pf_features, (0, 2, 1)),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            derived.jet_features.detach().cpu().numpy(),
            expected.jet_features,
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    def test_build_subtoken_inputs_appends_part_style_derived_features(self):
        tokens, mask = self.make_tokens()

        inputs = build_subtoken_inputs(tokens, mask)

        self.assertEqual(tuple(inputs.kin_values.shape), (2, 4, 4 + len(SUBTOKEN_DERIVED_KINEMATIC_FEATURE_NAMES)))
        self.assertEqual(tuple(inputs.id_values.shape), (2, 4, 6))
        self.assertEqual(tuple(inputs.track_values.shape), (2, 4, 4 + len(SUBTOKEN_DERIVED_TRACK_FEATURE_NAMES)))
        self.assertEqual(
            inputs.modality_feature_names[SUBTOKEN_MODALITY_KINEMATICS][-len(SUBTOKEN_DERIVED_KINEMATIC_FEATURE_NAMES) :],
            SUBTOKEN_DERIVED_KINEMATIC_FEATURE_NAMES,
        )
        self.assertEqual(
            inputs.modality_feature_names[SUBTOKEN_MODALITY_TRACK][-len(SUBTOKEN_DERIVED_TRACK_FEATURE_NAMES) :],
            SUBTOKEN_DERIVED_TRACK_FEATURE_NAMES,
        )
        self.assertIsNotNone(inputs.derived_kinematics)
        self.assertTrue(bool(require_torch().isfinite(inputs.kin_values).all()))
        self.assertTrue(bool(require_torch().isfinite(inputs.track_values).all()))
        self.assertTrue(bool((inputs.kin_values[~mask] == 0.0).all()))
        self.assertTrue(bool((inputs.track_values[~mask] == 0.0).all()))

    def test_build_subtoken_inputs_can_disable_part_style_derived_features(self):
        tokens, mask = self.make_tokens()
        config = SubtokenFeatureConfig(include_part_style_derived_features=False)

        inputs = build_subtoken_inputs(tokens, mask, config=config)

        self.assertEqual(tuple(inputs.kin_values.shape), (2, 4, 4))
        self.assertEqual(tuple(inputs.track_values.shape), (2, 4, 4))
        self.assertIsNone(inputs.derived_kinematics)

    def test_masked_nonfinite_values_are_sanitized_and_zeroed(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        tokens[0, 2, 0] = float("nan")
        tokens[1, 1, 3] = float("inf")

        inputs = build_subtoken_inputs(tokens, mask)

        self.assertTrue(bool(torch.isfinite(inputs.raw_tokens).all()))
        self.assertTrue(bool((inputs.raw_tokens[~mask] == 0.0).all()))


if __name__ == "__main__":
    unittest.main()

import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_MODALITY_IDENTITY,
    SUBTOKEN_MODALITY_KINEMATICS,
    SUBTOKEN_MODALITY_TRACK,
    SUBTOKEN_PART_ENCODER_CONTRACT,
    IdentityEncoder,
    KinematicsEncoder,
    ParticleAnchorEncoder,
    SubtokenFeatureConfig,
    SubtokenPartConfig,
    SubtokenParticleEncoder,
    TrackEncoder,
    build_pt_rank_features,
    subtoken_anchor_input_dim,
    subtoken_modality_input_dims,
)


class SubtokenPartStep3EncoderTests(unittest.TestCase):
    def make_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, False, True, True],
                [True, False, True, False, True],
            ],
            dtype=torch.bool,
        )
        tokens[:, :, 0] = torch.tensor(
            [
                [50.0, 20.0, 900.0, 8.0, 4.0],
                [30.0, 800.0, 12.0, 700.0, 6.0],
            ]
        )
        tokens[:, :, 1] = torch.tensor(
            [
                [0.2, -0.5, 9.0, 1.0, -1.2],
                [-0.3, 8.0, 0.7, -7.0, 0.4],
            ]
        )
        tokens[:, :, 2] = torch.tensor(
            [
                [0.1, 2.9, -8.0, -2.7, 0.8],
                [1.5, 8.0, -2.1, 7.0, -0.6],
            ]
        )
        tokens[:, :, 3] = tokens[:, :, 0] + 10.0
        tokens[:, :, 4] = torch.tensor(
            [
                [1.0, -1.0, 3.0, 0.0, 1.0],
                [0.0, -2.0, 1.0, 5.0, -1.0],
            ]
        )
        tokens[:, :, 5:10] = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [9, 9, 9, 9, 9],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                ],
                [
                    [0, 0, 0, 1, 0],
                    [8, 8, 8, 8, 8],
                    [0, 0, 0, 0, 1],
                    [7, 7, 7, 7, 7],
                    [1, 0, 0, 0, 0],
                ],
            ],
            dtype=torch.float32,
        )
        tokens[:, :, 10] = torch.tensor([[0.2, -0.3, 99.0, 0.4, -0.1], [0.5, 8.0, -0.2, 7.0, 0.2]])
        tokens[:, :, 11] = torch.tensor([[0.1, 0.2, 99.0, 0.3, 0.4], [0.4, 8.0, 0.5, 7.0, 0.6]])
        tokens[:, :, 12] = torch.tensor([[-0.1, 0.2, 99.0, -0.4, 0.3], [0.7, 8.0, 0.3, 7.0, -0.2]])
        tokens[:, :, 13] = torch.tensor([[0.6, 0.7, 99.0, 0.8, 0.9], [0.9, 8.0, 1.0, 7.0, 0.1]])
        return tokens, mask

    def make_config(self, **kwargs):
        defaults = {
            "num_classes": 2,
            "embed_dim": 16,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        }
        defaults.update(kwargs)
        return SubtokenPartConfig(**defaults)

    def test_individual_modality_encoders_map_to_embed_dim(self):
        torch = require_torch()
        kin = KinematicsEncoder(input_dim=11, embed_dim=16, dropout=0.0)
        identity = IdentityEncoder(input_dim=6, embed_dim=16, dropout=0.0)
        track = TrackEncoder(input_dim=4, embed_dim=16, dropout=0.0)
        anchor = ParticleAnchorEncoder(input_dim=RAW_TOKEN_DIM, embed_dim=16, dropout=0.0)
        mask = torch.tensor([[True, False, True]], dtype=torch.bool)

        self.assertEqual(tuple(kin(torch.randn(1, 3, 11), mask).shape), (1, 3, 16))
        self.assertEqual(tuple(identity(torch.randn(1, 3, 6), mask).shape), (1, 3, 16))
        self.assertEqual(tuple(track(torch.randn(1, 3, 4), mask).shape), (1, 3, 16))
        self.assertEqual(tuple(anchor(torch.randn(1, 3, RAW_TOKEN_DIM), mask).shape), (1, 3, 16))

    def test_subtoken_particle_encoder_shapes_and_masks(self):
        tokens, mask = self.make_tokens()
        model = SubtokenParticleEncoder(self.make_config())

        output = model(tokens, mask)

        self.assertEqual(tuple(output.subtokens.shape), (2, 5, 3, 16))
        self.assertEqual(tuple(output.anchor.shape), (2, 5, 16))
        self.assertEqual(tuple(output.modality_mask.shape), (2, 5, 3))
        self.assertEqual(output.modality_names, ("kinematics", "identity", "track"))
        self.assertEqual(output.summary()["contract"], SUBTOKEN_PART_ENCODER_CONTRACT)
        self.assertTrue(bool((output.subtokens[~mask] == 0.0).all()))
        self.assertTrue(bool((output.anchor[~mask] == 0.0).all()))
        self.assertTrue(bool(output.modality_mask[mask].all()))
        self.assertFalse(bool(output.modality_mask[~mask].any()))

    def test_subtokens_from_same_particle_share_anchor_contribution(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        model = SubtokenParticleEncoder(self.make_config())

        output = model(tokens, mask)
        recovered = output.recovered_anchor()
        expected = output.anchor[:, :, None, :].expand_as(recovered)

        self.assertTrue(bool(torch.allclose(recovered, expected, atol=1.0e-6, rtol=1.0e-6)))

    def test_different_modality_embeddings_change_outputs(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        config = self.make_config(use_particle_anchor=False, use_modality_type_embeddings=False)
        model = SubtokenParticleEncoder(config)

        output = model(tokens, mask)
        valid = mask
        kin_vs_id = (output.subtokens[:, :, 0, :] - output.subtokens[:, :, 1, :]).abs().sum(dim=-1)
        id_vs_track = (output.subtokens[:, :, 1, :] - output.subtokens[:, :, 2, :]).abs().sum(dim=-1)

        self.assertGreater(float(kin_vs_id[valid].mean().detach().cpu().item()), 0.0)
        self.assertGreater(float(id_vs_track[valid].mean().detach().cpu().item()), 0.0)
        self.assertTrue(bool(torch.isfinite(output.subtokens).all()))

    def test_gradients_flow_to_all_modality_encoders_and_anchor(self):
        tokens, mask = self.make_tokens()
        model = SubtokenParticleEncoder(self.make_config())

        output = model(tokens, mask)
        loss = output.subtokens[mask].square().mean() + output.anchor[mask].square().mean()
        loss.backward()

        for modality in (SUBTOKEN_MODALITY_KINEMATICS, SUBTOKEN_MODALITY_IDENTITY, SUBTOKEN_MODALITY_TRACK):
            grad_sum = sum(
                float(param.grad.detach().abs().sum().cpu().item())
                for param in model.modality_encoders[modality].parameters()
                if param.grad is not None
            )
            self.assertGreater(grad_sum, 0.0, modality)
        anchor_grad_sum = sum(
            float(param.grad.detach().abs().sum().cpu().item())
            for param in model.anchor_encoder.parameters()
            if param.grad is not None
        )
        self.assertGreater(anchor_grad_sum, 0.0)

    def test_anchor_source_dimensions_follow_feature_config(self):
        raw_cfg = SubtokenFeatureConfig(anchor_source="raw")
        part_cfg = SubtokenFeatureConfig(anchor_source="part_features")
        both_cfg = SubtokenFeatureConfig(anchor_source="raw_and_part_features")

        self.assertEqual(subtoken_anchor_input_dim(raw_cfg), RAW_TOKEN_DIM)
        self.assertEqual(subtoken_anchor_input_dim(part_cfg), len(PF_FEATURE_NAMES))
        self.assertEqual(subtoken_anchor_input_dim(both_cfg), RAW_TOKEN_DIM + len(PF_FEATURE_NAMES))

        dims = subtoken_modality_input_dims(raw_cfg)
        self.assertEqual(dims[SUBTOKEN_MODALITY_KINEMATICS], 11)
        self.assertEqual(dims[SUBTOKEN_MODALITY_IDENTITY], 6)
        self.assertEqual(dims[SUBTOKEN_MODALITY_TRACK], 8)

    def test_raw_and_part_feature_anchor_source_runs(self):
        tokens, mask = self.make_tokens()
        feature_config = SubtokenFeatureConfig(anchor_source="raw_and_part_features")
        config = self.make_config(feature_config=feature_config)
        model = SubtokenParticleEncoder(config)

        output = model(tokens, mask)

        self.assertEqual(tuple(output.anchor.shape), (2, 5, 16))
        self.assertTrue(bool(require_torch().isfinite(output.anchor).all()))

    def test_pt_rank_features_are_mask_safe_and_ordered(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()

        rank_features = build_pt_rank_features(tokens, mask)
        leading_indices = tokens[:, :, 0].masked_fill(~mask, -1.0e9).argmax(dim=1)

        self.assertEqual(tuple(rank_features.shape), (2, 5, 2))
        self.assertTrue(bool((rank_features[~mask] == 0.0).all()))
        for batch_index, leading_index in enumerate(leading_indices.tolist()):
            self.assertAlmostEqual(float(rank_features[batch_index, leading_index, 0].detach().cpu().item()), 0.0)
        self.assertTrue(bool(torch.isfinite(rank_features).all()))

    def test_pt_rank_embedding_and_modality_dropout_are_live(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        config = self.make_config(use_pt_rank_embedding=True, modality_dropout=0.95)
        model = SubtokenParticleEncoder(config)

        torch.manual_seed(123)
        model.train()
        output = model(tokens, mask)
        dropped_modality_mask = ~output.modality_mask
        dropped_modality_count = dropped_modality_mask[mask].sum()

        self.assertIsNotNone(output.pt_rank_embeddings)
        self.assertEqual(tuple(output.pt_rank_embeddings.shape), (2, 5, 16))
        self.assertTrue(bool((output.pt_rank_embeddings[~mask] == 0.0).all()))
        self.assertGreater(int(dropped_modality_count.detach().cpu().item()), 0)
        self.assertTrue(bool((output.subtokens[dropped_modality_mask] == 0.0).all()))
        self.assertTrue(bool(output.modality_mask[mask].any(dim=1).all()))


if __name__ == "__main__":
    unittest.main()

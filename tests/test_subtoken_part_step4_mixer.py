import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_LOCAL_MIXER_CONTRACT,
    SubtokenPartConfig,
    SubtokenParticleEncoder,
    WithinParticleSubtokenTransformer,
)


class SubtokenPartStep4MixerTests(unittest.TestCase):
    def make_config(self, **kwargs):
        defaults = {
            "num_classes": 2,
            "embed_dim": 16,
            "local_layers": 1,
            "local_heads": 4,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        }
        defaults.update(kwargs)
        return SubtokenPartConfig(**defaults)

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

    def test_mixer_accepts_encoder_output_and_preserves_shape(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        config = self.make_config()
        encoder = SubtokenParticleEncoder(config)
        mixer = WithinParticleSubtokenTransformer(config)

        encoded = encoder(tokens, mask)
        mixed = mixer(encoded)

        self.assertEqual(tuple(mixed.local_tokens.shape), tuple(encoded.subtokens.shape))
        self.assertEqual(tuple(mixed.mask.shape), tuple(mask.shape))
        self.assertEqual(tuple(mixed.modality_mask.shape), tuple(encoded.modality_mask.shape))
        self.assertEqual(mixed.modality_names, encoded.modality_names)
        self.assertEqual(mixed.summary()["contract"], SUBTOKEN_PART_LOCAL_MIXER_CONTRACT)
        self.assertTrue(bool(torch.isfinite(mixed.local_tokens).all()))
        self.assertTrue(bool((mixed.local_tokens[~mask] == 0.0).all()))

    def test_mixer_accepts_raw_subtoken_tensor(self):
        torch = require_torch()
        config = self.make_config()
        mixer = WithinParticleSubtokenTransformer(config)
        subtokens = torch.randn((2, 4, 3, 16), dtype=torch.float32)
        mask = torch.tensor([[True, False, True, True], [True, True, False, False]], dtype=torch.bool)

        mixed = mixer(subtokens, mask)

        self.assertEqual(tuple(mixed.local_tokens.shape), (2, 4, 3, 16))
        self.assertEqual(tuple(mixed.modality_mask.shape), (2, 4, 3))
        self.assertEqual(mixed.modality_names, ("modality_0", "modality_1", "modality_2"))
        self.assertTrue(bool((mixed.local_tokens[~mask] == 0.0).all()))
        self.assertTrue(bool(mixed.modality_mask[mask].all()))

    def test_attention_dropout_is_wired_into_local_transformer(self):
        config = self.make_config(dropout=0.0, attention_dropout=0.25)
        mixer = WithinParticleSubtokenTransformer(config)

        self.assertAlmostEqual(float(mixer.encoder.layers[0].dropout.p), 0.25)

    def test_mixer_respects_dropped_modality_mask_from_encoder(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        config = self.make_config(use_pt_rank_embedding=True, modality_dropout=0.95)
        encoder = SubtokenParticleEncoder(config)
        mixer = WithinParticleSubtokenTransformer(config)

        torch.manual_seed(123)
        encoder.train()
        encoded = encoder(tokens, mask)
        mixed = mixer(encoded)

        self.assertTrue(bool(torch.isfinite(mixed.local_tokens).all()))
        self.assertTrue(bool((encoded.subtokens[~encoded.modality_mask] == 0.0).all()))
        self.assertTrue(bool((mixed.local_tokens[~encoded.modality_mask] == 0.0).all()))
        self.assertTrue(bool(torch.equal(mixed.modality_mask, encoded.modality_mask)))

    def test_gradients_flow_to_input_and_mixer_parameters(self):
        torch = require_torch()
        config = self.make_config()
        mixer = WithinParticleSubtokenTransformer(config)
        subtokens = torch.randn((2, 4, 3, 16), dtype=torch.float32, requires_grad=True)
        mask = torch.tensor([[True, False, True, True], [True, True, False, False]], dtype=torch.bool)

        mixed = mixer(subtokens, mask)
        loss = mixed.local_tokens[mask].square().mean()
        loss.backward()

        self.assertIsNotNone(subtokens.grad)
        self.assertGreater(float(subtokens.grad[mask].abs().sum().detach().cpu().item()), 0.0)
        grad_sum = sum(
            float(param.grad.detach().abs().sum().cpu().item())
            for param in mixer.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)

    def test_particle_permutation_equivariance(self):
        torch = require_torch()
        config = self.make_config()
        mixer = WithinParticleSubtokenTransformer(config)
        mixer.eval()
        subtokens = torch.randn((2, 5, 3, 16), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, False, True, True],
                [True, False, True, False, True],
            ],
            dtype=torch.bool,
        )
        permutation = torch.tensor([3, 1, 0, 4, 2], dtype=torch.long)

        mixed = mixer(subtokens, mask)
        mixed_permuted = mixer(subtokens[:, permutation], mask[:, permutation])

        self.assertTrue(
            bool(torch.allclose(mixed_permuted.local_tokens, mixed.local_tokens[:, permutation], atol=1.0e-6, rtol=1.0e-6))
        )
        self.assertTrue(bool(torch.equal(mixed_permuted.mask, mixed.mask[:, permutation])))

    def test_invalid_shapes_raise_clear_errors(self):
        torch = require_torch()
        config = self.make_config()
        mixer = WithinParticleSubtokenTransformer(config)

        with self.assertRaises(ValueError):
            mixer(torch.zeros((2, 3, 16)), torch.ones((2, 3), dtype=torch.bool))
        with self.assertRaises(ValueError):
            mixer(torch.zeros((2, 3, 3, 15)), torch.ones((2, 3), dtype=torch.bool))
        with self.assertRaises(ValueError):
            mixer(torch.zeros((2, 3, 3, 16)), torch.ones((2, 4), dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()

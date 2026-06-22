import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.detr_slots import (
        DETR_SLOT_PARTICLE_NET_COORD_DIM,
        DETR_SLOT_PARTICLE_NET_ENCODER_STEP,
        DetrParticleNetEdgeConvBlock,
        DetrSlotDecoderConfig,
        ParticleNetHLTEncoderAdapter,
        ParticleNetHLTEncoderConfig,
        build_detr_slot_decoder_and_heads,
        detr_slot_particle_net_coordinates,
        detr_slot_token_embedding_features,
    )
else:  # pragma: no cover - environment dependent
    torch = None


def make_hlt_tokens(batch_size=2, particles=6):
    tokens = torch.randn(batch_size, particles, RAW_TOKEN_DIM)
    tokens[..., 0] = tokens[..., 0].abs() + 1.0
    tokens[..., 1] = 0.5 * torch.tanh(tokens[..., 1])
    tokens[..., 2] = 4.0 * tokens[..., 2]
    tokens[..., 3] = tokens[..., 3].abs() + 2.0
    tokens[..., 5:10] = torch.sigmoid(tokens[..., 5:10])
    tokens[..., 11] = torch.sigmoid(tokens[..., 11])
    tokens[..., 13] = torch.sigmoid(tokens[..., 13])
    return tokens


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DetrFreeSlotStep7ParticleNetEncoderTests(unittest.TestCase):
    def test_particle_net_coordinates_follow_mask_and_wrap_phi(self):
        tokens = make_hlt_tokens(batch_size=2, particles=5)
        mask = torch.tensor([[True, True, False, True, False], [True, True, True, True, True]])

        coords = detr_slot_particle_net_coordinates(tokens, mask)

        self.assertEqual(tuple(coords.shape), (2, 5, DETR_SLOT_PARTICLE_NET_COORD_DIM))
        self.assertTrue(torch.isfinite(coords).all())
        self.assertGreaterEqual(float(coords[..., 1].min()), -torch.pi)
        self.assertLessEqual(float(coords[..., 1].max()), torch.pi)
        self.assertTrue(torch.allclose(coords[0, 2], torch.zeros(DETR_SLOT_PARTICLE_NET_COORD_DIM)))

    def test_particle_net_config_rejects_bad_values(self):
        bad_configs = [
            {"input_dim": RAW_TOKEN_DIM + 1},
            {"memory_dim": 0},
            {"context_dim": 0},
            {"edgeconv_dims": ()},
            {"edgeconv_dims": (32, -1)},
            {"k": 0},
            {"dropout": 1.0},
            {"max_abs_eta": 0.0},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ParticleNetHLTEncoderConfig(**kwargs)

    def test_edgeconv_block_returns_masked_finite_features(self):
        torch.manual_seed(31)
        features = torch.randn(2, 5, 15)
        coords = torch.randn(2, 5, 3)
        mask = torch.tensor([[True, True, False, True, False], [True, True, True, True, True]])
        block = DetrParticleNetEdgeConvBlock(15, 12, k=8, dropout=0.0)

        output = block(features, coords, mask)

        self.assertEqual(tuple(output.shape), (2, 5, 12))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output[0, 2], torch.zeros(12)))
        self.assertTrue(torch.allclose(output[0, 4], torch.zeros(12)))

    def test_particle_net_encoder_returns_finite_contract(self):
        torch.manual_seed(32)
        encoder = ParticleNetHLTEncoderAdapter(
            ParticleNetHLTEncoderConfig(memory_dim=16, edgeconv_dims=(12, 16), k=8, dropout=0.0)
        )
        tokens = make_hlt_tokens(batch_size=3, particles=6)
        mask = torch.tensor(
            [
                [True, True, True, False, False, False],
                [False, False, False, False, False, False],
                [True, True, True, True, True, True],
            ]
        )

        output = encoder(tokens, mask)

        self.assertEqual(tuple(output.memory_tokens.shape), (3, 6, 16))
        self.assertEqual(tuple(output.memory_mask.shape), (3, 6))
        self.assertEqual(tuple(output.global_context.shape), (3, 16))
        self.assertTrue(torch.isfinite(output.memory_tokens).all())
        self.assertTrue(torch.isfinite(output.global_context).all())
        self.assertTrue(torch.equal(output.memory_mask, mask))
        self.assertTrue(torch.allclose(output.memory_tokens[0, 3:], torch.zeros(3, 16)))
        self.assertTrue(torch.allclose(output.memory_tokens[1], torch.zeros(6, 16)))
        self.assertTrue(torch.allclose(output.global_context[1], torch.zeros(16)))
        self.assertEqual(output.aux["encoder_step"], DETR_SLOT_PARTICLE_NET_ENCODER_STEP)
        self.assertEqual(output.aux["edgeconv_blocks"], 2.0)

    def test_particle_net_encoder_feeds_shared_decoder_with_context_projection(self):
        torch.manual_seed(33)
        encoder = ParticleNetHLTEncoderAdapter(memory_dim=16, context_dim=12, edgeconv_dims=(12, 16), k=4, dropout=0.0)
        decoder, heads = build_detr_slot_decoder_and_heads(
            decoder_config=DetrSlotDecoderConfig(
                num_slots=5,
                embed_dim=16,
                memory_dim=16,
                context_dim=12,
                num_layers=1,
                num_heads=4,
                dropout=0.0,
            ),
            heads_config={"hidden_dim": 32, "dropout": 0.0},
        )
        tokens = make_hlt_tokens(batch_size=2, particles=7)
        mask = torch.tensor([[True, True, True, True, False, False, False], [True, True, True, True, True, True, True]])

        encoded = encoder(tokens, mask)
        slots = decoder(encoded.memory_tokens, encoded.memory_mask, encoded.global_context)
        output = heads(slots, global_context=encoded.global_context)

        self.assertEqual(tuple(encoded.global_context.shape), (2, 12))
        self.assertEqual(tuple(slots.shape), (2, 5, 16))
        self.assertEqual(tuple(output.tokens.shape), (2, 5, RAW_TOKEN_DIM))
        self.assertEqual(output.diagnostics["aux_context_conditioned_heads"], 1.0)
        self.assertEqual(decoder.last_diagnostics["context_conditioned_queries"], 1.0)

    def test_particle_net_encoder_has_finite_gradients(self):
        torch.manual_seed(34)
        encoder = ParticleNetHLTEncoderAdapter(memory_dim=16, edgeconv_dims=(12, 16), k=4, dropout=0.0)
        tokens = make_hlt_tokens(batch_size=2, particles=5).requires_grad_()
        mask = torch.ones(2, 5, dtype=torch.bool)

        output = encoder(tokens, mask)
        loss = output.memory_tokens.mean() + output.global_context.mean()
        loss.backward()

        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())
        checked = 0
        for parameter in encoder.parameters():
            if parameter.grad is None:
                continue
            checked += 1
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(checked, 0)

    def test_particle_net_and_shared_embedding_features_agree_on_feature_shape(self):
        tokens = make_hlt_tokens(batch_size=1, particles=4)
        mask = torch.tensor([[True, False, True, True]])

        features = detr_slot_token_embedding_features(tokens, mask)
        coords = detr_slot_particle_net_coordinates(tokens, mask)

        self.assertEqual(features.shape[:2], coords.shape[:2])
        self.assertTrue(torch.allclose(features[0, 1], torch.zeros(features.shape[-1])))
        self.assertTrue(torch.allclose(coords[0, 1], torch.zeros(coords.shape[-1])))


if __name__ == "__main__":
    unittest.main()

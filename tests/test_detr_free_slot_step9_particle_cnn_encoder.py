import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.detr_slots import (
        DETR_SLOT_PARTICLE_CNN_ENCODER_STEP,
        DETR_SLOT_PARTICLE_CNN_FEATURE_DIM,
        DETR_SLOT_PARTICLE_CNN_ORDERING_ASSUMPTION,
        DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_DIM,
        DetrParticleCnnBlock,
        DetrSlotDecoderConfig,
        ParticleCnnHLTEncoderAdapter,
        ParticleCnnHLTEncoderConfig,
        build_detr_slot_decoder_and_heads,
        detr_slot_particle_cnn_features,
        detr_slot_particle_cnn_rank_features,
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
class DetrFreeSlotStep9ParticleCnnEncoderTests(unittest.TestCase):
    def test_particle_cnn_rank_features_follow_mask_and_cache_order(self):
        mask = torch.tensor([[True, True, False, True, False], [False, False, False, False, False]])

        rank_features = detr_slot_particle_cnn_rank_features(mask)

        self.assertEqual(tuple(rank_features.shape), (2, 5, DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_DIM))
        self.assertTrue(torch.isfinite(rank_features).all())
        self.assertTrue(torch.allclose(rank_features[0, 2], torch.zeros(DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_DIM)))
        self.assertTrue(torch.allclose(rank_features[1], torch.zeros(5, DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_DIM)))
        self.assertEqual(float(rank_features[0, 0, 0]), 0.0)
        self.assertEqual(float(rank_features[0, 0, 3]), 1.0)
        self.assertEqual(float(rank_features[0, 1, 3]), 0.0)

    def test_particle_cnn_features_append_rank_axis_features(self):
        tokens = make_hlt_tokens(batch_size=2, particles=5)
        mask = torch.tensor([[True, True, False, True, False], [True, True, True, True, True]])

        features = detr_slot_particle_cnn_features(tokens, mask)

        self.assertEqual(tuple(features.shape), (2, 5, DETR_SLOT_PARTICLE_CNN_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.allclose(features[0, 2], torch.zeros(DETR_SLOT_PARTICLE_CNN_FEATURE_DIM)))

    def test_particle_cnn_config_rejects_bad_values(self):
        bad_configs = [
            {"input_dim": RAW_TOKEN_DIM + 1},
            {"memory_dim": 0},
            {"context_dim": 0},
            {"hidden_channels": 0},
            {"kernel_sizes": ()},
            {"kernel_sizes": (4,), "dilations": (1,)},
            {"kernel_sizes": (3, 5), "dilations": (1,)},
            {"dilations": (0,)},
            {"context_mlp_dims": ()},
            {"dropout": 1.0},
            {"max_abs_eta": 0.0},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ParticleCnnHLTEncoderConfig(**kwargs)

    def test_particle_cnn_block_returns_masked_finite_features(self):
        torch.manual_seed(51)
        values = torch.randn(2, 12, 5)
        mask = torch.tensor([[True, True, False, True, False], [True, True, True, True, True]])
        block = DetrParticleCnnBlock(12, kernel_size=5, dilation=2, dropout=0.0)

        output = block(values, mask)

        self.assertEqual(tuple(output.shape), (2, 12, 5))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output[0, :, 2], torch.zeros(12)))
        self.assertTrue(torch.allclose(output[0, :, 4], torch.zeros(12)))

    def test_particle_cnn_encoder_returns_finite_contract(self):
        torch.manual_seed(52)
        encoder = ParticleCnnHLTEncoderAdapter(
            ParticleCnnHLTEncoderConfig(
                memory_dim=16,
                context_dim=12,
                hidden_channels=16,
                kernel_sizes=(5, 3),
                dilations=(1, 2),
                context_mlp_dims=(24,),
                dropout=0.0,
            )
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
        self.assertEqual(tuple(output.global_context.shape), (3, 12))
        self.assertTrue(torch.isfinite(output.memory_tokens).all())
        self.assertTrue(torch.isfinite(output.global_context).all())
        self.assertTrue(torch.equal(output.memory_mask, mask))
        self.assertTrue(torch.allclose(output.memory_tokens[0, 3:], torch.zeros(3, 16)))
        self.assertTrue(torch.allclose(output.memory_tokens[1], torch.zeros(6, 16)))
        self.assertTrue(torch.allclose(output.global_context[1], torch.zeros(12)))
        self.assertEqual(output.aux["encoder_step"], DETR_SLOT_PARTICLE_CNN_ENCODER_STEP)
        self.assertEqual(output.aux["ordering_assumption"], DETR_SLOT_PARTICLE_CNN_ORDERING_ASSUMPTION)
        self.assertEqual(output.aux["num_blocks"], 2.0)

    def test_particle_cnn_encoder_feeds_shared_decoder_with_context_projection(self):
        torch.manual_seed(53)
        encoder = ParticleCnnHLTEncoderAdapter(
            memory_dim=16,
            context_dim=12,
            hidden_channels=16,
            kernel_sizes=(5, 3),
            dilations=(1, 2),
            context_mlp_dims=(24,),
            dropout=0.0,
        )
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

    def test_particle_cnn_encoder_has_finite_gradients(self):
        torch.manual_seed(54)
        encoder = ParticleCnnHLTEncoderAdapter(
            memory_dim=16,
            context_dim=12,
            hidden_channels=16,
            kernel_sizes=(5, 3),
            dilations=(1, 2),
            context_mlp_dims=(24,),
            dropout=0.0,
        )
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


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.detr_slots import (
        DETR_SLOT_GLOBAL_TRANSFORMER_ENCODER_STEP,
        DETR_SLOT_TOKEN_EMBED_FEATURE_DIM,
        DetrSlotDecoderConfig,
        GlobalTransformerHLTEncoderAdapter,
        GlobalTransformerHLTEncoderConfig,
        build_detr_slot_decoder_and_heads,
        detr_slot_token_embedding_features,
    )
else:  # pragma: no cover - environment dependent
    torch = None


def make_hlt_tokens(batch_size=2, particles=6):
    tokens = torch.randn(batch_size, particles, RAW_TOKEN_DIM)
    tokens[..., 0] = tokens[..., 0].abs() + 1.0
    tokens[..., 1] = 0.5 * torch.tanh(tokens[..., 1])
    tokens[..., 3] = tokens[..., 3].abs() + 2.0
    tokens[..., 5:10] = torch.sigmoid(tokens[..., 5:10])
    tokens[..., 11] = torch.sigmoid(tokens[..., 11])
    tokens[..., 13] = torch.sigmoid(tokens[..., 13])
    return tokens


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DetrFreeSlotStep6GlobalTransformerEncoderTests(unittest.TestCase):
    def test_token_embedding_features_follow_raw_token_contract(self):
        tokens = make_hlt_tokens(batch_size=2, particles=5)
        mask = torch.tensor([[True, True, False, True, False], [True, True, True, True, True]])

        features = detr_slot_token_embedding_features(tokens, mask)

        self.assertEqual(tuple(features.shape), (2, 5, DETR_SLOT_TOKEN_EMBED_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.allclose(features[0, 2], torch.zeros(DETR_SLOT_TOKEN_EMBED_FEATURE_DIM)))

    def test_global_transformer_config_rejects_bad_values(self):
        bad_configs = [
            {"input_dim": RAW_TOKEN_DIM + 1},
            {"memory_dim": 10, "num_heads": 4},
            {"num_layers": 0},
            {"num_heads": 0},
            {"dropout": 1.0},
            {"max_abs_eta": 0.0},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    GlobalTransformerHLTEncoderConfig(**kwargs)

    def test_global_transformer_encoder_returns_finite_contract(self):
        torch.manual_seed(21)
        encoder = GlobalTransformerHLTEncoderAdapter(
            GlobalTransformerHLTEncoderConfig(memory_dim=16, num_layers=1, num_heads=4, dropout=0.0)
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
        self.assertEqual(output.aux["encoder_step"], DETR_SLOT_GLOBAL_TRANSFORMER_ENCODER_STEP)
        self.assertEqual(output.aux["forced_nonempty_attention_rows"], 1.0)

    def test_global_transformer_encoder_feeds_shared_decoder_with_context_projection(self):
        torch.manual_seed(22)
        encoder = GlobalTransformerHLTEncoderAdapter(memory_dim=16, context_dim=12, num_layers=1, num_heads=4, dropout=0.0)
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

    def test_global_transformer_encoder_has_finite_gradients(self):
        torch.manual_seed(23)
        encoder = GlobalTransformerHLTEncoderAdapter(memory_dim=16, num_layers=1, num_heads=4, dropout=0.0)
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

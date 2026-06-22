import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.detr_slots import (
        DETR_SLOT_ENCODER_OUTPUT_CONTRACT,
        BaseHLTEncoderAdapter,
        DetrPredictionHeads,
        DetrPredictionHeadsConfig,
        DetrSlotDecoder,
        DetrSlotDecoderConfig,
        DummyHLTEncoderAdapter,
        DummyHLTEncoderConfig,
        EncoderOutput,
        masked_mean_pool,
        validate_encoder_output,
    )
else:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DetrFreeSlotStep5EncoderTests(unittest.TestCase):
    def test_encoder_output_validates_shapes_and_reports_contract(self):
        memory = torch.randn(2, 5, 12)
        mask = torch.tensor([[True, True, False, True, False], [True, True, True, True, True]])
        context = torch.randn(2, 7)

        output = EncoderOutput(memory_tokens=memory, memory_mask=mask, global_context=context, aux={"epoch": 4})

        self.assertEqual(output.batch_size, 2)
        self.assertEqual(output.memory_size, 5)
        self.assertEqual(output.memory_dim, 12)
        self.assertEqual(output.context_dim, 7)
        self.assertTrue(output.has_global_context)
        self.assertEqual(output.shape_report()["contract"], DETR_SLOT_ENCODER_OUTPUT_CONTRACT)
        self.assertEqual(output.detached_float_diagnostics()["aux_epoch"], 4.0)

    def test_encoder_output_rejects_bad_shapes_and_nonfinite_values(self):
        memory = torch.zeros(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)

        with self.assertRaises(ValueError):
            EncoderOutput(memory_tokens=torch.zeros(2, 4), memory_mask=mask)
        with self.assertRaises(ValueError):
            EncoderOutput(memory_tokens=memory, memory_mask=torch.ones(2, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            EncoderOutput(memory_tokens=memory, memory_mask=mask, global_context=torch.zeros(3, 8))

        bad_memory = memory.clone()
        bad_memory[0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            EncoderOutput(memory_tokens=bad_memory, memory_mask=mask)

    def test_validator_rechecks_encoder_output(self):
        output = EncoderOutput(
            memory_tokens=torch.zeros(1, 2, 3),
            memory_mask=torch.ones(1, 2, dtype=torch.bool),
            global_context=None,
        )

        checked = validate_encoder_output(output)

        self.assertIsInstance(checked, EncoderOutput)
        self.assertEqual(checked.context_dim, 0)
        self.assertFalse(checked.has_global_context)

    def test_base_adapter_validates_inputs_and_defaults_mask_to_all_true(self):
        adapter = BaseHLTEncoderAdapter(input_dim=4, memory_dim=8)
        tokens = torch.randn(2, 3, 4)

        checked_tokens, checked_mask = adapter.validate_hlt_inputs(tokens)

        self.assertIs(checked_tokens, tokens)
        self.assertTrue(bool(checked_mask.all()))
        with self.assertRaises(ValueError):
            adapter.validate_hlt_inputs(torch.randn(2, 3, 5))
        with self.assertRaises(ValueError):
            adapter.validate_hlt_inputs(tokens, torch.ones(2, 2, dtype=torch.bool))

    def test_masked_mean_pool_handles_empty_rows(self):
        values = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 0.0], [5.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            ]
        )
        mask = torch.tensor([[True, True, False], [False, False, False]])

        pooled = masked_mean_pool(values, mask)

        self.assertTrue(torch.allclose(pooled[0], torch.tensor([2.0, 1.0])))
        self.assertTrue(torch.allclose(pooled[1], torch.zeros(2)))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_dummy_encoder_returns_shared_contract(self):
        encoder = DummyHLTEncoderAdapter(DummyHLTEncoderConfig(input_dim=RAW_TOKEN_DIM, memory_dim=16, hidden_dim=24))
        tokens = torch.randn(2, 7, RAW_TOKEN_DIM)
        mask = torch.tensor([[True, True, True, False, False, False, False], [True, True, True, True, True, True, True]])

        output = encoder(tokens, mask)

        self.assertIsInstance(output, EncoderOutput)
        self.assertEqual(tuple(output.memory_tokens.shape), (2, 7, 16))
        self.assertEqual(tuple(output.memory_mask.shape), (2, 7))
        self.assertEqual(tuple(output.global_context.shape), (2, 16))
        self.assertTrue(torch.isfinite(output.memory_tokens).all())
        self.assertTrue(torch.isfinite(output.global_context).all())
        self.assertTrue(torch.allclose(output.memory_tokens[0, 3:], torch.zeros(4, 16)))

    def test_dummy_encoder_can_feed_decoder_and_heads(self):
        torch.manual_seed(12)
        encoder = DummyHLTEncoderAdapter(input_dim=RAW_TOKEN_DIM, memory_dim=16, hidden_dim=32)
        decoder = DetrSlotDecoder(
            DetrSlotDecoderConfig(
                num_slots=5,
                embed_dim=16,
                memory_dim=16,
                context_dim=16,
                num_layers=1,
                num_heads=4,
                dropout=0.0,
            )
        )
        heads = DetrPredictionHeads(DetrPredictionHeadsConfig(embed_dim=16, hidden_dim=32, context_dim=16, dropout=0.0))
        tokens = torch.randn(2, 8, RAW_TOKEN_DIM)
        mask = torch.tensor([[True, True, True, True, False, False, False, False], [True, True, True, True, True, True, True, True]])

        encoded = encoder(tokens, mask)
        slot_embeddings = decoder(encoded.memory_tokens, encoded.memory_mask, encoded.global_context)
        output = heads(slot_embeddings, global_context=encoded.global_context)

        self.assertEqual(tuple(output.tokens.shape), (2, 5, RAW_TOKEN_DIM))
        self.assertEqual(tuple(output.existence_logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(output.tokens).all())
        self.assertEqual(decoder.last_diagnostics["context_conditioned_queries"], 1.0)
        self.assertEqual(output.diagnostics["aux_context_conditioned_heads"], 1.0)

    def test_dummy_encoder_decoder_heads_have_finite_gradients(self):
        torch.manual_seed(13)
        encoder = DummyHLTEncoderAdapter(input_dim=RAW_TOKEN_DIM, memory_dim=16, hidden_dim=32)
        decoder = DetrSlotDecoder(
            DetrSlotDecoderConfig(num_slots=4, embed_dim=16, memory_dim=16, num_layers=1, num_heads=4, dropout=0.0)
        )
        heads = DetrPredictionHeads(DetrPredictionHeadsConfig(embed_dim=16, hidden_dim=32, dropout=0.0))
        tokens = torch.randn(2, 6, RAW_TOKEN_DIM, requires_grad=True)
        mask = torch.ones(2, 6, dtype=torch.bool)

        encoded = encoder(tokens, mask)
        slot_embeddings = decoder(encoded.memory_tokens, encoded.memory_mask)
        output = heads(slot_embeddings)
        loss = output.tokens.mean() + output.existence_logits.mean() + encoded.global_context.mean()
        loss.backward()

        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())
        checked = 0
        for module in (encoder, decoder, heads):
            for parameter in module.parameters():
                if parameter.grad is None:
                    continue
                checked += 1
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()

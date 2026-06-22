import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.detr_slots import (
        DETR_SLOT_OUTPUT_CONTRACT,
        DetrPredictionHeads,
        DetrPredictionHeadsConfig,
        DetrSlotDecoder,
        DetrSlotDecoderConfig,
        DetrSlotOutput,
        LearnedSlotQueries,
        LearnedSlotQueryConfig,
        build_detr_slot_decoder_and_heads,
    )
    from teacher_logit_reco.set_matching.losses import SetMatchingLossConfig, compute_set_matching_loss
else:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DetrFreeSlotStep4DecoderTests(unittest.TestCase):
    def test_learned_queries_expand_across_batch(self):
        queries = LearnedSlotQueries(LearnedSlotQueryConfig(num_slots=5, embed_dim=16))
        expanded = queries(3)

        self.assertEqual(tuple(expanded.shape), (3, 5, 16))
        self.assertTrue(torch.allclose(expanded[0], expanded[1]))
        self.assertTrue(expanded.requires_grad)

    def test_decoder_forward_shape_with_fake_memory(self):
        decoder = DetrSlotDecoder(
            DetrSlotDecoderConfig(
                num_slots=6,
                embed_dim=16,
                memory_dim=12,
                num_layers=2,
                num_heads=4,
                dropout=0.0,
            )
        )
        memory = torch.randn(2, 7, 12)
        memory_mask = torch.tensor([[True, True, True, False, False, False, False], [True, True, True, True, True, True, True]])

        slots = decoder(memory, memory_mask)

        self.assertEqual(tuple(slots.shape), (2, 6, 16))
        self.assertTrue(torch.isfinite(slots).all())
        self.assertEqual(decoder.last_diagnostics["num_slots"], 6.0)

    def test_decoder_forces_empty_memory_rows_to_remain_finite(self):
        decoder = DetrSlotDecoder(
            DetrSlotDecoderConfig(num_slots=3, embed_dim=8, memory_dim=8, num_layers=1, num_heads=2, dropout=0.0)
        )
        memory = torch.randn(2, 4, 8)
        memory_mask = torch.tensor([[False, False, False, False], [True, False, False, False]])

        slots = decoder(memory, memory_mask)

        self.assertTrue(torch.isfinite(slots).all())
        self.assertEqual(decoder.last_diagnostics["forced_nonempty_memory_rows"], 1.0)

    def test_prediction_heads_return_detr_slot_output_contract(self):
        heads = DetrPredictionHeads(
            DetrPredictionHeadsConfig(embed_dim=16, hidden_dim=32, dropout=0.0, existence_bias=-1.5)
        )
        embeddings = torch.randn(2, 5, 16)
        slot_mask = torch.tensor([[True, True, False, True, True], [True, True, True, True, True]])

        output = heads(embeddings, slot_mask)

        self.assertIsInstance(output, DetrSlotOutput)
        self.assertEqual(tuple(output.tokens.shape), (2, 5, RAW_TOKEN_DIM))
        self.assertEqual(tuple(output.loss_features.shape), (2, 5, RAW_TOKEN_DIM))
        self.assertEqual(tuple(output.core_outputs.shape), (2, 5, 4))
        self.assertEqual(tuple(output.aux_outputs.shape), (2, 5, RAW_TOKEN_DIM - 4))
        self.assertEqual(tuple(output.existence_logits.shape), (2, 5))
        self.assertEqual(tuple(output.slot_mask.shape), (2, 5))
        self.assertEqual(output.shape_report()["contract"], DETR_SLOT_OUTPUT_CONTRACT)
        self.assertTrue(torch.isfinite(output.tokens).all())
        self.assertTrue(torch.isfinite(output.loss_features).all())
        self.assertTrue(torch.isfinite(output.existence_logits).all())

    def test_context_changes_decoder_and_head_outputs(self):
        torch.manual_seed(5)
        decoder = DetrSlotDecoder(
            DetrSlotDecoderConfig(
                num_slots=4,
                embed_dim=16,
                memory_dim=16,
                context_dim=6,
                num_layers=1,
                num_heads=4,
                dropout=0.0,
            )
        )
        heads = DetrPredictionHeads(
            DetrPredictionHeadsConfig(embed_dim=16, hidden_dim=32, context_dim=6, dropout=0.0)
        )
        memory = torch.randn(2, 7, 16)
        mask = torch.ones(2, 7, dtype=torch.bool)
        context_a = torch.zeros(2, 6)
        context_b = torch.ones(2, 6)

        slots_a = decoder(memory, mask, context_a)
        slots_b = decoder(memory, mask, context_b)
        out_a = heads(slots_a, global_context=context_a)
        out_b = heads(slots_a, global_context=context_b)

        self.assertFalse(torch.allclose(slots_a, slots_b))
        self.assertFalse(torch.allclose(out_a.loss_features, out_b.loss_features))
        self.assertEqual(decoder.last_diagnostics["context_conditioned_queries"], 1.0)
        self.assertEqual(out_b.diagnostics["aux_context_conditioned_heads"], 1.0)

    def test_decoder_and_heads_have_finite_gradients(self):
        torch.manual_seed(7)
        decoder, heads = build_detr_slot_decoder_and_heads(
            decoder_config={
                "num_slots": 4,
                "embed_dim": 16,
                "memory_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
                "dropout": 0.0,
            },
            heads_config={
                "hidden_dim": 32,
                "dropout": 0.0,
                "feature_config": {"feature_dim": 8},
            },
        )
        memory = torch.randn(2, 6, 16, requires_grad=True)
        memory_mask = torch.ones(2, 6, dtype=torch.bool)

        slots = decoder(memory, memory_mask)
        output = heads(slots)
        loss = output.loss_features.mean() + output.existence_logits.mean()
        loss.backward()

        self.assertIsNotNone(memory.grad)
        self.assertTrue(torch.isfinite(memory.grad).all())
        checked = 0
        for parameter in list(decoder.parameters()) + list(heads.parameters()):
            if parameter.grad is None:
                continue
            checked += 1
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(checked, 0)

    def test_output_to_loss_kwargs_smokes_existing_set_matching_loss(self):
        torch.manual_seed(8)
        heads = DetrPredictionHeads(
            {"embed_dim": 16, "hidden_dim": 32, "dropout": 0.0, "feature_config": {"feature_dim": RAW_TOKEN_DIM}}
        )
        output = heads(torch.randn(2, 3, 16))
        offline = output.loss_predicted_features.detach().clone()
        offline_mask = torch.ones(2, 3, dtype=torch.bool)

        loss_output = compute_set_matching_loss(
            **output.to_loss_kwargs(offline_features=offline, offline_mask=offline_mask),
            config=SetMatchingLossConfig(brute_force_fallback_limit=4),
        )

        self.assertTrue(torch.isfinite(loss_output.total_loss))
        self.assertIn("candidate_count_mean", loss_output.diagnostics)

    def test_all_encoder_names_can_share_same_decoder_head_contract(self):
        for name in ("gt", "pn", "pfn", "pcnn"):
            decoder = DetrSlotDecoder(
                DetrSlotDecoderConfig(num_slots=2, embed_dim=8, memory_dim=8, num_layers=1, num_heads=2, dropout=0.0)
            )
            heads = DetrPredictionHeads(DetrPredictionHeadsConfig(embed_dim=8, hidden_dim=16, dropout=0.0))
            memory = torch.randn(1, 3, 8)
            slots = decoder(memory, torch.ones(1, 3, dtype=torch.bool))
            output = heads(slots, aux={"encoder_name": name})

            self.assertEqual(tuple(output.tokens.shape), (1, 2, RAW_TOKEN_DIM))
            self.assertEqual(output.aux["encoder_name"], name)


if __name__ == "__main__":
    unittest.main()

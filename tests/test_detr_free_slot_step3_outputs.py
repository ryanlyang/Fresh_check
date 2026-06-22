import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.detr_slots import (
        DETR_SLOT_OUTPUT_CONTRACT,
        DetrSlotOutput,
        detr_slot_output_from_tensors,
        validate_detr_slot_output,
    )
else:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DetrFreeSlotStep3OutputTests(unittest.TestCase):
    def make_output(self):
        tokens = torch.zeros((2, 4, RAW_TOKEN_DIM), dtype=torch.float32)
        logits = torch.tensor([[2.0, 0.0, -1.0, 3.0], [1.0, -2.0, 0.5, 0.0]], dtype=torch.float32)
        mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
        return DetrSlotOutput(tokens=tokens, existence_logits=logits, slot_mask=mask, aux={"epoch": 3})

    def test_output_validates_shapes_and_exposes_dimensions(self):
        output = self.make_output()

        self.assertEqual(output.batch_size, 2)
        self.assertEqual(output.num_slots, 4)
        self.assertEqual(output.feature_dim, RAW_TOKEN_DIM)
        self.assertEqual(tuple(output.candidate_mask.shape), (2, 4))
        self.assertIs(output.predicted_features, output.tokens)
        report = output.shape_report()
        self.assertEqual(report["tokens_shape"], [2, 4, RAW_TOKEN_DIM])
        self.assertEqual(report["contract"], DETR_SLOT_OUTPUT_CONTRACT)

    def test_output_converts_mask_and_logits_to_tensor_contract(self):
        output = DetrSlotOutput(
            tokens=[[[0.0, 1.0, 2.0, 3.0]]],
            existence_logits=[[0]],
            slot_mask=[[1]],
        )

        self.assertEqual(tuple(output.tokens.shape), (1, 1, 4))
        self.assertTrue(torch.is_floating_point(output.existence_logits))
        self.assertEqual(output.slot_mask.dtype, torch.bool)

    def test_bad_shapes_raise_clear_errors(self):
        tokens = torch.zeros((2, 4, RAW_TOKEN_DIM), dtype=torch.float32)
        with self.assertRaises(ValueError):
            DetrSlotOutput(tokens=tokens, existence_logits=torch.zeros((2, 4, 1)), slot_mask=torch.ones((2, 4)))
        with self.assertRaises(ValueError):
            DetrSlotOutput(tokens=tokens, existence_logits=torch.zeros((2, 3)), slot_mask=torch.ones((2, 4)))
        with self.assertRaises(ValueError):
            DetrSlotOutput(
                tokens=torch.zeros((0, 4, RAW_TOKEN_DIM)),
                existence_logits=torch.zeros((0, 4)),
                slot_mask=torch.ones((0, 4)),
            )

    def test_nonfinite_values_raise(self):
        tokens = torch.zeros((1, 2, 4), dtype=torch.float32)
        logits = torch.zeros((1, 2), dtype=torch.float32)
        mask = torch.ones((1, 2), dtype=torch.bool)
        bad_tokens = tokens.clone()
        bad_tokens[0, 0, 0] = float("nan")
        bad_logits = logits.clone()
        bad_logits[0, 1] = float("inf")

        with self.assertRaises(FloatingPointError):
            DetrSlotOutput(tokens=bad_tokens, existence_logits=logits, slot_mask=mask)
        with self.assertRaises(FloatingPointError):
            DetrSlotOutput(tokens=tokens, existence_logits=bad_logits, slot_mask=mask)

    def test_probabilities_counts_and_diagnostics_are_masked(self):
        output = self.make_output()
        probs = output.existence_probabilities()
        masked = output.masked_existence_probabilities()
        diagnostics = output.detached_float_diagnostics()

        self.assertEqual(tuple(probs.shape), (2, 4))
        self.assertEqual(float(masked[0, 2]), 0.0)
        self.assertTrue(torch.allclose(output.active_slot_counts(), torch.tensor([3.0, 3.0])))
        self.assertIn("expected_particle_count_mean", diagnostics)
        self.assertEqual(diagnostics["aux_epoch"], 3.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values()))

    def test_loss_kwargs_match_existing_set_matching_vocabulary(self):
        aux_outputs = torch.zeros((2, 4, 10), dtype=torch.float32)
        output = DetrSlotOutput(
            tokens=torch.ones((2, 4, RAW_TOKEN_DIM), dtype=torch.float32),
            loss_features=torch.zeros((2, 4, RAW_TOKEN_DIM), dtype=torch.float32),
            existence_logits=torch.zeros((2, 4), dtype=torch.float32),
            slot_mask=torch.ones((2, 4), dtype=torch.bool),
            aux_outputs=aux_outputs,
        )
        offline = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        offline_mask = torch.ones((2, 5), dtype=torch.bool)

        kwargs = output.to_loss_kwargs(offline_features=offline, offline_mask=offline_mask)
        detr_kwargs = output.to_loss_kwargs(
            offline_features=offline,
            offline_mask=offline_mask,
            include_aux_logits=True,
        )

        self.assertIs(kwargs["predicted_features"], output.loss_features)
        self.assertIs(kwargs["existence_logits"], output.existence_logits)
        self.assertIs(kwargs["candidate_mask"], output.slot_mask)
        self.assertIs(kwargs["offline_features"], offline)
        self.assertIs(kwargs["offline_mask"], offline_mask)
        self.assertNotIn("predicted_aux_logits", kwargs)
        self.assertIs(detr_kwargs["predicted_aux_logits"], output.aux_outputs)
        self.assertTrue(torch.allclose(output.export_features, torch.ones((2, 4, RAW_TOKEN_DIM))))
        self.assertTrue(torch.allclose(output.predicted_features, torch.ones((2, 4, RAW_TOKEN_DIM))))
        self.assertTrue(torch.allclose(output.loss_predicted_features, torch.zeros((2, 4, RAW_TOKEN_DIM))))

    def test_output_exposes_existing_cache_and_trainer_compatibility_properties(self):
        output = self.make_output()

        diagnostics = output.diagnostics
        weights = output.candidate_weights

        self.assertIn("expected_particle_count_mean", diagnostics)
        self.assertEqual(tuple(weights.shape), (2, 4))
        self.assertEqual(float(weights[0, 2]), 0.0)
        self.assertTrue(torch.isfinite(weights).all())

    def test_factory_and_validator_return_validated_output(self):
        output = detr_slot_output_from_tensors(
            torch.zeros((1, 2, 6)),
            torch.zeros((1, 2)),
            torch.ones((1, 2)),
            aux={"loss": torch.tensor(1.25)},
            aux_outputs=torch.zeros((1, 2, 2)),
        )
        checked = validate_detr_slot_output(output)

        self.assertIsInstance(checked, DetrSlotOutput)
        self.assertEqual(checked.shape_report()["feature_dim"], 6)
        self.assertEqual(checked.shape_report()["aux_outputs_shape"], [1, 2, 2])
        self.assertEqual(checked.detached_float_diagnostics()["aux_loss"], 1.25)

    def test_factory_preserves_loss_features_and_core_outputs(self):
        output = detr_slot_output_from_tensors(
            torch.ones((1, 2, 6)),
            torch.zeros((1, 2)),
            torch.ones((1, 2)),
            loss_features=torch.zeros((1, 2, 6)),
            core_outputs=torch.zeros((1, 2, 4)),
            aux_outputs=torch.zeros((1, 2, 2)),
        )

        self.assertTrue(torch.allclose(output.predicted_features, torch.ones((1, 2, 6))))
        self.assertTrue(torch.allclose(output.loss_predicted_features, torch.zeros((1, 2, 6))))
        self.assertEqual(tuple(output.core_outputs.shape), (1, 2, 4))
        self.assertEqual(tuple(output.aux_outputs.shape), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()

import importlib.util
import math
import tempfile
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.set_matching.reconstructors import (
    SET_MATCHING_RECONSTRUCTOR_CONTRACT,
    SetMatchingReconstructorConfig,
    build_set_matching_reconstructor,
    load_set_matching_reconstructor_checkpoint,
    normalize_set_matching_reconstructor_architecture,
    save_set_matching_reconstructor_checkpoint,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def tiny_model_config(architecture):
    head = {"num_extra_candidates": 2, "dropout": 0.0}
    if architecture == "gt":
        return {
            "hidden_dim": 32,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "aggressive_head_config": head,
        }
    if architecture == "pn":
        return {
            "edgeconv_dims": (16,),
            "k": 2,
            "dropout": 0.0,
            "embedding_dim": 16,
            "aggressive_head_config": head,
        }
    if architecture == "pfn":
        return {
            "phi_dims": (16,),
            "context_dim": 16,
            "context_mlp_dims": (16,),
            "dropout": 0.0,
            "embedding_dim": 16,
            "aggressive_head_config": head,
        }
    if architecture == "pcnn":
        return {
            "hidden_channels": 16,
            "num_blocks": 1,
            "kernel_sizes": (3,),
            "dilations": (1,),
            "context_dim": 16,
            "context_mlp_dims": (16,),
            "dropout": 0.0,
            "embedding_dim": 16,
            "aggressive_head_config": head,
        }
    raise AssertionError(architecture)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SetMatchingReconstructorTests(unittest.TestCase):
    def make_batch(self):
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((2, 5), dtype=torch.bool)
        mask[0, :4] = True
        mask[1, :3] = True
        for batch in range(2):
            for index in range(5):
                if not bool(mask[batch, index]):
                    continue
                pt = 10.0 + batch + index
                eta = -0.2 + 0.05 * index
                phi = -0.3 + 0.1 * index
                tokens[batch, index, 0] = pt
                tokens[batch, index, 1] = eta
                tokens[batch, index, 2] = phi
                tokens[batch, index, 3] = pt * math.cosh(eta) + 0.5
                tokens[batch, index, 4] = 1.0
                tokens[batch, index, 5 + (index % 5)] = 1.0
                tokens[batch, index, 10:14] = torch.tensor([0.1, 0.01, -0.2, 0.02])
        labels = torch.tensor([1, 8], dtype=torch.long)
        return tokens, mask, labels

    def test_architecture_aliases_normalize(self):
        self.assertEqual(normalize_set_matching_reconstructor_architecture("aggressive_particle_net"), "pn")
        self.assertEqual(normalize_set_matching_reconstructor_architecture("particle_flow"), "pfn")
        self.assertEqual(normalize_set_matching_reconstructor_architecture("particle_transformer"), "gt")

    def test_all_four_reconstructors_emit_set_matching_contract(self):
        tokens, mask, labels = self.make_batch()
        for architecture in ("gt", "pn", "pfn", "pcnn"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(123)
                model = build_set_matching_reconstructor(
                    architecture,
                    SetMatchingReconstructorConfig(
                        architecture=architecture,
                        model_config=tiny_model_config(architecture),
                    ),
                )
                output = model(tokens, mask, labels=labels, split="model_train")
                self.assertEqual(output.diagnostics["output_contract"], SET_MATCHING_RECONSTRUCTOR_CONTRACT)
                self.assertEqual(tuple(output.predicted_features.shape), (2, 7, RAW_TOKEN_DIM))
                self.assertEqual(tuple(output.existence_logits.shape), (2, 7))
                self.assertEqual(tuple(output.candidate_mask.shape), (2, 7))
                self.assertTrue(torch.equal(output.candidate_mask[:, :5], mask))
                self.assertTrue(bool(output.candidate_mask[:, 5:].all()))
                self.assertTrue(bool(torch.isfinite(output.predicted_features).all()))
                self.assertTrue(bool(torch.isfinite(output.existence_logits).all()))
                self.assertGreaterEqual(float(output.candidate_weights[output.candidate_mask].min()), 0.0)
                self.assertLessEqual(float(output.candidate_weights[output.candidate_mask].max()), 1.0)

    def test_checkpoint_roundtrip_preserves_builder_config(self):
        tokens, mask, labels = self.make_batch()
        torch.manual_seed(123)
        model = build_set_matching_reconstructor(
            "gt",
            SetMatchingReconstructorConfig(
                architecture="gt",
                model_config=tiny_model_config("gt"),
            ),
        )
        before = model(tokens, mask, labels=labels)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_set_matching_reconstructor_checkpoint(f"{tmpdir}/best_model_val.pt", model)
            loaded, payload = load_set_matching_reconstructor_checkpoint(path, expected_architecture="gt")
        after = loaded(tokens, mask, labels=labels)

        self.assertEqual(payload["set_matching_architecture"], "gt")
        self.assertEqual(loaded.architecture, "gt")
        self.assertEqual(before.shape_report(), after.shape_report())
        self.assertTrue(torch.isfinite(after.predicted_features).all())


if __name__ == "__main__":
    unittest.main()


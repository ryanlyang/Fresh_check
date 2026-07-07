import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.hlt_baseline import require_torch
from teacher_logit_reco.hlt_self_dualview import (
    HLT_SDV_ALLOWED_INPUTS,
    HLT_SDV_DEPLOYMENT_INPUTS,
    HLT_SDV_MODEL_ARCHITECTURE,
    HLT_SDV_MODEL_CONTRACT,
    HLT_SDV_STEP4_EXPERIMENT_STEP,
    HLTSelfDualViewFusionModel,
    extract_model_state_dict,
    forward_hlt_sdv_batch,
    hlt_sdv_branch_dim_from_config,
    hlt_sdv_embedding_branch_config,
    initialize_hlt_sdv_branches_from_hlt_checkpoint,
    strip_compile_prefix_from_state_dict,
)


torch = require_torch()


class DummyEmbeddingBranch(torch.nn.Module):
    def __init__(self, *, in_dim: int = 6, branch_dim: int = 8, bias: float = 0.0) -> None:
        super().__init__()
        self.branch_dim = int(branch_dim)
        self.config = {"branch_dim": int(branch_dim), "input_dim": int(in_dim)}
        self.proj = torch.nn.Linear(int(in_dim), int(branch_dim))
        torch.nn.init.constant_(self.proj.bias, float(bias))

    def forward(self, inputs):
        x = inputs["features"].mean(dim=-1)
        return self.proj(x)


def make_inputs(*, batch_size: int = 3, in_dim: int = 6, particles: int = 5, offset: float = 0.0):
    features = torch.arange(batch_size * in_dim * particles, dtype=torch.float32).reshape(batch_size, in_dim, particles)
    features = features / 100.0 + float(offset)
    return {
        "points": torch.zeros(batch_size, 2, particles),
        "features": features,
        "lorentz_vectors": torch.ones(batch_size, 4, particles),
        "mask": torch.ones(batch_size, 1, particles, dtype=torch.bool),
    }


class HLTSDVStep4ModelTest(unittest.TestCase):
    def test_config_helpers_make_classifier_free_branch_config(self):
        cfg = hlt_sdv_embedding_branch_config(model_size="tiny")
        self.assertIsNone(cfg["num_classes"])
        self.assertIsNone(cfg["fc_params"])
        self.assertEqual(hlt_sdv_branch_dim_from_config(cfg), cfg["embed_dims"][-1])

    def test_model_forward_uses_two_branches_and_returns_representation(self):
        model = HLTSelfDualViewFusionModel(
            hlt_branch=DummyEmbeddingBranch(in_dim=6, branch_dim=8, bias=0.1),
            hlt2_branch=DummyEmbeddingBranch(in_dim=6, branch_dim=8, bias=0.2),
            branch_dim=8,
            fusion_hidden_dim=16,
            representation_dim=12,
            dropout=0.0,
        )
        hlt_inputs = make_inputs(offset=0.0)
        hlt2_inputs = make_inputs(offset=1.0)

        logits, representation = model(hlt_inputs, hlt2_inputs, return_representation=True)

        self.assertEqual(tuple(logits.shape), (3, 10))
        self.assertEqual(tuple(representation.shape), (3, 12))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.allclose(representation.norm(dim=1), torch.ones(3), atol=1e-5))
        self.assertEqual(model.config["contract"], HLT_SDV_MODEL_CONTRACT)
        self.assertEqual(model.config["architecture"], HLT_SDV_MODEL_ARCHITECTURE)
        self.assertEqual(model.config["experiment_step"], HLT_SDV_STEP4_EXPERIMENT_STEP)
        self.assertEqual(model.config["allowed_inputs"], HLT_SDV_ALLOWED_INPUTS)
        self.assertEqual(model.config["deployment_inputs"], HLT_SDV_DEPLOYMENT_INPUTS)
        self.assertFalse(model.config["requires_offline_inputs"])
        self.assertFalse(model.config["requires_teacher_features"])

    def test_forward_batch_and_branch_freezing(self):
        model = HLTSelfDualViewFusionModel(
            hlt_branch=DummyEmbeddingBranch(branch_dim=8),
            hlt2_branch=DummyEmbeddingBranch(branch_dim=8),
            branch_dim=8,
            fusion_hidden_dim=16,
            representation_dim=12,
            dropout=0.0,
        )
        batch = {"hlt_inputs": make_inputs(), "hlt2_inputs": make_inputs(offset=0.5)}
        logits = forward_hlt_sdv_batch(model, batch)
        self.assertEqual(tuple(logits.shape), (3, 10))

        model.set_branches_trainable(False)
        self.assertFalse(any(parameter.requires_grad for parameter in model.branch_parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.head_parameters()))
        model.set_branches_trainable(True)
        self.assertTrue(all(parameter.requires_grad for parameter in model.branch_parameters()))

    def test_initialize_both_branches_from_same_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hlt_teacher.pt"
            source = DummyEmbeddingBranch(branch_dim=8)
            with torch.no_grad():
                source.proj.weight.fill_(0.25)
                source.proj.bias.fill_(0.75)
            torch.save(
                {
                    "epoch": 7,
                    "experiment_step": "unit_hlt_teacher",
                    "model_state_dict": source.state_dict(),
                },
                path,
            )
            model = HLTSelfDualViewFusionModel(
                hlt_branch=DummyEmbeddingBranch(branch_dim=8),
                hlt2_branch=DummyEmbeddingBranch(branch_dim=8),
                branch_dim=8,
                fusion_hidden_dim=16,
                representation_dim=12,
                dropout=0.0,
            )

            report = initialize_hlt_sdv_branches_from_hlt_checkpoint(
                model,
                hlt_checkpoint=path,
                device=torch.device("cpu"),
                min_match_fraction=1.0,
            )

            self.assertTrue(report["both_branches_initialized_from_same_checkpoint"])
            self.assertEqual(report["hlt_branch"]["matched_tensors"], 2)
            self.assertEqual(report["hlt2_branch"]["matched_tensors"], 2)
            for branch in (model.hlt_branch, model.hlt2_branch):
                self.assertTrue(torch.allclose(branch.proj.weight, source.proj.weight))
                self.assertTrue(torch.allclose(branch.proj.bias, source.proj.bias))

    def test_state_dict_prefix_helpers(self):
        state = {
            "_orig_mod.module.proj.weight": torch.ones(2, 2),
            "module.proj.bias": torch.zeros(2),
        }
        stripped = strip_compile_prefix_from_state_dict(state)
        self.assertIn("proj.weight", stripped)
        self.assertIn("proj.bias", stripped)
        payload = {"model_state_dict": state}
        extracted = extract_model_state_dict(payload)
        self.assertEqual(set(extracted), {"proj.weight", "proj.bias"})


if __name__ == "__main__":
    unittest.main()

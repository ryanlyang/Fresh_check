import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.dualview_part.anchor import (
        HLTPartAnchorConfig,
        HLTPartSummaryEncoder,
        build_hlt_part_anchor,
        load_hlt_part_anchor,
        strip_compile_prefix_from_state_dict,
    )
else:  # pragma: no cover - environment dependent
    torch = None


if TORCH_AVAILABLE:

    class DummyPartModel(torch.nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.config = {"num_classes": int(num_classes), "input_dim": 17}
            self.proj = torch.nn.Linear(17, int(num_classes))

        def forward(self, points, features, lorentz_vectors, mask):
            del points, lorentz_vectors
            token_mask = mask[:, 0, :].float()
            denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (features * token_mask[:, None, :]).sum(dim=2) / denom
            return self.proj(pooled)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DualViewPartStep2AnchorTests(unittest.TestCase):
    def make_inputs(self, batch_size=3, n_tokens=5):
        features = torch.randn(batch_size, 17, n_tokens)
        mask = torch.ones(batch_size, 1, n_tokens, dtype=torch.bool)
        mask[-1, :, -2:] = False
        return {
            "points": torch.randn(batch_size, 2, n_tokens),
            "features": features,
            "lorentz_vectors": torch.randn(batch_size, 4, n_tokens),
            "mask": mask,
        }

    def test_summary_encoder_is_mask_safe_and_finite(self):
        encoder = HLTPartSummaryEncoder(input_dim=17, hidden_dim=16, context_dim=7)
        inputs = self.make_inputs(batch_size=2, n_tokens=4)
        context, summary = encoder(inputs["features"], inputs["mask"])

        self.assertEqual(tuple(context.shape), (2, 7))
        self.assertEqual(tuple(summary.shape), (2, 35))
        self.assertTrue(bool(torch.isfinite(context).all()))
        self.assertTrue(bool(torch.isfinite(summary).all()))

    def test_build_anchor_freezes_model_but_keeps_summary_trainable(self):
        cfg = HLTPartAnchorConfig(num_classes=2, context_dim=8, summary_hidden_dim=16, freeze_anchor=True)
        anchor = build_hlt_part_anchor(DummyPartModel(num_classes=2), config=cfg)
        output = anchor.forward_inputs(self.make_inputs())

        self.assertEqual(tuple(output.logits.shape), (3, 2))
        self.assertEqual(tuple(output.context.shape), (3, 8))
        self.assertTrue(anchor.anchor_parameters_frozen())
        self.assertGreater(anchor.trainable_parameter_count(), 0)
        self.assertTrue(output.diagnostics["anchor_context_available"])

    def test_freeze_all_parameters_freezes_summary_too(self):
        cfg = HLTPartAnchorConfig(num_classes=2, context_dim=8, summary_hidden_dim=16, freeze_anchor=True)
        anchor = build_hlt_part_anchor(DummyPartModel(num_classes=2), config=cfg)
        anchor.freeze_all_parameters()
        self.assertEqual(anchor.trainable_parameter_count(), 0)

    def test_strip_compile_prefix_from_state_dict(self):
        stripped = strip_compile_prefix_from_state_dict(
            {
                "_orig_mod.proj.weight": torch.ones(2, 2),
                "_orig_mod.proj.bias": torch.zeros(2),
            }
        )
        self.assertIn("proj.weight", stripped)
        self.assertIn("proj.bias", stripped)

    def test_load_hlt_part_anchor_uses_checkpoint_payload(self):
        model = DummyPartModel(num_classes=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "best_model_val.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 2,
                    "label_names": ["QCD", "Hgg"],
                    "hlt_degradation_strength": 0.6,
                    "model_config": {"num_classes": 2, "input_dim": 17},
                    "epoch": 3,
                },
                checkpoint,
            )
            with patch(
                "teacher_logit_reco.dualview_part.anchor._build_hlt_part_model_from_payload",
                return_value=DummyPartModel(num_classes=2),
            ):
                anchor = load_hlt_part_anchor(
                    checkpoint,
                    device="cpu",
                    context_dim=6,
                    summary_hidden_dim=12,
                    freeze_anchor=True,
                )

        output = anchor.forward_inputs(self.make_inputs(batch_size=2))
        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.context.shape), (2, 6))
        self.assertTrue(anchor.anchor_parameters_frozen())
        self.assertEqual(anchor.config.label_names, ("QCD", "Hgg"))
        self.assertEqual(anchor.metadata()["payload_epoch"], 3)

    def test_load_hlt_part_anchor_rejects_wrong_binary_labels(self):
        model = DummyPartModel(num_classes=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "wrong_labels.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 2,
                    "label_names": ["QCD", "Tbqq"],
                    "hlt_degradation_strength": 0.6,
                    "model_config": {"num_classes": 2, "input_dim": 17},
                },
                checkpoint,
            )
            with patch(
                "teacher_logit_reco.dualview_part.anchor._build_hlt_part_model_from_payload",
                return_value=DummyPartModel(num_classes=2),
            ):
                with self.assertRaises(ValueError):
                    load_hlt_part_anchor(checkpoint, device="cpu")

    def test_load_hlt_part_anchor_rejects_wrong_degradation(self):
        model = DummyPartModel(num_classes=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "wrong_degradation.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 2,
                    "label_names": ["QCD", "Hgg"],
                    "hlt_degradation_strength": 1.0,
                    "model_config": {"num_classes": 2, "input_dim": 17},
                },
                checkpoint,
            )
            with patch(
                "teacher_logit_reco.dualview_part.anchor._build_hlt_part_model_from_payload",
                return_value=DummyPartModel(num_classes=2),
            ):
                with self.assertRaises(ValueError):
                    load_hlt_part_anchor(checkpoint, device="cpu")

    def test_load_hlt_part_anchor_can_explicitly_bypass_contract_for_debug(self):
        model = DummyPartModel(num_classes=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "debug_noncanonical.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 2,
                    "label_names": ["QCD", "Tbqq"],
                    "model_config": {"num_classes": 2, "input_dim": 17},
                },
                checkpoint,
            )
            with patch(
                "teacher_logit_reco.dualview_part.anchor._build_hlt_part_model_from_payload",
                return_value=DummyPartModel(num_classes=2),
            ):
                anchor = load_hlt_part_anchor(
                    checkpoint,
                    device="cpu",
                    enforce_canonical_contract=False,
                )

        self.assertEqual(anchor.config.label_names, ("QCD", "Tbqq"))


if __name__ == "__main__":
    unittest.main()

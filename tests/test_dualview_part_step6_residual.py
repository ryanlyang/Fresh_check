import importlib.util
import math
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.dualview_part import (
        DUALVIEW_PART_GATE_PER_CLASS,
        DUALVIEW_PART_GATE_SCALAR,
        DUALVIEW_PART_RESIDUAL_CONTRACT,
        DUALVIEW_PART_STEP6,
        DualViewResidualParTConfig,
        HLTPartAnchorConfig,
        PNMemoryEncoderConfig,
        build_dualview_residual_part,
        build_hlt_part_anchor,
        build_pn_memory_encoder,
        reliability_feature_dim,
    )
else:  # pragma: no cover - environment dependent
    torch = None


if TORCH_AVAILABLE:

    class DummyPartModel(torch.nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.config = {"num_classes": int(num_classes), "input_dim": len(PF_FEATURE_NAMES)}
            self.proj = torch.nn.Linear(len(PF_FEATURE_NAMES), int(num_classes))

        def forward(self, points, features, lorentz_vectors, mask):
            del points, lorentz_vectors
            token_mask = mask[:, 0, :].float()
            denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (features * token_mask[:, None, :]).sum(dim=2) / denom
            return self.proj(pooled)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DualViewPartStep6ResidualTests(unittest.TestCase):
    def make_hlt_inputs(self, *, batch_size=3, n_tokens=7):
        features = torch.randn(batch_size, len(PF_FEATURE_NAMES), n_tokens)
        mask = torch.ones(batch_size, 1, n_tokens, dtype=torch.bool)
        mask[-1, :, -2:] = False
        return {
            "points": torch.randn(batch_size, 2, n_tokens),
            "features": features,
            "lorentz_vectors": torch.randn(batch_size, 4, n_tokens),
            "mask": mask,
        }

    def make_raw_tokens(self, *, batch_size=3, n_tokens=7, shift=0.0):
        tokens = torch.randn(batch_size, n_tokens, RAW_TOKEN_DIM) * 0.05
        mask = torch.ones(batch_size, n_tokens, dtype=torch.bool)
        mask[-1, -2:] = False
        tokens[:, :, 0] = torch.linspace(20.0, 80.0, n_tokens) + float(shift)
        tokens[:, :, 1] = torch.linspace(-0.4, 0.4, n_tokens)
        tokens[:, :, 2] = torch.linspace(-0.2, 0.2, n_tokens)
        tokens[:, :, 3] = tokens[:, :, 0] * 1.5
        tokens = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return tokens, mask

    def make_model(self, *, gate_mode=DUALVIEW_PART_GATE_SCALAR):
        anchor = build_hlt_part_anchor(
            DummyPartModel(num_classes=2),
            config=HLTPartAnchorConfig(
                num_classes=2,
                context_dim=8,
                summary_hidden_dim=16,
                summary_dropout=0.0,
                freeze_anchor=True,
            ),
        )
        pn_encoder = build_pn_memory_encoder(
            PNMemoryEncoderConfig(
                raw_token_dim=RAW_TOKEN_DIM,
                embed_dim=16,
                num_layers=1,
                num_heads=4,
                dropout=0.0,
                attention_dropout=0.0,
            )
        )
        return build_dualview_residual_part(
            anchor,
            pn_encoder=pn_encoder,
            config=DualViewResidualParTConfig(
                num_classes=2,
                hlt_context_dim=8,
                pn_context_dim=16,
                reliability_dim=reliability_feature_dim(),
                hidden_dim=32,
                num_hidden_layers=1,
                dropout=0.0,
                gate_mode=gate_mode,
                gate_bias_init=-5.0,
            ),
        )

    def make_forward_kwargs(self, *, batch_size=3, n_tokens=7):
        hlt_tokens, hlt_mask = self.make_raw_tokens(batch_size=batch_size, n_tokens=n_tokens)
        pn_tokens, pn_mask = self.make_raw_tokens(batch_size=batch_size, n_tokens=n_tokens, shift=5.0)
        confidence = torch.linspace(0.2, 1.0, n_tokens).expand(batch_size, n_tokens).clone()
        confidence = torch.where(pn_mask, confidence, torch.zeros_like(confidence))
        return {
            "hlt_inputs": self.make_hlt_inputs(batch_size=batch_size, n_tokens=n_tokens),
            "hlt_tokens": hlt_tokens,
            "hlt_mask": hlt_mask,
            "pn_reco_tokens": pn_tokens,
            "pn_reco_mask": pn_mask,
            "pn_reco_confidence": confidence,
        }

    def test_config_validation_and_contract(self):
        cfg = DualViewResidualParTConfig(hlt_context_dim=8, pn_context_dim=16)

        self.assertEqual(cfg.output_contract, DUALVIEW_PART_RESIDUAL_CONTRACT)
        self.assertEqual(cfg.experiment_step, DUALVIEW_PART_STEP6)
        self.assertEqual(cfg.gate_mode, DUALVIEW_PART_GATE_SCALAR)
        self.assertEqual(cfg.gate_dim, 1)
        self.assertEqual(cfg.fusion_input_dim, 8 + 16 + reliability_feature_dim())
        with self.assertRaises(ValueError):
            DualViewResidualParTConfig(gate_mode="wide_open")
        with self.assertRaises(ValueError):
            DualViewResidualParTConfig(hlt_context_dim=0, pn_context_dim=0, reliability_dim=0)

    def test_closed_initialization_matches_hlt_logits(self):
        model = self.make_model()
        model.eval()

        output = model(**self.make_forward_kwargs(), return_diagnostics=True)

        self.assertEqual(tuple(output.logits.shape), (3, 2))
        self.assertEqual(tuple(output.hlt_logits.shape), (3, 2))
        self.assertEqual(tuple(output.delta_logits.shape), (3, 2))
        self.assertEqual(tuple(output.gate.shape), (3, 1))
        self.assertEqual(tuple(output.hlt_context.shape), (3, 8))
        self.assertEqual(tuple(output.pn_context.shape), (3, 16))
        self.assertEqual(tuple(output.reliability_features.shape), (3, reliability_feature_dim()))
        self.assertTrue(bool(torch.isfinite(output.logits).all()))
        self.assertTrue(bool(torch.allclose(output.logits, output.hlt_logits, atol=1.0e-7)))
        self.assertTrue(bool(torch.allclose(output.delta_logits, torch.zeros_like(output.delta_logits), atol=1.0e-7)))
        self.assertAlmostEqual(float(output.gate.detach().mean()), 1.0 / (1.0 + math.exp(5.0)), places=6)
        self.assertEqual(output.diagnostics["output_contract"], DUALVIEW_PART_RESIDUAL_CONTRACT)
        self.assertEqual(output.diagnostics["prediction_changed_fraction"], 0.0)

    def test_per_class_gate_keeps_closed_initialization(self):
        model = self.make_model(gate_mode=DUALVIEW_PART_GATE_PER_CLASS)
        model.eval()

        output = model(**self.make_forward_kwargs(batch_size=2), return_diagnostics=True)

        self.assertEqual(tuple(output.gate.shape), (2, 2))
        self.assertTrue(bool(torch.allclose(output.logits, output.hlt_logits, atol=1.0e-7)))
        self.assertEqual(output.diagnostics["gate_mode"], DUALVIEW_PART_GATE_PER_CLASS)

    def test_forward_batch_accepts_collated_batch_shape(self):
        model = self.make_model()
        batch = self.make_forward_kwargs(batch_size=2, n_tokens=5)
        batch["labels"] = torch.tensor([0, 1], dtype=torch.long)

        output = model.forward_batch(batch)

        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.gate.shape), (2, 1))
        self.assertIs(model.anchor, model.hlt_anchor)
        self.assertFalse(any(key.startswith("anchor.") for key in model.state_dict()))

    def test_backward_updates_residual_head_without_anchor_gradients(self):
        model = self.make_model()
        model.train()
        kwargs = self.make_forward_kwargs(batch_size=4, n_tokens=6)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        output = model(**kwargs)
        loss = torch.nn.functional.cross_entropy(output.logits, labels)
        loss.backward()

        delta_final = list(model.delta_mlp.modules())[-1]
        self.assertIsNotNone(delta_final.weight.grad)
        self.assertGreater(float(delta_final.weight.grad.abs().sum()), 0.0)
        frozen_anchor_grads = [
            param.grad
            for param in model.hlt_anchor.model.parameters()
            if not param.requires_grad
        ]
        self.assertTrue(all(grad is None for grad in frozen_anchor_grads))

    def test_context_dim_mismatch_is_rejected(self):
        anchor = build_hlt_part_anchor(
            DummyPartModel(num_classes=2),
            config=HLTPartAnchorConfig(num_classes=2, context_dim=8, summary_hidden_dim=16),
        )
        pn_encoder = build_pn_memory_encoder(PNMemoryEncoderConfig(raw_token_dim=RAW_TOKEN_DIM, embed_dim=16))
        with self.assertRaises(ValueError):
            build_dualview_residual_part(
                anchor,
                pn_encoder=pn_encoder,
                infer_dims_from_modules=False,
                config=DualViewResidualParTConfig(hlt_context_dim=8, pn_context_dim=12),
            )


if __name__ == "__main__":
    unittest.main()

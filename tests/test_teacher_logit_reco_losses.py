import importlib.util
import math
import unittest

from teacher_logit_reco.losses import TeacherLogitRecoLossConfig

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
    from teacher_logit_reco.global_transformer import (
        GlobalTransformerReconstructor,
        GlobalTransformerReconstructorConfig,
    )
    from teacher_logit_reco.losses import (
        compute_teacher_logit_reco_loss,
        global_transformer_teacher_training_step,
        teacher_kl_loss,
        weak_jet_summary_loss,
    )
    from teacher_logit_reco.teachers import FrozenTeacher, assert_teacher_frozen


class TeacherLogitRecoLossConfigTests(unittest.TestCase):
    def test_config_validation_and_roundtrip(self):
        cfg = TeacherLogitRecoLossConfig(temperature=3.0, ce_weight=0.5)
        self.assertEqual(cfg.to_dict()["temperature"], 3.0)
        self.assertEqual(TeacherLogitRecoLossConfig.from_mapping(cfg.to_dict()).ce_weight, 0.5)
        with self.assertRaises(ValueError):
            TeacherLogitRecoLossConfig(temperature=0.0)
        with self.assertRaises(ValueError):
            TeacherLogitRecoLossConfig(ce_weight=-1.0)


if TORCH_AVAILABLE:
    class TinyFourArgTeacher(torch.nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.proj = torch.nn.Linear(17, num_classes)
            self.config = {"architecture": "pfn", "num_classes": num_classes}

        def forward(self, points, features, lorentz_vectors, mask):
            del points, lorentz_vectors
            valid = mask.float()
            denom = torch.clamp(valid.sum(dim=2), min=1.0)
            pooled = (features * valid).sum(dim=2) / denom
            return self.proj(pooled)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TeacherLogitRecoLossTests(unittest.TestCase):
    def make_batch(self):
        tokens = torch.zeros((3, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        offline = torch.zeros_like(tokens)
        mask = torch.zeros((3, 5), dtype=torch.bool)
        mask[:, :3] = True
        for b in range(3):
            for i in range(3):
                pt = 8.0 + b + i
                eta = 0.05 * (i + 1)
                phi = -0.15 + 0.1 * i
                tokens[b, i, 0] = pt
                tokens[b, i, 1] = eta
                tokens[b, i, 2] = phi
                tokens[b, i, 3] = pt * math.cosh(eta) + 0.2
                tokens[b, i, 4] = 1.0
                tokens[b, i, 5 + (i % 5)] = 1.0
                tokens[b, i, 10:14] = torch.tensor([0.1, 0.01, -0.2, 0.02])
                offline[b, i] = tokens[b, i]
                offline[b, i, 0] *= 1.08
                offline[b, i, 3] *= 1.08
        labels = torch.tensor([0, 1, 3], dtype=torch.long)
        return tokens, mask, offline, mask.clone(), labels

    def make_reconstructor(self):
        torch.manual_seed(10)
        return GlobalTransformerReconstructor(
            GlobalTransformerReconstructorConfig(
                hidden_dim=32,
                num_heads=4,
                num_layers=1,
                num_extra_candidates=2,
                dropout=0.0,
            )
        )

    def make_teacher(self):
        torch.manual_seed(20)
        return FrozenTeacher(model=TinyFourArgTeacher(), architecture="pfn", device=torch.device("cpu"))

    def test_teacher_kl_is_near_zero_for_matching_logits(self):
        logits = torch.randn(5, 4)
        loss = teacher_kl_loss(logits, logits.clone(), temperature=2.0)
        self.assertLess(float(loss), 1.0e-6)

    def test_combined_loss_is_finite(self):
        teacher = self.make_teacher()
        model = self.make_reconstructor()
        hlt_tokens, hlt_mask, offline_tokens, offline_mask, labels = self.make_batch()
        reco_view = model(hlt_tokens, hlt_mask, labels=labels)
        with torch.no_grad():
            offline_logits = teacher.forward_view_no_grad(offline_tokens, offline_mask)
        reco_logits = teacher.forward_soft_view(reco_view)
        loss = compute_teacher_logit_reco_loss(
            offline_logits=offline_logits,
            reco_logits=reco_logits,
            labels=labels,
            reco_view=reco_view,
            offline_tokens=offline_tokens,
            offline_mask=offline_mask,
        )
        self.assertTrue(bool(torch.isfinite(loss.total_loss)))
        self.assertGreaterEqual(float(loss.components["correction_budget"]), 0.0)
        self.assertGreaterEqual(float(loss.components["jet_summary"]), 0.0)
        self.assertIn("total_loss", loss.detached_float_dict())

    def test_training_step_sends_gradients_to_reconstructor_not_teacher(self):
        teacher = self.make_teacher()
        model = self.make_reconstructor()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        hlt_tokens, hlt_mask, offline_tokens, offline_mask, labels = self.make_batch()
        loss, _, _, _ = global_transformer_teacher_training_step(
            reconstructor=model,
            teacher=teacher,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            offline_tokens=offline_tokens,
            offline_mask=offline_mask,
            labels=labels,
            optimizer=optimizer,
        )
        self.assertTrue(bool(torch.isfinite(loss.total_loss.detach())))
        model_grad = sum(
            float(param.grad.detach().abs().sum())
            for param in model.parameters()
            if param.grad is not None
        )
        self.assertGreater(model_grad, 0.0)
        assert_teacher_frozen(teacher)
        for param in teacher.model.parameters():
            self.assertIsNone(param.grad)

    def test_weak_summary_loss_prefers_closer_view(self):
        teacher = self.make_teacher()
        del teacher
        model = self.make_reconstructor()
        hlt_tokens, hlt_mask, offline_tokens, offline_mask, labels = self.make_batch()
        reco_view = model(hlt_tokens, hlt_mask, labels=labels)
        far = weak_jet_summary_loss(reco_view, offline_tokens * 2.0, offline_mask)
        close = weak_jet_summary_loss(reco_view, reco_view.tokens.detach(), reco_view.mask)
        self.assertLess(float(close), float(far))


if __name__ == "__main__":
    unittest.main()

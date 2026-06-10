import importlib.util
import unittest

import numpy as np

from teacher_logit_reco.teachers import (
    FrozenTeacher,
    assert_teacher_frozen,
    infer_teacher_architecture,
    normalize_teacher_architecture,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


class TeacherArchitectureInferenceTests(unittest.TestCase):
    def test_normalizes_teacher_architecture_aliases(self):
        self.assertEqual(normalize_teacher_architecture("ParticleNet"), "pn")
        self.assertEqual(normalize_teacher_architecture("ParticleTransformer"), "part")
        self.assertEqual(normalize_teacher_architecture(None), "part")

    def test_infers_part_for_historical_checkpoints_without_architecture(self):
        payload = {"model_config": {"input_dim": 17, "num_classes": 10}}
        self.assertEqual(infer_teacher_architecture(payload), "part")

    def test_infers_explicit_architecture_from_model_config(self):
        payload = {"model_config": {"architecture": "pcnn"}}
        self.assertEqual(infer_teacher_architecture(payload), "pcnn")


if TORCH_AVAILABLE:
    class TinyFourArgTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))
            self.config = {"architecture": "pfn", "num_classes": 3}

        def forward(self, points, features, lorentz_vectors, mask):
            del points, lorentz_vectors
            pooled = (features * mask.float()).sum(dim=(1, 2))
            return torch.stack([pooled, -pooled, pooled * self.scale], dim=1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class FrozenTeacherTests(unittest.TestCase):
    def test_freezes_parameters_but_keeps_input_gradients(self):
        model = TinyFourArgTeacher()
        teacher = FrozenTeacher(
            model=model,
            architecture="pfn",
            device=torch.device("cpu"),
            max_constits=8,
        )
        assert_teacher_frozen(teacher)
        self.assertEqual(teacher.trainable_parameter_count(), 0)

        tokens = torch.zeros((2, 4, 14), dtype=torch.float32, requires_grad=True)
        mask = torch.zeros((2, 4), dtype=torch.bool)
        mask[:, :2] = True
        with torch.no_grad():
            tokens[:, :2, 0] = torch.tensor([[10.0, 5.0], [7.0, 3.0]])
            tokens[:, :2, 1] = 0.1
            tokens[:, :2, 2] = 0.2
            tokens[:, :2, 3] = tokens[:, :2, 0] * torch.cosh(tokens[:, :2, 1])
            tokens[:, :2, 5] = 1.0
        logits = teacher.forward_view(tokens, mask)
        self.assertEqual(tuple(logits.shape), (2, 3))
        loss = logits[:, 0].sum()
        loss.backward()
        self.assertIsNotNone(tokens.grad)
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)
        for param in teacher.model.parameters():
            self.assertIsNone(param.grad)

    def test_forward_numpy_view(self):
        teacher = FrozenTeacher(
            model=TinyFourArgTeacher(),
            architecture="pfn",
            device=torch.device("cpu"),
        )
        tokens = np.zeros((1, 3, 14), dtype=np.float32)
        mask = np.zeros((1, 3), dtype=bool)
        mask[:, :1] = True
        tokens[:, 0, 0] = 4.0
        tokens[:, 0, 3] = 4.1
        logits = teacher.forward_view_no_grad(tokens, mask)
        self.assertEqual(tuple(logits.shape), (1, 3))


if __name__ == "__main__":
    unittest.main()

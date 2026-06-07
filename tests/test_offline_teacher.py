import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from jetclass_fresh.hlt_baseline import ParticleViewTorchDataset, make_data_loader
    from jetclass_fresh.offline_teacher import OfflineTeacherTrainConfig, train_offline_teacher
else:  # pragma: no cover - environment dependent
    torch = None


def make_offline_view(split="model_train", n_jets=6):
    tokens = np.zeros((n_jets, 5, 14), dtype=np.float32)
    mask = np.zeros((n_jets, 5), dtype=bool)
    labels = np.zeros((n_jets,), dtype=np.int64)
    for jet_index in range(n_jets):
        mask[jet_index, :3] = True
        for part_index in range(3):
            pt = 3.0 + part_index + 0.2 * jet_index
            eta = -0.05 * part_index
            phi = 0.20 * part_index
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta)
            tokens[jet_index, part_index, 4] = 1.0
            tokens[jet_index, part_index, 5] = 1.0

    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=[JetIdentity(file="offline.root", entry=i, label=0) for i in range(n_jets)],
        split=split,
        metadata={
            "view": "offline",
            "source_manifest_hash": "synthetic_manifest_hash",
        },
    )


if TORCH_AVAILABLE:
    class ConstantClassZero(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(num_classes))

        def forward(self, points, features, lorentz_vectors, mask):
            batch_size = features.shape[0]
            del points, features, lorentz_vectors, mask
            return self.logits.unsqueeze(0).expand(batch_size, -1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class OfflineTeacherStep6Tests(unittest.TestCase):
    def test_offline_loader_collate_uses_offline_source_view(self):
        dataset = ParticleViewTorchDataset(make_offline_view(n_jets=3), expected_view="offline")
        loader = make_data_loader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            seed=707,
            source_view="offline",
        )
        batch = next(iter(loader))
        self.assertEqual(batch["features"].shape, (2, 17, 5))
        self.assertEqual(batch["mask"].shape, (2, 1, 5))
        self.assertEqual(batch["labels"].tolist(), [0, 0])

    def test_train_offline_teacher_smoke_with_injected_model(self):
        train_view = make_offline_view("model_train", n_jets=6)
        val_view = make_offline_view("model_val", n_jets=4)
        with tempfile.TemporaryDirectory() as tmp:
            config = OfflineTeacherTrainConfig(
                output_dir=tmp,
                manifest_path="/unused",
                epochs=1,
                batch_size=2,
                device="cpu",
                amp=False,
                early_stop_patience=2,
            )
            report = train_offline_teacher(
                config,
                model=ConstantClassZero(),
                train_view=train_view,
                val_view=val_view,
            )
            self.assertTrue((Path(tmp) / "best_model_val.pt").exists())
            self.assertTrue((Path(tmp) / "model_val_report.json").exists())

        self.assertEqual(report["experiment_step"], "step6_offline_teacher_reference")
        self.assertEqual(report["reference_role"], "offline_upper_reference_only")
        self.assertTrue(report["not_allowed_for_fusion_features"])
        self.assertTrue(report["no_final_test_evaluation"])
        self.assertEqual(report["best_model_val_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

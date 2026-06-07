import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from jetclass_fresh.hlt_baseline import (
        HLTBaselineTrainConfig,
        JetViewTorchDataset,
        collate_particle_transformer_batch,
        run_epoch,
        train_hlt_baseline,
    )
else:  # pragma: no cover - environment dependent
    torch = None


def make_fixed_hlt_view(n_jets=6):
    tokens = np.zeros((n_jets, 5, 14), dtype=np.float32)
    mask = np.zeros((n_jets, 5), dtype=bool)
    labels = np.zeros((n_jets,), dtype=np.int64)
    for jet_index in range(n_jets):
        mask[jet_index, :3] = True
        for part_index in range(3):
            pt = 2.0 + part_index + 0.1 * jet_index
            eta = 0.05 * part_index
            phi = 0.15 * part_index
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
        jet_ids=[JetIdentity(file="hlt.root", entry=i, label=0) for i in range(n_jets)],
        split="model_train",
        metadata={
            "view": "fixed_hlt",
            "hlt_content_hash": "synthetic_hash",
            "seed": 1053,
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
class HLTBaselineStep5Tests(unittest.TestCase):
    def test_dataset_collate_builds_torch_part_inputs(self):
        dataset = JetViewTorchDataset(make_fixed_hlt_view(3))
        batch = collate_particle_transformer_batch([dataset[0], dataset[1]])

        self.assertEqual(batch["points"].shape, (2, 2, 5))
        self.assertEqual(batch["features"].shape, (2, 17, 5))
        self.assertEqual(batch["lorentz_vectors"].shape, (2, 4, 5))
        self.assertEqual(batch["mask"].shape, (2, 1, 5))
        self.assertEqual(batch["labels"].tolist(), [0, 0])
        self.assertEqual(batch["features"].dtype, torch.float32)
        self.assertEqual(batch["labels"].dtype, torch.int64)

    def test_run_epoch_evaluates_loss_and_accuracy(self):
        dataset = JetViewTorchDataset(make_fixed_hlt_view(4))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_particle_transformer_batch,
        )
        model = ConstantClassZero()
        metrics = run_epoch(
            model,
            loader,
            device=torch.device("cpu"),
            criterion=torch.nn.CrossEntropyLoss(),
        )
        self.assertEqual(metrics["n_jets"], 4)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["loss"], 0.0)

    def test_train_hlt_baseline_smoke_with_injected_model(self):
        train_view = make_fixed_hlt_view(6)
        val_view = make_fixed_hlt_view(4)
        val_view.split = "model_val"
        with tempfile.TemporaryDirectory() as tmp:
            config = HLTBaselineTrainConfig(
                output_dir=tmp,
                cache_dir="/unused",
                epochs=1,
                batch_size=2,
                device="cpu",
                amp=False,
                early_stop_patience=2,
            )
            report = train_hlt_baseline(
                config,
                model=ConstantClassZero(),
                train_view=train_view,
                val_view=val_view,
            )
            self.assertTrue((Path(tmp) / "best_model_val.pt").exists())
            self.assertTrue((Path(tmp) / "model_val_report.json").exists())

        self.assertEqual(report["experiment_step"], "step5_single_hlt_baseline")
        self.assertTrue(report["no_final_test_evaluation"])
        self.assertEqual(report["best_model_val_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

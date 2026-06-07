import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.dual_view import DualViewTaggerTrainConfig
from jetclass_fresh.jetclass_data import JetIdentity, JetView
from jetclass_fresh.reconstructor import RECONSTRUCTOR_VARIANT_NAMES

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from jetclass_fresh.dual_view import (
        HLTTokenDataset,
        build_part_inputs_torch,
        make_hlt_token_loader,
        run_dual_view_epoch,
        train_dual_view_tagger,
    )
    from jetclass_fresh.reconstructor import ReconstructionOutput
else:  # pragma: no cover - environment dependent
    torch = None


def make_hlt_view(split="model_train", n_jets=6, n_constits=5):
    tokens = np.zeros((n_jets, n_constits, 14), dtype=np.float32)
    mask = np.zeros((n_jets, n_constits), dtype=bool)
    labels = np.zeros((n_jets,), dtype=np.int64)
    for jet_index in range(n_jets):
        mask[jet_index, :3] = True
        for part_index in range(3):
            pt = 2.0 + part_index + 0.1 * jet_index
            eta = 0.03 * part_index
            phi = 0.18 * part_index
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
        split=split,
        metadata={"view": "fixed_hlt", "hlt_content_hash": f"{split}_hash"},
    )


class DualViewStep8ConfigTests(unittest.TestCase):
    def test_config_defaults_to_m2_base_and_model_val_only(self):
        cfg = DualViewTaggerTrainConfig(
            output_dir="/tmp/out",
            hlt_cache_dir="/tmp/hlt",
            reconstructor_checkpoint="/tmp/reco.pt",
        )
        self.assertEqual(cfg.variant, "m2_base")
        self.assertEqual(cfg.train_split, "model_train")
        self.assertEqual(cfg.val_split, "model_val")
        self.assertEqual(cfg.max_constits, 128)
        self.assertIn("m2_antioverlap", RECONSTRUCTOR_VARIANT_NAMES)


if TORCH_AVAILABLE:
    class DummyReconstructor(torch.nn.Module):
        def forward(self, hlt_tokens, hlt_mask):
            weights = hlt_mask.float()
            return ReconstructionOutput(
                tokens=hlt_tokens,
                weights=weights,
                candidate_mask=hlt_mask,
                edited_tokens=hlt_tokens,
                split_tokens=hlt_tokens,
                generated_tokens=hlt_tokens[:, :0],
                edited_weights=weights,
                split_weights=weights,
                generated_weights=weights[:, :0],
                total_count_pred=weights.sum(dim=1),
                added_count_pred=torch.zeros_like(weights.sum(dim=1)),
            )

    class DummyDualTagger(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(num_classes))

        def forward(self, hlt_inputs, reco_inputs):
            batch_size = hlt_inputs["features"].shape[0]
            self._last_hlt_shape = tuple(hlt_inputs["features"].shape)
            self._last_reco_shape = tuple(reco_inputs["features"].shape)
            return self.logits.unsqueeze(0).expand(batch_size, -1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DualViewStep8TorchTests(unittest.TestCase):
    def test_build_part_inputs_torch_topk_reco_candidates(self):
        tokens = torch.zeros(2, 7, 14)
        mask = torch.ones(2, 7, dtype=torch.bool)
        weights = torch.linspace(0.1, 0.7, steps=7).unsqueeze(0).repeat(2, 1)
        for index in range(7):
            tokens[:, index, 0] = float(index + 1)
            tokens[:, index, 1] = 0.01 * index
            tokens[:, index, 2] = 0.1 * index
            tokens[:, index, 3] = float(index + 1)
            tokens[:, index, 5] = 1.0

        inputs = build_part_inputs_torch(tokens, mask, weights=weights, max_constits=4)
        self.assertEqual(inputs["features"].shape, (2, 17, 4))
        self.assertEqual(inputs["lorentz_vectors"].shape, (2, 4, 4))
        self.assertEqual(inputs["mask"].shape, (2, 1, 4))
        self.assertTrue(torch.all(inputs["mask"]))

    def test_build_part_inputs_torch_sanitizes_reco_four_vectors(self):
        tokens = torch.zeros(1, 3, 14)
        mask = torch.ones(1, 3, dtype=torch.bool)
        tokens[0, 0, 0] = 10.0
        tokens[0, 0, 1] = 5.0
        tokens[0, 0, 2] = 0.2
        tokens[0, 0, 3] = 1.0
        tokens[0, 0, 5] = 1.0
        tokens[0, 1, 0] = float("nan")
        tokens[0, 1, 3] = 4.0
        tokens[0, 2, 0] = 3.0
        tokens[0, 2, 1] = 0.1
        tokens[0, 2, 2] = -0.2
        tokens[0, 2, 3] = 3.0
        weights = torch.tensor([[1.0, 1.0, float("nan")]])

        inputs = build_part_inputs_torch(tokens, mask, weights=weights, max_constits=3)
        vectors = inputs["lorentz_vectors"]
        out_mask = inputs["mask"]
        self.assertTrue(torch.isfinite(inputs["features"]).all())
        self.assertTrue(torch.isfinite(vectors).all())
        self.assertEqual(int(out_mask.sum().item()), 1)
        px, py, pz, energy = vectors[0, :, 0]
        momentum = torch.sqrt(px * px + py * py + pz * pz)
        self.assertGreaterEqual(float(energy + 1.0e-6), float(momentum))

    def test_run_dual_view_epoch_with_frozen_dummy_reconstructor(self):
        dataset = HLTTokenDataset(make_hlt_view(n_jets=4))
        loader = make_hlt_token_loader(dataset, batch_size=2, shuffle=False, num_workers=0, seed=909)
        tagger = DummyDualTagger()
        metrics = run_dual_view_epoch(
            tagger,
            DummyReconstructor(),
            loader,
            device=torch.device("cpu"),
            criterion=torch.nn.CrossEntropyLoss(),
            max_constits=5,
        )
        self.assertEqual(metrics["n_jets"], 4)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(tagger._last_hlt_shape, (2, 17, 5))
        self.assertEqual(tagger._last_reco_shape, (2, 17, 5))

    def test_train_dual_view_tagger_smoke_with_injected_models(self):
        train_view = make_hlt_view("model_train", n_jets=6)
        val_view = make_hlt_view("model_val", n_jets=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = DualViewTaggerTrainConfig(
                output_dir=tmp,
                hlt_cache_dir="/unused",
                reconstructor_checkpoint="/unused",
                epochs=1,
                batch_size=2,
                device="cpu",
                amp=False,
                early_stop_patience=2,
                max_constits=5,
            )
            report = train_dual_view_tagger(
                cfg,
                tagger=DummyDualTagger(),
                reconstructor=DummyReconstructor(),
                train_view=train_view,
                val_view=val_view,
            )
            self.assertTrue((Path(tmp) / "best_model_val.pt").exists())
            self.assertTrue((Path(tmp) / "model_val_report.json").exists())
        self.assertEqual(report["experiment_step"], "step8_dual_view_tagger")
        self.assertEqual(report["variant"], "m2_base")
        self.assertTrue(report["no_final_test_evaluation"])
        self.assertTrue(report["reconstructor_frozen"])
        self.assertEqual(report["best_model_val_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

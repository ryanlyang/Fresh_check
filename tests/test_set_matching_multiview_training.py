import importlib.util
import unittest

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.set_matching.data import SetMatchingJetDataset, make_set_matching_loader
from teacher_logit_reco.set_matching.reconstructors import SetMatchingReconstructorOutput
from teacher_logit_reco.set_matching.train import (
    SetMatchingReconstructorTrainConfig,
    compute_core_normalization_from_dataset,
    run_set_matching_reco_epoch,
)
from teacher_logit_reco.views import PairedJetViews


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def make_view(*, view_name, n_jets=3, n_parts=5, offset=0.0, split="model_train"):
    tokens = np.zeros((n_jets, n_parts, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, n_parts), dtype=bool)
    labels = np.arange(n_jets, dtype=np.int64) % 10
    for jet_index in range(n_jets):
        valid = min(n_parts, 2 + jet_index)
        mask[jet_index, :valid] = True
        for part_index in range(valid):
            pt = 10.0 + offset + jet_index + part_index
            eta = 0.05 * (jet_index - part_index)
            phi = 0.2 * part_index
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta)
            tokens[jet_index, part_index, 4] = 1.0
            tokens[jet_index, part_index, 5 + (part_index % 5)] = 1.0
            tokens[jet_index, part_index, 10:14] = np.array([0.1, 0.01, -0.2, 0.02], dtype=np.float32)
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=[
            JetIdentity(file=f"ZJetsToNuNu_{index % 2}.root", entry=index, label=int(label))
            for index, label in enumerate(labels)
        ],
        split=split,
        metadata={"view": view_name},
    )


class SetMatchingTrainConfigTests(unittest.TestCase):
    def test_config_requires_explicit_split_confirmation(self):
        with self.assertRaises(ValueError):
            SetMatchingReconstructorTrainConfig(
                output_dir="out",
                manifest_path="manifest.json.gz",
                hlt_cache_dir="hlt_cache",
                architecture="particle_net",
            )

    def test_config_normalizes_architecture_aliases(self):
        cfg = SetMatchingReconstructorTrainConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt_cache",
            architecture="particle_net",
            confirm_split_settings=True,
        )
        self.assertEqual(cfg.architecture, "pn")
        self.assertEqual(cfg.wrapper_config().architecture, "pn")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SetMatchingEpochTests(unittest.TestCase):
    def make_dataset(self):
        hlt = make_view(view_name="fixed_hlt", n_jets=4, n_parts=5, offset=0.0, split="model_train")
        offline = make_view(view_name="offline", n_jets=4, n_parts=5, offset=0.5, split="model_train")
        return SetMatchingJetDataset(PairedJetViews(hlt=hlt, offline=offline), trim_to_valid=True)

    def test_core_normalization_uses_offline_valid_particles(self):
        dataset = self.make_dataset()
        stats = compute_core_normalization_from_dataset(dataset)
        self.assertEqual(stats["source"], "model_train_offline_core_features")
        self.assertEqual(len(stats["mean"]), 4)
        self.assertEqual(len(stats["std"]), 4)
        self.assertGreater(stats["count"], 0)
        self.assertTrue(np.isfinite(stats["mean"]).all())
        self.assertTrue(np.isfinite(stats["std"]).all())

    def test_epoch_runner_returns_loss_metrics(self):
        class EchoModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(()))

            def forward(self, hlt_tokens, hlt_mask, *, labels=None, jet_ids=None, split="in_memory"):
                del labels, jet_ids, split
                predicted = hlt_tokens.clone()
                predicted[:, :, 0] = predicted[:, :, 0] + self.bias
                logits = torch.where(
                    hlt_mask,
                    torch.full_like(hlt_mask.float(), 4.0),
                    torch.full_like(hlt_mask.float(), -4.0),
                )
                return SetMatchingReconstructorOutput(
                    predicted_features=predicted,
                    existence_logits=logits,
                    candidate_mask=hlt_mask,
                    diagnostics={"output_contract": "test", "candidate_count_mean": float(hlt_mask.float().sum(dim=1).mean())},
                )

        dataset = self.make_dataset()
        loader = make_set_matching_loader(dataset, batch_size=2, shuffle=False, num_workers=0)
        cfg = SetMatchingReconstructorTrainConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt_cache",
            confirm_split_settings=True,
            matched_aux_weight=0.0,
            correction_budget_weight=0.0,
            jet_summary_weight=0.0,
        )
        metrics = run_set_matching_reco_epoch(
            EchoModel(),
            loader,
            device=torch.device("cpu"),
            loss_config=cfg.loss_config(),
            amp=False,
            max_batches=1,
        )
        self.assertEqual(metrics["num_batches"], 1.0)
        self.assertEqual(metrics["num_jets"], 2.0)
        self.assertIn("total", metrics)
        self.assertIn("matched_core_loss", metrics)
        self.assertTrue(np.isfinite(metrics["total"]))


if __name__ == "__main__":
    unittest.main()

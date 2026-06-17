import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.set_matching.cache import (
    SET_MATCHING_CACHE_STEP,
    SetMatchingRecoViewCacheConfig,
    cache_set_matching_reco_views,
)
from teacher_logit_reco.set_matching.data import SetMatchingJetDataset
from teacher_logit_reco.set_matching.reconstructors import SetMatchingReconstructorOutput
from teacher_logit_reco.views import PairedJetViews


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def make_view(*, view_name, n_jets=3, n_parts=5, offset=0.0, split="stack_val"):
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


class SetMatchingRecoViewCacheConfigTests(unittest.TestCase):
    def test_final_test_cache_requires_explicit_confirmation(self):
        with self.assertRaises(ValueError):
            SetMatchingRecoViewCacheConfig(
                output_dir="out",
                manifest_path="manifest.json.gz",
                hlt_cache_dir="hlt_cache",
                reconstructor_checkpoint="best_model_val.pt",
                architecture="particle_net",
                splits=("stack_val", "final_test"),
            )

    def test_config_normalizes_architecture_and_split_aliases(self):
        cfg = SetMatchingRecoViewCacheConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt_cache",
            reconstructor_checkpoint="best_model_val.pt",
            architecture="particle_net",
            splits=("stack_train", "stack_val"),
        )
        self.assertEqual(cfg.architecture, "pn")
        self.assertEqual(cfg.splits, ("stack_train", "stack_val"))
        self.assertEqual(
            cfg.cache_path("p-cnn", "stack_val"),
            Path("out") / "reconstructed_views" / "pcnn" / "stack_val_reconstructed_view.npz",
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SetMatchingRecoViewCacheTests(unittest.TestCase):
    def test_cache_writer_saves_npz_metadata_and_alignment_hash(self):
        class EchoReco(torch.nn.Module):
            architecture = "gt"

            def forward(self, hlt_tokens, hlt_mask, *, labels=None, jet_ids=None, split="in_memory"):
                del labels, jet_ids, split
                predicted = hlt_tokens.clone()
                predicted[:, :, 0] = predicted[:, :, 0] + 0.25
                logits = torch.where(
                    hlt_mask,
                    torch.full_like(hlt_mask.float(), 4.0),
                    torch.full_like(hlt_mask.float(), -4.0),
                )
                return SetMatchingReconstructorOutput(
                    predicted_features=predicted,
                    existence_logits=logits,
                    candidate_mask=hlt_mask,
                    candidate_weights=torch.sigmoid(logits),
                    diagnostics={"candidate_count_mean": float(hlt_mask.float().sum(dim=1).mean())},
                )

        hlt = make_view(view_name="fixed_hlt", n_jets=3, n_parts=5, offset=0.0, split="stack_val")
        offline = make_view(view_name="offline", n_jets=3, n_parts=5, offset=0.5, split="stack_val")
        dataset = SetMatchingJetDataset(PairedJetViews(hlt=hlt, offline=offline), trim_to_valid=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "set_matching_multiview_test"
            config = SetMatchingRecoViewCacheConfig(
                output_dir=str(output_dir),
                manifest_path=str(Path(tmpdir) / "missing_manifest.json.gz"),
                hlt_cache_dir=str(Path(tmpdir) / "hlt_cache"),
                reconstructor_checkpoint=str(Path(tmpdir) / "best_model_val.pt"),
                architecture="gt",
                splits=("stack_val",),
                batch_size=2,
                device="cpu",
                amp=False,
            )
            report = cache_set_matching_reco_views(
                config,
                model=EchoReco(),
                checkpoint_payload={
                    "epoch": 3,
                    "experiment_step": "unit_test_checkpoint",
                    "loss_config": {"matched_aux_weight": 0.0, "correction_budget_weight": 0.0},
                },
                datasets={"stack_val": dataset},
            )

            cache_path = Path(report["cache_paths"]["stack_val"])
            metadata_path = Path(report["metadata_paths"]["stack_val"])
            self.assertTrue(cache_path.exists())
            self.assertTrue(metadata_path.exists())
            with np.load(cache_path, allow_pickle=False) as data:
                self.assertEqual(data["tokens"].shape, (3, 5, RAW_TOKEN_DIM))
                self.assertEqual(data["mask"].shape, (3, 5))
                self.assertEqual(data["confidence"].shape, (3, 5))
                self.assertEqual(data["labels"].tolist(), [0, 1, 2])
                self.assertEqual(data["indices"].tolist(), [0, 1, 2])

            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["experiment_step"], SET_MATCHING_CACHE_STEP)
            self.assertEqual(metadata["architecture"], "gt")
            self.assertEqual(metadata["split"], "stack_val")
            self.assertTrue(metadata["alignment_audit"]["ok"])
            self.assertEqual(metadata["n_jets"], 3)
            self.assertIn("cache_content_hash", metadata)
            self.assertIn("heldout_set_matching_metrics", metadata)
            self.assertEqual(metadata["source_checkpoint_epoch"], 3)


if __name__ == "__main__":
    unittest.main()

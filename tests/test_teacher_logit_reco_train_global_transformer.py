import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.train_global_transformer import (
    EXPERIMENT_STEP,
    TeacherLogitGlobalTransformerTrainConfig,
    source_metadata,
    train_teacher_logit_global_transformer_reco,
)
from teacher_logit_reco.views import PairedJetViews

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.teachers import FrozenTeacher


def make_tokens(n_jets=8, n_parts=6):
    tokens = np.zeros((n_jets, n_parts, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, n_parts), dtype=bool)
    labels = np.arange(n_jets, dtype=np.int64) % 4
    for jet_index in range(n_jets):
        n_valid = 2 + (jet_index % 3)
        mask[jet_index, :n_valid] = True
        for part_index in range(n_valid):
            pt = 6.0 + 0.3 * jet_index + part_index
            eta = 0.05 * (part_index + 1)
            phi = -0.2 + 0.08 * part_index
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta) + 0.2
            tokens[jet_index, part_index, 4] = 1.0
            tokens[jet_index, part_index, 5 + (part_index % 5)] = 1.0
            tokens[jet_index, part_index, 10:14] = np.array([0.1, 0.01, -0.2, 0.02], dtype=np.float32)
    return tokens, mask, labels


def make_pair(split="model_train", n_jets=8):
    hlt_tokens, hlt_mask, labels = make_tokens(n_jets=n_jets)
    offline_tokens = hlt_tokens.copy()
    offline_tokens[:, :, 0] *= np.where(hlt_mask, 1.05, 1.0)
    offline_tokens[:, :, 3] *= np.where(hlt_mask, 1.05, 1.0)
    jet_ids = [
        JetIdentity(file=f"{split}_{index // 2}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]
    hlt = JetView(
        tokens=hlt_tokens,
        mask=hlt_mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={"view": "fixed_hlt", "hlt_content_hash": f"{split}_hlt_hash"},
    )
    offline = JetView(
        tokens=offline_tokens,
        mask=hlt_mask.copy(),
        labels=labels.copy(),
        jet_ids=list(jet_ids),
        split=split,
        metadata={"view": "offline"},
    )
    return PairedJetViews(hlt=hlt, offline=offline, metadata={"source_manifest_hash": f"{split}_manifest_hash"})


class TeacherLogitGlobalTransformerTrainConfigTests(unittest.TestCase):
    def test_config_validates_splits_and_positive_values(self):
        with self.assertRaises(ValueError):
            TeacherLogitGlobalTransformerTrainConfig(
                output_dir="out",
                manifest_path="manifest.json.gz",
                hlt_cache_dir="hlt",
                teacher_checkpoint="teacher.pt",
                train_split="stack_train",
            )
        with self.assertRaises(ValueError):
            TeacherLogitGlobalTransformerTrainConfig(
                output_dir="out",
                manifest_path="manifest.json.gz",
                hlt_cache_dir="hlt",
                teacher_checkpoint="teacher.pt",
                batch_size=0,
            )

    def test_model_and_loss_config_factories(self):
        cfg = TeacherLogitGlobalTransformerTrainConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            teacher_checkpoint="teacher.pt",
            hidden_dim=64,
            num_heads=4,
            temperature=3.0,
        )
        self.assertEqual(cfg.model_config().hidden_dim, 64)
        self.assertEqual(cfg.loss_config().temperature, 3.0)

    def test_source_metadata_has_expected_keys(self):
        meta = source_metadata()
        self.assertIn("source_commit", meta)
        self.assertIn("source_status_hash", meta)


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
class TeacherLogitGlobalTransformerTinyTrainTests(unittest.TestCase):
    def test_tiny_training_writes_required_outputs(self):
        train_pair = make_pair(split="model_train", n_jets=8)
        val_pair = make_pair(split="model_val", n_jets=6)
        teacher = FrozenTeacher(
            model=TinyFourArgTeacher(),
            architecture="pfn",
            device=torch.device("cpu"),
            checkpoint_path="synthetic_teacher.pt",
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TeacherLogitGlobalTransformerTrainConfig(
                output_dir=tmp,
                manifest_path="synthetic_manifest.json.gz",
                hlt_cache_dir="synthetic_hlt_cache",
                teacher_checkpoint="synthetic_teacher.pt",
                teacher_architecture="pfn",
                epochs=2,
                batch_size=4,
                max_train_batches=1,
                max_val_batches=1,
                hidden_dim=32,
                num_heads=4,
                num_layers=1,
                num_extra_candidates=2,
                dropout=0.0,
                amp=False,
                early_stop_patience=2,
                device="cpu",
            )
            report = train_teacher_logit_global_transformer_reco(
                cfg,
                teacher=teacher,
                train_pair=train_pair,
                val_pair=val_pair,
            )
            out = Path(tmp)
            self.assertEqual(report["experiment_step"], EXPERIMENT_STEP)
            self.assertTrue((out / "best_model_val.pt").exists())
            self.assertTrue((out / "last.pt").exists())
            self.assertTrue((out / "training_curves.json").exists())
            self.assertTrue((out / "run_report.json").exists())
            self.assertTrue((out / "model_val_report.json").exists())
            payload = torch.load(out / "best_model_val.pt", map_location="cpu")
            self.assertEqual(payload["experiment_step"], EXPERIMENT_STEP)
            self.assertEqual(payload["reconstructor_architecture"], "global_transformer")
            self.assertEqual(payload["teacher_metadata"]["architecture"], "pfn")
            self.assertIn("model_state_dict", payload)


if __name__ == "__main__":
    unittest.main()

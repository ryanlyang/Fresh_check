import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.particle_cnn_reconstructor import PARTICLE_CNN_ORDERING_ASSUMPTION
from teacher_logit_reco.train_particle_cnn import (
    EXPERIMENT_STEP,
    RECONSTRUCTOR_ARCHITECTURE,
    TeacherLogitParticleCnnTrainConfig,
    teacher_logit_particle_cnn_checkpoint_payload,
    train_teacher_logit_particle_cnn_reco,
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


class DummyStateful:
    def state_dict(self):
        return {"dummy": 1}


class TeacherLogitParticleCnnTrainConfigTests(unittest.TestCase):
    def test_config_validates_splits_and_pcnn_shape_values(self):
        with self.assertRaises(ValueError):
            TeacherLogitParticleCnnTrainConfig(
                output_dir="out",
                manifest_path="manifest.json.gz",
                hlt_cache_dir="hlt",
                teacher_checkpoint="teacher.pt",
                train_split="stack_train",
            )
        with self.assertRaises(ValueError):
            TeacherLogitParticleCnnTrainConfig(
                output_dir="out",
                manifest_path="manifest.json.gz",
                hlt_cache_dir="hlt",
                teacher_checkpoint="teacher.pt",
                batch_size=0,
            )
        with self.assertRaises(ValueError):
            TeacherLogitParticleCnnTrainConfig(
                output_dir="out",
                manifest_path="manifest.json.gz",
                hlt_cache_dir="hlt",
                teacher_checkpoint="teacher.pt",
                num_blocks=2,
                kernel_sizes=(5,),
                dilations=(1, 2),
            )
        with self.assertRaises(ValueError):
            TeacherLogitParticleCnnTrainConfig(
                output_dir="out",
                manifest_path="manifest.json.gz",
                hlt_cache_dir="hlt",
                teacher_checkpoint="teacher.pt",
                num_blocks=1,
                kernel_sizes=(4,),
                dilations=(1,),
            )

    def test_model_and_loss_config_factories(self):
        cfg = TeacherLogitParticleCnnTrainConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            teacher_checkpoint="teacher.pt",
            hidden_channels=24,
            num_blocks=2,
            kernel_sizes=(5, 3),
            dilations=(1, 2),
            context_dim=48,
            context_mlp_dims=(64,),
            decoder_dims=(24,),
            slot_dim=12,
            temperature=3.0,
        )
        model_cfg = cfg.model_config()
        self.assertEqual(model_cfg.hidden_channels, 24)
        self.assertEqual(model_cfg.kernel_sizes, (5, 3))
        self.assertEqual(model_cfg.dilations, (1, 2))
        self.assertEqual(model_cfg.context_dim, 48)
        self.assertEqual(model_cfg.context_mlp_dims, (64,))
        self.assertEqual(model_cfg.decoder_dims, (24,))
        self.assertEqual(model_cfg.slot_dim, 12)
        self.assertEqual(cfg.loss_config().temperature, 3.0)
        self.assertEqual(cfg.loss_config().ce_weight, 0.25)
        self.assertEqual(cfg.loss_config().correction_budget_weight, 0.05)

    def test_checkpoint_payload_records_particle_cnn_architecture_and_ordering(self):
        cfg = TeacherLogitParticleCnnTrainConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            teacher_checkpoint="teacher.pt",
            hidden_channels=16,
            num_blocks=1,
            kernel_sizes=(3,),
            dilations=(1,),
        )
        payload = teacher_logit_particle_cnn_checkpoint_payload(
            DummyStateful(),
            DummyStateful(),
            epoch=2,
            config=cfg,
            model_config=cfg.model_config(),
            loss_config=cfg.loss_config(),
            teacher_metadata={"architecture": "part"},
            metrics={"model_val": {"total_loss": 1.0}},
            source={"source_commit": "abc"},
        )
        self.assertEqual(payload["experiment_step"], EXPERIMENT_STEP)
        self.assertEqual(payload["reconstructor_architecture"], RECONSTRUCTOR_ARCHITECTURE)
        self.assertEqual(payload["model_config"]["kernel_sizes"], [3])
        self.assertEqual(payload["model_config"]["reconstructor_architecture"], "particle_cnn")
        self.assertEqual(payload["ordering_assumption"], PARTICLE_CNN_ORDERING_ASSUMPTION)
        self.assertEqual(payload["teacher_metadata"]["architecture"], "part")


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
class TeacherLogitParticleCnnTinyTrainTests(unittest.TestCase):
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
            cfg = TeacherLogitParticleCnnTrainConfig(
                output_dir=tmp,
                manifest_path="synthetic_manifest.json.gz",
                hlt_cache_dir="synthetic_hlt_cache",
                teacher_checkpoint="synthetic_teacher.pt",
                teacher_architecture="pfn",
                epochs=2,
                batch_size=4,
                max_train_batches=1,
                max_val_batches=1,
                hidden_channels=12,
                num_blocks=1,
                kernel_sizes=(3,),
                dilations=(1,),
                context_dim=10,
                context_mlp_dims=(12,),
                decoder_dims=(10,),
                num_extra_candidates=1,
                dropout=0.0,
                amp=False,
                early_stop_patience=2,
                device="cpu",
            )
            report = train_teacher_logit_particle_cnn_reco(
                cfg,
                teacher=teacher,
                train_pair=train_pair,
                val_pair=val_pair,
            )
            out = Path(tmp)
            self.assertEqual(report["experiment_step"], EXPERIMENT_STEP)
            self.assertEqual(report["reconstructor_architecture"], RECONSTRUCTOR_ARCHITECTURE)
            self.assertEqual(report["ordering_assumption"], PARTICLE_CNN_ORDERING_ASSUMPTION)
            self.assertTrue((out / "best_model_val.pt").exists())
            self.assertTrue((out / "last.pt").exists())
            self.assertTrue((out / "training_curves.json").exists())
            self.assertTrue((out / "run_report.json").exists())
            self.assertTrue((out / "model_val_report.json").exists())
            payload = torch.load(out / "best_model_val.pt", map_location="cpu")
            self.assertEqual(payload["experiment_step"], EXPERIMENT_STEP)
            self.assertEqual(payload["reconstructor_architecture"], RECONSTRUCTOR_ARCHITECTURE)
            self.assertEqual(payload["teacher_metadata"]["architecture"], "pfn")
            self.assertEqual(payload["ordering_assumption"], PARTICLE_CNN_ORDERING_ASSUMPTION)
            self.assertIn("model_state_dict", payload)


if __name__ == "__main__":
    unittest.main()

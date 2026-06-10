import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.predict_particle_net import (
    PREDICT_EXPERIMENT_STEP,
    TeacherLogitParticleNetPredictionConfig,
    default_model_name_for_teacher_architecture,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from jetclass_fresh.fusion import load_prediction_block
    from teacher_logit_reco.particle_net_reconstructor import (
        ParticleNetReconstructor,
        ParticleNetReconstructorConfig,
    )
    from teacher_logit_reco.predict_particle_net import (
        collect_teacher_logit_particle_net_predictions,
        load_particle_net_reconstructor_checkpoint,
    )
    from teacher_logit_reco.teachers import FrozenTeacher
    from teacher_logit_reco.train_particle_net import EXPERIMENT_STEP as TRAIN_EXPERIMENT_STEP


def make_hlt_view(split="stack_val", n_jets=5, n_parts=6):
    tokens = np.zeros((n_jets, n_parts, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, n_parts), dtype=bool)
    labels = np.arange(n_jets, dtype=np.int64) % 4
    for jet_index in range(n_jets):
        n_valid = 2 + (jet_index % 3)
        mask[jet_index, :n_valid] = True
        for part_index in range(n_valid):
            pt = 6.0 + jet_index + part_index
            eta = 0.06 * part_index
            phi = -0.2 + 0.1 * part_index
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta) + 0.2
            tokens[jet_index, part_index, 4] = 1.0
            tokens[jet_index, part_index, 5 + (part_index % 5)] = 1.0
            tokens[jet_index, part_index, 10:14] = np.array([0.1, 0.01, -0.2, 0.02], dtype=np.float32)
    jet_ids = [
        JetIdentity(file=f"{split}_{index // 2}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={"view": "fixed_hlt", "hlt_content_hash": f"{split}_hlt_hash"},
    )


class TeacherLogitParticleNetPredictionConfigTests(unittest.TestCase):
    def test_final_test_requires_confirmation(self):
        with self.assertRaises(ValueError):
            TeacherLogitParticleNetPredictionConfig(
                output_dir="out",
                hlt_cache_dir="hlt",
                reconstructor_checkpoint="reco.pt",
            )
        cfg = TeacherLogitParticleNetPredictionConfig(
            output_dir="out",
            hlt_cache_dir="hlt",
            reconstructor_checkpoint="reco.pt",
            confirm_final_test=True,
        )
        self.assertIn("final_test", cfg.splits)

    def test_default_model_name(self):
        self.assertEqual(default_model_name_for_teacher_architecture("ParticleTransformer"), "pn_reco_to_part_teacher")
        self.assertEqual(default_model_name_for_teacher_architecture("pn"), "pn_reco_to_pn_teacher")
        self.assertEqual(default_model_name_for_teacher_architecture("PFN"), "pn_reco_to_pfn_teacher")


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
class TeacherLogitParticleNetPredictionTorchTests(unittest.TestCase):
    def make_reconstructor(self):
        torch.manual_seed(5)
        return ParticleNetReconstructor(
            ParticleNetReconstructorConfig(
                edgeconv_dims=(16,),
                k=3,
                num_extra_candidates=1,
                dropout=0.0,
            )
        )

    def make_teacher(self):
        torch.manual_seed(8)
        return FrozenTeacher(model=TinyFourArgTeacher(), architecture="pfn", device=torch.device("cpu"))

    def test_checkpoint_loader_reconstructs_particle_net_model(self):
        model = self.make_reconstructor()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model_val.pt"
            torch.save(
                {
                    "epoch": 2,
                    "reconstructor_architecture": "particle_net",
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config.to_dict(),
                    "experiment_step": TRAIN_EXPERIMENT_STEP,
                },
                path,
            )
            loaded, payload = load_particle_net_reconstructor_checkpoint(path, device=torch.device("cpu"))
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(payload["reconstructor_architecture"], "particle_net")
            view = make_hlt_view()
            tokens = torch.from_numpy(view.tokens)
            mask = torch.from_numpy(view.mask)
            out = loaded(tokens, mask)
            self.assertEqual(tuple(out.tokens.shape[:2]), (5, 7))

    def test_collect_predictions_writes_fusion_blocks(self):
        model = self.make_reconstructor()
        teacher = self.make_teacher()
        hlt_views = {
            "stack_train": make_hlt_view(split="stack_train", n_jets=4),
            "stack_val": make_hlt_view(split="stack_val", n_jets=4),
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TeacherLogitParticleNetPredictionConfig(
                output_dir=tmp,
                hlt_cache_dir="synthetic_hlt_cache",
                reconstructor_checkpoint="synthetic_pn_reco.pt",
                splits=["stack_train", "stack_val"],
                batch_size=2,
                device="cpu",
                amp=False,
            )
            report = collect_teacher_logit_particle_net_predictions(
                cfg,
                reconstructor=model,
                teacher=teacher,
                hlt_views=hlt_views,
            )
            prediction_dir = Path(report["prediction_dir"])
            model_name = report["model_name"]
            self.assertEqual(report["experiment_step"], PREDICT_EXPERIMENT_STEP)
            self.assertEqual(report["reconstructor_architecture"], "particle_net")
            self.assertEqual(model_name, "pn_reco_to_pfn_teacher")
            loaded = load_prediction_block(prediction_dir, model_name, "stack_val")
            self.assertEqual(loaded.logits.shape, (4, 4))
            self.assertEqual(loaded.metadata["experiment_step"], PREDICT_EXPERIMENT_STEP)
            self.assertEqual(loaded.metadata["model_kind"], "teacher_logit_particle_net_reco")
            self.assertEqual(loaded.metadata["training_step"], TRAIN_EXPERIMENT_STEP)
            self.assertEqual(loaded.metadata["reconstructor_architecture"], "particle_net")
            self.assertEqual(
                loaded.metadata["allowed_inputs"],
                "cached_fixed_hlt_only_then_reconstructed_soft_view_to_frozen_teacher",
            )
            self.assertTrue((Path(tmp) / "prediction_collection_report.json").exists())


if __name__ == "__main__":
    unittest.main()

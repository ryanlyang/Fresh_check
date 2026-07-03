import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.privileged_distill_10class import (
    PD10_DEFAULT_REPRESENTATION_BETA,
    PD10_NUM_CLASSES,
    PD10_REPRESENTATION_DIM,
    PD10_REPRESENTATION_MODE_COSINE,
    PD10_TARGET_FULL_LOGITS,
    PD10_TARGET_FULL_LOGITS_PLUS_REP,
    PD10_TARGET_REP_ONLY,
    PD10_TEACHER_PARTICLE_DUAL_VIEW,
    PD10_TOP_K,
    PD10StudentDistillationDataset,
    PD10StudentRepresentationAdapter,
    PD10StudentTrainConfig,
    PD10TeacherRepresentationCacheConfig,
    build_pd10_teacher_representation_block,
    default_pd10_experiment_layout,
    pd10_effective_representation_beta,
    pd10_extended_student_variant_name,
    pd10_student_loss,
    pd10_student_representation_loss,
    train_pd10_student,
)
import teacher_logit_reco.privileged_distill_10class.students as students_module


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_pd10_student.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("train_pd10_student", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_hlt_view(labels=(0, 1, 2, 3), *, split="model_train") -> JetView:
    labels_np = np.asarray(labels, dtype=np.int64)
    tokens = np.zeros((len(labels_np), 3, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.ones((len(labels_np), 3), dtype=bool)
    for row in range(len(labels_np)):
        tokens[row, :, 0] = float(row + 10)
        tokens[row, :, 1] = 0.01 * float(row + 1)
        tokens[row, :, 2] = 0.02 * float(row + 1)
        tokens[row, :, 3] = float(row + 11)
        tokens[row, :, 5] = 1.0
    jet_ids = [
        JetIdentity(file=f"v2_step4_class{int(label)}.root", entry=1000 + index, label=int(label))
        for index, label in enumerate(labels_np)
    ]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels_np,
        jet_ids=jet_ids,
        split=split,
        metadata={"view": "fixed_hlt", "hlt_content_hash": f"{split}-hlt-hash"},
    )


def make_representation_block(view: JetView, root: Path):
    cfg = PD10TeacherRepresentationCacheConfig(
        teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
        output_dir=str(root / "teacher_representations"),
        splits=(view.split,),
    )
    reps = np.zeros((len(view.labels), PD10_REPRESENTATION_DIM), dtype=np.float32)
    reps[:, 0] = 1.0
    return build_pd10_teacher_representation_block(
        cfg,
        view.split,
        representations=reps,
        labels=view.labels,
        jet_ids=view.jet_ids,
        source_metadata={"hlt_content_hash": "hlt-hash", "source_manifest_hash": "manifest-hash"},
    )


class TinyStudent(require_torch().nn.Module):
    def __init__(self):
        super().__init__()
        torch = require_torch()
        self.classifier = torch.nn.Linear(17, PD10_NUM_CLASSES)
        self.config = {"kind": "tiny_v2_step4_student", "num_classes": PD10_NUM_CLASSES}

    def forward(self, points, features, lorentz_vectors, mask):
        mask_float = mask.float()
        pooled = (features.float() * mask_float).sum(dim=-1) / torch.clamp(mask_float.sum(dim=-1), min=1.0)
        return self.classifier(pooled)


class TinyHiddenStudent(require_torch().nn.Module):
    def __init__(self):
        super().__init__()
        torch = require_torch()
        self.encoder = torch.nn.Linear(17, 8)
        self.classifier = torch.nn.Linear(8, PD10_NUM_CLASSES)
        self.config = {"kind": "tiny_hidden_v2_step4_student", "embed_dim": 8, "num_classes": PD10_NUM_CLASSES}

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        torch = require_torch()
        mask_float = mask.float()
        pooled = (features.float() * mask_float).sum(dim=-1) / torch.clamp(mask_float.sum(dim=-1), min=1.0)
        hidden = torch.tanh(self.encoder(pooled))
        return self.classifier(hidden)


class PD10V2Step4StudentVariantTests(unittest.TestCase):
    def test_v2_student_config_guardrails_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {
                "student_init": "warm_start",
                "teacher_target": "particle_dual_view",
                "output_dir": str(root / "student"),
                "hlt_cache_dir": str(root / "hlt_cache"),
                "baseline_checkpoint": str(root / "baseline.pt"),
                "evaluate_final_test": False,
            }
            logit = PD10StudentTrainConfig(
                **common,
                teacher_logit_cache=str(root / "teacher_logits"),
                target_mode=PD10_TARGET_FULL_LOGITS,
            )
            self.assertTrue(logit.uses_logit_teacher)
            self.assertFalse(logit.uses_representations)
            self.assertEqual(logit.variant_name, "warm_particle_dual_logit_kd")

            rep = PD10StudentTrainConfig(
                **common,
                teacher_representation_cache=str(root / "teacher_representations"),
                target_mode=PD10_TARGET_REP_ONLY,
            )
            self.assertFalse(rep.uses_logit_teacher)
            self.assertTrue(rep.uses_representations)
            self.assertEqual(rep.kd_alpha, 0.0)
            self.assertEqual(rep.representation_beta, PD10_DEFAULT_REPRESENTATION_BETA)
            self.assertEqual(rep.variant_name, "warm_particle_dual_rep_kd")

            both = PD10StudentTrainConfig(
                **common,
                teacher_logit_cache=str(root / "teacher_logits"),
                teacher_representation_cache=str(root / "teacher_representations"),
                target_mode=PD10_TARGET_FULL_LOGITS_PLUS_REP,
            )
            self.assertTrue(both.uses_logit_teacher)
            self.assertTrue(both.uses_representations)
            self.assertEqual(both.logit_target_mode, PD10_TARGET_FULL_LOGITS)
            self.assertEqual(both.variant_name, "warm_particle_dual_logit_rep_kd")

            self.assertEqual(
                pd10_extended_student_variant_name(
                    "warm_start",
                    "dual_view",
                    PD10_TARGET_FULL_LOGITS_PLUS_REP,
                    representation_beta=0.3,
                ),
                "warm_logit_fusion_dual_logit_rep_kd_beta0p3",
            )

            with self.assertRaises(ValueError):
                PD10StudentTrainConfig(**common, target_mode=PD10_TARGET_REP_ONLY)
            with self.assertRaises(ValueError):
                PD10StudentTrainConfig(
                    **common,
                    teacher_logit_cache=str(root / "teacher_logits"),
                    target_mode=PD10_TARGET_FULL_LOGITS_PLUS_REP,
                )
            with self.assertRaises(ValueError):
                PD10StudentTrainConfig(
                    **common,
                    teacher_representation_cache=str(root / "teacher_representations"),
                    target_mode=PD10_TARGET_REP_ONLY,
                    representation_mode="none",
                )

    def test_combined_student_loss_reports_representation_term(self):
        torch = require_torch()
        student_logits = torch.tensor([[0.2, 1.1, -0.3] + [0.0] * 7], dtype=torch.float32)
        teacher_logits = torch.tensor([[1.5, 0.7, -1.0] + [0.0] * 7], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.long)
        student_repr_raw = torch.ones((1, PD10_REPRESENTATION_DIM), dtype=torch.float32, requires_grad=True)
        student_repr = torch.nn.functional.normalize(student_repr_raw, dim=-1)
        teacher_repr = torch.zeros((1, PD10_REPRESENTATION_DIM), dtype=torch.float32)
        teacher_repr[:, 0] = 1.0

        loss, parts = pd10_student_loss(
            student_logits,
            labels,
            teacher_logits=teacher_logits,
            teacher_representations=teacher_repr,
            student_repr_projected=student_repr,
            target_mode=PD10_TARGET_FULL_LOGITS_PLUS_REP,
            kd_alpha=0.5,
            representation_beta=0.1,
            representation_mode=PD10_REPRESENTATION_MODE_COSINE,
            top_k=PD10_TOP_K,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts["kd_loss"], 0.0)
        self.assertGreater(parts["rep_loss"], 0.0)
        self.assertEqual(parts["effective_representation_beta"], 0.1)
        loss.backward()
        self.assertIsNotNone(student_repr_raw.grad)
        self.assertTrue(torch.isfinite(student_repr_raw.grad).all())
        self.assertGreater(float(torch.linalg.vector_norm(student_repr_raw.grad).detach().cpu().item()), 0.0)

    def test_representation_adapter_uses_base_hidden_state_and_backprops_to_encoder(self):
        torch = require_torch()
        base = TinyHiddenStudent()
        wrapped = PD10StudentRepresentationAdapter(base, representation_dim=PD10_REPRESENTATION_DIM)
        features = torch.randn(3, 17, 5)
        mask = torch.ones(3, 1, 5, dtype=torch.bool)
        points = torch.zeros(3, 2, 5)
        lorentz = torch.zeros(3, 4, 5)
        output = wrapped(points, features, lorentz, mask, return_representation=True)
        self.assertEqual(output.student_repr_raw.shape, (3, 8))
        self.assertNotEqual(wrapped.config["representation_source"], "masked_hlt_feature_mean")
        teacher = torch.zeros((3, PD10_REPRESENTATION_DIM), dtype=torch.float32)
        teacher[:, 0] = 1.0
        loss = pd10_student_representation_loss(
            output.student_repr_projected,
            teacher,
            mode=PD10_REPRESENTATION_MODE_COSINE,
        )
        loss.backward()
        self.assertIsNotNone(base.encoder.weight.grad)
        self.assertTrue(torch.isfinite(base.encoder.weight.grad).all())
        self.assertGreater(float(torch.linalg.vector_norm(base.encoder.weight.grad).detach().cpu().item()), 0.0)

    def test_representation_only_training_wraps_student_and_saves_deployable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_view = make_hlt_view(split="model_train")
            val_view = make_hlt_view(split="model_val")
            train_dataset = PD10StudentDistillationDataset(
                train_view,
                teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
                teacher_representation_block=make_representation_block(train_view, root),
            )
            val_dataset = PD10StudentDistillationDataset(
                val_view,
                teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
                teacher_representation_block=make_representation_block(val_view, root),
            )
            cfg = PD10StudentTrainConfig(
                student_init="scratch",
                teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
                output_dir=str(root / "student"),
                hlt_cache_dir=str(root / "hlt_cache"),
                teacher_representation_cache=str(root / "teacher_representations"),
                target_mode=PD10_TARGET_REP_ONLY,
                batch_size=2,
                epochs=1,
                lr=1.0e-3,
                device="cpu",
                amp=False,
                evaluate_final_test=False,
            )

            report = train_pd10_student(
                cfg,
                model=TinyStudent(),
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )

            torch = require_torch()
            payload = torch.load(cfg.checkpoint_path, map_location="cpu")
            self.assertEqual(report["variant_name"], "scratch_particle_dual_rep_kd")
            self.assertTrue(report["uses_representations"])
            self.assertFalse(report["uses_logit_teacher"])
            self.assertGreaterEqual(report["best_model_val_metrics"]["rep_loss"], 0.0)
            self.assertIsNotNone(payload["representation_projector_state_dict"])
            self.assertEqual(payload["deployable_checkpoint_state"], "base_hlt_student_only")
            self.assertFalse(any(key.startswith("base_model.") for key in payload["model_state_dict"]))

            warm_cfg = PD10StudentTrainConfig(
                student_init="scratch",
                teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
                output_dir=str(root / "student2"),
                hlt_cache_dir=str(root / "hlt_cache"),
                teacher_representation_cache=str(root / "teacher_representations"),
                target_mode=PD10_TARGET_REP_ONLY,
                kd_warmup_epochs=2,
                evaluate_final_test=False,
            )
            self.assertAlmostEqual(pd10_effective_representation_beta(warm_cfg, 1), 0.05)
            self.assertAlmostEqual(pd10_effective_representation_beta(warm_cfg, 2), 0.10)

    def test_prediction_dataset_default_is_teacher_cache_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = PD10StudentTrainConfig(
                student_init="scratch",
                teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
                output_dir=str(root / "student"),
                hlt_cache_dir=str(root / "hlt_cache"),
                teacher_logit_cache=str(root / "teacher_logits"),
                teacher_representation_cache=str(root / "teacher_representations"),
                target_mode=PD10_TARGET_FULL_LOGITS_PLUS_REP,
                evaluate_final_test=False,
            )
            sentinel = object()
            calls = []

            def fake_load(*args, **kwargs):
                calls.append((args, kwargs))
                return sentinel

            with patch.object(students_module, "load_pd10_student_dataset", side_effect=fake_load):
                result = students_module._load_teacher_free_prediction_dataset(
                    cfg,
                    "final_test",
                    max_jets=123,
                )

            self.assertIs(result, sentinel)
            self.assertEqual(len(calls), 1)
            _, kwargs = calls[0]
            self.assertEqual(kwargs["teacher_target"], "none")
            self.assertIsNone(kwargs["teacher_logit_dir"])
            self.assertIsNone(kwargs["teacher_representation_dir"])
            self.assertEqual(kwargs["max_jets"], 123)
            self.assertFalse(cfg.align_prediction_to_teacher_cache)

    def test_cli_accepts_v2_teacher_and_representation_flags(self):
        module = load_script_module()
        layout = default_pd10_experiment_layout(output_root="checkpoints")
        args = module.parse_args(
            [
                "--student-init",
                "warm_start",
                "--teacher-target",
                "particle_dual_view",
                "--target-mode",
                "full_logits_plus_rep",
                "--baseline-checkpoint",
                "baseline.pt",
                "--teacher-logit-cache",
                "teacher_logits",
                "--teacher-representation-cache",
                "teacher_representations",
                "--representation-beta",
                "0.3",
                "--representation-dim",
                "256",
                "--representation-mode",
                "cosine",
                "--confirm-final-test",
            ]
        )

        self.assertEqual(args.teacher_target, "particle_dual_view")
        self.assertEqual(args.target_mode, PD10_TARGET_FULL_LOGITS_PLUS_REP)
        self.assertEqual(args.teacher_representation_cache, "teacher_representations")
        self.assertEqual(args.representation_beta, 0.3)
        self.assertEqual(args.representation_dim, PD10_REPRESENTATION_DIM)
        self.assertEqual(args.representation_mode, PD10_REPRESENTATION_MODE_COSINE)
        self.assertFalse(args.align_prediction_to_teacher_cache)

        aligned_args = module.parse_args(
            [
                "--student-init",
                "scratch",
                "--teacher-target",
                "none",
                "--skip-final-test",
                "--align-prediction-to-teacher-cache",
            ]
        )
        self.assertTrue(aligned_args.align_prediction_to_teacher_cache)

        defaults = module.parse_args(
            [
                "--student-init",
                "scratch",
                "--teacher-target",
                "none",
                "--skip-final-test",
            ]
        )
        self.assertEqual(defaults.teacher_representation_cache, str(layout.root / "teacher_representations"))


if __name__ == "__main__":
    unittest.main()

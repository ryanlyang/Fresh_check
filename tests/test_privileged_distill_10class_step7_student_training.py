import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, softmax_np
from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.privileged_distill_10class import (
    PD10_DEFAULT_ALPHA,
    PD10_DEFAULT_TEMPERATURE,
    PD10_NUM_CLASSES,
    PD10_SPLIT_SIZES,
    PD10_STEP7_EXPERIMENT_STEP,
    PD10_STUDENT_DEFAULT_BATCH_SIZE,
    PD10_STUDENT_DEFAULT_SEED,
    PD10_STUDENT_SCRATCH_DEFAULT_EPOCHS,
    PD10_STUDENT_SCRATCH_DEFAULT_KD_WARMUP_EPOCHS,
    PD10_STUDENT_SCRATCH_DEFAULT_LR,
    PD10_STUDENT_PREDICTION_CACHE_CONTRACT,
    PD10_STUDENT_TRAINING_CONTRACT,
    PD10_STUDENT_WARM_START_DEFAULT_EPOCHS,
    PD10_STUDENT_WARM_START_DEFAULT_KD_WARMUP_EPOCHS,
    PD10_STUDENT_WARM_START_DEFAULT_LR,
    PD10_TARGET_CONFIDENCE_WEIGHTED,
    PD10_TARGET_FULL_LOGITS,
    PD10_TARGET_TOP3,
    PD10_TEACHER_NONE,
    PD10_TOP_K,
    PD10StudentDistillationDataset,
    PD10StudentTrainConfig,
    collect_and_save_pd10_student_predictions,
    default_pd10_experiment_layout,
    pd10_effective_kd_alpha,
    pd10_kd_loss,
    pd10_student_checkpoint,
    pd10_student_dir,
    pd10_student_loss,
    pd10_student_prediction_dir,
    pd10_student_variant_name,
    run_pd10_student_epoch,
    train_pd10_student,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_pd10_student.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("train_pd10_student", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_hlt_view(labels=(0, 1, 2, 3), *, split="model_val") -> JetView:
    labels_np = np.asarray(labels, dtype=np.int64)
    tokens = np.zeros((len(labels_np), 3, RAW_TOKEN_DIM), dtype=np.float32)
    for row in range(len(labels_np)):
        tokens[row, :, 0] = float(row + 1)
        tokens[row, :, 1] = 0.01 * float(row + 1)
        tokens[row, :, 2] = 0.02 * float(row + 1)
        tokens[row, :, 3] = float(row + 2)
    mask = np.ones((len(labels_np), 3), dtype=bool)
    jet_ids = [
        JetIdentity(file=f"class{int(label)}.root", entry=100 + index, label=int(label))
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


def make_teacher_block(target: str, view: JetView) -> PredictionBlock:
    labels = np.asarray(view.labels, dtype=np.int64)
    logits = np.full((len(labels), PD10_NUM_CLASSES), -1.0, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 2.0
    model_name = "dual_view_logit_teacher_10class" if target == "dual_view" else f"{target}_part_teacher_10class"
    return PredictionBlock(
        model_name=model_name,
        split=view.split,
        logits=logits,
        probs=softmax_np(logits),
        labels=labels,
        jet_ids=list(view.jet_ids),
        metadata={
            "teacher_target": target,
            "model_name": model_name,
            "split": view.split,
            "num_classes": PD10_NUM_CLASSES,
            "source_view": "fixed_hlt" if target == "hlt" else "offline",
            "student_deployment_inputs": "HLT_only",
            "teacher_logits_train_time_only": True,
        },
    )


class TinyStudent(require_torch().nn.Module):
    def __init__(self):
        super().__init__()
        torch = require_torch()
        self.bias = torch.nn.Parameter(torch.zeros(PD10_NUM_CLASSES))
        self.config = {"kind": "tiny_unit_test_student", "num_classes": PD10_NUM_CLASSES}

    def forward(self, points, features, lorentz_vectors, mask):
        batch_size = int(features.shape[0])
        return self.bias.unsqueeze(0).expand(batch_size, -1)


class PD10Step7StudentTrainingTests(unittest.TestCase):
    def test_student_config_guardrails_defaults_and_paths_are_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ce = PD10StudentTrainConfig(
                student_init="scratch",
                teacher_target="none",
                output_dir=str(root / "ce"),
                hlt_cache_dir=str(root / "hlt_cache"),
                evaluate_final_test=False,
            )

            self.assertEqual(ce.teacher_target, PD10_TEACHER_NONE)
            self.assertFalse(ce.uses_teacher)
            self.assertEqual(ce.kd_alpha, 0.0)
            self.assertEqual(ce.kd_warmup_epochs, 0)
            self.assertEqual(ce.epochs, PD10_STUDENT_SCRATCH_DEFAULT_EPOCHS)
            self.assertEqual(ce.lr, PD10_STUDENT_SCRATCH_DEFAULT_LR)
            self.assertEqual(ce.batch_size, PD10_STUDENT_DEFAULT_BATCH_SIZE)
            self.assertEqual(ce.seed, PD10_STUDENT_DEFAULT_SEED)
            self.assertEqual(ce.variant_name, "pd10_student_scratch_ce_only")
            self.assertEqual(ce.checkpoint_path, root / "ce" / "best_model_val.pt")

            warm = PD10StudentTrainConfig(
                student_init="warm_start",
                teacher_target="dual_view",
                output_dir=str(root / "warm"),
                hlt_cache_dir=str(root / "hlt_cache"),
                teacher_logit_cache=str(root / "teacher_logits"),
                baseline_checkpoint=str(root / "baseline.pt"),
                evaluate_final_test=False,
            )
            self.assertTrue(warm.uses_teacher)
            self.assertEqual(warm.epochs, PD10_STUDENT_WARM_START_DEFAULT_EPOCHS)
            self.assertEqual(warm.lr, PD10_STUDENT_WARM_START_DEFAULT_LR)
            self.assertEqual(warm.kd_warmup_epochs, PD10_STUDENT_WARM_START_DEFAULT_KD_WARMUP_EPOCHS)

            with self.assertRaises(ValueError):
                PD10StudentTrainConfig(
                    student_init="scratch",
                    teacher_target="hlt",
                    output_dir=str(root / "bad"),
                    hlt_cache_dir=str(root / "hlt_cache"),
                    evaluate_final_test=False,
                )
            with self.assertRaises(ValueError):
                PD10StudentTrainConfig(
                    student_init="warm_start",
                    teacher_target="none",
                    output_dir=str(root / "bad"),
                    hlt_cache_dir=str(root / "hlt_cache"),
                    evaluate_final_test=False,
                )
            with self.assertRaises(ValueError):
                PD10StudentTrainConfig(
                    student_init="scratch",
                    teacher_target="none",
                    output_dir=str(root / "bad"),
                    hlt_cache_dir=str(root / "hlt_cache"),
                    target_mode=PD10_TARGET_TOP3,
                    evaluate_final_test=False,
                )
            with self.assertRaises(ValueError):
                PD10StudentTrainConfig(
                    student_init="scratch",
                    teacher_target="none",
                    output_dir=str(root / "bad"),
                    hlt_cache_dir=str(root / "hlt_cache"),
                )
            with self.assertRaises(ValueError):
                PD10StudentTrainConfig(
                    student_init="scratch",
                    teacher_target="none",
                    output_dir=str(root / "bad"),
                    hlt_cache_dir=str(root / "hlt_cache"),
                    evaluate_final_test=False,
                    max_final_test_jets=PD10_SPLIT_SIZES["final_test"] + 1,
                )

            self.assertEqual(
                pd10_student_dir("warm_start", "dual_view", output_root=root),
                default_pd10_experiment_layout(output_root=root).student_dir(
                    pd10_student_variant_name(
                        "warm_start",
                        "dual_view",
                        PD10_TARGET_FULL_LOGITS,
                        temperature=PD10_DEFAULT_TEMPERATURE,
                        kd_alpha=PD10_DEFAULT_ALPHA,
                    )
                ),
            )
            self.assertEqual(
                pd10_student_checkpoint("scratch", output_root=root),
                default_pd10_experiment_layout(output_root=root).student_dir("pd10_student_scratch_ce_only")
                / "best_model_val.pt",
            )

    def test_kd_alpha_warmup_and_loss_modes(self):
        torch = require_torch()
        cfg = PD10StudentTrainConfig(
            student_init="scratch",
            teacher_target="offline",
            output_dir="out",
            hlt_cache_dir="hlt_cache",
            teacher_logit_cache="teacher_logits",
            kd_alpha=0.6,
            kd_warmup_epochs=3,
            evaluate_final_test=False,
        )
        self.assertAlmostEqual(pd10_effective_kd_alpha(cfg, 1), 0.2)
        self.assertAlmostEqual(pd10_effective_kd_alpha(cfg, 3), 0.6)
        self.assertEqual(cfg.kd_warmup_epochs, PD10_STUDENT_SCRATCH_DEFAULT_KD_WARMUP_EPOCHS)

        student_logits = torch.tensor([[0.2, 1.1, -0.3] + [0.0] * 7], dtype=torch.float32)
        teacher_logits = torch.tensor([[1.5, 0.7, -1.0] + [0.0] * 7], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.long)

        ce_only, ce_parts = pd10_student_loss(student_logits, labels)
        self.assertGreater(float(ce_only.item()), 0.0)
        self.assertEqual(ce_parts["kd_loss"], 0.0)

        for mode in (PD10_TARGET_FULL_LOGITS, PD10_TARGET_TOP3, PD10_TARGET_CONFIDENCE_WEIGHTED):
            loss = pd10_kd_loss(
                student_logits,
                teacher_logits,
                temperature=2.0,
                target_mode=mode,
                top_k=PD10_TOP_K,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(float(loss.item()), 0.0)

        uniform_teacher = torch.zeros_like(teacher_logits)
        weighted = pd10_kd_loss(
            student_logits,
            uniform_teacher,
            temperature=2.0,
            target_mode=PD10_TARGET_CONFIDENCE_WEIGHTED,
        )
        self.assertAlmostEqual(float(weighted.item()), 0.0, places=7)

    def test_training_runner_writes_student_reports_and_preserves_hlt_only_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_view = make_hlt_view(labels=(0, 1, 2, 3), split="model_train")
            val_view = make_hlt_view(labels=(0, 1, 2, 3), split="model_val")
            train_dataset = PD10StudentDistillationDataset(
                train_view,
                teacher_target="hlt",
                teacher_block=make_teacher_block("hlt", train_view),
            )
            val_dataset = PD10StudentDistillationDataset(
                val_view,
                teacher_target="hlt",
                teacher_block=make_teacher_block("hlt", val_view),
            )
            cfg = PD10StudentTrainConfig(
                student_init="scratch",
                teacher_target="hlt",
                output_dir=str(root / "student"),
                hlt_cache_dir=str(root / "hlt_cache"),
                teacher_logit_cache=str(root / "teacher_logits"),
                target_mode=PD10_TARGET_FULL_LOGITS,
                temperature=2.0,
                kd_alpha=0.5,
                kd_warmup_epochs=1,
                seed=42,
                batch_size=2,
                epochs=2,
                lr=1.0e-2,
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

            output = root / "student"
            self.assertTrue((output / "best_model_val.pt").exists())
            self.assertTrue((output / "last.pt").exists())
            self.assertTrue((output / "config.json").exists())
            self.assertTrue((output / "training_curves.json").exists())
            self.assertTrue((output / "run_report.json").exists())
            self.assertTrue((output / "model_val_report.json").exists())
            self.assertFalse((output / "final_test_report.json").exists())
            self.assertEqual(report["contract"], PD10_STUDENT_TRAINING_CONTRACT)
            self.assertEqual(report["experiment_step"], PD10_STEP7_EXPERIMENT_STEP)
            self.assertEqual(report["student_init"], "scratch")
            self.assertEqual(report["teacher_target"], "hlt")
            self.assertEqual(report["target_mode"], PD10_TARGET_FULL_LOGITS)
            self.assertEqual(report["kd_warmup_epochs"], 1)
            self.assertEqual(report["selection_metric"], "model_val_accuracy")
            self.assertIn("best_model_val_metrics", report)
            self.assertTrue(report["teacher_logits_train_time_only"])
            self.assertFalse(report["inference_requires_teacher_logits"])
            self.assertFalse(report["inference_requires_offline_inputs"])
            self.assertTrue(report["no_final_test_used_for_selection"])
            self.assertIn("final_test_evaluation_skipped", report)

            saved = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["variant_name"], cfg.variant_name)
            self.assertEqual(saved["model_val_dataset"]["teacher_target"], "hlt")
            self.assertIn("model_val_prediction_cache_skipped", report)

    def test_student_prediction_cache_writes_rich_hlt_only_metrics(self):
        torch = require_torch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = make_hlt_view(labels=(0, 1, 0, 1), split="model_val")
            dataset = PD10StudentDistillationDataset(view, teacher_target="none")
            cfg = PD10StudentTrainConfig(
                student_init="scratch",
                teacher_target="none",
                output_dir=str(root / "student"),
                hlt_cache_dir=str(root / "hlt_cache"),
                batch_size=2,
                device="cpu",
                amp=False,
                evaluate_final_test=False,
            )
            model = TinyStudent().to(torch.device("cpu"))

            metrics, metadata = collect_and_save_pd10_student_predictions(
                model,
                dataset,
                config=cfg,
                device=torch.device("cpu"),
                split="model_val",
            )

            prediction_dir = pd10_student_prediction_dir(cfg.output_dir)
            self.assertTrue((prediction_dir / cfg.variant_name / "model_val_predictions.npz").exists())
            self.assertEqual(metadata["contract"], PD10_STUDENT_PREDICTION_CACHE_CONTRACT)
            self.assertEqual(metadata["allowed_inputs"], "HLT_only")
            self.assertTrue(metadata["no_offline_inputs_loaded"])
            self.assertEqual(metrics["n_jets"], 4)
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("per_class_metrics", metrics)
            self.assertIn("binary_metrics", metrics)
            self.assertIn("QCD_vs_Hbb", metrics["binary_metrics"])
            self.assertIn("fpr_at_signal_eff_0p50_macro", metrics)
            self.assertIn("score_thresholds_by_class", metrics)

    def test_run_epoch_rejects_wrong_teacher_presence(self):
        torch = require_torch()
        view = make_hlt_view(labels=(0, 1), split="model_val")
        teacher_dataset = PD10StudentDistillationDataset(
            view,
            teacher_target="hlt",
            teacher_block=make_teacher_block("hlt", view),
        )
        loader = __import__(
            "teacher_logit_reco.privileged_distill_10class",
            fromlist=["make_pd10_student_data_loader"],
        ).make_pd10_student_data_loader(teacher_dataset, batch_size=2, shuffle=False, seed=1)
        ce_config = PD10StudentTrainConfig(
            student_init="scratch",
            teacher_target="none",
            output_dir="out",
            hlt_cache_dir="hlt_cache",
            evaluate_final_test=False,
        )

        with self.assertRaises(ValueError):
            run_pd10_student_epoch(
                TinyStudent(),
                loader,
                config=ce_config,
                device=torch.device("cpu"),
                epoch=1,
            )

    def test_cli_defaults_use_pd10_student_layout_and_canonical_sources(self):
        module = load_script_module()
        args = module.parse_args(
            [
                "--student-init",
                "scratch",
                "--teacher-target",
                "none",
                "--skip-final-test",
            ]
        )
        layout = default_pd10_experiment_layout(output_root="checkpoints")

        self.assertEqual(args.student_init, "scratch")
        self.assertEqual(args.teacher_target, "none")
        self.assertEqual(args.target_mode, PD10_TARGET_FULL_LOGITS)
        self.assertEqual(args.hlt_cache_dir, str(layout.hlt_cache_dir))
        self.assertEqual(args.teacher_logit_cache, str(layout.teacher_logits_dir))
        self.assertIsNone(args.baseline_checkpoint)
        self.assertIsNone(args.output_dir)
        self.assertFalse(args.confirm_final_test)
        self.assertTrue(args.skip_final_test)

        kd_args = module.parse_args(
            [
                "--student-init",
                "warm_start",
                "--teacher-target",
                "dual_view",
                "--target-mode",
                "top3",
                "--baseline-checkpoint",
                "baseline.pt",
                "--teacher-logit-cache",
                "teacher_logits",
                "--confirm-final-test",
            ]
        )
        self.assertEqual(kd_args.student_init, "warm_start")
        self.assertEqual(kd_args.teacher_target, "dual_view")
        self.assertEqual(kd_args.target_mode, PD10_TARGET_TOP3)
        self.assertEqual(kd_args.baseline_checkpoint, "baseline.pt")
        self.assertEqual(kd_args.teacher_logit_cache, "teacher_logits")
        self.assertTrue(kd_args.confirm_final_test)


if __name__ == "__main__":
    unittest.main()

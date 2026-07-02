import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.privileged_distill_10class import (
    PD10_DUAL_VIEW_DEFAULT_BATCH_SIZE,
    PD10_DUAL_VIEW_DEFAULT_EPOCHS,
    PD10_DUAL_VIEW_LOGIT_FEATURE_DIM,
    PD10_DUAL_VIEW_LOGIT_MODEL_NAME,
    PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT,
    PD10_NUM_CLASSES,
    PD10_STEP4_EXPERIMENT_STEP,
    PD10_STEP5_EXPERIMENT_STEP,
    PD10_TEACHER_LOGIT_CACHE_CONTRACT,
    PD10DualViewFeatureBlock,
    PD10DualViewLogitFusionTeacher,
    PD10DualViewLogitTeacherConfig,
    build_pd10_dual_view_feature_block,
    build_pd10_dual_view_prediction_block,
    default_pd10_experiment_layout,
    load_pd10_dual_view_logit_block,
    pd10_dual_view_teacher_checkpoint,
    pd10_dual_view_teacher_dir,
    pd10_dual_view_teacher_logit_cache_dir,
    pd10_part_teacher_model_name,
    train_pd10_dual_view_logit_model_from_features,
    validate_pd10_dual_view_input_blocks,
    validate_pd10_dual_view_logit_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_pd10_dual_view_logit_teacher.py"
SBATCH_DIR = REPO_ROOT / "sbatch"


def load_script_module():
    spec = importlib.util.spec_from_file_location("train_pd10_dual_view_logit_teacher", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_ids(labels, *, prefix="jet"):
    return [JetIdentity(file=f"{prefix}_{int(label)}.root", entry=index, label=int(label)) for index, label in enumerate(labels)]


def make_teacher_block(target: str, split: str, logits: np.ndarray, labels: np.ndarray) -> PredictionBlock:
    model_name = pd10_part_teacher_model_name(target)
    metadata = {
        "contract": PD10_TEACHER_LOGIT_CACHE_CONTRACT,
        "experiment_step": PD10_STEP4_EXPERIMENT_STEP,
        "teacher_target": target,
        "model_name": model_name,
        "split": split,
        "num_classes": PD10_NUM_CLASSES,
    }
    if target == "hlt":
        metadata.update(
            {
                "source_view": "fixed_hlt",
                "hlt_content_hash": f"{split}-hlt-hash",
                "no_offline_inputs_loaded": True,
            }
        )
    else:
        metadata.update(
            {
                "source_view": "offline",
                "no_hlt_inputs_loaded": True,
                "offline_privileged_inputs_loaded": True,
            }
        )
    return PredictionBlock(
        model_name=model_name,
        split=split,
        logits=logits.astype(np.float32),
        probs=softmax_np(logits),
        labels=labels.astype(np.int64),
        jet_ids=make_ids(labels),
        metadata=metadata,
    )


def make_feature_block(labels, *, split="model_val") -> PD10DualViewFeatureBlock:
    labels = np.asarray(labels, dtype=np.int64)
    features = np.zeros((len(labels), PD10_DUAL_VIEW_LOGIT_FEATURE_DIM), dtype=np.float32)
    base_logits = np.full((len(labels), PD10_NUM_CLASSES), -2.0, dtype=np.float32)
    base_logits[np.arange(len(labels)), labels] = 2.0
    return PD10DualViewFeatureBlock(
        split=split,
        features=features,
        base_logits=base_logits,
        labels=labels,
        jet_ids=make_ids(labels),
        metadata={
            "input_hlt_model_name": "hlt_part_teacher_10class",
            "input_offline_model_name": "offline_part_teacher_10class",
            "hlt_prediction_content_hash": "hlt-hash",
            "offline_prediction_content_hash": "offline-hash",
            "hlt_jet_identity_hash": "identity-hash",
            "offline_jet_identity_hash": "identity-hash",
        },
    )


class PD10Step5DualViewTeacherTests(unittest.TestCase):
    def test_feature_builder_uses_exact_58_column_contract(self):
        labels = np.asarray([0, 1, 2], dtype=np.int64)
        hlt_logits = np.zeros((3, PD10_NUM_CLASSES), dtype=np.float32)
        offline_logits = np.zeros((3, PD10_NUM_CLASSES), dtype=np.float32)
        hlt_logits[np.arange(3), labels] = 1.0
        offline_logits[np.arange(3), labels] = 3.0
        offline_logits[:, 9] = -1.0
        hlt_block = make_teacher_block("hlt", "model_val", hlt_logits, labels)
        offline_block = make_teacher_block("offline", "model_val", offline_logits, labels)

        feature_block = build_pd10_dual_view_feature_block(hlt_block, offline_block)

        self.assertEqual(feature_block.features.shape, (3, PD10_DUAL_VIEW_LOGIT_FEATURE_DIM))
        self.assertTrue(np.allclose(feature_block.features[:, 0:10], hlt_logits))
        self.assertTrue(np.allclose(feature_block.features[:, 10:20], offline_logits))
        self.assertTrue(np.allclose(feature_block.features[:, 20:30], offline_logits - hlt_logits))
        self.assertTrue(np.allclose(feature_block.features[:, 30:40], softmax_np(hlt_logits)))
        self.assertTrue(np.allclose(feature_block.features[:, 40:50], softmax_np(offline_logits)))
        self.assertTrue(np.allclose(feature_block.base_logits, 0.5 * (hlt_logits + offline_logits)))
        self.assertTrue(np.all(feature_block.features[:, 57] == 1.0))

    def test_alignment_mismatch_is_refused(self):
        labels = np.asarray([0, 1], dtype=np.int64)
        logits = np.zeros((2, PD10_NUM_CLASSES), dtype=np.float32)
        hlt_block = make_teacher_block("hlt", "model_val", logits, labels)
        offline_block = make_teacher_block("offline", "model_val", logits, labels)
        offline_block.jet_ids[1] = JetIdentity(file="different.root", entry=99, label=1)

        with self.assertRaises(ValueError):
            validate_pd10_dual_view_input_blocks(hlt_block, offline_block)

    def test_zero_initialized_delta_outputs_exact_base_logits(self):
        torch = require_torch()
        model = PD10DualViewLogitFusionTeacher(hidden_dim=16, dropout=0.0)
        model.eval()
        features = torch.randn(4, PD10_DUAL_VIEW_LOGIT_FEATURE_DIM)
        base_logits = torch.randn(4, PD10_NUM_CLASSES)

        with torch.no_grad():
            output = model(features, base_logits)

        final_linear = model.delta_net[-1]
        self.assertTrue(torch.allclose(final_linear.weight, torch.zeros_like(final_linear.weight)))
        self.assertTrue(torch.allclose(final_linear.bias, torch.zeros_like(final_linear.bias)))
        self.assertTrue(torch.allclose(output, base_logits, atol=1.0e-6))

    def test_training_keeps_initial_average_if_delta_does_not_help(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.asarray([index % PD10_NUM_CLASSES for index in range(50)], dtype=np.int64)
            train_block = make_feature_block(labels, split="model_train")
            val_block = make_feature_block(labels[:30], split="model_val")
            cfg = PD10DualViewLogitTeacherConfig(
                output_dir=str(root / "dual_teacher"),
                teacher_logit_dir=str(root / "teacher_logits"),
                prediction_splits=("model_val",),
                seed=123,
                batch_size=10,
                eval_batch_size=10,
                epochs=2,
                lr=1.0e-2,
                hidden_dim=16,
                dropout=0.0,
                device="cpu",
                max_train_jets=50,
                max_val_jets=30,
                overwrite=True,
            )

            report = train_pd10_dual_view_logit_model_from_features(cfg, train_block, val_block)

            self.assertTrue((root / "dual_teacher" / "best_model_val.pt").exists())
            self.assertLessEqual(
                report["best_model_val_cross_entropy"],
                report["initial_model_val"]["cross_entropy"] + 1.0e-8,
            )
            self.assertEqual(report["selection_metric"], "model_val_cross_entropy")
            self.assertTrue(report["no_final_test_used_for_selection"])

    def test_prediction_block_metadata_marks_train_time_privilege(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = PD10DualViewLogitTeacherConfig(
                output_dir=str(root / "dual_teacher"),
                teacher_logit_dir=str(root / "teacher_logits"),
                prediction_output_dir=str(root / "teacher_logits"),
                prediction_splits=("model_val",),
                overwrite=True,
            )
            cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            cfg.checkpoint_path.write_bytes(b"synthetic dual-view checkpoint")
            labels = np.asarray([0, 1, 2], dtype=np.int64)
            feature_block = make_feature_block(labels)
            logits = feature_block.base_logits.copy()

            block = build_pd10_dual_view_prediction_block(
                cfg,
                "model_val",
                logits=logits,
                feature_block=feature_block,
                checkpoint_payload={"epoch": 0, "experiment_step": PD10_STEP5_EXPERIMENT_STEP},
            )
            metadata = save_prediction_block(block, cfg.prediction_root)
            validate_pd10_dual_view_logit_metadata(metadata, split="model_val")

            loaded = load_pd10_dual_view_logit_block(cfg.prediction_root, "model_val")
            self.assertEqual(loaded.logits.shape, (3, PD10_NUM_CLASSES))
            self.assertEqual(loaded.metadata["contract"], PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT)
            self.assertEqual(loaded.metadata["model_name"], PD10_DUAL_VIEW_LOGIT_MODEL_NAME)
            self.assertEqual(loaded.metadata["student_deployment_inputs"], "HLT_only")
            self.assertTrue(loaded.metadata["teacher_logits_train_time_only"])
            self.assertFalse(loaded.metadata["uses_raw_offline_particles"])

    def test_config_and_cli_defaults_use_canonical_pd10_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = PD10DualViewLogitTeacherConfig(
                output_dir=str(root / "teacher"),
                teacher_logit_dir=str(root / "logits"),
                prediction_splits=("model_val",),
            )
            self.assertEqual(cfg.model_name, "dual_view_logit_teacher_10class")
            self.assertEqual(cfg.teacher_target, "dual_view")
            self.assertEqual(cfg.prediction_dir, root / "logits" / "dual_view_logit_teacher_10class")

        layout = default_pd10_experiment_layout(output_root="checkpoints")
        module = load_script_module()
        args = module.parse_args(["--confirm-final-test"])
        self.assertEqual(args.teacher_logit_dir, str(layout.teacher_logits_dir))
        self.assertEqual(args.output_dir, str(layout.teacher_dir("dual_view")))
        self.assertEqual(args.prediction_output_dir, str(layout.teacher_logits_dir))
        self.assertEqual(args.batch_size, PD10_DUAL_VIEW_DEFAULT_BATCH_SIZE)
        self.assertEqual(args.epochs, PD10_DUAL_VIEW_DEFAULT_EPOCHS)
        self.assertTrue(args.confirm_final_test)

        with self.assertRaises(ValueError):
            PD10DualViewLogitTeacherConfig(
                output_dir="out",
                teacher_logit_dir="logits",
                prediction_splits=("final_test",),
            )
        self.assertEqual(
            pd10_dual_view_teacher_dir(output_root=root),
            default_pd10_experiment_layout(output_root=root).teacher_dir("dual_view"),
        )
        self.assertEqual(
            pd10_dual_view_teacher_checkpoint(output_root=root),
            default_pd10_experiment_layout(output_root=root).teacher_checkpoint("dual_view"),
        )
        self.assertEqual(
            pd10_dual_view_teacher_logit_cache_dir(output_root=root),
            default_pd10_experiment_layout(output_root=root).teacher_logit_cache_dir("dual_view"),
        )

    def test_pd10_step5_sbatch_wiring(self):
        common = (SBATCH_DIR / "common.sh").read_text(encoding="utf-8")
        runner = (SBATCH_DIR / "run_pd10_train_dual_view_teacher.sh").read_text(encoding="utf-8")
        submitter = (SBATCH_DIR / "submit_pd10_step5_dual_view_teacher.sh").read_text(encoding="utf-8")

        self.assertIn("PD10_DUAL_VIEW_TEACHER_DIR:=${PD10_TEACHERS_DIR}/dual_view_logit_teacher_10class", common)
        self.assertIn("PD10_DUAL_VIEW_BATCH_SIZE:=8192", common)
        self.assertIn("PD10_DUAL_VIEW_EPOCHS:=20", common)
        self.assertIn("PD10_DUAL_VIEW_EARLY_STOP_PATIENCE:=4", common)
        self.assertIn('dual|dual_view) echo "dual_view_logit_teacher_10class"', common)

        self.assertIn("scripts/train_pd10_dual_view_logit_teacher.py", runner)
        self.assertIn("--teacher-logit-dir \"${PD10_TEACHER_LOGITS_DIR}\"", runner)
        self.assertIn("--output-dir \"${OUTPUT_DIR}\"", runner)
        self.assertIn("--prediction-output-dir \"${PD10_TEACHER_LOGITS_DIR}\"", runner)
        self.assertIn("--splits \"${split_args[@]}\"", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn("teacher_logit_manifest.json", runner)
        self.assertIn('${split}_predictions.npz', runner)

        self.assertIn("run_pd10_train_dual_view_teacher.sh", submitter)
        self.assertIn("pd10_step5_dual_view_teacher_submission", submitter)
        self.assertIn("UPSTREAM_DEPENDENCY", submitter)


if __name__ == "__main__":
    unittest.main()

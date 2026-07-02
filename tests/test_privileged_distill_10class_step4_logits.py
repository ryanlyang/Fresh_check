import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.fusion import save_prediction_block
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.privileged_distill_10class import (
    PD10_NUM_CLASSES,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_HLT,
    PD10_TEACHER_LOGIT_CACHE_CONTRACT,
    PD10_TEACHER_LOGIT_SPLITS,
    PD10_TEACHER_OFFLINE,
    PD10TeacherLogitCacheConfig,
    build_pd10_teacher_logit_block,
    default_pd10_experiment_layout,
    load_pd10_teacher_logit_block,
    pd10_teacher_logit_cache_dir,
    pd10_teacher_logit_prediction_paths,
    pd10_teacher_logit_selection_seed,
    pd10_part_teacher_model_name,
    validate_pd10_teacher_logit_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "cache_pd10_teacher_logits.py"
SBATCH_DIR = REPO_ROOT / "sbatch"


def load_script_module():
    spec = importlib.util.spec_from_file_location("cache_pd10_teacher_logits", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_test_view(*, labels=(0, 1, 2), split="model_val", view_name="fixed_hlt") -> JetView:
    labels_np = np.asarray(labels, dtype=np.int64)
    tokens = np.zeros((len(labels_np), 2, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.ones((len(labels_np), 2), dtype=bool)
    jet_ids = [
        JetIdentity(file=f"{view_name}_class{int(label)}.root", entry=index, label=int(label))
        for index, label in enumerate(labels_np)
    ]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels_np,
        jet_ids=jet_ids,
        split=split,
        metadata={"view": view_name},
    )


def make_config(tmp: Path, *, teacher="hlt", confirm_final_test=True) -> PD10TeacherLogitCacheConfig:
    checkpoint = tmp / f"{teacher}_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    return PD10TeacherLogitCacheConfig(
        teacher_target=teacher,
        checkpoint=str(checkpoint),
        output_dir=str(tmp / "teacher_logits"),
        manifest_path=str(tmp / "split_manifest.json.gz"),
        hlt_cache_dir=str(tmp / "hlt_cache"),
        data_dir=str(tmp / "data"),
        confirm_final_test=confirm_final_test,
        batch_size=2,
        num_workers=0,
    )


class PD10Step4TeacherLogitTests(unittest.TestCase):
    def test_cache_config_names_paths_splits_and_final_test_guardrails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_config(root, teacher="hlt")

            self.assertEqual(PD10_TEACHER_LOGIT_SPLITS, ("model_train", "model_val", "final_test"))
            self.assertEqual(cfg.teacher_target, PD10_TEACHER_HLT)
            self.assertEqual(cfg.model_name, "hlt_part_teacher_10class")
            self.assertEqual(cfg.source_view, "fixed_hlt")
            self.assertEqual(cfg.splits, PD10_TEACHER_LOGIT_SPLITS)
            self.assertEqual(cfg.teacher_output_dir, Path(cfg.output_dir) / cfg.model_name)
            self.assertEqual(
                pd10_teacher_logit_cache_dir("offline", output_root=root),
                default_pd10_experiment_layout(output_root=root).teacher_logit_cache_dir("offline"),
            )
            npz_path, metadata_path = pd10_teacher_logit_prediction_paths(cfg.output_dir, "hlt", "model_val")
            self.assertTrue(npz_path.as_posix().endswith("hlt_part_teacher_10class/model_val_predictions.npz"))
            self.assertTrue(metadata_path.as_posix().endswith("model_val_predictions_metadata.json"))
            self.assertNotEqual(
                pd10_teacher_logit_selection_seed(cfg, "model_train"),
                pd10_teacher_logit_selection_seed(cfg, "model_val"),
            )

            with self.assertRaises(ValueError):
                make_config(root, teacher="hlt", confirm_final_test=False)
            with self.assertRaises(ValueError):
                PD10TeacherLogitCacheConfig(
                    teacher_target="hlt",
                    checkpoint=str(root / "checkpoint.pt"),
                    output_dir="out",
                    manifest_path="manifest.json.gz",
                    hlt_cache_dir="hlt_cache",
                    splits=("stack_train",),
                )
            with self.assertRaises(ValueError):
                PD10TeacherLogitCacheConfig(
                    teacher_target="hlt",
                    checkpoint=str(root / "checkpoint.pt"),
                    output_dir="out",
                    manifest_path="manifest.json.gz",
                    hlt_cache_dir="hlt_cache",
                    splits=("model_train",),
                    max_model_train_jets=PD10_SPLIT_SIZES["model_train"] + 1,
                )

    def test_build_save_and_load_hlt_teacher_logit_block_preserves_alignment_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_config(root, teacher="hlt")
            view = make_test_view()
            logits = np.arange(len(view.labels) * PD10_NUM_CLASSES, dtype=np.float32).reshape(
                len(view.labels), PD10_NUM_CLASSES
            )
            block = build_pd10_teacher_logit_block(
                cfg,
                "model_val",
                logits=logits,
                labels=view.labels,
                view=view,
                source_metadata={
                    "source_view": "fixed_hlt",
                    "source_manifest_hash": "manifest-hash",
                    "hlt_content_hash": "hlt-hash",
                    "no_offline_inputs_loaded": True,
                    "offline_privileged_inputs_loaded": False,
                },
                checkpoint_payload={"epoch": 7, "experiment_step": "unit_test"},
            )
            metadata = save_prediction_block(block, cfg.output_dir)
            validate_pd10_teacher_logit_metadata(metadata, teacher_target="hlt", split="model_val")

            loaded = load_pd10_teacher_logit_block(cfg.output_dir, "hlt", "model_val")
            self.assertEqual(loaded.logits.shape, (len(view.labels), PD10_NUM_CLASSES))
            self.assertTrue(np.array_equal(loaded.labels, view.labels))
            self.assertEqual([jet.key() for jet in loaded.jet_ids], [jet.key() for jet in view.jet_ids])
            self.assertEqual(loaded.metadata["contract"], PD10_TEACHER_LOGIT_CACHE_CONTRACT)
            self.assertEqual(loaded.metadata["teacher_target"], PD10_TEACHER_HLT)
            self.assertEqual(loaded.metadata["source_view"], "fixed_hlt")
            self.assertEqual(loaded.metadata["allowed_inputs"], "HLT_only")
            self.assertEqual(loaded.metadata["hlt_content_hash"], "hlt-hash")
            self.assertTrue(loaded.metadata["no_offline_inputs_loaded"])
            self.assertFalse(loaded.metadata["offline_privileged_inputs_loaded"])

    def test_build_block_refuses_label_and_jet_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_config(root, teacher="hlt")
            view = make_test_view(labels=(0, 1, 2))
            bad_labels = np.asarray([0, 9, 2], dtype=np.int64)
            logits = np.zeros((3, PD10_NUM_CLASSES), dtype=np.float32)

            with self.assertRaises(ValueError):
                build_pd10_teacher_logit_block(
                    cfg,
                    "model_val",
                    logits=logits,
                    labels=bad_labels,
                    view=view,
                    source_metadata={
                        "source_view": "fixed_hlt",
                        "hlt_content_hash": "hlt-hash",
                        "no_offline_inputs_loaded": True,
                    },
                    checkpoint_payload={},
                )

    def test_metadata_validation_encodes_no_cross_view_input_contract(self):
        hlt_metadata = {
            "contract": PD10_TEACHER_LOGIT_CACHE_CONTRACT,
            "experiment_step": "pd10_step4_teacher_logit_cache",
            "teacher_target": PD10_TEACHER_HLT,
            "model_name": pd10_part_teacher_model_name("hlt"),
            "split": "model_val",
            "num_classes": PD10_NUM_CLASSES,
            "n_jets": 3,
            "source_view": "fixed_hlt",
            "allowed_inputs": "HLT_only",
            "hlt_content_hash": "hlt-hash",
            "no_offline_inputs_loaded": True,
        }
        validate_pd10_teacher_logit_metadata(hlt_metadata, teacher_target="hlt", split="model_val")

        bad_allowed_inputs = dict(hlt_metadata)
        bad_allowed_inputs["allowed_inputs"] = "offline_only_train_time_privileged"
        with self.assertRaises(ValueError):
            validate_pd10_teacher_logit_metadata(bad_allowed_inputs, teacher_target="hlt", split="model_val")

        missing_hlt_hash = dict(hlt_metadata)
        missing_hlt_hash["hlt_content_hash"] = None
        with self.assertRaises(ValueError):
            validate_pd10_teacher_logit_metadata(missing_hlt_hash, teacher_target="hlt", split="model_val")

        missing_no_offline = dict(hlt_metadata)
        missing_no_offline["no_offline_inputs_loaded"] = False
        with self.assertRaises(ValueError):
            validate_pd10_teacher_logit_metadata(missing_no_offline, teacher_target="hlt", split="model_val")

        offline_metadata = {
            "contract": PD10_TEACHER_LOGIT_CACHE_CONTRACT,
            "experiment_step": "pd10_step4_teacher_logit_cache",
            "teacher_target": PD10_TEACHER_OFFLINE,
            "model_name": pd10_part_teacher_model_name("offline"),
            "split": "model_val",
            "num_classes": PD10_NUM_CLASSES,
            "n_jets": 3,
            "source_view": "offline",
            "allowed_inputs": "offline_only_train_time_privileged",
            "no_hlt_inputs_loaded": True,
            "offline_privileged_inputs_loaded": True,
        }
        validate_pd10_teacher_logit_metadata(offline_metadata, teacher_target="offline", split="model_val")

        bad_offline = dict(offline_metadata)
        bad_offline["no_hlt_inputs_loaded"] = False
        with self.assertRaises(ValueError):
            validate_pd10_teacher_logit_metadata(bad_offline, teacher_target="offline", split="model_val")

    def test_cli_defaults_use_pd10_teacher_logit_layout(self):
        module = load_script_module()
        args = module.parse_args(["--teacher", "hlt", "--confirm-final-test"])
        layout = default_pd10_experiment_layout(output_root="checkpoints")

        self.assertEqual(args.teacher, "hlt")
        self.assertEqual(args.output_dir, str(layout.teacher_logits_dir))
        self.assertEqual(args.manifest, str(layout.split_manifest_path))
        self.assertEqual(args.hlt_cache_dir, str(layout.hlt_cache_dir))
        self.assertEqual(args.splits, list(PD10_TEACHER_LOGIT_SPLITS))
        self.assertTrue(args.confirm_final_test)
        self.assertFalse(args.no_skip_existing)

    def test_pd10_step4_sbatch_wiring(self):
        common = (SBATCH_DIR / "common.sh").read_text(encoding="utf-8")
        runner = (SBATCH_DIR / "run_pd10_cache_teacher_logits.sh").read_text(encoding="utf-8")
        submitter = (SBATCH_DIR / "submit_pd10_step4_teacher_logits.sh").read_text(encoding="utf-8")

        self.assertIn("PD10_TEACHER_LOGITS_DIR:=${PD10_ROOT}/teacher_logits", common)
        self.assertIn("PD10_TEACHER_LOGIT_TARGETS:=${PD10_TEACHER_TARGETS}", common)
        self.assertIn("PD10_TEACHER_LOGIT_SPLITS:=model_train model_val final_test", common)
        self.assertIn("PD10_TEACHER_LOGIT_NO_SKIP_EXISTING:=0", common)

        self.assertIn("scripts/cache_pd10_teacher_logits.py", runner)
        self.assertIn("--teacher \"${TEACHER}\"", runner)
        self.assertIn("--checkpoint \"${CHECKPOINT}\"", runner)
        self.assertIn("--output-dir \"${PD10_TEACHER_LOGITS_DIR}\"", runner)
        self.assertIn("--splits \"${split_args[@]}\"", runner)
        self.assertIn("--max-model-train-jets \"${PD10_MODEL_TRAIN_SIZE}\"", runner)
        self.assertIn("--max-model-val-jets \"${PD10_MODEL_VAL_SIZE}\"", runner)
        self.assertIn("--max-final-test-jets \"${PD10_FINAL_TEST_SIZE}\"", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn("--no-skip-existing", runner)
        self.assertIn("teacher_logit_manifest.json", runner)
        self.assertIn('${split}_predictions.npz', runner)

        self.assertIn("run_pd10_cache_teacher_logits.sh", submitter)
        self.assertIn('fresh_split_words teacher_args "${PD10_TEACHER_LOGIT_TARGETS}"', submitter)
        self.assertIn("pd10_step4_teacher_logits_submission", submitter)
        self.assertIn("hlt_part_teacher_10class", submitter)
        self.assertIn("offline_part_teacher_10class", submitter)


if __name__ == "__main__":
    unittest.main()

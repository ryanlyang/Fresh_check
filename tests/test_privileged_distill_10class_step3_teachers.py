import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from teacher_logit_reco.privileged_distill_10class import (
    PD10_HLT_TEACHER_SEED,
    PD10_OFFLINE_TEACHER_SEED,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_HLT,
    PD10_TEACHER_OFFLINE,
    PD10PartTeacherTrainConfig,
    default_pd10_experiment_layout,
    default_pd10_teacher_seed,
    normalize_pd10_part_teacher_target,
    pd10_hlt_params_dict,
    pd10_part_teacher_checkpoint,
    pd10_part_teacher_dir,
    pd10_part_teacher_model_name,
    register_pd10_part_teacher_checkpoint,
    sha256_file,
)
from teacher_logit_reco.privileged_distill_10class.teachers import _require_hlt_cache_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_or_register_pd10_teacher.py"
SBATCH_DIR = REPO_ROOT / "sbatch"


def load_script_module():
    spec = importlib.util.spec_from_file_location("train_or_register_pd10_teacher", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PD10Step3TeacherTests(unittest.TestCase):
    def test_teacher_names_paths_and_seeds_are_fixed(self):
        self.assertEqual(normalize_pd10_part_teacher_target("hlt_teacher"), PD10_TEACHER_HLT)
        self.assertEqual(normalize_pd10_part_teacher_target("offline_part"), PD10_TEACHER_OFFLINE)
        with self.assertRaises(ValueError):
            normalize_pd10_part_teacher_target(PD10_TEACHER_DUAL_VIEW)

        layout = default_pd10_experiment_layout(output_root="/tmp/checkpoints")
        self.assertEqual(pd10_part_teacher_model_name("hlt"), "hlt_part_teacher_10class")
        self.assertEqual(pd10_part_teacher_model_name("offline"), "offline_part_teacher_10class")
        self.assertEqual(pd10_part_teacher_dir("hlt", output_root="/tmp/checkpoints"), layout.teacher_dir("hlt"))
        self.assertEqual(
            pd10_part_teacher_checkpoint("offline", output_root="/tmp/checkpoints"),
            layout.teacher_checkpoint("offline"),
        )
        self.assertEqual(default_pd10_teacher_seed("hlt"), PD10_HLT_TEACHER_SEED)
        self.assertEqual(default_pd10_teacher_seed("offline"), PD10_OFFLINE_TEACHER_SEED)

    def test_teacher_train_config_requires_model_splits_and_explicit_final_test(self):
        cfg = PD10PartTeacherTrainConfig(
            teacher_target="hlt",
            output_dir="out",
            manifest_path="manifest.json.gz",
            cache_dir="hlt_cache",
            confirm_final_test=True,
        )
        self.assertEqual(cfg.teacher_target, "hlt")
        self.assertEqual(cfg.model_name, "hlt_part_teacher_10class")
        self.assertEqual(cfg.source_view, "fixed_hlt")
        self.assertEqual(cfg.max_train_jets, PD10_SPLIT_SIZES["model_train"])
        self.assertEqual(cfg.max_val_jets, PD10_SPLIT_SIZES["model_val"])
        self.assertEqual(cfg.max_final_test_jets, PD10_SPLIT_SIZES["final_test"])

        offline = PD10PartTeacherTrainConfig(
            teacher_target="offline",
            output_dir="out",
            manifest_path="manifest.json.gz",
            cache_dir="hlt_cache",
            confirm_final_test=True,
            max_train_jets=1000,
            max_val_jets=100,
            max_final_test_jets=100,
        )
        self.assertEqual(offline.source_view, "offline")
        self.assertEqual(offline.seed, PD10_OFFLINE_TEACHER_SEED)

        with self.assertRaises(ValueError):
            PD10PartTeacherTrainConfig(
                teacher_target="hlt",
                output_dir="out",
                manifest_path="manifest.json.gz",
                cache_dir="hlt_cache",
            )
        with self.assertRaises(ValueError):
            PD10PartTeacherTrainConfig(
                teacher_target="hlt",
                output_dir="out",
                manifest_path="manifest.json.gz",
                cache_dir="hlt_cache",
                confirm_final_test=True,
                train_split="stack_train",
            )
        with self.assertRaises(ValueError):
            PD10PartTeacherTrainConfig(
                teacher_target="offline",
                output_dir="out",
                manifest_path="manifest.json.gz",
                cache_dir="hlt_cache",
                confirm_final_test=True,
                max_final_test_jets=PD10_SPLIT_SIZES["final_test"] + 1,
            )

    def test_register_checkpoint_writes_step3_artifacts_without_final_eval_for_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pt"
            source.write_bytes(b"pretend torch checkpoint")
            source_report = root / "source_model_val.json"
            source_report.write_text('{"best_model_val_accuracy": 0.75}\n', encoding="utf-8")
            source_final = root / "source_final_test.json"
            source_final.write_text('{"accuracy": 0.70}\n', encoding="utf-8")
            output = root / "teacher"
            cfg = PD10PartTeacherTrainConfig(
                teacher_target="offline",
                output_dir=str(output),
                manifest_path=str(root / "manifest.json.gz"),
                cache_dir=str(root / "hlt_cache"),
                evaluate_final_test=False,
                confirm_final_test=True,
            )

            result = register_pd10_part_teacher_checkpoint(
                cfg,
                source_checkpoint=source,
                source_model_val_report=source_report,
                source_final_test_report=source_final,
            )

            self.assertTrue((output / "best_model_val.pt").exists())
            self.assertTrue((output / "run_report.json").exists())
            self.assertTrue((output / "model_val_report.json").exists())
            self.assertTrue((output / "final_test_report.json").exists())
            self.assertTrue((output / "source_metadata.json").exists())
            self.assertTrue((output / "config.json").exists())
            self.assertTrue((output / "registration_report.json").exists())
            self.assertEqual(result["teacher_target"], "offline")
            self.assertEqual(result["model_name"], "offline_part_teacher_10class")
            self.assertEqual(sha256_file(output / "best_model_val.pt"), sha256_file(source))

            with self.assertRaises(FileExistsError):
                register_pd10_part_teacher_checkpoint(
                    cfg,
                    source_checkpoint=source,
                    source_model_val_report=source_report,
                )

            unconfirmed = PD10PartTeacherTrainConfig(
                teacher_target="offline",
                output_dir=str(root / "unconfirmed"),
                manifest_path=str(root / "manifest.json.gz"),
                cache_dir=str(root / "hlt_cache"),
                evaluate_final_test=False,
            )
            with self.assertRaises(ValueError):
                register_pd10_part_teacher_checkpoint(
                    unconfirmed,
                    source_checkpoint=source,
                    source_model_val_report=source_report,
                    source_final_test_report=source_final,
                )

    def test_hlt_teacher_cache_contract_rejects_wrong_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "hlt_cache"
            cache.mkdir()
            for split in ("model_train", "model_val"):
                payload = {
                    "hlt_profile": "fixed_hlt_v2_realistic",
                    "hlt_degradation_strength": 0.6,
                    "hlt_params": pd10_hlt_params_dict(),
                }
                (cache / f"{split}_fixed_hlt_metadata.json").write_text(
                    json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            cfg = PD10PartTeacherTrainConfig(
                teacher_target="hlt",
                output_dir=str(root / "teacher"),
                manifest_path=str(root / "missing_manifest.json.gz"),
                cache_dir=str(cache),
                evaluate_final_test=False,
            )

            with self.assertRaisesRegex(ValueError, "hlt_profile"):
                _require_hlt_cache_contract(cfg)

    def test_hlt_cache_contract_can_force_final_test_split_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "hlt_cache"
            cache.mkdir()
            for split in ("model_train", "model_val"):
                payload = {
                    "hlt_profile": "fixed_hlt_v1",
                    "hlt_degradation_strength": 0.6,
                    "hlt_params": pd10_hlt_params_dict(),
                }
                (cache / f"{split}_fixed_hlt_metadata.json").write_text(
                    json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            cfg = PD10PartTeacherTrainConfig(
                teacher_target="hlt",
                output_dir=str(root / "teacher"),
                manifest_path=str(root / "missing_manifest.json.gz"),
                cache_dir=str(cache),
                evaluate_final_test=False,
            )

            self.assertTrue(_require_hlt_cache_contract(cfg)["ok"])
            with self.assertRaisesRegex(ValueError, "final_test"):
                _require_hlt_cache_contract(cfg, include_final_test=True)

    def test_cli_defaults_use_pd10_layout_and_require_teacher(self):
        module = load_script_module()
        args = module.parse_args(["--teacher", "hlt", "--confirm-final-test"])
        layout = default_pd10_experiment_layout(output_root="checkpoints")

        self.assertEqual(args.teacher, "hlt")
        self.assertEqual(args.manifest, str(layout.split_manifest_path))
        self.assertEqual(args.hlt_cache_dir, str(layout.hlt_cache_dir))
        self.assertIsNone(args.output_dir)
        self.assertTrue(args.confirm_final_test)
        self.assertFalse(args.skip_final_test)

    def test_pd10_step3_sbatch_wiring(self):
        common = (SBATCH_DIR / "common.sh").read_text(encoding="utf-8")
        runner = (SBATCH_DIR / "run_pd10_train_teacher.sh").read_text(encoding="utf-8")
        submitter = (SBATCH_DIR / "submit_pd10_step3_teachers.sh").read_text(encoding="utf-8")

        self.assertIn("PD10_TEACHERS_DIR:=${PD10_ROOT}/teachers", common)
        self.assertIn("PD10_TEACHER_TARGETS:=hlt offline", common)
        self.assertIn("PD10_HLT_TEACHER_SEED:=101", common)
        self.assertIn("PD10_OFFLINE_TEACHER_SEED:=707", common)
        self.assertIn("PD10_TEACHER_MODEL_SIZE:=base", common)
        self.assertIn("fresh_pd10_teacher_model_name", common)
        self.assertIn("fresh_pd10_teacher_source_checkpoint", common)

        self.assertIn("scripts/train_or_register_pd10_teacher.py", runner)
        self.assertIn("--teacher \"${TEACHER}\"", runner)
        self.assertIn("--manifest \"${PD10_MANIFEST_PATH}\"", runner)
        self.assertIn("--hlt-cache-dir \"${PD10_HLT_CACHE_DIR}\"", runner)
        self.assertIn("--max-train-jets \"${PD10_MODEL_TRAIN_SIZE}\"", runner)
        self.assertIn("--max-val-jets \"${PD10_MODEL_VAL_SIZE}\"", runner)
        self.assertIn("--max-final-test-jets \"${PD10_FINAL_TEST_SIZE}\"", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn("final_test_report.json", runner)

        self.assertIn("run_pd10_train_teacher.sh", submitter)
        self.assertIn('fresh_split_words teacher_args "${PD10_TEACHER_TARGETS}"', submitter)
        self.assertIn("pd10_step3_teachers_submission", submitter)
        self.assertIn("hlt_part_teacher_10class", submitter)
        self.assertIn("offline_part_teacher_10class", submitter)


if __name__ == "__main__":
    unittest.main()

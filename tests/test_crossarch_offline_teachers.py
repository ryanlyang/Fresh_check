from pathlib import Path
import tempfile
import unittest

from teacher_logit_reco.crossarch_experiment import CrossArchExperimentLayout
from teacher_logit_reco.crossarch_offline_teachers import (
    CrossArchOfflineTeacherTrainConfig,
    crossarch_offline_teacher_dir,
    normalize_crossarch_teacher_architecture,
    register_crossarch_offline_teacher_checkpoint,
    sha256_file,
)


class CrossArchOfflineTeacherConfigTests(unittest.TestCase):
    def test_normalizes_teacher_architecture_and_output_paths(self):
        self.assertEqual(normalize_crossarch_teacher_architecture("ParticleTransformer"), "part")
        self.assertEqual(normalize_crossarch_teacher_architecture("ParticleNet"), "pn")
        self.assertEqual(normalize_crossarch_teacher_architecture("P-CNN"), "pcnn")
        self.assertEqual(
            crossarch_offline_teacher_dir("PFN", output_root="/tmp/checkpoints"),
            CrossArchExperimentLayout(output_root="/tmp/checkpoints").offline_teacher_dir("pfn"),
        )
        with self.assertRaises(ValueError):
            normalize_crossarch_teacher_architecture("bad")

    def test_train_config_rejects_non_model_splits_and_bad_sizes(self):
        cfg = CrossArchOfflineTeacherTrainConfig(
            architecture="pfn",
            output_dir="out",
            manifest_path="manifest.json.gz",
        )
        self.assertEqual(cfg.architecture, "pfn")
        with self.assertRaises(ValueError):
            CrossArchOfflineTeacherTrainConfig(
                architecture="part",
                output_dir="out",
                manifest_path="manifest.json.gz",
                train_split="stack_train",
            )
        with self.assertRaises(ValueError):
            CrossArchOfflineTeacherTrainConfig(
                architecture="part",
                output_dir="out",
                manifest_path="manifest.json.gz",
                batch_size=0,
            )


class CrossArchOfflineTeacherRegistrationTests(unittest.TestCase):
    def test_register_copies_checkpoint_and_writes_required_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pt"
            source.write_bytes(b"pretend torch checkpoint")
            report = root / "source_report.json"
            report.write_text('{"ok": true}\n', encoding="utf-8")
            out = root / "registered"

            result = register_crossarch_offline_teacher_checkpoint(
                architecture="pcnn",
                source_checkpoint=source,
                output_dir=out,
                source_report=report,
            )

            self.assertEqual(result["architecture"], "pcnn")
            self.assertTrue((out / "best_model_val.pt").exists())
            self.assertTrue((out / "run_report.json").exists())
            self.assertTrue((out / "model_val_report.json").exists())
            self.assertTrue((out / "source_metadata.json").exists())
            self.assertTrue((out / "config.json").exists())
            self.assertTrue((out / "registration_report.json").exists())
            self.assertEqual(result["checkpoint_sha256"], sha256_file(out / "best_model_val.pt"))
            self.assertEqual(result["source_checkpoint_sha256"], sha256_file(source))

            with self.assertRaises(FileExistsError):
                register_crossarch_offline_teacher_checkpoint(
                    architecture="pcnn",
                    source_checkpoint=source,
                    output_dir=out,
                    source_report=report,
                )


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest

from teacher_logit_reco.crossarch_experiment import CrossArchExperimentLayout
from teacher_logit_reco.crossarch_reconstructors import (
    RECONSTRUCTOR_TRAIN_SCRIPTS,
    build_crossarch_reconstructor_spec,
    build_crossarch_reconstructor_specs,
    crossarch_reconstructor_checkpoint,
    crossarch_reconstructor_dir,
    crossarch_reconstructor_model_name,
    crossarch_teacher_checkpoint_path,
    train_script_for_reconstructor,
)


class CrossArchReconstructorHelperTests(unittest.TestCase):
    def test_train_script_mapping_and_model_names(self):
        self.assertEqual(
            train_script_for_reconstructor("gt"),
            "scripts/train_teacher_logit_global_transformer_reco.py",
        )
        self.assertEqual(
            train_script_for_reconstructor("ParticleNet"),
            "scripts/train_teacher_logit_particle_net_reco.py",
        )
        self.assertEqual(
            train_script_for_reconstructor("PFC"),
            "scripts/train_teacher_logit_particle_flow_reco.py",
        )
        self.assertEqual(
            train_script_for_reconstructor("particle_cnn"),
            "scripts/train_teacher_logit_particle_cnn_reco.py",
        )
        self.assertEqual(set(RECONSTRUCTOR_TRAIN_SCRIPTS), {"gt", "pn", "pfn", "pcnn"})
        self.assertEqual(crossarch_reconstructor_model_name("PFC", "PCNN"), "pfn_reco_to_pcnn_teacher")
        with self.assertRaises(ValueError):
            train_script_for_reconstructor("bad")

    def test_builds_exactly_sixteen_step5_specs(self):
        specs = build_crossarch_reconstructor_specs(output_root="/tmp/checkpoints")
        self.assertEqual(len(specs), 16)
        self.assertEqual(len({spec.model_name for spec in specs}), 16)
        self.assertEqual(specs[0].model_name, "gt_reco_to_part_teacher")
        self.assertEqual(specs[-1].model_name, "pcnn_reco_to_pcnn_teacher")
        self.assertTrue(all(spec.train_script.startswith("scripts/train_teacher_logit_") for spec in specs))
        self.assertTrue(all(spec.output_dir.parts[-3] == "reco_models" for spec in specs))
        self.assertTrue(all(spec.teacher_checkpoint.name == "best_model_val.pt" for spec in specs))

    def test_paths_match_fresh_crossarch_layout(self):
        layout = CrossArchExperimentLayout(output_root="/tmp/checkpoints")
        spec = build_crossarch_reconstructor_spec("gt", "part", output_root="/tmp/checkpoints")
        self.assertEqual(spec.output_dir, layout.reco_model_dir("gt", "part"))
        self.assertEqual(spec.teacher_checkpoint, layout.offline_teacher_checkpoint("part"))
        self.assertEqual(spec.reconstructor_implementation, "global_transformer")
        self.assertEqual(
            crossarch_reconstructor_dir("PN", "PFN", output_root="/tmp/checkpoints"),
            Path("/tmp/checkpoints") / "teacher_logit_reco_crossarch_500k" / "reco_models" / "pn" / "pfn",
        )
        self.assertEqual(
            crossarch_reconstructor_checkpoint("PN", "PFN", output_root="/tmp/checkpoints"),
            Path("/tmp/checkpoints")
            / "teacher_logit_reco_crossarch_500k"
            / "reco_models"
            / "pn"
            / "pfn"
            / "best_model_val.pt",
        )
        self.assertEqual(
            crossarch_teacher_checkpoint_path("ParticleTransformer", output_root="/tmp/checkpoints"),
            Path("/tmp/checkpoints")
            / "teacher_logit_reco_crossarch_500k"
            / "offline_teachers"
            / "part"
            / "best_model_val.pt",
        )
        self.assertNotIn("teacher_logit_reco_pn", str(spec.output_dir))
        self.assertNotIn("jetclass_fresh_reco7", str(spec.output_dir))


if __name__ == "__main__":
    unittest.main()

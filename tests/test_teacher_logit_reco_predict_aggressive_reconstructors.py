import unittest
from pathlib import Path

from teacher_logit_reco.predict_aggressive_reconstructors import (
    AGGRESSIVE_MODEL_NAME_PREFIXES,
    PREDICT_EXPERIMENT_STEP,
    TeacherLogitAggressivePredictionConfig,
    default_model_name_for_reco_teacher_architecture,
    validate_aggressive_reconstructor_architecture,
)


def make_config(**overrides):
    payload = {
        "output_dir": "tmp/aggressive_predictions",
        "hlt_cache_dir": "checkpoints/jetclass_fresh_hlt_cache",
        "reconstructor_checkpoint": "checkpoints/aggressive/best_model_val.pt",
        "splits": ["stack_train", "stack_val"],
    }
    payload.update(overrides)
    return TeacherLogitAggressivePredictionConfig(**payload)


class TeacherLogitAggressivePredictionConfigTests(unittest.TestCase):
    def test_step_metadata_and_prefixes_are_distinct(self):
        self.assertEqual(PREDICT_EXPERIMENT_STEP, "teacher_logit_reco_step9_aggressive_predictions")
        self.assertEqual(
            AGGRESSIVE_MODEL_NAME_PREFIXES,
            {
                "aggressive_global_transformer": "aggt",
                "aggressive_particle_net": "agpn",
                "aggressive_particle_flow": "agpfn",
                "aggressive_particle_cnn": "agpcnn",
            },
        )

    def test_default_model_names_do_not_collide_with_conservative_names(self):
        self.assertEqual(
            default_model_name_for_reco_teacher_architecture("aggressive_gt", "part"),
            "aggt_reco_to_part_teacher",
        )
        self.assertEqual(
            default_model_name_for_reco_teacher_architecture("agpn", "pn"),
            "agpn_reco_to_pn_teacher",
        )
        self.assertEqual(
            default_model_name_for_reco_teacher_architecture("aggressive-pfn", "pcnn"),
            "agpfn_reco_to_pcnn_teacher",
        )
        self.assertEqual(
            default_model_name_for_reco_teacher_architecture("agpcnn", "pfn"),
            "agpcnn_reco_to_pfn_teacher",
        )

    def test_validates_aggressive_architecture_aliases(self):
        self.assertEqual(validate_aggressive_reconstructor_architecture("aggressive_gt"), "aggressive_global_transformer")
        self.assertEqual(validate_aggressive_reconstructor_architecture("agpn"), "aggressive_particle_net")
        self.assertEqual(validate_aggressive_reconstructor_architecture("aggressive-pfn"), "aggressive_particle_flow")
        self.assertEqual(validate_aggressive_reconstructor_architecture("agpcnn"), "aggressive_particle_cnn")
        with self.assertRaises(ValueError):
            validate_aggressive_reconstructor_architecture("pn")
        with self.assertRaises(ValueError):
            validate_aggressive_reconstructor_architecture(None)

    def test_config_guards_final_test_and_bad_architectures(self):
        with self.assertRaises(ValueError):
            make_config(splits=["final_test"])
        with self.assertRaises(ValueError):
            make_config(reco_architecture="particle_net")
        cfg = make_config(splits=["final_test"], confirm_final_test=True, reco_architecture="agpcnn")
        self.assertEqual(cfg.expected_reco_architecture, "aggressive_particle_cnn")

    def test_prediction_dir_default_and_override(self):
        cfg = make_config()
        self.assertEqual(cfg.resolved_prediction_dir, Path("tmp/aggressive_predictions/predictions"))
        cfg = make_config(prediction_dir="tmp/custom_predictions")
        self.assertEqual(cfg.resolved_prediction_dir, Path("tmp/custom_predictions"))


if __name__ == "__main__":
    unittest.main()

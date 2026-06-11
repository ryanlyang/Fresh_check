from pathlib import Path
import unittest

from teacher_logit_reco.crossarch_predictions import (
    HLT_PREDICT_SCRIPT,
    RECONSTRUCTOR_PREDICT_SCRIPTS,
    build_crossarch_hlt_prediction_specs,
    build_crossarch_prediction_specs,
    build_crossarch_reco_prediction_specs,
    crossarch_hlt_prediction_spec,
    crossarch_reco_prediction_spec,
    predict_script_for_reconstructor,
)


class CrossArchPredictionHelperTests(unittest.TestCase):
    def test_predict_script_mapping(self):
        self.assertEqual(
            predict_script_for_reconstructor("gt"),
            "scripts/predict_teacher_logit_global_transformer_reco.py",
        )
        self.assertEqual(
            predict_script_for_reconstructor("ParticleNet"),
            "scripts/predict_teacher_logit_particle_net_reco.py",
        )
        self.assertEqual(
            predict_script_for_reconstructor("PFC"),
            "scripts/predict_teacher_logit_particle_flow_reco.py",
        )
        self.assertEqual(
            predict_script_for_reconstructor("particle_cnn"),
            "scripts/predict_teacher_logit_particle_cnn_reco.py",
        )
        self.assertEqual(set(RECONSTRUCTOR_PREDICT_SCRIPTS), {"gt", "pn", "pfn", "pcnn"})
        self.assertEqual(HLT_PREDICT_SCRIPT, "scripts/predict_crossarch_hlt_baseline.py")
        with self.assertRaises(ValueError):
            predict_script_for_reconstructor("bad")

    def test_builds_sixteen_reco_specs_four_hlt_specs_and_twenty_total(self):
        reco_specs = build_crossarch_reco_prediction_specs(output_root="/tmp/checkpoints")
        hlt_specs = build_crossarch_hlt_prediction_specs(output_root="/tmp/checkpoints")
        all_specs = build_crossarch_prediction_specs(output_root="/tmp/checkpoints")
        self.assertEqual(len(reco_specs), 16)
        self.assertEqual(len(hlt_specs), 4)
        self.assertEqual(len(all_specs), 20)
        self.assertEqual(len({spec.model_name for spec in all_specs}), 20)
        self.assertEqual(reco_specs[0].model_name, "gt_reco_to_part_teacher")
        self.assertEqual(reco_specs[-1].model_name, "pcnn_reco_to_pcnn_teacher")
        self.assertEqual([spec.model_name for spec in hlt_specs], ["hlt_part", "hlt_pn", "hlt_pfn", "hlt_pcnn"])
        self.assertTrue(all(spec.source_kind == "teacher_logit_reco" for spec in reco_specs))
        self.assertTrue(all(spec.source_kind == "direct_hlt" for spec in hlt_specs))

    def test_prediction_paths_use_fresh_crossarch_namespace(self):
        reco = crossarch_reco_prediction_spec("PN", "PFN", output_root="/tmp/checkpoints")
        self.assertEqual(reco.model_name, "pn_reco_to_pfn_teacher")
        self.assertEqual(
            reco.checkpoint,
            Path("/tmp/checkpoints")
            / "teacher_logit_reco_crossarch_500k"
            / "reco_models"
            / "pn"
            / "pfn"
            / "best_model_val.pt",
        )
        self.assertEqual(
            reco.prediction_source_dir,
            Path("/tmp/checkpoints")
            / "teacher_logit_reco_crossarch_500k"
            / "predictions"
            / "pn_reco_to_pfn_teacher",
        )
        self.assertEqual(
            reco.run_output_dir,
            Path("/tmp/checkpoints")
            / "teacher_logit_reco_crossarch_500k"
            / "prediction_runs"
            / "reco"
            / "pn_reco_to_pfn_teacher",
        )

        hlt = crossarch_hlt_prediction_spec("ParticleTransformer", output_root="/tmp/checkpoints")
        self.assertEqual(hlt.model_name, "hlt_part")
        self.assertEqual(
            hlt.checkpoint,
            Path("/tmp/checkpoints")
            / "teacher_logit_reco_crossarch_500k"
            / "hlt_baselines"
            / "part"
            / "best_model_val.pt",
        )
        self.assertEqual(
            hlt.run_output_dir,
            Path("/tmp/checkpoints") / "teacher_logit_reco_crossarch_500k" / "prediction_runs" / "hlt" / "hlt_part",
        )
        self.assertNotIn("teacher_logit_reco_gt", str(reco.prediction_source_dir))
        self.assertNotIn("jetclass_fresh_fusion", str(hlt.prediction_source_dir))


if __name__ == "__main__":
    unittest.main()

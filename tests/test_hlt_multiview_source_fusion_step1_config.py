import unittest
from pathlib import Path

from teacher_logit_reco.hlt_multiview_source_fusion import (
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_FUSION_HLT_RANDOM_4SEED,
    HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SOURCE_5VIEW,
    HLT_MV_TRIVIEW_MODEL_NAME,
    default_hlt_mv_experiment_config,
    default_hlt_mv_experiment_layout,
    hlt_mv_config_manifest,
)


class HLTMultiviewSourceFusionStep1ConfigTest(unittest.TestCase):
    def test_default_config_freezes_expected_run_grid(self):
        config = default_hlt_mv_experiment_config(
            pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME
        )

        self.assertEqual(config.pdv3_experiment_name, HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME)
        self.assertEqual(config.pdv3_hlt_profile, "fixed_hlt_v2_realistic")
        self.assertEqual(config.pdv3_hlt_degradation_strength, 1.0)
        self.assertEqual(config.strengths, (0.10, 0.20, 0.35, 1.00))
        self.assertEqual(
            config.source_model_names,
            (
                "hlt_part_seed8801",
                "hlt2_part_s0p10_seed8811",
                "hlt2_part_s0p20_seed8821",
                "hlt2_part_s0p35_seed8831",
                "hlt2_part_s1p00_seed8841",
            ),
        )
        self.assertEqual(
            config.random_hlt_source_names,
            (
                "hlt_part_seed9101",
                "hlt_part_seed9102",
                "hlt_part_seed9103",
                "hlt_part_seed9104",
            ),
        )
        self.assertEqual(
            config.pretrained_dualview_names,
            (
                "sdv_hlt_hlt2_s0p10",
                "sdv_hlt_hlt2_s0p20",
                "sdv_hlt_hlt2_s0p35",
                "sdv_hlt_hlt2_s1p00",
            ),
        )
        self.assertEqual(
            config.scratch_dualview_names,
            (
                "sdv_hlt_hlt2_s0p10_scratch",
                "sdv_hlt_hlt2_s0p20_scratch",
                "sdv_hlt_hlt2_s0p35_scratch",
                "sdv_hlt_hlt2_s1p00_scratch",
            ),
        )
        self.assertEqual(
            config.logit_fusion_names,
            (
                HLT_MV_FUSION_SOURCE_5VIEW,
                HLT_MV_FUSION_HLT_RANDOM_4SEED,
                HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL,
                HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL,
            ),
        )
        self.assertIn("sdv_hlt_hlt_same_view", config.control_names)
        self.assertIn("tta_hlt_part_hlt_plus_hlt2_s1p00", config.control_names)

    def test_layout_uses_pdv3_root_and_existing_input_caches(self):
        layout = default_hlt_mv_experiment_layout(
            output_root="/home/ryreu/atlas/Fresh_check/checkpoints",
            pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
        )
        expected_root = Path("/home/ryreu/atlas/Fresh_check/checkpoints") / HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME

        self.assertEqual(layout.pdv3_root, expected_root)
        self.assertEqual(layout.root, expected_root / "hlt_multiview_source_fusion")
        self.assertEqual(layout.split_manifest_path, expected_root / "inputs" / "split_manifest" / "split_manifest.json.gz")
        self.assertEqual(layout.hlt_cache_dir, expected_root / "inputs" / "hlt_cache")
        self.assertEqual(
            layout.hlt2_cache_dir(0.35),
            expected_root / "hlt_self_dualview" / "hlt2_cache" / "hlt_second_degrade_mild_v1_s0p35",
        )
        self.assertEqual(
            layout.source_model_dir("hlt2_part_s0p35_seed8831"),
            layout.root / "source_models" / "hlt2_part_s0p35_seed8831",
        )
        self.assertEqual(layout.triview_model_dir(), layout.root / "triview" / HLT_MV_TRIVIEW_MODEL_NAME)

    def test_manifest_contains_config_and_layout_contracts(self):
        config = default_hlt_mv_experiment_config()
        layout = default_hlt_mv_experiment_layout(output_root="checkpoints")
        manifest = hlt_mv_config_manifest(config=config, layout=layout)

        self.assertEqual(manifest["contract"], "deployable_hlt_multiview_source_fusion_layout_v1")
        self.assertEqual(manifest["config"]["pdv3_experiment_name"], HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME)
        self.assertEqual(manifest["config"]["allowed_inputs"], "HLT_only")
        self.assertEqual(manifest["config"]["deployment_inputs"], "HLT_plus_deterministic_HLT2_multiview")
        self.assertEqual(
            manifest["layout"]["root"],
            f"checkpoints/{HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME}/hlt_multiview_source_fusion",
        )
        self.assertIn("source_5view", manifest["layout"]["logit_fusion_dirs"])
        self.assertIn("sdv_hlt_hlt2_s0p20", manifest["layout"]["pretrained_dualview_model_dirs"])

    def test_identity_strength_is_rejected_for_model_grid(self):
        with self.assertRaises(ValueError):
            default_hlt_mv_experiment_config(strengths=(0.00, 0.20))


if __name__ == "__main__":
    unittest.main()

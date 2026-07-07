import unittest
from pathlib import Path

from teacher_logit_reco.hlt_self_dualview import (
    HLTSDVExperimentConfig,
    HLT_SDV_ALLOWED_INPUTS,
    HLT_SDV_CONTRACT,
    HLT_SDV_DEFAULT_STRENGTHS,
    HLT_SDV_DEPLOYMENT_INPUTS,
    HLT_SDV_HLT2_PROFILE_NAME,
    HLT_SDV_VARIANT_HLT2_ONLY,
    HLT_SDV_VARIANT_SAME_VIEW,
    HLT_SDV_VARIANT_TTA,
    build_hlt_sdv_required_variants,
    default_hlt_sdv_experiment_config,
    default_hlt_sdv_experiment_layout,
    hlt_sdv_config_manifest,
    hlt_sdv_dual_hlt2_variant_name,
    hlt_sdv_hlt2_cache_name,
    hlt_sdv_strength_tag,
)
from teacher_logit_reco.privileged_distill_10class.config import (
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
)


class HLTSDVStep1ConfigTest(unittest.TestCase):
    def test_default_config_reuses_pd10_split_contract(self):
        config = default_hlt_sdv_experiment_config()

        self.assertEqual(config.hlt2_strengths, HLT_SDV_DEFAULT_STRENGTHS)
        self.assertEqual(config.split_order, PD10_SPLIT_ORDER)
        self.assertEqual(config.split_sizes, PD10_SPLIT_SIZES)
        self.assertTrue(config.reuse_pd10_split_manifest)
        self.assertTrue(config.reuse_pd10_hlt_cache)
        self.assertEqual(config.allowed_inputs, HLT_SDV_ALLOWED_INPUTS)
        self.assertEqual(config.deployment_inputs, HLT_SDV_DEPLOYMENT_INPUTS)
        self.assertEqual(config.primary_variant, "sdv_hlt_hlt2_s0p20")
        self.assertEqual(
            config.hlt2_cache_names,
            (
                "hlt_second_degrade_mild_v1_s0p00",
                "hlt_second_degrade_mild_v1_s0p10",
                "hlt_second_degrade_mild_v1_s0p20",
                "hlt_second_degrade_mild_v1_s0p35",
                "hlt_second_degrade_mild_v1_s1p00",
            ),
        )

    def test_strength_and_variant_names_are_stable(self):
        self.assertEqual(hlt_sdv_strength_tag(0), "s0p00")
        self.assertEqual(hlt_sdv_strength_tag(0.2), "s0p20")
        self.assertEqual(hlt_sdv_strength_tag("s0p35"), "s0p35")
        self.assertEqual(hlt_sdv_strength_tag(1.0), "s1p00")
        self.assertEqual(hlt_sdv_hlt2_cache_name(0.2), "hlt_second_degrade_mild_v1_s0p20")
        self.assertEqual(hlt_sdv_dual_hlt2_variant_name(0.35), "sdv_hlt_hlt2_s0p35")
        self.assertEqual(hlt_sdv_dual_hlt2_variant_name(1.0), "sdv_hlt_hlt2_s1p00")

        variants = build_hlt_sdv_required_variants()
        self.assertEqual(
            variants,
            (
                HLT_SDV_VARIANT_SAME_VIEW,
                "sdv_hlt_hlt2_s0p10",
                "sdv_hlt_hlt2_s0p20",
                "sdv_hlt_hlt2_s0p35",
                "sdv_hlt_hlt2_s1p00",
                HLT_SDV_VARIANT_HLT2_ONLY,
                HLT_SDV_VARIANT_TTA,
            ),
        )

    def test_layout_points_under_existing_pd10_root(self):
        layout = default_hlt_sdv_experiment_layout(
            output_root=Path("checkpoints"),
            pd10_experiment_name="privileged_distill_10class_5m_hlt0p4_run1",
        )
        pd10_root = Path("checkpoints") / "privileged_distill_10class_5m_hlt0p4_run1"

        self.assertEqual(layout.pd10_root, pd10_root)
        self.assertEqual(layout.root, pd10_root / "hlt_self_dualview")
        self.assertEqual(layout.split_manifest_path, pd10_root / "split_manifest" / "split_manifest.json.gz")
        self.assertEqual(layout.parent_hlt_cache_dir, pd10_root / "hlt_cache")
        self.assertEqual(
            layout.hlt2_cache_dir(0.2),
            pd10_root / "hlt_self_dualview" / "hlt2_cache" / "hlt_second_degrade_mild_v1_s0p20",
        )
        self.assertEqual(
            layout.variant_dir(HLT_SDV_VARIANT_SAME_VIEW),
            pd10_root / "hlt_self_dualview" / "models" / HLT_SDV_VARIANT_SAME_VIEW,
        )
        self.assertEqual(
            layout.final_report_path,
            pd10_root / "hlt_self_dualview" / "final_report" / "summary.json",
        )

    def test_config_rejects_drift_from_step1_contract(self):
        with self.assertRaisesRegex(ValueError, "identity 0.00"):
            HLTSDVExperimentConfig(hlt2_strengths=(0.10, 0.20))
        with self.assertRaisesRegex(ValueError, "Primary"):
            HLTSDVExperimentConfig(hlt2_strengths=(0.0, 0.10), primary_strength=0.20)
        with self.assertRaisesRegex(ValueError, "unique"):
            HLTSDVExperimentConfig(hlt2_strengths=(0.0, 0.10, 0.10, 0.20))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            HLTSDVExperimentConfig(hlt2_strengths=(0.0, -0.10, 0.20))
        with self.assertRaisesRegex(ValueError, "reuse the existing PD10 split manifest"):
            HLTSDVExperimentConfig(reuse_pd10_split_manifest=False)
        with self.assertRaisesRegex(ValueError, "reuse the existing PD10 HLT cache"):
            HLTSDVExperimentConfig(reuse_pd10_hlt_cache=False)
        with self.assertRaisesRegex(ValueError, "allowed_inputs"):
            HLTSDVExperimentConfig(allowed_inputs="offline")

    def test_manifest_combines_config_and_layout(self):
        config = default_hlt_sdv_experiment_config(
            pd10_experiment_name="privileged_distill_10class_5m_hlt0p4_run1"
        )
        layout = default_hlt_sdv_experiment_layout(
            output_root="checkpoints",
            pd10_experiment_name=config.pd10_experiment_name,
        )
        manifest = hlt_sdv_config_manifest(config=config, layout=layout)

        self.assertEqual(manifest["contract"], HLT_SDV_CONTRACT)
        self.assertEqual(manifest["config"]["hlt2_profile_name"], HLT_SDV_HLT2_PROFILE_NAME)
        self.assertEqual(manifest["layout"]["split_manifest_path"], "checkpoints/privileged_distill_10class_5m_hlt0p4_run1/split_manifest/split_manifest.json.gz")
        self.assertEqual(manifest["layout"]["parent_hlt_cache_dir"], "checkpoints/privileged_distill_10class_5m_hlt0p4_run1/hlt_cache")
        self.assertIn("s0p20", manifest["layout"]["hlt2_cache_dirs"])
        self.assertIn("sdv_hlt_hlt2_s0p20", manifest["layout"]["variant_dirs"])


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from teacher_logit_reco.hlt_multiview_source_fusion import (
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_DEPLOYMENT_INPUTS,
    HLT_MV_EXPERIMENT_NAME,
    HLT_MV_SOURCE_CONTRACT,
    HLT_MV_SOURCE_PREDICTION_CONTRACT,
    HLT_MV_SOURCE_VIEW_HLT,
    HLT_MV_SOURCE_VIEW_HLT2,
    build_hlt_mv_source_config,
    default_hlt_mv_experiment_layout,
    hlt_mv_source_cache_dir,
    hlt_mv_source_output_dir,
    hlt_mv_source_seed_from_name,
    hlt_mv_source_view_from_name,
    hlt_mv_strength_from_source_name,
)


class HLTMultiviewSourceFusionStep2SourceTest(unittest.TestCase):
    def setUp(self):
        self.output_root = Path("/home/ryreu/atlas/Fresh_check/checkpoints")
        self.layout = default_hlt_mv_experiment_layout(
            output_root=self.output_root,
            pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
        )

    def test_hlt2_source_config_uses_existing_hlt2_cache_and_scratch_defaults(self):
        config = build_hlt_mv_source_config(
            source_name="hlt2_part_s0p35_seed8831",
            layout=self.layout,
            confirm_final_test=True,
        )

        self.assertEqual(config.source_view, HLT_MV_SOURCE_VIEW_HLT2)
        self.assertEqual(config.seed, 8831)
        self.assertEqual(config.warm_start_checkpoint, None)
        self.assertFalse(config.amp)
        self.assertEqual(config.lr, 3.0e-4)
        self.assertEqual(config.max_train_jets, 5_000_000)
        self.assertEqual(config.max_val_jets, 1_000_000)
        self.assertEqual(config.max_final_test_jets, 1_000_000)
        self.assertEqual(Path(config.cache_dir), self.layout.hlt2_cache_dir(0.35))
        self.assertEqual(Path(config.output_dir), self.layout.source_model_dir("hlt2_part_s0p35_seed8831"))
        self.assertEqual(config.source_contract, HLT_MV_SOURCE_CONTRACT)
        self.assertEqual(config.prediction_contract, HLT_MV_SOURCE_PREDICTION_CONTRACT)
        self.assertEqual(config.experiment_name, HLT_MV_EXPERIMENT_NAME)
        self.assertEqual(config.deployment_inputs, HLT_MV_DEPLOYMENT_INPUTS)

    def test_hlt_and_random_seed_source_paths_are_distinct(self):
        hlt_name = "hlt_part_seed8801"
        random_name = "hlt_part_seed9103"

        self.assertEqual(hlt_mv_source_view_from_name(hlt_name), HLT_MV_SOURCE_VIEW_HLT)
        self.assertEqual(hlt_mv_source_cache_dir(self.layout, hlt_name), self.layout.hlt_cache_dir)
        self.assertEqual(hlt_mv_source_output_dir(self.layout, hlt_name), self.layout.source_model_dir(hlt_name))
        self.assertEqual(hlt_mv_source_seed_from_name(hlt_name), 8801)

        self.assertEqual(hlt_mv_source_cache_dir(self.layout, random_name), self.layout.hlt_cache_dir)
        self.assertEqual(
            hlt_mv_source_output_dir(self.layout, random_name),
            self.layout.random_hlt_source_dir(random_name),
        )
        self.assertEqual(hlt_mv_source_seed_from_name(random_name), 9103)

    def test_strength_parsing_is_restricted_to_hlt2_sources(self):
        self.assertAlmostEqual(hlt_mv_strength_from_source_name("hlt2_part_s1p00_seed8841"), 1.0)
        self.assertAlmostEqual(hlt_mv_strength_from_source_name("hlt2_part_s0p10_seed8811"), 0.10)
        with self.assertRaises(ValueError):
            hlt_mv_strength_from_source_name("hlt_part_seed8801")
        with self.assertRaises(ValueError):
            build_hlt_mv_source_config(source_name="hlt2_part_s0p00_seed8800", confirm_final_test=True)

    def test_slurm_wrapper_reuses_existing_caches_and_defaults_no_amp(self):
        text = Path("sbatch/run_hlt_mv_train_source_model.sh").read_text()

        self.assertIn("#SBATCH --job-name=hlt_mv_src", text)
        self.assertIn("scripts/train_hlt_mv_source_model.py", text)
        self.assertIn("HLT_MV_HLT_CACHE_DIR:=${HLT_MV_PDV3_ROOT}/inputs/hlt_cache", text)
        self.assertIn("HLT_MV_HLT2_CACHE_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_self_dualview/hlt2_cache", text)
        self.assertIn("HLT_MV_SOURCE_LR:=0.0003", text)
        self.assertIn("HLT_MV_SOURCE_AMP:=0", text)
        self.assertIn("fresh_append_flag_if_enabled cmd --amp", text)
        self.assertNotIn("build_pd10_hlt2_cache.py", text)
        self.assertNotIn("build_fixed_hlt_cache.py", text)
        self.assertNotIn("warm-start", text.lower())


if __name__ == "__main__":
    unittest.main()

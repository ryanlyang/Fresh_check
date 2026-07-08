import unittest
from dataclasses import replace
from pathlib import Path

from teacher_logit_reco.hlt_multiview_source_fusion import (
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_PRETRAINED_DUALVIEW_CONTRACT,
    HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_AMP,
    HLT_MV_PRETRAINED_DUALVIEW_EXPERIMENT_STEP,
    build_hlt_mv_pretrained_dualview_config,
    default_hlt_mv_experiment_layout,
    hlt_mv_pretrained_dualview_checkpoint_paths,
    hlt_mv_pretrained_dualview_output_dir,
    hlt_mv_pretrained_dualview_prediction_paths,
    hlt_mv_pretrained_dualview_source_names,
    hlt_mv_pretrained_dualview_strength_from_name,
    train_hlt_mv_pretrained_dualview,
)


class HLTMultiviewSourceFusionStep5PretrainedDualviewTest(unittest.TestCase):
    def setUp(self):
        self.layout = default_hlt_mv_experiment_layout(
            output_root="/home/ryreu/atlas/Fresh_check/checkpoints",
            pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
        )

    def test_strength_source_and_checkpoint_mapping_is_explicit(self):
        model_name = "sdv_hlt_hlt2_s0p35"

        self.assertAlmostEqual(hlt_mv_pretrained_dualview_strength_from_name(model_name), 0.35)
        self.assertEqual(
            hlt_mv_pretrained_dualview_source_names(model_name),
            ("hlt_part_seed8801", "hlt2_part_s0p35_seed8831"),
        )
        hlt_checkpoint, hlt2_checkpoint = hlt_mv_pretrained_dualview_checkpoint_paths(self.layout, model_name)
        self.assertEqual(
            hlt_checkpoint,
            self.layout.source_model_dir("hlt_part_seed8801") / "best_model_val.pt",
        )
        self.assertEqual(
            hlt2_checkpoint,
            self.layout.source_model_dir("hlt2_part_s0p35_seed8831") / "best_model_val.pt",
        )
        self.assertEqual(
            hlt_mv_pretrained_dualview_output_dir(self.layout, model_name),
            self.layout.pretrained_dualview_model_dir(model_name),
        )

    def test_pretrained_dualview_config_uses_sdv_trainer_with_branch_initializers(self):
        config = build_hlt_mv_pretrained_dualview_config(
            model_name="sdv_hlt_hlt2_s1p00",
            layout=self.layout,
            confirm_final_test=True,
        )

        self.assertEqual(config.variant_name, "sdv_hlt_hlt2_s1p00")
        self.assertEqual(Path(config.output_dir), self.layout.pretrained_dualview_model_dir("sdv_hlt_hlt2_s1p00"))
        self.assertEqual(Path(config.hlt_cache_dir), self.layout.hlt_cache_dir)
        self.assertEqual(Path(config.hlt2_cache_dir), self.layout.hlt2_cache_dir(1.0))
        self.assertEqual(
            Path(config.hlt_teacher_checkpoint),
            self.layout.source_model_dir("hlt_part_seed8801") / "best_model_val.pt",
        )
        self.assertEqual(
            Path(config.hlt2_branch_checkpoint),
            self.layout.source_model_dir("hlt2_part_s1p00_seed8841") / "best_model_val.pt",
        )
        self.assertTrue(config.initialize_branches)
        self.assertEqual(config.head_warmup_epochs, 1)
        self.assertEqual(config.head_warmup_lr, 3.0e-4)
        self.assertFalse(config.amp)
        self.assertEqual(config.max_train_jets, 5_000_000)
        self.assertEqual(config.max_val_jets, 1_000_000)
        self.assertEqual(config.max_final_test_jets, 1_000_000)
        self.assertEqual(HLT_MV_PRETRAINED_DUALVIEW_CONTRACT, "hlt_multiview_pretrained_particle_dualview_v1")
        self.assertEqual(HLT_MV_PRETRAINED_DUALVIEW_EXPERIMENT_STEP, "hlt_mv_step5_pretrained_particle_dualview")
        self.assertFalse(HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_AMP)

    def test_scratch_and_control_names_are_rejected(self):
        with self.assertRaises(ValueError):
            hlt_mv_pretrained_dualview_strength_from_name("sdv_hlt_hlt_same_view")
        with self.assertRaises(ValueError):
            hlt_mv_pretrained_dualview_strength_from_name("sdv_hlt_hlt2_s0p20_scratch")
        with self.assertRaises(ValueError):
            build_hlt_mv_pretrained_dualview_config(
                model_name="sdv_hlt_hlt2_s0p00",
                layout=self.layout,
                confirm_final_test=True,
            )

    def test_prediction_paths_match_logit_fusion_layout(self):
        npz_path, meta_path = hlt_mv_pretrained_dualview_prediction_paths(
            self.layout,
            "sdv_hlt_hlt2_s0p20",
            "final_test",
        )
        expected_root = (
            self.layout.pretrained_dualview_model_dir("sdv_hlt_hlt2_s0p20")
            / "predictions"
            / "sdv_hlt_hlt2_s0p20"
        )

        self.assertEqual(npz_path, expected_root / "final_test_predictions.npz")
        self.assertEqual(meta_path, expected_root / "final_test_predictions_metadata.json")
        with self.assertRaises(ValueError):
            hlt_mv_pretrained_dualview_prediction_paths(self.layout, "sdv_hlt_hlt2_s0p20", "model_train")

    def test_training_wrapper_refuses_non_pretrained_settings_before_running(self):
        config = build_hlt_mv_pretrained_dualview_config(
            model_name="sdv_hlt_hlt2_s0p10",
            layout=self.layout,
            confirm_final_test=True,
        )

        with self.assertRaises(ValueError):
            train_hlt_mv_pretrained_dualview(replace(config, hlt2_branch_checkpoint=None))
        with self.assertRaises(ValueError):
            train_hlt_mv_pretrained_dualview(replace(config, head_warmup_epochs=0))

    def test_slurm_wrapper_reuses_sources_and_existing_caches(self):
        text = Path("sbatch/run_hlt_mv_train_pretrained_dualview.sh").read_text()

        self.assertIn("#SBATCH --job-name=hlt_mv_pdv", text)
        self.assertIn("scripts/train_hlt_mv_pretrained_dualview.py", text)
        self.assertIn("HLT_MV_CANONICAL_HLT_SOURCE_NAME:=hlt_part_seed8801", text)
        self.assertIn("HLT_MV_SOURCE_NAMES:=hlt_part_seed8801 hlt2_part_s0p10_seed8811", text)
        self.assertIn('^sdv_hlt_hlt2_(s[0-9]+p[0-9]+)$', text)
        self.assertIn('hlt_mv_source_name_for_tag()', text)
        self.assertIn("HLT_MV_SOURCE_MODELS_DIR}/${HLT_MV_CANONICAL_HLT_SOURCE_NAME}/best_model_val.pt", text)
        self.assertIn("HLT_MV_SOURCE_MODELS_DIR}/${hlt2_source_name}/best_model_val.pt", text)
        self.assertIn("hlt_second_degrade_mild_v1_${hlt2_tag}", text)
        self.assertIn("--hlt-checkpoint", text)
        self.assertIn("--hlt2-checkpoint", text)
        self.assertIn("HLT_MV_PRETRAINED_DUALVIEW_HEAD_WARMUP_LR:=0.0003", text)
        self.assertIn("fresh_require_file \"${HLT2_CHECKPOINT}\"", text)
        self.assertIn("hlt_mv_pretrained_dualview_report.json", text)
        self.assertNotIn("--no-branch-init", text)
        self.assertNotIn("build_pd10_hlt2_cache.py", text)
        self.assertNotIn("build_fixed_hlt_cache.py", text)


if __name__ == "__main__":
    unittest.main()

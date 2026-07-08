import unittest
from dataclasses import replace
from pathlib import Path

from teacher_logit_reco.hlt_multiview_source_fusion import (
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_SCRATCH_DUALVIEW_CONTRACT,
    HLT_MV_SCRATCH_DUALVIEW_DEFAULT_AMP,
    HLT_MV_SCRATCH_DUALVIEW_EXPERIMENT_STEP,
    HLT_MV_SCRATCH_DUALVIEW_UNUSED_CHECKPOINT,
    build_hlt_mv_scratch_dualview_config,
    default_hlt_mv_experiment_layout,
    hlt_mv_scratch_dualview_output_dir,
    hlt_mv_scratch_dualview_prediction_paths,
    hlt_mv_scratch_dualview_strength_from_name,
    train_hlt_mv_scratch_dualview,
)
from teacher_logit_reco.hlt_self_dualview import hlt_sdv_strength_from_variant


class HLTMultiviewSourceFusionStep6ScratchDualviewTest(unittest.TestCase):
    def setUp(self):
        self.layout = default_hlt_mv_experiment_layout(
            output_root="/home/ryreu/atlas/Fresh_check/checkpoints",
            pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
        )

    def test_scratch_strength_and_output_mapping(self):
        model_name = "sdv_hlt_hlt2_s0p35_scratch"

        self.assertAlmostEqual(hlt_mv_scratch_dualview_strength_from_name(model_name), 0.35)
        self.assertAlmostEqual(hlt_sdv_strength_from_variant(model_name), 0.35)
        self.assertEqual(
            hlt_mv_scratch_dualview_output_dir(self.layout, model_name),
            self.layout.scratch_dualview_model_dir(model_name),
        )

    def test_scratch_dualview_config_disables_branch_initializers_and_warmup(self):
        config = build_hlt_mv_scratch_dualview_config(
            model_name="sdv_hlt_hlt2_s1p00_scratch",
            layout=self.layout,
            confirm_final_test=True,
        )

        self.assertEqual(config.variant_name, "sdv_hlt_hlt2_s1p00_scratch")
        self.assertEqual(Path(config.output_dir), self.layout.scratch_dualview_model_dir("sdv_hlt_hlt2_s1p00_scratch"))
        self.assertEqual(Path(config.hlt_cache_dir), self.layout.hlt_cache_dir)
        self.assertEqual(Path(config.hlt2_cache_dir), self.layout.hlt2_cache_dir(1.0))
        self.assertEqual(config.hlt_teacher_checkpoint, HLT_MV_SCRATCH_DUALVIEW_UNUSED_CHECKPOINT)
        self.assertIsNone(config.hlt2_branch_checkpoint)
        self.assertFalse(config.initialize_branches)
        self.assertEqual(config.head_warmup_epochs, 0)
        self.assertEqual(config.head_warmup_lr, 3.0e-4)
        self.assertFalse(config.amp)
        self.assertEqual(config.max_train_jets, 5_000_000)
        self.assertEqual(config.max_val_jets, 1_000_000)
        self.assertEqual(config.max_final_test_jets, 1_000_000)
        self.assertEqual(HLT_MV_SCRATCH_DUALVIEW_CONTRACT, "hlt_multiview_scratch_particle_dualview_v1")
        self.assertEqual(HLT_MV_SCRATCH_DUALVIEW_EXPERIMENT_STEP, "hlt_mv_step6_scratch_particle_dualview")
        self.assertFalse(HLT_MV_SCRATCH_DUALVIEW_DEFAULT_AMP)

    def test_pretrained_and_control_names_are_rejected(self):
        with self.assertRaises(ValueError):
            hlt_mv_scratch_dualview_strength_from_name("sdv_hlt_hlt2_s0p20")
        with self.assertRaises(ValueError):
            hlt_mv_scratch_dualview_strength_from_name("sdv_hlt_hlt_same_view")
        with self.assertRaises(ValueError):
            build_hlt_mv_scratch_dualview_config(
                model_name="sdv_hlt_hlt2_s0p00_scratch",
                layout=self.layout,
                confirm_final_test=True,
            )
        with self.assertRaises(ValueError):
            build_hlt_mv_scratch_dualview_config(
                model_name="sdv_hlt_hlt2_s0p20_scratch",
                layout=self.layout,
                confirm_final_test=True,
                head_warmup_epochs=1,
            )

    def test_prediction_paths_match_logit_fusion_layout(self):
        npz_path, meta_path = hlt_mv_scratch_dualview_prediction_paths(
            self.layout,
            "sdv_hlt_hlt2_s0p20_scratch",
            "model_val",
        )
        expected_root = (
            self.layout.scratch_dualview_model_dir("sdv_hlt_hlt2_s0p20_scratch")
            / "predictions"
            / "sdv_hlt_hlt2_s0p20_scratch"
        )

        self.assertEqual(npz_path, expected_root / "model_val_predictions.npz")
        self.assertEqual(meta_path, expected_root / "model_val_predictions_metadata.json")
        with self.assertRaises(ValueError):
            hlt_mv_scratch_dualview_prediction_paths(self.layout, "sdv_hlt_hlt2_s0p20_scratch", "model_train")

    def test_training_wrapper_refuses_pretrained_settings_before_running(self):
        config = build_hlt_mv_scratch_dualview_config(
            model_name="sdv_hlt_hlt2_s0p10_scratch",
            layout=self.layout,
            confirm_final_test=True,
        )

        with self.assertRaises(ValueError):
            train_hlt_mv_scratch_dualview(replace(config, initialize_branches=True))
        with self.assertRaises(ValueError):
            train_hlt_mv_scratch_dualview(replace(config, head_warmup_epochs=1))

    def test_slurm_wrapper_reuses_caches_and_has_no_checkpoint_inputs(self):
        text = Path("sbatch/run_hlt_mv_train_scratch_dualview.sh").read_text()

        self.assertIn("#SBATCH --job-name=hlt_mv_sdv", text)
        self.assertIn("scripts/train_hlt_mv_scratch_dualview.py", text)
        self.assertIn("sdv_hlt_hlt2_s0p20_scratch", text)
        self.assertIn('^sdv_hlt_hlt2_(s[0-9]+p[0-9]+)_scratch$', text)
        self.assertIn("hlt_second_degrade_mild_v1_${BASH_REMATCH[1]}", text)
        self.assertIn("HLT_MV_SCRATCH_DUALVIEW_HEAD_WARMUP_EPOCHS:=0", text)
        self.assertIn("HLT_MV_SCRATCH_DUALVIEW_HEAD_WARMUP_LR:=0.0003", text)
        self.assertIn("--head-warmup-epochs", text)
        self.assertIn("hlt_mv_scratch_dualview_report.json", text)
        self.assertNotIn("--hlt-checkpoint", text)
        self.assertNotIn("--hlt2-checkpoint", text)
        self.assertNotIn("source_models", text)
        self.assertNotIn("--no-branch-init", text)
        self.assertNotIn("build_pd10_hlt2_cache.py", text)
        self.assertNotIn("build_fixed_hlt_cache.py", text)


if __name__ == "__main__":
    unittest.main()

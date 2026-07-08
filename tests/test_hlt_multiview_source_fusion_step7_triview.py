import unittest
from dataclasses import replace
from pathlib import Path

from teacher_logit_reco.hlt_multiview_source_fusion import (
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_TRIVIEW_CONTRACT,
    HLT_MV_TRIVIEW_DEFAULT_AMP,
    HLT_MV_TRIVIEW_EXPERIMENT_STEP,
    HLT_MV_TRIVIEW_MODEL_NAME,
    build_hlt_mv_triview_config,
    default_hlt_mv_experiment_layout,
    hlt_mv_triview_checkpoint_paths,
    hlt_mv_triview_output_dir,
    hlt_mv_triview_prediction_paths,
    hlt_mv_triview_source_names,
    normalize_hlt_mv_triview_name,
    train_hlt_mv_triview,
)


class HLTMultiviewSourceFusionStep7TriViewTest(unittest.TestCase):
    def setUp(self):
        self.layout = default_hlt_mv_experiment_layout(
            output_root="/home/ryreu/atlas/Fresh_check/checkpoints",
            pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
        )

    def test_triview_source_and_checkpoint_mapping_is_canonical(self):
        self.assertEqual(normalize_hlt_mv_triview_name(), HLT_MV_TRIVIEW_MODEL_NAME)
        self.assertEqual(
            hlt_mv_triview_source_names(),
            ("hlt_part_seed8801", "hlt2_part_s0p35_seed8831", "hlt2_part_s1p00_seed8841"),
        )

        hlt_checkpoint, s0p35_checkpoint, s1p00_checkpoint = hlt_mv_triview_checkpoint_paths(self.layout)
        self.assertEqual(
            hlt_checkpoint,
            self.layout.source_model_dir("hlt_part_seed8801") / "best_model_val.pt",
        )
        self.assertEqual(
            s0p35_checkpoint,
            self.layout.source_model_dir("hlt2_part_s0p35_seed8831") / "best_model_val.pt",
        )
        self.assertEqual(
            s1p00_checkpoint,
            self.layout.source_model_dir("hlt2_part_s1p00_seed8841") / "best_model_val.pt",
        )
        self.assertEqual(
            hlt_mv_triview_output_dir(self.layout),
            self.layout.triview_model_dir(HLT_MV_TRIVIEW_MODEL_NAME),
        )

    def test_triview_config_uses_existing_caches_sources_and_high_data_sizes(self):
        config = build_hlt_mv_triview_config(
            layout=self.layout,
            confirm_final_test=True,
        )

        self.assertEqual(config.model_name, HLT_MV_TRIVIEW_MODEL_NAME)
        self.assertEqual(Path(config.output_dir), self.layout.triview_model_dir(HLT_MV_TRIVIEW_MODEL_NAME))
        self.assertEqual(Path(config.hlt_cache_dir), self.layout.hlt_cache_dir)
        self.assertEqual(Path(config.hlt2_s0p35_cache_dir), self.layout.hlt2_cache_dir(0.35))
        self.assertEqual(Path(config.hlt2_s1p00_cache_dir), self.layout.hlt2_cache_dir(1.0))
        self.assertEqual(
            Path(config.hlt_source_checkpoint),
            self.layout.source_model_dir("hlt_part_seed8801") / "best_model_val.pt",
        )
        self.assertEqual(
            Path(config.hlt2_s0p35_source_checkpoint),
            self.layout.source_model_dir("hlt2_part_s0p35_seed8831") / "best_model_val.pt",
        )
        self.assertEqual(
            Path(config.hlt2_s1p00_source_checkpoint),
            self.layout.source_model_dir("hlt2_part_s1p00_seed8841") / "best_model_val.pt",
        )
        self.assertEqual(config.head_warmup_epochs, 1)
        self.assertFalse(config.amp)
        self.assertEqual(config.max_train_jets, 5_000_000)
        self.assertEqual(config.max_val_jets, 1_000_000)
        self.assertEqual(config.max_final_test_jets, 1_000_000)
        self.assertEqual(HLT_MV_TRIVIEW_CONTRACT, "hlt_multiview_triview_particle_fusion_v1")
        self.assertEqual(HLT_MV_TRIVIEW_EXPERIMENT_STEP, "hlt_mv_step7_triview_particle_fusion")
        self.assertFalse(HLT_MV_TRIVIEW_DEFAULT_AMP)

    def test_triview_rejects_wrong_name_and_missing_head_warmup(self):
        with self.assertRaises(ValueError):
            normalize_hlt_mv_triview_name("tri_hlt_hlt2_s0p20_s1p00")
        with self.assertRaises(ValueError):
            build_hlt_mv_triview_config(
                model_name="tri_hlt_hlt2_s0p20_s1p00",
                layout=self.layout,
                confirm_final_test=True,
            )
        with self.assertRaises(ValueError):
            build_hlt_mv_triview_config(
                layout=self.layout,
                confirm_final_test=True,
                head_warmup_epochs=0,
            )

    def test_prediction_paths_match_triview_layout(self):
        npz_path, meta_path = hlt_mv_triview_prediction_paths(self.layout, "final_test")
        expected_root = self.layout.triview_model_dir(HLT_MV_TRIVIEW_MODEL_NAME) / "predictions" / HLT_MV_TRIVIEW_MODEL_NAME

        self.assertEqual(npz_path, expected_root / "final_test_predictions.npz")
        self.assertEqual(meta_path, expected_root / "final_test_predictions_metadata.json")
        with self.assertRaises(ValueError):
            hlt_mv_triview_prediction_paths(self.layout, "model_train")

    def test_training_wrapper_refuses_noncanonical_settings_before_running(self):
        config = build_hlt_mv_triview_config(
            layout=self.layout,
            confirm_final_test=True,
        )

        with self.assertRaises(ValueError):
            train_hlt_mv_triview(replace(config, model_name="wrong_tri_view"))
        with self.assertRaises(ValueError):
            train_hlt_mv_triview(replace(config, head_warmup_epochs=0))

    def test_slurm_wrapper_reuses_sources_and_existing_caches(self):
        text = Path("sbatch/run_hlt_mv_train_triview.sh").read_text()

        self.assertIn("#SBATCH --job-name=hlt_mv_tri", text)
        self.assertIn("scripts/train_hlt_mv_triview.py", text)
        self.assertIn("tri_hlt_hlt2_s0p35_s1p00", text)
        self.assertIn("hlt_second_degrade_mild_v1_s0p35", text)
        self.assertIn("hlt_second_degrade_mild_v1_s1p00", text)
        self.assertIn("HLT_MV_SOURCE_MODELS_DIR}/hlt_part_seed8801/best_model_val.pt", text)
        self.assertIn("HLT_MV_SOURCE_MODELS_DIR}/hlt2_part_s0p35_seed8831/best_model_val.pt", text)
        self.assertIn("HLT_MV_SOURCE_MODELS_DIR}/hlt2_part_s1p00_seed8841/best_model_val.pt", text)
        self.assertIn("--hlt-checkpoint", text)
        self.assertIn("--hlt2-s0p35-checkpoint", text)
        self.assertIn("--hlt2-s1p00-checkpoint", text)
        self.assertIn("hlt_mv_triview_report.json", text)
        self.assertNotIn("build_pd10_hlt2_cache.py", text)
        self.assertNotIn("build_fixed_hlt_cache.py", text)
        self.assertNotIn("hlt_triview_debug", text)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from teacher_logit_reco.hlt_multiview_source_fusion import (
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_SOURCE_PREDICTION_CONTRACT,
    HLT_MV_SOURCE_PREDICTION_EXPERIMENT_STEP,
    HLT_MV_SOURCE_PREDICTION_REPORT,
    HLT_MV_SOURCE_PREDICTION_SPLITS,
    build_hlt_mv_source_config,
    default_hlt_mv_experiment_layout,
    hlt_mv_source_prediction_paths,
    normalize_hlt_mv_source_prediction_splits,
)


class HLTMultiviewSourceFusionStep3PredictionTest(unittest.TestCase):
    def setUp(self):
        self.layout = default_hlt_mv_experiment_layout(
            output_root="/home/ryreu/atlas/Fresh_check/checkpoints",
            pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
        )

    def test_prediction_paths_follow_source_model_layout(self):
        npz_path, meta_path = hlt_mv_source_prediction_paths(
            self.layout,
            "hlt2_part_s0p20_seed8821",
            "final_test",
        )

        expected_root = (
            self.layout.source_model_dir("hlt2_part_s0p20_seed8821")
            / "predictions"
            / "hlt2_part_s0p20_seed8821"
        )
        self.assertEqual(npz_path, expected_root / "final_test_predictions.npz")
        self.assertEqual(meta_path, expected_root / "final_test_predictions_metadata.json")

    def test_prediction_split_validation_is_model_val_and_final_test_only(self):
        self.assertEqual(HLT_MV_SOURCE_PREDICTION_SPLITS, ("model_val", "final_test"))
        self.assertEqual(
            normalize_hlt_mv_source_prediction_splits(["model_val", "final_test"]),
            ("model_val", "final_test"),
        )
        with self.assertRaises(ValueError):
            normalize_hlt_mv_source_prediction_splits(["model_train"])
        with self.assertRaises(ValueError):
            normalize_hlt_mv_source_prediction_splits(["model_val", "model_val"])

    def test_prediction_config_can_disable_final_test_confirmation_for_val_only(self):
        config = build_hlt_mv_source_config(
            source_name="hlt_part_seed8801",
            layout=self.layout,
            evaluate_final_test=False,
            confirm_final_test=False,
        )

        self.assertEqual(config.prediction_contract, HLT_MV_SOURCE_PREDICTION_CONTRACT)
        self.assertFalse(config.evaluate_final_test)
        self.assertFalse(config.confirm_final_test)

    def test_prediction_wrapper_reuses_models_and_caches_only(self):
        text = Path("sbatch/run_hlt_mv_cache_source_predictions.sh").read_text()

        self.assertIn("#SBATCH --job-name=hlt_mv_pred", text)
        self.assertIn("scripts/cache_hlt_mv_source_predictions.py", text)
        self.assertIn("fresh_require_file \"${OUTPUT_DIR}/best_model_val.pt\"", text)
        self.assertIn("HLT_MV_SOURCE_PREDICTION_SPLITS:=model_val final_test", text)
        self.assertIn("prediction_cache_report.json", text)
        self.assertIn("predictions/${SOURCE_NAME}/${split}_predictions.npz", text)
        self.assertNotIn("scripts/train_hlt_mv_source_model.py", text)
        self.assertNotIn("build_pd10_hlt2_cache.py", text)
        self.assertNotIn("build_fixed_hlt_cache.py", text)

    def test_prediction_cli_exposes_prediction_step_constants(self):
        text = Path("scripts/cache_hlt_mv_source_predictions.py").read_text()

        self.assertIn("cache_hlt_mv_source_predictions", text)
        self.assertIn("normalize_hlt_mv_source_prediction_splits", text)
        self.assertEqual(HLT_MV_SOURCE_PREDICTION_EXPERIMENT_STEP, "hlt_mv_step3_source_prediction_caching")
        self.assertEqual(HLT_MV_SOURCE_PREDICTION_REPORT, "prediction_cache_report.json")


if __name__ == "__main__":
    unittest.main()

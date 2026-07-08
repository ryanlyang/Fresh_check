import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.hlt_multiview_source_fusion import (
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_FUSION_HLT_RANDOM_4SEED,
    HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SOURCE_5VIEW,
    HLT_MV_LOGIT_FUSION_METRIC_TABLE_CSV,
    HLT_MV_LOGIT_FUSION_REPORT_JSON,
    HLT_MV_LOGIT_FUSION_SUMMARY_JSON,
    HLT_MV_LOGIT_FUSION_UNIFORM_METHOD,
    HLT_MV_LOGIT_FUSION_WEIGHTED_METHOD,
    HLTMVPredictionSpec,
    default_hlt_mv_experiment_layout,
    default_hlt_mv_logit_fusion_config,
    hlt_mv_builtin_logit_fusion_specs,
    parse_hlt_mv_prediction_specs,
    run_hlt_mv_logit_fusion,
)


def _labels(n: int = 80) -> np.ndarray:
    return np.asarray([idx % 10 for idx in range(n)], dtype=np.int64)


def _jet_ids(labels: np.ndarray, split: str) -> list[JetIdentity]:
    return [
        JetIdentity(file=f"{split}_class{int(label)}.root", entry=int(index), label=int(label))
        for index, label in enumerate(labels)
    ]


def _logits(labels: np.ndarray, *, shift: int, margin: float) -> np.ndarray:
    logits = np.full((labels.shape[0], 10), -1.0, dtype=np.float32)
    for row, label in enumerate(labels):
        pred = (int(label) + int(shift)) % 10
        logits[row, pred] = float(margin)
        logits[row, int(label)] += 0.35
    return logits


def _write_block(prediction_dir: Path, model_name: str, split: str, *, shift: int, margin: float) -> None:
    labels = _labels()
    logits = _logits(labels, shift=shift, margin=margin)
    block = PredictionBlock(
        model_name=model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=labels,
        jet_ids=_jet_ids(labels, split),
        metadata={
            "model_name": model_name,
            "split": split,
            "unit_test": True,
            "requires_offline_inputs": False,
            "requires_teacher_features": False,
        },
    )
    save_prediction_block(block, prediction_dir, overwrite=True)


class HLTMultiviewSourceFusionStep4LogitFusionTest(unittest.TestCase):
    def test_builtin_fusion_specs_match_plan_layout(self):
        layout = default_hlt_mv_experiment_layout(
            output_root="/home/ryreu/atlas/Fresh_check/checkpoints",
            pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
        )

        source_specs = hlt_mv_builtin_logit_fusion_specs(HLT_MV_FUSION_SOURCE_5VIEW, layout)
        random_specs = hlt_mv_builtin_logit_fusion_specs(HLT_MV_FUSION_HLT_RANDOM_4SEED, layout)
        pretrained_specs = hlt_mv_builtin_logit_fusion_specs(HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL, layout)
        scratch_specs = hlt_mv_builtin_logit_fusion_specs(HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL, layout)

        self.assertEqual(len(source_specs), 5)
        self.assertEqual(len(random_specs), 4)
        self.assertEqual(len(pretrained_specs), 4)
        self.assertEqual(len(scratch_specs), 4)
        self.assertEqual(source_specs[0].model_name, "hlt_part_seed8801")
        self.assertEqual(
            Path(source_specs[1].prediction_dir),
            layout.source_model_dir("hlt2_part_s0p10_seed8811") / "predictions",
        )
        self.assertEqual(
            Path(scratch_specs[-1].prediction_dir),
            layout.scratch_dualview_model_dir("sdv_hlt_hlt2_s1p00_scratch") / "predictions",
        )

    def test_custom_logit_fusion_writes_reports_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            specs = []
            for model_name, shift, margin in (
                ("hlt_part_seed8801", 0, 2.0),
                ("hlt2_part_s0p20_seed8821", 1, 1.8),
                ("hlt2_part_s0p35_seed8831", 0, 1.3),
            ):
                prediction_dir = tmp / model_name / "predictions"
                _write_block(prediction_dir, model_name, "model_val", shift=shift, margin=margin)
                _write_block(prediction_dir, model_name, "final_test", shift=shift, margin=margin)
                specs.append(HLTMVPredictionSpec(model_name=model_name, prediction_dir=str(prediction_dir)))

            config = default_hlt_mv_logit_fusion_config(
                "custom_three_view",
                model_specs=specs,
                output_dir=tmp / "fusion",
                confirm_final_test=True,
                overwrite=True,
                max_weight_steps=3,
            )
            report = run_hlt_mv_logit_fusion(config)

            self.assertTrue(report["ok"])
            self.assertEqual(set(report["methods"]), {HLT_MV_LOGIT_FUSION_UNIFORM_METHOD, HLT_MV_LOGIT_FUSION_WEIGHTED_METHOD})
            weighted = report["methods"][HLT_MV_LOGIT_FUSION_WEIGHTED_METHOD]
            self.assertAlmostEqual(sum(weighted["weights"]), 1.0, places=6)
            self.assertGreaterEqual(min(weighted["weights"]), 0.0)
            self.assertIn(
                "validation_threshold_fpr",
                weighted["metrics"]["final_test"],
            )
            self.assertTrue((tmp / "fusion" / HLT_MV_LOGIT_FUSION_REPORT_JSON).exists())
            self.assertTrue((tmp / "fusion" / HLT_MV_LOGIT_FUSION_SUMMARY_JSON).exists())
            self.assertTrue((tmp / "fusion" / HLT_MV_LOGIT_FUSION_METRIC_TABLE_CSV).exists())
            self.assertTrue(
                (
                    tmp
                    / "fusion"
                    / "predictions"
                    / "custom_three_view_uniform_logit_average"
                    / "final_test_predictions.npz"
                ).exists()
            )
            self.assertTrue(
                (
                    tmp
                    / "fusion"
                    / "predictions"
                    / "custom_three_view_weighted_logit_average"
                    / "model_val_predictions_metadata.json"
                ).exists()
            )

    def test_parse_custom_specs(self):
        specs = parse_hlt_mv_prediction_specs(["hlt_part_seed8801=/tmp/a", "hlt2_part_s0p10_seed8811=/tmp/b"])

        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].model_name, "hlt_part_seed8801")
        self.assertEqual(Path(specs[1].prediction_dir), Path("/tmp/b"))
        with self.assertRaises(ValueError):
            parse_hlt_mv_prediction_specs(["bad-spec"])

    def test_slurm_wrapper_is_fusion_only(self):
        text = Path("sbatch/run_hlt_mv_logit_fusion.sh").read_text()

        self.assertIn("#SBATCH --job-name=hlt_mv_fuse", text)
        self.assertIn("scripts/run_hlt_mv_logit_fusion.py", text)
        self.assertIn("source_5view", text)
        self.assertIn("hlt_random_4seed", text)
        self.assertIn("pretrained_dualview_4model", text)
        self.assertIn("scratch_dualview_4model", text)
        self.assertIn("fresh_require_file \"${prediction_dir}/${model_name}/${split}_predictions.npz\"", text)
        self.assertNotIn("train_hlt_mv_source_model.py", text)
        self.assertNotIn("cache_hlt_mv_source_predictions.py", text)
        self.assertNotIn("build_pd10_hlt2_cache.py", text)
        self.assertNotIn("#SBATCH --gres=gpu", text)


if __name__ == "__main__":
    unittest.main()

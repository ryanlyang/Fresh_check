import json
import tempfile
import unittest
from pathlib import Path

from teacher_logit_reco.hlt_multiview_source_fusion import (
    HLTMVFinalReportConfig,
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_FINAL_REPORT_CONTRACT,
    HLT_MV_FINAL_REPORT_JSON,
    HLT_MV_FINAL_REPORT_MD,
    HLT_MV_FINAL_REPORT_METRIC_TABLE_CSV,
    HLT_MV_FINAL_REPORT_RUN_JSON,
    HLT_MV_FINAL_REPORT_SUMMARY_JSON,
    default_hlt_mv_experiment_layout,
    write_hlt_mv_final_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"


def _read_sbatch(name: str) -> str:
    return (SBATCH_DIR / name).read_text(encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _minimal_model_report(*, val_acc: float, test_acc: float) -> dict:
    return {
        "ok": True,
        "model_val_prediction_metrics": {
            "accuracy": val_acc,
            "cross_entropy": 1.0 - val_acc,
        },
        "final_test_metrics": {
            "accuracy": test_acc,
            "cross_entropy": 1.0 - test_acc,
        },
    }


class HLTMultiviewSourceFusionStep8SubmitterTest(unittest.TestCase):
    def test_final_report_writer_aggregates_available_rows_and_marks_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "checkpoints"
            layout = default_hlt_mv_experiment_layout(
                output_root=output_root,
                pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
            )
            _write_json(
                layout.source_model_dir("hlt_part_seed8801") / "run_report.json",
                _minimal_model_report(val_acc=0.71, test_acc=0.70),
            )
            _write_json(
                layout.source_model_dir("hlt2_part_s0p35_seed8831") / "run_report.json",
                _minimal_model_report(val_acc=0.73, test_acc=0.72),
            )

            report = write_hlt_mv_final_report(
                HLTMVFinalReportConfig(
                    output_root=str(output_root),
                    pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
                    output_dir=str(layout.final_report_dir),
                    allow_missing=True,
                    overwrite=True,
                )
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["contract"], HLT_MV_FINAL_REPORT_CONTRACT)
            self.assertGreater(len(report["missing_artifacts"]), 0)
            self.assertGreaterEqual(report["n_rows"], 2)
            self.assertEqual(report["best_overall"]["model_name"], "hlt2_part_s0p35_seed8831")
            self.assertTrue((layout.final_report_dir / HLT_MV_FINAL_REPORT_JSON).exists())
            self.assertTrue((layout.final_report_dir / HLT_MV_FINAL_REPORT_SUMMARY_JSON).exists())
            self.assertTrue((layout.final_report_dir / HLT_MV_FINAL_REPORT_RUN_JSON).exists())
            self.assertTrue((layout.final_report_dir / HLT_MV_FINAL_REPORT_METRIC_TABLE_CSV).exists())
            self.assertTrue((layout.final_report_dir / HLT_MV_FINAL_REPORT_MD).exists())

    def test_final_report_writer_is_strict_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "checkpoints"
            layout = default_hlt_mv_experiment_layout(
                output_root=output_root,
                pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
            )
            with self.assertRaises(FileNotFoundError):
                write_hlt_mv_final_report(
                    HLTMVFinalReportConfig(
                        output_root=str(output_root),
                        pdv3_experiment_name=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
                        output_dir=str(layout.final_report_dir),
                    )
                )

    def test_final_report_wrapper_validates_all_expected_families(self):
        text = _read_sbatch("run_hlt_mv_final_report.sh")

        self.assertIn("#SBATCH --job-name=hlt_mv_report", text)
        self.assertIn("scripts/write_hlt_mv_final_report.py", text)
        self.assertIn("HLT_MV_FINAL_REPORT_ALLOW_MISSING:=0", text)
        self.assertIn("hlt_part_seed8801", text)
        self.assertIn("hlt2_part_s0p10_seed8811", text)
        self.assertIn("hlt_part_seed9104", text)
        self.assertIn("sdv_hlt_hlt2_s1p00", text)
        self.assertIn("hlt_mv_pretrained_dualview_report.json", text)
        self.assertIn("sdv_hlt_hlt2_s1p00_scratch", text)
        self.assertIn("hlt_mv_scratch_dualview_report.json", text)
        self.assertIn("sdv_hlt_hlt_same_view", text)
        self.assertIn("tta_hlt_part_hlt_plus_hlt2_s1p00", text)
        self.assertIn("tri_hlt_hlt2_s0p35_s1p00/hlt_mv_triview_report.json", text)
        self.assertIn("source_5view hlt_random_4seed pretrained_dualview_4model scratch_dualview_4model", text)
        self.assertIn('fresh_require_file "${HLT_MV_FINAL_REPORT_DIR}/metric_table.csv"', text)
        self.assertNotIn("build_pd10_hlt2_cache.py", text)
        self.assertNotIn("build_fixed_hlt_cache.py", text)

    def test_submitter_reuses_existing_caches_and_wires_the_full_graph(self):
        text = _read_sbatch("submit_hlt_mv_source_fusion.sh")

        self.assertIn("HLT_MV_PDV3_EXPERIMENT_NAME:=privileged_distill_v3_av10_adapter_fixed_hlt_v2_realistic_s1p0_highdata_20260705_190747", text)
        self.assertIn("HLT_MV_HLT_CACHE_DIR:=${HLT_MV_PDV3_ROOT}/inputs/hlt_cache", text)
        self.assertIn("HLT_MV_HLT2_CACHE_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_self_dualview/hlt2_cache", text)
        self.assertIn('fresh_require_dir "${HLT_MV_HLT_CACHE_DIR}"', text)
        self.assertIn('fresh_require_dir "${cache_dir}"', text)
        self.assertIn("hlt_second_degrade_mild_v1_${tag}", text)
        self.assertIn('fresh_split_words tta_strengths "${HLT_MV_TTA_STRENGTHS}"', text)
        self.assertIn('for strength in "${tta_strengths[@]}"; do', text)
        self.assertLess(
            text.index('fresh_split_words tta_strengths "${HLT_MV_TTA_STRENGTHS}"'),
            text.index('for strength in "${tta_strengths[@]}"; do'),
        )
        self.assertNotIn("for strength in ${HLT_MV_TTA_STRENGTHS}; do", text)
        self.assertIn("run_hlt_mv_train_source_model.sh", text)
        self.assertIn("run_hlt_mv_logit_fusion.sh", text)
        self.assertIn("run_hlt_mv_train_pretrained_dualview.sh", text)
        self.assertIn("run_hlt_mv_train_scratch_dualview.sh", text)
        self.assertIn("run_hlt_mv_train_same_view_control.sh", text)
        self.assertIn("run_hlt_mv_eval_tta_control.sh", text)
        self.assertIn("run_hlt_mv_train_triview.sh", text)
        self.assertIn("run_hlt_mv_final_report.sh", text)
        self.assertIn("source_5view", text)
        self.assertIn("hlt_random_4seed", text)
        self.assertIn("pretrained_dualview_4model", text)
        self.assertIn("scratch_dualview_4model", text)
        self.assertIn("hlt2_s0p10_source_job_id", text)
        self.assertIn('sdv_hlt_hlt2_s0p10) hlt2_source_job_id="${hlt2_s0p10_source_job_id}"', text)
        self.assertIn('variant_dep="$(join_nonempty_by_colon "${base_dep}" "${canonical_hlt_source_job_id}" "${hlt2_source_job_id}")"', text)
        self.assertIn('"${variant_dep}"', text)
        self.assertIn('"${base_dep}" \\', text)
        self.assertIn('"${canonical_hlt_source_job_id}" "${hlt2_s0p35_source_job_id}" "${hlt2_s1p00_source_job_id}"', text)
        self.assertIn("final_dep=\"$(join_nonempty_by_colon", text)
        self.assertIn("CONFIRM_FINAL_TEST", text)
        self.assertNotIn("run_pd10_build_hlt2_cache.sh", text)
        self.assertNotIn("build_pd10_hlt2_cache.py", text)
        self.assertNotIn("build_fixed_hlt_cache.py", text)

    def test_submitter_has_deep_skip_and_partial_output_guards(self):
        text = _read_sbatch("submit_hlt_mv_source_fusion.sh")

        self.assertIn("skip_existing_model_with_predictions()", text)
        self.assertIn("skip_existing_tta_control()", text)
        self.assertIn("skip_existing_fusion()", text)
        self.assertIn("skip_existing_final_report()", text)
        self.assertIn("refuse_partial_existing_output_dir()", text)
        self.assertIn('fresh_bool_enabled "${OVERWRITE}"', text)
        self.assertIn("best_model_val.pt", text)
        self.assertIn("training_curves.json", text)
        self.assertIn("final_test_predictions_metadata.json", text)
        self.assertIn("metric_table.csv", text)
        self.assertIn("hlt_multiview_source_fusion_report.md", text)

    def test_control_wrappers_are_hlt_only_and_reuse_source_checkpoint(self):
        same_view = _read_sbatch("run_hlt_mv_train_same_view_control.sh")
        tta = _read_sbatch("run_hlt_mv_eval_tta_control.sh")

        self.assertIn("scripts/train_pd10_hlt_self_dualview.py", same_view)
        self.assertIn("--hlt-cache-dir \"${HLT_MV_HLT_CACHE_DIR}\"", same_view)
        self.assertIn("hlt_part_seed8801/best_model_val.pt", same_view)
        self.assertIn("sdv_hlt_hlt_same_view", same_view)
        self.assertNotIn("teacher-logit", same_view.lower())
        self.assertNotIn("--offline", same_view.lower())
        self.assertNotIn("build_pd10_hlt2_cache.py", same_view)

        self.assertIn("scripts/evaluate_pd10_hlt_tta_control.py", tta)
        self.assertIn("--hlt-cache-dir \"${HLT_MV_HLT_CACHE_DIR}\"", tta)
        self.assertIn("--hlt2-cache-dir \"${HLT2_CACHE_DIR}\"", tta)
        self.assertIn("hlt_part_seed8801/best_model_val.pt", tta)
        self.assertIn("hlt_second_degrade_mild_v1_${STRENGTH_TAG}", tta)
        self.assertNotIn("--offline", tta.lower())
        self.assertNotIn("build_pd10_hlt2_cache.py", tta)


if __name__ == "__main__":
    unittest.main()

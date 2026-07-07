import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"


def _read(name: str) -> str:
    return (SBATCH_DIR / name).read_text(encoding="utf-8")


class HLTSDVStep8SlurmTest(unittest.TestCase):
    def test_common_defines_hlt_sdv_defaults_and_helpers(self):
        text = _read("common.sh")
        self.assertIn("PD10_ROOT:=${OUTPUT_ROOT}/privileged_distill_10class_5m_hlt0p4_run1", text)
        self.assertIn("PD10_HLT_SDV_ROOT:=${PD10_ROOT}/hlt_self_dualview", text)
        self.assertIn("PD10_HLT_SDV_STRENGTHS:=0.00 0.10 0.20 0.35 1.00", text)
        self.assertIn(
            "sdv_hlt_hlt_same_view sdv_hlt_hlt2_s0p10 sdv_hlt_hlt2_s0p20 sdv_hlt_hlt2_s0p35 sdv_hlt_hlt2_s1p00",
            text,
        )
        self.assertIn("hlt2_only_part_s0p20 tta_hlt_part_hlt_plus_hlt2_s0p20", text)
        self.assertIn("PD10_HLT_SDV_REPORT_REQUIRE_ANCHORS:=1", text)
        self.assertIn("PD10_HLT_SDV_SUBMIT_ANCHORS:=1", text)
        self.assertIn("PD10_HLT_SDV_RETRAIN_ANCHORS:=0", text)
        self.assertIn(
            "PD10_HLT_SDV_WARM_CE_ANCHOR_SPEC:=warm_start|none|full_logits|2.0|0.5|3|pd10_student_warm_start_ce_only",
            text,
        )
        self.assertIn(
            "PD10_HLT_SDV_WARM_DUAL_KD_ANCHOR_SPEC:=warm_start|dual_view|full_logits|2.0|0.5|3|pd10_student_warm_start_dual_view_full_logits_t2_a0p5",
            text,
        )
        self.assertIn("fresh_pd10_hlt_sdv_strength_tag()", text)
        self.assertIn("fresh_pd10_hlt_sdv_hlt2_cache_dir()", text)
        self.assertIn("fresh_pd10_hlt_sdv_strength_from_variant()", text)

    def test_wrappers_call_the_expected_python_entrypoints(self):
        expected = {
            "run_pd10_build_hlt2_cache.sh": "scripts/build_pd10_hlt2_cache.py",
            "run_pd10_audit_hlt2_cache.sh": "scripts/audit_pd10_hlt_self_dualview_inputs.py",
            "run_pd10_train_hlt_self_dualview.sh": "scripts/train_pd10_hlt_self_dualview.py",
            "run_pd10_train_hlt2_only_control.sh": "scripts/train_pd10_hlt2_only_control.py",
            "run_pd10_eval_hlt_tta_control.sh": "scripts/evaluate_pd10_hlt_tta_control.py",
            "run_pd10_write_hlt_self_dualview_report.sh": "scripts/write_pd10_hlt_self_dualview_report.py",
        }
        for wrapper, entrypoint in expected.items():
            with self.subTest(wrapper=wrapper):
                text = _read(wrapper)
                self.assertTrue(text.startswith("#!/usr/bin/env bash"))
                self.assertIn("source \"${SCRIPT_DIR}/common.sh\"", text)
                self.assertIn(entrypoint, text)
                self.assertIn("fresh_write_run_config", text)
                self.assertIn("fresh_run", text)

    def test_train_and_control_wrappers_are_hlt_only_deployable(self):
        train_text = _read("run_pd10_train_hlt_self_dualview.sh")
        self.assertIn("--hlt-cache-dir \"${PD10_HLT_CACHE_DIR}\"", train_text)
        self.assertIn("--hlt-teacher-checkpoint \"${PD10_HLT_SDV_HLT_TEACHER_CHECKPOINT}\"", train_text)
        self.assertIn("sdv_hlt_hlt_same_view", train_text)
        self.assertIn("fresh_pd10_hlt_sdv_strength_from_variant", train_text)
        self.assertNotIn("offline", train_text.lower())
        self.assertNotIn("teacher-logit", train_text.lower())

        hlt2_text = _read("run_pd10_train_hlt2_only_control.sh")
        self.assertIn("--hlt2-cache-dir \"${HLT2_CACHE_DIR}\"", hlt2_text)
        self.assertIn("--no-warm-start", hlt2_text)

        tta_text = _read("run_pd10_eval_hlt_tta_control.sh")
        self.assertIn("--hlt-cache-dir \"${PD10_HLT_CACHE_DIR}\"", tta_text)
        self.assertIn("--hlt2-cache-dir \"${HLT2_CACHE_DIR}\"", tta_text)

    def test_submitter_supports_smoke_skip_overwrite_and_pd10_root_override(self):
        text = _read("submit_pd10_hlt_self_dualview.sh")
        self.assertIn(": \"${SMOKE:=0}\"", text)
        self.assertIn("hlt_self_dualview_smoke_20k_5k_10k", text)
        self.assertIn("PD10_HLT_SDV_SMOKE_MODEL_TRAIN_SIZE:=20000", text)
        self.assertIn("PD10_HLT_SDV_SMOKE_MODEL_VAL_SIZE:=5000", text)
        self.assertIn("PD10_HLT_SDV_SMOKE_FINAL_TEST_SIZE:=10000", text)
        self.assertIn("SKIP_EXISTING", text)
        self.assertIn("OVERWRITE", text)
        self.assertIn("PD10_ROOT", text)
        self.assertIn("CONDA_ENV", text)

    def test_skip_existing_does_not_override_overwrite_and_uses_deep_sentinels(self):
        text = _read("submit_pd10_hlt_self_dualview.sh")
        self.assertIn('fresh_bool_enabled "${OVERWRITE}"', text)
        self.assertIn("json_ok_true()", text)
        self.assertIn("refuse_partial_existing_output_dir()", text)
        self.assertIn("Use OVERWRITE=1, remove the partial output, or choose a fresh output root", text)
        self.assertIn("skip_existing_hlt2_cache()", text)
        self.assertIn("skip_existing_trained_model()", text)
        self.assertIn("skip_existing_teacher_anchor()", text)
        self.assertIn("skip_existing_teacher_logits_anchor()", text)
        self.assertIn("skip_existing_dual_view_anchor()", text)
        self.assertIn("skip_existing_student_anchor()", text)
        self.assertIn("skip_existing_tta_control()", text)
        self.assertIn("skip_existing_final_report()", text)
        self.assertIn('fresh_bool_enabled "${PD10_HLT_SDV_RETRAIN_ANCHORS}"', text)
        self.assertIn("anchor_job_export_arg()", text)
        self.assertIn("--export=ALL,OVERWRITE=1", text)
        self.assertIn("model_train_fixed_hlt.npz", text)
        self.assertIn("model_val_fixed_hlt_metadata.json", text)
        self.assertIn("final_test_fixed_hlt.npz", text)
        self.assertIn("best_model_val.pt", text)
        self.assertIn("training_curves.json", text)
        self.assertIn("final_test_report.json", text)
        self.assertIn("metric_table.csv", text)
        self.assertIn("binary_projection_table.csv", text)
        self.assertIn('skip_existing_json_ok "pd10_hlt2_audit_${tag}" "${audit_dir}/hlt2_cache_audit_report.json"', text)
        self.assertIn('refuse_partial_existing_output_dir "pd10_hlt2_cache_${tag}" "${cache_dir}"', text)
        self.assertIn('refuse_partial_existing_output_dir "pd10_hlt_sdv_${variant}" "${model_dir}"', text)
        self.assertIn('refuse_partial_existing_output_dir "pd10_hlt_sdv_report" "${PD10_HLT_SDV_FINAL_REPORT_DIR}"', text)

    def test_anchor_skips_are_independent_of_global_skip_existing(self):
        text = _read("submit_pd10_hlt_self_dualview.sh")
        anchor_skip_condition = (
            'if fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}" '
            '|| fresh_bool_enabled "${PD10_HLT_SDV_RETRAIN_ANCHORS}"; then'
        )
        old_condition = (
            'if ! fresh_bool_enabled "${SKIP_EXISTING}" || fresh_is_dry_run '
            '|| fresh_bool_enabled "${OVERWRITE}" || fresh_bool_enabled "${PD10_HLT_SDV_RETRAIN_ANCHORS}"; then'
        )
        self.assertEqual(text.count(anchor_skip_condition), 4)
        self.assertNotIn(old_condition, text)

    def test_submitter_wires_cache_audit_models_controls_report(self):
        text = _read("submit_pd10_hlt_self_dualview.sh")
        self.assertIn("run_pd10_build_hlt2_cache.sh", text)
        self.assertIn("run_pd10_audit_hlt2_cache.sh", text)
        self.assertIn("run_pd10_train_hlt_self_dualview.sh", text)
        self.assertIn("run_pd10_train_hlt2_only_control.sh", text)
        self.assertIn("run_pd10_eval_hlt_tta_control.sh", text)
        self.assertIn("run_pd10_write_hlt_self_dualview_report.sh", text)
        self.assertIn(
            "model_base_dep=\"$(join_nonempty_by_colon \"${base_dep}\" \"${hlt_teacher_anchor_job_id}\" \"${cache_job_ids[@]}\" \"${audit_job_ids[@]}\")\"",
            text,
        )
        self.assertIn(
            "report_dep=\"$(join_nonempty_by_colon \"${base_dep}\" \"${cache_job_ids[@]}\" \"${audit_job_ids[@]}\" \"${anchor_job_ids[@]}\" \"${model_job_ids[@]}\" \"${control_job_ids[@]}\")\"",
            text,
        )
        self.assertIn("pd10_hlt_self_dualview_submission:", text)
        self.assertIn("dependency_summary:", text)

    def test_report_wrapper_validates_all_report_artifacts(self):
        text = _read("run_pd10_write_hlt_self_dualview_report.sh")
        self.assertIn('fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/summary.json"', text)
        self.assertIn('fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/hlt_self_dualview_report.json"', text)
        self.assertIn('fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/hlt_self_dualview_report.md"', text)
        self.assertIn('fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/metric_table.csv"', text)
        self.assertIn('fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/comparison_table.csv"', text)
        self.assertIn('fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/binary_projection_table.csv"', text)
        self.assertIn('fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/run_report.json"', text)

    def test_submitter_queues_required_report_anchor_graph(self):
        text = _read("submit_pd10_hlt_self_dualview.sh")
        self.assertIn("PD10_HLT_SDV_SUBMIT_ANCHORS", text)
        self.assertIn("PD10_HLT_SDV_RETRAIN_ANCHORS", text)
        self.assertIn("anchor_job_ids=()", text)
        self.assertIn("hlt_teacher_anchor_job_id", text)
        self.assertIn("offline_teacher_anchor_job_id", text)
        self.assertIn("run_pd10_train_teacher.sh", text)
        self.assertIn("run_pd10_cache_teacher_logits.sh", text)
        self.assertIn("run_pd10_train_dual_view_teacher.sh", text)
        self.assertIn("run_pd10_train_student.sh", text)
        self.assertIn("PD10_HLT_SDV_WARM_CE_ANCHOR_SPEC", text)
        self.assertIn("PD10_HLT_SDV_WARM_DUAL_KD_ANCHOR_SPEC", text)
        self.assertIn("warm_ce_variant", text)
        self.assertIn("warm_dual_kd_variant", text)
        self.assertIn("teacher_logit_manifest.json", text)
        self.assertIn("student_predictions/${variant}/model_val_predictions.npz", text)
        self.assertIn("student_predictions/${variant}/final_test_predictions.npz", text)


if __name__ == "__main__":
    unittest.main()

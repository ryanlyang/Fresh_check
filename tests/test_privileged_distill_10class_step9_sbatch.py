from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"
PD10_DIR = REPO_ROOT / "teacher_logit_reco" / "privileged_distill_10class"


class PD10Step9SbatchTests(unittest.TestCase):
    def read(self, name: str) -> str:
        return (SBATCH_DIR / name).read_text(encoding="utf-8")

    def read_pd10(self, name: str) -> str:
        return (PD10_DIR / name).read_text(encoding="utf-8")

    def test_common_defines_student_report_and_submission_controls(self):
        common = self.read("common.sh")

        self.assertIn("SKIP_EXISTING:=0", common)
        self.assertIn("CONFIRM_FINAL_TEST:=0", common)
        self.assertIn("PD10_STUDENTS_DIR:=${PD10_ROOT}/students", common)
        self.assertIn("PD10_FINAL_REPORT_DIR:=${PD10_ROOT}/final_report", common)
        self.assertIn("PD10_STUDENT_CORE_SPECS", common)
        self.assertIn("pd10_student_scratch_ce_only", common)
        self.assertIn("pd10_student_warm_start_dual_view_full_logits_t2_a0p5", common)
        self.assertIn("PD10_STUDENT_PRIORITY_SPECS", common)
        self.assertIn("pd10_student_warm_start_dual_view_top3_t2_a0p5", common)
        self.assertIn("PD10_STUDENT_WARM_START_BASELINE_CHECKPOINT", common)
        self.assertIn("PD10_REPORT_ALLOW_MISSING_CORE_STUDENTS", common)
        self.assertIn('"PD10_STUDENTS_DIR"', common)
        self.assertIn('"PD10_REPORT_SKIP_PREDICTION_METRICS"', common)
        self.assertIn('"SKIP_EXISTING"', common)

    def test_student_runner_trains_one_spec_and_checks_artifacts(self):
        runner = self.read("run_pd10_train_student.sh")

        self.assertIn("#SBATCH --job-name=pd10_student", runner)
        self.assertIn("Spec format: init|teacher|target_mode|temperature|kd_alpha|top_k|variant_name", runner)
        self.assertIn("scripts/train_pd10_student.py", runner)
        self.assertIn("--student-init \"${STUDENT_INIT}\"", runner)
        self.assertIn("--teacher-target \"${TEACHER_TARGET}\"", runner)
        self.assertIn("--target-mode \"${TARGET_MODE}\"", runner)
        self.assertIn("--teacher-logit-cache \"${PD10_TEACHER_LOGITS_DIR}\"", runner)
        self.assertIn("--output-dir \"${OUTPUT_DIR}\"", runner)
        self.assertIn("--baseline-checkpoint", runner)
        self.assertIn("PD10_STUDENT_WARM_START_BASELINE_CHECKPOINT", runner)
        self.assertIn("teacher_logit_manifest.json", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn("PD10_STUDENT_SKIP_FINAL_TEST", runner)
        self.assertIn("final_test_report.json", runner)
        self.assertIn("fresh_assert_json_ok \"${OUTPUT_DIR}/run_report.json\"", runner)

    def test_report_runner_writes_all_step8_outputs(self):
        runner = self.read("run_pd10_write_report.sh")

        self.assertIn("#SBATCH --job-name=pd10_report", runner)
        self.assertIn("scripts/write_pd10_report.py", runner)
        self.assertIn("--teachers-dir \"${PD10_TEACHERS_DIR}\"", runner)
        self.assertIn("--students-dir \"${PD10_STUDENTS_DIR}\"", runner)
        self.assertIn("--teacher-logit-dir \"${PD10_TEACHER_LOGITS_DIR}\"", runner)
        self.assertIn("--audit-dir \"${PD10_STEP2_AUDIT_DIR}\"", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn("PD10_REPORT_STUDENT_VARIANTS", runner)
        self.assertIn("--allow-missing-core-students", runner)
        self.assertIn("pd10_report.json", runner)
        self.assertIn("student_core_matrix.csv", runner)
        self.assertIn("gap_closure_table.csv", runner)
        self.assertIn("leakage_audit_summary.csv", runner)
        self.assertIn("fresh_assert_json_ok \"${PD10_FINAL_REPORT_DIR}/run_report.json\"", runner)

    def test_master_submitter_queues_dependency_graph_and_supports_skip_existing(self):
        submitter = self.read("submit_pd10_experiment.sh")

        self.assertIn("CONFIRM_FINAL_TEST=1", submitter)
        self.assertIn("SKIP_EXISTING", submitter)
        self.assertIn("skip_existing_artifact", submitter)
        self.assertIn("run_pd10_build_splits.sh", submitter)
        self.assertIn("run_pd10_build_hlt_cache.sh", submitter)
        self.assertIn("run_pd10_audit_splits_hlt_cache.sh", submitter)
        self.assertIn("run_pd10_train_teacher.sh", submitter)
        self.assertIn("run_pd10_cache_teacher_logits.sh", submitter)
        self.assertIn("run_pd10_train_dual_view_teacher.sh", submitter)
        self.assertIn("run_pd10_train_student.sh", submitter)
        self.assertIn("run_pd10_write_report.sh", submitter)
        self.assertIn("teacher_logits_after_teachers_and_audit", submitter)
        self.assertIn('"${teacher_logit_job_ids[@]}"', submitter)
        self.assertIn("students_afterok: teacher-specific plus audit", submitter)
        self.assertIn("final_report_afterok", submitter)
        self.assertIn("pd10_experiment_submission", submitter)
        self.assertIn("student_variants", submitter)
        self.assertIn("total_skipped_existing", submitter)
        self.assertIn("PD10_MODEL_TRAIN_SIZE", submitter)
        self.assertIn("PD10_FINAL_TEST_SIZE", submitter)

    def test_prediction_caches_disable_amp_and_sanitize_tiny_nonfinite_tails(self):
        logits = self.read_pd10("logits.py")
        students = self.read_pd10("students.py")

        for source in (logits, students):
            self.assertIn("sanitize_prediction_logits", source)
            self.assertIn("_disable_model_amp_for_eval", source)
            self.assertIn("amp_disabled_for_eval", source)
            self.assertIn("logit_sanitization", source)


if __name__ == "__main__":
    unittest.main()

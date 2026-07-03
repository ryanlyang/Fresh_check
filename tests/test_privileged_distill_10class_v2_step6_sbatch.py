from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"
SCRIPTS_DIR = REPO_ROOT / "scripts"


class PD10V2Step6SbatchTests(unittest.TestCase):
    def read_sbatch(self, name: str) -> str:
        return (SBATCH_DIR / name).read_text(encoding="utf-8")

    def read_script(self, name: str) -> str:
        return (SCRIPTS_DIR / name).read_text(encoding="utf-8")

    def test_common_defines_v2_roots_and_student_matrix(self):
        common = self.read_sbatch("common.sh")

        self.assertIn("PD10_V2_ROOT:=${PD10_ROOT}/v2_repkd_particle_dualview", common)
        self.assertIn("PD10_V2_TEACHER_REPRESENTATIONS_DIR:=${PD10_V2_ROOT}/teacher_representations", common)
        self.assertIn("PD10_V2_STUDENTS_DIR:=${PD10_V2_ROOT}/students", common)
        self.assertIn("PD10_V2_FINAL_REPORT_DIR:=${PD10_V2_ROOT}/final_report", common)
        self.assertIn("PD10_V2_PARTICLE_DUAL_VIEW_EPOCHS", common)
        self.assertIn("warm_particle_dual_logit_kd", common)
        self.assertIn("warm_particle_dual_rep_kd", common)
        self.assertIn("warm_particle_dual_logit_rep_kd", common)
        self.assertIn("warm_logit_fusion_dual_logit_rep_kd", common)
        self.assertIn("scratch_particle_dual_logit_rep_kd", common)
        self.assertIn("particle|particle_dual|particle_dual_view", common)
        self.assertIn("PD10_V2_REPORT_ALLOW_MISSING_CORE_STUDENTS:=0", common)
        self.assertIn("PD10_STUDENT_ALIGN_PREDICTION_TO_TEACHER_CACHE:=0", common)

    def test_particle_teacher_and_cache_runners_call_v2_clis(self):
        train = self.read_sbatch("run_pd10_train_particle_dual_view_teacher.sh")
        cache = self.read_sbatch("run_pd10_cache_particle_dual_view_teacher.sh")

        self.assertIn("CONDA_ENV:=atlas_kd", train)
        self.assertIn("#SBATCH --job-name=pd10_pdv_teacher", train)
        self.assertIn("scripts/train_pd10_particle_dual_view_teacher.py", train)
        self.assertIn("--hlt-teacher-checkpoint", train)
        self.assertIn("--offline-teacher-checkpoint", train)
        self.assertIn("--max-train-jets \"${PD10_MODEL_TRAIN_SIZE}\"", train)
        self.assertIn("fresh_assert_json_ok \"${OUTPUT_DIR}/run_report.json\"", train)

        self.assertIn("CONDA_ENV:=atlas_kd", cache)
        self.assertIn("#SBATCH --job-name=pd10_pdv_cache", cache)
        self.assertIn("scripts/cache_pd10_particle_dual_view_teacher.py", cache)
        self.assertIn("--logit-output-dir \"${PD10_V2_TEACHER_LOGITS_DIR}\"", cache)
        self.assertIn("--representation-output-dir \"${PD10_V2_TEACHER_REPRESENTATIONS_DIR}\"", cache)
        self.assertIn("particle_dual_view_cache_manifest.json", cache)
        self.assertIn("teacher_representation_manifest.json", cache)

    def test_dual_view_representation_cache_runner_and_cli_exist(self):
        runner = self.read_sbatch("run_pd10_cache_dual_view_representations.sh")
        script = self.read_script("cache_pd10_dual_view_representations.py")

        self.assertIn("#SBATCH --job-name=pd10_dual_rep", runner)
        self.assertIn("scripts/cache_pd10_dual_view_representations.py", runner)
        self.assertIn("--teacher-logit-dir \"${PD10_TEACHER_LOGITS_DIR}\"", runner)
        self.assertIn("--output-dir \"${PD10_V2_TEACHER_REPRESENTATIONS_DIR}\"", runner)
        self.assertIn("PD10_V2_REPRESENTATION_CACHE_NO_SKIP_EXISTING", runner)
        self.assertIn("teacher_representation_manifest.json", runner)

        self.assertIn("PD10DualViewRepresentationCacheConfig", script)
        self.assertIn("cache_pd10_dual_view_teacher_representations", script)
        self.assertIn("--confirm-final-test", script)

    def test_student_and_report_runners_understand_representations(self):
        student = self.read_sbatch("run_pd10_train_student.sh")
        report = self.read_sbatch("run_pd10_write_report.sh")

        self.assertIn("representation_beta|representation_mode|representation_dim", student)
        self.assertIn("REPRESENTATION_BETA", student)
        self.assertIn("--teacher-representation-cache \"${PD10_TEACHER_REPRESENTATIONS_DIR}\"", student)
        self.assertIn("--align-prediction-to-teacher-cache", student)
        self.assertIn("rep_only|full_logits_plus_rep|top3_plus_rep|confidence_weighted_plus_rep", student)
        self.assertIn("v2_comparisons.csv", report)
        self.assertIn("v2_diagnostics.csv", report)

    def test_v2_submitter_queues_dependency_graph(self):
        submitter = self.read_sbatch("submit_pd10_v2_repkd_particle_dualview.sh")

        self.assertIn("CONFIRM_FINAL_TEST=1", submitter)
        self.assertIn("CONDA_ENV:=atlas_kd", submitter)
        self.assertIn("UPSTREAM_DEPENDENCY", submitter)
        self.assertIn("PD10_V2_GLOBAL_UPSTREAM_DEPENDENCY", submitter)
        self.assertIn("PD10_V2_PARTICLE_TEACHER_UPSTREAM_DEPENDENCY", submitter)
        self.assertIn("PD10_V2_DUAL_REP_UPSTREAM_DEPENDENCY", submitter)
        self.assertIn("PD10_V2_REPORT_ANCHOR_UPSTREAM_DEPENDENCY", submitter)
        self.assertIn("particle_teacher_base_dep", submitter)
        self.assertIn("dual_rep_base_dep", submitter)
        self.assertIn("report_anchor_dep", submitter)
        self.assertIn("fresh_require_file_unless_deferred", submitter)
        self.assertIn("prepare_anchor_student_links", submitter)
        self.assertIn("prepare_anchor_teacher_links", submitter)
        self.assertIn("prepare_anchor_teacher_logit_links", submitter)
        self.assertIn("pd10_student_warm_start_hlt_full_logits_t2_a0p5", submitter)
        self.assertIn("pd10_student_warm_start_offline_full_logits_t2_a0p5", submitter)
        self.assertIn("run_pd10_train_particle_dual_view_teacher.sh", submitter)
        self.assertIn("run_pd10_cache_particle_dual_view_teacher.sh", submitter)
        self.assertIn("run_pd10_cache_dual_view_representations.sh", submitter)
        self.assertIn("run_pd10_train_student.sh", submitter)
        self.assertIn("run_pd10_write_report.sh", submitter)
        self.assertIn("PD10_STUDENTS_DIR=${PD10_V2_STUDENTS_DIR}", submitter)
        self.assertIn("PD10_TEACHER_LOGITS_DIR=${PD10_V2_TEACHER_LOGITS_DIR}", submitter)
        self.assertIn("PD10_TEACHERS_DIR=${PD10_V2_TEACHERS_DIR}", submitter)
        self.assertIn("PD10_FINAL_REPORT_DIR=${PD10_V2_FINAL_REPORT_DIR}", submitter)
        self.assertIn("particle_teacher_afterok: ${particle_teacher_base_dep:-none}", submitter)
        self.assertIn("dual_view_representations_afterok: ${dual_rep_dep:-none}", submitter)
        self.assertIn("students_afterok: teacher-specific representation/cache jobs", submitter)
        self.assertIn('report_dep="$(join_nonempty_by_colon "${report_anchor_dep}"', submitter)
        self.assertIn("pd10_v2_repkd_particle_dualview_submission", submitter)


if __name__ == "__main__":
    unittest.main()

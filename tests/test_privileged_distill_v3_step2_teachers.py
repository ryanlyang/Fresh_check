from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"


class PDV3Step2TeacherTests(unittest.TestCase):
    def test_common_defines_step2_teacher_cache_contract(self):
        common = (SBATCH_DIR / "common.sh").read_text(encoding="utf-8")

        self.assertIn("PDV3_TEACHERS_DIR:=${PDV3_ROOT}/teachers", common)
        self.assertIn("PDV3_TEACHER_LOGITS_DIR:=${PDV3_ROOT}/teacher_logits", common)
        self.assertIn("PDV3_V2_ROOT:=${PDV3_ROOT}/v2_repkd_particle_dualview", common)
        self.assertIn(
            "PDV3_TEACHER_REPRESENTATIONS_DIR:=${PDV3_V2_ROOT}/teacher_representations",
            common,
        )
        self.assertIn("PDV3_TEACHER_TARGETS:=hlt offline", common)
        self.assertIn("PDV3_TEACHER_LOGIT_TARGETS:=${PDV3_TEACHER_TARGETS}", common)
        self.assertIn("PDV3_TEACHER_LOGIT_SPLITS:=model_train model_val final_test", common)
        self.assertIn("PDV3_DUAL_VIEW_PREDICT_SPLITS:=model_train model_val final_test", common)
        self.assertIn(
            "PDV3_V2_PARTICLE_DUAL_VIEW_CACHE_SPLITS:=model_train model_val final_test",
            common,
        )
        self.assertIn("PDV3_HLT_DEGRADATION_STRENGTH:=0.2", common)

    def test_pd10_env_bridge_remaps_all_teacher_paths_to_pdv3_root(self):
        bridge = (SBATCH_DIR / "pdv3_pd10_env.sh").read_text(encoding="utf-8")

        self.assertIn('export PD10_ROOT="${PDV3_ROOT}"', bridge)
        self.assertIn('export PD10_MANIFEST_PATH="${PDV3_MANIFEST_PATH}"', bridge)
        self.assertIn('export PD10_HLT_CACHE_DIR="${PDV3_HLT_CACHE_DIR}"', bridge)
        self.assertIn('export PD10_OFFLINE_CACHE_DIR="${PDV3_OFFLINE_CACHE_DIR}"', bridge)
        self.assertIn('export PD10_TEACHERS_DIR="${PDV3_TEACHERS_DIR}"', bridge)
        self.assertIn('export PD10_TEACHER_LOGITS_DIR="${PDV3_TEACHER_LOGITS_DIR}"', bridge)
        self.assertIn(
            'export PD10_TEACHER_REPRESENTATIONS_DIR="${PDV3_TEACHER_REPRESENTATIONS_DIR}"',
            bridge,
        )
        self.assertIn('export PD10_V2_ROOT="${PDV3_V2_ROOT}"', bridge)
        self.assertIn('export PD10_V2_TEACHERS_DIR="${PDV3_TEACHERS_DIR}"', bridge)
        self.assertIn('export PD10_V2_TEACHER_LOGITS_DIR="${PDV3_TEACHER_LOGITS_DIR}"', bridge)
        self.assertIn(
            'export PD10_V2_TEACHER_REPRESENTATIONS_DIR="${PDV3_TEACHER_REPRESENTATIONS_DIR}"',
            bridge,
        )
        self.assertIn(
            'export PD10_DUAL_VIEW_TEACHER_DIR="${PDV3_TEACHERS_DIR}/dual_view_logit_teacher_10class"',
            bridge,
        )
        self.assertIn(
            'export PD10_DUAL_VIEW_TEACHER_LOGITS_DIR="${PDV3_TEACHER_LOGITS_DIR}/dual_view_logit_teacher_10class"',
            bridge,
        )
        self.assertIn(
            'export PD10_V2_PARTICLE_DUAL_VIEW_TEACHER_DIR="${PDV3_TEACHERS_DIR}/particle_dual_view_teacher_10class"',
            bridge,
        )
        self.assertIn('export PD10_HLT_DEGRADATION_STRENGTH="${PDV3_HLT_DEGRADATION_STRENGTH}"', bridge)
        self.assertIn('export PD10_MODEL_TRAIN_SIZE="${PDV3_MODEL_TRAIN_SIZE}"', bridge)
        self.assertIn('export PD10_MODEL_VAL_SIZE="${PDV3_MODEL_VAL_SIZE}"', bridge)
        self.assertIn('export PD10_FINAL_TEST_SIZE="${PDV3_FINAL_TEST_SIZE}"', bridge)

    def test_step2_wrappers_source_bridge_and_exec_pd10_tools(self):
        wrappers = {
            "run_pdv3_train_teacher.sh": "run_pd10_train_teacher.sh",
            "run_pdv3_cache_teacher_logits.sh": "run_pd10_cache_teacher_logits.sh",
            "run_pdv3_train_dual_view_teacher.sh": "run_pd10_train_dual_view_teacher.sh",
            "run_pdv3_train_particle_dual_view_teacher.sh": "run_pd10_train_particle_dual_view_teacher.sh",
            "run_pdv3_cache_particle_dual_view_teacher.sh": "run_pd10_cache_particle_dual_view_teacher.sh",
        }
        for wrapper_name, pd10_script in wrappers.items():
            with self.subTest(wrapper=wrapper_name):
                text = (SBATCH_DIR / wrapper_name).read_text(encoding="utf-8")
                self.assertIn('source "${SCRIPT_DIR}/common.sh"', text)
                self.assertIn('source "${SCRIPT_DIR}/pdv3_pd10_env.sh"', text)
                self.assertIn(f'exec bash "${{SCRIPT_DIR}}/{pd10_script}" "$@"', text)

    def test_step2_submitter_queues_teacher_graph_after_step1_contract(self):
        submitter = (SBATCH_DIR / "submit_pdv3_step2_teachers.sh").read_text(encoding="utf-8")

        self.assertIn("source \"${SCRIPT_DIR}/pdv3_pd10_env.sh\"", submitter)
        self.assertIn("PDV3_STEP1_DEPENDENCY", submitter)
        self.assertIn("pdv3_step1_input_audit_report.json", submitter)
        self.assertIn("fresh_assert_json_ok", submitter)
        self.assertIn("run_pdv3_train_teacher.sh", submitter)
        self.assertIn("run_pdv3_cache_teacher_logits.sh", submitter)
        self.assertIn("run_pdv3_train_dual_view_teacher.sh", submitter)
        self.assertIn("run_pdv3_train_particle_dual_view_teacher.sh", submitter)
        self.assertIn("run_pdv3_cache_particle_dual_view_teacher.sh", submitter)
        self.assertIn("dual_view_logit_teacher_10class", submitter)
        self.assertIn("particle_dual_view_teacher_10class", submitter)
        self.assertIn("teacher_representation_manifest.json", submitter)
        self.assertIn("particle_dual_view_cache_manifest.json", submitter)
        self.assertIn("pdv3_particle_dual_view_cache_representations", submitter)
        self.assertIn("pdv3_particle_dual_view_cache_logits", submitter)
        self.assertIn("--dependency=\"afterok:${dependency}\"", submitter)
        self.assertIn("pdv3_step2_teachers_submission", submitter)

    def test_step2_submitter_refuses_stale_default_outputs_without_skip_existing(self):
        submitter = (SBATCH_DIR / "submit_pdv3_step2_teachers.sh").read_text(encoding="utf-8")

        self.assertIn('if ! fresh_bool_enabled "${SKIP_EXISTING}"; then', submitter)
        self.assertIn('fresh_refuse_existing_dir "${PDV3_TEACHERS_DIR}/hlt_part_teacher_10class"', submitter)
        self.assertIn('fresh_refuse_existing_dir "${PDV3_TEACHERS_DIR}/offline_part_teacher_10class"', submitter)
        self.assertIn('fresh_refuse_existing_dir "${PDV3_TEACHERS_DIR}/dual_view_logit_teacher_10class"', submitter)
        self.assertIn('fresh_refuse_existing_dir "${PDV3_TEACHERS_DIR}/particle_dual_view_teacher_10class"', submitter)
        self.assertIn('fresh_refuse_existing_dir "${PDV3_TEACHER_LOGITS_DIR}/particle_dual_view_teacher_10class"', submitter)
        self.assertIn(
            'fresh_refuse_existing_dir "${PDV3_TEACHER_REPRESENTATIONS_DIR}/particle_dual_view_teacher_10class"',
            submitter,
        )


if __name__ == "__main__":
    unittest.main()

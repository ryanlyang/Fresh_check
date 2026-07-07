import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"


def _read(name: str) -> str:
    return (SBATCH_DIR / name).read_text(encoding="utf-8")


class HLTSDVStep8SlurmTest(unittest.TestCase):
    def test_common_defines_hlt_sdv_defaults_and_helpers(self):
        text = _read("common.sh")
        self.assertIn("PD10_HLT_SDV_ROOT:=${PD10_ROOT}/hlt_self_dualview", text)
        self.assertIn("PD10_HLT_SDV_STRENGTHS:=0.00 0.10 0.20 0.35 1.00", text)
        self.assertIn(
            "sdv_hlt_hlt_same_view sdv_hlt_hlt2_s0p10 sdv_hlt_hlt2_s0p20 sdv_hlt_hlt2_s0p35 sdv_hlt_hlt2_s1p00",
            text,
        )
        self.assertIn("hlt2_only_part_s0p20 tta_hlt_part_hlt_plus_hlt2_s0p20", text)
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

    def test_submitter_wires_cache_audit_models_controls_report(self):
        text = _read("submit_pd10_hlt_self_dualview.sh")
        self.assertIn("run_pd10_build_hlt2_cache.sh", text)
        self.assertIn("run_pd10_audit_hlt2_cache.sh", text)
        self.assertIn("run_pd10_train_hlt_self_dualview.sh", text)
        self.assertIn("run_pd10_train_hlt2_only_control.sh", text)
        self.assertIn("run_pd10_eval_hlt_tta_control.sh", text)
        self.assertIn("run_pd10_write_hlt_self_dualview_report.sh", text)
        self.assertIn("model_base_dep=\"$(join_nonempty_by_colon \"${base_dep}\" \"${cache_job_ids[@]}\" \"${audit_job_ids[@]}\")\"", text)
        self.assertIn("report_dep=\"$(join_nonempty_by_colon \"${base_dep}\" \"${cache_job_ids[@]}\" \"${audit_job_ids[@]}\" \"${model_job_ids[@]}\" \"${control_job_ids[@]}\")\"", text)
        self.assertIn("pd10_hlt_self_dualview_submission:", text)
        self.assertIn("dependency_summary:", text)


if __name__ == "__main__":
    unittest.main()

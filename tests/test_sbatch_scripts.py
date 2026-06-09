from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"


RUNNERS = [
    "run_build_fresh_splits.sh",
    "run_build_fresh_hlt_cache.sh",
    "run_train_fresh_hlt_baseline.sh",
    "run_train_fresh_hlt_seed.sh",
    "run_train_fresh_offline_teacher.sh",
    "run_train_fresh_reco7_variant.sh",
    "run_fuse_fresh_samehlt7_plus_hlt.sh",
    "run_fuse_fresh_hlt5_seed_control.sh",
    "run_audit_fresh_samehlt7_plus_hlt.sh",
    "run_write_fresh_final_report.sh",
    "run_v2_step6_train_m2_base.sh",
    "run_v2_step10_fuse_m2_base_plus_hlt.sh",
    "run_v2_step11_audit_m2_base_plus_hlt.sh",
    "run_v2_step7_train_variant.sh",
    "run_v2_step10_fuse_reco7_plus_hlt.sh",
    "run_v2_step11_audit_reco7_plus_hlt.sh",
    "run_independent_fusion_small.sh",
    "run_independent_fusion_large.sh",
]

SUBMITTERS = [
    "submit_fresh_hlt5_seed_control.sh",
    "submit_fresh_samehlt_reco7.sh",
    "submit_fresh_full_samehlt_reco7_vs_hlt5.sh",
    "submit_fresh_smoke_test.sh",
    "submit_v2_step6_m2_base_end_to_end.sh",
    "submit_v2_step7_reco7_plus_hlt.sh",
]


class SbatchStep14Tests(unittest.TestCase):
    def read(self, name):
        return (SBATCH_DIR / name).read_text(encoding="utf-8")

    def test_required_scripts_exist(self):
        self.assertTrue((SBATCH_DIR / "common.sh").exists())
        for name in RUNNERS + SUBMITTERS:
            self.assertTrue((SBATCH_DIR / name).exists(), name)

    def test_runners_have_sbatch_directives_and_strict_shell(self):
        for name in RUNNERS:
            text = self.read(name)
            self.assertIn("#!/usr/bin/env bash", text, name)
            self.assertIn("#SBATCH --job-name=", text, name)
            self.assertIn("#SBATCH --output=fresh_check_logs/%x_%j.out", text, name)
            self.assertIn("#SBATCH --error=fresh_check_logs/%x_%j.err", text, name)
            self.assertIn("#SBATCH --partition=", text, name)
            self.assertIn("#SBATCH --time=", text, name)
            self.assertIn("#SBATCH --mem=", text, name)
            self.assertIn("set -euo pipefail", text, name)
            self.assertIn("fresh_setup", text, name)
            self.assertIn("fresh_write_run_config", text, name)
            self.assertIn("fresh_run", text, name)

    def test_submitters_have_dry_run_and_afterok_dependencies(self):
        for name in SUBMITTERS:
            text = self.read(name)
            self.assertIn("set -euo pipefail", text, name)
            self.assertIn("fresh_prepare_submitter", text, name)
            self.assertIn("fresh_is_dry_run", text, name)
            self.assertIn("sbatch", text, name)
        master = self.read("submit_fresh_full_samehlt_reco7_vs_hlt5.sh")
        self.assertIn("afterok:${split_jid}", master)
        self.assertIn("afterok:${cache_jid}", master)
        self.assertIn("afterok:${hlt5_dep}", master)
        self.assertIn("afterok:${reco7_dep}", master)
        self.assertIn("afterok:${audit_dep}", master)

    def test_scripts_use_fresh_compute_defaults_not_old_project_code(self):
        combined = "\n".join((SBATCH_DIR / name).read_text(encoding="utf-8") for name in ["common.sh"] + RUNNERS + SUBMITTERS)
        self.assertIn("/home/ryreu/atlas/Fresh_check", combined)
        self.assertIn("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0", combined)
        self.assertNotIn("/home/ryreu/atlas/PracticeTagging/old", combined)
        self.assertNotIn("old_project", combined)

    def test_smoke_submitter_sets_protocol_tiny_sizes(self):
        text = self.read("submit_fresh_smoke_test.sh")
        self.assertIn('MODEL_TRAIN_SIZE="${MODEL_TRAIN_SIZE:-10000}"', text)
        self.assertIn('MODEL_VAL_SIZE="${MODEL_VAL_SIZE:-3000}"', text)
        self.assertIn('STACK_TRAIN_SIZE="${STACK_TRAIN_SIZE:-5000}"', text)
        self.assertIn('STACK_VAL_SIZE="${STACK_VAL_SIZE:-2000}"', text)
        self.assertIn('FINAL_TEST_SIZE="${FINAL_TEST_SIZE:-10000}"', text)
        self.assertIn('RECO7_VARIANTS="m2_base"', text)
        self.assertIn("pipeline correctness only", text)

    def test_space_separated_seed_and_variant_lists_use_helper(self):
        combined = "\n".join((SBATCH_DIR / name).read_text(encoding="utf-8") for name in RUNNERS + SUBMITTERS)
        self.assertNotIn('read -r -a seed_args <<< "${HLT5_SEEDS}"', combined)
        self.assertNotIn('read -r -a variant_args <<< "${RECO7_VARIANTS}"', combined)
        self.assertNotIn('read -r -a split_args <<< "${HLT_SPLITS}"', combined)
        for name in [
            "submit_fresh_full_samehlt_reco7_vs_hlt5.sh",
            "submit_fresh_hlt5_seed_control.sh",
            "submit_fresh_samehlt_reco7.sh",
            "submit_v2_step7_reco7_plus_hlt.sh",
            "run_fuse_fresh_hlt5_seed_control.sh",
            "run_fuse_fresh_samehlt7_plus_hlt.sh",
            "run_v2_step10_fuse_reco7_plus_hlt.sh",
            "run_independent_fusion_small.sh",
            "run_independent_fusion_large.sh",
            "run_build_fresh_hlt_cache.sh",
        ]:
            self.assertIn("fresh_split_words", self.read(name), name)
        self.assertIn("fresh_print_shell_command", self.read("common.sh"))

    def test_scripts_source_common_from_project_dir_not_slurm_spool_copy(self):
        for name in RUNNERS + SUBMITTERS:
            text = self.read(name)
            self.assertIn('SCRIPT_DIR="${PROJECT_DIR}/sbatch"', text, name)
            self.assertIn('source "${SCRIPT_DIR}/common.sh"', text, name)
            self.assertNotIn('dirname "${BASH_SOURCE[0]}"', text, name)

    def test_v2_step6_submitter_queues_training_fusion_and_audits(self):
        train = self.read("run_v2_step6_train_m2_base.sh")
        fusion = self.read("run_v2_step10_fuse_m2_base_plus_hlt.sh")
        audit = self.read("run_v2_step11_audit_m2_base_plus_hlt.sh")
        submitter = self.read("submit_v2_step6_m2_base_end_to_end.sh")
        self.assertIn("jetclass_v2_original_mechanism_step6", self.read("common.sh"))
        self.assertIn("--stage both", train)
        self.assertIn("--variants \"${V2_STEP6_VARIANT}\"", train)
        self.assertIn("--splits stack_train stack_val final_test", fusion)
        self.assertIn('CONFIRM_FINAL_TEST:=1', fusion)
        self.assertIn("--fusion-dir \"${V2_STEP6_FUSION_DIR}\"", audit)
        self.assertIn("run_v2_step6_train_m2_base.sh", submitter)
        self.assertIn("run_v2_step10_fuse_m2_base_plus_hlt.sh", submitter)
        self.assertIn("run_v2_step11_audit_m2_base_plus_hlt.sh", submitter)
        self.assertIn('--dependency="afterok:${train_jid}"', submitter)
        self.assertIn('--dependency="afterok:${fusion_jid}"', submitter)
        self.assertIn("hlt5_seed_control: true", submitter)

    def test_independent_fusion_handoff_scripts_run_small_and_large_sizes(self):
        small = self.read("run_independent_fusion_small.sh")
        large = self.read("run_independent_fusion_large.sh")
        for name, text in [
            ("run_independent_fusion_small.sh", small),
            ("run_independent_fusion_large.sh", large),
        ]:
            self.assertIn("#SBATCH --time=05:00:00", text, name)
            self.assertIn("#SBATCH --gres=gpu:1", text, name)
            self.assertIn("scripts/demo_load_and_score_models_no_fusion.py", text, name)
            self.assertIn("scripts/run_independent_fusion_from_predictions.py", text, name)
            self.assertIn("--confirm-final-test", text, name)
            self.assertIn("--feature-modes", text, name)
            self.assertIn('fresh_claim_new_dir "${RUN_OUTPUT_DIR}"', text, name)
            self.assertIn('fresh_require_file "${RUN_OUTPUT_DIR}/fusion/fusion_report.json"', text, name)
        self.assertIn('FUSION_STACK_TRAIN_SIZE:=50000', small)
        self.assertIn('FUSION_STACK_VAL_SIZE:=20000', small)
        self.assertIn('FUSION_FINAL_TEST_SIZE:=100000', small)
        self.assertIn('FUSION_MODEL_LOADING_SMALL_DIR', small)
        self.assertIn('FUSION_STACK_TRAIN_SIZE:=250000', large)
        self.assertIn('FUSION_STACK_VAL_SIZE:=50000', large)
        self.assertIn('FUSION_FINAL_TEST_SIZE:=500000', large)
        self.assertIn('FUSION_MODEL_LOADING_LARGE_DIR', large)

    def test_v2_step7_submitter_queues_seven_variants_fusion_and_audits(self):
        train = self.read("run_v2_step7_train_variant.sh")
        fusion = self.read("run_v2_step10_fuse_reco7_plus_hlt.sh")
        audit = self.read("run_v2_step11_audit_reco7_plus_hlt.sh")
        submitter = self.read("submit_v2_step7_reco7_plus_hlt.sh")
        self.assertIn("jetclass_v2_original_mechanism_step7", self.read("common.sh"))
        self.assertIn("#SBATCH --time=12:00:00", train)
        self.assertIn('VARIANT="${1:?Usage:', train)
        self.assertIn("--stage both", train)
        self.assertIn("--variants \"${VARIANT}\"", train)
        self.assertIn('fresh_claim_new_dir "${OUTPUT_DIR}"', train)
        self.assertIn('fresh_split_words variant_args "${V2_STEP7_VARIANTS}"', fusion)
        self.assertIn("--variants \"${variant_args[@]}\"", fusion)
        self.assertIn("--fusion-dir \"${V2_STEP7_FUSION_DIR}\"", audit)
        self.assertIn("run_v2_step7_train_variant.sh", submitter)
        self.assertIn("run_v2_step10_fuse_reco7_plus_hlt.sh", submitter)
        self.assertIn("run_v2_step11_audit_reco7_plus_hlt.sh", submitter)
        self.assertIn('submitter_lock_dir="${V2_STEP7_ROOT}/.submission_lock"', submitter)
        self.assertIn('fresh_claim_new_dir "${submitter_lock_dir}"', submitter)
        self.assertIn('fusion_dependency="$(fresh_join_by_colon "${variant_job_ids[@]}")"', submitter)
        self.assertIn('--dependency="afterok:${fusion_dependency}"', submitter)
        self.assertIn('--dependency="afterok:${fusion_jid}"', submitter)
        self.assertIn("hlt5_seed_control: true", submitter)


if __name__ == "__main__":
    unittest.main()

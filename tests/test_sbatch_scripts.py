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
]

SUBMITTERS = [
    "submit_fresh_hlt5_seed_control.sh",
    "submit_fresh_samehlt_reco7.sh",
    "submit_fresh_full_samehlt_reco7_vs_hlt5.sh",
    "submit_fresh_smoke_test.sh",
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


if __name__ == "__main__":
    unittest.main()

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
    "run_independent_fusion_ensemble_analysis.sh",
    "run_train_heterogeneous_hlt_arch.sh",
    "run_fuse_heterogeneous_hlt4.sh",
    "run_evaluate_offline_teacher_reference.sh",
    "run_train_teacher_logit_gt_reco.sh",
    "run_predict_teacher_logit_gt_reco.sh",
    "run_fuse_teacher_logit_gt_reco.sh",
    "run_train_teacher_logit_pn_reco.sh",
    "run_predict_teacher_logit_pn_reco.sh",
    "run_fuse_teacher_logit_pn_reco.sh",
    "run_train_teacher_logit_pfn_reco.sh",
    "run_predict_teacher_logit_pfn_reco.sh",
    "run_fuse_teacher_logit_pfn_reco.sh",
    "run_train_teacher_logit_pcnn_reco.sh",
    "run_predict_teacher_logit_pcnn_reco.sh",
    "run_fuse_teacher_logit_pcnn_reco.sh",
    "run_crossarch_build_splits.sh",
    "run_crossarch_build_hlt_cache.sh",
    "run_crossarch_audit_splits_hlt_cache.sh",
    "run_crossarch_train_offline_teacher.sh",
    "run_crossarch_train_hlt_baseline.sh",
    "run_crossarch_predict_hlt_baseline.sh",
    "run_crossarch_train_reconstructor.sh",
    "run_crossarch_predict_reconstructor.sh",
    "run_crossarch_fusion.sh",
    "run_crossarch_write_final_report.sh",
    "run_crossarch_train_reco_domain_tagger.sh",
    "run_crossarch_predict_reco_domain_tagger.sh",
    "run_crossarch_fusion_reco_domain_taggers.sh",
    "run_crossarch_split_fusion.sh",
    "run_crossarch_split_fusion_summary.sh",
    "run_crossarch_conditional_fusers_linear.sh",
    "run_crossarch_conditional_fusers_neural.sh",
    "run_crossarch_aggressive_train_reconstructor.sh",
    "run_crossarch_aggressive_predict_reconstructor.sh",
    "run_crossarch_aggressive_train_reco_domain_tagger.sh",
    "run_crossarch_aggressive_predict_reco_domain_tagger.sh",
    "run_crossarch_aggressive_fusion.sh",
    "run_crossarch_aggressive_audit.sh",
    "run_train_set_matching_reconstructor.sh",
    "run_cache_set_matching_multiview.sh",
    "run_train_five_view_tagger.sh",
    "run_audit_five_view_tagger.sh",
    "run_write_set_matching_multiview_final_report.sh",
]

SUBMITTERS = [
    "submit_fresh_hlt5_seed_control.sh",
    "submit_fresh_samehlt_reco7.sh",
    "submit_fresh_full_samehlt_reco7_vs_hlt5.sh",
    "submit_fresh_smoke_test.sh",
    "submit_v2_step6_m2_base_end_to_end.sh",
    "submit_v2_step7_reco7_plus_hlt.sh",
    "submit_heterogeneous_hlt4_fusion.sh",
    "submit_teacher_logit_gt_reco_experiment.sh",
    "submit_teacher_logit_pn_reco_experiment.sh",
    "submit_teacher_logit_pfn_reco_experiment.sh",
    "submit_teacher_logit_pcnn_reco_experiment.sh",
    "submit_crossarch_step3_offline_teachers.sh",
    "submit_crossarch_step4_hlt_baselines.sh",
    "submit_crossarch_step5_reconstructors.sh",
    "submit_crossarch_step6_predictions.sh",
    "submit_crossarch_full_experiment.sh",
    "submit_crossarch_reco_domain_taggers.sh",
    "submit_crossarch_split_fusions.sh",
    "submit_crossarch_aggressive_experiment.sh",
    "submit_crossarch_aggressive_smoke_test.sh",
    "submit_set_matching_multiview_experiment.sh",
    "submit_set_matching_multiview_smoke_test.sh",
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
            "run_independent_fusion_ensemble_analysis.sh",
            "run_fuse_heterogeneous_hlt4.sh",
            "submit_heterogeneous_hlt4_fusion.sh",
            "run_fuse_teacher_logit_gt_reco.sh",
            "submit_teacher_logit_gt_reco_experiment.sh",
            "run_train_teacher_logit_pn_reco.sh",
            "run_fuse_teacher_logit_pn_reco.sh",
            "submit_teacher_logit_pn_reco_experiment.sh",
            "run_train_teacher_logit_pfn_reco.sh",
            "run_fuse_teacher_logit_pfn_reco.sh",
            "submit_teacher_logit_pfn_reco_experiment.sh",
            "run_train_teacher_logit_pcnn_reco.sh",
            "run_fuse_teacher_logit_pcnn_reco.sh",
            "submit_teacher_logit_pcnn_reco_experiment.sh",
            "run_build_fresh_hlt_cache.sh",
            "run_crossarch_build_hlt_cache.sh",
            "submit_crossarch_step3_offline_teachers.sh",
            "submit_crossarch_step4_hlt_baselines.sh",
            "run_crossarch_predict_hlt_baseline.sh",
            "run_crossarch_train_reconstructor.sh",
            "submit_crossarch_step5_reconstructors.sh",
            "run_crossarch_predict_reconstructor.sh",
            "submit_crossarch_step6_predictions.sh",
            "run_crossarch_fusion.sh",
            "submit_crossarch_full_experiment.sh",
            "run_crossarch_predict_reco_domain_tagger.sh",
            "run_crossarch_fusion_reco_domain_taggers.sh",
            "submit_crossarch_reco_domain_taggers.sh",
            "run_crossarch_split_fusion.sh",
            "run_crossarch_split_fusion_summary.sh",
            "submit_crossarch_split_fusions.sh",
            "run_train_set_matching_reconstructor.sh",
            "run_cache_set_matching_multiview.sh",
            "submit_set_matching_multiview_experiment.sh",
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

    def test_heterogeneous_hlt4_submitter_queues_four_architectures_then_fusion(self):
        train = self.read("run_train_heterogeneous_hlt_arch.sh")
        fusion = self.read("run_fuse_heterogeneous_hlt4.sh")
        submitter = self.read("submit_heterogeneous_hlt4_fusion.sh")
        common = self.read("common.sh")
        self.assertIn("jetclass_hetero_hlt4_150k_50k_300k", common)
        self.assertIn("HETERO_HLT4_ARCHITECTURES:=part pn pfn pcnn", common)
        self.assertIn("HETERO_HLT4_TRAIN_SIZE:=150000", common)
        self.assertIn("HETERO_HLT4_VAL_SIZE:=50000", common)
        self.assertIn("HETERO_HLT4_FINAL_TEST_SIZE:=300000", common)
        self.assertIn("#SBATCH --time=12:00:00", train)
        self.assertIn("--max-train-jets \"${HETERO_HLT4_TRAIN_SIZE}\"", train)
        self.assertIn("--max-val-jets \"${HETERO_HLT4_VAL_SIZE}\"", train)
        self.assertIn("scripts/train_heterogeneous_hlt.py", train)
        self.assertIn("#SBATCH --time=23:00:00", fusion)
        self.assertIn("scripts/run_heterogeneous_hlt_fusion.py", fusion)
        self.assertIn("--stack-train-size \"${HETERO_HLT4_STACK_TRAIN_SIZE}\"", fusion)
        self.assertIn("--stack-val-size \"${HETERO_HLT4_STACK_VAL_SIZE}\"", fusion)
        self.assertIn("--final-test-size \"${HETERO_HLT4_FINAL_TEST_SIZE}\"", fusion)
        self.assertIn("--confirm-final-test", fusion)
        self.assertIn("run_train_heterogeneous_hlt_arch.sh", submitter)
        self.assertIn("run_fuse_heterogeneous_hlt4.sh", submitter)
        self.assertIn('fusion_dependency="$(fresh_join_by_colon "${train_job_ids[@]}")"', submitter)
        self.assertIn('--dependency="afterok:${fusion_dependency}"', submitter)

    def test_offline_teacher_reference_runner_scores_balanced_heldout_splits(self):
        runner = self.read("run_evaluate_offline_teacher_reference.sh")
        self.assertIn("#SBATCH --time=08:00:00", runner)
        self.assertIn("#SBATCH --gres=gpu:1", runner)
        self.assertIn("scripts/evaluate_offline_teacher_reference.py", runner)
        self.assertIn("--splits stack_val final_test", runner)
        self.assertIn("--stack-val-size \"${OFFLINE_REFERENCE_STACK_VAL_SIZE}\"", runner)
        self.assertIn("--final-test-size \"${OFFLINE_REFERENCE_FINAL_TEST_SIZE}\"", runner)
        self.assertIn("--control-seed \"${HETERO_HLT4_CONTROL_SEED}\"", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn('fresh_claim_new_dir "${OFFLINE_REFERENCE_EVAL_DIR}"', runner)

    def test_teacher_logit_gt_submitter_queues_training_prediction_and_fusion(self):
        train = self.read("run_train_teacher_logit_gt_reco.sh")
        predict = self.read("run_predict_teacher_logit_gt_reco.sh")
        fusion = self.read("run_fuse_teacher_logit_gt_reco.sh")
        submitter = self.read("submit_teacher_logit_gt_reco_experiment.sh")
        common = self.read("common.sh")
        self.assertIn("teacher_logit_reco_gt", common)
        self.assertIn("TEACHER_LOGIT_GT_TEACHERS:=part", common)
        self.assertIn("TEACHER_LOGIT_GT_PART_TEACHER_CHECKPOINT", common)
        self.assertIn("fresh_teacher_logit_gt_teacher_checkpoint", common)
        self.assertIn("fresh_teacher_logit_gt_model_name", common)
        self.assertIn("#SBATCH --time=12:00:00", train)
        self.assertIn("#SBATCH --gres=gpu:1", train)
        self.assertIn("scripts/train_teacher_logit_global_transformer_reco.py", train)
        self.assertIn("--teacher-architecture \"${ARCHITECTURE}\"", train)
        self.assertIn("--max-train-jets", train)
        self.assertIn('fresh_claim_new_dir "${OUTPUT_DIR}"', train)
        self.assertIn("#SBATCH --time=05:00:00", predict)
        self.assertIn("#SBATCH --gres=gpu:1", predict)
        self.assertIn("scripts/predict_teacher_logit_global_transformer_reco.py", predict)
        self.assertIn("--prediction-dir \"${TEACHER_LOGIT_GT_PREDICTION_DIR}\"", predict)
        self.assertIn("--splits stack_train stack_val final_test", predict)
        self.assertIn("--confirm-final-test", predict)
        self.assertIn("scripts/run_independent_fusion_from_predictions.py", fusion)
        self.assertIn("--group \"teacher_logit_gt:${group_models}\"", fusion)
        self.assertIn("--confirm-final-test", fusion)
        self.assertIn("run_train_teacher_logit_gt_reco.sh", submitter)
        self.assertIn("run_predict_teacher_logit_gt_reco.sh", submitter)
        self.assertIn("run_fuse_teacher_logit_gt_reco.sh", submitter)
        self.assertIn('fresh_split_words teacher_args "${TEACHER_LOGIT_GT_TEACHERS}"', submitter)
        self.assertIn('fresh_refuse_existing_dir "${TEACHER_LOGIT_GT_PREDICTION_DIR}/${model_name}"', submitter)
        self.assertIn('--dependency="afterok:${train_jid}"', submitter)
        self.assertIn('fusion_dependency="$(fresh_join_by_colon "${predict_job_ids[@]}")"', submitter)
        self.assertIn('--dependency="afterok:${fusion_dependency}"', submitter)

    def test_teacher_logit_pn_submitter_queues_training_prediction_and_fusion(self):
        train = self.read("run_train_teacher_logit_pn_reco.sh")
        predict = self.read("run_predict_teacher_logit_pn_reco.sh")
        fusion = self.read("run_fuse_teacher_logit_pn_reco.sh")
        submitter = self.read("submit_teacher_logit_pn_reco_experiment.sh")
        common = self.read("common.sh")
        self.assertIn("teacher_logit_reco_pn", common)
        self.assertIn("TEACHER_LOGIT_PN_TEACHERS:=part", common)
        self.assertIn("TEACHER_LOGIT_PN_PART_TEACHER_CHECKPOINT", common)
        self.assertIn("fresh_teacher_logit_pn_teacher_checkpoint", common)
        self.assertIn("fresh_teacher_logit_pn_model_name", common)
        self.assertIn("#SBATCH --time=12:00:00", train)
        self.assertIn("#SBATCH --gres=gpu:1", train)
        self.assertIn("scripts/train_teacher_logit_particle_net_reco.py", train)
        self.assertIn("--teacher-architecture \"${ARCHITECTURE}\"", train)
        self.assertIn("--edgeconv-dims \"${edgeconv_dim_args[@]}\"", train)
        self.assertIn("--k \"${TEACHER_LOGIT_PN_K}\"", train)
        self.assertIn("--max-train-jets", train)
        self.assertIn('fresh_claim_new_dir "${OUTPUT_DIR}"', train)
        self.assertIn("#SBATCH --time=05:00:00", predict)
        self.assertIn("#SBATCH --gres=gpu:1", predict)
        self.assertIn("scripts/predict_teacher_logit_particle_net_reco.py", predict)
        self.assertIn("--prediction-dir \"${TEACHER_LOGIT_PN_PREDICTION_DIR}\"", predict)
        self.assertIn("--splits stack_train stack_val final_test", predict)
        self.assertIn("--confirm-final-test", predict)
        self.assertIn("scripts/run_independent_fusion_from_predictions.py", fusion)
        self.assertIn("--group \"teacher_logit_pn:${group_models}\"", fusion)
        self.assertIn("--confirm-final-test", fusion)
        self.assertIn("run_train_teacher_logit_pn_reco.sh", submitter)
        self.assertIn("run_predict_teacher_logit_pn_reco.sh", submitter)
        self.assertIn("run_fuse_teacher_logit_pn_reco.sh", submitter)
        self.assertIn('fresh_split_words teacher_args "${TEACHER_LOGIT_PN_TEACHERS}"', submitter)
        self.assertIn('fresh_refuse_existing_dir "${TEACHER_LOGIT_PN_PREDICTION_DIR}/${model_name}"', submitter)
        self.assertIn('--dependency="afterok:${train_jid}"', submitter)
        self.assertIn('fusion_dependency="$(fresh_join_by_colon "${predict_job_ids[@]}")', submitter)
        self.assertIn('--dependency="afterok:${fusion_dependency}"', submitter)

    def test_teacher_logit_pfn_submitter_queues_training_prediction_and_fusion(self):
        train = self.read("run_train_teacher_logit_pfn_reco.sh")
        predict = self.read("run_predict_teacher_logit_pfn_reco.sh")
        fusion = self.read("run_fuse_teacher_logit_pfn_reco.sh")
        submitter = self.read("submit_teacher_logit_pfn_reco_experiment.sh")
        common = self.read("common.sh")
        self.assertIn("teacher_logit_reco_pfn", common)
        self.assertIn("TEACHER_LOGIT_PFN_TEACHERS:=part", common)
        self.assertIn("TEACHER_LOGIT_PFN_PART_TEACHER_CHECKPOINT", common)
        self.assertIn("fresh_teacher_logit_pfn_teacher_checkpoint", common)
        self.assertIn("fresh_teacher_logit_pfn_model_name", common)
        self.assertIn("#SBATCH --time=12:00:00", train)
        self.assertIn("#SBATCH --gres=gpu:1", train)
        self.assertIn("scripts/train_teacher_logit_particle_flow_reco.py", train)
        self.assertIn("--teacher-architecture \"${ARCHITECTURE}\"", train)
        self.assertIn("--phi-dims \"${phi_dim_args[@]}\"", train)
        self.assertIn("--context-dim \"${TEACHER_LOGIT_PFN_CONTEXT_DIM}\"", train)
        self.assertIn("--context-dims \"${context_dim_args[@]}\"", train)
        self.assertIn("--decoder-dims \"${decoder_dim_args[@]}\"", train)
        self.assertIn("--max-train-jets", train)
        self.assertIn('fresh_claim_new_dir "${OUTPUT_DIR}"', train)
        self.assertIn("#SBATCH --time=05:00:00", predict)
        self.assertIn("#SBATCH --gres=gpu:1", predict)
        self.assertIn("scripts/predict_teacher_logit_particle_flow_reco.py", predict)
        self.assertIn("--prediction-dir \"${TEACHER_LOGIT_PFN_PREDICTION_DIR}\"", predict)
        self.assertIn("--splits stack_train stack_val final_test", predict)
        self.assertIn("--confirm-final-test", predict)
        self.assertIn("scripts/run_independent_fusion_from_predictions.py", fusion)
        self.assertIn("--group \"teacher_logit_pfn:${group_models}\"", fusion)
        self.assertIn("--confirm-final-test", fusion)
        self.assertIn("run_train_teacher_logit_pfn_reco.sh", submitter)
        self.assertIn("run_predict_teacher_logit_pfn_reco.sh", submitter)
        self.assertIn("run_fuse_teacher_logit_pfn_reco.sh", submitter)
        self.assertIn('fresh_split_words teacher_args "${TEACHER_LOGIT_PFN_TEACHERS}"', submitter)
        self.assertIn('fresh_refuse_existing_dir "${TEACHER_LOGIT_PFN_PREDICTION_DIR}/${model_name}"', submitter)
        self.assertIn('--dependency="afterok:${train_jid}"', submitter)
        self.assertIn('fusion_dependency="$(fresh_join_by_colon "${predict_job_ids[@]}")', submitter)
        self.assertIn('--dependency="afterok:${fusion_dependency}"', submitter)

    def test_teacher_logit_pcnn_submitter_queues_training_prediction_and_fusion(self):
        train = self.read("run_train_teacher_logit_pcnn_reco.sh")
        predict = self.read("run_predict_teacher_logit_pcnn_reco.sh")
        fusion = self.read("run_fuse_teacher_logit_pcnn_reco.sh")
        submitter = self.read("submit_teacher_logit_pcnn_reco_experiment.sh")
        common = self.read("common.sh")
        self.assertIn("teacher_logit_reco_pcnn", common)
        self.assertIn("TEACHER_LOGIT_PCNN_TEACHERS:=part", common)
        self.assertIn("TEACHER_LOGIT_PCNN_PART_TEACHER_CHECKPOINT", common)
        self.assertIn("fresh_teacher_logit_pcnn_teacher_checkpoint", common)
        self.assertIn("fresh_teacher_logit_pcnn_model_name", common)
        self.assertIn("#SBATCH --time=12:00:00", train)
        self.assertIn("#SBATCH --gres=gpu:1", train)
        self.assertIn("scripts/train_teacher_logit_particle_cnn_reco.py", train)
        self.assertIn("--teacher-architecture \"${ARCHITECTURE}\"", train)
        self.assertIn("--hidden-channels \"${TEACHER_LOGIT_PCNN_HIDDEN_CHANNELS}\"", train)
        self.assertIn("--num-blocks \"${TEACHER_LOGIT_PCNN_NUM_BLOCKS}\"", train)
        self.assertIn("--kernel-sizes \"${kernel_size_args[@]}\"", train)
        self.assertIn("--dilations \"${dilation_args[@]}\"", train)
        self.assertIn("--context-dim \"${TEACHER_LOGIT_PCNN_CONTEXT_DIM}\"", train)
        self.assertIn("--context-dims \"${context_dim_args[@]}\"", train)
        self.assertIn("--decoder-dims \"${decoder_dim_args[@]}\"", train)
        self.assertIn("--max-train-jets", train)
        self.assertIn('fresh_claim_new_dir "${OUTPUT_DIR}"', train)
        self.assertIn("#SBATCH --time=05:00:00", predict)
        self.assertIn("#SBATCH --gres=gpu:1", predict)
        self.assertIn("scripts/predict_teacher_logit_particle_cnn_reco.py", predict)
        self.assertIn("--prediction-dir \"${TEACHER_LOGIT_PCNN_PREDICTION_DIR}\"", predict)
        self.assertIn("--splits stack_train stack_val final_test", predict)
        self.assertIn("--confirm-final-test", predict)
        self.assertIn("scripts/run_independent_fusion_from_predictions.py", fusion)
        self.assertIn("--group \"teacher_logit_pcnn:${group_models}\"", fusion)
        self.assertIn("--confirm-final-test", fusion)
        self.assertIn("run_train_teacher_logit_pcnn_reco.sh", submitter)
        self.assertIn("run_predict_teacher_logit_pcnn_reco.sh", submitter)
        self.assertIn("run_fuse_teacher_logit_pcnn_reco.sh", submitter)
        self.assertIn('fresh_split_words teacher_args "${TEACHER_LOGIT_PCNN_TEACHERS}"', submitter)
        self.assertIn('fresh_refuse_existing_dir "${TEACHER_LOGIT_PCNN_PREDICTION_DIR}/${model_name}"', submitter)
        self.assertIn('--dependency="afterok:${train_jid}"', submitter)
        self.assertIn('fusion_dependency="$(fresh_join_by_colon "${predict_job_ids[@]}")', submitter)
        self.assertIn('--dependency="afterok:${fusion_dependency}"', submitter)

    def test_crossarch_step2_runners_build_fresh_500k_150k_cache_and_audit(self):
        common = self.read("common.sh")
        split = self.read("run_crossarch_build_splits.sh")
        cache = self.read("run_crossarch_build_hlt_cache.sh")
        audit = self.read("run_crossarch_audit_splits_hlt_cache.sh")
        self.assertIn("teacher_logit_reco_crossarch_500k", common)
        self.assertIn("CROSSARCH_MANIFEST_PATH", common)
        self.assertIn("CROSSARCH_HLT_CACHE_DIR", common)
        self.assertIn("CROSSARCH_STEP2_AUDIT_DIR", common)
        self.assertIn("CROSSARCH_MODEL_TRAIN_SIZE:=500000", common)
        self.assertIn("CROSSARCH_MODEL_VAL_SIZE:=150000", common)
        self.assertIn("CROSSARCH_STACK_TRAIN_SIZE:=500000", common)
        self.assertIn("CROSSARCH_STACK_VAL_SIZE:=150000", common)
        self.assertIn("CROSSARCH_FINAL_TEST_SIZE:=500000", common)
        self.assertIn("CROSSARCH_HLT_SPLITS:=model_train model_val stack_train stack_val final_test", common)

        self.assertIn("scripts/build_jetclass_splits.py", split)
        self.assertIn("--out \"${CROSSARCH_MANIFEST_PATH}\"", split)
        self.assertIn("--model-train \"${CROSSARCH_MODEL_TRAIN_SIZE}\"", split)
        self.assertIn("--model-val \"${CROSSARCH_MODEL_VAL_SIZE}\"", split)
        self.assertIn("--stack-train \"${CROSSARCH_STACK_TRAIN_SIZE}\"", split)
        self.assertIn("--stack-val \"${CROSSARCH_STACK_VAL_SIZE}\"", split)
        self.assertIn("--final-test \"${CROSSARCH_FINAL_TEST_SIZE}\"", split)
        self.assertIn('fresh_refuse_existing_path "${CROSSARCH_MANIFEST_PATH}"', split)

        self.assertIn("scripts/build_fixed_hlt_cache.py", cache)
        self.assertIn("--manifest \"${CROSSARCH_MANIFEST_PATH}\"", cache)
        self.assertIn("--cache-dir \"${CROSSARCH_HLT_CACHE_DIR}\"", cache)
        self.assertIn('fresh_split_words split_args "${CROSSARCH_HLT_SPLITS}"', cache)
        self.assertIn("${CROSSARCH_HLT_CACHE_DIR}/${split}_fixed_hlt.npz", cache)

        self.assertIn("#SBATCH --time=06:00:00", audit)
        self.assertIn("scripts/audit_crossarch_step2_splits_hlt_cache.py", audit)
        self.assertIn("--manifest \"${CROSSARCH_MANIFEST_PATH}\"", audit)
        self.assertIn("--hlt-cache-dir \"${CROSSARCH_HLT_CACHE_DIR}\"", audit)
        self.assertIn("--output-dir \"${CROSSARCH_STEP2_AUDIT_DIR}\"", audit)
        self.assertIn('fresh_claim_new_dir "${CROSSARCH_STEP2_AUDIT_DIR}"', audit)
        self.assertIn('fresh_assert_json_ok "${CROSSARCH_STEP2_AUDIT_DIR}/crossarch_step2_audit_report.json"', audit)

    def test_crossarch_step3_runners_train_or_register_four_offline_teachers(self):
        common = self.read("common.sh")
        runner = self.read("run_crossarch_train_offline_teacher.sh")
        submitter = self.read("submit_crossarch_step3_offline_teachers.sh")
        self.assertIn("CROSSARCH_OFFLINE_TEACHER_DIR", common)
        self.assertIn("CROSSARCH_OFFLINE_TEACHER_ARCHITECTURES:=part pn pfn pcnn", common)
        self.assertIn("CROSSARCH_OFFLINE_TEACHER_SEED:=707", common)
        self.assertIn("CROSSARCH_OFFLINE_TEACHER_MODEL_SIZE:=base", common)
        self.assertIn("CROSSARCH_PART_TEACHER_SOURCE_CHECKPOINT", common)
        self.assertIn("CROSSARCH_PCNN_TEACHER_SOURCE_REPORT", common)
        self.assertIn("fresh_crossarch_offline_teacher_source_checkpoint", common)
        self.assertIn("fresh_crossarch_offline_teacher_source_report", common)

        self.assertIn("#SBATCH --time=2-00:00:00", runner)
        self.assertIn("#SBATCH --gres=gpu:1", runner)
        self.assertIn("scripts/train_or_register_crossarch_offline_teacher.py", runner)
        self.assertIn("--architecture \"${ARCHITECTURE}\"", runner)
        self.assertIn("--manifest \"${CROSSARCH_MANIFEST_PATH}\"", runner)
        self.assertIn("--data-dir \"${DATA_DIR}\"", runner)
        self.assertIn("--output-dir \"${OUTPUT_DIR}\"", runner)
        self.assertIn("--max-train-jets \"${CROSSARCH_MODEL_TRAIN_SIZE}\"", runner)
        self.assertIn("--max-val-jets \"${CROSSARCH_MODEL_VAL_SIZE}\"", runner)
        self.assertIn("--model-size \"${CROSSARCH_OFFLINE_TEACHER_MODEL_SIZE}\"", runner)
        self.assertIn('fresh_append_optional_arg cmd --register-checkpoint "${source_checkpoint}"', runner)
        self.assertIn('fresh_append_optional_arg cmd --register-source-report "${source_report}"', runner)
        self.assertIn('fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"', runner)
        self.assertIn('fresh_require_file "${OUTPUT_DIR}/source_metadata.json"', runner)

        self.assertIn('fresh_split_words teacher_args "${CROSSARCH_OFFLINE_TEACHER_ARCHITECTURES}"', submitter)
        self.assertIn("run_crossarch_train_offline_teacher.sh", submitter)
        self.assertIn('submitter_lock_dir="${CROSSARCH_ROOT}/.step3_offline_teacher_submission_lock"', submitter)
        self.assertIn('fresh_claim_new_dir "${submitter_lock_dir}"', submitter)
        self.assertIn('--dependency="afterok:${dependency}"', submitter)
        self.assertIn("crossarch_step3_offline_teachers_submission", submitter)

    def test_crossarch_step4_runners_train_four_hlt_baselines_and_predictions(self):
        common = self.read("common.sh")
        train = self.read("run_crossarch_train_hlt_baseline.sh")
        predict = self.read("run_crossarch_predict_hlt_baseline.sh")
        submitter = self.read("submit_crossarch_step4_hlt_baselines.sh")
        self.assertIn("CROSSARCH_HLT_BASELINE_DIR", common)
        self.assertIn("CROSSARCH_HLT_BASELINE_ARCHITECTURES:=part pn pfn pcnn", common)
        self.assertIn("CROSSARCH_HLT_BASELINE_SEED:=101", common)
        self.assertIn("CROSSARCH_HLT_BASELINE_MODEL_SIZE:=base", common)
        self.assertIn("CROSSARCH_PREDICTION_DIR", common)
        self.assertIn("CROSSARCH_HLT_PREDICT_SPLITS:=stack_train stack_val final_test", common)
        self.assertIn("fresh_crossarch_hlt_model_name", common)

        self.assertIn("#SBATCH --time=12:00:00", train)
        self.assertIn("#SBATCH --gres=gpu:1", train)
        self.assertIn("scripts/train_crossarch_hlt_baseline.py", train)
        self.assertIn("--architecture \"${ARCHITECTURE}\"", train)
        self.assertIn("--cache-dir \"${CROSSARCH_HLT_CACHE_DIR}\"", train)
        self.assertIn("--output-dir \"${OUTPUT_DIR}\"", train)
        self.assertIn("--max-train-jets \"${CROSSARCH_MODEL_TRAIN_SIZE}\"", train)
        self.assertIn("--max-val-jets \"${CROSSARCH_MODEL_VAL_SIZE}\"", train)
        self.assertIn("--model-size \"${CROSSARCH_HLT_BASELINE_MODEL_SIZE}\"", train)
        self.assertIn('fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"', train)
        self.assertIn('fresh_require_file "${OUTPUT_DIR}/source_metadata.json"', train)

        self.assertIn("#SBATCH --time=05:00:00", predict)
        self.assertIn("#SBATCH --gres=gpu:1", predict)
        self.assertIn("scripts/predict_crossarch_hlt_baseline.py", predict)
        self.assertIn('MODEL_NAME="$(fresh_crossarch_hlt_model_name "${ARCHITECTURE}")"', predict)
        self.assertIn("--checkpoint \"${CHECKPOINT}\"", predict)
        self.assertIn("--prediction-dir \"${CROSSARCH_PREDICTION_DIR}\"", predict)
        self.assertIn("--output-dir \"${RUN_OUTPUT_DIR}\"", predict)
        self.assertIn("--splits \"${split_args[@]}\"", predict)
        self.assertIn("--stack-train-size \"${CROSSARCH_STACK_TRAIN_SIZE}\"", predict)
        self.assertIn("--stack-val-size \"${CROSSARCH_STACK_VAL_SIZE}\"", predict)
        self.assertIn("--final-test-size \"${CROSSARCH_FINAL_TEST_SIZE}\"", predict)
        self.assertIn("--confirm-final-test", predict)
        self.assertIn('fresh_require_file "${SOURCE_PREDICTION_DIR}/${split}_predictions.npz"', predict)

        self.assertIn('fresh_split_words arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"', submitter)
        self.assertIn("run_crossarch_train_hlt_baseline.sh", submitter)
        self.assertIn("run_crossarch_predict_hlt_baseline.sh", submitter)
        self.assertIn('submitter_lock_dir="${CROSSARCH_ROOT}/.step4_hlt_baseline_submission_lock"', submitter)
        self.assertIn('fresh_claim_new_dir "${submitter_lock_dir}"', submitter)
        self.assertIn('--dependency="afterok:${train_jid}"', submitter)
        self.assertIn("crossarch_step4_hlt_baselines_submission", submitter)

    def test_crossarch_step5_submitter_queues_sixteen_reconstructors(self):
        common = self.read("common.sh")
        runner = self.read("run_crossarch_train_reconstructor.sh")
        submitter = self.read("submit_crossarch_step5_reconstructors.sh")
        self.assertIn("CROSSARCH_RECO_MODEL_DIR", common)
        self.assertIn("CROSSARCH_RECO_ARCHITECTURES:=gt pn pfn pcnn", common)
        self.assertIn("CROSSARCH_RECO_TEACHERS:=part pn pfn pcnn", common)
        self.assertIn("CROSSARCH_RECO_MAX_TRAIN_JETS:=${CROSSARCH_MODEL_TRAIN_SIZE}", common)
        self.assertIn("CROSSARCH_RECO_MAX_VAL_JETS:=${CROSSARCH_MODEL_VAL_SIZE}", common)
        self.assertIn("fresh_crossarch_reco_model_name", common)
        self.assertIn("fresh_crossarch_reco_train_script", common)

        self.assertIn("#SBATCH --time=2-00:00:00", runner)
        self.assertIn("#SBATCH --gres=gpu:1", runner)
        self.assertIn('RECO_ARCHITECTURE="${1:?Usage:', runner)
        self.assertIn('TEACHER_ARCHITECTURE="${2:?Usage:', runner)
        self.assertIn('MODEL_NAME="$(fresh_crossarch_reco_model_name "${RECO_ARCHITECTURE}" "${TEACHER_ARCHITECTURE}")"', runner)
        self.assertIn('TRAIN_SCRIPT="$(fresh_crossarch_reco_train_script "${RECO_ARCHITECTURE}")"', runner)
        self.assertIn('OUTPUT_DIR="${CROSSARCH_RECO_MODEL_DIR}/${RECO_ARCHITECTURE}/${TEACHER_ARCHITECTURE}"', runner)
        self.assertIn('TEACHER_CHECKPOINT="${CROSSARCH_OFFLINE_TEACHER_DIR}/${TEACHER_ARCHITECTURE}/best_model_val.pt"', runner)
        self.assertIn("--manifest-path \"${CROSSARCH_MANIFEST_PATH}\"", runner)
        self.assertIn("--hlt-cache-dir \"${CROSSARCH_HLT_CACHE_DIR}\"", runner)
        self.assertIn("--teacher-checkpoint \"${TEACHER_CHECKPOINT}\"", runner)
        self.assertIn("--teacher-architecture \"${TEACHER_ARCHITECTURE}\"", runner)
        self.assertIn("--max-train-jets \"${CROSSARCH_RECO_MAX_TRAIN_JETS}\"", runner)
        self.assertIn("--max-val-jets \"${CROSSARCH_RECO_MAX_VAL_JETS}\"", runner)
        self.assertIn("--batch-size \"${CROSSARCH_RECO_BATCH_SIZE}\"", runner)
        self.assertIn("--epochs \"${CROSSARCH_RECO_EPOCHS}\"", runner)
        self.assertIn("--edgeconv-dims \"${edgeconv_dim_args[@]}\"", runner)
        self.assertIn("--phi-dims \"${phi_dim_args[@]}\"", runner)
        self.assertIn("--kernel-sizes \"${kernel_size_args[@]}\"", runner)
        self.assertIn('fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"', runner)
        self.assertIn('fresh_require_file "${OUTPUT_DIR}/training_curves.json"', runner)
        self.assertIn('fresh_require_file "${OUTPUT_DIR}/run_report.json"', runner)

        self.assertIn('fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"', submitter)
        self.assertIn('fresh_split_words teacher_args "${CROSSARCH_RECO_TEACHERS}"', submitter)
        self.assertIn("run_crossarch_train_reconstructor.sh", submitter)
        self.assertIn('submitter_lock_dir="${CROSSARCH_ROOT}/.step5_reconstructor_submission_lock"', submitter)
        self.assertIn('fresh_claim_new_dir "${submitter_lock_dir}"', submitter)
        self.assertIn('for reco_architecture in "${reco_args[@]}"; do', submitter)
        self.assertIn('for teacher_architecture in "${teacher_args[@]}"; do', submitter)
        self.assertIn('fresh_refuse_existing_dir "${output_dir}"', submitter)
        self.assertIn('--dependency="afterok:${dependency}"', submitter)
        self.assertIn("crossarch_step5_reconstructors_submission", submitter)
        self.assertIn("expected_models: 16", submitter)

    def test_crossarch_step6_submitter_queues_prediction_blocks_for_all_sources(self):
        common = self.read("common.sh")
        runner = self.read("run_crossarch_predict_reconstructor.sh")
        submitter = self.read("submit_crossarch_step6_predictions.sh")
        self.assertIn("CROSSARCH_RECO_PREDICTION_RUN_DIR", common)
        self.assertIn("CROSSARCH_RECO_PREDICT_SPLITS:=stack_train stack_val final_test", common)
        self.assertIn("CROSSARCH_RECO_PREDICT_BATCH_SIZE:=128", common)
        self.assertIn("CROSSARCH_RECO_PREDICT_DEVICE:=${DEVICE}", common)
        self.assertIn("CROSSARCH_STEP6_SUBMIT_HLT_PREDICTIONS:=1", common)
        self.assertIn("CROSSARCH_STEP6_SKIP_EXISTING_PREDICTIONS:=0", common)
        self.assertIn("fresh_crossarch_reco_predict_script", common)

        self.assertIn("#SBATCH --time=05:00:00", runner)
        self.assertIn("#SBATCH --gres=gpu:1", runner)
        self.assertIn('RECO_ARCHITECTURE="${1:?Usage:', runner)
        self.assertIn('TEACHER_ARCHITECTURE="${2:?Usage:', runner)
        self.assertIn('PREDICT_SCRIPT="$(fresh_crossarch_reco_predict_script "${RECO_ARCHITECTURE}")"', runner)
        self.assertIn('MODEL_NAME="$(fresh_crossarch_reco_model_name "${RECO_ARCHITECTURE}" "${TEACHER_ARCHITECTURE}")"', runner)
        self.assertIn('RECONSTRUCTOR_CHECKPOINT="${CROSSARCH_RECO_MODEL_DIR}/${RECO_ARCHITECTURE}/${TEACHER_ARCHITECTURE}/best_model_val.pt"', runner)
        self.assertIn('TEACHER_CHECKPOINT="${CROSSARCH_OFFLINE_TEACHER_DIR}/${TEACHER_ARCHITECTURE}/best_model_val.pt"', runner)
        self.assertIn('RUN_OUTPUT_DIR="${CROSSARCH_RECO_PREDICTION_RUN_DIR}/${MODEL_NAME}"', runner)
        self.assertIn('SOURCE_PREDICTION_DIR="${CROSSARCH_PREDICTION_DIR}/${MODEL_NAME}"', runner)
        self.assertIn('fresh_split_words split_args "${CROSSARCH_RECO_PREDICT_SPLITS}"', runner)
        self.assertIn("--prediction-dir \"${CROSSARCH_PREDICTION_DIR}\"", runner)
        self.assertIn("--hlt-cache-dir \"${CROSSARCH_HLT_CACHE_DIR}\"", runner)
        self.assertIn("--reconstructor-checkpoint \"${RECONSTRUCTOR_CHECKPOINT}\"", runner)
        self.assertIn("--teacher-checkpoint \"${TEACHER_CHECKPOINT}\"", runner)
        self.assertIn("--model-name \"${MODEL_NAME}\"", runner)
        self.assertIn("--splits \"${split_args[@]}\"", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn('fresh_require_file "${SOURCE_PREDICTION_DIR}/${split}_predictions.npz"', runner)
        self.assertIn('fresh_require_file "${SOURCE_PREDICTION_DIR}/${split}_predictions_metadata.json"', runner)

        self.assertIn('fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"', submitter)
        self.assertIn('fresh_split_words teacher_args "${CROSSARCH_RECO_TEACHERS}"', submitter)
        self.assertIn('fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"', submitter)
        self.assertIn("run_crossarch_predict_hlt_baseline.sh", submitter)
        self.assertIn("run_crossarch_predict_reconstructor.sh", submitter)
        self.assertIn('submitter_lock_dir="${CROSSARCH_ROOT}/.step6_prediction_submission_lock"', submitter)
        self.assertIn('fresh_claim_new_dir "${submitter_lock_dir}"', submitter)
        self.assertIn('if fresh_bool_enabled "${CROSSARCH_STEP6_SUBMIT_HLT_PREDICTIONS}"; then', submitter)
        self.assertIn('for reco_architecture in "${reco_args[@]}"; do', submitter)
        self.assertIn('for teacher_architecture in "${teacher_args[@]}"; do', submitter)
        self.assertIn('maybe_skip_existing_prediction "${source_dir}" "${model_name}"', submitter)
        self.assertIn('--dependency="afterok:${dependency}"', submitter)
        self.assertIn("crossarch_step6_predictions_submission", submitter)
        self.assertIn("total: 20", submitter)

    def test_crossarch_step10_full_submitter_queues_full_graph(self):
        common = self.read("common.sh")
        fusion = self.read("run_crossarch_fusion.sh")
        final_runner = self.read("run_crossarch_write_final_report.sh")
        submitter = self.read("submit_crossarch_full_experiment.sh")

        self.assertIn("CROSSARCH_FUSION_DIR", common)
        self.assertIn("CROSSARCH_FINAL_REPORT_DIR", common)
        self.assertIn("CROSSARCH_FUSION_INCLUDE_OPTIONAL_GROUPS:=1", common)
        self.assertIn("CROSSARCH_FUSERS:=mean_logits mean_probs", common)

        self.assertIn("#SBATCH --time=1-00:00:00", fusion)
        self.assertIn("scripts/run_crossarch_fusion.py", fusion)
        self.assertIn("--fit-fusers", fusion)
        self.assertIn("--include-optional-groups", fusion)
        self.assertIn("--confirm-final-test", fusion)
        self.assertIn("--fusers \"${fuser_args[@]}\"", fusion)
        self.assertIn("--control-feature-modes \"${control_feature_mode_args[@]}\"", fusion)
        self.assertIn('fresh_assert_json_ok "${CROSSARCH_FUSION_DIR}/fusion_report.json"', fusion)

        self.assertIn("#SBATCH --time=02:00:00", final_runner)
        self.assertIn("scripts/write_crossarch_final_report.py", final_runner)
        self.assertIn("--fusion-report \"${CROSSARCH_FUSION_DIR}/fusion_report.json\"", final_runner)
        self.assertIn("--output-dir \"${CROSSARCH_FINAL_REPORT_DIR}\"", final_runner)
        self.assertIn('fresh_assert_json_ok "${CROSSARCH_FINAL_REPORT_DIR}/crossarch_final_report.json"', final_runner)

        self.assertIn("run_crossarch_build_splits.sh", submitter)
        self.assertIn("run_crossarch_build_hlt_cache.sh", submitter)
        self.assertIn("run_crossarch_audit_splits_hlt_cache.sh", submitter)
        self.assertIn("run_crossarch_train_offline_teacher.sh", submitter)
        self.assertIn("run_crossarch_train_hlt_baseline.sh", submitter)
        self.assertIn("run_crossarch_predict_hlt_baseline.sh", submitter)
        self.assertIn("run_crossarch_train_reconstructor.sh", submitter)
        self.assertIn("run_crossarch_predict_reconstructor.sh", submitter)
        self.assertIn("run_crossarch_fusion.sh", submitter)
        self.assertIn("run_crossarch_write_final_report.sh", submitter)
        self.assertIn('--dependency="afterok:${split_jid}"', submitter)
        self.assertIn('--dependency="afterok:${cache_jid}"', submitter)
        self.assertIn('--dependency="afterok:${audit_jid}"', submitter)
        self.assertIn('--dependency="afterok:${teacher_dep}"', submitter)
        self.assertIn("declare -A reco_train_job_id_by_model", submitter)
        self.assertIn('reco_train_job_id_by_model["${model_name}"]="${reco_train_jid}"', submitter)
        self.assertIn('model_train_jid="${reco_train_job_id_by_model[${model_name}]}"', submitter)
        self.assertIn('--dependency="afterok:${model_train_jid}"', submitter)
        self.assertIn("each_reco_predict_after_its_train: true", submitter)
        self.assertIn('--dependency="afterok:${prediction_dep}"', submitter)
        self.assertIn('--dependency="afterok:${fusion_jid}"', submitter)
        self.assertIn("crossarch_full_experiment_submission", submitter)
        self.assertIn("offline_teachers: 4", submitter)
        self.assertIn("hlt_train: 4", submitter)
        self.assertIn("hlt_predict: 4", submitter)
        self.assertIn("reco_train: 16", submitter)
        self.assertIn("reco_predict: 16", submitter)
        self.assertIn("total: 20", submitter)

    def test_crossarch_split_fusion_submitter_queues_forty_two_small_fusions(self):
        common = self.read("common.sh")
        runner = self.read("run_crossarch_split_fusion.sh")
        summary_runner = self.read("run_crossarch_split_fusion_summary.sh")
        submitter = self.read("submit_crossarch_split_fusions.sh")

        self.assertIn("CROSSARCH_SPLIT_FUSION_ROOT", common)
        self.assertIn("CROSSARCH_SPLIT_FUSION_FAMILIES:=frozen adapted", common)
        self.assertIn("CROSSARCH_SPLIT_FUSION_GROUPS:=hlt4 all16 all16_plus_hlt4 cross12_plus_hlt4 part_teacher4_plus_hlt4 pn_teacher4_plus_hlt4 mixed4_plus_hlt4", common)
        self.assertIn("CROSSARCH_SPLIT_FUSION_BUNDLES:=main gated controls", common)
        self.assertIn("CROSSARCH_SPLIT_FUSION_MAIN_FUSERS:=mean_logits mean_probs logistic_logits logistic_probs logistic_logits_probs uncertainty_logistic_logits_probs", common)
        self.assertIn("CROSSARCH_SPLIT_FUSION_GATED_FUSERS:=entropy_bin_gated_logistic margin_bin_gated_logistic multiplicity_bin_gated_logistic disagreement_bin_gated_logistic predicted_class_bin_gated_logistic", common)
        self.assertIn("CROSSARCH_SPLIT_FUSION_CONTROL_FUSERS:=logistic_logits_probs", common)

        self.assertIn("#SBATCH --time=12:00:00", runner)
        self.assertIn("scripts/run_crossarch_fusion.py", runner)
        self.assertIn('FAMILY="${1:?Usage:', runner)
        self.assertIn('GROUP="${2:?Usage:', runner)
        self.assertIn('BUNDLE="${3:?Usage:', runner)
        self.assertIn('fresh_crossarch_reco_model_name "${reco_architecture}" "${teacher_architecture}"', runner)
        self.assertIn('fresh_crossarch_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}"', runner)
        self.assertIn("part_teacher4_plus_hlt4", runner)
        self.assertIn("pn_teacher4_plus_hlt4", runner)
        self.assertIn("mixed4_plus_hlt4", runner)
        self.assertIn('OUTPUT_DIR="${CROSSARCH_SPLIT_FUSION_ROOT}/${FAMILY}/${GROUP}/${BUNDLE}"', runner)
        self.assertIn("--group \"${FAMILY}_${GROUP}:$(fresh_join_by_comma \"${group_models[@]}\")\"", runner)
        self.assertIn('if [[ "${skip_controls}" == "1" ]]; then', runner)
        self.assertIn('fresh_require_file "${OUTPUT_DIR}/fusion_report.json"', runner)

        self.assertIn("#SBATCH --partition=debug", summary_runner)
        self.assertIn("scripts/summarize_crossarch_split_fusions.py", summary_runner)
        self.assertIn("--split-root \"${CROSSARCH_SPLIT_FUSION_ROOT}\"", summary_runner)
        self.assertIn("--families \"${family_args[@]}\"", summary_runner)
        self.assertIn("--groups \"${group_args[@]}\"", summary_runner)
        self.assertIn("--bundles \"${bundle_args[@]}\"", summary_runner)

        self.assertIn("run_crossarch_split_fusion.sh", submitter)
        self.assertIn("run_crossarch_split_fusion_summary.sh", submitter)
        self.assertIn('fresh_split_words family_args "${CROSSARCH_SPLIT_FUSION_FAMILIES}"', submitter)
        self.assertIn('fresh_split_words group_args "${CROSSARCH_SPLIT_FUSION_GROUPS}"', submitter)
        self.assertIn('fresh_split_words bundle_args "${CROSSARCH_SPLIT_FUSION_BUNDLES}"', submitter)
        self.assertIn('for family in "${family_args[@]}"; do', submitter)
        self.assertIn('for group in "${group_args[@]}"; do', submitter)
        self.assertIn('for bundle in "${bundle_args[@]}"; do', submitter)
        self.assertIn('--dependency="afterok:${CROSSARCH_SPLIT_FUSION_DEPENDENCY}"', submitter)
        self.assertIn('fusion_dep="$(fresh_join_by_colon "${fusion_job_ids[@]}")"', submitter)
        self.assertIn('--dependency="afterok:${fusion_dep}"', submitter)
        self.assertIn("split_fusions: ${#fusion_job_ids[@]}", submitter)

    def test_crossarch_conditional_fuser_runners_are_debug_jobs(self):
        common = self.read("common.sh")
        linear = self.read("run_crossarch_conditional_fusers_linear.sh")
        neural = self.read("run_crossarch_conditional_fusers_neural.sh")

        self.assertIn("CROSSARCH_CONDITIONAL_FUSER_DIR", common)
        self.assertIn("CROSSARCH_CONDITIONAL_FUSER_RESIDUAL_PENALTIES", common)
        self.assertIn("CROSSARCH_CONDITIONAL_FUSER_NEURAL_EPOCHS", common)
        for name, text, suite in [
            ("run_crossarch_conditional_fusers_linear.sh", linear, "linear"),
            ("run_crossarch_conditional_fusers_neural.sh", neural, "neural"),
        ]:
            self.assertIn("#SBATCH --partition=debug", text, name)
            self.assertIn("scripts/run_crossarch_conditional_evidence_fusers.py", text, name)
            self.assertIn(f"--suite {suite}", text, name)
            self.assertIn('fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"', text, name)
            self.assertIn('fresh_crossarch_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}"', text, name)
            self.assertIn("--hlt-models \"${hlt_model_args[@]}\"", text, name)
            self.assertIn("--adapted-models \"${adapted_model_args[@]}\"", text, name)
            self.assertIn("--confirm-final-test", text, name)
            self.assertIn('fresh_require_file "${OUTPUT_DIR}/conditional_fuser_report.json"', text, name)
        self.assertIn("--skip-controls", neural)

    def test_crossarch_aggressive_step12_submitter_queues_full_aggressive_graph(self):
        common = self.read("common.sh")
        train = self.read("run_crossarch_aggressive_train_reconstructor.sh")
        predict = self.read("run_crossarch_aggressive_predict_reconstructor.sh")
        adapt_train = self.read("run_crossarch_aggressive_train_reco_domain_tagger.sh")
        adapt_predict = self.read("run_crossarch_aggressive_predict_reco_domain_tagger.sh")
        fusion = self.read("run_crossarch_aggressive_fusion.sh")
        audit = self.read("run_crossarch_aggressive_audit.sh")
        submitter = self.read("submit_crossarch_aggressive_experiment.sh")

        self.assertIn("CROSSARCH_AGGRESSIVE_ROOT:=${OUTPUT_ROOT}/teacher_logit_reco_crossarch_aggressive_v1_500k", common)
        self.assertIn("CROSSARCH_AGGRESSIVE_RECO_ARCHITECTURES:=aggt agpn agpfn agpcnn", common)
        self.assertIn("CROSSARCH_AGGRESSIVE_RECO_TEACHERS:=part pn pfn pcnn", common)
        self.assertIn("CROSSARCH_AGGRESSIVE_RECO_NUM_EXTRA_CANDIDATES:=64", common)
        self.assertIn("CROSSARCH_AGGRESSIVE_AUDIT_DIR", common)
        self.assertIn("CROSSARCH_AGGRESSIVE_AUDIT_GROUPS", common)
        self.assertIn("fresh_crossarch_aggressive_reco_model_name", common)
        self.assertIn("fresh_crossarch_aggressive_reco_domain_tagger_model_name", common)
        self.assertIn('echo "scripts/train_teacher_logit_aggressive_reco.py"', common)
        self.assertIn('echo "scripts/predict_teacher_logit_aggressive_reco.py"', common)

        self.assertIn("#SBATCH --time=2-00:00:00", train)
        self.assertIn("#SBATCH --gres=gpu:1", train)
        self.assertIn('TRAIN_SCRIPT="$(fresh_crossarch_aggressive_reco_train_script "${RECO_ARCHITECTURE}")"', train)
        self.assertIn("--num-extra-candidates \"${CROSSARCH_AGGRESSIVE_RECO_NUM_EXTRA_CANDIDATES}\"", train)
        self.assertIn("--max-global-logpt-scale \"${CROSSARCH_AGGRESSIVE_RECO_MAX_GLOBAL_LOGPT_SCALE}\"", train)
        self.assertIn('OUTPUT_DIR="${CROSSARCH_AGGRESSIVE_RECO_MODEL_DIR}/${RECO_ARCHITECTURE}/${TEACHER_ARCHITECTURE}"', train)

        self.assertIn("#SBATCH --time=08:00:00", predict)
        self.assertIn("#SBATCH --gres=gpu:1", predict)
        self.assertIn('PREDICT_SCRIPT="$(fresh_crossarch_aggressive_reco_predict_script "${RECO_ARCHITECTURE}")"', predict)
        self.assertIn('SOURCE_PREDICTION_DIR="${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}/${MODEL_NAME}"', predict)
        self.assertIn("--prediction-dir \"${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}\"", predict)

        self.assertIn("#SBATCH --time=2-00:00:00", adapt_train)
        self.assertIn("#SBATCH --gres=gpu:1", adapt_train)
        self.assertIn("scripts/train_crossarch_reco_domain_tagger.py", adapt_train)
        self.assertIn('RECONSTRUCTOR_CHECKPOINT="${CROSSARCH_AGGRESSIVE_RECO_MODEL_DIR}/${RECO_ARCHITECTURE}/${TEACHER_ARCHITECTURE}/best_model_val.pt"', adapt_train)

        self.assertIn("#SBATCH --time=08:00:00", adapt_predict)
        self.assertIn("#SBATCH --gres=gpu:1", adapt_predict)
        self.assertIn("scripts/predict_crossarch_reco_domain_tagger.py", adapt_predict)
        self.assertIn('TAGGER_CHECKPOINT="${CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_DIR}/${RECO_ARCHITECTURE}/${TEACHER_ARCHITECTURE}/best_model_val.pt"', adapt_predict)

        self.assertIn("#SBATCH --time=1-00:00:00", fusion)
        self.assertNotIn("#SBATCH --gres=gpu:1", fusion)
        self.assertIn("--include-optional-groups", fusion)
        self.assertIn("--groups \"${fusion_group_args[@]}\"", fusion)
        self.assertIn("--prediction-dir \"${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}\"", fusion)
        self.assertIn("--output-dir \"${CROSSARCH_AGGRESSIVE_FUSION_DIR}\"", fusion)

        self.assertIn("#SBATCH --time=02:00:00", audit)
        self.assertIn("#SBATCH --partition=debug", audit)
        self.assertIn("scripts/audit_crossarch_aggressive_experiment.py", audit)
        self.assertIn("--prediction-dir \"${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}\"", audit)
        self.assertIn("--reco-model-dir \"${CROSSARCH_AGGRESSIVE_RECO_MODEL_DIR}\"", audit)
        self.assertIn("--adapted-tagger-dir \"${CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_DIR}\"", audit)
        self.assertIn("--fusion-report \"${CROSSARCH_AGGRESSIVE_FUSION_DIR}/fusion_report.json\"", audit)
        self.assertIn("--fusion-groups \"${fusion_group_args[@]}\"", audit)

        self.assertIn('fresh_split_words reco_args "${CROSSARCH_AGGRESSIVE_RECO_ARCHITECTURES}"', submitter)
        self.assertIn('fresh_split_words teacher_args "${CROSSARCH_AGGRESSIVE_RECO_TEACHERS}"', submitter)
        self.assertIn("run_crossarch_aggressive_train_reconstructor.sh", submitter)
        self.assertIn("run_crossarch_aggressive_predict_reconstructor.sh", submitter)
        self.assertIn("run_crossarch_aggressive_train_reco_domain_tagger.sh", submitter)
        self.assertIn("run_crossarch_aggressive_predict_reco_domain_tagger.sh", submitter)
        self.assertIn("run_crossarch_aggressive_fusion.sh", submitter)
        self.assertIn("run_crossarch_aggressive_audit.sh", submitter)
        self.assertIn('--dependency="afterok:${train_jid}"', submitter)
        self.assertIn('--dependency="afterok:${adapt_train_jid}"', submitter)
        self.assertIn('--dependency="afterok:${fusion_dep}"', submitter)
        self.assertIn('--dependency="afterok:${fusion_jid}"', submitter)
        self.assertIn("crossarch_aggressive_experiment_submission", submitter)
        self.assertIn("aggressive_reco_train: 16", submitter)
        self.assertIn("aggressive_frozen_teacher_predict: 16", submitter)
        self.assertIn("aggressive_adapted_tagger_train: 16", submitter)
        self.assertIn("aggressive_adapted_tagger_predict: 16", submitter)
        self.assertIn("audit: 1", submitter)

    def test_crossarch_aggressive_step14_smoke_submitter_uses_tiny_isolated_outputs(self):
        smoke = self.read("submit_crossarch_aggressive_smoke_test.sh")
        common = self.read("common.sh")
        aggressive_submitter = self.read("submit_crossarch_aggressive_experiment.sh")
        audit = self.read("run_crossarch_aggressive_audit.sh")

        self.assertIn("CROSSARCH_AGGRESSIVE_REQUIRE_HLT_PREDICTIONS_AT_SUBMIT", common)
        self.assertIn("CROSSARCH_AGGRESSIVE_AUDIT_CHECK_PREDICTION_ARRAYS", common)
        self.assertIn("CROSSARCH_AGGRESSIVE_REQUIRE_HLT_PREDICTIONS_AT_SUBMIT", aggressive_submitter)
        self.assertIn("Skipping submit-time HLT prediction preflight", aggressive_submitter)
        self.assertIn("--check-prediction-arrays", audit)

        self.assertIn("teacher_logit_reco_crossarch_aggressive_v1_smoke_", smoke)
        self.assertIn('CROSSARCH_MODEL_TRAIN_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_MODEL_TRAIN_SIZE:-10000}"', smoke)
        self.assertIn('CROSSARCH_MODEL_VAL_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_MODEL_VAL_SIZE:-2000}"', smoke)
        self.assertIn('CROSSARCH_STACK_TRAIN_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_STACK_TRAIN_SIZE:-5000}"', smoke)
        self.assertIn('CROSSARCH_STACK_VAL_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_STACK_VAL_SIZE:-2000}"', smoke)
        self.assertIn('CROSSARCH_FINAL_TEST_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_FINAL_TEST_SIZE:-10000}"', smoke)
        self.assertIn('CROSSARCH_AGGRESSIVE_RECO_EPOCHS="${CROSSARCH_AGGRESSIVE_SMOKE_RECO_EPOCHS:-2}"', smoke)
        self.assertIn('CROSSARCH_AGGRESSIVE_RECO_MAX_TRAIN_JETS="${CROSSARCH_MODEL_TRAIN_SIZE}"', smoke)
        self.assertIn('CROSSARCH_AGGRESSIVE_RECO_MAX_VAL_JETS="${CROSSARCH_MODEL_VAL_SIZE}"', smoke)
        self.assertIn('CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_EPOCHS="${CROSSARCH_AGGRESSIVE_SMOKE_ADAPTED_EPOCHS:-2}"', smoke)
        self.assertIn('CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_MAX_TRAIN_JETS="${CROSSARCH_MODEL_TRAIN_SIZE}"', smoke)
        self.assertIn('CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_MAX_VAL_JETS="${CROSSARCH_MODEL_VAL_SIZE}"', smoke)
        self.assertIn("CROSSARCH_AGGRESSIVE_AUDIT_REQUIRE_OK=1", smoke)
        self.assertIn("CROSSARCH_AGGRESSIVE_AUDIT_CHECK_PREDICTION_ARRAYS=1", smoke)
        self.assertIn("CROSSARCH_AGGRESSIVE_REQUIRE_HLT_PREDICTIONS_AT_SUBMIT=0", smoke)
        self.assertIn("run_crossarch_predict_hlt_baseline.sh", smoke)
        self.assertIn("submit_crossarch_aggressive_experiment.sh", smoke)
        self.assertIn('export FUSION_UPSTREAM_DEPENDENCY="$(fresh_join_by_colon "${fusion_dependencies[@]}")"', smoke)
        self.assertIn("smoke metrics are for pipeline correctness only", smoke)

    def test_set_matching_step12_submitter_queues_multiview_graph(self):
        common = self.read("common.sh")
        train = self.read("run_train_set_matching_reconstructor.sh")
        cache = self.read("run_cache_set_matching_multiview.sh")
        tagger = self.read("run_train_five_view_tagger.sh")
        audit = self.read("run_audit_five_view_tagger.sh")
        final_report = self.read("run_write_set_matching_multiview_final_report.sh")
        submitter = self.read("submit_set_matching_multiview_experiment.sh")

        self.assertIn("SET_MATCHING_ROOT:=${OUTPUT_ROOT}/set_matching_multiview_500k", common)
        self.assertIn("SET_MATCHING_RECO_ARCHITECTURES:=gt pn pfn pcnn", common)
        self.assertIn("SET_MATCHING_TAGGER_VARIANTS:=hlt_only hlt_plus_gt hlt_plus_pn hlt_plus_pfn hlt_plus_pcnn five_view_plain five_view_geometry five_view_no_confidence view_label_shuffle_control", common)
        self.assertIn("SET_MATCHING_MODEL_TRAIN_SIZE:=500000", common)
        self.assertIn("SET_MATCHING_FINAL_TEST_SIZE:=500000", common)

        self.assertIn("#SBATCH --time=2-00:00:00", train)
        self.assertIn("#SBATCH --gres=gpu:1", train)
        self.assertIn("scripts/train_set_matching_reconstructor.py", train)
        self.assertIn('--architecture "${ARCHITECTURE}"', train)
        self.assertIn("--confirm-split-settings", train)
        self.assertIn("--max-train-jets \"${SET_MATCHING_MODEL_TRAIN_SIZE}\"", train)
        self.assertIn("--max-val-jets \"${SET_MATCHING_MODEL_VAL_SIZE}\"", train)

        self.assertIn("#SBATCH --time=12:00:00", cache)
        self.assertIn("#SBATCH --gres=gpu:1", cache)
        self.assertIn("scripts/cache_set_matching_reco_views.py", cache)
        self.assertIn('--output-dir "${SET_MATCHING_ROOT}"', cache)
        self.assertIn('--splits "${split_args[@]}"', cache)
        self.assertIn('fresh_append_flag_if_enabled cmd --confirm-final-test "${SET_MATCHING_CONFIRM_FINAL_TEST}"', cache)
        self.assertIn('fresh_require_file "${RUN_OUTPUT_DIR}/${split}_reconstructed_view.npz"', cache)

        self.assertIn("#SBATCH --time=2-00:00:00", tagger)
        self.assertIn("#SBATCH --gres=gpu:1", tagger)
        self.assertIn("scripts/train_five_view_tagger.py", tagger)
        self.assertIn('VARIANT="${1:?Usage:', tagger)
        self.assertIn("hlt_plus_pn", tagger)
        self.assertIn('cmd+=(--drop-views "${drop_views[@]}")', tagger)
        self.assertIn("five_view_geometry", tagger)
        self.assertIn("view_label_shuffle_control", tagger)
        self.assertIn('fresh_append_flag_if_enabled cmd --use-geometry-attention "${use_geometry_attention}"', tagger)
        self.assertIn('fresh_append_flag_if_enabled cmd --disable-confidence "${disable_confidence}"', tagger)
        self.assertIn('fresh_append_flag_if_enabled cmd --shuffle-view-labels "${shuffle_view_labels}"', tagger)

        self.assertIn("#SBATCH --time=08:00:00", audit)
        self.assertIn("#SBATCH --gres=gpu:1", audit)
        self.assertIn("scripts/evaluate_five_view_ablation.py", audit)
        self.assertIn("--require-all-canonical", audit)
        self.assertIn('fresh_require_file "${SET_MATCHING_ABLATION_DIR}/summary.csv"', audit)

        self.assertIn("#SBATCH --partition=debug", final_report)
        self.assertIn("#SBATCH --time=01:00:00", final_report)
        self.assertIn("scripts/write_set_matching_multiview_final_report.py", final_report)
        self.assertIn("--experiment-dir \"${SET_MATCHING_ROOT}\"", final_report)
        self.assertIn('fresh_require_file "${SET_MATCHING_FINAL_REPORT_DIR}/final_report.json"', final_report)

        self.assertIn("run_train_set_matching_reconstructor.sh", submitter)
        self.assertIn("run_cache_set_matching_multiview.sh", submitter)
        self.assertIn("run_train_five_view_tagger.sh", submitter)
        self.assertIn("run_audit_five_view_tagger.sh", submitter)
        self.assertIn("run_write_set_matching_multiview_final_report.sh", submitter)
        self.assertIn('submitter_lock_dir="${SET_MATCHING_ROOT}/.submission_lock"', submitter)
        self.assertIn('fresh_claim_new_dir "${submitter_lock_dir}"', submitter)
        self.assertIn('fresh_split_words reco_args "${SET_MATCHING_RECO_ARCHITECTURES}"', submitter)
        self.assertIn('fresh_split_words tagger_variant_args "${SET_MATCHING_TAGGER_VARIANTS}"', submitter)
        self.assertIn('cache_dep="$(fresh_join_by_colon "${cache_job_ids[@]}")', submitter)
        self.assertIn('--dependency="afterok:${cache_dep}"', submitter)
        self.assertIn('audit_dep="$(fresh_join_by_colon "${tagger_job_ids[@]}")', submitter)
        self.assertIn('--dependency="afterok:${audit_dep}"', submitter)
        self.assertIn('--dependency="afterok:${audit_jid}"', submitter)
        self.assertIn("reco_train: 4", submitter)
        self.assertIn("cache_reconstructed_views: 4", submitter)
        self.assertIn("tagger_train: ${#tagger_job_ids[@]}", submitter)
        self.assertIn("audit: 1", submitter)
        self.assertIn("final_report: 1", submitter)

    def test_set_matching_step13_smoke_submitter_uses_tiny_isolated_outputs(self):
        smoke = self.read("submit_set_matching_multiview_smoke_test.sh")

        self.assertIn("set_matching_multiview_smoke_", smoke)
        self.assertIn('SET_MATCHING_MODEL_TRAIN_SIZE="${SET_MATCHING_SMOKE_MODEL_TRAIN_SIZE:-10000}"', smoke)
        self.assertIn('SET_MATCHING_MODEL_VAL_SIZE="${SET_MATCHING_SMOKE_MODEL_VAL_SIZE:-2000}"', smoke)
        self.assertIn('SET_MATCHING_STACK_TRAIN_SIZE="${SET_MATCHING_SMOKE_STACK_TRAIN_SIZE:-5000}"', smoke)
        self.assertIn('SET_MATCHING_STACK_VAL_SIZE="${SET_MATCHING_SMOKE_STACK_VAL_SIZE:-2000}"', smoke)
        self.assertIn('SET_MATCHING_FINAL_TEST_SIZE="${SET_MATCHING_SMOKE_FINAL_TEST_SIZE:-10000}"', smoke)
        self.assertIn('SET_MATCHING_RECO_EPOCHS="${SET_MATCHING_SMOKE_RECO_EPOCHS:-2}"', smoke)
        self.assertIn('SET_MATCHING_RECO_EARLY_STOP_PATIENCE="${SET_MATCHING_SMOKE_RECO_EARLY_STOP_PATIENCE:-1}"', smoke)
        self.assertIn('SET_MATCHING_TAGGER_EPOCHS="${SET_MATCHING_SMOKE_TAGGER_EPOCHS:-2}"', smoke)
        self.assertIn('SET_MATCHING_TAGGER_EARLY_STOP_PATIENCE="${SET_MATCHING_SMOKE_TAGGER_EARLY_STOP_PATIENCE:-1}"', smoke)
        self.assertIn('SET_MATCHING_CACHE_MAX_JETS_PER_SPLIT="${SET_MATCHING_SMOKE_CACHE_MAX_JETS_PER_SPLIT:-${SET_MATCHING_FINAL_TEST_SIZE}}"', smoke)
        self.assertIn("SET_MATCHING_CONFIRM_FINAL_TEST=1", smoke)
        self.assertIn("SET_MATCHING_EVAL_REQUIRE_ALL_CANONICAL=1", smoke)
        self.assertIn('fresh_refuse_existing_dir "${SET_MATCHING_SMOKE_ROOT}"', smoke)
        self.assertIn('fresh_claim_new_dir "${SET_MATCHING_SMOKE_ROOT}/.smoke_submission_lock"', smoke)
        self.assertIn("smoke metrics are for pipeline correctness only", smoke)
        self.assertIn('bash "${SCRIPT_DIR}/submit_set_matching_multiview_experiment.sh"', smoke)


if __name__ == "__main__":
    unittest.main()

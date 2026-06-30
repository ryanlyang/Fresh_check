from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"


RUNNERS = [
    "run_build_fresh_splits.sh",
    "run_build_fresh_hlt_cache.sh",
    "run_build_label_filtered_split_manifest.sh",
    "run_build_label_filtered_fresh_splits.sh",
    "run_build_label_filtered_hlt_cache.sh",
    "run_train_fresh_hlt_baseline.sh",
    "run_train_fresh_hlt_seed.sh",
    "run_train_fresh_offline_teacher.sh",
    "run_train_eval_set_matching_binary_offline_teacher.sh",
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
    "run_v2_teacher_logit_dualview_debug.sh",
    "run_v2_teacher_logit_dualview_hbb_qcd.sh",
    "run_independent_fusion_small.sh",
    "run_independent_fusion_large.sh",
    "run_independent_fusion_ensemble_analysis.sh",
    "run_train_heterogeneous_hlt_arch.sh",
    "run_fuse_heterogeneous_hlt4.sh",
    "run_evaluate_offline_teacher_reference.sh",
    "run_diagnose_hlt_offline_disagreement.sh",
    "run_hlt_offline_router_specialists.sh",
    "run_hlt_offline_router_specialists_hbb_qcd.sh",
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
    "run_train_detr_slot_reconstructor.sh",
    "run_cache_detr_slot_reco_views.sh",
    "run_train_detr_slot_five_view_tagger.sh",
    "run_write_detr_slot_final_report.sh",
    "run_subtoken_part_compat.sh",
    "run_subtoken_part_distill.sh",
    "run_write_subtoken_part_report.sh",
    "run_train_local_graph_part_tagger.sh",
    "run_write_local_graph_part_report.sh",
    "run_cache_local_graph_baseline_logits.sh",
    "run_train_local_graph_residual_expert.sh",
    "run_write_local_graph_residual_expert_report.sh",
    "run_cache_local_graph_residual_v2_embeddings.sh",
    "run_train_local_graph_residual_expert_v2.sh",
    "run_write_local_graph_residual_expert_v2_report.sh",
    "run_train_multiscale_subjet_part_tagger.sh",
    "run_write_multiscale_subjet_part_report.sh",
    "run_local_graph_score_fusion.sh",
    "run_local_graph_multiscale_score_fusion.sh",
    "run_train_dualview_part_residual.sh",
    "run_write_dualview_part_report.sh",
    "run_train_local_compression_part.sh",
    "run_write_local_compression_part_report.sh",
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
    "submit_set_matching_hbb_qcd_binary_experiment.sh",
    "submit_offline_binary_qcd_tbqq_reference.sh",
    "submit_set_matching_qcd_tbqq_binary_experiment.sh",
    "submit_detr_slot_qcd_tbqq_binary_experiment.sh",
    "submit_detr_slot_qcd_hgg_binary_experiment.sh",
    "submit_detr_slot_smoke_test.sh",
    "submit_subtoken_part_qcd_hgg_binary_experiment.sh",
    "submit_subtoken_part_10class_experiment.sh",
    "submit_local_graph_qcd_hgg_binary_experiment.sh",
    "submit_local_graph_step10_first_serious_run.sh",
    "submit_local_graph_step10_3m1m1m_reuse_cache_with_fusion.sh",
    "submit_local_graph_residual_expert_experiment.sh",
    "submit_local_graph_residual_expert_v2_experiment.sh",
    "submit_local_graph_residual_expert_v2_3m1m1m_serious.sh",
    "submit_local_graph_residual_expert_v2_3m1m1m_ablation_suite.sh",
    "submit_multiscale_subjet_qcd_hgg_binary_experiment.sh",
    "submit_dualview_part_residual_smoke_test.sh",
    "submit_dualview_part_residual_500k_qcd_hgg.sh",
    "submit_local_compression_part_qcd_hgg_hlt0p6_experiment.sh",
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

    def test_common_mirrors_lightweight_diagnostics_outside_checkpoints(self):
        common = self.read("common.sh")
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn('DIAGNOSTICS_ROOT:=${PROJECT_DIR}/fresh_check_diagnostics', common)
        self.assertIn("MIRROR_DIAGNOSTICS:=1", common)
        self.assertIn("DIAGNOSTICS_MAX_FILE_MB:=25", common)
        self.assertIn("fresh_register_diagnostics_dir", common)
        self.assertIn("fresh_mirror_run_diagnostics", common)
        self.assertIn("fresh_mirror_current_slurm_logs", common)
        self.assertIn("fresh_on_exit", common)
        self.assertIn("RUN_CONFIG_DIAGNOSTICS_DIR", common)
        self.assertIn('"diagnostics_dir": os.environ.get("RUN_CONFIG_DIAGNOSTICS_DIR")', common)
        self.assertIn(".json", common)
        self.assertIn(".csv", common)
        self.assertIn(".pt", common)
        self.assertIn(".npz", common)
        self.assertIn("fresh_check_diagnostics/", gitignore)

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
            "run_build_label_filtered_split_manifest.sh",
            "run_build_label_filtered_hlt_cache.sh",
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

    def test_v2_teacher_logit_dualview_debug_runner_is_small_debug_probe(self):
        text = self.read("run_v2_teacher_logit_dualview_debug.sh")
        script = (REPO_ROOT / "scripts" / "run_v2_teacher_logit_dualview_debug.py").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=debug", text)
        self.assertIn("#SBATCH --time=24:00:00", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("scripts/run_v2_teacher_logit_dualview_debug.py", text)
        self.assertIn('V2_TLOG_DUALVIEW_DEBUG_TRAIN_SIZE:=20000', text)
        self.assertIn('V2_TLOG_DUALVIEW_DEBUG_VAL_SIZE:=5000', text)
        self.assertIn('V2_TLOG_DUALVIEW_DEBUG_TEST_SIZE:=20000', text)
        self.assertIn("--teacher-checkpoint", text)
        self.assertIn("--reco-epochs", text)
        self.assertIn("--dual-epochs", text)
        self.assertIn('fresh_claim_new_dir "${V2_TLOG_DUALVIEW_DEBUG_ROOT}"', text)
        self.assertIn('fresh_require_file "${V2_TLOG_DUALVIEW_DEBUG_ROOT}/evaluation_report.json"', text)
        self.assertIn("_balanced_indices_for_labels", script)
        self.assertIn("_limit_manifest_split_by_indices", script)
        self.assertIn("balanced_row_selection", script)
        self.assertIn("_safe_teacher_eval_logits", script)
        self.assertIn("logit_sanitization", script)
        self.assertNotIn("load_paired_jet_views(", script)

    def test_v2_teacher_logit_dualview_hbb_qcd_runner_uses_binary_normal_split(self):
        text = self.read("run_v2_teacher_logit_dualview_hbb_qcd.sh")
        self.assertIn("#SBATCH --partition=debug", text)
        self.assertIn("#SBATCH --time=24:00:00", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("scripts/run_v2_teacher_logit_dualview_debug.py", text)
        self.assertIn("--label-filter-names QCD Hbb", text)
        self.assertIn("V2_TLOG_HBB_QCD_TRAIN_SIZE:=100000", text)
        self.assertIn("V2_TLOG_HBB_QCD_VAL_SIZE:=30000", text)
        self.assertIn("V2_TLOG_HBB_QCD_TEST_SIZE:=100000", text)
        self.assertIn('fresh_claim_new_dir "${V2_TLOG_HBB_QCD_ROOT}"', text)
        self.assertIn('fresh_require_file "${V2_TLOG_HBB_QCD_ROOT}/evaluation_report.json"', text)

    def test_hlt_offline_disagreement_diagnostic_is_short_debug_runner(self):
        text = self.read("run_diagnose_hlt_offline_disagreement.sh")
        self.assertIn("#SBATCH --partition=debug", text)
        self.assertIn("#SBATCH --time=02:00:00", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("scripts/diagnose_hlt_offline_disagreement.py", text)
        self.assertIn('DISAGREE_DIAG_SPLIT:=stack_val', text)
        self.assertIn('DISAGREE_DIAG_MAX_JETS:=50000', text)
        self.assertIn("--hlt-checkpoint", text)
        self.assertIn("--offline-checkpoint", text)
        self.assertIn('fresh_require_file "${DISAGREE_DIAG_DIR}/disagreement_diagnostic_report.json"', text)

    def test_hlt_offline_router_specialists_runner_trains_two_specialists(self):
        text = self.read("run_hlt_offline_router_specialists.sh")
        hbb_qcd = self.read("run_hlt_offline_router_specialists_hbb_qcd.sh")
        self.assertIn("#SBATCH --partition=debug", text)
        self.assertIn("#SBATCH --time=12:00:00", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("scripts/run_hlt_offline_router_specialists.py", text)
        self.assertIn("ROUTER_SPECIALIST_MAX_TRAIN_JETS:=150000", text)
        self.assertIn("ROUTER_SPECIALIST_MAX_VAL_JETS:=50000", text)
        self.assertIn("ROUTER_SPECIALIST_MAX_TEST_JETS:=100000", text)
        self.assertIn('fresh_require_file "${ROUTER_SPECIALIST_ROOT}/specialists/agreement/best_model_val.pt"', text)
        self.assertIn('fresh_require_file "${ROUTER_SPECIALIST_ROOT}/specialists/disagreement/best_model_val.pt"', text)
        script = (REPO_ROOT / "scripts" / "run_hlt_offline_router_specialists.py").read_text(encoding="utf-8")
        self.assertIn("agreement", script)
        self.assertIn("disagreement", script)
        self.assertIn("delta_vs_hlt_probe_accuracy", script)
        self.assertIn("balanced_indices_for_labels", script)
        self.assertIn("balanced_row_selection_applied", script)
        self.assertIn("logit_sanitization", script)
        self.assertIn("repair_nonfinite_logits_for_eval", script)
        self.assertIn("#SBATCH --job-name=hlt_route_hbbqcd", hbb_qcd)
        self.assertIn("scripts/run_hlt_offline_router_specialists.py", hbb_qcd)
        self.assertIn("--label-filter-names QCD Hbb", hbb_qcd)
        self.assertIn("ROUTER_SPECIALIST_HBB_QCD_MAX_TRAIN_JETS:=150000", hbb_qcd)
        self.assertIn("ROUTER_SPECIALIST_HBB_QCD_MAX_VAL_JETS:=50000", hbb_qcd)
        self.assertIn("ROUTER_SPECIALIST_HBB_QCD_MAX_TEST_JETS:=100000", hbb_qcd)

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

    def test_set_matching_binary_offline_teacher_runner_trains_fresh_two_class_reference(self):
        runner = self.read("run_train_eval_set_matching_binary_offline_teacher.sh")
        script = (REPO_ROOT / "scripts" / "train_eval_set_matching_binary_offline_teacher.py").read_text(encoding="utf-8")

        self.assertIn("scripts/train_eval_set_matching_binary_offline_teacher.py", runner)
        self.assertIn("offline_teacher_reference/fresh_binary_part_", runner)
        self.assertIn("--manifest-path \"${SET_MATCHING_MANIFEST_PATH}\"", runner)
        self.assertIn("--data-dir \"${DATA_DIR}\"", runner)
        self.assertIn("--label-filter-names \"${label_filter_args[@]}\"", runner)
        self.assertIn("--label-names \"${label_name_args[@]}\"", runner)
        self.assertIn("--max-train-jets \"${BINARY_OFFLINE_TEACHER_MAX_TRAIN_JETS}\"", runner)
        self.assertIn("--max-val-jets \"${BINARY_OFFLINE_TEACHER_MAX_VAL_JETS}\"", runner)
        self.assertIn("--max-stack-val-jets \"${BINARY_OFFLINE_TEACHER_MAX_STACK_VAL_JETS}\"", runner)
        self.assertIn("--max-final-test-jets \"${BINARY_OFFLINE_TEACHER_MAX_FINAL_TEST_JETS}\"", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn("diagnostics/summary.csv", runner)
        self.assertIn("build_particle_transformer_classifier(num_classes=2", script)
        self.assertIn("classification_metrics_from_predictions", script)
        self.assertIn("fpr_at_signal_eff_0p30", script)
        self.assertIn("fresh_binary_offline_upper_reference", script)

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
        self.assertIn("--missing-target-weight \"${SET_MATCHING_MISSING_TARGET_WEIGHT}\"", train)

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
        self.assertIn('selection_mode="all_slots"', tagger)
        self.assertIn("--selection-metric \"${SET_MATCHING_TAGGER_SELECTION_METRIC}\"", tagger)
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

    def test_set_matching_hbb_qcd_binary_submitter_sets_two_class_task(self):
        submitter = self.read("submit_set_matching_hbb_qcd_binary_experiment.sh")
        manifest_runner = self.read("run_build_label_filtered_split_manifest.sh")
        binary_cache_runner = self.read("run_build_label_filtered_hlt_cache.sh")
        tagger = self.read("run_train_five_view_tagger.sh")
        train = self.read("run_train_set_matching_reconstructor.sh")
        cache = self.read("run_cache_set_matching_multiview.sh")
        audit = self.read("run_audit_five_view_tagger.sh")

        self.assertIn('SET_MATCHING_LABEL_FILTER_NAMES="QCD Hbb"', submitter)
        self.assertIn('SET_MATCHING_LABEL_NAMES="QCD Hbb"', submitter)
        self.assertIn("SET_MATCHING_NUM_CLASSES=2", submitter)
        self.assertIn("HBB_QCD_BUILD_BINARY_INPUTS:=1", submitter)
        self.assertIn("HBB_QCD_BINARY_MANIFEST_PATH", submitter)
        self.assertIn("HBB_QCD_BINARY_HLT_CACHE_DIR", submitter)
        self.assertIn('export SET_MATCHING_MANIFEST_PATH="${HBB_QCD_BINARY_MANIFEST_PATH}"', submitter)
        self.assertIn('export SET_MATCHING_HLT_CACHE_DIR="${HBB_QCD_BINARY_HLT_CACHE_DIR}"', submitter)
        self.assertIn("run_build_label_filtered_split_manifest.sh", submitter)
        self.assertIn("run_build_label_filtered_hlt_cache.sh", submitter)
        self.assertIn('input_dependency="${binary_hlt_cache_jid}"', submitter)
        self.assertIn("scripts/build_label_filtered_split_manifest.py", manifest_runner)
        self.assertIn("LABEL_FILTER_REMAP_LABELS", manifest_runner)
        self.assertIn("--remap-labels", manifest_runner)
        self.assertIn("scripts/build_fixed_hlt_cache.py", binary_cache_runner)
        self.assertIn("HLT_DEGRADATION_STRENGTH:=1.0", binary_cache_runner)
        self.assertIn("--hlt-degradation-strength", binary_cache_runner)
        self.assertIn('HBB_QCD_TAGGER_VARIANTS:=hlt_only hlt_plus_gt hlt_plus_pn hlt_plus_pfn hlt_plus_pcnn five_view_plain five_view_geometry five_view_no_confidence view_label_shuffle_control', submitter)
        self.assertIn("SET_MATCHING_EVAL_REQUIRE_ALL_CANONICAL=1", submitter)
        self.assertIn("HBB_QCD_BINARY_MANIFEST_MEM:=8G", submitter)
        self.assertIn("HBB_QCD_BINARY_HLT_CACHE_MEM:=64G", submitter)
        self.assertIn("HBB_QCD_RECO_MEM:=64G", submitter)
        self.assertIn("HBB_QCD_CACHE_MEM:=64G", submitter)
        self.assertIn("HBB_QCD_TAGGER_MEM:=64G", submitter)
        self.assertIn("HBB_QCD_AUDIT_MEM:=48G", submitter)
        self.assertIn("HBB_QCD_REPORT_MEM:=8G", submitter)
        self.assertIn("HBB_QCD_OFFLINE_TEACHER_MEM:=64G", submitter)
        self.assertIn("HBB_QCD_RECO_TIME:=2-00:00:00", submitter)
        self.assertIn("HBB_QCD_TAGGER_TIME:=2-00:00:00", submitter)
        self.assertIn("HBB_QCD_SUBMIT_OFFLINE_TEACHER_REFERENCE:=1", submitter)
        self.assertIn("HBB_QCD_TAGGER_SELECTION_METRIC:-fpr_at_signal_eff_0p50", submitter)
        self.assertIn('--mem="${HBB_QCD_RECO_MEM}"', submitter)
        self.assertIn('--mem="${HBB_QCD_CACHE_MEM}"', submitter)
        self.assertIn('--mem="${HBB_QCD_TAGGER_MEM}"', submitter)
        self.assertIn('--time="${HBB_QCD_RECO_TIME}"', submitter)
        self.assertIn('--time="${HBB_QCD_TAGGER_TIME}"', submitter)
        self.assertIn('--cpus-per-task="${HBB_QCD_RECO_CPUS}"', submitter)
        self.assertIn("run_train_eval_set_matching_binary_offline_teacher.sh", submitter)
        self.assertIn("hbbqcd_offline_teacher_reference", submitter)
        self.assertIn("offline_teacher_reference_job_id", submitter)
        self.assertIn("scripts/train_five_view_tagger.py", tagger)
        self.assertIn("--num-classes", tagger)
        self.assertIn("--label-names", tagger)
        self.assertIn("--label-filter-names", tagger)
        self.assertIn("--label-filter-names", train)
        self.assertIn("--label-filter-names", cache)
        self.assertIn("--label-filter-names", audit)
        metrics_code = (REPO_ROOT / "teacher_logit_reco" / "set_matching" / "five_view_train.py").read_text(encoding="utf-8")
        self.assertIn("fpr_at_signal_eff_0p30", metrics_code)
        self.assertIn("selection_metric", metrics_code)
        self.assertIn("run_write_set_matching_multiview_final_report.sh", submitter)
        self.assertIn("final_report_job_id", submitter)

    def test_label_filtered_manifest_builder_can_remap_noncontiguous_labels(self):
        script = (REPO_ROOT / "scripts" / "build_label_filtered_split_manifest.py").read_text(encoding="utf-8")

        self.assertIn("--remap-labels", script)
        self.assertIn("label_filter_remap_labels", script)
        self.assertIn("label_filter_source_to_filtered_label", script)
        self.assertIn("label_filter_filtered_to_source_label", script)
        self.assertIn("source_to_filtered_label", script)
        self.assertIn("Pass --remap-labels for labels like QCD Tbqq", script)

    def test_label_filtered_fresh_split_builder_samples_after_filtering(self):
        script = (REPO_ROOT / "scripts" / "build_label_filtered_fresh_splits.py").read_text(encoding="utf-8")
        runner = self.read("run_build_label_filtered_fresh_splits.sh")

        self.assertIn("split size caps", script)
        self.assertIn("after selecting QCD/Tbqq", script)
        self.assertIn("split_size_semantics", script)
        self.assertIn("after_label_filtering", script)
        self.assertIn("discover_file_records", script)
        self.assertIn("require_all_classes=False", script)
        self.assertIn("source_to_filtered", script)
        self.assertIn("requested_per_class_total", script)
        self.assertIn("multi_data_dir", script)
        self.assertIn("multi_data_dir_record_paths_are_absolute", script)
        self.assertIn("scripts/build_label_filtered_fresh_splits.py", runner)
        self.assertIn("DATA_DIRS", runner)
        self.assertIn("LABEL_FILTER_MODEL_TRAIN_SIZE", runner)
        self.assertIn("--model-train", runner)

    def test_qcd_tbqq_offline_reference_submitter_queues_manifest_and_offline_part(self):
        submitter = self.read("submit_offline_binary_qcd_tbqq_reference.sh")
        trainer = (REPO_ROOT / "scripts" / "train_eval_set_matching_binary_offline_teacher.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("QCD_vs_Tbqq", submitter)
        self.assertIn("QCD_TBQQ_ROOT", submitter)
        self.assertIn("offline_binary_qcd_tbqq_", submitter)
        self.assertIn("QCD_TBQQ_LABEL_NAMES:=QCD Tbqq", submitter)
        self.assertIn("QCD_TBQQ_MODEL_TRAIN_SIZE:=500000", submitter)
        self.assertIn("QCD_TBQQ_MODEL_VAL_SIZE:=150000", submitter)
        self.assertIn("QCD_TBQQ_STACK_TRAIN_SIZE:=500000", submitter)
        self.assertIn("QCD_TBQQ_STACK_VAL_SIZE:=150000", submitter)
        self.assertIn("QCD_TBQQ_FINAL_TEST_SIZE:=500000", submitter)
        self.assertIn("QCD_TBQQ_BUILD_DIRECT_BINARY_SPLITS:=1", submitter)
        self.assertIn("QCD_TBQQ_OFFLINE_EPOCHS:=45", submitter)
        self.assertIn("LABEL_FILTER_REMAP_LABELS=1", submitter)
        self.assertIn("run_build_label_filtered_fresh_splits.sh", submitter)
        self.assertIn("run_build_label_filtered_split_manifest.sh", submitter)
        self.assertIn("run_train_eval_set_matching_binary_offline_teacher.sh", submitter)
        self.assertIn('--dependency="afterok:${manifest_jid}"', submitter)
        self.assertIn("binary_manifest: 1", submitter)
        self.assertIn("offline_part_reference: 1", submitter)
        self.assertIn("offline_run_report", submitter)
        self.assertIn("load_split_manifest(args.manifest_path)", trainer)
        self.assertIn("class_names=manifest.class_names", trainer)

    def test_qcd_tbqq_set_matching_submitter_queues_full_binary_graph(self):
        submitter = self.read("submit_set_matching_qcd_tbqq_binary_experiment.sh")

        self.assertIn("QCD_vs_Tbqq", submitter)
        self.assertIn("set_matching_qcd_tbqq_binary_", submitter)
        self.assertIn("QCD_TBQQ_SOURCE_LABEL_NAMES:=QCD Tbqq", submitter)
        self.assertIn("QCD_TBQQ_BINARY_LABEL_FILTER:=0 1", submitter)
        self.assertIn('export SET_MATCHING_LABEL_FILTER_NAMES="${QCD_TBQQ_BINARY_LABEL_FILTER}"', submitter)
        self.assertIn('export SET_MATCHING_LABEL_NAMES="${QCD_TBQQ_SOURCE_LABEL_NAMES}"', submitter)
        self.assertIn("SET_MATCHING_NUM_CLASSES=2", submitter)
        self.assertIn("QCD_TBQQ_BUILD_DIRECT_BINARY_SPLITS:=1", submitter)
        self.assertIn("QCD_TBQQ_OFFLINE_TEACHER_EPOCHS:=45", submitter)
        self.assertIn('export SET_MATCHING_TAGGER_EPOCHS="${QCD_TBQQ_TAGGER_EPOCHS:-45}"', submitter)
        self.assertIn('export BINARY_OFFLINE_TEACHER_EPOCHS="${QCD_TBQQ_OFFLINE_TEACHER_EPOCHS}"', submitter)
        self.assertIn("LABEL_FILTER_REMAP_LABELS=1", submitter)
        self.assertIn('export LABEL_FILTER_NAMES="${QCD_TBQQ_SOURCE_LABEL_NAMES}"', submitter)
        self.assertIn("QCD_TBQQ_MODEL_TRAIN_SIZE:-500000", submitter)
        self.assertIn("QCD_TBQQ_MODEL_VAL_SIZE:-150000", submitter)
        self.assertIn("QCD_TBQQ_STACK_TRAIN_SIZE:-500000", submitter)
        self.assertIn("QCD_TBQQ_STACK_VAL_SIZE:-150000", submitter)
        self.assertIn("QCD_TBQQ_FINAL_TEST_SIZE:-500000", submitter)
        self.assertIn("QCD_TBQQ_TAGGER_SELECTION_METRIC:-fpr_at_signal_eff_0p50", submitter)
        self.assertIn("run_build_label_filtered_fresh_splits.sh", submitter)
        self.assertIn("run_build_label_filtered_split_manifest.sh", submitter)
        self.assertIn("run_build_label_filtered_hlt_cache.sh", submitter)
        self.assertIn("run_train_eval_set_matching_binary_offline_teacher.sh", submitter)
        self.assertIn("run_train_set_matching_reconstructor.sh", submitter)
        self.assertIn("run_cache_set_matching_multiview.sh", submitter)
        self.assertIn("run_train_five_view_tagger.sh", submitter)
        self.assertIn("run_audit_five_view_tagger.sh", submitter)
        self.assertIn("run_write_set_matching_multiview_final_report.sh", submitter)
        self.assertIn("qcdtbqq_binary_manifest", submitter)
        self.assertIn("qcdtbqq_binary_hlt_cache", submitter)
        self.assertIn("qcdtbqq_offline_teacher_reference", submitter)
        self.assertIn("tagger_train: ${#tagger_job_ids[@]}", submitter)
        self.assertIn("filtered_manifest_report", submitter)

    def test_detr_slot_runners_and_submitter_queue_full_binary_graph(self):
        common = self.read("common.sh")
        train = self.read("run_train_detr_slot_reconstructor.sh")
        cache = self.read("run_cache_detr_slot_reco_views.sh")
        tagger = self.read("run_train_detr_slot_five_view_tagger.sh")
        report = self.read("run_write_detr_slot_final_report.sh")
        submitter = self.read("submit_detr_slot_qcd_tbqq_binary_experiment.sh")
        hgg = self.read("submit_detr_slot_qcd_hgg_binary_experiment.sh")
        smoke = self.read("submit_detr_slot_smoke_test.sh")

        self.assertIn("DETR_SLOT_ROOT:=${OUTPUT_ROOT}/detr_slot_qcd_tbqq_binary_500k", common)
        self.assertIn("DETR_SLOT_ARCHITECTURES:=gt pn pfn pcnn", common)
        self.assertIn("DETR_SLOT_TAGGER_VARIANTS:=hlt_only hlt_plus_gt hlt_plus_pn hlt_plus_pfn hlt_plus_pcnn five_view_plain five_view_geometry five_view_no_confidence view_label_shuffle_control", common)
        self.assertIn("scripts/train_detr_slot_reconstructor.py", train)
        self.assertIn("--allow-bruteforce-fallback", train)
        self.assertIn("scripts/cache_detr_slot_reco_views.py", cache)
        self.assertIn('--output-dir "${DETR_SLOT_RECONSTRUCTED_VIEW_DIR}"', cache)
        self.assertIn("scripts/train_detr_slot_five_view_tagger.py", tagger)
        self.assertIn("--selection-metric", tagger)
        self.assertIn("scripts/write_detr_slot_final_report.py", report)
        self.assertIn("--five-view-audit-dir", report)
        self.assertIn("DETR_SLOT_REQUIRE_FIVE_VIEW_AUDIT", report)
        self.assertIn("DETR_SLOT_REQUIRE_OFFLINE_REFERENCE", report)
        self.assertIn("QCD_vs_Tbqq", submitter)
        self.assertIn("detr_slot_qcd_tbqq_binary_", submitter)
        self.assertIn("DETR_SLOT_BINARY_ROOT", submitter)
        self.assertIn("DETR_SLOT_TASK_NAME", submitter)
        self.assertIn("DETR_SLOT_SOURCE_LABEL_NAMES:=QCD Tbqq", submitter)
        self.assertIn("DETR_SLOT_BINARY_LABEL_FILTER:=0 1", submitter)
        self.assertIn("DETR_SLOT_HLT_DEGRADATION_STRENGTH", submitter)
        self.assertIn('export HLT_DEGRADATION_STRENGTH="${DETR_SLOT_HLT_DEGRADATION_STRENGTH}"', submitter)
        self.assertIn("DETR_SLOT_BINARY_RECO_EPOCHS", submitter)
        self.assertIn('export DETR_SLOT_LABEL_FILTER_NAMES="${DETR_SLOT_BINARY_LABEL_FILTER}"', submitter)
        self.assertIn('export DETR_SLOT_NUM_CLASSES=2', submitter)
        self.assertIn("run_build_label_filtered_fresh_splits.sh", submitter)
        self.assertIn("run_build_label_filtered_hlt_cache.sh", submitter)
        self.assertIn("run_train_eval_set_matching_binary_offline_teacher.sh", submitter)
        self.assertIn("run_train_detr_slot_reconstructor.sh", submitter)
        self.assertIn("run_cache_detr_slot_reco_views.sh", submitter)
        self.assertIn("run_train_detr_slot_five_view_tagger.sh", submitter)
        self.assertIn("run_audit_five_view_tagger.sh", submitter)
        self.assertIn("run_write_detr_slot_final_report.sh", submitter)
        self.assertIn("detrslot_binary_manifest", submitter)
        self.assertIn("detrslot_binary_hlt_cache", submitter)
        self.assertIn("detrslot_offline_teacher_reference", submitter)
        self.assertIn("detrslot_final_report", submitter)
        self.assertIn("detr_reco_train: ${#reco_train_job_ids[@]}", submitter)
        self.assertIn("detr_cache_reconstructed_views: ${#cache_job_ids[@]}", submitter)
        self.assertIn("detr_tagger_train: ${#tagger_job_ids[@]}", submitter)
        self.assertIn("submit_offline_teacher_reference=${DETR_SLOT_SUBMIT_OFFLINE_TEACHER_REFERENCE}", submitter)
        self.assertIn("QCD_vs_Hgg_DETR_free_slot", hgg)
        self.assertIn("detr_slot_qcd_hgg_binary_hlt", hgg)
        self.assertIn('export DETR_SLOT_SOURCE_LABEL_NAMES="QCD Hgg"', hgg)
        self.assertIn('DETR_SLOT_QCD_HGG_HLT_DEGRADATION_STRENGTH:=0.6', hgg)
        self.assertIn('export DETR_SLOT_HLT_DEGRADATION_STRENGTH="${DETR_SLOT_QCD_HGG_HLT_DEGRADATION_STRENGTH}"', hgg)
        self.assertIn('export DETR_SLOT_BINARY_RECO_EPOCHS="${DETR_SLOT_QCD_HGG_RECO_EPOCHS:-30}"', hgg)
        self.assertIn('export DETR_SLOT_BINARY_TAGGER_EPOCHS="${DETR_SLOT_QCD_HGG_TAGGER_EPOCHS:-45}"', hgg)
        self.assertIn('export DETR_SLOT_RECO_TIME="${DETR_SLOT_QCD_HGG_RECO_TIME:-2-12:00:00}"', hgg)
        self.assertIn('export DETR_SLOT_TAGGER_TIME="${DETR_SLOT_QCD_HGG_TAGGER_TIME:-2-12:00:00}"', hgg)
        self.assertIn('export DETR_SLOT_NUM_SLOTS="${DETR_SLOT_QCD_HGG_NUM_SLOTS:-160}"', hgg)
        self.assertIn('export DETR_SLOT_BINARY_MODEL_TRAIN_SIZE="${DETR_SLOT_QCD_HGG_MODEL_TRAIN_SIZE:-500000}"', hgg)
        self.assertIn('export DETR_SLOT_BINARY_MODEL_VAL_SIZE="${DETR_SLOT_QCD_HGG_MODEL_VAL_SIZE:-150000}"', hgg)
        self.assertIn('export DETR_SLOT_BINARY_STACK_TRAIN_SIZE="${DETR_SLOT_QCD_HGG_STACK_TRAIN_SIZE:-500000}"', hgg)
        self.assertIn('export DETR_SLOT_BINARY_STACK_VAL_SIZE="${DETR_SLOT_QCD_HGG_STACK_VAL_SIZE:-150000}"', hgg)
        self.assertIn('export DETR_SLOT_BINARY_FINAL_TEST_SIZE="${DETR_SLOT_QCD_HGG_FINAL_TEST_SIZE:-500000}"', hgg)
        self.assertIn('bash "${SCRIPT_DIR}/submit_detr_slot_qcd_tbqq_binary_experiment.sh"', hgg)
        self.assertIn("detr_slot_smoke_", smoke)
        self.assertIn("smoke metrics are for pipeline correctness only", smoke)
        self.assertIn('bash "${SCRIPT_DIR}/submit_detr_slot_qcd_tbqq_binary_experiment.sh"', smoke)

    def test_subtoken_part_qcd_hgg_submitter_can_queue_steps21_and22(self):
        compat = self.read("run_subtoken_part_compat.sh")
        distill = self.read("run_subtoken_part_distill.sh")
        report = self.read("run_write_subtoken_part_report.sh")
        submitter = self.read("submit_subtoken_part_qcd_hgg_binary_experiment.sh")

        self.assertIn("scripts/run_subtoken_part_compat.py", compat)
        self.assertIn("scripts/train_subtoken_part_distill.py", distill)
        self.assertIn("--offline-teacher-checkpoint", distill)
        self.assertIn("--distillation-weight", distill)
        self.assertIn("--modality-residual-weight", distill)
        self.assertIn("--gate-residual-regularization-weight", distill)
        self.assertIn("--masked-subtoken-max-match-delta-r", distill)
        self.assertIn("scripts/write_subtoken_part_report.py", report)
        self.assertIn("QCD Hgg", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH:=0.6", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_MODEL_TRAIN_SIZE:=500000", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_MODEL_VAL_SIZE:=150000", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_STACK_TRAIN_SIZE:=500000", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_STACK_VAL_SIZE:=150000", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_FINAL_TEST_SIZE:=500000", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_EPOCHS:=45", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_SELECTION_METRIC:=fpr_at_signal_eff_0p50", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_TIME:=1-00:00:00", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_COMPAT_TIME:=3-00:00:00", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_OFFLINE_TEACHER_TIME:=2-00:00:00", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_VERSION_B_TIME:=3-00:00:00", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_SUBMIT_VERSION_B:=0", submitter)
        self.assertIn("SUBTOKEN_PART_QCD_HGG_VERSION_B_VARIANTS:=subtoken_gate_context_distill subtoken_gate_context_residual subtoken_gate_context_distill_residual", submitter)
        self.assertIn("run_train_eval_set_matching_binary_offline_teacher.sh", submitter)
        self.assertIn("run_subtoken_part_distill.sh", submitter)
        self.assertIn("run_write_subtoken_part_report.sh", submitter)
        self.assertIn("subtoken_qcdhgg_version_a_compat", submitter)
        self.assertIn("subtoken_qcdhgg_offline_teacher_reference", submitter)
        self.assertIn("subtoken_qcdhgg_version_b_report", submitter)
        self.assertIn('"subtoken_gate_context": "../version_a_comparison/subtoken_gate_context/run_report.json"', submitter)
        self.assertIn('export SET_MATCHING_LABEL_FILTER_NAMES="${SUBTOKEN_PART_QCD_HGG_BINARY_LABEL_FILTER}"', submitter)
        self.assertIn('export SUBTOKEN_PART_DISTILL_LABEL_FILTER_NAMES="${SUBTOKEN_PART_QCD_HGG_BINARY_LABEL_FILTER}"', submitter)
        self.assertIn('version_b_report_dependency="${compat_jid}"', submitter)
        self.assertIn('SUBTOKEN_PART_REPORT_VARIANTS="subtoken_gate_context ${SUBTOKEN_PART_QCD_HGG_VERSION_B_VARIANTS}"', submitter)

    def test_subtoken_part_10class_submitter_builds_fresh_hlt06_cache(self):
        submitter = self.read("submit_subtoken_part_10class_experiment.sh")

        self.assertIn("10class_subtoken_part_version_a", submitter)
        self.assertIn("subtoken_part_10class_hlt", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_HLT_DEGRADATION_STRENGTH:=0.6", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_LABEL_NAMES:=QCD Hbb Hcc Hgg H4q Hqql Zqq Wqq Tbqq Tbl", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_VARIANTS:=hlt_part_baseline subtoken_no_gate subtoken_gate_local_only subtoken_gate_context", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_MODEL_TRAIN_SIZE:=500000", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_MODEL_VAL_SIZE:=150000", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_STACK_TRAIN_SIZE:=500000", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_STACK_VAL_SIZE:=150000", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_FINAL_TEST_SIZE:=500000", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_EPOCHS:=45", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_SELECTION_METRIC:=accuracy", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_HLT_CACHE_TIME:=1-00:00:00", submitter)
        self.assertIn("SUBTOKEN_PART_10CLASS_COMPAT_TIME:=4-00:00:00", submitter)
        self.assertIn('export SUBTOKEN_PART_LABEL_FILTER_NAMES="${SUBTOKEN_PART_10CLASS_LABEL_NAMES}"', submitter)
        self.assertIn("export SUBTOKEN_PART_NUM_CLASSES=10", submitter)
        self.assertIn('export SUBTOKEN_PART_REPORT_PRIMARY_METRIC="${SUBTOKEN_PART_10CLASS_SELECTION_METRIC}"', submitter)
        self.assertIn("run_build_fresh_splits.sh", submitter)
        self.assertIn("run_build_fresh_hlt_cache.sh", submitter)
        self.assertIn("run_subtoken_part_compat.sh", submitter)
        self.assertIn("subtoken_10class_splits", submitter)
        self.assertIn("subtoken_10class_hlt_cache", submitter)
        self.assertIn("subtoken_10class_version_a_compat", submitter)
        self.assertIn('--dependency="afterok:${split_jid}"', submitter)
        self.assertIn("afterok_args", submitter)

    def test_local_graph_part_qcd_hgg_submitter_queues_baseline_adapters_and_report(self):
        train = self.read("run_train_local_graph_part_tagger.sh")
        report = self.read("run_write_local_graph_part_report.sh")
        submitter = self.read("submit_local_graph_qcd_hgg_binary_experiment.sh")

        self.assertIn("scripts/train_local_graph_part_tagger.py", train)
        self.assertIn("local_point_attention_adapter_warmstart", train)
        self.assertIn("--warm-start-checkpoint", train)
        self.assertIn("--require-warm-start", train)
        self.assertIn("--freeze-part-epochs", train)
        self.assertIn("--confirm-final-test", train)
        self.assertIn("--expected-hlt-degradation-strength", train)
        self.assertIn("LOCAL_GRAPH_PART_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6", train)
        self.assertIn("--skip-hlt-params-check", train)
        self.assertIn("diagnostics/warm_start_report.json", train)

        self.assertIn("scripts/write_local_graph_part_report.py", report)
        self.assertIn("local_graph_part_report.json", report)
        self.assertIn("adapter_diagnostics.csv", report)
        self.assertIn("hlt_degradation_summary.csv", report)

        self.assertIn("QCD_vs_Hgg_local_graph_part", submitter)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH:=0.6", submitter)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_MODEL_TRAIN_SIZE:=500000", submitter)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_MODEL_VAL_SIZE:=150000", submitter)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_STACK_TRAIN_SIZE:=500000", submitter)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_STACK_VAL_SIZE:=150000", submitter)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_FINAL_TEST_SIZE:=500000", submitter)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_EPOCHS:=45", submitter)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_SELECTION_METRIC:=fpr_at_signal_eff_0p50", submitter)
        self.assertIn('export LOCAL_GRAPH_PART_EXPECTED_HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"', submitter)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_VARIANTS:=hlt_part_baseline local_edgeconv_adapter local_point_attention_adapter local_point_attention_adapter_warmstart", submitter)
        self.assertIn("run_build_label_filtered_fresh_splits.sh", submitter)
        self.assertIn("run_build_label_filtered_hlt_cache.sh", submitter)
        self.assertIn("run_train_local_graph_part_tagger.sh", submitter)
        self.assertIn("run_write_local_graph_part_report.sh", submitter)
        self.assertIn("localgraph_part_baseline", submitter)
        self.assertIn("localgraph_part_warmstart", submitter)
        self.assertIn('warmstart_dependency="${baseline_jid}"', submitter)
        self.assertIn('--dependency="afterok:${train_dep}"', submitter)
        self.assertIn("local_graph_train: ${#train_job_ids[@]}", submitter)
        self.assertIn("final_report_json", submitter)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_SUBMIT_SCORE_FUSION:=0", submitter)
        self.assertIn("run_local_graph_score_fusion.sh", submitter)
        self.assertIn("localgraph_score_fusion", submitter)
        self.assertIn("score_fusion_report", submitter)

    def test_local_graph_part_step10_wrapper_sets_first_serious_run_defaults(self):
        wrapper = self.read("submit_local_graph_step10_first_serious_run.sh")

        self.assertIn("QCD_vs_Hgg_local_graph_part_step10", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_HLT_DEGRADATION_STRENGTH:=0.6", wrapper)
        self.assertIn('export LOCAL_GRAPH_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_PART_STEP10_HLT_DEGRADATION_STRENGTH}"', wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_ROOT", wrapper)
        self.assertIn("local_graph_part_step10_qcd_hgg_binary_hlt", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_VARIANTS:-hlt_part_baseline local_edgeconv_adapter local_point_attention_adapter local_point_attention_adapter_warmstart", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_MODEL_TRAIN_SIZE:-500000", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_MODEL_VAL_SIZE:-150000", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_STACK_TRAIN_SIZE:-500000", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_STACK_VAL_SIZE:-150000", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_FINAL_TEST_SIZE:-500000", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_EPOCHS:-45", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_SELECTION_METRIC:-fpr_at_signal_eff_0p50", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_TRAIN_TIME:-2-12:00:00", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_STEP10_TRAIN_MEM:-160G", wrapper)
        self.assertIn('bash "${SCRIPT_DIR}/submit_local_graph_qcd_hgg_binary_experiment.sh"', wrapper)

    def test_local_graph_part_3m_wrapper_reuses_existing_cache_and_queues_fusion(self):
        wrapper = self.read("submit_local_graph_step10_3m1m1m_reuse_cache_with_fusion.sh")

        self.assertIn("qcd_hgg_hlt06_3m1m1m_full_20260628_194154", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_BUILD_BINARY_INPUTS=0", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_PATH", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_DIR", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_3M_MODEL_TRAIN_SIZE:-3000000", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_3M_MODEL_VAL_SIZE:-1000000", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_3M_STACK_TRAIN_SIZE:-3000000", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_3M_STACK_VAL_SIZE:-1000000", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_3M_FINAL_TEST_SIZE:-1000000", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_QCD_HGG_SUBMIT_SCORE_FUSION=1", wrapper)
        self.assertIn("LOCAL_GRAPH_PART_3M_SCORE_FUSION_TIME:-1-00:00:00", wrapper)
        self.assertIn('bash "${SCRIPT_DIR}/submit_local_graph_qcd_hgg_binary_experiment.sh"', wrapper)

    def test_multiscale_subjet_part_submitter_queues_step12_protocol(self):
        train = self.read("run_train_multiscale_subjet_part_tagger.sh")
        report = self.read("run_write_multiscale_subjet_part_report.sh")
        submitter = self.read("submit_multiscale_subjet_qcd_hgg_binary_experiment.sh")
        trainer = (REPO_ROOT / "scripts" / "train_multiscale_subjet_part_tagger.py").read_text(encoding="utf-8")
        reporter = (REPO_ROOT / "scripts" / "write_multiscale_subjet_part_report.py").read_text(encoding="utf-8")
        model = (REPO_ROOT / "teacher_logit_reco" / "multiscale_subjet_part" / "model.py").read_text(encoding="utf-8")

        self.assertIn("scripts/train_multiscale_subjet_part_tagger.py", train)
        self.assertIn("--confirm-final-test", train)
        self.assertIn("--expected-hlt-degradation-strength", train)
        self.assertIn("MULTISCALE_SUBJET_PART_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6", train)
        self.assertIn("--disable-subjet-pair-bias", train)
        self.assertIn("--ablation-profile", train)
        self.assertIn("--scale-profile", train)
        self.assertIn("--disable-assignment-scale-embedding", train)
        self.assertIn("--disable-token-scale-embedding", train)
        self.assertIn("--disable-scale-pair-embedding", train)
        self.assertIn("no_scale_bias)", train)
        self.assertIn("many_subjets)", train)
        self.assertIn("two_hlt_part_ensemble", train)
        self.assertIn("diagnostics/training_curves.json", train)
        self.assertIn("validation_threshold_final_test_fpr", trainer)
        self.assertIn("--assignment-geometry-bias-strength", trainer)
        self.assertIn("part_plus_subjet_late_fusion", model)
        self.assertIn("two_hlt_part_ensemble_control", model)

        self.assertIn("scripts/write_multiscale_subjet_part_report.py", report)
        self.assertIn("multiscale_subjet_part_report.json", report)
        self.assertIn("diagnostics.csv", report)
        self.assertIn("hlt_degradation.csv", report)
        self.assertIn("--primary-variant", report)
        self.assertIn("--allow-missing-default-controls", reporter)

        self.assertIn("QCD_vs_Hgg_multiscale_subjet_part", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH:=0.6", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_MODEL_TRAIN_SIZE:=500000", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_MODEL_VAL_SIZE:=150000", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_STACK_TRAIN_SIZE:=500000", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_STACK_VAL_SIZE:=150000", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_FINAL_TEST_SIZE:=500000", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_EPOCHS:=45", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_SELECTION_METRIC:=fpr_at_signal_eff_0p50", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_BASE_PROFILES:=hlt_part_baseline multiscale_subjet_residual_part_adapter pure_perceiver_latent_control part_plus_random_subjet_control subjet_branch_only", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_STEP14_PROFILES:=no_scale_bias one_scale_medium no_seeded_queries no_subjet_transformer no_particle_readback late_fusion cls_fusion cross_attention_branch_fusion few_subjets many_subjets physics_bias_removed large_hlt_part_control two_hlt_part_ensemble_control", submitter)
        self.assertIn("MULTISCALE_SUBJET_PART_QCD_HGG_INCLUDE_STEP14_ABLATIONS:=0", submitter)
        self.assertIn('export HLT_DEGRADATION_STRENGTH="${MULTISCALE_SUBJET_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"', submitter)
        self.assertIn("run_build_label_filtered_fresh_splits.sh", submitter)
        self.assertIn("run_build_label_filtered_hlt_cache.sh", submitter)
        self.assertIn("run_train_multiscale_subjet_part_tagger.sh", submitter)
        self.assertIn("run_write_multiscale_subjet_part_report.sh", submitter)
        self.assertIn("multiscale_subjet_train: ${#train_job_ids[@]}", submitter)
        self.assertIn('--dependency="afterok:${train_dep}"', submitter)
        self.assertIn("final_report_json", submitter)

    def test_local_compression_part_submitter_reuses_hlt06_cache_and_queues_variants(self):
        train = self.read("run_train_local_compression_part.sh")
        report = self.read("run_write_local_compression_part_report.sh")
        submitter = self.read("submit_local_compression_part_qcd_hgg_hlt0p6_experiment.sh")

        self.assertIn("scripts/train_local_compression_part_tagger.py", train)
        self.assertIn("LOCAL_COMPRESSION_PART_BASELINE_CHECKPOINT:?", train)
        self.assertIn("--manifest-path", train)
        self.assertIn("--hlt-cache-dir", train)
        self.assertIn("--baseline-checkpoint", train)
        self.assertIn("--confirm-split-settings", train)
        self.assertIn("--confirm-final-test", train)
        self.assertIn("--selection-metric", train)
        self.assertIn("--expected-hlt-degradation-strength", train)
        self.assertIn("--random-grouping-seed", train)
        self.assertIn("--require-baseline-split-manifest-hash", train)
        self.assertIn("--allow-missing-baseline-split-manifest-hash", train)
        self.assertIn("diagnostics/init_logit_diff_vs_baseline.json", train)

        self.assertIn("scripts/write_local_compression_part_report.py", report)
        self.assertIn("local_compression_part_final_report.json", report)
        self.assertIn("metric_table.csv", report)
        self.assertIn("baseline_comparison.csv", report)
        self.assertIn("diagnostics.csv", report)
        self.assertIn("runtime_summary.csv", report)

        self.assertIn("QCD_vs_Hgg_local_compression_part", submitter)
        self.assertIn("LOCAL_COMPRESSION_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH:=0.6", submitter)
        self.assertIn("multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_500k_full_20260628_194154", submitter)
        self.assertIn("local_graph_part_step10_qcd_hgg_binary_hlt0p6_20260627_075757/taggers/hlt_part_baseline/best_model_val.pt", submitter)
        self.assertIn("LOCAL_COMPRESSION_PART_QCD_HGG_MODEL_TRAIN_SIZE:=500000", submitter)
        self.assertIn("LOCAL_COMPRESSION_PART_QCD_HGG_MODEL_VAL_SIZE:=150000", submitter)
        self.assertIn("LOCAL_COMPRESSION_PART_QCD_HGG_STACK_TRAIN_SIZE:=500000", submitter)
        self.assertIn("LOCAL_COMPRESSION_PART_QCD_HGG_STACK_VAL_SIZE:=150000", submitter)
        self.assertIn("LOCAL_COMPRESSION_PART_QCD_HGG_FINAL_TEST_SIZE:=500000", submitter)
        self.assertIn("LOCAL_COMPRESSION_PART_QCD_HGG_EPOCHS:=45", submitter)
        self.assertIn("LOCAL_COMPRESSION_PART_QCD_HGG_SELECTION_METRIC:=fpr_at_signal_eff_0p50", submitter)
        self.assertIn("LOCAL_COMPRESSION_PART_QCD_HGG_VARIANTS:=hlt_part_baseline_recheck lc_mlp_delta lc_local_compression_no_context lc_context_gated lc_context_delta_no_modalities lc_random_grouping", submitter)
        self.assertIn('export LOCAL_COMPRESSION_PART_BASELINE_CHECKPOINT="${LOCAL_COMPRESSION_PART_QCD_HGG_BASELINE_CHECKPOINT}"', submitter)
        self.assertIn('export LOCAL_COMPRESSION_PART_REPORT_COMPARISON_SPLIT="final_test"', submitter)
        self.assertIn("run_train_local_compression_part.sh", submitter)
        self.assertIn("run_write_local_compression_part_report.sh", submitter)
        self.assertIn("local_compression_train: ${#train_job_ids[@]}", submitter)
        self.assertIn('--dependency="afterok:${train_dep}"', submitter)
        self.assertIn("final_report_json", submitter)

    def test_local_graph_score_fusion_runner_uses_frozen_step10_outputs(self):
        runner = self.read("run_local_graph_score_fusion.sh")

        self.assertIn("scripts/run_local_graph_score_fusion.py", runner)
        self.assertIn("LOCAL_GRAPH_PART_ROOT:=${OUTPUT_ROOT}/local_graph_part_step10_qcd_hgg_binary_hlt0p6", runner)
        self.assertIn("LOCAL_GRAPH_SCORE_FUSION_VARIANTS:=hlt_part_baseline local_edgeconv_adapter local_point_attention_adapter local_point_attention_adapter_warmstart", runner)
        self.assertIn("LOCAL_GRAPH_SCORE_FUSION_BASELINE_VARIANT:=hlt_part_baseline", runner)
        self.assertIn("LOCAL_GRAPH_SCORE_FUSION_PRIMARY_METRIC:=fpr_at_signal_eff_0p50", runner)
        self.assertIn("LOCAL_GRAPH_SCORE_FUSION_MAX_STACK_JETS:=150000", runner)
        self.assertIn("LOCAL_GRAPH_SCORE_FUSION_MAX_FINAL_TEST_JETS:=500000", runner)
        self.assertIn("LOCAL_GRAPH_SCORE_FUSION_CONFIRM_FINAL_TEST:=1", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn("best_model_val.pt", runner)
        self.assertIn("fusion_report.json", runner)
        self.assertIn("fusion_metric_table.csv", runner)
        self.assertIn("fusion_prediction_manifest.json", runner)

    def test_local_graph_residual_expert_runner_trains_one_loss_mode(self):
        runner = self.read("run_train_local_graph_residual_expert.sh")

        self.assertIn("scripts/train_local_graph_residual_expert.py", runner)
        self.assertIn("REQUESTED_LOSS_MODE", runner)
        self.assertIn("normalize_loss_mode", runner)
        self.assertIn("residual_weighted_bce", runner)
        self.assertIn("residual_boundary_pairwise", runner)
        self.assertIn("residual_boundary_pairwise_bce_anchor", runner)
        self.assertIn("residual_boundary_pairwise_soft_fpr_bce_anchor", runner)
        self.assertIn("residual_boundary_pairwise_soft_fpr_bce_anchor_alpha_shrink", runner)
        self.assertIn("not a training job", runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6", runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC:=fpr_at_signal_eff_0p50", runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_WARM_START_ENABLED:=1", runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_REQUIRE_WARM_START:=1", runner)
        self.assertIn("--warm-start-checkpoint", runner)
        self.assertIn("--require-warm-start", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn("--train-split model_train", runner)
        self.assertIn("--val-split model_val", runner)
        self.assertIn("${split}_baseline_logits.npz", runner)
        self.assertIn("${split}_baseline_logits_metadata.json", runner)
        self.assertIn("baseline_logit_manifest.json", runner)
        self.assertIn("diagnostics/residual_diagnostics_model_val.json", runner)
        self.assertNotIn("stack_train", runner)
        self.assertNotIn("stack_val", runner)
        self.assertNotIn("--final-test-split", runner)

    def test_local_graph_baseline_logit_cache_runner_writes_all_split_logits(self):
        runner = self.read("run_cache_local_graph_baseline_logits.sh")

        self.assertIn("scripts/cache_local_graph_baseline_logits.py", runner)
        self.assertIn("LOCAL_GRAPH_BASELINE_LOGIT_CACHE_SPLITS:=model_train model_val stack_train stack_val final_test", runner)
        self.assertIn("LOCAL_GRAPH_BASELINE_LOGIT_CACHE_METRIC_SPLITS:=model_train model_val stack_train stack_val", runner)
        self.assertIn("LOCAL_GRAPH_BASELINE_LOGIT_CACHE_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6", runner)
        self.assertIn("LOCAL_GRAPH_BASELINE_LOGIT_CACHE_EXPECTED_CHECKPOINT_VARIANT:=hlt_part_baseline", runner)
        self.assertIn("--checkpoint", runner)
        self.assertIn("--expected-checkpoint-variant", runner)
        self.assertIn("--splits", runner)
        self.assertIn("--metric-splits", runner)
        self.assertIn("--max-model-train-jets", runner)
        self.assertIn("--max-model-val-jets", runner)
        self.assertIn("--max-stack-train-jets", runner)
        self.assertIn("--max-stack-val-jets", runner)
        self.assertIn("--max-final-test-jets", runner)
        self.assertIn("baseline_logit_manifest.json", runner)
        self.assertIn("${split}_baseline_logits.npz", runner)
        self.assertIn("${split}_baseline_logits_metadata.json", runner)

    def test_local_graph_residual_expert_submitter_queues_baseline_logits_and_ladder(self):
        submitter = self.read("submit_local_graph_residual_expert_experiment.sh")
        report_runner = self.read("run_write_local_graph_residual_expert_report.sh")

        self.assertIn("local_graph_residual_expert_qcd_hgg_binary_hlt", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_HLT_DEGRADATION_STRENGTH:=0.6", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_LOSS_MODES:=A B C D", submitter)
        self.assertIn("alpha_shrinkage=reported_as_model_val_gamma_shrunk_rows", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_LOCAL_ADAPTER:=point_attention", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_TRAIN_BASELINE:=1", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_BUILD_BINARY_INPUTS:=1", submitter)
        self.assertIn("run_build_label_filtered_fresh_splits.sh", submitter)
        self.assertIn("run_build_label_filtered_hlt_cache.sh", submitter)
        self.assertIn("run_train_local_graph_part_tagger.sh", submitter)
        self.assertIn("run_cache_local_graph_baseline_logits.sh", submitter)
        self.assertIn("run_train_local_graph_residual_expert.sh", submitter)
        self.assertIn("run_write_local_graph_residual_expert_report.sh", submitter)
        self.assertIn("localgraph_residual_baseline_logits", submitter)
        self.assertIn("localgraph_residual_hlt_baseline", submitter)
        self.assertIn("localgraph_residual_final_report", submitter)
        self.assertIn("baseline_logit_cache_jid", submitter)
        self.assertIn("residual_job_ids", submitter)
        self.assertIn("residual_output_names", submitter)
        self.assertIn("afterok_args", submitter)
        self.assertIn('"${baseline_logit_cache_jid}"', submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_SUBMIT_FINAL_REPORT:=1", submitter)
        self.assertIn("afterok:${residual_dep}", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_SUBMIT_SCORE_FUSION:=0", submitter)
        self.assertIn("filtered_hlt_cache", submitter)
        self.assertIn("baseline_logit_cache", submitter)
        self.assertIn("residual_expert_root", submitter)
        self.assertIn("final_report", submitter)
        self.assertIn("scripts/write_local_graph_residual_expert_report.py", report_runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_REPORT_PRIMARY_METRIC:=fpr_at_signal_eff_0p50", report_runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_REPORT_ALLOW_PRECOMPUTED_EVALUATIONS:=0", report_runner)
        self.assertIn("--allow-precomputed-evaluations", report_runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_REPORT_CONFIRM_FINAL_TEST:=1", report_runner)
        self.assertIn("local_graph_residual_expert_report.json", report_runner)

    def test_local_graph_residual_v2_submitter_reuses_existing_inputs_and_queues_a_c_d(self):
        cache_runner = self.read("run_cache_local_graph_residual_v2_embeddings.sh")
        train_runner = self.read("run_train_local_graph_residual_expert_v2.sh")
        report_runner = self.read("run_write_local_graph_residual_expert_v2_report.sh")
        submitter = self.read("submit_local_graph_residual_expert_v2_experiment.sh")

        self.assertIn("scripts/cache_local_graph_residual_v2_embeddings.py", cache_runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_CACHE_SPLITS:=model_train model_val stack_train stack_val final_test", cache_runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_CACHE_METRIC_SPLITS:=model_train model_val stack_train stack_val", cache_runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT", cache_runner)
        self.assertIn("--checkpoint", cache_runner)
        self.assertIn("baseline_embedding_manifest.json", cache_runner)
        self.assertIn("${split}_baseline_embedding_cache.npz", cache_runner)
        self.assertIn("${split}_baseline_embedding_cache_metadata.json", cache_runner)

        self.assertIn("scripts/train_local_graph_residual_expert_v2.py", train_runner)
        self.assertIn("normalize_v2_loss_mode", train_runner)
        self.assertIn("residual_v2_weighted_bce", train_runner)
        self.assertIn("residual_v2_boundary_pairwise_bce_anchor", train_runner)
        self.assertIn("residual_v2_boundary_pairwise_soft_fpr_bce_anchor", train_runner)
        self.assertIn("not a training job", train_runner)
        self.assertIn("--baseline-embedding-cache-dir", train_runner)
        self.assertIn("--residual-input-mode", train_runner)
        self.assertIn("--condition-control-mode", train_runner)
        self.assertIn("--label-control-mode", train_runner)
        self.assertIn("--confirm-split-settings", train_runner)
        self.assertIn("--train-split model_train", train_runner)
        self.assertIn("--val-split model_val", train_runner)
        self.assertIn("diagnostics/model_val_learned_gamma_predictions.npz", train_runner)
        self.assertNotIn("--final-test-split", train_runner)

        self.assertIn("scripts/write_local_graph_residual_expert_v2_report.py", report_runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_REPORT_COMPARISON_SPLIT:=final_test", report_runner)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONFIRM_FINAL_TEST:=1", report_runner)
        self.assertIn("--hlt-cache-dir", report_runner)
        self.assertIn("--baseline-embedding-cache-dir", report_runner)
        self.assertIn("${split}_baseline_embedding_cache.npz", report_runner)
        self.assertIn("--confirm-final-test", report_runner)
        self.assertIn("local_graph_residual_expert_v2_report.json", report_runner)
        self.assertIn("metric_table.csv", report_runner)

        self.assertIn("local_graph_residual_expert_v2_qcd_hgg_binary_hlt", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES:=A C D", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_REPORT_COMPARISON_SPLIT:=final_test", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONFIRM_FINAL_TEST:=1", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_INPUT_MODE:=full", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODE:=normal", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODE:=normal", submitter)
        self.assertNotIn("LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES:=A B C D", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT must point", submitter)
        self.assertIn("run_cache_local_graph_residual_v2_embeddings.sh", submitter)
        self.assertIn("run_train_local_graph_residual_expert_v2.sh", submitter)
        self.assertIn("run_write_local_graph_residual_expert_v2_report.sh", submitter)
        self.assertIn("localgraph_residual_v2_embeddings", submitter)
        self.assertIn("localgraph_residual_v2_report", submitter)
        self.assertIn("gamma_shrinkage=reported_as_model_val_validation_shrunk_rows", submitter)
        self.assertIn("afterok:${residual_dep}", submitter)
        self.assertNotIn("run_build_label_filtered_fresh_splits.sh", submitter)
        self.assertNotIn("run_train_local_graph_part_tagger.sh", submitter)

    def test_local_graph_residual_v2_3m1m1m_serious_submitter_uses_existing_cache(self):
        submitter = self.read("submit_local_graph_residual_expert_v2_3m1m1m_serious.sh")

        self.assertIn("multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_3m1m1m_full_20260628_194154", submitter)
        self.assertIn("local_graph_part_qcd_hgg_hlt0p6_3m1m1m_20260629_015555", submitter)
        self.assertIn("taggers/hlt_part_baseline/best_model_val.pt", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE:=3000000", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE:=1000000", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE:=3000000", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE:=1000000", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE:=1000000", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES:=A C D", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH:=0.6", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_TRAIN_TIME:=5-00:00:00", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_CACHE_TIME:=1-12:00:00", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_REPORT_TIME:=1-00:00:00", submitter)
        self.assertIn("submit_local_graph_residual_expert_v2_experiment.sh", submitter)
        self.assertIn("fresh_require_file", submitter)
        self.assertIn("fresh_is_dry_run", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_ALLOW_BASELINE_SPLIT_MISMATCH:=0", submitter)
        self.assertIn("does not look like a 3M/1M/1M baseline", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH", submitter)
        self.assertIn("score_fusion_*/fusion_report.json", submitter)

    def test_local_graph_residual_v2_3m1m1m_ablation_suite_queues_controls(self):
        submitter = self.read("submit_local_graph_residual_expert_v2_3m1m1m_ablation_suite.sh")

        self.assertIn("multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_3m1m1m_full_20260628_194154", submitter)
        self.assertIn("local_graph_part_qcd_hgg_hlt0p6_3m1m1m_20260629_015555", submitter)
        self.assertIn("full_a|A|full|normal|normal", submitter)
        self.assertIn("full_c|C|full|normal|normal", submitter)
        self.assertIn("full_d|D|full|normal|normal", submitter)
        self.assertIn("embedding_only_d|D|embedding_only|normal|normal", submitter)
        self.assertIn("local_only_d|D|local_only|normal|normal", submitter)
        self.assertIn("condition_shuffled_d|D|full|shuffled|normal", submitter)
        self.assertIn("label_shuffled_d|D|full|normal|shuffled", submitter)
        self.assertIn("run_cache_local_graph_residual_v2_embeddings.sh", submitter)
        self.assertIn("run_train_local_graph_residual_expert_v2.sh", submitter)
        self.assertIn("run_write_local_graph_residual_expert_v2_report.sh", submitter)
        self.assertIn("LOCAL_GRAPH_RESIDUAL_V2_REPORT_VARIANTS", submitter)
        self.assertIn("total_submitted", submitter)
        self.assertIn("does not look like a 3M/1M/1M baseline", submitter)
        self.assertIn("score_fusion_*/fusion_report.json", submitter)

    def test_dualview_part_step10_smoke_runner_trains_real_and_shuffled_pn(self):
        runner = self.read("run_train_dualview_part_residual.sh")
        submitter = self.read("submit_dualview_part_residual_smoke_test.sh")
        trainer = (REPO_ROOT / "scripts" / "train_dualview_part_residual.py").read_text(encoding="utf-8")
        training = (REPO_ROOT / "teacher_logit_reco" / "dualview_part" / "training.py").read_text(encoding="utf-8")

        self.assertIn("scripts/train_dualview_part_residual.py", runner)
        self.assertIn("frozen_anchor_pn_residual", runner)
        self.assertIn("frozen_anchor_shuffled_pn_control", runner)
        self.assertIn("--confirm-final-test", runner)
        self.assertIn("--initialization-check-batches", runner)
        self.assertIn("DUALVIEW_PART_STACK_TRAIN_SIZE", runner)
        self.assertIn("DUALVIEW_PART_STACK_VAL_SIZE", runner)
        self.assertIn("DUALVIEW_PART_FINAL_TEST_SIZE", runner)
        self.assertIn("DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT", runner)
        self.assertIn("DUALVIEW_PART_HLT_CACHE_DIR", runner)
        self.assertIn("DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR", runner)
        self.assertIn("${split}_fixed_hlt.npz", runner)
        self.assertIn("${split}_reconstructed_view.npz", runner)
        self.assertIn("residual_diagnostics.json", runner)
        self.assertIn("gate_by_hlt_correctness.csv", runner)
        self.assertIn("fix_break_cases.csv", runner)
        self.assertIn("fix_cases.csv", runner)
        self.assertIn("break_cases.csv", runner)
        self.assertIn("--max-case-rows-per-type", runner)
        self.assertIn("--shuffle-pn-view", runner)

        report_runner = self.read("run_write_dualview_part_report.sh")
        self.assertIn("scripts/write_dualview_part_report.py", report_runner)
        self.assertIn("DUALVIEW_PART_REPORT_REQUIRE_REAL_BEATS_SHUFFLED", report_runner)
        self.assertIn("dualview_part_report.json", report_runner)
        self.assertIn("metric_table.csv", report_runner)

        self.assertIn("dualview_part_residual_smoke_", submitter)
        self.assertIn("DUALVIEW_PART_SMOKE_STACK_TRAIN_SIZE:-10000", submitter)
        self.assertIn("DUALVIEW_PART_SMOKE_STACK_VAL_SIZE:-5000", submitter)
        self.assertIn("DUALVIEW_PART_SMOKE_FINAL_TEST_SIZE:-10000", submitter)
        self.assertIn("DUALVIEW_PART_SMOKE_EPOCHS:-2", submitter)
        self.assertIn("DUALVIEW_PART_SMOKE_MAX_CASE_ROWS_PER_TYPE:-200", submitter)
        self.assertIn("DUALVIEW_PART_SMOKE_VARIANTS:=frozen_anchor_pn_residual frozen_anchor_shuffled_pn_control", submitter)
        self.assertIn("run_train_dualview_part_residual.sh", submitter)
        self.assertIn("run_write_dualview_part_report.sh", submitter)
        self.assertIn("shuffled_pn_control: submitted", submitter)
        self.assertIn("require_real_beats_shuffled", submitter)
        self.assertIn("afterok_args", submitter)

        self.assertIn("--skip-initialization-check", trainer)
        self.assertIn("--initialization-check-batches", trainer)
        self.assertIn("--max-case-rows-per-type", trainer)
        self.assertIn("run_initialization_check", training)
        self.assertIn("prediction_change_fraction", training)
        self.assertIn("initialization_check_passed", training)

    def test_dualview_part_step11_500k_submitter(self):
        submitter = self.read("submit_dualview_part_residual_500k_qcd_hgg.sh")

        self.assertIn("dualview_part_qcd_hgg_binary_hlt0p6_true500k_", submitter)
        self.assertIn("DUALVIEW_PART_500K_STACK_TRAIN_SIZE:=500000", submitter)
        self.assertIn("DUALVIEW_PART_500K_STACK_VAL_SIZE:=150000", submitter)
        self.assertIn("DUALVIEW_PART_500K_FINAL_TEST_SIZE:=500000", submitter)
        self.assertIn("DUALVIEW_PART_500K_EPOCHS:=45", submitter)
        self.assertIn("DUALVIEW_PART_500K_SELECTION_METRIC:=fpr_at_signal_eff_0p50", submitter)
        self.assertIn("DUALVIEW_PART_500K_VARIANTS:=frozen_anchor_pn_residual frozen_anchor_shuffled_pn_control", submitter)
        self.assertIn("DUALVIEW_PART_500K_REQUIRE_REAL_BEATS_SHUFFLED:-1", submitter)
        self.assertIn("run_train_dualview_part_residual.sh", submitter)
        self.assertIn("run_write_dualview_part_report.sh", submitter)
        self.assertIn("afterok_args", submitter)
        self.assertIn("expected_jobs:", submitter)


if __name__ == "__main__":
    unittest.main()

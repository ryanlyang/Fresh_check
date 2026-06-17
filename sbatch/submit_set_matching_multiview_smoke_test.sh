#!/usr/bin/env bash
# Submit a tiny set-matching multi-view smoke test for pipeline correctness only.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${TAGGER_UPSTREAM_DEPENDENCY:=}"
: "${SET_MATCHING_SMOKE_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${SET_MATCHING_SMOKE_ROOT:=${OUTPUT_ROOT}/set_matching_multiview_smoke_${SET_MATCHING_SMOKE_TAG}}"

export SET_MATCHING_ROOT="${SET_MATCHING_SMOKE_ROOT}"
export SET_MATCHING_RECONSTRUCTOR_DIR="${SET_MATCHING_ROOT}/reconstructors"
export SET_MATCHING_RECONSTRUCTED_VIEW_DIR="${SET_MATCHING_ROOT}/reconstructed_views"
export SET_MATCHING_TAGGER_ROOT="${SET_MATCHING_ROOT}/taggers"
export SET_MATCHING_ABLATION_DIR="${SET_MATCHING_ROOT}/ablations/five_view_ablation_eval"
export SET_MATCHING_FINAL_REPORT_DIR="${SET_MATCHING_ROOT}/final_report"

export SET_MATCHING_MODEL_TRAIN_SIZE="${SET_MATCHING_SMOKE_MODEL_TRAIN_SIZE:-10000}"
export SET_MATCHING_MODEL_VAL_SIZE="${SET_MATCHING_SMOKE_MODEL_VAL_SIZE:-2000}"
export SET_MATCHING_STACK_TRAIN_SIZE="${SET_MATCHING_SMOKE_STACK_TRAIN_SIZE:-5000}"
export SET_MATCHING_STACK_VAL_SIZE="${SET_MATCHING_SMOKE_STACK_VAL_SIZE:-2000}"
export SET_MATCHING_FINAL_TEST_SIZE="${SET_MATCHING_SMOKE_FINAL_TEST_SIZE:-10000}"

export SET_MATCHING_RECO_EPOCHS="${SET_MATCHING_SMOKE_RECO_EPOCHS:-2}"
export SET_MATCHING_RECO_EARLY_STOP_PATIENCE="${SET_MATCHING_SMOKE_RECO_EARLY_STOP_PATIENCE:-1}"
export SET_MATCHING_TAGGER_EPOCHS="${SET_MATCHING_SMOKE_TAGGER_EPOCHS:-2}"
export SET_MATCHING_TAGGER_EARLY_STOP_PATIENCE="${SET_MATCHING_SMOKE_TAGGER_EARLY_STOP_PATIENCE:-1}"
export SET_MATCHING_RECO_BATCH_SIZE="${SET_MATCHING_SMOKE_RECO_BATCH_SIZE:-64}"
export SET_MATCHING_CACHE_BATCH_SIZE="${SET_MATCHING_SMOKE_CACHE_BATCH_SIZE:-128}"
export SET_MATCHING_TAGGER_BATCH_SIZE="${SET_MATCHING_SMOKE_TAGGER_BATCH_SIZE:-32}"
export SET_MATCHING_EVAL_BATCH_SIZE="${SET_MATCHING_SMOKE_EVAL_BATCH_SIZE:-128}"

export SET_MATCHING_CACHE_MAX_JETS_PER_SPLIT="${SET_MATCHING_SMOKE_CACHE_MAX_JETS_PER_SPLIT:-${SET_MATCHING_FINAL_TEST_SIZE}}"
export SET_MATCHING_CONFIRM_FINAL_TEST=1
export SET_MATCHING_EVAL_REQUIRE_ALL_CANONICAL=1

fresh_prepare_submitter
if fresh_is_dry_run; then
  echo "set_matching_smoke_dry_run=1"
fi

fresh_refuse_existing_dir "${SET_MATCHING_SMOKE_ROOT}"
fresh_claim_new_dir "${SET_MATCHING_SMOKE_ROOT}/.smoke_submission_lock"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "warning=smoke metrics are for pipeline correctness only, not physics interpretation"
    echo "smoke_root=${SET_MATCHING_SMOKE_ROOT}"
    echo "manifest=${SET_MATCHING_MANIFEST_PATH}"
    echo "hlt_cache_dir=${SET_MATCHING_HLT_CACHE_DIR}"
    echo "model_train_size=${SET_MATCHING_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${SET_MATCHING_MODEL_VAL_SIZE}"
    echo "stack_train_size=${SET_MATCHING_STACK_TRAIN_SIZE}"
    echo "stack_val_size=${SET_MATCHING_STACK_VAL_SIZE}"
    echo "final_test_size=${SET_MATCHING_FINAL_TEST_SIZE}"
    echo "reco_epochs=${SET_MATCHING_RECO_EPOCHS}"
    echo "tagger_epochs=${SET_MATCHING_TAGGER_EPOCHS}"
    echo "cache_max_jets_per_split=${SET_MATCHING_CACHE_MAX_JETS_PER_SPLIT}"
    echo "confirm_final_test=${SET_MATCHING_CONFIRM_FINAL_TEST}"
    echo "upstream_dependency=${UPSTREAM_DEPENDENCY:-none}"
    echo "tagger_upstream_dependency=${TAGGER_UPSTREAM_DEPENDENCY:-none}"
  } > "${SET_MATCHING_SMOKE_ROOT}/.smoke_submission_lock/metadata.txt"
fi

cat <<SUMMARY
set_matching_multiview_smoke_submission:
  warning: smoke metrics are for pipeline correctness only, not physics interpretation
  smoke_root: ${SET_MATCHING_SMOKE_ROOT}
  smoke_sizes:
    model_train: ${SET_MATCHING_MODEL_TRAIN_SIZE}
    model_val: ${SET_MATCHING_MODEL_VAL_SIZE}
    stack_train: ${SET_MATCHING_STACK_TRAIN_SIZE}
    stack_val: ${SET_MATCHING_STACK_VAL_SIZE}
    final_test: ${SET_MATCHING_FINAL_TEST_SIZE}
  smoke_training:
    reco_epochs: ${SET_MATCHING_RECO_EPOCHS}
    tagger_epochs: ${SET_MATCHING_TAGGER_EPOCHS}
    cache_max_jets_per_split: ${SET_MATCHING_CACHE_MAX_JETS_PER_SPLIT}
  output_dirs:
    root: ${SET_MATCHING_ROOT}
    reconstructors: ${SET_MATCHING_RECONSTRUCTOR_DIR}
    reconstructed_views: ${SET_MATCHING_RECONSTRUCTED_VIEW_DIR}
    taggers: ${SET_MATCHING_TAGGER_ROOT}
    ablations: ${SET_MATCHING_ABLATION_DIR}
    final_report: ${SET_MATCHING_FINAL_REPORT_DIR}
SUMMARY

# The nested submitter performs the sbatch submissions and afterok wiring.
bash "${SCRIPT_DIR}/submit_set_matching_multiview_experiment.sh"

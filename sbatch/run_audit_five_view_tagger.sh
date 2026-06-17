#!/usr/bin/env bash
# Evaluate the trained five-view taggers and ablation controls.

#SBATCH --job-name=setmatch_audit
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${SKIP_HLT_HASH_CHECK:=0}"

fresh_setup "$@"
fresh_require_file "scripts/evaluate_five_view_ablation.py"
fresh_split_words variant_args "${SET_MATCHING_TAGGER_VARIANTS}"
if fresh_bool_enabled "${SET_MATCHING_EVAL_REQUIRE_ALL_CANONICAL}"; then
  for variant in "${variant_args[@]}"; do
    fresh_require_file "${SET_MATCHING_TAGGER_ROOT}/${variant}/best_model_val.pt"
  done
fi
for split in stack_val final_test; do
  fresh_require_file "${SET_MATCHING_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_claim_new_dir "${SET_MATCHING_ABLATION_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/evaluate_five_view_ablation.py"
  --output-dir "${SET_MATCHING_ABLATION_DIR}"
  --experiment-dir "${SET_MATCHING_ROOT}"
  --hlt-cache-dir "${SET_MATCHING_HLT_CACHE_DIR}"
  --tagger-root "${SET_MATCHING_TAGGER_ROOT}"
  --reconstructed-view-dir "${SET_MATCHING_RECONSTRUCTED_VIEW_DIR}"
  --batch-size "${SET_MATCHING_EVAL_BATCH_SIZE}"
  --num-workers "${SET_MATCHING_EVAL_NUM_WORKERS}"
  --device "${SET_MATCHING_EVAL_DEVICE}"
  --max-val-jets "${SET_MATCHING_STACK_VAL_SIZE}"
  --max-final-test-jets "${SET_MATCHING_FINAL_TEST_SIZE}"
  --max-tokens-per-view "${SET_MATCHING_MAX_TOKENS_PER_VIEW}"
  --min-tokens-per-view "${SET_MATCHING_MIN_TOKENS_PER_VIEW}"
  --confidence-threshold "${SET_MATCHING_CONFIDENCE_THRESHOLD}"
  --selection-mode topk_or_threshold
  --seed "${SET_MATCHING_TAGGER_SEED}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${SET_MATCHING_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --require-all-canonical "${SET_MATCHING_EVAL_REQUIRE_ALL_CANONICAL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${SKIP_HLT_HASH_CHECK}"
fresh_append_optional_arg cmd --max-val-batches "${SET_MATCHING_EVAL_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${SET_MATCHING_EVAL_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${SET_MATCHING_ABLATION_DIR}" "set_matching_five_view_audit" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${SET_MATCHING_ABLATION_DIR}/summary.csv"
  fresh_require_file "${SET_MATCHING_ABLATION_DIR}/summary.json"
  fresh_require_file "${SET_MATCHING_ABLATION_DIR}/per_class_metrics.csv"
  fresh_require_file "${SET_MATCHING_ABLATION_DIR}/run_report.json"
fi

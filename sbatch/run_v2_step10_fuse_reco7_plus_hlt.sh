#!/usr/bin/env bash
# V2 Step 10: collect frozen predictions and fuse Reco7 + HLT baseline.

#SBATCH --job-name=v2_step10_fuse7
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=08:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${FUSION_BATCH_SIZE:=128}"
: "${FUSION_NUM_WORKERS:=4}"
: "${FUSION_DEVICE:=${DEVICE}}"
: "${FUSION_MAX_JETS_PER_SPLIT:=}"
: "${FUSION_FEATURE_MODE:=logits_probs}"
: "${FUSION_MAX_ITER:=500}"
: "${CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "${HLT_CACHE_DIR}/stack_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/stack_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${HLT_BASELINE_DIR}/best_model_val.pt"
fresh_split_words variant_args "${V2_STEP7_VARIANTS}"
for variant in "${variant_args[@]}"; do
  fresh_require_file "${V2_STEP7_RECO_ROOT}/${variant}/stage2_dual_view/best_model_val.pt"
done
fresh_refuse_existing_dir "${V2_STEP7_FUSION_DIR}"

cmd=(
  "${PYTHON_BIN}" "scripts/run_reco7_fusion.py"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --hlt-checkpoint "${HLT_BASELINE_DIR}/best_model_val.pt"
  --reco-root "${V2_STEP7_RECO_ROOT}"
  --output-dir "${V2_STEP7_FUSION_DIR}"
  --variants "${variant_args[@]}"
  --splits stack_train stack_val final_test
  --batch-size "${FUSION_BATCH_SIZE}"
  --num-workers "${FUSION_NUM_WORKERS}"
  --device "${FUSION_DEVICE}"
  --feature-mode "${FUSION_FEATURE_MODE}"
  --max-iter "${FUSION_MAX_ITER}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_optional_arg cmd --max-jets-per-split "${FUSION_MAX_JETS_PER_SPLIT}"

fresh_write_run_config "${V2_STEP7_FUSION_DIR}" "v2_step10_fuse_reco7_plus_hlt" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${V2_STEP7_FUSION_DIR}/fusion_report.json"
  fresh_require_file "${V2_STEP7_FUSION_DIR}/run_report.json"
  fresh_require_file "${V2_STEP7_FUSION_DIR}/stacked_logistic_regression.npz"
fi

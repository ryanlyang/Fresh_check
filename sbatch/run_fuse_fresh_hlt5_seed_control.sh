#!/usr/bin/env bash
#SBATCH --job-name=fresh_fuse_hlt5
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${FUSION_BATCH_SIZE:=128}"
: "${FUSION_NUM_WORKERS:=4}"
: "${FUSION_DEVICE:=auto}"
: "${FUSION_MAX_JETS_PER_SPLIT:=}"
: "${FUSION_FEATURE_MODE:=logits_probs}"
: "${FUSION_MAX_ITER:=500}"
: "${CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_split_words seed_args "${HLT5_SEEDS}"
for seed in "${seed_args[@]}"; do
  fresh_require_file "${HLT5_ROOT}/seed${seed}/best_model_val.pt"
done
fresh_refuse_existing_dir "${HLT5_FUSION_DIR}"

cmd=(
  "${PYTHON_BIN}" "scripts/run_hlt5_fusion.py"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --hlt-checkpoint-root "${HLT5_ROOT}"
  --output-dir "${HLT5_FUSION_DIR}"
  --seeds "${seed_args[@]}"
  --splits stack_train stack_val final_test
  --batch-size "${FUSION_BATCH_SIZE}"
  --num-workers "${FUSION_NUM_WORKERS}"
  --device "${FUSION_DEVICE}"
  --feature-mode "${FUSION_FEATURE_MODE}"
  --max-iter "${FUSION_MAX_ITER}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_optional_arg cmd --max-jets-per-split "${FUSION_MAX_JETS_PER_SPLIT}"

fresh_write_run_config "${HLT5_FUSION_DIR}" "fuse_hlt5_seed_control" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${HLT5_FUSION_DIR}/fusion_report.json"
  fresh_require_file "${HLT5_FUSION_DIR}/stacked_logistic_regression.npz"
fi

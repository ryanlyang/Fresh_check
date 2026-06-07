#!/usr/bin/env bash
#SBATCH --job-name=fresh_hlt_seed
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${MODEL_SIZE:=base}"
: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${MAX_TRAIN_BATCHES:=}"
: "${MAX_VAL_BATCHES:=}"
: "${MAX_TRAIN_JETS:=}"
: "${MAX_VAL_JETS:=}"

TRAIN_SEED="${1:?Usage: sbatch/run_train_fresh_hlt_seed.sh SEED}"
export TRAIN_SEED
OUTPUT_DIR="${HLT5_ROOT}/seed${TRAIN_SEED}"

fresh_setup "$@"
fresh_require_file "${HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_refuse_existing_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "scripts/train_hlt_baseline.py"
  --cache-dir "${HLT_CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${TRAIN_SEED}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --model-size "${MODEL_SIZE}"
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_optional_arg cmd --max-train-batches "${MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-train-jets "${MAX_TRAIN_JETS}"
fresh_append_optional_arg cmd --max-val-jets "${MAX_VAL_JETS}"

fresh_write_run_config "${OUTPUT_DIR}" "train_hlt_seed_${TRAIN_SEED}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
fi

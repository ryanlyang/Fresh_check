#!/usr/bin/env bash
#SBATCH --job-name=crossarch_hlt_train
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ARCHITECTURE="${1:?Usage: sbatch run_crossarch_train_hlt_baseline.sh <part|pn|pfn|pcnn>}"
OUTPUT_DIR="${CROSSARCH_HLT_BASELINE_DIR}/${ARCHITECTURE}"

: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${MAX_TRAIN_BATCHES:=}"
: "${MAX_VAL_BATCHES:=}"

fresh_setup "$@"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_crossarch_hlt_baseline.py"
  --architecture "${ARCHITECTURE}"
  --cache-dir "${CROSSARCH_HLT_CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${CROSSARCH_HLT_BASELINE_SEED}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --max-train-jets "${CROSSARCH_MODEL_TRAIN_SIZE}"
  --max-val-jets "${CROSSARCH_MODEL_VAL_SIZE}"
  --model-size "${CROSSARCH_HLT_BASELINE_MODEL_SIZE}"
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_optional_arg cmd --max-train-batches "${MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${MAX_VAL_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "crossarch_hlt_train_${ARCHITECTURE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/source_metadata.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
fi

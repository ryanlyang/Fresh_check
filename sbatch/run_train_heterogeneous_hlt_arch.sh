#!/usr/bin/env bash
# Train one architecture in the heterogeneous fixed-HLT ensemble.

#SBATCH --job-name=hhlt_train
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

ARCHITECTURE="${1:?Usage: sbatch run_train_heterogeneous_hlt_arch.sh <part|pn|pfn|pcnn>}"
OUTPUT_DIR="${HETERO_HLT4_MODEL_ROOT}/${ARCHITECTURE}"

fresh_setup "$@"
fresh_require_file "${HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_heterogeneous_hlt.py"
  --architecture "${ARCHITECTURE}"
  --cache-dir "${HLT_CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${TRAIN_SEED:-101}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --max-train-jets "${HETERO_HLT4_TRAIN_SIZE}"
  --max-val-jets "${HETERO_HLT4_VAL_SIZE}"
  --model-size "${MODEL_SIZE:-base}"
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP:-0}"

fresh_write_run_config "${OUTPUT_DIR}" "train_heterogeneous_hlt_${ARCHITECTURE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/heterogeneous_hlt_report.json"
fi

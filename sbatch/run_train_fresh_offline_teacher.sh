#!/usr/bin/env bash
#SBATCH --job-name=fresh_offline_teacher
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

: "${OFFLINE_TEACHER_SEED:=707}"
: "${MODEL_SIZE:=base}"
: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${READ_CHUNK_SIZE:=50000}"
: "${MAX_TRAIN_BATCHES:=}"
: "${MAX_VAL_BATCHES:=}"
: "${MAX_TRAIN_JETS:=}"
: "${MAX_VAL_JETS:=}"

fresh_setup "$@"
export OFFLINE_TEACHER_SEED
fresh_require_data_dir
fresh_require_file "${MANIFEST_PATH}"
fresh_refuse_existing_dir "${OFFLINE_TEACHER_DIR}"

cmd=(
  "${PYTHON_BIN}" "scripts/train_offline_teacher.py"
  --manifest "${MANIFEST_PATH}"
  --data-dir "${DATA_DIR}"
  --output-dir "${OFFLINE_TEACHER_DIR}"
  --seed "${OFFLINE_TEACHER_SEED}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --model-size "${MODEL_SIZE}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_optional_arg cmd --max-train-batches "${MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-train-jets "${MAX_TRAIN_JETS}"
fresh_append_optional_arg cmd --max-val-jets "${MAX_VAL_JETS}"

fresh_write_run_config "${OFFLINE_TEACHER_DIR}" "train_offline_teacher" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OFFLINE_TEACHER_DIR}/best_model_val.pt"
  fresh_require_file "${OFFLINE_TEACHER_DIR}/model_val_report.json"
  fresh_require_file "${OFFLINE_TEACHER_DIR}/config.json"
fi

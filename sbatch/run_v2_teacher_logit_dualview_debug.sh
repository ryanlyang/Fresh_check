#!/usr/bin/env bash
# Debug probe: V2 m2-hybrid reconstructor trained with teacher-logit loss,
# followed by a fresh parent-aligned dual-view tagger.

#SBATCH --job-name=v2_tlogdv
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${V2_TLOG_DUALVIEW_DEBUG_ROOT:=${OUTPUT_ROOT}/v2_teacher_logit_dualview_debug_$(date +%Y%m%d_%H%M%S)}"
: "${V2_TLOG_DUALVIEW_DEBUG_TEACHER_CHECKPOINT:=${OFFLINE_TEACHER_DIR}/best_model_val.pt}"
: "${V2_TLOG_DUALVIEW_DEBUG_TEACHER_ARCHITECTURE:=part}"
: "${V2_TLOG_DUALVIEW_DEBUG_VARIANT:=m2_base}"
: "${V2_TLOG_DUALVIEW_DEBUG_TRAIN_SIZE:=20000}"
: "${V2_TLOG_DUALVIEW_DEBUG_VAL_SIZE:=5000}"
: "${V2_TLOG_DUALVIEW_DEBUG_TEST_SIZE:=20000}"
: "${V2_TLOG_DUALVIEW_DEBUG_RECO_EPOCHS:=8}"
: "${V2_TLOG_DUALVIEW_DEBUG_DUAL_EPOCHS:=8}"
: "${V2_TLOG_DUALVIEW_DEBUG_BATCH_SIZE:=64}"
: "${V2_TLOG_DUALVIEW_DEBUG_DUAL_BATCH_SIZE:=64}"
: "${V2_TLOG_DUALVIEW_DEBUG_EVAL_BATCH_SIZE:=128}"
: "${V2_TLOG_DUALVIEW_DEBUG_RECO_LR:=0.0003}"
: "${V2_TLOG_DUALVIEW_DEBUG_DUAL_LR:=0.001}"
: "${V2_TLOG_DUALVIEW_DEBUG_WEIGHT_DECAY:=0.0001}"
: "${V2_TLOG_DUALVIEW_DEBUG_RECO_PATIENCE:=3}"
: "${V2_TLOG_DUALVIEW_DEBUG_DUAL_PATIENCE:=3}"
: "${V2_TLOG_DUALVIEW_DEBUG_MODEL_SIZE:=base}"
: "${V2_TLOG_DUALVIEW_DEBUG_MAX_CONSTITS:=128}"
: "${V2_TLOG_DUALVIEW_DEBUG_NO_AMP:=0}"
: "${V2_TLOG_DUALVIEW_DEBUG_MAX_TRAIN_BATCHES:=}"
: "${V2_TLOG_DUALVIEW_DEBUG_MAX_VAL_BATCHES:=}"
: "${V2_TLOG_DUALVIEW_DEBUG_MAX_EVAL_BATCHES:=}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "${MANIFEST_PATH}"
fresh_require_file "${HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${V2_TLOG_DUALVIEW_DEBUG_TEACHER_CHECKPOINT}"
fresh_claim_new_dir "${V2_TLOG_DUALVIEW_DEBUG_ROOT}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_v2_teacher_logit_dualview_debug.py"
  --output-dir "${V2_TLOG_DUALVIEW_DEBUG_ROOT}"
  --manifest-path "${MANIFEST_PATH}"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --teacher-checkpoint "${V2_TLOG_DUALVIEW_DEBUG_TEACHER_CHECKPOINT}"
  --teacher-architecture "${V2_TLOG_DUALVIEW_DEBUG_TEACHER_ARCHITECTURE}"
  --variant "${V2_TLOG_DUALVIEW_DEBUG_VARIANT}"
  --train-size "${V2_TLOG_DUALVIEW_DEBUG_TRAIN_SIZE}"
  --val-size "${V2_TLOG_DUALVIEW_DEBUG_VAL_SIZE}"
  --test-size "${V2_TLOG_DUALVIEW_DEBUG_TEST_SIZE}"
  --seed "${TRAIN_SEED:-1407}"
  --reco-epochs "${V2_TLOG_DUALVIEW_DEBUG_RECO_EPOCHS}"
  --dual-epochs "${V2_TLOG_DUALVIEW_DEBUG_DUAL_EPOCHS}"
  --batch-size "${V2_TLOG_DUALVIEW_DEBUG_BATCH_SIZE}"
  --dual-batch-size "${V2_TLOG_DUALVIEW_DEBUG_DUAL_BATCH_SIZE}"
  --eval-batch-size "${V2_TLOG_DUALVIEW_DEBUG_EVAL_BATCH_SIZE}"
  --lr "${V2_TLOG_DUALVIEW_DEBUG_RECO_LR}"
  --dual-lr "${V2_TLOG_DUALVIEW_DEBUG_DUAL_LR}"
  --weight-decay "${V2_TLOG_DUALVIEW_DEBUG_WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --reco-early-stop-patience "${V2_TLOG_DUALVIEW_DEBUG_RECO_PATIENCE}"
  --dual-early-stop-patience "${V2_TLOG_DUALVIEW_DEBUG_DUAL_PATIENCE}"
  --model-size "${V2_TLOG_DUALVIEW_DEBUG_MODEL_SIZE}"
  --max-constits "${V2_TLOG_DUALVIEW_DEBUG_MAX_CONSTITS}"
)
fresh_append_flag_if_enabled cmd --no-amp "${V2_TLOG_DUALVIEW_DEBUG_NO_AMP}"
fresh_append_optional_arg cmd --max-train-batches "${V2_TLOG_DUALVIEW_DEBUG_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${V2_TLOG_DUALVIEW_DEBUG_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-eval-batches "${V2_TLOG_DUALVIEW_DEBUG_MAX_EVAL_BATCHES}"

fresh_write_run_config "${V2_TLOG_DUALVIEW_DEBUG_ROOT}" "v2_teacher_logit_dualview_debug" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${V2_TLOG_DUALVIEW_DEBUG_ROOT}/stage_a/best_model_val.pt"
  fresh_require_file "${V2_TLOG_DUALVIEW_DEBUG_ROOT}/stage_a/model_val_reconstruction_report.json"
  fresh_require_file "${V2_TLOG_DUALVIEW_DEBUG_ROOT}/stage2_dual_view/best_model_val.pt"
  fresh_require_file "${V2_TLOG_DUALVIEW_DEBUG_ROOT}/stage2_dual_view/model_val_report.json"
  fresh_require_file "${V2_TLOG_DUALVIEW_DEBUG_ROOT}/evaluation_report.json"
  fresh_require_file "${V2_TLOG_DUALVIEW_DEBUG_ROOT}/run_report.json"
fi

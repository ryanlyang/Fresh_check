#!/usr/bin/env bash
# V2 m2-hybrid teacher-logit + dual-view probe restricted to QCD vs Hbb.

#SBATCH --job-name=v2_tlog_hbbqcd
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

: "${V2_TLOG_HBB_QCD_ROOT:=${OUTPUT_ROOT}/v2_teacher_logit_dualview_hbb_qcd_$(date +%Y%m%d_%H%M%S)}"
: "${V2_TLOG_HBB_QCD_TEACHER_CHECKPOINT:=${OFFLINE_TEACHER_DIR}/best_model_val.pt}"
: "${V2_TLOG_HBB_QCD_TEACHER_ARCHITECTURE:=part}"
: "${V2_TLOG_HBB_QCD_VARIANT:=m2_base}"

# These are the QCD+Hbb portions of the normal 500k/150k/500k ten-class split.
: "${V2_TLOG_HBB_QCD_TRAIN_SIZE:=100000}"
: "${V2_TLOG_HBB_QCD_VAL_SIZE:=30000}"
: "${V2_TLOG_HBB_QCD_TEST_SIZE:=100000}"

: "${V2_TLOG_HBB_QCD_RECO_EPOCHS:=8}"
: "${V2_TLOG_HBB_QCD_DUAL_EPOCHS:=8}"
: "${V2_TLOG_HBB_QCD_BATCH_SIZE:=64}"
: "${V2_TLOG_HBB_QCD_DUAL_BATCH_SIZE:=64}"
: "${V2_TLOG_HBB_QCD_EVAL_BATCH_SIZE:=128}"
: "${V2_TLOG_HBB_QCD_RECO_LR:=0.0003}"
: "${V2_TLOG_HBB_QCD_DUAL_LR:=0.001}"
: "${V2_TLOG_HBB_QCD_WEIGHT_DECAY:=0.0001}"
: "${V2_TLOG_HBB_QCD_RECO_PATIENCE:=3}"
: "${V2_TLOG_HBB_QCD_DUAL_PATIENCE:=3}"
: "${V2_TLOG_HBB_QCD_MODEL_SIZE:=base}"
: "${V2_TLOG_HBB_QCD_MAX_CONSTITS:=128}"
: "${V2_TLOG_HBB_QCD_NO_AMP:=0}"
: "${V2_TLOG_HBB_QCD_MAX_TRAIN_BATCHES:=}"
: "${V2_TLOG_HBB_QCD_MAX_VAL_BATCHES:=}"
: "${V2_TLOG_HBB_QCD_MAX_EVAL_BATCHES:=}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "${MANIFEST_PATH}"
fresh_require_file "${HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${V2_TLOG_HBB_QCD_TEACHER_CHECKPOINT}"
fresh_claim_new_dir "${V2_TLOG_HBB_QCD_ROOT}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_v2_teacher_logit_dualview_debug.py"
  --output-dir "${V2_TLOG_HBB_QCD_ROOT}"
  --manifest-path "${MANIFEST_PATH}"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --teacher-checkpoint "${V2_TLOG_HBB_QCD_TEACHER_CHECKPOINT}"
  --teacher-architecture "${V2_TLOG_HBB_QCD_TEACHER_ARCHITECTURE}"
  --variant "${V2_TLOG_HBB_QCD_VARIANT}"
  --label-filter-names QCD Hbb
  --train-size "${V2_TLOG_HBB_QCD_TRAIN_SIZE}"
  --val-size "${V2_TLOG_HBB_QCD_VAL_SIZE}"
  --test-size "${V2_TLOG_HBB_QCD_TEST_SIZE}"
  --seed "${TRAIN_SEED:-1407}"
  --reco-epochs "${V2_TLOG_HBB_QCD_RECO_EPOCHS}"
  --dual-epochs "${V2_TLOG_HBB_QCD_DUAL_EPOCHS}"
  --batch-size "${V2_TLOG_HBB_QCD_BATCH_SIZE}"
  --dual-batch-size "${V2_TLOG_HBB_QCD_DUAL_BATCH_SIZE}"
  --eval-batch-size "${V2_TLOG_HBB_QCD_EVAL_BATCH_SIZE}"
  --lr "${V2_TLOG_HBB_QCD_RECO_LR}"
  --dual-lr "${V2_TLOG_HBB_QCD_DUAL_LR}"
  --weight-decay "${V2_TLOG_HBB_QCD_WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --reco-early-stop-patience "${V2_TLOG_HBB_QCD_RECO_PATIENCE}"
  --dual-early-stop-patience "${V2_TLOG_HBB_QCD_DUAL_PATIENCE}"
  --model-size "${V2_TLOG_HBB_QCD_MODEL_SIZE}"
  --max-constits "${V2_TLOG_HBB_QCD_MAX_CONSTITS}"
)
fresh_append_flag_if_enabled cmd --no-amp "${V2_TLOG_HBB_QCD_NO_AMP}"
fresh_append_optional_arg cmd --max-train-batches "${V2_TLOG_HBB_QCD_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${V2_TLOG_HBB_QCD_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-eval-batches "${V2_TLOG_HBB_QCD_MAX_EVAL_BATCHES}"

fresh_write_run_config "${V2_TLOG_HBB_QCD_ROOT}" "v2_teacher_logit_dualview_hbb_qcd" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${V2_TLOG_HBB_QCD_ROOT}/stage_a/best_model_val.pt"
  fresh_require_file "${V2_TLOG_HBB_QCD_ROOT}/stage_a/model_val_reconstruction_report.json"
  fresh_require_file "${V2_TLOG_HBB_QCD_ROOT}/stage2_dual_view/best_model_val.pt"
  fresh_require_file "${V2_TLOG_HBB_QCD_ROOT}/stage2_dual_view/model_val_report.json"
  fresh_require_file "${V2_TLOG_HBB_QCD_ROOT}/evaluation_report.json"
  fresh_require_file "${V2_TLOG_HBB_QCD_ROOT}/run_report.json"
fi

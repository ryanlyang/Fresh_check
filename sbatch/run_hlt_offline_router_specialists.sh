#!/usr/bin/env bash
# Train HLT-only agreement/disagreement specialists routed by HLT/offline-on-HLT probes.

#SBATCH --job-name=hlt_route_spec
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${ROUTER_SPECIALIST_ROOT:=${OUTPUT_ROOT}/hlt_offline_router_specialists_$(date +%Y%m%d_%H%M%S)}"
: "${ROUTER_SPECIALIST_HLT_CHECKPOINT:=${HLT_CHECKPOINT}}"
: "${ROUTER_SPECIALIST_HLT_ARCHITECTURE:=part}"
: "${ROUTER_SPECIALIST_OFFLINE_CHECKPOINT:=${OFFLINE_TEACHER_DIR}/best_model_val.pt}"
: "${ROUTER_SPECIALIST_OFFLINE_ARCHITECTURE:=part}"
: "${ROUTER_SPECIALIST_MAX_TRAIN_JETS:=150000}"
: "${ROUTER_SPECIALIST_MAX_VAL_JETS:=50000}"
: "${ROUTER_SPECIALIST_MAX_TEST_JETS:=100000}"
: "${ROUTER_SPECIALIST_EPOCHS:=8}"
: "${ROUTER_SPECIALIST_BATCH_SIZE:=128}"
: "${ROUTER_SPECIALIST_EVAL_BATCH_SIZE:=256}"
: "${ROUTER_SPECIALIST_LR:=0.001}"
: "${ROUTER_SPECIALIST_WEIGHT_DECAY:=0.0001}"
: "${ROUTER_SPECIALIST_PATIENCE:=3}"
: "${ROUTER_SPECIALIST_MODEL_SIZE:=base}"
: "${ROUTER_SPECIALIST_MAX_CONSTITS:=128}"
: "${ROUTER_SPECIALIST_WEIGHT_THRESHOLD:=0.0}"
: "${ROUTER_SPECIALIST_NO_AMP:=0}"
: "${ROUTER_SPECIALIST_MAX_TRAIN_BATCHES:=}"
: "${ROUTER_SPECIALIST_MAX_VAL_BATCHES:=}"

fresh_setup "$@"
fresh_require_file "${HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${ROUTER_SPECIALIST_HLT_CHECKPOINT}"
fresh_require_file "${ROUTER_SPECIALIST_OFFLINE_CHECKPOINT}"
fresh_claim_new_dir "${ROUTER_SPECIALIST_ROOT}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_hlt_offline_router_specialists.py"
  --output-dir "${ROUTER_SPECIALIST_ROOT}"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --hlt-checkpoint "${ROUTER_SPECIALIST_HLT_CHECKPOINT}"
  --hlt-architecture "${ROUTER_SPECIALIST_HLT_ARCHITECTURE}"
  --offline-checkpoint "${ROUTER_SPECIALIST_OFFLINE_CHECKPOINT}"
  --offline-architecture "${ROUTER_SPECIALIST_OFFLINE_ARCHITECTURE}"
  --max-train-jets "${ROUTER_SPECIALIST_MAX_TRAIN_JETS}"
  --max-val-jets "${ROUTER_SPECIALIST_MAX_VAL_JETS}"
  --max-test-jets "${ROUTER_SPECIALIST_MAX_TEST_JETS}"
  --seed "${TRAIN_SEED:-1777}"
  --epochs "${ROUTER_SPECIALIST_EPOCHS}"
  --batch-size "${ROUTER_SPECIALIST_BATCH_SIZE}"
  --eval-batch-size "${ROUTER_SPECIALIST_EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --lr "${ROUTER_SPECIALIST_LR}"
  --weight-decay "${ROUTER_SPECIALIST_WEIGHT_DECAY}"
  --early-stop-patience "${ROUTER_SPECIALIST_PATIENCE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --model-size "${ROUTER_SPECIALIST_MODEL_SIZE}"
  --max-constits "${ROUTER_SPECIALIST_MAX_CONSTITS}"
  --weight-threshold "${ROUTER_SPECIALIST_WEIGHT_THRESHOLD}"
)
fresh_append_flag_if_enabled cmd --no-amp "${ROUTER_SPECIALIST_NO_AMP}"
fresh_append_optional_arg cmd --max-train-batches "${ROUTER_SPECIALIST_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${ROUTER_SPECIALIST_MAX_VAL_BATCHES}"

fresh_write_run_config "${ROUTER_SPECIALIST_ROOT}" "hlt_offline_router_specialists" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${ROUTER_SPECIALIST_ROOT}/router/model_train_router_report.json"
  fresh_require_file "${ROUTER_SPECIALIST_ROOT}/router/model_val_router_report.json"
  fresh_require_file "${ROUTER_SPECIALIST_ROOT}/router/final_test_router_report.json"
  fresh_require_file "${ROUTER_SPECIALIST_ROOT}/specialists/agreement/best_model_val.pt"
  fresh_require_file "${ROUTER_SPECIALIST_ROOT}/specialists/disagreement/best_model_val.pt"
  fresh_require_file "${ROUTER_SPECIALIST_ROOT}/evaluation_report.json"
  fresh_require_file "${ROUTER_SPECIALIST_ROOT}/run_report.json"
fi

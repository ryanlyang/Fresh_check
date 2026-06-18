#!/usr/bin/env bash
# Diagnostic: compare normal HLT tagger with offline-trained tagger evaluated on HLT.

#SBATCH --job-name=hlt_off_diag
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${DISAGREE_DIAG_DIR:=${OUTPUT_ROOT}/hlt_offline_disagreement_diag_$(date +%Y%m%d_%H%M%S)}"
: "${DISAGREE_DIAG_SPLIT:=stack_val}"
: "${DISAGREE_DIAG_MAX_JETS:=50000}"
: "${DISAGREE_DIAG_BATCH_SIZE:=256}"
: "${DISAGREE_DIAG_NUM_WORKERS:=${NUM_WORKERS}}"
: "${DISAGREE_DIAG_DEVICE:=${DEVICE}}"
: "${DISAGREE_DIAG_HLT_CHECKPOINT:=${HLT_CHECKPOINT}}"
: "${DISAGREE_DIAG_HLT_ARCHITECTURE:=part}"
: "${DISAGREE_DIAG_OFFLINE_CHECKPOINT:=${OFFLINE_TEACHER_DIR}/best_model_val.pt}"
: "${DISAGREE_DIAG_OFFLINE_ARCHITECTURE:=part}"
: "${DISAGREE_DIAG_SAVE_PER_JET:=0}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "${MANIFEST_PATH}"
fresh_require_file "${HLT_CACHE_DIR}/${DISAGREE_DIAG_SPLIT}_fixed_hlt_metadata.json"
fresh_require_file "${DISAGREE_DIAG_HLT_CHECKPOINT}"
fresh_require_file "${DISAGREE_DIAG_OFFLINE_CHECKPOINT}"
fresh_claim_new_dir "${DISAGREE_DIAG_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/diagnose_hlt_offline_disagreement.py"
  --output-dir "${DISAGREE_DIAG_DIR}"
  --manifest-path "${MANIFEST_PATH}"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --hlt-checkpoint "${DISAGREE_DIAG_HLT_CHECKPOINT}"
  --hlt-architecture "${DISAGREE_DIAG_HLT_ARCHITECTURE}"
  --offline-checkpoint "${DISAGREE_DIAG_OFFLINE_CHECKPOINT}"
  --offline-architecture "${DISAGREE_DIAG_OFFLINE_ARCHITECTURE}"
  --split "${DISAGREE_DIAG_SPLIT}"
  --max-jets "${DISAGREE_DIAG_MAX_JETS}"
  --batch-size "${DISAGREE_DIAG_BATCH_SIZE}"
  --num-workers "${DISAGREE_DIAG_NUM_WORKERS}"
  --device "${DISAGREE_DIAG_DEVICE}"
)
fresh_append_flag_if_enabled cmd --save-per-jet "${DISAGREE_DIAG_SAVE_PER_JET}"

fresh_write_run_config "${DISAGREE_DIAG_DIR}" "diagnose_hlt_offline_disagreement" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${DISAGREE_DIAG_DIR}/disagreement_diagnostic_report.json"
  fresh_require_file "${DISAGREE_DIAG_DIR}/disagreement_logits.npz"
fi

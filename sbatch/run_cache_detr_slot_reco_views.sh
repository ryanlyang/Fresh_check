#!/usr/bin/env bash
# Cache reconstructed views from one trained DETR/free-slot reconstructor.

#SBATCH --job-name=detrslot_cache
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ARCHITECTURE="${1:?Usage: sbatch run_cache_detr_slot_reco_views.sh <gt|pn|pfn|pcnn>}"
RECONSTRUCTOR_CHECKPOINT="${DETR_SLOT_RECONSTRUCTOR_DIR}/${ARCHITECTURE}/best_model_val.pt"
RUN_OUTPUT_DIR="${DETR_SLOT_RECONSTRUCTED_VIEW_DIR}/${ARCHITECTURE}"

: "${NO_AMP:=0}"
: "${NON_STRICT_CHECKPOINT:=0}"
: "${SKIP_HLT_HASH_CHECK:=0}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${READ_CHUNK_SIZE:=50000}"
: "${DETR_SLOT_NO_SKIP_EXISTING:=0}"
: "${DETR_SLOT_TRIM_TO_VALID:=1}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "scripts/cache_detr_slot_reco_views.py"
fresh_require_file "${DETR_SLOT_MANIFEST_PATH}"
fresh_require_file "${RECONSTRUCTOR_CHECKPOINT}"
fresh_split_words split_args "${DETR_SLOT_CACHE_SPLITS}"
fresh_split_words label_filter_args "${DETR_SLOT_LABEL_FILTER_NAMES}"
for split in "${split_args[@]}"; do
  fresh_require_file "${DETR_SLOT_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_claim_new_dir "${RUN_OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_detr_slot_reco_views.py"
  --output-dir "${DETR_SLOT_RECONSTRUCTED_VIEW_DIR}"
  --manifest-path "${DETR_SLOT_MANIFEST_PATH}"
  --hlt-cache-dir "${DETR_SLOT_HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --reconstructor-checkpoint "${RECONSTRUCTOR_CHECKPOINT}"
  --architecture "${ARCHITECTURE}"
  --splits "${split_args[@]}"
  --batch-size "${DETR_SLOT_CACHE_BATCH_SIZE}"
  --num-workers "${DETR_SLOT_CACHE_NUM_WORKERS}"
  --device "${DETR_SLOT_CACHE_DEVICE}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --seed "${DETR_SLOT_RECO_SEED}"
  --export-max-tokens "${DETR_SLOT_CACHE_EXPORT_MAX_TOKENS}"
  --confidence-threshold "${DETR_SLOT_CACHE_CONFIDENCE_THRESHOLD}"
  --min-tokens-per-view "${DETR_SLOT_CACHE_MIN_TOKENS_PER_VIEW}"
)
if ((${#label_filter_args[@]})); then
  cmd+=(--label-filter-names "${label_filter_args[@]}")
fi
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${DETR_SLOT_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --non-strict-checkpoint "${NON_STRICT_CHECKPOINT}"
fresh_append_flag_if_enabled cmd --skip-detr-metrics "${DETR_SLOT_SKIP_DETR_METRICS}"
fresh_append_flag_if_enabled cmd --trim-to-valid "${DETR_SLOT_TRIM_TO_VALID}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_flag_if_enabled cmd --no-skip-existing "${DETR_SLOT_NO_SKIP_EXISTING}"
fresh_append_optional_arg cmd --max-jets-per-split "${DETR_SLOT_CACHE_MAX_JETS_PER_SPLIT}"

fresh_write_run_config "${RUN_OUTPUT_DIR}" "detr_slot_cache_${ARCHITECTURE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${RUN_OUTPUT_DIR}/${split}_reconstructed_view.npz"
    fresh_require_file "${RUN_OUTPUT_DIR}/${split}_reconstructed_view_metadata.json"
  done
  fresh_require_file "${RUN_OUTPUT_DIR}/cache_report.json"
  fresh_require_file "${RUN_OUTPUT_DIR}/cache_summary.csv"
fi

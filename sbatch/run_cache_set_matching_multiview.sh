#!/usr/bin/env bash
# Cache one reconstructed view from a trained set-matching reconstructor.

#SBATCH --job-name=setmatch_cache
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

ARCHITECTURE="${1:?Usage: sbatch run_cache_set_matching_multiview.sh <gt|pn|pfn|pcnn>}"
RECONSTRUCTOR_CHECKPOINT="${SET_MATCHING_RECONSTRUCTOR_DIR}/${ARCHITECTURE}/best_model_val.pt"
RUN_OUTPUT_DIR="${SET_MATCHING_RECONSTRUCTED_VIEW_DIR}/${ARCHITECTURE}"

: "${NO_AMP:=0}"
: "${NON_STRICT_CHECKPOINT:=0}"
: "${SKIP_SET_MATCHING_METRICS:=0}"
: "${TRIM_TO_VALID:=1}"
: "${SKIP_HLT_HASH_CHECK:=0}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${READ_CHUNK_SIZE:=50000}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "scripts/cache_set_matching_reco_views.py"
fresh_require_file "${SET_MATCHING_MANIFEST_PATH}"
fresh_require_file "${RECONSTRUCTOR_CHECKPOINT}"
fresh_split_words split_args "${SET_MATCHING_CACHE_SPLITS}"
for split in "${split_args[@]}"; do
  fresh_require_file "${SET_MATCHING_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_claim_new_dir "${RUN_OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_set_matching_reco_views.py"
  --output-dir "${SET_MATCHING_ROOT}"
  --manifest-path "${SET_MATCHING_MANIFEST_PATH}"
  --hlt-cache-dir "${SET_MATCHING_HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --reconstructor-checkpoint "${RECONSTRUCTOR_CHECKPOINT}"
  --architecture "${ARCHITECTURE}"
  --splits "${split_args[@]}"
  --batch-size "${SET_MATCHING_CACHE_BATCH_SIZE}"
  --num-workers "${SET_MATCHING_CACHE_NUM_WORKERS}"
  --device "${SET_MATCHING_CACHE_DEVICE}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --seed "${SET_MATCHING_RECO_SEED}"
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${SET_MATCHING_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --non-strict-checkpoint "${NON_STRICT_CHECKPOINT}"
fresh_append_flag_if_enabled cmd --skip-set-matching-metrics "${SKIP_SET_MATCHING_METRICS}"
fresh_append_flag_if_enabled cmd --trim-to-valid "${TRIM_TO_VALID}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_optional_arg cmd --max-jets-per-split "${SET_MATCHING_CACHE_MAX_JETS_PER_SPLIT}"

fresh_write_run_config "${RUN_OUTPUT_DIR}" "set_matching_cache_${ARCHITECTURE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${RUN_OUTPUT_DIR}/${split}_reconstructed_view.npz"
    fresh_require_file "${RUN_OUTPUT_DIR}/${split}_reconstructed_view_metadata.json"
  done
  fresh_require_file "${RUN_OUTPUT_DIR}/cache_report.json"
fi

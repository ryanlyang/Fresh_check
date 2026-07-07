#!/usr/bin/env bash
# Cache PD10-V2 particle dual-view teacher logits and representations.

#SBATCH --job-name=pd10_pdv_cache
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

CHECKPOINT="${PD10_V2_PARTICLE_DUAL_VIEW_TEACHER_CHECKPOINT}"
MODEL_NAME="particle_dual_view_teacher_10class"

fresh_setup "$@"
fresh_require_file "${CHECKPOINT}"
fresh_require_file "${PD10_MANIFEST_PATH}"
fresh_require_dir "${PD10_HLT_CACHE_DIR}"
if [[ -n "${PD10_OFFLINE_CACHE_DIR:-}" ]]; then
  fresh_require_dir "${PD10_OFFLINE_CACHE_DIR}"
else
  fresh_require_data_dir
fi
if fresh_bool_enabled "${PD10_V2_PARTICLE_DUAL_VIEW_CACHE_NO_SKIP_EXISTING}"; then
  fresh_refuse_existing_dir "${PD10_V2_TEACHER_LOGITS_DIR}/${MODEL_NAME}"
  fresh_refuse_existing_dir "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}"
fi

fresh_split_words split_args "${PD10_V2_PARTICLE_DUAL_VIEW_CACHE_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_pd10_particle_dual_view_teacher.py"
  --checkpoint "${CHECKPOINT}"
  --manifest "${PD10_MANIFEST_PATH}"
  --hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --data-dir "${PD10_DATA_DIR}"
  --logit-output-dir "${PD10_V2_TEACHER_LOGITS_DIR}"
  --representation-output-dir "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}"
  --splits "${split_args[@]}"
  --batch-size "${PD10_V2_PARTICLE_DUAL_VIEW_CACHE_BATCH_SIZE}"
  --num-workers "${PD10_V2_PARTICLE_DUAL_VIEW_CACHE_NUM_WORKERS}"
  --device "${PD10_V2_PARTICLE_DUAL_VIEW_CACHE_DEVICE}"
  --max-model-train-jets "${PD10_MODEL_TRAIN_SIZE}"
  --max-model-val-jets "${PD10_MODEL_VAL_SIZE}"
  --max-final-test-jets "${PD10_FINAL_TEST_SIZE}"
  --control-seed "${PD10_V2_PARTICLE_DUAL_VIEW_CACHE_CONTROL_SEED}"
  --read-chunk-size "${PD10_V2_PARTICLE_DUAL_VIEW_READ_CHUNK_SIZE}"
  --confirm-final-test
)
fresh_append_optional_arg cmd --max-batches "${PD10_V2_PARTICLE_DUAL_VIEW_CACHE_MAX_BATCHES}"
fresh_append_optional_arg cmd --offline-cache-dir "${PD10_OFFLINE_CACHE_DIR:-}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --no-skip-existing "${PD10_V2_PARTICLE_DUAL_VIEW_CACHE_NO_SKIP_EXISTING}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${PD10_V2_PARTICLE_DUAL_VIEW_VERIFY_LABEL_BRANCHES}"
fresh_append_flag_if_enabled cmd --no-verify-hlt-hash "${PD10_V2_PARTICLE_DUAL_VIEW_NO_VERIFY_HLT_HASH}"

fresh_write_run_config "${PD10_V2_TEACHER_LOGITS_DIR}/${MODEL_NAME}" "pd10_v2_particle_dual_view_cache" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${PD10_V2_TEACHER_LOGITS_DIR}/${MODEL_NAME}/particle_dual_view_cache_manifest.json"
  fresh_require_file "${PD10_V2_TEACHER_LOGITS_DIR}/${MODEL_NAME}/particle_dual_view_cache_report.json"
  fresh_require_file "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}/teacher_representation_manifest.json"
  fresh_require_file "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}/teacher_representation_cache_report.json"
  fresh_assert_json_ok "${PD10_V2_TEACHER_LOGITS_DIR}/${MODEL_NAME}/particle_dual_view_cache_manifest.json"
  fresh_assert_json_ok "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}/teacher_representation_manifest.json"
  for split in "${split_args[@]}"; do
    fresh_require_file "${PD10_V2_TEACHER_LOGITS_DIR}/${MODEL_NAME}/${split}_predictions.npz"
    fresh_require_file "${PD10_V2_TEACHER_LOGITS_DIR}/${MODEL_NAME}/${split}_predictions_metadata.json"
    fresh_require_file "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}/${split}_representations.npz"
    fresh_require_file "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}/${split}_representations_metadata.json"
  done
fi

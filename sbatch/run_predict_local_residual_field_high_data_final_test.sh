#!/usr/bin/env bash
# Evaluate one locked deployable checkpoint on the sealed 1M HLT-only final test.

#SBATCH --job-name=lprf_hd_final
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=1-00:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"

RECIPE="${1:?Usage: sbatch run_predict_local_residual_field_high_data_final_test.sh <A0|P7b> <seed>}"
SEED="${2:?Usage: sbatch run_predict_local_residual_field_high_data_final_test.sh <A0|P7b> <seed>}"
case "${RECIPE}" in A0|P7b) ;; *) echo "recipe must be A0 or P7b" >&2; exit 2 ;; esac
case "${SEED}" in 20421|20522|20623) ;; *) echo "unsupported high-data seed ${SEED}" >&2; exit 2 ;; esac
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT:?high-data campaign root is required}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST:=${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/study_manifest.json}"

fresh_setup
fresh_bool_enabled "${CONFIRM_FINAL_TEST}" || {
  echo "final-test prediction requires CONFIRM_FINAL_TEST=1" >&2
  exit 2
}
fresh_require_file "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/validation_report/run_report.json"
manifest_value() {
  "${PYTHON_BIN}" -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); [value := value[key] for key in sys.argv[2].split(".")]; print(value)' \
    "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}" "$1"
}
checkpoint="$(manifest_value "run_dirs.${SEED}.${RECIPE}")/best_model_val.pt"
manifest_path="$(manifest_value paths.split_manifest)"
hlt_cache_dir="$(manifest_value paths.hlt_cache_dir)"
prediction_parent="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/final_test_predictions/seed_${SEED}"
prediction_dir="${prediction_parent}/${RECIPE}"
fresh_require_file "${checkpoint}"
fresh_require_file "${manifest_path}"
fresh_require_dir "${hlt_cache_dir}"
if ! fresh_is_dry_run && [[ -d "${prediction_dir}" ]]; then
  partial_dir="${prediction_dir}.partial_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"
  echo "Quarantining incomplete final-test prediction directory: ${prediction_dir} -> ${partial_dir}"
  mv -- "${prediction_dir}" "${partial_dir}"
fi

cmd=(
  "${PYTHON_BIN}" -u scripts/predict_local_residual_field_tagger.py
  --checkpoint "${checkpoint}"
  --prediction-dir "${prediction_parent}"
  --model-name "${RECIPE}"
  --hlt-cache-dir "${hlt_cache_dir}"
  --manifest-path "${manifest_path}"
  --splits final_test
  --batch-size 128
  --num-workers 4
  --device "${DEVICE}"
  --confirm-final-test
)
fresh_write_run_config "${prediction_dir}" \
  "local_residual_field_high_data_final_${RECIPE}_${SEED}" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_require_file "${prediction_dir}/final_test_predictions.npz"
  fresh_require_file "${prediction_dir}/final_test_predictions_metadata.json"
fi

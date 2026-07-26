#!/usr/bin/env bash
# Train one P7b replicate using the shared frozen 3M Orobust/C0 sources.

#SBATCH --job-name=lprf_hd_P7b
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=3-00:00:00
#SBATCH --mem=500G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"

SEED="${1:?Usage: sbatch run_train_local_residual_field_high_data_p7b.sh <20421|20522|20623>}"
shift
case "${SEED}" in 20421|20522|20623) ;; *) echo "unsupported high-data P7b seed ${SEED}" >&2; exit 2 ;; esac
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT:?high-data campaign root is required}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST:=${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/study_manifest.json}"
OUTPUT_DIR="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/runs/seed_${SEED}/P7b"

fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}"
manifest_value() {
  "${PYTHON_BIN}" -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); [value := value[key] for key in sys.argv[2].split(".")]; print(value)' \
    "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}" "$1"
}
manifest_path="$(manifest_value paths.split_manifest)"
hlt_cache_dir="$(manifest_value paths.hlt_cache_dir)"
target_cache_dir="$(manifest_value paths.target_cache_dir)"
selected_consumer_json="$(manifest_value paths.selected_consumer_json)"
c0_checkpoint="$(manifest_value paths.c0_checkpoint)"
consumer_checkpoint="$(manifest_value paths.consumer_checkpoint)"
consumer_teacher_config="$(manifest_value paths.consumer_teacher_config)"
consumer_run_report="$(manifest_value paths.consumer_run_report)"
for file in "${manifest_path}" "${selected_consumer_json}" "${c0_checkpoint}" \
  "${consumer_checkpoint}" "${consumer_teacher_config}" "${consumer_run_report}"; do
  fresh_require_file "${file}"
done
fresh_require_dir "${hlt_cache_dir}"
fresh_require_dir "${target_cache_dir}"

reuse_completed_training=0
if ! fresh_is_dry_run && [[ -d "${OUTPUT_DIR}" ]]; then
  if [[ -f "${OUTPUT_DIR}/best_model_val.pt" \
      && -f "${OUTPUT_DIR}/run_report.json" \
      && -f "${OUTPUT_DIR}/source_metadata.json" \
      && -f "${OUTPUT_DIR}/curriculum_schedule.json" ]]; then
    reuse_completed_training=1
    echo "Reusing completed high-data P7b artifacts; only the audit will rerun"
  else
    partial_dir="${OUTPUT_DIR}.partial_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"
    echo "Quarantining incomplete P7b directory: ${OUTPUT_DIR} -> ${partial_dir}"
    mv -- "${OUTPUT_DIR}" "${partial_dir}"
  fi
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"

train_cmd=(
  "${PYTHON_BIN}" -u scripts/train_local_residual_field_curriculum_student.py
  --run-id P7b
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${hlt_cache_dir}"
  --target-cache-dir "${target_cache_dir}"
  --manifest-path "${manifest_path}"
  --predictor-warm-start-checkpoint "${c0_checkpoint}"
  --student-warm-start-checkpoint "${consumer_checkpoint}"
  --selected-consumer-json "${selected_consumer_json}"
  --oracle-teacher-checkpoint "${consumer_checkpoint}"
  --oracle-teacher-config-path "${consumer_teacher_config}"
  --oracle-run-report-path "${consumer_run_report}"
  --seed "${SEED}"
  --epochs 12
  --batch-size 24
  --eval-batch-size 64
  --num-workers 0
  --gradient-accumulation-steps 1
  --device "${DEVICE}"
)
record_cmd=(
  "${PYTHON_BIN}" -u scripts/record_local_residual_field_high_data_p7b.py
  --study-manifest "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}"
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
)
fresh_write_run_config "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/run_configs/P7b_seed_${SEED}" \
  "local_residual_field_high_data_P7b_${SEED}" "${train_cmd[@]}" "${record_cmd[@]}"
if [[ "${reuse_completed_training}" -eq 0 ]]; then
  fresh_run "${train_cmd[@]}"
fi
fresh_run "${record_cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_assert_json_ok "${OUTPUT_DIR}/high_data_completion.json"
fi

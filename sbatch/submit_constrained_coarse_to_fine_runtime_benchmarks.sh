#!/usr/bin/env bash
# Submit the full manifest-bound A-D C2F runtime benchmark matrix (Step 7).

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter
fresh_activate_env

: "${CONSTRAINED_C2F_CALIBRATION_ROOT:?Set CONSTRAINED_C2F_CALIBRATION_ROOT to a completed Step 6 root}"
: "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH:?Set CONSTRAINED_C2F_PARENT_MANIFEST_PATH used to build the calibration root}"
: "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/calibration_manifest.json.gz}"
: "${CONSTRAINED_C2F_CALIBRATION_HLT_CACHE_DIR:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/hlt_cache}"
: "${CONSTRAINED_C2F_CALIBRATION_OFFLINE_CACHE_DIR:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/offline_cache}"
: "${CONSTRAINED_C2F_CALIBRATION_TARGET_CACHE_DIR:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/targets}"
: "${CONSTRAINED_C2F_BENCHMARK_ROOT:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/runtime_benchmarks}"
: "${CONSTRAINED_C2F_BENCHMARK_EPOCHS:=3}"
: "${CONSTRAINED_C2F_BENCHMARK_LR:=2.0e-4}"
: "${CONSTRAINED_C2F_BENCHMARK_WARMUP_FRACTION:=0.10}"
: "${CONSTRAINED_C2F_BENCHMARK_MIN_LR_RATIO:=0.05}"
: "${CONSTRAINED_C2F_BENCHMARK_SINGLE_VIEW_CANDIDATES:=32:64 48:96 64:128}"
: "${CONSTRAINED_C2F_BENCHMARK_C6_CANDIDATES:=16:32 24:48 32:64}"
: "${CONSTRAINED_C2F_BENCHMARK_C4_CANDIDATES:=16:32}"
: "${CONSTRAINED_C2F_BENCHMARK_INPUT_WORKERS:=0 4 8 12}"
: "${CONSTRAINED_C2F_BENCHMARK_C4_HUNGARIAN_WORKERS:=1 4 8 12}"
: "${CONSTRAINED_C2F_BENCHMARK_PEAK_LEARNING_RATES:=2.0e-4 4.0e-4 6.0e-4}"
: "${CONSTRAINED_C2F_BENCHMARK_CPUS:=12}"
: "${CONSTRAINED_C2F_BENCHMARK_MEM:=220G}"
: "${CONSTRAINED_C2F_BENCHMARK_GPU_GRES:=gpu:1}"
: "${CONSTRAINED_C2F_SBATCH_ACCOUNT:=}"
: "${CONSTRAINED_C2F_SBATCH_PARTITION:=}"

IFS=' ' read -r -a single_view_candidates <<< "${CONSTRAINED_C2F_BENCHMARK_SINGLE_VIEW_CANDIDATES}"
IFS=' ' read -r -a c6_candidates <<< "${CONSTRAINED_C2F_BENCHMARK_C6_CANDIDATES}"
IFS=' ' read -r -a c4_candidates <<< "${CONSTRAINED_C2F_BENCHMARK_C4_CANDIDATES}"
IFS=' ' read -r -a input_workers <<< "${CONSTRAINED_C2F_BENCHMARK_INPUT_WORKERS}"
IFS=' ' read -r -a c4_hungarian_workers <<< "${CONSTRAINED_C2F_BENCHMARK_C4_HUNGARIAN_WORKERS}"
IFS=' ' read -r -a peak_learning_rates <<< "${CONSTRAINED_C2F_BENCHMARK_PEAK_LEARNING_RATES}"

for required in \
  "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH}" \
  "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH}" \
  "${CONSTRAINED_C2F_CALIBRATION_TARGET_CACHE_DIR}/hierarchy_target_cache_manifest.json"; do
  fresh_require_file "${required}"
done
fresh_require_dir "${CONSTRAINED_C2F_CALIBRATION_HLT_CACHE_DIR}"
fresh_require_dir "${CONSTRAINED_C2F_CALIBRATION_OFFLINE_CACHE_DIR}"
if ! fresh_is_dry_run && [[ -e "${CONSTRAINED_C2F_BENCHMARK_ROOT}" ]]; then
  echo "Refusing to reuse an existing runtime benchmark root: ${CONSTRAINED_C2F_BENCHMARK_ROOT}" >&2
  exit 2
fi

validation_cmd=(
  "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_calibration_slice.py
  --parent-manifest "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH}"
  --calibration-manifest "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH}"
  --hlt-cache-dir "${CONSTRAINED_C2F_CALIBRATION_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_CALIBRATION_OFFLINE_CACHE_DIR}"
  --target-cache-dir "${CONSTRAINED_C2F_CALIBRATION_TARGET_CACHE_DIR}"
  --output "${CONSTRAINED_C2F_CALIBRATION_ROOT}/runtime_benchmark_calibration_validation.json"
)
fresh_run "${validation_cmd[@]}"

mkdir -p "${CONSTRAINED_C2F_BENCHMARK_ROOT}"
plan_path="${CONSTRAINED_C2F_BENCHMARK_ROOT}/benchmark_plan.json"
plan_cmd=(
  "${PYTHON_BIN}" scripts/build_constrained_coarse_to_fine_runtime_benchmark_plan.py
  --output "${plan_path}"
  --calibration-root "${CONSTRAINED_C2F_CALIBRATION_ROOT}"
  --calibration-manifest "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH}"
  --epochs "${CONSTRAINED_C2F_BENCHMARK_EPOCHS}"
  --learning-rate "${CONSTRAINED_C2F_BENCHMARK_LR}"
  --warmup-fraction "${CONSTRAINED_C2F_BENCHMARK_WARMUP_FRACTION}"
  --min-lr-ratio "${CONSTRAINED_C2F_BENCHMARK_MIN_LR_RATIO}"
  --single-view-candidates "${single_view_candidates[@]}"
  --c6-candidates "${c6_candidates[@]}"
  --c4-candidates "${c4_candidates[@]}"
  --input-workers "${input_workers[@]}"
  --c4-hungarian-workers "${c4_hungarian_workers[@]}"
  --peak-learning-rates "${peak_learning_rates[@]}"
)
fresh_run "${plan_cmd[@]}"

tsv_path="${CONSTRAINED_C2F_BENCHMARK_ROOT}/benchmark_plan.tsv"
"${plan_cmd[@]}" --emit-tsv > "${tsv_path}"

sbatch_args=(--cpus-per-task="${CONSTRAINED_C2F_BENCHMARK_CPUS}" --mem="${CONSTRAINED_C2F_BENCHMARK_MEM}" --gres="${CONSTRAINED_C2F_BENCHMARK_GPU_GRES}")
[[ -n "${CONSTRAINED_C2F_SBATCH_ACCOUNT}" ]] && sbatch_args+=(--account="${CONSTRAINED_C2F_SBATCH_ACCOUNT}")
[[ -n "${CONSTRAINED_C2F_SBATCH_PARTITION}" ]] && sbatch_args+=(--partition="${CONSTRAINED_C2F_SBATCH_PARTITION}")

export CONSTRAINED_C2F_ROOT="${CONSTRAINED_C2F_CALIBRATION_ROOT}"
export CONSTRAINED_C2F_MANIFEST_PATH="${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH}"
export CONSTRAINED_C2F_HLT_CACHE_DIR="${CONSTRAINED_C2F_CALIBRATION_HLT_CACHE_DIR}"
export CONSTRAINED_C2F_OFFLINE_CACHE_DIR="${CONSTRAINED_C2F_CALIBRATION_OFFLINE_CACHE_DIR}"
export CONSTRAINED_C2F_TARGET_CACHE_DIR="${CONSTRAINED_C2F_CALIBRATION_TARGET_CACHE_DIR}"
export CONSTRAINED_C2F_RECON_ROOT="${CONSTRAINED_C2F_BENCHMARK_ROOT}/reconstructors"
export CONSTRAINED_C2F_RECO_EPOCHS="${CONSTRAINED_C2F_BENCHMARK_EPOCHS}"
export CONSTRAINED_C2F_RECO_SEED=22031
export CONSTRAINED_C2F_RECO_FIXED_HORIZON=1
export CONSTRAINED_C2F_RECO_MIN_EPOCHS="${CONSTRAINED_C2F_BENCHMARK_EPOCHS}"
export CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT=1
export CONSTRAINED_C2F_RECO_STACK_VAL_SPLIT=""
export CONSTRAINED_C2F_RECO_LR_SCHEDULE=constant
export CONSTRAINED_C2F_RECO_WARMUP_FRACTION="${CONSTRAINED_C2F_BENCHMARK_WARMUP_FRACTION}"
export CONSTRAINED_C2F_RECO_MIN_LR_RATIO="${CONSTRAINED_C2F_BENCHMARK_MIN_LR_RATIO}"
export CONSTRAINED_C2F_RECO_PROGRESS_INTERVAL_BATCHES=25
export CONSTRAINED_C2F_RECO_AMP=0
export OVERWRITE=0

job_ids=()
submission_path="${CONSTRAINED_C2F_BENCHMARK_ROOT}/benchmark_submission.tsv"
printf 'run_id\tvariant\tmatrix_case\tjob_id\n' > "${submission_path}"
while IFS=$'\t' read -r run_id variant matrix_case profile precision train_batch eval_batch schedule learning_rate hlt_encoder_lr_scale workers prefetch executor hungarian; do
  [[ -z "${run_id}" ]] && continue
  export CONSTRAINED_C2F_RECO_OUTPUT_ID="${run_id}"
  export CONSTRAINED_C2F_RUNTIME_PROFILE="${profile}"
  export CONSTRAINED_C2F_RECO_PRECISION_MODE="${precision}"
  export CONSTRAINED_C2F_RECO_BATCH_SIZE="${train_batch}"
  export CONSTRAINED_C2F_RECO_EVAL_BATCH_SIZE="${eval_batch}"
  export CONSTRAINED_C2F_RECO_LR_SCHEDULE="${schedule}"
  export CONSTRAINED_C2F_RECO_LEARNING_RATE="${learning_rate}"
  export CONSTRAINED_C2F_RECO_HLT_ENCODER_LR_SCALE="${hlt_encoder_lr_scale}"
  export CONSTRAINED_C2F_RECO_NUM_WORKERS="${workers}"
  export CONSTRAINED_C2F_HUNGARIAN_EXECUTOR="${executor}"
  export CONSTRAINED_C2F_HUNGARIAN_WORKERS="${hungarian}"
  if [[ "${prefetch}" == "None" ]]; then
    export CONSTRAINED_C2F_RECO_PREFETCH_FACTOR=""
  else
    export CONSTRAINED_C2F_RECO_PREFETCH_FACTOR="${prefetch}"
  fi
  if fresh_is_dry_run; then
    fresh_print_shell_command sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_train_constrained_coarse_to_fine_reconstructor.sh" "${variant}"
    continue
  fi
  submitted="$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_train_constrained_coarse_to_fine_reconstructor.sh" "${variant}")"
  echo "${submitted}"
  job_id="${submitted##* }"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Could not parse Slurm job id: ${submitted}" >&2; exit 2; }
  printf '%s\t%s\t%s\t%s\n' "${run_id}" "${variant}" "${matrix_case}" "${job_id}" >> "${submission_path}"
  job_ids+=("${job_id}")
done < "${tsv_path}"

if fresh_is_dry_run; then
  exit 0
fi
dependency="afterok:$(IFS=:; echo "${job_ids[*]}")"
export CONSTRAINED_C2F_CALIBRATION_ROOT CONSTRAINED_C2F_BENCHMARK_ROOT
report_args=(--cpus-per-task=4 --mem=32G)
[[ -n "${CONSTRAINED_C2F_SBATCH_ACCOUNT}" ]] && report_args+=(--account="${CONSTRAINED_C2F_SBATCH_ACCOUNT}")
[[ -n "${CONSTRAINED_C2F_SBATCH_PARTITION}" ]] && report_args+=(--partition="${CONSTRAINED_C2F_SBATCH_PARTITION}")
report_submitted="$(sbatch "${report_args[@]}" --dependency="${dependency}" "${SCRIPT_DIR}/run_write_constrained_coarse_to_fine_runtime_benchmark_report.sh")"
echo "${report_submitted}"
report_job_id="${report_submitted##* }"
printf 'report\tjob_id\t%s\n' "${report_job_id}" >> "${submission_path}"
echo "constrained_c2f_runtime_benchmark_submission_complete:"
echo "  root: ${CONSTRAINED_C2F_BENCHMARK_ROOT}"
echo "  plan: ${plan_path}"
echo "  benchmark_jobs: ${#job_ids[@]}"
echo "  report_job: ${report_job_id}"

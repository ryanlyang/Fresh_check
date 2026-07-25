#!/usr/bin/env bash
# Materialize the measured-safe dense caches and queue the full high-data pilot.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PAB_CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${PAB_CONDA_ENV:=atlas_kd_tigris}"
: "${PAB_HIGH_DATA_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${PAB_SBATCH_ACCOUNT:=reu-aisocial}"
: "${PAB_SBATCH_PARTITION:=tigris}"
: "${PAB_HIGH_DATA_ROOT:?Set PAB_HIGH_DATA_ROOT to the completed high_data_3m preparation root}"
: "${PAB_HIGH_DATA_MIN_FREE_GIB:=40}"
: "${PAB_HIGH_DATA_SOURCE_MODE:=auto}"

export PAB_CONDA_BASE PAB_CONDA_ENV
export CONDA_BASE="${PAB_CONDA_BASE}"
export CONDA_ENV="${PAB_CONDA_ENV}"
export PYTHONNOUSERSITE=1

cd "${PROJECT_DIR}"
mkdir -p fresh_check_logs

parent="${PAB_HIGH_DATA_ROOT}/inputs/split_manifest/split_manifest.json.gz"
hlt_cache="${PAB_HIGH_DATA_ROOT}/inputs/hlt_cache"
offline_cache="${PAB_HIGH_DATA_ROOT}/inputs/offline_cache"
baseline="${PAB_HIGH_DATA_ROOT}/taggers/A0/best_model_val.pt"
fresh_preflight="${PAB_HIGH_DATA_ROOT}/preflight"

[[ -f "${parent}" && ! -L "${parent}" ]] || {
  echo "Missing completed high-data parent manifest: ${parent}" >&2
  exit 2
}
[[ -f "${fresh_preflight}/prediction_anchored_high_data_preparation.json" ]] || {
  echo "High-data manifest preparation is incomplete: ${fresh_preflight}" >&2
  exit 2
}
source_complete=1
for split in model_train model_val stack_train stack_val final_test; do
  [[ -f "${hlt_cache}/${split}_fixed_hlt.npz" ]] || source_complete=0
  [[ -f "${hlt_cache}/${split}_fixed_hlt_metadata.json" ]] || source_complete=0
done
for split in model_train model_val stack_train stack_val; do
  [[ -f "${offline_cache}/${split}_offline.npz" ]] || source_complete=0
  [[ -f "${offline_cache}/${split}_offline_metadata.json" ]] || source_complete=0
done
[[ -f "${baseline}" && ! -L "${baseline}" ]] || source_complete=0

source_absent=1
for path in "${hlt_cache}" "${offline_cache}" "${PAB_HIGH_DATA_ROOT}/taggers/A0"; do
  [[ ! -e "${path}" ]] || source_absent=0
done
if [[ "${PAB_HIGH_DATA_SOURCE_MODE}" == "auto" ]]; then
  if [[ "${source_complete}" == "1" ]]; then
    PAB_HIGH_DATA_SOURCE_MODE=reuse
  elif [[ "${source_absent}" == "1" ]]; then
    PAB_HIGH_DATA_SOURCE_MODE=build
  else
    echo "High-data source is partial; refusing mixed build/reuse state" >&2
    exit 2
  fi
fi
if [[ "${PAB_HIGH_DATA_SOURCE_MODE}" == "build" && "${source_absent}" != "1" ]]; then
  echo "Build mode requires absent HLT/offline/A0 source paths" >&2
  exit 2
fi
if [[ "${PAB_HIGH_DATA_SOURCE_MODE}" == "reuse" && "${source_complete}" != "1" ]]; then
  echo "Reuse mode requires complete HLT/offline/A0 source artifacts" >&2
  exit 2
fi
if [[ "${PAB_HIGH_DATA_SOURCE_MODE}" != "build" && "${PAB_HIGH_DATA_SOURCE_MODE}" != "reuse" ]]; then
  echo "PAB_HIGH_DATA_SOURCE_MODE must be auto, build, or reuse" >&2
  exit 2
fi

available_kib="$(df -Pk "${PAB_HIGH_DATA_ROOT}" | awk 'NR==2 {print $4}')"
required_kib="$((PAB_HIGH_DATA_MIN_FREE_GIB * 1024 * 1024))"
if [[ "${PAB_HIGH_DATA_SOURCE_MODE}" == "build" ]]; then
  if [[ ! "${available_kib}" =~ ^[0-9]+$ || "${available_kib}" -lt "${required_kib}" ]]; then
    echo "High-data source build requires at least ${PAB_HIGH_DATA_MIN_FREE_GIB} GiB free" >&2
    exit 2
  fi
fi

stamp="$(date -u +%Y%m%d_%H%M%S)"
: "${PREDICTION_ANCHORED_ARTIFACT_ROOT:=${PAB_HIGH_DATA_ROOT}/full_pilot_${stamp}}"
: "${PAB_PREFLIGHT_ROOT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/preflight}"
[[ ! -e "${PREDICTION_ANCHORED_ARTIFACT_ROOT}" ]] || {
  echo "Fresh high-data campaign root already exists: ${PREDICTION_ANCHORED_ARTIFACT_ROOT}" >&2
  exit 2
}
mkdir -p "${PREDICTION_ANCHORED_ARTIFACT_ROOT}"

export DATA_DIR="${PAB_HIGH_DATA_DATA_DIR}"
export DATA_DIRS="${PAB_HIGH_DATA_DATA_DIR}"
export MANIFEST_PATH="${parent}"
export MAX_CONSTITS=128
export OVERWRITE=0

export HLT_CACHE_DIR="${hlt_cache}"
export HLT_SPLITS="model_train model_val stack_train stack_val final_test"
export HLT_PROFILE=fixed_hlt_v2_realistic
export HLT_DEGRADATION_STRENGTH=2.5

export ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH="${parent}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR="${offline_cache}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS="model_train model_val stack_train stack_val"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_DATA_DIRS="${PAB_HIGH_DATA_DATA_DIR}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_OVERWRITE=0
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_APPEND=0

export LOCAL_RESIDUAL_FIELD_ROOT="${PAB_HIGH_DATA_ROOT}"
export LOCAL_RESIDUAL_FIELD_MANIFEST_PATH="${parent}"
export LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR="${hlt_cache}"
export LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR="${PAB_HIGH_DATA_ROOT}/targets"
export LOCAL_RESIDUAL_FIELD_RECON_ROOT="${PAB_HIGH_DATA_ROOT}/reconstructors"
export LOCAL_RESIDUAL_FIELD_TAGGER_ROOT="${PAB_HIGH_DATA_ROOT}/taggers"
export DEVICE=cuda

export PAB_SOURCE_BASE="${PAB_HIGH_DATA_ROOT}"
export PREDICTION_ANCHORED_ARTIFACT_ROOT PAB_PREFLIGHT_ROOT
export PAB_SPLIT_PROFILE=high_data_3m
export PAB_BUDGET_GIB=6
export PAB_RECON_PHASE2_EPOCHS=4
export PAB_CONSUMER_BASELINE_STEPS=120000
export PAB_CONSUMER_FINETUNE_STEPS=24000

submit_job() {
  local label="$1"
  shift
  local output job_id
  output="$(sbatch --parsable "$@")"
  job_id="${output%%;*}"
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse numeric ${label} job ID from sbatch output: ${output}" >&2
    return 2
  fi
  printf 'submitted_%s=%s\n' "${label}" "${job_id}" >&2
  printf '%s\n' "${job_id}"
}

hlt_job=""
offline_job=""
a0_job=""
finalizer_dependency=()
if [[ "${PAB_HIGH_DATA_SOURCE_MODE}" == "build" ]]; then
  hlt_job="$(submit_job high_data_hlt \
    --account="${PAB_SBATCH_ACCOUNT}" --partition="${PAB_SBATCH_PARTITION}" \
    --nodes=1 --job-name=pab_hd_hlt --time=2-00:00:00 \
    --mem=512G --cpus-per-task=24 --export=ALL,PYTHONNOUSERSITE=1 \
    sbatch/run_build_fresh_hlt_cache.sh)"

  offline_job="$(submit_job high_data_offline \
    --account="${PAB_SBATCH_ACCOUNT}" --partition="${PAB_SBATCH_PARTITION}" \
    --nodes=1 --job-name=pab_hd_offline --time=2-00:00:00 \
    --mem=512G --cpus-per-task=24 --export=ALL,PYTHONNOUSERSITE=1 \
    sbatch/run_cache_architecture_view_offline_inputs.sh)"

  a0_job="$(submit_job high_data_a0 \
    --account="${PAB_SBATCH_ACCOUNT}" --partition="${PAB_SBATCH_PARTITION}" \
    --nodes=1 --job-name=pab_hd_source_a0 --time=3-00:00:00 \
    --mem=512G --cpus-per-task=16 --gres=gpu:gh200:1 \
    --dependency="afterok:${hlt_job}:${offline_job}" --export=ALL,PYTHONNOUSERSITE=1 \
    sbatch/run_train_local_residual_field_tagger.sh A0)"
  finalizer_dependency=(--dependency="afterok:${a0_job}")
fi

finalizer_job="$(submit_job high_data_finalizer "${finalizer_dependency[@]}" \
  --account="${PAB_SBATCH_ACCOUNT}" --partition="${PAB_SBATCH_PARTITION}" \
  --nodes=1 --job-name=pab_hd_finalize --time=08:00:00 \
  --mem=192G --cpus-per-task=16 \
  --export=ALL,PYTHONNOUSERSITE=1,PROJECT_DIR="${PROJECT_DIR}",PAB_SOURCE_BASE="${PAB_SOURCE_BASE}",PAB_PREFLIGHT_ROOT="${PAB_PREFLIGHT_ROOT}",PREDICTION_ANCHORED_ARTIFACT_ROOT="${PREDICTION_ANCHORED_ARTIFACT_ROOT}",PAB_SPLIT_PROFILE=high_data_3m,PAB_BUDGET_GIB=6,PAB_RECON_PHASE2_EPOCHS=4,PAB_CONSUMER_BASELINE_STEPS=120000,PAB_CONSUMER_FINETUNE_STEPS=24000 \
  sbatch/run_finalize_prediction_anchored_bridge_submission.sh)"

printf 'profile=high_data_3m\n'
printf 'source_mode=%s\n' "${PAB_HIGH_DATA_SOURCE_MODE}"
printf 'source_root=%s\n' "${PAB_HIGH_DATA_ROOT}"
printf 'artifact_root=%s\n' "${PREDICTION_ANCHORED_ARTIFACT_ROOT}"
printf 'hlt_job=%s\n' "${hlt_job:-REUSED_EXISTING}"
printf 'offline_job=%s\n' "${offline_job:-REUSED_EXISTING}"
printf 'a0_job=%s\n' "${a0_job:-REUSED_EXISTING}"
printf 'finalizer_job=%s\n' "${finalizer_job}"
printf 'consumer_baseline_steps=120000\n'
printf 'consumer_continuation_steps=24000\n'
printf 'reconstruction_phase2_epochs=4\n'
printf 'final_test_submission=SEALED_NOT_AUTOMATIC\n'

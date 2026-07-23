#!/usr/bin/env bash
# Queue a source-consistent 500k bridge pilot and then the complete B0--B6 graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PAB_CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${PAB_CONDA_ENV:=atlas_kd_tigris}"
: "${PAB_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${PAB_REBUILD_SOURCE:=1}"
: "${PAB_SBATCH_ACCOUNT:=reu-aisocial}"
: "${PAB_SBATCH_PARTITION:=tigris}"
export PAB_CONDA_BASE PAB_CONDA_ENV
export CONDA_BASE="${PAB_CONDA_BASE}"
export CONDA_ENV="${PAB_CONDA_ENV}"
export PYTHONNOUSERSITE=1

cd "${PROJECT_DIR}"
mkdir -p fresh_check_logs

stamp="$(date -u +%Y%m%d_%H%M%S)"
if [[ "${PAB_REBUILD_SOURCE}" == "1" ]]; then
  # Deliberately ignore an inherited PAB_SOURCE_BASE from an older 300k
  # residual-field campaign.  Reuse must be explicitly requested.
  PAB_SOURCE_BASE="${PAB_NEW_SOURCE_BASE:-${PROJECT_DIR}/checkpoints/prediction_anchored_bridge/source_500k_${stamp}}"
  PREDICTION_ANCHORED_ARTIFACT_ROOT="${PAB_NEW_ARTIFACT_ROOT:-${PROJECT_DIR}/checkpoints/prediction_anchored_bridge/full_pilot_${stamp}}"
elif [[ "${PAB_REBUILD_SOURCE}" == "0" ]]; then
  : "${PAB_SOURCE_BASE:?PAB_REBUILD_SOURCE=0 requires an explicit compatible PAB_SOURCE_BASE}"
  : "${PREDICTION_ANCHORED_ARTIFACT_ROOT:=${PROJECT_DIR}/checkpoints/prediction_anchored_bridge/full_pilot_${stamp}}"
else
  echo "PAB_REBUILD_SOURCE must be 0 or 1, got ${PAB_REBUILD_SOURCE}" >&2
  exit 2
fi
: "${PAB_PREFLIGHT_ROOT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/preflight}"
export PAB_SOURCE_BASE PREDICTION_ANCHORED_ARTIFACT_ROOT PAB_PREFLIGHT_ROOT

parent="${PAB_SOURCE_BASE}/inputs/split_manifest/split_manifest.json.gz"
hlt_cache="${PAB_SOURCE_BASE}/inputs/hlt_cache"
offline_cache="${PAB_SOURCE_BASE}/inputs/offline_cache"
baseline="${PAB_SOURCE_BASE}/taggers/A0/best_model_val.pt"

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

source_dependency=""
split_job=""
hlt_job=""
offline_job=""
a0_job=""
if [[ "${PAB_REBUILD_SOURCE}" == "1" ]]; then
  [[ ! -e "${PAB_SOURCE_BASE}" ]] || {
    echo "Fresh 500k source root already exists; refusing replacement: ${PAB_SOURCE_BASE}" >&2
    exit 2
  }

  export DATA_DIR="${PAB_DATA_DIR}"
  export DATA_DIRS="${PAB_DATA_DIR}"
  export MANIFEST_PATH="${parent}"
  export MODEL_TRAIN_SIZE=500000
  export MODEL_VAL_SIZE=150000
  export STACK_TRAIN_SIZE=500000
  export STACK_VAL_SIZE=150000
  export FINAL_TEST_SIZE=150000
  export MAX_CONSTITS=128
  export OVERWRITE=0

  export HLT_CACHE_DIR="${hlt_cache}"
  export HLT_SPLITS="model_train model_val stack_train stack_val final_test"
  export HLT_PROFILE=fixed_hlt_v2_realistic
  export HLT_DEGRADATION_STRENGTH=2.5

  export ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH="${parent}"
  export ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR="${offline_cache}"
  export ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS="model_train model_val stack_train stack_val"
  export ARCHITECTURE_VIEW_10CLASS_OFFLINE_DATA_DIRS="${PAB_DATA_DIR}"
  export ARCHITECTURE_VIEW_10CLASS_OFFLINE_OVERWRITE=0
  export ARCHITECTURE_VIEW_10CLASS_OFFLINE_APPEND=0

  export LOCAL_RESIDUAL_FIELD_ROOT="${PAB_SOURCE_BASE}"
  export LOCAL_RESIDUAL_FIELD_MANIFEST_PATH="${parent}"
  export LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR="${hlt_cache}"
  export LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR="${PAB_SOURCE_BASE}/targets"
  export LOCAL_RESIDUAL_FIELD_RECON_ROOT="${PAB_SOURCE_BASE}/reconstructors"
  export LOCAL_RESIDUAL_FIELD_TAGGER_ROOT="${PAB_SOURCE_BASE}/taggers"
  export DEVICE=cuda

  split_job="$(submit_job source_splits \
    --account="${PAB_SBATCH_ACCOUNT}" --partition="${PAB_SBATCH_PARTITION}" \
    --nodes=1 --job-name=pab_source_splits --time=04:00:00 \
    --mem=64G --cpus-per-task=8 --export=ALL \
    sbatch/run_build_fresh_splits.sh)"
  hlt_job="$(submit_job source_hlt \
    --account="${PAB_SBATCH_ACCOUNT}" --partition="${PAB_SBATCH_PARTITION}" \
    --nodes=1 --job-name=pab_source_hlt --time=1-00:00:00 \
    --mem=240G --cpus-per-task=12 --dependency="afterok:${split_job}" \
    --export=ALL sbatch/run_build_fresh_hlt_cache.sh)"
  offline_job="$(submit_job source_offline \
    --account="${PAB_SBATCH_ACCOUNT}" --partition="${PAB_SBATCH_PARTITION}" \
    --nodes=1 --job-name=pab_source_offline --time=1-00:00:00 \
    --mem=500G --cpus-per-task=16 --dependency="afterok:${split_job}" \
    --export=ALL sbatch/run_cache_architecture_view_offline_inputs.sh)"
  a0_job="$(submit_job source_a0 \
    --account="${PAB_SBATCH_ACCOUNT}" --partition="${PAB_SBATCH_PARTITION}" \
    --nodes=1 --job-name=pab_source_a0 --time=3-00:00:00 \
    --mem=500G --cpus-per-task=16 --gres=gpu:gh200:1 \
    --dependency="afterok:${hlt_job}:${offline_job}" --export=ALL \
    sbatch/run_train_local_residual_field_tagger.sh A0)"
  source_dependency="${a0_job}"
else
  for path in "${parent}" "${baseline}"; do
    [[ -f "${path}" && ! -L "${path}" ]] || {
      echo "Missing reusable source input: ${path}" >&2
      exit 2
    }
  done
  for split in model_train model_val stack_train stack_val final_test; do
    [[ -f "${hlt_cache}/${split}_fixed_hlt.npz" ]] || {
      echo "Reusable HLT cache is incomplete at split ${split}" >&2
      exit 2
    }
  done
  for split in model_train model_val stack_train stack_val; do
    [[ -f "${offline_cache}/${split}_offline.npz" ]] || {
      echo "Reusable offline cache is incomplete at development split ${split}" >&2
      exit 2
    }
  done
fi

dependency=()
if [[ -n "${source_dependency}" ]]; then
  dependency=(--dependency="afterok:${source_dependency}")
fi
finalizer_job="$(submit_job finalizer "${dependency[@]}" \
  --account="${PAB_SBATCH_ACCOUNT}" --partition="${PAB_SBATCH_PARTITION}" \
  --nodes=1 \
  --export=ALL,PYTHONNOUSERSITE=1,PROJECT_DIR="${PROJECT_DIR}",PAB_SOURCE_BASE="${PAB_SOURCE_BASE}",PAB_PREFLIGHT_ROOT="${PAB_PREFLIGHT_ROOT}",PREDICTION_ANCHORED_ARTIFACT_ROOT="${PREDICTION_ANCHORED_ARTIFACT_ROOT}" \
  sbatch/run_finalize_prediction_anchored_bridge_submission.sh)"

printf 'source_mode=%s\n' "$([[ "${PAB_REBUILD_SOURCE}" == "1" ]] && printf REBUILT_500K || printf REUSED_EXPLICIT)"
printf 'source_root=%s\n' "${PAB_SOURCE_BASE}"
printf 'split_job=%s\n' "${split_job:-REUSED_EXISTING}"
printf 'hlt_job=%s\n' "${hlt_job:-REUSED_EXISTING}"
printf 'offline_job=%s\n' "${offline_job:-REUSED_EXISTING}"
printf 'a0_job=%s\n' "${a0_job:-REUSED_EXISTING}"
printf 'finalizer_job=%s\n' "${finalizer_job}"
printf 'artifact_root=%s\n' "${PREDICTION_ANCHORED_ARTIFACT_ROOT}"
printf 'The finalizer submits the validated B0--B6 graph only after its source dependency succeeds.\n'

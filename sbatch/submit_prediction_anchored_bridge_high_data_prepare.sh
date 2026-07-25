#!/usr/bin/env bash
# Queue storage-safe preparation for the 3M/3M prediction-anchored campaign.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PAB_CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${PAB_CONDA_ENV:=atlas_kd_tigris}"
: "${PAB_HIGH_DATA_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${PAB_SBATCH_ACCOUNT:=reu-aisocial}"
: "${PAB_SBATCH_PARTITION:=tigris}"
: "${PAB_HIGH_DATA_SHARD_EVENTS:=100000}"

export PAB_CONDA_BASE PAB_CONDA_ENV
export CONDA_BASE="${PAB_CONDA_BASE}"
export CONDA_ENV="${PAB_CONDA_ENV}"
export PYTHONNOUSERSITE=1

cd "${PROJECT_DIR}"
mkdir -p fresh_check_logs

stamp="$(date -u +%Y%m%d_%H%M%S)"
: "${PAB_HIGH_DATA_ROOT:=${PROJECT_DIR}/checkpoints/prediction_anchored_bridge/high_data_3m_${stamp}}"
: "${PAB_HIGH_DATA_PARENT_MANIFEST:=${PAB_HIGH_DATA_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${PAB_HIGH_DATA_PREFLIGHT_ROOT:=${PAB_HIGH_DATA_ROOT}/preflight}"

for path in "${PAB_HIGH_DATA_ROOT}" "${PAB_HIGH_DATA_PREFLIGHT_ROOT}"; do
  [[ ! -e "${path}" ]] || {
    echo "Refusing to overwrite high-data preparation path: ${path}" >&2
    exit 2
  }
done
[[ -d "${PAB_HIGH_DATA_DATA_DIR}" ]] || {
  echo "High-data JetClass source directory is missing: ${PAB_HIGH_DATA_DATA_DIR}" >&2
  exit 2
}

mkdir -p "${PAB_HIGH_DATA_ROOT}/inputs/split_manifest"

export DATA_DIR="${PAB_HIGH_DATA_DATA_DIR}"
export DATA_DIRS="${PAB_HIGH_DATA_DATA_DIR}"
export MANIFEST_PATH="${PAB_HIGH_DATA_PARENT_MANIFEST}"
export MODEL_TRAIN_SIZE=500000
export MODEL_VAL_SIZE=500000
export STACK_TRAIN_SIZE=6000000
export STACK_VAL_SIZE=500000
export FINAL_TEST_SIZE=1000000
export MAX_CONSTITS=128
export OVERWRITE=0
export PAB_HIGH_DATA_PARENT_MANIFEST
export PAB_HIGH_DATA_PREFLIGHT_ROOT
export PAB_HIGH_DATA_DATA_DIR
export PAB_HIGH_DATA_SHARD_EVENTS

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

split_job="$(submit_job high_data_splits \
  --account="${PAB_SBATCH_ACCOUNT}" \
  --partition="${PAB_SBATCH_PARTITION}" \
  --nodes=1 \
  --job-name=pab_hd_splits \
  --time=12:00:00 \
  --mem=192G \
  --cpus-per-task=16 \
  --export=ALL,PYTHONNOUSERSITE=1 \
  sbatch/run_build_fresh_splits.sh)"

manifest_job="$(submit_job high_data_manifest \
  --account="${PAB_SBATCH_ACCOUNT}" \
  --partition="${PAB_SBATCH_PARTITION}" \
  --nodes=1 \
  --job-name=pab_hd_manifest \
  --time=1-00:00:00 \
  --mem=512G \
  --cpus-per-task=24 \
  --dependency="afterok:${split_job}" \
  --export=ALL,PYTHONNOUSERSITE=1,PAB_CONDA_BASE="${PAB_CONDA_BASE}",PAB_CONDA_ENV="${PAB_CONDA_ENV}",CONDA_BASE="${PAB_CONDA_BASE}",CONDA_ENV="${PAB_CONDA_ENV}",PAB_HIGH_DATA_PARENT_MANIFEST="${PAB_HIGH_DATA_PARENT_MANIFEST}",PAB_HIGH_DATA_PREFLIGHT_ROOT="${PAB_HIGH_DATA_PREFLIGHT_ROOT}",PAB_HIGH_DATA_DATA_DIR="${PAB_HIGH_DATA_DATA_DIR}",PAB_HIGH_DATA_SHARD_EVENTS="${PAB_HIGH_DATA_SHARD_EVENTS}" \
  sbatch/run_prepare_prediction_anchored_high_data_manifest.sh)"

printf 'profile=high_data_3m\n'
printf 'high_data_root=%s\n' "${PAB_HIGH_DATA_ROOT}"
printf 'parent_manifest=%s\n' "${PAB_HIGH_DATA_PARENT_MANIFEST}"
printf 'preflight_root=%s\n' "${PAB_HIGH_DATA_PREFLIGHT_ROOT}"
printf 'split_job=%s\n' "${split_job}"
printf 'manifest_job=%s\n' "${manifest_job}"
printf 'stack_train_consumer=3000000\n'
printf 'stack_train_distill=3000000\n'
printf 'validation_total=1000000\n'
printf 'final_test_sealed=1000000\n'
printf 'r0_disjoint_model_train=500000\n'
printf 'dense_npz_materialization=DEFERRED_TO_MEASURED_FULL_SUBMITTER\n'
printf 'scientific_training_submission=DEFERRED_UNTIL_SOURCE_CACHE_BUILD\n'

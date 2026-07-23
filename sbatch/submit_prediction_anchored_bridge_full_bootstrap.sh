#!/usr/bin/env bash
# Queue a missing offline source and then the complete B0--B6 pilot.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PAB_SOURCE_BASE:=${PROJECT_DIR}/checkpoints/local_particle_residual_field_curriculum/rebuild_and_pilot_20260720_185817}"
: "${PAB_CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${PAB_CONDA_ENV:=atlas_kd_tigris}"
export PAB_CONDA_BASE PAB_CONDA_ENV
export PYTHONNOUSERSITE=1
cd "${PROJECT_DIR}"
mkdir -p fresh_check_logs

stamp="$(date -u +%Y%m%d_%H%M%S)"
: "${PREDICTION_ANCHORED_ARTIFACT_ROOT:=${PROJECT_DIR}/checkpoints/prediction_anchored_bridge/full_pilot_${stamp}}"
: "${PAB_PREFLIGHT_ROOT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/preflight}"

parent="${PAB_SOURCE_BASE}/inputs/split_manifest/split_manifest.json.gz"
hlt_cache="${PAB_SOURCE_BASE}/inputs/hlt_cache"
offline_cache="${PAB_SOURCE_BASE}/inputs/offline_cache"
baseline="${PAB_SOURCE_BASE}/taggers/A0/best_model_val.pt"

for path in "${parent}" "${baseline}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
for split in model_train model_val stack_train stack_val final_test; do
  [[ -f "${hlt_cache}/${split}_fixed_hlt.npz" ]] || {
    echo "Existing HLT cache is incomplete at split ${split}; refusing a guessed rebuild" >&2
    exit 2
  }
done

dependency=()
offline_job=""
if [[ ! -f "${offline_cache}/stack_train_offline.npz" ]]; then
  offline_job="$(sbatch --parsable \
    --account=reu-aisocial --partition=tigris \
    --export=ALL,PYTHONNOUSERSITE=1,PROJECT_DIR="${PROJECT_DIR}",PAB_SOURCE_BASE="${PAB_SOURCE_BASE}" \
    sbatch/run_prediction_anchored_missing_offline_split.sh)"
  offline_job="${offline_job%%;*}"
  [[ "${offline_job}" =~ ^[0-9]+$ ]] || {
    echo "Could not parse numeric offline job ID from sbatch output: ${offline_job}" >&2
    exit 2
  }
  dependency=(--dependency="afterok:${offline_job}")
fi

finalizer_job="$(sbatch --parsable "${dependency[@]}" \
  --export=ALL,PYTHONNOUSERSITE=1,PROJECT_DIR="${PROJECT_DIR}",PAB_SOURCE_BASE="${PAB_SOURCE_BASE}",PAB_PREFLIGHT_ROOT="${PAB_PREFLIGHT_ROOT}",PREDICTION_ANCHORED_ARTIFACT_ROOT="${PREDICTION_ANCHORED_ARTIFACT_ROOT}" \
  sbatch/run_finalize_prediction_anchored_bridge_submission.sh)"
finalizer_job="${finalizer_job%%;*}"
[[ "${finalizer_job}" =~ ^[0-9]+$ ]] || {
  echo "Could not parse numeric finalizer job ID from sbatch output: ${finalizer_job}" >&2
  exit 2
}

printf 'offline_job=%s\n' "${offline_job:-REUSED_EXISTING}"
printf 'finalizer_job=%s\n' "${finalizer_job}"
printf 'artifact_root=%s\n' "${PREDICTION_ANCHORED_ARTIFACT_ROOT}"
printf 'The finalizer submits the complete validated B0--B6 graph after its dependency succeeds.\n'

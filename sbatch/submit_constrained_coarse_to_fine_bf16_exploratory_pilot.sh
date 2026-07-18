#!/usr/bin/env bash
# Submit the entire C2F pilot with the fixed BF16 runtime path, without the
# benchmark/candidate gate.  This is exploratory only: high-data and final
# claims remain restricted to an approved runtime profile.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints}"
: "${CONSTRAINED_C2F_MIN_FREE_GB:=40}"
: "${CONSTRAINED_C2F_REUSE_INPUTS_ROOT:=}"

available_kib="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ ]] || {
  echo "Could not determine free space for ${OUTPUT_ROOT}" >&2
  exit 2
}
minimum_kib=$(( CONSTRAINED_C2F_MIN_FREE_GB * 1024 * 1024 ))
if (( available_kib < minimum_kib )); then
  printf 'Refusing exploratory pilot: %s has %.1f GiB free; at least %s GiB is required.\n' \
    "${OUTPUT_ROOT}" "$(( available_kib / 1024 / 1024 ))" "${CONSTRAINED_C2F_MIN_FREE_GB}" >&2
  exit 2
fi

export CONSTRAINED_C2F_CAMPAIGN_MODE=pilot
export CONSTRAINED_C2F_STAGE_MODE=full
export CONSTRAINED_C2F_RUNTIME_PROFILE=bf16_exploratory_pilot_v1
export CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT=

# A finished exploratory run needs its selected model, not a second resumable
# copy of every model state.  Active jobs can still be restarted from scratch.
export CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT=0
export CONSTRAINED_C2F_RECO_SAVE_BEST_CHECKPOINT=1

if [[ -n "${CONSTRAINED_C2F_REUSE_INPUTS_ROOT}" ]]; then
  reuse_inputs_dir="${CONSTRAINED_C2F_REUSE_INPUTS_ROOT}/inputs"
  reuse_manifest="${reuse_inputs_dir}/split_manifest/split_manifest.json.gz"
  reuse_hlt="${reuse_inputs_dir}/hlt_cache"
  reuse_offline="${reuse_inputs_dir}/offline_cache"
  reuse_targets="${CONSTRAINED_C2F_REUSE_INPUTS_ROOT}/targets"
  for required in \
    "${reuse_manifest}" \
    "${reuse_hlt}/model_train_fixed_hlt_metadata.json" \
    "${reuse_offline}/model_train_offline_metadata.json" \
    "${reuse_targets}/hierarchy_target_cache_manifest.json"; do
    [[ -e "${required}" ]] || { echo "Reusable C2F input is missing: ${required}" >&2; exit 2; }
  done

  export CONSTRAINED_C2F_MANIFEST_PATH="${reuse_manifest}"
  export CONSTRAINED_C2F_HLT_CACHE_DIR="${reuse_hlt}"
  export CONSTRAINED_C2F_OFFLINE_CACHE_DIR="${reuse_offline}"
  export CONSTRAINED_C2F_TARGET_CACHE_DIR="${reuse_targets}"
  export CONSTRAINED_C2F_SUBMIT_SPLITS=0
  export CONSTRAINED_C2F_SUBMIT_HLT_CACHE=0
  export CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE=0
  export CONSTRAINED_C2F_SUBMIT_TARGETS=0
fi

exec bash "${SCRIPT_DIR}/submit_constrained_coarse_to_fine_experiment.sh"

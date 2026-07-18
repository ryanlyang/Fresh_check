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

exec bash "${SCRIPT_DIR}/submit_constrained_coarse_to_fine_experiment.sh"

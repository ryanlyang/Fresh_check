#!/usr/bin/env bash
# Submit the full accelerated pilot after the immutable candidate profile is written.

#SBATCH --job-name=c2f_rt_pilot
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH:?CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH is required}"
: "${CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT:?CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT is required}"
: "${CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION:?CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION is required}"

fresh_setup "$@"
fresh_require_file "${CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH}"
if [[ -e "${CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT}" ]]; then
  echo "Refusing to reuse existing accelerated pilot root: ${CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT}" >&2
  exit 2
fi

CONSTRAINED_C2F_CAMPAIGN_MODE=pilot \
CONSTRAINED_C2F_STAGE_MODE=full \
CONSTRAINED_C2F_RUNTIME_PROFILE=accelerated_candidate_v1 \
CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT="${CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH}" \
CONSTRAINED_C2F_ROOT="${CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT}" \
bash "${SCRIPT_DIR}/submit_constrained_coarse_to_fine_experiment.sh"

printf 'pilot_graph_submitted\t%s\n' "$(date -Is)" >> "${CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION}"

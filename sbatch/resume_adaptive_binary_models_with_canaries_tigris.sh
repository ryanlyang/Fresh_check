#!/usr/bin/env bash
# Compatibility entrypoint for the direct, preflighted production model resume.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${ABPH_ROOT:?Set ABPH_ROOT to the prepared streaming campaign root}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"

[[ "${ABPH_CONFIRM_CANARY_MODELS_RESUME:-0}" == "1" ]] || {
  echo "Set ABPH_CONFIRM_CANARY_MODELS_RESUME=1 to resume production models." >&2
  exit 2
}

export PROJECT_DIR
export CONDA_BASE
export CONDA_ENV
export PYTHONNOUSERSITE=1
export ABPH_ROOT
export ABPH_DATA_DIR
export ABPH_SBATCH_ACCOUNT
export ABPH_SBATCH_PARTITION
export ABPH_CONFIRM_MODELS_RESUME=1

# The DDP8 launch path has already completed a real update and distributed
# validation. Delegate to the production resume, whose dry-run preflight
# finishes before it cancels or submits any model jobs.
exec bash "${PROJECT_DIR}/sbatch/resume_adaptive_binary_models_from_contracts_tigris.sh"

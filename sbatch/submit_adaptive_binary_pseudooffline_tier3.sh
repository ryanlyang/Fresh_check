#!/usr/bin/env bash
# Submit any ABPH stage from sporcsubmit to tier3 GPUs.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${CONDA_ENV:=atlas_kd}"
: "${PYTHONNOUSERSITE:=1}"
export PROJECT_DIR ABPH_DATA_DIR CONDA_ENV PYTHONNOUSERSITE
export ABPH_CLUSTER=tier3
unset ABPH_SBATCH_ACCOUNT
if [[ "${CONDA_BASE:-}" == *miniforge3-aarch64* ]]; then unset CONDA_BASE; fi
exec bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_pseudooffline.sh" "$@"

#!/usr/bin/env bash
# Submit any ABPH stage to RIT Tigris GH200 nodes.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${PYTHONNOUSERSITE:=1}"
export PROJECT_DIR ABPH_DATA_DIR CONDA_BASE CONDA_ENV ABPH_SBATCH_ACCOUNT PYTHONNOUSERSITE
export ABPH_CLUSTER=tigris
exec bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_pseudooffline.sh" "$@"

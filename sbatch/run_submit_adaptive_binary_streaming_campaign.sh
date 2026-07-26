#!/usr/bin/env bash
# Submit the fresh 30 GB pilot after the external bootstrap gate succeeds.

#SBATCH --job-name=abph_bootstrap_continue
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
CONDA_BASE="${ABPH_CONDA_BASE:-/home/ryreu/miniforge3-aarch64}"
CONDA_ENV="${ABPH_CONDA_ENV:-atlas_kd_tigris}"
export CONDA_BASE CONDA_ENV
source "${PROJECT_DIR}/sbatch/common.sh"
: "${ABPH_BOOTSTRAP_CAMPAIGN_ROOT:?Set the fresh campaign root}"
: "${ABPH_BOOTSTRAP_STORAGE_PROJECTION:?Set the root-bound projection}"
: "${ABPH_BOOTSTRAP_RUNTIME_ACCEPTANCE:?Set the accepted runtime artifact}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
export PYTHONNOUSERSITE=1
fresh_setup

"${PYTHON_BIN}" - <<'PY'
import os
from teacher_logit_reco.adaptive_binary_pseudooffline.runtime_acceptance import require_runtime_acceptance
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import require_storage_projection

require_runtime_acceptance(os.environ["ABPH_BOOTSTRAP_RUNTIME_ACCEPTANCE"], scope="ddp4_runtime")
require_storage_projection(
    os.environ["ABPH_BOOTSTRAP_STORAGE_PROJECTION"],
    campaign_root=os.environ["ABPH_BOOTSTRAP_CAMPAIGN_ROOT"],
    campaign_mode="pilot",
    profile="streaming_30gb_v1",
)
PY

export ABPH_ROOT="${ABPH_BOOTSTRAP_CAMPAIGN_ROOT}"
export ABPH_CAMPAIGN_MODE=pilot
export ABPH_STAGE_MODE=full
export ABPH_RECONSTRUCTOR_PARALLELISM=ddp4
export ABPH_STORAGE_PROFILE=streaming_30gb_v1
export ABPH_STORAGE_PROJECTION_PATH="${ABPH_BOOTSTRAP_STORAGE_PROJECTION}"
export ABPH_RUNTIME_ACCEPTANCE_PATH="${ABPH_BOOTSTRAP_RUNTIME_ACCEPTANCE}"
export CONFIRM_FINAL_TEST=0
exec bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_pseudooffline_streaming30gb_tigris.sh"

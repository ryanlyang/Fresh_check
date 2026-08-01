#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${HOSD_LAUNCHER_ROOT:=${PROJECT_DIR}}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

hosd_setup() {
  : "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
  : "${HOSD_NODE_ID:?HOSD_NODE_ID is required}"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  cd "${PROJECT_DIR}"
  python -s -c 'import awkward_cpp, awkward, uproot; print("HOSD runtime OK", uproot.__version__, awkward.__version__)'
  python -c 'import sys; from pathlib import Path; from teacher_logit_reco.hlt_offline_structure_distillation import load_and_validate_campaign; load_and_validate_campaign(Path(sys.argv[1]), repo_root=Path(sys.argv[2])); print("HOSD source validation OK")' "${CAMPAIGN_ROOT}" "${PROJECT_DIR}"
}

hosd_run_registered_node() {
  hosd_setup
  python "${HOSD_LAUNCHER_ROOT}/scripts/run_hosd_registered_node.py" \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --campaign-source-root "${PROJECT_DIR}" \
    --node-id "${HOSD_NODE_ID}" \
    --coordinate "${SLURM_ARRAY_TASK_ID:-0}"
}

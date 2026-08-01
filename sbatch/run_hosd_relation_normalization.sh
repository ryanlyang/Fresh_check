#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HOSD_LAUNCHER_ROOT:=${PROJECT_DIR}}"
source "${HOSD_LAUNCHER_ROOT}/sbatch/hosd_common.sh"
hosd_run_registered_node

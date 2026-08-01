#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/hosd_common.sh"
hosd_run_registered_node

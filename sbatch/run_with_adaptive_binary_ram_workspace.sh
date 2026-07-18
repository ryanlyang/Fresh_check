#!/usr/bin/env bash
# Execute one worker command inside a verified rank-local tmpfs workspace.

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/sbatch/adaptive_binary_ram_workspace.sh"
abph_setup_ram_workspace
"$@"

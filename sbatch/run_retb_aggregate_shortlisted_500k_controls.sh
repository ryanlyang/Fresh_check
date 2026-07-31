#!/usr/bin/env bash
#SBATCH --job-name=retb_500k_control_aggregate
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retb_task_common.sh"
retb_run_task "shortlisted_500k_controls"

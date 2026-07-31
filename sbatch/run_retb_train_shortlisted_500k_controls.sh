#!/usr/bin/env bash
#SBATCH --job-name=retb_500k_controls
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=7-00:00:00
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retb_task_common.sh"
retb_run_task "shortlisted_500k_control_training"

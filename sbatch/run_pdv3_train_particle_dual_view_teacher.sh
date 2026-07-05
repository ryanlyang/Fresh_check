#!/usr/bin/env bash
#SBATCH --job-name=pdv3_pdv_t
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=4-00:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=pdv3_pd10_env.sh
source "${SCRIPT_DIR}/pdv3_pd10_env.sh"

exec bash "${SCRIPT_DIR}/run_pd10_train_particle_dual_view_teacher.sh" "$@"

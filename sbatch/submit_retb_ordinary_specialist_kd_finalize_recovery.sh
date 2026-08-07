#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retb_common.sh"

: "${RETB_ORDINARY_SPECIALIST_KD_ROOT:?Set the completed ordinary specialist-KD root}"

if [[ "${RETB_ORDINARY_SPECIALIST_KD_RECOVERY_FROZEN_REENTRY:-0}" != "1" ]]; then
  submission_project_dir="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel)"
  source_commit="$(git -C "${submission_project_dir}" rev-parse HEAD)"
  : "${RETB_SOURCE_WORKTREE_ROOT:=${submission_project_dir%/*}/retb_source_worktrees}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  frozen="${RETB_SOURCE_WORKTREE_ROOT}/retb_ordinary_kd_finalize_recovery_${stamp}_${source_commit:0:10}_$$"
  mkdir -p "${RETB_SOURCE_WORKTREE_ROOT}"
  git -C "${submission_project_dir}" worktree add --detach "${frozen}" "${source_commit}" >/dev/null
  exec env \
    PROJECT_DIR="${frozen}" OUTPUT_ROOT="${OUTPUT_ROOT}" DATA_DIR="${DATA_DIR}" \
    CONDA_BASE="${CONDA_BASE}" CONDA_ENV="${CONDA_ENV}" \
    SBATCH_ACCOUNT="${SBATCH_ACCOUNT}" SBATCH_PARTITION="${SBATCH_PARTITION}" \
    RETB_ORDINARY_SPECIALIST_KD_ROOT="${RETB_ORDINARY_SPECIALIST_KD_ROOT}" \
    RETB_ORDINARY_SPECIALIST_KD_RECOVERY_FROZEN_REENTRY=1 \
    RETB_FROZEN_REENTRY=1 RETB_FROZEN_SOURCE_COMMIT="${source_commit}" \
    RETB_SOURCE_WORKTREE_ROOT="${RETB_SOURCE_WORKTREE_ROOT}" \
    bash "${frozen}/sbatch/submit_retb_ordinary_specialist_kd_finalize_recovery.sh"
fi

retb_activate
root="$(cd -- "${RETB_ORDINARY_SPECIALIST_KD_ROOT}" && pwd)"
plan="${root}/registry/ordinary_specialist_kd_plan.json"
test -f "${plan}"

for condition in MATCHED_KD HYBRID_KD; do
  for expert in BASE4 PT TRACK REGION; do
    test -f "${root}/runs/students/${condition}/${expert}/result.json"
  done
done

log="${root}/job_ledgers/slurm/%x_%A_%a.out"
err="${root}/job_ledgers/slurm/%x_%A_%a.err"
exported="ALL,PROJECT_DIR=${PROJECT_DIR},RETB_ORDINARY_SPECIALIST_KD_ROOT=${root},RETB_ORDINARY_SPECIALIST_KD_FINALIZER_RECOVERY=1,RETB_FROZEN_REENTRY=1,RETB_FROZEN_SOURCE_COMMIT=${RETB_FROZEN_SOURCE_COMMIT}"

job_id="$(sbatch --parsable \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --cpus-per-task=4 --mem=32G --time=02:00:00 \
  --job-name=retb_ord_kd_finalize_recovery \
  --output="${log}" --error="${err}" \
  --export="${exported}" \
  "${PROJECT_DIR}/sbatch/run_retb_ordinary_specialist_kd_finalize.sh")"

printf 'Ordinary specialist-KD finalizer recovery job: %s\n' "${job_id}"

#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retb_common.sh"

: "${RETB_SUPP_KD_RECOVERY_ROOT:?Set RETB_SUPP_KD_RECOVERY_ROOT to the failed supplemental root}"
: "${RETB_SUPP_KD_ORIGINAL_ARRAY_JOB_ID:?Set RETB_SUPP_KD_ORIGINAL_ARRAY_JOB_ID to the original six-task array job}"
: "${RETB_SUPP_KD_GPU_CPUS:=8}"
: "${RETB_SUPP_KD_GPU_MEM:=220G}"

if [[ "${RETB_SUPP_KD_RECOVERY_FROZEN_REENTRY:-0}" != "1" ]]; then
  submission_project_dir="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel)"
  source_commit="$(git -C "${submission_project_dir}" rev-parse HEAD)"
  : "${RETB_SOURCE_WORKTREE_ROOT:=${submission_project_dir%/*}/retb_source_worktrees}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  frozen="${RETB_SOURCE_WORKTREE_ROOT}/retb_supp_kd_recovery_${stamp}_${source_commit:0:10}_$$"
  mkdir -p "${RETB_SOURCE_WORKTREE_ROOT}"
  git -C "${submission_project_dir}" worktree add --detach "${frozen}" "${source_commit}" >/dev/null
  exec env \
    PROJECT_DIR="${frozen}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DATA_DIR="${DATA_DIR}" \
    CONDA_BASE="${CONDA_BASE}" \
    CONDA_ENV="${CONDA_ENV}" \
    SBATCH_ACCOUNT="${SBATCH_ACCOUNT}" \
    SBATCH_PARTITION="${SBATCH_PARTITION}" \
    GPU_GRES="${GPU_GRES}" \
    RETB_SUPP_KD_RECOVERY_ROOT="${RETB_SUPP_KD_RECOVERY_ROOT}" \
    RETB_SUPP_KD_ORIGINAL_ARRAY_JOB_ID="${RETB_SUPP_KD_ORIGINAL_ARRAY_JOB_ID}" \
    RETB_SUPP_KD_GPU_CPUS="${RETB_SUPP_KD_GPU_CPUS}" \
    RETB_SUPP_KD_GPU_MEM="${RETB_SUPP_KD_GPU_MEM}" \
    RETB_FROZEN_REENTRY=1 \
    RETB_FROZEN_SOURCE_COMMIT="${source_commit}" \
    RETB_SUPP_KD_RECOVERY_FROZEN_REENTRY=1 \
    RETB_SOURCE_WORKTREE_ROOT="${RETB_SOURCE_WORKTREE_ROOT}" \
    bash "${frozen}/sbatch/submit_retb_supplemental_kd_recovery.sh"
fi

retb_activate
root="$(realpath "${RETB_SUPP_KD_RECOVERY_ROOT}")"
plan="${root}/registry/kd_baseline_plan.json"
test -f "${plan}" || {
  echo "Supplemental KD plan is absent: ${plan}" >&2
  exit 2
}
test -d "${root}/runs" || {
  echo "Supplemental KD runs directory is absent: ${root}/runs" >&2
  exit 2
}

common_export="ALL,PROJECT_DIR=${PROJECT_DIR},RETB_SUPP_KD_ROOT=${root},RETB_SUPP_KD_RECOVERY_MODE=1,RETB_FROZEN_REENTRY=1,RETB_FROZEN_SOURCE_COMMIT=${RETB_FROZEN_SOURCE_COMMIT}"
log="${root}/job_ledgers/slurm/%x_%A_%a.out"
err="${root}/job_ledgers/slurm/%x_%A_%a.err"

# The original tasks 0-2 are conventional O_BASE KD and remain valid.  Only
# tasks 3-5 (O_FULLREL) are replayed with authenticated REGION resources.
recovery="$(sbatch --parsable --array=3-5 \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --gres="${GPU_GRES}" --cpus-per-task="${RETB_SUPP_KD_GPU_CPUS}" \
  --mem="${RETB_SUPP_KD_GPU_MEM}" \
  --job-name="retb_supp_kd_fullrel_recovery" \
  --output="${log}" --error="${err}" \
  --export="${common_export}" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_kd_baseline_row.sh")"

# afterany on the original array waits for O_BASE tasks 0-2 even though its
# O_FULLREL tasks failed.  afterok on recovery requires all corrected tasks.
finalize="$(sbatch --parsable \
  --dependency="afterany:${RETB_SUPP_KD_ORIGINAL_ARRAY_JOB_ID},afterok:${recovery}" \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --cpus-per-task=4 --mem=32G \
  --job-name="retb_supp_kd_recovery_finalize" \
  --output="${log}" --error="${err}" \
  --export="${common_export}" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_kd_baseline_finalize.sh")"

python - "${root}" "${RETB_SUPP_KD_ORIGINAL_ARRAY_JOB_ID}" "${recovery}" "${finalize}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "supplemental_root": str(root),
    "original_array_job_id": sys.argv[2],
    "fullrel_recovery_array_job_id": sys.argv[3],
    "recovery_finalize_job_id": sys.argv[4],
    "reused_original_task_ids": [0, 1, 2],
    "replayed_task_ids": [3, 4, 5],
}
(root / "job_ledgers/recovery_submission.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

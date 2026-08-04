#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retb_common.sh"

: "${RETB_SUPPLEMENTAL_ROOT:?Set RETB_SUPPLEMENTAL_ROOT to the existing supplemental root}"
: "${RETB_READY_FUSION_JOB_ID:?Set RETB_READY_FUSION_JOB_ID to the ready fusion array ID}"
: "${RETB_LATE_FUSION_JOB_ID:?Set RETB_LATE_FUSION_JOB_ID to the late fusion array ID}"
: "${RETB_SUPPLEMENTAL_GPU_CPUS:=8}"
: "${RETB_SUPPLEMENTAL_OBASE_MEM:=220G}"

if [[ "${RETB_SUPPLEMENTAL_FROZEN_REENTRY:-0}" != "1" ]]; then
  submission_project_dir="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel)"
  source_commit="$(git -C "${submission_project_dir}" rev-parse HEAD)"
  : "${RETB_SOURCE_WORKTREE_ROOT:=${submission_project_dir%/*}/retb_source_worktrees}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  frozen="${RETB_SOURCE_WORKTREE_ROOT}/retb_supp_obase_recovery_${stamp}_${source_commit:0:10}_$$"
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
    RETB_SUPPLEMENTAL_ROOT="${RETB_SUPPLEMENTAL_ROOT}" \
    RETB_READY_FUSION_JOB_ID="${RETB_READY_FUSION_JOB_ID}" \
    RETB_LATE_FUSION_JOB_ID="${RETB_LATE_FUSION_JOB_ID}" \
    RETB_SUPPLEMENTAL_GPU_CPUS="${RETB_SUPPLEMENTAL_GPU_CPUS}" \
    RETB_SUPPLEMENTAL_OBASE_MEM="${RETB_SUPPLEMENTAL_OBASE_MEM}" \
    RETB_FROZEN_REENTRY=1 \
    RETB_FROZEN_SOURCE_COMMIT="${source_commit}" \
    RETB_SUPPLEMENTAL_FROZEN_REENTRY=1 \
    RETB_SOURCE_WORKTREE_ROOT="${RETB_SOURCE_WORKTREE_ROOT}" \
    bash "${frozen}/sbatch/submit_retb_supplemental_obase7_recovery.sh"
fi

retb_activate
test -f "${RETB_SUPPLEMENTAL_ROOT}/registry/ready_plan.json" || {
  echo "Existing supplemental ready plan is absent." >&2
  exit 2
}
test -f "${RETB_SUPPLEMENTAL_ROOT}/registry/late_plan.json" || {
  echo "Existing supplemental late plan is absent." >&2
  exit 2
}

source_commit="$(git rev-parse HEAD)"
common_export="ALL,PROJECT_DIR=${PROJECT_DIR},RETB_SUPPLEMENTAL_ROOT=${RETB_SUPPLEMENTAL_ROOT},RETB_FROZEN_REENTRY=1,RETB_FROZEN_SOURCE_COMMIT=${RETB_FROZEN_SOURCE_COMMIT}"
log="${RETB_SUPPLEMENTAL_ROOT}/job_ledgers/slurm/%x_%A_%a.out"
err="${RETB_SUPPLEMENTAL_ROOT}/job_ledgers/slurm/%x_%A_%a.err"

obase="$(sbatch --parsable --array=0-6 \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --gres="${GPU_GRES}" --cpus-per-task="${RETB_SUPPLEMENTAL_GPU_CPUS}" \
  --mem="${RETB_SUPPLEMENTAL_OBASE_MEM}" \
  --job-name="retb_supp_obase7_recovery_${source_commit:0:10}" \
  --output="${log}" --error="${err}" \
  --export="${common_export}" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_obase7_member.sh")"

finalize="$(sbatch --parsable \
  --dependency="afterok:${RETB_READY_FUSION_JOB_ID}:${RETB_LATE_FUSION_JOB_ID}:${obase}" \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --cpus-per-task=4 --mem=32G \
  --job-name="retb_supp_finalize_recovery_${source_commit:0:10}" \
  --output="${log}" --error="${err}" \
  --export="${common_export}" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_offline_fusion_finalize.sh")"

python - "${RETB_SUPPLEMENTAL_ROOT}" "${obase}" "${finalize}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
payload = {
    "supplemental_root": str(root),
    "obase7_recovery_array_job_id": sys.argv[2],
    "recovery_finalize_job_id": sys.argv[3],
}
path = root / "job_ledgers" / f"obase7_recovery_{sys.argv[2]}.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

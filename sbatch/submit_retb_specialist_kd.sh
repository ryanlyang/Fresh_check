#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retb_common.sh"

: "${RETB_PARENT_CAMPAIGN_ROOT:?Set the completed production RETB parent}"
: "${RETB_COMMON_FUSION_ROOT:?Set the completed CE4/KD4 supplemental root}"
: "${RETB_SPECIALIST_KD_GPU_CPUS:=8}"
: "${RETB_SPECIALIST_KD_GPU_MEM:=96G}"

if [[ "${RETB_SPECIALIST_KD_FROZEN_REENTRY:-0}" != "1" ]]; then
  submission_project_dir="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel)"
  source_commit="$(git -C "${submission_project_dir}" rev-parse HEAD)"
  : "${RETB_SOURCE_WORKTREE_ROOT:=${submission_project_dir%/*}/retb_source_worktrees}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  frozen="${RETB_SOURCE_WORKTREE_ROOT}/retb_specialist_kd_${stamp}_${source_commit:0:10}_$$"
  mkdir -p "${RETB_SOURCE_WORKTREE_ROOT}"
  git -C "${submission_project_dir}" worktree add --detach "${frozen}" "${source_commit}" >/dev/null
  exec env \
    PROJECT_DIR="${frozen}" OUTPUT_ROOT="${OUTPUT_ROOT}" DATA_DIR="${DATA_DIR}" \
    CONDA_BASE="${CONDA_BASE}" CONDA_ENV="${CONDA_ENV}" \
    SBATCH_ACCOUNT="${SBATCH_ACCOUNT}" SBATCH_PARTITION="${SBATCH_PARTITION}" \
    GPU_GRES="${GPU_GRES}" \
    RETB_PARENT_CAMPAIGN_ROOT="${RETB_PARENT_CAMPAIGN_ROOT}" \
    RETB_COMMON_FUSION_ROOT="${RETB_COMMON_FUSION_ROOT}" \
    RETB_SPECIALIST_KD_GPU_CPUS="${RETB_SPECIALIST_KD_GPU_CPUS}" \
    RETB_SPECIALIST_KD_GPU_MEM="${RETB_SPECIALIST_KD_GPU_MEM}" \
    RETB_SPECIALIST_KD_FROZEN_REENTRY=1 \
    RETB_FROZEN_REENTRY=1 RETB_FROZEN_SOURCE_COMMIT="${source_commit}" \
    RETB_SOURCE_WORKTREE_ROOT="${RETB_SOURCE_WORKTREE_ROOT}" \
    bash "${frozen}/sbatch/submit_retb_specialist_kd.sh"
fi

retb_activate
source_commit="$(git rev-parse HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
supplemental_id="retb_specialist_kd_${timestamp}_${source_commit:0:10}"
root="${OUTPUT_ROOT}/relation_expert_token_bridge_supplemental/${supplemental_id}"
mkdir -p "${root}/registry" "${root}/runs/teachers" "${root}/runs/students" \
  "${root}/reports" "${root}/job_ledgers/slurm"
exported="ALL,PROJECT_DIR=${PROJECT_DIR},RETB_SPECIALIST_KD_ROOT=${root},RETB_SPECIALIST_KD_ID=${supplemental_id},RETB_PARENT_CAMPAIGN_ROOT=${RETB_PARENT_CAMPAIGN_ROOT},RETB_COMMON_FUSION_ROOT=${RETB_COMMON_FUSION_ROOT},RETB_FROZEN_REENTRY=1,RETB_FROZEN_SOURCE_COMMIT=${RETB_FROZEN_SOURCE_COMMIT}"
log="${root}/job_ledgers/slurm/%x_%A_%a.out"
err="${root}/job_ledgers/slurm/%x_%A_%a.err"

bootstrap="$(sbatch --parsable --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --cpus-per-task=2 --mem=8G --time=01:00:00 \
  --job-name=retb_spec_kd_boot --output="${log}" --error="${err}" \
  --export="${exported}" "${PROJECT_DIR}/sbatch/run_retb_specialist_kd_bootstrap.sh")"

teachers="$(sbatch --parsable --dependency="afterok:${bootstrap}" --array=0-2 \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" --gres="${GPU_GRES}" \
  --cpus-per-task="${RETB_SPECIALIST_KD_GPU_CPUS}" --mem="${RETB_SPECIALIST_KD_GPU_MEM}" \
  --time=2-00:00:00 --job-name=retb_spec_teacher --output="${log}" --error="${err}" \
  --export="${exported}" "${PROJECT_DIR}/sbatch/run_retb_specialist_teacher.sh")"

base4_students="$(sbatch --parsable --dependency="afterok:${bootstrap}" --array=0,4 \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" --gres="${GPU_GRES}" \
  --cpus-per-task="${RETB_SPECIALIST_KD_GPU_CPUS}" --mem="${RETB_SPECIALIST_KD_GPU_MEM}" \
  --time=2-00:00:00 --job-name=retb_spec_student --output="${log}" --error="${err}" \
  --export="${exported}" "${PROJECT_DIR}/sbatch/run_retb_specialist_kd_student.sh")"

relation_students="$(sbatch --parsable --dependency="afterok:${teachers}" --array=1-3,5-7 \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" --gres="${GPU_GRES}" \
  --cpus-per-task="${RETB_SPECIALIST_KD_GPU_CPUS}" --mem="${RETB_SPECIALIST_KD_GPU_MEM}" \
  --time=2-00:00:00 --job-name=retb_spec_student --output="${log}" --error="${err}" \
  --export="${exported}" "${PROJECT_DIR}/sbatch/run_retb_specialist_kd_student.sh")"

finalize="$(sbatch --parsable --dependency="afterok:${base4_students}:${relation_students}" \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --cpus-per-task=4 --mem=32G --time=02:00:00 \
  --job-name=retb_spec_finalize --output="${log}" --error="${err}" \
  --export="${exported}" "${PROJECT_DIR}/sbatch/run_retb_specialist_kd_finalize.sh")"

python - "${root}" "${bootstrap}" "${teachers}" "${base4_students}" "${relation_students}" "${finalize}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
payload = {
    "supplemental_root": str(root),
    "bootstrap_job_id": sys.argv[2],
    "teacher_array_job_id": sys.argv[3],
    "base4_student_array_job_id": sys.argv[4],
    "relation_student_array_job_id": sys.argv[5],
    "finalize_job_id": sys.argv[6],
}
(root / "job_ledgers/submission.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

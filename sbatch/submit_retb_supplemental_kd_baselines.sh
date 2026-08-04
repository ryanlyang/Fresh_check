#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retb_common.sh"

: "${RETB_PARENT_CAMPAIGN_ROOT:?Set RETB_PARENT_CAMPAIGN_ROOT to the production campaign}"
: "${RETB_SUPP_KD_GPU_CPUS:=8}"
: "${RETB_SUPP_KD_GPU_MEM:=220G}"

if [[ "${RETB_SUPP_KD_FROZEN_REENTRY:-0}" != "1" ]]; then
  submission_project_dir="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel)"
  source_commit="$(git -C "${submission_project_dir}" rev-parse HEAD)"
  : "${RETB_SOURCE_WORKTREE_ROOT:=${submission_project_dir%/*}/retb_source_worktrees}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  frozen="${RETB_SOURCE_WORKTREE_ROOT}/retb_supp_kd_${stamp}_${source_commit:0:10}_$$"
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
    RETB_PARENT_CAMPAIGN_ROOT="${RETB_PARENT_CAMPAIGN_ROOT}" \
    RETB_SUPP_KD_GPU_CPUS="${RETB_SUPP_KD_GPU_CPUS}" \
    RETB_SUPP_KD_GPU_MEM="${RETB_SUPP_KD_GPU_MEM}" \
    RETB_FROZEN_REENTRY=1 \
    RETB_FROZEN_SOURCE_COMMIT="${source_commit}" \
    RETB_SUPP_KD_FROZEN_REENTRY=1 \
    RETB_SOURCE_WORKTREE_ROOT="${RETB_SOURCE_WORKTREE_ROOT}" \
    bash "${frozen}/sbatch/submit_retb_supplemental_kd_baselines.sh"
fi

retb_activate
test -f "${RETB_PARENT_CAMPAIGN_ROOT}/campaign_spec.json" || {
  echo "Parent RETB campaign is absent." >&2
  exit 2
}
source_commit="$(git rev-parse HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
supplemental_id="retb_supplemental_kd_baselines_${timestamp}_${source_commit:0:10}"
root="${OUTPUT_ROOT}/relation_expert_token_bridge_supplemental/${supplemental_id}"
mkdir -p "${root}/registry" "${root}/runs" "${root}/reports" "${root}/job_ledgers/slurm"
common_export="ALL,PROJECT_DIR=${PROJECT_DIR},RETB_SUPP_KD_ROOT=${root},RETB_SUPP_KD_ID=${supplemental_id},RETB_PARENT_CAMPAIGN_ROOT=${RETB_PARENT_CAMPAIGN_ROOT},RETB_FROZEN_REENTRY=1,RETB_FROZEN_SOURCE_COMMIT=${RETB_FROZEN_SOURCE_COMMIT}"
log="${root}/job_ledgers/slurm/%x_%A_%a.out"
err="${root}/job_ledgers/slurm/%x_%A_%a.err"

bootstrap="$(sbatch --parsable \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --cpus-per-task=2 --mem=8G \
  --job-name="retb_supp_kd_bootstrap" --output="${log}" --error="${err}" \
  --export="${common_export}" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_kd_baseline_bootstrap.sh")"

training="$(sbatch --parsable --dependency="afterok:${bootstrap}" --array=0-5 \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --gres="${GPU_GRES}" --cpus-per-task="${RETB_SUPP_KD_GPU_CPUS}" \
  --mem="${RETB_SUPP_KD_GPU_MEM}" \
  --job-name="retb_supp_kd_train" --output="${log}" --error="${err}" \
  --export="${common_export}" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_kd_baseline_row.sh")"

finalize="$(sbatch --parsable --dependency="afterok:${training}" \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --cpus-per-task=4 --mem=32G \
  --job-name="retb_supp_kd_finalize" --output="${log}" --error="${err}" \
  --export="${common_export}" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_kd_baseline_finalize.sh")"

python - "${root}" "${bootstrap}" "${training}" "${finalize}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
payload = {
    "supplemental_root": str(root),
    "bootstrap_job_id": sys.argv[2],
    "training_array_job_id": sys.argv[3],
    "finalize_job_id": sys.argv[4],
}
(root / "job_ledgers/submission.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

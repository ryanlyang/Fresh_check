#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retb_common.sh"

: "${RETB_PARENT_CAMPAIGN_ROOT:?Set RETB_PARENT_CAMPAIGN_ROOT to the completed production campaign}"
: "${RETB_REGION_KD_JOB_ID:=}"
: "${RETB_SUPPLEMENTAL_GPU_CPUS:=8}"
: "${RETB_SUPPLEMENTAL_FUSION_MEM:=220G}"
: "${RETB_SUPPLEMENTAL_OBASE_MEM:=220G}"

if [[ "${RETB_SUPPLEMENTAL_FROZEN_REENTRY:-0}" != "1" ]]; then
  submission_project_dir="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel)"
  source_commit="$(git -C "${submission_project_dir}" rev-parse HEAD)"
  : "${RETB_SOURCE_WORKTREE_ROOT:=${submission_project_dir%/*}/retb_source_worktrees}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  frozen="${RETB_SOURCE_WORKTREE_ROOT}/retb_supp_${stamp}_${source_commit:0:10}_$$"
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
    RETB_REGION_KD_JOB_ID="${RETB_REGION_KD_JOB_ID}" \
    RETB_SUPPLEMENTAL_GPU_CPUS="${RETB_SUPPLEMENTAL_GPU_CPUS}" \
    RETB_SUPPLEMENTAL_FUSION_MEM="${RETB_SUPPLEMENTAL_FUSION_MEM}" \
    RETB_SUPPLEMENTAL_OBASE_MEM="${RETB_SUPPLEMENTAL_OBASE_MEM}" \
    RETB_FROZEN_REENTRY=1 \
    RETB_FROZEN_SOURCE_COMMIT="${source_commit}" \
    RETB_SUPPLEMENTAL_FROZEN_REENTRY=1 \
    RETB_SOURCE_WORKTREE_ROOT="${RETB_SOURCE_WORKTREE_ROOT}" \
    bash "${frozen}/sbatch/submit_retb_supplemental_offline_fusion.sh"
fi

retb_activate
test -f "${RETB_PARENT_CAMPAIGN_ROOT}/campaign_spec.json" || {
  echo "Parent campaign_spec.json is absent." >&2
  exit 2
}

# If KD4 is not complete yet, the caller must bind the exact producing array
# task.  This is a runtime dependency only; the bootstrap later byte-seals the
# resulting registration and checkpoint.
kd4_ready="$(python - "${RETB_PARENT_CAMPAIGN_ROOT}" <<'PY'
from pathlib import Path
import sys
from teacher_logit_reco.relation_expert_token_bridge.contracts import load_hashed_json
from teacher_logit_reco.relation_expert_token_bridge.step4 import resolve_stage_b_run
from teacher_logit_reco.relation_expert_token_bridge.supplemental_offline_fusion import resolve_bank_parent
root = Path(sys.argv[1])
registry = load_hashed_json(root / "registry/retb_stage_b_runs.json")
row = resolve_bank_parent(registry, expert_id="REGION", loss_id="ELOSS_KD_DOMINANT")
path = root / "runs/stage_b" / row["run_id"] / "seed_101/checkpoint_registration.json"
print("1" if path.is_file() else "0")
PY
)"
dependency_arguments=()
if [[ "${kd4_ready}" != "1" ]]; then
  if [[ -z "${RETB_REGION_KD_JOB_ID}" ]]; then
    echo "REGION KD parent is pending; set RETB_REGION_KD_JOB_ID (for example 34884_110)." >&2
    exit 2
  fi
  dependency_arguments+=(--dependency="afterok:${RETB_REGION_KD_JOB_ID}")
fi

source_commit="$(git rev-parse HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
supplemental_id="retb_supplemental_offline_fusion_${timestamp}_${source_commit:0:10}"
root="${OUTPUT_ROOT}/relation_expert_token_bridge_supplemental/${supplemental_id}"
mkdir -p "${root}/registry" "${root}/runs" "${root}/reports" "${root}/job_ledgers/slurm"
export RETB_SUPPLEMENTAL_ROOT="${root}"
export RETB_SUPPLEMENTAL_ID="${supplemental_id}"

common_export="ALL,PROJECT_DIR=${PROJECT_DIR},RETB_SUPPLEMENTAL_ROOT=${root},RETB_SUPPLEMENTAL_ID=${supplemental_id},RETB_PARENT_CAMPAIGN_ROOT=${RETB_PARENT_CAMPAIGN_ROOT},RETB_FROZEN_REENTRY=1,RETB_FROZEN_SOURCE_COMMIT=${RETB_FROZEN_SOURCE_COMMIT}"
log="${root}/job_ledgers/slurm/%x_%A_%a.out"
err="${root}/job_ledgers/slurm/%x_%A_%a.err"

ready_bootstrap="$(sbatch --parsable \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --cpus-per-task=2 --mem=8G \
  --job-name="${supplemental_id}_ready_bootstrap" --output="${log}" --error="${err}" \
  --export="${common_export},RETB_SUPPLEMENTAL_PLAN_ROLE=ready" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_offline_fusion_bootstrap.sh")"

ready_fusion="$(sbatch --parsable --dependency="afterok:${ready_bootstrap}" --array=0-2 \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --gres="${GPU_GRES}" --cpus-per-task="${RETB_SUPPLEMENTAL_GPU_CPUS}" \
  --mem="${RETB_SUPPLEMENTAL_FUSION_MEM}" \
  --job-name="${supplemental_id}_ready_fusion" --output="${log}" --error="${err}" \
  --export="${common_export},RETB_SUPPLEMENTAL_PLAN_ROLE=ready,RETB_SUPPLEMENTAL_BANKS=CE4:CE7:KD3" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_offline_fusion_bank.sh")"

obase="$(sbatch --parsable --dependency="afterok:${ready_bootstrap}" --array=0-6 \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --gres="${GPU_GRES}" --cpus-per-task="${RETB_SUPPLEMENTAL_GPU_CPUS}" \
  --mem="${RETB_SUPPLEMENTAL_OBASE_MEM}" \
  --job-name="${supplemental_id}_obase7" --output="${log}" --error="${err}" \
  --export="${common_export}" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_obase7_member.sh")"

late_bootstrap="$(sbatch --parsable "${dependency_arguments[@]}" \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --cpus-per-task=2 --mem=8G \
  --job-name="${supplemental_id}_late_bootstrap" --output="${log}" --error="${err}" \
  --export="${common_export},RETB_SUPPLEMENTAL_PLAN_ROLE=late" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_offline_fusion_bootstrap.sh")"

late_fusion="$(sbatch --parsable --dependency="afterok:${late_bootstrap}" --array=0-1 \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --gres="${GPU_GRES}" --cpus-per-task="${RETB_SUPPLEMENTAL_GPU_CPUS}" \
  --mem="${RETB_SUPPLEMENTAL_FUSION_MEM}" \
  --job-name="${supplemental_id}_late_fusion" --output="${log}" --error="${err}" \
  --export="${common_export},RETB_SUPPLEMENTAL_PLAN_ROLE=late,RETB_SUPPLEMENTAL_BANKS=KD4:MIXED7" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_offline_fusion_bank.sh")"

finalize="$(sbatch --parsable --dependency="afterok:${ready_fusion}:${late_fusion}:${obase}" \
  --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
  --cpus-per-task=4 --mem=32G \
  --job-name="${supplemental_id}_finalize" --output="${log}" --error="${err}" \
  --export="${common_export}" \
  "${PROJECT_DIR}/sbatch/run_retb_supplemental_offline_fusion_finalize.sh")"

python - "${root}" "${ready_bootstrap}" "${ready_fusion}" "${obase}" "${late_bootstrap}" "${late_fusion}" "${finalize}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
payload = {
    "supplemental_root": str(root),
    "ready_bootstrap_job_id": sys.argv[2],
    "ready_fusion_array_job_id": sys.argv[3],
    "obase7_array_job_id": sys.argv[4],
    "late_bootstrap_job_id": sys.argv[5],
    "late_fusion_array_job_id": sys.argv[6],
    "finalize_job_id": sys.argv[7],
}
(root / "job_ledgers/submission.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

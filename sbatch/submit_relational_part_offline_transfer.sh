#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/relational_part_common.sh"

: "${RPT_OFFLINE_PARENT_ROOT:=${OUTPUT_ROOT}/relational_particle_transformer/rpt_attention_bias_20260729T155648Z_bfcda5bd6f_049b2555e6}"
: "${RPT_OFFLINE_TREE_CONCURRENCY:=16}"
: "${RPT_OFFLINE_TRAIN_CONCURRENCY:=4}"
: "${RPT_OFFLINE_FINAL_CONCURRENCY:=4}"
: "${RPT_OFFLINE_REGION_CONCURRENCY:=16}"
: "${RPT_OFFLINE_SOURCE_COMMIT:=}"

launch_root="$(git -C "${SCRIPT_DIR}/.." rev-parse --show-toplevel)"
source_commit="$(
  git -C "${launch_root}" rev-parse \
    "${RPT_OFFLINE_SOURCE_COMMIT:-HEAD}^{commit}"
)"
required_source_files=(
  sbatch/submit_relational_part_offline_transfer.sh
  scripts/prepare_relational_part_offline_transfer.py
  scripts/run_relational_part_offline_training_task.py
  scripts/run_relational_part_offline_final_task.py
  teacher_logit_reco/relational_part/offline_transfer.py
)
for relative in "${required_source_files[@]}"; do
  if ! git -C "${launch_root}" cat-file -e "${source_commit}:${relative}"; then
    echo "Pinned source commit lacks offline campaign file: ${relative}" >&2
    echo "Pinned commit: ${source_commit}" >&2
    echo "Set RPT_OFFLINE_SOURCE_COMMIT to the committed offline implementation." >&2
    exit 2
  fi
done
if ! cmp -s \
  <(git -C "${launch_root}" show "${source_commit}:sbatch/submit_relational_part_offline_transfer.sh") \
  "${BASH_SOURCE[0]}"; then
  echo "The executing offline submitter differs from its pinned source commit." >&2
  echo "Pinned commit: ${source_commit}" >&2
  echo "Pull/checkout that commit, or set RPT_OFFLINE_SOURCE_COMMIT correctly." >&2
  exit 2
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'parent campaign: %s\n' "${RPT_OFFLINE_PARENT_ROOT}"
  printf 'models: 4; seeds: 3; training tasks: 12; final tasks: 12\n'
  printf 'performance gate: disabled\n'
  printf 'pinned source commit: %s\n' "${source_commit}"
  exit 0
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for required in \
  "${RPT_OFFLINE_PARENT_ROOT}/campaign_spec.json" \
  "${RPT_OFFLINE_PARENT_ROOT}/inputs/split_manifest.json.gz" \
  "${RPT_OFFLINE_PARENT_ROOT}/inputs/hlt_cache/model_train_fixed_hlt.npz"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required parent artifact is absent: ${required}" >&2
    exit 2
  fi
done
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "JetClass data root is absent: ${DATA_DIR}" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
campaign_id="rpt_offline_transfer_${timestamp}_${source_commit:0:10}"
worktree_parent="$(dirname "${PROJECT_DIR}")/.rpt_worktrees"
campaign_source_root="${worktree_parent}/${campaign_id}"
campaign_root="${OUTPUT_ROOT}/relational_particle_transformer/${campaign_id}"
if [[ -e "${campaign_source_root}" || -e "${campaign_root}" ]]; then
  echo "Offline campaign destination already exists." >&2
  exit 2
fi
mkdir -p "${worktree_parent}" "${campaign_root}/job_ledgers/slurm"
git worktree add --detach "${campaign_source_root}" "${source_commit}"

export PROJECT_DIR="${campaign_source_root}"
export CAMPAIGN_ROOT="${campaign_root}"
export CAMPAIGN_ID="${campaign_id}"
export RPT_OFFLINE_PARENT_ROOT DATA_DIR
cd "${PROJECT_DIR}"
python scripts/prepare_relational_part_offline_transfer.py \
  --campaign-id "${campaign_id}" \
  --campaign-root "${campaign_root}" \
  --parent-campaign-root "${RPT_OFFLINE_PARENT_ROOT}" \
  --split-manifest "${RPT_OFFLINE_PARENT_ROOT}/inputs/split_manifest.json.gz"

log_pattern="${campaign_root}/job_ledgers/slurm/%x_%A_%a.out"
err_pattern="${campaign_root}/job_ledgers/slurm/%x_%A_%a.err"
submit_cpu() {
  local dependency="$1" script="$2"
  shift 2
  local dep=()
  [[ -n "${dependency}" ]] && dep=(--dependency="afterok:${dependency}")
  sbatch --parsable --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
    --cpus-per-task="${CPU_CPUS_PER_TASK}" --mem="${CPU_MEM}" \
    --output="${log_pattern}" --error="${err_pattern}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CAMPAIGN_ROOT=${CAMPAIGN_ROOT},RPT_OFFLINE_PARENT_ROOT=${RPT_OFFLINE_PARENT_ROOT},DATA_DIR=${DATA_DIR}" \
    "${dep[@]}" "$@" "${PROJECT_DIR}/sbatch/${script}"
}
submit_gpu() {
  local dependency="$1" script="$2"
  shift 2
  local dep=()
  [[ -n "${dependency}" ]] && dep=(--dependency="afterok:${dependency}")
  sbatch --parsable --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
    --gres="${GPU_GRES}" --cpus-per-task="${GPU_CPUS_PER_TASK}" --mem="${GPU_MEM}" \
    --output="${log_pattern}" --error="${err_pattern}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CAMPAIGN_ROOT=${CAMPAIGN_ROOT},RPT_OFFLINE_PARENT_ROOT=${RPT_OFFLINE_PARENT_ROOT},DATA_DIR=${DATA_DIR}" \
    "${dep[@]}" "$@" "${PROJECT_DIR}/sbatch/${script}"
}

declare -A jobs
jobs[offline_cache]="$(submit_cpu "" run_cache_relational_part_offline_inputs.sh)"
jobs[offline_binding]="$(submit_cpu "${jobs[offline_cache]}" run_bind_relational_part_offline_cache.sh)"
jobs[relation_normalization]="$(submit_cpu "${jobs[offline_binding]}" run_fit_relational_part_offline_normalization.sh)"

tree_final_ids=()
declare -A counts=( [model_train]=1000000 [model_val]=125000 [stack_val]=125000 [final_test]=500000 )
for split in model_train model_val stack_val final_test; do
  shard_count="$(((counts[${split}] + RPT_TREE_SHARD_SIZE - 1) / RPT_TREE_SHARD_SIZE))"
  array="$(submit_cpu "${jobs[offline_binding]}" run_build_relational_part_offline_tree_shard.sh \
    --array="0-$((shard_count - 1))%${RPT_OFFLINE_TREE_CONCURRENCY}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CAMPAIGN_ROOT=${CAMPAIGN_ROOT},RPT_OFFLINE_PARENT_ROOT=${RPT_OFFLINE_PARENT_ROOT},DATA_DIR=${DATA_DIR},RPT_TREE_SPLIT=${split}")"
  jobs["tree_${split}"]="${array}"
  final="$(submit_cpu "${array}" run_finalize_relational_part_offline_tree_split.sh \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CAMPAIGN_ROOT=${CAMPAIGN_ROOT},RPT_OFFLINE_PARENT_ROOT=${RPT_OFFLINE_PARENT_ROOT},DATA_DIR=${DATA_DIR},RPT_TREE_SPLIT=${split}")"
  jobs["tree_final_${split}"]="${final}"
  tree_final_ids+=("${final}")
done

jobs[region_plan]="$(submit_cpu "${jobs[relation_normalization]}:${jobs[tree_final_model_train]}" run_prepare_relational_part_offline_region_map.sh)"
jobs[region_shards]="$(submit_cpu "${jobs[region_plan]}" run_fit_relational_part_offline_region_shard.sh \
  --array="0-99%${RPT_OFFLINE_REGION_CONCURRENCY}" --cpus-per-task=2 --mem=24G)"
jobs[region_normalization]="$(submit_cpu "${jobs[region_shards]}" run_finalize_relational_part_offline_region.sh)"
model_dep="${jobs[region_normalization]}"
for job in "${tree_final_ids[@]}"; do model_dep="${model_dep}:${job}"; done
jobs[model_contracts]="$(submit_cpu "${model_dep}" run_prepare_relational_part_offline_models.sh)"
jobs[training]="$(submit_gpu "${jobs[model_contracts]}" run_train_relational_part_offline.sh --array="0-11%${RPT_OFFLINE_TRAIN_CONCURRENCY}")"
jobs[validation_lock]="$(submit_cpu "${jobs[training]}" run_aggregate_relational_part_offline_validation.sh)"
jobs[final_test]="$(submit_gpu "${jobs[validation_lock]}" run_evaluate_relational_part_offline_final.sh --array="0-11%${RPT_OFFLINE_FINAL_CONCURRENCY}")"
jobs[report]="$(submit_cpu "${jobs[final_test]}" run_write_relational_part_offline_report.sh)"

python - "${campaign_root}/job_ledgers/submission.json" "${campaign_id}" "${jobs[report]}" <<'PY'
import json, sys
path, campaign_id, report_job = sys.argv[1:]
with open(path, "w", encoding="utf-8") as stream:
    json.dump({"campaign_id": campaign_id, "final_report_job": report_job, "performance_gate": False}, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

printf 'offline campaign root: %s\n' "${campaign_root}"
printf 'pinned source: %s\n' "${campaign_source_root}"
printf 'cache/bind: %s / %s\n' "${jobs[offline_cache]}" "${jobs[offline_binding]}"
printf 'training array: %s\nfinal-test array: %s\nreport: %s\n' "${jobs[training]}" "${jobs[final_test]}" "${jobs[report]}"
printf 'No validation-performance threshold can cancel or prune these tasks.\n'

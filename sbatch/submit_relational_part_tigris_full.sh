#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"

: "${SCREENING_ARRAY_CONCURRENCY:=4}"
: "${TREE_ARRAY_CONCURRENCY:=16}"
: "${REGION_ARRAY_CONCURRENCY:=16}"
: "${RPT_CONFIRMATION_CONCURRENCY:=4}"
: "${RPT_FINAL_CONCURRENCY:=3}"
: "${RPT_MINIATURE:=0}"
: "${RPT_STORAGE_MEASUREMENTS:=${OUTPUT_ROOT}/relational_particle_transformer/bootstrap/storage_measurements.json}"

mode="submit"
case "${1:-}" in
  --dry-run) mode="dry-run" ;;
  --smoke-simulate) mode="smoke-simulate" ;;
  --smoke-submit)
    mode="submit"
    RPT_MINIATURE=1
    ;;
  "") ;;
  *)
    echo "Usage: $0 [--dry-run|--smoke-simulate|--smoke-submit]" >&2
    exit 2
    ;;
esac

cd "${PROJECT_DIR}"
if [[ "${mode}" != "submit" ]]; then
  args=(--dry-run)
  if [[ "${mode}" == "smoke-simulate" ]]; then
    args=(--smoke-simulate --miniature)
  elif [[ "${RPT_MINIATURE}" == "1" ]]; then
    args+=(--miniature)
  fi
  python scripts/submit_relational_part_graph.py \
    --output-root "${OUTPUT_ROOT}" \
    --screening-array-concurrency "${SCREENING_ARRAY_CONCURRENCY}" \
    --tree-array-concurrency "${TREE_ARRAY_CONCURRENCY}" \
    --region-array-concurrency "${REGION_ARRAY_CONCURRENCY}" \
    "${args[@]}"
  exit 0
fi

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Conda activation hook is absent: ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -c \
  'from torch.utils.cpp_extension import verify_ninja_availability; verify_ninja_availability(); print("PyTorch C++ extension toolchain: ninja OK")'
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "JetClass data root does not exist: ${DATA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${RPT_STORAGE_MEASUREMENTS}" ]]; then
  echo "Authenticated storage measurements are absent: ${RPT_STORAGE_MEASUREMENTS}" >&2
  echo "Set RPT_STORAGE_MEASUREMENTS to a source-evidence-bound Step-1 measurement artifact." >&2
  exit 2
fi

required_per_class=175000
if [[ "${RPT_MINIATURE}" == "1" ]]; then
  required_per_class=6
fi
python scripts/preflight_relational_part_data.py \
  --data-dir "${DATA_DIR}" \
  --tree-name tree \
  --required-per-class "${required_per_class}"

source_commit="$(python scripts/print_relational_part_source_snapshot.py --field source_commit)"
source_status_sha="$(python scripts/print_relational_part_source_snapshot.py --field source_status_sha256)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
campaign_id="rpt_attention_bias_${timestamp}_${source_commit:0:10}_${source_status_sha:0:10}"
campaign_root="${OUTPUT_ROOT}/relational_particle_transformer/${campaign_id}"
if [[ -e "${campaign_root}" ]]; then
  echo "Refusing existing campaign root: ${campaign_root}" >&2
  exit 2
fi
mkdir -p \
  "${campaign_root}/bootstrap" \
  "${campaign_root}/job_ledgers/slurm"
export CAMPAIGN_ID="${campaign_id}"
export CAMPAIGN_ROOT="${campaign_root}"
export RPT_STORAGE_MEASUREMENTS
export RPT_MINIATURE
export RPT_CONFIRMATION_CONCURRENCY
export RPT_FINAL_CONCURRENCY

graph_args=(--write-artifacts)
if [[ "${RPT_MINIATURE}" == "1" ]]; then
  graph_args+=(--miniature)
fi
python scripts/submit_relational_part_graph.py \
  --output-root "${OUTPUT_ROOT}" \
  --campaign-id "${campaign_id}" \
  --campaign-root "${campaign_root}" \
  --screening-array-concurrency "${SCREENING_ARRAY_CONCURRENCY}" \
  --tree-array-concurrency "${TREE_ARRAY_CONCURRENCY}" \
  --region-array-concurrency "${REGION_ARRAY_CONCURRENCY}" \
  "${graph_args[@]}"

log_pattern="${campaign_root}/job_ledgers/slurm/%x_%A_%a.out"
error_pattern="${campaign_root}/job_ledgers/slurm/%x_%A_%a.err"

submit_cpu() {
  local dependency="$1"
  local script="$2"
  shift 2
  local dependency_args=()
  if [[ -n "${dependency}" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi
  sbatch --parsable \
    --account="${SBATCH_ACCOUNT}" \
    --partition="${SBATCH_PARTITION}" \
    --cpus-per-task="${CPU_CPUS_PER_TASK}" \
    --mem="${CPU_MEM}" \
    --output="${log_pattern}" \
    --error="${error_pattern}" \
    "${dependency_args[@]}" \
    "$@" \
    "${SCRIPT_DIR}/${script}"
}

submit_gpu() {
  local dependency="$1"
  local script="$2"
  shift 2
  local dependency_args=()
  if [[ -n "${dependency}" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi
  sbatch --parsable \
    --account="${SBATCH_ACCOUNT}" \
    --partition="${SBATCH_PARTITION}" \
    --gres="${GPU_GRES}" \
    --cpus-per-task="${GPU_CPUS_PER_TASK}" \
    --mem="${GPU_MEM}" \
    --output="${log_pattern}" \
    --error="${error_pattern}" \
    "${dependency_args[@]}" \
    "$@" \
    "${SCRIPT_DIR}/${script}"
}

declare -A jobs
jobs[split_build]="$(submit_cpu "" run_build_relational_part_splits.sh)"
jobs[campaign_bootstrap]="$(submit_cpu "${jobs[split_build]}" run_build_relational_part_campaign.sh)"
jobs[preconstruction_raw_audit]="$(submit_cpu "${jobs[campaign_bootstrap]}" run_audit_relational_part_raw_inputs.sh)"
jobs[hlt_cache]="$(submit_cpu "${jobs[preconstruction_raw_audit]}" run_build_relational_part_hlt_cache.sh)"
jobs[tree_backend]="$(submit_cpu "${jobs[preconstruction_raw_audit]}" run_build_relational_part_tree_backend.sh)"
jobs[weaver_parity]="$(submit_cpu "${jobs[preconstruction_raw_audit]}" run_validate_relational_part_weaver_parity.sh)"
jobs[relation_normalization]="$(submit_cpu "${jobs[preconstruction_raw_audit]}:${jobs[hlt_cache]}" run_fit_relational_part_normalization.sh)"
jobs[tree_probe]="$(submit_cpu "${jobs[hlt_cache]}:${jobs[tree_backend]}" run_probe_relational_part_tree_backend.sh)"

if [[ "${RPT_MINIATURE}" == "1" ]]; then
  declare -A split_sizes=(
    [model_train]=20 [model_val]=10 [stack_val]=10 [final_test]=20
  )
else
  declare -A split_sizes=(
    [model_train]=1000000 [model_val]=125000
    [stack_val]=125000 [final_test]=500000
  )
fi
tree_final_ids=()
for split in model_train model_val stack_val final_test; do
  shard_count="$(((split_sizes[${split}] + RPT_TREE_SHARD_SIZE - 1) / RPT_TREE_SHARD_SIZE))"
  last="$((shard_count - 1))"
  array_id="$(submit_cpu "${jobs[tree_probe]}" \
    run_build_relational_part_angular_tree_shard.sh \
    --array="0-${last}%${TREE_ARRAY_CONCURRENCY}" \
    --export="ALL,RPT_TREE_SPLIT=${split}")"
  jobs["tree_shards_${split}"]="${array_id}"
  final_id="$(submit_cpu "${array_id}" \
    run_finalize_relational_part_angular_tree_cache.sh \
    --export="ALL,RPT_TREE_SPLIT=${split}")"
  jobs["tree_finalize_${split}"]="${final_id}"
  tree_final_ids+=("${final_id}")
done
region_shard_count="$(((split_sizes[model_train] + RPT_TREE_SHARD_SIZE - 1) / RPT_TREE_SHARD_SIZE))"
jobs[region_normalization_plan]="$(submit_cpu \
  "${jobs[relation_normalization]}:${jobs[tree_finalize_model_train]}" \
  run_prepare_relational_part_region_normalization_map.sh)"
jobs[region_normalization_shards]="$(submit_cpu \
  "${jobs[region_normalization_plan]}" \
  run_fit_relational_part_region_normalization_shard.sh \
  --array="0-$((region_shard_count - 1))%${REGION_ARRAY_CONCURRENCY}" \
  --cpus-per-task=2 \
  --mem=24G)"
jobs[region_normalization]="$(submit_cpu \
  "${jobs[region_normalization_shards]}" \
  run_finalize_relational_part_region_normalization.sh)"
post_dependency="${jobs[hlt_cache]}:${jobs[relation_normalization]}:${jobs[region_normalization]}:${jobs[tree_backend]}:${jobs[tree_probe]}"
for value in "${tree_final_ids[@]}"; do
  post_dependency="${post_dependency}:${value}"
done
jobs[postconstruction_input_audit]="$(submit_cpu "${post_dependency}" run_audit_relational_part_inputs.sh)"
jobs[screening_model_contracts]="$(submit_cpu \
  "${jobs[postconstruction_input_audit]}:${jobs[weaver_parity]}" \
  run_build_relational_part_model_contracts.sh)"
jobs[screening]="$(submit_gpu "${jobs[screening_model_contracts]}" \
  run_train_relational_part.sh \
  --array="0-20%${SCREENING_ARRAY_CONCURRENCY}" \
  --export="ALL,RPT_TRAIN_MODE=screening")"
jobs[screening_selection]="$(submit_cpu "${jobs[screening]}" run_select_relational_part_screening.sh)"
jobs[confirmation_submit]="$(submit_cpu "${jobs[screening_selection]}" run_submit_relational_part_confirmation.sh)"

ledger_args=()
for name in "${!jobs[@]}"; do
  ledger_args+=(--job "${name}=${jobs[${name}]}")
done
python scripts/write_relational_part_job_ledger.py \
  --production-graph "${campaign_root}/job_ledgers/production_graph.json" \
  "${ledger_args[@]}" \
  --output "${campaign_root}/job_ledgers/initial_submission_ledger.json"

printf 'campaign root: %s\n' "${campaign_root}"
printf 'source commit: %s\nsource dirty-status hash: %s\n' \
  "${source_commit}" "${source_status_sha}"
printf 'split job: %s\nHLT cache job: %s\nbackend manifest: %s\nthroughput probe: %s\n' \
  "${jobs[split_build]}" "${jobs[hlt_cache]}" \
  "${campaign_root}/backend/backend_manifest.json" \
  "${campaign_root}/backend/throughput_probe.json"
for split in model_train model_val stack_val final_test; do
  printf '%s tree array/finalizer: %s / %s\n' \
    "${split}" "${jobs[tree_shards_${split}]}" "${jobs[tree_finalize_${split}]}"
done
printf 'screening array: %s\nselector: %s\ncontinuation: %s\nledger: %s\n' \
  "${jobs[screening]}" "${jobs[screening_selection]}" \
  "${jobs[confirmation_submit]}" \
  "${campaign_root}/job_ledgers/initial_submission_ledger.json"
printf 'REGION plan / shard array / reducer: %s / %s / %s\n' \
  "${jobs[region_normalization_plan]}" \
  "${jobs[region_normalization_shards]}" \
  "${jobs[region_normalization]}"
printf 'monitor: squeue -u "$USER" -o "%%i %%j %%T %%R"\n'
printf 'accounting: sacct -X --starttime today --name rpt_ --format=JobID,JobName,State,Elapsed,ExitCode\n'
printf 'download: rsync -av tigris:%s/reports/ ./relational_part_reports/\n' "${campaign_root}"
printf 'final metrics: %s/final_test/<run_id>/seed_<seed>/metrics.json\n' "${campaign_root}"

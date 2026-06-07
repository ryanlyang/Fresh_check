#!/usr/bin/env bash
# Submit the five HLT seed-control training jobs and dependent HLT5 fusion.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"

fresh_prepare_submitter

submit_count=0
submit_job() {
  local label="$1"
  shift
  submit_count=$((submit_count + 1))
  if fresh_is_dry_run; then
    printf 'DRY_RUN sbatch %s: ' "${label}" >&2
    fresh_print_shell_command sbatch "$@" >&2
    printf '\n' >&2
    local clean_label="${label//[^A-Za-z0-9_]/_}"
    printf 'DRYRUN_%s\n' "${clean_label}"
    return 0
  fi
  local output
  output="$(sbatch "$@")"
  echo "${output}" >&2
  echo "${output}" | awk '{print $NF}'
}

with_dependency() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then
    echo --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

fresh_split_words seed_args "${HLT5_SEEDS}"
seed_job_ids=()
for seed in "${seed_args[@]}"; do
  fresh_refuse_existing_dir "${HLT5_ROOT}/seed${seed}"
  mapfile -t args < <(with_dependency "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_train_fresh_hlt_seed.sh" "${seed}")
  jid="$(submit_job "hlt_seed_${seed}" "${args[@]}")"
  seed_job_ids+=("${jid}")
  echo "submitted hlt_seed_${seed}=${jid}"
done

fresh_refuse_existing_dir "${HLT5_FUSION_DIR}"
dependency="$(fresh_join_by_colon "${seed_job_ids[@]}")"
fusion_jid="$(submit_job "hlt5_fusion" --dependency="afterok:${dependency}" "${SCRIPT_DIR}/run_fuse_fresh_hlt5_seed_control.sh")"

echo "hlt_seed_job_ids=$(fresh_join_by_space "${seed_job_ids[@]}")"
echo "hlt5_dependency=afterok:${dependency}"
echo "hlt5_fusion_job_id=${fusion_jid}"
echo "expected_hlt5_fusion_output=${HLT5_FUSION_DIR}"

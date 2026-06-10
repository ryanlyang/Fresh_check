#!/usr/bin/env bash
# Submit four heterogeneous HLT-only taggers, then dependent stacked fusion.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
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

dep_args() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

fresh_split_words arch_args "${HETERO_HLT4_ARCHITECTURES}"
submitter_lock_dir="${HETERO_HLT4_ROOT}/.submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "architectures=$(fresh_join_by_space "${arch_args[@]}")"
    echo "model_train_size=${HETERO_HLT4_TRAIN_SIZE}"
    echo "model_val_size=${HETERO_HLT4_VAL_SIZE}"
    echo "stack_train_size=${HETERO_HLT4_STACK_TRAIN_SIZE}"
    echo "stack_val_size=${HETERO_HLT4_STACK_VAL_SIZE}"
    echo "final_test_size=${HETERO_HLT4_FINAL_TEST_SIZE}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

train_job_ids=()
for architecture in "${arch_args[@]}"; do
  fresh_refuse_existing_dir "${HETERO_HLT4_MODEL_ROOT}/${architecture}"
  mapfile -t args < <(dep_args "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_train_heterogeneous_hlt_arch.sh" "${architecture}")
  jid="$(submit_job "hetero_hlt_${architecture}" "${args[@]}")"
  train_job_ids+=("${jid}")
  echo "submitted hetero_hlt_${architecture}=${jid}"
done

fusion_dependency="$(fresh_join_by_colon "${train_job_ids[@]}")"
fresh_refuse_existing_dir "${HETERO_HLT4_FUSION_DIR}"
fusion_jid="$(submit_job "hetero_hlt4_fusion" \
  --dependency="afterok:${fusion_dependency}" \
  "${SCRIPT_DIR}/run_fuse_heterogeneous_hlt4.sh")"

cat <<SUMMARY
heterogeneous_hlt4_submission:
  train_job_ids: $(fresh_join_by_space "${train_job_ids[@]}")
  fusion_job_id: ${fusion_jid}
  dependency_summary:
    fusion_afterok: ${fusion_dependency}
  architectures: $(fresh_join_by_space "${arch_args[@]}")
  split_sizes:
    model_train: ${HETERO_HLT4_TRAIN_SIZE}
    model_val: ${HETERO_HLT4_VAL_SIZE}
    stack_train: ${HETERO_HLT4_STACK_TRAIN_SIZE}
    stack_val: ${HETERO_HLT4_STACK_VAL_SIZE}
    final_test: ${HETERO_HLT4_FINAL_TEST_SIZE}
  output_dirs:
    root: ${HETERO_HLT4_ROOT}
    model_root: ${HETERO_HLT4_MODEL_ROOT}
    fusion: ${HETERO_HLT4_FUSION_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY

#!/usr/bin/env bash
# Submit all sixteen cross-architecture teacher-logit reconstructor jobs.

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

fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_RECO_TEACHERS}"

submitter_lock_dir="${CROSSARCH_ROOT}/.step5_reconstructor_submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "reconstructors=$(fresh_join_by_space "${reco_args[@]}")"
    echo "teachers=$(fresh_join_by_space "${teacher_args[@]}")"
    echo "manifest=${CROSSARCH_MANIFEST_PATH}"
    echo "hlt_cache_dir=${CROSSARCH_HLT_CACHE_DIR}"
    echo "offline_teacher_dir=${CROSSARCH_OFFLINE_TEACHER_DIR}"
    echo "reco_model_dir=${CROSSARCH_RECO_MODEL_DIR}"
    echo "model_train_size=${CROSSARCH_RECO_MAX_TRAIN_JETS}"
    echo "model_val_size=${CROSSARCH_RECO_MAX_VAL_JETS}"
    echo "upstream_dependency=${UPSTREAM_DEPENDENCY:-none}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

reco_job_ids=()
reco_pairs=()
for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    model_name="$(fresh_crossarch_reco_model_name "${reco_architecture}" "${teacher_architecture}")"
    output_dir="${CROSSARCH_RECO_MODEL_DIR}/${reco_architecture}/${teacher_architecture}"
    fresh_refuse_existing_dir "${output_dir}"
    mapfile -t train_args < <(
      dep_args \
        "${UPSTREAM_DEPENDENCY}" \
        "${SCRIPT_DIR}/run_crossarch_train_reconstructor.sh" \
        "${reco_architecture}" \
        "${teacher_architecture}"
    )
    train_jid="$(submit_job "crossarch_reco_${model_name}" "${train_args[@]}")"
    reco_job_ids+=("${train_jid}")
    reco_pairs+=("${model_name}")
    echo "submitted crossarch_reco_${model_name}=${train_jid}"
  done
done

cat <<SUMMARY
crossarch_step5_reconstructors_submission:
  reco_job_ids: $(fresh_join_by_space "${reco_job_ids[@]}")
  dependency_summary:
    reco_afterok: ${UPSTREAM_DEPENDENCY:-none}
  reconstructors: $(fresh_join_by_space "${reco_args[@]}")
  teachers: $(fresh_join_by_space "${teacher_args[@]}")
  model_names: $(fresh_join_by_space "${reco_pairs[@]}")
  expected_models: 16
  split_sizes:
    model_train: ${CROSSARCH_RECO_MAX_TRAIN_JETS}
    model_val: ${CROSSARCH_RECO_MAX_VAL_JETS}
  output_dirs:
    root: ${CROSSARCH_ROOT}
    reco_models: ${CROSSARCH_RECO_MODEL_DIR}
    offline_teachers: ${CROSSARCH_OFFLINE_TEACHER_DIR}
    hlt_cache: ${CROSSARCH_HLT_CACHE_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY

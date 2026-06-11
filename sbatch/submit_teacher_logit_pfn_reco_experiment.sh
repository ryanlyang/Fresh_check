#!/usr/bin/env bash
# Submit teacher-logit PFN reconstructor training, prediction, and fusion jobs.

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

fresh_split_words teacher_args "${TEACHER_LOGIT_PFN_TEACHERS}"
if [[ "${#teacher_args[@]}" -eq 0 ]]; then
  echo "TEACHER_LOGIT_PFN_TEACHERS is empty" >&2
  exit 2
fi

submitter_lock_dir="${TEACHER_LOGIT_PFN_ROOT}/.submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "teachers=$(fresh_join_by_space "${teacher_args[@]}")"
    echo "reco_root=${TEACHER_LOGIT_PFN_RECO_ROOT}"
    echo "prediction_dir=${TEACHER_LOGIT_PFN_PREDICTION_DIR}"
    echo "fusion_dir=${TEACHER_LOGIT_PFN_FUSION_DIR}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

train_job_ids=()
predict_job_ids=()
model_names=()
for architecture in "${teacher_args[@]}"; do
  case "${architecture}" in
    part|pn|pfn|pcnn) ;;
    *)
      echo "Unknown teacher-logit PFN architecture ${architecture}; expected part pn pfn pcnn" >&2
      exit 2
      ;;
  esac
  teacher_checkpoint="$(fresh_teacher_logit_pfn_teacher_checkpoint "${architecture}")"
  model_name="$(fresh_teacher_logit_pfn_model_name "${architecture}")"
  model_names+=("${model_name}")
  fresh_require_file "${teacher_checkpoint}"
  fresh_refuse_existing_dir "${TEACHER_LOGIT_PFN_RECO_ROOT}/${architecture}"
  fresh_refuse_existing_dir "${TEACHER_LOGIT_PFN_PREDICTION_RUN_ROOT}/${architecture}"
  fresh_refuse_existing_dir "${TEACHER_LOGIT_PFN_PREDICTION_DIR}/${model_name}"

  mapfile -t train_args < <(dep_args "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_train_teacher_logit_pfn_reco.sh" "${architecture}")
  train_jid="$(submit_job "teacher_logit_pfn_train_${architecture}" "${train_args[@]}")"
  train_job_ids+=("${train_jid}")
  echo "submitted teacher_logit_pfn_train_${architecture}=${train_jid}"

  predict_jid="$(submit_job "teacher_logit_pfn_predict_${architecture}" \
    --dependency="afterok:${train_jid}" \
    "${SCRIPT_DIR}/run_predict_teacher_logit_pfn_reco.sh" "${architecture}")"
  predict_job_ids+=("${predict_jid}")
  echo "submitted teacher_logit_pfn_predict_${architecture}=${predict_jid}"
done

fusion_dependency="$(fresh_join_by_colon "${predict_job_ids[@]}")"
fresh_refuse_existing_dir "${TEACHER_LOGIT_PFN_FUSION_DIR}"
fusion_jid="$(submit_job "teacher_logit_pfn_fusion" \
  --dependency="afterok:${fusion_dependency}" \
  "${SCRIPT_DIR}/run_fuse_teacher_logit_pfn_reco.sh")"

cat <<SUMMARY
teacher_logit_pfn_reco_submission:
  train_job_ids: $(fresh_join_by_space "${train_job_ids[@]}")
  predict_job_ids: $(fresh_join_by_space "${predict_job_ids[@]}")
  fusion_job_id: ${fusion_jid}
  dependency_summary:
    prediction_after_each_train: true
    fusion_afterok: ${fusion_dependency}
  teachers: $(fresh_join_by_space "${teacher_args[@]}")
  model_names: $(fresh_join_by_space "${model_names[@]}")
  output_dirs:
    root: ${TEACHER_LOGIT_PFN_ROOT}
    reco_root: ${TEACHER_LOGIT_PFN_RECO_ROOT}
    prediction_runs: ${TEACHER_LOGIT_PFN_PREDICTION_RUN_ROOT}
    prediction_dir: ${TEACHER_LOGIT_PFN_PREDICTION_DIR}
    fusion: ${TEACHER_LOGIT_PFN_FUSION_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY

#!/usr/bin/env bash
# Submit one target-conditioned denoising ParT experiment root.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter

: "${TARGET_DENOISING_PART_ROOT:=${OUTPUT_ROOT}/target_conditioned_denoising_part_hltv2_$(date +%Y%m%d_%H%M%S)}"
: "${TARGET_DENOISING_PART_UPSTREAM_DEPENDENCY:=${UPSTREAM_DEPENDENCY:-}}"
: "${TARGET_DENOISING_PART_SUBMIT_DENOISER:=1}"
: "${TARGET_DENOISING_PART_SUBMIT_TAGGERS:=1}"
: "${TARGET_DENOISING_PART_SUBMIT_REPORT:=1}"
: "${TARGET_DENOISING_PART_VARIANTS:=hlt_part_baseline feature_mlp_adapter_tag_only denoiser_features_frozen denoiser_features_joint denoiser_tag_only_same_arch denoiser_no_pair_bias}"
: "${TARGET_DENOISING_PART_DENOISER_ROOT:=${TARGET_DENOISING_PART_ROOT}/denoisers}"
: "${TARGET_DENOISING_PART_DENOISER_OUTPUT_DIR:=${TARGET_DENOISING_PART_DENOISER_ROOT}/real}"
: "${TARGET_DENOISING_PART_TAGGER_ROOT:=${TARGET_DENOISING_PART_ROOT}/taggers}"
: "${TARGET_DENOISING_PART_REPORT_DIR:=${TARGET_DENOISING_PART_ROOT}/final_report}"
: "${TARGET_DENOISING_PART_MANIFEST_PATH:=${MANIFEST_PATH:-}}"
: "${TARGET_DENOISING_PART_HLT_CACHE_DIR:=${HLT_CACHE_DIR:-}}"
: "${TARGET_DENOISING_PART_OFFLINE_CACHE_DIR:=${OFFLINE_CACHE_DIR:-}}"
: "${TARGET_DENOISING_PART_MODEL_TRAIN_SIZE:=500000}"
: "${TARGET_DENOISING_PART_MODEL_VAL_SIZE:=150000}"
: "${TARGET_DENOISING_PART_FINAL_TEST_SIZE:=150000}"
: "${TARGET_DENOISING_PART_HLT_PROFILE:=fixed_hlt_v2_realistic}"
: "${TARGET_DENOISING_PART_HLT_PROFILE_VERSION:=v1}"
: "${TARGET_DENOISING_PART_HLT_DEGRADATION_STRENGTH:=1.0}"

if [[ -z "${TARGET_DENOISING_PART_OFFLINE_CACHE_DIR}" && -n "${TARGET_DENOISING_PART_HLT_CACHE_DIR}" ]]; then
  inferred_offline_cache_dir="$(dirname "${TARGET_DENOISING_PART_HLT_CACHE_DIR}")/offline_cache"
  if [[ -d "${inferred_offline_cache_dir}" ]]; then
    TARGET_DENOISING_PART_OFFLINE_CACHE_DIR="${inferred_offline_cache_dir}"
  fi
fi

export TARGET_DENOISING_PART_ROOT
export TARGET_DENOISING_PART_DENOISER_ROOT
export TARGET_DENOISING_PART_DENOISER_OUTPUT_DIR
export TARGET_DENOISING_PART_TAGGER_ROOT
export TARGET_DENOISING_PART_REPORT_DIR
export TARGET_DENOISING_PART_VARIANTS
export TARGET_DENOISING_PART_MODEL_TRAIN_SIZE
export TARGET_DENOISING_PART_MODEL_VAL_SIZE
export TARGET_DENOISING_PART_FINAL_TEST_SIZE
export TARGET_DENOISING_PART_MANIFEST_PATH
export TARGET_DENOISING_PART_HLT_CACHE_DIR
export TARGET_DENOISING_PART_OFFLINE_CACHE_DIR
export TARGET_DENOISING_PART_HLT_PROFILE
export TARGET_DENOISING_PART_HLT_PROFILE_VERSION
export TARGET_DENOISING_PART_HLT_DEGRADATION_STRENGTH

dependency_token_is_valid() {
  local token="$1"
  if [[ "${token}" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  if fresh_is_dry_run && [[ "${token}" =~ ^DRYRUN_[A-Za-z0-9_]+$ ]]; then
    return 0
  fi
  return 1
}

validate_dependency_list() {
  local label="$1"
  local dependency="$2"
  if [[ -z "${dependency}" ]]; then
    return 0
  fi
  local old_ifs="${IFS}"
  local tokens=()
  IFS=':'
  read -r -a tokens <<< "${dependency}"
  IFS="${old_ifs}"
  local token
  for token in "${tokens[@]}"; do
    if [[ -z "${token}" ]] || ! dependency_token_is_valid "${token}"; then
      echo "Invalid Slurm dependency for ${label}: '${dependency}'." >&2
      return 2
    fi
  done
}

submit_job() {
  local label="$1"
  shift
  if fresh_is_dry_run; then
    printf 'DRY_RUN sbatch %s: ' "${label}" >&2
    fresh_print_shell_command sbatch "$@" >&2
    printf '\n' >&2
    local clean_label="${label//[^A-Za-z0-9_]/_}"
    printf 'DRYRUN_%s\n' "${clean_label}"
    return 0
  fi
  local output
  if ! output="$(sbatch "$@")"; then
    echo "Failed to submit ${label}." >&2
    return 2
  fi
  echo "${output}" >&2
  local job_id
  job_id="$(echo "${output}" | awk '{print $NF}')"
  if ! dependency_token_is_valid "${job_id}"; then
    echo "Failed to submit ${label}; expected a Slurm job ID but got '${job_id:-empty}'." >&2
    return 2
  fi
  printf '%s\n' "${job_id}"
}

join_nonempty_by_colon() {
  local values=()
  local item
  for item in "$@"; do
    if [[ -n "${item}" ]]; then
      values+=("${item}")
    fi
  done
  if [[ "${#values[@]}" -eq 0 ]]; then
    return 0
  fi
  fresh_join_by_colon "${values[@]}"
}

afterok_args() {
  local dependency="$1"
  shift
  validate_dependency_list "afterok" "${dependency}"
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

denoiser_output_complete() {
  local output_dir="$1"
  [[ -f "${output_dir}/best_denoiser_model_val.pt" ]] \
    && [[ -f "${output_dir}/last.pt" ]] \
    && [[ -f "${output_dir}/run_report.json" ]] \
    && [[ -f "${output_dir}/model_val_diagnostics.json" ]] \
    && [[ -f "${output_dir}/training_curves.json" ]] \
    && [[ -f "${output_dir}/diagnostics/epoch_metrics.csv" ]] \
    && grep -q '"ok": true' "${output_dir}/run_report.json"
}

tagger_output_complete() {
  local output_dir="$1"
  [[ -f "${output_dir}/best_model_val.pt" ]] \
    && [[ -f "${output_dir}/last.pt" ]] \
    && [[ -f "${output_dir}/run_report.json" ]] \
    && [[ -f "${output_dir}/model_val_report.json" ]] \
    && [[ -f "${output_dir}/config.json" ]] \
    && [[ -f "${output_dir}/training_curves.json" ]] \
    && [[ -f "${output_dir}/diagnostics/epoch_metrics.csv" ]] \
    && grep -q '"ok": true' "${output_dir}/run_report.json" \
    && { ! fresh_bool_enabled "${TARGET_DENOISING_PART_EVALUATE_FINAL_TEST:-1}" || [[ -f "${output_dir}/final_test_report.json" ]]; }
}

report_output_complete() {
  local output_dir="$1"
  [[ -f "${output_dir}/summary.json" ]] \
    && [[ -f "${output_dir}/tagger_metrics.csv" ]] \
    && [[ -f "${output_dir}/denoising_metrics.csv" ]] \
    && [[ -f "${output_dir}/mechanism_ablation_metrics.csv" ]] \
    && grep -q '"ok": true' "${output_dir}/summary.json"
}

archive_incomplete_output() {
  local output_dir="$1"
  local checker="$2"
  if fresh_is_dry_run || [[ ! -d "${output_dir}" ]]; then
    return 0
  fi
  if "${checker}" "${output_dir}"; then
    return 0
  fi
  local archived="${output_dir}_incomplete_$(date +%Y%m%d_%H%M%S)"
  echo "found incomplete target-denoising output; moving ${output_dir} to ${archived}" >&2
  mv "${output_dir}" "${archived}"
}

archive_existing_output() {
  local output_dir="$1"
  local reason="$2"
  if fresh_is_dry_run || [[ ! -d "${output_dir}" ]]; then
    return 0
  fi
  local archived="${output_dir}_${reason}_$(date +%Y%m%d_%H%M%S)"
  echo "found existing target-denoising output for ${reason}; moving ${output_dir} to ${archived}" >&2
  mv "${output_dir}" "${archived}"
}

denoiser_type_for_variant() {
  local variant="$1"
  case "${variant}" in
    denoiser_features_frozen|denoiser_features_joint)
      printf 'real\n'
      ;;
    denoiser_shuffled_targets)
      printf 'shuffled_targets\n'
      ;;
    denoiser_no_pair_bias)
      printf 'no_pair_bias\n'
      ;;
    denoiser_local_kernel_only)
      printf 'local_kernel_only\n'
      ;;
    *)
      printf '\n'
      ;;
  esac
}

denoiser_dir_for_type() {
  local denoiser_type="$1"
  printf '%s/%s\n' "${TARGET_DENOISING_PART_DENOISER_ROOT}" "${denoiser_type}"
}

denoiser_flags_for_type() {
  local denoiser_type="$1"
  case "${denoiser_type}" in
    shuffled_targets)
      printf '%s\n' "TARGET_DENOISING_PART_SHUFFLE_TARGET_RESIDUALS=1"
      ;;
    no_pair_bias)
      printf '%s\n' "TARGET_DENOISING_PART_DISABLE_PAIR_BIAS=1"
      printf '%s\n' "TARGET_DENOISING_PART_DISABLE_LOCAL_KERNEL=1"
      ;;
    local_kernel_only)
      printf '%s\n' "TARGET_DENOISING_PART_DISABLE_PAIR_BIAS=1"
      printf '%s\n' "TARGET_DENOISING_PART_DISABLE_LOCAL_KERNEL=0"
      ;;
  esac
}

export_arg_from_assignments() {
  local export_value="ALL"
  local assignment
  for assignment in "$@"; do
    if [[ -n "${assignment}" ]]; then
      export_value="${export_value},${assignment}"
    fi
  done
  printf '%s\n' "--export=${export_value}"
}

submitter_log_dir="${TARGET_DENOISING_PART_ROOT}/submission_logs/target_denoising_part_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_log_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "root=${TARGET_DENOISING_PART_ROOT}"
    echo "manifest=${TARGET_DENOISING_PART_MANIFEST_PATH:-${MANIFEST_PATH:-}}"
    echo "hlt_cache=${TARGET_DENOISING_PART_HLT_CACHE_DIR:-${HLT_CACHE_DIR:-}}"
    echo "offline_cache=${TARGET_DENOISING_PART_OFFLINE_CACHE_DIR:-none}"
    echo "variants=${TARGET_DENOISING_PART_VARIANTS}"
    echo "upstream_dependency=${TARGET_DENOISING_PART_UPSTREAM_DEPENDENCY}"
  } > "${submitter_log_dir}/metadata.txt"
fi

echo "target_denoising_part_submission_start:"
echo "  root: ${TARGET_DENOISING_PART_ROOT}"
echo "  variants: ${TARGET_DENOISING_PART_VARIANTS}"
echo "  offline_cache: ${TARGET_DENOISING_PART_OFFLINE_CACHE_DIR:-none}"
echo "  upstream_dependency: ${TARGET_DENOISING_PART_UPSTREAM_DEPENDENCY:-none}"

fresh_split_words variants "${TARGET_DENOISING_PART_VARIANTS}"
denoiser_types=()
for variant in "${variants[@]}"; do
  dtype="$(denoiser_type_for_variant "${variant}")"
  if [[ -z "${dtype}" ]]; then
    continue
  fi
  already_seen=0
  for existing in "${denoiser_types[@]:-}"; do
    if [[ "${existing}" == "${dtype}" ]]; then
      already_seen=1
      break
    fi
  done
  if [[ "${already_seen}" -eq 0 ]]; then
    denoiser_types+=("${dtype}")
  fi
done
denoiser_report_paths=()
for denoiser_type in "${denoiser_types[@]:-}"; do
  denoiser_report_paths+=("$(denoiser_dir_for_type "${denoiser_type}")/run_report.json")
done
if [[ "${#denoiser_report_paths[@]}" -gt 0 ]]; then
  TARGET_DENOISING_PART_DENOISER_REPORTS="$(fresh_join_by_space "${denoiser_report_paths[@]}")"
  export TARGET_DENOISING_PART_DENOISER_REPORTS
fi

denoiser_job_ids=()
if fresh_bool_enabled "${TARGET_DENOISING_PART_SUBMIT_DENOISER}"; then
  for denoiser_type in "${denoiser_types[@]:-}"; do
    denoiser_output_dir="$(denoiser_dir_for_type "${denoiser_type}")"
    if fresh_bool_enabled "${SKIP_EXISTING}" && ! fresh_is_dry_run && denoiser_output_complete "${denoiser_output_dir}"; then
      echo "skipped target_denoising_denoiser_${denoiser_type}; found complete output: ${denoiser_output_dir}" >&2
      continue
    fi
    if fresh_bool_enabled "${SKIP_EXISTING}"; then
      archive_incomplete_output "${denoiser_output_dir}" denoiser_output_complete
    fi
    denoiser_env=(
      "TARGET_DENOISING_PART_DENOISER_OUTPUT_DIR=${denoiser_output_dir}"
      "TARGET_DENOISING_PART_OUTPUT_DIR=${denoiser_output_dir}"
    )
    while IFS= read -r flag_assignment; do
      [[ -n "${flag_assignment}" ]] && denoiser_env+=("${flag_assignment}")
    done < <(denoiser_flags_for_type "${denoiser_type}")
    denoiser_export_arg="$(export_arg_from_assignments "${denoiser_env[@]:-}")"
    mapfile -t denoiser_args < <(afterok_args "${TARGET_DENOISING_PART_UPSTREAM_DEPENDENCY}" "${denoiser_export_arg}" "${SCRIPT_DIR}/run_train_target_conditioned_denoising_part.sh")
    denoiser_jid="$(submit_job "target_denoising_denoiser_${denoiser_type}" "${denoiser_args[@]}")"
    echo "submitted target_denoising_denoiser_${denoiser_type}=${denoiser_jid}"
    denoiser_job_ids+=("${denoiser_jid}")
  done
else
  for denoiser_type in "${denoiser_types[@]:-}"; do
    fresh_require_file "$(denoiser_dir_for_type "${denoiser_type}")/best_denoiser_model_val.pt"
  done
fi

tagger_dependency="$(join_nonempty_by_colon "${TARGET_DENOISING_PART_UPSTREAM_DEPENDENCY}" "${denoiser_job_ids[@]:-}")"
tagger_job_ids=()
tagger_skip_count=0
if fresh_bool_enabled "${TARGET_DENOISING_PART_SUBMIT_TAGGERS}"; then
  for variant in "${variants[@]}"; do
    output_dir="${TARGET_DENOISING_PART_TAGGER_ROOT}/${variant}"
    if fresh_bool_enabled "${SKIP_EXISTING}" && ! fresh_is_dry_run && tagger_output_complete "${output_dir}"; then
      tagger_skip_count=$((tagger_skip_count + 1))
      echo "skipped target_denoising_tagger_${variant}; found complete output: ${output_dir}" >&2
      continue
    fi
    if fresh_bool_enabled "${SKIP_EXISTING}"; then
      archive_incomplete_output "${output_dir}" tagger_output_complete
    fi
    denoiser_type="$(denoiser_type_for_variant "${variant}")"
    tagger_env=()
    if [[ -n "${denoiser_type}" ]]; then
      tagger_env+=("TARGET_DENOISING_PART_DENOISER_CHECKPOINT=$(denoiser_dir_for_type "${denoiser_type}")/best_denoiser_model_val.pt")
    fi
    tagger_export_arg="$(export_arg_from_assignments "${tagger_env[@]:-}")"
    mapfile -t tagger_args < <(afterok_args "${tagger_dependency}" "${tagger_export_arg}" "${SCRIPT_DIR}/run_train_target_denoising_part_tagger.sh" "${variant}")
    tagger_jid="$(submit_job "target_denoising_tagger_${variant}" "${tagger_args[@]}")"
    echo "submitted target_denoising_tagger_${variant}=${tagger_jid}"
    tagger_job_ids+=("${tagger_jid}")
  done
fi

report_dependency="$(join_nonempty_by_colon "${tagger_job_ids[@]:-}" "${denoiser_job_ids[@]:-}")"
report_jid=""
if fresh_bool_enabled "${TARGET_DENOISING_PART_SUBMIT_REPORT}"; then
  report_inputs_submitted=0
  if [[ "${#tagger_job_ids[@]}" -gt 0 ]] || [[ "${#denoiser_job_ids[@]}" -gt 0 ]]; then
    report_inputs_submitted=1
  fi
  if fresh_bool_enabled "${SKIP_EXISTING}" && [[ "${report_inputs_submitted}" -eq 0 ]] && ! fresh_is_dry_run && report_output_complete "${TARGET_DENOISING_PART_REPORT_DIR}"; then
    echo "skipped target_denoising_report; found complete output: ${TARGET_DENOISING_PART_REPORT_DIR}" >&2
  else
    if fresh_bool_enabled "${SKIP_EXISTING}"; then
      if [[ "${report_inputs_submitted}" -eq 1 ]]; then
        archive_existing_output "${TARGET_DENOISING_PART_REPORT_DIR}" stale_report
      else
        archive_incomplete_output "${TARGET_DENOISING_PART_REPORT_DIR}" report_output_complete
      fi
    fi
    mapfile -t report_args < <(afterok_args "${report_dependency}" "${SCRIPT_DIR}/run_write_target_denoising_part_report.sh")
    report_jid="$(submit_job target_denoising_report "${report_args[@]}")"
    echo "submitted target_denoising_report=${report_jid}"
  fi
fi

cat <<SUMMARY
target_denoising_part_submission:
  root: ${TARGET_DENOISING_PART_ROOT}
  variants: ${TARGET_DENOISING_PART_VARIANTS}
  job_ids:
    denoisers: $(fresh_join_by_space "${denoiser_job_ids[@]:-}")
    taggers: $(fresh_join_by_space "${tagger_job_ids[@]:-}")
    final_report: ${report_jid:-skipped_or_existing}
  expected_jobs:
    denoiser_types: $(fresh_join_by_space "${denoiser_types[@]:-}")
    denoiser_reports: ${TARGET_DENOISING_PART_DENOISER_REPORTS:-}
    tagger_variants: ${TARGET_DENOISING_PART_VARIANTS}
    taggers_submitted: ${#tagger_job_ids[@]}
    taggers_skipped_existing: ${tagger_skip_count}
  outputs:
    denoisers: ${TARGET_DENOISING_PART_DENOISER_ROOT}
    taggers: ${TARGET_DENOISING_PART_TAGGER_ROOT}
    final_report: ${TARGET_DENOISING_PART_REPORT_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY

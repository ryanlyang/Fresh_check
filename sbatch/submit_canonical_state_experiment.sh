#!/usr/bin/env bash
# Submit one Canonical Multi-Scale Jet State campaign graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter

: "${CANONICAL_STATE_CAMPAIGN_MODE:=highdata}"
case "${CANONICAL_STATE_CAMPAIGN_MODE}" in
  highdata)
    : "${CANONICAL_STATE_MODEL_TRAIN_SIZE:=5000000}"
    : "${CANONICAL_STATE_MODEL_VAL_SIZE:=1000000}"
    : "${CANONICAL_STATE_STACK_TRAIN_SIZE:=3000000}"
    : "${CANONICAL_STATE_STACK_VAL_SIZE:=1000000}"
    : "${CANONICAL_STATE_FINAL_TEST_SIZE:=1000000}"
    ;;
  pilot)
    : "${CANONICAL_STATE_MODEL_TRAIN_SIZE:=500000}"
    : "${CANONICAL_STATE_MODEL_VAL_SIZE:=150000}"
    : "${CANONICAL_STATE_STACK_TRAIN_SIZE:=300000}"
    : "${CANONICAL_STATE_STACK_VAL_SIZE:=150000}"
    : "${CANONICAL_STATE_FINAL_TEST_SIZE:=150000}"
    ;;
  *)
    echo "CANONICAL_STATE_CAMPAIGN_MODE must be pilot or highdata, got ${CANONICAL_STATE_CAMPAIGN_MODE}" >&2
    exit 2
    ;;
esac

if [[ -z "${CANONICAL_STATE_ROOT:-}" ]]; then
  CANONICAL_STATE_ROOT="${OUTPUT_ROOT}/canonical_multi_scale_jet_state_hltv2_s2p5_${CANONICAL_STATE_CAMPAIGN_MODE}_$(date +%Y%m%d_%H%M%S)"
fi
: "${CANONICAL_STATE_INPUTS_DIR:=${CANONICAL_STATE_ROOT}/inputs}"
: "${CANONICAL_STATE_SPLIT_MANIFEST_DIR:=${CANONICAL_STATE_INPUTS_DIR}/split_manifest}"
: "${CANONICAL_STATE_MANIFEST_PATH:=${CANONICAL_STATE_SPLIT_MANIFEST_DIR}/split_manifest.json.gz}"
: "${CANONICAL_STATE_HLT_CACHE_DIR:=${CANONICAL_STATE_INPUTS_DIR}/hlt_cache}"
: "${CANONICAL_STATE_OFFLINE_CACHE_DIR:=${CANONICAL_STATE_INPUTS_DIR}/offline_cache}"
: "${CANONICAL_STATE_AUDIT_DIR:=${CANONICAL_STATE_INPUTS_DIR}/audits}"
: "${CANONICAL_STATE_PHI_CACHE_DIR:=${CANONICAL_STATE_ROOT}/phi_cache}"
: "${CANONICAL_STATE_RUN_ROOT:=${CANONICAL_STATE_ROOT}/runs}"
: "${CANONICAL_STATE_REPORT_DIR:=${CANONICAL_STATE_ROOT}/final_report}"
: "${CANONICAL_STATE_HLT_PROFILE:=fixed_hlt_v2_realistic}"
: "${CANONICAL_STATE_HLT_DEGRADATION_STRENGTH:=2.5}"
: "${CANONICAL_STATE_HLT_SPLITS:=model_train model_val stack_train stack_val final_test}"
: "${CANONICAL_STATE_OFFLINE_SPLITS:=${CANONICAL_STATE_HLT_SPLITS}}"
: "${CANONICAL_STATE_DATA_DIR:=${PD10_DATA_DIR}}"
: "${CANONICAL_STATE_SUBMIT_SPLITS:=1}"
: "${CANONICAL_STATE_SUBMIT_HLT_CACHE:=1}"
: "${CANONICAL_STATE_SUBMIT_OFFLINE_CACHE:=1}"
: "${CANONICAL_STATE_SUBMIT_AUDIT:=1}"
: "${CANONICAL_STATE_SUBMIT_PHI_HLT:=1}"
: "${CANONICAL_STATE_SUBMIT_PHI_OFFLINE:=1}"
: "${CANONICAL_STATE_SUBMIT_SINGLE_MODELS:=1}"
: "${CANONICAL_STATE_SUBMIT_FUSION:=1}"
: "${CANONICAL_STATE_SUBMIT_ORACLE_DIAGNOSTICS:=1}"
: "${CANONICAL_STATE_SUBMIT_REPORT:=1}"
: "${CANONICAL_STATE_PREQUEUE_VALIDATE_INPUTS:=1}"
: "${CANONICAL_STATE_EMIT_PLANNING_STUB:=0}"
: "${CANONICAL_STATE_RUN_IDS:=A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 C4 C5 C6 D0 D1 D2 D3 D4 D5 E0 E1 E2 E3 E4 E5 E6 F0 F1 F2 F3 F4 Fseed Fshuffle G0 G1 G2 G3}"
: "${CANONICAL_STATE_SINGLE_RUN_IDS:=A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 C4 C5 C6 D0 D1 D2 D3 D4 D5 E0 E1 E2 E3 E4 E5 E6}"
: "${CANONICAL_STATE_FUSION_RUN_IDS:=F0 F1 F2 F3 F4 Fseed Fshuffle}"
: "${CANONICAL_STATE_ORACLE_RUN_IDS:=G0 G1 G2 G3}"
: "${CANONICAL_STATE_SBATCH_PARTITION:=}"
: "${CANONICAL_STATE_GPU_GRES:=}"
: "${CANONICAL_STATE_GPU_CPUS_PER_TASK:=}"
: "${CANONICAL_STATE_GPU_MEM:=}"
: "${CANONICAL_STATE_CPU_CPUS_PER_TASK:=}"
: "${CANONICAL_STATE_CPU_MEM:=}"
: "${CANONICAL_STATE_CHECKPOINT_POLICY:=all}"
: "${CANONICAL_STATE_SAVE_LAST_CHECKPOINT:=1}"

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

submit_job() {
  local label="$1"
  shift
  local sbatch_args=()
  if [[ -n "${CANONICAL_STATE_SBATCH_PARTITION}" ]]; then
    sbatch_args+=(--partition="${CANONICAL_STATE_SBATCH_PARTITION}")
  fi
  local gpu_job=0
  local arg
  for arg in "$@"; do
    case "${arg}" in
      */run_canonical_state_variant.sh|run_canonical_state_variant.sh)
        gpu_job=1
        ;;
    esac
  done
  if [[ "${gpu_job}" -eq 1 ]]; then
    if [[ -n "${CANONICAL_STATE_GPU_GRES}" ]]; then
      sbatch_args+=(--gres="${CANONICAL_STATE_GPU_GRES}")
    fi
    if [[ -n "${CANONICAL_STATE_GPU_CPUS_PER_TASK}" ]]; then
      sbatch_args+=(--cpus-per-task="${CANONICAL_STATE_GPU_CPUS_PER_TASK}")
    fi
    if [[ -n "${CANONICAL_STATE_GPU_MEM}" ]]; then
      sbatch_args+=(--mem="${CANONICAL_STATE_GPU_MEM}")
    fi
  else
    if [[ -n "${CANONICAL_STATE_CPU_CPUS_PER_TASK}" ]]; then
      sbatch_args+=(--cpus-per-task="${CANONICAL_STATE_CPU_CPUS_PER_TASK}")
    fi
    if [[ -n "${CANONICAL_STATE_CPU_MEM}" ]]; then
      sbatch_args+=(--mem="${CANONICAL_STATE_CPU_MEM}")
    fi
  fi
  if fresh_is_dry_run; then
    printf 'DRY_RUN sbatch %s: ' "${label}" >&2
    fresh_print_shell_command sbatch "${sbatch_args[@]}" "$@" >&2
    printf '\n' >&2
    local clean_label="${label//[^A-Za-z0-9_]/_}"
    printf 'DRYRUN_%s\n' "${clean_label}"
    return 0
  fi
  local output
  if ! output="$(sbatch "${sbatch_args[@]}" "$@")"; then
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
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

canonical_variant_output_complete() {
  local run_id="$1"
  local output_dir="$2"
  [[ -f "${output_dir}/run_report.json" ]] || return 1
  case "${run_id}" in
    F*)
      # All canonical-state variants, including fusion/prototype fusion runs,
      # write the canonical run report as the completion artifact.
      return 0
      ;;
    G*)
      [[ -f "${output_dir}/oracle_diagnostics.json" || -f "${output_dir}/run_report.json" ]] || return 1
      ;;
    C*)
      [[ -f "${output_dir}/model_val_report.json" || -f "${output_dir}/state_prediction_report.json" || -f "${output_dir}/run_report.json" ]] || return 1
      ;;
    *)
      [[ -f "${output_dir}/best_model_val.pt" || -f "${output_dir}/run_report.json" ]] || return 1
      ;;
  esac
}

archive_incomplete_variant_output() {
  local run_id="$1"
  local output_dir="$2"
  if fresh_is_dry_run || [[ ! -d "${output_dir}" ]]; then
    return 0
  fi
  if canonical_variant_output_complete "${run_id}" "${output_dir}"; then
    return 0
  fi
  local archived="${output_dir}_incomplete_$(date +%Y%m%d_%H%M%S)"
  echo "found incomplete canonical-state output; moving ${output_dir} to ${archived}" >&2
  mv "${output_dir}" "${archived}"
}

prequeue_validate_inputs() {
  if fresh_is_dry_run || ! fresh_bool_enabled "${CANONICAL_STATE_PREQUEUE_VALIDATE_INPUTS}"; then
    return 0
  fi
  "${PYTHON_BIN}" "scripts/audit_canonical_state_step1_inputs.py" \
    --manifest "${CANONICAL_STATE_MANIFEST_PATH}" \
    --hlt-cache-dir "${CANONICAL_STATE_HLT_CACHE_DIR}" \
    --output-dir "${CANONICAL_STATE_AUDIT_DIR}/prequeue" \
    --expected-model-train "${CANONICAL_STATE_MODEL_TRAIN_SIZE}" \
    --expected-model-val "${CANONICAL_STATE_MODEL_VAL_SIZE}" \
    --expected-stack-train "${CANONICAL_STATE_STACK_TRAIN_SIZE}" \
    --expected-stack-val "${CANONICAL_STATE_STACK_VAL_SIZE}" \
    --expected-final-test "${CANONICAL_STATE_FINAL_TEST_SIZE}" >/dev/null
}

submitter_log_dir="${CANONICAL_STATE_ROOT}/submission_logs/canonical_state_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_log_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "root=${CANONICAL_STATE_ROOT}"
    echo "mode=${CANONICAL_STATE_CAMPAIGN_MODE}"
    echo "data_dir=${CANONICAL_STATE_DATA_DIR}"
    echo "hlt_profile=${CANONICAL_STATE_HLT_PROFILE}"
    echo "hlt_degradation_strength=${CANONICAL_STATE_HLT_DEGRADATION_STRENGTH}"
    echo "model_train=${CANONICAL_STATE_MODEL_TRAIN_SIZE}"
    echo "model_val=${CANONICAL_STATE_MODEL_VAL_SIZE}"
    echo "stack_train=${CANONICAL_STATE_STACK_TRAIN_SIZE}"
    echo "stack_val=${CANONICAL_STATE_STACK_VAL_SIZE}"
    echo "final_test=${CANONICAL_STATE_FINAL_TEST_SIZE}"
    echo "run_ids=${CANONICAL_STATE_RUN_IDS}"
    echo "sbatch_partition=${CANONICAL_STATE_SBATCH_PARTITION}"
    echo "gpu_gres=${CANONICAL_STATE_GPU_GRES}"
    echo "gpu_cpus_per_task=${CANONICAL_STATE_GPU_CPUS_PER_TASK}"
    echo "gpu_mem=${CANONICAL_STATE_GPU_MEM}"
    echo "cpu_cpus_per_task=${CANONICAL_STATE_CPU_CPUS_PER_TASK}"
    echo "cpu_mem=${CANONICAL_STATE_CPU_MEM}"
    echo "checkpoint_policy=${CANONICAL_STATE_CHECKPOINT_POLICY}"
    echo "save_last_checkpoint=${CANONICAL_STATE_SAVE_LAST_CHECKPOINT}"
  } > "${submitter_log_dir}/metadata.txt"
fi

echo "canonical_state_submission_start:"
echo "  root: ${CANONICAL_STATE_ROOT}"
echo "  mode: ${CANONICAL_STATE_CAMPAIGN_MODE}"
echo "  data_dir: ${CANONICAL_STATE_DATA_DIR}"
echo "  hlt_profile: ${CANONICAL_STATE_HLT_PROFILE}"
echo "  hlt_degradation_strength: ${CANONICAL_STATE_HLT_DEGRADATION_STRENGTH}"
echo "  sizes: ${CANONICAL_STATE_MODEL_TRAIN_SIZE}/${CANONICAL_STATE_MODEL_VAL_SIZE}/${CANONICAL_STATE_STACK_TRAIN_SIZE}/${CANONICAL_STATE_STACK_VAL_SIZE}/${CANONICAL_STATE_FINAL_TEST_SIZE}"
if [[ -n "${CANONICAL_STATE_SBATCH_PARTITION}" ]]; then
  echo "  sbatch_partition: ${CANONICAL_STATE_SBATCH_PARTITION}"
fi
if [[ -n "${CANONICAL_STATE_GPU_GRES}" ]]; then
  echo "  gpu_gres: ${CANONICAL_STATE_GPU_GRES}"
fi
echo "  checkpoint_policy: ${CANONICAL_STATE_CHECKPOINT_POLICY}"
echo "  save_last_checkpoint: ${CANONICAL_STATE_SAVE_LAST_CHECKPOINT}"

split_jid=""
if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_SPLITS}"; then
  split_jid="$(submit_job "canonical_state_splits" \
    --export=ALL,PD10_DATA_DIR="${CANONICAL_STATE_DATA_DIR}",PD10_MODEL_TRAIN_SIZE="${CANONICAL_STATE_MODEL_TRAIN_SIZE}",PD10_MODEL_VAL_SIZE="${CANONICAL_STATE_MODEL_VAL_SIZE}",PD10_STACK_TRAIN_SIZE="${CANONICAL_STATE_STACK_TRAIN_SIZE}",PD10_STACK_VAL_SIZE="${CANONICAL_STATE_STACK_VAL_SIZE}",PD10_FINAL_TEST_SIZE="${CANONICAL_STATE_FINAL_TEST_SIZE}",PD10_SPLIT_MANIFEST_DIR="${CANONICAL_STATE_SPLIT_MANIFEST_DIR}",PD10_MANIFEST_PATH="${CANONICAL_STATE_MANIFEST_PATH}" \
    "${SCRIPT_DIR}/run_pd10_build_splits.sh")"
  echo "submitted canonical_state_splits=${split_jid}"
else
  fresh_require_file "${CANONICAL_STATE_MANIFEST_PATH}"
fi

hlt_dep="${split_jid}"
hlt_cache_jid=""
if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_HLT_CACHE}"; then
  mapfile -t hlt_args < <(afterok_args "${hlt_dep}" \
    --export=ALL,PD10_DATA_DIR="${CANONICAL_STATE_DATA_DIR}",PD10_MANIFEST_PATH="${CANONICAL_STATE_MANIFEST_PATH}",PD10_HLT_CACHE_DIR="${CANONICAL_STATE_HLT_CACHE_DIR}",PD10_HLT_SPLITS="${CANONICAL_STATE_HLT_SPLITS}",PD10_HLT_PROFILE="${CANONICAL_STATE_HLT_PROFILE}",PD10_HLT_DEGRADATION_STRENGTH="${CANONICAL_STATE_HLT_DEGRADATION_STRENGTH}" \
    "${SCRIPT_DIR}/run_pd10_build_hlt_cache.sh")
  hlt_cache_jid="$(submit_job "canonical_state_hlt_cache" "${hlt_args[@]}")"
  echo "submitted canonical_state_hlt_cache=${hlt_cache_jid}"
else
  fresh_require_dir "${CANONICAL_STATE_HLT_CACHE_DIR}"
fi

offline_dep="${split_jid}"
offline_cache_jid=""
if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_OFFLINE_CACHE}"; then
  mapfile -t offline_args < <(afterok_args "${offline_dep}" \
    --export=ALL,ARCHITECTURE_VIEW_10CLASS_OFFLINE_ROOT="${CANONICAL_STATE_ROOT}",ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH="${CANONICAL_STATE_MANIFEST_PATH}",ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR="${CANONICAL_STATE_OFFLINE_CACHE_DIR}",ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS="${CANONICAL_STATE_OFFLINE_SPLITS}",ARCHITECTURE_VIEW_10CLASS_OFFLINE_DATA_DIRS="${CANONICAL_STATE_DATA_DIR}" \
    "${SCRIPT_DIR}/run_cache_architecture_view_offline_inputs.sh")
  offline_cache_jid="$(submit_job "canonical_state_offline_cache" "${offline_args[@]}")"
  echo "submitted canonical_state_offline_cache=${offline_cache_jid}"
else
  if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_PHI_OFFLINE}"; then
    fresh_require_dir "${CANONICAL_STATE_OFFLINE_CACHE_DIR}"
  fi
fi

audit_dep="$(join_nonempty_by_colon "${hlt_cache_jid}" "${offline_cache_jid}")"
audit_jid=""
if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_AUDIT}"; then
  mapfile -t audit_args < <(afterok_args "${audit_dep}" \
    --export=ALL,CANONICAL_STATE_ROOT="${CANONICAL_STATE_ROOT}",CANONICAL_STATE_MANIFEST_PATH="${CANONICAL_STATE_MANIFEST_PATH}",CANONICAL_STATE_HLT_CACHE_DIR="${CANONICAL_STATE_HLT_CACHE_DIR}",CANONICAL_STATE_AUDIT_DIR="${CANONICAL_STATE_AUDIT_DIR}",CANONICAL_STATE_MODEL_TRAIN_SIZE="${CANONICAL_STATE_MODEL_TRAIN_SIZE}",CANONICAL_STATE_MODEL_VAL_SIZE="${CANONICAL_STATE_MODEL_VAL_SIZE}",CANONICAL_STATE_STACK_TRAIN_SIZE="${CANONICAL_STATE_STACK_TRAIN_SIZE}",CANONICAL_STATE_STACK_VAL_SIZE="${CANONICAL_STATE_STACK_VAL_SIZE}",CANONICAL_STATE_FINAL_TEST_SIZE="${CANONICAL_STATE_FINAL_TEST_SIZE}" \
    "${SCRIPT_DIR}/run_canonical_state_audit_inputs.sh")
  audit_jid="$(submit_job "canonical_state_audit" "${audit_args[@]}")"
  echo "submitted canonical_state_audit=${audit_jid}"
else
  prequeue_validate_inputs
fi

phi_base_dep="$(join_nonempty_by_colon "${audit_jid}")"
phi_hlt_jid=""
if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_PHI_HLT}"; then
  mapfile -t phi_hlt_args < <(afterok_args "${phi_base_dep}" \
    --export=ALL,CANONICAL_STATE_ROOT="${CANONICAL_STATE_ROOT}",CANONICAL_STATE_MANIFEST_PATH="${CANONICAL_STATE_MANIFEST_PATH}",CANONICAL_STATE_HLT_CACHE_DIR="${CANONICAL_STATE_HLT_CACHE_DIR}",CANONICAL_STATE_PHI_CACHE_DIR="${CANONICAL_STATE_PHI_CACHE_DIR}",CANONICAL_STATE_MODEL_TRAIN_SIZE="${CANONICAL_STATE_MODEL_TRAIN_SIZE}",CANONICAL_STATE_MODEL_VAL_SIZE="${CANONICAL_STATE_MODEL_VAL_SIZE}",CANONICAL_STATE_STACK_TRAIN_SIZE="${CANONICAL_STATE_STACK_TRAIN_SIZE}",CANONICAL_STATE_STACK_VAL_SIZE="${CANONICAL_STATE_STACK_VAL_SIZE}",CANONICAL_STATE_FINAL_TEST_SIZE="${CANONICAL_STATE_FINAL_TEST_SIZE}" \
    "${SCRIPT_DIR}/run_canonical_state_cache_phi.sh" hlt)
  phi_hlt_jid="$(submit_job "canonical_state_phi_hlt" "${phi_hlt_args[@]}")"
  echo "submitted canonical_state_phi_hlt=${phi_hlt_jid}"
fi

phi_offline_jid=""
if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_PHI_OFFLINE}"; then
  mapfile -t phi_off_args < <(afterok_args "${phi_base_dep}" \
    --export=ALL,CANONICAL_STATE_ROOT="${CANONICAL_STATE_ROOT}",CANONICAL_STATE_MANIFEST_PATH="${CANONICAL_STATE_MANIFEST_PATH}",CANONICAL_STATE_OFFLINE_CACHE_DIR="${CANONICAL_STATE_OFFLINE_CACHE_DIR}",CANONICAL_STATE_PHI_CACHE_DIR="${CANONICAL_STATE_PHI_CACHE_DIR}",CANONICAL_STATE_MODEL_TRAIN_SIZE="${CANONICAL_STATE_MODEL_TRAIN_SIZE}",CANONICAL_STATE_MODEL_VAL_SIZE="${CANONICAL_STATE_MODEL_VAL_SIZE}",CANONICAL_STATE_STACK_TRAIN_SIZE="${CANONICAL_STATE_STACK_TRAIN_SIZE}",CANONICAL_STATE_STACK_VAL_SIZE="${CANONICAL_STATE_STACK_VAL_SIZE}",CANONICAL_STATE_FINAL_TEST_SIZE="${CANONICAL_STATE_FINAL_TEST_SIZE}" \
    "${SCRIPT_DIR}/run_canonical_state_cache_phi.sh" offline)
  phi_offline_jid="$(submit_job "canonical_state_phi_offline" "${phi_off_args[@]}")"
  echo "submitted canonical_state_phi_offline=${phi_offline_jid}"
fi

phi_dep="$(join_nonempty_by_colon "${phi_hlt_jid}" "${phi_offline_jid}")"
fresh_split_words single_run_ids "${CANONICAL_STATE_SINGLE_RUN_IDS}"
single_job_ids=()
if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_SINGLE_MODELS}"; then
  for run_id in "${single_run_ids[@]}"; do
    output_dir="${CANONICAL_STATE_RUN_ROOT}/${run_id}"
    if fresh_bool_enabled "${SKIP_EXISTING}" && ! fresh_is_dry_run && canonical_variant_output_complete "${run_id}" "${output_dir}"; then
      echo "skipped canonical_state_${run_id}; complete output exists: ${output_dir}" >&2
      continue
    fi
    if fresh_bool_enabled "${SKIP_EXISTING}"; then
      archive_incomplete_variant_output "${run_id}" "${output_dir}"
    fi
    dep="${phi_dep}"
    if [[ "${run_id}" != "A0" && "${#single_job_ids[@]}" -gt 0 ]]; then
      dep="$(join_nonempty_by_colon "${phi_dep}" "${single_job_ids[0]}")"
    fi
    mapfile -t variant_args < <(afterok_args "${dep}" \
      --export=ALL,CANONICAL_STATE_ROOT="${CANONICAL_STATE_ROOT}",CANONICAL_STATE_MANIFEST_PATH="${CANONICAL_STATE_MANIFEST_PATH}",CANONICAL_STATE_HLT_CACHE_DIR="${CANONICAL_STATE_HLT_CACHE_DIR}",CANONICAL_STATE_PHI_HLT_CACHE_DIR="${CANONICAL_STATE_PHI_CACHE_DIR}/hlt",CANONICAL_STATE_PHI_OFFLINE_CACHE_DIR="${CANONICAL_STATE_PHI_CACHE_DIR}/offline",CANONICAL_STATE_RUN_ROOT="${CANONICAL_STATE_RUN_ROOT}",CANONICAL_STATE_EMIT_PLANNING_STUB="${CANONICAL_STATE_EMIT_PLANNING_STUB}",CANONICAL_STATE_MODEL_TRAIN_SIZE="${CANONICAL_STATE_MODEL_TRAIN_SIZE}",CANONICAL_STATE_MODEL_VAL_SIZE="${CANONICAL_STATE_MODEL_VAL_SIZE}",CANONICAL_STATE_STACK_TRAIN_SIZE="${CANONICAL_STATE_STACK_TRAIN_SIZE}",CANONICAL_STATE_STACK_VAL_SIZE="${CANONICAL_STATE_STACK_VAL_SIZE}",CANONICAL_STATE_FINAL_TEST_SIZE="${CANONICAL_STATE_FINAL_TEST_SIZE}",CONFIRM_FINAL_TEST="${CONFIRM_FINAL_TEST}" \
      "${SCRIPT_DIR}/run_canonical_state_variant.sh" "${run_id}")
    jid="$(submit_job "canonical_state_${run_id}" "${variant_args[@]}")"
    echo "submitted canonical_state_${run_id}=${jid}"
    single_job_ids+=("${jid}")
  done
fi

single_dep="$(join_nonempty_by_colon "${single_job_ids[@]:-}")"
fusion_job_ids=()
fresh_split_words fusion_run_ids "${CANONICAL_STATE_FUSION_RUN_IDS}"
if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_FUSION}"; then
  for run_id in "${fusion_run_ids[@]}"; do
    output_dir="${CANONICAL_STATE_RUN_ROOT}/${run_id}"
    if fresh_bool_enabled "${SKIP_EXISTING}" && ! fresh_is_dry_run && canonical_variant_output_complete "${run_id}" "${output_dir}"; then
      echo "skipped canonical_state_${run_id}; complete output exists: ${output_dir}" >&2
      continue
    fi
    if fresh_bool_enabled "${SKIP_EXISTING}"; then
      archive_incomplete_variant_output "${run_id}" "${output_dir}"
    fi
    mapfile -t fusion_args < <(afterok_args "${single_dep}" \
      --export=ALL,CANONICAL_STATE_ROOT="${CANONICAL_STATE_ROOT}",CANONICAL_STATE_MANIFEST_PATH="${CANONICAL_STATE_MANIFEST_PATH}",CANONICAL_STATE_HLT_CACHE_DIR="${CANONICAL_STATE_HLT_CACHE_DIR}",CANONICAL_STATE_PHI_HLT_CACHE_DIR="${CANONICAL_STATE_PHI_CACHE_DIR}/hlt",CANONICAL_STATE_PHI_OFFLINE_CACHE_DIR="${CANONICAL_STATE_PHI_CACHE_DIR}/offline",CANONICAL_STATE_RUN_ROOT="${CANONICAL_STATE_RUN_ROOT}",CANONICAL_STATE_EMIT_PLANNING_STUB="${CANONICAL_STATE_EMIT_PLANNING_STUB}",CANONICAL_STATE_MODEL_TRAIN_SIZE="${CANONICAL_STATE_MODEL_TRAIN_SIZE}",CANONICAL_STATE_MODEL_VAL_SIZE="${CANONICAL_STATE_MODEL_VAL_SIZE}",CANONICAL_STATE_STACK_TRAIN_SIZE="${CANONICAL_STATE_STACK_TRAIN_SIZE}",CANONICAL_STATE_STACK_VAL_SIZE="${CANONICAL_STATE_STACK_VAL_SIZE}",CANONICAL_STATE_FINAL_TEST_SIZE="${CANONICAL_STATE_FINAL_TEST_SIZE}",CONFIRM_FINAL_TEST="${CONFIRM_FINAL_TEST}" \
      "${SCRIPT_DIR}/run_canonical_state_variant.sh" "${run_id}")
    jid="$(submit_job "canonical_state_${run_id}" "${fusion_args[@]}")"
    echo "submitted canonical_state_${run_id}=${jid}"
    fusion_job_ids+=("${jid}")
  done
fi

oracle_job_ids=()
fresh_split_words oracle_run_ids "${CANONICAL_STATE_ORACLE_RUN_IDS}"
if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_ORACLE_DIAGNOSTICS}"; then
  oracle_dep="$(join_nonempty_by_colon "${single_dep}" "${fusion_job_ids[@]:-}")"
  for run_id in "${oracle_run_ids[@]}"; do
    mapfile -t oracle_args < <(afterok_args "${oracle_dep}" \
      --export=ALL,CANONICAL_STATE_ROOT="${CANONICAL_STATE_ROOT}",CANONICAL_STATE_MANIFEST_PATH="${CANONICAL_STATE_MANIFEST_PATH}",CANONICAL_STATE_HLT_CACHE_DIR="${CANONICAL_STATE_HLT_CACHE_DIR}",CANONICAL_STATE_PHI_HLT_CACHE_DIR="${CANONICAL_STATE_PHI_CACHE_DIR}/hlt",CANONICAL_STATE_PHI_OFFLINE_CACHE_DIR="${CANONICAL_STATE_PHI_CACHE_DIR}/offline",CANONICAL_STATE_RUN_ROOT="${CANONICAL_STATE_RUN_ROOT}",CANONICAL_STATE_EMIT_PLANNING_STUB="${CANONICAL_STATE_EMIT_PLANNING_STUB}",CANONICAL_STATE_MODEL_TRAIN_SIZE="${CANONICAL_STATE_MODEL_TRAIN_SIZE}",CANONICAL_STATE_MODEL_VAL_SIZE="${CANONICAL_STATE_MODEL_VAL_SIZE}",CANONICAL_STATE_STACK_TRAIN_SIZE="${CANONICAL_STATE_STACK_TRAIN_SIZE}",CANONICAL_STATE_STACK_VAL_SIZE="${CANONICAL_STATE_STACK_VAL_SIZE}",CANONICAL_STATE_FINAL_TEST_SIZE="${CANONICAL_STATE_FINAL_TEST_SIZE}",CONFIRM_FINAL_TEST="${CONFIRM_FINAL_TEST}" \
      "${SCRIPT_DIR}/run_canonical_state_variant.sh" "${run_id}")
    jid="$(submit_job "canonical_state_${run_id}" "${oracle_args[@]}")"
    echo "submitted canonical_state_${run_id}=${jid}"
    oracle_job_ids+=("${jid}")
  done
fi

report_dep="$(join_nonempty_by_colon "${single_job_ids[@]:-}" "${fusion_job_ids[@]:-}" "${oracle_job_ids[@]:-}")"
report_jid=""
if fresh_bool_enabled "${CANONICAL_STATE_SUBMIT_REPORT}"; then
  mapfile -t report_args < <(afterok_args "${report_dep}" \
    --export=ALL,CANONICAL_STATE_ROOT="${CANONICAL_STATE_ROOT}",CANONICAL_STATE_RUN_ROOT="${CANONICAL_STATE_RUN_ROOT}",CANONICAL_STATE_REPORT_DIR="${CANONICAL_STATE_REPORT_DIR}",CANONICAL_STATE_REPORT_RUN_IDS="${CANONICAL_STATE_RUN_IDS}",CONFIRM_FINAL_TEST="${CONFIRM_FINAL_TEST}" \
    "${SCRIPT_DIR}/run_write_canonical_state_report.sh")
  report_jid="$(submit_job "canonical_state_report" "${report_args[@]}")"
  echo "submitted canonical_state_report=${report_jid}"
fi

cat <<SUMMARY
canonical_state_submission_complete:
  root: ${CANONICAL_STATE_ROOT}
  mode: ${CANONICAL_STATE_CAMPAIGN_MODE}
  manifest: ${CANONICAL_STATE_MANIFEST_PATH}
  hlt_cache: ${CANONICAL_STATE_HLT_CACHE_DIR}
  offline_cache: ${CANONICAL_STATE_OFFLINE_CACHE_DIR}
  phi_cache: ${CANONICAL_STATE_PHI_CACHE_DIR}
  run_root: ${CANONICAL_STATE_RUN_ROOT}
  final_report: ${CANONICAL_STATE_REPORT_DIR}
  dependency_summary:
    split: ${split_jid:-none}
    hlt_cache: ${hlt_cache_jid:-none}
    offline_cache: ${offline_cache_jid:-none}
    audit: ${audit_jid:-none}
    phi_hlt: ${phi_hlt_jid:-none}
    phi_offline: ${phi_offline_jid:-none}
    single_models: ${single_job_ids[*]:-none}
    fusion: ${fusion_job_ids[*]:-none}
    oracle: ${oracle_job_ids[*]:-none}
    report: ${report_jid:-none}
  expected_jobs:
    inputs: canonical_state_splits canonical_state_hlt_cache canonical_state_offline_cache canonical_state_audit canonical_state_phi_hlt canonical_state_phi_offline
    single_models: ${CANONICAL_STATE_SINGLE_RUN_IDS}
    fusion: ${CANONICAL_STATE_FUSION_RUN_IDS}
    oracle_diagnostics: ${CANONICAL_STATE_ORACLE_RUN_IDS}
    report: canonical_state_report
SUMMARY

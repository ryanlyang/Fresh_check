#!/usr/bin/env bash
# Submit 10-class HLT0.6 standalone-vs-Architecture-View comparison.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_HLT_DEGRADATION_STRENGTH:=0.6}"
ARCHITECTURE_VIEW_PART_10CLASS_HLT_TAG="${ARCHITECTURE_VIEW_PART_10CLASS_HLT_DEGRADATION_STRENGTH//./p}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_ROOT:=${OUTPUT_ROOT}/architecture_view_part_10class_hlt${ARCHITECTURE_VIEW_PART_10CLASS_HLT_TAG}_${ARCHITECTURE_VIEW_PART_10CLASS_TAG}}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_INPUT_ROOT:=${ARCHITECTURE_VIEW_PART_10CLASS_ROOT}/inputs}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_MANIFEST_PATH:=${ARCHITECTURE_VIEW_PART_10CLASS_INPUT_ROOT}/split_manifest.json.gz}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_HLT_CACHE_DIR:=${ARCHITECTURE_VIEW_PART_10CLASS_INPUT_ROOT}/hlt_cache}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_ROOT:=${ARCHITECTURE_VIEW_PART_10CLASS_ROOT}/standalone_hlt4}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_MODEL_ROOT:=${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_ROOT}/models}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_FUSION_DIR:=${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_ROOT}/fusion_run}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_AV_ROOT:=${ARCHITECTURE_VIEW_PART_10CLASS_ROOT}/architecture_view}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_AV_TAGGER_ROOT:=${ARCHITECTURE_VIEW_PART_10CLASS_AV_ROOT}/taggers}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_AV_FUSION_DIR:=${ARCHITECTURE_VIEW_PART_10CLASS_AV_ROOT}/fusion_run}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_AV_FINAL_REPORT_DIR:=${ARCHITECTURE_VIEW_PART_10CLASS_AV_ROOT}/final_report}"

: "${ARCHITECTURE_VIEW_PART_10CLASS_LABEL_NAMES:=QCD Hbb Hcc Hgg H4q Hqql Zqq Wqq Tbqq Tbl}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_HETERO_ARCHITECTURES:=part pn pfn pcnn}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_AV_VARIANTS:=av10_baseline_recheck av10_context_mlp_control av10_part_context_to_part av10_pn_context_to_part av10_pfn_context_to_part av10_pcnn_context_to_part av10_all_views_to_part av10_random_view_control}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_AV_FUSION_VARIANTS:=av10_baseline_recheck av10_context_mlp_control av10_part_context_to_part av10_pn_context_to_part av10_pfn_context_to_part av10_pcnn_context_to_part av10_all_views_to_part}"

: "${ARCHITECTURE_VIEW_PART_10CLASS_MODEL_TRAIN_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_MODEL_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_STACK_TRAIN_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_STACK_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_FINAL_TEST_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_EPOCHS:=45}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_SELECTION_METRIC:=accuracy}"

: "${ARCHITECTURE_VIEW_PART_10CLASS_SPLIT_TIME:=04:00:00}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_CACHE_TIME:=1-00:00:00}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_HETERO_TIME:=2-00:00:00}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_AV_TRAIN_TIME:=3-00:00:00}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_FUSION_TIME:=23:00:00}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_REPORT_TIME:=02:00:00}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_SPLIT_MEM:=32G}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_CACHE_MEM:=160G}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_TRAIN_MEM:=160G}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_FUSION_MEM:=160G}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_REPORT_MEM:=8G}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_SPLIT_CPUS:=4}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_CACHE_CPUS:=8}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_TRAIN_CPUS:=8}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_FUSION_CPUS:=8}"
: "${ARCHITECTURE_VIEW_PART_10CLASS_REPORT_CPUS:=2}"

export MANIFEST_PATH="${ARCHITECTURE_VIEW_PART_10CLASS_MANIFEST_PATH}"
export HLT_CACHE_DIR="${ARCHITECTURE_VIEW_PART_10CLASS_HLT_CACHE_DIR}"
export HLT_DEGRADATION_STRENGTH="${ARCHITECTURE_VIEW_PART_10CLASS_HLT_DEGRADATION_STRENGTH}"
export MODEL_TRAIN_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_MODEL_TRAIN_SIZE}"
export MODEL_VAL_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_MODEL_VAL_SIZE}"
export STACK_TRAIN_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_STACK_TRAIN_SIZE}"
export STACK_VAL_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_STACK_VAL_SIZE}"
export FINAL_TEST_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_FINAL_TEST_SIZE}"

export HETERO_HLT4_ROOT="${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_ROOT}"
export HETERO_HLT4_MODEL_ROOT="${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_MODEL_ROOT}"
export HETERO_HLT4_FUSION_DIR="${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_FUSION_DIR}"
export HETERO_HLT4_ARCHITECTURES="${ARCHITECTURE_VIEW_PART_10CLASS_HETERO_ARCHITECTURES}"
export HETERO_HLT4_TRAIN_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_MODEL_TRAIN_SIZE}"
export HETERO_HLT4_VAL_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_MODEL_VAL_SIZE}"
export HETERO_HLT4_STACK_TRAIN_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_STACK_TRAIN_SIZE}"
export HETERO_HLT4_STACK_VAL_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_STACK_VAL_SIZE}"
export HETERO_HLT4_FINAL_TEST_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_FINAL_TEST_SIZE}"

export ARCHITECTURE_VIEW_PART_ROOT="${ARCHITECTURE_VIEW_PART_10CLASS_AV_ROOT}"
export ARCHITECTURE_VIEW_PART_TAGGER_ROOT="${ARCHITECTURE_VIEW_PART_10CLASS_AV_TAGGER_ROOT}"
export ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR="${ARCHITECTURE_VIEW_PART_10CLASS_AV_FINAL_REPORT_DIR}"
export ARCHITECTURE_VIEW_PART_MANIFEST_PATH="${ARCHITECTURE_VIEW_PART_10CLASS_MANIFEST_PATH}"
export ARCHITECTURE_VIEW_PART_HLT_CACHE_DIR="${ARCHITECTURE_VIEW_PART_10CLASS_HLT_CACHE_DIR}"
export ARCHITECTURE_VIEW_PART_BASELINE_CHECKPOINT="${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_MODEL_ROOT}/part/best_model_val.pt"
export ARCHITECTURE_VIEW_PART_LABEL_NAMES="${ARCHITECTURE_VIEW_PART_10CLASS_LABEL_NAMES}"
export ARCHITECTURE_VIEW_PART_LABEL_FILTER_NAMES="${ARCHITECTURE_VIEW_PART_10CLASS_LABEL_NAMES}"
export ARCHITECTURE_VIEW_PART_NUM_CLASSES=10
export ARCHITECTURE_VIEW_PART_MODEL_TRAIN_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_MODEL_TRAIN_SIZE}"
export ARCHITECTURE_VIEW_PART_MODEL_VAL_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_MODEL_VAL_SIZE}"
export ARCHITECTURE_VIEW_PART_STACK_VAL_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_STACK_VAL_SIZE}"
export ARCHITECTURE_VIEW_PART_FINAL_TEST_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_FINAL_TEST_SIZE}"
export ARCHITECTURE_VIEW_PART_EPOCHS="${ARCHITECTURE_VIEW_PART_10CLASS_EPOCHS}"
export ARCHITECTURE_VIEW_PART_SELECTION_METRIC="${ARCHITECTURE_VIEW_PART_10CLASS_SELECTION_METRIC}"
export ARCHITECTURE_VIEW_PART_EXPECTED_HLT_DEGRADATION_STRENGTH="${ARCHITECTURE_VIEW_PART_10CLASS_HLT_DEGRADATION_STRENGTH}"
export ARCHITECTURE_VIEW_PART_REPORT_VARIANTS="${ARCHITECTURE_VIEW_PART_10CLASS_AV_VARIANTS}"
export ARCHITECTURE_VIEW_PART_REPORT_PRIMARY_METRIC="${ARCHITECTURE_VIEW_PART_10CLASS_SELECTION_METRIC}"
export ARCHITECTURE_VIEW_PART_REPORT_COMPARISON_SPLIT="final_test"
export ARCHITECTURE_VIEW_PART_REPORT_BASELINE_VARIANT="av10_baseline_recheck"
export ARCHITECTURE_VIEW_PART_REPORT_CONFIRM_FINAL_TEST=1
export ARCHITECTURE_VIEW_PART_FUSION_DIR="${ARCHITECTURE_VIEW_PART_10CLASS_AV_FUSION_DIR}"
export ARCHITECTURE_VIEW_PART_FUSION_VARIANTS="${ARCHITECTURE_VIEW_PART_10CLASS_AV_FUSION_VARIANTS}"
export ARCHITECTURE_VIEW_PART_FUSION_STACK_TRAIN_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_STACK_TRAIN_SIZE}"
export ARCHITECTURE_VIEW_PART_FUSION_STACK_VAL_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_STACK_VAL_SIZE}"
export ARCHITECTURE_VIEW_PART_FUSION_FINAL_TEST_SIZE="${ARCHITECTURE_VIEW_PART_10CLASS_FINAL_TEST_SIZE}"

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

afterok_args() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

fresh_split_words hetero_arch_args "${ARCHITECTURE_VIEW_PART_10CLASS_HETERO_ARCHITECTURES}"
fresh_split_words av_variant_args "${ARCHITECTURE_VIEW_PART_10CLASS_AV_VARIANTS}"

submitter_lock_dir="${ARCHITECTURE_VIEW_PART_10CLASS_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${ARCHITECTURE_VIEW_PART_10CLASS_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=architecture_view_10class_vs_heterogeneous_hlt4"
    echo "root=${ARCHITECTURE_VIEW_PART_10CLASS_ROOT}"
    echo "manifest=${ARCHITECTURE_VIEW_PART_10CLASS_MANIFEST_PATH}"
    echo "hlt_cache=${ARCHITECTURE_VIEW_PART_10CLASS_HLT_CACHE_DIR}"
    echo "standalone_root=${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_ROOT}"
    echo "architecture_view_root=${ARCHITECTURE_VIEW_PART_10CLASS_AV_ROOT}"
    echo "hlt_degradation_strength=${ARCHITECTURE_VIEW_PART_10CLASS_HLT_DEGRADATION_STRENGTH}"
    echo "label_names=${ARCHITECTURE_VIEW_PART_10CLASS_LABEL_NAMES}"
    echo "heterogeneous_architectures=$(fresh_join_by_space "${hetero_arch_args[@]}")"
    echo "architecture_view_variants=$(fresh_join_by_space "${av_variant_args[@]}")"
    echo "selection_metric=${ARCHITECTURE_VIEW_PART_10CLASS_SELECTION_METRIC}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

mapfile -t split_args < <(
  afterok_args \
    "${UPSTREAM_DEPENDENCY}" \
    --time="${ARCHITECTURE_VIEW_PART_10CLASS_SPLIT_TIME}" \
    --cpus-per-task="${ARCHITECTURE_VIEW_PART_10CLASS_SPLIT_CPUS}" \
    --mem="${ARCHITECTURE_VIEW_PART_10CLASS_SPLIT_MEM}" \
    "${SCRIPT_DIR}/run_build_fresh_splits.sh"
)
split_jid="$(submit_job "archview_10class_splits" "${split_args[@]}")"
echo "submitted archview_10class_splits=${split_jid}"

cache_jid="$(submit_job "archview_10class_hlt_cache" \
  --time="${ARCHITECTURE_VIEW_PART_10CLASS_CACHE_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_PART_10CLASS_CACHE_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_PART_10CLASS_CACHE_MEM}" \
  --dependency="afterok:${split_jid}" \
  "${SCRIPT_DIR}/run_build_fresh_hlt_cache.sh")"
echo "submitted archview_10class_hlt_cache=${cache_jid}"

hetero_job_ids=()
part_train_jid=""
for architecture in "${hetero_arch_args[@]}"; do
  mapfile -t train_args < <(
    afterok_args \
      "${cache_jid}" \
      --time="${ARCHITECTURE_VIEW_PART_10CLASS_HETERO_TIME}" \
      --cpus-per-task="${ARCHITECTURE_VIEW_PART_10CLASS_TRAIN_CPUS}" \
      --mem="${ARCHITECTURE_VIEW_PART_10CLASS_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_heterogeneous_hlt_arch.sh" \
      "${architecture}"
  )
  jid="$(submit_job "hetero_hlt_${architecture}" "${train_args[@]}")"
  hetero_job_ids+=("${jid}")
  if [[ "${architecture}" == "part" ]]; then
    part_train_jid="${jid}"
  fi
  echo "submitted hetero_hlt_${architecture}=${jid}"
done

hetero_dep="$(fresh_join_by_colon "${hetero_job_ids[@]}")"
hetero_fusion_jid="$(submit_job "heterogeneous_hlt4_fusion" \
  --time="${ARCHITECTURE_VIEW_PART_10CLASS_FUSION_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_PART_10CLASS_FUSION_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_PART_10CLASS_FUSION_MEM}" \
  --dependency="afterok:${hetero_dep}" \
  "${SCRIPT_DIR}/run_fuse_heterogeneous_hlt4.sh")"
echo "submitted heterogeneous_hlt4_fusion=${hetero_fusion_jid}"

if [[ -z "${part_train_jid}" ]]; then
  echo "The architecture-view warm start requires the standalone part architecture in HETERO architectures." >&2
  exit 2
fi

av_job_ids=()
for variant in "${av_variant_args[@]}"; do
  label="${variant//[^A-Za-z0-9_]/_}"
  mapfile -t av_args < <(
    afterok_args \
      "${part_train_jid}" \
      --time="${ARCHITECTURE_VIEW_PART_10CLASS_AV_TRAIN_TIME}" \
      --cpus-per-task="${ARCHITECTURE_VIEW_PART_10CLASS_TRAIN_CPUS}" \
      --mem="${ARCHITECTURE_VIEW_PART_10CLASS_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_architecture_view_part.sh" \
      "${variant}"
  )
  jid="$(submit_job "archview10_${label}" "${av_args[@]}")"
  av_job_ids+=("${jid}")
  echo "submitted archview10_${label}=${jid}"
done

av_dep="$(fresh_join_by_colon "${av_job_ids[@]}")"
av_report_jid="$(submit_job "archview10_report" \
  --time="${ARCHITECTURE_VIEW_PART_10CLASS_REPORT_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_PART_10CLASS_REPORT_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_PART_10CLASS_REPORT_MEM}" \
  --dependency="afterok:${av_dep}" \
  "${SCRIPT_DIR}/run_write_architecture_view_part_report.sh")"
echo "submitted archview10_report=${av_report_jid}"

av_fusion_jid="$(submit_job "archview10_fusion" \
  --time="${ARCHITECTURE_VIEW_PART_10CLASS_FUSION_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_PART_10CLASS_FUSION_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_PART_10CLASS_FUSION_MEM}" \
  --dependency="afterok:${av_dep}" \
  "${SCRIPT_DIR}/run_fuse_architecture_view_part.sh")"
echo "submitted archview10_fusion=${av_fusion_jid}"

cat <<SUMMARY
architecture_view_part_10class_submission:
  task: 10class_architecture_view_vs_heterogeneous_hlt4
  root: ${ARCHITECTURE_VIEW_PART_10CLASS_ROOT}
  hlt_degradation_strength: ${ARCHITECTURE_VIEW_PART_10CLASS_HLT_DEGRADATION_STRENGTH}
  label_names: ${ARCHITECTURE_VIEW_PART_10CLASS_LABEL_NAMES}
  selection_metric: ${ARCHITECTURE_VIEW_PART_10CLASS_SELECTION_METRIC}
  job_ids:
    split_manifest: ${split_jid}
    hlt_cache: ${cache_jid}
    standalone_hlt4_train: $(fresh_join_by_space "${hetero_job_ids[@]}")
    standalone_hlt4_fusion: ${hetero_fusion_jid}
    architecture_view_train: $(fresh_join_by_space "${av_job_ids[@]}")
    architecture_view_report: ${av_report_jid}
    architecture_view_fusion: ${av_fusion_jid}
  expected_jobs:
    split_manifest: 1
    hlt_cache: 1
    standalone_train: ${#hetero_job_ids[@]}
    standalone_fusion: 1
    architecture_view_train: ${#av_job_ids[@]}
    architecture_view_report: 1
    architecture_view_fusion: 1
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${ARCHITECTURE_VIEW_PART_10CLASS_MODEL_TRAIN_SIZE}
    model_val: ${ARCHITECTURE_VIEW_PART_10CLASS_MODEL_VAL_SIZE}
    stack_train: ${ARCHITECTURE_VIEW_PART_10CLASS_STACK_TRAIN_SIZE}
    stack_val: ${ARCHITECTURE_VIEW_PART_10CLASS_STACK_VAL_SIZE}
    final_test: ${ARCHITECTURE_VIEW_PART_10CLASS_FINAL_TEST_SIZE}
  outputs:
    manifest: ${ARCHITECTURE_VIEW_PART_10CLASS_MANIFEST_PATH}
    hlt_cache: ${ARCHITECTURE_VIEW_PART_10CLASS_HLT_CACHE_DIR}
    standalone_models: ${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_MODEL_ROOT}
    standalone_fusion_report: ${ARCHITECTURE_VIEW_PART_10CLASS_STANDALONE_FUSION_DIR}/fusion/fusion_report.json
    architecture_view_taggers: ${ARCHITECTURE_VIEW_PART_10CLASS_AV_TAGGER_ROOT}
    architecture_view_report: ${ARCHITECTURE_VIEW_PART_10CLASS_AV_FINAL_REPORT_DIR}/architecture_view_part_final_report.json
    architecture_view_fusion_report: ${ARCHITECTURE_VIEW_PART_10CLASS_AV_FUSION_DIR}/fusion/fusion_report.json
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY

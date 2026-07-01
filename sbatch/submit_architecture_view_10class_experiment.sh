#!/usr/bin/env bash
# Submit the AV10 Architecture-View ensemble experiment.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${ARCHITECTURE_VIEW_10CLASS_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH:=0.6}"
ARCHITECTURE_VIEW_10CLASS_HLT_TAG="${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH//./p}"
: "${ARCHITECTURE_VIEW_10CLASS_ROOT:=${OUTPUT_ROOT}/architecture_view_10class_hlt${ARCHITECTURE_VIEW_10CLASS_HLT_TAG}_${ARCHITECTURE_VIEW_10CLASS_TAG}}"
: "${ARCHITECTURE_VIEW_10CLASS_INPUT_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/inputs}"
: "${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH:=${ARCHITECTURE_VIEW_10CLASS_INPUT_ROOT}/split_manifest.json.gz}"
: "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR:=${ARCHITECTURE_VIEW_10CLASS_INPUT_ROOT}/hlt_cache}"
: "${ARCHITECTURE_VIEW_10CLASS_STANDALONE_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/standalone_hlt4}"
: "${ARCHITECTURE_VIEW_10CLASS_STANDALONE_MODEL_ROOT:=${ARCHITECTURE_VIEW_10CLASS_STANDALONE_ROOT}/models}"
: "${ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_DIR:=${ARCHITECTURE_VIEW_10CLASS_STANDALONE_ROOT}/fusion_run}"
: "${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/taggers}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/prediction_cache}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/fusion}"
: "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/final_report}"

: "${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES:=QCD Hbb Hcc Hgg H4q Hqql Zqq Wqq Tbqq Tbl}"
: "${ARCHITECTURE_VIEW_10CLASS_VARIANTS:=av10_baseline_recheck av10_pn_context_to_part av10_pfn_context_to_part av10_pcnn_context_to_part av10_all_views_to_part av10_random_view_control av10_context_mlp_control}"
: "${ARCHITECTURE_VIEW_10CLASS_HETERO_ARCHITECTURES:=part pn pfn pcnn}"

: "${ARCHITECTURE_VIEW_10CLASS_MODEL_TRAIN_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_10CLASS_STACK_TRAIN_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_10CLASS_EPOCHS:=45}"
: "${ARCHITECTURE_VIEW_10CLASS_SELECTION_METRIC:=accuracy}"

: "${ARCHITECTURE_VIEW_10CLASS_SPLIT_TIME:=04:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_CACHE_TIME:=1-00:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_STANDALONE_TIME:=2-00:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_TRAIN_TIME:=3-00:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_TIME:=1-00:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_TIME:=04:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_REPORT_TIME:=02:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_SPLIT_MEM:=32G}"
: "${ARCHITECTURE_VIEW_10CLASS_CACHE_MEM:=160G}"
: "${ARCHITECTURE_VIEW_10CLASS_TRAIN_MEM:=160G}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_MEM:=160G}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_MEM:=32G}"
: "${ARCHITECTURE_VIEW_10CLASS_REPORT_MEM:=8G}"
: "${ARCHITECTURE_VIEW_10CLASS_SPLIT_CPUS:=4}"
: "${ARCHITECTURE_VIEW_10CLASS_CACHE_CPUS:=8}"
: "${ARCHITECTURE_VIEW_10CLASS_TRAIN_CPUS:=8}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_CPUS:=8}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_CPUS:=4}"
: "${ARCHITECTURE_VIEW_10CLASS_REPORT_CPUS:=2}"

export MANIFEST_PATH="${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH}"
export HLT_CACHE_DIR="${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}"
export HLT_DEGRADATION_STRENGTH="${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH}"
export MODEL_TRAIN_SIZE="${ARCHITECTURE_VIEW_10CLASS_MODEL_TRAIN_SIZE}"
export MODEL_VAL_SIZE="${ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE}"
export STACK_TRAIN_SIZE="${ARCHITECTURE_VIEW_10CLASS_STACK_TRAIN_SIZE}"
export STACK_VAL_SIZE="${ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE}"
export FINAL_TEST_SIZE="${ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE}"

export HETERO_HLT4_ROOT="${ARCHITECTURE_VIEW_10CLASS_STANDALONE_ROOT}"
export HETERO_HLT4_MODEL_ROOT="${ARCHITECTURE_VIEW_10CLASS_STANDALONE_MODEL_ROOT}"
export HETERO_HLT4_FUSION_DIR="${ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_DIR}"
export HETERO_HLT4_ARCHITECTURES="${ARCHITECTURE_VIEW_10CLASS_HETERO_ARCHITECTURES}"
export HETERO_HLT4_TRAIN_SIZE="${ARCHITECTURE_VIEW_10CLASS_MODEL_TRAIN_SIZE}"
export HETERO_HLT4_VAL_SIZE="${ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE}"
export HETERO_HLT4_STACK_TRAIN_SIZE="${ARCHITECTURE_VIEW_10CLASS_STACK_TRAIN_SIZE}"
export HETERO_HLT4_STACK_VAL_SIZE="${ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE}"
export HETERO_HLT4_FINAL_TEST_SIZE="${ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE}"

export ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT
export ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH
export ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR
export ARCHITECTURE_VIEW_10CLASS_BASELINE_CHECKPOINT="${ARCHITECTURE_VIEW_10CLASS_STANDALONE_MODEL_ROOT}/part/best_model_val.pt"
export ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES
export ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER_NAMES="${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES}"
export ARCHITECTURE_VIEW_10CLASS_MODEL_TRAIN_SIZE
export ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE
export ARCHITECTURE_VIEW_10CLASS_STACK_TRAIN_SIZE
export ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE
export ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE
export ARCHITECTURE_VIEW_10CLASS_EPOCHS
export ARCHITECTURE_VIEW_10CLASS_SELECTION_METRIC
export ARCHITECTURE_VIEW_10CLASS_EXPECTED_HLT_DEGRADATION_STRENGTH="${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH}"
export ARCHITECTURE_VIEW_10CLASS_VARIANTS
export ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT
export ARCHITECTURE_VIEW_10CLASS_PREDICTION_DIR="${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}/predictions"
export ARCHITECTURE_VIEW_10CLASS_FUSION_DIR
export ARCHITECTURE_VIEW_10CLASS_REPORT_DIR
export ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_REPORT="${ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_DIR}/fusion/fusion_report.json"
export ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST=1

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

fresh_split_words hetero_arch_args "${ARCHITECTURE_VIEW_10CLASS_HETERO_ARCHITECTURES}"
fresh_split_words av_variant_args "${ARCHITECTURE_VIEW_10CLASS_VARIANTS}"

submitter_lock_dir="${ARCHITECTURE_VIEW_10CLASS_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${ARCHITECTURE_VIEW_10CLASS_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=architecture_view_10class_ensemble"
    echo "root=${ARCHITECTURE_VIEW_10CLASS_ROOT}"
    echo "manifest=${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH}"
    echo "hlt_cache=${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}"
    echo "standalone_model_root=${ARCHITECTURE_VIEW_10CLASS_STANDALONE_MODEL_ROOT}"
    echo "tagger_root=${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT}"
    echo "prediction_root=${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}"
    echo "fusion_dir=${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}"
    echo "report_dir=${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}"
    echo "hlt_degradation_strength=${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH}"
    echo "label_names=${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES}"
    echo "heterogeneous_architectures=$(fresh_join_by_space "${hetero_arch_args[@]}")"
    echo "architecture_view_variants=$(fresh_join_by_space "${av_variant_args[@]}")"
    echo "selection_metric=${ARCHITECTURE_VIEW_10CLASS_SELECTION_METRIC}"
    echo "epochs=${ARCHITECTURE_VIEW_10CLASS_EPOCHS}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

mapfile -t split_args < <(
  afterok_args \
    "${UPSTREAM_DEPENDENCY}" \
    --time="${ARCHITECTURE_VIEW_10CLASS_SPLIT_TIME}" \
    --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_SPLIT_CPUS}" \
    --mem="${ARCHITECTURE_VIEW_10CLASS_SPLIT_MEM}" \
    "${SCRIPT_DIR}/run_build_fresh_splits.sh"
)
split_jid="$(submit_job "av10_splits" "${split_args[@]}")"
echo "submitted av10_splits=${split_jid}"

cache_jid="$(submit_job "av10_hlt_cache" \
  --time="${ARCHITECTURE_VIEW_10CLASS_CACHE_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_CACHE_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_10CLASS_CACHE_MEM}" \
  --dependency="afterok:${split_jid}" \
  "${SCRIPT_DIR}/run_build_fresh_hlt_cache.sh")"
echo "submitted av10_hlt_cache=${cache_jid}"

hetero_job_ids=()
part_train_jid=""
for architecture in "${hetero_arch_args[@]}"; do
  mapfile -t train_args < <(
    afterok_args \
      "${cache_jid}" \
      --time="${ARCHITECTURE_VIEW_10CLASS_STANDALONE_TIME}" \
      --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_TRAIN_CPUS}" \
      --mem="${ARCHITECTURE_VIEW_10CLASS_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_heterogeneous_hlt_arch.sh" \
      "${architecture}"
  )
  jid="$(submit_job "av10_standalone_${architecture}" "${train_args[@]}")"
  hetero_job_ids+=("${jid}")
  if [[ "${architecture}" == "part" ]]; then
    part_train_jid="${jid}"
  fi
  echo "submitted av10_standalone_${architecture}=${jid}"
done

if [[ -z "${part_train_jid}" ]]; then
  echo "AV10 Architecture-View needs standalone part in ARCHITECTURE_VIEW_10CLASS_HETERO_ARCHITECTURES." >&2
  exit 2
fi

hetero_dep="$(fresh_join_by_colon "${hetero_job_ids[@]}")"
standalone_fusion_jid="$(submit_job "av10_standalone_fusion" \
  --time="${ARCHITECTURE_VIEW_10CLASS_FUSION_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_FUSION_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_10CLASS_FUSION_MEM}" \
  --dependency="afterok:${hetero_dep}" \
  "${SCRIPT_DIR}/run_fuse_heterogeneous_hlt4.sh")"
echo "submitted av10_standalone_fusion=${standalone_fusion_jid}"

av_job_ids=()
for variant in "${av_variant_args[@]}"; do
  label="${variant//[^A-Za-z0-9_]/_}"
  mapfile -t av_args < <(
    afterok_args \
      "${part_train_jid}" \
      --time="${ARCHITECTURE_VIEW_10CLASS_TRAIN_TIME}" \
      --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_TRAIN_CPUS}" \
      --mem="${ARCHITECTURE_VIEW_10CLASS_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_architecture_view_10class_part.sh" \
      "${variant}"
  )
  jid="$(submit_job "av10_${label}" "${av_args[@]}")"
  av_job_ids+=("${jid}")
  echo "submitted av10_${label}=${jid}"
done

av_dep="$(fresh_join_by_colon "${av_job_ids[@]}")"
prediction_jid="$(submit_job "av10_prediction_cache" \
  --time="${ARCHITECTURE_VIEW_10CLASS_PREDICTION_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_PREDICTION_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_10CLASS_PREDICTION_MEM}" \
  --dependency="afterok:${av_dep}" \
  "${SCRIPT_DIR}/run_cache_architecture_view_10class_predictions.sh")"
echo "submitted av10_prediction_cache=${prediction_jid}"

fusion_jid="$(submit_job "av10_fusion" \
  --time="${ARCHITECTURE_VIEW_10CLASS_FUSION_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_FUSION_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_10CLASS_FUSION_MEM}" \
  --dependency="afterok:${prediction_jid}" \
  "${SCRIPT_DIR}/run_architecture_view_10class_fusion.sh")"
echo "submitted av10_fusion=${fusion_jid}"

report_dep="$(fresh_join_by_colon "${fusion_jid}" "${standalone_fusion_jid}")"
report_jid="$(submit_job "av10_report" \
  --time="${ARCHITECTURE_VIEW_10CLASS_REPORT_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_REPORT_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_10CLASS_REPORT_MEM}" \
  --dependency="afterok:${report_dep}" \
  "${SCRIPT_DIR}/run_write_architecture_view_10class_report.sh")"
echo "submitted av10_report=${report_jid}"

cat <<SUMMARY
architecture_view_10class_submission:
  task: architecture_view_10class_ensemble
  root: ${ARCHITECTURE_VIEW_10CLASS_ROOT}
  hlt_degradation_strength: ${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH}
  label_names: ${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES}
  selection_metric: ${ARCHITECTURE_VIEW_10CLASS_SELECTION_METRIC}
  job_ids:
    split_manifest: ${split_jid}
    hlt_cache: ${cache_jid}
    standalone_train: $(fresh_join_by_space "${hetero_job_ids[@]}")
    standalone_fusion: ${standalone_fusion_jid}
    architecture_view_train: $(fresh_join_by_space "${av_job_ids[@]}")
    prediction_cache: ${prediction_jid}
    architecture_view_fusion: ${fusion_jid}
    final_report: ${report_jid}
  expected_jobs:
    split_manifest: 1
    hlt_cache: 1
    standalone_train: ${#hetero_job_ids[@]}
    standalone_fusion: 1
    architecture_view_train: ${#av_job_ids[@]}
    prediction_cache: 1
    architecture_view_fusion: 1
    final_report: 1
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${ARCHITECTURE_VIEW_10CLASS_MODEL_TRAIN_SIZE}
    model_val: ${ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE}
    stack_train: ${ARCHITECTURE_VIEW_10CLASS_STACK_TRAIN_SIZE}
    stack_val: ${ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE}
    final_test: ${ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE}
  outputs:
    manifest: ${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH}
    hlt_cache: ${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}
    standalone_models: ${ARCHITECTURE_VIEW_10CLASS_STANDALONE_MODEL_ROOT}
    standalone_fusion_report: ${ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_DIR}/fusion/fusion_report.json
    architecture_view_taggers: ${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT}
    prediction_cache: ${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}
    architecture_view_fusion_report: ${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}/fusion_report.json
    final_report: ${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}/architecture_view_10class_report.json
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY

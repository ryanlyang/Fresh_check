#!/usr/bin/env bash
# Submit the AV10 ablation campaign with rebuilt splits, HLT cache, anchors, fusion, and offline transfer.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH:=0.6}"
ARCHITECTURE_VIEW_10CLASS_HLT_TAG="${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH//./p}"
: "${ARCHITECTURE_VIEW_10CLASS_ROOT:=${OUTPUT_ROOT}/architecture_view_10class_ablation_hlt${ARCHITECTURE_VIEW_10CLASS_HLT_TAG}_${ARCHITECTURE_VIEW_10CLASS_ABLATION_TAG}}"
: "${ARCHITECTURE_VIEW_10CLASS_INPUT_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/inputs}"
: "${ARCHITECTURE_VIEW_10CLASS_REUSE_EXISTING_INPUTS:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_EXISTING_INPUT_ROOT:=}"
: "${ARCHITECTURE_VIEW_10CLASS_EXISTING_MANIFEST_PATH:=}"
: "${ARCHITECTURE_VIEW_10CLASS_EXISTING_HLT_CACHE_DIR:=}"
if fresh_bool_enabled "${ARCHITECTURE_VIEW_10CLASS_REUSE_EXISTING_INPUTS}"; then
  if [[ -n "${ARCHITECTURE_VIEW_10CLASS_EXISTING_INPUT_ROOT}" ]]; then
    : "${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH:=${ARCHITECTURE_VIEW_10CLASS_EXISTING_INPUT_ROOT}/split_manifest.json.gz}"
    : "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR:=${ARCHITECTURE_VIEW_10CLASS_EXISTING_INPUT_ROOT}/hlt_cache}"
  else
    : "${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH:=${ARCHITECTURE_VIEW_10CLASS_EXISTING_MANIFEST_PATH}}"
    : "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR:=${ARCHITECTURE_VIEW_10CLASS_EXISTING_HLT_CACHE_DIR}}"
  fi
else
  : "${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH:=${ARCHITECTURE_VIEW_10CLASS_INPUT_ROOT}/split_manifest.json.gz}"
  : "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR:=${ARCHITECTURE_VIEW_10CLASS_INPUT_ROOT}/hlt_cache}"
fi
: "${ARCHITECTURE_VIEW_10CLASS_STANDALONE_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/standalone_hlt4}"
: "${ARCHITECTURE_VIEW_10CLASS_STANDALONE_MODEL_ROOT:=${ARCHITECTURE_VIEW_10CLASS_STANDALONE_ROOT}/models}"
: "${ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_DIR:=${ARCHITECTURE_VIEW_10CLASS_STANDALONE_ROOT}/fusion_run}"
: "${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/taggers}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/prediction_cache}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/fusion}"
: "${ARCHITECTURE_VIEW_10CLASS_LEGACY_REPORT_DIR:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/legacy_final_report}"
: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/final_report}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/offline_transfer}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH:=${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH}}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR:=${ARCHITECTURE_VIEW_10CLASS_OFFLINE_ROOT}/inputs/offline_cache}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TAGGER_ROOT:=${ARCHITECTURE_VIEW_10CLASS_OFFLINE_ROOT}/taggers}"

: "${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES:=QCD Hbb Hcc Hgg H4q Hqql Zqq Wqq Tbqq Tbl}"
: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS:=av10_hlt_baseline_recheck av10_larger_part av10_extra_part_block av10_part_only_adapter av10_feature_mlp_adapter av10_lc_mlp_delta_features av10_feature_mlp_adapter_wide av10_frozen_part_feature_adapter av10_shuffled_feature_adapter av10_pcnn_context_repeat av10_pfn_context_repeat av10_finetune_only_control av10_part_only_mlp_adapter av10_feature_deepsets_context_adapter av10_feature_self_attention_context_adapter av10_part_embedding_deepsets_adapter av10_part_embedding_self_attention_adapter av10_within_jet_shuffled_context_adapter av10_noise_context_adapter}"
: "${ARCHITECTURE_VIEW_10CLASS_LEGACY_VARIANTS:=av10_baseline_recheck av10_context_mlp_control av10_pfn_context_to_part av10_pcnn_context_to_part}"
: "${ARCHITECTURE_VIEW_10CLASS_VARIANTS:=${ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS} ${ARCHITECTURE_VIEW_10CLASS_LEGACY_VARIANTS}}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_VARIANTS:=av10_offline_part_baseline av10_offline_feature_mlp_adapter av10_offline_pcnn_context}"
: "${ARCHITECTURE_VIEW_10CLASS_HETERO_ARCHITECTURES:=part pn pfn pcnn}"

: "${ARCHITECTURE_VIEW_10CLASS_MODEL_TRAIN_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_10CLASS_STACK_TRAIN_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_10CLASS_EPOCHS:=45}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_EPOCHS:=${ARCHITECTURE_VIEW_10CLASS_EPOCHS}}"
: "${ARCHITECTURE_VIEW_10CLASS_SELECTION_METRIC:=accuracy}"

: "${ARCHITECTURE_VIEW_10CLASS_SPLIT_TIME:=04:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_CACHE_TIME:=1-00:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_STANDALONE_TIME:=2-00:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_TRAIN_TIME:=3-00:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_TIME:=1-00:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_TIME:=06:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_TIME:=1-00:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRAIN_TIME:=${ARCHITECTURE_VIEW_10CLASS_TRAIN_TIME}}"
: "${ARCHITECTURE_VIEW_10CLASS_REPORT_TIME:=03:00:00}"
: "${ARCHITECTURE_VIEW_10CLASS_SPLIT_MEM:=32G}"
: "${ARCHITECTURE_VIEW_10CLASS_CACHE_MEM:=180G}"
: "${ARCHITECTURE_VIEW_10CLASS_TRAIN_MEM:=180G}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_MEM:=180G}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_MEM:=48G}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_MEM:=180G}"
: "${ARCHITECTURE_VIEW_10CLASS_REPORT_MEM:=16G}"
: "${ARCHITECTURE_VIEW_10CLASS_SPLIT_CPUS:=4}"
: "${ARCHITECTURE_VIEW_10CLASS_CACHE_CPUS:=8}"
: "${ARCHITECTURE_VIEW_10CLASS_TRAIN_CPUS:=8}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_CPUS:=8}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_CPUS:=4}"
: "${ARCHITECTURE_VIEW_10CLASS_REPORT_CPUS:=2}"
: "${ARCHITECTURE_VIEW_10CLASS_SUBMIT_OFFLINE_TRANSFER:=1}"
: "${ARCHITECTURE_VIEW_10CLASS_SUBMIT_LEGACY_REPORT:=1}"
: "${ARCHITECTURE_VIEW_10CLASS_REQUIRE_ABLATION_FUSION:=1}"
: "${ARCHITECTURE_VIEW_10CLASS_REQUIRE_OFFLINE_TRANSFER:=1}"

fresh_split_words ablation_variant_args "${ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS}"
fresh_split_words legacy_variant_args "${ARCHITECTURE_VIEW_10CLASS_LEGACY_VARIANTS}"
fresh_split_words all_variant_args "${ARCHITECTURE_VIEW_10CLASS_VARIANTS}"
fresh_split_words offline_variant_args "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_VARIANTS}"
fresh_split_words hetero_arch_args "${ARCHITECTURE_VIEW_10CLASS_HETERO_ARCHITECTURES}"

ablation_variant_csv="$(fresh_join_by_comma "${ablation_variant_args[@]}")"
all_variant_csv="$(fresh_join_by_comma "${all_variant_args[@]}")"
legacy_variant_csv="$(fresh_join_by_comma "${legacy_variant_args[@]}")"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_GROUPS:=av10_architecture_view_core:av10_hlt_baseline_recheck,av10_feature_mlp_adapter,av10_lc_mlp_delta_features,av10_pcnn_context_repeat,av10_pfn_context_repeat av10_capacity_controls:av10_hlt_baseline_recheck,av10_larger_part,av10_extra_part_block,av10_part_only_adapter,av10_feature_mlp_adapter av10_contextual_adapters:av10_hlt_baseline_recheck,av10_feature_mlp_adapter,av10_feature_deepsets_context_adapter,av10_feature_self_attention_context_adapter,av10_part_embedding_deepsets_adapter,av10_part_embedding_self_attention_adapter,av10_part_only_mlp_adapter,av10_finetune_only_control av10_context_controls:av10_hlt_baseline_recheck,av10_within_jet_shuffled_context_adapter,av10_noise_context_adapter,av10_shuffled_feature_adapter av10_all_ablation:${ablation_variant_csv} av10_legacy_contexts:${legacy_variant_csv} av10_all_available:${all_variant_csv}}"

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

export ARCHITECTURE_VIEW_10CLASS_ROOT
export ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT
export ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH
export ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR
: "${ARCHITECTURE_VIEW_10CLASS_BASELINE_CHECKPOINT:=${ARCHITECTURE_VIEW_10CLASS_STANDALONE_MODEL_ROOT}/part/best_model_val.pt}"
export ARCHITECTURE_VIEW_10CLASS_BASELINE_CHECKPOINT
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
export ARCHITECTURE_VIEW_10CLASS_FUSION_GROUPS
export ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_REPORT="${ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_DIR}/fusion/fusion_report.json"
export ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST=1
export ARCHITECTURE_VIEW_10CLASS_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH=1

export ARCHITECTURE_VIEW_10CLASS_OFFLINE_ROOT
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_TAGGER_ROOT
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_BASELINE_CHECKPOINT="${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TAGGER_ROOT}/av10_offline_part_baseline/best_model_val.pt"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_LABEL_NAMES="${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_LABEL_FILTER_NAMES="${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_MODEL_TRAIN_SIZE="${ARCHITECTURE_VIEW_10CLASS_MODEL_TRAIN_SIZE}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_MODEL_VAL_SIZE="${ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_STACK_VAL_SIZE="${ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_FINAL_TEST_SIZE="${ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_EPOCHS
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_SELECTION_METRIC="${ARCHITECTURE_VIEW_10CLASS_SELECTION_METRIC}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH=1
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_CONFIRM_FINAL_TEST=1

export ARCHITECTURE_VIEW_10CLASS_ABLATION_ROOT="${ARCHITECTURE_VIEW_10CLASS_ROOT}"
export ARCHITECTURE_VIEW_10CLASS_ABLATION_TAGGER_ROOT="${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT}"
export ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR
export ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS
export ARCHITECTURE_VIEW_10CLASS_ABLATION_BASELINE_VARIANT="av10_hlt_baseline_recheck"
export ARCHITECTURE_VIEW_10CLASS_ABLATION_FUSION_REPORT="${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}/fusion_report.json"
export ARCHITECTURE_VIEW_10CLASS_ABLATION_REQUIRE_FUSION="${ARCHITECTURE_VIEW_10CLASS_REQUIRE_ABLATION_FUSION}"
export ARCHITECTURE_VIEW_10CLASS_ABLATION_REQUIRE_OFFLINE_TRANSFER="${ARCHITECTURE_VIEW_10CLASS_REQUIRE_OFFLINE_TRANSFER}"

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

av10_sha256_or_pending() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "${path}" | awk '{print $1}'
    else
      shasum -a 256 "${path}" | awk '{print $1}'
    fi
  else
    printf 'pending\n'
  fi
}

av10_require_reused_inputs() {
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH}"
  local split
  for split in model_train model_val stack_train stack_val final_test; do
    fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
    fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
  local python_bin="${PYTHON_BIN:-python}"
  "${python_bin}" - \
    "${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH}" \
    "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}" \
    "${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH}" <<'PY'
import json
import sys
from pathlib import Path

from jetclass_fresh.jetclass_data import load_split_manifest, manifest_hash

manifest_path = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
expected_strength = float(sys.argv[3])
expected_manifest_hash = manifest_hash(load_split_manifest(manifest_path))
problems = []
for split in ("model_train", "model_val", "stack_train", "stack_val", "final_test"):
    metadata_path = cache_dir / f"{split}_fixed_hlt_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    actual_manifest_hash = metadata.get("source_manifest_hash") or metadata.get("manifest_hash")
    if actual_manifest_hash != expected_manifest_hash:
        problems.append(
            f"{split}: source_manifest_hash {actual_manifest_hash!r} != active manifest {expected_manifest_hash!r}"
        )
    if metadata.get("view") not in (None, "fixed_hlt"):
        problems.append(f"{split}: expected fixed_hlt view metadata, got {metadata.get('view')!r}")
    actual = metadata.get("hlt_degradation_strength")
    try:
        matches_strength = abs(float(actual) - expected_strength) < 1.0e-12
    except (TypeError, ValueError):
        matches_strength = False
    if not matches_strength:
        problems.append(f"{split}: hlt_degradation_strength {actual!r} != {expected_strength!r}")
if problems:
    raise SystemExit("reused AV10 HLT cache failed identity checks:\n  " + "\n  ".join(problems))
print(f"reused AV10 HLT cache identity ok: manifest_hash={expected_manifest_hash}")
PY
}

submitter_lock_dir="${ARCHITECTURE_VIEW_10CLASS_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${ARCHITECTURE_VIEW_10CLASS_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
reuse_existing_inputs_flag=0
if fresh_bool_enabled "${ARCHITECTURE_VIEW_10CLASS_REUSE_EXISTING_INPUTS}"; then
  reuse_existing_inputs_flag=1
fi
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "task=architecture_view_10class_ablation"
    echo "root=${ARCHITECTURE_VIEW_10CLASS_ROOT}"
    echo "reuse_existing_inputs=${ARCHITECTURE_VIEW_10CLASS_REUSE_EXISTING_INPUTS}"
    echo "existing_input_root=${ARCHITECTURE_VIEW_10CLASS_EXISTING_INPUT_ROOT}"
    echo "manifest=${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH}"
    echo "hlt_cache=${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}"
    echo "hlt_cache_profile=fixed_hlt_v1"
    echo "standalone_model_root=${ARCHITECTURE_VIEW_10CLASS_STANDALONE_MODEL_ROOT}"
    echo "standalone_fusion_dir=${ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_DIR}"
    echo "tagger_root=${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT}"
    echo "prediction_root=${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}"
    echo "fusion_dir=${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}"
    echo "offline_root=${ARCHITECTURE_VIEW_10CLASS_OFFLINE_ROOT}"
    echo "ablation_report_dir=${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}"
    echo "legacy_report_dir=${ARCHITECTURE_VIEW_10CLASS_LEGACY_REPORT_DIR}"
    echo "baseline_checkpoint=${ARCHITECTURE_VIEW_10CLASS_BASELINE_CHECKPOINT}"
    echo "baseline_checkpoint_hash=$(av10_sha256_or_pending "${ARCHITECTURE_VIEW_10CLASS_BASELINE_CHECKPOINT}")"
    echo "hlt_degradation_strength=${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH}"
    echo "label_names=${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES}"
    echo "heterogeneous_architectures=$(fresh_join_by_space "${hetero_arch_args[@]}")"
    echo "ablation_variants=$(fresh_join_by_space "${ablation_variant_args[@]}")"
    echo "legacy_variants=$(fresh_join_by_space "${legacy_variant_args[@]}")"
    echo "offline_variants=$(fresh_join_by_space "${offline_variant_args[@]}")"
    echo "fusion_groups=${ARCHITECTURE_VIEW_10CLASS_FUSION_GROUPS}"
    echo "selection_metric=${ARCHITECTURE_VIEW_10CLASS_SELECTION_METRIC}"
    echo "epochs=${ARCHITECTURE_VIEW_10CLASS_EPOCHS}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

split_jid=""
cache_jid=""
if [[ "${reuse_existing_inputs_flag}" -eq 1 ]]; then
  av10_require_reused_inputs
  split_jid="${UPSTREAM_DEPENDENCY}"
  cache_jid="${UPSTREAM_DEPENDENCY}"
  echo "reusing av10 input manifest and HLT cache:"
  echo "  manifest=${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH}"
  echo "  hlt_cache=${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}"
  echo "  upstream_dependency=${UPSTREAM_DEPENDENCY:-none}"
else
  mapfile -t split_args < <(
    afterok_args \
      "${UPSTREAM_DEPENDENCY}" \
      --time="${ARCHITECTURE_VIEW_10CLASS_SPLIT_TIME}" \
      --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_SPLIT_CPUS}" \
      --mem="${ARCHITECTURE_VIEW_10CLASS_SPLIT_MEM}" \
      "${SCRIPT_DIR}/run_build_fresh_splits.sh"
  )
  split_jid="$(submit_job "av10_ablate_splits" "${split_args[@]}")"
  echo "submitted av10_ablate_splits=${split_jid}"

  cache_jid="$(submit_job "av10_ablate_hlt_cache" \
    --time="${ARCHITECTURE_VIEW_10CLASS_CACHE_TIME}" \
    --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_CACHE_CPUS}" \
    --mem="${ARCHITECTURE_VIEW_10CLASS_CACHE_MEM}" \
    --dependency="afterok:${split_jid}" \
    "${SCRIPT_DIR}/run_build_fresh_hlt_cache.sh")"
  echo "submitted av10_ablate_hlt_cache=${cache_jid}"
fi

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
  jid="$(submit_job "av10_ablate_standalone_${architecture}" "${train_args[@]}")"
  hetero_job_ids+=("${jid}")
  if [[ "${architecture}" == "part" ]]; then
    part_train_jid="${jid}"
  fi
  echo "submitted av10_ablate_standalone_${architecture}=${jid}"
done

if [[ -z "${part_train_jid}" ]]; then
  echo "AV10 ablation needs standalone part in ARCHITECTURE_VIEW_10CLASS_HETERO_ARCHITECTURES." >&2
  exit 2
fi

hetero_dep="$(fresh_join_by_colon "${hetero_job_ids[@]}")"
standalone_fusion_jid="$(submit_job "av10_ablate_standalone_fusion" \
  --time="${ARCHITECTURE_VIEW_10CLASS_FUSION_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_FUSION_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_10CLASS_FUSION_MEM}" \
  --dependency="afterok:${hetero_dep}" \
  "${SCRIPT_DIR}/run_fuse_heterogeneous_hlt4.sh")"
echo "submitted av10_ablate_standalone_fusion=${standalone_fusion_jid}"

av_job_ids=()
for variant in "${all_variant_args[@]}"; do
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
  jid="$(submit_job "av10_ablate_${label}" "${av_args[@]}")"
  av_job_ids+=("${jid}")
  echo "submitted av10_ablate_${label}=${jid}"
done

av_dep="$(fresh_join_by_colon "${av_job_ids[@]}")"
prediction_jid="$(submit_job "av10_ablate_prediction_cache" \
  --time="${ARCHITECTURE_VIEW_10CLASS_PREDICTION_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_PREDICTION_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_10CLASS_PREDICTION_MEM}" \
  --dependency="afterok:${av_dep}" \
  "${SCRIPT_DIR}/run_cache_architecture_view_10class_predictions.sh")"
echo "submitted av10_ablate_prediction_cache=${prediction_jid}"

fusion_jid="$(submit_job "av10_ablate_fusion" \
  --time="${ARCHITECTURE_VIEW_10CLASS_FUSION_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_FUSION_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_10CLASS_FUSION_MEM}" \
  --dependency="afterok:${prediction_jid}" \
  "${SCRIPT_DIR}/run_architecture_view_10class_fusion.sh")"
echo "submitted av10_ablate_fusion=${fusion_jid}"

offline_cache_jid=""
offline_train_job_ids=()
if fresh_bool_enabled "${ARCHITECTURE_VIEW_10CLASS_SUBMIT_OFFLINE_TRANSFER}"; then
  mapfile -t offline_cache_args < <(
    afterok_args \
      "${split_jid}" \
      --time="${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_TIME}" \
      --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_CACHE_CPUS}" \
      --mem="${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_MEM}" \
      "${SCRIPT_DIR}/run_cache_architecture_view_offline_inputs.sh"
  )
  offline_cache_jid="$(submit_job "av10_ablate_offline_cache" "${offline_cache_args[@]}")"
  echo "submitted av10_ablate_offline_cache=${offline_cache_jid}"

  offline_baseline_jid="$(submit_job "av10_ablate_av10_offline_part_baseline" \
    --time="${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRAIN_TIME}" \
    --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_TRAIN_CPUS}" \
    --mem="${ARCHITECTURE_VIEW_10CLASS_TRAIN_MEM}" \
    --dependency="afterok:${offline_cache_jid}" \
    "${SCRIPT_DIR}/run_train_architecture_view_10class_offline_part.sh" \
    "av10_offline_part_baseline")"
  offline_train_job_ids+=("${offline_baseline_jid}")
  echo "submitted av10_ablate_av10_offline_part_baseline=${offline_baseline_jid}"

  for variant in "${offline_variant_args[@]}"; do
    if [[ "${variant}" == "av10_offline_part_baseline" ]]; then
      continue
    fi
    label="${variant//[^A-Za-z0-9_]/_}"
    jid="$(submit_job "av10_ablate_${label}" \
      --time="${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRAIN_TIME}" \
      --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_TRAIN_CPUS}" \
      --mem="${ARCHITECTURE_VIEW_10CLASS_TRAIN_MEM}" \
      --dependency="afterok:${offline_baseline_jid}" \
      "${SCRIPT_DIR}/run_train_architecture_view_10class_offline_part.sh" \
      "${variant}")"
    offline_train_job_ids+=("${jid}")
    echo "submitted av10_ablate_${label}=${jid}"
  done
else
  export ARCHITECTURE_VIEW_10CLASS_ABLATION_REQUIRE_OFFLINE_TRANSFER=0
fi

legacy_report_jid=""
if fresh_bool_enabled "${ARCHITECTURE_VIEW_10CLASS_SUBMIT_LEGACY_REPORT}"; then
  report_variants_before="${ARCHITECTURE_VIEW_10CLASS_VARIANTS}"
  report_dir_before="${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR:-}"
  export ARCHITECTURE_VIEW_10CLASS_VARIANTS="${ARCHITECTURE_VIEW_10CLASS_LEGACY_VARIANTS}"
  export ARCHITECTURE_VIEW_10CLASS_REPORT_DIR="${ARCHITECTURE_VIEW_10CLASS_LEGACY_REPORT_DIR}"
  legacy_report_dep="$(fresh_join_by_colon "${fusion_jid}" "${standalone_fusion_jid}")"
  legacy_report_jid="$(submit_job "av10_legacy_report" \
    --time="${ARCHITECTURE_VIEW_10CLASS_REPORT_TIME}" \
    --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_REPORT_CPUS}" \
    --mem="${ARCHITECTURE_VIEW_10CLASS_REPORT_MEM}" \
    --dependency="afterok:${legacy_report_dep}" \
    "${SCRIPT_DIR}/run_write_architecture_view_10class_report.sh")"
  echo "submitted av10_legacy_report=${legacy_report_jid}"
  export ARCHITECTURE_VIEW_10CLASS_VARIANTS="${report_variants_before}"
  export ARCHITECTURE_VIEW_10CLASS_REPORT_DIR="${report_dir_before}"
fi

ablation_report_deps=("${fusion_jid}")
if fresh_bool_enabled "${ARCHITECTURE_VIEW_10CLASS_SUBMIT_OFFLINE_TRANSFER}"; then
  ablation_report_deps+=("${offline_train_job_ids[@]}")
fi
ablation_report_dep="$(fresh_join_by_colon "${ablation_report_deps[@]}")"
ablation_report_jid="$(submit_job "av10_ablation_report" \
  --time="${ARCHITECTURE_VIEW_10CLASS_REPORT_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_10CLASS_REPORT_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_10CLASS_REPORT_MEM}" \
  --dependency="afterok:${ablation_report_dep}" \
  "${SCRIPT_DIR}/run_write_architecture_view_10class_ablation_report.sh")"
echo "submitted av10_ablation_report=${ablation_report_jid}"

cat <<SUMMARY
architecture_view_10class_ablation_submission:
  task: architecture_view_10class_ablation
  root: ${ARCHITECTURE_VIEW_10CLASS_ROOT}
  hlt_degradation_strength: ${ARCHITECTURE_VIEW_10CLASS_HLT_DEGRADATION_STRENGTH}
  label_names: ${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES}
  selection_metric: ${ARCHITECTURE_VIEW_10CLASS_SELECTION_METRIC}
  job_ids:
    split_manifest: ${split_jid:-reused_existing}
    hlt_cache: ${cache_jid:-reused_existing}
    standalone_train: $(fresh_join_by_space "${hetero_job_ids[@]}")
    standalone_fusion: ${standalone_fusion_jid}
    hlt_ablation_and_legacy_train: $(fresh_join_by_space "${av_job_ids[@]}")
    prediction_cache: ${prediction_jid}
    av10_fusion: ${fusion_jid}
    offline_cache: ${offline_cache_jid:-skipped}
    offline_train: $(fresh_join_by_space "${offline_train_job_ids[@]:-}")
    legacy_report: ${legacy_report_jid:-skipped}
    ablation_report: ${ablation_report_jid}
  expected_jobs:
    split_manifest: $((1 - reuse_existing_inputs_flag))
    hlt_cache: $((1 - reuse_existing_inputs_flag))
    standalone_train: ${#hetero_job_ids[@]}
    standalone_fusion: 1
    hlt_ablation_and_legacy_train: ${#av_job_ids[@]}
    prediction_cache: 1
    av10_fusion: 1
    offline_cache: $(fresh_bool_enabled "${ARCHITECTURE_VIEW_10CLASS_SUBMIT_OFFLINE_TRANSFER}" && echo 1 || echo 0)
    offline_train: ${#offline_train_job_ids[@]}
    legacy_report: $(fresh_bool_enabled "${ARCHITECTURE_VIEW_10CLASS_SUBMIT_LEGACY_REPORT}" && echo 1 || echo 0)
    ablation_report: 1
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
    hlt_part_baseline_checkpoint: ${ARCHITECTURE_VIEW_10CLASS_BASELINE_CHECKPOINT}
    hlt4_fusion_report: ${ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_DIR}/fusion/fusion_report.json
    hlt_ablation_and_legacy_taggers: ${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT}
    prediction_cache: ${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}
    av10_fusion_report: ${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}/fusion_report.json
    offline_taggers: ${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TAGGER_ROOT}
    legacy_report: ${ARCHITECTURE_VIEW_10CLASS_LEGACY_REPORT_DIR}/architecture_view_10class_report.json
    ablation_report: ${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}/architecture_view_10class_ablation_report.json
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY

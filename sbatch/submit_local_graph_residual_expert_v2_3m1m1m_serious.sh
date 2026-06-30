#!/usr/bin/env bash
# Serious QCD-vs-Hgg HLT0.6 residual-expert V2 run on the 3M/1M/1M cache.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter

: "${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_INPUT_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_3m1m1m_full_20260628_194154}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_LOCAL_GRAPH_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints/local_graph_part_qcd_hgg_hlt0p6_3m1m1m_20260629_015555}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_BASELINE_CHECKPOINT:=${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_LOCAL_GRAPH_ROOT}/taggers/hlt_part_baseline/best_model_val.pt}"
: "${LOCAL_GRAPH_RESIDUAL_V2_ALLOW_BASELINE_SPLIT_MISMATCH:=0}"

: "${LOCAL_GRAPH_RESIDUAL_V2_EXISTING_INPUT_ROOT:=${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_INPUT_ROOT}}"
: "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_EXISTING_INPUT_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT:=${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_BASELINE_CHECKPOINT}}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TAG:=3m1m1m_$(date +%Y%m%d_%H%M%S)}"
: "${LOCAL_GRAPH_RESIDUAL_V2_ROOT:=${OUTPUT_ROOT}/local_graph_residual_expert_v2_qcd_hgg_binary_hlt0p6_3m1m1m_${LOCAL_GRAPH_RESIDUAL_V2_TAG}}"

: "${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES:=A C D}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE:=3000000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE:=1000000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE:=3000000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE:=1000000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE:=1000000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EPOCHS:=30}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"

: "${LOCAL_GRAPH_RESIDUAL_V2_CACHE_TIME:=1-12:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CACHE_MEM:=192G}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CACHE_CPUS:=8}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_TIME:=5-00:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_MEM:=192G}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CPUS:=8}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_TIME:=1-00:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_MEM:=96G}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CPUS:=4}"

if [[ -z "${LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH:-}" ]]; then
  candidate="${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_LOCAL_GRAPH_ROOT}/final_report/local_graph_part_report.json"
  if [[ -f "${candidate}" ]]; then
    LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH="${candidate}"
  fi
fi
if [[ -z "${LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH:-}" ]]; then
  shopt -s nullglob
  score_fusion_candidates=("${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_LOCAL_GRAPH_ROOT}"/score_fusion_*/fusion_report.json)
  shopt -u nullglob
  if [[ ${#score_fusion_candidates[@]} -gt 0 ]]; then
    IFS=$'\n' sorted_score_fusion_candidates=($(printf '%s\n' "${score_fusion_candidates[@]}" | sort))
    unset IFS
    LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH="${sorted_score_fusion_candidates[$((${#sorted_score_fusion_candidates[@]} - 1))]}"
  fi
fi

export LOCAL_GRAPH_RESIDUAL_V2_EXISTING_INPUT_ROOT
export LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR
export LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT
export LOCAL_GRAPH_RESIDUAL_V2_TAG
export LOCAL_GRAPH_RESIDUAL_V2_ROOT
export LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH
export LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES
export LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_EPOCHS
export LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC
export LOCAL_GRAPH_RESIDUAL_V2_CACHE_TIME
export LOCAL_GRAPH_RESIDUAL_V2_CACHE_MEM
export LOCAL_GRAPH_RESIDUAL_V2_CACHE_CPUS
export LOCAL_GRAPH_RESIDUAL_V2_TRAIN_TIME
export LOCAL_GRAPH_RESIDUAL_V2_TRAIN_MEM
export LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CPUS
export LOCAL_GRAPH_RESIDUAL_V2_REPORT_TIME
export LOCAL_GRAPH_RESIDUAL_V2_REPORT_MEM
export LOCAL_GRAPH_RESIDUAL_V2_REPORT_CPUS
export LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH="${LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH:-}"
export LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH="${LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH:-}"

fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}"
baseline_dir="$(dirname "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}")"
fresh_require_file "${baseline_dir}/run_report.json"
if [[ "${LOCAL_GRAPH_RESIDUAL_V2_ALLOW_BASELINE_SPLIT_MISMATCH}" != "1" ]]; then
  if [[ "${baseline_dir}" != *"3m1m1m"* ]] && ! grep -Fq "3m1m1m" "${baseline_dir}/run_report.json"; then
    cat >&2 <<ERROR
LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT does not look like a 3M/1M/1M baseline:
  checkpoint: ${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}
  run_report: ${baseline_dir}/run_report.json

Set LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_BASELINE_CHECKPOINT to the 3M hlt_part_baseline/best_model_val.pt,
or set LOCAL_GRAPH_RESIDUAL_V2_ALLOW_BASELINE_SPLIT_MISMATCH=1 only for deliberate debugging.
ERROR
    exit 2
  fi
fi
for split in model_train model_val stack_train stack_val final_test; do
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done

if fresh_is_dry_run; then
  echo "DRY_RUN serious V2 preset will delegate sbatch submission to submit_local_graph_residual_expert_v2_experiment.sh" >&2
fi

cat <<PRESET >&2
local_graph_residual_expert_v2_3m1m1m_serious_preset:
  input_root: ${LOCAL_GRAPH_RESIDUAL_V2_EXISTING_INPUT_ROOT}
  local_graph_root: ${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_LOCAL_GRAPH_ROOT}
  hlt_cache: ${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}
  baseline_checkpoint: ${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}
  optional_comparisons:
    standalone_report: ${LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH:-none}
    score_fusion_report: ${LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH:-none}
  output_root: ${LOCAL_GRAPH_RESIDUAL_V2_ROOT}
  loss_modes: ${LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES}
  split_caps:
    model_train: ${LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE}
    model_val: ${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE}
    stack_train: ${LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE}
    stack_val: ${LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE}
    final_test: ${LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE}
  resources:
    cache: time=${LOCAL_GRAPH_RESIDUAL_V2_CACHE_TIME} mem=${LOCAL_GRAPH_RESIDUAL_V2_CACHE_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_V2_CACHE_CPUS}
    train: time=${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_TIME} mem=${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CPUS}
    report: time=${LOCAL_GRAPH_RESIDUAL_V2_REPORT_TIME} mem=${LOCAL_GRAPH_RESIDUAL_V2_REPORT_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CPUS}
PRESET

# The generic submitter owns the actual sbatch graph and afterok dependencies.
exec "${SCRIPT_DIR}/submit_local_graph_residual_expert_v2_experiment.sh" "$@"

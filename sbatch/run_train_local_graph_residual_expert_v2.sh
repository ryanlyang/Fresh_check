#!/usr/bin/env bash
# Train one local-graph residual expert V2 mode.

#SBATCH --job-name=lgresidv2_train
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-12:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

REQUESTED_LOSS_MODE="${1:?Usage: sbatch run_train_local_graph_residual_expert_v2.sh <A|C|D|canonical-loss-mode>}"

normalize_v2_loss_mode() {
  local value="${1,,}"
  value="${value//-/_}"
  case "${value}" in
    a|weighted_bce|residual_v2_weighted_bce)
      echo "residual_v2_weighted_bce"
      ;;
    b|boundary_pairwise|residual_v2_boundary_pairwise)
      echo "residual_v2_boundary_pairwise"
      ;;
    c|boundary_pairwise_bce_anchor|residual_v2_boundary_pairwise_bce_anchor)
      echo "residual_v2_boundary_pairwise_bce_anchor"
      ;;
    d|boundary_pairwise_soft_fpr_bce_anchor|residual_v2_boundary_pairwise_soft_fpr_bce_anchor)
      echo "residual_v2_boundary_pairwise_soft_fpr_bce_anchor"
      ;;
    e|alpha_shrink|gamma_shrink|validation_shrinkage|residual_v2_boundary_pairwise_soft_fpr_bce_anchor_alpha_shrink)
      echo "V2 ladder E is a report policy, not a training job. Submit D and read validation_shrunk rows." >&2
      return 2
      ;;
    *)
      echo "Unknown local graph residual V2 loss mode: ${1}" >&2
      return 2
      ;;
  esac
}

safe_v2_label() {
  local value="${1,,}"
  value="${value//residual_v2_/}"
  value="${value//boundary_pairwise/bpair}"
  value="${value//soft_fpr_bce_anchor/sfpr_bce}"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s' "${value}"
}

LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODE="$(normalize_v2_loss_mode "${REQUESTED_LOSS_MODE}")"

: "${LOCAL_GRAPH_RESIDUAL_V2_ROOT:=${OUTPUT_ROOT}/local_graph_residual_expert_v2_qcd_hgg_binary_hlt0p6}"
: "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/baseline_embeddings}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/residual_experts}"
: "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_NAME:=$(safe_v2_label "${LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODE}")}"
: "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT}/${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_NAME}}"

: "${LOCAL_GRAPH_RESIDUAL_V2_SEED:=5207}"
: "${LOCAL_GRAPH_RESIDUAL_V2_BATCH_SIZE:=64}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EVAL_BATCH_SIZE:=128}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EPOCHS:=30}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LR:=0.0003}"
: "${LOCAL_GRAPH_RESIDUAL_V2_WEIGHT_DECAY:=0.0001}"
: "${LOCAL_GRAPH_RESIDUAL_V2_NUM_WORKERS:=${NUM_WORKERS}}"
: "${LOCAL_GRAPH_RESIDUAL_V2_DEVICE:=${DEVICE}}"
: "${LOCAL_GRAPH_RESIDUAL_V2_GRAD_CLIP_NORM:=${GRAD_CLIP_NORM}}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EARLY_STOP_PATIENCE:=6}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MAX_TRAIN_BATCHES:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MAX_VAL_BATCHES:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_EMBEDDING_DIM:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MAX_CONSTITS:=128}"
: "${LOCAL_GRAPH_RESIDUAL_V2_K:=16}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_EMBED_DIM:=128}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_HEADS:=8}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_HIDDEN_DIM:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_ADAPTER_GAMMA_INIT:=1.0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_POOL_MODE:=mean_max}"
: "${LOCAL_GRAPH_RESIDUAL_V2_DROPOUT:=0.05}"
: "${LOCAL_GRAPH_RESIDUAL_V2_ATTENTION_DROPOUT:=0.05}"
: "${LOCAL_GRAPH_RESIDUAL_V2_WEIGHT_THRESHOLD:=0.0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CONDITION_EMBED_DIM:=64}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_CONTEXT_DIM:=128}"
: "${LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_HIDDEN_DIM:=256}"
: "${LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_DROPOUT:=0.05}"
: "${LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_OUTPUT_SCALE:=1.0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_GATE_BIAS_INIT:=-1.0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_DELTA_INIT_STD:=0.001}"
: "${LOCAL_GRAPH_RESIDUAL_V2_GAMMA_INITIAL:=0.1}"
: "${LOCAL_GRAPH_RESIDUAL_V2_GAMMA_LEARNABLE:=1}"
: "${LOCAL_GRAPH_RESIDUAL_V2_GAMMA_MAX:=2.0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_DISABLE_GAMMA_MAX:=0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_INPUT_MODE:=full}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODE:=normal}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CONDITION_SHUFFLE_SEED:=520701}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODE:=normal}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LABEL_SHUFFLE_SEED:=520702}"

: "${LOCAL_GRAPH_RESIDUAL_V2_BCE_ANCHOR_WEIGHT:=0.05}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SOFT_FPR_WEIGHT:=0.25}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CORRECTION_L2_WEIGHT:=0.0001}"
: "${LOCAL_GRAPH_RESIDUAL_V2_PAIRWISE_WEIGHT:=1.0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_WEIGHTED_BCE_WEIGHT:=1.0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_PAIRWISE_TEMPERATURE:=0.20}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SOFT_FPR_EPSILON:=0.20}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CVAR_TOP_FRACTION:=0.50}"
: "${LOCAL_GRAPH_RESIDUAL_V2_HARD_BACKGROUND_FRACTION:=0.20}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SIGNAL_BOUNDARY_QUANTILE_LOW:=0.40}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SIGNAL_BOUNDARY_QUANTILE_HIGH:=0.60}"
: "${LOCAL_GRAPH_RESIDUAL_V2_BCE_BOUNDARY_SCALE:=}"

: "${LOCAL_GRAPH_RESIDUAL_V2_NO_AMP:=0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_COMPILE_MODEL:=0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SKIP_HLT_HASH_CHECK:=0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SKIP_HLT_PARAMS_CHECK:=0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/train_local_graph_residual_expert_v2.py"
for split in model_train model_val; do
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}/${split}_baseline_embedding_cache.npz"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}/${split}_baseline_embedding_cache_metadata.json"
done
fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}/baseline_embedding_manifest.json"
fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}/run_report.json"
fresh_claim_new_dir "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_local_graph_residual_expert_v2.py"
  --output-dir "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR}"
  --hlt-cache-dir "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}"
  --baseline-embedding-cache-dir "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}"
  --train-split model_train
  --val-split model_val
  --confirm-split-settings
  --seed "${LOCAL_GRAPH_RESIDUAL_V2_SEED}"
  --batch-size "${LOCAL_GRAPH_RESIDUAL_V2_BATCH_SIZE}"
  --eval-batch-size "${LOCAL_GRAPH_RESIDUAL_V2_EVAL_BATCH_SIZE}"
  --epochs "${LOCAL_GRAPH_RESIDUAL_V2_EPOCHS}"
  --lr "${LOCAL_GRAPH_RESIDUAL_V2_LR}"
  --weight-decay "${LOCAL_GRAPH_RESIDUAL_V2_WEIGHT_DECAY}"
  --num-workers "${LOCAL_GRAPH_RESIDUAL_V2_NUM_WORKERS}"
  --device "${LOCAL_GRAPH_RESIDUAL_V2_DEVICE}"
  --grad-clip-norm "${LOCAL_GRAPH_RESIDUAL_V2_GRAD_CLIP_NORM}"
  --early-stop-patience "${LOCAL_GRAPH_RESIDUAL_V2_EARLY_STOP_PATIENCE}"
  --max-train-jets "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE}"
  --max-val-jets "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE}"
  --selection-metric "${LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC}"
  --expected-hlt-degradation-strength "${LOCAL_GRAPH_RESIDUAL_V2_EXPECTED_HLT_DEGRADATION_STRENGTH}"
  --max-constits "${LOCAL_GRAPH_RESIDUAL_V2_MAX_CONSTITS}"
  --k "${LOCAL_GRAPH_RESIDUAL_V2_K}"
  --local-embed-dim "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_EMBED_DIM}"
  --local-heads "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_HEADS}"
  --local-adapter-gamma-init "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_ADAPTER_GAMMA_INIT}"
  --local-pool-mode "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_POOL_MODE}"
  --dropout "${LOCAL_GRAPH_RESIDUAL_V2_DROPOUT}"
  --attention-dropout "${LOCAL_GRAPH_RESIDUAL_V2_ATTENTION_DROPOUT}"
  --weight-threshold "${LOCAL_GRAPH_RESIDUAL_V2_WEIGHT_THRESHOLD}"
  --condition-embed-dim "${LOCAL_GRAPH_RESIDUAL_V2_CONDITION_EMBED_DIM}"
  --local-context-dim "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_CONTEXT_DIM}"
  --residual-hidden-dim "${LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_HIDDEN_DIM}"
  --residual-dropout "${LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_DROPOUT}"
  --residual-output-scale "${LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_OUTPUT_SCALE}"
  --gate-bias-init "${LOCAL_GRAPH_RESIDUAL_V2_GATE_BIAS_INIT}"
  --delta-init-std "${LOCAL_GRAPH_RESIDUAL_V2_DELTA_INIT_STD}"
  --gamma-initial "${LOCAL_GRAPH_RESIDUAL_V2_GAMMA_INITIAL}"
  --gamma-max "${LOCAL_GRAPH_RESIDUAL_V2_GAMMA_MAX}"
  --residual-input-mode "${LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_INPUT_MODE}"
  --condition-control-mode "${LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODE}"
  --condition-shuffle-seed "${LOCAL_GRAPH_RESIDUAL_V2_CONDITION_SHUFFLE_SEED}"
  --label-control-mode "${LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODE}"
  --label-shuffle-seed "${LOCAL_GRAPH_RESIDUAL_V2_LABEL_SHUFFLE_SEED}"
  --loss-mode "${LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODE}"
  --bce-anchor-weight "${LOCAL_GRAPH_RESIDUAL_V2_BCE_ANCHOR_WEIGHT}"
  --soft-fpr-weight "${LOCAL_GRAPH_RESIDUAL_V2_SOFT_FPR_WEIGHT}"
  --correction-l2-weight "${LOCAL_GRAPH_RESIDUAL_V2_CORRECTION_L2_WEIGHT}"
  --pairwise-weight "${LOCAL_GRAPH_RESIDUAL_V2_PAIRWISE_WEIGHT}"
  --weighted-bce-weight "${LOCAL_GRAPH_RESIDUAL_V2_WEIGHTED_BCE_WEIGHT}"
  --pairwise-temperature "${LOCAL_GRAPH_RESIDUAL_V2_PAIRWISE_TEMPERATURE}"
  --soft-fpr-epsilon "${LOCAL_GRAPH_RESIDUAL_V2_SOFT_FPR_EPSILON}"
  --cvar-top-fraction "${LOCAL_GRAPH_RESIDUAL_V2_CVAR_TOP_FRACTION}"
  --hard-background-fraction "${LOCAL_GRAPH_RESIDUAL_V2_HARD_BACKGROUND_FRACTION}"
  --signal-boundary-quantile-low "${LOCAL_GRAPH_RESIDUAL_V2_SIGNAL_BOUNDARY_QUANTILE_LOW}"
  --signal-boundary-quantile-high "${LOCAL_GRAPH_RESIDUAL_V2_SIGNAL_BOUNDARY_QUANTILE_HIGH}"
)
fresh_append_optional_arg cmd --baseline-embedding-dim "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_EMBEDDING_DIM}"
fresh_append_optional_arg cmd --local-hidden-dim "${LOCAL_GRAPH_RESIDUAL_V2_LOCAL_HIDDEN_DIM}"
fresh_append_optional_arg cmd --max-train-batches "${LOCAL_GRAPH_RESIDUAL_V2_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${LOCAL_GRAPH_RESIDUAL_V2_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --bce-boundary-scale "${LOCAL_GRAPH_RESIDUAL_V2_BCE_BOUNDARY_SCALE}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${LOCAL_GRAPH_RESIDUAL_V2_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --no-amp "${LOCAL_GRAPH_RESIDUAL_V2_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${LOCAL_GRAPH_RESIDUAL_V2_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${LOCAL_GRAPH_RESIDUAL_V2_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --skip-hlt-params-check "${LOCAL_GRAPH_RESIDUAL_V2_SKIP_HLT_PARAMS_CHECK}"
if ! fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_V2_GAMMA_LEARNABLE}"; then
  cmd+=(--disable-gamma-learnable)
fi
fresh_append_flag_if_enabled cmd --disable-gamma-max "${LOCAL_GRAPH_RESIDUAL_V2_DISABLE_GAMMA_MAX}"

fresh_write_run_config "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR}" "local_graph_residual_v2_${LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR}/last.pt"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR}/config.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR}/run_report.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_DIR}/diagnostics/model_val_learned_gamma_predictions.npz"
fi

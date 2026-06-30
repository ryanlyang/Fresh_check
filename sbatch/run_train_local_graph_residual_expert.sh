#!/usr/bin/env bash
# Train one local-graph residual expert against a frozen HLT ParT score.

#SBATCH --job-name=localgraph_resid
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

REQUESTED_LOSS_MODE="${1:?Usage: sbatch run_train_local_graph_residual_expert.sh <A|B|C|D|canonical-loss-mode> [none|edgeconv|point_attention]}"
REQUESTED_LOCAL_ADAPTER="${2:-point_attention}"

normalize_loss_mode() {
  local value="${1,,}"
  value="${value//-/_}"
  case "${value}" in
    a|weighted_bce|residual_weighted_bce)
      echo "residual_weighted_bce"
      ;;
    b|boundary_pairwise|residual_boundary_pairwise)
      echo "residual_boundary_pairwise"
      ;;
    c|boundary_pairwise_bce_anchor|residual_boundary_pairwise_bce_anchor)
      echo "residual_boundary_pairwise_bce_anchor"
      ;;
    d|boundary_pairwise_soft_fpr_bce_anchor|residual_boundary_pairwise_soft_fpr_bce_anchor)
      echo "residual_boundary_pairwise_soft_fpr_bce_anchor"
      ;;
    e|boundary_pairwise_soft_fpr_bce_anchor_alpha_shrink|residual_boundary_pairwise_soft_fpr_bce_anchor_alpha_shrink)
      echo "Residual ladder E is a report policy, not a training job. Submit D and read the val_shrunk rows." >&2
      return 2
      ;;
    *)
      echo "Unknown local graph residual loss mode: ${1}" >&2
      return 2
      ;;
  esac
}

normalize_local_adapter() {
  local value="${1,,}"
  value="${value//-/_}"
  case "${value}" in
    none|no_adapter|hlt_part_baseline)
      echo "none"
      ;;
    edgeconv|local_edgeconv_adapter)
      echo "edgeconv"
      ;;
    point|point_attention|local_point_attention_adapter|local_point_attention_adapter_warmstart)
      echo "point_attention"
      ;;
    *)
      echo "Unknown local graph residual local adapter: ${1}" >&2
      return 2
      ;;
  esac
}

LOCAL_GRAPH_RESIDUAL_LOSS_MODE="$(normalize_loss_mode "${REQUESTED_LOSS_MODE}")"
LOCAL_GRAPH_RESIDUAL_LOCAL_ADAPTER="$(normalize_local_adapter "${REQUESTED_LOCAL_ADAPTER}")"

: "${LOCAL_GRAPH_RESIDUAL_ROOT:=${OUTPUT_ROOT}/local_graph_residual_expert_qcd_hgg_binary_hlt0p6}"
: "${LOCAL_GRAPH_RESIDUAL_EXPERT_ROOT:=${LOCAL_GRAPH_RESIDUAL_ROOT}/residual_experts}"
: "${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_ROOT}/baseline_logits}"
: "${LOCAL_GRAPH_RESIDUAL_OUTPUT_NAME:=${LOCAL_GRAPH_RESIDUAL_LOSS_MODE}}"
: "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR:=${LOCAL_GRAPH_RESIDUAL_EXPERT_ROOT}/${LOCAL_GRAPH_RESIDUAL_OUTPUT_NAME}}"

: "${LOCAL_GRAPH_RESIDUAL_SEED:=4107}"
: "${LOCAL_GRAPH_RESIDUAL_BATCH_SIZE:=64}"
: "${LOCAL_GRAPH_RESIDUAL_EVAL_BATCH_SIZE:=128}"
: "${LOCAL_GRAPH_RESIDUAL_EPOCHS:=30}"
: "${LOCAL_GRAPH_RESIDUAL_LR:=0.0003}"
: "${LOCAL_GRAPH_RESIDUAL_WEIGHT_DECAY:=0.0001}"
: "${LOCAL_GRAPH_RESIDUAL_NUM_WORKERS:=${NUM_WORKERS}}"
: "${LOCAL_GRAPH_RESIDUAL_DEVICE:=${DEVICE}}"
: "${LOCAL_GRAPH_RESIDUAL_GRAD_CLIP_NORM:=${GRAD_CLIP_NORM}}"
: "${LOCAL_GRAPH_RESIDUAL_EARLY_STOP_PATIENCE:=6}"
: "${LOCAL_GRAPH_RESIDUAL_MAX_TRAIN_BATCHES:=}"
: "${LOCAL_GRAPH_RESIDUAL_MAX_VAL_BATCHES:=}"
: "${LOCAL_GRAPH_RESIDUAL_MODEL_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_RESIDUAL_MODEL_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_RESIDUAL_MODEL_SIZE:=base}"
: "${LOCAL_GRAPH_RESIDUAL_MAX_CONSTITS:=128}"
: "${LOCAL_GRAPH_RESIDUAL_K:=16}"
: "${LOCAL_GRAPH_RESIDUAL_LOCAL_EMBED_DIM:=128}"
: "${LOCAL_GRAPH_RESIDUAL_LOCAL_HEADS:=8}"
: "${LOCAL_GRAPH_RESIDUAL_LOCAL_HIDDEN_DIM:=}"
: "${LOCAL_GRAPH_RESIDUAL_DROPOUT:=0.05}"
: "${LOCAL_GRAPH_RESIDUAL_ATTENTION_DROPOUT:=0.05}"
: "${LOCAL_GRAPH_RESIDUAL_RESIDUAL_GAMMA_INIT:=0.0}"
: "${LOCAL_GRAPH_RESIDUAL_WEIGHT_THRESHOLD:=0.0}"
: "${LOCAL_GRAPH_RESIDUAL_BACKBONE_OUTPUT_DIM:=128}"
: "${LOCAL_GRAPH_RESIDUAL_CONDITION_EMBED_DIM:=64}"
: "${LOCAL_GRAPH_RESIDUAL_RESIDUAL_HIDDEN_DIM:=128}"
: "${LOCAL_GRAPH_RESIDUAL_RESIDUAL_DROPOUT:=0.05}"
: "${LOCAL_GRAPH_RESIDUAL_ALPHA_INITIAL:=0.1}"
: "${LOCAL_GRAPH_RESIDUAL_ALPHA_MAX:=2.0}"
: "${LOCAL_GRAPH_RESIDUAL_ALPHA_LEARNABLE:=1}"
: "${LOCAL_GRAPH_RESIDUAL_DISABLE_ALPHA_MAX:=0}"

: "${LOCAL_GRAPH_RESIDUAL_BCE_ANCHOR_WEIGHT:=0.10}"
: "${LOCAL_GRAPH_RESIDUAL_SOFT_FPR_WEIGHT:=0.25}"
: "${LOCAL_GRAPH_RESIDUAL_RESIDUAL_L2_WEIGHT:=0.0001}"
: "${LOCAL_GRAPH_RESIDUAL_PAIRWISE_WEIGHT:=1.0}"
: "${LOCAL_GRAPH_RESIDUAL_WEIGHTED_BCE_WEIGHT:=1.0}"
: "${LOCAL_GRAPH_RESIDUAL_PAIRWISE_TEMPERATURE:=0.20}"
: "${LOCAL_GRAPH_RESIDUAL_SOFT_FPR_EPSILON:=0.20}"
: "${LOCAL_GRAPH_RESIDUAL_CVAR_TOP_FRACTION:=0.50}"
: "${LOCAL_GRAPH_RESIDUAL_HARD_BACKGROUND_FRACTION:=0.20}"
: "${LOCAL_GRAPH_RESIDUAL_SIGNAL_BOUNDARY_QUANTILE_LOW:=0.40}"
: "${LOCAL_GRAPH_RESIDUAL_SIGNAL_BOUNDARY_QUANTILE_HIGH:=0.60}"
: "${LOCAL_GRAPH_RESIDUAL_BCE_BOUNDARY_SCALE:=}"

: "${LOCAL_GRAPH_RESIDUAL_WARM_START_ENABLED:=1}"
: "${LOCAL_GRAPH_RESIDUAL_WARM_START_CHECKPOINT:=${LOCAL_GRAPH_RESIDUAL_ROOT}/taggers/hlt_part_baseline/best_model_val.pt}"
: "${LOCAL_GRAPH_RESIDUAL_REQUIRE_WARM_START:=1}"
: "${LOCAL_GRAPH_RESIDUAL_FREEZE_PART_EPOCHS:=0}"
: "${LOCAL_GRAPH_RESIDUAL_NO_AMP:=0}"
: "${LOCAL_GRAPH_RESIDUAL_COMPILE_MODEL:=0}"
: "${LOCAL_GRAPH_RESIDUAL_SKIP_HLT_HASH_CHECK:=0}"
: "${LOCAL_GRAPH_RESIDUAL_SKIP_HLT_PARAMS_CHECK:=0}"
: "${LOCAL_GRAPH_RESIDUAL_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${LOCAL_GRAPH_RESIDUAL_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/train_local_graph_residual_expert.py"
for split in model_train model_val; do
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}/${split}_baseline_logits.npz"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}/${split}_baseline_logits_metadata.json"
done
fresh_require_file "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}/baseline_logit_manifest.json"
fresh_require_file "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}/run_report.json"
if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_WARM_START_ENABLED}"; then
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_WARM_START_CHECKPOINT}"
fi
fresh_claim_new_dir "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_local_graph_residual_expert.py"
  --output-dir "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}"
  --hlt-cache-dir "${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}"
  --baseline-logit-cache-dir "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}"
  --train-split model_train
  --val-split model_val
  --confirm-split-settings
  --seed "${LOCAL_GRAPH_RESIDUAL_SEED}"
  --batch-size "${LOCAL_GRAPH_RESIDUAL_BATCH_SIZE}"
  --eval-batch-size "${LOCAL_GRAPH_RESIDUAL_EVAL_BATCH_SIZE}"
  --epochs "${LOCAL_GRAPH_RESIDUAL_EPOCHS}"
  --lr "${LOCAL_GRAPH_RESIDUAL_LR}"
  --weight-decay "${LOCAL_GRAPH_RESIDUAL_WEIGHT_DECAY}"
  --num-workers "${LOCAL_GRAPH_RESIDUAL_NUM_WORKERS}"
  --device "${LOCAL_GRAPH_RESIDUAL_DEVICE}"
  --grad-clip-norm "${LOCAL_GRAPH_RESIDUAL_GRAD_CLIP_NORM}"
  --early-stop-patience "${LOCAL_GRAPH_RESIDUAL_EARLY_STOP_PATIENCE}"
  --max-train-jets "${LOCAL_GRAPH_RESIDUAL_MODEL_TRAIN_SIZE}"
  --max-val-jets "${LOCAL_GRAPH_RESIDUAL_MODEL_VAL_SIZE}"
  --selection-metric "${LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC}"
  --expected-hlt-degradation-strength "${LOCAL_GRAPH_RESIDUAL_EXPECTED_HLT_DEGRADATION_STRENGTH}"
  --model-size "${LOCAL_GRAPH_RESIDUAL_MODEL_SIZE}"
  --max-constits "${LOCAL_GRAPH_RESIDUAL_MAX_CONSTITS}"
  --local-adapter "${LOCAL_GRAPH_RESIDUAL_LOCAL_ADAPTER}"
  --k "${LOCAL_GRAPH_RESIDUAL_K}"
  --local-embed-dim "${LOCAL_GRAPH_RESIDUAL_LOCAL_EMBED_DIM}"
  --local-heads "${LOCAL_GRAPH_RESIDUAL_LOCAL_HEADS}"
  --dropout "${LOCAL_GRAPH_RESIDUAL_DROPOUT}"
  --attention-dropout "${LOCAL_GRAPH_RESIDUAL_ATTENTION_DROPOUT}"
  --residual-gamma-init "${LOCAL_GRAPH_RESIDUAL_RESIDUAL_GAMMA_INIT}"
  --weight-threshold "${LOCAL_GRAPH_RESIDUAL_WEIGHT_THRESHOLD}"
  --backbone-output-dim "${LOCAL_GRAPH_RESIDUAL_BACKBONE_OUTPUT_DIM}"
  --condition-embed-dim "${LOCAL_GRAPH_RESIDUAL_CONDITION_EMBED_DIM}"
  --residual-hidden-dim "${LOCAL_GRAPH_RESIDUAL_RESIDUAL_HIDDEN_DIM}"
  --residual-dropout "${LOCAL_GRAPH_RESIDUAL_RESIDUAL_DROPOUT}"
  --alpha-initial "${LOCAL_GRAPH_RESIDUAL_ALPHA_INITIAL}"
  --alpha-max "${LOCAL_GRAPH_RESIDUAL_ALPHA_MAX}"
  --loss-mode "${LOCAL_GRAPH_RESIDUAL_LOSS_MODE}"
  --bce-anchor-weight "${LOCAL_GRAPH_RESIDUAL_BCE_ANCHOR_WEIGHT}"
  --soft-fpr-weight "${LOCAL_GRAPH_RESIDUAL_SOFT_FPR_WEIGHT}"
  --residual-l2-weight "${LOCAL_GRAPH_RESIDUAL_RESIDUAL_L2_WEIGHT}"
  --pairwise-weight "${LOCAL_GRAPH_RESIDUAL_PAIRWISE_WEIGHT}"
  --weighted-bce-weight "${LOCAL_GRAPH_RESIDUAL_WEIGHTED_BCE_WEIGHT}"
  --pairwise-temperature "${LOCAL_GRAPH_RESIDUAL_PAIRWISE_TEMPERATURE}"
  --soft-fpr-epsilon "${LOCAL_GRAPH_RESIDUAL_SOFT_FPR_EPSILON}"
  --cvar-top-fraction "${LOCAL_GRAPH_RESIDUAL_CVAR_TOP_FRACTION}"
  --hard-background-fraction "${LOCAL_GRAPH_RESIDUAL_HARD_BACKGROUND_FRACTION}"
  --signal-boundary-quantile-low "${LOCAL_GRAPH_RESIDUAL_SIGNAL_BOUNDARY_QUANTILE_LOW}"
  --signal-boundary-quantile-high "${LOCAL_GRAPH_RESIDUAL_SIGNAL_BOUNDARY_QUANTILE_HIGH}"
  --freeze-part-epochs "${LOCAL_GRAPH_RESIDUAL_FREEZE_PART_EPOCHS}"
)
fresh_append_optional_arg cmd --local-hidden-dim "${LOCAL_GRAPH_RESIDUAL_LOCAL_HIDDEN_DIM}"
fresh_append_optional_arg cmd --max-train-batches "${LOCAL_GRAPH_RESIDUAL_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${LOCAL_GRAPH_RESIDUAL_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --bce-boundary-scale "${LOCAL_GRAPH_RESIDUAL_BCE_BOUNDARY_SCALE}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${LOCAL_GRAPH_RESIDUAL_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --no-amp "${LOCAL_GRAPH_RESIDUAL_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${LOCAL_GRAPH_RESIDUAL_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${LOCAL_GRAPH_RESIDUAL_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --skip-hlt-params-check "${LOCAL_GRAPH_RESIDUAL_SKIP_HLT_PARAMS_CHECK}"
if ! fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_ALPHA_LEARNABLE}"; then
  cmd+=(--disable-alpha-learnable)
fi
fresh_append_flag_if_enabled cmd --disable-alpha-max "${LOCAL_GRAPH_RESIDUAL_DISABLE_ALPHA_MAX}"
if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_WARM_START_ENABLED}"; then
  cmd+=(--warm-start-checkpoint "${LOCAL_GRAPH_RESIDUAL_WARM_START_CHECKPOINT}")
  fresh_append_flag_if_enabled cmd --require-warm-start "${LOCAL_GRAPH_RESIDUAL_REQUIRE_WARM_START}"
fi

fresh_write_run_config "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}" "local_graph_residual_${LOCAL_GRAPH_RESIDUAL_LOSS_MODE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}/last.pt"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}/config.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}/run_report.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}/diagnostics/residual_diagnostics_model_val.json"
  if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_WARM_START_ENABLED}"; then
    fresh_require_file "${LOCAL_GRAPH_RESIDUAL_OUTPUT_DIR}/diagnostics/warm_start_report.json"
  fi
fi

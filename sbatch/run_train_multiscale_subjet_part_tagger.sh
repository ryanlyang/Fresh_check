#!/usr/bin/env bash
# Train one multi-scale subjet HLT ParT comparison variant.

#SBATCH --job-name=multiscale_subjet
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

REQUESTED_PROFILE="${1:?Usage: sbatch run_train_multiscale_subjet_part_tagger.sh <profile-or-variant>}"

: "${MULTISCALE_SUBJET_PART_ROOT:=${OUTPUT_ROOT}/multiscale_subjet_part_qcd_hgg_binary_hlt0p6}"
: "${MULTISCALE_SUBJET_PART_TAGGER_ROOT:=${MULTISCALE_SUBJET_PART_ROOT}/taggers}"
: "${MULTISCALE_SUBJET_PART_HLT_CACHE_DIR:=${MULTISCALE_SUBJET_PART_ROOT}/binary_inputs/hlt_cache}"
: "${MULTISCALE_SUBJET_PART_SEED:=4107}"
: "${MULTISCALE_SUBJET_PART_BATCH_SIZE:=64}"
: "${MULTISCALE_SUBJET_PART_EVAL_BATCH_SIZE:=128}"
: "${MULTISCALE_SUBJET_PART_EPOCHS:=45}"
: "${MULTISCALE_SUBJET_PART_LR:=0.0003}"
: "${MULTISCALE_SUBJET_PART_WEIGHT_DECAY:=0.0001}"
: "${MULTISCALE_SUBJET_PART_NUM_WORKERS:=${NUM_WORKERS}}"
: "${MULTISCALE_SUBJET_PART_DEVICE:=${DEVICE}}"
: "${MULTISCALE_SUBJET_PART_GRAD_CLIP_NORM:=${GRAD_CLIP_NORM}}"
: "${MULTISCALE_SUBJET_PART_EARLY_STOP_PATIENCE:=6}"
: "${MULTISCALE_SUBJET_PART_MAX_TRAIN_BATCHES:=}"
: "${MULTISCALE_SUBJET_PART_MAX_VAL_BATCHES:=}"
: "${MULTISCALE_SUBJET_PART_MAX_STACK_VAL_BATCHES:=}"
: "${MULTISCALE_SUBJET_PART_MAX_FINAL_TEST_BATCHES:=}"
: "${MULTISCALE_SUBJET_PART_MODEL_TRAIN_SIZE:=500000}"
: "${MULTISCALE_SUBJET_PART_MODEL_VAL_SIZE:=150000}"
: "${MULTISCALE_SUBJET_PART_STACK_VAL_SIZE:=150000}"
: "${MULTISCALE_SUBJET_PART_FINAL_TEST_SIZE:=500000}"
: "${MULTISCALE_SUBJET_PART_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${MULTISCALE_SUBJET_PART_MODEL_SIZE:=base}"
: "${MULTISCALE_SUBJET_PART_MAX_CONSTITS:=128}"
: "${MULTISCALE_SUBJET_PART_TOKEN_DIM:=128}"
: "${MULTISCALE_SUBJET_PART_TOKEN_HIDDEN_DIM:=256}"
: "${MULTISCALE_SUBJET_PART_ASSIGNMENT_EMBED_DIM:=64}"
: "${MULTISCALE_SUBJET_PART_ASSIGNMENT_HIDDEN_DIM:=128}"
: "${MULTISCALE_SUBJET_PART_ASSIGNMENT_TEMPERATURE:=1.0}"
: "${MULTISCALE_SUBJET_PART_ASSIGNMENT_GEOMETRY_BIAS_STRENGTH:=2.0}"
: "${MULTISCALE_SUBJET_PART_TRANSFORMER_LAYERS:=2}"
: "${MULTISCALE_SUBJET_PART_TRANSFORMER_HEADS:=4}"
: "${MULTISCALE_SUBJET_PART_TRANSFORMER_FFN_DIM:=256}"
: "${MULTISCALE_SUBJET_PART_TRANSFORMER_PAIR_BIAS_HIDDEN_DIM:=64}"
: "${MULTISCALE_SUBJET_PART_READBACK_HIDDEN_DIM:=128}"
: "${MULTISCALE_SUBJET_PART_READBACK_HEADS:=4}"
: "${MULTISCALE_SUBJET_PART_READBACK_DELTA_HIDDEN_DIM:=256}"
: "${MULTISCALE_SUBJET_PART_BRANCH_HIDDEN_DIM:=256}"
: "${MULTISCALE_SUBJET_PART_RESIDUAL_GAMMA_INIT:=0.0}"
: "${MULTISCALE_SUBJET_PART_DROPOUT:=0.05}"
: "${MULTISCALE_SUBJET_PART_ATTENTION_DROPOUT:=0.05}"
: "${MULTISCALE_SUBJET_PART_SCALE_PROFILE:=default}"
: "${MULTISCALE_SUBJET_PART_DISABLE_ASSIGNMENT_SCALE_EMBEDDING:=0}"
: "${MULTISCALE_SUBJET_PART_DISABLE_TOKEN_SCALE_EMBEDDING:=0}"
: "${MULTISCALE_SUBJET_PART_DISABLE_SUBJET_PAIR_BIAS:=0}"
: "${MULTISCALE_SUBJET_PART_DISABLE_SCALE_PAIR_EMBEDDING:=0}"
: "${MULTISCALE_SUBJET_PART_RANDOM_CONTROL_SEED:=2027}"
: "${MULTISCALE_SUBJET_PART_WEIGHT_THRESHOLD:=0.0}"
: "${MULTISCALE_SUBJET_PART_NO_AMP:=0}"
: "${MULTISCALE_SUBJET_PART_COMPILE_MODEL:=0}"
: "${MULTISCALE_SUBJET_PART_SKIP_HLT_HASH_CHECK:=0}"
: "${MULTISCALE_SUBJET_PART_SKIP_HLT_PARAMS_CHECK:=0}"
: "${MULTISCALE_SUBJET_PART_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${MULTISCALE_SUBJET_PART_CONFIRM_FINAL_TEST:=1}"

MODEL_VARIANT="${REQUESTED_PROFILE}"
case "${REQUESTED_PROFILE}" in
  hlt_part_baseline|multiscale_subjet_residual_part_adapter|pure_perceiver_latent_control|part_plus_random_subjet_control|subjet_branch_only|part_plus_subjet_late_fusion|part_plus_subjet_cls_fusion|part_plus_subjet_cross_attention_fusion|two_hlt_part_ensemble_control)
    ;;
  main|primary|residual_adapter)
    MODEL_VARIANT="multiscale_subjet_residual_part_adapter"
    ;;
  no_scale_bias)
    MODEL_VARIANT="multiscale_subjet_residual_part_adapter"
    MULTISCALE_SUBJET_PART_DISABLE_ASSIGNMENT_SCALE_EMBEDDING=1
    MULTISCALE_SUBJET_PART_DISABLE_TOKEN_SCALE_EMBEDDING=1
    MULTISCALE_SUBJET_PART_DISABLE_SCALE_PAIR_EMBEDDING=1
    ;;
  one_scale|one_scale_medium)
    MODEL_VARIANT="multiscale_subjet_residual_part_adapter"
    MULTISCALE_SUBJET_PART_SCALE_PROFILE="one_scale_medium"
    ;;
  one_scale_small)
    MODEL_VARIANT="multiscale_subjet_residual_part_adapter"
    MULTISCALE_SUBJET_PART_SCALE_PROFILE="one_scale_small"
    ;;
  one_scale_large)
    MODEL_VARIANT="multiscale_subjet_residual_part_adapter"
    MULTISCALE_SUBJET_PART_SCALE_PROFILE="one_scale_large"
    ;;
  no_seeded_queries)
    MODEL_VARIANT="pure_perceiver_latent_control"
    ;;
  no_subjet_transformer)
    MODEL_VARIANT="multiscale_subjet_residual_part_adapter"
    MULTISCALE_SUBJET_PART_TRANSFORMER_LAYERS=0
    ;;
  no_particle_readback)
    MODEL_VARIANT="subjet_branch_only"
    ;;
  late_fusion)
    MODEL_VARIANT="part_plus_subjet_late_fusion"
    ;;
  cls_fusion)
    MODEL_VARIANT="part_plus_subjet_cls_fusion"
    ;;
  cross_attention_branch_fusion)
    MODEL_VARIANT="part_plus_subjet_cross_attention_fusion"
    ;;
  few_subjets)
    MODEL_VARIANT="multiscale_subjet_residual_part_adapter"
    MULTISCALE_SUBJET_PART_SCALE_PROFILE="few_subjets"
    ;;
  many_subjets)
    MODEL_VARIANT="multiscale_subjet_residual_part_adapter"
    MULTISCALE_SUBJET_PART_SCALE_PROFILE="many_subjets"
    ;;
  physics_bias_removed)
    MODEL_VARIANT="multiscale_subjet_residual_part_adapter"
    MULTISCALE_SUBJET_PART_DISABLE_SUBJET_PAIR_BIAS=1
    MULTISCALE_SUBJET_PART_ASSIGNMENT_GEOMETRY_BIAS_STRENGTH=0.0
    ;;
  large_hlt_part_control)
    MODEL_VARIANT="hlt_part_baseline"
    MULTISCALE_SUBJET_PART_MODEL_SIZE="large"
    ;;
  two_hlt_part_ensemble|two_hlt_part_ensemble_control)
    MODEL_VARIANT="two_hlt_part_ensemble_control"
    ;;
  *)
    echo "Unknown multiscale subjet profile/variant: ${REQUESTED_PROFILE}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${MULTISCALE_SUBJET_PART_TAGGER_ROOT}/${REQUESTED_PROFILE}"

fresh_setup "$@"
fresh_require_file "scripts/train_multiscale_subjet_part_tagger.py"
for split in model_train model_val stack_val final_test; do
  fresh_require_file "${MULTISCALE_SUBJET_PART_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${MULTISCALE_SUBJET_PART_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_multiscale_subjet_part_tagger.py"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${MULTISCALE_SUBJET_PART_HLT_CACHE_DIR}"
  --variant "${MODEL_VARIANT}"
  --ablation-profile "${REQUESTED_PROFILE}"
  --train-split model_train
  --val-split model_val
  --stack-val-split stack_val
  --final-test-split final_test
  --confirm-split-settings
  --seed "${MULTISCALE_SUBJET_PART_SEED}"
  --batch-size "${MULTISCALE_SUBJET_PART_BATCH_SIZE}"
  --eval-batch-size "${MULTISCALE_SUBJET_PART_EVAL_BATCH_SIZE}"
  --epochs "${MULTISCALE_SUBJET_PART_EPOCHS}"
  --lr "${MULTISCALE_SUBJET_PART_LR}"
  --weight-decay "${MULTISCALE_SUBJET_PART_WEIGHT_DECAY}"
  --num-workers "${MULTISCALE_SUBJET_PART_NUM_WORKERS}"
  --device "${MULTISCALE_SUBJET_PART_DEVICE}"
  --grad-clip-norm "${MULTISCALE_SUBJET_PART_GRAD_CLIP_NORM}"
  --early-stop-patience "${MULTISCALE_SUBJET_PART_EARLY_STOP_PATIENCE}"
  --max-train-jets "${MULTISCALE_SUBJET_PART_MODEL_TRAIN_SIZE}"
  --max-val-jets "${MULTISCALE_SUBJET_PART_MODEL_VAL_SIZE}"
  --max-stack-val-jets "${MULTISCALE_SUBJET_PART_STACK_VAL_SIZE}"
  --max-final-test-jets "${MULTISCALE_SUBJET_PART_FINAL_TEST_SIZE}"
  --selection-metric "${MULTISCALE_SUBJET_PART_SELECTION_METRIC}"
  --expected-hlt-degradation-strength "${MULTISCALE_SUBJET_PART_EXPECTED_HLT_DEGRADATION_STRENGTH}"
  --model-size "${MULTISCALE_SUBJET_PART_MODEL_SIZE}"
  --max-constits "${MULTISCALE_SUBJET_PART_MAX_CONSTITS}"
  --token-dim "${MULTISCALE_SUBJET_PART_TOKEN_DIM}"
  --token-hidden-dim "${MULTISCALE_SUBJET_PART_TOKEN_HIDDEN_DIM}"
  --assignment-embed-dim "${MULTISCALE_SUBJET_PART_ASSIGNMENT_EMBED_DIM}"
  --assignment-hidden-dim "${MULTISCALE_SUBJET_PART_ASSIGNMENT_HIDDEN_DIM}"
  --assignment-temperature "${MULTISCALE_SUBJET_PART_ASSIGNMENT_TEMPERATURE}"
  --assignment-geometry-bias-strength "${MULTISCALE_SUBJET_PART_ASSIGNMENT_GEOMETRY_BIAS_STRENGTH}"
  --transformer-layers "${MULTISCALE_SUBJET_PART_TRANSFORMER_LAYERS}"
  --transformer-heads "${MULTISCALE_SUBJET_PART_TRANSFORMER_HEADS}"
  --transformer-ffn-dim "${MULTISCALE_SUBJET_PART_TRANSFORMER_FFN_DIM}"
  --transformer-pair-bias-hidden-dim "${MULTISCALE_SUBJET_PART_TRANSFORMER_PAIR_BIAS_HIDDEN_DIM}"
  --readback-hidden-dim "${MULTISCALE_SUBJET_PART_READBACK_HIDDEN_DIM}"
  --readback-heads "${MULTISCALE_SUBJET_PART_READBACK_HEADS}"
  --readback-delta-hidden-dim "${MULTISCALE_SUBJET_PART_READBACK_DELTA_HIDDEN_DIM}"
  --branch-hidden-dim "${MULTISCALE_SUBJET_PART_BRANCH_HIDDEN_DIM}"
  --residual-gamma-init "${MULTISCALE_SUBJET_PART_RESIDUAL_GAMMA_INIT}"
  --dropout "${MULTISCALE_SUBJET_PART_DROPOUT}"
  --attention-dropout "${MULTISCALE_SUBJET_PART_ATTENTION_DROPOUT}"
  --scale-profile "${MULTISCALE_SUBJET_PART_SCALE_PROFILE}"
  --random-control-seed "${MULTISCALE_SUBJET_PART_RANDOM_CONTROL_SEED}"
  --weight-threshold "${MULTISCALE_SUBJET_PART_WEIGHT_THRESHOLD}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${MULTISCALE_SUBJET_PART_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --no-amp "${MULTISCALE_SUBJET_PART_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${MULTISCALE_SUBJET_PART_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${MULTISCALE_SUBJET_PART_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --skip-hlt-params-check "${MULTISCALE_SUBJET_PART_SKIP_HLT_PARAMS_CHECK}"
fresh_append_flag_if_enabled cmd --disable-assignment-scale-embedding "${MULTISCALE_SUBJET_PART_DISABLE_ASSIGNMENT_SCALE_EMBEDDING}"
fresh_append_flag_if_enabled cmd --disable-token-scale-embedding "${MULTISCALE_SUBJET_PART_DISABLE_TOKEN_SCALE_EMBEDDING}"
fresh_append_flag_if_enabled cmd --disable-subjet-pair-bias "${MULTISCALE_SUBJET_PART_DISABLE_SUBJET_PAIR_BIAS}"
fresh_append_flag_if_enabled cmd --disable-scale-pair-embedding "${MULTISCALE_SUBJET_PART_DISABLE_SCALE_PAIR_EMBEDDING}"
fresh_append_optional_arg cmd --max-train-batches "${MULTISCALE_SUBJET_PART_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${MULTISCALE_SUBJET_PART_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-stack-val-batches "${MULTISCALE_SUBJET_PART_MAX_STACK_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${MULTISCALE_SUBJET_PART_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "multiscale_subjet_part_${REQUESTED_PROFILE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/summary_metrics.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/best_metrics.csv"
fi

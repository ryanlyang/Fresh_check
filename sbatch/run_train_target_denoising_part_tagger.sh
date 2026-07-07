#!/usr/bin/env bash
# Train one target-conditioned denoising ParT tagger variant.

#SBATCH --job-name=tdenoise_tag
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
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

variant="${1:-${TARGET_DENOISING_PART_TAGGER_VARIANT:-}}"
if [[ -z "${variant}" ]]; then
  echo "Usage: sbatch run_train_target_denoising_part_tagger.sh <variant>" >&2
  exit 2
fi

: "${TARGET_DENOISING_PART_ROOT:=${OUTPUT_ROOT}/target_conditioned_denoising_part_hltv2}"
: "${TARGET_DENOISING_PART_TAGGER_ROOT:=${TARGET_DENOISING_PART_ROOT}/taggers}"
: "${TARGET_DENOISING_PART_OUTPUT_DIR:=${TARGET_DENOISING_PART_TAGGER_ROOT}/${variant}}"
: "${TARGET_DENOISING_PART_MANIFEST_PATH:=${MANIFEST_PATH}}"
: "${TARGET_DENOISING_PART_HLT_CACHE_DIR:=${HLT_CACHE_DIR}}"
: "${TARGET_DENOISING_PART_DATA_DIR:=}"
: "${TARGET_DENOISING_PART_DENOISER_CHECKPOINT:=${TARGET_DENOISING_PART_ROOT}/denoisers/real/best_denoiser_model_val.pt}"
: "${TARGET_DENOISING_PART_SEED:=7307}"
: "${TARGET_DENOISING_PART_BATCH_SIZE:=64}"
: "${TARGET_DENOISING_PART_EVAL_BATCH_SIZE:=128}"
: "${TARGET_DENOISING_PART_TAGGER_EPOCHS:=35}"
: "${TARGET_DENOISING_PART_TAGGER_LR:=0.0003}"
: "${TARGET_DENOISING_PART_TAGGER_WEIGHT_DECAY:=0.0001}"
: "${TARGET_DENOISING_PART_NUM_WORKERS:=${NUM_WORKERS}}"
: "${TARGET_DENOISING_PART_DEVICE:=${DEVICE}}"
: "${TARGET_DENOISING_PART_GRAD_CLIP_NORM:=${GRAD_CLIP_NORM}}"
: "${TARGET_DENOISING_PART_TAGGER_EARLY_STOP_PATIENCE:=6}"
: "${TARGET_DENOISING_PART_MODEL_TRAIN_SIZE:=500000}"
: "${TARGET_DENOISING_PART_MODEL_VAL_SIZE:=150000}"
: "${TARGET_DENOISING_PART_FINAL_TEST_SIZE:=500000}"
: "${TARGET_DENOISING_PART_MAX_TRAIN_BATCHES:=}"
: "${TARGET_DENOISING_PART_MAX_VAL_BATCHES:=}"
: "${TARGET_DENOISING_PART_MAX_FINAL_TEST_BATCHES:=}"
: "${TARGET_DENOISING_PART_EVALUATE_FINAL_TEST:=1}"
: "${TARGET_DENOISING_PART_CONFIRM_FINAL_TEST:=1}"
: "${TARGET_DENOISING_PART_SELECTION_METRIC:=accuracy}"
: "${TARGET_DENOISING_PART_HLT_PROFILE:=fixed_hlt_v2_realistic}"
: "${TARGET_DENOISING_PART_HLT_PROFILE_VERSION:=v1}"
: "${TARGET_DENOISING_PART_HLT_DEGRADATION_STRENGTH:=1.0}"
: "${TARGET_DENOISING_PART_NO_AMP:=0}"
: "${TARGET_DENOISING_PART_COMPILE_MODEL:=0}"
: "${TARGET_DENOISING_PART_SKIP_HLT_HASH_CHECK:=0}"
: "${TARGET_DENOISING_PART_ALLOW_HLT_METADATA_MISMATCH:=0}"
: "${TARGET_DENOISING_PART_ALLOW_MANIFEST_MISMATCH:=0}"
: "${TARGET_DENOISING_PART_ALLOW_JET_IDENTITY_MISMATCH:=0}"
: "${TARGET_DENOISING_PART_NUM_CLASSES:=10}"
: "${TARGET_DENOISING_PART_MODEL_SIZE:=base}"
: "${TARGET_DENOISING_PART_PART_EMBED_DIM:=128}"
: "${TARGET_DENOISING_PART_MAX_CONSTITS:=128}"
: "${TARGET_DENOISING_PART_WEIGHT_THRESHOLD:=0.0}"
: "${TARGET_DENOISING_PART_ADAPTER_HIDDEN_DIM:=128}"
: "${TARGET_DENOISING_PART_ADAPTER_DROPOUT:=0.0}"
: "${TARGET_DENOISING_PART_ADAPTER_GATE_BIAS_INIT:=-2.0}"
: "${TARGET_DENOISING_PART_FREEZE_DENOISER:=0}"
: "${TARGET_DENOISING_PART_TRAIN_DENOISER:=0}"
: "${TARGET_DENOISING_PART_ALLOW_MISSING_DENOISER_CHECKPOINT:=0}"
: "${TARGET_DENOISING_PART_NON_STRICT_DENOISER_CHECKPOINT:=0}"
: "${TARGET_DENOISING_PART_ALLOW_INCOMPATIBLE_DENOISER_CHECKPOINT:=0}"
: "${TARGET_DENOISING_PART_RECONSTRUCTION_ANCHOR_WEIGHT:=}"
: "${TARGET_DENOISING_PART_RECONSTRUCTION_ANCHOR_SMOOTH_L1_BETA:=1.0}"
: "${TARGET_DENOISING_PART_ALIGNMENT_MODE:=aligned_direct}"
: "${TARGET_DENOISING_PART_EMBED_DIM:=64}"
: "${TARGET_DENOISING_PART_NUM_HEADS:=4}"
: "${TARGET_DENOISING_PART_PAIR_HIDDEN_DIM:=64}"
: "${TARGET_DENOISING_PART_HEAD_HIDDEN_DIM:=128}"
: "${TARGET_DENOISING_PART_MLP_RATIO:=2.0}"
: "${TARGET_DENOISING_PART_DROPOUT:=0.0}"
: "${TARGET_DENOISING_PART_ATTENTION_DROPOUT:=0.0}"
: "${TARGET_DENOISING_PART_DISABLE_PAIR_BIAS:=0}"
: "${TARGET_DENOISING_PART_DISABLE_LOCAL_KERNEL:=0}"
: "${TARGET_DENOISING_PART_LOCAL_KERNEL_RADIUS:=0.12}"
: "${TARGET_DENOISING_PART_LOCAL_KERNEL_INIT:=0.0}"
: "${TARGET_DENOISING_PART_PAIR_BIAS_MAX_ABS:=4.0}"
: "${TARGET_DENOISING_PART_MAX_DELTA_LOG_PT:=0.30}"
: "${TARGET_DENOISING_PART_MAX_DELTA_ETA:=0.08}"
: "${TARGET_DENOISING_PART_MAX_DELTA_PHI:=0.08}"
: "${TARGET_DENOISING_PART_MAX_DELTA_LOG_ENERGY:=0.30}"

if [[ -z "${TARGET_DENOISING_PART_RECONSTRUCTION_ANCHOR_WEIGHT}" ]]; then
  case "${variant}" in
    denoiser_features_joint)
      TARGET_DENOISING_PART_RECONSTRUCTION_ANCHOR_WEIGHT="0.05"
      ;;
    *)
      TARGET_DENOISING_PART_RECONSTRUCTION_ANCHOR_WEIGHT="0.0"
      ;;
  esac
fi

fresh_setup "$@"
fresh_require_file "scripts/train_target_denoising_part_tagger.py"
fresh_require_file "${TARGET_DENOISING_PART_MANIFEST_PATH}"
for split in model_train model_val; do
  fresh_require_file "${TARGET_DENOISING_PART_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${TARGET_DENOISING_PART_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
if fresh_bool_enabled "${TARGET_DENOISING_PART_EVALUATE_FINAL_TEST}"; then
  fresh_require_file "${TARGET_DENOISING_PART_HLT_CACHE_DIR}/final_test_fixed_hlt.npz"
  fresh_require_file "${TARGET_DENOISING_PART_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fi

case "${variant}" in
  denoiser_features_frozen|denoiser_features_joint|denoiser_shuffled_targets|denoiser_no_pair_bias|denoiser_local_kernel_only)
    if ! fresh_bool_enabled "${TARGET_DENOISING_PART_ALLOW_MISSING_DENOISER_CHECKPOINT}"; then
      fresh_require_file "${TARGET_DENOISING_PART_DENOISER_CHECKPOINT}"
    fi
    ;;
esac

fresh_claim_new_dir "${TARGET_DENOISING_PART_OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_target_denoising_part_tagger.py"
  --output-dir "${TARGET_DENOISING_PART_OUTPUT_DIR}"
  --manifest-path "${TARGET_DENOISING_PART_MANIFEST_PATH}"
  --hlt-cache-dir "${TARGET_DENOISING_PART_HLT_CACHE_DIR}"
  --denoiser-checkpoint "${TARGET_DENOISING_PART_DENOISER_CHECKPOINT}"
  --variant "${variant}"
  --train-split model_train
  --val-split model_val
  --final-test-split final_test
  --seed "${TARGET_DENOISING_PART_SEED}"
  --batch-size "${TARGET_DENOISING_PART_BATCH_SIZE}"
  --eval-batch-size "${TARGET_DENOISING_PART_EVAL_BATCH_SIZE}"
  --epochs "${TARGET_DENOISING_PART_TAGGER_EPOCHS}"
  --lr "${TARGET_DENOISING_PART_TAGGER_LR}"
  --weight-decay "${TARGET_DENOISING_PART_TAGGER_WEIGHT_DECAY}"
  --num-workers "${TARGET_DENOISING_PART_NUM_WORKERS}"
  --device "${TARGET_DENOISING_PART_DEVICE}"
  --grad-clip-norm "${TARGET_DENOISING_PART_GRAD_CLIP_NORM}"
  --early-stop-patience "${TARGET_DENOISING_PART_TAGGER_EARLY_STOP_PATIENCE}"
  --max-train-jets "${TARGET_DENOISING_PART_MODEL_TRAIN_SIZE}"
  --max-val-jets "${TARGET_DENOISING_PART_MODEL_VAL_SIZE}"
  --max-final-test-jets "${TARGET_DENOISING_PART_FINAL_TEST_SIZE}"
  --selection-metric "${TARGET_DENOISING_PART_SELECTION_METRIC}"
  --expected-hlt-profile "${TARGET_DENOISING_PART_HLT_PROFILE}"
  --expected-hlt-profile-version "${TARGET_DENOISING_PART_HLT_PROFILE_VERSION}"
  --expected-hlt-degradation-strength "${TARGET_DENOISING_PART_HLT_DEGRADATION_STRENGTH}"
  --reconstruction-anchor-weight "${TARGET_DENOISING_PART_RECONSTRUCTION_ANCHOR_WEIGHT}"
  --reconstruction-anchor-smooth-l1-beta "${TARGET_DENOISING_PART_RECONSTRUCTION_ANCHOR_SMOOTH_L1_BETA}"
  --alignment-mode "${TARGET_DENOISING_PART_ALIGNMENT_MODE}"
  --num-classes "${TARGET_DENOISING_PART_NUM_CLASSES}"
  --model-size "${TARGET_DENOISING_PART_MODEL_SIZE}"
  --part-embed-dim "${TARGET_DENOISING_PART_PART_EMBED_DIM}"
  --max-constits "${TARGET_DENOISING_PART_MAX_CONSTITS}"
  --weight-threshold "${TARGET_DENOISING_PART_WEIGHT_THRESHOLD}"
  --adapter-hidden-dim "${TARGET_DENOISING_PART_ADAPTER_HIDDEN_DIM}"
  --adapter-dropout "${TARGET_DENOISING_PART_ADAPTER_DROPOUT}"
  --adapter-gate-bias-init "${TARGET_DENOISING_PART_ADAPTER_GATE_BIAS_INIT}"
  --embed-dim "${TARGET_DENOISING_PART_EMBED_DIM}"
  --num-heads "${TARGET_DENOISING_PART_NUM_HEADS}"
  --pair-hidden-dim "${TARGET_DENOISING_PART_PAIR_HIDDEN_DIM}"
  --head-hidden-dim "${TARGET_DENOISING_PART_HEAD_HIDDEN_DIM}"
  --mlp-ratio "${TARGET_DENOISING_PART_MLP_RATIO}"
  --dropout "${TARGET_DENOISING_PART_DROPOUT}"
  --attention-dropout "${TARGET_DENOISING_PART_ATTENTION_DROPOUT}"
  --local-kernel-radius "${TARGET_DENOISING_PART_LOCAL_KERNEL_RADIUS}"
  --local-kernel-init "${TARGET_DENOISING_PART_LOCAL_KERNEL_INIT}"
  --pair-bias-max-abs "${TARGET_DENOISING_PART_PAIR_BIAS_MAX_ABS}"
  --max-delta-log-pt "${TARGET_DENOISING_PART_MAX_DELTA_LOG_PT}"
  --max-delta-eta "${TARGET_DENOISING_PART_MAX_DELTA_ETA}"
  --max-delta-phi "${TARGET_DENOISING_PART_MAX_DELTA_PHI}"
  --max-delta-log-energy "${TARGET_DENOISING_PART_MAX_DELTA_LOG_ENERGY}"
)
fresh_append_optional_arg cmd --data-dir "${TARGET_DENOISING_PART_DATA_DIR}"
fresh_append_flag_if_enabled cmd --no-amp "${TARGET_DENOISING_PART_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${TARGET_DENOISING_PART_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${TARGET_DENOISING_PART_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --allow-hlt-metadata-mismatch "${TARGET_DENOISING_PART_ALLOW_HLT_METADATA_MISMATCH}"
fresh_append_flag_if_enabled cmd --allow-manifest-mismatch "${TARGET_DENOISING_PART_ALLOW_MANIFEST_MISMATCH}"
fresh_append_flag_if_enabled cmd --allow-jet-identity-mismatch "${TARGET_DENOISING_PART_ALLOW_JET_IDENTITY_MISMATCH}"
fresh_append_flag_if_enabled cmd --freeze-denoiser "${TARGET_DENOISING_PART_FREEZE_DENOISER}"
fresh_append_flag_if_enabled cmd --train-denoiser "${TARGET_DENOISING_PART_TRAIN_DENOISER}"
fresh_append_flag_if_enabled cmd --allow-missing-denoiser-checkpoint "${TARGET_DENOISING_PART_ALLOW_MISSING_DENOISER_CHECKPOINT}"
fresh_append_flag_if_enabled cmd --non-strict-denoiser-checkpoint "${TARGET_DENOISING_PART_NON_STRICT_DENOISER_CHECKPOINT}"
fresh_append_flag_if_enabled cmd --allow-incompatible-denoiser-checkpoint "${TARGET_DENOISING_PART_ALLOW_INCOMPATIBLE_DENOISER_CHECKPOINT}"
fresh_append_flag_if_enabled cmd --disable-pair-bias "${TARGET_DENOISING_PART_DISABLE_PAIR_BIAS}"
fresh_append_flag_if_enabled cmd --disable-local-kernel "${TARGET_DENOISING_PART_DISABLE_LOCAL_KERNEL}"
fresh_append_flag_if_enabled cmd --evaluate-final-test "${TARGET_DENOISING_PART_EVALUATE_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${TARGET_DENOISING_PART_CONFIRM_FINAL_TEST}"
fresh_append_optional_arg cmd --max-train-batches "${TARGET_DENOISING_PART_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${TARGET_DENOISING_PART_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${TARGET_DENOISING_PART_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${TARGET_DENOISING_PART_OUTPUT_DIR}" "target_denoising_part_tagger_${variant}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${TARGET_DENOISING_PART_OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${TARGET_DENOISING_PART_OUTPUT_DIR}/last.pt"
  fresh_require_file "${TARGET_DENOISING_PART_OUTPUT_DIR}/config.json"
  fresh_require_file "${TARGET_DENOISING_PART_OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${TARGET_DENOISING_PART_OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${TARGET_DENOISING_PART_OUTPUT_DIR}/run_report.json"
  fresh_require_file "${TARGET_DENOISING_PART_OUTPUT_DIR}/diagnostics/epoch_metrics.csv"
  if fresh_bool_enabled "${TARGET_DENOISING_PART_EVALUATE_FINAL_TEST}"; then
    fresh_require_file "${TARGET_DENOISING_PART_OUTPUT_DIR}/final_test_report.json"
  fi
fi

#!/usr/bin/env bash
# Collect Architecture-View Residual ParT predictions and run stacked fusion.

#SBATCH --job-name=archview_fuse
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=23:00:00
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

: "${ARCHITECTURE_VIEW_PART_ROOT:=${OUTPUT_ROOT}/architecture_view_part_10class_hlt0p6}"
: "${ARCHITECTURE_VIEW_PART_TAGGER_ROOT:=${ARCHITECTURE_VIEW_PART_ROOT}/taggers}"
: "${ARCHITECTURE_VIEW_PART_FUSION_DIR:=${ARCHITECTURE_VIEW_PART_ROOT}/fusion_run}"
: "${ARCHITECTURE_VIEW_PART_HLT_CACHE_DIR:=${ARCHITECTURE_VIEW_PART_ROOT}/inputs/hlt_cache}"
: "${ARCHITECTURE_VIEW_PART_FUSION_VARIANTS:=av_baseline_recheck av_context_mlp_control av_pn_only av_pfn_only av_pcnn_only av_all_views}"
: "${ARCHITECTURE_VIEW_PART_FUSION_BATCH_SIZE:=128}"
: "${ARCHITECTURE_VIEW_PART_FUSION_NUM_WORKERS:=4}"
: "${ARCHITECTURE_VIEW_PART_FUSION_DEVICE:=${DEVICE}}"
: "${ARCHITECTURE_VIEW_PART_FUSION_STACK_TRAIN_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_PART_FUSION_STACK_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_PART_FUSION_FINAL_TEST_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_PART_FUSION_FEATURE_MODES:=logits probs logits_probs}"
: "${ARCHITECTURE_VIEW_PART_FUSION_C_GRID:=}"
: "${ARCHITECTURE_VIEW_PART_FUSION_MAX_ITER:=2000}"
: "${ARCHITECTURE_VIEW_PART_FUSION_SKIP_CONTROLS:=0}"
: "${ARCHITECTURE_VIEW_PART_FUSION_CONTROL_SEED:=12345}"
: "${ARCHITECTURE_VIEW_PART_FUSION_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/run_architecture_view_part_fusion.py"
for split in stack_train stack_val final_test; do
  fresh_require_file "${ARCHITECTURE_VIEW_PART_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_split_words variant_args "${ARCHITECTURE_VIEW_PART_FUSION_VARIANTS}"
for variant in "${variant_args[@]}"; do
  fresh_require_file "${ARCHITECTURE_VIEW_PART_TAGGER_ROOT}/${variant}/best_model_val.pt"
done
fresh_claim_new_dir "${ARCHITECTURE_VIEW_PART_FUSION_DIR}"
fresh_split_words feature_mode_args "${ARCHITECTURE_VIEW_PART_FUSION_FEATURE_MODES}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_architecture_view_part_fusion.py"
  --cache-dir "${ARCHITECTURE_VIEW_PART_HLT_CACHE_DIR}"
  --checkpoint-root "${ARCHITECTURE_VIEW_PART_TAGGER_ROOT}"
  --output-dir "${ARCHITECTURE_VIEW_PART_FUSION_DIR}"
  --variants "${variant_args[@]}"
  --batch-size "${ARCHITECTURE_VIEW_PART_FUSION_BATCH_SIZE}"
  --num-workers "${ARCHITECTURE_VIEW_PART_FUSION_NUM_WORKERS}"
  --device "${ARCHITECTURE_VIEW_PART_FUSION_DEVICE}"
  --stack-train-size "${ARCHITECTURE_VIEW_PART_FUSION_STACK_TRAIN_SIZE}"
  --stack-val-size "${ARCHITECTURE_VIEW_PART_FUSION_STACK_VAL_SIZE}"
  --final-test-size "${ARCHITECTURE_VIEW_PART_FUSION_FINAL_TEST_SIZE}"
  --feature-modes "${feature_mode_args[@]}"
  --max-iter "${ARCHITECTURE_VIEW_PART_FUSION_MAX_ITER}"
  --control-seed "${ARCHITECTURE_VIEW_PART_FUSION_CONTROL_SEED}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${ARCHITECTURE_VIEW_PART_FUSION_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --skip-controls "${ARCHITECTURE_VIEW_PART_FUSION_SKIP_CONTROLS}"
fresh_append_flag_if_enabled cmd --overwrite-predictions "${OVERWRITE}"
if [[ -n "${ARCHITECTURE_VIEW_PART_FUSION_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${ARCHITECTURE_VIEW_PART_FUSION_C_GRID}"
  cmd+=(--c-grid "${c_grid_args[@]}")
fi

fresh_write_run_config "${ARCHITECTURE_VIEW_PART_FUSION_DIR}" "fuse_architecture_view_part" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for variant in "${variant_args[@]}"; do
    fresh_require_file "${ARCHITECTURE_VIEW_PART_FUSION_DIR}/predictions/${variant}/stack_train_predictions.npz"
    fresh_require_file "${ARCHITECTURE_VIEW_PART_FUSION_DIR}/predictions/${variant}/stack_val_predictions.npz"
    fresh_require_file "${ARCHITECTURE_VIEW_PART_FUSION_DIR}/predictions/${variant}/final_test_predictions.npz"
  done
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FUSION_DIR}/fusion/fusion_report.json"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FUSION_DIR}/fusion/group_fusion_metrics.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FUSION_DIR}/fusion/singleton_stacker_metrics.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FUSION_DIR}/fusion/controls.json"
fi

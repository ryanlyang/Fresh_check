#!/usr/bin/env bash
# Collect frozen predictions and fuse the heterogeneous fixed-HLT ensemble.

#SBATCH --job-name=hhlt_fuse
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=05:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${FUSION_BATCH_SIZE:=128}"
: "${FUSION_NUM_WORKERS:=4}"
: "${FUSION_DEVICE:=${DEVICE}}"

fresh_setup "$@"
fresh_require_file "${HLT_CACHE_DIR}/stack_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/stack_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"

fresh_split_words arch_args "${HETERO_HLT4_ARCHITECTURES}"
for architecture in "${arch_args[@]}"; do
  fresh_require_file "${HETERO_HLT4_MODEL_ROOT}/${architecture}/best_model_val.pt"
done
fresh_claim_new_dir "${HETERO_HLT4_FUSION_DIR}"
fresh_split_words feature_mode_args "${HETERO_HLT4_FEATURE_MODES}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_heterogeneous_hlt_fusion.py"
  --cache-dir "${HLT_CACHE_DIR}"
  --checkpoint-root "${HETERO_HLT4_MODEL_ROOT}"
  --output-dir "${HETERO_HLT4_FUSION_DIR}"
  --architectures "${arch_args[@]}"
  --batch-size "${FUSION_BATCH_SIZE}"
  --num-workers "${FUSION_NUM_WORKERS}"
  --device "${FUSION_DEVICE}"
  --stack-train-size "${HETERO_HLT4_STACK_TRAIN_SIZE}"
  --stack-val-size "${HETERO_HLT4_STACK_VAL_SIZE}"
  --final-test-size "${HETERO_HLT4_FINAL_TEST_SIZE}"
  --feature-modes "${feature_mode_args[@]}"
  --max-iter "${HETERO_HLT4_MAX_ITER}"
  --control-seed "${HETERO_HLT4_CONTROL_SEED}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --overwrite-predictions "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --skip-controls "${HETERO_HLT4_SKIP_CONTROLS}"
if [[ -n "${HETERO_HLT4_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${HETERO_HLT4_C_GRID}"
  cmd+=(--c-grid "${c_grid_args[@]}")
fi

fresh_write_run_config "${HETERO_HLT4_FUSION_DIR}" "fuse_heterogeneous_hlt4" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for architecture in "${arch_args[@]}"; do
    case "${architecture}" in
      part) model_name="hlt_part" ;;
      pn) model_name="hlt_pn" ;;
      pfn) model_name="hlt_pfn" ;;
      pcnn) model_name="hlt_pcnn" ;;
      *)
        echo "Unknown heterogeneous HLT architecture in output check: ${architecture}" >&2
        exit 2
        ;;
    esac
    fresh_require_file "${HETERO_HLT4_FUSION_DIR}/predictions/${model_name}/stack_train_predictions.npz"
    fresh_require_file "${HETERO_HLT4_FUSION_DIR}/predictions/${model_name}/stack_val_predictions.npz"
    fresh_require_file "${HETERO_HLT4_FUSION_DIR}/predictions/${model_name}/final_test_predictions.npz"
  done
  fresh_require_file "${HETERO_HLT4_FUSION_DIR}/fusion/fusion_report.json"
  fresh_require_file "${HETERO_HLT4_FUSION_DIR}/fusion/group_fusion_metrics.csv"
  fresh_require_file "${HETERO_HLT4_FUSION_DIR}/fusion/singleton_stacker_metrics.csv"
  fresh_require_file "${HETERO_HLT4_FUSION_DIR}/fusion/controls.json"
fi

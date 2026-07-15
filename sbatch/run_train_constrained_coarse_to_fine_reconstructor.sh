#!/usr/bin/env bash
# Train one B/C constrained coarse-to-fine reconstructor.

#SBATCH --job-name=c2f_reco
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

RUN_ID="${1:?Usage: sbatch run_train_constrained_coarse_to_fine_reconstructor.sh <B0-B7|C0-C6|C5-B1|C5-B2|C5-B3|C5-no-slot|Cdirect-unconstrained>}"
variant="${RUN_ID}"
slot_loss_weight="1.0"
unconstrained_slot_accounting=0
direct_particle_decoding=0
hierarchy_loss_weight="1.0"
case "${RUN_ID}" in
  B[0-7]|C[0-6]|C5-B1|C5-B2|C5-B3) ;;
  C5-no-slot) variant=C5; slot_loss_weight="0.0" ;;
  C5-unconstrained|Cdirect-unconstrained)
    variant=C5
    unconstrained_slot_accounting=1
    direct_particle_decoding=1
    hierarchy_loss_weight="0.0"
    ;;
  *) echo "Unsupported constrained coarse-to-fine reconstructor RUN_ID: ${RUN_ID}" >&2; exit 2 ;;
esac

: "${CONSTRAINED_C2F_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine}"
: "${CONSTRAINED_C2F_MANIFEST_PATH:=${CONSTRAINED_C2F_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${CONSTRAINED_C2F_HLT_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/inputs/hlt_cache}"
: "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/inputs/offline_cache}"
: "${CONSTRAINED_C2F_TARGET_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/targets}"
: "${CONSTRAINED_C2F_RECON_ROOT:=${CONSTRAINED_C2F_ROOT}/reconstructors}"
: "${CONSTRAINED_C2F_RECO_SEED:=22031}"
: "${CONSTRAINED_C2F_RECO_EPOCHS:=30}"
: "${CONSTRAINED_C2F_RECO_BATCH_SIZE:=64}"
: "${CONSTRAINED_C2F_RECO_EVAL_BATCH_SIZE:=128}"
: "${CONSTRAINED_C2F_RECO_NUM_WORKERS:=4}"
: "${CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT:=0}"
: "${CONSTRAINED_C2F_RECO_MAX_TRAIN_JETS:=}"
: "${CONSTRAINED_C2F_RECO_MAX_VAL_JETS:=}"
: "${CONSTRAINED_C2F_RECO_MAX_STACK_VAL_JETS:=}"
: "${CONSTRAINED_C2F_RECO_AMP:=0}"
: "${CONSTRAINED_C2F_TORCH_NATIVE_TRITON:=auto}"
: "${CONSTRAINED_C2F_TORCH_NATIVE_TRITON_PROBE:=1}"
export CONSTRAINED_C2F_TORCH_NATIVE_TRITON CONSTRAINED_C2F_TORCH_NATIVE_TRITON_PROBE

OUTPUT_DIR="${CONSTRAINED_C2F_RECON_ROOT}/${RUN_ID}"
fresh_setup "$@"
fresh_require_file "${CONSTRAINED_C2F_MANIFEST_PATH}"
fresh_require_dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
fresh_require_dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
fresh_require_file "${CONSTRAINED_C2F_TARGET_CACHE_DIR}/hierarchy_target_cache_manifest.json"
fresh_claim_new_dir "${OUTPUT_DIR}"

memory_cmd=(
  "${PYTHON_BIN}" scripts/audit_constrained_coarse_to_fine_memory.py
  --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
  --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"
  --splits model_train model_val
  --allocated-memory-mb "${SLURM_MEM_PER_NODE:-0}"
  --output "${OUTPUT_DIR}/memory_preflight.json"
)
fresh_run "${memory_cmd[@]}"

cmd=(
  "${PYTHON_BIN}" -u scripts/train_constrained_coarse_to_fine.py
  --output-dir "${OUTPUT_DIR}"
  --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}"
  --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
  --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"
  --variant "${variant}"
  --hierarchy-loss-weight "${hierarchy_loss_weight}"
  --slot-loss-weight "${slot_loss_weight}"
  --stack-val-split stack_val
  --seed "${CONSTRAINED_C2F_RECO_SEED}"
  --epochs "${CONSTRAINED_C2F_RECO_EPOCHS}"
  --batch-size "${CONSTRAINED_C2F_RECO_BATCH_SIZE}"
  --eval-batch-size "${CONSTRAINED_C2F_RECO_EVAL_BATCH_SIZE}"
  --num-workers "${CONSTRAINED_C2F_RECO_NUM_WORKERS}"
  --device "${DEVICE}"
)
if fresh_bool_enabled "${unconstrained_slot_accounting}"; then cmd+=(--unconstrained-slot-accounting); fi
if fresh_bool_enabled "${direct_particle_decoding}"; then cmd+=(--direct-particle-decoding); fi
if fresh_bool_enabled "${CONSTRAINED_C2F_RECO_AMP}"; then cmd+=(--amp); else cmd+=(--no-amp); fi
if ! fresh_bool_enabled "${CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT}"; then cmd+=(--no-save-last-checkpoint); fi
if [[ -n "${CONSTRAINED_C2F_RECO_MAX_TRAIN_JETS}" ]]; then cmd+=(--max-train-jets "${CONSTRAINED_C2F_RECO_MAX_TRAIN_JETS}"); fi
if [[ -n "${CONSTRAINED_C2F_RECO_MAX_VAL_JETS}" ]]; then cmd+=(--max-val-jets "${CONSTRAINED_C2F_RECO_MAX_VAL_JETS}"); fi
if [[ -n "${CONSTRAINED_C2F_RECO_MAX_STACK_VAL_JETS}" ]]; then cmd+=(--max-stack-val-jets "${CONSTRAINED_C2F_RECO_MAX_STACK_VAL_JETS}"); fi

fresh_write_run_config "${OUTPUT_DIR}" "constrained_c2f_reconstructor_${RUN_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi

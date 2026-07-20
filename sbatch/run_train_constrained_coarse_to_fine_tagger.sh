#!/usr/bin/env bash
# Train one frozen-control or staged fusion tagger.

#SBATCH --job-name=c2f_tagger
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

RUN_ID="${1:?Usage: sbatch run_train_constrained_coarse_to_fine_tagger.sh <A0/A1/A2/A4|D0-D8|D5-B1/B2/B3|E0-E6|D*-seedN>}"
variant="${RUN_ID}"
seed_offset=0
case "${RUN_ID}" in
  A0|A1|A2|A4|D[0-8]|D5-B1|D5-B2|D5-B3|E[0-6]) ;;
  *)
    if [[ "${RUN_ID}" =~ ^(D[0-8]|D5-B[12])-seed([12])$ ]]; then
      variant="${BASH_REMATCH[1]}"
      seed_offset="$((101 * BASH_REMATCH[2]))"
    else
      echo "Unsupported staged tagger RUN_ID: ${RUN_ID}" >&2
      exit 2
    fi
    ;;
esac

: "${CONSTRAINED_C2F_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine}"
: "${CONSTRAINED_C2F_MANIFEST_PATH:=${CONSTRAINED_C2F_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${CONSTRAINED_C2F_HLT_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/inputs/hlt_cache}"
: "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/inputs/offline_cache}"
: "${CONSTRAINED_C2F_TARGET_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/targets}"
: "${CONSTRAINED_C2F_RECON_ROOT:=${CONSTRAINED_C2F_ROOT}/reconstructors}"
: "${CONSTRAINED_C2F_TAGGER_ROOT:=${CONSTRAINED_C2F_ROOT}/taggers}"
: "${CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT:=}"
: "${CONSTRAINED_C2F_TAGGER_SEED:=28031}"
: "${CONSTRAINED_C2F_TAGGER_EPOCHS:=12}"
: "${CONSTRAINED_C2F_TAGGER_BATCH_SIZE:=32}"
: "${CONSTRAINED_C2F_TAGGER_EVAL_BATCH_SIZE:=64}"
: "${CONSTRAINED_C2F_TAGGER_NUM_WORKERS:=4}"
: "${CONSTRAINED_C2F_TAGGER_SAVE_LAST_CHECKPOINT:=0}"
: "${CONSTRAINED_C2F_TAGGER_MAX_TRAIN_JETS:=}"
: "${CONSTRAINED_C2F_TAGGER_MAX_VAL_JETS:=}"
: "${CONSTRAINED_C2F_TAGGER_AMP:=0}"
: "${CONSTRAINED_C2F_TORCH_NATIVE_TRITON:=auto}"
: "${CONSTRAINED_C2F_TORCH_NATIVE_TRITON_PROBE:=1}"
export CONSTRAINED_C2F_TORCH_NATIVE_TRITON CONSTRAINED_C2F_TORCH_NATIVE_TRITON_PROBE

if [[ ! "${variant}" =~ ^(A0|A1|A2|D0)$ && -z "${CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT}" ]]; then
  echo "CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT is required for schedule-matched taggers." >&2
  exit 2
fi

OUTPUT_DIR="${CONSTRAINED_C2F_TAGGER_ROOT}/${RUN_ID}"
fresh_setup "$@"
fresh_require_file "${CONSTRAINED_C2F_MANIFEST_PATH}"
fresh_require_dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
fresh_require_dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
fresh_require_file "${CONSTRAINED_C2F_TARGET_CACHE_DIR}/hierarchy_target_cache_manifest.json"
if [[ ! "${variant}" =~ ^(A0|A1|A2|D0)$ ]]; then fresh_require_file "${CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT}"; fi
fresh_claim_new_dir "${OUTPUT_DIR}"

memory_cmd=(
  "${PYTHON_BIN}" scripts/audit_constrained_coarse_to_fine_memory.py
  --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
  --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"
  --splits model_train model_val
  --loading-mode sequential
  --allocated-memory-mb "${SLURM_MEM_PER_NODE:-0}"
  --output "${OUTPUT_DIR}/memory_preflight.json"
)
fresh_run "${memory_cmd[@]}"

source_args=()
variant_args=()
alias_args=()
case "${variant}" in
  A0|A1|A2|A4) ;;
  D0|D1|D2|D3|D4|E0|E1|E2|E3)
    source_args+=(--reconstructor-source "canonical=${CONSTRAINED_C2F_RECON_ROOT}/C5-B3/best_model_val.pt")
    variant_args+=(--reconstructor-variant canonical=C5-B3)
    ;;
  D5)
    source_args+=(--reconstructor-source "canonical=${CONSTRAINED_C2F_RECON_ROOT}/C5-B3/best_model_val.pt")
    variant_args+=(--reconstructor-variant canonical=C5-B3)
    ;;
  D5-B1)
    source_args+=(--reconstructor-source "c5_b1=${CONSTRAINED_C2F_RECON_ROOT}/C5-B1/best_model_val.pt")
    variant_args+=(--reconstructor-variant c5_b1=C5-B1)
    ;;
  D5-B2)
    source_args+=(--reconstructor-source "c5_b2=${CONSTRAINED_C2F_RECON_ROOT}/C5-B2/best_model_val.pt")
    variant_args+=(--reconstructor-variant c5_b2=C5-B2)
    ;;
  D5-B3)
    source_args+=(--reconstructor-source "c5_b3=${CONSTRAINED_C2F_RECON_ROOT}/C5-B3/best_model_val.pt")
    variant_args+=(--reconstructor-variant c5_b3=C5-B3)
    ;;
  D6)
    for index in 0 1 2 3; do
      source_args+=(--reconstructor-source "stochastic_${index}@${index}=${CONSTRAINED_C2F_RECON_ROOT}/C6/best_model_val.pt")
      variant_args+=(--reconstructor-variant "stochastic_${index}=C6")
    done
    ;;
  D7)
    source_args+=(--reconstructor-source "grid=${CONSTRAINED_C2F_RECON_ROOT}/C5-B3/best_model_val.pt")
    variant_args+=(--reconstructor-variant grid=C5-B3)
    ;;
  D8)
    source_args+=(
      --reconstructor-source "best_c=${CONSTRAINED_C2F_RECON_ROOT}/C5-B3/best_model_val.pt"
      --reconstructor-source "c5_b1=${CONSTRAINED_C2F_RECON_ROOT}/C5-B1/best_model_val.pt"
      --reconstructor-source "c5_b2=${CONSTRAINED_C2F_RECON_ROOT}/C5-B2/best_model_val.pt"
      --reconstructor-source "c5_b3=${CONSTRAINED_C2F_RECON_ROOT}/C5-B3/best_model_val.pt"
    )
    variant_args+=(
      --reconstructor-variant best_c=C5-B3
      --reconstructor-variant c5_b1=C5-B1
      --reconstructor-variant c5_b2=C5-B2
      --reconstructor-variant c5_b3=C5-B3
    )
    # best_c and c5_b3 are the same selected checkpoint. Declare that identity
    # so the resolver drops the duplicate view instead of double-counting it.
    alias_args+=(--reconstructor-alias best_c=c5_b3)
    ;;
  E5)
    source_args+=(--reconstructor-source "canonical=${CONSTRAINED_C2F_RECON_ROOT}/C5-no-slot/best_model_val.pt")
    # C5-no-slot retains C5's decoder implementation; its saved slot-config
    # variant is C5_uncertainty while its zero slot-loss is checked separately.
    variant_args+=(--reconstructor-variant canonical=C5_uncertainty)
    ;;
  E4)
    source_args+=(--reconstructor-source "canonical=${CONSTRAINED_C2F_RECON_ROOT}/Cdirect-unconstrained/best_model_val.pt")
    # The direct control changes accounting/decoding flags, not the saved C5
    # slot-decoder variant identity.
    variant_args+=(--reconstructor-variant canonical=C5_uncertainty)
    ;;
  E6) ;;
esac

for value in "${source_args[@]}"; do
  if [[ "${value}" == *=*.pt ]]; then fresh_require_file "${value#*=}"; fi
done

cmd=(
  "${PYTHON_BIN}" -u scripts/train_constrained_coarse_to_fine_end_to_end.py
  --output-dir "${OUTPUT_DIR}"
  --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}"
  --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
  --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"
  --variant "${variant}"
  "${source_args[@]}"
  "${variant_args[@]}"
  "${alias_args[@]}"
  --seed "$((CONSTRAINED_C2F_TAGGER_SEED + seed_offset))"
  --epochs "${CONSTRAINED_C2F_TAGGER_EPOCHS}"
  --batch-size "${CONSTRAINED_C2F_TAGGER_BATCH_SIZE}"
  --eval-batch-size "${CONSTRAINED_C2F_TAGGER_EVAL_BATCH_SIZE}"
  --num-workers "${CONSTRAINED_C2F_TAGGER_NUM_WORKERS}"
  --device "${DEVICE}"
)
if [[ "${variant}" =~ ^(A0|A1|A2)$ ]]; then
  cmd+=(--allow-random-hlt-start)
elif [[ "${variant}" != "D0" && -n "${CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT}" ]]; then
  cmd+=(--hlt-warm-start-checkpoint "${CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT}")
fi
if [[ "${variant}" == "A1" ]]; then
  cmd+=(--d-model 320 --num-heads 10 --hlt-encoder-layers 8 --fusion-layers 4)
fi
if fresh_bool_enabled "${CONSTRAINED_C2F_TAGGER_AMP}"; then cmd+=(--amp); else cmd+=(--no-amp); fi
if ! fresh_bool_enabled "${CONSTRAINED_C2F_TAGGER_SAVE_LAST_CHECKPOINT}"; then cmd+=(--no-save-last-checkpoint); fi
if [[ -n "${CONSTRAINED_C2F_TAGGER_MAX_TRAIN_JETS}" ]]; then cmd+=(--max-train-jets "${CONSTRAINED_C2F_TAGGER_MAX_TRAIN_JETS}"); fi
if [[ -n "${CONSTRAINED_C2F_TAGGER_MAX_VAL_JETS}" ]]; then cmd+=(--max-val-jets "${CONSTRAINED_C2F_TAGGER_MAX_VAL_JETS}"); fi

fresh_write_run_config "${OUTPUT_DIR}" "constrained_c2f_tagger_${RUN_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi

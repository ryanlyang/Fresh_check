#!/usr/bin/env bash
# Submit both Step 17 500k pilot and Step 18 high-data local-compression runs.
#
# This wrapper reuses existing QCD/Hgg HLT0.6 binary inputs and HLT caches. It
# does not build splits, labels, or fixed-HLT caches.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_local_compression_part_qcd_hgg_hlt0p6_experiment.sh"

: "${LOCAL_COMPRESSION_PART_QCD_HGG_VARIANTS:=hlt_part_baseline_recheck lc_mlp_delta lc_local_compression_no_context lc_context_gated lc_context_delta_no_modalities lc_random_grouping}"

: "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_INPUT_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_500k_full_gradfix2_20260629_031038}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_BASELINE_CHECKPOINT:=${LOCAL_COMPRESSION_PART_QCD_HGG_500K_INPUT_ROOT}/taggers/hlt_part_baseline/best_model_val.pt}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_TAG:=500k_pilot_$(date +%Y%m%d_%H%M%S)}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_TRAIN_TIME:=2-12:00:00}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_REPORT_TIME:=03:00:00}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_TRAIN_MEM:=160G}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_REPORT_MEM:=16G}"

: "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_INPUT_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_3m1m1m_full_gradfix2_20260629_031038}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_BASELINE_CHECKPOINT:=${LOCAL_COMPRESSION_PART_QCD_HGG_3M_INPUT_ROOT}/taggers/hlt_part_baseline/best_model_val.pt}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_TAG:=3m1m1m_highdata_$(date +%Y%m%d_%H%M%S)}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_TRAIN_TIME:=5-00:00:00}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_REPORT_TIME:=06:00:00}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_TRAIN_MEM:=180G}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_REPORT_MEM:=24G}"

: "${LOCAL_COMPRESSION_PART_QCD_HGG_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH:=0}"
: "${LOCAL_COMPRESSION_PART_SUBMIT_500K:=1}"
: "${LOCAL_COMPRESSION_PART_SUBMIT_3M:=1}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "required file is missing: ${path}" >&2
    exit 1
  fi
}

submit_local_compression_size() {
  local label="$1"
  local tag="$2"
  local input_root="$3"
  local baseline_checkpoint="$4"
  local train_size="$5"
  local val_size="$6"
  local stack_train_size="$7"
  local stack_val_size="$8"
  local final_test_size="$9"
  local train_time="${10}"
  local report_time="${11}"
  local train_mem="${12}"
  local report_mem="${13}"

  require_file "${SUBMIT_SCRIPT}"
  require_file "${input_root}/binary_inputs/split_manifest.json.gz"
  require_file "${baseline_checkpoint}"
  for split in model_train model_val stack_train stack_val final_test; do
    require_file "${input_root}/binary_inputs/hlt_cache/${split}_fixed_hlt.npz"
    require_file "${input_root}/binary_inputs/hlt_cache/${split}_fixed_hlt_metadata.json"
  done

  echo "submitting ${label}:"
  echo "  input_root=${input_root}"
  echo "  baseline_checkpoint=${baseline_checkpoint}"
  echo "  tag=${tag}"
  echo "  variants=${LOCAL_COMPRESSION_PART_QCD_HGG_VARIANTS}"

  LOCAL_COMPRESSION_PART_QCD_HGG_TAG="${tag}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_INPUT_ROOT="${input_root}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_BASELINE_CHECKPOINT="${baseline_checkpoint}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_VARIANTS="${LOCAL_COMPRESSION_PART_QCD_HGG_VARIANTS}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_MODEL_TRAIN_SIZE="${train_size}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_MODEL_VAL_SIZE="${val_size}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_STACK_TRAIN_SIZE="${stack_train_size}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_STACK_VAL_SIZE="${stack_val_size}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_FINAL_TEST_SIZE="${final_test_size}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_TIME="${train_time}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_TIME="${report_time}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_MEM="${train_mem}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_MEM="${report_mem}" \
  LOCAL_COMPRESSION_PART_QCD_HGG_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH="${LOCAL_COMPRESSION_PART_QCD_HGG_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH}" \
  bash "${SUBMIT_SCRIPT}"
}

if [[ "${LOCAL_COMPRESSION_PART_SUBMIT_500K}" == "1" ]]; then
  submit_local_compression_size \
    "Step 17 500k pilot" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_TAG}" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_INPUT_ROOT}" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_BASELINE_CHECKPOINT}" \
    500000 150000 500000 150000 500000 \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_TRAIN_TIME}" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_REPORT_TIME}" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_TRAIN_MEM}" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_500K_REPORT_MEM}"
fi

if [[ "${LOCAL_COMPRESSION_PART_SUBMIT_3M}" == "1" ]]; then
  submit_local_compression_size \
    "Step 18 3M/1M/1M high-data" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_TAG}" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_INPUT_ROOT}" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_BASELINE_CHECKPOINT}" \
    3000000 1000000 3000000 1000000 1000000 \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_TRAIN_TIME}" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_REPORT_TIME}" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_TRAIN_MEM}" \
    "${LOCAL_COMPRESSION_PART_QCD_HGG_3M_REPORT_MEM}"
fi

#!/usr/bin/env bash
# Write the aggregate HLT-MV final report.

#SBATCH --job-name=hlt_mv_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${HLT_MV_PDV3_EXPERIMENT_NAME:=privileged_distill_v3_av10_adapter_fixed_hlt_v2_realistic_s1p0_highdata_20260705_190747}"
: "${HLT_MV_PDV3_ROOT:=${OUTPUT_ROOT}/${HLT_MV_PDV3_EXPERIMENT_NAME}}"
: "${HLT_MV_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_multiview_source_fusion}"
: "${HLT_MV_SOURCE_MODELS_DIR:=${HLT_MV_ROOT}/source_models}"
: "${HLT_MV_RANDOM_HLT_CONTROLS_DIR:=${HLT_MV_ROOT}/hlt_random_seed_controls}"
: "${HLT_MV_LOGIT_FUSIONS_DIR:=${HLT_MV_ROOT}/logit_fusions}"
: "${HLT_MV_PRETRAINED_DUALVIEW_DIR:=${HLT_MV_ROOT}/particle_dualview_pretrained}"
: "${HLT_MV_SCRATCH_DUALVIEW_DIR:=${HLT_MV_ROOT}/particle_dualview_scratch}"
: "${HLT_MV_CONTROLS_DIR:=${HLT_MV_ROOT}/controls}"
: "${HLT_MV_TRIVIEW_DIR:=${HLT_MV_ROOT}/triview}"
: "${HLT_MV_FINAL_REPORT_DIR:=${HLT_MV_ROOT}/final_report}"
: "${HLT_MV_FINAL_REPORT_ALLOW_MISSING:=0}"
: "${HLT_MV_FINAL_REPORT_REQUIRE_TRIVIEW:=0}"

fresh_setup "$@"
if ! fresh_bool_enabled "${HLT_MV_FINAL_REPORT_ALLOW_MISSING}"; then
  for source_name in \
    hlt_part_seed8801 \
    hlt2_part_s0p10_seed8811 \
    hlt2_part_s0p20_seed8821 \
    hlt2_part_s0p35_seed8831 \
    hlt2_part_s1p00_seed8841; do
    fresh_require_file "${HLT_MV_SOURCE_MODELS_DIR}/${source_name}/run_report.json"
  done
  for source_name in hlt_part_seed9101 hlt_part_seed9102 hlt_part_seed9103 hlt_part_seed9104; do
    fresh_require_file "${HLT_MV_RANDOM_HLT_CONTROLS_DIR}/${source_name}/run_report.json"
  done
  for variant in sdv_hlt_hlt2_s0p10 sdv_hlt_hlt2_s0p20 sdv_hlt_hlt2_s0p35 sdv_hlt_hlt2_s1p00; do
    fresh_require_file "${HLT_MV_PRETRAINED_DUALVIEW_DIR}/${variant}/hlt_mv_pretrained_dualview_report.json"
  done
  for variant in sdv_hlt_hlt2_s0p10_scratch sdv_hlt_hlt2_s0p20_scratch sdv_hlt_hlt2_s0p35_scratch sdv_hlt_hlt2_s1p00_scratch; do
    fresh_require_file "${HLT_MV_SCRATCH_DUALVIEW_DIR}/${variant}/hlt_mv_scratch_dualview_report.json"
  done
  for control_name in \
    sdv_hlt_hlt_same_view \
    tta_hlt_part_hlt_plus_hlt2_s0p10 \
    tta_hlt_part_hlt_plus_hlt2_s0p20 \
    tta_hlt_part_hlt_plus_hlt2_s0p35 \
    tta_hlt_part_hlt_plus_hlt2_s1p00; do
    fresh_require_file "${HLT_MV_CONTROLS_DIR}/${control_name}/run_report.json"
  done
  if fresh_bool_enabled "${HLT_MV_FINAL_REPORT_REQUIRE_TRIVIEW}"; then
    fresh_require_file "${HLT_MV_TRIVIEW_DIR}/tri_hlt_hlt2_s0p35_s1p00/hlt_mv_triview_report.json"
  fi
  for fusion_name in source_5view hlt_random_4seed pretrained_dualview_4model scratch_dualview_4model; do
    fresh_require_file "${HLT_MV_LOGIT_FUSIONS_DIR}/${fusion_name}/run_report.json"
  done
fi
fresh_claim_new_dir "${HLT_MV_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_hlt_mv_final_report.py"
  --output-root "${OUTPUT_ROOT}"
  --pdv3-experiment-name "${HLT_MV_PDV3_EXPERIMENT_NAME}"
  --output-dir "${HLT_MV_FINAL_REPORT_DIR}"
)
fresh_append_flag_if_enabled cmd --allow-missing "${HLT_MV_FINAL_REPORT_ALLOW_MISSING}"
fresh_append_flag_if_enabled cmd --require-triview "${HLT_MV_FINAL_REPORT_REQUIRE_TRIVIEW}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${HLT_MV_FINAL_REPORT_DIR}" "hlt_mv_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${HLT_MV_FINAL_REPORT_DIR}/summary.json"
  fresh_require_file "${HLT_MV_FINAL_REPORT_DIR}/hlt_multiview_source_fusion_report.json"
  fresh_require_file "${HLT_MV_FINAL_REPORT_DIR}/hlt_multiview_source_fusion_report.md"
  fresh_require_file "${HLT_MV_FINAL_REPORT_DIR}/metric_table.csv"
  fresh_require_file "${HLT_MV_FINAL_REPORT_DIR}/run_report.json"
  fresh_assert_json_ok "${HLT_MV_FINAL_REPORT_DIR}/run_report.json"
fi

#!/usr/bin/env bash
# Write the target-conditioned denoising ParT summary report.

#SBATCH --job-name=tdenoise_rep
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${TARGET_DENOISING_PART_ROOT:=${OUTPUT_ROOT}/target_conditioned_denoising_part_hltv2}"
: "${TARGET_DENOISING_PART_TAGGER_ROOT:=${TARGET_DENOISING_PART_ROOT}/taggers}"
: "${TARGET_DENOISING_PART_REPORT_DIR:=${TARGET_DENOISING_PART_ROOT}/final_report}"
: "${TARGET_DENOISING_PART_DENOISER_REPORT:=${TARGET_DENOISING_PART_ROOT}/denoisers/real/run_report.json}"
: "${TARGET_DENOISING_PART_DENOISER_REPORTS:=${TARGET_DENOISING_PART_DENOISER_REPORT}}"
: "${TARGET_DENOISING_PART_HLT_BASELINE_REPORT:=}"
: "${TARGET_DENOISING_PART_OFFLINE_BASELINE_REPORT:=}"
: "${TARGET_DENOISING_PART_VARIANTS:=hlt_part_baseline feature_mlp_adapter_tag_only denoiser_features_frozen denoiser_features_joint denoiser_tag_only_same_arch denoiser_no_pair_bias}"
: "${TARGET_DENOISING_PART_REQUIRE_VARIANTS:=1}"

fresh_setup "$@"
fresh_require_file "scripts/write_target_conditioned_denoising_part_report.py"
fresh_split_words denoiser_reports "${TARGET_DENOISING_PART_DENOISER_REPORTS}"
for denoiser_report in "${denoiser_reports[@]}"; do
  fresh_require_file "${denoiser_report}"
done
fresh_split_words report_variants "${TARGET_DENOISING_PART_VARIANTS}"
if fresh_bool_enabled "${TARGET_DENOISING_PART_REQUIRE_VARIANTS}"; then
  for variant in "${report_variants[@]}"; do
    fresh_require_file "${TARGET_DENOISING_PART_TAGGER_ROOT}/${variant}/run_report.json"
  done
fi
fresh_claim_new_dir "${TARGET_DENOISING_PART_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_target_conditioned_denoising_part_report.py"
  --output-dir "${TARGET_DENOISING_PART_REPORT_DIR}"
  --tagger-root "${TARGET_DENOISING_PART_TAGGER_ROOT}"
  --variants "${report_variants[@]}"
)
for denoiser_report in "${denoiser_reports[@]}"; do
  cmd+=(--denoiser-report "${denoiser_report}")
done
fresh_append_optional_arg cmd --hlt-baseline-report "${TARGET_DENOISING_PART_HLT_BASELINE_REPORT}"
fresh_append_optional_arg cmd --offline-baseline-report "${TARGET_DENOISING_PART_OFFLINE_BASELINE_REPORT}"
fresh_append_flag_if_enabled cmd --require-variants "${TARGET_DENOISING_PART_REQUIRE_VARIANTS}"

fresh_write_run_config "${TARGET_DENOISING_PART_REPORT_DIR}" "target_denoising_part_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${TARGET_DENOISING_PART_REPORT_DIR}/summary.json"
  fresh_require_file "${TARGET_DENOISING_PART_REPORT_DIR}/tagger_metrics.csv"
  fresh_require_file "${TARGET_DENOISING_PART_REPORT_DIR}/denoising_metrics.csv"
  fresh_require_file "${TARGET_DENOISING_PART_REPORT_DIR}/adapter_attention_diagnostics.csv"
  fresh_require_file "${TARGET_DENOISING_PART_REPORT_DIR}/mechanism_ablation_metrics.csv"
fi

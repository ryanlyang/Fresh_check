#!/usr/bin/env bash
# Summarize all split cross-architecture fusion reports.

#SBATCH --job-name=crossarch_split_sum
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_file "scripts/summarize_crossarch_split_fusions.py"

fresh_split_words family_args "${CROSSARCH_SPLIT_FUSION_FAMILIES}"
fresh_split_words group_args "${CROSSARCH_SPLIT_FUSION_GROUPS}"
fresh_split_words bundle_args "${CROSSARCH_SPLIT_FUSION_BUNDLES}"

OUTPUT_DIR="${CROSSARCH_SPLIT_FUSION_ROOT}/summary"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/summarize_crossarch_split_fusions.py"
  --split-root "${CROSSARCH_SPLIT_FUSION_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --families "${family_args[@]}"
  --groups "${group_args[@]}"
  --bundles "${bundle_args[@]}"
)
fresh_append_flag_if_enabled cmd --require-expected "${CROSSARCH_SPLIT_FUSION_REQUIRE_EXPECTED_SUMMARY}"

fresh_write_run_config "${OUTPUT_DIR}" "crossarch_split_fusion_summary" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/crossarch_split_fusion_summary.json"
  fresh_require_file "${OUTPUT_DIR}/crossarch_split_fusion_summary.md"
  fresh_assert_json_ok "${OUTPUT_DIR}/crossarch_split_fusion_summary.json"
fi

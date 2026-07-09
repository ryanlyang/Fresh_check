#!/usr/bin/env bash
# TIGRIS/GH200 runner for the strong HLT-MV source/fusion grid.
#
# Grid: 2.00, 3.00, 4.00, 5.00.
# The 2.00 HLT2 cache is reused from the completed heavy-grid cache root by
# default; 3.00/4.00/5.00 are built into this run root before the graph starts.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${OUTPUT_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints}"
: "${HLT_MV_PDV3_EXPERIMENT_NAME:=privileged_distill_v3_av10_adapter_fixed_hlt_v2_realistic_s1p0_highdata_20260705_190747}"
: "${HLT_MV_PDV3_ROOT:=${OUTPUT_ROOT}/${HLT_MV_PDV3_EXPERIMENT_NAME}}"
: "${HLT_MV_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_multiview_source_fusion_strong_$(date +%Y%m%d_%H%M%S)}"
: "${HLT_MV_REUSE_HLT2_CACHE_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_multiview_source_fusion_heavy_20260708_185728/hlt2_cache}"

: "${HLT_MV_HEAVY_REUSE_STRENGTHS:=2.00}"
: "${HLT_MV_HEAVY_BUILD_STRENGTHS:=3.00 4.00 5.00}"
: "${HLT_MV_STRENGTHS:=2.00 3.00 4.00 5.00}"
: "${HLT_MV_TTA_STRENGTHS:=2.00 3.00 4.00 5.00}"
: "${HLT_MV_HLT2_SOURCE_SEEDS:=2.00=8871 3.00=8881 4.00=8891 5.00=8901}"
: "${HLT_MV_CANONICAL_HLT_SOURCE_NAME:=hlt_part_seed8801}"
: "${HLT_MV_SOURCE_NAMES:=hlt_part_seed8801 hlt2_part_s2p00_seed8871 hlt2_part_s3p00_seed8881 hlt2_part_s4p00_seed8891 hlt2_part_s5p00_seed8901}"
: "${HLT_MV_RANDOM_HLT_SOURCE_NAMES:=hlt_part_seed9101 hlt_part_seed9102 hlt_part_seed9103 hlt_part_seed9104}"
: "${HLT_MV_PRETRAINED_DUALVIEW_NAMES:=sdv_hlt_hlt2_s2p00 sdv_hlt_hlt2_s3p00 sdv_hlt_hlt2_s4p00 sdv_hlt_hlt2_s5p00}"
: "${HLT_MV_SCRATCH_DUALVIEW_NAMES:=sdv_hlt_hlt2_s2p00_scratch sdv_hlt_hlt2_s3p00_scratch sdv_hlt_hlt2_s4p00_scratch sdv_hlt_hlt2_s5p00_scratch}"
: "${HLT_MV_TRIVIEW_MODEL_NAME:=tri_hlt_hlt2_s4p00_s5p00}"

export PROJECT_DIR OUTPUT_ROOT HLT_MV_PDV3_EXPERIMENT_NAME HLT_MV_PDV3_ROOT HLT_MV_ROOT
export HLT_MV_REUSE_HLT2_CACHE_ROOT HLT_MV_HEAVY_REUSE_STRENGTHS HLT_MV_HEAVY_BUILD_STRENGTHS
export HLT_MV_STRENGTHS HLT_MV_TTA_STRENGTHS HLT_MV_HLT2_SOURCE_SEEDS
export HLT_MV_CANONICAL_HLT_SOURCE_NAME HLT_MV_SOURCE_NAMES HLT_MV_RANDOM_HLT_SOURCE_NAMES
export HLT_MV_PRETRAINED_DUALVIEW_NAMES HLT_MV_SCRATCH_DUALVIEW_NAMES HLT_MV_TRIVIEW_MODEL_NAME

exec bash "${PROJECT_DIR}/sbatch/submit_hlt_mv_heavy_source_fusion_tigris.sh"

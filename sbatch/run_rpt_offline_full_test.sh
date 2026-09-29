#!/usr/bin/env bash
#SBATCH --job-name=rpt_off20m
set -euo pipefail

: "${RPT_FULL_TEST_OUTPUT:?Set RPT_FULL_TEST_OUTPUT}"
: "${RPT_FULL_TEST_PHASE:?Set RPT_FULL_TEST_PHASE}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
driver="${RPT_FULL_TEST_OUTPUT}/driver.py"
cd "${RPT_FULL_TEST_OUTPUT}"

if [[ "${RPT_FULL_TEST_PHASE}" == prepare ]]; then
  python -u "${driver}" prepare \
    --parent "${RPT_FULL_TEST_PARENT}" --source-root "${RPT_FULL_TEST_SOURCE}" \
    --data-dir "${RPT_FULL_TEST_DATA}" --output "${RPT_FULL_TEST_OUTPUT}" \
    --jobs "${RPT_FULL_TEST_JOBS}"
else
  python -u "${driver}" "${RPT_FULL_TEST_PHASE}" --output "${RPT_FULL_TEST_OUTPUT}"
fi

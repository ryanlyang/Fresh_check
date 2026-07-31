#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="$(cd -- "${script_dir}/.." && pwd -P)"
cd "${project_dir}"
: "${CAMPAIGN_ROOT:?Export CAMPAIGN_ROOT before submission}"
python scripts/submit_hosd_slurm.py --campaign-root "${CAMPAIGN_ROOT}" "$@"

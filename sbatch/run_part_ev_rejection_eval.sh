#!/usr/bin/env bash
set -euo pipefail
# Slurm runs a spool-directory copy of this script. BASH_SOURCE is NOT the
# frozen source path. Locate the verified driver using the explicit --output.
worker_args=("$@")
output=
while (( $# )); do
  case "$1" in
    --output)
      (( $# >= 2 )) || { echo 'Missing --output value' >&2; exit 2; }
      [[ -z "$output" ]] || { echo 'Duplicate --output' >&2; exit 2; }
      output=$2
      shift 2 ;;
    *) shift ;;
  esac
done
[[ "$output" == /* ]] || { echo 'Use an absolute --output campaign path' >&2; exit 2; }
driver="$output/source/scripts/run_part_ev_rejection_eval.py"
[[ -f "$output/plan.json" && -f "$driver" ]] || {
  echo "Missing evaluation plan or frozen driver under $output" >&2
  exit 2
}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Worker re-execs the frozen donor Python; report needs only stdlib Python.
exec python3 -B "$driver" "${worker_args[@]}"

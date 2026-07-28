#!/usr/bin/env bash
# Shared, bounded-retry DDP launcher for ABPH Slurm workers.

abph_ddp_master_port() {
  local numeric_job_id="${1:?Missing numeric Slurm job id}"
  local attempt="${2:?Missing launch attempt}"
  local port_min="${ABPH_DDP_PORT_MIN:-20000}"
  local port_span="${ABPH_DDP_PORT_SPAN:-40000}"

  [[ "${numeric_job_id}" =~ ^[0-9]+$ ]] || {
    echo "Invalid numeric Slurm job id: ${numeric_job_id}" >&2
    return 2
  }
  [[ "${attempt}" =~ ^[0-9]+$ ]] || {
    echo "Invalid DDP launch attempt: ${attempt}" >&2
    return 2
  }
  [[ "${port_min}" =~ ^[0-9]+$ && "${port_span}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ABPH DDP port range values must be positive integers" >&2
    return 2
  }
  ((port_min >= 1024 && port_min <= 65535)) || {
    echo "ABPH_DDP_PORT_MIN must be between 1024 and 65535" >&2
    return 2
  }
  ((port_span > 0 && port_min + port_span - 1 <= 65535)) || {
    echo "ABPH_DDP_PORT_SPAN exceeds the TCP port range" >&2
    return 2
  }

  # Coprime-ish multipliers spread adjacent Slurm IDs and retries over the range.
  echo "$((port_min + (numeric_job_id * 7919 + attempt * 104729) % port_span))"
}

abph_fresh_run_srun_with_port_retry() {
  local raw_job_id="${SLURM_JOB_ID:-}"
  local numeric_job_id="${raw_job_id%%_*}"
  local max_attempts="${ABPH_DDP_PORT_RETRY_ATTEMPTS:-4}"
  local quick_failure_seconds="${ABPH_DDP_PORT_RETRY_QUICK_FAILURE_SECONDS:-60}"
  local retry_delay_seconds="${ABPH_DDP_PORT_RETRY_DELAY_SECONDS:-3}"
  local attempt=0
  local status=1
  local started=0
  local elapsed=0

  [[ "${numeric_job_id}" =~ ^[0-9]+$ ]] || {
    echo "Invalid Slurm job id for DDP launch: ${SLURM_JOB_ID:-unset}" >&2
    return 2
  }
  [[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ABPH_DDP_PORT_RETRY_ATTEMPTS must be positive" >&2
    return 2
  }
  [[ "${quick_failure_seconds}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ABPH_DDP_PORT_RETRY_QUICK_FAILURE_SECONDS must be positive" >&2
    return 2
  }
  [[ "${retry_delay_seconds}" =~ ^[0-9]+$ ]] || {
    echo "ABPH_DDP_PORT_RETRY_DELAY_SECONDS must be nonnegative" >&2
    return 2
  }

  while ((attempt < max_attempts)); do
    export MASTER_PORT
    MASTER_PORT="$(abph_ddp_master_port "${numeric_job_id}" "${attempt}")"
    echo "ABPH DDP launch attempt $((attempt + 1))/${max_attempts}: MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT}" >&2
    started="$(date +%s)"
    if fresh_run srun "$@"; then
      return 0
    else
      status=$?
    fi
    elapsed="$(( $(date +%s) - started ))"
    attempt="$((attempt + 1))"

    if ((attempt >= max_attempts || elapsed > quick_failure_seconds)); then
      echo "ABPH DDP launch failed with status ${status} after ${elapsed}s; no further port retry" >&2
      return "${status}"
    fi
    echo "ABPH DDP startup failed with status ${status} after ${elapsed}s; retrying with a different rendezvous port" >&2
    sleep "${retry_delay_seconds}"
  done

  return "${status}"
}

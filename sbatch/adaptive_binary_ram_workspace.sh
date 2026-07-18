#!/usr/bin/env bash
# Source-only helper: verify tmpfs, create one private rank workspace, clean it on exit.

abph_cleanup_ram_workspace() {
  local status="${1:-$?}"
  trap - EXIT HUP INT TERM
  if [[ -n "${ABPH_RAM_WORKSPACE:-}" && -d "${ABPH_RAM_WORKSPACE}" ]]; then
    "${PYTHON_BIN:-python}" -u scripts/manage_adaptive_binary_ram_workspace.py cleanup \
      --workspace "${ABPH_RAM_WORKSPACE}" \
      --job-id "${SLURM_JOB_ID:-local}" \
      --rank "${SLURM_PROCID:-${RANK:-0}}" || true
  fi
  return 0
}

abph_setup_ram_workspace() {
  export ABPH_RAM_WORKSPACE
  ABPH_RAM_WORKSPACE="$("${PYTHON_BIN:-python}" -u scripts/manage_adaptive_binary_ram_workspace.py create --path-only)"
  [[ -n "${ABPH_RAM_WORKSPACE}" && -d "${ABPH_RAM_WORKSPACE}" ]] || {
    echo "Failed to create the verified ABPH RAM workspace" >&2
    return 1
  }
  trap 'status=$?; abph_cleanup_ram_workspace "${status}"; exit "${status}"' EXIT
  trap 'status=129; abph_cleanup_ram_workspace "${status}"; exit "${status}"' HUP
  trap 'status=130; abph_cleanup_ram_workspace "${status}"; exit "${status}"' INT
  trap 'status=143; abph_cleanup_ram_workspace "${status}"; exit "${status}"' TERM
}

abph_reserve_ram_workspace() {
  local role="${1:?RAM reservation role is required}"
  local expected_bytes="${2:?RAM reservation byte count is required}"
  [[ -n "${ABPH_RAM_WORKSPACE:-}" ]] || {
    echo "RAM workspace must exist before reserving" >&2
    return 1
  }
  export ABPH_RAM_STAGE_RESERVATION_ID
  ABPH_RAM_STAGE_RESERVATION_ID="$("${PYTHON_BIN:-python}" -u scripts/manage_adaptive_binary_ram_workspace.py reserve \
    --workspace "${ABPH_RAM_WORKSPACE}" \
    --job-id "${SLURM_JOB_ID:-local}" \
    --rank "${SLURM_PROCID:-${RANK:-0}}" \
    --owner "${SLURM_JOB_ID:-local}" \
    --role "${role}" \
    --expected-bytes "${expected_bytes}" \
    --id-only)"
  [[ -n "${ABPH_RAM_STAGE_RESERVATION_ID}" ]] || {
    echo "Failed to reserve RAM workspace capacity" >&2
    return 1
  }
}

abph_commit_ram_workspace() {
  local measured_path="${1:?RAM reservation measured path is required}"
  [[ -n "${ABPH_RAM_STAGE_RESERVATION_ID:-}" ]] || {
    echo "No active RAM stage reservation to commit" >&2
    return 1
  }
  "${PYTHON_BIN:-python}" -u scripts/manage_adaptive_binary_ram_workspace.py commit \
    --workspace "${ABPH_RAM_WORKSPACE}" \
    --job-id "${SLURM_JOB_ID:-local}" \
    --rank "${SLURM_PROCID:-${RANK:-0}}" \
    --reservation-id "${ABPH_RAM_STAGE_RESERVATION_ID}" \
    --measured-path "${measured_path}"
}

abph_release_ram_workspace() {
  if [[ -z "${ABPH_RAM_STAGE_RESERVATION_ID:-}" ]]; then
    return 0
  fi
  "${PYTHON_BIN:-python}" -u scripts/manage_adaptive_binary_ram_workspace.py release \
    --workspace "${ABPH_RAM_WORKSPACE}" \
    --job-id "${SLURM_JOB_ID:-local}" \
    --rank "${SLURM_PROCID:-${RANK:-0}}" \
    --reservation-id "${ABPH_RAM_STAGE_RESERVATION_ID}"
  unset ABPH_RAM_STAGE_RESERVATION_ID
}

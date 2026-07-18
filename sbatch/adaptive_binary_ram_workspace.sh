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

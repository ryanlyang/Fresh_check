#!/usr/bin/env bash
# Run under WSL/Linux, not PowerShell or Git Bash. No training jobs are submitted.
# Stages one archive at a time; the helper removes only a verified staging tar
# after every extracted ROOT file has been verified and a receipt committed.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/upload_jetclass_train_val.sh [options]

  --source-dir DIR   Local archive directory (default: /mnt/c/Users/22rya/Downloads)
  --remote HOST      SSH destination (default: ryreu@tigris.rc.rit.edu)
  --remote-root DIR  Existing dataset root (default: /home/ryreu/atlas/datasets/JetClass)
  --only PART        all (default), part0 through part9, or val
  --headroom-gib N   Free-space reserve during extraction (default: 10)
  --dry-run         List inputs and sizes; no SSH, hashing, uploads, or writes
  --help            Show this help

Rerun the SAME command after interruption. Completed archives are skipped,
partial tar uploads resume with rsync, and extraction can also be retried.
Keep the Windows terminal/computer awake. A reconnect may require Duo again.
Only temporary REMOTE tar copies are removed, after verified extraction.
Original local archives, unrelated datasets, and model checkpoints are untouched.
Run only one uploader against this dataset root at a time.
HELP
}

source_dir=/mnt/c/Users/22rya/Downloads
remote=ryreu@tigris.rc.rit.edu
remote_root=/home/ryreu/atlas/datasets/JetClass
only=all
headroom=10
dry_run=0
while (($#)); do
  case "$1" in
    --source-dir|--remote|--remote-root|--only|--headroom-gib)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --source-dir) source_dir=$2 ;;
        --remote) remote=$2 ;;
        --remote-root) remote_root=$2 ;;
        --only) only=$2 ;;
        --headroom-gib) headroom=$2 ;;
      esac
      shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$remote" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.@-]*$ ]] || { echo 'Invalid remote host.' >&2; exit 2; }
[[ "$remote_root" =~ ^/[a-zA-Z0-9_./-]+/JetClass$ && "$remote_root" != *'/../'* && "$remote_root" != *'/./'* ]] || {
  echo 'Use a simple absolute remote path ending in /JetClass, without dot components.' >&2; exit 2;
}
[[ "$only" == all || "$only" == val || "$only" =~ ^part[0-9]$ ]] || { echo 'Invalid --only value.' >&2; exit 2; }
[[ "$headroom" =~ ^[0-9]+$ && "$headroom" -ge 1 ]] || { echo 'Headroom must be a positive integer.' >&2; exit 2; }
[[ -d "$source_dir" ]] || { echo "Missing local directory: $source_dir" >&2; exit 1; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
helper="${script_dir}/stage_jetclass_archive.py"
[[ -f "$helper" ]] || { echo "Missing companion helper: $helper" >&2; exit 1; }

archives=()
for part in {0..9}; do
  if [[ "$only" == all || "$only" == "part${part}" ]]; then
    archives+=("JetClass_Pythia_train_100M_part${part}.tar")
  fi
done
if [[ "$only" == all || "$only" == val ]]; then archives+=(JetClass_Pythia_val_5M.tar); fi

total=0
for name in "${archives[@]}"; do
  path="${source_dir}/${name}"
  [[ -f "$path" && ! -L "$path" ]] || { echo "Missing or linked local archive: $path" >&2; exit 1; }
  bytes="$(stat -c %s -- "$path")"
  ((bytes > 0)) || { echo "Empty archive: $path" >&2; exit 1; }
  total=$((total + bytes))
  printf '%-43s %s bytes\n' "$name" "$bytes"
done
printf '\nDestination: %s:%s\nSelected archive bytes: %s\n' "$remote" "$remote_root" "$total"
if ((dry_run)); then echo 'Dry run only: no connection or changes made.'; exit 0; fi

for executable in ssh rsync python3 sha256sum flock; do
  command -v "$executable" >/dev/null || { echo "Install $executable in WSL before uploading." >&2; exit 1; }
done

# Prevent two local copies from writing the same remote staging archive.
lock_key="$(printf '%s\n%s\n' "$remote" "$remote_root" | sha256sum)"
lock_key=${lock_key%% *}
exec 9>"/tmp/jetclass-upload-${UID}-${lock_key}.lock"
flock -n 9 || { echo 'Another local upload to this dataset root is already running.' >&2; exit 1; }

control_dir="$(mktemp -d /tmp/jetclass-upload.XXXXXX)"
control_socket="${control_dir}/ssh"
ssh_options=(-o "ControlPath=${control_socket}" -o ControlMaster=no -o BatchMode=yes
             -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
cleanup() {
  code=$?
  trap - EXIT
  ssh "${ssh_options[@]}" -O exit "$remote" >/dev/null 2>&1 || true
  rmdir -- "$control_dir" 2>/dev/null || true
  if ((code != 0)); then
    echo 'Stopped safely. Rerun the same command to resume; do not delete partial files.' >&2
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo 'Opening one reusable SSH connection. Password/Duo may be requested.'
# Authentication remains interactive. No credentials or host-key exceptions are stored.
ssh -fN -o ControlMaster=yes -o "ControlPath=${control_socket}" -o ControlPersist=600 \
  -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$remote"

ssh "${ssh_options[@]}" "$remote" \
  "set -eu; command -v rsync; command -v python3; for p in '$remote_root' '$remote_root/incoming' '$remote_root/Pythia' '$remote_root/Pythia/train_100M' '$remote_root/Pythia/val_5M'; do test -d \"\$p\"; test \"\$(realpath -e -- \"\$p\")\" = \"\$p\"; done"
ssh "${ssh_options[@]}" "$remote" \
  "python3 -c 'import sys; sys.exit(\"Remote python3 must be Python 3.8 or newer.\" if sys.version_info < (3, 8) else 0)'"

helper_sha="$(sha256sum -- "$helper")"
helper_sha=${helper_sha%% *}
remote_helper="${remote_root}/incoming/.stage_jetclass_archive_${helper_sha}.py"
rsync_shell="ssh -o ControlMaster=no -o BatchMode=yes -o ControlPath=${control_socket} -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
rsync -t -e "$rsync_shell" -- "$helper" "${remote}:${remote_helper}"
ssh "${ssh_options[@]}" "$remote" "printf '%s  %s\n' '$helper_sha' '$remote_helper' | sha256sum -c -"

is_complete() {
  python3 -c 'import json,sys; value=json.load(sys.stdin)["complete"]; assert isinstance(value,bool); print("yes" if value else "no")'
}
remote_step() {
  ssh "${ssh_options[@]}" "$remote" \
    "python3 '$remote_helper' '$1' --root '$remote_root' --archive '$name' --sha256 '$digest' --archive-bytes '$bytes' --headroom-gib '$headroom'"
}

for name in "${archives[@]}"; do
  path="${source_dir}/${name}"
  bytes="$(stat -c %s -- "$path")"
  printf '\n=== %s ===\nComputing local SHA-256...\n' "$name"
  digest="$(sha256sum -- "$path")"
  digest=${digest%% *}
  done_status="$(remote_step status | is_complete)"
  if [[ "$done_status" == yes ]]; then
    echo 'Already extracted and recorded; skipping upload.'
    remote_step finalize
    continue
  fi
  done_status="$(remote_step prepare | is_complete)"
  if [[ "$done_status" == yes ]]; then
    echo 'Already complete; skipping upload.'
    remote_step finalize
    continue
  fi
  # In-place delta transfer retains partial uploads without a second archive copy.
  # --checksum also repairs a same-size corrupt staging file on the next retry.
  rsync --times --partial --inplace --checksum --info=progress2 --human-readable \
    -e "$rsync_shell" -- "$path" "${remote}:${remote_root}/incoming/${name}"
  remote_step finalize
done
echo 'All selected training/validation archives are staged and verified. No jobs were submitted.'

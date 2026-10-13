#!/usr/bin/env python3
"""Stream a complete RC backup to disk; never delete or change remote files.

Run in a local interactive terminal so OpenSSH can request password/Duo MFA.
The archive includes the small rest of Fresh_check (including Git history) as
well as RPT artifacts. RETB worktrees are preserved too, not cleanup targets.
Only the Python standard library, local OpenSSH, and remote GNU tar are needed.
"""

from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time


REMOTE_BASE = "/home/ryreu/atlas"
ROOTS = (
    "Fresh_check",
    ".rpt_worktrees",
    ".rpt_launch_worktrees",
    "rptabl_results_207556.tar.gz",
    "rpt-prepull-backup.t2Hynh",
)
MARKER = "RPT_BACKUP_V1 "
BLOCK = 4 * 1024 * 1024

# Executed remotely through python3 -c; neither an archive nor a helper file is
# created on RC. The digest travels on stderr, separate from the binary stream.
REMOTE_PROGRAM = r'''
import hashlib, json, pathlib, subprocess, sys
base = pathlib.Path(sys.argv[1])
roots = json.loads(sys.argv[2])
available = int(sys.argv[3])
marker = "RPT_BACKUP_V1 "
for name in roots:
    path = base / name
    if path.resolve(strict=True) != path:
        raise RuntimeError("Backup root is a symlink or resolves elsewhere: " + str(path))
sizes = subprocess.check_output(["du", "-sb", "--", *roots], cwd=base, text=True)
estimated = sum(int(line.split("\t", 1)[0]) for line in sizes.splitlines())
if estimated + 2 * 1024**3 > available:
    raise RuntimeError("Not enough local free space: need %.2f GiB including headroom; have %.2f GiB"
                       % ((estimated + 2 * 1024**3) / 1024**3, available / 1024**3))
print(marker + json.dumps({"stage": "inventory", "estimated_bytes": estimated,
                          "base": str(base), "roots": roots}), file=sys.stderr, flush=True)
digest = hashlib.sha256()
size = 0
child = subprocess.Popen(["tar", "--sparse", "-cf", "-", "--", *roots],
                         cwd=base, stdout=subprocess.PIPE)
try:
    while True:
        data = child.stdout.read(4 * 1024 * 1024)
        if not data:
            break
        sys.stdout.buffer.write(data)
        digest.update(data)
        size += len(data)
    sys.stdout.buffer.flush()
    code = child.wait()
    if code:
        raise RuntimeError("tar failed or detected changing files (exit %d); backup NOT verified" % code)
    print(marker + json.dumps({"stage": "complete", "sha256": digest.hexdigest(),
                              "archive_bytes": size}), file=sys.stderr, flush=True)
finally:
    if child.poll() is None:
        child.terminate()
        child.wait()
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(BLOCK), b""):
            digest.update(data)
    return digest.hexdigest()


def archive_inventory(path: Path) -> dict:
    """Check readable headers and expected roots without extracting 42 GB."""
    present = set()
    counts = {"files": 0, "checkpoint_pt_files": 0, "npz_files": 0,
              "symlinks": 0, "hardlinks": 0}
    checkpoints = []
    links = []
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not name.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if name.parts[0] not in ROOTS:
                raise ValueError(f"Unexpected archive root: {member.name}")
            present.add(name.parts[0])
            if member.isfile():
                counts["files"] += 1
                if name.suffix == ".pt":
                    counts["checkpoint_pt_files"] += 1
                    checkpoints.append(member.name)
                if name.suffix == ".npz":
                    counts["npz_files"] += 1
            elif member.issym():
                counts["symlinks"] += 1
                links.append({"path": member.name, "target": member.linkname})
            elif member.islnk():
                counts["hardlinks"] += 1
            elif not member.isdir():
                raise ValueError(f"Unexpected special file: {member.name}")
    if present != set(ROOTS):
        raise ValueError(f"Archive missing roots: {sorted(set(ROOTS) - present)}")
    return {**counts, "roots": sorted(present), "checkpoints": checkpoints,
            "symlink_targets": links}


def receipt_path(archive: Path) -> Path:
    return archive.with_name(archive.name + ".verified.json")


def verify(archive: Path) -> dict:
    receipt = json.loads(receipt_path(archive).read_text(encoding="utf-8"))
    if receipt.get("status") != "verified":
        raise ValueError("No successful verification receipt")
    if archive.stat().st_size != receipt["archive_bytes"]:
        raise ValueError("Archive size differs from verified download")
    print("Reading local archive to verify SHA-256...", flush=True)
    if sha256_file(archive) != receipt["remote_sha256"]:
        raise ValueError("Archive SHA-256 differs from verified remote stream")
    inventory = archive_inventory(archive)
    print(f"VERIFIED: {archive}\nSHA-256: {receipt['remote_sha256']}", flush=True)
    return inventory


def download(archive: Path, remote: str) -> dict:
    if shutil.which("ssh") is None:
        raise RuntimeError("OpenSSH ssh was not found on PATH")
    partial = archive.with_name(archive.name + ".partial")
    for target in (archive, partial, receipt_path(archive)):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(archive.parent).free
    command = "python3 -c " + shlex.quote(REMOTE_PROGRAM)
    command += " " + " ".join(shlex.quote(value) for value in (
        REMOTE_BASE, json.dumps(ROOTS), str(available)))
    events = []

    def read_stderr(pipe):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        # Password/Duo prompts may not end in a newline. Forward available bytes
        # immediately, rather than hiding the prompt behind a blocking readline.
        for raw in iter(lambda: pipe.read1(4096), b""):
            text = decoder.decode(raw)
            sys.stderr.write(text)
            sys.stderr.flush()
            pending += text
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                if line.startswith(MARKER):
                    try:
                        events.append(json.loads(line[len(MARKER):]))
                    except ValueError:
                        pass

    print(f"Connecting to {remote}; complete password/Duo in this terminal.", flush=True)
    print(f"Destination: {archive}\nNo remote files will be changed or deleted.", flush=True)
    started = last_progress = time.monotonic()
    received = 0
    digest = hashlib.sha256()
    with partial.open("xb") as destination:
        # No shell and no PowerShell text redirection: tar bytes stay binary.
        process = subprocess.Popen(
            ["ssh", "-T", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
             remote, command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        reader = threading.Thread(target=read_stderr, args=(process.stderr,), daemon=True)
        reader.start()
        try:
            for data in iter(lambda: process.stdout.read(BLOCK), b""):
                destination.write(data)
                digest.update(data)
                received += len(data)
                now = time.monotonic()
                if now - last_progress >= 30:
                    print(f"Downloaded {received / 1024**3:.2f} GiB "
                          f"({received / 1024**2 / (now - started):.1f} MiB/s average)", flush=True)
                    last_progress = now
            code = process.wait()
            reader.join()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            reader.join(timeout=5)
            process.stdout.close()
            process.stderr.close()
    if code:
        raise RuntimeError(f"SSH/tar failed (exit {code}). Unverified partial retained: {partial}")
    completed = [row for row in events if row.get("stage") == "complete"]
    inventories = [row for row in events if row.get("stage") == "inventory"]
    if len(completed) != 1 or len(inventories) != 1:
        raise RuntimeError("Remote verification messages missing or duplicated; do not delete RC data")
    remote_result = completed[0]
    if remote_result["archive_bytes"] != received or remote_result["sha256"] != digest.hexdigest():
        raise RuntimeError("Local stream differs from remote archive; do not delete RC data")
    print("Transfer finished. Re-reading local file to verify stored bytes...", flush=True)
    if sha256_file(partial) != remote_result["sha256"]:
        raise RuntimeError("On-disk SHA-256 mismatch; do not delete RC data")
    print("Checking archive contents (without extracting)...", flush=True)
    inventory = archive_inventory(partial)
    receipt = {"status": "verified", "verified_utc": datetime.now(timezone.utc).isoformat(),
               "remote": remote, "remote_base": REMOTE_BASE,
               "remote_roots": list(ROOTS), "archive": str(archive),
               "archive_bytes": received, "remote_sha256": remote_result["sha256"],
               "inventory": inventory,
               "note": "No remote deletion performed. Check live jobs and symlink dependencies before cleanup."}
    if archive.exists():
        raise FileExistsError(f"Refusing to replace existing archive: {archive}")
    partial.rename(archive)
    with receipt_path(archive).open("x", encoding="utf-8") as destination:
        json.dump(receipt, destination, indent=2)
        destination.write("\n")
    print(f"\nVERIFIED: {archive}\nSHA-256: {receipt['remote_sha256']}\n"
          f"Files: {inventory['files']}; .pt: {inventory['checkpoint_pt_files']}; "
          f".npz: {inventory['npz_files']}; symlinks: {inventory['symlinks']}\n"
          f"Receipt: {receipt_path(archive)}\nNothing deleted on RC.", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New local .tar filename (no overwrite)")
    parser.add_argument("--remote", default="ryreu@sporcsubmit.rc.rit.edu")
    parser.add_argument("--verify", type=Path, help="Recheck an existing archive against its receipt")
    args = parser.parse_args()
    if bool(args.output) == bool(args.verify):
        parser.error("Choose exactly one of --output or --verify")
    try:
        if args.verify:
            verify(args.verify.resolve())
        else:
            download(args.output.resolve(), args.remote)
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
        print(f"BACKUP NOT VERIFIED: {error}\nDo not delete remote originals.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

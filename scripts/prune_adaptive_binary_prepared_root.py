#!/usr/bin/env python3
"""Prune rebuildable prepared-root payloads before a fresh 30 GB campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.runtime_acceptance import (  # noqa: E402
    require_runtime_acceptance,
)


REBUILDABLE_DIRECTORIES = ("archives", "targets", "inputs")


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_symlink():
        raise ValueError(f"refusing to prune symlinked tree: {path}")
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--bootstrap-evidence-root", required=True)
    parser.add_argument("--runtime-acceptance", required=True)
    parser.add_argument("--maximum-retained-bytes", type=int, default=5_000_000_000)
    parser.add_argument("--approve-prune", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not args.approve_prune:
        raise PermissionError("prepared-root pruning requires explicit approval")
    root = Path(args.prepared_root).resolve()
    evidence = Path(args.bootstrap_evidence_root).resolve()
    if not root.is_dir() or not root.name.startswith("adaptive_binary_pseudooffline_"):
        raise ValueError("prepared root does not look like an ABPH campaign")
    if evidence.parent != root / "audits" or not evidence.name.startswith("bootstrap_"):
        raise ValueError("bootstrap evidence is not the expected retained subtree")
    require_runtime_acceptance(args.runtime_acceptance, scope="ddp4_runtime")

    before = _tree_bytes(root)
    removals = []
    removable_bytes = 0
    for name in REBUILDABLE_DIRECTORIES:
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError(f"prune path escaped prepared root: {path}")
        measured = _tree_bytes(path)
        removals.append({"path": str(path), "bytes": measured})
        removable_bytes += measured
    removable_files = []
    for path in evidence.rglob("best_model_val.pt"):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unexpected benchmark checkpoint path: {path}")
        measured = path.stat().st_size
        removable_files.append({"path": str(path), "bytes": measured})
        removable_bytes += measured
    stale_bootstrap_directories = []
    audits_root = (root / "audits").resolve()
    if audits_root.is_dir():
        for path in sorted(audits_root.glob("bootstrap_*")):
            resolved = path.resolve()
            if resolved == evidence:
                continue
            if resolved.parent != audits_root or path.is_symlink() or not path.is_dir():
                raise ValueError(f"unexpected stale bootstrap evidence path: {path}")
            measured = _tree_bytes(path)
            stale_bootstrap_directories.append(
                {"path": str(resolved), "bytes": measured}
            )
            removable_bytes += measured
    projected_retained = before - removable_bytes
    if projected_retained > int(args.maximum_retained_bytes):
        raise RuntimeError(
            f"prepared root would retain {projected_retained} bytes, exceeding the "
            f"{args.maximum_retained_bytes}-byte bootstrap allowance"
        )
    for row in removals:
        path = Path(row["path"])
        if path.exists():
            shutil.rmtree(path)
    for row in removable_files:
        Path(row["path"]).unlink()
    for row in stale_bootstrap_directories:
        shutil.rmtree(Path(row["path"]))
    after = _tree_bytes(root)
    if after > int(args.maximum_retained_bytes):
        raise RuntimeError("prepared root exceeds retained-byte allowance after pruning")
    payload = {
        "contract": "adaptive_binary_prepared_root_prune_v1",
        "ok": True,
        "prepared_root": str(root),
        "runtime_acceptance": str(Path(args.runtime_acceptance).resolve()),
        "bytes_before": before,
        "bytes_removed": removable_bytes,
        "bytes_after": after,
        "maximum_retained_bytes": int(args.maximum_retained_bytes),
        "removed_directories": removals,
        "removed_runtime_benchmark_checkpoints": removable_files,
        "removed_stale_bootstrap_directories": stale_bootstrap_directories,
        "preserved_runs": str(root / "runs"),
        "preserved_bootstrap_evidence": str(evidence),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

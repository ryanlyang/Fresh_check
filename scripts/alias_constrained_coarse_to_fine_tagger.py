#!/usr/bin/env python3
"""Create a provenance-explicit, storage-light alias of a selected tagger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--alias-dir", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--alias-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    alias_dir = Path(args.alias_dir)
    source_checkpoint = source_dir / "best_model_val.pt"
    source_report_path = source_dir / "run_report.json"
    if not source_checkpoint.is_file() or not source_report_path.is_file():
        raise FileNotFoundError(f"source tagger is incomplete: {source_dir}")
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias_checkpoint = alias_dir / "best_model_val.pt"
    alias_report_path = alias_dir / "run_report.json"
    if not args.overwrite and (alias_checkpoint.exists() or alias_report_path.exists()):
        raise FileExistsError(f"alias output already exists: {alias_dir}")
    if alias_checkpoint.exists() or alias_checkpoint.is_symlink():
        alias_checkpoint.unlink()
    relative_source = os.path.relpath(source_checkpoint, start=alias_dir)
    alias_checkpoint.symlink_to(relative_source)

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    checkpoint_hash = _sha256(source_checkpoint)
    report = {
        **source_report,
        "ok": True,
        "run_id": args.alias_name,
        "variant": args.alias_name,
        "alias_of": args.source_name,
        "independently_trained": False,
        "shared_checkpoint": True,
        "shared_checkpoint_sha256": checkpoint_hash,
        "shared_configuration_hash": source_report.get("configuration_hash"),
        "checkpoint": str(alias_checkpoint),
        "checkpoint_sha256": checkpoint_hash,
    }
    alias_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

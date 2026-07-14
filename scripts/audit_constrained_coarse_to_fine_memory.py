#!/usr/bin/env python3
"""Fail early when compressed campaign inputs cannot fit in a Slurm allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile


def _npz_uncompressed_bytes(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        return sum(int(row.file_size) for row in archive.infolist())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", required=True)
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=("model_train", "model_val", "stack_val"))
    parser.add_argument("--allocated-memory-mb", type=int, default=0)
    parser.add_argument("--safety-factor", type=float, default=1.35)
    parser.add_argument("--model-reserve-gb", type=float, default=20.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.safety_factor < 1.0 or args.model_reserve_gb < 0.0:
        raise SystemExit("invalid memory safety settings")

    hlt_root = Path(args.hlt_cache_dir)
    offline_root = Path(args.offline_cache_dir)
    target_root = Path(args.target_cache_dir)
    rows = {}
    resident = 0
    for split in args.splits:
        hlt = _npz_uncompressed_bytes(hlt_root / f"{split}_fixed_hlt.npz")
        offline = _npz_uncompressed_bytes(offline_root / f"{split}_offline.npz")
        rows[split] = {"hlt_uncompressed_bytes": hlt, "offline_uncompressed_bytes": offline}
        resident += hlt + offline
    target_candidates = tuple(target_root.rglob("*.npz"))
    largest_target = max((_npz_uncompressed_bytes(path) for path in target_candidates), default=0)
    reserve = int(float(args.model_reserve_gb) * 1024**3)
    estimated_peak = int(float(args.safety_factor) * resident + largest_target + reserve)
    allocated = int(args.allocated_memory_mb) * 1024**2
    ok = allocated <= 0 or estimated_peak <= int(0.90 * allocated)
    report = {
        "ok": ok,
        "splits": rows,
        "resident_cache_bytes": resident,
        "largest_target_shard_bytes": largest_target,
        "model_reserve_bytes": reserve,
        "safety_factor": float(args.safety_factor),
        "estimated_peak_bytes": estimated_peak,
        "allocated_bytes": allocated if allocated > 0 else None,
        "allocation_utilization_estimate": None if allocated <= 0 else estimated_peak / allocated,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit("estimated input/model memory exceeds 90% of the Slurm allocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Persist one atomic wall-time measurement for a runtime benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument("--profile-enabled", choices=("0", "1"), required=True)
    parser.add_argument("--run-report", required=True)
    parser.add_argument("--delete-selected-checkpoint", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = Path(args.run_report).resolve()
    if args.elapsed_seconds <= 0 or not report.is_file():
        raise ValueError("wall-time evidence requires a positive duration and run report")
    checkpoint = report.parent / "best_model_val.pt"
    checkpoint_bytes = checkpoint.stat().st_size if checkpoint.is_file() else 0
    if args.delete_selected_checkpoint and checkpoint.is_file():
        checkpoint.unlink()
    payload = {
        "contract": "adaptive_binary_runtime_walltime_v1",
        "elapsed_seconds": float(args.elapsed_seconds),
        "runtime_profile_enabled": args.profile_enabled == "1",
        "run_report": str(report),
        "selected_checkpoint_removed": bool(
            args.delete_selected_checkpoint and checkpoint_bytes > 0
        ),
        "selected_checkpoint_bytes_released": int(checkpoint_bytes),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

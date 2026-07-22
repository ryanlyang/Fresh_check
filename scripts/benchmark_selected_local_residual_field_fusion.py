#!/usr/bin/env python3
"""Benchmark only the immutable selected fusion set and its deployable members."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.fusion_runtime import benchmark_selected_fusion_runtime  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-fusion", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measured-batches", type=int, default=50)
    args = parser.parse_args(argv)
    report = benchmark_selected_fusion_runtime(
        args.selected_fusion, batch_size=args.batch_size, warmup_batches=args.warmup_batches,
        measured_batches=args.measured_batches,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

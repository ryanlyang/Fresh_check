#!/usr/bin/env python3
"""Build local per-HLT-particle residual-field target caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    DEFAULT_LOCAL_RESIDUAL_RADII,
    LOCAL_PARTICLE_RESIDUAL_FIELD_ALL_SPLITS,
    LOCAL_PARTICLE_RESIDUAL_FIELD_PRIMARY_SPLITS,
    build_local_particle_residual_field_caches,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Split manifest JSON or JSON.GZ path.")
    parser.add_argument("--hlt-cache-dir", required=True, help="Directory containing <split>_fixed_hlt.npz caches.")
    parser.add_argument("--offline-cache-dir", required=True, help="Directory containing <split>_offline.npz caches.")
    parser.add_argument("--output-cache-dir", required=True, help="Destination for local residual-field target caches.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to cache. Defaults to model_train/model_val/stack_val.",
    )
    parser.add_argument(
        "--include-final-test-targets",
        action="store_true",
        help="Also allow writing final_test targets as explicit oracle-only diagnostics.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Cache all splits, including final_test. Requires --include-final-test-targets.",
    )
    parser.add_argument(
        "--radii",
        nargs="+",
        type=float,
        default=list(DEFAULT_LOCAL_RESIDUAL_RADII),
        help="Local neighborhood radii in eta/phi space.",
    )
    parser.add_argument("--chunk-size", type=int, default=1024, help="Number of jets per target-building chunk.")
    parser.add_argument(
        "--target-dtype",
        default="float16",
        choices=("float16", "float32"),
        help="Storage dtype for target_fields. Loading promotes back to float32 for training.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_splits:
        splits = LOCAL_PARTICLE_RESIDUAL_FIELD_ALL_SPLITS
        if not args.include_final_test_targets:
            raise SystemExit("--all-splits requires --include-final-test-targets")
    else:
        splits = tuple(args.splits or LOCAL_PARTICLE_RESIDUAL_FIELD_PRIMARY_SPLITS)
    reports = build_local_particle_residual_field_caches(
        manifest_path=args.manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        output_cache_dir=args.output_cache_dir,
        splits=splits,
        radii=tuple(float(radius) for radius in args.radii),
        overwrite=bool(args.overwrite),
        allow_final_test_targets=bool(args.include_final_test_targets),
        chunk_size=int(args.chunk_size),
        target_dtype=str(args.target_dtype),
    )
    result = {
        "ok": True,
        "manifest": str(args.manifest),
        "hlt_cache_dir": str(args.hlt_cache_dir),
        "offline_cache_dir": str(args.offline_cache_dir),
        "output_cache_dir": str(args.output_cache_dir),
        "splits": list(splits),
        "radii": [float(radius) for radius in args.radii],
        "target_dtype": str(args.target_dtype),
        "reports": reports,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

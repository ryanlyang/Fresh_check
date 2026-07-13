#!/usr/bin/env python3
"""Build deterministic constrained coarse-to-fine offline hierarchy targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine import (  # noqa: E402
    HIERARCHY_TARGET_DEFAULT_SPLITS,
    build_hierarchy_target_caches,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Split manifest JSON or JSON.GZ path.")
    parser.add_argument("--hlt-cache-dir", required=True, help="HLT v2 strength-2.5 cache directory.")
    parser.add_argument("--offline-cache-dir", required=True, help="Aligned offline raw-token cache directory.")
    parser.add_argument("--output-cache-dir", required=True, help="Destination hierarchy-target cache directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(HIERARCHY_TARGET_DEFAULT_SPLITS),
        help="Offline-supervised splits to build. final_test is always forbidden.",
    )
    parser.add_argument(
        "--radial-boundary",
        type=float,
        default=None,
        help="Explicit inner/outer shell boundary. Default fits one pT-weighted median from model_train HLT.",
    )
    parser.add_argument("--coordinate-extent", type=float, default=0.8)
    parser.add_argument("--radial-histogram-bins", type=int, default=4096)
    parser.add_argument("--radial-fit-chunk-size", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=8192, help="Jets per independently hashed output shard.")
    parser.add_argument(
        "--target-dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Accounting storage dtype. float32 is the scientific default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_hierarchy_target_caches(
        manifest_path=args.manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        output_cache_dir=args.output_cache_dir,
        splits=tuple(args.splits),
        radial_boundary=args.radial_boundary,
        coordinate_extent=float(args.coordinate_extent),
        radial_histogram_bins=int(args.radial_histogram_bins),
        radial_fit_chunk_size=int(args.radial_fit_chunk_size),
        chunk_size=int(args.chunk_size),
        target_dtype=str(args.target_dtype),
        overwrite=bool(args.overwrite),
    )
    reports = result["reports"]
    summary = {
        "ok": True,
        "manifest": str(args.manifest),
        "hlt_cache_dir": str(args.hlt_cache_dir),
        "offline_cache_dir": str(args.offline_cache_dir),
        "output_cache_dir": str(args.output_cache_dir),
        "cache_set": result["cache_set"],
        "split_reports": {
            split: {
                "n_jets": report["n_jets"],
                "n_shards": report["n_shards"],
                "target_content_hash": report["target_content_hash"],
                "hlt_content_hash": report["hlt_content_hash"],
                "offline_content_hash": report["offline_content_hash"],
                "max_parent_child_closure_error": report["diagnostics_summary"][
                    "max_parent_child_closure_error"
                ],
            }
            for split, report in reports.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


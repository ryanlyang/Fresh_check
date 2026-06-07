#!/usr/bin/env python3
"""Generate and cache fixed HLT views for Step 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_cache import (  # noqa: E402
    DEFAULT_HLT_SEEDS,
    audit_hlt_cache,
    generate_and_cache_hlt_split,
)
from jetclass_fresh.jetclass_data import DEFAULT_DATA_DIR, SPLIT_ORDER, load_split_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Step 2 split manifest path (.json or .json.gz)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"JetClass data directory; defaults to manifest data_dir or {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--cache-dir",
        default="checkpoints/jetclass_fresh_hlt_cache",
        help="Directory for per-split HLT .npz files and metadata JSON",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLIT_ORDER),
        choices=list(SPLIT_ORDER),
        help="Splits to generate",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing HLT cache files")
    parser.add_argument("--show-progress", action="store_true", help="Show per-jet HLT progress if tqdm exists")
    parser.add_argument(
        "--verify-label-branches",
        action="store_true",
        help="Verify ROOT label branches agree with filename labels while loading offline views",
    )
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_split_manifest(args.manifest)
    data_dir = args.data_dir or manifest.data_dir or DEFAULT_DATA_DIR
    reports = {}

    for split in args.splits:
        metadata = generate_and_cache_hlt_split(
            manifest,
            split,
            args.cache_dir,
            data_dir=data_dir,
            seed=DEFAULT_HLT_SEEDS[split],
            overwrite=args.overwrite,
            show_progress=args.show_progress,
            verify_label_branches=args.verify_label_branches,
            read_chunk_size=args.read_chunk_size,
        )
        reports[split] = {
            "array_path": metadata["array_path"],
            "metadata_path": metadata["metadata_path"],
            "n_jets": metadata["n_jets"],
            "seed": metadata["seed"],
            "hlt_content_hash": metadata["hlt_content_hash"],
            "offline_constit_count_summary": metadata["offline_constit_count_summary"],
            "hlt_constit_count_summary": metadata["hlt_constit_count_summary"],
            "hlt_diagnostics_summary": metadata["hlt_diagnostics_summary"],
        }

    audit = audit_hlt_cache(manifest, args.cache_dir, splits=args.splits)
    result = {
        "cache_dir": str(Path(args.cache_dir)),
        "data_dir": str(data_dir),
        "splits": list(args.splits),
        "reports": reports,
        "audit": audit,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

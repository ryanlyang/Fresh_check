#!/usr/bin/env python3
"""Validate all REGION tree shards and atomically finalize one split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import finalize_tree_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-jet-count", type=int, required=True)
    parser.add_argument("--hlt-content-sha256", required=True)
    parser.add_argument("--tree-resource-sha256", required=True)
    parser.add_argument("--backend-manifest-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    expected_shards = (
        int(args.expected_jet_count) + 9_999
    ) // 10_000
    paths = [
        args.tree_dir / "shards" / f"shard_{index:05d}.metadata.json"
        for index in range(expected_shards)
    ]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("one or more expected REGION shard metadata files are absent")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "split": args.split,
                    "expected_jet_count": int(args.expected_jet_count),
                    "expected_shard_count": expected_shards,
                    "manifest_output": str(
                        (args.tree_dir / "manifest.json").resolve()
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = finalize_tree_split(
        args.tree_dir / "manifest.json",
        paths,
        split=args.split,
        expected_jet_count=args.expected_jet_count,
        hlt_content_sha256=args.hlt_content_sha256,
        tree_resource_sha256=args.tree_resource_sha256,
        backend_manifest_sha256=args.backend_manifest_sha256,
    )
    print(manifest["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

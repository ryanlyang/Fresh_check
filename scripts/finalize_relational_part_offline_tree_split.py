#!/usr/bin/env python3
"""Finalize one authenticated offline REGION-tree split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.architecture_view_part.train import load_cached_offline_view  # noqa: E402
from teacher_logit_reco.relational_part.ca_tree import finalize_view_tree_split  # noqa: E402
from teacher_logit_reco.relational_part.contracts import load_hashed_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tree-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-jet-count", type=int, required=True)
    parser.add_argument("--tree-resource", type=Path, required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    args = parser.parse_args()
    view = load_cached_offline_view(args.cache_dir, args.split, verify_hash=True)
    if len(view.tokens) != args.expected_jet_count:
        raise ValueError("offline cache size differs from finalizer expectation")
    resource = load_hashed_json(args.tree_resource)
    backend = load_hashed_json(args.backend_manifest)
    metadata = sorted((args.tree_dir / "shards").glob("shard_*.metadata.json"))
    result = finalize_view_tree_split(
        args.tree_dir / "manifest.json",
        metadata,
        split=args.split,
        expected_jet_count=args.expected_jet_count,
        input_view="offline",
        input_content_sha256=str(view.metadata["offline_content_hash"]),
        tree_resource_sha256=resource["content_hash"],
        backend_manifest_sha256=backend["content_hash"],
    )
    print(result["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

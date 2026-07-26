#!/usr/bin/env python3
"""Fit immutable REGION scalers from model_train compact tree sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    bind_source_provenance,
    fit_region_normalization,
    load_hashed_json,
    source_snapshot,
    unpack_tree_shard,
    write_immutable_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tree-dir", type=Path, required=True)
    parser.add_argument("--relation-normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = load_hashed_json(
        args.relation_normalization,
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    manifest = load_hashed_json(
        args.tree_dir / "manifest.json",
        expected_contract=ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    )
    if manifest.get("split") != "model_train":
        raise ValueError("REGION normalization may only access model_train")
    view = load_cached_hlt_view(args.cache_dir, "model_train", verify_hash=True)
    trees = []
    identities = []
    for row in manifest["shards"]:
        shard_path = (
            args.tree_dir / "shards" / f"shard_{int(row['shard_index']):05d}.npz"
        )
        shard_identities, shard_trees = unpack_tree_shard(shard_path)
        identities.extend(shard_identities)
        trees.extend(shard_trees)
    expected_identities = [identity.key() for identity in view.jet_ids]
    if identities != expected_identities:
        raise ValueError("tree sidecar identities differ from model_train HLT cache")
    artifact = fit_region_normalization(
        view.tokens,
        view.mask,
        view.jet_ids,
        trees,
        relation_normalization_artifact=base,
        angular_tree_resource_sha256=manifest["parents"][
            "tree_resource_sha256"
        ],
    )
    artifact = bind_source_provenance(
        artifact, source_snapshot=source_snapshot(REPO_ROOT)
    )
    publication = None
    if not args.dry_run:
        publication = write_immutable_json(args.output, artifact)
    print(json.dumps(
        {
            "dry_run": bool(args.dry_run),
            "fit_split": "model_train",
            "final_test_accessed": False,
            "artifact": artifact,
            "publication": publication,
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

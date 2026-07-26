#!/usr/bin/env python3
"""Build one deterministic contiguous compact REGION tree shard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from jetclass_fresh.part_inputs import (  # noqa: E402
    build_particle_transformer_inputs_from_tokens,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    ANGULAR_TREE_RESOURCE_CONTRACT,
    build_compiled_tree,
    load_hashed_json,
    load_tree_backend,
    write_tree_shard,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--tree-resource", type=Path, required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--backend-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    resource = load_hashed_json(
        args.tree_resource, expected_contract=ANGULAR_TREE_RESOURCE_CONTRACT
    )
    backend_manifest = load_hashed_json(
        args.backend_manifest,
        expected_contract=ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    )
    source = (
        REPO_ROOT / "teacher_logit_reco" / "relational_part"
        / "csrc" / "relational_ca_tree_v1.cpp"
    )
    backend = load_tree_backend(
        args.backend_binary, args.backend_manifest, source_path=source
    )
    view = load_cached_hlt_view(args.cache_dir, args.split, verify_hash=True)
    start, stop = int(args.start), int(args.stop)
    if start < 0 or stop <= start or stop > len(view.tokens) or stop - start > 10_000:
        raise ValueError("tree shard range differs from the locked contiguous policy")
    tokens = view.tokens[start:stop]
    mask = view.mask[start:stop]
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens, mask, source_view="fixed_hlt"
    )
    vectors = inputs.pf_vectors.transpose(0, 2, 1)
    trees = [
        build_compiled_tree(backend, vectors[row], tokens[row], mask[row])
        for row in range(stop - start)
    ]
    output = (
        args.output_dir / "shards"
        / f"shard_{int(args.shard_index):05d}.npz"
    )
    result = write_tree_shard(
        output,
        trees,
        view.jet_ids[start:stop],
        hlt_content_sha256=str(view.metadata["hlt_content_hash"]),
        tree_resource_sha256=resource["content_hash"],
        backend_manifest_sha256=backend_manifest["content_hash"],
    )
    print(result["metadata"]["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

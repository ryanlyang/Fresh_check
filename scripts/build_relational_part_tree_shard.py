#!/usr/bin/env python3
"""Build one deterministic contiguous compact REGION tree shard."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
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
    validate_existing_tree_shard,
    write_tree_shard,
)


_FORKED_BACKEND = None


def _build_one(payload):
    if _FORKED_BACKEND is None:
        raise RuntimeError("forked tree backend was not initialized")
    vectors, tokens, mask = payload
    return build_compiled_tree(_FORKED_BACKEND, vectors, tokens, mask)


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
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    resource = load_hashed_json(
        args.tree_resource, expected_contract=ANGULAR_TREE_RESOURCE_CONTRACT
    )
    backend_manifest = load_hashed_json(
        args.backend_manifest,
        expected_contract=ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    )
    view = load_cached_hlt_view(args.cache_dir, args.split, verify_hash=True)
    start, stop = int(args.start), int(args.stop)
    if start < 0 or stop <= start or stop > len(view.tokens) or stop - start > 10_000:
        raise ValueError("tree shard range differs from the locked contiguous policy")
    output = (
        args.output_dir / "shards"
        / f"shard_{int(args.shard_index):05d}.npz"
    )
    reused = validate_existing_tree_shard(
        output,
        view.jet_ids[start:stop],
        hlt_content_sha256=str(view.metadata["hlt_content_hash"]),
        tree_resource_sha256=resource["content_hash"],
        backend_manifest_sha256=backend_manifest["content_hash"],
        recover_unregistered_partial=True,
    )
    if reused is not None:
        print(reused["content_hash"])
        return 0
    source = (
        REPO_ROOT / "teacher_logit_reco" / "relational_part"
        / "csrc" / "relational_ca_tree_v1.cpp"
    )
    backend = load_tree_backend(
        args.backend_binary, args.backend_manifest, source_path=source
    )
    tokens = view.tokens[start:stop]
    mask = view.mask[start:stop]
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens, mask, source_view="fixed_hlt"
    )
    vectors = inputs.pf_vectors.transpose(0, 2, 1)
    workers = min(max(int(args.workers), 1), stop - start)
    payloads = [
        (vectors[row], tokens[row], mask[row]) for row in range(stop - start)
    ]
    if workers == 1:
        trees = [
            build_compiled_tree(backend, *payload) for payload in payloads
        ]
    else:
        # Tigris is Linux: fork inherits the authenticated read-only extension
        # and parallelizes only across jets, never within a merge sequence.
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError(
                "multi-process tree building requires the Linux fork method"
            )
        global _FORKED_BACKEND
        _FORKED_BACKEND = backend
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            trees = list(pool.map(_build_one, payloads, chunksize=8))
        _FORKED_BACKEND = None
    if args.dry_run:
        print(
            {
                "dry_run": True,
                "split": args.split,
                "start": start,
                "stop": stop,
                "shard_index": int(args.shard_index),
                "output": str(output.resolve()),
                "jet_count": len(trees),
                "across_jet_processes": workers,
            }
        )
        return 0
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

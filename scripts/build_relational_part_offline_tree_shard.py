#!/usr/bin/env python3
"""Build one deterministic REGION-tree shard from authenticated offline tokens."""

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

from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens  # noqa: E402
from teacher_logit_reco.architecture_view_part.train import load_cached_offline_view  # noqa: E402
from teacher_logit_reco.relational_part.ca_tree import (  # noqa: E402
    build_compiled_tree,
    load_tree_backend,
    write_view_tree_shard,
)
from teacher_logit_reco.relational_part.contracts import load_hashed_json  # noqa: E402


_BACKEND = None


def _build_one(payload):
    if _BACKEND is None:
        raise RuntimeError("forked tree backend was not initialized")
    return build_compiled_tree(_BACKEND, *payload)


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
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = parser.parse_args()

    resource = load_hashed_json(args.tree_resource)
    backend_manifest = load_hashed_json(args.backend_manifest)
    view = load_cached_offline_view(args.cache_dir, args.split, verify_hash=True)
    start, stop = int(args.start), int(args.stop)
    if start < 0 or stop <= start or stop > len(view.tokens) or stop - start > 10_000:
        raise ValueError("offline tree shard range differs from contiguous policy")
    source = REPO_ROOT / "teacher_logit_reco" / "relational_part" / "csrc" / "relational_ca_tree_v1.cpp"
    backend = load_tree_backend(args.backend_binary, args.backend_manifest, source_path=source)
    tokens = view.tokens[start:stop]
    mask = view.mask[start:stop]
    inputs = build_particle_transformer_inputs_from_tokens(tokens, mask, source_view="offline")
    vectors = inputs.pf_vectors.transpose(0, 2, 1)
    payloads = [(vectors[row], tokens[row], mask[row]) for row in range(stop - start)]
    workers = min(max(int(args.workers), 1), len(payloads))
    if workers == 1:
        trees = [build_compiled_tree(backend, *payload) for payload in payloads]
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("parallel offline tree building requires fork")
        global _BACKEND
        _BACKEND = backend
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
            trees = list(pool.map(_build_one, payloads, chunksize=8))
        _BACKEND = None
    result = write_view_tree_shard(
        args.output_dir / "shards" / f"shard_{args.shard_index:05d}.npz",
        trees,
        view.jet_ids[start:stop],
        input_view="offline",
        input_content_sha256=str(view.metadata["offline_content_hash"]),
        tree_resource_sha256=resource["content_hash"],
        backend_manifest_sha256=backend_manifest["content_hash"],
    )
    print(result["metadata"]["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

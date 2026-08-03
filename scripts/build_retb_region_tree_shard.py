#!/usr/bin/env python3
"""Build one deterministic RETB REGION-tree shard from an offline/HLT cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.part_inputs import (  # noqa: E402
    build_particle_transformer_inputs_from_tokens,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.replicas import (  # noqa: E402
    REALIZATION_POLICIES,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    ANGULAR_TREE_RESOURCE_CONTRACT,
    build_compiled_tree,
    load_tree_backend,
    validate_existing_tree_shard,
    write_tree_shard,
)


_WORKER_BACKEND = None


def _validate_view_coordinate(
    *,
    view_kind: str,
    logical_role: str,
    replica_id: int | None,
    realization_policy: str | None,
) -> None:
    if view_kind == "offline":
        if replica_id is not None or realization_policy is not None:
            raise ValueError("offline REGION view cannot declare HLT realization")
        return
    if replica_id is None or realization_policy not in REALIZATION_POLICIES:
        raise ValueError("HLT REGION view requires exact replica/policy")
    if logical_role in {"model_train", "scale_train"}:
        expected_replicas = REALIZATION_POLICIES[realization_policy][
            "training_replicas"
        ]
        if replica_id not in expected_replicas:
            raise ValueError("HLT REGION replica is incompatible with policy")
    elif realization_policy != "R_FIXED" or replica_id != 0:
        raise ValueError("evaluation REGION view must use replica-zero R_FIXED")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_worker_threads() -> None:
    """Keep one spawned process on one allocated CPU core."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)


def _initialize_worker(
    backend_binary: str,
    backend_manifest: str,
    source_path: str,
) -> None:
    """Load the authenticated extension inside a clean spawned process."""

    _configure_worker_threads()
    global _WORKER_BACKEND
    _WORKER_BACKEND = load_tree_backend(
        Path(backend_binary),
        Path(backend_manifest),
        source_path=Path(source_path),
    )


def _build_one(payload):
    if _WORKER_BACKEND is None:
        raise RuntimeError("spawned tree backend was not initialized")
    vectors, tokens, mask = payload
    return build_compiled_tree(_WORKER_BACKEND, vectors, tokens, mask)


def _load_view(
    *,
    view_kind: str,
    cache_dir: Path,
    logical_role: str,
    replica_id: int | None,
    realization_policy: str | None,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    if view_kind == "offline":
        manifest = load_hashed_json(
            cache_dir / "offline_input_manifest.json",
            expected_contract="retb_offline_input_cache_v1",
        )
        npz_path = cache_dir / manifest["npz_filename"]
        if (
            manifest["logical_role"] != logical_role
            or not npz_path.is_file()
            or _sha256(npz_path) != manifest["npz_sha256"]
        ):
            raise ValueError("offline REGION source cache differs")
        with np.load(npz_path, allow_pickle=False) as payload:
            tokens = np.asarray(payload["tokens"], dtype=np.float32)
            mask = np.asarray(payload["mask"], dtype=bool)
            identities = [
                str(value) for value in payload["identities"].tolist()
            ]
        return tokens, mask, identities, str(manifest["npz_sha256"])
    arrays, metadata = load_hlt_v3_cache(cache_dir)
    if (
        metadata["logical_role"] != logical_role
        or replica_id is None
        or int(metadata["replica_id"]) != int(replica_id)
        or metadata["degradation_profile_id"] != "D_NOMINAL"
        or metadata["realization_policy"] != realization_policy
    ):
        raise ValueError("HLT REGION source cache differs")
    return (
        np.asarray(arrays["tokens"], dtype=np.float32),
        np.asarray(arrays["mask"], dtype=bool),
        [str(value) for value in arrays["identities"].tolist()],
        str(metadata["array_content_sha256"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--view-kind", required=True, choices=("offline", "hlt"))
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--logical-role", required=True)
    parser.add_argument("--replica-id", type=int)
    parser.add_argument("--realization-policy")
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--stop", required=True, type=int)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tree-resource", required=True, type=Path)
    parser.add_argument("--backend-manifest", required=True, type=Path)
    parser.add_argument("--backend-binary", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    resource = load_hashed_json(
        args.tree_resource, expected_contract=ANGULAR_TREE_RESOURCE_CONTRACT
    )
    if resource.get("source") != campaign.get("source"):
        raise ValueError("REGION tree resource source lineage differs")
    backend_manifest = load_hashed_json(
        args.backend_manifest,
        expected_contract=ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    )
    backend_binary = args.backend_binary or (
        args.backend_manifest.parent / backend_manifest["binary_filename"]
    )
    _validate_view_coordinate(
        view_kind=args.view_kind,
        logical_role=args.logical_role,
        replica_id=args.replica_id,
        realization_policy=args.realization_policy,
    )
    tokens, mask, identities, view_content_sha = _load_view(
        view_kind=args.view_kind,
        cache_dir=args.cache_dir,
        logical_role=args.logical_role,
        replica_id=args.replica_id,
        realization_policy=args.realization_policy,
    )
    start, stop = int(args.start), int(args.stop)
    shard_size = int(args.shard_size)
    if (
        start < 0
        or stop <= start
        or stop > len(identities)
        or shard_size <= 0
        or shard_size > 10_000
    ):
        raise ValueError("RETB REGION view range differs")
    specifications = []
    for offset, shard_start in enumerate(range(start, stop, shard_size)):
        shard_stop = min(shard_start + shard_size, stop)
        shard_index = int(args.shard_index) + offset
        output = (
            args.output_dir
            / "shards"
            / f"shard_{shard_index:05d}.npz"
        )
        reused = validate_existing_tree_shard(
            output,
            identities[shard_start:shard_stop],
            hlt_content_sha256=view_content_sha,
            tree_resource_sha256=resource["content_hash"],
            backend_manifest_sha256=backend_manifest["content_hash"],
            recover_unregistered_partial=True,
        )
        specifications.append(
            {
                "shard_index": shard_index,
                "start": shard_start,
                "stop": shard_stop,
                "output": output,
                "reused": reused is not None,
            }
        )
    pending = [row for row in specifications if not row["reused"]]
    if not pending:
        print(
            json.dumps(
                {
                    "view_kind": args.view_kind,
                    "logical_role": args.logical_role,
                    "replica_id": args.replica_id,
                    "shard_count": len(specifications),
                    "reused_shard_count": len(specifications),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "view_kind": args.view_kind,
                    "logical_role": args.logical_role,
                    "replica_id": args.replica_id,
                    "start": start,
                    "stop": stop,
                    "shard_count": len(specifications),
                    "pending_shard_count": len(pending),
                },
                sort_keys=True,
            )
        )
        return 0
    source = (
        REPO_ROOT
        / "teacher_logit_reco"
        / "relational_part"
        / "csrc"
        / "relational_ca_tree_v1.cpp"
    )
    workers = min(max(int(args.workers), 1), shard_size)
    backend = None
    pool = None
    if workers > 1:
        if "spawn" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("parallel REGION building requires spawn")
        # Never fork a process after importing PyTorch or loading the C++
        # extension.  Inheriting those native thread/runtime states caused a
        # top-level SIGSEGV on a genuine Tigris miniature shard.  Each clean
        # child imports the compile-once campaign binary and authenticates it
        # against the same manifest and source before processing any jet.
        _configure_worker_threads()
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(
                str(backend_binary),
                str(args.backend_manifest),
                str(source),
            ),
        )
    else:
        _configure_worker_threads()
        backend = load_tree_backend(
            backend_binary, args.backend_manifest, source_path=source
        )
    published = 0
    try:
        for specification in pending:
            shard_start = int(specification["start"])
            shard_stop = int(specification["stop"])
            selected_tokens = tokens[shard_start:shard_stop]
            selected_mask = mask[shard_start:shard_stop]
            inputs = build_particle_transformer_inputs_from_tokens(
                selected_tokens,
                selected_mask,
                source_view=f"retb_{args.view_kind}",
            )
            vectors = inputs.pf_vectors.transpose(0, 2, 1)
            payloads = [
                (vectors[row], selected_tokens[row], selected_mask[row])
                for row in range(shard_stop - shard_start)
            ]
            if pool is None:
                trees = [
                    build_compiled_tree(backend, *row) for row in payloads
                ]
            else:
                trees = list(pool.map(_build_one, payloads, chunksize=8))
            write_tree_shard(
                specification["output"],
                trees,
                identities[shard_start:shard_stop],
                hlt_content_sha256=view_content_sha,
                tree_resource_sha256=resource["content_hash"],
                backend_manifest_sha256=backend_manifest["content_hash"],
            )
            published += 1
            print(
                json.dumps(
                    {
                        "published_shard_index": specification[
                            "shard_index"
                        ],
                        "published_shard_count": published,
                        "total_pending_shards": len(pending),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    print(
        json.dumps(
            {
                "view_kind": args.view_kind,
                "logical_role": args.logical_role,
                "replica_id": args.replica_id,
                "shard_count": len(specifications),
                "reused_shard_count": len(specifications) - len(pending),
                "published_shard_count": published,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

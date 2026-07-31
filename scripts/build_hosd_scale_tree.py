#!/usr/bin/env python3
"""Build one authenticated Stage-J scale tree view and close its wave."""

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

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    INPUT_VIEW_MANIFEST_CONTRACT,
    SCALE_INPUT_COMPLETION_CONTRACT,
    SCALE_TREE_WAVE_COMPLETION_CONTRACT,
    load_materialized_input_view,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    ANGULAR_TREE_RESOURCE_CONTRACT,
    build_compiled_tree,
    finalize_tree_split,
    load_tree_backend,
    validate_existing_tree_shard,
    write_tree_shard,
)


_FORKED_BACKEND = None
SHARD_SIZE = 10_000
COORDINATES = ("offline", "0", "1", "2", "3")


def _build_one(payload):
    if _FORKED_BACKEND is None:
        raise RuntimeError("forked tree backend was not initialized")
    vectors, tokens, mask = payload
    return build_compiled_tree(_FORKED_BACKEND, vectors, tokens, mask)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(root: Path, coordinate: str) -> tuple[Path, Path, str]:
    if coordinate == "offline":
        return (
            root / "scale_up" / "inputs" / "offline" / "scale_train.npz",
            root
            / "scale_up"
            / "trees"
            / "offline"
            / "scale_train_exclusive_ca_v1",
            "scale_train",
        )
    replica = int(coordinate)
    return (
        root / "scale_up" / "inputs" / "hlt" / f"replica_{replica}.npz",
        root / "scale_up" / "trees" / "hlt" / f"replica_{replica}",
        f"scale_train:r{replica}",
    )


def _close_wave(
    root: Path,
    campaign: dict,
    input_completion: dict,
    *,
    tree_resource_path: Path,
    tree_resource_sha256: str,
    backend_manifest_path: Path,
    backend_manifest_sha256: str,
) -> dict | None:
    rows = []
    for coordinate in COORDINATES:
        input_path, tree_dir, split = _paths(root, coordinate)
        manifest_path = tree_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = load_hashed_json(
            manifest_path,
            expected_contract="relational_ca_tree_split_manifest_v1",
        )
        view_manifest = load_hashed_json(
            input_path.with_suffix(input_path.suffix + ".json"),
            expected_contract=INPUT_VIEW_MANIFEST_CONTRACT,
        )
        if (
            manifest.get("split") != split
            or int(manifest.get("jet_count", -1))
            != int(view_manifest["identity_count"])
            or manifest["parents"]["hlt_content_sha256"]
            != view_manifest["npz_sha256"]
        ):
            raise ValueError("Stage-J tree manifest lineage differs")
        rows.append(
            {
                "coordinate": coordinate,
                "split": split,
                "view_manifest_sha256": view_manifest["content_hash"],
                "tree_manifest_path": str(manifest_path.resolve()),
                "tree_manifest_sha256": manifest["content_hash"],
                "jet_count": int(manifest["jet_count"]),
            }
        )
    completion = with_content_hash(
        {
            "contract": SCALE_TREE_WAVE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "source": campaign["source"],
            "campaign_spec_sha256": campaign["content_hash"],
            "scale_input_completion_sha256": input_completion["content_hash"],
            "rows": rows,
            "coordinate_count": len(rows),
            "coordinate_order": list(COORDINATES),
            "tree_resource_path": str(tree_resource_path.resolve()),
            "tree_resource_sha256": tree_resource_sha256,
            "backend_manifest_path": str(backend_manifest_path.resolve()),
            "backend_manifest_sha256": backend_manifest_sha256,
            "complete_exact_coverage": True,
        }
    )
    write_immutable_json(
        root / "scale_up" / "trees" / "completion.json", completion
    )
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--coordinate", required=True, choices=COORDINATES)
    parser.add_argument("--tree-resource", required=True, type=Path)
    parser.add_argument("--backend-manifest", required=True, type=Path)
    parser.add_argument("--backend-binary", required=True, type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    input_completion = load_hashed_json(
        root / "scale_up" / "inputs" / "completion.json",
        expected_contract=SCALE_INPUT_COMPLETION_CONTRACT,
    )
    resource = load_hashed_json(
        args.tree_resource, expected_contract=ANGULAR_TREE_RESOURCE_CONTRACT
    )
    backend_manifest = load_hashed_json(
        args.backend_manifest,
        expected_contract=ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    )
    if (
        input_completion.get("source") != campaign["source"]
        or resource.get("source") != campaign["source"]
        or backend_manifest.get("source") != campaign["source"]
    ):
        raise ValueError("Stage-J tree source lineage differs")
    input_path, output_dir, split = _paths(root, args.coordinate)
    view_manifest = load_hashed_json(
        input_path.with_suffix(input_path.suffix + ".json"),
        expected_contract=INPUT_VIEW_MANIFEST_CONTRACT,
    )
    if (
        view_manifest.get("source") != campaign["source"]
        or view_manifest.get("split") != "scale_train"
        or _sha256(input_path) != view_manifest.get("npz_sha256")
    ):
        raise ValueError("Stage-J tree input-view lineage differs")
    arrays, _ = load_materialized_input_view(
        input_path,
        expected_view_kind=(
            "canonical_offline"
            if args.coordinate == "offline"
            else "hlt_analogue"
        ),
        expected_source=campaign["source"],
    )
    identities = arrays["identities"]
    tokens = arrays["tokens"]
    mask = arrays["mask"]
    vectors = arrays["vectors"]
    identity_count = int(identities.shape[0])
    if (
        identity_count != int(view_manifest["identity_count"])
        or tokens.shape != (identity_count, 128, 14)
        or mask.shape != (identity_count, 128)
        or vectors.shape != (identity_count, 4, 128)
    ):
        raise ValueError("Stage-J tree input shapes differ")
    specifications = []
    for shard_index, start in enumerate(range(0, identity_count, SHARD_SIZE)):
        stop = min(start + SHARD_SIZE, identity_count)
        output = output_dir / "shards" / f"shard_{shard_index:05d}.npz"
        reused = validate_existing_tree_shard(
            output,
            identities[start:stop],
            hlt_content_sha256=view_manifest["npz_sha256"],
            tree_resource_sha256=resource["content_hash"],
            backend_manifest_sha256=backend_manifest["content_hash"],
            recover_unregistered_partial=True,
        )
        specifications.append(
            {
                "shard_index": shard_index,
                "start": start,
                "stop": stop,
                "output": output,
                "reused": reused is not None,
            }
        )
    pending = [row for row in specifications if not row["reused"]]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "coordinate": args.coordinate,
                    "jet_count": identity_count,
                    "shard_count": len(specifications),
                    "pending_shard_count": len(pending),
                },
                indent=2,
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
    backend = load_tree_backend(
        args.backend_binary, args.backend_manifest, source_path=source
    )
    workers = min(max(int(args.workers), 1), SHARD_SIZE)
    pool = None
    global _FORKED_BACKEND
    if workers > 1:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("parallel Stage-J tree building requires Linux fork")
        _FORKED_BACKEND = backend
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        )
    try:
        for position, row in enumerate(pending, start=1):
            start, stop = int(row["start"]), int(row["stop"])
            payloads = [
                (vectors[index].T, tokens[index], mask[index])
                for index in range(start, stop)
            ]
            trees = (
                [build_compiled_tree(backend, *payload) for payload in payloads]
                if pool is None
                else list(pool.map(_build_one, payloads, chunksize=8))
            )
            write_tree_shard(
                row["output"],
                trees,
                identities[start:stop],
                hlt_content_sha256=view_manifest["npz_sha256"],
                tree_resource_sha256=resource["content_hash"],
                backend_manifest_sha256=backend_manifest["content_hash"],
            )
            print(
                json.dumps(
                    {
                        "coordinate": args.coordinate,
                        "published_shard_count": position,
                        "total_pending_shards": len(pending),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        _FORKED_BACKEND = None
    shard_metadata = [
        output_dir / "shards" / f"shard_{index:05d}.metadata.json"
        for index in range(len(specifications))
    ]
    manifest = finalize_tree_split(
        output_dir / "manifest.json",
        shard_metadata,
        split=split,
        expected_jet_count=identity_count,
        hlt_content_sha256=view_manifest["npz_sha256"],
        tree_resource_sha256=resource["content_hash"],
        backend_manifest_sha256=backend_manifest["content_hash"],
    )
    wave = _close_wave(
        root,
        campaign,
        input_completion,
        tree_resource_path=args.tree_resource,
        tree_resource_sha256=resource["content_hash"],
        backend_manifest_path=args.backend_manifest,
        backend_manifest_sha256=backend_manifest["content_hash"],
    )
    print(
        json.dumps(
            {
                "coordinate": args.coordinate,
                "tree_manifest_sha256": manifest["content_hash"],
                "wave_complete": wave is not None,
                "wave_completion_sha256": (
                    None if wave is None else wave["content_hash"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

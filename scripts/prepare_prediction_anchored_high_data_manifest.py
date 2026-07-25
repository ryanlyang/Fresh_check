#!/usr/bin/env python3
"""Prepare the immutable 3M/3M bridge source and child manifests.

This command writes provenance and restart partitions only. Dense compressed
source caches are materialized later by the full submitter after a measured
free-space check; the partitions do not duplicate particle tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import load_split_manifest  # noqa: E402
from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    LOCKED_HIGH_DATA_3M_SPLIT_CONFIG,
    audit_child_split_manifest,
    build_child_split_manifest,
    canonical_sha256,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    sha256_file,
)


HIGH_DATA_PREPARATION_CONTRACT = (
    "prediction_anchored_high_data_3m_preparation_v1"
)
HIGH_DATA_STREAMING_PLAN_CONTRACT = (
    "prediction_anchored_high_data_3m_streaming_plan_v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-data-dir", required=True)
    parser.add_argument("--shard-events", type=int, default=100_000)
    return parser


def _child_shards(
    child: Mapping[str, Any], *, shard_events: int
) -> list[dict[str, Any]]:
    indices = child["parent_row_indices"]
    rows: list[dict[str, Any]] = []
    for ordinal, start in enumerate(range(0, len(indices), shard_events)):
        stop = min(start + shard_events, len(indices))
        rows.append(
            {
                "ordinal": int(ordinal),
                "child_row_start": int(start),
                "child_row_stop": int(stop),
                "count": int(stop - start),
                "parent_row_indices_sha256": canonical_sha256(indices[start:stop]),
            }
        )
    return rows


def _parent_shards(count: int, *, shard_events: int) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": int(ordinal),
            "parent_row_start": int(start),
            "parent_row_stop": int(min(start + shard_events, count)),
            "count": int(min(start + shard_events, count) - start),
        }
        for ordinal, start in enumerate(range(0, count, shard_events))
    ]


def build_streaming_plan(
    child_manifest: Mapping[str, Any],
    *,
    parent_manifest_file_sha256: str,
    source_data_dir: str,
    shard_events: int,
) -> dict[str, Any]:
    if int(shard_events) <= 0 or int(shard_events) % 10:
        raise ValueError("shard-events must be positive and divisible by ten classes")
    children = {
        name: {
            "parent_split": child["parent_split"],
            "child_manifest_sha256": child["content_hash"],
            "ordered_identity_sha256": child["ordered_identity_sha256"],
            "count": int(child["count"]),
            "shards": _child_shards(child, shard_events=int(shard_events)),
        }
        for name, child in sorted(child_manifest["children"].items())
    }
    counts = dict(LOCKED_HIGH_DATA_3M_SPLIT_CONFIG.parent_split_counts)
    parents = {
        name: {
            "count": int(counts[name]),
            "parent_split_order_sha256": child_manifest[
                "parent_split_order_sha256"
            ][name],
            "shards": _parent_shards(
                int(counts[name]), shard_events=int(shard_events)
            ),
        }
        for name in ("model_train", "final_test")
    }
    return with_content_hash(
        {
            "contract": HIGH_DATA_STREAMING_PLAN_CONTRACT,
            "profile": "high_data_3m",
            "parent_manifest_sha256": child_manifest["parent_manifest_sha256"],
            "parent_manifest_file_sha256": parent_manifest_file_sha256,
            "child_manifest_sha256": child_manifest["content_hash"],
            "split_config_sha256": child_manifest["split_config_sha256"],
            "source_data_dir": str(Path(source_data_dir)),
            "shard_events": int(shard_events),
            "children": children,
            "direct_parent_splits": parents,
            "dense_source_cache_policy": (
                "allowed_after_measured_projection_and_minimum_free_space_check"
            ),
            "final_test_offline_inputs_allowed": False,
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"high-data output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    parent_path = Path(args.parent_manifest).resolve()
    if parent_path.is_symlink() or not parent_path.is_file():
        raise FileNotFoundError(f"missing/unsafe parent manifest: {parent_path}")
    source_dir = Path(args.source_data_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source data directory does not exist: {source_dir}")

    parent = load_split_manifest(parent_path)
    child = build_child_split_manifest(
        parent, config=LOCKED_HIGH_DATA_3M_SPLIT_CONFIG
    )
    audit = audit_child_split_manifest(
        child,
        parent=parent,
        config=LOCKED_HIGH_DATA_3M_SPLIT_CONFIG,
    )
    child_path = output / "prediction_anchored_high_data_child_splits.json"
    write_immutable_json(child_path, child)

    parent_file_sha = sha256_file(parent_path)
    streaming = build_streaming_plan(
        child,
        parent_manifest_file_sha256=parent_file_sha,
        source_data_dir=str(source_dir),
        shard_events=int(args.shard_events),
    )
    streaming_path = output / "prediction_anchored_high_data_streaming_plan.json"
    write_immutable_json(streaming_path, streaming)

    preparation = with_content_hash(
        {
            "contract": HIGH_DATA_PREPARATION_CONTRACT,
            "status": "MANIFEST_AND_STREAMING_SHARDS_READY",
            "profile": "high_data_3m",
            "parent_manifest_path": str(parent_path),
            "parent_manifest_file_sha256": parent_file_sha,
            "parent_manifest_sha256": child["parent_manifest_sha256"],
            "child_manifest_path": str(child_path.resolve()),
            "child_manifest_sha256": child["content_hash"],
            "streaming_plan_path": str(streaming_path.resolve()),
            "streaming_plan_sha256": streaming["content_hash"],
            "split_config": LOCKED_HIGH_DATA_3M_SPLIT_CONFIG.to_payload(),
            "split_audit": audit,
            "requested_training_inventory": {
                "r0_disjoint_model_train": 500_000,
                "consumer": 3_000_000,
                "distill": 3_000_000,
                "validation_total": 1_000_000,
                "sealed_final_test": 1_000_000,
                "total_unique_parent_events": 8_500_000,
            },
            "validation_children": {
                "model_val_stop": 250_000,
                "model_val_select": 250_000,
                "stack_val_consumer": 250_000,
                "stack_val_deploy": 250_000,
            },
            "final_test_policy": "sealed_hlt_only_after_locked_deployable",
            "dense_npz_materialized": False,
            "persistent_storage_policy": (
                "manifest_only_until_dense_cache_projection_is_approved"
            ),
            "scientific_jobs_submitted": False,
        }
    )
    preparation_path = output / "prediction_anchored_high_data_preparation.json"
    write_immutable_json(preparation_path, preparation)
    print(
        json.dumps(
            {
                "ok": True,
                "status": preparation["status"],
                "child_manifest": str(child_path),
                "streaming_plan": str(streaming_path),
                "preparation": str(preparation_path),
                "dense_npz_materialized": False,
                "scientific_jobs_submitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

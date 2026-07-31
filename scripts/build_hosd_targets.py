#!/usr/bin/env python3
"""Publish one label-blind HOSD physical-target cache from an authenticated view."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    ExtractorResources,
    INPUT_VIEW_MANIFEST_CONTRACT,
    PHYSICAL_TARGET_IDS,
    STREAM_STORAGE_MODE,
    build_target_cache_spec,
    extract_registered_target,
    load_and_validate_campaign,
    load_hashed_json,
    publish_target_cache,
)
from teacher_logit_reco.hlt_offline_structure_distillation.extractors import (  # noqa: E402
    build_target_extractor_manifest,
)
from teacher_logit_reco.relational_part.ca_tree import unpack_tree_shard  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation.stage_b_runtime import (  # noqa: E402
    try_finalize_stage_b_wave,
)


def _hashes(values: list[str]) -> dict[str, str]:
    output = {}
    for value in values:
        name, separator, digest = value.partition("=")
        if not separator or len(digest) != 64:
            raise ValueError("--parent-hash entries must be NAME=SHA256")
        output[name] = digest
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--artifact-kind",
        required=True,
        choices=("canonical_offline", "hlt_analogue"),
    )
    parser.add_argument("--target-id", action="append", required=True)
    parser.add_argument("--parent-hash", action="append", default=[])
    parser.add_argument("--hlt-replica-id")
    parser.add_argument("--access-authorization-sha256")
    parser.add_argument("--relation-normalizer", type=Path)
    parser.add_argument("--tree-backend-manifest", type=Path)
    parser.add_argument("--tree-cache-dir", type=Path)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--cache-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    registry = load_hashed_json(
        args.campaign_root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    if registry.get("source") != campaign["source"]:
        raise ValueError("target registry source differs from active campaign")
    input_manifest = load_hashed_json(
        args.input_npz.with_suffix(args.input_npz.suffix + ".json"),
        expected_contract=INPUT_VIEW_MANIFEST_CONTRACT,
    )
    observed_input_sha = _sha256(args.input_npz)
    expected_replica = (
        None if args.hlt_replica_id is None else int(args.hlt_replica_id)
    )
    if (
        input_manifest.get("source") != campaign["source"]
        or input_manifest.get("split") != args.split
        or input_manifest.get("view_kind") != args.artifact_kind
        or input_manifest.get("replica_id") != expected_replica
        or input_manifest.get("npz_sha256") != observed_input_sha
        or input_manifest.get("contains_labels") is not False
    ):
        raise ValueError("target-builder input-view lineage differs")
    targets = tuple(args.target_id)
    if len(targets) != len(set(targets)) or any(
        target_id not in PHYSICAL_TARGET_IDS for target_id in targets
    ):
        raise ValueError("--target-id must contain unique Step-3 physical targets")
    registry_rows = {row["target_id"]: row for row in registry["targets"]}
    if any(not registry_rows[target_id]["executable_current_source"] for target_id in targets):
        raise ValueError("requested target is not executable for the current source")
    with np.load(args.input_npz, allow_pickle=False) as archive:
        forbidden = {
            name for name in archive.files
            if name.lower() in {"label", "labels", "class", "classes", "y"}
        }
        if forbidden:
            raise ValueError(
                f"label-blind target input contains forbidden fields: {sorted(forbidden)}"
            )
        required = {"identity", "raw_tokens", "mask"}
        if not required.issubset(archive.files):
            raise ValueError(f"input NPZ lacks fields {sorted(required - set(archive.files))}")
        identities = tuple(str(value) for value in archive["identity"].tolist())
        raw_tokens = np.asarray(archive["raw_tokens"], dtype=np.float32)
        mask = np.asarray(archive["mask"], dtype=bool)
        vectors = (
            np.asarray(archive["vectors"], dtype=np.float32)
            if "vectors" in archive.files
            else None
        )
    if len(identities) != raw_tokens.shape[0] or mask.shape[0] != len(identities):
        raise ValueError("input NPZ identity and tensor populations differ")
    relation = (
        load_hashed_json(args.relation_normalizer)
        if args.relation_normalizer is not None
        else None
    )
    if relation is not None and relation.get("source") != campaign["source"]:
        raise ValueError("relation normalizer source differs from active campaign")
    floors = (
        relation.get("track_uncertainty_floors", {})
        if relation is not None
        else {}
    )
    resources = ExtractorResources(
        d0_uncertainty_floor=float(floors.get("d0", {}).get("floor", 0.0)),
        dz_uncertainty_floor=float(floors.get("dz", {}).get("floor", 0.0)),
        sentinel_policy=(
            relation.get("track_sentinel_policy") if relation is not None else None
        ),
    )
    parent_hashes = _hashes(args.parent_hash)
    parent_hashes.update(
        {
            "campaign_spec": campaign["content_hash"],
            "target_registry": registry["content_hash"],
            "extractor_manifest": build_target_extractor_manifest()["content_hash"],
            "input_view": observed_input_sha,
            "input_view_manifest": input_manifest["content_hash"],
        }
    )
    if relation is not None:
        parent_hashes["relation_normalizer"] = relation["content_hash"]
    uses_tree = any(
        "CA_TREE" in target_id or "REGION" in target_id for target_id in targets
    )
    trees = None
    if uses_tree:
        if args.tree_backend_manifest is None or args.tree_cache_dir is None:
            raise ValueError(
                "tree-derived targets require --tree-backend-manifest and "
                "--tree-cache-dir"
            )
        tree_backend = load_hashed_json(args.tree_backend_manifest)
        if tree_backend.get("source") is not None and tree_backend.get(
            "source"
        ) != campaign["source"]:
            raise ValueError("tree backend source differs from active campaign")
        parent_hashes["tree_backend"] = tree_backend["content_hash"]
        tree_manifest = load_hashed_json(args.tree_cache_dir / "manifest.json")
        if tree_manifest.get("source") is not None and tree_manifest.get(
            "source"
        ) != campaign["source"]:
            raise ValueError("tree resource source differs from active campaign")
        parent_hashes["tree_resource"] = tree_manifest["content_hash"]
        tree_by_identity = {}
        for shard_path in sorted((args.tree_cache_dir / "shards").glob("shard_*.npz")):
            shard_identities, shard_trees = unpack_tree_shard(shard_path)
            for identity, tree in zip(shard_identities, shard_trees):
                if identity in tree_by_identity:
                    raise ValueError("tree cache contains duplicate identities")
                tree_by_identity[identity] = tree
        if set(tree_by_identity) != set(identities):
            raise ValueError("tree cache identity population differs from target input")
        trees = tuple(tree_by_identity[identity] for identity in identities)
    component_schema = {
        target_id: tuple(registry_rows[target_id]["component_names"])
        for target_id in targets
    }
    storage_modes = {
        target_id: (
            STREAM_STORAGE_MODE
            if registry_rows[target_id]["head_type"] == "pair"
            else "persist_compact_jet_target"
        )
        for target_id in targets
    }
    spec = build_target_cache_spec(
        cache_id=args.cache_id,
        split=args.split,
        artifact_kind=args.artifact_kind,
        identities=identities,
        target_components=component_schema,
        parent_hashes=parent_hashes,
        source=campaign["source"],
        shard_size=args.shard_size,
        storage_modes=storage_modes,
        hlt_replica_id=args.hlt_replica_id,
        access_authorization_hash=args.access_authorization_sha256,
    )

    persisted = tuple(spec["persisted_target_ids"])

    def generate(indices: np.ndarray):
        return {
            target_id: extract_registered_target(
                target_id,
                raw_tokens[indices],
                mask[indices],
                resources=resources,
                vectors=None if vectors is None else vectors[indices],
                trees=(
                    None if trees is None
                    else tuple(trees[int(index)] for index in indices)
                ),
            )
            for target_id in persisted
        }

    manifest = publish_target_cache(
        args.output_dir,
        cache_spec=spec,
        identities=identities,
        generator=generate,
    )
    wave = try_finalize_stage_b_wave(
        campaign_root=args.campaign_root,
        wave_kind=(
            "canonical"
            if args.artifact_kind == "canonical_offline"
            else "hlt_analogue"
        ),
        target_registry=registry,
        source=campaign["source"],
    )
    print(
        json.dumps(
            {
                "cache_spec_sha256": spec["content_hash"],
                "cache_manifest_sha256": manifest["content_hash"],
                "event_count": manifest["event_count"],
                "persisted_target_ids": manifest["persisted_target_ids"],
                "streamed_target_ids": manifest["streamed_target_ids"],
                "label_access_for_extraction": False,
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

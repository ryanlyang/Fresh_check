#!/usr/bin/env python3
"""Run the complete postconstruction provenance and input audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import load_split_manifest  # noqa: E402
from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    ANGULAR_TREE_PROBE_CONTRACT,
    ANGULAR_TREE_SHARD_CONTRACT,
    ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    POSTCONSTRUCTION_AUDIT_CONTRACT,
    PRECONSTRUCTION_AUDIT_CONTRACT,
    REGION_NORMALIZATION_CONTRACT,
    RELATIONAL_HLT_BINDING_CONTRACT,
    RELATIONAL_STORAGE_PROJECTION_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    bind_source_provenance,
    build_postconstruction_audit,
    load_hashed_json,
    sha256_file,
    source_snapshot,
    unpack_tree_shard,
    validate_content_hash,
    write_immutable_json,
)


_SPLITS = ("model_train", "model_val", "stack_val", "final_test")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preconstruction-audit", type=Path, required=True)
    parser.add_argument("--hlt-binding", type=Path, required=True)
    parser.add_argument("--relation-normalization", type=Path, required=True)
    parser.add_argument("--region-normalization", type=Path, required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--throughput-probe", type=Path, required=True)
    parser.add_argument(
        "--tree-split",
        action="append",
        required=True,
        metavar="SPLIT=DIR",
        help="Repeat exactly once for each nonempty split.",
    )
    parser.add_argument("--storage-projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _tree_dirs(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--tree-split must use SPLIT=DIR")
        split, raw_path = value.split("=", 1)
        if split not in _SPLITS or split in result:
            raise ValueError(f"invalid or duplicate tree split {split!r}")
        result[split] = Path(raw_path)
    if set(result) != set(_SPLITS):
        raise ValueError("--tree-split must cover every nonempty split")
    return result


def _audit_tree_split(
    *,
    split: str,
    tree_dir: Path,
    expected_identities: Sequence[str],
    expected_identity_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_hashed_json(
        tree_dir / "manifest.json",
        expected_contract=ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    )
    if manifest.get("split") != split:
        raise ValueError(f"{tree_dir} belongs to another split")
    identities: list[str] = []
    for expected_index, row in enumerate(manifest["shards"]):
        index = int(row["shard_index"])
        if index != expected_index:
            raise ValueError(f"{split} tree shard indices are not contiguous")
        npz_path = tree_dir / "shards" / f"shard_{index:05d}.npz"
        metadata_path = npz_path.with_suffix(".metadata.json")
        metadata = load_hashed_json(
            metadata_path, expected_contract=ANGULAR_TREE_SHARD_CONTRACT
        )
        if (
            metadata["content_hash"] != row["metadata_sha256"]
            or metadata["npz_sha256"] != row["npz_sha256"]
            or sha256_file(npz_path) != row["npz_sha256"]
            or int(metadata["jet_count"]) != int(row["jet_count"])
            or metadata["identity_sha256"] != row["identity_sha256"]
            or metadata["parents"] != manifest["parents"]
        ):
            raise ValueError(f"{split} tree shard {index} failed authentication")
        shard_identities, _ = unpack_tree_shard(npz_path)
        identities.extend(shard_identities)
    expected = list(expected_identities)
    if identities != expected:
        raise ValueError(f"{split} tree identity order differs from split manifest")
    duplicates = len(identities) - len(set(identities))
    report = {
        "split": split,
        "jet_count": len(identities),
        "shard_count": len(manifest["shards"]),
        "ordered_identity_sha256": expected_identity_sha256,
        "duplicate_identity_count": duplicates,
        "complete": (
            duplicates == 0
            and len(identities) == int(manifest["jet_count"])
        ),
        "event_level_output_persisted": False,
    }
    return manifest, report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    tree_dirs = _tree_dirs(args.tree_split)
    split_manifest = load_split_manifest(args.manifest)
    pre = load_hashed_json(
        args.preconstruction_audit,
        expected_contract=PRECONSTRUCTION_AUDIT_CONTRACT,
    )
    hlt = load_hashed_json(
        args.hlt_binding, expected_contract=RELATIONAL_HLT_BINDING_CONTRACT
    )
    relation = load_hashed_json(
        args.relation_normalization,
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    region = load_hashed_json(
        args.region_normalization,
        expected_contract=REGION_NORMALIZATION_CONTRACT,
    )
    backend = load_hashed_json(
        args.backend_manifest,
        expected_contract=ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    )
    probe = load_hashed_json(
        args.throughput_probe,
        expected_contract=ANGULAR_TREE_PROBE_CONTRACT,
    )
    storage = load_hashed_json(
        args.storage_projection,
        expected_contract=RELATIONAL_STORAGE_PROJECTION_CONTRACT,
    )
    tree_manifests = {}
    identity_audits = {}
    for split in _SPLITS:
        tree_manifests[split], identity_audits[split] = _audit_tree_split(
            split=split,
            tree_dir=tree_dirs[split],
            expected_identities=[
                identity.key() for identity in split_manifest.splits[split]
            ],
            expected_identity_sha256=hlt["split_reports"][split][
                "jet_identity_hash"
            ],
        )
    artifact = build_postconstruction_audit(
        preconstruction_audit=pre,
        hlt_binding=hlt,
        relation_normalization=relation,
        region_normalization=region,
        backend_manifest=backend,
        throughput_probe=probe,
        tree_manifests=tree_manifests,
        tree_identity_audits=identity_audits,
        storage_projection=storage,
    )
    if artifact["contract"] != POSTCONSTRUCTION_AUDIT_CONTRACT:
        raise AssertionError("postconstruction audit contract changed")
    artifact = bind_source_provenance(
        artifact, source_snapshot=source_snapshot(REPO_ROOT)
    )
    publication = None
    if not args.dry_run:
        publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "final_test_access": "provenance_only",
                "checkpoint_accessed": False,
                "inference_performed": False,
                "label_dependent_metric_computed": False,
                "artifact": artifact,
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

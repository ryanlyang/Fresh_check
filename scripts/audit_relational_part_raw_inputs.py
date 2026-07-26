#!/usr/bin/env python3
"""Run the immutable preconstruction raw-input audit for Step 8."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import (  # noqa: E402
    SPLIT_ORDER,
    SplitManifest,
    _resolve_data_file,
    load_offline_view,
    load_split_manifest,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    RAW_INPUT_SCHEMA_CONTRACT,
    bind_source_provenance,
    build_preconstruction_audit,
    load_hashed_json,
    select_raw_audit_identities,
    source_snapshot,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-input-schema", type=Path, required=True)
    parser.add_argument("--data-dir", nargs="+")
    parser.add_argument("--tree-name", default="tree")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--miniature", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _inventory(
    manifest: SplitManifest,
    *,
    data_dir: Any,
    tree_name: str,
    required_branches: Sequence[str],
) -> list[dict[str, Any]]:
    try:
        import uproot
    except ImportError as exc:  # pragma: no cover - compute environment
        raise ImportError("raw input audit requires uproot") from exc

    rows: list[dict[str, Any]] = []
    for record in sorted(manifest.file_records, key=lambda item: item.path):
        source = _resolve_data_file(data_dir, record.path)
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"raw source file is absent or unsafe: {source}")
        with uproot.open(source) as handle:
            if tree_name not in handle:
                raise KeyError(f"{source} lacks tree {tree_name!r}")
            tree = handle[tree_name]
            missing = [
                branch for branch in required_branches if branch not in tree.keys()
            ]
            if missing:
                raise KeyError(f"{source} lacks required branches {missing}")
            if int(tree.num_entries) != int(record.num_entries):
                raise ValueError(
                    f"{source} entry count changed: manifest={record.num_entries}, "
                    f"source={tree.num_entries}"
                )
            branches = {}
            for name in required_branches:
                branch = tree[name]
                typename = str(getattr(branch, "typename", ""))
                interpretation = str(getattr(branch, "interpretation", ""))
                lower = f"{typename} {interpretation}".lower()
                if not any(
                    token in lower
                    for token in ("float", "double", "int", "uint", "bool")
                ):
                    raise TypeError(f"{source}:{name} is not numeric: {lower}")
                if not any(
                    token in lower
                    for token in ("[]", "jagged", "asjagged", "listoffset")
                ):
                    raise TypeError(
                        f"{source}:{name} lacks a jagged particle axis: {lower}"
                    )
                branches[name] = {
                    "typename": typename,
                    "interpretation": interpretation,
                }
            rows.append(
                {
                    "path": record.path,
                    "resolved_path": str(source.resolve()),
                    "label": int(record.label),
                    "num_entries": int(tree.num_entries),
                    "required_branches": list(required_branches),
                    "branches": branches,
                    "shape_policy": "jagged_particle_axis",
                    "dtype_policy": "numeric",
                }
            )
    return rows


def _sample_manifest(
    manifest: SplitManifest,
    selected: dict[str, Any],
    split: str,
) -> SplitManifest:
    splits = {name: [] for name in SPLIT_ORDER}
    splits[split] = [
        manifest.splits[split][index]
        for index in selected["selected_indices"][split]
    ]
    sizes = {name: len(rows) for name, rows in splits.items()}
    return replace(manifest, splits=splits, split_sizes=sizes)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_split_manifest(args.manifest)
    schema = load_hashed_json(
        args.raw_input_schema,
        expected_contract=RAW_INPUT_SCHEMA_CONTRACT,
    )
    data_dir: Any = args.data_dir if args.data_dir is not None else manifest.data_dir
    selection = select_raw_audit_identities(
        manifest, miniature=bool(args.miniature)
    )
    inventory = _inventory(
        manifest,
        data_dir=data_dir,
        tree_name=args.tree_name,
        required_branches=schema["required_particle_branches"],
    )
    sampled_arrays = {}
    sampled_identity_hashes = {}
    for split in ("model_train", "model_val", "stack_val", "final_test"):
        subset = _sample_manifest(manifest, selection, split)
        view = load_offline_view(
            subset,
            split,
            data_dir=data_dir,
            tree_name=args.tree_name,
            verify_label_branches=True,
            read_chunk_size=int(args.read_chunk_size),
        )
        expected_keys = [
            manifest.splits[split][index].key()
            for index in selection["selected_indices"][split]
        ]
        if [identity.key() for identity in view.jet_ids] != expected_keys:
            raise ValueError(f"{split} raw-audit identity ordering changed")
        sampled_arrays[split] = {
            "tokens": view.tokens,
            "mask": view.mask,
            "labels": view.labels,
        }
        sampled_identity_hashes[split] = selection[
            "selected_identity_sha256"
        ][split]
    artifact = build_preconstruction_audit(
        manifest=manifest,
        raw_input_schema=schema,
        branch_inventory=inventory,
        sampled_arrays=sampled_arrays,
        miniature=bool(args.miniature),
    )
    if artifact["sample_selection"]["selected_identity_sha256"] != (
        sampled_identity_hashes
    ):
        raise AssertionError("raw audit sample identity binding changed")
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
                "resolved_configuration": {
                    "manifest": str(args.manifest.resolve()),
                    "raw_input_schema": str(args.raw_input_schema.resolve()),
                    "data_dir": (
                        list(data_dir)
                        if isinstance(data_dir, list)
                        else str(data_dir)
                    ),
                    "tree_name": args.tree_name,
                    "miniature": bool(args.miniature),
                    "output": str(args.output.resolve()),
                },
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

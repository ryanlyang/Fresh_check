#!/usr/bin/env python3
"""Fit the immutable non-tree relation normalizer from model_train HLT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from teacher_logit_reco.relational_part import (  # noqa: E402
    NORMALIZATION_CONTRACT,
    RELATIONAL_HLT_BINDING_CONTRACT,
    RELATION_FAMILY_REGISTRY_CONTRACT,
    RAW_INPUT_SCHEMA_CONTRACT,
    bind_source_provenance,
    fit_relation_normalization,
    load_hashed_json,
    source_snapshot,
    validate_content_hash,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--normalization-contract", type=Path, required=True)
    parser.add_argument("--relation-registry", type=Path, required=True)
    parser.add_argument("--raw-input-schema", type=Path, required=True)
    parser.add_argument("--hlt-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    normalization_contract = load_hashed_json(
        args.normalization_contract,
        expected_contract=NORMALIZATION_CONTRACT,
    )
    relation_registry = load_hashed_json(
        args.relation_registry,
        expected_contract=RELATION_FAMILY_REGISTRY_CONTRACT,
    )
    raw_input_schema = load_hashed_json(
        args.raw_input_schema,
        expected_contract=RAW_INPUT_SCHEMA_CONTRACT,
    )
    hlt_binding = load_hashed_json(
        args.hlt_binding,
        expected_contract=RELATIONAL_HLT_BINDING_CONTRACT,
    )
    hlt_binding_sha = validate_content_hash(hlt_binding)
    report = hlt_binding.get("split_reports", {}).get("model_train")
    if (
        hlt_binding.get("ok") is not True
        or not isinstance(report, dict)
        or int(report.get("n_jets", 0)) <= 0
    ):
        raise ValueError("HLT binding does not authenticate model_train")
    view = load_cached_hlt_view(args.cache_dir, "model_train", verify_hash=True)
    hlt_content_hash = str(report.get("hlt_content_hash"))
    if view.metadata.get("hlt_content_hash") != hlt_content_hash:
        raise ValueError("model_train HLT cache differs from the authenticated binding")
    if view.metadata.get("source_manifest_hash") != hlt_binding.get(
        "source_manifest_hash"
    ):
        raise ValueError("model_train HLT cache belongs to another split manifest")

    artifact = fit_relation_normalization(
        view.tokens,
        view.mask,
        view.jet_ids,
        normalization_contract=normalization_contract,
        relation_registry=relation_registry,
        raw_input_schema=raw_input_schema,
        hlt_binding_sha256=hlt_binding_sha,
        source_manifest_sha256=str(hlt_binding["source_manifest_hash"]),
        hlt_model_train_content_sha256=hlt_content_hash,
    )
    artifact = bind_source_provenance(
        artifact,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = None
    if not args.dry_run:
        publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "fit_split": "model_train",
                "final_test_accessed": False,
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

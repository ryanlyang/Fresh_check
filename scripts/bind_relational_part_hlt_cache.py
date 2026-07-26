#!/usr/bin/env python3
"""Authenticate fixed-HLT caches and publish the Step-8 HLT binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import load_split_manifest  # noqa: E402
from teacher_logit_reco.relational_part import (  # noqa: E402
    RELATIONAL_HLT_EXPECTATION_CONTRACT,
    RELATIONAL_SPLIT_BINDING_CONTRACT,
    bind_source_provenance,
    build_hlt_binding,
    load_hashed_json,
    source_snapshot,
    write_immutable_json,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-binding", type=Path, required=True)
    parser.add_argument("--hlt-expectation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_split_manifest(args.manifest)
    split_binding = load_hashed_json(
        args.split_binding,
        expected_contract=RELATIONAL_SPLIT_BINDING_CONTRACT,
    )
    expectation = load_hashed_json(
        args.hlt_expectation,
        expected_contract=RELATIONAL_HLT_EXPECTATION_CONTRACT,
    )
    artifact = build_hlt_binding(
        cache_dir=args.cache_dir,
        manifest=manifest,
        split_binding=split_binding,
        hlt_expectation=expectation,
    )
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
                "final_test_access": "sealed_preparation_only",
                "checkpoint_accessed": False,
                "inference_performed": False,
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

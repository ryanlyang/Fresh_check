#!/usr/bin/env python3
"""Materialize one real JetClass/HLT-v3 label-blind HOSD input view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
    materialize_hlt_input_view,
    materialize_offline_input_view,
)


def _pairs(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, digest = value.partition("=")
        if not separator or key in result:
            raise ValueError("parent hashes must be unique NAME=SHA256")
        result[key] = digest
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--kind", choices=("offline", "hlt"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--data-dir", action="append", default=[])
    parser.add_argument("--hlt-cache", type=Path)
    parser.add_argument("--replica-id", type=int)
    parser.add_argument("--parent-hash", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(
        args.campaign_root, repo_root=REPO_ROOT
    )
    parents = {
        "campaign_spec": campaign["content_hash"],
        **_pairs(args.parent_hash),
    }
    if args.kind == "offline":
        if args.split_manifest is None or args.hlt_cache is not None:
            raise ValueError("offline input view arguments differ")
        artifact = materialize_offline_input_view(
            split_manifest_path=args.split_manifest,
            split=args.split,
            data_dirs=args.data_dir or None,
            output=args.output,
            parent_hashes=parents,
            source=campaign["source"],
        )
    else:
        if args.hlt_cache is None or args.replica_id is None:
            raise ValueError("HLT input view arguments differ")
        artifact = materialize_hlt_input_view(
            hlt_cache_path=args.hlt_cache,
            split=args.split,
            replica_id=args.replica_id,
            output=args.output,
            parent_hashes=parents,
            source=campaign["source"],
        )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

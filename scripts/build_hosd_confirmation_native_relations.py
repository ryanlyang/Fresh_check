#!/usr/bin/env python3
"""Build the replica-0 design-confirm HLT-native relation target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_hosd_targets import main as build_targets  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
    materialize_native_relation_target,
    native_relation_target_ids,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--relation-normalizer", required=True, type=Path)
    parser.add_argument("--tree-backend-manifest", required=True, type=Path)
    parser.add_argument("--tree-cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    output = args.output_dir or (
        root / "targets" / "native_relations" / "design_confirm"
    )
    cache = output / "replica_0_cache"
    code = build_targets(
        [
            "--campaign-root",
            str(root),
            "--input-npz",
            str(args.input_npz),
            "--output-dir",
            str(cache),
            "--split",
            "design_confirm",
            "--artifact-kind",
            "hlt_analogue",
            "--hlt-replica-id",
            "0",
            "--relation-normalizer",
            str(args.relation_normalizer),
            "--tree-backend-manifest",
            str(args.tree_backend_manifest),
            "--tree-cache-dir",
            str(args.tree_cache_dir),
            "--cache-id",
            "design_confirm_native_relations_replica_0",
            *[
                item
                for target_id in native_relation_target_ids()
                for item in ("--target-id", target_id)
            ],
        ]
    )
    if code:
        raise RuntimeError("design-confirm native target-cache builder failed")
    artifact = materialize_native_relation_target(
        target_cache_root=cache,
        output_path=output / "replica_0.npz",
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
    )
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "output": str((output / "replica_0.npz").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

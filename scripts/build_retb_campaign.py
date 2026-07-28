#!/usr/bin/env python3
"""Build and publish the immutable RETB Step-1 campaign contracts."""

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
from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    RetbSplitConfig,
    build_step1_bundle,
    load_hashed_json,
    miniature_storage_measurements,
    publish_step1_bundle,
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.storage import (  # noqa: E402
    STORAGE_MEASUREMENTS_CONTRACT,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--storage-measurements", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--miniature",
        action="store_true",
        help="Use the non-scientific miniature identity/count profile.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = RetbSplitConfig.miniature() if args.miniature else RetbSplitConfig.production()
    if args.storage_measurements is None:
        if not args.miniature:
            raise ValueError("production campaign requires --storage-measurements")
        measurements = miniature_storage_measurements()
    else:
        measurements = load_hashed_json(
            args.storage_measurements,
            expected_contract=STORAGE_MEASUREMENTS_CONTRACT,
        )
    manifest = load_split_manifest(args.parent_manifest)
    snapshot = source_snapshot(REPO_ROOT)
    bundle = build_step1_bundle(
        campaign_id=args.campaign_id,
        manifest=manifest,
        split_config=config,
        source_snapshot=snapshot,
        storage_measurements=measurements,
    )
    result: dict[str, object] = {
        "campaign_id": args.campaign_id,
        "campaign_profile": config.profile,
        "campaign_spec_sha256": bundle["campaign_spec"]["content_hash"],
        "source": bundle["campaign_spec"]["source"],
        "scale_train_count": bundle["scale_train_manifest"]["count"],
        "registry_count": len(bundle["registries"]),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        result["publication"] = publish_step1_bundle(
            campaign_root=args.output_dir,
            manifest=manifest,
            bundle=bundle,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

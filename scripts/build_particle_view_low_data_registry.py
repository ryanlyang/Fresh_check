#!/usr/bin/env python3
"""Build the authoritative 500k particle-view pilot registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    PARTICLE_VIEW_LOW_DATA_CAMPAIGN_ID,
    build_low_data_campaign_inventory,
    build_low_data_campaign_registry,
    load_hashed_json,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inventory-output", required=True)
    parser.add_argument(
        "--existing-teacher-compatible",
        action="store_true",
        help="Allow the existing-offline-teacher target to enter selection.",
    )
    parser.add_argument(
        "--teacher-mix-compatible",
        action="store_true",
        help="Allow the two-teacher mixture target to enter selection.",
    )
    parser.add_argument(
        "--campaign-id",
        default=PARTICLE_VIEW_LOW_DATA_CAMPAIGN_ID,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    unified = load_hashed_json(args.unified_manifest)
    registry = build_low_data_campaign_registry(
        unified_split_manifest=unified,
        existing_teacher_compatible=args.existing_teacher_compatible,
        teacher_mix_compatible=args.teacher_mix_compatible,
        campaign_id=args.campaign_id,
    )
    inventory = build_low_data_campaign_inventory(registry)
    if not args.dry_run:
        write_immutable_json(args.output, registry)
        write_immutable_json(args.inventory_output, inventory)
    print(
        json.dumps(
            {
                "campaign_id": registry["campaign_id"],
                "registry_sha256": registry["content_hash"],
                "inventory_sha256": inventory["content_hash"],
                "declared_run_count": inventory["declared_run_count"],
                "seed_expanded_replica_count": inventory[
                    "seed_expanded_replica_count"
                ],
                "category_counts": inventory["category_counts"],
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

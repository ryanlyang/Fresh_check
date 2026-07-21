#!/usr/bin/env python3
"""Write one immutable virtual prediction-anchored bridge recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge import build_bridge_recipe  # noqa: E402
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import write_immutable_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rho", default="0.100")
    parser.add_argument("--channel-policy", choices=("physical45", "all50"), default="physical45")
    parser.add_argument("--r0-checkpoint-sha256", required=True)
    parser.add_argument("--hlt-source-sha256", required=True)
    parser.add_argument("--offline-source-sha256", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--target-schema-sha256", required=True)
    parser.add_argument("--preprocessing-sha256", required=True)
    parser.add_argument("--event-order-sha256", required=True)
    parser.add_argument("--control-type", default="")
    parser.add_argument("--control-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recipe = build_bridge_recipe(
        rho=args.rho,
        channel_policy=args.channel_policy,
        r0_checkpoint_sha256=args.r0_checkpoint_sha256,
        hlt_source_sha256=args.hlt_source_sha256,
        offline_source_sha256=args.offline_source_sha256,
        split_manifest_sha256=args.split_manifest_sha256,
        target_schema_sha256=args.target_schema_sha256,
        preprocessing_sha256=args.preprocessing_sha256,
        event_order_sha256=args.event_order_sha256,
        control_type=args.control_type or None,
        control_seed=args.control_seed,
    )
    if not args.dry_run:
        write_immutable_json(args.output, recipe)
    print(json.dumps({"dry_run": bool(args.dry_run), "output": args.output, "recipe": recipe}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

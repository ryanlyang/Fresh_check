#!/usr/bin/env python3
"""Build the immutable Step-7 target/loss/consumer interaction campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_target_loss_interaction_campaign,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", action="append", required=True)
    parser.add_argument("--canonical-target-id", required=True)
    parser.add_argument("--alternate-target-id", required=True)
    parser.add_argument(
        "--canonical-architecture-id",
        default="P_HIER_DECODER_REFINE",
    )
    parser.add_argument("--plain-architecture-id", default="P_PART_BASIC")
    parser.add_argument("--clean-consumer-id", default="C_CLEAN")
    parser.add_argument("--robust-consumer-id", default="C_ROBUST_MIX")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = build_target_loss_interaction_campaign(
        target_ids=args.target_id,
        canonical_target_id=args.canonical_target_id,
        alternate_target_id=args.alternate_target_id,
        canonical_architecture_id=args.canonical_architecture_id,
        plain_architecture_id=args.plain_architecture_id,
        clean_consumer_id=args.clean_consumer_id,
        robust_consumer_id=args.robust_consumer_id,
    )
    write_immutable_json(args.output, campaign)
    print(
        f"rows={campaign['row_count']} "
        f"content_hash={campaign['content_hash']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


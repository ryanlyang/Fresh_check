#!/usr/bin/env python3
"""Publish one explicit authorization for a claimed incomplete PV10 access."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_final_recovery_authorization,
    load_hashed_json,
    write_immutable_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-claim", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--previous-consumption")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = build_final_recovery_authorization(
        access_claim=load_hashed_json(args.access_claim),
        reason=args.reason,
        authorized_by=args.authorized_by,
        previous_recovery_consumption=(
            load_hashed_json(args.previous_consumption)
            if args.previous_consumption
            else None
        ),
    )
    write_immutable_json(args.output, artifact)
    print(f"content_hash={artifact['content_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

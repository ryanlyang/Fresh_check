#!/usr/bin/env python3
"""Audit shared 3M sources and freeze the matched A0/P7b study manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.high_data_seed_study import (  # noqa: E402
    build_high_data_study_manifest,
    save_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--reference-curriculum-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_high_data_study_manifest(
        campaign_id=args.campaign_id,
        campaign_root=args.campaign_root,
        reference_curriculum_root=args.reference_curriculum_root,
    )
    save_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

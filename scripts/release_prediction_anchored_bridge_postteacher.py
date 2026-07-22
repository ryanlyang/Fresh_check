#!/usr/bin/env python3
"""Validate and publish the explicit B5-to-B6 post-teacher release gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--selected-consumer", required=True)
    parser.add_argument("--primary-binding", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from teacher_logit_reco.local_particle_residual_field.bridge_semantic_evidence import (
        require_post_teacher_release,
    )
    from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
        load_hashed_json,
        write_immutable_json,
    )
    artifact = require_post_teacher_release(
        load_hashed_json(args.registry),
        selected_consumer=load_hashed_json(args.selected_consumer),
        primary_binding=load_hashed_json(args.primary_binding),
    )
    publication = None
    if not args.dry_run:
        if not args.output:
            raise ValueError("--output is required unless --dry-run is used")
        publication = write_immutable_json(args.output, artifact)
    print(json.dumps({"dry_run": bool(args.dry_run), "release": artifact,
                      "publication": publication}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

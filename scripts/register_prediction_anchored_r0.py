#!/usr/bin/env python3
"""Register an existing ordinary residual predictor as the frozen bridge R0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge_ram import (  # noqa: E402
    build_frozen_r0_registration,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    write_immutable_json,
)


def _json_object(value: str, *, name: str) -> dict:
    stripped = str(value).strip()
    if stripped.startswith("{"):
        payload = json.loads(stripped)
    else:
        path = Path(stripped)
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must resolve to a JSON object")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preprocessing-sha256", required=True)
    parser.add_argument("--target-schema-sha256", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--matching-policy", required=True, help="JSON text or JSON file")
    parser.add_argument("--validation-metrics", required=True, help="JSON text or JSON file")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registration = build_frozen_r0_registration(
        args.checkpoint,
        preprocessing_sha256=args.preprocessing_sha256,
        target_schema_sha256=args.target_schema_sha256,
        split_manifest_sha256=args.split_manifest_sha256,
        matching_policy=_json_object(args.matching_policy, name="matching-policy"),
        validation_metrics=_json_object(args.validation_metrics, name="validation-metrics"),
    )
    if not args.dry_run:
        write_immutable_json(args.output, registration)
    print(
        json.dumps(
            {"dry_run": bool(args.dry_run), "output": args.output, "registration": registration},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

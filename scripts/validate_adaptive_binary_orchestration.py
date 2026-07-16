#!/usr/bin/env python3
"""Validate ABPH campaign gates and freeze immutable final-claim membership."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.fusion import (  # noqa: E402
    load_frozen_fusion_artifact,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (  # noqa: E402
    freeze_final_claim_contract,
    load_final_claim_contract,
    require_actual_target_preflight,
    require_successful_selection_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--path", required=True)
    selection = commands.add_parser("selection-report")
    selection.add_argument("--path", required=True)
    freeze = commands.add_parser("freeze-final-claim")
    freeze.add_argument("--selection-report", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--claim-variant", action="append", required=True)
    freeze.add_argument(
        "--fusion-artifact",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
    )
    claim = commands.add_parser("final-claim")
    claim.add_argument("--path", required=True)
    claim.add_argument("--selection-report", required=True)
    claim.add_argument("--member")
    claim.add_argument("--fusion-artifact", metavar="VARIANT=PATH")
    return parser


def _fusion_artifacts(values: list[str]) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    hashes: dict[str, str] = {}
    memberships: dict[str, tuple[str, ...]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--fusion-artifact expects VARIANT=PATH")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        artifact = load_frozen_fusion_artifact(path)
        if artifact.fusion_variant != name:
            raise ValueError(f"{path} belongs to {artifact.fusion_variant}, not {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        memberships[name] = artifact.members
    return hashes, memberships


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        payload = require_actual_target_preflight(args.path)
    elif args.command == "selection-report":
        payload = require_successful_selection_report(args.path)
    elif args.command == "freeze-final-claim":
        hashes, memberships = _fusion_artifacts(args.fusion_artifact)
        payload = freeze_final_claim_contract(
            args.selection_report,
            args.output,
            claim_variants=args.claim_variant,
            fusion_artifact_hashes=hashes,
            fusion_memberships=memberships,
        )
    else:
        payload = load_final_claim_contract(
            args.path,
            selection_report_path=args.selection_report,
        )
        claims = tuple(str(name) for name in payload["claim_variants"])
        memberships = {
            str(name): tuple(str(member) for member in members)
            for name, members in dict(payload["fusion_memberships"]).items()
        }
        eligible_members = set(claims)
        eligible_members.update(member for members in memberships.values() for member in members)
        if args.member and args.member not in eligible_members:
            raise ValueError(f"{args.member} is not frozen into the final-claim contract")
        if args.fusion_artifact:
            hashes, actual_memberships = _fusion_artifacts([args.fusion_artifact])
            name = next(iter(hashes))
            if name not in claims:
                raise ValueError(f"{name} is not a frozen final-claim fusion")
            if hashes[name] != payload["fusion_artifact_hashes"].get(name):
                raise ValueError(f"{name} fusion artifact bytes changed after claim freeze")
            if list(actual_memberships[name]) != list(payload["fusion_memberships"].get(name, ())):
                raise ValueError(f"{name} fusion membership changed after claim freeze")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

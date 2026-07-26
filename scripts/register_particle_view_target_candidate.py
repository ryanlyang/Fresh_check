#!/usr/bin/env python3
"""Publish one immutable Step-3 particle-view target registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_target_candidate_registration,
    particle_view_generator_config_from_payload,
    validate_target_candidate_registration,
    write_immutable_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--selection-status", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generator-config", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--unified-split-manifest-sha256", required=True)
    parser.add_argument("--train-split-sha256", required=True)
    parser.add_argument("--train-identity-sha256", required=True)
    parser.add_argument("--query-tap-registration-sha256", required=True)
    parser.add_argument("--query-checkpoint-sha256", required=True)
    parser.add_argument("--memory-tap-registration-sha256", required=True)
    parser.add_argument("--memory-checkpoint-sha256", required=True)
    parser.add_argument("--staged-tap-source-role", required=True)
    parser.add_argument("--staged-tap-reservation-sha256", required=True)
    parser.add_argument("--staged-tap-manifest-sha256", required=True)
    parser.add_argument("--staged-tap-logical-content-sha256", required=True)
    parser.add_argument("--generator-checkpoint-sha256", required=True)
    parser.add_argument("--offline-source-sha256")
    parser.add_argument(
        "--privileged-claim-eligible", action="store_true"
    )
    parser.add_argument(
        "--deployment-control-eligible", action="store_true"
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_payload = json.loads(
        Path(args.generator_config).read_text(encoding="utf-8")
    )
    if not isinstance(config_payload, dict):
        raise ValueError("generator config must be a JSON object")
    config = particle_view_generator_config_from_payload(config_payload)
    registration = build_target_candidate_registration(
        target_id=args.target_id,
        campaign_id=args.campaign_id,
        selection_status=args.selection_status,
        seed=args.seed,
        generator_config=config,
        source_manifest_sha256=args.source_manifest_sha256,
        unified_split_manifest_sha256=args.unified_split_manifest_sha256,
        train_split_sha256=args.train_split_sha256,
        train_identity_sha256=args.train_identity_sha256,
        query_tap_registration_sha256=args.query_tap_registration_sha256,
        query_checkpoint_sha256=args.query_checkpoint_sha256,
        memory_tap_registration_sha256=args.memory_tap_registration_sha256,
        memory_checkpoint_sha256=args.memory_checkpoint_sha256,
        staged_tap_source_role=args.staged_tap_source_role,
        staged_tap_reservation_sha256=args.staged_tap_reservation_sha256,
        staged_tap_manifest_sha256=args.staged_tap_manifest_sha256,
        staged_tap_logical_content_sha256=(
            args.staged_tap_logical_content_sha256
        ),
        generator_checkpoint_sha256=args.generator_checkpoint_sha256,
        offline_source_sha256=args.offline_source_sha256,
        privileged_claim_eligible=args.privileged_claim_eligible,
        deployment_control_eligible=args.deployment_control_eligible,
    )
    validate_target_candidate_registration(registration)
    publication = write_immutable_json(args.output, registration)
    print(json.dumps(publication, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

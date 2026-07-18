#!/usr/bin/env python3
"""Manage ABPH storage-wave manifests, receipts, barriers, and audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.storage_lifecycle import (  # noqa: E402
    build_artifact_manifest,
    execute_cleanup_barrier,
    write_consumer_receipt,
    write_wave_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--storage-profile", default="streaming_30gb_v1")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest")
    manifest.add_argument("--data-dir", required=True)

    receipt = commands.add_parser("consumer-receipt")
    receipt.add_argument("--consumer", required=True)
    receipt.add_argument("--run-report", required=True)
    receipt.add_argument(
        "--consumer-kind",
        choices=("target_consumer", "offline_consumer"),
        default="target_consumer",
    )

    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--barrier", choices=("privileged", "deployable"), required=True)
    cleanup.add_argument("--expected-consumer", action="append", default=[])
    cleanup.add_argument("--scoring-member", action="append", default=[])

    wave = commands.add_parser("wave-receipt")
    wave.add_argument("--wave", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "campaign_root": args.campaign_root,
        "profile": args.storage_profile,
    }
    if args.command == "manifest":
        payload = build_artifact_manifest(
            **common,
            data_dir=args.data_dir,
        )
    elif args.command == "consumer-receipt":
        payload = write_consumer_receipt(
            **common,
            consumer=args.consumer,
            run_report=args.run_report,
            consumer_kind=args.consumer_kind,
        )
    elif args.command == "cleanup":
        payload = execute_cleanup_barrier(
            **common,
            barrier=args.barrier,
            expected_consumers=tuple(args.expected_consumer),
            scoring_members=tuple(args.scoring_member),
        )
    elif args.command == "wave-receipt":
        payload = write_wave_receipt(**common, wave=args.wave)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

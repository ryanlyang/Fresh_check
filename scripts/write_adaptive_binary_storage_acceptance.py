#!/usr/bin/env python3
"""Compile the real-data Step-10 storage acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.storage_acceptance import (  # noqa: E402
    build_storage_acceptance,
    write_storage_acceptance,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--campaign-mode", choices=("pilot", "highdata"), required=True)
    parser.add_argument("--storage-projection", required=True)
    parser.add_argument("--target-mode-selection", required=True)
    parser.add_argument("--target-feasibility", required=True)
    parser.add_argument("--wave-two-audit", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--runtime-acceptance", required=True)
    parser.add_argument("--test-evidence", required=True)
    parser.add_argument("--ram-lifecycle-smoke", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--source-status-hash", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_storage_acceptance(
        campaign_root=args.campaign_root,
        campaign_mode=args.campaign_mode,
        storage_projection=args.storage_projection,
        target_mode_selection=args.target_mode_selection,
        target_feasibility=args.target_feasibility,
        wave_two_audit=args.wave_two_audit,
        artifact_manifest=args.artifact_manifest,
        runtime_acceptance=args.runtime_acceptance,
        test_evidence=args.test_evidence,
        ram_lifecycle_smoke=args.ram_lifecycle_smoke,
        source_git_commit=args.source_git_commit,
        source_status_hash=args.source_status_hash,
    )
    write_storage_acceptance(args.campaign_root, args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

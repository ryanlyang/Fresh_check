#!/usr/bin/env python3
"""Audit and freeze the A0/P7b source artifacts used by the fusion campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    FusionSourceArtifactAuditConfig,
    audit_fusion_source_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--a0-report", required=True)
    parser.add_argument("--p7b-checkpoint", required=True)
    parser.add_argument("--p7b-report", required=True)
    parser.add_argument("--selected-consumer-json", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--hlt-cache-manifest", required=True)
    parser.add_argument("--a0-prediction-dir", required=True)
    parser.add_argument("--p7b-prediction-dir", required=True)
    parser.add_argument("--no-verify-prediction-hash", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_fusion_source_artifacts(
        FusionSourceArtifactAuditConfig(
            output_path=args.output,
            a0_checkpoint=args.a0_checkpoint,
            a0_report=args.a0_report,
            p7b_checkpoint=args.p7b_checkpoint,
            p7b_report=args.p7b_report,
            selected_consumer_json=args.selected_consumer_json,
            manifest_path=args.manifest_path,
            hlt_cache_manifest=args.hlt_cache_manifest,
            a0_prediction_dir=args.a0_prediction_dir,
            p7b_prediction_dir=args.p7b_prediction_dir,
            verify_prediction_hash=not bool(args.no_verify_prediction_hash),
            overwrite=bool(args.overwrite),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

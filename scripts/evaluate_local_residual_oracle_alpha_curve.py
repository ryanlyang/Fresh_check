#!/usr/bin/env python3
"""Evaluate a fixed Ofull/Orobust consumer over the first-stage alpha ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.curriculum_campaign import (  # noqa: E402
    evaluate_fixed_consumer_alpha_curve,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--consumer-id", required=True, choices=("Ofull", "Orobust_light"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--baseline-report", default="")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.10, 0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--splits", nargs="+", default=["model_val", "stack_val"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-jets", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--allow-missing-manifest-match", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_fixed_consumer_alpha_curve(
        checkpoint=args.checkpoint,
        consumer_id=args.consumer_id,
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        target_cache_dir=args.target_cache_dir,
        manifest_path=args.manifest_path or None,
        baseline_report=args.baseline_report or None,
        alphas=tuple(args.alphas),
        splits=tuple(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_jets=args.max_jets,
        device=args.device,
        amp=not args.disable_amp,
        verify_hash=not args.no_verify_hash,
        require_manifest_match=not args.allow_missing_manifest_match,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

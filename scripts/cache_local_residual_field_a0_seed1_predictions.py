#!/usr/bin/env python3
"""Cache hash-gated stack-only predictions for the independent A0 seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    FUSION_DEVELOPMENT_SPLITS,
    FUSION_MEMBER_A0_SEED1,
    LocalResidualFieldPredictionConfig,
    cache_a0_seed1_development_predictions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact-audit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--max-jets", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = cache_a0_seed1_development_predictions(
        LocalResidualFieldPredictionConfig(
            checkpoint=args.checkpoint,
            prediction_dir=args.prediction_dir,
            model_name=FUSION_MEMBER_A0_SEED1,
            hlt_cache_dir=args.hlt_cache_dir,
            target_cache_dir=None,
            manifest_path=args.manifest_path,
            splits=FUSION_DEVELOPMENT_SPLITS,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            amp=not bool(args.disable_amp),
            max_jets=args.max_jets,
            confirm_final_test=False,
            allow_oracle_final_test=False,
            overwrite=bool(args.overwrite),
        ),
        source_artifact_audit=args.source_artifact_audit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

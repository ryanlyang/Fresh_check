#!/usr/bin/env python3
"""Cache strict stack-only pre-classifier features for one fusion member."""

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
    FusionFeatureCacheConfig,
    cache_local_residual_field_fusion_features,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--member-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-sources", required=True)
    parser.add_argument("--source-artifact-audit", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = cache_local_residual_field_fusion_features(
        FusionFeatureCacheConfig(
            checkpoint=args.checkpoint,
            member_id=args.member_id,
            output_dir=args.output_dir,
            prediction_sources=args.prediction_sources,
            source_artifact_audit=args.source_artifact_audit,
            hlt_cache_dir=args.hlt_cache_dir,
            manifest_path=args.manifest_path,
            splits=FUSION_DEVELOPMENT_SPLITS,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            amp=not bool(args.disable_amp),
            storage_dtype=args.storage_dtype,
            overwrite=bool(args.overwrite),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

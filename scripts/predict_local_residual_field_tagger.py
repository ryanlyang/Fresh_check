#!/usr/bin/env python3
"""Cache prediction logits from a trained local residual-field tagger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    LocalResidualFieldPredictionConfig,
    cache_local_residual_field_tagger_predictions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument(
        "--target-cache-dir",
        default="",
        help="Required only for oracle or other target-dependent legacy taggers; curriculum checkpoints are HLT-only.",
    )
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--splits", nargs="+", default=["stack_train", "stack_val", "final_test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--max-jets", type=int, default=None)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--allow-oracle-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--allow-missing-manifest-match", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = cache_local_residual_field_tagger_predictions(
        LocalResidualFieldPredictionConfig(
            checkpoint=args.checkpoint,
            prediction_dir=args.prediction_dir,
            model_name=args.model_name,
            hlt_cache_dir=args.hlt_cache_dir,
            target_cache_dir=args.target_cache_dir or None,
            manifest_path=args.manifest_path or None,
            splits=tuple(args.splits),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
            amp=not bool(args.disable_amp),
            max_jets=args.max_jets,
            confirm_final_test=bool(args.confirm_final_test),
            allow_oracle_final_test=bool(args.allow_oracle_final_test),
            overwrite=bool(args.overwrite),
            verify_hash=not bool(args.no_verify_hash),
            require_manifest_match=not bool(args.allow_missing_manifest_match),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

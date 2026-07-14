#!/usr/bin/env python3
"""Cache HLT-only predictions and Step 9 diagnostics from a selected tagger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine import (  # noqa: E402
    DEFAULT_PREDICTION_SPLITS,
    EndToEndPredictionConfig,
    cache_end_to_end_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--manifest", dest="manifest_path", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--checkpoint", dest="checkpoint_path", required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_PREDICTION_SPLITS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--max-jets-per-split", type=int, default=None)
    parser.add_argument("--d8-view-ablation-max-jets", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=29117)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = cache_end_to_end_predictions(
        EndToEndPredictionConfig(
            prediction_dir=args.prediction_dir,
            model_name=args.model_name,
            manifest_path=args.manifest_path,
            hlt_cache_dir=args.hlt_cache_dir,
            checkpoint_path=args.checkpoint_path,
            splits=tuple(args.splits),
            batch_size=args.batch_size,
            device=args.device,
            amp=not args.no_amp,
            verify_hash=not args.no_verify_hash,
            overwrite=args.overwrite,
            confirm_final_test=args.confirm_final_test,
            max_jets_per_split=args.max_jets_per_split,
            d8_view_ablation_max_jets=args.d8_view_ablation_max_jets,
            seed=args.seed,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

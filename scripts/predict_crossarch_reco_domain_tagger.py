#!/usr/bin/env python3
"""Write prediction blocks for one reco-domain tagger source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import STACK_SPLITS  # noqa: E402
from teacher_logit_reco.crossarch_experiment import (  # noqa: E402
    RECONSTRUCTOR_ARCHITECTURES,
    TEACHER_ARCHITECTURES,
    CrossArchExperimentLayout,
)
from teacher_logit_reco.crossarch_reco_domain_taggers import (  # noqa: E402
    CrossArchRecoDomainTaggerPredictionConfig,
    collect_crossarch_reco_domain_tagger_predictions,
)


def parse_args() -> argparse.Namespace:
    layout = CrossArchExperimentLayout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reco-architecture", choices=RECONSTRUCTOR_ARCHITECTURES, required=True)
    parser.add_argument("--teacher-architecture", choices=TEACHER_ARCHITECTURES, required=True)
    parser.add_argument("--reconstructor-checkpoint", required=True)
    parser.add_argument("--tagger-checkpoint", required=True)
    parser.add_argument("--cache-dir", default=str(layout.hlt_cache_dir))
    parser.add_argument("--prediction-dir", default=str(layout.predictions_dir))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--splits", nargs="+", choices=STACK_SPLITS, default=list(STACK_SPLITS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--stack-train-size", type=int, default=500_000)
    parser.add_argument("--stack-val-size", type=int, default=150_000)
    parser.add_argument("--final-test-size", type=int, default=500_000)
    parser.add_argument("--max-jets-per-split", type=int, default=None)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--no-skip-existing-predictions", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--control-seed", type=int, default=12345)
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--teacher-weight-threshold", type=float, default=0.0)
    parser.add_argument("--non-strict-reconstructor-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CrossArchRecoDomainTaggerPredictionConfig(
        reco_architecture=args.reco_architecture,
        teacher_architecture=args.teacher_architecture,
        reconstructor_checkpoint=args.reconstructor_checkpoint,
        tagger_checkpoint=args.tagger_checkpoint,
        cache_dir=args.cache_dir,
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        splits=list(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
        stack_train_size=args.stack_train_size,
        stack_val_size=args.stack_val_size,
        final_test_size=args.final_test_size,
        max_jets_per_split=args.max_jets_per_split,
        overwrite_predictions=bool(args.overwrite_predictions),
        skip_existing_predictions=not bool(args.no_skip_existing_predictions),
        confirm_final_test=bool(args.confirm_final_test),
        control_seed=args.control_seed,
        max_constits=args.max_constits,
        teacher_weight_threshold=args.teacher_weight_threshold,
        strict_reconstructor_checkpoint=not args.non_strict_reconstructor_checkpoint,
    )
    report = collect_crossarch_reco_domain_tagger_predictions(config)
    print("crossarch_reco_domain_tagger_prediction_complete:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

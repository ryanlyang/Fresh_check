#!/usr/bin/env python3
"""Generate Step 6 prediction blocks from a trained teacher-logit reconstructor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import STACK_SPLITS  # noqa: E402
from teacher_logit_reco.predict_global_transformer import (  # noqa: E402
    TeacherLogitGlobalTransformerPredictionConfig,
    collect_teacher_logit_global_transformer_predictions,
)
from teacher_logit_reco.teachers import TEACHER_ARCHITECTURES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--reconstructor-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--teacher-architecture", choices=TEACHER_ARCHITECTURES, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--splits", nargs="+", default=list(STACK_SPLITS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-jets-per-split", type=int, default=None)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--no-skip-existing-predictions", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--teacher-weight-threshold", type=float, default=0.0)
    parser.add_argument("--non-strict-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TeacherLogitGlobalTransformerPredictionConfig(
        output_dir=args.output_dir,
        prediction_dir=args.prediction_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        reconstructor_checkpoint=args.reconstructor_checkpoint,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_architecture=args.teacher_architecture,
        model_name=args.model_name,
        splits=list(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=not bool(args.no_amp),
        max_jets_per_split=args.max_jets_per_split,
        overwrite_predictions=bool(args.overwrite_predictions),
        skip_existing_predictions=not bool(args.no_skip_existing_predictions),
        confirm_final_test=bool(args.confirm_final_test),
        max_constits=args.max_constits,
        teacher_weight_threshold=args.teacher_weight_threshold,
        strict_checkpoint=not bool(args.non_strict_checkpoint),
    )
    report = collect_teacher_logit_global_transformer_predictions(config)
    print("teacher_logit_global_transformer_predictions_complete:")
    print(f"  prediction_dir: {report['prediction_dir']}")
    print(f"  model_name: {report['model_name']}")
    for split in report["splits"]:
        meta = report["reports"][report["model_name"]][split]
        metrics = meta.get("metrics", {})
        print(
            f"  {split}: n={meta.get('n_jets')} "
            f"acc={metrics.get('accuracy')} ce={metrics.get('cross_entropy')}"
        )
    print(f"  report: {Path(args.output_dir) / 'prediction_collection_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

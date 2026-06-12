#!/usr/bin/env python3
"""Train one reco-domain tagger behind a frozen crossarch reconstructor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.crossarch_experiment import (  # noqa: E402
    RECONSTRUCTOR_ARCHITECTURES,
    TEACHER_ARCHITECTURES,
    CrossArchExperimentLayout,
)
from teacher_logit_reco.crossarch_reco_domain_taggers import (  # noqa: E402
    CrossArchRecoDomainTaggerTrainConfig,
    train_crossarch_reco_domain_tagger,
)


def parse_args() -> argparse.Namespace:
    layout = CrossArchExperimentLayout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reco-architecture", choices=RECONSTRUCTOR_ARCHITECTURES, required=True)
    parser.add_argument("--teacher-architecture", choices=TEACHER_ARCHITECTURES, required=True)
    parser.add_argument("--reconstructor-checkpoint", required=True)
    parser.add_argument("--cache-dir", default=str(layout.hlt_cache_dir))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2205)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=500_000)
    parser.add_argument("--max-val-jets", type=int, default=150_000)
    parser.add_argument("--model-size", choices=["base", "tiny"], default="base")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--teacher-weight-threshold", type=float, default=0.0)
    parser.add_argument("--non-strict-reconstructor-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CrossArchRecoDomainTaggerTrainConfig(
        reco_architecture=args.reco_architecture,
        teacher_architecture=args.teacher_architecture,
        reconstructor_checkpoint=args.reconstructor_checkpoint,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        model_size=args.model_size,
        compile_model=args.compile_model,
        max_constits=args.max_constits,
        teacher_weight_threshold=args.teacher_weight_threshold,
        strict_reconstructor_checkpoint=not args.non_strict_reconstructor_checkpoint,
    )
    report = train_crossarch_reco_domain_tagger(config)
    print("crossarch_reco_domain_tagger_training_complete:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

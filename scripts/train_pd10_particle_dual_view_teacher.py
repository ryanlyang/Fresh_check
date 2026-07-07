#!/usr/bin/env python3
"""Train the PD10-V2 particle-level HLT+offline dual-view teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.privileged_distill_10class import (  # noqa: E402
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_BATCH_SIZE,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_BRANCH_LR,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_DROPOUT,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_EARLY_STOP_PATIENCE,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_EPOCHS,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_FUSION_HIDDEN_DIM,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_LR,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_WARMUP_EPOCHS,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_WARMUP_LR,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_SEED,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_WEIGHT_DECAY,
    PD10_REPRESENTATION_DIM,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_HLT,
    PD10_TEACHER_OFFLINE,
    PD10ParticleDualViewTeacherTrainConfig,
    default_pd10_experiment_layout,
    pd10_particle_dual_view_teacher_dir,
    train_pd10_particle_dual_view_teacher,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(layout.split_manifest_path))
    parser.add_argument("--hlt-cache-dir", default=str(layout.hlt_cache_dir))
    parser.add_argument("--offline-cache-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--hlt-teacher-checkpoint",
        default=str(layout.teacher_checkpoint(PD10_TEACHER_HLT)),
    )
    parser.add_argument(
        "--offline-teacher-checkpoint",
        default=str(layout.teacher_checkpoint(PD10_TEACHER_OFFLINE)),
    )
    parser.add_argument(
        "--output-dir",
        default=str(pd10_particle_dual_view_teacher_dir(output_root="checkpoints")),
    )
    parser.add_argument("--seed", type=int, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_EPOCHS)
    parser.add_argument("--head-warmup-epochs", type=int, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_WARMUP_EPOCHS)
    parser.add_argument("--head-warmup-lr", type=float, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_WARMUP_LR)
    parser.add_argument("--branch-lr", type=float, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_BRANCH_LR)
    parser.add_argument("--head-lr", type=float, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_LR)
    parser.add_argument("--weight-decay", type=float, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=PD10_SPLIT_SIZES["model_train"])
    parser.add_argument("--max-val-jets", type=int, default=PD10_SPLIT_SIZES["model_val"])
    parser.add_argument("--model-size", choices=["base", "tiny", "large"], default="base")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--fusion-hidden-dim", type=int, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_FUSION_HIDDEN_DIM)
    parser.add_argument("--representation-dim", type=int, default=PD10_REPRESENTATION_DIM)
    parser.add_argument("--dropout", type=float, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_DROPOUT)
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--no-verify-hlt-hash", action="store_true")
    parser.add_argument("--no-branch-init", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing particle dual-view teacher artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PD10ParticleDualViewTeacherTrainConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        hlt_teacher_checkpoint=args.hlt_teacher_checkpoint,
        offline_teacher_checkpoint=args.offline_teacher_checkpoint,
        data_dir=args.data_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        head_warmup_epochs=args.head_warmup_epochs,
        head_warmup_lr=args.head_warmup_lr,
        branch_lr=args.branch_lr,
        head_lr=args.head_lr,
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
        fusion_hidden_dim=args.fusion_hidden_dim,
        representation_dim=args.representation_dim,
        dropout=args.dropout,
        verify_label_branches=bool(args.verify_label_branches),
        read_chunk_size=args.read_chunk_size,
        verify_hlt_hash=not bool(args.no_verify_hlt_hash),
        initialize_branches=not bool(args.no_branch_init),
        overwrite=bool(args.overwrite),
    )
    report = train_pd10_particle_dual_view_teacher(config)
    print("pd10_particle_dual_view_teacher_complete:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

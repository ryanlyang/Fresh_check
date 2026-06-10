#!/usr/bin/env python3
"""Train the Step 5 teacher-logit Global Transformer reconstructor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.teachers import TEACHER_ARCHITECTURES  # noqa: E402
from teacher_logit_reco.train_global_transformer import (  # noqa: E402
    TeacherLogitGlobalTransformerTrainConfig,
    train_teacher_logit_global_transformer_reco,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--teacher-architecture", choices=TEACHER_ARCHITECTURES, default=None)
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--seed", type=int, default=1205)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--teacher-weight-threshold", type=float, default=0.0)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-extra-candidates", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-delta-logpt", type=float, default=0.50)
    parser.add_argument("--max-delta-eta", type=float, default=0.25)
    parser.add_argument("--max-delta-phi", type=float, default=0.25)
    parser.add_argument("--max-delta-loge", type=float, default=0.50)
    parser.add_argument("--parent-weight-bias", type=float, default=4.0)
    parser.add_argument("--extra-weight-bias", type=float, default=-3.0)
    parser.add_argument("--max-total-extra-pt-fraction", type=float, default=0.20)
    parser.add_argument("--max-extra-delta-eta", type=float, default=1.25)
    parser.add_argument("--max-extra-delta-phi", type=float, default=1.25)

    parser.add_argument("--teacher-kl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.3)
    parser.add_argument("--correction-budget-weight", type=float, default=0.01)
    parser.add_argument("--jet-summary-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TeacherLogitGlobalTransformerTrainConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        data_dir=args.data_dir,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_architecture=args.teacher_architecture,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        amp=not bool(args.no_amp),
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_label_branches=args.verify_label_branches,
        read_chunk_size=args.read_chunk_size,
        compile_model=args.compile_model,
        max_constits=args.max_constits,
        teacher_weight_threshold=args.teacher_weight_threshold,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_extra_candidates=args.num_extra_candidates,
        dropout=args.dropout,
        max_delta_logpt=args.max_delta_logpt,
        max_delta_eta=args.max_delta_eta,
        max_delta_phi=args.max_delta_phi,
        max_delta_loge=args.max_delta_loge,
        parent_weight_bias=args.parent_weight_bias,
        extra_weight_bias=args.extra_weight_bias,
        max_total_extra_pt_fraction=args.max_total_extra_pt_fraction,
        max_extra_delta_eta=args.max_extra_delta_eta,
        max_extra_delta_phi=args.max_extra_delta_phi,
        teacher_kl_weight=args.teacher_kl_weight,
        ce_weight=args.ce_weight,
        correction_budget_weight=args.correction_budget_weight,
        jet_summary_weight=args.jet_summary_weight,
        temperature=args.temperature,
    )
    report = train_teacher_logit_global_transformer_reco(config)
    print("teacher_logit_global_transformer_training_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  best_model_val_total_loss: {report['best_model_val_total_loss']:.6f}")
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {Path(args.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

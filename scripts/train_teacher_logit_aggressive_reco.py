#!/usr/bin/env python3
"""Train a Step 8 aggressive teacher-logit reconstructor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.teachers import TEACHER_ARCHITECTURES  # noqa: E402
from teacher_logit_reco.train_aggressive_reconstructors import (  # noqa: E402
    TeacherLogitAggressiveReconstructorTrainConfig,
    train_teacher_logit_aggressive_reco,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--teacher-architecture", choices=TEACHER_ARCHITECTURES, default=None)
    parser.add_argument(
        "--reco-architecture",
        default="aggressive_gt",
        help="Aggressive reconstructor: aggressive_gt, aggressive_pn, aggressive_pfn, or aggressive_pcnn.",
    )
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
    parser.add_argument("--edgeconv-dims", type=int, nargs="+", default=[64, 128, 128])
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--phi-dims", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--context-mlp-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[5, 5, 3, 3, 3, 3])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 1, 2, 4])
    parser.add_argument("--embedding-dim", type=int, default=128)

    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num-extra-candidates", type=int, default=64)
    parser.add_argument("--max-delta-logpt", type=float, default=1.0)
    parser.add_argument("--max-delta-eta", type=float, default=0.20)
    parser.add_argument("--max-delta-phi", type=float, default=0.20)
    parser.add_argument("--max-delta-loge", type=float, default=1.0)
    parser.add_argument("--parent-weight-bias", type=float, default=2.0)
    parser.add_argument("--extra-weight-bias", type=float, default=-2.0)
    parser.add_argument("--max-total-extra-pt-fraction", type=float, default=0.50)
    parser.add_argument("--max-extra-delta-eta", type=float, default=1.50)
    parser.add_argument("--max-extra-delta-phi", type=float, default=1.50)
    parser.add_argument("--max-global-logpt-scale", type=float, default=0.35)
    parser.add_argument("--max-global-loge-scale", type=float, default=0.35)
    parser.add_argument("--max-global-eta-shift", type=float, default=0.05)
    parser.add_argument("--max-global-phi-shift", type=float, default=0.05)
    parser.add_argument("--extra-usage-weight-threshold", type=float, default=0.05)
    parser.add_argument("--eta-limit", type=float, default=5.0)
    parser.add_argument("--min-pt", type=float, default=1.0e-4)

    parser.add_argument("--teacher-kl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--correction-budget-weight", type=float, default=0.02)
    parser.add_argument("--jet-summary-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--aggressive-extra-budget-weight", type=float, default=0.05)
    parser.add_argument("--aggressive-parent-weight-budget-weight", type=float, default=0.02)
    parser.add_argument("--aggressive-global-calibration-budget-weight", type=float, default=0.02)
    parser.add_argument("--extra-count-budget-weight", type=float, default=0.05)
    parser.add_argument("--min-parent-weight-fraction", type=float, default=0.25)
    parser.add_argument("--parent-prune-budget-weight", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TeacherLogitAggressiveReconstructorTrainConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        data_dir=args.data_dir,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_architecture=args.teacher_architecture,
        reco_architecture=args.reco_architecture,
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
        edgeconv_dims=tuple(args.edgeconv_dims),
        k=args.k,
        phi_dims=tuple(args.phi_dims),
        context_dim=args.context_dim,
        context_mlp_dims=tuple(args.context_mlp_dims),
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        kernel_sizes=tuple(args.kernel_sizes),
        dilations=tuple(args.dilations),
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        num_extra_candidates=args.num_extra_candidates,
        max_delta_logpt=args.max_delta_logpt,
        max_delta_eta=args.max_delta_eta,
        max_delta_phi=args.max_delta_phi,
        max_delta_loge=args.max_delta_loge,
        parent_weight_bias=args.parent_weight_bias,
        extra_weight_bias=args.extra_weight_bias,
        max_total_extra_pt_fraction=args.max_total_extra_pt_fraction,
        max_extra_delta_eta=args.max_extra_delta_eta,
        max_extra_delta_phi=args.max_extra_delta_phi,
        max_global_logpt_scale=args.max_global_logpt_scale,
        max_global_loge_scale=args.max_global_loge_scale,
        max_global_eta_shift=args.max_global_eta_shift,
        max_global_phi_shift=args.max_global_phi_shift,
        extra_usage_weight_threshold=args.extra_usage_weight_threshold,
        eta_limit=args.eta_limit,
        min_pt=args.min_pt,
        teacher_kl_weight=args.teacher_kl_weight,
        ce_weight=args.ce_weight,
        correction_budget_weight=args.correction_budget_weight,
        jet_summary_weight=args.jet_summary_weight,
        temperature=args.temperature,
        aggressive_extra_budget_weight=args.aggressive_extra_budget_weight,
        aggressive_parent_weight_budget_weight=args.aggressive_parent_weight_budget_weight,
        aggressive_global_calibration_budget_weight=args.aggressive_global_calibration_budget_weight,
        extra_count_budget_weight=args.extra_count_budget_weight,
        min_parent_weight_fraction=args.min_parent_weight_fraction,
        parent_prune_budget_weight=args.parent_prune_budget_weight,
    )
    report = train_teacher_logit_aggressive_reco(config)
    print("teacher_logit_aggressive_training_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  reco_architecture: {report['reconstructor_architecture']}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  best_model_val_total_loss: {report['best_model_val_total_loss']:.6f}")
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {Path(args.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

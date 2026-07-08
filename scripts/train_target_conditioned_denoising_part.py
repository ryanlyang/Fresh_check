#!/usr/bin/env python3
"""Train a target-conditioned pairwise particle denoiser."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.target_denoising_part import (  # noqa: E402
    TARGET_DENOISING_SELECTION_METRICS,
    TargetDenoisingPretrainConfig,
    train_target_conditioned_denoiser,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", default=None)
    parser.add_argument("--data-dir", default=None)

    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--seed", type=int, default=7207)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--selection-metric", choices=TARGET_DENOISING_SELECTION_METRICS, default="normalized_rmse")
    parser.add_argument("--compile-model", action="store_true")

    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--alignment-mode", choices=("aligned_direct", "rank_direct"), default="aligned_direct")
    parser.add_argument("--shuffle-target-residuals", action="store_true")
    parser.add_argument("--target-shuffle-seed", type=int, default=93217)
    parser.add_argument("--expected-hlt-profile", default="fixed_hlt_v2_realistic")
    parser.add_argument("--expected-hlt-profile-version", default="v1")
    parser.add_argument("--expected-hlt-degradation-strength", type=float, default=1.0)

    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--pair-hidden-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--disable-pair-bias", action="store_true")
    parser.add_argument("--disable-local-kernel", action="store_true")
    parser.add_argument("--local-kernel-radius", type=float, default=0.12)
    parser.add_argument("--local-kernel-init", type=float, default=0.0)
    parser.add_argument("--pair-bias-max-abs", type=float, default=4.0)
    parser.add_argument("--max-delta-log-pt", type=float, default=0.30)
    parser.add_argument("--max-delta-eta", type=float, default=0.08)
    parser.add_argument("--max-delta-phi", type=float, default=0.08)
    parser.add_argument("--max-delta-log-energy", type=float, default=0.30)

    parser.add_argument("--smooth-l1-weight", type=float, default=0.5)
    parser.add_argument("--nll-weight", type=float, default=0.5)
    parser.add_argument("--reliability-weight", type=float, default=0.05)
    parser.add_argument("--delta-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--pair-bias-l2-weight", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TargetDenoisingPretrainConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        data_dir=args.data_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
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
        selection_metric=args.selection_metric,
        compile_model=bool(args.compile_model),
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_label_branches=bool(args.verify_label_branches),
        read_chunk_size=args.read_chunk_size,
        alignment_mode=args.alignment_mode,
        shuffle_target_residuals=bool(args.shuffle_target_residuals),
        target_shuffle_seed=args.target_shuffle_seed,
        expected_hlt_profile=args.expected_hlt_profile,
        expected_hlt_profile_version=args.expected_hlt_profile_version,
        expected_hlt_degradation_strength=args.expected_hlt_degradation_strength,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        pair_hidden_dim=args.pair_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        use_pair_bias=not bool(args.disable_pair_bias),
        use_local_kernel=not bool(args.disable_local_kernel),
        local_kernel_radius=args.local_kernel_radius,
        local_kernel_init=args.local_kernel_init,
        pair_bias_max_abs=args.pair_bias_max_abs,
        max_delta_log_pt=args.max_delta_log_pt,
        max_delta_eta=args.max_delta_eta,
        max_delta_phi=args.max_delta_phi,
        max_delta_log_energy=args.max_delta_log_energy,
        smooth_l1_weight=args.smooth_l1_weight,
        nll_weight=args.nll_weight,
        reliability_weight=args.reliability_weight,
        delta_l2_weight=args.delta_l2_weight,
        pair_bias_l2_weight=args.pair_bias_l2_weight,
    )
    report = train_target_conditioned_denoiser(config)
    print("target_conditioned_denoising_part_training_complete:")
    print(f"  output_dir: {report['output_dir']}")
    print(f"  output_contract: {report['output_contract']}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  selection_metric: {report['selection_metric']}")
    print(f"  best_model_selection_metric_value: {report['best_model_selection_metric_value']:.8g}")
    best_metrics = report.get("best_model_val_metrics", {})
    print(
        "  model_val: "
        f"loss={best_metrics.get('loss')} "
        f"normalized_rmse={best_metrics.get('normalized_rmse')} "
        f"target_count={best_metrics.get('target_count')}"
    )
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {report['run_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

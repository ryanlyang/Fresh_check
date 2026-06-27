#!/usr/bin/env python3
"""Train the reliability-gated dual-view ParT residual model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.dualview_part import (  # noqa: E402
    DUALVIEW_PART_PRIMARY_METRIC,
    DUALVIEW_PART_SELECTION_METRICS,
    DUALVIEW_PART_SOURCE_LABEL_NAMES,
    DualViewResidualTrainConfig,
    train_dualview_residual_part,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-anchor-checkpoint", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--pn-reconstructed-view-dir", required=True)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--diagnostics-mirror-dir", default=None)

    parser.add_argument("--train-split", default="stack_train")
    parser.add_argument("--val-split", default="stack_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--confirm-split-settings", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")

    parser.add_argument("--seed", type=int, default=2205)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--anchor-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--selection-metric", choices=DUALVIEW_PART_SELECTION_METRICS, default=DUALVIEW_PART_PRIMARY_METRIC)

    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-final-test-jets", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--skip-initialization-check", action="store_true")
    parser.add_argument("--initialization-check-batches", type=int, default=1)
    parser.add_argument("--max-case-rows-per-type", type=int, default=1000)

    parser.add_argument("--warm-anchor", action="store_true")
    parser.add_argument("--allow-noncanonical-anchor", action="store_true")
    parser.add_argument("--anchor-model-size", choices=("base", "tiny"), default="base")
    parser.add_argument("--anchor-context-dim", type=int, default=128)
    parser.add_argument("--anchor-summary-hidden-dim", type=int, default=128)
    parser.add_argument("--anchor-summary-dropout", type=float, default=0.0)
    parser.add_argument("--non-strict-anchor", action="store_true")

    parser.add_argument("--max-hlt-constits", type=int, default=128)
    parser.add_argument("--hlt-weight-threshold", type=float, default=0.0)
    parser.add_argument("--max-pn-tokens", type=int, default=128)
    parser.add_argument("--min-pn-tokens", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--selection-mode", choices=("topk_or_threshold", "all_slots"), default="topk_or_threshold")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--allow-noncanonical-dataset", action="store_true")
    parser.add_argument("--enforce-split-size", action="store_true")

    parser.add_argument("--pn-embed-dim", type=int, default=128)
    parser.add_argument("--pn-layers", type=int, default=2)
    parser.add_argument("--pn-heads", type=int, default=4)
    parser.add_argument("--pn-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--pn-dropout", type=float, default=0.05)
    parser.add_argument("--pn-attention-dropout", type=float, default=0.05)
    parser.add_argument("--disable-pn-confidence", action="store_true")

    parser.add_argument("--residual-hidden-dim", type=int, default=128)
    parser.add_argument("--residual-layers", type=int, default=2)
    parser.add_argument("--residual-dropout", type=float, default=0.05)
    parser.add_argument("--gate-bias-init", type=float, default=-5.0)
    parser.add_argument("--disable-anchor-context", action="store_true")
    parser.add_argument("--disable-reliability-features", action="store_true")

    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--shuffle-pn-view", action="store_true")
    parser.add_argument("--pn-view-shuffle-seed", type=int, default=2205)
    parser.add_argument("--label-names", nargs=2, default=list(DUALVIEW_PART_SOURCE_LABEL_NAMES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DualViewResidualTrainConfig(
        output_dir=args.output_dir,
        hlt_anchor_checkpoint=args.hlt_anchor_checkpoint,
        hlt_cache_dir=args.hlt_cache_dir,
        pn_reconstructed_view_dir=args.pn_reconstructed_view_dir,
        experiment_dir=args.experiment_dir,
        diagnostics_mirror_dir=args.diagnostics_mirror_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        final_test_split=args.final_test_split,
        confirm_split_settings=bool(args.confirm_split_settings),
        confirm_final_test=bool(args.confirm_final_test),
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        anchor_lr=args.anchor_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        amp=not bool(args.no_amp),
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        selection_metric=args.selection_metric,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_final_test_batches=args.max_final_test_batches,
        run_initialization_check=not bool(args.skip_initialization_check),
        initialization_check_batches=args.initialization_check_batches,
        max_case_rows_per_type=args.max_case_rows_per_type,
        freeze_anchor=not bool(args.warm_anchor),
        enforce_anchor_contract=not bool(args.allow_noncanonical_anchor),
        anchor_model_size=args.anchor_model_size,
        anchor_context_dim=args.anchor_context_dim,
        anchor_summary_hidden_dim=args.anchor_summary_hidden_dim,
        anchor_summary_dropout=args.anchor_summary_dropout,
        anchor_strict=not bool(args.non_strict_anchor),
        max_hlt_constits=args.max_hlt_constits,
        hlt_weight_threshold=args.hlt_weight_threshold,
        max_pn_tokens=args.max_pn_tokens,
        min_pn_tokens=args.min_pn_tokens,
        confidence_threshold=args.confidence_threshold,
        selection_mode=args.selection_mode,
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        enforce_dataset_contract=not bool(args.allow_noncanonical_dataset),
        enforce_split_size=bool(args.enforce_split_size),
        pn_embed_dim=args.pn_embed_dim,
        pn_layers=args.pn_layers,
        pn_heads=args.pn_heads,
        pn_mlp_ratio=args.pn_mlp_ratio,
        pn_dropout=args.pn_dropout,
        pn_attention_dropout=args.pn_attention_dropout,
        pn_use_confidence=not bool(args.disable_pn_confidence),
        residual_hidden_dim=args.residual_hidden_dim,
        residual_layers=args.residual_layers,
        residual_dropout=args.residual_dropout,
        gate_bias_init=args.gate_bias_init,
        use_anchor_context=not bool(args.disable_anchor_context),
        use_reliability_features=not bool(args.disable_reliability_features),
        compile_model=bool(args.compile_model),
        shuffle_pn_view=bool(args.shuffle_pn_view),
        pn_view_shuffle_seed=args.pn_view_shuffle_seed,
        label_names=tuple(args.label_names),
    )
    report = train_dualview_residual_part(config)
    print("dualview_part_residual_training_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  selection_metric: {report['selection_metric']}")
    print(f"  best_model_selection_metric_value: {report['best_model_selection_metric_value']:.8g}")
    print(f"  best_model_val_accuracy: {report['best_model_val_accuracy']:.6f}")
    print(f"  final_test_evaluated: {report['final_test_evaluated']}")
    print(f"  shuffle_pn_view: {report['shuffle_pn_view']}")
    if report.get("final_test_metrics"):
        final_metrics = report["final_test_metrics"]
        binary = final_metrics.get("binary_metrics", {})
        print(
            "  final_test: "
            f"acc={final_metrics.get('accuracy'):.6f} "
            f"auc={binary.get('auc')} "
            f"fpr30={binary.get('fpr_at_signal_eff_0p30')} "
            f"fpr50={binary.get('fpr_at_signal_eff_0p50')}"
        )
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {Path(args.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

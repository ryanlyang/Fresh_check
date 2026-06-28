#!/usr/bin/env python3
"""Train one Step 10 multi-scale subjet HLT ParT comparison variant."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.multiscale_subjet_part.model import MULTISCALE_SUBJET_CLASSIFIER_VARIANTS  # noqa: E402
from teacher_logit_reco.multiscale_subjet_part.train import (  # noqa: E402
    MULTISCALE_SUBJET_SELECTION_METRICS,
    MultiScaleSubjetTrainConfig,
    train_multiscale_subjet_tagger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--variant", choices=MULTISCALE_SUBJET_CLASSIFIER_VARIANTS, default="multiscale_subjet_residual_part_adapter")
    parser.add_argument("--ablation-profile", default="")

    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--stack-val-split", default="stack_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--confirm-split-settings", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")

    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-stack-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-stack-val-jets", type=int, default=None)
    parser.add_argument("--max-final-test-jets", type=int, default=None)
    parser.add_argument("--selection-metric", choices=MULTISCALE_SUBJET_SELECTION_METRICS, default="fpr_at_signal_eff_0p50")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--skip-hlt-params-check", action="store_true")
    parser.add_argument("--expected-hlt-degradation-strength", type=float, default=0.6)

    parser.add_argument("--model-size", choices=("base", "tiny", "large"), default="base")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--token-hidden-dim", type=int, default=256)
    parser.add_argument("--assignment-embed-dim", type=int, default=64)
    parser.add_argument("--assignment-hidden-dim", type=int, default=128)
    parser.add_argument("--assignment-temperature", type=float, default=1.0)
    parser.add_argument("--assignment-geometry-bias-strength", type=float, default=2.0)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-dim", type=int, default=256)
    parser.add_argument("--transformer-pair-bias-hidden-dim", type=int, default=64)
    parser.add_argument("--readback-hidden-dim", type=int, default=128)
    parser.add_argument("--readback-heads", type=int, default=4)
    parser.add_argument("--readback-delta-hidden-dim", type=int, default=256)
    parser.add_argument("--branch-hidden-dim", type=int, default=256)
    parser.add_argument("--residual-gamma-init", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--scale-profile", default="default")
    parser.add_argument("--disable-assignment-scale-embedding", action="store_true")
    parser.add_argument("--disable-token-scale-embedding", action="store_true")
    parser.add_argument("--disable-subjet-pair-bias", action="store_true")
    parser.add_argument("--disable-scale-pair-embedding", action="store_true")
    parser.add_argument("--random-control-seed", type=int, default=2027)
    parser.add_argument("--weight-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MultiScaleSubjetTrainConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        variant=args.variant,
        ablation_profile=args.ablation_profile or args.variant,
        train_split=args.train_split,
        val_split=args.val_split,
        stack_val_split=args.stack_val_split,
        final_test_split=args.final_test_split,
        confirm_split_settings=bool(args.confirm_split_settings),
        confirm_final_test=bool(args.confirm_final_test),
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
        max_stack_val_batches=args.max_stack_val_batches,
        max_final_test_batches=args.max_final_test_batches,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_stack_val_jets=args.max_stack_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        selection_metric=args.selection_metric,
        compile_model=bool(args.compile_model),
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_hlt_params=not bool(args.skip_hlt_params_check),
        expected_hlt_degradation_strength=args.expected_hlt_degradation_strength,
        model_size=args.model_size,
        max_constits=args.max_constits,
        token_dim=args.token_dim,
        token_hidden_dim=args.token_hidden_dim,
        assignment_embed_dim=args.assignment_embed_dim,
        assignment_hidden_dim=args.assignment_hidden_dim,
        assignment_temperature=args.assignment_temperature,
        assignment_geometry_bias_strength=args.assignment_geometry_bias_strength,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        transformer_pair_bias_hidden_dim=args.transformer_pair_bias_hidden_dim,
        readback_hidden_dim=args.readback_hidden_dim,
        readback_heads=args.readback_heads,
        readback_delta_hidden_dim=args.readback_delta_hidden_dim,
        branch_hidden_dim=args.branch_hidden_dim,
        residual_gamma_init=args.residual_gamma_init,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        scale_profile=args.scale_profile,
        use_assignment_scale_embedding=not bool(args.disable_assignment_scale_embedding),
        use_token_scale_embedding=not bool(args.disable_token_scale_embedding),
        use_subjet_pair_bias=not bool(args.disable_subjet_pair_bias),
        use_scale_pair_embedding=not bool(args.disable_scale_pair_embedding),
        random_control_seed=args.random_control_seed,
        weight_threshold=args.weight_threshold,
    )
    report = train_multiscale_subjet_tagger(config)
    print("multiscale_subjet_part_tagger_training_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  variant: {report['variant']}")
    print(f"  profile: {report.get('profile')}")
    print(f"  output_contract: {report['output_contract']}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  selection_metric: {report['selection_metric']}")
    print(f"  best_model_selection_metric_value: {report['best_model_selection_metric_value']:.8g}")
    print(f"  best_model_val_accuracy: {report['best_model_val_accuracy']:.6f}")
    if report.get("stack_val_metrics"):
        stack_binary = report["stack_val_metrics"].get("binary_metrics", {})
        print(
            "  stack_val: "
            f"acc={report['stack_val_metrics'].get('accuracy'):.6f} "
            f"auc={stack_binary.get('auc')} "
            f"fpr30={stack_binary.get('fpr_at_signal_eff_0p30')} "
            f"fpr50={stack_binary.get('fpr_at_signal_eff_0p50')}"
        )
    print(f"  final_test_evaluated: {report['final_test_evaluated']}")
    if report.get("final_test_metrics"):
        final_binary = report["final_test_metrics"].get("binary_metrics", {})
        print(
            "  final_test: "
            f"acc={report['final_test_metrics'].get('accuracy'):.6f} "
            f"auc={final_binary.get('auc')} "
            f"fpr30={final_binary.get('fpr_at_signal_eff_0p30')} "
            f"fpr50={final_binary.get('fpr_at_signal_eff_0p50')} "
            f"val_threshold_fpr={final_binary.get('validation_threshold_final_test_fpr')}"
        )
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {Path(args.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

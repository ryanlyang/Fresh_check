#!/usr/bin/env python3
"""Train one local-graph residual expert against a frozen HLT ParT score."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_graph_part.model import LOCAL_GRAPH_ADAPTERS  # noqa: E402
from teacher_logit_reco.local_graph_part.protocol import LOCAL_GRAPH_PART_PRIMARY_METRIC  # noqa: E402
from teacher_logit_reco.local_graph_part.residual_losses import (  # noqa: E402
    LOCAL_GRAPH_RESIDUAL_LOSS_MODES,
)
from teacher_logit_reco.local_graph_part.residual_train import (  # noqa: E402
    LocalGraphResidualExpertTrainConfig,
    train_local_graph_residual_expert,
)


def _add_common_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument(
        "--baseline-logit-cache-dir",
        required=True,
        help="Directory written by cache_local_graph_baseline_logits.py.",
    )
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--confirm-split-settings", action="store_true")
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help=(
            "Acknowledge the held-out final_test split for downstream final evaluation. "
            "This Step 6 trainer still loads only model_train/model_val."
        ),
    )

    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
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
    parser.add_argument("--selection-metric", default=LOCAL_GRAPH_PART_PRIMARY_METRIC)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--skip-hlt-params-check", action="store_true")
    parser.add_argument("--expected-hlt-degradation-strength", type=float, default=0.6)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-size", choices=("base", "tiny"), default="base")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--local-adapter", choices=LOCAL_GRAPH_ADAPTERS, default="point_attention")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--local-embed-dim", type=int, default=128)
    parser.add_argument("--local-heads", type=int, default=8)
    parser.add_argument("--local-hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--residual-gamma-init", type=float, default=0.0)
    parser.add_argument("--weight-threshold", type=float, default=0.0)
    parser.add_argument("--backbone-output-dim", type=int, default=128)
    parser.add_argument("--condition-embed-dim", type=int, default=64)
    parser.add_argument("--residual-hidden-dim", type=int, default=128)
    parser.add_argument("--residual-dropout", type=float, default=0.05)
    parser.add_argument("--alpha-initial", type=float, default=0.1)
    parser.add_argument("--disable-alpha-learnable", action="store_true")
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument(
        "--disable-alpha-max",
        action="store_true",
        help="Do not apply the positive-alpha upper clamp in the residual expert.",
    )


def _add_loss_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--loss-mode",
        default="residual_weighted_bce",
        metavar="MODE",
        help=(
            "Residual objective ladder mode. Supports A-E aliases and canonical modes: "
            + ", ".join(LOCAL_GRAPH_RESIDUAL_LOSS_MODES)
        ),
    )
    parser.add_argument("--bce-anchor-weight", type=float, default=0.10)
    parser.add_argument("--soft-fpr-weight", type=float, default=0.25)
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--weighted-bce-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-temperature", type=float, default=0.20)
    parser.add_argument("--soft-fpr-epsilon", type=float, default=0.20)
    parser.add_argument("--cvar-top-fraction", type=float, default=0.50)
    parser.add_argument("--hard-background-fraction", type=float, default=0.20)
    parser.add_argument("--signal-boundary-quantile-low", type=float, default=0.40)
    parser.add_argument("--signal-boundary-quantile-high", type=float, default=0.60)
    parser.add_argument("--bce-boundary-scale", type=float, default=None)


def _add_warm_start_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--warm-start-checkpoint",
        default=None,
        help="Optional HLT ParT baseline checkpoint for the residual expert backbone.",
    )
    parser.add_argument("--require-warm-start", action="store_true")
    parser.add_argument(
        "--freeze-part-epochs",
        type=int,
        default=0,
        help="Freeze the ParT backbone for this many initial residual-expert epochs.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_training_args(parser)
    _add_model_args(parser)
    _add_loss_args(parser)
    _add_warm_start_args(parser)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> LocalGraphResidualExpertTrainConfig:
    return LocalGraphResidualExpertTrainConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        baseline_logit_cache_dir=args.baseline_logit_cache_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        confirm_split_settings=bool(args.confirm_split_settings),
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
        verify_hlt_params=not bool(args.skip_hlt_params_check),
        expected_hlt_degradation_strength=args.expected_hlt_degradation_strength,
        model_size=args.model_size,
        max_constits=args.max_constits,
        local_adapter=args.local_adapter,
        k=args.k,
        local_embed_dim=args.local_embed_dim,
        local_heads=args.local_heads,
        local_hidden_dim=args.local_hidden_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        residual_gamma_init=args.residual_gamma_init,
        weight_threshold=args.weight_threshold,
        backbone_output_dim=args.backbone_output_dim,
        condition_embed_dim=args.condition_embed_dim,
        residual_hidden_dim=args.residual_hidden_dim,
        residual_dropout=args.residual_dropout,
        alpha_initial=args.alpha_initial,
        alpha_learnable=not bool(args.disable_alpha_learnable),
        alpha_max=None if bool(args.disable_alpha_max) else args.alpha_max,
        loss_mode=args.loss_mode,
        bce_anchor_weight=args.bce_anchor_weight,
        soft_fpr_weight=args.soft_fpr_weight,
        residual_l2_weight=args.residual_l2_weight,
        pairwise_weight=args.pairwise_weight,
        weighted_bce_weight=args.weighted_bce_weight,
        pairwise_temperature=args.pairwise_temperature,
        soft_fpr_epsilon=args.soft_fpr_epsilon,
        cvar_top_fraction=args.cvar_top_fraction,
        hard_background_fraction=args.hard_background_fraction,
        signal_boundary_quantile_low=args.signal_boundary_quantile_low,
        signal_boundary_quantile_high=args.signal_boundary_quantile_high,
        bce_boundary_scale=args.bce_boundary_scale,
        warm_start_checkpoint=args.warm_start_checkpoint,
        require_warm_start=bool(args.require_warm_start),
        freeze_part_epochs=args.freeze_part_epochs,
    )


def _nested_metric(metrics: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(metrics, dict):
        return None
    if key in metrics:
        return metrics.get(key)
    binary = metrics.get("binary_metrics")
    if isinstance(binary, dict):
        return binary.get(key)
    return None


def _fmt(value: Any, precision: int = 6) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    report = train_local_graph_residual_expert(config)
    fused = report.get("fused_model_val_metrics")
    baseline = report.get("baseline_model_val_metrics")
    diagnostics = report.get("residual_diagnostics_model_val") or {}
    alpha = report.get("alpha_shrinkage_model_val") or {}

    print("local_graph_residual_expert_training_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  variant: {report['variant']}")
    print(f"  loss_mode: {report['loss_config']['mode']}")
    print(f"  output_contract: {report['output_contract']}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  selection_metric: {report['selection_metric']}")
    print(f"  best_model_selection_metric_value: {_fmt(report['best_model_selection_metric_value'], 8)}")
    print(
        "  baseline_model_val: "
        f"acc={_fmt(_nested_metric(baseline, 'accuracy'))} "
        f"auc={_fmt(_nested_metric(baseline, 'auc'), 8)} "
        f"fpr30={_fmt(_nested_metric(baseline, 'fpr_at_signal_eff_0p30'), 8)} "
        f"fpr50={_fmt(_nested_metric(baseline, 'fpr_at_signal_eff_0p50'), 8)}"
    )
    print(
        "  fused_model_val: "
        f"acc={_fmt(_nested_metric(fused, 'accuracy'))} "
        f"auc={_fmt(_nested_metric(fused, 'auc'), 8)} "
        f"fpr30={_fmt(_nested_metric(fused, 'fpr_at_signal_eff_0p30'), 8)} "
        f"fpr50={_fmt(_nested_metric(fused, 'fpr_at_signal_eff_0p50'), 8)}"
    )
    print(
        "  residual_delta: "
        f"fpr50={_fmt(diagnostics.get('fused_delta_FPR50_vs_baseline'), 8)} "
        f"old_fp_removed={_fmt((diagnostics.get('false_positive_overlap') or {}).get('old_false_positives_removed'), 8)} "
        f"new_fp_introduced={_fmt((diagnostics.get('false_positive_overlap') or {}).get('new_false_positives_introduced'), 8)}"
    )
    print(
        "  alpha_shrinkage_model_val: "
        f"selected_alpha={_fmt(alpha.get('selected_alpha'), 8)} "
        f"delta_fpr={_fmt(alpha.get('delta_fpr_vs_baseline'), 8)} "
        f"collapsed_to_zero={alpha.get('collapsed_to_zero')}"
    )
    print("  final_test_evaluated: False")
    print("  final_test_note: Step 6 training intentionally loads only model_train/model_val.")
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  diagnostics: {report.get('residual_diagnostics_model_val_path')}")
    print(f"  run_report: {Path(args.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

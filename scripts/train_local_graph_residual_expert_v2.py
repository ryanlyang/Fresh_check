#!/usr/bin/env python3
"""Train one local-graph residual expert V2 mode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_graph_part.residual_v2_losses import (  # noqa: E402
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES,
)
from teacher_logit_reco.local_graph_part.residual_v2_train import (  # noqa: E402
    LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC,
    LocalGraphResidualExpertV2TrainConfig,
    train_local_graph_residual_expert_v2,
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument(
        "--baseline-embedding-cache-dir",
        required=True,
        help="Directory written by cache_local_graph_residual_v2_embeddings.py.",
    )
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--confirm-split-settings", action="store_true")
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Accepted for orchestration symmetry; this trainer still loads only model_train/model_val.",
    )
    parser.add_argument("--seed", type=int, default=5207)
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
    parser.add_argument("--selection-metric", default=LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--skip-hlt-params-check", action="store_true")
    parser.add_argument("--expected-hlt-degradation-strength", type=float, default=0.6)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-embedding-dim", type=int, default=None)
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--local-embed-dim", type=int, default=128)
    parser.add_argument("--local-heads", type=int, default=8)
    parser.add_argument("--local-hidden-dim", type=int, default=None)
    parser.add_argument("--local-adapter-gamma-init", type=float, default=1.0)
    parser.add_argument("--local-pool-mode", choices=("mean_max", "mean"), default="mean_max")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--weight-threshold", type=float, default=0.0)
    parser.add_argument("--condition-embed-dim", type=int, default=64)
    parser.add_argument("--local-context-dim", type=int, default=128)
    parser.add_argument("--residual-hidden-dim", type=int, default=256)
    parser.add_argument("--residual-dropout", type=float, default=0.05)
    parser.add_argument("--residual-output-scale", type=float, default=1.0)
    parser.add_argument("--gate-bias-init", type=float, default=-1.0)
    parser.add_argument("--delta-init-std", type=float, default=1.0e-3)
    parser.add_argument("--gamma-initial", type=float, default=0.1)
    parser.add_argument("--disable-gamma-learnable", action="store_true")
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--disable-gamma-max", action="store_true")
    parser.add_argument("--residual-input-mode", choices=("full", "embedding_only", "local_only"), default="full")
    parser.add_argument("--condition-control-mode", choices=("normal", "shuffled"), default="normal")
    parser.add_argument("--condition-shuffle-seed", type=int, default=520701)
    parser.add_argument("--label-control-mode", choices=("normal", "shuffled"), default="normal")
    parser.add_argument("--label-shuffle-seed", type=int, default=520702)


def _add_loss_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--loss-mode",
        default="residual_v2_weighted_bce",
        help=(
            "V2 objective mode. Supports A-D aliases; E is report-time shrinkage only. "
            "Canonical modes: " + ", ".join(LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES)
        ),
    )
    parser.add_argument("--bce-anchor-weight", type=float, default=0.05)
    parser.add_argument("--soft-fpr-weight", type=float, default=0.25)
    parser.add_argument("--correction-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--weighted-bce-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-temperature", type=float, default=0.20)
    parser.add_argument("--soft-fpr-epsilon", type=float, default=0.20)
    parser.add_argument("--cvar-top-fraction", type=float, default=0.50)
    parser.add_argument("--hard-background-fraction", type=float, default=0.20)
    parser.add_argument("--signal-boundary-quantile-low", type=float, default=0.40)
    parser.add_argument("--signal-boundary-quantile-high", type=float, default=0.60)
    parser.add_argument("--bce-boundary-scale", type=float, default=None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_args(parser)
    _add_model_args(parser)
    _add_loss_args(parser)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> LocalGraphResidualExpertV2TrainConfig:
    return LocalGraphResidualExpertV2TrainConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        baseline_embedding_cache_dir=args.baseline_embedding_cache_dir,
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
        baseline_embedding_dim=args.baseline_embedding_dim,
        max_constits=args.max_constits,
        k=args.k,
        local_embed_dim=args.local_embed_dim,
        local_heads=args.local_heads,
        local_hidden_dim=args.local_hidden_dim,
        local_adapter_gamma_init=args.local_adapter_gamma_init,
        local_pool_mode=args.local_pool_mode,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        weight_threshold=args.weight_threshold,
        condition_embed_dim=args.condition_embed_dim,
        local_context_dim=args.local_context_dim,
        residual_hidden_dim=args.residual_hidden_dim,
        residual_dropout=args.residual_dropout,
        residual_output_scale=args.residual_output_scale,
        gate_bias_init=args.gate_bias_init,
        delta_init_std=args.delta_init_std,
        gamma_initial=args.gamma_initial,
        gamma_learnable=not bool(args.disable_gamma_learnable),
        gamma_max=None if bool(args.disable_gamma_max) else args.gamma_max,
        residual_input_mode=args.residual_input_mode,
        condition_control_mode=args.condition_control_mode,
        condition_shuffle_seed=args.condition_shuffle_seed,
        label_control_mode=args.label_control_mode,
        label_shuffle_seed=args.label_shuffle_seed,
        loss_mode=args.loss_mode,
        bce_anchor_weight=args.bce_anchor_weight,
        soft_fpr_weight=args.soft_fpr_weight,
        correction_l2_weight=args.correction_l2_weight,
        pairwise_weight=args.pairwise_weight,
        weighted_bce_weight=args.weighted_bce_weight,
        pairwise_temperature=args.pairwise_temperature,
        soft_fpr_epsilon=args.soft_fpr_epsilon,
        cvar_top_fraction=args.cvar_top_fraction,
        hard_background_fraction=args.hard_background_fraction,
        signal_boundary_quantile_low=args.signal_boundary_quantile_low,
        signal_boundary_quantile_high=args.signal_boundary_quantile_high,
        bce_boundary_scale=args.bce_boundary_scale,
    )


def _nested_metric(metrics: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(metrics, dict):
        return None
    if key in metrics:
        return metrics[key]
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
    report = train_local_graph_residual_expert_v2(build_config(args))
    baseline = report.get("baseline_model_val_metrics")
    learned = report.get("fused_model_val_learned_gamma_metrics")
    shrunk = report.get("fused_model_val_val_shrunk_metrics")
    gamma = report.get("gamma_shrinkage_model_val") or {}

    print("local_graph_residual_expert_v2_training_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  variant: {report['variant']}")
    print(f"  loss_mode: {report['loss_config']['mode']}")
    print(f"  control_modes: {report['control_modes']}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  selection_metric: {report['selection_metric']}")
    print(f"  best_model_selection_metric_value: {_fmt(report['best_model_selection_metric_value'], 8)}")
    print(
        "  baseline_model_val: "
        f"auc={_fmt(_nested_metric(baseline, 'auc'), 8)} "
        f"fpr50={_fmt(_nested_metric(baseline, 'fpr_at_signal_eff_0p50'), 8)}"
    )
    print(
        "  learned_gamma_model_val: "
        f"auc={_fmt(_nested_metric(learned, 'auc'), 8)} "
        f"fpr50={_fmt(_nested_metric(learned, 'fpr_at_signal_eff_0p50'), 8)}"
    )
    print(
        "  val_shrunk_model_val: "
        f"gamma={_fmt(gamma.get('selected_gamma'), 8)} "
        f"auc={_fmt(_nested_metric(shrunk, 'auc'), 8)} "
        f"fpr50={_fmt(_nested_metric(shrunk, 'fpr_at_signal_eff_0p50'), 8)}"
    )
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {args.output_dir}/run_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

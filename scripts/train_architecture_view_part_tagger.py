#!/usr/bin/env python3
"""Train one Architecture-View Residual ParT tagger."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.architecture_view_part.config import ARCHITECTURE_VIEW_ALL_VARIANTS  # noqa: E402
from teacher_logit_reco.architecture_view_part.train import (  # noqa: E402
    ARCHITECTURE_VIEW_SELECTION_METRICS,
    ArchitectureViewTaggerTrainConfig,
    architecture_view_label_filter_names_to_indices,
    train_architecture_view_tagger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--hlt-cache-dir", default="")
    parser.add_argument("--offline-cache-dir", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--input-source", choices=("hlt", "offline"), default=None)

    parser.add_argument("--label-names", nargs="+", default=("QCD", "Hgg"))
    parser.add_argument("--label-filter-names", nargs="+", default=("QCD", "Hgg"))
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--stack-val-split", default="stack_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--confirm-split-settings", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")

    parser.add_argument("--seed", type=int, default=6207)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--adapter-lr", type=float, default=3.0e-4)
    parser.add_argument("--part-lr", type=float, default=1.0e-5)
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
    parser.add_argument(
        "--selection-metric",
        choices=ARCHITECTURE_VIEW_SELECTION_METRICS,
        default=None,
        help="Checkpoint-selection metric. Defaults to FPR@50 for binary variants and accuracy for AV10 variants.",
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--skip-hlt-params-check", action="store_true")
    parser.set_defaults(require_baseline_split_manifest_hash=True)
    parser.add_argument(
        "--require-baseline-split-manifest-hash",
        dest="require_baseline_split_manifest_hash",
        action="store_true",
    )
    parser.add_argument(
        "--allow-missing-baseline-split-manifest-hash",
        dest="require_baseline_split_manifest_hash",
        action="store_false",
    )
    parser.add_argument("--expected-hlt-degradation-strength", type=float, default=0.6)

    parser.add_argument("--variant", choices=ARCHITECTURE_VIEW_ALL_VARIANTS, default="av_all_views")
    parser.add_argument("--view-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--pn-k", type=int, default=16)
    parser.add_argument("--pn-layers", type=int, default=2)
    parser.add_argument("--pfn-hidden-dim", type=int, default=64)
    parser.add_argument("--pcnn-channels", type=int, default=64)
    parser.add_argument("--pcnn-layers", type=int, default=2)
    parser.add_argument("--fusion-hidden-dim", type=int, default=96)
    parser.add_argument("--part-embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--gate-bias-init", type=float, default=-5.0)
    parser.add_argument("--random-control-seed", type=int, default=2907)
    parser.add_argument("--delta-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--freeze-part-epochs", type=int, default=2)
    parser.add_argument("--input-delta-scale", type=float, default=1.0)
    parser.add_argument("--disable-feature-wise-input-delta-scales", action="store_true")
    parser.add_argument("--freeze-input-delta-pid", action="store_true")
    parser.add_argument("--freeze-input-delta-geometry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label_names = tuple(str(name) for name in args.label_names)
    label_filter = architecture_view_label_filter_names_to_indices(
        args.label_filter_names,
        manifest_path=args.manifest_path,
        label_names=label_names,
    )
    config = ArchitectureViewTaggerTrainConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        baseline_checkpoint=args.baseline_checkpoint,
        input_source=args.input_source,
        offline_cache_dir=args.offline_cache_dir,
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
        adapter_lr=args.adapter_lr,
        part_lr=args.part_lr,
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
        num_classes=args.num_classes,
        compile_model=bool(args.compile_model),
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_hlt_params=not bool(args.skip_hlt_params_check),
        require_baseline_split_manifest_hash=bool(args.require_baseline_split_manifest_hash),
        expected_hlt_degradation_strength=args.expected_hlt_degradation_strength,
        label_names=label_names,
        label_filter=label_filter,
        variant=args.variant,
        view_dim=args.view_dim,
        hidden_dim=args.hidden_dim,
        pn_k=args.pn_k,
        pn_layers=args.pn_layers,
        pfn_hidden_dim=args.pfn_hidden_dim,
        pcnn_channels=args.pcnn_channels,
        pcnn_layers=args.pcnn_layers,
        fusion_hidden_dim=args.fusion_hidden_dim,
        part_embed_dim=args.part_embed_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        gate_bias_init=args.gate_bias_init,
        random_control_seed=args.random_control_seed,
        delta_l2_weight=args.delta_l2_weight,
        freeze_part_epochs=args.freeze_part_epochs,
        input_delta_scale=args.input_delta_scale,
        use_feature_wise_input_delta_scales=not bool(args.disable_feature_wise_input_delta_scales),
        freeze_input_delta_pid=bool(args.freeze_input_delta_pid),
        freeze_input_delta_geometry=bool(args.freeze_input_delta_geometry),
    )
    report = train_architecture_view_tagger(config)
    print("architecture_view_part_tagger_training_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  output_contract: {report['output_contract']}")
    print(f"  variant: {report['variant']}")
    print(f"  input_source: {report.get('input_source')}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  selection_metric: {report['selection_metric']}")
    print(f"  best_model_selection_metric_value: {report['best_model_selection_metric_value']:.8g}")
    print(f"  best_model_val_accuracy: {report['best_model_val_accuracy']:.6f}")
    if report.get("baseline_checkpoint_hash"):
        print(f"  baseline_checkpoint_hash: {report['baseline_checkpoint_hash']}")
    if report.get("stack_val_metrics"):
        stack_binary = report["stack_val_metrics"].get("binary_metrics", {})
        print(
            "  stack_val: "
            f"acc={report['stack_val_metrics'].get('accuracy'):.6f} "
            f"auc={stack_binary.get('auc')} "
            f"fpr50={stack_binary.get('fpr_at_signal_eff_0p50')}"
        )
    print(f"  final_test_evaluated: {report['final_test_evaluated']}")
    if report.get("final_test_metrics"):
        final_binary = report["final_test_metrics"].get("binary_metrics", {})
        print(
            "  final_test: "
            f"acc={report['final_test_metrics'].get('accuracy'):.6f} "
            f"auc={final_binary.get('auc')} "
            f"fpr50={final_binary.get('fpr_at_signal_eff_0p50')}"
        )
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {Path(args.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

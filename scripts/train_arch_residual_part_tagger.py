#!/usr/bin/env python3
"""Train one frozen-HLT-ParT architecture residual expert."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.arch_residual_part.model import ARCH_RESIDUAL_ARCHITECTURES  # noqa: E402
from teacher_logit_reco.arch_residual_part.train import (  # noqa: E402
    ARCH_RESIDUAL_SELECTION_METRICS,
    ArchResidualTrainConfig,
    local_compression_label_filter_names_to_indices,
    train_arch_residual_tagger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)

    parser.add_argument("--label-names", nargs="+", default=("QCD", "Hgg"))
    parser.add_argument("--label-filter-names", nargs="+", default=("QCD", "Hgg"))
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--stack-val-split", default="stack_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--confirm-split-settings", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")

    parser.add_argument("--seed", type=int, default=7307)
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
    parser.add_argument("--max-stack-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-stack-val-jets", type=int, default=None)
    parser.add_argument("--max-final-test-jets", type=int, default=None)
    parser.add_argument("--selection-metric", choices=ARCH_RESIDUAL_SELECTION_METRICS, default="fpr_at_signal_eff_0p50")
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

    parser.add_argument("--architecture", choices=ARCH_RESIDUAL_ARCHITECTURES, default="pfn")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--particle-layers", type=int, default=3)
    parser.add_argument("--global-layers", type=int, default=2)
    parser.add_argument("--edge-k", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--no-baseline-conditioning", action="store_true")
    parser.add_argument("--gamma-init", type=float, default=1.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label_names = tuple(str(name) for name in args.label_names)
    label_filter = local_compression_label_filter_names_to_indices(
        args.label_filter_names,
        manifest_path=args.manifest_path,
        label_names=label_names,
    )
    config = ArchResidualTrainConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        baseline_checkpoint=args.baseline_checkpoint,
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
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_hlt_params=not bool(args.skip_hlt_params_check),
        require_baseline_split_manifest_hash=bool(args.require_baseline_split_manifest_hash),
        expected_hlt_degradation_strength=args.expected_hlt_degradation_strength,
        label_names=label_names,
        label_filter=label_filter,
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        particle_layers=args.particle_layers,
        global_layers=args.global_layers,
        edge_k=args.edge_k,
        dropout=args.dropout,
        condition_on_baseline=not bool(args.no_baseline_conditioning),
        gamma_init=args.gamma_init,
        residual_scale=args.residual_scale,
        residual_l2_weight=args.residual_l2_weight,
    )
    report = train_arch_residual_tagger(config)
    print("arch_residual_part_training_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  output_contract: {report['output_contract']}")
    print(f"  architecture: {report['architecture']}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  selection_metric: {report['selection_metric']}")
    print(f"  best_model_selection_metric_value: {report['best_model_selection_metric_value']:.8g}")
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

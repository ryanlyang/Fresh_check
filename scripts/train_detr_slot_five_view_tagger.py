#!/usr/bin/env python3
"""Train DETR/free-slot five-view taggers and ablations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402
from teacher_logit_reco.set_matching.detr_slots.five_view import (  # noqa: E402
    DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS,
    build_detr_slot_five_view_tagger_config,
    train_detr_slot_five_view_suite,
    train_detr_slot_five_view_tagger,
)
from teacher_logit_reco.set_matching.five_view_data import FIVE_VIEW_SELECTION_MODES  # noqa: E402
from teacher_logit_reco.set_matching.five_view_train import FiveViewTaggerTrainConfig  # noqa: E402


def label_names_to_indices(values: list[str]) -> tuple[int, ...]:
    if not values:
        return ()
    by_name = {name: index for index, name in enumerate(LABEL_NAMES)}
    output: list[int] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            output.append(int(text))
            continue
        if text not in by_name:
            raise ValueError(f"Unknown JetClass label {text!r}; expected one of {list(LABEL_NAMES)}")
        output.append(by_name[text])
    return tuple(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS, default=list(DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--reconstructed-view-dir", required=True)

    parser.add_argument("--train-split", default="stack_train")
    parser.add_argument("--val-split", default="stack_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--confirm-split-settings", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")

    parser.add_argument("--seed", type=int, default=1205)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-final-test-jets", type=int, default=None)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "accuracy",
            "loss",
            "macro_per_class_accuracy",
            "auc",
            "fpr_at_signal_eff_0p30",
            "fpr_at_signal_eff_0p50",
            "background_rejection_at_signal_eff_0p30",
            "background_rejection_at_signal_eff_0p50",
        ),
        default="accuracy",
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--label-names", nargs="*", default=())
    parser.add_argument("--label-filter-names", nargs="*", default=())

    parser.add_argument("--max-tokens-per-view", type=int, default=128)
    parser.add_argument("--min-tokens-per-view", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--selection-mode", choices=tuple(FIVE_VIEW_SELECTION_MODES), default="topk_or_threshold")
    parser.add_argument("--view-label-shuffle-seed", type=int, default=1205)
    parser.add_argument("--skip-hlt-hash-check", action="store_true")

    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--stage1-layers", type=int, default=2)
    parser.add_argument("--stage1-heads", type=int, default=4)
    parser.add_argument("--stage2-layers", type=int, default=4)
    parser.add_argument("--stage2-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--classifier-hidden-dim", type=int, default=None)
    parser.add_argument("--disable-confidence", action="store_true")
    parser.add_argument("--disable-view-embedding", action="store_true")
    parser.add_argument("--disable-source-embedding", action="store_true")
    parser.add_argument("--disable-view-summaries", action="store_true")
    parser.add_argument("--geometry-hidden-dim", type=int, default=64)
    parser.add_argument("--geometry-dropout", type=float, default=0.0)
    return parser.parse_args()


def base_config_from_args(args: argparse.Namespace) -> FiveViewTaggerTrainConfig:
    first_variant = args.variants[0]
    return FiveViewTaggerTrainConfig(
        output_dir=str(Path(args.output_root) / first_variant),
        hlt_cache_dir=args.hlt_cache_dir,
        experiment_dir=args.experiment_dir,
        reconstructed_view_dir=args.reconstructed_view_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        final_test_split=args.final_test_split,
        confirm_split_settings=args.confirm_split_settings,
        confirm_final_test=args.confirm_final_test,
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
        max_final_test_batches=args.max_final_test_batches,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        selection_metric=args.selection_metric,
        compile_model=args.compile_model,
        num_classes=args.num_classes,
        label_names=tuple(args.label_names),
        label_filter=label_names_to_indices(list(args.label_filter_names)),
        max_tokens_per_view=args.max_tokens_per_view,
        min_tokens_per_view=args.min_tokens_per_view,
        confidence_threshold=args.confidence_threshold,
        selection_mode=args.selection_mode,
        drop_views=(),
        shuffle_view_labels=False,
        view_label_shuffle_seed=args.view_label_shuffle_seed,
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        embed_dim=args.embed_dim,
        stage1_layers=args.stage1_layers,
        stage1_heads=args.stage1_heads,
        stage2_layers=args.stage2_layers,
        stage2_heads=args.stage2_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        classifier_hidden_dim=args.classifier_hidden_dim,
        use_confidence=not bool(args.disable_confidence),
        use_view_embedding=not bool(args.disable_view_embedding),
        use_source_embedding=not bool(args.disable_source_embedding),
        use_view_summaries=not bool(args.disable_view_summaries),
        use_geometry_attention=False,
        geometry_hidden_dim=args.geometry_hidden_dim,
        geometry_dropout=args.geometry_dropout,
    )


def main() -> int:
    args = parse_args()
    base_config = base_config_from_args(args)
    if len(args.variants) == 1:
        variant = args.variants[0]
        config = build_detr_slot_five_view_tagger_config(
            base_config,
            variant=variant,
            output_root=args.output_root,
        )
        report = train_detr_slot_five_view_tagger(
            config,
            variant=variant,
            output_dir=config.output_dir,
        )
        print("detr_slot_five_view_tagger_training_complete:")
        print(f"  variant: {variant}")
        print(f"  output_dir: {config.output_dir}")
        print(f"  best_epoch: {report['best_epoch']}")
        print(f"  selection_metric: {report['selection_metric']}")
        print(f"  best_model_selection_metric_value: {report['best_model_selection_metric_value']:.8g}")
        if report.get("final_test_metrics"):
            print(f"  final_test_accuracy: {report['final_test_metrics']['accuracy']:.6f}")
        print(f"  checkpoint: {report['checkpoint']}")
        return 0

    suite = train_detr_slot_five_view_suite(
        base_config,
        variants=tuple(args.variants),
        output_root=args.output_root,
    )
    print("detr_slot_five_view_tagger_suite_complete:")
    print(f"  output_root: {args.output_root}")
    for variant in suite["variants"]:
        report = suite["reports"][variant]
        print(
            f"  {variant}: best_epoch={report['best_epoch']} "
            f"best_{report['selection_metric']}={report['best_model_selection_metric_value']:.8g}"
        )
    print(f"  report: {Path(args.output_root) / 'five_view_tagger_suite_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

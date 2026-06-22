#!/usr/bin/env python3
"""Train a Version A HLT-only reliability-gated subtoken Particle Transformer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402
from teacher_logit_reco.subtoken_part.config import (  # noqa: E402
    SUBTOKEN_PART_GATE_MODES,
    SUBTOKEN_PART_POOL_MODES,
    SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    SUBTOKEN_PART_VARIANT_LOCAL_GATE,
    SUBTOKEN_PART_VARIANT_NO_GATE,
)
from teacher_logit_reco.subtoken_part.train import (  # noqa: E402
    SUBTOKEN_PART_SELECTION_METRICS,
    SubtokenTaggerTrainConfig,
    train_subtoken_tagger,
)


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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")

    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--stack-val-split", default="stack_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--confirm-split-settings", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")

    parser.add_argument("--seed", type=int, default=2607)
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
    parser.add_argument("--selection-metric", choices=SUBTOKEN_PART_SELECTION_METRICS, default="accuracy")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--label-names", nargs="*", default=())
    parser.add_argument("--label-filter-names", nargs="*", default=())

    parser.add_argument(
        "--variant",
        choices=(
            SUBTOKEN_PART_VARIANT_NO_GATE,
            SUBTOKEN_PART_VARIANT_LOCAL_GATE,
            SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
        ),
        default=SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--local-layers", type=int, default=1)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--context-layers", type=int, default=2)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--global-layers", type=int, default=6)
    parser.add_argument("--global-heads", type=int, default=8)
    parser.add_argument("--gate-mode", choices=SUBTOKEN_PART_GATE_MODES, default="context_softmax")
    parser.add_argument("--local-pool-mode", choices=SUBTOKEN_PART_POOL_MODES, default="learned_query")
    parser.add_argument("--disable-pairwise-bias", action="store_true")
    parser.add_argument("--disable-particle-anchor", action="store_true")
    parser.add_argument("--disable-modality-type-embeddings", action="store_true")
    parser.add_argument("--use-pt-rank-embedding", action="store_true")
    parser.add_argument("--modality-dropout", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--anchor-source", choices=("raw", "part_features", "raw_and_part_features"), default="raw")
    parser.add_argument("--disable-part-style-derived-features", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SubtokenTaggerTrainConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
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
        num_classes=args.num_classes,
        label_names=tuple(args.label_names),
        label_filter=label_names_to_indices(list(args.label_filter_names)),
        variant=args.variant,
        embed_dim=args.embed_dim,
        local_layers=args.local_layers,
        local_heads=args.local_heads,
        context_layers=args.context_layers,
        context_heads=args.context_heads,
        global_layers=args.global_layers,
        global_heads=args.global_heads,
        gate_mode=args.gate_mode,
        local_pool_mode=args.local_pool_mode,
        use_pairwise_bias=not bool(args.disable_pairwise_bias),
        use_particle_anchor=not bool(args.disable_particle_anchor),
        use_modality_type_embeddings=not bool(args.disable_modality_type_embeddings),
        use_pt_rank_embedding=bool(args.use_pt_rank_embedding),
        modality_dropout=args.modality_dropout,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        anchor_source=args.anchor_source,
        include_part_style_derived_features=not bool(args.disable_part_style_derived_features),
    )
    report = train_subtoken_tagger(config)
    print("subtoken_part_tagger_training_complete:")
    print(f"  output_dir: {args.output_dir}")
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
            f"fpr50={final_binary.get('fpr_at_signal_eff_0p50')}"
        )
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {Path(args.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

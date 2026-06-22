#!/usr/bin/env python3
"""Train a DETR/free-slot set reconstructor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402
from teacher_logit_reco.set_matching.detr_slots.training import (  # noqa: E402
    DetrSlotReconstructorTrainConfig,
    train_detr_slot_reconstructor,
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
    parser.add_argument("--architecture", choices=("gt", "pn", "pfn", "pcnn"), required=True)
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--manifest-path", default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--label-filter-names", nargs="*", default=())
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--confirm-split-settings", action="store_true")

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
    parser.add_argument("--no-trim-to-valid", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--compile-model", action="store_true")

    parser.add_argument("--num-slots", type=int, default=160)
    parser.add_argument("--export-max-tokens", type=int, default=128)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-abs-eta", type=float, default=5.0)

    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--decoder-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--existence-bias", type=float, default=-2.0)
    parser.add_argument("--core-output-scale", type=float, default=1.0)

    parser.add_argument("--gt-layers", type=int, default=4)
    parser.add_argument("--gt-heads", type=int, default=4)
    parser.add_argument("--gt-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--edgeconv-dims", type=int, nargs="+", default=[64, 128, 128])
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--phi-dims", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--context-mlp-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[5, 5, 3, 3, 3, 3])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 1, 2, 4])

    parser.add_argument("--assignment-aux-weight", type=float, default=0.05)
    parser.add_argument("--matched-core-weight", type=float, default=1.0)
    parser.add_argument("--matched-aux-weight", type=float, default=0.10)
    parser.add_argument("--existence-weight", type=float, default=1.0)
    parser.add_argument("--existence-positive-weight", type=float, default=1.0)
    parser.add_argument("--existence-negative-weight", type=float, default=0.20)
    parser.add_argument("--count-weight", type=float, default=0.10)
    parser.add_argument("--jet-summary-weight", type=float, default=0.05)
    parser.add_argument("--duplicate-weight", type=float, default=0.0)
    parser.add_argument("--hlt-support-weight", type=float, default=0.0)
    parser.add_argument("--max-nearest-hlt-delta-r", type=float, default=0.8)
    parser.add_argument("--duplicate-delta-r-scale", type=float, default=0.10)
    parser.add_argument("--duplicate-probability-threshold", type=float, default=0.25)
    parser.add_argument("--max-count-for-summary", type=float, default=128.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--brute-force-fallback-limit", type=int, default=8)
    parser.add_argument("--allow-bruteforce-fallback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DetrSlotReconstructorTrainConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        data_dir=args.data_dir,
        architecture=args.architecture,
        train_split=args.train_split,
        val_split=args.val_split,
        confirm_split_settings=args.confirm_split_settings,
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
        label_filter=label_names_to_indices(list(args.label_filter_names)),
        trim_to_valid=not bool(args.no_trim_to_valid),
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_label_branches=args.verify_label_branches,
        read_chunk_size=args.read_chunk_size,
        compile_model=args.compile_model,
        num_slots=args.num_slots,
        export_max_tokens=args.export_max_tokens,
        memory_dim=args.memory_dim,
        context_dim=args.context_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        max_abs_eta=args.max_abs_eta,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_mlp_ratio=args.decoder_mlp_ratio,
        head_hidden_dim=args.head_hidden_dim,
        existence_bias=args.existence_bias,
        core_output_scale=args.core_output_scale,
        gt_layers=args.gt_layers,
        gt_heads=args.gt_heads,
        gt_mlp_ratio=args.gt_mlp_ratio,
        edgeconv_dims=tuple(args.edgeconv_dims),
        k=args.k,
        phi_dims=tuple(args.phi_dims),
        context_mlp_dims=tuple(args.context_mlp_dims),
        hidden_channels=args.hidden_channels,
        kernel_sizes=tuple(args.kernel_sizes),
        dilations=tuple(args.dilations),
        assignment_aux_weight=args.assignment_aux_weight,
        matched_core_weight=args.matched_core_weight,
        matched_aux_weight=args.matched_aux_weight,
        existence_weight=args.existence_weight,
        existence_positive_weight=args.existence_positive_weight,
        existence_negative_weight=args.existence_negative_weight,
        count_weight=args.count_weight,
        jet_summary_weight=args.jet_summary_weight,
        duplicate_weight=args.duplicate_weight,
        hlt_support_weight=args.hlt_support_weight,
        max_nearest_hlt_delta_r=args.max_nearest_hlt_delta_r,
        duplicate_delta_r_scale=args.duplicate_delta_r_scale,
        duplicate_probability_threshold=args.duplicate_probability_threshold,
        max_count_for_summary=args.max_count_for_summary,
        huber_beta=args.huber_beta,
        brute_force_fallback_limit=args.brute_force_fallback_limit,
        allow_bruteforce_fallback=args.allow_bruteforce_fallback,
    )
    report = train_detr_slot_reconstructor(config)
    print("detr_slot_reconstructor_training_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  architecture: {report['architecture']}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  best_model_val_total_loss: {report['best_model_val_total_loss']:.6f}")
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {Path(args.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a Step 5 set-matching reconstructor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.set_matching.train import (  # noqa: E402
    SetMatchingReconstructorTrainConfig,
    train_set_matching_reconstructor,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402


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
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--label-filter-names", nargs="*", default=())
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-slots", type=int, default=None)
    parser.add_argument("--confirm-split-settings", action="store_true")

    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
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

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--edgeconv-dims", type=int, nargs="+", default=[64, 128, 128])
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--phi-dims", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--context-mlp-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[5, 5, 3, 3, 3, 3])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 1, 2, 4])
    parser.add_argument("--embedding-dim", type=int, default=128)

    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num-extra-candidates", type=int, default=64)
    parser.add_argument("--max-delta-logpt", type=float, default=1.0)
    parser.add_argument("--max-delta-eta", type=float, default=0.20)
    parser.add_argument("--max-delta-phi", type=float, default=0.20)
    parser.add_argument("--max-delta-loge", type=float, default=1.0)
    parser.add_argument("--parent-weight-bias", type=float, default=2.0)
    parser.add_argument("--extra-weight-bias", type=float, default=-2.0)
    parser.add_argument("--max-total-extra-pt-fraction", type=float, default=0.50)
    parser.add_argument("--max-extra-delta-eta", type=float, default=1.50)
    parser.add_argument("--max-extra-delta-phi", type=float, default=1.50)
    parser.add_argument("--max-global-logpt-scale", type=float, default=0.35)
    parser.add_argument("--max-global-loge-scale", type=float, default=0.35)
    parser.add_argument("--max-global-eta-shift", type=float, default=0.05)
    parser.add_argument("--max-global-phi-shift", type=float, default=0.05)
    parser.add_argument("--extra-usage-weight-threshold", type=float, default=0.05)
    parser.add_argument("--eta-limit", type=float, default=5.0)
    parser.add_argument("--min-pt", type=float, default=1.0e-4)
    parser.add_argument("--weight-logit-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--output-weight-threshold", type=float, default=0.0)

    parser.add_argument("--matched-core-weight", type=float, default=1.0)
    parser.add_argument("--matched-aux-weight", type=float, default=0.10)
    parser.add_argument("--existence-weight", type=float, default=1.0)
    parser.add_argument("--existence-positive-weight", type=float, default=1.0)
    parser.add_argument("--count-weight", type=float, default=0.10)
    parser.add_argument("--jet-summary-weight", type=float, default=0.05)
    parser.add_argument("--correction-budget-weight", type=float, default=0.02)
    parser.add_argument("--chamfer-weight", type=float, default=0.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--max-abs-eta", type=float, default=5.0)
    parser.add_argument("--hlt-support-budget-weight", type=float, default=0.0)
    parser.add_argument("--max-nearest-hlt-delta-r", type=float, default=0.8)
    parser.add_argument("--skip-core-normalization", action="store_true")
    parser.add_argument("--brute-force-fallback-limit", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SetMatchingReconstructorTrainConfig(
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
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        edgeconv_dims=tuple(args.edgeconv_dims),
        k=args.k,
        phi_dims=tuple(args.phi_dims),
        context_dim=args.context_dim,
        context_mlp_dims=tuple(args.context_mlp_dims),
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        kernel_sizes=tuple(args.kernel_sizes),
        dilations=tuple(args.dilations),
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        num_extra_candidates=args.num_extra_candidates,
        max_delta_logpt=args.max_delta_logpt,
        max_delta_eta=args.max_delta_eta,
        max_delta_phi=args.max_delta_phi,
        max_delta_loge=args.max_delta_loge,
        parent_weight_bias=args.parent_weight_bias,
        extra_weight_bias=args.extra_weight_bias,
        max_total_extra_pt_fraction=args.max_total_extra_pt_fraction,
        max_extra_delta_eta=args.max_extra_delta_eta,
        max_extra_delta_phi=args.max_extra_delta_phi,
        max_global_logpt_scale=args.max_global_logpt_scale,
        max_global_loge_scale=args.max_global_loge_scale,
        max_global_eta_shift=args.max_global_eta_shift,
        max_global_phi_shift=args.max_global_phi_shift,
        extra_usage_weight_threshold=args.extra_usage_weight_threshold,
        eta_limit=args.eta_limit,
        min_pt=args.min_pt,
        weight_logit_epsilon=args.weight_logit_epsilon,
        output_weight_threshold=args.output_weight_threshold,
        matched_core_weight=args.matched_core_weight,
        matched_aux_weight=args.matched_aux_weight,
        existence_weight=args.existence_weight,
        existence_positive_weight=args.existence_positive_weight,
        count_weight=args.count_weight,
        jet_summary_weight=args.jet_summary_weight,
        correction_budget_weight=args.correction_budget_weight,
        chamfer_weight=args.chamfer_weight,
        huber_beta=args.huber_beta,
        max_slots=args.max_slots,
        max_abs_eta=args.max_abs_eta,
        hlt_support_budget_weight=args.hlt_support_budget_weight,
        max_nearest_hlt_delta_r=args.max_nearest_hlt_delta_r,
        use_core_normalization=not bool(args.skip_core_normalization),
        brute_force_fallback_limit=args.brute_force_fallback_limit,
    )
    report = train_set_matching_reconstructor(config)
    print("set_matching_reconstructor_training_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  architecture: {report['architecture']}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  best_model_val_total_loss: {report['best_model_val_total_loss']:.6f}")
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {Path(args.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

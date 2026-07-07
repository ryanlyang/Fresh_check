#!/usr/bin/env python3
"""Train one target-conditioned denoising ParT tagger variant."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.target_denoising_part import (  # noqa: E402
    TARGET_DENOISING_TAGGER_SELECTION_METRICS,
    TARGET_DENOISING_TAGGER_VARIANTS,
    TargetDenoisingTaggerTrainConfig,
    train_target_denoising_tagger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--denoiser-checkpoint", default=None)
    parser.add_argument("--variant", choices=TARGET_DENOISING_TAGGER_VARIANTS, required=True)
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--seed", type=int, default=7307)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-final-test-jets", type=int, default=None)
    parser.add_argument("--evaluate-final-test", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--selection-metric", choices=TARGET_DENOISING_TAGGER_SELECTION_METRICS, default="accuracy")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--allow-hlt-metadata-mismatch", action="store_true")
    parser.add_argument("--allow-manifest-mismatch", action="store_true")
    parser.add_argument("--allow-jet-identity-mismatch", action="store_true")
    parser.add_argument("--expected-hlt-profile", default="fixed_hlt_v2_realistic")
    parser.add_argument("--expected-hlt-profile-version", default="v1")
    parser.add_argument("--expected-hlt-degradation-strength", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--model-size", default="base")
    parser.add_argument("--part-embed-dim", type=int, default=128)
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--weight-threshold", type=float, default=0.0)
    parser.add_argument("--adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--adapter-gate-bias-init", type=float, default=-2.0)
    parser.add_argument("--freeze-denoiser", action="store_true")
    parser.add_argument("--train-denoiser", action="store_true")
    parser.add_argument("--allow-missing-denoiser-checkpoint", action="store_true")
    parser.add_argument("--non-strict-denoiser-checkpoint", action="store_true")
    parser.add_argument("--allow-incompatible-denoiser-checkpoint", action="store_true")
    parser.add_argument("--reconstruction-anchor-weight", type=float, default=0.0)
    parser.add_argument("--reconstruction-anchor-smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--alignment-mode", choices=("aligned_direct", "rank_direct"), default="aligned_direct")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--pair-hidden-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--disable-pair-bias", action="store_true")
    parser.add_argument("--disable-local-kernel", action="store_true")
    parser.add_argument("--local-kernel-radius", type=float, default=0.12)
    parser.add_argument("--local-kernel-init", type=float, default=0.0)
    parser.add_argument("--pair-bias-max-abs", type=float, default=4.0)
    parser.add_argument("--max-delta-log-pt", type=float, default=0.30)
    parser.add_argument("--max-delta-eta", type=float, default=0.08)
    parser.add_argument("--max-delta-phi", type=float, default=0.08)
    parser.add_argument("--max-delta-log-energy", type=float, default=0.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    freeze_denoiser = None
    if args.freeze_denoiser:
        freeze_denoiser = True
    if args.train_denoiser:
        freeze_denoiser = False
    config = TargetDenoisingTaggerTrainConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        data_dir=args.data_dir,
        denoiser_checkpoint=args.denoiser_checkpoint,
        variant=args.variant,
        train_split=args.train_split,
        val_split=args.val_split,
        final_test_split=args.final_test_split,
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
        max_final_test_batches=args.max_final_test_batches,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        evaluate_final_test=bool(args.evaluate_final_test),
        confirm_final_test=bool(args.confirm_final_test),
        selection_metric=args.selection_metric,
        compile_model=bool(args.compile_model),
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        strict_hlt_metadata=not bool(args.allow_hlt_metadata_mismatch),
        require_same_manifest_hash=not bool(args.allow_manifest_mismatch),
        require_same_jet_identity=not bool(args.allow_jet_identity_mismatch),
        expected_hlt_profile=args.expected_hlt_profile,
        expected_hlt_profile_version=args.expected_hlt_profile_version,
        expected_hlt_degradation_strength=args.expected_hlt_degradation_strength,
        num_classes=args.num_classes,
        model_size=args.model_size,
        part_embed_dim=args.part_embed_dim,
        max_constits=args.max_constits,
        weight_threshold=args.weight_threshold,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_dropout=args.adapter_dropout,
        adapter_gate_bias_init=args.adapter_gate_bias_init,
        freeze_denoiser=freeze_denoiser,
        require_denoiser_checkpoint=not bool(args.allow_missing_denoiser_checkpoint),
        strict_denoiser_checkpoint=not bool(args.non_strict_denoiser_checkpoint),
        require_compatible_denoiser_checkpoint=not bool(args.allow_incompatible_denoiser_checkpoint),
        reconstruction_anchor_weight=args.reconstruction_anchor_weight,
        reconstruction_anchor_smooth_l1_beta=args.reconstruction_anchor_smooth_l1_beta,
        alignment_mode=args.alignment_mode,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        pair_hidden_dim=args.pair_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        use_pair_bias=not bool(args.disable_pair_bias),
        use_local_kernel=not bool(args.disable_local_kernel),
        local_kernel_radius=args.local_kernel_radius,
        local_kernel_init=args.local_kernel_init,
        pair_bias_max_abs=args.pair_bias_max_abs,
        max_delta_log_pt=args.max_delta_log_pt,
        max_delta_eta=args.max_delta_eta,
        max_delta_phi=args.max_delta_phi,
        max_delta_log_energy=args.max_delta_log_energy,
    )
    report = train_target_denoising_tagger(config)
    print("target_conditioned_denoising_part_tagger_training_complete:")
    print(f"  output_dir: {report['output_dir']}")
    print(f"  output_contract: {report['output_contract']}")
    print(f"  variant: {report['variant']}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  selection_metric: {report['selection_metric']}")
    print(f"  best_model_selection_metric_value: {report['best_model_selection_metric_value']}")
    best_metrics = report.get("best_model_val_metrics", {})
    print(
        "  model_val: "
        f"acc={best_metrics.get('accuracy')} "
        f"loss={best_metrics.get('loss')} "
        f"n_jets={best_metrics.get('n_jets')}"
    )
    if report.get("final_test_evaluated"):
        final = report.get("final_test_metrics") or {}
        print(
            "  final_test: "
            f"acc={final.get('accuracy')} "
            f"loss={final.get('loss')} "
            f"n_jets={final.get('n_jets')}"
        )
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  run_report: {report['run_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

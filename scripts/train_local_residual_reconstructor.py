#!/usr/bin/env python3
"""Train a local particle residual-field reconstructor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    LOCAL_RESIDUAL_RECONSTRUCTOR_SELECTION_METRICS,
    LOCAL_RESIDUAL_RECONSTRUCTOR_VARIANTS,
    LocalResidualReconstructorTrainConfig,
    train_local_residual_reconstructor,
)


def _parse_field_group_weights(values: list[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"field group weight {value!r} must have the form group=value"
            )
        group, raw_weight = value.split("=", 1)
        group = group.strip()
        if not group:
            raise argparse.ArgumentTypeError("field group weight group name must not be empty")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid weight in {value!r}") from exc
        output[group] = weight
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--stack-val-split", default="stack_val")
    parser.add_argument("--seed", type=int, default=10421)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-stack-val-jets", type=int, default=None)
    parser.add_argument("--variant", default="C0", choices=LOCAL_RESIDUAL_RECONSTRUCTOR_VARIANTS + ("C0", "C1", "C2", "C3", "C4", "C5", "C6"))
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--num-heads", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--context-layers", type=int, default=1)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--local-radius", type=float, default=0.12)
    parser.add_argument("--hard-local-radius", type=float, default=0.08)
    parser.add_argument("--disable-zero-init-output", action="store_true")
    parser.add_argument(
        "--field-subset",
        nargs="*",
        default=[],
        help="Field groups or field names to train on. Empty means all fields.",
    )
    parser.add_argument(
        "--field-group-weight",
        action="append",
        default=[],
        help="Per-group loss weight in the form group=value. May be repeated.",
    )
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--uncertainty-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.0)
    parser.add_argument("--selection-metric", default="mae", choices=LOCAL_RESIDUAL_RECONSTRUCTOR_SELECTION_METRICS)
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--allow-missing-manifest-match", action="store_true")
    parser.add_argument("--no-save-last-checkpoint", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = LocalResidualReconstructorTrainConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        target_cache_dir=args.target_cache_dir,
        manifest_path=args.manifest_path or None,
        train_split=args.train_split,
        val_split=args.val_split,
        stack_val_split=args.stack_val_split,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        device=str(args.device),
        amp=not bool(args.disable_amp),
        grad_clip_norm=float(args.grad_clip_norm),
        early_stop_patience=int(args.early_stop_patience),
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_stack_val_jets=args.max_stack_val_jets,
        variant=str(args.variant),
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        context_layers=int(args.context_layers),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        attention_dropout=float(args.attention_dropout),
        local_radius=float(args.local_radius),
        hard_local_radius=float(args.hard_local_radius),
        use_zero_init_output=not bool(args.disable_zero_init_output),
        field_subset=tuple(args.field_subset),
        field_group_weights=_parse_field_group_weights(list(args.field_group_weight)),
        huber_beta=float(args.huber_beta),
        uncertainty_loss_weight=float(args.uncertainty_loss_weight),
        consistency_loss_weight=float(args.consistency_loss_weight),
        selection_metric=str(args.selection_metric),
        verify_hash=not bool(args.no_verify_hash),
        require_manifest_match=not bool(args.allow_missing_manifest_match),
        save_last_checkpoint=not bool(args.no_save_last_checkpoint),
    )
    report = train_local_residual_reconstructor(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

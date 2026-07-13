#!/usr/bin/env python3
"""Train an augmented ParT tagger with local particle residual fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402
from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    LOCAL_RESIDUAL_FIELD_TAGGER_SELECTION_METRICS,
    RESIDUAL_FIELD_SOURCES,
    LocalResidualFieldTaggerTrainConfig,
    train_local_residual_field_tagger,
)


def _parse_group_weights(values: list[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"group weight {value!r} must have form group=value")
        group, raw_weight = value.split("=", 1)
        group = group.strip()
        if not group:
            raise argparse.ArgumentTypeError("group name must not be empty")
        try:
            output[group] = float(raw_weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid numeric weight in {value!r}") from exc
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
    parser.add_argument("--seed", type=int, default=20421)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--part-lr", type=float, default=3.0e-5)
    parser.add_argument("--reconstructor-lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-stack-val-jets", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=len(LABEL_NAMES))
    parser.add_argument("--label-names", nargs="*", default=list(LABEL_NAMES))
    parser.add_argument("--model-size", default="base", choices=("tiny", "base", "large"))
    parser.add_argument("--field-source", default="frozen_reconstructor", choices=RESIDUAL_FIELD_SOURCES)
    parser.add_argument("--reconstructor-checkpoint", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--require-baseline-warm-start", action="store_true")
    parser.add_argument("--residual-field-scale", type=float, default=1.0)
    parser.add_argument("--field-dropout", type=float, default=0.0)
    parser.add_argument("--control-seed", type=int, default=9173)
    parser.add_argument("--control-noise-scale", type=float, default=1.0)
    parser.add_argument("--control-random-unit-std", action="store_true")
    parser.add_argument("--learned-control-hidden-dim", type=int, default=128)
    parser.add_argument("--learned-control-dropout", type=float, default=0.05)
    parser.add_argument("--field-subset", nargs="*", default=[])
    parser.add_argument("--reconstructor-loss-weight", type=float, default=0.0)
    parser.add_argument("--reconstructor-field-group-weight", action="append", default=[])
    parser.add_argument("--reconstructor-huber-beta", type=float, default=0.1)
    parser.add_argument("--reconstructor-uncertainty-loss-weight", type=float, default=1.0)
    parser.add_argument("--reconstructor-consistency-loss-weight", type=float, default=0.0)
    parser.add_argument("--kd-loss-weight", type=float, default=0.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-logits-dir", default="")
    parser.add_argument("--teacher-logits-train-path", default="")
    parser.add_argument("--teacher-logits-val-path", default="")
    parser.add_argument("--teacher-logits-stack-val-path", default="")
    parser.add_argument("--selection-metric", default="accuracy", choices=LOCAL_RESIDUAL_FIELD_TAGGER_SELECTION_METRICS)
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--allow-missing-manifest-match", action="store_true")
    parser.add_argument("--no-save-last-checkpoint", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = LocalResidualFieldTaggerTrainConfig(
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
        part_lr=float(args.part_lr),
        reconstructor_lr=float(args.reconstructor_lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        device=str(args.device),
        amp=not bool(args.disable_amp),
        grad_clip_norm=float(args.grad_clip_norm),
        early_stop_patience=int(args.early_stop_patience),
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_stack_val_jets=args.max_stack_val_jets,
        num_classes=int(args.num_classes),
        label_names=tuple(args.label_names),
        model_size=str(args.model_size),
        field_source=str(args.field_source),
        reconstructor_checkpoint=args.reconstructor_checkpoint or None,
        baseline_checkpoint=args.baseline_checkpoint or None,
        require_baseline_warm_start=bool(args.require_baseline_warm_start),
        residual_field_scale=float(args.residual_field_scale),
        field_dropout=float(args.field_dropout),
        control_seed=int(args.control_seed),
        control_noise_scale=float(args.control_noise_scale),
        control_random_match_target_std=not bool(args.control_random_unit_std),
        learned_control_hidden_dim=int(args.learned_control_hidden_dim),
        learned_control_dropout=float(args.learned_control_dropout),
        field_subset=tuple(args.field_subset),
        reconstructor_loss_weight=float(args.reconstructor_loss_weight),
        reconstructor_field_group_weights=_parse_group_weights(list(args.reconstructor_field_group_weight)),
        reconstructor_huber_beta=float(args.reconstructor_huber_beta),
        reconstructor_uncertainty_loss_weight=float(args.reconstructor_uncertainty_loss_weight),
        reconstructor_consistency_loss_weight=float(args.reconstructor_consistency_loss_weight),
        kd_loss_weight=float(args.kd_loss_weight),
        kd_temperature=float(args.kd_temperature),
        teacher_logits_dir=args.teacher_logits_dir or None,
        teacher_logits_train_path=args.teacher_logits_train_path or None,
        teacher_logits_val_path=args.teacher_logits_val_path or None,
        teacher_logits_stack_val_path=args.teacher_logits_stack_val_path or None,
        selection_metric=str(args.selection_metric),
        verify_hash=not bool(args.no_verify_hash),
        require_manifest_match=not bool(args.allow_missing_manifest_match),
        save_last_checkpoint=not bool(args.no_save_last_checkpoint),
    )
    report = train_local_residual_field_tagger(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

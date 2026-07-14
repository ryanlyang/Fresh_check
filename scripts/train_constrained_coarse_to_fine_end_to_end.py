#!/usr/bin/env python3
"""Train one frozen or staged constrained pseudo-offline fusion tagger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine import (  # noqa: E402
    EndToEndLossConfig,
    EndToEndScheduleConfig,
    EndToEndTrainConfig,
    ReconstructorSourceSpec,
    train_end_to_end_tagger,
    fusion_variant_spec,
    normalize_fusion_variant,
)


def _sources(values: list[str], aliases: list[str], variants: list[str]) -> tuple[ReconstructorSourceSpec, ...]:
    """Parse NAME[@VIEW_INDEX]=CHECKPOINT without breaking Windows drive letters."""

    alias_map = dict(value.split("=", 1) for value in aliases)
    variant_map = dict(value.split("=", 1) for value in variants)
    grouped: dict[tuple[str, str], list[tuple[str, int]]] = {}
    order: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --reconstructor-source {value!r}; expected NAME[@INDEX]=PATH")
        left, checkpoint = value.split("=", 1)
        if "@" in left:
            view_name, raw_index = left.rsplit("@", 1)
            view_index = int(raw_index)
        else:
            view_name, view_index = left, 0
        source_name = view_name
        key = (source_name, checkpoint)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((view_name, view_index))
    specs: list[ReconstructorSourceSpec] = []
    for source_name, checkpoint in order:
        rows = grouped[(source_name, checkpoint)]
        specs.append(
            ReconstructorSourceSpec(
                name=source_name,
                checkpoint_path=checkpoint,
                view_names=tuple(name for name, _ in rows),
                view_indices=tuple(index for _, index in rows),
                expected_variant=variant_map.get(source_name),
            )
        )
    # Explicit aliases use ALIAS=TARGET and must also have their own source row.
    by_name = {row.name: row for row in specs}
    for alias, target in alias_map.items():
        if alias == target or alias not in by_name:
            continue
        row = by_name[alias]
        by_name[alias] = ReconstructorSourceSpec(**{**row.__dict__, "alias_of": target})
    return tuple(by_name[row.name] for row in specs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", dest="manifest_path", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", required=True)
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--variant", default="D5")
    parser.add_argument(
        "--reconstructor-source",
        action="append",
        default=[],
        metavar="VIEW[@INDEX]=CHECKPOINT",
        help="Repeat for every active pseudo view; repeated paths are loaded once.",
    )
    parser.add_argument(
        "--reconstructor-alias",
        action="append",
        default=[],
        metavar="ALIAS=TARGET",
        help="Require ALIAS to reuse TARGET's exact checkpoint/config and omit its duplicate view.",
    )
    parser.add_argument(
        "--reconstructor-variant",
        action="append",
        default=[],
        metavar="SOURCE=VARIANT",
        help="Optional strict checkpoint variant assertion.",
    )
    parser.add_argument("--hlt-warm-start-checkpoint", default=None)
    parser.add_argument("--allow-random-hlt-start", action="store_true")
    parser.add_argument("--teacher-logits-train-path", default=None)
    parser.add_argument("--teacher-logits-val-path", default=None)
    parser.add_argument("--seed", type=int, default=28031)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-nonfinite-batches", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--no-save-last-checkpoint", action="store_true")
    parser.add_argument("--fusion-only-warmup-epochs", type=int, default=1)
    parser.add_argument("--frozen-reconstructor-epochs", type=int, default=4)
    parser.add_argument("--terminal-decoder-epochs", type=int, default=4)
    parser.add_argument("--upper-hierarchy-epochs", type=int, default=2)
    parser.add_argument("--unfreeze-reconstructor-hlt-encoder", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--terminal-decoder-lr-scale", type=float, default=0.075)
    parser.add_argument("--upper-hierarchy-lr-scale", type=float, default=0.035)
    parser.add_argument("--reconstructor-hlt-encoder-lr-scale", type=float, default=0.03)
    parser.add_argument("--reconstruction-weight", type=float, default=0.10)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01)
    parser.add_argument("--reconstruction-slot-weight", type=float, default=1.0)
    parser.add_argument("--kd-loss-weight", type=float, default=0.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hlt-encoder-layers", type=int, default=6)
    parser.add_argument("--pseudo-local-layers", type=int, default=2)
    parser.add_argument("--pseudo-global-layers", type=int, default=3)
    parser.add_argument("--fusion-layers", type=int, default=3)
    parser.add_argument("--pseudo-view-dropout", type=float, default=0.15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    normalized_variant = normalize_fusion_variant(args.variant)
    if not args.reconstructor_source and fusion_variant_spec(normalized_variant).num_pseudo_views:
        raise SystemExit("at least one --reconstructor-source is required")
    sources = _sources(args.reconstructor_source, args.reconstructor_alias, args.reconstructor_variant)
    schedule = EndToEndScheduleConfig(
        fusion_only_warmup_epochs=args.fusion_only_warmup_epochs,
        frozen_reconstructor_epochs=args.frozen_reconstructor_epochs,
        terminal_decoder_epochs=args.terminal_decoder_epochs,
        upper_hierarchy_epochs=args.upper_hierarchy_epochs,
        unfreeze_reconstructor_hlt_encoder=args.unfreeze_reconstructor_hlt_encoder,
        tagger_learning_rate=args.learning_rate,
        terminal_decoder_lr_scale=args.terminal_decoder_lr_scale,
        upper_hierarchy_lr_scale=args.upper_hierarchy_lr_scale,
        reconstructor_hlt_encoder_lr_scale=args.reconstructor_hlt_encoder_lr_scale,
        reconstruction_weight=args.reconstruction_weight,
        gate_entropy_weight=args.gate_entropy_weight,
    )
    loss = EndToEndLossConfig(
        reconstruction_slot_weight=args.reconstruction_slot_weight,
        kd_loss_weight=args.kd_loss_weight,
        kd_temperature=args.kd_temperature,
    )
    fusion_overrides = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "hlt_encoder_layers": args.hlt_encoder_layers,
        "pseudo_local_layers": args.pseudo_local_layers,
        "pseudo_global_layers": args.pseudo_global_layers,
        "fusion_layers": args.fusion_layers,
        "pseudo_view_dropout": args.pseudo_view_dropout,
    }
    config = EndToEndTrainConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        target_cache_dir=args.target_cache_dir,
        reconstructor_sources=sources,
        variant=args.variant,
        hlt_warm_start_checkpoint=args.hlt_warm_start_checkpoint,
        allow_random_hlt_start=args.allow_random_hlt_start,
        teacher_logits_train_path=args.teacher_logits_train_path,
        teacher_logits_val_path=args.teacher_logits_val_path,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        max_nonfinite_batches=args.max_nonfinite_batches,
        device=args.device,
        amp=not args.no_amp,
        verify_hash=not args.no_verify_hash,
        pin_memory=not args.no_pin_memory,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        save_last_checkpoint=not args.no_save_last_checkpoint,
        schedule=schedule,
        loss=loss,
        fusion_overrides=fusion_overrides,
    )
    report = train_end_to_end_tagger(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

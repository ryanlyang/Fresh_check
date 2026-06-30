#!/usr/bin/env python3
"""Train the Step 13 local-compression pilot variants under one output root."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from teacher_logit_reco.local_compression_part.config import (  # noqa: E402
    LOCAL_COMPRESSION_DEFAULT_PILOT_VARIANTS,
    LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID,
    LOCAL_COMPRESSION_GATE_NONE,
    LOCAL_COMPRESSION_GATE_MODES,
    LOCAL_COMPRESSION_POOL_MODES,
    LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
    LOCAL_COMPRESSION_VARIANT_LARGER_HLT_PART_CONTROL,
    LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING,
    LOCAL_COMPRESSION_VARIANTS,
    normalize_local_compression_variant,
)
from teacher_logit_reco.local_compression_part.train import (  # noqa: E402
    LOCAL_COMPRESSION_SELECTION_METRICS,
    LOCAL_COMPRESSION_TRAIN_CONTRACT,
    LOCAL_COMPRESSION_TRAIN_STEP,
    LocalCompressionTaggerTrainConfig,
    local_compression_label_filter_names_to_indices,
    train_local_compression_tagger,
)


def _variant_gate_mode(variant: str, explicit_gate_mode: str | None) -> str:
    if explicit_gate_mode is not None:
        return str(explicit_gate_mode)
    if variant in {LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED, LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING}:
        return LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID
    return LOCAL_COMPRESSION_GATE_NONE


def _variant_output_dir(root: Path, variant: str, *, overwrite: bool) -> Path:
    output_dir = root / variant
    if output_dir.exists() and not bool(overwrite):
        raise FileExistsError(f"variant output already exists: {output_dir}; pass --overwrite to reuse it")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--variants", nargs="+", default=list(LOCAL_COMPRESSION_DEFAULT_PILOT_VARIANTS))
    parser.add_argument("--include-larger-control", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--label-names", nargs="+", default=("QCD", "Hgg"))
    parser.add_argument("--label-filter-names", nargs="+", default=("QCD", "Hgg"))
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--stack-val-split", default="stack_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--confirm-split-settings", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")

    parser.add_argument("--seed", type=int, default=3207)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--adapter-lr", type=float, default=3.0e-4)
    parser.add_argument("--part-lr", type=float, default=3.0e-5)
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
        choices=LOCAL_COMPRESSION_SELECTION_METRICS,
        default="fpr_at_signal_eff_0p50",
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--skip-hlt-params-check", action="store_true")
    parser.add_argument("--require-baseline-split-manifest-hash", action="store_true")
    parser.add_argument("--expected-hlt-degradation-strength", type=float, default=0.6)

    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--context-layers", type=int, default=1)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--pool-mode", choices=LOCAL_COMPRESSION_POOL_MODES, default="learned_query")
    parser.add_argument("--gate-mode", choices=LOCAL_COMPRESSION_GATE_MODES, default=None)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--freeze-pid-deltas", action="store_true")
    parser.add_argument("--freeze-geometry-deltas", action="store_true")
    parser.add_argument("--freeze-part-epochs", type=int, default=0)
    parser.add_argument("--random-grouping-seed", type=int, default=2907)
    return parser.parse_args()


def _normalize_variants(values: list[str], *, include_larger_control: bool) -> tuple[str, ...]:
    variants = [normalize_local_compression_variant(value) for value in values]
    if not include_larger_control:
        variants = [variant for variant in variants if variant != LOCAL_COMPRESSION_VARIANT_LARGER_HLT_PART_CONTROL]
    if LOCAL_COMPRESSION_VARIANT_LARGER_HLT_PART_CONTROL in variants:
        raise ValueError(
            "lc_larger_hlt_part_control is not implemented as a true larger canonical ParT; "
            "remove --include-larger-control until that control exists"
        )
    output: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant not in LOCAL_COMPRESSION_VARIANTS:
            raise ValueError(f"unknown local-compression variant {variant!r}")
        if variant not in seen:
            seen.add(variant)
            output.append(variant)
    if not output:
        raise ValueError("no local-compression variants requested")
    return tuple(output)


def _config_for_variant(args: argparse.Namespace, variant: str, output_dir: Path) -> LocalCompressionTaggerTrainConfig:
    label_names = tuple(str(name) for name in args.label_names)
    label_filter = local_compression_label_filter_names_to_indices(
        args.label_filter_names,
        manifest_path=args.manifest_path,
        label_names=label_names,
    )
    return LocalCompressionTaggerTrainConfig(
        output_dir=str(output_dir),
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        baseline_checkpoint=args.baseline_checkpoint,
        train_split=args.train_split,
        val_split=args.val_split,
        stack_val_split=args.stack_val_split,
        final_test_split=args.final_test_split,
        confirm_split_settings=bool(args.confirm_split_settings),
        confirm_final_test=bool(args.confirm_final_test),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        epochs=int(args.epochs),
        adapter_lr=float(args.adapter_lr),
        part_lr=float(args.part_lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        device=str(args.device),
        amp=not bool(args.no_amp),
        grad_clip_norm=float(args.grad_clip_norm),
        early_stop_patience=int(args.early_stop_patience),
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
        verify_hlt_params=not bool(args.skip_hlt_params_check),
        require_baseline_split_manifest_hash=bool(args.require_baseline_split_manifest_hash),
        expected_hlt_degradation_strength=float(args.expected_hlt_degradation_strength),
        label_names=label_names,
        label_filter=label_filter,
        variant=variant,
        random_grouping_seed=int(args.random_grouping_seed),
        embed_dim=int(args.embed_dim),
        local_layers=int(args.local_layers),
        local_heads=int(args.local_heads),
        context_layers=int(args.context_layers),
        context_heads=int(args.context_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        attention_dropout=float(args.attention_dropout),
        pool_mode=args.pool_mode,
        gate_mode=_variant_gate_mode(variant, args.gate_mode),
        delta_scale=float(args.delta_scale),
        freeze_pid_deltas=bool(args.freeze_pid_deltas),
        freeze_geometry_deltas=bool(args.freeze_geometry_deltas),
        freeze_part_epochs=int(args.freeze_part_epochs),
    )


def _summary_row(report: dict[str, Any]) -> dict[str, Any]:
    final_metrics = report.get("final_test_metrics") if isinstance(report.get("final_test_metrics"), dict) else {}
    final_binary = final_metrics.get("binary_metrics") if isinstance(final_metrics.get("binary_metrics"), dict) else {}
    val_metrics = report.get("best_model_val_metrics") if isinstance(report.get("best_model_val_metrics"), dict) else {}
    val_binary = val_metrics.get("binary_metrics") if isinstance(val_metrics.get("binary_metrics"), dict) else {}
    return {
        "variant": report.get("config", {}).get("variant"),
        "output_dir": str(Path(report.get("checkpoint", "")).parent) if report.get("checkpoint") else None,
        "best_epoch": report.get("best_epoch"),
        "model_val_fpr_at_signal_eff_0p50": val_binary.get("fpr_at_signal_eff_0p50"),
        "model_val_auc": val_binary.get("auc"),
        "final_test_fpr_at_signal_eff_0p50": final_binary.get("fpr_at_signal_eff_0p50"),
        "final_test_auc": final_binary.get("auc"),
        "final_test_accuracy": final_metrics.get("accuracy"),
        "checkpoint": report.get("checkpoint"),
        "run_report": str(Path(report.get("checkpoint", "")).parent / "run_report.json") if report.get("checkpoint") else None,
    }


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    variants = _normalize_variants(list(args.variants), include_larger_control=bool(args.include_larger_control))
    reports = []
    rows = []
    for variant in variants:
        output_dir = _variant_output_dir(output_root, variant, overwrite=bool(args.overwrite))
        config = _config_for_variant(args, variant, output_dir)
        print(f"local_compression_variant_start: {variant}", flush=True)
        report = train_local_compression_tagger(config)
        reports.append(report)
        row = _summary_row(report)
        rows.append(row)
        print(
            "local_compression_variant_complete: "
            f"{variant} final_fpr50={row['final_test_fpr_at_signal_eff_0p50']} "
            f"final_auc={row['final_test_auc']}",
            flush=True,
        )
    suite_report = {
        "experiment_step": "local_compression_part_step13_variants",
        "train_step": LOCAL_COMPRESSION_TRAIN_STEP,
        "train_contract": LOCAL_COMPRESSION_TRAIN_CONTRACT,
        "output_root": str(output_root),
        "variants": list(variants),
        "args": vars(args),
        "configs": [asdict(_config_for_variant(args, variant, output_root / variant)) for variant in variants],
        "rows": rows,
        "run_reports": [
            str(Path(report["checkpoint"]).parent / "run_report.json")
            for report in reports
            if report.get("checkpoint")
        ],
    }
    save_json(output_root / "variant_suite_report.json", suite_report)
    print("local_compression_variant_suite_complete:")
    print(f"  output_root: {output_root}")
    print(f"  variants: {' '.join(variants)}")
    print(f"  suite_report: {output_root / 'variant_suite_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

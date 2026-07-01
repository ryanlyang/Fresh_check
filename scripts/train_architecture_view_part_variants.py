#!/usr/bin/env python3
"""Train the Architecture-View Residual ParT pilot variants under one root."""

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
from teacher_logit_reco.architecture_view_part.config import (  # noqa: E402
    ARCHITECTURE_VIEW_DEFAULT_PILOT_VARIANTS,
    ARCHITECTURE_VIEW_VARIANTS,
    normalize_architecture_view_variant,
)
from teacher_logit_reco.architecture_view_part.train import (  # noqa: E402
    ARCHITECTURE_VIEW_SELECTION_METRICS,
    ARCHITECTURE_VIEW_TRAIN_CONTRACT,
    ARCHITECTURE_VIEW_TRAIN_STEP,
    ArchitectureViewTaggerTrainConfig,
    architecture_view_label_filter_names_to_indices,
    train_architecture_view_tagger,
)


def _variant_output_dir(root: Path, variant: str, *, overwrite: bool) -> Path:
    output_dir = root / variant
    if output_dir.exists() and not bool(overwrite):
        raise FileExistsError(f"variant output already exists: {output_dir}; pass --overwrite to reuse it")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _normalize_variants(values: list[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        variant = normalize_architecture_view_variant(value)
        if variant not in ARCHITECTURE_VIEW_VARIANTS:
            raise ValueError(f"unknown architecture-view variant {variant!r}")
        if variant not in seen:
            output.append(variant)
            seen.add(variant)
    if not output:
        raise ValueError("no architecture-view variants requested")
    return tuple(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--variants", nargs="+", default=list(ARCHITECTURE_VIEW_DEFAULT_PILOT_VARIANTS))
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--label-names", nargs="+", default=("QCD", "Hgg"))
    parser.add_argument("--label-filter-names", nargs="+", default=("QCD", "Hgg"))
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--stack-val-split", default="stack_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--confirm-split-settings", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")

    parser.add_argument("--seed", type=int, default=6207)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--adapter-lr", type=float, default=3.0e-4)
    parser.add_argument("--part-lr", type=float, default=1.0e-5)
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
        choices=ARCHITECTURE_VIEW_SELECTION_METRICS,
        default="fpr_at_signal_eff_0p50",
    )
    parser.add_argument("--compile-model", action="store_true")
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

    parser.add_argument("--view-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--pn-k", type=int, default=16)
    parser.add_argument("--pn-layers", type=int, default=2)
    parser.add_argument("--pfn-hidden-dim", type=int, default=64)
    parser.add_argument("--pcnn-channels", type=int, default=64)
    parser.add_argument("--pcnn-layers", type=int, default=2)
    parser.add_argument("--fusion-hidden-dim", type=int, default=96)
    parser.add_argument("--part-embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--gate-bias-init", type=float, default=-5.0)
    parser.add_argument("--random-control-seed", type=int, default=2907)
    parser.add_argument("--delta-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--freeze-part-epochs", type=int, default=2)
    return parser.parse_args()


def _config_for_variant(args: argparse.Namespace, variant: str, output_dir: Path) -> ArchitectureViewTaggerTrainConfig:
    label_names = tuple(str(name) for name in args.label_names)
    label_filter = architecture_view_label_filter_names_to_indices(
        args.label_filter_names,
        manifest_path=args.manifest_path,
        label_names=label_names,
    )
    return ArchitectureViewTaggerTrainConfig(
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
        view_dim=int(args.view_dim),
        hidden_dim=int(args.hidden_dim),
        pn_k=int(args.pn_k),
        pn_layers=int(args.pn_layers),
        pfn_hidden_dim=int(args.pfn_hidden_dim),
        pcnn_channels=int(args.pcnn_channels),
        pcnn_layers=int(args.pcnn_layers),
        fusion_hidden_dim=int(args.fusion_hidden_dim),
        part_embed_dim=int(args.part_embed_dim),
        dropout=float(args.dropout),
        attention_dropout=float(args.attention_dropout),
        gate_bias_init=float(args.gate_bias_init),
        random_control_seed=int(args.random_control_seed),
        delta_l2_weight=float(args.delta_l2_weight),
        freeze_part_epochs=int(args.freeze_part_epochs),
    )


def _summary_row(report: Mapping[str, Any]) -> dict[str, Any]:
    final_metrics = report.get("final_test_metrics") if isinstance(report.get("final_test_metrics"), Mapping) else {}
    final_binary = final_metrics.get("binary_metrics") if isinstance(final_metrics.get("binary_metrics"), Mapping) else {}
    val_metrics = report.get("best_model_val_metrics") if isinstance(report.get("best_model_val_metrics"), Mapping) else {}
    val_binary = val_metrics.get("binary_metrics") if isinstance(val_metrics.get("binary_metrics"), Mapping) else {}
    return {
        "variant": report.get("variant"),
        "best_epoch": report.get("best_epoch"),
        "model_val_fpr_at_signal_eff_0p50": val_binary.get("fpr_at_signal_eff_0p50"),
        "model_val_auc": val_binary.get("auc"),
        "final_test_fpr_at_signal_eff_0p50": final_binary.get("fpr_at_signal_eff_0p50"),
        "final_test_auc": final_binary.get("auc"),
        "final_test_accuracy": final_metrics.get("accuracy"),
        "checkpoint": report.get("checkpoint"),
        "run_report": str(Path(str(report.get("checkpoint", ""))).parent / "run_report.json")
        if report.get("checkpoint")
        else None,
    }


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    variants = _normalize_variants(list(args.variants))
    reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    child_reports: dict[str, str] = {}
    for variant in variants:
        output_dir = _variant_output_dir(output_root, variant, overwrite=bool(args.overwrite))
        config = _config_for_variant(args, variant, output_dir)
        print(f"architecture_view_variant_start: {variant}", flush=True)
        report = train_architecture_view_tagger(config)
        reports.append(report)
        row = _summary_row(report)
        rows.append(row)
        child_reports[variant] = str(output_dir / "run_report.json")
        print(
            "architecture_view_variant_complete: "
            f"{variant} final_fpr50={row['final_test_fpr_at_signal_eff_0p50']} "
            f"final_auc={row['final_test_auc']}",
            flush=True,
        )
    suite_report = {
        "experiment_step": "architecture_view_part_step3_variants",
        "train_step": ARCHITECTURE_VIEW_TRAIN_STEP,
        "train_contract": ARCHITECTURE_VIEW_TRAIN_CONTRACT,
        "output_root": str(output_root),
        "variants": list(variants),
        "args": vars(args),
        "configs": [asdict(_config_for_variant(args, variant, output_root / variant)) for variant in variants],
        "rows": rows,
        "child_reports": child_reports,
        "run_reports": list(child_reports.values()),
    }
    save_json(output_root / "variant_suite_report.json", suite_report)
    print("architecture_view_variant_suite_complete:")
    print(f"  output_root: {output_root}")
    print(f"  variants: {' '.join(variants)}")
    print(f"  variant_suite_report: {output_root / 'variant_suite_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect Architecture-View ParT predictions and run 10-class stacked fusion."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import DEFAULT_C_GRID, PredictionBlock, STACK_SPLITS, prediction_paths, save_prediction_block, softmax_np  # noqa: E402
from jetclass_fresh.hlt_baseline import resolve_device, save_json  # noqa: E402
from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from jetclass_fresh.independent_fusion import FEATURE_MODES, IndependentFusionConfig, run_independent_fusion  # noqa: E402
from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402
from teacher_logit_reco.architecture_view_part.config import (  # noqa: E402
    ARCHITECTURE_VIEW_VARIANTS,
    normalize_architecture_view_variant,
)
from teacher_logit_reco.architecture_view_part.train import (  # noqa: E402
    load_architecture_view_tagger_checkpoint,
)
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader  # noqa: E402


@dataclass(frozen=True)
class ArchitectureViewFusionRunConfig:
    cache_dir: str
    checkpoint_root: str
    output_dir: str
    variants: tuple[str, ...]
    splits: tuple[str, ...] = tuple(STACK_SPLITS)
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    stack_train_size: int | None = 500000
    stack_val_size: int | None = 150000
    final_test_size: int | None = 500000
    overwrite_predictions: bool = False
    skip_existing_predictions: bool = True
    confirm_final_test: bool = False
    feature_modes: tuple[str, ...] = tuple(FEATURE_MODES)
    c_grid: tuple[float, ...] = tuple(DEFAULT_C_GRID)
    max_iter: int = 2000
    run_controls: bool = True
    control_seed: int = 12345


def _split_size(config: ArchitectureViewFusionRunConfig, split: str) -> int | None:
    if split == "stack_train":
        return config.stack_train_size
    if split == "stack_val":
        return config.stack_val_size
    if split == "final_test":
        return config.final_test_size
    return None


def _checkpoint_path(config: ArchitectureViewFusionRunConfig, variant: str) -> Path:
    return Path(config.checkpoint_root) / variant / "best_model_val.pt"


def _prediction_block(
    model,
    view,
    *,
    variant: str,
    batch_size: int,
    num_workers: int,
    device,
    max_jets: int | None,
    label_names: tuple[str, ...],
    label_filter: tuple[int, ...],
    selection_seed: int,
) -> PredictionBlock:
    del selection_seed
    dataset = SubtokenHLTJetDataset(
        view,
        label_filter=label_filter,
        label_names=label_names,
        max_jets=max_jets,
    )
    loader = make_subtoken_hlt_loader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        seed=12345,
    )
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    import torch

    model.eval()
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True).float()
            mask = batch["mask"].to(device, non_blocking=True).bool()
            logits = model(tokens, mask, max_constits=int(tokens.shape[1]))
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
    logits_np = np.concatenate(logits_rows, axis=0)
    labels_np = np.concatenate(labels_rows, axis=0)
    return PredictionBlock(
        model_name=variant,
        split=view.split,
        logits=logits_np,
        probs=softmax_np(logits_np),
        labels=labels_np,
        jet_ids=list(view.jet_ids[: len(labels_np)]),
        metadata={
            "model_kind": "architecture_view_residual_part",
            "variant": str(variant),
            "hlt_content_hash": view.metadata.get("hlt_content_hash"),
            "allowed_inputs": "cached_fixed_hlt_only",
            "label_names": list(label_names),
            "label_filter": list(label_filter),
        },
    )


def collect_architecture_view_predictions(config: ArchitectureViewFusionRunConfig) -> dict[str, Any]:
    import torch

    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    prediction_dir = output_dir / "predictions"
    reports: dict[str, Any] = {}
    label_names = tuple(str(name) for name in LABEL_NAMES)
    label_filter = tuple(range(len(label_names)))
    for variant in config.variants:
        checkpoint = _checkpoint_path(config, variant)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing Architecture-View checkpoint for {variant}: {checkpoint}")
        model, payload = load_architecture_view_tagger_checkpoint(checkpoint, device=device)
        reports[variant] = {}
        payload_config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
        if isinstance(payload_config, Mapping):
            label_names = tuple(str(name) for name in payload_config.get("label_names", label_names))
            label_filter = tuple(int(index) for index in payload_config.get("label_filter", label_filter))
        for split in config.splits:
            npz_path, _ = prediction_paths(prediction_dir, variant, split)
            if npz_path.exists() and config.skip_existing_predictions and not config.overwrite_predictions:
                from jetclass_fresh.fusion import load_prediction_block

                reports[variant][split] = load_prediction_block(prediction_dir, variant, split).metadata
                continue
            view = load_cached_hlt_view(config.cache_dir, split)
            block = _prediction_block(
                model,
                view,
                variant=variant,
                batch_size=int(config.batch_size),
                num_workers=int(config.num_workers),
                device=device,
                max_jets=_split_size(config, split),
                label_names=label_names,
                label_filter=label_filter,
                selection_seed=int(config.control_seed) + 1009 * (list(STACK_SPLITS).index(split) + 1),
            )
            block.metadata.update(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_epoch": payload.get("epoch"),
                    "checkpoint_best_model_val_accuracy": (
                        (payload.get("metrics") or {}).get("model_val", {}) or {}
                    ).get("accuracy"),
                    "max_jets": _split_size(config, split),
                }
            )
            reports[variant][split] = save_prediction_block(
                block,
                prediction_dir,
                overwrite=config.overwrite_predictions,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True, help="Directory containing <variant>/best_model_val.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", nargs="+", choices=ARCHITECTURE_VIEW_VARIANTS, required=True)
    parser.add_argument("--splits", nargs="+", choices=STACK_SPLITS, default=list(STACK_SPLITS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stack-train-size", type=int, default=500000)
    parser.add_argument("--stack-val-size", type=int, default=150000)
    parser.add_argument("--final-test-size", type=int, default=500000)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--no-skip-existing-predictions", action="store_true")
    parser.add_argument("--feature-modes", nargs="+", choices=FEATURE_MODES, default=list(FEATURE_MODES))
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--control-seed", type=int, default=12345)
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = tuple(normalize_architecture_view_variant(variant) for variant in args.variants)
    config = ArchitectureViewFusionRunConfig(
        cache_dir=args.cache_dir,
        checkpoint_root=args.checkpoint_root,
        output_dir=args.output_dir,
        variants=variants,
        splits=tuple(args.splits),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=args.device,
        stack_train_size=int(args.stack_train_size),
        stack_val_size=int(args.stack_val_size),
        final_test_size=int(args.final_test_size),
        overwrite_predictions=bool(args.overwrite_predictions),
        skip_existing_predictions=not bool(args.no_skip_existing_predictions),
        confirm_final_test=bool(args.confirm_final_test),
        feature_modes=tuple(args.feature_modes),
        c_grid=tuple(float(value) for value in args.c_grid),
        max_iter=int(args.max_iter),
        run_controls=not bool(args.skip_controls),
        control_seed=int(args.control_seed),
    )
    if "final_test" in config.splits and not config.confirm_final_test:
        raise SystemExit("Refusing to evaluate final_test without --confirm-final-test")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "architecture_view_part_fusion_config.json",
        {
            "config": asdict(config),
            "groups": {
                "architecture_view_part4_baseline": [
                    "av_baseline_recheck",
                    "av_pn_only",
                    "av_pfn_only",
                    "av_pcnn_only",
                ],
                "architecture_view_part4_context": [
                    "av_context_mlp_control",
                    "av_pn_only",
                    "av_pfn_only",
                    "av_pcnn_only",
                ],
                "architecture_view_all_available": list(variants),
            },
            "leakage_rules": {
                "architecture_view_training": "Each variant trains on model_train and selects best checkpoint on model_val.",
                "stacker_fit_split": "stack_train",
                "stacker_selection_split": "stack_val",
                "final_test_evaluated_after_selection": True,
                "inputs": "cached fixed-HLT tokens only",
            },
        },
    )
    prediction_report = collect_architecture_view_predictions(config)
    save_json(output_dir / "prediction_collection_report.json", prediction_report)
    if tuple(config.splits) != tuple(STACK_SPLITS):
        print("Prediction collection complete. Fusion skipped because not all stack splits were requested.")
        return 0
    available = set(variants)
    groups = {
        "architecture_view_part4_baseline": [
            item
            for item in ("av_baseline_recheck", "av_pn_only", "av_pfn_only", "av_pcnn_only")
            if item in available
        ],
        "architecture_view_part4_context": [
            item
            for item in ("av_context_mlp_control", "av_pn_only", "av_pfn_only", "av_pcnn_only")
            if item in available
        ],
        "architecture_view_all_available": list(variants),
    }
    groups = {name: members for name, members in groups.items() if len(members) >= 2}
    fusion_config = IndependentFusionConfig(
        prediction_dir=str(output_dir / "predictions"),
        output_dir=str(output_dir / "fusion"),
        model_names=list(variants),
        groups=groups,
        feature_modes=list(config.feature_modes or FEATURE_MODES),
        c_grid=list(config.c_grid or DEFAULT_C_GRID),
        max_iter=int(config.max_iter),
        confirm_final_test=bool(config.confirm_final_test),
        run_controls=bool(config.run_controls),
        control_seed=int(config.control_seed),
        singleton_models=list(variants),
    )
    report = run_independent_fusion(fusion_config)
    print(f"Saved Architecture-View fusion report: {output_dir / 'fusion' / 'fusion_report.json'}")
    for group_name, group_report in report["group_fusion_metrics"].items():
        for mode, mode_report in group_report["feature_modes"].items():
            val = mode_report["metrics"]["stack_val"]
            final = mode_report["metrics"]["final_test"]
            selected_c = mode_report["selection"]["selected_C"]
            print(
                f"  {group_name:<36s} {mode:<12s} C={selected_c:g} "
                f"stack_val_acc={val['accuracy']:.6f} final_test_acc={final['accuracy']:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

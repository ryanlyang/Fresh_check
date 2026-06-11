#!/usr/bin/env python3
"""Evaluate the offline-only teacher as an upper-reference on held-out splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import (  # noqa: E402
    PredictionBlock,
    STACK_SPLITS,
    load_hlt_model_from_checkpoint,
    save_prediction_block,
    softmax_np,
)
from jetclass_fresh.heterogeneous_hlt import balanced_limit_jet_view  # noqa: E402
from jetclass_fresh.hlt_baseline import (  # noqa: E402
    ParticleViewTorchDataset,
    make_data_loader,
    require_torch,
    resolve_device,
    save_json,
)
from jetclass_fresh.independent_fusion import metrics_from_logits  # noqa: E402
from jetclass_fresh.jetclass_data import LABEL_NAMES, load_offline_view, load_split_manifest, manifest_hash  # noqa: E402


@dataclass
class OfflineTeacherEvalConfig:
    manifest_path: str
    checkpoint: str
    output_dir: str
    data_dir: str | None = None
    splits: List[str] | None = None
    stack_train_size: int | None = None
    stack_val_size: int | None = 50_000
    final_test_size: int | None = 300_000
    batch_size: int = 128
    num_workers: int = 4
    device: str = "auto"
    control_seed: int = 12345
    confirm_final_test: bool = False
    overwrite_predictions: bool = False
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--splits", nargs="+", choices=STACK_SPLITS, default=["stack_val", "final_test"])
    parser.add_argument("--stack-train-size", type=int, default=None)
    parser.add_argument("--stack-val-size", type=int, default=50_000)
    parser.add_argument("--final-test-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=12345)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    return parser.parse_args()


def split_size(config: OfflineTeacherEvalConfig, split: str) -> int | None:
    if split == "stack_train":
        return config.stack_train_size
    if split == "stack_val":
        return config.stack_val_size
    if split == "final_test":
        return config.final_test_size
    return None


def selection_seed(config: OfflineTeacherEvalConfig, split: str) -> int:
    return int(config.control_seed) + 1009 * (list(STACK_SPLITS).index(split) + 1)


def evaluate_split(
    model,
    offline_view,
    *,
    split: str,
    config: OfflineTeacherEvalConfig,
    device,
) -> tuple[PredictionBlock, Dict[str, Any]]:
    torch = require_torch()
    limited_view, selection = balanced_limit_jet_view(
        offline_view,
        split_size(config, split),
        seed=selection_seed(config, split),
    )
    dataset = ParticleViewTorchDataset(limited_view, expected_view="offline")
    loader = make_data_loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=selection_seed(config, split),
        source_view="offline",
    )
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            logits = model(batch["points"], batch["features"], batch["lorentz_vectors"], batch["mask"])
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
    logits_np = np.concatenate(logits_rows, axis=0)
    labels_np = np.concatenate(labels_rows, axis=0)
    block = PredictionBlock(
        model_name="offline_teacher",
        split=split,
        logits=logits_np,
        probs=softmax_np(logits_np),
        labels=labels_np,
        jet_ids=list(limited_view.jet_ids),
        metadata={
            "model_kind": "offline_teacher_reference",
            "reference_role": "offline_upper_reference_only",
            "allowed_inputs": "offline_constituents_only",
            "not_allowed_for_hlt_deployable_fusion_features": True,
            "subset_selection": selection,
        },
    )
    return block, metrics_from_logits(logits_np, labels_np)


def evaluate_offline_teacher(config: OfflineTeacherEvalConfig) -> Dict[str, Any]:
    if config.splits is None:
        config.splits = ["stack_val", "final_test"]
    if "final_test" in config.splits and not bool(config.confirm_final_test):
        raise ValueError("Refusing to evaluate final_test without confirm_final_test=True")

    torch = require_torch()
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_split_manifest(config.manifest_path)
    manifest_sha = manifest_hash(manifest)
    model, payload = load_hlt_model_from_checkpoint(config.checkpoint, device=device)

    split_reports: Dict[str, Any] = {}
    for split in config.splits:
        offline_view = load_offline_view(
            manifest,
            split,
            data_dir=config.data_dir,
            verify_label_branches=config.verify_label_branches,
            read_chunk_size=config.read_chunk_size,
        )
        block, metrics = evaluate_split(
            model,
            offline_view,
            split=split,
            config=config,
            device=device,
        )
        metadata = save_prediction_block(
            block,
            prediction_dir,
            overwrite=bool(config.overwrite_predictions),
        )
        split_reports[split] = {
            "metrics": metrics,
            "prediction_metadata": metadata,
            "subset_selection": block.metadata["subset_selection"],
        }

    report = {
        "purpose": "offline_teacher_upper_reference_evaluation",
        "config": asdict(config),
        "manifest_hash": manifest_sha,
        "checkpoint": str(config.checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_experiment_step": payload.get("experiment_step"),
        "label_names": list(LABEL_NAMES),
        "reference_role": "offline_upper_reference_only",
        "not_allowed_for_hlt_deployable_fusion_features": True,
        "splits": split_reports,
    }
    save_json(output_dir / "offline_teacher_reference_report.json", report)
    return report


def main() -> int:
    args = parse_args()
    config = OfflineTeacherEvalConfig(
        manifest_path=args.manifest_path,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        splits=list(args.splits),
        stack_train_size=args.stack_train_size,
        stack_val_size=args.stack_val_size,
        final_test_size=args.final_test_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        control_seed=args.control_seed,
        confirm_final_test=bool(args.confirm_final_test),
        overwrite_predictions=bool(args.overwrite_predictions),
        verify_label_branches=bool(args.verify_label_branches),
        read_chunk_size=args.read_chunk_size,
    )
    report = evaluate_offline_teacher(config)
    print("offline_teacher_reference:")
    print(f"  report: {Path(config.output_dir) / 'offline_teacher_reference_report.json'}")
    for split, row in report["splits"].items():
        metrics = row["metrics"]
        print(
            f"  {split:<11s} "
            f"acc={metrics['accuracy']:.6f} "
            f"ce={metrics['cross_entropy']:.6f} "
            f"auc={metrics['macro_ovr_auc']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

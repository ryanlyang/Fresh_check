#!/usr/bin/env python3
"""Train and evaluate a fresh binary offline-only ParT baseline.

This is an offline upper-reference for the Hbb-vs-QCD set-matching experiment.
It trains only on model_train, selects only on model_val, then evaluates the
selected checkpoint on stack_val/final_test with the same split manifest and
row limits used by the set-matching binary run.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import (  # noqa: E402
    ParticleViewTorchDataset,
    build_particle_transformer_classifier,
    make_data_loader,
    require_torch,
    resolve_device,
    run_epoch,
    save_json,
    set_training_seed,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES, JetView, load_offline_view, load_split_manifest, manifest_hash  # noqa: E402
from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions  # noqa: E402


EXPERIMENT_STEP = "set_matching_hbb_qcd_binary_fresh_offline_teacher"


def _label_names_to_indices(values: list[str]) -> tuple[int, ...]:
    by_name = {name: index for index, name in enumerate(LABEL_NAMES)}
    output: list[int] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            output.append(int(text))
        elif text in by_name:
            output.append(by_name[text])
        else:
            raise ValueError(f"Unknown JetClass label {text!r}; expected one of {list(LABEL_NAMES)}")
    if len(set(output)) != len(output):
        raise ValueError(f"Duplicate labels in filter: {values!r}")
    return tuple(output)


@dataclass
class BinaryOfflineTeacherConfig:
    output_dir: str
    manifest_path: str
    data_dir: str | None = None
    label_filter: tuple[int, ...] = (0, 1)
    label_names: tuple[str, ...] = ("QCD", "Hbb")
    train_split: str = "model_train"
    val_split: str = "model_val"
    stack_val_split: str = "stack_val"
    final_test_split: str = "final_test"
    seed: int = 2405
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 30
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 2
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    max_train_jets: int | None = 100_000
    max_val_jets: int | None = 30_000
    max_stack_val_jets: int | None = 10_000
    max_final_test_jets: int | None = 100_000
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_eval_batches: int | None = None
    model_size: str = "base"
    compile_model: bool = False
    confirm_final_test: bool = False
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000

    def __post_init__(self) -> None:
        self.label_filter = tuple(int(label) for label in self.label_filter)
        self.label_names = tuple(str(name) for name in self.label_names)
        if len(self.label_filter) != 2:
            raise ValueError("Binary offline teacher requires exactly two labels")
        if len(self.label_names) != 2:
            raise ValueError("Binary offline teacher requires exactly two label names")
        if len(set(self.label_filter)) != len(self.label_filter):
            raise ValueError(f"Duplicate label_filter entries: {self.label_filter}")
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Training may use only model_train and model_val")
        if self.stack_val_split != "stack_val" or self.final_test_split != "final_test":
            raise ValueError("Evaluation splits must be stack_val/final_test")
        if self.confirm_final_test is not True:
            raise ValueError("Set --confirm-final-test to evaluate final_test")


def _filter_and_remap_view(view: JetView, config: BinaryOfflineTeacherConfig, *, max_jets: int | None) -> JetView:
    labels = np.asarray(view.labels, dtype=np.int64)
    keep = np.isin(labels, np.asarray(config.label_filter, dtype=np.int64))
    if not np.all(keep):
        labels = labels[keep]
        tokens = np.asarray(view.tokens)[keep]
        mask = np.asarray(view.mask)[keep]
        jet_ids = [jet_id for jet_id, should_keep in zip(view.jet_ids, keep) if bool(should_keep)]
    else:
        tokens = np.asarray(view.tokens)
        mask = np.asarray(view.mask)
        jet_ids = list(view.jet_ids)

    label_to_binary = {int(source_label): index for index, source_label in enumerate(config.label_filter)}
    remapped = np.asarray([label_to_binary[int(label)] for label in labels], dtype=np.int64)
    if max_jets is not None:
        limit = min(int(max_jets), int(remapped.shape[0]))
        tokens = tokens[:limit]
        mask = mask[:limit]
        remapped = remapped[:limit]
        jet_ids = jet_ids[:limit]

    metadata = dict(view.metadata)
    metadata.update(
        {
            "binary_offline_teacher_label_filter": list(config.label_filter),
            "binary_offline_teacher_label_names": list(config.label_names),
            "binary_label_remap": {str(source): int(index) for index, source in enumerate(config.label_filter)},
            "n_jets_after_binary_filter": int(remapped.shape[0]),
            "max_jets_limit": None if max_jets is None else int(max_jets),
            "view": "offline",
        }
    )
    return JetView(
        tokens=np.asarray(tokens, dtype=np.float32),
        mask=np.asarray(mask, dtype=bool),
        labels=remapped,
        jet_ids=jet_ids,
        split=view.split,
        metadata=metadata,
    )


def _load_binary_offline_view(manifest, split: str, config: BinaryOfflineTeacherConfig, *, max_jets: int | None) -> JetView:
    view = load_offline_view(
        manifest,
        split,
        data_dir=config.data_dir,
        verify_label_branches=config.verify_label_branches,
        read_chunk_size=config.read_chunk_size,
    )
    return _filter_and_remap_view(view, config, max_jets=max_jets)


def _checkpoint_payload(model, optimizer, *, epoch: int, config: BinaryOfflineTeacherConfig, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": dict(metrics),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "model_config": getattr(model, "config", {}),
        "experiment_step": EXPERIMENT_STEP,
    }


def _evaluate_model(model, view: JetView, config: BinaryOfflineTeacherConfig, *, split: str, device) -> dict[str, Any]:
    torch = require_torch()
    dataset = ParticleViewTorchDataset(view, expected_view="offline")
    loader = make_data_loader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1009,
        source_view="offline",
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    logits_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    loss_sum = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if config.max_eval_batches is not None and batch_index >= int(config.max_eval_batches):
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=False):
                logits = model(batch["points"], batch["features"], batch["lorentz_vectors"], batch["mask"])
                loss = criterion(logits, batch["labels"])
            loss_sum += float(loss.detach().item())
            logits_np = logits.detach().cpu().numpy().astype(np.float32)
            logits_rows.append(logits_np)
            pred_rows.append(np.argmax(logits_np, axis=1).astype(np.int64))
            label_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))

    logits = np.concatenate(logits_rows, axis=0)
    preds = np.concatenate(pred_rows, axis=0)
    labels = np.concatenate(label_rows, axis=0)
    metrics = classification_metrics_from_predictions(
        preds=preds,
        labels=labels,
        loss_sum=loss_sum,
        logits=logits,
        label_names=tuple(config.label_names),
    )
    return {
        "split": split,
        "metrics": metrics,
        "n_jets": int(labels.shape[0]),
        "label_counts": {config.label_names[index]: int(np.sum(labels == index)) for index in range(2)},
    }


def _write_eval_diagnostics(output_dir: Path, report: Mapping[str, Any]) -> None:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for split, payload in report["evaluations"].items():
        metrics = payload["metrics"]
        binary = metrics.get("binary_metrics", {})
        rows.append(
            {
                "split": split,
                "accuracy": metrics.get("accuracy"),
                "loss": metrics.get("loss"),
                "auc": binary.get("auc"),
                "fpr_at_signal_eff_0p30": binary.get("fpr_at_signal_eff_0p30"),
                "fpr_at_signal_eff_0p50": binary.get("fpr_at_signal_eff_0p50"),
                "background_rejection_at_signal_eff_0p30": binary.get("background_rejection_at_signal_eff_0p30"),
                "background_rejection_at_signal_eff_0p50": binary.get("background_rejection_at_signal_eff_0p50"),
                "n_jets": payload.get("n_jets"),
            }
        )
        for class_row in metrics.get("per_class_accuracy", []):
            out = {"split": split}
            out.update(class_row)
            per_class_rows.append(out)
    with (diagnostics_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "accuracy",
                "loss",
                "auc",
                "fpr_at_signal_eff_0p30",
                "fpr_at_signal_eff_0p50",
                "background_rejection_at_signal_eff_0p30",
                "background_rejection_at_signal_eff_0p50",
                "n_jets",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (diagnostics_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_index", "class_name", "support", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(per_class_rows)


def train_and_evaluate(config: BinaryOfflineTeacherConfig) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_split_manifest(config.manifest_path)
    manifest_sha = manifest_hash(manifest)
    train_view = _load_binary_offline_view(manifest, config.train_split, config, max_jets=config.max_train_jets)
    val_view = _load_binary_offline_view(manifest, config.val_split, config, max_jets=config.max_val_jets)

    train_dataset = ParticleViewTorchDataset(train_view, expected_view="offline")
    val_dataset = ParticleViewTorchDataset(val_view, expected_view="offline")
    train_loader = make_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        source_view="offline",
    )
    val_loader = make_data_loader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
        source_view="offline",
    )

    model = build_particle_transformer_classifier(num_classes=2, model_size=config.model_size).to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))

    metadata = {
        "experiment_step": EXPERIMENT_STEP,
        "config": asdict(config),
        "manifest_hash": manifest_sha,
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "train_n_jets": len(train_dataset),
        "val_n_jets": len(val_dataset),
        "train_label_counts": {config.label_names[index]: int(np.sum(train_view.labels == index)) for index in range(2)},
        "val_label_counts": {config.label_names[index]: int(np.sum(val_view.labels == index)) for index in range(2)},
        "reference_role": "fresh_binary_offline_upper_reference",
        "leakage_rule": "Uses offline constituents intentionally as an offline-only upper reference.",
    }
    save_json(output_dir / "config.json", metadata)

    curves: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
        )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})

        improved = (
            val_metrics["accuracy"] > best_val_accuracy
            or (np.isclose(val_metrics["accuracy"], best_val_accuracy) and val_metrics["loss"] < best_val_loss)
        )
        torch.save(_checkpoint_payload(model, optimizer, epoch=epoch, config=config, metrics=row), output_dir / "last.pt")
        if improved:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_val_loss = float(val_metrics["loss"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(model, optimizer, epoch=epoch, config=config, metrics=row),
                output_dir / "best_model_val.pt",
            )
        else:
            epochs_without_improvement += 1
        if config.early_stop_patience >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    best_payload = torch.load(output_dir / "best_model_val.pt", map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])
    stack_val_view = _load_binary_offline_view(
        manifest,
        config.stack_val_split,
        config,
        max_jets=config.max_stack_val_jets,
    )
    final_test_view = _load_binary_offline_view(
        manifest,
        config.final_test_split,
        config,
        max_jets=config.max_final_test_jets,
    )
    eval_reports = {
        "stack_val": _evaluate_model(model, stack_val_view, config, split="stack_val", device=device),
        "final_test": _evaluate_model(model, final_test_view, config, split="final_test", device=device),
    }

    report = {
        "experiment_step": EXPERIMENT_STEP,
        "reference_role": "fresh_binary_offline_upper_reference",
        "best_epoch": int(best_epoch),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "final_test_evaluated": True,
        "evaluations": eval_reports,
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "manifest_hash": manifest_sha,
    }
    save_json(output_dir / "run_report.json", report)
    save_json(output_dir / "model_val_report.json", report)
    _write_eval_diagnostics(output_dir, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--label-filter-names", nargs="+", default=["QCD", "Hbb"])
    parser.add_argument("--label-names", nargs="+", default=["QCD", "Hbb"])
    parser.add_argument("--seed", type=int, default=2405)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-train-jets", type=int, default=100_000)
    parser.add_argument("--max-val-jets", type=int, default=30_000)
    parser.add_argument("--max-stack-val-jets", type=int, default=10_000)
    parser.add_argument("--max-final-test-jets", type=int, default=100_000)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--model-size", choices=["base", "tiny"], default="base")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BinaryOfflineTeacherConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        data_dir=args.data_dir,
        label_filter=_label_names_to_indices(list(args.label_filter_names)),
        label_names=tuple(args.label_names),
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
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_stack_val_jets=args.max_stack_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_eval_batches=args.max_eval_batches,
        model_size=args.model_size,
        compile_model=args.compile_model,
        confirm_final_test=bool(args.confirm_final_test),
        verify_label_branches=bool(args.verify_label_branches),
        read_chunk_size=args.read_chunk_size,
    )
    report = train_and_evaluate(config)
    print("fresh_binary_offline_teacher_complete:")
    print(f"  output_dir: {config.output_dir}")
    print(f"  best_epoch: {report['best_epoch']}")
    print(f"  model_val_acc: {report['best_model_val_accuracy']:.6f}")
    for split, row in report["evaluations"].items():
        metrics = row["metrics"]
        binary = metrics.get("binary_metrics", {})
        print(
            f"  {split}: "
            f"acc={metrics['accuracy']:.6f} "
            f"auc={binary.get('auc')} "
            f"fpr30={binary.get('fpr_at_signal_eff_0p30')} "
            f"fpr50={binary.get('fpr_at_signal_eff_0p50')}"
        )
    print(f"  report: {Path(config.output_dir) / 'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

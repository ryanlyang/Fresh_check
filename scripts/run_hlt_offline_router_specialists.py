#!/usr/bin/env python3
"""Train HLT-only specialists routed by HLT/offline-on-HLT agreement."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import classification_metrics_from_logits, load_hlt_model_from_checkpoint  # noqa: E402
from jetclass_fresh.hlt_baseline import (  # noqa: E402
    HLTBaselineTrainConfig,
    JetViewTorchDataset,
    make_data_loader,
    require_torch,
    resolve_device,
    save_json,
    train_hlt_baseline,
)
from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from jetclass_fresh.jetclass_data import LABEL_NAMES, JetView  # noqa: E402
from teacher_logit_reco.teachers import TEACHER_ARCHITECTURES, load_frozen_teacher  # noqa: E402
from teacher_logit_reco.train_global_transformer import source_metadata  # noqa: E402
from teacher_logit_reco.views import slice_jet_view  # noqa: E402


EXPERIMENT_STEP = "hlt_offline_agreement_router_specialists"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--hlt-checkpoint", required=True)
    parser.add_argument("--hlt-architecture", choices=TEACHER_ARCHITECTURES, default="part")
    parser.add_argument("--offline-checkpoint", required=True)
    parser.add_argument("--offline-architecture", choices=TEACHER_ARCHITECTURES, default="part")
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--test-split", default="final_test")
    parser.add_argument("--max-train-jets", type=int, default=150_000)
    parser.add_argument("--max-val-jets", type=int, default=50_000)
    parser.add_argument("--max-test-jets", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1777)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--model-size", choices=("tiny", "base"), default="base")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--weight-threshold", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def class_counts(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)
    return {
        LABEL_NAMES[index]: int(np.sum(labels == index))
        for index in range(len(LABEL_NAMES))
    }


def subset_view(view: JetView, indices: np.ndarray, *, route_name: str) -> JetView:
    indices = np.asarray(indices, dtype=np.int64)
    metadata = dict(view.metadata)
    metadata.update(
        {
            "router_subset": route_name,
            "source_n_jets_before_router_subset": int(len(view.labels)),
            "n_jets_after_router_subset": int(indices.shape[0]),
        }
    )
    return JetView(
        tokens=view.tokens[indices].copy(),
        mask=view.mask[indices].copy(),
        labels=view.labels[indices].copy(),
        jet_ids=[view.jet_ids[int(index)] for index in indices],
        split=view.split,
        metadata=metadata,
    )


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp, axis=1, keepdims=True), 1.0e-300, None)


def topk_overlap(left_probs: np.ndarray, right_probs: np.ndarray, *, k: int = 3) -> np.ndarray:
    left = np.argpartition(left_probs, kth=-k, axis=1)[:, -k:]
    right = np.argpartition(right_probs, kth=-k, axis=1)[:, -k:]
    return np.asarray([bool(set(a.tolist()) & set(b.tolist())) for a, b in zip(left, right)], dtype=bool)


def route_summary(labels: np.ndarray, hlt_logits: np.ndarray, offline_hlt_logits: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    hlt_probs = softmax_np(hlt_logits)
    offline_probs = softmax_np(offline_hlt_logits)
    hlt_pred = np.argmax(hlt_probs, axis=1)
    offline_pred = np.argmax(offline_probs, axis=1)
    agree = hlt_pred == offline_pred
    hlt_correct = hlt_pred == labels
    offline_correct = offline_pred == labels
    return {
        "n_jets": int(labels.shape[0]),
        "agreement_fraction": float(np.mean(agree)) if len(labels) else 0.0,
        "top3_overlap_fraction": float(np.mean(topk_overlap(hlt_probs, offline_probs, k=3))) if len(labels) else 0.0,
        "hlt_probe_accuracy": float(np.mean(hlt_correct)) if len(labels) else 0.0,
        "offline_on_hlt_probe_accuracy": float(np.mean(offline_correct)) if len(labels) else 0.0,
        "oracle_probe_a_or_b_accuracy": float(np.mean(hlt_correct | offline_correct)) if len(labels) else 0.0,
        "a_wrong_b_correct_fraction": float(np.mean((~hlt_correct) & offline_correct)) if len(labels) else 0.0,
        "routes": {
            "agreement": {
                "n_jets": int(np.sum(agree)),
                "fraction": float(np.mean(agree)) if len(labels) else 0.0,
                "class_counts": class_counts(labels[agree]),
                "hlt_probe_accuracy": float(np.mean(hlt_correct[agree])) if np.any(agree) else float("nan"),
                "offline_on_hlt_probe_accuracy": float(np.mean(offline_correct[agree])) if np.any(agree) else float("nan"),
            },
            "disagreement": {
                "n_jets": int(np.sum(~agree)),
                "fraction": float(np.mean(~agree)) if len(labels) else 0.0,
                "class_counts": class_counts(labels[~agree]),
                "hlt_probe_accuracy": float(np.mean(hlt_correct[~agree])) if np.any(~agree) else float("nan"),
                "offline_on_hlt_probe_accuracy": float(np.mean(offline_correct[~agree])) if np.any(~agree) else float("nan"),
            },
        },
    }


def load_hlt_view(args: argparse.Namespace, split: str, max_jets: int | None) -> JetView:
    return slice_jet_view(load_cached_hlt_view(args.hlt_cache_dir, split, verify_hash=True), max_jets)


def predict_teacher_on_view(teacher, view: JetView, *, batch_size: int, num_workers: int, seed: int) -> np.ndarray:
    torch = require_torch()
    dataset = JetViewTorchDataset(view)
    loader = make_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    rows: list[np.ndarray] = []
    teacher.model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {
                key: value.to(teacher.device, non_blocking=True)
                for key, value in batch.items()
                if key != "labels"
            }
            rows.append(teacher.forward_inputs(inputs).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, len(LABEL_NAMES)), dtype=np.float32)


def compute_router(
    *,
    args: argparse.Namespace,
    view: JetView,
    split: str,
    hlt_probe,
    offline_probe,
    output_dir: Path,
) -> dict[str, Any]:
    hlt_logits = predict_teacher_on_view(
        hlt_probe,
        view,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 10,
    )
    offline_hlt_logits = predict_teacher_on_view(
        offline_probe,
        view,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 11,
    )
    labels = np.asarray(view.labels, dtype=np.int64)
    hlt_pred = np.argmax(hlt_logits, axis=1)
    offline_pred = np.argmax(offline_hlt_logits, axis=1)
    agree = hlt_pred == offline_pred
    router_dir = output_dir / "router"
    router_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        router_dir / f"{split}_router_logits.npz",
        labels=labels,
        agree=agree.astype(bool),
        hlt_logits=hlt_logits.astype(np.float32),
        offline_on_hlt_logits=offline_hlt_logits.astype(np.float32),
    )
    summary = route_summary(labels, hlt_logits, offline_hlt_logits)
    save_json(router_dir / f"{split}_router_report.json", summary)
    return {
        "labels": labels,
        "hlt_logits": hlt_logits,
        "offline_on_hlt_logits": offline_hlt_logits,
        "agree": agree,
        "summary": summary,
    }


def train_specialist(
    *,
    args: argparse.Namespace,
    route_name: str,
    output_dir: Path,
    train_view: JetView,
    val_view: JetView,
    seed_offset: int,
) -> dict[str, Any]:
    if len(train_view.labels) == 0:
        raise ValueError(f"Cannot train {route_name} specialist on an empty train subset")
    if len(val_view.labels) == 0:
        raise ValueError(f"Cannot train {route_name} specialist on an empty val subset")
    config = HLTBaselineTrainConfig(
        output_dir=str(output_dir / "specialists" / route_name),
        cache_dir=args.hlt_cache_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed + seed_offset,
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
        model_size=args.model_size,
    )
    report = train_hlt_baseline(config, train_view=train_view, val_view=val_view)
    report["route_name"] = route_name
    report["train_subset_class_counts"] = class_counts(train_view.labels)
    report["val_subset_class_counts"] = class_counts(val_view.labels)
    save_json(Path(config.output_dir) / "router_specialist_report.json", report)
    return report


def predict_hlt_checkpoint(path: Path, view: JetView, *, args: argparse.Namespace, device) -> np.ndarray:
    torch = require_torch()
    model, _ = load_hlt_model_from_checkpoint(path, device=device)
    dataset = JetViewTorchDataset(view)
    loader = make_data_loader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 200,
    )
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            features = batch["features"].to(device, non_blocking=True)
            lorentz_vectors = batch["lorentz_vectors"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            rows.append(model(points, features, lorentz_vectors, mask).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, len(LABEL_NAMES)), dtype=np.float32)


def evaluate_routed_specialists(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    test_view: JetView,
    test_router: Mapping[str, Any],
    device,
) -> dict[str, Any]:
    agree = np.asarray(test_router["agree"], dtype=bool)
    agree_indices = np.flatnonzero(agree)
    disagree_indices = np.flatnonzero(~agree)
    labels = np.asarray(test_view.labels, dtype=np.int64)
    agree_view = subset_view(test_view, agree_indices, route_name="agreement")
    disagree_view = subset_view(test_view, disagree_indices, route_name="disagreement")

    agree_logits = predict_hlt_checkpoint(
        output_dir / "specialists" / "agreement" / "best_model_val.pt",
        agree_view,
        args=args,
        device=device,
    )
    disagree_logits = predict_hlt_checkpoint(
        output_dir / "specialists" / "disagreement" / "best_model_val.pt",
        disagree_view,
        args=args,
        device=device,
    )

    routed_logits = np.zeros((len(labels), len(LABEL_NAMES)), dtype=np.float32)
    routed_logits[agree_indices] = agree_logits
    routed_logits[disagree_indices] = disagree_logits
    hlt_probe_logits = np.asarray(test_router["hlt_logits"], dtype=np.float32)
    offline_hlt_logits = np.asarray(test_router["offline_on_hlt_logits"], dtype=np.float32)

    report = {
        "routed_specialists": classification_metrics_from_logits(routed_logits, labels),
        "hlt_probe": classification_metrics_from_logits(hlt_probe_logits, labels),
        "offline_on_hlt_probe": classification_metrics_from_logits(offline_hlt_logits, labels),
        "agreement_specialist_on_agreement_route": classification_metrics_from_logits(
            agree_logits,
            labels[agree_indices],
        )
        if len(agree_indices)
        else {"accuracy": float("nan"), "cross_entropy": float("nan"), "n_jets": 0},
        "disagreement_specialist_on_disagreement_route": classification_metrics_from_logits(
            disagree_logits,
            labels[disagree_indices],
        )
        if len(disagree_indices)
        else {"accuracy": float("nan"), "cross_entropy": float("nan"), "n_jets": 0},
        "route_counts": {
            "agreement": int(len(agree_indices)),
            "disagreement": int(len(disagree_indices)),
        },
    }
    report["delta_vs_hlt_probe_accuracy"] = float(
        report["routed_specialists"]["accuracy"] - report["hlt_probe"]["accuracy"]
    )
    np.savez_compressed(
        output_dir / "routed_specialist_final_test_logits.npz",
        labels=labels,
        agree=agree,
        routed_logits=routed_logits,
        hlt_probe_logits=hlt_probe_logits,
        offline_on_hlt_logits=offline_hlt_logits,
    )
    save_json(output_dir / "evaluation_report.json", report)
    return report


def main() -> int:
    args = parse_args()
    if args.train_split != "model_train" or args.val_split != "model_val":
        raise ValueError("Specialists must train on model_train and select on model_val")
    for name in ("max_train_jets", "max_val_jets", "max_test_jets", "epochs"):
        value = getattr(args, name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive when provided")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    hlt_probe = load_frozen_teacher(
        args.hlt_checkpoint,
        architecture=args.hlt_architecture,
        device=str(device),
        max_constits=args.max_constits,
        weight_threshold=args.weight_threshold,
    )
    offline_probe = load_frozen_teacher(
        args.offline_checkpoint,
        architecture=args.offline_architecture,
        device=str(device),
        max_constits=args.max_constits,
        weight_threshold=args.weight_threshold,
    )

    train_view = load_hlt_view(args, args.train_split, args.max_train_jets)
    val_view = load_hlt_view(args, args.val_split, args.max_val_jets)
    test_view = load_hlt_view(args, args.test_split, args.max_test_jets)

    run_config = {
        "experiment_step": EXPERIMENT_STEP,
        "args": vars(args),
        "label_names": list(LABEL_NAMES),
        "hlt_probe": dict(hlt_probe.metadata),
        "offline_probe": dict(offline_probe.metadata),
        "source": source_metadata(),
        "interpretation": {
            "router": "HLT probe and offline-trained probe evaluated on the same HLT jet; argmax agreement selects route.",
            "classification": "Final prediction comes from HLT-only specialists trained on the routed model_train subsets.",
        },
    }
    save_json(output_dir / "experiment_config.json", run_config)

    train_router = compute_router(
        args=args,
        view=train_view,
        split=args.train_split,
        hlt_probe=hlt_probe,
        offline_probe=offline_probe,
        output_dir=output_dir,
    )
    val_router = compute_router(
        args=args,
        view=val_view,
        split=args.val_split,
        hlt_probe=hlt_probe,
        offline_probe=offline_probe,
        output_dir=output_dir,
    )
    test_router = compute_router(
        args=args,
        view=test_view,
        split=args.test_split,
        hlt_probe=hlt_probe,
        offline_probe=offline_probe,
        output_dir=output_dir,
    )

    train_agree = np.flatnonzero(train_router["agree"])
    train_disagree = np.flatnonzero(~np.asarray(train_router["agree"], dtype=bool))
    val_agree = np.flatnonzero(val_router["agree"])
    val_disagree = np.flatnonzero(~np.asarray(val_router["agree"], dtype=bool))

    agreement_report = train_specialist(
        args=args,
        route_name="agreement",
        output_dir=output_dir,
        train_view=subset_view(train_view, train_agree, route_name="agreement"),
        val_view=subset_view(val_view, val_agree, route_name="agreement"),
        seed_offset=1000,
    )
    disagreement_report = train_specialist(
        args=args,
        route_name="disagreement",
        output_dir=output_dir,
        train_view=subset_view(train_view, train_disagree, route_name="disagreement"),
        val_view=subset_view(val_view, val_disagree, route_name="disagreement"),
        seed_offset=2000,
    )

    evaluation = evaluate_routed_specialists(
        args=args,
        output_dir=output_dir,
        test_view=test_view,
        test_router=test_router,
        device=device,
    )
    report = {
        "ok": True,
        "experiment_step": EXPERIMENT_STEP,
        "output_dir": str(output_dir),
        "router_reports": {
            args.train_split: train_router["summary"],
            args.val_split: val_router["summary"],
            args.test_split: test_router["summary"],
        },
        "specialists": {
            "agreement": agreement_report,
            "disagreement": disagreement_report,
        },
        "evaluation": evaluation,
        "source": source_metadata(),
    }
    save_json(output_dir / "run_report.json", report)

    print("hlt_offline_router_specialists_complete:")
    print(f"  output_dir: {output_dir}")
    print(f"  train_agree_fraction: {train_router['summary']['agreement_fraction']:.6f}")
    print(f"  val_agree_fraction: {val_router['summary']['agreement_fraction']:.6f}")
    print(f"  test_agree_fraction: {test_router['summary']['agreement_fraction']:.6f}")
    print(f"  agreement_specialist_val_acc: {agreement_report['best_model_val_accuracy']:.6f}")
    print(f"  disagreement_specialist_val_acc: {disagreement_report['best_model_val_accuracy']:.6f}")
    print(f"  hlt_probe_test_acc: {evaluation['hlt_probe']['accuracy']:.6f}")
    print(f"  routed_specialists_test_acc: {evaluation['routed_specialists']['accuracy']:.6f}")
    print(f"  delta_vs_hlt_probe_acc: {evaluation['delta_vs_hlt_probe_accuracy']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

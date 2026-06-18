#!/usr/bin/env python3
"""Diagnose complementarity between HLT and offline-trained taggers on HLT."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json  # noqa: E402
from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402
from teacher_logit_reco.teachers import TEACHER_ARCHITECTURES, load_frozen_teacher  # noqa: E402
from teacher_logit_reco.train_global_transformer import (  # noqa: E402
    PairedTeacherLogitDataset,
    make_teacher_logit_loader,
    source_metadata,
)
from teacher_logit_reco.views import load_paired_jet_views, summarize_paired_jet_views  # noqa: E402


FAMILIES = {
    "qcd": {"QCD"},
    "higgs": {"Hbb", "Hcc", "Hgg", "H4q", "Hqql"},
    "vector": {"Zqq", "Wqq"},
    "top": {"Tbqq", "Tbl"},
}
CLASS_TO_FAMILY = {
    class_name: family
    for family, class_names in FAMILIES.items()
    for class_name in class_names
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--hlt-checkpoint", required=True)
    parser.add_argument("--hlt-architecture", choices=TEACHER_ARCHITECTURES, default="part")
    parser.add_argument("--offline-checkpoint", required=True)
    parser.add_argument("--offline-architecture", choices=TEACHER_ARCHITECTURES, default="part")
    parser.add_argument("--split", default="stack_val")
    parser.add_argument("--max-jets", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--weight-threshold", type=float, default=0.0)
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--a-confident-threshold", type=float, default=0.70)
    parser.add_argument("--b-confident-threshold", type=float, default=0.70)
    parser.add_argument("--uncertain-threshold", type=float, default=0.40)
    parser.add_argument("--teacher-confident-threshold", type=float, default=0.80)
    parser.add_argument("--save-per-jet", action="store_true")
    return parser.parse_args()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp, axis=1, keepdims=True), 1.0e-300, None)


def entropy_np(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    return -np.sum(probs * np.log(np.clip(probs, 1.0e-12, 1.0)), axis=1)


def margin_np(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] < 2:
        return np.zeros((probs.shape[0],), dtype=np.float64)
    top2 = np.partition(probs, kth=-2, axis=1)[:, -2:]
    top2.sort(axis=1)
    return top2[:, 1] - top2[:, 0]


def kl_np(left_probs: np.ndarray, right_probs: np.ndarray) -> np.ndarray:
    left = np.clip(np.asarray(left_probs, dtype=np.float64), 1.0e-12, 1.0)
    right = np.clip(np.asarray(right_probs, dtype=np.float64), 1.0e-12, 1.0)
    return np.sum(left * (np.log(left) - np.log(right)), axis=1)


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = softmax_np(logits)
    pred = np.argmax(probs, axis=1)
    picked = np.clip(probs[np.arange(len(labels)), labels], 1.0e-12, 1.0)
    return {
        "accuracy": float(np.mean(pred == labels)) if len(labels) else 0.0,
        "cross_entropy": float(-np.mean(np.log(picked))) if len(labels) else float("nan"),
        "mean_confidence": float(np.mean(np.max(probs, axis=1))) if len(labels) else 0.0,
        "mean_entropy": float(np.mean(entropy_np(probs))) if len(labels) else 0.0,
        "mean_margin": float(np.mean(margin_np(probs))) if len(labels) else 0.0,
        "n_jets": int(len(labels)),
    }


def summarize_mask(mask: np.ndarray, labels: np.ndarray, *, logits_by_name: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    n_total = int(labels.shape[0])
    n = int(np.sum(mask))
    if n == 0:
        return {"n_jets": 0, "fraction": 0.0}
    payload: Dict[str, Any] = {
        "n_jets": n,
        "fraction": float(n / max(1, n_total)),
        "class_counts": {
            LABEL_NAMES[index]: int(np.sum(labels[mask] == index))
            for index in range(len(LABEL_NAMES))
        },
    }
    for name, logits in logits_by_name.items():
        payload[name] = metrics_from_logits(logits[mask], labels[mask])
    return payload


def class_family_indices() -> np.ndarray:
    families = [CLASS_TO_FAMILY.get(name, "other") for name in LABEL_NAMES]
    unique = {name: idx for idx, name in enumerate(sorted(set(families)))}
    return np.asarray([unique[name] for name in families], dtype=np.int64)


def topk_overlap(left_probs: np.ndarray, right_probs: np.ndarray, *, k: int = 3) -> np.ndarray:
    left = np.argpartition(left_probs, kth=-k, axis=1)[:, -k:]
    right = np.argpartition(right_probs, kth=-k, axis=1)[:, -k:]
    return np.asarray([bool(set(a.tolist()) & set(b.tolist())) for a, b in zip(left, right)], dtype=bool)


def evaluate_three_paths(args: argparse.Namespace, *, device):
    torch = require_torch()
    pair = load_paired_jet_views(
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        split=args.split,
        data_dir=args.data_dir,
        max_jets=args.max_jets,
        read_chunk_size=args.read_chunk_size,
    )
    dataset = PairedTeacherLogitDataset(pair)
    loader = make_teacher_logit_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=12345,
    )
    hlt_model = load_frozen_teacher(
        args.hlt_checkpoint,
        architecture=args.hlt_architecture,
        device=str(device),
        max_constits=args.max_constits,
        weight_threshold=args.weight_threshold,
    )
    offline_model = load_frozen_teacher(
        args.offline_checkpoint,
        architecture=args.offline_architecture,
        device=str(device),
        max_constits=args.max_constits,
        weight_threshold=args.weight_threshold,
    )

    rows: Dict[str, list[np.ndarray]] = {
        "hlt_on_hlt": [],
        "offline_on_hlt": [],
        "offline_on_offline": [],
        "labels": [],
    }
    with torch.no_grad():
        for batch in loader:
            hlt_tokens = batch["hlt_tokens"].to(device=device, non_blocking=True)
            hlt_mask = batch["hlt_mask"].to(device=device, non_blocking=True)
            offline_tokens = batch["offline_tokens"].to(device=device, non_blocking=True)
            offline_mask = batch["offline_mask"].to(device=device, non_blocking=True)
            labels = batch["labels"].detach().cpu().numpy().astype(np.int64)
            rows["hlt_on_hlt"].append(hlt_model.forward_view_no_grad(hlt_tokens, hlt_mask).detach().cpu().numpy())
            rows["offline_on_hlt"].append(
                offline_model.forward_view_no_grad(hlt_tokens, hlt_mask).detach().cpu().numpy()
            )
            rows["offline_on_offline"].append(
                offline_model.forward_view_no_grad(offline_tokens, offline_mask).detach().cpu().numpy()
            )
            rows["labels"].append(labels)
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in rows.items()}
    return pair, hlt_model, offline_model, arrays


def build_report(args: argparse.Namespace, *, pair, hlt_model, offline_model, arrays: Mapping[str, np.ndarray]):
    labels = arrays["labels"].astype(np.int64)
    logits_a = arrays["hlt_on_hlt"].astype(np.float32)
    logits_b = arrays["offline_on_hlt"].astype(np.float32)
    logits_t = arrays["offline_on_offline"].astype(np.float32)
    probs_a = softmax_np(logits_a)
    probs_b = softmax_np(logits_b)
    probs_t = softmax_np(logits_t)
    pred_a = np.argmax(probs_a, axis=1)
    pred_b = np.argmax(probs_b, axis=1)
    pred_t = np.argmax(probs_t, axis=1)
    conf_a = np.max(probs_a, axis=1)
    conf_b = np.max(probs_b, axis=1)
    conf_t = np.max(probs_t, axis=1)
    correct_a = pred_a == labels
    correct_b = pred_b == labels
    correct_t = pred_t == labels
    agree_ab = pred_a == pred_b
    top3_overlap_ab = topk_overlap(probs_a, probs_b, k=3)
    family_index = class_family_indices()
    family_agree_ab = family_index[pred_a] == family_index[pred_b]

    logits_by_name = {
        "hlt_on_hlt": logits_a,
        "offline_on_hlt": logits_b,
        "offline_on_offline": logits_t,
    }
    choose_conf = np.where(conf_b > conf_a, pred_b, pred_a)
    choose_margin = np.where(margin_np(probs_b) > margin_np(probs_a), pred_b, pred_a)
    oracle_ab = correct_a | correct_b

    buckets = {
        "a_b_agree": agree_ab,
        "a_b_disagree": ~agree_ab,
        "both_correct": correct_a & correct_b,
        "both_wrong": ~correct_a & ~correct_b,
        "a_correct_b_wrong": correct_a & ~correct_b,
        "a_wrong_b_correct": ~correct_a & correct_b,
        "a_confident_b_uncertain": (conf_a >= args.a_confident_threshold) & (conf_b <= args.uncertain_threshold),
        "a_uncertain_b_confident": (conf_a <= args.uncertain_threshold) & (conf_b >= args.b_confident_threshold),
        "offline_teacher_confident_a_wrong": (conf_t >= args.teacher_confident_threshold) & ~correct_a,
        "top3_overlap": top3_overlap_ab,
        "top3_disjoint": ~top3_overlap_ab,
        "family_agree": family_agree_ab,
        "family_disagree": ~family_agree_ab,
    }
    bucket_reports = {
        name: summarize_mask(mask, labels, logits_by_name=logits_by_name)
        for name, mask in buckets.items()
    }

    disagreement_pair_counts: Dict[str, int] = {}
    for left, right in zip(pred_a[~agree_ab], pred_b[~agree_ab]):
        key = f"{LABEL_NAMES[int(left)]}->{LABEL_NAMES[int(right)]}"
        disagreement_pair_counts[key] = disagreement_pair_counts.get(key, 0) + 1
    disagreement_pair_counts = dict(
        sorted(disagreement_pair_counts.items(), key=lambda item: item[1], reverse=True)[:50]
    )

    return {
        "experiment": "hlt_offline_disagreement_diagnostic",
        "split": args.split,
        "max_jets": None if args.max_jets is None else int(args.max_jets),
        "n_jets": int(labels.shape[0]),
        "label_names": list(LABEL_NAMES),
        "source": source_metadata(),
        "paired_views": summarize_paired_jet_views(pair),
        "models": {
            "A_hlt_on_hlt": dict(hlt_model.metadata),
            "B_offline_on_hlt": dict(offline_model.metadata),
            "T_offline_on_offline": dict(offline_model.metadata),
        },
        "overall": {
            "hlt_on_hlt": metrics_from_logits(logits_a, labels),
            "offline_on_hlt": metrics_from_logits(logits_b, labels),
            "offline_on_offline": metrics_from_logits(logits_t, labels),
            "a_b_argmax_agreement_fraction": float(np.mean(agree_ab)),
            "a_b_top3_overlap_fraction": float(np.mean(top3_overlap_ab)),
            "a_b_family_agreement_fraction": float(np.mean(family_agree_ab)),
            "oracle_a_or_b_accuracy": float(np.mean(oracle_ab)),
            "choose_higher_confidence_accuracy": float(np.mean(choose_conf == labels)),
            "choose_higher_margin_accuracy": float(np.mean(choose_margin == labels)),
            "mean_kl_a_to_b": float(np.mean(kl_np(probs_a, probs_b))),
            "mean_kl_b_to_a": float(np.mean(kl_np(probs_b, probs_a))),
        },
        "buckets": bucket_reports,
        "top_disagreement_pairs_a_to_b": disagreement_pair_counts,
        "interpretation": {
            "A_hlt_on_hlt": "normal HLT-trained tagger evaluated on HLT",
            "B_offline_on_hlt": "offline-trained tagger evaluated on HLT; still HLT-only at inference",
            "T_offline_on_offline": "offline upper-reference path",
            "oracle_a_or_b_accuracy": "best possible selector if it knew whether A or B was correct per jet",
            "a_wrong_b_correct": "golden bucket: B exposes complementary HLT-only signal if sizable",
        },
    }


def save_per_jet_summary(output_dir: Path, arrays: Mapping[str, np.ndarray]) -> None:
    labels = arrays["labels"].astype(np.int64)
    probs_a = softmax_np(arrays["hlt_on_hlt"])
    probs_b = softmax_np(arrays["offline_on_hlt"])
    probs_t = softmax_np(arrays["offline_on_offline"])
    pred_a = np.argmax(probs_a, axis=1)
    pred_b = np.argmax(probs_b, axis=1)
    pred_t = np.argmax(probs_t, axis=1)
    rows = np.stack(
        [
            labels,
            pred_a,
            pred_b,
            pred_t,
            np.max(probs_a, axis=1),
            np.max(probs_b, axis=1),
            np.max(probs_t, axis=1),
            entropy_np(probs_a),
            entropy_np(probs_b),
            entropy_np(probs_t),
            margin_np(probs_a),
            margin_np(probs_b),
            margin_np(probs_t),
            kl_np(probs_a, probs_b),
            kl_np(probs_b, probs_a),
        ],
        axis=1,
    )
    header = (
        "label,pred_hlt_on_hlt,pred_offline_on_hlt,pred_offline_on_offline,"
        "conf_hlt_on_hlt,conf_offline_on_hlt,conf_offline_on_offline,"
        "entropy_hlt_on_hlt,entropy_offline_on_hlt,entropy_offline_on_offline,"
        "margin_hlt_on_hlt,margin_offline_on_hlt,margin_offline_on_offline,"
        "kl_hlt_to_offline_on_hlt,kl_offline_on_hlt_to_hlt"
    )
    np.savetxt(output_dir / "per_jet_summary.csv", rows, delimiter=",", header=header, comments="")


def main() -> int:
    args = parse_args()
    if args.max_jets is not None and int(args.max_jets) <= 0:
        raise ValueError("max-jets must be positive when provided")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    pair, hlt_model, offline_model, arrays = evaluate_three_paths(args, device=device)
    report = build_report(args, pair=pair, hlt_model=hlt_model, offline_model=offline_model, arrays=arrays)
    save_json(output_dir / "disagreement_diagnostic_report.json", report)
    np.savez_compressed(
        output_dir / "disagreement_logits.npz",
        labels=arrays["labels"].astype(np.int64),
        hlt_on_hlt=arrays["hlt_on_hlt"].astype(np.float32),
        offline_on_hlt=arrays["offline_on_hlt"].astype(np.float32),
        offline_on_offline=arrays["offline_on_offline"].astype(np.float32),
    )
    if args.save_per_jet:
        save_per_jet_summary(output_dir, arrays)

    overall = report["overall"]
    print("hlt_offline_disagreement_diagnostic_complete:")
    print(f"  output_dir: {output_dir}")
    print(f"  split: {args.split}")
    print(f"  n_jets: {report['n_jets']}")
    print(f"  hlt_on_hlt_acc: {overall['hlt_on_hlt']['accuracy']:.6f}")
    print(f"  offline_on_hlt_acc: {overall['offline_on_hlt']['accuracy']:.6f}")
    print(f"  offline_on_offline_acc: {overall['offline_on_offline']['accuracy']:.6f}")
    print(f"  a_b_agreement_fraction: {overall['a_b_argmax_agreement_fraction']:.6f}")
    print(f"  oracle_a_or_b_acc: {overall['oracle_a_or_b_accuracy']:.6f}")
    print(f"  choose_higher_confidence_acc: {overall['choose_higher_confidence_accuracy']:.6f}")
    print(
        "  a_wrong_b_correct_fraction: "
        f"{report['buckets']['a_wrong_b_correct']['fraction']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

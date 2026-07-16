#!/usr/bin/env python3
"""Fit or apply an immutable stack-only ABPH logit fusion."""

from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    ABPH_FUSION_CANDIDATES,
    LogitPredictionBlock,
    apply_frozen_fusion,
    fit_frozen_stack_fusion,
    load_frozen_fusion_artifact,
    resolve_variant_config,
    write_frozen_fusion_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(ABPH_FUSION_CANDIDATES), required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument("--apply-split", choices=("stack_train", "stack_val", "final_test"))
    parser.add_argument("--frozen-artifact")
    return parser


def _json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _block(root: Path, member: str, split: str) -> LogitPredictionBlock:
    array_path = root / "logit_predictions" / member / f"{split}.npz"
    metadata_path = root / "logit_predictions" / member / f"{split}_metadata.json"
    metadata = _json(metadata_path)
    with np.load(array_path, allow_pickle=False) as arrays:
        block = LogitPredictionBlock(
            member=member,
            split=split,
            logits=np.asarray(arrays["logits"]),
            labels=np.asarray(arrays["labels"]),
            jet_ids=np.asarray(arrays["jet_ids"]),
            checkpoint_hash=str(metadata["checkpoint_hash"]),
            resolved_config_hash=str(metadata["resolved_config_hash"]),
            provenance=dict(metadata["provenance"]),
        )
    block.validate()
    if metadata.get("prediction_hash") != block.prediction_hash:
        raise ValueError(f"{member}/{split} prediction hash mismatch")
    return block


def _candidate_members(variant: str, supplied: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    candidates = tuple(str(name) for name in (supplied or ABPH_FUSION_CANDIDATES[variant]))
    if candidates != tuple(ABPH_FUSION_CANDIDATES[variant]):
        raise ValueError("fusion candidate membership/order differs from the frozen registry")
    if variant in {"G2_kt_ca_logit_fusion", "G4_seed_ensemble_primary"}:
        return (candidates,)
    if variant == "G3_particle_and_logit_fusion":
        anchor = "E7_dual_hierarchy_dualcross"
        return tuple((anchor, member) for member in candidates if member != anchor)
    anchors = {
        "E5_kt32_mh4_dualcross",
        "E6_ca32_mh4_dualcross",
        "E7_dual_hierarchy_dualcross",
    }
    rows = []
    for count in (2, 3, 4):
        rows.extend(
            members
            for members in combinations(candidates, count)
            if anchors.intersection(members)
        )
    return tuple(rows)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(labels.shape[0]), labels].mean())


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _fit(root: Path, variant: str, supplied: Sequence[str]) -> dict[str, Any]:
    best = None
    candidate_rows = []
    for members in _candidate_members(variant, supplied):
        train = tuple(_block(root, member, "stack_train") for member in members)
        val = tuple(_block(root, member, "stack_val") for member in members)
        artifact = fit_frozen_stack_fusion(variant, members, train, val)
        row = {
            "members": list(members),
            "stack_val_cross_entropy": artifact.stack_val_cross_entropy,
            "membership_hash": artifact.membership_hash,
        }
        candidate_rows.append(row)
        key = (artifact.stack_val_cross_entropy, len(members), artifact.membership_hash)
        if best is None or key < best[0]:
            best = (key, artifact, val)
    if best is None:
        raise RuntimeError("fusion search produced no candidate")
    _, artifact, val_blocks = best
    output_dir = root / "fusion" / variant
    artifact_path = output_dir / "frozen_fusion.json"
    write_frozen_fusion_artifact(artifact_path, artifact)
    fused = apply_frozen_fusion(artifact, val_blocks)
    labels = np.asarray(val_blocks[0].labels, dtype=np.int64)
    provenance = dict(val_blocks[0].provenance)
    artifact_payload = artifact.to_dict()
    report = {
        "ok": True,
        "variant_name": variant,
        "resolved_variant_config_hash": resolve_variant_config(variant)["resolved_config_hash"],
        "selected_checkpoint_hash": artifact_payload["artifact_hash"],
        "source_git_commit": provenance.get("source_git_commit", "unknown"),
        "source_status_hash": provenance.get("source_status_hash", "unknown"),
        "metrics": {
            "stack_val": {
                "available": True,
                "accuracy": _accuracy(fused, labels),
                "cross_entropy": _cross_entropy(fused, labels),
                "n_jets": int(labels.shape[0]),
            }
        },
        "provenance": {"stack_val": provenance},
        "fusion": {
            "members": list(artifact.members),
            "membership_hash": artifact.membership_hash,
            "artifact_hash": artifact_payload["artifact_hash"],
            "candidate_search": candidate_rows,
            "fit_split": "stack_train",
            "selection_split": "stack_val",
        },
    }
    _atomic_json(output_dir / "fusion_report.json", report)
    _atomic_json(root / "runs" / variant / "run_report.json", report)
    return report


def _apply(root: Path, variant: str, split: str, artifact_path: Path) -> dict[str, Any]:
    artifact = load_frozen_fusion_artifact(artifact_path)
    if artifact.fusion_variant != variant:
        raise ValueError("frozen fusion belongs to a different variant")
    blocks = tuple(_block(root, member, split) for member in artifact.members)
    logits = apply_frozen_fusion(artifact, blocks)
    labels = np.asarray(blocks[0].labels, dtype=np.int64)
    output_dir = root / "fusion" / variant
    np.savez_compressed(
        output_dir / f"{split}.npz",
        logits=np.asarray(logits, dtype=np.float32),
        labels=labels,
        jet_ids=np.asarray(blocks[0].jet_ids),
    )
    report_path = root / "runs" / variant / "run_report.json"
    report = dict(_json(report_path))
    metrics = dict(report.get("metrics", {}))
    metrics[split] = {
        "available": True,
        "accuracy": _accuracy(logits, labels),
        "cross_entropy": _cross_entropy(logits, labels),
        "n_jets": int(labels.shape[0]),
        "diagnostics": {
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "hypothesis_selection_used_offline_target": False,
            "fusion_fitted_on_final_test": False,
        },
    }
    report["metrics"] = metrics
    provenance = dict(report.get("provenance", {}))
    split_provenance = dict(blocks[0].provenance)
    split_provenance.update(
        {
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "hypothesis_selection_used_offline_target": False,
            "fusion_fitted_on_final_test": False,
        }
    )
    provenance[split] = split_provenance
    report["provenance"] = provenance
    _atomic_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.campaign_root)
    if args.apply_split:
        if not args.frozen_artifact:
            raise SystemExit("--apply-split requires --frozen-artifact")
        report = _apply(root, args.variant, args.apply_split, Path(args.frozen_artifact))
    else:
        report = _fit(root, args.variant, args.member)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

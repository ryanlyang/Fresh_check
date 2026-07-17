#!/usr/bin/env python3
"""Build a deterministic, balanced C2F runtime-calibration manifest subset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import (  # noqa: E402
    BALANCED_SPLIT_ROW_ORDERING,
    SPLIT_ORDER,
    SplitManifest,
    audit_split_manifest,
    load_split_manifest,
    manifest_hash,
    save_split_manifest,
)
from jetclass_fresh.hlt_cache import jet_identity_hash  # noqa: E402


CALIBRATION_SLICE_CONTRACT = "constrained_c2f_runtime_calibration_slice_v1"
DEFAULT_CALIBRATION_SEED = 81173


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _label_counts(rows: Sequence[Any], n_classes: int) -> dict[str, int]:
    counts = Counter(int(identity.label) for identity in rows)
    return {str(label): int(counts.get(label, 0)) for label in range(n_classes)}


def _balanced_subset(
    rows: Sequence[Any],
    *,
    split: str,
    requested_size: int,
    n_classes: int,
    seed: int,
) -> list[Any]:
    if requested_size <= 0:
        raise ValueError(f"{split} calibration size must be positive")
    if requested_size % n_classes:
        raise ValueError(f"{split} calibration size must be divisible by {n_classes}")
    per_class = requested_size // n_classes
    buckets: dict[int, list[Any]] = defaultdict(list)
    for identity in rows:
        buckets[int(identity.label)].append(identity)
    selected: list[Any] = []
    for label in range(n_classes):
        available = buckets.get(label, [])
        if len(available) < per_class:
            raise ValueError(
                f"{split} has only {len(available)} rows for label {label}, but calibration needs {per_class}"
            )
        # Parent manifests are already deterministically mixed. Preserve their
        # row membership decision, then reshuffle the selected balanced subset
        # with a documented independent seed.
        selected.extend(available[:per_class])
    permutation = np.random.RandomState(int(seed)).permutation(len(selected))
    return [selected[int(index)] for index in permutation]


def build_calibration_manifest(
    parent: SplitManifest,
    *,
    model_train_size: int,
    model_val_size: int,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> SplitManifest:
    """Derive a balanced train/validation-only manifest from a campaign manifest."""

    parent_audit = audit_split_manifest(parent)
    if not bool(parent_audit.get("ok")):
        raise ValueError("parent manifest failed its integrity audit")
    if parent.metadata.get("row_ordering") != BALANCED_SPLIT_ROW_ORDERING:
        raise ValueError("parent manifest does not use the required globally mixed row ordering")
    n_classes = len(parent.class_names)
    train_rows = _balanced_subset(
        parent.splits["model_train"],
        split="model_train",
        requested_size=int(model_train_size),
        n_classes=n_classes,
        seed=int(seed) + 101,
    )
    val_rows = _balanced_subset(
        parent.splits["model_val"],
        split="model_val",
        requested_size=int(model_val_size),
        n_classes=n_classes,
        seed=int(seed) + 211,
    )
    parent_sha = manifest_hash(parent)
    metadata = {
        "contract": CALIBRATION_SLICE_CONTRACT,
        "parent_manifest_hash": parent_sha,
        "selection_strategy": "deterministic_balanced_subset_then_globally_mixed",
        "selection_seed": int(seed),
        "row_ordering": BALANCED_SPLIT_ROW_ORDERING,
        "parent_split_membership": {
            "model_train": "parent.model_train",
            "model_val": "parent.model_val",
        },
        "requested_sizes": {
            "model_train": int(model_train_size),
            "model_val": int(model_val_size),
        },
        "selected_label_counts": {
            "model_train": _label_counts(train_rows, n_classes),
            "model_val": _label_counts(val_rows, n_classes),
        },
        "selected_jet_identity_hashes": {
            "model_train": jet_identity_hash(train_rows),
            "model_val": jet_identity_hash(val_rows),
        },
    }
    manifest = SplitManifest(
        data_dir=parent.data_dir,
        max_constits=int(parent.max_constits),
        class_names=list(parent.class_names),
        file_prefix_to_label=dict(parent.file_prefix_to_label),
        split_sizes={
            "model_train": int(model_train_size),
            "model_val": int(model_val_size),
            "stack_train": 0,
            "stack_val": 0,
            "final_test": 0,
        },
        split_seeds={split: int(parent.split_seeds.get(split, 0)) for split in SPLIT_ORDER},
        file_records=list(parent.file_records),
        splits={
            "model_train": train_rows,
            "model_val": val_rows,
            "stack_train": [],
            "stack_val": [],
            "final_test": [],
        },
        metadata=metadata,
    )
    audit = audit_split_manifest(manifest)
    if not bool(audit.get("ok")):
        raise ValueError(f"derived calibration manifest failed its integrity audit: {audit}")
    return manifest


def _report(parent: SplitManifest, calibration: SplitManifest, output_path: Path) -> dict[str, Any]:
    parent_sha = manifest_hash(parent)
    calibration_sha = manifest_hash(calibration)
    payload = {
        "ok": True,
        "contract": CALIBRATION_SLICE_CONTRACT,
        "parent_manifest_hash": parent_sha,
        "calibration_manifest_hash": calibration_sha,
        "calibration_manifest_path": str(output_path),
        "selected_sizes": dict(calibration.split_sizes),
        "selected_label_counts": calibration.metadata["selected_label_counts"],
        "selected_jet_identity_hashes": calibration.metadata["selected_jet_identity_hashes"],
    }
    payload["report_sha256"] = _sha256_json(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-train", type=int, default=50_000)
    parser.add_argument("--model-val", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    parent_path = Path(args.parent_manifest)
    output_path = Path(args.output_manifest)
    report_path = Path(args.report)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration manifest: {output_path}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration report: {report_path}")
    parent = load_split_manifest(parent_path)
    calibration = build_calibration_manifest(
        parent,
        model_train_size=int(args.model_train),
        model_val_size=int(args.model_val),
        seed=int(args.seed),
    )
    save_split_manifest(calibration, output_path, pretty=bool(args.pretty))
    report = _report(parent, calibration, output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

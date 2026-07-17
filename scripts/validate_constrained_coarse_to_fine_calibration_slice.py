#!/usr/bin/env python3
"""Fail closed on C2F runtime-calibration manifest and cache mismatches."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_cache import jet_identity_hash  # noqa: E402
from jetclass_fresh.jetclass_data import (  # noqa: E402
    BALANCED_SPLIT_ROW_ORDERING,
    audit_split_manifest,
    load_split_manifest,
    manifest_hash,
)
from scripts.build_constrained_coarse_to_fine_calibration_slice import (  # noqa: E402
    CALIBRATION_SLICE_CONTRACT,
)
from scripts.validate_constrained_coarse_to_fine_artifact import (  # noqa: E402
    _read_json,
    _validate_hlt_cache,
    _validate_offline_cache,
    _validate_target_cache,
)


CALIBRATION_SPLITS = ("model_train", "model_val")


def _identity_keys(rows: Sequence[Any]) -> set[tuple[str, int, int]]:
    return {identity.key() for identity in rows}


def _expected_counts(manifest: Any, split: str) -> dict[str, int]:
    counts = Counter(int(identity.label) for identity in manifest.splits[split])
    return {str(label): int(counts.get(label, 0)) for label in range(len(manifest.class_names))}


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _validate_manifest(parent_path: Path, calibration_path: Path) -> tuple[Any, Any, str]:
    parent = load_split_manifest(parent_path)
    calibration = load_split_manifest(calibration_path)
    parent_audit = audit_split_manifest(parent)
    calibration_audit = audit_split_manifest(calibration)
    if not bool(parent_audit.get("ok")):
        raise ValueError("parent manifest failed integrity audit")
    if not bool(calibration_audit.get("ok")):
        raise ValueError("calibration manifest failed integrity audit")
    metadata = calibration.metadata
    _require_equal(metadata.get("contract"), CALIBRATION_SLICE_CONTRACT, "calibration manifest contract")
    parent_sha = manifest_hash(parent)
    _require_equal(metadata.get("parent_manifest_hash"), parent_sha, "calibration parent manifest hash")
    _require_equal(metadata.get("row_ordering"), BALANCED_SPLIT_ROW_ORDERING, "calibration row ordering")
    for split in CALIBRATION_SPLITS:
        parent_keys = _identity_keys(parent.splits[split])
        calibration_rows = calibration.splits[split]
        selected_keys = _identity_keys(calibration_rows)
        if not selected_keys <= parent_keys:
            raise ValueError(f"calibration {split} contains rows outside parent {split}")
        _require_equal(len(selected_keys), len(calibration_rows), f"calibration {split} duplicate identities")
        _require_equal(calibration.split_sizes[split], len(calibration_rows), f"calibration {split} size")
        _require_equal(
            metadata.get("selected_jet_identity_hashes", {}).get(split),
            jet_identity_hash(calibration_rows),
            f"calibration {split} identity hash",
        )
        observed_counts = _expected_counts(calibration, split)
        _require_equal(
            metadata.get("selected_label_counts", {}).get(split),
            observed_counts,
            f"calibration {split} label counts",
        )
        values = set(observed_counts.values())
        if len(values) != 1:
            raise ValueError(f"calibration {split} is not balanced by label: {observed_counts}")
    for split in ("stack_train", "stack_val", "final_test"):
        _require_equal(calibration.split_sizes[split], 0, f"calibration {split} size")
        _require_equal(calibration.splits[split], [], f"calibration {split} rows")
    return parent, calibration, manifest_hash(calibration)


def _validate_rows_against_manifest(
    calibration: Any,
    *,
    hlt_cache_dir: Path,
    offline_cache_dir: Path,
    target_cache_dir: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in CALIBRATION_SPLITS:
        expected_rows = calibration.splits[split]
        expected_ids = jet_identity_hash(expected_rows)
        expected_labels = np.asarray([int(identity.label) for identity in expected_rows], dtype=np.int64)
        hlt = _read_json(hlt_cache_dir / f"{split}_fixed_hlt_metadata.json")
        offline = _read_json(offline_cache_dir / f"{split}_offline_metadata.json")
        target = _read_json(target_cache_dir / f"{split}_hierarchy_targets_metadata.json")
        for label, metadata in (("HLT", hlt), ("offline", offline), ("target", target)):
            _require_equal(metadata.get("n_jets"), len(expected_rows), f"{label}/{split} n_jets")
            _require_equal(metadata.get("jet_identity_hash"), expected_ids, f"{label}/{split} identity")
        hlt_labels = np.load(hlt_cache_dir / f"{split}_fixed_hlt.npz", allow_pickle=False)["labels"]
        offline_labels = np.load(offline_cache_dir / f"{split}_offline.npz", allow_pickle=False)["labels"]
        if not np.array_equal(hlt_labels, expected_labels):
            raise ValueError(f"HLT/{split} labels do not match calibration manifest")
        if not np.array_equal(offline_labels, expected_labels):
            raise ValueError(f"offline/{split} labels do not match calibration manifest")
        report[split] = {
            "n_jets": len(expected_rows),
            "jet_identity_hash": expected_ids,
            "hlt_content_hash": hlt["hlt_content_hash"],
            "offline_content_hash": offline.get("offline_content_hash") or offline.get("content_hash"),
            "target_content_hash": target["target_content_hash"],
        }
    return report


def validate_calibration_slice(
    *,
    parent_manifest_path: Path,
    calibration_manifest_path: Path,
    hlt_cache_dir: Path,
    offline_cache_dir: Path,
    target_cache_dir: Path,
) -> dict[str, Any]:
    parent, calibration, calibration_sha = _validate_manifest(parent_manifest_path, calibration_manifest_path)
    _validate_hlt_cache(hlt_cache_dir, CALIBRATION_SPLITS, calibration_sha)
    _validate_offline_cache(offline_cache_dir, CALIBRATION_SPLITS, calibration_sha)
    _validate_target_cache(
        target_cache_dir,
        CALIBRATION_SPLITS,
        calibration_sha,
        hlt_cache_dir=hlt_cache_dir,
        offline_cache_dir=offline_cache_dir,
    )
    return {
        "ok": True,
        "contract": CALIBRATION_SLICE_CONTRACT,
        "parent_manifest_hash": manifest_hash(parent),
        "calibration_manifest_hash": calibration_sha,
        "splits": _validate_rows_against_manifest(
            calibration,
            hlt_cache_dir=hlt_cache_dir,
            offline_cache_dir=offline_cache_dir,
            target_cache_dir=target_cache_dir,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", required=True)
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_calibration_slice(
        parent_manifest_path=Path(args.parent_manifest),
        calibration_manifest_path=Path(args.calibration_manifest),
        hlt_cache_dir=Path(args.hlt_cache_dir),
        offline_cache_dir=Path(args.offline_cache_dir),
        target_cache_dir=Path(args.target_cache_dir),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

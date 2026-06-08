#!/usr/bin/env python3
"""Verify V2 Step 1 split and fixed-HLT cache artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_cache import (  # noqa: E402
    DEFAULT_HLT_SEEDS,
    audit_hlt_cache,
    fixed_hlt_params_dict,
    hash_arrays,
    load_cached_hlt_view,
    load_hlt_metadata,
)
from jetclass_fresh.jetclass_data import (  # noqa: E402
    DEFAULT_SPLIT_TOTALS,
    LABEL_NAMES,
    SPLIT_ORDER,
    audit_split_manifest,
    load_split_manifest,
    manifest_hash,
    split_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Five-way split manifest path (.json or .json.gz)",
    )
    parser.add_argument(
        "--hlt-cache-dir",
        required=True,
        help="Directory containing per-split fixed-HLT .npz and metadata JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/jetclass_v2_step1_audit",
        help="Directory for Step 1 audit reports",
    )
    return parser.parse_args()


def count_summary(counts: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return {
            "n_jets": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    return {
        "n_jets": int(counts.size),
        "min": float(np.min(counts)),
        "max": float(np.max(counts)),
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "p10": float(np.percentile(counts, 10)),
        "p50": float(np.percentile(counts, 50)),
        "p90": float(np.percentile(counts, 90)),
    }


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = {name: 0 for name in LABEL_NAMES}
    for label in np.asarray(labels, dtype=np.int64):
        counts[LABEL_NAMES[int(label)]] += 1
    return counts


def read_cache_array_report(cache_dir: Path, split: str) -> dict[str, Any]:
    metadata = load_hlt_metadata(cache_dir, split)
    hlt_view = load_cached_hlt_view(cache_dir, split, verify_hash=True)
    array_path = cache_dir / f"{split}_fixed_hlt.npz"
    with np.load(array_path, allow_pickle=False) as data:
        files = sorted(data.files)
        file_indices = data["jet_file_indices"].astype(np.int32, copy=False)
        entries = data["jet_entries"]
        actual_content_hash = hash_arrays(
            {
                "tokens": data["tokens"],
                "mask": data["mask"],
                "labels": data["labels"],
                "jet_file_indices": file_indices,
                "jet_entries": entries,
            }
        )
        tokens_shape = list(data["tokens"].shape)
        mask_shape = list(data["mask"].shape)
        labels_shape = list(data["labels"].shape)

    hlt_counts = np.sum(hlt_view.mask, axis=1)
    return {
        "array_path": str(array_path),
        "metadata_path": str(cache_dir / f"{split}_fixed_hlt_metadata.json"),
        "array_keys": files,
        "tokens_shape": tokens_shape,
        "mask_shape": mask_shape,
        "labels_shape": labels_shape,
        "n_jets": int(hlt_view.tokens.shape[0]),
        "class_counts_from_cache": class_counts(hlt_view.labels),
        "metadata_class_count_source": "labels in fixed-HLT npz",
        "seed": int(metadata.get("seed", -1)),
        "expected_seed": int(DEFAULT_HLT_SEEDS[split]),
        "hlt_params": metadata.get("hlt_params"),
        "expected_hlt_params": fixed_hlt_params_dict(),
        "source_manifest_hash": metadata.get("source_manifest_hash"),
        "hlt_content_hash": metadata.get("hlt_content_hash"),
        "actual_hlt_content_hash": actual_content_hash,
        "content_hash_matches_metadata": actual_content_hash == metadata.get("hlt_content_hash"),
        "jet_identity_hash": metadata.get("jet_identity_hash"),
        "source_content_hash": metadata.get("source_content_hash"),
        "diagnostics_hash": metadata.get("diagnostics_hash"),
        "offline_constit_count_summary": metadata.get("offline_constit_count_summary"),
        "hlt_constit_count_summary": metadata.get("hlt_constit_count_summary"),
        "hlt_constit_count_summary_recomputed": count_summary(hlt_counts),
        "hlt_diagnostics_summary": metadata.get("hlt_diagnostics_summary"),
        "generator": metadata.get("generator"),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_summary(path: Path, split_report: Mapping[str, Any], hlt_report: Mapping[str, Any]) -> None:
    lines = [
        "# V2 Step 1 Split And HLT Cache Audit",
        "",
        f"overall_ok: {split_report['ok'] and hlt_report['ok']}",
        f"manifest_hash: `{split_report['manifest_hash']}`",
        "",
        "## Split Counts",
        "",
        "| split | jets | expected | ok |",
        "|---|---:|---:|---|",
    ]
    split_counts = split_report["split_summary"]["split_counts"]
    for split in SPLIT_ORDER:
        actual = int(split_counts[split])
        expected = int(DEFAULT_SPLIT_TOTALS[split])
        lines.append(f"| {split} | {actual} | {expected} | {actual == expected} |")

    lines.extend(
        [
            "",
            "## HLT Cache",
            "",
            "| split | jets | seed | offline mean count | HLT mean count | HLT hash | ok |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for split in SPLIT_ORDER:
        item = hlt_report["split_reports"][split]
        offline_mean = item["offline_constit_count_summary"]["mean"]
        hlt_mean = item["hlt_constit_count_summary"]["mean"]
        digest = item["hlt_content_hash"]
        lines.append(
            f"| {split} | {item['n_jets']} | {item['seed']} | "
            f"{offline_mean:.4f} | {hlt_mean:.4f} | `{digest[:12]}...` | {item['ok']} |"
        )

    lines.extend(
        [
            "",
            "## Leakage-Relevant Split Checks",
            "",
            f"duplicate_within_split_count: {split_report['split_audit']['duplicate_within_split_count']}",
            f"cross_split_overlap_count: {split_report['split_audit']['cross_split_overlap_count']}",
            (
                "file_level_separation_claimed: "
                f"{split_report['split_audit']['file_level_separation_claimed']}"
            ),
            "",
            "File overlap is expected for this manifest because separation is by `(file, entry)` jet identity, not by file.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    cache_dir = Path(args.hlt_cache_dir)
    output_dir = Path(args.output_dir)

    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    split_audit = audit_split_manifest(manifest)
    summary = split_summary(manifest)

    split_report = {
        "ok": bool(split_audit["ok"]),
        "experiment_step": "v2_step1_verify_data_splits",
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "protocol_expected_split_sizes": dict(DEFAULT_SPLIT_TOTALS),
        "split_summary": summary,
        "split_audit": split_audit,
    }

    base_hlt_audit = audit_hlt_cache(manifest, cache_dir, splits=SPLIT_ORDER)
    split_reports = {}
    for split in SPLIT_ORDER:
        item = read_cache_array_report(cache_dir, split)
        base_item = base_hlt_audit["split_reports"][split]
        problems = list(base_item.get("problems") or [])
        split_ok = bool(base_item.get("ok"))
        if item["n_jets"] != DEFAULT_SPLIT_TOTALS[split]:
            split_ok = False
            problems.append(f"n_jets is {item['n_jets']}, expected {DEFAULT_SPLIT_TOTALS[split]}")
        if item["seed"] != item["expected_seed"]:
            split_ok = False
            problems.append(f"seed is {item['seed']}, expected {item['expected_seed']}")
        if item["hlt_params"] != item["expected_hlt_params"]:
            split_ok = False
            problems.append("HLT params do not match fixed-HLT defaults")
        if item["source_manifest_hash"] != manifest_sha:
            split_ok = False
            problems.append("source_manifest_hash does not match manifest hash")
        if not item["content_hash_matches_metadata"]:
            split_ok = False
            problems.append("recomputed HLT content hash does not match metadata")
        item.update({"ok": bool(split_ok), "problems": problems})
        split_reports[split] = item

    hlt_report = {
        "ok": bool(base_hlt_audit["ok"] and all(item["ok"] for item in split_reports.values())),
        "experiment_step": "v2_step1_verify_fixed_hlt_cache",
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "expected_hlt_params": fixed_hlt_params_dict(),
        "expected_hlt_seeds": dict(DEFAULT_HLT_SEEDS),
        "base_audit": base_hlt_audit,
        "split_reports": split_reports,
        "all_splits_have_distinct_content_hashes": bool(
            base_hlt_audit.get("all_splits_have_distinct_content_hashes")
        ),
    }

    write_json(output_dir / "split_audit_report.json", split_report)
    write_json(output_dir / "hlt_cache_audit_report.json", hlt_report)
    write_summary(output_dir / "step1_audit_summary.md", split_report, hlt_report)

    result = {
        "ok": bool(split_report["ok"] and hlt_report["ok"]),
        "output_dir": str(output_dir),
        "split_audit_report": str(output_dir / "split_audit_report.json"),
        "hlt_cache_audit_report": str(output_dir / "hlt_cache_audit_report.json"),
        "summary": str(output_dir / "step1_audit_summary.md"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

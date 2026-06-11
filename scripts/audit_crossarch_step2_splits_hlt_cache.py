#!/usr/bin/env python3
"""Verify cross-architecture Step 2 split and fixed-HLT cache artifacts."""

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
    LABEL_NAMES,
    audit_split_manifest,
    load_split_manifest,
    manifest_hash,
    split_summary,
)
from teacher_logit_reco.crossarch_experiment import (  # noqa: E402
    EXPERIMENT_NAME,
    SPLIT_ORDER,
    SPLIT_SIZES,
    CrossArchExperimentLayout,
)


EXPERIMENT_STEP = "crossarch_step2_verify_splits_hlt_cache"


def parse_args() -> argparse.Namespace:
    layout = CrossArchExperimentLayout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(layout.split_manifest_path),
        help="Cross-architecture five-way split manifest path (.json or .json.gz)",
    )
    parser.add_argument(
        "--hlt-cache-dir",
        default=str(layout.hlt_cache_dir),
        help="Directory containing per-split fixed-HLT .npz and metadata JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layout.audits_dir / "step2_splits_hlt_cache"),
        help="Directory for cross-architecture Step 2 audit reports",
    )
    return parser.parse_args()


def expected_split_sizes() -> dict[str, int]:
    return {split: int(SPLIT_SIZES[split]) for split in SPLIT_ORDER}


def split_size_problems(
    declared_sizes: Mapping[str, int],
    actual_counts: Mapping[str, int],
    expected_counts: Mapping[str, int] | None = None,
) -> list[str]:
    expected_counts = expected_split_sizes() if expected_counts is None else expected_counts
    problems: list[str] = []
    for split in SPLIT_ORDER:
        expected = int(expected_counts[split])
        declared = int(declared_sizes.get(split, -1))
        actual = int(actual_counts.get(split, -1))
        if declared != expected:
            problems.append(f"{split} declared size is {declared}, expected {expected}")
        if actual != expected:
            problems.append(f"{split} actual count is {actual}, expected {expected}")
    extra_declared = sorted(set(declared_sizes) - set(SPLIT_ORDER))
    extra_actual = sorted(set(actual_counts) - set(SPLIT_ORDER))
    if extra_declared:
        problems.append(f"unexpected declared split sizes: {extra_declared}")
    if extra_actual:
        problems.append(f"unexpected actual split counts: {extra_actual}")
    return problems


def class_balance_problems(
    class_counts_by_split: Mapping[str, Mapping[str, int]],
    expected_counts: Mapping[str, int] | None = None,
) -> list[str]:
    expected_counts = expected_split_sizes() if expected_counts is None else expected_counts
    problems: list[str] = []
    n_classes = len(LABEL_NAMES)
    for split in SPLIT_ORDER:
        expected_per_class = int(expected_counts[split]) // n_classes
        if int(expected_counts[split]) % n_classes:
            problems.append(f"{split} expected size is not divisible by {n_classes}")
            continue
        counts = class_counts_by_split.get(split, {})
        for label_name in LABEL_NAMES:
            actual = int(counts.get(label_name, -1))
            if actual != expected_per_class:
                problems.append(
                    f"{split}/{label_name} count is {actual}, expected {expected_per_class}"
                )
    return problems


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
        files = sorted(data.files)
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


def hlt_cache_split_problems(
    item: Mapping[str, Any],
    *,
    base_problems: list[str] | None,
    expected_size: int,
    expected_seed: int,
    manifest_sha: str,
) -> list[str]:
    problems = list(base_problems or [])
    if int(item.get("n_jets", -1)) != int(expected_size):
        problems.append(f"n_jets is {item.get('n_jets')}, expected {expected_size}")
    if int(item.get("seed", -1)) != int(expected_seed):
        problems.append(f"seed is {item.get('seed')}, expected {expected_seed}")
    if item.get("hlt_params") != item.get("expected_hlt_params"):
        problems.append("HLT params do not match fixed-HLT defaults")
    if item.get("source_manifest_hash") != manifest_sha:
        problems.append("source_manifest_hash does not match manifest hash")
    if not bool(item.get("content_hash_matches_metadata")):
        problems.append("recomputed HLT content hash does not match metadata")
    return problems


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_summary(path: Path, combined_report: Mapping[str, Any]) -> None:
    split_report = combined_report["audits"]["split_manifest"]
    hlt_report = combined_report["audits"]["hlt_cache"]
    lines = [
        "# CrossArch Step 2 Split And HLT Cache Audit",
        "",
        f"experiment_name: `{combined_report['experiment_name']}`",
        f"overall_ok: {combined_report['ok']}",
        f"manifest_hash: `{combined_report['manifest_hash']}`",
        "",
        "## Split Counts",
        "",
        "| split | jets | expected | balanced classes | ok |",
        "|---|---:|---:|---|---|",
    ]
    split_counts = split_report["split_summary"]["split_counts"]
    class_counts_by_split = split_report["split_summary"]["class_counts"]
    expected_counts = split_report["expected_split_sizes"]
    for split in SPLIT_ORDER:
        actual = int(split_counts[split])
        expected = int(expected_counts[split])
        expected_per_class = expected // len(LABEL_NAMES)
        balanced = all(
            int(class_counts_by_split[split][label_name]) == expected_per_class
            for label_name in LABEL_NAMES
        )
        ok = actual == expected and balanced
        lines.append(f"| {split} | {actual} | {expected} | {balanced} | {ok} |")

    lines.extend(
        [
            "",
            "## HLT Cache",
            "",
            "| split | jets | seed | HLT mean count | HLT hash | ok |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for split in SPLIT_ORDER:
        item = hlt_report["split_reports"][split]
        hlt_summary = item.get("hlt_constit_count_summary") or {}
        digest = str(item.get("hlt_content_hash") or "")
        digest_text = f"`{digest[:12]}...`" if digest else "missing"
        lines.append(
            f"| {split} | {item.get('n_jets')} | {item.get('seed')} | "
            f"{float(hlt_summary.get('mean', 0.0)):.4f} | {digest_text} | {item.get('ok')} |"
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
            "Jet identity separation is by `(file, entry)`, so file overlap across splits is expected.",
        ]
    )
    if combined_report.get("problems"):
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in combined_report["problems"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_split_report(manifest_path: Path) -> tuple[Any, dict[str, Any]]:
    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    split_audit = audit_split_manifest(manifest)
    summary = split_summary(manifest)
    size_problems = split_size_problems(manifest.split_sizes, summary["split_counts"])
    balance_problems = class_balance_problems(summary["class_counts"])
    problems = list(size_problems) + list(balance_problems)
    if not split_audit["ok"]:
        problems.append("base split manifest audit failed")
    report = {
        "ok": bool(split_audit["ok"] and not problems),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": f"{EXPERIMENT_STEP}:split_manifest",
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "expected_split_sizes": expected_split_sizes(),
        "manifest_declared_split_sizes": {split: int(manifest.split_sizes[split]) for split in SPLIT_ORDER},
        "split_summary": summary,
        "split_audit": split_audit,
        "problems": problems,
    }
    return manifest, report


def build_hlt_report(manifest: Any, manifest_path: Path, cache_dir: Path) -> dict[str, Any]:
    manifest_sha = manifest_hash(manifest)
    base_hlt_audit = audit_hlt_cache(manifest, cache_dir, splits=SPLIT_ORDER)
    split_reports: dict[str, Any] = {}
    expected_counts = expected_split_sizes()

    for split in SPLIT_ORDER:
        base_item = base_hlt_audit["split_reports"].get(split, {})
        base_ok = bool(base_item.get("ok"))
        try:
            item = read_cache_array_report(cache_dir, split)
            problems = hlt_cache_split_problems(
                item,
                base_problems=list(base_item.get("problems") or []),
                expected_size=expected_counts[split],
                expected_seed=DEFAULT_HLT_SEEDS[split],
                manifest_sha=manifest_sha,
            )
            item.update({"ok": bool(base_ok and not problems), "problems": problems})
        except Exception as exc:  # pragma: no cover - exercised by compute-side failures
            item = {
                "ok": False,
                "split": split,
                "problems": list(base_item.get("problems") or []) + [str(exc)],
                "n_jets": 0,
                "seed": None,
                "hlt_content_hash": None,
            }
        split_reports[split] = item

    distinct_hashes_ok = bool(base_hlt_audit.get("all_splits_have_distinct_content_hashes"))
    problems: list[str] = []
    if not base_hlt_audit["ok"]:
        problems.append("base fixed-HLT cache audit failed")
    if not distinct_hashes_ok:
        problems.append("fixed-HLT content hashes are not distinct across all splits")
    for split, item in split_reports.items():
        for problem in item.get("problems") or []:
            problems.append(f"{split}: {problem}")

    return {
        "ok": bool(base_hlt_audit["ok"] and distinct_hashes_ok and all(item["ok"] for item in split_reports.values())),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": f"{EXPERIMENT_STEP}:fixed_hlt_cache",
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "expected_split_sizes": expected_counts,
        "expected_hlt_params": fixed_hlt_params_dict(),
        "expected_hlt_seeds": {split: int(DEFAULT_HLT_SEEDS[split]) for split in SPLIT_ORDER},
        "base_audit": base_hlt_audit,
        "split_reports": split_reports,
        "all_splits_have_distinct_content_hashes": distinct_hashes_ok,
        "problems": problems,
    }


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    cache_dir = Path(args.hlt_cache_dir)
    output_dir = Path(args.output_dir)

    manifest, split_report = build_split_report(manifest_path)
    hlt_report = build_hlt_report(manifest, manifest_path, cache_dir)
    combined = {
        "ok": bool(split_report["ok"] and hlt_report["ok"]),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": EXPERIMENT_STEP,
        "manifest_path": str(manifest_path),
        "hlt_cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "manifest_hash": split_report["manifest_hash"],
        "expected_split_sizes": expected_split_sizes(),
        "audits": {
            "split_manifest": split_report,
            "hlt_cache": hlt_report,
        },
        "problems": list(split_report.get("problems") or []) + list(hlt_report.get("problems") or []),
    }

    split_path = output_dir / "split_audit_report.json"
    hlt_path = output_dir / "hlt_cache_audit_report.json"
    combined_path = output_dir / "crossarch_step2_audit_report.json"
    summary_path = output_dir / "crossarch_step2_audit_summary.md"
    write_json(split_path, split_report)
    write_json(hlt_path, hlt_report)
    write_json(combined_path, combined)
    write_summary(summary_path, combined)

    result = {
        "ok": bool(combined["ok"]),
        "output_dir": str(output_dir),
        "split_audit_report": str(split_path),
        "hlt_cache_audit_report": str(hlt_path),
        "audit_report": str(combined_path),
        "summary": str(summary_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

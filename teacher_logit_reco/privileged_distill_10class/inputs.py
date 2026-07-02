"""PD10 Step 2 split and fixed-HLT cache audit helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_cache import (
    DEFAULT_HLT_SEEDS,
    audit_hlt_cache,
    fixed_hlt_params_dict,
    fixed_hlt_params_from_strength,
    hash_arrays,
    load_cached_hlt_view,
    load_hlt_metadata,
)
from jetclass_fresh.jetclass_data import (
    LABEL_NAMES,
    audit_split_manifest,
    load_split_manifest,
    manifest_hash,
    split_summary,
)

from .config import (
    PD10_EXPERIMENT_NAME,
    PD10_HLT_DEGRADATION_STRENGTH,
    PD10_MANIFEST_SPLIT_ORDER,
    PD10_MANIFEST_SPLIT_SIZES,
    PD10_MANIFEST_STACK_SPLIT_SIZES,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
)


PD10_STEP2_EXPERIMENT_STEP = "pd10_step2_splits_hlt_cache"
PD10_STEP2_SPLIT_REPORT = "split_audit_report.json"
PD10_STEP2_HLT_REPORT = "hlt_cache_audit_report.json"
PD10_STEP2_AUDIT_REPORT = "pd10_step2_audit_report.json"
PD10_STEP2_AUDIT_SUMMARY = "pd10_step2_audit_summary.md"


def pd10_expected_split_sizes(expected_counts: Mapping[str, int] | None = None) -> dict[str, int]:
    """Return the three model-facing PD10 split sizes."""

    source = PD10_SPLIT_SIZES if expected_counts is None else expected_counts
    return {split: int(source[split]) for split in PD10_SPLIT_ORDER}


def pd10_manifest_split_sizes(
    expected_counts: Mapping[str, int] | None = None,
    placeholder_counts: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Return the five-way manifest sizes used to build PD10 inputs."""

    model_counts = pd10_expected_split_sizes(expected_counts)
    stack_counts = pd10_stack_placeholder_split_sizes(placeholder_counts)
    source = {**model_counts, **stack_counts}
    return {split: int(source[split]) for split in PD10_MANIFEST_SPLIT_ORDER}


def pd10_stack_placeholder_split_sizes(expected_counts: Mapping[str, int] | None = None) -> dict[str, int]:
    """Return the stack split placeholders required by the shared manifest format."""

    source = PD10_MANIFEST_STACK_SPLIT_SIZES if expected_counts is None else expected_counts
    return {split: int(source[split]) for split in PD10_MANIFEST_STACK_SPLIT_SIZES}


def pd10_hlt_params_dict() -> dict[str, float]:
    """Return the configured canonical PD10 fixed-HLT profile as JSON-safe floats."""

    return fixed_hlt_params_dict(fixed_hlt_params_from_strength(PD10_HLT_DEGRADATION_STRENGTH))


def split_size_problems(
    declared_sizes: Mapping[str, int],
    actual_counts: Mapping[str, int],
    expected_counts: Mapping[str, int] | None = None,
) -> list[str]:
    """Check only the model-facing PD10 splits against the 5M/1M/1M contract."""

    expected_counts = pd10_expected_split_sizes() if expected_counts is None else expected_counts
    problems: list[str] = []
    for split in PD10_SPLIT_ORDER:
        expected = int(expected_counts[split])
        declared = int(declared_sizes.get(split, -1))
        actual = int(actual_counts.get(split, -1))
        if declared != expected:
            problems.append(f"{split} declared size is {declared}, expected {expected}")
        if actual != expected:
            problems.append(f"{split} actual count is {actual}, expected {expected}")
    return problems


def placeholder_split_size_problems(
    declared_sizes: Mapping[str, int],
    actual_counts: Mapping[str, int],
    expected_counts: Mapping[str, int] | None = None,
) -> list[str]:
    """Check the explicit tiny stack placeholders in the five-way manifest."""

    expected_counts = pd10_stack_placeholder_split_sizes() if expected_counts is None else expected_counts
    problems: list[str] = []
    for split, expected_value in expected_counts.items():
        expected = int(expected_value)
        declared = int(declared_sizes.get(split, -1))
        actual = int(actual_counts.get(split, -1))
        if declared != expected:
            problems.append(f"{split} placeholder declared size is {declared}, expected {expected}")
        if actual != expected:
            problems.append(f"{split} placeholder actual count is {actual}, expected {expected}")
    return problems


def class_balance_problems(
    class_counts_by_split: Mapping[str, Mapping[str, int]],
    expected_counts: Mapping[str, int] | None = None,
) -> list[str]:
    """Require balanced 10-class composition for the PD10 train/val/test splits."""

    expected_counts = pd10_expected_split_sizes() if expected_counts is None else expected_counts
    problems: list[str] = []
    n_classes = len(LABEL_NAMES)
    for split in PD10_SPLIT_ORDER:
        expected_total = int(expected_counts[split])
        expected_per_class = expected_total // n_classes
        if expected_total % n_classes:
            problems.append(f"{split} expected size is not divisible by {n_classes}")
            continue
        counts = class_counts_by_split.get(split, {})
        for label_name in LABEL_NAMES:
            actual = int(counts.get(label_name, -1))
            if actual != expected_per_class:
                problems.append(f"{split}/{label_name} count is {actual}, expected {expected_per_class}")
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


def _subset_mapping(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: mapping[key] for key in keys if key in mapping}


def read_cache_array_report(cache_dir: Path, split: str) -> dict[str, Any]:
    """Read one cached HLT split and recompute the critical metadata hashes."""

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
    expected_params = pd10_hlt_params_dict()
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
        "hlt_degradation_strength": float(PD10_HLT_DEGRADATION_STRENGTH),
        "hlt_params": metadata.get("hlt_params"),
        "expected_hlt_params": expected_params,
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
        problems.append(
            "HLT params do not match configured PD10 fixed-HLT profile "
            f"(strength={PD10_HLT_DEGRADATION_STRENGTH:g})"
        )
    if item.get("source_manifest_hash") != manifest_sha:
        problems.append("source_manifest_hash does not match manifest hash")
    if not bool(item.get("content_hash_matches_metadata")):
        problems.append("recomputed HLT content hash does not match metadata")
    return problems


def build_split_report(
    manifest_path: Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_placeholder_sizes: Mapping[str, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    split_audit = audit_split_manifest(manifest)
    summary = split_summary(manifest)
    actual_counts = summary["split_counts"]
    class_counts_by_split = summary["class_counts"]
    expected_counts = pd10_expected_split_sizes(expected_split_sizes)
    placeholder_counts = pd10_stack_placeholder_split_sizes(expected_placeholder_sizes)
    manifest_expected_counts = pd10_manifest_split_sizes(expected_counts, placeholder_counts)
    size_problems = split_size_problems(manifest.split_sizes, actual_counts, expected_counts=expected_counts)
    placeholder_problems = placeholder_split_size_problems(
        manifest.split_sizes,
        actual_counts,
        expected_counts=placeholder_counts,
    )
    balance_problems = class_balance_problems(class_counts_by_split, expected_counts=expected_counts)
    problems = list(size_problems) + list(placeholder_problems) + list(balance_problems)
    if not split_audit["ok"]:
        problems.append("base split manifest audit failed")
    report = {
        "ok": bool(split_audit["ok"] and not problems),
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": f"{PD10_STEP2_EXPERIMENT_STEP}:split_manifest",
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "pd10_splits": list(PD10_SPLIT_ORDER),
        "expected_split_sizes": expected_counts,
        "expected_manifest_split_sizes": manifest_expected_counts,
        "placeholder_split_sizes": placeholder_counts,
        "manifest_declared_split_sizes": _subset_mapping(manifest.split_sizes, PD10_MANIFEST_SPLIT_ORDER),
        "pd10_split_counts": _subset_mapping(actual_counts, PD10_SPLIT_ORDER),
        "pd10_class_counts": _subset_mapping(class_counts_by_split, PD10_SPLIT_ORDER),
        "split_summary": summary,
        "split_audit": split_audit,
        "problems": problems,
    }
    return manifest, report


def build_hlt_report(
    manifest: Any,
    manifest_path: Path,
    cache_dir: Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    manifest_sha = manifest_hash(manifest)
    expected_params = pd10_hlt_params_dict()
    base_hlt_audit = audit_hlt_cache(
        manifest,
        cache_dir,
        splits=PD10_SPLIT_ORDER,
        expected_params=expected_params,
    )
    split_reports: dict[str, Any] = {}
    expected_counts = pd10_expected_split_sizes(expected_split_sizes)

    for split in PD10_SPLIT_ORDER:
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

    hashes = [item.get("hlt_content_hash") for item in split_reports.values() if item.get("hlt_content_hash")]
    distinct_hashes_ok = len(hashes) == len(PD10_SPLIT_ORDER) and len(hashes) == len(set(hashes))
    problems: list[str] = []
    if not base_hlt_audit["ok"]:
        problems.append("base fixed-HLT cache audit failed")
    if not distinct_hashes_ok:
        problems.append("fixed-HLT content hashes are not distinct across PD10 splits")
    for split, item in split_reports.items():
        for problem in item.get("problems") or []:
            problems.append(f"{split}: {problem}")

    return {
        "ok": bool(base_hlt_audit["ok"] and distinct_hashes_ok and all(item["ok"] for item in split_reports.values())),
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": f"{PD10_STEP2_EXPERIMENT_STEP}:fixed_hlt_cache",
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "pd10_splits": list(PD10_SPLIT_ORDER),
        "expected_split_sizes": expected_counts,
        "hlt_degradation_strength": float(PD10_HLT_DEGRADATION_STRENGTH),
        "expected_hlt_params": expected_params,
        "expected_hlt_seeds": {split: int(DEFAULT_HLT_SEEDS[split]) for split in PD10_SPLIT_ORDER},
        "base_audit": base_hlt_audit,
        "split_reports": split_reports,
        "all_splits_have_distinct_content_hashes": distinct_hashes_ok,
        "problems": problems,
    }


def build_pd10_step2_audit_report(
    manifest_path: Path,
    hlt_cache_dir: Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_placeholder_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    expected_counts = pd10_expected_split_sizes(expected_split_sizes)
    placeholder_counts = pd10_stack_placeholder_split_sizes(expected_placeholder_sizes)
    manifest, split_report = build_split_report(
        manifest_path,
        expected_split_sizes=expected_counts,
        expected_placeholder_sizes=placeholder_counts,
    )
    hlt_report = build_hlt_report(
        manifest,
        manifest_path,
        hlt_cache_dir,
        expected_split_sizes=expected_counts,
    )
    return {
        "ok": bool(split_report["ok"] and hlt_report["ok"]),
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP2_EXPERIMENT_STEP,
        "manifest_path": str(manifest_path),
        "hlt_cache_dir": str(hlt_cache_dir),
        "manifest_hash": split_report["manifest_hash"],
        "pd10_splits": list(PD10_SPLIT_ORDER),
        "expected_split_sizes": expected_counts,
        "expected_manifest_split_sizes": pd10_manifest_split_sizes(expected_counts, placeholder_counts),
        "hlt_degradation_strength": float(PD10_HLT_DEGRADATION_STRENGTH),
        "expected_hlt_params": pd10_hlt_params_dict(),
        "audits": {
            "split_manifest": split_report,
            "hlt_cache": hlt_report,
        },
        "problems": list(split_report.get("problems") or []) + list(hlt_report.get("problems") or []),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_summary(path: Path, combined_report: Mapping[str, Any]) -> None:
    split_report = combined_report["audits"]["split_manifest"]
    hlt_report = combined_report["audits"]["hlt_cache"]
    lines = [
        "# PD10 Step 2 Split And HLT Cache Audit",
        "",
        f"experiment_name: `{combined_report['experiment_name']}`",
        f"overall_ok: {combined_report['ok']}",
        f"manifest_hash: `{combined_report['manifest_hash']}`",
        f"hlt_degradation_strength: {combined_report['hlt_degradation_strength']}",
        "",
        "## Split Counts",
        "",
        "| split | jets | expected | per class | balanced classes | ok |",
        "|---|---:|---:|---:|---|---|",
    ]
    split_counts = split_report["split_summary"]["split_counts"]
    class_counts_by_split = split_report["split_summary"]["class_counts"]
    expected_counts = split_report["expected_split_sizes"]
    for split in PD10_SPLIT_ORDER:
        actual = int(split_counts[split])
        expected = int(expected_counts[split])
        expected_per_class = expected // len(LABEL_NAMES)
        balanced = all(
            int(class_counts_by_split[split][label_name]) == expected_per_class
            for label_name in LABEL_NAMES
        )
        ok = actual == expected and balanced
        lines.append(f"| {split} | {actual} | {expected} | {expected_per_class} | {balanced} | {ok} |")

    placeholders = split_report["placeholder_split_sizes"]
    lines.extend(["", "## Manifest Placeholders", "", "| split | jets | expected |", "|---|---:|---:|"])
    for split, expected in placeholders.items():
        actual = int(split_counts.get(split, -1))
        lines.append(f"| {split} | {actual} | {expected} |")

    lines.extend(
        [
            "",
            "## HLT Cache",
            "",
            "| split | jets | seed | HLT mean count | HLT hash | ok |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for split in PD10_SPLIT_ORDER:
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


def write_pd10_step2_audit_reports(
    manifest_path: Path,
    hlt_cache_dir: Path,
    output_dir: Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_placeholder_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    combined = build_pd10_step2_audit_report(
        manifest_path,
        hlt_cache_dir,
        expected_split_sizes=expected_split_sizes,
        expected_placeholder_sizes=expected_placeholder_sizes,
    )
    split_report = combined["audits"]["split_manifest"]
    hlt_report = combined["audits"]["hlt_cache"]

    split_path = output_dir / PD10_STEP2_SPLIT_REPORT
    hlt_path = output_dir / PD10_STEP2_HLT_REPORT
    combined_path = output_dir / PD10_STEP2_AUDIT_REPORT
    summary_path = output_dir / PD10_STEP2_AUDIT_SUMMARY
    write_json(split_path, split_report)
    write_json(hlt_path, hlt_report)
    write_json(combined_path, combined)
    write_summary(summary_path, combined)

    return {
        "ok": bool(combined["ok"]),
        "output_dir": str(output_dir),
        "split_audit_report": str(split_path),
        "hlt_cache_audit_report": str(hlt_path),
        "audit_report": str(combined_path),
        "summary": str(summary_path),
    }


__all__ = [
    "PD10_STEP2_AUDIT_REPORT",
    "PD10_STEP2_AUDIT_SUMMARY",
    "PD10_STEP2_EXPERIMENT_STEP",
    "PD10_STEP2_HLT_REPORT",
    "PD10_STEP2_SPLIT_REPORT",
    "build_hlt_report",
    "build_pd10_step2_audit_report",
    "build_split_report",
    "class_balance_problems",
    "class_counts",
    "count_summary",
    "hlt_cache_split_problems",
    "pd10_expected_split_sizes",
    "pd10_hlt_params_dict",
    "pd10_manifest_split_sizes",
    "pd10_stack_placeholder_split_sizes",
    "placeholder_split_size_problems",
    "read_cache_array_report",
    "split_size_problems",
    "write_json",
    "write_pd10_step2_audit_reports",
    "write_summary",
]

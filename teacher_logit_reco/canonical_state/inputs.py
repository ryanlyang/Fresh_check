"""Step 1 split and HLT-cache audits for canonical jet-state experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_cache import (
    DEFAULT_HLT_SEEDS,
    audit_hlt_cache,
    hash_arrays,
    load_cached_hlt_view,
    load_hlt_metadata,
    normalize_hlt_profile,
)
from jetclass_fresh.jetclass_data import (
    LABEL_NAMES,
    audit_split_manifest,
    load_split_manifest,
    manifest_hash,
    split_summary,
)

from .config import (
    CANONICAL_STATE_EXPERIMENT_NAME,
    CANONICAL_STATE_EXPERIMENT_STEP,
    CANONICAL_STATE_HLT_DEGRADATION_STRENGTH,
    CANONICAL_STATE_HLT_PROFILE,
    CANONICAL_STATE_HLT_PROFILE_VERSION,
    CANONICAL_STATE_HLT_AUDIT_REPORT,
    CANONICAL_STATE_INPUTS_CONTRACT,
    CANONICAL_STATE_SPLIT_AUDIT_REPORT,
    CANONICAL_STATE_SPLIT_ORDER,
    CANONICAL_STATE_STEP1_AUDIT_REPORT,
    CANONICAL_STATE_STEP1_AUDIT_SUMMARY,
    canonical_state_hlt_params_dict,
    canonical_state_split_sizes,
)


def _subset_mapping(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: mapping[key] for key in keys if key in mapping}


def _json_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = {name: 0 for name in LABEL_NAMES}
    for label in np.asarray(labels, dtype=np.int64):
        counts[LABEL_NAMES[int(label)]] += 1
    return counts


def class_balance_problems(
    class_counts_by_split: Mapping[str, Mapping[str, int]],
    expected_counts: Mapping[str, int] | None = None,
) -> list[str]:
    expected_counts = canonical_state_split_sizes(expected_counts)
    problems: list[str] = []
    n_classes = len(LABEL_NAMES)
    for split in CANONICAL_STATE_SPLIT_ORDER:
        expected_total = int(expected_counts[split])
        if expected_total % n_classes:
            problems.append(f"{split} expected size is not divisible by {n_classes}")
            continue
        expected_per_class = expected_total // n_classes
        counts = class_counts_by_split.get(split, {})
        for label_name in LABEL_NAMES:
            actual = int(counts.get(label_name, -1))
            if actual != expected_per_class:
                problems.append(f"{split}/{label_name} count is {actual}, expected {expected_per_class}")
    return problems


def split_size_problems(
    declared_sizes: Mapping[str, int],
    actual_counts: Mapping[str, int],
    expected_counts: Mapping[str, int] | None = None,
) -> list[str]:
    expected_counts = canonical_state_split_sizes(expected_counts)
    problems: list[str] = []
    for split in CANONICAL_STATE_SPLIT_ORDER:
        expected = int(expected_counts[split])
        declared = int(declared_sizes.get(split, -1))
        actual = int(actual_counts.get(split, -1))
        if declared != expected:
            problems.append(f"{split} declared size is {declared}, expected {expected}")
        if actual != expected:
            problems.append(f"{split} actual count is {actual}, expected {expected}")
    return problems


def _cache_array_hash(cache_dir: Path, split: str) -> dict[str, Any]:
    array_path = cache_dir / f"{split}_fixed_hlt.npz"
    with np.load(array_path, allow_pickle=False) as data:
        file_indices = data["jet_file_indices"].astype(np.int32, copy=False)
        entries = data["jet_entries"]
        content_hash = hash_arrays(
            {
                "tokens": data["tokens"],
                "mask": data["mask"],
                "labels": data["labels"],
                "jet_file_indices": file_indices,
                "jet_entries": entries,
            }
        )
        return {
            "array_path": str(array_path),
            "array_keys": sorted(data.files),
            "tokens_shape": list(data["tokens"].shape),
            "mask_shape": list(data["mask"].shape),
            "labels_shape": list(data["labels"].shape),
            "hlt_content_hash": content_hash,
        }


def build_split_report(
    manifest_path: Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    base_audit = audit_split_manifest(manifest)
    summary = split_summary(manifest)
    actual_counts = summary["split_counts"]
    class_counts_by_split = summary["class_counts"]
    expected_counts = canonical_state_split_sizes(expected_split_sizes)
    problems = (
        split_size_problems(manifest.split_sizes, actual_counts, expected_counts=expected_counts)
        + class_balance_problems(class_counts_by_split, expected_counts=expected_counts)
    )
    if not bool(base_audit.get("ok")):
        problems.append("base split manifest audit failed")
    return manifest, {
        "ok": bool(base_audit.get("ok") and not problems),
        "experiment_name": CANONICAL_STATE_EXPERIMENT_NAME,
        "experiment_step": f"{CANONICAL_STATE_EXPERIMENT_STEP}:split_manifest",
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "split_order": list(CANONICAL_STATE_SPLIT_ORDER),
        "label_names": list(LABEL_NAMES),
        "expected_split_sizes": expected_counts,
        "manifest_declared_split_sizes": _subset_mapping(manifest.split_sizes, CANONICAL_STATE_SPLIT_ORDER),
        "split_counts": _subset_mapping(actual_counts, CANONICAL_STATE_SPLIT_ORDER),
        "class_counts": _subset_mapping(class_counts_by_split, CANONICAL_STATE_SPLIT_ORDER),
        "split_summary": summary,
        "split_audit": base_audit,
        "problems": problems,
    }


def build_hlt_cache_report(
    manifest: Any,
    manifest_path: Path,
    hlt_cache_dir: Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    manifest_sha = manifest_hash(manifest)
    expected_counts = canonical_state_split_sizes(expected_split_sizes)
    expected_params = canonical_state_hlt_params_dict()
    base_audit = audit_hlt_cache(
        manifest,
        hlt_cache_dir,
        splits=CANONICAL_STATE_SPLIT_ORDER,
        expected_params=expected_params,
        expected_hlt_profile=CANONICAL_STATE_HLT_PROFILE,
        expected_hlt_profile_version=CANONICAL_STATE_HLT_PROFILE_VERSION,
        expected_hlt_degradation_strength=CANONICAL_STATE_HLT_DEGRADATION_STRENGTH,
    )
    split_reports: dict[str, Any] = {}
    problems: list[str] = []
    for split in CANONICAL_STATE_SPLIT_ORDER:
        base_item = dict(base_audit.get("split_reports", {}).get(split, {}))
        metadata_path = hlt_cache_dir / f"{split}_fixed_hlt_metadata.json"
        try:
            metadata = load_hlt_metadata(hlt_cache_dir, split)
            view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True)
            array_report = _cache_array_hash(hlt_cache_dir, split)
            actual_profile = normalize_hlt_profile(metadata.get("hlt_profile"))
            item = {
                **array_report,
                "ok": True,
                "split": split,
                "metadata_path": str(metadata_path),
                "metadata_sha256": _json_file_sha256(metadata_path),
                "n_jets": int(view.tokens.shape[0]),
                "class_counts": class_counts(view.labels),
                "seed": int(metadata.get("seed", -1)),
                "expected_seed": int(DEFAULT_HLT_SEEDS[split]),
                "source_manifest_hash": metadata.get("source_manifest_hash"),
                "jet_identity_hash": metadata.get("jet_identity_hash"),
                "source_content_hash": metadata.get("source_content_hash"),
                "metadata_hlt_content_hash": metadata.get("hlt_content_hash"),
                "hlt_content_hash": array_report["hlt_content_hash"],
                "content_hash_matches_metadata": array_report["hlt_content_hash"] == metadata.get("hlt_content_hash"),
                "hlt_params": metadata.get("hlt_params"),
                "expected_hlt_params": expected_params,
                "hlt_profile": actual_profile,
                "expected_hlt_profile": CANONICAL_STATE_HLT_PROFILE,
                "hlt_profile_version": str(metadata.get("hlt_profile_version") or ""),
                "expected_hlt_profile_version": CANONICAL_STATE_HLT_PROFILE_VERSION,
                "hlt_degradation_strength": metadata.get("hlt_degradation_strength"),
                "expected_hlt_degradation_strength": float(CANONICAL_STATE_HLT_DEGRADATION_STRENGTH),
                "hlt_diagnostics_summary": metadata.get("hlt_diagnostics_summary"),
            }
            item_problems = list(base_item.get("problems") or [])
            if int(item["n_jets"]) != int(expected_counts[split]):
                item_problems.append(f"n_jets is {item['n_jets']}, expected {expected_counts[split]}")
            if int(item["seed"]) != int(item["expected_seed"]):
                item_problems.append(f"seed is {item['seed']}, expected {item['expected_seed']}")
            if item["source_manifest_hash"] != manifest_sha:
                item_problems.append("source_manifest_hash does not match manifest hash")
            if item["hlt_profile"] != CANONICAL_STATE_HLT_PROFILE:
                item_problems.append(
                    f"HLT profile is {item['hlt_profile']!r}, expected {CANONICAL_STATE_HLT_PROFILE!r}"
                )
            if item["hlt_profile_version"] != CANONICAL_STATE_HLT_PROFILE_VERSION:
                item_problems.append(
                    f"HLT profile version is {item['hlt_profile_version']!r}, "
                    f"expected {CANONICAL_STATE_HLT_PROFILE_VERSION!r}"
                )
            actual_strength = item.get("hlt_degradation_strength")
            if (
                actual_strength is None
                or abs(float(actual_strength) - float(CANONICAL_STATE_HLT_DEGRADATION_STRENGTH)) > 1.0e-12
            ):
                item_problems.append(
                    f"HLT degradation strength is {actual_strength}, "
                    f"expected {CANONICAL_STATE_HLT_DEGRADATION_STRENGTH:g}"
                )
            if item["hlt_params"] != expected_params:
                item_problems.append(
                    "HLT params do not match CMS-JS HLT v2 profile "
                    f"(profile={CANONICAL_STATE_HLT_PROFILE}, "
                    f"strength={CANONICAL_STATE_HLT_DEGRADATION_STRENGTH:g})"
                )
            if not bool(item["content_hash_matches_metadata"]):
                item_problems.append("recomputed HLT content hash does not match metadata")
            item["ok"] = bool(base_item.get("ok", True) and not item_problems)
            item["problems"] = item_problems
        except Exception as exc:  # pragma: no cover - compute-side failures
            item = {
                "ok": False,
                "split": split,
                "metadata_path": str(metadata_path),
                "problems": [str(exc)],
                "n_jets": 0,
            }
        split_reports[split] = item
        for problem in item.get("problems") or []:
            problems.append(f"{split}: {problem}")
    if not bool(base_audit.get("ok")):
        problems.append("base fixed-HLT cache audit failed")
    return {
        "ok": bool(not problems and all(bool(item.get("ok")) for item in split_reports.values())),
        "experiment_name": CANONICAL_STATE_EXPERIMENT_NAME,
        "experiment_step": f"{CANONICAL_STATE_EXPERIMENT_STEP}:hlt_cache",
        "cache_dir": str(hlt_cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "split_order": list(CANONICAL_STATE_SPLIT_ORDER),
        "expected_split_sizes": expected_counts,
        "hlt_profile": CANONICAL_STATE_HLT_PROFILE,
        "hlt_profile_version": CANONICAL_STATE_HLT_PROFILE_VERSION,
        "hlt_degradation_strength": float(CANONICAL_STATE_HLT_DEGRADATION_STRENGTH),
        "expected_hlt_params": expected_params,
        "split_reports": split_reports,
        "base_audit": base_audit,
        "problems": problems,
    }


def build_canonical_state_step1_input_audit_report(
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    hlt_cache_dir = Path(hlt_cache_dir)
    manifest, split_report = build_split_report(manifest_path, expected_split_sizes=expected_split_sizes)
    hlt_report = build_hlt_cache_report(
        manifest,
        manifest_path,
        hlt_cache_dir,
        expected_split_sizes=expected_split_sizes,
    )
    problems: list[str] = []
    for name, report in (("split_manifest", split_report), ("hlt_cache", hlt_report)):
        for problem in report.get("problems") or []:
            problems.append(f"{name}: {problem}")
    expected_counts = canonical_state_split_sizes(expected_split_sizes)
    return {
        "ok": bool(split_report.get("ok") and hlt_report.get("ok") and not problems),
        "contract": CANONICAL_STATE_INPUTS_CONTRACT,
        "experiment_name": CANONICAL_STATE_EXPERIMENT_NAME,
        "experiment_step": CANONICAL_STATE_EXPERIMENT_STEP,
        "manifest_path": str(manifest_path),
        "hlt_cache_dir": str(hlt_cache_dir),
        "manifest_hash": split_report.get("manifest_hash"),
        "split_order": list(CANONICAL_STATE_SPLIT_ORDER),
        "label_names": list(LABEL_NAMES),
        "expected_split_sizes": expected_counts,
        "hlt_profile": CANONICAL_STATE_HLT_PROFILE,
        "hlt_profile_version": CANONICAL_STATE_HLT_PROFILE_VERSION,
        "hlt_degradation_strength": float(CANONICAL_STATE_HLT_DEGRADATION_STRENGTH),
        "expected_hlt_params": canonical_state_hlt_params_dict(),
        "audits": {
            "split_manifest": split_report,
            "hlt_cache": hlt_report,
        },
        "problems": problems,
    }


def require_canonical_state_step1_input_contract(
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return the Step 1 audit report, or fail loudly on contract drift."""

    report = build_canonical_state_step1_input_audit_report(
        manifest_path,
        hlt_cache_dir,
        expected_split_sizes=expected_split_sizes,
    )
    if not bool(report.get("ok")):
        problems = report.get("problems") or ["unknown Step 1 input contract failure"]
        detail = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"Canonical state Step 1 input contract failed:\n{detail}")
    return report


def _audit_summary_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Canonical State Step 1 Input Audit",
        "",
        f"- ok: {bool(report.get('ok'))}",
        f"- contract: `{report.get('contract')}`",
        f"- manifest: `{report.get('manifest_path')}`",
        f"- hlt_cache_dir: `{report.get('hlt_cache_dir')}`",
        f"- manifest_hash: `{report.get('manifest_hash')}`",
        f"- hlt_profile: `{report.get('hlt_profile')}`",
        f"- hlt_profile_version: `{report.get('hlt_profile_version')}`",
        f"- hlt_degradation_strength: `{report.get('hlt_degradation_strength')}`",
        "",
        "## Split Sizes",
        "",
    ]
    split_report = dict(report.get("audits", {}).get("split_manifest", {}))
    split_counts = dict(split_report.get("split_counts") or {})
    expected = dict(report.get("expected_split_sizes") or {})
    for split in CANONICAL_STATE_SPLIT_ORDER:
        lines.append(f"- {split}: {split_counts.get(split)} / expected {expected.get(split)}")
    hlt_report = dict(report.get("audits", {}).get("hlt_cache", {}))
    hlt_splits = dict(hlt_report.get("split_reports") or {})
    lines.extend(["", "## HLT Cache", ""])
    for split in CANONICAL_STATE_SPLIT_ORDER:
        item = dict(hlt_splits.get(split) or {})
        lines.append(
            f"- {split}: ok={bool(item.get('ok'))}, "
            f"n_jets={item.get('n_jets')}, "
            f"content_hash=`{item.get('hlt_content_hash')}`"
        )
    problems = list(report.get("problems") or [])
    if problems:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in problems)
    return "\n".join(lines) + "\n"


def write_canonical_state_step1_input_audit_reports(
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_canonical_state_step1_input_audit_report(
        manifest_path,
        hlt_cache_dir,
        expected_split_sizes=expected_split_sizes,
    )
    paths = {
        "audit_report": output_dir / CANONICAL_STATE_STEP1_AUDIT_REPORT,
        "summary": output_dir / CANONICAL_STATE_STEP1_AUDIT_SUMMARY,
        "split_audit_report": output_dir / CANONICAL_STATE_SPLIT_AUDIT_REPORT,
        "hlt_audit_report": output_dir / CANONICAL_STATE_HLT_AUDIT_REPORT,
    }
    paths["audit_report"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["summary"].write_text(_audit_summary_markdown(report), encoding="utf-8")
    paths["split_audit_report"].write_text(
        json.dumps(report["audits"]["split_manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["hlt_audit_report"].write_text(
        json.dumps(report["audits"]["hlt_cache"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": bool(report["ok"]),
        "output_dir": str(output_dir),
        "audit_report": str(paths["audit_report"]),
        "summary": str(paths["summary"]),
        "split_audit_report": str(paths["split_audit_report"]),
        "hlt_audit_report": str(paths["hlt_audit_report"]),
        "problems": report.get("problems", []),
    }

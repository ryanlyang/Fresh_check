"""PDV3 Step 1 split, HLT0.2 cache, and paired-offline cache audits."""

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
    jet_identity_hash,
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
from teacher_logit_reco.architecture_view_part import load_cached_offline_view

from .config import (
    PDV3_EXPERIMENT_NAME,
    PDV3_EXPERIMENT_STEP,
    PDV3_HLT_AUDIT_REPORT,
    PDV3_HLT_DEGRADATION_STRENGTH,
    PDV3_INPUTS_CONTRACT,
    PDV3_MANIFEST_SPLIT_ORDER,
    PDV3_MODEL_SPLIT_ORDER,
    PDV3_OFFLINE_AUDIT_REPORT,
    PDV3_SPLIT_AUDIT_REPORT,
    PDV3_STEP1_AUDIT_REPORT,
    PDV3_STEP1_AUDIT_SUMMARY,
    pdv3_manifest_split_sizes,
    pdv3_model_split_sizes,
    pdv3_stack_placeholder_split_sizes,
)


def pdv3_hlt_params_dict() -> dict[str, float]:
    return fixed_hlt_params_dict(fixed_hlt_params_from_strength(PDV3_HLT_DEGRADATION_STRENGTH))


def _subset_mapping(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: mapping[key] for key in keys if key in mapping}


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = {name: 0 for name in LABEL_NAMES}
    for label in np.asarray(labels, dtype=np.int64):
        counts[LABEL_NAMES[int(label)]] += 1
    return counts


def class_balance_problems(
    class_counts_by_split: Mapping[str, Mapping[str, int]],
    expected_counts: Mapping[str, int] | None = None,
) -> list[str]:
    expected_counts = pdv3_model_split_sizes(expected_counts)
    problems: list[str] = []
    n_classes = len(LABEL_NAMES)
    for split in PDV3_MODEL_SPLIT_ORDER:
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


def split_size_problems(
    declared_sizes: Mapping[str, int],
    actual_counts: Mapping[str, int],
    expected_counts: Mapping[str, int] | None = None,
) -> list[str]:
    expected_counts = pdv3_model_split_sizes(expected_counts)
    problems: list[str] = []
    for split in PDV3_MODEL_SPLIT_ORDER:
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
    expected_counts = pdv3_stack_placeholder_split_sizes(expected_counts)
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


def _cache_array_hash(cache_dir: Path, split: str, *, kind: str) -> dict[str, Any]:
    if kind == "hlt":
        array_path = cache_dir / f"{split}_fixed_hlt.npz"
        token_key = "tokens"
        content_name = "hlt_content_hash"
    elif kind == "offline":
        array_path = cache_dir / f"{split}_offline.npz"
        token_key = "tokens"
        content_name = "offline_content_hash"
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown cache kind {kind!r}")
    with np.load(array_path, allow_pickle=False) as data:
        file_indices = data["jet_file_indices"].astype(np.int32, copy=False)
        entries = data["jet_entries"]
        content_hash = hash_arrays(
            {
                token_key: data["tokens"],
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
            content_name: content_hash,
        }


def build_split_report(
    manifest_path: Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_placeholder_sizes: Mapping[str, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    base_audit = audit_split_manifest(manifest)
    summary = split_summary(manifest)
    actual_counts = summary["split_counts"]
    class_counts_by_split = summary["class_counts"]
    expected_counts = pdv3_model_split_sizes(expected_split_sizes)
    placeholder_counts = pdv3_stack_placeholder_split_sizes(expected_placeholder_sizes)
    problems = (
        split_size_problems(manifest.split_sizes, actual_counts, expected_counts=expected_counts)
        + placeholder_split_size_problems(
            manifest.split_sizes,
            actual_counts,
            expected_counts=placeholder_counts,
        )
        + class_balance_problems(class_counts_by_split, expected_counts=expected_counts)
    )
    if not bool(base_audit.get("ok")):
        problems.append("base split manifest audit failed")
    report = {
        "ok": bool(base_audit.get("ok") and not problems),
        "experiment_name": PDV3_EXPERIMENT_NAME,
        "experiment_step": f"{PDV3_EXPERIMENT_STEP}:split_manifest",
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "model_splits": list(PDV3_MODEL_SPLIT_ORDER),
        "expected_split_sizes": expected_counts,
        "expected_manifest_split_sizes": pdv3_manifest_split_sizes(expected_counts, placeholder_counts),
        "manifest_declared_split_sizes": _subset_mapping(manifest.split_sizes, PDV3_MANIFEST_SPLIT_ORDER),
        "model_split_counts": _subset_mapping(actual_counts, PDV3_MODEL_SPLIT_ORDER),
        "model_class_counts": _subset_mapping(class_counts_by_split, PDV3_MODEL_SPLIT_ORDER),
        "split_summary": summary,
        "split_audit": base_audit,
        "problems": problems,
    }
    return manifest, report


def build_hlt_report(
    manifest: Any,
    manifest_path: Path,
    hlt_cache_dir: Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    manifest_sha = manifest_hash(manifest)
    expected_counts = pdv3_model_split_sizes(expected_split_sizes)
    expected_params = pdv3_hlt_params_dict()
    base_audit = audit_hlt_cache(
        manifest,
        hlt_cache_dir,
        splits=PDV3_MODEL_SPLIT_ORDER,
        expected_params=expected_params,
    )
    split_reports: dict[str, Any] = {}
    problems: list[str] = []
    for split in PDV3_MODEL_SPLIT_ORDER:
        base_item = dict(base_audit.get("split_reports", {}).get(split, {}))
        try:
            metadata = load_hlt_metadata(hlt_cache_dir, split)
            view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True)
            array_report = _cache_array_hash(hlt_cache_dir, split, kind="hlt")
            item = {
                **array_report,
                "ok": True,
                "split": split,
                "metadata_path": str(hlt_cache_dir / f"{split}_fixed_hlt_metadata.json"),
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
                "hlt_diagnostics_summary": metadata.get("hlt_diagnostics_summary"),
            }
            item_problems = list(base_item.get("problems") or [])
            if int(item["n_jets"]) != int(expected_counts[split]):
                item_problems.append(f"n_jets is {item['n_jets']}, expected {expected_counts[split]}")
            if int(item["seed"]) != int(item["expected_seed"]):
                item_problems.append(f"seed is {item['seed']}, expected {item['expected_seed']}")
            if item["source_manifest_hash"] != manifest_sha:
                item_problems.append("source_manifest_hash does not match manifest hash")
            if item["hlt_params"] != expected_params:
                item_problems.append(
                    "HLT params do not match PDV3 HLT0.2 profile "
                    f"(strength={PDV3_HLT_DEGRADATION_STRENGTH:g})"
                )
            if not bool(item["content_hash_matches_metadata"]):
                item_problems.append("recomputed HLT content hash does not match metadata")
            item["ok"] = bool(base_item.get("ok", True) and not item_problems)
            item["problems"] = item_problems
        except Exception as exc:  # pragma: no cover - compute-side failures
            item = {"ok": False, "split": split, "problems": [str(exc)], "n_jets": 0}
        split_reports[split] = item
        for problem in item.get("problems") or []:
            problems.append(f"{split}: {problem}")
    if not bool(base_audit.get("ok")):
        problems.append("base fixed-HLT cache audit failed")
    return {
        "ok": bool(not problems and all(bool(item.get("ok")) for item in split_reports.values())),
        "experiment_name": PDV3_EXPERIMENT_NAME,
        "experiment_step": f"{PDV3_EXPERIMENT_STEP}:hlt_cache",
        "cache_dir": str(hlt_cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "model_splits": list(PDV3_MODEL_SPLIT_ORDER),
        "expected_split_sizes": expected_counts,
        "hlt_degradation_strength": float(PDV3_HLT_DEGRADATION_STRENGTH),
        "expected_hlt_params": expected_params,
        "split_reports": split_reports,
        "base_audit": base_audit,
        "problems": problems,
    }


def build_offline_report(
    manifest: Any,
    manifest_path: Path,
    offline_cache_dir: Path,
    hlt_report: Mapping[str, Any],
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    manifest_sha = manifest_hash(manifest)
    expected_counts = pdv3_model_split_sizes(expected_split_sizes)
    split_reports: dict[str, Any] = {}
    problems: list[str] = []
    for split in PDV3_MODEL_SPLIT_ORDER:
        try:
            view = load_cached_offline_view(offline_cache_dir, split, verify_hash=True)
            array_report = _cache_array_hash(offline_cache_dir, split, kind="offline")
            metadata = dict(view.metadata)
            hlt_item = dict(hlt_report.get("split_reports", {}).get(split, {}))
            item = {
                **array_report,
                "ok": True,
                "split": split,
                "metadata_path": str(offline_cache_dir / f"{split}_offline_metadata.json"),
                "n_jets": int(view.tokens.shape[0]),
                "class_counts": class_counts(view.labels),
                "source_manifest_hash": metadata.get("source_manifest_hash"),
                "jet_identity_hash": metadata.get("jet_identity_hash"),
                "metadata_offline_content_hash": metadata.get("offline_content_hash"),
                "offline_content_hash": array_report["offline_content_hash"],
                "content_hash_matches_metadata": (
                    array_report["offline_content_hash"] == metadata.get("offline_content_hash")
                ),
                "paired_hlt_jet_identity_hash": hlt_item.get("jet_identity_hash"),
                "paired_hlt_labels_shape": hlt_item.get("labels_shape"),
            }
            item_problems: list[str] = []
            if int(item["n_jets"]) != int(expected_counts[split]):
                item_problems.append(f"n_jets is {item['n_jets']}, expected {expected_counts[split]}")
            if item["source_manifest_hash"] != manifest_sha:
                item_problems.append("source_manifest_hash does not match manifest hash")
            if not bool(item["content_hash_matches_metadata"]):
                item_problems.append("recomputed offline content hash does not match metadata")
            if item["jet_identity_hash"] != hlt_item.get("jet_identity_hash"):
                item_problems.append("offline jet_identity_hash does not match paired HLT cache")
            if list(item["labels_shape"]) != list(hlt_item.get("labels_shape") or []):
                item_problems.append("offline labels shape does not match paired HLT cache")
            item["ok"] = not item_problems
            item["problems"] = item_problems
        except Exception as exc:  # pragma: no cover - compute-side failures
            item = {"ok": False, "split": split, "problems": [str(exc)], "n_jets": 0}
        split_reports[split] = item
        for problem in item.get("problems") or []:
            problems.append(f"{split}: {problem}")
    return {
        "ok": bool(not problems and all(bool(item.get("ok")) for item in split_reports.values())),
        "experiment_name": PDV3_EXPERIMENT_NAME,
        "experiment_step": f"{PDV3_EXPERIMENT_STEP}:offline_cache",
        "cache_dir": str(offline_cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_sha,
        "model_splits": list(PDV3_MODEL_SPLIT_ORDER),
        "expected_split_sizes": expected_counts,
        "split_reports": split_reports,
        "problems": problems,
    }


def build_pdv3_step1_input_audit_report(
    manifest_path: Path,
    hlt_cache_dir: Path,
    offline_cache_dir: Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_placeholder_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    expected_counts = pdv3_model_split_sizes(expected_split_sizes)
    placeholder_counts = pdv3_stack_placeholder_split_sizes(expected_placeholder_sizes)
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
    offline_report = build_offline_report(
        manifest,
        manifest_path,
        offline_cache_dir,
        hlt_report,
        expected_split_sizes=expected_counts,
    )
    problems = (
        list(split_report.get("problems") or [])
        + list(hlt_report.get("problems") or [])
        + list(offline_report.get("problems") or [])
    )
    return {
        "ok": bool(split_report["ok"] and hlt_report["ok"] and offline_report["ok"] and not problems),
        "contract": PDV3_INPUTS_CONTRACT,
        "experiment_name": PDV3_EXPERIMENT_NAME,
        "experiment_step": PDV3_EXPERIMENT_STEP,
        "manifest_path": str(manifest_path),
        "hlt_cache_dir": str(hlt_cache_dir),
        "offline_cache_dir": str(offline_cache_dir),
        "manifest_hash": split_report["manifest_hash"],
        "model_splits": list(PDV3_MODEL_SPLIT_ORDER),
        "expected_split_sizes": expected_counts,
        "expected_manifest_split_sizes": pdv3_manifest_split_sizes(expected_counts, placeholder_counts),
        "hlt_degradation_strength": float(PDV3_HLT_DEGRADATION_STRENGTH),
        "expected_hlt_params": pdv3_hlt_params_dict(),
        "audits": {
            "split_manifest": split_report,
            "hlt_cache": hlt_report,
            "offline_cache": offline_report,
        },
        "problems": problems,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_summary(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# PDV3 Step 1 HLT0.2 Paired Input Audit",
        "",
        f"experiment_name: `{report['experiment_name']}`",
        f"contract: `{report['contract']}`",
        f"overall_ok: {report['ok']}",
        f"manifest_hash: `{report['manifest_hash']}`",
        f"hlt_degradation_strength: {report['hlt_degradation_strength']}",
        "",
        "## Split Counts",
        "",
        "| split | jets | expected | HLT hash | offline hash | identity match | ok |",
        "|---|---:|---:|---|---|---|---|",
    ]
    split_audit = report["audits"]["split_manifest"]
    hlt_audit = report["audits"]["hlt_cache"]
    offline_audit = report["audits"]["offline_cache"]
    split_counts = split_audit["split_summary"]["split_counts"]
    expected_counts = report["expected_split_sizes"]
    for split in PDV3_MODEL_SPLIT_ORDER:
        hlt_item = hlt_audit["split_reports"].get(split, {})
        offline_item = offline_audit["split_reports"].get(split, {})
        hlt_hash = str(hlt_item.get("hlt_content_hash") or "")
        offline_hash = str(offline_item.get("offline_content_hash") or "")
        identity_match = hlt_item.get("jet_identity_hash") == offline_item.get("jet_identity_hash")
        ok = bool(hlt_item.get("ok")) and bool(offline_item.get("ok"))
        lines.append(
            f"| {split} | {int(split_counts.get(split, 0))} | {int(expected_counts[split])} | "
            f"`{hlt_hash[:12]}...` | `{offline_hash[:12]}...` | {identity_match} | {ok} |"
        )
    if report.get("problems"):
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in report["problems"])
    else:
        lines.extend(["", "No problems found."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pdv3_step1_input_audit_reports(
    manifest_path: Path,
    hlt_cache_dir: Path,
    offline_cache_dir: Path,
    output_dir: Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_placeholder_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    report = build_pdv3_step1_input_audit_report(
        manifest_path,
        hlt_cache_dir,
        offline_cache_dir,
        expected_split_sizes=expected_split_sizes,
        expected_placeholder_sizes=expected_placeholder_sizes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / PDV3_SPLIT_AUDIT_REPORT
    hlt_path = output_dir / PDV3_HLT_AUDIT_REPORT
    offline_path = output_dir / PDV3_OFFLINE_AUDIT_REPORT
    report_path = output_dir / PDV3_STEP1_AUDIT_REPORT
    summary_path = output_dir / PDV3_STEP1_AUDIT_SUMMARY
    write_json(split_path, report["audits"]["split_manifest"])
    write_json(hlt_path, report["audits"]["hlt_cache"])
    write_json(offline_path, report["audits"]["offline_cache"])
    write_json(report_path, report)
    write_summary(summary_path, report)
    return {
        "ok": bool(report["ok"]),
        "output_dir": str(output_dir),
        "audit_report": str(report_path),
        "summary": str(summary_path),
        "split_audit_report": str(split_path),
        "hlt_cache_audit_report": str(hlt_path),
        "offline_cache_audit_report": str(offline_path),
    }


__all__ = [
    "build_hlt_report",
    "build_offline_report",
    "build_pdv3_step1_input_audit_report",
    "build_split_report",
    "class_balance_problems",
    "class_counts",
    "pdv3_hlt_params_dict",
    "placeholder_split_size_problems",
    "split_size_problems",
    "write_json",
    "write_pdv3_step1_input_audit_reports",
    "write_summary",
]

#!/usr/bin/env python3
"""Summarize the HLT v2 realistic-degradation pilot baseline sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def strength_tag(value: float | str) -> str:
    """Return the same path-safe strength tag as the Bash submitter.

    Keep the caller's spelling when possible so ``1.0`` maps to ``1p0``,
    matching ``submit_hlt_v2_baseline_sweep.sh``.
    """

    raw = str(value).strip()
    if not raw:
        raise ValueError("strength tag value cannot be empty")
    return raw.replace("-", "m").replace(".", "p")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _nested_value(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _metric(report: Mapping[str, Any], name: str) -> float | None:
    aliases = {
        "accuracy": (
            ("accuracy",),
            ("metrics", "accuracy"),
            ("best_model_val_accuracy",),
            ("final_epoch", "model_val", "accuracy"),
        ),
        "loss": (
            ("loss",),
            ("metrics", "loss"),
            ("best_model_val_loss",),
            ("final_epoch", "model_val", "loss"),
        ),
    }
    value = None
    for path in aliases.get(name, ((name,), ("metrics", name))):
        value = _nested_value(report, path)
        if value is not None:
            break
    if value is None:
        return None
    return float(value)


def _int_metric(report: Mapping[str, Any], name: str) -> int | None:
    aliases = {
        "n_jets": (
            ("n_jets",),
            ("metrics", "n_jets"),
            ("final_epoch", "model_val", "n_jets"),
        ),
    }
    value = None
    for path in aliases.get(name, ((name,), ("metrics", name))):
        value = _nested_value(report, path)
        if value is not None:
            break
    if value is None:
        return None
    return int(value)


def _summary_value(metadata: Mapping[str, Any], summary_key: str, field: str) -> float | None:
    summary = metadata.get(summary_key)
    if not isinstance(summary, Mapping):
        return None
    value = summary.get(field)
    return None if value is None else float(value)


def _diag_summary(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = metadata.get("hlt_diagnostics_summary")
    return summary if isinstance(summary, Mapping) else {}


def _diag_value(metadata: Mapping[str, Any], name: str, field: str = "mean") -> float | None:
    flat_aliases = {
        "drop_fraction": "drop_total_fraction",
        "merge_fraction": "drop_merge_fraction",
        "merge_count": "mean_merges_per_jet",
        "hlt_constits": "mean_hlt_constits",
        "offline_constits": "mean_offline_constits",
    }
    diag = _diag_summary(metadata).get(name)
    if not isinstance(diag, Mapping):
        flat_key = flat_aliases.get(name)
        if field == "mean" and flat_key is not None:
            value = _diag_summary(metadata).get(flat_key)
            return None if value is None else float(value)
        return None
    value = diag.get(field)
    return None if value is None else float(value)


def _metadata_constit_mean(metadata: Mapping[str, Any], key: str) -> float | None:
    summary_key = {
        "offline": "offline_constit_count_summary",
        "hlt": "hlt_constit_count_summary",
    }[key]
    value = _summary_value(metadata, summary_key, "mean")
    if value is not None:
        return value
    return _diag_value(metadata, f"{key}_constits")


def _metadata_constit_p50(metadata: Mapping[str, Any], key: str) -> float | None:
    summary_key = {
        "offline": "offline_constit_count_summary",
        "hlt": "hlt_constit_count_summary",
    }[key]
    return _summary_value(metadata, summary_key, "p50")


def _report_row(
    *,
    label: str,
    strength: float | None,
    report_path: Path,
    cache_metadata_path: Path | None,
    offline_accuracy: float | None,
) -> dict[str, Any]:
    report = _read_json(report_path)
    metadata: dict[str, Any] = {}
    if cache_metadata_path is not None and cache_metadata_path.exists():
        metadata = _read_json(cache_metadata_path)

    accuracy = _metric(report, "accuracy")
    loss = _metric(report, "loss")
    n_jets = _int_metric(report, "n_jets")
    gap = None
    if accuracy is not None and offline_accuracy is not None:
        gap = float(offline_accuracy) - float(accuracy)

    return {
        "baseline": label,
        "strength": "" if strength is None else float(strength),
        "tag": "offline" if strength is None else strength_tag(strength),
        "model_val_accuracy": accuracy,
        "model_val_loss": loss,
        "model_val_n_jets": n_jets,
        "accuracy_gap_vs_offline": gap,
        "report_path": str(report_path),
        "hlt_metadata_path": "" if cache_metadata_path is None else str(cache_metadata_path),
        "hlt_profile": metadata.get("hlt_profile", ""),
        "hlt_profile_version": metadata.get("hlt_profile_version", ""),
        "hlt_degradation_strength": metadata.get("hlt_degradation_strength", ""),
        "hlt_content_hash": metadata.get("hlt_content_hash", ""),
        "offline_constits_mean": _metadata_constit_mean(metadata, "offline"),
        "hlt_constits_mean": _metadata_constit_mean(metadata, "hlt"),
        "offline_constits_p50": _metadata_constit_p50(metadata, "offline"),
        "hlt_constits_p50": _metadata_constit_p50(metadata, "hlt"),
        "drop_fraction_mean": _diag_value(metadata, "drop_fraction"),
        "merge_fraction_mean": _diag_value(metadata, "merge_fraction"),
        "reassign_fraction_mean": _diag_value(metadata, "reassign_fraction"),
        "mean_merges_per_jet": _diag_value(metadata, "merge_count"),
        "pt_response_mean": _diag_value(metadata, "pt_response"),
        "pt_response_p10": _diag_value(metadata, "pt_response", "p10"),
        "pt_response_p90": _diag_value(metadata, "pt_response", "p90"),
    }


def build_sweep_report(
    sweep_root: Path,
    *,
    strengths: tuple[float | str, ...],
) -> dict[str, Any]:
    offline_report_path = (
        sweep_root / "offline_reference" / "teachers" / "offline_part_teacher_10class" / "model_val_report.json"
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    problems: list[str] = []
    offline_accuracy: float | None = None
    if offline_report_path.exists():
        offline_row = _report_row(
            label="offline_part",
            strength=None,
            report_path=offline_report_path,
            cache_metadata_path=None,
            offline_accuracy=None,
        )
        offline_accuracy = offline_row["model_val_accuracy"]
        if offline_accuracy is None:
            problems.append(f"offline model_val accuracy is missing in {offline_report_path}")
        rows.append(offline_row)
    else:
        missing.append(str(offline_report_path))

    for strength in strengths:
        strength_value = float(strength)
        run_root = sweep_root / f"hlt_v2_strength_{strength_tag(strength)}"
        report_path = run_root / "teachers" / "hlt_part_teacher_10class" / "model_val_report.json"
        metadata_path = run_root / "hlt_cache" / "model_val_fixed_hlt_metadata.json"
        if not report_path.exists():
            missing.append(str(report_path))
            continue
        if not metadata_path.exists():
            missing.append(str(metadata_path))
            continue
        row = _report_row(
            label="hlt_v2_part",
            strength=strength_value,
            report_path=report_path,
            cache_metadata_path=metadata_path,
            offline_accuracy=offline_accuracy,
        )
        if row["model_val_accuracy"] is None:
            problems.append(f"HLT v2 strength {strength_value:g} model_val accuracy is missing in {report_path}")
        if row["drop_fraction_mean"] is None:
            problems.append(f"HLT v2 strength {strength_value:g} drop diagnostics are missing in {metadata_path}")
        rows.append(row)

    return {
        "ok": not missing and not problems and len(rows) == len(strengths) + 1,
        "sweep_root": str(sweep_root),
        "strengths": [float(value) for value in strengths],
        "offline_report_path": str(offline_report_path),
        "rows": rows,
        "missing": missing,
        "problems": problems,
        "notes": [
            "This Step 4 pilot report intentionally uses model_val only.",
            "final_test remains untouched for HLT v2 calibration and strength selection.",
        ],
    }


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hlt_v2_baseline_sweep_model_val.json"
    csv_path = output_dir / "hlt_v2_baseline_sweep_model_val.csv"
    md_path = output_dir / "hlt_v2_baseline_sweep_model_val.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    rows = list(report.get("rows") or [])
    fieldnames = [
        "baseline",
        "strength",
        "tag",
        "model_val_accuracy",
        "accuracy_gap_vs_offline",
        "model_val_loss",
        "model_val_n_jets",
        "hlt_profile",
        "hlt_profile_version",
        "hlt_degradation_strength",
        "offline_constits_mean",
        "hlt_constits_mean",
        "drop_fraction_mean",
        "merge_fraction_mean",
        "reassign_fraction_mean",
        "mean_merges_per_jet",
        "pt_response_mean",
        "hlt_content_hash",
        "report_path",
        "hlt_metadata_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# HLT V2 Baseline Sweep Model-Val Report",
        "",
        f"ok: {bool(report.get('ok'))}",
        f"sweep_root: `{report.get('sweep_root')}`",
        "",
        "This report is model-val only. It is meant to choose a realistic HLT v2 strength before any final-test evaluation.",
        "",
        "| baseline | strength | model-val acc | gap vs offline | loss | HLT profile | HLT mean constits | drop mean | merge mean |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        strength = row["strength"] if row["strength"] != "" else "offline"
        accuracy = row["model_val_accuracy"]
        gap = row["accuracy_gap_vs_offline"]
        loss = row["model_val_loss"]
        hlt_mean = row["hlt_constits_mean"]
        drop = row["drop_fraction_mean"]
        merge = row["merge_fraction_mean"]
        lines.append(
            "| {baseline} | {strength} | {accuracy} | {gap} | {loss} | {profile} | {hlt_mean} | {drop} | {merge} |".format(
                baseline=row["baseline"],
                strength=strength,
                accuracy="" if accuracy is None else f"{float(accuracy):.6f}",
                gap="" if gap is None else f"{float(gap):.6f}",
                loss="" if loss is None else f"{float(loss):.6f}",
                profile=row["hlt_profile"],
                hlt_mean="" if hlt_mean is None else f"{float(hlt_mean):.3f}",
                drop="" if drop is None else f"{float(drop):.5f}",
                merge="" if merge is None else f"{float(merge):.5f}",
            )
        )
    if report.get("missing"):
        lines.extend(["", "## Missing Inputs", ""])
        lines.extend(f"- `{path}`" for path in report["missing"])
    if report.get("problems"):
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in report["problems"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--strengths", nargs="+", default=["0.0", "0.75", "1.0", "1.25"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sweep_root = Path(args.sweep_root)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_root / "baseline_sweep_report"
    report = build_sweep_report(sweep_root, strengths=tuple(str(value) for value in args.strengths))
    outputs = write_outputs(report, output_dir)
    print("hlt_v2_baseline_sweep_report_complete:")
    print(json.dumps({"ok": bool(report["ok"]), "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

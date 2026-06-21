#!/usr/bin/env python3
"""Write a compact final report for the set-matching multi-view branch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RECONSTRUCTOR_ARCHITECTURES = ("gt", "pn", "pfn", "pcnn")
TAGGER_VARIANTS = (
    "hlt_only",
    "hlt_plus_gt",
    "hlt_plus_pn",
    "hlt_plus_pfn",
    "hlt_plus_pcnn",
    "five_view_plain",
    "five_view_geometry",
    "five_view_no_confidence",
    "view_label_shuffle_control",
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_offline_teacher_references(experiment_dir: Path) -> list[dict[str, Any]]:
    reference_root = experiment_dir / "offline_teacher_reference"
    if not reference_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for report_path in sorted(reference_root.glob("*/run_report.json")):
        payload = read_json(report_path)
        if payload is None:
            continue
        rows.append(
            {
                "report_path": str(report_path),
                "output_dir": str(report_path.parent),
                "best_epoch": metric(payload, ("best_epoch",)),
                "model_val_accuracy": metric(payload, ("model_val_metrics", "accuracy")),
                "stack_val_accuracy": metric(payload, ("stack_val_metrics", "accuracy")),
                "stack_val_auc": metric(payload, ("stack_val_metrics", "binary_metrics", "auc")),
                "stack_val_fpr_at_signal_eff_0p30": metric(
                    payload,
                    ("stack_val_metrics", "binary_metrics", "fpr_at_signal_eff_0p30"),
                ),
                "stack_val_fpr_at_signal_eff_0p50": metric(
                    payload,
                    ("stack_val_metrics", "binary_metrics", "fpr_at_signal_eff_0p50"),
                ),
                "final_test_accuracy": metric(payload, ("final_test_metrics", "accuracy")),
                "final_test_auc": metric(payload, ("final_test_metrics", "binary_metrics", "auc")),
                "final_test_fpr_at_signal_eff_0p30": metric(
                    payload,
                    ("final_test_metrics", "binary_metrics", "fpr_at_signal_eff_0p30"),
                ),
                "final_test_fpr_at_signal_eff_0p50": metric(
                    payload,
                    ("final_test_metrics", "binary_metrics", "fpr_at_signal_eff_0p50"),
                ),
                "label_names": metric(payload, ("label_names",)),
                "config": metric(payload, ("config",)),
            }
        )
    return rows


def metric(payload: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--reconstructor-dir", default=None)
    parser.add_argument("--reconstructed-view-dir", default=None)
    parser.add_argument("--tagger-root", default=None)
    parser.add_argument("--ablation-dir", default=None)
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    reconstructor_dir = Path(args.reconstructor_dir) if args.reconstructor_dir else experiment_dir / "reconstructors"
    reconstructed_view_dir = Path(args.reconstructed_view_dir) if args.reconstructed_view_dir else experiment_dir / "reconstructed_views"
    tagger_root = Path(args.tagger_root) if args.tagger_root else experiment_dir / "taggers"
    ablation_dir = Path(args.ablation_dir) if args.ablation_dir else experiment_dir / "ablations" / "five_view_ablation_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    reconstructors: list[dict[str, Any]] = []
    for architecture in RECONSTRUCTOR_ARCHITECTURES:
        report_path = reconstructor_dir / architecture / "run_report.json"
        payload = read_json(report_path)
        if payload is None:
            problems.append(f"missing reconstructor report: {report_path}")
        reconstructors.append(
            {
                "architecture": architecture,
                "report_path": str(report_path),
                "exists": payload is not None,
                "best_epoch": metric(payload, ("best_epoch",)),
                "best_model_val_loss": metric(payload, ("best_model_val_loss",)),
                "best_model_val_total": metric(payload, ("best_model_val_metrics", "total")),
            }
        )

    caches: list[dict[str, Any]] = []
    for architecture in RECONSTRUCTOR_ARCHITECTURES:
        report_path = reconstructed_view_dir / architecture / "cache_report.json"
        payload = read_json(report_path)
        if payload is None:
            problems.append(f"missing reconstructed-view cache report: {report_path}")
        caches.append(
            {
                "architecture": architecture,
                "report_path": str(report_path),
                "exists": payload is not None,
                "splits": metric(payload, ("splits",)),
                "cache_paths": metric(payload, ("cache_paths",)),
            }
        )

    taggers: list[dict[str, Any]] = []
    for variant in TAGGER_VARIANTS:
        report_path = tagger_root / variant / "run_report.json"
        payload = read_json(report_path)
        if payload is None:
            problems.append(f"missing tagger report: {report_path}")
        final_test_metrics = metric(payload, ("final_test_metrics",))
        taggers.append(
            {
                "variant": variant,
                "report_path": str(report_path),
                "exists": payload is not None,
                "best_epoch": metric(payload, ("best_epoch",)),
                "selection_metric": metric(payload, ("selection_metric",)),
                "selection_metric_direction": metric(payload, ("selection_metric_direction",)),
                "best_model_selection_metric_value": metric(payload, ("best_model_selection_metric_value",)),
                "stack_val_accuracy": metric(payload, ("best_stack_val_metrics", "accuracy")),
                "stack_val_auc": metric(payload, ("best_stack_val_metrics", "binary_metrics", "auc")),
                "stack_val_fpr_at_signal_eff_0p30": metric(
                    payload,
                    ("best_stack_val_metrics", "binary_metrics", "fpr_at_signal_eff_0p30"),
                ),
                "stack_val_fpr_at_signal_eff_0p50": metric(
                    payload,
                    ("best_stack_val_metrics", "binary_metrics", "fpr_at_signal_eff_0p50"),
                ),
                "final_test_accuracy": final_test_metrics.get("accuracy") if isinstance(final_test_metrics, dict) else None,
                "final_test_auc": metric(payload, ("final_test_metrics", "binary_metrics", "auc")),
                "final_test_fpr_at_signal_eff_0p30": metric(
                    payload,
                    ("final_test_metrics", "binary_metrics", "fpr_at_signal_eff_0p30"),
                ),
                "final_test_fpr_at_signal_eff_0p50": metric(
                    payload,
                    ("final_test_metrics", "binary_metrics", "fpr_at_signal_eff_0p50"),
                ),
                "final_test_background_rejection_at_signal_eff_0p30": metric(
                    payload,
                    ("final_test_metrics", "binary_metrics", "background_rejection_at_signal_eff_0p30"),
                ),
                "final_test_background_rejection_at_signal_eff_0p50": metric(
                    payload,
                    ("final_test_metrics", "binary_metrics", "background_rejection_at_signal_eff_0p50"),
                ),
                "final_test_evaluated": bool(metric(payload, ("final_test_evaluated",))),
            }
        )

    ablation_report_path = ablation_dir / "run_report.json"
    ablation_report = read_json(ablation_report_path)
    if ablation_report is None:
        problems.append(f"missing ablation run report: {ablation_report_path}")
    elif bool(args.confirm_final_test) and not bool(ablation_report.get("final_test_evaluated")):
        problems.append("ablation report did not evaluate final_test despite --confirm-final-test")
    summary_rows = read_summary_csv(ablation_dir / "summary.csv")
    filtered_manifest_report_path = experiment_dir / "binary_inputs" / "filtered_manifest_report.json"
    filtered_manifest_report = read_json(filtered_manifest_report_path)
    offline_teacher_references = read_offline_teacher_references(experiment_dir)

    ok = not problems
    report = {
        "ok": ok,
        "experiment_step": "set_matching_multiview_step12_final_report",
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "confirm_final_test": bool(args.confirm_final_test),
        "problems": problems,
        "reconstructors": reconstructors,
        "reconstructed_view_caches": caches,
        "taggers": taggers,
        "ablation_report": str(ablation_report_path),
        "ablation_final_test_evaluated": bool((ablation_report or {}).get("final_test_evaluated")),
        "ablation_summary_rows": summary_rows,
        "filtered_manifest_report_path": str(filtered_manifest_report_path),
        "filtered_manifest_report_exists": filtered_manifest_report is not None,
        "filtered_manifest_report": filtered_manifest_report,
        "offline_teacher_references": offline_teacher_references,
    }

    json_path = output_dir / "final_report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Set-Matching Multi-View Final Report",
        "",
        f"ok: {ok}",
        f"experiment_dir: {experiment_dir}",
        f"ablation_report: {ablation_report_path}",
        "",
        "## Taggers",
        "",
        "| variant | selection_metric | selected_value | stack_val_accuracy | stack_val_auc | final_test_accuracy | final_test_auc | final_fpr@30 | final_fpr@50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in taggers:
        lines.append(
            f"| {row['variant']} | {row['selection_metric']} | {row['best_model_selection_metric_value']} | "
            f"{row['stack_val_accuracy']} | {row['stack_val_auc']} | "
            f"{row['final_test_accuracy']} | {row['final_test_auc']} | "
            f"{row['final_test_fpr_at_signal_eff_0p30']} | {row['final_test_fpr_at_signal_eff_0p50']} |"
        )
    if offline_teacher_references:
        lines += [
            "",
            "## Offline References",
            "",
            "| report | stack_val_auc | final_test_auc | final_fpr@30 | final_fpr@50 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in offline_teacher_references:
            lines.append(
                f"| {row['report_path']} | {row['stack_val_auc']} | {row['final_test_auc']} | "
                f"{row['final_test_fpr_at_signal_eff_0p30']} | {row['final_test_fpr_at_signal_eff_0p50']} |"
            )
    if problems:
        lines += ["", "## Problems", ""]
        lines.extend(f"- {problem}" for problem in problems)
    md_path = output_dir / "final_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("set_matching_multiview_final_report_complete:")
    print(f"  ok: {ok}")
    print(f"  final_report_json: {json_path}")
    print(f"  final_report_md: {md_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

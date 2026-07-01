#!/usr/bin/env python3
"""Write final report tables for AV10 architecture-view ensembles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from teacher_logit_reco.architecture_view_part import (  # noqa: E402
    ARCHITECTURE_VIEW_10CLASS_DEFAULT_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_FUSION_CONTRACT,
    ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
    normalize_architecture_view_variant,
)
from teacher_logit_reco.set_matching.train import source_metadata  # noqa: E402


REPORT_CONTRACT = "architecture_view_10class_final_report_v1"
REPORT_STEP = "architecture_view_10class_step5_final_report"
REPORT_SPLITS = ("model_val", "stack_val", "final_test")


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                keys.append(str(key))
    if not keys:
        keys = ["empty"]
        rows = [{"empty": ""}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if output == output and abs(output) != float("inf") else None


def _metrics_for_split(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    if split == "model_val":
        for key in ("best_model_val_metrics", "model_val_metrics"):
            value = report.get(key)
            if isinstance(value, Mapping):
                return value
    if split == "stack_val":
        for key in ("stack_val_metrics", "best_stack_val_metrics"):
            value = report.get(key)
            if isinstance(value, Mapping):
                return value
    if split == "final_test":
        value = report.get("final_test_metrics")
        if isinstance(value, Mapping):
            return value
    value = report.get(f"{split}_metrics")
    return value if isinstance(value, Mapping) else None


def _load_variant_reports(tagger_root: Path, variants: Sequence[str]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    reports: dict[str, Mapping[str, Any]] = {}
    problems: list[str] = []
    for variant in variants:
        canonical = normalize_architecture_view_variant(variant)
        path = tagger_root / canonical / "run_report.json"
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            problems.append(f"missing or invalid run_report for {canonical}: {path}")
            continue
        reports[canonical] = payload
    return reports, problems


def _individual_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, report in reports.items():
        for split in REPORT_SPLITS:
            metrics = _metrics_for_split(report, split)
            if not isinstance(metrics, Mapping):
                continue
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "accuracy": metrics.get("accuracy"),
                    "macro_per_class_accuracy": metrics.get("macro_per_class_accuracy"),
                    "cross_entropy": metrics.get("cross_entropy") or metrics.get("loss"),
                    "n_jets": metrics.get("n_jets"),
                    "best_epoch": report.get("best_epoch"),
                    "checkpoint": report.get("checkpoint"),
                }
            )
    return rows


def _per_class_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, report in reports.items():
        for split in REPORT_SPLITS:
            metrics = _metrics_for_split(report, split)
            if not isinstance(metrics, Mapping):
                continue
            for row in metrics.get("per_class_accuracy", ()):
                if not isinstance(row, Mapping):
                    continue
                payload = {"variant": variant, "split": split}
                payload.update(dict(row))
                rows.append(payload)
    return rows


def _fusion_rows(fusion_report: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    if not isinstance(fusion_report, Mapping):
        return rows, binary_rows
    for group_name, group in fusion_report.get("groups", {}).items():
        if not isinstance(group, Mapping):
            continue
        for mode, mode_report in group.get("fusion_modes", {}).items():
            if not isinstance(mode_report, Mapping):
                continue
            metrics = mode_report.get("metrics")
            if isinstance(metrics, Mapping):
                for split, split_metrics in metrics.items():
                    if not isinstance(split_metrics, Mapping):
                        continue
                    rows.append(
                        {
                            "group": group_name,
                            "mode": mode,
                            "split": split,
                            "accuracy": split_metrics.get("accuracy"),
                            "macro_per_class_accuracy": split_metrics.get("macro_per_class_accuracy"),
                            "cross_entropy": split_metrics.get("cross_entropy"),
                            "n_jets": split_metrics.get("n_jets"),
                        }
                    )
            binary = mode_report.get("binary_projection_results")
            if isinstance(binary, Mapping):
                for pair, pair_report in binary.items():
                    if not isinstance(pair_report, Mapping) or not pair_report.get("available"):
                        continue
                    for split, split_metrics in pair_report.get("metrics", {}).items():
                        if not isinstance(split_metrics, Mapping):
                            continue
                        binary_rows.append(
                            {
                                "group": group_name,
                                "mode": mode,
                                "pair": pair,
                                "split": split,
                                "auc": split_metrics.get("auc"),
                                "fpr_at_signal_eff_0p30": split_metrics.get("fpr_at_signal_eff_0p30"),
                                "fpr_at_signal_eff_0p50": split_metrics.get("fpr_at_signal_eff_0p50"),
                                "background_rejection_at_signal_eff_0p50": split_metrics.get(
                                    "background_rejection_at_signal_eff_0p50"
                                ),
                                "n_jets": split_metrics.get("n_jets"),
                            }
                        )
    return rows, binary_rows


def _best_row(rows: Sequence[Mapping[str, Any]], *, split: str = "final_test") -> Mapping[str, Any] | None:
    candidates = [row for row in rows if row.get("split") == split and _float(row.get("accuracy")) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (_float(row.get("accuracy")) or float("-inf"), -(_float(row.get("cross_entropy")) or 1.0e9)))


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report.get("summary", {})
    lines = [
        "# Architecture-View 10-Class Report",
        "",
        f"- Baseline variant: `{summary.get('baseline_variant')}`",
        f"- Best individual final-test variant: `{summary.get('best_individual_variant')}`",
        f"- Best individual final-test accuracy: `{summary.get('best_individual_accuracy')}`",
        f"- Baseline final-test accuracy: `{summary.get('baseline_accuracy')}`",
        f"- Best fusion final-test group/mode: `{summary.get('best_fusion_group')}` / `{summary.get('best_fusion_mode')}`",
        f"- Best fusion final-test accuracy: `{summary.get('best_fusion_accuracy')}`",
        "",
        "## Outputs",
        "",
    ]
    for name, value in report.get("outputs", {}).items():
        lines.append(f"- `{name}`: `{value}`")
    problems = report.get("problems") or []
    if problems:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in problems)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tagger-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", nargs="+", default=list(ARCHITECTURE_VIEW_10CLASS_DEFAULT_VARIANTS))
    parser.add_argument("--baseline-variant", default=ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK)
    parser.add_argument("--fusion-report", default=None)
    parser.add_argument("--standalone-fusion-report", default=None)
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not bool(args.confirm_final_test):
        raise ValueError("Refusing AV10 final report without --confirm-final-test")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = tuple(normalize_architecture_view_variant(variant) for variant in args.variants)
    baseline_variant = normalize_architecture_view_variant(args.baseline_variant)
    reports, problems = _load_variant_reports(Path(args.tagger_root), variants)
    individual_rows = _individual_rows(reports)
    per_class_rows = _per_class_rows(reports)
    fusion_report = _read_json(Path(args.fusion_report)) if args.fusion_report else None
    if args.fusion_report and not isinstance(fusion_report, Mapping):
        problems.append(f"missing or invalid fusion report: {args.fusion_report}")
    if isinstance(fusion_report, Mapping) and fusion_report.get("contract") != ARCHITECTURE_VIEW_10CLASS_FUSION_CONTRACT:
        problems.append(f"fusion report contract mismatch: {args.fusion_report}")
    standalone_report = _read_json(Path(args.standalone_fusion_report)) if args.standalone_fusion_report else None
    if args.standalone_fusion_report and not isinstance(standalone_report, Mapping):
        problems.append(f"missing or invalid standalone fusion report: {args.standalone_fusion_report}")
    fusion_rows, binary_rows = _fusion_rows(fusion_report)
    best_individual = _best_row(individual_rows)
    best_fusion = _best_row(fusion_rows)
    baseline_row = next(
        (row for row in individual_rows if row.get("variant") == baseline_variant and row.get("split") == "final_test"),
        None,
    )
    outputs = {
        "report_json": str(output_dir / "architecture_view_10class_report.json"),
        "report_md": str(output_dir / "architecture_view_10class_report.md"),
        "individual_model_table": str(output_dir / "individual_model_table.csv"),
        "fusion_metric_table": str(output_dir / "fusion_metric_table.csv"),
        "binary_projection_table": str(output_dir / "binary_projection_table.csv"),
        "per_class_table": str(output_dir / "per_class_accuracy.csv"),
        "run_report": str(output_dir / "run_report.json"),
    }
    summary = {
        "baseline_variant": baseline_variant,
        "baseline_accuracy": None if baseline_row is None else baseline_row.get("accuracy"),
        "best_individual_variant": None if best_individual is None else best_individual.get("variant"),
        "best_individual_accuracy": None if best_individual is None else best_individual.get("accuracy"),
        "best_fusion_group": None if best_fusion is None else best_fusion.get("group"),
        "best_fusion_mode": None if best_fusion is None else best_fusion.get("mode"),
        "best_fusion_accuracy": None if best_fusion is None else best_fusion.get("accuracy"),
        "label_names": list(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES),
    }
    report = {
        "contract": REPORT_CONTRACT,
        "step": REPORT_STEP,
        "ok": not problems,
        "source": source_metadata(),
        "tagger_root": str(args.tagger_root),
        "fusion_report": args.fusion_report,
        "standalone_fusion_report": args.standalone_fusion_report,
        "variants": list(variants),
        "summary": summary,
        "individual_rows": individual_rows,
        "fusion_rows": fusion_rows,
        "binary_projection_rows": binary_rows,
        "problems": problems,
        "outputs": outputs,
    }
    _write_csv(Path(outputs["individual_model_table"]), individual_rows)
    _write_csv(Path(outputs["fusion_metric_table"]), fusion_rows)
    _write_csv(Path(outputs["binary_projection_table"]), binary_rows)
    _write_csv(Path(outputs["per_class_table"]), per_class_rows)
    save_json(output_dir / "architecture_view_10class_report.json", _jsonable(report))
    save_json(output_dir / "run_report.json", _jsonable(report))
    _write_markdown(output_dir / "architecture_view_10class_report.md", report)
    print("architecture_view_10class_report_complete:")
    print(f"  ok: {report['ok']}")
    print(f"  output_dir: {output_dir}")
    print(f"  best_individual_variant: {summary['best_individual_variant']}")
    print(f"  best_individual_accuracy: {summary['best_individual_accuracy']}")
    print(f"  best_fusion_group: {summary['best_fusion_group']}")
    print(f"  best_fusion_mode: {summary['best_fusion_mode']}")
    print(f"  best_fusion_accuracy: {summary['best_fusion_accuracy']}")
    if problems:
        print("  problems:")
        for problem in problems:
            print(f"    - {problem}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

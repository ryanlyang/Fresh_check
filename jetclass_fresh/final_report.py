"""Step 13 final fresh-start report generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict

from .hlt_baseline import save_json
from .hlt_control import compare_hlt5_to_reco7_reports


DEFAULT_HLT_BASELINE_REPORT = "checkpoints/jetclass_fresh_hlt_baselines/single_hlt_seed101/model_val_report.json"
DEFAULT_OFFLINE_TEACHER_REPORT = "checkpoints/jetclass_fresh_offline_teacher/offline_teacher_seed707/model_val_report.json"
DEFAULT_RECO7_FUSION_REPORT = "checkpoints/jetclass_fresh_fusion/reco7_plus_hlt/fusion_report.json"
DEFAULT_HLT5_FUSION_REPORT = "checkpoints/jetclass_fresh_fusion/hlt5_seed_control/fusion_report.json"
DEFAULT_RECO7_AUDIT_REPORT = "checkpoints/jetclass_fresh_audits/reco7_plus_hlt/audit_report.json"
DEFAULT_HLT5_AUDIT_REPORT = "checkpoints/jetclass_fresh_audits/hlt5_seed_control/audit_report.json"

FUSION_METHOD_LABELS = {
    "uniform_probability_average": "Uniform probability average",
    "weighted_probability_average": "Weighted probability average",
    "weighted_logit_average": "Weighted logit average",
    "stacked_logistic_regression": "Stacked logistic regression",
}


@dataclass
class FinalReportConfig:
    """Configuration for the Step 13 report writer."""

    output_dir: str = "checkpoints/jetclass_fresh_final_report"
    hlt_baseline_report: str = DEFAULT_HLT_BASELINE_REPORT
    offline_teacher_report: str = DEFAULT_OFFLINE_TEACHER_REPORT
    reco7_fusion_report: str = DEFAULT_RECO7_FUSION_REPORT
    hlt5_fusion_report: str = DEFAULT_HLT5_FUSION_REPORT
    reco7_audit_report: str = DEFAULT_RECO7_AUDIT_REPORT
    hlt5_audit_report: str | None = DEFAULT_HLT5_AUDIT_REPORT
    markdown_filename: str = "FINAL_FRESH_START_REPORT.md"
    json_filename: str = "final_report_summary.json"
    allow_missing: bool = False
    substantial_accuracy_delta: float = 0.01
    require_cross_entropy_nonworse: bool = True


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_named_report(path: str | Path | None, name: str, *, allow_missing: bool, missing: list[Dict[str, str]]):
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        row = {"name": name, "path": str(path), "reason": "file not found"}
        missing.append(row)
        if allow_missing:
            return None
        raise FileNotFoundError(f"Missing required Step 13 input {name}: {path}")
    return _load_json(resolved)


def _model_val_summary(report: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if report is None:
        return None
    return {
        "experiment_step": report.get("experiment_step"),
        "reference_role": report.get("reference_role"),
        "best_epoch": report.get("best_epoch"),
        "accuracy": report.get("best_model_val_accuracy"),
        "loss": report.get("best_model_val_loss"),
        "epochs_completed": report.get("epochs_completed"),
        "checkpoint": report.get("checkpoint"),
        "no_final_test_evaluation": report.get("no_final_test_evaluation"),
    }


def fusion_method_metrics(report: Mapping[str, Any] | None, method: str, split: str = "final_test") -> Dict[str, Any] | None:
    """Extract one method/split metric row from a fusion report."""

    if report is None:
        return None
    section = report.get(method)
    if not isinstance(section, Mapping):
        return None
    if method == "uniform_probability_average":
        metrics = section.get(split)
    else:
        metrics = section.get("metrics", {}).get(split)
    return dict(metrics) if isinstance(metrics, Mapping) else None


def single_model_metric(report: Mapping[str, Any] | None, model_name: str, split: str = "final_test") -> Dict[str, Any] | None:
    if report is None:
        return None
    metrics = report.get("single_models", {}).get(split, {}).get(model_name)
    return dict(metrics) if isinstance(metrics, Mapping) else None


def _audit_status(report: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if report is None:
        return None
    audits = report.get("audits", {})
    return {
        "ok": bool(report.get("ok")),
        "audit_items": {
            str(name): bool(value.get("ok")) if isinstance(value, Mapping) else False
            for name, value in audits.items()
        },
    }


def _all_configured_audits_ok(*reports: Mapping[str, Any] | None) -> bool:
    configured = [report for report in reports if report is not None]
    return bool(configured) and all(bool(report.get("ok")) for report in configured)


def _metric_delta(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None, metric: str) -> float | None:
    if left is None or right is None or metric not in left or metric not in right:
        return None
    return float(right[metric]) - float(left[metric])


def _interpretation(
    *,
    hlt5_stack: Mapping[str, Any] | None,
    reco7_stack: Mapping[str, Any] | None,
    audits_ok: bool,
    missing_inputs: list[Dict[str, str]],
    substantial_accuracy_delta: float,
    require_cross_entropy_nonworse: bool,
) -> Dict[str, Any]:
    accuracy_delta = _metric_delta(hlt5_stack, reco7_stack, "accuracy")
    cross_entropy_delta = _metric_delta(hlt5_stack, reco7_stack, "cross_entropy")
    if missing_inputs:
        return {
            "state": "incomplete",
            "claim": "not_interpretable_yet",
            "reason": "Required result or audit inputs are missing.",
            "accuracy_delta_reco7_minus_hlt5": accuracy_delta,
            "cross_entropy_delta_reco7_minus_hlt5": cross_entropy_delta,
        }
    if not audits_ok:
        return {
            "state": "audit_failed",
            "claim": "not_interpretable_yet",
            "reason": "One or more configured audit reports did not pass.",
            "accuracy_delta_reco7_minus_hlt5": accuracy_delta,
            "cross_entropy_delta_reco7_minus_hlt5": cross_entropy_delta,
        }
    if accuracy_delta is None:
        return {
            "state": "insufficient_metrics",
            "claim": "not_interpretable_yet",
            "reason": "Locked final_test stacked metrics are unavailable.",
            "accuracy_delta_reco7_minus_hlt5": None,
            "cross_entropy_delta_reco7_minus_hlt5": cross_entropy_delta,
        }

    ce_ok = (
        True
        if not require_cross_entropy_nonworse or cross_entropy_delta is None
        else cross_entropy_delta <= 0.0
    )
    if accuracy_delta >= float(substantial_accuracy_delta) and ce_ok:
        state = "supports_reco7_stronger"
        claim = "7+HLT stack is substantially better than the 5-HLT seed stack under the configured threshold."
    elif accuracy_delta > 0.0:
        state = "modest_or_mixed_reco7_gain"
        claim = "7+HLT stack is higher-accuracy than HLT5, but not substantially under the configured threshold."
    else:
        state = "does_not_support_reco7_gain"
        claim = "The locked metrics do not show a positive 7+HLT accuracy gain over HLT5."

    return {
        "state": state,
        "claim": claim,
        "reason": (
            f"Substantial threshold is absolute accuracy delta >= {float(substantial_accuracy_delta):.4f}; "
            f"cross-entropy non-worse required={bool(require_cross_entropy_nonworse)}."
        ),
        "accuracy_delta_reco7_minus_hlt5": accuracy_delta,
        "cross_entropy_delta_reco7_minus_hlt5": cross_entropy_delta,
    }


def build_final_report_summary(config: FinalReportConfig) -> Dict[str, Any]:
    """Load saved artifacts and build a machine-readable Step 13 summary."""

    missing_inputs: list[Dict[str, str]] = []
    hlt_baseline = _load_named_report(
        config.hlt_baseline_report,
        "single_hlt_baseline",
        allow_missing=config.allow_missing,
        missing=missing_inputs,
    )
    offline_teacher = _load_named_report(
        config.offline_teacher_report,
        "offline_teacher",
        allow_missing=config.allow_missing,
        missing=missing_inputs,
    )
    reco7_fusion = _load_named_report(
        config.reco7_fusion_report,
        "reco7_plus_hlt_fusion",
        allow_missing=config.allow_missing,
        missing=missing_inputs,
    )
    hlt5_fusion = _load_named_report(
        config.hlt5_fusion_report,
        "hlt5_seed_control_fusion",
        allow_missing=config.allow_missing,
        missing=missing_inputs,
    )
    reco7_audits = _load_named_report(
        config.reco7_audit_report,
        "reco7_plus_hlt_audits",
        allow_missing=config.allow_missing,
        missing=missing_inputs,
    )
    hlt5_audits = _load_named_report(
        config.hlt5_audit_report,
        "hlt5_seed_control_audits",
        allow_missing=config.allow_missing,
        missing=missing_inputs,
    )

    hlt5_stack = fusion_method_metrics(hlt5_fusion, "stacked_logistic_regression")
    reco7_stack = fusion_method_metrics(reco7_fusion, "stacked_logistic_regression")
    audits_ok = _all_configured_audits_ok(reco7_audits, hlt5_audits)
    comparison = (
        compare_hlt5_to_reco7_reports(hlt5_fusion, reco7_fusion)
        if hlt5_fusion is not None and reco7_fusion is not None
        else None
    )

    fusion_summary: Dict[str, Any] = {
        "single_hlt_from_reco7_final_test": single_model_metric(reco7_fusion, "hlt_baseline"),
        "hlt5_stack_final_test": hlt5_stack,
        "reco7_plus_hlt_stack_final_test": reco7_stack,
        "simple_fusion_baselines": {
            "hlt5": {
                method: fusion_method_metrics(hlt5_fusion, method)
                for method in (
                    "uniform_probability_average",
                    "weighted_probability_average",
                    "weighted_logit_average",
                )
            },
            "reco7_plus_hlt": {
                method: fusion_method_metrics(reco7_fusion, method)
                for method in (
                    "uniform_probability_average",
                    "weighted_probability_average",
                    "weighted_logit_average",
                )
            },
        },
        "comparison": comparison,
    }

    interpretation = _interpretation(
        hlt5_stack=hlt5_stack,
        reco7_stack=reco7_stack,
        audits_ok=audits_ok,
        missing_inputs=missing_inputs,
        substantial_accuracy_delta=config.substantial_accuracy_delta,
        require_cross_entropy_nonworse=config.require_cross_entropy_nonworse,
    )

    return {
        "experiment_step": "step13_final_fresh_start_report",
        "config": asdict(config),
        "missing_inputs": missing_inputs,
        "model_val_references": {
            "single_hlt_baseline": _model_val_summary(hlt_baseline),
            "offline_teacher": _model_val_summary(offline_teacher),
        },
        "locked_fusion_results": fusion_summary,
        "audit_outcomes": {
            "all_configured_audits_ok": audits_ok,
            "reco7_plus_hlt": _audit_status(reco7_audits),
            "hlt5_seed_control": _audit_status(hlt5_audits),
        },
        "interpretation": interpretation,
        "discipline_note": (
            "This report summarizes clean-room outputs only. Do not tune implementation choices from final_test metrics."
        ),
    }


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _metrics_cells(metrics: Mapping[str, Any] | None) -> tuple[str, str, str]:
    if not metrics:
        return "missing", "missing", "missing"
    return (
        _fmt_metric(metrics.get("accuracy")),
        _fmt_metric(metrics.get("cross_entropy")),
        _fmt_metric(metrics.get("n_jets")),
    )


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_final_report_markdown(summary: Mapping[str, Any]) -> str:
    """Render the Step 13 summary as a Markdown report."""

    lines: list[str] = [
        "# JetClass Same-HLT Fresh-Start Report",
        "",
        "## Interpretation",
        "",
        str(summary["interpretation"]["claim"]),
        "",
        f"State: `{summary['interpretation']['state']}`",
        "",
        str(summary["interpretation"]["reason"]),
        "",
    ]

    missing = summary.get("missing_inputs", [])
    if missing:
        lines.extend(["## Missing Inputs", ""])
        lines.extend(
            _markdown_table(
                ["Input", "Path", "Reason"],
                [[str(row["name"]), str(row["path"]), str(row["reason"])] for row in missing],
            )
        )
        lines.append("")

    references = summary.get("model_val_references", {})
    ref_rows = []
    for name, report in references.items():
        if report is None:
            ref_rows.append([name, "missing", "missing", "missing", "missing"])
        else:
            ref_rows.append(
                [
                    name,
                    _fmt_metric(report.get("accuracy")),
                    _fmt_metric(report.get("loss")),
                    _fmt_metric(report.get("best_epoch")),
                    str(report.get("reference_role") or report.get("experiment_step")),
                ]
            )
    lines.extend(["## Model-Val References", ""])
    lines.extend(_markdown_table(["Model", "Accuracy", "Loss", "Best Epoch", "Role"], ref_rows))
    lines.append("")

    locked = summary.get("locked_fusion_results", {})
    fusion_rows = []
    for label, metrics in [
        ("Single HLT baseline from reco7 final_test", locked.get("single_hlt_from_reco7_final_test")),
        ("HLT5 stacked logistic final_test", locked.get("hlt5_stack_final_test")),
        ("7+HLT stacked logistic final_test", locked.get("reco7_plus_hlt_stack_final_test")),
    ]:
        acc, ce, n_jets = _metrics_cells(metrics)
        fusion_rows.append([label, acc, ce, n_jets])
    lines.extend(["## Locked Fusion Results", ""])
    lines.extend(_markdown_table(["Result", "Accuracy", "Cross Entropy", "N Jets"], fusion_rows))
    lines.append("")

    simple_rows = []
    simple = locked.get("simple_fusion_baselines", {})
    for family in ["hlt5", "reco7_plus_hlt"]:
        for method, label in FUSION_METHOD_LABELS.items():
            if method == "stacked_logistic_regression":
                continue
            acc, ce, n_jets = _metrics_cells(simple.get(family, {}).get(method))
            simple_rows.append([family, label, acc, ce, n_jets])
    lines.extend(["## Simple Fusion Baselines", ""])
    lines.extend(_markdown_table(["Family", "Method", "Accuracy", "Cross Entropy", "N Jets"], simple_rows))
    lines.append("")

    interpretation = summary.get("interpretation", {})
    lines.extend(["## HLT5 Vs 7+HLT", ""])
    lines.extend(
        _markdown_table(
            ["Delta", "Value"],
            [
                ["Accuracy, 7+HLT minus HLT5", _fmt_metric(interpretation.get("accuracy_delta_reco7_minus_hlt5"))],
                ["Cross entropy, 7+HLT minus HLT5", _fmt_metric(interpretation.get("cross_entropy_delta_reco7_minus_hlt5"))],
            ],
        )
    )
    lines.append("")

    audit_rows = []
    audit_outcomes = summary.get("audit_outcomes", {})
    for family in ["reco7_plus_hlt", "hlt5_seed_control"]:
        status = audit_outcomes.get(family)
        if status is None:
            audit_rows.append([family, "missing", "missing"])
            continue
        item_text = ", ".join(
            f"{name}={ok}" for name, ok in sorted(status.get("audit_items", {}).items())
        )
        audit_rows.append([family, str(status.get("ok")), item_text])
    lines.extend(["## Audit Outcomes", ""])
    lines.extend(_markdown_table(["Family", "OK", "Items"], audit_rows))
    lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            str(summary.get("discipline_note")),
            "",
            "The offline teacher is reported as an upper reference only and is not used as a fusion feature.",
            "",
        ]
    )
    return "\n".join(lines)


def write_final_report(config: FinalReportConfig) -> Dict[str, Any]:
    """Build and save the Step 13 JSON and Markdown reports."""

    summary = build_final_report_summary(config)
    markdown = render_final_report_markdown(summary)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / config.json_filename
    markdown_path = output_dir / config.markdown_filename
    save_json(json_path, summary)
    markdown_path.write_text(markdown, encoding="utf-8")
    return {
        "summary": summary,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }

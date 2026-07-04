"""Ablation reports for the AV10 architecture-view study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.set_matching.train import source_metadata

from .config import (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PFN_CONTEXT_REPEAT,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PCNN_CONTEXT,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
    architecture_view_variant_spec,
    normalize_architecture_view_variant,
)


ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_STEP = "architecture_view_10class_ablation_step5_report"
ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_CONTRACT = "architecture_view_10class_ablation_report_v1"
ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_SPLITS = ("model_val", "stack_val", "final_test")
ARCHITECTURE_VIEW_10CLASS_CORE_FUSION_GROUP = "av10_architecture_view_core"
ARCHITECTURE_VIEW_10CLASS_SCALAR_FUSION_MODE = "scalar_weighted_logit_mean"


ABLATION_LABELS = {
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK: "A0 HLT ParT baseline",
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART: "A1 larger vanilla HLT ParT",
    ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK: "A2 extra ParT block",
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER: "A3 ParT-only adapter",
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER: "A4 feature MLP adapter",
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES: "A5 LC MLP Delta feature-input adapter",
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE: "A6 wider/deeper feature MLP adapter",
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER: "A7 frozen-ParT adapter",
    ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER: "A8 shuffled-feature control",
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT: "A9 PCNN-context repeat",
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PFN_CONTEXT_REPEAT: "A10 PFN-context repeat",
}
OFFLINE_LABELS = {
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE: "O0 offline ParT baseline",
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER: "O1 offline feature MLP adapter",
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PCNN_CONTEXT: "O2 offline PCNN-context adapter",
}


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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return _jsonable(value)


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
    return output if math.isfinite(output) else None


def _metrics_for_split(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    if split == "model_val":
        keys = ("best_model_val_metrics", "model_val_metrics")
    elif split == "stack_val":
        keys = ("stack_val_metrics", "best_stack_val_metrics")
    elif split == "final_test":
        keys = ("final_test_metrics",)
    else:
        keys = (f"{split}_metrics",)
    for key in keys:
        value = report.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _metric(metrics: Mapping[str, Any] | None, name: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    value = _float(metrics.get(name))
    if value is not None:
        return value
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping):
        return _float(binary.get(name))
    return None


def _load_variant_reports(root: Path | None, variants: Sequence[str], problems: list[str], *, label: str) -> dict[str, Mapping[str, Any]]:
    reports: dict[str, Mapping[str, Any]] = {}
    if root is None:
        return reports
    for variant in variants:
        canonical = normalize_architecture_view_variant(variant)
        path = root / canonical / "run_report.json"
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            problems.append(f"missing or invalid {label} run_report for {canonical}: {path}")
            continue
        enriched = dict(payload)
        enriched["_run_report_path"] = str(path)
        reports[canonical] = enriched
    return reports


def _parameter_row(variant: str, report: Mapping[str, Any], *, family: str, baseline_total: float | None) -> dict[str, Any]:
    accounting = report.get("parameter_accounting")
    if not isinstance(accounting, Mapping):
        accounting = {}
    total_params = _float(accounting.get("total_params"))
    spec = architecture_view_variant_spec(variant)
    return {
        "family": family,
        "variant": variant,
        "display_name": ABLATION_LABELS.get(variant) or OFFLINE_LABELS.get(variant) or variant,
        "adapter_type": spec.adapter_type,
        "parameter_target": spec.parameter_target,
        "total_params": total_params,
        "trainable_params": accounting.get("trainable_params"),
        "part_params": accounting.get("part_params"),
        "trainable_part_params": accounting.get("trainable_part_params"),
        "part_head_params": accounting.get("part_head_params"),
        "trainable_part_head_params": accounting.get("trainable_part_head_params"),
        "adapter_params": accounting.get("adapter_params"),
        "trainable_adapter_params": accounting.get("trainable_adapter_params"),
        "all_adapter_params": accounting.get("all_adapter_params"),
        "dormant_adapter_params": accounting.get("dormant_adapter_params"),
        "active_adapter_module_names": " ".join(str(name) for name in accounting.get("active_adapter_module_names", []))
        if isinstance(accounting.get("active_adapter_module_names"), list)
        else accounting.get("active_adapter_module_names"),
        "ratio_vs_hlt_baseline_total_params": None
        if total_params is None or baseline_total is None or baseline_total <= 0.0
        else total_params / baseline_total,
    }


def _annotate_parameter_match_ratios(rows: list[dict[str, Any]]) -> None:
    reference_by_target = {
        "baseline_part": ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
        "larger_part_capacity_control": ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
        "match_successful_av10_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
        "match_feature_mlp_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
        "current_context_mlp_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
        "input_feature_repair_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
        "scaled_feature_mlp_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
        "offline_baseline_part": ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
        "offline_feature_mlp_adapter": ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
        "offline_pcnn_context_adapter": ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
    }
    row_by_family_variant = {
        (str(row.get("family")), str(row.get("variant"))): row
        for row in rows
    }
    for row in rows:
        target = str(row.get("parameter_target") or "")
        reference_variant = reference_by_target.get(target)
        if not reference_variant:
            continue
        reference = row_by_family_variant.get((str(row.get("family")), str(reference_variant)))
        if reference is None and str(row.get("family")) == "offline_transfer":
            reference = row_by_family_variant.get(("offline_transfer", str(reference_variant)))
        if reference is None:
            row["parameter_match_reference_variant"] = str(reference_variant)
            row["parameter_match_reference_missing"] = True
            continue
        row["parameter_match_reference_variant"] = str(reference.get("variant"))
        for field in ("adapter_params", "total_params"):
            value = _float(row.get(field))
            reference_value = _float(reference.get(field))
            key = field.replace("_params", "_param_ratio_vs_match_reference")
            row[key] = None if value is None or reference_value is None or reference_value <= 0.0 else value / reference_value
        adapter_ratio = _float(row.get("adapter_param_ratio_vs_match_reference"))
        row["adapter_param_match_within_20pct"] = (
            None if adapter_ratio is None else bool(0.8 <= adapter_ratio <= 1.25)
        )


def _diagnostic_rows(variant: str, report: Mapping[str, Any], *, family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_SPLITS:
        metrics = _metrics_for_split(report, split)
        diagnostics = metrics.get("diagnostics") if isinstance(metrics, Mapping) else None
        if not isinstance(diagnostics, Mapping):
            continue
        for key, value in diagnostics.items():
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "split": split,
                    "diagnostic": key,
                    "value": _float(value),
                    "raw_value": value,
                }
            )
    curves = _training_curves_for_report(report)
    epochs = curves.get("epochs") if isinstance(curves, Mapping) else None
    if isinstance(epochs, Sequence) and not isinstance(epochs, (str, bytes)):
        for row in epochs:
            if not isinstance(row, Mapping):
                continue
            epoch = row.get("epoch")
            for phase in ("train", "model_val"):
                metrics = row.get(phase)
                if not isinstance(metrics, Mapping):
                    continue
                diagnostics = metrics.get("diagnostics")
                if not isinstance(diagnostics, Mapping):
                    continue
                for key, value in diagnostics.items():
                    rows.append(
                        {
                            "family": family,
                            "variant": variant,
                            "split": phase,
                            "epoch": epoch,
                            "diagnostic": key,
                            "value": _float(value),
                            "raw_value": value,
                            "source": "training_curves",
                        }
                    )
    return rows


def _training_curves_for_report(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates: list[Path] = []
    path_value = report.get("training_curves")
    if isinstance(path_value, str) and path_value.strip():
        candidate = Path(path_value)
        candidates.append(candidate)
        run_report_path = report.get("_run_report_path")
        if isinstance(run_report_path, str) and not candidate.is_absolute():
            candidates.append(Path(run_report_path).parent / candidate)
    run_report_path = report.get("_run_report_path")
    if isinstance(run_report_path, str):
        candidates.append(Path(run_report_path).parent / "training_curves.json")
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        payload = _read_json(candidate)
        if isinstance(payload, Mapping):
            return payload
    return None


def _model_row(
    variant: str,
    report: Mapping[str, Any],
    *,
    family: str,
    baseline_accuracy: float | None,
) -> dict[str, Any]:
    final_metrics = _metrics_for_split(report, "final_test")
    val_metrics = _metrics_for_split(report, "model_val")
    accuracy = _metric(final_metrics, "accuracy")
    ce = _metric(final_metrics, "cross_entropy")
    if ce is None:
        ce = _metric(final_metrics, "ce_loss")
    return {
        "family": family,
        "variant": variant,
        "display_name": ABLATION_LABELS.get(variant) or OFFLINE_LABELS.get(variant) or variant,
        "input_source": report.get("input_source") or report.get("inference_input_source"),
        "final_test_accuracy": accuracy,
        "delta_vs_baseline": None if accuracy is None or baseline_accuracy is None else accuracy - baseline_accuracy,
        "cross_entropy": ce,
        "macro_per_class_accuracy": _metric(final_metrics, "macro_per_class_accuracy"),
        "model_val_accuracy": _metric(val_metrics, "accuracy"),
        "best_epoch": report.get("best_epoch"),
        "epochs_completed": report.get("epochs_completed"),
        "checkpoint": report.get("checkpoint"),
        "variant_behavior": report.get("variant_behavior"),
        "parameter_accounting": report.get("parameter_accounting"),
    }


def _best_model_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [row for row in rows if _float(row.get("final_test_accuracy")) is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            _float(row.get("final_test_accuracy")) or float("-inf"),
            -(_float(row.get("cross_entropy")) or 1.0e9),
        ),
    )


def _fusion_selected_payload(mode_report: Mapping[str, Any], members: Sequence[str]) -> dict[str, Any]:
    fit = mode_report.get("fit")
    if not isinstance(fit, Mapping):
        return {}
    if "weights" in fit:
        weights = fit["weights"]
        if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes, bytearray)):
            return {"weights": weights, "weighted_members": list(members)}
    if "temperatures" in fit:
        return {"temperatures": fit["temperatures"], "temperature_members": list(members)}
    if "selected_C" in fit:
        return {"selected_C": fit["selected_C"], "model": fit.get("model")}
    return {}


def _fusion_rows(fusion_report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(fusion_report, Mapping):
        return rows
    for group_name, group in fusion_report.get("groups", {}).items():
        if not isinstance(group, Mapping):
            continue
        members = tuple(str(name) for name in group.get("model_names", ()))
        for mode, mode_report in group.get("fusion_modes", {}).items():
            if not isinstance(mode_report, Mapping):
                continue
            metrics = mode_report.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            stack_val = metrics.get("stack_val") if isinstance(metrics.get("stack_val"), Mapping) else {}
            final_test = metrics.get("final_test") if isinstance(metrics.get("final_test"), Mapping) else {}
            rows.append(
                {
                    "group": group_name,
                    "mode": mode,
                    "included_models": list(members),
                    "stack_val_accuracy": _metric(stack_val, "accuracy"),
                    "final_test_accuracy": _metric(final_test, "accuracy"),
                    "final_test_cross_entropy": _metric(final_test, "cross_entropy"),
                    "selected_payload": _fusion_selected_payload(mode_report, members),
                }
            )
    return rows


def _best_fusion_row(rows: Sequence[Mapping[str, Any]], *, group: str | None = None, mode: str | None = None) -> Mapping[str, Any] | None:
    filtered = [
        row
        for row in rows
        if _float(row.get("final_test_accuracy")) is not None
        and (group is None or row.get("group") == group)
        and (mode is None or row.get("mode") == mode)
    ]
    if not filtered:
        return None
    return max(filtered, key=lambda row: _float(row.get("final_test_accuracy")) or float("-inf"))


def _ratio_text(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.6g}"


def _decision_answer(condition: bool | None, yes: str, no: str, missing: str = "unavailable") -> dict[str, Any]:
    if condition is None:
        return {"answer": missing, "satisfied": None}
    return {"answer": yes if condition else no, "satisfied": bool(condition)}


def _decision_summary(
    *,
    hlt_rows: Sequence[Mapping[str, Any]],
    fusion_rows: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_variant = {str(row.get("variant")): row for row in hlt_rows}
    offline_by_variant = {str(row.get("variant")): row for row in offline_rows}

    baseline = _float(by_variant.get(ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK, {}).get("final_test_accuracy"))
    feature = _float(by_variant.get(ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER, {}).get("final_test_accuracy"))
    input_delta = _float(by_variant.get(ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES, {}).get("final_test_accuracy"))
    larger = _float(by_variant.get(ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART, {}).get("final_test_accuracy"))
    extra = _float(by_variant.get(ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK, {}).get("final_test_accuracy"))
    part_only = _float(by_variant.get(ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER, {}).get("final_test_accuracy"))
    frozen = _float(by_variant.get(ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER, {}).get("final_test_accuracy"))
    shuffled = _float(by_variant.get(ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER, {}).get("final_test_accuracy"))
    pcnn = _float(by_variant.get(ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT, {}).get("final_test_accuracy"))
    pfn = _float(by_variant.get(ARCHITECTURE_VIEW_10CLASS_ABLATION_PFN_CONTEXT_REPEAT, {}).get("final_test_accuracy"))
    offline_base = _float(offline_by_variant.get(ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE, {}).get("final_test_accuracy"))
    offline_feature = _float(offline_by_variant.get(ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER, {}).get("final_test_accuracy"))
    core_scalar = _best_fusion_row(
        fusion_rows,
        group=ARCHITECTURE_VIEW_10CLASS_CORE_FUSION_GROUP,
        mode=ARCHITECTURE_VIEW_10CLASS_SCALAR_FUSION_MODE,
    )
    core_scalar_acc = None if core_scalar is None else _float(core_scalar.get("final_test_accuracy"))

    feature_gain = None if baseline is None or feature is None else feature - baseline
    input_delta_gain = None if baseline is None or input_delta is None else input_delta - baseline
    larger_gain = None if baseline is None or larger is None else larger - baseline
    extra_gain = None if baseline is None or extra is None else extra - baseline
    shuffled_gain = None if baseline is None or shuffled is None else shuffled - baseline
    frozen_gain = None if baseline is None or frozen is None else frozen - baseline
    offline_gain = None if offline_base is None or offline_feature is None else offline_feature - offline_base

    target_gain = feature_gain if feature_gain is not None and feature_gain > 0.0 else None
    capacity_match_larger = None if target_gain is None or larger_gain is None else larger_gain >= 0.8 * target_gain
    capacity_match_extra = None if target_gain is None or extra_gain is None else extra_gain >= 0.8 * target_gain
    shuffled_fails = None if shuffled_gain is None or target_gain is None else shuffled_gain < 0.25 * target_gain
    frozen_improves = None if frozen_gain is None else frozen_gain > 0.0
    feature_beats_part_only = None if feature is None or part_only is None else feature > part_only
    feature_beats_extra = None if feature is None or extra is None else feature > extra
    input_delta_works = None if input_delta_gain is None else input_delta_gain > 0.0
    input_delta_matches_feature = None if input_delta is None or feature is None or target_gain is None else (
        input_delta >= feature - 0.2 * target_gain
    )
    pcnn_or_pfn_strong = None if baseline is None or (pcnn is None and pfn is None) else max(pcnn or -1.0, pfn or -1.0) > baseline
    core_fusion_beats_feature = None if core_scalar_acc is None or feature is None else core_scalar_acc > feature
    offline_works = None if offline_gain is None else offline_gain > 0.0

    capacity_evidence = any(value is True for value in (capacity_match_larger, capacity_match_extra)) or (
        shuffled_fails is False
    )
    feature_conditioned_evidence = (
        feature_gain is not None
        and feature_gain > 0.0
        and feature_beats_extra is True
        and shuffled_fails is True
        and (feature_beats_part_only is True or feature_beats_part_only is None)
    )
    input_feature_repair_evidence = input_delta_works is True and input_delta_matches_feature is True
    architecture_view_evidence = pcnn_or_pfn_strong is True and core_fusion_beats_feature is True
    hlt_specific_evidence = feature_gain is not None and feature_gain > 0.0 and offline_works is False

    if capacity_evidence and not feature_conditioned_evidence:
        hypothesis = "capacity_or_extra_compute"
    elif input_feature_repair_evidence and (input_delta is not None and feature is not None and input_delta >= feature):
        hypothesis = "input_feature_repair_before_part"
    elif architecture_view_evidence:
        hypothesis = "architecture_view_complementarity"
    elif feature_conditioned_evidence:
        hypothesis = "feature_conditioned_embedding_adapter"
    elif hlt_specific_evidence:
        hypothesis = "hlt_specific_repair"
    else:
        hypothesis = "mixed_or_inconclusive"

    decisions = {
        "did_larger_part_close_gap": _decision_answer(
            capacity_match_larger,
            "Yes. The larger vanilla ParT recovered most of A4's gain.",
            "No. The larger vanilla ParT did not recover most of A4's gain.",
        ),
        "did_extra_part_block_close_gap": _decision_answer(
            capacity_match_extra,
            "Yes. Extra post-embedding Transformer compute recovered most of A4's gain.",
            "No. Extra post-embedding Transformer compute did not recover most of A4's gain.",
        ),
        "did_frozen_adapter_improve": _decision_answer(
            frozen_improves,
            "Yes. The adapter improved even with ParT frozen.",
            "No. The frozen-ParT adapter did not improve over baseline.",
        ),
        "did_shuffled_control_fail": _decision_answer(
            shuffled_fails,
            "Yes. The shuffled-feature control failed relative to the semantic feature adapter.",
            "No. The shuffled-feature control improved too much to dismiss capacity/regularization.",
        ),
        "did_lc_mlp_delta_input_repair_work": _decision_answer(
            input_delta_works,
            "Yes. The LC MLP Delta input-feature repair beat the HLT baseline.",
            "No. The LC MLP Delta input-feature repair did not beat the HLT baseline.",
        ),
        "did_input_repair_match_embedding_repair": _decision_answer(
            input_delta_matches_feature,
            "Yes. Input-feature repair matched most of the embedding-space feature adapter gain.",
            "No. Input-feature repair did not match the embedding-space feature adapter gain.",
        ),
        "did_offline_transfer_work": _decision_answer(
            offline_works,
            "Yes. The offline feature adapter beat the offline ParT baseline.",
            "No. The offline feature adapter did not beat the offline ParT baseline.",
            missing="offline transfer unavailable",
        ),
        "which_hypothesis_is_most_consistent": {
            "answer": hypothesis,
            "satisfied": None,
        },
    }
    return {
        "decisions": decisions,
        "evidence_values": {
            "hlt_baseline_accuracy": baseline,
            "feature_adapter_accuracy": feature,
            "feature_adapter_gain": feature_gain,
            "lc_mlp_delta_accuracy": input_delta,
            "lc_mlp_delta_gain": input_delta_gain,
            "larger_part_gain": larger_gain,
            "extra_part_block_gain": extra_gain,
            "part_only_accuracy": part_only,
            "shuffled_feature_gain": shuffled_gain,
            "frozen_adapter_gain": frozen_gain,
            "pcnn_context_accuracy": pcnn,
            "pfn_context_accuracy": pfn,
            "core_scalar_fusion_accuracy": core_scalar_acc,
            "offline_baseline_accuracy": offline_base,
            "offline_feature_adapter_accuracy": offline_feature,
            "offline_feature_adapter_gain": offline_gain,
        },
    }


def _write_decision_text(path: Path, decision_summary: Mapping[str, Any]) -> None:
    lines = ["Architecture-View 10-Class Ablation Decisions", ""]
    for key, payload in decision_summary.get("decisions", {}).items():
        if isinstance(payload, Mapping):
            lines.append(f"- {key}: {payload.get('answer')}")
    values = decision_summary.get("evidence_values")
    if isinstance(values, Mapping):
        lines.extend(["", "Evidence Values", ""])
        for key, value in values.items():
            lines.append(f"- {key}: {_ratio_text(_float(value))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report.get("summary", {})
    decisions = report.get("interpretation_summary", {}).get("decisions", {})
    lines = [
        "# Architecture-View 10-Class Ablation Report",
        "",
        f"- ok: {report.get('ok')}",
        f"- best HLT ablation: `{summary.get('best_hlt_variant')}` ({summary.get('best_hlt_accuracy')})",
        f"- HLT baseline: `{summary.get('hlt_baseline_variant')}` ({summary.get('hlt_baseline_accuracy')})",
        f"- best fusion: `{summary.get('best_fusion_group')}` / `{summary.get('best_fusion_mode')}` ({summary.get('best_fusion_accuracy')})",
        f"- best offline transfer: `{summary.get('best_offline_variant')}` ({summary.get('best_offline_accuracy')})",
        "",
        "## Decision Answers",
        "",
    ]
    for key, payload in decisions.items():
        if isinstance(payload, Mapping):
            lines.append(f"- **{key}**: {payload.get('answer')}")
    lines.extend(["", "## Outputs", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    problems = report.get("problems") or []
    if problems:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in problems)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ArchitectureView10ClassAblationReportConfig:
    """Configuration for the AV10 ablation report builder."""

    output_dir: str
    hlt_tagger_root: str
    hlt_variants: tuple[str, ...] = ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS
    hlt_baseline_variant: str = ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK
    fusion_report: str | None = None
    offline_tagger_root: str | None = None
    offline_variants: tuple[str, ...] = ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS
    offline_baseline_variant: str = ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE
    require_fusion: bool = False
    require_offline_transfer: bool = False
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if not str(self.output_dir):
            raise ValueError("output_dir is required")
        if not str(self.hlt_tagger_root):
            raise ValueError("hlt_tagger_root is required")
        if not bool(self.confirm_final_test):
            raise ValueError("Refusing AV10 ablation report without confirm_final_test=True")
        object.__setattr__(self, "hlt_variants", tuple(normalize_architecture_view_variant(v) for v in self.hlt_variants))
        object.__setattr__(self, "offline_variants", tuple(normalize_architecture_view_variant(v) for v in self.offline_variants))
        object.__setattr__(self, "hlt_baseline_variant", normalize_architecture_view_variant(self.hlt_baseline_variant))
        object.__setattr__(self, "offline_baseline_variant", normalize_architecture_view_variant(self.offline_baseline_variant))
        if self.fusion_report is not None:
            object.__setattr__(self, "fusion_report", str(self.fusion_report))
        if self.offline_tagger_root is not None:
            object.__setattr__(self, "offline_tagger_root", str(self.offline_tagger_root))


def build_architecture_view_10class_ablation_report(
    config: ArchitectureView10ClassAblationReportConfig,
) -> dict[str, Any]:
    """Build the Step 5 AV10 ablation report."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    hlt_reports = _load_variant_reports(
        Path(config.hlt_tagger_root),
        config.hlt_variants,
        problems,
        label="HLT ablation",
    )
    fusion_payload = _read_json(Path(config.fusion_report)) if config.fusion_report else None
    if config.fusion_report and not isinstance(fusion_payload, Mapping):
        problems.append(f"missing or invalid fusion report: {config.fusion_report}")
    elif config.require_fusion and not isinstance(fusion_payload, Mapping):
        problems.append("require_fusion=True but no fusion report was provided")

    offline_reports: dict[str, Mapping[str, Any]] = {}
    if config.offline_tagger_root:
        offline_reports = _load_variant_reports(
            Path(config.offline_tagger_root),
            config.offline_variants,
            problems,
            label="offline transfer",
        )
    elif config.require_offline_transfer:
        problems.append("require_offline_transfer=True but no offline_tagger_root was provided")

    hlt_baseline_report = hlt_reports.get(config.hlt_baseline_variant)
    hlt_baseline_acc = _metric(_metrics_for_split(hlt_baseline_report, "final_test") if hlt_baseline_report else None, "accuracy")
    hlt_baseline_params = None
    if isinstance(hlt_baseline_report, Mapping):
        accounting = hlt_baseline_report.get("parameter_accounting")
        if isinstance(accounting, Mapping):
            hlt_baseline_params = _float(accounting.get("total_params"))

    hlt_rows = [
        _model_row(variant, report, family="hlt_ablation", baseline_accuracy=hlt_baseline_acc)
        for variant, report in hlt_reports.items()
    ]
    hlt_rows = sorted(
        hlt_rows,
        key=lambda row: (
            row.get("variant") not in config.hlt_variants,
            config.hlt_variants.index(row.get("variant")) if row.get("variant") in config.hlt_variants else 999,
        ),
    )

    offline_baseline_report = offline_reports.get(config.offline_baseline_variant)
    offline_baseline_acc = _metric(
        _metrics_for_split(offline_baseline_report, "final_test") if offline_baseline_report else None,
        "accuracy",
    )
    offline_rows = [
        _model_row(variant, report, family="offline_transfer", baseline_accuracy=offline_baseline_acc)
        for variant, report in offline_reports.items()
    ]
    offline_rows = sorted(
        offline_rows,
        key=lambda row: (
            row.get("variant") not in config.offline_variants,
            config.offline_variants.index(row.get("variant")) if row.get("variant") in config.offline_variants else 999,
        ),
    )

    hlt_feature_gain = next(
        (row.get("delta_vs_baseline") for row in hlt_rows if row.get("variant") == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER),
        None,
    )
    for row in offline_rows:
        offline_gain = _float(row.get("delta_vs_baseline"))
        hlt_gain = _float(hlt_feature_gain)
        row["delta_compared_to_hlt_feature_adapter_gain"] = None if offline_gain is None or hlt_gain is None else offline_gain - hlt_gain

    fusion_rows = _fusion_rows(fusion_payload)
    best_hlt = _best_model_row(hlt_rows)
    best_offline = _best_model_row(offline_rows)
    best_fusion = _best_fusion_row(fusion_rows)
    parameter_rows = [
        _parameter_row(variant, report, family="hlt_ablation", baseline_total=hlt_baseline_params)
        for variant, report in hlt_reports.items()
    ] + [
        _parameter_row(variant, report, family="offline_transfer", baseline_total=hlt_baseline_params)
        for variant, report in offline_reports.items()
    ]
    _annotate_parameter_match_ratios(parameter_rows)
    diagnostic_rows: list[dict[str, Any]] = []
    for variant, report in hlt_reports.items():
        diagnostic_rows.extend(_diagnostic_rows(variant, report, family="hlt_ablation"))
    for variant, report in offline_reports.items():
        diagnostic_rows.extend(_diagnostic_rows(variant, report, family="offline_transfer"))

    interpretation = _decision_summary(hlt_rows=hlt_rows, fusion_rows=fusion_rows, offline_rows=offline_rows)
    if config.confirm_final_test:
        missing_final = [
            row["variant"]
            for row in (*hlt_rows, *offline_rows)
            if _float(row.get("final_test_accuracy")) is None
        ]
        if missing_final:
            problems.append(f"confirm_final_test=True but final_test accuracy is missing for: {missing_final}")
    if hlt_baseline_report is None:
        problems.append(f"HLT baseline variant was not found: {config.hlt_baseline_variant}")
    if config.require_offline_transfer and offline_baseline_report is None:
        problems.append(f"offline baseline variant was not found: {config.offline_baseline_variant}")

    outputs = {
        "report_json": str(output_dir / "architecture_view_10class_ablation_report.json"),
        "report_markdown": str(output_dir / "architecture_view_10class_ablation_report.md"),
        "hlt_ablation_table_csv": str(output_dir / "hlt_ablation_table.csv"),
        "fusion_complementarity_table_csv": str(output_dir / "fusion_complementarity_table.csv"),
        "offline_transfer_table_csv": str(output_dir / "offline_transfer_table.csv"),
        "parameter_accounting_csv": str(output_dir / "parameter_accounting.csv"),
        "diagnostics_csv": str(output_dir / "diagnostics.csv"),
        "decision_summary_txt": str(output_dir / "decision_summary.txt"),
        "run_report": str(output_dir / "run_report.json"),
    }
    report = {
        "contract": ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_CONTRACT,
        "step": ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_STEP,
        "ok": not problems,
        "problems": problems,
        "config": asdict(config),
        "source": source_metadata(),
        "summary": {
            "hlt_baseline_variant": config.hlt_baseline_variant,
            "hlt_baseline_accuracy": hlt_baseline_acc,
            "best_hlt_variant": None if best_hlt is None else best_hlt.get("variant"),
            "best_hlt_accuracy": None if best_hlt is None else best_hlt.get("final_test_accuracy"),
            "best_fusion_group": None if best_fusion is None else best_fusion.get("group"),
            "best_fusion_mode": None if best_fusion is None else best_fusion.get("mode"),
            "best_fusion_accuracy": None if best_fusion is None else best_fusion.get("final_test_accuracy"),
            "offline_baseline_variant": config.offline_baseline_variant,
            "offline_baseline_accuracy": offline_baseline_acc,
            "best_offline_variant": None if best_offline is None else best_offline.get("variant"),
            "best_offline_accuracy": None if best_offline is None else best_offline.get("final_test_accuracy"),
        },
        "hlt_ablation_rows": hlt_rows,
        "fusion_rows": fusion_rows,
        "offline_transfer_rows": offline_rows,
        "parameter_accounting_rows": parameter_rows,
        "diagnostic_rows": diagnostic_rows,
        "interpretation_summary": interpretation,
        "outputs": outputs,
    }
    _write_csv(Path(outputs["hlt_ablation_table_csv"]), hlt_rows)
    _write_csv(Path(outputs["fusion_complementarity_table_csv"]), fusion_rows)
    _write_csv(Path(outputs["offline_transfer_table_csv"]), offline_rows)
    _write_csv(Path(outputs["parameter_accounting_csv"]), parameter_rows)
    _write_csv(Path(outputs["diagnostics_csv"]), diagnostic_rows)
    _write_decision_text(Path(outputs["decision_summary_txt"]), interpretation)
    save_json(Path(outputs["report_json"]), _jsonable(report))
    save_json(Path(outputs["run_report"]), _jsonable(report))
    _write_markdown(Path(outputs["report_markdown"]), report)
    return _jsonable(report)


__all__ = [
    "ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_CONTRACT",
    "ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_STEP",
    "ARCHITECTURE_VIEW_10CLASS_CORE_FUSION_GROUP",
    "ARCHITECTURE_VIEW_10CLASS_SCALAR_FUSION_MODE",
    "ArchitectureView10ClassAblationReportConfig",
    "build_architecture_view_10class_ablation_report",
]

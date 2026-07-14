"""Strict Step 9 campaign tables and provenance audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cache import (
    HIERARCHY_TARGET_EXPECTED_HLT_PROFILE,
    HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION,
    HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH,
)
from .posthoc import COARSE_TO_FINE_FUSION_CONTRACT, REQUIRED_FUSION_GROUPS


COARSE_TO_FINE_REPORT_CONTRACT = "constrained_coarse_to_fine_step9_campaign_report_v1"
REQUIRED_RECONSTRUCTOR_RUNS = (
    *(f"B{index}" for index in range(8)),
    *(f"C{index}" for index in range(7)),
    "C5-B1",
    "C5-B2",
    "C5-B3",
    "C5-no-slot",
    "Cdirect-unconstrained",
)
REQUIRED_TAGGER_RUNS = (
    "A0",
    *(f"D{index}" for index in range(9)),
    "D5-B1",
    "D5-B2",
    "D5-B3",
    *(
        f"{run_id}-seed{seed}"
        for run_id in (*tuple(f"D{index}" for index in range(9)), "D5-B1", "D5-B2")
        for seed in (1, 2)
    ),
    *(f"E{index}" for index in range(7)),
)
REPORT_SPLITS = ("model_val", "stack_val", "final_test")


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, Mapping) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(name) for row in rows for name in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric(metrics: Mapping[str, Any] | None, *names: str) -> float | None:
    if not isinstance(metrics, Mapping) or metrics.get("available") is False:
        return None
    for name in names:
        value = metrics.get(name)
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            continue
    return None


@dataclass(frozen=True)
class CampaignReportConfig:
    campaign_root: str
    prediction_dir: str
    output_dir: str
    fusion_report_path: str
    reconstructor_runs: tuple[str, ...] = REQUIRED_RECONSTRUCTOR_RUNS
    tagger_runs: tuple[str, ...] = REQUIRED_TAGGER_RUNS
    required_fusion_groups: tuple[str, ...] = REQUIRED_FUSION_GROUPS
    baseline_run_id: str = "A0"
    capacity_run_id: str = "A1"
    offline_run_id: str = "A2"
    require_all_runs: bool = True
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if not self.reconstructor_runs or not self.tagger_runs:
            raise ValueError("campaign reports require non-empty reconstructor and tagger run lists")


def _training_report_path(root: Path, run_id: str) -> Path | None:
    candidates = (
        root / "reconstructors" / run_id / "run_report.json",
        root / "taggers" / run_id / "run_report.json",
        root / "runs" / run_id / "run_report.json",
        root / run_id / "run_report.json",
    )
    return next((path for path in candidates if path.exists()), None)


def _prediction_metadata(prediction_dir: Path, run_id: str, split: str) -> Mapping[str, Any] | None:
    return _read_json(prediction_dir / run_id / f"{split}_predictions_metadata.json")


def _reconstruction_row(run_id: str, report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report.get("best_model_val") or report.get("best_model_val_metrics") or {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    row: dict[str, Any] = {
        "run_id": run_id,
        "tier": run_id[0],
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "best_epoch": report.get("best_epoch"),
        "selection_metric": report.get("selection_metric"),
    }
    for name, value in metrics.items():
        if isinstance(value, (int, float, str)):
            row[str(name)] = value
    return row


def _audit_reconstructor_provenance(
    run_id: str,
    report: Mapping[str, Any],
    rows: list[dict[str, Any]],
    problems: list[str],
) -> None:
    provenance = report.get("provenance")
    source_state = report.get("source_state")
    if not isinstance(source_state, Mapping) or not source_state.get("source_commit") or not source_state.get("source_status_hash"):
        problems.append(f"{run_id} lacks source commit/status provenance")
    for split in ("model_train", "model_val"):
        source = provenance.get(split) if isinstance(provenance, Mapping) else None
        if not isinstance(source, Mapping):
            problems.append(f"{run_id} lacks {split} reconstruction provenance")
            continue
        required = (
            "source_manifest_hash",
            "hlt_content_hash",
            "offline_content_hash",
            "target_content_hash",
            "target_builder_version",
            "jet_identity_hash",
        )
        for field in required:
            value = source.get(field)
            rows.append({"run_id": run_id, "split": split, "field": field, "value": value})
            if value in (None, ""):
                problems.append(f"{run_id} {split} lacks required provenance {field}")
        if source.get("hlt_profile") != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE:
            problems.append(f"{run_id} {split} uses the wrong HLT profile")
        if source.get("hlt_profile_version") != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION:
            problems.append(f"{run_id} {split} uses the wrong HLT profile version")
        try:
            strength = float(source.get("hlt_degradation_strength"))
        except (TypeError, ValueError):
            strength = float("nan")
        if abs(strength - HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH) > 1.0e-12:
            problems.append(f"{run_id} {split} does not use HLT-v2 strength 2.5")


def _audit_prediction_provenance(
    run_id: str,
    split: str,
    metadata: Mapping[str, Any],
    rows: list[dict[str, Any]],
    problems: list[str],
) -> None:
    is_offline_reference = run_id == "A2"
    required = ["source_manifest_hash", "jet_identity_hash", "prediction_content_hash", "checkpoint_sha256"]
    required.append("offline_content_hash" if is_offline_reference else "hlt_content_hash")
    for field in required:
        value = metadata.get(field)
        rows.append({"run_id": run_id, "split": split, "field": field, "value": value})
        if value in (None, ""):
            problems.append(f"{run_id}/{split} lacks required prediction provenance {field}")
    if not is_offline_reference:
        if metadata.get("deployable_hlt_only") is not True:
            problems.append(f"{run_id}/{split} is not marked deployable_hlt_only")
        if metadata.get("hlt_profile") != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE:
            problems.append(f"{run_id}/{split} uses the wrong HLT profile")
        if metadata.get("hlt_profile_version") != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION:
            problems.append(f"{run_id}/{split} uses the wrong HLT profile version")
        try:
            strength = float(metadata.get("hlt_degradation_strength"))
        except (TypeError, ValueError):
            strength = float("nan")
        if abs(strength - HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH) > 1.0e-12:
            problems.append(f"{run_id}/{split} does not use HLT-v2 strength 2.5")
    if split == "final_test" and metadata.get("final_test_confirmed") is not True:
        problems.append(f"{run_id} final_test was not explicitly confirmed")
    source_state = metadata.get("source_state")
    if run_id.startswith(("D", "E")):
        if not isinstance(source_state, Mapping) or not source_state.get("source_commit") or not source_state.get("source_status_hash"):
            problems.append(f"{run_id}/{split} lacks source commit/status provenance")


def _alias_audit(
    metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    problems: list[str],
) -> dict[str, Any]:
    result = {
        "D5_checkpoint_sha256": None,
        "D5_B3_checkpoint_sha256": None,
        "D5_configuration_hash": None,
        "D5_B3_configuration_hash": None,
        "shared_configuration": False,
    }
    d5 = metadata.get(("D5", "model_val"))
    depth = metadata.get(("D5-B3", "model_val"))
    if not isinstance(d5, Mapping) or not isinstance(depth, Mapping):
        return result
    result["D5_checkpoint_sha256"] = d5.get("checkpoint_sha256")
    result["D5_B3_checkpoint_sha256"] = depth.get("checkpoint_sha256")
    result["D5_configuration_hash"] = d5.get("configuration_hash")
    result["D5_B3_configuration_hash"] = depth.get("configuration_hash")
    alias_declared = d5.get("alias_of") == "D5-B3" or depth.get("alias_of") == "D5"
    if alias_declared:
        if not d5.get("configuration_hash") or not depth.get("configuration_hash"):
            problems.append("D5/D5-B3 alias declaration lacks shared configuration hashes")
        elif (
            d5.get("checkpoint_sha256") != depth.get("checkpoint_sha256")
            or d5.get("configuration_hash") != depth.get("configuration_hash")
        ):
            problems.append("D5/D5-B3 alias declaration does not share checkpoint/configuration hashes")
        else:
            result["shared_configuration"] = True
    return result


def write_campaign_report(config: CampaignReportConfig) -> dict[str, Any]:
    root = Path(config.campaign_root)
    prediction_dir = Path(config.prediction_dir)
    output_dir = Path(config.output_dir)
    problems: list[str] = []
    provenance_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    training_reports: dict[str, Mapping[str, Any]] = {}
    for run_id in config.reconstructor_runs:
        path = _training_report_path(root, run_id)
        report = _read_json(path) if path is not None else None
        if not isinstance(report, Mapping):
            if config.require_all_runs:
                problems.append(f"missing required reconstructor run {run_id}")
            continue
        if report.get("ok") is False:
            problems.append(f"{run_id} reconstruction run_report has ok=false")
        training_reports[run_id] = report
        reconstruction_rows.append(_reconstruction_row(run_id, report))
        _audit_reconstructor_provenance(run_id, report, provenance_rows, problems)
    tagging_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    d8_rows: list[dict[str, Any]] = []
    prediction_metadata: dict[tuple[str, str], Mapping[str, Any]] = {}
    report_splits = REPORT_SPLITS if config.confirm_final_test else ("model_val", "stack_val")
    for run_id in config.tagger_runs:
        train_path = _training_report_path(root, run_id)
        train_report = _read_json(train_path) if train_path is not None else None
        if config.require_all_runs and not isinstance(train_report, Mapping):
            problems.append(f"missing required tagger run_report for {run_id}")
        elif isinstance(train_report, Mapping) and train_report.get("ok") is False:
            problems.append(f"{run_id} tagger run_report has ok=false")
        for split in report_splits:
            metadata = _prediction_metadata(prediction_dir, run_id, split)
            if not isinstance(metadata, Mapping):
                if config.require_all_runs:
                    problems.append(f"missing required predictions for {run_id}/{split}")
                continue
            prediction_metadata[(run_id, split)] = metadata
            _audit_prediction_provenance(run_id, split, metadata, provenance_rows, problems)
            metrics = metadata.get("metrics")
            if not isinstance(metrics, Mapping) or _metric(metrics, "accuracy") is None:
                problems.append(f"{run_id}/{split} lacks available classification metrics")
                continue
            tagging_rows.append(
                {
                    "run_id": run_id,
                    "tier": run_id[0],
                    "split": split,
                    "accuracy": _metric(metrics, "accuracy"),
                    "cross_entropy": _metric(metrics, "cross_entropy", "loss"),
                    "macro_ovr_auc": _metric(metrics, "macro_ovr_auc"),
                    "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                    "alias_of": metadata.get("alias_of"),
                }
            )
            for row in metrics.get("per_class_accuracy", ()): 
                if isinstance(row, Mapping):
                    per_class_rows.append({"run_id": run_id, "split": split, **row})
            mechanism = metadata.get("mechanism_diagnostics")
            if isinstance(mechanism, Mapping):
                gates = mechanism.get("pooled_gate")
                if isinstance(gates, Mapping):
                    names = gates.get("view_names") or []
                    for index, name in enumerate(names):
                        mechanism_rows.append(
                            {
                                "run_id": run_id,
                                "split": split,
                                "view_name": name,
                                "gate_mean": (gates.get("mean") or [None] * len(names))[index],
                                "gate_p10": (gates.get("p10") or [None] * len(names))[index],
                                "gate_p90": (gates.get("p90") or [None] * len(names))[index],
                                "dominant_fraction": (gates.get("dominant_view_fraction") or [None] * len(names))[index],
                                "uncertainty_gate_correlation": (
                                    mechanism.get("uncertainty_gate_correlation") or [None] * len(names)
                                )[index],
                            }
                        )
            if run_id == "D8" and split == "model_val":
                ablations = metadata.get("d8_model_val_view_ablations")
                if not isinstance(ablations, Mapping) or not ablations:
                    problems.append("D8 is missing required model_val drop-one/view-combination diagnostics")
                else:
                    for name, ablation_metrics in ablations.items():
                        if isinstance(ablation_metrics, Mapping):
                            d8_rows.append(
                                {
                                    "view_mask": name,
                                    "accuracy": _metric(ablation_metrics, "accuracy"),
                                    "cross_entropy": _metric(ablation_metrics, "cross_entropy", "loss"),
                                    "n_jets": _metric(ablation_metrics, "n_jets"),
                                }
                            )
    by_key = {(row["run_id"], row["split"]): row for row in tagging_rows}
    for row in tagging_rows:
        for label, baseline in (
            ("A0", config.baseline_run_id),
            ("A1", config.capacity_run_id),
            ("A2", config.offline_run_id),
        ):
            reference = by_key.get((baseline, row["split"]))
            row[f"accuracy_gap_vs_{label}"] = (
                None if reference is None else row["accuracy"] - reference["accuracy"]
            )
            row[f"cross_entropy_improvement_vs_{label}"] = (
                None
                if reference is None or row["cross_entropy"] is None or reference["cross_entropy"] is None
                else reference["cross_entropy"] - row["cross_entropy"]
            )
    alias_report = _alias_audit(prediction_metadata, problems)
    fusion_report = _read_json(Path(config.fusion_report_path))
    if not isinstance(fusion_report, Mapping):
        problems.append("missing Step 9 fusion report")
        fusion_groups = {}
    else:
        if fusion_report.get("ok") is False or fusion_report.get("contract") != COARSE_TO_FINE_FUSION_CONTRACT:
            problems.append("fusion report is failed or has the wrong contract")
        fusion_groups = fusion_report.get("groups") if isinstance(fusion_report.get("groups"), Mapping) else {}
        missing = sorted(set(config.required_fusion_groups) - set(fusion_groups))
        if missing:
            problems.append(f"fusion report is missing required groups {missing}")
        for group in config.required_fusion_groups:
            row = fusion_groups.get(group)
            claim_split = "final_test" if config.confirm_final_test else "model_val"
            claim = row.get("splits", {}).get(claim_split) if isinstance(row, Mapping) else None
            metrics = claim.get("metrics") if isinstance(claim, Mapping) else None
            if _metric(metrics, "accuracy") is None:
                problems.append(f"fusion group {group} lacks available {claim_split} metrics")
        f2 = fusion_groups.get("F2")
        if isinstance(f2, Mapping) and f2.get("spec", {}).get("method") != "representation_stacker":
            problems.append("F2 is not identified as learned D-tier representation fusion")
        selected_best_d = fusion_report.get("selected_best_d")
        if any(group in fusion_groups for group in ("F0", "F1", "F3", "F4", "F5")):
            if not isinstance(selected_best_d, str) or not selected_best_d:
                problems.append("fusion report does not declare the model_val-selected best D run")
            if fusion_report.get("best_d_selection_metric") != "model_val.cross_entropy":
                problems.append("best D selection is not based on model_val cross-entropy")
        if isinstance(selected_best_d, str) and selected_best_d:
            expected_f4 = (selected_best_d, f"{selected_best_d}-seed1", f"{selected_best_d}-seed2")
            f4 = fusion_groups.get("F4")
            if isinstance(f4, Mapping):
                observed_f4 = tuple(f4.get("spec", {}).get("members", ()))
                if observed_f4 != expected_f4:
                    problems.append(f"F4 is not the selected best-D seed ensemble: {observed_f4}")
            expected_f5 = tuple(
                dict.fromkeys(("D8", "D6", selected_best_d, *expected_f4[1:]))
            )
            f5 = fusion_groups.get("F5")
            if isinstance(f5, Mapping):
                observed_f5 = tuple(f5.get("spec", {}).get("members", ()))
                if observed_f5 != expected_f5:
                    problems.append(f"F5 lacks the planned D8/D6/selected-seed composition: {observed_f5}")
    for split in report_splits:
        for field in ("source_manifest_hash", "hlt_content_hash", "jet_identity_hash"):
            observed = {
                run_id: metadata.get(field)
                for (run_id, row_split), metadata in prediction_metadata.items()
                if row_split == split and run_id != "A2" and metadata.get(field) is not None
            }
            if len(set(observed.values())) > 1:
                problems.append(f"prediction runs disagree on {split}/{field}: {observed}")
    for split in ("model_train", "model_val"):
        for field in (
            "source_manifest_hash",
            "hlt_content_hash",
            "offline_content_hash",
            "target_content_hash",
            "target_builder_version",
            "jet_identity_hash",
        ):
            observed = {
                run_id: report.get("provenance", {}).get(split, {}).get(field)
                for run_id, report in training_reports.items()
                if report.get("provenance", {}).get(split, {}).get(field) is not None
            }
            if len(set(observed.values())) > 1:
                problems.append(f"reconstructor runs disagree on {split}/{field}: {observed}")
    outputs = {
        "tagging_metrics_csv": str(output_dir / "tagging_metrics.csv"),
        "per_class_metrics_csv": str(output_dir / "per_class_metrics.csv"),
        "reconstruction_metrics_csv": str(output_dir / "reconstruction_metrics.csv"),
        "mechanism_diagnostics_csv": str(output_dir / "mechanism_diagnostics.csv"),
        "d8_view_ablations_csv": str(output_dir / "d8_view_ablations.csv"),
        "provenance_audit_csv": str(output_dir / "provenance_audit.csv"),
        "confusion_matrices_json": str(output_dir / "confusion_matrices.json"),
    }
    _write_csv(Path(outputs["tagging_metrics_csv"]), tagging_rows)
    _write_csv(Path(outputs["per_class_metrics_csv"]), per_class_rows)
    _write_csv(Path(outputs["reconstruction_metrics_csv"]), reconstruction_rows)
    _write_csv(Path(outputs["mechanism_diagnostics_csv"]), mechanism_rows)
    _write_csv(Path(outputs["d8_view_ablations_csv"]), d8_rows)
    _write_csv(Path(outputs["provenance_audit_csv"]), provenance_rows)
    _write_json(
        Path(outputs["confusion_matrices_json"]),
        {
            run_id: {
                split: metadata.get("metrics", {}).get("confusion_matrix")
                for (candidate, split), metadata in prediction_metadata.items()
                if candidate == run_id
            }
            for run_id in config.tagger_runs
        },
    )
    report = {
        "ok": not problems,
        "contract": COARSE_TO_FINE_REPORT_CONTRACT,
        "config": asdict(config),
        "problems": problems,
        "required_reconstructor_runs": list(config.reconstructor_runs),
        "required_tagger_runs": list(config.tagger_runs),
        "required_fusion_groups": list(config.required_fusion_groups),
        "present_reconstructor_runs": sorted(training_reports),
        "d5_d5_b3_alias_audit": alias_report,
        "outputs": outputs,
        "fusion_report_path": config.fusion_report_path,
        "final_test_policy": {
            "confirmed": bool(config.confirm_final_test),
            "deployable_predictions_hlt_only": True,
            "selection_split": "model_val",
            "fusion_fit_split": "stack_train",
        },
    }
    _write_json(output_dir / "final_report.json", report)
    return report


__all__ = [
    "COARSE_TO_FINE_REPORT_CONTRACT",
    "REQUIRED_RECONSTRUCTOR_RUNS",
    "REQUIRED_TAGGER_RUNS",
    "REPORT_SPLITS",
    "CampaignReportConfig",
    "write_campaign_report",
]

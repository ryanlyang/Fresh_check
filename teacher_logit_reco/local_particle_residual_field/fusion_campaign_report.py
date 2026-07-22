"""Immutable human- and machine-readable report for the P7b fusion campaign."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from .fusion_campaign import (
    FUSION_GROUP_METHOD, FUSION_GROUP_SEED, default_fusion_candidate_specs, stable_fusion_json_hash,
)
from .fusion_bootstrap_audit import LOCAL_RESIDUAL_FIELD_FUSION_BOOTSTRAP_AUDIT_CONTRACT
from .fusion_metric_audit import LOCAL_RESIDUAL_FIELD_FUSION_METRIC_REPRODUCTION_CONTRACT
from .fusion_final import LOCAL_RESIDUAL_FIELD_FUSION_FINAL_EVALUATION_CONTRACT, _read_json, _validate_selection_dependencies
from .fusion_replay import LOCAL_RESIDUAL_FIELD_FUSION_RECIPE_REPLAY_CONTRACT
from .fusion_runtime import LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_CONTRACT
from .fusion_seed_control import sha256_file
from .fusion_selection import (
    _ranking_multiclass, _rejection_objective, _validate_candidate_report, load_selected_fusion_set,
)


LOCAL_RESIDUAL_FIELD_FUSION_REPORT_CONTRACT = "local_residual_field_fusion_campaign_report_v1"
FINAL_REPORT_FILENAMES = (
    "summary.md", "run_report.json", "provenance_audit.json", "member_metrics.csv",
    "fusion_candidate_stack_val.csv", "selected_fusion_metrics.csv", "paired_group_comparison.csv",
    "binary_rejection.csv", "complementarity.csv", "runtime_metrics.csv", "bootstrap_intervals.csv",
)


def _validated_hashed_json(path: Path, *, contract: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("ok") is not True or payload.get("contract") != contract:
        raise ValueError(f"artifact contract mismatch: {path}")
    unsigned = dict(payload)
    stored = unsigned.pop("artifact_hash", None)
    if stored != stable_fusion_json_hash(unsigned):
        raise ValueError(f"artifact logical hash mismatch: {path}")
    return payload


def _validated_metric_reproduction(path: Path, *, source_artifact_audit_hash: str) -> dict[str, Any]:
    payload = _read_json(path)
    unsigned = dict(payload)
    stored = unsigned.pop("audit_hash", None)
    if payload.get("ok") is not True or payload.get("contract") != LOCAL_RESIDUAL_FIELD_FUSION_METRIC_REPRODUCTION_CONTRACT:
        raise ValueError("raw A0/P7b metric reproduction gate did not pass")
    if stored != stable_fusion_json_hash(unsigned):
        raise ValueError("raw metric reproduction logical hash mismatch")
    if payload.get("source_artifact_audit_hash") != source_artifact_audit_hash:
        raise ValueError("raw metric reproduction is not bound to the selected source audit")
    return payload


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_cell(value) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _metric_columns(metrics: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "n_jets": metrics.get("n_jets"), "accuracy": metrics.get("accuracy"),
        "cross_entropy": metrics.get("cross_entropy"), "macro_one_vs_rest_auc": metrics.get("macro_one_vs_rest_auc"),
        "macro_per_class_accuracy": metrics.get("macro_per_class_accuracy"),
        "expected_calibration_error": metrics.get("expected_calibration_error"), "brier_score": metrics.get("brier_score"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }
    for row in metrics.get("per_class") or ():
        output[f"class_accuracy_{row['class_name']}"] = row.get("accuracy")
        output[f"class_auc_{row['class_name']}"] = row.get("one_vs_rest_auc")
    return output


def _candidate_ranking_columns(report: Mapping[str, Any]) -> dict[str, Any]:
    stability = report.get("head_stability")
    if isinstance(stability, Mapping) and int(stability.get("head_count", 0)) == 3:
        multiclass = stability["stack_val"]["multiclass"]
        rejection = stability["stack_val"]["rejection_objective"]
        ranking = _ranking_multiclass(report)
        return {
            "selection_accuracy": ranking["accuracy"],
            "selection_accuracy_variance": multiclass["accuracy"]["variance"],
            "selection_cross_entropy": ranking["cross_entropy"],
            "selection_cross_entropy_variance": multiclass["cross_entropy"]["variance"],
            "selection_rejection_objective": _rejection_objective(report),
            "selection_rejection_objective_variance": rejection["variance"],
            "selection_rule": (
                "deployed_fixed_seed_5101"
                if report.get("candidate_id") == "R0_linear_embeddings"
                else "mean_per_head_metrics"
            ),
        }
    multiclass = report["metrics"]["stack_val"]["multiclass"]
    return {
        "selection_accuracy": multiclass["accuracy"], "selection_accuracy_variance": None,
        "selection_cross_entropy": multiclass["cross_entropy"], "selection_cross_entropy_variance": None,
        "selection_rejection_objective": None, "selection_rejection_objective_variance": None,
        "selection_rule": "deployment_metrics",
    }


def _require_fusion_identity(row: Mapping[str, Any]) -> None:
    for key in ("run_id", "group_id", "candidate_id"):
        if not str(row.get(key) or "").strip():
            raise ValueError(f"fusion report row has empty {key}: {row}")


def _require_deployable_hlt_only(rows: Iterable[Mapping[str, Any]]) -> None:
    privileged = ("uses_true_fields", "uses_offline_particles", "uses_teacher_logits_at_runtime")
    if any(
        row.get("runtime_inputs") != "HLT_only" or row.get("deployable") is not True
        or any(row.get(key) is not False for key in privileged)
        for row in rows
    ):
        raise ValueError("final report cannot include non-deployable or non-HLT fusion rows")


def _require_current_final_bindings(
    final_sha256: str,
    runtime: Mapping[str, Any],
    bootstrap_audit: Mapping[str, Any],
) -> None:
    if (
        runtime.get("final_evaluation_sha256") != final_sha256
        or bootstrap_audit.get("final_evaluation_sha256") != final_sha256
    ):
        raise ValueError("runtime/bootstrap artifacts are stale relative to final_evaluation.json bytes")


def _candidate_reports(campaign_root: Path, selection: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    union = set(selection["representation_stability_union"])
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for group_id in (FUSION_GROUP_METHOD, FUSION_GROUP_SEED):
        for spec in default_fusion_candidate_specs():
            name = "candidate_stability_report.json" if spec.family == "representation" and spec.candidate_id in union else "candidate_report.json"
            report = _validate_candidate_report(campaign_root / "candidates" / group_id / spec.candidate_id / name)
            if (report["group_id"], report["candidate_id"]) != (group_id, spec.candidate_id):
                raise ValueError("candidate report identity drift")
            output[(group_id, spec.candidate_id)] = report
    return output


def _binary_rows(final: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in final["selected_results"]:
        identity = {key: result[key] for key in ("run_id", "group_id", "candidate_id")}
        _require_fusion_identity(identity)
        for projection, projection_row in result["binary_projection"]["projections"].items():
            for collection_key, convention in (
                ("operating_points", "within_split_matched_efficiency"),
                ("frozen_threshold_operating_points", "stack_val_frozen_threshold"),
            ):
                for efficiency_key, point in (projection_row.get(collection_key) or {}).items():
                    rows.append({
                        **identity, "champion_roles": result["champion_roles"], "projection": projection,
                        "signal_name": projection_row["positive_class_name"], "threshold_convention": convention,
                        "efficiency_key": efficiency_key, **dict(point),
                    })
    return rows


def _bootstrap_rows(final: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in final["selected_results"]:
        identity = {key: result[key] for key in ("run_id", "group_id", "candidate_id")}
        _require_fusion_identity(identity)
        multiclass = result["paired_bootstrap_vs_A0"]
        for metric in ("accuracy_delta_b_minus_a", "cross_entropy_delta_b_minus_a"):
            value = multiclass[metric]
            rows.append({**identity, "comparison": "fusion_vs_A0", "projection": "ten_class", "metric": metric,
                         "estimate": value["estimate"], "bootstrap_mean": value.get("mean"), "interval_95": value["interval_95"],
                         "replicates": multiclass["replicates"], "seed": multiclass["seed"], "sampled_index_hash": multiclass["sampled_index_hash"]})
        for signal, binary in result["paired_binary_bootstrap_vs_A0"].items():
            for metric in ("false_positive_rate_delta_b_minus_a", "log_smoothed_fpr_delta_b_minus_a", "smoothed_rejection_ratio_b_over_a"):
                value = binary[metric]
                rows.append({**identity, "comparison": "fusion_vs_A0", "projection": f"QCD_vs_{signal}", "metric": metric,
                             "estimate": value["estimate"], "interval_95": value["interval_95"],
                             "replicates": binary["replicates"], "seed": binary["seed"], "sampled_index_hash": binary["sampled_index_hash"]})
    return rows


def write_local_residual_field_fusion_campaign_report(selected_fusion_json: str | Path) -> dict[str, Any]:
    """Compose the final report from frozen selection, final, replay, and runtime artifacts."""

    selection_path = Path(selected_fusion_json).resolve()
    selection = load_selected_fusion_set(selection_path)
    audit, _prediction_registry = _validate_selection_dependencies(selection_path, selection)
    campaign_root = selection_path.parent.parent
    final_root = campaign_root / "final_evaluation" / selection["artifact_hash"][:16]
    final_path = final_root / "final_evaluation.json"
    replay_path = selection_path.parent / "recipe_replay" / "recipe_replay.json"
    runtime_path = final_root / "runtime_metrics.json"
    bootstrap_audit_path = final_root / "bootstrap_audit.json"
    metric_audit_path = campaign_root / "metric_reproduction_audit.json"
    final = _validated_hashed_json(final_path, contract=LOCAL_RESIDUAL_FIELD_FUSION_FINAL_EVALUATION_CONTRACT)
    replay = _validated_hashed_json(replay_path, contract=LOCAL_RESIDUAL_FIELD_FUSION_RECIPE_REPLAY_CONTRACT)
    runtime = _validated_hashed_json(runtime_path, contract=LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_CONTRACT)
    bootstrap_audit = _validated_hashed_json(bootstrap_audit_path, contract=LOCAL_RESIDUAL_FIELD_FUSION_BOOTSTRAP_AUDIT_CONTRACT)
    metric_audit = _validated_metric_reproduction(
        metric_audit_path, source_artifact_audit_hash=selection["source_artifact_audit_hash"],
    )
    if final["selected_fusion_artifact_hash"] != selection["artifact_hash"] or runtime["selected_fusion_artifact_hash"] != selection["artifact_hash"]:
        raise ValueError("final/runtime artifacts are not bound to selected_fusion.json")
    if replay["source_selected_fusion_artifact_hash"] != selection["artifact_hash"]:
        raise ValueError("recipe replay is not bound to selected_fusion.json")
    if bootstrap_audit["selected_fusion_artifact_hash"] != selection["artifact_hash"]:
        raise ValueError("bootstrap audit is not bound to selected_fusion.json")
    final_sha256 = sha256_file(final_path)
    _require_current_final_bindings(final_sha256, runtime, bootstrap_audit)
    if replay.get("hyperparameter_search_performed") is not False or replay.get("final_test_opened") is not False:
        raise ValueError("recipe replay was not a frozen development-only control")
    deployment_rows = [
        final, runtime, bootstrap_audit, *final["member_metrics"].values(),
        *final["selected_results"], *runtime["member_rows"], *runtime["fusion_rows"],
    ]
    _require_deployable_hlt_only(deployment_rows)

    candidates = _candidate_reports(campaign_root, selection)
    specs = {spec.candidate_id: spec for spec in default_fusion_candidate_specs()}
    member_rows = [
        {"run_id": f"member/{member}", "member_id": member, **_metric_columns(payload["multiclass"]),
         "checkpoint_hash": payload["checkpoint_hash"], "runtime_inputs": "HLT_only",
         "uses_true_fields": False, "uses_offline_particles": False,
         "uses_teacher_logits_at_runtime": False, "deployable": True}
        for member, payload in sorted(final["member_metrics"].items())
    ]
    candidate_rows: list[dict[str, Any]] = []
    for (group_id, candidate_id), report in candidates.items():
        row = {"run_id": f"{group_id}/{candidate_id}", "group_id": group_id, "candidate_id": candidate_id,
               "family": report["family"], "phase": report["phase"], "head_seeds": report["head_seeds"],
               "trainable_parameter_count": report["trainable_parameter_count"],
               "selected_hyperparameters": report["selected_hyperparameters"],
               **_metric_columns(report["metrics"]["stack_val"]["multiclass"]),
               **_candidate_ranking_columns(report),
               "candidate_report_artifact_hash": report["artifact_hash"], "runtime_inputs": "HLT_only",
               "uses_true_fields": False, "uses_offline_particles": False,
               "uses_teacher_logits_at_runtime": False, "deployable": True}
        _require_fusion_identity(row)
        candidate_rows.append(row)
    final_lookup = {(row["group_id"], row["candidate_id"]): row for row in final["selected_results"]}
    selected_rows: list[dict[str, Any]] = []
    for selected in selection["selections"]:
        result = final_lookup[(selected["group_id"], selected["candidate_id"])]
        row = {"run_id": result["run_id"], "group_id": selected["group_id"], "candidate_id": selected["candidate_id"],
               "champion_role": selected["champion_role"], "member_ids": selected["member_ids"],
               "hyperparameters": selected["hyperparameters"], **_metric_columns(result["multiclass"]),
               "final_test_status": final["final_test_status"], "runtime_inputs": "HLT_only",
               "uses_true_fields": False, "uses_offline_particles": False,
               "uses_teacher_logits_at_runtime": False, "deployable": True}
        _require_fusion_identity(row)
        selected_rows.append(row)
    paired_rows: list[dict[str, Any]] = []
    for candidate_id in specs:
        method = _candidate_ranking_columns(candidates[(FUSION_GROUP_METHOD, candidate_id)])
        seed = _candidate_ranking_columns(candidates[(FUSION_GROUP_SEED, candidate_id)])
        paired_rows.append({"run_id": f"matched/{candidate_id}", "group_id": "F_method_vs_F_seed", "candidate_id": candidate_id,
                            "comparison_view": "family_matched", "F_method_accuracy": method["selection_accuracy"], "F_seed_accuracy": seed["selection_accuracy"],
                            "accuracy_delta_method_minus_seed": method["selection_accuracy"] - seed["selection_accuracy"],
                            "F_method_accuracy_variance": method["selection_accuracy_variance"], "F_seed_accuracy_variance": seed["selection_accuracy_variance"],
                            "F_method_cross_entropy": method["selection_cross_entropy"], "F_seed_cross_entropy": seed["selection_cross_entropy"]})
    for selected in selection["selections"]:
        metrics = selected["selection_metrics"].get(
            "ranking_multiclass", selected["selection_metrics"]["multiclass"],
        )
        paired_rows.append({"run_id": f"champion/{selected['group_id']}/{selected['champion_role']}",
                            "group_id": selected["group_id"], "candidate_id": selected["candidate_id"],
                            "comparison_view": "best_achievable", "champion_role": selected["champion_role"],
                            "stack_val_accuracy": metrics["accuracy"], "stack_val_cross_entropy": metrics["cross_entropy"]})
    replay_metrics = replay["metrics"]["stack_val"]["multiclass"]
    paired_rows.append({"run_id": f"recipe_replay/F_seed/{replay['candidate_id']}", "group_id": FUSION_GROUP_SEED,
                        "candidate_id": replay["candidate_id"], "comparison_view": "F_method_recipe_replay_on_F_seed",
                        "stack_val_accuracy": replay_metrics["accuracy"], "stack_val_cross_entropy": replay_metrics["cross_entropy"],
                        "hyperparameter_search_performed": False})
    for row in paired_rows:
        _require_fusion_identity(row)
    complementarity_rows: list[dict[str, Any]] = []
    for result in final["selected_results"]:
        identity = {key: result[key] for key in ("run_id", "group_id", "candidate_id")}
        comp = result["member_complementarity"]
        complementarity_rows.append({**identity, "class_name": "ALL", **{key: comp.get(key) for key in (
            "member_a", "member_b", "n_jets", "disagreement_count", "disagreement_rate", "both_correct", "a_only_correct",
            "b_only_correct", "both_wrong", "error_overlap_jaccard", "gain_on_a_error_count", "gain_on_a_error_rate",
            "loss_on_a_correct_count", "loss_on_a_correct_rate", "flattened_logit_correlation", "flattened_probability_correlation")}})
        for class_row in comp["per_class"]:
            complementarity_rows.append({**identity, **class_row})
    runtime_rows = [{"row_type": "member", **row} for row in runtime["member_rows"]] + [
        {"row_type": "fusion", **row} for row in runtime["fusion_rows"]
    ]
    for row in runtime["fusion_rows"]:
        _require_fusion_identity(row)
    binary_rows = _binary_rows(final)
    bootstrap_rows = _bootstrap_rows(final)

    raw_best = max(member_rows, key=lambda row: float(row["accuracy"]))
    late_best = max((row for row in candidate_rows if row["group_id"] == FUSION_GROUP_METHOD and row["family"] == "late"), key=lambda row: float(row["selection_accuracy"]))
    representation_best = max((row for row in candidate_rows if row["group_id"] == FUSION_GROUP_METHOD and row["family"] == "representation"), key=lambda row: float(row["selection_accuracy"]))
    seed_best = max((row for row in candidate_rows if row["group_id"] == FUSION_GROUP_SEED), key=lambda row: float(row["selection_accuracy"]))
    g0_fit = _read_json(candidates[(FUSION_GROUP_METHOD, "L0_mean_logits")]["fit_artifacts"][0]["path"])
    g0 = g0_fit.get("g0_development_reproduction")
    if not isinstance(g0, Mapping) or g0.get("ok") is not True:
        raise ValueError("G0 reproduction audit is absent or failed")
    fastest_selected = min(runtime["fusion_rows"], key=lambda row: float(row["median_batch_latency_ms"]))
    method_accuracy_selection = next(
        row for row in selection["selections"]
        if row["group_id"] == FUSION_GROUP_METHOD and row["champion_role"] == "accuracy_champion"
    )
    method_accuracy_final = final_lookup[(FUSION_GROUP_METHOD, method_accuracy_selection["candidate_id"])]
    accuracy_interval = method_accuracy_final["paired_bootstrap_vs_A0"]["accuracy_delta_b_minus_a"]
    method_rejection_selection = next(
        row for row in selection["selections"]
        if row["group_id"] == FUSION_GROUP_METHOD and row["champion_role"] == "rejection_champion"
    )
    method_rejection_final = final_lookup[(FUSION_GROUP_METHOD, method_rejection_selection["candidate_id"])]
    rejection_point = method_rejection_final["binary_projection"]["projections"]["QCD_vs_Hgg"]["operating_points"]["signal_efficiency_0.50"]
    representation_selection_label = (
        "fixed-seed-5101 selection accuracy"
        if representation_best.get("selection_rule") == "deployed_fixed_seed_5101"
        else "three-seed mean selection accuracy"
    )

    report_root = campaign_root / "final_report"
    if report_root.exists():
        if (report_root / "run_report.json").is_file():
            raise FileExistsError(f"refusing to overwrite immutable final report: {report_root}")
        quarantine = report_root.with_name(
            f"{report_root.name}.partial_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        report_root.replace(quarantine)
    report_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".final_report.", dir=str(report_root.parent)))
    try:
        _write_csv(temporary_root / "member_metrics.csv", member_rows)
        _write_csv(temporary_root / "fusion_candidate_stack_val.csv", candidate_rows)
        _write_csv(temporary_root / "selected_fusion_metrics.csv", selected_rows)
        _write_csv(temporary_root / "paired_group_comparison.csv", paired_rows)
        _write_csv(temporary_root / "binary_rejection.csv", binary_rows)
        _write_csv(temporary_root / "complementarity.csv", complementarity_rows)
        _write_csv(temporary_root / "runtime_metrics.csv", runtime_rows)
        _write_csv(temporary_root / "bootstrap_intervals.csv", bootstrap_rows)
        report_input_paths = {
            "selected_fusion": selection_path,
            "final_evaluation": final_path,
            "recipe_replay": replay_path,
            "runtime_metrics": runtime_path,
            "bootstrap_audit": bootstrap_audit_path,
            "metric_reproduction_audit": metric_audit_path,
        }
        report_input_artifacts = [
            {"name": name, "path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in report_input_paths.items()
        ]
        provenance = {
            "ok": True, "campaign_id": selection["campaign_id"], "selected_fusion_artifact_hash": selection["artifact_hash"],
            "source_artifact_audit_hash": selection["source_artifact_audit_hash"],
            "split_manifest_hash": audit.get("split_manifest_hash"), "hlt_content_hashes": audit.get("hlt_content_hashes"),
            "inputs": {
                row["name"]: {"path": row["path"], "sha256": row["sha256"]}
                for row in report_input_artifacts
            },
            "deployable_results": [row["run_id"] for row in final["selected_results"]], "oracle_diagnostics": [],
            "raw_metric_reproduction_audit_hash": metric_audit["audit_hash"],
            "final_test_deployable_rows_are_hlt_only": True, "final_test_status": final["final_test_status"],
            "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
            "uses_teacher_logits_at_runtime": False, "deployable": True,
        }
        (temporary_root / "provenance_audit.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = (
            f"# Local residual-field P7b fusion campaign `{selection['campaign_id']}`\n\n"
            f"Final-test status: **{final['final_test_status']}**. All deployable final rows are HLT-only; oracle diagnostics are reported separately and none are included here.\n\n"
            f"- Best raw member: `{raw_best['member_id']}` (final accuracy {raw_best['accuracy']:.6f}).\n"
            f"- G0 reproduction: passed (`{g0['reference_id']}`).\n"
            f"- Best late F_method: `{late_best['candidate_id']}` (selection accuracy {late_best['selection_accuracy']:.6f}).\n"
            f"- Best representation F_method: `{representation_best['candidate_id']}` ({representation_selection_label} {representation_best['selection_accuracy']:.6f}, three-seed diagnostic variance {representation_best['selection_accuracy_variance']}).\n"
            f"- Best F_seed: `{seed_best['candidate_id']}` (selection accuracy {seed_best['selection_accuracy']:.6f}).\n"
            f"- Frozen F_method recipe replay on F_seed: `{replay['candidate_id']}` (stack-val accuracy {replay_metrics['accuracy']:.6f}; no hyperparameter search).\n"
            f"- Paired F_method accuracy gain versus A0: {accuracy_interval['estimate']:+.6f} (95% interval {accuracy_interval['interval_95']}).\n"
            f"- F_method rejection champion, QCD vs Hgg at 50% signal efficiency: {rejection_point['qcd_false_positive_count']}/{rejection_point['qcd_support']} QCD passes; rejection {rejection_point['background_rejection']} (95% lower bound {rejection_point['background_rejection_lower_bound_95']}).\n"
            f"- Lowest measured selected end-to-end runtime: `{fastest_selected['run_id']}` ({fastest_selected['median_batch_latency_ms']:.3f} ms per batch, excluding loading and host-to-device transfer).\n\n"
            "Paired confidence intervals, QCD rejection counts/intervals, complementarity, and full runtime accounting are in the accompanying CSV files.\n"
        )
        (temporary_root / "summary.md").write_text(summary, encoding="utf-8")
        run_report: dict[str, Any] = {
            "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_REPORT_CONTRACT,
            "campaign_id": selection["campaign_id"], "created_at": datetime.now(timezone.utc).isoformat(),
            "selected_fusion_artifact_hash": selection["artifact_hash"], "final_test_status": final["final_test_status"],
            "headline": {"best_raw_member": raw_best, "g0_reproduction": g0, "best_late_F_method": late_best,
                         "best_representation_F_method": representation_best, "best_F_seed": seed_best,
                         "method_recipe_replay_F_seed": {"candidate_id": replay["candidate_id"], "metrics": replay_metrics},
                         "paired_accuracy_vs_A0": accuracy_interval, "headline_rejection": rejection_point,
                         "runtime": fastest_selected},
            "row_counts": {"members": len(member_rows), "candidates": len(candidate_rows), "selected": len(selected_rows),
                           "paired": len(paired_rows), "binary_rejection": len(binary_rows), "complementarity": len(complementarity_rows),
                           "runtime": len(runtime_rows), "bootstrap": len(bootstrap_rows)},
            "artifacts": list(FINAL_REPORT_FILENAMES),
            "input_artifacts": report_input_artifacts,
            "output_artifacts": [
                {
                    "path": str((report_root / name).resolve()),
                    "sha256": sha256_file(temporary_root / name),
                }
                for name in FINAL_REPORT_FILENAMES
                if name != "run_report.json"
            ],
            "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
            "uses_teacher_logits_at_runtime": False, "deployable": True,
        }
        run_report["artifact_hash"] = stable_fusion_json_hash(run_report)
        (temporary_root / "run_report.json").write_text(json.dumps(run_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        missing = [name for name in FINAL_REPORT_FILENAMES if not (temporary_root / name).is_file()]
        if missing:
            raise RuntimeError(f"final report is incomplete: {missing}")
        os.replace(temporary_root, report_root)
        return run_report
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


__all__ = ["LOCAL_RESIDUAL_FIELD_FUSION_REPORT_CONTRACT", "FINAL_REPORT_FILENAMES", "write_local_residual_field_fusion_campaign_report"]

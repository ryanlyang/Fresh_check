"""Step 9 matched candidate execution and stack-val-only champion selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import load_prediction_block
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .fusion_atomic import publish_temporary_file
from .fusion_campaign import (
    FUSION_CANDIDATE_IDS,
    FUSION_CHAMPION_ACCURACY,
    FUSION_CHAMPION_REJECTION,
    FUSION_FAMILY_LATE,
    FUSION_FAMILY_REPRESENTATION,
    FUSION_FINAL_TEST_STATUS,
    FUSION_GROUP_METHOD,
    FUSION_GROUP_SEED,
    FUSION_HEAD_SEEDS,
    FUSION_SELECTION_SPLIT,
    FusionCampaignConfig,
    SelectedFusionArtifact,
    default_fusion_candidate_specs,
    default_fusion_group_specs,
    stable_fusion_json_hash,
    validate_selected_fusion_artifact,
)
from .fusion_features import require_development_prediction_sources
from .fusion_late import LateFusionCampaignFitConfig, fit_late_fusion_campaign_candidate
from .fusion_metrics import local_residual_field_multiclass_metrics, local_residual_field_binary_projection_metrics
from .fusion_seed_control import sha256_file
from .fusion_train import (
    RepresentationFusionTrainConfig,
    load_representation_fusion_development_data,
    load_representation_fusion_head_from_checkpoint,
    predict_representation_fusion_head,
    train_representation_fusion_campaign_candidate,
)


LOCAL_RESIDUAL_FIELD_FUSION_CANDIDATE_REPORT_CONTRACT = "local_residual_field_fusion_candidate_report_v1"
LOCAL_RESIDUAL_FIELD_FUSION_SELECTION_SET_CONTRACT = "local_residual_field_selected_fusion_set_v1"
LOCAL_RESIDUAL_FIELD_FUSION_CANDIDATE_REGISTRY_CONTRACT = "local_residual_field_fusion_candidate_registry_v1"
FUSION_HEADLINE_SIGNALS = ("Hgg", "Hbb", "Tbqq", "Wqq", "Zqq")


def _atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_temporary_file(temporary, path, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return dict(payload)


def _validate_candidate_report(path: str | Path) -> dict[str, Any]:
    report = _load_json(path)
    if report.get("contract") != LOCAL_RESIDUAL_FIELD_FUSION_CANDIDATE_REPORT_CONTRACT or report.get("ok") is not True:
        raise ValueError(f"invalid fusion candidate report: {path}")
    if report.get("final_test_opened") is not False:
        raise ValueError(f"candidate report opened final-test before selection: {path}")
    if set(report.get("metrics") or {}) != {"stack_train", "stack_val"}:
        raise ValueError(f"candidate report does not contain exactly the two development splits: {path}")
    def find_final_metric(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).lower()
                if "final_test" in normalized and normalized != "final_test_opened":
                    return True
                if normalized in {"split", "selection_split", "fit_split"} and item == "final_test":
                    return True
                if find_final_metric(item):
                    return True
        elif isinstance(value, list):
            return any(find_final_metric(item) for item in value)
        return False
    if find_final_metric(report):
        raise ValueError(f"candidate report contains final-test metrics before selection: {path}")
    if report.get("runtime_inputs") != "HLT_only" or report.get("deployable") is not True or any(
        bool(report.get(key)) for key in ("uses_true_fields", "uses_offline_particles", "uses_teacher_logits_at_runtime")
    ):
        raise ValueError(f"candidate report is not deployable HLT-only: {path}")
    fit_artifacts = report.get("fit_artifacts")
    if not isinstance(fit_artifacts, list) or not fit_artifacts:
        raise ValueError(f"candidate report does not bind fit artifacts: {path}")
    for row in fit_artifacts:
        fit_path = Path(str(row.get("path") or ""))
        if not fit_path.is_file() or sha256_file(fit_path) != row.get("sha256"):
            raise ValueError(f"candidate fit artifact changed or disappeared: {fit_path}")
    unsigned = dict(report)
    stored_hash = unsigned.pop("artifact_hash", None)
    if stored_hash != stable_fusion_json_hash(unsigned):
        raise ValueError(f"candidate report hash mismatch: {path}")
    return report


def _member_checkpoint_hashes(registry: Mapping[str, Any], members: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for member in members:
        metadata_path = registry["members"][member]["splits"][FUSION_SELECTION_SPLIT]["metadata_path"]
        metadata = _load_json(metadata_path)
        value = metadata.get("checkpoint_hash") or metadata.get("student_checkpoint_hash")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"prediction metadata does not bind a checkpoint hash for {member}")
        output[member] = value
    return output


def _representation_hyperparameters(candidate_id: str) -> list[dict[str, Any]]:
    spec = next(spec for spec in default_fusion_candidate_specs() if spec.candidate_id == candidate_id)
    if candidate_id == "R0_linear_embeddings":
        return [
            {"hidden_width": 64, "dropout": 0.0, "weight_decay": float(weight_decay)}
            for weight_decay in spec.hyperparameter_grid["weight_decay"]
        ]
    return [
        {"hidden_width": int(width), "dropout": float(dropout), "weight_decay": float(weight_decay)}
        for width in spec.hyperparameter_grid["hidden_width"]
        for dropout in spec.hyperparameter_grid["dropout"]
        for weight_decay in spec.hyperparameter_grid["weight_decay"]
    ]


def _ranking_multiclass(row: Mapping[str, Any]) -> dict[str, float]:
    stability = row.get("head_stability")
    if isinstance(stability, Mapping) and int(stability.get("head_count", 0)) == len(FUSION_HEAD_SEEDS):
        if row.get("candidate_id") == "R0_linear_embeddings":
            per_head = stability.get("stack_val", {}).get("per_head", ())
            deployed = next(
                (head for head in per_head if int(head.get("seed", -1)) == FUSION_HEAD_SEEDS[0]),
                None,
            )
            if isinstance(deployed, Mapping):
                metrics = deployed["multiclass"]
                return {
                    "accuracy": float(metrics["accuracy"]),
                    "cross_entropy": float(metrics["cross_entropy"]),
                }
        multiclass = stability.get("stack_val", {}).get("multiclass", {})
        accuracy = multiclass.get("accuracy", {})
        cross_entropy = multiclass.get("cross_entropy", {})
        if isinstance(accuracy, Mapping) and isinstance(cross_entropy, Mapping):
            return {"accuracy": float(accuracy["mean"]), "cross_entropy": float(cross_entropy["mean"])}
    metrics = row["metrics"]["stack_val"]["multiclass"]
    return {"accuracy": float(metrics["accuracy"]), "cross_entropy": float(metrics["cross_entropy"])}


def _accuracy_choice(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    best_accuracy = max(_ranking_multiclass(row)["accuracy"] for row in rows)
    tied = [
        row for row in rows
        if best_accuracy - _ranking_multiclass(row)["accuracy"] < 0.0005
    ]
    selected = min(
        tied,
        key=lambda row: (
            _ranking_multiclass(row)["cross_entropy"],
            int(row["trainable_parameter_count"]),
            str(row.get("candidate_id") or row.get("trial_id")),
        ),
    )
    trace = [
        {"rule": "highest_locked_stack_val_ranking_accuracy", "best_accuracy": best_accuracy},
        {"rule": "accuracy_tie_window", "strict_difference_less_than": 0.0005, "eligible_count": len(tied)},
        {"rule": "lower_cross_entropy_then_parameters", "selected": selected.get("candidate_id") or selected.get("trial_id")},
    ]
    return selected, trace


@dataclass
class FusionCandidateRunConfig:
    campaign_id: str
    group_id: str
    candidate_id: str
    output_dir: str
    prediction_sources: str
    source_artifact_audit: str
    feature_root: str | None = None
    phase: str = "screening"
    device: str = "auto"
    learning_rate: float = 1.0e-3
    epochs: int = 80
    patience: int = 10
    batch_size: int = 2048
    stacker_max_steps: int = 80
    classwise_steps: int = 600
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.group_id not in {FUSION_GROUP_METHOD, FUSION_GROUP_SEED}:
            raise ValueError("group_id must be F_method or F_seed")
        if self.candidate_id not in FUSION_CANDIDATE_IDS:
            raise ValueError("candidate_id is outside the locked registry")
        if self.phase not in {"screening", "stability"}:
            raise ValueError("phase must be screening or stability")


def _representation_trial(
    config: FusionCandidateRunConfig,
    *,
    seed: int,
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    trial_id = stable_fusion_json_hash({"seed": seed, **dict(hyperparameters)})[:16]
    root = Path(config.output_dir) / "trials" / f"seed_{seed}" / trial_id
    checkpoint_path = root / "head.pt"
    train_report_path = root / "train_report.json"
    if checkpoint_path.is_file() and train_report_path.is_file() and not config.overwrite:
        completed = _load_json(train_report_path)
        expected_hyperparameters = {
            "seed": int(seed), "hidden_width": int(hyperparameters["hidden_width"]),
            "dropout": float(hyperparameters["dropout"]), "weight_decay": float(hyperparameters["weight_decay"]),
            "learning_rate": float(config.learning_rate),
        }
        if (
            completed.get("ok") is True
            and completed.get("contract") == "local_residual_field_representation_fusion_train_v1"
            and completed.get("campaign_id") == config.campaign_id
            and completed.get("group_id") == config.group_id
            and completed.get("candidate_id") == config.candidate_id
            and completed.get("hyperparameters") == expected_hyperparameters
            and completed.get("checkpoint_hash") == sha256_file(checkpoint_path)
        ):
            return {
                "trial_id": trial_id, "seed": seed, "hyperparameters": dict(hyperparameters),
                "trainable_parameter_count": completed["training"]["trainable_parameter_count"],
                "metrics": completed["metrics"], "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_hash": completed["checkpoint_hash"],
                "train_report_path": str(train_report_path.resolve()),
                "train_report_sha256": sha256_file(train_report_path), "resumed_completed_trial": True,
            }
    if root.exists() and not config.overwrite:
        quarantine = root.with_name(
            f"{root.name}.partial_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        root.replace(quarantine)
    report = train_representation_fusion_campaign_candidate(
        RepresentationFusionTrainConfig(
            campaign_id=config.campaign_id, group_id=config.group_id, candidate_id=config.candidate_id,
            feature_root=str(config.feature_root), prediction_sources=config.prediction_sources,
            source_artifact_audit=config.source_artifact_audit,
            checkpoint_path=str(checkpoint_path), report_path=str(train_report_path),
            seed=seed, hidden_width=int(hyperparameters["hidden_width"]),
            dropout=float(hyperparameters["dropout"]), weight_decay=float(hyperparameters["weight_decay"]),
            learning_rate=float(config.learning_rate), epochs=int(config.epochs), patience=int(config.patience),
            batch_size=int(config.batch_size), device=config.device, overwrite=bool(config.overwrite),
        )
    )
    return {
        "trial_id": trial_id,
        "seed": seed,
        "hyperparameters": dict(hyperparameters),
        "trainable_parameter_count": report["training"]["trainable_parameter_count"],
        "metrics": report["metrics"],
        "checkpoint_path": report["checkpoint_path"],
        "checkpoint_hash": report["checkpoint_hash"],
        "train_report_path": str(train_report_path.resolve()),
        "train_report_sha256": sha256_file(train_report_path),
    }


def _average_head_metrics(
    config: FusionCandidateRunConfig,
    head_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    data, _hashes = load_representation_fusion_development_data(
        group_id=config.group_id, feature_root=str(config.feature_root),
        prediction_sources=config.prediction_sources, source_artifact_audit=config.source_artifact_audit,
    )
    metrics: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    stability: dict[str, Any] = {"head_count": len(head_rows), "head_seeds": [int(row["seed"]) for row in head_rows]}
    for split in ("stack_train", "stack_val"):
        rows, split_diagnostics, per_head_metrics = [], [], []
        for head in head_rows:
            model, _payload = load_representation_fusion_head_from_checkpoint(head["checkpoint_path"], device=config.device)
            logits, diagnostic = predict_representation_fusion_head(
                model, data[split], device=config.device, batch_size=config.batch_size,
            )
            rows.append(logits)
            split_diagnostics.append({"seed": head["seed"], **diagnostic})
            per_head_metrics.append({
                "seed": int(head["seed"]),
                "multiclass": local_residual_field_multiclass_metrics(logits, data[split].labels, label_names=LABEL_NAMES),
                "binary_projection": local_residual_field_binary_projection_metrics(logits, data[split].labels, label_names=LABEL_NAMES),
            })
        deployed_rows = rows[:1] if config.candidate_id == "R0_linear_embeddings" else rows
        averaged = np.mean(np.stack(deployed_rows, axis=0).astype(np.float64), axis=0).astype(np.float32)
        metrics[split] = {
            "multiclass": local_residual_field_multiclass_metrics(averaged, data[split].labels, label_names=LABEL_NAMES),
            "binary_projection": local_residual_field_binary_projection_metrics(averaged, data[split].labels, label_names=LABEL_NAMES),
        }
        diagnostics[split] = split_diagnostics
        scalar_names = (
            "accuracy", "cross_entropy", "macro_one_vs_rest_auc", "macro_per_class_accuracy",
            "expected_calibration_error", "brier_score",
        )
        stability[split] = {
            "per_head": per_head_metrics,
            "multiclass": {
                name: {
                    "mean": float(np.mean([float(row["multiclass"][name]) for row in per_head_metrics])),
                    "variance": float(np.var([float(row["multiclass"][name]) for row in per_head_metrics])),
                    "values": [float(row["multiclass"][name]) for row in per_head_metrics],
                }
                for name in scalar_names
                if all(row["multiclass"].get(name) is not None for row in per_head_metrics)
            },
            "deployment_head_seeds": [int(row["seed"]) for row in (head_rows[:1] if config.candidate_id == "R0_linear_embeddings" else head_rows)],
            "deployment_rule": "single_fixed_seed_linear_head" if config.candidate_id == "R0_linear_embeddings" else "mean_head_logits",
            "ranking_rule": "deployed_fixed_seed_5101" if config.candidate_id == "R0_linear_embeddings" else "mean_per_head_metrics",
        }
        if split == FUSION_SELECTION_SPLIT:
            objectives = [_rejection_objective_from_binary(row["binary_projection"]) for row in per_head_metrics]
            stability[split]["rejection_objective"] = {
                "mean": float(np.mean(objectives)), "variance": float(np.var(objectives)),
                "values": [float(value) for value in objectives],
            }
    diagnostics["stability"] = stability
    return metrics, diagnostics


def run_fusion_candidate(config: FusionCandidateRunConfig) -> dict[str, Any]:
    """Fit exactly one locked candidate/group pair without any final-test access."""

    campaign = FusionCampaignConfig(campaign_id=config.campaign_id)
    spec = next(spec for spec in campaign.candidates if spec.candidate_id == config.candidate_id)
    group = next(group for group in campaign.groups if group.group_id == config.group_id)
    registry = require_development_prediction_sources(
        config.prediction_sources, source_artifact_audit=config.source_artifact_audit,
    )
    output_root = Path(config.output_dir)
    checkpoint_hashes = _member_checkpoint_hashes(registry, group.member_ids)
    if spec.family == FUSION_FAMILY_LATE:
        if config.phase != "screening":
            raise ValueError("late-fusion candidates do not have a stability phase")
        fit_path = output_root / "fit_artifact.json"
        fit = None
        if fit_path.is_file() and not config.overwrite:
            completed_fit = _load_json(fit_path)
            unsigned_fit = dict(completed_fit)
            stored_fit_hash = unsigned_fit.pop("artifact_hash", None)
            if (
                completed_fit.get("ok") is True
                and completed_fit.get("contract") == "local_residual_field_late_fusion_fit_v1"
                and completed_fit.get("campaign_id") == config.campaign_id
                and completed_fit.get("group_id") == config.group_id
                and completed_fit.get("candidate_id") == config.candidate_id
                and stored_fit_hash == stable_fusion_json_hash(unsigned_fit)
            ):
                fit = completed_fit
            else:
                quarantine = fit_path.with_name(
                    f"{fit_path.name}.partial_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                )
                fit_path.replace(quarantine)
        if fit is None:
            fit = fit_late_fusion_campaign_candidate(
                LateFusionCampaignFitConfig(
                    campaign_id=config.campaign_id, group_id=config.group_id, candidate_id=config.candidate_id,
                    prediction_sources=config.prediction_sources, source_artifact_audit=config.source_artifact_audit,
                    output_path=str(fit_path), device=config.device, stacker_max_steps=config.stacker_max_steps,
                    classwise_steps=config.classwise_steps, overwrite=config.overwrite,
                )
            )
        report_path = output_root / "candidate_report.json"
        report: dict[str, Any] = {
            "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_CANDIDATE_REPORT_CONTRACT,
            "campaign_id": config.campaign_id, "group_id": config.group_id,
            "member_ids": list(group.member_ids), "member_checkpoint_hashes": checkpoint_hashes,
            "candidate_id": config.candidate_id, "family": spec.family,
            "candidate_spec": spec.to_dict(), "candidate_spec_hash": stable_fusion_json_hash(spec.to_dict()),
            "phase": "complete", "head_seeds": [], "selected_hyperparameters": fit["parameters"],
            "metrics": fit["metrics"], "trainable_parameter_count": fit["trainable_parameter_count"],
            "fit_artifacts": [{"path": str(fit_path.resolve()), "sha256": sha256_file(fit_path), "artifact_hash": fit["artifact_hash"]}],
            "prediction_sources_path": str(Path(config.prediction_sources).resolve()),
            "prediction_sources_hash": registry["manifest_hash"],
            "source_artifact_audit_path": str(Path(config.source_artifact_audit).resolve()),
            "source_artifact_audit_hash": registry["source_artifact_audit_hash"],
            "feature_root": None, "development_splits": ["stack_train", "stack_val"],
            "final_test_opened": False, "runtime_inputs": "HLT_only", "uses_true_fields": False,
            "uses_offline_particles": False, "uses_teacher_logits_at_runtime": False, "deployable": True,
        }
    else:
        if not config.feature_root:
            raise ValueError("representation candidates require feature_root")
        screening_path = output_root / "candidate_report.json"
        screening_candidate_binding = None
        if config.phase == "screening":
            trials = [
                _representation_trial(config, seed=FUSION_HEAD_SEEDS[0], hyperparameters=hyperparameters)
                for hyperparameters in _representation_hyperparameters(config.candidate_id)
            ]
            selected, trace = _accuracy_choice(trials)
            heads = [selected]
            report_path = screening_path
            phase = "screening"
        else:
            screening = _validate_candidate_report(screening_path)
            screening_candidate_binding = {
                "path": str(screening_path.resolve()), "sha256": sha256_file(screening_path),
                "artifact_hash": screening["artifact_hash"],
            }
            if screening.get("phase") != "screening" or screening.get("head_seeds") != [FUSION_HEAD_SEEDS[0]]:
                raise ValueError("stability phase requires the immutable seed-5101 screening report")
            selected_hyperparameters = screening["selected_hyperparameters"]
            first = screening["head_artifacts"][0]
            if sha256_file(first["checkpoint_path"]) != first["checkpoint_hash"]:
                raise ValueError("screening head checkpoint changed before stability rerun")
            heads = [first] + [
                _representation_trial(config, seed=seed, hyperparameters=selected_hyperparameters)
                for seed in FUSION_HEAD_SEEDS[1:]
            ]
            trials = screening["hyperparameter_trials"]
            trace = screening["hyperparameter_selection_trace"]
            selected = {"hyperparameters": selected_hyperparameters, "trainable_parameter_count": screening["trainable_parameter_count"]}
            report_path = output_root / "candidate_stability_report.json"
            phase = "stability"
        metrics, diagnostics = _average_head_metrics(config, heads)
        feature_manifest_bindings = []
        for member in group.member_ids:
            feature_manifest_path = Path(config.feature_root) / member / "representation_manifest.json"
            feature_manifest = _load_json(feature_manifest_path)
            unsigned_feature_manifest = dict(feature_manifest)
            feature_manifest_hash = unsigned_feature_manifest.pop("manifest_hash", None)
            if feature_manifest_hash != stable_fusion_json_hash(unsigned_feature_manifest):
                raise ValueError(f"representation manifest logical hash mismatch for {member}")
            feature_manifest_bindings.append({
                "member_id": member, "path": str(feature_manifest_path.resolve()),
                "sha256": sha256_file(feature_manifest_path), "manifest_hash": feature_manifest_hash,
            })
        report = {
            "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_CANDIDATE_REPORT_CONTRACT,
            "campaign_id": config.campaign_id, "group_id": config.group_id,
            "member_ids": list(group.member_ids), "member_checkpoint_hashes": checkpoint_hashes,
            "candidate_id": config.candidate_id, "family": spec.family,
            "candidate_spec": spec.to_dict(), "candidate_spec_hash": stable_fusion_json_hash(spec.to_dict()),
            "phase": phase, "head_seeds": [int(head["seed"]) for head in heads],
            "selected_hyperparameters": dict(selected["hyperparameters"]),
            "hyperparameter_trials": trials, "hyperparameter_selection_trace": trace,
            "head_artifacts": heads, "metrics": metrics, "head_diagnostics": diagnostics,
            "head_stability": diagnostics.get("stability"),
            "screening_candidate_binding": screening_candidate_binding,
            "feature_manifest_bindings": feature_manifest_bindings,
            "trainable_parameter_count": int(selected["trainable_parameter_count"]),
            "fit_artifacts": [
                {"path": head["train_report_path"], "sha256": head["train_report_sha256"], "artifact_hash": head["checkpoint_hash"]}
                for head in heads
            ],
            "prediction_sources_path": str(Path(config.prediction_sources).resolve()),
            "prediction_sources_hash": registry["manifest_hash"],
            "source_artifact_audit_path": str(Path(config.source_artifact_audit).resolve()),
            "source_artifact_audit_hash": registry["source_artifact_audit_hash"],
            "feature_root": str(Path(config.feature_root).resolve()),
            "development_splits": ["stack_train", "stack_val"], "final_test_opened": False,
            "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
            "uses_teacher_logits_at_runtime": False, "deployable": True,
        }
    report["artifact_hash"] = stable_fusion_json_hash(report)
    _atomic_json(report_path, report, overwrite=config.overwrite)
    return report


def _rejection_objective_from_binary(binary_projection: Mapping[str, Any]) -> float:
    projections = binary_projection["projections"]
    values: list[float] = []
    for signal in FUSION_HEADLINE_SIGNALS:
        point = projections[f"QCD_vs_{signal}"]["operating_points"]["signal_efficiency_0.50"]
        if point.get("available") is not True:
            return float("inf")
        false_positives = int(point["qcd_false_positive_count"])
        support = int(point["qcd_support"])
        values.append(math.log((false_positives + 0.5) / float(support + 1)))
    return float(np.mean(values))


def _rejection_objective(report: Mapping[str, Any]) -> float:
    stability = report.get("head_stability")
    if isinstance(stability, Mapping) and int(stability.get("head_count", 0)) == len(FUSION_HEAD_SEEDS):
        if report.get("candidate_id") == "R0_linear_embeddings":
            per_head = stability.get("stack_val", {}).get("per_head", ())
            deployed = next(
                (head for head in per_head if int(head.get("seed", -1)) == FUSION_HEAD_SEEDS[0]),
                None,
            )
            if isinstance(deployed, Mapping):
                return _rejection_objective_from_binary(deployed["binary_projection"])
        objective = stability.get("stack_val", {}).get("rejection_objective", {})
        if isinstance(objective, Mapping) and objective.get("mean") is not None:
            return float(objective["mean"])
    return _rejection_objective_from_binary(report["metrics"]["stack_val"]["binary_projection"])


@dataclass
class FusionSelectionConfig:
    campaign_id: str
    candidates_root: str
    prediction_sources: str
    source_artifact_audit: str
    output_path: str
    pristine_confirmation_manifest: str | None = None
    overwrite: bool = False


def _champions_for_group(
    reports: Sequence[Mapping[str, Any]],
    *,
    best_raw_accuracy: float,
) -> tuple[Mapping[str, Any], list[dict[str, Any]], Mapping[str, Any], list[dict[str, Any]]]:
    accuracy, accuracy_trace = _accuracy_choice(reports)
    eligible = [
        report for report in reports
        if _ranking_multiclass(report)["accuracy"] >= best_raw_accuracy - 0.001
    ]
    if not eligible:
        raise ValueError("no rejection candidate satisfies the locked stack-val accuracy floor")
    rejection = min(
        eligible,
        key=lambda report: (
            _rejection_objective(report),
            _ranking_multiclass(report)["cross_entropy"],
            int(report["trainable_parameter_count"]),
            report["candidate_id"],
        ),
    )
    rejection_trace = [
        {"rule": "accuracy_floor", "best_raw_accuracy": best_raw_accuracy, "maximum_drop": 0.001, "eligible_count": len(eligible)},
        {"rule": "minimum_locked_log_jeffreys_fpr_at_0.50", "signals": list(FUSION_HEADLINE_SIGNALS), "selected_objective": _rejection_objective(rejection)},
        {"rule": "lower_cross_entropy_then_parameters", "selected": rejection["candidate_id"]},
    ]
    return accuracy, accuracy_trace, rejection, rejection_trace


def select_fusion_champions(config: FusionSelectionConfig) -> dict[str, Any]:
    """Require symmetric coverage, finish stability screening, and freeze both roles."""

    campaign = FusionCampaignConfig(campaign_id=config.campaign_id)
    registry = require_development_prediction_sources(
        config.prediction_sources, source_artifact_audit=config.source_artifact_audit,
    )
    candidate_root = Path(config.candidates_root)
    screening: dict[str, dict[str, dict[str, Any]]] = {}
    paths: dict[tuple[str, str, str], Path] = {}
    for group in campaign.groups:
        screening[group.group_id] = {}
        for candidate in campaign.candidates:
            path = candidate_root / group.group_id / candidate.candidate_id / "candidate_report.json"
            report = _validate_candidate_report(path)
            if (report.get("campaign_id"), report.get("group_id"), report.get("candidate_id")) != (
                config.campaign_id, group.group_id, candidate.candidate_id,
            ):
                raise ValueError(f"candidate report identity mismatch: {path}")
            if report.get("candidate_spec_hash") != stable_fusion_json_hash(candidate.to_dict()):
                raise ValueError(f"candidate registry drift in report: {path}")
            if report.get("prediction_sources_hash") != registry["manifest_hash"]:
                raise ValueError(f"candidate report uses different prediction sources: {path}")
            screening[group.group_id][candidate.candidate_id] = report
            paths[(group.group_id, candidate.candidate_id, "screening")] = path
    representation_ids = [
        candidate.candidate_id for candidate in campaign.candidates if candidate.family == FUSION_FAMILY_REPRESENTATION
    ]
    raw_accuracy: dict[str, float] = {}
    for group in campaign.groups:
        values = []
        for member in group.member_ids:
            block = load_prediction_block(registry["members"][member]["prediction_root"], member, "stack_val", verify_hash=True)
            values.append(local_residual_field_multiclass_metrics(block.logits, block.labels, label_names=LABEL_NAMES)["accuracy"])
        raw_accuracy[group.group_id] = max(values)
    union: set[str] = set()
    for group_id in (FUSION_GROUP_METHOD, FUSION_GROUP_SEED):
        representation_rows = [screening[group_id][candidate_id] for candidate_id in representation_ids]
        accuracy_representation, _trace = _accuracy_choice(representation_rows)
        union.add(accuracy_representation["candidate_id"])
        rejection_eligible = [
            report for report in representation_rows
            if _ranking_multiclass(report)["accuracy"] >= raw_accuracy[group_id] - 0.001
        ]
        if rejection_eligible:
            rejection_representation = min(
                rejection_eligible,
                key=lambda report: (
                    _rejection_objective(report),
                    _ranking_multiclass(report)["cross_entropy"],
                    int(report["trainable_parameter_count"]), report["candidate_id"],
                ),
            )
            union.add(rejection_representation["candidate_id"])
    active = {group: dict(rows) for group, rows in screening.items()}
    for group_id in (FUSION_GROUP_METHOD, FUSION_GROUP_SEED):
        for candidate_id in sorted(union):
            path = candidate_root / group_id / candidate_id / "candidate_stability_report.json"
            report = _validate_candidate_report(path)
            if report.get("head_seeds") != list(FUSION_HEAD_SEEDS) or report.get("phase") != "stability":
                raise ValueError(f"required three-seed stability report is incomplete: {path}")
            if report.get("selected_hyperparameters") != screening[group_id][candidate_id].get("selected_hyperparameters"):
                raise ValueError(f"stability run changed screened hyperparameters: {path}")
            active[group_id][candidate_id] = report
            paths[(group_id, candidate_id, "stability")] = path
    stability_plan_binding = None
    stability_plan_path = candidate_root.parent / "selection" / "stability_plan.json"
    if stability_plan_path.is_file():
        stability_plan = _load_json(stability_plan_path)
        unsigned_plan = dict(stability_plan)
        stored_plan_hash = unsigned_plan.pop("artifact_hash", None)
        if stability_plan.get("contract") != "local_residual_field_fusion_stability_plan_v1" or stability_plan.get("ok") is not True:
            raise ValueError("stability plan contract mismatch")
        if stored_plan_hash != stable_fusion_json_hash(unsigned_plan):
            raise ValueError("stability plan logical hash mismatch")
        if stability_plan.get("campaign_id") != config.campaign_id or set(stability_plan.get("required_candidate_ids") or ()) != union:
            raise ValueError("stability plan differs from selector's independently recomputed union")
        expected_screening_hashes = {
            group_id: {candidate_id: screening[group_id][candidate_id]["artifact_hash"] for candidate_id in representation_ids}
            for group_id in (FUSION_GROUP_METHOD, FUSION_GROUP_SEED)
        }
        if stability_plan.get("screening_artifact_hashes") != expected_screening_hashes:
            raise ValueError("stability plan is not bound to the active screening artifacts")
        stability_plan_binding = {
            "path": str(stability_plan_path.resolve()), "sha256": sha256_file(stability_plan_path),
            "artifact_hash": stored_plan_hash,
        }
    timestamp = datetime.now(timezone.utc).isoformat()
    selected_records: list[SelectedFusionArtifact] = []
    binding_rows: list[dict[str, Any]] = []
    for group in campaign.groups:
        rows = list(active[group.group_id].values())
        accuracy, accuracy_trace, rejection, rejection_trace = _champions_for_group(
            rows, best_raw_accuracy=raw_accuracy[group.group_id],
        )
        for role, report, trace in (
            (FUSION_CHAMPION_ACCURACY, accuracy, accuracy_trace),
            (FUSION_CHAMPION_REJECTION, rejection, rejection_trace),
        ):
            phase = "stability" if report.get("phase") == "stability" else "screening"
            path = paths[(group.group_id, report["candidate_id"], phase)]
            record = SelectedFusionArtifact(
                campaign_id=config.campaign_id, selection_timestamp=timestamp, champion_role=role,
                group_id=group.group_id, member_ids=tuple(group.member_ids),
                member_checkpoint_hashes=report["member_checkpoint_hashes"], candidate_id=report["candidate_id"],
                hyperparameters=report["selected_hyperparameters"],
                fit_artifact_hashes={"candidate_report": report["artifact_hash"]},
                selection_metrics={
                    "multiclass": report["metrics"]["stack_val"]["multiclass"],
                    "binary_projection": report["metrics"]["stack_val"]["binary_projection"],
                    "rejection_objective": _rejection_objective(report),
                    "ranking_multiclass": _ranking_multiclass(report),
                    "head_stability": report.get("head_stability"),
                },
                tie_break_trace=tuple(trace), candidate_registry_hash=campaign.candidate_registry_hash,
            )
            selected_records.append(record)
            binding_rows.append({
                "group_id": group.group_id, "champion_role": role, "candidate_id": report["candidate_id"],
                "candidate_report_path": str(path.resolve()), "candidate_report_sha256": sha256_file(path),
                "candidate_report_artifact_hash": report["artifact_hash"], "phase": report["phase"],
            })
    selection_input_bindings = []
    for group_id, rows in active.items():
        for candidate_id, report in rows.items():
            phase = "stability" if report.get("phase") == "stability" else "screening"
            path = paths[(group_id, candidate_id, phase)]
            selection_input_bindings.append({
                "group_id": group_id, "candidate_id": candidate_id, "phase": phase,
                "path": str(path.resolve()), "sha256": sha256_file(path),
                "artifact_hash": report["artifact_hash"],
            })
    registry_artifact = {
        "contract": LOCAL_RESIDUAL_FIELD_FUSION_CANDIDATE_REGISTRY_CONTRACT,
        "campaign": campaign.to_dict(), "candidate_registry_hash": campaign.candidate_registry_hash,
        "coverage": {group: list(rows) for group, rows in screening.items()},
        "representation_stability_union": sorted(union),
        "stability_plan_binding": stability_plan_binding,
        "selection_input_bindings": selection_input_bindings,
    }
    registry_artifact["artifact_hash"] = stable_fusion_json_hash(registry_artifact)
    registry_path = Path(config.output_path).parent / "candidate_registry.json"
    if registry_path.is_file() and not config.overwrite:
        if _load_json(registry_path) != registry_artifact:
            raise ValueError("existing candidate registry differs from the recoverable selection attempt")
    else:
        _atomic_json(registry_path, registry_artifact, overwrite=config.overwrite)
    feature_roots = {
        str(report["feature_root"]) for rows in screening.values() for report in rows.values()
        if report.get("feature_root")
    }
    if len(feature_roots) != 1:
        raise ValueError("candidate reports must bind exactly one common representation feature root")
    pristine = None
    if config.pristine_confirmation_manifest:
        pristine_path = Path(config.pristine_confirmation_manifest)
        pristine = {"path": str(pristine_path.resolve()), "sha256": sha256_file(pristine_path)}
    selection: dict[str, Any] = {
        "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_SELECTION_SET_CONTRACT,
        "campaign_id": config.campaign_id, "selection_timestamp": timestamp,
        "selection_source": FUSION_SELECTION_SPLIT, "candidate_registry_hash": campaign.candidate_registry_hash,
        "candidate_registry_path": str(registry_path.resolve()), "candidate_registry_sha256": sha256_file(registry_path),
        "representation_stability_union": sorted(union),
        "stability_plan_binding": stability_plan_binding,
        "selection_input_bindings": selection_input_bindings,
        "selections": [record.to_dict() for record in selected_records], "selection_bindings": binding_rows,
        "prediction_sources_path": str(Path(config.prediction_sources).resolve()),
        "prediction_sources_hash": registry["manifest_hash"],
        "source_artifact_audit_path": str(Path(config.source_artifact_audit).resolve()),
        "source_artifact_audit_hash": registry["source_artifact_audit_hash"],
        "member_prediction_roots": {
            member: registry["members"][member]["prediction_root"] for member in registry["members"]
        },
        "feature_root": next(iter(feature_roots)), "pristine_confirmation_manifest": pristine,
        "final_test_status": FUSION_FINAL_TEST_STATUS, "final_test_opened": False,
    }
    selection["artifact_hash"] = stable_fusion_json_hash(selection)
    _atomic_json(Path(config.output_path), selection, overwrite=config.overwrite)
    return selection


def load_selected_fusion_set(path: str | Path) -> dict[str, Any]:
    selection = _load_json(path)
    if selection.get("contract") != LOCAL_RESIDUAL_FIELD_FUSION_SELECTION_SET_CONTRACT or selection.get("ok") is not True:
        raise ValueError("selected_fusion.json contract mismatch")
    unsigned = dict(selection)
    stored = unsigned.pop("artifact_hash", None)
    if stored != stable_fusion_json_hash(unsigned):
        raise ValueError("selected_fusion.json artifact hash mismatch")
    campaign = FusionCampaignConfig(campaign_id=selection["campaign_id"])
    if selection.get("candidate_registry_hash") != campaign.candidate_registry_hash:
        raise ValueError("selected_fusion.json candidate registry hash mismatch")
    records = [
        validate_selected_fusion_artifact(SelectedFusionArtifact(**row), campaign)
        for row in selection.get("selections") or ()
    ]
    expected = {(group, role) for group in (FUSION_GROUP_METHOD, FUSION_GROUP_SEED) for role in (FUSION_CHAMPION_ACCURACY, FUSION_CHAMPION_REJECTION)}
    if {(record.group_id, record.champion_role) for record in records} != expected or len(records) != 4:
        raise ValueError("selected_fusion.json must contain both champion roles for both groups")
    if selection.get("final_test_opened") is not False:
        raise ValueError("selected_fusion.json was not frozen before final-test access")
    return selection


__all__ = [
    "LOCAL_RESIDUAL_FIELD_FUSION_CANDIDATE_REPORT_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_FUSION_SELECTION_SET_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_FUSION_CANDIDATE_REGISTRY_CONTRACT",
    "FUSION_HEADLINE_SIGNALS",
    "FusionCandidateRunConfig",
    "FusionSelectionConfig",
    "run_fusion_candidate",
    "select_fusion_champions",
    "load_selected_fusion_set",
]

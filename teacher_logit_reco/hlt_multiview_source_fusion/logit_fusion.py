"""Logit-fusion reporting for deployable HLT multiview experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import (
    PredictionBlock,
    load_prediction_block,
    save_prediction_block,
    select_weighted_average_weights,
    softmax_np,
    validate_prediction_alignment,
)
from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.privileged_distill_10class.metrics import pd10_prediction_metrics_from_logits

from .config import (
    HLTMVExperimentConfig,
    HLTMVExperimentLayout,
    HLT_MV_ALLOWED_INPUTS,
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_DEPLOYMENT_INPUTS,
    HLT_MV_EXPERIMENT_NAME,
    HLT_MV_FUSION_HLT_RANDOM_4SEED,
    HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SOURCE_5VIEW,
    default_hlt_mv_experiment_config,
    default_hlt_mv_experiment_layout,
    normalize_hlt_mv_source_name,
)
from .source import HLT_MV_SOURCE_PREDICTION_SPLITS, normalize_hlt_mv_source_prediction_splits


HLT_MV_LOGIT_FUSION_EXPERIMENT_STEP = "hlt_mv_step4_logit_fusion_reporting"
HLT_MV_LOGIT_FUSION_CONTRACT = "hlt_multiview_logit_fusion_report_v1"
HLT_MV_LOGIT_FUSION_PREDICTION_CONTRACT = "hlt_multiview_logit_fusion_prediction_v1"
HLT_MV_LOGIT_FUSION_REPORT_JSON = "fusion_report.json"
HLT_MV_LOGIT_FUSION_SUMMARY_JSON = "summary.json"
HLT_MV_LOGIT_FUSION_RUN_JSON = "run_report.json"
HLT_MV_LOGIT_FUSION_METRIC_TABLE_CSV = "metric_table.csv"
HLT_MV_LOGIT_FUSION_UNIFORM_METHOD = "uniform_logit_average"
HLT_MV_LOGIT_FUSION_WEIGHTED_METHOD = "weighted_logit_average"


@dataclass(frozen=True)
class HLTMVPredictionSpec:
    """One cached prediction source used by an HLT-MV logit fusion."""

    model_name: str
    prediction_dir: str
    role: str = "member"

    def __post_init__(self) -> None:
        name = normalize_hlt_mv_source_name(self.model_name)
        prediction_dir = str(Path(self.prediction_dir))
        if not prediction_dir:
            raise ValueError("prediction_dir cannot be empty")
        object.__setattr__(self, "model_name", name)
        object.__setattr__(self, "prediction_dir", prediction_dir)
        object.__setattr__(self, "role", str(self.role or "member"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "prediction_dir": self.prediction_dir,
            "role": self.role,
        }


@dataclass(frozen=True)
class HLTMVLogitFusionConfig:
    """Configuration for one HLT-MV cached-logit fusion report."""

    output_dir: str
    fusion_name: str
    model_specs: tuple[HLTMVPredictionSpec, ...]
    splits: tuple[str, ...] = HLT_MV_SOURCE_PREDICTION_SPLITS
    confirm_final_test: bool = False
    overwrite: bool = False
    fit_weighted_average: bool = True
    max_weight_steps: int = 30
    allowed_inputs: str = HLT_MV_ALLOWED_INPUTS
    deployment_inputs: str = HLT_MV_DEPLOYMENT_INPUTS

    def __post_init__(self) -> None:
        fusion_name = normalize_hlt_mv_source_name(self.fusion_name)
        specs = tuple(self.model_specs)
        if len(specs) < 2:
            raise ValueError("HLT-MV logit fusion requires at least two prediction specs.")
        names = [spec.model_name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"HLT-MV logit fusion model names must be unique, got {names!r}.")
        splits = normalize_hlt_mv_source_prediction_splits(self.splits)
        if "final_test" in splits and not bool(self.confirm_final_test):
            raise ValueError("HLT-MV logit fusion final-test evaluation requires confirm_final_test=True")
        if int(self.max_weight_steps) < 0:
            raise ValueError("max_weight_steps cannot be negative")
        object.__setattr__(self, "fusion_name", fusion_name)
        object.__setattr__(self, "model_specs", specs)
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "max_weight_steps", int(self.max_weight_steps))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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
                keys.append(str(key))
                seen.add(str(key))
    if not keys:
        rows = [{"available": False, "reason": "no rows"}]
        keys = ["available", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _config_for(config: HLTMVExperimentConfig | None) -> HLTMVExperimentConfig:
    return config or default_hlt_mv_experiment_config()


def _layout_for(
    *,
    layout: HLTMVExperimentLayout | None = None,
    output_root: str | Path = "checkpoints",
    pdv3_experiment_name: str = HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
) -> HLTMVExperimentLayout:
    return layout or default_hlt_mv_experiment_layout(
        output_root=output_root,
        pdv3_experiment_name=pdv3_experiment_name,
    )


def hlt_mv_builtin_logit_fusion_specs(
    fusion_name: str,
    layout: HLTMVExperimentLayout,
    *,
    config: HLTMVExperimentConfig | None = None,
) -> tuple[HLTMVPredictionSpec, ...]:
    """Return canonical prediction specs for one named HLT-MV fusion family."""

    cfg = _config_for(config)
    name = normalize_hlt_mv_source_name(fusion_name)
    if name == HLT_MV_FUSION_SOURCE_5VIEW:
        return tuple(
            HLTMVPredictionSpec(
                model_name=model_name,
                prediction_dir=str(layout.source_model_dir(model_name) / "predictions"),
                role="source_model",
            )
            for model_name in cfg.source_model_names
        )
    if name == HLT_MV_FUSION_HLT_RANDOM_4SEED:
        return tuple(
            HLTMVPredictionSpec(
                model_name=model_name,
                prediction_dir=str(layout.random_hlt_source_dir(model_name) / "predictions"),
                role="random_hlt_source_model",
            )
            for model_name in cfg.random_hlt_source_names
        )
    if name == HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL:
        return tuple(
            HLTMVPredictionSpec(
                model_name=model_name,
                prediction_dir=str(layout.pretrained_dualview_model_dir(model_name) / "predictions"),
                role="pretrained_particle_dualview",
            )
            for model_name in cfg.pretrained_dualview_names
        )
    if name == HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL:
        return tuple(
            HLTMVPredictionSpec(
                model_name=model_name,
                prediction_dir=str(layout.scratch_dualview_model_dir(model_name) / "predictions"),
                role="scratch_particle_dualview",
            )
            for model_name in cfg.scratch_dualview_names
        )
    raise ValueError(f"Unknown HLT-MV built-in logit fusion name: {fusion_name!r}")


def default_hlt_mv_logit_fusion_config(
    fusion_name: str,
    *,
    output_root: str | Path = "checkpoints",
    pdv3_experiment_name: str = HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    layout: HLTMVExperimentLayout | None = None,
    run_config: HLTMVExperimentConfig | None = None,
    model_specs: Sequence[HLTMVPredictionSpec] | None = None,
    output_dir: str | Path | None = None,
    splits: tuple[str, ...] | list[str] = HLT_MV_SOURCE_PREDICTION_SPLITS,
    confirm_final_test: bool = False,
    overwrite: bool = False,
    fit_weighted_average: bool = True,
    max_weight_steps: int = 30,
) -> HLTMVLogitFusionConfig:
    cfg = _config_for(run_config)
    lay = _layout_for(layout=layout, output_root=output_root, pdv3_experiment_name=pdv3_experiment_name)
    name = normalize_hlt_mv_source_name(fusion_name)
    specs = tuple(model_specs) if model_specs is not None else hlt_mv_builtin_logit_fusion_specs(name, lay, config=cfg)
    return HLTMVLogitFusionConfig(
        output_dir=str(Path(output_dir) if output_dir is not None else lay.logit_fusion_dir(name)),
        fusion_name=name,
        model_specs=specs,
        splits=normalize_hlt_mv_source_prediction_splits(splits),
        confirm_final_test=bool(confirm_final_test),
        overwrite=bool(overwrite),
        fit_weighted_average=bool(fit_weighted_average),
        max_weight_steps=int(max_weight_steps),
    )


def parse_hlt_mv_prediction_specs(values: Sequence[str]) -> tuple[HLTMVPredictionSpec, ...]:
    """Parse CLI specs of the form model_name=/path/to/predictions."""

    specs: list[HLTMVPredictionSpec] = []
    for value in values:
        if "=" not in str(value):
            raise ValueError(f"Prediction spec must be model_name=/path/to/predictions, got {value!r}")
        name, path = str(value).split("=", 1)
        specs.append(HLTMVPredictionSpec(model_name=name, prediction_dir=path, role="custom"))
    return tuple(specs)


def _load_blocks_for_specs(specs: Sequence[HLTMVPredictionSpec], split: str) -> list[PredictionBlock]:
    blocks = [
        load_prediction_block(spec.prediction_dir, spec.model_name, split, verify_hash=True)
        for spec in specs
    ]
    validate_prediction_alignment(blocks)
    return blocks


def _weighted_average_logits(blocks: Sequence[PredictionBlock], weights: np.ndarray) -> np.ndarray:
    validate_prediction_alignment(blocks)
    logits = np.stack([block.logits for block in blocks], axis=0).astype(np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != logits.shape[0]:
        raise ValueError(f"weights length {weights.shape[0]} does not match {logits.shape[0]} blocks")
    if np.any(weights < -1.0e-12):
        raise ValueError("HLT-MV logit fusion weights must be nonnegative")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("HLT-MV logit fusion weights must have positive finite sum")
    weights = weights / total
    combined = np.tensordot(weights, logits, axes=(0, 0)).astype(np.float32)
    if not np.isfinite(combined).all():
        raise FloatingPointError("HLT-MV fused logits contain non-finite values")
    return combined


def _metrics_for_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    validation_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if validation_metrics is None:
        return pd10_prediction_metrics_from_logits(logits, labels)
    return pd10_prediction_metrics_from_logits(
        logits,
        labels,
        validation_thresholds_by_class=validation_metrics.get("score_thresholds_by_class"),
        validation_binary_thresholds=validation_metrics.get("binary_score_thresholds"),
    )


def _save_fused_block(
    *,
    output_dir: Path,
    fusion_model_name: str,
    split: str,
    logits: np.ndarray,
    template_block: PredictionBlock,
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    block = PredictionBlock(
        model_name=fusion_model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=template_block.labels,
        jet_ids=list(template_block.jet_ids),
        metadata=dict(metadata),
    )
    return save_prediction_block(block, output_dir / "predictions", overwrite=overwrite)


def _metric_row(
    *,
    fusion_name: str,
    method: str,
    split: str,
    metrics: Mapping[str, Any],
    weights: Sequence[float],
) -> dict[str, Any]:
    return {
        "fusion_name": fusion_name,
        "method": method,
        "split": split,
        "accuracy": metrics.get("accuracy"),
        "cross_entropy": metrics.get("cross_entropy", metrics.get("loss")),
        "loss": metrics.get("loss"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "macro_ovr_auc": metrics.get("macro_ovr_auc"),
        "expected_calibration_error": metrics.get("expected_calibration_error"),
        "validation_threshold_fpr_at_signal_eff_0p30_macro": metrics.get(
            "validation_threshold_fpr_at_signal_eff_0p30_macro"
        ),
        "validation_threshold_fpr_at_signal_eff_0p50_macro": metrics.get(
            "validation_threshold_fpr_at_signal_eff_0p50_macro"
        ),
        "validation_binary_fpr_at_signal_eff_0p30_macro": metrics.get(
            "validation_binary_fpr_at_signal_eff_0p30_macro"
        ),
        "validation_binary_fpr_at_signal_eff_0p50_macro": metrics.get(
            "validation_binary_fpr_at_signal_eff_0p50_macro"
        ),
        "weights": list(weights),
    }


def run_hlt_mv_logit_fusion(config: HLTMVLogitFusionConfig) -> dict[str, Any]:
    """Evaluate uniform and model-val-selected weighted logit fusion."""

    output_dir = Path(config.output_dir)
    report_path = output_dir / HLT_MV_LOGIT_FUSION_REPORT_JSON
    if report_path.exists() and not bool(config.overwrite):
        raise FileExistsError(f"HLT-MV logit fusion report already exists: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks_by_split = {
        split: _load_blocks_for_specs(config.model_specs, split)
        for split in config.splits
    }
    model_names = [spec.model_name for spec in config.model_specs]
    uniform_weights = np.full((len(config.model_specs),), 1.0 / float(len(config.model_specs)), dtype=np.float64)

    report: dict[str, Any] = {
        "ok": True,
        "contract": HLT_MV_LOGIT_FUSION_CONTRACT,
        "experiment_name": HLT_MV_EXPERIMENT_NAME,
        "experiment_step": HLT_MV_LOGIT_FUSION_EXPERIMENT_STEP,
        "fusion_name": config.fusion_name,
        "model_names": model_names,
        "model_specs": [spec.to_dict() for spec in config.model_specs],
        "splits": list(config.splits),
        "allowed_inputs": config.allowed_inputs,
        "deployment_inputs": config.deployment_inputs,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "selection_split": "model_val",
        "final_test_not_used_for_weight_selection": True,
        "methods": {},
        "prediction_metadata": {},
    }
    metric_rows: list[dict[str, Any]] = []

    method_weights: dict[str, np.ndarray] = {
        HLT_MV_LOGIT_FUSION_UNIFORM_METHOD: uniform_weights,
    }
    if bool(config.fit_weighted_average):
        val_blocks = blocks_by_split.get("model_val")
        if val_blocks is None:
            raise ValueError("Weighted HLT-MV logit fusion requires model_val predictions.")
        selected_weights, selection_report = select_weighted_average_weights(
            val_blocks,
            mode="logits",
            max_steps=config.max_weight_steps,
        )
        method_weights[HLT_MV_LOGIT_FUSION_WEIGHTED_METHOD] = selected_weights
        report["weighted_logit_selection"] = selection_report

    for method, weights in method_weights.items():
        method_report: dict[str, Any] = {
            "weights": [float(item) for item in np.asarray(weights).tolist()],
            "metrics": {},
            "prediction_model_name": f"{config.fusion_name}_{method}",
        }
        validation_metrics = None
        for split in config.splits:
            blocks = blocks_by_split[split]
            fused_logits = _weighted_average_logits(blocks, weights)
            metrics = _metrics_for_logits(
                fused_logits,
                blocks[0].labels,
                validation_metrics=validation_metrics if split != "model_val" else None,
            )
            if split == "model_val":
                validation_metrics = metrics
            method_report["metrics"][split] = metrics
            metric_rows.append(
                _metric_row(
                    fusion_name=config.fusion_name,
                    method=method,
                    split=split,
                    metrics=metrics,
                    weights=method_report["weights"],
                )
            )
            prediction_metadata = _save_fused_block(
                output_dir=output_dir,
                fusion_model_name=method_report["prediction_model_name"],
                split=split,
                logits=fused_logits,
                template_block=blocks[0],
                metadata={
                    "contract": HLT_MV_LOGIT_FUSION_PREDICTION_CONTRACT,
                    "experiment_name": HLT_MV_EXPERIMENT_NAME,
                    "experiment_step": HLT_MV_LOGIT_FUSION_EXPERIMENT_STEP,
                    "fusion_name": config.fusion_name,
                    "method": method,
                    "model_names": model_names,
                    "weights": method_report["weights"],
                    "source_prediction_metadata": [block.metadata for block in blocks],
                    "rich_metrics": metrics,
                    "allowed_inputs": config.allowed_inputs,
                    "deployment_inputs": config.deployment_inputs,
                    "requires_offline_inputs": False,
                    "requires_teacher_features": False,
                    "requires_deterministic_hlt2_transform": True,
                    "final_test_not_used_for_weight_selection": True,
                },
                overwrite=bool(config.overwrite),
            )
            report["prediction_metadata"].setdefault(method, {})[split] = prediction_metadata
        report["methods"][method] = method_report

    summary = {
        "ok": True,
        "contract": HLT_MV_LOGIT_FUSION_CONTRACT,
        "fusion_name": config.fusion_name,
        "model_names": model_names,
        "methods": {
            method: {
                "weights": payload["weights"],
                "model_val_accuracy": payload["metrics"].get("model_val", {}).get("accuracy"),
                "model_val_cross_entropy": payload["metrics"].get("model_val", {}).get("cross_entropy"),
                "final_test_accuracy": payload["metrics"].get("final_test", {}).get("accuracy"),
                "final_test_cross_entropy": payload["metrics"].get("final_test", {}).get("cross_entropy"),
            }
            for method, payload in report["methods"].items()
        },
    }
    save_json(report_path, _jsonable(report))
    save_json(output_dir / HLT_MV_LOGIT_FUSION_SUMMARY_JSON, _jsonable(summary))
    save_json(output_dir / HLT_MV_LOGIT_FUSION_RUN_JSON, _jsonable(report))
    _write_csv(output_dir / HLT_MV_LOGIT_FUSION_METRIC_TABLE_CSV, metric_rows)
    return report


__all__ = [
    "HLTMVLogitFusionConfig",
    "HLTMVPredictionSpec",
    "HLT_MV_LOGIT_FUSION_CONTRACT",
    "HLT_MV_LOGIT_FUSION_EXPERIMENT_STEP",
    "HLT_MV_LOGIT_FUSION_METRIC_TABLE_CSV",
    "HLT_MV_LOGIT_FUSION_PREDICTION_CONTRACT",
    "HLT_MV_LOGIT_FUSION_REPORT_JSON",
    "HLT_MV_LOGIT_FUSION_RUN_JSON",
    "HLT_MV_LOGIT_FUSION_SUMMARY_JSON",
    "HLT_MV_LOGIT_FUSION_UNIFORM_METHOD",
    "HLT_MV_LOGIT_FUSION_WEIGHTED_METHOD",
    "default_hlt_mv_logit_fusion_config",
    "hlt_mv_builtin_logit_fusion_specs",
    "parse_hlt_mv_prediction_specs",
    "run_hlt_mv_logit_fusion",
]

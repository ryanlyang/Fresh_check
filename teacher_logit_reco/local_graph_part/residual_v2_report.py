"""Final report builder for local-graph residual expert V2 experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device

from teacher_logit_reco.set_matching.train import source_metadata
from teacher_logit_reco.subtoken_part.train import make_subtoken_hlt_loader

from .fusion import binary_logits_from_log_odds, binary_margin_from_logits, binary_metrics_from_signal_scores
from .protocol import (
    LOCAL_GRAPH_PART_CONTRACT,
    LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
)
from .residual_diagnostics import residual_correction_diagnostics
from .residual_report import (
    _diagnostic_rows,
    _discover_variant_reports,
    _jsonable,
    _metric,
    _metric_direction,
    _metric_row,
    _read_json,
    _write_csv,
)
from .residual_v2_cache import (
    LocalGraphResidualV2BaselineEmbeddingBlock,
    load_residual_v2_embedding_block,
    verify_residual_v2_embedding_block_alignment,
    verify_residual_v2_embedding_cache_family,
)
from .residual_v2_protocol import (
    LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT,
    LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_EVAL_SPLITS,
    LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC,
    LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_STEP,
    LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_REPORT_STEP,
    default_local_graph_residual_v2_protocol,
    local_graph_residual_v2_protocol_manifest,
)
from .residual_v2_train import (
    LocalGraphResidualExpertV2TrainConfig,
    _load_residual_v2_dataset,
    _shrunk_logits_from_prediction_arrays,
    _strip_prediction_arrays,
    load_local_graph_residual_expert_v2_checkpoint,
    run_local_graph_residual_expert_v2_epoch,
    select_local_graph_residual_v2_gamma_shrinkage,
)
from .train import LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS, LOCAL_GRAPH_SELECTION_METRICS


LOCAL_GRAPH_RESIDUAL_V2_REPORT_SPLITS = ("model_val", "stack_val", "final_test")
LOCAL_GRAPH_RESIDUAL_V2_BASELINE_REPORT_VARIANT = "exact_hlt_part_baseline"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _score_from_metric(value: float | None, direction: str) -> float:
    if value is None or not math.isfinite(float(value)):
        return float("-inf")
    return -float(value) if direction == "minimize" else float(value)


def _best_row(rows: Sequence[Mapping[str, Any]], direction: str) -> Mapping[str, Any] | None:
    scored = [(_score_from_metric(_float_or_none(row.get("primary_metric_value")), direction), row) for row in rows]
    scored = [item for item in scored if math.isfinite(item[0])]
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _split_max_jets(config: "LocalGraphResidualExpertV2ReportConfig", split: str) -> int | None:
    if split == "model_val":
        return config.max_model_val_jets
    if split == "stack_train":
        return config.max_stack_train_jets
    if split == "stack_val":
        return config.max_stack_val_jets
    if split == "final_test":
        return config.max_final_test_jets
    return None


def _baseline_block_for_report_subset(
    block: LocalGraphResidualV2BaselineEmbeddingBlock,
    *,
    max_jets: int | None,
) -> LocalGraphResidualV2BaselineEmbeddingBlock:
    if max_jets is None or int(max_jets) >= int(block.labels.shape[0]):
        return block
    n_jets = max(int(max_jets), 0)
    condition_features = None
    if block.condition_features_array is not None:
        condition_features = block.condition_features_array[:n_jets]
    return LocalGraphResidualV2BaselineEmbeddingBlock(
        split=block.split,
        logits=block.logits[:n_jets],
        embedding=block.embedding[:n_jets],
        labels=block.labels[:n_jets],
        indices=block.indices[:n_jets],
        metadata=dict(block.metadata),
        condition_features_array=condition_features,
    )


def _baseline_row(
    *,
    block: LocalGraphResidualV2BaselineEmbeddingBlock,
    split: str,
    primary_metric: str,
    direction: str,
) -> dict[str, Any]:
    metrics = block.metrics()
    return _metric_row(
        source_type="baseline",
        variant=LOCAL_GRAPH_RESIDUAL_V2_BASELINE_REPORT_VARIANT,
        split=split,
        metrics=metrics,
        primary_metric=primary_metric,
        direction=direction,
        extra={
            "score_type": "baseline_margin",
            "baseline_variant_contract": LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT,
            "n_jets": int(block.labels.shape[0]),
            "metrics": metrics,
        },
    )


def _logit_np(probability: float) -> float:
    clipped = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(clipped / (1.0 - clipped))


def _binary_logloss(labels: np.ndarray, margin: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    margin = np.asarray(margin, dtype=np.float64).reshape(-1)
    y_signed = 2.0 * labels - 1.0
    return float(np.mean(np.logaddexp(0.0, -y_signed * margin))) if labels.size else float("inf")


def _fit_baseline_calibration(block: LocalGraphResidualV2BaselineEmbeddingBlock) -> dict[str, Any]:
    labels = np.asarray(block.labels, dtype=np.int64)
    scores = np.asarray(block.margin, dtype=np.float64)
    if labels.size == 0:
        return {"scale": 1.0, "bias": 0.0, "model_val_logloss": None, "n_jets": 0}
    positive_rate = float(np.mean(labels == 1))
    center = float(np.mean(scores))
    base_bias = _logit_np(positive_rate) - center
    scale_grid = np.asarray((0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0), dtype=np.float64)
    bias_offsets = np.linspace(-4.0, 4.0, 81, dtype=np.float64)
    best: tuple[float, float, float] | None = None
    for scale in scale_grid:
        for bias_offset in bias_offsets:
            bias = float(_logit_np(positive_rate) - float(scale) * center + float(bias_offset))
            loss = _binary_logloss(labels, float(scale) * scores + bias)
            if best is None or loss < best[0]:
                best = (loss, float(scale), float(bias))
    assert best is not None
    return {
        "type": "positive_slope_grid_logistic_calibration",
        "fit_split": "model_val",
        "scale": float(best[1]),
        "bias": float(best[2]),
        "model_val_logloss": float(best[0]),
        "uncalibrated_model_val_logloss": _binary_logloss(labels, scores),
        "positive_rate": positive_rate,
        "n_jets": int(labels.size),
        "note": "Positive slope preserves score ranking, so FPR@50/AUC should match the baseline unless ties move.",
    }


def _calibrated_baseline_row(
    *,
    block: LocalGraphResidualV2BaselineEmbeddingBlock,
    split: str,
    calibration: Mapping[str, Any],
    primary_metric: str,
    direction: str,
) -> dict[str, Any]:
    scores = float(calibration["scale"]) * np.asarray(block.margin, dtype=np.float64) + float(calibration["bias"])
    metrics = binary_metrics_from_signal_scores(scores, block.labels)
    return _metric_row(
        source_type="calibration_only_control",
        variant="baseline_logistic_calibration_model_val",
        split=split,
        metrics=metrics,
        primary_metric=primary_metric,
        direction=direction,
        extra={
            "score_type": "calibrated_baseline_margin",
            "calibration_fit_split": calibration.get("fit_split"),
            "calibration_scale": calibration.get("scale"),
            "calibration_bias": calibration.get("bias"),
            "metrics": metrics,
        },
    )


def _apply_baseline_deltas(rows: list[dict[str, Any]], *, baseline_by_split: Mapping[str, float | None]) -> None:
    for row in rows:
        baseline_value = baseline_by_split.get(str(row.get("split")))
        value = _float_or_none(row.get("primary_metric_value"))
        direction = str(row.get("primary_metric_direction", "minimize"))
        row["baseline_variant"] = LOCAL_GRAPH_RESIDUAL_V2_BASELINE_REPORT_VARIANT
        row["baseline_primary_metric_value"] = baseline_value
        if baseline_value is None or value is None:
            row["primary_metric_delta_vs_baseline"] = None
            row["primary_metric_improvement_vs_baseline"] = None
            row["beats_baseline"] = None
            continue
        delta = float(value) - float(baseline_value)
        improvement = -delta if direction == "minimize" else delta
        row["primary_metric_delta_vs_baseline"] = delta
        row["primary_metric_improvement_vs_baseline"] = improvement
        row["beats_baseline"] = improvement > 0.0


def _selected_gamma_metrics_and_diagnostics(
    *,
    prediction_arrays: Mapping[str, Any],
    selected_gamma: float,
    gamma_report: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(prediction_arrays.get("labels"), dtype=np.int64)
    baseline_logits = np.asarray(prediction_arrays.get("baseline_logits"), dtype=np.float32)
    correction_logits = np.asarray(prediction_arrays.get("correction_logits"), dtype=np.float32)
    indices = np.asarray(prediction_arrays.get("indices"), dtype=np.int64)
    shrunk_logits = _shrunk_logits_from_prediction_arrays(prediction_arrays, gamma=float(selected_gamma))
    shrunk_margin = binary_margin_from_logits(shrunk_logits)
    learned_correction = binary_margin_from_logits(correction_logits)
    metrics = binary_metrics_from_signal_scores(shrunk_margin, labels)
    diagnostics = residual_correction_diagnostics(
        labels=labels,
        baseline_logits=baseline_logits,
        fused_logits=shrunk_logits,
        residual_logits=binary_logits_from_log_odds(float(selected_gamma) * learned_correction),
        indices=indices if indices.ndim == 1 else None,
        alpha_report={
            "selected_alpha": float(selected_gamma),
            "selected_gamma": float(selected_gamma),
            "shrinkage_applies_to": "learned_correction_delta",
            **(dict(gamma_report) if isinstance(gamma_report, Mapping) else {}),
        },
    )
    return metrics, diagnostics


def _learned_diagnostics_from_predictions(prediction_arrays: Mapping[str, Any]) -> dict[str, Any]:
    labels = np.asarray(prediction_arrays.get("labels"), dtype=np.int64)
    indices = np.asarray(prediction_arrays.get("indices"), dtype=np.int64)
    return residual_correction_diagnostics(
        labels=labels,
        baseline_logits=np.asarray(prediction_arrays.get("baseline_logits"), dtype=np.float32),
        fused_logits=np.asarray(prediction_arrays.get("fused_logits"), dtype=np.float32),
        residual_logits=np.asarray(prediction_arrays.get("correction_logits"), dtype=np.float32),
        indices=indices if indices.ndim == 1 else None,
    )


def _configure_train_config_for_report(
    payload_config: Mapping[str, Any],
    report_config: "LocalGraphResidualExpertV2ReportConfig",
) -> LocalGraphResidualExpertV2TrainConfig:
    train_config = LocalGraphResidualExpertV2TrainConfig(**dict(payload_config))
    train_config.hlt_cache_dir = str(report_config.hlt_cache_dir)
    train_config.baseline_embedding_cache_dir = str(report_config.baseline_embedding_cache_dir)
    train_config.verify_hlt_hash = bool(report_config.verify_hlt_hash)
    train_config.verify_hlt_params = bool(report_config.verify_hlt_params)
    train_config.expected_hlt_degradation_strength = float(report_config.expected_hlt_degradation_strength)
    train_config.num_workers = int(report_config.num_workers)
    train_config.eval_batch_size = int(report_config.batch_size)
    train_config.device = str(report_config.device)
    train_config.amp = bool(report_config.amp)
    return train_config


def _evaluate_v2_checkpoint(
    *,
    checkpoint_path: Path,
    split: str,
    report_config: "LocalGraphResidualExpertV2ReportConfig",
    baseline_block: LocalGraphResidualV2BaselineEmbeddingBlock,
    train_baseline_block: LocalGraphResidualV2BaselineEmbeddingBlock,
    selected_gamma: float | None,
    gamma_report: Mapping[str, Any] | None,
    device: Any,
) -> dict[str, Any]:
    model, payload = load_local_graph_residual_expert_v2_checkpoint(checkpoint_path, device=device)
    train_payload = payload.get("config")
    if not isinstance(train_payload, Mapping):
        raise ValueError(f"V2 residual checkpoint has no train config: {checkpoint_path}")
    train_config = _configure_train_config_for_report(train_payload, report_config)
    dataset = _load_residual_v2_dataset(train_config, split, max_jets=_split_max_jets(report_config, split))
    alignment = verify_residual_v2_embedding_block_alignment(
        baseline_block,
        dataset.metadata,
        split=split,
        dataset_length=len(dataset),
        expected_indices=np.arange(len(dataset), dtype=np.int64),
        expected_labels=np.asarray(dataset.labels, dtype=np.int64),
        expected_checkpoint_identity=train_baseline_block.metadata.get("checkpoint_identity"),
        expected_condition_reference=train_baseline_block.condition_reference(require=True),
        expected_embedding_dim=int(model.config.baseline_embedding_dim),
        require_hashes=bool(report_config.verify_hlt_hash),
    )
    loader = make_subtoken_hlt_loader(
        dataset,
        batch_size=int(report_config.batch_size),
        shuffle=False,
        num_workers=int(report_config.num_workers),
        seed=int(report_config.seed) + 211,
    )
    metrics = run_local_graph_residual_expert_v2_epoch(
        model,
        loader,
        baseline_block=baseline_block,
        device=device,
        loss_config=train_config.loss_config(train_baseline_block),
        amp=bool(report_config.amp),
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
        condition_control_mode=str(train_config.condition_control_mode),
        condition_shuffle_seed=int(train_config.condition_shuffle_seed),
        label_control_mode=str(train_config.label_control_mode),
        label_shuffle_seed=int(train_config.label_shuffle_seed),
    )
    prediction_arrays = metrics.get("_prediction_arrays", {})
    learned_diagnostics = None
    selected_diagnostics = None
    selected_metrics = None
    effective_gamma = selected_gamma
    effective_gamma_report = gamma_report
    if isinstance(prediction_arrays, Mapping) and np.asarray(prediction_arrays.get("labels")).size:
        if split == "model_val":
            effective_gamma_report = select_local_graph_residual_v2_gamma_shrinkage(
                labels=np.asarray(prediction_arrays.get("labels"), dtype=np.int64),
                baseline_logits=np.asarray(prediction_arrays.get("baseline_logits"), dtype=np.float32),
                correction_logits=np.asarray(prediction_arrays.get("correction_logits"), dtype=np.float32),
            )
            effective_gamma = _float_or_none(effective_gamma_report.get("selected_gamma"))
        learned_diagnostics = _learned_diagnostics_from_predictions(prediction_arrays)
        if effective_gamma is not None:
            selected_metrics, selected_diagnostics = _selected_gamma_metrics_and_diagnostics(
                prediction_arrays=prediction_arrays,
                selected_gamma=float(effective_gamma),
                gamma_report=effective_gamma_report,
            )
    return {
        "source": "checkpoint_evaluation",
        "fused_metrics": _strip_prediction_arrays(metrics.get("fused_metrics", metrics)),
        "baseline_metrics": _strip_prediction_arrays(metrics.get("baseline_metrics", {})),
        "residual_metrics": _strip_prediction_arrays(metrics.get("residual_metrics", {})),
        "correction_metrics": _strip_prediction_arrays(metrics.get("correction_metrics", {})),
        "learned_gamma_metrics": _strip_prediction_arrays(metrics.get("fused_metrics", metrics)),
        "selected_gamma_metrics": selected_metrics,
        "selected_gamma": effective_gamma,
        "gamma_shrinkage_model_val": effective_gamma_report if split == "model_val" else None,
        "learned_gamma_diagnostics": learned_diagnostics,
        "selected_gamma_diagnostics": selected_diagnostics,
        "baseline_cache_alignment": alignment,
        "dataset_metadata": dict(dataset.metadata),
    }


def _resolve_checkpoint_path(report_path: Path, run_report: Mapping[str, Any]) -> Path:
    checkpoint = run_report.get("checkpoint") or str(report_path.parent / "best_model_val.pt")
    checkpoint_path = Path(str(checkpoint))
    if checkpoint_path.is_absolute():
        return checkpoint_path
    direct = report_path.parent / checkpoint_path
    if direct.exists():
        return direct
    return report_path.parent / "best_model_val.pt"


def _append_external_metric_table(
    *,
    metric_rows: list[dict[str, Any]],
    report_path: Path,
    source_prefix: str,
    splits: Sequence[str],
    primary_metric: str,
    direction: str,
    problems: list[str],
) -> None:
    report = _read_json(report_path)
    if not isinstance(report, Mapping):
        problems.append(f"external report was not readable: {report_path}")
        return
    table = report.get("metric_table") or report.get("rows") or report.get("fusion_metric_table")
    if not isinstance(table, Sequence) or isinstance(table, (str, bytes)):
        problems.append(f"external report has no metric table: {report_path}")
        return
    for item in table:
        if not isinstance(item, Mapping):
            continue
        split = str(item.get("split") or "final_test")
        if split not in splits:
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {
            "accuracy": item.get("accuracy") or item.get(f"{split}_accuracy"),
            "auc": item.get("auc") or item.get(f"{split}_auc"),
            "fpr_at_signal_eff_0p30": item.get("fpr_at_signal_eff_0p30")
            or item.get(f"{split}_fpr_at_signal_eff_0p30"),
            "fpr_at_signal_eff_0p50": item.get("fpr_at_signal_eff_0p50")
            or item.get(f"{split}_fpr_at_signal_eff_0p50"),
            "background_rejection_at_signal_eff_0p50": item.get("background_rejection_at_signal_eff_0p50")
            or item.get(f"{split}_background_rejection_at_signal_eff_0p50"),
        }
        metric_rows.append(
            _metric_row(
                source_type=f"{source_prefix}_{item.get('source_type', item.get('method', 'external'))}",
                variant=str(item.get("variant") or item.get("method") or source_prefix),
                split=split,
                metrics=metrics,
                primary_metric=primary_metric,
                direction=direction,
                report_path=report_path,
                extra={
                    "score_type": item.get("score_type"),
                    "external_report_contract": report.get("output_contract") or report.get("contract"),
                },
            )
        )


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report.get("comparison_summary", {})
    rows = [row for row in report.get("metric_table", []) if row.get("split") == summary.get("comparison_split")]
    lines = [
        "# Local Graph Residual Expert V2 Report",
        "",
        f"- ok: {report.get('ok')}",
        f"- comparison split: {summary.get('comparison_split')}",
        f"- primary metric: {summary.get('primary_metric')} ({summary.get('primary_metric_direction')})",
        f"- baseline: {summary.get('baseline_metric_value')}",
        f"- best source: {summary.get('best_source_type')}",
        f"- best variant: {summary.get('best_variant')}",
        f"- best value: {summary.get('best_metric_value')}",
        "",
        "## Comparison",
        "",
        "| source | variant | primary | delta vs baseline | beats baseline |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {source} | {variant} | {primary} | {delta} | {beats} |".format(
                source=row.get("source_type"),
                variant=row.get("variant"),
                primary=row.get("primary_metric_value"),
                delta=row.get("primary_metric_delta_vs_baseline"),
                beats=row.get("beats_baseline"),
            )
        )
    problems = report.get("problems") or []
    if problems:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in problems)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class LocalGraphResidualExpertV2ReportConfig:
    """Configuration for Step 12 V2 residual-expert reporting."""

    output_dir: str
    hlt_cache_dir: str
    baseline_embedding_cache_dir: str
    residual_expert_root: str
    residual_variants: tuple[str, ...] = ()
    v1_residual_report_path: str | None = None
    score_fusion_report_path: str | None = None
    standalone_report_path: str | None = None
    primary_metric: str = LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC
    comparison_split: str = "final_test"
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    amp: bool = False
    seed: int = 9821
    max_model_val_jets: int | None = None
    max_stack_train_jets: int | None = None
    max_stack_val_jets: int | None = None
    max_final_test_jets: int | None = None
    confirm_final_test: bool = False
    include_calibration_control: bool = True
    require_all_residual_variants: bool = True
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    expected_hlt_degradation_strength: float = LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH

    def __post_init__(self) -> None:
        protocol = default_local_graph_residual_v2_protocol()
        if self.primary_metric not in LOCAL_GRAPH_SELECTION_METRICS:
            raise ValueError(f"primary_metric must be one of {LOCAL_GRAPH_SELECTION_METRICS}")
        if self.primary_metric != LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC:
            raise ValueError("V2 binary report must rank by final-test FPR@50, not accuracy")
        if self.comparison_split not in LOCAL_GRAPH_RESIDUAL_V2_REPORT_SPLITS:
            raise ValueError(f"comparison_split must be one of {LOCAL_GRAPH_RESIDUAL_V2_REPORT_SPLITS}")
        if self.comparison_split == "final_test" and not bool(self.confirm_final_test):
            raise ValueError("Set confirm_final_test=True before writing a final_test V2 residual report")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if self.primary_metric not in LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS:
            raise ValueError("V2 report primary metric must be lower-is-better FPR@50")
        if abs(float(self.expected_hlt_degradation_strength) - float(protocol.hlt_degradation_strength)) > 1.0e-12:
            raise ValueError("V2 residual report protocol is frozen to HLT degradation strength 0.6")


def build_local_graph_residual_expert_v2_report(config: LocalGraphResidualExpertV2ReportConfig) -> dict[str, Any]:
    """Build Step 12 V2 final comparison tables."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    direction = _metric_direction(config.primary_metric)
    if direction != "minimize":
        raise ValueError("V2 residual report must minimize FPR@50")
    splits = tuple(LOCAL_GRAPH_RESIDUAL_V2_REPORT_SPLITS)
    problems: list[str] = []

    train_block = load_residual_v2_embedding_block(config.baseline_embedding_cache_dir, "model_train", require_metadata=True)
    baseline_blocks = {
        split: load_residual_v2_embedding_block(config.baseline_embedding_cache_dir, split, require_metadata=True)
        for split in splits
    }
    cache_family = verify_residual_v2_embedding_cache_family(
        [train_block, *baseline_blocks.values()],
        require_condition_reference=True,
    )
    report_baseline_blocks = {
        split: _baseline_block_for_report_subset(block, max_jets=_split_max_jets(config, split))
        for split, block in baseline_blocks.items()
    }

    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for split, block in report_baseline_blocks.items():
        metric_rows.append(
            _baseline_row(
                block=block,
                split=split,
                primary_metric=config.primary_metric,
                direction=direction,
            )
        )
    calibration_report = None
    if bool(config.include_calibration_control):
        calibration_report = _fit_baseline_calibration(report_baseline_blocks["model_val"])
        for split, block in report_baseline_blocks.items():
            metric_rows.append(
                _calibrated_baseline_row(
                    block=block,
                    split=split,
                    calibration=calibration_report,
                    primary_metric=config.primary_metric,
                    direction=direction,
                )
            )

    device = resolve_device(config.device)
    require_torch()
    residual_reports = _discover_variant_reports(Path(config.residual_expert_root), config.residual_variants)
    for variant, report_path in residual_reports.items():
        run_report = _read_json(report_path)
        if not isinstance(run_report, Mapping):
            message = f"missing V2 residual run_report for {variant}: {report_path}"
            if bool(config.require_all_residual_variants):
                problems.append(message)
            continue
        checkpoint_path = _resolve_checkpoint_path(report_path, run_report)
        if not checkpoint_path.exists():
            problems.append(f"missing V2 checkpoint for {variant}: {checkpoint_path}")
            continue
        selected_gamma: float | None = None
        gamma_report: Mapping[str, Any] | None = None
        for split in splits:
            evaluation = _evaluate_v2_checkpoint(
                checkpoint_path=checkpoint_path,
                split=split,
                report_config=config,
                baseline_block=baseline_blocks[split],
                train_baseline_block=train_block,
                selected_gamma=selected_gamma,
                gamma_report=gamma_report,
                device=device,
            )
            if split == "model_val":
                selected_gamma = _float_or_none(evaluation.get("selected_gamma"))
                gamma_report = evaluation.get("gamma_shrinkage_model_val")

            row_extra = {
                "score_type": "learned_gamma",
                "loss_mode": run_report.get("loss_mode")
                or ((run_report.get("loss_config") or {}).get("mode") if isinstance(run_report.get("loss_config"), Mapping) else None),
                "best_epoch": run_report.get("best_epoch"),
                "checkpoint": str(checkpoint_path),
                "selected_gamma_model_val": selected_gamma,
                "evaluation_source": evaluation.get("source"),
                "runtime_seconds": ((run_report.get("runtime") or {}).get("elapsed_seconds") if isinstance(run_report.get("runtime"), Mapping) else run_report.get("walltime_seconds")),
            }
            metric_rows.append(
                _metric_row(
                    source_type="v2_residual_fused_learned_gamma",
                    variant=str(variant),
                    split=split,
                    metrics=evaluation.get("learned_gamma_metrics"),
                    primary_metric=config.primary_metric,
                    direction=direction,
                    report_path=report_path,
                    extra=row_extra,
                )
            )
            if isinstance(evaluation.get("selected_gamma_metrics"), Mapping):
                metric_rows.append(
                    _metric_row(
                        source_type="v2_residual_fused_val_shrunk",
                        variant=f"{variant}__val_shrunk",
                        split=split,
                        metrics=evaluation.get("selected_gamma_metrics"),
                        primary_metric=config.primary_metric,
                        direction=direction,
                        report_path=report_path,
                        extra={
                            **row_extra,
                            "score_type": "validation_shrunk",
                            "shrinkage_applies_to": "learned_correction_delta",
                        },
                    )
                )
            if isinstance(evaluation.get("residual_metrics"), Mapping):
                metric_rows.append(
                    _metric_row(
                        source_type="v2_residual_only_control",
                        variant=f"{variant}__residual_only",
                        split=split,
                        metrics=evaluation.get("residual_metrics"),
                        primary_metric=config.primary_metric,
                        direction=direction,
                        report_path=report_path,
                        extra={**row_extra, "score_type": "residual_only"},
                    )
                )
            learned_diagnostics = evaluation.get("learned_gamma_diagnostics")
            selected_diagnostics = evaluation.get("selected_gamma_diagnostics")
            diagnostic_rows.extend(
                _diagnostic_rows(
                    variant=str(variant),
                    split=split,
                    diagnostics=learned_diagnostics,
                    report_path=report_path,
                )
            )
            diagnostic_rows.extend(
                _diagnostic_rows(
                    variant=f"{variant}__val_shrunk",
                    split=split,
                    diagnostics=selected_diagnostics,
                    report_path=report_path,
                )
            )
            if isinstance(learned_diagnostics, Mapping):
                _write_json(diagnostics_dir / f"{variant}_{split}_learned_gamma_diagnostics.json", learned_diagnostics)
            if isinstance(selected_diagnostics, Mapping):
                _write_json(diagnostics_dir / f"{variant}_{split}_val_shrunk_diagnostics.json", selected_diagnostics)

    for external_path, prefix in (
        (config.v1_residual_report_path, "v1"),
        (config.score_fusion_report_path, "score_fusion_control"),
        (config.standalone_report_path, "standalone_control"),
    ):
        if external_path:
            _append_external_metric_table(
                metric_rows=metric_rows,
                report_path=Path(external_path),
                source_prefix=prefix,
                splits=splits,
                primary_metric=config.primary_metric,
                direction=direction,
                problems=problems,
            )

    baseline_by_split = {
        split: _metric(report_baseline_blocks[split].metrics(), config.primary_metric)
        for split in splits
    }
    _apply_baseline_deltas(metric_rows, baseline_by_split=baseline_by_split)
    comparison_rows = [row for row in metric_rows if row.get("split") == config.comparison_split]
    best = _best_row(comparison_rows, direction)
    baseline_value = baseline_by_split.get(config.comparison_split)
    if best is None:
        problems.append(f"no rows had {config.comparison_split} {config.primary_metric}")

    output_paths = {
        "report_json": str(output_dir / "local_graph_residual_expert_v2_report.json"),
        "report_markdown": str(output_dir / "local_graph_residual_expert_v2_report.md"),
        "metric_table_csv": str(output_dir / "metric_table.csv"),
        "diagnostics_csv": str(output_dir / "residual_v2_diagnostics.csv"),
        "baseline_comparison_csv": str(output_dir / "baseline_comparison.csv"),
        "run_report": str(output_dir / "run_report.json"),
    }
    report = {
        "experiment_step": LOCAL_GRAPH_RESIDUAL_V2_REPORT_STEP,
        "output_contract": LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONTRACT,
        "protocol_step": LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_STEP,
        "protocol_contract": LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_CONTRACT,
        "base_protocol_contract": LOCAL_GRAPH_PART_CONTRACT,
        "baseline_embedding_cache_contract": LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
        "ok": len(problems) == 0,
        "problems": problems,
        "config": asdict(config),
        "protocol": local_graph_residual_v2_protocol_manifest(),
        "source": source_metadata(),
        "comparison_summary": {
            "comparison_split": config.comparison_split,
            "primary_metric": config.primary_metric,
            "primary_metric_direction": direction,
            "baseline_variant": LOCAL_GRAPH_RESIDUAL_V2_BASELINE_REPORT_VARIANT,
            "baseline_metric_value": baseline_value,
            "best_source_type": best.get("source_type") if best is not None else None,
            "best_variant": best.get("variant") if best is not None else None,
            "best_metric_value": best.get("primary_metric_value") if best is not None else None,
            "best_improvement_vs_baseline": best.get("primary_metric_improvement_vs_baseline")
            if best is not None
            else None,
            "hlt_degradation_strength": LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH,
            "rule": "Primary comparison is final-test FPR@50, lower is better.",
            "gamma_reporting_rule": (
                "V2 rows are split into learned-gamma and model-val validation-shrunk rows; "
                "gamma shrinkage is applied to the learned correction delta."
            ),
        },
        "baseline_embedding_cache_family": cache_family,
        "calibration_only_control": calibration_report,
        "metric_table": metric_rows,
        "residual_diagnostics": diagnostic_rows,
        "outputs": output_paths,
    }
    _write_csv(Path(output_paths["metric_table_csv"]), metric_rows)
    _write_csv(Path(output_paths["diagnostics_csv"]), diagnostic_rows)
    _write_csv(Path(output_paths["baseline_comparison_csv"]), comparison_rows)
    _write_json(Path(output_paths["report_json"]), report)
    _write_json(Path(output_paths["run_report"]), report)
    _write_markdown(Path(output_paths["report_markdown"]), report)
    return report


__all__ = [
    "LOCAL_GRAPH_RESIDUAL_V2_BASELINE_REPORT_VARIANT",
    "LOCAL_GRAPH_RESIDUAL_V2_REPORT_SPLITS",
    "LocalGraphResidualExpertV2ReportConfig",
    "build_local_graph_residual_expert_v2_report",
]

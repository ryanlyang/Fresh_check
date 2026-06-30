"""Final report builder for local-graph residual expert experiments."""

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

from .fusion import binary_margin_from_logits, binary_metrics_from_signal_scores
from .protocol import (
    LOCAL_GRAPH_PART_CONTRACT,
    LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_PART_PROTOCOL_STEP,
    LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
    local_graph_part_protocol_manifest,
)
from .residual_cache import (
    LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CONTRACT,
    LocalGraphBaselineLogitBlock,
    load_baseline_logit_block,
    save_json,
    verify_baseline_logit_block_alignment,
    verify_baseline_logit_cache_family,
)
from .residual_diagnostics import residual_correction_diagnostics
from .residual_losses import select_alpha_shrinkage
from .residual_train import (
    LocalGraphResidualExpertTrainConfig,
    _load_residual_dataset,
    _strip_prediction_arrays,
    load_local_graph_residual_expert_checkpoint,
    run_local_graph_residual_expert_epoch,
)
from .train import LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS, LOCAL_GRAPH_SELECTION_METRICS


LOCAL_GRAPH_RESIDUAL_REPORT_STEP = "local_graph_residual_expert_step9_report"
LOCAL_GRAPH_RESIDUAL_REPORT_CONTRACT = "local_graph_residual_expert_report_v1"
LOCAL_GRAPH_RESIDUAL_PRECOMPUTED_EVAL_CONTRACT = "local_graph_residual_precomputed_eval_v2"
LOCAL_GRAPH_RESIDUAL_REPORT_SPLITS = ("model_val", "stack_val", "final_test")
LOCAL_GRAPH_RESIDUAL_BASELINE_VARIANT = "frozen_hlt_part_baseline"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
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
            if str(key) not in seen:
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


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _float_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _metric(metrics: Mapping[str, Any] | None, metric_name: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    direct = _float_or_none(metrics.get(metric_name))
    if direct is not None:
        return direct
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping):
        return _float_or_none(binary.get(metric_name))
    return None


def _metric_direction(metric_name: str) -> str:
    return "minimize" if metric_name in LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS else "maximize"


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


def _split_max_jets(config: "LocalGraphResidualExpertReportConfig", split: str) -> int | None:
    if split == "model_val":
        return config.max_model_val_jets
    if split == "stack_val":
        return config.max_stack_val_jets
    if split == "final_test":
        return config.max_final_test_jets
    return None


def _metrics_for_standalone_split(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    if split == "model_val":
        for key in ("best_model_val_metrics", "model_val_metrics"):
            metrics = report.get(key)
            if isinstance(metrics, Mapping):
                return metrics
    if split == "stack_val":
        for key in ("stack_val_metrics", "best_stack_val_metrics"):
            metrics = report.get(key)
            if isinstance(metrics, Mapping):
                return metrics
    if split == "final_test":
        metrics = report.get("final_test_metrics")
        if isinstance(metrics, Mapping):
            return metrics
    nested = report.get("evaluations")
    if isinstance(nested, Mapping):
        split_payload = nested.get(split)
        if isinstance(split_payload, Mapping):
            metrics = split_payload.get("metrics")
            if isinstance(metrics, Mapping):
                return metrics
    return None


def _residual_precomputed_eval(report: Mapping[str, Any], split: str) -> dict[str, Any] | None:
    evaluations = report.get("evaluations")
    if isinstance(evaluations, Mapping):
        payload = evaluations.get(split)
        if isinstance(payload, Mapping):
            return dict(payload)
    if split == "model_val":
        fused = report.get("fused_model_val_metrics")
        baseline = report.get("baseline_model_val_metrics")
        residual = report.get("residual_model_val_metrics")
        diagnostics = report.get("residual_diagnostics_model_val")
        if any(isinstance(item, Mapping) for item in (fused, baseline, residual, diagnostics)):
            return {
                "fused_metrics": fused if isinstance(fused, Mapping) else None,
                "baseline_metrics": baseline if isinstance(baseline, Mapping) else None,
                "residual_metrics": residual if isinstance(residual, Mapping) else None,
                "residual_diagnostics": diagnostics if isinstance(diagnostics, Mapping) else None,
                "source": "model_val_training_report",
            }
    return None


def _validate_precomputed_residual_report(report: Mapping[str, Any], *, variant: str) -> list[str]:
    problems: list[str] = []
    if report.get("precomputed_evaluation_contract") != LOCAL_GRAPH_RESIDUAL_PRECOMPUTED_EVAL_CONTRACT:
        problems.append(
            f"{variant}: precomputed residual evaluations require contract "
            f"{LOCAL_GRAPH_RESIDUAL_PRECOMPUTED_EVAL_CONTRACT!r}"
        )
    alpha = report.get("alpha_shrinkage_model_val")
    if not isinstance(alpha, Mapping) or alpha.get("shrinkage_applies_to") != "learned_correction_delta":
        problems.append(f"{variant}: alpha_shrinkage_model_val must use learned_correction_delta semantics")
    alignment = report.get("baseline_cache_alignment")
    if not isinstance(alignment, Mapping):
        problems.append(f"{variant}: precomputed payload is missing baseline_cache_alignment")
    else:
        family = alignment.get("family")
        if not isinstance(family, Mapping):
            problems.append(f"{variant}: precomputed payload is missing baseline_cache_alignment.family")
        else:
            reference = family.get("condition_reference")
            if not isinstance(reference, Mapping) or reference.get("source_split") != "model_train":
                problems.append(f"{variant}: precomputed condition_reference is not sourced from model_train")
            checkpoint_identity = family.get("checkpoint_identity")
            if not isinstance(checkpoint_identity, Mapping):
                problems.append(f"{variant}: precomputed payload is missing baseline checkpoint identity")
            elif checkpoint_identity.get("checkpoint_variant") != "hlt_part_baseline":
                problems.append(f"{variant}: precomputed baseline checkpoint is not hlt_part_baseline")
    evaluations = report.get("evaluations")
    if isinstance(evaluations, Mapping):
        for split, payload in evaluations.items():
            if not isinstance(payload, Mapping):
                continue
            if payload.get("selected_alpha_metrics") is not None and not isinstance(alpha, Mapping):
                problems.append(f"{variant}/{split}: selected-alpha metrics exist without alpha_shrinkage_model_val")
    return problems


def _baseline_row(
    *,
    block: LocalGraphBaselineLogitBlock,
    split: str,
    primary_metric: str,
    direction: str,
) -> dict[str, Any]:
    metrics = block.metrics()
    return {
        "source_type": "baseline",
        "variant": LOCAL_GRAPH_RESIDUAL_BASELINE_VARIANT,
        "split": split,
        "primary_metric": primary_metric,
        "primary_metric_direction": direction,
        "primary_metric_value": _metric(metrics, primary_metric),
        "accuracy": _metric(metrics, "accuracy"),
        "auc": _metric(metrics, "auc"),
        "fpr_at_signal_eff_0p30": _metric(metrics, "fpr_at_signal_eff_0p30"),
        "fpr_at_signal_eff_0p50": _metric(metrics, "fpr_at_signal_eff_0p50"),
        "background_rejection_at_signal_eff_0p50": _metric(
            metrics, "background_rejection_at_signal_eff_0p50"
        ),
        "n_jets": int(block.labels.shape[0]),
        "metrics": metrics,
    }


def _baseline_block_for_report_subset(
    block: LocalGraphBaselineLogitBlock,
    *,
    max_jets: int | None,
) -> LocalGraphBaselineLogitBlock:
    if max_jets is None or int(max_jets) >= int(block.labels.shape[0]):
        return block
    n_jets = max(int(max_jets), 0)
    condition_features = None
    if block.condition_features_array is not None:
        condition_features = block.condition_features_array[:n_jets]
    return LocalGraphBaselineLogitBlock(
        split=block.split,
        logits=block.logits[:n_jets],
        labels=block.labels[:n_jets],
        indices=block.indices[:n_jets],
        metadata=dict(block.metadata),
        condition_features_array=condition_features,
    )


def _metric_row(
    *,
    source_type: str,
    variant: str,
    split: str,
    metrics: Mapping[str, Any] | None,
    primary_metric: str,
    direction: str,
    report_path: Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "source_type": source_type,
        "variant": variant,
        "split": split,
        "report_path": str(report_path) if report_path is not None else None,
        "primary_metric": primary_metric,
        "primary_metric_direction": direction,
        "primary_metric_value": _metric(metrics, primary_metric),
        "accuracy": _metric(metrics, "accuracy"),
        "auc": _metric(metrics, "auc"),
        "fpr_at_signal_eff_0p30": _metric(metrics, "fpr_at_signal_eff_0p30"),
        "fpr_at_signal_eff_0p50": _metric(metrics, "fpr_at_signal_eff_0p50"),
        "background_rejection_at_signal_eff_0p50": _metric(
            metrics, "background_rejection_at_signal_eff_0p50"
        ),
        "n_jets": int(metrics.get("n_jets", 0)) if isinstance(metrics, Mapping) else None,
        "metrics": metrics if isinstance(metrics, Mapping) else None,
    }
    if extra:
        row.update(dict(extra))
    return row


def _apply_baseline_deltas(rows: list[dict[str, Any]], *, baseline_by_split: Mapping[str, float | None]) -> None:
    for row in rows:
        baseline_value = baseline_by_split.get(str(row.get("split")))
        value = _float_or_none(row.get("primary_metric_value"))
        direction = str(row.get("primary_metric_direction", "minimize"))
        row["baseline_variant"] = LOCAL_GRAPH_RESIDUAL_BASELINE_VARIANT
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


def _diagnostic_rows(
    *,
    variant: str,
    split: str,
    diagnostics: Mapping[str, Any] | None,
    report_path: Path | None,
) -> list[dict[str, Any]]:
    if not isinstance(diagnostics, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for key, value in diagnostics.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "diagnostic": f"{key}.{sub_key}",
                        "value": _float_or_none(sub_value),
                        "raw_value": sub_value,
                        "report_path": str(report_path) if report_path is not None else None,
                    }
                )
        else:
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "diagnostic": key,
                    "value": _float_or_none(value),
                    "raw_value": value,
                    "report_path": str(report_path) if report_path is not None else None,
                }
            )
    return rows


def _resolve_report_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    return path


def _discover_variant_reports(root: Path, variants: Sequence[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    requested = [str(item) for item in variants if str(item)]
    if requested:
        for variant in requested:
            paths[variant] = root / variant / "run_report.json"
    else:
        for report_path in sorted(root.glob("*/run_report.json")):
            if report_path.parent.name not in {"final_report", "score_fusion"}:
                paths[report_path.parent.name] = report_path
    return paths


def _selected_alpha_from_report(report: Mapping[str, Any]) -> float | None:
    alpha = report.get("alpha_shrinkage_model_val")
    if isinstance(alpha, Mapping):
        if alpha.get("shrinkage_applies_to") != "learned_correction_delta":
            raise ValueError("alpha_shrinkage_model_val must use learned_correction_delta semantics")
        return _float_or_none(alpha.get("selected_alpha"))
    return None


def _selected_score_diagnostics(
    *,
    labels: np.ndarray,
    baseline_logits: np.ndarray,
    correction_logits: np.ndarray,
    selected_gamma: float | None,
    indices: np.ndarray | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    learned_fused = binary_margin_from_logits(baseline_logits) + binary_margin_from_logits(correction_logits)
    learned_fused_logits_np = np.stack((-0.5 * learned_fused, 0.5 * learned_fused), axis=1).astype(np.float32)
    learned_diagnostics = residual_correction_diagnostics(
        labels=labels,
        baseline_logits=baseline_logits,
        fused_logits=learned_fused_logits_np,
        residual_logits=correction_logits,
        indices=indices,
    )
    if selected_gamma is None:
        return learned_diagnostics, None
    selected_fused = binary_margin_from_logits(baseline_logits) + float(selected_gamma) * binary_margin_from_logits(
        correction_logits
    )
    selected_fused_logits_np = np.stack((-0.5 * selected_fused, 0.5 * selected_fused), axis=1).astype(np.float32)
    selected_diagnostics = residual_correction_diagnostics(
        labels=labels,
        baseline_logits=baseline_logits,
        fused_logits=selected_fused_logits_np,
        residual_logits=np.stack(
            (
                -0.5 * float(selected_gamma) * binary_margin_from_logits(correction_logits),
                0.5 * float(selected_gamma) * binary_margin_from_logits(correction_logits),
            ),
            axis=1,
        ).astype(np.float32),
        indices=indices,
        alpha_report={
            "selected_alpha": float(selected_gamma),
            "shrinkage_applies_to": "learned_correction_delta",
        },
    )
    return learned_diagnostics, selected_diagnostics


def _evaluate_residual_checkpoint(
    *,
    checkpoint_path: Path,
    split: str,
    report_config: "LocalGraphResidualExpertReportConfig",
    baseline_block: LocalGraphBaselineLogitBlock,
    train_baseline_block: LocalGraphBaselineLogitBlock,
    selected_alpha: float | None,
    device: Any,
) -> dict[str, Any]:
    model, payload = load_local_graph_residual_expert_checkpoint(checkpoint_path, device=device)
    train_payload = payload.get("config")
    if not isinstance(train_payload, Mapping):
        raise ValueError(f"residual checkpoint has no train config: {checkpoint_path}")
    train_config = LocalGraphResidualExpertTrainConfig(**dict(train_payload))
    train_config.hlt_cache_dir = str(report_config.hlt_cache_dir)
    train_config.baseline_logit_cache_dir = str(report_config.baseline_logit_cache_dir)
    train_config.verify_hlt_hash = bool(report_config.verify_hlt_hash)
    train_config.verify_hlt_params = bool(report_config.verify_hlt_params)
    train_config.expected_hlt_degradation_strength = float(report_config.expected_hlt_degradation_strength)
    train_config.num_workers = int(report_config.num_workers)
    dataset = _load_residual_dataset(train_config, split, max_jets=_split_max_jets(report_config, split))
    alignment = verify_baseline_logit_block_alignment(
        baseline_block,
        dataset.metadata,
        split=split,
        dataset_length=len(dataset),
    )
    loader = make_subtoken_hlt_loader(
        dataset,
        batch_size=int(report_config.batch_size),
        shuffle=False,
        num_workers=int(report_config.num_workers),
        seed=int(report_config.seed) + 101,
    )
    metrics = run_local_graph_residual_expert_epoch(
        model,
        loader,
        baseline_block=baseline_block,
        device=device,
        loss_config=train_config.loss_config(train_baseline_block),
        amp=bool(report_config.amp),
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
    )
    prediction_arrays = metrics.get("_prediction_arrays", {})
    learned_diagnostics = None
    selected_diagnostics = None
    shrink_metrics = None
    alpha_report = None
    effective_selected_alpha = selected_alpha
    if isinstance(prediction_arrays, Mapping):
        labels = np.asarray(prediction_arrays.get("labels"), dtype=np.int64)
        baseline_logits = np.asarray(prediction_arrays.get("baseline_logits"), dtype=np.float32)
        fused_logits = np.asarray(prediction_arrays.get("fused_logits"), dtype=np.float32)
        residual_logits = np.asarray(prediction_arrays.get("residual_logits"), dtype=np.float32)
        correction_logits = np.asarray(prediction_arrays.get("correction_logits", residual_logits), dtype=np.float32)
        indices = np.asarray(prediction_arrays.get("indices"), dtype=np.int64)
        if labels.ndim == 1 and baseline_logits.ndim == 2 and fused_logits.ndim == 2 and correction_logits.ndim == 2:
            if split == "model_val":
                alpha_report = select_alpha_shrinkage(
                    labels=labels,
                    baseline_logit=binary_margin_from_logits(baseline_logits),
                    residual_logit=binary_margin_from_logits(correction_logits),
                )
                alpha_report["shrinkage_applies_to"] = "learned_correction_delta"
                effective_selected_alpha = _float_or_none(alpha_report.get("selected_alpha"))
            learned_diagnostics, selected_diagnostics = _selected_score_diagnostics(
                labels=labels,
                baseline_logits=baseline_logits,
                correction_logits=correction_logits,
                selected_gamma=effective_selected_alpha,
                indices=indices if indices.ndim == 1 else None,
            )
            if effective_selected_alpha is not None:
                scores = binary_margin_from_logits(baseline_logits) + float(effective_selected_alpha) * binary_margin_from_logits(correction_logits)
                shrink_metrics = binary_metrics_from_signal_scores(scores, labels)
    output = {
        "fused_metrics": _strip_prediction_arrays(metrics.get("fused_metrics", metrics)),
        "baseline_metrics": _strip_prediction_arrays(metrics.get("baseline_metrics", {})),
        "residual_metrics": _strip_prediction_arrays(metrics.get("residual_metrics", {})),
        "learned_alpha_metrics": _strip_prediction_arrays(metrics.get("fused_metrics", metrics)),
        "selected_alpha_metrics": shrink_metrics,
        "selected_alpha": effective_selected_alpha,
        "alpha_shrinkage_model_val": alpha_report,
        "learned_alpha_diagnostics": learned_diagnostics,
        "selected_alpha_diagnostics": selected_diagnostics,
        "residual_diagnostics": selected_diagnostics or learned_diagnostics,
        "baseline_cache_alignment": alignment,
        "dataset_metadata": dict(dataset.metadata),
        "source": "checkpoint_evaluation",
    }
    return output


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report.get("comparison_summary", {})
    rows = [row for row in report.get("metric_table", []) if row.get("split") == summary.get("comparison_split")]
    lines = [
        "# Local Graph Residual Expert Report",
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
class LocalGraphResidualExpertReportConfig:
    """Configuration for Step 9 residual-expert reporting."""

    output_dir: str
    hlt_cache_dir: str
    baseline_logit_cache_dir: str
    residual_expert_root: str
    residual_variants: tuple[str, ...] = ()
    standalone_tagger_root: str | None = None
    standalone_variants: tuple[str, ...] = ()
    score_fusion_report_path: str | None = None
    primary_metric: str = LOCAL_GRAPH_PART_PRIMARY_METRIC
    comparison_split: str = "final_test"
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    amp: bool = False
    seed: int = 9713
    max_model_val_jets: int | None = None
    max_stack_val_jets: int | None = None
    max_final_test_jets: int | None = None
    confirm_final_test: bool = False
    evaluate_checkpoints: bool = True
    allow_precomputed_evaluations: bool = False
    require_all_residual_variants: bool = True
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    expected_hlt_degradation_strength: float = LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH

    def __post_init__(self) -> None:
        if self.primary_metric not in LOCAL_GRAPH_SELECTION_METRICS:
            raise ValueError(f"primary_metric must be one of {LOCAL_GRAPH_SELECTION_METRICS}")
        if self.comparison_split not in LOCAL_GRAPH_RESIDUAL_REPORT_SPLITS:
            raise ValueError(f"comparison_split must be one of {LOCAL_GRAPH_RESIDUAL_REPORT_SPLITS}")
        if self.comparison_split == "final_test" and not bool(self.confirm_final_test):
            raise ValueError("Set confirm_final_test=True before writing a final_test residual report")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if abs(float(self.expected_hlt_degradation_strength) - LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
            raise ValueError("residual report protocol is frozen to HLT degradation strength 0.6")
        if not bool(self.evaluate_checkpoints) and not bool(self.allow_precomputed_evaluations):
            raise ValueError(
                "skip-checkpoint-evaluation is unsafe for final numbers unless "
                "allow_precomputed_evaluations=True and each run_report carries the precomputed-eval contract"
            )


def build_local_graph_residual_expert_report(config: LocalGraphResidualExpertReportConfig) -> dict[str, Any]:
    """Build Step 9 comparison tables for residual-expert experiments."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    direction = _metric_direction(config.primary_metric)
    splits = tuple(LOCAL_GRAPH_RESIDUAL_REPORT_SPLITS)
    problems: list[str] = []

    baseline_blocks: dict[str, LocalGraphBaselineLogitBlock] = {}
    for split in splits:
        baseline_blocks[split] = load_baseline_logit_block(config.baseline_logit_cache_dir, split, require_metadata=True)
    train_baseline_block = load_baseline_logit_block(config.baseline_logit_cache_dir, "model_train", require_metadata=True)
    baseline_cache_family = verify_baseline_logit_cache_family(
        [train_baseline_block, *baseline_blocks.values()],
        require_condition_reference=True,
    )

    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    report_baseline_blocks = {
        split: _baseline_block_for_report_subset(block, max_jets=_split_max_jets(config, split))
        for split, block in baseline_blocks.items()
    }
    for split, block in baseline_blocks.items():
        metric_rows.append(
            _baseline_row(
                block=report_baseline_blocks[split],
                split=split,
                primary_metric=config.primary_metric,
                direction=direction,
            )
        )

    device = resolve_device(config.device) if bool(config.evaluate_checkpoints) else None
    residual_reports = _discover_variant_reports(Path(config.residual_expert_root), config.residual_variants)
    for variant, report_path in residual_reports.items():
        run_report = _read_json(report_path)
        if not isinstance(run_report, Mapping):
            message = f"missing residual run_report for {variant}: {report_path}"
            if bool(config.require_all_residual_variants):
                problems.append(message)
            continue
        if not bool(config.evaluate_checkpoints):
            precomputed_problems = _validate_precomputed_residual_report(run_report, variant=str(variant))
            if precomputed_problems:
                problems.extend(precomputed_problems)
                continue
        selected_alpha = _selected_alpha_from_report(run_report) if not bool(config.evaluate_checkpoints) else None
        checkpoint = run_report.get("checkpoint") or str(report_path.parent / "best_model_val.pt")
        checkpoint_path = Path(str(checkpoint))
        if not checkpoint_path.is_absolute():
            checkpoint_path = report_path.parent / checkpoint_path

        for split in splits:
            if bool(config.evaluate_checkpoints):
                evaluation = _evaluate_residual_checkpoint(
                    checkpoint_path=checkpoint_path,
                    split=split,
                    report_config=config,
                    baseline_block=baseline_blocks[split],
                    train_baseline_block=train_baseline_block,
                    selected_alpha=selected_alpha,
                    device=device,
                )
                if split == "model_val":
                    selected_alpha = evaluation.get("selected_alpha", selected_alpha)
            else:
                evaluation = _residual_precomputed_eval(run_report, split)
                if evaluation is None:
                    problems.append(f"missing precomputed {split} evaluation for residual variant {variant}")
                    continue
            learned_metrics = evaluation.get("learned_alpha_metrics") or evaluation.get("fused_metrics")
            selected_metrics = evaluation.get("selected_alpha_metrics")
            row_extra = {
                "loss_mode": (run_report.get("loss_config") or {}).get("mode")
                if isinstance(run_report.get("loss_config"), Mapping)
                else None,
                "best_epoch": run_report.get("best_epoch"),
                "selected_alpha": evaluation.get("selected_alpha", selected_alpha),
                "shrinkage_applies_to": "learned_correction_delta",
                "evaluation_source": evaluation.get("source"),
            }
            metric_rows.append(
                _metric_row(
                    source_type="residual_fused_learned_alpha",
                    variant=str(variant),
                    split=split,
                    metrics=learned_metrics if isinstance(learned_metrics, Mapping) else None,
                    primary_metric=config.primary_metric,
                    direction=direction,
                    report_path=report_path,
                    extra={**row_extra, "alpha_policy": "learned_alpha"},
                )
            )
            if isinstance(selected_metrics, Mapping):
                metric_rows.append(
                    _metric_row(
                        source_type="residual_fused_val_shrunk",
                        variant=f"{variant}__val_shrunk",
                        split=split,
                        metrics=selected_metrics,
                        primary_metric=config.primary_metric,
                        direction=direction,
                        report_path=report_path,
                        extra={**row_extra, "alpha_policy": "model_val_gamma_shrink"},
                    )
                )
            residual_metrics = evaluation.get("residual_metrics")
            if isinstance(residual_metrics, Mapping):
                metric_rows.append(
                    _metric_row(
                        source_type="residual_only_control",
                        variant=f"{variant}__residual_only",
                        split=split,
                        metrics=residual_metrics,
                        primary_metric=config.primary_metric,
                        direction=direction,
                        report_path=report_path,
                        extra=row_extra,
                    )
                )
            diagnostics = evaluation.get("learned_alpha_diagnostics") or evaluation.get("residual_diagnostics")
            diagnostic_rows.extend(
                _diagnostic_rows(
                    variant=str(variant),
                    split=split,
                    diagnostics=diagnostics,
                    report_path=report_path,
                )
            )
            selected_diagnostics = evaluation.get("selected_alpha_diagnostics")
            diagnostic_rows.extend(
                _diagnostic_rows(
                    variant=f"{variant}__val_shrunk",
                    split=split,
                    diagnostics=selected_diagnostics,
                    report_path=report_path,
                )
            )
            if isinstance(diagnostics, Mapping):
                save_json(diagnostics_dir / f"{variant}_{split}_residual_diagnostics.json", diagnostics)
            if isinstance(selected_diagnostics, Mapping):
                save_json(
                    diagnostics_dir / f"{variant}_{split}_val_shrunk_residual_diagnostics.json",
                    selected_diagnostics,
                )

    standalone_root = Path(config.standalone_tagger_root) if config.standalone_tagger_root else None
    if standalone_root is not None:
        for variant, report_path in _discover_variant_reports(standalone_root, config.standalone_variants).items():
            run_report = _read_json(report_path)
            if not isinstance(run_report, Mapping):
                problems.append(f"missing standalone local-graph report for {variant}: {report_path}")
                continue
            for split in splits:
                metrics = _metrics_for_standalone_split(run_report, split)
                if metrics is None:
                    continue
                metric_rows.append(
                    _metric_row(
                        source_type="standalone_local_graph",
                        variant=str(variant),
                        split=split,
                        metrics=metrics,
                        primary_metric=config.primary_metric,
                        direction=direction,
                        report_path=report_path,
                        extra={"best_epoch": run_report.get("best_epoch")},
                    )
                )

    fusion_control_rows: list[dict[str, Any]] = []
    if config.score_fusion_report_path:
        fusion_report_path = Path(config.score_fusion_report_path)
        fusion_report = _read_json(fusion_report_path)
        if isinstance(fusion_report, Mapping):
            for row in fusion_report.get("fusion_metric_table", []) or []:
                if not isinstance(row, Mapping):
                    continue
                metrics = {
                    "accuracy": row.get("final_test_accuracy"),
                    "auc": row.get("final_test_auc"),
                    "fpr_at_signal_eff_0p30": row.get("final_test_fpr_at_signal_eff_0p30"),
                    "fpr_at_signal_eff_0p50": row.get("final_test_fpr_at_signal_eff_0p50"),
                    "background_rejection_at_signal_eff_0p50": row.get(
                        "final_test_background_rejection_at_signal_eff_0p50"
                    ),
                }
                metric_row = _metric_row(
                    source_type="score_fusion_control",
                    variant=str(row.get("method", "score_fusion")),
                    split="final_test",
                    metrics=metrics,
                    primary_metric=config.primary_metric,
                    direction=direction,
                    report_path=fusion_report_path,
                    extra={"model_set": row.get("model_set"), "is_control": row.get("is_control")},
                )
                metric_rows.append(metric_row)
                fusion_control_rows.append(metric_row)
        else:
            problems.append(f"score fusion report was not readable: {fusion_report_path}")

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
        "report_json": str(output_dir / "local_graph_residual_expert_report.json"),
        "report_markdown": str(output_dir / "local_graph_residual_expert_report.md"),
        "metric_table_csv": str(output_dir / "residual_metric_table.csv"),
        "diagnostics_csv": str(output_dir / "residual_diagnostics.csv"),
        "baseline_comparison_csv": str(output_dir / "baseline_comparison.csv"),
        "fusion_controls_csv": str(output_dir / "fusion_controls.csv"),
        "run_report": str(output_dir / "run_report.json"),
    }
    report = {
        "experiment_step": LOCAL_GRAPH_RESIDUAL_REPORT_STEP,
        "output_contract": LOCAL_GRAPH_RESIDUAL_REPORT_CONTRACT,
        "protocol_step": LOCAL_GRAPH_PART_PROTOCOL_STEP,
        "protocol_contract": LOCAL_GRAPH_PART_CONTRACT,
        "baseline_logit_cache_contract": LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CONTRACT,
        "ok": len(problems) == 0,
        "problems": problems,
        "config": asdict(config),
        "protocol": local_graph_part_protocol_manifest(),
        "source": source_metadata(),
        "comparison_summary": {
            "comparison_split": config.comparison_split,
            "primary_metric": config.primary_metric,
            "primary_metric_direction": direction,
            "baseline_variant": LOCAL_GRAPH_RESIDUAL_BASELINE_VARIANT,
            "baseline_metric_value": baseline_value,
            "best_source_type": best.get("source_type") if best is not None else None,
            "best_variant": best.get("variant") if best is not None else None,
            "best_metric_value": best.get("primary_metric_value") if best is not None else None,
            "best_improvement_vs_baseline": best.get("primary_metric_improvement_vs_baseline")
            if best is not None
            else None,
            "evaluate_checkpoints": bool(config.evaluate_checkpoints),
            "allow_precomputed_evaluations": bool(config.allow_precomputed_evaluations),
            "precomputed_evaluation_contract": LOCAL_GRAPH_RESIDUAL_PRECOMPUTED_EVAL_CONTRACT,
            "hlt_degradation_strength": LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH,
            "rule": "Primary comparison is final-test FPR@50, lower is better.",
            "alpha_reporting_rule": (
                "Residual rows are split into learned-alpha and model-val gamma-shrunk rows; "
                "gamma shrinkage is applied to the learned correction delta=alpha*r."
            ),
        },
        "baseline_cache_family": baseline_cache_family,
        "metric_table": metric_rows,
        "residual_diagnostics": diagnostic_rows,
        "fusion_controls": fusion_control_rows,
        "outputs": output_paths,
    }
    _write_csv(Path(output_paths["metric_table_csv"]), metric_rows)
    _write_csv(Path(output_paths["diagnostics_csv"]), diagnostic_rows)
    _write_csv(Path(output_paths["baseline_comparison_csv"]), comparison_rows)
    _write_csv(Path(output_paths["fusion_controls_csv"]), fusion_control_rows)
    save_json(output_paths["report_json"], report)
    save_json(output_paths["run_report"], report)
    _write_markdown(Path(output_paths["report_markdown"]), report)
    return report


__all__ = [
    "LOCAL_GRAPH_RESIDUAL_BASELINE_VARIANT",
    "LOCAL_GRAPH_RESIDUAL_REPORT_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_REPORT_SPLITS",
    "LOCAL_GRAPH_RESIDUAL_REPORT_STEP",
    "LocalGraphResidualExpertReportConfig",
    "build_local_graph_residual_expert_report",
]

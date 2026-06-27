#!/usr/bin/env python3
"""Run frozen-score fusion for local-graph HLT ParT comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, set_training_seed  # noqa: E402
from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from teacher_logit_reco.local_graph_part.fusion import (  # noqa: E402
    LOCAL_GRAPH_SCORE_FUSION_CONTRACT,
    LOCAL_GRAPH_SCORE_FUSION_DEFAULT_C_GRID,
    LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT,
    LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT,
    LOCAL_GRAPH_SCORE_FUSION_STEP,
    FusionFeatureBlock,
    LocalGraphPredictionBlock,
    binary_metrics_from_signal_scores,
    build_disagreement_feature_block,
    build_score_feature_block,
    fit_binary_logistic_stackers_selecting_c,
    fusion_metric_score,
    load_prediction_block,
    save_prediction_block,
    select_weighted_average_on_stack,
    shuffle_non_baseline_columns,
    shuffled_labels,
    write_prediction_manifest,
)
from teacher_logit_reco.local_graph_part.protocol import (  # noqa: E402
    LOCAL_GRAPH_PART_DEFAULT_VARIANTS,
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
    local_graph_part_protocol_manifest,
)
from teacher_logit_reco.local_graph_part.train import (  # noqa: E402
    load_local_graph_tagger_checkpoint,
)
from teacher_logit_reco.set_matching.train import source_metadata  # noqa: E402
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader  # noqa: E402


DEFAULT_MODEL_SETS = (
    ("hlt_part_baseline", "local_edgeconv_adapter"),
    ("hlt_part_baseline", "local_point_attention_adapter"),
    ("hlt_part_baseline", "local_edgeconv_adapter", "local_point_attention_adapter"),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metric(metrics: Mapping[str, Any], key: str) -> Any:
    if key in metrics:
        return metrics.get(key)
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping):
        return binary.get(key)
    return None


def _row_from_metrics(
    *,
    method: str,
    model_set: Sequence[str],
    stack_metrics: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    primary_metric: str,
    negative_control: bool = False,
    control_type: str | None = None,
    selection: Mapping[str, Any] | None = None,
    weights: Sequence[float] | None = None,
    coefficients: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stack_score, stack_value = fusion_metric_score(stack_metrics, primary_metric)
    final_score, final_value = fusion_metric_score(final_metrics, primary_metric)
    return {
        "method": str(method),
        "model_set": " ".join(str(item) for item in model_set),
        "n_models": int(len(model_set)),
        "negative_control": bool(negative_control),
        "control_type": control_type,
        "primary_metric": str(primary_metric),
        "stack_val_primary_metric_value": float(stack_value),
        "stack_val_selection_score": float(stack_score),
        "final_test_primary_metric_value": float(final_value),
        "final_test_selection_score": float(final_score),
        "stack_val_accuracy": _metric(stack_metrics, "accuracy"),
        "stack_val_auc": _metric(stack_metrics, "auc"),
        "stack_val_fpr_at_signal_eff_0p30": _metric(stack_metrics, "fpr_at_signal_eff_0p30"),
        "stack_val_fpr_at_signal_eff_0p50": _metric(stack_metrics, "fpr_at_signal_eff_0p50"),
        "stack_val_background_rejection_at_signal_eff_0p50": _metric(
            stack_metrics,
            "background_rejection_at_signal_eff_0p50",
        ),
        "final_test_accuracy": _metric(final_metrics, "accuracy"),
        "final_test_auc": _metric(final_metrics, "auc"),
        "final_test_fpr_at_signal_eff_0p30": _metric(final_metrics, "fpr_at_signal_eff_0p30"),
        "final_test_fpr_at_signal_eff_0p50": _metric(final_metrics, "fpr_at_signal_eff_0p50"),
        "final_test_background_rejection_at_signal_eff_0p50": _metric(
            final_metrics,
            "background_rejection_at_signal_eff_0p50",
        ),
        "selection": dict(selection or {}),
        "weights": None if weights is None else [float(value) for value in weights],
        "coefficients": dict(coefficients or {}),
        "stack_metrics": stack_metrics,
        "final_metrics": final_metrics,
    }


def _select_best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    usable = [
        row
        for row in rows
        if row.get("final_test_selection_score") is not None
        and np.isfinite(float(row.get("final_test_selection_score")))
    ]
    if not usable:
        return None
    return max(usable, key=lambda row: float(row["final_test_selection_score"]))


def _safe_model_set_name(model_set: Sequence[str]) -> str:
    return "__".join(str(item).replace("local_", "").replace("_adapter", "") for item in model_set)


def _available_variants(tagger_root: Path, variants: Sequence[str], *, require_all: bool) -> tuple[list[str], list[str]]:
    available: list[str] = []
    missing: list[str] = []
    for variant in variants:
        checkpoint = tagger_root / str(variant) / "best_model_val.pt"
        if checkpoint.exists():
            available.append(str(variant))
        else:
            missing.append(str(variant))
    if missing and bool(require_all):
        raise FileNotFoundError(f"missing local graph checkpoints for variants: {missing}")
    return available, missing


def _load_dataset(hlt_cache_dir: Path, split: str, *, max_jets: int | None, verify_hash: bool) -> SubtokenHLTJetDataset:
    view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=bool(verify_hash))
    return SubtokenHLTJetDataset(
        view,
        label_filter=(0, 1),
        label_names=LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
        max_jets=max_jets,
    )


def _evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    variant: str,
    split: str,
    hlt_cache_dir: Path,
    max_jets: int | None,
    batch_size: int,
    num_workers: int,
    device,
    verify_hlt_hash: bool,
) -> LocalGraphPredictionBlock:
    torch = require_torch()
    model, payload = load_local_graph_tagger_checkpoint(checkpoint_path, device=device)
    dataset = _load_dataset(hlt_cache_dir, split, max_jets=max_jets, verify_hash=bool(verify_hlt_hash))
    loader = make_subtoken_hlt_loader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        seed=7717,
    )
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    indices_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = model(tokens, mask)
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
            indices_rows.append(batch["indices"].detach().cpu().numpy().astype(np.int64))
    logits_np = np.concatenate(logits_rows, axis=0) if logits_rows else np.zeros((0, 2), dtype=np.float32)
    labels_np = np.concatenate(labels_rows, axis=0) if labels_rows else np.zeros((0,), dtype=np.int64)
    indices_np = np.concatenate(indices_rows, axis=0) if indices_rows else np.zeros((0,), dtype=np.int64)
    return LocalGraphPredictionBlock(
        variant=str(variant),
        split=str(split),
        logits=logits_np,
        labels=labels_np,
        indices=indices_np,
        metadata={
            "checkpoint": str(checkpoint_path),
            "run_report": str(checkpoint_path.parent / "run_report.json"),
            "checkpoint_variant": payload.get("variant"),
            "checkpoint_epoch": payload.get("epoch"),
            "dataset": dict(dataset.metadata),
        },
    )


def _ensure_prediction_block(
    *,
    prediction_dir: Path,
    tagger_root: Path,
    hlt_cache_dir: Path,
    variant: str,
    split: str,
    max_jets: int | None,
    batch_size: int,
    num_workers: int,
    device,
    overwrite: bool,
    verify_hlt_hash: bool,
) -> LocalGraphPredictionBlock:
    npz_path, _meta_path = (prediction_dir / variant / f"{split}_predictions.npz", prediction_dir / variant / f"{split}_predictions_metadata.json")
    if npz_path.exists() and not bool(overwrite):
        return load_prediction_block(prediction_dir, variant, split)
    block = _evaluate_checkpoint(
        checkpoint_path=tagger_root / variant / "best_model_val.pt",
        variant=variant,
        split=split,
        hlt_cache_dir=hlt_cache_dir,
        max_jets=max_jets,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device=device,
        verify_hlt_hash=bool(verify_hlt_hash),
    )
    save_prediction_block(block, prediction_dir, overwrite=bool(overwrite))
    return block


def _model_sets(variants: Sequence[str], baseline_variant: str) -> list[tuple[str, ...]]:
    available = set(variants)
    sets: list[tuple[str, ...]] = []
    for candidate in DEFAULT_MODEL_SETS:
        if all(item in available for item in candidate):
            sets.append(tuple(candidate))
    if baseline_variant in available:
        all_set = tuple([baseline_variant, *[variant for variant in variants if variant != baseline_variant]])
        if len(all_set) > 1:
            sets.append(all_set)
    deduped: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for model_set in sets:
        if model_set not in seen:
            seen.add(model_set)
            deduped.append(model_set)
    return deduped


def _blocks(blocks_by_split: Mapping[str, Mapping[str, LocalGraphPredictionBlock]], split: str, variants: Sequence[str]) -> list[LocalGraphPredictionBlock]:
    return [blocks_by_split[split][variant] for variant in variants]


def _add_raw_model_rows(
    rows: list[dict[str, Any]],
    *,
    blocks_by_split: Mapping[str, Mapping[str, LocalGraphPredictionBlock]],
    variants: Sequence[str],
    primary_metric: str,
) -> None:
    for variant in variants:
        stack_block = blocks_by_split[LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT][variant]
        final_block = blocks_by_split[LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT][variant]
        rows.append(
            _row_from_metrics(
                method=f"raw_{variant}",
                model_set=[variant],
                stack_metrics=stack_block.metrics(),
                final_metrics=final_block.metrics(),
                primary_metric=primary_metric,
            )
        )


def _add_equal_average_rows(
    rows: list[dict[str, Any]],
    *,
    blocks_by_split: Mapping[str, Mapping[str, LocalGraphPredictionBlock]],
    model_sets: Sequence[Sequence[str]],
    primary_metric: str,
) -> None:
    for model_set in model_sets:
        for mode, probs in (("margin", False), ("prob_signal", True), ("log_odds", False), ("rank", True)):
            stack_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT, model_set)
            final_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT, model_set)
            stack_feature = build_score_feature_block(stack_blocks, mode=mode)
            final_feature = build_score_feature_block(
                final_blocks,
                mode=mode,
                rank_reference_blocks=stack_blocks if mode == "rank" else None,
            )
            weights = np.full((len(model_set),), 1.0 / float(len(model_set)), dtype=np.float64)
            stack_scores = stack_feature.features @ weights
            final_scores = final_feature.features @ weights
            rows.append(
                _row_from_metrics(
                    method=f"equal_average_{mode}",
                    model_set=model_set,
                    stack_metrics=binary_metrics_from_signal_scores(
                        stack_scores,
                        stack_feature.labels,
                        scores_are_probabilities=probs,
                    ),
                    final_metrics=binary_metrics_from_signal_scores(
                        final_scores,
                        final_feature.labels,
                        scores_are_probabilities=probs,
                    ),
                    primary_metric=primary_metric,
                    weights=weights,
                )
            )


def _add_weighted_grid_rows(
    rows: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
    *,
    blocks_by_split: Mapping[str, Mapping[str, LocalGraphPredictionBlock]],
    model_sets: Sequence[Sequence[str]],
    primary_metric: str,
    weight_grid_step: float,
) -> None:
    for model_set in model_sets:
        for mode, probs in (("margin", False), ("prob_signal", True), ("log_odds", False), ("rank", True)):
            stack_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT, model_set)
            final_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT, model_set)
            stack_feature = build_score_feature_block(stack_blocks, mode=mode)
            final_feature = build_score_feature_block(
                final_blocks,
                mode=mode,
                rank_reference_blocks=stack_blocks if mode == "rank" else None,
            )
            selection, candidates = select_weighted_average_on_stack(
                stack_feature,
                step=float(weight_grid_step),
                selection_metric=primary_metric,
                scores_are_probabilities=probs,
            )
            final_scores = selection.predict_score(final_feature.features)
            final_metrics = binary_metrics_from_signal_scores(
                final_scores,
                final_feature.labels,
                scores_are_probabilities=probs,
            )
            method = f"grid_weighted_{mode}"
            rows.append(
                _row_from_metrics(
                    method=method,
                    model_set=model_set,
                    stack_metrics=selection.stack_metrics,
                    final_metrics=final_metrics,
                    primary_metric=primary_metric,
                    selection={"n_candidates": len(candidates), **selection.to_dict()},
                    weights=selection.weights,
                )
            )
            for index, weight in enumerate(selection.weights):
                weight_rows.append(
                    {
                        "method": method,
                        "model_set": " ".join(model_set),
                        "variant": model_set[index],
                        "weight": float(weight),
                        "score_mode": mode,
                    }
                )


def _feature_block_for_mode(
    blocks: Sequence[LocalGraphPredictionBlock],
    *,
    mode: str,
    rank_reference_blocks: Sequence[LocalGraphPredictionBlock] | None = None,
) -> FusionFeatureBlock:
    if mode == "margins":
        return build_score_feature_block(blocks, mode="margin")
    if mode == "probabilities":
        return build_score_feature_block(blocks, mode="prob_signal")
    if mode == "log_odds":
        return build_score_feature_block(blocks, mode="log_odds")
    if mode == "ranks":
        return build_score_feature_block(blocks, mode="rank", rank_reference_blocks=rank_reference_blocks)
    if mode == "disagreement":
        return build_disagreement_feature_block(blocks)
    raise ValueError(f"unknown feature mode: {mode}")


def _add_logistic_rows(
    rows: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
    *,
    blocks_by_split: Mapping[str, Mapping[str, LocalGraphPredictionBlock]],
    model_sets: Sequence[Sequence[str]],
    primary_metric: str,
    c_grid: Sequence[float],
    max_iter: int,
    prefer_sklearn: bool,
) -> None:
    for model_set in model_sets:
        for mode in ("margins", "probabilities", "log_odds", "ranks", "disagreement"):
            stack_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT, model_set)
            final_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT, model_set)
            stack_feature = _feature_block_for_mode(stack_blocks, mode=mode)
            final_feature = _feature_block_for_mode(
                final_blocks,
                mode=mode,
                rank_reference_blocks=stack_blocks if mode == "ranks" else None,
            )
            stacker, selection = fit_binary_logistic_stackers_selecting_c(
                stack_feature,
                c_grid=c_grid,
                max_iter=int(max_iter),
                selection_metric=primary_metric,
                prefer_sklearn=bool(prefer_sklearn),
            )
            final_metrics = stacker.metrics(final_feature.features, final_feature.labels)
            method = f"logistic_{mode}"
            rows.append(
                _row_from_metrics(
                    method=method,
                    model_set=model_set,
                    stack_metrics=selection["selected_metrics"],
                    final_metrics=final_metrics,
                    primary_metric=primary_metric,
                    selection=selection,
                    coefficients=stacker.to_dict(),
                )
            )
            for name, coef in zip(stacker.feature_names, stacker.coef):
                weight_rows.append(
                    {
                        "method": method,
                        "model_set": " ".join(model_set),
                        "variant": _feature_variant_from_name(name),
                        "feature": name,
                        "coefficient": float(coef),
                        "selected_C": float(stacker.C),
                        "solver": stacker.solver,
                    }
                )


def _feature_variant_from_name(name: str) -> str:
    text = str(name)
    if "__" in text:
        parts = text.split("__")
        if parts[0] == "abs_margin_diff" and len(parts) >= 3:
            return f"{parts[1]} {parts[2]}"
        return parts[0].replace("row_shuffled__", "")
    return text


def _add_control_rows(
    rows: list[dict[str, Any]],
    *,
    blocks_by_split: Mapping[str, Mapping[str, LocalGraphPredictionBlock]],
    model_sets: Sequence[Sequence[str]],
    baseline_variant: str,
    primary_metric: str,
    c_grid: Sequence[float],
    max_iter: int,
    prefer_sklearn: bool,
    seed: int,
) -> None:
    if baseline_variant in blocks_by_split[LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT]:
        stack_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT, [baseline_variant])
        final_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT, [baseline_variant])
        stack_feature = build_score_feature_block(stack_blocks, mode="margin")
        final_feature = build_score_feature_block(final_blocks, mode="margin")
        stacker, selection = fit_binary_logistic_stackers_selecting_c(
            stack_feature,
            c_grid=c_grid,
            max_iter=int(max_iter),
            selection_metric=primary_metric,
            prefer_sklearn=bool(prefer_sklearn),
        )
        rows.append(
            _row_from_metrics(
                method="control_baseline_only_logistic_margin",
                model_set=[baseline_variant],
                stack_metrics=selection["selected_metrics"],
                final_metrics=stacker.metrics(final_feature.features, final_feature.labels),
                primary_metric=primary_metric,
                negative_control=False,
                control_type="baseline_only_calibration",
                selection=selection,
                coefficients=stacker.to_dict(),
            )
        )

    for model_set in model_sets:
        if baseline_variant not in model_set or len(model_set) < 2:
            continue
        stack_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT, model_set)
        final_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT, model_set)
        stack_feature = shuffle_non_baseline_columns(
            build_score_feature_block(stack_blocks, mode="margin"),
            baseline_variant=baseline_variant,
            seed=int(seed),
        )
        final_feature = shuffle_non_baseline_columns(
            build_score_feature_block(final_blocks, mode="margin"),
            baseline_variant=baseline_variant,
            seed=int(seed) + 1,
        )
        stacker, selection = fit_binary_logistic_stackers_selecting_c(
            stack_feature,
            c_grid=c_grid,
            max_iter=int(max_iter),
            selection_metric=primary_metric,
            prefer_sklearn=bool(prefer_sklearn),
        )
        rows.append(
            _row_from_metrics(
                method="control_row_shuffled_local_scores",
                model_set=model_set,
                stack_metrics=selection["selected_metrics"],
                final_metrics=stacker.metrics(final_feature.features, final_feature.labels),
                primary_metric=primary_metric,
                negative_control=True,
                control_type="row_shuffled_local_scores",
                selection=selection,
                coefficients=stacker.to_dict(),
            )
        )

    for model_set in model_sets:
        stack_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT, model_set)
        final_blocks = _blocks(blocks_by_split, LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT, model_set)
        stack_feature = build_score_feature_block(stack_blocks, mode="margin")
        shuffled_stack = FusionFeatureBlock(
            features=stack_feature.features,
            labels=shuffled_labels(stack_feature.labels, seed=int(seed)),
            feature_names=stack_feature.feature_names,
            variants=stack_feature.variants,
            split=stack_feature.split,
        )
        final_feature = build_score_feature_block(final_blocks, mode="margin")
        stacker, selection = fit_binary_logistic_stackers_selecting_c(
            shuffled_stack,
            c_grid=c_grid,
            max_iter=int(max_iter),
            selection_metric=primary_metric,
            prefer_sklearn=bool(prefer_sklearn),
        )
        rows.append(
            _row_from_metrics(
                method="control_label_shuffled_logistic_margin",
                model_set=model_set,
                stack_metrics=selection["selected_metrics"],
                final_metrics=stacker.metrics(final_feature.features, final_feature.labels),
                primary_metric=primary_metric,
                negative_control=True,
                control_type="label_shuffled_stacker",
                selection=selection,
                coefficients=stacker.to_dict(),
            )
        )


def _add_delta_columns(rows: list[dict[str, Any]], *, baseline_variant: str) -> None:
    raw_baseline = next((row for row in rows if row["method"] == f"raw_{baseline_variant}"), None)
    calibrated = next((row for row in rows if row["method"] == "control_baseline_only_logistic_margin"), None)
    raw_value = raw_baseline.get("final_test_primary_metric_value") if raw_baseline else None
    calibrated_value = calibrated.get("final_test_primary_metric_value") if calibrated else None
    for row in rows:
        value = row.get("final_test_primary_metric_value")
        if value is None:
            continue
        if raw_value is not None:
            row["delta_vs_raw_hlt_baseline"] = float(value) - float(raw_value)
        if calibrated_value is not None:
            row["delta_vs_calibrated_hlt_baseline"] = float(value) - float(calibrated_value)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report.get("summary", {})
    rows = report.get("fusion_metric_table", [])
    top = [row for row in rows if not row.get("negative_control")]
    top = sorted(top, key=lambda row: float(row.get("final_test_selection_score", float("-inf"))), reverse=True)[:12]
    lines = [
        "# Local Graph Score Fusion Report",
        "",
        f"- ok: `{report.get('ok')}`",
        f"- primary metric: `{summary.get('primary_metric')}`",
        f"- raw baseline FPR50: `{summary.get('raw_baseline_final_test_primary_metric')}`",
        f"- calibrated baseline FPR50: `{summary.get('calibrated_baseline_final_test_primary_metric')}`",
        f"- best valid method: `{summary.get('best_valid_method')}`",
        f"- best valid final metric: `{summary.get('best_valid_final_test_primary_metric')}`",
        "",
        "| method | models | final FPR50 | stack FPR50 | delta raw | delta calibrated |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            "| {method} | {models} | {final} | {stack} | {dr} | {dc} |".format(
                method=row.get("method"),
                models=row.get("model_set"),
                final=row.get("final_test_fpr_at_signal_eff_0p50"),
                stack=row.get("stack_val_fpr_at_signal_eff_0p50"),
                dr=row.get("delta_vs_raw_hlt_baseline"),
                dc=row.get("delta_vs_calibrated_hlt_baseline"),
            )
        )
    if report.get("problems"):
        lines.extend(["", "## Problems", ""])
        for problem in report["problems"]:
            lines.append(f"- {problem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_score_fusion(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    set_training_seed(int(args.seed))
    device = resolve_device(args.device)
    experiment_root = Path(args.experiment_root)
    tagger_root = Path(args.tagger_root or experiment_root / "taggers")
    hlt_cache_dir = Path(args.hlt_cache_dir or experiment_root / "binary_inputs" / "hlt_cache")
    output_dir = Path(args.output_dir or experiment_root / "score_fusion")
    prediction_dir = Path(args.prediction_dir or output_dir / "predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants, missing_variants = _available_variants(
        tagger_root,
        args.variants,
        require_all=bool(args.require_all_variants),
    )
    if args.baseline_variant not in variants:
        raise FileNotFoundError(f"baseline variant is not available: {args.baseline_variant}")
    blocks_by_split: dict[str, dict[str, LocalGraphPredictionBlock]] = {
        LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT: {},
        LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT: {},
    }
    all_blocks: list[LocalGraphPredictionBlock] = []
    for split, max_jets in (
        (LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT, args.max_stack_jets),
        (LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT, args.max_final_test_jets),
    ):
        if split == LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT and not bool(args.confirm_final_test):
            raise ValueError("Refusing to evaluate final_test fusion without --confirm-final-test")
        for variant in variants:
            block = _ensure_prediction_block(
                prediction_dir=prediction_dir,
                tagger_root=tagger_root,
                hlt_cache_dir=hlt_cache_dir,
                variant=variant,
                split=split,
                max_jets=max_jets,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                device=device,
                overwrite=bool(args.overwrite_predictions),
                verify_hlt_hash=not bool(args.skip_hlt_hash_check),
            )
            blocks_by_split[split][variant] = block
            all_blocks.append(block)

    primary_metric = str(args.primary_metric)
    model_sets = _model_sets(variants, args.baseline_variant)
    if not model_sets:
        raise ValueError("No fusion model sets could be built from the available variants")

    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    _add_raw_model_rows(rows, blocks_by_split=blocks_by_split, variants=variants, primary_metric=primary_metric)
    _add_equal_average_rows(rows, blocks_by_split=blocks_by_split, model_sets=model_sets, primary_metric=primary_metric)
    _add_weighted_grid_rows(
        rows,
        weight_rows,
        blocks_by_split=blocks_by_split,
        model_sets=model_sets,
        primary_metric=primary_metric,
        weight_grid_step=float(args.weight_grid_step),
    )
    _add_logistic_rows(
        rows,
        weight_rows,
        blocks_by_split=blocks_by_split,
        model_sets=model_sets,
        primary_metric=primary_metric,
        c_grid=tuple(float(value) for value in args.c_grid),
        max_iter=int(args.max_iter),
        prefer_sklearn=not bool(args.no_sklearn),
    )
    if bool(args.run_controls):
        _add_control_rows(
            rows,
            blocks_by_split=blocks_by_split,
            model_sets=model_sets,
            baseline_variant=args.baseline_variant,
            primary_metric=primary_metric,
            c_grid=tuple(float(value) for value in args.c_grid),
            max_iter=int(args.max_iter),
            prefer_sklearn=not bool(args.no_sklearn),
            seed=int(args.control_seed),
        )
    _add_delta_columns(rows, baseline_variant=args.baseline_variant)

    valid_rows = [row for row in rows if not row.get("negative_control")]
    control_rows = [row for row in rows if row.get("negative_control")]
    best_valid = _select_best_row(valid_rows)
    best_control = _select_best_row(control_rows)
    raw_baseline = next((row for row in rows if row["method"] == f"raw_{args.baseline_variant}"), None)
    calibrated_baseline = next((row for row in rows if row["method"] == "control_baseline_only_logistic_margin"), None)
    raw_metric = raw_baseline.get("final_test_primary_metric_value") if raw_baseline else None
    calibrated_metric = calibrated_baseline.get("final_test_primary_metric_value") if calibrated_baseline else None

    problems = []
    if best_valid is None:
        problems.append("no valid fusion rows were produced")
    if raw_baseline is None:
        problems.append("raw HLT baseline row was not produced")
    if calibrated_baseline is None and bool(args.run_controls):
        problems.append("calibrated HLT-only baseline row was not produced")

    output_paths = {
        "fusion_report_json": str(output_dir / "fusion_report.json"),
        "fusion_report_md": str(output_dir / "fusion_report.md"),
        "fusion_metric_table_csv": str(output_dir / "fusion_metric_table.csv"),
        "fusion_weights_csv": str(output_dir / "fusion_weights.csv"),
        "fusion_controls_csv": str(output_dir / "fusion_controls.csv"),
        "fusion_prediction_manifest_json": str(output_dir / "fusion_prediction_manifest.json"),
        "run_report_json": str(output_dir / "run_report.json"),
    }
    manifest = write_prediction_manifest(
        output_paths["fusion_prediction_manifest_json"],
        blocks=all_blocks,
        extra={
            "experiment_root": str(experiment_root),
            "tagger_root": str(tagger_root),
            "hlt_cache_dir": str(hlt_cache_dir),
            "available_variants": variants,
            "missing_variants": missing_variants,
            "stack_split": LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT,
            "final_test_split": LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT,
            "max_stack_jets": args.max_stack_jets,
            "max_final_test_jets": args.max_final_test_jets,
        },
    )
    report = {
        "step": LOCAL_GRAPH_SCORE_FUSION_STEP,
        "contract": LOCAL_GRAPH_SCORE_FUSION_CONTRACT,
        "ok": len(problems) == 0,
        "problems": problems,
        "source": source_metadata(),
        "protocol": local_graph_part_protocol_manifest(),
        "config": vars(args),
        "experiment_root": str(experiment_root),
        "tagger_root": str(tagger_root),
        "hlt_cache_dir": str(hlt_cache_dir),
        "prediction_manifest": manifest,
        "summary": {
            "primary_metric": primary_metric,
            "baseline_variant": args.baseline_variant,
            "available_variants": variants,
            "missing_variants": missing_variants,
            "model_sets": [" ".join(model_set) for model_set in model_sets],
            "raw_baseline_final_test_primary_metric": raw_metric,
            "calibrated_baseline_final_test_primary_metric": calibrated_metric,
            "best_valid_method": best_valid.get("method") if best_valid else None,
            "best_valid_model_set": best_valid.get("model_set") if best_valid else None,
            "best_valid_final_test_primary_metric": best_valid.get("final_test_primary_metric_value")
            if best_valid
            else None,
            "best_valid_delta_vs_raw_baseline": best_valid.get("delta_vs_raw_hlt_baseline") if best_valid else None,
            "best_valid_delta_vs_calibrated_baseline": best_valid.get("delta_vs_calibrated_hlt_baseline")
            if best_valid
            else None,
            "best_control_method": best_control.get("method") if best_control else None,
            "best_control_final_test_primary_metric": best_control.get("final_test_primary_metric_value")
            if best_control
            else None,
            "interpretation_rule": (
                "A fusion is interesting only if it improves final_test FPR@50 versus both raw and calibrated HLT baselines."
            ),
        },
        "fusion_metric_table": rows,
        "fusion_weights": weight_rows,
        "fusion_controls": control_rows,
        "outputs": output_paths,
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - start),
            "device": str(device),
        },
    }

    _write_csv(Path(output_paths["fusion_metric_table_csv"]), rows)
    _write_csv(Path(output_paths["fusion_weights_csv"]), weight_rows)
    _write_csv(Path(output_paths["fusion_controls_csv"]), control_rows)
    save_json(Path(output_paths["fusion_report_json"]), report)
    save_json(Path(output_paths["run_report_json"]), report)
    _write_markdown(Path(output_paths["fusion_report_md"]), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--tagger-root", default=None)
    parser.add_argument("--hlt-cache-dir", default=None)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variants", nargs="*", default=list(LOCAL_GRAPH_PART_DEFAULT_VARIANTS))
    parser.add_argument("--baseline-variant", default="hlt_part_baseline")
    parser.add_argument("--primary-metric", default=LOCAL_GRAPH_PART_PRIMARY_METRIC)
    parser.add_argument("--max-stack-jets", type=int, default=150000)
    parser.add_argument("--max-final-test-jets", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7717)
    parser.add_argument("--c-grid", nargs="*", type=float, default=list(LOCAL_GRAPH_SCORE_FUSION_DEFAULT_C_GRID))
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--weight-grid-step", type=float, default=0.05)
    parser.add_argument("--control-seed", type=int, default=17717)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--require-all-variants", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--no-sklearn", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    args.run_controls = not bool(args.skip_controls)
    return args


def main() -> int:
    report = run_score_fusion(parse_args())
    summary = report["summary"]
    print("local_graph_score_fusion_complete:")
    print(f"  ok: {report['ok']}")
    print(f"  output_dir: {Path(report['outputs']['fusion_report_json']).parent}")
    print(f"  primary_metric: {summary['primary_metric']}")
    print(f"  raw_baseline: {summary['raw_baseline_final_test_primary_metric']}")
    print(f"  calibrated_baseline: {summary['calibrated_baseline_final_test_primary_metric']}")
    print(f"  best_valid_method: {summary['best_valid_method']}")
    print(f"  best_valid_model_set: {summary['best_valid_model_set']}")
    print(f"  best_valid_final_metric: {summary['best_valid_final_test_primary_metric']}")
    print(f"  fusion_report: {report['outputs']['fusion_report_json']}")
    print(f"  metric_table: {report['outputs']['fusion_metric_table_csv']}")
    if report["problems"]:
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


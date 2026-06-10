"""Diversity and uncertainty analyses for frozen JetClass prediction blocks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .fusion import (
    DEFAULT_C_GRID,
    STACK_SPLITS,
    PredictionBlock,
    load_blocks_for_split,
    softmax_np,
    validate_prediction_alignment,
)
from .hlt_baseline import save_json
from .independent_fusion import (
    StandardizedLinearStacker,
    default_groups_for_models,
    discover_prediction_models,
    fit_stacker_selecting_c_on_val,
    metrics_from_logits,
    metrics_from_probs,
    save_stacker,
    split_hash_audit,
    validate_fusion_groups,
)
from .jetclass_data import LABEL_NAMES


UNCERTAINTY_FEATURE_MODES = (
    "uncertainty",
    "mean_outputs",
    "mean_uncertainty",
    "logits_uncertainty",
    "probs_uncertainty",
    "logits_probs_uncertainty",
)


@dataclass
class UncertaintyStackerConfig:
    prediction_dir: str
    output_dir: str
    model_names: List[str]
    groups: Dict[str, List[str]]
    feature_modes: List[str]
    c_grid: List[float]
    max_iter: int = 2000
    confirm_final_test: bool = False


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_ready(value), sort_keys=True)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else _json_ready(value)
                    for key, value in row.items()
                }
            )


def _class_names(num_classes: int) -> List[str]:
    names = list(LABEL_NAMES[:num_classes])
    if len(names) < num_classes:
        names.extend([f"class_{idx}" for idx in range(len(names), num_classes)])
    return names


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return None
    if float(np.std(a)) < 1.0e-12 or float(np.std(b)) < 1.0e-12:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1.0e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def _kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_safe = np.clip(np.asarray(p, dtype=np.float64), 1.0e-12, 1.0)
    q_safe = np.clip(np.asarray(q, dtype=np.float64), 1.0e-12, 1.0)
    return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=-1)


def _js_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (np.asarray(p, dtype=np.float64) + np.asarray(q, dtype=np.float64))
    return 0.5 * _kl_rows(p, midpoint) + 0.5 * _kl_rows(q, midpoint)


def _mean_pairwise_jsd(probs_stack: np.ndarray) -> np.ndarray:
    n_models = int(probs_stack.shape[0])
    if n_models < 2:
        return np.zeros((probs_stack.shape[1],), dtype=np.float64)
    rows = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            rows.append(_js_rows(probs_stack[i], probs_stack[j]))
    return np.mean(np.stack(rows, axis=0), axis=0)


def _mean_pairwise_prediction_disagreement(preds_stack: np.ndarray) -> np.ndarray:
    n_models = int(preds_stack.shape[0])
    if n_models < 2:
        return np.zeros((preds_stack.shape[1],), dtype=np.float64)
    rows = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            rows.append((preds_stack[i] != preds_stack[j]).astype(np.float64))
    return np.mean(np.stack(rows, axis=0), axis=0)


def _vote_fractions(preds_stack: np.ndarray, num_classes: int) -> np.ndarray:
    n_models, n_rows = preds_stack.shape
    votes = np.zeros((n_rows, num_classes), dtype=np.float64)
    row_index = np.arange(n_rows)
    for model_index in range(n_models):
        votes[row_index, preds_stack[model_index].astype(np.int64)] += 1.0
    return votes / float(max(n_models, 1))


def per_class_accuracy(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.argmax(probs, axis=1).astype(np.int64)
    num_classes = int(probs.shape[1])
    values: Dict[str, Any] = {}
    for class_index in range(num_classes):
        mask = labels == class_index
        values[str(class_index)] = {
            "class_name": _class_names(num_classes)[class_index],
            "n_jets": int(np.sum(mask)),
            "accuracy": float(np.mean(preds[mask] == labels[mask])) if np.any(mask) else None,
        }
    return values


def pairwise_diversity_metrics(block_a: PredictionBlock, block_b: PredictionBlock) -> Dict[str, Any]:
    validate_prediction_alignment([block_a, block_b])
    labels = block_a.labels.astype(np.int64)
    probs_a = softmax_np(block_a.logits)
    probs_b = softmax_np(block_b.logits)
    pred_a = np.argmax(probs_a, axis=1).astype(np.int64)
    pred_b = np.argmax(probs_b, axis=1).astype(np.int64)
    correct_a = pred_a == labels
    correct_b = pred_b == labels
    n_rows = int(len(labels))

    n11 = int(np.sum(correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    q_denom = float(n11 * n00 + n10 * n01)
    q_statistic = None if q_denom == 0.0 else float((n11 * n00 - n10 * n01) / q_denom)
    wrong_union = np.sum((~correct_a) | (~correct_b))
    error_jaccard = None if wrong_union == 0 else float(n00 / float(wrong_union))

    return {
        "split": block_a.split,
        "model_a": block_a.model_name,
        "model_b": block_b.model_name,
        "n_jets": n_rows,
        "accuracy_a": float(np.mean(correct_a)) if n_rows else 0.0,
        "accuracy_b": float(np.mean(correct_b)) if n_rows else 0.0,
        "prediction_agreement": float(np.mean(pred_a == pred_b)) if n_rows else 0.0,
        "prediction_disagreement": float(np.mean(pred_a != pred_b)) if n_rows else 0.0,
        "both_correct_rate": float(n11 / float(n_rows)) if n_rows else 0.0,
        "both_wrong_rate": float(n00 / float(n_rows)) if n_rows else 0.0,
        "a_only_correct_rate": float(n10 / float(n_rows)) if n_rows else 0.0,
        "b_only_correct_rate": float(n01 / float(n_rows)) if n_rows else 0.0,
        "error_overlap_jaccard": error_jaccard,
        "q_statistic": q_statistic,
        "correctness_correlation": _safe_corr(correct_a.astype(np.float64), correct_b.astype(np.float64)),
        "flattened_logit_correlation": _safe_corr(block_a.logits, block_b.logits),
        "flattened_prob_correlation": _safe_corr(probs_a, probs_b),
        "mean_row_jensen_shannon": float(np.mean(_js_rows(probs_a, probs_b))) if n_rows else 0.0,
    }


def group_oracle_summary(blocks: Sequence[PredictionBlock]) -> Dict[str, Any]:
    validate_prediction_alignment(blocks)
    labels = blocks[0].labels.astype(np.int64)
    logits_stack = np.stack([block.logits.astype(np.float64) for block in blocks], axis=0)
    probs_stack = np.stack([softmax_np(block.logits).astype(np.float64) for block in blocks], axis=0)
    preds_stack = np.argmax(probs_stack, axis=2).astype(np.int64)
    correct_stack = preds_stack == labels[None, :]
    mean_probs = np.mean(probs_stack, axis=0)
    mean_logits_probs = softmax_np(np.mean(logits_stack, axis=0))
    vote_probs = _vote_fractions(preds_stack, int(probs_stack.shape[2]))
    mean_pred = np.argmax(mean_probs, axis=1).astype(np.int64)
    unanimous = np.all(preds_stack == preds_stack[:1], axis=0)
    disagreement = ~unanimous
    raw_model_acc = [float(np.mean(correct_stack[index])) for index in range(len(blocks))]
    n_rows = int(len(labels))

    def subset_accuracy(mask: np.ndarray) -> float | None:
        if not np.any(mask):
            return None
        return float(np.mean(mean_pred[mask] == labels[mask]))

    return {
        "split": blocks[0].split,
        "model_names": [block.model_name for block in blocks],
        "n_jets": n_rows,
        "best_single_model_accuracy": float(max(raw_model_acc)) if raw_model_acc else None,
        "worst_single_model_accuracy": float(min(raw_model_acc)) if raw_model_acc else None,
        "mean_single_model_accuracy": float(np.mean(raw_model_acc)) if raw_model_acc else None,
        "oracle_any_model_correct_accuracy": float(np.mean(np.any(correct_stack, axis=0))) if n_rows else 0.0,
        "all_models_correct_rate": float(np.mean(np.all(correct_stack, axis=0))) if n_rows else 0.0,
        "all_models_wrong_rate": float(np.mean(~np.any(correct_stack, axis=0))) if n_rows else 0.0,
        "unanimous_prediction_rate": float(np.mean(unanimous)) if n_rows else 0.0,
        "disagreement_prediction_rate": float(np.mean(disagreement)) if n_rows else 0.0,
        "mean_probability_accuracy": metrics_from_probs(mean_probs, labels)["accuracy"],
        "mean_logit_accuracy": metrics_from_probs(mean_logits_probs, labels)["accuracy"],
        "majority_vote_accuracy": metrics_from_probs(vote_probs, labels)["accuracy"],
        "unanimous_subset_mean_probability_accuracy": subset_accuracy(unanimous),
        "disagreement_subset_mean_probability_accuracy": subset_accuracy(disagreement),
        "mean_pairwise_prediction_disagreement": float(np.mean(_mean_pairwise_prediction_disagreement(preds_stack))) if n_rows else 0.0,
        "mean_pairwise_jensen_shannon": float(np.mean(_mean_pairwise_jsd(probs_stack))) if n_rows else 0.0,
    }


def run_diversity_audit(
    *,
    prediction_dir: str | Path,
    output_dir: str | Path,
    model_names: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    confirm_final_test: bool,
) -> Dict[str, Any]:
    if not confirm_final_test:
        raise ValueError("Refusing to evaluate final_test without confirm_final_test=True")
    prediction_dir = Path(prediction_dir)
    output_dir = Path(output_dir)
    model_names = list(model_names)
    groups = {name: list(models) for name, models in groups.items()}
    validate_fusion_groups(groups, model_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    hash_audit = split_hash_audit(prediction_dir, model_names)
    if not hash_audit["ok"]:
        raise ValueError(f"Prediction split hash audit failed: {hash_audit['problems']}")

    raw_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []

    for split in STACK_SPLITS:
        all_blocks = load_blocks_for_split(prediction_dir, model_names, split)
        block_by_name = {block.model_name: block for block in all_blocks}
        num_classes = int(all_blocks[0].logits.shape[1])
        for block in all_blocks:
            metrics = metrics_from_logits(block.logits, block.labels)
            raw_rows.append(
                {
                    "split": split,
                    "model": block.model_name,
                    "accuracy": metrics["accuracy"],
                    "cross_entropy": metrics["cross_entropy"],
                    "macro_ovr_auc": metrics["macro_ovr_auc"],
                    "n_jets": metrics["n_jets"],
                }
            )
            for class_index, class_payload in per_class_accuracy(softmax_np(block.logits), block.labels).items():
                per_class_rows.append(
                    {
                        "split": split,
                        "model": block.model_name,
                        "class_index": int(class_index),
                        **class_payload,
                    }
                )
        for i, block_a in enumerate(all_blocks):
            for block_b in all_blocks[i + 1 :]:
                pairwise_rows.append(pairwise_diversity_metrics(block_a, block_b))
        for group_name, group_models in groups.items():
            group_blocks = [block_by_name[name] for name in group_models]
            oracle_rows.append({"group": group_name, **group_oracle_summary(group_blocks)})

    report = {
        "experiment": "frozen_prediction_diversity_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_dir": str(prediction_dir),
        "model_names": model_names,
        "groups": groups,
        "splits": list(STACK_SPLITS),
        "class_names": _class_names(num_classes),
        "stack_split_hash_audit": hash_audit,
        "raw_model_metrics": raw_rows,
        "per_class_model_metrics": per_class_rows,
        "pairwise_diversity": pairwise_rows,
        "group_oracle_summary": oracle_rows,
        "interpretation_hints": [
            "High oracle_any_model_correct_accuracy with high pairwise disagreement means a smarter selector may help.",
            "Low pairwise disagreement and high correctness correlation means the seven models are mostly redundant.",
            "A large disagreement_subset accuracy gap means ambiguity/disagreement features may be useful.",
        ],
        "output_files": {
            "raw_model_metrics_csv": str(output_dir / "raw_model_metrics.csv"),
            "per_class_model_metrics_csv": str(output_dir / "per_class_model_metrics.csv"),
            "pairwise_diversity_csv": str(output_dir / "pairwise_diversity.csv"),
            "group_oracle_summary_csv": str(output_dir / "group_oracle_summary.csv"),
        },
    }
    _write_csv(output_dir / "raw_model_metrics.csv", raw_rows)
    _write_csv(output_dir / "per_class_model_metrics.csv", per_class_rows)
    _write_csv(output_dir / "pairwise_diversity.csv", pairwise_rows)
    _write_csv(output_dir / "group_oracle_summary.csv", oracle_rows)
    save_json(output_dir / "diversity_report.json", _json_ready(report))
    return _json_ready(report)


def build_uncertainty_feature_matrix(
    blocks: Sequence[PredictionBlock],
    *,
    feature_mode: str,
) -> tuple[np.ndarray, List[str]]:
    validate_prediction_alignment(blocks)
    if feature_mode not in UNCERTAINTY_FEATURE_MODES:
        raise ValueError(
            f"Unknown uncertainty feature_mode {feature_mode!r}; "
            f"expected one of {list(UNCERTAINTY_FEATURE_MODES)}"
        )
    model_names = [block.model_name for block in blocks]
    logits_stack = np.stack([block.logits.astype(np.float64) for block in blocks], axis=0)
    probs_stack = np.stack([softmax_np(block.logits).astype(np.float64) for block in blocks], axis=0)
    preds_stack = np.argmax(probs_stack, axis=2).astype(np.int64)
    num_classes = int(logits_stack.shape[2])
    class_names = _class_names(num_classes)

    components: List[np.ndarray] = []
    columns: List[str] = []

    def add(matrix: np.ndarray, names: Sequence[str]) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.shape[1] != len(names):
            raise ValueError(f"Column-name mismatch: matrix has {matrix.shape[1]} columns, got {len(names)} names")
        components.append(matrix)
        columns.extend(names)

    def add_class_matrix(prefix: str, matrix: np.ndarray) -> None:
        add(matrix, [f"{prefix}:{name}" for name in class_names])

    def add_raw(kind: str, stack: np.ndarray) -> None:
        matrix = np.concatenate([stack[index] for index in range(len(model_names))], axis=1)
        names = [
            f"{model}:{kind}:{class_name}"
            for model in model_names
            for class_name in class_names
        ]
        add(matrix, names)

    logits_mean = np.mean(logits_stack, axis=0)
    probs_mean = np.mean(probs_stack, axis=0)

    if feature_mode in ("logits_uncertainty", "logits_probs_uncertainty"):
        add_raw("logit", logits_stack)
    if feature_mode in ("probs_uncertainty", "logits_probs_uncertainty"):
        add_raw("prob", probs_stack)
    if feature_mode in ("mean_outputs", "mean_uncertainty"):
        add_class_matrix("mean_logit", logits_mean)
        add_class_matrix("mean_prob", probs_mean)

    if feature_mode != "mean_outputs":
        probs_std = np.std(probs_stack, axis=0)
        logits_std = np.std(logits_stack, axis=0)
        add_class_matrix("std_prob", probs_std)
        add_class_matrix("range_prob", np.max(probs_stack, axis=0) - np.min(probs_stack, axis=0))
        add_class_matrix("std_logit", logits_std)
        add_class_matrix("range_logit", np.max(logits_stack, axis=0) - np.min(logits_stack, axis=0))
        add_class_matrix("vote_fraction", _vote_fractions(preds_stack, num_classes))

        entropy_by_model = _entropy(probs_stack).T
        max_prob_by_model = np.max(probs_stack, axis=2).T
        sorted_probs = np.sort(probs_stack, axis=2)
        margin_by_model = (sorted_probs[:, :, -1] - sorted_probs[:, :, -2]).T
        add(entropy_by_model, [f"{model}:entropy" for model in model_names])
        add(max_prob_by_model, [f"{model}:max_prob" for model in model_names])
        add(margin_by_model, [f"{model}:margin" for model in model_names])

        ensemble_entropy = _entropy(probs_mean)
        mean_model_entropy = np.mean(_entropy(probs_stack), axis=0)
        scalar_features = np.stack(
            [
                ensemble_entropy,
                mean_model_entropy,
                ensemble_entropy - mean_model_entropy,
                np.std(_entropy(probs_stack), axis=0),
                np.max(_vote_fractions(preds_stack, num_classes), axis=1),
                1.0 - np.max(_vote_fractions(preds_stack, num_classes), axis=1),
                _mean_pairwise_prediction_disagreement(preds_stack),
                _mean_pairwise_jsd(probs_stack),
                np.sum(np.var(probs_stack, axis=0), axis=1),
                np.sum(np.var(logits_stack, axis=0), axis=1),
            ],
            axis=1,
        )
        add(
            scalar_features,
            [
                "ensemble_entropy",
                "mean_model_entropy",
                "mutual_information",
                "std_model_entropy",
                "max_vote_fraction",
                "vote_disagreement",
                "mean_pairwise_prediction_disagreement",
                "mean_pairwise_jensen_shannon",
                "sum_prob_variance",
                "sum_logit_variance",
            ],
        )

    features = np.concatenate(components, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise FloatingPointError("Uncertainty feature matrix contains non-finite values")
    return features, columns


def _fit_uncertainty_group(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    feature_mode: str,
    c_grid: Sequence[float],
    max_iter: int,
) -> tuple[StandardizedLinearStacker, Dict[str, Any], Dict[str, Dict[str, Any]], List[str]]:
    blocks_by_split = {
        split: load_blocks_for_split(prediction_dir, model_names, split)
        for split in STACK_SPLITS
    }
    x_train, columns = build_uncertainty_feature_matrix(blocks_by_split["stack_train"], feature_mode=feature_mode)
    x_val, val_columns = build_uncertainty_feature_matrix(blocks_by_split["stack_val"], feature_mode=feature_mode)
    x_test, test_columns = build_uncertainty_feature_matrix(blocks_by_split["final_test"], feature_mode=feature_mode)
    if columns != val_columns or columns != test_columns:
        raise ValueError("Uncertainty feature columns changed across splits")
    y_train = blocks_by_split["stack_train"][0].labels
    y_val = blocks_by_split["stack_val"][0].labels
    y_test = blocks_by_split["final_test"][0].labels
    num_classes = int(blocks_by_split["stack_train"][0].logits.shape[1])
    stacker, selection = fit_stacker_selecting_c_on_val(
        x_train,
        y_train,
        x_val,
        y_val,
        c_grid=c_grid,
        max_iter=max_iter,
        feature_mode=feature_mode,
        model_names=model_names,
        num_classes=num_classes,
    )
    metrics = {
        "stack_train": metrics_from_probs(stacker.predict_probs(x_train), y_train),
        "stack_val": metrics_from_probs(stacker.predict_probs(x_val), y_val),
        "final_test": metrics_from_probs(stacker.predict_probs(x_test), y_test),
    }
    return stacker, selection, metrics, columns


def flatten_uncertainty_metrics(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group_name, group in report.items():
        for mode, mode_report in group["feature_modes"].items():
            for split, metrics in mode_report["metrics"].items():
                rows.append(
                    {
                        "group": group_name,
                        "feature_mode": mode,
                        "split": split,
                        "models": " ".join(group["model_names"]),
                        "selected_C": mode_report["selection"]["selected_C"],
                        "n_features": mode_report["n_features"],
                        "accuracy": metrics.get("accuracy"),
                        "cross_entropy": metrics.get("cross_entropy"),
                        "macro_ovr_auc": metrics.get("macro_ovr_auc"),
                        "n_jets": metrics.get("n_jets"),
                    }
                )
    return rows


def run_uncertainty_feature_stackers(config: UncertaintyStackerConfig) -> Dict[str, Any]:
    if not config.confirm_final_test:
        raise ValueError("Refusing to evaluate final_test without confirm_final_test=True")
    prediction_dir = Path(config.prediction_dir)
    output_dir = Path(config.output_dir)
    report_path = output_dir / "uncertainty_stacker_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite locked uncertainty stacker report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stacker_dir = output_dir / "stackers"

    model_names = list(config.model_names)
    groups = {name: list(models) for name, models in config.groups.items()}
    validate_fusion_groups(groups, model_names)
    hash_audit = split_hash_audit(prediction_dir, model_names)
    if not hash_audit["ok"]:
        raise ValueError(f"Prediction split hash audit failed: {hash_audit['problems']}")

    group_report: Dict[str, Any] = {}
    feature_columns: Dict[str, Any] = {}
    for group_name, group_models in groups.items():
        group_report[group_name] = {"model_names": list(group_models), "feature_modes": {}}
        feature_columns[group_name] = {}
        for mode in config.feature_modes:
            stacker, selection, metrics, columns = _fit_uncertainty_group(
                prediction_dir,
                group_models,
                feature_mode=mode,
                c_grid=config.c_grid,
                max_iter=config.max_iter,
            )
            save_stacker(stacker, stacker_dir / f"group__{group_name}__{mode}.npz")
            feature_columns[group_name][mode] = columns
            group_report[group_name]["feature_modes"][mode] = {
                "method": "multiclass_logistic_regression_on_uncertainty_features",
                "feature_mode": mode,
                "selection": selection,
                "metrics": metrics,
                "n_features": int(len(columns)),
                "feature_columns_path": str(output_dir / "feature_columns.json"),
                "weight_matrix_shape": list(stacker.coef.shape),
            }

    rows = flatten_uncertainty_metrics(group_report)
    report = {
        "experiment": "uncertainty_feature_stacker_from_frozen_prediction_blocks",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_dir": str(prediction_dir),
        "model_names": model_names,
        "groups": groups,
        "feature_modes": list(config.feature_modes),
        "leakage_rules": {
            "stacker_fit_split": "stack_train",
            "stacker_selection_split": "stack_val",
            "final_test_evaluated_after_selection": True,
            "features": "frozen model logits/probabilities plus within-row ensemble uncertainty summaries only",
        },
        "stack_split_hash_audit": hash_audit,
        "group_uncertainty_stacker_metrics": group_report,
        "output_files": {
            "uncertainty_stacker_metrics_csv": str(output_dir / "uncertainty_stacker_metrics.csv"),
            "feature_columns_json": str(output_dir / "feature_columns.json"),
            "stacker_dir": str(stacker_dir),
        },
    }
    _write_csv(output_dir / "uncertainty_stacker_metrics.csv", rows)
    save_json(output_dir / "feature_columns.json", _json_ready(feature_columns))
    save_json(report_path, _json_ready(report))
    return _json_ready(report)


def resolve_models_and_groups(
    prediction_dir: str | Path,
    *,
    model_names: Sequence[str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
) -> tuple[List[str], Dict[str, List[str]]]:
    resolved_models = list(model_names or discover_prediction_models(prediction_dir))
    resolved_groups = (
        {name: list(models) for name, models in groups.items()}
        if groups
        else default_groups_for_models(resolved_models)
    )
    if not resolved_groups:
        resolved_groups = {"all_models": list(resolved_models)}
    validate_fusion_groups(resolved_groups, resolved_models)
    return resolved_models, resolved_groups


def parse_group_specs(group_specs: Sequence[str] | None) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for spec in group_specs or []:
        if ":" not in spec:
            raise ValueError(f"Group spec must look like name:model_a,model_b, got {spec!r}")
        name, members = spec.split(":", 1)
        models = [member.strip() for member in members.split(",") if member.strip()]
        if not name.strip() or not models:
            raise ValueError(f"Invalid group spec {spec!r}")
        groups[name.strip()] = models
    return groups


def default_c_grid() -> List[float]:
    return [float(value) for value in DEFAULT_C_GRID]

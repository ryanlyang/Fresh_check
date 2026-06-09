"""Independent stacked-fusion evaluation from frozen prediction blocks.

This module consumes prediction blocks produced by
``scripts/demo_load_and_score_models_no_fusion.py``.  It deliberately starts
after model loading/inference: the only stacker inputs are frozen logits and/or
probabilities loaded from saved prediction blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .fusion import (
    DEFAULT_C_GRID,
    STACK_SPLITS,
    PredictionBlock,
    load_blocks_for_split,
    softmax_np,
    stack_feature_matrix,
    validate_prediction_alignment,
)
from .hlt_baseline import save_json
from .jetclass_data import LABEL_NAMES


FEATURE_MODES = ("logits", "probs", "logits_probs")
DEPLOYABLE_FORBIDDEN_MODELS = {"offline_teacher"}


@dataclass
class IndependentFusionConfig:
    prediction_dir: str
    output_dir: str
    model_names: List[str]
    groups: Dict[str, List[str]]
    feature_modes: List[str]
    c_grid: List[float]
    max_iter: int = 2000
    confirm_final_test: bool = False
    run_controls: bool = True
    control_seed: int = 12345
    singleton_models: List[str] | None = None


@dataclass
class StandardizedLinearStacker:
    coef: np.ndarray
    intercept: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    C: float
    solver: str
    feature_mode: str
    model_names: List[str]

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        x = (features.astype(np.float64) - self.mean) / self.scale
        return x @ self.coef.T + self.intercept

    def predict_probs(self, features: np.ndarray) -> np.ndarray:
        return softmax_np(self.predict_logits(features))


def discover_prediction_models(prediction_dir: str | Path, *, required_splits: Sequence[str] = STACK_SPLITS) -> List[str]:
    root = Path(prediction_dir)
    if not root.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {root}")
    models = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if all((child / f"{split}_predictions.npz").exists() for split in required_splits):
            models.append(child.name)
    if not models:
        raise FileNotFoundError(f"No complete prediction blocks found under {root}")
    return models


def default_groups_for_models(model_names: Sequence[str]) -> Dict[str, List[str]]:
    names = list(model_names)
    m2_models = [name for name in names if name.startswith("m2_")]
    groups: Dict[str, List[str]] = {}
    if m2_models:
        groups["m2_only"] = m2_models
    if "hlt_baseline" in names and m2_models:
        groups["hlt_plus_m2"] = ["hlt_baseline", *m2_models]
    if "hlt_baseline" in names:
        groups["hlt_only"] = ["hlt_baseline"]
    return groups


def validate_fusion_groups(groups: Mapping[str, Sequence[str]], available_models: Sequence[str]) -> None:
    available = set(available_models)
    for group_name, model_names in groups.items():
        missing = sorted(set(model_names) - available)
        if missing:
            raise ValueError(f"Fusion group {group_name!r} references missing models: {missing}")
        forbidden = sorted(set(model_names) & DEPLOYABLE_FORBIDDEN_MODELS)
        if forbidden:
            raise ValueError(
                f"Fusion group {group_name!r} includes forbidden offline-reference model(s): {forbidden}"
            )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ensure_splits(confirm_final_test: bool) -> None:
    if not confirm_final_test:
        raise ValueError("Refusing to evaluate final_test without confirm_final_test=True")


def labels_hash(labels: np.ndarray) -> str:
    import hashlib

    arr = np.asarray(labels, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def split_hash_audit(prediction_dir: str | Path, model_names: Sequence[str]) -> Dict[str, Any]:
    split_rows: Dict[str, Any] = {}
    identity_sets: Dict[str, set[str]] = {}
    ok = True
    problems: List[str] = []
    for split in STACK_SPLITS:
        blocks = load_blocks_for_split(prediction_dir, model_names, split)
        validate_prediction_alignment(blocks)
        first = blocks[0]
        identities = {identity.key() for identity in first.jet_ids}
        identity_sets[split] = identities
        per_model_hashes = {
            block.model_name: block.metadata.get("jet_identity_hash")
            for block in blocks
        }
        if len(set(per_model_hashes.values())) != 1:
            ok = False
            problems.append(f"{split}: model prediction blocks disagree on jet_identity_hash")
        split_rows[split] = {
            "n_jets": int(len(first.labels)),
            "jet_identity_hash": first.metadata.get("jet_identity_hash"),
            "label_hash": labels_hash(first.labels),
            "per_model_jet_identity_hash": per_model_hashes,
        }

    overlaps = {}
    for i, split_a in enumerate(STACK_SPLITS):
        for split_b in STACK_SPLITS[i + 1 :]:
            count = len(identity_sets[split_a] & identity_sets[split_b])
            overlaps[f"{split_a}__{split_b}"] = int(count)
            if count:
                ok = False
                problems.append(f"{split_a} and {split_b} overlap in {count} jet identities")
    return {
        "ok": bool(ok),
        "problems": problems,
        "splits": split_rows,
        "cross_split_overlap_counts": overlaps,
        "leakage_rule": "stack_train, stack_val, and final_test identities must be disjoint.",
    }


def confusion_matrix_np(probs: np.ndarray, labels: np.ndarray, *, num_classes: int) -> np.ndarray:
    preds = np.argmax(probs, axis=1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(labels.astype(np.int64), preds.astype(np.int64)):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            matrix[int(truth), int(pred)] += 1
    return matrix


def macro_ovr_auc(probs: np.ndarray, labels: np.ndarray, *, num_classes: int) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    y_true = np.eye(num_classes, dtype=np.int64)[labels.astype(np.int64)]
    try:
        return float(roc_auc_score(y_true, probs, average="macro", multi_class="ovr"))
    except Exception:
        return None


def fpr_at_signal_efficiency(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    signal_efficiency: float = 0.50,
) -> Dict[str, Any]:
    values: Dict[str, float | None] = {}
    for class_index in range(num_classes):
        signal = labels == class_index
        background = ~signal
        if not np.any(signal) or not np.any(background):
            values[str(class_index)] = None
            continue
        scores = probs[:, class_index]
        threshold = float(np.quantile(scores[signal], 1.0 - float(signal_efficiency)))
        values[str(class_index)] = float(np.mean(scores[background] >= threshold))
    finite = [value for value in values.values() if value is not None]
    return {
        "signal_efficiency": float(signal_efficiency),
        "per_class": values,
        "macro": float(np.mean(finite)) if finite else None,
    }


def metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape {probs.shape}")
    num_classes = int(probs.shape[1])
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Labels are outside the probability dimension")
    preds = np.argmax(probs, axis=1)
    picked = np.clip(probs[np.arange(len(labels)), labels], 1.0e-12, 1.0)
    return {
        "accuracy": float(np.mean(preds == labels)) if len(labels) else 0.0,
        "cross_entropy": float(-np.mean(np.log(picked))) if len(labels) else float("nan"),
        "n_jets": int(len(labels)),
        "macro_ovr_auc": macro_ovr_auc(probs, labels, num_classes=num_classes),
        "fpr_at_50pct_signal_efficiency": fpr_at_signal_efficiency(probs, labels, num_classes=num_classes),
        "confusion_matrix": confusion_matrix_np(probs, labels, num_classes=num_classes).tolist(),
    }


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    return metrics_from_probs(softmax_np(logits), labels)


def _standardize_train(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    return (features - mean) / scale, mean, scale


def _fit_numpy_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float,
    max_iter: int,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_features = x_train.shape
    y_onehot = np.eye(num_classes, dtype=np.float64)[y_train.astype(np.int64)]
    coef = np.zeros((num_classes, n_features), dtype=np.float64)
    intercept = np.zeros((num_classes,), dtype=np.float64)
    lr = 0.25
    l2 = 1.0 / max(float(C), 1.0e-9)
    for _ in range(int(max_iter)):
        logits = x_train @ coef.T + intercept
        probs = softmax_np(logits).astype(np.float64)
        error = (probs - y_onehot) / float(max(n_rows, 1))
        coef -= lr * (error.T @ x_train + l2 * coef / float(max(n_rows, 1)))
        intercept -= lr * error.sum(axis=0)
    return coef, intercept


def _fit_one_stacker(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float,
    max_iter: int,
    num_classes: int,
    feature_mode: str,
    model_names: Sequence[str],
) -> StandardizedLinearStacker:
    x_train, mean, scale = _standardize_train(x_train_raw)
    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=int(max_iter),
            multi_class="multinomial",
            n_jobs=1,
        )
        clf.fit(x_train, y_train)
        if clf.coef_.shape[0] != num_classes or not np.array_equal(
            clf.classes_.astype(np.int64),
            np.arange(num_classes, dtype=np.int64),
        ):
            raise ValueError("sklearn fitted a reduced class set; falling back to fixed-dimension numpy solver")
        coef = clf.coef_.astype(np.float64)
        intercept = clf.intercept_.astype(np.float64)
        solver = "sklearn_lbfgs"
    except Exception:
        coef, intercept = _fit_numpy_logistic(
            x_train,
            y_train,
            C=float(C),
            max_iter=max_iter,
            num_classes=num_classes,
        )
        solver = "numpy_gd"
    return StandardizedLinearStacker(
        coef=coef,
        intercept=intercept,
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        C=float(C),
        solver=solver,
        feature_mode=feature_mode,
        model_names=list(model_names),
    )


def fit_stacker_selecting_c_on_val(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    c_grid: Sequence[float],
    max_iter: int,
    feature_mode: str,
    model_names: Sequence[str],
    num_classes: int | None = None,
) -> tuple[StandardizedLinearStacker, Dict[str, Any]]:
    if num_classes is None:
        num_classes = int(max(np.max(y_train), np.max(y_val)) + 1)
    num_classes = int(num_classes)
    candidates = []
    for c_value in c_grid:
        stacker = _fit_one_stacker(
            x_train,
            y_train,
            C=float(c_value),
            max_iter=max_iter,
            num_classes=num_classes,
            feature_mode=feature_mode,
            model_names=model_names,
        )
        metrics = metrics_from_probs(stacker.predict_probs(x_val), y_val)
        candidates.append(
            {
                "C": float(c_value),
                "solver": stacker.solver,
                "metrics": metrics,
                "stacker": stacker,
            }
        )
    best = max(candidates, key=lambda row: (row["metrics"]["accuracy"], -row["metrics"]["cross_entropy"]))
    return best["stacker"], {
        "selection_split": "stack_val",
        "selected_C": float(best["C"]),
        "selected_solver": str(best["solver"]),
        "selected_stack_val_metrics": best["metrics"],
        "candidates": [
            {"C": row["C"], "solver": row["solver"], "metrics": row["metrics"]}
            for row in candidates
        ],
    }


def save_stacker(stacker: StandardizedLinearStacker, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coef=stacker.coef,
        intercept=stacker.intercept,
        mean=stacker.mean,
        scale=stacker.scale,
        C=np.asarray([stacker.C], dtype=np.float64),
        solver=np.asarray([stacker.solver]),
        feature_mode=np.asarray([stacker.feature_mode]),
        model_names=np.asarray(stacker.model_names),
    )


def block_features(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    split: str,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, List[PredictionBlock]]:
    blocks = load_blocks_for_split(prediction_dir, model_names, split)
    validate_prediction_alignment(blocks)
    return stack_feature_matrix(blocks, feature_mode=feature_mode), blocks[0].labels, blocks


def fit_group_stacker(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    feature_mode: str,
    c_grid: Sequence[float],
    max_iter: int,
) -> tuple[StandardizedLinearStacker, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    x_train, y_train, train_blocks = block_features(prediction_dir, model_names, split="stack_train", feature_mode=feature_mode)
    x_val, y_val, _ = block_features(prediction_dir, model_names, split="stack_val", feature_mode=feature_mode)
    x_test, y_test, _ = block_features(prediction_dir, model_names, split="final_test", feature_mode=feature_mode)
    num_classes = int(train_blocks[0].logits.shape[1])
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
    return stacker, selection, metrics


def raw_source_metrics(prediction_dir: str | Path, model_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    for model_name in model_names:
        report[model_name] = {}
        for split in STACK_SPLITS:
            block = load_blocks_for_split(prediction_dir, [model_name], split)[0]
            report[model_name][split] = metrics_from_logits(block.logits, block.labels)
    return report


def fit_temperature_on_val(logits_val: np.ndarray, labels_val: np.ndarray, temperatures: Sequence[float]) -> Dict[str, Any]:
    candidates = []
    for temp in temperatures:
        probs = softmax_np(np.asarray(logits_val, dtype=np.float64) / float(temp))
        metrics = metrics_from_probs(probs, labels_val)
        candidates.append({"temperature": float(temp), "metrics": metrics})
    best = min(candidates, key=lambda row: row["metrics"]["cross_entropy"])
    return {"selected_temperature": float(best["temperature"]), "candidates": candidates}


def temperature_metrics_for_model(prediction_dir: str | Path, model_name: str) -> Dict[str, Any]:
    temps = [0.50, 0.67, 0.80, 0.90, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
    val_block = load_blocks_for_split(prediction_dir, [model_name], "stack_val")[0]
    selection = fit_temperature_on_val(val_block.logits, val_block.labels, temps)
    selected = float(selection["selected_temperature"])
    metrics = {}
    for split in STACK_SPLITS:
        block = load_blocks_for_split(prediction_dir, [model_name], split)[0]
        metrics[split] = metrics_from_probs(softmax_np(block.logits / selected), block.labels)
    return {
        "method": "temperature_scaling_only",
        "selection_split": "stack_val",
        "selection": selection,
        "metrics": metrics,
    }


def singleton_stacker_metrics(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    feature_modes: Sequence[str],
    c_grid: Sequence[float],
    max_iter: int,
    stacker_dir: str | Path | None = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for model_name in model_names:
        if model_name in DEPLOYABLE_FORBIDDEN_MODELS:
            continue
        report[model_name] = {"temperature_scaling": temperature_metrics_for_model(prediction_dir, model_name)}
        for mode in feature_modes:
            stacker, selection, metrics = fit_group_stacker(
                prediction_dir,
                [model_name],
                feature_mode=mode,
                c_grid=c_grid,
                max_iter=max_iter,
            )
            if stacker_dir is not None:
                save_stacker(stacker, Path(stacker_dir) / f"singleton__{model_name}__{mode}.npz")
            report[model_name][mode] = {
                "method": "full_multiclass_logistic_regression",
                "model_names": [model_name],
                "feature_mode": mode,
                "selection": selection,
                "metrics": metrics,
                "weight_matrix_shape": list(stacker.coef.shape),
            }
    return report


def group_fusion_metrics(
    prediction_dir: str | Path,
    groups: Mapping[str, Sequence[str]],
    *,
    feature_modes: Sequence[str],
    c_grid: Sequence[float],
    max_iter: int,
    stacker_dir: str | Path,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for group_name, model_names in groups.items():
        report[group_name] = {"model_names": list(model_names), "feature_modes": {}}
        for mode in feature_modes:
            stacker, selection, metrics = fit_group_stacker(
                prediction_dir,
                model_names,
                feature_mode=mode,
                c_grid=c_grid,
                max_iter=max_iter,
            )
            save_stacker(stacker, Path(stacker_dir) / f"group__{group_name}__{mode}.npz")
            report[group_name]["feature_modes"][mode] = {
                "method": "full_multiclass_logistic_regression",
                "feature_mode": mode,
                "selection": selection,
                "metrics": metrics,
                "weight_matrix_shape": list(stacker.coef.shape),
            }
    return report


def _column_shuffle(features: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    shuffled = np.asarray(features).copy()
    for column in range(shuffled.shape[1]):
        shuffled[:, column] = shuffled[rng.permutation(shuffled.shape[0]), column]
    return shuffled


def negative_controls(
    prediction_dir: str | Path,
    groups: Mapping[str, Sequence[str]],
    *,
    feature_modes: Sequence[str],
    c_grid: Sequence[float],
    max_iter: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.RandomState(int(seed))
    report: Dict[str, Any] = {}
    for group_name, model_names in groups.items():
        report[group_name] = {}
        for mode in feature_modes:
            x_train, y_train, train_blocks = block_features(prediction_dir, model_names, split="stack_train", feature_mode=mode)
            x_val, y_val, _ = block_features(prediction_dir, model_names, split="stack_val", feature_mode=mode)
            x_test, y_test, _ = block_features(prediction_dir, model_names, split="final_test", feature_mode=mode)
            num_classes = int(train_blocks[0].logits.shape[1])

            shuffled_y = y_train.copy()
            rng.shuffle(shuffled_y)
            perm_stacker, perm_selection = fit_stacker_selecting_c_on_val(
                x_train,
                shuffled_y,
                x_val,
                y_val,
                c_grid=c_grid,
                max_iter=max_iter,
                feature_mode=mode,
                model_names=model_names,
                num_classes=num_classes,
            )
            perm_metrics = {
                "stack_val": metrics_from_probs(perm_stacker.predict_probs(x_val), y_val),
                "final_test": metrics_from_probs(perm_stacker.predict_probs(x_test), y_test),
            }

            shuffled_x_train = _column_shuffle(x_train, rng)
            row_stacker, row_selection = fit_stacker_selecting_c_on_val(
                shuffled_x_train,
                y_train,
                x_val,
                y_val,
                c_grid=c_grid,
                max_iter=max_iter,
                feature_mode=mode,
                model_names=model_names,
                num_classes=num_classes,
            )
            row_metrics = {
                "stack_val": metrics_from_probs(row_stacker.predict_probs(x_val), y_val),
                "final_test": metrics_from_probs(row_stacker.predict_probs(x_test), y_test),
            }

            report[group_name][mode] = {
                "permuted_labels": {
                    "selection": perm_selection,
                    "metrics": perm_metrics,
                    "note": "Only stack_train labels are permuted. stack_val/final_test labels remain locked.",
                },
                "row_shuffled_features": {
                    "selection": row_selection,
                    "metrics": row_metrics,
                    "note": "Each stack_train feature column is shuffled independently before fitting.",
                },
            }
    return report


def flatten_raw_metrics(raw: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for model_name, by_split in raw.items():
        for split, metrics in by_split.items():
            rows.append({"model": model_name, "split": split, **_flat_metric_row(metrics)})
    return rows


def _flat_metric_row(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy": metrics.get("accuracy"),
        "cross_entropy": metrics.get("cross_entropy"),
        "n_jets": metrics.get("n_jets"),
        "macro_ovr_auc": metrics.get("macro_ovr_auc"),
        "fpr50_macro": (metrics.get("fpr_at_50pct_signal_efficiency") or {}).get("macro"),
    }


def flatten_group_metrics(group_report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for group_name, group in group_report.items():
        for mode, mode_report in group["feature_modes"].items():
            for split, metrics in mode_report["metrics"].items():
                rows.append(
                    {
                        "group": group_name,
                        "feature_mode": mode,
                        "split": split,
                        "models": " ".join(group["model_names"]),
                        "selected_C": mode_report["selection"]["selected_C"],
                        **_flat_metric_row(metrics),
                    }
                )
    return rows


def flatten_singleton_metrics(singleton_report: Mapping[str, Any], raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for model_name, model_report in singleton_report.items():
        raw_final = raw.get(model_name, {}).get("final_test", {})
        for mode, mode_report in model_report.items():
            if mode == "temperature_scaling":
                metrics_by_split = mode_report["metrics"]
                selected_c = ""
                feature_mode = "temperature_only"
            else:
                metrics_by_split = mode_report["metrics"]
                selected_c = mode_report["selection"]["selected_C"]
                feature_mode = mode
            for split, metrics in metrics_by_split.items():
                rows.append(
                    {
                        "model": model_name,
                        "method": "temperature_scaling_only" if mode == "temperature_scaling" else "full_logistic",
                        "feature_mode": feature_mode,
                        "split": split,
                        "selected_C": selected_c,
                        "raw_final_test_accuracy": raw_final.get("accuracy"),
                        **_flat_metric_row(metrics),
                    }
                )
    return rows


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
            writer.writerow({key: _json_dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def suspicious_flags(
    raw: Mapping[str, Any],
    singleton: Mapping[str, Any],
    controls: Mapping[str, Any],
    *,
    chance_accuracy: float,
) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    hlt_raw = raw.get("hlt_baseline", {}).get("final_test", {}).get("accuracy")
    hlt_singleton = (
        singleton.get("hlt_baseline", {})
        .get("logits_probs", {})
        .get("metrics", {})
        .get("final_test", {})
        .get("accuracy")
    )
    if hlt_raw is not None and hlt_singleton is not None:
        delta = float(hlt_singleton) - float(hlt_raw)
        if delta > 0.05:
            flags.append(
                {
                    "name": "large_hlt_singleton_stacker_gain",
                    "severity": "inspect",
                    "raw_hlt_final_test_accuracy": hlt_raw,
                    "singleton_hlt_final_test_accuracy": hlt_singleton,
                    "delta": delta,
                }
            )
    for group_name, by_mode in controls.items():
        for mode, control_report in by_mode.items():
            for control_name in ("permuted_labels", "row_shuffled_features"):
                acc = (
                    control_report.get(control_name, {})
                    .get("metrics", {})
                    .get("final_test", {})
                    .get("accuracy")
                )
                if acc is not None and float(acc) > max(0.20, chance_accuracy + 0.10):
                    flags.append(
                        {
                            "name": f"{control_name}_did_not_collapse",
                            "severity": "warning",
                            "group": group_name,
                            "feature_mode": mode,
                            "final_test_accuracy": acc,
                        }
                    )
    return flags


def run_independent_fusion(config: IndependentFusionConfig) -> Dict[str, Any]:
    _ensure_splits(config.confirm_final_test)
    prediction_dir = Path(config.prediction_dir)
    output_dir = Path(config.output_dir)
    report_path = output_dir / "fusion_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite locked fusion report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stacker_dir = output_dir / "stackers"

    model_names = list(config.model_names)
    groups = {name: list(models) for name, models in config.groups.items()}
    feature_modes = list(config.feature_modes)
    validate_fusion_groups(groups, model_names)

    hash_audit = split_hash_audit(prediction_dir, model_names)
    if not hash_audit["ok"]:
        raise ValueError(f"Prediction split hash audit failed: {hash_audit['problems']}")

    raw = raw_source_metrics(prediction_dir, model_names)
    group_metrics = group_fusion_metrics(
        prediction_dir,
        groups,
        feature_modes=feature_modes,
        c_grid=config.c_grid,
        max_iter=config.max_iter,
        stacker_dir=stacker_dir,
    )
    singleton_models = list(config.singleton_models or model_names)
    singleton = singleton_stacker_metrics(
        prediction_dir,
        singleton_models,
        feature_modes=feature_modes,
        c_grid=config.c_grid,
        max_iter=config.max_iter,
        stacker_dir=stacker_dir,
    )
    controls = (
        negative_controls(
            prediction_dir,
            groups,
            feature_modes=feature_modes,
            c_grid=config.c_grid,
            max_iter=config.max_iter,
            seed=config.control_seed,
        )
        if config.run_controls
        else {}
    )
    first_confusion = next(iter(next(iter(raw.values())).values()))["confusion_matrix"]
    num_classes = len(first_confusion)
    chance_accuracy = 1.0 / float(num_classes)
    flags = suspicious_flags(raw, singleton, controls, chance_accuracy=chance_accuracy)

    write_csv(output_dir / "raw_source_metrics.csv", flatten_raw_metrics(raw))
    write_csv(output_dir / "group_fusion_metrics.csv", flatten_group_metrics(group_metrics))
    write_csv(output_dir / "singleton_stacker_metrics.csv", flatten_singleton_metrics(singleton, raw))
    save_json(output_dir / "controls.json", controls)
    save_json(output_dir / "stack_split_hash_audit.json", hash_audit)
    save_json(
        output_dir / "confusion_matrices.json",
        {
            "raw": {
                model: {
                    split: metrics["confusion_matrix"]
                    for split, metrics in by_split.items()
                }
                for model, by_split in raw.items()
            },
            "hlt_singleton_logits_probs": (
                singleton.get("hlt_baseline", {})
                .get("logits_probs", {})
                .get("metrics", {})
                .get("final_test", {})
                .get("confusion_matrix")
            ),
        },
    )

    report = {
        "experiment": "independent_fusion_from_frozen_prediction_blocks",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_dir": str(prediction_dir),
        "model_names": model_names,
        "groups": groups,
        "feature_modes": feature_modes,
        "leakage_rules": {
            "stacker_fit_split": "stack_train",
            "stacker_selection_split": "stack_val",
            "final_test_evaluated_after_selection": True,
            "offline_teacher_allowed_in_fusion_groups": False,
            "stacker_features": "logits/probabilities only",
        },
        "class_names": list(LABEL_NAMES[:num_classes]),
        "class_to_idx": {name: idx for idx, name in enumerate(LABEL_NAMES[:num_classes])},
        "config": asdict(config),
        "stack_split_hash_audit": hash_audit,
        "raw_source_metrics": raw,
        "singleton_stacker_metrics": singleton,
        "group_fusion_metrics": group_metrics,
        "controls": controls,
        "suspicious_flags": flags,
        "output_files": {
            "raw_source_metrics_csv": str(output_dir / "raw_source_metrics.csv"),
            "singleton_stacker_metrics_csv": str(output_dir / "singleton_stacker_metrics.csv"),
            "group_fusion_metrics_csv": str(output_dir / "group_fusion_metrics.csv"),
            "controls_json": str(output_dir / "controls.json"),
            "stack_split_hash_audit_json": str(output_dir / "stack_split_hash_audit.json"),
            "confusion_matrices_json": str(output_dir / "confusion_matrices.json"),
            "stacker_dir": str(stacker_dir),
        },
    }
    save_json(report_path, report)
    return report

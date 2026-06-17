"""Conditional evidence fusers anchored on the direct HLT ensemble.

These fusers consume frozen prediction blocks only.  The intended use is to
test whether reco-domain adapted taggers carry class-specific evidence that is
useful beyond ordinary stacked logistic regression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import (
    DEFAULT_C_GRID,
    STACK_SPLITS,
    PredictionBlock,
    load_blocks_for_split,
    softmax_np,
    stack_feature_matrix,
    validate_prediction_alignment,
)
from jetclass_fresh.independent_fusion import (
    StandardizedLinearStacker,
    fit_stacker_selecting_c_on_val,
    metrics_from_probs,
    save_stacker,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .crossarch_experiment import (
    DIRECT_HLT_ARCHITECTURES,
    RECONSTRUCTOR_ARCHITECTURES,
    TEACHER_ARCHITECTURES,
    build_reco_domain_tagger_model_names,
    hlt_model_name,
)
from .crossarch_fusion import (
    entropy_from_probs,
    pairwise_disagreement_fraction,
    predicted_classes,
    top1_margin_from_probs,
)


EXPERIMENT_STEP = "crossarch_conditional_evidence_fusers"
LINEAR_SUITE = "linear"
NEURAL_SUITE = "neural"
SUITES = (LINEAR_SUITE, NEURAL_SUITE)


@dataclass(frozen=True)
class ConditionalEvidenceFuserConfig:
    """Configuration for conditional evidence fuser experiments."""

    prediction_dir: str
    output_dir: str
    suite: str
    hlt_model_names: tuple[str, ...] = tuple(hlt_model_name(arch) for arch in DIRECT_HLT_ARCHITECTURES)
    adapted_model_names: tuple[str, ...] = tuple(
        build_reco_domain_tagger_model_names(RECONSTRUCTOR_ARCHITECTURES, TEACHER_ARCHITECTURES)
    )
    splits: tuple[str, ...] = tuple(STACK_SPLITS)
    c_grid: tuple[float, ...] = tuple(DEFAULT_C_GRID)
    max_iter: int = 2000
    confirm_final_test: bool = False
    run_controls: bool = True
    control_seed: int = 12345
    residual_penalties: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.1)
    weight_decays: tuple[float, ...] = (0.0, 0.0001)
    confusion_pair_counts: tuple[int, ...] = (4, 8, 12, 20)
    neural_epochs: int = 40
    neural_batch_size: int = 8192
    neural_lr: float = 0.001
    neural_hidden_dims: tuple[int, ...] = (64, 128)
    neural_dropout: float = 0.05
    neural_device: str = "auto"
    neural_patience: int = 8

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"suite must be one of {SUITES}, got {self.suite!r}")
        unknown_splits = sorted(set(self.splits) - set(STACK_SPLITS))
        if unknown_splits:
            raise ValueError(f"Unknown splits: {unknown_splits}")
        if "final_test" in self.splits and not self.confirm_final_test:
            raise ValueError("Refusing to evaluate final_test without confirm_final_test=True")
        if tuple(self.splits) != tuple(STACK_SPLITS):
            raise ValueError(f"Conditional fusers require exactly {tuple(STACK_SPLITS)}")
        if not self.hlt_model_names:
            raise ValueError("At least one HLT model is required")
        if not self.adapted_model_names:
            raise ValueError("At least one adapted model is required")
        if any(float(value) <= 0.0 for value in self.c_grid):
            raise ValueError("c_grid values must be positive")
        if any(float(value) < 0.0 for value in self.residual_penalties):
            raise ValueError("residual_penalties must be non-negative")
        if any(float(value) < 0.0 for value in self.weight_decays):
            raise ValueError("weight_decays must be non-negative")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if int(self.neural_epochs) <= 0:
            raise ValueError("neural_epochs must be positive")
        if int(self.neural_batch_size) <= 0:
            raise ValueError("neural_batch_size must be positive")
        if float(self.neural_lr) <= 0.0:
            raise ValueError("neural_lr must be positive")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                }
            )


def _split_blocks(
    prediction_dir: str | Path,
    *,
    model_names: Sequence[str],
    splits: Sequence[str] = STACK_SPLITS,
) -> dict[str, list[PredictionBlock]]:
    return {
        split: load_blocks_for_split(prediction_dir, model_names, split)
        for split in splits
    }


def _labels_by_split(blocks_by_split: Mapping[str, Sequence[PredictionBlock]]) -> dict[str, np.ndarray]:
    return {split: blocks[0].labels for split, blocks in blocks_by_split.items()}


def _metrics_by_split(probs_by_split: Mapping[str, np.ndarray], labels_by_split: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        split: metrics_from_probs(probs_by_split[split], labels_by_split[split])
        for split in STACK_SPLITS
    }


def _metric_summary(method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    return {
        "method": method,
        "stack_val_accuracy": (metrics.get("stack_val") or {}).get("accuracy"),
        "stack_val_cross_entropy": (metrics.get("stack_val") or {}).get("cross_entropy"),
        "final_test_accuracy": (metrics.get("final_test") or {}).get("accuracy"),
        "final_test_cross_entropy": (metrics.get("final_test") or {}).get("cross_entropy"),
        "status": payload.get("status", "ok"),
    }


def _per_class_rows(
    method: str,
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    baseline_probs: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    preds = np.argmax(probs, axis=1)
    baseline_preds = np.argmax(baseline_probs, axis=1) if baseline_probs is not None else None
    rows = []
    for class_index, class_name in enumerate(LABEL_NAMES[: probs.shape[1]]):
        mask = labels == class_index
        if not np.any(mask):
            continue
        accuracy = float(np.mean(preds[mask] == labels[mask]))
        row = {
            "method": method,
            "class_idx": int(class_index),
            "class_name": str(class_name),
            "n": int(np.sum(mask)),
            "class_accuracy": accuracy,
        }
        if baseline_preds is not None:
            baseline_accuracy = float(np.mean(baseline_preds[mask] == labels[mask]))
            row["delta_vs_hlt4_anchor"] = accuracy - baseline_accuracy
        rows.append(row)
    return rows


def _audit_blocks(blocks_by_split: Mapping[str, Sequence[PredictionBlock]]) -> dict[str, Any]:
    problems: list[str] = []
    identity_sets: dict[str, set[str]] = {}
    split_rows: dict[str, Any] = {}
    for split in STACK_SPLITS:
        blocks = list(blocks_by_split[split])
        validate_prediction_alignment(blocks)
        first = blocks[0]
        identities = {identity.key() for identity in first.jet_ids}
        identity_sets[split] = identities
        split_rows[split] = {
            "n_jets": int(len(first.labels)),
            "model_names": [block.model_name for block in blocks],
            "jet_identity_hashes": {
                block.model_name: block.metadata.get("jet_identity_hash")
                for block in blocks
            },
        }
    overlaps = {}
    for index, left in enumerate(STACK_SPLITS):
        for right in STACK_SPLITS[index + 1 :]:
            count = len(identity_sets[left] & identity_sets[right])
            overlaps[f"{left}__{right}"] = int(count)
            if count:
                problems.append(f"{left} and {right} overlap in {count} jet identities")
    return {
        "ok": not problems,
        "problems": problems,
        "splits": split_rows,
        "cross_split_overlap_counts": overlaps,
    }


def _anchor_hlt4(
    prediction_dir: str | Path,
    hlt_model_names: Sequence[str],
    *,
    c_grid: Sequence[float],
    max_iter: int,
) -> dict[str, Any]:
    blocks_by_split = _split_blocks(prediction_dir, model_names=hlt_model_names)
    x_train = stack_feature_matrix(blocks_by_split["stack_train"], feature_mode="logits_probs")
    x_val = stack_feature_matrix(blocks_by_split["stack_val"], feature_mode="logits_probs")
    y_train = blocks_by_split["stack_train"][0].labels
    y_val = blocks_by_split["stack_val"][0].labels
    num_classes = int(blocks_by_split["stack_train"][0].logits.shape[1])
    stacker, selection = fit_stacker_selecting_c_on_val(
        x_train,
        y_train,
        x_val,
        y_val,
        c_grid=c_grid,
        max_iter=max_iter,
        feature_mode="hlt4_logits_probs_anchor",
        model_names=hlt_model_names,
        num_classes=num_classes,
    )
    logits_by_split = {}
    probs_by_split = {}
    for split in STACK_SPLITS:
        x = stack_feature_matrix(blocks_by_split[split], feature_mode="logits_probs")
        logits = stacker.predict_logits(x).astype(np.float32)
        logits_by_split[split] = logits
        probs_by_split[split] = softmax_np(logits)
    labels = _labels_by_split(blocks_by_split)
    return {
        "blocks_by_split": blocks_by_split,
        "stacker": stacker,
        "selection": selection,
        "logits_by_split": logits_by_split,
        "probs_by_split": probs_by_split,
        "metrics": _metrics_by_split(probs_by_split, labels),
    }


def _model_uncertainty_columns(
    blocks: Sequence[PredictionBlock],
    *,
    prefix: str,
) -> tuple[list[np.ndarray], list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for block in blocks:
        entropy = entropy_from_probs(block.probs)
        margin = top1_margin_from_probs(block.probs)
        max_prob = np.max(block.probs, axis=1).astype(np.float32)
        columns.extend([entropy, margin, max_prob])
        names.extend(
            [
                f"{prefix}entropy::{block.model_name}",
                f"{prefix}margin::{block.model_name}",
                f"{prefix}max_prob::{block.model_name}",
            ]
        )
    return columns, names


def build_conditional_features(
    adapted_blocks: Sequence[PredictionBlock],
    hlt_blocks: Sequence[PredictionBlock],
    *,
    anchor_logits: np.ndarray,
    anchor_probs: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build HLT-anchor and adapted-evidence features for one split."""

    validate_prediction_alignment([*adapted_blocks, *hlt_blocks])
    num_classes = int(anchor_logits.shape[1])
    columns: list[np.ndarray] = []
    names: list[str] = []

    def add_matrix(matrix: np.ndarray, *, prefix: str) -> None:
        for class_index in range(num_classes):
            columns.append(matrix[:, class_index].astype(np.float32))
            names.append(f"{prefix}class_{class_index}")

    add_matrix(anchor_logits, prefix="anchor_logit::")
    add_matrix(anchor_probs, prefix="anchor_prob::")
    columns.extend(
        [
            entropy_from_probs(anchor_probs),
            top1_margin_from_probs(anchor_probs),
            np.max(anchor_probs, axis=1).astype(np.float32),
        ]
    )
    names.extend(["anchor_entropy", "anchor_margin", "anchor_max_prob"])

    for block in adapted_blocks:
        add_matrix(block.logits, prefix=f"logit::{block.model_name}::")
        add_matrix(block.probs, prefix=f"prob::{block.model_name}::")
    uncertainty_columns, uncertainty_names = _model_uncertainty_columns(adapted_blocks, prefix="adapted_")
    columns.extend(uncertainty_columns)
    names.extend(uncertainty_names)

    adapted_logits = np.stack([block.logits.astype(np.float32) for block in adapted_blocks], axis=1)
    adapted_probs = np.stack([block.probs.astype(np.float32) for block in adapted_blocks], axis=1)
    for class_index in range(num_classes):
        for stat_name, values in (
            ("adapted_mean_logit", np.mean(adapted_logits[:, :, class_index], axis=1)),
            ("adapted_std_logit", np.std(adapted_logits[:, :, class_index], axis=1)),
            ("adapted_max_logit", np.max(adapted_logits[:, :, class_index], axis=1)),
            ("adapted_min_logit", np.min(adapted_logits[:, :, class_index], axis=1)),
            ("adapted_mean_prob", np.mean(adapted_probs[:, :, class_index], axis=1)),
            ("adapted_std_prob", np.std(adapted_probs[:, :, class_index], axis=1)),
            ("adapted_max_prob", np.max(adapted_probs[:, :, class_index], axis=1)),
            ("adapted_min_prob", np.min(adapted_probs[:, :, class_index], axis=1)),
        ):
            columns.append(values.astype(np.float32))
            names.append(f"{stat_name}::class_{class_index}")

    adapted_preds = predicted_classes(adapted_blocks)
    anchor_pred = np.argmax(anchor_probs, axis=1).astype(np.int64)
    for class_index in range(num_classes):
        columns.append(np.mean(adapted_preds == class_index, axis=1).astype(np.float32))
        names.append(f"adapted_vote_fraction::class_{class_index}")
        columns.append((anchor_pred == class_index).astype(np.float32))
        names.append(f"anchor_pred_onehot::class_{class_index}")
    columns.extend(
        [
            pairwise_disagreement_fraction(adapted_preds),
            np.mean(adapted_preds == anchor_pred[:, None], axis=1).astype(np.float32),
        ]
    )
    names.extend(["adapted_pairwise_disagreement", "adapted_anchor_agreement_fraction"])

    # Include direct HLT raw evidence after the explicit anchor, so the plain
    # feature fusers can decide whether individual HLT models add residual value.
    for block in hlt_blocks:
        add_matrix(block.logits, prefix=f"hlt_logit::{block.model_name}::")
        add_matrix(block.probs, prefix=f"hlt_prob::{block.model_name}::")
    hlt_uncertainty_columns, hlt_uncertainty_names = _model_uncertainty_columns(hlt_blocks, prefix="hlt_")
    columns.extend(hlt_uncertainty_columns)
    names.extend(hlt_uncertainty_names)

    features = np.stack(columns, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise FloatingPointError("Conditional evidence features contain non-finite values")
    return features, tuple(names)


def _conditional_feature_sets(
    adapted_blocks_by_split: Mapping[str, Sequence[PredictionBlock]],
    hlt_blocks_by_split: Mapping[str, Sequence[PredictionBlock]],
    anchor: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    features_by_split: dict[str, np.ndarray] = {}
    names: tuple[str, ...] | None = None
    for split in STACK_SPLITS:
        features, split_names = build_conditional_features(
            adapted_blocks_by_split[split],
            hlt_blocks_by_split[split],
            anchor_logits=anchor["logits_by_split"][split],
            anchor_probs=anchor["probs_by_split"][split],
        )
        if names is None:
            names = split_names
        elif names != split_names:
            raise ValueError(f"Feature names changed across splits at {split}")
        features_by_split[split] = features
    assert names is not None
    return features_by_split, names


def _fit_logistic_method(
    *,
    method_name: str,
    features_by_split: Mapping[str, np.ndarray],
    labels_by_split: Mapping[str, np.ndarray],
    c_grid: Sequence[float],
    max_iter: int,
    model_names: Sequence[str],
    stacker_dir: Path,
) -> dict[str, Any]:
    num_classes = int(np.max(labels_by_split["stack_train"]) + 1)
    stacker, selection = fit_stacker_selecting_c_on_val(
        features_by_split["stack_train"],
        labels_by_split["stack_train"],
        features_by_split["stack_val"],
        labels_by_split["stack_val"],
        c_grid=c_grid,
        max_iter=max_iter,
        feature_mode=method_name,
        model_names=model_names,
        num_classes=num_classes,
    )
    probs_by_split = {
        split: stacker.predict_probs(features_by_split[split])
        for split in STACK_SPLITS
    }
    save_stacker(stacker, stacker_dir / f"{method_name}.npz")
    return {
        "status": "ok",
        "method": "multinomial_logistic_regression",
        "selection": selection,
        "metrics": _metrics_by_split(probs_by_split, labels_by_split),
        "probs_by_split": probs_by_split,
        "stacker_path": str(stacker_dir / f"{method_name}.npz"),
    }


def _standardize(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.astype(np.float64).mean(axis=0)
    scale = train_x.astype(np.float64).std(axis=0)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def _standardize_with(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((features.astype(np.float32) - mean) / scale).astype(np.float32)


def _fit_residual_linear_torch(
    *,
    features_by_split: Mapping[str, np.ndarray],
    labels_by_split: Mapping[str, np.ndarray],
    anchor_logits_by_split: Mapping[str, np.ndarray],
    residual_penalties: Sequence[float],
    weight_decays: Sequence[float],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - depends on optional torch env
        return {"status": "skipped", "reason": f"torch unavailable: {exc}"}

    resolved = torch.device("cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(int(seed))
    train_x = features_by_split["stack_train"]
    mean, scale = _standardize(train_x)
    train_y = labels_by_split["stack_train"].astype(np.int64)
    val_y = labels_by_split["stack_val"].astype(np.int64)
    num_classes = int(anchor_logits_by_split["stack_train"].shape[1])
    candidates = []
    best_state: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None

    for residual_penalty in residual_penalties:
        for weight_decay in weight_decays:
            model = nn.Linear(train_x.shape[1], num_classes).to(resolved)
            nn.init.zeros_(model.weight)
            nn.init.zeros_(model.bias)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
            criterion = nn.CrossEntropyLoss()
            x_train_t = torch.from_numpy(_standardize_with(train_x, mean, scale)).to(resolved)
            y_train_t = torch.from_numpy(train_y).to(resolved)
            anchor_train_t = torch.from_numpy(anchor_logits_by_split["stack_train"].astype(np.float32)).to(resolved)
            n_rows = int(x_train_t.shape[0])
            for _ in range(int(epochs)):
                order = torch.randperm(n_rows, device=resolved)
                for start in range(0, n_rows, int(batch_size)):
                    idx = order[start : start + int(batch_size)]
                    delta = model(x_train_t[idx])
                    logits = anchor_train_t[idx] + delta
                    loss = criterion(logits, y_train_t[idx])
                    if float(residual_penalty) > 0.0:
                        loss = loss + float(residual_penalty) * torch.mean(delta * delta)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            probs_by_split = {}
            with torch.no_grad():
                for split in STACK_SPLITS:
                    x_t = torch.from_numpy(_standardize_with(features_by_split[split], mean, scale)).to(resolved)
                    anchor_t = torch.from_numpy(anchor_logits_by_split[split].astype(np.float32)).to(resolved)
                    logits = anchor_t + model(x_t)
                    probs_by_split[split] = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            metrics = _metrics_by_split(probs_by_split, labels_by_split)
            key = (
                float(metrics["stack_val"]["accuracy"]),
                -float(metrics["stack_val"]["cross_entropy"]),
            )
            candidates.append(
                {
                    "residual_penalty": float(residual_penalty),
                    "weight_decay": float(weight_decay),
                    "metrics": metrics["stack_val"],
                }
            )
            if best_key is None or key > best_key:
                best_key = key
                best_state = {
                    "selection": {
                        "selection_split": "stack_val",
                        "residual_penalty": float(residual_penalty),
                        "weight_decay": float(weight_decay),
                        "epochs": int(epochs),
                    },
                    "metrics": metrics,
                    "probs_by_split": probs_by_split,
                    "coef": model.weight.detach().cpu().numpy().astype(np.float32),
                    "bias": model.bias.detach().cpu().numpy().astype(np.float32),
                    "mean": mean,
                    "scale": scale,
                }
    assert best_state is not None
    return {
        "status": "ok",
        "method": "anchored_linear_residual",
        "selection": {**best_state["selection"], "candidates": candidates, "device": str(resolved)},
        "metrics": best_state["metrics"],
        "probs_by_split": best_state["probs_by_split"],
        "linear_state": {
            "coef_shape": list(best_state["coef"].shape),
            "bias_shape": list(best_state["bias"].shape),
            "mean_shape": list(best_state["mean"].shape),
            "scale_shape": list(best_state["scale"].shape),
        },
    }


def _fit_ovr_calibrator(
    *,
    features_by_split: Mapping[str, np.ndarray],
    labels_by_split: Mapping[str, np.ndarray],
    c_grid: Sequence[float],
    max_iter: int,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - depends on optional sklearn env
        return {"status": "skipped", "reason": f"sklearn unavailable: {exc}"}

    train_x = features_by_split["stack_train"]
    mean, scale = _standardize(train_x)
    x_train = _standardize_with(train_x, mean, scale)
    x_val = _standardize_with(features_by_split["stack_val"], mean, scale)
    labels_train = labels_by_split["stack_train"].astype(np.int64)
    labels_val = labels_by_split["stack_val"].astype(np.int64)
    num_classes = int(np.max(labels_train) + 1)
    candidates = []
    best: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None
    for c_value in c_grid:
        for class_weight in ("balanced",):
            class_models = []
            val_logits = np.zeros((len(labels_val), num_classes), dtype=np.float32)
            for class_index in range(num_classes):
                binary_y = (labels_train == class_index).astype(np.int64)
                clf = LogisticRegression(
                    C=float(c_value),
                    solver="lbfgs",
                    max_iter=int(max_iter),
                    class_weight=class_weight,
                )
                clf.fit(x_train, binary_y)
                class_models.append(clf)
                val_logits[:, class_index] = clf.decision_function(x_val).astype(np.float32)
            val_probs = softmax_np(val_logits)
            val_metrics = metrics_from_probs(val_probs, labels_val)
            candidate = {
                "C": float(c_value),
                "class_weight": class_weight or "none",
                "metrics": val_metrics,
            }
            candidates.append(candidate)
            key = (float(val_metrics["accuracy"]), -float(val_metrics["cross_entropy"]))
            if best_key is None or key > best_key:
                best_key = key
                best = {"candidate": candidate, "models": class_models}
    assert best is not None
    probs_by_split = {}
    for split in STACK_SPLITS:
        x = _standardize_with(features_by_split[split], mean, scale)
        logits = np.zeros((x.shape[0], num_classes), dtype=np.float32)
        for class_index, clf in enumerate(best["models"]):
            logits[:, class_index] = clf.decision_function(x).astype(np.float32)
        probs_by_split[split] = softmax_np(logits)
    return {
        "status": "ok",
        "method": "one_vs_rest_class_calibrators",
        "selection": {"selection_split": "stack_val", **best["candidate"], "candidates": candidates},
        "metrics": _metrics_by_split(probs_by_split, labels_by_split),
        "probs_by_split": probs_by_split,
    }


def _top_confusion_pairs(anchor_probs: np.ndarray, labels: np.ndarray, *, max_pairs: int) -> list[tuple[int, int]]:
    preds = np.argmax(anchor_probs, axis=1).astype(np.int64)
    counts: dict[tuple[int, int], int] = {}
    for truth, pred in zip(labels.astype(np.int64), preds):
        if truth == pred:
            continue
        pair = tuple(sorted((int(truth), int(pred))))
        counts[pair] = counts.get(pair, 0) + 1
    return [pair for pair, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[: int(max_pairs)]]


def _pair_key_from_anchor(anchor_probs: np.ndarray) -> np.ndarray:
    top2 = np.argpartition(anchor_probs, kth=-2, axis=1)[:, -2:]
    return np.sort(top2, axis=1).astype(np.int64)


def _fit_confusion_pair_corrector(
    *,
    features_by_split: Mapping[str, np.ndarray],
    labels_by_split: Mapping[str, np.ndarray],
    anchor_probs_by_split: Mapping[str, np.ndarray],
    c_grid: Sequence[float],
    pair_counts: Sequence[int],
    max_iter: int,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover
        return {"status": "skipped", "reason": f"sklearn unavailable: {exc}"}

    mean, scale = _standardize(features_by_split["stack_train"])
    x_train = _standardize_with(features_by_split["stack_train"], mean, scale)
    y_train = labels_by_split["stack_train"].astype(np.int64)
    y_val = labels_by_split["stack_val"].astype(np.int64)
    all_pairs = _top_confusion_pairs(
        anchor_probs_by_split["stack_val"],
        y_val,
        max_pairs=max(pair_counts),
    )
    candidates = []
    best: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None
    for pair_count in pair_counts:
        pairs = all_pairs[: int(pair_count)]
        for c_value in c_grid:
            models = {}
            for left, right in pairs:
                mask = (y_train == left) | (y_train == right)
                if np.sum(mask) < 2 or len(np.unique(y_train[mask])) < 2:
                    continue
                binary_y = (y_train[mask] == right).astype(np.int64)
                clf = LogisticRegression(
                    C=float(c_value),
                    solver="lbfgs",
                    max_iter=int(max_iter),
                    class_weight="balanced",
                )
                clf.fit(x_train[mask], binary_y)
                models[(int(left), int(right))] = clf
            probs_by_split = _apply_pair_correctors(
                features_by_split,
                anchor_probs_by_split,
                models,
                mean=mean,
                scale=scale,
            )
            val_metrics = metrics_from_probs(probs_by_split["stack_val"], y_val)
            candidate = {
                "pair_count": int(pair_count),
                "C": float(c_value),
                "pairs": [list(pair) for pair in pairs],
                "fitted_pair_count": int(len(models)),
                "metrics": val_metrics,
            }
            candidates.append(candidate)
            key = (float(val_metrics["accuracy"]), -float(val_metrics["cross_entropy"]))
            if best_key is None or key > best_key:
                best_key = key
                best = {"candidate": candidate, "models": models, "probs_by_split": probs_by_split}
    if best is None:
        return {"status": "skipped", "reason": "No trainable confusion pairs were found"}
    return {
        "status": "ok",
        "method": "hlt_anchor_confusion_pair_corrector",
        "selection": {"selection_split": "stack_val", **best["candidate"], "candidates": candidates},
        "metrics": _metrics_by_split(best["probs_by_split"], labels_by_split),
        "probs_by_split": best["probs_by_split"],
    }


def _apply_pair_correctors(
    features_by_split: Mapping[str, np.ndarray],
    anchor_probs_by_split: Mapping[str, np.ndarray],
    models: Mapping[tuple[int, int], Any],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    probs_by_split = {}
    for split in STACK_SPLITS:
        probs = anchor_probs_by_split[split].astype(np.float64).copy()
        pair_keys = _pair_key_from_anchor(probs)
        x = _standardize_with(features_by_split[split], mean, scale)
        for pair, clf in models.items():
            left, right = pair
            mask = (pair_keys[:, 0] == left) & (pair_keys[:, 1] == right)
            if not np.any(mask):
                continue
            pair_prob_right = clf.predict_proba(x[mask])[:, 1]
            pair_mass = probs[mask, left] + probs[mask, right]
            probs[mask, left] = pair_mass * (1.0 - pair_prob_right)
            probs[mask, right] = pair_mass * pair_prob_right
        probs = np.clip(probs, 1.0e-12, None)
        probs_by_split[split] = (probs / np.sum(probs, axis=1, keepdims=True)).astype(np.float32)
    return probs_by_split


def _fit_neural_method(
    *,
    method_name: str,
    gated: bool,
    features_by_split: Mapping[str, np.ndarray],
    labels_by_split: Mapping[str, np.ndarray],
    anchor_logits_by_split: Mapping[str, np.ndarray],
    hidden_dims: Sequence[int],
    residual_penalties: Sequence[float],
    weight_decays: Sequence[float],
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    device: str,
    patience: int,
    seed: int,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        return {"status": "skipped", "reason": f"torch unavailable: {exc}"}

    resolved = torch.device("cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(int(seed))
    mean, scale = _standardize(features_by_split["stack_train"])
    x_train_np = _standardize_with(features_by_split["stack_train"], mean, scale)
    x_val_np = _standardize_with(features_by_split["stack_val"], mean, scale)
    y_train_np = labels_by_split["stack_train"].astype(np.int64)
    y_val_np = labels_by_split["stack_val"].astype(np.int64)
    num_classes = int(anchor_logits_by_split["stack_train"].shape[1])

    class ResidualMlp(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            output_dim = num_classes * 2 if gated else num_classes
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, output_dim),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, x, anchor):
            raw = self.net(x)
            if gated:
                residual, gate_logits = torch.chunk(raw, 2, dim=1)
                gate = torch.sigmoid(gate_logits)
                return anchor + gate * residual, residual, gate
            return anchor + raw, raw, None

    train_x = torch.from_numpy(x_train_np).to(resolved)
    train_y = torch.from_numpy(y_train_np).to(resolved)
    train_anchor = torch.from_numpy(anchor_logits_by_split["stack_train"].astype(np.float32)).to(resolved)
    val_x = torch.from_numpy(x_val_np).to(resolved)
    val_y = torch.from_numpy(y_val_np).to(resolved)
    val_anchor = torch.from_numpy(anchor_logits_by_split["stack_val"].astype(np.float32)).to(resolved)
    criterion = nn.CrossEntropyLoss()
    n_rows = int(train_x.shape[0])
    candidates = []
    best: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None

    for hidden_dim in hidden_dims:
        for residual_penalty in residual_penalties:
            for weight_decay in weight_decays:
                model = ResidualMlp(train_x.shape[1], int(hidden_dim)).to(resolved)
                optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
                local_best_state = None
                local_best_key = None
                stale = 0
                history = []
                for epoch in range(1, int(epochs) + 1):
                    model.train()
                    order = torch.randperm(n_rows, device=resolved)
                    for start in range(0, n_rows, int(batch_size)):
                        idx = order[start : start + int(batch_size)]
                        logits, residual, gate = model(train_x[idx], train_anchor[idx])
                        loss = criterion(logits, train_y[idx])
                        if float(residual_penalty) > 0.0:
                            loss = loss + float(residual_penalty) * torch.mean(residual * residual)
                        if gated and gate is not None:
                            loss = loss + 0.001 * torch.mean(gate)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        logits, _, _ = model(val_x, val_anchor)
                        val_probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
                    val_metrics = metrics_from_probs(val_probs, y_val_np)
                    key = (float(val_metrics["accuracy"]), -float(val_metrics["cross_entropy"]))
                    history.append({"epoch": int(epoch), "metrics": val_metrics})
                    if local_best_key is None or key > local_best_key:
                        local_best_key = key
                        local_best_state = {key_: value.detach().cpu().clone() for key_, value in model.state_dict().items()}
                        stale = 0
                    else:
                        stale += 1
                        if stale >= int(patience):
                            break
                assert local_best_state is not None
                model.load_state_dict(local_best_state)
                probs_by_split = {}
                model.eval()
                with torch.no_grad():
                    for split in STACK_SPLITS:
                        x = torch.from_numpy(_standardize_with(features_by_split[split], mean, scale)).to(resolved)
                        anchor = torch.from_numpy(anchor_logits_by_split[split].astype(np.float32)).to(resolved)
                        logits, _, _ = model(x, anchor)
                        probs_by_split[split] = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
                metrics = _metrics_by_split(probs_by_split, labels_by_split)
                candidate = {
                    "hidden_dim": int(hidden_dim),
                    "residual_penalty": float(residual_penalty),
                    "weight_decay": float(weight_decay),
                    "epochs_completed": int(history[-1]["epoch"]),
                    "best_stack_val_metrics": metrics["stack_val"],
                }
                candidates.append(candidate)
                key = (float(metrics["stack_val"]["accuracy"]), -float(metrics["stack_val"]["cross_entropy"]))
                if best_key is None or key > best_key:
                    best_key = key
                    best = {
                        "candidate": candidate,
                        "metrics": metrics,
                        "probs_by_split": probs_by_split,
                    }
    assert best is not None
    return {
        "status": "ok",
        "method": method_name,
        "selection": {
            "selection_split": "stack_val",
            "device": str(resolved),
            **best["candidate"],
            "candidates": candidates,
        },
        "metrics": best["metrics"],
        "probs_by_split": best["probs_by_split"],
    }


def _negative_controls(
    *,
    features_by_split: Mapping[str, np.ndarray],
    labels_by_split: Mapping[str, np.ndarray],
    c_grid: Sequence[float],
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.RandomState(int(seed))
    x_train = features_by_split["stack_train"]
    y_train = labels_by_split["stack_train"]
    x_val = features_by_split["stack_val"]
    y_val = labels_by_split["stack_val"]
    x_test = features_by_split["final_test"]
    y_test = labels_by_split["final_test"]
    num_classes = int(np.max(y_train) + 1)

    shuffled_y = y_train.copy()
    rng.shuffle(shuffled_y)
    perm_stacker, perm_selection = fit_stacker_selecting_c_on_val(
        x_train,
        shuffled_y,
        x_val,
        y_val,
        c_grid=c_grid,
        max_iter=max_iter,
        feature_mode="conditional_permuted_labels",
        model_names=["conditional_features"],
        num_classes=num_classes,
    )
    shuffled_x = x_train.copy()
    for column in range(shuffled_x.shape[1]):
        shuffled_x[:, column] = shuffled_x[rng.permutation(shuffled_x.shape[0]), column]
    row_stacker, row_selection = fit_stacker_selecting_c_on_val(
        shuffled_x,
        y_train,
        x_val,
        y_val,
        c_grid=c_grid,
        max_iter=max_iter,
        feature_mode="conditional_row_shuffled_features",
        model_names=["conditional_features"],
        num_classes=num_classes,
    )
    return {
        "enabled": True,
        "permuted_labels": {
            "selection": perm_selection,
            "metrics": {
                "stack_val": metrics_from_probs(perm_stacker.predict_probs(x_val), y_val),
                "final_test": metrics_from_probs(perm_stacker.predict_probs(x_test), y_test),
            },
        },
        "row_shuffled_features": {
            "selection": row_selection,
            "metrics": {
                "stack_val": metrics_from_probs(row_stacker.predict_probs(x_val), y_val),
                "final_test": metrics_from_probs(row_stacker.predict_probs(x_test), y_test),
            },
        },
    }


def run_conditional_evidence_fusers(config: ConditionalEvidenceFuserConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    report_path = output_dir / "conditional_fuser_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite locked conditional fuser report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stacker_dir = output_dir / "stackers"
    stacker_dir.mkdir(parents=True, exist_ok=True)

    prediction_dir = Path(config.prediction_dir)
    hlt_blocks_by_split = _split_blocks(prediction_dir, model_names=config.hlt_model_names)
    adapted_blocks_by_split = _split_blocks(prediction_dir, model_names=config.adapted_model_names)
    all_blocks_by_split = {
        split: [*adapted_blocks_by_split[split], *hlt_blocks_by_split[split]]
        for split in STACK_SPLITS
    }
    audit = _audit_blocks(all_blocks_by_split)
    anchor = _anchor_hlt4(
        prediction_dir,
        config.hlt_model_names,
        c_grid=config.c_grid,
        max_iter=config.max_iter,
    )
    features_by_split, feature_names = _conditional_feature_sets(adapted_blocks_by_split, hlt_blocks_by_split, anchor)
    labels_by_split = _labels_by_split(hlt_blocks_by_split)

    methods: dict[str, dict[str, Any]] = {
        "hlt4_anchor": {
            "status": "ok",
            "method": "fixed_hlt4_logistic_anchor",
            "selection": anchor["selection"],
            "metrics": anchor["metrics"],
            "probs_by_split": anchor["probs_by_split"],
        }
    }

    all20_raw_features = {
        split: stack_feature_matrix(all_blocks_by_split[split], feature_mode="logits_probs")
        for split in STACK_SPLITS
    }
    adapted16_raw_features = {
        split: stack_feature_matrix(adapted_blocks_by_split[split], feature_mode="logits_probs")
        for split in STACK_SPLITS
    }
    methods["plain_logistic_all20_logits_probs"] = _fit_logistic_method(
        method_name="plain_logistic_all20_logits_probs",
        features_by_split=all20_raw_features,
        labels_by_split=labels_by_split,
        c_grid=config.c_grid,
        max_iter=config.max_iter,
        model_names=[*config.adapted_model_names, *config.hlt_model_names],
        stacker_dir=stacker_dir,
    )
    methods["plain_logistic_adapted16_logits_probs"] = _fit_logistic_method(
        method_name="plain_logistic_adapted16_logits_probs",
        features_by_split=adapted16_raw_features,
        labels_by_split=labels_by_split,
        c_grid=config.c_grid,
        max_iter=config.max_iter,
        model_names=config.adapted_model_names,
        stacker_dir=stacker_dir,
    )
    methods["plain_logistic_conditional_features"] = _fit_logistic_method(
        method_name="plain_logistic_conditional_features",
        features_by_split=features_by_split,
        labels_by_split=labels_by_split,
        c_grid=config.c_grid,
        max_iter=config.max_iter,
        model_names=[*config.adapted_model_names, *config.hlt_model_names],
        stacker_dir=stacker_dir,
    )

    if config.suite == LINEAR_SUITE:
        methods["anchored_linear_residual"] = _fit_residual_linear_torch(
            features_by_split=features_by_split,
            labels_by_split=labels_by_split,
            anchor_logits_by_split=anchor["logits_by_split"],
            residual_penalties=config.residual_penalties,
            weight_decays=config.weight_decays,
            epochs=min(int(config.neural_epochs), 20),
            batch_size=config.neural_batch_size,
            lr=config.neural_lr,
            device=config.neural_device,
            seed=config.control_seed,
        )
        methods["one_vs_rest_class_calibrators"] = _fit_ovr_calibrator(
            features_by_split=features_by_split,
            labels_by_split=labels_by_split,
            c_grid=config.c_grid,
            max_iter=config.max_iter,
        )
        methods["confusion_pair_corrector"] = _fit_confusion_pair_corrector(
            features_by_split=features_by_split,
            labels_by_split=labels_by_split,
            anchor_probs_by_split=anchor["probs_by_split"],
            c_grid=config.c_grid,
            pair_counts=config.confusion_pair_counts,
            max_iter=config.max_iter,
        )
    else:
        methods["anchored_residual_mlp"] = _fit_neural_method(
            method_name="anchored_residual_mlp",
            gated=False,
            features_by_split=features_by_split,
            labels_by_split=labels_by_split,
            anchor_logits_by_split=anchor["logits_by_split"],
            hidden_dims=config.neural_hidden_dims,
            residual_penalties=config.residual_penalties,
            weight_decays=config.weight_decays,
            epochs=config.neural_epochs,
            batch_size=config.neural_batch_size,
            lr=config.neural_lr,
            dropout=config.neural_dropout,
            device=config.neural_device,
            patience=config.neural_patience,
            seed=config.control_seed,
        )
        methods["class_gated_residual_mlp"] = _fit_neural_method(
            method_name="class_gated_residual_mlp",
            gated=True,
            features_by_split=features_by_split,
            labels_by_split=labels_by_split,
            anchor_logits_by_split=anchor["logits_by_split"],
            hidden_dims=config.neural_hidden_dims,
            residual_penalties=config.residual_penalties,
            weight_decays=config.weight_decays,
            epochs=config.neural_epochs,
            batch_size=config.neural_batch_size,
            lr=config.neural_lr,
            dropout=config.neural_dropout,
            device=config.neural_device,
            patience=config.neural_patience,
            seed=config.control_seed,
        )

    controls = (
        _negative_controls(
            features_by_split=features_by_split,
            labels_by_split=labels_by_split,
            c_grid=config.c_grid,
            max_iter=config.max_iter,
            seed=config.control_seed,
        )
        if config.run_controls
        else {"enabled": False, "reason": "disabled by configuration"}
    )

    summary_rows = [_metric_summary(name, payload) for name, payload in methods.items()]
    summary_rows = sorted(
        summary_rows,
        key=lambda row: (row["final_test_accuracy"] is not None, row["final_test_accuracy"] or -1.0),
        reverse=True,
    )
    final_labels = labels_by_split["final_test"]
    anchor_final_probs = anchor["probs_by_split"]["final_test"]
    per_class_rows: list[dict[str, Any]] = []
    for method_name, payload in methods.items():
        probs_by_split = payload.get("probs_by_split")
        if not isinstance(probs_by_split, dict) or "final_test" not in probs_by_split:
            continue
        per_class_rows.extend(
            _per_class_rows(
                method_name,
                probs_by_split["final_test"],
                final_labels,
                baseline_probs=anchor_final_probs,
            )
        )

    public_methods = {
        name: {key: value for key, value in payload.items() if key != "probs_by_split"}
        for name, payload in methods.items()
    }
    report = {
        "ok": bool(audit["ok"]),
        "experiment_step": EXPERIMENT_STEP,
        "suite": config.suite,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "model_groups": {
            "hlt4_anchor_models": list(config.hlt_model_names),
            "adapted_models": list(config.adapted_model_names),
            "all20_models": [*config.adapted_model_names, *config.hlt_model_names],
        },
        "feature_summary": {
            "n_features": int(len(feature_names)),
            "feature_names_path": str(output_dir / "conditional_feature_names.json"),
            "shapes": {split: list(features_by_split[split].shape) for split in STACK_SPLITS},
        },
        "audit": audit,
        "methods": public_methods,
        "method_summary": summary_rows,
        "controls": controls,
        "leakage_rules": {
            "inputs": "frozen prediction blocks only",
            "anchor_fit_split": "stack_train",
            "anchor_selection_split": "stack_val",
            "fuser_fit_split": "stack_train",
            "fuser_selection_split": "stack_val",
            "final_test_evaluated_after_selection": True,
            "no_model_checkpoints_loaded": True,
            "no_training_data_loaded": True,
        },
        "output_files": {
            "report": str(report_path),
            "method_summary_csv": str(output_dir / "method_summary.csv"),
            "per_class_final_test_csv": str(output_dir / "per_class_final_test.csv"),
            "conditional_feature_names": str(output_dir / "conditional_feature_names.json"),
            "stacker_dir": str(stacker_dir),
        },
    }
    _write_json(output_dir / "conditional_feature_names.json", {"feature_names": list(feature_names)})
    _write_csv(output_dir / "method_summary.csv", summary_rows)
    _write_csv(output_dir / "per_class_final_test.csv", per_class_rows)
    _write_json(report_path, report)
    return report


__all__ = [
    "EXPERIMENT_STEP",
    "LINEAR_SUITE",
    "NEURAL_SUITE",
    "SUITES",
    "ConditionalEvidenceFuserConfig",
    "build_conditional_features",
    "run_conditional_evidence_fusers",
]

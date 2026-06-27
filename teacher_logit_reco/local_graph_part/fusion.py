"""Frozen-score fusion helpers for local-graph HLT ParT comparisons.

This module starts after neural inference.  It consumes cached binary logits
from the HLT ParT baseline and local-graph variants, builds small fusion
features, and selects simple score-level fusion rules on ``stack_val``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions

from .protocol import LOCAL_GRAPH_PART_PRIMARY_METRIC, LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
from .train import LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS


LOCAL_GRAPH_SCORE_FUSION_STEP = "local_graph_score_fusion_step1_core"
LOCAL_GRAPH_SCORE_FUSION_CONTRACT = "local_graph_score_fusion_core_v1"
LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT = "stack_val"
LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT = "final_test"
LOCAL_GRAPH_SCORE_FUSION_DEFAULT_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
LOCAL_GRAPH_SCORE_FUSION_SCORE_MODES = (
    "margin",
    "prob_signal",
    "log_prob_signal",
    "log_odds",
    "rank",
)
LOCAL_GRAPH_SCORE_FUSION_FEATURE_MODES = (
    "margins",
    "probabilities",
    "log_odds",
    "ranks",
    "disagreement",
)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {logits.shape}")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    return (exp / np.clip(denom, 1.0e-300, None)).astype(np.float64)


def sigmoid_np(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores >= 0.0
    output = np.empty_like(scores, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
    exp_scores = np.exp(scores[~positive])
    output[~positive] = exp_scores / (1.0 + exp_scores)
    return output


def logit_np(probs: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    clipped = np.clip(probs, float(eps), 1.0 - float(eps))
    return np.log(clipped / (1.0 - clipped))


def binary_margin_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"binary logits must have shape [N, 2], got {logits.shape}")
    return (logits[:, 1] - logits[:, 0]).astype(np.float64)


def binary_logits_from_log_odds(log_odds: np.ndarray) -> np.ndarray:
    log_odds = np.asarray(log_odds, dtype=np.float64).reshape(-1)
    return np.stack([-0.5 * log_odds, 0.5 * log_odds], axis=1).astype(np.float64)


def sanitize_binary_logits(logits: np.ndarray, *, source: str = "logits") -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"{source} must have shape [N, 2], got {logits.shape}")
    if not np.isfinite(logits).all():
        raise FloatingPointError(f"{source} contains non-finite values")
    return logits.astype(np.float32, copy=False)


def binary_metrics_from_signal_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    scores_are_probabilities: bool = False,
    label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
) -> dict[str, Any]:
    """Compute project-standard binary metrics from a scalar signal score.

    When ``scores_are_probabilities`` is false, ``scores`` are interpreted as
    log-odds/margins.  When true, ``scores`` are interpreted as signal
    probabilities.  In both cases the FPR operating points are computed from
    the resulting signal probability ranking.
    """

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("local-graph score fusion expects binary labels encoded as 0/1")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(f"score/label length mismatch: {scores.shape[0]} vs {labels.shape[0]}")
    if not np.isfinite(scores).all():
        raise FloatingPointError("fusion scores contain non-finite values")
    if bool(scores_are_probabilities):
        probs_signal = np.clip(scores, 1.0e-12, 1.0 - 1.0e-12)
        log_odds = logit_np(probs_signal)
    else:
        log_odds = scores
        probs_signal = sigmoid_np(log_odds)
    logits = binary_logits_from_log_odds(log_odds)
    preds = (probs_signal >= 0.5).astype(np.int64)
    return classification_metrics_from_predictions(
        preds=preds,
        labels=labels,
        logits=logits,
        label_names=tuple(label_names),
    )


def lookup_fusion_metric(metrics: Mapping[str, Any], metric_name: str = LOCAL_GRAPH_PART_PRIMARY_METRIC) -> float:
    if metric_name in metrics:
        return float(metrics[metric_name])
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping) and metric_name in binary:
        return float(binary[metric_name])
    raise KeyError(f"metrics do not contain {metric_name!r}")


def fusion_metric_score(metrics: Mapping[str, Any], metric_name: str = LOCAL_GRAPH_PART_PRIMARY_METRIC) -> tuple[float, float]:
    value = lookup_fusion_metric(metrics, metric_name)
    if not np.isfinite(value):
        return float("-inf"), value
    if metric_name in LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS:
        return -float(value), float(value)
    return float(value), float(value)


@dataclass
class LocalGraphPredictionBlock:
    """Frozen predictions for one local-graph variant on one split."""

    variant: str
    split: str
    logits: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.variant = str(self.variant)
        self.split = str(self.split)
        self.logits = sanitize_binary_logits(self.logits, source=f"{self.variant}/{self.split}/logits")
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        self.indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        if self.logits.shape[0] != self.labels.shape[0]:
            raise ValueError(f"{self.variant}/{self.split}: logits and labels have different lengths")
        if self.indices.shape[0] != self.labels.shape[0]:
            raise ValueError(f"{self.variant}/{self.split}: indices and labels have different lengths")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError(f"{self.variant}/{self.split}: labels must be encoded as 0/1")

    @property
    def probs(self) -> np.ndarray:
        return softmax_np(self.logits)

    @property
    def prob_signal(self) -> np.ndarray:
        return self.probs[:, 1]

    @property
    def prob_background(self) -> np.ndarray:
        return self.probs[:, 0]

    @property
    def margin(self) -> np.ndarray:
        return binary_margin_from_logits(self.logits)

    @property
    def log_odds(self) -> np.ndarray:
        return logit_np(self.prob_signal)

    @property
    def confidence(self) -> np.ndarray:
        return np.max(self.probs, axis=1)

    @property
    def entropy(self) -> np.ndarray:
        probs = np.clip(self.probs, 1.0e-12, 1.0)
        return -np.sum(probs * np.log(probs), axis=1)

    def metrics(self) -> dict[str, Any]:
        return binary_metrics_from_signal_scores(self.log_odds, self.labels)

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "split": self.split,
            "n_jets": int(self.labels.shape[0]),
            "logits_shape": list(self.logits.shape),
            "index_min": int(np.min(self.indices)) if self.indices.size else None,
            "index_max": int(np.max(self.indices)) if self.indices.size else None,
            "metadata": dict(self.metadata),
        }


def prediction_cache_paths(prediction_dir: str | Path, variant: str, split: str) -> tuple[Path, Path]:
    root = Path(prediction_dir) / str(variant)
    return root / f"{split}_predictions.npz", root / f"{split}_predictions_metadata.json"


def save_prediction_block(
    block: LocalGraphPredictionBlock,
    prediction_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    npz_path, meta_path = prediction_cache_paths(prediction_dir, block.variant, block.split)
    if not bool(overwrite) and (npz_path.exists() or meta_path.exists()):
        raise FileExistsError(f"prediction cache already exists for {block.variant}/{block.split}: {npz_path}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        logits=block.logits.astype(np.float32, copy=False),
        labels=block.labels.astype(np.int64, copy=False),
        indices=block.indices.astype(np.int64, copy=False),
        score_margin=block.margin.astype(np.float32),
        prob_signal=block.prob_signal.astype(np.float32),
        prob_background=block.prob_background.astype(np.float32),
    )
    metadata = {
        "step": LOCAL_GRAPH_SCORE_FUSION_STEP,
        "contract": LOCAL_GRAPH_SCORE_FUSION_CONTRACT,
        **block.to_manifest_row(),
        "npz_path": str(npz_path),
        "metadata_path": str(meta_path),
        "metrics": block.metrics(),
    }
    metadata.update(dict(block.metadata))
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def load_prediction_block(
    prediction_dir: str | Path,
    variant: str,
    split: str,
    *,
    require_metadata: bool = False,
) -> LocalGraphPredictionBlock:
    npz_path, meta_path = prediction_cache_paths(prediction_dir, variant, split)
    if not npz_path.exists():
        raise FileNotFoundError(f"missing prediction cache: {npz_path}")
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    elif bool(require_metadata):
        raise FileNotFoundError(f"missing prediction metadata: {meta_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        logits = data["logits"]
        labels = data["labels"]
        if "indices" in data:
            indices = data["indices"]
        else:
            indices = np.arange(labels.shape[0], dtype=np.int64)
    return LocalGraphPredictionBlock(
        variant=str(variant),
        split=str(split),
        logits=logits,
        labels=labels,
        indices=indices,
        metadata=metadata,
    )


def validate_prediction_alignment(blocks: Sequence[LocalGraphPredictionBlock]) -> None:
    if not blocks:
        raise ValueError("at least one prediction block is required")
    first = blocks[0]
    for block in blocks[1:]:
        if block.split != first.split:
            raise ValueError(f"split mismatch: {block.variant} has {block.split}, expected {first.split}")
        if not np.array_equal(block.labels, first.labels):
            raise ValueError(f"label mismatch between {first.variant} and {block.variant}")
        if not np.array_equal(block.indices, first.indices):
            raise ValueError(f"index mismatch between {first.variant} and {block.variant}")


def load_blocks_for_split(
    prediction_dir: str | Path,
    variants: Sequence[str],
    split: str,
    *,
    require_metadata: bool = False,
) -> list[LocalGraphPredictionBlock]:
    blocks = [
        load_prediction_block(prediction_dir, variant, split, require_metadata=require_metadata)
        for variant in variants
    ]
    validate_prediction_alignment(blocks)
    return blocks


def percentile_ranks_from_reference(reference_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    reference_scores = np.asarray(reference_scores, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if reference_scores.size == 0:
        raise ValueError("reference_scores cannot be empty")
    sorted_ref = np.sort(reference_scores)
    right = np.searchsorted(sorted_ref, scores, side="right")
    left = np.searchsorted(sorted_ref, scores, side="left")
    return (0.5 * (left + right) / float(reference_scores.size)).astype(np.float64)


def score_vector_for_block(
    block: LocalGraphPredictionBlock,
    *,
    mode: str = "margin",
    rank_reference: LocalGraphPredictionBlock | None = None,
) -> np.ndarray:
    mode = str(mode)
    if mode == "margin":
        return block.margin
    if mode == "prob_signal":
        return block.prob_signal
    if mode == "log_prob_signal":
        return np.log(np.clip(block.prob_signal, 1.0e-12, 1.0))
    if mode == "log_odds":
        return block.log_odds
    if mode == "rank":
        reference = rank_reference or block
        return percentile_ranks_from_reference(reference.margin, block.margin)
    raise ValueError(f"score mode must be one of {LOCAL_GRAPH_SCORE_FUSION_SCORE_MODES}, got {mode!r}")


@dataclass(frozen=True)
class FusionFeatureBlock:
    features: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    variants: tuple[str, ...]
    split: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "variants": list(self.variants),
            "feature_names": list(self.feature_names),
            "features_shape": list(self.features.shape),
            "n_jets": int(self.labels.shape[0]),
        }


def build_score_feature_block(
    blocks: Sequence[LocalGraphPredictionBlock],
    *,
    mode: str = "margin",
    rank_reference_blocks: Sequence[LocalGraphPredictionBlock] | None = None,
) -> FusionFeatureBlock:
    validate_prediction_alignment(blocks)
    refs: dict[str, LocalGraphPredictionBlock] = {}
    if rank_reference_blocks is not None:
        refs = {block.variant: block for block in rank_reference_blocks}
    columns = []
    names = []
    for block in blocks:
        columns.append(score_vector_for_block(block, mode=mode, rank_reference=refs.get(block.variant)))
        names.append(f"{block.variant}__{mode}")
    features = np.stack(columns, axis=1).astype(np.float64)
    if not np.isfinite(features).all():
        raise FloatingPointError("fusion score feature matrix contains non-finite values")
    return FusionFeatureBlock(
        features=features,
        labels=blocks[0].labels.copy(),
        feature_names=tuple(names),
        variants=tuple(block.variant for block in blocks),
        split=blocks[0].split,
    )


def build_disagreement_feature_block(
    blocks: Sequence[LocalGraphPredictionBlock],
    *,
    include_entropy: bool = True,
) -> FusionFeatureBlock:
    validate_prediction_alignment(blocks)
    columns: list[np.ndarray] = []
    names: list[str] = []
    margins: dict[str, np.ndarray] = {}
    for block in blocks:
        margin = block.margin
        margins[block.variant] = margin
        columns.append(margin)
        names.append(f"{block.variant}__margin")
        columns.append(block.confidence)
        names.append(f"{block.variant}__confidence")
        if bool(include_entropy):
            columns.append(block.entropy)
            names.append(f"{block.variant}__entropy")
    variants = [block.variant for block in blocks]
    for i, left in enumerate(variants):
        for right in variants[i + 1 :]:
            columns.append(np.abs(margins[left] - margins[right]))
            names.append(f"abs_margin_diff__{left}__{right}")
    features = np.stack(columns, axis=1).astype(np.float64)
    if not np.isfinite(features).all():
        raise FloatingPointError("disagreement feature matrix contains non-finite values")
    return FusionFeatureBlock(
        features=features,
        labels=blocks[0].labels.copy(),
        feature_names=tuple(names),
        variants=tuple(variants),
        split=blocks[0].split,
    )


def standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    return (features - mean) / scale, mean, scale


@dataclass
class BinaryLogisticStacker:
    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    C: float
    feature_names: tuple[str, ...]
    variants: tuple[str, ...]
    solver: str = "numpy_gd"

    def predict_score(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        x = (features - self.mean) / self.scale
        return (x @ self.coef.reshape(-1) + float(self.intercept)).astype(np.float64)

    def predict_prob_signal(self, features: np.ndarray) -> np.ndarray:
        return sigmoid_np(self.predict_score(features))

    def metrics(self, features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        return binary_metrics_from_signal_scores(self.predict_score(features), labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coef": self.coef.astype(float).tolist(),
            "intercept": float(self.intercept),
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
            "C": float(self.C),
            "feature_names": list(self.feature_names),
            "variants": list(self.variants),
            "solver": str(self.solver),
        }


def _fit_numpy_binary_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float,
    max_iter: int,
    learning_rate: float = 0.25,
) -> tuple[np.ndarray, float]:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train/y_train length mismatch")
    coef = np.zeros((x_train.shape[1],), dtype=np.float64)
    intercept = float(logit_np(np.asarray([np.clip(np.mean(y_train), 1.0e-4, 1.0 - 1.0e-4)]))[0])
    l2 = 1.0 / max(float(C), 1.0e-12)
    denom = float(max(x_train.shape[0], 1))
    for _ in range(int(max_iter)):
        scores = x_train @ coef + intercept
        probs = sigmoid_np(scores)
        error = probs - y_train
        grad_coef = (x_train.T @ error) / denom + (l2 / denom) * coef
        grad_intercept = float(np.mean(error))
        coef -= float(learning_rate) * grad_coef
        intercept -= float(learning_rate) * grad_intercept
    return coef, float(intercept)


def fit_binary_logistic_stacker(
    feature_block: FusionFeatureBlock,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    prefer_sklearn: bool = True,
) -> BinaryLogisticStacker:
    x_train, mean, scale = standardize_features(feature_block.features)
    y_train = np.asarray(feature_block.labels, dtype=np.int64)
    if not np.isin(y_train, [0, 1]).all():
        raise ValueError("binary logistic stacker expects 0/1 labels")
    solver = "numpy_gd"
    coef: np.ndarray
    intercept: float
    if bool(prefer_sklearn):
        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(
                C=float(C),
                solver="lbfgs",
                max_iter=int(max_iter),
                n_jobs=1,
            )
            clf.fit(x_train, y_train)
            if not np.array_equal(clf.classes_.astype(np.int64), np.asarray([0, 1], dtype=np.int64)):
                raise ValueError("sklearn logistic regression fitted an unexpected class set")
            coef = clf.coef_.reshape(-1).astype(np.float64)
            intercept = float(clf.intercept_.reshape(-1)[0])
            solver = "sklearn_lbfgs"
        except Exception:
            coef, intercept = _fit_numpy_binary_logistic(x_train, y_train, C=float(C), max_iter=int(max_iter))
    else:
        coef, intercept = _fit_numpy_binary_logistic(x_train, y_train, C=float(C), max_iter=int(max_iter))
    return BinaryLogisticStacker(
        coef=coef.astype(np.float64),
        intercept=float(intercept),
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        C=float(C),
        feature_names=tuple(feature_block.feature_names),
        variants=tuple(feature_block.variants),
        solver=solver,
    )


def fit_binary_logistic_stackers_selecting_c(
    stack_block: FusionFeatureBlock,
    *,
    c_grid: Sequence[float] = LOCAL_GRAPH_SCORE_FUSION_DEFAULT_C_GRID,
    max_iter: int = 1000,
    selection_metric: str = LOCAL_GRAPH_PART_PRIMARY_METRIC,
    prefer_sklearn: bool = True,
) -> tuple[BinaryLogisticStacker, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for c_value in c_grid:
        stacker = fit_binary_logistic_stacker(
            stack_block,
            C=float(c_value),
            max_iter=int(max_iter),
            prefer_sklearn=bool(prefer_sklearn),
        )
        metrics = stacker.metrics(stack_block.features, stack_block.labels)
        score, value = fusion_metric_score(metrics, selection_metric)
        candidates.append(
            {
                "C": float(c_value),
                "solver": stacker.solver,
                "selection_score": float(score),
                "selection_metric_value": float(value),
                "metrics": metrics,
                "stacker": stacker,
            }
        )
    best = max(candidates, key=lambda row: (row["selection_score"], -float(row["metrics"].get("loss", 0.0))))
    return best["stacker"], {
        "selection_split": stack_block.split,
        "selection_metric": str(selection_metric),
        "selected_C": float(best["C"]),
        "selected_solver": str(best["solver"]),
        "selected_metric_value": float(best["selection_metric_value"]),
        "selected_metrics": best["metrics"],
        "candidates": [
            {
                "C": row["C"],
                "solver": row["solver"],
                "selection_score": row["selection_score"],
                "selection_metric_value": row["selection_metric_value"],
                "metrics": row["metrics"],
            }
            for row in candidates
        ],
    }


def simplex_weight_grid(num_models: int, *, step: float = 0.05) -> np.ndarray:
    num_models = int(num_models)
    if num_models <= 0:
        raise ValueError("num_models must be positive")
    step = float(step)
    if step <= 0.0 or step > 1.0:
        raise ValueError("step must be in (0, 1]")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must evenly divide 1.0")
    rows: list[list[int]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append([*prefix, remaining])
            return
        for value in range(remaining + 1):
            rec([*prefix, value], remaining - value, slots - 1)

    rec([], units, num_models)
    return (np.asarray(rows, dtype=np.float64) / float(units)).astype(np.float64)


def weighted_average_scores(score_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if score_matrix.ndim != 2:
        raise ValueError(f"score_matrix must be 2D, got shape {score_matrix.shape}")
    if score_matrix.shape[1] != weights.shape[0]:
        raise ValueError(f"score_matrix has {score_matrix.shape[1]} columns but got {weights.shape[0]} weights")
    if not np.isfinite(score_matrix).all() or not np.isfinite(weights).all():
        raise FloatingPointError("weighted average received non-finite values")
    return score_matrix @ weights


@dataclass
class WeightedFusionSelection:
    weights: np.ndarray
    variants: tuple[str, ...]
    score_mode: str
    selection_metric: str
    stack_metrics: dict[str, Any]

    def predict_score(self, score_matrix: np.ndarray) -> np.ndarray:
        return weighted_average_scores(score_matrix, self.weights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.astype(float).tolist(),
            "variants": list(self.variants),
            "score_mode": self.score_mode,
            "selection_metric": self.selection_metric,
            "stack_metrics": self.stack_metrics,
        }


def select_weighted_average_on_stack(
    stack_score_block: FusionFeatureBlock,
    *,
    step: float = 0.05,
    selection_metric: str = LOCAL_GRAPH_PART_PRIMARY_METRIC,
    scores_are_probabilities: bool = False,
) -> tuple[WeightedFusionSelection, list[dict[str, Any]]]:
    weights = simplex_weight_grid(stack_score_block.features.shape[1], step=float(step))
    rows: list[dict[str, Any]] = []
    for weight_row in weights:
        scores = weighted_average_scores(stack_score_block.features, weight_row)
        metrics = binary_metrics_from_signal_scores(
            scores,
            stack_score_block.labels,
            scores_are_probabilities=bool(scores_are_probabilities),
        )
        score, value = fusion_metric_score(metrics, selection_metric)
        rows.append(
            {
                "weights": weight_row.astype(float).tolist(),
                "selection_score": float(score),
                "selection_metric_value": float(value),
                "metrics": metrics,
            }
        )
    best = max(rows, key=lambda row: row["selection_score"])
    selection = WeightedFusionSelection(
        weights=np.asarray(best["weights"], dtype=np.float64),
        variants=tuple(stack_score_block.variants),
        score_mode="weighted_features",
        selection_metric=str(selection_metric),
        stack_metrics=best["metrics"],
    )
    return selection, rows


def shuffle_non_baseline_columns(
    feature_block: FusionFeatureBlock,
    *,
    baseline_variant: str,
    seed: int = 12345,
) -> FusionFeatureBlock:
    rng = np.random.default_rng(int(seed))
    features = np.asarray(feature_block.features, dtype=np.float64).copy()
    for column, name in enumerate(feature_block.feature_names):
        if str(name).startswith(f"{baseline_variant}__"):
            continue
        features[:, column] = features[rng.permutation(features.shape[0]), column]
    return FusionFeatureBlock(
        features=features,
        labels=feature_block.labels.copy(),
        feature_names=tuple(f"row_shuffled__{name}" if not str(name).startswith(f"{baseline_variant}__") else name for name in feature_block.feature_names),
        variants=tuple(feature_block.variants),
        split=feature_block.split,
    )


def shuffled_labels(labels: np.ndarray, *, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    labels = np.asarray(labels, dtype=np.int64).copy()
    return labels[rng.permutation(labels.shape[0])]


def write_prediction_manifest(
    path: str | Path,
    *,
    blocks: Sequence[LocalGraphPredictionBlock],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "step": LOCAL_GRAPH_SCORE_FUSION_STEP,
        "contract": LOCAL_GRAPH_SCORE_FUSION_CONTRACT,
        "blocks": [block.to_manifest_row() for block in blocks],
    }
    if extra:
        manifest.update(dict(extra))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


__all__ = [
    "LOCAL_GRAPH_SCORE_FUSION_CONTRACT",
    "LOCAL_GRAPH_SCORE_FUSION_DEFAULT_C_GRID",
    "LOCAL_GRAPH_SCORE_FUSION_FEATURE_MODES",
    "LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT",
    "LOCAL_GRAPH_SCORE_FUSION_SCORE_MODES",
    "LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT",
    "LOCAL_GRAPH_SCORE_FUSION_STEP",
    "BinaryLogisticStacker",
    "FusionFeatureBlock",
    "LocalGraphPredictionBlock",
    "WeightedFusionSelection",
    "binary_logits_from_log_odds",
    "binary_margin_from_logits",
    "binary_metrics_from_signal_scores",
    "build_disagreement_feature_block",
    "build_score_feature_block",
    "fit_binary_logistic_stacker",
    "fit_binary_logistic_stackers_selecting_c",
    "fusion_metric_score",
    "load_blocks_for_split",
    "load_prediction_block",
    "logit_np",
    "lookup_fusion_metric",
    "percentile_ranks_from_reference",
    "prediction_cache_paths",
    "sanitize_binary_logits",
    "save_prediction_block",
    "score_vector_for_block",
    "select_weighted_average_on_stack",
    "shuffle_non_baseline_columns",
    "shuffled_labels",
    "sigmoid_np",
    "simplex_weight_grid",
    "softmax_np",
    "standardize_features",
    "validate_prediction_alignment",
    "weighted_average_scores",
    "write_prediction_manifest",
]


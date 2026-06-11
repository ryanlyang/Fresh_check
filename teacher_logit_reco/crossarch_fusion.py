"""Fresh cross-architecture fusion features, fusers, controls, and audits.

The Step 7 path only builds aligned feature metadata from frozen prediction
blocks.  The Step 8/9 path fits fusers, runs negative controls, and writes
source-alignment, group-size, split-leakage, and final-test guardrail audits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    fit_stacker_selecting_c_on_val,
    metrics_from_probs,
    save_stacker,
)

from .crossarch_experiment import (
    EXPERIMENT_NAME,
    FusionGroupSpec,
    build_fusion_groups,
)


EXPERIMENT_STEP = "crossarch_step7_fusion_feature_builder"
FUSER_EXPERIMENT_STEP = "crossarch_step8_fusers"
FEATURE_MODES = ("logits", "probs", "logits_probs")
UNCERTAINTY_FEATURE_MODE = "uncertainty"
LOGITS_PROBS_UNCERTAINTY_FEATURE_MODE = "logits_probs_uncertainty"
ALL_FEATURE_MODES = FEATURE_MODES + (UNCERTAINTY_FEATURE_MODE, LOGITS_PROBS_UNCERTAINTY_FEATURE_MODE)
DEFAULT_BIN_SCORE_NAMES = ("anchor_entropy", "anchor_margin", "disagreement_fraction")
DEFAULT_CONTROL_FEATURE_MODES = ("logits", "probs", "logits_probs", LOGITS_PROBS_UNCERTAINTY_FEATURE_MODE)
CONTROL_WARNING_MIN_ACCURACY = 0.20
CONTROL_WARNING_CHANCE_MARGIN = 0.10
DEFAULT_CROSSARCH_FUSERS = (
    "mean_logits",
    "mean_probs",
    "logistic_logits",
    "logistic_probs",
    "logistic_logits_probs",
    "uncertainty_logistic_logits_probs",
    "entropy_bin_gated_logistic",
    "margin_bin_gated_logistic",
    "multiplicity_bin_gated_logistic",
    "disagreement_bin_gated_logistic",
    "predicted_class_bin_gated_logistic",
)
LOGISTIC_FUSER_FEATURE_MODES = {
    "logistic_logits": "logits",
    "logistic_probs": "probs",
    "logistic_logits_probs": "logits_probs",
    "uncertainty_logistic_logits_probs": "logits_probs_uncertainty",
}
BIN_GATED_FUSER_FEATURE_MODE = LOGITS_PROBS_UNCERTAINTY_FEATURE_MODE


@dataclass(frozen=True)
class NamedFeatureMatrix:
    """A feature matrix with stable column names."""

    values: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Feature matrix must be 2D, got shape {values.shape}")
        if values.shape[1] != len(self.names):
            raise ValueError(f"Feature matrix has {values.shape[1]} columns but {len(self.names)} names")
        if not np.isfinite(values).all():
            raise FloatingPointError("Feature matrix contains non-finite values")
        object.__setattr__(self, "values", values)

    def summary(self) -> dict[str, Any]:
        return {
            "shape": [int(dim) for dim in self.values.shape],
            "n_features": int(self.values.shape[1]),
            "feature_names": list(self.names),
        }


@dataclass
class CrossArchSplitFeatureSet:
    """Aligned feature matrices and bin metadata for one group/split."""

    group_name: str
    split: str
    model_names: tuple[str, ...]
    labels: np.ndarray
    feature_matrices: dict[str, NamedFeatureMatrix]
    bin_scores: dict[str, np.ndarray]
    anchor_model_name: str
    jet_identity_hash: str | None = None
    label_hash: str | None = None
    bin_assignments: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "group_name": self.group_name,
            "split": self.split,
            "model_names": list(self.model_names),
            "n_models": len(self.model_names),
            "n_jets": int(len(self.labels)),
            "anchor_model_name": self.anchor_model_name,
            "jet_identity_hash": self.jet_identity_hash,
            "label_hash": self.label_hash,
            "features": {name: matrix.summary() for name, matrix in self.feature_matrices.items()},
            "bin_scores": {
                name: {
                    "shape": [int(dim) for dim in values.shape],
                    "min": float(np.min(values)) if values.size else None,
                    "max": float(np.max(values)) if values.size else None,
                    "mean": float(np.mean(values)) if values.size else None,
                }
                for name, values in self.bin_scores.items()
            },
            "bin_assignments": {
                name: {
                    "shape": [int(dim) for dim in values.shape],
                    "unique_bins": sorted(int(value) for value in np.unique(values)),
                }
                for name, values in self.bin_assignments.items()
            },
        }


@dataclass(frozen=True)
class CrossArchFusionFeatureBuildConfig:
    """Configuration for Step 7 feature-matrix construction."""

    prediction_dir: str
    output_dir: str
    groups: Mapping[str, Sequence[str]]
    splits: Sequence[str] = tuple(STACK_SPLITS)
    feature_modes: Sequence[str] = ALL_FEATURE_MODES
    include_optional_groups: bool = False
    write_feature_matrices: bool = False
    confirm_final_test: bool = False
    anchor_model_name: str | None = None
    quantile_bins: int = 3

    def __post_init__(self) -> None:
        if int(self.quantile_bins) < 2:
            raise ValueError("quantile_bins must be at least 2")
        unknown_splits = sorted(set(self.splits) - set(STACK_SPLITS))
        if unknown_splits:
            raise ValueError(f"Unknown splits: {unknown_splits}")
        if "final_test" in self.splits and not self.confirm_final_test:
            raise ValueError("Refusing to touch final_test without confirm_final_test=True")
        unknown_modes = sorted(set(self.feature_modes) - set(ALL_FEATURE_MODES))
        if unknown_modes:
            raise ValueError(f"Unknown feature modes: {unknown_modes}")


@dataclass(frozen=True)
class CrossArchFusionFitConfig(CrossArchFusionFeatureBuildConfig):
    """Configuration for Step 8 F0-F3 fuser fitting."""

    fusers: Sequence[str] = DEFAULT_CROSSARCH_FUSERS
    c_grid: Sequence[float] = tuple(DEFAULT_C_GRID)
    max_iter: int = 2000
    min_bin_train_rows: int = 2
    run_controls: bool = True
    control_seed: int = 12345
    control_feature_modes: Sequence[str] = DEFAULT_CONTROL_FEATURE_MODES
    control_warning_min_accuracy: float = CONTROL_WARNING_MIN_ACCURACY
    control_warning_chance_margin: float = CONTROL_WARNING_CHANCE_MARGIN

    def __post_init__(self) -> None:
        super().__post_init__()
        unknown_fusers = sorted(set(self.fusers) - set(DEFAULT_CROSSARCH_FUSERS))
        if unknown_fusers:
            raise ValueError(f"Unknown crossarch fusers: {unknown_fusers}")
        if not self.fusers:
            raise ValueError("At least one fuser is required")
        if tuple(self.splits) != tuple(STACK_SPLITS):
            raise ValueError(f"Step 8 fusers require exactly {tuple(STACK_SPLITS)}")
        if any(float(value) <= 0.0 for value in self.c_grid):
            raise ValueError("c_grid values must be positive")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if int(self.min_bin_train_rows) <= 0:
            raise ValueError("min_bin_train_rows must be positive")
        unknown_control_modes = sorted(set(self.control_feature_modes) - set(ALL_FEATURE_MODES))
        if unknown_control_modes:
            raise ValueError(f"Unknown control feature modes: {unknown_control_modes}")
        if self.run_controls and not self.control_feature_modes:
            raise ValueError("control_feature_modes cannot be empty when run_controls=True")
        if int(self.control_seed) < 0:
            raise ValueError("control_seed must be non-negative")
        if float(self.control_warning_min_accuracy) < 0.0:
            raise ValueError("control_warning_min_accuracy must be non-negative")
        if float(self.control_warning_chance_margin) < 0.0:
            raise ValueError("control_warning_chance_margin must be non-negative")


def labels_hash(labels: np.ndarray) -> str:
    arr = np.asarray(labels, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def entropy_from_probs(probs: np.ndarray, *, normalized: bool = True) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    clipped = np.clip(probs, 1.0e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    if normalized:
        entropy = entropy / np.log(float(probs.shape[1]))
    return entropy.astype(np.float32)


def top1_margin_from_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[1] < 2:
        return np.ones((probs.shape[0],), dtype=np.float32)
    top2 = np.partition(probs, kth=-2, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).astype(np.float32)


def _raw_feature_names(blocks: Sequence[PredictionBlock], *, feature_mode: str) -> tuple[str, ...]:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown raw feature mode {feature_mode!r}")
    names: list[str] = []
    for block in blocks:
        num_classes = int(block.logits.shape[1])
        if feature_mode in ("logits", "logits_probs"):
            names.extend(f"logit::{block.model_name}::class_{idx}" for idx in range(num_classes))
        if feature_mode in ("probs", "logits_probs"):
            names.extend(f"prob::{block.model_name}::class_{idx}" for idx in range(num_classes))
    return tuple(names)


def raw_feature_matrix(blocks: Sequence[PredictionBlock], *, feature_mode: str) -> NamedFeatureMatrix:
    return NamedFeatureMatrix(
        values=stack_feature_matrix(blocks, feature_mode=feature_mode),
        names=_raw_feature_names(blocks, feature_mode=feature_mode),
    )


def predicted_classes(blocks: Sequence[PredictionBlock]) -> np.ndarray:
    validate_prediction_alignment(blocks)
    return np.stack([np.argmax(block.probs, axis=1).astype(np.int64) for block in blocks], axis=1)


def pairwise_disagreement_fraction(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.int64)
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")
    n_models = predictions.shape[1]
    if n_models < 2:
        return np.zeros((predictions.shape[0],), dtype=np.float32)
    total_pairs = n_models * (n_models - 1) // 2
    disagreements = np.zeros((predictions.shape[0],), dtype=np.float64)
    for left in range(n_models):
        for right in range(left + 1, n_models):
            disagreements += predictions[:, left] != predictions[:, right]
    return (disagreements / float(total_pairs)).astype(np.float32)


def distinct_predicted_class_count(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.int64)
    return np.asarray([len(set(row.tolist())) for row in predictions], dtype=np.float32)


def build_uncertainty_feature_matrix(blocks: Sequence[PredictionBlock]) -> NamedFeatureMatrix:
    """Build row-wise uncertainty/diversity features from aligned blocks."""

    validate_prediction_alignment(blocks)
    model_names = [block.model_name for block in blocks]
    logits_stack = np.stack([block.logits.astype(np.float32) for block in blocks], axis=1)
    probs_stack = np.stack([block.probs.astype(np.float32) for block in blocks], axis=1)
    entropy = np.stack([entropy_from_probs(block.probs) for block in blocks], axis=1)
    max_prob = np.stack([np.max(block.probs, axis=1).astype(np.float32) for block in blocks], axis=1)
    margin = np.stack([top1_margin_from_probs(block.probs) for block in blocks], axis=1)
    preds = predicted_classes(blocks)

    columns: list[np.ndarray] = []
    names: list[str] = []
    for index, model_name in enumerate(model_names):
        columns.extend([entropy[:, index], max_prob[:, index], margin[:, index]])
        names.extend(
            [
                f"entropy::{model_name}",
                f"max_prob::{model_name}",
                f"top1_margin::{model_name}",
            ]
        )

    mean_probs = np.mean(probs_stack, axis=1)
    ensemble_entropy = entropy_from_probs(mean_probs)
    ensemble_margin = top1_margin_from_probs(mean_probs)
    classwise_mean_logits = np.mean(logits_stack, axis=1)
    classwise_std_logits = np.std(logits_stack, axis=1)
    classwise_mean_probs = np.mean(probs_stack, axis=1)
    classwise_std_probs = np.std(probs_stack, axis=1)
    aggregate_columns = [
        np.mean(entropy, axis=1),
        np.std(entropy, axis=1),
        np.mean(max_prob, axis=1),
        np.std(max_prob, axis=1),
        np.mean(margin, axis=1),
        np.std(margin, axis=1),
        ensemble_entropy,
        ensemble_margin,
        distinct_predicted_class_count(preds),
        pairwise_disagreement_fraction(preds),
    ]
    aggregate_names = [
        "mean_model_entropy",
        "std_model_entropy",
        "mean_model_max_prob",
        "std_model_max_prob",
        "mean_model_margin",
        "std_model_margin",
        "ensemble_entropy_mean_prob",
        "ensemble_margin_mean_prob",
        "distinct_predicted_class_count",
        "pairwise_disagreement_fraction",
    ]
    columns.extend(aggregate_columns)
    names.extend(aggregate_names)

    num_classes = int(blocks[0].logits.shape[1])
    for class_index in range(num_classes):
        columns.extend(
            [
                classwise_mean_logits[:, class_index],
                classwise_std_logits[:, class_index],
                classwise_mean_probs[:, class_index],
                classwise_std_probs[:, class_index],
            ]
        )
        names.extend(
            [
                f"mean_logit::class_{class_index}",
                f"std_logit::class_{class_index}",
                f"mean_prob::class_{class_index}",
                f"std_prob::class_{class_index}",
            ]
        )

    values = np.stack(columns, axis=1).astype(np.float32)
    return NamedFeatureMatrix(values=values, names=tuple(names))


def _select_anchor_block(blocks: Sequence[PredictionBlock], anchor_model_name: str | None) -> PredictionBlock:
    if anchor_model_name is None:
        return blocks[0]
    for block in blocks:
        if block.model_name == anchor_model_name:
            return block
    return blocks[0]


def build_bin_scores(
    blocks: Sequence[PredictionBlock],
    *,
    anchor_model_name: str | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    validate_prediction_alignment(blocks)
    anchor = _select_anchor_block(blocks, anchor_model_name)
    preds = predicted_classes(blocks)
    scores = {
        "anchor_entropy": entropy_from_probs(anchor.probs),
        "anchor_margin": top1_margin_from_probs(anchor.probs),
        "disagreement_fraction": pairwise_disagreement_fraction(preds),
        "distinct_predicted_class_count": distinct_predicted_class_count(preds),
        "anchor_predicted_class": np.argmax(anchor.probs, axis=1).astype(np.int64),
    }
    return scores, anchor.model_name


def quantile_edges(values: np.ndarray, *, n_bins: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("quantile_edges expects a 1D array")
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.unique(edges.astype(np.float64))


def assign_quantile_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    return np.searchsorted(edges, values, side="right").astype(np.int16)


def fit_bin_specs(
    train_scores: Mapping[str, np.ndarray],
    *,
    n_bins: int = 3,
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for score_name in DEFAULT_BIN_SCORE_NAMES:
        specs[f"{score_name}_quantile{int(n_bins)}"] = {
            "source_score": score_name,
            "kind": "train_quantile",
            "n_bins_requested": int(n_bins),
            "edges": quantile_edges(train_scores[score_name], n_bins=int(n_bins)).tolist(),
            "reference_split": "stack_train",
        }
    specs["anchor_predicted_class"] = {
        "source_score": "anchor_predicted_class",
        "kind": "integer_class_id",
        "reference_split": "per_split_anchor_prediction",
    }
    return specs


def assign_bins_from_specs(
    scores: Mapping[str, np.ndarray],
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    assignments: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        score_name = str(spec["source_score"])
        if spec["kind"] == "train_quantile":
            assignments[name] = assign_quantile_bins(scores[score_name], np.asarray(spec["edges"], dtype=np.float64))
        elif spec["kind"] == "integer_class_id":
            assignments[name] = np.asarray(scores[score_name], dtype=np.int16)
        else:
            raise ValueError(f"Unknown bin spec kind {spec['kind']!r}")
    return assignments


def build_split_feature_set(
    prediction_dir: str | Path,
    *,
    group_name: str,
    model_names: Sequence[str],
    split: str,
    feature_modes: Sequence[str] = ALL_FEATURE_MODES,
    anchor_model_name: str | None = None,
    verify_hash: bool = True,
) -> CrossArchSplitFeatureSet:
    blocks = load_blocks_for_split(prediction_dir, model_names, split)
    if not verify_hash:
        # load_blocks_for_split verifies hashes internally. This branch exists
        # for future compatibility without widening the public function shape.
        validate_prediction_alignment(blocks)
    return build_split_feature_set_from_blocks(
        blocks,
        group_name=group_name,
        split=split,
        feature_modes=feature_modes,
        anchor_model_name=anchor_model_name,
    )


def build_split_feature_set_from_blocks(
    blocks: Sequence[PredictionBlock],
    *,
    group_name: str,
    split: str,
    feature_modes: Sequence[str] = ALL_FEATURE_MODES,
    anchor_model_name: str | None = None,
) -> CrossArchSplitFeatureSet:
    validate_prediction_alignment(blocks)
    feature_matrices: dict[str, NamedFeatureMatrix] = {}
    for mode in feature_modes:
        if mode in FEATURE_MODES:
            feature_matrices[mode] = raw_feature_matrix(blocks, feature_mode=mode)
    if UNCERTAINTY_FEATURE_MODE in feature_modes or LOGITS_PROBS_UNCERTAINTY_FEATURE_MODE in feature_modes:
        uncertainty = build_uncertainty_feature_matrix(blocks)
        if UNCERTAINTY_FEATURE_MODE in feature_modes:
            feature_matrices[UNCERTAINTY_FEATURE_MODE] = uncertainty
        if LOGITS_PROBS_UNCERTAINTY_FEATURE_MODE in feature_modes:
            raw = feature_matrices.get("logits_probs") or raw_feature_matrix(blocks, feature_mode="logits_probs")
            feature_matrices[LOGITS_PROBS_UNCERTAINTY_FEATURE_MODE] = NamedFeatureMatrix(
                values=np.concatenate([raw.values, uncertainty.values], axis=1),
                names=raw.names + uncertainty.names,
            )
    bin_scores, selected_anchor = build_bin_scores(blocks, anchor_model_name=anchor_model_name)
    first = blocks[0]
    return CrossArchSplitFeatureSet(
        group_name=group_name,
        split=split,
        model_names=tuple(block.model_name for block in blocks),
        labels=first.labels.astype(np.int64, copy=False),
        feature_matrices=feature_matrices,
        bin_scores=bin_scores,
        anchor_model_name=selected_anchor,
        jet_identity_hash=first.metadata.get("jet_identity_hash"),
        label_hash=labels_hash(first.labels),
    )


def cross_split_identity_overlap(split_sets: Mapping[str, CrossArchSplitFeatureSet]) -> dict[str, int]:
    # The compact prediction block format stores full JetIdentity objects only
    # during load, not in CrossArchSplitFeatureSet.  For Step 7 reports we rely
    # on per-split identity hashes and leave exact overlap checks to Step 9.
    splits = list(split_sets)
    return {f"{a}__{b}": 0 for idx, a in enumerate(splits) for b in splits[idx + 1 :]}


def save_feature_set(
    feature_set: CrossArchSplitFeatureSet,
    output_dir: str | Path,
    *,
    write_matrices: bool,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    group_dir = output_dir / "features" / feature_set.group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = group_dir / f"{feature_set.split}_feature_matrices.npz"
    metadata_path = group_dir / f"{feature_set.split}_feature_metadata.json"
    summary = feature_set.summary()
    summary["matrix_path"] = str(matrix_path) if write_matrices else None
    summary["metadata_path"] = str(metadata_path)
    if write_matrices:
        arrays: dict[str, np.ndarray] = {
            "labels": feature_set.labels.astype(np.int64, copy=False),
        }
        for mode, matrix in feature_set.feature_matrices.items():
            arrays[f"features__{mode}"] = matrix.values
        for name, values in feature_set.bin_scores.items():
            arrays[f"bin_score__{name}"] = np.asarray(values)
        for name, values in feature_set.bin_assignments.items():
            arrays[f"bin_assignment__{name}"] = np.asarray(values)
        np.savez_compressed(matrix_path, **arrays)
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_group_feature_report(
    prediction_dir: str | Path,
    *,
    group_name: str,
    model_names: Sequence[str],
    splits: Sequence[str] = STACK_SPLITS,
    feature_modes: Sequence[str] = ALL_FEATURE_MODES,
    output_dir: str | Path | None = None,
    write_matrices: bool = False,
    anchor_model_name: str | None = None,
    quantile_bins: int = 3,
) -> dict[str, Any]:
    if "stack_train" not in splits:
        raise ValueError("Step 7 bin specs require stack_train in splits")
    split_reports: dict[str, Any] = {}
    train_set = build_split_feature_set(
        prediction_dir,
        group_name=group_name,
        model_names=model_names,
        split="stack_train",
        feature_modes=feature_modes,
        anchor_model_name=anchor_model_name,
    )
    bin_specs = fit_bin_specs(train_set.bin_scores, n_bins=quantile_bins)
    train_set.bin_assignments = assign_bins_from_specs(train_set.bin_scores, bin_specs)
    if output_dir is not None:
        split_reports["stack_train"] = save_feature_set(train_set, output_dir, write_matrices=write_matrices)
    else:
        split_reports["stack_train"] = train_set.summary()

    for split in splits:
        if split == "stack_train":
            continue
        split_set = build_split_feature_set(
            prediction_dir,
            group_name=group_name,
            model_names=model_names,
            split=split,
            feature_modes=feature_modes,
            anchor_model_name=anchor_model_name,
        )
        split_set.bin_assignments = assign_bins_from_specs(split_set.bin_scores, bin_specs)
        if output_dir is not None:
            split_reports[split] = save_feature_set(split_set, output_dir, write_matrices=write_matrices)
        else:
            split_reports[split] = split_set.summary()

    return {
        "group_name": group_name,
        "model_names": list(model_names),
        "n_models": len(model_names),
        "splits": split_reports,
        "bin_specs": bin_specs,
        "cross_split_identity_overlap_counts": cross_split_identity_overlap({}),
    }


def _groups_from_config(config: CrossArchFusionFeatureBuildConfig) -> dict[str, list[str]]:
    return {str(name): [str(model) for model in models] for name, models in config.groups.items()}


def run_crossarch_feature_builder(config: CrossArchFusionFeatureBuildConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _groups_from_config(config)
    config_payload = {
        **asdict(config),
        "groups": groups,
        "feature_modes": list(config.feature_modes),
        "splits": list(config.splits),
    }
    (output_dir / "feature_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    group_reports = {}
    for group_name, model_names in groups.items():
        group_reports[group_name] = build_group_feature_report(
            config.prediction_dir,
            group_name=group_name,
            model_names=model_names,
            splits=config.splits,
            feature_modes=config.feature_modes,
            output_dir=output_dir,
            write_matrices=config.write_feature_matrices,
            anchor_model_name=config.anchor_model_name,
            quantile_bins=config.quantile_bins,
        )

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": EXPERIMENT_STEP,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_dir": str(config.prediction_dir),
        "output_dir": str(output_dir),
        "feature_modes": list(config.feature_modes),
        "splits": list(config.splits),
        "write_feature_matrices": bool(config.write_feature_matrices),
        "groups": group_reports,
        "leakage_rules": {
            "inputs": "frozen prediction blocks only",
            "no_model_checkpoints_loaded": True,
            "no_training_data_loaded": True,
            "no_fuser_fit_in_step7": True,
            "final_test_guarded_by_confirm_final_test": True,
        },
    }
    (output_dir / "feature_build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def mean_logits_probs(blocks: Sequence[PredictionBlock]) -> np.ndarray:
    validate_prediction_alignment(blocks)
    logits = np.mean(np.stack([block.logits.astype(np.float32) for block in blocks], axis=0), axis=0)
    return softmax_np(logits)


def mean_probs_probs(blocks: Sequence[PredictionBlock]) -> np.ndarray:
    validate_prediction_alignment(blocks)
    probs = np.mean(np.stack([block.probs.astype(np.float32) for block in blocks], axis=0), axis=0)
    probs = np.clip(probs, 1.0e-12, None)
    return (probs / np.sum(probs, axis=1, keepdims=True)).astype(np.float32)


def load_group_fusion_data(
    prediction_dir: str | Path,
    *,
    group_name: str,
    model_names: Sequence[str],
    splits: Sequence[str] = STACK_SPLITS,
    feature_modes: Sequence[str] = ALL_FEATURE_MODES,
    anchor_model_name: str | None = None,
    quantile_bins: int = 3,
) -> tuple[dict[str, list[PredictionBlock]], dict[str, CrossArchSplitFeatureSet], dict[str, dict[str, Any]]]:
    blocks_by_split: dict[str, list[PredictionBlock]] = {}
    split_sets: dict[str, CrossArchSplitFeatureSet] = {}
    for split in splits:
        blocks = load_blocks_for_split(prediction_dir, model_names, split)
        blocks_by_split[split] = blocks
        split_sets[split] = build_split_feature_set_from_blocks(
            blocks,
            group_name=group_name,
            split=split,
            feature_modes=feature_modes,
            anchor_model_name=anchor_model_name,
        )
    bin_specs = fit_bin_specs(split_sets["stack_train"].bin_scores, n_bins=quantile_bins)
    for split_set in split_sets.values():
        split_set.bin_assignments = assign_bins_from_specs(split_set.bin_scores, bin_specs)
    return blocks_by_split, split_sets, bin_specs


def evaluate_mean_fuser(
    blocks_by_split: Mapping[str, Sequence[PredictionBlock]],
    *,
    fuser_name: str,
) -> dict[str, Any]:
    if fuser_name == "mean_logits":
        predictor = mean_logits_probs
        method = "mean_logits_then_softmax"
    elif fuser_name == "mean_probs":
        predictor = mean_probs_probs
        method = "mean_probabilities"
    else:
        raise ValueError(f"Unknown mean fuser {fuser_name!r}")
    metrics = {}
    for split, blocks in blocks_by_split.items():
        metrics[split] = metrics_from_probs(predictor(blocks), blocks[0].labels)
    return {
        "method": method,
        "fit_split": None,
        "selection_split": None,
        "metrics": metrics,
    }


def fit_global_logistic_fuser(
    split_sets: Mapping[str, CrossArchSplitFeatureSet],
    *,
    fuser_name: str,
    feature_mode: str,
    c_grid: Sequence[float],
    max_iter: int,
    stacker_dir: str | Path | None = None,
) -> dict[str, Any]:
    train_set = split_sets["stack_train"]
    val_set = split_sets["stack_val"]
    train_x = train_set.feature_matrices[feature_mode].values
    val_x = val_set.feature_matrices[feature_mode].values
    train_y = train_set.labels
    val_y = val_set.labels
    num_classes = int(train_set.feature_matrices["logits"].values.shape[1] // max(len(train_set.model_names), 1))
    stacker, selection = fit_stacker_selecting_c_on_val(
        train_x,
        train_y,
        val_x,
        val_y,
        c_grid=c_grid,
        max_iter=max_iter,
        feature_mode=feature_mode,
        model_names=train_set.model_names,
        num_classes=num_classes,
    )
    if stacker_dir is not None:
        save_stacker(stacker, Path(stacker_dir) / f"group__{train_set.group_name}__{fuser_name}.npz")
    metrics = {}
    for split, split_set in split_sets.items():
        features = split_set.feature_matrices[feature_mode].values
        metrics[split] = metrics_from_probs(stacker.predict_probs(features), split_set.labels)
    return {
        "method": "regularized_multinomial_logistic_regression",
        "feature_mode": feature_mode,
        "fit_split": "stack_train",
        "selection_split": "stack_val",
        "selection": selection,
        "metrics": metrics,
        "weight_matrix_shape": list(stacker.coef.shape),
    }


def bin_assignment_name_for_fuser(fuser_name: str, *, quantile_bins: int = 3) -> str:
    if fuser_name == "entropy_bin_gated_logistic":
        return f"anchor_entropy_quantile{int(quantile_bins)}"
    if fuser_name == "margin_bin_gated_logistic":
        return f"anchor_margin_quantile{int(quantile_bins)}"
    if fuser_name == "disagreement_bin_gated_logistic":
        return f"disagreement_fraction_quantile{int(quantile_bins)}"
    if fuser_name == "predicted_class_bin_gated_logistic":
        return "anchor_predicted_class"
    if fuser_name == "multiplicity_bin_gated_logistic":
        return f"hlt_valid_constituent_count_quantile{int(quantile_bins)}"
    raise ValueError(f"Unknown bin-gated fuser {fuser_name!r}")


def _predict_with_bin_stackers(
    *,
    split_set: CrossArchSplitFeatureSet,
    feature_mode: str,
    bin_assignment_name: str,
    fallback_stacker,
    bin_stackers: Mapping[int, Any],
) -> np.ndarray:
    features = split_set.feature_matrices[feature_mode].values
    probs = fallback_stacker.predict_probs(features)
    assignments = split_set.bin_assignments[bin_assignment_name].astype(np.int64)
    for bin_id, stacker in bin_stackers.items():
        if stacker is None:
            continue
        mask = assignments == int(bin_id)
        if np.any(mask):
            probs[mask] = stacker.predict_probs(features[mask])
    return probs


def fit_bin_gated_logistic_fuser(
    split_sets: Mapping[str, CrossArchSplitFeatureSet],
    *,
    fuser_name: str,
    c_grid: Sequence[float],
    max_iter: int,
    quantile_bins: int = 3,
    min_bin_train_rows: int = 2,
    stacker_dir: str | Path | None = None,
) -> dict[str, Any]:
    feature_mode = BIN_GATED_FUSER_FEATURE_MODE
    bin_assignment_name = bin_assignment_name_for_fuser(fuser_name, quantile_bins=quantile_bins)
    train_set = split_sets["stack_train"]
    if bin_assignment_name not in train_set.bin_assignments:
        return {
            "method": "bin_gated_logistic_regression",
            "status": "skipped",
            "reason": (
                f"Bin assignment {bin_assignment_name!r} is unavailable. "
                "Current prediction blocks do not carry row-wise HLT constituent multiplicity."
            ),
            "feature_mode": feature_mode,
            "bin_assignment_name": bin_assignment_name,
            "metrics": {},
        }

    fallback = fit_global_logistic_fuser(
        split_sets,
        fuser_name=f"{fuser_name}__fallback",
        feature_mode=feature_mode,
        c_grid=c_grid,
        max_iter=max_iter,
        stacker_dir=None,
    )
    train_x = train_set.feature_matrices[feature_mode].values
    train_y = train_set.labels
    val_set = split_sets["stack_val"]
    val_x = val_set.feature_matrices[feature_mode].values
    val_y = val_set.labels
    num_classes = int(train_set.feature_matrices["logits"].values.shape[1] // max(len(train_set.model_names), 1))

    fallback_stacker, fallback_selection = fit_stacker_selecting_c_on_val(
        train_x,
        train_y,
        val_x,
        val_y,
        c_grid=c_grid,
        max_iter=max_iter,
        feature_mode=feature_mode,
        model_names=train_set.model_names,
        num_classes=num_classes,
    )
    if stacker_dir is not None:
        save_stacker(fallback_stacker, Path(stacker_dir) / f"group__{train_set.group_name}__{fuser_name}__fallback.npz")

    all_bin_ids = sorted({
        int(value)
        for split_set in split_sets.values()
        for value in np.unique(split_set.bin_assignments[bin_assignment_name])
    })
    bin_stackers: dict[int, Any] = {}
    bin_reports: dict[str, Any] = {}
    for bin_id in all_bin_ids:
        train_mask = train_set.bin_assignments[bin_assignment_name] == int(bin_id)
        val_mask = val_set.bin_assignments[bin_assignment_name] == int(bin_id)
        train_count = int(np.sum(train_mask))
        val_count = int(np.sum(val_mask))
        if train_count < int(min_bin_train_rows) or val_count == 0:
            bin_stackers[bin_id] = None
            bin_reports[str(bin_id)] = {
                "status": "fallback",
                "train_rows": train_count,
                "stack_val_rows": val_count,
                "reason": "too few train or stack_val rows for a separate bin stacker",
            }
            continue
        stacker, selection = fit_stacker_selecting_c_on_val(
            train_x[train_mask],
            train_y[train_mask],
            val_x[val_mask],
            val_y[val_mask],
            c_grid=c_grid,
            max_iter=max_iter,
            feature_mode=feature_mode,
            model_names=train_set.model_names,
            num_classes=num_classes,
        )
        bin_stackers[bin_id] = stacker
        if stacker_dir is not None:
            save_stacker(stacker, Path(stacker_dir) / f"group__{train_set.group_name}__{fuser_name}__bin_{bin_id}.npz")
        bin_reports[str(bin_id)] = {
            "status": "fit",
            "train_rows": train_count,
            "stack_val_rows": val_count,
            "selection": selection,
            "weight_matrix_shape": list(stacker.coef.shape),
        }

    metrics = {}
    bin_populations = {}
    for split, split_set in split_sets.items():
        probs = _predict_with_bin_stackers(
            split_set=split_set,
            feature_mode=feature_mode,
            bin_assignment_name=bin_assignment_name,
            fallback_stacker=fallback_stacker,
            bin_stackers=bin_stackers,
        )
        metrics[split] = metrics_from_probs(probs, split_set.labels)
        assignments = split_set.bin_assignments[bin_assignment_name]
        bin_populations[split] = {
            str(int(value)): int(np.sum(assignments == value))
            for value in sorted(np.unique(assignments))
        }

    return {
        "method": "bin_gated_logistic_regression",
        "status": "ok",
        "feature_mode": feature_mode,
        "bin_assignment_name": bin_assignment_name,
        "fit_split": "stack_train",
        "selection_split": "stack_val",
        "fallback_selection": fallback_selection,
        "fallback_global_metrics": fallback["metrics"],
        "bin_reports": bin_reports,
        "bin_populations": bin_populations,
        "metrics": metrics,
    }


def _num_classes_from_split_set(split_set: CrossArchSplitFeatureSet) -> int:
    return int(split_set.feature_matrices["logits"].values.shape[1] // max(len(split_set.model_names), 1))


def _stable_control_seed(seed: int, group_name: str) -> int:
    digest = hashlib.sha256(str(group_name).encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return int((int(seed) + offset) % (2**31 - 1))


def _column_shuffle(features: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    shuffled = np.asarray(features).copy()
    for column in range(shuffled.shape[1]):
        shuffled[:, column] = shuffled[rng.permutation(shuffled.shape[0]), column]
    return shuffled


def run_group_negative_controls(
    split_sets: Mapping[str, CrossArchSplitFeatureSet],
    *,
    feature_modes: Sequence[str] = DEFAULT_CONTROL_FEATURE_MODES,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    max_iter: int = 2000,
    seed: int = 12345,
    warning_min_accuracy: float = CONTROL_WARNING_MIN_ACCURACY,
    warning_chance_margin: float = CONTROL_WARNING_CHANCE_MARGIN,
) -> dict[str, Any]:
    """Fit label-permutation and row-shuffled controls for one aligned group."""

    train_set = split_sets["stack_train"]
    val_set = split_sets["stack_val"]
    test_set = split_sets["final_test"]
    num_classes = _num_classes_from_split_set(train_set)
    chance_accuracy = 1.0 / float(max(num_classes, 1))
    warning_threshold = max(float(warning_min_accuracy), chance_accuracy + float(warning_chance_margin))
    rng = np.random.RandomState(_stable_control_seed(seed, train_set.group_name))
    mode_reports: dict[str, Any] = {}
    flags: list[dict[str, Any]] = []

    for mode in feature_modes:
        if mode not in train_set.feature_matrices:
            mode_reports[str(mode)] = {
                "status": "skipped",
                "reason": f"Feature mode {mode!r} is unavailable for controls",
            }
            continue

        train_x = train_set.feature_matrices[mode].values
        val_x = val_set.feature_matrices[mode].values
        test_x = test_set.feature_matrices[mode].values
        train_y = train_set.labels
        val_y = val_set.labels
        test_y = test_set.labels

        shuffled_y = train_y.copy()
        rng.shuffle(shuffled_y)
        perm_stacker, perm_selection = fit_stacker_selecting_c_on_val(
            train_x,
            shuffled_y,
            val_x,
            val_y,
            c_grid=c_grid,
            max_iter=max_iter,
            feature_mode=str(mode),
            model_names=train_set.model_names,
            num_classes=num_classes,
        )
        perm_metrics = {
            "stack_val": metrics_from_probs(perm_stacker.predict_probs(val_x), val_y),
            "final_test": metrics_from_probs(perm_stacker.predict_probs(test_x), test_y),
        }

        shuffled_train_x = _column_shuffle(train_x, rng)
        row_stacker, row_selection = fit_stacker_selecting_c_on_val(
            shuffled_train_x,
            train_y,
            val_x,
            val_y,
            c_grid=c_grid,
            max_iter=max_iter,
            feature_mode=str(mode),
            model_names=train_set.model_names,
            num_classes=num_classes,
        )
        row_metrics = {
            "stack_val": metrics_from_probs(row_stacker.predict_probs(val_x), val_y),
            "final_test": metrics_from_probs(row_stacker.predict_probs(test_x), test_y),
        }

        mode_reports[str(mode)] = {
            "status": "ok",
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

        for control_name, control_metrics in (
            ("permuted_labels", perm_metrics),
            ("row_shuffled_features", row_metrics),
        ):
            final_accuracy = control_metrics["final_test"]["accuracy"]
            if final_accuracy is not None and float(final_accuracy) > warning_threshold:
                flags.append(
                    {
                        "name": f"{control_name}_did_not_collapse",
                        "severity": "warning",
                        "group": train_set.group_name,
                        "feature_mode": str(mode),
                        "final_test_accuracy": float(final_accuracy),
                        "warning_threshold": float(warning_threshold),
                    }
                )

    return {
        "enabled": True,
        "ok": not flags,
        "seed": int(seed),
        "feature_modes": list(feature_modes),
        "chance_accuracy": float(chance_accuracy),
        "warning_threshold": float(warning_threshold),
        "mode_reports": mode_reports,
        "suspicious_flags": flags,
    }


def audit_group_prediction_blocks(
    blocks_by_split: Mapping[str, Sequence[PredictionBlock]],
    split_sets: Mapping[str, CrossArchSplitFeatureSet],
    *,
    expected_splits: Sequence[str] = STACK_SPLITS,
) -> dict[str, Any]:
    """Audit source alignment, group sizing, and split identity leakage."""

    problems: list[str] = []
    split_rows: dict[str, Any] = {}
    identity_sets: dict[str, set[str]] = {}
    model_names = tuple(next(iter(split_sets.values())).model_names) if split_sets else tuple()
    duplicate_model_names = sorted({name for name in model_names if model_names.count(name) > 1})

    for split in expected_splits:
        blocks = list(blocks_by_split.get(split, []))
        if not blocks:
            problems.append(f"{split}: no prediction blocks loaded")
            continue
        validate_prediction_alignment(blocks)
        first = blocks[0]
        identities = {identity.key() for identity in first.jet_ids}
        identity_sets[split] = identities
        per_model_rows = {block.model_name: int(len(block.labels)) for block in blocks}
        per_model_num_classes = {block.model_name: int(block.logits.shape[1]) for block in blocks}
        per_model_label_hash = {block.model_name: labels_hash(block.labels) for block in blocks}
        per_model_identity_hash = {
            block.model_name: str(block.metadata.get("jet_identity_hash", ""))
            for block in blocks
        }
        per_model_content_hash = {
            block.model_name: str(block.metadata.get("prediction_content_hash", ""))
            for block in blocks
        }
        if len(set(per_model_rows.values())) != 1:
            problems.append(f"{split}: model prediction blocks disagree on row count")
        if len(set(per_model_num_classes.values())) != 1:
            problems.append(f"{split}: model prediction blocks disagree on class count")
        if len(set(per_model_label_hash.values())) != 1:
            problems.append(f"{split}: model prediction blocks disagree on label hash")
        nonempty_identity_hashes = [value for value in per_model_identity_hash.values() if value]
        if nonempty_identity_hashes and len(set(nonempty_identity_hashes)) != 1:
            problems.append(f"{split}: model prediction blocks disagree on jet identity hash")
        split_rows[split] = {
            "n_jets": int(len(first.labels)),
            "num_classes": int(first.logits.shape[1]),
            "label_hash": labels_hash(first.labels),
            "jet_identity_hash": first.metadata.get("jet_identity_hash"),
            "per_model_n_jets": per_model_rows,
            "per_model_num_classes": per_model_num_classes,
            "per_model_label_hash": per_model_label_hash,
            "per_model_jet_identity_hash": per_model_identity_hash,
            "per_model_prediction_content_hash": per_model_content_hash,
        }

    overlap_rows: dict[str, Any] = {}
    leakage_ok = True
    split_list = list(expected_splits)
    for index, split_a in enumerate(split_list):
        for split_b in split_list[index + 1 :]:
            overlap = sorted(identity_sets.get(split_a, set()) & identity_sets.get(split_b, set()))
            key = f"{split_a}__{split_b}"
            overlap_rows[key] = {
                "count": int(len(overlap)),
                "examples": overlap[:5],
            }
            if overlap:
                leakage_ok = False
                problems.append(f"{split_a} and {split_b} overlap in {len(overlap)} jet identities")

    row_counts = {
        split: int(split_rows[split]["n_jets"])
        for split in split_rows
    }
    group_size_ok = bool(model_names) and not duplicate_model_names and all(count > 0 for count in row_counts.values())
    if not model_names:
        problems.append("group has no model names")
    if duplicate_model_names:
        problems.append(f"group has duplicate model names: {duplicate_model_names}")
    if any(count <= 0 for count in row_counts.values()):
        problems.append("one or more requested splits has no rows")

    source_alignment_ok = not any(
        "disagree" in problem or "no prediction blocks" in problem
        for problem in problems
    )
    return {
        "ok": bool(source_alignment_ok and group_size_ok and leakage_ok),
        "problems": problems,
        "source_alignment": {
            "ok": bool(source_alignment_ok),
            "splits": split_rows,
            "rule": "All model prediction blocks inside a group must share row order, labels, and jet identities per split.",
        },
        "group_size": {
            "ok": bool(group_size_ok),
            "n_models": int(len(model_names)),
            "model_names": list(model_names),
            "duplicate_model_names": duplicate_model_names,
            "rows_by_split": row_counts,
        },
        "split_leakage": {
            "ok": bool(leakage_ok),
            "cross_split_overlaps": overlap_rows,
            "rule": "stack_train, stack_val, and final_test jet identities must be disjoint.",
        },
    }


def final_test_guardrail_audit(config: CrossArchFusionFitConfig) -> dict[str, Any]:
    final_test_requested = "final_test" in set(config.splits)
    ok = (not final_test_requested) or bool(config.confirm_final_test)
    return {
        "ok": bool(ok),
        "final_test_requested": bool(final_test_requested),
        "confirm_final_test": bool(config.confirm_final_test),
        "rule": "final_test can only be evaluated when confirm_final_test=True.",
    }


def fit_group_fusers(
    prediction_dir: str | Path,
    *,
    group_name: str,
    model_names: Sequence[str],
    fusers: Sequence[str] = DEFAULT_CROSSARCH_FUSERS,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    max_iter: int = 2000,
    anchor_model_name: str | None = None,
    quantile_bins: int = 3,
    min_bin_train_rows: int = 2,
    stacker_dir: str | Path | None = None,
    run_controls: bool = True,
    control_seed: int = 12345,
    control_feature_modes: Sequence[str] = DEFAULT_CONTROL_FEATURE_MODES,
    control_warning_min_accuracy: float = CONTROL_WARNING_MIN_ACCURACY,
    control_warning_chance_margin: float = CONTROL_WARNING_CHANCE_MARGIN,
) -> dict[str, Any]:
    required_feature_modes = set(ALL_FEATURE_MODES)
    blocks_by_split, split_sets, bin_specs = load_group_fusion_data(
        prediction_dir,
        group_name=group_name,
        model_names=model_names,
        splits=STACK_SPLITS,
        feature_modes=tuple(required_feature_modes),
        anchor_model_name=anchor_model_name,
        quantile_bins=quantile_bins,
    )
    audits = audit_group_prediction_blocks(blocks_by_split, split_sets, expected_splits=STACK_SPLITS)
    fuser_reports: dict[str, Any] = {}
    for fuser_name in fusers:
        if fuser_name in ("mean_logits", "mean_probs"):
            fuser_reports[fuser_name] = evaluate_mean_fuser(blocks_by_split, fuser_name=fuser_name)
        elif fuser_name in LOGISTIC_FUSER_FEATURE_MODES:
            fuser_reports[fuser_name] = fit_global_logistic_fuser(
                split_sets,
                fuser_name=fuser_name,
                feature_mode=LOGISTIC_FUSER_FEATURE_MODES[fuser_name],
                c_grid=c_grid,
                max_iter=max_iter,
                stacker_dir=stacker_dir,
            )
        elif fuser_name.endswith("_bin_gated_logistic"):
            fuser_reports[fuser_name] = fit_bin_gated_logistic_fuser(
                split_sets,
                fuser_name=fuser_name,
                c_grid=c_grid,
                max_iter=max_iter,
                quantile_bins=quantile_bins,
                min_bin_train_rows=min_bin_train_rows,
                stacker_dir=stacker_dir,
            )
        else:
            raise ValueError(f"Unknown fuser {fuser_name!r}")

    controls = (
        run_group_negative_controls(
            split_sets,
            feature_modes=control_feature_modes,
            c_grid=c_grid,
            max_iter=max_iter,
            seed=control_seed,
            warning_min_accuracy=control_warning_min_accuracy,
            warning_chance_margin=control_warning_chance_margin,
        )
        if run_controls
        else {
            "enabled": False,
            "ok": True,
            "reason": "Controls disabled by CrossArchFusionFitConfig.run_controls=False",
            "mode_reports": {},
            "suspicious_flags": [],
        }
    )
    reported_fusers = set(fuser_reports)
    fuser_completion = {
        "ok": reported_fusers == set(fusers),
        "requested_fusers": list(fusers),
        "reported_fusers": sorted(reported_fusers),
        "missing_fusers": sorted(set(fusers) - reported_fusers),
        "skipped_fusers": sorted(
            name for name, payload in fuser_reports.items() if payload.get("status") == "skipped"
        ),
    }
    group_ok = bool(audits["ok"] and controls.get("ok", True) and fuser_completion["ok"])
    return {
        "group_name": group_name,
        "model_names": list(model_names),
        "n_models": len(model_names),
        "ok": group_ok,
        "audits": audits,
        "controls": controls,
        "suspicious_flags": list(controls.get("suspicious_flags", [])),
        "fuser_completion": fuser_completion,
        "feature_summary": {split: split_set.summary() for split, split_set in split_sets.items()},
        "bin_specs": bin_specs,
        "fusers": fuser_reports,
    }


def run_crossarch_fusers(config: CrossArchFusionFitConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    report_path = output_dir / "fusion_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite locked crossarch fusion report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _groups_from_config(config)
    config_payload = {
        **asdict(config),
        "groups": groups,
        "feature_modes": list(config.feature_modes),
        "splits": list(config.splits),
        "fusers": list(config.fusers),
        "c_grid": [float(value) for value in config.c_grid],
        "control_feature_modes": list(config.control_feature_modes),
    }
    (output_dir / "fusion_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    group_reports = {}
    stacker_dir = output_dir / "stackers"
    for group_name, model_names in groups.items():
        group_reports[group_name] = fit_group_fusers(
            config.prediction_dir,
            group_name=group_name,
            model_names=model_names,
            fusers=config.fusers,
            c_grid=config.c_grid,
            max_iter=config.max_iter,
            anchor_model_name=config.anchor_model_name,
            quantile_bins=config.quantile_bins,
            min_bin_train_rows=config.min_bin_train_rows,
            stacker_dir=stacker_dir,
            run_controls=config.run_controls,
            control_seed=config.control_seed,
            control_feature_modes=config.control_feature_modes,
            control_warning_min_accuracy=config.control_warning_min_accuracy,
            control_warning_chance_margin=config.control_warning_chance_margin,
        )

    final_test_guardrail = final_test_guardrail_audit(config)
    suspicious_flags = [
        dict(flag, group=group_name) if "group" not in flag else flag
        for group_name, group_report in group_reports.items()
        for flag in group_report.get("suspicious_flags", [])
    ]
    audit_summary = {
        "ok": bool(
            final_test_guardrail["ok"]
            and all(group_report.get("audits", {}).get("ok", False) for group_report in group_reports.values())
        ),
        "final_test_guardrail": final_test_guardrail,
        "groups": {
            group_name: {
                "ok": bool(group_report.get("audits", {}).get("ok", False)),
                "source_alignment_ok": bool(
                    group_report.get("audits", {}).get("source_alignment", {}).get("ok", False)
                ),
                "group_size_ok": bool(group_report.get("audits", {}).get("group_size", {}).get("ok", False)),
                "split_leakage_ok": bool(group_report.get("audits", {}).get("split_leakage", {}).get("ok", False)),
            }
            for group_name, group_report in group_reports.items()
        },
    }
    controls_summary = {
        "enabled": bool(config.run_controls),
        "ok": all(group_report.get("controls", {}).get("ok", True) for group_report in group_reports.values()),
        "suspicious_flag_count": int(len(suspicious_flags)),
        "feature_modes": list(config.control_feature_modes),
        "seed": int(config.control_seed),
    }
    ok = bool(
        audit_summary["ok"]
        and controls_summary["ok"]
        and all(group_report.get("ok", False) for group_report in group_reports.values())
    )
    report = {
        "ok": ok,
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": FUSER_EXPERIMENT_STEP,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_dir": str(config.prediction_dir),
        "output_dir": str(output_dir),
        "groups": group_reports,
        "fusers": list(config.fusers),
        "c_grid": [float(value) for value in config.c_grid],
        "max_iter": int(config.max_iter),
        "audit_summary": audit_summary,
        "controls_summary": controls_summary,
        "suspicious_flags": suspicious_flags,
        "leakage_rules": {
            "inputs": "frozen prediction blocks only",
            "fuser_fit_split": "stack_train",
            "regularization_selection_split": "stack_val",
            "final_test_evaluated_after_selection": True,
            "no_model_checkpoints_loaded": True,
            "no_training_data_loaded": True,
            "controls_use_stack_train_only_for_fit": True,
            "source_alignment_checked_before_fusion": True,
            "split_identity_overlap_checked": True,
        },
        "output_files": {
            "fusion_config": str(output_dir / "fusion_config.json"),
            "fusion_report": str(report_path),
            "stacker_dir": str(stacker_dir),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def default_crossarch_feature_groups(*, include_optional: bool = False) -> dict[str, list[str]]:
    groups = build_fusion_groups(include_optional=include_optional)
    return {name: list(group.model_names) for name, group in groups.items()}


def validate_crossarch_feature_groups(groups: Mapping[str, Sequence[str]]) -> dict[str, FusionGroupSpec]:
    if not groups:
        raise ValueError("At least one feature group is required")
    return {
        str(name): FusionGroupSpec(name=str(name), model_names=tuple(models))
        for name, models in groups.items()
    }


__all__ = [
    "ALL_FEATURE_MODES",
    "BIN_GATED_FUSER_FEATURE_MODE",
    "CONTROL_WARNING_CHANCE_MARGIN",
    "CONTROL_WARNING_MIN_ACCURACY",
    "DEFAULT_CONTROL_FEATURE_MODES",
    "DEFAULT_CROSSARCH_FUSERS",
    "DEFAULT_BIN_SCORE_NAMES",
    "EXPERIMENT_STEP",
    "FEATURE_MODES",
    "FUSER_EXPERIMENT_STEP",
    "LOGITS_PROBS_UNCERTAINTY_FEATURE_MODE",
    "LOGISTIC_FUSER_FEATURE_MODES",
    "UNCERTAINTY_FEATURE_MODE",
    "CrossArchFusionFeatureBuildConfig",
    "CrossArchFusionFitConfig",
    "CrossArchSplitFeatureSet",
    "NamedFeatureMatrix",
    "assign_bins_from_specs",
    "assign_quantile_bins",
    "audit_group_prediction_blocks",
    "bin_assignment_name_for_fuser",
    "build_bin_scores",
    "build_group_feature_report",
    "build_split_feature_set",
    "build_split_feature_set_from_blocks",
    "build_uncertainty_feature_matrix",
    "default_crossarch_feature_groups",
    "distinct_predicted_class_count",
    "entropy_from_probs",
    "evaluate_mean_fuser",
    "fit_bin_gated_logistic_fuser",
    "fit_bin_specs",
    "fit_global_logistic_fuser",
    "fit_group_fusers",
    "final_test_guardrail_audit",
    "labels_hash",
    "load_group_fusion_data",
    "mean_logits_probs",
    "mean_probs_probs",
    "pairwise_disagreement_fraction",
    "predicted_classes",
    "quantile_edges",
    "raw_feature_matrix",
    "run_group_negative_controls",
    "run_crossarch_feature_builder",
    "run_crossarch_fusers",
    "save_feature_set",
    "top1_margin_from_probs",
    "validate_crossarch_feature_groups",
]

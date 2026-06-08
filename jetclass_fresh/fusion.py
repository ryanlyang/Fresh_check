"""Step 10 frozen-model prediction collection and stacked fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .dual_view import (
    HLTTokenDataset,
    build_dual_view_tagger,
    build_part_inputs_torch,
    build_soft_corrected_view_torch,
    load_stage_a_reconstructor_checkpoint,
    make_hlt_token_loader,
)
from .hlt_baseline import (
    JetViewTorchDataset,
    ParticleTransformerHLTClassifier,
    make_data_loader,
    require_torch,
    resolve_device,
    save_json,
)
from .hlt_cache import hash_arrays, jet_identity_hash, load_cached_hlt_view
from .jetclass_data import JetIdentity, LABEL_NAMES
from .reconstructor import RECONSTRUCTOR_VARIANT_NAMES


STACK_SPLITS = ["stack_train", "stack_val", "final_test"]
DEFAULT_C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
MAX_REPAIRED_PREDICTION_ROW_FRACTION = 1.0e-5


@dataclass(frozen=True)
class FusionModelSpec:
    """Frozen model to evaluate for Step 10 fusion."""

    name: str
    kind: str
    checkpoint: str

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "kind": self.kind, "checkpoint": self.checkpoint}


@dataclass
class PredictionBlock:
    """One frozen model's logits/probabilities on one split."""

    model_name: str
    split: str
    logits: np.ndarray
    probs: np.ndarray
    labels: np.ndarray
    jet_ids: List[JetIdentity]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinearStacker:
    """Standardized linear multinomial stacker."""

    coef: np.ndarray
    intercept: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    C: float
    model_names: List[str]
    feature_mode: str = "logits_probs"
    solver: str = "sklearn"

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        x = (features.astype(np.float64) - self.mean) / self.scale
        return x @ self.coef.T + self.intercept

    def predict_probs(self, features: np.ndarray) -> np.ndarray:
        return softmax_np(self.predict_logits(features))


@dataclass
class FusionRunConfig:
    """CLI-level Step 10 configuration."""

    output_dir: str
    hlt_cache_dir: str
    hlt_checkpoint: str
    reco_root: str = "checkpoints/jetclass_fresh_reco7"
    variants: List[str] = field(default_factory=lambda: list(RECONSTRUCTOR_VARIANT_NAMES))
    splits: List[str] = field(default_factory=lambda: list(STACK_SPLITS))
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    max_jets_per_split: int | None = None
    overwrite_predictions: bool = False
    skip_existing_predictions: bool = True
    confirm_final_test: bool = False
    C_grid: List[float] = field(default_factory=lambda: list(DEFAULT_C_GRID))
    feature_mode: str = "logits_probs"
    max_iter: int = 500


def default_reco7_plus_hlt_specs(
    *,
    hlt_checkpoint: str,
    reco_root: str = "checkpoints/jetclass_fresh_reco7",
    variants: Sequence[str] = RECONSTRUCTOR_VARIANT_NAMES,
) -> List[FusionModelSpec]:
    specs = [FusionModelSpec(name="hlt_baseline", kind="hlt", checkpoint=str(hlt_checkpoint))]
    for variant in variants:
        specs.append(
            FusionModelSpec(
                name=variant,
                kind="dual_view",
                checkpoint=str(Path(reco_root) / variant / "stage2_dual_view" / "best_model_val.pt"),
            )
        )
    return specs


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


def sanitize_prediction_logits(
    logits: np.ndarray,
    *,
    model_name: str,
    split: str,
    max_bad_row_fraction: float = MAX_REPAIRED_PREDICTION_ROW_FRACTION,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Repair tiny non-finite prediction tails and reject numerically broken models."""

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError(f"{model_name}/{split} logits must be 2D, got shape {logits.shape}")
    finite = np.isfinite(logits)
    bad_rows = ~np.all(finite, axis=1)
    bad_row_count = int(np.sum(bad_rows))
    bad_value_count = int(logits.size - np.sum(finite))
    row_fraction = bad_row_count / float(max(logits.shape[0], 1))
    report = {
        "nonfinite_value_count": bad_value_count,
        "nonfinite_row_count": bad_row_count,
        "nonfinite_row_fraction": row_fraction,
        "max_allowed_nonfinite_row_fraction": float(max_bad_row_fraction),
        "repaired": bad_row_count > 0,
    }
    if bad_row_count == 0:
        return logits, report
    if row_fraction > float(max_bad_row_fraction):
        raise FloatingPointError(
            f"Non-finite predictions for {model_name}/{split}: "
            f"{bad_row_count}/{logits.shape[0]} rows ({row_fraction:.6g}) exceed "
            f"the repair limit {float(max_bad_row_fraction):.6g}"
        )
    repaired = logits.copy()
    repaired[bad_rows, :] = 0.0
    return repaired, report


def classification_metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not np.isfinite(probs).all():
        raise FloatingPointError("Cannot compute classification metrics from non-finite probabilities")
    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == labels)) if len(labels) else 0.0
    picked = np.clip(probs[np.arange(len(labels)), labels], 1.0e-12, 1.0)
    return {
        "accuracy": accuracy,
        "cross_entropy": float(-np.mean(np.log(picked))) if len(labels) else float("nan"),
        "n_jets": int(len(labels)),
    }


def classification_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return classification_metrics_from_probs(softmax_np(logits), labels)


def _identity_arrays(jet_ids: Sequence[JetIdentity]) -> tuple[list[str], np.ndarray, np.ndarray]:
    files: list[str] = []
    file_to_index: Dict[str, int] = {}
    file_indices = np.zeros((len(jet_ids),), dtype=np.int32)
    entries = np.zeros((len(jet_ids),), dtype=np.int64)
    for index, identity in enumerate(jet_ids):
        if identity.file not in file_to_index:
            file_to_index[identity.file] = len(files)
            files.append(identity.file)
        file_indices[index] = file_to_index[identity.file]
        entries[index] = int(identity.entry)
    return files, file_indices, entries


def prediction_paths(prediction_dir: str | Path, model_name: str, split: str) -> tuple[Path, Path]:
    root = Path(prediction_dir) / model_name
    return root / f"{split}_predictions.npz", root / f"{split}_predictions_metadata.json"


def save_prediction_block(block: PredictionBlock, prediction_dir: str | Path, *, overwrite: bool = False) -> Dict[str, Any]:
    npz_path, meta_path = prediction_paths(prediction_dir, block.model_name, block.split)
    if not overwrite and (npz_path.exists() or meta_path.exists()):
        raise FileExistsError(f"Prediction block already exists: {npz_path}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    jet_files, file_indices, entries = _identity_arrays(block.jet_ids)
    logits, finite_report = sanitize_prediction_logits(
        block.logits,
        model_name=block.model_name,
        split=block.split,
    )
    probs = softmax_np(logits)
    arrays = {
        "logits": logits.astype(np.float32, copy=False),
        "probs": probs.astype(np.float32, copy=False),
        "labels": block.labels.astype(np.int64, copy=False),
        "jet_file_indices": file_indices,
        "jet_entries": entries,
    }
    np.savez_compressed(npz_path, **arrays)
    content_hash = hash_arrays(arrays)
    metadata = {
        **block.metadata,
        "model_name": block.model_name,
        "split": block.split,
        "npz_path": str(npz_path),
        "metadata_path": str(meta_path),
        "jet_files": jet_files,
        "jet_identity_hash": jet_identity_hash(block.jet_ids),
        "prediction_content_hash": content_hash,
        "n_jets": int(len(block.labels)),
        "num_classes": int(logits.shape[1]),
        "prediction_finite_check": finite_report,
        "metrics": classification_metrics_from_logits(logits, block.labels),
    }
    save_json(meta_path, metadata)
    return metadata


def load_prediction_block(prediction_dir: str | Path, model_name: str, split: str, *, verify_hash: bool = True) -> PredictionBlock:
    npz_path, meta_path = prediction_paths(prediction_dir, model_name, split)
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with np.load(npz_path, allow_pickle=False) as data:
        logits = data["logits"].astype(np.float32)
        probs = data["probs"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        file_indices = data["jet_file_indices"].astype(np.int64)
        entries = data["jet_entries"].astype(np.int64)
    jet_files = [str(path) for path in metadata["jet_files"]]
    jet_ids = [
        JetIdentity(file=jet_files[int(file_index)], entry=int(entry), label=int(label))
        for file_index, entry, label in zip(file_indices, entries, labels)
    ]
    if verify_hash:
        actual = hash_arrays(
            {
                "logits": logits,
                "probs": probs,
                "labels": labels,
                "jet_file_indices": file_indices.astype(np.int32),
                "jet_entries": entries,
            }
        )
        if actual != metadata.get("prediction_content_hash"):
            raise ValueError(f"Prediction hash mismatch for {model_name}/{split}: {actual}")
    logits, finite_report = sanitize_prediction_logits(logits, model_name=model_name, split=split)
    probs = softmax_np(logits)
    if finite_report["repaired"]:
        metadata = dict(metadata)
        metadata["loaded_prediction_finite_check"] = finite_report
    return PredictionBlock(
        model_name=model_name,
        split=split,
        logits=logits,
        probs=probs,
        labels=labels,
        jet_ids=jet_ids,
        metadata=metadata,
    )


def validate_prediction_alignment(blocks: Sequence[PredictionBlock]) -> None:
    if not blocks:
        raise ValueError("No prediction blocks provided")
    labels = blocks[0].labels
    jet_ids = blocks[0].jet_ids
    split = blocks[0].split
    for block in blocks[1:]:
        if block.split != split:
            raise ValueError(f"Split mismatch: {block.split} != {split}")
        if not np.array_equal(block.labels, labels):
            raise ValueError(f"Label mismatch between {blocks[0].model_name} and {block.model_name}")
        if block.jet_ids != jet_ids:
            raise ValueError(f"Jet identity mismatch between {blocks[0].model_name} and {block.model_name}")


def stack_feature_matrix(blocks: Sequence[PredictionBlock], *, feature_mode: str = "logits_probs") -> np.ndarray:
    validate_prediction_alignment(blocks)
    columns: list[np.ndarray] = []
    for block in blocks:
        if feature_mode in ("logits", "logits_probs"):
            columns.append(block.logits)
        if feature_mode in ("probs", "logits_probs"):
            columns.append(block.probs)
    if not columns:
        raise ValueError(f"Unknown feature_mode {feature_mode!r}")
    features = np.concatenate(columns, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise FloatingPointError("Stacked fusion feature matrix contains non-finite values")
    return features


def load_hlt_model_from_checkpoint(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    model_config = payload.get("model_config") or {}
    if not model_config:
        cfg = payload.get("config", {})
        from .hlt_baseline import build_hlt_classifier

        model = build_hlt_classifier(num_classes=len(LABEL_NAMES), model_size=cfg.get("model_size", "base"))
    else:
        model = ParticleTransformerHLTClassifier(**model_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def load_dual_view_model_from_checkpoint(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    cfg = payload.get("config", {})
    model_cfg = payload.get("model_config", {})
    tagger_kwargs = {
        "num_classes": int(model_cfg.get("num_classes", len(LABEL_NAMES))),
        "model_size": model_cfg.get("model_size", cfg.get("model_size", "base")),
        "hidden_dim": model_cfg.get("hidden_dim"),
        "num_heads": model_cfg.get("num_heads"),
        "num_layers": model_cfg.get("num_layers"),
        "feedforward_dim": model_cfg.get("feedforward_dim"),
        "dropout": model_cfg.get("dropout", 0.05),
        "architecture": model_cfg.get("architecture", "cross_attention_fusion"),
    }
    tagger = build_dual_view_tagger(**tagger_kwargs)
    tagger.load_state_dict(payload["model_state_dict"], strict=True)
    tagger = tagger.to(device)
    tagger.eval()
    reco_path = payload.get("reconstructor_checkpoint") or cfg.get("reconstructor_checkpoint")
    if not reco_path:
        raise ValueError(f"Dual-view checkpoint {path} does not record a reconstructor checkpoint")
    reconstructor, reco_payload = load_stage_a_reconstructor_checkpoint(reco_path, device=device)
    return tagger, reconstructor, payload, reco_payload


def _maybe_limit_view(view, max_jets: int | None):
    if max_jets is None:
        return view
    from .jetclass_data import JetView

    limit = min(int(max_jets), len(view.labels))
    return JetView(
        tokens=view.tokens[:limit],
        mask=view.mask[:limit],
        labels=view.labels[:limit],
        jet_ids=view.jet_ids[:limit],
        split=view.split,
        metadata=dict(view.metadata),
    )


def evaluate_hlt_model(
    model,
    view,
    *,
    model_name: str = "hlt_baseline",
    batch_size: int,
    num_workers: int,
    device,
    max_jets: int | None = None,
) -> PredictionBlock:
    torch = require_torch()
    view = _maybe_limit_view(view, max_jets)
    dataset = JetViewTorchDataset(view)
    loader = make_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=12345,
    )
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            logits = model(batch["points"], batch["features"], batch["lorentz_vectors"], batch["mask"])
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
    logits_np = np.concatenate(logits_rows, axis=0)
    labels_np = np.concatenate(labels_rows, axis=0)
    return PredictionBlock(
        model_name=model_name,
        split=view.split,
        logits=logits_np,
        probs=softmax_np(logits_np),
        labels=labels_np,
        jet_ids=list(view.jet_ids),
        metadata={
            "model_kind": "hlt",
            "hlt_content_hash": view.metadata.get("hlt_content_hash"),
            "allowed_inputs": "cached_fixed_hlt_only",
        },
    )


def evaluate_dual_view_model(
    model_name: str,
    tagger,
    reconstructor,
    view,
    *,
    batch_size: int,
    num_workers: int,
    device,
    max_constits: int = 128,
    max_jets: int | None = None,
) -> PredictionBlock:
    torch = require_torch()
    view = _maybe_limit_view(view, max_jets)
    dataset = HLTTokenDataset(view)
    loader = make_hlt_token_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=12345,
    )
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            reco = reconstructor(batch["hlt_tokens"], batch["hlt_mask"])
            hlt_inputs = build_part_inputs_torch(batch["hlt_tokens"], batch["hlt_mask"], max_constits=max_constits)
            corrected_inputs = build_soft_corrected_view_torch(
                batch["hlt_tokens"],
                batch["hlt_mask"],
                reco,
            )
            logits = tagger(hlt_inputs, corrected_inputs)
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
    logits_np = np.concatenate(logits_rows, axis=0)
    labels_np = np.concatenate(labels_rows, axis=0)
    return PredictionBlock(
        model_name=model_name,
        split=view.split,
        logits=logits_np,
        probs=softmax_np(logits_np),
        labels=labels_np,
        jet_ids=list(view.jet_ids),
        metadata={
            "model_kind": "dual_view",
            "dual_view_architecture": "cross_attention_fusion",
            "hlt_content_hash": view.metadata.get("hlt_content_hash"),
            "allowed_inputs": "cached_fixed_hlt_and_parent_aligned_corrected_view_from_cached_fixed_hlt",
        },
    )


def collect_frozen_predictions(
    specs: Sequence[FusionModelSpec],
    *,
    hlt_cache_dir: str | Path,
    prediction_dir: str | Path,
    splits: Sequence[str] = STACK_SPLITS,
    batch_size: int = 128,
    num_workers: int = 0,
    device: str = "auto",
    max_jets_per_split: int | None = None,
    overwrite: bool = False,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    torch = require_torch()
    resolved_device = resolve_device(device)
    reports: Dict[str, Any] = {}
    for spec in specs:
        reports[spec.name] = {}
        if spec.kind == "hlt":
            model, payload = load_hlt_model_from_checkpoint(spec.checkpoint, device=resolved_device)
            for split in splits:
                npz_path, _ = prediction_paths(prediction_dir, spec.name, split)
                if npz_path.exists() and skip_existing and not overwrite:
                    reports[spec.name][split] = load_prediction_block(prediction_dir, spec.name, split).metadata
                    continue
                view = load_cached_hlt_view(hlt_cache_dir, split)
                block = evaluate_hlt_model(
                    model,
                    view,
                    model_name=spec.name,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=resolved_device,
                    max_jets=max_jets_per_split,
                )
                block.metadata.update({"checkpoint": spec.checkpoint, "checkpoint_epoch": payload.get("epoch")})
                reports[spec.name][split] = save_prediction_block(block, prediction_dir, overwrite=overwrite)
            del model
        elif spec.kind == "dual_view":
            tagger, reconstructor, payload, reco_payload = load_dual_view_model_from_checkpoint(spec.checkpoint, device=resolved_device)
            for split in splits:
                npz_path, _ = prediction_paths(prediction_dir, spec.name, split)
                if npz_path.exists() and skip_existing and not overwrite:
                    reports[spec.name][split] = load_prediction_block(prediction_dir, spec.name, split).metadata
                    continue
                view = load_cached_hlt_view(hlt_cache_dir, split)
                block = evaluate_dual_view_model(
                    spec.name,
                    tagger,
                    reconstructor,
                    view,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=resolved_device,
                    max_jets=max_jets_per_split,
                )
                block.metadata.update(
                    {
                        "checkpoint": spec.checkpoint,
                        "checkpoint_epoch": payload.get("epoch"),
                        "reconstructor_checkpoint": payload.get("reconstructor_checkpoint"),
                        "reconstructor_epoch": reco_payload.get("epoch"),
                    }
                )
                reports[spec.name][split] = save_prediction_block(block, prediction_dir, overwrite=overwrite)
            del tagger
            del reconstructor
        else:
            raise ValueError(f"Unknown model kind {spec.kind!r} for {spec.name}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return reports


def load_blocks_for_split(prediction_dir: str | Path, model_names: Sequence[str], split: str) -> List[PredictionBlock]:
    blocks = [load_prediction_block(prediction_dir, name, split) for name in model_names]
    validate_prediction_alignment(blocks)
    return blocks


def uniform_probability_average(blocks: Sequence[PredictionBlock]) -> np.ndarray:
    validate_prediction_alignment(blocks)
    return np.mean(np.stack([block.probs for block in blocks], axis=0), axis=0)


def weighted_average_values(blocks: Sequence[PredictionBlock], weights: np.ndarray, *, mode: str) -> np.ndarray:
    validate_prediction_alignment(blocks)
    values = np.stack([block.probs if mode == "probs" else block.logits for block in blocks], axis=0)
    weights = np.asarray(weights, dtype=np.float64)
    combined = np.tensordot(weights, values, axes=(0, 0))
    if mode == "probs":
        combined = np.clip(combined, 1.0e-12, None)
        return combined / np.sum(combined, axis=1, keepdims=True)
    if mode == "logits":
        return softmax_np(combined)
    raise ValueError(f"Unknown weighted average mode {mode!r}")


def select_weighted_average_weights(
    blocks: Sequence[PredictionBlock],
    *,
    mode: str,
    max_steps: int = 30,
) -> tuple[np.ndarray, Dict[str, Any]]:
    validate_prediction_alignment(blocks)
    n_models = len(blocks)
    labels = blocks[0].labels
    candidate_values = np.stack([block.probs if mode == "probs" else block.logits for block in blocks], axis=0)

    def score(weights: np.ndarray) -> tuple[float, float]:
        combined = np.tensordot(weights, candidate_values, axes=(0, 0))
        probs = combined if mode == "probs" else softmax_np(combined)
        if mode == "probs":
            probs = np.clip(probs, 1.0e-12, None)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
        metrics = classification_metrics_from_probs(probs, labels)
        return metrics["accuracy"], -metrics["cross_entropy"]

    single_scores = []
    for idx in range(n_models):
        weights = np.zeros(n_models, dtype=np.float64)
        weights[idx] = 1.0
        single_scores.append(score(weights))
    best_idx = int(np.argmax([item[0] for item in single_scores]))
    weights = np.zeros(n_models, dtype=np.float64)
    weights[best_idx] = 1.0
    best_score = score(weights)
    grid = np.linspace(0.05, 0.95, 19)
    history = [{"step": 0, "score": list(best_score), "weights": weights.tolist()}]
    for step in range(1, int(max_steps) + 1):
        step_best = best_score
        step_weights = weights
        for model_idx in range(n_models):
            basis = np.zeros(n_models, dtype=np.float64)
            basis[model_idx] = 1.0
            for alpha in grid:
                trial = (1.0 - float(alpha)) * weights + float(alpha) * basis
                trial = trial / np.sum(trial)
                trial_score = score(trial)
                if trial_score > step_best:
                    step_best = trial_score
                    step_weights = trial
        if np.allclose(step_weights, weights):
            break
        weights = step_weights
        best_score = step_best
        history.append({"step": step, "score": list(best_score), "weights": weights.tolist()})
    return weights, {"mode": mode, "history": history, "selected_score": list(best_score)}


def _standardize_train(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    return (features - mean) / scale, mean, scale


def _fit_sklearn_logistic(features: np.ndarray, labels: np.ndarray, *, C: float, max_iter: int) -> LinearStacker | None:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None
    x, mean, scale = _standardize_train(features.astype(np.float64))
    clf = LogisticRegression(
        C=float(C),
        penalty="l2",
        solver="lbfgs",
        max_iter=int(max_iter),
        multi_class="multinomial",
    )
    clf.fit(x, labels)
    return LinearStacker(
        coef=clf.coef_.astype(np.float64),
        intercept=clf.intercept_.astype(np.float64),
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        C=float(C),
        model_names=[],
        solver="sklearn",
    )


def _fit_numpy_logistic(features: np.ndarray, labels: np.ndarray, *, C: float, max_iter: int, num_classes: int) -> LinearStacker:
    x, mean, scale = _standardize_train(features.astype(np.float64))
    n_rows, n_features = x.shape
    y = np.eye(num_classes, dtype=np.float64)[labels]
    coef = np.zeros((num_classes, n_features), dtype=np.float64)
    intercept = np.zeros((num_classes,), dtype=np.float64)
    lr = 0.25
    l2 = 1.0 / max(float(C), 1.0e-6)
    for _ in range(int(max_iter)):
        logits = x @ coef.T + intercept
        probs = softmax_np(logits).astype(np.float64)
        error = (probs - y) / float(n_rows)
        grad_w = error.T @ x + l2 * coef / float(n_rows)
        grad_b = error.sum(axis=0)
        coef -= lr * grad_w
        intercept -= lr * grad_b
    return LinearStacker(
        coef=coef,
        intercept=intercept,
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        C=float(C),
        model_names=[],
        solver="numpy_gd",
    )


def fit_logistic_stacker(
    train_blocks: Sequence[PredictionBlock],
    val_blocks: Sequence[PredictionBlock],
    *,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    feature_mode: str = "logits_probs",
    max_iter: int = 500,
) -> tuple[LinearStacker, Dict[str, Any]]:
    train_x = stack_feature_matrix(train_blocks, feature_mode=feature_mode)
    val_x = stack_feature_matrix(val_blocks, feature_mode=feature_mode)
    train_y = train_blocks[0].labels
    val_y = val_blocks[0].labels
    model_names = [block.model_name for block in train_blocks]
    candidates = []
    num_classes = int(max(np.max(train_y), np.max(val_y)) + 1)
    for c_value in C_grid:
        stacker = _fit_sklearn_logistic(train_x, train_y, C=float(c_value), max_iter=max_iter)
        if stacker is None:
            stacker = _fit_numpy_logistic(train_x, train_y, C=float(c_value), max_iter=max_iter, num_classes=num_classes)
        stacker.model_names = list(model_names)
        stacker.feature_mode = feature_mode
        val_probs = stacker.predict_probs(val_x)
        metrics = classification_metrics_from_probs(val_probs, val_y)
        candidates.append({"C": float(c_value), "metrics": metrics, "solver": stacker.solver, "stacker": stacker})
    best = max(candidates, key=lambda row: (row["metrics"]["accuracy"], -row["metrics"]["cross_entropy"]))
    report = {
        "feature_mode": feature_mode,
        "model_names": model_names,
        "candidates": [
            {"C": row["C"], "metrics": row["metrics"], "solver": row["solver"]}
            for row in candidates
        ],
        "selected_C": best["C"],
        "selected_metrics": best["metrics"],
        "selected_solver": best["solver"],
    }
    return best["stacker"], report


def save_linear_stacker(stacker: LinearStacker, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coef=stacker.coef.astype(np.float64),
        intercept=stacker.intercept.astype(np.float64),
        mean=stacker.mean.astype(np.float64),
        scale=stacker.scale.astype(np.float64),
        C=np.asarray([stacker.C], dtype=np.float64),
        model_names=np.asarray(stacker.model_names),
        feature_mode=np.asarray([stacker.feature_mode]),
        solver=np.asarray([stacker.solver]),
    )


def evaluate_fusion_methods(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    feature_mode: str = "logits_probs",
    max_iter: int = 500,
) -> Dict[str, Any]:
    blocks_by_split = {
        split: load_blocks_for_split(prediction_dir, model_names, split)
        for split in STACK_SPLITS
    }
    labels_by_split = {split: blocks[0].labels for split, blocks in blocks_by_split.items()}
    report: Dict[str, Any] = {"model_names": list(model_names), "splits": list(STACK_SPLITS), "single_models": {}}
    for split, blocks in blocks_by_split.items():
        report["single_models"][split] = {
            block.model_name: classification_metrics_from_logits(block.logits, block.labels)
            for block in blocks
        }

    uniform_metrics = {}
    for split, blocks in blocks_by_split.items():
        uniform_metrics[split] = classification_metrics_from_probs(uniform_probability_average(blocks), labels_by_split[split])

    prob_weights, prob_weight_report = select_weighted_average_weights(blocks_by_split["stack_val"], mode="probs")
    logit_weights, logit_weight_report = select_weighted_average_weights(blocks_by_split["stack_val"], mode="logits")
    weighted_prob_metrics = {}
    weighted_logit_metrics = {}
    for split, blocks in blocks_by_split.items():
        weighted_prob_metrics[split] = classification_metrics_from_probs(
            weighted_average_values(blocks, prob_weights, mode="probs"),
            labels_by_split[split],
        )
        weighted_logit_metrics[split] = classification_metrics_from_probs(
            weighted_average_values(blocks, logit_weights, mode="logits"),
            labels_by_split[split],
        )

    stacker, stacker_report = fit_logistic_stacker(
        blocks_by_split["stack_train"],
        blocks_by_split["stack_val"],
        C_grid=C_grid,
        feature_mode=feature_mode,
        max_iter=max_iter,
    )
    stacker_metrics = {}
    for split, blocks in blocks_by_split.items():
        features = stack_feature_matrix(blocks, feature_mode=feature_mode)
        stacker_metrics[split] = classification_metrics_from_probs(stacker.predict_probs(features), labels_by_split[split])

    report.update(
        {
            "uniform_probability_average": uniform_metrics,
            "weighted_probability_average": {
                "weights": prob_weights.tolist(),
                "selection": prob_weight_report,
                "metrics": weighted_prob_metrics,
            },
            "weighted_logit_average": {
                "weights": logit_weights.tolist(),
                "selection": logit_weight_report,
                "metrics": weighted_logit_metrics,
            },
            "stacked_logistic_regression": {
                "selection": stacker_report,
                "metrics": stacker_metrics,
            },
            "final_test_evaluated": True,
            "final_test_evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "locked_final_test_note": (
                "This Step 10 report evaluates final_test after stack choices are fixed by stack_train/stack_val."
            ),
        }
    )
    return {"report": report, "stacker": stacker}


def run_reco7_fusion(config: FusionRunConfig) -> Dict[str, Any]:
    if "final_test" in config.splits and not config.confirm_final_test:
        raise ValueError("Refusing to evaluate final_test without confirm_final_test=True")
    output_dir = Path(config.output_dir)
    report_path = output_dir / "fusion_report.json"
    if report_path.exists():
        raise FileExistsError(f"Fusion report already exists; refusing to overwrite locked result: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    specs = default_reco7_plus_hlt_specs(
        hlt_checkpoint=config.hlt_checkpoint,
        reco_root=config.reco_root,
        variants=config.variants,
    )
    save_json(
        output_dir / "fusion_config.json",
        {
            "config": asdict(config),
            "model_specs": [spec.to_dict() for spec in specs],
            "leakage_rule": "Fusion inputs are frozen model logits/probabilities plus class labels only.",
        },
    )
    prediction_report = collect_frozen_predictions(
        specs,
        hlt_cache_dir=config.hlt_cache_dir,
        prediction_dir=prediction_dir,
        splits=config.splits,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=config.device,
        max_jets_per_split=config.max_jets_per_split,
        overwrite=config.overwrite_predictions,
        skip_existing=config.skip_existing_predictions,
    )
    if list(config.splits) != STACK_SPLITS:
        return {"prediction_report": prediction_report, "fusion_report": None}
    fusion = evaluate_fusion_methods(
        prediction_dir,
        [spec.name for spec in specs],
        C_grid=config.C_grid,
        feature_mode=config.feature_mode,
        max_iter=config.max_iter,
    )
    save_linear_stacker(fusion["stacker"], output_dir / "stacked_logistic_regression.npz")
    final_report = {"prediction_report": prediction_report, **fusion["report"]}
    save_json(report_path, final_report)
    return final_report

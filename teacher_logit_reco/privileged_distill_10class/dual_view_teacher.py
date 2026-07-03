"""PD10 Step 5 dual-view logit-fusion teacher."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import (
    PredictionBlock,
    classification_metrics_from_logits,
    load_prediction_block,
    prediction_paths,
    save_prediction_block,
    softmax_np,
    validate_prediction_alignment,
)
from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.jetclass_data import JetIdentity, LABEL_NAMES

from .config import (
    PD10_EXPERIMENT_NAME,
    PD10_NUM_CLASSES,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_HLT,
    PD10_TEACHER_OFFLINE,
    default_pd10_experiment_layout,
    pd10_teacher_model_name,
)
from .logits import (
    PD10_TEACHER_LOGIT_CACHE_CONTRACT,
    PD10_TEACHER_LOGIT_CACHE_MANIFEST,
    PD10_TEACHER_LOGIT_CACHE_REPORT,
    PD10_TEACHER_LOGIT_SPLITS,
    load_pd10_teacher_logit_block,
)
from .representations import (
    PD10TeacherRepresentationCacheConfig,
    build_pd10_teacher_representation_block,
    load_pd10_teacher_representation_block,
    save_pd10_teacher_representation_block,
    write_pd10_teacher_representation_manifest,
)
from .teachers import sha256_file


PD10_STEP5_EXPERIMENT_STEP = "pd10_step5_dual_view_logit_teacher"
PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT = "pd10_dual_view_logit_teacher_v1"
PD10_DUAL_VIEW_LOGIT_FEATURE_DIM = 58
PD10_DUAL_VIEW_LOGIT_SUMMARY_DIM = 8
PD10_DUAL_VIEW_LOGIT_MODEL_NAME = pd10_teacher_model_name(PD10_TEACHER_DUAL_VIEW)
PD10_DUAL_VIEW_DEFAULT_SEED = 1709
PD10_DUAL_VIEW_DEFAULT_BATCH_SIZE = 8192
PD10_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE = 16384
PD10_DUAL_VIEW_DEFAULT_EPOCHS = 20
PD10_DUAL_VIEW_DEFAULT_LR = 1.0e-3
PD10_DUAL_VIEW_DEFAULT_WEIGHT_DECAY = 1.0e-4
PD10_DUAL_VIEW_DEFAULT_HIDDEN_DIM = 128
PD10_DUAL_VIEW_DEFAULT_DROPOUT = 0.05
PD10_DUAL_VIEW_DEFAULT_EARLY_STOP_PATIENCE = 4

PD10_DUAL_VIEW_SUMMARY_FEATURE_NAMES: tuple[str, ...] = (
    "hlt_entropy",
    "offline_entropy",
    "hlt_top1_top2_margin",
    "offline_top1_top2_margin",
    "hlt_max_probability",
    "offline_max_probability",
    "offline_to_hlt_kl",
    "top1_agreement",
)
PD10_DUAL_VIEW_LOGIT_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"hlt_logit_{name}" for name in LABEL_NAMES)
    + tuple(f"offline_logit_{name}" for name in LABEL_NAMES)
    + tuple(f"offline_minus_hlt_logit_{name}" for name in LABEL_NAMES)
    + tuple(f"hlt_probability_{name}" for name in LABEL_NAMES)
    + tuple(f"offline_probability_{name}" for name in LABEL_NAMES)
    + PD10_DUAL_VIEW_SUMMARY_FEATURE_NAMES
)


@dataclass(frozen=True)
class PD10DualViewFeatureBlock:
    """Aligned HLT/offline teacher-logit features for one split."""

    split: str
    features: np.ndarray
    base_logits: np.ndarray
    labels: np.ndarray
    jet_ids: list[JetIdentity]
    hlt_metadata: dict[str, Any] = field(default_factory=dict)
    offline_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        base_logits = np.asarray(self.base_logits, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if features.ndim != 2 or int(features.shape[1]) != PD10_DUAL_VIEW_LOGIT_FEATURE_DIM:
            raise ValueError(
                f"dual-view features must have shape [N, {PD10_DUAL_VIEW_LOGIT_FEATURE_DIM}], "
                f"got {features.shape}"
            )
        if base_logits.shape != (int(features.shape[0]), PD10_NUM_CLASSES):
            raise ValueError(f"base_logits shape {base_logits.shape} does not match features/PD10 classes")
        if int(labels.shape[0]) != int(features.shape[0]):
            raise ValueError("labels length does not match dual-view features")
        if len(self.jet_ids) != int(features.shape[0]):
            raise ValueError("jet_ids length does not match dual-view features")
        identity_labels = np.asarray([int(identity.label) for identity in self.jet_ids], dtype=np.int64)
        if not np.array_equal(labels, identity_labels):
            raise ValueError("labels and jet ids are not aligned")
        if not np.isfinite(features).all():
            raise FloatingPointError("dual-view features contain non-finite values")
        if not np.isfinite(base_logits).all():
            raise FloatingPointError("dual-view base logits contain non-finite values")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "base_logits", base_logits)
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class PD10DualViewLogitTeacherConfig:
    """Training and prediction config for the Step 5 logit-fusion teacher."""

    output_dir: str
    teacher_logit_dir: str
    prediction_output_dir: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    prediction_splits: tuple[str, ...] = field(default_factory=lambda: PD10_TEACHER_LOGIT_SPLITS)
    seed: int = PD10_DUAL_VIEW_DEFAULT_SEED
    batch_size: int = PD10_DUAL_VIEW_DEFAULT_BATCH_SIZE
    eval_batch_size: int = PD10_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE
    num_workers: int = 0
    epochs: int = PD10_DUAL_VIEW_DEFAULT_EPOCHS
    lr: float = PD10_DUAL_VIEW_DEFAULT_LR
    weight_decay: float = PD10_DUAL_VIEW_DEFAULT_WEIGHT_DECAY
    hidden_dim: int = PD10_DUAL_VIEW_DEFAULT_HIDDEN_DIM
    dropout: float = PD10_DUAL_VIEW_DEFAULT_DROPOUT
    early_stop_patience: int = PD10_DUAL_VIEW_DEFAULT_EARLY_STOP_PATIENCE
    grad_clip_norm: float = 1.0
    device: str = "auto"
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_train_jets: int | None = PD10_SPLIT_SIZES["model_train"]
    max_val_jets: int | None = PD10_SPLIT_SIZES["model_val"]
    max_final_test_jets: int | None = PD10_SPLIT_SIZES["final_test"]
    overwrite: bool = False
    skip_existing_predictions: bool = True
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if self.train_split != "model_train":
            raise ValueError("PD10 dual-view teacher may train only on model_train")
        if self.val_split != "model_val":
            raise ValueError("PD10 dual-view teacher may select only on model_val")
        splits = tuple(str(split) for split in self.prediction_splits)
        if not splits:
            raise ValueError("at least one prediction split is required")
        unknown = [split for split in splits if split not in PD10_TEACHER_LOGIT_SPLITS]
        if unknown:
            raise ValueError(f"unknown PD10 dual-view prediction splits: {unknown}")
        if "final_test" in splits and not bool(self.confirm_final_test):
            raise ValueError("Refusing to predict final_test dual-view logits without confirm_final_test=True")
        if int(self.batch_size) <= 0 or int(self.eval_batch_size) <= 0:
            raise ValueError("batch_size and eval_batch_size must be positive")
        if int(self.epochs) < 0:
            raise ValueError("epochs cannot be negative")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for split, value in (
            ("model_train", self.max_train_jets),
            ("model_val", self.max_val_jets),
            ("final_test", self.max_final_test_jets),
        ):
            if value is not None and int(value) > int(PD10_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PD10_SPLIT_SIZES[split]}")
        prediction_output_dir = self.prediction_output_dir or self.teacher_logit_dir
        object.__setattr__(self, "prediction_splits", splits)
        object.__setattr__(self, "prediction_output_dir", str(prediction_output_dir))

    @property
    def model_name(self) -> str:
        return PD10_DUAL_VIEW_LOGIT_MODEL_NAME

    @property
    def teacher_target(self) -> str:
        return PD10_TEACHER_DUAL_VIEW

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "best_model_val.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "last.pt"

    @property
    def prediction_root(self) -> Path:
        return Path(str(self.prediction_output_dir))

    @property
    def prediction_dir(self) -> Path:
        return self.prediction_root / self.model_name


@dataclass(frozen=True)
class PD10DualViewRepresentationCacheConfig:
    """Cache the logit-fusion teacher hidden representation for representation KD controls."""

    checkpoint: str
    teacher_logit_dir: str
    output_dir: str
    splits: tuple[str, ...] = field(default_factory=lambda: PD10_TEACHER_LOGIT_SPLITS)
    batch_size: int = PD10_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE
    device: str = "auto"
    max_model_train_jets: int | None = PD10_SPLIT_SIZES["model_train"]
    max_model_val_jets: int | None = PD10_SPLIT_SIZES["model_val"]
    max_final_test_jets: int | None = PD10_SPLIT_SIZES["final_test"]
    overwrite: bool = False
    skip_existing: bool = True
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        splits = tuple(str(split) for split in self.splits)
        if not splits:
            raise ValueError("at least one representation split is required")
        unknown = [split for split in splits if split not in PD10_TEACHER_LOGIT_SPLITS]
        if unknown:
            raise ValueError(f"unknown PD10 dual-view representation splits: {unknown}")
        if "final_test" in splits and not bool(self.confirm_final_test):
            raise ValueError("Refusing to cache final_test dual-view representations without confirm_final_test=True")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        object.__setattr__(self, "splits", splits)

    @property
    def teacher_target(self) -> str:
        return PD10_TEACHER_DUAL_VIEW

    @property
    def model_name(self) -> str:
        return PD10_DUAL_VIEW_LOGIT_MODEL_NAME

    @property
    def representation_dir(self) -> Path:
        return Path(self.output_dir) / self.model_name


def pd10_dual_view_teacher_dir(*, output_root: str | Path = "checkpoints") -> Path:
    return default_pd10_experiment_layout(output_root=output_root).teacher_dir(PD10_TEACHER_DUAL_VIEW)


def pd10_dual_view_teacher_checkpoint(*, output_root: str | Path = "checkpoints") -> Path:
    return pd10_dual_view_teacher_dir(output_root=output_root) / "best_model_val.pt"


def pd10_dual_view_teacher_logit_cache_dir(*, output_root: str | Path = "checkpoints") -> Path:
    return default_pd10_experiment_layout(output_root=output_root).teacher_logit_cache_dir(PD10_TEACHER_DUAL_VIEW)


def pd10_dual_view_teacher_representation_cache_dir(*, output_root: str | Path = "checkpoints") -> Path:
    return default_pd10_experiment_layout(output_root=output_root).root / "teacher_representations" / PD10_DUAL_VIEW_LOGIT_MODEL_NAME


def _max_jets_for_split(config: PD10DualViewLogitTeacherConfig, split: str) -> int | None:
    if split == "model_train":
        return config.max_train_jets
    if split == "model_val":
        return config.max_val_jets
    if split == "final_test":
        return config.max_final_test_jets
    return None


def _check_logits(name: str, logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or int(logits.shape[1]) != PD10_NUM_CLASSES:
        raise ValueError(f"{name} logits must have shape [N, {PD10_NUM_CLASSES}], got {logits.shape}")
    if not np.isfinite(logits).all():
        raise FloatingPointError(f"{name} logits contain non-finite values")
    return logits


def _entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs.astype(np.float64), 1.0e-12, 1.0)
    return (-np.sum(clipped * np.log(clipped), axis=1)).astype(np.float32)


def _top1_top2_margin(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] < 2:
        return np.zeros((probs.shape[0],), dtype=np.float32)
    top2 = np.partition(probs, kth=probs.shape[1] - 2, axis=1)[:, -2:]
    top2.sort(axis=1)
    return (top2[:, 1] - top2[:, 0]).astype(np.float32)


def pd10_dual_view_summary_features(hlt_logits: np.ndarray, offline_logits: np.ndarray) -> np.ndarray:
    """Build the eight scalar summary features from the plan."""

    hlt_logits = _check_logits("HLT", hlt_logits)
    offline_logits = _check_logits("offline", offline_logits)
    if hlt_logits.shape[0] != offline_logits.shape[0]:
        raise ValueError("HLT/offline logits row count mismatch")
    hlt_probs = softmax_np(hlt_logits)
    offline_probs = softmax_np(offline_logits)
    hlt_top1 = np.argmax(hlt_probs, axis=1)
    offline_top1 = np.argmax(offline_probs, axis=1)
    clipped_hlt = np.clip(hlt_probs.astype(np.float64), 1.0e-12, 1.0)
    clipped_offline = np.clip(offline_probs.astype(np.float64), 1.0e-12, 1.0)
    offline_to_hlt_kl = np.sum(clipped_offline * (np.log(clipped_offline) - np.log(clipped_hlt)), axis=1)
    summary = np.stack(
        [
            _entropy(hlt_probs),
            _entropy(offline_probs),
            _top1_top2_margin(hlt_probs),
            _top1_top2_margin(offline_probs),
            np.max(hlt_probs, axis=1).astype(np.float32),
            np.max(offline_probs, axis=1).astype(np.float32),
            offline_to_hlt_kl.astype(np.float32),
            (hlt_top1 == offline_top1).astype(np.float32),
        ],
        axis=1,
    )
    if summary.shape[1] != PD10_DUAL_VIEW_LOGIT_SUMMARY_DIM:
        raise AssertionError("PD10 dual-view summary feature dimension changed unexpectedly")
    return summary.astype(np.float32, copy=False)


def validate_pd10_dual_view_input_blocks(hlt_block: PredictionBlock, offline_block: PredictionBlock) -> None:
    if hlt_block.metadata.get("contract") != PD10_TEACHER_LOGIT_CACHE_CONTRACT:
        raise ValueError("HLT input block is not a PD10 teacher-logit cache")
    if offline_block.metadata.get("contract") != PD10_TEACHER_LOGIT_CACHE_CONTRACT:
        raise ValueError("offline input block is not a PD10 teacher-logit cache")
    if hlt_block.metadata.get("teacher_target") != PD10_TEACHER_HLT:
        raise ValueError("first dual-view input block must be the HLT teacher")
    if offline_block.metadata.get("teacher_target") != PD10_TEACHER_OFFLINE:
        raise ValueError("second dual-view input block must be the offline teacher")
    validate_prediction_alignment([hlt_block, offline_block])
    _check_logits("HLT", hlt_block.logits)
    _check_logits("offline", offline_block.logits)


def build_pd10_dual_view_feature_block(
    hlt_block: PredictionBlock,
    offline_block: PredictionBlock,
) -> PD10DualViewFeatureBlock:
    validate_pd10_dual_view_input_blocks(hlt_block, offline_block)
    z_hlt = hlt_block.logits.astype(np.float32, copy=False)
    z_offline = offline_block.logits.astype(np.float32, copy=False)
    p_hlt = softmax_np(z_hlt)
    p_offline = softmax_np(z_offline)
    summary = pd10_dual_view_summary_features(z_hlt, z_offline)
    features = np.concatenate(
        [z_hlt, z_offline, z_offline - z_hlt, p_hlt, p_offline, summary],
        axis=1,
    ).astype(np.float32, copy=False)
    base_logits = (0.5 * (z_hlt + z_offline)).astype(np.float32, copy=False)
    return PD10DualViewFeatureBlock(
        split=hlt_block.split,
        features=features,
        base_logits=base_logits,
        labels=hlt_block.labels,
        jet_ids=list(hlt_block.jet_ids),
        hlt_metadata=dict(hlt_block.metadata),
        offline_metadata=dict(offline_block.metadata),
        metadata={
            "feature_names": list(PD10_DUAL_VIEW_LOGIT_FEATURE_NAMES),
            "input_hlt_model_name": hlt_block.model_name,
            "input_offline_model_name": offline_block.model_name,
            "hlt_prediction_content_hash": hlt_block.metadata.get("prediction_content_hash"),
            "offline_prediction_content_hash": offline_block.metadata.get("prediction_content_hash"),
            "hlt_jet_identity_hash": hlt_block.metadata.get("jet_identity_hash"),
            "offline_jet_identity_hash": offline_block.metadata.get("jet_identity_hash"),
        },
    )


def _balanced_subset_indices(labels: np.ndarray, max_rows: int, *, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n_rows = int(labels.shape[0])
    if max_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    max_rows = int(max_rows)
    per_class = max_rows // PD10_NUM_CLASSES
    remainder = max_rows % PD10_NUM_CLASSES
    chosen: list[np.ndarray] = []
    selected_mask = np.zeros((n_rows,), dtype=bool)
    for label in range(PD10_NUM_CLASSES):
        label_indices = np.flatnonzero(labels == label)
        if label_indices.size == 0:
            continue
        target = per_class + (1 if label < remainder else 0)
        take = min(int(target), int(label_indices.size))
        if take <= 0:
            continue
        picked = rng.choice(label_indices, size=take, replace=False)
        chosen.append(picked.astype(np.int64))
        selected_mask[picked] = True
    chosen_indices = np.concatenate(chosen, axis=0) if chosen else np.zeros((0,), dtype=np.int64)
    if chosen_indices.size < max_rows:
        remaining = np.flatnonzero(~selected_mask)
        fill_count = min(max_rows - int(chosen_indices.size), int(remaining.size))
        if fill_count > 0:
            fill = rng.choice(remaining, size=fill_count, replace=False)
            chosen_indices = np.concatenate([chosen_indices, fill.astype(np.int64)], axis=0)
    return np.sort(chosen_indices.astype(np.int64, copy=False))


def limit_pd10_dual_view_feature_block(
    block: PD10DualViewFeatureBlock,
    max_rows: int | None,
    *,
    seed: int,
) -> PD10DualViewFeatureBlock:
    if max_rows is None or int(max_rows) >= int(block.labels.shape[0]):
        return block
    indices = _balanced_subset_indices(block.labels, int(max_rows), seed=int(seed))
    return PD10DualViewFeatureBlock(
        split=block.split,
        features=block.features[indices],
        base_logits=block.base_logits[indices],
        labels=block.labels[indices],
        jet_ids=[block.jet_ids[int(index)] for index in indices],
        hlt_metadata=dict(block.hlt_metadata),
        offline_metadata=dict(block.offline_metadata),
        metadata={
            **dict(block.metadata),
            "subset_selection": {
                "max_rows": int(max_rows),
                "selected_rows": int(indices.shape[0]),
                "seed": int(seed),
            },
        },
    )


def load_pd10_dual_view_feature_block(
    teacher_logit_dir: str | Path,
    split: str,
    *,
    max_rows: int | None = None,
    seed: int = PD10_DUAL_VIEW_DEFAULT_SEED,
    verify_hash: bool = True,
) -> PD10DualViewFeatureBlock:
    hlt_block = load_pd10_teacher_logit_block(teacher_logit_dir, PD10_TEACHER_HLT, split, verify_hash=verify_hash)
    offline_block = load_pd10_teacher_logit_block(
        teacher_logit_dir,
        PD10_TEACHER_OFFLINE,
        split,
        verify_hash=verify_hash,
    )
    block = build_pd10_dual_view_feature_block(hlt_block, offline_block)
    return limit_pd10_dual_view_feature_block(block, max_rows, seed=seed)


class PD10DualViewLogitFusionTeacher:
    """Thin factory wrapper to keep PyTorch optional until construction time."""

    def __new__(
        cls,
        *,
        input_dim: int = PD10_DUAL_VIEW_LOGIT_FEATURE_DIM,
        hidden_dim: int = PD10_DUAL_VIEW_DEFAULT_HIDDEN_DIM,
        num_classes: int = PD10_NUM_CLASSES,
        dropout: float = PD10_DUAL_VIEW_DEFAULT_DROPOUT,
    ):
        torch = require_torch()
        nn = torch.nn

        class _PD10DualViewLogitFusionTeacher(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = {
                    "input_dim": int(input_dim),
                    "hidden_dim": int(hidden_dim),
                    "num_classes": int(num_classes),
                    "dropout": float(dropout),
                }
                self.delta_net = nn.Sequential(
                    nn.LayerNorm(int(input_dim)),
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(num_classes)),
                )
                final_linear = self.delta_net[-1]
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)

            def forward(self, features, base_logits):
                return base_logits + self.delta_net(features)

        return _PD10DualViewLogitFusionTeacher()


def _make_feature_loader(
    block: PD10DualViewFeatureBlock,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
):
    torch = require_torch()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(block.features.astype(np.float32, copy=False)),
        torch.from_numpy(block.base_logits.astype(np.float32, copy=False)),
        torch.from_numpy(block.labels.astype(np.int64, copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _run_dual_view_epoch(
    model,
    loader,
    *,
    device,
    optimizer=None,
    grad_clip_norm: float = 0.0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    total_correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for batch_index, (features, base_logits, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        features = features.to(device, non_blocking=True)
        base_logits = base_logits.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features, base_logits)
        loss_sum = criterion(logits, labels)
        if training:
            loss = loss_sum / max(int(labels.numel()), 1)
            loss.backward()
            if float(grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
        with torch.no_grad():
            total_loss += float(loss_sum.detach().cpu().item())
            total_rows += int(labels.numel())
            total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu().item())
    mean_loss = total_loss / float(max(total_rows, 1))
    return {
        "loss": float(mean_loss),
        "cross_entropy": float(mean_loss),
        "accuracy": float(total_correct / float(total_rows)) if total_rows else 0.0,
        "n_jets": int(total_rows),
    }


def collect_pd10_dual_view_logits_for_feature_block(
    model,
    block: PD10DualViewFeatureBlock,
    *,
    batch_size: int,
    device,
) -> np.ndarray:
    torch = require_torch()
    loader = _make_feature_loader(
        block,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        seed=PD10_DUAL_VIEW_DEFAULT_SEED,
    )
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for features, base_logits, _labels in loader:
            features = features.to(device, non_blocking=True)
            base_logits = base_logits.to(device, non_blocking=True)
            logits = model(features, base_logits)
            rows.append(logits.detach().cpu().numpy().astype(np.float32))
    logits_np = (
        np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        if rows
        else np.zeros((0, PD10_NUM_CLASSES), dtype=np.float32)
    )
    _check_logits("dual-view teacher", logits_np)
    return logits_np


def collect_pd10_dual_view_representations_for_feature_block(
    model,
    block: PD10DualViewFeatureBlock,
    *,
    batch_size: int,
    device,
) -> np.ndarray:
    torch = require_torch()
    if not hasattr(model, "delta_net"):
        raise TypeError("dual-view representation cache requires a PD10DualViewLogitFusionTeacher")
    hidden_net = model.delta_net[:-1]
    loader = _make_feature_loader(
        block,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        seed=PD10_DUAL_VIEW_DEFAULT_SEED,
    )
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for features, _base_logits, _labels in loader:
            features = features.to(device, non_blocking=True)
            hidden = hidden_net(features)
            hidden = torch.nn.functional.normalize(hidden, dim=1)
            rows.append(hidden.detach().cpu().numpy().astype(np.float32))
    representations = (
        np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        if rows
        else np.zeros((0, int(getattr(model, "config", {}).get("hidden_dim", PD10_DUAL_VIEW_DEFAULT_HIDDEN_DIM))), dtype=np.float32)
    )
    if representations.ndim != 2:
        raise ValueError(f"dual-view representations must be 2D [N, D], got {representations.shape}")
    if not np.isfinite(representations).all():
        raise FloatingPointError("dual-view representations contain non-finite values")
    return representations


def _checkpoint_payload(
    model,
    optimizer,
    *,
    config: PD10DualViewLogitTeacherConfig,
    epoch: int,
    metrics: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP5_EXPERIMENT_STEP,
        "teacher_target": PD10_TEACHER_DUAL_VIEW,
        "model_name": config.model_name,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(config),
        "model_config": dict(getattr(model, "config", {})),
        "metrics": dict(metrics),
        "source_metadata": dict(source_metadata),
        "label_names": list(LABEL_NAMES),
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
    }


def load_pd10_dual_view_logit_teacher_checkpoint(checkpoint: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(checkpoint, map_location=device)
    model_config = dict(payload.get("model_config") or {})
    model = PD10DualViewLogitFusionTeacher(
        input_dim=int(model_config.get("input_dim", PD10_DUAL_VIEW_LOGIT_FEATURE_DIM)),
        hidden_dim=int(model_config.get("hidden_dim", PD10_DUAL_VIEW_DEFAULT_HIDDEN_DIM)),
        num_classes=int(model_config.get("num_classes", PD10_NUM_CLASSES)),
        dropout=float(model_config.get("dropout", PD10_DUAL_VIEW_DEFAULT_DROPOUT)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def _input_source_metadata(train_block: PD10DualViewFeatureBlock, val_block: PD10DualViewFeatureBlock) -> dict[str, Any]:
    return {
        "input_contract": PD10_TEACHER_LOGIT_CACHE_CONTRACT,
        "input_hlt_teacher": train_block.metadata.get("input_hlt_model_name"),
        "input_offline_teacher": train_block.metadata.get("input_offline_model_name"),
        "train_split": train_block.split,
        "val_split": val_block.split,
        "train_hlt_prediction_content_hash": train_block.metadata.get("hlt_prediction_content_hash"),
        "train_offline_prediction_content_hash": train_block.metadata.get("offline_prediction_content_hash"),
        "val_hlt_prediction_content_hash": val_block.metadata.get("hlt_prediction_content_hash"),
        "val_offline_prediction_content_hash": val_block.metadata.get("offline_prediction_content_hash"),
        "train_jet_identity_hash": train_block.metadata.get("hlt_jet_identity_hash"),
        "val_jet_identity_hash": val_block.metadata.get("hlt_jet_identity_hash"),
        "allowed_inputs": "HLT_plus_offline_teacher_logits_train_time_privileged",
        "uses_raw_hlt_particles": False,
        "uses_raw_offline_particles": False,
        "uses_hlt_teacher_logits": True,
        "uses_offline_teacher_logits": True,
    }


def train_pd10_dual_view_logit_model_from_features(
    config: PD10DualViewLogitTeacherConfig,
    train_block: PD10DualViewFeatureBlock,
    val_block: PD10DualViewFeatureBlock,
) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    if config.checkpoint_path.exists() and not config.overwrite:
        raise FileExistsError(f"PD10 dual-view teacher checkpoint already exists: {config.checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", {"config": asdict(config)})

    model = PD10DualViewLogitFusionTeacher(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    train_loader = _make_feature_loader(
        train_block,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = _make_feature_loader(
        val_block,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
    )
    source_metadata = _input_source_metadata(train_block, val_block)
    initial_val = _run_dual_view_epoch(
        model,
        val_loader,
        device=device,
        max_batches=config.max_val_batches,
    )
    best_val_ce = float(initial_val["cross_entropy"])
    best_val_accuracy = float(initial_val["accuracy"])
    best_epoch = 0
    initial_row = {"epoch": 0, "train": None, "model_val": initial_val, "zero_delta_initialization": True}
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            config=config,
            epoch=0,
            metrics=initial_row,
            source_metadata=source_metadata,
        ),
        config.checkpoint_path,
    )
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            config=config,
            epoch=0,
            metrics=initial_row,
            source_metadata=source_metadata,
        ),
        config.last_checkpoint_path,
    )

    curves: list[dict[str, Any]] = [initial_row]
    epochs_without_improvement = 0
    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = _run_dual_view_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        val_metrics = _run_dual_view_epoch(
            model,
            val_loader,
            device=device,
            max_batches=config.max_val_batches,
        )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})
        torch.save(
            _checkpoint_payload(
                model,
                optimizer,
                config=config,
                epoch=epoch,
                metrics=row,
                source_metadata=source_metadata,
            ),
            config.last_checkpoint_path,
        )
        val_ce = float(val_metrics["cross_entropy"])
        val_accuracy = float(val_metrics["accuracy"])
        improved = val_ce < best_val_ce or (np.isclose(val_ce, best_val_ce) and val_accuracy > best_val_accuracy)
        if improved:
            best_val_ce = val_ce
            best_val_accuracy = val_accuracy
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    config=config,
                    epoch=epoch,
                    metrics=row,
                    source_metadata=source_metadata,
                ),
                config.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if config.early_stop_patience >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    report = {
        "ok": True,
        "contract": PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP5_EXPERIMENT_STEP,
        "teacher_target": PD10_TEACHER_DUAL_VIEW,
        "model_name": config.model_name,
        "best_epoch": int(best_epoch),
        "best_model_val_cross_entropy": float(best_val_ce),
        "best_model_val_accuracy": float(best_val_accuracy),
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
        "initial_model_val": initial_val,
        "epochs_completed": int(len(curves) - 1),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(config.checkpoint_path),
        "last_checkpoint": str(config.last_checkpoint_path),
        "config": asdict(config),
        "source_metadata": source_metadata,
        "no_final_test_used_for_selection": True,
    }
    save_json(output_dir / "source_metadata.json", source_metadata)
    save_json(output_dir / "model_val_report.json", report)
    return report


def build_pd10_dual_view_prediction_block(
    config: PD10DualViewLogitTeacherConfig,
    split: str,
    *,
    logits: np.ndarray,
    feature_block: PD10DualViewFeatureBlock,
    checkpoint_payload: Mapping[str, Any],
) -> PredictionBlock:
    logits = _check_logits("dual-view teacher", logits)
    if int(logits.shape[0]) != int(feature_block.labels.shape[0]):
        raise ValueError("dual-view prediction row count does not match labels")
    metadata = {
        "contract": PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP5_EXPERIMENT_STEP,
        "teacher_target": PD10_TEACHER_DUAL_VIEW,
        "model_name": config.model_name,
        "model_kind": "pd10_dual_view_logit_teacher",
        "architecture": "logit_fusion_mlp",
        "source_view": "hlt_plus_offline_teacher_logits",
        "split": split,
        "split_expected_size": int(PD10_SPLIT_SIZES[split]),
        "max_jets": _max_jets_for_split(config, split),
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": sha256_file(config.checkpoint_path),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_experiment_step": checkpoint_payload.get("experiment_step"),
        "input_hlt_teacher_model_name": feature_block.metadata.get("input_hlt_model_name"),
        "input_offline_teacher_model_name": feature_block.metadata.get("input_offline_model_name"),
        "input_hlt_prediction_content_hash": feature_block.metadata.get("hlt_prediction_content_hash"),
        "input_offline_prediction_content_hash": feature_block.metadata.get("offline_prediction_content_hash"),
        "input_hlt_jet_identity_hash": feature_block.metadata.get("hlt_jet_identity_hash"),
        "input_offline_jet_identity_hash": feature_block.metadata.get("offline_jet_identity_hash"),
        "label_names": list(LABEL_NAMES),
        "num_classes": PD10_NUM_CLASSES,
        "allowed_inputs": "HLT_plus_offline_train_time_privileged",
        "student_deployment_inputs": "HLT_only",
        "teacher_logits_train_time_only": True,
        "uses_hlt_teacher_logits": True,
        "uses_offline_teacher_logits": True,
        "uses_raw_hlt_particles": False,
        "uses_raw_offline_particles": False,
    }
    return PredictionBlock(
        model_name=config.model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=feature_block.labels,
        jet_ids=list(feature_block.jet_ids),
        metadata=metadata,
    )


def validate_pd10_dual_view_logit_metadata(
    metadata: Mapping[str, Any],
    *,
    split: str | None = None,
) -> None:
    if metadata.get("contract") != PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT:
        raise ValueError("PD10 dual-view teacher logit contract mismatch")
    if metadata.get("experiment_step") != PD10_STEP5_EXPERIMENT_STEP:
        raise ValueError("PD10 dual-view teacher logit step mismatch")
    if metadata.get("teacher_target") != PD10_TEACHER_DUAL_VIEW:
        raise ValueError("PD10 dual-view teacher target mismatch")
    if metadata.get("model_name") != PD10_DUAL_VIEW_LOGIT_MODEL_NAME:
        raise ValueError("PD10 dual-view model name mismatch")
    if split is not None and metadata.get("split") != split:
        raise ValueError(f"split mismatch: {metadata.get('split')} != {split}")
    if int(metadata.get("num_classes", -1)) != PD10_NUM_CLASSES:
        raise ValueError("PD10 dual-view teacher logits must contain 10 classes")
    if int(metadata.get("n_jets", -1)) <= 0:
        raise ValueError("PD10 dual-view teacher logit cache contains no rows")
    if metadata.get("source_view") != "hlt_plus_offline_teacher_logits":
        raise ValueError("PD10 dual-view teacher logits must declare HLT+offline teacher-logit source")
    if not bool(metadata.get("uses_hlt_teacher_logits")) or not bool(metadata.get("uses_offline_teacher_logits")):
        raise ValueError("PD10 dual-view teacher logits must declare both input teacher-logit sources")
    if bool(metadata.get("uses_raw_offline_particles")):
        raise ValueError("PD10 logit-fusion teacher must not declare raw offline particle use")
    if metadata.get("input_hlt_prediction_content_hash") is None:
        raise ValueError("PD10 dual-view teacher logits missing HLT input prediction hash")
    if metadata.get("input_offline_prediction_content_hash") is None:
        raise ValueError("PD10 dual-view teacher logits missing offline input prediction hash")


def load_pd10_dual_view_logit_block(
    prediction_output_dir: str | Path,
    split: str,
    *,
    verify_hash: bool = True,
) -> PredictionBlock:
    block = load_prediction_block(prediction_output_dir, PD10_DUAL_VIEW_LOGIT_MODEL_NAME, split, verify_hash=verify_hash)
    validate_pd10_dual_view_logit_metadata(block.metadata, split=split)
    return block


def _existing_dual_view_metadata_if_valid(
    config: PD10DualViewLogitTeacherConfig,
    split: str,
) -> dict[str, Any] | None:
    npz_path, metadata_path = prediction_paths(config.prediction_root, config.model_name, split)
    if not npz_path.exists() or not metadata_path.exists():
        return None
    if not config.skip_existing_predictions or config.overwrite:
        return None
    block = load_pd10_dual_view_logit_block(config.prediction_root, split)
    return dict(block.metadata)


def _max_jets_for_dual_view_representation_split(config: PD10DualViewRepresentationCacheConfig, split: str) -> int | None:
    if split == "model_train":
        return config.max_model_train_jets
    if split == "model_val":
        return config.max_model_val_jets
    if split == "final_test":
        return config.max_final_test_jets
    return None


def _existing_dual_view_representation_metadata_if_valid(
    config: PD10DualViewRepresentationCacheConfig,
    split: str,
    *,
    representation_dim: int,
) -> dict[str, Any] | None:
    if not config.skip_existing or config.overwrite:
        return None
    npz_path = config.representation_dir / f"{split}_representations.npz"
    meta_path = config.representation_dir / f"{split}_representations_metadata.json"
    if not (npz_path.exists() and meta_path.exists()):
        return None
    block = load_pd10_teacher_representation_block(config.output_dir, config.teacher_target, split)
    if int(block.representations.shape[1]) != int(representation_dim):
        return None
    return dict(block.metadata)


def cache_pd10_dual_view_teacher_representations(config: PD10DualViewRepresentationCacheConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    model, payload = load_pd10_dual_view_logit_teacher_checkpoint(config.checkpoint, device=device)
    representation_dim = int(getattr(model, "config", {}).get("hidden_dim", PD10_DUAL_VIEW_DEFAULT_HIDDEN_DIM))
    rep_config = PD10TeacherRepresentationCacheConfig(
        teacher_target=PD10_TEACHER_DUAL_VIEW,
        output_dir=config.output_dir,
        splits=config.splits,
        representation_dim=representation_dim,
        overwrite=bool(config.overwrite),
        skip_existing=bool(config.skip_existing),
        confirm_final_test=bool(config.confirm_final_test),
    )
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    for split in config.splits:
        existing = _existing_dual_view_representation_metadata_if_valid(
            config,
            split,
            representation_dim=representation_dim,
        )
        if existing is not None:
            rows.append(existing)
            split_reports[split] = {"skipped_existing": True, "representation_metadata": existing}
            continue
        feature_block = load_pd10_dual_view_feature_block(
            config.teacher_logit_dir,
            split,
            max_rows=_max_jets_for_dual_view_representation_split(config, split),
            seed=PD10_DUAL_VIEW_DEFAULT_SEED,
        )
        representations = collect_pd10_dual_view_representations_for_feature_block(
            model,
            feature_block,
            batch_size=config.batch_size,
            device=device,
        )
        block = build_pd10_teacher_representation_block(
            rep_config,
            split,
            representations=representations,
            labels=feature_block.labels,
            jet_ids=feature_block.jet_ids,
            source_metadata={
                **dict(feature_block.metadata),
                "representation_source": "dual_view_logit_fusion_hidden_pre_classifier",
                "teacher_logit_dir": str(config.teacher_logit_dir),
                "max_jets": _max_jets_for_dual_view_representation_split(config, split),
            },
            extra_metadata={
                "checkpoint": str(config.checkpoint),
                "checkpoint_sha256": sha256_file(config.checkpoint),
                "checkpoint_epoch": payload.get("epoch"),
                "checkpoint_experiment_step": payload.get("experiment_step"),
            },
        )
        metadata = save_pd10_teacher_representation_block(block, config.output_dir, overwrite=bool(config.overwrite))
        rows.append(metadata)
        split_reports[split] = {"skipped_existing": False, "representation_metadata": metadata}
    manifest = write_pd10_teacher_representation_manifest(rep_config, rows)
    return {
        "ok": True,
        "teacher_target": PD10_TEACHER_DUAL_VIEW,
        "model_name": config.model_name,
        "checkpoint": str(config.checkpoint),
        "teacher_logit_dir": str(config.teacher_logit_dir),
        "representation_output_dir": str(config.output_dir),
        "representation_dim": int(representation_dim),
        "splits": list(config.splits),
        "split_reports": split_reports,
        "manifest": manifest,
    }


def cache_pd10_dual_view_teacher_logits(
    config: PD10DualViewLogitTeacherConfig,
    *,
    model=None,
    checkpoint_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    device = resolve_device(config.device)
    if model is None:
        model, payload = load_pd10_dual_view_logit_teacher_checkpoint(config.checkpoint_path, device=device)
    else:
        payload = dict(checkpoint_payload or {})
        model = model.to(device)
        model.eval()
    prediction_rows: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    for split in config.prediction_splits:
        existing = _existing_dual_view_metadata_if_valid(config, split)
        if existing is not None:
            split_reports[split] = {"skipped_existing": True, "metadata": existing}
            prediction_rows.append(existing)
            continue
        feature_block = load_pd10_dual_view_feature_block(
            config.teacher_logit_dir,
            split,
            max_rows=_max_jets_for_split(config, split),
            seed=int(config.seed) + 2003 * (PD10_SPLIT_ORDER.index(split) + 1),
        )
        logits = collect_pd10_dual_view_logits_for_feature_block(
            model,
            feature_block,
            batch_size=config.eval_batch_size,
            device=device,
        )
        block = build_pd10_dual_view_prediction_block(
            config,
            split,
            logits=logits,
            feature_block=feature_block,
            checkpoint_payload=payload,
        )
        metadata = save_prediction_block(block, config.prediction_root, overwrite=bool(config.overwrite))
        validate_pd10_dual_view_logit_metadata(metadata, split=split)
        split_reports[split] = {"skipped_existing": False, "metadata": metadata}
        prediction_rows.append(metadata)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    manifest = {
        "ok": True,
        "contract": PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP5_EXPERIMENT_STEP,
        "teacher_target": PD10_TEACHER_DUAL_VIEW,
        "model_name": config.model_name,
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": sha256_file(config.checkpoint_path),
        "teacher_logit_input_dir": str(config.teacher_logit_dir),
        "prediction_output_dir": str(config.prediction_root),
        "teacher_logit_dir": str(config.prediction_dir),
        "splits": list(config.prediction_splits),
        "split_sizes": {split: int(PD10_SPLIT_SIZES[split]) for split in PD10_SPLIT_ORDER},
        "prediction_rows": prediction_rows,
        "split_reports": split_reports,
        "config": asdict(config),
    }
    save_json(config.prediction_dir / PD10_TEACHER_LOGIT_CACHE_MANIFEST, manifest)
    save_json(config.prediction_dir / PD10_TEACHER_LOGIT_CACHE_REPORT, manifest)
    return manifest


def train_pd10_dual_view_logit_teacher(config: PD10DualViewLogitTeacherConfig) -> dict[str, Any]:
    train_block = load_pd10_dual_view_feature_block(
        config.teacher_logit_dir,
        config.train_split,
        max_rows=config.max_train_jets,
        seed=int(config.seed) + 11,
    )
    val_block = load_pd10_dual_view_feature_block(
        config.teacher_logit_dir,
        config.val_split,
        max_rows=config.max_val_jets,
        seed=int(config.seed) + 17,
    )
    report = train_pd10_dual_view_logit_model_from_features(config, train_block, val_block)
    device = resolve_device(config.device)
    model, payload = load_pd10_dual_view_logit_teacher_checkpoint(config.checkpoint_path, device=device)
    prediction_manifest = cache_pd10_dual_view_teacher_logits(config, model=model, checkpoint_payload=payload)
    final_report = {**report, "prediction_manifest": prediction_manifest}
    save_json(Path(config.output_dir) / "run_report.json", final_report)
    save_json(Path(config.output_dir) / "model_val_report.json", final_report)
    return final_report


__all__ = [
    "PD10_DUAL_VIEW_DEFAULT_BATCH_SIZE",
    "PD10_DUAL_VIEW_DEFAULT_DROPOUT",
    "PD10_DUAL_VIEW_DEFAULT_EARLY_STOP_PATIENCE",
    "PD10_DUAL_VIEW_DEFAULT_EPOCHS",
    "PD10_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE",
    "PD10_DUAL_VIEW_DEFAULT_HIDDEN_DIM",
    "PD10_DUAL_VIEW_DEFAULT_LR",
    "PD10_DUAL_VIEW_DEFAULT_SEED",
    "PD10_DUAL_VIEW_DEFAULT_WEIGHT_DECAY",
    "PD10_DUAL_VIEW_LOGIT_FEATURE_DIM",
    "PD10_DUAL_VIEW_LOGIT_FEATURE_NAMES",
    "PD10_DUAL_VIEW_LOGIT_MODEL_NAME",
    "PD10_DUAL_VIEW_LOGIT_SUMMARY_DIM",
    "PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT",
    "PD10_DUAL_VIEW_SUMMARY_FEATURE_NAMES",
    "PD10_STEP5_EXPERIMENT_STEP",
    "PD10DualViewFeatureBlock",
    "PD10DualViewLogitFusionTeacher",
    "PD10DualViewRepresentationCacheConfig",
    "PD10DualViewLogitTeacherConfig",
    "build_pd10_dual_view_feature_block",
    "build_pd10_dual_view_prediction_block",
    "cache_pd10_dual_view_teacher_logits",
    "cache_pd10_dual_view_teacher_representations",
    "classification_metrics_from_logits",
    "collect_pd10_dual_view_logits_for_feature_block",
    "collect_pd10_dual_view_representations_for_feature_block",
    "limit_pd10_dual_view_feature_block",
    "load_pd10_dual_view_feature_block",
    "load_pd10_dual_view_logit_block",
    "load_pd10_dual_view_logit_teacher_checkpoint",
    "pd10_dual_view_summary_features",
    "pd10_dual_view_teacher_checkpoint",
    "pd10_dual_view_teacher_dir",
    "pd10_dual_view_teacher_logit_cache_dir",
    "pd10_dual_view_teacher_representation_cache_dir",
    "train_pd10_dual_view_logit_model_from_features",
    "train_pd10_dual_view_logit_teacher",
    "validate_pd10_dual_view_input_blocks",
    "validate_pd10_dual_view_logit_metadata",
]

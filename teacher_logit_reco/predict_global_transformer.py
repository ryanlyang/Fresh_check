"""Prediction collection for teacher-logit Global Transformer reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import (
    STACK_SPLITS,
    PredictionBlock,
    load_prediction_block,
    prediction_paths,
    save_prediction_block,
    softmax_np,
)
from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetIdentity, JetView

from .global_transformer import GlobalTransformerReconstructor
from .reconstructor_builders import (
    infer_reconstructor_architecture_from_payload,
    load_teacher_logit_reconstructor_checkpoint,
)
from .teachers import assert_teacher_frozen, load_frozen_teacher, normalize_teacher_architecture
from .train_global_transformer import EXPERIMENT_STEP as TRAIN_EXPERIMENT_STEP
from .train_global_transformer import source_metadata


PREDICT_EXPERIMENT_STEP = "teacher_logit_reco_step6_global_transformer_predictions"


@dataclass
class TeacherLogitGlobalTransformerPredictionConfig:
    """Configuration for Step 6 prediction block generation."""

    output_dir: str
    hlt_cache_dir: str
    reconstructor_checkpoint: str
    prediction_dir: str | None = None
    teacher_checkpoint: str | None = None
    teacher_architecture: str | None = None
    model_name: str | None = None
    splits: list[str] = field(default_factory=lambda: list(STACK_SPLITS))
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    max_jets_per_split: int | None = None
    overwrite_predictions: bool = False
    skip_existing_predictions: bool = True
    confirm_final_test: bool = False
    max_constits: int = 128
    teacher_weight_threshold: float = 0.0
    strict_checkpoint: bool = True

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers must be non-negative")
        if self.max_jets_per_split is not None and int(self.max_jets_per_split) < 0:
            raise ValueError("max_jets_per_split must be non-negative when provided")
        if "final_test" in list(self.splits) and not bool(self.confirm_final_test):
            raise ValueError("Refusing to generate final_test predictions without confirm_final_test=True")

    @property
    def resolved_prediction_dir(self) -> Path:
        if self.prediction_dir is not None:
            return Path(self.prediction_dir)
        return Path(self.output_dir) / "predictions"


class HLTPredictionDataset:
    """Torch dataset over one cached fixed-HLT view, preserving jet identities."""

    def __init__(self, view: JetView, *, max_jets: int | None = None) -> None:
        require_torch()
        limit = len(view.labels) if max_jets is None else min(int(max_jets), len(view.labels))
        self.tokens = np.asarray(view.tokens[:limit], dtype=np.float32)
        self.mask = np.asarray(view.mask[:limit], dtype=bool)
        self.labels = np.asarray(view.labels[:limit], dtype=np.int64)
        self.jet_ids = list(view.jet_ids[:limit])
        self.split = view.split
        self.metadata = dict(view.metadata)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.tokens[index], self.mask[index], self.labels[index], self.jet_ids[index]


def collate_hlt_prediction(samples):
    torch = require_torch()
    return {
        "hlt_tokens": torch.from_numpy(np.stack([row[0] for row in samples], axis=0)).float(),
        "hlt_mask": torch.from_numpy(np.stack([row[1] for row in samples], axis=0)).bool(),
        "labels": torch.as_tensor([row[2] for row in samples], dtype=torch.long),
        "jet_ids": [row[3] for row in samples],
    }


def make_hlt_prediction_loader(
    dataset: HLTPredictionDataset,
    *,
    batch_size: int,
    num_workers: int,
):
    torch = require_torch()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_hlt_prediction,
        pin_memory=torch.cuda.is_available(),
    )


def load_global_transformer_reconstructor_checkpoint(
    checkpoint_path: str | Path,
    *,
    device,
    strict: bool = True,
) -> tuple[GlobalTransformerReconstructor, Dict[str, Any]]:
    """Load a Step 5 Global Transformer reconstructor checkpoint."""

    model, payload = load_teacher_logit_reconstructor_checkpoint(
        checkpoint_path,
        device=device,
        strict=bool(strict),
        expected_architecture="global_transformer",
    )
    return model, payload


def default_model_name_for_teacher_architecture(architecture: str | None) -> str:
    arch = normalize_teacher_architecture(architecture)
    return f"gt_reco_to_{arch}_teacher"


def teacher_checkpoint_from_payload(
    payload: Mapping[str, Any],
    *,
    override_checkpoint: str | None = None,
) -> str:
    if override_checkpoint:
        return str(override_checkpoint)
    teacher_metadata = dict(payload.get("teacher_metadata") or {})
    if teacher_metadata.get("checkpoint_path"):
        return str(teacher_metadata["checkpoint_path"])
    train_config = dict(payload.get("config") or {})
    if train_config.get("teacher_checkpoint"):
        return str(train_config["teacher_checkpoint"])
    raise KeyError("Teacher checkpoint path was not provided and could not be recovered from reconstructor checkpoint")


def teacher_architecture_from_payload(
    payload: Mapping[str, Any],
    *,
    override_architecture: str | None = None,
) -> str | None:
    if override_architecture:
        return normalize_teacher_architecture(override_architecture)
    teacher_metadata = dict(payload.get("teacher_metadata") or {})
    if teacher_metadata.get("architecture"):
        return normalize_teacher_architecture(str(teacher_metadata["architecture"]))
    train_config = dict(payload.get("config") or {})
    if train_config.get("teacher_architecture"):
        return normalize_teacher_architecture(str(train_config["teacher_architecture"]))
    return None


def evaluate_teacher_logit_reco_model(
    model_name: str,
    reconstructor,
    teacher,
    view: JetView,
    *,
    batch_size: int,
    num_workers: int,
    device,
    amp: bool = True,
    max_jets: int | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> PredictionBlock:
    """Evaluate HLT -> reconstructor -> frozen teacher on one split."""

    torch = require_torch()
    dataset = HLTPredictionDataset(view, max_jets=max_jets)
    loader = make_hlt_prediction_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    reconstructor.eval()
    teacher.model.eval()
    assert_teacher_frozen(teacher)
    autocast_enabled = bool(amp and getattr(device, "type", None) == "cuda")
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    jet_ids: list[JetIdentity] = []

    with torch.no_grad():
        for batch in loader:
            hlt_tokens = batch["hlt_tokens"].to(device=device, non_blocking=True)
            hlt_mask = batch["hlt_mask"].to(device=device, non_blocking=True)
            labels = batch["labels"].to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                reco_view = reconstructor(
                    hlt_tokens,
                    hlt_mask,
                    labels=labels,
                    jet_ids=batch["jet_ids"],
                    split=dataset.split,
                )
                logits = teacher.forward_soft_view(reco_view)
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
            jet_ids.extend(batch["jet_ids"])

    if not logits_rows:
        raise ValueError(f"No predictions were produced for {model_name}/{view.split}")
    logits_np = np.concatenate(logits_rows, axis=0)
    labels_np = np.concatenate(labels_rows, axis=0)
    metadata = {
        "experiment_step": PREDICT_EXPERIMENT_STEP,
        "model_kind": "teacher_logit_global_transformer_reco",
        "teacher_architecture": teacher.metadata.get("architecture"),
        "teacher_metadata": dict(teacher.metadata),
        "hlt_content_hash": view.metadata.get("hlt_content_hash"),
        "allowed_inputs": "cached_fixed_hlt_only_then_reconstructed_soft_view_to_frozen_teacher",
        "training_step": TRAIN_EXPERIMENT_STEP,
        "max_jets": None if max_jets is None else int(max_jets),
    }
    metadata.update(dict(checkpoint_metadata or {}))
    return PredictionBlock(
        model_name=model_name,
        split=view.split,
        logits=logits_np,
        probs=softmax_np(logits_np),
        labels=labels_np,
        jet_ids=jet_ids,
        metadata=metadata,
    )


def collect_teacher_logit_global_transformer_predictions(
    config: TeacherLogitGlobalTransformerPredictionConfig,
    *,
    reconstructor=None,
    teacher=None,
    hlt_views: Mapping[str, JetView] | None = None,
) -> Dict[str, Any]:
    """Generate fusion-compatible prediction blocks for requested splits."""

    torch = require_torch()
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = config.resolved_prediction_dir
    reports: Dict[str, Any] = {}

    payload: Dict[str, Any] = {}
    if reconstructor is None:
        reconstructor, payload = load_global_transformer_reconstructor_checkpoint(
            config.reconstructor_checkpoint,
            device=device,
            strict=bool(config.strict_checkpoint),
        )
    else:
        reconstructor = reconstructor.to(device).eval()

    if teacher is None:
        teacher_checkpoint = teacher_checkpoint_from_payload(payload, override_checkpoint=config.teacher_checkpoint)
        teacher_architecture = teacher_architecture_from_payload(
            payload,
            override_architecture=config.teacher_architecture,
        )
        teacher = load_frozen_teacher(
            teacher_checkpoint,
            architecture=teacher_architecture,
            device=str(device),
            max_constits=int(config.max_constits),
            weight_threshold=float(config.teacher_weight_threshold),
        )
    else:
        teacher.model = teacher.model.to(device).eval()
        teacher.device = device
        assert_teacher_frozen(teacher)

    model_name = config.model_name or default_model_name_for_teacher_architecture(teacher.metadata.get("architecture"))
    checkpoint_metadata = {
        "reconstructor_checkpoint": str(config.reconstructor_checkpoint),
        "reconstructor_architecture": infer_reconstructor_architecture_from_payload(payload),
        "reconstructor_checkpoint_epoch": payload.get("epoch"),
        "reconstructor_experiment_step": payload.get("experiment_step"),
        "reconstructor_model_config": dict(payload.get("model_config") or {}),
        "reconstructor_loss_config": dict(payload.get("loss_config") or {}),
        "teacher_checkpoint": teacher.metadata.get("checkpoint_path"),
        "source": source_metadata(),
    }

    reports[model_name] = {}
    for split in list(config.splits):
        npz_path, _ = prediction_paths(prediction_dir, model_name, split)
        if npz_path.exists() and config.skip_existing_predictions and not config.overwrite_predictions:
            reports[model_name][split] = load_prediction_block(prediction_dir, model_name, split).metadata
            continue
        view = hlt_views[split] if hlt_views is not None and split in hlt_views else load_cached_hlt_view(config.hlt_cache_dir, split)
        block = evaluate_teacher_logit_reco_model(
            model_name,
            reconstructor,
            teacher,
            view,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=device,
            amp=config.amp,
            max_jets=config.max_jets_per_split,
            checkpoint_metadata=checkpoint_metadata,
        )
        reports[model_name][split] = save_prediction_block(
            block,
            prediction_dir,
            overwrite=bool(config.overwrite_predictions),
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    output = {
        "experiment_step": PREDICT_EXPERIMENT_STEP,
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "splits": list(config.splits),
        "config": asdict(config),
        "teacher": dict(teacher.metadata),
        "reports": reports,
        "leakage_rule": (
            "Prediction generation loads cached fixed-HLT views only. Offline constituents are not loaded; "
            "the frozen teacher sees only the reconstructed soft view produced from HLT tokens."
        ),
    }
    save_json(output_dir / "prediction_collection_report.json", output)
    return output

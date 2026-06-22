"""Version B privileged-offline distillation for subtoken HLT taggers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import (
    LABEL_NAMES,
    JetView,
    RAW_TOKEN_DIM,
    load_offline_view,
    load_split_manifest,
    manifest_hash,
)

from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions
from teacher_logit_reco.set_matching.train import source_metadata
from teacher_logit_reco.teachers import load_frozen_teacher

from .classifier import SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP, SubtokenParticleTransformerClassifier
from .classifier import build_subtoken_particle_transformer_classifier
from .config import SUBTOKEN_PART_VERSION_B
from .masked import (
    SUBTOKEN_PART_MASKED_STEP,
    MaskedSubtokenPredictionHead,
    build_masked_subtoken_targets,
    compute_masked_subtoken_loss,
    normalize_masked_subtoken_target_mode,
    sample_masked_subtoken_modality_mask,
)
from .residuals import (
    SUBTOKEN_PART_RESIDUAL_STEP,
    ModalityResidualHead,
    compute_gate_residual_regularization,
    compute_modality_residual_loss,
    compute_modality_residual_targets,
    normalize_modality_residual_target_mode,
)
from .train import (
    SUBTOKEN_PART_LOWER_IS_BETTER_SELECTION_METRICS,
    SubtokenHLTJetDataset,
    SubtokenTaggerTrainConfig,
    _finite_float,
    _selection_metric_requires_predictions,
    _selection_score,
    _write_best_metrics_csv,
    _write_epoch_metrics_csv,
    _write_per_class_csv,
    _write_summary_csv,
    load_subtoken_tagger_checkpoint,
    make_subtoken_hlt_loader,
    run_subtoken_tagger_epoch,
)


SUBTOKEN_PART_DISTILL_STEP = "subtoken_part_step16_offline_teacher_distillation"
SUBTOKEN_PART_DISTILL_CONTRACT = "hlt_student_privileged_offline_teacher_distillation_v1"


@dataclass
class SubtokenDistillTrainConfig(SubtokenTaggerTrainConfig):
    """Configuration for Version B HLT student training with an offline teacher."""

    manifest_path: str = ""
    data_dir: str | None = None
    offline_teacher_checkpoint: str = ""
    offline_teacher_architecture: str = "part"
    teacher_device: str | None = None
    teacher_max_constits: int = 128
    teacher_weight_threshold: float = 0.0
    strict_teacher_checkpoint: bool = True
    distillation_temperature: float = 2.0
    distillation_weight: float = 0.5
    classification_weight: float = 1.0
    modality_residual_weight: float = 0.0
    modality_residual_target_mode: str = "jet"
    modality_residual_huber_beta: float = 1.0
    gate_residual_regularization_weight: float = 0.0
    gate_residual_temperature: float = 1.0
    masked_subtoken_weight: float = 0.0
    masked_subtoken_target_mode: str = "hlt_self"
    masked_subtoken_probability: float = 0.15
    masked_subtoken_huber_beta: float = 1.0
    masked_subtoken_max_match_delta_r: float | None = 0.4
    verify_offline_label_branches: bool = False
    read_chunk_size: int = 50_000

    def __post_init__(self) -> None:
        super().__post_init__()
        self.manifest_path = str(self.manifest_path)
        self.offline_teacher_checkpoint = str(self.offline_teacher_checkpoint)
        self.offline_teacher_architecture = str(self.offline_teacher_architecture or "part")
        if not self.manifest_path:
            raise ValueError("manifest_path is required for offline-teacher distillation")
        if not self.offline_teacher_checkpoint:
            raise ValueError("offline_teacher_checkpoint is required for offline-teacher distillation")
        self.data_dir = None if self.data_dir is None else str(self.data_dir)
        self.teacher_device = None if self.teacher_device is None else str(self.teacher_device)
        self.teacher_max_constits = int(self.teacher_max_constits)
        if self.teacher_max_constits <= 0:
            raise ValueError("teacher_max_constits must be positive")
        self.teacher_weight_threshold = float(self.teacher_weight_threshold)
        if self.teacher_weight_threshold < 0.0:
            raise ValueError("teacher_weight_threshold cannot be negative")
        self.distillation_temperature = float(self.distillation_temperature)
        if self.distillation_temperature <= 0.0:
            raise ValueError("distillation_temperature must be positive")
        self.distillation_weight = float(self.distillation_weight)
        if self.distillation_weight < 0.0:
            raise ValueError("distillation_weight cannot be negative")
        self.classification_weight = float(self.classification_weight)
        if self.classification_weight < 0.0:
            raise ValueError("classification_weight cannot be negative")
        self.modality_residual_weight = float(self.modality_residual_weight)
        if self.modality_residual_weight < 0.0:
            raise ValueError("modality_residual_weight cannot be negative")
        self.modality_residual_target_mode = normalize_modality_residual_target_mode(self.modality_residual_target_mode)
        self.modality_residual_huber_beta = float(self.modality_residual_huber_beta)
        if self.modality_residual_huber_beta <= 0.0:
            raise ValueError("modality_residual_huber_beta must be positive")
        self.gate_residual_regularization_weight = float(self.gate_residual_regularization_weight)
        if self.gate_residual_regularization_weight < 0.0:
            raise ValueError("gate_residual_regularization_weight cannot be negative")
        self.gate_residual_temperature = float(self.gate_residual_temperature)
        if self.gate_residual_temperature <= 0.0:
            raise ValueError("gate_residual_temperature must be positive")
        self.masked_subtoken_weight = float(self.masked_subtoken_weight)
        if self.masked_subtoken_weight < 0.0:
            raise ValueError("masked_subtoken_weight cannot be negative")
        self.masked_subtoken_target_mode = normalize_masked_subtoken_target_mode(self.masked_subtoken_target_mode)
        self.masked_subtoken_probability = float(self.masked_subtoken_probability)
        if self.masked_subtoken_probability < 0.0 or self.masked_subtoken_probability >= 1.0:
            raise ValueError("masked_subtoken_probability must be in [0, 1)")
        self.masked_subtoken_huber_beta = float(self.masked_subtoken_huber_beta)
        if self.masked_subtoken_huber_beta <= 0.0:
            raise ValueError("masked_subtoken_huber_beta must be positive")
        if self.masked_subtoken_max_match_delta_r is not None:
            self.masked_subtoken_max_match_delta_r = float(self.masked_subtoken_max_match_delta_r)
            if self.masked_subtoken_max_match_delta_r <= 0.0:
                raise ValueError("masked_subtoken_max_match_delta_r must be positive when set")
        if (
            self.classification_weight == 0.0
            and self.distillation_weight == 0.0
            and self.modality_residual_weight == 0.0
            and self.gate_residual_regularization_weight == 0.0
            and self.masked_subtoken_weight == 0.0
        ):
            raise ValueError("at least one Version B training loss weight must be positive")
        self.read_chunk_size = int(self.read_chunk_size)
        if self.read_chunk_size <= 0:
            raise ValueError("read_chunk_size must be positive")

    def model_config(self):
        return replace(super().model_config(), version=SUBTOKEN_PART_VERSION_B)


class SubtokenDistillJetDataset:
    """Paired HLT/offline dataset for teacher distillation.

    The student receives only ``tokens``/``mask`` from the HLT cache.  The
    offline tensors are present only so the frozen teacher can produce training
    targets inside the Step 16 training loop.
    """

    def __init__(
        self,
        hlt_view: JetView,
        offline_view: JetView,
        *,
        label_filter: Sequence[int],
        label_names: Sequence[str],
        max_jets: int | None = None,
        verify_identity: bool = True,
    ) -> None:
        if hlt_view.metadata.get("view") not in (None, "fixed_hlt"):
            raise ValueError(f"Expected fixed_hlt cached view, got {hlt_view.metadata.get('view')!r}")
        if offline_view.metadata.get("view") not in (None, "offline"):
            raise ValueError(f"Expected offline view, got {offline_view.metadata.get('view')!r}")
        if hlt_view.split != offline_view.split:
            raise ValueError(f"HLT/offline split mismatch: {hlt_view.split!r} != {offline_view.split!r}")
        if len(hlt_view.labels) != len(offline_view.labels):
            raise ValueError("HLT/offline views must have the same number of rows before filtering")
        if not np.array_equal(np.asarray(hlt_view.labels), np.asarray(offline_view.labels)):
            raise ValueError("HLT/offline labels differ before filtering")
        if verify_identity and list(hlt_view.jet_ids) != list(offline_view.jet_ids):
            raise ValueError("HLT/offline jet identities differ; distillation requires one-to-one paired rows")

        labels = np.asarray(hlt_view.labels, dtype=np.int64)
        label_filter = tuple(int(label) for label in label_filter)
        label_names = tuple(str(name) for name in label_names)
        if len(label_filter) != len(label_names):
            raise ValueError("label_filter and label_names must have the same length")
        keep = np.isin(labels, np.asarray(label_filter, dtype=np.int64))
        hlt_tokens = np.asarray(hlt_view.tokens, dtype=np.float32)[keep]
        hlt_mask = np.asarray(hlt_view.mask, dtype=bool)[keep]
        offline_tokens = np.asarray(offline_view.tokens, dtype=np.float32)[keep]
        offline_mask = np.asarray(offline_view.mask, dtype=bool)[keep]
        kept_labels = labels[keep]
        jet_ids = [jet_id for jet_id, should_keep in zip(hlt_view.jet_ids, keep) if bool(should_keep)]

        remap = {source_label: index for index, source_label in enumerate(label_filter)}
        remapped = np.asarray([remap[int(label)] for label in kept_labels], dtype=np.int64)
        if max_jets is not None:
            limit = min(int(max_jets), int(remapped.shape[0]))
            hlt_tokens = hlt_tokens[:limit]
            hlt_mask = hlt_mask[:limit]
            offline_tokens = offline_tokens[:limit]
            offline_mask = offline_mask[:limit]
            remapped = remapped[:limit]
            jet_ids = jet_ids[:limit]

        self.tokens = np.asarray(hlt_tokens, dtype=np.float32)
        self.mask = np.asarray(hlt_mask, dtype=bool)
        self.offline_tokens = np.asarray(offline_tokens, dtype=np.float32)
        self.offline_mask = np.asarray(offline_mask, dtype=bool)
        self.labels = remapped
        self.jet_ids = jet_ids
        self.split = hlt_view.split
        self.label_names = label_names
        self.label_filter = label_filter
        self.metadata = {
            "split": hlt_view.split,
            "source_view": "fixed_hlt",
            "teacher_source_view": "offline",
            "n_jets": int(remapped.shape[0]),
            "raw_token_dim": int(self.tokens.shape[-1]),
            "max_constits": int(self.tokens.shape[1]),
            "offline_raw_token_dim": int(self.offline_tokens.shape[-1]),
            "offline_max_constits": int(self.offline_tokens.shape[1]),
            "label_filter": list(label_filter),
            "label_names": list(label_names),
            "label_counts": self.label_counts(),
            "max_jets_limit": None if max_jets is None else int(max_jets),
            "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
            "jet_identity_hash": hlt_view.metadata.get("jet_identity_hash"),
            "source_manifest_hash": offline_view.metadata.get("source_manifest_hash")
            or hlt_view.metadata.get("source_manifest_hash"),
            "offline_view_metadata": {
                "data_dir": offline_view.metadata.get("data_dir"),
                "tree_name": offline_view.metadata.get("tree_name"),
                "max_constits": offline_view.metadata.get("max_constits"),
            },
        }

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "offline_tokens": self.offline_tokens[index],
            "offline_mask": self.offline_mask[index],
            "labels": self.labels[index],
            "indices": np.int64(index),
        }

    def label_counts(self) -> dict[str, int]:
        return {
            self.label_names[index]: int(np.sum(self.labels == index))
            for index in range(len(self.label_names))
        }


def collate_subtoken_distill_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    torch = require_torch()
    tokens = np.stack([sample["tokens"] for sample in samples], axis=0).astype(np.float32, copy=False)
    mask = np.stack([sample["mask"] for sample in samples], axis=0).astype(bool, copy=False)
    offline_tokens = np.stack([sample["offline_tokens"] for sample in samples], axis=0).astype(np.float32, copy=False)
    offline_mask = np.stack([sample["offline_mask"] for sample in samples], axis=0).astype(bool, copy=False)
    labels = np.asarray([sample["labels"] for sample in samples], dtype=np.int64)
    indices = np.asarray([sample["indices"] for sample in samples], dtype=np.int64)
    return {
        "tokens": torch.from_numpy(tokens).float(),
        "mask": torch.from_numpy(mask).bool(),
        "offline_tokens": torch.from_numpy(offline_tokens).float(),
        "offline_mask": torch.from_numpy(offline_mask).bool(),
        "labels": torch.from_numpy(labels).long(),
        "indices": torch.from_numpy(indices).long(),
    }


def make_subtoken_distill_loader(
    dataset: SubtokenDistillJetDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_subtoken_distill_batch,
        generator=generator,
    )


def _move_distill_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("tokens", "mask", "offline_tokens", "offline_mask", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _load_subtoken_distill_dataset(
    config: SubtokenDistillTrainConfig,
    split: str,
    *,
    max_jets: int | None,
) -> tuple[SubtokenDistillJetDataset, str]:
    manifest = load_split_manifest(config.manifest_path)
    manifest_sha = manifest_hash(manifest)
    hlt_view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    hlt_manifest_sha = hlt_view.metadata.get("source_manifest_hash")
    if hlt_manifest_sha not in (None, manifest_sha):
        raise ValueError(f"HLT cache source_manifest_hash {hlt_manifest_sha} does not match {manifest_sha}")
    offline_view = load_offline_view(
        manifest,
        split,
        data_dir=config.data_dir,
        verify_label_branches=bool(config.verify_offline_label_branches),
        read_chunk_size=int(config.read_chunk_size),
    )
    return (
        SubtokenDistillJetDataset(
            hlt_view,
            offline_view,
            label_filter=config.resolved_label_filter,
            label_names=config.resolved_label_names,
            max_jets=max_jets,
        ),
        manifest_sha,
    )


def align_teacher_logits_to_student(student_logits, teacher_logits, *, label_filter: Sequence[int]):
    """Align binary/multiclass teacher logits with the student output space."""

    torch = require_torch()
    if int(teacher_logits.shape[1]) == int(student_logits.shape[1]):
        return teacher_logits
    label_filter = tuple(int(label) for label in label_filter)
    if len(label_filter) != int(student_logits.shape[1]):
        raise ValueError("label_filter length must match student output dimension")
    max_label = max(label_filter, default=-1)
    if int(teacher_logits.shape[1]) <= max_label:
        raise ValueError(
            f"Teacher logits have {int(teacher_logits.shape[1])} classes, cannot select label_filter={label_filter}"
        )
    indices = torch.as_tensor(label_filter, dtype=torch.long, device=teacher_logits.device)
    return teacher_logits.index_select(dim=1, index=indices)


def compute_subtoken_distillation_kl_loss(student_logits, teacher_logits, *, temperature: float):
    """Temperature-scaled KL distillation loss for offline-teacher targets."""

    torch = require_torch()
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
    loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    loss = loss * (temperature * temperature)
    if not bool(torch.isfinite(loss).detach().cpu().item()):
        raise FloatingPointError("distillation KL loss is non-finite")
    return loss


def run_subtoken_distill_epoch(
    model: SubtokenParticleTransformerClassifier,
    teacher,
    loader,
    *,
    device,
    criterion,
    residual_head: ModalityResidualHead | None = None,
    masked_head: MaskedSubtokenPredictionHead | None = None,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    max_batches: int | None = None,
    label_filter: tuple[int, ...],
    label_names: tuple[str, ...] | None = None,
    feature_config=None,
    distillation_temperature: float = 2.0,
    distillation_weight: float = 0.5,
    classification_weight: float = 1.0,
    modality_residual_weight: float = 0.0,
    modality_residual_target_mode: str = "jet",
    modality_residual_huber_beta: float = 1.0,
    gate_residual_regularization_weight: float = 0.0,
    gate_residual_temperature: float = 1.0,
    masked_subtoken_weight: float = 0.0,
    masked_subtoken_target_mode: str = "hlt_self",
    masked_subtoken_probability: float = 0.15,
    masked_subtoken_huber_beta: float = 1.0,
    masked_subtoken_max_match_delta_r: float | None = 0.4,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    """Run one train/eval epoch where the offline teacher is a training-only target."""

    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    if residual_head is not None:
        residual_head.train(training)
    if masked_head is not None:
        masked_head.train(training)
    total_loss = 0.0
    total_ce_loss = 0.0
    total_distill_loss = 0.0
    total_residual_loss = 0.0
    total_gate_residual_loss = 0.0
    total_masked_subtoken_loss = 0.0
    total_correct = 0
    total_teacher_correct = 0
    total_teacher_agree = 0
    total_seen = 0
    collected_preds: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
    collected_logits: list[np.ndarray] = []
    residual_diagnostic_totals: dict[str, float] = {}
    residual_diagnostic_weight_sum = 0.0
    masked_diagnostic_totals: dict[str, float] = {}
    masked_diagnostic_weight_sum = 0.0
    autocast_enabled = bool(amp and device.type == "cuda")
    context = torch.enable_grad() if training else torch.no_grad()
    use_residual_targets = (
        residual_head is not None and float(modality_residual_weight) > 0.0
    ) or float(gate_residual_regularization_weight) > 0.0
    use_masked_subtokens = masked_head is not None and float(masked_subtoken_weight) > 0.0
    grad_parameters = list(model.parameters())
    if residual_head is not None:
        grad_parameters.extend(list(residual_head.parameters()))
    if masked_head is not None:
        grad_parameters.extend(list(masked_head.parameters()))

    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_distill_batch_to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            teacher_device = getattr(teacher, "device", device)
            with torch.no_grad():
                teacher_logits = teacher.forward_view_no_grad(
                    batch["offline_tokens"].to(teacher_device, non_blocking=True),
                    batch["offline_mask"].to(teacher_device, non_blocking=True),
                )

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                if use_residual_targets:
                    student_output = model(batch["tokens"], batch["mask"], return_outputs=True)
                    student_logits = student_output.logits
                else:
                    student_output = None
                    student_logits = model(batch["tokens"], batch["mask"])
                aligned_teacher_logits = align_teacher_logits_to_student(
                    student_logits,
                    teacher_logits.to(student_logits.device),
                    label_filter=label_filter,
                )
                ce_loss = criterion(student_logits, batch["labels"])
                distill_loss = compute_subtoken_distillation_kl_loss(
                    student_logits,
                    aligned_teacher_logits,
                    temperature=float(distillation_temperature),
                )
                loss = float(classification_weight) * ce_loss + float(distillation_weight) * distill_loss
                residual_loss = student_logits.new_zeros(())
                gate_residual_loss = student_logits.new_zeros(())
                masked_subtoken_loss = student_logits.new_zeros(())
                residual_targets = None
                residual_prediction = None
                if use_residual_targets:
                    residual_targets = compute_modality_residual_targets(
                        batch["tokens"],
                        batch["mask"],
                        batch["offline_tokens"],
                        batch["offline_mask"],
                        feature_config=feature_config,
                        target_mode=str(modality_residual_target_mode),
                    )
                    if residual_head is not None and float(modality_residual_weight) > 0.0:
                        residual_prediction = residual_head(student_output)
                        residual_loss = compute_modality_residual_loss(
                            residual_prediction,
                            residual_targets,
                            huber_beta=float(modality_residual_huber_beta),
                        )
                        loss = loss + float(modality_residual_weight) * residual_loss
                    if (
                        student_output is not None
                        and student_output.gates is not None
                        and float(gate_residual_regularization_weight) > 0.0
                    ):
                        gate_residual_loss = compute_gate_residual_regularization(
                            student_output.gates.gates,
                            residual_targets.targets,
                            student_output.gates.modality_mask & residual_targets.mask,
                            temperature=float(gate_residual_temperature),
                        )
                        loss = loss + float(gate_residual_regularization_weight) * gate_residual_loss
                masked_mask_output = None
                masked_prediction = None
                masked_targets = None
                if use_masked_subtokens:
                    modality_names = tuple(feature_config.modality_names) if feature_config is not None else (
                        "kinematics",
                        "identity",
                        "track",
                    )
                    masked_mask_output = sample_masked_subtoken_modality_mask(
                        batch["mask"],
                        num_modalities=len(modality_names),
                        mask_probability=float(masked_subtoken_probability),
                        modality_names=modality_names,
                        force_at_least_one=True,
                    )
                    masked_output = model(
                        batch["tokens"],
                        batch["mask"],
                        return_outputs=True,
                        modality_mask_override=masked_mask_output.input_modality_mask,
                    )
                    target_mode = normalize_masked_subtoken_target_mode(masked_subtoken_target_mode)
                    use_offline_target = target_mode in {"offline", "offline_slot"}
                    target_tokens = batch["offline_tokens"] if use_offline_target else batch["tokens"]
                    target_mask = batch["offline_mask"] if use_offline_target else batch["mask"]
                    masked_targets = build_masked_subtoken_targets(
                        target_tokens,
                        target_mask,
                        masked_mask_output.prediction_mask,
                        feature_config=feature_config,
                        target_mode=target_mode,
                        reference_tokens=batch["tokens"] if target_mode == "offline" else None,
                        reference_mask=batch["mask"] if target_mode == "offline" else None,
                        max_match_delta_r=masked_subtoken_max_match_delta_r,
                    )
                    masked_prediction = masked_head(masked_output, masked_mask_output)
                    masked_subtoken_loss = compute_masked_subtoken_loss(
                        masked_prediction,
                        masked_targets,
                        huber_beta=float(masked_subtoken_huber_beta),
                    )
                    loss = loss + float(masked_subtoken_weight) * masked_subtoken_loss

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(grad_parameters, float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(grad_parameters, float(grad_clip_norm))
                    optimizer.step()

            labels = batch["labels"]
            preds = student_logits.detach().argmax(dim=1)
            teacher_preds = aligned_teacher_logits.detach().argmax(dim=1)
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_ce_loss += float(ce_loss.detach().item()) * batch_size
            total_distill_loss += float(distill_loss.detach().item()) * batch_size
            total_residual_loss += float(residual_loss.detach().item()) * batch_size
            total_gate_residual_loss += float(gate_residual_loss.detach().item()) * batch_size
            total_masked_subtoken_loss += float(masked_subtoken_loss.detach().item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_teacher_correct += int((teacher_preds == labels).sum().item())
            total_teacher_agree += int((teacher_preds == preds).sum().item())
            total_seen += batch_size
            if collect_predictions:
                collected_preds.append(preds.detach().cpu().numpy().astype(np.int64))
                collected_labels.append(labels.detach().cpu().numpy().astype(np.int64))
                collected_logits.append(student_logits.detach().cpu().numpy().astype(np.float32))
            if residual_targets is not None:
                diagnostics = {
                    f"target_{key}": value
                    for key, value in residual_targets.diagnostics().items()
                    if hasattr(value, "detach") and int(value.numel()) == 1
                }
                if residual_prediction is not None:
                    diagnostics.update(
                        {
                            f"prediction_{key}": value
                            for key, value in residual_prediction.diagnostics().items()
                            if hasattr(value, "detach") and int(value.numel()) == 1
                        }
                    )
                for key, value in diagnostics.items():
                    residual_diagnostic_totals[key] = (
                        residual_diagnostic_totals.get(key, 0.0) + float(value.detach().cpu().item()) * batch_size
                    )
                residual_diagnostic_weight_sum += float(batch_size)
            if masked_mask_output is not None:
                diagnostics = {
                    f"mask_{key}": value
                    for key, value in masked_mask_output.diagnostics().items()
                    if hasattr(value, "detach") and int(value.numel()) == 1
                }
                if masked_prediction is not None:
                    diagnostics.update(
                        {
                            f"prediction_{key}": value
                            for key, value in masked_prediction.diagnostics().items()
                            if hasattr(value, "detach") and int(value.numel()) == 1
                        }
                    )
                if masked_targets is not None:
                    for key, value in masked_targets.matching_metadata.items():
                        if isinstance(value, (int, float)) and value is not None:
                            diagnostics[f"target_{key}"] = student_logits.new_tensor(float(value))
                for key, value in diagnostics.items():
                    masked_diagnostic_totals[key] = (
                        masked_diagnostic_totals.get(key, 0.0) + float(value.detach().cpu().item()) * batch_size
                    )
                masked_diagnostic_weight_sum += float(batch_size)

    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}
    metrics: dict[str, Any] = {
        "loss": total_loss / float(total_seen),
        "ce_loss": total_ce_loss / float(total_seen),
        "distillation_loss": total_distill_loss / float(total_seen),
        "modality_residual_loss": total_residual_loss / float(total_seen),
        "gate_residual_regularization_loss": total_gate_residual_loss / float(total_seen),
        "masked_subtoken_loss": total_masked_subtoken_loss / float(total_seen),
        "accuracy": total_correct / float(total_seen),
        "teacher_accuracy_on_offline": total_teacher_correct / float(total_seen),
        "student_teacher_agreement": total_teacher_agree / float(total_seen),
        "n_jets": int(total_seen),
    }
    if collect_predictions:
        preds_np = np.concatenate(collected_preds, axis=0) if collected_preds else np.asarray([], dtype=np.int64)
        labels_np = np.concatenate(collected_labels, axis=0) if collected_labels else np.asarray([], dtype=np.int64)
        logits_np = np.concatenate(collected_logits, axis=0) if collected_logits else None
        metrics.update(
            classification_metrics_from_predictions(
                preds=preds_np,
                labels=labels_np,
                loss_sum=total_loss,
                logits=logits_np,
                label_names=label_names,
            )
        )
    if residual_diagnostic_weight_sum > 0.0:
        metrics["modality_residual_diagnostics"] = {
            key: value / residual_diagnostic_weight_sum
            for key, value in sorted(residual_diagnostic_totals.items())
        }
    if masked_diagnostic_weight_sum > 0.0:
        metrics["masked_subtoken_diagnostics"] = {
            key: value / masked_diagnostic_weight_sum
            for key, value in sorted(masked_diagnostic_totals.items())
        }
    return metrics


def subtoken_distill_checkpoint_payload(
    model: SubtokenParticleTransformerClassifier,
    optimizer,
    *,
    residual_head: ModalityResidualHead | None = None,
    masked_head: MaskedSubtokenPredictionHead | None = None,
    epoch: int,
    config: SubtokenDistillTrainConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
    teacher_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "residual_head_state_dict": None if residual_head is None else residual_head.state_dict(),
        "masked_subtoken_head_state_dict": None if masked_head is None else masked_head.state_dict(),
        "config": asdict(config),
        "model_config": model.to_config_dict(),
        "metrics": dict(metrics),
        "label_names": list(config.resolved_label_names),
        "label_filter": list(config.resolved_label_filter),
        "num_classes": int(config.resolved_num_classes),
        "source": dict(source),
        "experiment_step": SUBTOKEN_PART_DISTILL_STEP,
        "tagger_model_step": SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP,
        "output_contract": model.output_contract,
        "distillation_contract": SUBTOKEN_PART_DISTILL_CONTRACT,
        "offline_teacher_metadata": dict(teacher_metadata),
        "modality_residual_supervision": {
            "step": SUBTOKEN_PART_RESIDUAL_STEP,
            "enabled": bool(float(config.modality_residual_weight) > 0.0),
            "target_mode": str(config.modality_residual_target_mode),
            "modality_residual_weight": float(config.modality_residual_weight),
            "modality_residual_huber_beta": float(config.modality_residual_huber_beta),
            "gate_residual_regularization_weight": float(config.gate_residual_regularization_weight),
            "gate_residual_temperature": float(config.gate_residual_temperature),
            "residual_head_saved_in_checkpoint": residual_head is not None,
        },
        "masked_subtoken_objective": {
            "step": SUBTOKEN_PART_MASKED_STEP,
            "enabled": bool(float(config.masked_subtoken_weight) > 0.0),
            "target_mode": str(config.masked_subtoken_target_mode),
            "offline_target_alignment": (
                "nearest_delta_r" if str(config.masked_subtoken_target_mode) == "offline" else "slot_aligned"
            ),
            "masked_subtoken_weight": float(config.masked_subtoken_weight),
            "masked_subtoken_probability": float(config.masked_subtoken_probability),
            "masked_subtoken_huber_beta": float(config.masked_subtoken_huber_beta),
            "masked_subtoken_max_match_delta_r": config.masked_subtoken_max_match_delta_r,
            "masked_head_saved_in_checkpoint": masked_head is not None,
        },
        "offline_teacher_used_for_training_only": True,
        "inference_consumes_hlt_only": True,
    }


def _teacher_metadata(teacher, config: SubtokenDistillTrainConfig) -> dict[str, Any]:
    metadata = dict(getattr(teacher, "metadata", {}) or {})
    metadata.setdefault("architecture", str(config.offline_teacher_architecture))
    metadata.setdefault("checkpoint_path", str(config.offline_teacher_checkpoint))
    metadata["distillation_temperature"] = float(config.distillation_temperature)
    metadata["distillation_weight"] = float(config.distillation_weight)
    metadata["classification_weight"] = float(config.classification_weight)
    metadata["teacher_max_constits"] = int(config.teacher_max_constits)
    metadata["teacher_weight_threshold"] = float(config.teacher_weight_threshold)
    return metadata


def train_subtoken_distilled_tagger(
    config: SubtokenDistillTrainConfig,
    *,
    model: SubtokenParticleTransformerClassifier | None = None,
    teacher=None,
    train_dataset: SubtokenDistillJetDataset | None = None,
    val_dataset: SubtokenHLTJetDataset | None = None,
    stack_val_dataset: SubtokenHLTJetDataset | None = None,
    final_test_dataset: SubtokenHLTJetDataset | None = None,
) -> dict[str, Any]:
    """Train Version B: HLT-only student with offline-teacher distillation."""

    run_start_time = time.perf_counter()
    config.validate_label_metadata()
    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    manifest_sha = None
    if train_dataset is None:
        train_dataset, manifest_sha = _load_subtoken_distill_dataset(
            config,
            config.train_split,
            max_jets=config.max_train_jets,
        )
    else:
        manifest_sha = train_dataset.metadata.get("source_manifest_hash")
    val_dataset = val_dataset or SubtokenHLTJetDataset(
        load_cached_hlt_view(config.hlt_cache_dir, config.val_split, verify_hash=bool(config.verify_hlt_hash)),
        label_filter=config.resolved_label_filter,
        label_names=config.resolved_label_names,
        max_jets=config.max_val_jets,
    )
    if train_dataset.tokens.shape[-1] != RAW_TOKEN_DIM or train_dataset.offline_tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"Distillation dataset expects raw token dim {RAW_TOKEN_DIM}")
    if val_dataset.tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"Validation HLT dataset expects raw token dim {RAW_TOKEN_DIM}")

    train_loader = make_subtoken_distill_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    val_loader = make_subtoken_hlt_loader(
        val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )

    checkpoint_model = model or build_subtoken_particle_transformer_classifier(config.model_config())
    checkpoint_model = checkpoint_model.to(device)
    train_model = checkpoint_model
    if config.compile_model and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)
    residual_head = None
    if float(config.modality_residual_weight) > 0.0:
        residual_head = ModalityResidualHead(config.model_config()).to(device)
    masked_head = None
    if float(config.masked_subtoken_weight) > 0.0:
        masked_head = MaskedSubtokenPredictionHead(config.model_config()).to(device)
    if teacher is None:
        teacher = load_frozen_teacher(
            config.offline_teacher_checkpoint,
            architecture=config.offline_teacher_architecture,
            device=config.teacher_device or str(device),
            max_constits=int(config.teacher_max_constits),
            weight_threshold=float(config.teacher_weight_threshold),
            strict=bool(config.strict_teacher_checkpoint),
        )
    teacher_meta = _teacher_metadata(teacher, config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_params = list(checkpoint_model.parameters())
    if residual_head is not None:
        optimizer_params.extend(list(residual_head.parameters()))
    if masked_head is not None:
        optimizer_params.extend(list(masked_head.parameters()))
    optimizer = torch.optim.AdamW(optimizer_params, lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": SUBTOKEN_PART_DISTILL_STEP,
        "tagger_model_step": SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP,
        "distillation_contract": SUBTOKEN_PART_DISTILL_CONTRACT,
        "output_contract": checkpoint_model.output_contract,
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "manifest_hash": manifest_sha,
        "label_names": list(config.resolved_label_names),
        "label_filter": list(config.resolved_label_filter),
        "num_classes": int(config.resolved_num_classes),
        "selection_metric": str(config.selection_metric),
        "offline_teacher_metadata": teacher_meta,
        "modality_residual_supervision": {
            "step": SUBTOKEN_PART_RESIDUAL_STEP,
            "enabled": bool(float(config.modality_residual_weight) > 0.0),
            "target_mode": str(config.modality_residual_target_mode),
            "modality_residual_weight": float(config.modality_residual_weight),
            "gate_residual_regularization_weight": float(config.gate_residual_regularization_weight),
            "gate_residual_temperature": float(config.gate_residual_temperature),
        },
        "masked_subtoken_objective": {
            "step": SUBTOKEN_PART_MASKED_STEP,
            "enabled": bool(float(config.masked_subtoken_weight) > 0.0),
            "target_mode": str(config.masked_subtoken_target_mode),
            "offline_target_alignment": (
                "nearest_delta_r" if str(config.masked_subtoken_target_mode) == "offline" else "slot_aligned"
            ),
            "masked_subtoken_weight": float(config.masked_subtoken_weight),
            "masked_subtoken_probability": float(config.masked_subtoken_probability),
            "masked_subtoken_huber_beta": float(config.masked_subtoken_huber_beta),
            "masked_subtoken_max_match_delta_r": config.masked_subtoken_max_match_delta_r,
            "masked_head_saved_in_checkpoint": masked_head is not None,
        },
        "offline_teacher_used_for_training_only": True,
        "offline_teacher_splits_loaded": [str(config.train_split)],
        "leakage_rule": (
            "Version B uses offline tokens only inside the training loop to query a frozen teacher. "
            "The student forward path, model_val selection, stack_val/final_test evaluation, and saved "
            "inference checkpoint consume cached fixed-HLT tokens only."
        ),
        "final_test_loaded_during_training": False,
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_score = float("-inf")
    best_selection_metric_value = float("nan")
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_subtoken_distill_epoch(
            train_model,
            teacher,
            train_loader,
            device=device,
            criterion=criterion,
            residual_head=residual_head,
            masked_head=masked_head,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(config.amp),
            grad_clip_norm=float(config.grad_clip_norm),
            max_batches=config.max_train_batches,
            label_filter=tuple(config.resolved_label_filter),
            label_names=tuple(config.resolved_label_names),
            feature_config=checkpoint_model.config.feature_config,
            distillation_temperature=float(config.distillation_temperature),
            distillation_weight=float(config.distillation_weight),
            classification_weight=float(config.classification_weight),
            modality_residual_weight=float(config.modality_residual_weight),
            modality_residual_target_mode=str(config.modality_residual_target_mode),
            modality_residual_huber_beta=float(config.modality_residual_huber_beta),
            gate_residual_regularization_weight=float(config.gate_residual_regularization_weight),
            gate_residual_temperature=float(config.gate_residual_temperature),
            masked_subtoken_weight=float(config.masked_subtoken_weight),
            masked_subtoken_target_mode=str(config.masked_subtoken_target_mode),
            masked_subtoken_probability=float(config.masked_subtoken_probability),
            masked_subtoken_huber_beta=float(config.masked_subtoken_huber_beta),
            masked_subtoken_max_match_delta_r=config.masked_subtoken_max_match_delta_r,
            collect_predictions=False,
        )
        val_metrics = run_subtoken_tagger_epoch(
            train_model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
            collect_predictions=_selection_metric_requires_predictions(str(config.selection_metric)),
            collect_diagnostics=False,
            label_names=tuple(config.resolved_label_names),
        )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})
        _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)

        val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
        val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        improved = val_score > best_val_score or (np.isclose(val_score, best_val_score) and val_loss < best_val_loss)
        payload = subtoken_distill_checkpoint_payload(
            checkpoint_model,
            optimizer,
            residual_head=residual_head,
            masked_head=masked_head,
            epoch=epoch,
            config=config,
            metrics=row,
            source=source,
            teacher_metadata=teacher_meta,
        )
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best_val_score = float(val_score)
            best_selection_metric_value = float(selection_value)
            best_val_accuracy = float(val_accuracy)
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(payload, output_dir / "best_model_val.pt")
        else:
            epochs_without_improvement += 1
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise FloatingPointError("subtoken distilled tagger did not produce a valid model_val checkpoint")

    best_model, _ = load_subtoken_tagger_checkpoint(output_dir / "best_model_val.pt", device=device)
    best_val_metrics = run_subtoken_tagger_epoch(
        best_model,
        val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_val_batches,
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.resolved_label_names),
    )
    metrics_by_split: dict[str, Mapping[str, Any]] = {"model_val": best_val_metrics}

    stack_val_dataset = stack_val_dataset or SubtokenHLTJetDataset(
        load_cached_hlt_view(config.hlt_cache_dir, config.stack_val_split, verify_hash=bool(config.verify_hlt_hash)),
        label_filter=config.resolved_label_filter,
        label_names=config.resolved_label_names,
        max_jets=config.max_stack_val_jets,
    )
    stack_val_loader = make_subtoken_hlt_loader(
        stack_val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 2,
    )
    stack_val_metrics = run_subtoken_tagger_epoch(
        best_model,
        stack_val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_stack_val_batches,
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.resolved_label_names),
    )
    metrics_by_split["stack_val"] = stack_val_metrics

    final_test_metrics = None
    final_test_metadata = None
    if bool(config.confirm_final_test):
        final_test_dataset = final_test_dataset or SubtokenHLTJetDataset(
            load_cached_hlt_view(config.hlt_cache_dir, config.final_test_split, verify_hash=bool(config.verify_hlt_hash)),
            label_filter=config.resolved_label_filter,
            label_names=config.resolved_label_names,
            max_jets=config.max_final_test_jets,
        )
        final_test_loader = make_subtoken_hlt_loader(
            final_test_dataset,
            batch_size=int(config.eval_batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            seed=int(config.seed) + 3,
        )
        final_test_metrics = run_subtoken_tagger_epoch(
            best_model,
            final_test_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_final_test_batches,
            collect_predictions=True,
            collect_diagnostics=True,
            label_names=tuple(config.resolved_label_names),
        )
        metrics_by_split["final_test"] = final_test_metrics
        final_test_metadata = dict(final_test_dataset.metadata)

    elapsed_seconds = float(time.perf_counter() - run_start_time)
    _write_per_class_csv(diagnostics_dir / "per_class_metrics.csv", metrics_by_split)
    _write_summary_csv(diagnostics_dir / "summary_metrics.csv", metrics_by_split)
    report = {
        "experiment_step": SUBTOKEN_PART_DISTILL_STEP,
        "tagger_model_step": SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP,
        "distillation_contract": SUBTOKEN_PART_DISTILL_CONTRACT,
        "output_contract": best_model.output_contract,
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": (
            "minimize" if str(config.selection_metric) in SUBTOKEN_PART_LOWER_IS_BETTER_SELECTION_METRICS else "maximize"
        ),
        "best_model_selection_metric_value": float(best_selection_metric_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "best_model_val_metrics": best_val_metrics,
        "stack_val_metrics": stack_val_metrics,
        "final_test_metrics": final_test_metrics,
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(output_dir / "training_curves.json"),
        "diagnostic_csv": str(diagnostics_dir / "epoch_metrics.csv"),
        "summary_metrics_csv": str(diagnostics_dir / "summary_metrics.csv"),
        "per_class_metrics_csv": str(diagnostics_dir / "per_class_metrics.csv"),
        "config": asdict(config),
        "model_config": best_model.to_config_dict(),
        "label_names": list(config.resolved_label_names),
        "label_filter": list(config.resolved_label_filter),
        "num_classes": int(config.resolved_num_classes),
        "source": source,
        "manifest_hash": manifest_sha,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "stack_val_dataset": dict(stack_val_dataset.metadata),
        "final_test_dataset": final_test_metadata,
        "final_test_evaluated": bool(config.confirm_final_test),
        "no_final_test_evaluation": not bool(config.confirm_final_test),
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "epochs_completed": len(curves),
            "seconds_per_completed_epoch": elapsed_seconds / float(len(curves)) if curves else None,
        },
        "walltime_seconds": elapsed_seconds,
        "offline_teacher_metadata": teacher_meta,
        "modality_residual_supervision": {
            "step": SUBTOKEN_PART_RESIDUAL_STEP,
            "enabled": bool(float(config.modality_residual_weight) > 0.0),
            "target_mode": str(config.modality_residual_target_mode),
            "modality_residual_weight": float(config.modality_residual_weight),
            "modality_residual_huber_beta": float(config.modality_residual_huber_beta),
            "gate_residual_regularization_weight": float(config.gate_residual_regularization_weight),
            "gate_residual_temperature": float(config.gate_residual_temperature),
            "residual_head_saved_in_checkpoint": residual_head is not None,
        },
        "masked_subtoken_objective": {
            "step": SUBTOKEN_PART_MASKED_STEP,
            "enabled": bool(float(config.masked_subtoken_weight) > 0.0),
            "target_mode": str(config.masked_subtoken_target_mode),
            "offline_target_alignment": (
                "nearest_delta_r" if str(config.masked_subtoken_target_mode) == "offline" else "slot_aligned"
            ),
            "masked_subtoken_weight": float(config.masked_subtoken_weight),
            "masked_subtoken_probability": float(config.masked_subtoken_probability),
            "masked_subtoken_huber_beta": float(config.masked_subtoken_huber_beta),
            "masked_subtoken_max_match_delta_r": config.masked_subtoken_max_match_delta_r,
            "masked_head_saved_in_checkpoint": masked_head is not None,
        },
        "offline_teacher_used_for_training_only": True,
        "offline_teacher_splits_loaded": [str(config.train_split)],
        "inference_consumes_hlt_only": True,
        "inference_requires_offline": False,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    _write_best_metrics_csv(diagnostics_dir / "best_metrics.csv", report)
    return report


__all__ = [
    "SUBTOKEN_PART_DISTILL_CONTRACT",
    "SUBTOKEN_PART_DISTILL_STEP",
    "SubtokenDistillJetDataset",
    "SubtokenDistillTrainConfig",
    "align_teacher_logits_to_student",
    "collate_subtoken_distill_batch",
    "compute_subtoken_distillation_kl_loss",
    "make_subtoken_distill_loader",
    "run_subtoken_distill_epoch",
    "subtoken_distill_checkpoint_payload",
    "train_subtoken_distilled_tagger",
]

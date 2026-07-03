"""PD10-V2 particle-level dual-view teacher training and caching."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, load_prediction_block, save_prediction_block, softmax_np
from jetclass_fresh.heterogeneous_hlt import balanced_limit_jet_view
from jetclass_fresh.hlt_baseline import (
    default_part_config,
    require_torch,
    resolve_device,
    save_json,
    set_training_seed,
)
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetIdentity, JetView, LABEL_NAMES, load_offline_view, load_split_manifest, manifest_hash
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES, build_particle_transformer_inputs_from_tokens

from .config import (
    PD10_EXPERIMENT_NAME,
    PD10_EXTENDED_TEACHER_ALLOWED_INPUTS,
    PD10_NUM_CLASSES,
    PD10_REPRESENTATION_DIM,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_HLT,
    PD10_TEACHER_OFFLINE,
    PD10_TEACHER_PARTICLE_DUAL_VIEW,
    default_pd10_experiment_layout,
    pd10_extended_teacher_model_name,
)
from .inputs import pd10_hlt_params_dict
from .representations import (
    PD10_TEACHER_REPRESENTATION_SPLITS,
    PD10TeacherRepresentationCacheConfig,
    build_pd10_teacher_representation_block,
    load_pd10_teacher_representation_block,
    save_pd10_teacher_representation_block,
    validate_pd10_teacher_representation_metadata,
    write_pd10_teacher_representation_manifest,
)
from .teachers import load_pd10_part_teacher_model_from_checkpoint, sha256_file


PD10_V2_STEP2_EXPERIMENT_STEP = "pd10_v2_step2_particle_dual_view_teacher"
PD10_V2_STEP2_TRAIN_EXPERIMENT_STEP = f"{PD10_V2_STEP2_EXPERIMENT_STEP}:train"
PD10_V2_STEP2_CACHE_EXPERIMENT_STEP = f"{PD10_V2_STEP2_EXPERIMENT_STEP}:cache"
PD10_PARTICLE_DUAL_VIEW_TEACHER_CONTRACT = "pd10_particle_dual_view_teacher_v1"
PD10_PARTICLE_DUAL_VIEW_LOGIT_CACHE_CONTRACT = "pd10_particle_dual_view_logit_cache_v1"
PD10_PARTICLE_DUAL_VIEW_CACHE_MANIFEST = "particle_dual_view_cache_manifest.json"
PD10_PARTICLE_DUAL_VIEW_CACHE_REPORT = "particle_dual_view_cache_report.json"
PD10_PARTICLE_DUAL_VIEW_MODEL_NAME = pd10_extended_teacher_model_name(PD10_TEACHER_PARTICLE_DUAL_VIEW)
PD10_PARTICLE_DUAL_VIEW_DEFAULT_SEED = 2707
PD10_PARTICLE_DUAL_VIEW_DEFAULT_BATCH_SIZE = 128
PD10_PARTICLE_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE = 128
PD10_PARTICLE_DUAL_VIEW_DEFAULT_EPOCHS = 10
PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_WARMUP_EPOCHS = 1
PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_WARMUP_LR = 1.0e-3
PD10_PARTICLE_DUAL_VIEW_DEFAULT_BRANCH_LR = 3.0e-5
PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_LR = 3.0e-4
PD10_PARTICLE_DUAL_VIEW_DEFAULT_WEIGHT_DECAY = 1.0e-4
PD10_PARTICLE_DUAL_VIEW_DEFAULT_DROPOUT = 0.05
PD10_PARTICLE_DUAL_VIEW_DEFAULT_FUSION_HIDDEN_DIM = 512
PD10_PARTICLE_DUAL_VIEW_DEFAULT_EARLY_STOP_PATIENCE = 3
PD10_PARTICLE_DUAL_VIEW_LOGIT_SPLITS: tuple[str, ...] = PD10_SPLIT_ORDER


@dataclass(frozen=True)
class PD10ParticleDualViewTeacherTrainConfig:
    """Training config for the particle-level HLT+offline teacher."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    hlt_teacher_checkpoint: str
    offline_teacher_checkpoint: str
    data_dir: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    seed: int = PD10_PARTICLE_DUAL_VIEW_DEFAULT_SEED
    batch_size: int = PD10_PARTICLE_DUAL_VIEW_DEFAULT_BATCH_SIZE
    eval_batch_size: int = PD10_PARTICLE_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE
    epochs: int = PD10_PARTICLE_DUAL_VIEW_DEFAULT_EPOCHS
    head_warmup_epochs: int = PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_WARMUP_EPOCHS
    head_warmup_lr: float = PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_WARMUP_LR
    branch_lr: float = PD10_PARTICLE_DUAL_VIEW_DEFAULT_BRANCH_LR
    head_lr: float = PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_LR
    weight_decay: float = PD10_PARTICLE_DUAL_VIEW_DEFAULT_WEIGHT_DECAY
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = PD10_PARTICLE_DUAL_VIEW_DEFAULT_EARLY_STOP_PATIENCE
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_train_jets: int | None = PD10_SPLIT_SIZES["model_train"]
    max_val_jets: int | None = PD10_SPLIT_SIZES["model_val"]
    model_size: str = "base"
    compile_model: bool = False
    fusion_hidden_dim: int = PD10_PARTICLE_DUAL_VIEW_DEFAULT_FUSION_HIDDEN_DIM
    representation_dim: int = PD10_REPRESENTATION_DIM
    dropout: float = PD10_PARTICLE_DUAL_VIEW_DEFAULT_DROPOUT
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000
    verify_hlt_hash: bool = True
    initialize_branches: bool = True

    def __post_init__(self) -> None:
        if (self.train_split, self.val_split) != ("model_train", "model_val"):
            raise ValueError("particle dual-view teacher may train only on model_train and select only on model_val")
        if int(self.batch_size) <= 0 or int(self.eval_batch_size) <= 0:
            raise ValueError("batch_size and eval_batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if int(self.head_warmup_epochs) < 0:
            raise ValueError("head_warmup_epochs cannot be negative")
        if float(self.head_warmup_lr) <= 0.0 or float(self.branch_lr) <= 0.0 or float(self.head_lr) <= 0.0:
            raise ValueError("learning rates must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.model_size not in {"base", "tiny", "large"}:
            raise ValueError("model_size must be 'base', 'tiny', or 'large'")
        if int(self.fusion_hidden_dim) <= 0:
            raise ValueError("fusion_hidden_dim must be positive")
        if int(self.representation_dim) <= 0:
            raise ValueError("representation_dim must be positive")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for split, value in (("model_train", self.max_train_jets), ("model_val", self.max_val_jets)):
            if value is not None and int(value) > int(PD10_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PD10_SPLIT_SIZES[split]}")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "head_warmup_epochs", int(self.head_warmup_epochs))
        object.__setattr__(self, "representation_dim", int(self.representation_dim))

    @property
    def model_name(self) -> str:
        return PD10_PARTICLE_DUAL_VIEW_MODEL_NAME

    @property
    def teacher_target(self) -> str:
        return PD10_TEACHER_PARTICLE_DUAL_VIEW

    @property
    def allowed_inputs(self) -> str:
        return PD10_EXTENDED_TEACHER_ALLOWED_INPUTS[self.teacher_target]

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "best_model_val.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "last.pt"


@dataclass(frozen=True)
class PD10ParticleDualViewTeacherCacheConfig:
    """Prediction/representation cache config for the selected particle dual-view teacher."""

    checkpoint: str
    manifest_path: str
    hlt_cache_dir: str
    logit_output_dir: str
    representation_output_dir: str
    data_dir: str | None = None
    splits: tuple[str, ...] = field(default_factory=lambda: PD10_PARTICLE_DUAL_VIEW_LOGIT_SPLITS)
    batch_size: int = PD10_PARTICLE_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE
    num_workers: int = 0
    device: str = "auto"
    max_model_train_jets: int | None = PD10_SPLIT_SIZES["model_train"]
    max_model_val_jets: int | None = PD10_SPLIT_SIZES["model_val"]
    max_final_test_jets: int | None = PD10_SPLIT_SIZES["final_test"]
    max_batches: int | None = None
    overwrite: bool = False
    skip_existing: bool = True
    confirm_final_test: bool = False
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000
    control_seed: int = 9901
    verify_hlt_hash: bool = True

    def __post_init__(self) -> None:
        splits = tuple(str(split) for split in self.splits)
        if not splits:
            raise ValueError("at least one split is required")
        unknown = [split for split in splits if split not in PD10_PARTICLE_DUAL_VIEW_LOGIT_SPLITS]
        if unknown:
            raise ValueError(f"unknown particle dual-view cache splits: {unknown}")
        if "final_test" in splits and not bool(self.confirm_final_test):
            raise ValueError("Refusing to cache final_test particle dual-view outputs without confirm_final_test=True")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        for split, value in (
            ("model_train", self.max_model_train_jets),
            ("model_val", self.max_model_val_jets),
            ("final_test", self.max_final_test_jets),
        ):
            if value is not None and int(value) > int(PD10_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PD10_SPLIT_SIZES[split]}")
        object.__setattr__(self, "splits", splits)

    @property
    def model_name(self) -> str:
        return PD10_PARTICLE_DUAL_VIEW_MODEL_NAME

    @property
    def teacher_target(self) -> str:
        return PD10_TEACHER_PARTICLE_DUAL_VIEW

    @property
    def logit_dir(self) -> Path:
        return Path(self.logit_output_dir) / self.model_name

    @property
    def representation_dir(self) -> Path:
        return Path(self.representation_output_dir) / self.model_name


class PD10ParticleDualViewTeacher:
    """Factory wrapper for a two-branch Particle Transformer privileged teacher."""

    def __new__(
        cls,
        *,
        num_classes: int = PD10_NUM_CLASSES,
        model_size: str = "base",
        branch_config: Mapping[str, Any] | None = None,
        fusion_hidden_dim: int = PD10_PARTICLE_DUAL_VIEW_DEFAULT_FUSION_HIDDEN_DIM,
        representation_dim: int = PD10_REPRESENTATION_DIM,
        dropout: float = PD10_PARTICLE_DUAL_VIEW_DEFAULT_DROPOUT,
    ):
        torch = require_torch()
        nn = torch.nn
        try:
            from weaver.nn.model.ParticleTransformer import ParticleTransformer
        except ImportError as exc:  # pragma: no cover - depends on research env
            raise ImportError(
                "Particle dual-view teacher requires weaver-core on the research compute."
            ) from exc

        cfg = default_part_config(num_classes=int(num_classes), model_size=model_size)
        if branch_config:
            cfg.update(dict(branch_config))
        cfg["num_classes"] = None
        cfg["fc_params"] = None
        branch_dim = int(cfg["embed_dims"][-1])

        class _EmbeddingBranch(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = dict(cfg)
                self.mod = ParticleTransformer(**cfg)

            def forward(self, inputs: Mapping[str, Any]):
                return self.mod(inputs["features"], v=inputs["lorentz_vectors"], mask=inputs["mask"])

        class _PD10ParticleDualViewTeacher(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hlt_branch = _EmbeddingBranch()
                self.offline_branch = _EmbeddingBranch()
                self.fusion = nn.Sequential(
                    nn.LayerNorm(branch_dim * 4),
                    nn.Linear(branch_dim * 4, int(fusion_hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(fusion_hidden_dim), int(representation_dim)),
                    nn.GELU(),
                    nn.LayerNorm(int(representation_dim)),
                )
                self.classifier = nn.Linear(int(representation_dim), int(num_classes))
                self.config = {
                    "contract": PD10_PARTICLE_DUAL_VIEW_TEACHER_CONTRACT,
                    "architecture": "particle_dual_view_part_concat_abs_product",
                    "num_classes": int(num_classes),
                    "model_size": model_size,
                    "branch_config": dict(cfg),
                    "branch_dim": int(branch_dim),
                    "fusion_input_dim": int(branch_dim * 4),
                    "fusion_hidden_dim": int(fusion_hidden_dim),
                    "representation_dim": int(representation_dim),
                    "dropout": float(dropout),
                    "hlt_feature_names": list(PF_FEATURE_NAMES),
                    "offline_feature_names": list(PF_FEATURE_NAMES),
                    "allowed_inputs": PD10_EXTENDED_TEACHER_ALLOWED_INPUTS[PD10_TEACHER_PARTICLE_DUAL_VIEW],
                    "student_deployment_inputs": "HLT_only",
                    "returns_offline_particles": False,
                    "teacher_is_train_time_only_for_distillation": True,
                    "inference_requires_offline_inputs": True,
                    "teacher_inference_requires_offline_inputs": True,
                    "inference_export_requires_teacher_features": False,
                }

            def no_weight_decay(self) -> set[str]:
                return {"hlt_branch.mod.cls_token", "offline_branch.mod.cls_token"}

            def branch_parameters(self):
                yield from self.hlt_branch.parameters()
                yield from self.offline_branch.parameters()

            def head_parameters(self):
                yield from self.fusion.parameters()
                yield from self.classifier.parameters()

            def set_branches_trainable(self, trainable: bool) -> None:
                for parameter in self.hlt_branch.parameters():
                    parameter.requires_grad_(bool(trainable))
                for parameter in self.offline_branch.parameters():
                    parameter.requires_grad_(bool(trainable))

            def forward(
                self,
                hlt_inputs: Mapping[str, Any],
                offline_inputs: Mapping[str, Any],
                *,
                return_representation: bool = False,
            ):
                hlt_embedding = self.hlt_branch(hlt_inputs)
                offline_embedding = self.offline_branch(offline_inputs)
                fused_input = torch.cat(
                    [
                        hlt_embedding,
                        offline_embedding,
                        torch.abs(offline_embedding - hlt_embedding),
                        hlt_embedding * offline_embedding,
                    ],
                    dim=1,
                )
                representation = self.fusion(fused_input)
                logits = self.classifier(representation)
                if return_representation:
                    return logits, torch.nn.functional.normalize(representation, dim=1)
                return logits

        return _PD10ParticleDualViewTeacher()


def pd10_particle_dual_view_teacher_dir(*, output_root: str | Path = "checkpoints") -> Path:
    layout = default_pd10_experiment_layout(output_root=output_root)
    return layout.root / "teachers" / PD10_PARTICLE_DUAL_VIEW_MODEL_NAME


def pd10_particle_dual_view_teacher_checkpoint(*, output_root: str | Path = "checkpoints") -> Path:
    return pd10_particle_dual_view_teacher_dir(output_root=output_root) / "best_model_val.pt"


def pd10_particle_dual_view_logit_cache_dir(*, output_root: str | Path = "checkpoints") -> Path:
    layout = default_pd10_experiment_layout(output_root=output_root)
    return layout.root / "teacher_logits" / PD10_PARTICLE_DUAL_VIEW_MODEL_NAME


def pd10_particle_dual_view_representation_cache_dir(*, output_root: str | Path = "checkpoints") -> Path:
    layout = default_pd10_experiment_layout(output_root=output_root)
    return layout.root / "teacher_representations" / PD10_PARTICLE_DUAL_VIEW_MODEL_NAME


def _identity_key(identity: JetIdentity) -> tuple[str, int, int]:
    return str(identity.file), int(identity.entry), int(identity.label)


def align_pd10_hlt_offline_views(hlt_view: JetView, offline_view: JetView) -> tuple[JetView, JetView]:
    """Return HLT/offline views ordered by the HLT rows and matched by identity."""

    if hlt_view.split != offline_view.split:
        raise ValueError(f"split mismatch: {hlt_view.split} != {offline_view.split}")
    if hlt_view.metadata.get("view") not in (None, "fixed_hlt"):
        raise ValueError(f"Expected fixed_hlt HLT view, got {hlt_view.metadata.get('view')!r}")
    if offline_view.metadata.get("view") not in (None, "offline"):
        raise ValueError(f"Expected offline view, got {offline_view.metadata.get('view')!r}")
    offline_index_by_id: dict[tuple[str, int, int], int] = {}
    for index, identity in enumerate(offline_view.jet_ids):
        key = _identity_key(identity)
        if key in offline_index_by_id:
            raise ValueError(f"duplicate offline jet identity in {offline_view.split}: {identity}")
        offline_index_by_id[key] = int(index)
    offline_indices: list[int] = []
    for row, identity in enumerate(hlt_view.jet_ids):
        key = _identity_key(identity)
        if key not in offline_index_by_id:
            raise ValueError(f"HLT row {row} missing from offline view: {identity}")
        offline_index = offline_index_by_id[key]
        if int(hlt_view.labels[row]) != int(offline_view.labels[offline_index]):
            raise ValueError(f"HLT/offline label mismatch at HLT row {row}")
        offline_indices.append(offline_index)
    aligned_offline = JetView(
        tokens=offline_view.tokens[offline_indices],
        mask=offline_view.mask[offline_indices],
        labels=offline_view.labels[offline_indices],
        jet_ids=[offline_view.jet_ids[index] for index in offline_indices],
        split=offline_view.split,
        metadata={**dict(offline_view.metadata), "aligned_to": "fixed_hlt", "aligned_rows": int(len(offline_indices))},
    )
    if not np.array_equal(hlt_view.labels, aligned_offline.labels):
        raise ValueError("aligned HLT/offline labels differ")
    if hlt_view.jet_ids != aligned_offline.jet_ids:
        raise ValueError("aligned HLT/offline jet identities differ")
    return hlt_view, aligned_offline


def _load_paired_views(
    *,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    split: str,
    data_dir: str | None,
    max_jets: int | None,
    seed: int,
    verify_label_branches: bool,
    read_chunk_size: int,
    verify_hlt_hash: bool,
) -> tuple[JetView, JetView, dict[str, Any]]:
    hlt_view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=bool(verify_hlt_hash))
    if hlt_view.metadata.get("hlt_params") != pd10_hlt_params_dict():
        raise ValueError(f"HLT cache for {split} does not match configured PD10 fixed-HLT params")
    manifest = load_split_manifest(manifest_path)
    offline_view = load_offline_view(
        manifest,
        split,
        data_dir=data_dir,
        verify_label_branches=verify_label_branches,
        read_chunk_size=read_chunk_size,
    )
    hlt_view, selection_report = balanced_limit_jet_view(hlt_view, max_jets, seed=int(seed))
    hlt_view, offline_view = align_pd10_hlt_offline_views(hlt_view, offline_view)
    source_metadata = {
        "source_view": "paired_hlt_offline_particles",
        "source_manifest_hash": manifest_hash(manifest),
        "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
        "hlt_jet_identity_hash": hlt_view.metadata.get("jet_identity_hash"),
        "hlt_degradation_strength": hlt_view.metadata.get("hlt_degradation_strength"),
        "expected_hlt_params": pd10_hlt_params_dict(),
        "offline_privileged_inputs_loaded": True,
        "uses_raw_offline_particles": True,
        "student_deployment_inputs": "HLT_only",
        "returns_offline_particles": False,
        "teacher_is_train_time_only_for_distillation": True,
        "inference_requires_offline_inputs": True,
        "subset_selection": selection_report,
    }
    return hlt_view, offline_view, source_metadata


class PD10PairedParticleViewDataset:
    """Paired HLT/offline dataset for privileged teacher training."""

    def __init__(self, hlt_view: JetView, offline_view: JetView) -> None:
        require_torch()
        hlt_view, offline_view = align_pd10_hlt_offline_views(hlt_view, offline_view)
        self.hlt_tokens = np.asarray(hlt_view.tokens, dtype=np.float32)
        self.hlt_mask = np.asarray(hlt_view.mask, dtype=bool)
        self.offline_tokens = np.asarray(offline_view.tokens, dtype=np.float32)
        self.offline_mask = np.asarray(offline_view.mask, dtype=bool)
        self.labels = np.asarray(hlt_view.labels, dtype=np.int64)
        self.jet_ids = list(hlt_view.jet_ids)
        self.split = hlt_view.split

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return (
            self.hlt_tokens[index],
            self.hlt_mask[index],
            self.offline_tokens[index],
            self.offline_mask[index],
            self.labels[index],
        )


def collate_pd10_particle_dual_view_batch(samples):
    torch = require_torch()
    hlt_tokens = np.stack([sample[0] for sample in samples], axis=0)
    hlt_mask = np.stack([sample[1] for sample in samples], axis=0)
    offline_tokens = np.stack([sample[2] for sample in samples], axis=0)
    offline_mask = np.stack([sample[3] for sample in samples], axis=0)
    labels = np.asarray([sample[4] for sample in samples], dtype=np.int64)
    hlt_inputs = build_particle_transformer_inputs_from_tokens(
        hlt_tokens,
        hlt_mask,
        labels=labels,
        source_view="fixed_hlt",
    )
    offline_inputs = build_particle_transformer_inputs_from_tokens(
        offline_tokens,
        offline_mask,
        labels=labels,
        source_view="offline",
    )
    return {
        "hlt": {
            "points": torch.from_numpy(hlt_inputs.pf_points).float(),
            "features": torch.from_numpy(hlt_inputs.pf_features).float(),
            "lorentz_vectors": torch.from_numpy(hlt_inputs.pf_vectors).float(),
            "mask": torch.from_numpy(hlt_inputs.pf_mask).bool(),
        },
        "offline": {
            "points": torch.from_numpy(offline_inputs.pf_points).float(),
            "features": torch.from_numpy(offline_inputs.pf_features).float(),
            "lorentz_vectors": torch.from_numpy(offline_inputs.pf_vectors).float(),
            "mask": torch.from_numpy(offline_inputs.pf_mask).bool(),
        },
        "labels": torch.from_numpy(labels).long(),
    }


def make_pd10_particle_dual_view_loader(
    dataset: PD10PairedParticleViewDataset,
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
        collate_fn=collate_pd10_particle_dual_view_batch,
        generator=generator,
    )


def move_pd10_particle_dual_view_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {
        "hlt": {key: value.to(device, non_blocking=True) for key, value in batch["hlt"].items()},
        "offline": {key: value.to(device, non_blocking=True) for key, value in batch["offline"].items()},
        "labels": batch["labels"].to(device, non_blocking=True),
    }


def _load_matching_branch_weights(branch, checkpoint: str | Path, *, device, branch_label: str) -> dict[str, Any]:
    torch = require_torch()
    payload = torch.load(checkpoint, map_location=device)
    source_state = dict(payload.get("model_state_dict") or {})
    target_state = branch.state_dict()
    matched: dict[str, Any] = {}
    skipped_shape: list[str] = []
    skipped_missing: list[str] = []
    for key, value in source_state.items():
        if key not in target_state:
            skipped_missing.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shape.append(key)
            continue
        matched[key] = value
    min_matched = max(1, int(0.50 * float(len(target_state))))
    if len(matched) < min_matched:
        raise RuntimeError(
            f"{branch_label} initialization from {checkpoint} matched only {len(matched)} tensors; "
            f"expected at least {min_matched} of {len(target_state)} target tensors"
        )
    branch.load_state_dict(matched, strict=False)
    return {
        "branch": branch_label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_experiment_step": payload.get("experiment_step"),
        "matched_tensors": int(len(matched)),
        "target_tensors": int(len(target_state)),
        "min_required_matched_tensors": int(min_matched),
        "skipped_missing_count": int(len(skipped_missing)),
        "skipped_shape_count": int(len(skipped_shape)),
        "skipped_shape_keys": skipped_shape[:20],
    }


def initialize_pd10_particle_dual_view_branches(
    model,
    *,
    hlt_checkpoint: str | Path,
    offline_checkpoint: str | Path,
    device,
) -> dict[str, Any]:
    return {
        "hlt_branch": _load_matching_branch_weights(
            model.hlt_branch,
            hlt_checkpoint,
            device=device,
            branch_label="hlt_branch",
        ),
        "offline_branch": _load_matching_branch_weights(
            model.offline_branch,
            offline_checkpoint,
            device=device,
            branch_label="offline_branch",
        ),
    }


def _optimizer_for_stage(model, config: PD10ParticleDualViewTeacherTrainConfig, *, stage: str):
    torch = require_torch()
    if stage == "head_warmup":
        model.set_branches_trainable(False)
        return torch.optim.AdamW(
            [parameter for parameter in model.head_parameters() if parameter.requires_grad],
            lr=float(config.head_warmup_lr),
            weight_decay=float(config.weight_decay),
        )
    model.set_branches_trainable(True)
    return torch.optim.AdamW(
        [
            {"params": list(model.branch_parameters()), "lr": float(config.branch_lr)},
            {"params": list(model.head_parameters()), "lr": float(config.head_lr)},
        ],
        weight_decay=float(config.weight_decay),
    )


def run_pd10_particle_dual_view_epoch(
    model,
    loader,
    *,
    device,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    grad_clip_norm: float = 0.0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    context = nullcontext() if is_train else torch.no_grad()
    criterion = torch.nn.CrossEntropyLoss()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_pd10_particle_dual_view_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            autocast_enabled = bool(amp and device.type == "cuda")
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(batch["hlt"], batch["offline"])
                loss = criterion(logits, batch["labels"])
            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    if float(grad_clip_norm) > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if float(grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    optimizer.step()
            labels = batch["labels"]
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_correct += int((logits.detach().argmax(dim=1) == labels).sum().detach().cpu().item())
            total_seen += batch_size
    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}
    return {
        "loss": total_loss / float(total_seen),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }


def _checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: PD10ParticleDualViewTeacherTrainConfig,
    metrics: Mapping[str, Any],
    branch_initialization: Mapping[str, Any],
):
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": dict(metrics),
        "label_names": list(LABEL_NAMES),
        "pf_feature_names": list(PF_FEATURE_NAMES),
        "model_config": getattr(model, "config", {}),
        "branch_initialization": dict(branch_initialization),
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_V2_STEP2_TRAIN_EXPERIMENT_STEP,
        "teacher_target": PD10_TEACHER_PARTICLE_DUAL_VIEW,
        "model_name": PD10_PARTICLE_DUAL_VIEW_MODEL_NAME,
        "allowed_inputs": PD10_EXTENDED_TEACHER_ALLOWED_INPUTS[PD10_TEACHER_PARTICLE_DUAL_VIEW],
        "student_deployment_inputs": "HLT_only",
        "returns_offline_particles": False,
        "uses_raw_offline_particles": True,
        "offline_privileged_inputs_loaded": True,
        "teacher_is_train_time_only_for_distillation": True,
        "inference_requires_offline_inputs": True,
        "teacher_inference_requires_offline_inputs": True,
        "inference_export_requires_teacher_features": False,
        "teacher_logits_train_time_only": True,
        "teacher_representations_train_time_only": True,
    }


def train_pd10_particle_dual_view_teacher(
    config: PD10ParticleDualViewTeacherTrainConfig,
    *,
    model=None,
) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_hlt, train_offline, train_source = _load_paired_views(
        manifest_path=config.manifest_path,
        hlt_cache_dir=config.hlt_cache_dir,
        split=config.train_split,
        data_dir=config.data_dir,
        max_jets=config.max_train_jets,
        seed=int(config.seed) + 101,
        verify_label_branches=config.verify_label_branches,
        read_chunk_size=config.read_chunk_size,
        verify_hlt_hash=config.verify_hlt_hash,
    )
    val_hlt, val_offline, val_source = _load_paired_views(
        manifest_path=config.manifest_path,
        hlt_cache_dir=config.hlt_cache_dir,
        split=config.val_split,
        data_dir=config.data_dir,
        max_jets=config.max_val_jets,
        seed=int(config.seed) + 202,
        verify_label_branches=config.verify_label_branches,
        read_chunk_size=config.read_chunk_size,
        verify_hlt_hash=config.verify_hlt_hash,
    )
    train_dataset = PD10PairedParticleViewDataset(train_hlt, train_offline)
    val_dataset = PD10PairedParticleViewDataset(val_hlt, val_offline)
    train_loader = make_pd10_particle_dual_view_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=int(config.seed) + 303,
    )
    val_loader = make_pd10_particle_dual_view_loader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=int(config.seed) + 404,
    )

    model = model or PD10ParticleDualViewTeacher(
        num_classes=PD10_NUM_CLASSES,
        model_size=config.model_size,
        fusion_hidden_dim=config.fusion_hidden_dim,
        representation_dim=config.representation_dim,
        dropout=config.dropout,
    )
    model = model.to(device)
    branch_initialization: dict[str, Any] = {"enabled": bool(config.initialize_branches)}
    if config.initialize_branches:
        branch_initialization.update(
            initialize_pd10_particle_dual_view_branches(
                model,
                hlt_checkpoint=config.hlt_teacher_checkpoint,
                offline_checkpoint=config.offline_teacher_checkpoint,
                device=device,
            )
        )
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    run_metadata = {
        "config": asdict(config),
        "contract": PD10_PARTICLE_DUAL_VIEW_TEACHER_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_V2_STEP2_TRAIN_EXPERIMENT_STEP,
        "teacher_target": PD10_TEACHER_PARTICLE_DUAL_VIEW,
        "model_name": config.model_name,
        "allowed_inputs": config.allowed_inputs,
        "student_deployment_inputs": "HLT_only",
        "returns_offline_particles": False,
        "uses_raw_offline_particles": True,
        "offline_privileged_inputs_loaded": True,
        "teacher_is_train_time_only_for_distillation": True,
        "inference_requires_offline_inputs": True,
        "teacher_inference_requires_offline_inputs": True,
        "inference_export_requires_teacher_features": False,
        "teacher_logits_train_time_only": True,
        "teacher_representations_train_time_only": True,
        "train_source": train_source,
        "val_source": val_source,
        "train_n_jets": len(train_dataset),
        "val_n_jets": len(val_dataset),
        "branch_initialization": branch_initialization,
        "selection_metric": "model_val_cross_entropy",
        "no_final_test_used_for_selection": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_accuracy = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    current_stage: str | None = None
    optimizer = None
    for epoch in range(1, int(config.epochs) + 1):
        stage = "head_warmup" if epoch <= int(config.head_warmup_epochs) else "full_finetune"
        if optimizer is None or stage != current_stage:
            optimizer = _optimizer_for_stage(model, config, stage=stage)
            current_stage = stage
        train_metrics = run_pd10_particle_dual_view_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_pd10_particle_dual_view_epoch(
            model,
            val_loader,
            device=device,
            amp=False,
            max_batches=config.max_val_batches,
        )
        row = {"epoch": int(epoch), "stage": stage, "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})

        improved = (
            val_metrics["loss"] < best_val_loss
            or (np.isclose(val_metrics["loss"], best_val_loss) and val_metrics["accuracy"] > best_val_accuracy)
        )
        torch.save(
            _checkpoint_payload(
                model,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=row,
                branch_initialization=branch_initialization,
            ),
            config.last_checkpoint_path,
        )
        if improved:
            best_val_loss = float(val_metrics["loss"])
            best_val_accuracy = float(val_metrics["accuracy"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=row,
                    branch_initialization=branch_initialization,
                ),
                config.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    checkpoint_sha = sha256_file(config.checkpoint_path)
    report = {
        "ok": True,
        "contract": PD10_PARTICLE_DUAL_VIEW_TEACHER_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_V2_STEP2_TRAIN_EXPERIMENT_STEP,
        "teacher_target": PD10_TEACHER_PARTICLE_DUAL_VIEW,
        "model_name": config.model_name,
        "allowed_inputs": config.allowed_inputs,
        "student_deployment_inputs": "HLT_only",
        "returns_offline_particles": False,
        "best_epoch": int(best_epoch),
        "best_model_val_cross_entropy": float(best_val_loss),
        "best_model_val_accuracy": float(best_val_accuracy),
        "epochs_completed": int(len(curves)),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "last_checkpoint": str(config.last_checkpoint_path),
        "selection_metric": "model_val_cross_entropy",
        "no_final_test_used_for_selection": True,
        "uses_raw_offline_particles": True,
        "offline_privileged_inputs_loaded": True,
        "teacher_is_train_time_only_for_distillation": True,
        "inference_requires_offline_inputs": True,
        "teacher_inference_requires_offline_inputs": True,
        "inference_export_requires_teacher_features": False,
        "teacher_logits_train_time_only": True,
        "teacher_representations_train_time_only": True,
        "branch_initialization": branch_initialization,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


def load_pd10_particle_dual_view_teacher_checkpoint(checkpoint: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(checkpoint, map_location=device)
    model_config = dict(payload.get("model_config") or {})
    model = PD10ParticleDualViewTeacher(
        num_classes=int(model_config.get("num_classes", PD10_NUM_CLASSES)),
        model_size=str(model_config.get("model_size", "base")),
        branch_config=model_config.get("branch_config"),
        fusion_hidden_dim=int(model_config.get("fusion_hidden_dim", PD10_PARTICLE_DUAL_VIEW_DEFAULT_FUSION_HIDDEN_DIM)),
        representation_dim=int(model_config.get("representation_dim", PD10_REPRESENTATION_DIM)),
        dropout=float(model_config.get("dropout", PD10_PARTICLE_DUAL_VIEW_DEFAULT_DROPOUT)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def _max_jets_for_cache_split(config: PD10ParticleDualViewTeacherCacheConfig, split: str) -> int | None:
    if split == "model_train":
        return config.max_model_train_jets
    if split == "model_val":
        return config.max_model_val_jets
    if split == "final_test":
        return config.max_final_test_jets
    return None


def pd10_particle_dual_view_cache_selection_seed(config: PD10ParticleDualViewTeacherCacheConfig, split: str) -> int:
    return int(config.control_seed) + 1009 * (PD10_PARTICLE_DUAL_VIEW_LOGIT_SPLITS.index(split) + 1)


def collect_pd10_particle_dual_view_outputs(
    model,
    hlt_view: JetView,
    offline_view: JetView,
    *,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch = require_torch()
    dataset = PD10PairedParticleViewDataset(hlt_view, offline_view)
    loader = make_pd10_particle_dual_view_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    logits_rows: list[np.ndarray] = []
    representation_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_pd10_particle_dual_view_batch_to_device(batch, device)
            logits, representations = model(batch["hlt"], batch["offline"], return_representation=True)
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            representation_rows.append(representations.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
    logits_np = (
        np.concatenate(logits_rows, axis=0).astype(np.float32, copy=False)
        if logits_rows
        else np.zeros((0, PD10_NUM_CLASSES), dtype=np.float32)
    )
    reps_np = (
        np.concatenate(representation_rows, axis=0).astype(np.float32, copy=False)
        if representation_rows
        else np.zeros((0, PD10_REPRESENTATION_DIM), dtype=np.float32)
    )
    labels_np = (
        np.concatenate(labels_rows, axis=0).astype(np.int64, copy=False)
        if labels_rows
        else np.zeros((0,), dtype=np.int64)
    )
    if logits_np.ndim != 2 or int(logits_np.shape[1]) != PD10_NUM_CLASSES:
        raise ValueError(f"particle dual-view logits must have shape [N, {PD10_NUM_CLASSES}], got {logits_np.shape}")
    if reps_np.ndim != 2:
        raise ValueError(f"particle dual-view representations must have shape [N, D], got {reps_np.shape}")
    if int(labels_np.shape[0]) != int(logits_np.shape[0]) or int(labels_np.shape[0]) != int(reps_np.shape[0]):
        raise ValueError("labels/logits/representations row count mismatch")
    if not np.isfinite(logits_np).all():
        raise FloatingPointError("particle dual-view logits contain non-finite values")
    if not np.isfinite(reps_np).all():
        raise FloatingPointError("particle dual-view representations contain non-finite values")
    return logits_np, reps_np, labels_np


def build_pd10_particle_dual_view_logit_block(
    config: PD10ParticleDualViewTeacherCacheConfig,
    split: str,
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    jet_ids: Sequence[JetIdentity],
    source_metadata: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
) -> PredictionBlock:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits.ndim != 2 or int(logits.shape[1]) != PD10_NUM_CLASSES:
        raise ValueError(f"particle dual-view logits must have shape [N, {PD10_NUM_CLASSES}], got {logits.shape}")
    if int(labels.shape[0]) != int(logits.shape[0]) or len(jet_ids) != int(labels.shape[0]):
        raise ValueError("labels/logits/jet_ids row count mismatch")
    identity_labels = np.asarray([int(identity.label) for identity in jet_ids], dtype=np.int64)
    if not np.array_equal(labels, identity_labels):
        raise ValueError("labels and jet ids are not aligned")
    metadata = {
        "contract": PD10_PARTICLE_DUAL_VIEW_LOGIT_CACHE_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_V2_STEP2_CACHE_EXPERIMENT_STEP,
        "teacher_target": config.teacher_target,
        "model_name": config.model_name,
        "model_kind": "pd10_particle_dual_view_teacher_logits",
        "architecture": "particle_dual_view_part_concat_abs_product",
        "source_view": "paired_hlt_offline_particles",
        "allowed_inputs": PD10_EXTENDED_TEACHER_ALLOWED_INPUTS[config.teacher_target],
        "split": split,
        "split_expected_size": int(PD10_SPLIT_SIZES[split]),
        "max_jets": _max_jets_for_cache_split(config, split),
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_experiment_step": checkpoint_payload.get("experiment_step"),
        "label_names": list(LABEL_NAMES),
        "num_classes": PD10_NUM_CLASSES,
        "student_deployment_inputs": "HLT_only",
        "teacher_logits_train_time_only": True,
        "teacher_representations_train_time_only": True,
        "returns_offline_particles": False,
        "uses_raw_offline_particles": True,
        "offline_privileged_inputs_loaded": True,
        "teacher_is_train_time_only_for_distillation": True,
        "inference_requires_offline_inputs": True,
        "teacher_inference_requires_offline_inputs": True,
        "inference_export_requires_teacher_features": False,
        **dict(source_metadata),
    }
    return PredictionBlock(
        model_name=config.model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=labels,
        jet_ids=list(jet_ids),
        metadata=metadata,
    )


def validate_pd10_particle_dual_view_logit_metadata(
    metadata: Mapping[str, Any],
    *,
    split: str | None = None,
) -> None:
    if metadata.get("contract") != PD10_PARTICLE_DUAL_VIEW_LOGIT_CACHE_CONTRACT:
        raise ValueError("particle dual-view logit cache contract mismatch")
    if metadata.get("experiment_step") != PD10_V2_STEP2_CACHE_EXPERIMENT_STEP:
        raise ValueError("particle dual-view logit cache step mismatch")
    if metadata.get("teacher_target") != PD10_TEACHER_PARTICLE_DUAL_VIEW:
        raise ValueError("particle dual-view logit cache teacher_target mismatch")
    if metadata.get("model_name") != PD10_PARTICLE_DUAL_VIEW_MODEL_NAME:
        raise ValueError("particle dual-view logit cache model_name mismatch")
    if split is not None and metadata.get("split") != split:
        raise ValueError(f"split mismatch: {metadata.get('split')} != {split}")
    if int(metadata.get("num_classes", -1)) != PD10_NUM_CLASSES:
        raise ValueError("particle dual-view logit cache must contain 10 classes")
    if int(metadata.get("n_jets", -1)) <= 0:
        raise ValueError("particle dual-view logit cache contains no rows")
    if metadata.get("allowed_inputs") != PD10_EXTENDED_TEACHER_ALLOWED_INPUTS[PD10_TEACHER_PARTICLE_DUAL_VIEW]:
        raise ValueError("particle dual-view logit cache allowed_inputs mismatch")
    if metadata.get("student_deployment_inputs") != "HLT_only":
        raise ValueError("particle dual-view logit cache must declare HLT-only student deployment")
    if not bool(metadata.get("teacher_logits_train_time_only")):
        raise ValueError("particle dual-view logits must be train-time only")
    if not bool(metadata.get("teacher_representations_train_time_only")):
        raise ValueError("particle dual-view representations must be train-time only")
    if bool(metadata.get("returns_offline_particles")):
        raise ValueError("particle dual-view caches must not return offline particles")
    if not bool(metadata.get("uses_raw_offline_particles")):
        raise ValueError("particle dual-view teacher must declare raw offline particle privilege")
    if not bool(metadata.get("inference_requires_offline_inputs")):
        raise ValueError("particle dual-view teacher must declare offline inputs are required for teacher inference")
    if not bool(metadata.get("teacher_inference_requires_offline_inputs")):
        raise ValueError("particle dual-view teacher must declare offline inputs are required for teacher inference")
    if bool(metadata.get("inference_export_requires_teacher_features")):
        raise ValueError("particle dual-view logit cache must not be required for student inference/export")


def load_pd10_particle_dual_view_logit_block(
    output_dir: str | Path,
    split: str,
    *,
    verify_hash: bool = True,
) -> PredictionBlock:
    block = load_prediction_block(output_dir, PD10_PARTICLE_DUAL_VIEW_MODEL_NAME, split, verify_hash=verify_hash)
    validate_pd10_particle_dual_view_logit_metadata(block.metadata, split=split)
    return block


def _existing_particle_cache_if_valid(config: PD10ParticleDualViewTeacherCacheConfig, split: str) -> dict[str, Any] | None:
    if not config.skip_existing or config.overwrite:
        return None
    logit_npz = config.logit_dir / f"{split}_predictions.npz"
    logit_meta = config.logit_dir / f"{split}_predictions_metadata.json"
    rep_npz = config.representation_dir / f"{split}_representations.npz"
    rep_meta = config.representation_dir / f"{split}_representations_metadata.json"
    if not (logit_npz.exists() and logit_meta.exists() and rep_npz.exists() and rep_meta.exists()):
        return None
    logit_block = load_pd10_particle_dual_view_logit_block(config.logit_output_dir, split)
    rep_block = load_pd10_teacher_representation_block(config.representation_output_dir, config.teacher_target, split)
    return {
        "logit_metadata": dict(logit_block.metadata),
        "representation_metadata": dict(rep_block.metadata),
    }


def cache_pd10_particle_dual_view_teacher(config: PD10ParticleDualViewTeacherCacheConfig) -> dict[str, Any]:
    torch = require_torch()
    device = resolve_device(config.device)
    Path(config.logit_output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.representation_output_dir).mkdir(parents=True, exist_ok=True)
    model, payload = load_pd10_particle_dual_view_teacher_checkpoint(config.checkpoint, device=device)
    logit_rows: list[dict[str, Any]] = []
    rep_rows: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}

    for split in config.splits:
        existing = _existing_particle_cache_if_valid(config, split)
        if existing is not None:
            split_reports[split] = {"skipped_existing": True, **existing}
            logit_rows.append(existing["logit_metadata"])
            rep_rows.append(existing["representation_metadata"])
            continue
        seed = pd10_particle_dual_view_cache_selection_seed(config, split)
        hlt_view, offline_view, source_metadata = _load_paired_views(
            manifest_path=config.manifest_path,
            hlt_cache_dir=config.hlt_cache_dir,
            split=split,
            data_dir=config.data_dir,
            max_jets=_max_jets_for_cache_split(config, split),
            seed=seed,
            verify_label_branches=config.verify_label_branches,
            read_chunk_size=config.read_chunk_size,
            verify_hlt_hash=config.verify_hlt_hash,
        )
        logits, representations, labels = collect_pd10_particle_dual_view_outputs(
            model,
            hlt_view,
            offline_view,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=device,
            seed=seed + 1,
            max_batches=config.max_batches,
        )
        if config.max_batches is not None:
            hlt_view = JetView(
                tokens=hlt_view.tokens[: labels.shape[0]],
                mask=hlt_view.mask[: labels.shape[0]],
                labels=hlt_view.labels[: labels.shape[0]],
                jet_ids=hlt_view.jet_ids[: labels.shape[0]],
                split=hlt_view.split,
                metadata=dict(hlt_view.metadata),
            )
        logit_block = build_pd10_particle_dual_view_logit_block(
            config,
            split,
            logits=logits,
            labels=labels,
            jet_ids=hlt_view.jet_ids,
            source_metadata=source_metadata,
            checkpoint_payload=payload,
        )
        logit_metadata = save_prediction_block(logit_block, config.logit_output_dir, overwrite=bool(config.overwrite))
        validate_pd10_particle_dual_view_logit_metadata(logit_metadata, split=split)
        rep_cfg = PD10TeacherRepresentationCacheConfig(
            teacher_target=config.teacher_target,
            output_dir=config.representation_output_dir,
            splits=(split,),
            representation_dim=int(representations.shape[1]),
            overwrite=bool(config.overwrite),
            skip_existing=bool(config.skip_existing),
            confirm_final_test=bool(config.confirm_final_test),
        )
        rep_block = build_pd10_teacher_representation_block(
            rep_cfg,
            split,
            representations=representations,
            labels=labels,
            jet_ids=hlt_view.jet_ids,
            source_metadata=source_metadata,
            extra_metadata={
                "checkpoint": str(config.checkpoint),
                "checkpoint_sha256": sha256_file(config.checkpoint),
                "checkpoint_epoch": payload.get("epoch"),
                "checkpoint_experiment_step": payload.get("experiment_step"),
                "teacher_logits_dir": str(config.logit_dir),
            },
        )
        rep_metadata = save_pd10_teacher_representation_block(
            rep_block,
            config.representation_output_dir,
            overwrite=bool(config.overwrite),
        )
        validate_pd10_teacher_representation_metadata(
            rep_metadata,
            teacher_target=config.teacher_target,
            split=split,
            representation_dim=int(representations.shape[1]),
        )
        split_reports[split] = {
            "skipped_existing": False,
            "logit_metadata": logit_metadata,
            "representation_metadata": rep_metadata,
        }
        logit_rows.append(logit_metadata)
        rep_rows.append(rep_metadata)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rep_manifest = write_pd10_teacher_representation_manifest(
        PD10TeacherRepresentationCacheConfig(
            teacher_target=config.teacher_target,
            output_dir=config.representation_output_dir,
            splits=tuple(config.splits),
            representation_dim=int((rep_rows[0] if rep_rows else {}).get("representation_dim", PD10_REPRESENTATION_DIM)),
            overwrite=bool(config.overwrite),
            skip_existing=bool(config.skip_existing),
            confirm_final_test=bool(config.confirm_final_test),
        ),
        rep_rows,
    )
    manifest = {
        "ok": True,
        "contract": PD10_PARTICLE_DUAL_VIEW_LOGIT_CACHE_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_V2_STEP2_CACHE_EXPERIMENT_STEP,
        "teacher_target": config.teacher_target,
        "model_name": config.model_name,
        "allowed_inputs": PD10_EXTENDED_TEACHER_ALLOWED_INPUTS[config.teacher_target],
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "logit_output_dir": str(config.logit_output_dir),
        "representation_output_dir": str(config.representation_output_dir),
        "teacher_logit_dir": str(config.logit_dir),
        "teacher_representation_dir": str(config.representation_dir),
        "splits": list(config.splits),
        "split_sizes": {split: int(PD10_SPLIT_SIZES[split]) for split in PD10_SPLIT_ORDER},
        "logit_rows": logit_rows,
        "representation_rows": rep_rows,
        "representation_manifest": rep_manifest,
        "split_reports": split_reports,
        "config": asdict(config),
    }
    save_json(config.logit_dir / PD10_PARTICLE_DUAL_VIEW_CACHE_MANIFEST, manifest)
    save_json(config.logit_dir / PD10_PARTICLE_DUAL_VIEW_CACHE_REPORT, manifest)
    return manifest


__all__ = [
    "PD10_PARTICLE_DUAL_VIEW_CACHE_MANIFEST",
    "PD10_PARTICLE_DUAL_VIEW_CACHE_REPORT",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_BATCH_SIZE",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_BRANCH_LR",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_DROPOUT",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_EARLY_STOP_PATIENCE",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_EPOCHS",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_FUSION_HIDDEN_DIM",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_LR",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_WARMUP_EPOCHS",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_HEAD_WARMUP_LR",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_SEED",
    "PD10_PARTICLE_DUAL_VIEW_DEFAULT_WEIGHT_DECAY",
    "PD10_PARTICLE_DUAL_VIEW_LOGIT_CACHE_CONTRACT",
    "PD10_PARTICLE_DUAL_VIEW_LOGIT_SPLITS",
    "PD10_PARTICLE_DUAL_VIEW_MODEL_NAME",
    "PD10_PARTICLE_DUAL_VIEW_TEACHER_CONTRACT",
    "PD10ParticleDualViewTeacher",
    "PD10ParticleDualViewTeacherCacheConfig",
    "PD10ParticleDualViewTeacherTrainConfig",
    "PD10PairedParticleViewDataset",
    "PD10_V2_STEP2_CACHE_EXPERIMENT_STEP",
    "PD10_V2_STEP2_EXPERIMENT_STEP",
    "PD10_V2_STEP2_TRAIN_EXPERIMENT_STEP",
    "align_pd10_hlt_offline_views",
    "build_pd10_particle_dual_view_logit_block",
    "cache_pd10_particle_dual_view_teacher",
    "collect_pd10_particle_dual_view_outputs",
    "collate_pd10_particle_dual_view_batch",
    "initialize_pd10_particle_dual_view_branches",
    "load_pd10_particle_dual_view_logit_block",
    "load_pd10_particle_dual_view_teacher_checkpoint",
    "make_pd10_particle_dual_view_loader",
    "move_pd10_particle_dual_view_batch_to_device",
    "pd10_particle_dual_view_cache_selection_seed",
    "pd10_particle_dual_view_logit_cache_dir",
    "pd10_particle_dual_view_representation_cache_dir",
    "pd10_particle_dual_view_teacher_checkpoint",
    "pd10_particle_dual_view_teacher_dir",
    "run_pd10_particle_dual_view_epoch",
    "train_pd10_particle_dual_view_teacher",
    "validate_pd10_particle_dual_view_logit_metadata",
]

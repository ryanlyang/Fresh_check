"""Step 4 training loop for PDV3 AV10-adapter privileged distillation.

The PDV3 student is always deployable from HLT inputs alone.  Privileged
teacher logits/representations are loaded only into the training/evaluation
loop as supervision targets and are never passed to the AV10 student forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import time
from typing import Any, Mapping, Sequence
import re

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.hlt_cache import (
    fixed_hlt_params_dict,
    fixed_hlt_params_from_profile,
    jet_identity_hash,
    load_cached_hlt_view,
    normalize_hlt_profile,
)
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.architecture_view_part.checkpoint import (
    compute_architecture_view_init_logit_diff_vs_baseline,
    sha256_file,
    warm_start_architecture_view_part_model,
)
from teacher_logit_reco.architecture_view_part.model import (
    ArchitectureViewResidualParT,
    build_architecture_view_residual_part,
)
from teacher_logit_reco.architecture_view_part.train import (
    ArchitectureViewTaggerTrainConfig,
    _delta_l2_mean_from_output,
    _finite_float,
    _flatten_scalar_diagnostics,
    _gradient_norm_diagnostics,
    _manifest_metadata,
    _metrics_without_prediction_arrays,
    _move_batch_to_device,
    _selection_score,
    _set_part_model_trainable,
    architecture_view_binary_projection_metrics,
    architecture_view_selection_metric_direction,
)
from teacher_logit_reco.local_compression_part.train import _optional_nonnegative_int, _optional_positive_int
from teacher_logit_reco.privileged_distill_10class.config import (
    PD10_REPRESENTATION_MODE_COSINE,
    PD10_TARGET_FULL_LOGITS,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_NONE,
    PD10_TEACHER_PARTICLE_DUAL_VIEW,
)
from teacher_logit_reco.privileged_distill_10class.student_data import (
    load_pd10_student_teacher_block,
    load_pd10_student_teacher_representation_block,
)
from teacher_logit_reco.privileged_distill_10class.students import pd10_student_loss
from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions
from teacher_logit_reco.set_matching.train import source_metadata
from teacher_logit_reco.subtoken_part.train import make_subtoken_hlt_loader

from .config import (
    PDV3_HLT_DEGRADATION_STRENGTH,
    PDV3_HLT_PROFILE,
    PDV3_LABEL_FILTER,
    PDV3_LABEL_NAMES,
    PDV3_MODEL_SPLIT_ORDER,
    PDV3_MODEL_SPLIT_SIZES,
    PDV3_NUM_CLASSES,
    normalize_pdv3_split_name,
)
from .students import (
    PDV3_LOSS_CE,
    PDV3_LOSS_LOGIT_KD,
    PDV3_LOSS_LOGIT_REP_KD,
    PDV3_TEACHER_NONE,
    PDV3_TEACHER_V1_DUAL_VIEW,
    PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW,
    PDV3_TRAINING_SCHEDULE_REP_FROZEN,
    PDV3_TRAINING_SCHEDULE_REP_FULL_UNFREEZE,
    PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
    PDV3_TRAINING_SCHEDULE_STAGED,
    PDV3StudentVariantSpec,
    normalize_pdv3_student_variant,
    pdv3_student_variant_spec,
)


PDV3_STEP4_TRAIN_STEP = "pdv3_step4_av10_adapter_student_kd_train"
PDV3_STUDENT_TRAIN_CONTRACT = "pdv3_av10_adapter_student_kd_train_v1"
PDV3_STUDENT_REPRESENTATION_SOURCE = "preclassifier_part_jet_representation"
PDV3_RESIDUAL_REPRESENTATION_PROJECTOR_CONTRACT = "pdv3_residual_delta_z_representation_projector_v1"


def _pd10_teacher_target_for_spec(spec: PDV3StudentVariantSpec) -> str:
    if spec.teacher_family == PDV3_TEACHER_NONE:
        return PD10_TEACHER_NONE
    if spec.teacher_family == PDV3_TEACHER_V1_DUAL_VIEW:
        return PD10_TEACHER_DUAL_VIEW
    if spec.teacher_family == PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW:
        return PD10_TEACHER_PARTICLE_DUAL_VIEW
    raise AssertionError(f"Unhandled PDV3 teacher family {spec.teacher_family!r}")


def _resolve_teacher_root(root: str | Path, model_name: str) -> Path:
    path = Path(root)
    if not model_name:
        return path
    return path.parent if path.name == model_name else path


def pdv3_effective_weight(target: float, warmup_epochs: int, epoch: int) -> float:
    """Linear warmup helper for KD alpha and representation beta."""

    target = float(target)
    warmup = int(warmup_epochs)
    if target <= 0.0:
        return 0.0
    if warmup <= 0:
        return target
    return target * min(1.0, float(epoch) / float(warmup))


def _pdv3_expected_hlt_params(profile: str, strength: float) -> dict[str, Any]:
    return fixed_hlt_params_dict(fixed_hlt_params_from_profile(profile, strength))


def _verify_pdv3_hlt_metadata(
    metadata: Mapping[str, Any],
    *,
    split: str,
    expected_profile: str,
    expected_strength: float,
    required: bool,
) -> dict[str, Any]:
    expected_profile = normalize_hlt_profile(expected_profile)
    expected_strength = float(expected_strength)
    expected_params = _pdv3_expected_hlt_params(expected_profile, expected_strength)
    actual_profile = metadata.get("hlt_profile")
    actual_strength = metadata.get("hlt_degradation_strength")
    actual_params = metadata.get("hlt_params") if isinstance(metadata.get("hlt_params"), Mapping) else {}
    problems: list[str] = []
    if actual_profile != expected_profile:
        problems.append(f"HLT profile is {actual_profile!r}, expected {expected_profile!r}")
    if actual_strength is None or abs(float(actual_strength) - expected_strength) > 1.0e-12:
        problems.append(f"HLT degradation strength is {actual_strength}, expected {expected_strength:g}")
    if actual_params != expected_params:
        problems.append("HLT parameters do not match the expected PDV3 HLT profile/strength")
    report = {
        "ok": not problems,
        "split": split,
        "expected_hlt_profile": expected_profile,
        "actual_hlt_profile": actual_profile,
        "expected_hlt_degradation_strength": expected_strength,
        "actual_hlt_degradation_strength": actual_strength,
        "expected_hlt_params": expected_params,
        "actual_hlt_params": actual_params,
        "problems": problems,
    }
    if bool(required) and problems:
        raise ValueError(f"PDV3 HLT cache contract failed for {split}: {'; '.join(problems)}")
    return report


@dataclass
class PDV3StudentTrainConfig:
    """Config for one Step 4 AV10-adapter student training run."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    baseline_checkpoint: str
    student_variant: str
    teacher_logit_root: str = ""
    teacher_representation_root: str = ""
    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    confirm_split_settings: bool = False
    confirm_final_test: bool = True
    seed: int = 7707
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 45
    adapter_lr: float | None = None
    part_lr: float | None = None
    weight_decay: float | None = None
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 6
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = PDV3_MODEL_SPLIT_SIZES["model_train"]
    max_val_jets: int | None = PDV3_MODEL_SPLIT_SIZES["model_val"]
    max_final_test_jets: int | None = PDV3_MODEL_SPLIT_SIZES["final_test"]
    selection_metric: str = "accuracy"
    compile_model: bool = False
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    require_baseline_split_manifest_hash: bool = True
    expected_hlt_profile: str = PDV3_HLT_PROFILE
    expected_hlt_degradation_strength: float = PDV3_HLT_DEGRADATION_STRENGTH
    view_dim: int = 32
    hidden_dim: int = 64
    pn_k: int = 16
    pn_layers: int = 2
    pfn_hidden_dim: int = 64
    pcnn_channels: int = 64
    pcnn_layers: int = 2
    fusion_hidden_dim: int = 96
    part_embed_dim: int = 128
    dropout: float = 0.05
    attention_dropout: float = 0.05
    gate_bias_init: float = -5.0
    random_control_seed: int = 2907
    delta_l2_weight: float = 1.0e-4
    input_delta_scale: float = 1.0
    use_feature_wise_input_delta_scales: bool = True
    freeze_input_delta_pid: bool = False
    freeze_input_delta_geometry: bool = False
    representation_dim: int = 128
    use_teacher_logits: bool | None = None
    use_teacher_representations: bool | None = None
    allow_baseline_from_scratch: bool = True
    final_test_teacher_diagnostics: bool = False
    representation_delta_l2_weight: float = 1.0e-4
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.output_dir = str(self.output_dir)
        self.manifest_path = str(self.manifest_path)
        self.hlt_cache_dir = str(self.hlt_cache_dir)
        self.baseline_checkpoint = str(self.baseline_checkpoint)
        self.student_variant = normalize_pdv3_student_variant(self.student_variant)
        self.train_split = normalize_pdv3_split_name(self.train_split)
        self.val_split = normalize_pdv3_split_name(self.val_split)
        self.final_test_split = normalize_pdv3_split_name(self.final_test_split)
        if (self.train_split, self.val_split, self.final_test_split) != PDV3_MODEL_SPLIT_ORDER:
            raise ValueError(f"PDV3 training split order must be {PDV3_MODEL_SPLIT_ORDER}")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set confirm_split_settings=True to acknowledge model_train/model_val selection")
        if not bool(self.confirm_final_test):
            raise ValueError("PDV3 Step 4 requires confirm_final_test=True for guarded final evaluation")
        if int(self.batch_size) <= 0 or int(self.eval_batch_size) <= 0 or int(self.epochs) <= 0:
            raise ValueError("batch_size, eval_batch_size, and epochs must be positive")
        self.batch_size = int(self.batch_size)
        self.eval_batch_size = int(self.eval_batch_size)
        self.epochs = int(self.epochs)
        self.num_workers = int(self.num_workers)
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        self.max_train_batches = _optional_nonnegative_int(self.max_train_batches, field_name="max_train_batches")
        self.max_val_batches = _optional_nonnegative_int(self.max_val_batches, field_name="max_val_batches")
        self.max_final_test_batches = _optional_nonnegative_int(
            self.max_final_test_batches,
            field_name="max_final_test_batches",
        )
        self.max_train_jets = _optional_positive_int(self.max_train_jets, field_name="max_train_jets")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.max_final_test_jets = _optional_positive_int(self.max_final_test_jets, field_name="max_final_test_jets")
        for split, value in (
            ("model_train", self.max_train_jets),
            ("model_val", self.max_val_jets),
            ("final_test", self.max_final_test_jets),
        ):
            if value is not None and int(value) > int(PDV3_MODEL_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PDV3_MODEL_SPLIT_SIZES[split]}")
        if self.selection_metric not in ("accuracy", "macro_per_class_accuracy", "loss"):
            raise ValueError("PDV3 Step 4 selection_metric must be accuracy, macro_per_class_accuracy, or loss")
        self.expected_hlt_profile = normalize_hlt_profile(self.expected_hlt_profile)
        if self.expected_hlt_profile != PDV3_HLT_PROFILE:
            raise ValueError(f"PDV3 Step 4 is locked to HLT profile {PDV3_HLT_PROFILE}")
        self.expected_hlt_degradation_strength = float(self.expected_hlt_degradation_strength)
        if abs(self.expected_hlt_degradation_strength - PDV3_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
            raise ValueError(f"PDV3 Step 4 is locked to HLT degradation {PDV3_HLT_DEGRADATION_STRENGTH:g}")
        self.representation_dim = int(self.representation_dim)
        if self.representation_dim <= 0:
            raise ValueError("representation_dim must be positive")
        if self.use_teacher_logits is not None:
            self.use_teacher_logits = bool(self.use_teacher_logits)
        if self.use_teacher_representations is not None:
            self.use_teacher_representations = bool(self.use_teacher_representations)
        spec = self.spec
        active_logits = self.uses_teacher_logits
        active_reps = self.uses_teacher_representations
        if spec.teacher_family == PDV3_TEACHER_NONE and (active_logits or active_reps):
            raise ValueError("CE-only PDV3 variants cannot enable teacher supervision")
        if active_reps and spec.teacher_family != PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW:
            raise ValueError("representation KD is only supported for V2 particle-dual-view teacher variants")
        if active_logits and not spec.teacher_logit_name:
            raise ValueError(f"{self.student_variant} has no teacher_logit_name configured")
        if active_reps and not spec.teacher_representation_name:
            raise ValueError(f"{self.student_variant} has no teacher_representation_name configured")
        if spec.teacher_family != PDV3_TEACHER_NONE and not active_logits and not active_reps:
            raise ValueError("KD variants must keep at least one teacher signal active")
        for field_name in (
            "grad_clip_norm",
            "delta_l2_weight",
            "dropout",
            "attention_dropout",
            "input_delta_scale",
            "representation_delta_l2_weight",
        ):
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
            setattr(self, field_name, value)
        if self.dropout >= 1.0 or self.attention_dropout >= 1.0:
            raise ValueError("dropout values must be in [0, 1)")

    @property
    def spec(self) -> PDV3StudentVariantSpec:
        return pdv3_student_variant_spec(self.student_variant)

    @property
    def adapter_lr_value(self) -> float:
        return float(self.spec.adapter_lr if self.adapter_lr is None else self.adapter_lr)

    @property
    def baseline_checkpoint_exists(self) -> bool:
        return bool(self.baseline_checkpoint and Path(self.baseline_checkpoint).is_file())

    @property
    def trains_baseline_from_scratch(self) -> bool:
        spec = self.spec
        return bool(
            not self.baseline_checkpoint_exists
            and bool(self.allow_baseline_from_scratch)
            and bool(spec.is_baseline)
            and spec.loss_mode == PDV3_LOSS_CE
        )

    @property
    def part_lr_value(self) -> float:
        if self.trains_baseline_from_scratch and self.part_lr is None:
            return 1.0e-3
        return float(self.spec.part_lr if self.part_lr is None else self.part_lr)

    @property
    def weight_decay_value(self) -> float:
        return float(self.spec.weight_decay if self.weight_decay is None else self.weight_decay)

    @property
    def teacher_target(self) -> str:
        return _pd10_teacher_target_for_spec(self.spec)

    @property
    def uses_teacher_logits(self) -> bool:
        if self.use_teacher_logits is None:
            return bool(self.spec.requires_teacher_logits)
        return bool(self.use_teacher_logits)

    @property
    def uses_teacher_representations(self) -> bool:
        if self.use_teacher_representations is None:
            return bool(self.spec.requires_teacher_representations)
        return bool(self.use_teacher_representations)

    @property
    def teacher_logit_cache_root(self) -> str:
        if not self.uses_teacher_logits:
            return ""
        if not self.teacher_logit_root:
            raise ValueError(f"{self.student_variant} requires teacher_logit_root")
        return str(_resolve_teacher_root(self.teacher_logit_root, self.spec.teacher_logit_name))

    @property
    def teacher_representation_cache_root(self) -> str:
        if not self.uses_teacher_representations:
            return ""
        if not self.teacher_representation_root:
            raise ValueError(f"{self.student_variant} requires teacher_representation_root")
        return str(_resolve_teacher_root(self.teacher_representation_root, self.spec.teacher_representation_name))

    def architecture_config(self) -> ArchitectureViewTaggerTrainConfig:
        spec = self.spec
        return ArchitectureViewTaggerTrainConfig(
            output_dir=self.output_dir,
            manifest_path=self.manifest_path,
            hlt_cache_dir=self.hlt_cache_dir,
            baseline_checkpoint=self.baseline_checkpoint or "__pdv3_from_scratch_baseline__",
            input_source="hlt",
            train_split=self.train_split,
            val_split=self.val_split,
            stack_val_split="stack_val",
            final_test_split=self.final_test_split,
            confirm_split_settings=True,
            confirm_final_test=True,
            seed=int(self.seed),
            batch_size=int(self.batch_size),
            eval_batch_size=int(self.eval_batch_size),
            epochs=int(self.epochs),
            adapter_lr=self.adapter_lr_value,
            part_lr=self.part_lr_value,
            weight_decay=self.weight_decay_value,
            num_workers=int(self.num_workers),
            device=str(self.device),
            amp=bool(self.amp),
            grad_clip_norm=float(self.grad_clip_norm),
            early_stop_patience=int(self.early_stop_patience),
            max_train_batches=self.max_train_batches,
            max_val_batches=self.max_val_batches,
            max_stack_val_batches=0,
            max_final_test_batches=self.max_final_test_batches,
            max_train_jets=self.max_train_jets,
            max_val_jets=self.max_val_jets,
            max_stack_val_jets=10,
            max_final_test_jets=self.max_final_test_jets,
            selection_metric=str(self.selection_metric),
            num_classes=PDV3_NUM_CLASSES,
            compile_model=bool(self.compile_model),
            verify_hlt_hash=bool(self.verify_hlt_hash),
            verify_hlt_params=bool(self.verify_hlt_params),
            require_baseline_split_manifest_hash=bool(self.require_baseline_split_manifest_hash),
            expected_hlt_degradation_strength=float(self.expected_hlt_degradation_strength),
            label_names=PDV3_LABEL_NAMES,
            label_filter=PDV3_LABEL_FILTER,
            variant=spec.architecture_view_variant,
            view_dim=int(self.view_dim),
            hidden_dim=int(self.hidden_dim),
            pn_k=int(self.pn_k),
            pn_layers=int(self.pn_layers),
            pfn_hidden_dim=int(self.pfn_hidden_dim),
            pcnn_channels=int(self.pcnn_channels),
            pcnn_layers=int(self.pcnn_layers),
            fusion_hidden_dim=int(self.fusion_hidden_dim),
            part_embed_dim=int(self.part_embed_dim),
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            gate_bias_init=float(self.gate_bias_init),
            random_control_seed=int(self.random_control_seed),
            delta_l2_weight=float(self.delta_l2_weight),
            freeze_part_epochs=int(spec.freeze_part_epochs),
            input_delta_scale=float(self.input_delta_scale),
            use_feature_wise_input_delta_scales=bool(self.use_feature_wise_input_delta_scales),
            freeze_input_delta_pid=bool(self.freeze_input_delta_pid),
            freeze_input_delta_geometry=bool(self.freeze_input_delta_geometry),
        )


class PDV3ArchitectureViewStudentDataset:
    """HLT raw-token dataset with optional teacher logits/representations."""

    def __init__(
        self,
        *,
        tokens: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
        jet_ids: Sequence[Any],
        split: str,
        label_names: Sequence[str],
        metadata: Mapping[str, Any],
        teacher_logits: np.ndarray | None = None,
        teacher_representations: np.ndarray | None = None,
        teacher_metadata: Mapping[str, Any] | None = None,
        teacher_representation_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.tokens = np.asarray(tokens, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=bool)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.jet_ids = list(jet_ids)
        self.split = str(split)
        self.label_names = tuple(str(name) for name in label_names)
        self.label_filter = PDV3_LABEL_FILTER
        self.teacher_logits = None if teacher_logits is None else np.asarray(teacher_logits, dtype=np.float32)
        self.teacher_representations = (
            None if teacher_representations is None else np.asarray(teacher_representations, dtype=np.float32)
        )
        self.teacher_metadata = dict(teacher_metadata or {})
        self.teacher_representation_metadata = dict(teacher_representation_metadata or {})
        self.metadata = dict(metadata)
        self.metadata.update(
            {
                "split": self.split,
                "n_jets": int(self.labels.shape[0]),
                "label_names": list(self.label_names),
                "label_filter": list(PDV3_LABEL_FILTER),
                "label_counts": self.label_counts(),
                "student_allowed_inputs": "HLT_only",
                "teacher_logits_train_time_only": self.teacher_logits is not None,
                "teacher_representations_train_time_only": self.teacher_representations is not None,
                "jet_identity_hash": jet_identity_hash(self.jet_ids),
            }
        )
        if self.teacher_logits is not None and int(self.teacher_logits.shape[0]) != int(self.labels.shape[0]):
            raise ValueError("teacher logits row count does not match dataset labels")
        if self.teacher_representations is not None and int(self.teacher_representations.shape[0]) != int(self.labels.shape[0]):
            raise ValueError("teacher representations row count does not match dataset labels")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "labels": self.labels[index],
            "indices": np.int64(index),
        }

    def label_counts(self) -> dict[str, int]:
        return {
            self.label_names[index]: int(np.sum(self.labels == index))
            for index in range(len(self.label_names))
        }


def _limit_first_n(array: np.ndarray, max_jets: int | None) -> np.ndarray:
    if max_jets is None:
        return array
    return array[: min(int(max_jets), int(array.shape[0]))]


def _check_block_alignment(
    *,
    split: str,
    dataset_labels: np.ndarray,
    dataset_jet_ids: Sequence[Any],
    teacher_labels: np.ndarray,
    teacher_jet_ids: Sequence[Any],
    teacher_name: str,
) -> None:
    n = int(dataset_labels.shape[0])
    if int(teacher_labels.shape[0]) < n:
        raise ValueError(f"{teacher_name}/{split} has fewer rows than the student dataset")
    teacher_labels = np.asarray(teacher_labels[:n], dtype=np.int64)
    if not np.array_equal(dataset_labels, teacher_labels):
        raise ValueError(f"{teacher_name}/{split} labels are not aligned with the student dataset")
    teacher_subset = list(teacher_jet_ids[:n])
    if list(dataset_jet_ids) != teacher_subset:
        raise ValueError(f"{teacher_name}/{split} jet identities are not aligned with the student dataset")


def _load_pdv3_student_dataset(
    config: PDV3StudentTrainConfig,
    split: str,
    *,
    max_jets: int | None,
    load_teacher_supervision: bool = True,
) -> PDV3ArchitectureViewStudentDataset:
    view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    audit = _verify_pdv3_hlt_metadata(
        view.metadata,
        split=split,
        expected_profile=str(config.expected_hlt_profile),
        expected_strength=float(config.expected_hlt_degradation_strength),
        required=bool(config.verify_hlt_params),
    )
    if view.metadata.get("view") not in (None, "fixed_hlt"):
        raise ValueError(f"PDV3 students require fixed_hlt cache rows, got {view.metadata.get('view')!r}")
    labels = np.asarray(view.labels, dtype=np.int64)
    if not np.array_equal(np.unique(labels), np.asarray(PDV3_LABEL_FILTER, dtype=np.int64)):
        # Some tiny smoke tests may not contain all classes, but every observed label
        # must still be one of the fixed JetClass indices.
        if not set(int(v) for v in np.unique(labels)).issubset(set(PDV3_LABEL_FILTER)):
            raise ValueError("PDV3 student cache contains labels outside the 10-class JetClass filter")
    tokens = np.asarray(view.tokens, dtype=np.float32)
    mask = np.asarray(view.mask, dtype=bool)
    jet_ids = list(view.jet_ids)
    if max_jets is not None:
        limit = min(int(max_jets), int(labels.shape[0]))
        tokens = tokens[:limit]
        mask = mask[:limit]
        labels = labels[:limit]
        jet_ids = jet_ids[:limit]
    spec = config.spec
    teacher_block = None
    teacher_rep_block = None
    teacher_logits = None
    teacher_reps = None
    if bool(load_teacher_supervision) and config.uses_teacher_logits:
        teacher_block = load_pd10_student_teacher_block(
            config.teacher_logit_cache_root,
            config.teacher_target,
            split,
            verify_hash=True,
        )
        if teacher_block is None:
            raise FileNotFoundError(f"teacher logits are required for {config.student_variant}")
        _check_block_alignment(
            split=split,
            dataset_labels=labels,
            dataset_jet_ids=jet_ids,
            teacher_labels=teacher_block.labels,
            teacher_jet_ids=teacher_block.jet_ids,
            teacher_name=spec.teacher_logit_name,
        )
        teacher_logits = _limit_first_n(np.asarray(teacher_block.logits, dtype=np.float32), len(labels))
    if bool(load_teacher_supervision) and config.uses_teacher_representations:
        teacher_rep_block = load_pd10_student_teacher_representation_block(
            config.teacher_representation_cache_root,
            config.teacher_target,
            split,
            verify_hash=True,
        )
        if teacher_rep_block is None:
            raise FileNotFoundError(f"teacher representations are required for {config.student_variant}")
        _check_block_alignment(
            split=split,
            dataset_labels=labels,
            dataset_jet_ids=jet_ids,
            teacher_labels=teacher_rep_block.labels,
            teacher_jet_ids=teacher_rep_block.jet_ids,
            teacher_name=spec.teacher_representation_name,
        )
        teacher_reps = _limit_first_n(np.asarray(teacher_rep_block.representations, dtype=np.float32), len(labels))
    metadata = {
        **dict(view.metadata),
        "input_source": "hlt",
        "source_view": "fixed_hlt",
        "raw_token_dim": int(tokens.shape[-1]),
        "max_constits": int(tokens.shape[1]),
        "max_jets_limit": None if max_jets is None else int(max_jets),
        "hlt_protocol_audit": audit,
        "source_manifest_hash": view.metadata.get("source_manifest_hash"),
        "hlt_content_hash": view.metadata.get("hlt_content_hash"),
        "source_hlt_jet_identity_hash": view.metadata.get("jet_identity_hash"),
        "teacher_supervision_loaded": bool(load_teacher_supervision)
        and (config.uses_teacher_logits or config.uses_teacher_representations),
        "teacher_supervision_policy": "train_or_model_val_only" if bool(load_teacher_supervision) else "disabled",
    }
    return PDV3ArchitectureViewStudentDataset(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        label_names=PDV3_LABEL_NAMES,
        metadata=metadata,
        teacher_logits=teacher_logits,
        teacher_representations=teacher_reps,
        teacher_metadata={} if teacher_block is None else teacher_block.metadata,
        teacher_representation_metadata={} if teacher_rep_block is None else teacher_rep_block.metadata,
    )


def _verify_pdv3_dataset_manifest_hash(
    dataset: PDV3ArchitectureViewStudentDataset,
    *,
    expected_manifest_hash: str | None,
    split: str,
    cache_dir: str,
) -> None:
    if not expected_manifest_hash:
        return
    actual = dataset.metadata.get("source_manifest_hash")
    if not actual:
        raise ValueError(
            f"PDV3 {split} cache loaded from {cache_dir!r} is missing source_manifest_hash; "
            f"expected {expected_manifest_hash}"
        )
    if str(actual) != str(expected_manifest_hash):
        raise ValueError(
            f"PDV3 {split} cache source_manifest_hash mismatch for {cache_dir!r}: "
            f"{actual} != {expected_manifest_hash}"
        )


def _teacher_tensor_for_batch(loader, batch: Mapping[str, Any], field_name: str, device) -> Any | None:
    torch = require_torch()
    array = getattr(loader.dataset, field_name, None)
    if array is None:
        return None
    indices = batch["indices"].detach().cpu().numpy().astype(np.int64, copy=False)
    return torch.as_tensor(np.asarray(array)[indices], dtype=torch.float32, device=device)


class PDV3RepresentationProjector(require_torch().nn.Module):
    """Residual ``delta_z`` head for V2 representation KD.

    The student representation source is a deployable HLT-only jet vector
    ``z_hlt``.  This module projects it into teacher-representation space,
    predicts a zero-initialized residual ``delta_z``, and returns
    ``z_corr = z_hlt_projected + gate * delta_z`` for the cosine RepKD loss.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.05) -> None:
        torch = require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_value = float(dropout)
        if self.input_dim <= 0 or self.output_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim, output_dim, and hidden_dim must be positive")
        self.source_norm = torch.nn.LayerNorm(self.input_dim)
        self.base_projection = (
            torch.nn.Identity()
            if self.input_dim == self.output_dim
            else torch.nn.Linear(self.input_dim, self.output_dim)
        )
        self.delta_in = torch.nn.Linear(self.input_dim, self.hidden_dim * 2)
        self.dropout = torch.nn.Dropout(self.dropout_value)
        self.delta_out = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.gate_logit = torch.nn.Parameter(torch.tensor(-5.0, dtype=torch.float32))
        torch.nn.init.zeros_(self.delta_out.weight)
        torch.nn.init.zeros_(self.delta_out.bias)
        self._last_output: dict[str, Any] = {}

    def forward(self, representation_source):
        torch = require_torch()
        source = representation_source.float()
        normalized = self.source_norm(source)
        base = self.base_projection(source)
        hidden_pair = self.delta_in(normalized)
        hidden, gate_hidden = hidden_pair.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate_hidden) * hidden
        hidden = self.dropout(hidden)
        raw_delta = self.delta_out(hidden)
        gate = torch.sigmoid(self.gate_logit).to(dtype=raw_delta.dtype, device=raw_delta.device)
        delta_z = raw_delta * gate
        corrected = base + delta_z
        self._last_output = {
            "z_hlt_projected": base,
            "raw_delta_z": raw_delta,
            "delta_z": delta_z,
            "z_corr": corrected,
            "gate": gate,
        }
        return corrected

    def delta_z_l2_mean(self):
        torch = require_torch()
        delta_z = self._last_output.get("delta_z")
        if delta_z is None:
            return torch.zeros((), device=self.gate_logit.device, dtype=self.gate_logit.dtype)
        return delta_z.float().square().sum(dim=-1).mean()

    def diagnostics(self) -> dict[str, Any]:
        output = self._last_output
        if not output:
            return {
                "representation_projector_contract": PDV3_RESIDUAL_REPRESENTATION_PROJECTOR_CONTRACT,
                "representation_projector_has_forward_state": False,
            }
        delta_z = output["delta_z"].detach().float()
        raw_delta = output["raw_delta_z"].detach().float()
        base = output["z_hlt_projected"].detach().float()
        corrected = output["z_corr"].detach().float()
        gate = output["gate"].detach().float()
        delta_norm = delta_z.norm(dim=-1)
        raw_delta_norm = raw_delta.norm(dim=-1)
        base_norm = base.norm(dim=-1)
        corrected_norm = corrected.norm(dim=-1)
        ratio = delta_norm / base_norm.clamp_min(1.0e-12)
        return {
            "representation_projector_contract": PDV3_RESIDUAL_REPRESENTATION_PROJECTOR_CONTRACT,
            "representation_projector_has_forward_state": True,
            "representation_projector_kind": "residual_delta_z",
            "representation_projector_zero_init_delta": True,
            "representation_projector_residual_form": "z_corr=z_hlt_projected+sigmoid(gate_logit)*delta_z",
            "representation_projector_gate": float(gate.detach().cpu().item()),
            "z_hlt_projected_norm_mean": float(base_norm.mean().detach().cpu().item()),
            "z_corr_norm_mean": float(corrected_norm.mean().detach().cpu().item()),
            "raw_delta_z_norm_mean": float(raw_delta_norm.mean().detach().cpu().item()),
            "delta_z_norm_mean": float(delta_norm.mean().detach().cpu().item()),
            "delta_z_norm_max": float(delta_norm.max().detach().cpu().item()) if int(delta_norm.numel()) else 0.0,
            "delta_z_l2_mean": float(delta_z.square().sum(dim=-1).mean().detach().cpu().item()),
            "delta_z_to_z_hlt_norm_ratio_mean": float(ratio.mean().detach().cpu().item()),
        }

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "contract": PDV3_RESIDUAL_REPRESENTATION_PROJECTOR_CONTRACT,
            "source": PDV3_STUDENT_REPRESENTATION_SOURCE,
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "hidden_dim": int(self.hidden_dim),
            "dropout": float(self.dropout_value),
            "base_projection": "identity" if self.input_dim == self.output_dim else "linear",
            "residual_form": "z_corr=z_hlt_projected+sigmoid(gate_logit)*delta_z",
            "zero_init_delta_projection": True,
            "gate_init": -5.0,
            "class_name": type(self).__name__,
        }


def _masked_mean(rows, mask):
    torch = require_torch()
    mask = mask.bool()
    valid = mask[:, : int(rows.shape[1])].to(dtype=rows.dtype)
    rows = rows[:, : int(valid.shape[1])]
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (rows * valid[:, :, None]).sum(dim=1) / denom


def pdv3_student_representation_source_from_output(output) -> Any:
    """Return the HLT-only jet vector immediately before the ParT classifier head."""

    torch = require_torch()
    representation = getattr(output, "part_jet_representation", None)
    if representation is None:
        raise ValueError(
            "PDV3 representation KD requires ArchitectureViewResidualPartOutput.part_jet_representation. "
            "The AV10/ParT wrapper must capture the pre-classifier ParT jet vector during forward."
        )
    representation = representation.float()
    if int(representation.ndim) != 2:
        raise ValueError(
            "PDV3 representation KD expects a rank-2 [batch, dim] pre-classifier jet representation; "
            f"got shape {tuple(representation.shape)}"
        )
    return representation


def _representation_source_dim(model: ArchitectureViewResidualParT) -> int:
    return int(model.config.part_embed_dim)


def _set_module_trainable(module: Any | None, trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(bool(trainable))
    module.train(bool(trainable))


def _optimizer_for_pdv3(
    model: ArchitectureViewResidualParT,
    representation_projector: Any | None,
    config: PDV3StudentTrainConfig,
):
    torch = require_torch()
    adapter_params: list[Any] = []
    for module in model.adapter_modules(active_only=True):
        adapter_params.extend([param for param in module.parameters() if param.requires_grad])
    if representation_projector is not None:
        adapter_params.extend([param for param in representation_projector.parameters() if param.requires_grad])
    head_params = [param for param in _part_head_parameters(model) if param.requires_grad]
    head_param_ids = {id(param) for param in head_params}
    part_params = [
        param for param in model.part_model.parameters() if param.requires_grad and id(param) not in head_param_ids
    ]
    groups: list[dict[str, Any]] = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": config.adapter_lr_value, "name": "adapter_and_rep_head"})
    if head_params:
        groups.append({"params": head_params, "lr": config.part_lr_value, "name": "classifier_head"})
    if part_params:
        groups.append({"params": part_params, "lr": config.part_lr_value, "name": "part"})
    if not groups:
        return None
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay_value)


def _set_active_adapter_modules_trainable(
    model: ArchitectureViewResidualParT,
    trainable_names: Sequence[str],
) -> tuple[str, ...]:
    """Set trainability for active adapter modules and keep dormant modules frozen."""

    active_names = tuple(str(name) for name in model.active_adapter_module_names())
    requested = {str(name) for name in trainable_names}
    enabled: list[str] = []
    for name, module in model.adapter_module_map().items():
        trainable = str(name) in requested and str(name) in set(active_names)
        _set_module_trainable(module, trainable)
        if trainable:
            enabled.append(str(name))
    return tuple(name for name in active_names if name in set(enabled))


def _set_part_head_trainable(model: ArchitectureViewResidualParT, trainable: bool) -> None:
    if not hasattr(model, "_part_head_parameter_modules"):
        return
    for module in model._part_head_parameter_modules():
        _set_module_trainable(module, bool(trainable))


def _part_head_parameters(model: ArchitectureViewResidualParT) -> tuple[Any, ...]:
    if not hasattr(model, "_part_head_parameter_modules"):
        return ()
    output: list[Any] = []
    seen: set[int] = set()
    for module in model._part_head_parameter_modules():
        for parameter in module.parameters():
            ident = id(parameter)
            if ident in seen:
                continue
            seen.add(ident)
            output.append(parameter)
    return tuple(output)


def _set_part_model_trainable_scope(
    model: ArchitectureViewResidualParT,
    scope: str,
    *,
    upper_block_count: int = 2,
) -> dict[str, Any]:
    """Apply full/frozen/upper-block trainability to the wrapped ParT body."""

    scope = str(scope)
    upper_block_count = max(1, int(upper_block_count))
    if scope == "full":
        _set_part_model_trainable(model, True)
        return {
            "part_trainability_scope": "full",
            "upper_unfreeze_block_count": int(upper_block_count),
            "trainable_part_module_names": ["part_model"],
            "trainable_part_module_count": 1,
        }
    _set_part_model_trainable(model, False)
    if scope != "upper":
        return {
            "part_trainability_scope": "frozen",
            "upper_unfreeze_block_count": int(upper_block_count),
            "trainable_part_module_names": [],
            "trainable_part_module_count": 0,
        }

    config = getattr(model.part_model, "config", {}) or {}
    num_layers = int(config.get("num_layers") or 0)
    threshold = max(0, num_layers - upper_block_count) if num_layers > 0 else None
    selected_modules: list[str] = []
    selected_parameter_ids: set[int] = set()
    block_re = re.compile(r"(?:^|\.)(?:blocks|particle_blocks)\.(\d+)(?:\.|$)")
    cls_block_re = re.compile(r"(?:^|\.)(?:cls_blocks|class_blocks)\.(\d+)(?:\.|$)")
    for name, module in model.part_model.named_modules():
        if not name:
            continue
        match = block_re.search(name)
        cls_match = cls_block_re.search(name)
        selected = False
        if match is not None:
            block_index = int(match.group(1))
            selected = threshold is None or block_index >= int(threshold)
        elif cls_match is not None:
            selected = True
        if not selected:
            continue
        any_parameter = False
        for parameter in module.parameters(recurse=True):
            selected_parameter_ids.add(id(parameter))
            parameter.requires_grad_(True)
            any_parameter = True
        if any_parameter:
            selected_modules.append(str(name))
    if selected_parameter_ids:
        model.part_model.train(True)
        setattr(model, "_force_part_model_eval", False)
    return {
        "part_trainability_scope": "upper",
        "upper_unfreeze_block_count": int(upper_block_count),
        "trainable_part_module_names": selected_modules[:64],
        "trainable_part_module_count": len(selected_modules),
        "trainable_part_parameter_count": len(selected_parameter_ids),
    }


def _phase_weight_metadata(spec: PDV3StudentVariantSpec) -> dict[str, Any]:
    return {
        "configured_kd_alpha": float(spec.kd_alpha),
        "configured_rep_beta": float(spec.rep_beta),
        "kd_warmup_epochs": int(spec.kd_warmup_epochs),
        "rep_warmup_epochs": int(spec.rep_warmup_epochs),
    }


def _with_phase_weight_metadata(
    phases: Sequence[Mapping[str, Any]],
    spec: PDV3StudentVariantSpec,
) -> list[dict[str, Any]]:
    weight_metadata = _phase_weight_metadata(spec)
    return [{**dict(phase), **weight_metadata} for phase in phases]


def pdv3_student_training_phase_plan(spec: PDV3StudentVariantSpec) -> list[dict[str, Any]]:
    """Human-readable training phase plan for Step 6 combined adapters."""

    if str(spec.training_schedule) == PDV3_TRAINING_SCHEDULE_REP_FROZEN:
        adapter_label: Any = [] if spec.architecture_adapter_type == "none" else "all_active"
        return _with_phase_weight_metadata(
            [
                {
                    "phase_name": "delta_z_frozen_repkd",
                    "epoch_start": 1,
                    "epoch_end": "end",
                    "part_model_trainable": False,
                    "part_trainability_scope": "frozen",
                    "classifier_head_trainable": True,
                    "representation_projector_trainable": bool(spec.requires_teacher_representations),
                    "trainable_adapter_module_names": adapter_label,
                }
            ],
            spec,
        )
    if str(spec.training_schedule) in (
        PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
        PDV3_TRAINING_SCHEDULE_REP_FULL_UNFREEZE,
    ):
        warmup_epochs = max(1, int(spec.freeze_part_epochs))
        second_phase_name = (
            "delta_z_upper_unfreeze_repkd"
            if str(spec.training_schedule) == PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE
            else "delta_z_full_gentle_unfreeze_repkd"
        )
        second_scope = (
            "upper"
            if str(spec.training_schedule) == PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE
            else "full"
        )
        adapter_label: Any = [] if spec.architecture_adapter_type == "none" else "all_active"
        second_adapter_label = (
            []
            if "freeze_adapter_after_warmup" in str(spec.freeze_policy)
            else adapter_label
        )
        return _with_phase_weight_metadata(
            [
                {
                    "phase_name": "delta_z_frozen_warmup",
                    "epoch_start": 1,
                    "epoch_end": int(warmup_epochs),
                    "part_model_trainable": False,
                    "part_trainability_scope": "frozen",
                    "classifier_head_trainable": True,
                    "representation_projector_trainable": bool(spec.requires_teacher_representations),
                    "trainable_adapter_module_names": adapter_label,
                },
                {
                    "phase_name": second_phase_name,
                    "epoch_start": int(warmup_epochs) + 1,
                    "epoch_end": "end",
                    "part_model_trainable": True,
                    "part_trainability_scope": second_scope,
                    "classifier_head_trainable": True,
                    "representation_projector_trainable": bool(spec.requires_teacher_representations),
                    "trainable_adapter_module_names": second_adapter_label,
                },
            ],
            spec,
        )
    if str(spec.training_schedule) == PDV3_TRAINING_SCHEDULE_STAGED:
        warmup_epochs = max(2, int(spec.freeze_part_epochs))
        input_end = max(1, warmup_epochs // 2)
        return _with_phase_weight_metadata([
            {
                "phase_name": "input_repair_warmup",
                "epoch_start": 1,
                "epoch_end": int(input_end),
                "part_model_trainable": False,
                "part_trainability_scope": "frozen",
                "classifier_head_trainable": False,
                "representation_projector_trainable": bool(spec.requires_teacher_representations),
                "trainable_adapter_module_names": ["feature_delta_adapter"],
            },
            {
                "phase_name": "embedding_residual_warmup",
                "epoch_start": int(input_end + 1),
                "epoch_end": int(warmup_epochs),
                "part_model_trainable": False,
                "part_trainability_scope": "frozen",
                "classifier_head_trainable": False,
                "representation_projector_trainable": bool(spec.requires_teacher_representations),
                "trainable_adapter_module_names": ["context_control", "context_control_gate"],
            },
            {
                "phase_name": "joint_finetune",
                "epoch_start": int(warmup_epochs + 1),
                "epoch_end": "end",
                "part_model_trainable": True,
                "part_trainability_scope": "full",
                "classifier_head_trainable": True,
                "representation_projector_trainable": bool(spec.requires_teacher_representations),
                "trainable_adapter_module_names": "all_active",
            },
        ], spec)
    if int(spec.freeze_part_epochs) > 0:
        return _with_phase_weight_metadata([
            {
                "phase_name": "adapter_warmup",
                "epoch_start": 1,
                "epoch_end": int(spec.freeze_part_epochs),
                "part_model_trainable": False,
                "part_trainability_scope": "frozen",
                "classifier_head_trainable": True,
                "representation_projector_trainable": bool(spec.requires_teacher_representations),
                "trainable_adapter_module_names": "all_active",
            },
            {
                "phase_name": "joint_finetune",
                "epoch_start": int(spec.freeze_part_epochs) + 1,
                "epoch_end": "end",
                "part_model_trainable": True,
                "part_trainability_scope": "full",
                "classifier_head_trainable": True,
                "representation_projector_trainable": bool(spec.requires_teacher_representations),
                "trainable_adapter_module_names": "all_active",
            },
        ], spec)
    phase_name = "delta_z_joint_from_start_repkd" if str(spec.freeze_policy) == "delta_z_joint_from_start" else "joint_finetune"
    return _with_phase_weight_metadata([
        {
            "phase_name": phase_name,
            "epoch_start": 1,
            "epoch_end": "end",
            "part_model_trainable": True,
            "part_trainability_scope": "full",
            "classifier_head_trainable": True,
            "representation_projector_trainable": bool(spec.requires_teacher_representations),
            "trainable_adapter_module_names": "all_active",
        }
    ], spec)


def apply_pdv3_student_training_phase(
    model: ArchitectureViewResidualParT,
    representation_projector: Any | None,
    spec: PDV3StudentVariantSpec,
    *,
    epoch: int,
    evaluation_only: bool = False,
) -> dict[str, Any]:
    """Apply the CE/KD student trainability schedule for one epoch."""

    active_adapters = tuple(str(name) for name in model.active_adapter_module_names())
    part_trainability_scope = "full"
    if bool(evaluation_only):
        part_trainable = False
        trainable_adapters: tuple[str, ...] = ()
        part_trainability_scope = "frozen"
        phase_name = "evaluation_only"
    elif str(spec.training_schedule) == PDV3_TRAINING_SCHEDULE_REP_FROZEN:
        phase_name = "delta_z_frozen_repkd"
        part_trainable = False
        part_trainability_scope = "frozen"
        trainable_adapters = active_adapters
    elif str(spec.training_schedule) in (
        PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
        PDV3_TRAINING_SCHEDULE_REP_FULL_UNFREEZE,
    ):
        warmup_epochs = max(1, int(spec.freeze_part_epochs))
        current_epoch = max(1, int(epoch))
        freeze_adapter_after_warmup = "freeze_adapter_after_warmup" in str(spec.freeze_policy)
        if current_epoch <= warmup_epochs:
            phase_name = "delta_z_frozen_warmup"
            part_trainable = False
            part_trainability_scope = "frozen"
            trainable_adapters = active_adapters
        elif str(spec.training_schedule) == PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE:
            phase_name = "delta_z_upper_unfreeze_repkd"
            part_trainable = True
            part_trainability_scope = "upper"
            trainable_adapters = () if freeze_adapter_after_warmup else active_adapters
        else:
            phase_name = "delta_z_full_gentle_unfreeze_repkd"
            part_trainable = True
            part_trainability_scope = "full"
            trainable_adapters = () if freeze_adapter_after_warmup else active_adapters
    elif str(spec.training_schedule) == PDV3_TRAINING_SCHEDULE_STAGED:
        warmup_epochs = max(2, int(spec.freeze_part_epochs))
        input_end = max(1, warmup_epochs // 2)
        current_epoch = max(1, int(epoch))
        if current_epoch <= input_end:
            phase_name = "input_repair_warmup"
            part_trainable = False
            part_trainability_scope = "frozen"
            trainable_adapters = tuple(name for name in active_adapters if name == "feature_delta_adapter")
        elif current_epoch <= warmup_epochs:
            phase_name = "embedding_residual_warmup"
            part_trainable = False
            part_trainability_scope = "frozen"
            trainable_adapters = tuple(
                name for name in active_adapters if name in {"context_control", "context_control_gate"}
            )
        else:
            phase_name = "joint_finetune"
            part_trainable = True
            part_trainability_scope = "full"
            trainable_adapters = active_adapters
    else:
        current_epoch = max(1, int(epoch))
        part_trainable = current_epoch > int(spec.freeze_part_epochs)
        part_trainability_scope = "full" if part_trainable else "frozen"
        if not part_trainable and int(spec.freeze_part_epochs) > 0:
            phase_name = "adapter_warmup"
        else:
            phase_name = (
                "delta_z_joint_from_start_repkd"
                if str(spec.freeze_policy) == "delta_z_joint_from_start"
                else "joint_finetune"
            )
        trainable_adapters = active_adapters

    train_head_during_frozen_adapter_warmup = bool(
        not evaluation_only
        and not part_trainable
        and str(spec.training_schedule) != "staged"
        and int(spec.freeze_part_epochs) > 0
        and bool(active_adapters or representation_projector is not None or spec.requires_teacher_logits)
        and not bool(spec.is_baseline)
    )
    part_scope_info = _set_part_model_trainable_scope(model, part_trainability_scope)
    _set_part_head_trainable(model, bool(part_trainable or train_head_during_frozen_adapter_warmup))
    enabled_adapters = _set_active_adapter_modules_trainable(model, trainable_adapters)
    representation_projector_trainable = bool(not evaluation_only and representation_projector is not None)
    _set_module_trainable(representation_projector, representation_projector_trainable)
    frozen_adapters = tuple(name for name in active_adapters if name not in set(enabled_adapters))
    part_head_trainable = False
    if hasattr(model, "_part_head_parameter_modules"):
        for module in model._part_head_parameter_modules():
            if any(parameter.requires_grad for parameter in module.parameters()):
                part_head_trainable = True
                break
    module_group_trainability = {
        "part": bool(part_trainable),
        "classifier_head": bool(part_head_trainable),
        "input_delta_adapter": "feature_delta_adapter" in set(enabled_adapters),
        "embedding_delta_adapter": bool(
            {"context_control", "context_control_gate"}.intersection(set(enabled_adapters))
        ),
        "representation_projector": bool(representation_projector_trainable),
    }
    return {
        "training_schedule": str(spec.training_schedule),
        "freeze_policy": str(spec.freeze_policy),
        "phase_name": str(phase_name),
        "part_model_trainable": bool(part_trainable),
        "part_trainability_scope": str(part_scope_info.get("part_trainability_scope", part_trainability_scope)),
        "upper_unfreeze_block_count": int(part_scope_info.get("upper_unfreeze_block_count", 2)),
        "trainable_part_module_names": list(part_scope_info.get("trainable_part_module_names", [])),
        "trainable_part_module_count": int(part_scope_info.get("trainable_part_module_count", 0)),
        "trainable_part_parameter_count": int(part_scope_info.get("trainable_part_parameter_count", 0)),
        "classifier_head_trainable": bool(part_head_trainable),
        "representation_projector_trainable": bool(representation_projector_trainable),
        **_phase_weight_metadata(spec),
        "module_group_trainability": module_group_trainability,
        "active_adapter_module_names": list(active_adapters),
        "trainable_adapter_module_names": list(enabled_adapters),
        "frozen_adapter_module_names": list(frozen_adapters),
        "optimizer_signature": (
            bool(part_trainable),
            tuple(enabled_adapters),
            bool(representation_projector_trainable),
        ),
    }


def _pdv3_phase_optimizer_group_lrs(
    phase_info: Mapping[str, Any],
    config: PDV3StudentTrainConfig,
) -> dict[str, float]:
    """Record the optimizer groups implied by a phase without peeking into Adam internals."""

    output: dict[str, float] = {}
    if phase_info.get("trainable_adapter_module_names") or bool(phase_info.get("representation_projector_trainable")):
        output["adapter_and_rep_head"] = float(config.adapter_lr_value)
    if bool(phase_info.get("classifier_head_trainable")):
        output["classifier_head"] = float(config.part_lr_value)
    if bool(phase_info.get("part_model_trainable")):
        output["part"] = float(config.part_lr_value)
    return output


def _representation_projector_grad_norm(projector: Any | None) -> float:
    if projector is None:
        return 0.0
    total = None
    for parameter in projector.parameters():
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        term = grad.detach().float().square().sum()
        total = term if total is None else total + term
    if total is None:
        return 0.0
    return float(total.clamp_min(0.0).sqrt().detach().cpu().item())


def _pdv3_combined_adapter_class_diagnostics(output: Any, labels: Any) -> dict[str, tuple[float, float]]:
    """Per-class absolute delta diagnostics for the combined input/embedding adapter."""

    feature_delta_output = getattr(output, "feature_delta_output", None)
    view_output = getattr(output, "view_output", None)
    if feature_delta_output is None or view_output is None:
        return {}
    delta_f = getattr(feature_delta_output, "delta_F_rows", None)
    delta_h = getattr(view_output, "delta_h", None)
    mask_f = getattr(feature_delta_output, "mask", None)
    mask_h = getattr(view_output, "mask", None)
    if delta_f is None or delta_h is None or mask_f is None or mask_h is None:
        return {}
    delta_f = delta_f.detach().float()
    delta_h = delta_h.detach().float()
    mask_f = mask_f.detach().bool()
    mask_h = mask_h.detach().bool()
    particles = min(int(delta_f.shape[1]), int(delta_h.shape[1]), int(mask_f.shape[1]), int(mask_h.shape[1]))
    if particles <= 0:
        return {}
    valid = mask_f[:, :particles] & mask_h[:, :particles]
    valid_counts = valid.sum(dim=1).clamp_min(1).to(dtype=delta_f.dtype)
    jet_delta_f = (delta_f[:, :particles].abs().mean(dim=-1) * valid.to(dtype=delta_f.dtype)).sum(dim=1)
    jet_delta_f = jet_delta_f / valid_counts
    jet_delta_h = (delta_h[:, :particles].abs().mean(dim=-1) * valid.to(dtype=delta_h.dtype)).sum(dim=1)
    jet_delta_h = jet_delta_h / valid_counts.to(dtype=delta_h.dtype)
    output_diag: dict[str, tuple[float, float]] = {}
    labels = labels.detach().long()
    for class_index, class_name in enumerate(PDV3_LABEL_NAMES):
        selected = labels == int(class_index)
        count = int(selected.sum().detach().cpu().item())
        if count <= 0:
            continue
        suffix = f"class_{class_index}_{str(class_name)}"
        output_diag[f"combined_delta_F_abs_mean.{suffix}"] = (
            float(jet_delta_f[selected].mean().detach().cpu().item()),
            float(count),
        )
        output_diag[f"combined_delta_h_abs_mean.{suffix}"] = (
            float(jet_delta_h[selected].mean().detach().cpu().item()),
            float(count),
        )
    return output_diag


def run_pdv3_student_epoch(
    model: ArchitectureViewResidualParT,
    loader,
    *,
    config: PDV3StudentTrainConfig,
    device,
    epoch: int,
    representation_projector=None,
    optimizer=None,
    scaler=None,
    max_batches: int | None = None,
    collect_predictions: bool = False,
    collect_diagnostics: bool = False,
) -> dict[str, Any]:
    torch = require_torch()
    spec = config.spec
    training = optimizer is not None
    model.train(training)
    if representation_projector is not None:
        representation_projector.train(training)
    if bool(getattr(model, "_force_part_model_eval", False)):
        model.part_model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total_rep_loss = 0.0
    total_teacher_logit_kl = 0.0
    total_teacher_representation_cosine = 0.0
    teacher_logit_weight_sum = 0.0
    teacher_representation_weight_sum = 0.0
    total_teacher_top1_agreement = 0.0
    total_teacher_entropy = 0.0
    total_student_entropy_with_teacher = 0.0
    total_teacher_confidence = 0.0
    total_student_confidence_with_teacher = 0.0
    teacher_agreement_weight_sum = 0.0
    teacher_agree_correct = 0.0
    teacher_agree_count = 0.0
    teacher_disagree_correct = 0.0
    teacher_disagree_count = 0.0
    teacher_high_conf_correct = 0.0
    teacher_high_conf_count = 0.0
    teacher_low_conf_correct = 0.0
    teacher_low_conf_count = 0.0
    total_delta_l2_loss = 0.0
    total_delta_l2_mean = 0.0
    total_representation_delta_l2_loss = 0.0
    total_representation_delta_l2_mean = 0.0
    total_effective_alpha = 0.0
    total_effective_beta = 0.0
    total_nonfinite_batches = 0
    total_nonfinite_jets = 0
    total_correct = 0
    total_seen = 0
    collected_preds: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
    collected_logits: list[np.ndarray] = []
    diagnostic_totals: dict[str, float] = {}
    diagnostic_weight_totals: dict[str, float] = {}
    gradient_norm_totals: dict[str, float] = {}
    gradient_norm_weight_sum = 0.0
    configured_alpha = (
        pdv3_effective_weight(spec.kd_alpha, spec.kd_warmup_epochs, epoch)
        if bool(config.uses_teacher_logits)
        else 0.0
    )
    configured_beta = (
        pdv3_effective_weight(spec.rep_beta, spec.rep_warmup_epochs, epoch)
        if bool(config.uses_teacher_representations)
        else 0.0
    )
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch_to_device(batch, device)
            labels = batch["labels"]
            batch_size = int(labels.numel())
            if training:
                optimizer.zero_grad(set_to_none=True)
            teacher_logits = _teacher_tensor_for_batch(loader, batch, "teacher_logits", device)
            teacher_representations = _teacher_tensor_for_batch(loader, batch, "teacher_representations", device)
            effective_alpha = float(configured_alpha) if teacher_logits is not None else 0.0
            effective_beta = float(configured_beta) if teacher_representations is not None else 0.0
            with torch.cuda.amp.autocast(enabled=bool(config.amp and device.type == "cuda")):
                output = model(
                    batch["tokens"],
                    batch["mask"],
                    return_outputs=True,
                    max_constits=int(batch["tokens"].shape[1]),
                    sample_indices=batch.get("indices"),
                )
                logits = output.logits
                student_repr_projected = None
                representation_delta_l2_mean = logits.new_zeros(())
                if teacher_representations is not None:
                    if representation_projector is None:
                        raise ValueError("teacher representations require a representation projector")
                    repr_source = pdv3_student_representation_source_from_output(output)
                    student_repr_projected = representation_projector(repr_source)
                    representation_delta_l2_mean = representation_projector.delta_z_l2_mean()
                kd_loss, loss_parts = pd10_student_loss(
                    logits,
                    labels,
                    teacher_logits=teacher_logits,
                    teacher_representations=teacher_representations,
                    student_repr_projected=student_repr_projected,
                    target_mode=PD10_TARGET_FULL_LOGITS,
                    temperature=float(spec.kd_temperature),
                    kd_alpha=float(effective_alpha),
                    representation_beta=float(effective_beta),
                    representation_mode=PD10_REPRESENTATION_MODE_COSINE
                    if teacher_representations is not None
                    else "none",
                    top_k=3,
                )
                delta_l2_mean = _delta_l2_mean_from_output(output)
                applies_delta_l2 = getattr(output, "feature_delta_output", None) is not None
                delta_l2_weight = float(config.delta_l2_weight) if applies_delta_l2 else 0.0
                delta_l2_loss = delta_l2_mean * delta_l2_weight
                representation_delta_l2_weight = (
                    float(config.representation_delta_l2_weight)
                    if teacher_representations is not None and representation_projector is not None
                    else 0.0
                )
                representation_delta_l2_loss = representation_delta_l2_mean * representation_delta_l2_weight
                loss = kd_loss + delta_l2_loss + representation_delta_l2_loss
            finite_terms = [
                torch.isfinite(loss.detach()).all(),
                torch.isfinite(logits.detach()).all(),
                torch.isfinite(delta_l2_loss.detach()).all(),
                torch.isfinite(representation_delta_l2_loss.detach()).all(),
            ]
            if teacher_logits is not None:
                finite_terms.append(torch.isfinite(teacher_logits.detach()).all())
            if teacher_representations is not None:
                finite_terms.append(torch.isfinite(teacher_representations.detach()).all())
            if not all(bool(term.cpu().item()) for term in finite_terms):
                total_nonfinite_batches += 1
                total_nonfinite_jets += batch_size
                if training:
                    optimizer.zero_grad(set_to_none=True)
                continue
            if training:
                if scaler is not None and bool(scaler.is_enabled()):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norms = _gradient_norm_diagnostics(model)
                    grad_norms["grad_norm.representation_projector"] = _representation_projector_grad_norm(
                        representation_projector
                    )
                    for key, value in grad_norms.items():
                        gradient_norm_totals[key] = gradient_norm_totals.get(key, 0.0) + float(value) * batch_size
                    gradient_norm_weight_sum += float(batch_size)
                    if float(config.grad_clip_norm) > 0.0:
                        params = list(model.parameters())
                        if representation_projector is not None:
                            params.extend(list(representation_projector.parameters()))
                        torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norms = _gradient_norm_diagnostics(model)
                    grad_norms["grad_norm.representation_projector"] = _representation_projector_grad_norm(
                        representation_projector
                    )
                    for key, value in grad_norms.items():
                        gradient_norm_totals[key] = gradient_norm_totals.get(key, 0.0) + float(value) * batch_size
                    gradient_norm_weight_sum += float(batch_size)
                    if float(config.grad_clip_norm) > 0.0:
                        params = list(model.parameters())
                        if representation_projector is not None:
                            params.extend(list(representation_projector.parameters()))
                        torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip_norm))
                    optimizer.step()
            preds = logits.detach().argmax(dim=1)
            total_loss += float(loss.detach().item()) * batch_size
            total_ce_loss += float(loss_parts["ce_loss"]) * batch_size
            total_kd_loss += float(loss_parts["kd_loss"]) * batch_size
            total_rep_loss += float(loss_parts["rep_loss"]) * batch_size
            if teacher_logits is not None:
                total_teacher_logit_kl += float(loss_parts["teacher_student_logit_kl"]) * batch_size
                teacher_logit_weight_sum += float(batch_size)
                teacher_probs = torch.softmax(teacher_logits.detach().float(), dim=-1)
                student_probs = torch.softmax(logits.detach().float(), dim=-1)
                teacher_pred = teacher_probs.argmax(dim=-1)
                teacher_conf = teacher_probs.max(dim=-1).values
                student_conf = student_probs.max(dim=-1).values
                agreement = teacher_pred == preds
                correct = preds == labels
                total_teacher_top1_agreement += float(agreement.float().sum().detach().cpu().item())
                total_teacher_entropy += float(
                    (-(teacher_probs * teacher_probs.clamp_min(1.0e-12).log()).sum(dim=-1)).sum().cpu().item()
                )
                total_student_entropy_with_teacher += float(
                    (-(student_probs * student_probs.clamp_min(1.0e-12).log()).sum(dim=-1)).sum().cpu().item()
                )
                total_teacher_confidence += float(teacher_conf.sum().detach().cpu().item())
                total_student_confidence_with_teacher += float(student_conf.sum().detach().cpu().item())
                teacher_agreement_weight_sum += float(batch_size)
                if bool(agreement.any().item()):
                    teacher_agree_correct += float(correct[agreement].float().sum().detach().cpu().item())
                    teacher_agree_count += float(agreement.float().sum().detach().cpu().item())
                disagreement = ~agreement
                if bool(disagreement.any().item()):
                    teacher_disagree_correct += float(correct[disagreement].float().sum().detach().cpu().item())
                    teacher_disagree_count += float(disagreement.float().sum().detach().cpu().item())
                high_conf = teacher_conf >= 0.70
                low_conf = teacher_conf < 0.50
                if bool(high_conf.any().item()):
                    teacher_high_conf_correct += float(correct[high_conf].float().sum().detach().cpu().item())
                    teacher_high_conf_count += float(high_conf.float().sum().detach().cpu().item())
                if bool(low_conf.any().item()):
                    teacher_low_conf_correct += float(correct[low_conf].float().sum().detach().cpu().item())
                    teacher_low_conf_count += float(low_conf.float().sum().detach().cpu().item())
            if teacher_representations is not None:
                total_teacher_representation_cosine += (
                    float(loss_parts["teacher_student_representation_cosine"]) * batch_size
                )
                teacher_representation_weight_sum += float(batch_size)
            total_delta_l2_loss += float(delta_l2_loss.detach().item()) * batch_size
            total_delta_l2_mean += float(delta_l2_mean.detach().item()) * batch_size
            total_representation_delta_l2_loss += float(representation_delta_l2_loss.detach().item()) * batch_size
            total_representation_delta_l2_mean += float(representation_delta_l2_mean.detach().item()) * batch_size
            total_effective_alpha += float(effective_alpha) * batch_size
            total_effective_beta += float(effective_beta) * batch_size
            total_correct += int((preds == labels).sum().detach().cpu().item())
            total_seen += batch_size
            if collect_predictions:
                collected_preds.append(preds.detach().cpu().numpy().astype(np.int64))
                collected_labels.append(labels.detach().cpu().numpy().astype(np.int64))
                collected_logits.append(logits.detach().cpu().numpy().astype(np.float32))
            if collect_diagnostics:
                flat = _flatten_scalar_diagnostics(output.diagnostics())
                flat.update(
                    {
                        "effective_kd_alpha": float(effective_alpha),
                        "effective_representation_beta": float(effective_beta),
                        "configured_kd_alpha": float(configured_alpha),
                        "configured_representation_beta": float(configured_beta),
                        "student_representation_source_dim": float(_representation_source_dim(model)),
                        "student_representation_source": PDV3_STUDENT_REPRESENTATION_SOURCE,
                        "uses_teacher_logits": float(teacher_logits is not None),
                        "uses_teacher_representations": float(teacher_representations is not None),
                    }
                )
                if student_repr_projected is not None:
                    flat["student_repr_projected_norm_mean"] = float(
                        student_repr_projected.detach().float().norm(dim=-1).mean().cpu().item()
                    )
                if representation_projector is not None and student_repr_projected is not None:
                    for key, value in representation_projector.diagnostics().items():
                        try:
                            flat[key] = float(value)
                        except (TypeError, ValueError):
                            continue
                    flat["representation_delta_l2_loss"] = float(representation_delta_l2_loss.detach().cpu().item())
                    flat["representation_delta_l2_weight"] = float(config.representation_delta_l2_weight)
                for key, value in flat.items():
                    diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value) * batch_size
                    diagnostic_weight_totals[key] = diagnostic_weight_totals.get(key, 0.0) + float(batch_size)
                for key, (value, weight) in _pdv3_combined_adapter_class_diagnostics(output, labels).items():
                    diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value) * float(weight)
                    diagnostic_weight_totals[key] = diagnostic_weight_totals.get(key, 0.0) + float(weight)
    if total_seen == 0:
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "n_jets": 0,
            "nonfinite_batches_skipped": int(total_nonfinite_batches),
            "nonfinite_jets_skipped": int(total_nonfinite_jets),
        }
    metrics: dict[str, Any] = {
        "loss": total_loss / float(total_seen),
        "ce_loss": total_ce_loss / float(total_seen),
        "kd_loss": total_kd_loss / float(total_seen),
        "rep_loss": total_rep_loss / float(total_seen),
        "teacher_student_logit_kl": None
        if teacher_logit_weight_sum <= 0.0
        else total_teacher_logit_kl / teacher_logit_weight_sum,
        "teacher_student_representation_cosine": None
        if teacher_representation_weight_sum <= 0.0
        else total_teacher_representation_cosine / teacher_representation_weight_sum,
        "teacher_student_top1_agreement": None
        if teacher_agreement_weight_sum <= 0.0
        else total_teacher_top1_agreement / teacher_agreement_weight_sum,
        "teacher_entropy_mean": None
        if teacher_agreement_weight_sum <= 0.0
        else total_teacher_entropy / teacher_agreement_weight_sum,
        "student_entropy_mean_with_teacher": None
        if teacher_agreement_weight_sum <= 0.0
        else total_student_entropy_with_teacher / teacher_agreement_weight_sum,
        "teacher_confidence_mean": None
        if teacher_agreement_weight_sum <= 0.0
        else total_teacher_confidence / teacher_agreement_weight_sum,
        "student_confidence_mean_with_teacher": None
        if teacher_agreement_weight_sum <= 0.0
        else total_student_confidence_with_teacher / teacher_agreement_weight_sum,
        "student_accuracy_when_teacher_student_agree": None
        if teacher_agree_count <= 0.0
        else teacher_agree_correct / teacher_agree_count,
        "student_accuracy_when_teacher_student_disagree": None
        if teacher_disagree_count <= 0.0
        else teacher_disagree_correct / teacher_disagree_count,
        "teacher_student_agree_count": int(teacher_agree_count),
        "teacher_student_disagree_count": int(teacher_disagree_count),
        "student_accuracy_teacher_confidence_ge_0p70": None
        if teacher_high_conf_count <= 0.0
        else teacher_high_conf_correct / teacher_high_conf_count,
        "student_accuracy_teacher_confidence_lt_0p50": None
        if teacher_low_conf_count <= 0.0
        else teacher_low_conf_correct / teacher_low_conf_count,
        "teacher_confidence_ge_0p70_count": int(teacher_high_conf_count),
        "teacher_confidence_lt_0p50_count": int(teacher_low_conf_count),
        "delta_l2_loss": total_delta_l2_loss / float(total_seen),
        "delta_l2_mean": total_delta_l2_mean / float(total_seen),
        "delta_l2_weight": float(config.delta_l2_weight),
        "delta_l2_scope": "feature_delta_only",
        "representation_delta_l2_loss": total_representation_delta_l2_loss / float(total_seen),
        "representation_delta_l2_mean": total_representation_delta_l2_mean / float(total_seen),
        "representation_delta_l2_weight": float(config.representation_delta_l2_weight),
        "representation_delta_l2_scope": "delta_z_residual_only",
        "configured_kd_alpha": float(configured_alpha),
        "configured_representation_beta": float(configured_beta),
        "effective_kd_alpha": total_effective_alpha / float(total_seen),
        "effective_representation_beta": total_effective_beta / float(total_seen),
        "nonfinite_batches_skipped": int(total_nonfinite_batches),
        "nonfinite_jets_skipped": int(total_nonfinite_jets),
        "teacher_family": spec.teacher_family,
        "loss_mode": spec.loss_mode,
        "active_loss_components": {
            "ce": True,
            "logit_kd": bool(teacher_logit_weight_sum > 0.0),
            "representation_kd": bool(teacher_representation_weight_sum > 0.0),
        },
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }
    if collect_predictions:
        preds_np = np.concatenate(collected_preds, axis=0) if collected_preds else np.asarray([], dtype=np.int64)
        labels_np = np.concatenate(collected_labels, axis=0) if collected_labels else np.asarray([], dtype=np.int64)
        logits_np = np.concatenate(collected_logits, axis=0) if collected_logits else None
        if logits_np is not None:
            metrics["_prediction_arrays"] = {"preds": preds_np, "labels": labels_np, "logits": logits_np}
        metrics.update(
            classification_metrics_from_predictions(
                preds=preds_np,
                labels=labels_np,
                loss_sum=total_ce_loss,
                logits=logits_np,
                label_names=PDV3_LABEL_NAMES,
            )
        )
        if logits_np is not None:
            metrics["binary_projection_metrics"] = architecture_view_binary_projection_metrics(
                logits_np,
                labels_np,
                label_names=PDV3_LABEL_NAMES,
                pairs=(("QCD", "Hgg"), ("QCD", "Hbb"), ("QCD", "Tbqq")),
            )
    if diagnostic_weight_totals:
        metrics["diagnostics"] = {
            key: value / diagnostic_weight_totals[key]
            for key, value in sorted(diagnostic_totals.items())
            if diagnostic_weight_totals.get(key, 0.0) > 0.0
        }
    if gradient_norm_weight_sum > 0.0:
        diagnostics = dict(metrics.get("diagnostics") or {})
        diagnostics.update(
            {key: value / gradient_norm_weight_sum for key, value in sorted(gradient_norm_totals.items())}
        )
        metrics["diagnostics"] = diagnostics
    return metrics


def _write_epoch_metrics_csv(path: Path, curves: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in curves:
        train = row.get("train", {}) if isinstance(row.get("train"), Mapping) else {}
        val = row.get("model_val", {}) if isinstance(row.get("model_val"), Mapping) else {}
        payload = {
            "epoch": row.get("epoch"),
            "training_schedule": row.get("training_schedule"),
            "train_phase": row.get("train_phase"),
            "freeze_policy": row.get("freeze_policy"),
            "part_model_trainable": row.get("part_model_trainable"),
            "classifier_head_trainable": row.get("classifier_head_trainable"),
            "representation_projector_trainable": row.get("representation_projector_trainable"),
            "optimizer_group_lrs": row.get("optimizer_group_lrs"),
            "module_group_trainability": row.get("module_group_trainability"),
            "trainable_adapter_module_names": " ".join(
                str(name) for name in row.get("trainable_adapter_module_names", [])
            )
            if isinstance(row.get("trainable_adapter_module_names"), list)
            else row.get("trainable_adapter_module_names"),
            "frozen_adapter_module_names": " ".join(
                str(name) for name in row.get("frozen_adapter_module_names", [])
            )
            if isinstance(row.get("frozen_adapter_module_names"), list)
            else row.get("frozen_adapter_module_names"),
            "train_loss": train.get("loss"),
            "train_ce_loss": train.get("ce_loss"),
            "train_kd_loss": train.get("kd_loss"),
            "train_rep_loss": train.get("rep_loss"),
            "train_teacher_student_logit_kl": train.get("teacher_student_logit_kl"),
            "train_teacher_student_representation_cosine": train.get("teacher_student_representation_cosine"),
            "train_teacher_student_top1_agreement": train.get("teacher_student_top1_agreement"),
            "train_teacher_entropy_mean": train.get("teacher_entropy_mean"),
            "train_student_entropy_mean_with_teacher": train.get("student_entropy_mean_with_teacher"),
            "train_nonfinite_batches_skipped": train.get("nonfinite_batches_skipped"),
            "train_delta_l2_loss": train.get("delta_l2_loss"),
            "train_representation_delta_l2_loss": train.get("representation_delta_l2_loss"),
            "train_representation_delta_l2_mean": train.get("representation_delta_l2_mean"),
            "train_accuracy": train.get("accuracy"),
            "model_val_loss": val.get("loss"),
            "model_val_ce_loss": val.get("ce_loss"),
            "model_val_kd_loss": val.get("kd_loss"),
            "model_val_rep_loss": val.get("rep_loss"),
            "model_val_teacher_student_logit_kl": val.get("teacher_student_logit_kl"),
            "model_val_teacher_student_representation_cosine": val.get("teacher_student_representation_cosine"),
            "model_val_teacher_student_top1_agreement": val.get("teacher_student_top1_agreement"),
            "model_val_teacher_entropy_mean": val.get("teacher_entropy_mean"),
            "model_val_student_entropy_mean_with_teacher": val.get("student_entropy_mean_with_teacher"),
            "model_val_nonfinite_batches_skipped": val.get("nonfinite_batches_skipped"),
            "model_val_delta_l2_loss": val.get("delta_l2_loss"),
            "model_val_representation_delta_l2_loss": val.get("representation_delta_l2_loss"),
            "model_val_representation_delta_l2_mean": val.get("representation_delta_l2_mean"),
            "model_val_accuracy": val.get("accuracy"),
            "selection_metric_value": row.get("selection_metric_value"),
        }
        for prefix, metrics in (("train", train), ("model_val", val)):
            diagnostics = metrics.get("diagnostics") if isinstance(metrics.get("diagnostics"), Mapping) else {}
            for key, value in sorted(diagnostics.items()):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric):
                    payload[f"{prefix}_diag_{str(key).replace('.', '_').replace('/', '_')}"] = numeric
        rows.append(payload)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["epoch"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_payload(
    *,
    model: ArchitectureViewResidualParT,
    representation_projector,
    optimizer,
    epoch: int,
    config: PDV3StudentTrainConfig,
    architecture_config: ArchitectureViewTaggerTrainConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
    baseline_load_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "representation_projector_state_dict": None
        if representation_projector is None
        else representation_projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "student_variant": str(config.student_variant),
        "student_spec": config.spec.to_dict(),
        "architecture_view_variant": str(architecture_config.variant),
        "variant_behavior": model.variant_behavior(),
        "config": asdict(config),
        "architecture_config": asdict(architecture_config),
        "model_config": model.to_config_dict(),
        "representation_projector_config": None
        if representation_projector is None
        else representation_projector.to_config_dict(),
        "metrics": dict(metrics),
        "label_names": list(PDV3_LABEL_NAMES),
        "label_filter": list(PDV3_LABEL_FILTER),
        "num_classes": PDV3_NUM_CLASSES,
        "source": dict(source),
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": dict(baseline_load_report),
        "experiment_step": PDV3_STEP4_TRAIN_STEP,
        "train_contract": PDV3_STUDENT_TRAIN_CONTRACT,
        "output_contract": model.output_contract,
    }


def _save_metrics(path: Path, metrics: Mapping[str, Any]) -> None:
    save_json(path, _metrics_without_prediction_arrays(metrics))


def train_pdv3_student(config: PDV3StudentTrainConfig) -> dict[str, Any]:
    """Train or recheck one PDV3 AV10-adapter student."""

    run_start_time = time.perf_counter()
    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    if output_dir.exists() and not bool(config.overwrite):
        # Keep this deliberately narrow: existing empty dirs are common on Slurm
        # after fresh_claim_new_dir, but existing reports mean accidental reuse.
        if (output_dir / "run_report.json").exists() or (output_dir / "best_model_val.pt").exists():
            raise FileExistsError(f"PDV3 student output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    architecture_config = config.architecture_config()
    spec = config.spec
    manifest_info = _manifest_metadata(config.manifest_path)
    manifest_sha = manifest_info.get("manifest_hash") if manifest_info.get("load_ok") else None
    if bool(config.require_baseline_split_manifest_hash):
        if not bool(manifest_info.get("load_ok")):
            raise ValueError(
                "require_baseline_split_manifest_hash=True requires a readable split manifest; "
                f"failed to load {config.manifest_path!r}: {manifest_info.get('load_error', 'unknown error')}"
            )
        if not manifest_sha:
            raise ValueError("require_baseline_split_manifest_hash=True requires a non-empty manifest_hash")

    train_dataset = _load_pdv3_student_dataset(
        config,
        config.train_split,
        max_jets=config.max_train_jets,
        load_teacher_supervision=True,
    )
    val_dataset = _load_pdv3_student_dataset(
        config,
        config.val_split,
        max_jets=config.max_val_jets,
        load_teacher_supervision=True,
    )
    _verify_pdv3_dataset_manifest_hash(
        train_dataset,
        expected_manifest_hash=str(manifest_sha) if manifest_sha else None,
        split=str(config.train_split),
        cache_dir=str(config.hlt_cache_dir),
    )
    _verify_pdv3_dataset_manifest_hash(
        val_dataset,
        expected_manifest_hash=str(manifest_sha) if manifest_sha else None,
        split=str(config.val_split),
        cache_dir=str(config.hlt_cache_dir),
    )
    if train_dataset.tokens.shape[-1] != RAW_TOKEN_DIM or val_dataset.tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"PDV3 students expect raw token dim {RAW_TOKEN_DIM}")
    train_loader = make_subtoken_hlt_loader(
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

    model = build_architecture_view_residual_part(
        architecture_config.model_config(),
        variant=str(spec.architecture_view_variant),
    ).to(device)
    baseline_checkpoint_path = Path(config.baseline_checkpoint) if config.baseline_checkpoint else None
    baseline_warm_start_available = bool(baseline_checkpoint_path is not None and baseline_checkpoint_path.is_file())
    baseline_from_scratch = bool(
        not baseline_warm_start_available
        and bool(config.allow_baseline_from_scratch)
        and bool(spec.is_baseline)
        and spec.loss_mode == PDV3_LOSS_CE
    )
    if baseline_warm_start_available:
        baseline_report = warm_start_architecture_view_part_model(
            model,
            config.baseline_checkpoint,
            map_location=device,
            expected_selection_metric=str(config.selection_metric),
            expected_split_manifest_hash=manifest_sha if bool(config.require_baseline_split_manifest_hash) else None,
            expected_label_names=PDV3_LABEL_NAMES,
            expected_label_filter=PDV3_LABEL_FILTER,
            expected_num_classes=PDV3_NUM_CLASSES,
            expected_hlt_degradation_strength=float(config.expected_hlt_degradation_strength),
            require_metadata=True,
            require_all_part_keys=True,
        ).to_dict()
        baseline_report["weight_warm_start_skipped"] = False
        baseline_initialization = "warm_start_checkpoint"
    elif baseline_from_scratch:
        baseline_report = {
            "checkpoint_step": "pdv3_step4_baseline_from_scratch",
            "checkpoint_contract": PDV3_STUDENT_TRAIN_CONTRACT,
            "baseline_checkpoint_path": str(config.baseline_checkpoint),
            "baseline_checkpoint_exists": False,
            "baseline_checkpoint_hash": None,
            "baseline_checkpoint_selection_metric": None,
            "baseline_checkpoint_hlt_degradation_strength": None,
            "baseline_checkpoint_split_manifest_hash": None,
            "weight_warm_start_skipped": True,
            "from_scratch_baseline_fallback": True,
            "reason": "missing_or_empty_baseline_checkpoint_for_pdv3_hlt_part_ce",
        }
        baseline_initialization = "from_scratch_fallback"
    else:
        raise FileNotFoundError(f"PDV3 baseline checkpoint is required but missing: {config.baseline_checkpoint}")
    baseline_report["input_source"] = "hlt"
    model.baseline_checkpoint_report = dict(baseline_report)
    save_json(diagnostics_dir / "baseline_load_report.json", baseline_report)

    init_logit_sample_count = min(4, int(len(train_dataset)))
    init_logit_diff: dict[str, Any] = {}
    if init_logit_sample_count > 0:
        sample_tokens = torch.as_tensor(train_dataset.tokens[:init_logit_sample_count], device=device).float()
        sample_mask = torch.as_tensor(train_dataset.mask[:init_logit_sample_count], device=device).bool()
        init_logit_diff = compute_architecture_view_init_logit_diff_vs_baseline(
            model,
            sample_tokens,
            sample_mask,
            max_constits=int(sample_tokens.shape[1]),
            attach=True,
        )
        save_json(diagnostics_dir / "init_logit_diff_vs_baseline.json", init_logit_diff)

    representation_projector = None
    if config.uses_teacher_representations:
        representation_projector = PDV3RepresentationProjector(
            _representation_source_dim(model),
            int(config.representation_dim),
            hidden_dim=max(256, int(config.fusion_hidden_dim) * 2),
            dropout=float(config.dropout),
        ).to(device)

    evaluation_only = bool(spec.is_baseline and spec.loss_mode == PDV3_LOSS_CE and baseline_warm_start_available)
    phase_plan = pdv3_student_training_phase_plan(spec)
    initial_phase = apply_pdv3_student_training_phase(
        model,
        representation_projector,
        spec,
        epoch=0 if evaluation_only else 1,
        evaluation_only=evaluation_only,
    )
    initial_phase["optimizer_group_lrs"] = _pdv3_phase_optimizer_group_lrs(initial_phase, config)
    optimizer_signature = initial_phase["optimizer_signature"]
    optimizer = None if evaluation_only else _optimizer_for_pdv3(model, representation_projector, config)
    train_model = model
    if bool(config.compile_model) and not evaluation_only and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": PDV3_STEP4_TRAIN_STEP,
        "train_contract": PDV3_STUDENT_TRAIN_CONTRACT,
        "output_contract": model.output_contract,
        "student_variant": str(config.student_variant),
        "student_spec": spec.to_dict(),
        "architecture_view_variant": str(spec.architecture_view_variant),
        "training_schedule": str(spec.training_schedule),
        "freeze_policy": str(spec.freeze_policy),
        "phase_plan": phase_plan,
        "optimizer_lr_policy": {
            "adapter_and_rep_head": float(config.adapter_lr_value),
            "part": float(config.part_lr_value),
            "weight_decay": float(config.weight_decay_value),
        },
        "initial_train_phase": dict(initial_phase),
        "config": asdict(config),
        "architecture_config": asdict(architecture_config),
        "model_config": model.to_config_dict(),
        "representation_projector_config": None
        if representation_projector is None
        else representation_projector.to_config_dict(),
        "source": source,
        "manifest": manifest_info,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "hlt_input_contract": {
            "profile": str(config.expected_hlt_profile),
            "degradation_strength": float(config.expected_hlt_degradation_strength),
            "label": f"{config.expected_hlt_profile}@{float(config.expected_hlt_degradation_strength):g}",
            "matches_pdv3_default": bool(
                str(config.expected_hlt_profile) == PDV3_HLT_PROFILE
                and abs(float(config.expected_hlt_degradation_strength) - float(PDV3_HLT_DEGRADATION_STRENGTH))
                <= 1.0e-12
            ),
        },
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": baseline_report,
        "baseline_initialization": str(baseline_initialization),
        "baseline_warm_start_available": bool(baseline_warm_start_available),
        "baseline_from_scratch": bool(baseline_from_scratch),
        "label_names": list(PDV3_LABEL_NAMES),
        "label_filter": list(PDV3_LABEL_FILTER),
        "num_classes": PDV3_NUM_CLASSES,
        "selection_metric": str(config.selection_metric),
        "teacher_target": config.teacher_target,
        "active_loss_components": {
            "ce": True,
            "logit_kd": bool(config.uses_teacher_logits),
            "representation_kd": bool(config.uses_teacher_representations),
        },
        "teacher_logit_root": config.teacher_logit_cache_root if config.uses_teacher_logits else "",
        "teacher_representation_root": config.teacher_representation_cache_root
        if config.uses_teacher_representations
        else "",
        "leakage_rule": (
            "PDV3 Step 4 student forward consumes fixed-HLT tokens only. Teacher logits/representations are "
            "train-time/model_val supervision only; model_train trains, model_val selects, and final_test "
            "evaluation disables teacher caches after checkpoint selection."
        ),
        "final_test_loaded_during_training": False,
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    if evaluation_only:
        val_metrics = run_pdv3_student_epoch(
            model,
            val_loader,
            config=config,
            device=device,
            epoch=0,
            representation_projector=representation_projector,
            max_batches=config.max_val_batches,
            collect_predictions=True,
            collect_diagnostics=True,
        )
        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        curves = [
            {
                "epoch": 0,
                "train": {"loss": None, "accuracy": None, "n_jets": 0, "mode": "evaluation_only_baseline_recheck"},
                "model_val": _metrics_without_prediction_arrays(val_metrics),
                "selection_metric": str(config.selection_metric),
                "selection_metric_value": float(selection_value),
                "selection_score": float(val_score),
                "training_schedule": str(spec.training_schedule),
                "train_phase": "evaluation_only",
                "part_model_trainable": False,
                "classifier_head_trainable": False,
                "representation_projector_trainable": False,
                "optimizer_group_lrs": {},
                "module_group_trainability": dict(initial_phase.get("module_group_trainability", {})),
                "trainable_adapter_module_names": [],
                "frozen_adapter_module_names": list(model.active_adapter_module_names()),
            }
        ]
        best_epoch = 0
        best_val_metrics = val_metrics
        best_val_score = float(val_score)
        best_val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
        best_val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        payload = _checkpoint_payload(
            model=model,
            representation_projector=representation_projector,
            optimizer=optimizer,
            epoch=0,
            config=config,
            architecture_config=architecture_config,
            metrics=curves[0],
            source=source,
            baseline_load_report=baseline_report,
        )
        torch.save(payload, output_dir / "best_model_val.pt")
        torch.save(payload, output_dir / "last.pt")
    else:
        curves = []
        best_val_score = float("-inf")
        selection_value = float("nan")
        best_val_accuracy = -1.0
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        for epoch in range(1, int(config.epochs) + 1):
            phase_info = apply_pdv3_student_training_phase(
                model,
                representation_projector,
                spec,
                epoch=epoch,
                evaluation_only=False,
            )
            phase_info["optimizer_group_lrs"] = _pdv3_phase_optimizer_group_lrs(phase_info, config)
            if phase_info["optimizer_signature"] != optimizer_signature:
                optimizer = _optimizer_for_pdv3(model, representation_projector, config)
                optimizer_signature = phase_info["optimizer_signature"]
            train_metrics = run_pdv3_student_epoch(
                train_model,
                train_loader,
                config=config,
                device=device,
                epoch=epoch,
                representation_projector=representation_projector,
                optimizer=optimizer,
                scaler=scaler,
                max_batches=config.max_train_batches,
                collect_predictions=False,
                collect_diagnostics=True,
            )
            val_metrics = run_pdv3_student_epoch(
                model,
                val_loader,
                config=config,
                device=device,
                epoch=epoch,
                representation_projector=representation_projector,
                max_batches=config.max_val_batches,
                collect_predictions=True,
                collect_diagnostics=True,
            )
            val_score, current_selection_value = _selection_score(val_metrics, str(config.selection_metric))
            row = {
                "epoch": int(epoch),
                "train": _metrics_without_prediction_arrays(train_metrics),
                "model_val": _metrics_without_prediction_arrays(val_metrics),
                "selection_metric": str(config.selection_metric),
                "selection_metric_value": float(current_selection_value),
                "selection_score": float(val_score),
                "training_schedule": str(spec.training_schedule),
                "train_phase": str(phase_info["phase_name"]),
                "freeze_policy": str(spec.freeze_policy),
                "part_model_trainable": bool(phase_info["part_model_trainable"]),
                "classifier_head_trainable": bool(phase_info["classifier_head_trainable"]),
                "representation_projector_trainable": bool(phase_info["representation_projector_trainable"]),
                "optimizer_group_lrs": dict(phase_info["optimizer_group_lrs"]),
                "module_group_trainability": dict(phase_info["module_group_trainability"]),
                "trainable_adapter_module_names": list(phase_info["trainable_adapter_module_names"]),
                "frozen_adapter_module_names": list(phase_info["frozen_adapter_module_names"]),
                "effective_kd_alpha": pdv3_effective_weight(spec.kd_alpha, spec.kd_warmup_epochs, epoch)
                if bool(config.uses_teacher_logits)
                else 0.0,
                "effective_representation_beta": pdv3_effective_weight(
                    spec.rep_beta,
                    spec.rep_warmup_epochs,
                    epoch,
                )
                if bool(config.uses_teacher_representations)
                else 0.0,
            }
            curves.append(row)
            improved = val_score > best_val_score
            if improved:
                best_val_score = float(val_score)
                selection_value = float(current_selection_value)
                best_val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
                best_val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
                best_epoch = int(epoch)
                epochs_without_improvement = 0
                payload = _checkpoint_payload(
                    model=model,
                    representation_projector=representation_projector,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    architecture_config=architecture_config,
                    metrics=row,
                    source=source,
                    baseline_load_report=baseline_report,
                )
                torch.save(payload, output_dir / "best_model_val.pt")
                _save_metrics(output_dir / "model_val_report.json", val_metrics)
            else:
                epochs_without_improvement += 1
            payload = _checkpoint_payload(
                model=model,
                representation_projector=representation_projector,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                architecture_config=architecture_config,
                metrics=row,
                source=source,
                baseline_load_report=baseline_report,
            )
            torch.save(payload, output_dir / "last.pt")
            save_json(output_dir / "training_curves.json", {"epochs": curves})
            _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)
            if int(config.early_stop_patience) >= 0 and epochs_without_improvement > int(config.early_stop_patience):
                break
        if best_epoch < 0:
            raise RuntimeError("PDV3 training completed without selecting a best checkpoint")
        best_payload = torch.load(output_dir / "best_model_val.pt", map_location=device)
        model.load_state_dict(best_payload["model_state_dict"])
        if representation_projector is not None and best_payload.get("representation_projector_state_dict") is not None:
            representation_projector.load_state_dict(best_payload["representation_projector_state_dict"])
        best_phase = apply_pdv3_student_training_phase(
            model,
            representation_projector,
            spec,
            epoch=max(1, int(best_epoch)),
            evaluation_only=False,
        )
        best_phase["optimizer_group_lrs"] = _pdv3_phase_optimizer_group_lrs(best_phase, config)
        best_val_metrics = run_pdv3_student_epoch(
            model,
            val_loader,
            config=config,
            device=device,
            epoch=best_epoch,
            representation_projector=representation_projector,
            max_batches=config.max_val_batches,
            collect_predictions=True,
            collect_diagnostics=True,
        )
        val_score, selection_value = _selection_score(best_val_metrics, str(config.selection_metric))
        best_val_score = float(val_score)
        best_val_accuracy = _finite_float(best_val_metrics.get("accuracy"), default=-1.0)
        best_val_loss = _finite_float(best_val_metrics.get("loss"), default=float("inf"))
        _save_metrics(output_dir / "model_val_report.json", best_val_metrics)
    if evaluation_only:
        best_phase = initial_phase

    save_json(output_dir / "training_curves.json", {"epochs": curves})
    _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)

    final_test_dataset = _load_pdv3_student_dataset(
        config,
        config.final_test_split,
        max_jets=config.max_final_test_jets,
        load_teacher_supervision=False,
    )
    _verify_pdv3_dataset_manifest_hash(
        final_test_dataset,
        expected_manifest_hash=str(manifest_sha) if manifest_sha else None,
        split=str(config.final_test_split),
        cache_dir=str(config.hlt_cache_dir),
    )
    final_test_loader = make_subtoken_hlt_loader(
        final_test_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 2,
    )
    final_test_metrics = run_pdv3_student_epoch(
        model,
        final_test_loader,
        config=config,
        device=device,
        epoch=int(best_epoch),
        representation_projector=representation_projector,
        max_batches=config.max_final_test_batches,
        collect_predictions=True,
        collect_diagnostics=True,
    )
    final_test_teacher_diagnostic_metrics = None
    if bool(config.final_test_teacher_diagnostics) and (config.uses_teacher_logits or config.uses_teacher_representations):
        final_test_teacher_dataset = _load_pdv3_student_dataset(
            config,
            config.final_test_split,
            max_jets=config.max_final_test_jets,
            load_teacher_supervision=True,
        )
        _verify_pdv3_dataset_manifest_hash(
            final_test_teacher_dataset,
            expected_manifest_hash=str(manifest_sha) if manifest_sha else None,
            split=str(config.final_test_split),
            cache_dir=str(config.hlt_cache_dir),
        )
        final_test_teacher_loader = make_subtoken_hlt_loader(
            final_test_teacher_dataset,
            batch_size=int(config.eval_batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            seed=int(config.seed) + 3,
        )
        final_test_teacher_diagnostic_metrics = run_pdv3_student_epoch(
            model,
            final_test_teacher_loader,
            config=config,
            device=device,
            epoch=int(best_epoch),
            representation_projector=representation_projector,
            max_batches=config.max_final_test_batches,
            collect_predictions=False,
            collect_diagnostics=True,
        )
        final_test_teacher_diagnostic_metrics["diagnostic_only"] = True
        final_test_teacher_diagnostic_metrics["used_for_selection"] = False
        final_test_teacher_diagnostic_metrics["used_for_primary_final_test"] = False
        _save_metrics(output_dir / "final_test_teacher_diagnostics_report.json", final_test_teacher_diagnostic_metrics)
    elapsed_seconds = float(time.perf_counter() - run_start_time)
    checkpoint_path = output_dir / "best_model_val.pt"
    last_checkpoint_path = output_dir / "last.pt"
    checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    last_checkpoint_hash = sha256_file(last_checkpoint_path) if last_checkpoint_path.is_file() else None
    report = {
        "experiment_step": PDV3_STEP4_TRAIN_STEP,
        "train_contract": PDV3_STUDENT_TRAIN_CONTRACT,
        "output_contract": model.output_contract,
        "student_variant": str(config.student_variant),
        "student_spec": spec.to_dict(),
        "architecture_view_variant": str(spec.architecture_view_variant),
        "training_schedule": str(spec.training_schedule),
        "freeze_policy": str(spec.freeze_policy),
        "phase_plan": phase_plan,
        "best_train_phase": dict(best_phase),
        "variant_behavior": model.variant_behavior(),
        "parameter_accounting": model.parameter_accounting(),
        "representation_projector_config": None
        if representation_projector is None
        else representation_projector.to_config_dict(),
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": architecture_view_selection_metric_direction(str(config.selection_metric)),
        "best_model_selection_metric_value": float(selection_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "best_model_val_metrics": _metrics_without_prediction_arrays(best_val_metrics),
        "final_test_metrics": _metrics_without_prediction_arrays(final_test_metrics),
        "final_test_teacher_diagnostic_metrics": None
        if final_test_teacher_diagnostic_metrics is None
        else _metrics_without_prediction_arrays(final_test_teacher_diagnostic_metrics),
        "final_test_evaluated": True,
        "epochs_completed": 0 if evaluation_only else len(curves),
        "checkpoint": str(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "last_checkpoint": str(last_checkpoint_path),
        "last_checkpoint_hash": last_checkpoint_hash,
        "training_curves": str(output_dir / "training_curves.json"),
        "diagnostic_csv": str(diagnostics_dir / "epoch_metrics.csv"),
        "config": asdict(config),
        "architecture_config": asdict(architecture_config),
        "model_config": model.to_config_dict(),
        "label_names": list(PDV3_LABEL_NAMES),
        "label_filter": list(PDV3_LABEL_FILTER),
        "num_classes": PDV3_NUM_CLASSES,
        "source": source,
        "manifest": manifest_info,
        "hlt_input_contract": {
            "profile": str(config.expected_hlt_profile),
            "degradation_strength": float(config.expected_hlt_degradation_strength),
            "label": f"{config.expected_hlt_profile}@{float(config.expected_hlt_degradation_strength):g}",
            "matches_pdv3_default": bool(
                str(config.expected_hlt_profile) == PDV3_HLT_PROFILE
                and abs(float(config.expected_hlt_degradation_strength) - float(PDV3_HLT_DEGRADATION_STRENGTH))
                <= 1.0e-12
            ),
        },
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": baseline_report,
        "baseline_initialization": str(baseline_initialization),
        "baseline_warm_start_available": bool(baseline_warm_start_available),
        "baseline_from_scratch": bool(baseline_from_scratch),
        "baseline_checkpoint_path": baseline_report.get("baseline_checkpoint_path"),
        "baseline_checkpoint_hash": baseline_report.get("baseline_checkpoint_hash"),
        "baseline_checkpoint_selection_metric": baseline_report.get("baseline_checkpoint_selection_metric"),
        "baseline_checkpoint_hlt_degradation_strength": baseline_report.get(
            "baseline_checkpoint_hlt_degradation_strength"
        ),
        "baseline_checkpoint_split_manifest_hash": baseline_report.get("baseline_checkpoint_split_manifest_hash"),
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "final_test_dataset": dict(final_test_dataset.metadata),
        "teacher_target": config.teacher_target,
        "active_loss_components": {
            "ce": True,
            "logit_kd": bool(config.uses_teacher_logits),
            "representation_kd": bool(config.uses_teacher_representations),
        },
        "teacher_logit_root": config.teacher_logit_cache_root if config.uses_teacher_logits else "",
        "teacher_representation_root": config.teacher_representation_cache_root
        if config.uses_teacher_representations
        else "",
        "inference_consumes_hlt_only": True,
        "final_test_loaded_during_training": False,
        "final_test_evaluation_policy": "hlt_only_no_teacher_caches",
        "final_test_teacher_diagnostics_enabled": bool(config.final_test_teacher_diagnostics),
        "final_test_uses_teacher_logits": bool(final_test_dataset.teacher_logits is not None),
        "final_test_uses_teacher_representations": bool(final_test_dataset.teacher_representations is not None),
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "epochs_completed": 0 if evaluation_only else len(curves),
        },
    }
    _save_metrics(output_dir / "model_val_report.json", best_val_metrics)
    _save_metrics(output_dir / "final_test_report.json", final_test_metrics)
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "PDV3_STEP4_TRAIN_STEP",
    "PDV3_RESIDUAL_REPRESENTATION_PROJECTOR_CONTRACT",
    "PDV3_STUDENT_REPRESENTATION_SOURCE",
    "PDV3_STUDENT_TRAIN_CONTRACT",
    "PDV3ArchitectureViewStudentDataset",
    "PDV3RepresentationProjector",
    "PDV3StudentTrainConfig",
    "apply_pdv3_student_training_phase",
    "pdv3_effective_weight",
    "pdv3_student_training_phase_plan",
    "pdv3_student_representation_source_from_output",
    "run_pdv3_student_epoch",
    "train_pdv3_student",
]

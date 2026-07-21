"""Step 7 training runner for local residual-field curriculum distillation.

The runner deliberately keeps oracle consumers and teacher-logit caches on the
training/evaluation side of the boundary.  Selected checkpoints contain only
the deployable reconstructor, confidence heads, and HLT student.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import (
    amp_autocast_context,
    amp_grad_scaler,
    require_torch,
    resolve_device,
    save_json,
    set_training_seed,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES
from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions

from .curriculum import (
    ALPHA_SCHEDULE_FIXED,
    ALPHA_SCHEDULE_PIECEWISE,
    FIELD_GATE_MODE_LEARNED_SIGMOID,
    FIELD_GATE_MODE_NONE,
    FIELD_GATE_MODES,
    FREEZE_SCHEDULE_NONE,
    FREEZE_SCHEDULE_RESIDUAL_WARMUP_THEN_UPPER,
    LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT,
    LocalResidualFieldCurriculumJointConfig,
    LocalResidualFieldCurriculumJointModel,
    LocalResidualFieldCurriculumScheduler,
    LocalResidualFieldCurriculumSchedulerConfig,
    RESIDUAL_PROJECTION_RESET_NONE,
    RESIDUAL_PROJECTION_RESET_SCALE,
    load_selected_consumer_record,
    paired_consumers_confirmed_from_env,
)
from .data import (
    LocalParticleResidualFieldDatasetConfig,
    TeacherLogitBlock,
    _load_teacher_logits,
    load_local_particle_residual_field_dataset,
    load_local_particle_residual_field_hlt_only_dataset,
    make_local_particle_residual_field_loader,
    move_local_particle_residual_field_batch_to_device,
)
from .model import LocalResidualFieldReconstructorConfig
from .oracle import FrozenLocalResidualFieldOracleConsumer, FrozenOracleConsumerConfig, _sha256_file
from .tagger import (
    RESIDUAL_FIELD_SOURCE_ZERO,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldTaggerConfig,
)
from .train import _jsonable, _torch_load_checkpoint


LOCAL_RESIDUAL_FIELD_CURRICULUM_TRAIN_CONTRACT = "local_residual_field_curriculum_train_v1"
CURRICULUM_PILOT_RUN_IDS = ("P0", "P2", "P4", "P7a", "P7b", "Q0", "Q3")
CURRICULUM_STAGE1B_RUN_IDS = ("P2", "P4", "P7a", "P7b", "Q0", "Q3")
CURRICULUM_STUDENT_KD_SOURCES = ("oracle_true", "offline_teacher")
CURRICULUM_LOSS_NAMES = ("ce", "student_kd", "oracle_path", "field", "gate", "reg")


PILOT_LOSS_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    "P0": {"ce": 1.0, "student_kd": 0.0, "oracle_path": 0.0, "field": 1.0, "gate": 0.0, "reg": 0.01},
    "P2": {"ce": 1.0, "student_kd": 0.0, "oracle_path": 1.0, "field": 0.0, "gate": 0.0, "reg": 0.01},
    "P4": {"ce": 1.0, "student_kd": 0.0, "oracle_path": 1.0, "field": 0.1, "gate": 0.0, "reg": 0.01},
    "P7a": {"ce": 1.0, "student_kd": 0.0, "oracle_path": 1.0, "field": 0.2, "gate": 0.05, "reg": 0.01},
    "P7b": {"ce": 1.0, "student_kd": 0.0, "oracle_path": 1.0, "field": 0.2, "gate": 0.05, "reg": 0.01},
    "Q0": {"ce": 1.0, "student_kd": 0.0, "oracle_path": 0.0, "field": 0.2, "gate": 0.05, "reg": 0.01},
    "Q3": {"ce": 1.0, "student_kd": 0.0, "oracle_path": 1.0, "field": 0.2, "gate": 0.05, "reg": 0.01},
}


@dataclass
class LocalResidualFieldCurriculumTrainConfig:
    """Configuration for one Step 7 pilot curriculum run."""

    output_dir: str
    hlt_cache_dir: str
    target_cache_dir: str
    run_id: str
    manifest_path: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    stack_val_split: str = "stack_val"
    final_test_split: str = "final_test"
    seed: int = 30421
    batch_size: int = 24
    eval_batch_size: int = 64
    epochs: int = 12
    gradient_accumulation_steps: int = 1
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    early_stop_patience: int = 4
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    max_stack_val_jets: int | None = None
    max_final_test_jets: int | None = None
    num_classes: int = len(LABEL_NAMES)
    label_names: Sequence[str] = tuple(LABEL_NAMES)
    model_size: str = "base"
    reconstructor_d_model: int = 160
    reconstructor_num_heads: int = 5
    reconstructor_num_layers: int = 4
    reconstructor_context_layers: int = 1
    reconstructor_dropout: float = 0.05
    reconstructor_attention_dropout: float = 0.05
    residual_field_clip_value: float = 8.0
    selected_consumer_json: str | None = None
    consumer_id: str | None = None
    selected_alpha_endpoint: float | None = None
    confirm_paired_consumers: bool = False
    oracle_teacher_checkpoint: str | None = None
    oracle_teacher_config_path: str | None = None
    oracle_run_report_path: str | None = None
    oracle_forward_microbatch_size: int | None = None
    oracle_logit_only_fallback: bool = False
    oracle_teacher_logits_dir: str | None = None
    oracle_teacher_logits_paths: Mapping[str, str] = field(default_factory=dict)
    offline_teacher_logits_dir: str | None = None
    offline_teacher_logits_paths: Mapping[str, str] = field(default_factory=dict)
    student_kd_source: str = "oracle_true"
    student_warm_start_checkpoint: str | None = None
    predictor_warm_start_checkpoint: str | None = None
    field_gate_mode: str | None = None
    initial_gate_bias_prob: float = 0.1
    gate_reliability_error_scale: float = 1.0
    residual_projection_reset: str | None = None
    residual_projection_scale: float = 0.1
    freeze_schedule: str | None = None
    freeze_phase1_epochs: int = 2
    freeze_phase2_epochs: int = 3
    optimizer_group_learning_rates: Mapping[str, float] = field(default_factory=dict)
    alpha_schedule: str | None = None
    fixed_alpha: float | None = None
    piecewise_alpha: Sequence[Any] = field(default_factory=tuple)
    sigmoid_alpha_start: float = 0.25
    sigmoid_alpha_end: float | None = None
    sigmoid_alpha_midpoint: float = 0.5
    sigmoid_alpha_sharpness: float = 12.0
    loss_weight_overrides: Mapping[str, float] = field(default_factory=dict)
    loss_weight_schedule: Mapping[str, Any] = field(default_factory=dict)
    kd_temperature: float = 2.0
    field_huber_beta: float = 0.1
    min_validation_valid_fraction: float = 0.99
    verify_hash: bool = True
    require_manifest_match: bool = True
    save_last_checkpoint: bool = True
    evaluate_final_test: bool = False
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        self.output_dir = str(self.output_dir)
        self.hlt_cache_dir = str(self.hlt_cache_dir)
        self.target_cache_dir = str(self.target_cache_dir)
        self.run_id = str(self.run_id)
        if self.run_id not in CURRICULUM_PILOT_RUN_IDS:
            raise ValueError(f"run_id must be one of {CURRICULUM_PILOT_RUN_IDS}")
        self.manifest_path = None if not self.manifest_path else str(self.manifest_path)
        for name in ("batch_size", "eval_batch_size", "epochs", "gradient_accumulation_steps", "num_classes"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if int(self.early_stop_patience) < -1:
            raise ValueError("early_stop_patience must be -1 or greater")
        if int(self.reconstructor_d_model) % int(self.reconstructor_num_heads) != 0:
            raise ValueError("reconstructor_d_model must be divisible by reconstructor_num_heads")
        for name in ("weight_decay", "grad_clip_norm", "residual_field_clip_value"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, value)
        for name in ("kd_temperature", "field_huber_beta", "gate_reliability_error_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        self.min_validation_valid_fraction = float(self.min_validation_valid_fraction)
        if not (0.0 < self.min_validation_valid_fraction <= 1.0):
            raise ValueError("min_validation_valid_fraction must be in (0, 1]")
        self.initial_gate_bias_prob = float(self.initial_gate_bias_prob)
        if not (0.0 < self.initial_gate_bias_prob <= 1.0):
            raise ValueError("initial_gate_bias_prob must be in (0, 1]")
        self.label_names = tuple(str(name) for name in self.label_names)
        if len(self.label_names) != int(self.num_classes):
            raise ValueError("label_names length must match num_classes")
        self.student_kd_source = str(self.student_kd_source)
        if self.student_kd_source not in CURRICULUM_STUDENT_KD_SOURCES:
            raise ValueError(f"student_kd_source must be one of {CURRICULUM_STUDENT_KD_SOURCES}")
        unknown_weights = sorted(set(self.loss_weight_overrides) - set(CURRICULUM_LOSS_NAMES))
        if unknown_weights:
            raise ValueError(f"unknown curriculum loss weights: {unknown_weights}")
        self.loss_weight_overrides = {
            str(name): _finite_nonnegative(value, name=f"loss_weight_overrides.{name}")
            for name, value in dict(self.loss_weight_overrides).items()
        }
        self.loss_weight_schedule = dict(self.loss_weight_schedule or {})
        self.oracle_teacher_logits_paths = {str(key): str(value) for key, value in dict(self.oracle_teacher_logits_paths).items()}
        self.offline_teacher_logits_paths = {str(key): str(value) for key, value in dict(self.offline_teacher_logits_paths).items()}
        for name in (
            "selected_consumer_json",
            "oracle_teacher_checkpoint",
            "oracle_teacher_config_path",
            "oracle_run_report_path",
            "oracle_teacher_logits_dir",
            "offline_teacher_logits_dir",
            "student_warm_start_checkpoint",
            "predictor_warm_start_checkpoint",
        ):
            value = getattr(self, name)
            setattr(self, name, None if not value else str(value))
        self.consumer_id = None if not self.consumer_id else str(self.consumer_id)
        if self.field_gate_mode is not None and str(self.field_gate_mode) not in FIELD_GATE_MODES:
            raise ValueError(f"field_gate_mode must be one of {FIELD_GATE_MODES}")
        if bool(self.evaluate_final_test) and not bool(self.confirm_final_test):
            raise ValueError("final-test deployable evaluation requires confirm_final_test=True")
        if str(self.final_test_split) in {str(self.train_split), str(self.val_split), str(self.stack_val_split)}:
            raise ValueError("final_test_split must be disjoint from training and validation splits")
        for name in ("max_train_jets", "max_val_jets", "max_stack_val_jets", "max_final_test_jets"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when provided")
            setattr(self, name, None if value is None else int(value))


@dataclass(frozen=True)
class ResolvedCurriculumRun:
    run_id: str
    selected_consumer_id: str | None
    selected_alpha_endpoint: float | None
    paired_consumer_mode: bool
    gate_mode: str
    student_init_source: str
    loss_weights: Mapping[str, float]
    scheduler: Any
    oracle_path_fallback_downgrade: bool
    recipe_equivalent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "selected_consumer_id": self.selected_consumer_id,
            "selected_alpha_endpoint": self.selected_alpha_endpoint,
            "paired_consumer_mode": self.paired_consumer_mode,
            "gate_mode": self.gate_mode,
            "student_init_source": self.student_init_source,
            "loss_weights": dict(self.loss_weights),
            "oracle_path_fallback_downgrade": self.oracle_path_fallback_downgrade,
            "scientific_recipe_equivalent": self.recipe_equivalent,
        }


class _Stage1aP0Scheduler:
    """Minimal scheduler interface for P0, which intentionally has no consumer."""

    def __init__(self, loss_weights: Mapping[str, float], total_epochs: int) -> None:
        self.loss_weights = dict(loss_weights)
        self.total_epochs = int(total_epochs)

    def state_for_epoch(self, epoch: int, *, total_epochs: int | None = None) -> dict[str, Any]:
        del total_epochs
        return {
            "epoch": int(epoch),
            "alpha": 0.0,
            "active_consumer_id": None,
            "loss_weights": dict(self.loss_weights),
            "teacher": {},
            "selected_consumer_id": None,
            "selected_alpha_endpoint": None,
            "paired_consumer_mode": False,
            "stage": "1a",
        }

    def run_report_payload(self, num_epochs: int | None = None) -> dict[str, Any]:
        count = self.total_epochs if num_epochs is None else int(num_epochs)
        return {
            "stage": "1a",
            "consumer_required": False,
            "epochs": [self.state_for_epoch(epoch) for epoch in range(count)],
            "epoch_count": count,
        }


class _AlignedCurriculumDataset:
    """Adds two independently provenance-checked teacher-logit views."""

    def __init__(
        self,
        base: Any,
        *,
        oracle_logits: TeacherLogitBlock | None = None,
        offline_logits: TeacherLogitBlock | None = None,
    ) -> None:
        self.base = base
        self.oracle_logits = None if oracle_logits is None else np.asarray(oracle_logits.logits[: len(base)], dtype=np.float32)
        self.offline_logits = None if offline_logits is None else np.asarray(offline_logits.logits[: len(base)], dtype=np.float32)
        self.oracle_logits_metadata = {} if oracle_logits is None else dict(oracle_logits.metadata)
        self.offline_logits_metadata = {} if offline_logits is None else dict(offline_logits.metadata)

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        if self.oracle_logits is not None:
            item["oracle_teacher_logits"] = self.oracle_logits[index]
        if self.offline_logits is not None:
            item["offline_teacher_logits"] = self.offline_logits[index]
        return item


def _finite_nonnegative(value: Any, *, name: str) -> float:
    output = float(value)
    if not math.isfinite(output) or output < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return output


def _piecewise_pilot_schedule(endpoint: float) -> tuple[dict[str, Any], ...]:
    endpoint = float(endpoint)
    if endpoint <= 0.25:
        return ({"epoch": 0, "alpha": endpoint},)
    if endpoint <= 0.50:
        return ({"epoch": 0, "alpha": 0.25}, {"epoch": 3, "alpha": endpoint})
    return (
        {"epoch": 0, "alpha": 0.25},
        {"epoch": 3, "alpha": 0.50},
        {"epoch": 6, "alpha": endpoint},
    )


def resolve_curriculum_run(config: LocalResidualFieldCurriculumTrainConfig) -> ResolvedCurriculumRun:
    """Resolve and validate a named pilot recipe without guessing a consumer."""

    run_id = str(config.run_id)
    paired = bool(config.confirm_paired_consumers or paired_consumers_confirmed_from_env(os.environ))
    selected_id: str | None = None
    endpoint: float | None = None
    selected_path = config.selected_consumer_json
    if run_id in CURRICULUM_STAGE1B_RUN_IDS:
        if selected_path:
            record = load_selected_consumer_record(selected_path)
            selected_id = record.selected_consumer_id
            endpoint = record.selected_alpha_endpoint
            if config.consumer_id and str(config.consumer_id) != selected_id and not paired:
                raise ValueError(
                    f"{run_id} consumer_id {config.consumer_id!r} differs from selected P7a consumer {selected_id!r}"
                )
            if config.selected_alpha_endpoint is not None and not math.isclose(
                float(config.selected_alpha_endpoint), float(endpoint), rel_tol=0.0, abs_tol=1.0e-8
            ) and not paired:
                raise ValueError(
                    f"{run_id} selected_alpha_endpoint {config.selected_alpha_endpoint} differs from "
                    f"selected_consumer.json endpoint {endpoint}"
                )
            if paired:
                selected_id = str(config.consumer_id or selected_id)
                endpoint = float(config.selected_alpha_endpoint if config.selected_alpha_endpoint is not None else endpoint)
        elif paired:
            if not config.consumer_id or config.selected_alpha_endpoint is None:
                raise ValueError("paired-consumer mode requires explicit consumer_id and selected_alpha_endpoint")
            selected_id = str(config.consumer_id)
            endpoint = float(config.selected_alpha_endpoint)
        else:
            raise ValueError(
                f"{run_id} requires selected_consumer.json unless paired-consumer mode is explicitly enabled"
            )
        if selected_id not in {"Ofull", "Orobust_light"}:
            raise ValueError("Stage 1b selected consumer must be Ofull or Orobust_light")
        if endpoint is None or not (0.0 <= float(endpoint) <= 1.0):
            raise ValueError("Stage 1b requires selected_alpha_endpoint in [0, 1]")
    elif any((config.consumer_id, config.selected_consumer_json, config.selected_alpha_endpoint is not None)):
        raise ValueError("P0 is a Stage 1a baseline and must not be launched with a selected consumer")

    weights = dict(PILOT_LOSS_WEIGHTS[run_id])
    if run_id in {"Q0", "Q3"} and config.loss_weight_overrides:
        raise ValueError(f"{run_id} loss weights are fixed by the pilot ablation contract")
    weights.update(dict(config.loss_weight_overrides))
    fallback_downgrade = bool(config.oracle_logit_only_fallback and weights["oracle_path"] > 0.0)
    if fallback_downgrade:
        weights["oracle_path"] = 0.0

    default_gate = FIELD_GATE_MODE_LEARNED_SIGMOID if run_id in {"P7a", "P7b", "Q0", "Q3"} else FIELD_GATE_MODE_NONE
    gate_mode = str(config.field_gate_mode or default_gate)
    if run_id in {"Q0", "Q3"} and gate_mode != FIELD_GATE_MODE_LEARNED_SIGMOID:
        raise ValueError(f"{run_id} must use the selected P7a learned confidence-gate recipe")
    if gate_mode == FIELD_GATE_MODE_NONE and float(weights["gate"]) > 0.0:
        raise ValueError("a positive gate loss requires a confidence gate mode")

    if run_id == "P0":
        scheduler: Any = _Stage1aP0Scheduler(weights, int(config.epochs))
    else:
        assert selected_id is not None and endpoint is not None
        if run_id == "P2":
            schedule_payload: dict[str, Any] = {"alpha_schedule": ALPHA_SCHEDULE_FIXED, "fixed_alpha": 0.25}
        elif run_id == "Q3":
            schedule_payload = {"alpha_schedule": ALPHA_SCHEDULE_FIXED, "fixed_alpha": float(endpoint)}
        else:
            schedule_payload = {
                "alpha_schedule": ALPHA_SCHEDULE_PIECEWISE,
                "piecewise_alpha": _piecewise_pilot_schedule(float(endpoint)),
            }
        if config.alpha_schedule is not None:
            if run_id in {"Q0", "Q3"}:
                raise ValueError(f"{run_id} alpha schedule is fixed by the pilot ablation contract")
            schedule_payload["alpha_schedule"] = str(config.alpha_schedule)
        if config.fixed_alpha is not None:
            schedule_payload["fixed_alpha"] = float(config.fixed_alpha)
        if config.piecewise_alpha:
            schedule_payload["piecewise_alpha"] = tuple(config.piecewise_alpha)
        schedule_payload.update(
            {
                "sigmoid_alpha_start": float(config.sigmoid_alpha_start),
                "sigmoid_alpha_end": config.sigmoid_alpha_end,
                "sigmoid_alpha_midpoint": float(config.sigmoid_alpha_midpoint),
                "sigmoid_alpha_sharpness": float(config.sigmoid_alpha_sharpness),
                "loss_weights": weights,
                "loss_weight_schedule": dict(config.loss_weight_schedule),
                "selected_consumer_id": selected_id,
                "selected_alpha_endpoint": float(endpoint),
                "paired_consumer_mode": bool(paired),
            }
        )
        if selected_path and not paired:
            scheduler_config = LocalResidualFieldCurriculumSchedulerConfig.from_selected_consumer(
                schedule_payload,
                selected_consumer_path=selected_path,
                require_selected_consumer=True,
            )
        else:
            scheduler_config = LocalResidualFieldCurriculumSchedulerConfig(**schedule_payload)
        scheduler = LocalResidualFieldCurriculumScheduler(scheduler_config, total_epochs=int(config.epochs))
        if run_id == "Q0":
            expected = dict(PILOT_LOSS_WEIGHTS["P7a"])
            expected["oracle_path"] = 0.0
            if weights != expected:
                raise ValueError("Q0 must exactly match P7a except oracle-path KD is disabled")
        if run_id == "Q3":
            first = scheduler.state_for_epoch(0)
            last = scheduler.state_for_epoch(max(int(config.epochs) - 1, 0))
            if not math.isclose(float(first["alpha"]), float(endpoint), abs_tol=1.0e-8) or first["alpha"] != last["alpha"]:
                raise ValueError("Q3 must use selected_alpha_endpoint from epoch 0 with no ramp")

    student_source = selected_id if run_id == "P7b" else "A0"
    if run_id in {"P7a", "P7b", "Q0", "Q3"} and not config.student_warm_start_checkpoint:
        raise ValueError(f"{run_id} requires the declared student warm-start checkpoint")
    return ResolvedCurriculumRun(
        run_id=run_id,
        selected_consumer_id=selected_id,
        selected_alpha_endpoint=endpoint,
        paired_consumer_mode=paired,
        gate_mode=gate_mode,
        student_init_source=str(student_source),
        loss_weights=weights,
        scheduler=scheduler,
        oracle_path_fallback_downgrade=fallback_downgrade,
        recipe_equivalent=not fallback_downgrade,
    )


def _candidate_logit_path(root: str | None, split: str) -> str | None:
    if not root:
        return None
    directory = Path(root)
    for name in (
        f"{split}_predictions.npz",
        f"{split}_teacher_logits.npz",
        f"{split}_logits.npz",
        f"{split}.npz",
    ):
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    return None


def _resolve_logit_paths(
    *,
    root: str | None,
    direct: Mapping[str, str],
    splits: Sequence[str],
    required: bool,
    cache_name: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    missing: list[str] = []
    for split in splits:
        path = direct.get(str(split)) or _candidate_logit_path(root, str(split))
        if path and Path(path).is_file():
            paths[str(split)] = str(path)
        elif required:
            missing.append(str(split))
    if missing:
        raise FileNotFoundError(f"{cache_name} is required but missing for splits {missing}")
    return paths


def _validate_logit_block(
    block: TeacherLogitBlock,
    *,
    dataset: Any,
    split: str,
    cache_name: str,
) -> None:
    if block.logits.ndim != 2 or int(block.logits.shape[0]) < len(dataset):
        raise ValueError(f"{cache_name} rows/shape do not cover {split}: {block.logits.shape}")
    if block.labels is None or not np.array_equal(np.asarray(block.labels[: len(dataset)]), np.asarray(dataset.labels)):
        raise ValueError(f"{cache_name} labels do not match {split}")
    metadata = dict(block.metadata)
    if str(metadata.get("split") or "") != str(split):
        raise ValueError(f"{cache_name} split metadata {metadata.get('split')!r} != {split!r}")
    alignment = dataset.metadata.get("alignment_report", {})
    expected_identity = alignment.get("jet_identity_hash")
    if not metadata.get("jet_identity_hash"):
        raise ValueError(f"{cache_name} requires jet_identity_hash metadata")
    if expected_identity and str(metadata["jet_identity_hash"]) != str(expected_identity):
        raise ValueError(f"{cache_name} jet_identity_hash does not match {split}")


def _load_datasets(
    config: LocalResidualFieldCurriculumTrainConfig,
    resolved: ResolvedCurriculumRun,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    splits = (str(config.train_split), str(config.val_split), str(config.stack_val_split))
    states = [resolved.scheduler.state_for_epoch(epoch, total_epochs=int(config.epochs)) for epoch in range(int(config.epochs))]
    oracle_cache_required = bool(config.oracle_logit_only_fallback)
    offline_cache_required = bool(
        config.student_kd_source == "offline_teacher"
        and any(float(state["loss_weights"].get("student_kd", 0.0)) > 0.0 for state in states)
    )
    oracle_paths = _resolve_logit_paths(
        root=config.oracle_teacher_logits_dir,
        direct=config.oracle_teacher_logits_paths,
        splits=splits,
        required=oracle_cache_required,
        cache_name="oracle teacher logits cache",
    )
    offline_paths = _resolve_logit_paths(
        root=config.offline_teacher_logits_dir,
        direct=config.offline_teacher_logits_paths,
        splits=splits,
        required=offline_cache_required,
        cache_name="offline teacher logits cache",
    )
    max_by_split = {
        str(config.train_split): config.max_train_jets,
        str(config.val_split): config.max_val_jets,
        str(config.stack_val_split): config.max_stack_val_jets,
    }
    datasets: dict[str, Any] = {}
    for split in splits:
        base = load_local_particle_residual_field_dataset(
            LocalParticleResidualFieldDatasetConfig(
                hlt_cache_dir=config.hlt_cache_dir,
                target_cache_dir=config.target_cache_dir,
                split=split,
                manifest_path=config.manifest_path,
                max_jets=max_by_split[split],
                verify_hash=bool(config.verify_hash),
                require_manifest_match=bool(config.require_manifest_match),
            )
        )
        oracle_block = _load_teacher_logits(oracle_paths[split]) if split in oracle_paths else None
        offline_block = _load_teacher_logits(offline_paths[split]) if split in offline_paths else None
        if oracle_block is not None:
            _validate_logit_block(oracle_block, dataset=base, split=split, cache_name="oracle teacher logits cache")
        if offline_block is not None:
            _validate_logit_block(offline_block, dataset=base, split=split, cache_name="offline teacher logits cache")
        datasets[split] = _AlignedCurriculumDataset(base, oracle_logits=oracle_block, offline_logits=offline_block)
    return datasets, oracle_paths, offline_paths


def _dataset_provenance(datasets: Mapping[str, Any]) -> dict[str, Any]:
    return {str(split): _jsonable(dataset.metadata) for split, dataset in datasets.items()}


def _provenance_hashes(dataset: Any) -> dict[str, Any]:
    alignment = dataset.metadata.get("alignment_report", {})
    return {
        key: alignment.get(key)
        for key in (
            "source_manifest_hash",
            "hlt_content_hash",
            "offline_content_hash",
            "target_content_hash",
            "jet_identity_hash",
        )
        if alignment.get(key) not in (None, "")
    }


def _build_joint_model(
    config: LocalResidualFieldCurriculumTrainConfig,
    resolved: ResolvedCurriculumRun,
    train_dataset: Any,
    *,
    device: Any,
) -> tuple[LocalResidualFieldCurriculumJointModel, dict[str, Any]]:
    field_names = tuple(str(name) for name in train_dataset.field_names)
    field_groups = {str(name): tuple(int(index) for index in indices) for name, indices in train_dataset.field_groups.items()}
    predictor_payload = _load_predictor_warm_start_payload(config.predictor_warm_start_checkpoint)
    checkpoint_reconstructor_config = _predictor_reconstructor_config(predictor_payload)
    dataset_max_particles = int(train_dataset.tokens.shape[1])
    reconstructor_max_particles = int(
        checkpoint_reconstructor_config.get("max_particles", dataset_max_particles)
    )
    if reconstructor_max_particles < dataset_max_particles:
        raise ValueError(
            "predictor warm-start max_particles is smaller than the curriculum dataset width: "
            f"{reconstructor_max_particles} < {dataset_max_particles}"
        )
    reconstructor_config = LocalResidualFieldReconstructorConfig(
        field_dim=len(field_names),
        field_names=field_names,
        field_groups=field_groups,
        d_model=int(config.reconstructor_d_model),
        num_heads=int(config.reconstructor_num_heads),
        num_layers=int(config.reconstructor_num_layers),
        context_layers=int(config.reconstructor_context_layers),
        dropout=float(config.reconstructor_dropout),
        attention_dropout=float(config.reconstructor_attention_dropout),
        # Preserve the checkpoint's rank-embedding capacity.  The standalone
        # C0 trainer intentionally uses the model default (256), even when the
        # cached HLT tensors contain only 128 particle slots.
        max_particles=reconstructor_max_particles,
    )
    student_config = LocalResidualFieldTaggerConfig(
        num_classes=int(config.num_classes),
        field_dim=len(field_names),
        field_source=RESIDUAL_FIELD_SOURCE_ZERO,
        model_size=str(config.model_size),
        residual_field_clip_value=float(config.residual_field_clip_value),
        field_names=field_names,
        field_groups=field_groups,
    )
    epoch_states = [resolved.scheduler.state_for_epoch(epoch) for epoch in range(int(config.epochs))]
    needs_oracle_path = any(
        float(state["loss_weights"].get("oracle_path", 0.0)) > 0.0
        for state in epoch_states
    )
    needs_oracle_student_kd = config.student_kd_source == "oracle_true" and any(
        float(state["loss_weights"].get("student_kd", 0.0)) > 0.0
        for state in epoch_states
    )
    needs_oracle = needs_oracle_path or needs_oracle_student_kd or bool(config.oracle_logit_only_fallback)
    oracle_config = None
    if needs_oracle:
        if not config.oracle_teacher_checkpoint:
            raise FileNotFoundError("this curriculum recipe requires oracle_teacher_checkpoint")
        oracle_config = FrozenOracleConsumerConfig(
            checkpoint=config.oracle_teacher_checkpoint,
            consumer_id=resolved.selected_consumer_id,
            alpha=float(resolved.selected_alpha_endpoint or 0.0),
            teacher_config_path=config.oracle_teacher_config_path,
            run_report_path=config.oracle_run_report_path,
            oracle_logit_only_fallback=bool(config.oracle_logit_only_fallback),
            oracle_forward_microbatch_size=config.oracle_forward_microbatch_size,
        )
    reset_mode = config.residual_projection_reset
    freeze_schedule = config.freeze_schedule
    if resolved.run_id == "P7b":
        reset_mode = reset_mode or RESIDUAL_PROJECTION_RESET_SCALE
        freeze_schedule = freeze_schedule or FREEZE_SCHEDULE_RESIDUAL_WARMUP_THEN_UPPER
    else:
        reset_mode = reset_mode or RESIDUAL_PROJECTION_RESET_NONE
        freeze_schedule = freeze_schedule or FREEZE_SCHEDULE_NONE
    joint_config = LocalResidualFieldCurriculumJointConfig(
        reconstructor_config=reconstructor_config,
        student_config=student_config,
        oracle_consumer_config=oracle_config,
        field_gate_mode=resolved.gate_mode,
        initial_gate_bias_prob=float(config.initial_gate_bias_prob),
        gate_reliability_loss_weight=1.0 if resolved.gate_mode != FIELD_GATE_MODE_NONE else 0.0,
        gate_reliability_error_scale=float(config.gate_reliability_error_scale),
        student_init_source=resolved.student_init_source,
        student_init_checkpoint=config.student_warm_start_checkpoint,
        require_student_init_checkpoint=resolved.run_id in {"P7a", "P7b", "Q0", "Q3"},
        residual_projection_reset=reset_mode,
        residual_projection_scale=float(config.residual_projection_scale),
        freeze_schedule=freeze_schedule,
        freeze_phase1_epochs=int(config.freeze_phase1_epochs),
        freeze_phase2_epochs=int(config.freeze_phase2_epochs),
        optimizer_group_learning_rates=dict(config.optimizer_group_learning_rates),
        field_names=field_names,
        field_groups=field_groups,
        normalization_metadata=dict(train_dataset.metadata.get("target_metadata") or {}),
        provenance_hashes=_provenance_hashes(train_dataset),
    )
    model = LocalResidualFieldCurriculumJointModel(joint_config).to(device)
    if model.oracle_consumer is not None and str(model.oracle_consumer.consumer_id) != str(resolved.selected_consumer_id):
        raise ValueError(
            f"oracle teacher checkpoint consumer {model.oracle_consumer.consumer_id!r} differs from "
            f"selected P7a consumer {resolved.selected_consumer_id!r}"
        )
    warm_report = _warm_start_predictor(
        model,
        config.predictor_warm_start_checkpoint,
        payload=predictor_payload,
    )
    return model, warm_report


def _load_predictor_warm_start_payload(path: str | None) -> Mapping[str, Any] | None:
    if not path:
        return None
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"predictor warm-start checkpoint does not exist: {checkpoint}")
    payload = _torch_load_checkpoint(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("predictor warm-start checkpoint must contain a mapping")
    return payload


def _predictor_reconstructor_config(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if payload is None:
        return {}
    raw_model_config = payload.get("model_config")
    if not isinstance(raw_model_config, Mapping):
        return {}
    nested = raw_model_config.get("reconstructor_config")
    return nested if isinstance(nested, Mapping) else raw_model_config


def _warm_start_predictor(
    model: LocalResidualFieldCurriculumJointModel,
    path: str | None,
    *,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = {"requested": bool(path), "applied": False, "checkpoint": path, "checkpoint_hash": None, "confidence_heads_loaded": False}
    if not path:
        return report
    checkpoint = Path(path)
    if payload is None:
        payload = _load_predictor_warm_start_payload(path)
    assert payload is not None
    checkpoint_reconstructor_config = _predictor_reconstructor_config(payload)
    checkpoint_field_names = tuple(str(name) for name in checkpoint_reconstructor_config.get("field_names", ()))
    current_field_names = tuple(str(name) for name in model.config.field_names)
    if checkpoint_field_names and checkpoint_field_names != current_field_names:
        raise ValueError("predictor warm-start field_names do not match the current field target cache ordering")
    checkpoint_provenance = payload.get("provenance_hashes")
    if isinstance(checkpoint_provenance, Mapping):
        for key, current_value in model.config.provenance_hashes.items():
            checkpoint_value = checkpoint_provenance.get(key)
            if current_value not in (None, "") and checkpoint_value not in (None, "") and str(current_value) != str(checkpoint_value):
                raise ValueError(f"predictor warm-start provenance mismatch for {key}")
    state = payload.get("reconstructor_state_dict") or payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("predictor warm-start checkpoint has no reconstructor state")
    model.reconstructor.load_state_dict(state, strict=True)
    confidence_state = payload.get("confidence_heads_state_dict")
    if isinstance(confidence_state, Mapping):
        model.confidence_heads.load_state_dict(confidence_state, strict=True)
        report["confidence_heads_loaded"] = True
    report.update({"applied": True, "checkpoint_hash": _sha256_file(checkpoint)})
    return report


def _temperature_kd(student_logits: Any, teacher_logits: Any, *, temperature: float) -> Any:
    torch = require_torch()
    temp = float(temperature)
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / temp, dim=-1),
        torch.nn.functional.softmax(teacher_logits.detach().to(student_logits) / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)


def _masked_field_huber(pred: Any, target: Any, mask: Any, *, beta: float) -> Any:
    torch = require_torch()
    target = target.to(device=pred.device, dtype=pred.dtype)
    valid = mask.to(device=pred.device, dtype=torch.bool).unsqueeze(-1).expand_as(pred)
    if not bool(valid.detach().any().cpu().item()):
        return pred.new_zeros(())
    return torch.nn.functional.smooth_l1_loss(pred[valid], target[valid], beta=float(beta), reduction="mean")


def _residual_regularization(fields: Any, mask: Any) -> tuple[Any, dict[str, float]]:
    torch = require_torch()
    valid = mask.to(device=fields.device, dtype=torch.bool)
    expanded = valid.unsqueeze(-1).expand_as(fields)
    magnitude = fields[expanded].square().mean() if bool(expanded.detach().any().cpu().item()) else fields.new_zeros(())
    if int(fields.shape[1]) > 1:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        pair_expanded = pair_valid.unsqueeze(-1).expand(-1, -1, int(fields.shape[-1]))
        differences = fields[:, 1:] - fields[:, :-1]
        smoothness = differences[pair_expanded].square().mean() if bool(pair_expanded.detach().any().cpu().item()) else fields.new_zeros(())
    else:
        smoothness = fields.new_zeros(())
    total = magnitude + smoothness
    return total, {
        "residual_magnitude": float(magnitude.detach().cpu().item()),
        "residual_smoothness": float(smoothness.detach().cpu().item()),
    }


def compute_curriculum_batch_loss(
    output: Any,
    batch: Mapping[str, Any],
    *,
    loss_weights: Mapping[str, float],
    student_kd_source: str,
    kd_temperature: float,
    field_huber_beta: float,
) -> tuple[Any, dict[str, float]]:
    """Compose all Step 7 loss terms and fail if a requested teacher is absent."""

    torch = require_torch()
    weights = {name: float(loss_weights.get(name, 0.0)) for name in CURRICULUM_LOSS_NAMES}
    ce = torch.nn.functional.cross_entropy(output.student_logits, batch["labels"])
    zero = ce.new_zeros(())
    student_kd = zero
    if weights["student_kd"] > 0.0:
        teacher = output.oracle_true_logits if student_kd_source == "oracle_true" else batch.get("offline_teacher_logits")
        if teacher is None:
            raise ValueError(f"student KD requested from {student_kd_source}, but aligned teacher logits are absent")
        student_kd = _temperature_kd(output.student_logits, teacher, temperature=float(kd_temperature))
    oracle_path = zero
    if weights["oracle_path"] > 0.0:
        if output.oracle_pred_logits is None or output.oracle_true_logits is None:
            raise ValueError("oracle-path KD requested, but differentiable predicted/true oracle logits are absent")
        oracle_path = _temperature_kd(output.oracle_pred_logits, output.oracle_true_logits, temperature=float(kd_temperature))
    field_loss = zero
    if weights["field"] > 0.0:
        if batch.get("target_fields") is None:
            raise ValueError("field loss requested without field target cache")
        field_loss = _masked_field_huber(
            output.pred_fields_raw,
            batch["target_fields"],
            batch.get("target_mask", batch["raw_mask"]),
            beta=float(field_huber_beta),
        )
    gate_loss = zero
    if weights["gate"] > 0.0:
        if output.field_gate_loss is None:
            raise ValueError("gate loss requested but the joint model did not produce reliability supervision")
        gate_loss = output.field_gate_loss
    regularization, reg_diagnostics = _residual_regularization(output.pred_fields_effective, batch["raw_mask"])
    total = (
        weights["ce"] * ce
        + weights["student_kd"] * student_kd
        + weights["oracle_path"] * oracle_path
        + weights["field"] * field_loss
        + weights["gate"] * gate_loss
        + weights["reg"] * regularization
    )
    diagnostics = {
        "loss": float(total.detach().cpu().item()),
        "cross_entropy": float(ce.detach().cpu().item()),
        "student_kd_loss": float(student_kd.detach().cpu().item()),
        "oracle_path_kd_loss": float(oracle_path.detach().cpu().item()),
        "field_huber_loss": float(field_loss.detach().cpu().item()),
        "gate_loss": float(gate_loss.detach().cpu().item()),
        "regularization_loss": float(regularization.detach().cpu().item()),
        **reg_diagnostics,
    }
    return total, diagnostics


def _gradients_finite(model: Any) -> bool:
    torch = require_torch()
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().detach().cpu().item()) for parameter in model.parameters())


def _run_epoch(
    model: LocalResidualFieldCurriculumJointModel,
    loader: Any,
    *,
    device: Any,
    optimizer: Any | None,
    scaler: Any | None,
    amp_enabled: bool,
    grad_clip_norm: float,
    accumulation_steps: int,
    epoch_state: Mapping[str, Any],
    config: LocalResidualFieldCurriculumTrainConfig,
    oracle_logits_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    if model.oracle_consumer is not None:
        model.oracle_consumer._enforce_frozen_eval()
    if training:
        optimizer.zero_grad(set_to_none=True)
    labels_chunks: list[np.ndarray] = []
    logits_chunks: list[np.ndarray] = []
    attempted_jets = seen = finite_batches = nonfinite_batches = nonfinite_grad_batches = optimizer_steps = 0
    sums = {name: 0.0 for name in (
        "loss", "cross_entropy", "student_kd_loss", "oracle_path_kd_loss", "field_huber_loss",
        "gate_loss", "regularization_loss", "residual_magnitude", "residual_smoothness",
    )}
    total_batches = len(loader)
    pending = 0
    for batch_index, batch in enumerate(loader):
        batch = move_local_particle_residual_field_batch_to_device(batch, device)
        batch_n = int(batch["labels"].shape[0])
        attempted_jets += batch_n
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            with amp_autocast_context(bool(amp_enabled)):
                output = model(
                    batch["points"],
                    batch["features"],
                    batch["lorentz_vectors"],
                    batch["mask"],
                    tokens=batch["tokens"],
                    raw_mask=batch["raw_mask"],
                    indices=batch.get("indices"),
                    target_fields=batch.get("target_fields"),
                    oracle_alpha=float(epoch_state["alpha"]),
                    cached_oracle_true_logits=(
                        batch.get("oracle_teacher_logits") if bool(config.oracle_logit_only_fallback) else None
                    ),
                    cached_oracle_true_logits_metadata=(
                        oracle_logits_metadata if bool(config.oracle_logit_only_fallback) else None
                    ),
                    return_outputs=True,
                )
                loss, components = compute_curriculum_batch_loss(
                    output,
                    batch,
                    loss_weights=epoch_state["loss_weights"],
                    student_kd_source=str(config.student_kd_source),
                    kd_temperature=float(config.kd_temperature),
                    field_huber_beta=float(config.field_huber_beta),
                )
        finite = bool(torch.isfinite(loss).detach().cpu().item()) and bool(
            torch.isfinite(output.student_logits).all().detach().cpu().item()
        )
        if not finite:
            nonfinite_batches += 1
            pending = 0
            if training:
                optimizer.zero_grad(set_to_none=True)
            continue
        if training:
            scaled_loss = loss / float(accumulation_steps)
            if scaler is not None and bool(amp_enabled):
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            pending += 1
            should_step = pending >= int(accumulation_steps) or batch_index + 1 == total_batches
            if should_step:
                if scaler is not None and bool(amp_enabled):
                    scaler.unscale_(optimizer)
                if pending < int(accumulation_steps):
                    correction = float(accumulation_steps) / float(max(pending, 1))
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                if float(grad_clip_norm) > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    grads_ok = bool(torch.isfinite(grad_norm).detach().cpu().item())
                else:
                    grads_ok = _gradients_finite(model)
                if not grads_ok:
                    nonfinite_grad_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None and bool(amp_enabled):
                        scaler.update()
                    pending = 0
                    continue
                if scaler is not None and bool(amp_enabled):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending = 0
        labels_np = batch["labels"].detach().cpu().numpy().astype(np.int64)
        logits_np = output.student_logits.detach().float().cpu().numpy()
        labels_chunks.append(labels_np)
        logits_chunks.append(logits_np)
        seen += batch_n
        finite_batches += 1
        for name in sums:
            sums[name] += float(components[name]) * batch_n
    valid_fraction = float(seen) / float(attempted_jets) if attempted_jets else 0.0
    base = {
        "n_jets": int(seen),
        "attempted_jets": int(attempted_jets),
        "valid_fraction": valid_fraction,
        "total_batches": int(total_batches),
        "finite_batches": int(finite_batches),
        "nonfinite_batches": int(nonfinite_batches),
        "nonfinite_grad_batches": int(nonfinite_grad_batches),
        "nonfinite_fraction": float(nonfinite_batches + nonfinite_grad_batches) / float(max(total_batches, 1)),
        "optimizer_steps": int(optimizer_steps),
        "alpha": float(epoch_state["alpha"]),
        "loss_weights": dict(epoch_state["loss_weights"]),
    }
    if seen == 0:
        return {**base, "loss": float("nan"), "accuracy": 0.0}
    labels_all = np.concatenate(labels_chunks)
    logits_all = np.concatenate(logits_chunks)
    predictions = np.argmax(logits_all, axis=1).astype(np.int64)
    metrics = classification_metrics_from_predictions(
        preds=predictions,
        labels=labels_all,
        loss_sum=sums["cross_entropy"],
        logits=logits_all if int(config.num_classes) == 2 else None,
        label_names=tuple(config.label_names),
    )
    metrics.update({name: value / float(seen) for name, value in sums.items()})
    metrics.update(base)
    return metrics


def _coverage_valid(metrics: Mapping[str, Any], *, expected: int, required_fraction: float) -> tuple[bool, str]:
    seen = int(metrics.get("n_jets", 0) or 0)
    minimum = int(math.ceil(int(expected) * float(required_fraction)))
    if seen < minimum:
        return False, f"finite validation coverage {seen}/{expected} below required {minimum} ({required_fraction:.4f})"
    try:
        loss = float(metrics.get("loss"))
    except (TypeError, ValueError):
        return False, "validation loss is missing"
    if not math.isfinite(loss):
        return False, "validation loss is nonfinite"
    return True, ""


def _load_deployable_state(model: LocalResidualFieldCurriculumJointModel, path: str | Path) -> None:
    payload = _torch_load_checkpoint(path, map_location="cpu")
    if payload.get("contract") != LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT:
        raise ValueError("best checkpoint is not an oracle-free deployable curriculum checkpoint")
    if payload.get("oracle_consumer_included") is not False:
        raise ValueError("best deployable checkpoint unexpectedly contains an oracle consumer")
    model.reconstructor.load_state_dict(payload["reconstructor_state_dict"], strict=True)
    model.student.load_state_dict(payload["student_state_dict"], strict=True)
    model.confidence_heads.load_state_dict(payload["confidence_heads_state_dict"], strict=True)


def _write_epoch_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = {
            "epoch": row.get("epoch"),
            "alpha": row.get("schedule", {}).get("alpha"),
            "freeze_phase": row.get("freeze", {}).get("phase"),
        }
        for name, weight in dict(row.get("schedule", {}).get("loss_weights", {})).items():
            output[f"weight_{name}"] = weight
        for split in ("train", "model_val"):
            for name in (
                "loss", "cross_entropy", "student_kd_loss", "oracle_path_kd_loss", "field_huber_loss",
                "gate_loss", "regularization_loss", "accuracy", "n_jets", "attempted_jets", "valid_fraction",
                "nonfinite_batches", "nonfinite_grad_batches", "nonfinite_fraction", "valid_for_selection",
            ):
                if name in row.get(split, {}):
                    output[f"{split}_{name}"] = row[split][name]
        flattened.append(output)
    fieldnames: list[str] = []
    for row in flattened:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["epoch"])
        writer.writeheader()
        writer.writerows(flattened)


def _evaluate_final_test_deployable(
    model: LocalResidualFieldCurriculumJointModel,
    config: LocalResidualFieldCurriculumTrainConfig,
    *,
    device: Any,
) -> dict[str, Any]:
    if not bool(config.confirm_final_test):
        raise ValueError("final-test deployable evaluation requires explicit confirmation")
    dataset = load_local_particle_residual_field_hlt_only_dataset(
        LocalParticleResidualFieldDatasetConfig(
            hlt_cache_dir=config.hlt_cache_dir,
            target_cache_dir=config.target_cache_dir,
            split=str(config.final_test_split),
            manifest_path=config.manifest_path,
            max_jets=config.max_final_test_jets,
            allow_final_test_targets=False,
            verify_hash=bool(config.verify_hash),
            require_manifest_match=bool(config.require_manifest_match),
        )
    )
    loader = make_local_particle_residual_field_loader(
        dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 3,
    )
    oracle = model.oracle_consumer
    model.oracle_consumer = None
    try:
        state = {"alpha": 0.0, "loss_weights": {"ce": 1.0}}
        metrics = _run_epoch(
            model,
            loader,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=bool(config.amp and getattr(device, "type", str(device)) == "cuda"),
            grad_clip_norm=0.0,
            accumulation_steps=1,
            epoch_state=state,
            config=config,
            oracle_logits_metadata=None,
        )
    finally:
        model.oracle_consumer = oracle
    ok, reason = _coverage_valid(
        metrics,
        expected=len(dataset),
        required_fraction=float(config.min_validation_valid_fraction),
    )
    if not ok:
        raise RuntimeError(f"final-test deployable evaluation failed coverage: {reason}")
    return {
        **metrics,
        "split": str(config.final_test_split),
        "runtime_inputs": "HLT_only",
        "uses_true_fields": False,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "oracle_teacher_loaded": False,
        "field_target_cache_loaded": False,
        "deployable": True,
        "selection_allowed": False,
    }


def train_local_residual_field_curriculum(
    config: LocalResidualFieldCurriculumTrainConfig,
    *,
    model: LocalResidualFieldCurriculumJointModel | None = None,
    datasets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a named curriculum run and emit only oracle-free selected checkpoints."""

    torch = require_torch()
    set_training_seed(int(config.seed))
    resolved = resolve_curriculum_run(config)
    output_dir = Path(config.output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.device))
    amp_enabled = bool(config.amp and getattr(device, "type", str(device)) == "cuda")
    if datasets is None:
        loaded_datasets, oracle_paths, offline_paths = _load_datasets(config, resolved)
    else:
        loaded_datasets = dict(datasets)
        oracle_paths = dict(config.oracle_teacher_logits_paths)
        offline_paths = dict(config.offline_teacher_logits_paths)
    required_splits = (str(config.train_split), str(config.val_split), str(config.stack_val_split))
    missing_splits = [split for split in required_splits if split not in loaded_datasets]
    if missing_splits:
        raise ValueError(f"curriculum datasets are missing splits {missing_splits}")
    train_dataset = loaded_datasets[str(config.train_split)]
    for split in required_splits[1:]:
        dataset = loaded_datasets[split]
        if tuple(dataset.field_names) != tuple(train_dataset.field_names) or dict(dataset.field_groups) != dict(train_dataset.field_groups):
            raise ValueError("field target cache schema differs across curriculum splits")
    if model is None:
        model, predictor_warm_start_report = _build_joint_model(config, resolved, train_dataset, device=device)
    else:
        model = model.to(device)
        predictor_warm_start_report = {"requested": False, "applied": False, "model_supplied": True}
    if bool(config.oracle_logit_only_fallback):
        states = [resolved.scheduler.state_for_epoch(epoch) for epoch in range(int(config.epochs))]
        alphas = {round(float(state["alpha"]), 12) for state in states}
        if len(alphas) != 1:
            raise ValueError("oracle_logit_only_fallback requires a fixed alpha cache; curriculum ramps are unsupported")

    train_loader = make_local_particle_residual_field_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    val_loader = make_local_particle_residual_field_loader(
        loaded_datasets[str(config.val_split)],
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )
    stack_loader = make_local_particle_residual_field_loader(
        loaded_datasets[str(config.stack_val_split)],
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 2,
    )
    dataset_metadata = _dataset_provenance({split: loaded_datasets[split] for split in required_splits})
    source_metadata = {
        "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_TRAIN_CONTRACT,
        "config": _jsonable(asdict(config)),
        "resolved_run": resolved.to_dict(),
        "scheduler": resolved.scheduler.run_report_payload(int(config.epochs)),
        "dataset_metadata": dataset_metadata,
        "oracle_teacher_logits_paths": oracle_paths,
        "offline_teacher_logits_paths": offline_paths,
        "predictor_warm_start": predictor_warm_start_report,
        "student_initialization": model.student_initialization_report,
        "adaptation": model.adaptation_report(),
        "deployment_contract": {
            "runtime_inputs": "HLT_only",
            "uses_true_fields": False,
            "uses_offline_particles": False,
            "uses_teacher_logits_at_runtime": False,
            "deployable": True,
        },
    }
    save_json(output_dir / "source_metadata.json", source_metadata)
    save_json(output_dir / "curriculum_schedule.json", resolved.scheduler.run_report_payload(int(config.epochs)))

    curves: list[dict[str, Any]] = []
    best_epoch = -1
    best_accuracy = float("-inf")
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    validation_stable = False
    current_phase = None
    optimizer = None
    optimizer_report: dict[str, Any] = {}
    scaler = amp_grad_scaler(bool(amp_enabled))
    best_path = output_dir / "best_model_val.pt"
    last_path = output_dir / "last.pt"
    for epoch in range(int(config.epochs)):
        freeze_report = model.apply_freeze_schedule(epoch, validation_stable=validation_stable)
        if freeze_report["phase"] != current_phase or optimizer is None:
            optimizer, optimizer_report = model.build_optimizer(weight_decay=float(config.weight_decay))
            current_phase = freeze_report["phase"]
        epoch_state = resolved.scheduler.state_for_epoch(epoch, total_epochs=int(config.epochs))
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip_norm=float(config.grad_clip_norm),
            accumulation_steps=int(config.gradient_accumulation_steps),
            epoch_state=epoch_state,
            config=config,
            oracle_logits_metadata=getattr(train_dataset, "oracle_logits_metadata", None),
        )
        val_dataset = loaded_datasets[str(config.val_split)]
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
            grad_clip_norm=0.0,
            accumulation_steps=1,
            epoch_state=epoch_state,
            config=config,
            oracle_logits_metadata=getattr(val_dataset, "oracle_logits_metadata", None),
        )
        val_ok, rejection = _coverage_valid(
            val_metrics,
            expected=len(val_dataset),
            required_fraction=float(config.min_validation_valid_fraction),
        )
        val_metrics["valid_for_selection"] = bool(val_ok)
        val_metrics["selection_expected_n_jets"] = len(val_dataset)
        val_metrics["selection_valid_fraction_required"] = float(config.min_validation_valid_fraction)
        if rejection:
            val_metrics["selection_rejection_reason"] = rejection
        validation_stable = bool(val_ok and int(val_metrics.get("nonfinite_batches", 0)) == 0)
        row = {
            "epoch": int(epoch),
            "schedule": _jsonable(epoch_state),
            "freeze": _jsonable(freeze_report),
            "optimizer": _jsonable(optimizer_report),
            "train": _jsonable(train_metrics),
            "model_val": _jsonable(val_metrics),
        }
        curves.append(row)
        accuracy = float(val_metrics.get("accuracy", float("nan")))
        if val_ok and math.isfinite(accuracy) and accuracy > best_accuracy:
            best_epoch = int(epoch)
            best_accuracy = accuracy
            best_metrics = dict(val_metrics)
            epochs_without_improvement = 0
            model.save_deployable_checkpoint(
                best_path,
                extra_metadata={
                    "training_contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_TRAIN_CONTRACT,
                    "run_id": resolved.run_id,
                    "epoch": int(epoch),
                    "model_val": _jsonable(val_metrics),
                    "schedule": _jsonable(epoch_state),
                    "selected_consumer_id": resolved.selected_consumer_id,
                    "selected_alpha_endpoint": resolved.selected_alpha_endpoint,
                    "teacher_used_during_training": (
                        None if model.oracle_consumer is None else str(model.oracle_consumer.consumer_id)
                    ),
                    "runtime_inputs": "HLT_only",
                    "uses_teacher_logits_at_runtime": False,
                    "scientific_recipe_equivalent": resolved.recipe_equivalent,
                },
            )
        else:
            epochs_without_improvement += 1
        if bool(config.save_last_checkpoint):
            model.save_deployable_checkpoint(
                last_path,
                extra_metadata={
                    "training_contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_TRAIN_CONTRACT,
                    "run_id": resolved.run_id,
                    "epoch": int(epoch),
                    "selected_consumer_id": resolved.selected_consumer_id,
                    "selected_alpha_endpoint": resolved.selected_alpha_endpoint,
                    "teacher_used_during_training": (
                        None if model.oracle_consumer is None else str(model.oracle_consumer.consumer_id)
                    ),
                    "runtime_inputs": "HLT_only",
                    "uses_teacher_logits_at_runtime": False,
                    "scientific_recipe_equivalent": resolved.recipe_equivalent,
                },
            )
        save_json(output_dir / "training_curves.json", {"epochs": curves, "selection_metric": "accuracy"})
        _write_epoch_csv(diagnostics_dir / "curriculum_epoch_metrics.csv", curves)
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement > int(config.early_stop_patience):
            break
    if best_epoch < 0 or not best_path.is_file():
        raise RuntimeError(
            "curriculum training produced no selectable checkpoint; every model_val epoch failed finite coverage"
        )
    _load_deployable_state(model, best_path)
    stack_dataset = loaded_datasets[str(config.stack_val_split)]
    best_state = resolved.scheduler.state_for_epoch(best_epoch, total_epochs=int(config.epochs))
    stack_metrics = _run_epoch(
        model,
        stack_loader,
        device=device,
        optimizer=None,
        scaler=None,
        amp_enabled=amp_enabled,
        grad_clip_norm=0.0,
        accumulation_steps=1,
        epoch_state=best_state,
        config=config,
        oracle_logits_metadata=getattr(stack_dataset, "oracle_logits_metadata", None),
    )
    stack_ok, stack_reason = _coverage_valid(
        stack_metrics,
        expected=len(stack_dataset),
        required_fraction=float(config.min_validation_valid_fraction),
    )
    stack_metrics["valid_for_selection"] = bool(stack_ok)
    if not stack_ok:
        raise RuntimeError(f"best checkpoint failed stack_val finite coverage: {stack_reason}")
    final_test = None
    if bool(config.evaluate_final_test):
        deployable_eval_model = LocalResidualFieldCurriculumJointModel.from_deployable_checkpoint(
            best_path,
            device=device,
        )
        final_test = _evaluate_final_test_deployable(deployable_eval_model, config, device=device)
    loaded_oracle_checkpoint = (
        None if model.oracle_consumer is None else str(model.oracle_consumer.config.checkpoint)
    )
    report = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_TRAIN_CONTRACT,
        "run_id": resolved.run_id,
        "output_dir": str(output_dir),
        "best_epoch": int(best_epoch),
        "best_model_val": _jsonable(best_metrics),
        "stack_val": _jsonable(stack_metrics),
        "final_test": _jsonable(final_test),
        "checkpoint": str(best_path),
        "checkpoint_hash": _sha256_file(best_path),
        "last_checkpoint": str(last_path) if bool(config.save_last_checkpoint) else None,
        "training_curves": str(output_dir / "training_curves.json"),
        "diagnostic_csv": str(diagnostics_dir / "curriculum_epoch_metrics.csv"),
        "source_metadata": str(output_dir / "source_metadata.json"),
        "curriculum_schedule": str(output_dir / "curriculum_schedule.json"),
        "resolved_run": resolved.to_dict(),
        "selected_consumer_id": resolved.selected_consumer_id,
        "selected_alpha_endpoint": resolved.selected_alpha_endpoint,
        "oracle_teacher_checkpoint": loaded_oracle_checkpoint,
        "oracle_teacher_checkpoint_hash": (
            None if model.oracle_consumer is None else str(model.oracle_consumer.checkpoint_hash)
        ),
        "oracle_teacher_logits_paths": oracle_paths,
        "offline_teacher_logits_paths": offline_paths,
        "teacher_used_during_training": (
            None if model.oracle_consumer is None else str(model.oracle_consumer.consumer_id)
        ),
        "oracle_teacher_loaded_during_training": bool(model.oracle_consumer is not None),
        "training_uses_true_fields": True,
        "training_uses_field_target_cache": True,
        "predictor_warm_start": predictor_warm_start_report,
        "student_initialization": _jsonable(model.student_initialization_report),
        "runtime_inputs": "HLT_only",
        "uses_true_fields": False,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "deployable": True,
        "selection_allowed": True,
        "oracle_diagnostics_separate": True,
        "scientific_recipe_equivalent": resolved.recipe_equivalent,
    }
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "LOCAL_RESIDUAL_FIELD_CURRICULUM_TRAIN_CONTRACT",
    "CURRICULUM_PILOT_RUN_IDS",
    "CURRICULUM_STAGE1B_RUN_IDS",
    "CURRICULUM_STUDENT_KD_SOURCES",
    "CURRICULUM_LOSS_NAMES",
    "PILOT_LOSS_WEIGHTS",
    "LocalResidualFieldCurriculumTrainConfig",
    "ResolvedCurriculumRun",
    "resolve_curriculum_run",
    "compute_curriculum_batch_loss",
    "train_local_residual_field_curriculum",
]

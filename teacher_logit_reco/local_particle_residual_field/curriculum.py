"""Joint curriculum model for residual-field distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import require_torch

from .model import (
    LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT,
    LocalResidualFieldReconstructorConfig,
    LocalResidualFieldReconstructorOutput,
    build_local_residual_field_reconstructor,
)
from .oracle import (
    LOCAL_RESIDUAL_FIELD_FROZEN_ORACLE_CONSUMER_CONTRACT,
    FrozenLocalResidualFieldOracleConsumer,
    FrozenOracleConsumerConfig,
    FrozenOracleConsumerOutput,
)
from .tagger import (
    LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
    RESIDUAL_FIELD_SOURCE_HLT_ONLY,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldTaggerConfig,
    warm_start_local_residual_field_tagger_part,
)

try:  # Keep metadata/report imports usable without a local torch install.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_RESIDUAL_FIELD_CURRICULUM_JOINT_CONTRACT = "local_residual_field_curriculum_joint_model_v1"
LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT = "local_residual_field_curriculum_deployable_v1"
LOCAL_RESIDUAL_FIELD_CURRICULUM_SCHEDULER_CONTRACT = "local_residual_field_curriculum_scheduler_v1"
LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT = "local_residual_field_selected_consumer_v1"

FIELD_GATE_MODE_NONE = "none"
FIELD_GATE_MODE_LEARNED_SIGMOID = "learned_sigmoid"
FIELD_GATE_MODE_SUPERVISED_RELIABILITY = "supervised_reliability"
FIELD_GATE_MODE_UNCERTAINTY_INVERSE = "uncertainty_inverse"
FIELD_GATE_MODES = (
    FIELD_GATE_MODE_NONE,
    FIELD_GATE_MODE_LEARNED_SIGMOID,
    FIELD_GATE_MODE_SUPERVISED_RELIABILITY,
    FIELD_GATE_MODE_UNCERTAINTY_INVERSE,
)

RESIDUAL_PROJECTION_RESET_NONE = "none"
RESIDUAL_PROJECTION_RESET_SCALE = "scale"
RESIDUAL_PROJECTION_RESET_RESET = "reset"
RESIDUAL_PROJECTION_RESET_MODES = (
    RESIDUAL_PROJECTION_RESET_NONE,
    RESIDUAL_PROJECTION_RESET_SCALE,
    RESIDUAL_PROJECTION_RESET_RESET,
)

FREEZE_PHASE_RESIDUAL_PATH_WARMUP = "residual_path_warmup"
FREEZE_PHASE_UPPER_UNFREEZE = "upper_unfreeze"
FREEZE_PHASE_FULL_GENTLE_UNFREEZE = "full_gentle_unfreeze"
FREEZE_PHASES = (
    FREEZE_PHASE_RESIDUAL_PATH_WARMUP,
    FREEZE_PHASE_UPPER_UNFREEZE,
    FREEZE_PHASE_FULL_GENTLE_UNFREEZE,
)

FREEZE_SCHEDULE_NONE = "none"
FREEZE_SCHEDULE_RESIDUAL_WARMUP_THEN_UPPER = "residual_path_warmup_then_upper_unfreeze"
FREEZE_SCHEDULES = (
    FREEZE_SCHEDULE_NONE,
    FREEZE_SCHEDULE_RESIDUAL_WARMUP_THEN_UPPER,
)

STUDENT_INIT_SOURCES = ("A0", "Ofull", "Orobust_light")

DEFAULT_OPTIMIZER_GROUP_LEARNING_RATES = {
    "predictor": 3.0e-4,
    "confidence_heads": 3.0e-4,
    "student_residual_projection": 3.0e-5,
    "student_head": 3.0e-5,
    "student_upper": 1.0e-5,
    "student_body": 3.0e-6,
}

ALPHA_SCHEDULE_FIXED = "fixed_alpha"
ALPHA_SCHEDULE_PIECEWISE = "piecewise_alpha"
ALPHA_SCHEDULE_SIGMOID = "sigmoid_alpha"
ALPHA_SCHEDULES = (
    ALPHA_SCHEDULE_FIXED,
    ALPHA_SCHEDULE_PIECEWISE,
    ALPHA_SCHEDULE_SIGMOID,
)

LOSS_WEIGHT_SCHEDULE_FIXED = "fixed"
LOSS_WEIGHT_SCHEDULE_LINEAR = "linear"
LOSS_WEIGHT_SCHEDULE_PIECEWISE = "piecewise"
LOSS_WEIGHT_SCHEDULE_SIGMOID = "sigmoid"
LOSS_WEIGHT_SCHEDULES = (
    LOSS_WEIGHT_SCHEDULE_FIXED,
    LOSS_WEIGHT_SCHEDULE_LINEAR,
    LOSS_WEIGHT_SCHEDULE_PIECEWISE,
    LOSS_WEIGHT_SCHEDULE_SIGMOID,
)


def normalize_field_gate_mode(value: str | None) -> str:
    key = str(value or FIELD_GATE_MODE_NONE).strip().lower().replace("-", "_")
    aliases = {
        "": FIELD_GATE_MODE_NONE,
        "off": FIELD_GATE_MODE_NONE,
        "none": FIELD_GATE_MODE_NONE,
        "identity": FIELD_GATE_MODE_NONE,
        "learned": FIELD_GATE_MODE_LEARNED_SIGMOID,
        "sigmoid": FIELD_GATE_MODE_LEARNED_SIGMOID,
        "learned_sigmoid": FIELD_GATE_MODE_LEARNED_SIGMOID,
        "field_gate": FIELD_GATE_MODE_LEARNED_SIGMOID,
        "supervised": FIELD_GATE_MODE_SUPERVISED_RELIABILITY,
        "supervised_reliability": FIELD_GATE_MODE_SUPERVISED_RELIABILITY,
        "reliability": FIELD_GATE_MODE_SUPERVISED_RELIABILITY,
        "uncertainty": FIELD_GATE_MODE_UNCERTAINTY_INVERSE,
        "uncertainty_inverse": FIELD_GATE_MODE_UNCERTAINTY_INVERSE,
        "inverse_uncertainty": FIELD_GATE_MODE_UNCERTAINTY_INVERSE,
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(f"field_gate_mode must be one of {FIELD_GATE_MODES}, got {value!r}")


def normalize_residual_projection_reset(value: str | None) -> str:
    key = str(value or RESIDUAL_PROJECTION_RESET_NONE).strip().lower().replace("-", "_")
    aliases = {
        "": RESIDUAL_PROJECTION_RESET_NONE,
        "none": RESIDUAL_PROJECTION_RESET_NONE,
        "off": RESIDUAL_PROJECTION_RESET_NONE,
        "keep": RESIDUAL_PROJECTION_RESET_NONE,
        "scale": RESIDUAL_PROJECTION_RESET_SCALE,
        "shrink": RESIDUAL_PROJECTION_RESET_SCALE,
        "reset": RESIDUAL_PROJECTION_RESET_RESET,
        "zero": RESIDUAL_PROJECTION_RESET_RESET,
        "zero_residual": RESIDUAL_PROJECTION_RESET_RESET,
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(
        f"residual_projection_reset must be one of {RESIDUAL_PROJECTION_RESET_MODES}, got {value!r}"
    )


def normalize_freeze_schedule(value: str | None) -> str:
    key = str(value or FREEZE_SCHEDULE_NONE).strip().lower().replace("-", "_")
    aliases = {
        "": FREEZE_SCHEDULE_NONE,
        "none": FREEZE_SCHEDULE_NONE,
        "off": FREEZE_SCHEDULE_NONE,
        "manual": FREEZE_SCHEDULE_NONE,
        "residual_path_warmup_then_upper_unfreeze": FREEZE_SCHEDULE_RESIDUAL_WARMUP_THEN_UPPER,
        "warmup_then_upper": FREEZE_SCHEDULE_RESIDUAL_WARMUP_THEN_UPPER,
        "p7b": FREEZE_SCHEDULE_RESIDUAL_WARMUP_THEN_UPPER,
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(f"freeze_schedule must be one of {FREEZE_SCHEDULES}, got {value!r}")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_alpha_schedule(value: str | None) -> str:
    key = str(value or ALPHA_SCHEDULE_FIXED).strip().lower().replace("-", "_")
    aliases = {
        "": ALPHA_SCHEDULE_FIXED,
        "fixed": ALPHA_SCHEDULE_FIXED,
        "fixed_alpha": ALPHA_SCHEDULE_FIXED,
        "constant": ALPHA_SCHEDULE_FIXED,
        "piecewise": ALPHA_SCHEDULE_PIECEWISE,
        "piecewise_alpha": ALPHA_SCHEDULE_PIECEWISE,
        "stair": ALPHA_SCHEDULE_PIECEWISE,
        "staircase": ALPHA_SCHEDULE_PIECEWISE,
        "sigmoid": ALPHA_SCHEDULE_SIGMOID,
        "sigmoid_alpha": ALPHA_SCHEDULE_SIGMOID,
        "logistic": ALPHA_SCHEDULE_SIGMOID,
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(f"alpha_schedule must be one of {ALPHA_SCHEDULES}, got {value!r}")


def normalize_loss_weight_schedule(value: str | None) -> str:
    key = str(value or LOSS_WEIGHT_SCHEDULE_FIXED).strip().lower().replace("-", "_")
    aliases = {
        "": LOSS_WEIGHT_SCHEDULE_FIXED,
        "fixed": LOSS_WEIGHT_SCHEDULE_FIXED,
        "constant": LOSS_WEIGHT_SCHEDULE_FIXED,
        "linear": LOSS_WEIGHT_SCHEDULE_LINEAR,
        "ramp": LOSS_WEIGHT_SCHEDULE_LINEAR,
        "piecewise": LOSS_WEIGHT_SCHEDULE_PIECEWISE,
        "stair": LOSS_WEIGHT_SCHEDULE_PIECEWISE,
        "sigmoid": LOSS_WEIGHT_SCHEDULE_SIGMOID,
        "logistic": LOSS_WEIGHT_SCHEDULE_SIGMOID,
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(f"loss weight schedule type must be one of {LOSS_WEIGHT_SCHEDULES}, got {value!r}")


def _config_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"cannot serialize config object of type {type(value).__name__}")


def _field_groups_to_dict(value: Mapping[str, Sequence[int]] | None) -> dict[str, list[int]]:
    return {
        str(group): [int(index) for index in indices]
        for group, indices in dict(value or {}).items()
    }


@dataclass(frozen=True)
class LocalResidualFieldCurriculumJointConfig:
    """Configuration for ``R_theta + S_phi + optional T_consumer``.

    The oracle consumer is a training-only teacher.  Deployment payload helpers
    below intentionally omit it so the saved model requires only HLT particles
    plus the learned residual-field predictor.
    """

    reconstructor_config: LocalResidualFieldReconstructorConfig | Mapping[str, Any] = field(
        default_factory=LocalResidualFieldReconstructorConfig
    )
    student_config: LocalResidualFieldTaggerConfig | Mapping[str, Any] = field(default_factory=LocalResidualFieldTaggerConfig)
    oracle_consumer_config: FrozenOracleConsumerConfig | Mapping[str, Any] | None = None
    field_gate_mode: str = FIELD_GATE_MODE_LEARNED_SIGMOID
    initial_gate_bias_prob: float = 0.1
    gate_reliability_loss_weight: float = 0.05
    gate_reliability_error_scale: float = 1.0
    gate_log_var_min: float = -8.0
    gate_log_var_max: float = 8.0
    student_init_source: str = "A0"
    student_init_checkpoint: str | None = None
    require_student_init_checkpoint: bool = False
    residual_projection_reset: str = RESIDUAL_PROJECTION_RESET_NONE
    residual_projection_scale: float = 1.0
    freeze_schedule: str = FREEZE_SCHEDULE_NONE
    freeze_phase1_epochs: int = 2
    freeze_phase2_epochs: int = 3
    optimizer_group_learning_rates: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_OPTIMIZER_GROUP_LEARNING_RATES)
    )
    field_names: Sequence[str] = field(default_factory=tuple)
    field_groups: Mapping[str, Sequence[int]] = field(default_factory=dict)
    normalization_metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance_hashes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reco = self.reconstructor_config
        if not isinstance(reco, LocalResidualFieldReconstructorConfig):
            payload = dict(reco)
            payload.pop("contract", None)
            reco = LocalResidualFieldReconstructorConfig(**payload)
            object.__setattr__(self, "reconstructor_config", reco)
        student = self.student_config
        if not isinstance(student, LocalResidualFieldTaggerConfig):
            payload = dict(student)
            payload.pop("contract", None)
            payload.pop("augmented_feature_dim", None)
            student = LocalResidualFieldTaggerConfig(**payload)
            object.__setattr__(self, "student_config", student)
        if student.field_source == RESIDUAL_FIELD_SOURCE_HLT_ONLY:
            raise ValueError("curriculum joint student must be an augmented residual-field tagger, not hlt_only")
        if int(reco.field_dim) != int(student.field_dim):
            raise ValueError("reconstructor and student residual field dimensions must match")
        gate_mode = normalize_field_gate_mode(self.field_gate_mode)
        object.__setattr__(self, "field_gate_mode", gate_mode)
        gate_prob = float(self.initial_gate_bias_prob)
        if not (0.0 < gate_prob <= 1.0):
            raise ValueError("initial_gate_bias_prob must be in (0, 1]")
        object.__setattr__(self, "initial_gate_bias_prob", gate_prob)
        gate_weight = float(self.gate_reliability_loss_weight)
        if gate_weight < 0.0 or not math.isfinite(gate_weight):
            raise ValueError("gate_reliability_loss_weight must be finite and non-negative")
        object.__setattr__(self, "gate_reliability_loss_weight", gate_weight)
        gate_scale = float(self.gate_reliability_error_scale)
        if gate_scale <= 0.0 or not math.isfinite(gate_scale):
            raise ValueError("gate_reliability_error_scale must be finite and positive")
        object.__setattr__(self, "gate_reliability_error_scale", gate_scale)
        log_var_min = float(self.gate_log_var_min)
        log_var_max = float(self.gate_log_var_max)
        if not math.isfinite(log_var_min) or not math.isfinite(log_var_max) or log_var_min >= log_var_max:
            raise ValueError("gate_log_var_min must be finite and less than gate_log_var_max")
        object.__setattr__(self, "gate_log_var_min", log_var_min)
        object.__setattr__(self, "gate_log_var_max", log_var_max)
        reset_mode = normalize_residual_projection_reset(self.residual_projection_reset)
        object.__setattr__(self, "residual_projection_reset", reset_mode)
        reset_scale = float(self.residual_projection_scale)
        if reset_scale < 0.0:
            raise ValueError("residual_projection_scale must be non-negative")
        object.__setattr__(self, "residual_projection_scale", reset_scale)
        oracle = self.oracle_consumer_config
        if oracle is not None and not isinstance(oracle, FrozenOracleConsumerConfig):
            payload = dict(oracle)
            payload.pop("contract", None)
            oracle = FrozenOracleConsumerConfig(**payload)
            object.__setattr__(self, "oracle_consumer_config", oracle)
        names = tuple(str(name) for name in (self.field_names or reco.field_names or student.field_names or ()))
        if names and len(names) != int(reco.field_dim):
            raise ValueError("field_names must match the reconstructor field_dim")
        object.__setattr__(self, "field_names", names)
        groups = _field_groups_to_dict(self.field_groups or reco.field_groups or student.field_groups)
        object.__setattr__(self, "field_groups", groups)
        student_init_source = str(self.student_init_source or "").strip()
        if student_init_source not in STUDENT_INIT_SOURCES:
            raise ValueError(f"student_init_source must be one of {STUDENT_INIT_SOURCES}")
        object.__setattr__(self, "student_init_source", student_init_source)
        init_checkpoint = None if not self.student_init_checkpoint else str(self.student_init_checkpoint)
        if bool(self.require_student_init_checkpoint) and init_checkpoint is None:
            raise ValueError("student_init_checkpoint is required by this curriculum recipe")
        object.__setattr__(self, "student_init_checkpoint", init_checkpoint)
        object.__setattr__(self, "require_student_init_checkpoint", bool(self.require_student_init_checkpoint))
        freeze_schedule = normalize_freeze_schedule(self.freeze_schedule)
        object.__setattr__(self, "freeze_schedule", freeze_schedule)
        if student_init_source in {"Ofull", "Orobust_light"}:
            if init_checkpoint is None:
                raise ValueError("oracle-initialized P7b requires student_init_checkpoint")
            if reset_mode == RESIDUAL_PROJECTION_RESET_NONE:
                raise ValueError("oracle-initialized P7b requires an explicit residual projection reset policy")
            if freeze_schedule != FREEZE_SCHEDULE_RESIDUAL_WARMUP_THEN_UPPER:
                raise ValueError(
                    "oracle-initialized P7b requires freeze_schedule="
                    f"{FREEZE_SCHEDULE_RESIDUAL_WARMUP_THEN_UPPER}"
                )
            if oracle is not None and oracle.consumer_id and str(oracle.consumer_id) != student_init_source:
                raise ValueError(
                    f"student_init_source {student_init_source!r} != oracle consumer {oracle.consumer_id!r}"
                )
        for name in ("freeze_phase1_epochs", "freeze_phase2_epochs"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        learning_rates = dict(DEFAULT_OPTIMIZER_GROUP_LEARNING_RATES)
        learning_rates.update({str(key): float(value) for key, value in dict(self.optimizer_group_learning_rates).items()})
        unknown_groups = sorted(set(learning_rates) - set(DEFAULT_OPTIMIZER_GROUP_LEARNING_RATES))
        if unknown_groups:
            raise ValueError(f"unknown optimizer groups: {unknown_groups}")
        for group_name, learning_rate in learning_rates.items():
            if not math.isfinite(learning_rate) or learning_rate <= 0.0:
                raise ValueError(f"optimizer learning rate for {group_name} must be finite and positive")
        object.__setattr__(self, "optimizer_group_learning_rates", learning_rates)
        object.__setattr__(self, "normalization_metadata", dict(self.normalization_metadata or {}))
        object.__setattr__(self, "provenance_hashes", dict(self.provenance_hashes or {}))

    def to_dict(self) -> dict[str, Any]:
        oracle = None
        if self.oracle_consumer_config is not None:
            oracle = _config_to_dict(self.oracle_consumer_config)
            oracle["contract"] = LOCAL_RESIDUAL_FIELD_FROZEN_ORACLE_CONSUMER_CONTRACT
        return {
            "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_JOINT_CONTRACT,
            "reconstructor_config": _config_to_dict(self.reconstructor_config),
            "student_config": _config_to_dict(self.student_config),
            "oracle_consumer_config": oracle,
            "field_gate_mode": str(self.field_gate_mode),
            "initial_gate_bias_prob": float(self.initial_gate_bias_prob),
            "gate_reliability_loss_weight": float(self.gate_reliability_loss_weight),
            "gate_reliability_error_scale": float(self.gate_reliability_error_scale),
            "gate_log_var_min": float(self.gate_log_var_min),
            "gate_log_var_max": float(self.gate_log_var_max),
            "student_init_source": str(self.student_init_source),
            "student_init_checkpoint": self.student_init_checkpoint,
            "require_student_init_checkpoint": bool(self.require_student_init_checkpoint),
            "residual_projection_reset": str(self.residual_projection_reset),
            "residual_projection_scale": float(self.residual_projection_scale),
            "freeze_schedule": str(self.freeze_schedule),
            "freeze_phase1_epochs": int(self.freeze_phase1_epochs),
            "freeze_phase2_epochs": int(self.freeze_phase2_epochs),
            "optimizer_group_learning_rates": dict(self.optimizer_group_learning_rates),
            "field_names": list(self.field_names),
            "field_groups": _field_groups_to_dict(self.field_groups),
            "normalization_metadata": dict(self.normalization_metadata),
            "provenance_hashes": dict(self.provenance_hashes),
        }


def _finite_float(value: Any, *, name: str, minimum: float | None = None) -> float:
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and output < float(minimum):
        raise ValueError(f"{name} must be >= {minimum}")
    return output


def _coerce_epoch_value_points(
    points: Any,
    *,
    value_key: str,
    allow_selected_endpoint: bool = False,
) -> tuple[tuple[int, Any], ...]:
    output: list[tuple[int, Any]] = []
    for item in points or ():
        if isinstance(item, Mapping):
            epoch = int(item.get("epoch", item.get("start_epoch", 0)))
            value = item.get(value_key, item.get("value"))
        else:
            epoch, value = item
            epoch = int(epoch)
        if epoch < 0:
            raise ValueError("schedule point epochs must be non-negative")
        if (
            bool(allow_selected_endpoint)
            and isinstance(value, str)
            and value.strip().lower().replace("-", "_") in {"selected", "endpoint", "selected_endpoint"}
        ):
            normalized_value: Any = "selected_endpoint"
        else:
            normalized_value = _finite_float(value, name=value_key, minimum=0.0)
        output.append((epoch, normalized_value))
    output.sort(key=lambda pair: pair[0])
    return tuple(output)


def _piecewise_value(
    points: Sequence[tuple[int, Any]],
    *,
    epoch: int,
    default: float,
    selected_endpoint: float | None = None,
) -> float:
    output = float(default)
    for start_epoch, value in points:
        if int(epoch) >= int(start_epoch):
            if isinstance(value, str) and value == "selected_endpoint":
                if selected_endpoint is None:
                    raise ValueError("piecewise schedule requested selected_endpoint, but no endpoint is configured")
                output = float(selected_endpoint)
            else:
                output = float(value)
        else:
            break
    return output


def _schedule_progress(epoch: int, total_epochs: int | None) -> float:
    if total_epochs is None or int(total_epochs) <= 1:
        return 0.0
    return max(0.0, min(1.0, float(epoch) / float(max(int(total_epochs) - 1, 1))))


def _sigmoid_value(
    *,
    epoch: int,
    total_epochs: int | None,
    start: float,
    end: float,
    midpoint: float,
    sharpness: float,
) -> float:
    progress = _schedule_progress(epoch, total_epochs)
    if float(midpoint) > 1.0:
        denom = float(max((total_epochs or int(math.ceil(midpoint))) - 1, 1))
        center = float(midpoint) / denom
    else:
        center = float(midpoint)
    x = float(sharpness) * (progress - center)
    sigma = 1.0 / (1.0 + math.exp(-max(min(x, 60.0), -60.0)))
    return float(start) + (float(end) - float(start)) * sigma


def _scheduled_scalar(
    spec: Any,
    *,
    epoch: int,
    total_epochs: int | None,
    default: float,
    value_name: str,
) -> float:
    if spec is None:
        return float(default)
    if isinstance(spec, (int, float)):
        return _finite_float(spec, name=value_name, minimum=0.0)
    if not isinstance(spec, Mapping):
        raise TypeError(f"{value_name} schedule must be a number or mapping")
    kind = normalize_loss_weight_schedule(str(spec.get("type", spec.get("schedule", LOSS_WEIGHT_SCHEDULE_FIXED))))
    if kind == LOSS_WEIGHT_SCHEDULE_FIXED:
        return _finite_float(spec.get("value", default), name=value_name, minimum=0.0)
    if kind == LOSS_WEIGHT_SCHEDULE_LINEAR:
        start = _finite_float(spec.get("start", default), name=f"{value_name}.start", minimum=0.0)
        end = _finite_float(spec.get("end", default), name=f"{value_name}.end", minimum=0.0)
        start_epoch = int(spec.get("start_epoch", 0))
        end_epoch = int(spec.get("end_epoch", max(int(total_epochs or 1) - 1, start_epoch + 1)))
        if end_epoch <= start_epoch:
            return end if int(epoch) >= end_epoch else start
        progress = max(0.0, min(1.0, float(int(epoch) - start_epoch) / float(end_epoch - start_epoch)))
        return start + (end - start) * progress
    if kind == LOSS_WEIGHT_SCHEDULE_PIECEWISE:
        points = _coerce_epoch_value_points(spec.get("points", ()), value_key="value")
        return _piecewise_value(points, epoch=int(epoch), default=float(default))
    if kind == LOSS_WEIGHT_SCHEDULE_SIGMOID:
        start = _finite_float(spec.get("start", default), name=f"{value_name}.start", minimum=0.0)
        end = _finite_float(spec.get("end", default), name=f"{value_name}.end", minimum=0.0)
        midpoint = _finite_float(spec.get("midpoint", 0.5), name=f"{value_name}.midpoint")
        sharpness = _finite_float(spec.get("sharpness", 12.0), name=f"{value_name}.sharpness", minimum=0.0)
        return _sigmoid_value(
            epoch=int(epoch),
            total_epochs=total_epochs,
            start=start,
            end=end,
            midpoint=midpoint,
            sharpness=sharpness,
        )
    raise AssertionError(f"unhandled loss weight schedule {kind!r}")


def _extract_selected_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("selected", "selection", "selected_consumer"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


@dataclass(frozen=True)
class SelectedConsumerRecord:
    """Consumer selected by Stage 1a for Stage 1b students."""

    selected_consumer_id: str
    selected_alpha_endpoint: float | None = None
    source_path: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        consumer_id = str(self.selected_consumer_id or "").strip()
        if not consumer_id:
            raise ValueError("selected_consumer_id is required")
        object.__setattr__(self, "selected_consumer_id", consumer_id)
        if self.selected_alpha_endpoint is not None:
            object.__setattr__(
                self,
                "selected_alpha_endpoint",
                _finite_float(self.selected_alpha_endpoint, name="selected_alpha_endpoint", minimum=0.0),
            )
        object.__setattr__(self, "source_path", None if self.source_path is None else str(self.source_path))
        object.__setattr__(self, "payload", dict(self.payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
            "selected_consumer_id": str(self.selected_consumer_id),
            "selected_alpha_endpoint": self.selected_alpha_endpoint,
            "source_path": self.source_path,
            "payload": dict(self.payload),
        }


def load_selected_consumer_record(path: str | Path) -> SelectedConsumerRecord:
    selected_path = Path(path)
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"selected consumer artifact {selected_path} is not a JSON object")
    selected = _extract_selected_mapping(payload)
    consumer_id = (
        selected.get("selected_consumer_id")
        or selected.get("consumer_id")
        or selected.get("consumer")
        or selected.get("selected_id")
    )
    endpoint = (
        selected.get("selected_alpha_endpoint")
        or selected.get("alpha_endpoint")
        or selected.get("selected_alpha")
        or selected.get("alpha")
    )
    return SelectedConsumerRecord(
        selected_consumer_id=str(consumer_id or ""),
        selected_alpha_endpoint=None if endpoint is None else float(endpoint),
        source_path=str(selected_path),
        payload=dict(payload),
    )


def paired_consumers_confirmed_from_env(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    raw = str(values.get("LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS", "")).strip().lower()
    return raw in {"1", "true", "yes", "y", "confirm", "confirmed"}


@dataclass(frozen=True)
class LocalResidualFieldCurriculumSchedulerConfig:
    """Epoch scheduler for Stage 1b deployable curriculum students."""

    alpha_schedule: str = ALPHA_SCHEDULE_FIXED
    fixed_alpha: float | None = None
    piecewise_alpha: Sequence[Any] = field(default_factory=tuple)
    sigmoid_alpha_start: float = 0.0
    sigmoid_alpha_end: float | None = None
    sigmoid_alpha_midpoint: float = 0.5
    sigmoid_alpha_sharpness: float = 12.0
    teacher_sequence: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    loss_weights: Mapping[str, float] = field(default_factory=lambda: {"ce": 1.0})
    loss_weight_schedule: Mapping[str, Any] = field(default_factory=dict)
    selected_consumer_id: str | None = None
    selected_alpha_endpoint: float | None = None
    selected_consumer_path: str | None = None
    selected_consumer_payload: Mapping[str, Any] = field(default_factory=dict)
    paired_consumer_mode: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha_schedule", normalize_alpha_schedule(self.alpha_schedule))
        if self.fixed_alpha is not None:
            object.__setattr__(self, "fixed_alpha", _finite_float(self.fixed_alpha, name="fixed_alpha", minimum=0.0))
        if self.selected_alpha_endpoint is not None:
            object.__setattr__(
                self,
                "selected_alpha_endpoint",
                _finite_float(self.selected_alpha_endpoint, name="selected_alpha_endpoint", minimum=0.0),
            )
        object.__setattr__(
            self,
            "sigmoid_alpha_start",
            _finite_float(self.sigmoid_alpha_start, name="sigmoid_alpha_start", minimum=0.0),
        )
        if self.sigmoid_alpha_end is not None:
            object.__setattr__(
                self,
                "sigmoid_alpha_end",
                _finite_float(self.sigmoid_alpha_end, name="sigmoid_alpha_end", minimum=0.0),
            )
        object.__setattr__(
            self,
            "sigmoid_alpha_midpoint",
            _finite_float(self.sigmoid_alpha_midpoint, name="sigmoid_alpha_midpoint"),
        )
        object.__setattr__(
            self,
            "sigmoid_alpha_sharpness",
            _finite_float(self.sigmoid_alpha_sharpness, name="sigmoid_alpha_sharpness", minimum=0.0),
        )
        points = _coerce_epoch_value_points(
            self.piecewise_alpha,
            value_key="alpha",
            allow_selected_endpoint=True,
        )
        object.__setattr__(self, "piecewise_alpha", points)
        weights = {
            str(name): _finite_float(value, name=f"loss_weights.{name}", minimum=0.0)
            for name, value in dict(self.loss_weights or {}).items()
        }
        object.__setattr__(self, "loss_weights", weights)
        object.__setattr__(self, "loss_weight_schedule", dict(self.loss_weight_schedule or {}))
        teacher_sequence: list[dict[str, Any]] = []
        for item in self.teacher_sequence or ():
            payload = dict(item)
            start_epoch = int(payload.get("epoch", payload.get("start_epoch", 0)))
            if start_epoch < 0:
                raise ValueError("teacher_sequence epochs must be non-negative")
            payload["epoch"] = start_epoch
            if "consumer_id" not in payload and "selected_consumer_id" in payload:
                payload["consumer_id"] = payload["selected_consumer_id"]
            if "consumer_id" not in payload:
                raise ValueError("teacher_sequence entries must include consumer_id")
            payload["consumer_id"] = str(payload["consumer_id"])
            teacher_sequence.append(payload)
        teacher_sequence.sort(key=lambda item: int(item["epoch"]))
        object.__setattr__(self, "teacher_sequence", tuple(teacher_sequence))
        consumer_id = None if self.selected_consumer_id is None else str(self.selected_consumer_id).strip()
        object.__setattr__(self, "selected_consumer_id", consumer_id or None)
        object.__setattr__(
            self,
            "selected_consumer_path",
            None if self.selected_consumer_path is None else str(self.selected_consumer_path),
        )
        object.__setattr__(self, "selected_consumer_payload", dict(self.selected_consumer_payload or {}))
        object.__setattr__(self, "paired_consumer_mode", bool(self.paired_consumer_mode))

    @classmethod
    def from_selected_consumer(
        cls,
        config: "LocalResidualFieldCurriculumSchedulerConfig | Mapping[str, Any] | None" = None,
        *,
        selected_consumer_path: str | Path | None = None,
        require_selected_consumer: bool = False,
        confirm_paired_consumers: bool | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "LocalResidualFieldCurriculumSchedulerConfig":
        payload = config.to_dict() if isinstance(config, cls) else dict(config or {})
        confirm = paired_consumers_confirmed_from_env(env) if confirm_paired_consumers is None else bool(confirm_paired_consumers)
        path = selected_consumer_path or payload.get("selected_consumer_path")
        if path:
            record = load_selected_consumer_record(path)
            payload["selected_consumer_id"] = record.selected_consumer_id
            payload["selected_alpha_endpoint"] = record.selected_alpha_endpoint
            payload["selected_consumer_path"] = record.source_path
            payload["selected_consumer_payload"] = record.payload
            payload["paired_consumer_mode"] = False
            if bool(require_selected_consumer) and record.selected_alpha_endpoint is None:
                raise ValueError("selected_consumer.json must include selected_alpha_endpoint for Stage 1b")
        elif bool(require_selected_consumer) and not bool(confirm):
            raise ValueError(
                "Stage 1b curriculum students require selected_consumer.json unless "
                "LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS=1 is set"
            )
        else:
            payload["paired_consumer_mode"] = bool(confirm)
        payload.pop("contract", None)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_SCHEDULER_CONTRACT,
            "alpha_schedule": str(self.alpha_schedule),
            "fixed_alpha": self.fixed_alpha,
            "piecewise_alpha": [
                {"epoch": int(epoch), "alpha": alpha if isinstance(alpha, str) else float(alpha)}
                for epoch, alpha in self.piecewise_alpha
            ],
            "sigmoid_alpha_start": float(self.sigmoid_alpha_start),
            "sigmoid_alpha_end": self.sigmoid_alpha_end,
            "sigmoid_alpha_midpoint": float(self.sigmoid_alpha_midpoint),
            "sigmoid_alpha_sharpness": float(self.sigmoid_alpha_sharpness),
            "teacher_sequence": [dict(item) for item in self.teacher_sequence],
            "loss_weights": dict(self.loss_weights),
            "loss_weight_schedule": dict(self.loss_weight_schedule),
            "selected_consumer_id": self.selected_consumer_id,
            "selected_alpha_endpoint": self.selected_alpha_endpoint,
            "selected_consumer_path": self.selected_consumer_path,
            "selected_consumer_payload": dict(self.selected_consumer_payload or {}),
            "paired_consumer_mode": bool(self.paired_consumer_mode),
        }


class LocalResidualFieldCurriculumScheduler:
    """Computes alpha, teacher, and loss weights for each training epoch."""

    def __init__(
        self,
        config: LocalResidualFieldCurriculumSchedulerConfig | Mapping[str, Any] | None = None,
        *,
        total_epochs: int | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, LocalResidualFieldCurriculumSchedulerConfig)
            else LocalResidualFieldCurriculumSchedulerConfig(**dict(config or {}))
        )
        self.total_epochs = None if total_epochs is None else int(total_epochs)
        if self.total_epochs is not None and self.total_epochs <= 0:
            raise ValueError("total_epochs must be positive when provided")

    def _alpha_endpoint(self) -> float:
        if self.config.selected_alpha_endpoint is not None:
            return float(self.config.selected_alpha_endpoint)
        if self.config.sigmoid_alpha_end is not None:
            return float(self.config.sigmoid_alpha_end)
        if self.config.fixed_alpha is not None:
            return float(self.config.fixed_alpha)
        if self.config.piecewise_alpha:
            return float(self.config.piecewise_alpha[-1][1])
        return 1.0

    def alpha_for_epoch(self, epoch: int, *, total_epochs: int | None = None) -> float:
        selected_total = self.total_epochs if total_epochs is None else int(total_epochs)
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        schedule = str(self.config.alpha_schedule)
        endpoint = self._alpha_endpoint()
        if schedule == ALPHA_SCHEDULE_FIXED:
            return float(self.config.fixed_alpha if self.config.fixed_alpha is not None else endpoint)
        if schedule == ALPHA_SCHEDULE_PIECEWISE:
            return _piecewise_value(
                self.config.piecewise_alpha,
                epoch=epoch,
                default=float(self.config.fixed_alpha if self.config.fixed_alpha is not None else 0.0),
                selected_endpoint=endpoint,
            )
        if schedule == ALPHA_SCHEDULE_SIGMOID:
            return _sigmoid_value(
                epoch=epoch,
                total_epochs=selected_total,
                start=float(self.config.sigmoid_alpha_start),
                end=float(self.config.sigmoid_alpha_end if self.config.sigmoid_alpha_end is not None else endpoint),
                midpoint=float(self.config.sigmoid_alpha_midpoint),
                sharpness=float(self.config.sigmoid_alpha_sharpness),
            )
        raise AssertionError(f"unhandled alpha schedule {schedule!r}")

    def teacher_for_epoch(self, epoch: int) -> dict[str, Any]:
        selected: dict[str, Any] = {}
        for item in self.config.teacher_sequence:
            start_epoch = int(item.get("epoch", item.get("start_epoch", 0)))
            if int(epoch) >= start_epoch:
                selected = dict(item)
            else:
                break
        consumer_id = (
            selected.get("consumer_id")
            or selected.get("selected_consumer_id")
            or self.config.selected_consumer_id
        )
        if consumer_id is not None:
            selected["consumer_id"] = str(consumer_id)
        selected.setdefault("selected_consumer_id", self.config.selected_consumer_id)
        selected.setdefault("selected_alpha_endpoint", self.config.selected_alpha_endpoint)
        selected.setdefault("alpha", self.alpha_for_epoch(epoch))
        return selected

    def loss_weights_for_epoch(self, epoch: int, *, total_epochs: int | None = None) -> dict[str, float]:
        selected_total = self.total_epochs if total_epochs is None else int(total_epochs)
        output = dict(self.config.loss_weights)
        for name, spec in dict(self.config.loss_weight_schedule).items():
            output[str(name)] = _scheduled_scalar(
                spec,
                epoch=int(epoch),
                total_epochs=selected_total,
                default=float(output.get(str(name), 0.0)),
                value_name=f"loss_weight.{name}",
            )
        return {str(name): float(value) for name, value in sorted(output.items())}

    def state_for_epoch(self, epoch: int, *, total_epochs: int | None = None) -> dict[str, Any]:
        alpha = self.alpha_for_epoch(epoch, total_epochs=total_epochs)
        teacher = self.teacher_for_epoch(epoch)
        return {
            "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_SCHEDULER_CONTRACT,
            "epoch": int(epoch),
            "alpha": float(alpha),
            "active_consumer_id": teacher.get("consumer_id"),
            "alpha_schedule": str(self.config.alpha_schedule),
            "loss_weight_schedule": dict(self.config.loss_weight_schedule),
            "loss_weights": self.loss_weights_for_epoch(epoch, total_epochs=total_epochs),
            "teacher": teacher,
            "selected_consumer_id": self.config.selected_consumer_id,
            "selected_alpha_endpoint": self.config.selected_alpha_endpoint,
            "selected_consumer_path": self.config.selected_consumer_path,
            "paired_consumer_mode": bool(self.config.paired_consumer_mode),
        }

    def epoch_report(self, num_epochs: int | None = None) -> list[dict[str, Any]]:
        count = self.total_epochs if num_epochs is None else int(num_epochs)
        if count is None:
            raise ValueError("num_epochs is required when scheduler.total_epochs is unset")
        if count <= 0:
            raise ValueError("num_epochs must be positive")
        return [self.state_for_epoch(epoch, total_epochs=count) for epoch in range(count)]

    def run_report_payload(self, num_epochs: int | None = None) -> dict[str, Any]:
        return {
            "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_SCHEDULER_CONTRACT,
            "config": self.config.to_dict(),
            "selected_consumer": dict(self.config.selected_consumer_payload or {}),
            "epochs": self.epoch_report(num_epochs),
        }

    def write_report(self, path: str | Path, *, num_epochs: int | None = None) -> dict[str, Any]:
        payload = self.run_report_payload(num_epochs)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload


@dataclass(frozen=True)
class LocalResidualFieldCurriculumJointOutput:
    """Forward output required by Step 4."""

    pred_fields_raw: Any
    pred_fields_effective: Any
    field_gate: Any
    field_uncertainty: Any | None
    student_logits: Any
    oracle_pred_logits: Any | None
    diagnostics: Mapping[str, Any]
    field_delta: Any | None = None
    field_log_var: Any | None = None
    field_gate_loss: Any | None = None
    field_reliability_target: Any | None = None
    student_output: Any | None = None
    reconstructor_output: LocalResidualFieldReconstructorOutput | None = None
    oracle_true_logits: Any | None = None
    oracle_output: FrozenOracleConsumerOutput | None = None


@dataclass(frozen=True)
class _ConfidenceHeadOutput:
    pred_fields_raw: Any
    field_delta: Any
    field_log_var: Any
    field_gate: Any
    field_uncertainty: Any


class _FieldGate(_ModuleBase):
    def __init__(self, *, field_dim: int, mode: str, initial_prob: float) -> None:
        torch = require_torch()
        super().__init__()
        self.field_dim = int(field_dim)
        self.mode = normalize_field_gate_mode(mode)
        if self.mode in {FIELD_GATE_MODE_LEARNED_SIGMOID, FIELD_GATE_MODE_SUPERVISED_RELIABILITY}:
            self.norm = torch.nn.LayerNorm(self.field_dim)
            self.proj = torch.nn.Linear(self.field_dim, self.field_dim)
            torch.nn.init.zeros_(self.proj.weight)
            bias = math.log(float(initial_prob) / max(1.0 - float(initial_prob), 1.0e-12))
            torch.nn.init.constant_(self.proj.bias, bias)
        else:
            self.norm = None
            self.proj = None

    def forward(self, fields: Any, mask: Any) -> Any:
        torch = require_torch()
        if self.mode == FIELD_GATE_MODE_NONE:
            gate = torch.ones_like(fields)
        elif self.mode == FIELD_GATE_MODE_UNCERTAINTY_INVERSE:
            gate = torch.ones_like(fields)
        else:
            gate = torch.sigmoid(self.proj(self.norm(fields)))
        return gate * mask.unsqueeze(-1).to(dtype=fields.dtype)


class _ConfidenceHeads(_ModuleBase):
    """Step 6 residual-field delta, uncertainty, and gate heads."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        field_dim: int,
        mode: str,
        initial_prob: float,
        log_var_min: float,
        log_var_max: float,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.field_dim = int(field_dim)
        self.mode = normalize_field_gate_mode(mode)
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)

        def head() -> Any:
            return torch.nn.Sequential(
                torch.nn.LayerNorm(self.hidden_dim),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.hidden_dim, self.field_dim),
            )

        self.field_delta_head = head()
        self.field_log_var_head = head()
        self.field_gate_head = head()
        for module in (self.field_delta_head, self.field_log_var_head, self.field_gate_head):
            last = module[-1]
            if hasattr(last, "weight"):
                torch.nn.init.zeros_(last.weight)
            if hasattr(last, "bias"):
                torch.nn.init.zeros_(last.bias)
        if hasattr(self.field_gate_head[-1], "bias"):
            bias = math.log(float(initial_prob) / max(1.0 - float(initial_prob), 1.0e-12))
            torch.nn.init.constant_(self.field_gate_head[-1].bias, bias)

    def forward(self, *, hidden: Any, base_fields: Any, mask: Any) -> _ConfidenceHeadOutput:
        torch = require_torch()
        valid = mask.unsqueeze(-1).to(dtype=base_fields.dtype)
        field_delta = self.field_delta_head(hidden).to(dtype=base_fields.dtype) * valid
        pred_fields = (base_fields + field_delta) * valid
        field_log_var = self.field_log_var_head(hidden).to(dtype=base_fields.dtype)
        field_log_var = field_log_var.clamp(min=float(self.log_var_min), max=float(self.log_var_max)) * valid
        if self.mode == FIELD_GATE_MODE_NONE:
            gate = torch.ones_like(pred_fields)
        elif self.mode == FIELD_GATE_MODE_UNCERTAINTY_INVERSE:
            gate = torch.exp(-torch.nn.functional.softplus(field_log_var))
        else:
            gate = torch.sigmoid(self.field_gate_head(hidden).to(dtype=base_fields.dtype))
        gate = gate * valid
        uncertainty = torch.exp(0.5 * field_log_var) * valid
        return _ConfidenceHeadOutput(
            pred_fields_raw=pred_fields,
            field_delta=field_delta,
            field_log_var=field_log_var,
            field_gate=gate,
            field_uncertainty=uncertainty,
        )


def confidence_reliability_target(
    *,
    pred_fields: Any,
    target_fields: Any,
    mask: Any,
    error_scale: float = 1.0,
) -> Any:
    """Soft reliability target: close field predictions should receive high gate values."""

    torch = require_torch()
    scale = max(float(error_scale), 1.0e-8)
    target = target_fields.to(device=pred_fields.device, dtype=pred_fields.dtype)
    error = (pred_fields - target).abs()
    reliability = torch.exp(-error / scale)
    reliability = torch.nan_to_num(reliability, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
    return reliability * mask.unsqueeze(-1).to(dtype=reliability.dtype)


def compute_confidence_gate_loss(
    *,
    gate: Any,
    pred_fields: Any,
    target_fields: Any | None,
    mask: Any,
    mode: str,
    loss_weight: float,
    error_scale: float,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    """Compute the light reliability loss used by Step 6 confidence gates."""

    torch = require_torch()
    gate_mode = normalize_field_gate_mode(mode)
    weight = float(loss_weight)
    diagnostics = {
        "gate_supervision_enabled": False,
        "gate_supervision_mode": gate_mode,
        "gate_reliability_loss_weight": weight,
    }
    if target_fields is None or weight <= 0.0 or gate_mode == FIELD_GATE_MODE_NONE:
        return None, None, diagnostics
    valid = mask.unsqueeze(-1).expand_as(gate)
    if not bool(valid.detach().any().cpu().item()):
        zero = gate.new_zeros(())
        diagnostics.update({"gate_supervision_enabled": True, "gate_reliability_loss": 0.0, "valid_gate_values": 0})
        return zero, torch.zeros_like(gate), diagnostics
    reliability = confidence_reliability_target(
        pred_fields=pred_fields,
        target_fields=target_fields,
        mask=mask,
        error_scale=float(error_scale),
    )
    gate_clamped = gate.clamp(min=1.0e-6, max=1.0 - 1.0e-6)
    bce = -(reliability * torch.log(gate_clamped) + (1.0 - reliability) * torch.log(1.0 - gate_clamped))
    loss = bce[valid].mean() * weight
    diagnostics.update(
        {
            "gate_supervision_enabled": True,
            "gate_reliability_loss": float(loss.detach().cpu().item()),
            "gate_reliability_target_mean": float(reliability.detach()[valid].mean().cpu().item()),
            "valid_gate_values": int(valid.detach().sum().cpu().item()),
        }
    )
    return loss, reliability, diagnostics


def _field_valid_mask(raw_mask: Any | None, fallback_mask: Any, fields: Any) -> Any:
    torch = require_torch()
    mask = raw_mask if raw_mask is not None else fallback_mask
    if mask.ndim == 3:
        mask = mask.squeeze(1)
    mask = mask.to(device=fields.device, dtype=torch.bool)
    if mask.shape != fields.shape[:2]:
        raise ValueError(f"field mask shape {tuple(mask.shape)} is incompatible with fields {tuple(fields.shape)}")
    return mask


def _is_head_parameter(name: str) -> bool:
    key = name.lower()
    return any(part in key for part in ("classifier", "class_head", "cls_head", "head", "fc", "output"))


def _is_upper_block_parameter(name: str) -> bool:
    key = name.lower()
    if _is_head_parameter(key):
        return True
    digits = [int(chunk) for chunk in key.replace("_", ".").split(".") if chunk.isdigit()]
    return bool(digits and max(digits) >= 6)


def reset_or_scale_student_residual_projection(
    student: LocalResidualFieldAugmentedParT,
    *,
    mode: str,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Zero or shrink residual-field input columns in a warm-started student.

    The ParticleTransformer implementation hides its exact input embedding
    layout behind ``weaver``.  This helper finds weight tensors whose input
    channel axis equals the augmented feature dimension and edits only the
    residual-field columns ``[base_feature_dim:augmented_feature_dim]``.
    """

    torch = require_torch()
    reset_mode = normalize_residual_projection_reset(mode)
    if reset_mode == RESIDUAL_PROJECTION_RESET_NONE:
        return {
            "mode": reset_mode,
            "scale": float(scale),
            "matched_parameter_names": [],
            "matched_parameter_count": 0,
            "edited_value_count": 0,
        }
    reset_scale = float(scale)
    if reset_scale < 0.0:
        raise ValueError("scale must be non-negative")
    base_dim = int(student.config.base_feature_dim)
    augmented_dim = int(student.config.augmented_feature_dim)
    if augmented_dim <= base_dim:
        return {
            "mode": reset_mode,
            "scale": reset_scale,
            "matched_parameter_names": [],
            "matched_parameter_count": 0,
            "edited_value_count": 0,
        }
    matched: list[str] = []
    edited = 0
    with torch.no_grad():
        for name, parameter in student.part_model.named_parameters():
            if parameter.ndim < 2:
                continue
            if int(parameter.shape[1]) != augmented_dim:
                continue
            slices = [slice(None)] * int(parameter.ndim)
            slices[1] = slice(base_dim, augmented_dim)
            view = parameter[tuple(slices)]
            if reset_mode == RESIDUAL_PROJECTION_RESET_RESET:
                view.zero_()
            elif reset_mode == RESIDUAL_PROJECTION_RESET_SCALE:
                view.mul_(reset_scale)
            matched.append(str(name))
            edited += int(view.numel())
    return {
        "mode": reset_mode,
        "scale": reset_scale,
        "matched_parameter_names": matched,
        "matched_parameter_count": len(matched),
        "edited_value_count": int(edited),
    }


class LocalResidualFieldCurriculumJointModel(_ModuleBase):
    """Deployable residual-field predictor plus augmented ParT student.

    ``oracle_consumer`` is optional and training-only.  It is intentionally not
    a registered submodule, and deployment payloads omit it.
    """

    def __init__(
        self,
        config: LocalResidualFieldCurriculumJointConfig | Mapping[str, Any] | None = None,
        *,
        reconstructor: Any | None = None,
        student: LocalResidualFieldAugmentedParT | None = None,
        oracle_consumer: FrozenLocalResidualFieldOracleConsumer | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.config = (
            config
            if isinstance(config, LocalResidualFieldCurriculumJointConfig)
            else LocalResidualFieldCurriculumJointConfig(**dict(config or {}))
        )
        self.reconstructor = reconstructor or build_local_residual_field_reconstructor(self.config.reconstructor_config)
        self.student = student or LocalResidualFieldAugmentedParT(self.config.student_config)
        if getattr(self.student, "reconstructor", None) is not None or getattr(
            self.student, "control_generator", None
        ) is not None:
            raise ValueError(
                "deployable curriculum student must consume explicit predicted fields and cannot own a second "
                "reconstructor or control generator"
            )
        self.student_initialization_report: dict[str, Any] = {
            "student_init_source": str(self.config.student_init_source),
            "student_init_checkpoint": self.config.student_init_checkpoint,
            "student_init_checkpoint_hash": None,
            "warm_start_applied": False,
            "student_supplied": bool(student is not None),
        }
        if self.config.student_init_checkpoint is not None:
            init_path = Path(self.config.student_init_checkpoint)
            if not init_path.is_file():
                raise FileNotFoundError(f"student initialization checkpoint does not exist: {init_path}")
            if self.config.student_init_source in {"Ofull", "Orobust_light"}:
                teacher_config_path = init_path.with_name("teacher_config.json")
                if not teacher_config_path.is_file():
                    raise FileNotFoundError(
                        f"oracle student initialization is missing teacher_config.json: {teacher_config_path}"
                    )
                teacher_config = json.loads(teacher_config_path.read_text(encoding="utf-8"))
                if not isinstance(teacher_config, Mapping):
                    raise ValueError("oracle student teacher_config.json must contain an object")
                artifact_teacher_id = str(teacher_config.get("teacher_id") or "")
                if artifact_teacher_id != self.config.student_init_source:
                    raise ValueError(
                        f"student initialization artifact teacher_id {artifact_teacher_id!r} != "
                        f"student_init_source {self.config.student_init_source!r}"
                    )
            warm_start_report = warm_start_local_residual_field_tagger_part(
                self.student,
                init_path,
                map_location="cpu",
                require=True,
            )
            self.student_initialization_report.update(
                {
                    "student_init_checkpoint_hash": _sha256_file(init_path),
                    "warm_start_applied": True,
                    "warm_start_report": warm_start_report,
                }
            )
        self.confidence_heads = _ConfidenceHeads(
            hidden_dim=int(self.config.reconstructor_config.d_model),
            field_dim=int(self.config.reconstructor_config.field_dim),
            mode=str(self.config.field_gate_mode),
            initial_prob=float(self.config.initial_gate_bias_prob),
            log_var_min=float(self.config.gate_log_var_min),
            log_var_max=float(self.config.gate_log_var_max),
        )
        if oracle_consumer is not None:
            self.oracle_consumer = oracle_consumer
        elif self.config.oracle_consumer_config is not None:
            self.oracle_consumer = FrozenLocalResidualFieldOracleConsumer(self.config.oracle_consumer_config)
        else:
            self.oracle_consumer = None
        self.residual_projection_reset_report = reset_or_scale_student_residual_projection(
            self.student,
            mode=str(self.config.residual_projection_reset),
            scale=float(self.config.residual_projection_scale),
        )
        if (
            self.config.residual_projection_reset != RESIDUAL_PROJECTION_RESET_NONE
            and int(self.residual_projection_reset_report["matched_parameter_count"]) == 0
        ):
            raise ValueError(
                "residual projection adaptation was requested but no augmented input projection was found"
            )
        self.current_freeze_phase: str | None = None

    def apply_freeze_phase(self, phase: str) -> dict[str, Any]:
        """Set trainability for the Step 4 phase schedule.

        Phase 1 trains the predictor/gate and a small student adaptation slice
        (input residual projection plus obvious classifier/head parameters).
        Phase 2 additionally releases upper student blocks by name heuristic.
        Phase 3 gently unfreezes the whole deployable model.
        """

        key = str(phase).strip().lower().replace("-", "_")
        aliases = {
            "phase1": FREEZE_PHASE_RESIDUAL_PATH_WARMUP,
            "phase_1": FREEZE_PHASE_RESIDUAL_PATH_WARMUP,
            "residual_warmup": FREEZE_PHASE_RESIDUAL_PATH_WARMUP,
            "residual_path_warmup": FREEZE_PHASE_RESIDUAL_PATH_WARMUP,
            "phase2": FREEZE_PHASE_UPPER_UNFREEZE,
            "phase_2": FREEZE_PHASE_UPPER_UNFREEZE,
            "upper": FREEZE_PHASE_UPPER_UNFREEZE,
            "upper_unfreeze": FREEZE_PHASE_UPPER_UNFREEZE,
            "phase3": FREEZE_PHASE_FULL_GENTLE_UNFREEZE,
            "phase_3": FREEZE_PHASE_FULL_GENTLE_UNFREEZE,
            "full": FREEZE_PHASE_FULL_GENTLE_UNFREEZE,
            "full_gentle_unfreeze": FREEZE_PHASE_FULL_GENTLE_UNFREEZE,
        }
        key = aliases.get(key, key)
        if key not in FREEZE_PHASES:
            raise ValueError(f"phase must be one of {FREEZE_PHASES}, got {phase!r}")

        for parameter in self.reconstructor.parameters():
            parameter.requires_grad_(True)
        for parameter in self.confidence_heads.parameters():
            parameter.requires_grad_(True)
        for parameter in self.student.parameters():
            parameter.requires_grad_(key == FREEZE_PHASE_FULL_GENTLE_UNFREEZE)

        if key in {FREEZE_PHASE_RESIDUAL_PATH_WARMUP, FREEZE_PHASE_UPPER_UNFREEZE}:
            base_dim = int(self.student.config.base_feature_dim)
            augmented_dim = int(self.student.config.augmented_feature_dim)
            for name, parameter in self.student.part_model.named_parameters():
                release = _is_head_parameter(name)
                if key == FREEZE_PHASE_UPPER_UNFREEZE:
                    release = release or _is_upper_block_parameter(name)
                if parameter.ndim >= 2 and int(parameter.shape[1]) == augmented_dim and augmented_dim > base_dim:
                    release = True
                parameter.requires_grad_(bool(release))

        self.current_freeze_phase = key
        return self.trainability_report(phase=key)

    def trainability_report(self, *, phase: str | None = None) -> dict[str, Any]:
        def count(module: Any) -> dict[str, int]:
            total = 0
            trainable = 0
            for parameter in module.parameters():
                values = int(parameter.numel())
                total += values
                if bool(parameter.requires_grad):
                    trainable += values
            return {"total": int(total), "trainable": int(trainable)}

        student_named = list(self.student.part_model.named_parameters())
        return {
            "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_JOINT_CONTRACT,
            "phase": None if phase is None else str(phase),
            "reconstructor": count(self.reconstructor),
            "confidence_heads": count(self.confidence_heads),
            "field_delta_head": count(self.confidence_heads.field_delta_head),
            "field_log_var_head": count(self.confidence_heads.field_log_var_head),
            "field_gate_head": count(self.confidence_heads.field_gate_head),
            "field_gate": count(self.confidence_heads.field_gate_head),
            "student": count(self.student),
            "student_head_trainable_names": [
                str(name) for name, parameter in student_named if bool(parameter.requires_grad) and _is_head_parameter(name)
            ],
            "student_upper_trainable_names": [
                str(name) for name, parameter in student_named if bool(parameter.requires_grad) and _is_upper_block_parameter(name)
            ],
            "residual_projection_reset": dict(self.residual_projection_reset_report),
            "student_initialization": dict(self.student_initialization_report),
            "freeze_schedule": self.freeze_schedule_report(),
        }

    def freeze_phase_for_epoch(self, epoch: int, *, validation_stable: bool = False) -> str:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self.config.freeze_schedule == FREEZE_SCHEDULE_NONE:
            return FREEZE_PHASE_FULL_GENTLE_UNFREEZE
        if epoch < int(self.config.freeze_phase1_epochs):
            return FREEZE_PHASE_RESIDUAL_PATH_WARMUP
        phase3_start = int(self.config.freeze_phase1_epochs) + int(self.config.freeze_phase2_epochs)
        if epoch >= phase3_start and bool(validation_stable):
            return FREEZE_PHASE_FULL_GENTLE_UNFREEZE
        return FREEZE_PHASE_UPPER_UNFREEZE

    def freeze_schedule_report(self) -> dict[str, Any]:
        return {
            "name": str(self.config.freeze_schedule),
            "phase1": {
                "phase": FREEZE_PHASE_RESIDUAL_PATH_WARMUP,
                "start_epoch": 0,
                "epochs": int(self.config.freeze_phase1_epochs),
            },
            "phase2": {
                "phase": FREEZE_PHASE_UPPER_UNFREEZE,
                "start_epoch": int(self.config.freeze_phase1_epochs),
                "epochs": int(self.config.freeze_phase2_epochs),
            },
            "phase3": {
                "phase": FREEZE_PHASE_FULL_GENTLE_UNFREEZE,
                "earliest_start_epoch": int(self.config.freeze_phase1_epochs)
                + int(self.config.freeze_phase2_epochs),
                "requires_validation_stable": True,
            },
        }

    def apply_freeze_schedule(self, epoch: int, *, validation_stable: bool = False) -> dict[str, Any]:
        phase = self.freeze_phase_for_epoch(epoch, validation_stable=validation_stable)
        report = self.apply_freeze_phase(phase)
        report["epoch"] = int(epoch)
        report["validation_stable"] = bool(validation_stable)
        report["optimizer_groups"] = self.optimizer_group_report()
        return report

    def optimizer_group_specs(self) -> list[dict[str, Any]]:
        """Return disjoint optimizer groups for the currently active phase."""

        learning_rates = dict(self.config.optimizer_group_learning_rates)
        groups: dict[str, list[Any]] = {name: [] for name in learning_rates}
        groups["predictor"] = [parameter for parameter in self.reconstructor.parameters() if parameter.requires_grad]
        groups["confidence_heads"] = [
            parameter for parameter in self.confidence_heads.parameters() if parameter.requires_grad
        ]
        base_dim = int(self.student.config.base_feature_dim)
        augmented_dim = int(self.student.config.augmented_feature_dim)
        claimed: set[int] = set()
        for name, parameter in self.student.part_model.named_parameters():
            if not bool(parameter.requires_grad):
                continue
            group_name = "student_body"
            if parameter.ndim >= 2 and int(parameter.shape[1]) == augmented_dim and augmented_dim > base_dim:
                group_name = "student_residual_projection"
            elif _is_head_parameter(name):
                group_name = "student_head"
            elif _is_upper_block_parameter(name):
                group_name = "student_upper"
            identifier = id(parameter)
            if identifier in claimed:
                raise AssertionError(f"student parameter {name!r} was assigned to more than one optimizer group")
            claimed.add(identifier)
            groups[group_name].append(parameter)
        specs = [
            {
                "name": name,
                "params": parameters,
                "lr": float(learning_rates[name]),
            }
            for name, parameters in groups.items()
            if parameters
        ]
        expected = {
            id(parameter)
            for module in (self.reconstructor, self.confidence_heads, self.student)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        assigned = {id(parameter) for group in specs for parameter in group["params"]}
        if assigned != expected:
            raise AssertionError(
                f"optimizer groups do not exactly cover trainable parameters: missing={len(expected - assigned)}, "
                f"extra={len(assigned - expected)}"
            )
        return specs

    def optimizer_group_report(self) -> list[dict[str, Any]]:
        return [
            {
                "name": str(group["name"]),
                "learning_rate": float(group["lr"]),
                "parameter_tensors": len(group["params"]),
                "parameter_count": sum(int(parameter.numel()) for parameter in group["params"]),
            }
            for group in self.optimizer_group_specs()
        ]

    def build_optimizer(self, *, weight_decay: float = 1.0e-4) -> tuple[Any, dict[str, Any]]:
        torch = require_torch()
        decay = float(weight_decay)
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        specs = self.optimizer_group_specs()
        if not specs:
            raise ValueError("no trainable parameters are active for the current freeze phase")
        optimizer = torch.optim.AdamW(specs, weight_decay=decay)
        report = {
            "phase": self.current_freeze_phase,
            "weight_decay": decay,
            "groups": self.optimizer_group_report(),
            "trainability": self.trainability_report(phase=self.current_freeze_phase),
            "residual_projection_reset": dict(self.residual_projection_reset_report),
        }
        return optimizer, report

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        *,
        tokens: Any,
        raw_mask: Any | None = None,
        indices: Any | None = None,
        target_fields: Any | None = None,
        oracle_alpha: float | None = None,
        cached_oracle_true_logits: Any | None = None,
        cached_oracle_true_logits_metadata: Mapping[str, Any] | None = None,
        return_outputs: bool = True,
    ) -> LocalResidualFieldCurriculumJointOutput | Any:
        torch = require_torch()
        reco_output = self.reconstructor(tokens.to(device=features.device, dtype=features.dtype), raw_mask)
        base_fields = reco_output.predicted_fields.to(device=features.device, dtype=features.dtype)
        valid_mask = _field_valid_mask(raw_mask, mask, base_fields)
        head_output = self.confidence_heads(
            hidden=reco_output.hidden.to(device=features.device, dtype=features.dtype),
            base_fields=base_fields,
            mask=valid_mask,
        )
        pred_raw = head_output.pred_fields_raw
        gate = head_output.field_gate
        pred_effective = pred_raw * gate
        pred_effective = torch.nan_to_num(pred_effective, nan=0.0, posinf=0.0, neginf=0.0)
        pred_effective = pred_effective * valid_mask.unsqueeze(-1).to(dtype=pred_effective.dtype)
        field_uncertainty = head_output.field_uncertainty
        field_gate_loss, field_reliability_target, gate_loss_diagnostics = compute_confidence_gate_loss(
            gate=gate,
            pred_fields=pred_raw,
            target_fields=target_fields,
            mask=valid_mask,
            mode=str(self.config.field_gate_mode),
            loss_weight=float(self.config.gate_reliability_loss_weight),
            error_scale=float(self.config.gate_reliability_error_scale),
        )

        student_output = self.student(
            points,
            features,
            lorentz_vectors,
            mask,
            tokens=tokens,
            raw_mask=raw_mask,
            indices=indices,
            residual_fields=pred_effective,
            return_outputs=True,
        )
        oracle_output = None
        oracle_true_logits = None
        oracle_pred_logits = None
        if self.oracle_consumer is not None:
            self.oracle_consumer.to(features.device)
            oracle_output = self.oracle_consumer(
                points=points,
                features=features,
                lorentz_vectors=lorentz_vectors,
                mask=mask,
                tokens=tokens,
                raw_mask=raw_mask,
                indices=indices,
                true_fields=target_fields,
                predicted_fields=pred_effective,
                cached_true_logits=cached_oracle_true_logits,
                cached_true_logits_metadata=cached_oracle_true_logits_metadata,
                alpha=oracle_alpha,
            )
            oracle_true_logits = oracle_output.teacher_logits_true
            oracle_pred_logits = oracle_output.teacher_logits_pred

        diagnostics = {
            "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_JOINT_CONTRACT,
            "reconstructor_contract": LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT,
            "student_contract": LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
            "field_gate_mode": str(self.config.field_gate_mode),
            "field_gate_mean": float(gate.detach()[valid_mask].mean().cpu().item()) if bool(valid_mask.detach().any().cpu().item()) else 0.0,
            "field_gate_min": float(gate.detach()[valid_mask].min().cpu().item()) if bool(valid_mask.detach().any().cpu().item()) else 0.0,
            "field_gate_max": float(gate.detach()[valid_mask].max().cpu().item()) if bool(valid_mask.detach().any().cpu().item()) else 0.0,
            "pred_fields_abs_mean": (
                float(pred_effective.detach()[valid_mask].abs().mean().cpu().item())
                if bool(valid_mask.detach().any().cpu().item())
                else 0.0
            ),
            "field_delta_abs_mean": (
                float(head_output.field_delta.detach()[valid_mask].abs().mean().cpu().item())
                if bool(valid_mask.detach().any().cpu().item())
                else 0.0
            ),
            "field_log_var_mean": (
                float(head_output.field_log_var.detach()[valid_mask].mean().cpu().item())
                if bool(valid_mask.detach().any().cpu().item())
                else 0.0
            ),
            "field_uncertainty_present": bool(field_uncertainty is not None),
            "oracle_consumer_present": bool(self.oracle_consumer is not None),
            "oracle_pred_logits_present": bool(oracle_pred_logits is not None),
            "oracle_true_logits_present": bool(oracle_true_logits is not None),
            "residual_projection_reset": dict(self.residual_projection_reset_report),
            "student_initialization": dict(self.student_initialization_report),
            "freeze_phase": self.current_freeze_phase,
            "freeze_schedule": self.freeze_schedule_report(),
            "confidence_heads": {
                "field_delta_head": True,
                "field_log_var_head": True,
                "field_gate_head": True,
                "gate_reliability_loss_weight": float(self.config.gate_reliability_loss_weight),
                "gate_reliability_error_scale": float(self.config.gate_reliability_error_scale),
            },
            "gate_supervision": gate_loss_diagnostics,
        }
        if oracle_output is not None:
            diagnostics["oracle_consumer"] = dict(oracle_output.diagnostics)
        if not bool(return_outputs):
            return student_output.logits
        return LocalResidualFieldCurriculumJointOutput(
            pred_fields_raw=pred_raw,
            pred_fields_effective=pred_effective,
            field_gate=gate,
            field_uncertainty=field_uncertainty,
            student_logits=student_output.logits,
            oracle_pred_logits=oracle_pred_logits,
            oracle_true_logits=oracle_true_logits,
            field_delta=head_output.field_delta,
            field_log_var=head_output.field_log_var,
            field_gate_loss=field_gate_loss,
            field_reliability_target=field_reliability_target,
            diagnostics=diagnostics,
            student_output=student_output,
            reconstructor_output=reco_output,
            oracle_output=oracle_output,
        )

    def deployable_checkpoint_payload(self, *, extra_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return an oracle-free checkpoint payload for inference."""

        deployable_config = self.config.to_dict()
        deployable_config["oracle_consumer_config"] = None
        deployable_config["oracle_consumer_included"] = False
        deployable_config["student_init_checkpoint"] = None
        deployable_config["require_student_init_checkpoint"] = False
        deployable_config["student_init_source"] = "A0"
        deployable_config["residual_projection_reset"] = RESIDUAL_PROJECTION_RESET_NONE
        deployable_config["residual_projection_scale"] = 1.0
        return {
            "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT,
            "model_contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_JOINT_CONTRACT,
            "oracle_consumer_included": False,
            "model_config": deployable_config,
            "reconstructor_state_dict": self.reconstructor.state_dict(),
            "student_state_dict": self.student.state_dict(),
            "confidence_heads_state_dict": self.confidence_heads.state_dict(),
            "normalization_metadata": dict(self.config.normalization_metadata),
            "field_names": list(self.config.field_names),
            "field_groups": _field_groups_to_dict(self.config.field_groups),
            "provenance_hashes": dict(self.config.provenance_hashes),
            "residual_projection_reset": dict(self.residual_projection_reset_report),
            "student_initialization": {
                "student_init_source": str(self.config.student_init_source),
                "student_init_checkpoint_hash": self.student_initialization_report.get(
                    "student_init_checkpoint_hash"
                ),
                "warm_start_applied": bool(self.student_initialization_report.get("warm_start_applied")),
            },
            "freeze_schedule": self.freeze_schedule_report(),
            "metadata": dict(extra_metadata or {}),
        }

    def save_deployable_checkpoint(
        self,
        path: str | Path,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        torch = require_torch()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.deployable_checkpoint_payload(extra_metadata=extra_metadata)
        torch.save(payload, output_path)
        return {
            "path": str(output_path),
            "checkpoint_hash": _sha256_file(output_path),
            "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT,
            "oracle_consumer_included": False,
        }

    @classmethod
    def from_deployable_checkpoint(
        cls,
        checkpoint: str | Path | Mapping[str, Any],
        *,
        device: Any = "cpu",
        reconstructor: Any | None = None,
        student: LocalResidualFieldAugmentedParT | None = None,
    ) -> "LocalResidualFieldCurriculumJointModel":
        torch = require_torch()
        payload = (
            torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, (str, Path))
            else dict(checkpoint)
        )
        if not isinstance(payload, Mapping):
            raise ValueError("deployable curriculum checkpoint must contain a mapping")
        if payload.get("contract") != LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT:
            raise ValueError("checkpoint is not a deployable local residual-field curriculum artifact")
        if payload.get("oracle_consumer_included") is not False:
            raise ValueError("deployable curriculum checkpoint must explicitly exclude the oracle consumer")
        model_config = payload.get("model_config")
        if not isinstance(model_config, Mapping):
            raise ValueError("deployable curriculum checkpoint is missing model_config")
        clean_config = dict(model_config)
        clean_config.pop("contract", None)
        clean_config.pop("oracle_consumer_included", None)
        if clean_config.get("oracle_consumer_config") is not None:
            raise ValueError("deployable curriculum model_config contains an oracle consumer")
        clean_config["oracle_consumer_config"] = None
        clean_config["student_init_checkpoint"] = None
        clean_config["require_student_init_checkpoint"] = False
        clean_config["student_init_source"] = "A0"
        clean_config["residual_projection_reset"] = RESIDUAL_PROJECTION_RESET_NONE
        clean_config["residual_projection_scale"] = 1.0
        model = cls(
            LocalResidualFieldCurriculumJointConfig(**clean_config),
            reconstructor=reconstructor,
            student=student,
            oracle_consumer=None,
        )
        for name, module in (
            ("reconstructor_state_dict", model.reconstructor),
            ("student_state_dict", model.student),
            ("confidence_heads_state_dict", model.confidence_heads),
        ):
            state = payload.get(name)
            if not isinstance(state, Mapping):
                raise ValueError(f"deployable curriculum checkpoint is missing {name}")
            module.load_state_dict(state, strict=True)
        model.residual_projection_reset_report = dict(payload.get("residual_projection_reset") or {})
        model.student_initialization_report = dict(payload.get("student_initialization") or {})
        model.oracle_consumer = None
        model.to(device)
        model.eval()
        return model


__all__ = [
    "LOCAL_RESIDUAL_FIELD_CURRICULUM_JOINT_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_CURRICULUM_SCHEDULER_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT",
    "FIELD_GATE_MODE_NONE",
    "FIELD_GATE_MODE_LEARNED_SIGMOID",
    "FIELD_GATE_MODE_SUPERVISED_RELIABILITY",
    "FIELD_GATE_MODE_UNCERTAINTY_INVERSE",
    "FIELD_GATE_MODES",
    "ALPHA_SCHEDULE_FIXED",
    "ALPHA_SCHEDULE_PIECEWISE",
    "ALPHA_SCHEDULE_SIGMOID",
    "ALPHA_SCHEDULES",
    "LOSS_WEIGHT_SCHEDULE_FIXED",
    "LOSS_WEIGHT_SCHEDULE_LINEAR",
    "LOSS_WEIGHT_SCHEDULE_PIECEWISE",
    "LOSS_WEIGHT_SCHEDULE_SIGMOID",
    "LOSS_WEIGHT_SCHEDULES",
    "RESIDUAL_PROJECTION_RESET_NONE",
    "RESIDUAL_PROJECTION_RESET_SCALE",
    "RESIDUAL_PROJECTION_RESET_RESET",
    "RESIDUAL_PROJECTION_RESET_MODES",
    "FREEZE_PHASE_RESIDUAL_PATH_WARMUP",
    "FREEZE_PHASE_UPPER_UNFREEZE",
    "FREEZE_PHASE_FULL_GENTLE_UNFREEZE",
    "FREEZE_PHASES",
    "FREEZE_SCHEDULE_NONE",
    "FREEZE_SCHEDULE_RESIDUAL_WARMUP_THEN_UPPER",
    "FREEZE_SCHEDULES",
    "STUDENT_INIT_SOURCES",
    "DEFAULT_OPTIMIZER_GROUP_LEARNING_RATES",
    "SelectedConsumerRecord",
    "LocalResidualFieldCurriculumSchedulerConfig",
    "LocalResidualFieldCurriculumScheduler",
    "LocalResidualFieldCurriculumJointConfig",
    "LocalResidualFieldCurriculumJointModel",
    "LocalResidualFieldCurriculumJointOutput",
    "load_selected_consumer_record",
    "paired_consumers_confirmed_from_env",
    "normalize_alpha_schedule",
    "confidence_reliability_target",
    "compute_confidence_gate_loss",
    "normalize_field_gate_mode",
    "normalize_freeze_schedule",
    "normalize_loss_weight_schedule",
    "normalize_residual_projection_reset",
    "reset_or_scale_student_residual_projection",
]

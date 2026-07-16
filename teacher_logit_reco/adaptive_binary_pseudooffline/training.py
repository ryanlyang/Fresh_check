"""Update-count curriculum training for adaptive binary pseudo-offline models.

This module owns the scientific training contract rather than the model's
forward graph.  A concrete reconstructor supplies named modules and a step
function; the trainer supplies the locked phase schedule, objective contract,
optimizer state, EMA selection, nonfinite handling, and restart semantics.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, set_training_seed

from .hypothesis_distribution import conditional_distribution_weight
from .targets import ABPH_LEVEL_CAPACITIES


ABPH_RECONSTRUCTOR_TRAINING_CONTRACT = (
    "adaptive_binary_pseudooffline_reconstructor_training_v1"
)
ABPH_RECONSTRUCTION_LOSS_NAMES: tuple[str, ...] = (
    "root",
    "group_2",
    "group_4",
    "group_8",
    "group_16",
    "group_32",
    "topology",
    "frontier",
    "particle",
    "particle_feature",
    "distribution",
    "calibration",
    "auxiliary",
)
ABPH_RECONSTRUCTOR_MODULE_GROUPS: tuple[str, ...] = (
    "hlt_encoder",
    "root",
    "hierarchy_2",
    "hierarchy_4",
    "hierarchy_8",
    "hierarchy_16",
    "hierarchy_32",
    "renderer",
    "distribution",
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReconstructionLossWeights:
    """Locked normalized objective-family weights from Section 13."""

    root: float = 1.0
    group_2: float = 1.0
    group_4: float = 1.0
    group_8: float = 0.75
    group_16: float = 0.50
    group_32: float = 0.50
    topology: float = 0.50
    frontier: float = 1.0
    particle: float = 1.0
    particle_feature: float = 0.50
    distribution: float = 0.25
    calibration: float = 0.10
    auxiliary: float = 0.25

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"required reconstruction weight {name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
            **asdict(self),
        }


@dataclass(frozen=True)
class ReconstructorCurriculumConfig:
    """Maximum update budgets and the locked progressive-depth schedule."""

    root_updates: int = 150_000
    hierarchy_updates_per_depth: int = 80_000
    renderer_updates: int = 200_000
    distribution_updates: int = 200_000
    evaluation_interval: int = 2_000
    root_patience_evaluations: int = 12
    hierarchy_patience_evaluations: int = 12
    renderer_patience_evaluations: int = 15
    distribution_patience_evaluations: int = 15
    renderer_true_parent_fraction: float = 0.25
    renderer_transition_fraction: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "root_updates",
            "hierarchy_updates_per_depth",
            "renderer_updates",
            "distribution_updates",
            "evaluation_interval",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "root_patience_evaluations",
            "hierarchy_patience_evaluations",
            "renderer_patience_evaluations",
            "distribution_patience_evaluations",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least one")
        first = float(self.renderer_true_parent_fraction)
        transition = float(self.renderer_transition_fraction)
        if first < 0.0 or transition < 0.0 or first + transition > 1.0:
            raise ValueError("renderer teacher-forcing fractions are invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizerGroupPolicy:
    peak_lr: float
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.peak_lr)) or float(self.peak_lr) <= 0.0:
            raise ValueError("optimizer peak_lr must be positive")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0.0:
            raise ValueError("optimizer weight_decay must be nonnegative")


def default_optimizer_group_policies() -> dict[str, OptimizerGroupPolicy]:
    new_module = OptimizerGroupPolicy(peak_lr=3.0e-4)
    return {
        "hlt_encoder": OptimizerGroupPolicy(peak_lr=3.0e-5),
        "root": new_module,
        **{
            f"hierarchy_{capacity}": OptimizerGroupPolicy(peak_lr=3.0e-4)
            for capacity in ABPH_LEVEL_CAPACITIES
        },
        "renderer": OptimizerGroupPolicy(peak_lr=3.0e-4),
        "distribution": OptimizerGroupPolicy(peak_lr=3.0e-4),
    }


@dataclass(frozen=True)
class ReconstructorTrainerConfig:
    output_dir: str
    seed: int = 24731
    device: str = "auto"
    amp: bool = True
    amp_dtype: str = "bfloat16"
    gradient_accumulation_steps: int = 1
    distributed_world_size: int = 1
    root_hierarchy_effective_batch_size: int = 1024
    renderer_distribution_effective_batch_size: int = 512
    enforce_effective_batch_size: bool = True
    gradient_clip_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    warmup_fraction: float = 0.05
    cosine_minimum_fraction: float = 0.05
    ema_decay: float = 0.9999
    maximum_nonfinite_updates: int = 8
    save_last_checkpoint: bool = True
    curriculum: ReconstructorCurriculumConfig = ReconstructorCurriculumConfig()
    loss_weights: ReconstructionLossWeights = ReconstructionLossWeights()

    def __post_init__(self) -> None:
        if not str(self.output_dir).strip():
            raise ValueError("output_dir is required")
        if int(self.gradient_accumulation_steps) <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        for name in (
            "distributed_world_size",
            "root_hierarchy_effective_batch_size",
            "renderer_distribution_effective_batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if float(self.gradient_clip_norm) <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 < float(self.warmup_fraction) < 1.0:
            raise ValueError("warmup_fraction must lie in (0, 1)")
        if not 0.0 < float(self.cosine_minimum_fraction) <= 1.0:
            raise ValueError("cosine_minimum_fraction must lie in (0, 1]")
        if not 0.0 <= float(self.ema_decay) < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if int(self.maximum_nonfinite_updates) < 0:
            raise ValueError("maximum_nonfinite_updates must be nonnegative")
        if str(self.amp_dtype) not in {"bfloat16", "float16"}:
            raise ValueError("amp_dtype must be bfloat16 or float16")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = ABPH_RECONSTRUCTOR_TRAINING_CONTRACT
        payload["config_hash"] = _canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class CurriculumStage:
    key: str
    phase: int
    phase_name: str
    maximum_updates: int
    patience_evaluations: int
    active_capacity: int


@dataclass(frozen=True)
class CurriculumState:
    stage_index: int
    stage_key: str
    phase: int
    phase_name: str
    global_update: int
    stage_update: int
    stage_maximum_updates: int
    active_capacity: int
    stage_progress: float
    teacher_forcing_probability: float
    distribution_weight: float
    complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CurriculumController:
    """Serializable state machine for phases 1-4 and all five depths."""

    def __init__(self, config: ReconstructorCurriculumConfig) -> None:
        self.config = config
        self.stages = self._build_stages(config)
        self.stage_index = 0
        self.stage_update = 0
        self.global_update = 0

    @staticmethod
    def _build_stages(config: ReconstructorCurriculumConfig) -> tuple[CurriculumStage, ...]:
        return (
            CurriculumStage(
                "phase1_root",
                1,
                "root_pretraining",
                int(config.root_updates),
                int(config.root_patience_evaluations),
                1,
            ),
            *tuple(
                CurriculumStage(
                    f"phase2_hierarchy_{capacity}",
                    2,
                    "progressive_hierarchy",
                    int(config.hierarchy_updates_per_depth),
                    int(config.hierarchy_patience_evaluations),
                    int(capacity),
                )
                for capacity in ABPH_LEVEL_CAPACITIES
            ),
            CurriculumStage(
                "phase3_renderer",
                3,
                "deterministic_particle_rendering",
                int(config.renderer_updates),
                int(config.renderer_patience_evaluations),
                32,
            ),
            CurriculumStage(
                "phase4_distribution",
                4,
                "probabilistic_multi_hypothesis",
                int(config.distribution_updates),
                int(config.distribution_patience_evaluations),
                32,
            ),
        )

    @property
    def complete(self) -> bool:
        return self.stage_index >= len(self.stages)

    def _progress(self, stage: CurriculumStage) -> float:
        if int(stage.maximum_updates) <= 1:
            return 1.0
        return min(max(self.stage_update / float(stage.maximum_updates - 1), 0.0), 1.0)

    def _teacher_forcing(self, stage: CurriculumStage, progress: float) -> float:
        if stage.phase == 1:
            return 1.0
        if stage.phase == 2:
            if progress < 0.20:
                return 1.0
            if progress < 0.70:
                return 1.0 - 0.75 * ((progress - 0.20) / 0.50)
            return 0.25
        if stage.phase == 3:
            first = float(self.config.renderer_true_parent_fraction)
            transition = float(self.config.renderer_transition_fraction)
            if progress < first:
                return 1.0
            if transition > 0.0 and progress < first + transition:
                return 1.0 - (progress - first) / transition
            return 0.0
        return 0.0

    def state(self) -> CurriculumState:
        if self.complete:
            return CurriculumState(
                stage_index=len(self.stages),
                stage_key="complete",
                phase=4,
                phase_name="complete",
                global_update=int(self.global_update),
                stage_update=0,
                stage_maximum_updates=0,
                active_capacity=32,
                stage_progress=1.0,
                teacher_forcing_probability=0.0,
                distribution_weight=0.25,
                complete=True,
            )
        stage = self.stages[self.stage_index]
        progress = self._progress(stage)
        return CurriculumState(
            stage_index=int(self.stage_index),
            stage_key=stage.key,
            phase=int(stage.phase),
            phase_name=stage.phase_name,
            global_update=int(self.global_update),
            stage_update=int(self.stage_update),
            stage_maximum_updates=int(stage.maximum_updates),
            active_capacity=int(stage.active_capacity),
            stage_progress=float(progress),
            teacher_forcing_probability=float(self._teacher_forcing(stage, progress)),
            distribution_weight=(
                float(conditional_distribution_weight(progress))
                if stage.phase == 4
                else 0.0
            ),
            complete=False,
        )

    def advance(self) -> bool:
        if self.complete:
            raise RuntimeError("cannot advance a completed curriculum")
        stage = self.stages[self.stage_index]
        self.global_update += 1
        self.stage_update += 1
        transitioned = self.stage_update >= int(stage.maximum_updates)
        if transitioned:
            self.stage_index += 1
            self.stage_update = 0
        return transitioned

    def finish_stage_early(self) -> None:
        if self.complete:
            raise RuntimeError("cannot finish a completed curriculum")
        self.stage_index += 1
        self.stage_update = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
            "config": self.config.to_dict(),
            "stage_index": int(self.stage_index),
            "stage_update": int(self.stage_update),
            "global_update": int(self.global_update),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("contract") != ABPH_RECONSTRUCTOR_TRAINING_CONTRACT:
            raise ValueError("curriculum checkpoint contract mismatch")
        if dict(payload.get("config", {})) != self.config.to_dict():
            raise ValueError("curriculum checkpoint configuration mismatch")
        stage_index = int(payload["stage_index"])
        stage_update = int(payload["stage_update"])
        global_update = int(payload["global_update"])
        if not 0 <= stage_index <= len(self.stages):
            raise ValueError("checkpoint stage index is out of range")
        if stage_index < len(self.stages):
            if not 0 <= stage_update < self.stages[stage_index].maximum_updates:
                raise ValueError("checkpoint stage update is out of range")
        elif stage_update != 0:
            raise ValueError("completed curriculum must have zero stage update")
        if global_update < 0:
            raise ValueError("checkpoint global update is negative")
        self.stage_index = stage_index
        self.stage_update = stage_update
        self.global_update = global_update


@dataclass(frozen=True)
class ReconstructorStepContext:
    curriculum: CurriculumState
    split: str
    mode: str
    validation: bool
    teacher_forcing_probability: float

    def __post_init__(self) -> None:
        if self.split not in {"model_train", "model_val"}:
            raise ValueError("reconstructor curriculum permits model_train/model_val only")
        if self.mode not in {"teacher_forced", "rollout"}:
            raise ValueError("step mode must be teacher_forced or rollout")
        if self.validation and (self.split != "model_val" or self.mode != "rollout"):
            raise ValueError("model validation must be zero-teacher-forcing rollout")


@dataclass(frozen=True)
class ReconstructorStepResult:
    loss_terms: Mapping[str, Any]
    metrics: Mapping[str, Any]
    batch_size: int
    tensors_to_check: tuple[Any, ...] = ()
    hard_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("step result batch_size must be positive")


@dataclass(frozen=True)
class ComposedReconstructionLoss:
    total: Any
    raw_terms: Mapping[str, Any]
    effective_weights: Mapping[str, float]
    weighted_terms: Mapping[str, Any]
    required_terms: tuple[str, ...]


def assemble_reconstruction_loss_terms(
    *,
    root_loss: Any,
    hierarchy_supervision: Sequence[Any] = (),
    rollout_alignment: Any | None = None,
    particle_matching: Any | None = None,
    particle_auxiliary: Any | None = None,
    distribution_loss: Any | None = None,
) -> dict[str, Any]:
    """Map the typed Step 3-7 outputs onto the locked Section 13 families.

    Per-level accounting totals contain topology NLL internally.  It is
    removed from each group term and averaged into the distinct topology
    family so the top-level weights do not silently double count it.  The
    same separation is applied to calibration inside the variational loss.
    """

    torch = require_torch()
    if root_loss is None or not hasattr(root_loss, "total"):
        raise TypeError("root_loss must be a RootLossOutput-like object")
    if len(hierarchy_supervision) > len(ABPH_LEVEL_CAPACITIES):
        raise ValueError("hierarchy supervision exceeds the five declared depths")
    terms: dict[str, Any] = {"root": root_loss.total}
    topology_terms = []
    for capacity, supervision in zip(ABPH_LEVEL_CAPACITIES, hierarchy_supervision):
        if not hasattr(supervision, "accounting_loss") or not hasattr(
            supervision, "total_loss"
        ):
            raise TypeError("hierarchy supervision has the wrong contract")
        accounting = supervision.accounting_loss
        if "topology_nll" not in accounting.components:
            raise KeyError("hierarchy accounting loss lacks topology_nll")
        topology = accounting.components["topology_nll"]
        topology_weight = float(accounting.weights.topology_nll)
        group = supervision.total_loss - topology_weight * topology
        if not bool(torch.isfinite(group)):
            raise FloatingPointError(f"group_{capacity} loss is nonfinite")
        terms[f"group_{capacity}"] = group
        topology_terms.append(topology)
    if topology_terms:
        terms["topology"] = torch.stack(tuple(topology_terms)).mean()
    if rollout_alignment is not None:
        if getattr(rollout_alignment, "mode", None) != "rollout":
            raise ValueError("frontier objective must come from rollout alignment")
        terms["frontier"] = rollout_alignment.total_frontier_loss
    if particle_matching is not None:
        terms["particle"] = particle_matching.total
        components = tuple(particle_matching.component_losses.values())
        if not components:
            raise ValueError("particle matching exposes no feature components")
        terms["particle_feature"] = torch.stack(components).mean()
    if particle_auxiliary is not None:
        terms["auxiliary"] = particle_auxiliary.total
    if distribution_loss is not None:
        calibration_weight = 0.10
        metrics = getattr(distribution_loss, "metrics", {})
        declared_weights = metrics.get("weights", {}) if isinstance(metrics, Mapping) else {}
        if "calibration" in declared_weights:
            calibration_weight = float(declared_weights["calibration"])
        terms["distribution"] = (
            distribution_loss.total
            - calibration_weight * distribution_loss.calibration_loss
        )
        terms["calibration"] = distribution_loss.calibration_loss
    return terms


def active_reconstruction_loss_names(context: ReconstructorStepContext) -> tuple[str, ...]:
    state = context.curriculum
    names = ["root"]
    if state.phase >= 2:
        names.extend(
            f"group_{capacity}"
            for capacity in ABPH_LEVEL_CAPACITIES
            if int(capacity) <= int(state.active_capacity)
        )
        names.append("topology")
        if context.mode == "rollout":
            names.append("frontier")
    if state.phase >= 3:
        names.extend(("particle", "particle_feature", "auxiliary"))
    if state.phase >= 4:
        names.extend(("distribution", "calibration"))
    return tuple(names)


def compose_reconstruction_loss(
    result: ReconstructorStepResult,
    context: ReconstructorStepContext,
    weights: ReconstructionLossWeights | None = None,
) -> ComposedReconstructionLoss:
    """Fail closed on missing, unknown, non-scalar, or nonfinite objective terms."""

    torch = require_torch()
    resolved = weights or ReconstructionLossWeights()
    unknown = sorted(set(result.loss_terms) - set(ABPH_RECONSTRUCTION_LOSS_NAMES))
    if unknown:
        raise KeyError(f"unknown reconstruction loss terms: {unknown}")
    required = active_reconstruction_loss_names(context)
    missing = [name for name in required if name not in result.loss_terms]
    if missing:
        raise KeyError(f"required reconstruction losses disappeared: {missing}")
    if result.hard_failures:
        raise RuntimeError(
            "hard accounting/projection failures: " + "; ".join(result.hard_failures)
        )
    raw: dict[str, Any] = {}
    effective: dict[str, float] = {}
    weighted: dict[str, Any] = {}
    for name in required:
        value = torch.as_tensor(result.loss_terms[name])
        if value.numel() != 1:
            raise ValueError(f"loss term {name} must be scalar")
        if not value.is_floating_point():
            raise TypeError(f"loss term {name} must be floating point")
        # Forward AMP is bfloat16, while all objective weighting/accumulation
        # is explicitly float32 under the locked optimization contract.
        value = value.reshape(()).float()
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f"loss term {name} is nonfinite")
        weight = float(getattr(resolved, name))
        if name == "distribution":
            # The stochastic objective warms up during optimization, but
            # model-val checkpoint scores must remain comparable over time.
            weight = (
                float(resolved.distribution)
                if context.validation
                else float(context.curriculum.distribution_weight)
            )
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"active loss {name} has an invalid effective weight")
        raw[name] = value
        effective[name] = weight
        weighted[name] = value * weight
    for value in result.tensors_to_check:
        tensor = torch.as_tensor(value)
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError("step output contains nonfinite tensors")
    total = sum(weighted.values())
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("composed reconstruction loss is nonfinite")
    return ComposedReconstructionLoss(
        total=total,
        raw_terms=raw,
        effective_weights=effective,
        weighted_terms=weighted,
        required_terms=required,
    )


class StatefulBatchSource(Protocol):
    def next_batch(self) -> Any: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, payload: Mapping[str, Any]) -> None: ...


class CyclingSequenceBatchSource:
    """Small deterministic restart-safe source used by smoke/overfit jobs."""

    def __init__(self, batches: Sequence[Any]) -> None:
        if not batches:
            raise ValueError("batch source requires at least one batch")
        self.batches = tuple(batches)
        self.position = 0
        self.cycles = 0

    def next_batch(self) -> Any:
        value = self.batches[self.position]
        self.position += 1
        if self.position == len(self.batches):
            self.position = 0
            self.cycles += 1
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "cycling_sequence",
            "length": len(self.batches),
            "position": int(self.position),
            "cycles": int(self.cycles),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("kind") != "cycling_sequence":
            raise ValueError("batch-source checkpoint kind mismatch")
        if int(payload.get("length", -1)) != len(self.batches):
            raise ValueError("batch-source checkpoint length mismatch")
        position = int(payload.get("position", -1))
        cycles = int(payload.get("cycles", -1))
        if not 0 <= position < len(self.batches) or cycles < 0:
            raise ValueError("batch-source checkpoint cursor is invalid")
        self.position = position
        self.cycles = cycles


class ExponentialMovingAverage:
    def __init__(self, model: Any, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    def update(self, model: Any) -> None:
        torch = require_torch()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                current = value.detach()
                if current.is_floating_point():
                    self.shadow[name].mul_(self.decay).add_(
                        current, alpha=1.0 - self.decay
                    )
                else:
                    self.shadow[name].copy_(current)

    def copy_to(self, model: Any) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def reset_from(self, model: Any) -> None:
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @contextmanager
    def applied(self, model: Any):
        online = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.copy_to(model)
        try:
            yield
        finally:
            model.load_state_dict(online, strict=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": float(self.decay),
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if not math.isclose(float(payload["decay"]), self.decay, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("EMA decay differs from checkpoint")
        shadow = dict(payload["shadow"])
        if set(shadow) != set(self.shadow):
            raise ValueError("EMA checkpoint parameter names differ")
        self.shadow = {name: value.detach().clone() for name, value in shadow.items()}


def build_reconstructor_optimizer(
    model: Any,
    module_groups: Mapping[str, Any],
    *,
    policies: Mapping[str, OptimizerGroupPolicy] | None = None,
    beta1: float = 0.9,
    beta2: float = 0.95,
    epsilon: float = 1.0e-8,
) -> Any:
    """Build exact, non-overlapping named groups covering every model parameter."""

    torch = require_torch()
    resolved = dict(policies or default_optimizer_group_policies())
    missing_groups = sorted(set(ABPH_RECONSTRUCTOR_MODULE_GROUPS) - set(module_groups))
    extra_groups = sorted(set(module_groups) - set(ABPH_RECONSTRUCTOR_MODULE_GROUPS))
    if missing_groups or extra_groups:
        raise ValueError(
            f"module groups must match the Step 8 contract; missing={missing_groups}, "
            f"extra={extra_groups}"
        )
    model_parameters = {id(parameter): parameter for parameter in model.parameters()}
    claimed: dict[int, str] = {}
    optimizer_groups = []
    for name in ABPH_RECONSTRUCTOR_MODULE_GROUPS:
        parameters = list(module_groups[name].parameters())
        if not parameters:
            raise ValueError(f"module group {name} has no parameters")
        for parameter in parameters:
            identifier = id(parameter)
            if identifier not in model_parameters:
                raise ValueError(f"module group {name} contains a parameter outside model")
            if identifier in claimed:
                raise ValueError(
                    f"parameter is shared by optimizer groups {claimed[identifier]} and {name}"
                )
            claimed[identifier] = name
        policy = resolved[name]
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": 0.0,
                "peak_lr": float(policy.peak_lr),
                "weight_decay": float(policy.weight_decay),
                "group_name": name,
            }
        )
    unclaimed = set(model_parameters) - set(claimed)
    if unclaimed:
        raise ValueError(f"optimizer groups leave {len(unclaimed)} model parameters unclaimed")
    return torch.optim.AdamW(
        optimizer_groups,
        betas=(float(beta1), float(beta2)),
        eps=float(epsilon),
    )


def _stage_lr_multiplier(
    state: CurriculumState,
    *,
    warmup_fraction: float,
    minimum_fraction: float,
) -> float:
    progress = float(state.stage_progress)
    if progress < float(warmup_fraction):
        return max(progress / float(warmup_fraction), 1.0e-3)
    cosine_progress = (progress - float(warmup_fraction)) / (1.0 - float(warmup_fraction))
    cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return float(minimum_fraction) + (1.0 - float(minimum_fraction)) * cosine


def _group_phase_factor(group_name: str, state: CurriculumState) -> float:
    if group_name == "hlt_encoder":
        return 1.0 if state.phase == 1 else 0.25
    if group_name == "root":
        return 1.0 if state.phase == 1 else 0.25
    if group_name.startswith("hierarchy_"):
        capacity = int(group_name.rsplit("_", 1)[1])
        if state.phase < 2 or capacity > int(state.active_capacity):
            return 0.0
        return 1.0 if state.phase == 2 and capacity == int(state.active_capacity) else 0.25
    if group_name == "renderer":
        if state.phase < 3:
            return 0.0
        return 1.0 if state.phase == 3 else 0.25
    if group_name == "distribution":
        return 1.0 if state.phase == 4 else 0.0
    raise KeyError(group_name)


def configure_reconstructor_optimizer(
    optimizer: Any,
    state: CurriculumState,
    config: ReconstructorTrainerConfig,
) -> list[dict[str, Any]]:
    """Set phase trainability/LRs and return the complete auditable contract."""

    schedule = _stage_lr_multiplier(
        state,
        warmup_fraction=float(config.warmup_fraction),
        minimum_fraction=float(config.cosine_minimum_fraction),
    )
    metadata = []
    for group in optimizer.param_groups:
        name = str(group["group_name"])
        phase_factor = _group_phase_factor(name, state)
        trainable = phase_factor > 0.0
        learning_rate = float(group["peak_lr"]) * phase_factor * schedule if trainable else 0.0
        group["lr"] = learning_rate
        for parameter in group["params"]:
            parameter.requires_grad_(trainable)
        metadata.append(
            {
                "group_name": name,
                "trainable": trainable,
                "peak_lr": float(group["peak_lr"]),
                "phase_factor": float(phase_factor),
                "schedule_factor": float(schedule),
                "learning_rate": float(learning_rate),
                "weight_decay": float(group["weight_decay"]),
                "parameter_count": int(sum(p.numel() for p in group["params"])),
            }
        )
    return metadata


def _training_mode(state: CurriculumState, seed: int) -> str:
    probability = float(state.teacher_forcing_probability)
    if probability <= 0.0:
        return "rollout"
    if probability >= 1.0:
        return "teacher_forced"
    digest = hashlib.sha256(f"{int(seed)}:{int(state.global_update)}".encode("ascii")).digest()
    draw = int.from_bytes(digest[:8], "big") / float(2**64)
    return "teacher_forced" if draw < probability else "rollout"


def _rng_state() -> dict[str, Any]:
    torch = require_torch()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    torch = require_torch()
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch"])
    if payload.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda"])


def _autocast(device: Any, config: ReconstructorTrainerConfig):
    torch = require_torch()
    enabled = bool(config.amp and device.type == "cuda")
    if not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _make_scaler(device: Any, config: ReconstructorTrainerConfig) -> Any:
    torch = require_torch()
    enabled = bool(config.amp and device.type == "cuda" and config.amp_dtype == "float16")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover - older torch
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_to_device(value: Any, device: Any) -> Any:
    torch = require_torch()
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _scalar_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    torch = require_torch()
    result: dict[str, float] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), item)
            return
        if isinstance(value, (float, int, np.number)) and not isinstance(value, bool):
            number = float(value)
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            number = float(value.detach().cpu())
        else:
            return
        if math.isfinite(number):
            result[prefix] = number

    visit("", values)
    return result


def _objective_gradient_norms(
    composed: ComposedReconstructionLoss,
    parameters: Sequence[Any],
) -> dict[str, float]:
    """Measure each active weighted objective without consuming the graph."""

    torch = require_torch()
    result: dict[str, float] = {}
    for name, objective in composed.weighted_terms.items():
        if not objective.requires_grad:
            result[name] = 0.0
            continue
        gradients = torch.autograd.grad(
            objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared = sum(
            gradient.detach().float().square().sum()
            for gradient in gradients
            if gradient is not None
        )
        if isinstance(squared, int):
            result[name] = 0.0
        else:
            result[name] = float(torch.sqrt(squared).cpu())
    return result


def _optimizer_group_gradient_norms(optimizer: Any) -> dict[str, float]:
    torch = require_torch()
    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        squared = sum(
            parameter.grad.detach().float().square().sum()
            for parameter in group["params"]
            if parameter.grad is not None
        )
        result[str(group["group_name"])] = (
            0.0 if isinstance(squared, int) else float(torch.sqrt(squared).cpu())
        )
    return result


def evaluate_reconstructor_rollout(
    model: Any,
    batches: Iterable[Any],
    state: CurriculumState,
    step_function: Callable[[Any, Any, ReconstructorStepContext], ReconstructorStepResult],
    config: ReconstructorTrainerConfig,
    device: Any,
) -> dict[str, Any]:
    """Evaluate only a complete zero-teacher-forcing model-val rollout."""

    torch = require_torch()
    model.eval()
    context = ReconstructorStepContext(
        curriculum=state,
        split="model_val",
        mode="rollout",
        validation=True,
        teacher_forcing_probability=0.0,
    )
    sums: dict[str, float] = {}
    total_jets = 0
    batches_seen = 0
    effective_weights: dict[str, float] | None = None
    with torch.no_grad():
        for cpu_batch in batches:
            batch = _move_to_device(cpu_batch, device)
            with _autocast(device, config):
                result = step_function(model, batch, context)
            with torch.autocast(device_type=device.type, enabled=False):
                composed = compose_reconstruction_loss(result, context, config.loss_weights)
            weight = int(result.batch_size)
            current_effective_weights = dict(composed.effective_weights)
            if effective_weights is None:
                effective_weights = current_effective_weights
            elif current_effective_weights != effective_weights:
                raise RuntimeError("model-val objective weights changed within one evaluation")
            row = {
                "loss.total": float(composed.total.detach().float().cpu()),
                **{
                    f"loss.raw.{name}": float(value.detach().float().cpu())
                    for name, value in composed.raw_terms.items()
                },
                **{
                    f"loss.weighted.{name}": float(value.detach().float().cpu())
                    for name, value in composed.weighted_terms.items()
                },
                **_scalar_metrics(result.metrics),
            }
            for name, value in row.items():
                sums[name] = sums.get(name, 0.0) + value * weight
            total_jets += weight
            batches_seen += 1
    if batches_seen == 0 or total_jets == 0:
        raise RuntimeError("rollout model validation produced no batches")
    averages = {name: value / total_jets for name, value in sums.items()}
    selection = float(averages["loss.total"])
    if not math.isfinite(selection):
        raise FloatingPointError("rollout model-validation score is nonfinite")
    return {
        "contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
        "split": "model_val",
        "mode": "rollout",
        "teacher_forcing_probability": 0.0,
        "offline_targets_loaded": True,
        "teacher_logits_loaded": False,
        "checkpoint_selection_eligible": True,
        "selection_metric": "model_val.rollout.loss.total",
        "selection_score": selection,
        "n_jets": int(total_jets),
        "n_batches": int(batches_seen),
        "required_losses": list(active_reconstruction_loss_names(context)),
        "effective_weights": effective_weights,
        "metrics": averages,
    }


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    ema: ExponentialMovingAverage,
    controller: CurriculumController,
    train_source: StatefulBatchSource,
    config: ReconstructorTrainerConfig,
    role: str,
    validation: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
    optimizer_metadata: Sequence[Mapping[str, Any]],
    nonfinite_updates: int,
    trainer_state: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    selected = role in {"best_stage_model_val", "best_model_val"}
    return {
        "checkpoint_contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
        "checkpoint_role": role,
        "selection_split": "model_val" if selected else None,
        "selection_mode": "rollout" if selected else None,
        "model_state_dict": (
            {name: value.detach().clone() for name, value in ema.shadow.items()}
            if selected
            else model.state_dict()
        ),
        "online_model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "curriculum_state_dict": controller.state_dict(),
        "train_source_state_dict": dict(train_source.state_dict()),
        "rng_state": _rng_state(),
        "config": config.to_dict(),
        "model_metadata": _jsonable(model_metadata),
        "validation": _jsonable(validation),
        "provenance": _jsonable(provenance),
        "optimizer_groups": _jsonable(optimizer_metadata),
        "nonfinite_updates": int(nonfinite_updates),
        "trainer_state": _jsonable(trainer_state),
        "final_test_loaded": False,
        "teacher_logits_loaded": False,
    }


def _load_torch_payload(path: Path, device: Any) -> Mapping[str, Any]:
    torch = require_torch()
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location=device)


def load_reconstructor_curriculum_checkpoint(
    path: str | Path,
    *,
    device: str | Any = "cpu",
    require_selected: bool = False,
) -> Mapping[str, Any]:
    resolved = resolve_device(device) if isinstance(device, str) else device
    payload = _load_torch_payload(Path(path), resolved)
    if payload.get("checkpoint_contract") != ABPH_RECONSTRUCTOR_TRAINING_CONTRACT:
        raise ValueError("reconstructor checkpoint contract mismatch")
    if require_selected and payload.get("checkpoint_role") not in {
        "best_stage_model_val",
        "best_model_val",
    }:
        raise ValueError("a model-val selected reconstructor checkpoint is required")
    if payload.get("final_test_loaded") is not False:
        raise ValueError("reconstructor checkpoint does not attest final-test isolation")
    return payload


def _component_config(module: Any) -> Any:
    config = getattr(module, "config", None)
    if config is None:
        return None
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    return repr(config)


def describe_reconstructor_model(model: Any, module_groups: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "model_config": _component_config(model),
        "module_groups": {
            name: {
                "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
                "config": _component_config(module),
                "parameter_count": int(sum(value.numel() for value in module.parameters())),
            }
            for name, module in module_groups.items()
        },
    }
    canonical = _jsonable(payload)
    canonical["model_metadata_hash"] = _canonical_hash(canonical)
    return canonical


def train_reconstructor_curriculum(
    model: Any,
    module_groups: Mapping[str, Any],
    train_source: StatefulBatchSource,
    validation_batches: Callable[[], Iterable[Any]],
    step_function: Callable[[Any, Any, ReconstructorStepContext], ReconstructorStepResult],
    config: ReconstructorTrainerConfig,
    *,
    provenance: Mapping[str, Any] | None = None,
    resume_from: str | Path | None = None,
    maximum_optimizer_updates: int | None = None,
    optimizer_policies: Mapping[str, OptimizerGroupPolicy] | None = None,
) -> dict[str, Any]:
    """Run phases 1-4 with exact restart and rollout-only model selection."""

    torch = require_torch()
    set_training_seed(int(config.seed))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    model.to(device)
    optimizer = build_reconstructor_optimizer(
        model,
        module_groups,
        policies=optimizer_policies,
        beta1=float(config.adam_beta1),
        beta2=float(config.adam_beta2),
        epsilon=float(config.adam_epsilon),
    )
    scaler = _make_scaler(device, config)
    ema = ExponentialMovingAverage(model, float(config.ema_decay))
    controller = CurriculumController(config.curriculum)
    source_provenance = _jsonable(dict(provenance or {}))
    model_metadata = describe_reconstructor_model(model, module_groups)
    config_hash = config.to_dict()["config_hash"]
    nonfinite_updates = 0
    objective_skip_counts = {name: 0 for name in ABPH_RECONSTRUCTION_LOSS_NAMES}
    curves: list[dict[str, Any]] = []
    best_by_stage: dict[str, dict[str, Any]] = {}
    evaluations_without_improvement: dict[str, int] = {}

    if resume_from is not None:
        payload = load_reconstructor_curriculum_checkpoint(resume_from, device=device)
        if payload.get("config", {}).get("config_hash") != config_hash:
            raise ValueError("resume checkpoint training configuration mismatch")
        if dict(payload.get("provenance", {})) != source_provenance:
            raise ValueError("resume checkpoint provenance mismatch")
        if dict(payload.get("model_metadata", {})) != model_metadata:
            raise ValueError("resume checkpoint model/module configuration mismatch")
        model.load_state_dict(payload["online_model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        ema.load_state_dict(payload["ema_state_dict"])
        controller.load_state_dict(payload["curriculum_state_dict"])
        train_source.load_state_dict(payload["train_source_state_dict"])
        _restore_rng_state(payload["rng_state"])
        nonfinite_updates = int(payload.get("nonfinite_updates", 0))
        saved_trainer_state = dict(payload.get("trainer_state", {}))
        curves = list(saved_trainer_state.get("curves", []))
        best_by_stage = {
            str(name): dict(value)
            for name, value in dict(saved_trainer_state.get("best_by_stage", {})).items()
        }
        evaluations_without_improvement = {
            str(name): int(value)
            for name, value in dict(
                saved_trainer_state.get("evaluations_without_improvement", {})
            ).items()
        }
        objective_skip_counts = {
            name: int(value)
            for name, value in dict(
                saved_trainer_state.get("objective_skip_counts", objective_skip_counts)
            ).items()
        }

    _atomic_json(
        output_dir / "config.json",
        {
            "contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
            "config": config.to_dict(),
            "provenance": source_provenance,
            "module_groups": list(ABPH_RECONSTRUCTOR_MODULE_GROUPS),
            "model_metadata": model_metadata,
            "step_function": (
                f"{step_function.__module__}."
                f"{getattr(step_function, '__qualname__', step_function.__class__.__qualname__)}"
            ),
            "final_test_loaded": False,
            "teacher_logits_loaded": False,
        },
    )

    updates_this_call = 0
    while not controller.complete:
        if maximum_optimizer_updates is not None and updates_this_call >= int(maximum_optimizer_updates):
            break
        state = controller.state()
        optimizer_metadata = configure_reconstructor_optimizer(optimizer, state, config)
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        accumulated_jets = 0
        update_failed = False
        train_rows: list[dict[str, float]] = []
        training_required_losses: tuple[str, ...] = ()
        will_evaluate = (
            state.stage_update + 1 >= state.stage_maximum_updates
            or (state.global_update + 1) % int(config.curriculum.evaluation_interval) == 0
        )
        objective_gradient_sums: dict[str, float] = {}
        active_parameters_for_update = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        for _ in range(int(config.gradient_accumulation_steps)):
            context = ReconstructorStepContext(
                curriculum=state,
                split="model_train",
                mode=_training_mode(state, int(config.seed)),
                validation=False,
                teacher_forcing_probability=float(state.teacher_forcing_probability),
            )
            try:
                batch = _move_to_device(train_source.next_batch(), device)
                with _autocast(device, config):
                    result = step_function(model, batch, context)
                with torch.autocast(device_type=device.type, enabled=False):
                    composed = compose_reconstruction_loss(result, context, config.loss_weights)
                    training_required_losses = composed.required_terms
                    scaled_loss = composed.total / float(config.gradient_accumulation_steps)
                if will_evaluate:
                    objective_norms = _objective_gradient_norms(
                        composed, active_parameters_for_update
                    )
                    for name, value in objective_norms.items():
                        objective_gradient_sums[name] = (
                            objective_gradient_sums.get(name, 0.0) + value
                        )
                scaler.scale(scaled_loss).backward()
                accumulated_jets += int(result.batch_size)
                train_rows.append(
                    {
                        "loss.total": float(composed.total.detach().float().cpu()),
                        **{
                            f"loss.raw.{name}": float(value.detach().float().cpu())
                            for name, value in composed.raw_terms.items()
                        },
                        **{
                            f"loss.normalized.{name}": float(value.detach().float().cpu())
                            for name, value in composed.raw_terms.items()
                        },
                        **{
                            f"loss.effective_weight.{name}": float(value)
                            for name, value in composed.effective_weights.items()
                        },
                    }
                )
            except FloatingPointError as exc:
                update_failed = True
                nonfinite_updates += 1
                message = str(exc)
                for name in ABPH_RECONSTRUCTION_LOSS_NAMES:
                    if f"loss term {name} " in message:
                        objective_skip_counts[name] += 1
                        break
                optimizer.zero_grad(set_to_none=True)
                if nonfinite_updates > int(config.maximum_nonfinite_updates):
                    raise RuntimeError(
                        "reconstructor exceeded the maximum skipped nonfinite updates"
                    ) from exc
                break
        if update_failed:
            continue
        effective_batch_size = int(accumulated_jets) * int(config.distributed_world_size)
        expected_effective_batch_size = (
            int(config.root_hierarchy_effective_batch_size)
            if state.phase <= 2
            else int(config.renderer_distribution_effective_batch_size)
        )
        if (
            bool(config.enforce_effective_batch_size)
            and effective_batch_size != expected_effective_batch_size
        ):
            raise ValueError(
                f"stage {state.stage_key} accumulated effective batch size "
                f"{effective_batch_size}, expected {expected_effective_batch_size}"
            )
        scaler.unscale_(optimizer)
        active_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradients = [parameter.grad for parameter in active_parameters if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            nonfinite_updates += 1
            optimizer.zero_grad(set_to_none=True)
            if nonfinite_updates > int(config.maximum_nonfinite_updates):
                raise RuntimeError("reconstructor gradients remained absent/nonfinite")
            continue
        optimizer_group_gradient_norms = _optimizer_group_gradient_norms(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            active_parameters, float(config.gradient_clip_norm)
        )
        if not bool(torch.isfinite(gradient_norm)):
            nonfinite_updates += 1
            optimizer.zero_grad(set_to_none=True)
            if nonfinite_updates > int(config.maximum_nonfinite_updates):
                raise RuntimeError("reconstructor gradient norm remained nonfinite")
            continue
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        transitioned = controller.advance()
        updates_this_call += 1

        should_evaluate = (
            transitioned
            or controller.global_update % int(config.curriculum.evaluation_interval) == 0
        )
        if not should_evaluate:
            continue
        with ema.applied(model):
            validation = evaluate_reconstructor_rollout(
                model,
                validation_batches(),
                state,
                step_function,
                config,
                device,
            )
        train_averages = {
            name: sum(row.get(name, 0.0) for row in train_rows) / len(train_rows)
            for name in sorted({name for row in train_rows for name in row})
        }
        curve = {
            "global_update": int(controller.global_update),
            "curriculum": state.to_dict(),
            "train": {
                "n_jets": int(accumulated_jets),
                "effective_batch_size": int(effective_batch_size),
                "expected_effective_batch_size": int(expected_effective_batch_size),
                "mode": _training_mode(state, int(config.seed)),
                "required_losses": list(training_required_losses),
                "metrics": train_averages,
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "objective_gradient_norms": {
                    name: value / float(config.gradient_accumulation_steps)
                    for name, value in objective_gradient_sums.items()
                },
                "optimizer_group_gradient_norms": optimizer_group_gradient_norms,
                "objective_skip_counts": dict(objective_skip_counts),
                "nonfinite_updates": int(nonfinite_updates),
            },
            "model_val_rollout": validation,
            "optimizer_groups": optimizer_metadata,
        }
        curves.append(curve)
        score = float(validation["selection_score"])
        best = best_by_stage.get(state.stage_key)
        improved = best is None or score < float(best["selection_score"])
        if improved:
            evaluations_without_improvement[state.stage_key] = 0
            stage_path = output_dir / f"best_{state.stage_key}.pt"
            best_by_stage[state.stage_key] = {
                "selection_score": score,
                "global_update": int(controller.global_update),
                "checkpoint": str(stage_path),
            }
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                controller=controller,
                train_source=train_source,
                config=config,
                role=(
                    "best_model_val"
                    if state.stage_key == "phase4_distribution"
                    else "best_stage_model_val"
                ),
                validation=validation,
                provenance=source_provenance,
                optimizer_metadata=optimizer_metadata,
                nonfinite_updates=nonfinite_updates,
                trainer_state={
                    "curves": curves,
                    "best_by_stage": best_by_stage,
                    "evaluations_without_improvement": evaluations_without_improvement,
                    "objective_skip_counts": objective_skip_counts,
                },
                model_metadata=model_metadata,
            )
            _atomic_torch_save(stage_path, payload)
            if state.stage_key == "phase4_distribution":
                _atomic_torch_save(output_dir / "best_model_val.pt", payload)
            best_by_stage[state.stage_key]["checkpoint_sha256"] = _file_sha256(stage_path)
        else:
            evaluations_without_improvement[state.stage_key] = (
                evaluations_without_improvement.get(state.stage_key, 0) + 1
            )

        if config.save_last_checkpoint:
            _atomic_torch_save(
                output_dir / "last.pt",
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    ema=ema,
                    controller=controller,
                    train_source=train_source,
                    config=config,
                    role="last",
                    validation=validation,
                    provenance=source_provenance,
                    optimizer_metadata=optimizer_metadata,
                    nonfinite_updates=nonfinite_updates,
                    trainer_state={
                        "curves": curves,
                        "best_by_stage": best_by_stage,
                        "evaluations_without_improvement": evaluations_without_improvement,
                        "objective_skip_counts": objective_skip_counts,
                    },
                    model_metadata=model_metadata,
                ),
            )
        _atomic_json(
            output_dir / "training_curves.json",
            {
                "contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
                "selection_metric": "model_val.rollout.loss.total",
                "rollout_validation_required": True,
                "evaluations": curves,
            },
        )

        early_transition = False
        if not transitioned:
            stage_definition = controller.stages[state.stage_index]
            early_transition = (
                evaluations_without_improvement.get(state.stage_key, 0)
                >= int(stage_definition.patience_evaluations)
            )
            if early_transition:
                controller.finish_stage_early()
        if transitioned or early_transition:
            selected_path = Path(best_by_stage[state.stage_key]["checkpoint"])
            selected = load_reconstructor_curriculum_checkpoint(
                selected_path, device=device, require_selected=True
            )
            model.load_state_dict(selected["model_state_dict"], strict=True)
            ema.reset_from(model)
            optimizer.state.clear()
            handoff_optimizer_metadata = (
                configure_reconstructor_optimizer(optimizer, controller.state(), config)
                if not controller.complete
                else optimizer_metadata
            )
            if config.save_last_checkpoint:
                _atomic_torch_save(
                    output_dir / "last.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        ema=ema,
                        controller=controller,
                        train_source=train_source,
                        config=config,
                        role="last",
                        validation=validation,
                        provenance=source_provenance,
                        optimizer_metadata=handoff_optimizer_metadata,
                        nonfinite_updates=nonfinite_updates,
                        trainer_state={
                            "curves": curves,
                            "best_by_stage": best_by_stage,
                            "evaluations_without_improvement": evaluations_without_improvement,
                            "objective_skip_counts": objective_skip_counts,
                        },
                        model_metadata=model_metadata,
                    ),
                )

    optimizer_metadata = (
        configure_reconstructor_optimizer(optimizer, controller.state(), config)
        if not controller.complete
        else [
            {
                "group_name": str(group["group_name"]),
                "trainable": bool(any(p.requires_grad for p in group["params"])),
                "learning_rate": float(group["lr"]),
                "peak_lr": float(group["peak_lr"]),
                "parameter_count": int(sum(p.numel() for p in group["params"])),
            }
            for group in optimizer.param_groups
        ]
    )
    if config.save_last_checkpoint:
        _atomic_torch_save(
            output_dir / "last.pt",
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                controller=controller,
                train_source=train_source,
                config=config,
                role="last",
                validation=curves[-1]["model_val_rollout"] if curves else None,
                provenance=source_provenance,
                optimizer_metadata=optimizer_metadata,
                nonfinite_updates=nonfinite_updates,
                trainer_state={
                    "curves": curves,
                    "best_by_stage": best_by_stage,
                    "evaluations_without_improvement": evaluations_without_improvement,
                    "objective_skip_counts": objective_skip_counts,
                },
                model_metadata=model_metadata,
            ),
        )
    complete = bool(controller.complete)
    if complete and not (output_dir / "best_model_val.pt").exists():
        raise RuntimeError("completed curriculum produced no phase-4 model-val checkpoint")
    report = {
        "contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
        "ok": complete,
        "status": "complete" if complete else "interrupted",
        "config_hash": config_hash,
        "curriculum": controller.state_dict(),
        "best_by_stage": best_by_stage,
        "best_model_val_checkpoint": (
            str(output_dir / "best_model_val.pt") if complete else None
        ),
        "best_model_val_checkpoint_sha256": (
            _file_sha256(output_dir / "best_model_val.pt") if complete else None
        ),
        "selection_split": "model_val",
        "selection_mode": "rollout",
        "rollout_validation_count": len(curves),
        "teacher_forced_validation_count": 0,
        "nonfinite_updates": int(nonfinite_updates),
        "objective_skip_counts": objective_skip_counts,
        "optimizer_groups": optimizer_metadata,
        "model_metadata": model_metadata,
        "provenance": source_provenance,
        "final_test_loaded": False,
        "teacher_logits_loaded": False,
        "stage_handoff": "restore_best_ema_and_reset_optimizer_moments",
    }
    _atomic_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "ABPH_RECONSTRUCTION_LOSS_NAMES",
    "ABPH_RECONSTRUCTOR_MODULE_GROUPS",
    "ABPH_RECONSTRUCTOR_TRAINING_CONTRACT",
    "ComposedReconstructionLoss",
    "CurriculumController",
    "CurriculumStage",
    "CurriculumState",
    "CyclingSequenceBatchSource",
    "ExponentialMovingAverage",
    "OptimizerGroupPolicy",
    "ReconstructionLossWeights",
    "ReconstructorCurriculumConfig",
    "ReconstructorStepContext",
    "ReconstructorStepResult",
    "ReconstructorTrainerConfig",
    "StatefulBatchSource",
    "active_reconstruction_loss_names",
    "assemble_reconstruction_loss_terms",
    "build_reconstructor_optimizer",
    "compose_reconstruction_loss",
    "configure_reconstructor_optimizer",
    "default_optimizer_group_policies",
    "describe_reconstructor_model",
    "evaluate_reconstructor_rollout",
    "load_reconstructor_curriculum_checkpoint",
    "train_reconstructor_curriculum",
]

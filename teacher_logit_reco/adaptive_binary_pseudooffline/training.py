"""Update-count curriculum training for adaptive binary pseudo-offline models.

This module owns the scientific training contract rather than the model's
forward graph.  A concrete reconstructor supplies named modules and a step
function; the trainer supplies the locked phase schedule, objective contract,
optimizer state, EMA selection, nonfinite handling, and restart semantics.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, set_training_seed

from .convergence_schedule import (
    ABPH_ACCELERATED_SCHEDULE_CONTRACT,
    ABPH_LEGACY_SCHEDULE_CONTRACT,
    ABPH_STAGE_ROLES,
    StageScheduleBudget,
    decide_stage_continuation,
)
from .checkpoints import (
    ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT,
    ABPH_EPHEMERAL_RESUME_CHECKPOINT_CONTRACT,
    ABPH_RESTART_SEMANTICS_EXACT,
    ABPH_RESTART_SEMANTICS_WARM_START,
    build_compact_selected_checkpoint,
    ephemeral_checkpoint_path,
    require_exact_resume_checkpoint,
    streaming_storage_enabled,
    validate_compact_selected_checkpoint,
    write_selected_checkpoint,
)
from .distributed import (
    DistributedRuntime,
    ReconstructorTrainingModule,
    abort_distributed_runtime,
    all_gather_objects,
    all_reduce_min_bool,
    all_reduce_sum_int,
    any_structural_error,
    barrier,
    broadcast_object,
    build_stage_ddp_wrapper,
    gather_error_summaries,
    initialize_distributed_runtime,
    require_standard_tensor_mapping,
    tensor_mapping_is_finite,
    verify_common_parameter_state,
)
from .distributed_validation import (
    ABPH_DISTRIBUTED_VALIDATION_CONTRACT,
    TypedValidationAccumulator,
    finalize_typed_validation,
)
from .hypothesis_distribution import conditional_distribution_weight
from .input_pipeline import (
    ABPH_INPUT_PIPELINE_CONTRACT,
    ABPH_PREFETCH_QUEUE_DEPTH,
    BatchTransferStager,
    RankLocalBatchPrefetcher,
    prepare_contiguous_cpu_batch,
)
from .runtime_profile import RuntimeProfileConfig, RuntimeProfiler, profile_span
from .targets import ABPH_LEVEL_CAPACITIES


ABPH_RECONSTRUCTOR_TRAINING_CONTRACT = (
    "adaptive_binary_pseudooffline_reconstructor_training_v1"
)
ABPH_DISTRIBUTED_CHECKPOINT_CONTRACT = "adaptive_binary_distributed_checkpoint_v1"
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


class _SynchronizedForwardRetry(RuntimeError):
    pass


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
    """Progressive-depth schedule with explicit legacy/accelerated semantics."""

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
    maximum_capacity: int = 32
    hierarchy_capacities: tuple[int, ...] = ABPH_LEVEL_CAPACITIES
    renderer_enabled: bool = True
    distribution_enabled: bool = True
    schedule_contract: str = ABPH_LEGACY_SCHEDULE_CONTRACT
    campaign_schedule_profile: str = "legacy"
    root_budget: StageScheduleBudget | None = None
    hierarchy_budget: StageScheduleBudget | None = None
    renderer_budget: StageScheduleBudget | None = None
    distribution_budget: StageScheduleBudget | None = None
    root_stage_role: str = "trained"
    hierarchy_stage_role: str = "trained"
    renderer_stage_role: str = "trained"
    distribution_stage_role: str = "trained"

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
        if int(self.maximum_capacity) not in (1, *ABPH_LEVEL_CAPACITIES):
            raise ValueError(
                f"maximum_capacity must be root-only (1) or one of {ABPH_LEVEL_CAPACITIES}"
            )
        capacities = tuple(int(value) for value in self.hierarchy_capacities)
        if not capacities and int(self.maximum_capacity) != 1:
            raise ValueError("hierarchy_capacities cannot be empty when hierarchy training is enabled")
        if any(value not in ABPH_LEVEL_CAPACITIES for value in capacities):
            raise ValueError(f"hierarchy_capacities must be drawn from {ABPH_LEVEL_CAPACITIES}")
        if capacities != tuple(sorted(set(capacities))):
            raise ValueError("hierarchy_capacities must be unique and increasing")
        if any(value > int(self.maximum_capacity) for value in capacities):
            raise ValueError("hierarchy_capacities cannot exceed maximum_capacity")
        if bool(self.distribution_enabled) and not bool(self.renderer_enabled):
            raise ValueError("distribution training requires the particle renderer phase")
        roles = {
            "root": self.root_stage_role,
            "hierarchy": self.hierarchy_stage_role,
            "renderer": self.renderer_stage_role,
            "distribution": self.distribution_stage_role,
        }
        for family, role in roles.items():
            if str(role) not in ABPH_STAGE_ROLES:
                raise ValueError(f"invalid {family} stage role {role!r}")
        if self.schedule_contract == ABPH_ACCELERATED_SCHEDULE_CONTRACT:
            if self.campaign_schedule_profile not in {"pilot", "highdata"}:
                raise ValueError(
                    "accelerated schedule requires pilot or highdata campaign profile"
                )
            for family in ("root", "hierarchy", "renderer", "distribution"):
                role = str(roles[family])
                budget = getattr(self, f"{family}_budget")
                enabled = (
                    family == "root"
                    or (family == "hierarchy" and int(self.maximum_capacity) > 1)
                    or (family == "renderer" and bool(self.renderer_enabled))
                    or (family == "distribution" and bool(self.distribution_enabled))
                )
                if enabled and role in {"trained", "warm_started_handoff"}:
                    if not isinstance(budget, StageScheduleBudget):
                        raise ValueError(
                            f"accelerated {family} stage requires an explicit budget"
                        )
                elif enabled and role in {"disabled", "oracle"}:
                    raise ValueError(
                        f"{family} role {role!r} conflicts with an enabled curriculum stage"
                    )
            if self.root_stage_role == "warm_started_handoff" and self.root_budget != StageScheduleBudget(1, 0, 1):
                raise ValueError("root warm-start handoff must remain exactly one update")
            if self.hierarchy_stage_role == "warm_started_handoff" and self.hierarchy_budget != StageScheduleBudget(1, 0, 1):
                raise ValueError("hierarchy warm-start handoff must remain exactly one update per depth")
        elif self.schedule_contract != ABPH_LEGACY_SCHEDULE_CONTRACT:
            raise ValueError(f"unknown schedule contract {self.schedule_contract!r}")
        elif any(
            budget is not None
            for budget in (
                self.root_budget,
                self.hierarchy_budget,
                self.renderer_budget,
                self.distribution_budget,
            )
        ):
            raise ValueError("legacy schedules may not carry accelerated stage budgets")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReconstructorCurriculumConfig":
        """Read old schedules as explicitly legacy, never as accelerated."""

        values = dict(payload)
        values.pop("contract", None)
        values.setdefault("schedule_contract", ABPH_LEGACY_SCHEDULE_CONTRACT)
        values.setdefault("campaign_schedule_profile", "legacy")
        for family in ("root", "hierarchy", "renderer", "distribution"):
            key = f"{family}_budget"
            value = values.get(key)
            if isinstance(value, Mapping):
                values[key] = StageScheduleBudget.from_dict(value)
            values.setdefault(f"{family}_stage_role", "trained")
        return cls(**values)

    @property
    def accelerated(self) -> bool:
        return self.schedule_contract == ABPH_ACCELERATED_SCHEDULE_CONTRACT

    def stage_budget(self, family: str) -> StageScheduleBudget:
        normalized = str(family)
        if self.accelerated:
            budget = getattr(self, f"{normalized}_budget")
            if not isinstance(budget, StageScheduleBudget):
                raise ValueError(f"missing accelerated budget for {normalized}")
            return budget
        legacy_updates = {
            "root": self.root_updates,
            "hierarchy": self.hierarchy_updates_per_depth,
            "renderer": self.renderer_updates,
            "distribution": self.distribution_updates,
        }[normalized]
        return StageScheduleBudget(int(legacy_updates), 0, int(legacy_updates))

    def stage_role(self, family: str) -> str:
        return str(getattr(self, f"{family}_stage_role"))


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
    root_hierarchy_gradient_accumulation_steps: int | None = None
    renderer_distribution_gradient_accumulation_steps: int | None = None
    distributed_world_size: int = 1
    root_hierarchy_local_batch_size: int | None = None
    renderer_distribution_local_batch_size: int | None = None
    root_hierarchy_effective_batch_size: int = 1024
    renderer_distribution_effective_batch_size: int = 512
    enforce_effective_batch_size: bool = True
    runtime_batch_contract_path: str | None = None
    runtime_batch_contract_hash: str | None = None
    asynchronous_prefetch: bool = True
    prefetch_queue_depth: int = ABPH_PREFETCH_QUEUE_DEPTH
    pin_memory: bool = True
    nonblocking_transfer: bool = True
    ddp_find_unused_parameters: bool = True
    ddp_broadcast_buffers: bool = False
    gradient_clip_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    warmup_fraction: float = 0.05
    cosine_minimum_fraction: float = 0.05
    ema_decay: float = 0.9999
    maximum_nonfinite_updates: int = 8
    save_last_checkpoint: bool = True
    runtime_profile: RuntimeProfileConfig = RuntimeProfileConfig()
    curriculum: ReconstructorCurriculumConfig = ReconstructorCurriculumConfig()
    loss_weights: ReconstructionLossWeights = ReconstructionLossWeights()

    def __post_init__(self) -> None:
        if not str(self.output_dir).strip():
            raise ValueError("output_dir is required")
        if int(self.gradient_accumulation_steps) <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        for name in (
            "root_hierarchy_gradient_accumulation_steps",
            "renderer_distribution_gradient_accumulation_steps",
            "root_hierarchy_local_batch_size",
            "renderer_distribution_local_batch_size",
        ):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when supplied")
        if bool(self.runtime_batch_contract_path) != bool(self.runtime_batch_contract_hash):
            raise ValueError("runtime batch contract path and hash must be supplied together")
        if int(self.prefetch_queue_depth) != ABPH_PREFETCH_QUEUE_DEPTH:
            raise ValueError(
                f"prefetch_queue_depth is locked to {ABPH_PREFETCH_QUEUE_DEPTH}"
            )
        if bool(self.ddp_broadcast_buffers):
            raise ValueError("ABPH DDP broadcast_buffers is locked off")
        for family in ("root_hierarchy", "renderer_distribution"):
            local = getattr(self, f"{family}_local_batch_size")
            accumulation = getattr(self, f"{family}_gradient_accumulation_steps")
            effective = getattr(self, f"{family}_effective_batch_size")
            if local is not None and accumulation is not None:
                actual = int(self.distributed_world_size) * int(local) * int(accumulation)
                if actual != int(effective):
                    raise ValueError(
                        f"{family} runtime batch arithmetic gives {actual}, expected {effective}"
                    )
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


def _normalized_trainer_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize pre-runtime schedule payloads for explicit legacy resume only."""

    values = dict(payload)
    values.pop("contract", None)
    values.pop("config_hash", None)
    values["curriculum"] = ReconstructorCurriculumConfig.from_dict(
        dict(values.get("curriculum", {}))
    ).to_dict()
    values.setdefault("runtime_profile", asdict(RuntimeProfileConfig()))
    return values


@dataclass(frozen=True)
class CurriculumStage:
    key: str
    phase: int
    phase_name: str
    family: str
    role: str
    budget: StageScheduleBudget
    patience_evaluations: int
    active_capacity: int

    @property
    def maximum_updates(self) -> int:
        return int(self.budget.hard_max_updates)


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
    supervised_capacities: tuple[int, ...] = ABPH_LEVEL_CAPACITIES
    stage_nominal_updates: int = 0
    stage_extension_updates: int = 0
    stage_hard_max_updates: int = 0
    stage_extension_blocks: int = 0
    stage_role: str = "trained"
    schedule_contract: str = ABPH_LEGACY_SCHEDULE_CONTRACT

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
        self.extension_blocks: dict[str, int] = {
            stage.key: 0 for stage in self.stages
        }
        self.stage_outcomes: dict[str, str] = {}
        self.schedule_events: list[dict[str, Any]] = []
        self.schedule_truncated = False
        self._awaiting_boundary_decision = False

    @staticmethod
    def _build_stages(config: ReconstructorCurriculumConfig) -> tuple[CurriculumStage, ...]:
        stages = [
            CurriculumStage(
                key="phase1_root",
                phase=1,
                phase_name="root_pretraining",
                family="root",
                role=config.stage_role("root"),
                budget=config.stage_budget("root"),
                patience_evaluations=int(config.root_patience_evaluations),
                active_capacity=1,
            )
        ]
        stages.extend(
            CurriculumStage(
                key=f"phase2_hierarchy_{capacity}",
                phase=2,
                phase_name="progressive_hierarchy",
                family="hierarchy",
                role=config.stage_role("hierarchy"),
                budget=config.stage_budget("hierarchy"),
                patience_evaluations=int(config.hierarchy_patience_evaluations),
                active_capacity=int(capacity),
            )
            for capacity in config.hierarchy_capacities
            if int(capacity) <= int(config.maximum_capacity)
        )
        if bool(config.renderer_enabled):
            stages.append(CurriculumStage(
                key="phase3_renderer",
                phase=3,
                phase_name="deterministic_particle_rendering",
                family="renderer",
                role=config.stage_role("renderer"),
                budget=config.stage_budget("renderer"),
                patience_evaluations=int(config.renderer_patience_evaluations),
                active_capacity=32,
            ))
        if bool(config.distribution_enabled):
            stages.append(CurriculumStage(
                key="phase4_distribution",
                phase=4,
                phase_name="probabilistic_multi_hypothesis",
                family="distribution",
                role=config.stage_role("distribution"),
                budget=config.stage_budget("distribution"),
                patience_evaluations=int(config.distribution_patience_evaluations),
                active_capacity=32,
            ))
        return tuple(stages)

    def is_final_stage(self, state: CurriculumState | None = None) -> bool:
        if state is not None:
            return int(state.stage_index) == len(self.stages) - 1
        if self.complete:
            return False
        resolved = self.state()
        return int(resolved.stage_index) == len(self.stages) - 1

    @property
    def complete(self) -> bool:
        return self.stage_index >= len(self.stages)

    def _progress(self, stage: CurriculumStage) -> float:
        nominal = int(stage.budget.nominal_updates)
        if nominal <= 1:
            return 1.0
        return min(max(self.stage_update / float(nominal - 1), 0.0), 1.0)

    def _current_stage_limit(self, stage: CurriculumStage) -> int:
        if not self.config.accelerated:
            return int(stage.maximum_updates)
        blocks = int(self.extension_blocks[stage.key])
        return min(
            int(stage.budget.nominal_updates)
            + blocks * int(stage.budget.extension_updates),
            int(stage.budget.hard_max_updates),
        )

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
                supervised_capacities=tuple(self.config.hierarchy_capacities),
                stage_nominal_updates=0,
                stage_extension_updates=0,
                stage_hard_max_updates=0,
                stage_extension_blocks=0,
                stage_role="disabled",
                schedule_contract=self.config.schedule_contract,
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
            stage_maximum_updates=self._current_stage_limit(stage),
            active_capacity=int(stage.active_capacity),
            stage_progress=float(progress),
            teacher_forcing_probability=float(self._teacher_forcing(stage, progress)),
            distribution_weight=(
                float(conditional_distribution_weight(progress))
                if stage.phase == 4
                else 0.0
            ),
            complete=False,
            supervised_capacities=tuple(self.config.hierarchy_capacities),
            stage_nominal_updates=int(stage.budget.nominal_updates),
            stage_extension_updates=int(stage.budget.extension_updates),
            stage_hard_max_updates=int(stage.budget.hard_max_updates),
            stage_extension_blocks=int(self.extension_blocks[stage.key]),
            stage_role=stage.role,
            schedule_contract=self.config.schedule_contract,
        )

    def advance(self) -> bool:
        if self.complete:
            raise RuntimeError("cannot advance a completed curriculum")
        if self._awaiting_boundary_decision:
            raise RuntimeError("cannot advance before resolving the schedule boundary")
        stage = self.stages[self.stage_index]
        self.global_update += 1
        self.stage_update += 1
        boundary = self.stage_update >= self._current_stage_limit(stage)
        if boundary and self.config.accelerated:
            self._awaiting_boundary_decision = True
        elif boundary:
            self.stage_index += 1
            self.stage_update = 0
        return boundary

    def approve_extension(self, event: Mapping[str, Any]) -> None:
        if not self.config.accelerated or not self._awaiting_boundary_decision:
            raise RuntimeError("no accelerated schedule boundary awaits extension")
        stage = self.stages[self.stage_index]
        current = self._current_stage_limit(stage)
        if current >= int(stage.budget.hard_max_updates):
            raise RuntimeError("cannot extend a stage beyond its hard maximum")
        self.extension_blocks[stage.key] += 1
        if self._current_stage_limit(stage) <= current:
            raise RuntimeError("approved extension did not increase the stage limit")
        self.schedule_events.append(dict(event))
        self._awaiting_boundary_decision = False

    def finish_accelerated_stage(
        self, outcome: str, event: Mapping[str, Any], *, schedule_truncated: bool
    ) -> None:
        if not self.config.accelerated or not self._awaiting_boundary_decision:
            raise RuntimeError("no accelerated schedule boundary awaits completion")
        stage = self.stages[self.stage_index]
        self.stage_outcomes[stage.key] = str(outcome)
        self.schedule_events.append(dict(event))
        self.schedule_truncated = bool(
            self.schedule_truncated or schedule_truncated
        )
        self.stage_index += 1
        self.stage_update = 0
        self._awaiting_boundary_decision = False

    def finish_stage_early(self) -> None:
        if self.complete:
            raise RuntimeError("cannot finish a completed curriculum")
        if self.config.accelerated:
            raise RuntimeError("accelerated stages use deterministic boundary decisions")
        self.stage_index += 1
        self.stage_update = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
            "config": self.config.to_dict(),
            "stage_index": int(self.stage_index),
            "stage_update": int(self.stage_update),
            "global_update": int(self.global_update),
            "extension_blocks": dict(self.extension_blocks),
            "stage_outcomes": dict(self.stage_outcomes),
            "schedule_events": list(self.schedule_events),
            "schedule_truncated": bool(self.schedule_truncated),
            "awaiting_boundary_decision": bool(self._awaiting_boundary_decision),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("contract") != ABPH_RECONSTRUCTOR_TRAINING_CONTRACT:
            raise ValueError("curriculum checkpoint contract mismatch")
        saved_config = ReconstructorCurriculumConfig.from_dict(
            dict(payload.get("config", {}))
        )
        if saved_config.to_dict() != self.config.to_dict():
            raise ValueError("curriculum checkpoint configuration mismatch")
        stage_index = int(payload["stage_index"])
        stage_update = int(payload["stage_update"])
        global_update = int(payload["global_update"])
        if not 0 <= stage_index <= len(self.stages):
            raise ValueError("checkpoint stage index is out of range")
        if stage_index < len(self.stages):
            stage = self.stages[stage_index]
            blocks = {
                str(name): int(value)
                for name, value in dict(payload.get("extension_blocks", {})).items()
            }
            current_blocks = int(blocks.get(stage.key, 0))
            current_limit = min(
                int(stage.budget.nominal_updates)
                + current_blocks * int(stage.budget.extension_updates),
                int(stage.budget.hard_max_updates),
            )
            if not 0 <= stage_update < current_limit:
                raise ValueError("checkpoint stage update is out of range")
        elif stage_update != 0:
            raise ValueError("completed curriculum must have zero stage update")
        if global_update < 0:
            raise ValueError("checkpoint global update is negative")
        self.stage_index = stage_index
        self.stage_update = stage_update
        self.global_update = global_update
        saved_blocks = dict(payload.get("extension_blocks", {}))
        self.extension_blocks = {
            stage.key: int(saved_blocks.get(stage.key, 0)) for stage in self.stages
        }
        self.stage_outcomes = {
            str(name): str(value)
            for name, value in dict(payload.get("stage_outcomes", {})).items()
        }
        self.schedule_events = [
            dict(value) for value in payload.get("schedule_events", [])
        ]
        self.schedule_truncated = bool(payload.get("schedule_truncated", False))
        self._awaiting_boundary_decision = bool(
            payload.get("awaiting_boundary_decision", False)
        )
        if self._awaiting_boundary_decision:
            raise ValueError("checkpoints may not persist an unresolved schedule boundary")


@dataclass(frozen=True)
class ReconstructorStepContext:
    curriculum: CurriculumState
    split: str
    mode: str
    validation: bool
    teacher_forcing_probability: float
    stochastic_seed: int | None = None
    runtime_profiler: RuntimeProfiler | None = field(
        default=None, compare=False, repr=False
    )

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
        terms["particle_feature"] = (
            torch.stack(components).mean()
            if components
            else particle_matching.real_particle_loss
        )
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
            for capacity in state.supervised_capacities
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
        self.shadow = {
            name: value.detach().to(self.shadow[name].device).clone()
            for name, value in shadow.items()
        }


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
        if not parameters and not bool(getattr(module_groups[name], "allow_empty", False)):
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
                "shared_across_depths": bool(
                    getattr(module_groups[name], "shared_across_depths", False)
                ),
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


def _group_phase_factor(
    group_name: str,
    state: CurriculumState,
    *,
    shared_across_depths: bool = False,
) -> float:
    if group_name == "hlt_encoder":
        return 1.0 if state.phase == 1 else 0.25
    if group_name == "root":
        return 1.0 if state.phase == 1 else 0.25
    if group_name.startswith("hierarchy_"):
        if shared_across_depths:
            return 1.0 if state.phase == 2 else (0.25 if state.phase > 2 else 0.0)
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
        phase_factor = _group_phase_factor(
            name,
            state,
            shared_across_depths=bool(group.get("shared_across_depths", False)),
        )
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
    runtime_profiler: RuntimeProfiler | None = None,
    distributed_runtime: DistributedRuntime | None = None,
    validation_owner: Any | None = None,
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
        runtime_profiler=runtime_profiler,
    )
    runtime = distributed_runtime or DistributedRuntime(0, 1, 0, "none", device.type)
    accumulator = TypedValidationAccumulator()
    effective_weights: dict[str, float] | None = None
    iterator = iter(batches)
    transfer_stager = BatchTransferStager(
        device, non_blocking=bool(config.nonblocking_transfer)
    )
    local_error: BaseException | None = None
    try:
        with torch.no_grad():
            while True:
                try:
                    with profile_span(runtime_profiler, "validation_source_wait"):
                        cpu_batch = next(iterator)
                except StopIteration:
                    break
                with profile_span(runtime_profiler, "pinned_memory_staging"):
                    prepared_batch = prepare_contiguous_cpu_batch(
                        cpu_batch,
                        pin_memory=bool(config.pin_memory and device.type == "cuda"),
                    )
                with profile_span(runtime_profiler, "host_to_device"):
                    staged_batch = transfer_stager.stage(prepared_batch)
                    batch = staged_batch.wait()
                with profile_span(runtime_profiler, "model_step_total"):
                    with _autocast(device, config):
                        result = step_function(model, batch, context)
                with profile_span(runtime_profiler, "loss_composition"):
                    with torch.autocast(device_type=device.type, enabled=False):
                        composed = compose_reconstruction_loss(
                            result, context, config.loss_weights
                        )
                weight = int(result.batch_size)
                current_effective_weights = dict(composed.effective_weights)
                if effective_weights is None:
                    effective_weights = current_effective_weights
                elif current_effective_weights != effective_weights:
                    raise RuntimeError(
                        "model-val objective weights changed within one evaluation"
                    )
                additive_row = {
                    "loss.total": float(composed.total.detach().float().cpu()),
                    **{
                        f"loss.raw.{name}": float(value.detach().float().cpu())
                        for name, value in composed.raw_terms.items()
                    },
                    **{
                        f"loss.weighted.{name}": float(value.detach().float().cpu())
                        for name, value in composed.weighted_terms.items()
                    },
                }
                for name, value in additive_row.items():
                    accumulator.add_mean(
                        name,
                        value,
                        weight,
                        selection_eligible=name == "loss.total",
                    )
                for name, value in _scalar_metrics(result.metrics).items():
                    accumulator.add_non_additive(name, value, weight)
                accumulator.finish_batch(weight)
        if effective_weights is None:
            raise RuntimeError("rollout model validation produced no batches")
    except BaseException as exc:
        local_error = exc
    validation_succeeded = all_reduce_min_bool(
        runtime, local_error is None, device=device
    )
    if not validation_succeeded:
        errors = gather_error_summaries(
            runtime,
            phase="distributed_validation",
            error=local_error,
            structural=True,
        )
        raise RuntimeError(
            "distributed validation failed before reduction: "
            + json.dumps(list(errors), sort_keys=True)
        )
    assert effective_weights is not None
    validation_range = getattr(validation_owner, "last_validation_range", None)
    hlt_view = getattr(validation_owner, "hlt_view", None)
    expected_jet_ids = None if hlt_view is None else tuple(hlt_view.jet_ids)
    reduced = finalize_typed_validation(
        accumulator,
        runtime=runtime,
        device=device,
        required_losses=active_reconstruction_loss_names(context),
        effective_weights=effective_weights,
        validation_range=validation_range,
        expected_jet_ids=expected_jet_ids,
        split="model_val",
    )
    return {
        "contract": ABPH_DISTRIBUTED_VALIDATION_CONTRACT,
        "training_contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
        "split": "model_val",
        "mode": "rollout",
        "teacher_forcing_probability": 0.0,
        "offline_targets_loaded": True,
        "teacher_logits_loaded": False,
        "selection_metric": "model_val.rollout.loss.total",
        **reduced,
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
    distributed_runtime: Mapping[str, Any] | None = None,
    rank_runtime_states: Sequence[Mapping[str, Any]] | None = None,
    runtime_profile_hash: str | None = None,
) -> dict[str, Any]:
    selected = role in {"best_stage_model_val", "best_model_val"}
    rows = [dict(row) for row in (rank_runtime_states or ())]
    if not rows:
        rows = [
            {
                "rank": 0,
                "train_source_state_dict": dict(train_source.state_dict()),
                "rng_state": _rng_state(),
            }
        ]
    rows.sort(key=lambda row: int(row["rank"]))
    if [int(row["rank"]) for row in rows] != list(range(len(rows))):
        raise ValueError("checkpoint rank state rows are incomplete or reordered")
    global_cursors = [
        dict(row["train_source_state_dict"]).get("global_cursor") for row in rows
    ]
    if global_cursors[0] is not None and any(
        cursor != global_cursors[0] for cursor in global_cursors
    ):
        raise ValueError("checkpoint ranks disagree on the committed global cursor")
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
        # Retained for single-rank checkpoint-reader compatibility.
        "train_source_state_dict": dict(rows[0]["train_source_state_dict"]),
        "rng_state": rows[0]["rng_state"],
        "distributed_checkpoint_state": {
            "contract": ABPH_DISTRIBUTED_CHECKPOINT_CONTRACT,
            "world_size": len(rows),
            "rank_states": rows,
            "global_source_cursor": global_cursors[0],
            "source_state_semantics": "committed_cursor_after_successful_update",
            "rng_state_semantics": "rank_local_python_numpy_torch_cuda",
        },
        "config": config.to_dict(),
        "schedule_contract": config.curriculum.schedule_contract,
        "campaign_schedule_profile": config.curriculum.campaign_schedule_profile,
        "model_metadata": _jsonable(model_metadata),
        "validation": _jsonable(validation),
        "provenance": _jsonable(provenance),
        "optimizer_groups": _jsonable(optimizer_metadata),
        "nonfinite_updates": int(nonfinite_updates),
        "trainer_state": _jsonable(trainer_state),
        "distributed_runtime": _jsonable(distributed_runtime or {}),
        "runtime_contracts": {
            "distributed_runtime_contract": (
                dict(distributed_runtime or {}).get("contract")
            ),
            "runtime_batch_contract_hash": config.runtime_batch_contract_hash,
            "runtime_profile_snapshot_hash": runtime_profile_hash,
        },
        "final_test_loaded": False,
        "teacher_logits_loaded": False,
        "exact_resume_supported": True,
        "restart_semantics": ABPH_RESTART_SEMANTICS_EXACT,
    }


def _compact_selected_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a full selected training payload onto the persistent contract."""

    if payload.get("checkpoint_role") not in {"best_stage_model_val", "best_model_val"}:
        raise ValueError("only selected reconstructor payloads may be compacted")
    config = dict(payload.get("config", {}))
    config_hash = str(config.get("config_hash", ""))
    if not config_hash:
        raise ValueError("selected reconstructor payload lacks its resolved config hash")
    return build_compact_selected_checkpoint(
        model_state_dict=payload["model_state_dict"],
        checkpoint_role=str(payload["checkpoint_role"]),
        model_metadata=dict(payload.get("model_metadata", {})),
        resolved_variant_config=config,
        resolved_variant_config_hash=config_hash,
        validation=payload.get("validation"),
        provenance=dict(payload.get("provenance", {})),
        runtime_contracts=dict(payload.get("runtime_contracts", {})),
        schedule_contracts={
            "schedule_contract": payload.get("schedule_contract"),
            "campaign_schedule_profile": payload.get("campaign_schedule_profile"),
        },
        extra_metadata={
            "training_contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
            "optimizer_groups": payload.get("optimizer_groups", []),
            "distributed_runtime": payload.get("distributed_runtime", {}),
            "nonfinite_updates": int(payload.get("nonfinite_updates", 0)),
            "teacher_logits_loaded": False,
        },
    )


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
    contract = payload.get("checkpoint_contract")
    if contract == ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT:
        validate_compact_selected_checkpoint(payload, require_selected=True)
    elif contract not in {
        ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
        ABPH_EPHEMERAL_RESUME_CHECKPOINT_CONTRACT,
    }:
        raise ValueError("reconstructor checkpoint contract mismatch")
    if require_selected and payload.get("checkpoint_role") not in {
        "best_stage_model_val",
        "best_model_val",
    }:
        raise ValueError("a model-val selected reconstructor checkpoint is required")
    if payload.get("final_test_loaded") is not False:
        raise ValueError("reconstructor checkpoint does not attest final-test isolation")
    return payload


def _collective_checkpoint_load(
    path: str | Path,
    *,
    runtime: DistributedRuntime,
    require_selected: bool = False,
) -> Mapping[str, Any]:
    """Have rank zero validate one checkpoint, then broadcast that exact payload."""

    local: Mapping[str, Any] | None = None
    error: str | None = None
    if runtime.is_primary:
        try:
            local = load_reconstructor_curriculum_checkpoint(
                path, device="cpu", require_selected=require_selected
            )
        except BaseException as exc:  # broadcast failure instead of stranding peers
            error = f"{type(exc).__name__}: {exc}"
    control = broadcast_object(runtime, {"error": error, "payload": local})
    if control["error"] is not None:
        raise RuntimeError(f"rank-zero checkpoint load failed: {control['error']}")
    payload = control["payload"]
    if not isinstance(payload, Mapping):
        raise RuntimeError("broadcast checkpoint payload is missing")
    return payload


def _rank_state_for_resume(
    payload: Mapping[str, Any], runtime: DistributedRuntime
) -> Mapping[str, Any]:
    distributed = dict(payload.get("distributed_checkpoint_state", {}))
    if runtime.world_size == 1 and not distributed:
        return {
            "rank": 0,
            "train_source_state_dict": payload["train_source_state_dict"],
            "rng_state": payload["rng_state"],
        }
    if distributed.get("contract") != ABPH_DISTRIBUTED_CHECKPOINT_CONTRACT:
        raise ValueError("distributed resume checkpoint state contract mismatch")
    if int(distributed.get("world_size", -1)) != runtime.world_size:
        raise ValueError("distributed resume checkpoint world size mismatch")
    rows = [dict(row) for row in distributed.get("rank_states", ())]
    if [int(row.get("rank", -1)) for row in rows] != list(range(runtime.world_size)):
        raise ValueError("distributed resume checkpoint rank states are incomplete")
    local = rows[runtime.rank]
    saved_runtime = dict(local.get("runtime", {}))
    if saved_runtime and (
        int(saved_runtime.get("rank", -1)) != runtime.rank
        or int(saved_runtime.get("world_size", -1)) != runtime.world_size
        or str(saved_runtime.get("backend")) != runtime.backend
    ):
        raise ValueError("distributed resume rank topology mismatch")
    return local


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
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    streaming_checkpoints = streaming_storage_enabled()
    campaign_root = (
        output_dir.parent.parent
        if output_dir.parent.name == "runs"
        else output_dir.parent
    )
    checkpoint_variant_name = output_dir.name
    device = resolve_device(config.device)
    distributed_runtime = initialize_distributed_runtime(
        requested_world_size=int(config.distributed_world_size), device=device
    )
    set_training_seed(int(config.seed) + int(distributed_runtime.rank))
    for source_name, source in (("train", train_source),):
        source_rank = getattr(source, "rank", distributed_runtime.rank)
        source_world = getattr(source, "world_size", distributed_runtime.world_size)
        if (
            int(source_rank) != distributed_runtime.rank
            or int(source_world) != distributed_runtime.world_size
        ):
            raise ValueError(
                f"{source_name} source rank topology differs from the distributed runtime"
            )
    if distributed_runtime.distributed and not bool(config.asynchronous_prefetch):
        raise ValueError("distributed ABPH training requires deferred-commit prefetch")
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
    profiler_output_dir = (
        output_dir
        if distributed_runtime.is_primary
        else output_dir / "distributed_runtime" / f"rank_{distributed_runtime.rank:04d}"
    )
    runtime_profiler = RuntimeProfiler(
        config.runtime_profile,
        device=device,
        output_dir=profiler_output_dir,
        provenance=source_provenance,
        training_config_hash=config_hash,
        model_metadata_hash=model_metadata.get("model_metadata_hash"),
    )
    if hasattr(train_source, "set_runtime_profiler"):
        train_source.set_runtime_profiler(runtime_profiler)
    if hasattr(train_source, "set_plan_log_dir"):
        train_source.set_plan_log_dir(output_dir / "global_batch_plans")
    validation_owner = getattr(validation_batches, "__self__", None)
    if validation_owner is not None:
        validation_rank = getattr(validation_owner, "rank", distributed_runtime.rank)
        validation_world = getattr(
            validation_owner, "world_size", distributed_runtime.world_size
        )
        if (
            int(validation_rank) != distributed_runtime.rank
            or int(validation_world) != distributed_runtime.world_size
        ):
            raise ValueError(
                "validation source rank topology differs from the distributed runtime"
            )
    if hasattr(validation_owner, "set_runtime_profiler"):
        validation_owner.set_runtime_profiler(runtime_profiler)
    nonfinite_updates = 0
    compiler_failure_updates = 0
    objective_skip_counts = {name: 0 for name in ABPH_RECONSTRUCTION_LOSS_NAMES}
    curves: list[dict[str, Any]] = []
    best_by_stage: dict[str, dict[str, Any]] = {}
    evaluations_without_improvement: dict[str, int] = {}
    distributed_error_events: list[dict[str, Any]] = []

    if resume_from is not None:
        payload = _collective_checkpoint_load(
            resume_from, runtime=distributed_runtime
        )
        require_exact_resume_checkpoint(payload)
        saved_distributed = dict(payload.get("distributed_runtime", {}))
        if saved_distributed and (
            int(saved_distributed.get("world_size", -1))
            != distributed_runtime.world_size
            or str(saved_distributed.get("backend")) != distributed_runtime.backend
        ):
            raise ValueError("resume checkpoint distributed runtime mismatch")
        if _normalized_trainer_config(
            dict(payload.get("config", {}))
        ) != _normalized_trainer_config(config.to_dict()):
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
        local_resume_state = _rank_state_for_resume(payload, distributed_runtime)
        train_source.load_state_dict(local_resume_state["train_source_state_dict"])
        _restore_rng_state(local_resume_state["rng_state"])
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
        compiler_failure_updates = int(
            saved_trainer_state.get("compiler_failure_updates", 0)
        )
        distributed_error_events = [
            dict(value)
            for value in saved_trainer_state.get("distributed_error_events", ())
        ]

    transfer_stager = BatchTransferStager(
        device, non_blocking=bool(config.nonblocking_transfer)
    )
    prefetcher: RankLocalBatchPrefetcher | None = None
    if bool(config.asynchronous_prefetch) and all(
        hasattr(train_source, name)
        for name in (
            "current_cursor",
            "derive_next_plan",
            "agree_plan_hash",
            "prepare_planned_batch",
            "commit_planned_batch",
        )
    ):
        prefetcher = RankLocalBatchPrefetcher(
            train_source,
            queue_depth=int(config.prefetch_queue_depth),
            pin_memory=bool(config.pin_memory and device.type == "cuda"),
        )
    if distributed_runtime.distributed and prefetcher is None:
        raise TypeError(
            "distributed ABPH training requires a planned deferred-commit target source"
        )
    training_module = ReconstructorTrainingModule(
        model,
        step_function,
        compose_reconstruction_loss,
        config.loss_weights,
    )
    stage_wrapper: Any | None = None
    wrapper_stage_key: str | None = None

    def current_trainer_state() -> dict[str, Any]:
        return {
            "curves": curves,
            "best_by_stage": best_by_stage,
            "evaluations_without_improvement": evaluations_without_improvement,
            "objective_skip_counts": objective_skip_counts,
            "compiler_failure_updates": int(compiler_failure_updates),
            "distributed_error_events": distributed_error_events,
        }

    def collective_checkpoint_payload(
        *,
        role: str,
        validation: Mapping[str, Any] | None,
        optimizer_metadata: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any] | None:
        rank_states = all_gather_objects(
            distributed_runtime,
            {
                "rank": distributed_runtime.rank,
                "runtime": distributed_runtime.to_dict(),
                "train_source_state_dict": dict(train_source.state_dict()),
                "rng_state": _rng_state(),
            },
        )
        payload: Mapping[str, Any] | None = None
        payload_error: str | None = None
        if distributed_runtime.is_primary:
            try:
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    ema=ema,
                    controller=controller,
                    train_source=train_source,
                    config=config,
                    role=role,
                    validation=validation,
                    provenance=source_provenance,
                    optimizer_metadata=optimizer_metadata,
                    nonfinite_updates=nonfinite_updates,
                    trainer_state=current_trainer_state(),
                    model_metadata=model_metadata,
                    distributed_runtime=distributed_runtime.to_dict(),
                    rank_runtime_states=rank_states,
                    runtime_profile_hash=runtime_profiler.payload()[
                        "profile_content_hash"
                    ],
                )
                if streaming_checkpoints:
                    payload["checkpoint_contract"] = (
                        ABPH_EPHEMERAL_RESUME_CHECKPOINT_CONTRACT
                    )
                    payload["storage_profile"] = "streaming_30gb_v1"
                    payload["persistent"] = False
            except BaseException as exc:
                payload_error = f"{type(exc).__name__}: {exc}"
        payload_error = broadcast_object(distributed_runtime, payload_error)
        if payload_error is not None:
            raise RuntimeError(
                f"rank-zero checkpoint assembly failed: {payload_error}"
            )
        return payload

    def write_checkpoint_collective(
        path: Path,
        payload: Mapping[str, Any] | None,
        *,
        aliases: Sequence[Path] = (),
    ) -> str:
        local_error: str | None = None
        if distributed_runtime.is_primary:
            try:
                if payload is None:
                    raise RuntimeError("rank zero did not build a checkpoint payload")
                _atomic_torch_save(path, payload)
                for alias in aliases:
                    if streaming_checkpoints:
                        compact = _compact_selected_payload(payload)
                        write_selected_checkpoint(
                            alias,
                            compact,
                            campaign_root=campaign_root,
                            artifact_role="selected_reconstructor_checkpoint",
                            run_id=checkpoint_variant_name,
                        )
                    else:
                        _atomic_torch_save(alias, payload)
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
        error = broadcast_object(distributed_runtime, local_error)
        if error is not None:
            raise RuntimeError(f"rank-zero checkpoint write failed: {error}")
        barrier(distributed_runtime)
        digest_path = aliases[0] if aliases and streaming_checkpoints else path
        digest = _file_sha256(digest_path) if distributed_runtime.is_primary else None
        digest = broadcast_object(distributed_runtime, digest)
        if not isinstance(digest, str) or not digest:
            raise RuntimeError("checkpoint content hash broadcast failed")
        return digest

    if distributed_runtime.is_primary:
        _atomic_json(
            output_dir / "config.json",
            {
            "contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
            "config": config.to_dict(),
            "distributed_runtime": distributed_runtime.to_dict(),
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
    barrier(distributed_runtime)

    updates_this_call = 0
    while not controller.complete:
        if maximum_optimizer_updates is not None and updates_this_call >= int(maximum_optimizer_updates):
            break
        state = controller.state()
        optimizer_metadata = configure_reconstructor_optimizer(optimizer, state, config)
        if stage_wrapper is None or wrapper_stage_key != state.stage_key:
            if stage_wrapper is not None:
                barrier(distributed_runtime)
                del stage_wrapper
                stage_wrapper = None
            stage_wrapper = build_stage_ddp_wrapper(
                training_module,
                distributed_runtime,
                device=device,
                find_unused_parameters=bool(config.ddp_find_unused_parameters),
            )
            verify_common_parameter_state(distributed_runtime, model)
            barrier(distributed_runtime)
            wrapper_stage_key = state.stage_key
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
        global_batch_plan_hashes: list[str] = []
        active_parameters_for_update = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        accumulation_steps = int(
            (
                config.root_hierarchy_gradient_accumulation_steps
                if state.phase <= 2
                else config.renderer_distribution_gradient_accumulation_steps
            )
            or config.gradient_accumulation_steps
        )
        local_batch_size = (
            config.root_hierarchy_local_batch_size
            if state.phase <= 2
            else config.renderer_distribution_local_batch_size
        )
        if local_batch_size is not None:
            if not hasattr(train_source, "set_batch_size"):
                raise TypeError(
                    "phase-specific runtime contract requires a resizable train source"
                )
            train_source.set_batch_size(int(local_batch_size))
        runtime_profiler.begin_train_update(
            state,
            local_batch_size=getattr(train_source, "batch_size", None),
            accumulation_steps=accumulation_steps,
            distributed_world_size=int(config.distributed_world_size),
        )
        update_requests = tuple(
            (int(state.global_update), int(index))
            for index in range(accumulation_steps)
        )
        next_update_requests = (
            ()
            if will_evaluate
            else tuple(
                (int(state.global_update) + 1, int(index))
                for index in range(accumulation_steps)
            )
        )
        pipeline_requests = update_requests + next_update_requests
        if prefetcher is not None:
            prefetcher.prime(pipeline_requests)
        for accumulation_index in range(accumulation_steps):
            context = ReconstructorStepContext(
                curriculum=state,
                split="model_train",
                mode=_training_mode(state, int(config.seed)),
                validation=False,
                teacher_forcing_probability=float(state.teacher_forcing_probability),
                stochastic_seed=(
                    int(config.seed) + int(distributed_runtime.rank)
                ),
                runtime_profiler=runtime_profiler,
            )
            cpu_batch = None
            batch = None
            input_error: BaseException | None = None
            try:
                with profile_span(runtime_profiler, "target_source_wait"):
                    if prefetcher is not None:
                        cpu_batch = prefetcher.next_planned_batch(
                            global_update=int(state.global_update),
                            accumulation_index=int(accumulation_index),
                        )
                    elif hasattr(train_source, "next_planned_batch"):
                        cpu_batch = train_source.next_planned_batch(
                            global_update=int(state.global_update),
                            accumulation_index=int(accumulation_index),
                        )
                    else:
                        cpu_batch = train_source.next_batch()
                if prefetcher is not None:
                    timing = prefetcher.last_timing
                    if timing is not None:
                        runtime_profiler.record_cpu_duration(
                            "target_shard_decompression",
                            timing.target_shard_decompression_seconds,
                        )
                        runtime_profiler.record_cpu_duration(
                            "cpu_batch_assembly",
                            max(
                                0.0,
                                timing.source_prepare_seconds
                                - timing.target_shard_decompression_seconds,
                            ),
                        )
                        runtime_profiler.record_cpu_duration(
                            "pinned_memory_staging", timing.pinned_staging_seconds
                        )
                    prefetcher.prime(
                        pipeline_requests[accumulation_index + 1 :]
                    )
                plan_hash = (
                    cpu_batch.get("global_batch_plan_hash")
                    if isinstance(cpu_batch, Mapping)
                    else None
                )
                if plan_hash is not None:
                    global_batch_plan_hashes.append(str(plan_hash))
                    runtime_profiler.record_global_batch_plan_hash(str(plan_hash))
                with profile_span(runtime_profiler, "host_to_device"):
                    if prefetcher is None:
                        with profile_span(runtime_profiler, "pinned_memory_staging"):
                            cpu_batch = prepare_contiguous_cpu_batch(
                                cpu_batch,
                                pin_memory=bool(
                                    config.pin_memory and device.type == "cuda"
                                ),
                            )
                    staged_batch = transfer_stager.stage(cpu_batch)
                    batch = staged_batch.wait()
            except BaseException as exc:
                input_error = exc
            globally_ready = all_reduce_min_bool(
                distributed_runtime, input_error is None, device=device
            )
            if not globally_ready:
                errors = gather_error_summaries(
                    distributed_runtime,
                    phase="input_readiness",
                    error=input_error,
                    structural=True,
                )
                distributed_error_events.append(
                    {
                        "global_update": int(state.global_update),
                        "accumulation_index": int(accumulation_index),
                        "phase": "input_readiness",
                        "errors": list(errors),
                    }
                )
                optimizer.zero_grad(set_to_none=True)
                if prefetcher is not None:
                    prefetcher.reset_to_committed_cursor()
                runtime_profiler.end_train_update(
                    success=False, jets=accumulated_jets
                )
                raise RuntimeError(
                    "distributed ABPH input readiness failed: "
                    + json.dumps(list(errors), sort_keys=True)
                ) from input_error

            synchronize_backward = accumulation_index + 1 == accumulation_steps
            synchronization_context = (
                nullcontext()
                if not distributed_runtime.distributed or synchronize_backward
                else stage_wrapper.no_sync()
            )
            forward_error: BaseException | None = None
            ddp_output: Mapping[str, Any] | None = None
            try:
                with synchronization_context:
                    try:
                        with profile_span(runtime_profiler, "model_step_total"):
                            with _autocast(device, config):
                                ddp_output = require_standard_tensor_mapping(
                                    stage_wrapper(batch, context)
                                )
                        if not tensor_mapping_is_finite(ddp_output):
                            raise FloatingPointError(
                                "DDP reconstructor forward produced nonfinite tensors"
                            )
                    except BaseException as exc:
                        forward_error = exc
                    globally_forward_ok = all_reduce_min_bool(
                        distributed_runtime, forward_error is None, device=device
                    )
                    if not globally_forward_ok:
                        errors = gather_error_summaries(
                            distributed_runtime,
                            phase="forward",
                            error=forward_error,
                            structural=(
                                forward_error is not None
                                and not isinstance(forward_error, FloatingPointError)
                            ),
                        )
                        structural = any_structural_error(errors)
                        distributed_error_events.append(
                            {
                                "global_update": int(state.global_update),
                                "accumulation_index": int(accumulation_index),
                                "phase": "forward",
                                "structural": structural,
                                "errors": list(errors),
                            }
                        )
                        optimizer.zero_grad(set_to_none=True)
                        if prefetcher is not None:
                            prefetcher.reset_to_committed_cursor()
                        update_failed = True
                        if structural:
                            runtime_profiler.end_train_update(
                                success=False, jets=accumulated_jets
                            )
                            raise RuntimeError(
                                "distributed ABPH structural forward failure: "
                                + json.dumps(list(errors), sort_keys=True)
                            ) from forward_error
                        nonfinite_updates += 1
                        message = " ".join(
                            str(row.get("message") or "") for row in errors
                        )
                        if (
                            "accounting/projection" in message
                            or "compiler" in message
                        ):
                            compiler_failure_updates += 1
                        for name in ABPH_RECONSTRUCTION_LOSS_NAMES:
                            if f"loss term {name} " in message:
                                objective_skip_counts[name] += 1
                                break
                        if nonfinite_updates > int(config.maximum_nonfinite_updates):
                            runtime_profiler.end_train_update(
                                success=False, jets=accumulated_jets
                            )
                            raise RuntimeError(
                                "reconstructor exceeded the maximum synchronized "
                                "nonfinite updates"
                            ) from forward_error
                        raise _SynchronizedForwardRetry from forward_error

                    metadata = training_module.last_metadata
                    if metadata is None or ddp_output is None:
                        raise RuntimeError("DDP forward omitted its local metadata contract")
                    composed = ComposedReconstructionLoss(
                        total=ddp_output["total_loss"],
                        raw_terms=ddp_output["raw_loss_terms"],
                        effective_weights=metadata["effective_weights"],
                        weighted_terms=ddp_output["weighted_loss_terms"],
                        required_terms=tuple(metadata["required_terms"]),
                    )
                    training_required_losses = composed.required_terms
                    scaled_loss = composed.total / float(accumulation_steps)
                    if will_evaluate:
                        with profile_span(
                            runtime_profiler, "objective_gradient_diagnostics"
                        ):
                            objective_norms = _objective_gradient_norms(
                                composed, active_parameters_for_update
                            )
                        for name, value in objective_norms.items():
                            objective_gradient_sums[name] = (
                                objective_gradient_sums.get(name, 0.0) + value
                            )
                    backward_error: BaseException | None = None
                    try:
                        with profile_span(runtime_profiler, "backward"):
                            synchronization_span = (
                                profile_span(
                                    runtime_profiler, "gradient_synchronization"
                                )
                                if distributed_runtime.distributed
                                and synchronize_backward
                                else nullcontext()
                            )
                            with synchronization_span:
                                scaler.scale(scaled_loss).backward()
                    except BaseException as exc:
                        backward_error = exc
                    if backward_error is not None:
                        distributed_error_events.append(
                            {
                                "global_update": int(state.global_update),
                                "accumulation_index": int(accumulation_index),
                                "phase": "backward_abort",
                                "structural": True,
                                "errors": [
                                    {
                                        "rank": distributed_runtime.rank,
                                        "phase": "backward",
                                        "structural": True,
                                        "error_type": type(backward_error).__name__,
                                        "message": str(backward_error)[:1024],
                                    }
                                ],
                            }
                        )
                        optimizer.zero_grad(set_to_none=True)
                        if prefetcher is not None:
                            prefetcher.reset_to_committed_cursor()
                        runtime_profiler.end_train_update(
                            success=False, jets=accumulated_jets
                        )
                        abort_distributed_runtime(distributed_runtime)
                        raise RuntimeError(
                            "distributed ABPH rank-local backward failure; "
                            "the process group was aborted"
                        ) from backward_error
                    try:
                        globally_backward_ok = all_reduce_min_bool(
                            distributed_runtime,
                            True,
                            device=device,
                        )
                    except BaseException as exc:
                        abort_distributed_runtime(distributed_runtime)
                        raise RuntimeError(
                            "distributed ABPH peer failed during backward; "
                            "the process group was aborted"
                        ) from exc
                    if not globally_backward_ok:
                        errors = gather_error_summaries(
                            distributed_runtime,
                            phase="backward",
                            error=backward_error,
                            structural=True,
                        )
                        distributed_error_events.append(
                            {
                                "global_update": int(state.global_update),
                                "accumulation_index": int(accumulation_index),
                                "phase": "backward",
                                "structural": True,
                                "errors": list(errors),
                            }
                        )
                        optimizer.zero_grad(set_to_none=True)
                        if prefetcher is not None:
                            prefetcher.reset_to_committed_cursor()
                        runtime_profiler.end_train_update(
                            success=False, jets=accumulated_jets
                        )
                        raise RuntimeError(
                            "distributed ABPH backward failure: "
                            + json.dumps(list(errors), sort_keys=True)
                        ) from backward_error
                    batch_size = int(ddp_output["batch_size_tensor"].detach().cpu())
                    accumulated_jets += batch_size
                    train_rows.append(
                        {
                            "loss.total": float(
                                composed.total.detach().float().cpu()
                            ),
                            **{
                                f"loss.raw.{name}": float(
                                    value.detach().float().cpu()
                                )
                                for name, value in composed.raw_terms.items()
                            },
                            **{
                                f"loss.normalized.{name}": float(
                                    value.detach().float().cpu()
                                )
                                for name, value in composed.raw_terms.items()
                            },
                            **{
                                f"loss.effective_weight.{name}": float(value)
                                for name, value in composed.effective_weights.items()
                            },
                        }
                    )
            except _SynchronizedForwardRetry:
                barrier(distributed_runtime)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                stage_wrapper = None
                gc.collect()
                barrier(distributed_runtime)
                stage_wrapper = build_stage_ddp_wrapper(
                    training_module,
                    distributed_runtime,
                    device=device,
                    find_unused_parameters=bool(config.ddp_find_unused_parameters),
                )
                verify_common_parameter_state(distributed_runtime, model)
                barrier(distributed_runtime)
                break
        if update_failed:
            if prefetcher is not None:
                prefetcher.reset_to_committed_cursor()
            runtime_profiler.end_train_update(success=False, jets=accumulated_jets)
            continue
        effective_batch_size = all_reduce_sum_int(
            distributed_runtime, accumulated_jets, device=device
        )
        expected_effective_batch_size = (
            int(config.root_hierarchy_effective_batch_size)
            if state.phase <= 2
            else int(config.renderer_distribution_effective_batch_size)
        )
        if (
            bool(config.enforce_effective_batch_size)
            and effective_batch_size != expected_effective_batch_size
        ):
            if prefetcher is not None:
                prefetcher.reset_to_committed_cursor()
            runtime_profiler.end_train_update(
                success=False, jets=accumulated_jets
            )
            raise ValueError(
                f"stage {state.stage_key} accumulated effective batch size "
                f"{effective_batch_size}, expected {expected_effective_batch_size}"
            )
        scaler.unscale_(optimizer)
        active_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradients = [
            parameter.grad
            for parameter in active_parameters
            if parameter.grad is not None
        ]
        local_gradients_finite = bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        globally_gradients_finite = all_reduce_min_bool(
            distributed_runtime, local_gradients_finite, device=device
        )
        if not globally_gradients_finite:
            errors = gather_error_summaries(
                distributed_runtime,
                phase="post_backward_gradients",
                error=(
                    None
                    if local_gradients_finite
                    else FloatingPointError("gradients are absent or nonfinite")
                ),
                structural=False,
            )
            distributed_error_events.append(
                {
                    "global_update": int(state.global_update),
                    "phase": "post_backward_gradients",
                    "structural": False,
                    "errors": list(errors),
                }
            )
            nonfinite_updates += 1
            optimizer.zero_grad(set_to_none=True)
            if prefetcher is not None:
                prefetcher.reset_to_committed_cursor()
            runtime_profiler.end_train_update(success=False, jets=accumulated_jets)
            if nonfinite_updates > int(config.maximum_nonfinite_updates):
                raise RuntimeError("reconstructor gradients remained absent/nonfinite")
            continue
        optimizer_group_gradient_norms = _optimizer_group_gradient_norms(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            active_parameters, float(config.gradient_clip_norm)
        )
        globally_gradient_norm_finite = all_reduce_min_bool(
            distributed_runtime,
            bool(torch.isfinite(gradient_norm)),
            device=device,
        )
        if not globally_gradient_norm_finite:
            errors = gather_error_summaries(
                distributed_runtime,
                phase="gradient_clip_norm",
                error=(
                    None
                    if bool(torch.isfinite(gradient_norm))
                    else FloatingPointError("gradient norm is nonfinite")
                ),
                structural=False,
            )
            distributed_error_events.append(
                {
                    "global_update": int(state.global_update),
                    "phase": "gradient_clip_norm",
                    "structural": False,
                    "errors": list(errors),
                }
            )
            nonfinite_updates += 1
            optimizer.zero_grad(set_to_none=True)
            if prefetcher is not None:
                prefetcher.reset_to_committed_cursor()
            runtime_profiler.end_train_update(success=False, jets=accumulated_jets)
            if nonfinite_updates > int(config.maximum_nonfinite_updates):
                raise RuntimeError("reconstructor gradient norm remained nonfinite")
            continue
        with profile_span(runtime_profiler, "optimizer_ema_update"):
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        if prefetcher is not None:
            committed_hashes = prefetcher.commit_consumed()
            if tuple(committed_hashes) != tuple(global_batch_plan_hashes):
                raise RuntimeError(
                    "committed global batch plans differ from the completed update"
                )
        runtime_profiler.end_train_update(
            success=True, jets=effective_batch_size
        )
        transitioned = controller.advance()
        updates_this_call += 1

        should_evaluate = (
            transitioned
            or controller.global_update % int(config.curriculum.evaluation_interval) == 0
        )
        if not should_evaluate:
            continue
        runtime_profiler.begin_validation(state)
        validation_local_batch_size = (
            config.root_hierarchy_local_batch_size
            if state.phase <= 2
            else config.renderer_distribution_local_batch_size
        )
        if validation_local_batch_size is not None and hasattr(
            validation_owner, "set_batch_size"
        ):
            validation_owner.set_batch_size(int(validation_local_batch_size))
        with ema.applied(model):
            verify_common_parameter_state(distributed_runtime, model)
            validation = evaluate_reconstructor_rollout(
                model,
                validation_batches(),
                state,
                step_function,
                config,
                device,
                runtime_profiler=runtime_profiler,
                distributed_runtime=distributed_runtime,
                validation_owner=validation_owner,
            )
        runtime_profiler.end_validation(
            jets=int(validation["n_jets"]), batches=int(validation["n_batches"])
        )
        runtime_profiler.write(persist=distributed_runtime.is_primary)
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
                "gradient_accumulation_steps": int(accumulation_steps),
                "distributed_runtime": distributed_runtime.to_dict(),
                "global_batch_plan_hashes": list(global_batch_plan_hashes),
                "mode": _training_mode(state, int(config.seed)),
                "required_losses": list(training_required_losses),
                "metrics": train_averages,
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "objective_gradient_norms": {
                    name: value / float(accumulation_steps)
                    for name, value in objective_gradient_sums.items()
                },
                "optimizer_group_gradient_norms": optimizer_group_gradient_norms,
                "objective_skip_counts": dict(objective_skip_counts),
                "nonfinite_updates": int(nonfinite_updates),
                "compiler_failure_updates": int(compiler_failure_updates),
            },
            "model_val_rollout": validation,
            "optimizer_groups": optimizer_metadata,
        }
        curves.append(curve)
        if validation.get("checkpoint_selection_eligible") is not True:
            raise RuntimeError(
                "model-val result is not eligible for checkpoint selection; "
                "distributed reduction/coverage must complete first"
            )
        score = float(validation["selection_score"])
        best = best_by_stage.get(state.stage_key)
        selection_decision = None
        if distributed_runtime.is_primary:
            selection_decision = {
                "stage_key": state.stage_key,
                "selection_score": score,
                "improved": best is None or score < float(best["selection_score"]),
                "previous_best": None
                if best is None
                else float(best["selection_score"]),
            }
        selection_decision = broadcast_object(
            distributed_runtime, selection_decision
        )
        if (
            selection_decision["stage_key"] != state.stage_key
            or float(selection_decision["selection_score"]) != score
        ):
            raise RuntimeError("rank-zero checkpoint selection decision is inconsistent")
        improved = bool(selection_decision["improved"])
        curve["checkpoint_selection_decision"] = selection_decision
        if improved:
            evaluations_without_improvement[state.stage_key] = 0
            stage_path = (
                ephemeral_checkpoint_path(
                    variant_name=checkpoint_variant_name,
                    filename=f"best_{state.stage_key}.pt",
                )
                if streaming_checkpoints
                else output_dir / f"best_{state.stage_key}.pt"
            )
            best_by_stage[state.stage_key] = {
                "selection_score": score,
                "global_update": int(controller.global_update),
                "checkpoint": str(stage_path),
                "checkpoint_storage": (
                    "rank0_ram_ephemeral"
                    if streaming_checkpoints
                    else "persistent_shared"
                ),
            }
            if streaming_checkpoints and controller.is_final_stage(state):
                best_by_stage[state.stage_key]["persistent_selected_checkpoint"] = str(
                    output_dir / "best_model_val.pt"
                )
        else:
            evaluations_without_improvement[state.stage_key] = (
                evaluations_without_improvement.get(state.stage_key, 0) + 1
            )

        stage_finished = False
        if config.curriculum.accelerated and transitioned:
            stage_definition = controller.stages[state.stage_index]
            stage_history = [
                row
                for row in curves
                if row["curriculum"]["stage_key"] == state.stage_key
            ]
            if stage_definition.role == "warm_started_handoff":
                schedule_event = {
                    "stage_key": state.stage_key,
                    "global_update": int(controller.global_update),
                    "stage_update": int(controller.stage_update),
                    "outcome": "warm_started_handoff",
                    "continue_training": False,
                    "schedule_truncated": False,
                    "checks": {"warm_started_handoff": True},
                }
                controller.finish_accelerated_stage(
                    "warm_started_handoff",
                    schedule_event,
                    schedule_truncated=False,
                )
                stage_finished = True
            else:
                decision = decide_stage_continuation(
                    stage_history,
                    required_objectives=validation["required_losses"],
                    best_checkpoint_global_update=best_by_stage[state.stage_key][
                        "global_update"
                    ],
                    stage_update=int(controller.stage_update),
                    nominal_updates=int(state.stage_nominal_updates),
                    extension_blocks_completed=int(state.stage_extension_blocks),
                    hard_max_updates=int(state.stage_hard_max_updates),
                    nonfinite_updates=int(nonfinite_updates),
                    compiler_failure_updates=int(compiler_failure_updates),
                    stage_role=stage_definition.role,
                )
                schedule_event = {
                    "stage_key": state.stage_key,
                    "global_update": int(controller.global_update),
                    "stage_update": int(controller.stage_update),
                    "nominal_updates": int(state.stage_nominal_updates),
                    "extension_updates": int(state.stage_extension_updates),
                    "hard_max_updates": int(state.stage_hard_max_updates),
                    "extension_blocks_before_decision": int(
                        state.stage_extension_blocks
                    ),
                    **decision.to_dict(),
                }
                if decision.continue_training:
                    controller.approve_extension(schedule_event)
                else:
                    controller.finish_accelerated_stage(
                        decision.outcome,
                        schedule_event,
                        schedule_truncated=decision.schedule_truncated,
                    )
                    stage_finished = True
            curve["schedule_decision"] = schedule_event

        if improved:
            payload = collective_checkpoint_payload(
                role=("best_model_val" if controller.is_final_stage(state) else "best_stage_model_val"),
                validation=validation,
                optimizer_metadata=optimizer_metadata,
            )
            with runtime_profiler.standalone_span(
                "checkpoint_serialization", stage_key=state.stage_key
            ):
                checkpoint_hash = write_checkpoint_collective(
                    stage_path,
                    payload,
                    aliases=(output_dir / "best_model_val.pt",)
                    if controller.is_final_stage(state)
                    else (),
                )
            best_by_stage[state.stage_key]["checkpoint_sha256"] = checkpoint_hash

        if config.save_last_checkpoint:
            payload = collective_checkpoint_payload(
                role="last",
                validation=validation,
                optimizer_metadata=optimizer_metadata,
            )
            with runtime_profiler.standalone_span(
                "checkpoint_serialization", stage_key=state.stage_key
            ):
                write_checkpoint_collective(
                    (
                        ephemeral_checkpoint_path(
                            variant_name=checkpoint_variant_name,
                            filename="last.pt",
                        )
                        if streaming_checkpoints
                        else output_dir / "last.pt"
                    ),
                    payload,
                )
        if distributed_runtime.is_primary:
            _atomic_json(
                output_dir / "training_curves.json",
                {
                "contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
                "selection_metric": "model_val.rollout.loss.total",
                "rollout_validation_required": True,
                "schedule_contract": config.curriculum.schedule_contract,
                "campaign_schedule_profile": (
                    config.curriculum.campaign_schedule_profile
                ),
                "evaluations": curves,
                },
            )
        barrier(distributed_runtime)

        early_transition = False
        if not config.curriculum.accelerated and not transitioned:
            stage_definition = controller.stages[state.stage_index]
            early_transition = (
                evaluations_without_improvement.get(state.stage_key, 0)
                >= int(stage_definition.patience_evaluations)
            )
            if early_transition:
                controller.finish_stage_early()
        if not config.curriculum.accelerated:
            stage_finished = bool(transitioned or early_transition)
        if stage_finished:
            selected_path = Path(best_by_stage[state.stage_key]["checkpoint"])
            selected = _collective_checkpoint_load(
                selected_path,
                runtime=distributed_runtime,
                require_selected=True,
            )
            model.load_state_dict(selected["model_state_dict"], strict=True)
            ema.reset_from(model)
            if streaming_checkpoints and distributed_runtime.is_primary:
                selected_path.unlink(missing_ok=True)
                best_by_stage[state.stage_key]["ephemeral_checkpoint_deleted"] = True
                if controller.is_final_stage(state):
                    best_by_stage[state.stage_key]["checkpoint"] = str(
                        output_dir / "best_model_val.pt"
                    )
            optimizer.state.clear()
            handoff_optimizer_metadata = (
                configure_reconstructor_optimizer(optimizer, controller.state(), config)
                if not controller.complete
                else optimizer_metadata
            )
            if config.save_last_checkpoint:
                payload = collective_checkpoint_payload(
                    role="last",
                    validation=validation,
                    optimizer_metadata=handoff_optimizer_metadata,
                )
                with runtime_profiler.standalone_span(
                    "checkpoint_serialization", stage_key=state.stage_key
                ):
                    write_checkpoint_collective(
                        (
                            ephemeral_checkpoint_path(
                                variant_name=checkpoint_variant_name,
                                filename="last.pt",
                            )
                            if streaming_checkpoints
                            else output_dir / "last.pt"
                        ),
                        payload,
                    )

    prefetch_maximum_resident_batches = (
        0 if prefetcher is None else int(prefetcher.maximum_resident_batches)
    )
    if prefetcher is not None:
        prefetcher.close()

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
        final_stage_key = (
            controller.state().stage_key if not controller.complete else "complete"
        )
        payload = collective_checkpoint_payload(
            role="last",
            validation=curves[-1]["model_val_rollout"] if curves else None,
            optimizer_metadata=optimizer_metadata,
        )
        with runtime_profiler.standalone_span(
            "checkpoint_serialization", stage_key=final_stage_key
        ):
            write_checkpoint_collective(
                (
                    ephemeral_checkpoint_path(
                        variant_name=checkpoint_variant_name,
                        filename="last.pt",
                    )
                    if streaming_checkpoints
                    else output_dir / "last.pt"
                ),
                payload,
            )
    complete = bool(controller.complete)
    terminal_checkpoint_present = (
        (output_dir / "best_model_val.pt").exists()
        if distributed_runtime.is_primary
        else None
    )
    terminal_checkpoint_present = broadcast_object(
        distributed_runtime, terminal_checkpoint_present
    )
    if complete and not terminal_checkpoint_present:
        raise RuntimeError("completed curriculum produced no terminal model-val checkpoint")
    if complete and streaming_checkpoints and distributed_runtime.is_primary:
        ephemeral_checkpoint_path(
            variant_name=checkpoint_variant_name, filename="last.pt"
        ).unlink(missing_ok=True)
    runtime_profile = runtime_profiler.write(persist=distributed_runtime.is_primary)
    runtime_profile = broadcast_object(
        distributed_runtime,
        runtime_profile if distributed_runtime.is_primary else None,
    )
    stage_schedule: dict[str, Any] = {}
    for stage in controller.stages:
        events = [
            row for row in controller.schedule_events if row.get("stage_key") == stage.key
        ]
        if config.curriculum.accelerated:
            status = controller.stage_outcomes.get(stage.key, "in_progress")
        else:
            status = (
                "legacy_completed"
                if stage.key in best_by_stage
                and (
                    controller.complete
                    or controller.stage_index
                    > next(
                        index
                        for index, candidate in enumerate(controller.stages)
                        if candidate.key == stage.key
                    )
                )
                else "in_progress"
            )
        stage_schedule[stage.key] = {
            "family": stage.family,
            "role": stage.role,
            "status": status,
            "budget": stage.budget.to_dict(),
            "extension_blocks": int(controller.extension_blocks[stage.key]),
            "last_boundary_update": (
                int(events[-1]["stage_update"]) if events else None
            ),
            "events": events,
        }
    present_families = {stage.family for stage in controller.stages}
    for family, key in (
        ("hierarchy", "phase2_hierarchy_disabled"),
        ("renderer", "phase3_renderer"),
        ("distribution", "phase4_distribution"),
    ):
        if family not in present_families:
            stage_schedule[key] = {
                "family": family,
                "role": config.curriculum.stage_role(family),
                "status": config.curriculum.stage_role(family),
                "budget": None,
                "extension_blocks": 0,
                "last_boundary_update": None,
                "events": [],
            }
    best_model_val_sha256 = None
    best_model_val_content_hash = None
    if complete and distributed_runtime.is_primary:
        best_model_val_sha256 = _file_sha256(output_dir / "best_model_val.pt")
        if streaming_checkpoints:
            selected_payload = load_reconstructor_curriculum_checkpoint(
                output_dir / "best_model_val.pt", device="cpu", require_selected=True
            )
            best_model_val_content_hash = selected_payload.get("content_hash")
    best_model_val_sha256 = broadcast_object(
        distributed_runtime, best_model_val_sha256
    )
    best_model_val_content_hash = broadcast_object(
        distributed_runtime, best_model_val_content_hash
    )
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
            best_model_val_sha256 if complete else None
        ),
        "best_model_val_checkpoint_content_hash": (
            best_model_val_content_hash if complete else None
        ),
        "selection_split": "model_val",
        "selection_mode": "rollout",
        "rollout_validation_count": len(curves),
        "teacher_forced_validation_count": 0,
        "nonfinite_updates": int(nonfinite_updates),
        "compiler_failure_updates": int(compiler_failure_updates),
        "objective_skip_counts": objective_skip_counts,
        "optimizer_groups": optimizer_metadata,
        "model_metadata": model_metadata,
        "provenance": source_provenance,
        "final_test_loaded": False,
        "teacher_logits_loaded": False,
        "stage_handoff": "restore_best_ema_and_reset_optimizer_moments",
        "checkpoint_storage": {
            "profile": (
                "streaming_30gb_v1" if streaming_checkpoints else "cache_heavy_v1"
            ),
            "selected_contract": (
                ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT
                if streaming_checkpoints
                else ABPH_RECONSTRUCTOR_TRAINING_CONTRACT
            ),
            "ephemeral_resume_contract": (
                ABPH_EPHEMERAL_RESUME_CHECKPOINT_CONTRACT
                if streaming_checkpoints
                else ABPH_RECONSTRUCTOR_TRAINING_CONTRACT
            ),
            "persistent_last_checkpoint": not streaming_checkpoints,
            "restart_semantics": (
                ABPH_RESTART_SEMANTICS_WARM_START
                if streaming_checkpoints
                else ABPH_RESTART_SEMANTICS_EXACT
            ),
            "failed_allocation_loses_optimizer_state": bool(streaming_checkpoints),
            "compact_selected_is_exact_resume": False,
        },
        "schedule": {
            "contract": config.curriculum.schedule_contract,
            "policy_label": (
                "accelerated_screening_v1"
                if config.curriculum.accelerated
                else "legacy_fixed_v1"
            ),
            "campaign_profile": config.curriculum.campaign_schedule_profile,
            "stages": stage_schedule,
            "events": list(controller.schedule_events),
            "schedule_truncated": bool(controller.schedule_truncated),
            "negative_mechanism_conclusion_valid": bool(
                not controller.schedule_truncated
            ),
            "automatic_highdata_promotion_allowed": bool(
                complete and not controller.schedule_truncated
            ),
            "warning": (
                "Hard maximum reached while convergence rule still requested extension; "
                "a negative mechanism conclusion is invalid."
                if controller.schedule_truncated
                else None
            ),
        },
        "runtime_profile": {
            "path": str(runtime_profiler.output_path),
            "profile_content_hash": runtime_profile["profile_content_hash"],
            "sampled_training_updates": runtime_profile["summary"][
                "sampled_training_updates"
            ],
        },
        "distributed_runtime": {
            **distributed_runtime.to_dict(),
            "find_unused_parameters": bool(config.ddp_find_unused_parameters),
            "broadcast_buffers": False,
            "stage_aware_wrapper": True,
            "deferred_source_commit": bool(prefetcher is not None),
            "error_events": distributed_error_events,
        },
        "input_pipeline": {
            "contract": ABPH_INPUT_PIPELINE_CONTRACT,
            "asynchronous_prefetch": bool(prefetcher is not None),
            "queue_depth": int(config.prefetch_queue_depth),
            "maximum_resident_batches": prefetch_maximum_resident_batches,
            "pin_memory": bool(config.pin_memory and device.type == "cuda"),
            "nonblocking_transfer": bool(
                config.nonblocking_transfer and device.type == "cuda"
            ),
            "dedicated_cuda_transfer_stream": bool(device.type == "cuda"),
            "cursor_commit": "training_thread_after_successful_dequeue",
        },
    }
    if distributed_runtime.is_primary:
        _atomic_json(output_dir / "run_report.json", report)
    barrier(distributed_runtime)
    report = broadcast_object(
        distributed_runtime, report if distributed_runtime.is_primary else None
    )
    return report


__all__ = [
    "ABPH_DISTRIBUTED_CHECKPOINT_CONTRACT",
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

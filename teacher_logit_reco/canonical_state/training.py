"""Training schedule and safety utilities for Canonical State runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import require_torch

from .losses import CanonicalStateLossWeights


CANONICAL_STATE_TRAINING_CONTRACT = "canonical_state_training_schedule_v1"

SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE = "warmstart_frozen_warmup_to_upper_unfreeze"
SCHEDULE_WARMSTART_JOINT_FROM_START = "warmstart_joint_from_start"
SCHEDULE_FULL_PART_FROZEN_ADAPTER_HEAD = "full_part_frozen_adapter_head"
SCHEDULE_FROM_SCRATCH_CANONICAL_STATE = "from_scratch_canonical_state_model"
SCHEDULE_FROM_SCRATCH_PART_BASELINE = "from_scratch_part_baseline"
SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER = "predictor_pretrain_then_tagger"
SCHEDULE_JOINT_END_TO_END_NO_PRETRAINING = "joint_end_to_end_no_pretraining"

CANONICAL_STATE_TRAINING_SCHEDULES: tuple[str, ...] = (
    SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
    SCHEDULE_WARMSTART_JOINT_FROM_START,
    SCHEDULE_FULL_PART_FROZEN_ADAPTER_HEAD,
    SCHEDULE_FROM_SCRATCH_CANONICAL_STATE,
    SCHEDULE_FROM_SCRATCH_PART_BASELINE,
    SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
    SCHEDULE_JOINT_END_TO_END_NO_PRETRAINING,
)

_SCHEDULE_ALIASES = {
    "warmstart": SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
    "frozen_warmup": SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
    "joint": SCHEDULE_WARMSTART_JOINT_FROM_START,
    "part_frozen": SCHEDULE_FULL_PART_FROZEN_ADAPTER_HEAD,
    "from_scratch_canonical": SCHEDULE_FROM_SCRATCH_CANONICAL_STATE,
    "from_scratch_part": SCHEDULE_FROM_SCRATCH_PART_BASELINE,
    "pretrain": SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
    "joint_no_pretrain": SCHEDULE_JOINT_END_TO_END_NO_PRETRAINING,
}


def normalize_canonical_state_training_schedule(value: str) -> str:
    key = str(value).strip()
    if key in CANONICAL_STATE_TRAINING_SCHEDULES:
        return key
    if key in _SCHEDULE_ALIASES:
        return _SCHEDULE_ALIASES[key]
    raise ValueError(f"unknown canonical-state training schedule {value!r}")


@dataclass(frozen=True)
class CanonicalStateTrainPhase:
    """One trainability/loss phase in a Step 7 schedule."""

    name: str
    start_epoch: int
    end_epoch: int | None
    part_model_trainable: bool
    classifier_head_trainable: bool
    state_predictor_trainable: bool
    state_encoder_trainable: bool
    state_adapter_trainable: bool
    feature_adapter_trainable: bool = False
    state_only_head_trainable: bool = False
    requires_warm_start: bool = True
    allow_teacher_logits: bool = False
    loss_weights: CanonicalStateLossWeights = field(default_factory=CanonicalStateLossWeights)
    lr_by_group: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if int(self.start_epoch) <= 0:
            raise ValueError("start_epoch must be 1-based and positive")
        if self.end_epoch is not None and int(self.end_epoch) < int(self.start_epoch):
            raise ValueError("end_epoch must be >= start_epoch")
        object.__setattr__(self, "start_epoch", int(self.start_epoch))
        object.__setattr__(self, "end_epoch", None if self.end_epoch is None else int(self.end_epoch))
        object.__setattr__(self, "requires_warm_start", bool(self.requires_warm_start))
        object.__setattr__(self, "allow_teacher_logits", bool(self.allow_teacher_logits))
        if not isinstance(self.loss_weights, CanonicalStateLossWeights):
            object.__setattr__(self, "loss_weights", CanonicalStateLossWeights(**dict(self.loss_weights)))
        object.__setattr__(self, "lr_by_group", {str(k): float(v) for k, v in dict(self.lr_by_group).items()})

    def contains_epoch(self, epoch: int) -> bool:
        epoch = int(epoch)
        return epoch >= int(self.start_epoch) and (self.end_epoch is None or epoch <= int(self.end_epoch))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loss_weights"] = self.loss_weights.to_dict()
        return payload


@dataclass(frozen=True)
class CanonicalStateTrainingSchedule:
    name: str
    phases: tuple[CanonicalStateTrainPhase, ...]
    total_epochs: int

    def __post_init__(self) -> None:
        name = normalize_canonical_state_training_schedule(self.name)
        if int(self.total_epochs) <= 0:
            raise ValueError("total_epochs must be positive")
        if not self.phases:
            raise ValueError("schedule must include at least one phase")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "total_epochs", int(self.total_epochs))
        object.__setattr__(self, "phases", tuple(self.phases))

    @property
    def requires_warm_start(self) -> bool:
        return any(bool(phase.requires_warm_start) for phase in self.phases)

    def phase_for_epoch(self, epoch: int) -> CanonicalStateTrainPhase:
        for phase in self.phases:
            if phase.contains_epoch(int(epoch)):
                return phase
        raise ValueError(f"no phase covers epoch {epoch}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": CANONICAL_STATE_TRAINING_CONTRACT,
            "name": self.name,
            "total_epochs": int(self.total_epochs),
            "requires_warm_start": bool(self.requires_warm_start),
            "phases": [phase.to_dict() for phase in self.phases],
        }


def canonical_state_schedule_requires_warm_start(schedule_name: str) -> bool:
    schedule_name = normalize_canonical_state_training_schedule(schedule_name)
    return schedule_name not in {
        SCHEDULE_FROM_SCRATCH_CANONICAL_STATE,
        SCHEDULE_FROM_SCRATCH_PART_BASELINE,
    }


def build_canonical_state_training_schedule(
    schedule_name: str,
    *,
    total_epochs: int = 45,
    warmup_epochs: int = 2,
    adapter_warmup_epochs: int = 2,
    adapter_lr: float = 3.0e-4,
    part_lr: float = 3.0e-5,
    head_lr: float = 1.0e-4,
    predictor_lr: float = 3.0e-4,
    loss_weights: CanonicalStateLossWeights | Mapping[str, Any] | None = None,
) -> CanonicalStateTrainingSchedule:
    """Build one of the Step 7 training phase plans."""

    name = normalize_canonical_state_training_schedule(schedule_name)
    total_epochs = int(total_epochs)
    warmup = max(1, int(warmup_epochs))
    adapter_warmup = max(1, int(adapter_warmup_epochs))
    base_weights = loss_weights if isinstance(loss_weights, CanonicalStateLossWeights) else CanonicalStateLossWeights(**dict(loss_weights or {}))
    state_pretrain_weights = CanonicalStateLossWeights(
        ce=0.0,
        state_huber=max(float(base_weights.state_huber), 1.0),
        state_l1=float(base_weights.state_l1),
        delta_norm=max(float(base_weights.delta_norm), 1.0e-4),
        smoothness=float(base_weights.smoothness),
        uncertainty_state=float(base_weights.uncertainty_state),
    )
    lrs = {
        "state_predictor": float(predictor_lr),
        "state_encoder": float(adapter_lr),
        "state_adapter": float(adapter_lr),
        "feature_adapter": float(adapter_lr),
        "state_only_head": float(head_lr),
        "part_head": float(head_lr),
        "part_model": float(part_lr),
    }

    def phase(
        phase_name: str,
        start: int,
        end: int | None,
        *,
        part: bool,
        head: bool,
        predictor: bool,
        encoder: bool,
        adapter: bool,
        feature: bool = False,
        state_head: bool = False,
        warm_start: bool = True,
        teacher: bool = False,
        weights: CanonicalStateLossWeights | None = None,
        notes: str = "",
    ) -> CanonicalStateTrainPhase:
        return CanonicalStateTrainPhase(
            name=phase_name,
            start_epoch=start,
            end_epoch=end,
            part_model_trainable=part,
            classifier_head_trainable=head,
            state_predictor_trainable=predictor,
            state_encoder_trainable=encoder,
            state_adapter_trainable=adapter,
            feature_adapter_trainable=feature,
            state_only_head_trainable=state_head,
            requires_warm_start=warm_start,
            allow_teacher_logits=teacher,
            loss_weights=weights or base_weights,
            lr_by_group=lrs,
            notes=notes,
        )

    if name == SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE:
        phases = (
            phase("adapter_warmup_part_frozen", 1, warmup, part=False, head=True, predictor=True, encoder=True, adapter=True),
            phase("upper_unfreeze_finetune", warmup + 1, None, part=True, head=True, predictor=True, encoder=True, adapter=True),
        )
    elif name == SCHEDULE_WARMSTART_JOINT_FROM_START:
        phases = (
            phase("joint_from_start", 1, None, part=True, head=True, predictor=True, encoder=True, adapter=True),
        )
    elif name == SCHEDULE_FULL_PART_FROZEN_ADAPTER_HEAD:
        phases = (
            phase("full_part_frozen_adapter_head", 1, None, part=False, head=True, predictor=True, encoder=True, adapter=True),
        )
    elif name == SCHEDULE_FROM_SCRATCH_CANONICAL_STATE:
        phases = (
            phase(
                "from_scratch_canonical_state",
                1,
                None,
                part=True,
                head=True,
                predictor=True,
                encoder=True,
                adapter=True,
                warm_start=False,
            ),
        )
    elif name == SCHEDULE_FROM_SCRATCH_PART_BASELINE:
        phases = (
            phase(
                "from_scratch_part_baseline",
                1,
                None,
                part=True,
                head=True,
                predictor=False,
                encoder=False,
                adapter=False,
                warm_start=False,
                weights=CanonicalStateLossWeights(ce=max(float(base_weights.ce), 1.0)),
            ),
        )
    elif name == SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER:
        phases = (
            phase(
                "state_predictor_pretrain",
                1,
                warmup,
                part=False,
                head=False,
                predictor=True,
                encoder=True,
                adapter=False,
                weights=state_pretrain_weights,
                notes="train residual predictor before tagger consumes Phi_pred",
            ),
            phase(
                "tagger_adapter_warmup",
                warmup + 1,
                warmup + adapter_warmup,
                part=False,
                head=True,
                predictor=True,
                encoder=True,
                adapter=True,
            ),
            phase(
                "joint_finetune_after_pretrain",
                warmup + adapter_warmup + 1,
                None,
                part=True,
                head=True,
                predictor=True,
                encoder=True,
                adapter=True,
            ),
        )
    elif name == SCHEDULE_JOINT_END_TO_END_NO_PRETRAINING:
        phases = (
            phase("joint_end_to_end_no_pretraining", 1, None, part=True, head=True, predictor=True, encoder=True, adapter=True),
        )
    else:  # pragma: no cover - normalize guards this.
        raise AssertionError(name)
    return CanonicalStateTrainingSchedule(name=name, phases=phases, total_epochs=total_epochs)


def _set_module_trainable(module: Any | None, trainable: bool) -> None:
    if module is None or not hasattr(module, "parameters"):
        return
    for parameter in module.parameters():
        parameter.requires_grad_(bool(trainable))
    if hasattr(module, "train"):
        module.train(bool(trainable))


def _part_head_modules(part_model: Any) -> tuple[Any, ...]:
    candidates = (
        getattr(getattr(part_model, "mod", None), "fc", None),
        getattr(getattr(part_model, "mod", None), "head", None),
        getattr(getattr(part_model, "mod", None), "classifier", None),
        getattr(part_model, "fc", None),
        getattr(part_model, "head", None),
        getattr(part_model, "classifier", None),
    )
    modules = []
    seen: set[int] = set()
    for module in candidates:
        if module is not None and hasattr(module, "parameters") and id(module) not in seen:
            modules.append(module)
            seen.add(id(module))
    if modules:
        return tuple(modules)
    if hasattr(part_model, "named_modules"):
        for name, module in part_model.named_modules():
            leaf = str(name).lower().rsplit(".", 1)[-1]
            if leaf in {"fc", "head", "classifier"} and hasattr(module, "parameters") and id(module) not in seen:
                modules.append(module)
                seen.add(id(module))
    return tuple(modules)


def _trainable_count(module: Any | None) -> int:
    if module is None or not hasattr(module, "parameters"):
        return 0
    return int(sum(int(parameter.numel()) for parameter in module.parameters() if bool(parameter.requires_grad)))


def _total_count(module: Any | None) -> int:
    if module is None or not hasattr(module, "parameters"):
        return 0
    return int(sum(int(parameter.numel()) for parameter in module.parameters()))


def apply_canonical_state_train_phase(model: Any, phase: CanonicalStateTrainPhase) -> dict[str, Any]:
    """Apply a Step 7 phase to a CanonicalStateConditionedParT-like model."""

    _set_module_trainable(getattr(model, "part_model", None), bool(phase.part_model_trainable))
    if not bool(phase.part_model_trainable):
        for module in _part_head_modules(getattr(model, "part_model", None)):
            _set_module_trainable(module, bool(phase.classifier_head_trainable))
    _set_module_trainable(getattr(model, "state_predictor", None), bool(phase.state_predictor_trainable))
    _set_module_trainable(getattr(model, "state_encoder", None), bool(phase.state_encoder_trainable))
    _set_module_trainable(getattr(model, "state_adapter", None), bool(phase.state_adapter_trainable))
    _set_module_trainable(getattr(model, "feature_adapter", None), bool(phase.feature_adapter_trainable))
    _set_module_trainable(getattr(model, "state_only_head", None), bool(phase.state_only_head_trainable))
    return canonical_state_phase_report(model, phase)


def canonical_state_trainable_module_groups(model: Any) -> dict[str, dict[str, int]]:
    part_model = getattr(model, "part_model", None)
    head_modules = _part_head_modules(part_model)
    head_ids = {id(parameter) for module in head_modules for parameter in module.parameters()}
    part_non_head_trainable = 0
    part_non_head_total = 0
    if part_model is not None and hasattr(part_model, "parameters"):
        for parameter in part_model.parameters():
            if id(parameter) in head_ids:
                continue
            part_non_head_total += int(parameter.numel())
            if bool(parameter.requires_grad):
                part_non_head_trainable += int(parameter.numel())
    groups = {
        "part_model": {"trainable": part_non_head_trainable, "total": part_non_head_total},
        "part_head": {
            "trainable": sum(_trainable_count(module) for module in head_modules),
            "total": sum(_total_count(module) for module in head_modules),
        },
        "state_predictor": {
            "trainable": _trainable_count(getattr(model, "state_predictor", None)),
            "total": _total_count(getattr(model, "state_predictor", None)),
        },
        "state_encoder": {
            "trainable": _trainable_count(getattr(model, "state_encoder", None)),
            "total": _total_count(getattr(model, "state_encoder", None)),
        },
        "state_adapter": {
            "trainable": _trainable_count(getattr(model, "state_adapter", None)),
            "total": _total_count(getattr(model, "state_adapter", None)),
        },
        "feature_adapter": {
            "trainable": _trainable_count(getattr(model, "feature_adapter", None)),
            "total": _total_count(getattr(model, "feature_adapter", None)),
        },
        "state_only_head": {
            "trainable": _trainable_count(getattr(model, "state_only_head", None)),
            "total": _total_count(getattr(model, "state_only_head", None)),
        },
    }
    return groups


def canonical_state_phase_report(
    model: Any,
    phase: CanonicalStateTrainPhase,
    *,
    state_cache_identity: Mapping[str, Any] | None = None,
    teacher_cache_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    groups = canonical_state_trainable_module_groups(model)
    return {
        "contract": CANONICAL_STATE_TRAINING_CONTRACT,
        "phase_name": str(phase.name),
        "phase": phase.to_dict(),
        "module_group_trainability": groups,
        "optimizer_group_lrs": dict(phase.lr_by_group),
        "loss_weights": phase.loss_weights.to_dict(),
        "requires_warm_start": bool(phase.requires_warm_start),
        "allow_teacher_logits": bool(phase.allow_teacher_logits),
        "state_cache_identity": dict(state_cache_identity or {}),
        "teacher_cache_identity": dict(teacher_cache_identity or {}),
    }


def canonical_state_optimizer_group_specs(model: Any, phase: CanonicalStateTrainPhase) -> list[dict[str, Any]]:
    """Return optimizer groups with names and parameters for a phase."""

    groups: list[dict[str, Any]] = []
    module_by_name = {
        "state_predictor": getattr(model, "state_predictor", None),
        "state_encoder": getattr(model, "state_encoder", None),
        "state_adapter": getattr(model, "state_adapter", None),
        "feature_adapter": getattr(model, "feature_adapter", None),
        "state_only_head": getattr(model, "state_only_head", None),
    }
    part_model = getattr(model, "part_model", None)
    head_modules = _part_head_modules(part_model)
    head_ids = {id(parameter) for module in head_modules for parameter in module.parameters()}
    if part_model is not None and hasattr(part_model, "parameters"):
        params = [
            parameter
            for parameter in part_model.parameters()
            if bool(parameter.requires_grad) and id(parameter) not in head_ids
        ]
        if params:
            groups.append({"name": "part_model", "lr": float(phase.lr_by_group.get("part_model", 0.0)), "params": params})
    head_params = [parameter for module in head_modules for parameter in module.parameters() if bool(parameter.requires_grad)]
    if head_params:
        groups.append({"name": "part_head", "lr": float(phase.lr_by_group.get("part_head", 0.0)), "params": head_params})
    for name, module in module_by_name.items():
        if module is None or not hasattr(module, "parameters"):
            continue
        params = [parameter for parameter in module.parameters() if bool(parameter.requires_grad)]
        if params:
            groups.append({"name": name, "lr": float(phase.lr_by_group.get(name, 0.0)), "params": params})
    return groups


def tensor_tree_isfinite(value: Any) -> bool:
    """Recursively check whether tensors/numeric values are finite."""

    torch = require_torch()
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().detach().cpu().item())
    if isinstance(value, Mapping):
        return all(tensor_tree_isfinite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensor_tree_isfinite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(torch.isfinite(torch.tensor(float(value))).item())
    return True


def canonical_state_nonfinite_batch_report(**named_values: Any) -> dict[str, Any]:
    problems = [name for name, value in named_values.items() if not tensor_tree_isfinite(value)]
    return {
        "ok": not problems,
        "skip_batch": bool(problems),
        "nonfinite_fields": problems,
        "nonfinite_field_count": int(len(problems)),
    }


def canonical_state_should_skip_batch(*, loss: Any | None = None, logits: Any | None = None, **named_values: Any) -> bool:
    values = dict(named_values)
    if loss is not None:
        values["loss"] = loss
    if logits is not None:
        values["logits"] = logits
    return bool(canonical_state_nonfinite_batch_report(**values)["skip_batch"])


def assert_teacher_free_final_test(
    *,
    split: str | None,
    teacher_logits: Any | None = None,
    teacher_representations: Any | None = None,
    allow_teacher_diagnostics: bool = False,
) -> None:
    if str(split or "").strip().lower() != "final_test":
        return
    if bool(allow_teacher_diagnostics):
        return
    if teacher_logits is not None or teacher_representations is not None:
        raise ValueError("primary final_test evaluation must be teacher-free")


__all__ = [
    "CANONICAL_STATE_TRAINING_CONTRACT",
    "CANONICAL_STATE_TRAINING_SCHEDULES",
    "SCHEDULE_FULL_PART_FROZEN_ADAPTER_HEAD",
    "SCHEDULE_FROM_SCRATCH_CANONICAL_STATE",
    "SCHEDULE_FROM_SCRATCH_PART_BASELINE",
    "SCHEDULE_JOINT_END_TO_END_NO_PRETRAINING",
    "SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER",
    "SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE",
    "SCHEDULE_WARMSTART_JOINT_FROM_START",
    "CanonicalStateTrainPhase",
    "CanonicalStateTrainingSchedule",
    "apply_canonical_state_train_phase",
    "assert_teacher_free_final_test",
    "build_canonical_state_training_schedule",
    "canonical_state_nonfinite_batch_report",
    "canonical_state_optimizer_group_specs",
    "canonical_state_phase_report",
    "canonical_state_schedule_requires_warm_start",
    "canonical_state_should_skip_batch",
    "canonical_state_trainable_module_groups",
    "normalize_canonical_state_training_schedule",
    "tensor_tree_isfinite",
]

"""Configuration and naming layer for PD10 privileged distillation.

Step 1 is intentionally config-only.  It freezes the high-data 10-class
experiment contract, declares teacher/student target names, and provides path
helpers for later training, caching, and reporting scripts.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM


PD10_EXPERIMENT_NAME = "privileged_distill_10class_5m"
PD10_EXPERIMENT_STEP = "pd10_step1_config"
PD10_CONTRACT = "pd10_hlt_only_student_privileged_teacher_10class_5m_v1"

PD10_LABEL_NAMES: tuple[str, ...] = tuple(str(name) for name in LABEL_NAMES)
PD10_LABEL_FILTER: tuple[int, ...] = tuple(range(len(PD10_LABEL_NAMES)))
PD10_NUM_CLASSES = len(PD10_LABEL_NAMES)

PD10_SPLIT_ORDER: tuple[str, ...] = ("model_train", "model_val", "final_test")
PD10_SPLIT_SIZES: dict[str, int] = {
    "model_train": 5_000_000,
    "model_val": 1_000_000,
    "final_test": 1_000_000,
}


def _float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


PD10_HLT_DEGRADATION_STRENGTH = _float_from_env("PD10_HLT_DEGRADATION_STRENGTH", 0.6)

PD10_MANIFEST_SPLIT_ORDER: tuple[str, ...] = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)
PD10_MANIFEST_STACK_SPLIT_SIZES: dict[str, int] = {
    "stack_train": 10,
    "stack_val": 10,
}
PD10_MANIFEST_SPLIT_SIZES: dict[str, int] = {
    "model_train": PD10_SPLIT_SIZES["model_train"],
    "model_val": PD10_SPLIT_SIZES["model_val"],
    "stack_train": PD10_MANIFEST_STACK_SPLIT_SIZES["stack_train"],
    "stack_val": PD10_MANIFEST_STACK_SPLIT_SIZES["stack_val"],
    "final_test": PD10_SPLIT_SIZES["final_test"],
}

PD10_TEACHER_NONE = "none"
PD10_TEACHER_HLT = "hlt"
PD10_TEACHER_OFFLINE = "offline"
PD10_TEACHER_DUAL_VIEW = "dual_view"
PD10_TEACHER_TARGETS: tuple[str, ...] = (
    PD10_TEACHER_NONE,
    PD10_TEACHER_HLT,
    PD10_TEACHER_OFFLINE,
    PD10_TEACHER_DUAL_VIEW,
)
PD10_REAL_TEACHERS: tuple[str, ...] = (
    PD10_TEACHER_HLT,
    PD10_TEACHER_OFFLINE,
    PD10_TEACHER_DUAL_VIEW,
)
PD10_TEACHER_MODEL_NAMES: dict[str, str] = {
    PD10_TEACHER_HLT: "hlt_part_teacher_10class",
    PD10_TEACHER_OFFLINE: "offline_part_teacher_10class",
    PD10_TEACHER_DUAL_VIEW: "dual_view_logit_teacher_10class",
}
PD10_TEACHER_ALLOWED_INPUTS: dict[str, str] = {
    PD10_TEACHER_HLT: "HLT_only",
    PD10_TEACHER_OFFLINE: "offline_only_train_time_privileged",
    PD10_TEACHER_DUAL_VIEW: "HLT_plus_offline_train_time_privileged",
}

PD10_STUDENT_INIT_SCRATCH = "scratch"
PD10_STUDENT_INIT_WARM_START = "warm_start"
PD10_STUDENT_INIT_MODES: tuple[str, ...] = (
    PD10_STUDENT_INIT_SCRATCH,
    PD10_STUDENT_INIT_WARM_START,
)

PD10_TARGET_FULL_LOGITS = "full_logits"
PD10_TARGET_TOP3 = "top3"
PD10_TARGET_CONFIDENCE_WEIGHTED = "confidence_weighted"
PD10_STUDENT_TARGET_MODES: tuple[str, ...] = (
    PD10_TARGET_FULL_LOGITS,
    PD10_TARGET_TOP3,
    PD10_TARGET_CONFIDENCE_WEIGHTED,
)

PD10_DEFAULT_TEMPERATURE = 2.0
PD10_DEFAULT_ALPHA = 0.5
PD10_TOP_K = 3

PD10_CORE_STUDENT_VARIANTS = "core"
PD10_PRIORITY_STUDENT_VARIANTS = "priority"

_REPORT_ONLY_CONFIG_KEYS: tuple[str, ...] = (
    "contract",
    "experiment_step",
    "experiment_name",
    "teacher_specs",
    "core_student_variants",
    "priority_student_variants",
)


def _alias_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_string_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = values.split()
    try:
        normalized = tuple(str(value).strip() for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if any(not value for value in normalized):
        raise ValueError(f"{name} cannot contain empty values")
    return normalized


def _normalize_int_tuple(values: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(values, str):
        values = values.replace(",", " ").split()
    try:
        normalized = tuple(int(value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc
    if any(value < 0 for value in normalized):
        raise ValueError(f"{name} cannot contain negative values")
    return normalized


def float_tag(value: float) -> str:
    """Return a stable path/name-safe tag for a scalar hyperparameter."""

    return f"{float(value):0.3g}".replace("-", "m").replace(".", "p")


def normalize_pd10_split_name(value: str) -> str:
    split = str(value).strip()
    if split not in PD10_SPLIT_ORDER:
        raise ValueError(f"Unknown PD10 split {value!r}; expected one of {PD10_SPLIT_ORDER}")
    return split


def normalize_pd10_teacher_target(value: str) -> str:
    key = _alias_key(value)
    aliases = {
        "ce": PD10_TEACHER_NONE,
        "ce_only": PD10_TEACHER_NONE,
        "hard_labels": PD10_TEACHER_NONE,
        "no_teacher": PD10_TEACHER_NONE,
        "self": PD10_TEACHER_HLT,
        "self_distill": PD10_TEACHER_HLT,
        "self_distillation": PD10_TEACHER_HLT,
        "hlt_teacher": PD10_TEACHER_HLT,
        "hlt_part": PD10_TEACHER_HLT,
        "offline_teacher": PD10_TEACHER_OFFLINE,
        "offline_part": PD10_TEACHER_OFFLINE,
        "dual": PD10_TEACHER_DUAL_VIEW,
        "dualview": PD10_TEACHER_DUAL_VIEW,
        "dual_view_teacher": PD10_TEACHER_DUAL_VIEW,
        "xz": PD10_TEACHER_DUAL_VIEW,
        "hlt+offline": PD10_TEACHER_DUAL_VIEW,
        "hlt_offline": PD10_TEACHER_DUAL_VIEW,
        "hlt_plus_offline": PD10_TEACHER_DUAL_VIEW,
    }
    normalized = aliases.get(key, key)
    if normalized not in PD10_TEACHER_TARGETS:
        raise ValueError(f"Unknown PD10 teacher target {value!r}; expected one of {PD10_TEACHER_TARGETS}")
    return normalized


def normalize_pd10_student_init_mode(value: str) -> str:
    key = _alias_key(value)
    aliases = {
        "from_scratch": PD10_STUDENT_INIT_SCRATCH,
        "random": PD10_STUDENT_INIT_SCRATCH,
        "random_init": PD10_STUDENT_INIT_SCRATCH,
        "warm": PD10_STUDENT_INIT_WARM_START,
        "warmstart": PD10_STUDENT_INIT_WARM_START,
        "warm_started": PD10_STUDENT_INIT_WARM_START,
        "baseline": PD10_STUDENT_INIT_WARM_START,
        "hlt_checkpoint": PD10_STUDENT_INIT_WARM_START,
    }
    normalized = aliases.get(key, key)
    if normalized not in PD10_STUDENT_INIT_MODES:
        raise ValueError(f"Unknown PD10 student init mode {value!r}; expected one of {PD10_STUDENT_INIT_MODES}")
    return normalized


def normalize_pd10_student_target_mode(value: str) -> str:
    key = _alias_key(value)
    aliases = {
        "full": PD10_TARGET_FULL_LOGITS,
        "all": PD10_TARGET_FULL_LOGITS,
        "all_logits": PD10_TARGET_FULL_LOGITS,
        "top_k": PD10_TARGET_TOP3,
        "topk": PD10_TARGET_TOP3,
        "top_3": PD10_TARGET_TOP3,
        "top_k_3": PD10_TARGET_TOP3,
        "confidence": PD10_TARGET_CONFIDENCE_WEIGHTED,
        "conf": PD10_TARGET_CONFIDENCE_WEIGHTED,
        "weighted": PD10_TARGET_CONFIDENCE_WEIGHTED,
        "confidence_weight": PD10_TARGET_CONFIDENCE_WEIGHTED,
    }
    normalized = aliases.get(key, key)
    if normalized not in PD10_STUDENT_TARGET_MODES:
        raise ValueError(f"Unknown PD10 target mode {value!r}; expected one of {PD10_STUDENT_TARGET_MODES}")
    return normalized


def pd10_teacher_model_name(teacher_target: str) -> str:
    target = normalize_pd10_teacher_target(teacher_target)
    if target == PD10_TEACHER_NONE:
        raise ValueError("CE-only variants do not have a teacher model name")
    return PD10_TEACHER_MODEL_NAMES[target]


def pd10_student_variant_name(
    init_mode: str,
    teacher_target: str = PD10_TEACHER_NONE,
    target_mode: str = PD10_TARGET_FULL_LOGITS,
    *,
    temperature: float = PD10_DEFAULT_TEMPERATURE,
    kd_alpha: float = PD10_DEFAULT_ALPHA,
    top_k: int = PD10_TOP_K,
) -> str:
    init = normalize_pd10_student_init_mode(init_mode)
    teacher = normalize_pd10_teacher_target(teacher_target)
    mode = normalize_pd10_student_target_mode(target_mode)
    if teacher == PD10_TEACHER_NONE:
        if mode != PD10_TARGET_FULL_LOGITS:
            raise ValueError("CE-only student variants must use target_mode='full_logits'")
        return f"pd10_student_{init}_ce_only"
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    alpha = float(kd_alpha)
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("kd_alpha must be in (0, 1] for KD variants")
    if mode == PD10_TARGET_TOP3 and int(top_k) != PD10_TOP_K:
        return (
            f"pd10_student_{init}_{teacher}_{mode}_k{int(top_k)}"
            f"_t{float_tag(temperature)}_a{float_tag(alpha)}"
        )
    return f"pd10_student_{init}_{teacher}_{mode}_t{float_tag(temperature)}_a{float_tag(alpha)}"


@dataclass(frozen=True)
class PD10TeacherSpec:
    """One train-time teacher source for the PD10 experiment."""

    teacher_target: str
    model_name: str | None = None
    allowed_inputs: str | None = None
    uses_hlt: bool | None = None
    uses_offline: bool | None = None
    description: str = ""

    def __post_init__(self) -> None:
        target = normalize_pd10_teacher_target(self.teacher_target)
        if target == PD10_TEACHER_NONE:
            raise ValueError("PD10TeacherSpec requires a real teacher target, not 'none'")
        model_name = str(self.model_name or pd10_teacher_model_name(target))
        expected_name = pd10_teacher_model_name(target)
        if model_name != expected_name:
            raise ValueError(f"teacher {target!r} model_name must be {expected_name!r}, got {model_name!r}")
        allowed_inputs = str(self.allowed_inputs or PD10_TEACHER_ALLOWED_INPUTS[target])
        if allowed_inputs != PD10_TEACHER_ALLOWED_INPUTS[target]:
            raise ValueError(
                f"teacher {target!r} allowed_inputs must be {PD10_TEACHER_ALLOWED_INPUTS[target]!r}"
            )
        expected_uses_hlt = target in {PD10_TEACHER_HLT, PD10_TEACHER_DUAL_VIEW}
        expected_uses_offline = target in {PD10_TEACHER_OFFLINE, PD10_TEACHER_DUAL_VIEW}
        uses_hlt = expected_uses_hlt if self.uses_hlt is None else bool(self.uses_hlt)
        uses_offline = expected_uses_offline if self.uses_offline is None else bool(self.uses_offline)
        if uses_hlt != expected_uses_hlt:
            raise ValueError(f"teacher {target!r} uses_hlt must be {expected_uses_hlt}")
        if uses_offline != expected_uses_offline:
            raise ValueError(f"teacher {target!r} uses_offline must be {expected_uses_offline}")
        object.__setattr__(self, "teacher_target", target)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "allowed_inputs", allowed_inputs)
        object.__setattr__(self, "uses_hlt", uses_hlt)
        object.__setattr__(self, "uses_offline", uses_offline)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PD10StudentVariantSpec:
    """One HLT-only student condition in the PD10 matrix."""

    init_mode: str
    teacher_target: str = PD10_TEACHER_NONE
    target_mode: str = PD10_TARGET_FULL_LOGITS
    temperature: float = PD10_DEFAULT_TEMPERATURE
    kd_alpha: float = PD10_DEFAULT_ALPHA
    top_k: int = PD10_TOP_K
    group: str = PD10_CORE_STUDENT_VARIANTS
    description: str = ""

    def __post_init__(self) -> None:
        init = normalize_pd10_student_init_mode(self.init_mode)
        teacher = normalize_pd10_teacher_target(self.teacher_target)
        target_mode = normalize_pd10_student_target_mode(self.target_mode)
        temperature = float(self.temperature)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        kd_alpha = float(self.kd_alpha)
        top_k = int(self.top_k)
        if top_k <= 0 or top_k > PD10_NUM_CLASSES:
            raise ValueError(f"top_k must be in [1, {PD10_NUM_CLASSES}]")
        if teacher == PD10_TEACHER_NONE:
            if target_mode != PD10_TARGET_FULL_LOGITS:
                raise ValueError("CE-only student variants must use target_mode='full_logits'")
            kd_alpha = 0.0
        else:
            if kd_alpha <= 0.0 or kd_alpha > 1.0:
                raise ValueError("kd_alpha must be in (0, 1] for KD variants")
        group = str(self.group).strip()
        if group not in {PD10_CORE_STUDENT_VARIANTS, PD10_PRIORITY_STUDENT_VARIANTS}:
            raise ValueError(
                f"group must be {PD10_CORE_STUDENT_VARIANTS!r} or {PD10_PRIORITY_STUDENT_VARIANTS!r}"
            )
        object.__setattr__(self, "init_mode", init)
        object.__setattr__(self, "teacher_target", teacher)
        object.__setattr__(self, "target_mode", target_mode)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "kd_alpha", kd_alpha)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "group", group)

    @property
    def requires_teacher(self) -> bool:
        return self.teacher_target != PD10_TEACHER_NONE

    @property
    def name(self) -> str:
        return pd10_student_variant_name(
            self.init_mode,
            self.teacher_target,
            self.target_mode,
            temperature=self.temperature,
            kd_alpha=self.kd_alpha if self.requires_teacher else PD10_DEFAULT_ALPHA,
            top_k=self.top_k,
        )

    @property
    def teacher_model_name(self) -> str | None:
        if not self.requires_teacher:
            return None
        return pd10_teacher_model_name(self.teacher_target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "init_mode": self.init_mode,
            "teacher_target": self.teacher_target,
            "teacher_model_name": self.teacher_model_name,
            "target_mode": self.target_mode,
            "temperature": float(self.temperature),
            "kd_alpha": float(self.kd_alpha),
            "top_k": int(self.top_k),
            "requires_teacher": bool(self.requires_teacher),
            "group": self.group,
            "description": self.description,
            "student_allowed_inputs": "HLT_only",
        }


def build_pd10_teacher_specs() -> tuple[PD10TeacherSpec, ...]:
    return (
        PD10TeacherSpec(
            PD10_TEACHER_HLT,
            description="HLT-only ParT teacher for self-distillation controls.",
        ),
        PD10TeacherSpec(
            PD10_TEACHER_OFFLINE,
            description="Offline-only ParT privileged teacher.",
        ),
        PD10TeacherSpec(
            PD10_TEACHER_DUAL_VIEW,
            description="HLT+offline logit-fusion privileged teacher.",
        ),
    )


def build_pd10_core_student_variants() -> tuple[PD10StudentVariantSpec, ...]:
    variants: list[PD10StudentVariantSpec] = []
    for init_mode in PD10_STUDENT_INIT_MODES:
        variants.append(
            PD10StudentVariantSpec(
                init_mode=init_mode,
                teacher_target=PD10_TEACHER_NONE,
                description=f"{init_mode} hard-label CE-only control.",
            )
        )
        for teacher_target in (PD10_TEACHER_HLT, PD10_TEACHER_OFFLINE, PD10_TEACHER_DUAL_VIEW):
            variants.append(
                PD10StudentVariantSpec(
                    init_mode=init_mode,
                    teacher_target=teacher_target,
                    target_mode=PD10_TARGET_FULL_LOGITS,
                    temperature=PD10_DEFAULT_TEMPERATURE,
                    kd_alpha=PD10_DEFAULT_ALPHA,
                    description=f"{init_mode} student distilled from {teacher_target} teacher.",
                )
            )
    return tuple(variants)


def build_pd10_priority_student_variants() -> tuple[PD10StudentVariantSpec, ...]:
    return (
        PD10StudentVariantSpec(
            init_mode=PD10_STUDENT_INIT_WARM_START,
            teacher_target=PD10_TEACHER_DUAL_VIEW,
            target_mode=PD10_TARGET_FULL_LOGITS,
            temperature=4.0,
            kd_alpha=PD10_DEFAULT_ALPHA,
            group=PD10_PRIORITY_STUDENT_VARIANTS,
            description="Warm-start dual-view KD with higher temperature.",
        ),
        PD10StudentVariantSpec(
            init_mode=PD10_STUDENT_INIT_WARM_START,
            teacher_target=PD10_TEACHER_DUAL_VIEW,
            target_mode=PD10_TARGET_FULL_LOGITS,
            temperature=PD10_DEFAULT_TEMPERATURE,
            kd_alpha=0.3,
            group=PD10_PRIORITY_STUDENT_VARIANTS,
            description="Warm-start dual-view KD with lower KD weight.",
        ),
        PD10StudentVariantSpec(
            init_mode=PD10_STUDENT_INIT_WARM_START,
            teacher_target=PD10_TEACHER_DUAL_VIEW,
            target_mode=PD10_TARGET_TOP3,
            temperature=PD10_DEFAULT_TEMPERATURE,
            kd_alpha=PD10_DEFAULT_ALPHA,
            group=PD10_PRIORITY_STUDENT_VARIANTS,
            description="Warm-start dual-view KD using top-3 teacher probabilities.",
        ),
        PD10StudentVariantSpec(
            init_mode=PD10_STUDENT_INIT_WARM_START,
            teacher_target=PD10_TEACHER_DUAL_VIEW,
            target_mode=PD10_TARGET_CONFIDENCE_WEIGHTED,
            temperature=PD10_DEFAULT_TEMPERATURE,
            kd_alpha=PD10_DEFAULT_ALPHA,
            group=PD10_PRIORITY_STUDENT_VARIANTS,
            description="Warm-start dual-view KD with per-jet confidence weighting.",
        ),
    )


@dataclass(frozen=True)
class PD10ExperimentConfig:
    """Strict experiment contract for the first high-data PD10 run."""

    label_names: tuple[str, ...] = PD10_LABEL_NAMES
    label_filter: tuple[int, ...] = PD10_LABEL_FILTER
    num_classes: int = PD10_NUM_CLASSES
    split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(PD10_SPLIT_SIZES))
    hlt_degradation_strength: float = PD10_HLT_DEGRADATION_STRENGTH
    teacher_targets: tuple[str, ...] = PD10_TEACHER_TARGETS
    student_init_modes: tuple[str, ...] = PD10_STUDENT_INIT_MODES
    target_modes: tuple[str, ...] = PD10_STUDENT_TARGET_MODES
    default_temperature: float = PD10_DEFAULT_TEMPERATURE
    default_alpha: float = PD10_DEFAULT_ALPHA
    top_k: int = PD10_TOP_K
    raw_token_dim: int = RAW_TOKEN_DIM
    confirm_final_test: bool = True

    def __post_init__(self) -> None:
        label_names = _normalize_string_tuple(self.label_names, name="label_names")
        label_filter = _normalize_int_tuple(self.label_filter, name="label_filter")
        if label_names != PD10_LABEL_NAMES:
            raise ValueError(f"PD10 is locked to full JetClass label order {PD10_LABEL_NAMES}")
        if label_filter != PD10_LABEL_FILTER:
            raise ValueError(f"PD10 label_filter must be {PD10_LABEL_FILTER}")
        if int(self.num_classes) != PD10_NUM_CLASSES:
            raise ValueError(f"PD10 num_classes must be {PD10_NUM_CLASSES}")
        split_sizes = {normalize_pd10_split_name(key): int(value) for key, value in self.split_sizes.items()}
        if tuple(split_sizes.keys()) != PD10_SPLIT_ORDER:
            raise ValueError(f"split_sizes keys must be exactly {PD10_SPLIT_ORDER} in order")
        if dict(split_sizes) != PD10_SPLIT_SIZES:
            raise ValueError(f"PD10 first run is locked to split sizes {PD10_SPLIT_SIZES}")
        if any(size <= 0 for size in split_sizes.values()):
            raise ValueError("split sizes must be positive")
        hlt_degradation_strength = float(self.hlt_degradation_strength)
        if hlt_degradation_strength < 0.0:
            raise ValueError("hlt_degradation_strength must be non-negative")
        if abs(hlt_degradation_strength - PD10_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
            raise ValueError(
                "PD10 run is locked to configured HLT degradation strength "
                f"{PD10_HLT_DEGRADATION_STRENGTH:g}"
            )
        teacher_targets = tuple(normalize_pd10_teacher_target(value) for value in self.teacher_targets)
        if teacher_targets != PD10_TEACHER_TARGETS:
            raise ValueError(f"teacher_targets must be {PD10_TEACHER_TARGETS}")
        student_init_modes = tuple(normalize_pd10_student_init_mode(value) for value in self.student_init_modes)
        if student_init_modes != PD10_STUDENT_INIT_MODES:
            raise ValueError(f"student_init_modes must be {PD10_STUDENT_INIT_MODES}")
        target_modes = tuple(normalize_pd10_student_target_mode(value) for value in self.target_modes)
        if target_modes != PD10_STUDENT_TARGET_MODES:
            raise ValueError(f"target_modes must be {PD10_STUDENT_TARGET_MODES}")
        default_temperature = float(self.default_temperature)
        if default_temperature <= 0.0:
            raise ValueError("default_temperature must be positive")
        default_alpha = float(self.default_alpha)
        if default_alpha <= 0.0 or default_alpha > 1.0:
            raise ValueError("default_alpha must be in (0, 1]")
        top_k = int(self.top_k)
        if top_k != PD10_TOP_K:
            raise ValueError(f"PD10 first run is locked to top_k={PD10_TOP_K}")
        if int(self.raw_token_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"raw_token_dim must match the repo raw token contract {RAW_TOKEN_DIM}")
        if not bool(self.confirm_final_test):
            raise ValueError("PD10 requires explicit final-test confirmation")
        object.__setattr__(self, "label_names", label_names)
        object.__setattr__(self, "label_filter", label_filter)
        object.__setattr__(self, "split_sizes", split_sizes)
        object.__setattr__(self, "hlt_degradation_strength", hlt_degradation_strength)
        object.__setattr__(self, "teacher_targets", teacher_targets)
        object.__setattr__(self, "student_init_modes", student_init_modes)
        object.__setattr__(self, "target_modes", target_modes)
        object.__setattr__(self, "default_temperature", default_temperature)
        object.__setattr__(self, "default_alpha", default_alpha)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "raw_token_dim", int(self.raw_token_dim))
        object.__setattr__(self, "confirm_final_test", bool(self.confirm_final_test))

    @property
    def core_student_variants(self) -> tuple[PD10StudentVariantSpec, ...]:
        return build_pd10_core_student_variants()

    @property
    def priority_student_variants(self) -> tuple[PD10StudentVariantSpec, ...]:
        return build_pd10_priority_student_variants()

    @property
    def teacher_specs(self) -> tuple[PD10TeacherSpec, ...]:
        return build_pd10_teacher_specs()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "PD10ExperimentConfig" | None) -> "PD10ExperimentConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        field_names = {field.name for field in fields(cls)}
        clean_value = {key: item for key, item in value.items() if key not in _REPORT_ONLY_CONFIG_KEYS}
        unknown = sorted(set(clean_value) - field_names)
        if unknown:
            raise ValueError(f"Unknown PD10ExperimentConfig keys: {unknown}")
        return cls(**clean_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": PD10_CONTRACT,
            "experiment_step": PD10_EXPERIMENT_STEP,
            "experiment_name": PD10_EXPERIMENT_NAME,
            "label_names": list(self.label_names),
            "label_filter": list(self.label_filter),
            "num_classes": int(self.num_classes),
            "split_sizes": {split: int(self.split_sizes[split]) for split in PD10_SPLIT_ORDER},
            "hlt_degradation_strength": float(self.hlt_degradation_strength),
            "teacher_targets": list(self.teacher_targets),
            "teacher_specs": [spec.to_dict() for spec in self.teacher_specs],
            "student_init_modes": list(self.student_init_modes),
            "target_modes": list(self.target_modes),
            "default_temperature": float(self.default_temperature),
            "default_alpha": float(self.default_alpha),
            "top_k": int(self.top_k),
            "raw_token_dim": int(self.raw_token_dim),
            "confirm_final_test": bool(self.confirm_final_test),
            "core_student_variants": [spec.to_dict() for spec in self.core_student_variants],
            "priority_student_variants": [spec.to_dict() for spec in self.priority_student_variants],
        }


@dataclass(frozen=True)
class PD10ExperimentLayout:
    """Path helper for later PD10 scripts and reports."""

    output_root: str | Path = "checkpoints"
    experiment_name: str = PD10_EXPERIMENT_NAME

    @property
    def root(self) -> Path:
        return Path(self.output_root) / self.experiment_name

    @property
    def split_manifest_dir(self) -> Path:
        return self.root / "split_manifest"

    @property
    def split_manifest_path(self) -> Path:
        return self.split_manifest_dir / "split_manifest.json.gz"

    @property
    def hlt_cache_dir(self) -> Path:
        return self.root / "hlt_cache"

    @property
    def teachers_dir(self) -> Path:
        return self.root / "teachers"

    @property
    def teacher_logits_dir(self) -> Path:
        return self.root / "teacher_logits"

    @property
    def students_dir(self) -> Path:
        return self.root / "students"

    @property
    def audits_dir(self) -> Path:
        return self.root / "audits"

    @property
    def step2_audit_dir(self) -> Path:
        return self.audits_dir / "step2_splits_hlt_cache"

    @property
    def final_report_dir(self) -> Path:
        return self.root / "final_report"

    def teacher_dir(self, teacher_target: str) -> Path:
        return self.teachers_dir / pd10_teacher_model_name(teacher_target)

    def teacher_checkpoint(self, teacher_target: str) -> Path:
        return self.teacher_dir(teacher_target) / "best_model_val.pt"

    def teacher_logit_cache_dir(self, teacher_target: str) -> Path:
        return self.teacher_logits_dir / pd10_teacher_model_name(teacher_target)

    def student_dir(self, variant: str | PD10StudentVariantSpec) -> Path:
        name = variant.name if isinstance(variant, PD10StudentVariantSpec) else str(variant)
        if not name:
            raise ValueError("student variant name must be non-empty")
        return self.students_dir / name

    def to_dict(self) -> dict[str, str]:
        return {
            "root": self.root.as_posix(),
            "split_manifest_dir": self.split_manifest_dir.as_posix(),
            "split_manifest_path": self.split_manifest_path.as_posix(),
            "hlt_cache_dir": self.hlt_cache_dir.as_posix(),
            "teachers_dir": self.teachers_dir.as_posix(),
            "teacher_logits_dir": self.teacher_logits_dir.as_posix(),
            "students_dir": self.students_dir.as_posix(),
            "audits_dir": self.audits_dir.as_posix(),
            "step2_audit_dir": self.step2_audit_dir.as_posix(),
            "final_report_dir": self.final_report_dir.as_posix(),
        }


def default_pd10_experiment_config() -> PD10ExperimentConfig:
    return PD10ExperimentConfig()


def default_pd10_experiment_layout(
    *,
    output_root: str | Path = "checkpoints",
    experiment_name: str = PD10_EXPERIMENT_NAME,
) -> PD10ExperimentLayout:
    return PD10ExperimentLayout(output_root=output_root, experiment_name=experiment_name)


def pd10_config_manifest(
    *,
    config: PD10ExperimentConfig | None = None,
    layout: PD10ExperimentLayout | None = None,
) -> dict[str, Any]:
    cfg = config or default_pd10_experiment_config()
    lay = layout or default_pd10_experiment_layout()
    return {
        "config": cfg.to_dict(),
        "layout": lay.to_dict(),
    }


__all__ = [
    "PD10_CONTRACT",
    "PD10_CORE_STUDENT_VARIANTS",
    "PD10_DEFAULT_ALPHA",
    "PD10_DEFAULT_TEMPERATURE",
    "PD10_EXPERIMENT_NAME",
    "PD10_EXPERIMENT_STEP",
    "PD10_HLT_DEGRADATION_STRENGTH",
    "PD10_LABEL_FILTER",
    "PD10_LABEL_NAMES",
    "PD10_MANIFEST_SPLIT_ORDER",
    "PD10_MANIFEST_SPLIT_SIZES",
    "PD10_MANIFEST_STACK_SPLIT_SIZES",
    "PD10_NUM_CLASSES",
    "PD10_PRIORITY_STUDENT_VARIANTS",
    "PD10_REAL_TEACHERS",
    "PD10_SPLIT_ORDER",
    "PD10_SPLIT_SIZES",
    "PD10_STUDENT_INIT_MODES",
    "PD10_STUDENT_INIT_SCRATCH",
    "PD10_STUDENT_INIT_WARM_START",
    "PD10_STUDENT_TARGET_MODES",
    "PD10_TARGET_CONFIDENCE_WEIGHTED",
    "PD10_TARGET_FULL_LOGITS",
    "PD10_TARGET_TOP3",
    "PD10_TEACHER_ALLOWED_INPUTS",
    "PD10_TEACHER_DUAL_VIEW",
    "PD10_TEACHER_HLT",
    "PD10_TEACHER_MODEL_NAMES",
    "PD10_TEACHER_NONE",
    "PD10_TEACHER_OFFLINE",
    "PD10_TEACHER_TARGETS",
    "PD10_TOP_K",
    "PD10ExperimentConfig",
    "PD10ExperimentLayout",
    "PD10StudentVariantSpec",
    "PD10TeacherSpec",
    "build_pd10_core_student_variants",
    "build_pd10_priority_student_variants",
    "build_pd10_teacher_specs",
    "default_pd10_experiment_config",
    "default_pd10_experiment_layout",
    "float_tag",
    "normalize_pd10_split_name",
    "normalize_pd10_student_init_mode",
    "normalize_pd10_student_target_mode",
    "normalize_pd10_teacher_target",
    "pd10_config_manifest",
    "pd10_student_variant_name",
    "pd10_teacher_model_name",
]

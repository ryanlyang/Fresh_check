"""Student variant registry for PDV3 AV10-adapter distillation.

Step 3 is intentionally a registry/contract layer.  It names the deployable
HLT-only student variants, maps each one onto an already-implemented AV10
model path, and records the CE/KD recipe that Step 4 will train.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from teacher_logit_reco.architecture_view_part.config import (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_SELF_ATTENTION_ADAPTER,
    architecture_view_variant_num_classes,
    architecture_view_variant_spec,
    normalize_architecture_view_variant,
)

from .config import PDV3_NUM_CLASSES


PDV3_STUDENT_REGISTRY_STEP = "pdv3_step3_student_variant_registry"
PDV3_STUDENT_REGISTRY_CONTRACT = "pdv3_av10_adapter_student_variant_registry_v1"

PDV3_HLT_TEACHER_LOGIT_NAME = "hlt_part_teacher_10class"
PDV3_OFFLINE_TEACHER_LOGIT_NAME = "offline_part_teacher_10class"
PDV3_V1_DUAL_VIEW_TEACHER_NAME = "dual_view_logit_teacher_10class"
PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME = "particle_dual_view_teacher_10class"

PDV3_TEACHER_NONE = "none"
PDV3_TEACHER_V1_DUAL_VIEW = "v1_dual_view"
PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW = "v2_particle_dual_view"
PDV3_TEACHER_FAMILIES = (
    PDV3_TEACHER_NONE,
    PDV3_TEACHER_V1_DUAL_VIEW,
    PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW,
)

PDV3_LOSS_CE = "ce"
PDV3_LOSS_LOGIT_KD = "ce_plus_logit_kd"
PDV3_LOSS_LOGIT_REP_KD = "ce_plus_logit_rep_kd"
PDV3_LOSS_MODES = (
    PDV3_LOSS_CE,
    PDV3_LOSS_LOGIT_KD,
    PDV3_LOSS_LOGIT_REP_KD,
)
PDV3_TRAINING_SCHEDULE_JOINT = "joint"
PDV3_TRAINING_SCHEDULE_STAGED = "staged"
PDV3_TRAINING_SCHEDULE_REP_FROZEN = "rep_frozen"
PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE = "rep_upper_unfreeze"
PDV3_TRAINING_SCHEDULE_REP_FULL_UNFREEZE = "rep_full_unfreeze"
PDV3_TRAINING_SCHEDULES = (
    PDV3_TRAINING_SCHEDULE_JOINT,
    PDV3_TRAINING_SCHEDULE_STAGED,
    PDV3_TRAINING_SCHEDULE_REP_FROZEN,
    PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
    PDV3_TRAINING_SCHEDULE_REP_FULL_UNFREEZE,
)

PDV3_STUDENT_FAMILY_HLT_PART = "hlt_part"
PDV3_STUDENT_FAMILY_FEATURE_MLP = "feature_mlp_embedding_delta"
PDV3_STUDENT_FAMILY_CONTEXT_ADAPTER = "contextual_embedding_delta"
PDV3_STUDENT_FAMILY_LC_MLP_DELTA = "lc_mlp_input_feature_delta"
PDV3_STUDENT_FAMILY_LC_PLUS_FEATURE_MLP = "lc_plus_feature_mlp_combined_adapter"
PDV3_STUDENT_FAMILIES = (
    PDV3_STUDENT_FAMILY_HLT_PART,
    PDV3_STUDENT_FAMILY_FEATURE_MLP,
    PDV3_STUDENT_FAMILY_CONTEXT_ADAPTER,
    PDV3_STUDENT_FAMILY_LC_MLP_DELTA,
    PDV3_STUDENT_FAMILY_LC_PLUS_FEATURE_MLP,
)
PDV3_DEFAULT_CONTEXT_ADAPTER_VARIANT = ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_SELF_ATTENTION_ADAPTER

PDV3_STUDENT_HLT_PART_CE = "pdv3_hlt_part_ce"
PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD = "pdv3_hlt_part_v1_dual_logit_kd"
PDV3_STUDENT_HLT_PART_V2_LOGIT_REP_KD = "pdv3_hlt_part_v2_logit_rep_kd"
PDV3_STUDENT_R0_HLT_PART_CE_BASELINE = "r0_hlt_part_ce_baseline"
PDV3_STUDENT_R1_HLT_PART_LOGIT_KD = "r1_hlt_part_logit_kd"
PDV3_STUDENT_R2_FROZEN_DELTA_Z_REPKD = "r2_frozen_delta_z_repkd"
PDV3_STUDENT_R3_DELTA_Z_UPPER_UNFREEZE_REPKD = "r3_delta_z_upper_unfreeze_repkd"
PDV3_STUDENT_R4_DELTA_Z_FULL_GENTLE_UNFREEZE_REPKD = "r4_delta_z_full_gentle_unfreeze_repkd"
PDV3_STUDENT_R5_DELTA_Z_JOINT_FROM_START_REPKD = "r5_delta_z_joint_from_start_repkd"
PDV3_STUDENT_FEATURE_MLP_CE = "pdv3_feature_mlp_ce"
PDV3_STUDENT_FEATURE_MLP_V1_DUAL_LOGIT_KD = "pdv3_feature_mlp_v1_dual_logit_kd"
PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD = "pdv3_feature_mlp_v2_logit_rep_kd"
PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START = (
    "pdv3_feature_mlp_v2_logit_rep_kd_frozen_start"
)
PDV3_STUDENT_LC_MLP_DELTA_CE = "pdv3_lc_mlp_delta_ce"
PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD = "pdv3_lc_mlp_delta_v2_logit_rep_kd"
PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START = (
    "pdv3_lc_mlp_delta_v2_logit_rep_kd_frozen_start"
)
PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_JOINT = "pdv3_lc_plus_feature_mlp_ce_joint"
PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_STAGED = "pdv3_lc_plus_feature_mlp_ce_staged"
PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_JOINT = (
    "pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_joint"
)
PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED = (
    "pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_staged"
)
PDV3_STUDENT_CONTEXT_ADAPTER_CE = "context_adapter_ce"
PDV3_STUDENT_HLT_PART_LOGIT_KD = "hlt_part_logit_kd"
PDV3_STUDENT_HLT_PART_LOGIT_KD_REPKD = "hlt_part_logit_kd_repkd"
PDV3_STUDENT_FEATURE_MLP_ADAPTER_REPKD = "feature_mlp_adapter_repkd"
PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD = "feature_mlp_adapter_logit_kd_repkd"
PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD_FREEZE_ADAPTER_AFTER_WARMUP = (
    "feature_mlp_adapter_logit_kd_repkd_freeze_adapter_after_warmup"
)
PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD = "context_adapter_logit_kd"
PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD_REPKD = "context_adapter_logit_kd_repkd"
PDV3_STUDENT_CONTEXT_ADAPTER_REPKD_ONLY = "context_adapter_repkd_only"

PDV3_STUDENT_VARIANTS = (
    PDV3_STUDENT_HLT_PART_CE,
    PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
    PDV3_STUDENT_HLT_PART_V2_LOGIT_REP_KD,
    PDV3_STUDENT_R0_HLT_PART_CE_BASELINE,
    PDV3_STUDENT_R1_HLT_PART_LOGIT_KD,
    PDV3_STUDENT_R2_FROZEN_DELTA_Z_REPKD,
    PDV3_STUDENT_R3_DELTA_Z_UPPER_UNFREEZE_REPKD,
    PDV3_STUDENT_R4_DELTA_Z_FULL_GENTLE_UNFREEZE_REPKD,
    PDV3_STUDENT_R5_DELTA_Z_JOINT_FROM_START_REPKD,
    PDV3_STUDENT_FEATURE_MLP_CE,
    PDV3_STUDENT_FEATURE_MLP_V1_DUAL_LOGIT_KD,
    PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
    PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START,
    PDV3_STUDENT_LC_MLP_DELTA_CE,
    PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD,
    PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START,
    PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_JOINT,
    PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_STAGED,
    PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_JOINT,
    PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED,
    PDV3_STUDENT_CONTEXT_ADAPTER_CE,
    PDV3_STUDENT_HLT_PART_LOGIT_KD,
    PDV3_STUDENT_HLT_PART_LOGIT_KD_REPKD,
    PDV3_STUDENT_FEATURE_MLP_ADAPTER_REPKD,
    PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD,
    PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD_FREEZE_ADAPTER_AFTER_WARMUP,
    PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD,
    PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD_REPKD,
    PDV3_STUDENT_CONTEXT_ADAPTER_REPKD_ONLY,
)

PDV3_STUDENT_DEFAULT_VARIANTS = PDV3_STUDENT_VARIANTS


def _alias_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


PDV3_STUDENT_VARIANT_ALIASES: dict[str, str] = {
    "hlt_part_ce": PDV3_STUDENT_HLT_PART_CE,
    "part_ce": PDV3_STUDENT_HLT_PART_CE,
    "baseline": PDV3_STUDENT_HLT_PART_CE,
    "pdv3_baseline": PDV3_STUDENT_HLT_PART_CE,
    "hlt_part_v1": PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
    "pdv3_hlt_v1": PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
    "hlt_part_v1_dual": PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
    "hlt_part_v2": PDV3_STUDENT_HLT_PART_V2_LOGIT_REP_KD,
    "pdv3_hlt_v2": PDV3_STUDENT_HLT_PART_V2_LOGIT_REP_KD,
    "r0": PDV3_STUDENT_R0_HLT_PART_CE_BASELINE,
    "r0_hlt": PDV3_STUDENT_R0_HLT_PART_CE_BASELINE,
    "R0_hlt_part_ce_baseline": PDV3_STUDENT_R0_HLT_PART_CE_BASELINE,
    "r1": PDV3_STUDENT_R1_HLT_PART_LOGIT_KD,
    "r1_logit_kd": PDV3_STUDENT_R1_HLT_PART_LOGIT_KD,
    "R1_hlt_part_logit_kd": PDV3_STUDENT_R1_HLT_PART_LOGIT_KD,
    "r2": PDV3_STUDENT_R2_FROZEN_DELTA_Z_REPKD,
    "r2_repkd_frozen": PDV3_STUDENT_R2_FROZEN_DELTA_Z_REPKD,
    "R2_frozen_delta_z_repkd": PDV3_STUDENT_R2_FROZEN_DELTA_Z_REPKD,
    "r3": PDV3_STUDENT_R3_DELTA_Z_UPPER_UNFREEZE_REPKD,
    "r3_upper_unfreeze": PDV3_STUDENT_R3_DELTA_Z_UPPER_UNFREEZE_REPKD,
    "R3_delta_z_upper_unfreeze_repkd": PDV3_STUDENT_R3_DELTA_Z_UPPER_UNFREEZE_REPKD,
    "r4": PDV3_STUDENT_R4_DELTA_Z_FULL_GENTLE_UNFREEZE_REPKD,
    "r4_full_unfreeze": PDV3_STUDENT_R4_DELTA_Z_FULL_GENTLE_UNFREEZE_REPKD,
    "R4_delta_z_full_gentle_unfreeze_repkd": PDV3_STUDENT_R4_DELTA_Z_FULL_GENTLE_UNFREEZE_REPKD,
    "r5": PDV3_STUDENT_R5_DELTA_Z_JOINT_FROM_START_REPKD,
    "r5_joint": PDV3_STUDENT_R5_DELTA_Z_JOINT_FROM_START_REPKD,
    "R5_delta_z_joint_from_start_repkd": PDV3_STUDENT_R5_DELTA_Z_JOINT_FROM_START_REPKD,
    "feature_mlp_ce": PDV3_STUDENT_FEATURE_MLP_CE,
    "pdv3_feature_ce": PDV3_STUDENT_FEATURE_MLP_CE,
    "feature_mlp_v1": PDV3_STUDENT_FEATURE_MLP_V1_DUAL_LOGIT_KD,
    "feature_mlp_v2": PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
    "feature_mlp_v2_frozen": PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START,
    "lc_mlp_delta_ce": PDV3_STUDENT_LC_MLP_DELTA_CE,
    "input_delta_ce": PDV3_STUDENT_LC_MLP_DELTA_CE,
    "lc_mlp_delta_v2": PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD,
    "input_delta_v2": PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD,
    "lc_mlp_delta_v2_frozen": PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START,
    "input_delta_v2_frozen": PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START,
    "lc_plus_feature_mlp_ce_joint": PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_JOINT,
    "combined_ce_joint": PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_JOINT,
    "lc_plus_feature_mlp_ce_staged": PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_STAGED,
    "combined_ce_staged": PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_STAGED,
    "lc_plus_feature_mlp_v2_joint": PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_JOINT,
    "combined_v2_joint": PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_JOINT,
    "lc_plus_feature_mlp_v2_staged": PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED,
    "combined_v2_staged": PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED,
    "context_adapter_ce": PDV3_STUDENT_CONTEXT_ADAPTER_CE,
    "context_ce": PDV3_STUDENT_CONTEXT_ADAPTER_CE,
    "hlt_part_logit_kd": PDV3_STUDENT_HLT_PART_LOGIT_KD,
    "hlt_logit_kd": PDV3_STUDENT_HLT_PART_LOGIT_KD,
    "hlt_part_logit_kd_repkd": PDV3_STUDENT_HLT_PART_LOGIT_KD_REPKD,
    "hlt_part_repkd": PDV3_STUDENT_HLT_PART_LOGIT_KD_REPKD,
    "feature_mlp_adapter_repkd": PDV3_STUDENT_FEATURE_MLP_ADAPTER_REPKD,
    "feature_mlp_repkd": PDV3_STUDENT_FEATURE_MLP_ADAPTER_REPKD,
    "feature_mlp_adapter_logit_kd_repkd": PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD,
    "feature_mlp_logit_kd_repkd": PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD,
    "feature_mlp_adapter_logit_kd_repkd_freeze_adapter_after_warmup": (
        PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD_FREEZE_ADAPTER_AFTER_WARMUP
    ),
    "feature_mlp_logit_kd_repkd_freeze_adapter_after_warmup": (
        PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD_FREEZE_ADAPTER_AFTER_WARMUP
    ),
    "context_adapter_logit_kd": PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD,
    "context_logit_kd": PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD,
    "context_adapter_logit_kd_repkd": PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD_REPKD,
    "context_logit_kd_repkd": PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD_REPKD,
    "context_adapter_repkd_only": PDV3_STUDENT_CONTEXT_ADAPTER_REPKD_ONLY,
    "context_repkd_only": PDV3_STUDENT_CONTEXT_ADAPTER_REPKD_ONLY,
    "best_expected": PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED,
}


def normalize_pdv3_student_variant(value: str) -> str:
    key = _alias_key(value)
    normalized = PDV3_STUDENT_VARIANT_ALIASES.get(key, key)
    if normalized not in PDV3_STUDENT_VARIANTS:
        raise ValueError(f"Unknown PDV3 student variant {value!r}; expected one of {PDV3_STUDENT_VARIANTS}")
    return normalized


@dataclass(frozen=True)
class PDV3StudentVariantSpec:
    """A deployable HLT-only student plus its Step-4 distillation recipe."""

    name: str
    student_family: str
    architecture_view_variant: str
    teacher_family: str = PDV3_TEACHER_NONE
    loss_mode: str = PDV3_LOSS_CE
    teacher_logit_name: str = ""
    teacher_representation_name: str = ""
    default_use_teacher_logits: bool | None = None
    default_use_teacher_representations: bool | None = None
    kd_temperature: float = 2.0
    kd_alpha: float = 0.0
    rep_beta: float = 0.0
    kd_warmup_epochs: int = 0
    rep_warmup_epochs: int = 0
    freeze_part_epochs: int = 0
    freeze_policy: str = "none"
    training_schedule: str = PDV3_TRAINING_SCHEDULE_JOINT
    combined_adapter: bool = False
    adapter_lr: float = 3.0e-4
    part_lr: float = 1.0e-5
    weight_decay: float = 1.0e-4
    description: str = ""
    expected_rank: int = 0
    is_baseline: bool = False
    is_candidate: bool = True

    def __post_init__(self) -> None:
        name = normalize_pdv3_student_variant(self.name)
        family = str(self.student_family)
        if family not in PDV3_STUDENT_FAMILIES:
            raise ValueError(f"student_family must be one of {PDV3_STUDENT_FAMILIES}, got {family!r}")
        architecture_variant = normalize_architecture_view_variant(self.architecture_view_variant)
        if architecture_view_variant_num_classes(architecture_variant) != PDV3_NUM_CLASSES:
            raise ValueError(f"PDV3 students require 10-class AV variants, got {architecture_variant!r}")
        teacher_family = str(self.teacher_family)
        if teacher_family not in PDV3_TEACHER_FAMILIES:
            raise ValueError(f"teacher_family must be one of {PDV3_TEACHER_FAMILIES}, got {teacher_family!r}")
        loss_mode = str(self.loss_mode)
        if loss_mode not in PDV3_LOSS_MODES:
            raise ValueError(f"loss_mode must be one of {PDV3_LOSS_MODES}, got {loss_mode!r}")
        kd_temperature = float(self.kd_temperature)
        if kd_temperature <= 0.0:
            raise ValueError("kd_temperature must be positive")
        kd_alpha = float(self.kd_alpha)
        rep_beta = float(self.rep_beta)
        if kd_alpha < 0.0 or rep_beta < 0.0:
            raise ValueError("kd_alpha and rep_beta must be non-negative")
        if teacher_family == PDV3_TEACHER_NONE:
            if loss_mode != PDV3_LOSS_CE:
                raise ValueError("teacher_family='none' must use CE loss")
            if kd_alpha != 0.0 or rep_beta != 0.0:
                raise ValueError("CE-only variants must have zero KD weights")
            teacher_logit_name = ""
            teacher_representation_name = ""
            default_use_teacher_logits = False
            default_use_teacher_representations = False
        elif teacher_family == PDV3_TEACHER_V1_DUAL_VIEW:
            if loss_mode != PDV3_LOSS_LOGIT_KD:
                raise ValueError("V1 dual-view teacher variants must use logit KD")
            if not self.teacher_logit_name:
                raise ValueError("V1 dual-view variants require teacher_logit_name")
            if self.teacher_representation_name:
                raise ValueError("V1 dual-view variants must not require representation caches")
            teacher_logit_name = str(self.teacher_logit_name)
            teacher_representation_name = ""
            default_use_teacher_logits = True if self.default_use_teacher_logits is None else bool(self.default_use_teacher_logits)
            default_use_teacher_representations = (
                False
                if self.default_use_teacher_representations is None
                else bool(self.default_use_teacher_representations)
            )
        else:
            if loss_mode not in (PDV3_LOSS_LOGIT_KD, PDV3_LOSS_LOGIT_REP_KD):
                raise ValueError("V2 particle-dual-view variants must use logit KD or logit+representation KD")
            default_use_teacher_logits = (
                loss_mode == PDV3_LOSS_LOGIT_KD
                if self.default_use_teacher_logits is None
                else bool(self.default_use_teacher_logits)
            )
            default_use_teacher_representations = (
                loss_mode == PDV3_LOSS_LOGIT_REP_KD
                if self.default_use_teacher_representations is None
                else bool(self.default_use_teacher_representations)
            )
            if default_use_teacher_logits and not self.teacher_logit_name:
                raise ValueError("V2 logit-KD variants require teacher logits")
            if default_use_teacher_representations and not self.teacher_representation_name:
                raise ValueError("V2 representation variants require teacher representations")
            if loss_mode == PDV3_LOSS_LOGIT_KD and self.teacher_representation_name:
                raise ValueError("V2 logit-only variants must not require teacher representations")
            teacher_logit_name = str(self.teacher_logit_name)
            teacher_representation_name = str(self.teacher_representation_name)
        if teacher_family != PDV3_TEACHER_NONE and not (default_use_teacher_logits or default_use_teacher_representations):
            raise ValueError("KD variants must keep at least one default teacher signal active")
        if default_use_teacher_logits and kd_alpha <= 0.0:
            raise ValueError("logit-KD variants must have positive kd_alpha")
        if not default_use_teacher_logits and kd_alpha != 0.0:
            raise ValueError("variants that disable default teacher logits must have kd_alpha=0")
        if loss_mode == PDV3_LOSS_LOGIT_REP_KD and rep_beta <= 0.0:
            raise ValueError("representation KD variants must have positive rep_beta")
        if loss_mode == PDV3_LOSS_LOGIT_REP_KD and not default_use_teacher_representations:
            raise ValueError("representation KD variants must enable default teacher representations")
        if loss_mode != PDV3_LOSS_LOGIT_REP_KD and rep_beta != 0.0:
            raise ValueError("non-representation variants must have zero rep_beta")
        if teacher_family == PDV3_TEACHER_V1_DUAL_VIEW and default_use_teacher_representations:
            raise ValueError("V1 dual-view variants cannot use representation caches")
        freeze_part_epochs = int(self.freeze_part_epochs)
        if freeze_part_epochs < 0:
            raise ValueError("freeze_part_epochs must be non-negative")
        for name_float in ("adapter_lr", "part_lr", "weight_decay"):
            value = float(getattr(self, name_float))
            if value < 0.0:
                raise ValueError(f"{name_float} must be non-negative")
            object.__setattr__(self, name_float, value)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "student_family", family)
        object.__setattr__(self, "architecture_view_variant", architecture_variant)
        object.__setattr__(self, "teacher_family", teacher_family)
        object.__setattr__(self, "loss_mode", loss_mode)
        object.__setattr__(self, "teacher_logit_name", teacher_logit_name)
        object.__setattr__(self, "teacher_representation_name", teacher_representation_name)
        object.__setattr__(self, "default_use_teacher_logits", bool(default_use_teacher_logits))
        object.__setattr__(self, "default_use_teacher_representations", bool(default_use_teacher_representations))
        object.__setattr__(self, "kd_temperature", kd_temperature)
        object.__setattr__(self, "kd_alpha", kd_alpha)
        object.__setattr__(self, "rep_beta", rep_beta)
        object.__setattr__(self, "kd_warmup_epochs", int(self.kd_warmup_epochs))
        object.__setattr__(self, "rep_warmup_epochs", int(self.rep_warmup_epochs))
        object.__setattr__(self, "freeze_part_epochs", freeze_part_epochs)
        object.__setattr__(self, "freeze_policy", str(self.freeze_policy))
        training_schedule = str(self.training_schedule)
        if training_schedule not in PDV3_TRAINING_SCHEDULES:
            raise ValueError(f"training_schedule must be one of {PDV3_TRAINING_SCHEDULES}")
        combined_adapter = bool(self.combined_adapter)
        if combined_adapter and family != PDV3_STUDENT_FAMILY_LC_PLUS_FEATURE_MLP:
            raise ValueError("combined_adapter=True requires the combined LC+feature MLP family")
        object.__setattr__(self, "training_schedule", training_schedule)
        object.__setattr__(self, "combined_adapter", combined_adapter)
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "expected_rank", int(self.expected_rank))
        object.__setattr__(self, "is_baseline", bool(self.is_baseline))
        object.__setattr__(self, "is_candidate", bool(self.is_candidate))

    @property
    def requires_teacher_logits(self) -> bool:
        return bool(self.default_use_teacher_logits)

    @property
    def requires_teacher_representations(self) -> bool:
        return bool(self.default_use_teacher_representations)

    @property
    def architecture_adapter_type(self) -> str:
        return architecture_view_variant_spec(self.architecture_view_variant).adapter_type

    def architecture_train_overrides(self) -> dict[str, Any]:
        return {
            "variant": self.architecture_view_variant,
            "freeze_part_epochs": int(self.freeze_part_epochs),
            "adapter_lr": float(self.adapter_lr),
            "part_lr": float(self.part_lr),
            "weight_decay": float(self.weight_decay),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requires_teacher_logits"] = bool(self.requires_teacher_logits)
        payload["requires_teacher_representations"] = bool(self.requires_teacher_representations)
        payload["architecture_adapter_type"] = self.architecture_adapter_type
        payload["architecture_view_train_overrides"] = self.architecture_train_overrides()
        payload["combined_adapter"] = bool(self.combined_adapter)
        payload["training_schedule"] = str(self.training_schedule)
        return payload


def _ce_spec(
    *,
    name: str,
    family: str,
    architecture_view_variant: str,
    description: str,
    expected_rank: int,
    freeze_part_epochs: int = 0,
    freeze_policy: str | None = None,
    training_schedule: str = "joint",
    combined_adapter: bool = False,
    is_baseline: bool = False,
) -> PDV3StudentVariantSpec:
    return PDV3StudentVariantSpec(
        name=name,
        student_family=family,
        architecture_view_variant=architecture_view_variant,
        teacher_family=PDV3_TEACHER_NONE,
        loss_mode=PDV3_LOSS_CE,
        kd_temperature=2.0,
        kd_alpha=0.0,
        rep_beta=0.0,
        kd_warmup_epochs=0,
        rep_warmup_epochs=0,
        freeze_part_epochs=freeze_part_epochs,
        freeze_policy=str(freeze_policy or ("none" if freeze_part_epochs == 0 else "adapter_warmup")),
        training_schedule=training_schedule,
        combined_adapter=combined_adapter,
        description=description,
        expected_rank=expected_rank,
        is_baseline=is_baseline,
        is_candidate=not is_baseline,
    )


def _v1_spec(
    *,
    name: str,
    family: str,
    architecture_view_variant: str,
    description: str,
    expected_rank: int,
    freeze_part_epochs: int = 0,
) -> PDV3StudentVariantSpec:
    return PDV3StudentVariantSpec(
        name=name,
        student_family=family,
        architecture_view_variant=architecture_view_variant,
        teacher_family=PDV3_TEACHER_V1_DUAL_VIEW,
        loss_mode=PDV3_LOSS_LOGIT_KD,
        teacher_logit_name=PDV3_V1_DUAL_VIEW_TEACHER_NAME,
        default_use_teacher_logits=True,
        default_use_teacher_representations=False,
        kd_temperature=2.0,
        kd_alpha=0.5,
        rep_beta=0.0,
        kd_warmup_epochs=1,
        rep_warmup_epochs=0,
        freeze_part_epochs=freeze_part_epochs,
        freeze_policy="none" if freeze_part_epochs == 0 else "adapter_warmup",
        description=description,
        expected_rank=expected_rank,
    )


def _v2_logit_only_spec(
    *,
    name: str,
    family: str,
    architecture_view_variant: str,
    description: str,
    expected_rank: int,
    freeze_part_epochs: int = 0,
    freeze_policy: str | None = None,
    training_schedule: str = PDV3_TRAINING_SCHEDULE_JOINT,
) -> PDV3StudentVariantSpec:
    return PDV3StudentVariantSpec(
        name=name,
        student_family=family,
        architecture_view_variant=architecture_view_variant,
        teacher_family=PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW,
        loss_mode=PDV3_LOSS_LOGIT_KD,
        teacher_logit_name=PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME,
        teacher_representation_name="",
        default_use_teacher_logits=True,
        default_use_teacher_representations=False,
        kd_temperature=2.0,
        kd_alpha=0.5,
        rep_beta=0.0,
        kd_warmup_epochs=1,
        rep_warmup_epochs=0,
        freeze_part_epochs=freeze_part_epochs,
        freeze_policy=str(freeze_policy or ("none" if freeze_part_epochs == 0 else "adapter_warmup")),
        training_schedule=training_schedule,
        combined_adapter=False,
        description=description,
        expected_rank=expected_rank,
    )


def _v2_rep_only_spec(
    *,
    name: str,
    family: str,
    architecture_view_variant: str,
    description: str,
    expected_rank: int,
    freeze_part_epochs: int = 0,
    freeze_policy: str | None = None,
    training_schedule: str = PDV3_TRAINING_SCHEDULE_JOINT,
) -> PDV3StudentVariantSpec:
    return PDV3StudentVariantSpec(
        name=name,
        student_family=family,
        architecture_view_variant=architecture_view_variant,
        teacher_family=PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW,
        loss_mode=PDV3_LOSS_LOGIT_REP_KD,
        teacher_logit_name="",
        teacher_representation_name=PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME,
        default_use_teacher_logits=False,
        default_use_teacher_representations=True,
        kd_temperature=2.0,
        kd_alpha=0.0,
        rep_beta=0.10,
        kd_warmup_epochs=0,
        rep_warmup_epochs=2,
        freeze_part_epochs=freeze_part_epochs,
        freeze_policy=str(freeze_policy or ("none" if freeze_part_epochs == 0 else "adapter_warmup")),
        training_schedule=training_schedule,
        combined_adapter=False,
        description=description,
        expected_rank=expected_rank,
    )


def _v2_spec(
    *,
    name: str,
    family: str,
    architecture_view_variant: str,
    description: str,
    expected_rank: int,
    freeze_part_epochs: int = 0,
    freeze_policy: str | None = None,
    training_schedule: str = PDV3_TRAINING_SCHEDULE_JOINT,
    combined_adapter: bool = False,
) -> PDV3StudentVariantSpec:
    return PDV3StudentVariantSpec(
        name=name,
        student_family=family,
        architecture_view_variant=architecture_view_variant,
        teacher_family=PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW,
        loss_mode=PDV3_LOSS_LOGIT_REP_KD,
        teacher_logit_name=PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME,
        teacher_representation_name=PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME,
        default_use_teacher_logits=True,
        default_use_teacher_representations=True,
        kd_temperature=2.0,
        kd_alpha=0.5,
        rep_beta=0.10,
        kd_warmup_epochs=1,
        rep_warmup_epochs=2,
        freeze_part_epochs=freeze_part_epochs,
        freeze_policy=str(freeze_policy or ("none" if freeze_part_epochs == 0 else "adapter_warmup")),
        training_schedule=training_schedule,
        combined_adapter=combined_adapter,
        description=description,
        expected_rank=expected_rank,
    )


def pdv3_student_variant_specs() -> dict[str, PDV3StudentVariantSpec]:
    specs = (
        _ce_spec(
            name=PDV3_STUDENT_HLT_PART_CE,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="Plain warm-started HLT ParT with hard-label CE only.",
            expected_rank=14,
            freeze_part_epochs=0,
            is_baseline=True,
        ),
        _v1_spec(
            name=PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="Plain HLT ParT trained with V1 dual-view logit KD.",
            expected_rank=10,
            freeze_part_epochs=0,
        ),
        _v2_spec(
            name=PDV3_STUDENT_HLT_PART_V2_LOGIT_REP_KD,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="Plain HLT ParT trained with V2 logit and representation KD.",
            expected_rank=8,
            freeze_part_epochs=0,
        ),
        _ce_spec(
            name=PDV3_STUDENT_R0_HLT_PART_CE_BASELINE,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="R0 ladder anchor: plain warm-started HLT ParT CE baseline.",
            expected_rank=20,
            freeze_part_epochs=0,
            is_baseline=True,
        ),
        _v2_logit_only_spec(
            name=PDV3_STUDENT_R1_HLT_PART_LOGIT_KD,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="R1 ladder: HLT ParT with V2 particle-dual teacher logits only.",
            expected_rank=19,
            freeze_part_epochs=0,
            freeze_policy="delta_z_ladder_logit_only",
        ),
        _v2_spec(
            name=PDV3_STUDENT_R2_FROZEN_DELTA_Z_REPKD,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="R2 ladder: frozen HLT ParT body with trainable head and residual delta-z RepKD head.",
            expected_rank=18,
            freeze_part_epochs=45,
            freeze_policy="delta_z_frozen_part",
            training_schedule=PDV3_TRAINING_SCHEDULE_REP_FROZEN,
        ),
        _v2_spec(
            name=PDV3_STUDENT_R3_DELTA_Z_UPPER_UNFREEZE_REPKD,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="R3 ladder: frozen delta-z warmup followed by upper ParT block unfreeze.",
            expected_rank=17,
            freeze_part_epochs=2,
            freeze_policy="delta_z_upper_unfreeze",
            training_schedule=PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
        ),
        _v2_spec(
            name=PDV3_STUDENT_R4_DELTA_Z_FULL_GENTLE_UNFREEZE_REPKD,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="R4 ladder: frozen delta-z warmup followed by full gentle ParT unfreeze.",
            expected_rank=16,
            freeze_part_epochs=2,
            freeze_policy="delta_z_full_gentle_unfreeze",
            training_schedule=PDV3_TRAINING_SCHEDULE_REP_FULL_UNFREEZE,
        ),
        _v2_spec(
            name=PDV3_STUDENT_R5_DELTA_Z_JOINT_FROM_START_REPKD,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="R5 ladder control: HLT ParT and residual delta-z RepKD head trained jointly from epoch 0.",
            expected_rank=15,
            freeze_part_epochs=0,
            freeze_policy="delta_z_joint_from_start",
            training_schedule=PDV3_TRAINING_SCHEDULE_JOINT,
        ),
        _ce_spec(
            name=PDV3_STUDENT_FEATURE_MLP_CE,
            family=PDV3_STUDENT_FAMILY_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
            description="Feature-MLP embedding residual adapter with hard-label CE only.",
            expected_rank=12,
            freeze_part_epochs=1,
        ),
        _v1_spec(
            name=PDV3_STUDENT_FEATURE_MLP_V1_DUAL_LOGIT_KD,
            family=PDV3_STUDENT_FAMILY_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
            description="Feature-MLP embedding residual adapter with V1 dual-view logit KD.",
            expected_rank=9,
            freeze_part_epochs=1,
        ),
        _v2_spec(
            name=PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD,
            family=PDV3_STUDENT_FAMILY_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
            description="Feature-MLP embedding residual adapter with V2 logit and representation KD.",
            expected_rank=2,
            freeze_part_epochs=1,
        ),
        _v2_spec(
            name=PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START,
            family=PDV3_STUDENT_FAMILY_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
            description="Feature-MLP V2 KD with a longer frozen-ParT adapter warm start.",
            expected_rank=4,
            freeze_part_epochs=2,
            freeze_policy="frozen_start",
        ),
        _ce_spec(
            name=PDV3_STUDENT_LC_MLP_DELTA_CE,
            family=PDV3_STUDENT_FAMILY_LC_MLP_DELTA,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
            description="LC-style bounded input-feature delta adapter with hard-label CE only.",
            expected_rank=13,
            freeze_part_epochs=1,
        ),
        _v2_spec(
            name=PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD,
            family=PDV3_STUDENT_FAMILY_LC_MLP_DELTA,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
            description="LC-style bounded input-feature delta adapter with V2 logit and representation KD.",
            expected_rank=7,
            freeze_part_epochs=1,
        ),
        _v2_spec(
            name=PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START,
            family=PDV3_STUDENT_FAMILY_LC_MLP_DELTA,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
            description="LC-style input-feature delta V2 KD with a longer frozen-ParT adapter warm start.",
            expected_rank=5,
            freeze_part_epochs=2,
            freeze_policy="frozen_start",
        ),
        _ce_spec(
            name=PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_JOINT,
            family=PDV3_STUDENT_FAMILY_LC_PLUS_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER,
            description="Combined LC input-feature delta plus feature-MLP embedding delta, trained jointly with CE.",
            expected_rank=11,
            freeze_part_epochs=0,
            freeze_policy="combined_joint",
            training_schedule="joint",
            combined_adapter=True,
        ),
        _ce_spec(
            name=PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_STAGED,
            family=PDV3_STUDENT_FAMILY_LC_PLUS_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER,
            description="Combined LC input-feature delta plus feature-MLP embedding delta, trained with staged CE.",
            expected_rank=6,
            freeze_part_epochs=4,
            freeze_policy="combined_staged",
            training_schedule="staged",
            combined_adapter=True,
        ),
        _v2_spec(
            name=PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_JOINT,
            family=PDV3_STUDENT_FAMILY_LC_PLUS_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER,
            description=(
                "Combined LC input-feature delta plus feature-MLP embedding delta with V2 logit and "
                "representation KD, trained jointly."
            ),
            expected_rank=3,
            freeze_part_epochs=0,
            freeze_policy="combined_joint",
            training_schedule="joint",
            combined_adapter=True,
        ),
        _v2_spec(
            name=PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED,
            family=PDV3_STUDENT_FAMILY_LC_PLUS_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER,
            description=(
                "Combined LC input-feature delta plus feature-MLP embedding delta with V2 logit and "
                "representation KD, trained with the staged adapter schedule."
            ),
            expected_rank=1,
            freeze_part_epochs=4,
            freeze_policy="combined_staged",
            training_schedule="staged",
            combined_adapter=True,
        ),
        _ce_spec(
            name=PDV3_STUDENT_CONTEXT_ADAPTER_CE,
            family=PDV3_STUDENT_FAMILY_CONTEXT_ADAPTER,
            architecture_view_variant=PDV3_DEFAULT_CONTEXT_ADAPTER_VARIANT,
            description="Step 7 context adapter baseline with hard-label CE only.",
            expected_rank=21,
            freeze_part_epochs=1,
            freeze_policy="context_adapter_warmup",
        ),
        _v2_logit_only_spec(
            name=PDV3_STUDENT_HLT_PART_LOGIT_KD,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="Step 7 HLT ParT logit-KD-only baseline using the V2 particle-dual teacher logits.",
            expected_rank=22,
            freeze_part_epochs=0,
            freeze_policy="step7_hlt_part_logit_kd",
        ),
        _v2_spec(
            name=PDV3_STUDENT_HLT_PART_LOGIT_KD_REPKD,
            family=PDV3_STUDENT_FAMILY_HLT_PART,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            description="Step 7 HLT ParT with V2 logit KD plus residual delta-z RepKD.",
            expected_rank=23,
            freeze_part_epochs=2,
            freeze_policy="step7_delta_z_upper_unfreeze",
            training_schedule=PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
        ),
        _v2_rep_only_spec(
            name=PDV3_STUDENT_FEATURE_MLP_ADAPTER_REPKD,
            family=PDV3_STUDENT_FAMILY_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
            description="Step 7 feature-MLP adapter with residual delta-z RepKD only, no logit KD.",
            expected_rank=24,
            freeze_part_epochs=2,
            freeze_policy="feature_mlp_repkd_upper_unfreeze",
            training_schedule=PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
        ),
        _v2_spec(
            name=PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD,
            family=PDV3_STUDENT_FAMILY_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
            description="Step 7 feature-MLP adapter with V2 logit KD plus residual delta-z RepKD.",
            expected_rank=25,
            freeze_part_epochs=2,
            freeze_policy="feature_mlp_logit_kd_repkd_upper_unfreeze",
            training_schedule=PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
        ),
        _v2_spec(
            name=PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD_FREEZE_ADAPTER_AFTER_WARMUP,
            family=PDV3_STUDENT_FAMILY_FEATURE_MLP,
            architecture_view_variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
            description=(
                "Step 7 feature-MLP adapter with V2 logit KD plus RepKD; freezes the embedding adapter "
                "after warmup so upper ParT blocks and delta-z do the later co-adaptation."
            ),
            expected_rank=26,
            freeze_part_epochs=2,
            freeze_policy="freeze_adapter_after_warmup",
            training_schedule=PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
        ),
        _v2_logit_only_spec(
            name=PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD,
            family=PDV3_STUDENT_FAMILY_CONTEXT_ADAPTER,
            architecture_view_variant=PDV3_DEFAULT_CONTEXT_ADAPTER_VARIANT,
            description="Step 7 context adapter with V2 particle-dual teacher logit KD only.",
            expected_rank=27,
            freeze_part_epochs=1,
            freeze_policy="context_adapter_logit_kd_warmup",
        ),
        _v2_spec(
            name=PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD_REPKD,
            family=PDV3_STUDENT_FAMILY_CONTEXT_ADAPTER,
            architecture_view_variant=PDV3_DEFAULT_CONTEXT_ADAPTER_VARIANT,
            description="Step 7 context adapter with V2 logit KD plus residual delta-z RepKD.",
            expected_rank=28,
            freeze_part_epochs=2,
            freeze_policy="context_adapter_logit_kd_repkd_upper_unfreeze",
            training_schedule=PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
        ),
        _v2_rep_only_spec(
            name=PDV3_STUDENT_CONTEXT_ADAPTER_REPKD_ONLY,
            family=PDV3_STUDENT_FAMILY_CONTEXT_ADAPTER,
            architecture_view_variant=PDV3_DEFAULT_CONTEXT_ADAPTER_VARIANT,
            description="Step 7 context adapter with residual delta-z RepKD only, no logit KD.",
            expected_rank=29,
            freeze_part_epochs=2,
            freeze_policy="context_adapter_repkd_upper_unfreeze",
            training_schedule=PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE,
        ),
    )
    return {spec.name: spec for spec in specs}


def pdv3_student_variant_spec(value: str) -> PDV3StudentVariantSpec:
    return pdv3_student_variant_specs()[normalize_pdv3_student_variant(value)]


def pdv3_student_variants(*, candidates_only: bool = False) -> tuple[str, ...]:
    if not candidates_only:
        return PDV3_STUDENT_VARIANTS
    specs = pdv3_student_variant_specs()
    return tuple(name for name in PDV3_STUDENT_VARIANTS if specs[name].is_candidate)


def pdv3_student_registry_manifest() -> dict[str, Any]:
    return {
        "step": PDV3_STUDENT_REGISTRY_STEP,
        "contract": PDV3_STUDENT_REGISTRY_CONTRACT,
        "student_variants": list(PDV3_STUDENT_VARIANTS),
        "default_student_variants": list(PDV3_STUDENT_DEFAULT_VARIANTS),
        "student_families": list(PDV3_STUDENT_FAMILIES),
        "teacher_families": list(PDV3_TEACHER_FAMILIES),
        "loss_modes": list(PDV3_LOSS_MODES),
        "teacher_names": {
            "hlt": PDV3_HLT_TEACHER_LOGIT_NAME,
            "offline": PDV3_OFFLINE_TEACHER_LOGIT_NAME,
            "v1_dual_view": PDV3_V1_DUAL_VIEW_TEACHER_NAME,
            "v2_particle_dual_view": PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME,
        },
        "variants": {name: spec.to_dict() for name, spec in pdv3_student_variant_specs().items()},
    }


__all__ = [
    "PDV3_HLT_TEACHER_LOGIT_NAME",
    "PDV3_LOSS_CE",
    "PDV3_LOSS_LOGIT_KD",
    "PDV3_LOSS_LOGIT_REP_KD",
    "PDV3_LOSS_MODES",
    "PDV3_OFFLINE_TEACHER_LOGIT_NAME",
    "PDV3_STUDENT_DEFAULT_VARIANTS",
    "PDV3_STUDENT_FAMILIES",
    "PDV3_STUDENT_FAMILY_CONTEXT_ADAPTER",
    "PDV3_STUDENT_FAMILY_FEATURE_MLP",
    "PDV3_STUDENT_FAMILY_HLT_PART",
    "PDV3_STUDENT_FAMILY_LC_MLP_DELTA",
    "PDV3_STUDENT_FAMILY_LC_PLUS_FEATURE_MLP",
    "PDV3_STUDENT_FEATURE_MLP_CE",
    "PDV3_STUDENT_CONTEXT_ADAPTER_CE",
    "PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD",
    "PDV3_STUDENT_CONTEXT_ADAPTER_LOGIT_KD_REPKD",
    "PDV3_STUDENT_CONTEXT_ADAPTER_REPKD_ONLY",
    "PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD",
    "PDV3_STUDENT_FEATURE_MLP_ADAPTER_LOGIT_KD_REPKD_FREEZE_ADAPTER_AFTER_WARMUP",
    "PDV3_STUDENT_FEATURE_MLP_ADAPTER_REPKD",
    "PDV3_STUDENT_FEATURE_MLP_V1_DUAL_LOGIT_KD",
    "PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD",
    "PDV3_STUDENT_FEATURE_MLP_V2_LOGIT_REP_KD_FROZEN_START",
    "PDV3_STUDENT_HLT_PART_CE",
    "PDV3_STUDENT_HLT_PART_LOGIT_KD",
    "PDV3_STUDENT_HLT_PART_LOGIT_KD_REPKD",
    "PDV3_STUDENT_HLT_PART_V1_DUAL_LOGIT_KD",
    "PDV3_STUDENT_HLT_PART_V2_LOGIT_REP_KD",
    "PDV3_STUDENT_R0_HLT_PART_CE_BASELINE",
    "PDV3_STUDENT_R1_HLT_PART_LOGIT_KD",
    "PDV3_STUDENT_R2_FROZEN_DELTA_Z_REPKD",
    "PDV3_STUDENT_R3_DELTA_Z_UPPER_UNFREEZE_REPKD",
    "PDV3_STUDENT_R4_DELTA_Z_FULL_GENTLE_UNFREEZE_REPKD",
    "PDV3_STUDENT_R5_DELTA_Z_JOINT_FROM_START_REPKD",
    "PDV3_STUDENT_LC_MLP_DELTA_CE",
    "PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD",
    "PDV3_STUDENT_LC_MLP_DELTA_V2_LOGIT_REP_KD_FROZEN_START",
    "PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_JOINT",
    "PDV3_STUDENT_LC_PLUS_FEATURE_MLP_CE_STAGED",
    "PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_JOINT",
    "PDV3_STUDENT_LC_PLUS_FEATURE_MLP_V2_LOGIT_REP_KD_STAGED",
    "PDV3_STUDENT_REGISTRY_CONTRACT",
    "PDV3_STUDENT_REGISTRY_STEP",
    "PDV3_STUDENT_VARIANTS",
    "PDV3_STUDENT_VARIANT_ALIASES",
    "PDV3_TEACHER_FAMILIES",
    "PDV3_TEACHER_NONE",
    "PDV3_TEACHER_V1_DUAL_VIEW",
    "PDV3_TEACHER_V2_PARTICLE_DUAL_VIEW",
    "PDV3_TRAINING_SCHEDULE_JOINT",
    "PDV3_TRAINING_SCHEDULE_REP_FROZEN",
    "PDV3_TRAINING_SCHEDULE_REP_FULL_UNFREEZE",
    "PDV3_TRAINING_SCHEDULE_REP_UPPER_UNFREEZE",
    "PDV3_TRAINING_SCHEDULE_STAGED",
    "PDV3_TRAINING_SCHEDULES",
    "PDV3_V1_DUAL_VIEW_TEACHER_NAME",
    "PDV3_V2_PARTICLE_DUAL_VIEW_TEACHER_NAME",
    "PDV3_DEFAULT_CONTEXT_ADAPTER_VARIANT",
    "PDV3StudentVariantSpec",
    "normalize_pdv3_student_variant",
    "pdv3_student_registry_manifest",
    "pdv3_student_variant_spec",
    "pdv3_student_variant_specs",
    "pdv3_student_variants",
]

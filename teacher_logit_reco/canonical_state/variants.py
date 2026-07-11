"""Variant registry for Canonical Multi-Scale Jet State A0-G3 runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .losses import CanonicalStateLossWeights
from .predictor import (
    PREDICTOR_VARIANT_DEEPSETS,
    PREDICTOR_VARIANT_GEOMETRY_BIASED,
    PREDICTOR_VARIANT_HARD_LOCALITY,
    PREDICTOR_VARIANT_NO_GEOMETRY_BIAS,
    PREDICTOR_VARIANT_NO_STATE_SELF_ATTENTION,
    PREDICTOR_VARIANT_PARTICLE_ONLY_QUERIES,
    PREDICTOR_VARIANT_STATE_ONLY,
    PREDICTOR_VARIANT_UNCERTAINTY,
)
from .tagger import (
    STATE_CONTEXT_ALL,
    STATE_CONTEXT_DELTA_PHI,
    STATE_CONTEXT_FEATURE_MLP_PLUS_STATE,
    STATE_CONTEXT_NOISE,
    STATE_CONTEXT_ORACLE_PHI_OFF,
    STATE_CONTEXT_PARTICLES_ONLY,
    STATE_CONTEXT_PHI_HLT,
    STATE_CONTEXT_PHI_PRED,
    STATE_CONTEXT_SHUFFLED,
    STATE_CONTEXT_STATE_ONLY,
)
from .training import (
    SCHEDULE_FULL_PART_FROZEN_ADAPTER_HEAD,
    SCHEDULE_FROM_SCRATCH_CANONICAL_STATE,
    SCHEDULE_FROM_SCRATCH_PART_BASELINE,
    SCHEDULE_JOINT_END_TO_END_NO_PRETRAINING,
    SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
    SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
    SCHEDULE_WARMSTART_JOINT_FROM_START,
    normalize_canonical_state_training_schedule,
)


CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT = "canonical_state_a0_g3_variant_registry_v1"

MODEL_KIND_PART_BASELINE = "part_baseline"
MODEL_KIND_AV10_FEATURE_MLP = "av10_feature_mlp_adapter_repeat"
MODEL_KIND_STATE_CONDITIONED_PART = "state_conditioned_part"
MODEL_KIND_STATE_PREDICTOR_ONLY = "state_predictor_only"
MODEL_KIND_STATE_ONLY_TAGGER = "state_only_tagger"
MODEL_KIND_LOGIT_FUSION = "logit_fusion"
MODEL_KIND_STATE_TOKEN_FUSION = "state_token_fusion"
MODEL_KIND_PARTICLE_VIEW_FUSION = "particle_view_fusion"
MODEL_KIND_SEED_ENSEMBLE = "seed_ensemble"
MODEL_KIND_ORACLE_DIAGNOSTIC = "oracle_diagnostic"
MODEL_KIND_ANALYSIS_REPORT = "analysis_report"

WARM_START_NONE = "none"
WARM_START_HLT_PART_BASELINE = "hlt_part_baseline"
WARM_START_AV10_REFERENCE = "av10_reference"
WARM_START_PRETRAINED_PREDICTOR = "pretrained_state_predictor"

FINAL_TEST_POLICY_PRIMARY_TEACHER_FREE = "primary_teacher_free"
FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE = "diagnostic_teacher_free"
FINAL_TEST_POLICY_MODEL_VAL_ONLY = "model_val_only"
FINAL_TEST_POLICY_STACK_ONLY = "stack_only"
FINAL_TEST_POLICY_REPORT_ONLY = "report_only"


CANONICAL_STATE_EXPECTED_RUN_IDS: tuple[str, ...] = (
    "A0",
    "A1",
    "A2",
    "A3",
    "B0",
    "B1",
    "B2",
    "B3",
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "E0",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "F0",
    "F1",
    "F2",
    "F3",
    "F4",
    "Fseed",
    "Fshuffle",
    "G0",
    "G1",
    "G2",
    "G3",
)


def _ce() -> CanonicalStateLossWeights:
    return CanonicalStateLossWeights(ce=1.0)


def _state_aux(*, kd: bool = False, ce: float = 1.0, state: float = 1.0) -> CanonicalStateLossWeights:
    return CanonicalStateLossWeights(
        ce=float(ce),
        state_huber=float(state),
        state_l1=0.25 * float(state),
        logit_kd=0.5 if bool(kd) else 0.0,
        delta_norm=1.0e-4,
        smoothness=1.0e-3,
        kd_temperature=2.0,
    )


def _predictor_loss(*, uncertainty: bool = False) -> CanonicalStateLossWeights:
    return CanonicalStateLossWeights(
        ce=0.0,
        state_huber=0.0 if bool(uncertainty) else 1.0,
        state_l1=0.25,
        delta_norm=1.0e-4,
        smoothness=1.0e-3,
        uncertainty_state=1.0 if bool(uncertainty) else 0.0,
    )


@dataclass(frozen=True)
class CanonicalStateVariantSpec:
    """One A0-G3 run specification."""

    run_id: str
    tier: str
    title: str
    model_kind: str
    schedule: str
    tagger_mode: str | None = None
    predictor_variant: str | None = None
    state_context_families: tuple[str, ...] = ()
    warm_start_policy: str = WARM_START_HLT_PART_BASELINE
    loss_weights: CanonicalStateLossWeights = field(default_factory=_ce)
    uses_teacher_logits: bool = False
    uses_offline_phi: bool = False
    oracle_inputs_allowed: bool = False
    primary: bool = True
    diagnostic_only: bool = False
    requires_stack_splits: bool = False
    seed_count: int = 1
    fusion_inputs: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    final_test_policy: str = FINAL_TEST_POLICY_PRIMARY_TEACHER_FREE
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        run_id = str(self.run_id)
        if run_id not in CANONICAL_STATE_EXPECTED_RUN_IDS:
            raise ValueError(f"unknown canonical-state run id {run_id!r}")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "tier", str(self.tier))
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "model_kind", str(self.model_kind))
        object.__setattr__(self, "schedule", normalize_canonical_state_training_schedule(self.schedule))
        object.__setattr__(
            self,
            "state_context_families",
            tuple(str(item) for item in self.state_context_families),
        )
        object.__setattr__(self, "fusion_inputs", tuple(str(item) for item in self.fusion_inputs))
        object.__setattr__(self, "dependencies", tuple(str(item) for item in self.dependencies))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))
        object.__setattr__(self, "uses_teacher_logits", bool(self.uses_teacher_logits))
        object.__setattr__(self, "uses_offline_phi", bool(self.uses_offline_phi))
        object.__setattr__(self, "oracle_inputs_allowed", bool(self.oracle_inputs_allowed))
        object.__setattr__(self, "primary", bool(self.primary))
        object.__setattr__(self, "diagnostic_only", bool(self.diagnostic_only))
        object.__setattr__(self, "requires_stack_splits", bool(self.requires_stack_splits))
        object.__setattr__(self, "seed_count", int(self.seed_count))
        if int(self.seed_count) <= 0:
            raise ValueError("seed_count must be positive")
        if not isinstance(self.loss_weights, CanonicalStateLossWeights):
            object.__setattr__(self, "loss_weights", CanonicalStateLossWeights(**dict(self.loss_weights)))
        if self.oracle_inputs_allowed and self.primary:
            raise ValueError(f"{run_id} uses oracle inputs and cannot be primary")
        if self.uses_offline_phi and not self.oracle_inputs_allowed and self.model_kind != MODEL_KIND_STATE_PREDICTOR_ONLY:
            raise ValueError(f"{run_id} uses offline Phi but is not marked as oracle/predictor")
        if self.diagnostic_only and self.primary:
            raise ValueError(f"{run_id} cannot be both primary and diagnostic_only")
        if self.final_test_policy in {FINAL_TEST_POLICY_MODEL_VAL_ONLY, FINAL_TEST_POLICY_STACK_ONLY, FINAL_TEST_POLICY_REPORT_ONLY} and self.primary:
            raise ValueError(f"{run_id} has non-primary final-test policy but is marked primary")
        missing_dependencies = [item for item in self.dependencies if item not in CANONICAL_STATE_EXPECTED_RUN_IDS]
        if missing_dependencies:
            raise ValueError(f"{run_id} has unknown dependencies: {missing_dependencies}")
        missing_fusion_inputs = [item for item in self.fusion_inputs if item not in CANONICAL_STATE_EXPECTED_RUN_IDS]
        if missing_fusion_inputs:
            raise ValueError(f"{run_id} has unknown fusion inputs: {missing_fusion_inputs}")

    @property
    def is_fusion(self) -> bool:
        return bool(self.fusion_inputs) or self.model_kind in {
            MODEL_KIND_LOGIT_FUSION,
            MODEL_KIND_STATE_TOKEN_FUSION,
            MODEL_KIND_PARTICLE_VIEW_FUSION,
            MODEL_KIND_SEED_ENSEMBLE,
        }

    @property
    def is_oracle(self) -> bool:
        return bool(self.oracle_inputs_allowed)

    @property
    def teacher_for_training_only(self) -> bool:
        return bool(self.uses_teacher_logits and self.final_test_policy == FINAL_TEST_POLICY_PRIMARY_TEACHER_FREE)

    def allows_primary_final_test(self) -> bool:
        return (
            bool(self.primary)
            and not bool(self.oracle_inputs_allowed)
            and self.final_test_policy == FINAL_TEST_POLICY_PRIMARY_TEACHER_FREE
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT
        payload["loss_weights"] = self.loss_weights.to_dict()
        payload["is_fusion"] = bool(self.is_fusion)
        payload["is_oracle"] = bool(self.is_oracle)
        payload["teacher_for_training_only"] = bool(self.teacher_for_training_only)
        payload["allows_primary_final_test"] = bool(self.allows_primary_final_test())
        return payload


def _spec(
    run_id: str,
    tier: str,
    title: str,
    model_kind: str,
    schedule: str,
    *,
    tagger_mode: str | None = None,
    predictor_variant: str | None = None,
    state_context_families: tuple[str, ...] = (),
    warm_start_policy: str = WARM_START_HLT_PART_BASELINE,
    loss_weights: CanonicalStateLossWeights | None = None,
    uses_teacher_logits: bool = False,
    uses_offline_phi: bool = False,
    oracle_inputs_allowed: bool = False,
    primary: bool = True,
    diagnostic_only: bool = False,
    requires_stack_splits: bool = False,
    seed_count: int = 1,
    fusion_inputs: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
    final_test_policy: str = FINAL_TEST_POLICY_PRIMARY_TEACHER_FREE,
    notes: tuple[str, ...] = (),
) -> CanonicalStateVariantSpec:
    return CanonicalStateVariantSpec(
        run_id=run_id,
        tier=tier,
        title=title,
        model_kind=model_kind,
        schedule=schedule,
        tagger_mode=tagger_mode,
        predictor_variant=predictor_variant,
        state_context_families=state_context_families,
        warm_start_policy=warm_start_policy,
        loss_weights=loss_weights or _ce(),
        uses_teacher_logits=uses_teacher_logits,
        uses_offline_phi=uses_offline_phi,
        oracle_inputs_allowed=oracle_inputs_allowed,
        primary=primary,
        diagnostic_only=diagnostic_only,
        requires_stack_splits=requires_stack_splits,
        seed_count=seed_count,
        fusion_inputs=fusion_inputs,
        dependencies=dependencies,
        final_test_policy=final_test_policy,
        notes=notes,
    )


def canonical_state_variant_registry() -> dict[str, CanonicalStateVariantSpec]:
    """Return the complete A0-G3 campaign registry."""

    registry = {
        "A0": _spec(
            "A0",
            "A",
            "HLT ParT baseline",
            MODEL_KIND_PART_BASELINE,
            SCHEDULE_FROM_SCRATCH_PART_BASELINE,
            tagger_mode=STATE_CONTEXT_PARTICLES_ONLY,
            warm_start_policy=WARM_START_NONE,
            notes=("canonical HLT particle-only baseline",),
        ),
        "A1": _spec(
            "A1",
            "A",
            "HLT ParT fine-tune-only control",
            MODEL_KIND_PART_BASELINE,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_PARTICLES_ONLY,
            dependencies=("A0",),
            notes=("same schedule budget as adapter models, no Phi tokens",),
        ),
        "A2": _spec(
            "A2",
            "A",
            "canonical feature_mlp_plus_state adapter control",
            MODEL_KIND_AV10_FEATURE_MLP,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            warm_start_policy=WARM_START_AV10_REFERENCE,
            dependencies=("A0",),
            notes=("canonical-state runner approximation of the known strong AV10 feature MLP adapter",),
        ),
        "A3": _spec(
            "A3",
            "A",
            "HLT ParT seed ensemble baseline",
            MODEL_KIND_SEED_ENSEMBLE,
            SCHEDULE_FROM_SCRATCH_PART_BASELINE,
            tagger_mode=STATE_CONTEXT_PARTICLES_ONLY,
            warm_start_policy=WARM_START_NONE,
            seed_count=3,
            requires_stack_splits=True,
            fusion_inputs=("A0",),
            dependencies=("A0",),
            notes=("same-size seed ensemble control for fusion claims",),
        ),
        "B0": _spec(
            "B0",
            "B",
            "HLT particles plus Phi_hlt context",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_PHI_HLT,
            state_context_families=("phi_hlt",),
            dependencies=("A0",),
        ),
        "B1": _spec(
            "B1",
            "B",
            "Phi_hlt-only tagger",
            MODEL_KIND_STATE_ONLY_TAGGER,
            SCHEDULE_FROM_SCRATCH_CANONICAL_STATE,
            tagger_mode=STATE_CONTEXT_STATE_ONLY,
            state_context_families=("phi_hlt",),
            warm_start_policy=WARM_START_NONE,
            notes=("measures standalone information in canonical state tokens",),
        ),
        "B2": _spec(
            "B2",
            "B",
            "HLT particles plus shuffled Phi_hlt control",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_SHUFFLED,
            state_context_families=("phi_hlt_shuffled",),
            primary=False,
            diagnostic_only=True,
            dependencies=("A0", "B0"),
            final_test_policy=FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE,
            notes=("breaks state semantics while keeping adapter capacity",),
        ),
        "B3": _spec(
            "B3",
            "B",
            "HLT particles plus random/noise state control",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_NOISE,
            state_context_families=("noise_state",),
            primary=False,
            diagnostic_only=True,
            dependencies=("A0", "B0"),
            final_test_policy=FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE,
            notes=("capacity/regularization-only state adapter control",),
        ),
        "C0": _spec(
            "C0",
            "C",
            "geometry-biased decoder predictor",
            MODEL_KIND_STATE_PREDICTOR_ONLY,
            SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            warm_start_policy=WARM_START_NONE,
            loss_weights=_predictor_loss(),
            uses_offline_phi=True,
            primary=False,
            diagnostic_only=True,
            final_test_policy=FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE,
        ),
        "C1": _spec(
            "C1",
            "C",
            "no-geometry-bias decoder predictor",
            MODEL_KIND_STATE_PREDICTOR_ONLY,
            SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
            predictor_variant=PREDICTOR_VARIANT_NO_GEOMETRY_BIAS,
            warm_start_policy=WARM_START_NONE,
            loss_weights=_predictor_loss(),
            uses_offline_phi=True,
            primary=False,
            diagnostic_only=True,
            final_test_policy=FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE,
            dependencies=("C0",),
        ),
        "C2": _spec(
            "C2",
            "C",
            "DeepSets/global pooled predictor",
            MODEL_KIND_STATE_PREDICTOR_ONLY,
            SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
            predictor_variant=PREDICTOR_VARIANT_DEEPSETS,
            warm_start_policy=WARM_START_NONE,
            loss_weights=_predictor_loss(),
            uses_offline_phi=True,
            primary=False,
            diagnostic_only=True,
            final_test_policy=FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE,
            dependencies=("C0",),
        ),
        "C3": _spec(
            "C3",
            "C",
            "state-only predictor",
            MODEL_KIND_STATE_PREDICTOR_ONLY,
            SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
            predictor_variant=PREDICTOR_VARIANT_STATE_ONLY,
            warm_start_policy=WARM_START_NONE,
            loss_weights=_predictor_loss(),
            uses_offline_phi=True,
            primary=False,
            diagnostic_only=True,
            final_test_policy=FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE,
            dependencies=("C0",),
        ),
        "C4": _spec(
            "C4",
            "C",
            "particle-only learned-query predictor",
            MODEL_KIND_STATE_PREDICTOR_ONLY,
            SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
            predictor_variant=PREDICTOR_VARIANT_PARTICLE_ONLY_QUERIES,
            warm_start_policy=WARM_START_NONE,
            loss_weights=_predictor_loss(),
            uses_offline_phi=True,
            primary=False,
            diagnostic_only=True,
            final_test_policy=FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE,
            dependencies=("C0",),
        ),
        "C5": _spec(
            "C5",
            "C",
            "hard-locality predictor",
            MODEL_KIND_STATE_PREDICTOR_ONLY,
            SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
            predictor_variant=PREDICTOR_VARIANT_HARD_LOCALITY,
            warm_start_policy=WARM_START_NONE,
            loss_weights=_predictor_loss(),
            uses_offline_phi=True,
            primary=False,
            diagnostic_only=True,
            final_test_policy=FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE,
            dependencies=("C0",),
        ),
        "C6": _spec(
            "C6",
            "C",
            "uncertainty predictor",
            MODEL_KIND_STATE_PREDICTOR_ONLY,
            SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
            predictor_variant=PREDICTOR_VARIANT_UNCERTAINTY,
            warm_start_policy=WARM_START_NONE,
            loss_weights=_predictor_loss(uncertainty=True),
            uses_offline_phi=True,
            primary=False,
            diagnostic_only=True,
            final_test_policy=FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE,
            dependencies=("C0",),
        ),
        "D0": _spec(
            "D0",
            "D",
            "HLT particles plus Phi_hlt and delta_Phi_hat",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_DELTA_PHI,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat"),
            loss_weights=_state_aux(),
            dependencies=("A0", "C0"),
        ),
        "D1": _spec(
            "D1",
            "D",
            "HLT particles plus Phi_hlt and Phi_pred",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_PHI_PRED,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "phi_pred"),
            loss_weights=_state_aux(),
            dependencies=("A0", "C0"),
        ),
        "D2": _spec(
            "D2",
            "D",
            "full residual-state context",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat", "phi_pred"),
            loss_weights=_state_aux(),
            dependencies=("A0", "B0", "C0"),
            notes=("main clean residual-state model",),
        ),
        "D3": _spec(
            "D3",
            "D",
            "full residual-state context repeat without logit KD",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat", "phi_pred"),
            loss_weights=_state_aux(kd=False),
            dependencies=("A0", "D2"),
            notes=("teacher-logit KD is intentionally disabled until a cache path is implemented",),
        ),
        "D4": _spec(
            "D4",
            "D",
            "D2 with auxiliary state residual supervision and no logit KD",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat", "phi_pred"),
            loss_weights=_state_aux(state=0.5),
            dependencies=("A0", "D2"),
        ),
        "D5": _spec(
            "D5",
            "D",
            "D2 architecture with CE only",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat", "phi_pred"),
            loss_weights=_ce(),
            dependencies=("A0", "D2"),
            notes=("same capacity but no explicit HLT-to-offline state supervision",),
        ),
        "E0": _spec(
            "E0",
            "E",
            "warm-start frozen warmup to upper-unfreeze",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat", "phi_pred"),
            loss_weights=_state_aux(),
            dependencies=("A0", "D2"),
        ),
        "E1": _spec(
            "E1",
            "E",
            "warm-start joint from start",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_WARMSTART_JOINT_FROM_START,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat", "phi_pred"),
            loss_weights=_state_aux(),
            dependencies=("A0", "E0"),
        ),
        "E2": _spec(
            "E2",
            "E",
            "warm-start full ParT frozen",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_FULL_PART_FROZEN_ADAPTER_HEAD,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat", "phi_pred"),
            loss_weights=_state_aux(),
            dependencies=("A0", "E0"),
        ),
        "E3": _spec(
            "E3",
            "E",
            "from-scratch canonical-state model",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_FROM_SCRATCH_CANONICAL_STATE,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat", "phi_pred"),
            warm_start_policy=WARM_START_NONE,
            loss_weights=_state_aux(),
            dependencies=("E4",),
        ),
        "E4": _spec(
            "E4",
            "E",
            "from-scratch ParT baseline",
            MODEL_KIND_PART_BASELINE,
            SCHEDULE_FROM_SCRATCH_PART_BASELINE,
            tagger_mode=STATE_CONTEXT_PARTICLES_ONLY,
            warm_start_policy=WARM_START_NONE,
            loss_weights=_ce(),
        ),
        "E5": _spec(
            "E5",
            "E",
            "residual predictor pretrain then tagger train",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_PREDICTOR_PRETRAIN_THEN_TAGGER,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat", "phi_pred"),
            warm_start_policy=WARM_START_PRETRAINED_PREDICTOR,
            loss_weights=_state_aux(),
            dependencies=("C0", "E6"),
        ),
        "E6": _spec(
            "E6",
            "E",
            "joint end-to-end without predictor pretraining",
            MODEL_KIND_STATE_CONDITIONED_PART,
            SCHEDULE_JOINT_END_TO_END_NO_PRETRAINING,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "delta_phi_hat", "phi_pred"),
            loss_weights=_state_aux(),
            dependencies=("A0",),
        ),
        "F0": _spec(
            "F0",
            "F",
            "logit fusion HLT ParT plus best canonical-state model",
            MODEL_KIND_LOGIT_FUSION,
            SCHEDULE_WARMSTART_JOINT_FROM_START,
            primary=True,
            requires_stack_splits=True,
            fusion_inputs=("A0", "D2"),
            dependencies=("A0", "D2"),
        ),
        "F1": _spec(
            "F1",
            "F",
            "logit fusion HLT ParT plus AV10 plus canonical-state",
            MODEL_KIND_LOGIT_FUSION,
            SCHEDULE_WARMSTART_JOINT_FROM_START,
            primary=True,
            requires_stack_splits=True,
            fusion_inputs=("A0", "A2", "D2", "D3"),
            dependencies=("A0", "A2", "D2", "D3"),
        ),
        "F2": _spec(
            "F2",
            "F",
            "predictor-diversity logit fusion diagnostic",
            MODEL_KIND_LOGIT_FUSION,
            SCHEDULE_WARMSTART_JOINT_FROM_START,
            primary=False,
            diagnostic_only=True,
            requires_stack_splits=True,
            fusion_inputs=("D2", "C1", "C2", "C3"),
            dependencies=("D2", "C1", "C2", "C3"),
            final_test_policy=FINAL_TEST_POLICY_STACK_ONLY,
            notes=(
                "diagnostic only: C inputs are predictor-only and emit no logits unless separate tagger counterparts are added",
            ),
        ),
        "F3": _spec(
            "F3",
            "F",
            "state-token view fusion prototype",
            MODEL_KIND_STATE_TOKEN_FUSION,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_ALL,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("phi_hlt", "phi_pred_p0", "phi_pred_p2", "delta_p0", "delta_p2"),
            primary=False,
            diagnostic_only=True,
            requires_stack_splits=False,
            dependencies=("C0", "C2", "D2"),
            loss_weights=_state_aux(),
            notes=("diagnostic prototype: current runner trains one P0 state-conditioned tagger, not multi-predictor fusion",),
        ),
        "F4": _spec(
            "F4",
            "F",
            "particle-view gated adapter fusion prototype",
            MODEL_KIND_PARTICLE_VIEW_FUSION,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_FEATURE_MLP_PLUS_STATE,
            predictor_variant=PREDICTOR_VARIANT_GEOMETRY_BIASED,
            state_context_families=("feature_mlp_delta_h", "state_context_delta_h"),
            primary=False,
            diagnostic_only=True,
            dependencies=("A2", "D2"),
            loss_weights=_state_aux(),
            notes=("diagnostic prototype: current runner trains feature-MLP plus P0 state context, not separate view fusion",),
        ),
        "Fseed": _spec(
            "Fseed",
            "F",
            "same-size HLT ParT seed ensemble",
            MODEL_KIND_SEED_ENSEMBLE,
            SCHEDULE_FROM_SCRATCH_PART_BASELINE,
            tagger_mode=STATE_CONTEXT_PARTICLES_ONLY,
            warm_start_policy=WARM_START_NONE,
            seed_count=3,
            primary=True,
            requires_stack_splits=True,
            fusion_inputs=("A0",),
            dependencies=("A0",),
            notes=("required advisor-facing fusion control",),
        ),
        "Fshuffle": _spec(
            "Fshuffle",
            "F",
            "fusion with one shuffled model/view",
            MODEL_KIND_LOGIT_FUSION,
            SCHEDULE_WARMSTART_JOINT_FROM_START,
            primary=False,
            diagnostic_only=True,
            requires_stack_splits=True,
            fusion_inputs=("A0", "D2"),
            dependencies=("F1",),
            final_test_policy=FINAL_TEST_POLICY_STACK_ONLY,
            notes=("leakage/capacity fusion control",),
        ),
        "G0": _spec(
            "G0",
            "G",
            "oracle Phi_off context on non-final validation splits",
            MODEL_KIND_ORACLE_DIAGNOSTIC,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_ORACLE_PHI_OFF,
            state_context_families=("phi_off",),
            loss_weights=_ce(),
            uses_offline_phi=True,
            oracle_inputs_allowed=True,
            primary=False,
            diagnostic_only=True,
            dependencies=("B0",),
            final_test_policy=FINAL_TEST_POLICY_MODEL_VAL_ONLY,
            notes=("evaluates model_val and stack_val only; oracle gaps are reported from model_val",),
        ),
        "G1": _spec(
            "G1",
            "G",
            "oracle delta_Phi_true context on non-final validation splits",
            MODEL_KIND_ORACLE_DIAGNOSTIC,
            SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            tagger_mode=STATE_CONTEXT_DELTA_PHI,
            state_context_families=("delta_phi_true",),
            loss_weights=_ce(),
            uses_offline_phi=True,
            oracle_inputs_allowed=True,
            primary=False,
            diagnostic_only=True,
            dependencies=("D0",),
            final_test_policy=FINAL_TEST_POLICY_MODEL_VAL_ONLY,
            notes=("uses Phi_off - Phi_hlt with the offline-valid state mask on model_val and stack_val only",),
        ),
        "G2": _spec(
            "G2",
            "G",
            "predicted Phi_pred versus oracle Phi_off gap",
            MODEL_KIND_ANALYSIS_REPORT,
            SCHEDULE_WARMSTART_JOINT_FROM_START,
            primary=False,
            diagnostic_only=True,
            dependencies=("D2", "G0", "G1"),
            final_test_policy=FINAL_TEST_POLICY_REPORT_ONLY,
        ),
        "G3": _spec(
            "G3",
            "G",
            "per-class and token-family residual error analysis",
            MODEL_KIND_ANALYSIS_REPORT,
            SCHEDULE_WARMSTART_JOINT_FROM_START,
            primary=False,
            diagnostic_only=True,
            dependencies=("C0", "C1", "C2", "C3", "C4", "C5", "C6", "D2", "D3"),
            final_test_policy=FINAL_TEST_POLICY_REPORT_ONLY,
        ),
    }
    if tuple(registry) != CANONICAL_STATE_EXPECTED_RUN_IDS:
        missing = [run_id for run_id in CANONICAL_STATE_EXPECTED_RUN_IDS if run_id not in registry]
        extra = [run_id for run_id in registry if run_id not in CANONICAL_STATE_EXPECTED_RUN_IDS]
        raise AssertionError(f"canonical-state registry drift: missing={missing}, extra={extra}")
    return registry


def canonical_state_variant_spec(run_id: str) -> CanonicalStateVariantSpec:
    registry = canonical_state_variant_registry()
    key = str(run_id)
    if key not in registry:
        raise KeyError(f"unknown canonical-state run id {run_id!r}")
    return registry[key]


def canonical_state_primary_run_ids() -> tuple[str, ...]:
    return tuple(run_id for run_id, spec in canonical_state_variant_registry().items() if bool(spec.primary))


def canonical_state_diagnostic_run_ids() -> tuple[str, ...]:
    return tuple(run_id for run_id, spec in canonical_state_variant_registry().items() if bool(spec.diagnostic_only))


def canonical_state_oracle_run_ids() -> tuple[str, ...]:
    return tuple(run_id for run_id, spec in canonical_state_variant_registry().items() if bool(spec.is_oracle))


def canonical_state_fusion_run_ids() -> tuple[str, ...]:
    return tuple(run_id for run_id, spec in canonical_state_variant_registry().items() if bool(spec.is_fusion))


def canonical_state_required_dependencies(run_id: str) -> tuple[str, ...]:
    spec = canonical_state_variant_spec(run_id)
    return tuple(dict.fromkeys((*spec.dependencies, *spec.fusion_inputs)))


def require_canonical_state_primary_final_test_allowed(run_id: str) -> CanonicalStateVariantSpec:
    spec = canonical_state_variant_spec(run_id)
    if not spec.allows_primary_final_test():
        raise ValueError(
            f"{run_id} is not allowed as a primary final_test claim "
            f"(policy={spec.final_test_policy}, primary={spec.primary}, oracle={spec.is_oracle})"
        )
    return spec


def canonical_state_registry_manifest() -> dict[str, Any]:
    registry = canonical_state_variant_registry()
    return {
        "contract": CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT,
        "run_ids": list(CANONICAL_STATE_EXPECTED_RUN_IDS),
        "primary_run_ids": list(canonical_state_primary_run_ids()),
        "diagnostic_run_ids": list(canonical_state_diagnostic_run_ids()),
        "oracle_run_ids": list(canonical_state_oracle_run_ids()),
        "fusion_run_ids": list(canonical_state_fusion_run_ids()),
        "variants": {run_id: spec.to_dict() for run_id, spec in registry.items()},
    }


__all__ = [
    "CANONICAL_STATE_EXPECTED_RUN_IDS",
    "CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT",
    "FINAL_TEST_POLICY_DIAGNOSTIC_TEACHER_FREE",
    "FINAL_TEST_POLICY_MODEL_VAL_ONLY",
    "FINAL_TEST_POLICY_PRIMARY_TEACHER_FREE",
    "FINAL_TEST_POLICY_REPORT_ONLY",
    "FINAL_TEST_POLICY_STACK_ONLY",
    "MODEL_KIND_ANALYSIS_REPORT",
    "MODEL_KIND_AV10_FEATURE_MLP",
    "MODEL_KIND_LOGIT_FUSION",
    "MODEL_KIND_ORACLE_DIAGNOSTIC",
    "MODEL_KIND_PART_BASELINE",
    "MODEL_KIND_PARTICLE_VIEW_FUSION",
    "MODEL_KIND_SEED_ENSEMBLE",
    "MODEL_KIND_STATE_CONDITIONED_PART",
    "MODEL_KIND_STATE_ONLY_TAGGER",
    "MODEL_KIND_STATE_PREDICTOR_ONLY",
    "MODEL_KIND_STATE_TOKEN_FUSION",
    "WARM_START_AV10_REFERENCE",
    "WARM_START_HLT_PART_BASELINE",
    "WARM_START_NONE",
    "WARM_START_PRETRAINED_PREDICTOR",
    "CanonicalStateVariantSpec",
    "canonical_state_diagnostic_run_ids",
    "canonical_state_expected_run_ids",
    "canonical_state_fusion_run_ids",
    "canonical_state_oracle_run_ids",
    "canonical_state_primary_run_ids",
    "canonical_state_registry_manifest",
    "canonical_state_required_dependencies",
    "canonical_state_variant_registry",
    "canonical_state_variant_spec",
    "require_canonical_state_primary_final_test_allowed",
]


def canonical_state_expected_run_ids() -> tuple[str, ...]:
    return CANONICAL_STATE_EXPECTED_RUN_IDS

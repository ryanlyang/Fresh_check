"""Campaign-aware convergence schedules for ABPH screening runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Mapping, Sequence

from .config import canonical_hash


ABPH_ACCELERATED_SCHEDULE_CONTRACT = (
    "adaptive_binary_pseudooffline_accelerated_screening_v1"
)
ABPH_SEVEN_DAY_SCHEDULE_CONTRACT = (
    "adaptive_binary_pseudooffline_accelerated_screening_v2_7day"
)
ABPH_ACCELERATED_SCHEDULE_CONTRACTS: tuple[str, ...] = (
    ABPH_ACCELERATED_SCHEDULE_CONTRACT,
    ABPH_SEVEN_DAY_SCHEDULE_CONTRACT,
)
ABPH_SCHEDULE_POLICY_BY_CONTRACT: Mapping[str, str] = {
    ABPH_ACCELERATED_SCHEDULE_CONTRACT: "accelerated_screening_v1",
    ABPH_SEVEN_DAY_SCHEDULE_CONTRACT: "accelerated_screening_v2_7day",
}
ABPH_EXTENSION_COMPARISON_CONTRACT = (
    "adaptive_binary_pseudooffline_extension_comparison_v1"
)
ABPH_LEGACY_SCHEDULE_CONTRACT = "adaptive_binary_pseudooffline_legacy_fixed_v1"
ABPH_SCHEDULE_PROFILES: tuple[str, ...] = ("pilot", "highdata")
ABPH_STAGE_FAMILIES: tuple[str, ...] = (
    "root",
    "hierarchy",
    "renderer",
    "distribution",
)
ABPH_STAGE_ROLES: tuple[str, ...] = (
    "trained",
    "warm_started_handoff",
    "disabled",
    "oracle",
)


@dataclass(frozen=True)
class StageScheduleBudget:
    nominal_updates: int
    extension_updates: int
    hard_max_updates: int

    def __post_init__(self) -> None:
        nominal = int(self.nominal_updates)
        extension = int(self.extension_updates)
        hard_max = int(self.hard_max_updates)
        if nominal <= 0:
            raise ValueError("nominal_updates must be positive")
        if hard_max < nominal:
            raise ValueError("hard_max_updates cannot be below nominal_updates")
        if extension < 0:
            raise ValueError("extension_updates must be nonnegative")
        if hard_max > nominal and extension <= 0:
            raise ValueError("a schedule with extension headroom needs a positive block")
        if extension > 0 and (hard_max - nominal) % extension:
            raise ValueError("hard maximum must contain an integer number of extensions")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageScheduleBudget":
        return cls(
            nominal_updates=int(payload["nominal_updates"]),
            extension_updates=int(payload["extension_updates"]),
            hard_max_updates=int(payload["hard_max_updates"]),
        )

    def to_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


ABPH_ACCELERATED_STAGE_BUDGETS: Mapping[
    str, Mapping[str, StageScheduleBudget]
] = {
    "pilot": {
        "root": StageScheduleBudget(12_000, 4_000, 24_000),
        "hierarchy": StageScheduleBudget(8_000, 4_000, 16_000),
        "renderer": StageScheduleBudget(15_000, 5_000, 30_000),
        "distribution": StageScheduleBudget(10_000, 5_000, 20_000),
    },
    "highdata": {
        "root": StageScheduleBudget(50_000, 10_000, 80_000),
        "hierarchy": StageScheduleBudget(40_000, 10_000, 60_000),
        "renderer": StageScheduleBudget(80_000, 20_000, 120_000),
        "distribution": StageScheduleBudget(50_000, 10_000, 80_000),
    },
}

ABPH_SEVEN_DAY_STAGE_BUDGETS: Mapping[
    str, Mapping[str, StageScheduleBudget]
] = {
    "pilot": {
        # Locked global batches make these approximately 6.1, 2.0, 3.1,
        # and 2.0 passes over the unchanged 500k-jet model_train split.
        "root": StageScheduleBudget(3_000, 1_000, 4_000),
        "hierarchy": StageScheduleBudget(1_000, 1_000, 2_000),
        "renderer": StageScheduleBudget(3_000, 1_000, 4_000),
        "distribution": StageScheduleBudget(2_000, 1_000, 3_000),
    },
    # High-data keeps the established policy until the pilot demonstrates
    # signal and resolves whether its extension blocks were needed.
    "highdata": dict(ABPH_ACCELERATED_STAGE_BUDGETS["highdata"]),
}

ABPH_STAGE_BUDGETS_BY_CONTRACT: Mapping[
    str, Mapping[str, Mapping[str, StageScheduleBudget]]
] = {
    ABPH_ACCELERATED_SCHEDULE_CONTRACT: ABPH_ACCELERATED_STAGE_BUDGETS,
    ABPH_SEVEN_DAY_SCHEDULE_CONTRACT: ABPH_SEVEN_DAY_STAGE_BUDGETS,
}


def infer_campaign_schedule_profile(
    *, model_train_jets: int, model_val_jets: int
) -> str:
    observed = (int(model_train_jets), int(model_val_jets))
    expected = {
        "pilot": (500_000, 150_000),
        "highdata": (5_000_000, 1_000_000),
    }
    matches = [name for name, sizes in expected.items() if observed == sizes]
    if len(matches) != 1:
        raise ValueError(
            "ABPH accelerated schedule must be inferred from an immutable pilot or "
            f"high-data split; observed model_train/model_val={observed}"
        )
    return matches[0]


def accelerated_stage_budget(
    profile: str,
    stage_family: str,
    *,
    schedule_contract: str = ABPH_ACCELERATED_SCHEDULE_CONTRACT,
) -> StageScheduleBudget:
    normalized_profile = str(profile).strip().lower()
    normalized_family = str(stage_family).strip().lower()
    budgets = ABPH_STAGE_BUDGETS_BY_CONTRACT.get(str(schedule_contract))
    if budgets is None:
        raise ValueError(f"unknown accelerated schedule contract {schedule_contract!r}")
    if normalized_profile not in budgets:
        raise ValueError(f"unknown accelerated schedule profile {profile!r}")
    if normalized_family not in ABPH_STAGE_FAMILIES:
        raise ValueError(f"unknown stage family {stage_family!r}")
    return budgets[normalized_profile][normalized_family]


def schedule_contract_for_policy(policy_label: str) -> str:
    normalized = str(policy_label).strip()
    matches = [
        contract
        for contract, label in ABPH_SCHEDULE_POLICY_BY_CONTRACT.items()
        if label == normalized
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown ABPH schedule policy {policy_label!r}")
    return matches[0]


def schedule_policy_for_contract(schedule_contract: str) -> str:
    try:
        return ABPH_SCHEDULE_POLICY_BY_CONTRACT[str(schedule_contract)]
    except KeyError as exc:
        raise ValueError(
            f"unknown ABPH schedule contract {schedule_contract!r}"
        ) from exc


def budget_for_stage_role(
    budget: StageScheduleBudget, role: str
) -> StageScheduleBudget:
    normalized = str(role).strip().lower()
    if normalized not in ABPH_STAGE_ROLES:
        raise ValueError(f"unknown ABPH stage role {role!r}")
    if normalized == "trained":
        return budget
    if normalized == "warm_started_handoff":
        return StageScheduleBudget(1, 0, 1)
    raise ValueError(f"{normalized} stages must not be added to the curriculum")


@dataclass(frozen=True)
class StageContinuationDecision:
    continue_training: bool
    outcome: str
    schedule_truncated: bool
    checks: Mapping[str, bool]
    robust_relative_slope: float | None
    best_checkpoint_global_update: int | None
    recent_global_updates: tuple[int, ...]
    objective_relative_degradation: Mapping[str, float]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "checks": dict(self.checks),
            "recent_global_updates": list(self.recent_global_updates),
            "objective_relative_degradation": dict(
                self.objective_relative_degradation
            ),
            "reasons": list(self.reasons),
        }


def _validation_score(row: Mapping[str, Any]) -> float:
    validation = row.get("model_val_rollout", row.get("validation", row))
    score = float(validation["selection_score"])
    if not math.isfinite(score):
        raise FloatingPointError("stage validation history contains a nonfinite score")
    return score


def _validation_metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    validation = row.get("model_val_rollout", row.get("validation", row))
    metrics = validation.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise TypeError("stage validation metrics must be a mapping")
    return metrics


def decide_stage_continuation(
    history: Sequence[Mapping[str, Any]],
    *,
    required_objectives: Sequence[str],
    best_checkpoint_global_update: int | None,
    stage_update: int,
    nominal_updates: int,
    extension_blocks_completed: int,
    hard_max_updates: int,
    nonfinite_updates: int,
    compiler_failure_updates: int,
    stage_role: str = "trained",
) -> StageContinuationDecision:
    """Apply the pre-registered nominal/extension continuation rule."""

    if not history:
        raise ValueError("stage continuation requires validation history")
    recent = tuple(history[-3:])
    recent_updates = tuple(int(row["global_update"]) for row in history[-2:])
    scores = [_validation_score(row) for row in recent]
    slope = None
    if len(scores) == 3:
        pairwise = (
            scores[1] - scores[0],
            (scores[2] - scores[0]) / 2.0,
            scores[2] - scores[1],
        )
        slope = float(statistics.median(pairwise))
    current_score = scores[-1]
    robust_relative_slope = (
        slope / max(abs(current_score), 1.0e-12) if slope is not None else None
    )
    best_is_recent = (
        best_checkpoint_global_update is not None
        and int(best_checkpoint_global_update) in recent_updates
    )
    improving = (
        robust_relative_slope is not None
        and robust_relative_slope <= -0.002
    )

    degradations: dict[str, float] = {}
    objectives_clean = True
    for name in required_objectives:
        key = f"loss.raw.{name}"
        values = []
        for row in history:
            metrics = _validation_metrics(row)
            if key not in metrics:
                objectives_clean = False
                continue
            value = float(metrics[key])
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"stage validation history contains nonfinite {key}"
                )
            values.append(value)
        if len(values) != len(history):
            degradations[name] = math.inf
            continue
        best_value = min(values)
        degradation = (values[-1] - best_value) / max(abs(best_value), 1.0e-12)
        degradations[name] = float(degradation)
        if degradation > 0.05:
            objectives_clean = False

    numerical_health_clean = (
        int(nonfinite_updates) == 0 and int(compiler_failure_updates) == 0
    )
    at_hard_max = int(stage_update) >= int(hard_max_updates)
    trained_role = str(stage_role) == "trained"
    checks = {
        "trained_stage": trained_role,
        "three_evaluations_available": len(scores) == 3,
        "best_checkpoint_in_last_two_evaluations": best_is_recent,
        "robust_relative_slope_at_most_minus_0p002": improving,
        "required_objectives_within_5_percent_of_best": objectives_clean,
        "numerical_and_compiler_health_clean": numerical_health_clean,
        "hard_max_not_reached": not at_hard_max,
    }
    eligible_except_cap = all(
        checks[name]
        for name in (
            "trained_stage",
            "three_evaluations_available",
            "best_checkpoint_in_last_two_evaluations",
            "robust_relative_slope_at_most_minus_0p002",
            "required_objectives_within_5_percent_of_best",
            "numerical_and_compiler_health_clean",
        )
    )
    continue_training = bool(eligible_except_cap and not at_hard_max)
    if continue_training:
        outcome = "extended_for_convergence"
    elif at_hard_max:
        outcome = "hard_max_reached"
    elif int(extension_blocks_completed) > 0:
        outcome = "plateau_stopped"
    else:
        outcome = "nominal_completed"
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return StageContinuationDecision(
        continue_training=continue_training,
        outcome=outcome,
        schedule_truncated=bool(at_hard_max and eligible_except_cap),
        checks=checks,
        robust_relative_slope=robust_relative_slope,
        best_checkpoint_global_update=(
            None
            if best_checkpoint_global_update is None
            else int(best_checkpoint_global_update)
        ),
        recent_global_updates=recent_updates,
        objective_relative_degradation=degradations,
        reasons=reasons,
    )


def relative_reconstruction_improvement(
    nominal_best_loss: float, extension_best_loss: float
) -> float:
    nominal = float(nominal_best_loss)
    extension = float(extension_best_loss)
    if not math.isfinite(nominal) or not math.isfinite(extension) or nominal <= 0.0:
        raise ValueError("reconstruction losses must be finite and nominal must be positive")
    return (nominal - extension) / nominal


def reconstruction_materially_improved(
    nominal_best_loss: float, extension_best_loss: float
) -> bool:
    return relative_reconstruction_improvement(
        nominal_best_loss, extension_best_loss
    ) >= 0.005


def tagging_signal_category(gain: float) -> str:
    value = float(gain)
    if not math.isfinite(value):
        raise ValueError("tagging gain must be finite")
    if value >= 0.002:
        return "positive_signal"
    if value <= -0.002:
        return "negative_signal"
    return "no_clear_signal"


def tagging_conclusion_changed(nominal_gain: float, extension_gain: float) -> bool:
    return tagging_signal_category(nominal_gain) != tagging_signal_category(
        extension_gain
    )


def build_extension_comparison_report(
    *,
    variant_name: str,
    nominal_checkpoint_hash: str,
    extension_checkpoint_hash: str,
    matched_a0_artifact_hash: str,
    frozen_tagger_recipe_hash: str,
    nominal_best_loss: float,
    extension_best_loss: float,
    nominal_tagging_gain: float,
    extension_tagging_gain: float,
    initialization_seed: int,
    training_budget_hash: str,
    evaluation_split: str = "model_val",
) -> dict[str, Any]:
    """Build the frozen nominal-versus-extension screening decision artifact."""

    hashes = {
        "nominal_checkpoint_hash": nominal_checkpoint_hash,
        "extension_checkpoint_hash": extension_checkpoint_hash,
        "matched_a0_artifact_hash": matched_a0_artifact_hash,
        "frozen_tagger_recipe_hash": frozen_tagger_recipe_hash,
        "training_budget_hash": training_budget_hash,
    }
    if not str(variant_name).strip():
        raise ValueError("variant_name is required")
    if str(evaluation_split) != "model_val":
        raise ValueError("extension comparison is restricted to model_val")
    if any(not str(value).strip() for value in hashes.values()):
        raise ValueError("extension comparison requires every artifact hash")
    relative = relative_reconstruction_improvement(
        nominal_best_loss, extension_best_loss
    )
    reconstruction_changed = relative >= 0.005
    nominal_category = tagging_signal_category(nominal_tagging_gain)
    extension_category = tagging_signal_category(extension_tagging_gain)
    tagging_changed = nominal_category != extension_category
    schedule_truncated = bool(reconstruction_changed or tagging_changed)
    report = {
        "contract": ABPH_EXTENSION_COMPARISON_CONTRACT,
        "ok": True,
        "variant_name": str(variant_name),
        "evaluation_split": "model_val",
        "final_test_loaded": False,
        **hashes,
        "initialization_seed": int(initialization_seed),
        "raw": {
            "nominal_best_loss": float(nominal_best_loss),
            "extension_best_loss": float(extension_best_loss),
            "relative_reconstruction_improvement": float(relative),
            "nominal_tagging_gain_vs_A0": float(nominal_tagging_gain),
            "extension_tagging_gain_vs_A0": float(extension_tagging_gain),
        },
        "thresholds": {
            "material_relative_reconstruction_improvement": 0.005,
            "positive_signal_accuracy_gain": 0.002,
            "negative_signal_accuracy_gain": -0.002,
        },
        "categories": {
            "nominal": nominal_category,
            "extension": extension_category,
            "tagging_conclusion_changed": tagging_changed,
        },
        "material_reconstruction_improvement": reconstruction_changed,
        "schedule_truncated": schedule_truncated,
        "screening_checkpoint_policy": (
            "use_extension_cap" if schedule_truncated else "retain_nominal"
        ),
    }
    report["report_content_hash"] = canonical_hash(report)
    return report


__all__ = [
    "ABPH_ACCELERATED_SCHEDULE_CONTRACT",
    "ABPH_ACCELERATED_SCHEDULE_CONTRACTS",
    "ABPH_ACCELERATED_STAGE_BUDGETS",
    "ABPH_EXTENSION_COMPARISON_CONTRACT",
    "ABPH_LEGACY_SCHEDULE_CONTRACT",
    "ABPH_SCHEDULE_PROFILES",
    "ABPH_SEVEN_DAY_SCHEDULE_CONTRACT",
    "ABPH_SEVEN_DAY_STAGE_BUDGETS",
    "ABPH_STAGE_FAMILIES",
    "ABPH_STAGE_ROLES",
    "StageContinuationDecision",
    "StageScheduleBudget",
    "accelerated_stage_budget",
    "budget_for_stage_role",
    "build_extension_comparison_report",
    "decide_stage_continuation",
    "infer_campaign_schedule_profile",
    "reconstruction_materially_improved",
    "relative_reconstruction_improvement",
    "schedule_contract_for_policy",
    "schedule_policy_for_contract",
    "tagging_conclusion_changed",
    "tagging_signal_category",
]

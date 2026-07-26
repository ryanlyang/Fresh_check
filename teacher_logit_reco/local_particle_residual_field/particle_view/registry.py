"""Step-1 campaign registry schema for particle-view runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .splits import PARTICLE_VIEW_FORBIDDEN_TRAINING_SPLIT_FRAGMENTS


PARTICLE_VIEW_REGISTRY_CONTRACT = "particle_view_campaign_registry_v1"
PARTICLE_VIEW_RUN_CONTRACT = "particle_view_campaign_run_v1"
PARTICLE_VIEW_QUALITY_POLICY = "warn_and_continue_scientific_metrics_v1"

PARTICLE_VIEW_SELECTION_FAMILIES = (
    "privileged_scientific",
    "pre_stage_g_deployable",
    "diagnostic",
    "infrastructure",
)
PARTICLE_VIEW_STAGES = (
    "source",
    "baseline",
    "target_screen",
    "view_publication",
    "representation",
    "predictor",
    "confirmation",
    "fairness",
    "stack",
    "report_export",
    "final_test",
)
PARTICLE_VIEW_SEEDS = (101, 202, 303)


def _validate_run_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("run_id must be a non-empty string")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise ValueError(f"run_id contains unsupported characters: {value!r}")
    lowered = value.lower()
    if "crossfit" in lowered or "cross_fit" in lowered:
        raise ValueError("cross-fit run identities are forbidden")
    return value


@dataclass(frozen=True)
class ParticleViewRunSpec:
    run_id: str
    stage: str
    scientific_role: str
    selection_family: str
    seed_ids: tuple[int, ...] = (101,)
    parent_run_ids: tuple[str, ...] = ()
    uses_labels: bool = True
    train_split: str | None = "train"
    selectable: bool = False
    diagnostic: bool = False
    clean_consumer_paired: bool = False
    robust_consumer_paired: bool = False
    stack_val_eligible: bool = False
    final_test_eligible: bool = False

    def to_payload(self) -> dict[str, Any]:
        _validate_run_spec(self)
        return {
            "contract": PARTICLE_VIEW_RUN_CONTRACT,
            "run_id": self.run_id,
            "stage": self.stage,
            "scientific_role": self.scientific_role,
            "selection_family": self.selection_family,
            "seed_ids": list(self.seed_ids),
            "parent_run_ids": list(self.parent_run_ids),
            "uses_labels": bool(self.uses_labels),
            "train_split": self.train_split,
            "selectable": bool(self.selectable),
            "diagnostic": bool(self.diagnostic),
            "single_seed_screen": tuple(self.seed_ids) == (101,),
            "three_seed_confirmation": tuple(self.seed_ids)
            == PARTICLE_VIEW_SEEDS,
            "clean_consumer_paired": bool(self.clean_consumer_paired),
            "robust_consumer_paired": bool(self.robust_consumer_paired),
            "stack_val_eligible": bool(self.stack_val_eligible),
            "final_test_eligible": bool(self.final_test_eligible),
            "quality_policy": PARTICLE_VIEW_QUALITY_POLICY,
        }


def _validate_run_spec(spec: ParticleViewRunSpec) -> None:
    _validate_run_id(spec.run_id)
    if spec.stage not in PARTICLE_VIEW_STAGES:
        raise ValueError(f"unknown particle-view stage {spec.stage!r}")
    if not isinstance(spec.scientific_role, str) or not spec.scientific_role:
        raise ValueError("scientific_role must be a non-empty string")
    if spec.selection_family not in PARTICLE_VIEW_SELECTION_FAMILIES:
        raise ValueError(f"unknown selection family {spec.selection_family!r}")
    if not spec.seed_ids or len(set(spec.seed_ids)) != len(spec.seed_ids):
        raise ValueError("seed_ids must be non-empty and unique")
    if any(seed not in PARTICLE_VIEW_SEEDS for seed in spec.seed_ids):
        raise ValueError(f"seed_ids must be drawn from {PARTICLE_VIEW_SEEDS}")
    if tuple(spec.seed_ids) not in ((101,), PARTICLE_VIEW_SEEDS):
        raise ValueError("runs must use seed 101 or the locked 101/202/303 replicas")
    for name, value in (
        ("uses_labels", spec.uses_labels),
        ("stack_val_eligible", spec.stack_val_eligible),
        ("final_test_eligible", spec.final_test_eligible),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
    if spec.uses_labels and spec.train_split != "train":
        raise ValueError("every label-using run must use the one logical train pool")
    if not spec.uses_labels and spec.train_split is not None:
        raise ValueError("non-training/inference rows must set train_split=None")
    if spec.train_split is not None:
        lowered = spec.train_split.lower()
        if any(
            fragment in lowered
            for fragment in PARTICLE_VIEW_FORBIDDEN_TRAINING_SPLIT_FRAGMENTS
        ):
            raise ValueError("consumer/distill/fold/cross-fit training splits are forbidden")
    if spec.final_test_eligible and spec.selection_family not in {
        "privileged_scientific",
        "pre_stage_g_deployable",
    }:
        raise ValueError("diagnostic/infrastructure rows cannot be final-test eligible")
    boolean_fields = (
        spec.selectable,
        spec.diagnostic,
        spec.clean_consumer_paired,
        spec.robust_consumer_paired,
        spec.stack_val_eligible,
        spec.final_test_eligible,
    )
    if any(not isinstance(value, bool) for value in boolean_fields):
        raise ValueError("registry run flags must be boolean")
    if spec.diagnostic and (
        spec.selectable or spec.stack_val_eligible or spec.final_test_eligible
    ):
        raise ValueError("diagnostic rows cannot be selectable or sealed-split eligible")
    if spec.final_test_eligible and not spec.selectable:
        raise ValueError("final-test eligibility requires selectability")
    for parent in spec.parent_run_ids:
        _validate_run_id(parent)
        if parent == spec.run_id:
            raise ValueError("run cannot depend on itself")


def _assert_acyclic(runs: Mapping[str, Mapping[str, Any]]) -> None:
    state: dict[str, int] = {}

    def visit(run_id: str) -> None:
        marker = state.get(run_id, 0)
        if marker == 1:
            raise ValueError(f"campaign registry contains a cycle at {run_id}")
        if marker == 2:
            return
        state[run_id] = 1
        for parent in runs[run_id]["parent_run_ids"]:
            if parent not in runs:
                raise ValueError(f"run {run_id} references unknown parent {parent}")
            visit(parent)
        state[run_id] = 2

    for run_id in sorted(runs):
        visit(run_id)


def build_particle_view_registry(
    *,
    unified_split_manifest_sha256: str,
    train_identity_sha256: str,
    run_specs: Sequence[ParticleViewRunSpec],
    campaign_id: str = "particle_view_500k_v1",
) -> dict[str, Any]:
    require_sha256(
        "unified_split_manifest_sha256", unified_split_manifest_sha256
    )
    require_sha256("train_identity_sha256", train_identity_sha256)
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign_id must be a non-empty string")
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in run_specs:
        payload = spec.to_payload()
        if spec.run_id in seen:
            raise ValueError(f"duplicate particle-view run_id {spec.run_id}")
        seen.add(spec.run_id)
        runs.append(payload)
    runs.sort(key=lambda row: row["run_id"])
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_REGISTRY_CONTRACT,
            "campaign_id": campaign_id,
            "unified_split_manifest_sha256": unified_split_manifest_sha256,
            "train_identity_sha256": train_identity_sha256,
            "training_topology": "single_pool_no_crossfit_v1",
            "quality_policy": PARTICLE_VIEW_QUALITY_POLICY,
            "seed_policy": {
                "screen": [101],
                "confirmation": [101, 202, 303],
            },
            "runs": runs,
        }
    )
    validate_particle_view_registry(artifact)
    return artifact


def validate_particle_view_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_hash(payload, expected_contract=PARTICLE_VIEW_REGISTRY_CONTRACT)
    expected_fields = {
        "contract",
        "campaign_id",
        "unified_split_manifest_sha256",
        "train_identity_sha256",
        "training_topology",
        "quality_policy",
        "seed_policy",
        "runs",
        "content_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("campaign registry field inventory mismatch")
    if not isinstance(payload.get("campaign_id"), str) or not payload["campaign_id"]:
        raise ValueError("campaign registry campaign_id is invalid")
    require_sha256(
        "unified_split_manifest_sha256",
        payload.get("unified_split_manifest_sha256"),
    )
    require_sha256(
        "train_identity_sha256",
        payload.get("train_identity_sha256"),
    )
    if payload.get("training_topology") != "single_pool_no_crossfit_v1":
        raise ValueError("registry training topology must be single-pool")
    if payload.get("quality_policy") != PARTICLE_VIEW_QUALITY_POLICY:
        raise ValueError("scientific metrics must use warn-and-continue policy")
    if payload.get("seed_policy") != {
        "screen": [101],
        "confirmation": [101, 202, 303],
    }:
        raise ValueError("registry seed policy mismatch")
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("registry runs must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    family_counts = {family: 0 for family in PARTICLE_VIEW_SELECTION_FAMILIES}
    for raw in raw_runs:
        if not isinstance(raw, Mapping):
            raise ValueError("registry run must be an object")
        expected_fields = {
            "contract",
            "run_id",
            "stage",
            "scientific_role",
            "selection_family",
            "seed_ids",
            "parent_run_ids",
            "uses_labels",
            "train_split",
            "selectable",
            "diagnostic",
            "single_seed_screen",
            "three_seed_confirmation",
            "clean_consumer_paired",
            "robust_consumer_paired",
            "stack_val_eligible",
            "final_test_eligible",
            "quality_policy",
        }
        if set(raw) != expected_fields or raw.get("contract") != PARTICLE_VIEW_RUN_CONTRACT:
            raise ValueError("registry run schema mismatch")
        spec = ParticleViewRunSpec(
            run_id=str(raw["run_id"]),
            stage=str(raw["stage"]),
            scientific_role=str(raw["scientific_role"]),
            selection_family=str(raw["selection_family"]),
            seed_ids=tuple(raw["seed_ids"]),
            parent_run_ids=tuple(raw["parent_run_ids"]),
            uses_labels=raw["uses_labels"],
            train_split=raw["train_split"],
            selectable=raw["selectable"],
            diagnostic=raw["diagnostic"],
            clean_consumer_paired=raw["clean_consumer_paired"],
            robust_consumer_paired=raw["robust_consumer_paired"],
            stack_val_eligible=raw["stack_val_eligible"],
            final_test_eligible=raw["final_test_eligible"],
        )
        _validate_run_spec(spec)
        if raw.get("quality_policy") != PARTICLE_VIEW_QUALITY_POLICY:
            raise ValueError("run quality policy mismatch")
        if spec.to_payload() != dict(raw):
            raise ValueError(f"run payload is not canonical: {spec.run_id}")
        if spec.run_id in by_id:
            raise ValueError(f"duplicate run_id {spec.run_id}")
        by_id[spec.run_id] = raw
        family_counts[spec.selection_family] += 1
    if list(by_id) != sorted(by_id):
        raise ValueError("campaign registry runs must be sorted by run_id")
    _assert_acyclic(by_id)
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "run_count": len(by_id),
        "selection_family_counts": family_counts,
        "single_training_pool": all(
            (not bool(run["uses_labels"])) or run["train_split"] == "train"
            for run in by_id.values()
        ),
        "train_identity_sha256": payload["train_identity_sha256"],
    }


__all__ = [
    "PARTICLE_VIEW_QUALITY_POLICY",
    "PARTICLE_VIEW_REGISTRY_CONTRACT",
    "PARTICLE_VIEW_RUN_CONTRACT",
    "PARTICLE_VIEW_SEEDS",
    "PARTICLE_VIEW_SELECTION_FAMILIES",
    "PARTICLE_VIEW_STAGES",
    "ParticleViewRunSpec",
    "build_particle_view_registry",
    "validate_particle_view_registry",
]

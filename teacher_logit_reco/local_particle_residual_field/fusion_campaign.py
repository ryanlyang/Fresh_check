"""Locked contracts for the A0/P7b fusion and independent-seed control campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping


LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_CONTRACT = "local_residual_field_p7b_fusion_campaign_v1"
LOCAL_RESIDUAL_FIELD_SELECTED_FUSION_CONTRACT = "local_residual_field_selected_fusion_v1"

FUSION_MEMBER_A0 = "A0"
FUSION_MEMBER_A0_SEED1 = "A0_seed1"
FUSION_MEMBER_P7B = "P7b"
LEGACY_WARM_START_A1 = "A1"
INDEPENDENT_SEED_DISPLAY_ALIAS = "A1_independent"
INDEPENDENT_SEED_CANONICAL_CONFIG_ID = "A_recipe-A0-from-scratch_seed-20522"

FUSION_GROUP_METHOD = "F_method"
FUSION_GROUP_SEED = "F_seed"

FUSION_FAMILY_LATE = "late"
FUSION_FAMILY_REPRESENTATION = "representation"
FUSION_FAMILIES = (FUSION_FAMILY_LATE, FUSION_FAMILY_REPRESENTATION)

FUSION_FIT_SPLIT = "stack_train"
FUSION_SELECTION_SPLIT = "stack_val"
FUSION_FINAL_SPLIT = "final_test"
FUSION_FINAL_TEST_STATUS = "exploratory_previously_partially_opened"
FUSION_HEAD_SEEDS = (5101, 5102, 5103)

FUSION_CHAMPION_ACCURACY = "accuracy_champion"
FUSION_CHAMPION_REJECTION = "rejection_champion"
FUSION_CHAMPION_ROLES = (FUSION_CHAMPION_ACCURACY, FUSION_CHAMPION_REJECTION)

FUSION_CANDIDATE_IDS = (
    "L0_mean_logits",
    "L1_mean_probs",
    "L2_temp_mean_logits",
    "L3_scalar_simplex_logits",
    "L4_classwise_simplex_logits",
    "L5_linear_stacker",
    "R0_linear_embeddings",
    "R1_mlp_embeddings_logits",
    "R2_scalar_event_gate",
    "R3_classwise_event_gate",
    "R4_A0_anchored_residual",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stable_fusion_json_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean_nonempty(value: str, *, field_name: str) -> str:
    output = str(value).strip()
    if not output:
        raise ValueError(f"{field_name} must be non-empty")
    return output


def _clean_sha256(value: str, *, field_name: str) -> str:
    output = _clean_nonempty(value, field_name=field_name).lower()
    if len(output) != 64 or any(character not in "0123456789abcdef" for character in output):
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    return output


@dataclass(frozen=True)
class FusionMemberSpec:
    """One frozen deployable member available to the fusion campaign."""

    run_id: str
    display_alias: str
    role: str
    canonical_config_id: str
    training_seed: int | None
    from_scratch: bool
    warm_start_source: str | None = None
    runtime_inputs: str = "HLT_only"
    uses_true_fields: bool = False
    uses_offline_particles: bool = False
    uses_teacher_logits_at_runtime: bool = False
    deployable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _clean_nonempty(self.run_id, field_name="run_id"))
        object.__setattr__(self, "display_alias", _clean_nonempty(self.display_alias, field_name="display_alias"))
        object.__setattr__(self, "role", _clean_nonempty(self.role, field_name="role"))
        object.__setattr__(
            self,
            "canonical_config_id",
            _clean_nonempty(self.canonical_config_id, field_name="canonical_config_id"),
        )
        if self.role not in {"anchor", "independent_seed_control", "curriculum_student"}:
            raise ValueError(f"unsupported fusion member role {self.role!r}")
        if self.role == "independent_seed_control":
            if self.run_id == LEGACY_WARM_START_A1:
                raise ValueError("legacy A1 is a warm-start control and cannot be the independent seed member")
            if self.run_id != FUSION_MEMBER_A0_SEED1:
                raise ValueError(f"independent seed member must use run_id {FUSION_MEMBER_A0_SEED1!r}")
            if self.display_alias != INDEPENDENT_SEED_DISPLAY_ALIAS:
                raise ValueError(f"independent seed display alias must be {INDEPENDENT_SEED_DISPLAY_ALIAS!r}")
            if not bool(self.from_scratch) or self.warm_start_source is not None:
                raise ValueError("independent seed member must be from scratch with no warm-start source")
        if self.training_seed is not None:
            object.__setattr__(self, "training_seed", int(self.training_seed))
        object.__setattr__(
            self,
            "warm_start_source",
            None if self.warm_start_source is None else str(self.warm_start_source),
        )
        if self.runtime_inputs != "HLT_only":
            raise ValueError("fusion members must declare runtime_inputs='HLT_only'")
        if self.uses_true_fields or self.uses_offline_particles or self.uses_teacher_logits_at_runtime:
            raise ValueError("fusion members cannot require privileged inputs at runtime")
        if not self.deployable:
            raise ValueError("fusion members must be deployable")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class FusionGroupSpec:
    """Ordered two-member comparison group."""

    group_id: str
    member_ids: tuple[str, ...]
    purpose: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _clean_nonempty(self.group_id, field_name="group_id"))
        members = tuple(_clean_nonempty(item, field_name="member_id") for item in self.member_ids)
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "purpose", _clean_nonempty(self.purpose, field_name="purpose"))
        if len(members) != 2 or len(set(members)) != 2:
            raise ValueError("fusion groups must contain exactly two distinct ordered members")
        if members[0] != FUSION_MEMBER_A0:
            raise ValueError("A0 must be member A (the first member) in every primary fusion group")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class FusionCandidateSpec:
    """One predeclared candidate in the finite fusion search space."""

    candidate_id: str
    family: str
    formula: str
    trainable: bool
    requires_representations: bool
    hyperparameter_grid: Mapping[str, Any] = field(default_factory=dict)
    max_trainable_parameters: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _clean_nonempty(self.candidate_id, field_name="candidate_id"))
        object.__setattr__(self, "family", _clean_nonempty(self.family, field_name="family"))
        object.__setattr__(self, "formula", _clean_nonempty(self.formula, field_name="formula"))
        if self.family not in FUSION_FAMILIES:
            raise ValueError(f"fusion candidate family must be one of {FUSION_FAMILIES}")
        if bool(self.requires_representations) != (self.family == FUSION_FAMILY_REPRESENTATION):
            raise ValueError("representation requirement must agree with the candidate family")
        grid = _jsonable(dict(self.hyperparameter_grid or {}))
        object.__setattr__(self, "hyperparameter_grid", grid)
        limit = int(self.max_trainable_parameters)
        if limit < 0:
            raise ValueError("max_trainable_parameters cannot be negative")
        if not self.trainable and limit != 0:
            raise ValueError("non-trainable candidates must have a zero parameter limit")
        if self.family == FUSION_FAMILY_REPRESENTATION and limit > 1_000_000:
            raise ValueError("representation fusion parameter cap cannot exceed 1,000,000")
        object.__setattr__(self, "max_trainable_parameters", limit)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def default_fusion_member_specs() -> tuple[FusionMemberSpec, ...]:
    return (
        FusionMemberSpec(
            run_id=FUSION_MEMBER_A0,
            display_alias=FUSION_MEMBER_A0,
            role="anchor",
            canonical_config_id="A_recipe-A0-from-scratch_seed-20421",
            training_seed=20421,
            from_scratch=True,
        ),
        FusionMemberSpec(
            run_id=FUSION_MEMBER_A0_SEED1,
            display_alias=INDEPENDENT_SEED_DISPLAY_ALIAS,
            role="independent_seed_control",
            canonical_config_id=INDEPENDENT_SEED_CANONICAL_CONFIG_ID,
            training_seed=20522,
            from_scratch=True,
        ),
        FusionMemberSpec(
            run_id=FUSION_MEMBER_P7B,
            display_alias=FUSION_MEMBER_P7B,
            role="curriculum_student",
            canonical_config_id="P_recipe-P7b_consumer-selected_studentinit-selected-consumer",
            training_seed=None,
            from_scratch=False,
            warm_start_source="selected_consumer",
        ),
    )


def default_fusion_group_specs() -> tuple[FusionGroupSpec, ...]:
    return (
        FusionGroupSpec(
            group_id=FUSION_GROUP_METHOD,
            member_ids=(FUSION_MEMBER_A0, FUSION_MEMBER_P7B),
            purpose="curriculum-method complementarity",
        ),
        FusionGroupSpec(
            group_id=FUSION_GROUP_SEED,
            member_ids=(FUSION_MEMBER_A0, FUSION_MEMBER_A0_SEED1),
            purpose="ordinary independent-seed diversity control",
        ),
    )


def default_fusion_candidate_specs() -> tuple[FusionCandidateSpec, ...]:
    late = (
        FusionCandidateSpec("L0_mean_logits", "late", "(z_A + z_B) / 2", False, False),
        FusionCandidateSpec("L1_mean_probs", "late", "(p_A + p_B) / 2", False, False),
        FusionCandidateSpec(
            "L2_temp_mean_logits",
            "late",
            "mean of stack-train temperature-scaled member logits",
            True,
            False,
            {"temperature_bounds": [0.25, 5.0]},
            2,
        ),
        FusionCandidateSpec(
            "L3_scalar_simplex_logits",
            "late",
            "w*z_A + (1-w)*z_B",
            True,
            False,
            {"weight_grid": [round(index / 100.0, 2) for index in range(101)]},
            1,
        ),
        FusionCandidateSpec(
            "L4_classwise_simplex_logits",
            "late",
            "w_c*z_A,c + (1-w_c)*z_B,c",
            True,
            False,
            {"l2": [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]},
            10,
        ),
        FusionCandidateSpec(
            "L5_linear_stacker",
            "late",
            "regularized multiclass linear stacker",
            True,
            False,
            {
                "feature_modes": ["logits", "probabilities", "logits+probabilities"],
                "C": [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0],
            },
            1_000,
        ),
    )
    representation_grid = {
        "hidden_width": [64, 128],
        "dropout": [0.0, 0.1],
        "weight_decay": [1.0e-5, 1.0e-4],
        "max_hidden_layers": 1,
    }
    representation = (
        FusionCandidateSpec(
            "R0_linear_embeddings",
            "representation",
            "linear classifier over frozen embedding interactions",
            True,
            True,
            {"weight_decay": [1.0e-5, 1.0e-4]},
            1_000_000,
        ),
        FusionCandidateSpec(
            "R1_mlp_embeddings_logits",
            "representation",
            "one-hidden-layer GELU head",
            True,
            True,
            representation_grid,
            1_000_000,
        ),
        FusionCandidateSpec(
            "R2_scalar_event_gate",
            "representation",
            "g(x)*z_A + (1-g(x))*z_B",
            True,
            True,
            representation_grid,
            1_000_000,
        ),
        FusionCandidateSpec(
            "R3_classwise_event_gate",
            "representation",
            "g_c(x)*z_A,c + (1-g_c(x))*z_B,c",
            True,
            True,
            representation_grid,
            1_000_000,
        ),
        FusionCandidateSpec(
            "R4_A0_anchored_residual",
            "representation",
            "z_A + delta_z(x)",
            True,
            True,
            representation_grid,
            1_000_000,
        ),
    )
    return late + representation


@dataclass(frozen=True)
class FusionCampaignConfig:
    """Validated finite campaign registry and split policy."""

    campaign_id: str
    members: tuple[FusionMemberSpec, ...] = field(default_factory=default_fusion_member_specs)
    groups: tuple[FusionGroupSpec, ...] = field(default_factory=default_fusion_group_specs)
    candidates: tuple[FusionCandidateSpec, ...] = field(default_factory=default_fusion_candidate_specs)
    fit_split: str = FUSION_FIT_SPLIT
    selection_split: str = FUSION_SELECTION_SPLIT
    final_split: str = FUSION_FINAL_SPLIT
    final_test_status: str = FUSION_FINAL_TEST_STATUS
    head_seeds: tuple[int, ...] = FUSION_HEAD_SEEDS
    contract: str = LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_CONTRACT

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_id", _clean_nonempty(self.campaign_id, field_name="campaign_id"))
        if self.contract != LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_CONTRACT:
            raise ValueError("unexpected fusion campaign contract")
        members = tuple(self.members)
        groups = tuple(self.groups)
        candidates = tuple(self.candidates)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "candidates", candidates)
        member_ids = tuple(member.run_id for member in members)
        if member_ids != (FUSION_MEMBER_A0, FUSION_MEMBER_A0_SEED1, FUSION_MEMBER_P7B):
            raise ValueError("campaign members must be ordered exactly as A0, A0_seed1, P7b")
        if members[0].training_seed != 20421 or members[1].training_seed != 20522:
            raise ValueError("campaign anchor and independent seed are locked to seeds 20421 and 20522")
        expected_groups = {
            FUSION_GROUP_METHOD: (FUSION_MEMBER_A0, FUSION_MEMBER_P7B),
            FUSION_GROUP_SEED: (FUSION_MEMBER_A0, FUSION_MEMBER_A0_SEED1),
        }
        actual_groups = {group.group_id: group.member_ids for group in groups}
        if actual_groups != expected_groups or len(groups) != len(expected_groups):
            raise ValueError(f"campaign groups must match {expected_groups}")
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if candidate_ids != FUSION_CANDIDATE_IDS:
            raise ValueError("campaign candidate registry is not the locked L0-L5/R0-R4 search space")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("campaign candidate IDs must be unique")
        expected_members = [member.to_dict() for member in default_fusion_member_specs()]
        if [member.to_dict() for member in members] != expected_members:
            raise ValueError("campaign member specifications differ from the locked member registry")
        expected_groups_full = [group.to_dict() for group in default_fusion_group_specs()]
        if [group.to_dict() for group in groups] != expected_groups_full:
            raise ValueError("campaign group specifications differ from the locked group registry")
        expected_candidates = [candidate.to_dict() for candidate in default_fusion_candidate_specs()]
        if [candidate.to_dict() for candidate in candidates] != expected_candidates:
            raise ValueError("campaign candidate specifications differ from the locked candidate registry")
        if (self.fit_split, self.selection_split, self.final_split) != (
            FUSION_FIT_SPLIT,
            FUSION_SELECTION_SPLIT,
            FUSION_FINAL_SPLIT,
        ):
            raise ValueError("campaign split roles are locked to stack_train/stack_val/final_test")
        if self.final_split in {self.fit_split, self.selection_split}:
            raise ValueError("final split cannot be used for fitting or selection")
        if self.final_test_status != FUSION_FINAL_TEST_STATUS:
            raise ValueError("current final-test must be labeled as previously partially opened")
        seeds = tuple(int(seed) for seed in self.head_seeds)
        if seeds != FUSION_HEAD_SEEDS:
            raise ValueError(f"fusion head seeds are locked to {FUSION_HEAD_SEEDS}")
        object.__setattr__(self, "head_seeds", seeds)

    @property
    def candidate_registry_hash(self) -> str:
        return stable_fusion_json_hash([candidate.to_dict() for candidate in self.candidates])

    def to_dict(self) -> dict[str, Any]:
        payload = _jsonable(asdict(self))
        payload["candidate_registry_hash"] = self.candidate_registry_hash
        return payload


@dataclass(frozen=True)
class SelectedFusionArtifact:
    """Selection record that must exist before a final split can be opened."""

    campaign_id: str
    selection_timestamp: str
    champion_role: str
    group_id: str
    member_ids: tuple[str, ...]
    member_checkpoint_hashes: Mapping[str, str]
    candidate_id: str
    hyperparameters: Mapping[str, Any]
    fit_artifact_hashes: Mapping[str, str]
    selection_metrics: Mapping[str, Any]
    tie_break_trace: tuple[Mapping[str, Any], ...]
    candidate_registry_hash: str
    final_test_status: str = FUSION_FINAL_TEST_STATUS
    selection_source: str = FUSION_SELECTION_SPLIT
    contract: str = LOCAL_RESIDUAL_FIELD_SELECTED_FUSION_CONTRACT

    def __post_init__(self) -> None:
        for name in ("campaign_id", "selection_timestamp", "group_id", "candidate_id", "candidate_registry_hash"):
            object.__setattr__(self, name, _clean_nonempty(getattr(self, name), field_name=name))
        if self.contract != LOCAL_RESIDUAL_FIELD_SELECTED_FUSION_CONTRACT:
            raise ValueError("unexpected selected-fusion contract")
        if self.selection_source != FUSION_SELECTION_SPLIT:
            raise ValueError("selected fusion must use stack_val as selection_source")
        if self.champion_role not in FUSION_CHAMPION_ROLES:
            raise ValueError(f"champion_role must be one of {FUSION_CHAMPION_ROLES}")
        if self.group_id not in {FUSION_GROUP_METHOD, FUSION_GROUP_SEED}:
            raise ValueError("selected fusion has an unknown primary group")
        if self.candidate_id not in FUSION_CANDIDATE_IDS:
            raise ValueError("selected fusion candidate is not in the locked registry")
        members = tuple(str(item) for item in self.member_ids)
        expected = {
            FUSION_GROUP_METHOD: (FUSION_MEMBER_A0, FUSION_MEMBER_P7B),
            FUSION_GROUP_SEED: (FUSION_MEMBER_A0, FUSION_MEMBER_A0_SEED1),
        }[self.group_id]
        if members != expected:
            raise ValueError(f"selected fusion members must match ordered group {expected}")
        object.__setattr__(self, "member_ids", members)
        checkpoint_hashes = {
            str(key): _clean_sha256(value, field_name=f"checkpoint hash for {key}")
            for key, value in dict(self.member_checkpoint_hashes).items()
        }
        if set(checkpoint_hashes) != set(members):
            raise ValueError("selected fusion checkpoint hashes must cover exactly the selected members")
        if not self.fit_artifact_hashes:
            raise ValueError("selected fusion must bind at least one fit artifact hash")
        object.__setattr__(self, "member_checkpoint_hashes", checkpoint_hashes)
        object.__setattr__(
            self,
            "fit_artifact_hashes",
            {
                str(key): _clean_sha256(value, field_name=f"fit artifact hash for {key}")
                for key, value in dict(self.fit_artifact_hashes).items()
            },
        )
        object.__setattr__(self, "hyperparameters", _jsonable(dict(self.hyperparameters)))
        object.__setattr__(self, "selection_metrics", _jsonable(dict(self.selection_metrics)))
        object.__setattr__(self, "tie_break_trace", tuple(_jsonable(dict(row)) for row in self.tie_break_trace))
        if self.final_test_status != FUSION_FINAL_TEST_STATUS:
            raise ValueError("selected fusion must preserve the current final-test status label")
        _clean_sha256(self.candidate_registry_hash, field_name="candidate_registry_hash")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def validate_selected_fusion_artifact(
    artifact: SelectedFusionArtifact,
    campaign: FusionCampaignConfig,
) -> SelectedFusionArtifact:
    if artifact.campaign_id != campaign.campaign_id:
        raise ValueError("selected fusion campaign_id does not match the campaign")
    if artifact.candidate_registry_hash != campaign.candidate_registry_hash:
        raise ValueError("selected fusion candidate registry hash does not match the campaign")
    return artifact


__all__ = [
    "LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_SELECTED_FUSION_CONTRACT",
    "FUSION_MEMBER_A0",
    "FUSION_MEMBER_A0_SEED1",
    "FUSION_MEMBER_P7B",
    "LEGACY_WARM_START_A1",
    "INDEPENDENT_SEED_DISPLAY_ALIAS",
    "INDEPENDENT_SEED_CANONICAL_CONFIG_ID",
    "FUSION_GROUP_METHOD",
    "FUSION_GROUP_SEED",
    "FUSION_FAMILY_LATE",
    "FUSION_FAMILY_REPRESENTATION",
    "FUSION_FAMILIES",
    "FUSION_FIT_SPLIT",
    "FUSION_SELECTION_SPLIT",
    "FUSION_FINAL_SPLIT",
    "FUSION_FINAL_TEST_STATUS",
    "FUSION_HEAD_SEEDS",
    "FUSION_CHAMPION_ACCURACY",
    "FUSION_CHAMPION_REJECTION",
    "FUSION_CHAMPION_ROLES",
    "FUSION_CANDIDATE_IDS",
    "FusionMemberSpec",
    "FusionGroupSpec",
    "FusionCandidateSpec",
    "FusionCampaignConfig",
    "SelectedFusionArtifact",
    "default_fusion_member_specs",
    "default_fusion_group_specs",
    "default_fusion_candidate_specs",
    "stable_fusion_json_hash",
    "validate_selected_fusion_artifact",
]

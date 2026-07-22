from __future__ import annotations

from dataclasses import replace

import pytest

from teacher_logit_reco.local_particle_residual_field import (
    FUSION_CANDIDATE_IDS,
    FUSION_CHAMPION_ACCURACY,
    FUSION_FINAL_TEST_STATUS,
    FUSION_GROUP_METHOD,
    FUSION_GROUP_SEED,
    FUSION_MEMBER_A0,
    FUSION_MEMBER_A0_SEED1,
    FUSION_MEMBER_P7B,
    FusionCampaignConfig,
    FusionMemberSpec,
    SelectedFusionArtifact,
    validate_selected_fusion_artifact,
)


def test_step1_default_registry_locks_members_groups_candidates_and_splits() -> None:
    campaign = FusionCampaignConfig(campaign_id="pilot_001")

    assert tuple(member.run_id for member in campaign.members) == ("A0", "A0_seed1", "P7b")
    assert {group.group_id: group.member_ids for group in campaign.groups} == {
        FUSION_GROUP_METHOD: (FUSION_MEMBER_A0, FUSION_MEMBER_P7B),
        FUSION_GROUP_SEED: (FUSION_MEMBER_A0, FUSION_MEMBER_A0_SEED1),
    }
    assert tuple(candidate.candidate_id for candidate in campaign.candidates) == FUSION_CANDIDATE_IDS
    assert campaign.fit_split == "stack_train"
    assert campaign.selection_split == "stack_val"
    assert campaign.final_split == "final_test"
    assert campaign.final_test_status == FUSION_FINAL_TEST_STATUS
    assert len(campaign.candidate_registry_hash) == 64


def test_step1_rejects_legacy_warm_start_a1_as_independent_seed() -> None:
    with pytest.raises(ValueError, match="legacy A1 is a warm-start control"):
        FusionMemberSpec(
            run_id="A1",
            display_alias="A1_independent",
            role="independent_seed_control",
            canonical_config_id="bad",
            training_seed=20522,
            from_scratch=True,
        )


def test_step1_rejects_privileged_runtime_member() -> None:
    with pytest.raises(ValueError, match="privileged inputs"):
        FusionMemberSpec(
            run_id="P7b",
            display_alias="P7b",
            role="curriculum_student",
            canonical_config_id="P7b",
            training_seed=None,
            from_scratch=False,
            warm_start_source="selected_consumer",
            uses_teacher_logits_at_runtime=True,
        )


def test_step1_rejects_any_split_policy_change() -> None:
    with pytest.raises(ValueError, match="split roles are locked"):
        FusionCampaignConfig(campaign_id="bad", fit_split="final_test")


def test_step1_rejects_candidate_drift_even_when_ids_are_unchanged() -> None:
    campaign = FusionCampaignConfig(campaign_id="locked")
    changed = (replace(campaign.candidates[0], formula="retrospective new formula"), *campaign.candidates[1:])

    with pytest.raises(ValueError, match="candidate specifications differ"):
        FusionCampaignConfig(campaign_id="bad", candidates=changed)


def test_step1_selected_artifact_binds_campaign_registry_and_ordered_members() -> None:
    campaign = FusionCampaignConfig(campaign_id="pilot_001")
    selected = SelectedFusionArtifact(
        campaign_id=campaign.campaign_id,
        selection_timestamp="2026-07-22T12:00:00Z",
        champion_role=FUSION_CHAMPION_ACCURACY,
        group_id=FUSION_GROUP_METHOD,
        member_ids=(FUSION_MEMBER_A0, FUSION_MEMBER_P7B),
        member_checkpoint_hashes={FUSION_MEMBER_A0: "a" * 64, FUSION_MEMBER_P7B: "b" * 64},
        candidate_id="L0_mean_logits",
        hyperparameters={},
        fit_artifact_hashes={"candidate": "c" * 64},
        selection_metrics={"accuracy": 0.75},
        tie_break_trace=({"rule": "accuracy"},),
        candidate_registry_hash=campaign.candidate_registry_hash,
    )

    assert validate_selected_fusion_artifact(selected, campaign) is selected
    with pytest.raises(ValueError, match="candidate.registry.hash"):
        validate_selected_fusion_artifact(replace(selected, candidate_registry_hash="bad"), campaign)


def test_step1_selected_artifact_rejects_reordered_group_members() -> None:
    campaign = FusionCampaignConfig(campaign_id="pilot_001")
    with pytest.raises(ValueError, match="ordered group"):
        SelectedFusionArtifact(
            campaign_id=campaign.campaign_id,
            selection_timestamp="2026-07-22T12:00:00Z",
            champion_role=FUSION_CHAMPION_ACCURACY,
            group_id=FUSION_GROUP_SEED,
            member_ids=(FUSION_MEMBER_A0_SEED1, FUSION_MEMBER_A0),
            member_checkpoint_hashes={FUSION_MEMBER_A0: "a", FUSION_MEMBER_A0_SEED1: "b"},
            candidate_id="L0_mean_logits",
            hyperparameters={},
            fit_artifact_hashes={"candidate": "c"},
            selection_metrics={"accuracy": 0.75},
            tie_break_trace=(),
            candidate_registry_hash=campaign.candidate_registry_hash,
        )

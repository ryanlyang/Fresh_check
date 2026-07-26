from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from teacher_logit_reco.local_particle_residual_field.particle_view import (
    CANONICAL_TARGET_DISCOVERY_RUN_ID,
    CandidateViewSet,
    IndexedCandidateViewProvider,
    MatchingFreeParticleViewGenerator,
    ParticleViewGeneratorConfig,
    RecoveryProbeConfig,
    FixedCapacityRecoveryProbe,
    StagedContextualMemory,
    StagedDiscoveryViewProvider,
    TARGET_SCREEN_IDS,
    build_tap_stage_reservation,
    build_target_screen_recipe,
    fit_particle_view_normalizer,
    lorentz_vectors_to_particle_geometry,
    particle_view_generator_config_from_payload,
    predict_recovery_probe_views,
    stage_teacher_tap_float16,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _geometry(
    pt: torch.Tensor,
    eta: torch.Tensor,
    phi: torch.Tensor,
    mass: torch.Tensor,
) -> torch.Tensor:
    return torch.stack((pt, eta, phi, mass), dim=-1)


def test_target_runtime_compiles_every_declared_screen_deterministically():
    recipes = {
        run_id: build_target_screen_recipe(run_id)
        for run_id in TARGET_SCREEN_IDS
    }
    assert len(recipes) == 36
    assert len({recipe.content_hash for recipe in recipes.values()}) == 36
    canonical = recipes[CANONICAL_TARGET_DISCOVERY_RUN_ID]
    assert canonical.query_tap_choice == "penultimate"
    assert canonical.memory_tap_choice == "penultimate"
    assert canonical.generator_config.num_cross_attention_blocks == 2
    assert canonical.generator_config.bottleneck_width == 4
    assert canonical.selection_status == "canonical_selectable"

    assert recipes["VGEN_QUERY_RAW"].generator_config.query_dim == 17
    assert recipes["VGEN_TAP_MIX3"].memory_tap_choice == "mix_last3"
    assert recipes["VGEN_XATTN4"].generator_config.num_cross_attention_blocks == 4
    assert recipes["VGEN_NO_PAIR"].generator_config.use_pair_bias is False
    assert recipes["VGEN_LOCAL02"].generator_config.hard_local_radius == 0.2
    assert recipes["VGEN_LOCAL04"].generator_config.hard_local_radius == 0.4
    assert recipes["VGEN_UNCENTERED"].generator_config.center_output is False
    assert recipes["VGEN_DIM8"].generator_config.bottleneck_width == 8
    assert recipes["VGEN_KD100"].oracle_objective.offline_kd_weight == 1.0
    assert recipes["VGEN_NO_RATE"].oracle_objective.rate_budget_enabled is False
    assert recipes["VGEN_TEACHER_LARGE"].generator_config.memory_dim == 192
    assert recipes["VGEN_TEACHER_MIX2"].generator_config.memory_dim == 160
    assert recipes["VGEN_MEMORY_HLT"].generator_config.memory_source == "hlt"
    assert (
        recipes["VGEN_MEMORY_HLT_SELFMASK"].generator_config.self_mask_same_particle
        is True
    )
    assert (
        recipes["VGEN_MEMORY_HLT"].oracle_objective.offline_kd_weight == 0.0
    )


def test_generator_config_round_trip_and_local_radius_masks_only_far_keys():
    config = ParticleViewGeneratorConfig(
        query_dim=3,
        memory_dim=3,
        width=8,
        num_heads=2,
        num_cross_attention_blocks=1,
        bottleneck_width=2,
        hard_local_radius=0.2,
    )
    assert particle_view_generator_config_from_payload(
        config.to_payload()
    ) == config
    with pytest.raises(ValueError, match="0.2/0.4"):
        ParticleViewGeneratorConfig(
            query_dim=3,
            memory_dim=3,
            hard_local_radius=0.3,
        )

    query = torch.randn(1, 1, 3)
    memory = torch.randn(1, 2, 3)
    query_geometry = _geometry(
        torch.tensor([[10.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
    )
    memory_geometry = _geometry(
        torch.tensor([[8.0, 7.0]]),
        torch.tensor([[0.1, 1.0]]),
        torch.tensor([[0.0, 0.0]]),
        torch.zeros(1, 2),
    )
    output = MatchingFreeParticleViewGenerator(config).eval()(
        query,
        memory,
        query_geometry=query_geometry,
        memory_geometry=memory_geometry,
        query_mask=torch.ones(1, 1, dtype=torch.bool),
        memory_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    # The final attention key is the learned null token.
    assert torch.all(output.attention[..., 0] > 0)
    assert torch.count_nonzero(output.attention[..., 1]) == 0
    assert torch.all(output.attention[..., 2] > 0)


def test_lorentz_geometry_is_physical_masked_and_finite():
    vectors = torch.tensor(
        [
            [
                [3.0, 0.0],
                [4.0, 0.0],
                [0.0, 0.0],
                [5.0, 0.0],
            ]
        ]
    )
    mask = torch.tensor([[[True, False]]])
    geometry = lorentz_vectors_to_particle_geometry(vectors, mask)
    assert geometry.shape == (1, 2, 4)
    assert geometry[0, 0].tolist() == pytest.approx([5.0, 0.0, 0.9272952, 0.0])
    assert torch.count_nonzero(geometry[0, 1]) == 0
    assert torch.isfinite(geometry).all()


class _FrozenQuery:
    def __call__(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        tokens = features.transpose(1, 2).detach()
        return SimpleNamespace(
            single_layer_tokens=tokens,
            particle_tokens=tokens[:, None],
            particle_mask=mask[:, 0],
            logits=torch.zeros(features.shape[0], 3, device=features.device),
        )


def test_staged_provider_reorders_parent_rows_and_injects_offline_logits():
    tokens = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones(2, 2, dtype=torch.bool)
    reservation = build_tap_stage_reservation(
        source_role="offline_teacher",
        source_manifest_sha256=_sha("source"),
        logical_split_sha256=_sha("split"),
        ordered_identity_sha256=_sha("identity"),
        teacher_checkpoint_sha256=_sha("teacher"),
        tap_spec_sha256=_sha("tap"),
        jets=2,
        max_particles=2,
        token_width=3,
        identity_columns=2,
    )
    staged = stage_teacher_tap_float16(
        tokens,
        mask,
        torch.tensor([[8, 0], [4, 1]], dtype=torch.int64),
        reservation=reservation,
    )
    memory = StagedContextualMemory(
        staged_tap=staged,
        logits=torch.tensor([[8.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        parent_indices=torch.tensor([8, 4], dtype=torch.int64),
    )
    generator = MatchingFreeParticleViewGenerator(
        ParticleViewGeneratorConfig(
            query_dim=3,
            memory_dim=3,
            width=8,
            num_heads=2,
            num_cross_attention_blocks=1,
            bottleneck_width=2,
        )
    )
    provider = StagedDiscoveryViewProvider(
        generator=generator,
        query_teacher=_FrozenQuery(),
        staged_memory=memory,
        query_tap_choice="penultimate",
    )
    momentum = torch.tensor(
        [
            [[1.0, 0.5], [0.0, 0.0], [0.0, 0.0]],
            [[1.5, 0.7], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    energy = torch.sqrt(momentum.square().sum(dim=1, keepdim=True) + 1.0)
    lorentz = torch.cat((momentum, energy), dim=1)
    batch = {
        "points": torch.zeros(2, 2, 2),
        "features": torch.randn(2, 3, 2),
        "lorentz_vectors": lorentz,
        "mask": torch.ones(2, 1, 2, dtype=torch.bool),
        "offline_lorentz_vectors": lorentz.clone(),
        "offline_mask": torch.ones(2, 1, 2, dtype=torch.bool),
        "parent_indices": torch.tensor([4, 8]),
    }
    provided = provider(batch)
    assert provided["view"].shape == (2, 2, 2)
    assert provided["raw_centered_view"].shape == (2, 2, 2)
    assert batch["offline_logits"][:, 0].tolist() == [4.0, 8.0]
    provided["view"].square().sum().backward()
    assert generator.memory_projection.weight.grad is not None


def test_candidate_view_set_normalizes_train_only_and_reorders_by_parent():
    view_set = CandidateViewSet(
        views=torch.tensor(
            [
                [[1.0, -1.0], [3.0, 1.0], [0.0, 0.0]],
                [[5.0, 3.0], [7.0, 5.0], [9.0, 7.0]],
            ],
            dtype=torch.float32,
        ),
        mask=torch.tensor(
            [[True, True, False], [True, True, True]],
            dtype=torch.bool,
        ),
        parent_indices=torch.tensor([8, 4], dtype=torch.int64),
        logical_split_sha256=_sha("train-split"),
        ordered_identity_sha256=_sha("train-order"),
    )
    normalizer = fit_particle_view_normalizer(
        view_set.views,
        view_set.mask,
        train_split_sha256=_sha("train-split"),
        generator_checkpoint_sha256=_sha("generator"),
    )
    normalized = view_set.normalized(normalizer)
    assert normalized.views[normalized.mask].mean(dim=0).tolist() == pytest.approx(
        [0.0, 0.0], abs=1.0e-6
    )
    assert torch.count_nonzero(normalized.views[~normalized.mask]) == 0
    provider = IndexedCandidateViewProvider(normalized)
    provided = provider(
        {
            "features": torch.zeros(2, 17, 3),
            "mask": torch.tensor(
                [
                    [[True, True, True]],
                    [[True, True, False]],
                ]
            ),
            "parent_indices": torch.tensor([4, 8]),
        }
    )
    assert torch.equal(provided["view"][0], normalized.views[1])
    assert torch.equal(provided["view"][1], normalized.views[0])
    assert provided["raw_centered_view"] is provided["view"]


def test_recovery_prediction_loader_is_label_free_and_preserves_order(
    monkeypatch,
):
    import teacher_logit_reco.local_particle_residual_field.particle_view.target_runtime as runtime

    truth = CandidateViewSet(
        views=torch.randn(2, 3, 1),
        mask=torch.ones(2, 3, dtype=torch.bool),
        parent_indices=torch.tensor([5, 9], dtype=torch.int64),
        logical_split_sha256=_sha("select-split"),
        ordered_identity_sha256=_sha("select-order"),
    )
    batch = {
        "features": torch.randn(2, 17, 3),
        "mask": truth.mask.clone(),
        "true_view": truth.views.clone(),
    }
    observed = {}

    def loader(*args, **kwargs):
        observed["mode"] = kwargs["mode"]
        observed["true_views"] = kwargs["true_views"]
        return [batch]

    monkeypatch.setattr(runtime, "make_logical_data_loader", loader)
    config = RecoveryProbeConfig(view_dim=1)
    model = FixedCapacityRecoveryProbe(config).eval()
    aligned = SimpleNamespace(
        logical_split_sha256=truth.logical_split_sha256,
    )
    prediction = predict_recovery_probe_views(
        model=model,
        aligned=aligned,
        true_views=truth,
        device="cpu",
        num_workers=0,
    )
    assert observed["mode"] == "recovery_probe"
    assert "labels" not in batch
    assert prediction.parent_indices.tolist() == [5, 9]
    assert torch.equal(prediction.mask, truth.mask)

    batch["labels"] = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="unexpected fields"):
        predict_recovery_probe_views(
            model=model,
            aligned=aligned,
            true_views=truth,
            device="cpu",
            num_workers=0,
        )

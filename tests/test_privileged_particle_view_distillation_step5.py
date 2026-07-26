from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from teacher_logit_reco.local_particle_residual_field.particle_view.predictor import (
    NONSELECTABLE_PREDICTOR_ARCHITECTURES,
    PARTICLE_VIEW_FLOP_COUNTER,
    PARTICLE_VIEW_PREDICTOR_ARCHITECTURES,
    PVA3_CANONICAL_ARCHITECTURE,
    HierarchicalParticleViewPredictor,
    RegionLevel,
    build_canonical_particle_view_predictor,
    build_particle_pair_features,
    build_predictor_architecture_config,
    build_predictor_architecture_screen,
    build_soft_hierarchy_relations,
    cartesian_four_vector_kinematics,
    count_unique_parameters,
    flop_fixture_sha256,
    predictor_semantic_flops,
    profile_predictor_resources,
    validate_predictor_resource_profile,
)


def _batch(batch: int = 2, particles: int = 9):
    generator = torch.Generator().manual_seed(5105)
    features = torch.randn(
        (batch, 17, particles), generator=generator
    )
    momentum = torch.randn(
        (batch, 3, particles), generator=generator
    )
    energy = torch.sqrt(momentum.square().sum(dim=1, keepdim=True) + 1.0)
    vectors = torch.cat((momentum, energy), dim=1)
    mask = torch.ones((batch, 1, particles), dtype=torch.bool)
    mask[0, 0, -2:] = False
    features[0, :, -2:] = 123.0
    vectors[0, :, -2:] = -456.0
    return features, vectors, mask


def _manual_level(
    embeddings,
    vectors,
    valid,
    assignment,
):
    return RegionLevel(
        embeddings=embeddings,
        four_vectors=vectors,
        weight=valid.float(),
        occupancy=valid.float(),
        valid=valid,
        assignment=assignment,
        provisional_assignment=assignment,
    )


def test_step5_architecture_screen_is_complete_and_locked():
    screen = build_predictor_architecture_screen(view_dim=4)
    assert tuple(screen) == PARTICLE_VIEW_PREDICTOR_ARCHITECTURES
    canonical = screen[PVA3_CANONICAL_ARCHITECTURE]
    assert canonical.width == 192
    assert canonical.local_blocks == 3
    assert canonical.global_region_blocks == 4
    assert canonical.decoder_blocks == 2
    assert canonical.final_refinement_blocks == 1
    assert canonical.hierarchy_sizes == (16, 8, 4)
    assert canonical.dropout == 0.05
    assert canonical.balance_weight == 0.01
    assert canonical.to_payload()["pooling"] == {
        "sequential": True,
        "embedding_weight": "normalized_particle_pt",
        "physical_four_vector_weight": "assignment_once",
        "particle_to_fine_centroid_refinement_passes": 1,
        "higher_transition_geometry": (
            "single_embedding_proposal_then_registered_pair_assignment"
        ),
        "iterative_refinement": False,
        "invented_initial_coordinates": False,
    }
    assert screen["P_NO_BALANCE"].balance_weight == 0
    assert screen["P_NO_PAIR_BIAS"].use_pair_bias is False
    assert screen["P_NO_REFINEMENT"].final_refinement_blocks == 0
    assert screen["P_REGIONS_8_4"].hierarchy_sizes == (8, 4)
    assert screen["P_REGIONS_16_8_4_2"].hierarchy_sizes == (16, 8, 4, 2)
    assert screen["P_WIDTH128"].width == 128
    assert screen["P_WIDTH256"].width == 256
    assert {
        key for key, config in screen.items() if not config.selectable
    } == NONSELECTABLE_PREDICTOR_ARCHITECTURES
    assert len({config.content_hash for config in screen.values()}) == len(screen)
    assert (
        screen["P_HIER_DECODER_REFINE"].structural_hash
        == screen["P_REGIONS_16_8_4"].structural_hash
        == screen["P_WIDTH192"].structural_hash
    )


def test_step5_cartesian_geometry_and_wrapped_pair_features():
    vectors = torch.tensor(
        [[[3.0, 4.0, 0.0, 6.0], [-1.0, 0.0, 0.0, 2.0]]]
    )
    kinematics = cartesian_four_vector_kinematics(vectors)
    assert torch.allclose(kinematics[0, 0, 0], torch.tensor(5.0))
    assert torch.allclose(kinematics[0, 0, 1], torch.tensor(0.0))
    assert torch.allclose(
        kinematics[0, 0, 3], torch.sqrt(torch.tensor(11.0))
    )
    pair = build_particle_pair_features(vectors)
    assert pair.shape == (1, 2, 2, 6)
    assert torch.all(pair[..., 1].abs() <= torch.pi + 1.0e-7)
    assert torch.isfinite(pair).all()


def test_step5_canonical_forward_shapes_masks_bounds_and_centering():
    torch.manual_seed(5106)
    model = build_canonical_particle_view_predictor(view_dim=4).eval()
    features, vectors, mask = _batch()
    with torch.no_grad():
        output = model(features, vectors, mask)
    valid = mask[:, 0]
    assert output.mean.shape == (2, 9, 4)
    assert output.log_variance.shape == (2, 9, 4)
    assert output.trust.shape == (2, 9, 1)
    assert torch.equal(output.mean[~valid], torch.zeros_like(output.mean[~valid]))
    assert torch.equal(
        output.log_variance[~valid],
        torch.zeros_like(output.log_variance[~valid]),
    )
    assert torch.equal(
        output.trust[~valid], torch.zeros_like(output.trust[~valid])
    )
    assert output.log_variance[valid].min() >= -6.0
    assert output.log_variance[valid].max() <= 3.0
    assert output.trust[valid].min() >= 0
    assert output.trust[valid].max() <= 1
    masked_sum = (output.mean * valid[:, :, None]).sum(dim=1)
    assert torch.allclose(masked_sum, torch.zeros_like(masked_sum), atol=2e-5)
    assert output.hierarchy is not None
    assert [level.embeddings.shape[1] for level in output.hierarchy.levels] == [
        16,
        8,
        4,
    ]
    assert output.hierarchy.balance_terms.shape == (2, 3)
    assert output.hierarchy.assignment_entropy.shape == (2, 3)
    assert output.hierarchy.maximum_slot_mass.shape == (2, 3)
    assert output.hierarchy.empty_rate.shape == (2, 3)
    assert output.balance_loss.item() >= 0


def test_step5_pooling_is_sequential_conservative_and_not_double_weighted():
    torch.manual_seed(5107)
    model = build_canonical_particle_view_predictor(view_dim=2).eval()
    features, vectors, mask = _batch(batch=1, particles=7)
    with torch.no_grad():
        output = model(features, vectors, mask)
    hierarchy = output.hierarchy
    assert hierarchy is not None
    valid = mask[:, 0]
    expected_p4 = torch.where(
        valid[:, None, :], vectors, torch.zeros_like(vectors)
    ).sum(dim=2)
    expected_count = valid.sum(dim=1).float()
    source_valid = valid
    for level in hierarchy.levels:
        row_sum = level.assignment.sum(dim=-1)
        assert torch.allclose(
            row_sum[source_valid],
            torch.ones_like(row_sum[source_valid]),
            atol=1e-6,
        )
        assert torch.equal(
            row_sum[~source_valid], torch.zeros_like(row_sum[~source_valid])
        )
        # A second pT weight on physical vectors would violate this equality.
        assert torch.allclose(
            level.four_vectors.sum(dim=1), expected_p4, atol=2e-4, rtol=2e-4
        )
        assert torch.allclose(
            level.weight.sum(dim=1), torch.ones(1), atol=2e-5
        )
        assert torch.allclose(
            level.occupancy.sum(dim=1), expected_count, atol=2e-4
        )
        source_valid = level.valid
    assert hierarchy.levels[1].assignment.shape == (1, 16, 8)
    assert hierarchy.levels[2].assignment.shape == (1, 8, 4)


def test_step5_soft_relations_are_exact_products_and_empty_rows_are_zero():
    generator = torch.Generator().manual_seed(5108)
    batch, width = 1, 3
    fine_valid = torch.tensor([[True, True, False]])
    mid_valid = torch.tensor([[True, True]])
    coarse_valid = torch.tensor([[True]])
    a = torch.ones((batch, 5, 3)) / 3
    b = torch.tensor([[[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]])
    c = torch.tensor([[[0.25], [0.75]]])
    fine = _manual_level(
        torch.randn((batch, 3, width), generator=generator),
        torch.randn((batch, 3, 4), generator=generator),
        fine_valid,
        a,
    )
    middle = _manual_level(
        torch.randn((batch, 2, width), generator=generator),
        torch.randn((batch, 2, 4), generator=generator),
        mid_valid,
        b,
    )
    coarse = _manual_level(
        torch.randn((batch, 1, width), generator=generator),
        torch.randn((batch, 1, 4), generator=generator),
        coarse_valid,
        c,
    )
    relation, complement, level_ids = build_soft_hierarchy_relations(
        (fine, middle, coarse)
    )
    assert relation.shape == (1, 6, 6)
    expected_b = b.clone()
    expected_b[:, 2] = 0
    expected_bc = (b @ c).clone()
    expected_bc[:, 2] = 0
    assert torch.allclose(relation[:, :3, 3:5], expected_b)
    assert torch.allclose(relation[:, 3:5, 5:6], c)
    assert torch.allclose(relation[:, :3, 5:6], expected_bc)
    assert torch.allclose(
        relation[:, 3:5, :3], expected_b.transpose(1, 2)
    )
    assert torch.equal(relation[:, 2], torch.zeros_like(relation[:, 2]))
    assert torch.equal(relation[:, :, 2], torch.zeros_like(relation[:, :, 2]))
    assert torch.equal(complement[:, 2], torch.zeros_like(complement[:, 2]))
    assert level_ids.tolist() == [0, 0, 0, 1, 1, 2]


def test_step5_predictor_is_particle_permutation_equivariant():
    torch.manual_seed(5109)
    model = build_canonical_particle_view_predictor(view_dim=2).eval()
    features, vectors, mask = _batch(batch=1, particles=8)
    permutation = torch.tensor([5, 0, 7, 2, 3, 1, 6, 4])
    with torch.no_grad():
        original = model(features, vectors, mask)
        permuted = model(
            features[:, :, permutation],
            vectors[:, :, permutation],
            mask[:, :, permutation],
        )
    assert torch.allclose(
        permuted.mean,
        original.mean[:, permutation],
        atol=2e-5,
        rtol=2e-5,
    )
    for original_level, permuted_level in zip(
        original.hierarchy.levels, permuted.hierarchy.levels
    ):
        assert torch.allclose(
            original_level.four_vectors,
            permuted_level.four_vectors,
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            original_level.embeddings,
            permuted_level.embeddings,
            atol=2e-5,
            rtol=2e-5,
        )


def test_step5_gradients_reach_hlt_features_and_geometry_only():
    torch.manual_seed(5110)
    config = build_predictor_architecture_config(
        "P_REGIONS_8_4", view_dim=2
    )
    model = HierarchicalParticleViewPredictor(config)
    features, vectors, mask = _batch(batch=1, particles=6)
    features.requires_grad_()
    vectors.requires_grad_()
    output = model(features, vectors, mask)
    loss = (
        output.mean.square().mean()
        + output.log_variance.square().mean()
        + output.balance_loss
    )
    loss.backward()
    assert features.grad is not None
    assert vectors.grad is not None
    assert features.grad.abs().sum() > 0
    assert vectors.grad.abs().sum() > 0
    assert set(inspect.signature(model.forward).parameters) == {
        "features",
        "lorentz_vectors",
        "mask",
    }


def test_step5_no_balance_control_and_fail_closed_inputs():
    config = build_predictor_architecture_config("P_NO_BALANCE", view_dim=1)
    model = HierarchicalParticleViewPredictor(config).eval()
    features, vectors, mask = _batch(batch=1, particles=5)
    with torch.no_grad():
        output = model(features, vectors, mask)
    assert output.balance_loss.item() == 0
    with pytest.raises(ValueError, match="all-padding"):
        model(features, vectors, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="finite"):
        bad = features.clone()
        bad[0, 0, 0] = float("nan")
        model(bad, vectors, mask)


def test_step5_resource_counter_parameters_and_profile_contract():
    torch.manual_seed(5111)
    config = build_predictor_architecture_config("P_C0_PARTICLE", view_dim=1)
    model = HierarchicalParticleViewPredictor(config)
    assert count_unique_parameters(model) > 0
    assert count_unique_parameters(model, trainable_only=True) == (
        count_unique_parameters(model)
    )
    flops = predictor_semantic_flops(model, particles=8)
    assert flops["counter"] == PARTICLE_VIEW_FLOP_COUNTER
    assert flops["fixture_sha256"] == flop_fixture_sha256(particles=8)
    assert flops["exact_integer_total"] == sum(flops["per_operator"].values())
    assert flops["exact_integer_total"] > 0
    profile = profile_predictor_resources(
        model, particles=128, warmup=0, repetitions=1
    )
    validate_predictor_resource_profile(
        profile, expected_config_sha256=config.content_hash
    )
    stale = dict(profile)
    stale["total_parameters"] += 1
    with pytest.raises(ValueError, match="content hash"):
        validate_predictor_resource_profile(stale)


def test_step5_shared_stem_is_explicit_and_nonselectable():
    config = build_predictor_architecture_config(
        "P_SHARED_CONSUMER_STEM", view_dim=2
    )
    with pytest.raises(ValueError, match="requires"):
        HierarchicalParticleViewPredictor(config)
    shared = nn.Linear(17, 192)
    model = HierarchicalParticleViewPredictor(
        config, shared_particle_embedding=shared
    )
    assert model.input_embedding is shared
    assert model.config.selectable is False


def test_step5_every_declared_architecture_control_executes():
    torch.manual_seed(5112)
    features, vectors, mask = _batch(batch=1, particles=4)
    for architecture_id in PARTICLE_VIEW_PREDICTOR_ARCHITECTURES:
        config = build_predictor_architecture_config(
            architecture_id, view_dim=1
        )
        shared = (
            nn.Sequential(
                nn.Linear(17, config.width),
                nn.LayerNorm(config.width),
                nn.GELU(),
            )
            if config.shared_consumer_stem
            else None
        )
        model = HierarchicalParticleViewPredictor(
            config, shared_particle_embedding=shared
        ).eval()
        with torch.no_grad():
            output = model(features, vectors, mask)
        assert output.mean.shape == (1, 4, 1), architecture_id
        assert torch.isfinite(output.mean).all(), architecture_id
        assert torch.isfinite(output.log_variance).all(), architecture_id

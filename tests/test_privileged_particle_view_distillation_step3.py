from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import teacher_logit_reco.local_particle_residual_field.particle_view.tap_staging as tap_staging_module
from teacher_logit_reco.local_particle_residual_field.particle_view import (
    ContextualHLTQueryTap,
    ContextualQueryTapConfig,
    MatchingFreeParticleViewGenerator,
    PARTICLE_VIEW_COORDINATE_PARENT_HASH_FIELDS,
    PARTICLE_VIEW_PAIR_FEATURE_ORDER,
    ParticleViewGeneratorConfig,
    StreamingTeacherTapBuilder,
    audit_live_tap_equivalence,
    build_hlt_memory_pair_features,
    build_mandatory_hlt_memory_control_configs,
    build_tap_stage_reservation,
    build_target_candidate_registration,
    load_hashed_json,
    particle_view_rate_covariance_losses,
    stage_teacher_tap_float16,
    validate_staged_teacher_tap,
    validate_target_candidate_registration,
    with_content_hash,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rehash(payload: dict) -> dict:
    unhashed = deepcopy(payload)
    unhashed.pop("content_hash", None)
    return with_content_hash(unhashed)


def _geometry(batch: int, particles: int, *, offset: float = 0.0) -> torch.Tensor:
    torch.manual_seed(7 + particles)
    pt = torch.rand(batch, particles) * 10.0 + 0.1
    eta = torch.randn(batch, particles) * 0.4 + offset
    phi = (torch.rand(batch, particles) * 2.0 - 1.0) * math.pi
    mass = torch.rand(batch, particles)
    return torch.stack((pt, eta, phi, mass), dim=-1)


def _generator_config(
    *,
    bottleneck_width: int = 4,
    memory_source: str = "offline",
    self_mask: bool = False,
    coordinate_dropout_probability: float = 0.0,
    coordinate_noise_sigma: float = 0.0,
) -> ParticleViewGeneratorConfig:
    return ParticleViewGeneratorConfig(
        query_dim=12,
        memory_dim=10 if memory_source == "offline" else 12,
        width=32,
        num_heads=4,
        num_cross_attention_blocks=2,
        feed_forward_expansion=2,
        bottleneck_width=bottleneck_width,
        memory_source=memory_source,
        self_mask_same_particle=self_mask,
        coordinate_dropout_probability=coordinate_dropout_probability,
        coordinate_noise_sigma=coordinate_noise_sigma,
    )


def _offline_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(23)
    query = torch.randn(2, 6, 12)
    memory = torch.randn(2, 7, 10)
    query_mask = torch.tensor(
        [[True, True, True, True, False, False], [True, True, True, False, False, False]]
    )
    memory_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, True, False, False, False],
        ]
    )
    return {
        "query_tokens": query,
        "memory_tokens": memory,
        "query_geometry": _geometry(2, 6),
        "memory_geometry": _geometry(2, 7, offset=0.2),
        "query_mask": query_mask,
        "memory_mask": memory_mask,
    }


def test_canonical_generator_profile_and_all_bottleneck_widths() -> None:
    canonical = ParticleViewGeneratorConfig(query_dim=160, memory_dim=160)
    assert canonical.width == 160
    assert canonical.num_heads == 8
    assert canonical.num_cross_attention_blocks == 2
    assert canonical.feed_forward_expansion == 4
    assert canonical.bottleneck_width == 4
    assert canonical.use_null_token is True
    assert canonical.use_pair_bias is True
    assert canonical.center_output is True
    assert canonical.coordinate_dropout_probability == pytest.approx(0.10)
    assert canonical.coordinate_noise_sigma == pytest.approx(0.05)
    assert canonical.to_payload()["matching_policy"].startswith("none_")
    assert tuple(canonical.to_payload()["pair_feature_order"]) == (
        PARTICLE_VIEW_PAIR_FEATURE_ORDER
    )

    inputs = _offline_inputs()
    for width in (1, 2, 4, 8):
        model = MatchingFreeParticleViewGenerator(
            _generator_config(bottleneck_width=width)
        ).eval()
        output = model(**inputs)
        assert output.view.shape == (2, 6, width)
        assert torch.count_nonzero(output.view[~inputs["query_mask"]]) == 0


def test_pair_geometry_wraps_phi_and_zeros_padding() -> None:
    query_geometry = torch.tensor([[[2.0, 0.5, math.pi - 0.1, 0.0]]])
    memory_geometry = torch.tensor(
        [[[1.0, -0.5, -math.pi + 0.1, 0.0], [1.0, 10.0, 0.0, 0.0]]]
    )
    query_mask = torch.tensor([[True]])
    memory_mask = torch.tensor([[True, False]])
    pair = build_hlt_memory_pair_features(
        query_geometry,
        memory_geometry,
        query_mask=query_mask,
        memory_mask=memory_mask,
    )
    assert pair.shape == (1, 1, 2, 4)
    assert pair[0, 0, 0, 0] == pytest.approx(1.0)
    assert pair[0, 0, 0, 1] == pytest.approx(-0.2, abs=1.0e-6)
    assert pair[0, 0, 0, 3] == pytest.approx(math.log(2.0), abs=1.0e-6)
    assert torch.count_nonzero(pair[0, 0, 1]) == 0


def test_hlt_permutation_equivariance_and_offline_permutation_invariance() -> None:
    inputs = _offline_inputs()
    torch.manual_seed(31)
    model = MatchingFreeParticleViewGenerator(_generator_config()).eval()
    reference = model(**inputs).view
    repeated = model(**inputs).view
    assert torch.equal(repeated, reference)

    query_permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    permuted_query_inputs = dict(inputs)
    for name in ("query_tokens", "query_geometry", "query_mask"):
        permuted_query_inputs[name] = inputs[name][:, query_permutation]
    permuted_query = model(**permuted_query_inputs).view
    assert torch.allclose(
        permuted_query,
        reference[:, query_permutation],
        atol=2.0e-6,
        rtol=2.0e-6,
    )

    memory_permutation = torch.tensor([6, 2, 0, 5, 1, 4, 3])
    permuted_memory_inputs = dict(inputs)
    for name in ("memory_tokens", "memory_geometry", "memory_mask"):
        permuted_memory_inputs[name] = inputs[name][:, memory_permutation]
    permuted_memory = model(**permuted_memory_inputs).view
    assert torch.allclose(permuted_memory, reference, atol=2.0e-6, rtol=2.0e-6)


def test_padding_cannot_change_valid_views() -> None:
    torch.manual_seed(41)
    model = MatchingFreeParticleViewGenerator(_generator_config()).eval()
    query = torch.randn(1, 3, 12)
    memory = torch.randn(1, 4, 10)
    query_geometry = _geometry(1, 3)
    memory_geometry = _geometry(1, 4)
    short = model(
        query,
        memory,
        query_geometry=query_geometry,
        memory_geometry=memory_geometry,
        query_mask=torch.ones(1, 3, dtype=torch.bool),
        memory_mask=torch.ones(1, 4, dtype=torch.bool),
    ).view

    padded_query = torch.cat((query, torch.full((1, 2, 12), float("nan"))), dim=1)
    padded_memory = torch.cat((memory, torch.full((1, 3, 10), float("nan"))), dim=1)
    padded_query_geometry = torch.cat(
        (query_geometry, torch.full((1, 2, 4), float("nan"))), dim=1
    )
    padded_memory_geometry = torch.cat(
        (memory_geometry, torch.full((1, 3, 4), float("nan"))), dim=1
    )
    padded = model(
        padded_query,
        padded_memory,
        query_geometry=padded_query_geometry,
        memory_geometry=padded_memory_geometry,
        query_mask=torch.tensor([[True, True, True, False, False]]),
        memory_mask=torch.tensor([[True, True, True, True, False, False, False]]),
    ).view
    assert torch.allclose(padded[:, :3], short, atol=2.0e-6, rtol=2.0e-6)
    assert torch.count_nonzero(padded[:, 3:]) == 0


def test_centering_bounds_null_behavior_and_training_augmentation_order() -> None:
    inputs = _offline_inputs()
    config = _generator_config(
        coordinate_dropout_probability=0.5,
        coordinate_noise_sigma=0.0,
    )
    torch.manual_seed(51)
    model = MatchingFreeParticleViewGenerator(config).eval()
    deterministic = model(**inputs).view
    assert float(deterministic.detach().abs().max()) <= 2.0
    for batch_index in range(deterministic.shape[0]):
        valid = inputs["query_mask"][batch_index]
        assert torch.allclose(
            deterministic[batch_index, valid].mean(dim=0),
            torch.zeros(config.bottleneck_width),
            atol=1.0e-6,
        )

    generator = torch.Generator().manual_seed(101)
    augmented = model(
        **inputs,
        apply_output_augmentation=True,
        augmentation_generator=generator,
    ).view
    for batch_index in range(augmented.shape[0]):
        valid = inputs["query_mask"][batch_index]
        for dimension in range(config.bottleneck_width):
            actual = augmented[batch_index, valid, dimension]
            expected = deterministic[batch_index, valid, dimension]
            assert torch.allclose(actual, expected, atol=1.0e-6) or torch.count_nonzero(
                actual
            ) == 0
        assert torch.allclose(
            augmented[batch_index, valid].mean(dim=0),
            torch.zeros(config.bottleneck_width),
            atol=1.0e-6,
        )

    null_inputs = dict(inputs)
    null_inputs["memory_mask"] = torch.zeros_like(inputs["memory_mask"])
    null_only = model(**null_inputs)
    assert torch.isfinite(null_only.view).all()
    assert null_only.null_attention_fraction.detach().item() == pytest.approx(
        1.0, abs=1.0e-6
    )


def test_hlt_memory_self_inclusive_and_exact_self_mask_controls() -> None:
    locked_controls = build_mandatory_hlt_memory_control_configs(
        token_dim=12,
        width=32,
        num_heads=4,
        feed_forward_expansion=2,
    )
    assert set(locked_controls) == {
        "VGEN_MEMORY_HLT",
        "VGEN_MEMORY_HLT_SELFMASK",
    }
    assert locked_controls["VGEN_MEMORY_HLT"].self_mask_same_particle is False
    assert (
        locked_controls["VGEN_MEMORY_HLT_SELFMASK"].self_mask_same_particle
        is True
    )
    inclusive_payload = locked_controls["VGEN_MEMORY_HLT"].to_payload()
    masked_payload = locked_controls["VGEN_MEMORY_HLT_SELFMASK"].to_payload()
    assert {
        key: value
        for key, value in inclusive_payload.items()
        if key != "self_mask_same_particle"
    } == {
        key: value
        for key, value in masked_payload.items()
        if key != "self_mask_same_particle"
    }

    torch.manual_seed(61)
    tokens = torch.randn(2, 5, 12)
    geometry = _geometry(2, 5)
    mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, False, False]]
    )
    common = {
        "query_tokens": tokens,
        "memory_tokens": tokens,
        "query_geometry": geometry,
        "memory_geometry": geometry,
        "query_mask": mask,
        "memory_mask": mask,
    }
    inclusive = MatchingFreeParticleViewGenerator(
        _generator_config(memory_source="hlt")
    ).eval()(**common)
    self_masked = MatchingFreeParticleViewGenerator(
        _generator_config(memory_source="hlt", self_mask=True)
    ).eval()(**common)
    diagonal = torch.arange(tokens.shape[1])
    inclusive_diagonal = inclusive.attention[:, :, diagonal, diagonal]
    masked_diagonal = self_masked.attention[:, :, diagonal, diagonal]
    valid_diagonal = mask[:, None, :].expand_as(masked_diagonal)
    assert torch.all(inclusive_diagonal[valid_diagonal] > 0)
    assert torch.count_nonzero(masked_diagonal[valid_diagonal]) == 0
    assert torch.isfinite(self_masked.view).all()
    assert self_masked.attention.shape[-1] == tokens.shape[1] + 1

    wrong_memory = dict(common)
    wrong_memory["memory_tokens"] = tokens.clone()
    wrong_memory["memory_tokens"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="exact frozen query tokens"):
        MatchingFreeParticleViewGenerator(
            _generator_config(memory_source="hlt")
        ).eval()(**wrong_memory)


def test_contextual_query_tap_detaches_frozen_a0_and_mixes_deterministically() -> None:
    config = ContextualQueryTapConfig(
        source="mix3",
        available_layers=("final_minus2", "final_minus1", "final"),
        checkpoint_sha256=_sha("a0"),
        input_normalization_sha256=_sha("hlt-normalization"),
    )
    tap = ContextualHLTQueryTap(config)
    torch.manual_seed(71)
    tensors = {
        name: torch.randn(2, 4, 6, requires_grad=True)
        for name in config.available_layers
    }
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    output = tap(tensors, mask)
    assert torch.allclose(
        output[mask],
        torch.stack([tensor.detach() for tensor in tensors.values()]).mean(dim=0)[
            mask
        ],
        atol=1.0e-6,
    )
    output.square().sum().backward()
    assert tap.mixture_logits.grad is not None
    assert all(tensor.grad is None for tensor in tensors.values())
    assert torch.count_nonzero(output[~mask]) == 0
    first = tap.provenance_payload()
    second = tap.provenance_payload()
    assert first == second
    assert first["mixture_weights"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_rate_variance_and_covariance_losses_match_hand_calculations() -> None:
    mask = torch.tensor([[True, True]])
    view = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    losses = particle_view_rate_covariance_losses(view, mask)
    assert losses["variances"].tolist() == pytest.approx([1.0, 0.0])
    assert losses["variance_floor_loss"] == pytest.approx(0.01)
    assert losses["rate_loss"] == pytest.approx(0.0)
    assert losses["covariance_loss"] == pytest.approx(0.0)

    correlated = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]])
    correlated_losses = particle_view_rate_covariance_losses(correlated, mask)
    assert correlated_losses["rate_loss"] == pytest.approx(1.0)
    assert correlated_losses["covariance_loss"] == pytest.approx(1.0)

    one_dimensional = particle_view_rate_covariance_losses(view[..., :1], mask)
    assert one_dimensional["covariance_loss"].item() == 0.0


def test_generator_paths_receive_gradients_without_teacher_updates() -> None:
    inputs = _offline_inputs()
    inputs["query_tokens"].requires_grad_(False)
    inputs["memory_tokens"].requires_grad_(False)
    torch.manual_seed(79)
    model = MatchingFreeParticleViewGenerator(_generator_config()).train()
    output = model(**inputs, apply_output_augmentation=False)
    regularizers = particle_view_rate_covariance_losses(
        output.view, inputs["query_mask"]
    )
    loss = (
        output.view.square().mean()
        + regularizers["variance_floor_loss"]
        + regularizers["rate_loss"]
        + regularizers["covariance_loss"]
    )
    loss.backward()
    assert model.query_projection.weight.grad is not None
    assert torch.count_nonzero(model.query_projection.weight.grad) > 0
    assert model.memory_projection.weight.grad is not None
    assert torch.count_nonzero(model.memory_projection.weight.grad) > 0
    assert model.blocks[0].pair_bias[-1].weight.grad is not None
    assert torch.count_nonzero(model.blocks[0].pair_bias[-1].weight.grad) > 0
    assert model.bottleneck[-1].weight.grad is not None
    assert torch.count_nonzero(model.bottleneck[-1].weight.grad) > 0
    assert inputs["query_tokens"].grad is None
    assert inputs["memory_tokens"].grad is None


def test_ram_stage_is_bit_exact_nearest_even_and_detects_mutation() -> None:
    lower = np.float16(1.0)
    upper = np.nextafter(lower, np.float16(np.inf))
    midpoint = np.float32(
        (np.float32(lower) + np.float32(upper)) * np.float32(0.5)
    )
    tokens = torch.tensor(
        [
            [
                [float(midpoint), 0.33333334, -2.125],
                [float("nan"), float("nan"), float("nan")],
            ],
            [[10.001, -0.00003, 4.5], [0.25, -0.5, 0.75]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, False], [True, True]])
    identities = torch.tensor([[0, 10], [1, 11]], dtype=torch.int64)
    reservation = build_tap_stage_reservation(
        source_role="offline_teacher",
        source_manifest_sha256=_sha("source"),
        logical_split_sha256=_sha("train-split"),
        ordered_identity_sha256=_sha("train-identities"),
        teacher_checkpoint_sha256=_sha("teacher"),
        tap_spec_sha256=_sha("tap"),
        jets=2,
        max_particles=2,
        token_width=3,
        identity_columns=2,
    )
    staged = stage_teacher_tap_float16(
        tokens, mask, identities, reservation=reservation
    )
    assert staged.tokens[0, 0, 0].item() == 1.0
    assert torch.count_nonzero(staged.tokens[0, 1]) == 0
    logical_hash = validate_staged_teacher_tap(staged)
    audit = audit_live_tap_equivalence(staged, tokens, mask, identities)
    assert audit["logical_content_sha256"] == logical_hash
    assert audit["bitwise_equal_float16"] is True
    assert audit["max_half_bracket_ratio"] <= 1.0 + 1.0e-6
    assert staged.manifest["persistent_storage_allowed"] is False

    staged.tokens[0, 0, 1] += torch.tensor(0.25, dtype=torch.float16)
    with pytest.raises(ValueError, match="logical content changed"):
        validate_staged_teacher_tap(staged)


@pytest.mark.parametrize(
    "source_role",
    (
        "offline_teacher",
        "offline_teacher_secondary",
        "hlt_memory_control",
    ),
)
def test_all_declared_teacher_tap_source_roles_are_reservable(source_role):
    reservation = build_tap_stage_reservation(
        source_role=source_role,
        source_manifest_sha256=_sha("source"),
        logical_split_sha256=_sha("split"),
        ordered_identity_sha256=_sha("identities"),
        teacher_checkpoint_sha256=_sha("teacher"),
        tap_spec_sha256=_sha("tap"),
        jets=2,
        max_particles=3,
        token_width=4,
        identity_columns=2,
    )
    assert reservation["source_role"] == source_role


def test_streaming_tap_staging_matches_one_shot_and_validates_without_promotion(
    monkeypatch,
):
    torch.manual_seed(41)
    tokens = torch.randn(5, 4, 6, dtype=torch.float32)
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, False, False, False],
            [True, True, True, False],
        ]
    )
    tokens[~mask] = float("nan")
    identities = torch.stack(
        (torch.arange(5), torch.arange(10, 15)), dim=1
    ).to(torch.int64)
    reservation = build_tap_stage_reservation(
        source_role="offline_teacher_secondary",
        source_manifest_sha256=_sha("source"),
        logical_split_sha256=_sha("split"),
        ordered_identity_sha256=_sha("identities"),
        teacher_checkpoint_sha256=_sha("teacher"),
        tap_spec_sha256=_sha("tap"),
        jets=5,
        max_particles=4,
        token_width=6,
        identity_columns=2,
    )
    one_shot = stage_teacher_tap_float16(
        tokens, mask, identities, reservation=reservation
    )
    builder = StreamingTeacherTapBuilder(reservation=reservation)
    builder.append(tokens[:2], mask[:2], identities[:2])
    builder.append(tokens[2:], mask[2:], identities[2:])
    streamed = builder.finalize()
    assert torch.equal(streamed.tokens, one_shot.tokens)
    assert torch.equal(streamed.mask, one_shot.mask)
    assert torch.equal(streamed.jet_identities, one_shot.jet_identities)
    assert (
        streamed.manifest["logical_content_sha256"]
        == one_shot.manifest["logical_content_sha256"]
    )

    # Validation must operate on the retained half tensor in bounded chunks,
    # never by recreating the full float32 canonicalization input.
    monkeypatch.setattr(
        tap_staging_module,
        "_canonical_stage_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full canonicalization was called")
        ),
    )
    assert (
        validate_staged_teacher_tap(streamed)
        == streamed.manifest["logical_content_sha256"]
    )


def _registration_kwargs() -> dict:
    return {
        "target_id": "VGEN_TAP_PENULT",
        "campaign_id": "particle_view_500k_v1",
        "selection_status": "canonical_selectable",
        "seed": 101,
        "source_manifest_sha256": _sha("source"),
        "unified_split_manifest_sha256": _sha("unified"),
        "train_split_sha256": _sha("train-split"),
        "train_identity_sha256": _sha("train-identities"),
        "query_tap_registration_sha256": _sha("query-tap"),
        "query_checkpoint_sha256": _sha("a0-checkpoint"),
        "memory_tap_registration_sha256": _sha("offline-tap"),
        "memory_checkpoint_sha256": _sha("offline-checkpoint"),
        "staged_tap_source_role": "offline_teacher",
        "staged_tap_reservation_sha256": _sha("reservation"),
        "staged_tap_manifest_sha256": _sha("staged-manifest"),
        "staged_tap_logical_content_sha256": _sha("staged-content"),
        "generator_checkpoint_sha256": _sha("generator"),
        "offline_source_sha256": _sha("offline-source"),
        "privileged_claim_eligible": True,
        "deployment_control_eligible": False,
    }


def test_candidate_target_provenance_is_acyclic_and_control_specific() -> None:
    offline = build_target_candidate_registration(
        generator_config=_generator_config(),
        **_registration_kwargs(),
    )
    audit = validate_target_candidate_registration(offline)
    assert audit["matching_free"] is True
    assert audit["privileged_claim_eligible"] is True
    serialized = repr(offline).lower()
    assert "consumer_checkpoint" not in serialized
    assert "predictor_checkpoint" not in serialized
    assert "target_logits" not in serialized

    control_kwargs = _registration_kwargs()
    control_kwargs.update(
        {
            "target_id": "VGEN_MEMORY_HLT",
            "selection_status": "performance_control",
            "memory_tap_registration_sha256": control_kwargs[
                "query_tap_registration_sha256"
            ],
            "memory_checkpoint_sha256": control_kwargs[
                "query_checkpoint_sha256"
            ],
            "staged_tap_source_role": "hlt_memory_control",
            "offline_source_sha256": None,
            "privileged_claim_eligible": False,
            "deployment_control_eligible": True,
        }
    )
    hlt_control = build_target_candidate_registration(
        generator_config=_generator_config(memory_source="hlt"),
        **control_kwargs,
    )
    control_audit = validate_target_candidate_registration(hlt_control)
    assert control_audit["memory_source"] == "hlt"
    assert control_audit["deployment_control_eligible"] is True
    assert hlt_control["offline_source_sha256"] is None

    self_mask_kwargs = dict(control_kwargs)
    self_mask_kwargs.update(
        {
            "target_id": "VGEN_MEMORY_HLT_SELFMASK",
            "selection_status": "diagnostic_nonselectable",
            "deployment_control_eligible": False,
        }
    )
    self_mask = build_target_candidate_registration(
        generator_config=_generator_config(
            memory_source="hlt", self_mask=True
        ),
        **self_mask_kwargs,
    )
    assert validate_target_candidate_registration(self_mask)["ok"]

    wrong_hlt = dict(control_kwargs)
    wrong_hlt["memory_checkpoint_sha256"] = _sha("different-checkpoint")
    with pytest.raises(ValueError, match="exact A0 checkpoint"):
        build_target_candidate_registration(
            generator_config=_generator_config(memory_source="hlt"),
            **wrong_hlt,
        )

    descendant = deepcopy(offline)
    descendant["consumer_dependencies"] = [_sha("consumer")]
    with pytest.raises(ValueError, match="not canonical"):
        validate_target_candidate_registration(_rehash(descendant))


def test_coordinate_parent_schema_has_generator_but_no_descendant() -> None:
    assert "generator_checkpoint_sha256" in PARTICLE_VIEW_COORDINATE_PARENT_HASH_FIELDS
    assert all(
        "consumer" not in field and "predictor" not in field
        for field in PARTICLE_VIEW_COORDINATE_PARENT_HASH_FIELDS
    )


def test_target_registration_cli_publishes_immutable_artifact(
    tmp_path: Path,
) -> None:
    config = _generator_config()
    config_path = tmp_path / "generator_config.json"
    config_path.write_text(
        json.dumps(config.to_payload(), sort_keys=True), encoding="utf-8"
    )
    output_path = tmp_path / "candidate.json"
    kwargs = _registration_kwargs()
    command = [
        sys.executable,
        "scripts/register_particle_view_target_candidate.py",
        "--target-id",
        kwargs["target_id"],
        "--campaign-id",
        kwargs["campaign_id"],
        "--selection-status",
        kwargs["selection_status"],
        "--seed",
        str(kwargs["seed"]),
        "--generator-config",
        str(config_path),
    ]
    for key in (
        "source_manifest_sha256",
        "unified_split_manifest_sha256",
        "train_split_sha256",
        "train_identity_sha256",
        "query_tap_registration_sha256",
        "query_checkpoint_sha256",
        "memory_tap_registration_sha256",
        "memory_checkpoint_sha256",
        "staged_tap_source_role",
        "staged_tap_reservation_sha256",
        "staged_tap_manifest_sha256",
        "staged_tap_logical_content_sha256",
        "generator_checkpoint_sha256",
        "offline_source_sha256",
    ):
        command.extend(("--" + key.replace("_", "-"), kwargs[key]))
    command.extend(
        (
            "--privileged-claim-eligible",
            "--output",
            str(output_path),
        )
    )
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout)["status"] == "published"
    registration = load_hashed_json(output_path)
    assert validate_target_candidate_registration(registration)["ok"]
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(second.stdout)["status"] == "already_present"

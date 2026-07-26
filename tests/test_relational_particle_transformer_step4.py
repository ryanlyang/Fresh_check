from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.relational_part import (
    DENSITY_ENCODED_DIMENSION,
    DENSITY_NODE_FEATURE_NAMES,
    TRACK_ENCODED_DIMENSION,
    TRACK_VALIDITY_STATE_NAMES,
    DensityEncoder,
    RelationalPairBuilder,
    TrackEncoder,
    build_density_node_features,
    build_density_relation_contract,
    build_global_determinism_contract,
    build_normalization_contract,
    build_raw_input_schema_contract,
    build_relation_family_registry,
    build_registered_step4_model,
    build_screening_registry,
    build_track_compatibility,
    build_track_node_features,
    build_track_relation_contract,
    build_step4_model_contract,
    canonical_json_bytes,
    fit_relation_normalization,
    validate_content_hash,
    validate_relation_normalization_artifact,
)


class _MiniPairEmbed(torch.nn.Module):
    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.pairwise_lv_dim = 0
        self.pairwise_input_dim = input_dimension
        self.out_dim = 8
        self.remove_self_pair = False
        self.fts_embed = torch.nn.Conv1d(input_dimension, 8, 1)


class _MiniParticleTransformer(torch.nn.Module):
    def __init__(self, **config) -> None:
        super().__init__()
        self.use_amp = False
        self.pair_extra_dim = int(config["pair_extra_dim"])
        self.pair_embed = _MiniPairEmbed(self.pair_extra_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 17))
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(
                    17, 1, dropout=0, batch_first=True
                )
                for _ in range(8)
            ]
        )
        self.classifier = torch.nn.Linear(25, int(config["num_classes"]))

    def forward(self, x, v=None, mask=None, uu=None):
        pair_bias = self.pair_embed(v, uu=uu, mask=mask)
        tokens = x.transpose(1, 2)
        for attention in self.blocks:
            update, _ = attention(
                tokens,
                tokens,
                tokens,
                key_padding_mask=~mask[:, 0].bool(),
                need_weights=False,
            )
            tokens = tokens + update
        x = tokens.transpose(1, 2)
        weights = mask.to(x.dtype)
        particle_summary = (x * weights).sum(dim=-1) / weights.sum(
            dim=-1
        ).clamp_min(1)
        pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).to(x.dtype)
        pair_summary = (pair_bias * pair_mask).sum(dim=(-1, -2)) / pair_mask.sum(
            dim=(-1, -2)
        ).clamp_min(1)
        return self.classifier(torch.cat((particle_summary, pair_summary), dim=1))


def _mini_weaver():
    def pairwise(xi, xj, num_outputs=4):
        base = xi[:, :1] + xj[:, :1]
        return torch.cat(tuple(base + index for index in range(num_outputs)), dim=1)

    return SimpleNamespace(
        ParticleTransformer=_MiniParticleTransformer,
        pairwise_lv_fts=pairwise,
    )


def _fit_input() -> tuple[np.ndarray, np.ndarray, list[JetIdentity]]:
    jets, particles = 4, 5
    tokens = np.zeros((jets, particles, 14), dtype=np.float32)
    mask = np.ones((jets, particles), dtype=bool)
    identities = [
        JetIdentity(file=f"fit/class_{row}.root", entry=row, label=row)
        for row in range(jets)
    ]
    categories = (0, 1, 2, 3, 4)
    for row in range(jets):
        tokens[row, :, 0] = np.asarray([10, 6, 4, 2, 1], dtype=np.float32)
        tokens[row, :, 1] = np.asarray([0, .03, .08, .15, .3], dtype=np.float32)
        tokens[row, :, 2] = np.asarray([3.13, -3.13, .2, -.4, 1.2], dtype=np.float32)
        tokens[row, :, 3] = tokens[row, :, 0] * np.cosh(tokens[row, :, 1])
        tokens[row, :, 4] = np.asarray([1, 0, 0, -1, 1], dtype=np.float32)
        for particle, category in enumerate(categories):
            tokens[row, particle, 5 + category] = 1
        tokens[row, :, 10] = np.asarray([.01, 0, 0, -.03, .05]) * (row + 1)
        tokens[row, :, 11] = np.asarray([.001, .002, .003, .004, .005]) * (row + 1)
        tokens[row, :, 12] = np.asarray([.02, 0, 0, -.04, .08]) * (row + 1)
        tokens[row, :, 13] = np.asarray([.006, .007, .008, .009, .010]) * (row + 1)
    return tokens, mask, identities


def _artifact():
    tokens, mask, identities = _fit_input()
    normalizer_contract = build_normalization_contract(
        split_binding_sha256="1" * 64
    )
    relation_registry = build_relation_family_registry()
    raw_schema = build_raw_input_schema_contract()
    artifact = fit_relation_normalization(
        tokens,
        mask,
        identities,
        normalization_contract=normalizer_contract,
        relation_registry=relation_registry,
        raw_input_schema=raw_schema,
        hlt_binding_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        hlt_model_train_content_sha256="4" * 64,
    )
    return tokens, mask, identities, relation_registry, raw_schema, artifact


def _raw_track_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.zeros(1, 4, 14)
    mask = torch.ones(1, 1, 4, dtype=torch.bool)
    raw[0, :, 0] = torch.tensor([10., 5., 3., 2.])
    raw[0, :, 1] = torch.tensor([0., .1, .2, .3])
    raw[0, :, 2] = torch.tensor([np.pi - .01, -np.pi + .01, .2, .4])
    raw[0, :, 3] = raw[0, :, 0] * torch.cosh(raw[0, :, 1])
    raw[0, :, 5] = 1
    raw[0, 1, 5] = 0
    raw[0, 1, 6] = 1  # neutral hadron: never a valid track
    raw[0, :, 10] = torch.tensor([2., 99., .3, -1.])
    raw[0, :, 11] = torch.tensor([.5, .01, 0., .25])
    raw[0, :, 12] = torch.tensor([1., 99., .2, -2.])
    raw[0, :, 13] = torch.tensor([.25, .01, .2, .5])
    return raw, mask


def test_step4_normalizer_fits_global_floors_and_distinct_sample_sets() -> None:
    tokens, mask, identities, registry, raw_schema, artifact = _artifact()
    assert artifact["schema_version"] == 2
    assert artifact["fit_families"] == ["PT", "TRACK", "CHARGE", "DENSITY"]
    assert validate_relation_normalization_artifact(
        artifact,
        relation_registry_sha256=registry["content_hash"],
        raw_input_schema_sha256=raw_schema["content_hash"],
    ) == artifact["content_hash"]
    # Charged hadron/electron/muon are indices 0,3,4 in every jet.
    expected_d0err = tokens[:, (0, 3, 4), 11].astype(np.float64).reshape(-1)
    expected_dzerr = tokens[:, (0, 3, 4), 13].astype(np.float64).reshape(-1)
    assert artifact["track_uncertainty_floors"]["d0"]["floor"] == pytest.approx(
        max(float(np.quantile(expected_d0err, .01, method="linear")), 1e-6)
    )
    assert artifact["track_uncertainty_floors"]["dz"]["floor"] == pytest.approx(
        max(float(np.quantile(expected_dzerr, .01, method="linear")), 1e-6)
    )
    assert artifact["track_uncertainty_floors"]["d0"]["applicable_count"] == 12
    assert artifact["sample_sets"]["TRACK_node"]["applicable_count"] == 12
    assert artifact["sample_sets"]["DENSITY_node"]["applicable_count"] == int(mask.sum())
    assert set(artifact["track_uncertainty_floors"]["d0"]["quantiles"]) == {
        "q01", "q05", "q50", "q95", "q99"
    }

    order = np.asarray([2, 0, 3, 1])
    reordered = fit_relation_normalization(
        tokens[order],
        mask[order],
        [identities[index] for index in order],
        normalization_contract=build_normalization_contract(
            split_binding_sha256="1" * 64
        ),
        relation_registry=registry,
        raw_input_schema=raw_schema,
        hlt_binding_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        hlt_model_train_content_sha256="4" * 64,
    )
    assert reordered == artifact


def test_track_raw_significance_validity_direction_wrapping_and_gradients() -> None:
    raw, mask = _raw_track_fixture()
    nodes = build_track_node_features(
        raw,
        mask,
        d0_uncertainty_floor=.1,
        dz_uncertainty_floor=.1,
    )
    assert nodes["track_valid"].tolist() == [[True, False, False, True]]
    assert nodes["continuous"][0, 0, 0] == 2.0  # raw d0, never tanh(d0)
    expected_sigma = np.sqrt(.5**2 + .1**2)
    assert nodes["sigma_d0_effective"][0, 0] == pytest.approx(expected_sigma)
    assert nodes["continuous"][0, 4, 0] == pytest.approx(
        np.arcsinh(2.0 / expected_sigma)
    )

    pairs = build_track_compatibility(
        raw,
        mask,
        d0_uncertainty_floor=.1,
        dz_uncertainty_floor=.1,
    )
    assert TRACK_VALIDITY_STATE_NAMES == (
        "invalid_invalid", "valid_invalid", "invalid_valid", "valid_valid"
    )
    assert pairs["validity_index"][0, 0, 1] == 1
    assert pairs["validity_index"][0, 1, 0] == 2
    assert pairs["validity_index"][0, 0, 3] == 3
    assert pairs["compatibility"][0, :, 0, 1].count_nonzero() == 0
    assert pairs["compatibility"][0, 8, 0, 3] == pytest.approx(
        -pairs["compatibility"][0, 8, 3, 0].item()
    )
    assert pairs["delta_phi"][0, 0, 1] == pytest.approx(-.02, abs=2e-5)

    sentinel_raw = raw.clone()
    sentinel_raw[0, 0, 10] = -999
    sentinel_nodes = build_track_node_features(
        sentinel_raw,
        mask,
        d0_uncertainty_floor=.1,
        dz_uncertainty_floor=.1,
        sentinel_policy={"d0": -999, "d0err": None, "dz": None, "dzerr": None},
    )
    assert sentinel_nodes["track_valid"][0, 0].item() is False

    *_, artifact = _artifact()
    encoder = TrackEncoder(artifact)
    raw_for_grad = raw.clone().requires_grad_(True)
    encoded = encoder(raw_for_grad, mask)
    assert encoded.shape == (1, TRACK_ENCODED_DIMENSION, 4, 4)
    assert torch.isfinite(encoded).all()
    encoded.square().sum().backward()
    assert encoder.track_encoder[0].weight.grad is not None
    assert encoder.pair_encoder[0].weight.grad is not None
    neutral_changed = raw.detach().clone()
    neutral_changed[0, 1, 10:14] = torch.tensor([-700., .7, 800., .8])
    torch.testing.assert_close(
        encoder(raw.detach(), mask),
        encoder(neutral_changed, mask),
    )


def test_density_exact_annuli_composition_self_exclusion_and_edge_cases() -> None:
    raw = torch.zeros(1, 5, 14)
    mask = torch.ones(1, 1, 5, dtype=torch.bool)
    raw[0, :, 0] = torch.tensor([10., 1., 2., 3., 4.])
    raw[0, :, 1] = torch.tensor([0., .05, .10, .20, .40])
    raw[0, :, 3] = raw[0, :, 0] * torch.cosh(raw[0, :, 1])
    for particle, category in enumerate((0, 1, 2, 3, 4)):
        raw[0, particle, 5 + category] = 1
    raw[0, :, 11] = .01
    raw[0, :, 13] = .02
    raw[0, 3, 10] = .1  # displaced electron in the local R<=.20 cone
    details = build_density_node_features(
        raw,
        mask,
        d0_uncertainty_floor=1e-6,
        dz_uncertainty_floor=1e-6,
    )
    descriptor = details["descriptor"][0, :, 0]
    expected_count = np.log1p(1) / np.log1p(128)
    torch.testing.assert_close(
        descriptor[[0, 2, 4, 6]],
        torch.full((4,), expected_count),
    )
    torch.testing.assert_close(
        descriptor[[1, 3, 5, 7]],
        torch.tensor([1., 2., 3., 4.]) / 20.0,
    )
    assert descriptor[16] == pytest.approx(3 / 6)
    assert descriptor[17] == pytest.approx(1 / 6)
    assert descriptor[18] == pytest.approx(2 / 6)
    assert descriptor[19] == pytest.approx(3 / 6)
    assert descriptor[20] == pytest.approx(1.0)
    assert descriptor[21] == pytest.approx(10 / 16)
    assert all(
        not details["annulus_masks"][0, index, 0, 0]
        for index in range(4)
    )

    single = build_density_node_features(
        raw[:, :1],
        mask[:, :, :1],
        d0_uncertainty_floor=1e-6,
        dz_uncertainty_floor=1e-6,
    )["descriptor"]
    assert single[0, :21].count_nonzero() == 0
    assert single[0, 21, 0] == 1
    zero_pt = raw[:, :1].clone()
    zero_pt[:, :, 0] = 0
    zero = build_density_node_features(
        zero_pt,
        mask[:, :, :1],
        d0_uncertainty_floor=1e-6,
        dz_uncertainty_floor=1e-6,
    )["descriptor"]
    assert zero.count_nonzero() == 0
    nonfinite = raw.clone()
    nonfinite[0, 0, 1] = float("nan")
    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        build_density_node_features(
            nonfinite,
            mask,
            d0_uncertainty_floor=1e-6,
            dz_uncertainty_floor=1e-6,
        )


def test_step4_encoders_and_pair_builder_are_permutation_equivariant() -> None:
    tokens, mask_np, *_rest, artifact = _artifact()
    raw = torch.from_numpy(tokens[:2])
    mask = torch.from_numpy(mask_np[:2]).unsqueeze(1)
    track = TrackEncoder(artifact).eval()
    density = DensityEncoder(artifact).eval()
    permutation = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.argsort(permutation)
    for encoder in (track, density):
        reference = encoder(raw, mask)
        permuted = encoder(raw[:, permutation], mask[:, :, permutation])
        restored = permuted[:, :, inverse][:, :, :, inverse]
        torch.testing.assert_close(reference, restored, atol=2e-6, rtol=2e-6)

    inputs = build_particle_transformer_inputs_from_tokens(
        tokens[:2], mask_np[:2], source_view="fixed_hlt"
    )

    def pairwise(xi, xj, num_outputs=4):
        base = xi[:, :1] + xj[:, :1]
        return torch.cat(tuple(base + index for index in range(num_outputs)), dim=1)

    builder = RelationalPairBuilder(
        ("TRACK", "DENSITY"),
        normalization_artifact=artifact,
        weaver_module=SimpleNamespace(pairwise_lv_fts=pairwise),
    )
    combined = builder(
        torch.from_numpy(inputs.pf_features),
        torch.from_numpy(inputs.pf_vectors),
        torch.from_numpy(inputs.pf_mask),
        raw,
    )
    assert combined.shape == (
        2, 4 + TRACK_ENCODED_DIMENSION + DENSITY_ENCODED_DIMENSION, 5, 5
    )
    assert torch.isfinite(combined).all()
    with pytest.raises(ValueError, match="raw HLT"):
        builder(
            torch.from_numpy(inputs.pf_features),
            torch.from_numpy(inputs.pf_vectors),
            torch.from_numpy(inputs.pf_mask),
        )


def test_registered_step4_singles_train_and_preserve_event_logits_under_permutation() -> None:
    tokens, mask_np, _, registry, _, artifact = _artifact()
    screening = build_screening_registry(
        relation_registry_sha256=registry["content_hash"]
    )
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens[:2], mask_np[:2], source_view="fixed_hlt"
    )
    points = torch.from_numpy(inputs.pf_points)
    features = torch.from_numpy(inputs.pf_features)
    vectors = torch.from_numpy(inputs.pf_vectors)
    mask = torch.from_numpy(inputs.pf_mask)
    raw = torch.from_numpy(tokens[:2])
    permutation = torch.tensor([3, 0, 4, 1, 2])
    for run_id, family in (
        ("RPT_TRACK", "TRACK"),
        ("RPT_DENSITY", "DENSITY"),
    ):
        torch.manual_seed(91)
        model = build_registered_step4_model(
            run_id,
            normalization_artifact=artifact,
            screening_registry=screening,
            weaver_module=_mini_weaver(),
        )
        assert model.families == (family,)
        model.eval()
        reference = model(points, features, vectors, mask, raw)
        permuted = model(
            points[:, :, permutation],
            features[:, :, permutation],
            vectors[:, :, permutation],
            mask[:, :, permutation],
            raw[:, permutation],
        )
        torch.testing.assert_close(reference, permuted, atol=2e-6, rtol=2e-6)
        diagnostics = model.diagnostics(
            features,
            vectors,
            mask,
            raw,
            labels=torch.tensor([1, 8]),
        )
        canonical_json_bytes(diagnostics)
        if family == "TRACK":
            assert (
                "bias_by_minimum_absolute_displacement_significance"
                in diagnostics["TRACK"]
            )
            assert (
                "asinh_absolute_significance_distributions"
                in diagnostics["TRACK"]
            )
            assert (
                diagnostics["TRACK"]["required_class_performance"][
                    "event_counts"
                ]
                == [1, 0, 1, 0]
            )
        else:
            assert (
                "bias_by_context_local_activity_fraction"
                in diagnostics["DENSITY"]
            )
            assert (
                "performance_by_jet_multiplicity"
                in diagnostics["DENSITY"]
            )
            assert set(
                diagnostics["DENSITY"][
                    "annulus_occupancy_count_distributions"
                ]
            ) == {"0", "1", "2", "3"}
        model.train()
        loss = torch.nn.functional.cross_entropy(
            model(points, features, vectors, mask, raw),
            torch.tensor([0, 1]),
        )
        loss.backward()
        parameters = list(model.pair_builder.encoders[family].parameters())
        assert parameters and any(parameter.grad is not None for parameter in parameters)


def test_step4_family_contracts_bind_normalizer_registry_and_raw_schema() -> None:
    _, _, _, registry, raw_schema, artifact = _artifact()
    track_contract = build_track_relation_contract(
        relation_registry_sha256=registry["content_hash"],
        relation_normalization_sha256=artifact["content_hash"],
        raw_input_schema_sha256=raw_schema["content_hash"],
    )
    density_contract = build_density_relation_contract(
        relation_registry_sha256=registry["content_hash"],
        relation_normalization_sha256=artifact["content_hash"],
        track_relation_sha256=track_contract["content_hash"],
    )
    assert track_contract["encoded_dimension"] == 12
    assert track_contract["pair_input_dimension"] == 81
    assert density_contract["node_feature_names"] == list(
        DENSITY_NODE_FEATURE_NAMES
    )
    assert density_contract["annulus_endpoint_rule"] == (
        "[0,0.05],(0.05,0.10],(0.10,0.20],(0.20,0.40]"
    )
    assert validate_content_hash(track_contract) == track_contract["content_hash"]
    assert validate_content_hash(density_contract) == density_contract["content_hash"]
    screening = build_screening_registry(
        relation_registry_sha256=registry["content_hash"]
    )
    assert any(row["run_id"] == "RPT_TRACK" for row in screening["rows"])
    assert any(row["run_id"] == "RPT_DENSITY" for row in screening["rows"])
    determinism = build_global_determinism_contract()
    track_model = build_step4_model_contract(
        "RPT_TRACK",
        normalization_artifact=artifact,
        screening_registry=screening,
        relation_registry_sha256=registry["content_hash"],
        raw_input_schema_sha256=raw_schema["content_hash"],
        pair_base_sha256="5" * 64,
        family_contract=track_contract,
        weaver_runtime_sha256="6" * 64,
        global_determinism_sha256=determinism["content_hash"],
    )
    density_model = build_step4_model_contract(
        "RPT_DENSITY",
        normalization_artifact=artifact,
        screening_registry=screening,
        relation_registry_sha256=registry["content_hash"],
        raw_input_schema_sha256=raw_schema["content_hash"],
        pair_base_sha256="5" * 64,
        family_contract=density_contract,
        weaver_runtime_sha256="6" * 64,
        global_determinism_sha256=determinism["content_hash"],
    )
    assert track_model["pair_path"]["combined_dimension"] == 16
    assert density_model["pair_path"]["combined_dimension"] == 16
    assert validate_content_hash(track_model) == track_model["content_hash"]
    assert validate_content_hash(density_model) == density_model["content_hash"]

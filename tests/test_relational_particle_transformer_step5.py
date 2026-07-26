from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.relational_part import (
    EXCLUSIVE_RESOLUTIONS,
    REGION_RAW_FEATURE_NAMES,
    RegionEncoder,
    RelationalFamilyParticleTransformer,
    RelationalParticleTransformer,
    WideBaseParticleTransformer,
    build_angular_tree_resource_contract,
    build_batched_region_raw_features,
    build_global_determinism_contract,
    build_tree_probe_artifact,
    build_normalization_contract,
    build_raw_input_schema_contract,
    build_reference_tree,
    build_region_raw_features,
    build_relation_family_registry,
    build_screening_registry,
    build_registered_screening_model,
    build_registered_wide_model,
    build_step5_model_contract,
    canonical_json_bytes,
    fit_region_normalization,
    fit_relation_normalization,
    finalize_tree_split,
    full_active_incremental_parameters,
    pair_encoder_flops,
    pair_encoder_parameter_count,
    select_tree_probe,
    select_wide_widths,
    tree_content_sha256,
    unpack_tree_shard,
    validate_backend_manifest,
    validate_region_normalization,
    write_tree_shard,
    with_content_hash,
)
from teacher_logit_reco.relational_part.train import _capture_diagnostics


def _sample(
    jets: int = 4,
    particles: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[JetIdentity]]:
    tokens = np.zeros((jets, particles, 14), dtype=np.float32)
    mask = np.ones((jets, particles), dtype=bool)
    vectors = np.zeros((jets, particles, 4), dtype=np.float64)
    identities = [
        JetIdentity(file=f"region/class_{row}.root", entry=row, label=row)
        for row in range(jets)
    ]
    for row in range(jets):
        pt = np.asarray([12., 8., 5., 3., 2., 1.])[:particles] + row * .1
        eta = np.asarray([0., .04, .3, -.5, .8, -1.])[:particles]
        phi = np.asarray([0., .03, .5, -1., 2., -2.5])[:particles]
        px, py = pt * np.cos(phi), pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(px**2 + py**2 + pz**2 + (.2 + .1 * row) ** 2)
        vectors[row] = np.stack((px, py, pz, energy), axis=1)
        tokens[row, :, 0] = pt
        tokens[row, :, 1] = eta
        tokens[row, :, 2] = phi
        tokens[row, :, 3] = energy
        tokens[row, :, 4] = np.asarray([1, 0, 0, -1, 1, -1])[:particles]
        for particle in range(particles):
            tokens[row, particle, 5 + particle % 5] = 1
        tokens[row, :, 10] = np.linspace(-.02, .03, particles)
        tokens[row, :, 11] = .002 + row * .0001
        tokens[row, :, 12] = np.linspace(-.03, .05, particles)
        tokens[row, :, 13] = .004 + row * .0001
    return tokens, mask, vectors, identities


def _artifacts():
    tokens, mask, vectors, identities = _sample()
    registry = build_relation_family_registry()
    raw_schema = build_raw_input_schema_contract()
    base = fit_relation_normalization(
        tokens,
        mask,
        identities,
        normalization_contract=build_normalization_contract(
            split_binding_sha256="1" * 64
        ),
        relation_registry=registry,
        raw_input_schema=raw_schema,
        hlt_binding_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        hlt_model_train_content_sha256="4" * 64,
    )
    trees = [
        build_reference_tree(vectors[row], tokens[row], mask[row])
        for row in range(len(tokens))
    ]
    resource = build_angular_tree_resource_contract(
        split_binding_sha256="1" * 64
    )
    region = fit_region_normalization(
        tokens,
        mask,
        identities,
        trees,
        relation_normalization_artifact=base,
        angular_tree_resource_sha256=resource["content_hash"],
    )
    return tokens, mask, vectors, identities, trees, registry, base, region


def test_reference_tree_topology_escheme_n_less_than_k_and_wide_separation() -> None:
    tokens, mask, vectors, _, trees, *_ = _artifacts()
    tree = trees[0]
    assert tree["n_nodes"] == 2 * tree["n_valid"] - 1
    first, second = tree["leaf_to_node"][:2]
    assert tree["parent"][first] == tree["parent"][second]
    root_vector = vectors[0].sum(axis=0)
    np.testing.assert_allclose(tree["vectors"][tree["root"]], root_vector, rtol=2e-6)
    assert tree["pt"][tree["root"]] == pytest.approx(
        np.hypot(root_vector[0], root_vector[1]), rel=2e-6
    )
    assert tree["pt"][tree["root"]] != pytest.approx(tokens[0, :, 0].sum())

    short_tree = build_reference_tree(vectors[0, :3], tokens[0, :3], mask[0, :3])
    assert short_tree["actual_cluster_counts"] == {"2": 2, "4": 3, "8": 3}
    for resolution in (4, 8):
        assigned = short_tree["assignments"][str(resolution)]
        assert len(np.unique(assigned)) == 3

    wide_vectors = vectors[0, :2].copy()
    wide_tokens = tokens[0, :2].copy()
    wide_tokens[:, 1] = np.asarray([-2., 2.])
    wide_tokens[:, 2] = 0
    pt = wide_tokens[:, 0]
    wide_vectors[:, 0] = pt
    wide_vectors[:, 1] = 0
    wide_vectors[:, 2] = pt * np.sinh(wide_tokens[:, 1])
    wide_vectors[:, 3] = np.sqrt(
        wide_vectors[:, 0] ** 2 + wide_vectors[:, 2] ** 2
    )
    wide = build_reference_tree(
        wide_vectors, wide_tokens, np.ones(2, dtype=bool)
    )
    assert wide["n_nodes"] == 3
    assert wide["merge_delta_r"][wide["root"]] > .8


def test_region_41_channels_diagonal_merge_and_permutation_equivariance() -> None:
    tokens, mask, vectors, _, trees, *_ = _artifacts()
    raw = build_region_raw_features(
        trees[0], torch.from_numpy(tokens[0]), torch.from_numpy(mask[0])
    )
    assert raw.shape == (41, 6, 6)
    assert len(REGION_RAW_FEATURE_NAMES) == 41
    assert raw[4:8, torch.arange(6), torch.arange(6)].count_nonzero() == 0
    assert torch.all(raw[:3, torch.arange(6), torch.arange(6)] == 1)
    assert raw[38:41].transpose(-1, -2).equal(-raw[38:41])

    permutation = np.asarray([3, 0, 5, 1, 4, 2])
    inverse = np.argsort(permutation)
    permuted_tree = build_reference_tree(
        vectors[0, permutation],
        tokens[0, permutation],
        mask[0, permutation],
    )
    permuted = build_region_raw_features(
        permuted_tree,
        torch.from_numpy(tokens[0, permutation]),
        torch.from_numpy(mask[0, permutation]),
    )
    restored = permuted[:, inverse][:, :, inverse]
    torch.testing.assert_close(raw, restored, atol=2e-6, rtol=2e-6)


def test_region_production_builder_is_batched_and_never_roundtrips_tokens() -> None:
    tokens, mask, vectors, _ = _sample(jets=2)
    trees = [
        build_reference_tree(vectors[row], tokens[row], mask[row])
        for row in range(2)
    ]
    token_tensor = torch.from_numpy(tokens)
    mask_tensor = torch.from_numpy(mask).unsqueeze(1)
    batched = build_batched_region_raw_features(
        trees, token_tensor, mask_tensor
    )
    singles = torch.stack(
        [
            build_region_raw_features(
                trees[row], token_tensor[row], mask_tensor[row, 0]
            )
            for row in range(2)
        ]
    )
    torch.testing.assert_close(batched, singles, atol=0, rtol=0)
    source = inspect.getsource(build_batched_region_raw_features)
    assert ".cpu(" not in source
    assert ".numpy(" not in source
    assert "for query" not in source
    assert "for context" not in source


def test_region_normalizer_encoder_and_masking() -> None:
    tokens, mask, _, _, trees, _, base, region = _artifacts()
    assert validate_region_normalization(
        region,
        relation_normalization_sha256=base["content_hash"],
    ) == region["content_hash"]
    assert len(region["records"]) == 38
    encoder = RegionEncoder(region)
    raw = torch.from_numpy(tokens[:2])
    valid = torch.from_numpy(mask[:2]).unsqueeze(1)
    details = encoder(raw, valid, trees[:2], return_details=True)
    assert details["encoded"].shape == (2, 12, 6, 6)
    assert torch.isfinite(details["encoded"]).all()
    diagonal = torch.arange(6)
    assert details["normalized"][:, 4:8, diagonal, diagonal].count_nonzero() == 0
    ablated = encoder(
        raw,
        valid,
        trees[:2],
        return_details=True,
        disabled_resolutions=(4,),
    )
    k4_channels = [1, *range(14, 20), *range(28, 30), *range(34, 36), 39]
    assert ablated["normalized"][:, k4_channels].count_nonzero() == 0
    assert (
        ablated["resolution_ablation_domain"]
        == "registered_normalized_K_specific_channels_before_encoder"
    )
    details["encoded"].square().sum().backward()
    assert encoder.encoder[0].weight.grad is not None

    empty_mask = torch.zeros(1, 1, 6, dtype=torch.bool)
    empty_tree = build_reference_tree(
        np.zeros((6, 4)), np.zeros((6, 14)), np.zeros(6, dtype=bool)
    )
    assert encoder(
        torch.zeros(1, 6, 14), empty_mask, [empty_tree]
    ).count_nonzero() == 0


def test_compact_sidecar_atomic_resume_and_runtime_round_trip(tmp_path: Path) -> None:
    _, _, _, identities, trees, *_ = _artifacts()
    path = tmp_path / "shards" / "shard_00000.npz"
    first = write_tree_shard(
        path,
        trees,
        identities,
        hlt_content_sha256="a" * 64,
        tree_resource_sha256="b" * 64,
        backend_manifest_sha256="c" * 64,
    )
    assert first["reused"] is False
    second = write_tree_shard(
        path,
        trees,
        identities,
        hlt_content_sha256="a" * 64,
        tree_resource_sha256="b" * 64,
        backend_manifest_sha256="c" * 64,
    )
    assert second["reused"] is True
    restored_ids, restored = unpack_tree_shard(path)
    assert restored_ids == [identity.key() for identity in identities]
    assert [tree_content_sha256(tree) for tree in restored] == [
        tree_content_sha256(tree) for tree in trees
    ]
    with pytest.raises(FileExistsError, match="stale"):
        write_tree_shard(
            path,
            trees,
            identities,
            hlt_content_sha256="d" * 64,
            tree_resource_sha256="b" * 64,
            backend_manifest_sha256="c" * 64,
        )
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    assert metadata["jet_count"] == 4
    manifest = finalize_tree_split(
        tmp_path / "manifest.json",
        [path.with_suffix(".metadata.json")],
        split="model_train",
        expected_jet_count=4,
        hlt_content_sha256="a" * 64,
        tree_resource_sha256="b" * 64,
        backend_manifest_sha256="c" * 64,
    )
    assert manifest["jet_count"] == 4
    assert finalize_tree_split(
        tmp_path / "manifest.json",
        [path.with_suffix(".metadata.json")],
        split="model_train",
        expected_jet_count=4,
        hlt_content_sha256="a" * 64,
        tree_resource_sha256="b" * 64,
        backend_manifest_sha256="c" * 64,
    ) == manifest


def test_backend_source_and_abi_manifest_fail_closed() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "teacher_logit_reco" / "relational_part" / "csrc"
        / "relational_ca_tree_v1.cpp"
    ).read_text(encoding="utf-8")
    for value in (
        "std::priority_queue",
        "torch::kFloat64",
        "backend_manifest",
        "self_test",
        "build_tree",
        "_OPENMP",
    ):
        assert value in source
    manifest = with_content_hash(
        {
            "contract": "relational_ca_tree_backend_manifest_v2",
            "schema_version": 2,
            "contract_id": "relational_ca_tree_v1",
            "backend_schema_version": 1,
            "source_sha256": "1" * 64,
            "binary_sha256": "2" * 64,
            "compiler_identity": "gcc",
            "compiler_major_version": 13,
            "compiler_version": "13.2.0",
            "compiler_executable": "/usr/bin/g++",
            "compiler_driver_version_line": "g++ 13.2.0",
            "compiler_flags": [
                "-O3", "-std=c++17", "-fopenmp", "-fno-fast-math",
                "-fno-associative-math", "-ffp-contract=off",
            ],
            "platform_architecture": "x86_64",
            "python_major_minor": "3.11",
            "pytorch_version": "2.5",
            "pytorch_cxx11_abi": True,
            "openmp_available": True,
            "self_test_sha256": "3" * 64,
            "compiled_reference_smoke_tree_sha256": "4" * 64,
        }
    )
    assert validate_backend_manifest(manifest) == manifest["content_hash"]
    unsafe = dict(manifest)
    unsafe.pop("content_hash")
    unsafe["compiler_flags"] = [*unsafe["compiler_flags"], "-ffast-math"]
    with pytest.raises(ValueError, match="compiler flags"):
        validate_backend_manifest(with_content_hash(unsafe))


def test_probe_largest_remainder_and_wide_capacity_tie_breaks() -> None:
    identities = [
        f"probe/file_{index // 100}.root#{index}" for index in range(20_000)
    ]
    valid_counts = [index % 100 for index in range(20_000)]
    probe = select_tree_probe(identities, valid_counts)
    assert len(probe["selected_indices"]) == 20_000
    assert len(probe["parity_indices"]) == 1_000
    assert sum(probe["final_quotas"]) == 20_000
    probe_artifact = build_tree_probe_artifact(
        probe,
        valid_counts,
        np.full(20_000, .01),
        np.full(20_000, 512),
        peak_resident_bytes=1_000_000,
        parity_topology_exact=True,
        parity_max_continuous_absolute_error=1e-7,
    )
    assert probe_artifact["limits"]["passed"] is True
    assert probe_artifact["parity"]["topology_exact"] is True

    wide = select_wide_widths()
    assert wide["selected_widths"] == [183, 64, 155]
    assert wide["target_full_incremental_parameters"] == (
        full_active_incremental_parameters()
    )
    assert wide["relative_incremental_mismatch"] <= .02
    selected = tuple(wide["selected_widths"])
    selected_key = (
        wide["absolute_incremental_mismatch"],
        pair_encoder_flops(4, selected),
        sum(selected),
        selected,
    )
    target = wide["target_full_incremental_parameters"]
    base = pair_encoder_parameter_count(4, (64, 64, 64))
    for candidate in (
        (182, 64, 156), (183, 65, 153), (184, 64, 154), (183, 64, 156)
    ):
        increment = pair_encoder_parameter_count(4, candidate) - base
        key = (
            abs(increment - target),
            pair_encoder_flops(4, candidate),
            sum(candidate),
            candidate,
        )
        assert selected_key <= key


class _PairEmbed(torch.nn.Module):
    def __init__(
        self,
        *,
        physical_dimension: int,
        extra_dimension: int,
        hidden: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.pairwise_lv_dim = physical_dimension
        self.pairwise_input_dim = extra_dimension
        self.out_dim = 8
        self.remove_self_pair = False
        self.is_symmetric = physical_dimension == 4 and extra_dimension == 0
        self.sparse_eval = (True, True)

        def stem(dimension):
            modules = [torch.nn.BatchNorm1d(dimension)]
            for output in (*hidden, 8):
                modules.extend(
                    (
                        torch.nn.Conv1d(dimension, output, 1),
                        torch.nn.BatchNorm1d(output),
                        torch.nn.GELU(),
                    )
                )
                dimension = output
            return torch.nn.Sequential(*modules)

        if physical_dimension:
            self.embed = stem(physical_dimension)
        if extra_dimension:
            self.fts_embed = stem(extra_dimension)


class _Transformer(torch.nn.Module):
    def __init__(self, **config) -> None:
        super().__init__()
        self.use_amp = False
        self.pair_extra_dim = int(config.get("pair_extra_dim", 0))
        physical = int(config["pair_input_dim"])
        extra = self.pair_extra_dim
        self.pair_embed = _PairEmbed(
            physical_dimension=physical,
            extra_dimension=extra,
            hidden=tuple(config["pair_embed_dims"]),
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1))
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(
                    17, 1, dropout=0, batch_first=True
                )
                for _ in range(8)
            ]
        )
        self.head = torch.nn.Linear(25, 10)

    def forward(self, x, v=None, mask=None, uu=None):
        bias = self.pair_embed(v, uu=uu, mask=mask)
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
        particle = (x * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).to(x)
        pair = (bias * pair_mask).sum((-1, -2)) / pair_mask.sum(
            (-1, -2)
        ).clamp_min(1)
        return self.head(torch.cat((particle, pair), 1))


def _weaver():
    def pairwise(xi, xj, num_outputs=4):
        value = xi[:, :1] + xj[:, :1]
        return torch.cat([value + index for index in range(num_outputs)], 1)

    return SimpleNamespace(ParticleTransformer=_Transformer, pairwise_lv_fts=pairwise)


def test_all_standard_screening_combinations_and_full_zero_shape_control() -> None:
    tokens, mask_np, vectors_np, _, trees, registry, base, region = _artifacts()
    screening = build_screening_registry(
        relation_registry_sha256=registry["content_hash"]
    )
    base_model = RelationalParticleTransformer(weaver_module=_weaver())
    assert base_model.mod.pair_embed.pairwise_lv_dim == 4
    wide_model = WideBaseParticleTransformer(weaver_module=_weaver())
    assert wide_model.capacity_artifact["selected_widths"] == [183, 64, 155]
    registered_wide = build_registered_wide_model(
        "RPT_BASE_WIDE_MAX",
        screening_registry=screening,
        capacity_artifact=select_wide_widths(),
        weaver_module=_weaver(),
    )
    assert registered_wide.run_id == "RPT_BASE_WIDE_MAX"
    supported_rows = [
        row for row in screening["rows"]
        if row["run_id"] not in ("RPT_BASE", "RPT_BASE_WIDE_MAX")
    ]
    for row in supported_rows:
        families = tuple(row["new_relation_families"])
        model = RelationalFamilyParticleTransformer(
            families,
            normalization_artifact=base,
            region_normalization_artifact=region if "REGION" in families else None,
            force_zero_relations=row["relation_input_mode"] == "forced_zero",
            weaver_module=_weaver(),
        )
        expected = 4 + sum(
            {"PT": 8, "TRACK": 12, "PID": 8, "CHARGE": 6, "DENSITY": 12, "REGION": 12}[f]
            for f in families
        )
        assert model.pair_builder.output_dimension == expected

    inputs = build_particle_transformer_inputs_from_tokens(
        tokens[:2], mask_np[:2], source_view="fixed_hlt"
    )
    common = dict(
        normalization_artifact=base,
        region_normalization_artifact=region,
        weaver_module=_weaver(),
    )
    torch.manual_seed(7)
    full = RelationalFamilyParticleTransformer(
        ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"), **common
    )
    torch.manual_seed(7)
    zero = RelationalFamilyParticleTransformer(
        ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"),
        force_zero_relations=True,
        **common,
    )
    registered_zero = build_registered_screening_model(
        "RPT_FULL_ZERO_REL",
        normalization_artifact=base,
        screening_registry=screening,
        region_normalization_artifact=region,
        weaver_module=_weaver(),
    )
    assert registered_zero.force_zero_relations is True
    assert {
        name: tuple(value.shape) for name, value in full.state_dict().items()
    } == {
        name: tuple(value.shape) for name, value in zero.state_dict().items()
    }
    full.eval()
    diagnostics = full.diagnostics(
        torch.from_numpy(inputs.pf_features),
        torch.from_numpy(inputs.pf_vectors),
        torch.from_numpy(inputs.pf_mask),
        torch.from_numpy(tokens[:2]),
        trees[:2],
        labels=torch.tensor([1, 8]),
    )
    canonical_json_bytes(diagnostics)
    assert set(diagnostics["REGION"]["lca_distributions"]) == {
        "normalized_depth",
        "log_merge_delta_r",
        "log_merge_kt",
        "merge_z",
        "log_merge_mass_fraction",
    }
    assert set(diagnostics["REGION"]["cluster_property_distributions"]) == {
        "2", "4", "8"
    }
    assert diagnostics["REGION"]["performance_by_tree_depth"][
        "event_counts"
    ]
    assert (
        diagnostics["CHARGE"]["pid_conditioned_performance"]
        is not None
    )
    repeated = {
        "points": torch.from_numpy(inputs.pf_points[:1]).repeat(65, 1, 1),
        "features": torch.from_numpy(inputs.pf_features[:1]).repeat(65, 1, 1),
        "lorentz_vectors": torch.from_numpy(
            inputs.pf_vectors[:1]
        ).repeat(65, 1, 1),
        "mask": torch.from_numpy(inputs.pf_mask[:1]).repeat(65, 1, 1),
        "raw_tokens": torch.from_numpy(tokens[:1]).repeat(65, 1, 1),
        "labels": torch.arange(65) % 10,
        "region_trees": [trees[0]] * 65,
    }
    population = _capture_diagnostics(
        full,
        [
            {name: value[:64] for name, value in repeated.items()},
            {name: value[64:] for name, value in repeated.items()},
        ],
        torch.device("cpu"),
    )
    assert population["event_count"] == 65
    assert len(population["values"]["REGION"]["node_counts"]) == 65
    assert len(
        population["values"]["REGION"]["actual_cluster_counts"]["8"]
    ) == 65
    details = zero.pair_features(
        torch.from_numpy(inputs.pf_features),
        torch.from_numpy(inputs.pf_vectors),
        torch.from_numpy(inputs.pf_mask),
        torch.from_numpy(tokens[:2]),
        trees[:2],
        return_details=True,
    )
    assert details["combined"][:, :4].count_nonzero() > 0
    assert details["combined"][:, 4:].count_nonzero() == 0
    determinism = build_global_determinism_contract()
    family_hashes = {
        family: str(index) * 64
        for index, family in enumerate(
            ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"), start=1
        )
    }
    full_contract = build_step5_model_contract(
        "RPT_FULL_ALL",
        normalization_artifact=base,
        screening_registry=screening,
        relation_registry_sha256=registry["content_hash"],
        pair_base_sha256="7" * 64,
        family_contract_sha256=family_hashes,
        weaver_runtime_sha256="8" * 64,
        global_determinism_sha256=determinism["content_hash"],
        region_normalization_artifact=region,
    )
    zero_contract = build_step5_model_contract(
        "RPT_FULL_ZERO_REL",
        normalization_artifact=base,
        screening_registry=screening,
        relation_registry_sha256=registry["content_hash"],
        pair_base_sha256="7" * 64,
        family_contract_sha256=family_hashes,
        weaver_runtime_sha256="8" * 64,
        global_determinism_sha256=determinism["content_hash"],
        region_normalization_artifact=region,
    )
    wide_contract = build_step5_model_contract(
        "RPT_BASE_WIDE_MAX",
        normalization_artifact=base,
        screening_registry=screening,
        relation_registry_sha256=registry["content_hash"],
        pair_base_sha256="7" * 64,
        family_contract_sha256={},
        weaver_runtime_sha256="8" * 64,
        global_determinism_sha256=determinism["content_hash"],
        wide_capacity_artifact=select_wide_widths(),
    )
    assert full_contract["combined_dimension"] == 62
    assert zero_contract["full_zero_exact_shape"] is True
    assert wide_contract["combined_dimension"] == 4

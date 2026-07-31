from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.hlt_offline_structure_distillation import (
    ExtractorResources,
    PHYSICAL_TARGET_IDS,
    build_structure_target_registry,
    build_target_capability_audit,
    build_target_extractor_manifest,
    extract_registered_target,
    load_materialized_hlt_input_view,
    materialize_hlt_input_view,
)
from teacher_logit_reco.hlt_offline_structure_distillation.target_schemas import (
    RELATION_COMPONENTS,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    build_raw_input_schema,
)
from teacher_logit_reco.relational_part import (
    DENSITY_NODE_FEATURE_NAMES,
    TRACK_COMPATIBILITY_FEATURE_NAMES,
    build_density_node_features,
    build_reference_trees,
    build_track_compatibility,
)


def _pairwise_lv_fts(xi, xj, num_outputs):
    assert num_outputs == 4
    shape = torch.broadcast_shapes(xi.shape, xj.shape)
    left = xi.expand(shape)
    right = xj.expand(shape)
    distance = (left - right).square().sum(dim=1)
    return torch.stack(
        tuple(torch.log1p(distance + offset) for offset in range(4)),
        dim=1,
    )


WEAVER_FIXTURE = SimpleNamespace(pairwise_lv_fts=_pairwise_lv_fts)
RESOURCES = ExtractorResources(
    d0_uncertainty_floor=0.001,
    dz_uncertainty_floor=0.002,
    sentinel_policy=None,
    weaver_module=WEAVER_FIXTURE,
)


def _source() -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "status_sha256": "b" * 64,
        "dirty": False,
        "status_hash_policy": (
            "git_diff_binary_HEAD_plus_sorted_untracked_file_bytes_v2"
        ),
    }


def _fixture(*, include_empty: bool = True) -> tuple[np.ndarray, np.ndarray]:
    jets, particles = (3 if include_empty else 2), 6
    raw = np.zeros((jets, particles, 14), dtype=np.float32)
    mask = np.zeros((jets, particles), dtype=bool)
    mask[0] = True
    mask[1, :3] = True
    for row in range(2):
        pt = np.asarray([12.0, 8.0, 5.0, 3.0, 2.0, 1.0])
        eta = np.asarray([0.0, 0.04, 0.3, -0.5, 0.8, -1.0])
        phi = np.asarray([0.0, 0.03, 0.5, -1.0, 2.0, -2.5])
        raw[row, :, 0] = pt
        raw[row, :, 1] = eta
        raw[row, :, 2] = phi
        raw[row, :, 3] = np.sqrt(
            np.square(pt * np.cosh(eta)) + np.square(0.2 + 0.1 * row)
        )
        raw[row, :, 4] = np.asarray([1, 0, 0, -1, 1, -1])
        for particle in range(particles):
            raw[row, particle, 5 + particle % 5] = 1
        raw[row, :, 10] = np.linspace(-0.02, 0.03, particles)
        raw[row, :, 11] = 0.002 + row * 0.0001
        raw[row, :, 12] = np.linspace(-0.03, 0.05, particles)
        raw[row, :, 13] = 0.004 + row * 0.0001
    return raw, mask


def _extract(target_id: str, raw: np.ndarray, mask: np.ndarray, **kwargs):
    return extract_registered_target(
        target_id,
        raw,
        mask,
        resources=RESOURCES,
        **kwargs,
    )


def test_manifest_and_registry_bind_every_physical_extractor() -> None:
    manifest = build_target_extractor_manifest()
    assert manifest["target_ids"] == list(PHYSICAL_TARGET_IDS)
    assert len(PHYSICAL_TARGET_IDS) == len(set(PHYSICAL_TARGET_IDS)) == 28
    assert manifest["label_access"] is False
    assert manifest["constituent_matching_required"] is False

    raw_schema = build_raw_input_schema()
    audit = build_target_capability_audit(
        raw_input_schema=raw_schema, source=_source()
    )
    registry = build_structure_target_registry(
        capability_audit=audit,
        raw_input_schema=raw_schema,
        source=_source(),
    )
    rows = {row["target_id"]: row for row in registry["targets"]}
    for target_id in PHYSICAL_TARGET_IDS:
        assert rows[target_id]["extractor_implementation_status"] == (
            "implemented_step_3"
        )
        assert rows[target_id]["physical_extractor_manifest_sha256"] == (
            manifest["content_hash"]
        )
        assert rows[target_id]["extractor_entrypoint"].endswith(
            ":extract_registered_target"
        )


def test_every_physical_target_runs_on_reader_shaped_miniature_jets() -> None:
    raw, mask = _fixture()
    part_inputs = build_particle_transformer_inputs_from_tokens(raw, mask)
    vectors = np.asarray(part_inputs.pf_vectors).transpose(0, 2, 1)
    for target_id in PHYSICAL_TARGET_IDS:
        result = _extract(target_id, raw, mask, vectors=vectors)
        result.validate()
        assert result.values.shape[0] == raw.shape[0]
        assert result.values.shape[1] == len(result.component_names)
        assert torch.isfinite(result.values).all()
        assert not bool(result.loss_mask[-1].any()) or target_id in {
            "T_OFFLINE_TRACK_32",
            "T_HLT_SELF_TRACK_32",
            "T_OFFLINE_TRACK_COMPONENT_PROXY_17",
            "T_HLT_SELF_TRACK_COMPONENT_PROXY_17",
        }


def test_jet_target_uses_summed_four_vector_and_exact_empty_masks() -> None:
    raw = np.zeros((2, 2, 14), dtype=np.float32)
    mask = np.asarray([[True, True], [False, False]])
    raw[0, :, 0] = [3.0, 4.0]
    raw[0, :, 1] = 0.0
    raw[0, :, 2] = [0.0, np.pi / 2]
    raw[0, :, 3] = [5.0, 6.0]
    raw[0, :, 5] = 1.0
    raw[0, :, 11] = raw[0, :, 13] = 1.0
    result = _extract("T_OFFLINE_JET_10", raw, mask)
    jet_pt = 5.0
    jet_mass = np.sqrt(11.0**2 - 3.0**2 - 4.0**2)
    assert result.values[0, 0] == pytest.approx(np.log1p(jet_pt))
    assert result.values[0, 4] == pytest.approx(np.log1p(jet_mass))
    assert result.values[0, 7] == pytest.approx(4.0 / (7.0 + 1e-6))
    assert result.values[0, 8] == pytest.approx(3.0 / (7.0 + 1e-6))
    assert result.availability_groups[1:4] == ("jet_direction",) * 3
    assert result.loss_mask[0].all()
    assert not result.loss_mask[1].any()
    assert result.values[1].count_nonzero() == 0


def test_zero_vector_pt_masks_only_direction_for_nonempty_jet() -> None:
    raw = np.zeros((1, 2, 14), dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    raw[0, :, 0] = 2.0
    raw[0, :, 2] = [0.0, np.pi]
    raw[0, :, 3] = 3.0
    raw[0, :, 5] = 1.0
    raw[0, :, 11] = raw[0, :, 13] = 1.0
    vectors = np.asarray(
        [[[2.0, 0.0, 0.0, 3.0], [-2.0, 0.0, 0.0, 3.0]]],
        dtype=np.float64,
    )
    result = _extract("T_OFFLINE_JET_10", raw, mask, vectors=vectors)
    assert not result.loss_mask[0, 1:4].any()
    assert result.loss_mask[0, [0, 4, 5, 6, 7, 8, 9]].all()
    assert result.values[0, 1:4].count_nonzero() == 0


def test_composition_and_track_masks_follow_pid_and_measurement_domains() -> None:
    raw = np.zeros((1, 4, 14), dtype=np.float32)
    mask = np.ones((1, 4), dtype=bool)
    raw[0, :, 0] = [10.0, 5.0, 3.0, 2.0]
    raw[0, :, 3] = raw[0, :, 0]
    raw[0, :, 4] = [1.0, 0.0, -1.0, 1.0]
    raw[0, 0, 5] = 1.0
    raw[0, 1, 6] = 1.0
    raw[0, 2, 8] = 1.0
    # Particle 3 is canonical unknown.
    raw[0, :, 10] = [0.2, 9.0, -0.4, 1.0]
    raw[0, :, 11] = [0.1, 0.1, 0.2, 0.0]
    raw[0, :, 12] = [0.1, 9.0, 0.5, 1.0]
    raw[0, :, 13] = [0.1, 0.1, 0.25, 0.0]
    composition = _extract("T_OFFLINE_COMPOSITION_16", raw, mask)
    assert composition.values[0, :6].tolist() == pytest.approx(
        [0.25, 0.25, 0.0, 0.25, 0.0, 0.25]
    )
    assert composition.values[0, 12:15].tolist() == pytest.approx(
        [0.25, 0.25, 0.5]
    )
    track = _extract("T_OFFLINE_TRACK_32", raw, mask)
    assert track.values[0, 0] == pytest.approx(0.5)
    # Unknown with invalid errors is outside charged PID domain, so only no
    # unavailable charged-domain particle remains here.
    assert track.values[0, 2] == pytest.approx(0.0)
    assert track.loss_mask[0, :4].all()
    assert track.loss_mask[0, 4:].all()
    assert track.availability_groups[:4] == (
        "track_availability_observation",
    ) * 4
    assert track.availability_groups[4:] == ("has_valid_track",) * 28

    empty = _extract(
        "T_OFFLINE_TRACK_32",
        np.zeros((1, 2, 14), np.float32),
        np.zeros((1, 2), bool),
    )
    assert empty.loss_mask[0, :4].all()
    assert not empty.loss_mask[0, 4:].any()


def test_density_is_exact_pt_weighted_reduction_of_rpt_node_builder() -> None:
    raw, mask = _fixture(include_empty=False)
    target = _extract("T_OFFLINE_DENSITY_22", raw, mask)
    details = build_density_node_features(
        torch.from_numpy(raw),
        torch.from_numpy(mask).unsqueeze(1),
        d0_uncertainty_floor=RESOURCES.d0_uncertainty_floor,
        dz_uncertainty_floor=RESOURCES.dz_uncertainty_floor,
    )
    weights = torch.from_numpy(raw[0, :, 0]).double()
    expected = (
        details["descriptor"][0].double() * weights.unsqueeze(0)
    ).sum(dim=1) / weights.sum()
    torch.testing.assert_close(target.values[0].double(), expected)
    assert target.component_names == tuple(
        f"scalar_pt_weighted_mean__{name}"
        for name in DENSITY_NODE_FEATURE_NAMES
    )


def test_tree_summary_has_exact_dimensions_entropy_and_scalar_pt_ranking() -> None:
    raw, mask = _fixture()
    result = _extract("T_OFFLINE_CA_TREE_26", raw, mask)
    assert result.values.shape == (3, 26)
    assert result.values[0, 0] == pytest.approx(6 / 128)
    assert result.values[0, 1] == pytest.approx(11 / 255)
    # Cluster fractions and normalized entropies are bounded.
    assert torch.all((result.values[0, 8:17] >= 0) & (result.values[0, 8:17] <= 1))
    assert torch.all((result.values[0, 20:26] >= 0) & (result.values[0, 20:26] <= 1))
    assert not result.loss_mask[2].any()


def test_relation_aggregates_use_exact_coordinates_and_applicability() -> None:
    raw, mask = _fixture()
    for family, components in RELATION_COMPONENTS.items():
        result = _extract(f"T_OFFLINE_RELATION_{family}", raw, mask)
        assert result.component_names == tuple(components)
        assert result.values.shape == (3, len(components))
        assert not result.loss_mask[2].any()
    # Six valid particles yield 15 unordered BASE4 pairs and 30 directed PT
    # pairs; the fake Weaver output is symmetric so its population std is
    # still calculated only once per unordered pair.
    base = _extract("T_OFFLINE_RELATION_BASE4", raw, mask)
    assert base.diagnostics["self_pairs_included"] is False
    pid = _extract("T_OFFLINE_RELATION_PID", raw, mask)
    assert torch.allclose(pid.values[0, :6].sum(), torch.tensor(1.0))


def test_track_pair_is_exact_rpt_compatibility_with_directed_mask() -> None:
    raw, mask = _fixture(include_empty=False)
    target = _extract("T_HLT_TRACK_PAIR_13", raw, mask)
    details = build_track_compatibility(
        torch.from_numpy(raw),
        torch.from_numpy(mask).unsqueeze(1),
        d0_uncertainty_floor=RESOURCES.d0_uncertainty_floor,
        dz_uncertainty_floor=RESOURCES.dz_uncertainty_floor,
    )
    assert target.component_names == TRACK_COMPATIBILITY_FEATURE_NAMES
    torch.testing.assert_close(
        target.values.masked_select(target.loss_mask),
        details["compatibility"].masked_select(target.loss_mask),
    )
    assert target.values.masked_select(~target.loss_mask).count_nonzero() == 0
    diagonal = torch.arange(raw.shape[1])
    assert not target.loss_mask[:, :, diagonal, diagonal].any()
    assert target.loss_mask[0, :, 0, 3].all()
    assert target.loss_mask[0, :, 3, 0].all()


def test_region_pair_is_upper_triangular_and_uses_target_specific_transforms() -> None:
    raw, mask = _fixture(include_empty=False)
    vectors = build_particle_transformer_inputs_from_tokens(
        raw, mask
    ).pf_vectors.transpose(0, 2, 1)
    trees = build_reference_trees(vectors, raw, mask)
    target = _extract(
        "T_HLT_REGION_PAIR_8",
        raw,
        mask,
        vectors=vectors,
        trees=trees,
    )
    assert target.values.shape == (2, 8, 6, 6)
    assert not target.loss_mask.tril(diagonal=0).any()
    assert target.loss_mask[0, :, 0, 1].all()
    assert torch.all((target.values[:, 0:3] == 0) | (target.values[:, 0:3] == 1))
    assert torch.all((target.values[:, 6] >= 0) & (target.values[:, 6] <= 0.5))
    tree = trees[0]
    left = int(tree["leaf_to_node"][0])
    right = int(tree["leaf_to_node"][1])
    ancestors = set()
    while left >= 0:
        ancestors.add(left)
        left = int(tree["parent"][left])
    while right not in ancestors:
        right = int(tree["parent"][right])
    assert target.values[0, 3, 0, 1] == pytest.approx(
        float(tree["depth"][right]) / 127
    )
    assert target.values[0, 4, 0, 1] == pytest.approx(
        np.log1p(float(tree["merge_delta_r"][right]))
    )


def test_offline_and_hlt_self_targets_share_identical_extractors() -> None:
    raw, mask = _fixture()
    suffixes = (
        "JET_10",
        "COMPOSITION_16",
        "TRACK_32",
        "DENSITY_22",
        "CA_TREE_26",
        "TRACK_COMPONENT_PROXY_17",
    )
    for suffix in suffixes:
        offline = _extract(f"T_OFFLINE_{suffix}", raw, mask)
        hlt = _extract(f"T_HLT_SELF_{suffix}", raw, mask)
        assert torch.equal(offline.values, hlt.values)
        assert torch.equal(offline.loss_mask, hlt.loss_mask)
    for family in RELATION_COMPONENTS:
        offline = _extract(f"T_OFFLINE_RELATION_{family}", raw, mask)
        hlt = _extract(f"T_HLT_SELF_RELATION_{family}", raw, mask)
        assert torch.equal(offline.values, hlt.values)
        assert torch.equal(offline.loss_mask, hlt.loss_mask)


def test_track_component_proxy_is_deterministic_and_masks_padded_slots() -> None:
    raw = np.zeros((1, 4, 14), dtype=np.float32)
    mask = np.ones((1, 4), dtype=bool)
    raw[0, :, 0] = [10.0, 8.0, 4.0, 2.0]
    raw[0, :, 1] = [0.0, 0.01, 1.0, -1.0]
    raw[0, :, 2] = [0.0, 0.01, 1.0, -1.0]
    raw[0, :, 3] = raw[0, :, 0] * np.cosh(raw[0, :, 1])
    raw[0, :, 5] = 1.0
    raw[0, :, 10] = [0.1, 0.1, 1.0, -1.0]
    raw[0, :, 11] = 0.1
    raw[0, :, 12] = [0.2, 0.2, 1.0, -1.0]
    raw[0, :, 13] = 0.1
    first = _extract("T_OFFLINE_TRACK_COMPONENT_PROXY_17", raw, mask)
    second = _extract("T_OFFLINE_TRACK_COMPONENT_PROXY_17", raw, mask)
    assert torch.equal(first.values, second.values)
    assert first.values[0, 0] == pytest.approx(1 / 8)
    assert first.values[0, 1] == pytest.approx(2 / 40)
    assert first.values[0, 2] == pytest.approx(18 / 24)
    assert first.loss_mask[0, 1:5].all()
    assert not first.loss_mask[0, 5:].any()


def test_nonfinite_valid_source_and_unregistered_targets_fail_closed() -> None:
    raw, mask = _fixture(include_empty=False)
    broken = raw.copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="valid raw tokens"):
        _extract("T_OFFLINE_JET_10", broken, mask)
    with pytest.raises(ValueError, match="not a Step-3 physical target"):
        _extract("T_HLT_TRACK_ORIGIN_TRUTH", raw, mask)


def test_real_hlt_cache_materializer_is_label_blind_and_deterministic(
    tmp_path, monkeypatch
) -> None:
    identities = np.asarray(["jet-a", "jet-b"])
    tokens = np.zeros((2, 128, 14), dtype=np.float32)
    tokens[:, :2, 0] = 1.0
    mask = np.zeros((2, 128), dtype=bool)
    mask[:, :2] = True
    measurement_states = np.zeros((2, 128, 14), dtype=np.int8)
    cache = tmp_path / "hlt_cache"
    cache.mkdir()
    (cache / "hlt_v3_arrays.npz").write_bytes(b"authenticated-cache")
    monkeypatch.setattr(
        "teacher_logit_reco.hlt_offline_structure_distillation.input_views.load_hlt_v3_cache",
        lambda path: (
            {
                "identities": identities,
                "tokens": tokens,
                "mask": mask,
                "measurement_states": measurement_states,
            },
            {
                "content_hash": "d" * 64,
                "logical_role": "model_train",
                "replica_id": 0,
                "realization_policy": "R_MULTI",
                "degradation_profile_id": "D_NOMINAL",
                "source": _source(),
            },
        ),
    )
    first = materialize_hlt_input_view(
        hlt_cache_path=cache,
        split="model_train",
        replica_id=0,
        output=tmp_path / "view.npz",
        parent_hashes={"campaign": "a" * 64},
        source=_source(),
    )
    second = materialize_hlt_input_view(
        hlt_cache_path=cache,
        split="model_train",
        replica_id=0,
        output=tmp_path / "view.npz",
        parent_hashes={"campaign": "a" * 64},
        source=_source(),
    )
    assert first == second
    independent = materialize_hlt_input_view(
        hlt_cache_path=cache,
        split="model_train",
        replica_id=0,
        output=tmp_path / "independent-view.npz",
        parent_hashes={"campaign": "a" * 64},
        source=_source(),
    )
    assert independent["npz_sha256"] == first["npz_sha256"]
    assert first["contract"] == "hosd_label_blind_input_view_v4"
    assert first["mmap_store"]["contract"] == "hosd_npy_mmap_store_v2"
    assert set(first["mmap_store"]["members"]) == {
        "identities",
        "tokens",
        "mask",
        "vectors",
        "measurement_states",
    }
    assert first["schema_version"] == 4
    assert first["storage_layout"] == (
        "deterministic_npz_plus_authenticated_npy_mmap_v2"
    )
    assert (
        (tmp_path / "independent-view.npz").read_bytes()
        == (tmp_path / "view.npz").read_bytes()
    )
    assert not first["contains_labels"]
    with np.load(tmp_path / "view.npz", allow_pickle=False) as payload:
        assert set(payload.files) == {
            "identity",
            "mask",
            "measurement_states",
            "raw_tokens",
            "vectors",
        }
    loaded, loaded_metadata = load_materialized_hlt_input_view(
        tmp_path / "view.npz", expected_source=_source()
    )
    assert np.array_equal(loaded["measurement_states"], measurement_states)
    assert isinstance(loaded["tokens"], np.memmap)
    assert isinstance(loaded["mask"], np.memmap)
    assert loaded_metadata["logical_role"] == "model_train"
    assert loaded_metadata["replica_id"] == 0
    assert loaded_metadata["realization_policy"] == "R_MULTI"
    sidecar = (
        tmp_path
        / independent["mmap_store"]["directory"]
        / independent["mmap_store"]["members"]["tokens"]["filename"]
    )
    sidecar_bytes = sidecar.read_bytes()
    sidecar.write_bytes(sidecar_bytes[:-1] + bytes([sidecar_bytes[-1] ^ 1]))
    with pytest.raises(ValueError, match="memory-map member differs"):
        load_materialized_hlt_input_view(tmp_path / "independent-view.npz")
    sidecar.write_bytes(sidecar_bytes)
    # The adjacent manifest authenticates bytes, not merely shapes or names.
    with np.load(tmp_path / "view.npz", allow_pickle=False) as payload:
        drifted = {name: np.asarray(payload[name]) for name in payload.files}
    drifted["raw_tokens"] = drifted["raw_tokens"].copy()
    drifted["raw_tokens"][0, 0, 0] += 1
    np.savez(tmp_path / "view.npz", **drifted)
    with pytest.raises(ValueError, match="lineage differs"):
        load_materialized_hlt_input_view(tmp_path / "view.npz")
    monkeypatch.setattr(
        "teacher_logit_reco.hlt_offline_structure_distillation.input_views.load_hlt_v3_cache",
        lambda path: (
            {
                "identities": identities,
                "tokens": tokens,
                "mask": mask,
                "measurement_states": measurement_states,
            },
            {
                "content_hash": "d" * 64,
                "logical_role": "val_stop",
                "replica_id": 0,
                "realization_policy": "R_FIXED",
                "degradation_profile_id": "D_NOMINAL",
                "source": _source(),
            },
        ),
    )
    with pytest.raises(ValueError, match="source coordinate"):
        materialize_hlt_input_view(
            hlt_cache_path=cache,
            split="model_train",
            replica_id=0,
            output=tmp_path / "wrong-view.npz",
            parent_hashes={"campaign": "a" * 64},
            source=_source(),
        )

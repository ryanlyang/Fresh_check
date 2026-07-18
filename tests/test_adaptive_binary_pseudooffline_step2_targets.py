from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from jetclass_fresh.hlt_cache import (
    DEFAULT_HLT_SEEDS,
    fixed_hlt_params_from_profile,
    generate_and_cache_hlt_view,
    load_cached_hlt_view,
)
from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    FileRecord,
    JetIdentity,
    JetView,
    LABEL_NAMES,
    RAW_TOKEN_DIM,
    SPLIT_ORDER,
    SplitManifest,
    manifest_hash,
    save_split_manifest,
)
from teacher_logit_reco.architecture_view_part import save_cached_offline_view
from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_COMPACT_TARGET_CODEC_NAME,
    ABPH_HLT_DEGRADATION_STRENGTH,
    ABPH_HLT_PROFILE,
    ABPH_LEVEL_CAPACITIES,
    PARTICLE_TARGET_NAMES,
    TOPOLOGY_ACTIVE_TERMINAL,
    AdaptiveBinaryHierarchyLayout,
    adaptive_binary_target_invariant_report,
    audit_adaptive_binary_target_cache,
    build_adaptive_binary_target_cache,
    build_adaptive_binary_targets,
    exclusive_binary_partition,
    load_adaptive_binary_target_shard,
    wrap_phi,
)


def _token(pt: float, eta: float, phi: float, pid: int = 0, charge: float = 1.0) -> np.ndarray:
    result = np.zeros(RAW_TOKEN_DIM, dtype=np.float32)
    result[:5] = (pt, eta, phi, pt * np.cosh(eta) + 0.2, charge)
    if pid < 5:
        result[5 + pid] = 1.0
    result[10:] = (0.01, 0.002, -0.03, 0.004)
    return result


def _toy_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[JetIdentity, ...]]:
    n_jets = 3
    hlt = np.zeros((n_jets, 128, RAW_TOKEN_DIM), dtype=np.float32)
    offline = np.zeros_like(hlt)
    hlt_mask = np.zeros((n_jets, 128), dtype=bool)
    offline_mask = np.zeros_like(hlt_mask)
    counts = (7, 1, 8)
    for jet_index, count in enumerate(counts):
        hlt_count = max(1, count - 2)
        hlt_mask[jet_index, :hlt_count] = True
        offline_mask[jet_index, :count] = True
        for particle_index in range(count):
            phi = float(wrap_phi(3.10 + 0.035 * particle_index)) if jet_index == 0 else -0.35 + 0.11 * particle_index
            row = _token(
                35.0 - 2.0 * particle_index + jet_index,
                -0.28 + 0.08 * particle_index,
                phi,
                pid=particle_index % 6,
                charge=1.0 if particle_index % 2 == 0 else -1.0,
            )
            offline[jet_index, particle_index] = row
            if particle_index < hlt_count:
                hlt[jet_index, particle_index] = row
    jet_ids = tuple(
        JetIdentity(file=f"HToBB_{index:03d}.root", entry=100 + index, label=1)
        for index in range(n_jets)
    )
    return hlt, hlt_mask, offline, offline_mask, jet_ids


@pytest.mark.parametrize("grouping", ("exclusive_kt", "cambridge_aachen"))
def test_every_frontier_is_an_exact_nonempty_partition(grouping: str):
    hlt, hlt_mask, offline, offline_mask, jet_ids = _toy_arrays()
    output = build_adaptive_binary_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        jet_ids=jet_ids,
        layout=AdaptiveBinaryHierarchyLayout(grouping=grouping),
    )
    report = adaptive_binary_target_invariant_report(output)
    assert report["ok"], report["problems"]
    assert report["max_frontier_particle_multiplicity"] == 1
    assert tuple(level.shape[1] for level in output.level_masks) == ABPH_LEVEL_CAPACITIES
    for jet_index in range(output.n_jets):
        expected = offline_mask[jet_index].astype(np.int64)
        for level_mask, membership in zip(output.level_masks, output.level_membership):
            active = membership[jet_index, level_mask[jet_index]]
            assert np.all(active.sum(axis=1) > 0)
            np.testing.assert_array_equal(active.sum(axis=0), expected)


def test_singleton_is_carried_without_a_physical_empty_sibling():
    hlt, hlt_mask, offline, offline_mask, jet_ids = _toy_arrays()
    output = build_adaptive_binary_targets(
        hlt[1:2],
        hlt_mask[1:2],
        offline[1:2],
        offline_mask[1:2],
        jet_ids=jet_ids[1:2],
    )
    for mask, topology, membership in zip(
        output.level_masks, output.level_topology, output.level_membership
    ):
        assert mask[0].sum() == 1
        assert topology[0, 0] == TOPOLOGY_ACTIVE_TERMINAL
        assert membership[0, 0].sum() == 1
        assert not membership[0, 1:].any()


def test_wrapped_phi_is_used_for_targets_and_clustering():
    hlt = np.zeros((1, 128, RAW_TOKEN_DIM), dtype=np.float32)
    offline = np.zeros_like(hlt)
    hlt_mask = np.zeros((1, 128), dtype=bool)
    offline_mask = np.zeros_like(hlt_mask)
    hlt[0, 0] = _token(50.0, 0.0, np.pi - 0.01)
    hlt_mask[0, 0] = True
    offline[0, 0] = _token(30.0, 0.0, -np.pi + 0.01)
    offline[0, 1] = _token(20.0, 0.01, np.pi - 0.02)
    offline_mask[0, :2] = True
    output = build_adaptive_binary_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        jet_ids=(JetIdentity(file="HToBB_phi.root", entry=7, label=1),),
    )
    phi_index = PARTICLE_TARGET_NAMES.index("phi_hlt_relative")
    assert output.particle_targets[0, 0, phi_index] == pytest.approx(0.02, abs=2.0e-6)
    assert np.all(output.particle_targets[0, :2, phi_index] >= -np.pi)
    assert np.all(output.particle_targets[0, :2, phi_index] < np.pi)
    assert wrap_phi(np.pi) == pytest.approx(-np.pi)


@pytest.mark.parametrize("grouping", ("exclusive_kt", "cambridge_aachen"))
def test_clustering_treats_particles_across_phi_boundary_as_neighbors(grouping: str):
    tokens = np.zeros((128, RAW_TOKEN_DIM), dtype=np.float32)
    tokens[0] = _token(20.0, 0.0, np.pi - 0.01)
    tokens[1] = _token(20.0, 0.0, -np.pi + 0.01)
    tokens[2] = _token(20.0, 0.0, 0.0)
    children = exclusive_binary_partition(
        tokens,
        (0, 1, 2),
        grouping=grouping,
        jet_identity=("HToBB_phi.root", 9, 1),
    )
    assert any(set(child) == {0, 1} for child in children)


@pytest.mark.parametrize("grouping", ("exclusive_kt", "cambridge_aachen"))
def test_binary_partition_and_target_identities_are_byte_deterministic(grouping: str):
    hlt, hlt_mask, offline, offline_mask, jet_ids = _toy_arrays()
    first_children = exclusive_binary_partition(
        offline[0], range(7), grouping=grouping, jet_identity=jet_ids[0]
    )
    second_children = exclusive_binary_partition(
        offline[0], range(7), grouping=grouping, jet_identity=jet_ids[0]
    )
    assert first_children == second_children
    assert set(first_children[0]).isdisjoint(first_children[1])
    assert set(first_children[0]).union(first_children[1]) == set(range(7))

    first = build_adaptive_binary_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        jet_ids=jet_ids,
        layout=AdaptiveBinaryHierarchyLayout(grouping=grouping),
    )
    second = build_adaptive_binary_targets(
        hlt.copy(),
        hlt_mask.copy(),
        offline.copy(),
        offline_mask.copy(),
        jet_ids=jet_ids,
        layout=AdaptiveBinaryHierarchyLayout(grouping=grouping),
    )
    assert first.root_identities.tobytes() == second.root_identities.tobytes()
    for left, right in zip(first.level_identities, second.level_identities):
        assert left.tobytes() == right.tobytes()

    relabeled = JetIdentity(file=jet_ids[0].file, entry=jet_ids[0].entry, label=9)
    assert exclusive_binary_partition(
        offline[0], range(7), grouping=grouping, jet_identity=relabeled
    ) == first_children


def test_kt_and_ca_targets_share_the_exact_root_identity_and_values():
    hlt, hlt_mask, offline, offline_mask, jet_ids = _toy_arrays()
    kt = build_adaptive_binary_targets(
        hlt, hlt_mask, offline, offline_mask, jet_ids=jet_ids,
        layout=AdaptiveBinaryHierarchyLayout(grouping="exclusive_kt"),
    )
    ca = build_adaptive_binary_targets(
        hlt, hlt_mask, offline, offline_mask, jet_ids=jet_ids,
        layout=AdaptiveBinaryHierarchyLayout(grouping="cambridge_aachen"),
    )
    np.testing.assert_array_equal(kt.root_features, ca.root_features)
    assert kt.root_identities.tobytes() == ca.root_identities.tobytes()


def _cache_manifest(n_jets: int = 3) -> SplitManifest:
    prefixes = tuple(FILE_PREFIX_TO_LABEL)
    splits: dict[str, list[JetIdentity]] = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        splits[split] = [
            JetIdentity(
                file=f"{prefixes[index % len(prefixes)]}_{split_index:03d}.root",
                entry=1000 * split_index + index,
                label=index % len(LABEL_NAMES),
            )
            for index in range(n_jets)
        ]
    records = [
        FileRecord(path=row.file, label=row.label, num_entries=10_000)
        for rows in splits.values()
        for row in rows
    ]
    return SplitManifest(
        data_dir="toy",
        max_constits=128,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: n_jets for split in SPLIT_ORDER},
        split_seeds={split: 900 + index for index, split in enumerate(SPLIT_ORDER)},
        file_records=records,
        splits=splits,
        metadata={"step2_test": True},
    )


def _offline_view(manifest: SplitManifest, split: str) -> JetView:
    identities = tuple(manifest.splits[split])
    n_jets = len(identities)
    tokens = np.zeros((n_jets, 128, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 128), dtype=bool)
    for jet_index in range(n_jets):
        count = 4 + jet_index
        mask[jet_index, :count] = True
        for particle_index in range(count):
            tokens[jet_index, particle_index] = _token(
                30.0 - particle_index,
                -0.2 + 0.08 * particle_index,
                -0.3 + 0.12 * particle_index,
                pid=particle_index % 6,
            )
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=np.asarray([row.label for row in identities], dtype=np.int64),
        jet_ids=list(identities),
        split=split,
        metadata={"source_manifest_hash": manifest_hash(manifest), "view": "offline"},
    )


def test_sharded_cache_binds_provenance_round_trips_and_detects_tampering():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = _cache_manifest()
        manifest_path = root / "split_manifest.json.gz"
        hlt_cache = root / "hlt"
        offline_cache = root / "offline"
        target_cache = root / "targets"
        save_split_manifest(manifest, manifest_path)
        offline = _offline_view(manifest, "model_train")
        save_cached_offline_view(offline, offline_cache)
        generate_and_cache_hlt_view(
            offline,
            hlt_cache,
            seed=DEFAULT_HLT_SEEDS["model_train"],
            params=fixed_hlt_params_from_profile(
                ABPH_HLT_PROFILE, ABPH_HLT_DEGRADATION_STRENGTH
            ),
            hlt_degradation_strength=ABPH_HLT_DEGRADATION_STRENGTH,
        )
        metadata = build_adaptive_binary_target_cache(
            manifest_path=manifest_path,
            hlt_cache_dir=hlt_cache,
            offline_cache_dir=offline_cache,
            output_cache_dir=target_cache,
            split="model_train",
            grouping="exclusive_kt",
            chunk_size=2,
        )
        assert metadata["offline_final_test_loaded"] is False
        assert metadata["n_shards"] == 2
        assert metadata["hlt_content_hash"]
        assert metadata["offline_content_hash"]
        assert metadata["target_content_hash"]
        first = load_adaptive_binary_target_shard(
            target_cache, "model_train", "exclusive_kt", 0
        )
        assert first.targets.n_jets == 2
        assert adaptive_binary_target_invariant_report(first.targets)["ok"]
        audit = audit_adaptive_binary_target_cache(
            target_cache,
            manifest_path=manifest_path,
            splits=("model_train",),
            groupings=("exclusive_kt",),
        )
        assert audit["ok"], audit["problems"]

        shard_path = next((target_cache / "model_train_exclusive_kt_adaptive_binary_targets").glob("*.npz"))
        with np.load(shard_path, allow_pickle=False) as source:
            tampered = {key: np.asarray(source[key]) for key in source.files}
        tampered["root_features"] = tampered["root_features"].copy()
        tampered["root_features"][0, 0] += 1.0
        np.savez_compressed(shard_path, **tampered)
        with pytest.raises(ValueError, match="shard hash mismatch"):
            load_adaptive_binary_target_shard(
                target_cache, "model_train", "exclusive_kt", 0
            )


def test_target_cache_refuses_offline_final_test():
    with pytest.raises(ValueError, match="restricted"):
        build_adaptive_binary_target_cache(
            manifest_path="unused",
            hlt_cache_dir="unused",
            offline_cache_dir="unused",
            output_cache_dir="unused",
            split="final_test",
            grouping="exclusive_kt",
        )


def test_compact_cache_round_trips_real_targets_and_audits_forensic_identities(tmp_path: Path):
    manifest = _cache_manifest(n_jets=4)
    manifest_path = tmp_path / "split_manifest.json.gz"
    hlt_cache = tmp_path / "hlt"
    offline_cache = tmp_path / "offline"
    target_cache = tmp_path / "targets"
    save_split_manifest(manifest, manifest_path)
    offline = _offline_view(manifest, "model_train")
    save_cached_offline_view(offline, offline_cache)
    generate_and_cache_hlt_view(
        offline,
        hlt_cache,
        seed=DEFAULT_HLT_SEEDS["model_train"],
        params=fixed_hlt_params_from_profile(
            ABPH_HLT_PROFILE, ABPH_HLT_DEGRADATION_STRENGTH
        ),
        hlt_degradation_strength=ABPH_HLT_DEGRADATION_STRENGTH,
    )
    metadata = build_adaptive_binary_target_cache(
        manifest_path=manifest_path,
        hlt_cache_dir=hlt_cache,
        offline_cache_dir=offline_cache,
        output_cache_dir=target_cache,
        split="model_train",
        grouping="exclusive_kt",
        chunk_size=2,
        storage_codec=ABPH_COMPACT_TARGET_CODEC_NAME,
        forensic_jets_per_class=1,
    )
    assert metadata["storage_codec"] == ABPH_COMPACT_TARGET_CODEC_NAME
    assert metadata["feature_dtype"] == "float32"
    assert metadata["forensic_identity_sample"]["n_jets"] == 4
    assert all(row["encoded_content_hash"] for row in metadata["shards"])
    shard = load_adaptive_binary_target_shard(
        target_cache, "model_train", "exclusive_kt", 0
    )
    hlt = load_cached_hlt_view(hlt_cache, "model_train", verify_hash=True)
    expected = build_adaptive_binary_targets(
        hlt.tokens[:2],
        hlt.mask[:2],
        offline.tokens[:2],
        offline.mask[:2],
        jet_ids=tuple(offline.jet_ids[:2]),
    )
    expected_arrays = expected.array_dict()
    observed_arrays = shard.targets.array_dict()
    assert set(observed_arrays) == set(expected_arrays)
    for key in expected_arrays:
        assert observed_arrays[key].dtype == expected_arrays[key].dtype, key
        assert observed_arrays[key].shape == expected_arrays[key].shape, key
        assert observed_arrays[key].tobytes() == expected_arrays[key].tobytes(), key
    assert adaptive_binary_target_invariant_report(shard.targets)["ok"]
    audit = audit_adaptive_binary_target_cache(
        target_cache,
        manifest_path=manifest_path,
        splits=("model_train",),
        groupings=("exclusive_kt",),
    )
    assert audit["ok"], audit["problems"]
    report = audit["reports"]["model_train/exclusive_kt"]
    assert report["storage_codec"] == ABPH_COMPACT_TARGET_CODEC_NAME
    assert report["forensic_identity_sample_count"] == 4

    shard_path = next(
        (target_cache / "model_train_exclusive_kt_adaptive_binary_targets").glob(
            "shard_*.npz"
        )
    )
    with np.load(shard_path, allow_pickle=False) as source:
        tampered = {key: np.asarray(source[key]) for key in source.files}
    tampered["payload__root_features"] = tampered["payload__root_features"].copy()
    tampered["payload__root_features"][0] ^= np.uint8(1)
    np.savez_compressed(shard_path, **tampered)
    with pytest.raises(ValueError, match="shard hash mismatch"):
        load_adaptive_binary_target_shard(
            target_cache, "model_train", "exclusive_kt", 0
        )

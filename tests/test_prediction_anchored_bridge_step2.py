from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.local_particle_residual_field import (
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    BridgeScalers,
    AllocationNpzStager,
    AllocationRamLedger,
    DerivedShardLRU,
    FrozenR0Runner,
    PredictionAnchoredBridgeProvider,
    StreamedR0TrainConfig,
    apply_bridge_control,
    bridge_response,
    build_bridge_recipe,
    build_matched_wrong_event_map,
    compute_local_particle_residual_fields,
    deterministic_rank_range,
    fit_bridge_scalers,
    physical_loss_groups,
    require_single_node,
    train_streamed_r0,
    validate_bridge_recipe,
    virtual_bridge,
)
from teacher_logit_reco.local_particle_residual_field.model import (
    LocalResidualFieldReconstructorConfig,
    build_local_residual_field_reconstructor,
)


def _tokens(n: int, p: int, *, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((n, p, 14), dtype=np.float32)
    mask = np.ones((n, p), dtype=bool)
    if p > 2:
        mask[::3, -1] = False
    for event in range(n):
        for particle in range(p):
            if not mask[event, particle]:
                continue
            pt = 1.0 + event * 0.2 + particle * 0.1 + offset
            eta = -0.4 + 0.08 * particle
            phi = -0.8 + 0.11 * event + 0.03 * particle
            tokens[event, particle, 0] = pt
            tokens[event, particle, 1] = eta
            tokens[event, particle, 2] = phi
            tokens[event, particle, 3] = pt * np.cosh(eta)
            tokens[event, particle, 5 + particle % 5] = 1.0
    return tokens, mask


def _write_source_pair(root: Path, *, n: int = 10, p: int = 5):
    hlt_tokens, hlt_mask = _tokens(n, p)
    offline_tokens, offline_mask = _tokens(n, p, offset=0.15)
    labels = (np.arange(n) % 2).astype(np.int64)
    file_indices = np.zeros(n, dtype=np.int32)
    entries = np.arange(100, 100 + n, dtype=np.int64)
    files = ["synthetic.root"]
    ids = tuple(
        JetIdentity(file=files[0], entry=int(entry), label=int(label))
        for entry, label in zip(entries, labels)
    )
    paths = {}
    arrays_by_source = {}
    for name, tokens, mask in (
        ("hlt", hlt_tokens, hlt_mask),
        ("offline", offline_tokens, offline_mask),
    ):
        arrays = {
            "tokens": tokens,
            "mask": mask,
            "labels": labels,
            "jet_file_indices": file_indices,
            "jet_entries": entries,
        }
        npz = root / f"{name}.npz"
        metadata_path = root / f"{name}.json"
        np.savez_compressed(npz, **arrays)
        content = hash_arrays(arrays)
        metadata = {
            "jet_files": files,
            "jet_identity_hash": jet_identity_hash(ids),
            ("hlt_content_hash" if name == "hlt" else "offline_content_hash"): content,
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        paths[name] = (npz, metadata_path)
        arrays_by_source[name] = arrays
    return paths, arrays_by_source


class _FakeR0:
    checkpoint_sha256 = "f" * 64

    def predict_numpy(self, tokens, mask):
        tokens = np.asarray(tokens, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        fields = np.zeros((*mask.shape, 50), dtype=np.float32)
        fields[..., 0] = tokens[..., 0] * 0.01
        fields[..., 45] = 0.25
        hidden = np.repeat(tokens[..., :1], 4, axis=-1).astype(np.float32)
        fields[~mask] = 0
        hidden[~mask] = 0
        return fields, hidden


def test_single_node_and_deterministic_rank_ranges():
    assert require_single_node({"SLURM_NNODES": "1"}) == 1
    with pytest.raises(RuntimeError, match="exactly one node"):
        require_single_node({"SLURM_NNODES": "2"})
    ranges = [deterministic_rank_range(10, rank, 3) for rank in range(3)]
    assert ranges == [(0, 4), (4, 7), (7, 10)]


def test_allocation_ledger_is_locked_and_raw_is_non_evictable(tmp_path):
    kwargs = dict(
        allocation_id="locked",
        capacity_bytes=1000,
        allow_unverified_test_root=True,
    )
    first = AllocationRamLedger(tmp_path, **kwargs)
    second = AllocationRamLedger(tmp_path, **kwargs)

    def reserve(ledger):
        try:
            return ledger.reserve(owner="rank", role="derived", expected_bytes=500, category="derived")
        except MemoryError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, (first, second)))
    assert sum(value is not None for value in results) == 1
    for value in results:
        if value is not None:
            first.release(value)
    raw = first.reserve(owner="rank0", role="raw", expected_bytes=100, category="raw")
    first.commit(raw, measured_bytes=100)
    first.finalize_raw_stage()
    with pytest.raises(RuntimeError, match="non-evictable"):
        first.release(raw)
    assert first.snapshot()["mandatory_headroom_fraction"] == 0.20
    first.cleanup()


def test_one_open_staging_streamed_truth_and_r0_agree(tmp_path):
    paths, source = _write_source_pair(tmp_path)
    ledger = AllocationRamLedger(
        tmp_path / "ram",
        allocation_id="stage",
        capacity_bytes=32 * 1024 * 1024,
        allow_unverified_test_root=True,
    )
    stager = AllocationNpzStager(ledger, rank=0, world_size=2)
    hlt, offline, report = stager.stage_pair(
        hlt_npz=paths["hlt"][0],
        hlt_metadata=paths["hlt"][1],
        offline_npz=paths["offline"][0],
        offline_metadata=paths["offline"][1],
        shard_size=4,
    )
    assert set(report["persistent_npz_open_counts"].values()) == {1}
    assert hlt.manifest["raw_non_evictable"] is True
    assert hlt.manifest["rank_ownership"] == [
        {"rank": 0, "start": 0, "stop": 5},
        {"rank": 1, "start": 5, "stop": 10},
    ]
    staged = hlt.read_indices([8, 1, 4])
    np.testing.assert_array_equal(staged["tokens"], source["hlt"]["tokens"][[8, 1, 4]])
    provider = PredictionAnchoredBridgeProvider(
        hlt=hlt,
        offline=offline,
        r0=_FakeR0(),
        ledger=ledger,
        rank=0,
        world_size=1,
        derived_capacity_bytes=128 * 1024,
    )
    indices = [0, 3, 7]
    streamed_truth, streamed_mask = provider.truth_for_indices(indices)
    direct_truth, direct_mask, *_ = compute_local_particle_residual_fields(
        source["hlt"]["tokens"][indices],
        source["hlt"]["mask"][indices],
        source["offline"]["tokens"][indices],
        source["offline"]["mask"][indices],
    )
    np.testing.assert_array_equal(streamed_truth, direct_truth)
    np.testing.assert_array_equal(streamed_mask, direct_mask)
    f0, h0 = provider.r0_for_indices(indices)
    expected_f0, expected_h0 = _FakeR0().predict_numpy(
        source["hlt"]["tokens"][indices], source["hlt"]["mask"][indices]
    )
    np.testing.assert_array_equal(f0, expected_f0)
    np.testing.assert_array_equal(h0, expected_h0)
    wrong_map = provider.matched_wrong_event_map([0, 1, 2, 3], seed=11)
    controlled = provider.control_shard(
        0,
        control_type="event_shuffled_delta",
        seed=11,
        wrong_event_map=wrong_map,
    )
    assert controlled["fields"].shape == (4, 5, 50)
    assert np.isfinite(controlled["fields"]).all()
    assert provider.telemetry()["h0_cached"] is False
    assert set(stager.persistent_npz_open_counts.values()) == {1}
    second_stager = AllocationNpzStager(ledger, rank=0, world_size=2)
    with pytest.raises(RuntimeError, match="refusing a persistent reopen"):
        second_stager.stage_pair(
            hlt_npz=paths["hlt"][0],
            hlt_metadata=paths["hlt"][1],
            offline_npz=paths["offline"][0],
            offline_metadata=paths["offline"][1],
            shard_size=4,
        )
    assert second_stager.persistent_npz_open_counts == {}
    provider.close()
    ledger.cleanup()


def test_named_pair_staging_opens_each_parent_source_once_and_finalizes_together(tmp_path):
    model_root = tmp_path / "model"
    stack_root = tmp_path / "stack"
    model_root.mkdir()
    stack_root.mkdir()
    model_paths, model_arrays = _write_source_pair(model_root, n=7)
    stack_paths, stack_arrays = _write_source_pair(stack_root, n=9)
    ledger = AllocationRamLedger(
        tmp_path / "ram_named",
        allocation_id="named",
        capacity_bytes=64 * 1024 * 1024,
        allow_unverified_test_root=True,
    )
    stager = AllocationNpzStager(ledger, rank=0, world_size=1)
    staged, report = stager.stage_named_pairs(
        {
            "model_train": {
                "hlt_npz": model_paths["hlt"][0],
                "hlt_metadata": model_paths["hlt"][1],
                "offline_npz": model_paths["offline"][0],
                "offline_metadata": model_paths["offline"][1],
            },
            "stack_train": {
                "hlt_npz": stack_paths["hlt"][0],
                "hlt_metadata": stack_paths["hlt"][1],
                "offline_npz": stack_paths["offline"][0],
                "offline_metadata": stack_paths["offline"][1],
            },
        },
        shard_size=3,
    )
    assert list(staged) == ["model_train", "stack_train"]
    assert report["source_namespaces"] == ["model_train", "stack_train"]
    assert report["all_persistent_npz_open_counts_equal_one"] is True
    assert len(report["persistent_npz_open_counts"]) == 4
    assert ledger.snapshot()["raw_stage_finalized"] is True
    assert len(stager.raw_reservation_ids) == 2
    np.testing.assert_array_equal(
        staged["model_train"][0].read_indices([6, 0])["tokens"],
        model_arrays["hlt"]["tokens"][[6, 0]],
    )
    np.testing.assert_array_equal(
        staged["stack_train"][1].read_indices([8, 1])["tokens"],
        stack_arrays["offline"]["tokens"][[8, 1]],
    )
    with pytest.raises(RuntimeError, match="refusing a persistent reopen"):
        AllocationNpzStager(ledger).stage_named_pairs(
            {
                "again": {
                    "hlt_npz": model_paths["hlt"][0],
                    "hlt_metadata": model_paths["hlt"][1],
                    "offline_npz": model_paths["offline"][0],
                    "offline_metadata": model_paths["offline"][1],
                }
            }
        )
    ledger.cleanup()


def test_derived_lru_regenerates_without_touching_raw(tmp_path):
    ledger = AllocationRamLedger(
        tmp_path,
        allocation_id="lru",
        capacity_bytes=10_000,
        allow_unverified_test_root=True,
    )
    calls = []

    def generate(key):
        calls.append(key)
        return {"value": np.full(100, int(key), dtype=np.float32)}

    cache = DerivedShardLRU(ledger=ledger, owner="rank0", capacity_bytes=500, generator=generate)
    cache.get(1)
    cache.get(2)
    cache.get(1)
    telemetry = cache.telemetry()
    assert calls == [1, 2, 1]
    assert telemetry["evictions"] == 2
    assert telemetry["regenerations"] == 1
    assert telemetry["raw_shards_evictable"] is False
    assert telemetry["persistent_field_tensors_written"] is False
    cache.close()
    ledger.cleanup()


def _field_fixture(n=6, p=4):
    rng = np.random.default_rng(4)
    mask = np.ones((n, p), dtype=bool)
    mask[0, -1] = False
    f0 = rng.normal(size=(n, p, 50)).astype(np.float32)
    truth = f0 + rng.normal(scale=0.5, size=f0.shape).astype(np.float32)
    f0[~mask] = 0
    truth[~mask] = 0
    return f0, truth, mask


def test_virtual_bridge_endpoints_response_and_recipe():
    f0, truth, mask = _field_fixture()
    at_zero = virtual_bridge(f0, truth, mask, rho="0", channel_policy=BRIDGE_CHANNEL_PHYSICAL45)
    at_one = virtual_bridge(f0, truth, mask, rho="1", channel_policy=BRIDGE_CHANNEL_PHYSICAL45)
    np.testing.assert_array_equal(at_zero, f0)
    np.testing.assert_array_equal(at_one[..., :45], truth[..., :45])
    np.testing.assert_array_equal(at_one[..., 45:], f0[..., 45:])
    all_one = virtual_bridge(f0, truth, mask, rho="1", channel_policy=BRIDGE_CHANNEL_ALL50)
    np.testing.assert_array_equal(all_one, truth)
    response = bridge_response(f0, truth, mask)
    assert tuple(response) == ("0.000", "0.025", "0.050", "0.075", "0.100")
    recipe = build_bridge_recipe(
        rho="0.10",
        channel_policy="physical45",
        r0_checkpoint_sha256="a" * 64,
        hlt_source_sha256="b" * 64,
        offline_source_sha256="c" * 64,
        split_manifest_sha256="d" * 64,
        target_schema_sha256="e" * 64,
        preprocessing_sha256="f" * 64,
        event_order_sha256="1" * 64,
    )
    validate_bridge_recipe(recipe)
    assert recipe["rho_decimal"] == "0.100"
    assert recipe["dense_bridge_artifact"] is False


def test_matched_derangement_and_all_five_controls():
    tokens, mask = _tokens(16, 4)
    labels = np.repeat([0, 1], 8)
    event_ids = [f"event-{index}" for index in range(16)]
    mapping = build_matched_wrong_event_map(
        tokens=tokens,
        mask=mask,
        labels=labels,
        event_ids=event_ids,
        seed=17,
        logical_block_size=16,
    )
    permutation = np.asarray(mapping["permutation"])
    assert np.all(permutation != np.arange(16))
    np.testing.assert_array_equal(labels, labels[permutation])
    f0, truth, field_mask = _field_fixture(n=16, p=4)
    event_control = apply_bridge_control(
        f0,
        truth,
        field_mask,
        control_type="event_shuffled_delta",
        seed=17,
        event_ids=event_ids,
        wrong_event_map=mapping,
    )
    expected = f0.copy()
    expected[..., :45] += 0.1 * (truth - f0)[permutation, ..., :45]
    expected[~field_mask] = 0
    # Donor padding is also zeroed; this fixture has a single recipient/donor
    # mismatch, so compare their common valid support.
    common = field_mask & field_mask[permutation]
    np.testing.assert_allclose(event_control[..., :45][common], expected[..., :45][common], atol=1e-6)
    particle = apply_bridge_control(
        f0, truth, field_mask, control_type="particle_shuffled_delta", seed=4, event_ids=event_ids
    )
    sign = apply_bridge_control(f0, truth, field_mask, control_type="sign_reversed_delta", seed=4)
    random = apply_bridge_control(f0, truth, field_mask, control_type="same_norm_random_delta", seed=4)
    radius = apply_bridge_control(f0, truth, field_mask, control_type="radius_group_permuted_delta", seed=4)
    assert all(np.isfinite(value).all() for value in (particle, sign, random, radius))
    np.testing.assert_allclose(sign[..., :45], (f0 - 0.1 * (truth - f0))[..., :45], atol=1e-6)
    true_delta = 0.1 * (truth - f0)
    random_delta = random - f0
    for indices in physical_loss_groups().values():
        np.testing.assert_allclose(
            np.linalg.norm(random_delta[..., indices][field_mask]),
            np.linalg.norm(true_delta[..., indices][field_mask]),
            rtol=1e-5,
            atol=1e-6,
        )
    np.testing.assert_array_equal(radius[..., 45:], f0[..., 45:])


def test_scaler_sparse_fallback_inactive_ordering_and_inverse():
    n = 101
    f0 = np.zeros((n, 1, 50), dtype=np.float32)
    f0[:, 0, 0] = np.linspace(-2, 3, n)
    truth = f0.copy()
    truth[-1, 0, 0] += 10.0
    truth[:, 0, 2] += 2.0
    truth[-1, 0, 49] += 5.0
    mask = np.ones((n, 1), dtype=bool)
    scalers = fit_bridge_scalers(
        [(f0, truth, mask)],
        parent_hashes={"source": "a" * 64, "r0": "b" * 64},
        channel_policy=BRIDGE_CHANNEL_PHYSICAL45,
    )
    assert scalers.sparse_nonzero_fallback[0]
    assert scalers.q99_delta[0] == pytest.approx(1.0)
    assert scalers.sigma_delta[0] == pytest.approx(1.0)
    assert scalers.trust_scale[0] == pytest.approx(2.0)
    assert not scalers.active[1]
    assert scalers.q99_delta[1] == 0
    assert scalers.sigma_delta[1] == scalers.epsilon[1]
    assert not scalers.active[49]
    standardized = scalers.conditioning_standardize(f0, mask)
    restored = scalers.conditioning_inverse(standardized, mask)
    np.testing.assert_allclose(restored, f0, atol=2e-7)
    raw = np.ones_like(f0)
    bounded = scalers.bounded_physical_correction(raw, mask)
    assert np.all(bounded[..., 1] == 0)
    assert np.all(bounded[..., 45:] == 0)
    corrected = scalers.corrected_physical_fields(f0, raw, mask)
    np.testing.assert_array_equal(corrected[..., 45:], f0[..., 45:])
    with pytest.raises(ValueError, match="physical-space"):
        scalers.corrected_physical_fields(f0, raw, mask, input_space="standardized")
    artifact = scalers.to_artifact()
    loaded = BridgeScalers.from_artifact(artifact)
    np.testing.assert_array_equal(loaded.active, scalers.active)


def _small_checkpoint(path: Path) -> None:
    import torch

    config = LocalResidualFieldReconstructorConfig(
        particle_dim=14,
        field_dim=50,
        d_model=10,
        num_heads=2,
        num_layers=1,
        context_layers=1,
        max_particles=8,
    )
    model = build_local_residual_field_reconstructor(config)
    torch.save(
        {
            "checkpoint_contract": "test_r0",
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "metrics": {"mae": 1.0},
        },
        path,
    )


def test_streamed_r0_training_publication_and_frozen_determinism(tmp_path):
    tokens, mask = _tokens(4, 3)
    targets = np.zeros((4, 3, 50), dtype=np.float32)
    targets[..., 0] = tokens[..., 0] * 0.1
    targets[~mask] = 0

    def batches(_epoch):
        return iter(
            [
                {
                    "hlt_tokens": tokens,
                    "hlt_mask": mask,
                    "target_fields": targets,
                    "target_mask": mask,
                }
            ]
        )

    output = tmp_path / "r0"
    report = train_streamed_r0(
        StreamedR0TrainConfig(
            output_dir=str(output),
            epochs=1,
            device="cpu",
            d_model=10,
            num_heads=2,
            num_layers=1,
            context_layers=1,
            early_stop_patience=-1,
        ),
        train_batches=batches,
        model_val_stop_batches=batches,
        provenance_hashes={
            "preprocessing_sha256": "a" * 64,
            "target_schema_sha256": "b" * 64,
            "split_manifest_sha256": "c" * 64,
        },
        matching_policy={"query": "hlt_particles", "target": "offline_local_summary"},
    )
    assert report["persistent_artifacts"] == [
        "r0_metrics.json",
        "r0_registration.json",
        "r0_weights.pt",
    ]
    assert "optimizer_state_dict" not in __import__("torch").load(
        output / "r0_weights.pt", map_location="cpu", weights_only=False
    )
    runner = FrozenR0Runner(output / "r0_weights.pt", device="cpu")
    first = runner.predict_numpy(tokens, mask)
    second = runner.predict_numpy(tokens, mask)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.all(first[0][~mask] == 0)
    assert np.all(first[1][~mask] == 0)


def test_production_audit_persists_only_small_contracts(tmp_path):
    from scripts.audit_prediction_anchored_bridge_inputs import main

    paths, _ = _write_source_pair(tmp_path, n=6, p=3)
    checkpoint = tmp_path / "r0.pt"
    _small_checkpoint(checkpoint)
    output = tmp_path / "published"
    code = main(
        [
            "--hlt-npz", str(paths["hlt"][0]),
            "--hlt-metadata", str(paths["hlt"][1]),
            "--offline-npz", str(paths["offline"][0]),
            "--offline-metadata", str(paths["offline"][1]),
            "--r0-checkpoint", str(checkpoint),
            "--ram-root", str(tmp_path / "audit_ram"),
            "--allocation-id", "smoke",
            "--output-dir", str(output),
            "--split-manifest-sha256", "d" * 64,
            "--max-fit-jets", "4",
            "--fit-batch-size", "4",
            "--shard-size", "4",
            "--derived-capacity-bytes", str(128 * 1024),
            "--test-capacity-bytes", str(32 * 1024 * 1024),
            "--allow-unverified-test-root",
        ]
    )
    assert code == 0
    assert sorted(path.name for path in output.iterdir()) == [
        "bridge_recipe_all50.json",
        "bridge_recipe_physical45.json",
        "bridge_scalers_all50.json",
        "bridge_scalers_physical45.json",
        "step2_audit_metrics.json",
    ]
    assert not list(output.glob("*.npy")) and not list(output.glob("*.npz"))
    metrics = json.loads((output / "step2_audit_metrics.json").read_text())
    assert metrics["persistent_field_tensors_written"] is False
    assert set(metrics["raw_stage"]["persistent_npz_open_counts"].values()) == {1}
    assert not (tmp_path / "audit_ram" / "prediction_anchored_bridge_smoke").exists()

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.hlt_offline_structure_distillation import (
    ExtractorResources,
    LoadedTargetCache,
    TeacherInferenceAdapter,
    align_conditional_context_to_cache,
    apply_conditional_residual,
    apply_target_shuffle,
    build_heteroscedastic_metadata,
    build_hlt_conditional_context,
    build_storage_measurements,
    build_stage_b_wave_completion,
    build_target_cache_spec,
    build_target_shuffle_plan,
    build_teacher_lock,
    build_teacher_output_manifest,
    build_teacher_training_manifest,
    complete_teacher_training,
    fit_conditional_residual,
    fit_latent_ridge_adapter,
    fit_latent_whitening,
    fit_streamed_target_normalizer,
    fit_target_normalizer,
    extract_registered_target,
    infer_teacher_batch,
    iter_authenticated_target_shard_layouts,
    load_target_cache,
    load_target_cache_sharded,
    normalize_target,
    publish_target_cache,
    publish_target_cache_shard,
    stage_b_wave_rows,
    target_mean_values,
    validate_target_cache,
    validate_target_normalizer,
    validate_teacher_lock,
    whiten_latents,
)
from teacher_logit_reco.hlt_offline_structure_distillation.extractors import (
    PHYSICAL_TARGET_IDS,
    TargetBatch,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    STRUCTURE_TARGET_REGISTRY_CONTRACT,
    TARGET_CACHE_MANIFEST_CONTRACT,
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.teachers import (
    training_protocol,
)
from teacher_logit_reco.relational_part import TrainingConfig
from scripts.execute_hosd_scale_graph import _teacher_logits_npz
from scripts.train_hosd_baseline import _privileged


H = "a" * 64
SOURCE = {
    "commit": "b" * 40,
    "status_sha256": "c" * 64,
    "dirty": True,
    "status_hash_policy": "test",
}


def test_stage_b_wave_coordinates_and_completion_are_exhaustive():
    registry = with_content_hash(
        {
            "contract": STRUCTURE_TARGET_REGISTRY_CONTRACT,
            "schema_version": 1,
            "source": SOURCE,
            "targets": [
                {
                    "target_id": target_id,
                    "executable_current_source": True,
                }
                for target_id in PHYSICAL_TARGET_IDS
            ],
        }
    )
    rows = stage_b_wave_rows(
        wave_kind="hlt_analogue", target_registry=registry
    )
    assert len(rows) == 6
    assert {
        (row["split"], row["replica"]) for row in rows
    } == {
        *(("model_train", replica) for replica in range(4)),
        ("val_stop", 0),
        ("val_design", 0),
    }
    manifests = []
    for row in rows:
        manifests.append(
            with_content_hash(
                {
                    "contract": TARGET_CACHE_MANIFEST_CONTRACT,
                    "schema_version": 1,
                    "source": SOURCE,
                    "split": row["split"],
                    "hlt_replica_id": str(row["replica"]),
                    "artifact_kind": "hlt_analogue",
                    "persisted_target_ids": list(row["target_ids"]),
                    "streamed_target_ids": [],
                }
            )
        )
    completion = build_stage_b_wave_completion(
        wave_kind="hlt_analogue",
        target_registry=registry,
        manifests=manifests,
        source=SOURCE,
    )
    assert completion["row_count"] == 6
    assert completion["exact_coordinate_coverage"]
    with pytest.raises(ValueError, match="count"):
        build_stage_b_wave_completion(
            wave_kind="hlt_analogue",
            target_registry=registry,
            manifests=manifests[:-1],
            source=SOURCE,
        )


def _generator(source_values: np.ndarray):
    def generate(indices: np.ndarray):
        selected = source_values[indices]
        values = np.stack([selected, selected * 2], axis=1).astype(np.float32)
        masks = np.ones_like(values, dtype=bool)
        masks[selected == 3, 1] = False
        values[~masks] = 0
        return {
            "T_OFFLINE_TEST": TargetBatch(
                target_id="T_OFFLINE_TEST",
                component_names=("a", "b"),
                availability_groups=("target_available",),
                values=torch.from_numpy(values),
                loss_mask=torch.from_numpy(masks),
                diagnostics={},
            )
        }

    return generate


def _cache(tmp_path: Path, name: str = "cache") -> tuple[dict, LoadedTargetCache]:
    identities = ["jet-c", "jet-a", "jet-e", "jet-b", "jet-d"]
    values = np.asarray([3, 1, 5, 2, 4], dtype=np.float32)
    spec = build_target_cache_spec(
        cache_id=name,
        split="model_train",
        artifact_kind="canonical_offline",
        identities=identities,
        target_components={"T_OFFLINE_TEST": ("a", "b")},
        parent_hashes={"campaign": H, "split": "d" * 64, "registry": "e" * 64},
        source=SOURCE,
        shard_size=2,
    )
    publish_target_cache(
        tmp_path / name,
        cache_spec=spec,
        identities=identities,
        generator=_generator(values),
        shard_order=(2, 0, 1),
    )
    return spec, load_target_cache(tmp_path / name, cache_spec=spec)


def test_cache_is_identity_sorted_label_blind_byte_deterministic_and_resumable(
    tmp_path: Path,
) -> None:
    identities = ["jet-c", "jet-a", "jet-e", "jet-b", "jet-d"]
    values = np.asarray([3, 1, 5, 2, 4], dtype=np.float32)
    spec = build_target_cache_spec(
        cache_id="canonical",
        split="model_train",
        artifact_kind="canonical_offline",
        identities=identities,
        target_components={"T_OFFLINE_TEST": ("a", "b")},
        parent_hashes={"campaign": H},
        source=SOURCE,
        shard_size=2,
    )
    left = publish_target_cache(
        tmp_path / "left",
        cache_spec=spec,
        identities=identities,
        generator=_generator(values),
        shard_order=(2, 0, 1),
    )
    right = publish_target_cache(
        tmp_path / "right",
        cache_spec=spec,
        identities=identities,
        generator=_generator(values),
        shard_order=(0, 1, 2),
    )
    assert left == right
    assert left["complete_exact_identity_coverage"]
    assert not left["labels_stored"]
    loaded = load_target_cache(tmp_path / "left", cache_spec=spec)
    assert loaded.identities == tuple(sorted(identities))
    assert loaded.values["T_OFFLINE_TEST"][:, 0].tolist() == [1, 2, 3, 4, 5]
    for index in range(3):
        a = tmp_path / "left" / "shards" / f"shard_{index:06d}.npz"
        b = tmp_path / "right" / "shards" / f"shard_{index:06d}.npz"
        assert a.read_bytes() == b.read_bytes()
    resumed = publish_target_cache(
        tmp_path / "left",
        cache_spec=spec,
        identities=identities,
        generator=_generator(values),
    )
    assert resumed == left


def test_target_layout_iterator_rejects_shard_byte_drift(tmp_path: Path) -> None:
    spec, _ = _cache(tmp_path, "layout")
    rows = list(iter_authenticated_target_shard_layouts(tmp_path / "layout"))
    assert len(rows) == int(spec["shard_count"])
    shard = tmp_path / "layout" / "shards" / "shard_000000.npz"
    shard.write_bytes(shard.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="attestation differs"):
        list(iter_authenticated_target_shard_layouts(tmp_path / "layout"))


def test_canonical_mmap_shards_publish_without_population_sort_or_index_copy(
    tmp_path: Path,
) -> None:
    identities = np.asarray(["jet-a", "jet-b", "jet-c", "jet-d"])
    values = np.arange(1, 5, dtype=np.float32)
    spec = build_target_cache_spec(
        cache_id="canonical-mmap",
        split="scale_train",
        artifact_kind="canonical_offline",
        identities=identities,
        identities_are_canonical=True,
        target_components={"T_OFFLINE_TEST": ("a", "b")},
        parent_hashes={"campaign": H},
        source=SOURCE,
        shard_size=2,
    )
    for shard_index in range(spec["shard_count"]):
        publish_target_cache_shard(
            tmp_path / "canonical-mmap",
            cache_spec=spec,
            canonical_identities=identities,
            canonical_to_source=None,
            shard_index=shard_index,
            generator=_generator(values),
            identity_population_attestation=spec[
                "canonical_identity_order_sha256"
            ],
        )
    manifest = validate_target_cache(
        tmp_path / "canonical-mmap", cache_spec=spec
    )
    assert manifest["event_count"] == 4
    with pytest.raises(ValueError, match="unique, and sorted"):
        build_target_cache_spec(
            cache_id="not-canonical",
            split="scale_train",
            artifact_kind="canonical_offline",
            identities=np.asarray(["jet-b", "jet-a"]),
            identities_are_canonical=True,
            target_components={"T_OFFLINE_TEST": ("a", "b")},
            parent_hashes={"campaign": H},
            source=SOURCE,
        )


def test_shard_publication_identity_work_is_linear_and_reuse_is_constant(
    tmp_path: Path,
) -> None:
    class CountingSequence:
        def __init__(self, values):
            self.values = tuple(values)
            self.accesses = 0

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            if isinstance(index, slice):
                selected = self.values[index]
                self.accesses += len(selected)
                return selected
            if index >= len(self.values):
                raise IndexError(index)
            self.accesses += 1
            return self.values[index]

    count, shard_size = 10_001, 257
    identities = CountingSequence(f"jet-{index:05d}" for index in range(count))
    source_values = np.arange(count, dtype=np.float32)
    spec = build_target_cache_spec(
        cache_id="linear-identities",
        split="scale_train",
        artifact_kind="teacher_output",
        identities=identities,
        target_components={"T_OFFLINE_TEST": ("a", "b")},
        parent_hashes={"campaign": H},
        source=SOURCE,
        shard_size=shard_size,
        identities_are_canonical=True,
    )
    identities.accesses = 0
    for shard_index in range(int(spec["shard_count"])):
        publish_target_cache_shard(
            tmp_path / "linear",
            cache_spec=spec,
            canonical_identities=identities,
            canonical_to_source=None,
            shard_index=shard_index,
            generator=_generator(source_values),
            identity_population_attestation=spec[
                "canonical_identity_order_sha256"
            ],
        )
    assert identities.accesses <= 3 * count
    identities.accesses = 0
    for shard_index in range(int(spec["shard_count"])):
        publish_target_cache_shard(
            tmp_path / "linear",
            cache_spec=spec,
            canonical_identities=identities,
            canonical_to_source=None,
            shard_index=shard_index,
            generator=_generator(source_values),
            identity_population_attestation=spec[
                "canonical_identity_order_sha256"
            ],
        )
    assert identities.accesses == 0


def test_canonical_identity_attestation_avoids_population_materialization() -> None:
    class AttestedPopulation:
        def __len__(self):
            return 3_000_000

        def __iter__(self):
            raise AssertionError("attested canonical identities were iterated")

        def __getitem__(self, index):
            raise AssertionError("attested canonical identities were indexed")

    spec = build_target_cache_spec(
        cache_id="attested-scale-residual",
        split="scale_train",
        artifact_kind="residual",
        identities=AttestedPopulation(),
        target_components={"T_OFFLINE_TEST__RES__0": ("a", "b")},
        parent_hashes={"campaign": H},
        source=SOURCE,
        hlt_replica_id="0",
        identities_are_canonical=True,
        canonical_identity_order_attestation="f" * 64,
    )
    assert spec["event_count"] == 3_000_000
    assert spec["canonical_identity_order_sha256"] == "f" * 64


def test_sharded_cache_keeps_only_one_identity_shard_resident(tmp_path: Path) -> None:
    spec, _ = _cache(tmp_path, "lazy-identities")
    loaded = load_target_cache_sharded(tmp_path / "lazy-identities", cache_spec=spec)
    assert not isinstance(loaded.identities, tuple)
    assert not hasattr(loaded.identities.store, "identities")
    assert loaded.identities[:2] == ("jet-a", "jet-b")
    assert len(loaded.identities.store._cached_identities) <= int(spec["shard_size"])
    assert loaded.identities[-1] == "jet-e"
    assert len(loaded.identities.store._cached_identities) <= int(spec["shard_size"])
    target_array_names = set(
        loaded.identities.store.records[
            loaded.identities.store._cached_index
        ]["arrays"].values()
    )
    assert all(
        loaded.identities.store._cached_arrays[name].shape[0]
        <= int(spec["shard_size"])
        for name in target_array_names
    )


def test_scale_kd_logits_stream_to_authenticated_mmap(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    teacher_id = "O_BASE"
    cache_root = root / "scale_up" / "teacher_outputs" / teacher_id
    identities = tuple(f"jet-{index:05d}" for index in range(25))
    expected = np.arange(250, dtype=np.float32).reshape(25, 10)
    coordinate = f"T_OFFLINE_LOGITS_{teacher_id}"
    spec = build_target_cache_spec(
        cache_id="scale-teacher-logits",
        split="scale_train",
        artifact_kind="teacher_output",
        identities=identities,
        target_components={coordinate: tuple(f"class_{index}" for index in range(10))},
        parent_hashes={"campaign": H},
        source=SOURCE,
        shard_size=4,
        identities_are_canonical=True,
    )

    def generate(indices):
        values = expected[indices]
        return {
            coordinate: TargetBatch(
                target_id=coordinate,
                component_names=tuple(f"class_{index}" for index in range(10)),
                availability_groups=("teacher_logits_available",),
                values=torch.from_numpy(values),
                loss_mask=torch.ones_like(torch.from_numpy(values), dtype=torch.bool),
                diagnostics={},
            )
        }

    for shard_index in range(int(spec["shard_count"])):
        publish_target_cache_shard(
            cache_root,
            cache_spec=spec,
            canonical_identities=identities,
            canonical_to_source=None,
            shard_index=shard_index,
            generator=generate,
            identity_population_attestation=spec["canonical_identity_order_sha256"],
        )
    manifest = validate_target_cache(cache_root, cache_spec=spec)
    from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
        write_immutable_json,
    )

    write_immutable_json(cache_root / "cache_spec.json", spec)
    write_immutable_json(cache_root / "target_manifest.json", manifest)
    labels_path = root / "scale_labels.npz"
    np.savez_compressed(
        labels_path,
        identities=np.asarray(identities),
        labels=np.arange(25, dtype=np.int64) % 10,
    )
    output = _teacher_logits_npz(root, teacher_id, labels_path)
    loaded = _privileged(output, identities, "logits")
    assert isinstance(loaded, np.memmap)
    assert np.array_equal(loaded, expected)
    assert _teacher_logits_npz(root, teacher_id, labels_path) == output


def test_real_physical_extractor_miniature_cache_passes_lineage_and_resume(
    tmp_path: Path,
) -> None:
    raw = np.zeros((4, 5, 14), dtype=np.float32)
    mask = np.zeros((4, 5), dtype=bool)
    for row, count in enumerate((5, 3, 2, 1)):
        mask[row, :count] = True
        raw[row, :count, 0] = np.arange(1, count + 1, dtype=np.float32)
        raw[row, :count, 1] = np.linspace(-0.2, 0.2, count)
        raw[row, :count, 2] = np.linspace(-1.0, 1.0, count)
        raw[row, :count, 3] = raw[row, :count, 0] * np.cosh(
            raw[row, :count, 1]
        )
        raw[row, :count, 5] = 1
        raw[row, :count, 11] = 0.01
        raw[row, :count, 13] = 0.02
    identities = ("jet-3", "jet-1", "jet-4", "jet-2")
    resources = ExtractorResources(0.001, 0.002)
    first = extract_registered_target(
        "T_OFFLINE_JET_10", raw[:1], mask[:1], resources=resources
    )
    spec = build_target_cache_spec(
        cache_id="physical-miniature",
        split="model_train",
        artifact_kind="canonical_offline",
        identities=identities,
        target_components={
            "T_OFFLINE_JET_10": first.component_names,
        },
        parent_hashes={"campaign": H, "extractor": "9" * 64},
        source=SOURCE,
        shard_size=2,
    )

    def generate(indices):
        return {
            "T_OFFLINE_JET_10": extract_registered_target(
                "T_OFFLINE_JET_10",
                raw[indices],
                mask[indices],
                resources=resources,
            )
        }

    original = publish_target_cache(
        tmp_path / "physical",
        cache_spec=spec,
        identities=identities,
        generator=generate,
        shard_order=(1, 0),
    )
    resumed = publish_target_cache(
        tmp_path / "physical",
        cache_spec=spec,
        identities=identities,
        generator=generate,
    )
    assert resumed == original
    assert original["event_count"] == 4


def test_cache_rejects_postlock_without_authorization_and_corruption(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="access_authorization"):
        build_target_cache_spec(
            cache_id="late",
            split="stack_val",
            artifact_kind="canonical_offline",
            identities=("jet",),
            target_components={"T": ("x",)},
            parent_hashes={"campaign": H},
            source=SOURCE,
        )
    spec, _ = _cache(tmp_path)
    shard = tmp_path / "cache" / "shards" / "shard_000000.npz"
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="bytes"):
        validate_target_cache(tmp_path / "cache", cache_spec=spec)


def test_train_only_normalizer_mean_and_heteroscedastic_contract(tmp_path: Path) -> None:
    _, cache = _cache(tmp_path)
    artifact = fit_target_normalizer(
        cache,
        fitting_population="target_500k",
        source=SOURCE,
        component_kinds={"T_OFFLINE_TEST": ("continuous", "binary")},
    )
    normalized = normalize_target(
        cache.values["T_OFFLINE_TEST"],
        cache.masks["T_OFFLINE_TEST"],
        target_id="T_OFFLINE_TEST",
        normalizer=artifact,
    )
    assert np.all(normalized[~cache.masks["T_OFFLINE_TEST"]] == 0)
    assert np.array_equal(
        normalized[:, 1][cache.masks["T_OFFLINE_TEST"][:, 1]],
        cache.values["T_OFFLINE_TEST"][:, 1][cache.masks["T_OFFLINE_TEST"][:, 1]],
    )
    means = target_mean_values(
        cache.masks["T_OFFLINE_TEST"],
        target_id="T_OFFLINE_TEST",
        normalizer=artifact,
    )
    assert np.all(means[~cache.masks["T_OFFLINE_TEST"]] == 0)
    hetero = build_heteroscedastic_metadata(artifact, source=SOURCE)
    assert hetero["log_variance_clip"] == [-8.0, 5.0]
    val_cache = LoadedTargetCache(
        identities=cache.identities,
        values=cache.values,
        masks=cache.masks,
        manifest={**cache.manifest, "split": "val_stop"},
    )
    with pytest.raises(ValueError, match="must be fit"):
        fit_target_normalizer(
            val_cache, fitting_population="target_500k", source=SOURCE
        )


def test_streamed_pair_normalizer_is_sample_bound_and_never_persists_pairs() -> None:
    artifact = fit_streamed_target_normalizer(
        target_id="T_HLT_REGION_PAIR_8",
        component_names=("same_k2", "merge_depth"),
        component_kinds=("binary", "continuous"),
        component_samples=(
            np.asarray([0, 1, 1, 0], dtype=np.float64),
            np.asarray([0.1, 0.2, 0.4, 0.8], dtype=np.float64),
        ),
        fitting_population="target_scale",
        split="scale_train",
        selected_jet_count=2,
        selected_jet_identity_sha256="d" * 64,
        sampled_pair_identity_sha256="e" * 64,
        parent_hashes={"scale_hlt_views": "f" * 64},
        source=SOURCE,
    )
    assert validate_target_normalizer(artifact) == artifact["content_hash"]
    assert artifact["dense_pair_target_persisted"] is False
    assert artifact["sampling_contract"]["pair_limit_per_jet"] == 64
    assert artifact["targets"][0]["components"][0]["normalize"] is False
    values = np.asarray([[1.0, 0.4]], dtype=np.float32)
    masks = np.ones_like(values, dtype=bool)
    transformed = normalize_target(
        values,
        masks,
        target_id="T_HLT_REGION_PAIR_8",
        normalizer=artifact,
    )
    assert transformed[0, 0] == 1.0
    with pytest.raises(ValueError, match="must be fit"):
        fit_streamed_target_normalizer(
            target_id="T",
            component_names=("x",),
            component_kinds=("continuous",),
            component_samples=(np.asarray([1.0]),),
            fitting_population="target_scale",
            split="model_train",
            selected_jet_count=1,
            selected_jet_identity_sha256="d" * 64,
            sampled_pair_identity_sha256="e" * 64,
            parent_hashes={"view": "f" * 64},
            source=SOURCE,
        )


def test_conditional_residual_backoff_is_deterministic_and_identity_free() -> None:
    context = np.asarray(
        [[index, index % 3, index % 4, (index % 5) / 4] for index in range(40)],
        dtype=np.float64,
    )
    values = np.stack([np.arange(40), -np.arange(40)], axis=1).astype(np.float64)
    masks = np.ones_like(values, dtype=bool)
    masks[:20, 1] = False
    artifact = fit_conditional_residual(
        values,
        masks,
        context,
        target_id="T",
        train_cache_hashes={"offline": H, "hlt": "d" * 64},
        source=SOURCE,
    )
    assert not artifact["identity_values_stored"]
    assert not artifact["event_values_stored"]
    assert artifact["bin_counts"] == [8, 4, 4, 4]
    output = apply_conditional_residual(context[:3], artifact=artifact)
    assert output.shape == (3, 2)
    assert np.isfinite(output).all()


def test_conditional_context_alignment_uses_compact_identity_permutation() -> None:
    context = np.asarray(
        [[20.0, 21.0, 22.0, 23.0], [10.0, 11.0, 12.0, 13.0]],
        dtype=np.float64,
    )
    aligned = align_conditional_context_to_cache(
        ("jet-b", "jet-a"),
        context,
        ("jet-a", "jet-b"),
    )
    assert np.array_equal(aligned, context[[1, 0]])
    with pytest.raises(ValueError, match="identity coverage differs"):
        align_conditional_context_to_cache(
            ("jet-b", "jet-a"),
            context,
            ("jet-a", "jet-c"),
        )


def test_conditional_context_uses_vector_jet_and_registered_track_validity() -> None:
    raw = np.zeros((1, 2, 14), dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    raw[0, :, 0] = [3.0, 4.0]
    raw[0, :, 1] = [0.0, 0.0]
    raw[0, :, 2] = [0.0, np.pi / 2]
    raw[0, :, 3] = [3.0, 4.0]
    raw[0, :, 5] = 1.0
    raw[0, 0, 11] = raw[0, 0, 13] = 0.01
    # Particle one has zero errors and is not a valid track.
    context = build_hlt_conditional_context(
        raw,
        mask,
        d0_uncertainty_floor=0.001,
        dz_uncertainty_floor=0.001,
        sentinel_policy=None,
    )
    assert context[0, 0] == pytest.approx(np.log(5.0))
    assert context[0, 1] == pytest.approx(0.0)
    assert context[0, 2] == 2
    assert context[0, 3] == pytest.approx(0.5)


def test_global_and_within_class_shuffle_are_distinct_and_immutable() -> None:
    identities = tuple(f"jet-{index:02d}" for index in range(8))
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    global_plan = build_target_shuffle_plan(
        identities,
        labels=labels,
        target_id="T",
        split="model_train",
        shuffle_kind="global",
        label_manifest_sha256=H,
        canonical_cache_manifest_sha256="d" * 64,
        source=SOURCE,
    )
    class_plan = build_target_shuffle_plan(
        identities,
        labels=labels,
        target_id="T",
        split="model_train",
        shuffle_kind="within_class",
        label_manifest_sha256=H,
        canonical_cache_manifest_sha256="d" * 64,
        source=SOURCE,
    )
    assert global_plan["content_hash"] != class_plan["content_hash"]
    mapping = np.asarray(class_plan["mapping_recipient_to_donor"])
    assert np.array_equal(np.asarray(labels), np.asarray(labels)[mapping])
    values = np.arange(16, dtype=np.float32).reshape(8, 2)
    masks = np.ones_like(values, dtype=bool)
    original = values.copy()
    shuffled, _ = apply_target_shuffle(values, masks, plan=global_plan)
    assert np.array_equal(values, original)
    assert not np.array_equal(shuffled, original)


def test_whitening_and_ridge_are_deterministic() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(300, 128))
    whitening_a = fit_latent_whitening(
        values,
        teacher_lock_sha256=H,
        fitting_population="target_500k",
        source=SOURCE,
    )
    whitening_b = fit_latent_whitening(
        values,
        teacher_lock_sha256=H,
        fitting_population="target_500k",
        source=SOURCE,
    )
    assert whitening_a == whitening_b
    whitened = whiten_latents(values, whitening=whitening_a)
    assert whitened.shape == values.shape
    ridge = fit_latent_ridge_adapter(
        whitened,
        values,
        whitening_sha256=whitening_a["content_hash"],
        teacher_lock_sha256=H,
        source=SOURCE,
    )
    assert ridge["lambda"] == 1.0e-4
    assert not ridge["labels_used"]


class _Teacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 10)
        self.latent = torch.empty(0)

    def forward(self, batch):
        raw = batch["x"].float()
        self.latent = raw.mean(dim=1, keepdim=True).expand(-1, 128).contiguous()
        return self.linear(raw)


def test_teacher_production_lock_and_fp32_inference(tmp_path: Path) -> None:
    training = build_teacher_training_manifest(
        campaign_spec_sha256=H,
        split_manifest_sha256="d" * 64,
        model_contract_hashes={"O_BASE": "e" * 64, "O_FULLREL": "f" * 64},
        normalizer_hashes={
            "O_BASE": {"input": "1" * 64},
            "O_FULLREL": {"input": "2" * 64, "relation": "3" * 64},
        },
        source=SOURCE,
    )
    completions = {}
    for teacher_id in ("O_BASE", "O_FULLREL"):
        checkpoint = tmp_path / f"{teacher_id}.pt"
        checkpoint.write_bytes(f"checkpoint-{teacher_id}".encode())
        completions[teacher_id] = complete_teacher_training(
            training,
            teacher_id=teacher_id,
            checkpoint_path=checkpoint,
            selector_trace={"selected_epoch": 4},
            architecture={"teacher_id": teacher_id},
            source=SOURCE,
        )
    lock = build_teacher_lock(training, completions=completions, source=SOURCE)
    validate_teacher_lock(lock, source=SOURCE)
    model = _Teacher()
    model.train()
    adapter = TeacherInferenceAdapter(
        teacher_id="O_BASE",
        model=model,
        forward=model,
        pooled_latent_tap=lambda: model.latent,
        tap_contract="exact_normalized_preclassifier_o_base_pooled_representation",
    )
    output = infer_teacher_batch(adapter, {"x": torch.randn(3, 4)})
    assert model.training
    assert output["T_OFFLINE_LOGITS_O_BASE"].values.dtype == torch.float32
    assert output["T_OFFLINE_POOLED_LATENT"].values.shape == (3, 128)
    manifest = build_teacher_output_manifest(
        teacher_lock=lock,
        cache_manifest_hashes_by_split={
            split: {
                "T_OFFLINE_LOGITS_O_BASE": "4" * 64,
                "T_OFFLINE_LOGITS_O_FULLREL": "5" * 64,
                "T_OFFLINE_POOLED_LATENT": "6" * 64,
            }
            for split in ("model_train", "val_stop", "val_design")
        },
        source=SOURCE,
    )
    assert manifest["teacher_lock_sha256"] == lock["content_hash"]
    assert manifest["split_order"] == [
        "model_train",
        "val_stop",
        "val_design",
    ]
    (tmp_path / "O_BASE.pt").write_bytes(b"drift")
    with pytest.raises(ValueError, match="drifted"):
        validate_teacher_lock(lock, source=SOURCE)


def test_teacher_training_protocol_is_exact_40_epoch_and_exact_selector() -> None:
    protocol = training_protocol()
    assert protocol["maximum_epochs"] == protocol["minimum_epochs"] == 40
    assert protocol["early_stop_before_epoch_40"] is False
    assert protocol["checkpoint_selection"][0].startswith("exact_maximum")
    config = TrainingConfig(
        seed=101,
        minimum_epochs=40,
        accuracy_window=0.0,
        campaign_profile="hosd_teacher",
    )
    config.validate()
    artifact = config.artifact(global_determinism_sha256=H)
    assert artifact["checkpoint_selector"].startswith("exact_max")


def test_storage_projection_is_complete_and_performance_blind() -> None:
    artifact = build_storage_measurements(
        family_measurements={
            "JET": {
                "storage_class": "compact_jet",
                "sample_events": 100,
                "bytes_written": 1000,
                "elapsed_seconds": 2,
                "valid_components": 800,
                "total_components": 1000,
                "maximum_shard_rebuild_seconds": 4,
            },
            "PAIR": {
                "storage_class": "same_view_pair",
                "sample_events": 100,
                "bytes_written": 100_000,
                "elapsed_seconds": 10,
                "valid_components": 50,
                "total_components": 1000,
            },
        },
        available_storage_bytes=1_000_000,
        parent_hashes={"campaign": H},
        source=SOURCE,
    )
    assert set(artifact["projected_storage_bytes"]) == {
        "production_500k",
        "scale_3m",
    }
    assert artifact["decision_uses_scientific_results"] is False
    assert artifact["projection_exceeds_available_storage"]
    assert artifact["pair_target_bytes_excluded_from_persistent_projection"]
    assert artifact["measured_streamed_dense_bytes_per_jet"] == 1000.0
    assert artifact["persistent_cache_multiplicity"] == 9


def test_real_storage_probe_is_label_blind_and_measures_extractor_bytes(
    tmp_path,
) -> None:
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "measure_hosd_storage.py"
    )
    spec = importlib.util.spec_from_file_location("measure_hosd_storage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw_tokens = np.zeros((2, 4, 14), dtype=np.float32)
    raw_tokens[:, :, 2] = 1.0
    mask = np.ones((2, 4), dtype=bool)
    probe = tmp_path / "probe.npz"
    np.savez(
        probe,
        identity=np.asarray(["a", "b"]),
        raw_tokens=raw_tokens,
        mask=mask,
    )
    identities, selected, selected_mask, vectors = module._load_probe_input(
        probe
    )
    measured = module._measure_families(
        target_ids=("T_OFFLINE_JET_10",),
        raw_tokens=selected,
        mask=selected_mask,
        vectors=vectors,
        resources=ExtractorResources(
            d0_uncertainty_floor=0.0,
            dz_uncertainty_floor=0.0,
        ),
    )
    assert identities == ("a", "b")
    assert measured["T_OFFLINE_JET_10"]["bytes_written"] > 0
    assert measured["T_OFFLINE_JET_10"]["storage_class"] == "compact_jet"

    np.savez(
        tmp_path / "labeled.npz",
        identity=np.asarray(["a"]),
        raw_tokens=raw_tokens[:1],
        mask=mask[:1],
        labels=np.asarray([0]),
    )
    with pytest.raises(ValueError, match="label blind"):
        module._load_probe_input(tmp_path / "labeled.npz")

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from teacher_logit_reco.adaptive_binary_pseudooffline.bundled_scoring import (
    ABPH_LOGIT_ONLY_ARRAY_NAMES,
    encode_logit_only_npz,
    group_scoring_members,
    scoring_source_family,
    source_generation_hash,
    validate_logit_only_npz,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    ABPH_BUNDLED_SCORING_FAMILIES,
    ABPH_FUSION_CANDIDATES,
    AdaptiveBinarySubmissionConfig,
    build_submission_graph,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    StorageProjectionRow,
    build_storage_projection,
    write_storage_projection,
)


def _projection(root: Path, path: Path) -> None:
    payload = build_storage_projection(
        campaign_root=root,
        campaign_mode="pilot",
        profile=ABPH_STREAMING_STORAGE_PROFILE,
        rows=(
            StorageProjectionRow(
                artifact_family="test",
                artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
                expected_bytes=1_000_000,
                active_from_wave=0,
                active_through_wave=6,
                retained=True,
                atomic_write_overhead_bytes=1_000,
                measurement_source="test",
            ),
        ),
        measurement_contract="test",
        sample_provenance_hash="test",
    )
    write_storage_projection(path, payload)


def test_members_group_only_when_their_pseudo_source_is_identical() -> None:
    members = (
        "E5_kt32_mh4_dualcross",
        "F4_ce_logit_kd",
        "E6_ca32_mh4_dualcross",
        "E7_dual_hierarchy_dualcross",
        "F0_ce_reco_primary",
        "F0_ce_reco_primary__seed2",
    )
    grouped = group_scoring_members(members)
    assert grouped["d1_exclusive_kt"] == (
        "E5_kt32_mh4_dualcross",
        "F4_ce_logit_kd",
    )
    assert grouped["d2_cambridge_aachen"] == ("E6_ca32_mh4_dualcross",)
    assert grouped["shared_root_dual"] == ("E7_dual_hierarchy_dualcross",)
    assert scoring_source_family("F0_ce_reco_primary").key != scoring_source_family(
        "F0_ce_reco_primary__seed2"
    ).key


def test_logit_only_codec_rejects_extra_pseudo_arrays(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"
    path.write_bytes(
        encode_logit_only_npz(
            logits=np.ones((5, 10), dtype=np.float64),
            labels=np.arange(5, dtype=np.int64),
            jet_ids=np.asarray([f"jet:{index}" for index in range(5)]),
            source_indices=np.arange(5, dtype=np.int64),
        )
    )
    report = validate_logit_only_npz(path)
    assert report["array_names"] == list(ABPH_LOGIT_ONLY_ARRAY_NAMES)
    with np.load(path, allow_pickle=False) as payload:
        assert payload["logits"].dtype == np.float32

    np.savez_compressed(
        path,
        logits=np.ones((5, 10), dtype=np.float32),
        labels=np.arange(5, dtype=np.int64),
        jet_ids=np.asarray([f"jet:{index}" for index in range(5)]),
        source_indices=np.arange(5, dtype=np.int64),
        pseudo_particles=np.ones((5, 128, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="non-logit payload"):
        validate_logit_only_npz(path)


def test_source_generation_hash_binds_split_checkpoint_and_identity() -> None:
    family = scoring_source_family("E5_kt32_mh4_dualcross")
    common = {
        "family": family,
        "split": "model_val",
        "hlt_content_hash": "hlt",
        "jet_identity_hash": "identities",
        "generator_source_hash": "generator",
        "checkpoint_hashes": {"D1": "checkpoint"},
        "consumer_schema_hashes": ("schema",),
    }
    baseline = source_generation_hash(**common)
    assert baseline != source_generation_hash(
        **{**common, "split": "stack_train"}
    )
    assert baseline != source_generation_hash(
        **{**common, "checkpoint_hashes": {"D1": "changed"}}
    )
    assert baseline != source_generation_hash(
        **{**common, "jet_identity_hash": "changed"}
    )


def test_streaming_graph_queues_source_families_and_not_per_member(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    projection = tmp_path / "projection.json"
    _projection(root, projection)
    graph = build_submission_graph(
        AdaptiveBinarySubmissionConfig(
            campaign_root=root,
            data_dir=tmp_path / "data",
            reconstructor_parallelism="single",
            allow_debug_single_reconstructor=True,
            storage_profile=ABPH_STREAMING_STORAGE_PROFILE,
            storage_projection_path=projection,
        )
    )
    jobs = {job.key: job for job in graph}
    bundle_keys = {key for key in jobs if key.startswith("logit_bundle:")}
    assert bundle_keys == {
        f"logit_bundle:{name}" for name in ABPH_BUNDLED_SCORING_FAMILIES
    }
    assert not any(key.startswith("logit_prediction:") for key in jobs)
    d1 = jobs["logit_bundle:d1_exclusive_kt"]
    assert d1.executable == "run_adaptive_binary_bundled_scoring.sh"
    assert d1.arguments == (
        "E5_kt32_mh4_dualcross",
        "F4_ce_logit_kd",
    )
    for fusion_name, members in ABPH_FUSION_CANDIDATES.items():
        fusion = jobs[f"variant:{fusion_name}"]
        expected = {
            f"logit_bundle:{scoring_source_family(member).key}"
            for member in members
        }
        assert set(fusion.dependencies) == expected

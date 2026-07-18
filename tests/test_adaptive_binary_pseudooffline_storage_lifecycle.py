from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from teacher_logit_reco.adaptive_binary_pseudooffline.bundled_scoring import (
    encode_logit_only_npz,
    persisted_identity_hash,
    validate_logit_only_npz,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    ABPH_JOINT_TARGET_TAGGER_MEMBERS,
    ABPH_LOGIT_PREDICTION_MEMBERS,
    ABPH_PRIVILEGED_CONSUMERS,
    AdaptiveBinarySubmissionConfig,
    build_submission_graph,
    require_partial_stage_inputs,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_lifecycle import (
    artifact_manifest_path,
    build_artifact_manifest,
    cleanup_receipt_path,
    execute_cleanup_barrier,
    require_cleanup_receipt,
    write_consumer_receipt,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    StorageProjectionRow,
    build_storage_projection,
    initialize_storage_accounting,
    write_storage_projection,
)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
                measurement_source="test",
            ),
        ),
        measurement_contract="test",
        sample_provenance_hash="test",
    )
    write_storage_projection(path, payload)


def _campaign_inputs(root: Path) -> None:
    hlt = root / "inputs" / "hlt_cache"
    for index, split in enumerate(
        ("model_train", "model_val", "stack_train", "stack_val", "final_test")
    ):
        hlt.mkdir(parents=True, exist_ok=True)
        (hlt / f"{split}_fixed_hlt.npz").write_bytes(f"hlt-{split}".encode())
        _json(hlt / f"{split}_fixed_hlt_metadata.json", {"split": split})
    offline = root / "inputs" / "offline_cache"
    for split in ("model_train", "model_val"):
        offline.mkdir(parents=True, exist_ok=True)
        (offline / f"{split}_offline.npz").write_bytes(
            f"offline-{split}".encode()
        )
        _json(offline / f"{split}_offline_metadata.json", {"split": split})
    _json(
        root / "audits" / "target_mode_selection.json",
        {"selected_mode": "rank_local_build", "content_hash": "mode"},
    )


def _logits(root: Path, member: str) -> None:
    identities = np.asarray(["file:0:0", "file:1:1"], dtype=np.str_)
    for split in ("model_val", "stack_train", "stack_val"):
        path = root / "logit_predictions" / member / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            encode_logit_only_npz(
                logits=np.ones((2, 10), dtype=np.float32),
                labels=np.asarray([0, 1], dtype=np.int64),
                jet_ids=identities,
                source_indices=np.asarray([0, 1], dtype=np.int64),
            )
        )
        artifact = validate_logit_only_npz(path)
        _json(
            path.with_name(f"{split}_metadata.json"),
            {
                "ok": True,
                "prediction_sha256": artifact["sha256"],
                "provenance": {
                    "persisted_identity_hash": persisted_identity_hash(identities),
                    "source_generation_hash": "source",
                },
            },
        )


def _target_consumer_report(path: Path, name: str = "B1") -> None:
    _json(
        path,
        {
            "ok": True,
            "variant_name": name,
            "provenance": {
                "model_val": {
                    "source_manifest_hash": "manifest",
                    "offline_cache_content_hash": "offline",
                    "hierarchy_target_content_hash": "targets",
                }
            },
        },
    )
def test_exact_cleanup_requires_receipts_and_retains_final_test_hlt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    initialize_storage_accounting(root, profile=ABPH_STREAMING_STORAGE_PROFILE)
    _campaign_inputs(root)
    manifest = build_artifact_manifest(
        root, data_dir=tmp_path / "data"
    )
    assert manifest["target_mode"] == "rank_local_build"
    report = root / "runs" / "B1" / "run_report.json"
    _target_consumer_report(report)
    write_consumer_receipt(root, consumer="B1", run_report=report)
    receipt = execute_cleanup_barrier(
        root, barrier="privileged", expected_consumers=("B1",)
    )
    assert receipt["ok"] is True
    assert not (root / "inputs" / "offline_cache" / "model_train_offline.npz").exists()
    assert (root / "inputs" / "hlt_cache" / "model_train_fixed_hlt.npz").is_file()

    _logits(root, "member")
    deployable = execute_cleanup_barrier(
        root, barrier="deployable", scoring_members=("member",)
    )
    assert deployable["ok"] is True
    assert not (root / "inputs" / "hlt_cache" / "model_train_fixed_hlt.npz").exists()
    assert (root / "inputs" / "hlt_cache" / "final_test_fixed_hlt.npz").is_file()
    assert require_cleanup_receipt(root, "privileged")["ok"] is True
    assert require_cleanup_receipt(root, "deployable")["ok"] is True


def test_cleanup_refuses_changed_manifest_artifact(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    initialize_storage_accounting(root, profile=ABPH_STREAMING_STORAGE_PROFILE)
    _campaign_inputs(root)
    build_artifact_manifest(root, data_dir=tmp_path / "data")
    report = root / "runs" / "B1" / "run_report.json"
    _target_consumer_report(report)
    write_consumer_receipt(root, consumer="B1", run_report=report)
    (root / "inputs" / "offline_cache" / "model_train_offline.npz").write_bytes(
        b"tampered"
    )
    with pytest.raises(ValueError, match="hash changed"):
        execute_cleanup_barrier(
            root, barrier="privileged", expected_consumers=("B1",)
        )
    assert not cleanup_receipt_path(root, "privileged").exists()


def test_cleanup_intent_allows_retry_after_partial_exact_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    initialize_storage_accounting(root, profile=ABPH_STREAMING_STORAGE_PROFILE)
    _campaign_inputs(root)
    build_artifact_manifest(root, data_dir=tmp_path / "data")
    report = root / "runs" / "B1" / "run_report.json"
    _target_consumer_report(report)
    write_consumer_receipt(root, consumer="B1", run_report=report)
    original_unlink = Path.unlink
    injected = {"done": False}

    def fail_after_first_delete(path: Path, *args, **kwargs):
        result = original_unlink(path, *args, **kwargs)
        if path.name == "model_train_offline.npz" and not injected["done"]:
            injected["done"] = True
            raise RuntimeError("simulated node loss")
        return result

    monkeypatch.setattr(Path, "unlink", fail_after_first_delete)
    with pytest.raises(RuntimeError, match="simulated node loss"):
        execute_cleanup_barrier(
            root, barrier="privileged", expected_consumers=("B1",)
        )
    monkeypatch.setattr(Path, "unlink", original_unlink)
    receipt = execute_cleanup_barrier(
        root, barrier="privileged", expected_consumers=("B1",)
    )
    assert receipt["ok"] is True
    assert not (root / "inputs" / "offline_cache" / "model_train_offline.npz").exists()
    assert not (root / "inputs" / "offline_cache" / "model_val_offline.npz").exists()


def test_streaming_graph_has_all_waves_receipts_and_cleanup_barriers(
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
    assert {f"wave:{index}" for index in range(7)}.issubset(jobs)
    assert set(jobs["barrier:privileged_cleanup"].arguments[1:]) == set(
        ABPH_PRIVILEGED_CONSUMERS
    )
    assert set(jobs["barrier:deployable_cleanup"].arguments[1:]) == set(
        ABPH_LOGIT_PREDICTION_MEMBERS
    )
    for member in ABPH_JOINT_TARGET_TAGGER_MEMBERS:
        assert f"receipt:{member}" in jobs
    assert jobs["report:model_selection"].dependencies == ("wave:5",)
    assert jobs["wave:6"].dependencies == ("report:model_selection",)


def test_models_reuse_fails_after_privileged_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    projection = tmp_path / "projection.json"
    _projection(root, projection)
    initialize_storage_accounting(root, profile=ABPH_STREAMING_STORAGE_PROFILE)
    _campaign_inputs(root)
    build_artifact_manifest(root, data_dir=tmp_path / "data")
    _json(
        cleanup_receipt_path(root, "privileged"),
        {"contract": "placeholder"},
    )
    config = AdaptiveBinarySubmissionConfig(
        campaign_root=root,
        data_dir=tmp_path / "data",
        stage_mode="models",
        rebuild_inputs=False,
        rebuild_targets=False,
        rebuild_models=True,
        reconstructor_parallelism="single",
        allow_debug_single_reconstructor=True,
        storage_profile=ABPH_STREAMING_STORAGE_PROFILE,
        storage_projection_path=projection,
    )
    with pytest.raises(PermissionError, match="after privileged inputs were cleaned"):
        require_partial_stage_inputs(config)


def test_streaming_tigris_wrapper_locks_profile_and_account() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "sbatch"
        / "submit_adaptive_binary_pseudooffline_streaming30gb_tigris.sh"
    ).read_text(encoding="utf-8")
    assert "ABPH_SBATCH_ACCOUNT:=reu-aisocial" in source
    assert "ABPH_STORAGE_PROFILE:=streaming_30gb_v1" in source
    assert "ABPH_STORAGE_PROJECTION_PATH:?" in source
    assert "ABPH_RUNTIME_ACCEPTANCE_PATH:?" in source

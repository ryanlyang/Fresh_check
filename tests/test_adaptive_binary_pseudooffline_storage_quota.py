from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    AdaptiveBinarySubmissionConfig,
    build_submission_graph,
    submission_manifest,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (
    ABPH_MAX_PERSISTENT_BYTES,
    ABPH_MAX_PROJECTED_PEAK_BYTES,
    ABPH_STREAMING_STORAGE_CONTRACT,
    ABPH_STREAMING_STORAGE_PROFILE,
    ABPH_TARGET_STEADY_STATE_BYTES,
    StorageArtifactClass,
    StorageProfile,
    StorageProjectionRow,
    build_storage_projection,
    campaign_storage_audit,
    cleanup_quota_managed_artifact,
    initialize_storage_accounting,
    publish_quota_managed_file,
    release_persistent_reservation,
    require_storage_projection,
    reserve_persistent_artifact,
    storage_paths,
    write_quota_managed_bytes,
    write_storage_projection,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_quota_managed_file_publication_streams_ephemeral_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    profile = _small_profile()
    initialize_storage_accounting(root, profile=profile)
    source = tmp_path / "tmpfs-simulation" / "model_train_fixed_hlt.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"cache-payload" * 100)
    destination = root / "inputs" / "hlt_cache" / source.name
    receipt = publish_quota_managed_file(
        root,
        source,
        destination,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="input_hlt_cache",
        source_provenance_hash="source-hash",
        run_id="hlt-cache-test",
        profile=profile,
        copy_buffer_bytes=17,
    )
    assert destination.read_bytes() == source.read_bytes()
    assert receipt["actual_bytes"] == source.stat().st_size
    ledger = json.loads(storage_paths(root)["ledger"].read_text(encoding="utf-8"))
    assert not ledger["active_reservations"]
    assert str(destination.resolve()) in ledger["committed_artifacts"]


def test_quota_managed_cleanup_requires_exact_role_and_hash(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    profile = _small_profile()
    initialize_storage_accounting(root, profile=profile)
    source = tmp_path / "ram" / "payload.bin"
    source.parent.mkdir()
    source.write_bytes(b"persistent-boundary-smoke")
    destination = root / "storage" / "lifecycle_smoke_payload" / "payload.bin"
    receipt = publish_quota_managed_file(
        root,
        source,
        destination,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="ram_lifecycle_smoke_payload",
        source_provenance_hash="smoke-source",
        run_id="smoke",
        profile=profile,
    )
    with pytest.raises(ValueError, match="role mismatch"):
        cleanup_quota_managed_artifact(
            root,
            destination,
            expected_sha256=receipt["sha256"],
            expected_artifact_role="wrong-role",
            profile=profile,
        )
    with pytest.raises(ValueError, match="ledger hash mismatch"):
        cleanup_quota_managed_artifact(
            root,
            destination,
            expected_sha256="wrong-hash",
            expected_artifact_role="ram_lifecycle_smoke_payload",
            profile=profile,
        )
    cleanup = cleanup_quota_managed_artifact(
        root,
        destination,
        expected_sha256=receipt["sha256"],
        expected_artifact_role="ram_lifecycle_smoke_payload",
        profile=profile,
    )
    assert cleanup["destination_removed"] is True
    assert cleanup["ledger_reconciled"] is True
    ledger = json.loads(storage_paths(root)["ledger"].read_text(encoding="utf-8"))
    assert str(destination.resolve()) not in ledger["committed_artifacts"]


def _projection(root: Path, path: Path, *, oversized: bool = False) -> dict:
    expected = ABPH_MAX_PROJECTED_PEAK_BYTES + 1 if oversized else 4_000_000
    payload = build_storage_projection(
        campaign_root=root,
        campaign_mode="pilot",
        profile=ABPH_STREAMING_STORAGE_PROFILE,
        rows=(
            StorageProjectionRow(
                artifact_family="input_and_selected_artifacts",
                artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
                expected_bytes=expected,
                active_from_wave=0,
                active_through_wave=6,
                retained=True,
                atomic_write_overhead_bytes=1_000_000,
                measurement_source="deterministic_real_sample_v1",
            ),
        ),
        measurement_contract="production_codec_measurement_v1",
        sample_provenance_hash="sample-hash",
    )
    write_storage_projection(path, payload)
    return payload


def _small_profile() -> StorageProfile:
    return StorageProfile(
        name="test_220kb",
        implementation_contract="test_storage_v1",
        enforce_quota=True,
        max_persistent_bytes=220_000,
        max_projected_peak_bytes=180_000,
        target_steady_state_bytes=150_000,
    )


def test_streaming_profile_has_locked_30gb_limits() -> None:
    assert ABPH_MAX_PERSISTENT_BYTES == 30_000_000_000
    assert ABPH_MAX_PROJECTED_PEAK_BYTES == 24_000_000_000
    assert ABPH_TARGET_STEADY_STATE_BYTES == 20_000_000_000


def test_projection_is_wave_aware_and_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    path = tmp_path / "projection.json"
    payload = _projection(root, path)
    assert payload["ok"] is True
    assert payload["projected_peak_persistent_bytes"] == 5_000_000
    assert payload["projected_final_retained_bytes"] == 4_000_000
    assert require_storage_projection(
        path,
        campaign_root=root,
        campaign_mode="pilot",
        profile=ABPH_STREAMING_STORAGE_PROFILE,
    )["content_hash"] == payload["content_hash"]

    oversized_path = tmp_path / "oversized.json"
    oversized = _projection(root, oversized_path, oversized=True)
    assert oversized["ok"] is False
    with pytest.raises(ValueError, match="projected peak"):
        require_storage_projection(
            oversized_path,
            campaign_root=root,
            campaign_mode="pilot",
            profile=ABPH_STREAMING_STORAGE_PROFILE,
        )

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["projected_peak_persistent_bytes"] = 1
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        require_storage_projection(
            path,
            campaign_root=root,
            campaign_mode="pilot",
            profile=ABPH_STREAMING_STORAGE_PROFILE,
        )


def test_initialization_creates_contract_ledger_and_exact_layout(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    profile = _small_profile()
    result = initialize_storage_accounting(root, profile=profile)
    paths = storage_paths(root)
    assert result["dry_run"] is False
    assert paths["contract"].is_file()
    assert paths["ledger"].is_file()
    assert paths["lock"].is_file()
    assert paths["reservations"].is_dir()
    assert paths["cleanup_receipts"].is_dir()
    assert paths["audits"].is_dir()
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    assert contract["profile"]["max_persistent_bytes"] == 220_000


def test_dry_run_makes_no_directory_or_reservation(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    profile = _small_profile()
    preview = initialize_storage_accounting(root, profile=profile, dry_run=True)
    reservation = reserve_persistent_artifact(
        root,
        destination=root / "runs" / "model.pt",
        expected_bytes=100,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="selected_checkpoint",
        source_provenance_hash="source",
        run_id="B1",
        profile=profile,
        dry_run=True,
    )
    assert preview["dry_run"] is True
    assert reservation["status"] == "dry_run"
    assert not root.exists()


def test_atomic_quota_managed_write_commits_hash_and_audits_bytes(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    profile = _small_profile()
    initialize_storage_accounting(root, profile=profile)
    destination = root / "runs" / "B1" / "selected.pt"
    receipt = write_quota_managed_bytes(
        root,
        destination,
        b"selected-weights",
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="selected_checkpoint",
        source_provenance_hash="checkpoint-source",
        run_id="B1",
        profile=profile,
    )
    assert receipt["status"] == "committed"
    assert receipt["actual_bytes"] == len(b"selected-weights")
    assert destination.read_bytes() == b"selected-weights"
    ledger = json.loads(storage_paths(root)["ledger"].read_text(encoding="utf-8"))
    assert ledger["active_reservations"] == {}
    assert str(destination.resolve()) in ledger["committed_artifacts"]
    audit = campaign_storage_audit(root, profile=profile)
    assert audit["ok"] is True
    assert audit["measured_persistent_bytes"] == sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )


def test_concurrent_reservations_cannot_cross_hard_cap(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    profile = _small_profile()
    initialize_storage_accounting(root, profile=profile)
    barrier = threading.Barrier(2)
    accepted: list[dict] = []
    rejected: list[BaseException] = []

    def reserve(index: int) -> None:
        barrier.wait()
        try:
            accepted.append(
                reserve_persistent_artifact(
                    root,
                    destination=root / "runs" / f"model-{index}.pt",
                    expected_bytes=90_000,
                    artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
                    artifact_role="selected_checkpoint",
                    source_provenance_hash=f"source-{index}",
                    run_id=f"B{index}",
                    profile=profile,
                )
            )
        except BaseException as exc:
            rejected.append(exc)

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert "reservation rejected" in str(rejected[0])
    release_persistent_reservation(
        root,
        accepted[0]["reservation_id"],
        profile=profile,
    )


def test_persistent_reservation_rejects_escape_and_forbidden_classes(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    profile = _small_profile()
    initialize_storage_accounting(root, profile=profile)
    common = {
        "expected_bytes": 1,
        "artifact_role": "debug_dump",
        "source_provenance_hash": "source",
        "run_id": "debug",
        "profile": profile,
    }
    with pytest.raises(ValueError, match="escapes campaign root"):
        reserve_persistent_artifact(
            root,
            destination=tmp_path / "outside.pt",
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            **common,
        )
    with pytest.raises(PermissionError, match="cannot be persisted"):
        reserve_persistent_artifact(
            root,
            destination=root / "debug.pt",
            artifact_class=StorageArtifactClass.FORBIDDEN_PERSISTENT,
            **common,
        )


def test_streaming_config_binds_projection_into_graph_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    projection_path = tmp_path / "projection.json"
    projection = _projection(root, projection_path)
    config = AdaptiveBinarySubmissionConfig(
        campaign_root=root,
        data_dir=tmp_path / "data",
        reconstructor_parallelism="single",
        allow_debug_single_reconstructor=True,
        storage_profile=ABPH_STREAMING_STORAGE_PROFILE,
        storage_projection_path=projection_path,
    )
    graph = build_submission_graph(config)
    jobs = {job.key: job for job in graph}
    assert jobs["target:mode_preflight"].dependencies == ("wave:1",)
    assert jobs["target:cache"].dependencies == ("target:mode_preflight",)
    assert all(
        job.environment["ABPH_TARGET_MODE_REPORT"]
        == str(root / "audits" / "target_mode_selection.json")
        for job in graph
    )
    assert all(
        job.environment["ABPH_STORAGE_PROFILE"] == ABPH_STREAMING_STORAGE_PROFILE
        for job in graph
    )
    assert all(
        job.environment["ABPH_RAM_STAGE_RESERVATION_BYTES"]
        == str(projection["projected_peak_persistent_bytes"])
        for job in graph
    )
    assert not any(job.stage == "pseudo_prediction" for job in graph)
    e5 = jobs["variant:E5_kt32_mh4_dualcross"]
    e7 = jobs["variant:E7_dual_hierarchy_dualcross"]
    assert "variant:D1_kt32_mh4_particles" in e5.dependencies
    assert "variant:D1_kt32_mh4_particles" in e7.dependencies
    assert "variant:D2_ca32_mh4_particles" in e7.dependencies
    assert not any(
        dependency.startswith("prediction:")
        for job in graph
        for dependency in job.dependencies
    )
    manifest = submission_manifest(config, graph)
    assert manifest["storage_profile"] == ABPH_STREAMING_STORAGE_PROFILE
    assert manifest["storage_projection"]["content_hash"] == projection["content_hash"]


def test_streaming_submitter_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    projection_path = tmp_path / "projection.json"
    _projection(root, projection_path)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "submit_adaptive_binary_pseudooffline.py"),
            "--campaign-root",
            str(root),
            "--data-dir",
            str(tmp_path / "data"),
            "--campaign-mode",
            "pilot",
            "--stage-mode",
            "full",
            "--cluster",
            "tigris",
            "--reconstructor-parallelism",
            "single",
            "--allow-debug-single-reconstructor",
            "--storage-profile",
            ABPH_STREAMING_STORAGE_PROFILE,
            "--storage-projection",
            str(projection_path),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["storage_accounting"]["profile"]["implementation_contract"] == (
        ABPH_STREAMING_STORAGE_CONTRACT
    )
    for row in payload["submission_commands"]:
        assert "--output=/dev/null" in row["command"]
        assert "--error=/dev/null" in row["command"]
        assert row["environment"]["MIRROR_DIAGNOSTICS"] == "0"
        assert row["environment"]["LOG_DIR"].startswith("/dev/shm/")
        assert row["environment"]["DIAGNOSTICS_ROOT"].startswith("/dev/shm/")
    assert not root.exists()


def test_streaming_config_rejects_missing_or_oversized_projection(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="measured storage projection"):
        AdaptiveBinarySubmissionConfig(
            campaign_root=tmp_path / "missing",
            data_dir=tmp_path / "data",
            reconstructor_parallelism="single",
            allow_debug_single_reconstructor=True,
            storage_profile=ABPH_STREAMING_STORAGE_PROFILE,
        )
    root = tmp_path / "oversized"
    projection_path = tmp_path / "oversized.json"
    _projection(root, projection_path, oversized=True)
    with pytest.raises(ValueError, match="projected peak"):
        AdaptiveBinarySubmissionConfig(
            campaign_root=root,
            data_dir=tmp_path / "data",
            reconstructor_parallelism="single",
            allow_debug_single_reconstructor=True,
            storage_profile=ABPH_STREAMING_STORAGE_PROFILE,
            storage_projection_path=projection_path,
        )

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
    publish_quota_managed_file,
    storage_paths,
    write_storage_projection,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.ram_workspace import (
    RamCapacityProbe,
    RankLocalWorkspace,
)
from scripts.run_adaptive_binary_ram_lifecycle_smoke import (
    SMOKE_ARTIFACT_ROLE,
    main as run_ram_lifecycle_smoke,
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
        _json(
            hlt / f"{split}_fixed_hlt_metadata.json",
            {
                "split": split,
                "source_manifest_hash": "manifest",
                "hlt_content_hash": f"hlt-hash-{split}",
                "jet_identity_hash": f"identity-{split}",
                "label_hash": f"labels-{split}",
            },
        )
    offline = root / "inputs" / "offline_cache"
    for split in ("model_train", "model_val"):
        offline.mkdir(parents=True, exist_ok=True)
        (offline / f"{split}_offline.npz").write_bytes(
            f"offline-{split}".encode()
        )
        _json(
            offline / f"{split}_offline_metadata.json",
            {
                "split": split,
                "source_manifest_hash": "manifest",
                "offline_content_hash": f"offline-hash-{split}",
                "jet_identity_hash": f"identity-{split}",
                "label_hash": f"labels-{split}",
            },
        )
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
                    "hlt_content_hash": "hlt-hash-model_val",
                    "offline_cache_content_hash": "offline-hash-model_val",
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


def test_consumer_receipt_is_bound_to_frozen_hlt_source(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    initialize_storage_accounting(root, profile=ABPH_STREAMING_STORAGE_PROFILE)
    _campaign_inputs(root)
    build_artifact_manifest(root, data_dir=tmp_path / "data")
    report = root / "runs" / "B1" / "run_report.json"
    _target_consumer_report(report)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["provenance"]["model_val"]["hlt_content_hash"] = "other-hlt-cache"
    _json(report, payload)
    with pytest.raises(ValueError, match="HLT cache hash mismatch"):
        write_consumer_receipt(root, consumer="B1", run_report=report)


def test_privileged_cleanup_refuses_empty_consumer_membership(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    initialize_storage_accounting(root, profile=ABPH_STREAMING_STORAGE_PROFILE)
    _campaign_inputs(root)
    build_artifact_manifest(root, data_dir=tmp_path / "data")
    with pytest.raises(ValueError, match="requires frozen consumer membership"):
        execute_cleanup_barrier(root, barrier="privileged")


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
    assert jobs["wave:6"].dependencies == ("wave:5",)
    assert jobs["report:model_selection"].dependencies == ("wave:6",)


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


def test_material_streaming_writers_stage_then_quota_publish() -> None:
    root = Path(__file__).resolve().parents[1]
    inputs = (root / "sbatch" / "run_adaptive_binary_inputs.sh").read_text(
        encoding="utf-8"
    )
    targets = (root / "sbatch" / "run_adaptive_binary_targets.sh").read_text(
        encoding="utf-8"
    )
    report = (root / "sbatch" / "run_adaptive_binary_report.sh").read_text(
        encoding="utf-8"
    )
    for source in (inputs, targets, report):
        assert "abph_setup_ram_workspace" in source
        assert "publish_adaptive_binary_quota_tree.py" in source
    for source in (inputs, targets):
        assert "abph_reserve_ram_workspace" in source
        assert "abph_commit_ram_workspace" in source
        assert "abph_release_ram_workspace" in source
    assert 'hlt_cache_dir="${ABPH_RAM_WORKSPACE}/hlt_cache"' in inputs
    assert 'offline_cache_dir="${ABPH_RAM_WORKSPACE}/offline_cache"' in inputs
    assert 'target_cache_dir="${ABPH_RAM_WORKSPACE}/targets"' in targets
    assert 'staged_report="${ABPH_RAM_WORKSPACE}/' in report


def test_storage_acceptance_runs_real_ram_lifecycle_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "sbatch" / "run_adaptive_binary_storage_acceptance.sh").read_text(
        encoding="utf-8"
    )
    smoke = (root / "scripts" / "run_adaptive_binary_ram_lifecycle_smoke.py").read_text(
        encoding="utf-8"
    )
    assert ': "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"' in worker
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in worker
    assert '${SCRIPT_DIR}/common.sh' not in worker
    assert 'CONDA_BASE="${ABPH_CONDA_BASE:-/home/ryreu/miniforge3-aarch64}"' in worker
    assert 'CONDA_ENV="${ABPH_CONDA_ENV:-atlas_kd_tigris}"' in worker
    assert worker.index("CONDA_BASE=") < worker.index(
        'source "${PROJECT_DIR}/sbatch/common.sh"'
    )
    assert "fresh_setup\n" in worker
    assert "fresh_setup_job" not in worker
    assert "fresh_require_conda_python_package pytest" in worker
    assert "ram_lifecycle_smoke)" in worker
    ordered_actions = (
        "abph_reserve_ram_workspace",
        "run_adaptive_binary_ram_lifecycle_smoke.py prepare",
        "abph_commit_ram_workspace",
        "publish_adaptive_binary_quota_tree.py",
        "abph_release_ram_workspace",
        "run_adaptive_binary_ram_lifecycle_smoke.py verify",
    )
    positions = [worker.index(action) for action in ordered_actions]
    assert positions == sorted(positions)
    assert '--campaign-root "${ABPH_ROOT}"' in worker
    assert '--destination-dir "${ABPH_ROOT}/storage/lifecycle_smoke_payload"' in worker
    assert "workspace_filesystem_required" in smoke
    assert "cleanup_quota_managed_artifact" in smoke
    acceptance = (
        root
        / "teacher_logit_reco"
        / "adaptive_binary_pseudooffline"
        / "storage_acceptance.py"
    ).read_text(encoding="utf-8")
    assert "ABPH_RAM_LIFECYCLE_SMOKE_CONTRACT" in acceptance
    assert '"ram_lifecycle_smoke"' in acceptance


def test_storage_acceptance_repair_reanchors_cached_slurm_graph(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    from scripts.repair_adaptive_binary_storage_acceptance_graph import (
        TIGRIS_CONDA_BASE,
        TIGRIS_CONDA_ENV,
        _job_environment,
        _remaining_dependencies,
        _source_identity,
        repaired_sbatch_command,
    )

    command = repaired_sbatch_command(
        {
            "key": "acceptance:component_parity_tests",
            "command": [
                "sbatch",
                "--parsable",
                "--job-name=abph_storage_acceptance_tests",
                "--output=/dev/null",
                "--error=/dev/null",
                "--dependency=afterok:10",
                "/repo/sbatch/run_adaptive_binary_storage_acceptance.sh",
                "tests",
            ],
        },
        label="storage_tests",
        dependencies=("20",),
        log_dir=tmp_path,
    )
    assert "--job-name=abph_repair_storage_tests" in command
    assert f"--output={tmp_path}/abph_repair_storage_tests_%j.out" in command
    assert f"--error={tmp_path}/abph_repair_storage_tests_%j.err" in command
    assert "--dependency=afterok:20" in command
    assert "--dependency=afterok:10" not in command
    assert "--output=/dev/null" not in command
    assert "--error=/dev/null" not in command
    environment = _job_environment(
        {
            "key": "acceptance:component_parity_tests",
            "environment": {
                "ABPH_CONDA_BASE": "/stale/abph",
                "ABPH_CONDA_ENV": "stale_abph",
                "CONDA_BASE": "/stale/conda",
                "CONDA_ENV": "stale_conda",
            },
        },
        tmp_path,
    )
    assert environment["ABPH_CONDA_BASE"] == TIGRIS_CONDA_BASE
    assert environment["CONDA_BASE"] == TIGRIS_CONDA_BASE
    assert environment["ABPH_CONDA_ENV"] == TIGRIS_CONDA_ENV
    assert environment["CONDA_ENV"] == TIGRIS_CONDA_ENV
    source_identity = _source_identity(root)
    frozen_environment = _job_environment(
        {
            "key": "acceptance:component_parity_tests",
            "environment": {},
        },
        root,
        source_identity=source_identity,
    )
    assert (
        frozen_environment["ABPH_ACCEPTANCE_SOURCE_GIT_COMMIT"]
        == source_identity[0]
    )
    assert (
        frozen_environment["ABPH_ACCEPTANCE_SOURCE_STATUS_HASH"]
        == source_identity[1]
    )

    worker = (
        root / "sbatch" / "run_adaptive_binary_storage_acceptance.sh"
    ).read_text(encoding="utf-8")
    assert "abph_acceptance_source_identity" in worker
    assert "ABPH_ACCEPTANCE_SOURCE_GIT_COMMIT" in worker
    assert "ABPH_ACCEPTANCE_SOURCE_STATUS_HASH" in worker

    class _Result:
        def __init__(self, returncode: int, stdout: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **_kwargs):
        job_id = command[5] if command[0] == "sacct" else command[-1]
        if command[0] == "scontrol" and job_id == "10":
            return _Result(1)
        if command[0] == "sacct" and job_id == "10":
            return _Result(0, "10|COMPLETED|0:0|\n")
        if command[0] == "scontrol" and job_id == "20":
            return _Result(0, "JobId=20 JobState=RUNNING Reason=None")
        raise AssertionError(command)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "scripts.repair_adaptive_binary_storage_acceptance_graph.subprocess.run",
            fake_run,
        )
        assert _remaining_dependencies(("10", "20"), dry_run=False) == ["20"]


def test_ram_lifecycle_smoke_publishes_to_campaign_then_cleans(tmp_path: Path) -> None:
    campaign = tmp_path / "persistent_campaign"
    initialize_storage_accounting(campaign, profile=ABPH_STREAMING_STORAGE_PROFILE)
    workspace_root = tmp_path / "tmpfs_simulation" / "abph-job-0"
    probe = RamCapacityProbe(
        root=workspace_root.parent,
        filesystem_type="tmpfs",
        filesystem_free_bytes=8_000_000,
        cgroup_limit_bytes=8_000_000,
        cgroup_current_bytes=0,
        slurm_allocation_bytes=8_000_000,
    )
    workspace = RankLocalWorkspace(
        workspace_root, job_id="job", rank=0, probe=probe
    )
    reservation = workspace.reserve(
        owner="job", role="storage_lifecycle_smoke", expected_bytes=2_000_000
    )
    assert (
        run_ram_lifecycle_smoke(
            [
                "prepare",
                "--workspace",
                str(workspace.root),
                "--campaign-root",
                str(campaign),
            ]
        )
        == 0
    )
    stage = workspace.root / "codec_scratch" / "ram_lifecycle_stage"
    source = stage / "payload.bin"
    workspace.commit_tree(reservation, stage)
    destination = campaign / "storage" / "lifecycle_smoke_payload" / "payload.bin"
    publish_quota_managed_file(
        campaign,
        source,
        destination,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role=SMOKE_ARTIFACT_ROLE,
        source_provenance_hash="smoke-source",
        run_id="smoke",
        profile=ABPH_STREAMING_STORAGE_PROFILE,
    )
    workspace.release(reservation)
    output = campaign / "storage" / "ram_lifecycle_smoke.json"
    assert (
        run_ram_lifecycle_smoke(
            [
                "verify",
                "--workspace",
                str(workspace.root),
                "--campaign-root",
                str(campaign),
                "--output",
                str(output),
                "--source-git-commit",
                "commit",
                "--source-status-hash",
                "status",
            ]
        )
        == 0
    )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["persistent_publication_verified"] is True
    assert evidence["persistent_cleanup"]["destination_removed"] is True
    assert not destination.exists()
    ledger = json.loads(storage_paths(campaign)["ledger"].read_text(encoding="utf-8"))
    assert str(destination.resolve()) not in ledger["committed_artifacts"]

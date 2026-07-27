from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_adaptive_binary_storage_acceptance_tests import (
    isolated_test_environment,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.accounting_preflight import (
    ABPH_STEP4_PREFLIGHT_CONTRACT,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    AdaptiveBinarySubmissionConfig,
    build_submission_graph,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_acceptance import (
    ABPH_RAM_LIFECYCLE_SMOKE_CONTRACT,
    ABPH_STORAGE_ACCEPTANCE_TEST_FILES,
    ABPH_STORAGE_TEST_EVIDENCE_CONTRACT,
    build_storage_acceptance,
    canonical_hash,
    require_storage_acceptance,
    write_storage_acceptance,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_lifecycle import (
    build_artifact_manifest,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    StorageProjectionRow,
    build_storage_projection,
    initialize_storage_accounting,
    write_storage_projection,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.target_mode import (
    ABPH_RANK_LOCAL_TARGET_MODE,
    ABPH_TARGET_MODE_SELECTION_CONTRACT,
)


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _hashed(payload: dict) -> dict:
    payload = dict(payload)
    payload["content_hash"] = canonical_hash(payload)
    return payload


def _projection(root: Path, path: Path) -> dict:
    payload = build_storage_projection(
        campaign_root=root,
        campaign_mode="pilot",
        profile=ABPH_STREAMING_STORAGE_PROFILE,
        rows=(
            StorageProjectionRow(
                artifact_family="real_smoke",
                artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
                expected_bytes=10_000_000,
                active_from_wave=0,
                active_through_wave=6,
                retained=True,
                measurement_source="real_sample",
            ),
        ),
        measurement_contract="real_sample_v1",
        sample_provenance_hash="sample",
    )
    write_storage_projection(path, payload)
    return payload


def _inputs(root: Path) -> None:
    for split in ("model_train", "model_val", "stack_train", "stack_val", "final_test"):
        hlt = root / "inputs" / "hlt_cache"
        (hlt / f"{split}_fixed_hlt.npz").parent.mkdir(parents=True, exist_ok=True)
        (hlt / f"{split}_fixed_hlt.npz").write_bytes(f"hlt-{split}".encode())
        _json(
            hlt / f"{split}_fixed_hlt_metadata.json",
            {
                "split": split,
                "source_manifest_hash": "manifest",
                "hlt_content_hash": f"hlt-{split}",
                "jet_identity_hash": f"ids-{split}",
            },
        )
    for split in ("model_train", "model_val"):
        offline = root / "inputs" / "offline_cache"
        (offline / f"{split}_offline.npz").parent.mkdir(parents=True, exist_ok=True)
        (offline / f"{split}_offline.npz").write_bytes(f"offline-{split}".encode())
        _json(
            offline / f"{split}_offline_metadata.json",
            {
                "split": split,
                "source_manifest_hash": "manifest",
                "offline_content_hash": f"offline-{split}",
                "jet_identity_hash": f"ids-{split}",
            },
        )


def _acceptance_inputs(root: Path, projection_path: Path) -> dict[str, Path]:
    initialize_storage_accounting(
        root,
        profile=ABPH_STREAMING_STORAGE_PROFILE,
        projection=json.loads(projection_path.read_text(encoding="utf-8")),
    )
    _inputs(root)
    submission_path = _json(
        root / "submission_logs" / "abph_full_submission.json",
        {
            "submission_commands": [
                {
                    "command": [
                        "sbatch",
                        "--output=/dev/null",
                        "--error=/dev/null",
                    ],
                    "environment": {
                        "MIRROR_DIAGNOSTICS": "0",
                        "LOG_DIR": "/dev/shm/abph-test/logs",
                        "DIAGNOSTICS_ROOT": "/dev/shm/abph-test/diagnostics",
                    },
                }
            ]
        },
    )
    target_mode = _hashed(
        {
            "contract": ABPH_TARGET_MODE_SELECTION_CONTRACT,
            "campaign_root": str(root.resolve()),
            "campaign_mode": "pilot",
            "selected_mode": ABPH_RANK_LOCAL_TARGET_MODE,
            "selection_is_worker_overrideable": False,
            "measurement_content_hash": "measurement",
            "storage_projection_content_hash": json.loads(
                projection_path.read_text(encoding="utf-8")
            )["content_hash"],
            "source_manifest_hash": "manifest",
            "hlt_content_hash": "hlt-model_train",
            "offline_content_hash": "offline-model_train",
            "source_provenance_by_split": {
                split: {
                    "source_manifest_hash": "manifest",
                    "hlt_content_hash": f"hlt-{split}",
                    "offline_content_hash": f"offline-{split}",
                    "jet_identity_hash": f"ids-{split}",
                }
                for split in ("model_train", "model_val")
            },
            "workspace_capacity": {"reservation_limit_bytes": 1_000_000},
            "rank_local_target_working_set_bytes": 500_000,
            "minimum_ram_headroom_fraction": 0.20,
        }
    )
    target_mode_path = _json(
        root / "audits" / "target_mode_selection.json", target_mode
    )
    feasibility_path = _json(
        root / "audits" / "actual_target_feasibility.json",
        {
            "contract": ABPH_STEP4_PREFLIGHT_CONTRACT,
            "ok": True,
            "target_mode_selection_hash": target_mode["content_hash"],
        },
    )
    manifest = build_artifact_manifest(root, data_dir=root / "data")
    audit_path = _json(
        root / "storage" / "storage_audits" / "wave_2.json",
        _hashed(
            {
                "campaign_root": str(root.resolve()),
                "storage_profile": ABPH_STREAMING_STORAGE_PROFILE,
                "measured_persistent_bytes": 1_000_000,
                "ok": True,
            }
        ),
    )
    tests_path = _json(
        root / "storage" / "storage_acceptance_tests.json",
        _hashed(
            {
                "contract": ABPH_STORAGE_TEST_EVIDENCE_CONTRACT,
                "ok": True,
                "return_code": 0,
                "test_files": list(ABPH_STORAGE_ACCEPTANCE_TEST_FILES),
                "passed": 100,
                "skipped": 0,
                "source_git_commit": "commit",
                "source_status_hash": "status",
            }
        ),
    )
    runtime_path = _json(
        root / "runtime.json",
        {
            "runtime": "accepted",
            "promotion": {
                "production_reconstructor_parallelism": "ddp4",
            },
        },
    )
    ram_smoke_path = _json(
        root / "storage" / "ram_lifecycle_smoke.json",
        _hashed(
            {
                "contract": ABPH_RAM_LIFECYCLE_SMOKE_CONTRACT,
                "ok": True,
                "campaign_root": str(root.resolve()),
                "persistent_destination": str(
                    (
                        root
                        / "storage"
                        / "lifecycle_smoke_payload"
                        / "payload.bin"
                    ).resolve()
                ),
                "persistent_publication_verified": True,
                "persistent_cleanup": {
                    "contract": "adaptive_binary_quota_managed_cleanup_v1",
                    "ok": True,
                    "artifact_path": str(
                        (
                            root
                            / "storage"
                            / "lifecycle_smoke_payload"
                            / "payload.bin"
                        ).resolve()
                    ),
                    "artifact_role": "ram_lifecycle_smoke_payload",
                    "sha256": "smoke-sha256",
                    "destination_removed": True,
                    "ledger_reconciled": True,
                },
                "source_bytes": 262144,
                "source_sha256": "smoke-sha256",
                "published_sha256": "smoke-sha256",
                "source_git_commit": "commit",
                "source_status_hash": "status",
                "problems": [],
            }
        ),
    )
    return {
        "target_mode": target_mode_path,
        "feasibility": feasibility_path,
        "manifest": root / "storage" / "artifact_manifest.json",
        "audit": audit_path,
        "tests": tests_path,
        "runtime": runtime_path,
        "ram_smoke": ram_smoke_path,
        "submission": submission_path,
    }


def test_storage_acceptance_binds_real_smoke_and_component_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    projection_path = tmp_path / "projection.json"
    _projection(root, projection_path)
    paths = _acceptance_inputs(root, projection_path)
    monkeypatch.setattr(
        "teacher_logit_reco.adaptive_binary_pseudooffline.storage_acceptance.require_runtime_acceptance",
        lambda *_args, **_kwargs: {"acceptance_content_hash": "runtime-hash"},
    )
    payload = build_storage_acceptance(
        campaign_root=root,
        campaign_mode="pilot",
        storage_projection=projection_path,
        target_mode_selection=paths["target_mode"],
        target_feasibility=paths["feasibility"],
        wave_two_audit=paths["audit"],
        artifact_manifest=paths["manifest"],
        runtime_acceptance=paths["runtime"],
        test_evidence=paths["tests"],
        ram_lifecycle_smoke=paths["ram_smoke"],
        source_git_commit="commit",
        source_status_hash="status",
    )
    assert payload["ok"] is True
    assert payload["gates"]["no_persistent_pseudo_cache"] is True
    output = root / "storage" / "storage_acceptance.json"
    write_storage_acceptance(root, output, payload)
    assert require_storage_acceptance(output, campaign_root=root)["ok"] is True


@pytest.mark.parametrize(
    "evidence_key",
    (
        "target_mode",
        "feasibility",
        "audit",
        "manifest",
        "submission",
        "ram_smoke",
    ),
)
def test_storage_acceptance_reuse_rejects_changed_real_data_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_key: str,
) -> None:
    root = tmp_path / "campaign"
    projection_path = tmp_path / "projection.json"
    _projection(root, projection_path)
    paths = _acceptance_inputs(root, projection_path)
    monkeypatch.setattr(
        "teacher_logit_reco.adaptive_binary_pseudooffline.storage_acceptance.require_runtime_acceptance",
        lambda *_args, **_kwargs: {"acceptance_content_hash": "runtime-hash"},
    )
    payload = build_storage_acceptance(
        campaign_root=root,
        campaign_mode="pilot",
        storage_projection=projection_path,
        target_mode_selection=paths["target_mode"],
        target_feasibility=paths["feasibility"],
        wave_two_audit=paths["audit"],
        artifact_manifest=paths["manifest"],
        runtime_acceptance=paths["runtime"],
        test_evidence=paths["tests"],
        ram_lifecycle_smoke=paths["ram_smoke"],
        source_git_commit="commit",
        source_status_hash="status",
    )
    output = root / "storage" / "storage_acceptance.json"
    write_storage_acceptance(root, output, payload)
    paths[evidence_key].write_bytes(paths[evidence_key].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="evidence changed"):
        require_storage_acceptance(output, campaign_root=root)


def test_storage_acceptance_rejects_persistent_pseudo_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    projection_path = tmp_path / "projection.json"
    _projection(root, projection_path)
    paths = _acceptance_inputs(root, projection_path)
    pseudo = root / "pseudo_predictions" / "D1" / "model_train" / "shard.npz"
    pseudo.parent.mkdir(parents=True)
    pseudo.write_bytes(b"forbidden")
    monkeypatch.setattr(
        "teacher_logit_reco.adaptive_binary_pseudooffline.storage_acceptance.require_runtime_acceptance",
        lambda *_args, **_kwargs: {"acceptance_content_hash": "runtime-hash"},
    )
    payload = build_storage_acceptance(
        campaign_root=root,
        campaign_mode="pilot",
        storage_projection=projection_path,
        target_mode_selection=paths["target_mode"],
        target_feasibility=paths["feasibility"],
        wave_two_audit=paths["audit"],
        artifact_manifest=paths["manifest"],
        runtime_acceptance=paths["runtime"],
        test_evidence=paths["tests"],
        ram_lifecycle_smoke=paths["ram_smoke"],
        source_git_commit="commit",
        source_status_hash="status",
    )
    assert payload["ok"] is False
    assert "persistent pseudo-view artifacts" in " ".join(payload["problems"])


def test_full_streaming_graph_gates_models_on_storage_acceptance(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    projection_path = tmp_path / "projection.json"
    _projection(root, projection_path)
    graph = build_submission_graph(
        AdaptiveBinarySubmissionConfig(
            campaign_root=root,
            data_dir=tmp_path / "data",
            storage_profile=ABPH_STREAMING_STORAGE_PROFILE,
            storage_projection_path=projection_path,
            reconstructor_parallelism="single",
            allow_debug_single_reconstructor=True,
        )
    )
    jobs = {job.key: job for job in graph}
    assert jobs["acceptance:storage_smoke"].dependencies == (
        "wave:2",
        "acceptance:component_parity_tests",
        "acceptance:ram_lifecycle_smoke",
    )
    assert jobs["acceptance:ram_lifecycle_smoke"].dependencies == ("wave:0",)
    for job in graph:
        if job.stage == "runtime_batch_probe":
            assert "acceptance:storage_smoke" in job.dependencies
        assert job.environment["ABPH_SUPPRESS_SLURM_LOGS"] == "1"
        assert job.environment["MIRROR_DIAGNOSTICS"] == "0"


def test_storage_component_tests_isolate_live_campaign_and_slurm_environment() -> None:
    environment, removed = isolated_test_environment(
        {
            "ABPH_ROOT": "/live/campaign",
            "ABPH_STORAGE_PROFILE": "streaming_30gb_v1",
            "CONDA_PREFIX": "/keep/conda",
            "DATA_DIR": "/live/data",
            "PATH": "/keep/bin",
            "SLURM_JOB_ID": "19084",
            "WORLD_SIZE": "8",
        }
    )
    assert set(removed) == {
        "ABPH_ROOT",
        "ABPH_STORAGE_PROFILE",
        "DATA_DIR",
        "SLURM_JOB_ID",
        "WORLD_SIZE",
    }
    assert environment["CONDA_PREFIX"] == "/keep/conda"
    assert environment["PATH"] == "/keep/bin"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"

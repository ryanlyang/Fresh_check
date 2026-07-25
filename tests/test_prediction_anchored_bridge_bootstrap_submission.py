from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import scripts.bootstrap_prediction_anchored_bridge_preflight as bootstrap
import scripts.refresh_prediction_anchored_bridge_preflight as refresh
from scripts.bootstrap_prediction_anchored_bridge_preflight import _baseline_model_size
from tests.test_prediction_anchored_bridge_execution import (
    _ToyConsumer,
    _config as miniature_split_config,
    _fixture as execution_fixture,
)
from teacher_logit_reco.local_particle_residual_field import (
    build_child_split_manifest,
    record_registry_measurements,
    require_production_ready,
    render_tigris_sbatch_commands,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
    with_content_hash,
)


ROOT = Path(__file__).resolve().parents[1]


def test_clean_start_bootstrap_cli_imports_without_campaign_outputs():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/bootstrap_prediction_anchored_bridge_preflight.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--baseline-checkpoint" in completed.stdout
    assert "--artifact-root" in completed.stdout


def test_refresh_cli_imports_when_launched_outside_repository(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts/refresh_prediction_anchored_bridge_preflight.py"
            ),
            "--help",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--source-preflight" in completed.stdout
    assert "--artifact-root" in completed.stdout


def test_bootstrap_derives_and_requires_the_baseline_model_size(tmp_path):
    checkpoint = tmp_path / "baseline.pt"
    torch.save({"model_state_dict": {}, "model_config": {"model_size": "tiny"}}, checkpoint)
    assert _baseline_model_size(checkpoint) == "tiny"
    torch.save({"model_state_dict": {}}, tmp_path / "unknown.pt")
    with pytest.raises(ValueError, match="model_size"):
        _baseline_model_size(tmp_path / "unknown.pt")


def test_full_bootstrap_submitter_builds_locked_500k_source_before_finalizer():
    submitter = (ROOT / "sbatch/submit_prediction_anchored_bridge_full_bootstrap.sh").read_text()
    finalizer = (ROOT / "sbatch/run_finalize_prediction_anchored_bridge_submission.sh").read_text()
    assert "rebuild_and_pilot_20260720_185817" not in submitter
    assert 'PAB_REBUILD_SOURCE:=1' in submitter
    assert 'PAB_SOURCE_BASE="${PAB_NEW_SOURCE_BASE:-' in submitter
    assert "source_500k_${stamp}" in submitter
    for setting in (
        "MODEL_TRAIN_SIZE=500000",
        "MODEL_VAL_SIZE=150000",
        "STACK_TRAIN_SIZE=500000",
        "STACK_VAL_SIZE=150000",
        "FINAL_TEST_SIZE=150000",
    ):
        assert f"export {setting}" in submitter
    assert (
        'ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS='
        '"model_train model_val stack_train stack_val"'
    ) in submitter
    assert "run_build_fresh_splits.sh" in submitter
    assert "run_build_fresh_hlt_cache.sh" in submitter
    assert "run_cache_architecture_view_offline_inputs.sh" in submitter
    assert "run_train_local_residual_field_tagger.sh A0" in submitter
    assert '--dependency="afterok:${split_job}"' in submitter
    assert '--dependency="afterok:${hlt_job}:${offline_job}"' in submitter
    assert 'source_dependency="${a0_job}"' in submitter
    assert 'dependency=(--dependency="afterok:${source_dependency}")' in submitter
    assert 'job_id="${output%%;*}"' in submitter
    assert '--account="${PAB_SBATCH_ACCOUNT}"' in submitter
    assert "PAB_SBATCH_ACCOUNT:=reu-aisocial" in submitter
    assert "PAB_SBATCH_PARTITION:=tigris" in submitter
    assert "PREDICTION_ANCHORED_EXECUTE=1" in finalizer
    assert "submit_prediction_anchored_bridge_pilot.sh" in finalizer
    assert "PYTHONNOUSERSITE=1" in submitter + finalizer
    assert "PAB_CONDA_BASE:=/home/ryreu/miniforge3-aarch64" in submitter
    assert "PAB_CONDA_ENV:=atlas_kd_tigris" in submitter
    assert "PAB_CONDA_BASE:=/home/ryreu/miniforge3-aarch64" in finalizer
    assert "PAB_CONDA_ENV:=atlas_kd_tigris" in finalizer
    assert 'export CONDA_ENV="${PAB_CONDA_ENV}"' in finalizer
    assert 'export CONDA_BASE="${PAB_CONDA_BASE}"' in finalizer
    assert "SKIP_CONDA=1" not in finalizer
    assert "#SBATCH --nodes=1" in finalizer


def test_recovery_wrapper_refreshes_graph_and_reuses_exact_b0_b3_nodes():
    wrapper = (
        ROOT / "sbatch/run_refresh_prediction_anchored_bridge_recovery.sh"
    ).read_text()
    refresher = (
        ROOT / "scripts/refresh_prediction_anchored_bridge_preflight.py"
    ).read_text()
    assert "#SBATCH --account=reu-aisocial" in wrapper
    assert "PYTHONNOUSERSITE=1" in wrapper
    assert "PAB_CONDA_ENV:=atlas_kd_tigris" in wrapper
    assert "refresh_prediction_anchored_bridge_preflight.py" in wrapper
    assert '"b0_validate_preflight=${PAB_B0_JOB_ID}"' in wrapper
    assert '"b1_train_register_r0=${PAB_B1_JOB_ID}"' in wrapper
    assert '"b2_stage_recipes_scalers=${PAB_B2_JOB_ID}"' in wrapper
    assert '"b3_consumers_paired3=${PAB_B3_CONSUMER_JOB_ID}"' in wrapper
    assert '"b3_l0_paired3=${PAB_B3_L0_JOB_ID}"' in wrapper
    assert '"b4_select_consumer=${PAB_B4_SELECT_JOB_ID}"' in wrapper
    assert '"b4_confirm_consumer=${PAB_B4_CONFIRM_JOB_ID}"' in wrapper
    assert 'build_step3_consumer_model("T10_robust"' in refresher
    assert "--execute" in wrapper


def test_finalizer_binds_all_four_immutable_submission_controls():
    finalizer = (ROOT / "sbatch/run_finalize_prediction_anchored_bridge_submission.sh").read_text()
    for name in (
        "PREDICTION_ANCHORED_GRAPH",
        "PAB_REGISTRY",
        "PAB_RESERVATIONS",
        "PAB_EXECUTION_SPEC",
        "PAB_REPRESENTATIVE_RESOURCE_REFERENCE",
    ):
        assert f"export {name}=" in finalizer
    assert "--budget-gib 5" in finalizer


def test_b4_confirmation_publishes_checkpoint_bound_runtime_resource_reference():
    runner = (ROOT / "sbatch/run_prepare_prediction_anchored_bridge_ram.sh").read_text()
    bootstrap = (
        ROOT / "scripts/bootstrap_prediction_anchored_bridge_preflight.py"
    ).read_text()
    registration_initializer = (
        ': "${PAB_R0_REGISTRATION:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}'
        '/r0/r0_registration.json}"'
    )
    b2_branch = runner.split("  B2)", maxsplit=1)[1].split("    ;;", maxsplit=1)[0]
    b4_runtime_branch = runner.split(
        "  B4_RUNTIME_RESOURCES)", maxsplit=1
    )[1].split("    ;;", maxsplit=1)[0]
    b4_select_branch = runner.split(
        "  B4_SELECT)", maxsplit=1
    )[1].split("    ;;", maxsplit=1)[0]
    assert "representative_architecture_resource_reference.json" in bootstrap
    assert 'build_step3_consumer_model("T10_robust"' in bootstrap
    assert 'build_step3_consumer_model("T10_clean"' not in bootstrap
    assert "canonical_a3_bundle_resources.json" not in bootstrap
    assert "measurements/deployed_resource_reference.json" not in bootstrap
    assert "publish_prediction_anchored_runtime_resources.py" in runner
    assert "PAB_REPRESENTATIVE_RESOURCE_REFERENCE" in runner
    assert '--physical45-recipe "${PAB_PHYSICAL45_RECIPE}"' in runner
    assert '--execution-spec "${PAB_EXECUTION_SPEC}"' in runner
    assert '--r0-registration "${PAB_R0_REGISTRATION}"' in runner
    assert "--device cpu" in runner
    assert "measurements/deployed_resource_reference.json" in runner
    assert b2_branch.count(registration_initializer) == 1
    assert b4_runtime_branch.count(registration_initializer) == 1
    assert b4_runtime_branch.index(registration_initializer) < b4_runtime_branch.index(
        '--r0-registration "${PAB_R0_REGISTRATION}"'
    )
    assert "PAB_CONSUMER_QUALITY_GATE_POLICY:=warn_and_continue" in b4_select_branch
    assert "PAB_PREFERRED_CONSUMER:=T10_robust" in b4_select_branch
    assert '--quality-gate-policy "${PAB_CONSUMER_QUALITY_GATE_POLICY}"' in b4_select_branch
    assert '--preferred-consumer "${PAB_PREFERRED_CONSUMER}"' in b4_select_branch


def test_miniature_clean_start_reaches_registry_reservations_graph_and_rendering(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    spec, _spec_path = execution_fixture(source_root)
    parent_path = Path(spec["parent_manifest"]["path"])
    baseline_path = Path(spec["baseline_checkpoint"]["path"])
    monkeypatch.setattr(bootstrap, "_baseline_model_size", lambda _path: "tiny")
    monkeypatch.setattr(
        bootstrap,
        "build_child_split_manifest",
        lambda parent: build_child_split_manifest(
            parent, config=miniature_split_config()
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "build_prediction_anchored_execution_spec",
        lambda **_kwargs: spec,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_step3_consumer_model",
        lambda run_id, model_size: _ToyConsumer(run_id),
    )
    monkeypatch.setattr(
        bootstrap,
        "initialize_step3_root_from_reference",
        lambda *args, **kwargs: {"ok": True},
    )

    def miniature_measure(registry, *, fixed_storage, selected_budget_bytes, **_kwargs):
        sizes = {
            row["canonical_run_id"]: 1024
            for row in registry["runs"]
            if row["measurement_status"] == "UNMEASURED"
        }
        measured = record_registry_measurements(registry, sizes)
        readiness = require_production_ready(
            measured,
            fixed_persistent_bytes=fixed_storage.total_bytes,
            selected_budget_bytes=selected_budget_bytes,
        )
        return measured, with_content_hash(
            {
                "contract": "miniature_clean_start_measurement_v1",
                "production_readiness": readiness,
            }
        )

    monkeypatch.setattr(
        bootstrap, "measure_step8_registry_states", miniature_measure
    )
    output = tmp_path / "preflight"
    artifact_root = tmp_path / "campaign"
    assert (
        bootstrap.main(
            [
                "--parent-manifest", str(parent_path),
                "--hlt-cache-dir", str(source_root / "hlt"),
                "--offline-cache-dir", str(source_root / "offline"),
                "--baseline-checkpoint", str(baseline_path),
                "--output-dir", str(output),
                "--artifact-root", str(artifact_root),
                "--budget-gib", "5",
            ]
        )
        == 0
    )
    registry = json.loads(
        (output / "campaign_registry_step8.json").read_text()
    )
    reservations = json.loads(
        (output / "campaign_reservations.json").read_text()
    )
    graph = json.loads(
        (output / "prediction_anchored_tigris_graph.json").read_text()
    )
    representative = json.loads(
        (
            output / "representative_architecture_resource_reference.json"
        ).read_text()
    )
    rendered = render_tigris_sbatch_commands(graph)
    assert registry["configuration_count"] == 55
    assert reservations["projected_persistent_bytes"] <= 5 * 1024**3
    assert reservations["representative_reference_sha256"] == representative[
        "content_hash"
    ]
    assert graph["representative_reference_sha256"] == representative["content_hash"]
    assert graph["covered_runnable_configuration_count"] == 55
    assert graph["conditional_skips"] == []
    assert any(
        node["teacher_namespace"] == "physical45_alternate_bridge_teacher"
        for node in graph["nodes"]
    )
    assert len(rendered["commands"]) == len(graph["nodes"]) - 1
    assert representative["checkpoint_hashes_present"] is False
    assert not (
        artifact_root / "measurements" / "deployed_resource_reference.json"
    ).exists()

    monkeypatch.setattr(
        refresh,
        "build_step3_consumer_model",
        lambda run_id, model_size: _ToyConsumer(run_id),
    )
    monkeypatch.setattr(refresh, "_baseline_model_size", lambda _path: "tiny")
    monkeypatch.setattr(
        refresh,
        "initialize_step3_root_from_reference",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        refresh, "measure_step8_registry_states", miniature_measure
    )
    refreshed = tmp_path / "preflight_clean_pair_v2"
    artifact_root.mkdir()
    assert (
        refresh.main(
            [
                "--source-preflight",
                str(output),
                "--output-dir",
                str(refreshed),
                "--artifact-root",
                str(artifact_root),
                "--budget-gib",
                "5",
            ]
        )
        == 0
    )
    original_spec = json.loads(
        (output / "prediction_anchored_execution_spec.json").read_text()
    )
    refreshed_spec = json.loads(
        (refreshed / "prediction_anchored_execution_spec.json").read_text()
    )
    refreshed_graph = json.loads(
        (refreshed / "prediction_anchored_tigris_graph.json").read_text()
    )
    assert refreshed_spec["content_hash"] == original_spec["content_hash"]
    assert refreshed_graph["execution_spec_sha256"] == original_spec["content_hash"]
    assert refreshed_graph["configuration_count"] == 55
    assert refreshed_graph["covered_runnable_configuration_count"] == 55

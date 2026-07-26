from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from jetclass_fresh.hlt_cache import generate_and_cache_hlt_view
from jetclass_fresh.jetclass_data import (
    JetView,
    save_split_manifest,
)
from teacher_logit_reco.architecture_view_part import save_cached_offline_view
from teacher_logit_reco.local_particle_residual_field.particle_view import (
    ParticleViewRunSpec,
    DirectControlTrainConfig,
    build_baseline_factory,
    build_baseline_factory_config,
    build_direct_control_factory,
    build_direct_control_factory_config,
    build_direct_control_recipe,
    build_particle_view_registry,
    build_runtime_data_config,
    build_runtime_task_result,
    build_scientific_task_catalog,
    build_stage_a_teacher_task_specs,
    build_stage_a_direct_resource_plan,
    build_stage_a_direct_task_specs,
    build_unified_split_manifest,
    execute_scientific_task,
    load_aligned_logical_jet_view,
    make_logical_data_loader,
    miniature_parent_manifest,
    miniature_split_config,
    resolve_parent_task_artifacts,
    register_existing_teacher_source,
    sha256_file,
    validate_runtime_data_config,
    validate_baseline_factory_config,
    validate_stage_a_direct_resource_plan,
    train_direct_hlt_control,
    write_immutable_json,
)


def _offline_view(parent, split: str) -> JetView:
    identities = list(parent.splits[split])
    count = len(identities)
    tokens = np.zeros((count, 128, 14), dtype=np.float32)
    mask = np.zeros((count, 128), dtype=bool)
    labels = np.asarray([row.label for row in identities], dtype=np.int64)
    for index in range(count):
        particles = 5 + index % 3
        mask[index, :particles] = True
        tokens[index, :particles, 0] = np.linspace(
            5.0, 1.5, particles, dtype=np.float32
        )
        tokens[index, :particles, 1] = np.linspace(
            -0.3, 0.3, particles, dtype=np.float32
        )
        tokens[index, :particles, 2] = np.linspace(
            -0.5, 0.5, particles, dtype=np.float32
        )
        tokens[index, :particles, 3] = (
            tokens[index, :particles, 0]
            * np.cosh(tokens[index, :particles, 1])
        )
        tokens[index, :particles, 5 + (index % 5)] = 1.0
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=identities,
        split=split,
        metadata={"view": "offline", "fixture": "particle_view_runtime_data"},
    )


def _runtime_sources(tmp_path: Path, *, reorder_offline: str | None = None):
    parent = miniature_parent_manifest(rows_per_class=2)
    split_config = miniature_split_config(rows_per_class=2)
    unified = build_unified_split_manifest(parent, config=split_config)
    parent_path = tmp_path / "parent_manifest.json"
    unified_path = tmp_path / "unified_manifest.json"
    save_split_manifest(parent, parent_path, pretty=True)
    write_immutable_json(unified_path, unified)
    hlt_root = tmp_path / "hlt"
    offline_root = tmp_path / "offline"
    used = ("model_train", "model_val", "stack_val", "final_test")
    for offset, split in enumerate(used):
        canonical = _offline_view(parent, split)
        generate_and_cache_hlt_view(
            canonical,
            hlt_root,
            seed=800 + offset,
        )
        offline = canonical
        if split == reorder_offline:
            order = np.arange(len(canonical.labels))[::-1]
            offline = JetView(
                tokens=canonical.tokens[order],
                mask=canonical.mask[order],
                labels=canonical.labels[order],
                jet_ids=[canonical.jet_ids[int(index)] for index in order],
                split=split,
                metadata=dict(canonical.metadata),
            )
        save_cached_offline_view(offline, offline_root)
    runtime_config = build_runtime_data_config(
        parent_manifest_path=parent_path,
        unified_manifest_path=unified_path,
        hlt_cache_dir=hlt_root,
        offline_cache_dir=offline_root,
    )
    return parent, unified, runtime_config


def test_runtime_data_aligns_logical_slices_and_builds_label_free_probe(tmp_path):
    _, unified, config = _runtime_sources(tmp_path)
    audit = validate_runtime_data_config(config)
    assert audit["parent_splits"] == [
        "final_test",
        "model_train",
        "model_val",
        "stack_val",
    ]

    stop = load_aligned_logical_jet_view(config, "model_val_stop")
    select = load_aligned_logical_jet_view(config, "model_val_select")
    assert len(stop) == len(select) == 10
    assert {row.key() for row in stop.identities}.isdisjoint(
        {row.key() for row in select.identities}
    )
    assert stop.logical_split_sha256 == unified["logical_splits"][
        "model_val_stop"
    ]["content_hash"]

    aligned_batch = next(
        iter(
            make_logical_data_loader(
                stop,
                mode="aligned",
                batch_size=4,
                shuffle=False,
                num_workers=0,
                seed=101,
            )
        )
    )
    assert aligned_batch["features"].shape[0] == 4
    assert aligned_batch["offline_features"].shape[0] == 4
    assert aligned_batch["labels"].shape == (4,)

    true_views = np.zeros((len(stop), 128, 2), dtype=np.float32)
    true_views[stop.hlt.mask[stop.parent_row_indices], :] = 0.25
    probe_batch = next(
        iter(
            make_logical_data_loader(
                stop,
                mode="recovery_probe",
                true_views=true_views,
                batch_size=4,
                shuffle=False,
                num_workers=0,
                seed=101,
            )
        )
    )
    assert set(probe_batch) == {"features", "mask", "true_view"}
    assert probe_batch["features"].shape[1] == 17
    assert probe_batch["mask"].shape == (4, 128)


def test_runtime_data_rejects_mutually_consistent_but_parent_reordered_cache(
    tmp_path,
):
    _, _, config = _runtime_sources(
        tmp_path, reorder_offline="model_val"
    )
    with pytest.raises(ValueError, match="offline cache identity order"):
        load_aligned_logical_jet_view(config, "model_val_stop")


def test_runtime_data_config_rejects_cache_changed_after_binding(tmp_path):
    _, _, config = _runtime_sources(tmp_path)
    path = Path(config["parent_cache_records"][0]["hlt_metadata"]["path"])
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        validate_runtime_data_config(config)


def test_parent_artifact_resolver_uses_seed_fallback_and_authenticates(tmp_path):
    registry = build_particle_view_registry(
        unified_split_manifest_sha256="a" * 64,
        train_identity_sha256="b" * 64,
        run_specs=[
            ParticleViewRunSpec(
                run_id="parent",
                stage="source",
                scientific_role="source:parent",
                selection_family="infrastructure",
                seed_ids=(101,),
                uses_labels=False,
                train_split=None,
            ),
            ParticleViewRunSpec(
                run_id="child",
                stage="baseline",
                scientific_role="baseline:child",
                selection_family="infrastructure",
                seed_ids=(101, 202, 303),
                parent_run_ids=("parent",),
            ),
        ],
        campaign_id="runtime_parent_resolution",
    )
    task_id = "parent__seed_101"
    task_root = tmp_path / "runtime_tasks" / task_id
    task_root.mkdir(parents=True)
    artifact = task_root / "source.json"
    artifact.write_text("source", encoding="utf-8")
    result = build_runtime_task_result(
        task_id=task_id,
        artifacts=[{"path": str(artifact), "sha256": sha256_file(artifact)}],
    )
    write_immutable_json(task_root / "task_result.json", result)
    resolved = resolve_parent_task_artifacts(
        registry=registry,
        artifact_root=tmp_path,
        run_id="child",
        seed=303,
    )
    assert resolved["parent"]["seed"] == 101
    assert resolved["parent"]["artifacts"]["source.json"]["sha256"] == (
        sha256_file(artifact)
    )
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        resolve_parent_task_artifacts(
            registry=registry,
            artifact_root=tmp_path,
            run_id="child",
            seed=303,
        )


def test_source_preflight_scientific_factory_runs_real_cache_audit(tmp_path):
    _, unified, config = _runtime_sources(tmp_path / "sources")
    config_path = tmp_path / "runtime_data_config.json"
    write_immutable_json(config_path, config)
    registry = build_particle_view_registry(
        unified_split_manifest_sha256=unified["content_hash"],
        train_identity_sha256=unified["logical_splits"]["train"][
            "ordered_identity_sha256"
        ],
        run_specs=[
            ParticleViewRunSpec(
                run_id="PV_SOURCE_PREFLIGHT",
                stage="source",
                scientific_role="source:unified_manifest_sources_storage",
                selection_family="infrastructure",
                uses_labels=False,
                train_split=None,
            )
        ],
        campaign_id="runtime_source_factory",
    )
    catalog = build_scientific_task_catalog(
        registry=registry,
        task_specs={
            "PV_SOURCE_PREFLIGHT": {
                "operation": "source_preflight",
                "factory": (
                    "teacher_logit_reco.local_particle_residual_field."
                    "particle_view.production_factories:"
                    "build_source_preflight_factory"
                ),
                "factory_config_path": str(config_path),
                "factory_config_sha256": sha256_file(config_path),
            }
        },
    )
    output = tmp_path / "source_task"
    result = execute_scientific_task(
        catalog=catalog,
        registry=registry,
        run_id="PV_SOURCE_PREFLIGHT",
        seed=101,
        task_id="PV_SOURCE_PREFLIGHT__seed_101",
        output_dir=output,
    )
    assert result["status"] == "complete"
    audit_path = output / "source_cache_audit.json"
    assert audit_path.is_file()
    assert "source_cache_audit" in audit_path.read_text(encoding="utf-8")


def test_runtime_data_config_cli_publishes_authenticated_config(tmp_path):
    _, _, config = _runtime_sources(tmp_path / "sources")
    output = tmp_path / "cli_runtime_data_config.json"
    result = subprocess.run(
        [
            str(Path(".venv/Scripts/python.exe")),
            "scripts/build_particle_view_runtime_data_config.py",
            "--parent-manifest",
            config["parent_manifest"]["path"],
            "--unified-manifest",
            config["unified_manifest"]["path"],
            "--hlt-cache-dir",
            config["hlt_cache_dir"],
            "--offline-cache-dir",
            config["offline_cache_dir"],
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert "parent_splits=4" in result.stdout


def _stage_a_registry(unified):
    return build_particle_view_registry(
        unified_split_manifest_sha256=unified["content_hash"],
        train_identity_sha256=unified["logical_splits"]["train"][
            "ordered_identity_sha256"
        ],
        run_specs=[
            ParticleViewRunSpec(
                run_id="PV_SOURCE_PREFLIGHT",
                stage="source",
                scientific_role="source:unified_manifest_sources_storage",
                selection_family="infrastructure",
                uses_labels=False,
                train_split=None,
            ),
            ParticleViewRunSpec(
                run_id="A0_VIEW",
                stage="baseline",
                scientific_role="baseline:A0_VIEW",
                selection_family="infrastructure",
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
            ),
            ParticleViewRunSpec(
                run_id="TOFF_VIEW_BASE",
                stage="baseline",
                scientific_role="baseline:TOFF_VIEW_BASE",
                selection_family="infrastructure",
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
            ),
            ParticleViewRunSpec(
                run_id="TOFF_VIEW_LARGE",
                stage="baseline",
                scientific_role="baseline:TOFF_VIEW_LARGE",
                selection_family="infrastructure",
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
            ),
            ParticleViewRunSpec(
                run_id="TOFF_VIEW_EXISTING",
                stage="baseline",
                scientific_role="baseline:TOFF_VIEW_EXISTING",
                selection_family="diagnostic",
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
                diagnostic=True,
            ),
            ParticleViewRunSpec(
                run_id="STAGE_A_PARAMETER_MATCH",
                stage="baseline",
                scientific_role="baseline:STAGE_A_PARAMETER_MATCH",
                selection_family="infrastructure",
                seed_ids=(101, 202, 303),
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
            ),
            ParticleViewRunSpec(
                run_id="STAGE_A_FLOP_MATCH",
                stage="baseline",
                scientific_role="baseline:STAGE_A_FLOP_MATCH",
                selection_family="infrastructure",
                seed_ids=(101, 202, 303),
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
            ),
        ],
        campaign_id="stage_a_factory_test",
    )


def _publish_source_parent_result(artifact_root: Path):
    task_id = "PV_SOURCE_PREFLIGHT__seed_101"
    root = artifact_root / "runtime_tasks" / task_id
    root.mkdir(parents=True)
    audit = root / "source_cache_audit.json"
    audit.write_text("authenticated-source-audit", encoding="utf-8")
    result = build_runtime_task_result(
        task_id=task_id,
        artifacts=[{"path": str(audit), "sha256": sha256_file(audit)}],
    )
    write_immutable_json(root / "task_result.json", result)


def test_stage_a_teacher_factory_builds_real_hlt_and_offline_loaders(tmp_path):
    _, unified, data = _runtime_sources(tmp_path / "sources")
    artifact_root = tmp_path / "campaign"
    _publish_source_parent_result(artifact_root)
    registry = _stage_a_registry(unified)
    existing_checkpoint = tmp_path / "existing.pt"
    existing_checkpoint.write_bytes(b"existing-offline-teacher")
    config = build_baseline_factory_config(
        runtime_data_config=data,
        device="cpu",
        num_workers=0,
        max_train_batches=1,
        max_val_batches=1,
        existing_checkpoint_path=existing_checkpoint,
        existing_provenance_metadata_sha256="c" * 64,
    )
    assert validate_baseline_factory_config(config)["trained_teacher_count"] == 3

    a0 = build_baseline_factory(
        operation="teacher_training",
        config=config,
        registry=registry,
        run_id="A0_VIEW",
        seed=101,
        task_id="A0_VIEW__seed_101",
        output_dir=str(
            artifact_root / "runtime_tasks" / "A0_VIEW__seed_101"
        ),
    )
    assert a0["action"] is None
    assert a0["kwargs"]["recipe"].role == "A0_view"
    assert a0["kwargs"]["recipe"].particle_source == "fixed_hlt"
    assert a0["kwargs"]["train_loader"].batch_size == 128
    assert a0["kwargs"]["config"].max_train_batches == 1
    assert len(a0["artifact_paths"]) == 4

    offline = build_baseline_factory(
        operation="teacher_training",
        config=config,
        registry=registry,
        run_id="TOFF_VIEW_LARGE",
        seed=101,
        task_id="TOFF_VIEW_LARGE__seed_101",
        output_dir=str(
            artifact_root / "runtime_tasks" / "TOFF_VIEW_LARGE__seed_101"
        ),
    )
    assert offline["kwargs"]["recipe"].role == "Toff_view"
    assert offline["kwargs"]["recipe"].architecture == "large"
    assert offline["kwargs"]["train_loader"].batch_size == 64
    assert (
        a0["kwargs"]["recipe"].unified_split_manifest_sha256
        == offline["kwargs"]["recipe"].unified_split_manifest_sha256
    )


def test_existing_teacher_factory_is_diagnostic_and_source_bound(tmp_path):
    _, unified, data = _runtime_sources(tmp_path / "sources")
    artifact_root = tmp_path / "campaign"
    _publish_source_parent_result(artifact_root)
    registry = _stage_a_registry(unified)
    checkpoint = tmp_path / "existing.pt"
    checkpoint.write_bytes(b"existing")
    config = build_baseline_factory_config(
        runtime_data_config=data,
        device="cpu",
        existing_checkpoint_path=checkpoint,
        existing_provenance_metadata_sha256="d" * 64,
    )
    prepared = build_baseline_factory(
        operation="existing_teacher_registration",
        config=config,
        registry=registry,
        run_id="TOFF_VIEW_EXISTING",
        seed=101,
        task_id="TOFF_VIEW_EXISTING__seed_101",
        output_dir=str(
            artifact_root / "runtime_tasks" / "TOFF_VIEW_EXISTING__seed_101"
        ),
    )
    assert prepared["action"] is None
    register_existing_teacher_source(**prepared["kwargs"])
    registration = Path(prepared["artifact_paths"][0])
    lineage = Path(prepared["artifact_paths"][1])
    assert registration.is_file() and lineage.is_file()
    assert '"selectable": false' in registration.read_text(encoding="utf-8")
    assert sha256_file(
        artifact_root
        / "runtime_tasks"
        / "PV_SOURCE_PREFLIGHT__seed_101"
        / "source_cache_audit.json"
    ) in lineage.read_text(encoding="utf-8")


def test_stage_a_teacher_task_specs_exclude_unresolved_direct_controls(tmp_path):
    _, unified, data = _runtime_sources(tmp_path / "sources")
    config = build_baseline_factory_config(
        runtime_data_config=data,
        device="cpu",
    )
    config_path = tmp_path / "baseline_factory_config.json"
    write_immutable_json(config_path, config)
    specs = build_stage_a_teacher_task_specs(
        factory_config_path=config_path
    )
    assert set(specs) == {
        "A0_VIEW",
        "TOFF_VIEW_BASE",
        "TOFF_VIEW_LARGE",
        "TOFF_VIEW_EXISTING",
    }
    assert specs["A0_VIEW"]["operation"] == "teacher_training"
    assert (
        specs["TOFF_VIEW_EXISTING"]["operation"]
        == "existing_teacher_registration"
    )
    assert "STAGE_A_PARAMETER_MATCH" not in specs

    artifact_root = tmp_path / "campaign"
    _publish_source_parent_result(artifact_root)
    unavailable = build_baseline_factory(
        operation="existing_teacher_registration",
        config=config,
        registry=_stage_a_registry(unified),
        run_id="TOFF_VIEW_EXISTING",
        seed=101,
        task_id="TOFF_VIEW_EXISTING__seed_101",
        output_dir=str(
            artifact_root / "runtime_tasks" / "TOFF_VIEW_EXISTING__seed_101"
        ),
    )
    register_existing_teacher_source(**unavailable["kwargs"])
    text = Path(unavailable["artifact_paths"][0]).read_text(encoding="utf-8")
    assert '"selection_status": "diagnostic_unavailable"' in text
    assert '"warning_is_non_gating": true' in text


def test_baseline_factory_config_cli_publishes_teacher_specs(tmp_path):
    _, _, data = _runtime_sources(tmp_path / "sources")
    data_path = tmp_path / "runtime_data_config.json"
    write_immutable_json(data_path, data)
    config_path = tmp_path / "baseline_factory_config.json"
    specs_path = tmp_path / "teacher_task_specs.json"
    result = subprocess.run(
        [
            str(Path(".venv/Scripts/python.exe")),
            "scripts/build_particle_view_baseline_factory_config.py",
            "--runtime-data-config",
            str(data_path),
            "--output",
            str(config_path),
            "--task-specs-output",
            str(specs_path),
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-val-batches",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert config_path.is_file() and specs_path.is_file()
    assert "trained_teacher_roles=3" in result.stdout


def test_stage_a_direct_resource_plan_profiles_all_candidates_without_gating():
    first = build_stage_a_direct_resource_plan()
    second = build_stage_a_direct_resource_plan()
    assert first == second
    audit = validate_stage_a_direct_resource_plan(first)
    assert audit["candidate_count"] == 16
    assert first["canonical_target"]["view_dim"] == 4
    assert (
        first["canonical_target"]["predictor_architecture_id"]
        == "P_HIER_DECODER_REFINE"
    )
    assert first["canonical_target"]["deployed_parameters"] == sum(
        first["canonical_target"]["parameter_breakdown"].values()
    )
    for quantity in ("parameters", "flops"):
        selection = first["selections"][quantity]
        assert selection["selected"]["config_id"].startswith("w")
        assert selection["warning_is_non_gating"] is True
        if selection["quality_warning"] is not None:
            assert selection["quality_warning"] == "WARN_CONTROL_MATCH_TOLERANCE"


def test_stage_a_direct_factory_binds_selection_source_and_hlt_loaders(tmp_path):
    _, unified, data = _runtime_sources(tmp_path / "sources")
    artifact_root = tmp_path / "campaign"
    _publish_source_parent_result(artifact_root)
    resource_plan = build_stage_a_direct_resource_plan()
    config = build_direct_control_factory_config(
        runtime_data_config=data,
        resource_plan=resource_plan,
        device="cpu",
        num_workers=0,
        max_train_batches=1,
        max_val_batches=1,
    )
    prepared = build_direct_control_factory(
        operation="direct_control_training",
        config=config,
        registry=_stage_a_registry(unified),
        run_id="STAGE_A_PARAMETER_MATCH",
        seed=303,
        task_id="STAGE_A_PARAMETER_MATCH__seed_303",
        output_dir=str(
            artifact_root
            / "runtime_tasks"
            / "STAGE_A_PARAMETER_MATCH__seed_303"
        ),
    )
    recipe = prepared["kwargs"]["recipe"]
    assert recipe.seed == 303
    assert recipe.selection["requested_quantity"] == "parameters"
    assert recipe.resource_plan_sha256 == resource_plan["content_hash"]
    assert prepared["kwargs"]["train_loader"].batch_size == 128
    assert prepared["kwargs"]["config"].max_train_batches == 1
    assert prepared["action"] is None

    config_path = tmp_path / "direct_factory_config.json"
    write_immutable_json(config_path, config)
    specs = build_stage_a_direct_task_specs(
        factory_config_path=config_path
    )
    assert set(specs) == {
        "STAGE_A_PARAMETER_MATCH",
        "STAGE_A_FLOP_MATCH",
    }
    assert all(
        row["operation"] == "direct_control_training"
        for row in specs.values()
    )


def test_direct_control_trainer_smoke_publishes_hlt_only_registration(tmp_path):
    torch = pytest.importorskip("torch")
    parent = miniature_parent_manifest(rows_per_class=2)
    unified = build_unified_split_manifest(
        parent, config=miniature_split_config(rows_per_class=2)
    )
    plan = build_stage_a_direct_resource_plan()
    recipe = build_direct_control_recipe(
        run_id="STAGE_A_FLOP_MATCH",
        seed=101,
        resource_plan=plan,
        unified_split_manifest=unified,
        preprocessing_sha256="1" * 64,
        source_sha256="2" * 64,
        library_versions_sha256="3" * 64,
    )

    class TinyDirect(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(17, 10)

        def forward(self, points, features, lorentz_vectors, mask):
            del points, lorentz_vectors
            valid = mask.float()
            pooled = (features * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1)
            return self.head(pooled)

    batch = {
        "points": torch.randn(2, 2, 5),
        "features": torch.randn(2, 17, 5),
        "lorentz_vectors": torch.randn(2, 4, 5),
        "mask": torch.ones(2, 1, 5, dtype=torch.bool),
        "labels": torch.tensor([0, 1]),
    }

    class Loader:
        batch_size = 128

        def __len__(self):
            return 1

        def __iter__(self):
            return iter((batch,))

    report = train_direct_hlt_control(
        recipe=recipe,
        train_loader=Loader(),
        model_val_stop_loader=Loader(),
        config=DirectControlTrainConfig(
            output_dir=str(tmp_path / "direct_train"),
            device="cpu",
            max_train_batches=1,
            max_val_batches=1,
            amp=False,
        ),
        model=TinyDirect(),
    )
    assert report["status"] == "COMPLETE"
    registration = (
        tmp_path / "direct_train" / "direct_control_registration.json"
    )
    text = registration.read_text(encoding="utf-8")
    assert '"hlt_only_inference": true' in text
    assert '"privileged_inputs": false' in text


def test_direct_control_factory_cli_writes_resource_plan_config_and_specs(
    tmp_path,
):
    _, _, data = _runtime_sources(tmp_path / "sources")
    data_path = tmp_path / "runtime_data_config.json"
    write_immutable_json(data_path, data)
    plan = build_stage_a_direct_resource_plan()
    input_plan = tmp_path / "input_resource_plan.json"
    write_immutable_json(input_plan, plan)
    output_plan = tmp_path / "published_resource_plan.json"
    config_path = tmp_path / "direct_factory_config.json"
    specs_path = tmp_path / "direct_task_specs.json"
    result = subprocess.run(
        [
            str(Path(".venv/Scripts/python.exe")),
            "scripts/build_particle_view_direct_control_factory_config.py",
            "--runtime-data-config",
            str(data_path),
            "--resource-plan",
            str(input_plan),
            "--resource-plan-output",
            str(output_plan),
            "--output",
            str(config_path),
            "--task-specs-output",
            str(specs_path),
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-val-batches",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output_plan.is_file() and config_path.is_file() and specs_path.is_file()
    assert "candidates=16" in result.stdout

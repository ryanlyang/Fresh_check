from __future__ import annotations

import json
from pathlib import Path

import pytest

from jetclass_fresh.jetclass_data import (
    DEFAULT_SPLIT_SEEDS,
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    FileRecord,
    JetIdentity,
    SplitManifest,
    load_split_manifest,
    manifest_hash,
    save_split_manifest,
)
from teacher_logit_reco.relation_expert_token_bridge import (
    RetbSplitConfig,
    authorize_dataset_access,
    build_global_determinism,
    build_registries,
    build_step1_bundle,
    build_storage_measurements,
    event_rng_seed,
    miniature_storage_measurements,
    optimizer_update_counts,
    publish_step1_bundle,
    replica_for,
    resolve_run_id,
    validate_content_hash,
    validate_step1_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    SEMANTIC_CONTROL_POLICY,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (
    EXPERT_ORDER,
    TOKEN_SHAPES,
    validate_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.splits import (
    build_scale_train_manifest,
    build_validation_partition,
    validate_source_split_manifest,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (
    validate_campaign_source,
)


def _source() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }


def _manifest(*, reverse_model_val: bool = False) -> tuple[SplitManifest, RetbSplitConfig]:
    config = RetbSplitConfig.miniature()
    splits: dict[str, list[JetIdentity]] = {
        name: []
        for name in (
            "model_train",
            "model_val",
            "stack_train",
            "stack_val",
            "final_test",
        )
    }
    records: list[FileRecord] = []
    for label in range(10):
        path = f"class_{label}.root"
        records.append(FileRecord(path=path, label=label, num_entries=9))
        splits["model_train"].extend(
            JetIdentity(path, entry, label) for entry in (0, 1)
        )
        validation = [JetIdentity(path, entry, label) for entry in (2, 3)]
        if reverse_model_val:
            validation.reverse()
        splits["model_val"].extend(validation)
        splits["stack_val"].append(JetIdentity(path, 4, label))
        splits["final_test"].extend(
            JetIdentity(path, entry, label) for entry in (5, 6)
        )
    return (
        SplitManifest(
            data_dir="fixture",
            max_constits=128,
            class_names=list(LABEL_NAMES),
            file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
            split_sizes=dict(config.split_sizes),
            split_seeds=dict(DEFAULT_SPLIT_SEEDS),
            file_records=records,
            splits=splits,
        ),
        config,
    )


def _bundle() -> tuple[SplitManifest, RetbSplitConfig, dict]:
    manifest, config = _manifest()
    bundle = build_step1_bundle(
        campaign_id="retb_test",
        manifest=manifest,
        split_config=config,
        source_snapshot=_source(),
        storage_measurements=miniature_storage_measurements(),
    )
    return manifest, config, bundle


def test_split_roles_counts_balance_and_scale_pool_are_exact() -> None:
    manifest, config, bundle = _bundle()
    audit = validate_source_split_manifest(manifest, config=config)
    assert audit["split_counts"] == {
        "model_train": 20,
        "model_val": 20,
        "stack_train": 0,
        "stack_val": 10,
        "final_test": 20,
    }
    validation = bundle["validation_partition_manifest"]
    assert validation["counts"] == {"val_stop": 10, "val_design": 10}
    stop = {
        JetIdentity.from_dict(row).key() for row in validation["roles"]["val_stop"]
    }
    design = {
        JetIdentity.from_dict(row).key()
        for row in validation["roles"]["val_design"]
    }
    assert stop.isdisjoint(design)
    assert stop | design == {row.key() for row in manifest.splits["model_val"]}

    scale = bundle["scale_train_manifest"]
    scale_rows = [JetIdentity.from_dict(row) for row in scale["identities"]]
    scale_keys = {row.key() for row in scale_rows}
    assert scale["count"] == 40
    assert {row.key() for row in manifest.splits["model_train"]} <= scale_keys
    held_out = {
        row.key()
        for split in ("model_val", "stack_val", "final_test")
        for row in manifest.splits[split]
    }
    assert scale_keys.isdisjoint(held_out)
    assert bundle["scale_train_audit"]["duplicate_count"] == 0


def test_production_profile_freezes_requested_counts() -> None:
    config = RetbSplitConfig.production()
    assert dict(config.split_sizes) == {
        "model_train": 500_000,
        "model_val": 100_000,
        "stack_train": 0,
        "stack_val": 50_000,
        "final_test": 300_000,
    }
    assert config.val_stop_per_class == 5_000
    assert config.val_design_per_class == 5_000
    assert config.scale_train_per_class == 300_000


def test_validation_partition_is_input_order_invariant() -> None:
    first, config = _manifest(reverse_model_val=False)
    second, _ = _manifest(reverse_model_val=True)
    first_artifact = build_validation_partition(
        first,
        source_manifest_sha256=manifest_hash(first),
        config=config,
    )
    second_artifact = build_validation_partition(
        second,
        source_manifest_sha256=manifest_hash(second),
        config=config,
    )
    assert first_artifact["identity_hashes"] == second_artifact["identity_hashes"]
    assert first_artifact["roles"] == second_artifact["roles"]


def test_scale_selection_is_deterministic_and_fails_when_capacity_is_absent() -> None:
    manifest, config = _manifest()
    first = build_scale_train_manifest(
        manifest,
        source_manifest_sha256=manifest_hash(manifest),
        config=config,
    )
    second = build_scale_train_manifest(
        manifest,
        source_manifest_sha256=manifest_hash(manifest),
        config=config,
    )
    assert first == second
    manifest.file_records[0] = FileRecord("class_0.root", 0, 7)
    with pytest.raises(ValueError, match="insufficient unused"):
        build_scale_train_manifest(
            manifest,
            source_manifest_sha256=manifest_hash(manifest),
            config=config,
        )


def test_final_select_labels_are_separate_and_selector_only() -> None:
    _, _, bundle = _bundle()
    artifact = bundle["final_select_label_manifest"]
    assert artifact["role"] == "stage_n_selector_only"
    assert artifact["feature_access_allowed"] is False
    assert artifact["selection_inference_access_allowed"] is False
    assert set(artifact["rows"][0]) == {"identity", "label"}
    assert artifact["rows"] == sorted(
        artifact["rows"], key=lambda row: (row["identity"], row["label"])
    )


def test_replica_cycle_covers_all_realizations_without_label_multiplication() -> None:
    identity = "sample.root#42"
    cycle = [
        replica_for(
            policy="R_MULTI",
            logical_role="model_train",
            epoch=epoch,
            canonical_identity=identity,
        )
        for epoch in range(4)
    ]
    assert sorted(cycle) == [0, 1, 2, 3]
    assert {
        replica_for(
            policy="R_RANDOM",
            logical_role="scale_train",
            epoch=epoch,
            canonical_identity=identity,
        )
        for epoch in range(4)
    } == {0, 1, 2, 3}
    for role in ("val_stop", "val_design", "stack_val", "final_test"):
        assert (
            replica_for(
                policy="R_MULTI",
                logical_role=role,
                epoch=39,
                canonical_identity=identity,
            )
            == 0
        )
    assert event_rng_seed(
        logical_role="model_train", replica_id=2, canonical_identity=identity
    ) == event_rng_seed(
        logical_role="scale_train", replica_id=2, canonical_identity=identity
    )


def test_all_static_registries_and_run_ids_are_frozen() -> None:
    registries = build_registries()
    assert len(registries) == 18
    assert tuple(registries["expert_registry"]["canonical_order"]) == EXPERT_ORDER
    assert registries["token_shape_registry"]["shapes"] == TOKEN_SHAPES
    assert registries["campaign_stage_registry"]["stage_order"] == list(
        "ABCDEFGHIJKLMN"
    )
    assert (
        registries["deployed_graph_registry"][
            "performance_failure_blocks_future_runs"
        ]
        is False
    )
    for name, artifact in registries.items():
        assert validate_registry(artifact, name=name) == artifact["content_hash"]

    arguments = {
        "stage": "G",
        "component": "PREDICTOR",
        "seed": 101,
        "configuration": {
            "expert": "PT",
            "shape": "S8_128",
            "predictor": "A3_SLOT_DECODER_DIRECT",
        },
    }
    first = resolve_run_id(**arguments)
    assert first == resolve_run_id(**arguments)
    assert first != resolve_run_id(
        **{**arguments, "configuration": {**arguments["configuration"], "expert": "TRACK"}}
    )
    with pytest.raises(ValueError, match="not registered"):
        resolve_run_id(
            stage="G",
            component="UNDECLARED",
            seed=101,
            configuration={},
        )


def test_determinism_contract_freezes_metrics_and_short_schedule() -> None:
    contract = build_global_determinism()
    assert contract["paired_bootstrap"]["seed"] == 917_301
    assert contract["ece"]["edges"] == [index / 15 for index in range(16)]
    assert contract["qcd_signal_rejection"]["discriminant"] == (
        "p_signal/(p_signal+p_QCD)"
    )
    assert contract["scientific_performance_failure_stops_run"] is False
    one = optimizer_update_counts(
        training_event_count=1,
        maximum_epochs=1,
        microbatch_size=128,
        gradient_accumulation_steps=1,
    )
    assert one == {
        "microbatches_per_epoch": 1,
        "optimizer_updates_per_epoch": 1,
        "total_optimizer_updates": 1,
        "warmup_updates": 1,
    }
    ordinary = optimizer_update_counts(
        training_event_count=500,
        maximum_epochs=40,
        microbatch_size=128,
        gradient_accumulation_steps=2,
    )
    assert ordinary["warmup_updates"] == 4


def test_campaign_parents_source_and_tamper_detection() -> None:
    manifest, _, bundle = _bundle()
    digest = validate_step1_bundle(bundle, manifest=manifest)
    assert digest == bundle["campaign_spec"]["content_hash"]
    spec = bundle["campaign_spec"]
    assert spec["parent_artifact_hashes"]["split_manifest"] == manifest_hash(manifest)
    assert spec["source"]["commit"] == "a" * 40
    assert spec["access_policy"]["performance_based_run_termination"] is False
    assert spec["contract"] == "retb_campaign_spec_v2"
    assert spec["semantic_control_policy"] == SEMANTIC_CONTROL_POLICY
    assert set(spec["registry_hashes"]) == set(bundle["registries"])

    changed = dict(bundle["global_determinism"])
    changed["scientific_performance_failure_stops_run"] = True
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_content_hash(changed)


def test_source_and_dataset_access_authorization_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, bundle = _bundle()
    monkeypatch.setattr(
        "teacher_logit_reco.relation_expert_token_bridge.workflow.source_snapshot",
        lambda _root: _source(),
    )
    assert validate_campaign_source(bundle["campaign_spec"], repo_root=".")[
        "commit"
    ] == "a" * 40
    monkeypatch.setattr(
        "teacher_logit_reco.relation_expert_token_bridge.workflow.source_snapshot",
        lambda _root: {**_source(), "source_status_sha256": "c" * 64},
    )
    with pytest.raises(ValueError, match="source snapshot differs"):
        validate_campaign_source(bundle["campaign_spec"], repo_root=".")

    authorize_dataset_access(
        worker_role="training_worker", requested_resource="val_stop"
    )
    with pytest.raises(PermissionError):
        authorize_dataset_access(
            worker_role="training_worker", requested_resource="val_design"
        )
    with pytest.raises(PermissionError):
        authorize_dataset_access(
            worker_role="stage_n_selection_inference",
            requested_resource="final_select_label_manifest",
        )


def test_publish_is_atomic_idempotent_and_complete(tmp_path: Path) -> None:
    manifest, _, bundle = _bundle()
    root = tmp_path / "campaign"
    first = publish_step1_bundle(
        campaign_root=root,
        manifest=manifest,
        bundle=bundle,
    )
    second = publish_step1_bundle(
        campaign_root=root,
        manifest=manifest,
        bundle=bundle,
    )
    assert first["campaign_spec_sha256"] == second["campaign_spec_sha256"]
    assert second["publications"]["campaign_spec"]["status"] == "already_present"
    loaded_manifest = load_split_manifest(root / "inputs" / "split_manifest.json.gz")
    assert manifest_hash(loaded_manifest) == manifest_hash(manifest)
    assert (
        load_hashed_json(root / "campaign_spec.json")["content_hash"]
        == bundle["campaign_spec"]["content_hash"]
    )
    assert (root / "inputs" / "final_select_label_manifest.json.gz").is_file()
    assert (root / "registry" / "run_id_registry.json").is_file()
    assert (root / "selection_predictions" / "stack_val").is_dir()

    raw = json.loads((root / "campaign_spec.json").read_text(encoding="utf-8"))
    raw["campaign_id"] = "different"
    (root / "campaign_spec.json").write_text(
        json.dumps(raw, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_step1_bundle(
            campaign_root=root,
            manifest=manifest,
            bundle=bundle,
        )


def test_storage_measurements_require_complete_authenticated_production_evidence(
    tmp_path: Path,
) -> None:
    mini = miniature_storage_measurements()
    measurements = dict(mini["measurements"])
    with pytest.raises(ValueError, match="source-evidence"):
        build_storage_measurements(
            measurements=measurements,
            evidence_hashes={},
            measurement_profile="production_source_evidence",
        )
    evidence = tmp_path / "representative.npz"
    evidence.write_bytes(b"measured format")
    artifact = build_storage_measurements(
        measurements=measurements,
        source_evidence={
            "sample": {
                "path": str(evidence),
                "purpose": "representative_compressed_input",
            }
        },
        measurement_profile="production_source_evidence",
    )
    assert validate_content_hash(artifact) == artifact["content_hash"]
    evidence.write_bytes(b"drifted")
    from teacher_logit_reco.relation_expert_token_bridge.storage import (
        validate_storage_measurements,
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_storage_measurements(artifact)


def test_invalid_split_overlap_or_balance_fails_closed() -> None:
    manifest, config = _manifest()
    manifest.splits["stack_val"][0] = manifest.splits["model_train"][0]
    with pytest.raises(ValueError, match="RETB split validation failed"):
        validate_source_split_manifest(manifest, config=config)


def test_campaign_cli_supports_dry_run_and_publication(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.build_retb_campaign import main

    manifest, _ = _manifest()
    parent = tmp_path / "source_manifest.json.gz"
    output = tmp_path / "campaign"
    save_split_manifest(manifest, parent)
    common = [
        "--parent-manifest",
        str(parent),
        "--output-dir",
        str(output),
        "--campaign-id",
        "cli_mini",
        "--miniature",
    ]
    assert main([*common, "--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["dry_run"] is True
    assert not output.exists()
    assert main(common) == 0
    published = json.loads(capsys.readouterr().out)
    assert published["dry_run"] is False
    assert (output / "campaign_spec.json").is_file()


def test_storage_measurement_cli_authenticates_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.measure_retb_storage import main
    from teacher_logit_reco.relation_expert_token_bridge.storage import (
        STORAGE_MEASUREMENTS_CONTRACT,
    )

    measurement_values = miniature_storage_measurements()["measurements"]
    measurement_path = tmp_path / "measurements.json"
    evidence = tmp_path / "representative.bin"
    output = tmp_path / "storage_measurements.json"
    measurement_path.write_text(
        json.dumps(measurement_values), encoding="utf-8"
    )
    evidence.write_bytes(b"representative format bytes")
    assert (
        main(
            [
                "--measurements-json",
                str(measurement_path),
                "--evidence",
                f"sample={evidence}",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    artifact = load_hashed_json(
        output, expected_contract=STORAGE_MEASUREMENTS_CONTRACT
    )
    assert result["content_hash"] == artifact["content_hash"]
    assert artifact["source_evidence"]["sample"]["bytes"] == evidence.stat().st_size

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from jetclass_fresh.hlt_cache import (
    HLT_ARRAY_FILENAME,
    HLT_METADATA_FILENAME,
    hash_arrays,
    jet_identity_hash,
)
from jetclass_fresh.jetclass_data import (
    DEFAULT_SPLIT_SEEDS,
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    SPLIT_ORDER,
    JetIdentity,
    SplitManifest,
    save_split_manifest,
)
from teacher_logit_reco.relational_part import (
    CAMPAIGN_SPEC_CONTRACT,
    CANONICAL_FAMILY_ORDER,
    GIB,
    GLOBAL_DETERMINISM_CONTRACT,
    LOCKED_HLT_SEEDS,
    PRODUCTION_SPLIT_SIZES,
    RELATION_FAMILY_REGISTRY_CONTRACT,
    SCREENING_REGISTRY_CONTRACT,
    RAW_INPUT_SCHEMA_CONTRACT,
    RelationalSplitConfig,
    StorageMeasurements,
    build_confirmation_architecture_registry,
    build_global_determinism_contract,
    build_hlt_binding,
    build_hlt_expectation,
    build_relation_family_registry,
    build_raw_input_schema_contract,
    build_screening_registry,
    build_semantic_control_registry,
    build_split_binding,
    build_step1_bundle,
    build_storage_measurements,
    build_storage_projection,
    optimizer_update_counts,
    publish_step1_bundle,
    resolve_registered_run,
    scheduled_learning_rate,
    validate_content_hash,
    validate_global_determinism_contract,
    validate_relational_split_manifest,
    validate_screening_registry,
    with_content_hash,
)


def _manifest(
    config: RelationalSplitConfig | None = None,
) -> tuple[SplitManifest, RelationalSplitConfig]:
    config = config or RelationalSplitConfig.miniature()
    splits: dict[str, list[JetIdentity]] = {}
    offset = 0
    for split in SPLIT_ORDER:
        per_class = int(config.split_sizes[split]) // len(LABEL_NAMES)
        rows: list[JetIdentity] = []
        for label in range(len(LABEL_NAMES)):
            for local in range(per_class):
                rows.append(
                    JetIdentity(
                        file=f"/data/class_{label}.root",
                        entry=offset + label * 1000 + local,
                        label=label,
                    )
                )
        splits[split] = rows
        offset += 100_000
    manifest = SplitManifest(
        data_dir="/data",
        max_constits=128,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes=config.normalized_sizes(),
        split_seeds=dict(DEFAULT_SPLIT_SEEDS),
        file_records=[],
        splits=splits,
        metadata={"purpose": "relational_step1_fixture"},
    )
    return manifest, config


def _measurements() -> StorageMeasurements:
    return StorageMeasurements(
        hlt_sample_jets=100,
        hlt_sample_bytes=10_000,
        tree_sample_jets=100,
        tree_sample_bytes=2_000,
        checkpoint_sample_count=1,
        checkpoint_sample_bytes=1_000_000,
        prediction_sample_events=100,
        prediction_sample_bytes=4_000,
        fixed_overhead_bytes=1_000_000,
    )


def _measurement_artifact() -> dict:
    measurements = _measurements()
    return build_storage_measurements(
        measurements,
        source_evidence={
            "hlt_cache": {
                "path": "representative_hlt_cache.npz",
                "sha256": "1" * 64,
                "bytes": measurements.hlt_sample_bytes,
                "purpose": "compressed HLT bytes-per-jet sample",
            },
            "tree_sidecar": {
                "path": "representative_tree_sidecar.npz",
                "sha256": "2" * 64,
                "bytes": measurements.tree_sample_bytes,
                "purpose": "locked tree-probe bytes-per-jet sample",
            },
            "checkpoint": {
                "path": "representative_checkpoint.pt",
                "sha256": "3" * 64,
                "bytes": measurements.checkpoint_sample_bytes,
                "purpose": "retained best-checkpoint sample",
            },
            "predictions": {
                "path": "representative_predictions.npz",
                "sha256": "4" * 64,
                "bytes": measurements.prediction_sample_bytes,
                "purpose": "final prediction bytes-per-event sample",
            },
        },
    )


def _write_hlt_cache(
    root: Path,
    manifest: SplitManifest,
    split_binding: dict,
    expectation: dict,
) -> None:
    root.mkdir(parents=True)
    for split in expectation["required_splits"]:
        rows = manifest.splits[split]
        count = len(rows)
        tokens = np.zeros((count, 128, 14), dtype=np.float32)
        mask = np.zeros((count, 128), dtype=bool)
        mask[:, 0] = True
        labels = np.asarray([row.label for row in rows], dtype=np.int64)
        jet_files: list[str] = []
        file_index: dict[str, int] = {}
        file_indices = np.empty((count,), dtype=np.int32)
        entries = np.asarray([row.entry for row in rows], dtype=np.int64)
        for index, row in enumerate(rows):
            if row.file not in file_index:
                file_index[row.file] = len(jet_files)
                jet_files.append(row.file)
            file_indices[index] = file_index[row.file]
        arrays = {
            "tokens": tokens,
            "mask": mask,
            "labels": labels,
            "jet_file_indices": file_indices,
            "jet_entries": entries,
        }
        np.savez_compressed(
            root / HLT_ARRAY_FILENAME.format(split=split),
            **arrays,
        )
        metadata = {
            "version": 1,
            "view": "fixed_hlt",
            "split": split,
            "seed": LOCKED_HLT_SEEDS[split],
            "hlt_profile": expectation["hlt_profile"],
            "hlt_profile_version": expectation["hlt_profile_version"],
            "hlt_degradation_strength": expectation["hlt_degradation_strength"],
            "hlt_params": expectation["hlt_params"],
            "source_manifest_hash": split_binding["source_manifest_hash"],
            "source_view": "offline",
            "max_constits": 128,
            "raw_token_dim": 14,
            "n_jets": count,
            "jet_files": jet_files,
            "jet_identity_hash": jet_identity_hash(rows),
            "generator": {
                "module": "jetclass_fixed_hlt",
                "function": "build_fixed_hlt_view",
                "params_class": "FixedHLTParams",
            },
            "source_content_hash": "1" * 64,
            "hlt_content_hash": hash_arrays(arrays),
            "diagnostics_hash": hash_arrays({}),
            "generator": {
                "module": "jetclass_fixed_hlt",
                "function": "build_fixed_hlt_view",
                "params_class": "FixedHLTParams",
            },
        }
        (root / HLT_METADATA_FILENAME.format(split=split)).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_production_split_and_hlt_contract_are_exact() -> None:
    config = RelationalSplitConfig.production()
    assert config.normalized_sizes() == PRODUCTION_SPLIT_SIZES
    assert config.max_constituents == 128
    manifest, mini = _manifest()
    binding = build_split_binding(manifest, config=mini)
    expectation = build_hlt_expectation(
        split_binding_sha256=binding["content_hash"]
    )
    assert expectation["required_splits"] == [
        "model_train",
        "model_val",
        "stack_val",
        "final_test",
    ]
    assert expectation["forbidden_splits"] == ["stack_train"]
    assert expectation["seeds"] == LOCKED_HLT_SEEDS
    assert expectation["hlt_degradation_strength"] == 0.6
    assert expectation["raw_token_dimension"] == 14
    assert expectation["derived_part_particle_input_dimension"] == 17


@pytest.mark.parametrize(
    "failure", ["imbalance", "overlap", "count", "seed", "negative_entry"]
)
def test_split_audit_fails_closed(failure: str) -> None:
    manifest, config = _manifest()
    if failure == "imbalance":
        manifest.splits["model_train"][0] = JetIdentity(
            file="/data/class_0.root", entry=0, label=1
        )
    elif failure == "overlap":
        manifest.splits["model_val"][0] = manifest.splits["model_train"][0]
    elif failure == "count":
        manifest.splits["stack_val"].pop()
    elif failure == "negative_entry":
        original = manifest.splits["model_train"][0]
        manifest.splits["model_train"][0] = JetIdentity(
            file=original.file, entry=-1, label=original.label
        )
    else:
        manifest.split_seeds["model_train"] += 1
    with pytest.raises(ValueError, match="split provenance"):
        validate_relational_split_manifest(manifest, config=config)


def test_hlt_binding_authenticates_metadata_identities_and_arrays(
    tmp_path: Path,
) -> None:
    manifest, config = _manifest()
    binding = build_split_binding(manifest, config=config)
    expectation = build_hlt_expectation(
        split_binding_sha256=binding["content_hash"]
    )
    cache = tmp_path / "cache"
    _write_hlt_cache(cache, manifest, binding, expectation)
    result = build_hlt_binding(
        cache_dir=cache,
        manifest=manifest,
        split_binding=binding,
        hlt_expectation=expectation,
    )
    assert result["ok"] is True
    assert set(result["split_reports"]) == set(expectation["required_splits"])

    metadata_path = cache / HLT_METADATA_FILENAME.format(split="model_val")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_manifest_hash"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="source_manifest_hash"):
        build_hlt_binding(
            cache_dir=cache,
            manifest=manifest,
            split_binding=binding,
            hlt_expectation=expectation,
        )


def test_registries_are_complete_hashed_and_training_independent() -> None:
    relations = build_relation_family_registry()
    screening = build_screening_registry(
        relation_registry_sha256=relations["content_hash"]
    )
    confirmation = build_confirmation_architecture_registry(
        relation_registry_sha256=relations["content_hash"],
        screening_registry_sha256=screening["content_hash"],
    )
    semantic = build_semantic_control_registry(
        relation_registry_sha256=relations["content_hash"],
        confirmation_registry_sha256=confirmation["content_hash"],
    )
    assert relations["canonical_family_order"] == list(CANONICAL_FAMILY_ORDER)
    assert relations["contract"] == RELATION_FAMILY_REGISTRY_CONTRACT
    assert relations["schema_version"] == 4
    region = next(
        row for row in relations["families"] if row["family_id"] == "REGION"
    )
    assert region["raw_feature_groups"] == {
        "same_cluster_indicators_K2_K4_K8": 3,
        "lca_depth": 1,
        "lca_merge": 4,
        "endpoint_cluster_descriptors": 18,
        "within_cluster_particle_pt_fractions": 6,
        "endpoint_to_axis_distances": 6,
        "signed_cluster_pt_rank_differences": 3,
    }
    assert sum(region["raw_feature_groups"].values()) == region["raw_dimension"] == 41
    assert screening["contract"] == SCREENING_REGISTRY_CONTRACT
    assert screening["schema_version"] == 2
    assert screening["wide_capacity_search"]["tie_breaks"] == [
        "minimum_absolute_incremental_parameter_mismatch",
        "lower_analytically_calculated_pair_encoder_FLOPs_at_128_valid_particles",
        "smaller_width_sum",
        "smaller_width_tuple_lexicographically",
    ]
    assert (
        screening["wide_capacity_search"]["flops_source"]
        == "locked_exact_symbolic_pair_encoder_formula"
    )
    assert screening["row_count"] == 21
    assert len({row["run_id"] for row in screening["rows"]}) == 21
    assert sum(row["relational_selection_eligible"] for row in screening["rows"]) == 18
    assert {
        row["configuration_role"] for row in screening["rows"]
    } == {"reference_baseline", "capacity_control", "scientific_finalist"}
    assert resolve_registered_run(
        "RPT_PT_TRACK",
        screening_registry=screening,
    )["new_relation_families"] == ["PT", "TRACK"]
    assert resolve_registered_run(
        "RPT_SELECTED_UNARY",
        screening_registry=screening,
        confirmation_registry=confirmation,
        semantic_registry=semantic,
    )["configuration_role"] == "semantic_control"
    for artifact in (relations, screening, confirmation, semantic):
        assert validate_content_hash(artifact) == artifact["content_hash"]


def test_corrected_registry_versions_reject_legacy_contracts() -> None:
    relations = build_relation_family_registry()
    legacy_relations = dict(relations)
    legacy_relations.pop("content_hash")
    legacy_relations["contract"] = "relational_part_relation_family_registry_v1"
    legacy_relations["schema_version"] = 1
    legacy_relations = with_content_hash(legacy_relations)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_content_hash(
            legacy_relations,
            expected_contract=RELATION_FAMILY_REGISTRY_CONTRACT,
        )

    screening = build_screening_registry(
        relation_registry_sha256=relations["content_hash"]
    )
    legacy_screening = dict(screening)
    legacy_screening.pop("content_hash")
    legacy_screening["contract"] = "relational_part_screening_registry_v1"
    legacy_screening["schema_version"] = 1
    legacy_screening = with_content_hash(legacy_screening)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_screening_registry(legacy_screening)


def test_global_determinism_policy_is_complete_exact_and_not_tunable() -> None:
    policy = build_global_determinism_contract()
    assert policy["contract"] == GLOBAL_DETERMINISM_CONTRACT
    assert validate_global_determinism_contract(policy) == policy["content_hash"]
    assert policy["fixed_before_scientific_results"] is True
    assert policy["model_specific_override_allowed"] is False

    weaver = policy["parity"]["authoritative_weaver_explicit_uu"]
    assert weaver == {
        "device_path": "real_installed_weaver",
        "dtype": "float32",
        "autocast_enabled": False,
        "gradient_scaler_enabled": False,
        "evaluation_mode": True,
        "atol": 1.0e-6,
        "rtol": 1.0e-6,
        "applies_to": [
            "standard_four_pair_features",
            "logits",
            "input_gradients",
            "parameter_gradients",
            "valid_token_padding_invariance",
        ],
        "nonfinite_is_failure": True,
    }
    exact = set(policy["parity"]["exact_fields"])
    assert {
        "integer_tree_topology",
        "particle_masks",
        "categorical_states",
        "event_identities",
    } <= exact

    bootstrap = policy["paired_bootstrap"]
    assert bootstrap["seed"] == 917_301
    assert bootstrap["replicates"] == 10_000
    assert bootstrap["sampling_unit"] == "aligned_event_identity"
    assert bootstrap["seed_reused_for_every_paired_comparison"] is True
    assert bootstrap["stratification"]["draws_per_class"] == (
        "original_event_count_in_that_class"
    )
    assert bootstrap["stratification"]["within_class_source_order"] == (
        "split_manifest_event_identity_order"
    )
    assert bootstrap["interval"] == {
        "kind": "two_sided_percentile",
        "lower_percent": 2.5,
        "upper_percent": 97.5,
        "quantile_method": "Hyndman_Fan_type_7_linear",
        "numpy_method": "linear",
        "bootstrap_mean_used_as_endpoint": False,
    }

    ece = policy["calibration"]["ece"]
    assert ece["kind"] == "top_label_multiclass"
    assert ece["bin_count"] == 15
    assert ece["membership"] == (
        "left_closed_right_open_except_final_bin_closed"
    )
    assert ece["empty_bin_contribution"] == 0.0

    rejection = policy["qcd_signal_rejection"]
    assert rejection["discriminant"] == "logit_signal_minus_logit_QCD"
    assert rejection["threshold"]["pass_rule"] == (
        "score_greater_than_or_equal_to_threshold"
    )
    assert rejection["zero_background_behavior"] == {
        "background_rejection": None,
        "background_rejection_is_infinite": True,
        "qcd_false_positive_rate": 0.0,
        "qcd_false_positive_count": 0,
        "reason": "avoid_nonfinite_JSON_while_preserving_exact_meaning",
    }

    changed = copy.deepcopy(policy)
    changed.pop("content_hash")
    changed["paired_bootstrap"]["seed"] += 1
    changed = with_content_hash(changed)
    with pytest.raises(ValueError, match="differ from the locked"):
        validate_global_determinism_contract(changed)


def test_raw_input_charge_quantization_is_versioned_and_exact() -> None:
    schema = build_raw_input_schema_contract()
    assert schema["contract"] == RAW_INPUT_SCHEMA_CONTRACT
    assert schema["schema_version"] == 2
    assert schema["charge_states"] == [-1, 0, 1]
    assert schema["charge_integer_tolerance"] == 1.0e-6
    assert schema["charge_quantization"] == (
        "nearest_locked_state_after_tolerance_validation"
    )
    legacy = dict(schema)
    legacy.pop("content_hash")
    legacy["contract"] = "relational_part_raw_input_schema_v1"
    legacy["schema_version"] = 1
    legacy.pop("charge_integer_tolerance")
    legacy.pop("charge_quantization")
    legacy = with_content_hash(legacy)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_content_hash(legacy, expected_contract=RAW_INPUT_SCHEMA_CONTRACT)


def test_locked_optimizer_update_and_warmup_integer_rules() -> None:
    schedule = build_global_determinism_contract()["optimizer_update_schedule"]
    assert schedule["accumulation_groups_cross_epoch_boundary"] is False
    assert schedule["dataloader_drop_last"] is False
    assert schedule["gradient_normalization"] == (
        "sum_of_event_losses_divided_by_actual_event_count_"
        "in_accumulation_group"
    )
    assert optimizer_update_counts(training_event_count=0, maximum_epochs=40) == {
        "microbatches_per_epoch": 0,
        "optimizer_updates_per_epoch": 0,
        "total_optimizer_updates": 0,
        "warmup_updates": 0,
    }
    assert optimizer_update_counts(training_event_count=1, maximum_epochs=1) == {
        "microbatches_per_epoch": 1,
        "optimizer_updates_per_epoch": 1,
        "total_optimizer_updates": 1,
        "warmup_updates": 1,
    }
    assert optimizer_update_counts(training_event_count=128, maximum_epochs=1) == {
        "microbatches_per_epoch": 2,
        "optimizer_updates_per_epoch": 1,
        "total_optimizer_updates": 1,
        "warmup_updates": 1,
    }
    production = optimizer_update_counts(
        training_event_count=1_000_000,
        maximum_epochs=40,
    )
    assert production == {
        "microbatches_per_epoch": 15_625,
        "optimizer_updates_per_epoch": 7_813,
        "total_optimizer_updates": 312_520,
        "warmup_updates": 15_626,
    }
    assert scheduled_learning_rate(
        update_ordinal=1,
        total_optimizer_updates=1,
        warmup_updates=1,
        base_lr=1.0e-3,
        minimum_lr=1.0e-5,
    ) == pytest.approx(1.0e-3)
    assert scheduled_learning_rate(
        update_ordinal=15_626,
        total_optimizer_updates=312_520,
        warmup_updates=15_626,
        base_lr=1.0e-3,
        minimum_lr=1.0e-5,
    ) == pytest.approx(1.0e-3)
    assert scheduled_learning_rate(
        update_ordinal=312_520,
        total_optimizer_updates=312_520,
        warmup_updates=15_626,
        base_lr=1.0e-3,
        minimum_lr=1.0e-5,
    ) == pytest.approx(1.0e-5)


def test_storage_projection_is_measured_and_fails_closed() -> None:
    projection = build_storage_projection(
        _measurements(),
        available_bytes=100 * GIB,
        total_hlt_jets=1000,
        total_tree_jets=1000,
        final_test_events=100,
    )
    assert projection["ok"] is True
    assert projection["checks"]["uses_measured_tree_bytes_per_jet"] is True
    with pytest.raises(ValueError, match="storage preflight failed"):
        build_storage_projection(
            _measurements(),
            available_bytes=21 * GIB,
            total_hlt_jets=1_750_000,
            total_tree_jets=1_750_000,
        )


def test_production_bundle_rejects_unbound_storage_estimates() -> None:
    manifest, _ = _manifest()
    with pytest.raises(ValueError, match="source-bound"):
        build_step1_bundle(
            campaign_id="production_rejects_estimate",
            manifest=manifest,
            measurements=_measurements(),
            available_bytes=100 * GIB,
            source_commit="a" * 40,
            source_status_sha256="b" * 64,
        )


def test_storage_measurements_are_source_bound_and_tamper_evident() -> None:
    artifact = _measurement_artifact()
    assert validate_content_hash(artifact) == artifact["content_hash"]
    projection = build_storage_projection(
        artifact,
        available_bytes=100 * GIB,
        total_hlt_jets=1000,
        total_tree_jets=1000,
        final_test_events=100,
    )
    assert projection["measurement_artifact_sha256"] == artifact["content_hash"]

    changed = copy.deepcopy(artifact)
    changed["measurements"]["tree_sample_bytes"] += 1
    with pytest.raises(ValueError, match="content hash mismatch"):
        build_storage_projection(
            changed,
            available_bytes=100 * GIB,
            total_hlt_jets=1000,
            total_tree_jets=1000,
            final_test_events=100,
        )


def test_step1_bundle_is_deterministic_and_publishes_immutably(
    tmp_path: Path,
) -> None:
    manifest, config = _manifest()
    arguments = {
        "campaign_id": "mini_step1",
        "manifest": manifest,
        "measurements": _measurement_artifact(),
        "available_bytes": 100 * GIB,
        "source_commit": "a" * 64,
        "source_status_sha256": "b" * 64,
        "split_config": config,
    }
    first = build_step1_bundle(**arguments)
    second = build_step1_bundle(**arguments)
    assert first["campaign_spec"]["content_hash"] == second["campaign_spec"]["content_hash"]
    assert first["step1_report"]["ready_for_step2"] is True
    assert first["step1_report"]["schema_version"] == 3
    assert first["step1_report"]["screening_row_count"] == 21
    assert first["campaign_spec"]["campaign_profile"] == "nonproduction_miniature_test"
    assert first["campaign_spec"]["contract"] == CAMPAIGN_SPEC_CONTRACT
    assert first["campaign_spec"]["schema_version"] == 3
    assert (
        first["campaign_spec"]["global_determinism"]
        == first["global_determinism"]
    )
    assert first["global_determinism"]["schema_version"] == 3
    assert (
        first["global_determinism"]["attention_diagnostics"][
            "region_resolution_dropouts"
        ]
        == [2, 4, 8]
    )
    assert (
        first["campaign_spec"]["parent_artifact_hashes"]["global_determinism"]
        == first["global_determinism"]["content_hash"]
    )
    assert first["campaign_spec"]["scientific_results_allowed"] is False
    assert (
        first["storage_projection"]["measurement_artifact_sha256"]
        == first["storage_measurements"]["content_hash"]
    )

    root = tmp_path / "campaign"
    published = publish_step1_bundle(
        campaign_root=root, manifest=manifest, bundle=first
    )
    repeated = publish_step1_bundle(
        campaign_root=root, manifest=manifest, bundle=first
    )
    assert published["campaign_spec_sha256"] == first["campaign_spec"]["content_hash"]
    assert repeated["publications"]["campaign_spec"]["status"] == "already_present"
    assert (root / "inputs" / "split_manifest.json.gz").is_file()
    assert (root / "storage_measurements.json").is_file()

    changed = copy.deepcopy(first)
    changed["campaign_spec"] = {
        **changed["campaign_spec"],
        "campaign_id": "different",
    }
    with pytest.raises(ValueError, match="content hash mismatch"):
        publish_step1_bundle(campaign_root=root, manifest=manifest, bundle=changed)

    stale = build_step1_bundle(
        **{**arguments, "source_status_sha256": "c" * 64}
    )
    mixed = copy.deepcopy(first)
    mixed["screening_registry"] = stale["screening_registry"]
    with pytest.raises(ValueError, match="campaign spec parent hashes"):
        publish_step1_bundle(campaign_root=root, manifest=manifest, bundle=mixed)


def test_git_sha1_source_identity_is_supported_for_real_repositories() -> None:
    manifest, config = _manifest()
    bundle = build_step1_bundle(
        campaign_id="sha1_source",
        manifest=manifest,
        measurements=_measurement_artifact(),
        available_bytes=100 * GIB,
        source_commit="a" * 40,
        source_status_sha256="b" * 64,
        source_dirty=False,
        split_config=config,
    )
    assert bundle["campaign_spec"]["source"]["commit"] == "a" * 40
    assert bundle["campaign_spec"]["source"]["dirty"] is False


def test_step1_cli_dry_run_and_nonproduction_publication(tmp_path: Path) -> None:
    manifest, _ = _manifest()
    manifest_path = tmp_path / "source_manifest.json.gz"
    save_split_manifest(manifest, manifest_path)
    measurements = _measurements()
    evidence_specs = {
        "hlt_cache": (
            "representative_hlt_cache.npz",
            measurements.hlt_sample_bytes,
            "compressed HLT bytes-per-jet sample",
        ),
        "tree_sidecar": (
            "representative_tree_sidecar.npz",
            measurements.tree_sample_bytes,
            "locked tree-probe bytes-per-jet sample",
        ),
        "checkpoint": (
            "representative_checkpoint.pt",
            measurements.checkpoint_sample_bytes,
            "retained best-checkpoint sample",
        ),
        "predictions": (
            "representative_predictions.npz",
            measurements.prediction_sample_bytes,
            "final prediction bytes-per-event sample",
        ),
    }
    evidence = {}
    for name, (filename, size, purpose) in evidence_specs.items():
        (tmp_path / filename).write_bytes(b"\0" * size)
        evidence[name] = {"path": filename, "purpose": purpose}
    measurement_path = tmp_path / "measurements.json"
    measurement_path.write_text(
        json.dumps(
            {
                "measurements": measurements.to_dict(),
                "source_evidence": evidence,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "campaign"
    command = [
        sys.executable,
        "scripts/build_relational_part_campaign.py",
        "--parent-manifest",
        str(manifest_path),
        "--output-dir",
        str(output),
        "--campaign-id",
        "mini_cli",
        "--storage-measurements",
        str(measurement_path),
        "--available-bytes",
        str(100 * GIB),
        "--miniature",
    ]
    dry = subprocess.run(
        [*command, "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    dry_payload = json.loads(dry.stdout)
    assert dry_payload["dry_run"] is True
    assert not output.exists()
    assert (
        dry_payload["campaign_spec"]["campaign_profile"]
        == "nonproduction_miniature_test"
    )

    subprocess.run(command, check=True, capture_output=True, text=True)
    published_spec = json.loads(
        (output / "campaign_spec.json").read_text(encoding="utf-8")
    )
    assert published_spec["scientific_results_allowed"] is False

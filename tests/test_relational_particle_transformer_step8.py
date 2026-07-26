from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scripts.build_relational_part_model_contracts import (
    EXPECTED_TRIMMING_DIAGNOSTIC,
    validate_parity_execution_binding,
)

from jetclass_fresh.jetclass_data import (
    DEFAULT_SPLIT_SEEDS,
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    SPLIT_ORDER,
    FileRecord,
    JetIdentity,
    SplitManifest,
)
from teacher_logit_reco.relational_part import (
    ANGULAR_TREE_PROBE_CONTRACT,
    JOB_LEDGER_CONTRACT,
    POSTCONSTRUCTION_AUDIT_CONTRACT,
    PRECONSTRUCTION_AUDIT_CONTRACT,
    RAW_AUDIT_SALT,
    StorageMeasurements,
    build_preconstruction_audit,
    build_production_graph,
    build_raw_input_schema_contract,
    build_storage_projection,
    build_tree_probe_artifact,
    select_raw_audit_identities,
    select_tree_probe,
    validate_existing_tree_shard,
    validate_campaign_source,
    validate_content_hash,
    validate_production_graph,
)


ROOT = Path(__file__).resolve().parents[1]


def test_parity_consumption_requires_exact_source_and_active_trim_fixture() -> None:
    source = {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": True,
    }
    parity = {
        "source": source,
        "sequence_trimming_diagnostic": dict(
            EXPECTED_TRIMMING_DIAGNOSTIC
        ),
    }
    validate_parity_execution_binding(parity, current_source=source)
    with pytest.raises(ValueError, match="source snapshot"):
        validate_parity_execution_binding(
            parity,
            current_source=source | {"source_status_sha256": "c" * 64},
        )
    with pytest.raises(ValueError, match="active trimming"):
        validate_parity_execution_binding(
            parity
            | {
                "sequence_trimming_diagnostic": (
                    EXPECTED_TRIMMING_DIAGNOSTIC
                    | {"ordinary_trimmed_width": 7}
                )
            },
            current_source=source,
        )


def _miniature_manifest() -> SplitManifest:
    files = [
        FileRecord(
            path=f"root/class_{label}.root",
            label=label,
            num_entries=6,
        )
        for label in range(len(LABEL_NAMES))
    ]
    splits = {name: [] for name in SPLIT_ORDER}
    offsets = {
        "model_train": (0, 2),
        "model_val": (2, 3),
        "stack_train": (3, 3),
        "stack_val": (3, 4),
        "final_test": (4, 6),
    }
    for split, (start, stop) in offsets.items():
        for label in range(len(LABEL_NAMES)):
            splits[split].extend(
                JetIdentity(
                    file=files[label].path,
                    entry=entry,
                    label=label,
                )
                for entry in range(start, stop)
            )
    return SplitManifest(
        data_dir="/synthetic",
        max_constits=128,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={name: len(splits[name]) for name in SPLIT_ORDER},
        split_seeds=dict(DEFAULT_SPLIT_SEEDS),
        file_records=files,
        splits=splits,
    )


def _raw_arrays(manifest: SplitManifest, selection: dict, split: str) -> dict:
    indices = selection["selected_indices"][split]
    count = len(indices)
    tokens = np.zeros((count, 128, 14), dtype=np.float32)
    mask = np.zeros((count, 128), dtype=np.bool_)
    labels = np.asarray(
        [manifest.splits[split][index].label for index in indices],
        dtype=np.int64,
    )
    mask[:, 0] = True
    tokens[:, 0, 0] = 1.0
    tokens[:, 0, 3] = 1.0
    tokens[:, 0, 5] = 1.0
    return {"tokens": tokens, "mask": mask, "labels": labels}


def test_step8_production_graph_is_complete_and_final_test_is_sealed() -> None:
    graph = build_production_graph(
        campaign_root="/tmp/rpt_campaign",
        campaign_id="rpt_campaign",
        source_commit="1" * 40,
        source_status_sha256="2" * 64,
        screening_array_concurrency=7,
        tree_array_concurrency=11,
    )
    assert validate_production_graph(graph) == graph["content_hash"]
    assert graph["split_sizes"] == {
        "model_train": 1_000_000,
        "model_val": 125_000,
        "stack_train": 0,
        "stack_val": 125_000,
        "final_test": 500_000,
    }
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    assert nodes["screening"]["array"] == "0-20%7"
    assert nodes["tree_shards_model_train"]["array"] == "0-99%11"
    assert nodes["tree_shards_final_test"]["array"] == "0-49%11"
    assert nodes["screening"]["dependencies"] == [
        "screening_model_contracts"
    ]
    assert nodes["finalist_lock"]["dependencies"] == [
        "semantic_controls",
        "unary_training",
    ]
    assert nodes["final_test_evaluation"]["dependencies"] == [
        "final_test_submit"
    ]
    assert nodes["final_test_evaluation"][
        "final_test_access"
    ] == "sealed_scientific_evaluation"
    assert nodes["tree_shards_final_test"][
        "final_test_access"
    ] == "sealed_preparation_only"
    assert all(
        nodes[name]["final_test_access"] == "none"
        for name in (
            "screening",
            "screening_selection",
            "confirmation_training",
            "confirmation_summary",
            "semantic_controls",
            "unary_training",
            "finalist_lock",
        )
    )


def test_raw_audit_uses_exact_salt_and_locked_miniature_samples() -> None:
    manifest = _miniature_manifest()
    first = select_raw_audit_identities(manifest, miniature=True)
    second = select_raw_audit_identities(manifest, miniature=True)
    assert first == second
    assert first["salt"] == RAW_AUDIT_SALT == "rpt_raw_audit_v1"
    assert first["selected_counts"] == {
        "model_train": 20,
        "model_val": 10,
        "stack_val": 10,
        "final_test": 20,
    }
    schema = build_raw_input_schema_contract()
    inventory = [
        {
            "path": record.path,
            "num_entries": record.num_entries,
            "required_branches": list(
                schema["required_particle_branches"]
            ),
            "shape_policy": "jagged_particle_axis",
            "dtype_policy": "numeric",
        }
        for record in manifest.file_records
    ]
    artifact = build_preconstruction_audit(
        manifest=manifest,
        raw_input_schema=schema,
        branch_inventory=inventory,
        sampled_arrays={
            split: _raw_arrays(manifest, first, split)
            for split in ("model_train", "model_val", "stack_val", "final_test")
        },
        miniature=True,
    )
    assert validate_content_hash(
        artifact, expected_contract=PRECONSTRUCTION_AUDIT_CONTRACT
    ) == artifact["content_hash"]
    assert artifact["final_test_access"] == (
        "sealed_preparation_input_validation_only"
    )
    assert artifact["inference_performed"] is False
    assert artifact["normalizer_fitted"] is False
    assert all(
        report["pid_multi_hot_particle_count"] == 0
        for report in artifact["sample_reports"].values()
    )


def test_raw_audit_rejects_multihot_pid_and_nonzero_padding() -> None:
    manifest = _miniature_manifest()
    selection = select_raw_audit_identities(manifest, miniature=True)
    sampled = {
        split: _raw_arrays(manifest, selection, split)
        for split in ("model_train", "model_val", "stack_val", "final_test")
    }
    sampled["model_train"]["tokens"][0, 0, 6] = 1.0
    schema = build_raw_input_schema_contract()
    inventory = [
        {
            "path": record.path,
            "num_entries": record.num_entries,
            "required_branches": list(
                schema["required_particle_branches"]
            ),
            "shape_policy": "jagged_particle_axis",
            "dtype_policy": "numeric",
        }
        for record in manifest.file_records
    ]
    with pytest.raises(ValueError, match="multi-hot"):
        build_preconstruction_audit(
            manifest=manifest,
            raw_input_schema=schema,
            branch_inventory=inventory,
            sampled_arrays=sampled,
            miniature=True,
        )


def test_tree_probe_v2_authenticates_parents_and_storage() -> None:
    identities = [f"sample/file.root#{index}" for index in range(20)]
    counts = [index % 10 + 1 for index in range(20)]
    selection = select_tree_probe(identities, counts, sample_count=20)
    storage = build_storage_projection(
        StorageMeasurements(
            hlt_sample_jets=1,
            hlt_sample_bytes=1,
            tree_sample_jets=1,
            tree_sample_bytes=1,
            checkpoint_sample_count=1,
            checkpoint_sample_bytes=1,
            prediction_sample_events=1,
            prediction_sample_bytes=1,
            fixed_overhead_bytes=0,
        ),
        available_bytes=100 * 1024**3,
        total_hlt_jets=60,
        total_tree_jets=60,
        retained_checkpoint_count=1,
        retained_final_prediction_sets=1,
        final_test_events=20,
    )
    artifact = build_tree_probe_artifact(
        selection,
        counts,
        np.full(20, 0.01),
        np.full(20, 32),
        peak_resident_bytes=1_000_000,
        parity_topology_exact=True,
        parity_max_continuous_absolute_error=1.0e-8,
        total_campaign_jets=60,
        hlt_content_sha256="1" * 64,
        tree_resource_sha256="2" * 64,
        backend_manifest_sha256="3" * 64,
        storage_projection=storage,
    )
    assert validate_content_hash(
        artifact, expected_contract=ANGULAR_TREE_PROBE_CONTRACT
    ) == artifact["content_hash"]
    assert artifact["schema_version"] == 2
    assert artifact["scientific_provenance_complete"] is True
    assert artifact["storage_projection_sha256"] == storage["content_hash"]
    assert artifact["limits"]["passed"] is True
    assert (
        storage["component_bytes"][
            "peak_concurrent_resumable_last_checkpoints"
        ]
        > 0
    )
    assert sum(artifact["initial_quotas"]) + sum(
        artifact["redistributed_additions"]
    ) == 20


def test_resumable_tree_worker_recovers_only_unregistered_partial(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shards" / "shard_00000.npz"
    shard.parent.mkdir()
    shard.write_bytes(b"interrupted-unregistered-output")
    with pytest.raises(FileExistsError, match="partial"):
        validate_existing_tree_shard(
            shard,
            ["file.root#0"],
            hlt_content_sha256="1" * 64,
            tree_resource_sha256="2" * 64,
            backend_manifest_sha256="3" * 64,
        )
    assert shard.is_file()
    assert (
        validate_existing_tree_shard(
            shard,
            ["file.root#0"],
            hlt_content_sha256="1" * 64,
            tree_resource_sha256="2" * 64,
            backend_manifest_sha256="3" * 64,
            recover_unregistered_partial=True,
        )
        is None
    )
    assert not shard.exists()


def test_step8_worker_surface_and_tigris_defaults_are_present() -> None:
    required = {
        "run_build_relational_part_splits.sh",
        "run_audit_relational_part_raw_inputs.sh",
        "run_build_relational_part_hlt_cache.sh",
        "run_audit_relational_part_inputs.sh",
        "run_fit_relational_part_normalization.sh",
        "run_build_relational_part_tree_backend.sh",
        "run_probe_relational_part_tree_backend.sh",
        "run_build_relational_part_angular_tree_shard.sh",
        "run_finalize_relational_part_angular_tree_cache.sh",
        "run_train_relational_part.sh",
        "run_select_relational_part_screening.sh",
        "run_submit_relational_part_confirmation.sh",
        "run_aggregate_relational_part_confirmation.sh",
        "run_evaluate_relational_part_semantic_controls.sh",
        "run_submit_relational_part_final_test.sh",
        "run_evaluate_relational_part_final_test.sh",
        "run_write_relational_part_report.sh",
        "submit_relational_part_tigris_full.sh",
    }
    sbatch = ROOT / "sbatch"
    assert required.issubset({path.name for path in sbatch.iterdir()})
    common = (sbatch / "relational_part_common.sh").read_text(
        encoding="utf-8"
    )
    for value in (
        "/home/ryreu/atlas/Fresh_check",
        "/home/ryreu/atlas/PracticeTagging/data",
        "/home/ryreu/atlas/Fresh_check/checkpoints",
        "/home/ryreu/miniforge3-aarch64",
        "atlas_kd_tigris",
        "reu-aisocial",
        "gpu:gh200:1",
        "220G",
        "192G",
    ):
        assert value in common
    top = (sbatch / "submit_relational_part_tigris_full.sh").read_text(
        encoding="utf-8"
    )
    assert "--dry-run" in top
    assert "--smoke-submit" in top
    assert "0-20%${SCREENING_ARRAY_CONCURRENCY}" in top
    assert "preflight_relational_part_data.py" in top
    assert "afterok:" in top
    assert "RPT_STORAGE_MEASUREMENTS" in top
    semantic_runner = (
        ROOT / "scripts" / "run_relational_part_semantic_perturbation.py"
    ).read_text(encoding="utf-8")
    semantic_worker = (
        ROOT / "scripts" / "evaluate_relational_part_semantic_controls.py"
    ).read_text(encoding="utf-8")
    report_worker = (
        ROOT / "scripts" / "write_relational_part_report.py"
    ).read_text(encoding="utf-8")
    report_sbatch = (
        sbatch / "run_write_relational_part_report.sh"
    ).read_text(encoding="utf-8")
    assert "--campaign-spec" in semantic_runner
    assert "validate_campaign_source" in semantic_runner
    assert "checkpoint_registration_hashes" in semantic_worker
    assert "validate_campaign_source" in semantic_worker
    assert "validate_campaign_source" in report_worker
    assert "--campaign-spec" in report_sbatch


def test_continuation_source_snapshot_and_dynamic_ledger_fail_closed(
    tmp_path: Path,
) -> None:
    from teacher_logit_reco.relational_part import source_snapshot, with_content_hash

    source = source_snapshot(ROOT)
    campaign = with_content_hash(
        {
            "contract": "test_campaign",
            "schema_version": 1,
            "source": {
                "commit": source["source_commit"],
                "status_sha256": source["source_status_sha256"],
                "dirty": source["source_dirty"],
            },
        }
    )
    validate_campaign_source(campaign, repo_root=ROOT)
    stale = {
        **campaign,
        "source": {**campaign["source"], "status_sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="source snapshot"):
        validate_campaign_source(stale, repo_root=ROOT)

    campaign_path = tmp_path / "campaign_spec.json"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    ledger = tmp_path / "dynamic_jobs.json"
    script = ROOT / "scripts" / "register_relational_part_dynamic_job.py"
    command = [
        sys.executable,
        str(script),
        "--ledger",
        str(ledger),
        "--campaign-spec",
        str(campaign_path),
        "--logical-name",
        "confirmation_training",
        "--job-id",
        "12345",
        "--dependency",
        "afterok:100",
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["job_count"] == 1
    conflict = list(command)
    conflict[conflict.index("12345")] = "99999"
    failed = subprocess.run(
        conflict, cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert failed.returncode != 0
    assert "another binding" in failed.stderr


def test_smoke_simulation_prints_complete_nonmutating_ledger(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "smoke"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "submit_relational_part_graph.py"),
            "--campaign-id",
            "rpt_step8_smoke",
            "--campaign-root",
            str(campaign),
            "--miniature",
            "--smoke-simulate",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    graph = payload["production_graph"]
    ledger = payload["job_ledger"]
    assert payload["production_submission_performed"] is False
    assert not campaign.exists()
    assert validate_content_hash(ledger, expected_contract=JOB_LEDGER_CONTRACT)
    assert ledger["submission_mode"] == "smoke_simulation"
    assert ledger["submitted_node_count"] == len(graph["nodes"])
    assert all(value is not None for value in ledger["jobs"].values())
    assert graph["scientific_results_allowed"] is False


def test_postconstruction_contract_is_versioned_for_full_audit() -> None:
    assert (
        POSTCONSTRUCTION_AUDIT_CONTRACT
        == "relational_part_postconstruction_input_audit_v1"
    )

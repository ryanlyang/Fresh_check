from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jetclass_fresh.jetclass_data import (
    DEFAULT_SPLIT_SEEDS,
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    FileRecord,
    JetIdentity,
    SplitManifest,
    save_split_manifest,
)
from teacher_logit_reco.hlt_offline_structure_distillation import (
    PARENT_REQUIREMENTS,
    STAGE_ORDER,
    authorize_access,
    build_parent_status,
    build_registries,
    build_step1_bundle,
    canonical_sha256,
    load_hashed_json,
    load_authorized_identity_labels,
    publish_step1_bundle,
    require_parents_ready,
    validate_content_hash,
    validate_step1_bundle,
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    PARENT_REBUILD_PLAN_CONTRACT,
    bind_source,
    source_record,
)
from teacher_logit_reco.hlt_offline_structure_distillation.parent_submission import (
    build_parent_submission_plan,
    ensure_shared_bootstrap_split,
    finalize_parent_group,
    shared_parent_runtime_commands,
    submit_parent_plan,
)
from teacher_logit_reco.hlt_offline_structure_distillation.parents import (
    HLT_CACHE_SET_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge import (
    RetbSplitConfig,
    build_global_determinism,
    build_production_graph,
    miniature_storage_measurements,
    validate_production_campaign_binding,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (
    HOSD_MINIATURE_SPLIT_PROFILE,
)


def _source(*, status: str = "b" * 64) -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "source_status_sha256": status,
        "source_dirty": status != "b" * 64,
    }


def _manifest(
    *, reverse_validation: bool = False
) -> tuple[SplitManifest, RetbSplitConfig]:
    config = RetbSplitConfig.miniature(
        train_per_class=2,
        validation_role_per_class=2,
        final_select_per_class=1,
        final_test_per_class=2,
        scale_train_per_class=4,
    )
    splits: dict[str, list[JetIdentity]] = {
        role: []
        for role in (
            "model_train",
            "model_val",
            "stack_train",
            "stack_val",
            "final_test",
        )
    }
    records = []
    for label in range(10):
        path = f"class_{label}.root"
        records.append(FileRecord(path=path, label=label, num_entries=12))
        splits["model_train"].extend(
            JetIdentity(path, entry, label) for entry in (0, 1)
        )
        validation = [
            JetIdentity(path, entry, label) for entry in (2, 3, 4, 5)
        ]
        if reverse_validation:
            validation.reverse()
        splits["model_val"].extend(validation)
        splits["stack_val"].append(JetIdentity(path, 6, label))
        splits["final_test"].extend(
            JetIdentity(path, entry, label) for entry in (7, 8)
        )
    manifest = SplitManifest(
        data_dir="fixture",
        max_constits=128,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes=dict(config.split_sizes),
        split_seeds=dict(DEFAULT_SPLIT_SEEDS),
        file_records=records,
        splits=splits,
    )
    return manifest, config


def _bundle(tmp_path: Path) -> tuple[SplitManifest, dict]:
    manifest, config = _manifest()
    bundle = build_step1_bundle(
        campaign_id="hosd_step1_test",
        campaign_root=tmp_path / "campaign",
        manifest=manifest,
        split_config=config,
        source_snapshot=_source(),
        storage_measurements=miniature_storage_measurements(),
    )
    return manifest, bundle


def test_canonical_hashing_is_order_independent_and_tamper_evident() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256(
        {"a": 1, "b": 2}
    )
    artifact = with_content_hash(
        {"contract": "fixture_v1", "schema_version": 1, "value": 3}
    )
    assert validate_content_hash(artifact) == artifact["content_hash"]
    changed = dict(artifact)
    changed["value"] = 4
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_content_hash(changed)


def test_design_select_and_confirm_are_balanced_disjoint_and_deterministic(
    tmp_path: Path,
) -> None:
    first_manifest, first_config = _manifest(reverse_validation=False)
    second_manifest, second_config = _manifest(reverse_validation=True)
    first = build_step1_bundle(
        campaign_id="first",
        campaign_root=tmp_path / "first",
        manifest=first_manifest,
        split_config=first_config,
        source_snapshot=_source(),
        storage_measurements=miniature_storage_measurements(),
    )["design_partition_manifest"]
    second = build_step1_bundle(
        campaign_id="second",
        campaign_root=tmp_path / "second",
        manifest=second_manifest,
        split_config=second_config,
        source_snapshot=_source(),
        storage_measurements=miniature_storage_measurements(),
    )["design_partition_manifest"]
    assert first["roles"] == second["roles"]
    assert first["counts"] == {"design_select": 10, "design_confirm": 10}
    assert first["per_class_counts"] == {
        "design_select": 1,
        "design_confirm": 1,
    }
    selected = {
        JetIdentity.from_dict(row).key()
        for row in first["roles"]["design_select"]
    }
    confirmed = {
        JetIdentity.from_dict(row).key()
        for row in first["roles"]["design_confirm"]
    }
    assert selected.isdisjoint(confirmed)


def test_access_roles_fail_closed() -> None:
    authorize_access(
        worker_role="train_worker",
        requested_resource="model_train_targets",
    )
    authorize_access(
        worker_role="stack_inference",
        requested_resource="stack_val_hlt",
    )
    with pytest.raises(PermissionError):
        authorize_access(
            worker_role="stack_inference",
            requested_resource="final_select_label_manifest",
        )
    with pytest.raises(PermissionError):
        authorize_access(
            worker_role="target_builder",
            requested_resource="model_train_labels",
        )
    with pytest.raises(ValueError):
        authorize_access(worker_role="unknown", requested_resource="anything")


def test_postlock_final_labels_require_typed_authorized_loader(tmp_path) -> None:
    artifact = tmp_path / "identity_labels.npz"
    np.savez(
        artifact,
        identities=np.asarray(["a", "b"]),
        labels=np.asarray([1, 2], dtype=np.int64),
    )
    identities, labels = load_authorized_identity_labels(
        artifact,
        worker_role="final_inference",
        requested_resource="postlock_final_test_identity_labels",
    )
    assert identities == ("a", "b")
    assert labels.tolist() == [1, 2]
    with pytest.raises(PermissionError):
        load_authorized_identity_labels(
            artifact,
            worker_role="stack_inference",
            requested_resource="postlock_final_test_identity_labels",
        )


def test_stage_a_to_k_and_every_artifact_producer_are_enumerated() -> None:
    registries = build_registries(source=source_record(_source()))
    jobs = registries["stage_job_registry"]
    assert jobs["stage_order"] == list(STAGE_ORDER)
    assert {node["stage"] for node in jobs["nodes"]} == set(STAGE_ORDER)
    assert all(not node["performance_can_omit_or_cancel"] for node in jobs["nodes"])
    outputs = [
        output for node in jobs["nodes"] for output in node["outputs"]
    ]
    producer_rows = registries["producer_registry"]["entries"]
    assert len(outputs) == len(set(outputs))
    assert {row["artifact"] for row in producer_rows} == set(outputs)
    assert len(PARENT_REQUIREMENTS) == len(
        registries["parent_requirement_registry"]["requirements"]
    )


def test_parent_lock_requires_both_offline_and_shared_hlt_normalizer_families() -> None:
    rows = {row.parent_id: row for row in PARENT_REQUIREMENTS}
    assert {
        "relation_500k_normalizer",
        "region_500k_normalizer",
        "hlt_shared_500k_normalizer",
        "hlt_shared_region_500k_normalizer",
    }.issubset(rows)
    assert rows["hlt_shared_500k_normalizer"].canonical_path.endswith(
        "hlt_shared_500k/relation.json"
    )
    assert rows["hlt_shared_region_500k_normalizer"].canonical_path.endswith(
        "hlt_shared_500k/region.json"
    )


def test_parent_reuse_requires_exact_contract_hash_and_source() -> None:
    artifact = bind_source(
        build_global_determinism(),
        source_snapshot=_source(),
    )
    current = build_parent_status(
        source_snapshot=_source(),
        in_memory_artifacts={"global_determinism": artifact},
    )
    current_row = next(
        row
        for row in current["requirements"]
        if row["parent_id"] == "global_determinism"
    )
    assert current_row["status"] == "reusable_exact"

    drifted = build_parent_status(
        source_snapshot=_source(status="c" * 64),
        in_memory_artifacts={"global_determinism": artifact},
    )
    drifted_row = next(
        row
        for row in drifted["requirements"]
        if row["parent_id"] == "global_determinism"
    )
    assert drifted_row["status"] == "rebuild_required_source_drift"
    with pytest.raises(RuntimeError, match="not ready"):
        require_parents_ready(drifted, before_stage="A")

    tampered = dict(artifact)
    tampered["schema_version"] = 2
    with pytest.raises(ValueError, match="content hash mismatch"):
        build_parent_status(
            source_snapshot=_source(),
            in_memory_artifacts={"global_determinism": tampered},
        )


def test_bundle_binds_source_parents_registries_and_nonexecuting_dag(
    tmp_path: Path,
) -> None:
    manifest, bundle = _bundle(tmp_path)
    assert not (tmp_path / "campaign").exists()
    digest = validate_step1_bundle(bundle, manifest=manifest)
    spec = bundle["campaign_spec"]
    assert digest == spec["content_hash"]
    assert spec["scientific_training_or_inference_authorized"] is False
    assert spec["seeds"]["confirmation"] == [202, 303, 404]
    assert spec["access_policy"]["performance_based_termination"] is False
    assert bundle["dry_run_plan"]["execution_allowed"] is False
    assert bundle["dry_run_plan"]["scientific_outputs_created"] is False
    assert set(bundle["dry_run_plan"]["stage_job_counts"]) == set(STAGE_ORDER)
    assert bundle["parent_rebuild_plan"]["unresolved_parent_count"] > 0

    changed = copy.deepcopy(bundle)
    changed["campaign_spec"]["registry_hashes"]["stage_job_registry"] = "f" * 64
    with pytest.raises(ValueError):
        validate_step1_bundle(changed, manifest=manifest)


def test_publication_is_atomic_idempotent_and_preserves_shared_contracts(
    tmp_path: Path,
) -> None:
    manifest, bundle = _bundle(tmp_path)
    root = tmp_path / "campaign"
    first = publish_step1_bundle(
        campaign_root=root, manifest=manifest, bundle=bundle
    )
    second = publish_step1_bundle(
        campaign_root=root, manifest=manifest, bundle=bundle
    )
    assert first["campaign_spec_sha256"] == second["campaign_spec_sha256"]
    assert second["publications"]["campaign_spec"]["status"] == "already_present"
    assert (root / "campaign_spec.json").is_file()
    assert (root / "job_ledgers" / "stage_a_to_k_dry_run_plan.json").is_file()
    assert (
        root
        / "inputs"
        / "shared_retb_parent_campaign"
        / "campaign_spec.json"
    ).is_file()
    shared_spec = json.loads(
        (
            root
            / "inputs"
            / "shared_retb_parent_campaign"
            / "campaign_spec.json"
        ).read_text(encoding="utf-8")
    )
    assert shared_spec["campaign_id"] == "shared_retb_parent_campaign"
    shared_root = root / "inputs" / "shared_retb_parent_campaign"
    graph = build_production_graph(
        campaign_root=shared_root,
        campaign_id=shared_spec["campaign_id"],
        source_commit=shared_spec["source"]["commit"],
        source_status_sha256=shared_spec["source"]["status_sha256"],
        storage_measurements_sha256=shared_spec["parent_artifact_hashes"][
            "storage_measurements"
        ],
        miniature=True,
        miniature_split_profile=HOSD_MINIATURE_SPLIT_PROFILE,
        split_profile_parent_sha256=shared_spec["parent_artifact_hashes"][
            "split_audit"
        ],
    )
    validate_production_campaign_binding(graph, shared_spec)
    assert graph["split_sizes"]["model_val"] == 40
    drifted_graph = {
        **graph,
        "split_profile_parent_sha256": "e" * 64,
    }
    drifted_graph.pop("content_hash")
    drifted_graph = with_content_hash(drifted_graph)
    with pytest.raises(ValueError, match="campaign binding"):
        validate_production_campaign_binding(drifted_graph, shared_spec)
    assert not (root / "baselines" / "predictions.npz").exists()


def test_identical_concurrent_immutable_publication_is_race_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from teacher_logit_reco.relation_expert_token_bridge import contracts

    destination = tmp_path / "wave.json"
    real_link = contracts.os.link

    def concurrent_winner(source, target):
        Path(target).write_bytes(Path(source).read_bytes())
        raise FileExistsError("simulated identical winner")

    monkeypatch.setattr(contracts.os, "link", concurrent_winner)
    assert contracts.write_immutable_bytes(destination, b"same")["status"] == (
        "already_present"
    )
    monkeypatch.setattr(contracts.os, "link", real_link)


def test_cli_dry_run_enumerates_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import build_hosd_campaign

    manifest, _ = _manifest()
    manifest_path = tmp_path / "split_manifest.json.gz"
    save_split_manifest(manifest, manifest_path)
    output = tmp_path / "dry_run_campaign"
    monkeypatch.setattr(build_hosd_campaign, "source_snapshot", lambda _root: _source())
    assert (
        build_hosd_campaign.main(
            [
                "--parent-manifest",
                str(manifest_path),
                "--output-dir",
                str(output),
                "--campaign-id",
                "hosd_cli_dry_run",
                "--miniature",
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["scientific_outputs_created"] is False
    assert set(payload["stage_job_counts"]) == set(STAGE_ORDER)
    assert not output.exists()


def test_parent_group_completion_reuse_is_immutable_and_idempotent(
    tmp_path: Path,
) -> None:
    plan = {
        "campaign_spec_sha256": "a" * 64,
        "rebuild_plan_sha256": "b" * 64,
        "group": "hlt",
        "parent_ids": [],
    }
    first = finalize_parent_group(
        plan, campaign_root=tmp_path, submitted_job_ids=[]
    )
    second = finalize_parent_group(
        plan, campaign_root=tmp_path, submitted_job_ids=["999"]
    )
    assert second == first
    assert second["parents"] == {}
    assert second["task_manifest_completions"] == {}


def test_shared_parent_runtime_bootstraps_graph_before_task_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        parent_submission,
    )

    root = tmp_path / "hosd"
    shared = root / "inputs" / "shared_retb_parent_campaign"
    shared.mkdir(parents=True)
    source = source_record(_source())
    hosd = {"source": source}
    retb = with_content_hash(
        {
            "contract": "retb_campaign_spec_v1",
            "schema_version": 1,
            "campaign_id": "shared_retb_parent_campaign",
            "campaign_profile": "miniature_test",
            "parent_artifact_hashes": {"split_audit": "c" * 64},
            "source": source,
        }
    )
    (shared / "campaign_spec.json").write_text(
        json.dumps(retb), encoding="utf-8"
    )
    monkeypatch.setattr(
        parent_submission, "load_and_validate_campaign", lambda *_args, **_kwargs: hosd
    )
    commands = shared_parent_runtime_commands(
        campaign_root=root,
        repo_root=tmp_path,
        data_dir=tmp_path / "jetclass",
    )
    assert "--miniature" in commands[0]
    assert "submit_retb_graph.py" in commands[0][2]
    assert "--write-artifacts" in commands[0]
    assert "--miniature-split-profile" in commands[0]
    assert "hosd_real_miniature_v1" in commands[0]
    assert "--split-profile-parent-sha256" in commands[0]
    assert "bootstrap_retb_input_tasks.py" in commands[1][2]
    assert commands[1].index("--production-graph") < commands[1].index(
        "--data-dir"
    )


def test_shared_parent_runtime_materializes_byte_identical_bootstrap_split(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hosd"
    shared = root / "inputs" / "shared_retb_parent_campaign"
    source = shared / "inputs" / "split_manifest.json.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"authenticated-gzip-fixture")
    first = ensure_shared_bootstrap_split(campaign_root=root)
    second = ensure_shared_bootstrap_split(campaign_root=root)
    destination = shared / "bootstrap" / "split_manifest.json.gz"
    assert destination.read_bytes() == source.read_bytes()
    assert first["publication"] == "published"
    assert second["publication"] == "already_present"
    destination.write_bytes(b"drifted")
    with pytest.raises(FileExistsError, match="immutable"):
        ensure_shared_bootstrap_split(campaign_root=root)


def test_parent_plan_routes_logs_and_submits_complete_prerequisite_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        parent_submission,
    )

    root = tmp_path / "hosd"
    shared = root / "inputs" / "shared_retb_parent_campaign"
    tasks = shared / "job_ledgers" / "tasks"
    tasks.mkdir(parents=True)
    source = source_record(_source())
    campaign = {"campaign_id": "hosd", "content_hash": "a" * 64, "source": source}
    monkeypatch.setattr(
        parent_submission, "load_and_validate_campaign", lambda *_args, **_kwargs: campaign
    )
    rebuild = with_content_hash(
        {
            "contract": PARENT_REBUILD_PLAN_CONTRACT,
            "schema_version": 1,
            "source": source,
            "groups": [
                {
                    "group": "hlt",
                    "parent_ids": ["hlt_v3_cache", "hlt_v3_profile"],
                }
            ],
        }
    )
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "inputs" / "inherited_parent_rebuild_plan.json").write_text(
        json.dumps(rebuild), encoding="utf-8"
    )
    for node, array in (
        ("offline_input_cache", "0-5%2"),
        ("hlt_v3_cache", "0-8%2"),
    ):
        artifact = with_content_hash(
            {
                "contract": "retb_tigris_task_manifest_v1",
                "schema_version": 1,
                "node_id": node,
                "slurm_array": array,
            }
        )
        (tasks / f"{node}.json").write_text(json.dumps(artifact), encoding="utf-8")
    plan = build_parent_submission_plan(
        campaign_root=root, repo_root=tmp_path, group="hlt"
    )
    assert plan["runtime_ready"] is True
    assert [row["task_node"] for row in plan["commands"]] == [
        "offline_input_cache",
        "hlt_v3_cache",
    ]
    flattened = [value for row in plan["commands"] for value in row["argv"]]
    assert "--array=0-5%2" in flattened
    assert "--array=0-8%2" in flattened
    assert "--job-name=hosd_parent_offline_inputs" in flattened
    assert "--job-name=hosd_parent_hlt_v3" in flattened
    assert all(
        value.startswith("--job-name=hosd_parent_")
        for value in flattened
        if value.startswith("--job-name=")
    )
    assert any(value.startswith("--output=") for value in flattened)
    assert all("parent_controllers" in value for value in flattened if value.startswith("--error="))
    assert all(row["completion_argv"] for row in plan["commands"])
    assert all(
        "attest_retb_task_manifest_completion.py" in row["completion_argv"][2]
        for row in plan["commands"]
    )


def test_parent_submit_attests_each_array_before_submitting_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        parent_submission,
    )

    calls: list[list[str]] = []

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[0] == "sbatch":
            job_id = "101" if "first.sh" in argv else "102"
            return subprocess.CompletedProcess(
                argv, 0, stdout=job_id + "\n", stderr=""
            )
        return subprocess.CompletedProcess(argv, 0, stdout="{}\n", stderr="")

    monkeypatch.setattr(parent_submission.subprocess, "run", fake_run)
    job_ids = submit_parent_plan(
        {
            "runtime_ready": True,
            "commands": [
                {
                    "wrapper": "first.sh",
                    "argv": ["sbatch", "--parsable", "first.sh"],
                    "completion_argv": ["python", "attest.py", "first"],
                },
                {
                    "wrapper": "second.sh",
                    "argv": ["sbatch", "--parsable", "second.sh"],
                    "completion_argv": ["python", "attest.py", "second"],
                },
            ],
        }
    )
    assert job_ids == ["101", "102"]
    assert calls == [
        ["sbatch", "--parsable", "first.sh"],
        ["python", "attest.py", "first"],
        ["sbatch", "--parsable", "--dependency=afterok:101", "second.sh"],
        ["python", "attest.py", "second"],
    ]


def test_normalization_parent_finalizer_publishes_hosd_canonical_aliases(
    tmp_path, monkeypatch
) -> None:
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        parent_submission,
    )

    root = tmp_path / "campaign"
    shared = root / "inputs" / "shared_retb_parent_campaign"
    parent_ids = (
        "relation_500k_normalizer",
        "region_500k_normalizer",
        "hlt_shared_500k_normalizer",
        "hlt_shared_region_500k_normalizer",
    )
    requirements = []
    for index, parent_id in enumerate(parent_ids):
        contract = f"test_normalizer_contract_{index}"
        canonical = f"inputs/normalization/test_{index}/normalizer.json"
        artifact = with_content_hash(
            {"contract": contract, "schema_version": 1, "value": index}
        )
        source = shared / canonical
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(artifact), encoding="utf-8")
        requirements.append(
            SimpleNamespace(
                parent_id=parent_id,
                canonical_path=canonical,
                expected_contract=contract,
            )
        )
    monkeypatch.setattr(parent_submission, "PARENT_REQUIREMENTS", tuple(requirements))
    published = parent_submission._publish_normalization_aliases(
        campaign_root=root, shared_root=shared
    )
    assert set(published) == set(parent_ids)
    for requirement in requirements:
        destination = root / requirement.canonical_path
        assert destination.is_file()
        assert load_hashed_json(
            destination, expected_contract=requirement.expected_contract
        )["content_hash"] == published[requirement.parent_id]


def test_parent_submit_fails_closed_when_aggregate_attestation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        parent_submission,
    )

    responses = iter(
        [
            subprocess.CompletedProcess(["sbatch"], 0, stdout="101\n", stderr=""),
            subprocess.CompletedProcess(
                ["python", "attest.py"],
                1,
                stdout="",
                stderr="missing row completion 3",
            ),
        ]
    )
    monkeypatch.setattr(
        parent_submission.subprocess,
        "run",
        lambda *_args, **_kwargs: next(responses),
    )
    with pytest.raises(RuntimeError, match="missing row completion 3"):
        submit_parent_plan(
            {
                "runtime_ready": True,
                "commands": [
                    {
                        "wrapper": "worker.sh",
                        "argv": ["sbatch", "worker.sh"],
                        "completion_argv": ["python", "attest.py"],
                    }
                ],
            }
        )


def test_parent_submit_failure_includes_worker_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        parent_submission,
    )

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "retb_hlt_v3_42.err").write_text(
        "missing production_graph.json", encoding="utf-8"
    )
    monkeypatch.setattr(
        parent_submission.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["sbatch"], returncode=2, stdout="42", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="missing production_graph.json"):
        submit_parent_plan(
            {
                "runtime_ready": True,
                "log_directory": str(logs),
                "commands": [
                    {"wrapper": "worker.sh", "argv": ["sbatch", "worker.sh"]}
                ],
            }
        )


def test_hlt_cache_set_requires_complete_unique_nested_lineage(
    tmp_path: Path,
) -> None:
    source = source_record(_source())
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "hlt_v3_arrays.npz").write_bytes(b"fixture")
    metadata = with_content_hash(
        {
            "contract": "retb_hlt_v3_cache_v1",
            "schema_version": 1,
            "source": source,
        }
    )
    (cache / "hlt_v3_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    artifact = with_content_hash(
        {
            "contract": HLT_CACHE_SET_CONTRACT,
            "schema_version": 1,
            "source": source,
            "cache_count": 1,
            "caches": [
                {
                    "logical_role": "model_train",
                    "replica_id": 0,
                    "realization_policy": "R_MULTI",
                    "path": str(cache),
                    "metadata_sha256": metadata["content_hash"],
                }
            ],
        }
    )
    status = build_parent_status(
        source_snapshot=_source(), in_memory_artifacts={"hlt_v3_cache": artifact}
    )
    row = next(
        value for value in status["requirements"] if value["parent_id"] == "hlt_v3_cache"
    )
    assert row["reusable"] is True
    duplicated = dict(artifact)
    duplicated.pop("content_hash")
    duplicated["cache_count"] = 2
    duplicated["caches"] = artifact["caches"] * 2
    duplicated = with_content_hash(duplicated)
    with pytest.raises(ValueError, match="duplicated"):
        build_parent_status(
            source_snapshot=_source(),
            in_memory_artifacts={"hlt_v3_cache": duplicated},
        )

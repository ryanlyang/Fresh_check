from __future__ import annotations

import copy
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
    publish_step1_bundle,
    require_parents_ready,
    validate_content_hash,
    validate_step1_bundle,
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    bind_source,
    source_record,
)
from teacher_logit_reco.hlt_offline_structure_distillation.parent_submission import (
    finalize_parent_group,
)
from teacher_logit_reco.relation_expert_token_bridge import (
    RetbSplitConfig,
    build_global_determinism,
    miniature_storage_measurements,
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

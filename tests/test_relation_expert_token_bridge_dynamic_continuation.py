from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from teacher_logit_reco.relation_expert_token_bridge import (
    DYNAMIC_CONTINUATION_BINDING_CONTRACT,
    build_dynamic_continuation,
    build_production_graph,
    publish_dynamic_continuation,
    validate_dynamic_continuation,
    validate_published_dynamic_continuation,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    build_campaign_spec,
    with_content_hash,
    write_immutable_json,
)


ROOT = Path(__file__).resolve().parents[1]


def _source() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }


def _campaign_and_graph(root: Path) -> tuple[dict, dict]:
    parents = (
        "artifact_layout",
        "final_select_label_manifest",
        "global_determinism",
        "hlt_replica_manifest",
        "raw_input_schema",
        "scale_train_manifest",
        "split_audit",
        "split_manifest",
        "storage_measurements",
        "validation_partition_manifest",
    )
    campaign = build_campaign_spec(
        campaign_id=root.name,
        campaign_profile="miniature_test",
        source_snapshot=_source(),
        parent_artifact_hashes={
            name: f"{index + 1:064x}"
            for index, name in enumerate(parents)
        },
        run_registry_hashes={"dynamic-test": "f" * 64},
    )
    graph = build_production_graph(
        campaign_root=root,
        campaign_id=root.name,
        source_commit="a" * 40,
        source_status_sha256="b" * 64,
        storage_measurements_sha256=campaign["parent_artifact_hashes"][
            "storage_measurements"
        ],
        miniature=True,
    )
    return campaign, graph


def _selection(campaign: dict) -> dict:
    return with_content_hash(
        {
            "contract": "retb_test_selector_v1",
            "schema_version": 1,
            "selected": "best_available_even_when_negative",
            "source": campaign["source"],
        }
    )


def _rows(root: Path) -> list[dict]:
    return [
        {
            "task_id": "ignored-and-canonicalized",
            "argv": [
                sys.executable,
                "scripts/materialize_retb_stage_e_run.py",
                "--campaign-root",
                str(root),
            ],
            "environment": {"RETB_TEST_ROW": "0"},
            "expected_outputs": [
                str(root / "runs" / "dynamic" / "row_0.json")
            ],
            "input_artifact_hashes": {"parent": "1" * 64},
        },
        {
            "task_id": "also-canonicalized",
            "argv": [
                sys.executable,
                "scripts/materialize_retb_stage_e_run.py",
                "--campaign-root",
                str(root),
            ],
            "environment": {"RETB_TEST_ROW": "1"},
            "expected_outputs": [
                str(root / "runs" / "dynamic" / "row_1.json")
            ],
            "input_artifact_hashes": {"parent": "2" * 64},
        },
    ]


def test_dynamic_continuation_seals_selection_manifest_and_rows(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "retb_dynamic").resolve()
    root.mkdir()
    campaign, graph = _campaign_and_graph(root)
    selection = _selection(campaign)
    selection_path = root / "selection" / "pilot_selection.json"
    manifest_path = (
        root / "job_ledgers" / "tasks" / "stage_e_bridge_targets.json"
    )
    binding_path = (
        root
        / "selection"
        / "continuations"
        / "pilot_selection_to_bridge_target_training.json"
    )
    bundle = build_dynamic_continuation(
        campaign=campaign,
        production_graph=graph,
        selector_output=selection,
        selector_output_path=selection_path,
        downstream_node_id="bridge_target_training",
        rows=_rows(root),
        campaign_root=root,
    )
    validate_dynamic_continuation(
        bundle,
        campaign=campaign,
        production_graph=graph,
        selector_output=selection,
        selector_output_path=selection_path,
        rows=_rows(root),
        campaign_root=root,
    )
    assert bundle["continuation_binding"]["contract"] == (
        DYNAMIC_CONTINUATION_BINDING_CONTRACT
    )
    assert bundle["task_manifest"]["task_count"] == 2
    assert all(
        row["input_artifact_hashes"]["selector_output"]
        == selection["content_hash"]
        for row in bundle["task_manifest"]["rows"]
    )
    assert all(
        row["input_artifact_hashes"]["continuation_intent"]
        == bundle["continuation_intent"]["content_hash"]
        for row in bundle["task_manifest"]["rows"]
    )

    write_immutable_json(root / "campaign_spec.json", campaign)
    write_immutable_json(
        root / "job_ledgers" / "production_graph.json", graph
    )
    write_immutable_json(selection_path, selection)
    publish_dynamic_continuation(
        bundle=bundle,
        downstream_manifest_path=manifest_path,
        binding_path=binding_path,
    )
    assert validate_published_dynamic_continuation(
        campaign=campaign,
        production_graph=graph,
        task_manifest=bundle["task_manifest"],
        campaign_root=root,
    ) == bundle["continuation_binding"]["content_hash"]


def test_dynamic_continuation_rejects_missing_binding_and_selection_drift(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "retb_dynamic_tamper").resolve()
    root.mkdir()
    campaign, graph = _campaign_and_graph(root)
    selection = _selection(campaign)
    selection_path = root / "selection" / "pilot_selection.json"
    bundle = build_dynamic_continuation(
        campaign=campaign,
        production_graph=graph,
        selector_output=selection,
        selector_output_path=selection_path,
        downstream_node_id="bridge_target_training",
        rows=_rows(root),
        campaign_root=root,
    )
    write_immutable_json(selection_path, selection)
    manifest_path = (
        root / "job_ledgers" / "tasks" / "stage_e_bridge_targets.json"
    )
    write_immutable_json(manifest_path, bundle["task_manifest"])
    with pytest.raises(ValueError, match="unique continuation binding"):
        validate_published_dynamic_continuation(
            campaign=campaign,
            production_graph=graph,
            task_manifest=bundle["task_manifest"],
            campaign_root=root,
        )
    binding_path = (
        root
        / "selection"
        / "continuations"
        / "pilot_selection_to_bridge_target_training.json"
    )
    write_immutable_json(binding_path, bundle["continuation_binding"])
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    payload["selected"] = "drifted"
    selection_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_published_dynamic_continuation(
            campaign=campaign,
            production_graph=graph,
            task_manifest=bundle["task_manifest"],
            campaign_root=root,
        )


def test_dynamic_continuation_rejects_static_or_unbound_rows(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "retb_dynamic_contract").resolve()
    root.mkdir()
    campaign, graph = _campaign_and_graph(root)
    selection = _selection(campaign)
    with pytest.raises(ValueError, match="dynamic manifest-driven"):
        build_dynamic_continuation(
            campaign=campaign,
            production_graph=graph,
            selector_output=selection,
            selector_output_path=root / "selection" / "selection.json",
            downstream_node_id="offline_fusion_training",
            rows=_rows(root),
            campaign_root=root,
        )
    bad = _rows(root)
    bad[0]["input_artifact_hashes"]["selector_output"] = "3" * 64
    with pytest.raises(ValueError, match="reserved lineage"):
        build_dynamic_continuation(
            campaign=campaign,
            production_graph=graph,
            selector_output=selection,
            selector_output_path=root / "selection" / "selection.json",
            downstream_node_id="bridge_target_training",
            rows=bad,
            campaign_root=root,
        )


def test_stage_b_e_selection_clis_expose_the_continuation_contract() -> None:
    scripts = (
        "select_retb_offline_shapes.py",
        "select_retb_expert_optimization.py",
        "search_retb_offline_loss_bundle.py",
        "search_retb_heterogeneous_budget.py",
        "materialize_retb_stage_e_run.py",
        "certify_retb_bridge_content.py",
        "certify_retb_bridge_noninferiority.py",
        "select_retb_bridge_coordinates.py",
    )
    for name in scripts:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "add_dynamic_continuation_arguments(parser)" in source
        assert "resolve_selector_continuation(" in source

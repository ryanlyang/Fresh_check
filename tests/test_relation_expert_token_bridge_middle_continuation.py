from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from teacher_logit_reco.relation_expert_token_bridge import (
    MIDDLE_CONTINUATION_MANIFEST_NODES,
    MIDDLE_NODE_ENTRYPOINTS,
    MIDDLE_WAVE_PREREQUISITES,
    build_middle_continuation,
    build_production_graph,
    publish_middle_continuation,
    publish_task_manifest_completion,
    publish_task_row_completion,
    validate_middle_continuation,
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
        run_registry_hashes={"middle-test": "f" * 64},
    )
    graph = build_production_graph(
        campaign_root=root,
        campaign_id=root.name,
        source_commit="a" * 40,
        source_status_sha256="b" * 64,
        storage_measurements_sha256=campaign[
            "parent_artifact_hashes"
        ]["storage_measurements"],
        miniature=True,
    )
    return campaign, graph


def _trigger(campaign: dict, identity: str) -> dict:
    return with_content_hash(
        {
            "contract": f"retb_test_{identity}_v1",
            "schema_version": 1,
            "identity": identity,
            "source": campaign["source"],
        }
    )


def _rows(root: Path, node_id: str, count: int = 2) -> list[dict]:
    return [
        {
            "task_id": "canonicalized-by-continuation",
            "argv": [
                sys.executable,
                MIDDLE_NODE_ENTRYPOINTS[node_id],
                "--campaign-root",
                str(root),
            ],
            "environment": {"RETB_TEST_ROW": str(index)},
            "expected_outputs": [
                str(root / "outputs" / node_id / f"row_{index}.json")
            ],
            "input_artifact_hashes": {
                "resolved_configuration": f"{index + 17:064x}"
            },
        }
        for index in range(count)
    ]


def _publish_campaign(root: Path, campaign: dict, graph: dict) -> None:
    write_immutable_json(root / "campaign_spec.json", campaign)
    write_immutable_json(
        root / "job_ledgers" / "production_graph.json", graph
    )


def _publish_outputs_and_completion(
    *,
    root: Path,
    campaign: dict,
    manifest: dict,
) -> dict:
    for row in manifest["rows"]:
        output = with_content_hash(
            {
                "contract": "retb_test_middle_output_v1",
                "schema_version": 1,
                "task_id": row["task_id"],
                "source": campaign["source"],
            }
        )
        write_immutable_json(Path(row["expected_outputs"][0]), output)
        publish_task_row_completion(
            campaign_root=root,
            campaign=campaign,
            task_manifest=manifest,
            task_index=int(row["task_index"]),
        )
    return publish_task_manifest_completion(
        campaign_root=root,
        campaign=campaign,
        task_manifest=manifest,
    )["artifact"]


def test_stage_f_j_registry_is_complete_dynamic_and_stage_specific() -> None:
    root = Path("C:/campaign/retb_middle_registry")
    _, graph = _campaign_and_graph(root)
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    entries = {
        row["node_id"]: row
        for row in graph["node_execution_registry"]["entries"]
    }
    assert set(MIDDLE_NODE_ENTRYPOINTS) == set(
        MIDDLE_CONTINUATION_MANIFEST_NODES
    )
    assert set(MIDDLE_WAVE_PREREQUISITES) == set(
        MIDDLE_CONTINUATION_MANIFEST_NODES
    )
    for node_id in MIDDLE_CONTINUATION_MANIFEST_NODES:
        assert nodes[node_id]["dynamic_continuation"] is True
        assert entries[node_id]["row_resolution"] == "dynamic"
        assert entries[node_id]["manifest_producer"]["entrypoint"] == (
            "scripts/continue_retb_stage_f_j.py"
        )
        assert (ROOT / MIDDLE_NODE_ENTRYPOINTS[node_id]).is_file()


def test_middle_continuation_publishes_executable_bound_manifest(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "retb_middle").resolve()
    root.mkdir()
    campaign, graph = _campaign_and_graph(root)
    _publish_campaign(root, campaign, graph)
    trigger = _trigger(campaign, "step8")
    rows = _rows(root, "target_cache_build")
    payload = build_middle_continuation(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        node_id="target_cache_build",
        trigger_artifact=trigger,
        rows=rows,
    )
    validate_middle_continuation(
        payload,
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        trigger_artifact=trigger,
        rows=rows,
    )
    publications = publish_middle_continuation(
        campaign_root=root, payload=payload
    )
    manifest = payload["dynamic_continuation"]["task_manifest"]
    assert publications["gate"]["path"].endswith("gate.json")
    assert manifest["task_count"] == 2
    assert payload["gate"]["prerequisite_node_order"] == []
    assert validate_published_dynamic_continuation(
        campaign=campaign,
        production_graph=graph,
        task_manifest=manifest,
        campaign_root=root,
    ) == payload["dynamic_continuation"][
        "continuation_binding"
    ]["content_hash"]


def test_dependent_wave_requires_complete_revalidated_parent(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "retb_middle_complete").resolve()
    root.mkdir()
    campaign, graph = _campaign_and_graph(root)
    first = build_middle_continuation(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        node_id="target_cache_build",
        trigger_artifact=_trigger(campaign, "step8"),
        rows=_rows(root, "target_cache_build"),
    )
    publish_middle_continuation(campaign_root=root, payload=first)
    manifest = first["dynamic_continuation"]["task_manifest"]
    completion = _publish_outputs_and_completion(
        root=root, campaign=campaign, manifest=manifest
    )
    second = build_middle_continuation(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        node_id="target_normalizers",
        trigger_artifact=_trigger(campaign, "target-cache-complete"),
        rows=_rows(root, "target_normalizers", count=1),
    )
    assert second["gate"]["prerequisite_completion_hashes"] == {
        "target_cache_build": completion["content_hash"]
    }
    assert second["gate"][
        "all_prerequisite_rows_and_outputs_revalidated"
    ] is True

    output_path = Path(manifest["rows"][0]["expected_outputs"][0])
    replacement = with_content_hash(
        {
            "contract": "retb_test_middle_output_v1",
            "schema_version": 1,
            "task_id": "tampered-but-internally-hashed",
            "source": campaign["source"],
        }
    )
    output_path.write_text(json.dumps(replacement), encoding="utf-8")
    with pytest.raises(
        ValueError, match="task row completion semantics differ"
    ):
        build_middle_continuation(
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
            node_id="target_normalizers",
            trigger_artifact=_trigger(campaign, "target-cache-complete"),
            rows=_rows(root, "target_normalizers", count=1),
        )


def test_incomplete_parent_and_wrong_worker_fail_closed(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "retb_middle_incomplete").resolve()
    root.mkdir()
    campaign, graph = _campaign_and_graph(root)
    first = build_middle_continuation(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        node_id="target_cache_build",
        trigger_artifact=_trigger(campaign, "step8"),
        rows=_rows(root, "target_cache_build"),
    )
    publish_middle_continuation(campaign_root=root, payload=first)
    with pytest.raises(FileNotFoundError, match="prerequisite"):
        build_middle_continuation(
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
            node_id="target_normalizers",
            trigger_artifact=_trigger(campaign, "target-cache-complete"),
            rows=_rows(root, "target_normalizers", count=1),
        )
    wrong = _rows(root, "target_cache_build", count=1)
    wrong[0]["argv"][1] = "scripts/train_retb_predictor.py"
    with pytest.raises(ValueError, match="entry point differs"):
        build_middle_continuation(
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
            node_id="target_cache_build",
            trigger_artifact=_trigger(campaign, "step8"),
            rows=wrong,
        )


def test_stage_f_j_cli_and_completion_attestation_are_wired() -> None:
    continuation = (
        ROOT / "scripts" / "continue_retb_stage_f_j.py"
    ).read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "run_retb_task.py").read_text(
        encoding="utf-8"
    )
    launcher = (
        ROOT / "sbatch" / "run_retb_array_launcher.sh"
    ).read_text(encoding="utf-8")
    assert "build_middle_continuation(" in continuation
    assert "validate_middle_continuation(" in continuation
    assert "publish_middle_continuation(" in continuation
    assert "reusable_task_row_completion(" in runner
    assert "publish_task_row_completion(" in runner
    assert "publish_task_manifest_completion(" in runner
    assert "sbatch --parsable --wait" in launcher
    assert "attest_retb_task_manifest_completion.py" in launcher
    assert (
        ROOT / "scripts" / "calibrate_retb_uncertainty.py"
    ).is_file()

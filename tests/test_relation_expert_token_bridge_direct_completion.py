from __future__ import annotations

from pathlib import Path

import pytest

from teacher_logit_reco.relation_expert_token_bridge.campaign import (
    build_campaign_spec,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.direct_completion import (
    build_direct_node_completion,
    direct_node_completion_path,
    validate_direct_node_completion,
)
from teacher_logit_reco.relation_expert_token_bridge.early_continuation import (
    _producer_completion,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (
    build_production_graph,
)


SOURCE = {
    "source_commit": "1" * 40,
    "source_status_sha256": "2" * 64,
    "source_dirty": True,
}


def _campaign_and_graph(root: Path) -> tuple[dict, dict]:
    parent_names = (
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
        campaign_id="direct-completion",
        campaign_profile="miniature_test",
        source_snapshot=SOURCE,
        parent_artifact_hashes={
            name: f"{index + 1:064x}"
            for index, name in enumerate(parent_names)
        },
        run_registry_hashes={"runs": "f" * 64},
    )
    graph = build_production_graph(
        campaign_root=root,
        campaign_id=campaign["campaign_id"],
        source_commit=SOURCE["source_commit"],
        source_status_sha256=SOURCE["source_status_sha256"],
        storage_measurements_sha256=campaign[
            "parent_artifact_hashes"
        ]["storage_measurements"],
        miniature=True,
    )
    return campaign, graph


def test_direct_completion_authenticates_output_and_normalizes_for_factory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "direct-completion"
    root.mkdir()
    campaign, graph = _campaign_and_graph(root)
    artifact = root / "registry" / "step8.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("immutable\n", encoding="utf-8")
    completion = build_direct_node_completion(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        node_id="step8_target_cache_contracts",
        output_paths=[artifact],
    )
    validate_direct_node_completion(
        completion,
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
    )
    write_immutable_json(
        direct_node_completion_path(
            root, node_id="step8_target_cache_contracts"
        ),
        completion,
    )
    _, normalized = _producer_completion(
        root, producer_node_id="step8_target_cache_contracts"
    )
    assert normalized["completed_task_count"] == 1
    assert normalized["task_manifest_sha256"] == completion["content_hash"]

    artifact.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="semantics differ"):
        validate_direct_node_completion(
            completion,
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
        )


def test_step_contract_wrappers_attest_then_produce_then_materialize() -> None:
    root = Path(__file__).resolve().parents[1]
    for step in range(8, 15):
        source = (
            root / "sbatch" / f"run_retb_build_step{step}_contracts.sh"
        ).read_text(encoding="utf-8")
        assert (
            source.index("attest_retb_direct_node_completion.py")
            < source.index("produce_retb_downstream_manifest_plans.py")
            < source.index("materialize_retb_downstream_manifests.py")
        )

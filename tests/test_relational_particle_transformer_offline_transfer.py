from pathlib import Path

from teacher_logit_reco.relational_part.contracts import with_content_hash
from teacher_logit_reco.relational_part.offline_transfer import (
    OFFLINE_TRANSFER_MODEL_SPECS,
    OFFLINE_TRANSFER_SEEDS,
    build_offline_model_contract,
    build_offline_task_registry,
    build_offline_transfer_campaign,
    validate_offline_task_registry,
    validate_offline_transfer_campaign,
)
from teacher_logit_reco.relational_part.region_normalization_parallel import (
    REGION_NORMALIZATION_PLAN_CONTRACT_V2,
    build_region_normalization_plan,
    validate_region_normalization_plan,
)


def test_offline_campaign_freezes_matrix_and_has_no_performance_gate(tmp_path):
    split = tmp_path / "split_manifest.json.gz"
    split.write_bytes(b"locked split bytes")
    parent = with_content_hash(
        {"contract": "test_parent", "campaign_id": "parent_hlt"}
    )
    campaign = build_offline_transfer_campaign(
        campaign_id="offline_test",
        parent_campaign=parent,
        parent_campaign_path=tmp_path / "parent.json",
        split_manifest_path=split,
        source={
            "source_commit": "a" * 40,
            "source_status_sha256": "b" * 64,
            "source_dirty": False,
        },
    )
    validate_offline_transfer_campaign(campaign)
    assert campaign["failure_policy"]["performance_gate"] is False
    assert campaign["selection_policy"][
        "validation_cannot_remove_a_predeclared_final_task"
    ] is True
    assert campaign["model_matrix"] == OFFLINE_TRANSFER_MODEL_SPECS


def test_offline_task_registry_is_four_models_times_three_seeds():
    contracts = {
        run_id: build_offline_model_contract(
            run_id,
            campaign_sha256="a" * 64,
            relation_normalization_sha256="b" * 64,
            region_normalization_sha256="c" * 64,
        )
        for run_id in OFFLINE_TRANSFER_MODEL_SPECS
    }
    registry = build_offline_task_registry(
        campaign_sha256="a" * 64, model_contracts=contracts
    )
    validate_offline_task_registry(registry)
    assert registry["task_count"] == 12
    assert {
        (row["run_id"], row["seed"]) for row in registry["tasks"]
    } == {
        (run_id, seed)
        for run_id in OFFLINE_TRANSFER_MODEL_SPECS
        for seed in OFFLINE_TRANSFER_SEEDS
    }
    assert registry["performance_gate"] is False


def test_submission_graph_contains_lock_before_final_and_no_metric_gate():
    source = Path("sbatch/submit_relational_part_offline_transfer.sh").read_text(
        encoding="utf-8"
    )
    assert "run_aggregate_relational_part_offline_validation.sh" in source
    assert "run_evaluate_relational_part_offline_final.sh" in source
    assert source.index("jobs[validation_lock]") < source.index("jobs[final_test]")
    assert "performance_gate" not in source.lower() or "performance gate: disabled" in source.lower()
    assert "RPT_OFFLINE_SOURCE_COMMIT" in source
    assert 'cat-file -e "${source_commit}:${relative}"' in source
    assert "executing offline submitter differs" in source


def test_offline_region_plan_uses_generic_input_lineage():
    plan = build_region_normalization_plan(
        tree_manifest_sha256="a" * 64,
        tree_resource_sha256="b" * 64,
        relation_normalization_sha256="c" * 64,
        hlt_content_sha256="d" * 64,
        input_view="offline",
        selected_identities=["jet-0"],
        shard_rows=[
            {
                "shard_index": 0,
                "shard_jet_count": 1,
                "global_start": 0,
                "global_stop": 1,
                "selected_count": 1,
                "selected_local_indices": [0],
                "selection_ranks": [0],
                "selected_identity_sha256": "e" * 64,
                "selected_input_filename": "shard_00000.npz",
                "selected_input_npz_sha256": "f" * 64,
                "tree_shard_metadata_sha256": "1" * 64,
            }
        ],
    )
    validate_region_normalization_plan(plan)
    assert plan["contract"] == REGION_NORMALIZATION_PLAN_CONTRACT_V2
    assert plan["parents"]["input_view"] == "offline"
    assert plan["parents"]["input_content_sha256"] == "d" * 64
    assert "hlt_content_sha256" not in plan["parents"]

from __future__ import annotations

import importlib.util
from pathlib import Path

from teacher_logit_reco.local_particle_residual_field import (
    ConsumerCampaignConfig,
    LOCKED_HIGH_DATA_3M_SPLIT_CONFIG,
    consumer_run_specs,
    prediction_anchored_split_config,
    with_content_hash,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_prepare_module():
    path = REPO_ROOT / "scripts" / "prepare_prediction_anchored_high_data_manifest.py"
    spec = importlib.util.spec_from_file_location("pab_high_data_prepare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_high_data_3m_split_profile_has_exact_locked_inventory():
    config = prediction_anchored_split_config("high_data_3m")
    assert config == LOCKED_HIGH_DATA_3M_SPLIT_CONFIG
    assert dict(config.parent_split_counts) == {
        "model_train": 500_000,
        "model_val": 500_000,
        "stack_train": 6_000_000,
        "stack_val": 500_000,
        "final_test": 1_000_000,
    }
    children = {
        child.name: child.count
        for partition in config.partitions
        for child in partition.children
    }
    assert children == {
        "stack_train_consumer": 3_000_000,
        "stack_train_distill": 3_000_000,
        "model_val_stop": 250_000,
        "model_val_select": 250_000,
        "stack_val_consumer": 250_000,
        "stack_val_deploy": 250_000,
    }
    assert next(
        child
        for partition in config.partitions
        for child in partition.children
        if child.name == "stack_val_deploy"
    ).seal_kind == "deployable_preconfirmation"


def test_high_data_streaming_plan_does_not_duplicate_dense_fields():
    module = _load_prepare_module()
    digest = "a" * 64
    child = with_content_hash(
        {
            "contract": "prediction_anchored_child_splits_v1",
            "parent_manifest_sha256": digest,
            "split_config_sha256": "b" * 64,
            "parent_split_order_sha256": {
                "model_train": "c" * 64,
                "model_val": "d" * 64,
                "stack_train": "e" * 64,
                "stack_val": "f" * 64,
                "final_test": "0" * 64,
            },
            "children": {
                "stack_train_consumer": {
                    "parent_split": "stack_train",
                    "content_hash": "1" * 64,
                    "ordered_identity_sha256": "2" * 64,
                    "count": 20,
                    "parent_row_indices": list(range(20)),
                }
            },
        }
    )
    plan = module.build_streaming_plan(
        child,
        parent_manifest_file_sha256="3" * 64,
        source_data_dir="/data/jetclass",
        shard_events=10,
    )
    row = plan["children"]["stack_train_consumer"]
    assert row["count"] == 20
    assert [shard["count"] for shard in row["shards"]] == [10, 10]
    assert all("parent_row_indices" not in shard for shard in row["shards"])
    assert plan["dense_source_cache_policy"] == (
        "allowed_after_measured_projection_and_minimum_free_space_check"
    )
    assert plan["final_test_offline_inputs_allowed"] is False


def test_high_data_submitter_is_storage_safe_and_tigris_bound():
    submitter = (
        REPO_ROOT
        / "sbatch"
        / "submit_prediction_anchored_bridge_high_data_prepare.sh"
    ).read_text(encoding="utf-8")
    runner = (
        REPO_ROOT
        / "sbatch"
        / "run_prepare_prediction_anchored_high_data_manifest.sh"
    ).read_text(encoding="utf-8")

    for expected in (
        "MODEL_TRAIN_SIZE=500000",
        "MODEL_VAL_SIZE=500000",
        "STACK_TRAIN_SIZE=6000000",
        "STACK_VAL_SIZE=500000",
        "FINAL_TEST_SIZE=1000000",
        "PAB_SBATCH_ACCOUNT:=reu-aisocial",
        "PYTHONNOUSERSITE=1",
        "dense_npz_materialization=DEFERRED_TO_MEASURED_FULL_SUBMITTER",
        "scientific_training_submission=DEFERRED_UNTIL_SOURCE_CACHE_BUILD",
    ):
        assert expected in submitter
    assert "run_build_fresh_hlt_cache.sh" not in submitter
    assert "run_cache_architecture_view_offline_inputs.sh" not in submitter
    assert "--dependency=\"afterok:${split_job}\"" in submitter

    assert "#SBATCH --account=reu-aisocial" in runner
    assert "atlas_kd_tigris" in runner
    assert "export PYTHONNOUSERSITE=1" in runner
    assert "prepare_prediction_anchored_high_data_manifest.py" in runner


def test_high_data_consumer_profile_binds_3m_fairness_and_scaled_steps():
    config = ConsumerCampaignConfig(
        baseline_steps=120_000,
        bridge_finetune_steps=24_000,
        batch_size=128,
        evaluation_interval_steps=200,
        data_profile="high_data_3m",
    )
    specs = consumer_run_specs(config.data_profile)
    assert specs["A0_C250"].unique_jet_count == 3_000_000
    assert specs["A0_C250_LONG"].unique_jet_count == 3_000_000
    assert specs["A0_S500"].unique_jet_count == 6_000_000
    assert specs["Tpred"].unique_jet_count == 3_000_000
    assert config.to_artifact()["run_specs"]["T10_robust"]["unique_jet_count"] == 3_000_000


def test_high_data_full_submitter_builds_dense_sources_then_full_graph():
    submitter = (
        REPO_ROOT
        / "sbatch"
        / "submit_prediction_anchored_bridge_high_data_full.sh"
    ).read_text(encoding="utf-8")
    for expected in (
        "PAB_HIGH_DATA_ROOT:?",
        "PAB_HIGH_DATA_MIN_FREE_GIB:=40",
        "PAB_HIGH_DATA_SOURCE_MODE:=auto",
        "PAB_SBATCH_ACCOUNT:=reu-aisocial",
        "PYTHONNOUSERSITE=1",
        "run_build_fresh_hlt_cache.sh",
        "run_cache_architecture_view_offline_inputs.sh",
        "run_train_local_residual_field_tagger.sh A0",
        "run_finalize_prediction_anchored_bridge_submission.sh",
        "PAB_SPLIT_PROFILE=high_data_3m",
        "PAB_BUDGET_GIB=6",
        "PAB_RECON_PHASE2_EPOCHS=4",
        "PAB_CONSUMER_BASELINE_STEPS=120000",
        "PAB_CONSUMER_FINETUNE_STEPS=24000",
        'dependency="afterok:${hlt_job}:${offline_job}"',
        'dependency="afterok:${a0_job}"',
    ):
        assert expected in submitter
    assert "PAB_HIGH_DATA_MIN_FREE_GIB * 1024 * 1024" in submitter
    assert "High-data source is partial; refusing mixed build/reuse state" in submitter
    assert 'source_mode=%s\\n\' "${PAB_HIGH_DATA_SOURCE_MODE}"' in submitter
    assert "final_test_submission=SEALED_NOT_AUTOMATIC" in submitter
    consumer_runner = (
        REPO_ROOT
        / "sbatch"
        / "run_train_prediction_anchored_bridge_consumer.sh"
    ).read_text(encoding="utf-8")
    assert '--data-profile "${PAB_SPLIT_PROFILE}"' in consumer_runner
    assert '--baseline-steps "${PAB_CONSUMER_BASELINE_STEPS}"' in consumer_runner
    assert '--bridge-finetune-steps "${PAB_CONSUMER_FINETUNE_STEPS}"' in consumer_runner
    reconstructor_runner = (
        REPO_ROOT
        / "sbatch"
        / "run_train_prediction_anchored_bridge_reconstructor.sh"
    ).read_text(encoding="utf-8")
    assert '--phase2-epochs "${PAB_RECON_PHASE2_EPOCHS}"' in reconstructor_runner

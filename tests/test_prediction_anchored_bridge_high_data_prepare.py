from __future__ import annotations

import importlib.util
from pathlib import Path

from teacher_logit_reco.local_particle_residual_field import (
    LOCKED_HIGH_DATA_3M_SPLIT_CONFIG,
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
    assert plan["dense_source_cache_policy"].startswith("forbidden_in_home")
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
        "dense_npz_materialization=DISABLED",
        "scientific_training_submission=DEFERRED_UNTIL_STREAMING_RUNTIME",
    ):
        assert expected in submitter
    assert "run_build_fresh_hlt_cache.sh" not in submitter
    assert "run_cache_architecture_view_offline_inputs.sh" not in submitter
    assert "--dependency=\"afterok:${split_job}\"" in submitter

    assert "#SBATCH --account=reu-aisocial" in runner
    assert "atlas_kd_tigris" in runner
    assert "export PYTHONNOUSERSITE=1" in runner
    assert "prepare_prediction_anchored_high_data_manifest.py" in runner


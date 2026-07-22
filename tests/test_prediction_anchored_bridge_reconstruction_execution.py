from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.local_particle_residual_field.bridge import (
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    BridgeScalers,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import with_content_hash
from teacher_logit_reco.local_particle_residual_field.bridge_reconstruction_execution import (
    PREDICTION_ANCHORED_L0_POSTTEACHER_LINEAGE_CONTRACT,
    PREDICTION_ANCHORED_RECONSTRUCTION_METRICS_CONTRACT,
    PREDICTION_ANCHORED_RECONSTRUCTION_REPLICA_CONTRACT,
    RECONSTRUCTION_RUN_IDS,
    ReconstructionReplicaResult,
    aggregate_reconstruction_replicas,
    build_reconstruction_model,
    publish_reconstruction_paired_replicas,
    publish_l0_early_replay_manifest,
    resolve_reconstruction_run,
)
from teacher_logit_reco.local_particle_residual_field.hierarchical_global_reconstructor import (
    ARCH_A3_HLG_PRIMARY,
    ARCH_A5_HLG_ABSOLUTE,
    DIRECT_HLT,
)
from teacher_logit_reco.local_particle_residual_field.hierarchical_reconstructor import (
    ARCH_A1_MULTISCALE_LOCAL,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _scaler(policy: str) -> dict[str, object]:
    values = np.ones(50, dtype=np.float64)
    return BridgeScalers(
        mu_f0=np.zeros(50),
        sigma_f0=values,
        q99_delta=values,
        sigma_delta=values,
        trust_scale=2.0 * values,
        epsilon=1.0e-3 * values,
        active=np.asarray([True] * 50),
        sparse_nonzero_fallback=np.asarray([False] * 50),
        valid_count=100,
        parent_hashes={"source": "1" * 64},
        channel_policy=policy,
    ).to_artifact()


def _absolute() -> dict[str, object]:
    return with_content_hash(
        {
            "contract": "prediction_anchored_absolute_output_scaler_v1",
            "source_manifest_sha256": "2" * 64,
            "bridge_recipe_sha256": "3" * 64,
            "fit_partition": "stack_train_distill",
            "channel_policy": "physical45",
            "quantiles": [0.001, 0.999],
            "lo": [-1.0] * 45,
            "hi": [1.0] * 45,
            "center": [0.0] * 45,
            "half_range": [1.0] * 45,
            "epsilon": [1.0e-3] * 45,
            "valid_particle_count": 100,
            "batch_count": 1,
            "accumulation_dtype": "float64",
            "quantile_method": "numpy_linear_exact_from_ram_batches",
            "derived_dense_fields_persisted": False,
        }
    )


def test_all_46_reconstruction_rows_have_one_repository_executor_mapping():
    assert len(RECONSTRUCTION_RUN_IDS) == 46
    assert len(set(RECONSTRUCTION_RUN_IDS)) == 46
    resolved = {run_id: resolve_reconstruction_run(run_id) for run_id in RECONSTRUCTION_RUN_IDS}
    assert set(resolved) == set(RECONSTRUCTION_RUN_IDS)
    assert resolved["D10_L0_bridge_only"].binding_kind is None
    assert resolved["D10_L1_ce_only"].cache_namespace is None
    assert resolved["D10_L2_kd_only"].requires_cache is True
    assert resolved["D10_N3_nonprivileged_teacher_kd"].cache_namespace.endswith("f0_control")
    assert resolved[DIRECT_HLT].direct is True
    assert resolved["D10_B1_all50_fullhead"].channel_policy == "all50"


def test_representative_c0_local_hlg_absolute_and_all50_models_build_from_locked_artifacts():
    physical = _scaler(BRIDGE_CHANNEL_PHYSICAL45)
    all50 = _scaler(BRIDGE_CHANNEL_ALL50)
    c0, _ = build_reconstruction_model(
        "D10_L10_no_trust", physical45_scaler=physical, c0_model_width=24, dropout=0.0
    )
    local, _ = build_reconstruction_model(
        ARCH_A1_MULTISCALE_LOCAL, physical45_scaler=physical, dropout=0.0
    )
    hlg, _ = build_reconstruction_model(
        ARCH_A3_HLG_PRIMARY, physical45_scaler=physical, dropout=0.0
    )
    absolute, _ = build_reconstruction_model(
        ARCH_A5_HLG_ABSOLUTE,
        physical45_scaler=physical,
        absolute_scaler=_absolute(),
        dropout=0.0,
    )
    all50_model, _ = build_reconstruction_model(
        "D10_B1_all50_fullhead",
        physical45_scaler=physical,
        all50_scaler=all50,
        dropout=0.0,
    )
    assert c0.config.trust_bound_enabled is False
    assert local.config.architecture_id == ARCH_A1_MULTISCALE_LOCAL
    assert hlg.config.architecture_id == ARCH_A3_HLG_PRIMARY
    assert absolute.config.architecture_id == ARCH_A5_HLG_ABSOLUTE
    assert all50_model.full50_reachable is True
    try:
        build_reconstruction_model(DIRECT_HLT, physical45_scaler=physical)
    except FileNotFoundError as error:
        assert "resource reference" in str(error)
    else:  # pragma: no cover
        raise AssertionError("direct control accepted a guessed capacity reference")


def _replica(run_id: str, seed: int, accuracy: float) -> ReconstructionReplicaResult:
    metrics = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_RECONSTRUCTION_METRICS_CONTRACT,
            "run_id": run_id,
            "seed_id": seed,
            "model_val_select": {
                "accuracy": accuracy,
                "deployable_gain": accuracy - 0.5,
                "cross_entropy": 1.0 - accuracy,
            },
        }
    )
    weights = {
        "checkpoint_contract": PREDICTION_ANCHORED_RECONSTRUCTION_REPLICA_CONTRACT,
        "run_id": run_id,
        "seed_id": seed,
        "epoch": 2,
        "model_family": "c0",
        "architecture_id": run_id,
        "model_config": {"toy": True},
        "model_state_dict": {"weight": torch.tensor([float(seed)])},
        "parent_hashes": {"source": "4" * 64},
        "selected_teacher_checkpoint_sha256": "5" * 64,
        "target_cache_sha256": "6" * 64,
    }
    return ReconstructionReplicaResult(run_id, seed, metrics, weights)


def test_general_paired_publication_retains_only_ordered_median_weights(tmp_path):
    replicas = [
        _replica("D10_A1_multiscale_local", 101, 0.60),
        _replica("D10_A1_multiscale_local", 202, 0.70),
        _replica("D10_A1_multiscale_local", 303, 0.80),
    ]
    aggregate = aggregate_reconstruction_replicas(replicas)
    assert aggregate["ordered_seed_ids"] == [101, 202, 303]
    assert aggregate["median_seed_id"] == 202
    result = publish_reconstruction_paired_replicas(replicas, output_dir=tmp_path / "published")
    assert result["median_seed_id"] == 202
    assert sorted(path.name for path in (tmp_path / "published").iterdir()) == [
        "aggregate_metrics.json",
        "median_weights.pt",
        "publication.json",
    ]
    payload = torch.load(tmp_path / "published" / "median_weights.pt", weights_only=False)
    assert payload["seed_id"] == 202
    assert payload["weights_only"] is True
    assert payload["optimizer_state_persisted"] is False
    assert payload["generated_fields_persisted"] is False


def _l0_replica(
    seed: int,
    accuracy: float,
    bridge_loss: float,
    *,
    postteacher: bool,
) -> ReconstructionReplicaResult:
    teacher_sha = "5" * 64
    replay_sha = "6" * 64
    metrics = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_RECONSTRUCTION_METRICS_CONTRACT,
            "run_id": "D10_L0_bridge_only",
            "seed_id": seed,
            "checkpoint_selection": {
                "selected_epoch": 2,
                "selected_model_state_sha256": str(seed % 10) * 64,
            },
            "campaign_config": {"field_warmup_steps": 1, "phase2_epochs": 1},
            "teacher_lineage": with_content_hash(
                {
                    "contract": "prediction_anchored_c0_teacher_lineage_v1",
                    "run_id": "D10_L0_bridge_only",
                    "mode": "preteacher_l0_exception",
                    "teacher_checkpoint_sha256": None,
                }
            ),
            "postteacher_evaluation_lineage": (
                with_content_hash(
                    {
                        "contract": PREDICTION_ANCHORED_L0_POSTTEACHER_LINEAGE_CONTRACT,
                        "run_id": "D10_L0_bridge_only",
                        "teacher_checkpoint_sha256": teacher_sha,
                        "training_teacher_free": True,
                        "teacher_used_only_for_model_val_select_evaluation": True,
                        "target_logit_cache_used": False,
                        "final_test_accessed": False,
                    }
                )
                if postteacher
                else None
            ),
            "model_val_stop": {
                "bridge_loss": bridge_loss,
                "normalized_bridge_mse": bridge_loss + 0.1,
            },
            "model_val_select": {
                "accuracy": accuracy,
                "deployable_gain": accuracy - 0.5,
                "cross_entropy": 1.0 - accuracy,
            },
        }
    )
    weights = {
        "checkpoint_contract": PREDICTION_ANCHORED_RECONSTRUCTION_REPLICA_CONTRACT,
        "run_id": "D10_L0_bridge_only",
        "seed_id": seed,
        "epoch": 2,
        "model_family": "c0",
        "architecture_id": "D10_L0_bridge_only",
        "model_config": {"toy": True},
        "model_state_dict": {"weight": torch.tensor([float(seed)])},
        "parent_hashes": (
            {
                "execution_spec_sha256": "4" * 64,
                "teacher_checkpoint_sha256": teacher_sha,
                "l0_early_replay_sha256": replay_sha,
            }
            if postteacher
            else {"execution_spec_sha256": "4" * 64}
        ),
        "selected_teacher_checkpoint_sha256": teacher_sha if postteacher else None,
        "target_cache_sha256": None,
    }
    return ReconstructionReplicaResult(
        "D10_L0_bridge_only", seed, metrics, weights, str(seed % 10) * 64
    )


def test_l0_early_is_metrics_only_then_postteacher_publishes_accuracy_median(tmp_path):
    early_replicas = [
        _l0_replica(101, 0.80, 0.30, postteacher=False),
        _l0_replica(202, 0.60, 0.10, postteacher=False),
        _l0_replica(303, 0.70, 0.20, postteacher=False),
    ]
    early = publish_l0_early_replay_manifest(
        early_replicas, output_dir=tmp_path / "l0_early"
    )
    assert early["persistent_weights"] == 0
    assert early["persistent_artifacts"] == ["replay_manifest.json"]
    assert not list((tmp_path / "l0_early").glob("*.pt"))
    replay = json.loads((tmp_path / "l0_early" / "replay_manifest.json").read_text())
    assert replay["early_median_seed_id"] == 303
    assert replay["early_ordered_seed_ids"] == [101, 303, 202]
    assert replay["replicas"][0]["model_val_stop"]["bridge_loss"] == 0.3
    early_aggregate = aggregate_reconstruction_replicas(early_replicas)
    assert early_aggregate["median_seed_id"] == 303
    assert early_aggregate["aggregation_phase"] == "early_reachability_bridge_loss"
    with pytest.raises(ValueError, match="replay evidence"):
        publish_reconstruction_paired_replicas(
            early_replicas, output_dir=tmp_path / "invalid_early"
        )
    with pytest.raises(ValueError, match="evaluation lineage"):
        publish_reconstruction_paired_replicas(
            early_replicas,
            output_dir=tmp_path / "invalid_postteacher",
            l0_postteacher=True,
        )
    replicas = [
        _l0_replica(101, 0.80, 0.30, postteacher=True),
        _l0_replica(202, 0.60, 0.10, postteacher=True),
        _l0_replica(303, 0.70, 0.20, postteacher=True),
    ]
    with pytest.raises(ValueError, match="post-teacher lineage"):
        publish_l0_early_replay_manifest(
            replicas, output_dir=tmp_path / "invalid_replay"
        )
    final = publish_reconstruction_paired_replicas(
        replicas,
        output_dir=tmp_path / "l0_final",
        l0_postteacher=True,
    )
    assert final["median_seed_id"] == 303
    aggregate = aggregate_reconstruction_replicas(replicas, l0_postteacher=True)
    assert aggregate["ordered_seed_ids"] == [202, 303, 101]
    assert aggregate["aggregation_phase"] == "postteacher_common_model_val_select"


def test_slurm_and_submitter_use_repository_executor_not_external_placeholder():
    runner = (REPO_ROOT / "sbatch" / "run_train_prediction_anchored_bridge_reconstructor.sh").read_text()
    submitter = (REPO_ROOT / "scripts" / "submit_prediction_anchored_bridge_graph.py").read_text()
    assert "--mode execute" in runner
    assert "PAB_RECONSTRUCTOR_EXECUTOR" not in runner
    assert "PAB_RECONSTRUCTOR_EXECUTOR" not in submitter
    assert "publish-l0-early" in runner
    assert "publish-l0-postteacher" in runner
    assert "b6_l0_postteacher_eval_paired3" in runner
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in runner
    assert "#SBATCH --account=reu-aisocial" in runner
    assert "f\"--account={TIGRIS_ACCOUNT}\"" in submitter
    assert "PYTHONNOUSERSITE=1" in submitter

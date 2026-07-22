from __future__ import annotations

import numpy as np
import pytest
import torch

from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.local_particle_residual_field.bridge import (
    BRIDGE_CHANNEL_PHYSICAL45,
    BridgeScalers,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign_policy import (
    PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT,
    PredictionAnchoredDeployableBundle,
    build_deployable_bundle_manifest,
    export_deployable_bundle,
    load_deployable_bundle,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign import (
    build_campaign_registry,
    record_registry_measurements,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
    sha256_file,
    with_content_hash,
)
from teacher_logit_reco.local_particle_residual_field.bridge_deployment_execution import (
    select_deployable_from_publications,
    build_deployable_semantic_replica_evidence,
    repository_bundle_factory,
)
from teacher_logit_reco.local_particle_residual_field.bridge_evaluation import (
    PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
)
from teacher_logit_reco.local_particle_residual_field.bridge_reconstruction_execution import (
    PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT,
    PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import write_immutable_json
from teacher_logit_reco.local_particle_residual_field.bridge_reconstruction_execution import (
    build_reconstruction_model,
)
from teacher_logit_reco.local_particle_residual_field.model import (
    LocalResidualFieldReconstructorConfig,
    build_local_residual_field_reconstructor,
)
from teacher_logit_reco.local_particle_residual_field.tagger import (
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldTaggerConfig,
    RESIDUAL_FIELD_SOURCE_ORACLE,
)


def _scaler():
    ones = np.ones(50, dtype=np.float64)
    return BridgeScalers(
        mu_f0=np.zeros(50),
        sigma_f0=ones,
        q99_delta=ones,
        sigma_delta=ones,
        trust_scale=2.0 * ones,
        epsilon=1.0e-3 * ones,
        active=np.asarray([True] * 45 + [False] * 5),
        sparse_nonzero_fallback=np.asarray([False] * 50),
        valid_count=20,
        parent_hashes={"source": "1" * 64},
        channel_policy=BRIDGE_CHANNEL_PHYSICAL45,
    ).to_artifact()


def _hlt_batch():
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randn(2, 4, 14, generator=generator)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    tokens = tokens * mask.unsqueeze(-1)
    part = build_particle_transformer_inputs_from_tokens(
        tokens.numpy(), mask.numpy(), source_view="fixed_hlt"
    )
    return {
        "tokens": tokens,
        "raw_mask": mask,
        "points": torch.as_tensor(part.pf_points),
        "features": torch.as_tensor(part.pf_features),
        "lorentz_vectors": torch.as_tensor(part.pf_vectors),
        "mask": torch.as_tensor(part.pf_mask),
    }


def test_semantic_evidence_locks_four_seeds_and_thresholds():
    evidence = build_deployable_semantic_replica_evidence(
        run_id="D10_L1_ce_only",
        seed_id=101,
        perturbation_mean_accuracy_loss=0.001,
        perturbation_worst_accuracy_loss=0.002,
        alignment_finite=True,
        distribution_distance_finite=True,
    )
    assert evidence["perturbation_audit_seeds"] == [9101, 9102, 9103, 9104]
    assert evidence["perturbation_threshold_passed"] is True
    assert evidence["final_test_accessed"] is False
    with pytest.raises(ValueError, match="paired seed"):
        build_deployable_semantic_replica_evidence(
            run_id="bad",
            seed_id=404,
            perturbation_mean_accuracy_loss=0.0,
            perturbation_worst_accuracy_loss=0.0,
            alignment_finite=True,
            distribution_distance_finite=True,
        )


def test_real_repository_bundle_factory_reloads_without_parent_paths(tmp_path):
    pytest.importorskip("weaver")
    scaler = _scaler()
    r0_config = LocalResidualFieldReconstructorConfig(
        d_model=160,
        num_heads=5,
        num_layers=1,
        context_layers=1,
        dropout=0.0,
        attention_dropout=0.0,
        max_particles=4,
    )
    r0 = build_local_residual_field_reconstructor(r0_config)
    correction, _ = build_reconstruction_model(
        "D10_L1_ce_only", physical45_scaler=scaler, c0_model_width=20, dropout=0.0
    )
    consumer_config = LocalResidualFieldTaggerConfig(
        field_source=RESIDUAL_FIELD_SOURCE_ORACLE,
        model_size="tiny",
        field_dim=50,
    )
    consumer = LocalResidualFieldAugmentedParT(consumer_config)
    source = PredictionAnchoredDeployableBundle(r0, correction, consumer).eval()

    component_paths = {}
    for role, payload in {
        "r0": {"state": r0.state_dict()},
        "correction": {"state": correction.state_dict()},
        "consumer": {"state": consumer.state_dict()},
    }.items():
        path = tmp_path / f"{role}.pt"
        torch.save(payload, path)
        component_paths[role] = sha256_file(path)
    locked = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT,
            "status": "CONFIRMED_LOCKED",
            "selected_run_id": "D10_L1_ce_only",
            "median_seed_id": 202,
            "epoch": 1,
            "checkpoint_sha256": component_paths["correction"],
            "teacher_sha256": component_paths["consumer"],
            "scaler_sha256": scaler["content_hash"],
            "recipe_sha256": "4" * 64,
        }
    )
    manifest = build_deployable_bundle_manifest(
        locked,
        component_sha256=component_paths,
        preprocessing={"source": "fixed_hlt"},
        residual_normalization=scaler,
        target_schema={"field_dim": 50},
        class_order=[f"class_{index}" for index in range(10)],
        architecture_manifest={
            "r0_model_config": r0_config.to_dict(),
            "correction_model_config": correction.config.to_artifact(),
            "consumer_model_config": consumer_config.to_dict(),
        },
        bundle_reservation_bytes=256 * 1024 * 1024,
    )
    checkpoint = tmp_path / "bundle.pt"
    export_deployable_bundle(source, manifest=manifest, output_path=checkpoint)
    loaded, loaded_manifest = load_deployable_bundle(
        checkpoint, bundle_factory=repository_bundle_factory
    )
    assert loaded_manifest["input_availability"] == "hlt_only"
    with torch.no_grad():
        assert torch.allclose(source(_hlt_batch()), loaded(_hlt_batch()), atol=1e-6, rtol=1e-5)


def test_publication_selector_uses_three_seed_metrics_and_persistent_median_only(tmp_path):
    registry = build_campaign_registry()
    registry = record_registry_measurements(
        registry, {row["canonical_run_id"]: 1_000_000 for row in registry["runs"]}
    )
    r0_path = tmp_path / "r0.pt"
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": {"weight": torch.zeros(3)}}, r0_path)
    torch.save({"model_state_dict": {"weight": torch.zeros(5)}}, teacher_path)
    selected = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
            "status": "CONFIRMED_LOCKED",
            "checkpoint_path": str(teacher_path),
            "checkpoint_sha256": sha256_file(teacher_path),
        }
    )
    selection_path = tmp_path / "selection" / "selected_bridge_consumer.json"
    write_immutable_json(selection_path, selected)
    binding = with_content_hash(
        {
            "contract": "synthetic_primary_binding_v1",
            "checkpoint_sha256": sha256_file(teacher_path),
            "bridge_recipe_sha256": "4" * 64,
        }
    )
    write_immutable_json(tmp_path / "bindings" / "primary.json", binding)

    published = []
    for row in registry["runs"]:
        if not row["selectable_for_primary_deployment"]:
            continue
        run_id = row["canonical_run_id"]
        run_root = tmp_path / "reconstructors" / run_id
        run_root.mkdir(parents=True)
        median_path = run_root / "median_weights.pt"
        torch.save(
            {
                "model_state_dict": {"weight": torch.zeros(7)},
                "parent_hashes": {
                    "teacher_checkpoint_sha256": sha256_file(teacher_path),
                    "r0_checkpoint_sha256": sha256_file(r0_path),
                    "physical45_scaler_sha256": "5" * 64,
                },
            },
            median_path,
        )
        replica_metrics = []
        for seed, accuracy in zip((101, 202, 303), (0.709, 0.710, 0.711)):
            semantic = build_deployable_semantic_replica_evidence(
                run_id=run_id,
                seed_id=seed,
                perturbation_mean_accuracy_loss=0.001,
                perturbation_worst_accuracy_loss=0.002,
                alignment_finite=True,
                distribution_distance_finite=True,
            )
            metrics = {
                "checkpoint_selection": {"selected_epoch": 2},
                "model_val_select": {
                    "accuracy": accuracy,
                    "macro_per_class_accuracy": accuracy - 0.01,
                    "cross_entropy": 1.0 - accuracy,
                    "f0_accuracy": 0.700,
                    "privileged_bridge_accuracy": 0.720,
                    "recovery_fraction": (accuracy - 0.700) / 0.020,
                    "trust_saturation_fraction": 0.001,
                    "reliability_channels_exact_pass_through": True,
                    "semantic_evidence": semantic,
                },
            }
            replica_metrics.append(
                {
                    "seed_id": seed,
                    "metrics": metrics,
                    "source_checkpoint_sha256": f"{seed % 10}" * 64,
                    "weights_persisted": seed == 202,
                }
            )
        aggregate = with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT,
                "run_id": run_id,
                "aggregation_phase": (
                    "postteacher_common_model_val_select"
                    if run_id == "D10_L0_bridge_only"
                    else "ordinary_model_val_select"
                ),
                "paired_seed_ids": [101, 202, 303],
                "median_seed_id": 202,
                "replica_metrics": replica_metrics,
            }
        )
        write_immutable_json(run_root / "aggregate_metrics.json", aggregate)
        publication = with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT,
                "run_id": run_id,
                "aggregate_sha256": aggregate["content_hash"],
                "median_seed_id": 202,
                "retained_checkpoint": "median_weights.pt",
                "retained_checkpoint_sha256": sha256_file(median_path),
                "measured_state_bytes": median_path.stat().st_size,
                "weights_payload_reload_verified": True,
                "l0_postteacher_common_evaluation": run_id == "D10_L0_bridge_only",
            }
        )
        write_immutable_json(run_root / "publication.json", publication)
        published.append(run_id)

    evidence, preconfirmation = select_deployable_from_publications(
        registry,
        artifact_root=tmp_path,
        r0_checkpoint_path=r0_path,
        selected_consumer_path=selection_path,
        semantic_evidence_root=tmp_path / "semantic_evidence_not_needed",
    )
    assert len(evidence["replicas"]) == 3 * len(published)
    assert evidence["all_nonmedian_weights_metrics_only"] is True
    assert evidence["excluded_selectable_runs"] == []
    assert any(
        row["run_id"] == "D10_L0_bridge_only" for row in evidence["aggregates"]
    )
    assert preconfirmation["median_seed_id"] == 202
    assert preconfirmation["checkpoint"].endswith("median_weights.pt")

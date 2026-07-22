from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.validate_local_residual_field_fusion_completion import validate_completion
from teacher_logit_reco.local_particle_residual_field import fusion_campaign_report as report_module
from teacher_logit_reco.local_particle_residual_field import fusion_selection as selection_module
from teacher_logit_reco.local_particle_residual_field import fusion_sources as source_module
from teacher_logit_reco.local_particle_residual_field.fusion_campaign import (
    FUSION_HEAD_SEEDS,
    stable_fusion_json_hash,
)
from teacher_logit_reco.local_particle_residual_field.fusion_late import apply_late_fusion_candidate
from teacher_logit_reco.local_particle_residual_field.fusion_atomic import publish_temporary_file
from teacher_logit_reco.local_particle_residual_field.fusion_metric_audit import (
    LOCAL_RESIDUAL_FIELD_FUSION_METRIC_REPRODUCTION_CONTRACT,
)
from teacher_logit_reco.local_particle_residual_field.fusion_runtime import _torch_late_fusion, _verified_deployed_heads
from teacher_logit_reco.local_particle_residual_field.fusion_selection import FusionCandidateRunConfig
from tests.test_local_residual_field_fusion_campaign_step3 import _OracleFreeModel, build_source_fixture


ROOT = Path(__file__).resolve().parents[1]


def test_completion_validator_accepts_actual_source_audit_raw_source_hashes(tmp_path: Path) -> None:
    config, checkpoint_payload = build_source_fixture(tmp_path)
    audit = source_module.audit_fusion_source_artifacts(
        config,
        model_loader=lambda path, device="cpu": (_OracleFreeModel(), checkpoint_payload),
    )
    assert audit["ok"] is True

    validated = validate_completion(
        config.output_path,
        expected_contract="local_residual_field_fusion_source_artifact_audit_v1",
    )
    assert validated["ok"] is True


def test_immutable_publication_is_atomic_no_replace(tmp_path: Path) -> None:
    destination = tmp_path / "candidate_report.json"
    first = tmp_path / ".first.tmp"
    second = tmp_path / ".second.tmp"
    first.write_bytes(b"first-complete-artifact")
    second.write_bytes(b"second-complete-artifact")

    publish_temporary_file(first, destination, overwrite=False)
    with pytest.raises(FileExistsError, match="immutable artifact"):
        publish_temporary_file(second, destination, overwrite=False)

    assert destination.read_bytes() == b"first-complete-artifact"


def test_runtime_hash_binds_exact_deployed_representation_heads(tmp_path: Path) -> None:
    heads = []
    for seed in (5101, 5102, 5103):
        checkpoint = tmp_path / f"head_{seed}.pt"
        checkpoint.write_bytes(f"head-{seed}".encode())
        heads.append({
            "seed": seed,
            "checkpoint_path": str(checkpoint),
            "checkpoint_hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        })

    deployed, bindings = _verified_deployed_heads("R0_linear_embeddings", {"head_artifacts": heads})
    assert deployed == heads[:1]
    assert bindings == [{
        "seed": 5101,
        "checkpoint_path": str((tmp_path / "head_5101.pt").resolve()),
        "checkpoint_sha256": heads[0]["checkpoint_hash"],
    }]

    (tmp_path / "head_5101.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed before runtime benchmark"):
        _verified_deployed_heads("R0_linear_embeddings", {"head_artifacts": heads})


def _binary(false_positives: int) -> dict:
    return {
        "projections": {
            f"QCD_vs_{signal}": {
                "operating_points": {
                    "signal_efficiency_0.50": {
                        "available": True,
                        "qcd_false_positive_count": false_positives,
                        "qcd_support": 100,
                    }
                }
            }
            for signal in selection_module.FUSION_HEADLINE_SIGNALS
        }
    }


def _stability(mean_accuracy: float, mean_ce: float, mean_rejection: float) -> dict:
    return {
        "head_count": len(FUSION_HEAD_SEEDS),
        "stack_val": {
            "multiclass": {
                "accuracy": {"mean": mean_accuracy, "variance": 0.001},
                "cross_entropy": {"mean": mean_ce, "variance": 0.002},
            },
            "rejection_objective": {"mean": mean_rejection, "variance": 0.003},
        },
    }


def test_stability_ranking_uses_per_head_metric_mean_not_ensemble_metric() -> None:
    ensemble_favorite = {
        "candidate_id": "R1", "trainable_parameter_count": 10,
        "metrics": {"stack_val": {"multiclass": {"accuracy": 0.99, "cross_entropy": 0.01}}},
        "head_stability": _stability(0.80, 0.30, -2.0),
    }
    stable_favorite = {
        "candidate_id": "R2", "trainable_parameter_count": 20,
        "metrics": {"stack_val": {"multiclass": {"accuracy": 0.90, "cross_entropy": 0.10}}},
        "head_stability": _stability(0.82, 0.28, -2.2),
    }
    selected, _trace = selection_module._accuracy_choice([ensemble_favorite, stable_favorite])
    assert selected["candidate_id"] == "R2"
    assert selection_module._rejection_objective(stable_favorite) == -2.2
    columns = report_module._candidate_ranking_columns(stable_favorite)
    assert columns["selection_accuracy"] == 0.82
    assert columns["selection_accuracy_variance"] == 0.001


def test_r0_deploys_and_ranks_one_fixed_seed_head_but_records_three_head_stability(monkeypatch) -> None:
    data = {
        split: SimpleNamespace(labels=np.array([0, 1], dtype=np.int64))
        for split in ("stack_train", "stack_val")
    }
    monkeypatch.setattr(selection_module, "load_representation_fusion_development_data", lambda **_kwargs: (data, {}))
    monkeypatch.setattr(selection_module, "load_representation_fusion_head_from_checkpoint", lambda path, device: (path, {}))

    def fake_predict(model, _data, **_kwargs):
        value = float(str(model)[-1])
        return np.full((2, 10), value, dtype=np.float32), {}

    def fake_multiclass(logits, _labels, **_kwargs):
        value = float(logits[0, 0])
        return {
            "accuracy": value, "cross_entropy": 10.0 - value,
            "macro_one_vs_rest_auc": value, "macro_per_class_accuracy": value,
            "expected_calibration_error": value, "brier_score": value,
        }

    monkeypatch.setattr(selection_module, "predict_representation_fusion_head", fake_predict)
    monkeypatch.setattr(selection_module, "local_residual_field_multiclass_metrics", fake_multiclass)
    monkeypatch.setattr(
        selection_module,
        "local_residual_field_binary_projection_metrics",
        lambda logits, _labels, **_kwargs: _binary(int(round(float(logits[0, 0])))),
    )
    heads = [
        {"seed": seed, "checkpoint_path": f"head{index}"}
        for index, seed in enumerate(FUSION_HEAD_SEEDS, start=1)
    ]
    config = FusionCandidateRunConfig(
        campaign_id="review", group_id="F_method", candidate_id="R0_linear_embeddings",
        output_dir="unused", prediction_sources="predictions.json",
        source_artifact_audit="audit.json", feature_root="features", phase="stability",
    )
    metrics, diagnostics = selection_module._average_head_metrics(config, heads)
    stability = diagnostics["stability"]
    assert metrics["stack_val"]["multiclass"]["accuracy"] == 1.0
    assert stability["stack_val"]["multiclass"]["accuracy"]["mean"] == 2.0
    assert stability["stack_val"]["deployment_head_seeds"] == [FUSION_HEAD_SEEDS[0]]
    assert stability["stack_val"]["deployment_rule"] == "single_fixed_seed_linear_head"
    report = {
        "candidate_id": "R0_linear_embeddings", "head_stability": stability,
        "metrics": metrics,
    }
    assert selection_module._ranking_multiclass(report)["accuracy"] == 1.0
    assert stability["stack_val"]["ranking_rule"] == "deployed_fixed_seed_5101"


def test_raw_metric_gate_is_queued_and_validated(tmp_path: Path) -> None:
    submitter = (ROOT / "sbatch" / "submit_lprf_p7b_fusion_campaign.sh").read_text(encoding="utf-8")
    assert 'submit_metric_audit "${JOBS[preflight]}"' in submitter
    assert "metric_reproduction_audit.json" in submitter
    assert "metric_parent" in submitter

    payload = {
        "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_METRIC_REPRODUCTION_CONTRACT,
        "source_artifact_audit_hash": "a" * 64, "members": {}, "problems": [],
    }
    payload["audit_hash"] = stable_fusion_json_hash(payload)
    path = tmp_path / "metric_reproduction_audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert report_module._validated_metric_reproduction(
        path, source_artifact_audit_hash="a" * 64,
    )["audit_hash"] == payload["audit_hash"]
    with pytest.raises(ValueError, match="source audit"):
        report_module._validated_metric_reproduction(path, source_artifact_audit_hash="b" * 64)


def test_completion_validator_detects_referenced_mutation_and_submitter_quarantines(tmp_path: Path) -> None:
    referenced = tmp_path / "head.pt"
    referenced.write_bytes(b"checkpoint")
    payload = {
        "ok": True, "contract": "review_completion_v1",
        "checkpoint_path": str(referenced.resolve()),
        "checkpoint_hash": hashlib.sha256(referenced.read_bytes()).hexdigest(),
    }
    payload["artifact_hash"] = stable_fusion_json_hash(payload)
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_completion(completion)["referenced_files_checked"] == [str(referenced.resolve())]
    referenced.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        validate_completion(completion)
    submitter = (ROOT / "sbatch" / "submit_lprf_p7b_fusion_campaign.sh").read_text(encoding="utf-8")
    assert "Quarantining invalid completion artifact" in submitter
    assert "seed_control_completion.json" in submitter


def test_completion_validator_enforces_stage_contract_and_logical_parent_hash(tmp_path: Path) -> None:
    parent = {
        "ok": True,
        "contract": "local_residual_field_fusion_source_artifact_audit_v1",
        "generation": 1,
    }
    parent["audit_hash"] = stable_fusion_json_hash(parent)
    parent_path = tmp_path / "source_artifact_audit.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    child = {
        "ok": True, "contract": "candidate_completion_v1",
        "source_artifact_audit_path": str(parent_path.resolve()),
        "source_artifact_audit_hash": parent["audit_hash"],
    }
    child["artifact_hash"] = stable_fusion_json_hash(child)
    child_path = tmp_path / "candidate_report.json"
    child_path.write_text(json.dumps(child), encoding="utf-8")
    assert validate_completion(
        child_path, expected_contract="candidate_completion_v1",
    )["contract"] == "candidate_completion_v1"
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_completion(child_path, expected_contract="wrong_stage_v1")

    parent["generation"] = 2
    parent.pop("audit_hash")
    parent["audit_hash"] = stable_fusion_json_hash(parent)
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(ValueError, match="logical parent hash changed"):
        validate_completion(child_path, expected_contract="candidate_completion_v1")


def test_standalone_fit_candidates_requires_valid_metric_audit() -> None:
    submitter = (ROOT / "sbatch" / "submit_lprf_p7b_fusion_campaign.sh").read_text(encoding="utf-8")
    assert "completion_contract" in submitter
    assert "--expected-contract" in submitter
    assert 'artifact_done metric_audit "${LOCAL_RESIDUAL_FIELD_FUSION_METRIC_AUDIT}"' in submitter
    assert "fit_candidates requires a valid completed raw-metric reproduction audit" in submitter


def test_report_completion_binds_upstream_inputs_and_generated_outputs(tmp_path: Path) -> None:
    upstream = tmp_path / "final_evaluation.bin"
    generated = tmp_path / "summary.md"
    upstream.write_bytes(b"final-v1")
    generated.write_text("summary-v1", encoding="utf-8")
    report = {
        "ok": True, "contract": "local_residual_field_fusion_campaign_report_v1",
        "input_artifacts": [{
            "name": "final_evaluation", "path": str(upstream.resolve()),
            "sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
        }],
        "output_artifacts": [{
            "path": str(generated.resolve()),
            "sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
        }],
    }
    report["artifact_hash"] = stable_fusion_json_hash(report)
    report_path = tmp_path / "run_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    validate_completion(
        report_path,
        expected_contract="local_residual_field_fusion_campaign_report_v1",
    )
    upstream.write_bytes(b"final-v2")
    with pytest.raises(ValueError, match="changed"):
        validate_completion(
            report_path,
            expected_contract="local_residual_field_fusion_campaign_report_v1",
        )


def test_completion_validation_cascades_through_byte_bound_json_parents(tmp_path: Path) -> None:
    source_audit = {
        "ok": True, "contract": "local_residual_field_fusion_source_artifact_audit_v1",
    }
    source_audit["audit_hash"] = stable_fusion_json_hash(source_audit)
    source_audit_path = tmp_path / "source_artifact_audit.json"
    source_audit_path.write_text(json.dumps(source_audit), encoding="utf-8")
    checkpoint_path = tmp_path / "best_model_val.pt"
    checkpoint_path.write_bytes(b"seed-checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    seed_completion = {
        "ok": True, "contract": "local_residual_field_a0_seed1_completion_v1",
        "source_artifact_audit_path": str(source_audit_path.resolve()),
        "source_artifact_audit_hash": source_audit["audit_hash"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
    }
    seed_completion["artifact_hash"] = stable_fusion_json_hash(seed_completion)
    seed_completion_path = tmp_path / "seed_control_completion.json"
    seed_completion_path.write_text(json.dumps(seed_completion), encoding="utf-8")
    prediction = {
        "ok": True, "contract": "local_residual_field_fusion_prediction_sources_v1",
        "generation": 1,
        "source_artifact_audit_path": str(source_audit_path.resolve()),
        "source_artifact_audit_hash": source_audit["audit_hash"],
        "seed_control_completion_path": str(seed_completion_path.resolve()),
        "seed_control_completion_hash": seed_completion["artifact_hash"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_hash": checkpoint_hash,
    }
    prediction["manifest_hash"] = stable_fusion_json_hash(prediction)
    prediction_path = tmp_path / "development_prediction_sources.json"
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    feature = {
        "ok": True, "contract": "local_residual_field_fusion_feature_manifest_v1",
        "source_artifact_audit_path": str(source_audit_path.resolve()),
        "source_artifact_audit_hash": source_audit["audit_hash"],
        "prediction_sources_path": str(prediction_path.resolve()),
        "prediction_sources_hash": prediction["manifest_hash"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_hash": checkpoint_hash,
    }
    feature["manifest_hash"] = stable_fusion_json_hash(feature)
    feature_path = tmp_path / "representation_manifest.json"
    feature_path.write_text(json.dumps(feature), encoding="utf-8")
    candidate = {
        "ok": True, "contract": "local_residual_field_fusion_candidate_report_v1",
        "source_artifact_audit_path": str(source_audit_path.resolve()),
        "source_artifact_audit_hash": source_audit["audit_hash"],
        "prediction_sources_path": str(prediction_path.resolve()),
        "prediction_sources_hash": prediction["manifest_hash"],
        "feature_manifest_bindings": [{
            "path": str(feature_path.resolve()),
            "sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            "manifest_hash": feature["manifest_hash"],
        }],
    }
    candidate["artifact_hash"] = stable_fusion_json_hash(candidate)
    candidate_path = tmp_path / "candidate_report.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    validate_completion(
        candidate_path,
        expected_contract="local_residual_field_fusion_candidate_report_v1",
    )

    prediction["generation"] = 2
    prediction.pop("manifest_hash")
    prediction["manifest_hash"] = stable_fusion_json_hash(prediction)
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    with pytest.raises(ValueError, match="logical parent hash changed"):
        validate_completion(
            candidate_path,
            expected_contract="local_residual_field_fusion_candidate_report_v1",
        )


def test_torch_runtime_late_fusion_matches_frozen_numpy_recipe() -> None:
    rng = np.random.default_rng(17)
    logits_a = rng.normal(size=(7, 10)).astype(np.float32)
    logits_b = rng.normal(size=(7, 10)).astype(np.float32)
    parameters = {
        "L0_mean_logits": {},
        "L1_mean_probs": {},
        "L2_temp_mean_logits": {"temperatures": [1.2, 0.8]},
        "L3_scalar_simplex_logits": {"weight": 0.35},
        "L4_classwise_simplex_logits": {"weights": np.linspace(0.1, 0.9, 10).tolist()},
        "L5_linear_stacker": {
            "feature_mode": "logits", "feature_mean": np.zeros(20).tolist(),
            "weight": rng.normal(size=(10, 20)).tolist(), "bias": rng.normal(size=10).tolist(),
        },
    }
    tensor_a, tensor_b = torch.from_numpy(logits_a), torch.from_numpy(logits_b)
    for candidate_id, recipe in parameters.items():
        observed = _torch_late_fusion(candidate_id, recipe, tensor_a, tensor_b).detach().numpy()
        expected = apply_late_fusion_candidate(candidate_id, recipe, logits_a, logits_b)
        np.testing.assert_allclose(observed, expected, atol=2.0e-5, rtol=2.0e-5)


def test_report_requires_current_final_sha_and_explicit_privilege_flags() -> None:
    valid = {
        "runtime_inputs": "HLT_only", "deployable": True,
        "uses_true_fields": False, "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
    }
    report_module._require_deployable_hlt_only([valid])
    with pytest.raises(ValueError, match="non-deployable"):
        report_module._require_deployable_hlt_only([{key: value for key, value in valid.items() if key != "uses_true_fields"}])
    report_module._require_current_final_bindings(
        "f" * 64, {"final_evaluation_sha256": "f" * 64}, {"final_evaluation_sha256": "f" * 64},
    )
    with pytest.raises(ValueError, match="stale"):
        report_module._require_current_final_bindings(
            "f" * 64, {"final_evaluation_sha256": "e" * 64}, {"final_evaluation_sha256": "f" * 64},
        )

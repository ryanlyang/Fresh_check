from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teacher_logit_reco.local_particle_residual_field import (
    FUSION_CANDIDATE_IDS,
    FUSION_GROUP_METHOD,
    FUSION_GROUP_SEED,
    FUSION_HEAD_SEEDS,
    FusionCampaignConfig,
    FusionCandidateRunConfig,
    FusionSelectionConfig,
    LOCAL_RESIDUAL_FIELD_FUSION_CANDIDATE_REPORT_CONTRACT,
    load_selected_fusion_set,
    run_fusion_candidate,
    select_fusion_champions,
    stable_fusion_json_hash,
)
from teacher_logit_reco.local_particle_residual_field import fusion_selection as selection_module


def _binary(false_positives: int = 2) -> dict:
    projections = {}
    for signal in ("Hgg", "Hbb", "Tbqq", "Wqq", "Zqq", "Hcc", "Tbcq", "Tbqq2", "Wcq"):
        projections[f"QCD_vs_{signal}"] = {
            "operating_points": {
                "signal_efficiency_0.30": {
                    "available": True, "threshold": 0.5,
                    "qcd_false_positive_count": false_positives, "qcd_support": 100,
                },
                "signal_efficiency_0.50": {
                    "available": True, "threshold": 0.0,
                    "qcd_false_positive_count": false_positives, "qcd_support": 100,
                }
            }
        }
    return {"contract": "local_residual_field_binary_projection_metrics_v2", "projections": projections}


def _candidate_report(campaign: FusionCampaignConfig, group_id: str, candidate_id: str, *, accuracy: float) -> dict:
    spec = next(spec for spec in campaign.candidates if spec.candidate_id == candidate_id)
    members = next(group.member_ids for group in campaign.groups if group.group_id == group_id)
    checkpoint_hashes = {member: ("a" if member == "A0" else "b" if member == "P7b" else "c") * 64 for member in members}
    phase = "complete" if spec.family == "late" else "screening"
    report = {
        "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_CANDIDATE_REPORT_CONTRACT,
        "campaign_id": campaign.campaign_id, "group_id": group_id, "member_ids": list(members),
        "member_checkpoint_hashes": checkpoint_hashes, "candidate_id": candidate_id, "family": spec.family,
        "candidate_spec": spec.to_dict(), "candidate_spec_hash": stable_fusion_json_hash(spec.to_dict()),
        "phase": phase, "head_seeds": [] if spec.family == "late" else [5101],
        "selected_hyperparameters": {}, "trainable_parameter_count": 0,
        "metrics": {
            split: {
                "multiclass": {"accuracy": accuracy, "cross_entropy": 1.0 - accuracy, "n_jets": 100},
                "binary_projection": _binary(1 if candidate_id == "L0_mean_logits" else 3),
            }
            for split in ("stack_train", "stack_val")
        },
        "fit_artifacts": [], "prediction_sources_path": "predictions.json",
        "prediction_sources_hash": "d" * 64, "source_artifact_audit_path": "audit.json",
        "source_artifact_audit_hash": "e" * 64,
        "feature_root": None if spec.family == "late" else "features",
        "development_splits": ["stack_train", "stack_val"], "final_test_opened": False,
        "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False, "deployable": True,
    }
    report["artifact_hash"] = stable_fusion_json_hash(report)
    return report


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not report.get("fit_artifacts"):
        fit_path = path.parent / f"{path.stem}_fit.json"
        fit_path.write_text("{}\n", encoding="utf-8")
        import hashlib
        report["fit_artifacts"] = [{
            "path": str(fit_path.resolve()),
            "sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
            "artifact_hash": "f" * 64,
        }]
        report.pop("artifact_hash", None)
        report["artifact_hash"] = stable_fusion_json_hash(report)
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


def _selection_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    campaign = FusionCampaignConfig(campaign_id="toy")
    candidate_root = tmp_path / "campaign" / "candidates"
    for group in (FUSION_GROUP_METHOD, FUSION_GROUP_SEED):
        for index, candidate_id in enumerate(FUSION_CANDIDATE_IDS):
            accuracy = 0.82 if candidate_id == "L0_mean_logits" else 0.80 - index * 1.0e-5
            report = _candidate_report(campaign, group, candidate_id, accuracy=accuracy)
            root = candidate_root / group / candidate_id
            _write_report(root / "candidate_report.json", report)
            if candidate_id.startswith("R"):
                stable = dict(report)
                stable["phase"] = "stability"
                stable["head_seeds"] = list(FUSION_HEAD_SEEDS)
                stable.pop("artifact_hash")
                stable["artifact_hash"] = stable_fusion_json_hash(stable)
                _write_report(root / "candidate_stability_report.json", stable)
    registry = {
        "manifest_hash": "d" * 64, "source_artifact_audit_hash": "e" * 64,
        "members": {
            member: {"prediction_root": member, "splits": {}}
            for member in ("A0", "A0_seed1", "P7b")
        },
    }
    monkeypatch.setattr(selection_module, "require_development_prediction_sources", lambda *_args, **_kwargs: registry)
    monkeypatch.setattr(
        selection_module, "load_prediction_block",
        lambda _root, member, _split, verify_hash=True: SimpleNamespace(
            logits=np.zeros((10, 10), dtype=np.float32), labels=np.arange(10), metadata={"member": member}
        ),
    )
    monkeypatch.setattr(
        selection_module, "local_residual_field_multiclass_metrics",
        lambda _logits, _labels, label_names: {"accuracy": 0.81},
    )
    output = tmp_path / "campaign" / "selection" / "selected_fusion.json"
    select_fusion_champions(
        FusionSelectionConfig(
            campaign_id="toy", candidates_root=str(candidate_root), prediction_sources="predictions.json",
            source_artifact_audit="audit.json", output_path=str(output),
        )
    )
    return output


def test_step9_selector_requires_symmetric_coverage_and_writes_four_locked_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _selection_fixture(tmp_path, monkeypatch)
    selected = load_selected_fusion_set(output)

    assert len(selected["selections"]) == 4
    assert {(row["group_id"], row["champion_role"]) for row in selected["selections"]} == {
        ("F_method", "accuracy_champion"), ("F_method", "rejection_champion"),
        ("F_seed", "accuracy_champion"), ("F_seed", "rejection_champion"),
    }
    assert {row["candidate_id"] for row in selected["selections"]} == {"L0_mean_logits"}
    assert selected["final_test_opened"] is False
    assert selected["representation_stability_union"]


def test_step9_selector_rejects_missing_or_final_contaminated_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _selection_fixture(tmp_path, monkeypatch)
    candidate = tmp_path / "campaign" / "candidates" / "F_seed" / "L5_linear_stacker" / "candidate_report.json"
    candidate.unlink()
    with pytest.raises(FileNotFoundError):
        select_fusion_champions(
            FusionSelectionConfig(
                campaign_id="toy", candidates_root=str(tmp_path / "campaign" / "candidates"),
                prediction_sources="predictions.json", source_artifact_audit="audit.json",
                output_path=str(output.with_name("second.json")),
            )
        )


def test_step9_selector_rejects_final_metrics_before_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _selection_fixture(tmp_path, monkeypatch)
    candidate = tmp_path / "campaign" / "candidates" / "F_method" / "L1_mean_probs" / "candidate_report.json"
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["metrics"]["final_test"] = payload["metrics"]["stack_val"]
    payload.pop("artifact_hash")
    payload["artifact_hash"] = stable_fusion_json_hash(payload)
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="development splits|final-test"):
        select_fusion_champions(
            FusionSelectionConfig(
                campaign_id="toy", candidates_root=str(tmp_path / "campaign" / "candidates"),
                prediction_sources="predictions.json", source_artifact_audit="audit.json",
                output_path=str(output.with_name("contaminated.json")),
            )
        )


def test_step9_late_runner_writes_stack_only_candidate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"checkpoint_hash": "a" * 64}), encoding="utf-8")
    registry = {
        "manifest_hash": "d" * 64, "source_artifact_audit_hash": "e" * 64,
        "members": {
            "A0": {"splits": {"stack_val": {"metadata_path": str(metadata)}}},
            "P7b": {"splits": {"stack_val": {"metadata_path": str(metadata)}}},
        },
    }
    monkeypatch.setattr(selection_module, "require_development_prediction_sources", lambda *_args, **_kwargs: registry)

    def fake_fit(config):
        payload = {
            "parameters": {}, "metrics": {
                split: {"multiclass": {"accuracy": 0.8}, "binary_projection": _binary()}
                for split in ("stack_train", "stack_val")
            },
            "trainable_parameter_count": 0, "artifact_hash": "f" * 64,
        }
        Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config.output_path).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(selection_module, "fit_late_fusion_campaign_candidate", fake_fit)
    report = run_fusion_candidate(
        FusionCandidateRunConfig(
            campaign_id="toy", group_id="F_method", candidate_id="L0_mean_logits",
            output_dir=str(tmp_path / "candidate"), prediction_sources="predictions.json",
            source_artifact_audit="audit.json",
        )
    )
    assert set(report["metrics"]) == {"stack_train", "stack_val"}
    assert report["final_test_opened"] is False
    assert (tmp_path / "candidate" / "candidate_report.json").is_file()


def test_step9_representation_screening_uses_seed_5101_and_locked_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"checkpoint_hash": "a" * 64}), encoding="utf-8")
    registry = {
        "manifest_hash": "d" * 64, "source_artifact_audit_hash": "e" * 64,
        "members": {
            "A0": {"splits": {"stack_val": {"metadata_path": str(metadata)}}},
            "P7b": {"splits": {"stack_val": {"metadata_path": str(metadata)}}},
        },
    }
    monkeypatch.setattr(selection_module, "require_development_prediction_sources", lambda *_args, **_kwargs: registry)
    feature_root = tmp_path / "features"
    for member in ("A0", "P7b"):
        manifest = {
            "ok": True, "contract": "local_residual_field_fusion_feature_manifest_v1",
            "member_id": member,
        }
        manifest["manifest_hash"] = stable_fusion_json_hash(manifest)
        member_root = feature_root / member
        member_root.mkdir(parents=True)
        (member_root / "representation_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
    trial_counter = {"value": 0}

    def fake_trial(_config, *, seed, hyperparameters):
        trial_counter["value"] += 1
        fit = tmp_path / f"trial_{trial_counter['value']}.json"
        checkpoint = tmp_path / f"head_{trial_counter['value']}.pt"
        fit.write_text("{}", encoding="utf-8")
        checkpoint.write_text("head", encoding="utf-8")
        import hashlib
        return {
            "trial_id": str(trial_counter["value"]), "seed": seed,
            "hyperparameters": dict(hyperparameters), "trainable_parameter_count": 100,
            "metrics": {
                split: {"multiclass": {"accuracy": 0.8, "cross_entropy": float(hyperparameters["weight_decay"])}}
                for split in ("stack_train", "stack_val")
            },
            "checkpoint_path": str(checkpoint), "checkpoint_hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "train_report_path": str(fit), "train_report_sha256": hashlib.sha256(fit.read_bytes()).hexdigest(),
        }

    averaged_metrics = {
        split: {"multiclass": {"accuracy": 0.8, "cross_entropy": 0.2}, "binary_projection": _binary()}
        for split in ("stack_train", "stack_val")
    }
    monkeypatch.setattr(selection_module, "_representation_trial", fake_trial)
    monkeypatch.setattr(selection_module, "_average_head_metrics", lambda *_args, **_kwargs: (averaged_metrics, {}))
    report = run_fusion_candidate(
        FusionCandidateRunConfig(
            campaign_id="toy", group_id="F_method", candidate_id="R1_mlp_embeddings_logits",
            output_dir=str(tmp_path / "candidate"), prediction_sources="predictions.json",
            source_artifact_audit="audit.json", feature_root=str(feature_root),
        )
    )
    assert report["head_seeds"] == [5101]
    assert len(report["hyperparameter_trials"]) == 8
    assert report["selected_hyperparameters"]["weight_decay"] == 1.0e-5

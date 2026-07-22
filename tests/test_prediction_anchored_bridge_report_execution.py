from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teacher_logit_reco.local_particle_residual_field import (
    PREDICTION_ANCHORED_CLEAN_RELOAD_AUDIT_CONTRACT,
    PREDICTION_ANCHORED_CONSUMER_PUBLICATION_CONTRACT,
    PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT,
    PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT,
    PREDICTION_ANCHORED_REPLICA_AGGREGATE_CONTRACT,
    PREDICTION_ANCHORED_REPORT_EVIDENCE_CONTRACT,
    STEP3_RUN_IDS,
    T10_ALL50_CLEAN,
    T10_CLEAN,
    T10_ROBUST,
    build_campaign_registry,
    build_step9_reports_from_publications,
)
from teacher_logit_reco.local_particle_residual_field.bridge_evaluation import (
    PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
    PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
)
from teacher_logit_reco.local_particle_residual_field.bridge_reconstruction_execution import (
    PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT,
    PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT,
)
from teacher_logit_reco.local_particle_residual_field.bridge_splits import (
    PREDICTION_ANCHORED_SPLIT_CONTRACT,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
    sha256_file,
    with_content_hash,
    write_immutable_json,
)


PAIRED = (101, 202, 303)
TEACHERS = {T10_CLEAN, T10_ROBUST, T10_ALL50_CLEAN}


def _checkpoint(path: Path, *, training_metadata=None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {"weight": torch.zeros(2)},
            **({} if training_metadata is None else {"training_metadata": training_metadata}),
        },
        path,
    )
    return sha256_file(path)


def _classification(accuracy: float) -> dict:
    return {
        "accuracy": accuracy,
        "macro_per_class_accuracy": accuracy - 0.01,
        "cross_entropy": 1.0 - accuracy,
    }


def _teacher_aggregate(run_id: str) -> dict:
    replicas = []
    for seed, offset in zip(PAIRED, (-0.001, 0.0, 0.001)):
        f0 = _classification(0.70 + offset)
        bridge = _classification(0.72 + offset)
        replicas.append(
            {
                "seed_id": seed,
                "checkpoint_sha256": str(seed % 10) * 64,
                "f0": f0,
                "bridge_0p10": bridge,
                "delta_same": 0.02,
                "diagnostics": {
                    "oracle_physical45": _classification(0.75 + offset),
                    "oracle_all50": _classification(0.76 + offset),
                    "zero_field_consumer_diagnostic": _classification(0.20 + offset),
                },
                "negative_controls": {"zero_delta": _classification(0.70 + offset)},
            }
        )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
            "run_id": run_id,
            "paired_seed_ids": list(PAIRED),
            "median_seed_id": 202,
            "mean_delta_same": 0.02,
            "replica_metrics": replicas,
        }
    )


def _generic_aggregate(run_id: str, *, reconstruction: bool) -> dict:
    replicas = []
    for seed, offset in zip(PAIRED, (-0.001, 0.0, 0.001)):
        metrics = {
            "optimizer_steps_completed": 25,
            "model_val_stop": _classification(0.69 + offset),
        }
        replicas.append({"seed_id": seed, "metrics": metrics})
    return with_content_hash(
        {
            "contract": (
                PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT
                if reconstruction
                else PREDICTION_ANCHORED_REPLICA_AGGREGATE_CONTRACT
            ),
            "run_id": run_id,
            "paired_seed_ids": list(PAIRED),
            "median_seed_id": 202,
            "replica_metrics": replicas,
        }
    )


def _publication(root: Path, run_id: str, aggregate: dict, *, reconstruction: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "median_weights.pt"
    digest = _checkpoint(checkpoint)
    publication = with_content_hash(
        {
            "contract": (
                PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT
                if reconstruction
                else PREDICTION_ANCHORED_CONSUMER_PUBLICATION_CONTRACT
            ),
            "run_id": run_id,
            "aggregate_sha256": aggregate["content_hash"],
            "median_seed_id": 202,
            "retained_checkpoint": checkpoint.name,
            "retained_checkpoint_sha256": digest,
        }
    )
    write_immutable_json(root / "aggregate_metrics.json", aggregate)
    write_immutable_json(root / "publication.json", publication)


def _campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import teacher_logit_reco.local_particle_residual_field.bridge_report_execution as reports

    monkeypatch.setattr(
        reports, "validate_prediction_anchored_execution_spec", lambda *args, **kwargs: {"ok": True}
    )
    registry = build_campaign_registry(alternate_teacher_valid=False)
    child = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_SPLIT_CONTRACT,
            "children": {
                "stack_train_consumer": {"content_hash": "1" * 64},
                "stack_train_distill": {"content_hash": "2" * 64},
            },
        }
    )
    child_path = tmp_path / "child.json"
    write_immutable_json(child_path, child)
    baseline_path = tmp_path / "baseline.pt"
    baseline_sha = _checkpoint(baseline_path)
    execution_spec = with_content_hash(
        {
            "contract": "prediction_anchored_execution_spec_v1",
            "child_manifest": {"path": str(child_path), "content_hash": child["content_hash"]},
            "baseline_checkpoint": {"path": str(baseline_path), "sha256": baseline_sha},
            "consumer_training": {"baseline_steps": 100, "bridge_finetune_steps": 20},
        }
    )
    execution_path = tmp_path / "execution.json"
    write_immutable_json(execution_path, execution_spec)

    root = tmp_path / "artifacts"
    aggregates = {}
    for run_id in STEP3_RUN_IDS:
        aggregate = (
            _teacher_aggregate(run_id)
            if run_id in TEACHERS
            else _generic_aggregate(run_id, reconstruction=False)
        )
        aggregates[run_id] = aggregate
        _publication(root / "consumers" / run_id, run_id, aggregate, reconstruction=False)
        if run_id in TEACHERS:
            write_immutable_json(
                root / "consumer_evaluations" / run_id / "selection_aggregate.json",
                aggregate,
            )
    for row in registry["runs"]:
        run_id = row["canonical_run_id"]
        if run_id in STEP3_RUN_IDS or row["execution_status"] != "RUNNABLE":
            continue
        aggregate = _generic_aggregate(run_id, reconstruction=True)
        aggregates[run_id] = aggregate
        _publication(root / "reconstructors" / run_id, run_id, aggregate, reconstruction=True)

    selected = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
            "status": "CONFIRMED_LOCKED",
            "selected_consumer_recipe": T10_CLEAN,
            "recipe_aggregate_metrics": aggregates[T10_CLEAN],
        }
    )
    write_immutable_json(root / "selection" / "selected_bridge_consumer.json", selected)

    selected_run = "D10_A3_hlg_primary"
    replicas = []
    for seed, accuracy in zip(PAIRED, (0.709, 0.710, 0.711)):
        replicas.append(
            with_content_hash(
                {
                    "contract": "prediction_anchored_deployable_replica_evidence_v1",
                    "run_id": selected_run,
                    "seed_id": seed,
                    "accuracy": accuracy,
                    "macro_per_class_accuracy": accuracy - 0.01,
                    "cross_entropy": 1.0 - accuracy,
                    "teacher_bridge_gain": 0.02,
                    "deployable_gain": accuracy - 0.70,
                    "recovery_fraction": (accuracy - 0.70) / 0.02,
                }
            )
        )
    deployable_aggregate = with_content_hash(
        {
            "contract": "prediction_anchored_deployable_configuration_aggregate_v1",
            "run_id": selected_run,
            "replicas": replicas,
            "median_seed_id": 202,
            "mean_recovery_fraction": 0.5,
        }
    )
    pre = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT,
            "configuration_aggregate": deployable_aggregate,
        }
    )
    write_immutable_json(root / "selection" / "deployable_preconfirmation.json", pre)
    locked = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT,
            "preconfirmation_sha256": pre["content_hash"],
            "selected_aggregate_sha256": deployable_aggregate["content_hash"],
        }
    )
    write_immutable_json(root / "selection" / "locked_deployable.json", locked)
    audit = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CLEAN_RELOAD_AUDIT_CONTRACT,
            "locked_deployable_sha256": locked["content_hash"],
            "passed": True,
        }
    )
    write_immutable_json(root / "deployable_bundle" / "clean_reload_audit.json", audit)
    write_immutable_json(
        root / "telemetry" / "pack.json",
        with_content_hash({"contract": "synthetic_telemetry_v1", "ram_peak_reserved_bytes": 1234}),
    )
    return registry, execution_path, root


def test_automatic_report_evidence_uses_publications_and_clean_reload(tmp_path, monkeypatch):
    registry, execution_path, root = _campaign(tmp_path, monkeypatch)
    evidence, report = build_step9_reports_from_publications(
        registry, execution_spec_path=execution_path, artifact_root=root
    )
    assert evidence["contract"] == PREDICTION_ANCHORED_REPORT_EVIDENCE_CONTRACT
    assert evidence["manual_report_evidence_used"] is False
    assert len(evidence["run_outcomes"]) == 54
    assert set(evidence["run_outcomes"].values()) == {"COMPLETED", "SKIPPED_INVALID_PARENT"}
    baseline = {row["row_id"]: row for row in evidence["baseline_deployable_rows"]}
    assert baseline["A0_legacy"]["status"] == "REFERENCE_ONLY_UNPAIRED"
    assert baseline["selected_T10(fhat:D10_A3_hlg_primary)"]["hlt_only_reload_passed"] is True
    privileged = {row["row_id"] for row in evidence["privileged_rows"]}
    assert "selected_T10(physical45_oracle_ceiling)" in privileged
    assert "selected_T10(full50_oracle_ceiling)" in privileged
    assert report["all_registry_rows_visible"] is True
    assert report["baseline_deployable_row_count"] == 10
    assert evidence["persistent_telemetry"]["checkpoint_bytes"] > 0
    assert evidence["ram_telemetry"]["observed_peak_ram_bytes"] == 1234


def test_automatic_report_fails_before_clean_reload_passes(tmp_path, monkeypatch):
    registry, execution_path, root = _campaign(tmp_path, monkeypatch)
    audit_path = root / "deployable_bundle" / "clean_reload_audit.json"
    audit_path.unlink()
    write_immutable_json(
        audit_path,
        with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_CLEAN_RELOAD_AUDIT_CONTRACT,
                "locked_deployable_sha256": "0" * 64,
                "passed": False,
            }
        ),
    )
    with pytest.raises(PermissionError, match="clean HLT-only"):
        build_step9_reports_from_publications(
            registry, execution_spec_path=execution_path, artifact_root=root
        )


def test_report_export_shell_no_longer_requires_manual_evidence():
    text = Path("sbatch/run_prepare_prediction_anchored_bridge_ram.sh").read_text(
        encoding="utf-8"
    )
    block = text.split("REPORT_EXPORT)", 1)[1].split(";;", 1)[0]
    assert "PAB_REPORT_EVIDENCE" not in block
    assert "--artifact-root" in block and "--execution-spec" in block
    assert block.index("deploy_prediction_anchored_bridge.py export") < block.index(
        "evaluate_prediction_anchored_bridge_campaign.py reports"
    )

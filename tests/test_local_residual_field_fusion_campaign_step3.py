from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block
from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    SPLIT_ORDER,
    JetIdentity,
    SplitManifest,
    manifest_hash,
    save_split_manifest,
)
from teacher_logit_reco.local_particle_residual_field import (
    LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT,
    LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT_CONTRACT,
    LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT,
    FusionSourceArtifactAuditConfig,
    audit_fusion_source_artifacts,
    require_fusion_source_artifact_audit,
    sha256_file,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(path: Path) -> tuple[SplitManifest, dict[str, list[JetIdentity]]]:
    splits = {name: [] for name in SPLIT_ORDER}
    labels = (0, 3, 1, 8, 7, 6)
    for split in ("stack_train", "stack_val"):
        splits[split] = [
            JetIdentity(file=f"{split}.root", entry=index, label=label)
            for index, label in enumerate(labels)
        ]
    manifest = SplitManifest(
        data_dir="/synthetic/jetclass",
        max_constits=128,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={name: len(rows) for name, rows in splits.items()},
        split_seeds={name: index + 1 for index, name in enumerate(SPLIT_ORDER)},
        file_records=[],
        splits=splits,
    )
    save_split_manifest(manifest, path)
    return manifest, splits


def _prediction_block(
    root: Path,
    *,
    member: str,
    split: str,
    checkpoint_hash: str,
    manifest_sha: str,
    hlt_hash: str,
    identities: list[JetIdentity],
) -> None:
    labels = np.asarray([identity.label for identity in identities], dtype=np.int64)
    logits = np.zeros((len(labels), len(LABEL_NAMES)), dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 2.0
    save_prediction_block(
        PredictionBlock(
            model_name=member,
            split=split,
            logits=logits,
            probs=np.zeros_like(logits),
            labels=labels,
            jet_ids=identities,
            metadata={
                "contract": LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT,
                "checkpoint_hash": checkpoint_hash,
                "runtime_inputs": "HLT_only",
                "uses_true_fields": False,
                "uses_offline_particles": False,
                "uses_teacher_logits_at_runtime": False,
                "deployable": True,
                "selection_allowed": True,
                "dataset_metadata": {
                    "alignment_report": {
                        "source_manifest_hash": manifest_sha,
                        "hlt_content_hash": hlt_hash,
                    }
                },
            },
        ),
        root,
    )


def build_source_fixture(tmp_path: Path) -> tuple[FusionSourceArtifactAuditConfig, dict]:
    manifest_path = tmp_path / "split_manifest.json.gz"
    manifest, split_ids = _manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    hlt_hashes = {split: f"hlt-content-{split}" for split in ("stack_train", "stack_val")}
    hlt_manifest = tmp_path / "reused_inputs_report.json"
    _write_json(
        hlt_manifest,
        {
            "ok": True,
            "source_manifest_hash": manifest_sha,
            "audits": {
                "hlt_cache": [
                    {
                        "path": str(tmp_path / f"{split}_fixed_hlt_metadata.json"),
                        "content_hashes": {"hlt_content_hash": hlt_hashes[split]},
                    }
                    for split in ("stack_train", "stack_val")
                ]
            },
        },
    )
    a0_checkpoint = tmp_path / "A0" / "best_model_val.pt"
    p7b_checkpoint = tmp_path / "P7b" / "best_model_val.pt"
    a0_checkpoint.parent.mkdir(parents=True)
    p7b_checkpoint.parent.mkdir(parents=True)
    a0_checkpoint.write_bytes(b"synthetic-a0-checkpoint")
    p7b_checkpoint.write_bytes(b"synthetic-p7b-deployable-checkpoint")
    a0_hash = sha256_file(a0_checkpoint)
    p7b_hash = sha256_file(p7b_checkpoint)
    a0_report = tmp_path / "A0" / "run_report.json"
    p7b_report = tmp_path / "P7b" / "run_report.json"
    _write_json(a0_report, {"ok": True, "field_source": "hlt_only", "checkpoint_hash": a0_hash})
    _write_json(
        p7b_report,
        {
            "ok": True,
            "checkpoint_hash": p7b_hash,
            "selected_consumer_id": "Ofull",
            "selected_alpha_endpoint": 0.75,
            "runtime_inputs": "HLT_only",
            "uses_true_fields": False,
            "uses_offline_particles": False,
            "uses_teacher_logits_at_runtime": False,
            "deployable": True,
        },
    )
    selected = tmp_path / "selected_consumer.json"
    curve = {"0.0": {"accuracy": 0.7}, "0.75": {"accuracy": 0.75}}
    _write_json(
        selected,
        {
            "contract": "local_residual_field_selected_consumer_v1",
            "selected_consumer_id": "Ofull",
            "selected_alpha_endpoint": 0.75,
            "selection_source": "stack_val",
            "selection_reason": "predeclared pilot selector",
            "model_val_alpha_curve": curve,
            "stack_val_alpha_curve": curve,
        },
    )
    predictions = tmp_path / "predictions"
    for member, checkpoint_hash in (("A0", a0_hash), ("P7b", p7b_hash)):
        for split in ("stack_train", "stack_val"):
            _prediction_block(
                predictions,
                member=member,
                split=split,
                checkpoint_hash=checkpoint_hash,
                manifest_sha=manifest_sha,
                hlt_hash=hlt_hashes[split],
                identities=split_ids[split],
            )
        _write_json(
            predictions / member / "prediction_manifest.json",
            {"ok": True, "contract": LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT, "model_name": member},
        )
    config = FusionSourceArtifactAuditConfig(
        output_path=str(tmp_path / "campaign" / "source_artifact_audit.json"),
        a0_checkpoint=str(a0_checkpoint),
        a0_report=str(a0_report),
        p7b_checkpoint=str(p7b_checkpoint),
        p7b_report=str(p7b_report),
        selected_consumer_json=str(selected),
        manifest_path=str(manifest_path),
        hlt_cache_manifest=str(hlt_manifest),
        a0_prediction_dir=str(predictions),
        p7b_prediction_dir=str(predictions),
    )
    payload = {
        "contract": LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT,
        "oracle_consumer_included": False,
        "metadata": {"runtime_inputs": "HLT_only"},
    }
    return config, payload


class _OracleFreeModel:
    oracle_consumer = None


def test_step3_audit_hashes_sources_predictions_and_loads_p7b_without_oracle(tmp_path: Path) -> None:
    config, checkpoint_payload = build_source_fixture(tmp_path)
    calls = []

    def loader(path: str, *, device: str):
        calls.append((path, device))
        return _OracleFreeModel(), checkpoint_payload

    report = audit_fusion_source_artifacts(config, model_loader=loader)

    assert report["ok"] is True
    assert report["contract"] == LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT_CONTRACT
    assert report["p7b_oracle_free_load"]["ok"] is True
    assert report["final_test_opened"] is False
    assert calls == [(config.p7b_checkpoint, "cpu")]
    assert set(report["reusable_predictions"]) == {"A0", "P7b"}
    assert len(report["hashed_files"]) == 17
    assert require_fusion_source_artifact_audit(config.output_path)["audit_hash"] == report["audit_hash"]


def test_step3_gpu_gate_rejects_any_source_drift_after_audit(tmp_path: Path) -> None:
    config, payload = build_source_fixture(tmp_path)
    report = audit_fusion_source_artifacts(
        config,
        model_loader=lambda path, device="cpu": (_OracleFreeModel(), payload),
    )
    assert report["ok"] is True
    Path(config.a0_report).write_text("{}\n", encoding="utf-8")

    try:
        require_fusion_source_artifact_audit(config.output_path)
    except ValueError as exc:
        assert "changed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tampered source report was accepted")


def test_step3_audit_fails_if_loaded_p7b_contains_oracle_consumer(tmp_path: Path) -> None:
    config, payload = build_source_fixture(tmp_path)

    class BadModel:
        oracle_consumer = object()

    report = audit_fusion_source_artifacts(
        config,
        model_loader=lambda path, device="cpu": (BadModel(), payload),
    )

    assert report["ok"] is False
    assert report["p7b_oracle_free_load"]["ok"] is False
    assert "oracle consumer" in " ".join(report["problems"])

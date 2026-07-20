from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block
from jetclass_fresh.jetclass_data import JetIdentity, SPLIT_ORDER, SplitManifest, manifest_hash, save_split_manifest
from teacher_logit_reco.local_particle_residual_field.oracle_teacher import (
    ORACLE_TEACHER_LOGIT_SPLITS,
    ORACLE_TEACHER_TRAIN_SPLITS,
    build_oracle_teacher_reuse_contract,
    validate_oracle_teacher_reuse_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _teacher_payload(*, target_suffix: str = "", field_source: str = "oracle") -> dict:
    dataset_metadata = {}
    for split in ORACLE_TEACHER_TRAIN_SPLITS:
        dataset_metadata[split] = {
            "alignment_report": {
                "source_manifest_hash": "manifest_hash",
                "hlt_content_hash": f"hlt_{split}",
                "offline_content_hash": f"offline_{split}",
                "target_content_hash": f"target_{split}{target_suffix}",
                "jet_identity_hash": f"identity_{split}",
            }
        }
    return {
        "contract": "local_residual_field_oracle_teacher_config_v1",
        "teacher_id": "B1",
        "field_source": field_source,
        "oracle_field_alpha": 1.0,
        "oracle_field_noise_std": 0.0,
        "oracle_field_dropout": 0.0,
        "oracle_field_group_dropout": 0.0,
        "field_subset": [],
        "selected_field_indices": [0, 1],
        "selected_field_names": ["r0p02.delta_log_pt_sum", "r0p02.delta_pt_frac"],
        "selected_field_groups": {"pt_density": [0, 1]},
        "model_config": {
            "contract": "local_particle_residual_field_augmented_part_v1",
            "num_classes": 2,
            "field_dim": 2,
            "base_feature_dim": 17,
            "augmented_feature_dim": 19,
            "model_size": "base",
            "field_source": field_source,
            "field_names": ["r0p02.delta_log_pt_sum", "r0p02.delta_pt_frac"],
            "field_groups": {"pt_density": [0, 1]},
        },
        "train_config": {
            "num_classes": 2,
            "label_names": ["a", "b"],
            "model_size": "base",
            "field_subset": [],
        },
        "dataset_metadata": dataset_metadata,
    }


def _write_source_run(path: Path, payload: dict) -> None:
    path.mkdir()
    (path / "best_model_val.pt").write_bytes(b"checkpoint")
    (path / "training_curves.json").write_text('{"epochs": []}', encoding="utf-8")
    (path / "teacher_config.json").write_text(json.dumps(payload), encoding="utf-8")
    (path / "run_report.json").write_text(
        json.dumps(
            {
                "ok": True,
                "field_source": payload["field_source"],
                "model_config": payload["model_config"],
                "config": payload["train_config"],
                "dataset_metadata": payload["dataset_metadata"],
                "selected_field_indices": payload["selected_field_indices"],
                "selected_field_names": payload["selected_field_names"],
                "selected_field_groups": payload["selected_field_groups"],
            }
        ),
        encoding="utf-8",
    )


def _registration_command(source: Path, output: Path, expected_config: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "register_local_residual_oracle_teacher.py"),
        "--source-run-dir",
        str(source),
        "--output-dir",
        str(output),
        "--teacher-id",
        "Ofull",
        "--expected-field-source",
        "oracle_scaled",
        "--expected-alpha",
        "1.0",
        "--expected-noise-std",
        "0.0",
        "--expected-field-dropout",
        "0.0",
        "--expected-group-dropout",
        "0.0",
        "--link-mode",
        "copy",
    ]
    if expected_config is not None:
        command.extend(("--expected-teacher-config", str(expected_config)))
    return command


def test_step2_reuse_contract_covers_all_required_provenance_dimensions():
    contract = build_oracle_teacher_reuse_contract(_teacher_payload())

    assert contract["source_manifest_hash"] == "manifest_hash"
    assert set(contract["split_provenance"]) == set(ORACLE_TEACHER_TRAIN_SPLITS)
    assert all(contract[name] for name in (
        "split_provenance_hash",
        "field_schema_hash",
        "label_order_hash",
        "field_recipe_hash",
        "model_architecture_hash",
        "reuse_contract_hash",
    ))

    mismatch = build_oracle_teacher_reuse_contract(_teacher_payload(target_suffix="_different"))
    validation = validate_oracle_teacher_reuse_contract(contract, mismatch)
    assert validation["ok"] is False
    assert "split_provenance_hash" in {item["field"] for item in validation["mismatches"]}


def test_step2_register_existing_oracle_teacher_validates_and_materializes(tmp_path: Path):
    source = tmp_path / "B1"
    source_payload = _teacher_payload(field_source="oracle")
    _write_source_run(source, source_payload)
    expected_path = tmp_path / "expected_teacher_config.json"
    expected_path.write_text(json.dumps(_teacher_payload(field_source="oracle_scaled")), encoding="utf-8")
    output = tmp_path / "Ofull"

    result = subprocess.run(
        _registration_command(source, output, expected_path),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"teacher_id": "Ofull"' in result.stdout
    assert (output / "best_model_val.pt").read_bytes() == b"checkpoint"
    assert (output / "training_curves.json").exists()
    registered = json.loads((output / "teacher_config.json").read_text(encoding="utf-8"))
    report = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
    registration = json.loads((output / "registration_report.json").read_text(encoding="utf-8"))
    assert registered["teacher_id"] == "Ofull"
    assert registered["registered_from"] == str(source)
    assert registered["reuse_validation"]["ok"] is True
    assert report["teacher_reuse_contract_hash"] == registered["reuse_contract"]["reuse_contract_hash"]
    assert registration["contract"] == "local_residual_field_oracle_teacher_registration_v2"
    assert registration["unverified_legacy_override"] is False


def test_step2_registration_fails_closed_without_provenance_or_on_cache_mismatch(tmp_path: Path):
    source = tmp_path / "B1"
    _write_source_run(source, _teacher_payload())

    no_provenance = subprocess.run(
        _registration_command(source, tmp_path / "no_provenance"),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert no_provenance.returncode != 0
    assert "provenance verification is required" in no_provenance.stderr

    mismatched_path = tmp_path / "mismatched.json"
    mismatched_path.write_text(json.dumps(_teacher_payload(target_suffix="_other")), encoding="utf-8")
    mismatch = subprocess.run(
        _registration_command(source, tmp_path / "mismatch", mismatched_path),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode != 0
    assert "reuse provenance mismatch" in mismatch.stderr


def test_step2_registration_scans_candidates_for_first_provenance_match(tmp_path: Path):
    stale_source = tmp_path / "B1_stale"
    matching_source = tmp_path / "B4_matching"
    _write_source_run(stale_source, _teacher_payload(target_suffix="_stale"))
    _write_source_run(matching_source, _teacher_payload(field_source="oracle"))
    expected_path = tmp_path / "expected_teacher_config.json"
    expected_path.write_text(
        json.dumps(_teacher_payload(field_source="oracle_scaled")),
        encoding="utf-8",
    )
    output = tmp_path / "Ofull"
    command = _registration_command(stale_source, output, expected_path)
    command.extend(("--candidate-run-dir", str(matching_source)))

    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    registered = json.loads((output / "teacher_config.json").read_text(encoding="utf-8"))
    registration = json.loads((output / "registration_report.json").read_text(encoding="utf-8"))
    assert registered["registered_from"] == str(matching_source)
    assert [item["ok"] for item in registration["candidate_audit"]] == [False, True]
    assert registration["candidate_audit"][0]["validation"]["mismatches"]
    assert registration["candidate_audit"][1]["validation"]["ok"] is True


def test_step2_registration_builds_expected_contract_from_active_cache_metadata(tmp_path: Path):
    source = tmp_path / "B1"
    source_payload = _teacher_payload(field_source="oracle")
    _write_source_run(source, source_payload)

    manifest = SplitManifest(
        data_dir="unused",
        max_constits=128,
        class_names=["a", "b"],
        file_prefix_to_label={},
        split_sizes={split: 0 for split in SPLIT_ORDER},
        split_seeds={split: index for index, split in enumerate(SPLIT_ORDER)},
        file_records=[],
        splits={split: [] for split in SPLIT_ORDER},
    )
    manifest_path = tmp_path / "manifest.json.gz"
    save_split_manifest(manifest, manifest_path)
    source_manifest_hash = manifest_hash(manifest)
    # Match the source teacher to the exact active manifest used by registration.
    for metadata in source_payload["dataset_metadata"].values():
        metadata["alignment_report"]["source_manifest_hash"] = source_manifest_hash
    _write_source_run(tmp_path / "B1_active", source_payload)
    active_source = tmp_path / "B1_active"

    hlt_dir = tmp_path / "hlt"
    target_dir = tmp_path / "targets"
    hlt_dir.mkdir()
    target_dir.mkdir()
    for split in ORACLE_TEACHER_TRAIN_SPLITS:
        (hlt_dir / f"{split}_fixed_hlt_metadata.json").write_text(
            json.dumps(
                {
                    "source_manifest_hash": source_manifest_hash,
                    "hlt_content_hash": f"hlt_{split}",
                    "jet_identity_hash": f"identity_{split}",
                }
            ),
            encoding="utf-8",
        )
        (target_dir / f"{split}_local_particle_residual_fields_metadata.json").write_text(
            json.dumps(
                {
                    "source_manifest_hash": source_manifest_hash,
                    "hlt_content_hash": f"hlt_{split}",
                    "offline_content_hash": f"offline_{split}",
                    "target_content_hash": f"target_{split}",
                    "hlt_jet_identity_hash": f"identity_{split}",
                    "field_names": ["r0p02.delta_log_pt_sum", "r0p02.delta_pt_frac"],
                    "field_groups": {"pt_density": [0, 1]},
                    "label_names": ["a", "b"],
                }
            ),
            encoding="utf-8",
        )

    command = _registration_command(active_source, tmp_path / "Ofull_active")
    command.extend(
        (
            "--manifest-path",
            str(manifest_path),
            "--hlt-cache-dir",
            str(hlt_dir),
            "--target-cache-dir",
            str(target_dir),
            "--expected-label-names",
            "a",
            "b",
            "--expected-num-classes",
            "2",
        )
    )
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    registration = json.loads(
        (tmp_path / "Ofull_active" / "registration_report.json").read_text(encoding="utf-8")
    )
    assert result.returncode == 0
    assert registration["reuse_validation"]["ok"] is True
    assert registration["actual_reuse_contract"]["source_manifest_hash"] == source_manifest_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_step2_oracle_logit_cache_validator_checks_all_splits_and_provenance(tmp_path: Path):
    teacher_id = "Ofull"
    teacher_dir = tmp_path / "teachers" / teacher_id
    teacher_dir.mkdir(parents=True)
    checkpoint = teacher_dir / "best_model_val.pt"
    checkpoint.write_bytes(b"checkpoint")
    teacher_config = _teacher_payload(field_source="oracle_scaled")
    teacher_config["teacher_id"] = teacher_id
    teacher_config["role"] = "oracle_teacher_candidate"
    teacher_config["reuse_contract"] = build_oracle_teacher_reuse_contract(teacher_config)
    teacher_config_path = teacher_dir / "teacher_config.json"
    teacher_config_path.write_text(json.dumps(teacher_config, sort_keys=True), encoding="utf-8")
    (teacher_dir / "run_report.json").write_text('{"ok": true}', encoding="utf-8")

    prediction_dir = tmp_path / "oracle_teacher_logits"
    checkpoint_hash = _sha256(checkpoint)
    teacher_config_hash = _sha256(teacher_config_path)
    jet_ids = [
        JetIdentity(file="sample.root", entry=index, label=index % 2)
        for index in range(4)
    ]
    labels = np.asarray([identity.label for identity in jet_ids], dtype=np.int64)
    logits = np.asarray([[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5], [-0.5, 0.5]], dtype=np.float32)
    for split in ORACLE_TEACHER_LOGIT_SPLITS:
        metadata = {
            "contract": "local_particle_residual_field_predictions_v1",
            "model_contract": "local_particle_residual_field_augmented_part_v1",
            "checkpoint": str(checkpoint),
            "checkpoint_hash": checkpoint_hash,
            "teacher_id": teacher_id,
            "teacher_role": "oracle_teacher_candidate",
            "teacher_config": str(teacher_config_path),
            "teacher_config_hash": teacher_config_hash,
            "teacher_reuse_contract_hash": teacher_config["reuse_contract"]["reuse_contract_hash"],
            "field_source": "oracle_scaled",
            "runtime_inputs": "HLT_plus_true_residual_fields",
            "uses_true_fields": True,
            "deployable": False,
        }
        save_prediction_block(
            PredictionBlock(
                model_name=teacher_id,
                split=split,
                logits=logits,
                probs=np.zeros_like(logits),
                labels=labels,
                jet_ids=jet_ids,
                metadata=metadata,
            ),
            prediction_dir,
        )
        source_meta = prediction_dir / teacher_id / f"{split}_predictions_metadata.json"
        shutil.copy2(source_meta, prediction_dir / teacher_id / f"{split}_metadata.json")
    (prediction_dir / teacher_id / "prediction_manifest.json").write_text(
        json.dumps({"model_name": teacher_id, "splits": list(ORACLE_TEACHER_LOGIT_SPLITS)}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_local_residual_oracle_teacher_logits.py"),
            "--teacher-dir",
            str(teacher_dir),
            "--prediction-dir",
            str(prediction_dir),
            "--teacher-id",
            teacher_id,
            "--splits",
            *ORACLE_TEACHER_LOGIT_SPLITS,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"final_test_oracle_logits_present": false' in result.stdout
    report = json.loads(
        (prediction_dir / teacher_id / "cache_validation_report.json").read_text(encoding="utf-8")
    )
    assert report["ok"] is True
    assert report["splits"] == list(ORACLE_TEACHER_LOGIT_SPLITS)
    assert set(report["split_reports"]) == set(ORACLE_TEACHER_LOGIT_SPLITS)

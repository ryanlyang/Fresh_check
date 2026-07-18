from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from teacher_logit_reco.constrained_coarse_to_fine.runtime import (
    build_runtime_profile,
    precision_mode_metadata,
    profile_requires_last_checkpoint,
)
from teacher_logit_reco.constrained_coarse_to_fine.runtime_profiles import (
    write_approved_profile,
)


def _canonical_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_bf16_exploratory_pilot_profile_is_nonresumable_bf16() -> None:
    profile = build_runtime_profile(
        profile="bf16_exploratory_pilot_v1",
        precision_mode="bf16_forward_fp32_loss",
        batch_size=16,
        eval_batch_size=32,
        num_workers=4,
        prefetch_factor=2,
        learning_rate=2.0e-4,
        hlt_encoder_lr_scale=0.05,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        lr_schedule="constant",
        warmup_fraction=0.1,
        min_lr_ratio=0.05,
        min_epochs=0,
        early_stop_patience=8,
        fixed_horizon=False,
        max_epochs=30,
        hungarian_workers=1,
        hungarian_executor="serial",
    )

    assert profile["precision"] == precision_mode_metadata("bf16_forward_fp32_loss")
    assert profile_requires_last_checkpoint("bf16_exploratory_pilot_v1") is False


def _environment() -> dict[str, object]:
    payload: dict[str, object] = {
        "contract": "constrained_c2f_code_environment_v1",
        "source_commit": "test-commit",
        "source_tree_clean": True,
        "source_status_hash": "clean-status",
        "python_implementation": "CPython",
        "python_version": "3.10.test",
        "torch_version": "test",
        "torch_cuda_version": "test",
        "scipy_version": "test",
    }
    payload["code_environment_hash"] = _canonical_hash(payload)
    return payload


def _candidate(path, environment: dict[str, object]) -> dict[str, object]:
    profiles = {
        name: build_runtime_profile(
            profile="accelerated_candidate_v1",
            precision_mode="bf16_forward_fp32_loss",
            batch_size=16 if name != "C6" else 8,
            eval_batch_size=32 if name != "C6" else 16,
            num_workers=4,
            prefetch_factor=2,
            learning_rate=2.0e-4,
            hlt_encoder_lr_scale=0.05,
            weight_decay=1.0e-4,
            grad_clip_norm=1.0,
            lr_schedule="warmup_cosine",
            warmup_fraction=0.1,
            min_lr_ratio=0.05,
            min_epochs=5,
            early_stop_patience=2,
            fixed_horizon=False,
            max_epochs=10,
            hungarian_workers=4 if name == "C4" else 1,
            hungarian_executor="thread" if name == "C4" else "serial",
        )
        for name in ("C1", "C5-B3", "C6", "C4")
    }
    payload: dict[str, object] = {
        "contract": "constrained_c2f_accelerated_candidate_v1",
        "status": "accelerated_candidate_v1",
        "runtime_profiles_by_variant": profiles,
        "execution_by_variant": {name: {"runtime_profile_hash": profile["runtime_profile_hash"]} for name, profile in profiles.items()},
        "shared_optimizer": {"learning_rate": 2.0e-4},
        "code_environment": environment,
        "code_environment_hash": environment["code_environment_hash"],
    }
    payload["candidate_profile_hash"] = _canonical_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _pilot_inputs(root) -> tuple[object, object, object, object]:
    manifest = root / "manifest.json.gz"
    with gzip.open(manifest, "wt", encoding="utf-8") as handle:
        json.dump({"splits": {"model_train": [], "model_val": []}}, handle)
    hlt, offline, targets = root / "hlt", root / "offline", root / "targets"
    for path in (hlt, offline, targets):
        path.mkdir()
    payload = {"splits": {"model_train": [], "model_val": []}}
    manifest_hash = _canonical_hash(payload)
    for split in ("model_train", "model_val"):
        hlt_meta = {"source_manifest_hash": manifest_hash, "hlt_content_hash": f"hlt-{split}", "jet_identity_hash": f"id-{split}"}
        offline_meta = {"source_manifest_hash": manifest_hash, "offline_content_hash": f"off-{split}", "jet_identity_hash": f"id-{split}"}
        target_meta = {
            "source_manifest_hash": manifest_hash,
            "hlt_content_hash": hlt_meta["hlt_content_hash"],
            "offline_content_hash": offline_meta["offline_content_hash"],
            "target_content_hash": f"target-{split}",
            "jet_identity_hash": hlt_meta["jet_identity_hash"],
        }
        (hlt / f"{split}_fixed_hlt_metadata.json").write_text(json.dumps(hlt_meta), encoding="utf-8")
        (offline / f"{split}_offline_metadata.json").write_text(json.dumps(offline_meta), encoding="utf-8")
        (targets / f"{split}_hierarchy_targets_metadata.json").write_text(json.dumps(target_meta), encoding="utf-8")
    (targets / "hierarchy_target_cache_manifest.json").write_text("{}", encoding="utf-8")
    return manifest, hlt, offline, targets


def test_approved_profile_rejects_hand_authored_tagger_sanity_evidence(monkeypatch, tmp_path) -> None:
    environment = _environment()
    monkeypatch.setattr(
        "teacher_logit_reco.constrained_coarse_to_fine.runtime_profiles.collect_code_environment",
        lambda: environment,
    )
    monkeypatch.setattr(
        "teacher_logit_reco.constrained_coarse_to_fine.runtime_certification.validate_reconstructor_certification",
        lambda **kwargs: {"promotion_gate": {"ok": True}},
    )
    candidate_path = tmp_path / "candidate.json"
    candidate = _candidate(candidate_path, environment)
    evidence = {}
    for name, path_name in {
        "c5_ten_epoch_certification": "C5-B3",
        "c6_ten_epoch_certification": "C6",
        "c5_fp32_reference": "C5-B3",
        "c6_fp32_reference": "C6",
        "c5_tagger_sanity": "C5-B3",
        "c6_tagger_sanity": "C6",
    }.items():
        tagger_sanity = name.endswith("tagger_sanity")
        path = tmp_path / f"{name}.json"
        certification_files = {}
        if not tagger_sanity:
            candidate_dir, reference_dir = tmp_path / f"{name}_candidate", tmp_path / f"{name}_reference"
            candidate_dir.mkdir(); reference_dir.mkdir()
            for directory, profile_name, epochs in ((candidate_dir, "accelerated_candidate_v1", 10), (reference_dir, "fp32_reference", 10 if "ten_epoch" in name else 30)):
                (directory / "best_model_val.pt").write_bytes(b"checkpoint")
                (directory / "run_report.json").write_text(json.dumps({
                    "ok": True, "variant": path_name, "runtime_profile": {"name": profile_name},
                    "training_config": {"fixed_horizon": True, "epochs": epochs},
                    "completed_epochs": epochs,
                    "checkpoint_sha256": hashlib.sha256((directory / "best_model_val.pt").read_bytes()).hexdigest(),
                }), encoding="utf-8")
            certification_files = {
                "candidate": {
                    "run_report_path": str(candidate_dir / "run_report.json"),
                    "run_report_sha256": hashlib.sha256((candidate_dir / "run_report.json").read_bytes()).hexdigest(),
                    "checkpoint_path": str(candidate_dir / "best_model_val.pt"),
                    "checkpoint_sha256": hashlib.sha256((candidate_dir / "best_model_val.pt").read_bytes()).hexdigest(),
                },
                "fp32_reference": {
                    "run_report_path": str(reference_dir / "run_report.json"),
                    "run_report_sha256": hashlib.sha256((reference_dir / "run_report.json").read_bytes()).hexdigest(),
                    "checkpoint_path": str(reference_dir / "best_model_val.pt"),
                    "checkpoint_sha256": hashlib.sha256((reference_dir / "best_model_val.pt").read_bytes()).hexdigest(),
                },
                "candidate_epochs": 10,
                "fp32_reference_epochs": 10 if "ten_epoch" in name else 30,
            }
        path.write_text(
            json.dumps({
                "contract": "constrained_c2f_runtime_tagger_sanity_v1" if tagger_sanity else "constrained_c2f_runtime_reconstructor_certification_v1",
                "ok": True,
                "path": path_name,
                "candidate_profile_hash": candidate["candidate_profile_hash"],
                "code_environment_hash": environment["code_environment_hash"],
                "promotion_evidence_kind": "tagger_sanity" if tagger_sanity else ("ten_epoch_certification" if "ten_epoch" in name else "fp32_reference_promotion"),
                "promotion_gate": {"ok": True},
                **certification_files,
            }),
            encoding="utf-8",
        )
        evidence[name] = path
    manifest, hlt, offline, targets = _pilot_inputs(tmp_path)
    approved_path = tmp_path / "approved.json"
    with pytest.raises(ValueError, match="wrong evaluation contract"):
        write_approved_profile(
            candidate_profile_path=candidate_path,
            manifest_path=manifest,
            hlt_cache_dir=hlt,
            offline_cache_dir=offline,
            target_cache_dir=targets,
            output_path=approved_path,
            **evidence,
        )


def test_approved_profile_rejects_evidence_with_the_wrong_gate_kind(monkeypatch, tmp_path) -> None:
    environment = _environment()
    monkeypatch.setattr(
        "teacher_logit_reco.constrained_coarse_to_fine.runtime_profiles.collect_code_environment",
        lambda: environment,
    )
    monkeypatch.setattr(
        "teacher_logit_reco.constrained_coarse_to_fine.runtime_certification.validate_reconstructor_certification",
        lambda **kwargs: {"promotion_gate": {"ok": True}},
    )
    candidate_path = tmp_path / "candidate.json"
    candidate = _candidate(candidate_path, environment)
    evidence = {}
    for name, path_name in {
        "c5_ten_epoch_certification": "C5-B3", "c6_ten_epoch_certification": "C6",
        "c5_fp32_reference": "C5-B3", "c6_fp32_reference": "C6",
        "c5_tagger_sanity": "C5-B3", "c6_tagger_sanity": "C6",
    }.items():
        tagger_sanity = name.endswith("tagger_sanity")
        path = tmp_path / f"{name}.json"
        certification_files = {}
        if not tagger_sanity:
            candidate_dir, reference_dir = tmp_path / f"{name}_candidate", tmp_path / f"{name}_reference"
            candidate_dir.mkdir(); reference_dir.mkdir()
            for directory, profile_name, epochs in ((candidate_dir, "accelerated_candidate_v1", 10), (reference_dir, "fp32_reference", 10 if "ten_epoch" in name else 30)):
                (directory / "best_model_val.pt").write_bytes(b"checkpoint")
                (directory / "run_report.json").write_text(json.dumps({
                    "ok": True, "variant": path_name, "runtime_profile": {"name": profile_name},
                    "training_config": {"fixed_horizon": True, "epochs": epochs},
                    "completed_epochs": epochs,
                    "checkpoint_sha256": hashlib.sha256((directory / "best_model_val.pt").read_bytes()).hexdigest(),
                }), encoding="utf-8")
            certification_files = {
                "candidate": {"run_report_path": str(candidate_dir / "run_report.json"), "run_report_sha256": hashlib.sha256((candidate_dir / "run_report.json").read_bytes()).hexdigest(), "checkpoint_path": str(candidate_dir / "best_model_val.pt"), "checkpoint_sha256": hashlib.sha256((candidate_dir / "best_model_val.pt").read_bytes()).hexdigest()},
                "fp32_reference": {"run_report_path": str(reference_dir / "run_report.json"), "run_report_sha256": hashlib.sha256((reference_dir / "run_report.json").read_bytes()).hexdigest(), "checkpoint_path": str(reference_dir / "best_model_val.pt"), "checkpoint_sha256": hashlib.sha256((reference_dir / "best_model_val.pt").read_bytes()).hexdigest()},
                "candidate_epochs": 10, "fp32_reference_epochs": 10 if "ten_epoch" in name else 30,
            }
        path.write_text(json.dumps({
            "contract": "constrained_c2f_runtime_tagger_sanity_v1" if tagger_sanity else "constrained_c2f_runtime_reconstructor_certification_v1",
            "ok": True, "path": path_name,
            "candidate_profile_hash": candidate["candidate_profile_hash"],
            "code_environment_hash": environment["code_environment_hash"],
            "promotion_evidence_kind": "tagger_sanity" if tagger_sanity else "ten_epoch_certification",
            "promotion_gate": {"ok": True},
            **certification_files,
        }), encoding="utf-8")
        evidence[name] = path
    manifest, hlt, offline, targets = _pilot_inputs(tmp_path)
    with pytest.raises(ValueError, match="wrong promotion evidence kind"):
        write_approved_profile(
            candidate_profile_path=candidate_path, manifest_path=manifest,
            hlt_cache_dir=hlt, offline_cache_dir=offline, target_cache_dir=targets,
            output_path=tmp_path / "approved.json", **evidence,
        )

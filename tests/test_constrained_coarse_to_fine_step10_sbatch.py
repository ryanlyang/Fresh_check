from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays
from scripts.validate_constrained_coarse_to_fine_artifact import _npz_logical_hash

from teacher_logit_reco.constrained_coarse_to_fine import (
    D4_UNCERTAINTY_GATED,
    EndToEndScheduleConfig,
    end_to_end_phase,
)


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "sbatch"


def _read(name: str) -> str:
    return (SBATCH / name).read_text(encoding="utf-8")


def test_streaming_npz_hash_matches_project_logical_hash() -> None:
    arrays = {
        "tokens": np.arange(48, dtype=np.float32).reshape(2, 3, 8),
        "mask": np.asarray([[True, False, True], [True, True, False]], dtype=bool),
        "labels": np.asarray([1, 2], dtype=np.int64),
        "jet_file_indices": np.asarray([0, 1], dtype=np.int32),
        "jet_entries": np.asarray([4, 9], dtype=np.int64),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.npz"
        np.savez_compressed(path, **arrays)
        assert _npz_logical_hash(path, tuple(arrays)) == hash_arrays(arrays)


RUNNERS = (
    "run_cache_constrained_coarse_to_fine_targets.sh",
    "run_train_constrained_coarse_to_fine_reconstructor.sh",
    "run_train_constrained_coarse_to_fine_tagger.sh",
    "run_cache_constrained_coarse_to_fine_predictions.sh",
    "run_alias_constrained_coarse_to_fine_predictions.sh",
    "run_alias_constrained_coarse_to_fine_tagger.sh",
    "run_constrained_coarse_to_fine_fusion.sh",
    "run_write_constrained_coarse_to_fine_report.sh",
)


def test_step10_workers_are_strict_slurm_jobs() -> None:
    for name in RUNNERS:
        text = _read(name)
        assert "#!/usr/bin/env bash" in text
        assert "#SBATCH --job-name=" in text
        assert "#SBATCH --output=fresh_check_logs/%x_%j.out" in text
        assert "#SBATCH --error=fresh_check_logs/%x_%j.err" in text
        assert "set -euo pipefail" in text
        assert "fresh_setup" in text
        assert "fresh_write_run_config" in text
        assert "fresh_run" in text


def test_step10_all_requested_queue_entry_points_exist() -> None:
    names = (
        "submit_constrained_coarse_to_fine_experiment.sh",
        "submit_constrained_coarse_to_fine_pilot.sh",
        "submit_constrained_coarse_to_fine_highdata.sh",
        "submit_constrained_coarse_to_fine_pilot_and_highdata.sh",
        "submit_constrained_coarse_to_fine_targets_only.sh",
        "submit_constrained_coarse_to_fine_reconstructors_only.sh",
        "submit_constrained_coarse_to_fine_taggers_only.sh",
        "submit_constrained_coarse_to_fine_depth_d5.sh",
        "submit_constrained_coarse_to_fine_d8_only.sh",
        "submit_constrained_coarse_to_fine_report_only.sh",
        "submit_constrained_coarse_to_fine_final_claims.sh",
        "submit_constrained_coarse_to_fine_tigris_pilot_and_highdata.sh",
        "submit_constrained_coarse_to_fine_sporcsubmit_pilot_and_highdata.sh",
    )
    for name in names:
        assert (SBATCH / name).is_file(), name


def test_step10_graph_uses_hlt_v2_s2p5_and_protocol_sizes() -> None:
    text = _read("submit_constrained_coarse_to_fine_experiment.sh")
    assert "CONSTRAINED_C2F_HLT_PROFILE:=fixed_hlt_v2_realistic" in text
    assert "CONSTRAINED_C2F_HLT_DEGRADATION_STRENGTH:=2.5" in text
    assert "CONSTRAINED_C2F_MODEL_TRAIN_SIZE:=500000" in text
    assert "CONSTRAINED_C2F_MODEL_TRAIN_SIZE:=5000000" in text
    assert "CONSTRAINED_C2F_STACK_TRAIN_SIZE:=2000000" in text
    assert "CONSTRAINED_C2F_FINAL_TEST_SIZE:=1000000" in text


def test_step10_graph_queues_complete_reconstructor_and_depth_families() -> None:
    text = _read("submit_constrained_coarse_to_fine_experiment.sh")
    assert "A0 A1 A2 A4" in text
    assert "B0 B1 B2 B3 B4 B5 B6 B7" in text
    assert "C0 C1 C2 C3 C4 C5 C6 C5-B1 C5-B2 C5-B3 C5-no-slot" in text
    assert "Cdirect-unconstrained" in text
    for run_id in ("D0", "D1", "D2", "D3", "D4", "D5", "D5-B1", "D5-B2", "D5-B3", "D6", "D7", "D8"):
        assert run_id in text
    for run_id in ("E0", "E1", "E2", "E3", "E4", "E5", "E6"):
        assert run_id in text
    for run_id in ("D0", "D3", "D5-B1", "D5-B2", "D6", "D7", "D8"):
        assert f"{run_id}-seed1" in text
        assert f"{run_id}-seed2" in text
    tagger = _read("run_train_constrained_coarse_to_fine_tagger.sh")
    reconstructor = _read("run_train_constrained_coarse_to_fine_reconstructor.sh")
    assert "best_c=c5_b3" in tagger
    assert "stochastic_${index}@${index}" in tagger
    assert "c2f_tagger_D5-B3_alias" in text
    assert "run_alias_constrained_coarse_to_fine_tagger.sh" in text
    assert "run_alias_constrained_coarse_to_fine_predictions.sh" in text
    assert "--direct-particle-decoding" in reconstructor
    assert 'hierarchy_loss_weight="0.0"' in reconstructor
    assert "Cdirect-unconstrained/best_model_val.pt" in tagger
    assert "--d-model 320 --num-heads 10 --hlt-encoder-layers 8" in tagger


def test_step10_submitter_is_dependency_aware_and_records_job_ids() -> None:
    text = _read("submit_constrained_coarse_to_fine_experiment.sh")
    assert '--dependency="afterok:${dependency}"' in text
    assert 'cache_dep="$(join_dependencies "${hlt_jid}" "${offline_jid}")"' in text
    assert "declare -A recon_jids" in text
    assert "declare -A tagger_jids" in text
    assert "job_ids.tsv" in text
    assert "source_status_hash" in text


def test_step10_reuse_and_posthoc_paths_fail_closed() -> None:
    text = _read("submit_constrained_coarse_to_fine_experiment.sh")
    assert "Required HLT cache is incomplete" in text
    assert "Required hierarchy target cache is incomplete" in text
    assert "requires incomplete reconstructor" in text
    assert "CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT" in text
    assert "F2:representation_stacker:D3,D4,D5,D6,D8" in text
    assert "F4:mean_logits:BEST_D,BEST_D_SEED1,BEST_D_SEED2" in text
    assert "F5:linear_stacker:D8,D6,BEST_D,BEST_D_SEED1,BEST_D_SEED2" in text
    assert "validate_constrained_coarse_to_fine_artifact.py" in text
    assert "Report-only rerun requires complete predictions" in text
    assert "CONSTRAINED_C2F_SUBMIT_FUSION:=1" in text
    assert "CONSTRAINED_C2F_SUBMIT_REPORT:=1" in text


def test_step10_campaign_is_staged_and_final_test_is_separate() -> None:
    text = _read("submit_constrained_coarse_to_fine_pilot_and_highdata.sh")
    assert "CONSTRAINED_C2F_CAMPAIGN_MODE=pilot" in text
    assert "CONSTRAINED_C2F_CAMPAIGN_MODE=highdata" in text
    assert "--dependency" not in text
    assert "CONSTRAINED_C2F_APPROVE_HIGHDATA" in text
    assert "pilot final report is not ok" in text
    final = _read("submit_constrained_coarse_to_fine_final_claims.sh")
    assert "CONSTRAINED_C2F_APPROVE_FINAL_TEST" in final
    assert "CONSTRAINED_C2F_STAGE_MODE=final_claims" in final
    assert "final_test_claim_receipt.json" in final
    assert 'claim_ids=("${predict_ids[@]}" F0 F1 F2 F3 F4 F5)' in final
    for runner in (
        "run_cache_constrained_coarse_to_fine_predictions.sh",
        "run_alias_constrained_coarse_to_fine_predictions.sh",
    ):
        runner_text = _read(runner)
        assert "CONSTRAINED_C2F_PREDICT_SPLITS:=model_val stack_train stack_val}" in runner_text
        assert "CONSTRAINED_C2F_PREDICT_SPLITS:=model_val stack_train stack_val final_test}" not in runner_text


def test_step10_cluster_wrappers_set_correct_environments_and_resources() -> None:
    tigris = _read("submit_constrained_coarse_to_fine_tigris_pilot_and_highdata.sh")
    assert "atlas_kd_tigris" in tigris
    assert "miniforge3-aarch64" in tigris
    assert "gpu:gh200:1" in tigris
    assert "220G" in tigris
    sporc = _read("submit_constrained_coarse_to_fine_sporcsubmit_pilot_and_highdata.sh")
    assert "atlas_kd" in sporc
    assert "tier3" in sporc
    assert "gpu:1" in sporc
    assert "300G" in sporc


def test_step10_saves_only_selected_checkpoints_by_default() -> None:
    recon = _read("run_train_constrained_coarse_to_fine_reconstructor.sh")
    tagger = _read("run_train_constrained_coarse_to_fine_tagger.sh")
    assert "CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT:=0" in recon
    assert "CONSTRAINED_C2F_TAGGER_SAVE_LAST_CHECKPOINT:=0" in tagger
    assert "--no-save-last-checkpoint" in recon
    assert "--no-save-last-checkpoint" in tagger


def test_completion_validator_rejects_stale_manifest_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        manifest_payload = {"splits": {"model_train": [], "model_val": []}}
        manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
        manifest_sha = hashlib.sha256(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        run = root / "A0"
        run.mkdir()
        (run / "best_model_val.pt").write_bytes(b"checkpoint")
        checkpoint_sha = hashlib.sha256(b"checkpoint").hexdigest()

        def report(source_manifest_hash: str) -> dict:
            provenance = {
                split: {
                    "source_manifest_hash": source_manifest_hash,
                    "hlt_profile": "fixed_hlt_v2_realistic",
                    "hlt_profile_version": "v1",
                    "hlt_degradation_strength": 2.5,
                    "hlt_content_hash": f"hlt-{split}",
                    "jet_identity_hash": f"ids-{split}",
                }
                for split in ("model_train", "model_val")
            }
            return {
                "ok": True,
                "variant": "A0",
                "checkpoint_sha256": checkpoint_sha,
                "provenance": provenance,
            }

        report_path = run / "run_report.json"
        report_path.write_text(json.dumps(report(manifest_sha)), encoding="utf-8")
        command = (
            sys.executable,
            str(ROOT / "scripts" / "validate_constrained_coarse_to_fine_artifact.py"),
            "--kind",
            "tagger",
            "--path",
            str(run),
            "--run-id",
            "A0",
            "--manifest",
            str(manifest),
        )
        assert subprocess.run(command, capture_output=True, text=True).returncode == 0
        report_path.write_text(json.dumps(report("stale-manifest")), encoding="utf-8")
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode != 0
        assert "source_manifest_hash mismatch" in failed.stderr


def test_completion_validator_accepts_normalized_control_and_seed_variants() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        payload = {"splits": {"model_train": [], "model_val": []}}
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        manifest_sha = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        provenance = {
            split: {
                "source_manifest_hash": manifest_sha,
                "hlt_profile": "fixed_hlt_v2_realistic",
                "hlt_profile_version": "v1",
                "hlt_degradation_strength": 2.5,
                "hlt_content_hash": f"hlt-{split}",
                "jet_identity_hash": f"ids-{split}",
            }
            for split in ("model_train", "model_val")
        }
        cases = (
            (
                "reconstructor",
                "C5-no-slot",
                "C5_uncertainty",
                {"training_config": {"slot_loss_weight": 0.0}},
            ),
            (
                "reconstructor",
                "Cdirect-unconstrained",
                "C5_uncertainty",
                {"model": {"slot_config": {"direct_particle_decoding": True, "constrain_accounting": False}}},
            ),
            ("tagger", "D8-seed2", "D8", {}),
            ("tagger", "D5-B3", "D5-B3", {"run_id": "D5-B3"}),
        )
        for kind, run_id, variant, extra in cases:
            run = root / run_id
            run.mkdir()
            (run / "best_model_val.pt").write_bytes(b"checkpoint")
            checkpoint_sha = hashlib.sha256(b"checkpoint").hexdigest()
            report = {
                "ok": True,
                "variant": variant,
                "checkpoint_sha256": checkpoint_sha,
                "provenance": provenance,
                **extra,
            }
            (run / "run_report.json").write_text(json.dumps(report), encoding="utf-8")
            command = (
                sys.executable,
                str(ROOT / "scripts" / "validate_constrained_coarse_to_fine_artifact.py"),
                "--kind",
                kind,
                "--path",
                str(run),
                "--run-id",
                run_id,
                "--manifest",
                str(manifest),
            )
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr


def test_completion_validator_rejects_wrong_no_slot_attestation_and_modified_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        payload = {"splits": {"model_train": [], "model_val": []}}
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        manifest_sha = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        provenance = {
            split: {
                "source_manifest_hash": manifest_sha,
                "hlt_profile": "fixed_hlt_v2_realistic",
                "hlt_profile_version": "v1",
                "hlt_degradation_strength": 2.5,
                "hlt_content_hash": f"hlt-{split}",
                "jet_identity_hash": f"ids-{split}",
            }
            for split in ("model_train", "model_val")
        }
        run = root / "C5-no-slot"
        run.mkdir()
        checkpoint = run / "best_model_val.pt"
        checkpoint.write_bytes(b"checkpoint")
        report = {
            "ok": True,
            "variant": "C5_uncertainty",
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            "training_config": {"slot_loss_weight": 1.0},
            "provenance": provenance,
        }
        report_path = run / "run_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        command = (
            sys.executable,
            str(ROOT / "scripts" / "validate_constrained_coarse_to_fine_artifact.py"),
            "--kind", "reconstructor", "--path", str(run), "--run-id", "C5-no-slot",
            "--manifest", str(manifest),
        )
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode != 0
        assert "slot_loss_weight == 0" in failed.stderr
        report["training_config"]["slot_loss_weight"] = 0.0
        report_path.write_text(json.dumps(report), encoding="utf-8")
        checkpoint.write_bytes(b"modified checkpoint")
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode != 0
        assert "checkpoint_sha256 mismatch" in failed.stderr


def test_frozen_tagger_sweep_never_unfreezes_reconstructor() -> None:
    schedule = EndToEndScheduleConfig(
        frozen_reconstructor_epochs=1,
        terminal_decoder_epochs=1,
        upper_hierarchy_epochs=1,
    )
    for epoch in (0, 1, 2, 20):
        phase = end_to_end_phase(epoch, D4_UNCERTAINTY_GATED, schedule)
        assert phase.name == "frozen_reconstructor_tagger"
        assert not phase.terminal_decoder_trainable
        assert not phase.upper_hierarchy_trainable

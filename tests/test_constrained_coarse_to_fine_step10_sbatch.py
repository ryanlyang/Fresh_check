from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays
from scripts.validate_constrained_coarse_to_fine_artifact import (
    _TARGET_SHARD_ARRAY_KEYS,
    _json_hash,
    _npz_logical_hash,
    _target_content_hash_payload,
)

from teacher_logit_reco.constrained_coarse_to_fine import (
    D4_UNCERTAINTY_GATED,
    EndToEndScheduleConfig,
    end_to_end_phase,
)


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "sbatch"


def _read(name: str) -> str:
    return (SBATCH / name).read_text(encoding="utf-8")


def _write_active_input_metadata(
    root: Path,
    manifest_sha: str,
    splits: tuple[str, ...] = ("model_train", "model_val"),
) -> tuple[Path, Path, Path, dict[str, dict[str, object]]]:
    hlt_root = root / "hlt"
    offline_root = root / "offline"
    target_root = root / "targets"
    for path in (hlt_root, offline_root, target_root):
        path.mkdir()
    provenance: dict[str, dict[str, object]] = {}
    for split in splits:
        hlt_hash = f"hlt-{split}"
        offline_hash = f"offline-{split}"
        target_hash = f"target-{split}"
        identity_hash = f"ids-{split}"
        hlt = {
            "source_manifest_hash": manifest_sha,
            "hlt_profile": "fixed_hlt_v2_realistic",
            "hlt_profile_version": "v1",
            "hlt_degradation_strength": 2.5,
            "hlt_content_hash": hlt_hash,
            "jet_identity_hash": identity_hash,
        }
        offline = {
            "source_manifest_hash": manifest_sha,
            "offline_content_hash": offline_hash,
            "jet_identity_hash": identity_hash,
        }
        target = {
            **hlt,
            "cache_contract": "constrained_coarse_to_fine_hierarchy_target_cache_v1",
            "offline_content_hash": offline_hash,
            "target_content_hash": target_hash,
        }
        (hlt_root / f"{split}_fixed_hlt_metadata.json").write_text(json.dumps(hlt), encoding="utf-8")
        (offline_root / f"{split}_offline_metadata.json").write_text(json.dumps(offline), encoding="utf-8")
        (target_root / f"{split}_hierarchy_targets_metadata.json").write_text(json.dumps(target), encoding="utf-8")
        provenance[split] = {
            **hlt,
            "offline_content_hash": offline_hash,
            "target_content_hash": target_hash,
        }
    return hlt_root, offline_root, target_root, provenance


def _active_input_args(hlt_root: Path, offline_root: Path, target_root: Path) -> tuple[str, ...]:
    return (
        "--hlt-cache-dir", str(hlt_root),
        "--offline-cache-dir", str(offline_root),
        "--target-cache-dir", str(target_root),
    )


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


def test_target_validator_recomputes_split_and_cache_set_aggregate_hashes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        payload = {"splits": {"model_train": []}}
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        manifest_sha = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        hlt_root, offline_root, target_root, _ = _write_active_input_metadata(
            root, manifest_sha, ("model_train",)
        )
        shard_root = target_root / "model_train_hierarchy_targets"
        shard_root.mkdir()
        arrays = {name: np.zeros((1,), dtype=np.float32) for name in _TARGET_SHARD_ARRAY_KEYS}
        shard_path = shard_root / "shard_00000.npz"
        np.savez_compressed(shard_path, **arrays)
        shard_hash = hash_arrays(arrays)
        metadata = {
            "cache_contract": "constrained_coarse_to_fine_hierarchy_target_cache_v1",
            "builder_version": "test-v1",
            "split": "model_train",
            "n_jets": 1,
            "target_dtype": "float32",
            "source_manifest_hash": manifest_sha,
            "hlt_profile": "fixed_hlt_v2_realistic",
            "hlt_profile_version": "v1",
            "hlt_degradation_strength": 2.5,
            "hlt_content_hash": "hlt-model_train",
            "offline_content_hash": "offline-model_train",
            "jet_identity_hash": "ids-model_train",
            "layout": {"layout_version": "test"},
            "shards": [
                {
                    "filename": shard_path.name,
                    "shard_index": 0,
                    "start": 0,
                    "stop": 1,
                    "n_jets": 1,
                    "content_hash": shard_hash,
                }
            ],
        }
        metadata["target_content_hash"] = _json_hash(_target_content_hash_payload(metadata))
        metadata_path = target_root / "model_train_hierarchy_targets_metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        cache_set = {
            "cache_set_contract": "constrained_coarse_to_fine_hierarchy_target_cache_set_v1",
            "builder_version": "test-v1",
            "source_manifest_hash": manifest_sha,
            "hlt_profile": "fixed_hlt_v2_realistic",
            "hlt_profile_version": "v1",
            "hlt_degradation_strength": 2.5,
            "splits": ["model_train"],
            "split_target_content_hashes": {"model_train": metadata["target_content_hash"]},
        }
        cache_set["cache_set_content_hash"] = _json_hash(cache_set)
        cache_set_path = target_root / "hierarchy_target_cache_manifest.json"
        cache_set_path.write_text(json.dumps(cache_set), encoding="utf-8")
        command = (
            sys.executable,
            str(ROOT / "scripts" / "validate_constrained_coarse_to_fine_artifact.py"),
            "--kind", "target-cache", "--path", str(target_root),
            "--manifest", str(manifest), "--splits", "model_train",
            "--hlt-cache-dir", str(hlt_root), "--offline-cache-dir", str(offline_root),
        )
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        cache_set["cache_set_content_hash"] = "stale-aggregate"
        cache_set_path.write_text(json.dumps(cache_set), encoding="utf-8")
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode != 0
        assert "target cache set aggregate content mismatch" in failed.stderr


def test_prediction_validator_binds_reuse_to_active_input_and_tagger_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        payload = {"splits": {"model_val": []}}
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        manifest_sha = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        hlt_root, offline_root, target_root, provenance = _write_active_input_metadata(
            root, manifest_sha, ("model_train", "model_val")
        )
        tagger_root = root / "taggers"
        active_tagger = tagger_root / "A0"
        active_tagger.mkdir(parents=True)
        checkpoint = active_tagger / "best_model_val.pt"
        checkpoint.write_bytes(b"active checkpoint")
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (active_tagger / "run_report.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "variant": "A0",
                    "checkpoint_sha256": checkpoint_hash,
                    "provenance": provenance,
                }
            ),
            encoding="utf-8",
        )
        prediction_root = root / "predictions" / "A0"
        prediction_root.mkdir(parents=True)
        arrays = {
            "logits": np.zeros((1, 10), dtype=np.float32),
            "probs": np.full((1, 10), 0.1, dtype=np.float32),
            "labels": np.zeros((1,), dtype=np.int64),
            "jet_file_indices": np.zeros((1,), dtype=np.int32),
            "jet_entries": np.zeros((1,), dtype=np.int64),
        }
        np.savez_compressed(prediction_root / "model_val_predictions.npz", **arrays)
        representation = prediction_root / "model_val_representations.npz"
        np.savez_compressed(representation, representation=np.zeros((1, 4), dtype=np.float32))
        representation_hash = hashlib.sha256(representation.read_bytes()).hexdigest()
        metadata = {
            "run_id": "A0",
            "source_manifest_hash": manifest_sha,
            "hlt_profile": "fixed_hlt_v2_realistic",
            "hlt_profile_version": "v1",
            "hlt_degradation_strength": 2.5,
            "deployable_hlt_only": True,
            "hlt_content_hash": "hlt-model_val",
            "jet_identity_hash": "ids-model_val",
            "prediction_content_hash": hash_arrays(arrays),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "fusion_representation_sha256": representation_hash,
        }
        (prediction_root / "model_val_predictions_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (prediction_root / "prediction_run_report.json").write_text(
            json.dumps({"ok": True, "splits": {"model_val": metadata}}), encoding="utf-8"
        )
        command = (
            sys.executable,
            str(ROOT / "scripts" / "validate_constrained_coarse_to_fine_artifact.py"),
            "--kind", "prediction", "--path", str(prediction_root), "--run-id", "A0",
            "--manifest", str(manifest), "--splits", "model_val",
            "--hlt-cache-dir", str(hlt_root), "--offline-cache-dir", str(offline_root),
            "--target-cache-dir", str(target_root),
            "--tagger-root", str(tagger_root),
        )
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        active_hlt_path = hlt_root / "model_val_fixed_hlt_metadata.json"
        active_hlt = json.loads(active_hlt_path.read_text(encoding="utf-8"))
        active_hlt["hlt_content_hash"] = "replacement-hlt-model_val"
        active_hlt_path.write_text(json.dumps(active_hlt), encoding="utf-8")
        active_target_path = target_root / "model_val_hierarchy_targets_metadata.json"
        active_target = json.loads(active_target_path.read_text(encoding="utf-8"))
        active_target["hlt_content_hash"] = "replacement-hlt-model_val"
        active_target_path.write_text(json.dumps(active_target), encoding="utf-8")
        tagger_report_path = active_tagger / "run_report.json"
        tagger_report = json.loads(tagger_report_path.read_text(encoding="utf-8"))
        tagger_report["provenance"]["model_val"]["hlt_content_hash"] = "replacement-hlt-model_val"
        tagger_report_path.write_text(json.dumps(tagger_report), encoding="utf-8")
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode != 0
        assert "active input content mismatch" in failed.stderr


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
    assert '--hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"' in text
    assert '--offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"' in text
    assert '--target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"' in text
    assert '--tagger-root "${CONSTRAINED_C2F_TAGGER_ROOT}"' in text
    assert "Cannot reuse hierarchy targets while active HLT/offline caches are being rebuilt." in text
    assert '[[ -z "${hlt_jid}" && -z "${offline_jid}" && -z "${target_jid}" ]]' in text
    assert "Report-only rerun requires complete predictions" in text
    assert "CONSTRAINED_C2F_SUBMIT_FUSION:=1" in text
    assert "CONSTRAINED_C2F_SUBMIT_REPORT:=1" in text


def test_step10_campaign_is_staged_and_final_test_is_separate() -> None:
    text = _read("submit_constrained_coarse_to_fine_pilot_and_highdata.sh")
    assert "CONSTRAINED_C2F_CAMPAIGN_MODE=pilot" in text
    assert "CONSTRAINED_C2F_CAMPAIGN_MODE=highdata" in text
    assert "--dependency" not in text
    assert "CONSTRAINED_C2F_APPROVE_HIGHDATA" in text
    assert "CONSTRAINED_C2F_PILOT_REPORT_PATH" in text
    assert "pilot final report is not ok" in text
    canonical = _read("submit_constrained_coarse_to_fine_experiment.sh")
    assert '"${CONSTRAINED_C2F_CAMPAIGN_MODE}" == "highdata"' in canonical
    assert "High-data submission requires CONSTRAINED_C2F_APPROVE_HIGHDATA=1" in canonical
    assert "High-data submission requires CONSTRAINED_C2F_PILOT_REPORT_PATH" in canonical
    assert "pilot final report is not ok" in canonical
    standalone = _read("submit_constrained_coarse_to_fine_highdata.sh")
    assert "submit_constrained_coarse_to_fine_experiment.sh" in standalone
    final = _read("submit_constrained_coarse_to_fine_final_claims.sh")
    assert "CONSTRAINED_C2F_APPROVE_FINAL_TEST" in final
    assert "CONSTRAINED_C2F_STAGE_MODE=final_claims" in final
    assert "CONSTRAINED_C2F_CAMPAIGN_MODE=highdata" in final
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


def test_step10_end_to_end_memory_preflight_matches_sequential_split_loading() -> None:
    tagger = _read("run_train_constrained_coarse_to_fine_tagger.sh")
    assert "--loading-mode sequential" in tagger
    source = (ROOT / "teacher_logit_reco" / "constrained_coarse_to_fine" / "end_to_end.py").read_text(
        encoding="utf-8"
    )
    assert '"split_loading": "sequential_reload_per_epoch"' in source
    assert "del train_source" in source
    assert "del val_source" in source


def test_completion_validator_rejects_stale_manifest_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        manifest_payload = {"splits": {"model_train": [], "model_val": []}}
        manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
        manifest_sha = hashlib.sha256(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        hlt_root, offline_root, target_root, active_provenance = _write_active_input_metadata(root, manifest_sha)
        run = root / "A0"
        run.mkdir()
        (run / "best_model_val.pt").write_bytes(b"checkpoint")
        checkpoint_sha = hashlib.sha256(b"checkpoint").hexdigest()

        def report(source_manifest_hash: str) -> dict:
            provenance = {
                split: {**row, "source_manifest_hash": source_manifest_hash}
                for split, row in active_provenance.items()
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
            *_active_input_args(hlt_root, offline_root, target_root),
        )
        assert subprocess.run(command, capture_output=True, text=True).returncode == 0
        hlt_path = hlt_root / "model_train_fixed_hlt_metadata.json"
        target_path = target_root / "model_train_hierarchy_targets_metadata.json"
        hlt_metadata = json.loads(hlt_path.read_text(encoding="utf-8"))
        target_metadata = json.loads(target_path.read_text(encoding="utf-8"))
        hlt_metadata["hlt_content_hash"] = "replacement-hlt-model_train"
        target_metadata["hlt_content_hash"] = "replacement-hlt-model_train"
        hlt_path.write_text(json.dumps(hlt_metadata), encoding="utf-8")
        target_path.write_text(json.dumps(target_metadata), encoding="utf-8")
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode != 0
        assert "active hlt_content_hash mismatch" in failed.stderr
        hlt_metadata["hlt_content_hash"] = "hlt-model_train"
        target_metadata["hlt_content_hash"] = "hlt-model_train"
        hlt_path.write_text(json.dumps(hlt_metadata), encoding="utf-8")
        target_path.write_text(json.dumps(target_metadata), encoding="utf-8")
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
        hlt_root, offline_root, target_root, provenance = _write_active_input_metadata(root, manifest_sha)
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
                *_active_input_args(hlt_root, offline_root, target_root),
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
        hlt_root, offline_root, target_root, provenance = _write_active_input_metadata(root, manifest_sha)
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
            *_active_input_args(hlt_root, offline_root, target_root),
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

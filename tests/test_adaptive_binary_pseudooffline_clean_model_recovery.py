from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "sbatch" / "reset_adaptive_binary_models_recovery_tigris.sh"
).read_text(encoding="utf-8")
PROBE_SUBMITTER = (
    ROOT / "sbatch" / "submit_adaptive_binary_runtime_batch_probes_tigris.sh"
).read_text(encoding="utf-8")


def test_clean_model_recovery_requires_explicit_confirmation() -> None:
    assert "ABPH_CONFIRM_CLEAN_MODEL_RECOVERY" in SCRIPT
    assert "Set ABPH_CONFIRM_CLEAN_MODEL_RECOVERY=1" in SCRIPT


def test_clean_model_recovery_cancels_only_named_abph_jobs() -> None:
    assert "squeue --me" in SCRIPT
    assert "$2 ~ /^abph_/" in SCRIPT
    assert 'scancel "${stale_jobs[@]}"' in SCRIPT


def test_clean_model_recovery_preserves_preparation_and_rebuilds_contracts() -> None:
    assert "runtime_batch_measurements runtime_batch_contracts" in SCRIPT
    assert "submit_adaptive_binary_runtime_batch_probes_tigris.sh" in SCRIPT
    assert "ABPH_STAGE_MODE=models" in SCRIPT
    assert "retained: inputs, baselines, storage acceptance" in SCRIPT


def test_clean_model_recovery_gates_models_on_every_contract() -> None:
    assert """awk -F $'\\t' '$2 == "compile" {print $4}'""" in SCRIPT
    assert 'dependency="afterok:$(IFS=:; echo "${contract_jobs[*]}")"' in SCRIPT
    assert "--dependency=" in SCRIPT


def test_clean_model_recovery_rebuilds_missing_shared_targets_first() -> None:
    assert "model_train_exclusive_kt_adaptive_binary_targets_metadata.json" in SCRIPT
    assert "run_adaptive_binary_targets.sh\" cache" in SCRIPT
    assert "run_adaptive_binary_targets.sh\" preflight" in SCRIPT
    assert 'ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY="afterok:${target_preflight_job}"' in SCRIPT


def test_runtime_probes_accept_the_target_rebuild_dependency() -> None:
    assert "ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY" in PROBE_SUBMITTER
    assert '"--dependency=${ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY}"' in PROBE_SUBMITTER

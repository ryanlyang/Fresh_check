from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.bootstrap_prediction_anchored_bridge_preflight import _baseline_model_size


ROOT = Path(__file__).resolve().parents[1]


def test_clean_start_bootstrap_cli_imports_without_campaign_outputs():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/bootstrap_prediction_anchored_bridge_preflight.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--baseline-checkpoint" in completed.stdout
    assert "--artifact-root" in completed.stdout


def test_bootstrap_derives_and_requires_the_baseline_model_size(tmp_path):
    checkpoint = tmp_path / "baseline.pt"
    torch.save({"model_state_dict": {}, "model_config": {"model_size": "tiny"}}, checkpoint)
    assert _baseline_model_size(checkpoint) == "tiny"
    torch.save({"model_state_dict": {}}, tmp_path / "unknown.pt")
    with pytest.raises(ValueError, match="model_size"):
        _baseline_model_size(tmp_path / "unknown.pt")


def test_full_bootstrap_submitter_queues_afterok_and_reuses_complete_hlt():
    submitter = (ROOT / "sbatch/submit_prediction_anchored_bridge_full_bootstrap.sh").read_text()
    finalizer = (ROOT / "sbatch/run_finalize_prediction_anchored_bridge_submission.sh").read_text()
    missing = (ROOT / "sbatch/run_prediction_anchored_missing_offline_split.sh").read_text()
    assert "afterok:${offline_job}" in submitter
    assert "run_prediction_anchored_missing_offline_split.sh" in submitter
    assert "for split in model_train model_val stack_train stack_val final_test" in submitter
    assert "refusing a guessed rebuild" in submitter
    assert "--account=reu-aisocial" in submitter
    assert "PREDICTION_ANCHORED_EXECUTE=1" in finalizer
    assert "submit_prediction_anchored_bridge_pilot.sh" in finalizer
    assert "stack_train_offline.npz" in missing
    assert "--splits stack_train" in missing
    assert "PYTHONNOUSERSITE=1" in submitter + finalizer + missing
    assert "SKIP_CONDA=1" in finalizer
    assert "SKIP_CONDA=1" in missing
    assert '/envs/${CONDA_ENV}/bin/python' in finalizer
    assert '/envs/${CONDA_ENV}/bin/python' in missing


def test_finalizer_binds_all_four_immutable_submission_controls():
    finalizer = (ROOT / "sbatch/run_finalize_prediction_anchored_bridge_submission.sh").read_text()
    for name in (
        "PREDICTION_ANCHORED_GRAPH",
        "PAB_REGISTRY",
        "PAB_RESERVATIONS",
        "PAB_EXECUTION_SPEC",
    ):
        assert f"export {name}=" in finalizer
    assert "--budget-gib 5" in finalizer

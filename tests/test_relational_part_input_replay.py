"""Bounded tests for cross-platform diagnosis, not scientific parity evidence."""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/replay_relational_part_cross_platform_inputs.py"
SPEC = importlib.util.spec_from_file_location("input_replay_under_test", SCRIPT)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
LAUNCHER = REPO / "sbatch/submit_relational_part_input_replay.sh"


def snapshot():
    return {"features": np.zeros((64, 17, 128), np.float32),
            "points": np.zeros((64, 2, 128), np.float32),
            "lorentz_vectors": np.zeros((64, 4, 128), np.float32),
            "mask": np.ones((64, 1, 128), bool)}


def test_snapshot_and_nonempty_exact_masks():
    values = snapshot()
    R.validate_snapshot(values)
    values["mask"][0] = False
    with pytest.raises(ValueError, match="empty jet"):
        R.validate_snapshot(values)


@pytest.mark.parametrize("field,kind", [("features", "shape"), ("mask", "dtype"),
                                        ("lorentz_vectors", "nonfinite"), ("points", "dtype")])
def test_snapshot_rejects_invalid_inputs(field, kind):
    values = snapshot()
    if kind == "shape":
        values[field] = values[field][:8]
    elif kind == "dtype":
        values[field] = values[field].astype(np.float64)
    else:
        values[field][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid input snapshot"):
        R.validate_snapshot(values)


def test_replay_isolates_input_and_pair_changes_and_resets_state():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.count = 99
            self.pair_calls = 0

        def explicit_standard_four(self, vectors, mask):
            self.pair_calls += 1
            return vectors.unsqueeze(-1) + vectors.unsqueeze(-2)

        def mod(self, features, *, v, mask, uu):
            assert not torch.is_grad_enabled() and not self.training
            value = (features.sum((1, 2)) + uu.sum((1, 2, 3)) + self.count)[:, None].expand(-1, 10).clone()
            self.count += 1
            features.fill_(999)
            v.fill_(999)
            mask.fill_(False)
            uu.fill_(-999)
            return value

    values = {"features": np.ones((2, 17, 3), np.float32),
              "points": np.zeros((2, 2, 3), np.float32),
              "lorentz_vectors": np.ones((2, 4, 3), np.float32),
              "mask": np.ones((2, 1, 3), bool)}
    before = copy.deepcopy(values)
    model = Model().eval()
    resets = []

    def reset(models):
        assert models == [model]
        model.count = 0
        resets.append(True)

    first, pairs = R.replay(model, values, torch.device("cpu"), reset)
    second, fixed = R.replay(model, values, torch.device("cpu"), reset, pairs=pairs)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(pairs, np.full((2, 4, 3, 3), 2, np.float32))
    np.testing.assert_array_equal(fixed, pairs)
    np.testing.assert_array_equal(first, np.full((2, 10), 123, np.float32))
    changed, _ = R.replay(model, values, torch.device("cpu"), reset, pairs=pairs + 1)
    np.testing.assert_array_equal(changed - first, np.full((2, 10), 36, np.float32))
    assert model.pair_calls == 1 and len(resets) == 3
    for key in before:
        np.testing.assert_array_equal(values[key], before[key])


def binding():
    return dict(contract=R.VERSION, phase="capture", architecture="aarch64", device="cpu",
                events=64, campaign_sha256="campaign", diagnostic_sha256="diagnostic",
                authorizes_inference=False)


@pytest.mark.parametrize("field,value", [("phase", "replay"), ("architecture", "x86_64"),
    ("device", "cuda"), ("events", 640), ("campaign_sha256", "another"),
    ("diagnostic_sha256", "another"), ("contract", "older"), ("authorizes_inference", True)])
def test_capture_must_match_the_exact_diagnostic(field, value):
    capture = binding()
    R.require_capture_binding(capture, {"content_hash": "campaign"}, {"content_hash": "diagnostic"})
    capture[field] = value
    with pytest.raises(ValueError, match="another diagnostic"):
        R.require_capture_binding(capture, {"content_hash": "campaign"}, {"content_hash": "diagnostic"})


def test_artifact_hash_is_checked_before_reading(tmp_path):
    from scripts import evaluate_relational_part_offline_full_test as F
    path = tmp_path / "replay_arrays.npz"
    np.savez(path, example=np.ones(2))
    report = {"artifacts": {path.name: F.digest(path)}}
    values = R.read_arrays(F, tmp_path, report, path.name)
    np.testing.assert_array_equal(values["example"], np.ones(2))
    np.savez(path, example=np.zeros(2))
    with pytest.raises(ValueError, match="bytes changed"):
        R.read_arrays(F, tmp_path, report, path.name)


def test_helpers_load_without_a_campaign_and_cli_help_works():
    assert R.load_probe_helpers().BATCH == 64
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--capture" in result.stdout and "--diagnostic" in result.stdout


@pytest.fixture
def shell():
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash unavailable")
    return str(bash)


def shell_path(path):
    value = str(path).replace("\\", "/")
    return "/" + value[0].lower() + value[2:] if len(value) > 1 and value[1] == ":" else value


@pytest.mark.parametrize("phase", ["capture", "replay"])
def test_real_shell_dry_run_resources_and_no_mutations(tmp_path, shell, phase):
    root = tmp_path / "campaign"
    (root / "source").mkdir(parents=True)
    (root / "sporc_campaign.json").write_text("{}")
    diag = tmp_path / "previous"
    diag.mkdir()
    for name in ("diagnostic_report.json", "first_batch_inputs_and_pairs.npz", "diagnostic_logits.npz"):
        (diag / name).touch()
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "replay_report.json").touch()
    (capture / "replay_arrays.npz").touch()
    args = [shell, shell_path(LAUNCHER), "--phase", phase, "--campaign", shell_path(root),
            "--diagnostic", shell_path(diag), "--dry-run"]
    if phase == "replay":
        args += ["--capture", shell_path(capture)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--cpus-per-task=2" in result.stdout and "--mem=16G" in result.stdout
    if phase == "capture":
        assert "--partition=tigris" in result.stdout and "--gres" not in result.stdout
    else:
        assert "--partition=tier3" in result.stdout and "--gres=gpu:a100:1" in result.stdout
    assert not (root / "diagnostics").exists()


def test_scripts_do_not_write_production_receipts_or_submit_full_graph():
    source, launcher = SCRIPT.read_text(), LAUNCHER.read_text()
    assert '"authorizes_inference": False' in source
    assert '"production_settings_changed": False' in source
    assert '"used_for_model_selection": False' in source
    assert "portability_validation.json" not in source
    assert "scancel" not in launcher and "--dependency" not in launcher
    assert "compgen -A variable SBATCH_" in launcher

"""Fake Slurm only; never contacts a cluster."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from tests.test_part_fullscale_launcher import BASH, shell_path

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not BASH, reason="Bash unavailable")


@pytest.fixture
def launcher(tmp_path):
    source, output, fakebin = tmp_path/"source repo", tmp_path/"campaign", tmp_path/"bin"
    for p in (source/"sbatch", source/"scripts", source/"part_selected_screen", fakebin): p.mkdir(parents=True)
    for name in ("submit_part_selected_screen_sporc.sh", "run_part_selected_screen_sporc.sh"):
        shutil.copyfile(REPO/"sbatch"/name, source/"sbatch"/name)
    for name in ("model.py", "reporting.py"):
        (source/"part_selected_screen"/name).write_text("# fixture\n")
    (source/"scripts/run_part_selected_screen.py").write_text("# fixture\n")
    donor = tmp_path/"donor"
    donor.mkdir(); (donor/"campaign.json").write_text("{}")
    layerwise = tmp_path/"layerwise donor"
    layerwise.mkdir(); (layerwise/"experiment_spec.json").write_text("{}")
    calls = tmp_path/"calls.txt"
    stubs = {
        "flock": "exit 0\n",
        "uname": 'echo "${TEST_ARCH:-x86_64}"\n',
        "squeue": 'printf "%s\\n" "${TEST_LIVE:-}"\n',
        "scontrol": '[[ "${TEST_PARTITION_BAD:-}" != 1 ]] || exit 1\necho PartitionName=tier3\n',
        "sbatch": 'printf "%s|%s\\n" "$PART_SELECTED_PHASE" "$*" >> "$TEST_CALLS"\necho "$((750000 + $(wc -l < "$TEST_CALLS")))"\n',
        "python": '''if [[ "$*" == *run_part_selected_screen.py* ]]; then
  printf 'WORKER %s\\n' "$*"
else
  exec "$TEST_PYTHON" "$@"
fi
''',
    }
    for name, body in stubs.items():
        path = fakebin/name
        path.write_text("#!/usr/bin/env bash\n"+body, newline="\n"); path.chmod(0o755)
    result = subprocess.run([BASH, "-c", 'git init -q "$1" && git -C "$1" add . && git -C "$1" -c user.name=Test -c user.email=test@example.invalid -c commit.gpgsign=false commit -qm fixture',
        "init", shell_path(source)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    def run(*args, env=None, worker=False):
        extra = dict(os.environ, TEST_CALLS=shell_path(calls), TEST_PYTHON=shell_path(sys.executable))
        extra.update(env or {})
        script = output/"source/sbatch/run_part_selected_screen_sporc.sh" if worker else source/"sbatch/submit_part_selected_screen_sporc.sh"
        defaults = [] if worker else ["--output", shell_path(output), "--python", shell_path(fakebin/"python"), "--donor", shell_path(donor), "--layerwise-donor", shell_path(layerwise)]
        return subprocess.run([BASH, "--noprofile", "--norc", "-c", 'export PATH="$1:$PATH"; shift; exec bash "$@"', "test",
            shell_path(fakebin), shell_path(script), *defaults, *args], env=extra, capture_output=True, text=True, timeout=120)
    return run, source, output, calls



def test_dry_run_is_non_mutating(launcher):
    run, _, output, calls = launcher
    result = run("--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--array=0-3%2" in result.stdout and "--partition=tier3" in result.stdout
    assert "prepare 5h -> benchmark[4] 5h -> gate 2h" in result.stdout
    assert not output.exists() and not calls.exists()


def test_full_dependency_graph_four_new_models_only(launcher):
    run, _, output, calls = launcher
    result = run("--concurrency", "4")
    assert result.returncode == 0, result.stderr
    rows = calls.read_text().splitlines()
    assert [r.split("|")[0] for r in rows] == ["prepare", "benchmark", "gate", "train", "lock", "evaluate", "report"]
    assert "--dependency" not in rows[0]
    for i in range(1, 7):
        assert f"--dependency=afterok:{750000+i}" in rows[i]
        assert "--kill-on-invalid-dep=yes" in rows[i]
    for i in (1, 3, 5):
        assert "--array=0-3%4" in rows[i] and "--gres=gpu:a100:1" in rows[i]
    for i in (0, 2, 4, 6):
        assert "--array" not in rows[i] and "--gres" not in rows[i]
    assert "--time=5:00:00" in rows[0] and "--time=5:00:00" in rows[1]
    assert "--time=96:00:00" in rows[3] and "--time=12:00:00" in rows[5]
    assert all("--partition=tier3" in r for r in rows)
    assert (output/"launch.json").is_file()
    import json
    launch = json.loads((output/"launch.json").read_text())
    assert len(launch["controls"]) == 4
    assert launch["train_hours"] == "96"
    assert (output/"source/part_selected_screen/model.py").read_text() == "# fixture\n"


def test_frozen_worker_and_invalid_task(launcher):
    run, source, output, calls = launcher
    assert run().returncode == 0
    (source/"part_selected_screen/model.py").write_text("# later pull\n")
    env = dict(PART_SELECTED_OUTPUT=shell_path(output), PART_SELECTED_PHASE="train", SLURM_ARRAY_TASK_ID="3")
    result = run(worker=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "--task-index 3" in result.stdout
    assert run(worker=True, env=dict(env, SLURM_ARRAY_TASK_ID="4")).returncode != 0
    assert run(worker=True, env=dict(env, TEST_ARCH="aarch64")).returncode != 0
    (output/"source/part_selected_screen/model.py").write_text("# tampered\n")
    assert run(worker=True, env=env).returncode != 0


def test_no_duplicate_jobs_or_changed_frozen_settings(launcher):
    run, _, _, calls = launcher
    assert run().returncode == 0
    assert run().returncode != 0
    assert run("--resume", env={"TEST_LIVE": "750004_[0-3%2]"}).returncode != 0
    assert run("--resume", "--mem", "128G").returncode != 0
    assert run("--resume", "--train-hours", "72").returncode != 0
    assert len(calls.read_text().splitlines()) == 7
    result = run("--mode", "train", "--resume")
    assert result.returncode == 0, result.stderr
    assert [r.split("|")[0] for r in calls.read_text().splitlines()[7:]] == ["gate", "train", "lock", "evaluate", "report"]


def test_status_before_preparation_and_report_retry(launcher):
    run, _, _, calls = launcher
    assert run().returncode == 0
    result = run("--mode", "status")
    assert result.returncode == 0 and "possibly still queued" in result.stdout
    assert len(calls.read_text().splitlines()) == 7
    assert run("--mode", "report", "--resume").returncode == 0
    assert len(calls.read_text().splitlines()) == 8


def test_dirty_source_and_wrong_partition_refused(launcher):
    run, source, _, calls = launcher
    (source/"part_selected_screen/model.py").write_text("# dirty\n")
    assert run().returncode != 0
    assert not calls.exists()
    (source/"part_selected_screen/model.py").write_text("# fixture\n")
    assert run(env={"TEST_PARTITION_BAD": "1"}).returncode != 0
    assert not calls.exists()

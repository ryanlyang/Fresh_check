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
    for p in (source/"sbatch", source/"scripts", source/"part_layerwise_screen", fakebin): p.mkdir(parents=True)
    for name in ("submit_part_layerwise_screen_sporc.sh", "run_part_layerwise_screen_sporc.sh"):
        shutil.copyfile(REPO/"sbatch"/name, source/"sbatch"/name)
    for name in ("model.py", "reporting.py"):
        (source/"part_layerwise_screen"/name).write_text("# fixture\n")
    (source/"scripts/run_part_layerwise_screen.py").write_text("# fixture\n")
    donor = tmp_path/"donor"
    donor.mkdir(); (donor/"campaign.json").write_text("{}")
    calls = tmp_path/"calls.txt"
    stubs = {
        "flock": "exit 0\n",
        "uname": 'echo "${TEST_ARCH:-x86_64}"\n',
        "squeue": 'printf "%s\\n" "${TEST_LIVE:-}"\n',
        "scontrol": '[[ "${TEST_PARTITION_BAD:-}" != 1 ]] || exit 1\necho PartitionName=tier3\n',
        "sbatch": 'printf "%s|%s\\n" "$PART_LAYERWISE_PHASE" "$*" >> "$TEST_CALLS"\necho "$((750000 + $(wc -l < "$TEST_CALLS")))"\n',
        "python": '''if [[ "$*" == *run_part_layerwise_screen.py* ]]; then
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
        script = output/"source/sbatch/run_part_layerwise_screen_sporc.sh" if worker else source/"sbatch/submit_part_layerwise_screen_sporc.sh"
        defaults = [] if worker else ["--output", shell_path(output), "--python", shell_path(fakebin/"python"), "--donor", shell_path(donor)]
        return subprocess.run([BASH, "--noprofile", "--norc", "-c", 'export PATH="$1:$PATH"; shift; exec bash "$@"', "test",
            shell_path(fakebin), shell_path(script), *defaults, *args], env=extra, capture_output=True, text=True, timeout=40)
    return run, source, output, calls


def test_dry_run_no_files_or_jobs(launcher):
    run, _, output, calls = launcher
    result = run("--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--array=0-1%2" in result.stdout and "--partition=tier3" in result.stdout
    assert "No separate preflight" in result.stdout
    assert not output.exists() and not calls.exists()


def test_one_command_queues_exactly_two_train_and_eval_tasks(launcher):
    run, _, output, calls = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    rows = calls.read_text().splitlines()
    assert [r.split("|")[0] for r in rows] == ["train", "lock", "evaluate", "report"]
    assert "--dependency" not in rows[0]
    assert "--time=48:00:00" in rows[0] and "--array=0-1%2" in rows[0]
    assert "--dependency=afterok:750001" in rows[1]
    assert "--dependency=afterok:750002" in rows[2] and "--array=0-1%2" in rows[2]
    assert "--dependency=afterok:750003" in rows[3]
    assert "--time=4:00:00" in rows[2]
    assert all("--gres=gpu:a100:1" in rows[i] for i in (0, 2))
    assert all("--gres" not in rows[i] and "--time=2:00:00" in rows[i] for i in (1, 3))
    assert all("--partition=tier3" in row and "--account=reu-aisocial" in row for row in rows)
    assert all("PART_RANGER" not in r and "EV_RANGER" not in r for r in rows)
    assert (output/"source/part_layerwise_screen/model.py").read_text() == "# fixture\n"


def test_frozen_worker_ignores_later_pull_and_rejects_bad_task(launcher):
    run, source, output, calls = launcher
    assert run().returncode == 0
    (source/"part_layerwise_screen/model.py").write_text("# later changes\n")
    env = dict(PART_LAYERWISE_OUTPUT=shell_path(output), PART_LAYERWISE_PHASE="train", SLURM_ARRAY_TASK_ID="1")
    result = run(worker=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "--phase train" in result.stdout and "--task-index 1" in result.stdout
    assert run(worker=True, env=dict(env, SLURM_ARRAY_TASK_ID="2")).returncode != 0
    assert run(worker=True, env=dict(env, TEST_ARCH="aarch64")).returncode != 0
    (output/"source/part_layerwise_screen/model.py").write_text("# tampered\n")
    assert run(worker=True, env=env).returncode != 0
    assert len(calls.read_text().splitlines()) == 4


def test_resume_blocks_live_duplicates_and_changes_to_resources(launcher):
    run, _, _, calls = launcher
    assert run().returncode == 0
    assert run().returncode != 0
    assert run("--resume", env={"TEST_LIVE": "750001_[0-1%2]"}).returncode != 0
    assert run("--resume", "--mem", "128G").returncode != 0
    assert len(calls.read_text().splitlines()) == 4
    result = run("--resume", "--train-hours", "72")
    assert result.returncode == 0, result.stderr
    assert len(calls.read_text().splitlines()) == 8
    assert "--time=72:00:00" in calls.read_text().splitlines()[4]


def test_retry_report_only_and_status_before_start(launcher):
    run, _, _, calls = launcher
    assert run().returncode == 0
    result = run("--mode", "status")
    assert result.returncode == 0 and "possibly still queued" in result.stdout
    assert len(calls.read_text().splitlines()) == 4
    result = run("--mode", "report", "--resume")
    assert result.returncode == 0, result.stderr
    assert len(calls.read_text().splitlines()) == 5
    assert calls.read_text().splitlines()[-1].startswith("report|")


def test_dirty_sources_and_unavailable_partition_refused(launcher):
    run, source, _, calls = launcher
    (source/"part_layerwise_screen/model.py").write_text("# uncommitted\n")
    assert run().returncode != 0
    assert not calls.exists()
    (source/"part_layerwise_screen/model.py").write_text("# fixture\n")
    assert run(env={"TEST_PARTITION_BAD": "1"}).returncode != 0
    assert not calls.exists()

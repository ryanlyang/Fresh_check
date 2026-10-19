"""Fake Slurm only; no cluster access or real submissions."""
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
    repo, campaign, fakebin = tmp_path / "repo", tmp_path / "campaign", tmp_path / "bin"
    for path in (repo / "scripts", repo / "sbatch", campaign / "source", fakebin):
        path.mkdir(parents=True)
    for name in ("submit_part_ev_message_parity.sh", "run_part_ev_message_parity.sh"):
        shutil.copyfile(REPO / "sbatch" / name, repo / "sbatch" / name)
    shutil.copyfile(REPO / "scripts/diagnose_part_ev_message_parity.py", repo / "scripts/diagnose_part_ev_message_parity.py")
    calls = tmp_path / "calls"
    stubs = {
        "uname": 'echo "${TEST_ARCH:-x86_64}"\n',
        "flock": "exit 0\n",
        "scontrol": '[[ "${TEST_PARTITION_BAD:-}" != 1 ]] || exit 1\necho tier3\n',
        "squeue": '[[ "${TEST_QUEUE_FAIL:-}" != 1 ]] || exit 1\nprintf "%s\\n" "${TEST_LIVE:-}"\n',
        "sbatch": 'printf "%s\\n" "$*" >> "$TEST_CALLS"\necho "$((760000 + $(wc -l < "$TEST_CALLS")))"\n',
        "python": '''if [[ "$*" == *diagnose_part_ev_message_parity.py* ]]; then
  [[ "${TEST_PREPARE_FAIL:-}" != 1 ]] || exit 1
  if [[ "$*" == *"--phase prepare"* ]]; then
    for ((i=1;i<=$#;i++)); do
      if [[ "${!i}" == --output ]]; then j=$((i+1)); diagnostic="${!j}"; fi
      if [[ "${!i}" == --campaign ]]; then j=$((i+1)); campaign="${!j}"; fi
    done
    printf '{"campaign":"%s"}\\n' "$campaign" > "$diagnostic/diagnostic_plan.json"
  else
    printf 'FROZEN %s\\n' "$*"
  fi
else
  exec "$TEST_PYTHON" "$@"
fi
''',
    }
    for name, body in stubs.items():
        path = fakebin / name
        path.write_text("#!/usr/bin/env bash\n" + body, newline="\n")
        path.chmod(0o755)
    (campaign / "launch.env").write_text(
        f'MESSAGE_PYTHON="{shell_path(fakebin / "python")}"\n'
        "MESSAGE_GPU=a100\nMESSAGE_CPUS=8\nMESSAGE_MEMORY=64G\n"
        "MESSAGE_ACCOUNT=reu-aisocial\nMESSAGE_PARTITION=tier3\n", newline="\n")
    (campaign / "launch.json").write_text("{}\n")
    (campaign / "campaign.json").write_text("{}\n")
    (campaign / "source/sentinel").write_text("frozen campaign\n")
    (campaign / "submitted_job_ids.tsv").write_text("benchmark\t21919334\n", newline="\n")
    result = subprocess.run([BASH, "-c",
        'git init -q "$1" && git -C "$1" add . && git -C "$1" -c user.name=Test -c user.email=test@example.invalid '
        '-c commit.gpgsign=false commit -qm fixture && cd "$2" && sha256sum launch.env launch.json > launch.sha256 '
        '&& cd source && sha256sum sentinel > ../source.sha256',
        "init", shell_path(repo), shell_path(campaign)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr

    def run(*args, env=None, worker=None):
        extra = dict(os.environ, TEST_CALLS=shell_path(calls), TEST_PYTHON=shell_path(sys.executable), **(env or {}))
        script = worker / "run_part_ev_message_parity.sh" if worker else repo / "sbatch/submit_part_ev_message_parity.sh"
        options = [shell_path(worker)] if worker else ["--campaign", shell_path(campaign), *args]
        return subprocess.run([BASH, "--noprofile", "--norc", "-c",
            'export PATH="$1:$PATH"; shift; exec bash "$@"', "test", shell_path(fakebin), shell_path(script), *options],
            env=extra, capture_output=True, text=True, timeout=40)
    return run, repo, campaign, calls


def test_dry_run_is_read_only_and_single_one_hour_gpu(launcher):
    run, _, campaign, calls = launcher
    result = run("--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--time=01:00:00" in result.stdout and "--partition=tier3" in result.stdout
    assert "--array" not in result.stdout and not calls.exists()
    assert not (campaign / "diagnostics").exists()


def test_submission_can_coexist_with_readonly_original_benchmarks(launcher):
    run, repo, campaign, calls = launcher
    result = run(env={"TEST_LIVE": "21919334_[2-4%2]", "SBATCH_TIME": "9-00:00:00"})
    assert result.returncode == 0, result.stderr
    rows = calls.read_text().splitlines()
    assert len(rows) == 1 and "--time=01:00:00" in rows[0] and "--gres=gpu:a100:1" in rows[0]
    assert "--array" not in rows[0] and "--dependency" not in rows[0]
    assert (campaign / "submitted_job_ids.tsv").read_text() == "benchmark\t21919334\n"
    output = next((campaign / "diagnostics").glob("message_parity.*"))
    (repo / "scripts/diagnose_part_ev_message_parity.py").write_text("# changed live code\n")
    result = run(worker=output)
    assert result.returncode == 0, result.stderr
    assert "--phase run" in result.stdout
    assert run(worker=output, env={"TEST_ARCH": "aarch64"}).returncode != 0
    (output / "diagnose_part_ev_message_parity.py").write_text("# changed frozen diagnostic\n")
    assert run(worker=output).returncode != 0


@pytest.mark.parametrize("env", [{"TEST_ARCH": "aarch64"}, {"TEST_PREPARE_FAIL": "1"}, {"TEST_PARTITION_BAD": "1"}])
def test_failed_setup_never_submits(launcher, env):
    run, _, _, calls = launcher
    assert run(env=env).returncode != 0
    assert not calls.exists()


def test_duplicate_task_diagnostic_is_blocked_but_other_task_allowed(launcher):
    run, _, _, calls = launcher
    assert run().returncode == 0
    assert run(env={"TEST_LIVE": "760001"}).returncode != 0
    assert len(calls.read_text().splitlines()) == 1
    assert run("--task-index", "1", env={"TEST_LIVE": "760001"}).returncode == 0
    assert len(calls.read_text().splitlines()) == 2
    assert run(env={"TEST_QUEUE_FAIL": "1"}).returncode != 0


def test_frozen_campaign_and_resource_tampering_blocked(launcher):
    run, _, campaign, calls = launcher
    (campaign / "launch.env").write_text("MESSAGE_PARTITION=debug\n")
    assert run().returncode != 0 and not calls.exists()


def test_dirty_diagnostic_and_changed_original_source_blocked(launcher):
    run, repo, campaign, calls = launcher
    (campaign / "source/sentinel").write_text("modified\n")
    assert run().returncode != 0 and not calls.exists()
    (repo / "scripts/diagnose_part_ev_message_parity.py").write_text("# dirty\n")
    assert run().returncode != 0 and not calls.exists()


def test_invalid_index_is_rejected(launcher):
    run, _, _, calls = launcher
    assert run("--task-index", "5").returncode != 0 and not calls.exists()

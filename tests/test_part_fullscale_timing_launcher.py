"""Read-only campaign and isolated timing launcher tests using fake Slurm."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
BASH = next((str(p) for p in (Path("C:/Program Files/Git/bin/bash.exe"),)
             if p.is_file()), None) if os.name == "nt" else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH, reason="Local Bash unavailable")


def shell_path(path):
    path = Path(path).resolve()
    return "/" + path.drive[0].lower() + path.as_posix()[2:] if os.name == "nt" else str(path)


def snapshot(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*")
            if p.is_file() and "diagnostics" not in p.relative_to(path).parts}


@pytest.fixture
def timing_launcher(tmp_path):
    repo, campaign, fakebin = tmp_path / "source repo", tmp_path / "campaign", tmp_path / "bin"
    for path in (repo / "sbatch", repo / "scripts", campaign / "source/scripts", fakebin):
        path.mkdir(parents=True)
    for name in ("submit_part_fullscale_timing.sh", "run_part_fullscale_timing.sh"):
        shutil.copyfile(REPO / "sbatch" / name, repo / "sbatch" / name)
    (repo / "scripts/diagnose_part_fullscale_timing.py").write_text("# standalone diagnostic snapshot\n")
    original = campaign / "source/scripts/run_part_fullscale.py"
    original.write_text("# immutable original training recipe\n")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    (campaign / "source.sha256").write_text(f"{digest}  ./scripts/run_part_fullscale.py\n", newline="\n")
    (campaign / "campaign.json").write_text("{}\n")
    (campaign / "submitted_job_ids.tsv").write_text("train\t283716\t2026-10-13\n")
    (campaign / "benchmark_1gpu.json").write_text('{"preserve": true}\n')
    (campaign / "checkpoint.pt").write_bytes(b"preserve-production-checkpoint")
    (campaign / "launch.env").write_text(
        f"PART_FULLSCALE_PYTHON='{shell_path(fakebin / 'python')}'\n"
        "PART_FULLSCALE_DATA_ROOT=/data/JetClass\n", newline="\n")
    calls = tmp_path / "calls.txt"
    stubs = {
        "sbatch": (
            'printf "ENV_DEP=%s|" "${SBATCH_DEPENDENCY:-}" >> "$TEST_CALLS"\n'
            'printf "%s " "$@" >> "$TEST_CALLS"\nprintf "\\n" >> "$TEST_CALLS"\n'
            'if [[ "${TEST_SBATCH_BAD:-}" == 1 ]]; then echo invalid; else echo "700001;cluster"; fi\n'
        ),
        "python": (
            'printf "PYTHON %s\\n" "$*"\n'
            'printf "RANK=%s LOCAL_RANK=%s WORLD_SIZE=%s LOCAL_WORLD_SIZE=%s\\n" "$RANK" "$LOCAL_RANK" "$WORLD_SIZE" "$LOCAL_WORLD_SIZE"\n'
            'printf "PYTHONPATH=%s PYTHONNOUSERSITE=%s PYTHONDONTWRITEBYTECODE=%s\\n" "$PYTHONPATH" "$PYTHONNOUSERSITE" "$PYTHONDONTWRITEBYTECODE"\n'
        ),
        "uname": 'echo "${TEST_ARCH:-aarch64}"\n',
        "srun": (
            'printf "SRUN %s\\n" "$*"\n'
            'printf "SRUN_ENV_NTASKS=%s\\n" "${SRUN_NTASKS:-}"\n'
            'while [[ "$1" == --* ]]; do shift; done\nexec "$@"\n'
        ),
        "squeue": 'echo "Unexpected production queue query" >&2; exit 90\n',
        "scancel": 'echo "Forbidden cancellation" >&2; exit 91\n',
        "scontrol": 'echo "Forbidden production mutation" >&2; exit 92\n',
    }
    for name, body in stubs.items():
        path = fakebin / name
        path.write_text("#!/usr/bin/env bash\n" + body, newline="\n")
        path.chmod(0o755)

    def run(*args, extra_env=None, worker=None):
        env = dict(os.environ, TEST_CALLS=shell_path(calls))
        env.update(extra_env or {})
        script = worker / "source/sbatch/run_part_fullscale_timing.sh" if worker else repo / "sbatch/submit_part_fullscale_timing.sh"
        defaults = [] if worker else ["--campaign", shell_path(campaign)]
        if worker:
            env["PART_FULLSCALE_TIMING_OUTPUT"] = shell_path(worker)
        return subprocess.run([BASH, "--noprofile", "--norc", "-c",
                               'export PATH="$1:$PATH"; shift; exec bash "$@"', "test",
                               shell_path(fakebin), shell_path(script), *defaults, *args],
                              env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)

    return run, repo, campaign, calls


def submit(timing_launcher):
    run, _, campaign, _ = timing_launcher
    result = run()
    assert result.returncode == 0, result.stderr
    return next((campaign / "diagnostics").glob("timing.*"))


def test_dry_run_is_read_only_and_fixed_at_two_hours(timing_launcher):
    run, _, campaign, calls = timing_launcher
    before = snapshot(campaign)
    result = run("--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--gpus-per-node=1" in result.stdout
    assert "--cpus-per-task=8 --mem=128G --time=02:00:00 --signal=USR1@300" in result.stdout
    assert "train-steps=10000 validation-steps=2500 budget-minutes=100" in result.stdout
    assert snapshot(campaign) == before
    assert not (campaign / "diagnostics").exists()
    assert not calls.exists()


@pytest.mark.parametrize("args", [
    ("--train-steps", "0"), ("--validation-steps", "-1"),
    ("--train-steps", "20001"), ("--validation-steps", "2501"),
    ("--budget-minutes", "101"), ("--budget-minutes", "1.5"),
    ("--train-steps", "100000000000000000000"), ("--train-steps", "01"),
    ("--dependency", "283716"), ("--time", "3-00:00:00"),
    ("--resume",), ("--gpus", "2"), ("--account", "bad;name"),
])
def test_invalid_options_do_not_create_or_submit(timing_launcher, args):
    run, _, campaign, calls = timing_launcher
    before = snapshot(campaign)
    result = run(*args)
    assert result.returncode != 0
    assert snapshot(campaign) == before
    assert not (campaign / "diagnostics").exists() and not calls.exists()


def test_submission_is_separate_snapshot_and_does_not_change_campaign(timing_launcher):
    run, repo, campaign, calls = timing_launcher
    before = snapshot(campaign)
    result = run(extra_env={"SBATCH_DEPENDENCY": "afterok:283716", "SBATCH_ARRAY_INX": "0-5%2",
                            "SBATCH_TIMELIMIT": "3-00:00:00"})
    assert result.returncode == 0, result.stderr
    diagnostic = next((campaign / "diagnostics").glob("timing.*"))
    row, = calls.read_text().splitlines()
    assert row.startswith("ENV_DEP=|")
    assert "--job-name=parte_timing" in row
    assert "--nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=1" in row
    assert "--time=02:00:00 --signal=USR1@300" in row
    assert "--dependency" not in row and "--array" not in row
    assert "283716" not in row and "3-00:00:00" not in row
    assert snapshot(campaign) == before
    assert (diagnostic / "submitted_job_id.txt").read_text().strip() == "700001"
    assert "report.json" in result.stdout and "report.md" in result.stdout
    assert "cat " in result.stdout and "squeue -j 700001" in result.stdout
    assert "launch.env" in (diagnostic / "diagnostic.sha256").read_text()
    assert "campaign.json" in (diagnostic / "campaign.sha256").read_text()
    (repo / "scripts/diagnose_part_fullscale_timing.py").write_text("# later edit\n")
    assert "standalone diagnostic snapshot" in (diagnostic / "source/scripts/diagnose_part_fullscale_timing.py").read_text()


def test_each_submission_gets_new_diagnostic_directory(timing_launcher):
    run, _, campaign, calls = timing_launcher
    submit(timing_launcher)
    assert run().returncode == 0
    assert len(list((campaign / "diagnostics").glob("timing.*"))) == 2
    assert len(calls.read_text().splitlines()) == 2
    assert (campaign / "submitted_job_ids.tsv").read_text().startswith("train\t283716\t")


def test_worker_uses_original_source_pinned_python_and_single_rank(timing_launcher):
    run, _, campaign, _ = timing_launcher
    result = run("--train-steps", "1234", "--validation-steps", "456", "--budget-minutes", "99")
    assert result.returncode == 0, result.stderr
    diagnostic = next((campaign / "diagnostics").glob("timing.*"))
    before = snapshot(campaign)
    result = run(worker=diagnostic, extra_env={
        "RANK": "9", "LOCAL_RANK": "4", "WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8",
        "PYTHONPATH": "/wrong/source", "SRUN_NTASKS": "16",
    })
    assert result.returncode == 0, result.stderr
    assert "SRUN --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --ntasks-per-node=1" in result.stdout
    assert "SRUN_ENV_NTASKS=\n" in result.stdout
    assert "RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1" in result.stdout
    assert f"PYTHONPATH={shell_path(campaign / 'source')}" in result.stdout
    assert "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1" in result.stdout
    assert "--train-steps 1234 --validation-steps 456 --budget-minutes 99" in result.stdout
    assert "source/scripts/diagnose_part_fullscale_timing.py" in result.stdout
    assert snapshot(campaign) == before


@pytest.mark.parametrize("relative", [
    "source/scripts/diagnose_part_fullscale_timing.py", "source/sbatch/run_part_fullscale_timing.sh",
    "launch.env", "campaign.sha256",
])
def test_worker_rejects_tampered_diagnostic_before_python(timing_launcher, relative):
    run, _, _, _ = timing_launcher
    diagnostic = submit(timing_launcher)
    with (diagnostic / relative).open("a") as handle:
        handle.write("# tampered\n")
    result = run(worker=diagnostic)
    assert result.returncode != 0 and "Diagnostic checksum" in result.stderr
    assert "PYTHON" not in result.stdout


@pytest.mark.parametrize("relative", [
    "source/scripts/run_part_fullscale.py", "launch.env", "source.sha256", "campaign.json",
])
def test_worker_rejects_tampered_campaign_before_python(timing_launcher, relative):
    run, _, campaign, _ = timing_launcher
    diagnostic = submit(timing_launcher)
    with (campaign / relative).open("a") as handle:
        handle.write("# tampered\n")
    result = run(worker=diagnostic)
    assert result.returncode != 0 and "checksum" in result.stderr
    assert "PYTHON" not in result.stdout


@pytest.mark.parametrize("env, message", [
    ({"TEST_ARCH": "x86_64"}, "requires Tigris aarch64"),
    ({"SLURM_NTASKS": "2"}, "exactly one node and one task"),
    ({"SLURM_NNODES": "2"}, "exactly one node and one task"),
])
def test_worker_rejects_wrong_architecture_and_allocation(timing_launcher, env, message):
    run, _, _, _ = timing_launcher
    diagnostic = submit(timing_launcher)
    result = run(worker=diagnostic, extra_env=env)
    assert result.returncode != 0 and message in result.stderr
    assert "PYTHON" not in result.stdout


def test_submit_rejects_invalid_original_source_without_creating_diagnostic(timing_launcher):
    run, _, campaign, calls = timing_launcher
    (campaign / "source/scripts/run_part_fullscale.py").write_text("# changed\n")
    result = run()
    assert result.returncode != 0 and "source checksum" in result.stderr
    assert not (campaign / "diagnostics").exists() and not calls.exists()


def test_invalid_scheduler_reply_retains_only_diagnostic_snapshot(timing_launcher):
    run, _, campaign, _ = timing_launcher
    before = snapshot(campaign)
    result = run(extra_env={"TEST_SBATCH_BAD": "1"})
    assert result.returncode != 0 and "Inspect the queue before retrying" in result.stderr
    diagnostic = next((campaign / "diagnostics").glob("timing.*"))
    assert not (diagnostic / "submitted_job_id.txt").exists()
    assert (diagnostic / "diagnostic.sha256").is_file()
    assert snapshot(campaign) == before

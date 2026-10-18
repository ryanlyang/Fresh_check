"""Validation-only launch isolation and recovery tests with fake Slurm."""
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
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def validation_launcher(tmp_path):
    repo, campaign, output, fakebin = (tmp_path / name for name in
                                      ("source repo", "campaign", "validation", "bin"))
    for path in (repo / "sbatch", repo / "scripts", campaign / "source/scripts", fakebin):
        path.mkdir(parents=True)
    for name in ("submit_part_fullscale_validation.sh", "run_part_fullscale_validation.sh"):
        shutil.copyfile(REPO / "sbatch" / name, repo / "sbatch" / name)
    for name in ("evaluate_part_fullscale_validation.py", "part_fullscale_validation_support.py"):
        (repo / "scripts" / name).write_text("# standalone validation snapshot\n")
    original = campaign / "source/scripts/run_part_fullscale.py"
    original.write_text("# immutable original training implementation\n")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    (campaign / "source.sha256").write_text(f"{digest}  ./scripts/run_part_fullscale.py\n", newline="\n")
    (campaign / "campaign.json").write_text("{}\n")
    (campaign / "submitted_job_ids.tsv").write_text("train\t283716\t2026-10-13\n")
    (campaign / "checkpoint.pt").write_bytes(b"preserve original weights")
    (campaign / "launch.env").write_text(
        f"PART_FULLSCALE_PYTHON='{shell_path(fakebin / 'python')}'\n"
        "PART_FULLSCALE_DATA_ROOT=/data/JetClass\n", newline="\n")
    calls = tmp_path / "calls.txt"
    stubs = {
        "flock": "exit 0\n",  # Git Bash lacks flock; real locking is not asserted.
        "squeue": '[[ "${TEST_QUEUE_FAIL:-}" != 1 ]] || exit 3\nprintf "%s\\n" "${TEST_LIVE_JOBS:-}"\n',
        "sbatch": (
            'printf "%s|ENV_DEP=%s|" "$PART_VALIDATION_PHASE" "${SBATCH_DEPENDENCY:-}" >> "$TEST_CALLS"\n'
            'printf "%s " "$@" >> "$TEST_CALLS"\nprintf "\\n" >> "$TEST_CALLS"\n'
            'if [[ "${TEST_SBATCH_BAD:-}" == 1 ]]; then echo invalid; else '
            'echo "$((700000 + $(wc -l < "$TEST_CALLS")));cluster"; fi\n'
        ),
        "python": (
            'printf "PYTHON %s\\n" "$*"\n'
            'printf "RANK=%s LOCAL_RANK=%s WORLD_SIZE=%s LOCAL_WORLD_SIZE=%s\\n" "$RANK" "$LOCAL_RANK" "$WORLD_SIZE" "$LOCAL_WORLD_SIZE"\n'
            'printf "PYTHONPATH=%s PYTHONNOUSERSITE=%s PYTHONDONTWRITEBYTECODE=%s\\n" "$PYTHONPATH" "$PYTHONNOUSERSITE" "$PYTHONDONTWRITEBYTECODE"\n'
        ),
        "uname": 'echo "${TEST_ARCH:-aarch64}"\n',
        "srun": (
            'printf "SRUN %s\\n" "$*"\nprintf "SRUN_ENV_NTASKS=%s\\n" "${SRUN_NTASKS:-}"\n'
            'while [[ "$1" == --* ]]; do shift; done\nexec "$@"\n'
        ),
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
        script = worker / "source/sbatch/run_part_fullscale_validation.sh" if worker else repo / "sbatch/submit_part_fullscale_validation.sh"
        defaults = [] if worker else ["--campaign", shell_path(campaign), "--output", shell_path(output)]
        if worker:
            env["PART_VALIDATION_OUTPUT"] = shell_path(worker)
        return subprocess.run([BASH, "--noprofile", "--norc", "-c",
                               'export PATH="$1:$PATH"; shift; exec bash "$@"', "test",
                               shell_path(fakebin), shell_path(script), *defaults, *args],
                              env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)

    return run, repo, campaign, output, calls


def submit(fixture):
    run, _, _, output, _ = fixture
    result = run()
    assert result.returncode == 0, result.stderr
    return output


def test_validation_dry_run_leaves_original_and_output_untouched(validation_launcher):
    run, _, campaign, output, calls = validation_launcher
    before = snapshot(campaign)
    result = run("--dry-run", "--dependency", "283716")
    assert result.returncode == 0, result.stderr
    assert "--array=0-5%3" in result.stdout and "--mem=32G --time=04:00:00" in result.stdout
    assert "--time=00:30:00" in result.stdout and "--time=01:00:00" in result.stdout
    assert "Prepare dependency: afterok:283716" in result.stdout
    assert snapshot(campaign) == before and not output.exists() and not calls.exists()


@pytest.mark.parametrize("args", [
    ("--time", "06:00:00"), ("--time", "3-00:00:00"), ("--time", "00:00:00"),
    ("--concurrency", "0"), ("--concurrency", "7"), ("--cpus", "0"),
    ("--dependency", "283716;evil"), ("--partition", "tigris;evil"),
    ("--mode", "train"), ("--gpus", "2"),
])
def test_invalid_options_do_not_submit_or_write(validation_launcher, args):
    run, _, campaign, output, calls = validation_launcher
    before = snapshot(campaign)
    result = run(*args)
    assert result.returncode != 0
    assert snapshot(campaign) == before and not output.exists() and not calls.exists()


@pytest.mark.parametrize("relative", ["", "source/evaluation", "runs/evaluation", "final_test/evaluation"])
def test_output_cannot_be_original_or_inside_it(validation_launcher, relative):
    run, _, campaign, output, calls = validation_launcher
    before = snapshot(campaign)
    result = run("--output", shell_path(campaign / relative), "--dry-run")
    assert result.returncode != 0
    assert snapshot(campaign) == before and not output.exists() and not calls.exists()


def test_submission_freezes_new_tools_without_modifying_training(validation_launcher):
    run, repo, campaign, output, calls = validation_launcher
    before = snapshot(campaign)
    result = run("--dependency", "283716", extra_env={"SBATCH_DEPENDENCY": "afterok:999", "SBATCH_TIMELIMIT": "3-00:00:00"})
    assert result.returncode == 0, result.stderr
    rows = calls.read_text().splitlines()
    assert [row.split("|", 1)[0] for row in rows] == ["prepare", "evaluate", "report"]
    assert "ENV_DEP=|" in rows[0] and "999" not in rows[0]
    assert "--dependency=afterok:283716" in rows[0]
    assert "--dependency=afterok:700001" in rows[1] and "--array=0-5%3" in rows[1]
    assert "--dependency=afterok:700002" in rows[2]
    assert "--gpus-per-node=1" in rows[1] and "--gpus" not in rows[0] + rows[2]
    assert snapshot(campaign) == before
    assert (output / "original_metadata/campaign.json").read_bytes() == (campaign / "campaign.json").read_bytes()
    assert len((output / "submitted_job_ids.tsv").read_text().splitlines()) == 3
    assert not (output / "submission_uncertain").exists()
    assert "reports/validation_report.{json,md}" in result.stdout
    (repo / "scripts/evaluate_part_fullscale_validation.py").write_text("# changed later\n")
    assert "standalone validation snapshot" in (output / "source/scripts/evaluate_part_fullscale_validation.py").read_text()


def test_default_output_is_an_independent_campaign_sibling(validation_launcher):
    run, _, campaign, _, calls = validation_launcher
    result = run("--output", "")
    assert result.returncode == 0, result.stderr
    outputs = list(campaign.parent.glob("campaign_validation_*"))
    assert len(outputs) == 1 and outputs[0].parent == campaign.parent
    assert len(calls.read_text().splitlines()) == 3


@pytest.mark.parametrize("phase", ["prepare", "evaluate", "report"])
def test_worker_routes_read_only_phases_and_single_rank(validation_launcher, phase):
    run, _, campaign, _, _ = validation_launcher
    output = submit(validation_launcher)
    before = snapshot(campaign)
    result = run(worker=output, extra_env={
        "PART_VALIDATION_PHASE": phase, "SLURM_ARRAY_TASK_ID": "4", "SRUN_NTASKS": "9",
        "WORLD_SIZE": "16", "RANK": "15", "LOCAL_RANK": "3", "PYTHONPATH": "/wrong",
    })
    assert result.returncode == 0, result.stderr
    assert f"{phase} --output {shell_path(output)}" in result.stdout
    assert "RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1" in result.stdout
    assert f"PYTHONPATH={shell_path(campaign / 'source')}" in result.stdout
    assert "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1" in result.stdout
    if phase == "evaluate":
        assert "--task-index 4" in result.stdout and "SRUN_ENV_NTASKS=\n" in result.stdout
    else:
        assert "SRUN " not in result.stdout
    if phase == "prepare":
        assert f"--campaign {shell_path(campaign)}" in result.stdout
    assert snapshot(campaign) == before


@pytest.mark.parametrize("relative", [
    "source/scripts/evaluate_part_fullscale_validation.py",
    "source/scripts/part_fullscale_validation_support.py", "source/sbatch/run_part_fullscale_validation.sh",
    "validation_launch.env", "original.sha256", "original_metadata/launch.env",
])
def test_worker_rejects_changed_evaluation_snapshot(validation_launcher, relative):
    run, _, _, _, _ = validation_launcher
    output = submit(validation_launcher)
    with (output / relative).open("a") as handle:
        handle.write("# tampered\n")
    result = run(worker=output, extra_env={"PART_VALIDATION_PHASE": "prepare"})
    assert result.returncode != 0 and "checksum" in result.stderr
    assert "PYTHON" not in result.stdout


@pytest.mark.parametrize("relative", ["campaign.json", "source.sha256", "launch.env", "source/scripts/run_part_fullscale.py"])
def test_worker_rejects_changed_original_metadata_or_source(validation_launcher, relative):
    run, _, campaign, _, _ = validation_launcher
    output = submit(validation_launcher)
    with (campaign / relative).open("a") as handle:
        handle.write("# tampered\n")
    result = run(worker=output, extra_env={"PART_VALIDATION_PHASE": "prepare"})
    assert result.returncode != 0 and "checksum" in result.stderr
    assert "PYTHON" not in result.stdout


@pytest.mark.parametrize("env, message", [
    ({"TEST_ARCH": "x86_64"}, "requires Tigris aarch64"),
    ({"SLURM_NTASKS": "2"}, "exactly one node and one task"),
    ({"SLURM_NNODES": "2"}, "exactly one node and one task"),
    ({"SLURM_ARRAY_TASK_ID": "6"}, "task index must be 0..5"),
    ({"PART_VALIDATION_PHASE": "train"}, "Unknown validation-only phase"),
])
def test_worker_rejects_wrong_environment_or_phase(validation_launcher, env, message):
    run, _, _, _, _ = validation_launcher
    output = submit(validation_launcher)
    result = run(worker=output, extra_env={"PART_VALIDATION_PHASE": "evaluate", **env})
    assert result.returncode != 0 and message in result.stderr
    assert "PYTHON" not in result.stdout


@pytest.mark.parametrize("line", ["PART_FULLSCALE_COMPARISON=plain", "PART_FULLSCALE_TRAINING_RECIPE=adamw"])
def test_other_campaign_profiles_are_rejected_before_output_creation(validation_launcher, line):
    run, _, campaign, output, calls = validation_launcher
    with (campaign / "launch.env").open("a") as handle:
        handle.write(line + "\n")
    result = run()
    assert result.returncode != 0 and "standard/Ranger" in result.stderr
    assert not output.exists() and not calls.exists()


def test_resume_keeps_frozen_source_and_refuses_live_jobs(validation_launcher):
    run, repo, campaign, _, calls = validation_launcher
    output = submit(validation_launcher)
    source_before = snapshot(output / "source")
    before = snapshot(campaign)
    result = run("--resume", shell_path(output), extra_env={"TEST_LIVE_JOBS": "700002_[0-5%3]"})
    assert result.returncode != 0 and "still live" in result.stderr
    assert len(calls.read_text().splitlines()) == 3
    (repo / "scripts/evaluate_part_fullscale_validation.py").write_text("# later repository change\n")
    result = run("--resume", shell_path(output))
    assert result.returncode == 0, result.stderr
    assert len(calls.read_text().splitlines()) == 6
    assert snapshot(output / "source") == source_before and snapshot(campaign) == before


@pytest.mark.parametrize("args", [("--time", "02:00:00"), ("--concurrency", "2"), ("--mem", "64G"), ("--account", "other")])
def test_resume_cannot_change_frozen_resources(validation_launcher, args):
    run, _, _, _, calls = validation_launcher
    output = submit(validation_launcher)
    result = run("--resume", shell_path(output), *args)
    assert result.returncode != 0 and "differs from frozen" in result.stderr
    assert len(calls.read_text().splitlines()) == 3


def test_resume_dry_run_is_read_only(validation_launcher):
    run, _, campaign, _, calls = validation_launcher
    output = submit(validation_launcher)
    original_before, evaluation_before = snapshot(campaign), snapshot(output)
    result = run("--resume", shell_path(output), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert snapshot(campaign) == original_before and snapshot(output) == evaluation_before
    assert len(calls.read_text().splitlines()) == 3


def test_ambiguous_submission_blocks_blind_resume(validation_launcher):
    run, _, _, output, calls = validation_launcher
    result = run(extra_env={"TEST_SBATCH_BAD": "1"})
    assert result.returncode != 0 and "Inspect the queue" in result.stderr
    assert (output / "submission_uncertain").is_file()
    result = run("--resume", shell_path(output))
    assert result.returncode != 0 and "interrupted/ambiguous" in result.stderr
    assert len(calls.read_text().splitlines()) == 1


def test_queue_failure_blocks_resume_without_new_submissions(validation_launcher):
    run, _, _, _, calls = validation_launcher
    output = submit(validation_launcher)
    result = run("--resume", shell_path(output), extra_env={"TEST_QUEUE_FAIL": "1"})
    assert result.returncode != 0 and "Cannot inspect live jobs" in result.stderr
    assert len(calls.read_text().splitlines()) == 3

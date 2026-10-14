"""Local launcher tests with a temporary source repository and fake Slurm only."""
from __future__ import annotations

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


@pytest.fixture
def launcher(tmp_path):
    source, output, fakebin = tmp_path / "source repo", tmp_path / "campaign", tmp_path / "bin"
    for path in (source / "sbatch", source / "scripts", source / "part_ev_fullscale", fakebin):
        path.mkdir(parents=True)
    for name in ("submit_part_fullscale.sh", "run_part_fullscale.sh"):
        shutil.copyfile(REPO / "sbatch" / name, source / "sbatch" / name)
    (source / "scripts/run_part_fullscale.py").write_text("# fixture driver\n")
    (source / "part_ev_fullscale/__init__.py").write_text("# fixture package\n")
    (source / "requirements-part-fullscale.txt").write_text("# fixture requirements\n")
    data = tmp_path / "data"
    data.mkdir()
    calls = tmp_path / "calls.txt"
    stubs = {
        "flock": "exit 0\n",  # Git Bash lacks flock. Real lock behavior is not mocked as proven.
        "python": 'printf "PYTHON %s\\n" "$*"\nprintf "RANK=%s LOCAL_RANK=%s WORLD_SIZE=%s\\n" "${RANK:-}" "${LOCAL_RANK:-}" "${WORLD_SIZE:-}"\n',
        "squeue": 'printf "%s\\n" "${TEST_LIVE_JOBS:-}"\n',
        "sbatch": (
            'printf "%s|" "$PART_FULLSCALE_PHASE" >> "$TEST_CALLS"\n'
            'printf "%s " "$@" >> "$TEST_CALLS"\nprintf "\\n" >> "$TEST_CALLS"\n'
            'if [[ "${TEST_SBATCH_BAD:-}" == 1 ]]; then echo invalid; else '
            'echo "$((700000 + $(wc -l < "$TEST_CALLS")))"; fi\n'
        ),
        "uname": 'echo "${TEST_ARCH:-aarch64}"\n',
        "scontrol": 'echo gh-a-001\n',
        "srun": (
            'printf "SRUN %s\\n" "$*"\n'
            'while [[ "$1" == --* ]]; do shift; done\n'
            'export SLURM_PROCID=1\nexec "$@"\n'
        ),
    }
    for name, body in stubs.items():
        path = fakebin / name
        path.write_text("#!/usr/bin/env bash\n" + body, newline="\n")
        path.chmod(0o755)
    setup = subprocess.run([BASH, "-c", 'git init -q "$1" && git -C "$1" add . && '
                            'git -C "$1" -c user.name=Test -c user.email=test@example.invalid '
                            '-c commit.gpgsign=false commit -qm fixture', "init", shell_path(source)],
                           capture_output=True, text=True, timeout=30)
    assert setup.returncode == 0, setup.stderr

    def run(*args, extra_env=None, worker=False):
        env = dict(os.environ, TEST_CALLS=shell_path(calls))
        env.update(extra_env or {})
        script = output / "source/sbatch/run_part_fullscale.sh" if worker else source / "sbatch/submit_part_fullscale.sh"
        defaults = [] if worker else ["--output", shell_path(output), "--python", shell_path(fakebin / "python"),
                                     "--data-root", shell_path(data)]
        return subprocess.run([BASH, "--noprofile", "--norc", "-c",
                               'export PATH="$1:$PATH"; shift; exec bash "$@"', "test",
                               shell_path(fakebin), shell_path(script), *defaults, *args],
                              env=env, capture_output=True, text=True, timeout=30)

    return run, source, output, calls, data


def prepare(launcher):
    run, _, output, _, _ = launcher
    result = run("--mode", "prepare")
    assert result.returncode == 0, result.stderr
    (output / "campaign.json").write_text("{}\n")


def test_dry_run_does_not_create_campaign_or_submit(launcher):
    run, _, output, calls, _ = launcher
    result = run("--mode", "train", "--gpus", "4", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--nodes=4 --ntasks-per-node=1 --gpus-per-node=1" in result.stdout
    assert "--mem=128G" in result.stdout
    assert "--array=0-5%2" in result.stdout
    assert "original-recipe preview" in result.stdout
    assert not output.exists() and not calls.exists()


@pytest.mark.parametrize("args", [("--gpus", "3"), ("--dependency", "123;456"),
                                  ("--concurrency", "7"), ("--time", "bad"),
                                  ("--mode", "test")])
def test_invalid_arguments_never_submit(launcher, args):
    run, _, output, calls, _ = launcher
    result = run("--mode", "train", *args)
    assert result.returncode != 0
    assert not output.exists() and not calls.exists()


def test_prepare_freezes_source_and_no_training_is_queued(launcher):
    run, source, output, calls, _ = launcher
    prepare(launcher)
    assert (output / "source/scripts/run_part_fullscale.py").is_file()
    assert len((output / "source_commit.txt").read_text().strip()) == 40
    assert "./scripts/run_part_fullscale.py" in (output / "source.sha256").read_text()
    assert calls.read_text().count("\n") == 1
    assert calls.read_text().startswith("prepare|")
    assert "--gpus-per-node" not in calls.read_text()
    # An unrelated working-tree edit must not change the frozen source.
    (source / "paper_notes.txt").write_text("unrelated\n")
    result = run("--mode", "benchmark", "--gpus", "2")
    assert result.returncode == 0, result.stderr
    assert not (output / "source/paper_notes.txt").exists()


def test_dirty_relevant_source_is_not_frozen(launcher):
    run, source, output, calls, _ = launcher
    (source / "scripts/run_part_fullscale.py").write_text("changed\n")
    result = run("--mode", "prepare")
    assert result.returncode != 0 and "dirty source" in result.stderr
    assert not output.exists() and not calls.exists()


def test_train_fixed_six_tasks_multinode_and_explicit_dependency(launcher):
    run, _, _, calls, _ = launcher
    prepare(launcher)
    result = run("--mode", "train", "--gpus", "4", "--dependency", "700001", extra_env={"SBATCH_DEPENDENCY": "afterok:999"})
    assert result.returncode == 0, result.stderr
    row = calls.read_text().splitlines()[-1]
    assert "--nodes=4 --ntasks-per-node=1 --gpus-per-node=1" in row
    assert "--array=0-5%2" in row and "--dependency=afterok:700001" in row
    assert "--signal=USR1@180" in row and "999" not in row


def test_resume_refuses_live_tasks_and_requires_explicit_intent(launcher):
    run, _, _, calls, _ = launcher
    prepare(launcher)
    assert run("--mode", "train").returncode == 0
    no_resume = run("--mode", "train")
    assert no_resume.returncode != 0 and "prior submission" in no_resume.stderr
    live = run("--mode", "train", "--resume", extra_env={"TEST_LIVE_JOBS": "700002_[2-5%2]"})
    assert live.returncode != 0 and "still live" in live.stderr
    assert len(calls.read_text().splitlines()) == 2
    assert run("--mode", "train", "--resume").returncode == 0


def test_testing_is_opt_in_and_report_depends_on_whole_array(launcher):
    run, _, _, calls, data = launcher
    prepare(launcher)
    (data / "ZJetsToNuNu_100.root").write_bytes(b"stub; driver validates actual inventory")
    result = run("--mode", "test", "--test-data", shell_path(data), "--dependency", "12345")
    assert result.returncode == 0, result.stderr
    rows = calls.read_text().splitlines()
    assert [r.split("|", 1)[0] for r in rows] == ["prepare", "test", "report"]
    assert "--dependency=afterok:12345" in rows[1]
    assert "--array=0-5%2" in rows[1]
    assert "--dependency=afterok:700002" in rows[2]
    assert "--kill-on-invalid-dep=yes" in rows[2]


def test_worker_maps_one_rank_per_node_and_authenticates_frozen_source(launcher):
    run, _, output, _, _ = launcher
    prepare(launcher)
    env = {"PART_FULLSCALE_OUTPUT": shell_path(output), "PART_FULLSCALE_PHASE": "train",
           "SLURM_ARRAY_TASK_ID": "3", "SLURM_JOB_NODELIST": "gh-a-[001-002]",
           "SLURM_JOB_ID": "12345", "SLURM_NTASKS": "2", "SLURM_NNODES": "2"}
    result = run(worker=True, extra_env=env)
    assert result.returncode == 0, result.stderr
    assert "SRUN --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=1" in result.stdout
    assert "--task-index 3" in result.stdout
    assert "RANK=1 LOCAL_RANK=0 WORLD_SIZE=2" in result.stdout
    (output / "source/scripts/run_part_fullscale.py").write_text("tampered\n")
    result = run(worker=True, extra_env=env)
    assert result.returncode != 0
    assert "PYTHON" not in result.stdout


def test_worker_rejects_wrong_architecture(launcher):
    run, _, output, _, _ = launcher
    prepare(launcher)
    result = run(worker=True, extra_env={"PART_FULLSCALE_OUTPUT": shell_path(output),
                 "PART_FULLSCALE_PHASE": "prepare", "TEST_ARCH": "x86_64"})
    assert result.returncode != 0 and "requires Tigris aarch64" in result.stderr


def test_prepare_must_complete_or_be_an_explicit_dependency(launcher):
    run, _, _, calls, _ = launcher
    assert run("--mode", "prepare").returncode == 0
    result = run("--mode", "train")
    assert result.returncode != 0 and "Prepare must finish" in result.stderr
    assert len(calls.read_text().splitlines()) == 1
    result = run("--mode", "train", "--dependency", "700001")
    assert result.returncode == 0, result.stderr


def test_empty_test_directory_is_rejected(launcher):
    run, _, _, calls, data = launcher
    prepare(launcher)
    result = run("--mode", "test", "--test-data", shell_path(data))
    assert result.returncode != 0 and "No staged ROOT files" in result.stderr
    assert len(calls.read_text().splitlines()) == 1


def test_invalid_scheduler_response_is_not_recorded_as_a_job(launcher):
    run, _, output, _, _ = launcher
    prepare(launcher)
    result = run("--mode", "train", extra_env={"TEST_SBATCH_BAD": "1"})
    assert result.returncode != 0 and "Inspect the queue before retrying" in result.stderr
    assert len((output / "submitted_job_ids.tsv").read_text().splitlines()) == 1


def test_worker_rejects_more_than_one_rank_per_node(launcher):
    run, _, output, _, _ = launcher
    prepare(launcher)
    result = run(worker=True, extra_env={"PART_FULLSCALE_OUTPUT": shell_path(output),
                 "PART_FULLSCALE_PHASE": "benchmark", "SLURM_JOB_NODELIST": "gh-a-001",
                 "SLURM_JOB_ID": "45678", "SLURM_NTASKS": "2", "SLURM_NNODES": "1"})
    assert result.returncode != 0 and "exactly one rank/GPU per node" in result.stderr


def test_distinct_gpu_benchmarks_are_not_duplicate_phases(launcher):
    run, _, _, calls, _ = launcher
    prepare(launcher)
    assert run("--mode", "benchmark", "--gpus", "1").returncode == 0
    result = run("--mode", "benchmark", "--gpus", "2", extra_env={"TEST_LIVE_JOBS": "700002"})
    assert result.returncode == 0, result.stderr
    assert len(calls.read_text().splitlines()) == 3


def test_plain_prepare_freezes_comparison_and_worker_passes_it(launcher):
    run, _, output, calls, _ = launcher
    result = run("--mode", "prepare", "--comparison", "plain")
    assert result.returncode == 0, result.stderr
    assert "PART_FULLSCALE_COMPARISON=plain" in (output / "launch.env").read_text()
    assert "--job-name=parte_plain_prepare" in calls.read_text()
    assert "--time=02:00:00" in calls.read_text()
    result = run(worker=True, extra_env={
        "PART_FULLSCALE_OUTPUT": shell_path(output), "PART_FULLSCALE_PHASE": "prepare",
        "PART_FULLSCALE_COMPARISON": "standard",  # inherited shell cannot change it
    })
    assert result.returncode == 0, result.stderr
    assert "--comparison plain" in result.stdout


def test_plain_defaults_to_a_distinct_output_directory_without_writes(launcher):
    run, source, output, calls, _ = launcher
    result = run("--mode", "prepare", "--comparison", "plain", "--output", "", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "/checkpoints/part_fullscale/part_plain_ev_" in result.stdout
    assert "Comparison for a new campaign: plain" in result.stdout
    assert not output.exists() and not calls.exists() and not (source / "checkpoints").exists()


@pytest.mark.parametrize("mode", ["benchmark", "train", "test"])
def test_comparison_cannot_be_selected_after_prepare(launcher, mode):
    run, _, output, calls, _ = launcher
    result = run("--mode", mode, "--comparison", "plain", "--dry-run")
    assert result.returncode != 0 and "prepare-only" in result.stderr
    assert not output.exists() and not calls.exists()


def test_invalid_comparison_creates_nothing(launcher):
    run, _, output, calls, _ = launcher
    result = run("--mode", "prepare", "--comparison", "typo")
    assert result.returncode != 0 and "standard or plain" in result.stderr
    assert not output.exists() and not calls.exists()


def test_prepare_resume_cannot_change_the_frozen_comparison(launcher):
    run, _, output, calls, _ = launcher
    prepare(launcher)
    original_env = (output / "launch.env").read_bytes()
    result = run("--mode", "prepare", "--comparison", "plain", "--resume")
    assert result.returncode != 0 and "differs from the frozen campaign" in result.stderr
    assert len(calls.read_text().splitlines()) == 1
    assert (output / "launch.env").read_bytes() == original_env


def test_plain_training_uses_frozen_choice_with_six_tasks_and_four_days(launcher):
    run, _, output, calls, _ = launcher
    assert run("--mode", "prepare", "--comparison", "plain").returncode == 0
    (output / "campaign.json").write_text("{}\n")
    result = run("--mode", "benchmark", "--gpus", "1", "--dependency", "700001")
    assert result.returncode == 0, result.stderr
    benchmark = calls.read_text().splitlines()[-1]
    assert "--job-name=parte_plain_benchmark" in benchmark and "--time=01:00:00" in benchmark
    result = run("--mode", "train", "--gpus", "1", "--concurrency", "6",
                 "--time", "4-00:00:00", "--dependency", "700002",
                 extra_env={"PART_FULLSCALE_COMPARISON": "standard"})
    assert result.returncode == 0, result.stderr
    training = calls.read_text().splitlines()[-1]
    assert "--job-name=parte_plain_train" in training
    assert "--array=0-5%6" in training and "--time=4-00:00:00" in training
    assert "--dependency=afterok:700002" in training
    assert "Comparison: plain" in result.stdout


def test_legacy_launch_environment_defaults_to_standard_despite_inherited_value(launcher):
    run, _, output, calls, _ = launcher
    prepare(launcher)
    env_path = output / "launch.env"
    env_path.write_text("\n".join(
        line for line in env_path.read_text().splitlines()
        if not line.startswith("PART_FULLSCALE_COMPARISON=")
    ) + "\n")
    result = run("--mode", "benchmark", extra_env={"PART_FULLSCALE_COMPARISON": "plain"})
    assert result.returncode == 0, result.stderr
    assert "--job-name=parte_benchmark" in calls.read_text().splitlines()[-1]
    result = run(worker=True, extra_env={
        "PART_FULLSCALE_OUTPUT": shell_path(output), "PART_FULLSCALE_PHASE": "prepare",
        "PART_FULLSCALE_COMPARISON": "plain",
    })
    assert result.returncode == 0, result.stderr
    assert "--comparison standard" in result.stdout


def test_adamw_prepare_freezes_recipe_and_worker_passes_it(launcher):
    run, _, output, calls, _ = launcher
    result = run("--mode", "prepare", "--training-recipe", "adamw")
    assert result.returncode == 0, result.stderr
    assert "PART_FULLSCALE_TRAINING_RECIPE=adamw" in (output / "launch.env").read_text()
    assert "PART_FULLSCALE_COMPARISON=standard" in (output / "launch.env").read_text()
    assert "--job-name=parte_adamw_prepare" in calls.read_text()
    assert "--time=02:00:00" in calls.read_text()
    result = run(worker=True, extra_env={
        "PART_FULLSCALE_OUTPUT": shell_path(output), "PART_FULLSCALE_PHASE": "prepare",
        "PART_FULLSCALE_TRAINING_RECIPE": "original",
    })
    assert result.returncode == 0, result.stderr
    assert "--comparison standard --training-recipe adamw" in result.stdout


def test_adamw_defaults_to_distinct_campaign_directory_without_writes(launcher):
    run, source, output, calls, _ = launcher
    result = run("--mode", "prepare", "--training-recipe", "adamw", "--output", "", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "/checkpoints/part_fullscale/part_ev_adamw_" in result.stdout
    assert "training recipe: adamw; tasks: 2" in result.stdout
    assert not output.exists() and not calls.exists() and not (source / "checkpoints").exists()


@pytest.mark.parametrize("mode", ["benchmark", "train", "test"])
def test_training_recipe_cannot_be_selected_after_prepare(launcher, mode):
    run, _, output, calls, _ = launcher
    result = run("--mode", mode, "--training-recipe", "adamw", "--dry-run")
    assert result.returncode != 0 and "prepare-only" in result.stderr
    assert not output.exists() and not calls.exists()


@pytest.mark.parametrize("args", [
    ("--training-recipe", "adamw", "--comparison", "plain"),
    ("--training-recipe", "typo"),
    ("--training-recipe", "adamw", "--concurrency", "3"),
])
def test_invalid_adamw_configuration_creates_nothing(launcher, args):
    run, _, output, calls, _ = launcher
    result = run("--mode", "prepare", *args)
    assert result.returncode != 0
    assert not output.exists() and not calls.exists()


def test_original_prepare_resume_cannot_switch_to_adamw(launcher):
    run, _, output, calls, _ = launcher
    prepare(launcher)
    original_env = (output / "launch.env").read_bytes()
    result = run("--mode", "prepare", "--training-recipe", "adamw", "--resume")
    assert result.returncode != 0 and "recipe differs from the frozen campaign" in result.stderr
    assert len(calls.read_text().splitlines()) == 1
    assert (output / "launch.env").read_bytes() == original_env


def test_adamw_prepare_resume_without_recipe_retains_its_frozen_profile(launcher):
    run, _, output, calls, _ = launcher
    assert run("--mode", "prepare", "--training-recipe", "adamw").returncode == 0
    original_env = (output / "launch.env").read_bytes()
    result = run("--mode", "prepare", "--resume")
    assert result.returncode == 0, result.stderr
    assert "--job-name=parte_adamw_prepare" in calls.read_text().splitlines()[-1]
    assert (output / "launch.env").read_bytes() == original_env


def test_adamw_dependent_training_has_exactly_two_tasks_before_preflight_completes(launcher):
    run, _, output, calls, _ = launcher
    assert run("--mode", "prepare", "--training-recipe", "adamw").returncode == 0
    assert not (output / "campaign.json").exists()
    result = run("--mode", "benchmark", "--gpus", "1", "--dependency", "700001")
    assert result.returncode == 0, result.stderr
    benchmark = calls.read_text().splitlines()[-1]
    assert "--job-name=parte_adamw_benchmark" in benchmark and "--time=01:00:00" in benchmark
    result = run("--mode", "train", "--gpus", "1", "--concurrency", "2",
                 "--time", "4-00:00:00", "--dependency", "700002",
                 extra_env={"PART_FULLSCALE_TRAINING_RECIPE": "original"})
    assert result.returncode == 0, result.stderr
    training = calls.read_text().splitlines()[-1]
    assert "--job-name=parte_adamw_train" in training
    assert "--array=0-1%2" in training and "--time=4-00:00:00" in training
    assert "--dependency=afterok:700002" in training
    assert "Training recipe: adamw (2 tasks)" in result.stdout


def test_adamw_existing_dry_run_resolves_two_tasks_and_does_not_submit(launcher):
    run, _, output, calls, _ = launcher
    assert run("--mode", "prepare", "--training-recipe", "adamw").returncode == 0
    original_calls = calls.read_bytes()
    result = run("--mode", "train", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "Frozen comparison: standard; training recipe: adamw; tasks: 2" in result.stdout
    assert "--array=0-1%2" in result.stdout and "original-recipe preview" not in result.stdout
    assert calls.read_bytes() == original_calls
    assert not (output / "campaign.json").exists()


def test_adamw_existing_campaign_rejects_three_concurrent_tasks(launcher):
    run, _, _, calls, _ = launcher
    assert run("--mode", "prepare", "--training-recipe", "adamw").returncode == 0
    result = run("--mode", "train", "--concurrency", "3", "--dependency", "700001")
    assert result.returncode != 0 and "only 2 train/test tasks" in result.stderr
    assert len(calls.read_text().splitlines()) == 1


@pytest.mark.parametrize("index, valid", [("0", True), ("1", True), ("2", False), ("5", False)])
def test_adamw_worker_validates_two_task_indices(launcher, index, valid):
    run, _, output, _, _ = launcher
    assert run("--mode", "prepare", "--training-recipe", "adamw").returncode == 0
    result = run(worker=True, extra_env={
        "PART_FULLSCALE_OUTPUT": shell_path(output), "PART_FULLSCALE_PHASE": "train",
        "SLURM_ARRAY_TASK_ID": index, "SLURM_JOB_NODELIST": "gh-a-001",
        "SLURM_JOB_ID": "12345", "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
    })
    if valid:
        assert result.returncode == 0, result.stderr
        assert f"--task-index {index}" in result.stdout
        assert "--training-recipe" not in result.stdout  # later phases load campaign.json
    else:
        assert result.returncode != 0 and "Task index must be 0..1" in result.stderr
        assert "PYTHON" not in result.stdout


def test_adamw_testing_is_two_tasks_with_a_dependent_report(launcher):
    run, _, output, calls, data = launcher
    assert run("--mode", "prepare", "--training-recipe", "adamw").returncode == 0
    (output / "campaign.json").write_text("{}\n")
    (data / "ZJetsToNuNu_100.root").write_bytes(b"fixture ROOT placeholder")
    result = run("--mode", "test", "--test-data", shell_path(data), "--dependency", "12345")
    assert result.returncode == 0, result.stderr
    rows = calls.read_text().splitlines()
    assert [r.split("|", 1)[0] for r in rows] == ["prepare", "test", "report"]
    assert "--job-name=parte_adamw_test" in rows[1] and "--array=0-1%2" in rows[1]
    assert "--dependency=afterok:12345" in rows[1]
    assert "--job-name=parte_adamw_report" in rows[2] and "--dependency=afterok:700002" in rows[2]


def test_legacy_environment_defaults_to_original_despite_inherited_adamw(launcher):
    run, _, output, calls, _ = launcher
    prepare(launcher)
    env_path = output / "launch.env"
    env_path.write_text("\n".join(
        line for line in env_path.read_text().splitlines()
        if not line.startswith("PART_FULLSCALE_TRAINING_RECIPE=")
    ) + "\n")
    result = run("--mode", "train", extra_env={"PART_FULLSCALE_TRAINING_RECIPE": "adamw"})
    assert result.returncode == 0, result.stderr
    assert "--job-name=parte_train" in calls.read_text().splitlines()[-1]
    assert "--array=0-5%2" in calls.read_text().splitlines()[-1]
    result = run(worker=True, extra_env={
        "PART_FULLSCALE_OUTPUT": shell_path(output), "PART_FULLSCALE_PHASE": "prepare",
        "PART_FULLSCALE_TRAINING_RECIPE": "adamw",
    })
    assert result.returncode == 0, result.stderr
    assert "--training-recipe original" in result.stdout

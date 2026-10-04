"""Run the real Bash launcher against isolated, fake Conda and Slurm commands."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SUBMITTER = REPO / "sbatch/submit_relational_part_sporc_inference.sh"


def shell_path(path):
    value = str(Path(path).resolve()).replace("\\", "/")
    if len(value) > 2 and value[1] == ":":
        return "/" + value[0].lower() + value[2:]
    return value


def executable(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8", newline="\n")
    path.chmod(0o755)


@pytest.fixture
def launcher(tmp_path):
    if os.name == "nt":
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.is_file():
            pytest.skip("Git Bash is required for launcher behavior tests on Windows")
    else:
        bash = shutil.which("bash")
        if not bash:
            pytest.skip("Bash is unavailable")
    bin_dir = tmp_path / "bin"
    helper = tmp_path / "fake_python.py"
    # Execute actual embedded config/discovery programs. Only the remote driver
    # and x86/platform smoke check are mocked; never import torch or contact Slurm.
    executable(helper, r'''import contextlib, io, json, os, pathlib, sys
sys.stdout.reconfigure(newline="\n")
args = sys.argv[1:]
def shell(value):
    value = str(value).replace("\\", "/")
    return "/" + value[0].lower() + value[2:] if len(value) > 2 and value[1] == ":" else value
os.environ["CONDA_BASE"] = shell(os.environ["CONDA_BASE"])
if args[0] == "__mkdir__":
    pathlib.Path(args[-1]).mkdir(parents=True, exist_ok=True)
elif args[0] == "__realpath__":
    print(shell(pathlib.Path(args[-1]).resolve(strict="-e" in args)))
elif args[0] == "__sbatch__":
    ledger = pathlib.Path(os.environ["RPT_TEST_CALLS"])
    previous = ledger.read_text().splitlines() if ledger.exists() else []
    phase = os.environ["RPT_SPORC_PHASE"]
    with ledger.open("a") as stream:
        stream.write(json.dumps({"phase": phase, "arguments": args[1:]}) + "\n")
    if phase == os.environ.get("RPT_TEST_FAIL_PHASE"):
        print("simulated sbatch failure", file=sys.stderr)
        raise SystemExit(1)
    print(700001 + len(previous))
elif args[0] == "-c":
    if "platform.machine()" in args[1]:
        print("SPORC Python: isolated test stub 3.12")
    else:
        sys.argv = ["-c", *args[2:]]
        exec(args[1], {"__name__": "__main__"})
elif args[0] == "-":
    sys.argv = ["-", *args[1:]]
    captured = io.StringIO()
    code = sys.stdin.read()
    with contextlib.redirect_stdout(captured):
        exec(code, {"__name__": "__main__"})
    lines = captured.getvalue().splitlines()
    for index, line in enumerate(lines):
        path_result = "print(matches[0])" in code or (index == len(lines) - 1 and 'print(manifest["source_campaign"])' in code)
        print(shell(line) if path_result else line)
else:
    if args[0] == "-u":
        args.pop(0)
    assert args[0].endswith("run_relational_part_sporc_inference.py"), args
    phase = args[1]
    root = pathlib.Path(args[args.index("--output") + 1]).resolve()
    if phase == "bootstrap":
        root.mkdir()
        (root / "source/sbatch").mkdir(parents=True)
        (root / "source/scripts").mkdir()
        (root / "source/sbatch/run_relational_part_sporc_inference.sh").write_text("#!/bin/bash\n")
        (root / "source/scripts/run_relational_part_sporc_inference.py").write_text("# stub\n")
        spec = dict(output=str(root), jobs=int(args[args.index("--jobs") + 1]),
                    source_campaign=args[args.index("--source-campaign") + 1])
        (root / "sporc_campaign.json").write_text(json.dumps(spec))
    elif phase == "check":
        assert (root / "sporc_campaign.json").is_file()
    else:
        raise AssertionError("Launcher must not run a worker phase locally: " + phase)
''')
    wrapper = '#!/usr/bin/env bash\nexec "$RPT_TEST_PYTHON" "$RPT_TEST_HELPER" "$@"\n'
    executable(bin_dir / "python", wrapper)
    for command in ("mkdir", "realpath"):
        executable(bin_dir / command, f'#!/usr/bin/env bash\nexec "$RPT_TEST_PYTHON" "$RPT_TEST_HELPER" __{command}__ "$@"\n')
    executable(bin_dir / "sbatch", '#!/usr/bin/env bash\nexec "$RPT_TEST_PYTHON" "$RPT_TEST_HELPER" __sbatch__ "$@"\n')
    executable(bin_dir / "squeue", '#!/usr/bin/env bash\n[[ "${RPT_TEST_SQUEUE_FAIL:-0}" != 1 ]] || exit 1\nprintf "%s\\n" "${RPT_TEST_SQUEUE:-}"\n')
    executable(bin_dir / "flock", '#!/usr/bin/env bash\nexit "${RPT_TEST_LOCK_FAIL:-0}"\n')
    executable(bin_dir / "uname", '#!/usr/bin/env bash\nprintf "x86_64\\n"\n')
    conda = tmp_path / "conda"
    executable(conda / "etc/profile.d/conda.sh", 'conda() { export CONDA_PREFIX="${CONDA_BASE}/envs/${2}"; }\n')
    source = tmp_path / "original"
    (source / "selection").mkdir(parents=True)
    (source / "test").mkdir()
    (source / "selection/locked_models.json").write_text("{}")
    (source / "test/evaluation_plan.json").write_text("{}")
    output = tmp_path / "sporc"
    calls = tmp_path / "calls.jsonl"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("RPT_", "SBATCH_")) and key not in ("GPU_GRES", "CONDA_BASE", "CONDA_ENV")}
    env.update(CONDA_BASE=shell_path(conda), CONDA_ENV="atlas_kd_sporc",
               RPT_TEST_BIN=shell_path(bin_dir), RPT_TEST_PYTHON=shell_path(sys.executable),
               RPT_TEST_HELPER=shell_path(helper), RPT_TEST_SUBMITTER=shell_path(SUBMITTER),
               RPT_TEST_CALLS=str(calls))

    def run(*arguments, extra_env=None):
        return subprocess.run([str(bash), "--noprofile", "--norc", "-c",
            'export PATH="$RPT_TEST_BIN:$PATH"; exec bash "$RPT_TEST_SUBMITTER" "$@"',
            "launcher-test", *arguments], env={**env, **(extra_env or {})},
            capture_output=True, text=True, timeout=30)

    def read_calls():
        return [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []

    return run, source, output, read_calls


def fresh_args(source, output):
    return ("--source-campaign", shell_path(source), "--output", shell_path(output))


def test_dry_run_is_read_only_and_prints_requested_graph(launcher):
    run, source, output, calls = launcher
    result = run(*fresh_args(source, output), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    assert calls() == []
    assert "--array=0-39%2" in result.stderr
    assert "--array=0-11%2" in result.stderr
    assert "--time=23:30:00" in result.stderr
    assert "--partition=debug" in result.stderr
    assert "--gres=gpu:a100:1" in result.stderr


def test_real_shell_submits_exact_afterok_graph_and_appends_ledger(launcher):
    run, source, output, calls = launcher
    result = run(*fresh_args(source, output), extra_env={"RPT_SPORC_PHASE": "stale_inherited_phase"})
    assert result.returncode == 0, result.stderr
    records = calls()
    assert [r["phase"] for r in records] == ["build", "validate", "infer", "aggregate", "report"]
    assert "--array=0-39%2" in records[2]["arguments"]
    assert "--array=0-11%2" in records[3]["arguments"]
    assert not any(a.startswith("--dependency=") for a in records[0]["arguments"])
    for index, record in enumerate(records[1:], 1):
        assert f"--dependency=afterok:{700000 + index}" in record["arguments"]
        assert "--kill-on-invalid-dep=yes" in record["arguments"]
    assert (output / "submitted_job_ids.txt").read_text().splitlines() == [
        f"{phase} {700001 + i}" for i, phase in enumerate(["build", "validate", "infer", "aggregate", "report"])]


def test_sbatch_failure_stops_graph_and_preserves_partial_ledger(launcher):
    run, source, output, calls = launcher
    result = run(*fresh_args(source, output), extra_env={"RPT_TEST_FAIL_PHASE": "infer"})
    assert result.returncode != 0
    assert [r["phase"] for r in calls()] == ["build", "validate", "infer"]
    assert (output / "submitted_job_ids.txt").read_text().splitlines() == ["build 700001", "validate 700002"]


def test_fresh_tier3_graph_keeps_parity_dependency_and_cannot_relabel_debug_resume(launcher):
    run, source, output, calls = launcher
    result = run(*fresh_args(source, output), extra_env={"SBATCH_PARTITION": "tier3"})
    assert result.returncode == 0, result.stderr
    records = calls()
    assert all("--partition=tier3" in row["arguments"] for row in records)
    assert "--dependency=afterok:700002" in records[2]["arguments"]
    changed = run("--resume", shell_path(output), extra_env={"SBATCH_PARTITION": "debug"})
    assert changed.returncode != 0 and "Cannot change frozen partition" in changed.stderr
    assert len(calls()) == 5


def test_resume_rejects_live_jobs_and_changed_jobs_then_can_restart(launcher):
    run, source, output, calls = launcher
    assert run(*fresh_args(source, output)).returncode == 0
    count = len(calls())
    live = run("--resume", shell_path(output), extra_env={"RPT_TEST_SQUEUE": "700003_4|anything"})
    assert live.returncode != 0 and "remains queued/running" in live.stderr
    assert len(calls()) == count
    changed = run("--resume", shell_path(output), "--jobs", "20")
    assert changed.returncode != 0 and "Cannot change frozen jobs" in changed.stderr
    assert len(calls()) == count
    resumed = run("--resume", shell_path(output))
    assert resumed.returncode == 0, resumed.stderr
    assert len(calls()) == 10
    assert len((output / "submitted_job_ids.txt").read_text().splitlines()) == 10


@pytest.mark.parametrize("extra_env", [{"RPT_TEST_SQUEUE_FAIL": "1"}, {"RPT_TEST_LOCK_FAIL": "1"}])
def test_queue_or_lock_failure_is_fail_closed(launcher, extra_env):
    run, source, output, calls = launcher
    result = run(*fresh_args(source, output), extra_env=extra_env)
    assert result.returncode != 0
    assert not calls()


@pytest.mark.parametrize("arguments", [("--jobs", "0"), ("--jobs", "201"), ("--concurrency", "41")])
def test_invalid_array_bounds_never_bootstrap_or_submit(launcher, arguments):
    run, source, output, calls = launcher
    result = run(*fresh_args(source, output), *arguments)
    assert result.returncode != 0
    assert not output.exists() and not calls()


def test_job_discovery_requires_exact_unique_ledger(launcher):
    run, source, output, calls = launcher
    ledger = source / "submitted_job_ids.txt"
    ledger.write_text("infer 2075560\nother 207556\n")
    arguments = ("--source-job", "207556", "--search-root", shell_path(source.parent), "--dry-run")
    missing = run(*arguments)
    assert missing.returncode != 0 and "found 0" in missing.stderr
    ledger.write_text("infer 207556\n")
    found = run(*arguments)
    assert found.returncode == 0, found.stderr
    other = source.parent / "other"
    other.mkdir()
    (other / "submitted_job_ids.txt").write_text("infer 207556\n")
    ambiguous = run(*arguments)
    assert ambiguous.returncode != 0 and "found 2" in ambiguous.stderr
    assert not calls()

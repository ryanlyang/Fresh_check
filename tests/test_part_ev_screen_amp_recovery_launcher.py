"""No real scheduler calls: verify the frozen, diagnostic-only submission graph."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from tests.test_part_fullscale_launcher import BASH, shell_path

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not BASH, reason='Bash unavailable')


@pytest.fixture
def launcher(tmp_path):
    repo, campaign, fakebin = tmp_path/'repo', tmp_path/'campaign', tmp_path/'bin'
    for path in (repo/'scripts', repo/'sbatch', campaign/'source', fakebin):
        path.mkdir(parents=True)
    for name in ('submit_part_ev_screen_amp_recovery.sh', 'run_part_ev_screen_amp_recovery.sh'):
        shutil.copyfile(REPO/'sbatch'/name, repo/'sbatch'/name)
    shutil.copyfile(REPO/'scripts/check_part_ev_screen_amp_recovery.py', repo/'scripts/check_part_ev_screen_amp_recovery.py')
    calls = tmp_path/'calls'
    stubs = {
        'uname': 'echo "${TEST_ARCH:-x86_64}"\n',
        'flock': 'exit 0\n',
        'squeue': '[[ "${TEST_QUEUE_FAIL:-}" != 1 ]] || exit 1\nprintf "%s\\n" "${TEST_LIVE:-}"\n',
        'scontrol': 'echo partition=tier3\n',
        'sbatch': 'printf "%s\\n" "$*" >> "$TEST_CALLS"\necho "$((750000 + $(wc -l < "$TEST_CALLS")))"\n',
        'python': '''[[ "${TEST_PREPARE_FAIL:-}" != 1 ]] || exit 1
if [[ "$*" == *"--phase prepare"* ]]; then
  for ((i=1;i<=$#;i++)); do
    if [[ "${!i}" == --output ]]; then j=$((i+1)); diagnostic="${!j}"; fi
  done
  printf '{}\\n' > "$diagnostic/recovery.json"
else
  printf 'FROZEN %s\\n' "$*"
fi
''',
    }
    for name, body in stubs.items():
        path = fakebin/name
        path.write_text('#!/usr/bin/env bash\n'+body, newline='\n')
        path.chmod(0o755)
    (campaign/'launch.env').write_text(
        f'SCREEN_PYTHON="{shell_path(fakebin / "python")}"\n'
        'SCREEN_GPU=a100\nSCREEN_CPUS=8\nSCREEN_MEMORY=64G\n'
        'SCREEN_ACCOUNT=reu-aisocial\nSCREEN_PARTITION=tier3\n', newline='\n')
    (campaign/'launch.json').write_text('{}\n')
    (campaign/'campaign.json').write_text('{}\n')
    (campaign/'source/sentinel').write_text('original source\n')
    (campaign/'submitted_job_ids.tsv').write_text('benchmark\t123456\n', newline='\n')
    result = subprocess.run([BASH, '-c',
        'git init -q "$1" && git -C "$1" add . && git -C "$1" -c user.name=Test -c user.email=test@example.invalid '
        '-c commit.gpgsign=false commit -qm fixture && cd "$2/source" && sha256sum sentinel > ../source.sha256',
        'init', shell_path(repo), shell_path(campaign)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr

    def run(*args, env=None, worker=None):
        extra = dict(os.environ, TEST_CALLS=shell_path(calls), **(env or {}))
        if worker:
            script = worker/'run_part_ev_screen_amp_recovery.sh'
            options = [shell_path(worker), *args]
        else:
            script = repo/'sbatch/submit_part_ev_screen_amp_recovery.sh'
            options = ['--campaign', shell_path(campaign), *args]
        return subprocess.run([BASH, '--noprofile', '--norc', '-c',
            'export PATH="$1:$PATH"; shift; exec bash "$@"', 'test', shell_path(fakebin), shell_path(script), *options],
            env=extra, capture_output=True, text=True, timeout=40)
    return run, repo, campaign, calls


def test_dry_run_is_read_only(launcher):
    run, _, campaign, calls = launcher
    result = run('--dry-run')
    assert result.returncode == 0, result.stderr
    assert '--array=0-6%2' in result.stdout and '--time=01:00:00' in result.stdout
    assert not (campaign/'diagnostics').exists() and not calls.exists()


def test_frozen_short_array_then_afterany_summary(launcher):
    run, repo, campaign, calls = launcher
    result = run('--concurrency', '3', env={'SBATCH_TIME':'9-00:00:00'})
    assert result.returncode == 0, result.stderr
    rows = calls.read_text().splitlines()
    assert len(rows) == 2
    assert '--array=0-6%3' in rows[0] and '--gres=gpu:a100:1' in rows[0]
    assert '--time=01:00:00' in rows[0] and '--mem=64G' in rows[0]
    assert '--time=00:10:00' in rows[1] and '--dependency=afterany:750001' in rows[1]
    assert '--gres' not in rows[1] and '--array' not in rows[1]
    assert '--account=reu-aisocial' in rows[0] and '--partition=tier3' in rows[0]
    assert (campaign/'submitted_job_ids.tsv').read_text() == 'benchmark\t123456\n'
    output = next(p for p in (campaign/'diagnostics').glob('amp_recovery.*') if p.is_dir())
    (repo/'scripts/check_part_ev_screen_amp_recovery.py').write_text('# changed live code\n')
    worker = run('run', worker=output, env={'SLURM_ARRAY_TASK_ID':'4'})
    assert worker.returncode == 0, worker.stderr
    assert '--phase run' in worker.stdout and '--task-index 4' in worker.stdout
    assert run('run', worker=output, env={'SLURM_ARRAY_TASK_ID':'7'}).returncode != 0
    assert run('summary', worker=output).returncode == 0
    assert run('train', worker=output).returncode != 0
    (output/'check_part_ev_screen_amp_recovery.py').write_text('# tamper frozen code\n')
    assert run('summary', worker=output).returncode != 0


@pytest.mark.parametrize('env', [{'TEST_LIVE':'123456_[0-6%2]'}, {'TEST_ARCH':'aarch64'},
    {'TEST_PREPARE_FAIL':'1'}, {'TEST_QUEUE_FAIL':'1'}])
def test_no_submission_on_failed_preflight_or_live_original_jobs(launcher, env):
    run, _, _, calls = launcher
    assert run(env=env).returncode != 0
    assert not calls.exists()


def test_duplicate_recovery_pipeline_is_blocked(launcher):
    run, _, _, calls = launcher
    assert run().returncode == 0
    assert run(env={'TEST_LIVE':'750001_2'}).returncode != 0
    assert run(env={'TEST_LIVE':'750002'}).returncode != 0
    assert len(calls.read_text().splitlines()) == 2


def test_dirty_recovery_source_and_original_tampering_are_blocked(launcher):
    run, repo, campaign, calls = launcher
    (campaign/'source/sentinel').write_text('changed\n')
    assert run().returncode != 0
    assert not calls.exists()
    (repo/'scripts/check_part_ev_screen_amp_recovery.py').write_text('# dirty\n')
    assert run().returncode != 0
    assert not calls.exists()

"""Fake Slurm integration; no real submissions, environment changes or GPU use."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from part_selected_tigris import protocol as P
from tests.test_part_fullscale_launcher import BASH, shell_path
from tests.test_part_selected_tigris import benchmark_fixture
from tests.test_part_selected_tigris_launcher import launcher

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not BASH, reason='Bash unavailable')
FILES = ('scripts/recover_part_selected_tigris_walltime.py',
         'sbatch/submit_part_selected_tigris_recovery.sh', 'sbatch/run_part_selected_tigris_recovery.sh')


@pytest.fixture
def recovery_launcher(launcher):
    original_run, source, output, calls = launcher
    assert original_run('--mode', 'benchmark').returncode == 0
    original_launch = P.read_json(output/'launch.json')
    spec, rows = benchmark_fixture(output)
    spec.update(schema=P.SCHEMA, tasks=copy.deepcopy(P.TASKS), launch=original_launch)
    spec.pop('content_hash')
    spec = P.hashed(spec)
    P.write_json(output/'experiment_spec.json', spec)
    for task, row in zip(P.TASKS, rows):
        row.update(campaign_sha256=spec['content_hash'], projected_training_hours=114.66)
        row['runtime']['campaign_sha256'] = spec['content_hash']
        P.write_json(output/'benchmarks'/f"{task['run_id']}.json", row)
    for name in FILES:
        shutil.copyfile(REPO/name, source/name)
    fakebin = output.parent/'bin'
    (fakebin/'scontrol').write_text('#!/usr/bin/env bash\necho "PartitionName=tigris MaxTime=${TEST_MAXTIME:-7-00:00:00}"\n', newline='\n')
    (fakebin/'python').write_text('''#!/usr/bin/env bash
if [[ "$*" == *recover_part_selected_tigris_walltime.py* && "$*" != *"--phase configure"* ]]; then
  printf 'RECOVERY_WORKER %s\\n' "$*"
else
  exec "$TEST_PYTHON" "$@"
fi
''', newline='\n')
    commit = subprocess.run([BASH, '-c', 'git -C "$1" add . && git -C "$1" -c user.name=Test -c user.email=test@example.invalid -c commit.gpgsign=false commit -qm recovery',
        'commit', shell_path(source)], capture_output=True, text=True, timeout=30)
    assert commit.returncode == 0, commit.stderr
    recovery = output/'recoveries/walltime_168h_12h'

    def run(*args, env=None, worker=False):
        environ = dict(os.environ, TEST_CALLS=shell_path(calls), TEST_PYTHON=shell_path(sys.executable))
        environ.update(env or {})
        script = recovery/'source/sbatch/run_part_selected_tigris_recovery.sh' if worker else source/'sbatch/submit_part_selected_tigris_recovery.sh'
        defaults = [] if worker else ['--output', shell_path(output)]
        return subprocess.run([BASH, '--noprofile', '--norc', '-c',
            'export PATH="$1:$PATH"; shift; exec bash "$@"', 'test', shell_path(fakebin), shell_path(script), *defaults, *args],
            env=environ, capture_output=True, text=True, timeout=120)
    return run, source, output, recovery, calls, original_run


def test_recovery_queues_only_gate_training_lock_evaluation_report(recovery_launcher):
    run, _, output, recovery, calls, original_run = recovery_launcher
    before = {p: p.read_bytes() for p in (output/'launch.env', output/'launch.json', output/'experiment_spec.json',
        output/'source.sha256', *sorted((output/'benchmarks').glob('*.json')))}
    result = run('--concurrency', '4')
    assert result.returncode == 0, result.stdout+result.stderr
    rows = calls.read_text().splitlines()[3:]
    assert [r.split('|')[0] for r in rows] == ['gate', 'train', 'lock', 'evaluate', 'report']
    assert '--dependency' not in rows[0]
    for index in range(1, 5):
        assert f'--dependency=afterok:{750003+index}' in rows[index]
        assert '--kill-on-invalid-dep=yes' in rows[index]
    assert '--time=168:00:00' in rows[1]
    assert '--time=12:00:00' in rows[3]
    for index in (1, 3):
        assert '--array=0-3%4' in rows[index] and '--gpus-per-node=1' in rows[index]
        assert '--mem=128G' in rows[index] and '--cpus-per-task=8' in rows[index]
    for index in (0, 2, 4):
        assert '--time=2:00:00' in rows[index] and '--gpus' not in rows[index]
    assert len((recovery/'submitted_job_ids.tsv').read_text().splitlines()) == 5
    assert len((output/'submitted_job_ids.tsv').read_text().splitlines()) == 8
    assert before == {p: p.read_bytes() for p in before}
    assert len(list((recovery/'source').rglob('*.py'))) == 1  # No whole-repo/environment copy.
    assert original_run('--mode', 'train', '--resume', env={'TEST_LIVE': '750005_[0-3%4]'}).returncode != 0


def test_dry_run_does_not_write_or_submit(recovery_launcher):
    run, _, output, recovery, calls, _ = recovery_launcher
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    result = run('--dry-run')
    assert result.returncode == 0, result.stderr
    assert '149.06 h' in result.stdout
    assert not recovery.exists() and len(calls.read_text().splitlines()) == 3
    assert before == {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}


@pytest.mark.parametrize('env', [
    {'TEST_LIVE': '750002_[0-3%2]|selgh_benchmark'},
    {'TEST_LIVE': '484463|selgh_autostart_750003'},
    {'TEST_ARCH': 'x86_64'}, {'TEST_MAXTIME': '4-00:00:00'},
])
def test_active_jobs_wrong_platform_or_partition_limit_stop_submission(recovery_launcher, env):
    run, _, _, recovery, calls, _ = recovery_launcher
    result = run(env=env)
    assert result.returncode != 0
    assert not recovery.exists() and len(calls.read_text().splitlines()) == 3


def test_no_duplicates_and_frozen_limits_on_resume(recovery_launcher):
    run, _, output, recovery, calls, _ = recovery_launcher
    assert run().returncode == 0
    assert run().returncode != 0
    assert run('--resume', env={'TEST_LIVE': '750005_0|selgh_recover_train'}).returncode != 0
    # If interruption occurred between the two ledger appends, either ledger
    # is sufficient to prevent duplicate jobs.
    ledger = output/'submitted_job_ids.tsv'
    recorded = ledger.read_bytes()
    ledger.write_text('prepare\t750001\nbenchmark\t750002\ngate\t750003\n')
    assert run('--resume', env={'TEST_LIVE': '750005_0|selgh_recover_train'}).returncode != 0
    ledger.write_bytes(recorded)
    assert run('--resume', '--recovery', shell_path(recovery), '--train-hours', '170',
               env={'TEST_MAXTIME': 'UNLIMITED'}).returncode != 0
    assert len(calls.read_text().splitlines()) == 8
    result = run('--resume')
    assert result.returncode == 0, result.stderr
    assert len(calls.read_text().splitlines()) == 13


def test_evaluation_walltime_extension_is_submitted(recovery_launcher):
    run, _, output, _, calls, _ = recovery_launcher
    for path in (output/'benchmarks').glob('*.json'):
        row = P.read_json(path)
        row['validation']['seconds'] = 1000
        P.write_json(path, row)
    assert run().returncode != 0
    result = run('--evaluate-hours', '16')
    assert result.returncode == 0, result.stderr
    assert '--time=16:00:00' in calls.read_text().splitlines()[6]


def test_worker_uses_original_environment_and_checks_frozen_sources(recovery_launcher):
    run, source, output, recovery, _, _ = recovery_launcher
    assert run().returncode == 0
    (source/'scripts/recover_part_selected_tigris_walltime.py').write_text('# later pull\n')
    env = dict(PART_SELECTED_TIGRIS_OUTPUT=shell_path(output), SELECTED_WALLTIME_RECOVERY=shell_path(recovery),
               PART_SELECTED_TIGRIS_PHASE='train', SLURM_ARRAY_TASK_ID='2')
    result = run(worker=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'RECOVERY_WORKER' in result.stdout and '--task-index 2' in result.stdout
    assert run(worker=True, env=dict(env, PART_SELECTED_TIGRIS_PHASE='benchmark')).returncode != 0
    assert run(worker=True, env=dict(env, SLURM_ARRAY_TASK_ID='4')).returncode != 0
    helper = recovery/'source/scripts/recover_part_selected_tigris_walltime.py'
    old = helper.read_bytes()
    helper.write_text('# tampered helper\n')
    assert run(worker=True, env=env).returncode != 0
    helper.write_bytes(old)
    (output/'source/part_selected_screen/model.py').write_text('# tampered original\n')
    assert run(worker=True, env=env).returncode != 0


def test_dirty_tooling_and_bad_benchmark_refused(recovery_launcher):
    run, source, output, recovery, calls, _ = recovery_launcher
    path = source/'scripts/recover_part_selected_tigris_walltime.py'
    old = path.read_bytes()
    path.write_bytes(old+b'\n# dirty\n')
    assert run().returncode != 0
    assert not recovery.exists()
    path.write_bytes(old)
    row = P.read_json(output/'benchmarks/SELECTED_EV_LAYERWISE.json')
    row['passed'] = False
    P.write_json(output/'benchmarks/SELECTED_EV_LAYERWISE.json', row)
    assert run().returncode != 0
    assert len(calls.read_text().splitlines()) == 3

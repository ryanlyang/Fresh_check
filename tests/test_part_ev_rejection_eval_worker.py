"""Simulate Slurm copying the worker outside its source tree; no real jobs."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.test_part_fullscale_launcher import BASH, shell_path

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not BASH, reason='Bash unavailable')


@pytest.fixture
def worker(tmp_path):
    spool = tmp_path / 'spool/job123/slurm_script'
    spool.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / 'sbatch/run_part_ev_rejection_eval.sh', spool)
    campaign = tmp_path / 'evaluation with spaces'
    driver = campaign / 'source/scripts/run_part_ev_rejection_eval.py'
    driver.parent.mkdir(parents=True)
    driver.write_text('# fixture; fake Python validates path and arguments\n')
    (campaign / 'plan.json').write_text('{}\n')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    python = bin_dir / 'python3'
    python.write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == -B && -f "$2" ]] || exit 97
printf '%s\\n' "$@"
printf 'BYTECODE=%s USER_SITE=%s\\n' "$PYTHONDONTWRITEBYTECODE" "$PYTHONNOUSERSITE"
''', newline='\n')
    python.chmod(0o755)

    def run(*args):
        return subprocess.run([BASH, '--noprofile', '--norc', '-c',
            'export PATH="$1:$PATH"; shift; exec bash "$@"', 'fixture',
            shell_path(bin_dir), shell_path(spool), *args],
            cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True, timeout=20)
    return run, campaign, driver


@pytest.mark.parametrize('phase,model', [('worker', 'part'), ('worker', 'state'), ('report', None)])
def test_relocated_slurm_script_uses_explicit_frozen_campaign(worker, phase, model):
    run, campaign, driver = worker
    args = ['--phase', phase, '--output', shell_path(campaign)]
    if model:
        args += ['--model', model]
    result = run(*args)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['-B', shell_path(driver), *args,
                                          'BYTECODE=1 USER_SITE=1']


@pytest.mark.parametrize('args', [[], ['--output'], ['--output', 'relative'],
                                ['--output', '/one', '--output', '/two']])
def test_invalid_campaign_argument_fails_before_python(worker, args):
    run, _, _ = worker
    result = run(*args)
    assert result.returncode == 2
    assert not result.stdout


def test_missing_frozen_driver_fails_before_python(worker):
    run, campaign, driver = worker
    driver.unlink()
    result = run('--phase', 'worker', '--output', shell_path(campaign), '--model', 'part')
    assert result.returncode == 2
    assert 'Missing evaluation plan or frozen driver' in result.stderr

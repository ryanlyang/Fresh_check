"""Fake-scheduler tests only: no real SSH, Slurm or cluster mutation."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from tests.test_part_fullscale_launcher import BASH, shell_path

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not BASH,reason='Bash unavailable')


@pytest.fixture
def launcher(tmp_path):
    source,output,fakebin = tmp_path/'source repo',tmp_path/'campaign',tmp_path/'bin'
    for p in (source/'sbatch',source/'scripts',source/'part_ev_screen',fakebin): p.mkdir(parents=True)
    for name in ('submit_part_ev_screen_sporc.sh','run_part_ev_screen_sporc.sh'):
        shutil.copyfile(REPO/'sbatch'/name,source/'sbatch'/name)
    (source/'scripts/run_part_ev_screen.py').write_text('# fixture\n')
    (source/'part_ev_screen/model.py').write_text('# fixture\n')
    data = tmp_path/'data'
    (data/'Pythia/train_100M').mkdir(parents=True)
    (data/'Pythia/val_5M').mkdir()
    calls = tmp_path/'calls.txt'
    stubs = {
        'flock':'exit 0\n',
        'uname':'echo "${TEST_ARCH:-x86_64}"\n',
        'squeue':'printf "%s\\n" "${TEST_LIVE:-}"\n',
        'scontrol':'[[ "${TEST_PARTITION_BAD:-}" != 1 ]] || exit 1\necho PartitionName=tier3\n',
        'sbatch': 'printf "%s|%s\\n" "$PART_EV_SCREEN_PHASE" "$*" >> "$TEST_CALLS"\necho "$((740000 + $(wc -l < "$TEST_CALLS")))"\n',
        'python': '''if [[ "$*" == *run_part_ev_screen.py* ]]; then
  if [[ "$*" == *review* ]]; then
    [[ "${TEST_BENCHMARK_BAD:-}" != 1 ]] || exit 1
    echo '{"recommended_train_hours":11,"models":[{"validation":{"seconds":100,"events":128000}}]}'
  else
    printf 'WORKER %s\\n' "$*"
  fi
else
  exec "$TEST_PYTHON" "$@"
fi
''',
    }
    for name,body in stubs.items():
        path = fakebin/name
        path.write_text('#!/usr/bin/env bash\n'+body,newline='\n'); path.chmod(0o755)
    initialized = subprocess.run([BASH,'-c','git init -q "$1" && git -C "$1" add . && git -C "$1" -c user.name=Test -c user.email=test@example.invalid -c commit.gpgsign=false commit -qm fixture',
        'init',shell_path(source)],capture_output=True,text=True,timeout=30)
    assert initialized.returncode == 0,initialized.stderr
    def run(*args,env=None,worker=False):
        extra = dict(os.environ, TEST_CALLS=shell_path(calls), TEST_PYTHON=shell_path(sys.executable))
        extra.update(env or {})
        script = output/'source/sbatch/run_part_ev_screen_sporc.sh' if worker else source/'sbatch/submit_part_ev_screen_sporc.sh'
        defaults = [] if worker else ['--output',shell_path(output),'--python',shell_path(fakebin/'python'),'--data-root',shell_path(data)]
        return subprocess.run([BASH,'--noprofile','--norc','-c','export PATH="$1:$PATH"; shift; exec bash "$@"','test',
            shell_path(fakebin),shell_path(script),*defaults,*args],env=extra,capture_output=True,text=True,timeout=40)
    return run,source,output,calls


def test_dry_run_no_files_or_jobs(launcher):
    run,_,output,calls = launcher
    result = run('--mode','prepare','--dry-run')
    assert result.returncode == 0,result.stderr
    assert '--array=0-6%2' in result.stdout and '--gres=gpu:a100:1' in result.stdout
    assert not output.exists() and not calls.exists()


def test_prepare_and_training_dependency_graph(launcher):
    run,source,output,calls = launcher
    result = run('--mode','prepare')
    assert result.returncode == 0,result.stderr
    rows = calls.read_text().splitlines()
    assert len(rows) == 2
    assert rows[0].startswith('prepare|') and '--gres' not in rows[0]
    assert rows[1].startswith('benchmark|') and '--dependency=afterok:740001' in rows[1]
    assert all('--time=2:00:00' in row for row in rows)
    assert (output/'source/part_ev_screen/model.py').read_text() == '# fixture\n'
    (source/'part_ev_screen/model.py').write_text('# changed after prepare\n')
    result = run('--mode','train','--reviewed')
    assert result.returncode == 0,result.stderr
    rows = calls.read_text().splitlines()
    assert len(rows) == 5
    assert '--time=11:00:00' in rows[2]
    assert rows[3].startswith('evaluate|') and '--dependency=afterok:740003' in rows[3]
    assert rows[4].startswith('report|') and '--dependency=afterok:740004' in rows[4]
    assert all('--cluster' not in r for r in rows)
    assert all('--array=0-6%2' in rows[i] for i in (1,2,3))


@pytest.mark.parametrize('options,env', [([],{}),(['--reviewed'],{'TEST_BENCHMARK_BAD':'1'}),
    (['--reviewed','--train-hours','10'],{}), (['--reviewed'],{'TEST_LIVE':'740002_3'})])
def test_training_gates(launcher,options,env):
    run,_,_,calls = launcher
    assert run('--mode','prepare').returncode == 0
    result = run('--mode','train',*options,env=env)
    assert result.returncode != 0
    assert len(calls.read_text().splitlines()) == 2


def test_resume_needs_terminal_jobs_and_explicit_flag(launcher):
    run,_,_,calls = launcher
    assert run('--mode','prepare').returncode == 0
    assert run('--mode','benchmark').returncode != 0
    assert run('--mode','benchmark','--resume',env={'TEST_LIVE':'740002_[3-6%2]'}).returncode != 0
    assert len(calls.read_text().splitlines()) == 2
    result = run('--mode','benchmark','--resume')
    assert result.returncode == 0,result.stderr
    assert len(calls.read_text().splitlines()) == 3


def test_worker_uses_frozen_paths_and_x86_guard(launcher):
    run,_,output,_ = launcher
    assert run('--mode','prepare').returncode == 0
    env = dict(PART_EV_SCREEN_OUTPUT=shell_path(output),PART_EV_SCREEN_PHASE='train',SLURM_ARRAY_TASK_ID='4')
    result = run(worker=True,env=env)
    assert result.returncode == 0,result.stderr
    assert '--phase train' in result.stdout and '--task-index 4' in result.stdout
    assert run(worker=True,env=dict(env,TEST_ARCH='aarch64')).returncode != 0
    assert run(worker=True,env=dict(env,SLURM_ARRAY_TASK_ID='7')).returncode != 0


def test_source_tampering_and_invalid_partition_fail(launcher):
    run,_,output,calls = launcher
    assert run('--mode','prepare').returncode == 0
    assert run('--mode','train','--reviewed',env={'TEST_PARTITION_BAD':'1'}).returncode != 0
    (output/'source/part_ev_screen/model.py').write_text('# tampered\n')
    env = dict(PART_EV_SCREEN_OUTPUT=shell_path(output),PART_EV_SCREEN_PHASE='benchmark',SLURM_ARRAY_TASK_ID='0')
    assert run(worker=True,env=env).returncode != 0
    assert len(calls.read_text().splitlines()) == 2


def test_resource_overrides_never_silently_change_existing_campaign(launcher):
    run,_,_,calls = launcher
    assert run('--mode','prepare').returncode == 0
    result = run('--mode','benchmark','--resume','--mem','128G')
    assert result.returncode != 0 and 'frozen campaign' in result.stderr
    assert len(calls.read_text().splitlines()) == 2

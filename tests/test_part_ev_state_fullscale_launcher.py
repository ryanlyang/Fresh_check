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
    for p in (source/'sbatch',source/'scripts',source/'part_ev_state_fullscale',fakebin): p.mkdir(parents=True)
    for name in ('submit_part_ev_state_fullscale.sh','run_part_ev_state_fullscale.sh'):
        shutil.copyfile(REPO/'sbatch'/name,source/'sbatch'/name)
    (source/'scripts/run_part_ev_state_fullscale.py').write_text('# fixture\n')
    for name in ('model.py','engine.py','protocol.py','reporting.py'):
        (source/'part_ev_state_fullscale'/name).write_text('# fixture\n')
    data = tmp_path/'data'
    (data/'Pythia/train_100M').mkdir(parents=True)
    (data/'Pythia/val_5M').mkdir()
    calls = tmp_path/'calls.txt'
    stubs = {
        'flock':'exit 0\n',
        'uname':'echo "${TEST_ARCH:-x86_64}"\n',
        'squeue':'printf "%s\\n" "${TEST_LIVE:-}"\n',
        'scontrol':'[[ "${TEST_PARTITION_BAD:-}" != 1 ]] || exit 1\necho "PartitionName=tier3 MaxTime=${TEST_MAXTIME:-14-00:00:00}"\n',
        'sbatch': 'printf "%s|%s\\n" "$PART_EV_STATE_FULLSCALE_PHASE" "$*" >> "$TEST_CALLS"\necho "$((740000 + $(wc -l < "$TEST_CALLS")))"\n',
        'python': '''if [[ "$*" == *run_part_ev_state_fullscale.py* ]]; then
  if [[ "$*" == *review* ]]; then
    [[ "${TEST_BENCHMARK_BAD:-}" != 1 ]] || exit 1
    echo '{"recommended_train_hours":11,"recommended_evaluate_hours":5,"models":[{"validation":{"seconds":100,"events":128000}}]}'
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
        script = output/'source/sbatch/run_part_ev_state_fullscale.sh' if worker else source/'sbatch/submit_part_ev_state_fullscale.sh'
        defaults = [] if worker else ['--output',shell_path(output),'--python',shell_path(fakebin/'python'),'--data-root',shell_path(data)]
        return subprocess.run([BASH,'--noprofile','--norc','-c','export PATH="$1:$PATH"; shift; exec bash "$@"','test',
            shell_path(fakebin),shell_path(script),*defaults,*args],env=extra,capture_output=True,text=True,timeout=40)
    return run,source,output,calls


def test_dry_run_no_files_or_jobs(launcher):
    run,_,output,calls = launcher
    result = run('--mode','prepare','--dry-run')
    assert result.returncode == 0,result.stderr
    assert '--array=0-6%3' in result.stdout and '--gres=gpu:a100:1' in result.stdout
    assert not output.exists() and not calls.exists()


def test_prepare_and_training_dependency_graph(launcher):
    run,source,output,calls = launcher
    result = run('--mode','prepare')
    assert result.returncode == 0,result.stderr
    rows = calls.read_text().splitlines()
    assert len(rows) == 2
    assert rows[0].startswith('prepare|') and '--gres' not in rows[0]
    assert rows[1].startswith('benchmark|') and '--dependency=afterok:740001' in rows[1]
    assert all('--time=5:00:00' in row for row in rows)
    assert (output/'source/part_ev_state_fullscale/model.py').read_text() == '# fixture\n'
    (source/'part_ev_state_fullscale/model.py').write_text('# changed after prepare\n')
    result = run('--mode','train','--reviewed')
    assert result.returncode == 0,result.stderr
    rows = calls.read_text().splitlines()
    assert len(rows) == 5
    assert '--time=11:00:00' in rows[2]
    assert rows[3].startswith('evaluate|') and '--dependency=afterok:740003' in rows[3]
    assert rows[4].startswith('report|') and '--dependency=afterok:740004' in rows[4]
    assert all('--cluster' not in r for r in rows)
    assert '--array=0,3,6%3' in rows[1]
    assert all('--array=0-6%3' in rows[i] for i in (2,3))


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
    env = dict(PART_EV_STATE_FULLSCALE_OUTPUT=shell_path(output),PART_EV_STATE_FULLSCALE_PHASE='train',SLURM_ARRAY_TASK_ID='2')
    result = run(worker=True,env=env)
    assert result.returncode == 0,result.stderr
    assert '--phase train' in result.stdout and '--task-index 2' in result.stdout
    assert run(worker=True,env=dict(env,TEST_ARCH='aarch64')).returncode != 0
    assert run(worker=True,env=dict(env,SLURM_ARRAY_TASK_ID='7')).returncode != 0


def test_source_tampering_and_invalid_partition_fail(launcher):
    run,_,output,calls = launcher
    assert run('--mode','prepare').returncode == 0
    assert run('--mode','train','--reviewed',env={'TEST_PARTITION_BAD':'1'}).returncode != 0
    (output/'source/part_ev_state_fullscale/model.py').write_text('# tampered\n')
    env = dict(PART_EV_STATE_FULLSCALE_OUTPUT=shell_path(output),PART_EV_STATE_FULLSCALE_PHASE='benchmark',SLURM_ARRAY_TASK_ID='0')
    assert run(worker=True,env=env).returncode != 0
    assert len(calls.read_text().splitlines()) == 2


def test_resource_overrides_never_silently_change_existing_campaign(launcher):
    run,_,_,calls = launcher
    assert run('--mode','prepare').returncode == 0
    result = run('--mode','benchmark','--resume','--mem','128G')
    assert result.returncode != 0 and 'frozen campaign' in result.stderr
    assert len(calls.read_text().splitlines()) == 2


def test_seven_new_runs_and_review_approval_flags(launcher):
    run,_,output,calls = launcher
    assert run('--mode','prepare').returncode == 0
    assert run('--mode','train','--reviewed').returncode == 0
    rows = calls.read_text()
    assert '--array=0-6%3' in rows
    assert 'PART_RANGER' not in rows and 'EV_RANGER' not in rows
    assert '--partition=tier3' in rows
    assert '--array=0-4' not in rows
    assert '--array=0,3,6%3' in rows
    worker = (output/'source/sbatch/run_part_ev_state_fullscale.sh').read_text()
    assert 'run_part_ev_state_fullscale.py' in worker

def test_environment_override_is_frozen(launcher):
    run,_,_,calls = launcher
    assert run('--mode','prepare').returncode == 0
    assert run('--mode','benchmark','--resume','--python','/different/python').returncode != 0
    assert len(calls.read_text().splitlines()) == 2

def test_launch_env_tampering_is_rejected_before_submission(launcher):
    run,_,output,calls = launcher
    assert run('--mode','prepare').returncode == 0
    with (output/'launch.env').open('a') as stream:
        stream.write('\nSTATE_FULLSCALE_PARTITION=debug\n')
    assert run('--mode','benchmark','--resume').returncode != 0
    assert len(calls.read_text().splitlines()) == 2


@pytest.mark.parametrize('limit',['01:00:00','1-00:00:00'])
def test_partition_walltime_is_checked(launcher,limit):
    run,_,_,calls = launcher
    if limit == '01:00:00':
        assert run('--mode','prepare',env={'TEST_MAXTIME':limit}).returncode != 0
        assert not calls.exists()
    else:
        assert run('--mode','prepare').returncode == 0
        result = run('--mode','train','--reviewed','--train-hours','216',env={'TEST_MAXTIME':limit})
        assert result.returncode != 0 and 'exceeds partition MaxTime' in result.stderr
        assert len(calls.read_text().splitlines()) == 2


def test_eval_override_and_7_way_concurrency(launcher):
    run,_,_,calls = launcher
    assert run('--mode','prepare','--concurrency','7').returncode == 0
    assert '--array=0,3,6%3' in calls.read_text()
    result = run('--mode','train','--reviewed','--train-hours','216','--evaluate-hours','8','--concurrency','7')
    assert result.returncode == 0,result.stderr
    rows = calls.read_text().splitlines()
    assert '--time=216:00:00' in rows[2] and '--array=0-6%7' in rows[2]
    assert '--time=8:00:00' in rows[3]
    assert '--signal=B:USR1@180' in rows[2]


def test_eval_override_cannot_undercut_recommendation(launcher):
    run,_,_,calls = launcher
    assert run('--mode','prepare').returncode == 0
    assert run('--mode','train','--reviewed','--evaluate-hours','2').returncode != 0
    assert len(calls.read_text().splitlines()) == 2


def test_readonly_review_allowed_while_jobs_active(launcher):
    run,_,_,calls = launcher
    assert run('--mode','prepare').returncode == 0
    result = run('--mode','review',env={'TEST_LIVE':'740002_0'})
    assert result.returncode == 0,result.stderr
    assert len(calls.read_text().splitlines()) == 2


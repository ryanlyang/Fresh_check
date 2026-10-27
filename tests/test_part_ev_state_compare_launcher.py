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
    for p in (source/'sbatch',source/'scripts',source/'part_ev_state_compare',fakebin): p.mkdir(parents=True)
    for name in ('submit_part_ev_state_compare.sh','run_part_ev_state_compare.sh'):
        shutil.copyfile(REPO/'sbatch'/name,source/'sbatch'/name)
    (source/'scripts/run_part_ev_state_compare.py').write_text('# fixture\n')
    for name in ('model.py','engine.py','protocol.py','reporting.py'):
        (source/'part_ev_state_compare'/name).write_text('# fixture\n')
    (source/'scripts/part_ev_state_compare_dependencies.py').write_text('# fixture\n')
    controls = tmp_path/'old'
    controls.mkdir()
    data = tmp_path/'data'
    (data/'Pythia/train_100M').mkdir(parents=True)
    (data/'Pythia/val_5M').mkdir()
    calls = tmp_path/'calls.txt'
    stubs = {
        'flock':'exit 0\n',
        'uname':'echo "${TEST_ARCH:-x86_64}"\n',
        'squeue':'printf "%s\\n" "${TEST_LIVE:-}"\n',
        'scontrol':'[[ "${TEST_PARTITION_BAD:-}" != 1 ]] || exit 1\necho "PartitionName=tier3 MaxTime=${TEST_MAXTIME:-14-00:00:00}"\n',
        'sbatch': 'printf "%s|%s\\n" "$PART_EV_STATE_COMPARE_PHASE" "$*" >> "$TEST_CALLS"\necho "$((740000 + $(wc -l < "$TEST_CALLS")))"\n',
        'python': '''if [[ "$*" == *part_ev_state_compare_dependencies.py* ]]; then
  [[ "${TEST_SWITCH_BAD:-}" != 1 ]] || exit 1
  echo '21964940_0:21964940_6'
elif [[ "$*" == *run_part_ev_state_compare.py* ]]; then
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
        script = output/'source/sbatch/run_part_ev_state_compare.sh' if worker else source/'sbatch/submit_part_ev_state_compare.sh'
        defaults = [] if worker else ['--output',shell_path(output),'--python',shell_path(fakebin/'python'),'--data-root',shell_path(data),'--controls',shell_path(controls)]
        return subprocess.run([BASH,'--noprofile','--norc','-c','export PATH="$1:$PATH"; shift; exec bash "$@"','test',
            shell_path(fakebin),shell_path(script),*defaults,*args],env=extra,capture_output=True,text=True,timeout=40)
    return run,source,output,calls


def test_dry_run_has_no_effects(launcher):
    run,_,output,calls=launcher
    result=run('--mode','prepare','--dry-run')
    assert result.returncode==0,result.stderr
    assert 'Train[3]' in result.stdout and 'evaluate[5]' in result.stdout
    assert not output.exists() and not calls.exists()


def test_three_new_trainings_five_evaluations_individual_kept_dependencies(launcher):
    run,source,output,calls=launcher
    result=run('--mode','prepare')
    assert result.returncode==0,result.stderr
    rows=calls.read_text().splitlines()
    assert len(rows)==2 and all('--time=5:00:00' in r for r in rows)
    assert '--array=0-2%2' in rows[1] and '--dependency=afterok:740001' in rows[1]
    (source/'part_ev_state_compare/model.py').write_text('# changed later\n')
    result=run('--mode','train','--reviewed','--concurrency','3')
    assert result.returncode==0,result.stderr
    rows=calls.read_text().splitlines()
    assert len(rows)==6
    assert rows[2].startswith('train|') and '--array=0-2%3' in rows[2]
    assert '--time=11:00:00' in rows[2] and '--dependency' not in rows[2]
    assert rows[3].startswith('lock|') and '--gres' not in rows[3]
    assert '--dependency=afterok:740003:21964940_0:21964940_6 ' in rows[3]
    assert rows[4].startswith('evaluate|') and '--array=0-4%3' in rows[4]
    assert '--dependency=afterok:740004' in rows[4]
    assert rows[5].startswith('report|') and '--dependency=afterok:740005' in rows[5]
    assert all('scancel' not in r and '21964940_3' not in r for r in rows)
    assert (output/'source/part_ev_state_compare/model.py').read_text()=='# fixture\n'


@pytest.mark.parametrize('options,env',[
    ([],{}),(['--reviewed'],{'TEST_SWITCH_BAD':'1'}),(['--reviewed'],{'TEST_BENCHMARK_BAD':'1'}),
    (['--reviewed'],{'TEST_LIVE':'740002_3'}),(['--reviewed','--train-hours','10'],{}),
    (['--reviewed','--evaluate-hours','2'],{}),(['--reviewed','--mem','128G'],{}),
    (['--reviewed','--train-hours','400'],{'TEST_MAXTIME':'14-00:00:00'})])
def test_submission_guards(launcher,options,env):
    run,_,_,calls=launcher
    assert run('--mode','prepare').returncode==0
    result=run('--mode','train',*options,env=env)
    assert result.returncode!=0
    assert len(calls.read_text().splitlines())==2


def test_retry_and_frozen_worker(launcher):
    run,_,output,calls=launcher
    assert run('--mode','prepare').returncode==0
    assert run('--mode','benchmark').returncode!=0
    assert run('--mode','benchmark','--resume',env={'TEST_LIVE':'740002_[0-2%2]'}).returncode!=0
    assert run('--mode','benchmark','--resume').returncode==0
    env=dict(PART_EV_STATE_COMPARE_OUTPUT=shell_path(output),PART_EV_STATE_COMPARE_PHASE='evaluate',SLURM_ARRAY_TASK_ID='4')
    result=run(worker=True,env=env)
    assert result.returncode==0,result.stderr
    assert '--task-index 4' in result.stdout
    assert run(worker=True,env=dict(env,TEST_ARCH='aarch64')).returncode!=0
    (output/'source/part_ev_state_compare/model.py').write_text('# changed\n')
    assert run(worker=True,env=env).returncode!=0


def test_old_controls_override_rejected(launcher):
    run,_,_,calls=launcher
    assert run('--mode','prepare').returncode==0
    assert run('--mode','benchmark','--resume','--controls','/different').returncode!=0
    assert len(calls.read_text().splitlines())==2


def test_worker_cannot_train_reused_controls(launcher):
    run,_,output,_=launcher
    assert run('--mode','prepare').returncode==0
    for index in ('3','4','5'):
        result=run(worker=True,env=dict(PART_EV_STATE_COMPARE_OUTPUT=shell_path(output),
            PART_EV_STATE_COMPARE_PHASE='train',SLURM_ARRAY_TASK_ID=index))
        assert result.returncode != 0


def test_reject_old_six_task_concurrency(launcher):
    run,_,_,calls=launcher
    assert run('--mode','prepare','--concurrency','6').returncode != 0
    assert not calls.exists()

"""Fake scheduler only; never requires an RC login or submits real jobs."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from tests.test_part_fullscale_launcher import BASH, shell_path

REPO=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not BASH,reason='Bash unavailable')


@pytest.fixture
def launcher(tmp_path):
    source,output,parent,bin_dir=(tmp_path/n for n in ('repo with space','diagnostic','old campaign','bin'))
    for p in (source/'sbatch',source/'scripts',source/'part_ev_gpu_scaling',parent,bin_dir):
        p.mkdir(parents=True)
    (parent/'campaign.json').write_text('{"frozen":true}\n')
    for name in ('submit_part_ev_gpu_scaling.sh','run_part_ev_gpu_scaling.sh'):
        shutil.copyfile(REPO/'sbatch'/name,source/'sbatch'/name)
    (source/'scripts/benchmark_part_ev_gpu_scaling.py').write_text('# driver fixture\n')
    for name in ('protocol.py','benchmark.py'):
        (source/'part_ev_gpu_scaling'/name).write_text('# fixture\n')
    calls=tmp_path/'calls.txt'
    stubs=dict(
        uname='echo "${TEST_ARCH:-x86_64}"\n',
        scontrol='echo "PartitionName=tier3 MaxTime=${TEST_MAXTIME:-14-00:00:00}"\n',
        sbatch='printf "%s|%s|%s\\n" "$EVSCALE_GPUS" "$EVSCALE_OUTPUT" "$*" >> "$TEST_CALLS"\necho "$((100+$(wc -l < "$TEST_CALLS")))"\n',
        python='''printf 'PYTHON %s\n' "$*"
printf 'RANK=%s WORLD=%s OMP=%s\n' "${RANK:-}" "${WORLD_SIZE:-}" "${OMP_NUM_THREADS:-}"
if [[ "$*" == *'--phase prepare'* && "${TEST_PREPARE_FAIL:-}" == 1 ]]; then exit 1; fi
''')
    for name,body in stubs.items():
        p=bin_dir/name; p.write_text('#!/usr/bin/env bash\n'+body,newline='\n');p.chmod(0o755)
    initialized=subprocess.run([BASH,'-c','git init -q "$1" && git -C "$1" add . && git -C "$1" -c user.name=Test -c user.email=test@example.invalid -c commit.gpgsign=false commit -qm fixture','init',shell_path(source)],capture_output=True,text=True,timeout=30)
    assert initialized.returncode==0,initialized.stderr
    def run(*args,env=None,worker=False):
        environment=dict(os.environ,TEST_CALLS=shell_path(calls))
        environment.update(env or {})
        script=(output/'source' if worker else source)/'sbatch'/('run_part_ev_gpu_scaling.sh' if worker else 'submit_part_ev_gpu_scaling.sh')
        defaults=[] if worker else ['--output',shell_path(output),'--campaign',shell_path(parent),'--python',shell_path(bin_dir/'python')]
        return subprocess.run([BASH,'--noprofile','--norc','-c','export PATH="$1:$PATH"; shift; exec bash "$@"','test',shell_path(bin_dir),shell_path(script),*defaults,*args],
                              env=environment,capture_output=True,text=True,timeout=40)
    return run,source,output,parent,calls


def test_dry_run_and_help_no_writes(launcher):
    run,_,output,_,calls=launcher
    r=run('--dry-run')
    assert r.returncode==0,r.stderr
    assert '--gres=gpu:a100:1' in r.stdout and '--gres=gpu:a100:2' in r.stdout
    assert '--nodes=1' in r.stdout and '04:30:00' in r.stdout
    assert not output.exists() and not calls.exists()
    assert run('--help').returncode==0


def test_two_short_jobs_same_node_afterany_and_frozen_source(launcher):
    run,source,output,parent,calls=launcher
    before=(parent/'campaign.json').read_bytes()
    r=run()
    assert r.returncode==0,r.stderr
    lines=calls.read_text().splitlines()
    assert len(lines)==2
    for world,line in zip((1,2),lines):
        assert f'--gres=gpu:a100:{world}' in line
        assert '--nodes=1' in line and '--ntasks=1' in line
        assert '--cpus-per-task=8' in line and '--mem=64G' in line
        assert '--time=04:30:00' in line
        assert '--array' not in line and 'scancel' not in line
    assert '--dependency' not in lines[0]
    assert '--dependency=afterany:101' in lines[1]
    assert (parent/'campaign.json').read_bytes()==before
    assert len(list(parent.iterdir()))==1
    (source/'part_ev_gpu_scaling/benchmark.py').write_text('# changed locally\n')
    env=dict(EVSCALE_OUTPUT=shell_path(output),EVSCALE_GPUS='2',WORLD_SIZE='99',RANK='17')
    worker=run(worker=True,env=env)
    assert worker.returncode==0,worker.stderr
    assert '--nproc-per-node=2' in worker.stdout and '--standalone --nnodes=1' in worker.stdout
    assert 'RANK= WORLD= OMP=2' in worker.stdout
    assert (output/'source/part_ev_gpu_scaling/benchmark.py').read_text()=='# fixture\n'
    assert run('--mode','review').returncode==0
    assert len(calls.read_text().splitlines())==2


@pytest.mark.parametrize('args,env',[
    ([],{'TEST_ARCH':'aarch64'}),([],{'TEST_MAXTIME':'04:00:00'}),
    (['--mode','train'],{}),(['--partition','bad name'],{}),([],{'TEST_PREPARE_FAIL':'1'})])
def test_invalid_requests_never_submit(launcher,args,env):
    run,_,_,_,calls=launcher
    assert run(*args,env=env).returncode!=0
    assert not calls.exists()


def test_refuse_overwrite_inside_campaign_or_changed_snapshot(launcher):
    run,_,output,parent,calls=launcher
    assert run('--output',shell_path(parent/'nested')).returncode!=0
    assert not calls.exists() and not (parent/'nested').exists()
    assert run().returncode==0
    assert run().returncode!=0
    assert len(calls.read_text().splitlines())==2
    (output/'source/part_ev_gpu_scaling/benchmark.py').write_text('# modified\n')
    assert run('--mode','review').returncode!=0
    assert run(worker=True,env=dict(EVSCALE_OUTPUT=shell_path(output),EVSCALE_GPUS='2')).returncode!=0


def test_uncommitted_relevant_source_refused(launcher):
    run,source,_,_,calls=launcher
    (source/'part_ev_gpu_scaling/benchmark.py').write_text('# uncommitted\n')
    assert run().returncode!=0
    assert not calls.exists()

"""Temporary git repos and fake Slurm. Never submits a real job."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import submit_part_ev_support as S
from tests.test_part_fullscale_launcher import BASH, shell_path


@pytest.fixture
def launcher(tmp_path,monkeypatch):
    repo=tmp_path/'repo with spaces';repo.mkdir()
    donor=tmp_path/'immutable donor';donor.mkdir()
    (donor/'campaign.json').write_text('{"keep":"unchanged"}')
    for rel in ('sbatch/run_part_ev_support.sh','scripts/run_part_ev_support.py','part_ev_support/model.py'):
        path=repo/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
    for args in (['init','-q'],['add','.'],['-c','user.name=Test','-c','user.email=test@example.invalid',
                  '-c','commit.gpgsign=false','commit','-qm','fixture']):
        subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
    monkeypatch.setattr(S,'ROOT',repo)
    monkeypatch.setattr(S.platform,'system',lambda:'Linux')
    monkeypatch.setattr(S.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(S.P,'reference',lambda *a:None)
    monkeypatch.setattr(S.P,'check_storage',lambda *a:8*2**30)
    monkeypatch.setattr(S.shutil,'which',lambda name:name)
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2))
    output=repo/'checkpoints/part_ev_support/new'
    a=S.parse(['--mode','benchmark','--output',str(output),'--reference',str(donor),'--python',sys.executable])
    calls=[];original=S.command;behavior=dict(reply=None,partition='PartitionName=tier3 State=UP MaxTime=20-00:00:00')
    def fake(args,**kw):
        args=[str(x) for x in args]
        if args[0]=='git':return original(args,**kw)
        calls.append((args,kw))
        if args[0]=='scontrol':return behavior['partition']
        assert args[0]=='sbatch'
        if '--test-only' in args:return 'valid'
        return behavior['reply'] or str(100+len([r for r,_ in calls if r[0]=='sbatch' and '--test-only' not in r]))
    monkeypatch.setattr(S,'command',fake)
    return a,repo,donor,calls,behavior


def actual_jobs(calls):return [r for r,_ in calls if r[0]=='sbatch' and '--test-only' not in r]


def test_independent_chain_and_frozen_snapshot(launcher):
    a,repo,donor,calls,_=launcher
    before=(donor/'campaign.json').read_bytes()
    S.launch(a);jobs=actual_jobs(calls)
    assert len(jobs)==3
    assert '--time=120' in jobs[0] and not any('--gres' in s for s in jobs[0])
    assert '--time=330' in jobs[1] and '--gres=gpu:a100:2' in jobs[1]
    assert '--nodes=1' in jobs[1] and '--ntasks=1' in jobs[1]
    assert '--cpus-per-task=8' in jobs[1] and '--mem=64G' in jobs[1]
    assert '--dependency=afterok:101' in jobs[1] and '--dependency=afterany:102' in jobs[2]
    assert '--time=15' in jobs[2]
    assert (donor/'campaign.json').read_bytes()==before
    assert (a.output/'submitted_job_ids.tsv').read_text().splitlines()==['prepare\t101','benchmark\t102','summary\t103']
    snapshot=a.output/'source/part_ev_support/model.py'
    (repo/'part_ev_support/model.py').write_text('later git pull')
    assert snapshot.read_text()=='# fixture\n'
    assert S.P.launch_config(a.output)['authorizes_training'] is False
    snapshot.write_text('unexpected change')
    with pytest.raises(ValueError,match='source'):S.P.launch_config(a.output)


def test_plan_reads_only(launcher):
    a,repo,donor,calls,_=launcher;a.mode='plan'
    S.launch(a)
    assert not calls and not (repo/'checkpoints').exists()


@pytest.mark.parametrize('fault',['existing','dirty','down','limit','architecture','interpreter','storage'])
def test_fail_without_actual_submission(launcher,monkeypatch,fault):
    a,repo,donor,calls,b=launcher
    if fault=='existing':a.output.mkdir(parents=True)
    if fault=='dirty':(repo/'part_ev_support/model.py').write_text('dirty')
    if fault=='down':b['partition']='PartitionName=tier3 State=DOWN MaxTime=20-00:00:00'
    if fault=='limit':b['partition']='PartitionName=tier3 State=UP MaxTime=05:00:00'
    if fault=='architecture':monkeypatch.setattr(S.platform,'machine',lambda:'aarch64')
    if fault=='interpreter':a.python=str(repo/'not-python')
    if fault=='storage':
        def fail(*a):raise OSError('headroom')
        monkeypatch.setattr(S.P,'check_storage',fail)
    with pytest.raises((ValueError,FileExistsError,OSError)):S.launch(a)
    assert not actual_jobs(calls)


def test_uncertain_reply_keeps_intent_blocks_duplicate(launcher):
    a,_,_,calls,b=launcher;b['reply']='not-a-job-id'
    with pytest.raises(ValueError,match='Uncertain'):S.launch(a)
    assert len(actual_jobs(calls))==1
    assert S.P.read_json(a.output/'submission_intent.json')['status']=='unknown'
    assert not (a.output/'submitted_job_ids.tsv').exists()
    with pytest.raises(FileExistsError):S.launch(a)
    assert len(actual_jobs(calls))==1


def test_unsafe_output_and_production_rejected(launcher):
    a,repo,donor,calls,b=launcher
    for mode,output in [('train',a.output),('benchmark',donor),('benchmark',repo/'checkpoints'),
                        ('benchmark',a.output/'nested')]:
        with pytest.raises(SystemExit):S.parse(['--mode',mode,'--output',str(output)])


def test_export_sanitization_and_no_implicit_gpu_visibility(monkeypatch):
    for k in ('SBATCH_GRES','SLURM_JOB_ID','CUDA_VISIBLE_DEVICES','WORLD_SIZE','RANK','PYTHONPATH','NCCL_DEBUG'):
        monkeypatch.setenv(k,'contaminated')
    env=S.environment(sys.executable,{'NCCL_IB_DISABLE':'1'})
    assert not any(k.startswith(('SBATCH_','SLURM_')) for k in env)
    assert not {'CUDA_VISIBLE_DEVICES','WORLD_SIZE','RANK','PYTHONPATH','NCCL_DEBUG'} & env.keys()
    assert env['NCCL_IB_DISABLE']=='1' and env['OMP_NUM_THREADS']=='2'


@pytest.mark.skipif(not BASH,reason='Bash unavailable')
def test_shell_syntax():
    repo=Path(__file__).resolve().parents[1]
    for script in ('submit_part_ev_support.sh','run_part_ev_support.sh'):
        result=subprocess.run([BASH,'-n',shell_path(repo/'sbatch'/script)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr


@pytest.mark.skipif(not BASH,reason='Bash unavailable')
def test_frozen_worker_dispatch_and_inherited_rank_cleanup(tmp_path):
    # Fake Python checks dispatch only; snapshot tamper checks are exercised
    # with the real Python protocol in test_independent_chain_and_frozen_snapshot.
    repo=Path(__file__).resolve().parents[1]
    output=tmp_path/'benchmark with spaces';(output/'source').mkdir(parents=True)
    binary=tmp_path/'bin';binary.mkdir()
    python=binary/'python'
    python.write_text('''#!/usr/bin/env bash
if [[ "$2" == -c ]]; then
  if [[ "$3" == *shlex* ]]; then printf 'export NCCL_DEBUG=INFO\\n'; fi
  exit 0
fi
printf 'ARGS=%s\\n' "$*"
printf 'WORLD=%s RANK=%s OMP=%s NCCL=%s VISIBLE=%s\\n' "${WORLD_SIZE:-}" "${RANK:-}" "$OMP_NUM_THREADS" "$NCCL_DEBUG" "$CUDA_VISIBLE_DEVICES"
''',newline='\n')
    (binary/'python3').write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$READOUT_TEST_PYTHON"\n',newline='\n')
    for p in binary.iterdir():p.chmod(0o755)
    (output/'launch.json').write_text('{}')
    (output/'launch.sha256').write_text(S.P.file_sha256(output/'launch.json')+'  launch.json\n',newline='\n')
    env=dict(os.environ,READOUT_TEST_PYTHON=shell_path(python),WORLD_SIZE='99',RANK='99',
             NCCL_DEBUG='bad-inherited',CUDA_VISIBLE_DEVICES='0,1')
    result=subprocess.run([BASH,'--noprofile','--norc','-c',
        'export PATH="$1:$PATH"; shift; exec bash "$@"','test',shell_path(binary),
        shell_path(repo/'sbatch/run_part_ev_support.sh'),shell_path(output),'benchmark'],
        env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert '--nproc-per-node=2' in result.stdout and '--max-restarts=0' in result.stdout
    assert 'WORLD= RANK= OMP=2 NCCL=INFO VISIBLE=0,1' in result.stdout

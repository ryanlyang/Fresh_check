import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import submit_part_selected_production as S
from part_selected_production import protocol as P
from part_selected_fullscale import protocol as science
from tests.test_part_selected_production import review
from tests.test_part_fullscale_launcher import BASH, shell_path


@pytest.fixture
def setup(tmp_path,monkeypatch):
    repo=tmp_path/'repo'; repo.mkdir()
    for name in P.FILES:
        path=repo/name; path.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(S.ROOT/name,path)
    subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-m','fixture'],check=True,capture_output=True)
    monkeypatch.setattr(S,'ROOT',repo)
    monkeypatch.setattr(platform,'machine',lambda:'aarch64')
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=18*2**30))
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None))
    old_which=shutil.which
    monkeypatch.setattr(shutil,'which',lambda n:'/fake/'+n if n in ('sbatch','squeue','scontrol') else old_which(n))
    monkeypatch.setenv('SBATCH_GRES','gpu:wrong:99')
    monkeypatch.setenv('RANK','99'); monkeypatch.setenv('WORLD_SIZE','99')
    bench=tmp_path/'benchmark'; bench.mkdir(); retry=bench/'retry'; retry.mkdir()
    output=repo/'checkpoints/part_selected_production/new'
    evidence=dict(science=dict(content_hash='original',tasks=science.TASKS),review=review(),
                  evidence={},python=sys.executable,account='reu-aisocial')
    evidence['review']['storage']=dict(required_free_bytes=12*2**30)
    monkeypatch.setattr(S,'inspect',lambda _:evidence)
    state=dict(jobs=[],checks=[],live='',fail=None,test_fail=False)
    real=S.command
    def command(argv,**kwargs):
        if argv[0]=='squeue': return state['live']
        if '--phase' in argv: return '{"verified":true}'
        if argv[0]=='sbatch':
            if '--test-only' in argv:
                state['checks'].append(argv)
                if state['test_fail']: raise subprocess.CalledProcessError(1,argv)
                return 'Valid'
            if state['fail']==len(state['jobs']): raise subprocess.CalledProcessError(1,argv)
            job=str(1000+len(state['jobs'])); state['jobs'].append((argv,kwargs['env'])); return job+';tigris'
        return real(argv,**kwargs)
    monkeypatch.setattr(S,'command',command)
    args=['--benchmark',str(bench),'--retry',str(retry),'--output',str(output)]
    return repo,output,args,state


def test_plan_is_read_only(setup,capsys):
    repo,output,args,state=setup
    before={p:P.sha(p) for p in repo.rglob('*') if p.is_file()}
    S.launch(S.parse(['--mode','plan',*args]))
    assert not output.exists() and not state['jobs'] and not state['checks']
    assert 'No files or jobs changed' in capsys.readouterr().out
    assert before=={p:P.sha(p) for p in repo.rglob('*') if p.is_file()}


def test_seven_full_individual_jobs_and_afterok_evaluations(setup):
    repo,output,args,state=setup
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert len(state['jobs'])==15 and len(state['checks'])==15
    plan=P.checked(output/'production.json')
    assert plan['source']=={n:P.sha(output/'source'/n) for n in P.FILES}
    for i in range(7):
        train,env=state['jobs'][2*i]; evaluate,_=state['jobs'][2*i+1]
        assert f'--time={P.TRAIN_HOURS[i]*60}' in train
        assert f'--time={P.EVALUATE_HOURS[i]*60}' in evaluate
        assert '--gres=gpu:gh200:1' in train and '--mem=128G' in train
        assert not any(x.startswith('--array') for x in train)
        assert '--dependency=afterok:'+str(1000+2*i) in evaluate
        assert '--kill-on-invalid-dep=yes' in train and '--signal=B:USR1@900' in train
        assert not any(k.startswith('SBATCH_') for k in env)
        assert env['WORLD_SIZE']=='1' and 'RANK' not in env
        if i>=3: assert '--dependency=afterany:'+str(1000+2*(i-3)) in train
    report,_=state['jobs'][-1]
    assert '--dependency=afterok:'+':'.join(str(1001+2*i) for i in range(7)) in report
    assert not any(x.startswith('--gres') for x in report)
    assert '--time=30' in report
    assert not any(x.startswith('--signal') for x in report)
    state['live']='1000\n1001'
    with pytest.raises(ValueError,match='active/pending'):
        S.launch(S.parse(['--mode','resume','--reviewed','--output',str(output)]))
    with pytest.raises(ValueError,match='production owner'):
        S.launch(S.parse(['--mode','submit','--reviewed',*args[:-1],str(output.parent/'duplicate')]))


def test_testonly_failure_submits_nothing(setup):
    _,output,args,state=setup; state['test_fail']=True
    with pytest.raises(subprocess.CalledProcessError): S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert state['jobs']==[] and not (output/'submitted_job_ids.tsv').exists()


def test_partial_submission_is_durable_and_blocks_ambiguous_retry(setup):
    _,output,args,state=setup; state['fail']=1
    with pytest.raises(subprocess.CalledProcessError): S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert len(state['jobs'])==1
    assert '1000' in (output/'submitted_job_ids.tsv').read_text()
    assert P.read(output/'submission_intent.json')['status']=='unknown'
    with pytest.raises(ValueError,match='Unresolved sbatch'):
        S.launch(S.parse(['--mode','resume','--reviewed','--output',str(output)]))


def test_resume_only_missing_stages(setup):
    _,output,args,state=setup
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    plan=P.read(output/'production.json')
    for i,t in enumerate(plan['science']['tasks']):
        run=P.run_directory(output,t)
        if i!=6: P.write(run/'training_complete.json',{})
        if i<5: P.write(run/'full_validation/metrics.json',{})
    state['jobs'].clear(); state['checks'].clear()
    S.launch(S.parse(['--mode','resume','--reviewed','--output',str(output)]))
    names=[r[-2:] for r,_ in state['jobs']]
    assert names==[['evaluate','5'],['train','6'],['evaluate','6'],['report','-']]


def test_explicit_review_and_no_short_budget_options(setup):
    _,_,args,_=setup
    with pytest.raises(SystemExit): S.parse(['--mode','submit',*args])
    with pytest.raises(SystemExit): S.parse(['--mode','submit','--reviewed','--indices','0',*args])
    with pytest.raises(SystemExit): S.parse(['--mode','plan','--total-steps','200000',*args])


@pytest.mark.skipif(BASH is None,reason='Bash unavailable')
def test_bash_wrappers_parse():
    for name in ('run','submit'):
        path=S.ROOT/f'sbatch/{name}_part_selected_production_tigris.sh'
        subprocess.run([BASH,'-n',shell_path(path)],check=True,capture_output=True)


def test_actual_worker_import_boundary(tmp_path):
    """A newer checkout cannot leak into the original scientific namespace."""
    parent=tmp_path/'parent'; source=parent/'source'; retry=parent/'benchmark_retries/retry.test'
    (source/'part_selected_fullscale').mkdir(parents=True); retry.mkdir(parents=True)
    (source/'part_selected_fullscale/__init__.py').write_text('')
    (source/'part_selected_fullscale/protocol.py').write_text('MARKER="original_science"\n')
    P.write(parent/'launch.json',dict(python=sys.executable))
    # Stub only the evidence loader; the subprocess's namespace ordering is real.
    (retry/'retry_part_selected_fullscale.py').write_text(
        'def load_plan(output):\n    return '+repr(dict(campaign=str(parent),python=sys.executable))+'\n')
    program='\n'.join(['from part_selected_production.protocol import activate',
        f'driver,science=activate({str(parent)!r},{str(retry)!r})',
        'assert science.MARKER=="original_science"',
        f'assert science.__file__.startswith({str(source)!r})'])
    subprocess.run([sys.executable,'-B','-c',program],cwd=S.ROOT,check=True,capture_output=True)


def test_allocation_deadline_requires_actual_slurm_end(monkeypatch):
    from datetime import datetime
    from scripts import run_part_selected_production as W
    monkeypatch.setenv('SLURM_JOB_ID','123')
    now=1800000000.
    monkeypatch.setattr(W.time,'time',lambda:now)
    monkeypatch.setattr(W.time,'monotonic',lambda:200.)
    end=datetime.fromtimestamp(now+10000).isoformat()
    monkeypatch.setattr(W.subprocess,'check_output',lambda *a,**k:'JobId=123 EndTime='+end)
    assert W.deadline()==9300.
    monkeypatch.setattr(W.subprocess,'check_output',lambda *a,**k:'EndTime=Unknown')
    with pytest.raises(ValueError,match='deadline'): W.deadline()


def test_storage_budget_accounts_for_remaining_outputs(tmp_path):
    plan=dict(review=dict(storage=dict(required_free_bytes=12*2**30,reserve_bytes=4*2**30)))
    assert S.remaining_storage(tmp_path,plan)==12*2**30
    (tmp_path/'output.bin').write_bytes(b'x'*100)
    assert S.remaining_storage(tmp_path,plan)==12*2**30-100


def test_sources_support_cluster_python_310():
    import ast
    for name in P.FILES:
        if name.endswith('.py'):
            ast.parse((S.ROOT/name).read_text(encoding='utf-8'),filename=name,feature_version=(3,10))

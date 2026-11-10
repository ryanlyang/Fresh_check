import ast
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import submit_part_ev_compact_production as S
from part_ev_compact_production import protocol as P
from part_ev_compact import protocol as C
from tests.test_part_fullscale_launcher import BASH,shell_path


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
    old=shutil.which
    monkeypatch.setattr(shutil,'which',lambda n:'/fake/'+n if n in ('sbatch','squeue','scontrol') else old(n))
    monkeypatch.setenv('SBATCH_GRES','gpu:wrong:99'); monkeypatch.setenv('SLURM_MEM_PER_CPU','9999')
    monkeypatch.setenv('RANK','99'); monkeypatch.setenv('WORLD_SIZE','99')
    benchmark=tmp_path/'benchmark'; benchmark.mkdir()
    output=repo/'checkpoints/part_ev_compact_production/new'
    evidence=dict(benchmark=str(benchmark),benchmark_spec={'content_hash':'benchmark'},
        science=dict(tasks=[dict(t,recipe=t['architecture']) for t in C.TASKS],
            recipes={a:C.recipe(a,n) for a,n in [('compact',1000000),('matched',1243800)]}),
        python=sys.executable,account='reu-aisocial',train_hours=[150,75],evaluate_hours=[5,5],
        storage=dict(required_free_bytes=8*2**30,reserve_bytes=4*2**30),evidence={})
    monkeypatch.setattr(S,'inspect',lambda _:evidence)
    state=dict(jobs=[],checks=[],live='',fail=None,test_fail=False,ready=[[0,True,True],[1,True,True]])
    real=S.command
    def command(argv,**kwargs):
        if argv[0]=='squeue': return state['live']
        if '--phase' in argv:
            phase=argv[argv.index('--phase')+1]
            return json.dumps(state['ready'] if phase=='ready' else {'verified':True})
        if argv[0]=='sbatch':
            if '--test-only' in argv:
                state['checks'].append(argv)
                if state['test_fail']: raise subprocess.CalledProcessError(1,argv)
                return 'Valid'
            if state['fail']==len(state['jobs']): raise subprocess.CalledProcessError(1,argv)
            job=str(1000+len(state['jobs'])); state['jobs'].append((argv,kwargs['env'])); return job+';tigris'
        return real(argv,**kwargs)
    monkeypatch.setattr(S,'command',command)
    args=['--benchmark',str(benchmark),'--output',str(output)]
    return repo,output,args,state


def test_plan_is_read_only(setup,capsys):
    repo,output,args,state=setup
    before={p:P.sha(p) for p in repo.rglob('*') if p.is_file()}
    S.launch(S.parse(['--mode','plan',*args]))
    assert not output.exists() and not state['jobs'] and not state['checks']
    assert 'No files or jobs changed' in capsys.readouterr().out
    assert before=={p:P.sha(p) for p in repo.rglob('*') if p.is_file()}


def test_five_jobs_no_cross_training_dependency_and_freeze(setup):
    repo,output,args,state=setup
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert len(state['jobs'])==len(state['checks'])==5
    for i in range(2):
        train,env=state['jobs'][2*i]; evaluate,_=state['jobs'][2*i+1]
        assert f'--time={[9000,4500][i]}' in train
        assert '--time=300' in evaluate
        assert not any(s.startswith(('--dependency','--array')) for s in train)
        assert '--dependency=afterok:'+str(1000+2*i) in evaluate
        assert '--gres=gpu:gh200:1' in train and '--mem=128G' in train
        assert '--signal=B:USR1@900' in train
        assert not any(k.startswith(('SBATCH_','SLURM_')) for k in env)
        assert env['WORLD_SIZE']=='1' and 'RANK' not in env
        assert str(output/'source/sbatch/run_part_ev_compact_production_tigris.sh') in train
    report,_=state['jobs'][4]
    assert '--dependency=afterok:1001:1003' in report and '--time=60' in report
    assert not any(s.startswith('--gres') for s in report)
    plan=P.checked(output/'production.json')
    before=P.inventory(output/'source')
    (repo/'part_ev_compact_production/engine.py').write_text('# subsequent live pull')
    assert P.inventory(output/'source')==before==plan['source']
    assert '--benchmark' not in state['jobs'][0][0]  # Identity from frozen plan, not mutable CLI.


def test_active_duplicate_and_benchmark_owner_guard(setup):
    _,output,args,state=setup
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    state['live']='1000\n1001'
    with pytest.raises(ValueError,match='active/pending'):
        S.launch(S.parse(['--mode','resume','--reviewed','--output',str(output)]))
    with pytest.raises(ValueError,match='production owner'):
        S.launch(S.parse(['--mode','submit','--reviewed',*args[:-1],str(output.parent/'duplicate')]))


def test_testonly_failure_submits_nothing(setup):
    _,output,args,state=setup; state['test_fail']=True
    with pytest.raises(subprocess.CalledProcessError): S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert not state['jobs'] and not (output/'submitted_job_ids.tsv').exists()


def test_uncertain_partial_submission_journal(setup):
    _,output,args,state=setup; state['fail']=1
    with pytest.raises(subprocess.CalledProcessError): S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert len(state['jobs'])==1 and '1000' in (output/'submitted_job_ids.tsv').read_text()
    assert P.read(output/'submission_intent.json')['status']=='unknown'
    with pytest.raises(ValueError,match='Unresolved sbatch'):
        S.launch(S.parse(['--mode','resume','--reviewed','--output',str(output)]))


def test_resume_only_audited_missing_stages(setup):
    _,output,args,state=setup
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    state['jobs'].clear(); state['checks'].clear(); state['ready']=[[1,False,True]]
    S.launch(S.parse(['--mode','resume','--reviewed','--output',str(output)]))
    assert [r[-2:] for r,_ in state['jobs']]==[['evaluate','1'],['report','-']]
    assert not any(s.startswith('--dependency') for s in state['jobs'][0][0])


def test_subset_resume_defers_report(setup):
    _,output,args,state=setup
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    state['jobs'].clear(); state['checks'].clear()
    S.launch(S.parse(['--mode','resume','--reviewed','--indices','0','--output',str(output)]))
    assert [r[-2:] for r,_ in state['jobs']]==[['train','0'],['evaluate','0']]


def test_no_source_mutation_short_budget_or_unreviewed_submission(setup):
    _,output,args,_=setup
    for extras in ([],['--total-steps','200000'],['--indices','1'],['--concurrency','1']):
        with pytest.raises(SystemExit): S.parse(['--mode','submit',*extras,*args])
    with pytest.raises(SystemExit): S.parse(['--mode','resume','--reviewed','--output',str(output.parents[2])])


def test_no_storage_bypass(setup,monkeypatch):
    _,output,args,state=setup
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=6*2**30))
    with pytest.raises(OSError,match='headroom'): S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert not output.exists() and not state['jobs']


def test_remaining_storage(tmp_path):
    plan={'storage':{'required_free_bytes':8*2**30,'reserve_bytes':4*2**30}}
    (tmp_path/'partial.bin').write_bytes(b'abc')
    assert S.remaining_storage(tmp_path,plan)==8*2**30-3


def test_actual_benchmark_import_boundary_and_tamper(tmp_path):
    bench=tmp_path/'benchmark'; source=bench/'source'
    (source/'part_ev_compact').mkdir(parents=True)
    (source/'part_ev_compact/__init__.py').write_text('')
    (source/'part_ev_compact/protocol.py').write_text('MARKER="frozen_science"\n')
    P.write(bench/'snapshot.json',P.inventory(source)); P.write(bench/'launch.json',{'python':sys.executable})
    program='\n'.join(['from part_ev_compact_production.protocol import activate',
        f'science=activate({str(bench)!r})','assert science.MARKER=="frozen_science"',
        f'assert science.__file__.startswith({str(source)!r})'])
    subprocess.run([sys.executable,'-B','-c',program],cwd=S.ROOT,check=True,capture_output=True)
    (source/'part_ev_compact/protocol.py').write_text('MARKER="changed"\n')
    result=subprocess.run([sys.executable,'-B','-c',program],cwd=S.ROOT,capture_output=True,text=True)
    assert result.returncode and 'frozen source changed' in result.stderr


def test_allocation_deadline(monkeypatch):
    from datetime import datetime
    from scripts import run_part_ev_compact_production as W
    now=1800000000.
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setattr(W.time,'time',lambda:now); monkeypatch.setattr(W.time,'monotonic',lambda:200.)
    end=datetime.fromtimestamp(now+10000).isoformat()
    monkeypatch.setattr(W.subprocess,'check_output',lambda *a,**kw:'JobId=123 EndTime='+end)
    assert W.deadline()==9300.
    monkeypatch.setattr(W.subprocess,'check_output',lambda *a,**kw:'EndTime=Unknown')
    with pytest.raises(ValueError,match='deadline'): W.deadline()


def test_python310_syntax():
    for name in P.FILES:
        if name.endswith('.py'): ast.parse((S.ROOT/name).read_text(),filename=name,feature_version=(3,10))


@pytest.mark.skipif(BASH is None,reason='Bash unavailable')
def test_shell_syntax():
    for name in ('run','submit'):
        subprocess.run([BASH,'-n',shell_path(S.ROOT/f'sbatch/{name}_part_ev_compact_production_tigris.sh')],
                       check=True,capture_output=True)

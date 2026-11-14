import ast
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import submit_part_ev_state_production as S
from part_ev_state_production import protocol as P
from part_ev_state_bias import protocol as C
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
    monkeypatch.setattr(platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(platform,'system',lambda:'Linux')
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=18*2**30))
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None))
    old=shutil.which
    monkeypatch.setattr(shutil,'which',lambda n:'/fake/'+n if n in ('sbatch','squeue','scontrol') else old(n))
    monkeypatch.setenv('SBATCH_GRES','gpu:wrong:99'); monkeypatch.setenv('SLURM_MEM_PER_CPU','9999')
    monkeypatch.setenv('RANK','99'); monkeypatch.setenv('WORLD_SIZE','99')
    benchmark=tmp_path/'benchmark'; benchmark.mkdir()
    output=repo/'checkpoints/part_ev_state_production/new'
    evidence=dict(benchmark=str(benchmark),benchmark_sha256='benchmark',model_kind='bias',
        task=C.TASK,recipes=C.recipes(),python=sys.executable,account='reu-aisocial',partition='tier3',
        train_hours=308,evaluate_hours=4,expected_gpu={'nccl_environment':{}},
        storage=dict(required_free_bytes=7*2**30,reserve_bytes=4*2**30),evidence={})
    monkeypatch.setattr(S,'inspect',lambda _:evidence)
    state=dict(jobs=[],checks=[],live='',fail=None,test_fail=False,ready=['train','evaluate','report'],partition='State=UP MaxTime=20-00:00:00')
    real=S.command
    def command(argv,**kwargs):
        if argv[0]=='squeue': return state['live']
        if argv[0]=='scontrol': return state['partition']
        if '--phase' in argv:
            phase=argv[argv.index('--phase')+1]
            return json.dumps(state['ready'] if phase=='ready' else {'verified':True})
        if argv[0]=='sbatch':
            if '--test-only' in argv:
                state['checks'].append(argv)
                if state['test_fail']: raise subprocess.CalledProcessError(1,argv)
                return 'Valid'
            if state['fail']==len(state['jobs']): raise subprocess.CalledProcessError(1,argv)
            job=str(1000+len(state['jobs'])); state['jobs'].append((argv,kwargs['env'])); return job+';sporc'
        return real(argv,**kwargs)
    monkeypatch.setattr(S,'command',command)
    args=['--model','bias','--benchmark',str(benchmark),'--output',str(output)]
    return repo,output,args,state


def test_plan_is_read_only(setup,capsys):
    repo,output,args,state=setup
    before={p:P.sha(p) for p in repo.rglob('*') if p.is_file()}
    S.launch(S.parse(['--mode','plan',*args]))
    assert not output.exists() and not state['jobs'] and not state['checks']
    assert 'No files/jobs changed' in capsys.readouterr().out
    assert before=={p:P.sha(p) for p in repo.rglob('*') if p.is_file()}


def test_three_independent_jobs_resources_and_freeze(setup):
    repo,output,args,state=setup
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert len(state['jobs'])==len(state['checks'])==3
    train,env=state['jobs'][0]; evaluate,_=state['jobs'][1]; report,_=state['jobs'][2]
    assert '--time=18480' in train and '--gres=gpu:a100:2' in train
    assert '--time=240' in evaluate and '--gres=gpu:a100:1' in evaluate
    assert '--dependency=afterok:1000' in evaluate and '--dependency=afterok:1001' in report
    assert not any(s.startswith(('--dependency','--array')) for s in train)
    assert '--mem=64G' in train and '--cpus-per-task=8' in train
    assert '--signal=B:USR1@900' in train
    assert not any(k.startswith(('SBATCH_','SLURM_')) for k in env)
    assert env['WORLD_SIZE']=='1' and 'RANK' not in env
    assert str(output/'source/sbatch/run_part_ev_state_production.sh') in train
    assert '--time=60' in report and not any(s.startswith('--gres') for s in report)
    plan=P.checked(output/'campaign.json')
    before=P.inventory(output/'source')
    (repo/'part_ev_state_production/engine.py').write_text('# subsequent live pull')
    assert P.inventory(output/'source')==before==plan['source']
    assert '--benchmark' not in train


def test_multiply_time_is_independent(setup,monkeypatch):
    _,output,args,state=setup
    evidence=S.inspect(None).copy()
    evidence.update(model_kind='multiply',train_hours=226,evaluate_hours=2)
    monkeypatch.setattr(S,'inspect',lambda _:evidence)
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert '--time=13560' in state['jobs'][0][0]
    assert '--time=120' in state['jobs'][1][0]


def memory_evidence(evidence):
    from part_ev_pair_memory.protocol import TASK,PARAMETERS,CONFIG
    return dict(evidence,model_kind='memory',task=TASK,benchmark_sha256='memory-benchmark',
        parameters={TASK['architecture']:PARAMETERS},model_config=CONFIG,
        train_hours=372,evaluate_hours=4)


def test_memory_submits_only_its_three_jobs_after_existing_chains(setup,monkeypatch):
    from part_ev_state_multiply.protocol import TASK as MUL_TASK
    repo,bias,args,state=setup
    original=S.inspect(None).copy()
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    multiply=bias.parent/'multiply'
    monkeypatch.setattr(S,'inspect',lambda _:dict(original,model_kind='multiply',task=MUL_TASK,
        benchmark_sha256='multiply-benchmark',train_hours=226,evaluate_hours=2))
    S.launch(S.parse(['--mode','submit','--reviewed','--model','multiply',
        '--benchmark',original['benchmark'],'--output',str(multiply)]))
    state['live']='\n'.join(str(i) for i in range(1000,1006))
    old={p:P.sha(p) for root in (bias,multiply) for p in root.rglob('*') if p.is_file()}
    # Simulate a later committed pull in the live checkout, not old snapshots.
    (repo/'part_ev_state_production/__init__.py').write_text('# newer production commit')
    subprocess.run(['git','-C',str(repo),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid',
        'commit','-m','new independent campaign'],check=True,capture_output=True)
    memory=bias.parent/'memory'
    monkeypatch.setattr(S,'inspect',lambda _:memory_evidence(original))
    S.launch(S.parse(['--mode','submit','--reviewed','--model','memory',
        '--benchmark',original['benchmark'],'--output',str(memory)]))
    assert len(state['jobs'])==len(state['checks'])==9
    train,_=state['jobs'][6];evaluate,_=state['jobs'][7];report,_=state['jobs'][8]
    assert '--job-name=evstateprod_memory_train' in train
    assert '--time=22320' in train and '--gres=gpu:a100:2' in train
    assert '--mem=64G' in train and '--cpus-per-task=8' in train
    assert not any(s.startswith(('--dependency','--array')) for s in train)
    assert '--time=240' in evaluate and '--gres=gpu:a100:1' in evaluate
    assert '--dependency=afterok:1006' in evaluate
    assert '--time=60' in report and '--dependency=afterok:1007' in report
    assert not any(s.startswith('--gres') for s in report)
    assert old=={p:P.sha(p) for root in (bias,multiply) for p in root.rglob('*') if p.is_file()}
    plan=P.checked(memory/'campaign.json')
    assert plan['task']['run_id']=='STATE_MEMORY_EV_LAYERWISE'
    assert plan['parameters']=={'state_pair_memory_layerwise':2364294}
    assert plan['recipes'][plan['task']['recipe']]['total_steps']==1000000
    assert plan['source']!=P.checked(bias/'campaign.json')['source']


def test_memory_respects_partition_and_other_campaign_storage(setup,monkeypatch):
    _,output,args,state=setup
    original=S.inspect(None).copy()
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    old={p:P.sha(p) for p in output.rglob('*') if p.is_file()}
    memory=output.parent/'memory'
    monkeypatch.setattr(S,'inspect',lambda _:memory_evidence(original))
    arguments=['--mode','submit','--reviewed','--model','memory',
        '--benchmark',original['benchmark'],'--output',str(memory)]
    # Bias fits 14 days, but memory's measured requirement does not.
    state['partition']='State=UP MaxTime=14-00:00:00'
    with pytest.raises(ValueError,match='walltime'):S.launch(S.parse(arguments))
    assert len(state['jobs'])==3 and not memory.exists()
    state['partition']='State=UP MaxTime=20-00:00:00'
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=9*2**30))
    with pytest.raises(OSError,match='aggregate'):S.launch(S.parse(arguments))
    assert len(state['jobs'])==3 and not memory.exists()
    assert old=={p:P.sha(p) for p in output.rglob('*') if p.is_file()}


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
    state['jobs'].clear(); state['checks'].clear(); state['ready']=['evaluate','report']
    S.launch(S.parse(['--mode','resume','--reviewed','--output',str(output)]))
    assert [r[-1] for r,_ in state['jobs']]==['evaluate','report']
    assert not any(s.startswith('--dependency') for s in state['jobs'][0][0])


def test_complete_resume_is_noop(setup):
    _,output,args,state=setup
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    state['jobs'].clear(); state['checks'].clear();state['ready']=[]
    S.launch(S.parse(['--mode','resume','--reviewed','--output',str(output)]))
    assert not state['jobs'] and not state['checks']


@pytest.mark.parametrize('partition',['State=DOWN MaxTime=20-00:00:00',
    'State=UP MaxTime=12-00:00:00','State=UP MaxTime=Unknown'])
def test_partition_cannot_fit_refuses_before_freeze(setup,partition):
    _,output,args,state=setup;state['partition']=partition
    with pytest.raises(ValueError):
        S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert not state['jobs'] and not output.exists()


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


@pytest.mark.parametrize('kind',['bias','multiply','memory'])
def test_actual_benchmark_import_boundary_and_tamper(tmp_path,kind):
    bench=tmp_path/'benchmark'; source=bench/'source'
    package=source/P.PACKAGES[kind]
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'protocol.py').write_text('MARKER="frozen_science"\n')
    P.write(bench/'source_manifest.json',P.inventory(source)); P.write(bench/'launch.json',{'python':sys.executable})
    program='\n'.join(['from part_ev_state_production.protocol import activate',
        f'science=activate({str(bench)!r},{kind!r})','assert science.MARKER=="frozen_science"',
        f'assert science.__file__.startswith({str(source)!r})'])
    subprocess.run([sys.executable,'-B','-c',program],cwd=S.ROOT,check=True,capture_output=True)
    (package/'protocol.py').write_text('MARKER="changed"\n')
    result=subprocess.run([sys.executable,'-B','-c',program],cwd=S.ROOT,capture_output=True,text=True)
    assert result.returncode and 'frozen source changed' in result.stderr


def test_allocation_deadline(monkeypatch):
    from datetime import datetime
    from scripts import run_part_ev_state_production as W
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
        subprocess.run([BASH,'-n',shell_path(S.ROOT/f'sbatch/{name}_part_ev_state_production.sh')],
                       check=True,capture_output=True)


def test_other_registered_campaigns_count_toward_storage(setup,monkeypatch):
    repo,output,args,state=setup
    S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    evidence=S.inspect(None).copy();evidence['benchmark_sha256']='other'
    monkeypatch.setattr(S,'inspect',lambda _:evidence)
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=9*2**30))
    with pytest.raises(OSError,match='aggregate'):
        S.launch(S.parse(['--mode','submit','--reviewed',*args[:-1],str(output.parent/'second')]))
    assert len(state['jobs'])==3


def test_dirty_production_refused(setup):
    repo,output,args,state=setup
    (repo/'part_ev_state_production/engine.py').write_text('# uncommitted')
    with pytest.raises(ValueError,match='Commit/push/pull'):
        S.launch(S.parse(['--mode','submit','--reviewed',*args]))
    assert not state['jobs'] and not output.exists()


def test_no_new_benchmark_or_test_or_segment_phases():
    # Scientific implementation remains in the benchmark snapshot; the small
    # production snapshot contains orchestration only, not a new model copy.
    assert not any('pair_memory/' in x for x in P.FILES)
    script=(S.ROOT/'sbatch/run_part_ev_state_production.sh').read_text()
    assert '--nproc-per-node=2' in script and '--max-restarts=0' in script
    assert 'trap ' in script and 'EVPROD_STOP_FILE' in script
    assert 'exec "$evprod_python"' not in script  # batch shell owns warning signal

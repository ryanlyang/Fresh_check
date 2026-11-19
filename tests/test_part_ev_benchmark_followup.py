import ast
import copy
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from part_ev_benchmark_followup import protocol as F
from part_ev_state_production import protocol as P
from scripts import submit_part_ev_benchmark_followup as S
from scripts import run_part_ev_benchmark_followup as W
from tests.test_part_fullscale_launcher import BASH,shell_path


def benchmark(repo,kind='readout',job='22138551'):
    bench=repo/'checkpoints'/F.MODELS[kind][0]/'bench'
    source=bench/'source';source.mkdir(parents=True)
    (source/'science.py').write_text('# immutable scientific fixture\n')
    manifest=P.inventory(source)
    P.write(bench/'source_manifest.json',manifest)
    P.write(bench/'launch.json',P.hashed(dict(schema=F.MODELS[kind][0]+'_benchmark_v1',
        output=str(bench),source_sha256=P.digest(manifest),partition='tier3',account='reu-aisocial',
        authorizes_training=False,python=sys.executable,output_budget_bytes=2**30)))
    (bench/'submitted_job_ids.tsv').write_text(f'prepare\t{int(job)-2}\nbenchmark\t{int(job)-1}\nsummary\t{job}\n')
    return bench


@pytest.fixture
def setup(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    for name in F.FILES:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(S.ROOT/name,path)
    subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-m','fixture'],check=True,capture_output=True)
    monkeypatch.setattr(S,'ROOT',repo)
    monkeypatch.setattr(platform,'system',lambda:'Linux');monkeypatch.setattr(platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(os,'getuid',lambda:123,raising=False)
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=23*2**30))
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None))
    old=shutil.which
    monkeypatch.setattr(shutil,'which',lambda n:'/fake/'+n if n in ('sbatch','squeue','scontrol') else old(n))
    bench=benchmark(repo)
    output=repo/'checkpoints/part_ev_benchmark_followup/readout'
    prod=repo/'checkpoints/part_ev_state_production/readout'
    state=dict(jobs=[],checks=[],live='',fail=False,job_state='PENDING',job_override={},aged=False)
    real=S.command
    def command(argv,**kwargs):
        if argv[0]=='squeue':return state['live']
        if argv[0]=='scontrol':
            if state['aged']:raise subprocess.CalledProcessError(1,argv)
            job=argv[3]
            roots=[p.parent for p in (repo/'checkpoints').glob('part_ev_*/bench/submitted_job_ids.tsv')
                   if f'summary\t{job}\n' in p.read_text()]
            root=roots[0];prefix=next(v[1] for v in F.MODELS.values() if v[0]==root.parent.name)
            row=dict(JobId=job,JobName=prefix+'_summary',UserId='ryreu(123)',Partition='tier3',
                JobState=state['job_state'],ExitCode='0:0',StdOut=str(root/'logs'/f'{prefix}_summary_{job}.out'))
            row.update(state['job_override']);return ' '.join(f'{k}={v}' for k,v in row.items())
        if argv[0]=='sbatch':
            if '--test-only' in argv:state['checks'].append(argv);return 'Valid'
            if state['fail']:raise subprocess.CalledProcessError(1,argv)
            job=str(1000+len(state['jobs']));state['jobs'].append((argv,kwargs['env']));return job+';sporc'
        return real(argv,**kwargs)
    monkeypatch.setattr(S,'command',command)
    args=['--model','readout','--benchmark',str(bench),'--output',str(output),'--production',str(prod)]
    return repo,bench,output,prod,args,state


def arm(setup):
    _,_,output,_,args,_=setup
    S.launch(S.parse(['--mode','arm','--reviewed',*args]))
    return F.load(output)


def admitted(arm):
    review=dict(passed=True,complete=True,authorizes_training=False,projected_train_hours=200.,
        projected_validation_hours=20.,projected_checkpoint_hours=.1,
        recommended_train_hours=math.ceil(1.3*220.1),recommended_evaluate_hours=4)
    return dict(model_kind=arm['model_kind'],benchmark=arm['benchmark'],
        python=arm['benchmark_identity']['python'],account=arm['benchmark_identity']['account'],
        task=dict(seed=101,recipe='ranger_full_v1'),
        recipes={'ranger_full_v1':dict(total_steps=1000000,epochs=50,steps_per_epoch=20000,
            validation_events_per_epoch=1280000,global_batch_size=512)},review=review,
        train_hours=review['recommended_train_hours'],evaluate_hours=4)


def test_plan_read_only_and_unreviewed_rejected(setup,capsys):
    repo,_,output,_,args,state=setup
    before={p:P.sha(p) for p in repo.rglob('*') if p.is_file()}
    S.launch(S.parse(['--mode','plan',*args]))
    assert not output.exists() and not state['jobs'] and not state['checks']
    assert before=={p:P.sha(p) for p in repo.rglob('*') if p.is_file()}
    assert 'Read-only' in capsys.readouterr().out
    with pytest.raises(SystemExit):S.parse(['--mode','arm',*args])


def test_arm_only_one_cpu_job_freezes_orchestration_and_preserves_benchmark(setup):
    repo,bench,output,_,_,state=setup
    before={p:P.sha(p) for p in bench.rglob('*') if p.is_file()}
    intent=arm(setup)
    assert len(state['jobs'])==len(state['checks'])==1
    row,env=state['jobs'][0]
    assert '--dependency=afterok:22138551' in row
    assert '--time=120' in row and '--mem=16G' in row and '--cpus-per-task=4' in row
    assert not any(x.startswith(('--gres','--array')) for x in row)
    assert row[-2:]==[str(output/'source/sbatch/run_part_ev_benchmark_followup.sh'),str(output)]
    assert set(intent['source'])==set(F.FILES)
    (repo/'part_ev_state_production/engine.py').write_text('# subsequent pull')
    assert F.load(output)==intent
    assert before=={p:P.sha(p) for p in bench.rglob('*') if p.is_file()}
    assert not (output/'result.json').exists()


def test_three_independent_controllers(setup):
    repo,_,output,prod,_,state=setup
    arm(setup)
    for kind,job in [('triangles','22138557'),('support','22114968')]:
        bench=benchmark(repo,kind,job)
        S.launch(S.parse(['--mode','arm','--reviewed','--model',kind,'--benchmark',str(bench),
            '--output',str(output.parent/kind),'--production',str(prod.parent/kind)]))
    assert len(state['jobs'])==3
    for (row,_),job in zip(state['jobs'],['22138551','22138557','22114968']):
        assert '--dependency=afterok:'+job in row
        assert not any(x.startswith('--array') for x in row)


@pytest.mark.parametrize('override',[dict(JobName='other'),dict(UserId='other(999)'),
    dict(Partition='tigris'),dict(StdOut='/other.out'),dict(JobState='FAILED')])
def test_summary_identity_and_failure_guard(setup,override):
    *_,state=setup;state['job_override']=override
    with pytest.raises(ValueError):arm(setup)
    assert not state['jobs'] and not setup[2].exists()


def test_duplicate_benchmark_and_target_guard(setup):
    repo,bench,output,prod,args,state=setup
    arm(setup)
    args[args.index(str(output))]=str(output.parent/'duplicate')
    with pytest.raises(ValueError,match='already has'):S.launch(S.parse(['--mode','arm','--reviewed',*args]))
    other=benchmark(repo,'support','22114968')
    args[args.index(str(bench))]=str(other);args[args.index('readout')]='support'
    with pytest.raises(ValueError,match='target already'):S.launch(S.parse(['--mode','arm','--reviewed',*args]))
    assert len(state['jobs'])==1


@pytest.mark.parametrize('fault',['launch','source','ledger','controller'])
def test_tamper_guards(setup,fault):
    _,bench,output,_,_,_=setup
    arm(setup)
    path={'launch':bench/'launch.json','source':bench/'source/science.py',
          'ledger':bench/'submitted_job_ids.tsv','controller':output/'source'/F.FILES[0]}[fault]
    path.write_text('changed')
    with pytest.raises((ValueError,KeyError)):F.load(output)


@pytest.mark.parametrize('fault',['failed','incomplete','budget','seed','batch','ceiling','estimate'])
def test_admission_requires_full_recipe_pass_and_approved_ceilings(setup,fault):
    intent=arm(setup);e=admitted(intent)
    if fault=='failed':e['review']['passed']=False
    if fault=='incomplete':e['review']['complete']=False
    if fault=='budget':e['recipes']['ranger_full_v1']['total_steps']=200000
    if fault=='seed':e['task']['seed']=202
    if fault=='batch':e['recipes']['ranger_full_v1']['global_batch_size']=256
    if fault=='ceiling':intent['max_train_hours']=1
    if fault=='estimate':e['train_hours']=1
    with pytest.raises(ValueError):F.admit(intent,e)


@pytest.mark.parametrize('aged',[False,True])
def test_completed_or_aged_summary_requires_real_review_before_queue(setup,monkeypatch,aged):
    _,_,output,_,args,state=setup
    state.update(aged=aged,job_state='COMPLETED')
    def failed(_):raise ValueError('benchmark gates failed')
    monkeypatch.setattr(S.S,'inspect',failed)
    with pytest.raises(ValueError,match='gates'):arm(setup)
    assert not output.exists() and not state['jobs']
    intent=S.proposal(S.parse(['--mode','plan',*args]))
    monkeypatch.setattr(S.S,'inspect',lambda _:admitted(intent))
    arm(setup)
    assert not any(x.startswith('--dependency') for x in state['jobs'][0][0])


def test_retry_only_known_inactive_blocked_controller(setup):
    _,_,output,_,_,state=setup
    arm(setup)
    retry=S.parse(['--mode','retry','--reviewed','--output',str(output)])
    state['live']='1000'
    with pytest.raises(ValueError,match='still active'):S.launch(retry)
    state['live']=''
    P.write(output/'result.json',dict(status='blocked',error='headroom'))
    S.launch(retry)
    assert len(state['jobs'])==2
    P.write(output/'result.json',dict(status='needs_reconciliation'))
    with pytest.raises(ValueError,match='handoff'):S.launch(retry)
    assert len(state['jobs'])==2


def test_unknown_sbatch_is_not_retried(setup):
    _,_,output,_,_,state=setup;state['fail']=True
    with pytest.raises(subprocess.CalledProcessError):arm(setup)
    assert P.read(output/'submission_intent.json')['status']=='unknown'
    state['fail']=False
    with pytest.raises(ValueError,match='Unresolved'):
        S.launch(S.parse(['--mode','retry','--reviewed','--output',str(output)]))
    assert not state['jobs']


def test_worker_dispatches_once_and_uses_frozen_production_code(setup,monkeypatch):
    repo,_,output,prod,_,_=setup;intent=arm(setup);e=admitted(intent)
    monkeypatch.setattr(W.S,'inspect',lambda _:e)
    calls=[]
    def dispatch(args,**kw):
        calls.append((args,kw));prod.mkdir(parents=True)
        (prod/'submitted_job_ids.tsv').write_text('train\t2000\nevaluate\t2001\nreport\t2002\n')
    monkeypatch.setattr(W.S,'launch',dispatch)
    (repo/'part_ev_state_production/engine.py').write_text('# live changes irrelevant')
    row=W.run(output)
    assert row['status']=='submitted' and len(calls)==1
    assert calls[0][1]['frozen_source']['manifest']==intent['source']
    assert calls[0][1]['frozen_source']['path']==str(output/'source')
    assert calls[0][1]['expected_evidence']==e
    assert W.run(output)==row and len(calls)==1


@pytest.mark.parametrize('fault',['review','partial','pre_dispatch'])
def test_worker_never_resubmits_uncertain_production(setup,monkeypatch,fault):
    _,_,output,prod,_,_=setup;intent=arm(setup)
    calls=[]
    def inspect(_):
        if fault=='review':raise ValueError('benchmark failed')
        return admitted(intent)
    def dispatch(*a,**kw):
        calls.append(True)
        if fault=='partial':
            prod.mkdir(parents=True);(prod/'submission_intent.json').write_text('{"status":"unknown"}')
        raise ValueError('synthetic refusal')
    monkeypatch.setattr(W.S,'inspect',inspect);monkeypatch.setattr(W.S,'launch',dispatch)
    with pytest.raises(ValueError):W.run(output)
    assert bool(calls)==(fault!='review')
    assert P.read(output/'result.json')['status']==('needs_reconciliation' if fault=='partial' else 'blocked')
    if fault=='partial':
        with pytest.raises(ValueError,match='uncertain'):W.run(output)
        assert len(calls)==1


@pytest.mark.parametrize('drift',[False,True])
def test_worker_real_production_launcher_end_to_end(setup,monkeypatch,drift):
    repo,bench,output,prod,_,state=setup;intent=arm(setup)
    evidence=admitted(intent)
    evidence.update(benchmark_sha256='science-identity',storage=P.storage_budget(2349702))
    evidence['task']['run_id']='PAIR_READOUT_STATE_EV_LAYERWISE'
    calls=[]; inspections=[]
    def inspect(_):
        row=copy.deepcopy(evidence)
        if inspections and drift:row['train_hours']+=1
        inspections.append(True);return row
    def command(argv,**kw):
        if argv[0]=='squeue':return ''
        if argv[0]=='scontrol':return 'State=UP MaxTime=20-00:00:00'
        if argv[0]=='sbatch':
            if '--test-only' in argv:return 'Valid'
            calls.append(argv);return str(2000+len(calls)-1)
        raise AssertionError(argv)
    monkeypatch.setattr(W.S,'inspect',inspect)
    monkeypatch.setattr(W.S,'command',command)
    monkeypatch.setattr(W.S,'worker_query',lambda out,plan,phase:
                        ['train','evaluate','report'] if phase=='ready' else dict(verified=True))
    (repo/'part_ev_state_production/engine.py').write_text('# later checkout edit')
    if drift:
        with pytest.raises(ValueError,match='changed during'):W.run(output)
        assert not calls and not prod.exists()
        return
    row=W.run(output)
    assert row['status']=='submitted' and len(calls)==3
    assert '--gres=gpu:a100:2' in calls[0] and '--gres=gpu:a100:1' in calls[1]
    assert '--dependency=afterok:2000' in calls[1] and '--dependency=afterok:2001' in calls[2]
    assert not any(x.startswith('--dependency') for x in calls[0])
    assert P.inventory(prod/'source')=={k:intent['source'][k] for k in P.FILES}
    assert W.run(output)==row and len(calls)==3


def test_launcher_does_not_inherit_gpu_or_submission_options(setup,monkeypatch):
    monkeypatch.setenv('SBATCH_GRES','gpu:a100:9')
    monkeypatch.setenv('SLURM_JOB_ID','999')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    monkeypatch.setenv('RANK','5')
    arm(setup)
    env=setup[-1]['jobs'][0][1]
    assert not any(k.startswith(('SBATCH_','SLURM_')) for k in env)
    assert 'CUDA_VISIBLE_DEVICES' not in env and 'RANK' not in env


def test_no_automatic_pending_production_when_existing_manual_campaign(setup):
    _,_,output,prod,_,state=setup
    prod.mkdir(parents=True)
    with pytest.raises(FileExistsError,match='Production output'):arm(setup)
    assert not output.exists() and not state['jobs']


def test_dirty_source_and_low_storage_refuse(setup,monkeypatch):
    repo,_,output,_,_,state=setup
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=4*2**30))
    with pytest.raises(OSError):arm(setup)
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=23*2**30))
    (repo/F.FILES[0]).write_text('# dirty')
    with pytest.raises(ValueError,match='Commit/push/pull'):arm(setup)
    assert not output.exists() and not state['jobs']


def test_syntax():
    for name in F.FILES:
        if name.endswith('.py'):ast.parse((S.ROOT/name).read_text(),filename=name,feature_version=(3,10))
        elif name.endswith('.sh') and BASH:
            subprocess.run([BASH,'-n',shell_path(S.ROOT/name)],check=True,capture_output=True)

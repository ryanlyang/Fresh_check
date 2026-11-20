import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from part_ev_rts_tigris import protocol as P, submission as S, evidence as E
from tests.test_part_ev_rts_tigris_evidence import evidence


@pytest.mark.parametrize('kind',list(P.PACKAGES))
def test_prepared_reference_does_not_need_sporc_benchmark_pass(tmp_path,monkeypatch,kind):
    science=importlib.import_module(P.PACKAGES[kind]+'.protocol')
    parent=dict(task=science.TASK,parameters=P.COUNTS[kind],recipe=science.recipes()['ranger_full_v1'],content_hash='old')
    for name in ('campaign.json','launch.json','source_manifest.json','data_contents.json'):
        P.write(tmp_path/name,{'preserve':name})
    before=P.inventory(tmp_path)
    fake=SimpleNamespace(TASK=science.TASK,recipes=science.recipes,
        load=lambda path,check_environment:parent,verify_inventory_metadata=lambda *_:None,
        review=lambda *_:pytest.fail('Must not depend on SPORC GPU completion'))
    monkeypatch.setattr(P,'activate',lambda *_:fake)
    _,actual,donor=P.donor_contract(tmp_path,kind)
    assert actual==parent and len(donor['evidence'])==4 and P.inventory(tmp_path)==before
    P.write(tmp_path/'benchmark.json',{'passed':False})
    assert P.donor_contract(tmp_path,kind)[2]==donor
    P.write(tmp_path/'benchmark.json',{'passed':True})
    assert P.donor_contract(tmp_path,kind)[2]==donor
    parent['parameters']+=1
    with pytest.raises(ValueError):P.donor_contract(tmp_path,kind)


def admitted(tmp_path,monkeypatch,kind='readout'):
    root=tmp_path/'checkpoints/part_ev_rts_tigris'/kind
    root.mkdir(parents=True)
    science=importlib.import_module(P.PACKAGES[kind]+'.protocol')
    launch=dict(repo=str(tmp_path),output=str(root),python=sys.executable,model_kind=kind,
        account='reu-aisocial',auto_production=True,max_train_hours=480,max_evaluate_hours=24,
        admission={'donor':{'content_hash':kind}})
    spec=dict(content_hash='spec',model_kind=kind,task=science.TASK,recipe=science.recipes()['ranger_full_v1'],
        parameters=P.COUNTS[kind],model=science.CONFIG,environment={'arch':'aarch64'},inventory={},launch=launch,
        parameter_inventory_sha256='inventory')
    P.write(root/'launch.json',P.hashed(launch));P.write(root/'campaign.json',spec)
    P.write(root/'benchmark.json',P.hashed(evidence(kind)))
    monkeypatch.setattr(P,'load',lambda *_a,**_kw:spec)
    monkeypatch.setattr(P,'launch_config',lambda *_a,**_kw:launch)
    P.write(root/'summary.json',P.summary(root))
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *_:None))
    monkeypatch.setattr(P.shutil,'disk_usage',lambda _:SimpleNamespace(free=50*2**30))
    state=dict(jobs=[],test_only=[],fail=None,limit='20-00:00:00')
    def command(argv,**kw):
        if argv[0]=='scontrol':return 'State=UP MaxTime='+state['limit']
        assert argv[0]=='sbatch'
        if '--test-only' in argv:state['test_only'].append(argv);return 'ok'
        if len(state['jobs'])==state['fail']:raise OSError('submission response lost')
        state['jobs'].append(argv)
        return str(3000+len(state['jobs']))+';tigris'
    monkeypatch.setattr(S,'command',command)
    return root,launch,spec,state


@pytest.mark.parametrize('kind',list(P.PACKAGES))
def test_real_admission_dispatch_idempotent_and_budgeted(tmp_path,monkeypatch,kind):
    root,launch,spec,state=admitted(tmp_path,monkeypatch,kind)
    result=S.dispatch(root)
    assert result['status']=='submitted' and len(state['jobs'])==3
    plan=P.load_production(root)
    assert plan['parameters'][spec['task']['architecture']]==P.COUNTS[kind]
    assert plan['recipes'][spec['task']['recipe']]['total_steps']==1000000
    assert plan['policy']['validation_checks']==50 and plan['policy']['local_batch']==512
    assert P.checked(root/'production/campaign.json')==plan
    train,evaluate,report=state['jobs']
    assert '--gres=gpu:gh200:1' in train and '--gres=gpu:gh200:1' in evaluate
    assert not any(x.startswith('--gres') for x in report)
    assert '--dependency=afterok:3001' in evaluate and '--dependency=afterok:3002' in report
    assert '--time='+str(plan['train_hours']*60) in train
    assert '--mem=128G' in train
    assert S.dispatch(root)['status']=='already_submitted' and len(state['jobs'])==3
    assert P.required_storage(tmp_path,root)>=P.RESERVE_BYTES


@pytest.mark.parametrize('fault',('failed','missing','stale','runtime','optout','cap','partition','storage'))
def test_no_failed_or_overbudget_handoff(tmp_path,monkeypatch,fault):
    root,launch,spec,state=admitted(tmp_path,monkeypatch)
    if fault=='failed':
        row=evidence('readout');row['passed']=False;P.write(root/'benchmark.json',P.hashed(row))
    if fault=='missing':(root/'benchmark.json').unlink()
    if fault=='stale':P.write(root/'summary.json',{'passed':True})
    if fault=='runtime':
        row=evidence('readout');row['runtime']['world_size']=2;P.write(root/'benchmark.json',P.hashed(row))
    if fault=='optout':launch['auto_production']=False
    if fault=='cap':launch['max_train_hours']=1
    if fault=='partition':state['limit']='1-00:00:00'
    if fault=='storage':monkeypatch.setattr(P.shutil,'disk_usage',lambda _:SimpleNamespace(free=1))
    with pytest.raises((ValueError,OSError)):S.dispatch(root)
    assert not state['jobs'] and not (root/'production.json').exists()


@pytest.mark.parametrize('failure_index',(0,1,2))
def test_partial_dispatch_never_duplicates(tmp_path,monkeypatch,failure_index):
    root,_,_,state=admitted(tmp_path,monkeypatch)
    state['fail']=failure_index
    with pytest.raises(OSError):S.dispatch(root)
    assert P.read(root/'production_intent.json')['status']=='unknown'
    state['fail']=None
    with pytest.raises(ValueError,match='Partial'):S.dispatch(root)
    assert len(state['jobs'])==failure_index


def test_different_output_cannot_dispatch_same_reference_twice(tmp_path,monkeypatch):
    root,launch,_,state=admitted(tmp_path,monkeypatch)
    owner=root.parent/(P.digest(dict(model='readout',donor='readout'))+'.production-owner.json')
    P.write(owner,dict(output=str(root.parent/'already_started')))
    with pytest.raises(ValueError,match='already owns'):S.dispatch(root)
    assert not state['jobs']


def test_production_rejects_changed_native_evidence(tmp_path,monkeypatch):
    root,_,_,_=admitted(tmp_path,monkeypatch)
    S.dispatch(root)
    row=evidence('readout');row['checkpoint_seconds']=2.
    P.write(root/'benchmark.json',P.hashed(row))
    P.write(root/'summary.json',P.summary(root))
    with pytest.raises(ValueError,match='changed'):P.load_production(root)


@pytest.mark.parametrize('kind',list(P.PACKAGES))
def test_prepare_load_freezes_independent_arm_environment(tmp_path,monkeypatch,kind):
    science=importlib.import_module(P.PACKAGES[kind]+'.protocol')
    parent=dict(task=science.TASK,parameters=P.COUNTS[kind],recipe=science.recipes()['ranger_full_v1'],
        model=science.CONFIG,inventory={'unchanged':'data'},parameter_inventory_sha256='inventory')
    donor=dict(path='read-only-sporc',content_hash='donor',evidence={'campaign.json':'sha'})
    env={'architecture':'aarch64','torch_cuda':'12.6','weaver_sources':{'model':'same'}}
    launch=dict(model_kind=kind,benchmark='read-only-sporc',admission=dict(donor=donor,environment=copy.deepcopy(env),
        task=parent['task'],parameters=parent['parameters']))
    monkeypatch.setattr(P,'launch_config',lambda *_:launch)
    monkeypatch.setattr(P,'donor_contract',lambda *_:(science,parent,donor))
    monkeypatch.setattr(P,'runtime_environment',lambda *_:env)
    monkeypatch.setattr(P,'check_storage',lambda *_:None)
    prepared=P.prepare(tmp_path)
    assert prepared==P.load(tmp_path) and prepared['environment']['architecture']=='aarch64'
    assert prepared['recipe']['total_steps']==1000000 and prepared['recipe']['epochs']==50
    if kind=='support':assert prepared['parameter_inventory_sha256']=='inventory'
    with pytest.raises(FileExistsError):P.prepare(tmp_path)
    env['torch_cuda']='changed'
    with pytest.raises(ValueError,match='environment changed'):P.load(tmp_path)


@pytest.mark.parametrize('change',('arch','cuda','weaver'))
def test_rejects_wrong_arm_runtime(monkeypatch,change):
    from part_ev_fullscale import model as M
    monkeypatch.setattr(M,'reference_metadata',lambda:None)
    actual=dict(architecture='aarch64',torch_cuda='12.6',weaver_sources={'model':'same'})
    parent=dict(environment=dict(weaver_sources={'model':'same'}))
    if change=='arch':actual['architecture']='x86_64'
    if change=='cuda':actual['torch_cuda']=None
    if change=='weaver':actual['weaver_sources']['model']='different'
    with pytest.raises(ValueError):P.runtime_environment(SimpleNamespace(runtime_environment=lambda:actual),parent)

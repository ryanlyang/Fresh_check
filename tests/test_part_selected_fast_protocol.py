import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from part_selected_fast import protocol as P
from part_selected_fast import trees
from part_selected_screen.normalization import identity_normalization
from tests.test_part_selected_tigris import benchmark_fixture as old_fixture


def launch_options():
    return dict(cpus='12',tree_threads='4',memory_gib='128',cache_bytes=0,compile_region=False,
                train_hours='168',evaluate_hours='12')


def test_contract_and_prepared_artifact_drift(tmp_path,monkeypatch):
    parent=tmp_path/'reference'; output=tmp_path/'fast'
    parent.mkdir(); output.mkdir()
    P.write_json(parent/'experiment_spec.json',{'content_hash':'old'})
    old=dict(content_hash='old',precision=P.S.POLICY,recipes={'ranger_short_v1':P.recipes()['ranger_short_v1']},
             environment={'architecture':'aarch64'},inventory={},donor={},layerwise_donor={},
             input_reference={},hardware_transfer={},environment_differences_from_donor=[])
    old['inputs']={}
    for name in ('inputs/selected.yaml','inputs/normalization.json','inputs/native_stream.json'):
        P.write_json(output/name,{'fixture':name})
        old['inputs'][name]=P.file_sha256(output/name)
    P.write_json(output/'backend/manifest.json',{'fixture':True})
    P.write_json(output/'backend/parity.json',dict(passed=True,exact=True,invalid_input_rejected=True,
                 cases=[dict(particles=n) for n in (1,2,11,32,64,128)]))
    P.write_json(output/'launch.json',dict(reference=str(parent),**launch_options()))
    monkeypatch.setattr(P,'reference',lambda *a,**kw:old)
    monkeypatch.setattr(P,'runtime_environment',lambda:old['environment'])
    spec=P.create_spec(output); P.write_json(output/'experiment_spec.json',spec)
    assert P.load_campaign(output)==spec
    for field in ('execution','parameters','parity','source','recipes'):
        bad=copy.deepcopy(spec); bad[field]={}; bad.pop('content_hash')
        P.write_json(output/'experiment_spec.json',P.hashed(bad))
        with pytest.raises(ValueError): P.load_campaign(output)
    P.write_json(output/'experiment_spec.json',spec)
    P.write_json(output/'backend/parity.json',dict(passed=False))
    with pytest.raises(ValueError,match='artifact changed'): P.load_campaign(output)


def fixture_gate(tmp_path):
    spec,rows=old_fixture(tmp_path)
    spec['launch'].update(launch_options()); spec['execution']=P.execution(spec['launch'])
    spec['recipes']={'ranger_short_v1':P.recipes()['ranger_short_v1']}
    spec['input_reference']['fingerprints']={'0':'same','20':'same'}
    for row in rows:
        row['input_fingerprints']=spec['input_reference']['fingerprints']
        row['runtime']['execution']=spec['execution']
        row['initial_parity']['fast_execution']=dict(passed=True,policy=P.POLICY)
        row.update(measured_steps=5000,windows=[{}]*5,host_peak_bytes=8*2**30,gpu_peak_reserved_bytes=30*2**30)
        P.write_json(tmp_path/'benchmarks'/f"{row['task']['run_id']}.json",row)
        measurements={v:[dict(trial=i,steps=200,seconds=s,input_fingerprints=spec['input_reference']['fingerprints'])
                           for i in range(3)] for v,s in [('reference',100.),('candidate',50.)]}
        measurements['reference'][0]['validation_cold']={'seconds':2.,'events':4096}
        profile=dict(schema='selected_fast_profile_v1',task=row['task'],campaign_sha256='selected',
                     execution=spec['execution'],runtime=row['runtime'],passed=True,inputs_match=True,
                     host_peak_bytes=10*2**30,conservative_speedup=2.,measurements=measurements)
        P.write_json(tmp_path/'profiles'/f"{row['task']['run_id']}.json",P.hashed(profile))
    return spec,rows


@pytest.mark.parametrize('change',['missing_profile','failed_profile','fake_speed','slow','incomplete_profile',
                                  'missing_parity','execution','windows','memory','reserved','runtime','old_schema'])
def test_four_model_gates_fail_closed(tmp_path,change):
    spec,rows=fixture_gate(tmp_path)
    assert len(P.require_approval(tmp_path,spec))==4
    task=rows[-1]['task']; path=tmp_path/'profiles'/f"{task['run_id']}.json"
    profile=P.read_json(path); row=rows[-1]
    if change=='missing_profile': path.unlink()
    elif change=='failed_profile': profile['passed']=False
    elif change=='fake_speed': profile['conservative_speedup']=3
    elif change=='slow': profile['conservative_speedup']=1.01
    elif change=='incomplete_profile': profile['measurements']['candidate'].pop()
    elif change=='missing_parity': row['initial_parity'].pop('fast_execution')
    elif change=='execution': row['runtime']['execution']={}
    elif change=='windows': row['windows']=[]
    elif change=='memory': row['host_peak_bytes']=128*2**30
    elif change=='reserved': row['gpu_peak_reserved_bytes']=100*2**30
    elif change=='runtime': row['runtime']['gpu']='A100'
    elif change=='old_schema': profile['schema']='old'
    if path.exists():
        profile.pop('content_hash'); P.write_json(path,P.hashed(profile))
    P.write_json(tmp_path/'benchmarks'/f"{task['run_id']}.json",row)
    with pytest.raises((ValueError,FileNotFoundError)): P.require_approval(tmp_path,spec)


def test_resources_and_source_manifest():
    config=launch_options(); assert P.execution(config)['cache_bytes']==0
    for key,value in [('cpus','4'),('tree_threads','0'),('cache_bytes',1000*2**30)]:
        with pytest.raises(ValueError): P.execution(dict(config,**{key:value}))
    source=P.source_manifest(Path(__file__).resolve().parents[1])
    assert 'scripts/submit_part_selected_fast.py' in source
    assert 'part_selected_fast/csrc/tree_core.h' in source
    assert all(t['seed']==101 for t in P.TASKS)
    assert set(P.MATCHED.values())==set(P.SHARED+P.LAYERWISE)


def test_factory_does_not_change_other_engine_or_model_bindings(tmp_path,monkeypatch):
    from part_selected_fast import engine as E
    import part_screen_common.engine as common
    import part_selected_tigris.engine as old
    old_factory,old_loader=old.engine_for,common.build_loader
    normal=identity_normalization()
    monkeypatch.setattr(P,'load_campaign',lambda *a,**k:dict(reference={'path':str(tmp_path)},
                        content_hash='fast',execution=P.execution(launch_options())))
    monkeypatch.setattr(P,'checked_json',lambda *a,**kw:normal)
    monkeypatch.setattr(E,'reference_backend',lambda *a:'reference_for_tests')
    monkeypatch.setattr(E,'load_backend',lambda *a:None)
    monkeypatch.setattr(E,'PackedTrees',lambda *a,**kw:trees.PackedTrees(None,oracle='reference_for_tests'))
    engine=E.engine_for(tmp_path)
    assert engine.P is P and old.engine_for is old_factory and common.build_loader is old_loader
    assert type(engine).train.__globals__['build_loader'] is not old_loader
    assert type(engine).train.__globals__['build_loader'].keywords['config']==tmp_path/'inputs/selected.yaml'
    assert type(engine).benchmark is not old.benchmark


def test_checkpoint_rejects_old_campaign_and_cache_not_serialized(tmp_path):
    from tests.test_part_selected_fast_model import factory
    from part_screen_common.engine import ScreenEngine
    from part_ev_fullscale.optimizer import build_optimizer
    _,model=factory('selected_shared')
    engine=ScreenEngine(P,None,None,None)
    optimizer=build_optimizer(model,P.recipes()['ranger_short_v1'])
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    spec={'content_hash':'fast'}; runtime={'execution':'fast'}; task=P.TASKS[0]
    path=tmp_path/'own.pt'; engine.save(path,model,optimizer,scaler,{'task':task,'step':3},spec,runtime)
    saved=torch.load(path,weights_only=False)
    assert all('cache' not in k and 'fast_provider' not in k for k in saved['model'])
    state,rng=engine.load(path,model,optimizer,scaler,spec,runtime,task)
    assert state['step']==3 and rng['trimmers']
    with pytest.raises(ValueError,match='another task/campaign/runtime'):
        engine.load(path,model,optimizer,scaler,{'content_hash':'old'},runtime,task)


def test_profile_summary_uses_conservative_repeated_ranges():
    from scripts.profile_part_selected_fast import summarize
    result=summarize([{'seconds':x} for x in (9,10,11)],[{'seconds':x} for x in (4,5,6)])
    assert result['conservative_speedup']==1.5 and result['median_speedup']==2
    with pytest.raises(ValueError): summarize([],[])
    with pytest.raises(ValueError): summarize([{'seconds':0}]*3,[{'seconds':1}]*3)

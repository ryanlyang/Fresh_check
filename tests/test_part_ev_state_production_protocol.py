import copy
import importlib
import math
from pathlib import Path
import sys

import pytest

from part_ev_state_production import protocol as P


@pytest.fixture(params=['bias','multiply','memory'])
def admitted(request,tmp_path,monkeypatch):
    kind=request.param
    science=importlib.import_module(P.PACKAGES[kind]+'.protocol')
    fixture=importlib.import_module('tests.test_'+P.PACKAGES[kind]+'_protocol')
    spec=dict(content_hash='spec',task=science.TASK,recipe=science.recipes()['ranger_full_v1'],
        parameters=science.PARAMETERS,model=science.CONFIG,environment={'torch':'pinned'},
        inventory={'splits':{}},launch=dict(nccl_environment={},python=sys.executable,account='reu-aisocial'))
    monkeypatch.setattr(P,'activate',lambda b,k:science)
    monkeypatch.setattr(science,'load',lambda *a,**kw:spec)
    monkeypatch.setattr(science,'verify_inventory_metadata',lambda _:None)
    for name in ('campaign.json','launch.json','source_manifest.json','data_contents.json'):
        P.write(tmp_path/name,{'frozen':name})
    P.write(tmp_path/'benchmark.json',P.hashed(fixture.evidence()))
    P.write(tmp_path/'summary.json',science.review(tmp_path))
    return kind,tmp_path,science,spec


def test_reuses_real_benchmark_review_without_new_preflight_or_scientific_changes(admitted):
    kind,root,science,old=admitted
    before={p:P.sha(p) for p in root.iterdir()}
    row=P.inspect(root,kind)
    assert row['task']==science.TASK and row['model_config']==science.CONFIG
    assert row['recipes']=={science.TASK['recipe']:old['recipe']}
    assert row['parameters'][science.TASK['architecture']]==science.PARAMETERS
    assert row['train_hours']==science.review(root)['recommended_train_hours']
    assert row['evaluate_hours']==science.review(root)['recommended_evaluate_hours']
    assert row['expected_gpu']['name']=='NVIDIA A100'
    assert before=={p:P.sha(p) for p in root.iterdir()}
    recipe=row['recipes'][row['task']['recipe']]
    assert recipe['total_steps']==1000000 and recipe['epochs']==50
    assert recipe['steps_per_epoch']==20000 and recipe['validation_events_per_epoch']==1280000
    assert P.POLICY['local_batch']*P.POLICY['world_size']==512
    assert not P.POLICY['test_data'] and not P.POLICY['persistent_pair_cache']


@pytest.mark.parametrize('fault',['incomplete','failed','short_timing','restart','memory','precision','summary'])
def test_no_admission_from_slurm_completion_or_partial_evidence(admitted,fault):
    kind,root,_,_=admitted
    row=P.read(root/'benchmark.json');row.pop('content_hash')
    if fault=='incomplete':row['complete']=False
    if fault=='failed':row['passed']=False
    if fault=='short_timing':row['stream']['measured_steps']=20
    if fault=='restart':row['restart']['full_state']=False
    if fault=='memory':row['gpus'][0]['peak_allocated']=40*2**30
    if fault=='precision':row['validation']['precision']['passed']=False
    P.write(root/'benchmark.json',P.hashed(row))
    if fault=='summary':P.write(root/'summary.json',{'passed':True,'complete':True})
    with pytest.raises(ValueError):P.inspect(root,kind)


@pytest.mark.parametrize('fault',['source','evidence','policy','recipe','output'])
def test_frozen_production_admission_rechecked(admitted,tmp_path,fault):
    kind,benchmark,_,_=admitted
    evidence=P.inspect(benchmark,kind)
    output=tmp_path/'production'
    for name in P.FILES:
        path=output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen')
    plan=P.hashed(dict(schema=P.SCHEMA,authorizes_training=True,output=str(output.resolve()),
        policy=P.POLICY,source=P.inventory(output/'source'),**evidence))
    P.write(output/'campaign.json',plan)
    assert P.load(output)==plan
    if fault=='source':(output/'source'/P.FILES[0]).write_text('# changed')
    if fault=='evidence':P.write(benchmark/'data_contents.json',{'changed':True})
    if fault in ('policy','recipe','output'):
        row=copy.deepcopy(plan);row.pop('content_hash')
        if fault=='policy':row['policy']['world_size']=1
        if fault=='recipe':row['recipes'][row['task']['recipe']]['total_steps']=200000
        if fault=='output':row['output']=str(tmp_path)
        P.write(output/'campaign.json',P.hashed(row))
    with pytest.raises(ValueError):P.load(output)


def test_real_reported_walltimes_preserved():
    for train,val,ckpt,recommend,evaluate in (
        (205.8833917724971,30.603806885972315,.09225429535843432,308,4),
        (157.4573128024996,16.017702139583385,.14510114207863808,226,2),
        (251.0837766313893,34.49662491472231,.09106525503098965,372,4)):
        row=dict(passed=True,complete=True,authorizes_training=False,projected_train_hours=train,
            projected_validation_hours=val,projected_checkpoint_hours=ckpt,
            recommended_train_hours=recommend,recommended_evaluate_hours=evaluate)
        P.validate_review(row)
        assert recommend==math.ceil(1.3*(train+val+ckpt))
        row['recommended_train_hours']-=1
        with pytest.raises(ValueError):P.validate_review(row)


@pytest.mark.parametrize('n',[2329510,2343846,2364294])
def test_storage_accounts_for_all_candidates_and_atomic_restarts(n):
    row=P.storage_budget(n)
    assert row['weights_bytes']>=51*n*4
    assert row['restart_bytes']>=3*n*20
    assert row['predictions_bytes']>=5000000*44
    assert row['reserve_bytes']==4*2**30
    assert 6*2**30<row['required_free_bytes']<8*2**30


@pytest.mark.parametrize('fault',['history','boundaries','writers','reserved'])
def test_memory_production_keeps_memory_specific_benchmark_gates(tmp_path,monkeypatch,fault):
    from part_ev_pair_memory import protocol as memory
    from tests.test_part_ev_pair_memory_protocol import evidence,setup_review
    row=evidence()
    if fault=='history':row['correctness'][0]['checks'][-2]['final_reader_first_writer_gradient']=False
    if fault=='boundaries':row['correctness'][0]['checks'][-1]['overflow_rejected']=False
    if fault=='writers':row['correctness'][0]['delayed_activation']['all_writers_and_gates_updated']=False
    if fault=='reserved':row['gpus'][0]['peak_reserved']=41*2**30
    setup_review(tmp_path,monkeypatch,row)
    monkeypatch.setattr(P,'activate',lambda *a:memory)
    monkeypatch.setattr(memory,'verify_inventory_metadata',lambda _:None)
    with pytest.raises(ValueError):P.inspect(tmp_path,'memory')

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from part_ev_state_bias import protocol as P
from scripts.run_part_ev_state_bias import main


def precision(dtype):
    return dict(passed=True,expected_activation_dtype=dtype,parameter_dtypes=['torch.float32'],
                observed_activations={'projection':dtype})


def evidence():
    from part_ev_message_study.parity import POLICY
    comparison=dict(passed=True,maximum_absolute_error=0.)
    initialization=dict(passed=True,policy=POLICY,device_type='cuda',events=8,
        optimizer_steps=0,initial_backbone_exact=True,modes={mode:dict(
            warmup=[dict(pair=i+1,rng_matches=True,comparison=comparison) for i in range(2)],
            measured=[dict(order=o,rng_matches=True,dtype='torch.float16' if mode=='train_fp16' else 'torch.float32',
                comparison=comparison,reference_repeat=comparison,new_repeat=comparison)
                for o in POLICY['measured_orders']]) for mode in POLICY['modes']})
    def measured(count):
        windows=[dict(steps=min(P.POLICY['window'],count-start),seconds=min(P.POLICY['window'],count-start)/2)
                 for start in range(0,count,P.POLICY['window'])]
        rank=dict(windows=windows,precision=precision('torch.float16'),
                  parameters_sha256='same',scaler=32768,skips=0,lookahead_counter=2)
        return dict(measured_steps=count,healthy=True,seconds=count/2,jets_per_second=1024,
                    skips=0,ranks=[copy.deepcopy(rank),copy.deepcopy(rank)])
    return dict(passed=True,complete=True,campaign_sha256='spec',stream=measured(4200),fixed=measured(40),
        correctness=[dict(passed=True,rank=r,tolerances=P.PARITY,initialization=initialization,
            checks=[dict(kind=k,passed=True) for k in ('zero_identity','active_chunk','active_SyncBN_DDP')]) for r in (0,1)],
        restart=dict(passed=True,updates=7,ranks=2,full_state=True),finite_state=True,
        gpus=[dict(rank=r,name='NVIDIA A100',total_memory=40*1024**3,
                   peak_allocated=25*1024**3,peak_reserved=29*1024**3) for r in (0,1)],
        validation=dict(events=128000,seconds=30,precision=precision('torch.float32')),
        diagnostics=[dict(events=8,particles=17,precision='fp32',dropout=False,optimizer_steps=0,
            layers=[dict(**{k:[.1]*8 for k in P.DIAGNOSTIC_HEAD_FIELDS},new_to_ev_norm=.1,
                         correction_to_native_norm=.1,state_output_parameter_norm=.1) for _ in range(8)]) for _ in range(2)],
        inference=dict(batch_size=256,particles=128,batches=20,seconds=4.,precision='fp32'),
        checkpoint_seconds=1.,runtime=dict(world_size=2,local_batch=256,global_batch=512,workers_per_rank=1,
                        batchnorm='SyncBatchNorm',tf32=False,cudnn_benchmark=False,nccl_environment={}))


def setup_review(tmp_path,monkeypatch,row):
    P.write_json(tmp_path/'campaign.json',{})
    spec=dict(content_hash='spec',recipe=P.recipes()['ranger_full_v1'],launch=dict(nccl_environment={}))
    monkeypatch.setattr(P,'load',lambda *a,**kw:spec)
    row['content_hash']=P.digest(row)
    P.write_json(tmp_path/'benchmark.json',row)


def test_complete_review_projects_all_50_validations_but_never_approves(tmp_path,monkeypatch):
    setup_review(tmp_path,monkeypatch,evidence())
    row=P.review(tmp_path)
    assert row['passed'] and not row['authorizes_training']
    assert row['projected_train_hours']==pytest.approx(500000/3600)
    assert row['projected_validation_hours']==pytest.approx(30*50*10/3600)
    assert row['fixed128_inference_jets_per_second']==1280
    assert row['parameters']==2329510
    assert row['recommended_train_hours']>row['projected_train_hours']


@pytest.mark.parametrize('fault', ['rank','precision','checks','recipe','memory','windows','skips','restart','inference','diagnostics','hash'])
def test_review_fails_closed(tmp_path,monkeypatch,fault):
    row=evidence()
    if fault=='rank': row['stream']['ranks'][1]['parameters_sha256']='different'
    if fault=='precision': row['validation']['precision']['passed']=False
    if fault=='checks': row['correctness'][0]['checks'].pop()
    if fault=='recipe': row['runtime']['global_batch']=1024
    if fault=='memory': row['gpus'][0]['peak_allocated']=39*1024**3
    if fault=='windows': row['stream']['ranks'][0]['windows'][0]['steps']=100
    if fault=='skips': row['fixed']['skips']=2
    if fault=='restart': row['restart']['full_state']=False
    if fault=='inference': row['inference']['seconds']=0
    if fault=='diagnostics': row['diagnostics'][0]['layers'].pop()
    setup_review(tmp_path,monkeypatch,row)
    if fault=='hash':
        row['passed']=False
        P.write_json(tmp_path/'benchmark.json',row)
    with pytest.raises(ValueError): P.review(tmp_path)


def test_missing_and_failed_never_authorize_and_summary_fails(tmp_path,monkeypatch):
    assert P.review(tmp_path)['missing']==['campaign.json','benchmark.json']
    assert main(['--phase','summary','--output',str(tmp_path)])==1
    row=evidence(); row.update(passed=False,complete=False,error='OOM')
    setup_review(tmp_path,monkeypatch,row)
    assert not P.review(tmp_path)['passed']
    with pytest.raises(SystemExit): main(['--phase','train','--output',str(tmp_path)])


def test_snapshot_rejects_modified_missing_or_extra_files(tmp_path):
    source=tmp_path/'source';source.mkdir()
    file=source/'code.py'; file.write_text('original')
    P.write_json(tmp_path/'source_manifest.json',{'code.py':P.file_sha256(file)})
    original=P.verify_snapshot(tmp_path)
    file.write_text('modified')
    with pytest.raises(ValueError,match='source'): P.verify_snapshot(tmp_path)
    file.write_text('original'); (source/'extra.py').write_text('extra')
    with pytest.raises(ValueError,match='source'): P.verify_snapshot(tmp_path)
    assert original


def test_environment_ignores_lazy_import_paths_not_real_package_changes(monkeypatch):
    calls=[]
    monkeypatch.setattr(P,'environment',lambda:dict(packages={'torch':'pinned'}))
    def distributions(*,path):
        calls.append(path)
        return [SimpleNamespace(metadata={'Name':'Package'},version='1')]
    monkeypatch.setattr(P.importlib.metadata,'distributions',distributions)
    original=P.runtime_environment()
    monkeypatch.setattr(sys,'path',sys.path+['/temporary/setuptools/_vendor'])
    assert P.runtime_environment()==original
    assert all('/temporary/setuptools/_vendor' not in paths for paths in calls)


def test_prepare_is_independent_preserves_reference_and_freezes_data(tmp_path,monkeypatch):
    output=tmp_path/'new';output.mkdir(); donor=tmp_path/'old';donor.mkdir()
    P.write_json(donor/'campaign.json',{'old':'unchanged'})
    (output/'source').mkdir(); (output/'source'/'a.py').write_text('# source')
    P.write_json(output/'source_manifest.json',{'a.py':P.file_sha256(output/'source'/'a.py')})
    P.write_json(output/'launch.json',dict(reference=str(donor),nccl_environment={}))
    old=dict(inventory={'same':'inventory'},environment={'same':'env'})
    monkeypatch.setattr(P,'reference',lambda *a,**k:old)
    monkeypatch.setattr(P,'runtime_environment',lambda:old['environment'])
    monkeypatch.setattr(P,'verify_inventory_metadata',lambda spec:None)
    monkeypatch.setattr(P,'verify_data_contents',lambda inv:{'verified':'bytes'})
    monkeypatch.setattr(P.shutil,'disk_usage',lambda path:SimpleNamespace(free=8*1024**3))
    before=(donor/'campaign.json').read_bytes()
    spec=P.prepare(output)
    assert spec['recipe']['total_steps']==1_000_000 and spec['recipe']['epochs']==50
    assert not spec['authorizes_training'] and not spec['historical_control_reuse_authorized']
    assert P.load(output)==spec
    assert (donor/'campaign.json').read_bytes()==before and len(list(donor.iterdir()))==1
    with pytest.raises(FileExistsError): P.prepare(output)


@pytest.mark.parametrize('fault',[None,'runtime','recipe','source','environment','config','population'])
def test_reference_contract_is_checked_without_mutating_or_loading_weights(tmp_path,monkeypatch,fault):
    key='particle_transformer/data/JetClass/JetClass_full.yaml'
    config=tmp_path/'source'/key;config.parent.mkdir(parents=True)
    config.write_bytes(P.DATA_CONFIG.read_bytes())
    (tmp_path/'source'/'model.py').write_text('# frozen old model')
    inventory=dict(splits={s:dict(files=[dict(num_entries=100000) for _ in range(n)])
                           for s,n in (('train',1000),('val',50))})
    if fault=='population':inventory['splits']['val']['files'].pop()
    inventory['inventory_sha256']=P.digest(inventory)
    row=dict(schema=P.REFERENCE_SCHEMA,task=dict(architecture='state_layerwise',seed=101),
        policy=dict(world_size=2,local_batch=256,global_batch=512,train_workers_per_rank=1,
                    validation_workers=2,validation_batch=512,batchnorm='SyncBatchNorm'),
        recipes=P.recipes(),parameters={'state_layerwise':2327462},environment={'env':'pinned'},
        inventory=inventory,source={key:P.file_sha256(config),'model.py':P.file_sha256(tmp_path/'source/model.py')})
    if fault=='runtime':row['policy']['train_workers_per_rank']=2
    if fault=='recipe':row['recipes']['ranger_full_v1']['total_steps']=200000
    if fault=='config':
        config.write_text('changed physics preprocessing')
        row['source'][key]=P.file_sha256(config)
    row['content_hash']=P.digest(row)
    P.write_json(tmp_path/'campaign.json',row)
    if fault=='source':(tmp_path/'source/model.py').write_text('mutated after freezing')
    monkeypatch.setattr(P,'runtime_environment',lambda:{'env':'changed' if fault=='environment' else 'pinned'})
    before={p.relative_to(tmp_path):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    if fault is None:assert P.reference(tmp_path)==row
    else:
        with pytest.raises(ValueError):P.reference(tmp_path)
    assert before=={p.relative_to(tmp_path):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}

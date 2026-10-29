import copy
import json
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from part_ev_layerwise_ddp import protocol as P, engine as E
from part_ev_state_compare import protocol as C


def scaling_fixture():
    precision=dict(passed=True,expected_activation_dtype='torch.float16')
    sections={}
    for key,count in (('stream',4200),('fixed',40)):
        sections[key]=dict(measured_steps=count,seconds=count/2,jets_per_second=1024,
            healthy=True,skips=0,ranks=[dict(parameters_sha256='same',scaler=32768,skips=0,
                lookahead_counter=0,precision=precision,windows=[dict(steps=count,seconds=count/2)])]*2)
    return dict(completed=True,numerically_healthy=True,memory_headroom_passed=True,world_size=2,
        plan_sha256='plan',**sections,parity=[dict(passed=True,nonzero_state=True)]*2,
        gpus=[dict(rank=i,name='A100',total_memory=40*1024**3,peak_allocated=18*1024**3) for i in range(2)],
        runtime=dict(global_batch=512,local_batch=256,workers_per_rank=1,batchnorm='SyncBatchNorm',tf32=False,nccl_environment={}))


def spec_fixture():
    return dict(content_hash='production',task=P.TASK,tasks=[P.TASK],policy=copy.deepcopy(P.POLICY),
                recipes={P.TASK['recipe']:C.recipes()['ranger_full_v1']},parameters={'state_layerwise':2327462},
                expected_gpu=dict(name='A100',total_memory=40*1024**3,nccl_environment={}))


def test_scope_recipe_and_historical_sources_unchanged():
    assert P.TASK==C.TASKS[0] and P.TASK['seed']==101
    assert P.POLICY['world_size']*P.POLICY['local_batch']==512
    assert P.POLICY['train_workers_per_rank']==1
    recipe=spec_fixture()['recipes'][P.TASK['recipe']]
    assert recipe['epochs']==50 and recipe['total_steps']==1_000_000
    assert recipe['steps_per_epoch']==20_000 and recipe['validation_events_per_epoch']==1_280_000
    assert not any('layerwise_ddp' in name for name in C.source_manifest(P.ROOT))
    assert not any('layerwise_ddp' in name for name in P.G.source_manifest())


def test_only_successful_two_gpu_measurement_admitted():
    assert P.validate_scaling(scaling_fixture(),dict(content_hash='plan'))['name']=='A100'


@pytest.mark.parametrize('change',[
    lambda r:r.update(world_size=1), lambda r:r.update(completed=False),
    lambda r:r.update(numerically_healthy=False),lambda r:r.update(memory_headroom_passed=False),
    lambda r:r['stream'].update(measured_steps=10),lambda r:r['fixed'].update(healthy=False),
    lambda r:r['runtime'].update(local_batch=512),lambda r:r['runtime'].update(batchnorm='native'),
    lambda r:r['gpus'][0].update(peak_allocated=39*1024**3),
    lambda r:r['parity'][0].update(passed=False),
])
def test_bad_scaling_evidence_refused(change):
    row=copy.deepcopy(scaling_fixture());change(row)
    with pytest.raises(ValueError): P.validate_scaling(row,dict(content_hash='plan'))


def test_estimate_includes_all_validation_checkpoint_overhead_and_margin():
    spec=spec_fixture()
    preflight=dict(validation=dict(events=128000,seconds=25),checkpoint_seconds=.5)
    estimate=P.recommendation(spec,preflight,scaling_fixture())
    # 1M updates at .5s, 50*1.28M validation jets at 128k/25s, 1,152 writes.
    import math
    assert estimate['train_hours']==math.ceil(1.3*(500000+50*10*25+1152*.5)/3600)
    assert estimate['evaluate_hours']==5
    assert estimate['train_hours']>estimate['projected_train_compute_hours']


def test_prepare_is_new_directory_only_and_never_writes_old_evidence(tmp_path,monkeypatch):
    scaling=tmp_path/'scaling';scaling.mkdir()
    parent=tmp_path/'parent';parent.mkdir()
    plan=dict(content_hash='plan',parent=str(parent),recipe=spec_fixture()['recipes'][P.TASK['recipe']])
    original=dict(environment={},inventory={'frozen':True})
    row=scaling_fixture();row['content_hash']=P.digest(row)
    P.write_json(scaling/'gpus_2.json',row);P.write_json(scaling/'plan.json',plan)
    before={p.name:p.read_bytes() for p in scaling.iterdir()}
    monkeypatch.setattr(P.G,'load',lambda *a,**k:(plan,original))
    monkeypatch.setattr(P.G.C,'verify_inventory_metadata',lambda _:None)
    for bad in (scaling,scaling/'child',parent,parent/'child',tmp_path):
        with pytest.raises(ValueError): P.prepare(bad,scaling)
    spec=P.prepare(tmp_path/'production',scaling)
    assert spec['tasks']==[P.TASK] and spec['recipes'][P.TASK['recipe']]==plan['recipe']
    assert {p.name:p.read_bytes() for p in scaling.iterdir()}==before
    assert not list(parent.iterdir())
    with pytest.raises(FileExistsError): P.prepare(tmp_path/'production',scaling)


def preflight_fixture(spec):
    return dict(campaign_sha256=spec['content_hash'],runtime=P.expected_runtime(spec),passed=True,
        restart=dict(passed=True,updates_checked=7,rank_count=2),peak_bytes=[18*1024**3]*2,
        checkpoint_seconds=1.,validation=dict(events=128000,seconds=120.,
            precision=dict(passed=True,expected_activation_dtype='torch.float32')))


def test_preflight_and_approval_are_campaign_bound(tmp_path):
    spec=spec_fixture();spec['scaling']=str(tmp_path/'scaling')
    P.write_json(tmp_path/'scaling/gpus_2.json',scaling_fixture())
    result=preflight_fixture(spec);result['content_hash']=P.digest(result)
    P.write_json(tmp_path/'preflight/result.json',result)
    with pytest.raises(ValueError,match='Review'): P.approve(tmp_path,spec,False)
    estimate=P.approve(tmp_path,spec,True)
    assert P.require_approval(tmp_path,spec)==estimate
    changed=dict(result,checkpoint_seconds=2.)
    del changed['content_hash'];changed['content_hash']=P.digest(changed)
    P.write_json(tmp_path/'preflight/result.json',changed)
    with pytest.raises(ValueError,match='approval changed'): P.require_approval(tmp_path,spec)


@pytest.mark.parametrize('change',[
    lambda r:r.update(passed=False), lambda r:r['runtime'].update(world_size=1),
    lambda r:r['restart'].update(rank_count=1), lambda r:r['restart'].update(updates_checked=1),
    lambda r:r['validation']['precision'].update(expected_activation_dtype='torch.float16'),
    lambda r:r['validation'].update(events=1000),lambda r:r.update(peak_bytes=[39*1024**3]*2),
])
def test_incomplete_production_preflight_refused(tmp_path,change):
    spec=spec_fixture();spec['scaling']=str(tmp_path/'scaling')
    P.write_json(tmp_path/'scaling/gpus_2.json',scaling_fixture())
    result=preflight_fixture(spec);change(result);result['content_hash']=P.digest(result)
    P.write_json(tmp_path/'preflight/result.json',result)
    with pytest.raises(ValueError): P.require_preflight(tmp_path,spec)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Linear(17,10)
        self.dropout=torch.nn.Dropout(.15)
        self.trimmer=torch.nn.Identity();self.trimmer._counter=0
        self.force_nan=False
    def forward(self,points,features,vectors,mask):
        self.trimmer._counter+=1
        result=self.fc(self.dropout(features.mean(-1)))
        return result*float('nan') if self.force_nan else result


class ToyLoader:
    def __init__(self,batch,rank=0,offset=0):
        self.batch,self.rank,self.index=batch,rank,offset
    def __iter__(self): return self
    def __next__(self):
        self.index+=1
        row=E.B.fixed_batch(self.batch,7)
        row[0]['pf_features']+=self.index*.001+self.rank*.01
        return row


def _distributed_recovery_worker(rank,url,directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=url,rank=rank,world_size=2)
    try:
        device=torch.device('cpu')
        runtime=dict(world_size=2,campaign_sha256='production')
        spec=spec_fixture()
        spec['parameters']['state_layerwise']=180
        spec['recipes'][P.TASK['recipe']].update(epochs=2,total_steps=8,steps_per_epoch=4,validation_events_per_epoch=512)
        spec=json.loads(json.dumps(spec))
        P.require_approval=lambda *a:None
        E.shutdown_loader=lambda _:None
        E.build_loader=lambda split,inv,batch,**kw:(ToyLoader(batch,kw.get('rank',0)),None)
        spec['inventory']={}
        def fresh(_spec,_device,world):
            E.B.seed_everything(101)
            base=ToyModel()
            wrapper=torch.nn.parallel.DistributedDataParallel(base)
            return base,wrapper,E.build_optimizer(base,spec['recipes'][P.TASK['recipe']]),torch.amp.GradScaler('cpu',enabled=False)
        E.fresh=fresh
        root=Path(directory)
        for name in ('continuous','resumed','after_val','mid_val'):
            E.root_call(lambda:P.write_json(root/name/'preflight/result.json',dict(runtime=runtime)),rank,2)
            E.root_call(lambda:P.write_json(root/name/'campaign.json',spec),rank,2)
        E.train(root/'continuous',spec,rank,2,device,runtime)
        original_step=E.B.step
        def interrupt(*args,**kwargs):
            row=original_step(*args,**kwargs)
            if rank==1 and args[7]==3: E.B.STOP=True
            return row
        E.B.step=interrupt
        try:
            E.train(root/'resumed',spec,rank,2,device,runtime)
        except SystemExit as exc:
            assert exc.code==75
        else: raise AssertionError('Expected a coordinated stop from rank one')
        E.B.step=original_step;E.B.STOP=False
        E.train(root/'resumed',spec,rank,2,device,runtime)
        # Also resume AFTER a completed validation, exercising nonzero validation replay.
        def interrupt_after_val(*args,**kwargs):
            row=original_step(*args,**kwargs)
            if rank==1 and args[7]==6: E.B.STOP=True
            return row
        E.B.step=interrupt_after_val
        with pytest.raises(SystemExit) as stopped:
            E.train(root/'after_val',spec,rank,2,device,runtime)
        assert stopped.value.code==75
        E.B.step=original_step;E.B.STOP=False
        E.train(root/'after_val',spec,rank,2,device,runtime)
        # Interrupted validation has consumed data and mutated eval counters, but
        # must restart from the pre-validation commit, retaining the earlier selection.
        original_validate=E.validate
        validations=0
        def interrupt_validation(*args,**kwargs):
            nonlocal validations
            metric=original_validate(*args,**kwargs)
            validations+=1
            if validations==2:
                E.B.STOP=True
                return None
            return metric
        E.validate=interrupt_validation
        with pytest.raises(SystemExit) as stopped:
            E.train(root/'mid_val',spec,rank,2,device,runtime)
        assert stopped.value.code==75
        partial=E.trusted_load(P.run_directory(root/'mid_val')/'last.pt')['state']
        assert partial['validated_checks']==1 and partial['step']==8
        assert not (P.run_directory(root/'mid_val')/'validation/check_02.json').exists()
        E.validate=original_validate;E.B.STOP=False
        E.train(root/'mid_val',spec,rank,2,device,runtime)
        # Both runs include exact train and validation stream replay and per-rank RNG.
        a=E.trusted_load(P.run_directory(root/'continuous')/'last.pt')
        for name in ('resumed','after_val','mid_val'):
            b=E.trusted_load(P.run_directory(root/name)/'last.pt')
            for key,tensor in a['model'].items(): torch.testing.assert_close(tensor,b['model'][key],rtol=0,atol=0)
            for x,y in zip(a['optimizer']['slow'],b['optimizer']['slow']): torch.testing.assert_close(x,y,rtol=0,atol=0)
            assert a['optimizer']['step_counter']==b['optimizer']['step_counter']
            assert a['scaler']==b['scaler']
            for i in range(2):
                assert torch.equal(a['rank_states'][i]['torch'],b['rank_states'][i]['torch'])
                assert a['rank_states'][i]['trimmers']==b['rank_states'][i]['trimmers']
            for field in ('step','validated_checks','best_check','best_accuracy','amp_skipped_steps'):
                assert a['state'][field]==b['state'][field]
            assert b['state']['step']==8 and b['state']['validated_checks']==2
        # Completed no-op validates existing artifacts, never starts another run.
        E.train(root/'resumed',spec,rank,2,device,runtime)
        # An exception on rank-zero validation reaches BOTH ranks without hanging.
        broken=ToyModel();broken.force_nan=True
        try: E.validate(broken,ToyLoader(20) if rank==0 else None,device,rank,2,20,batch_size=20)
        except RuntimeError as exc: assert 'Nonfinite' in str(exc)
        else: raise AssertionError('Validation failure was not propagated')
        E.root_call(lambda:P.write_json(root/'passed.json',dict(passed=True)),rank,2)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo unavailable')
def test_full_distributed_stop_resume_and_rank_zero_validation(tmp_path):
    context=mp.spawn(_distributed_recovery_worker,args=((tmp_path/'rendezvous').as_uri(),str(tmp_path)),nprocs=2,join=False)
    deadline=time.monotonic()+90
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline: pytest.fail('Distributed recovery timed out')
        assert P.read_json(tmp_path/'passed.json')['passed']
    finally:
        for child in context.processes:
            if child.is_alive(): child.terminate()


def test_reject_checkpoint_runtime_change(tmp_path):
    torch.manual_seed(1)
    base=ToyModel();spec=spec_fixture();recipe=spec['recipes'][P.TASK['recipe']]
    opt=E.build_optimizer(base,recipe);scaler=torch.amp.GradScaler('cpu',enabled=False)
    path=tmp_path/'last.pt'
    E.save(path,base,opt,scaler,E.initial_state(),spec,dict(world_size=1),0,1)
    with pytest.raises(ValueError,match='runtime'):
        E.restore(path,base,opt,scaler,spec,dict(world_size=2),0,2)


def test_resume_rejects_invalid_candidate_path(tmp_path):
    base=ToyModel();spec=spec_fixture()
    opt=E.build_optimizer(base,spec['recipes'][P.TASK['recipe']]);scaler=torch.amp.GradScaler('cpu',enabled=False)
    state=E.initial_state();state.update(best_check=1,best_file='../outside.pt')
    E.save(tmp_path/'last.pt',base,opt,scaler,state,spec,{},0,1)
    with pytest.raises(ValueError,match='path'): E.restore(tmp_path/'last.pt',base,opt,scaler,spec,{},0,1)


def test_evaluation_locks_selection_reuses_files_and_rejects_partial_population(tmp_path,monkeypatch):
    model=ToyModel().eval()
    spec=spec_fixture()
    spec['inventory']=dict(splits=dict(val=dict(files=[dict(name=f'file_{i}.root',num_entries=10) for i in range(2)])))
    checkpoint=P.run_directory(tmp_path)/'best_model_val.pt'
    E.atomic_torch_save(dict(model=model.state_dict()),checkpoint)
    checksum=P.file_sha256(checkpoint)
    monkeypatch.setattr(E,'audit_training',lambda *a:dict(checkpoint_sha256=checksum))
    monkeypatch.setattr(E.B,'build_model',lambda *a,**kw:ToyModel())
    monkeypatch.setattr(E.B,'STOP',False)
    calls=[]
    def loader(*args):
        calls.append(True)
        return [next(ToyLoader(10))]
    monkeypatch.setattr(E,'finite_validation_loader',loader)
    with pytest.raises(ValueError,match='Expected 5M'): E.evaluate(tmp_path,spec,torch.device('cpu'))
    lock=P.read_json(tmp_path/'selection/locked_checkpoints.json')
    assert lock['campaign_sha256']==spec['content_hash'] and len(lock['checkpoints'])==1
    assert lock['checkpoints'][0]['sha256']==checksum
    assert len(calls)==2
    with pytest.raises(ValueError,match='Expected 5M'): E.evaluate(tmp_path,spec,torch.device('cpu'))
    assert len(calls)==2 and P.file_sha256(checkpoint)==checksum
    assert not (P.run_directory(tmp_path)/'full_validation/metrics.json').exists()
    sidecar=P.run_directory(tmp_path)/'full_validation/file_000.json'
    row=P.read_json(sidecar);row['contract']['checkpoint_sha256']='different'
    P.write_json(sidecar,row)
    with pytest.raises(ValueError,match='prediction changed'): E.evaluate(tmp_path,spec,torch.device('cpu'))
    assert len(calls)==2

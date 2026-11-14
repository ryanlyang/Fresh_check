import copy
import json
import os
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from part_ev_state_production import protocol as P, engine as E
from part_ev_state_bias.protocol import TASK
from part_ev_state_fullscale.protocol import recipes


def spec_fixture():
    return dict(content_hash='production',task=TASK,tasks=[TASK],policy=copy.deepcopy(P.POLICY),
        recipes=recipes(),parameters={TASK['architecture']:2329510},model_kind='bias',
        expected_gpu=dict(name='A100',total_memory=40*1024**3,nccl_environment={}))


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
        runtime=P.expected_runtime(spec_fixture())
        spec=spec_fixture()
        spec['parameters'][TASK['architecture']]=180
        spec['recipes'][TASK['recipe']].update(epochs=2,total_steps=8,steps_per_epoch=4,validation_events_per_epoch=512)
        spec=json.loads(json.dumps(spec))
        E.shutdown_loader=lambda _:None
        E.build_loader=lambda split,inv,batch,**kw:(ToyLoader(batch,kw.get('rank',0)),None)
        spec['inventory']={}
        def fresh(_spec,_device,world):
            E.B.seed_everything(101)
            base=ToyModel()
            wrapper=torch.nn.parallel.DistributedDataParallel(base)
            return base,wrapper,E.build_optimizer(base,spec['recipes'][TASK['recipe']]),torch.amp.GradScaler('cpu',enabled=False)
        E.fresh=fresh
        root=Path(directory)
        for name in ('continuous','resumed','after_val','mid_val'):
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
        partial=E.trusted_load(P.run_directory(root/'mid_val',spec)/'last.pt')['state']
        assert partial['validated_checks']==1 and partial['step']==8
        assert not (P.run_directory(root/'mid_val',spec)/'validation/check_02.json').exists()
        E.validate=original_validate;E.B.STOP=False
        E.train(root/'mid_val',spec,rank,2,device,runtime)
        # Both runs include exact train and validation stream replay and per-rank RNG.
        a=E.trusted_load(P.run_directory(root/'continuous',spec)/'last.pt')
        for name in ('resumed','after_val','mid_val'):
            b=E.trusted_load(P.run_directory(root/name,spec)/'last.pt')
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
        # A competing rank-zero lock failure must propagate to rank one.
        with pytest.raises(RuntimeError,match='already|Permission|temporarily unavailable'):
            with E.ExitStack() as stack:
                E.root_call(lambda:stack.enter_context(P.file_lock(P.run_directory(root/'resumed',spec)/'.training.lock',blocking=False)),rank,2)
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
    base=ToyModel();spec=spec_fixture();recipe=spec['recipes'][TASK['recipe']]
    opt=E.build_optimizer(base,recipe);scaler=torch.amp.GradScaler('cpu',enabled=False)
    path=tmp_path/'last.pt'
    E.save(path,base,opt,scaler,E.initial_state(spec),spec,dict(world_size=1),0,1)
    with pytest.raises(ValueError,match='runtime'):
        E.restore(path,base,opt,scaler,spec,dict(world_size=2),0,2)


def test_resume_rejects_invalid_candidate_path(tmp_path):
    base=ToyModel();spec=spec_fixture()
    opt=E.build_optimizer(base,spec['recipes'][TASK['recipe']]);scaler=torch.amp.GradScaler('cpu',enabled=False)
    state=E.initial_state(spec);state.update(best_check=1,best_file='../outside.pt')
    E.save(tmp_path/'last.pt',base,opt,scaler,state,spec,{},0,1)
    with pytest.raises(ValueError,match='path'): E.restore(tmp_path/'last.pt',base,opt,scaler,spec,{},0,1)


def test_evaluation_locks_selection_reuses_files_and_rejects_partial_population(tmp_path,monkeypatch):
    model=ToyModel().eval()
    spec=spec_fixture()
    spec['inventory']=dict(splits=dict(val=dict(files=[dict(name=f'file_{i}.root',num_entries=10) for i in range(2)])))
    checkpoint=P.run_directory(tmp_path,spec)/'best_model_val.pt'
    E.atomic_torch_save(dict(model=model.state_dict()),checkpoint)
    checksum=P.file_sha256(checkpoint)
    monkeypatch.setattr(E,'audit_training',lambda *a:dict(checkpoint_sha256=checksum))
    monkeypatch.setattr(E,'build_model',lambda *a,**kw:ToyModel())
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
    assert not (P.run_directory(tmp_path,spec)/'full_validation/metrics.json').exists()
    sidecar=P.run_directory(tmp_path,spec)/'full_validation/file_000.json'
    row=P.read_json(sidecar);row['contract']['checkpoint_sha256']='different'
    P.write_json(sidecar,row)
    with pytest.raises(ValueError,match='prediction changed'): E.evaluate(tmp_path,spec,torch.device('cpu'))
    assert len(calls)==2
    sidecar.with_suffix('.npz').unlink()
    with pytest.raises(ValueError,match='prediction is missing'): E.evaluate(tmp_path,spec,torch.device('cpu'))
    assert len(calls)==2


def test_deadline_and_slurm_marker_stop_without_shortening_recipe(tmp_path,monkeypatch):
    monkeypatch.setattr(E.B,'STOP',False)
    monkeypatch.setattr(E,'DEADLINE',float('inf'))
    monkeypatch.setenv('EVPROD_STOP_FILE',str(tmp_path/'stop'))
    assert not E.stopped()
    (tmp_path/'stop').touch()
    assert E.stopped()
    monkeypatch.delenv('EVPROD_STOP_FILE')
    monkeypatch.setattr(E,'DEADLINE',0.)
    assert E.stopping(torch.device('cpu'),1)
    assert spec_fixture()['recipes'][TASK['recipe']]['total_steps']==1_000_000


@pytest.mark.parametrize('kind,count',[('bias',2329510),('multiply',2343846),('memory',2364294)])
def test_production_constructs_exact_benchmarked_model(kind,count):
    spec=spec_fixture();spec['model_kind']=kind
    base=E.build_model(spec)
    assert sum(p.numel() for p in base.parameters())==count
    assert len(base.mod.blocks)==8


def test_ready_no_false_completion(tmp_path):
    spec=spec_fixture()
    assert E.ready(tmp_path,spec)==['train','evaluate','report']
    P.write(P.run_directory(tmp_path,spec)/'training_complete.json',{})
    with pytest.raises(KeyError): E.ready(tmp_path,spec)


def test_restore_rejects_other_model_and_budget(tmp_path):
    base=ToyModel();spec=spec_fixture()
    opt=E.build_optimizer(base,spec['recipes'][TASK['recipe']])
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    state=E.initial_state(spec);state['step']=1000001
    E.save(tmp_path/'last.pt',base,opt,scaler,state,spec,{},0,1)
    with pytest.raises(ValueError,match='cursor'):
        E.restore(tmp_path/'last.pt',base,opt,scaler,spec,{},0,1)
    state=E.initial_state(spec);state['task']=dict(TASK,seed=202)
    E.save(tmp_path/'last.pt',base,opt,scaler,state,spec,{},0,1)
    with pytest.raises(ValueError,match='task'):
        E.restore(tmp_path/'last.pt',base,opt,scaler,spec,{},0,1)


@pytest.mark.parametrize('fault',[None,'budget','missing','tie','optimizer','runtime','skips'])
def test_production_audits_fifty_checks_before_evaluation(tmp_path,fault):
    spec=spec_fixture();runtime=P.expected_runtime(spec);run=P.run_directory(tmp_path,spec)
    checks=[]
    for i in range(1,51):
        row=dict(check=i,step=i*20000,events=1280000,accuracy=.8+min(i,49)/1000,
                 macro_auc_ovo=.98,macro_auc_ovr=.98,loss=.2)
        P.write(run/'validation'/f'check_{i:02d}.json',row);checks.append(row)
    E.atomic_torch_save(dict(task=TASK,campaign_sha256=spec['content_hash'],runtime=runtime,
        validation=checks[48],model={}),run/'best_model_val.pt')
    complete=dict(task=TASK,step=1000000,validated_checks=50,best_check=49,
        best_accuracy=checks[48]['accuracy'],campaign_sha256=spec['content_hash'],runtime=runtime,
        amp_skipped_steps=2,train_seconds=100.,validation_seconds=10.,
        best_checkpoint_sha256=P.sha(run/'best_model_val.pt'))
    P.write(tmp_path/'campaign.json',spec)
    optimizer=dict(optimizer='Ranger',parameters=2329510,
                   groups=[dict(weight_decay=0,betas=[.95,.999],eps=1e-5)])
    if fault=='budget':complete['step']=200000;complete['validated_checks']=10
    if fault=='missing':(run/'validation/check_50.json').unlink()
    if fault=='tie':complete['best_check']=50
    if fault=='optimizer':optimizer['groups'][0]['weight_decay']=.01
    if fault=='runtime':complete['runtime']['world_size']=1
    if fault=='skips':complete['amp_skipped_steps']=50000
    P.write(run/'optimizer.json',optimizer);P.write(run/'training_complete.json',complete)
    if fault is None:
        assert E.audit_training(tmp_path,spec)['best_check']==49
        assert E.ready(tmp_path,spec)==['evaluate','report']
    else:
        with pytest.raises((ValueError,FileNotFoundError)):E.audit_training(tmp_path,spec)


def test_report_uses_audited_accuracy_checkpoint_and_validation_disclosure(tmp_path,monkeypatch):
    from part_ev_message_study import controls
    spec=spec_fixture()
    checks=[dict(check=i,step=i*20000,accuracy=.85,macro_auc_ovo=.98) for i in range(1,51)]
    trained=dict(validation_checks=checks,best_check=49)
    metric=dict(accuracy=.863,macro_auc_ovo=.989,qcd_rejection={
        'Hbb':{'0.5':dict(rejection=20000)},'Tbqq':{'0.5':dict(rejection=None)}})
    row=dict(task=TASK,best_check=49,metrics=metric)
    monkeypatch.setattr(E,'audit_training',lambda *a:trained)
    seen=[]
    def audit(*a,recompute):
        seen.append(recompute)
        return row,None
    monkeypatch.setattr(controls,'validate_evaluation',audit)
    result=E.report(tmp_path,spec)
    assert seen==[E.classification_metrics]
    assert result['model']['best_check']==49 and result['independent_test'] is False
    assert result['training_gpus']==2 and result['historical_controls_runtime_matched'] is False
    text=(tmp_path/'reports/production_report.md').read_text()
    assert '2,329,510' in text and '86.3000' in text and '| 50 | 1000000 |' in text
    assert 'overlapping selection' in text and 'ZB' in text and '20000.00' in text
    assert P.checked(tmp_path/'reports/production_report.json')==result

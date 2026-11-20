import copy
from contextlib import contextmanager
import importlib
import json
import time

import pytest
import torch
from part_ev_rts_tigris import protocol as P, production as E
from tests.test_part_ev_state_production_engine import ToyModel, ToyLoader


@contextmanager
def local_lock(path,**kwargs):
    path.parent.mkdir(parents=True,exist_ok=True)
    yield


def specification(kind='readout'):
    science=importlib.import_module(P.PACKAGES[kind]+'.protocol')
    return dict(content_hash='native-production',task=science.TASK,tasks=[science.TASK],model_kind=kind,
        parameters={science.TASK['architecture']:P.COUNTS[kind]},recipes=science.recipes(),inventory={},
        expected_gpu=dict(name='NVIDIA GH200',total_memory=96*2**30,nccl_environment={}))


@pytest.mark.parametrize('kind',list(P.PACKAGES))
def test_exact_models_native_bn_batch_and_optimizer(kind):
    spec=specification(kind)
    base,model,opt,scaler=E.fresh(spec,torch.device('cpu'),1)
    assert base is model and not any(isinstance(m,torch.nn.SyncBatchNorm) for m in base.modules())
    assert len(base.mod.blocks)==8 and sum(p.numel() for p in base.parameters())==P.COUNTS[kind]
    assert E.optimizer_metadata(base,opt)['optimizer']=='Ranger'
    assert P.expected_runtime(spec)['local_batch_size']==512
    with pytest.raises(ValueError):E.fresh(spec,torch.device('cpu'),2)


def test_native_stream_and_rng_resume_exact_before_after_and_during_validation(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(P,'file_lock',local_lock)
    monkeypatch.setattr(E.B,'STOP',False)
    monkeypatch.setattr(E,'DEADLINE',float('inf'))
    spec=specification();task=spec['task'];spec['parameters'][task['architecture']]=180
    spec['recipes'][task['recipe']].update(epochs=2,total_steps=8,steps_per_epoch=4,validation_events_per_epoch=512)
    spec=json.loads(json.dumps(spec))
    runtime=P.expected_runtime(spec);device=torch.device('cpu')
    batches=[]
    def loader(split,inventory,batch,**kw):
        batches.append((split,batch,kw))
        return ToyLoader(batch),None
    monkeypatch.setattr(E,'build_loader',loader)
    monkeypatch.setattr(E,'shutdown_loader',lambda _:None)
    monkeypatch.setattr(E,'build_model',lambda _:ToyModel())
    original_step=E.B.step;original_validate=E.validate
    for name in ('continuous','before','after','midval'):
        P.write(tmp_path/name/'campaign.json',spec)
    E.train(tmp_path/'continuous',spec,0,1,device,runtime)
    for name,step in (('before',3),('after',6)):
        def stop_step(*args,**kw):
            row=original_step(*args,**kw)
            if args[7]==step:E.B.STOP=True
            return row
        monkeypatch.setattr(E.B,'step',stop_step)
        with pytest.raises(SystemExit) as exc:E.train(tmp_path/name,spec,0,1,device,runtime)
        assert exc.value.code==75
        monkeypatch.setattr(E.B,'step',original_step);E.B.STOP=False
        E.train(tmp_path/name,spec,0,1,device,runtime)
    validations=0
    def stop_validation(*args,**kw):
        nonlocal validations
        result=original_validate(*args,**kw);validations+=1
        if validations==2:E.B.STOP=True;return None
        return result
    monkeypatch.setattr(E,'validate',stop_validation)
    with pytest.raises(SystemExit):E.train(tmp_path/'midval',spec,0,1,device,runtime)
    partial=E.trusted_load(P.run_directory(tmp_path/'midval',spec)/'last.pt')
    assert partial['state']['step']==8 and partial['state']['validated_checks']==1
    monkeypatch.setattr(E,'validate',original_validate);E.B.STOP=False
    E.train(tmp_path/'midval',spec,0,1,device,runtime)
    expected=E.trusted_load(P.run_directory(tmp_path/'continuous',spec)/'last.pt')
    for name in ('before','after','midval'):
        actual=E.trusted_load(P.run_directory(tmp_path/name,spec)/'last.pt')
        for key,tensor in expected['model'].items():torch.testing.assert_close(tensor,actual['model'][key],rtol=0,atol=0)
        for x,y in zip(expected['optimizer']['slow'],actual['optimizer']['slow']):torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert expected['optimizer']['step_counter']==actual['optimizer']['step_counter']
        assert torch.equal(expected['rank_states'][0]['torch'],actual['rank_states'][0]['torch'])
        assert expected['rank_states'][0]['trimmers']==actual['rank_states'][0]['trimmers']
        for field in ('step','validated_checks','best_accuracy','best_check','amp_skipped_steps'):
            assert expected['state'][field]==actual['state'][field]
        assert E.audit_training(tmp_path/name,spec)['training']['step']==8
    assert all(batch==512 and kw['workers']==2 for _,batch,kw in batches)
    assert all(kw['world_size']==1 and kw['rank']==0 for split,_,kw in batches if split=='train')
    E.train(tmp_path/'after',spec,0,1,device,runtime)  # Audited completed no-op.


def test_restart_rejects_cross_cluster_checkpoint(tmp_path):
    base=ToyModel();spec=specification();runtime=P.expected_runtime(spec)
    opt=E.build_optimizer(base,spec['recipes'][spec['task']['recipe']])
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    E.save(tmp_path/'last.pt',base,opt,scaler,E.initial_state(spec),spec,runtime,0,1)
    with pytest.raises(ValueError,match='runtime'):
        E.restore(tmp_path/'last.pt',base,opt,scaler,spec,dict(runtime,batchnorm='SyncBatchNorm'),0,2)


def test_full_validation_reuses_committed_chunks_and_rejects_partial_population(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'file_lock',local_lock)
    monkeypatch.setattr(E.B,'STOP',False);monkeypatch.setattr(E,'DEADLINE',float('inf'))
    spec=specification();model=ToyModel()
    spec['inventory']=dict(splits=dict(val=dict(files=[dict(name='file.root',num_entries=10)])))
    path=P.run_directory(tmp_path,spec)/'best_model_val.pt'
    E.atomic_torch_save(dict(model=model.state_dict()),path)
    monkeypatch.setattr(E,'audit_training',lambda *_:dict(checkpoint_sha256=P.sha(path)))
    monkeypatch.setattr(E,'build_model',lambda *_:ToyModel())
    calls=[]
    def batches(*_):calls.append(1);return [next(ToyLoader(10))]
    monkeypatch.setattr(E,'finite_validation_loader',batches)
    for _ in range(2):
        with pytest.raises(ValueError,match='Expected 5M'):E.evaluate(tmp_path,spec,torch.device('cpu'))
    assert len(calls)==1
    assert not (path.parent/'full_validation/metrics.json').exists()


@pytest.mark.parametrize('fault',(None,'checks','optimizer','runtime','skips'))
def test_fifty_check_audit(tmp_path,fault):
    spec=specification();task=spec['task'];runtime=P.expected_runtime(spec);run=P.run_directory(tmp_path,spec)
    checks=[]
    for i in range(1,51):
        row=dict(check=i,step=i*20000,events=1280000,accuracy=.8+min(i,49)/1000,macro_auc_ovo=.98)
        P.write(run/'validation'/f'check_{i:02d}.json',row);checks.append(row)
    E.atomic_torch_save(dict(task=task,campaign_sha256=spec['content_hash'],runtime=runtime,
        validation=checks[48],model={}),run/'best_model_val.pt')
    done=dict(task=task,step=1000000,validated_checks=50,best_check=49,best_accuracy=checks[48]['accuracy'],
        campaign_sha256=spec['content_hash'],runtime=runtime,amp_skipped_steps=2,train_seconds=100.,validation_seconds=10.,
        best_checkpoint_sha256=P.sha(run/'best_model_val.pt'))
    optimizer=dict(optimizer='Ranger',parameters=P.COUNTS['readout'],groups=[dict(weight_decay=0,betas=[.95,.999],eps=1e-5)])
    if fault=='checks':done['validated_checks']=49
    if fault=='optimizer':optimizer['groups'][0]['weight_decay']=.01
    if fault=='runtime':done['runtime']['world_size']=2
    if fault=='skips':done['amp_skipped_steps']=50000
    P.write(tmp_path/'campaign.json',spec);P.write(run/'optimizer.json',optimizer);P.write(run/'training_complete.json',done)
    if fault:
        with pytest.raises(ValueError):E.audit_training(tmp_path,spec)
    else:assert E.audit_training(tmp_path,spec)['best_check']==49

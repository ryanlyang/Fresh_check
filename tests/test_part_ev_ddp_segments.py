import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from part_ev_ddp_segments import protocol as S, engine as M
from part_ev_layerwise_ddp import engine as E
from tests.test_part_ev_layerwise_ddp import spec_fixture, ToyModel, ToyLoader


def hashed(row):
    row = dict(row)
    row.pop('content_hash', None)
    return dict(row, content_hash=S.P.digest(row))


def test_deadline_uses_allocation_not_process_start():
    env = dict(SLURM_JOB_START_TIME='1000', SLURM_JOB_END_TIME=str(1000+72*3600))
    # Startup already used 30 minutes; one hour is reserved for shutdown.
    assert M.allocation_deadline(env, wall_clock=lambda:2800, monotonic=lambda:10) == 10+70.5*3600
    for bad in ({}, dict(env, SLURM_JOB_END_TIME='nan'),
                dict(env, SLURM_JOB_END_TIME=str(1000+71*3600)),
                dict(env, SLURM_JOB_START_TIME='garbled')):
        with pytest.raises(ValueError): M.allocation_deadline(bad)
    with pytest.raises(ValueError, match='No safe'):
        M.allocation_deadline(env, wall_clock=lambda:1000+72*3600)


def test_deadline_and_external_signal_are_distinct():
    backend = SimpleNamespace(STOP=False, agree=lambda value,*args:value)
    controller = M.Deadline(SimpleNamespace(B=backend), 100, clock=lambda:101)
    assert controller.stopping('cpu', 1) and controller.expired and not controller.external_stop
    backend.STOP = True
    assert controller.stopping('cpu', 1) and controller.external_stop


def test_private_adapter_does_not_change_historical_engine_or_source_contract():
    private = M.create_engine()
    assert private is not E and private.train is not E.train
    assert private.train.__code__.co_code == E.train.__code__.co_code
    original = S.P.source_manifest()
    assert not any('ddp_segments' in name for name in original)
    assert original.items() <= S.source_manifest().items()
    recipe = spec_fixture()['recipes'][S.P.TASK['recipe']]
    assert (recipe['total_steps'],recipe['epochs'],recipe['steps_per_epoch']) == (1_000_000,50,20_000)
    assert S.P.TASK['architecture'] == 'state_layerwise' and S.P.TASK['seed'] == 101
    assert S.POLICY['default_segments'] == 4 and S.POLICY['shutdown_margin_seconds'] == 3600


@pytest.mark.parametrize('change',[
    lambda row:row.update(campaign_sha256='other'),
    lambda row:row['runtime'].update(world_size=1),
    lambda row:row.update(rank_states=[{}]),
    lambda row:row['state'].update(step=1_000_001),
    lambda row:row['state'].update(step=40000,validated_checks=0),
])
def test_snapshot_rejects_invalid_restart_identity(tmp_path,monkeypatch,change):
    from part_ev_fullscale import checkpoint
    spec=spec_fixture()
    row=dict(schema=S.P.SCHEMA,campaign_sha256=spec['content_hash'],runtime=S.P.expected_runtime(spec),
             state=E.initial_state(),rank_states=[{},{}])
    run=S.P.run_directory(tmp_path);run.mkdir(parents=True)
    (run/'last.pt').write_bytes(b'fixture')
    monkeypatch.setattr(checkpoint,'trusted_load',lambda _:row)
    assert S.snapshot(tmp_path,spec)['step']==0
    change(row)
    with pytest.raises(ValueError,match='identity/progress'): S.snapshot(tmp_path,spec)


def test_stale_last_checkpoint_cannot_pass_completion(tmp_path,monkeypatch):
    from part_ev_fullscale import checkpoint
    spec=spec_fixture()
    row=dict(schema=S.P.SCHEMA,campaign_sha256=spec['content_hash'],runtime=S.P.expected_runtime(spec),
             state=E.initial_state(),rank_states=[{},{}])
    run=S.P.run_directory(tmp_path);run.mkdir(parents=True)
    (run/'last.pt').write_bytes(b'fixture')
    S.P.write_json(run/'training_complete.json',dict(row['state'],step=1000000,validated_checks=50))
    monkeypatch.setattr(checkpoint,'trusted_load',lambda _:row)
    monkeypatch.setattr(E,'audit_training',lambda *a:None)
    with pytest.raises(ValueError,match='does not match completed'): S.snapshot(tmp_path,spec)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    campaign = tmp_path/'campaign'; campaign.mkdir()
    chain = campaign/'segment_chains'/'first'
    spec = spec_fixture()
    S.P.write_json(campaign/'production_approval.json', {'fixture':True})
    S.P.write_json(campaign/'preflight/result.json', {'fixture':True})
    monkeypatch.setattr(S.P, 'load', lambda *a,**k:spec)
    monkeypatch.setattr(S.P, 'require_approval', lambda *a:dict(evaluate_hours=5,train_hours=216))
    monkeypatch.setattr(S, 'source_manifest', lambda:{'adapter':'hash'})
    plan = S.prepare(chain,campaign,4)
    return campaign,chain,spec,plan


def test_frozen_chain_reuses_approval_and_rejects_mutations(prepared):
    campaign,chain,spec,plan = prepared
    original = {p:p.read_bytes() for p in (campaign/'production_approval.json',campaign/'preflight/result.json')}
    assert S.load(chain) == (plan,spec)
    assert plan['initial'] == dict(step=0,validated_checks=0,checkpoint_sha256=None,complete=False)
    with pytest.raises(FileExistsError): S.prepare(chain,campaign,4)
    with pytest.raises(ValueError): S.prepare(campaign/'wrong',campaign,4)
    for count in (0,9,True):
        with pytest.raises(ValueError): S.prepare(chain.parent/'new',campaign,count)
    assert all(p.read_bytes()==value for p,value in original.items())
    S.P.write_json(campaign/'production_approval.json', {'fixture':'changed'})
    with pytest.raises(ValueError, match='approval changed'): S.load(chain)


def test_completed_run_can_be_adopted_after_receipt_commit_failure(prepared,monkeypatch):
    campaign,chain,_,_=prepared
    complete=dict(step=1000000,validated_checks=50,checkpoint_sha256='done',complete=True)
    monkeypatch.setattr(S,'snapshot',lambda *a:complete)
    plan=S.prepare(chain.parent/'recovery',campaign,1)
    assert plan['initial']==complete and plan['segments']==1


def test_missing_wrong_or_overwritten_handoff_cannot_continue(prepared):
    _,chain,_,plan = prepared
    assert S.previous_snapshot(chain,plan,0) == plan['initial']
    with pytest.raises(FileNotFoundError): S.previous_snapshot(chain,plan,1)
    row = hashed(dict(chain_sha256=plan['content_hash'],index=0,status='paused',
                      after=dict(step=123,validated_checks=0,checkpoint_sha256='new',complete=False)))
    S.P.write_json(S.receipt_path(chain,0),row)
    assert S.previous_snapshot(chain,plan,1) == row['after']
    with pytest.raises(FileExistsError): S.previous_snapshot(chain,plan,0)
    S.P.write_json(S.receipt_path(chain,0),hashed(dict(row,status='complete')))
    with pytest.raises(ValueError): S.previous_snapshot(chain,plan,1)


def test_final_gate_never_accepts_a_successful_but_incomplete_chain(prepared,monkeypatch):
    _,chain,spec,plan = prepared
    cursor = dict(step=900000,validated_checks=45,checkpoint_sha256='latest',complete=False)
    monkeypatch.setattr(S,'snapshot',lambda *a:cursor)
    row = dict(chain_sha256=plan['content_hash'],index=3,status='paused',after=cursor)
    S.P.write_json(S.receipt_path(chain,3),hashed(row))
    with pytest.raises(ValueError,match='not complete'): S.completion_gate(chain,plan,spec)
    cursor.update(step=1000000,validated_checks=50,complete=True)
    S.P.write_json(S.receipt_path(chain,3),hashed(dict(row,status='complete',after=cursor)))
    assert S.completion_gate(chain,plan,spec)==cursor
    cursor['checkpoint_sha256']='changed'
    with pytest.raises(ValueError): S.completion_gate(chain,plan,spec)


def assert_nested_equal(a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: assert_nested_equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_nested_equal(x,y)
    else:
        assert a==b


def _segmented_worker(rank,url,directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=url,rank=rank,world_size=2)
    root = Path(directory)
    try:
        device = torch.device('cpu')
        spec = spec_fixture()
        spec['parameters']['state_layerwise']=180
        spec['recipes'][S.P.TASK['recipe']].update(epochs=3,total_steps=12,steps_per_epoch=4,
                                                validation_events_per_epoch=512)
        spec['inventory']={}
        spec=json.loads(json.dumps(spec))
        runtime=S.P.expected_runtime(spec)
        S.P.require_approval=lambda *a:None
        def configured_engine():
            engine=M.create_engine()
            engine.shutdown_loader=lambda _:None
            engine.build_loader=lambda split,inv,batch,**kw:(ToyLoader(batch,kw.get('rank',0)),None)
            def fresh(_spec,_device,world):
                E.B.seed_everything(101)
                base=ToyModel()
                return (base,torch.nn.parallel.DistributedDataParallel(base),
                        E.build_optimizer(base,spec['recipes'][S.P.TASK['recipe']]),
                        torch.amp.GradScaler('cpu',enabled=False))
            engine.fresh=fresh
            return engine
        for name in ('continuous','segmented','signal','error'):
            E.root_call(lambda:S.P.write_json(root/name/'preflight/result.json',dict(runtime=runtime)),rank,2)
            E.root_call(lambda:S.P.write_json(root/name/'campaign.json',spec),rank,2)
        configured_engine().train(root/'continuous',spec,rank,2,device,runtime)
        campaign=root/'segmented';chain=campaign/'segment_chains/first'
        plan=dict(campaign=str(campaign),segments=4,content_hash='chain',
                  initial=S.snapshot(campaign,spec))
        original_step=E.B.step
        for index in range(4):
            engine=configured_engine();now=[0.]
            if index==0:
                def step(*args,**kwargs):
                    result=original_step(*args,**kwargs)
                    if rank==1 and args[7]==2: now[0]=200
                    return result
                engine.B.step=step
            if index==1:
                original_validate=engine.validate
                checks=[0]
                def mid_validation(*args,**kwargs):
                    result=original_validate(*args,**kwargs)
                    checks[0]+=1
                    if checks[0]==2:
                        if rank==1: now[0]=200
                        assert engine.stopping(device,2)
                        return None
                    return result
                engine.validate=mid_validation
            before_hash=S.snapshot(campaign,spec)['checkpoint_sha256']
            result=M.train_segment(chain,plan,spec,index,rank,2,device,runtime,100,
                                   engine=engine,clock=lambda:now[0])
            E.B.step=original_step
            assert result['status']==('paused','paused','complete','already_complete')[index]
            if index==0:
                assert result['after']['step']==3 and result['after']['validated_checks']==0
            if index==1:
                assert result['after']['step']==8 and result['after']['validated_checks']==1
                assert not (S.P.run_directory(campaign)/'validation/check_02.json').exists()
            if index==3: assert result['after']['checkpoint_sha256']==before_hash
        E.root_call(lambda:S.completion_gate(chain,plan,spec),rank,2)
        a=E.trusted_load(S.P.run_directory(root/'continuous')/'last.pt')
        b=E.trusted_load(S.P.run_directory(campaign)/'last.pt')
        for key in ('model','optimizer','scaler','rank_states','runtime'):
            assert_nested_equal(a[key],b[key])
        for key in ('step','validated_checks','best_accuracy','best_check','amp_skipped_steps'):
            assert a['state'][key]==b['state'][key]
        for check in range(1,4):
            for key in ('accuracy','macro_auc_ovo','step','check'):
                left=S.P.read_json(S.P.run_directory(root/'continuous')/f'validation/check_{check:02d}.json')
                right=S.P.read_json(S.P.run_directory(campaign)/f'validation/check_{check:02d}.json')
                assert left.get(key)==right.get(key)
        # A signal is not a successful planned handoff, even with a saved checkpoint.
        for name in ('signal','error'):
            engine=configured_engine()
            target=root/name;new_chain=target/'segment_chains/first'
            new_plan=dict(campaign=str(target),segments=4,content_hash=name,initial=S.snapshot(target,spec))
            def bad_step(*args,**kwargs):
                if name=='error': raise RuntimeError('deliberate failure')
                result=original_step(*args,**kwargs)
                if rank==1: E.B.STOP=True
                return result
            E.B.step=bad_step
            if name=='signal':
                with pytest.raises(SystemExit) as exc:
                    M.train_segment(new_chain,new_plan,spec,0,rank,2,device,runtime,100,
                                    engine=engine,clock=lambda:0.)
                assert exc.value.code==75
            else:
                with pytest.raises(RuntimeError,match='deliberate failure'):
                    M.train_segment(new_chain,new_plan,spec,0,rank,2,device,runtime,100,
                                    engine=engine,clock=lambda:0.)
            E.B.step=original_step;E.B.STOP=False
            assert not S.receipt_path(new_chain,0).exists()
        # Deadline during replay cannot masquerade as a productive handoff.
        target=root/'signal';new_chain=target/'segment_chains/replay_only'
        new_plan=dict(campaign=str(target),segments=4,content_hash='replay',initial=S.snapshot(target,spec))
        with pytest.raises(RuntimeError,match='No committed training progress'):
            M.train_segment(new_chain,new_plan,spec,0,rank,2,device,runtime,100,
                            engine=configured_engine(),clock=lambda:200.)
        assert S.snapshot(target,spec)==new_plan['initial']
        assert not S.receipt_path(new_chain,0).exists()
        # No silently continuing from an unexpected checkpoint.
        new_plan['initial']['step']+=1
        with pytest.raises(ValueError,match='Checkpoint changed'):
            M.train_segment(new_chain,new_plan,spec,0,rank,2,device,runtime,100,
                            engine=configured_engine(),clock=lambda:0.)
        E.root_call(lambda:S.P.write_json(root/'passed.json',dict(passed=True)),rank,2)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo unavailable')
def test_two_rank_segment_handoffs_exactly_match_uninterrupted_training(tmp_path):
    context=mp.spawn(_segmented_worker,args=((tmp_path/'rendezvous').as_uri(),str(tmp_path)),nprocs=2,join=False)
    deadline=time.monotonic()+120
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline: pytest.fail('Segmented distributed test timed out')
        assert S.P.read_json(tmp_path/'passed.json')['passed']
    finally:
        for child in context.processes:
            if child.is_alive(): child.terminate()
